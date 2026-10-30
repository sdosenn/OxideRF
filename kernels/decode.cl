// Resident PAL RF processing. All transform batches share one in-order queue.
#define RF_N 32768u
#define RF_EDGE 2048u
#define RF_HOP (RF_N - 2u * RF_EDGE)
#define TBC_WIDTH 1135u
#define TBC_HEIGHT 313u
#define TAU_F 6.2831853071795864769f
float2 cmul(float2 a, float2 b) {
    return (float2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
// RF arrives in its capture format and is scaled here exactly as on the CPU:
// the scales are powers of two. One work-group per block sums its mean.
#define RF_INPUT(NAME, T, OFFSET, SCALE) \
__kernel void block_means_##NAME(__global const T *raw, __global float *means, \
                                 __local float *sums) { \
    uint lid = get_local_id(0), size = get_local_size(0), b = get_group_id(0); \
    float sum = 0.0f; \
    for (uint i = lid; i < RF_N; i += size) sum += ((float)raw[b*RF_HOP+i]-OFFSET)*SCALE; \
    sums[lid] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (uint stride = size/2; stride > 0; stride >>= 1) { \
        if (lid < stride) sums[lid] += sums[lid+stride]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (lid == 0) means[b] = sums[0]/RF_N; \
} \
__kernel void pack_rf_##NAME(__global const T *raw, __global const float *means, \
                             __global float2 *dst) { \
    uint i = get_global_id(0), b = get_global_id(1); \
    dst[b*RF_N+i] = (float2)(((float)raw[b*RF_HOP+i]-OFFSET)*SCALE-means[b], 0.0f); \
}
RF_INPUT(f32, float, 0.0f, 1.0f)
RF_INPUT(u8, uchar, 128.0f, (1.0f/128.0f))
RF_INPUT(u16, ushort, 32768.0f, (1.0f/32768.0f))
__kernel void rf_bands(__global const float2 *spectrum, __global const float *rf_gain,
                        __global const float *chroma_gain, __global float2 *analytic,
                        __global float2 *chroma, int luma_only) {
    uint i = get_global_id(0), b = get_global_id(1), p = b*RF_N+i;
    analytic[p] = spectrum[p]*rf_gain[i];
    if (!luma_only) chroma[p] = spectrum[p]*chroma_gain[i];
}
__kernel void fm_phase(__global const float2 *analytic, __global float2 *frequency,
                        float phase_to_hz) {
    uint i = get_global_id(0), b = get_global_id(1), j = max(i, 1u);
    float2 a = analytic[b*RF_N+j-1], v = analytic[b*RF_N+j];
    float phase = atan2(v.y*a.x-v.x*a.y, v.x*a.x+v.y*a.y);
    if (phase < 0.0f) phase += TAU_F;
    frequency[b*RF_N+i] = (float2)(phase*phase_to_hz, 0.0f);
}
__kernel void video_bands(__global const float2 *frequency, __global const float2 *video_gain,
                           __global const float *sync_gain, __global float2 *luma,
                           __global float2 *sync) {
    uint i = get_global_id(0), p = get_global_id(1)*RF_N+i;
    float2 v = cmul(frequency[p], video_gain[i]);
    luma[p] = v; sync[p] = v*sync_gain[i];
}
__kernel void store_signal(__global const float2 *luma, __global const float2 *sync,
                            __global const float2 *chroma, __global const float2 *oscillator,
                            __global const float2 *block_phase, __global float4 *ring,
                            ulong absolute, uint mask, int luma_only,
                            __global const float2 *analytic, __global float *envelope, int detect_dropouts) {
    uint i = get_global_id(0)+RF_EDGE, b = get_global_id(1), p = b*RF_N+i;
    float2 c = (float2)(0.0f);
    if (!luma_only) {
        float2 rotation = cmul(oscillator[i], block_phase[b]);
        c = cmul(chroma[p], rotation)*(1.0f/RF_N);
    }
    uint dst = (uint)(absolute+(ulong)b*RF_HOP+i)&mask;
    ring[dst] = (float4)(luma[p].x/RF_N, sync[p].x/RF_N, c.x, c.y);
    if (detect_dropouts) envelope[dst]=length(analytic[p])/RF_N;
}

// Repair only active picture samples. Keep VBI, sync and burst byte-identical.
__kernel void repair_dropouts(__global const ushort *original, __global const uchar *mask,
                               __global ushort *output) {
    uint id=get_global_id(0), plane=id/(TBC_WIDTH*TBC_HEIGHT), at=id%(TBC_WIDTH*TBC_HEIGHT);
    int row=at/TBC_WIDTH, x=at%TBC_WIDTH;
    ushort value=original[id];
    if(mask[at] && row>=22 && row<310 && x>=185 && x<1107) {
        int step=plane ? 4 : 1;
        for(int d=step;d<=16;d+=step) {
            uint sum=0,count=0;
            for(int side=-1;side<=1;side+=2) {
                int donor=row+side*d;
                if(donor>=22 && donor<310 && !mask[donor*TBC_WIDTH+x]) {
                    sum+=original[plane*TBC_WIDTH*TBC_HEIGHT+donor*TBC_WIDTH+x];++count;
                }
            }
            if(count){value=(ushort)((sum+count/2)/count);break;}
        }
    }
    output[id]=value;
}
// Estimate the sync floor from a low-tail quantile, independent of RF DC shift.
__kernel void sync_histogram(__global const float4 *ring,uint mask,ulong start,
                             __global uint *histogram) {
    __local uint bins[128];
    uint lid=get_local_id(0);
    if(lid<128) bins[lid]=0;
    barrier(CLK_LOCAL_MEM_FENCE);
    float value=ring[(uint)(start+get_global_id(0))&mask].y;
    uint bin=(uint)clamp(floor((value-3200000.0f)/16000.0f),0.0f,127.0f);
    atomic_inc(&bins[bin]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid<128 && bins[lid]) atomic_add(&histogram[lid],bins[lid]);
}
__kernel void choose_sync_slice(__global const uint *histogram,uint samples,
                                __global float *slice) {
    uint target=max(samples*3u/100u,1u), count=0;
    for(uint bin=0;bin<128;++bin) {
        count+=histogram[bin];
        if(count>=target) {slice[0]=clamp(3200000.0f+(bin+0.5f)*16000.0f+150000.0f,3650000.0f,4400000.0f);return;}
    }
}
// Only threshold crossings scan backwards. Ordinary samples do constant work.
// The event list is bounded by the minimum accepted pulse width, not tape length.
__kernel void find_pulses(__global const float4 *ring, uint mask, ulong start,
                           ulong first_valid, float samples_per_us,
                           __global uint *count, __global ulong2 *events, uint capacity, __global const float *slice) {
    ulong p = start+get_global_id(0);
    float level=slice[0];
    if (p <= first_valid || ring[(uint)p&mask].y < level ||
        ring[(uint)(p-1)&mask].y >= level) return;
    uint limit = (uint)ceil(samples_per_us*30.5f)+1;
    uint length = 0;
    while (length <= limit && p > first_valid+length &&
           ring[(uint)(p-length-1)&mask].y < level) ++length;
    if (length < (uint)floor(samples_per_us*3.0f) || length > limit) return;
    uint slot = atomic_inc(count);
    if (slot < capacity) events[slot] = (ulong2)(p-length, length);
}
float4 sample_ring(__global const float4 *ring, uint mask, uint origin, float offset) {
    int i = (int)floor(offset);
    float t = offset-floor(offset);
    float4 a = ring[(origin+(uint)i)&mask], b = ring[(origin+(uint)(i+1))&mask];
    return a+(b-a)*t;
}
float4 sample_video(__global const float4 *ring,uint mask,uint origin,float offset, __global const float *sinc_lut) {
    int i=(int)floor(offset);float t=offset-floor(offset);
    float4 a=ring[(origin+(uint)(i-1))&mask],b=ring[(origin+(uint)i)&mask];
    float4 c=ring[(origin+(uint)(i+1))&mask],d=ring[(origin+(uint)(i+2))&mask];
    float4 value=b+0.5f*t*(c-a+t*(2.0f*a-5.0f*b+4.0f*c-d+t*(3.0f*(b-c)+d-a)));
    uint phase=convert_uint_rtz(t*65536.0f+0.5f);
    float luma=0.0f;
    for(int tap=0;tap<16;++tap) luma+=ring[(origin+(uint)(i+tap-7))&mask].x*sinc_lut[phase*16u+(uint)tap];
    value.x=luma;return value;
}
float median8(float *v) {
    for (uint i=1; i<8; ++i) {
        float x=v[i]; uint j=i;
        while (j>0 && v[j-1]>x) { v[j]=v[j-1]; --j; }
        v[j]=x;
    }
    return v[4];
}
// Coordinates stay relative to a single line to preserve subpixel precision
// with FP32 even when a recording contains billions of RF samples.
float4 refine_row(__global const float4 *ring, uint mask, uint origin, float edge,
                  float samples_per_us, float threshold) {
    float tips[8], porches[8],tip=0.0f,porch=0.0f,best=edge;
    float valid=0.0f;
    for(uint pass=0;pass<2;++pass) {
    valid=0.0f;
    for (uint i=0;i<8;++i) {
        tips[i]=sample_ring(ring,mask,origin,edge+(0.8f+i*0.18f)*samples_per_us).y;
        porches[i]=sample_ring(ring,mask,origin,edge+(6.8f+i*0.2f)*samples_per_us).y;
    }
    tip=median8(tips); porch=median8(porches); best=edge;
    if (porch-tip>=100000.0f && porch-tip<600000.0f) {
        float level=threshold>0.0f ? threshold : (tip+porch)*0.5f, distance=INFINITY;
        int start=(int)floor(edge-3.0f*samples_per_us);
        int end=(int)(edge+samples_per_us);
        for (int i=start;i<end;++i) {
            float a=ring[(origin+(uint)(i-1))&mask].y;
            float b=ring[(origin+(uint)i)&mask].y;
            if (a>level && b<=level) {
                float pos=(float)(i-1)+(a-level)/(a-b);
                if (fabs(pos-edge)<distance) { best=pos; distance=fabs(pos-edge); }
            }
        }
        start=(int)(edge+3.0f*samples_per_us);
        end=(int)(edge+6.5f*samples_per_us);
        for (int i=start;i<end;++i) {
            float a=ring[(origin+(uint)(i-1))&mask].y;
            float b=ring[(origin+(uint)i)&mask].y;
            if (a<level && b>=level) {
                float right=(float)(i-1)+(level-a)/(b-a);
                float from_right=right-(4.7f-2.25f/40.0f+3.0f/17.734475f)*samples_per_us;
                if (isfinite(distance) && fabs((right-best)/samples_per_us-4.7f)<0.7f) {
                    best=from_right; valid=1.0f;
                }
                break;
            }
        }
    }
    if(valid==0.0f)break;
    edge=best;
    }
    for(uint i=0;i<8;++i) {
        tips[i]=sample_ring(ring,mask,origin,best+(0.8f+i*0.18f)*samples_per_us).y;
        porches[i]=sample_ring(ring,mask,origin,best+(6.8f+i*0.2f)*samples_per_us).y;
    }
    return (float4)(best,median8(tips),median8(porches),valid);
}
__kernel void refine_lines(__global const float4 *ring, uint mask,
                            __global const uint *origins, __global const float4 *lines,
                            float samples_per_us, __global float4 *result, float threshold) {
    uint row=get_global_id(0);
    float edge=lines[row].x;
    if (lines[row].z == 0.0f) { result[row]=(float4)(edge,0,0,0); return; }
    result[row]=refine_row(ring,mask,origins[row],edge,samples_per_us,threshold);
}
// Field-common sync threshold from the first refinement: the median tip and
// porch levels over rows 22..299 still locked, as on the CPU. One work item per
// row finds its rank; the rows ranked at the middle write the medians, and
// item 0 writes the number of locked rows.
__kernel void refine_threshold(__global const float4 *result, __global float *threshold) {
    uint i=get_global_id(0);
    float4 own=result[22+i];
    uint n=0, tip_rank=0, porch_rank=0;
    for (uint j=0; j<278; ++j) {
        float4 other=result[22+j];
        if (other.w > 0.0f) {
            ++n;
            if (other.y<own.y || (other.y==own.y && j<i)) ++tip_rank;
            if (other.z<own.z || (other.z==own.z && j<i)) ++porch_rank;
        }
    }
    if (i == 0) threshold[2]=(float)n;
    if (own.w > 0.0f) {
        if (tip_rank == n/2) threshold[0]=own.y;
        if (porch_rank == n/2) threshold[1]=own.z;
    }
}
// Second refinement at the field threshold, starting from the first result
// without a host round trip. Positions stay relative to each line's origin.
__kernel void refine_lines_at(__global const float4 *ring, uint mask,
                               __global const uint *origins, __global const float4 *first,
                               float samples_per_us, __global float4 *result,
                               __global const float *threshold) {
    uint row=get_global_id(0);
    float4 previous=first[row];
    float whole=floor(previous.x), edge=previous.x-whole;
    if (previous.w == 0.0f) { result[row]=(float4)(edge,0,0,0); return; }
    float level=(threshold[0]+threshold[1])*0.5f;
    result[row]=refine_row(ring,mask,origins[row]+(uint)(int)whole,edge,samples_per_us,level);
}
// Chunked forward and backward passes of the dropout envelope low-pass, as
// in dropout.rs: each work item filters 256 samples after a 512-sample
// warm-up, which reproduces the serial filter. Contraction into FMA would
// change rounding, so it stays off.
#define SMOOTH_CHUNK 256u
#define SMOOTH_WARMUP 512u
float clean_envelope(float v) { return isfinite(v) ? fmax(v,0.0f) : 0.0f; }
__kernel void smooth_forward(__global const float *envelope, uint mask, uint origin, uint count,
                             float b, float a, __global float *out) {
    #pragma OPENCL FP_CONTRACT OFF
    uint begin=get_global_id(0)*SMOOTH_CHUNK;
    if (begin>=count) return;
    uint end=min(begin+SMOOTH_CHUNK,count), from=begin>SMOOTH_WARMUP ? begin-SMOOTH_WARMUP : 0u;
    float previous_input=clean_envelope(envelope[(origin+from)&mask]), previous_output=previous_input;
    for (uint k=from; k<end; ++k) {
        float current=clean_envelope(envelope[(origin+k)&mask]);
        float value=b*(current+previous_input)-a*previous_output;
        previous_input=current; previous_output=value;
        if (k>=begin) out[k]=value;
    }
}
__kernel void smooth_backward(__global const float *in, uint count, float b, float a,
                              __global float *out) {
    #pragma OPENCL FP_CONTRACT OFF
    uint begin=get_global_id(0)*SMOOTH_CHUNK;
    if (begin>=count) return;
    uint end=min(begin+SMOOTH_CHUNK,count), from=min(end+SMOOTH_WARMUP,count);
    float previous_input=in[from-1], previous_output=previous_input;
    for (uint k=from; k>begin; --k) {
        float current=in[k-1];
        float value=b*(current+previous_input)-a*previous_output;
        previous_input=current; previous_output=value;
        if (k-1<end) out[k-1]=value;
    }
}
// Dropout threshold and intervals on the smoothed envelope, as
// dropout::raw_intervals. One work-group of 256 finds the median of every
// 32nd positive level exactly with a radix select: positive floats order
// like their bit patterns. It stores the median and the number of levels.
__kernel void dropout_median(__global const float *smoothed, uint count,
                             __global float *level, __local uint *hist) {
    __local uint total, chosen, remaining;
    uint lid=get_local_id(0), size=get_local_size(0), m=(count+31u)/32u;
    if (lid==0) total=0;
    barrier(CLK_LOCAL_MEM_FENCE);
    uint mine=0;
    for (uint j=lid; j<m; j+=size) {
        float v=smoothed[j*32u];
        if (isfinite(v) && v>0.0f) ++mine;
    }
    atomic_add(&total,mine);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint n=total;
    if (n<32u) { if (lid==0) level[1]=(float)n; return; }
    uint prefix=0, k=n/2u;
    for (int shift=24; shift>=0; shift-=8) {
        uint high=shift==24 ? 0u : 0xffffffffu<<(shift+8);
        hist[lid]=0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j=lid; j<m; j+=size) {
            float v=smoothed[j*32u];
            uint bits=as_uint(v);
            if (isfinite(v) && v>0.0f && (bits&high)==(prefix&high))
                atomic_inc(&hist[(bits>>shift)&255u]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid==0) {
            uint below=0, bucket=0;
            while (below+hist[bucket]<=k) { below+=hist[bucket]; ++bucket; }
            chosen=prefix|(bucket<<shift);
            remaining=k-below;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        prefix=chosen; k=remaining;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid==0) { level[0]=as_float(prefix); level[1]=(float)n; }
}
#define DROPOUT_CHUNK 256u
// 0: no low or recovered sample, 1: last decisive sample low, 2: recovered.
uint dropout_class(float v, float threshold) {
    if (!isfinite(v) || v<=threshold) return 1u;
    return v>=threshold*1.25f ? 2u : 0u;
}
__kernel void dropout_chunks(__global const float *smoothed, uint count,
                             __global const float *level, float fraction,
                             __global uint *state) {
    uint chunk=get_global_id(0), begin=chunk*DROPOUT_CHUNK;
    if (begin>=count) return;
    float threshold=level[0]*fraction;
    uint last=0, end=min(begin+DROPOUT_CHUNK,count);
    for (uint k=begin; k<end; ++k) {
        uint c=dropout_class(smoothed[k],threshold);
        if (c) last=c;
    }
    state[chunk]=last;
}
// One work-group turns per-chunk states into each chunk's entry state:
// whether the scan is inside a dropout when it reaches the chunk.
__kernel void dropout_entries(__global const uint *state, uint chunks,
                              __global uint *entry, __local uint *scan) {
    uint lid=get_local_id(0), size=get_local_size(0);
    uint per=(chunks+size-1u)/size, first=lid*per, end=min(first+per,chunks);
    uint last=0;
    for (uint c=first; c<end; ++c) if (state[c]) last=state[c];
    scan[lid]=last;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset=1; offset<size; offset<<=1) {
        uint v=lid>=offset ? scan[lid-offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (scan[lid]==0u) scan[lid]=v;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    uint inside=lid>0 && scan[lid-1]==1u;
    for (uint c=first; c<end; ++c) {
        entry[c]=inside;
        if (state[c]) inside=state[c]==1u;
    }
}
// Boundary events: the sample index where a dropout starts, or with the top
// bit set where the level recovers. Order is restored on the host.
__kernel void dropout_events(__global const float *smoothed, uint count,
                             __global const float *level, float fraction,
                             __global const uint *entry, __global uint *events,
                             volatile __global uint *total, uint capacity) {
    uint chunk=get_global_id(0), begin=chunk*DROPOUT_CHUNK;
    if (begin>=count || level[1]<32.0f) return;
    float threshold=level[0]*fraction;
    uint inside=entry[chunk], end=min(begin+DROPOUT_CHUNK,count);
    for (uint k=begin; k<end; ++k) {
        uint c=dropout_class(smoothed[k],threshold);
        if (c==1u && !inside) {
            uint at=atomic_inc(total);
            if (at<capacity) events[at]=k;
            inside=1u;
        } else if (c==2u && inside) {
            uint at=atomic_inc(total);
            if (at<capacity) events[at]=k|0x80000000u;
            inside=0u;
        }
    }
}
__kernel void burst_rows(__global const float4 *ring, uint mask,
                          __global const uint *origins, __global const float4 *lines,
                          __global float4 *burst) {
    uint row=get_global_id(0); float2 sum=(float2)(0,0);
    float fraction=lines[row].x, step=lines[row].y;
    for (uint x=98;x<138;++x)
        sum += sample_ring(ring,mask,origins[row],fraction+x*step).zw;
    float phase=atan2(sum.y,sum.x), cosine;
    float target=(row&3u)<2u ? 2.356194490192345f : 5.497787143782138f;
    float sine=sincos(target+phase,&cosine);
    float gain=lines[row].z>0.0f ? 5000.0f/fmax(length(sum)/40.0f,1e-8f) : 0.0f;
    burst[row]=(float4)(cosine,sine,gain,phase);
}
// PAL VHS alternates unrotated and -90-degree/line recording heads. Infer
// the head from the two-line burst correlation, then recover V-switch parity.
// One work-group of 256 sums the track and both polarity candidates, so the
// field does not wait on a single work item.
__kernel void pal_chroma_state(__global const float4 *burst,__global int *state,
                               __local float *sums) {
    uint lid=get_local_id(0), size=get_local_size(0);
    float track=0.0f,rotated=0.0f,unrotated=0.0f;
    for(uint row=18+lid;row<TBC_HEIGHT-10;row+=size) {
        if(burst[row].z>0.0f && burst[row-2].z>0.0f)
            track+=cos(burst[row].w-burst[row-2].w);
        if(burst[row].z>0.0f && burst[row-1].z>0.0f) {
            float delta=burst[row].w-burst[row-1].w, sign=(row&1u)?-1.0f:1.0f;
            rotated+=sign*cos(delta);
            unrotated+=sign*sin(delta);
        }
    }
    sums[lid]=track; sums[size+lid]=rotated; sums[2*size+lid]=unrotated;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride=size/2;stride>0;stride>>=1) {
        if(lid<stride) {
            sums[lid]+=sums[lid+stride];
            sums[size+lid]+=sums[size+lid+stride];
            sums[2*size+lid]+=sums[2*size+lid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid==0) state[0]=(sums[0]<0.0f ? sums[size] : sums[2*size])<0.0f;
}
void sort_gain512(__local float *values) {
    for(uint size=2;size<=512;size<<=1) for(uint stride=size>>1;stride>0;stride>>=1) {
        for(uint i=get_local_id(0);i<512;i+=256) {
            uint j=i^stride;
            if(j>i) {float a=values[i],b=values[j];if((a>b)==((i&size)==0u)){values[i]=b;values[j]=a;}}
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
// Parallel bitonic median/MAD avoids a long serial sort in private GPU memory.
__kernel void burst_gain_stats(__global const float4 *burst,__global float *cap) {
    __local float values[512];__local uint count;
    if(get_local_id(0)==0)count=0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i=get_local_id(0);i<512;i+=256) {
        float gain=i<285 ? burst[18+i].z : 0.0f;
        values[i]=gain>0.0f ? gain : INFINITY;
        if(gain>0.0f)atomic_inc(&count);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(count==0){if(get_local_id(0)==0)cap[0]=-1.0f;return;}
    sort_gain512(values);
    float mid=values[count/2];
    // Every work item must read the median before any slot is overwritten.
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i=get_local_id(0);i<512;i+=256)values[i]=i<count ? fabs(values[i]-mid) : INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    sort_gain512(values);
    if(get_local_id(0)==0)cap[0]=mid+2.0f*values[count/2];
}
__kernel void smooth_burst_gain(__global const float4 *burst,__global const float *cap,__global float *gain) {
    uint row=get_global_id(0);
    if(burst[row].z<=0.0f || cap[0]<0.0f){gain[row]=burst[row].z;return;}
    float sum=0.0f;uint count=0;
    for(int i=max(0,(int)row-4);i<min(313,(int)row+5);++i)if(burst[i].z>0.0f){sum+=fmin(burst[i].z,cap[0]);++count;}
    gain[row]=sum/count;
}
__kernel void tbc_field(__global const float4 *ring, uint mask,
                         __global const uint *origins, __global const float4 *lines,
                         __global const float4 *burst, float blank,
                         __global ushort *output,__global const int *state,__global const float *gain,__global const float *sinc_lut) {
    uint id=get_global_id(0);
    if (id>=TBC_WIDTH*TBC_HEIGHT) return;
    uint row=id/TBC_WIDTH, x=id%TBC_WIDTH;
    float4 sample=sample_video(ring,mask,origins[row],lines[row].x+x*lines[row].y,sinc_lut);
    float code=16384.0f+(sample.x-blank)*(376.32f/7000.0f);
    float2 under=sample_video(ring,mask,origins[row],lines[row].x+(x-2.0f)*lines[row].y,sinc_lut).zw;
    float2 baseband=cmul((float2)(under.x,-under.y),burst[row].xy);
    if(state[0]) baseband=(row&1u)?(float2)(baseband.y,-baseband.x):(float2)(-baseband.y,baseband.x);
    float carrier=(x&3u)==0u ? baseband.x : (x&3u)==1u ? -baseband.y :
                  (x&3u)==2u ? -baseband.x : baseband.y;
    output[id]=convert_ushort_sat(round(code));
    output[TBC_WIDTH*TBC_HEIGHT+id]=convert_ushort_sat(round(32767.0f+carrier*gain[row]));
}
// Preserve the original TBC levels and horizontal samples for PAL lines 7–22.
__kernel void vbi_luma(__global const float4 *ring, uint mask,
                       __global const uint *origins, __global const float4 *lines,
                       float blank, __global ushort *output,__global const float *sinc_lut) {
    uint x=get_global_id(0), row=get_global_id(1)+6u;
    float hz=sample_video(ring,mask,origins[row],lines[row].x+x*lines[row].y,sinc_lut).x;
    output[get_global_id(1)*1135u+x]=convert_ushort_sat(round(16384.0f+(hz-blank)*(376.32f/7000.0f)));
}
// Render only the visible luma crop. No full TBC/chroma plane crosses to the CPU.
__kernel void preview_luma(__global const float4 *ring, uint mask,
                            __global const uint *origins, __global const float4 *lines,
                            float blank, __global uchar *output,__global const float *sinc_lut) {
    uint x=get_global_id(0),row=get_global_id(1)+22u;
    float src=185.0f+(x+0.5f)*922.0f/720.0f-0.5f;
    float hz=sample_video(ring,mask,origins[row],lines[row].x+src*lines[row].y,sinc_lut).x;
    float code=16384.0f+(hz-blank)*(376.32f/7000.0f);
    output[get_global_id(1)*720u+x]=convert_uchar_sat((code-14745.6f)*(255.0f/44672.0f));
}
