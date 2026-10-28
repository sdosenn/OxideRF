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
__kernel void pack_rf(__global const float *raw, __global const float *means,
                       __global float2 *dst) {
    uint i = get_global_id(0), b = get_global_id(1);
    dst[b*RF_N+i] = (float2)(raw[b*RF_HOP+i]-means[b], 0.0f);
}
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
__kernel void refine_lines(__global const float4 *ring, uint mask,
                            __global const uint *origins, __global const float4 *lines,
                            float samples_per_us, __global float4 *result, float threshold) {
    uint row=get_global_id(0), origin=origins[row];
    float edge=lines[row].x;
    if (lines[row].z == 0.0f) { result[row]=(float4)(edge,0,0,0); return; }
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
    result[row]=(float4)(best,median8(tips),median8(porches),valid);
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
__kernel void pal_chroma_state(__global const float4 *burst,__global int *state) {
    float track=0.0f,polarity=0.0f;
    for(uint row=18;row<TBC_HEIGHT-10;++row)
        if(burst[row].z>0.0f && burst[row-2].z>0.0f)
            track+=cos(burst[row].w-burst[row-2].w);
    for(uint row=18;row<TBC_HEIGHT-10;++row) {
        if(burst[row].z>0.0f && burst[row-1].z>0.0f) {
            float delta=burst[row].w-burst[row-1].w;
            polarity+=((row&1u)?-1.0f:1.0f)*(track<0.0f?cos(delta):sin(delta));
        }
    }
    state[0]=polarity<0.0f;
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
