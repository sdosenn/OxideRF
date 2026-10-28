// PAL 2D colour decoding adapted from PALcolour (GPL-3.0-or-later).
// Copyright 2018 William Andrew Steer; 2018-2019 Simon Inns; 2019 Adam Sampson.
// See THIRD_PARTY_NOTICES.md for the upstream source and attribution.
#define EW 1135
#define EH 313
#define OW 720
#define OH 576
#define PLANE (EW * EH)

float esin(int x) { return (x & 1) ? ((x & 2) ? -1.0f : 1.0f) : 0.0f; }
float ecos(int x) { return (x & 1) ? 0.0f : ((x & 2) ? -1.0f : 1.0f); }
float ec(__global const ushort *input, int row, int x, int active) {
    if (row < (active ? 22 : 0) || row >= (active ? 310 : EH)) return 0.0f;
    return (float)input[PLANE + row * EW + x] - 32767.0f;
}
__kernel void export_burst(__global const ushort *input, __global float *burst,
                           int begin, int end) {
    int row = get_global_id(0);
    float p=0, q=0, po=0, qo=0;
    for (int x=begin; x<end; ++x) {
        float a=(ec(input,row,x,0)-(ec(input,row-2,x,0)+ec(input,row+2,x,0))*0.5f)*0.5f;
        float b=(ec(input,row+1,x,0)-ec(input,row-1,x,0))*0.5f;
        p+=a*esin(x); q+=a*ecos(x); po+=b*esin(x); qo+=b*ecos(x);
    }
    p/=(end-begin); q/=(end-begin); po/=(end-begin); qo/=(end-begin);
    float sign=((p-po)*(p-po)+(q-qo)*(q-qo)<2*(p*p+q*q)) ? 1.0f : -1.0f;
    float bp=(p-qo)*0.5f, bq=(q+po)*0.5f;
    float norm=fmax(hypot(bp,bq),130000.0f/128.0f);
    burst[row*3]=bp/norm; burst[row*3+1]=bq/norm; burst[row*3+2]=sign;
}
__kernel void export_pal(__global const ushort *input, __global const float *burst,
                         __global const float *coeff, __global float *component,
                         int start, int end) {
    int x=get_global_id(0)+start, row=get_global_id(1)+22;
    if (x>=end) return;
    float pu=0,qu=0,pv=0,qv=0;
    for(int b=0;b<=7;++b) {
        float m0=0,m1=0,m2=0,m3=0,n0=0,n1=0,n2=0,n3=0;
        for(int side=-1;side<=1;side+=2) {
            int i=x+side*b;
            float a=ec(input,row,i,1);
            float c=-ec(input,row-2,i,1)-ec(input,row+2,i,1);
            float d=ec(input,row-1,i,1)-ec(input,row+1,i,1);
            float e=-ec(input,row-3,i,1)+ec(input,row+3,i,1);
            float s=esin(i),t=ecos(i);
            m0+=a*s; m1+=c*s; m2+=d*s; m3+=e*s;
            n0+=a*t; n1+=c*t; n2+=d*t; n3+=e*t;
        }
        float a=coeff[b*4],c=coeff[b*4+1],d=coeff[b*4+2],e=coeff[b*4+3];
        pu+=m0*a+m1*c+n2*d+n3*e; qu+=n0*a+n1*c-m2*d-m3*e;
        pv+=m0*a+m1*c-n2*d-n3*e; qv+=n0*a+n1*c+m2*d+m3*e;
    }
    float bp=burst[row*3],bq=burst[row*3+1],sign=burst[row*3+2];
    int at=row*EW+x;
    component[at]=(float)input[at];
    component[PLANE+at]=-2*(pu*bp+qu*bq);
    component[2*PLANE+at]=-2*sign*(qv*bp-pv*bq);
}
float ecomponent(__global const float *c, int plane, float x, float y, int start, int end) {
    x=clamp(x,(float)start,(float)(end-1)); y=clamp(y,22.0f,309.0f);
    int a=(int)x,b=(int)y, r=min(a+1,end-1),d=min(b+1,309);
    float tx=x-a,ty=y-b;
    int off=plane*PLANE;
    return mix(mix(c[off+b*EW+a],c[off+b*EW+r],tx),
               mix(c[off+d*EW+a],c[off+d*EW+r],tx),ty);
}
__kernel void export_nv12(__global const float *c, __global uchar *output,
                          int start,int end,int parity,float black,float range) {
    int x=get_global_id(0), y=get_global_id(1);
    float sx=start+(x+0.5f)*(end-start)/OW-0.5f;
    float sy=(44.0f+y-parity)*0.5f;
    float luma=ecomponent(c,0,sx,sy,start,end);
    output[y*OW+x]=convert_uchar_sat_rte(clamp(16.0f+(luma-black)*219.0f/range,1.0f,254.75f));
    if ((x&1)==0 && (y&1)==0) {
        float u=0,v=0;
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
            float px=start+(x+dx+0.5f)*(end-start)/OW-0.5f;
            float py=(44.0f+y+dy-parity)*0.5f;
            u+=ecomponent(c,1,px,py,start,end); v+=ecomponent(c,2,px,py,start,end);
        }
        int at=OW*OH+(y/2)*OW+x;
        output[at]=convert_uchar_sat_rte(clamp(128.0f+u*0.25f*112.0f/(0.886f*0.492111041f*range),1.0f,254.75f));
        output[at+1]=convert_uchar_sat_rte(clamp(128.0f+v*0.25f*112.0f/(0.701f*0.877283220f*range),1.0f,254.75f));
    }
}

// A single original field, without bob interpolation or opposite-field mixing.
// NV12 limits the memory-only GUI transport to 311040 bytes per PAL field.
__kernel void preview_colour_nv12(__global const float *c, __global uchar *output) {
    int x=get_global_id(0),y=get_global_id(1);
    float sx=185.0f+(x+0.5f)*922.0f/720.0f-0.5f;
    float luma=ecomponent(c,0,sx,22.0f+y,185,1107);
    output[y*720+x]=convert_uchar_sat_rte(clamp(16.0f+(luma-14745.6f)*219.0f/44672.0f,1.0f,254.75f));
    if((x&1)==0 && (y&1)==0) {
        float u=0.0f,v=0.0f;
        for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
            float px=185.0f+(x+dx+0.5f)*922.0f/720.0f-0.5f;
            u+=ecomponent(c,1,px,22.0f+y+dy,185,1107);
            v+=ecomponent(c,2,px,22.0f+y+dy,185,1107);
        }
        int at=720*288+(y/2)*720+x;
        output[at]=convert_uchar_sat_rte(clamp(128.0f+u*0.25f*112.0f/(0.886f*0.492111041f*44672.0f),1.0f,254.75f));
        output[at+1]=convert_uchar_sat_rte(clamp(128.0f+v*0.25f*112.0f/(0.701f*0.877283220f*44672.0f),1.0f,254.75f));
    }
}

// Full raster: every TBC sample and line, including sync, blanking and VBI.
// Luma comes straight from the TBC field; PAL colour exists only in the active
// picture, so chroma outside it is neutral. Line 313 pads NV12 to an even height.
// Chroma modes: 0 = colour in the active picture only, 1 = PAL colour decoded
// across the whole raster (burst and blanking included), 2 = composite Y+C in
// grey, which shows the burst and colour subcarrier as they are recorded.
#define FULL_H 314
float full_sample(__global const ushort *input, int plane, int row, float x) {
    x = clamp(x, 0.0f, (float)(EW - 1));
    int a = (int)x, b = min(a + 1, EW - 1);
    int at = plane * PLANE + row * EW;
    return mix((float)input[at + a], (float)input[at + b], x - a);
}
float full_luma(__global const ushort *input, int row, float x, int mode) {
    if (row >= EH) return 14745.6f;
    float luma = full_sample(input, 0, row, x);
    return mode == 2 ? luma + full_sample(input, 1, row, x) - 32767.0f : luma;
}
float ecomponent_whole(__global const float *c, int plane, float x, float y) {
    x = clamp(x, 0.0f, (float)(EW - 1)); y = clamp(y, 0.0f, (float)(EH - 1));
    int a = (int)x, b = (int)y, r = min(a + 1, EW - 1), d = min(b + 1, EH - 1);
    float tx = x - a, ty = y - b;
    int off = plane * PLANE;
    return mix(mix(c[off + b * EW + a], c[off + b * EW + r], tx),
               mix(c[off + d * EW + a], c[off + d * EW + r], tx), ty);
}
float full_chroma(__global const float *c, int plane, int row, float x, int mode) {
    if (mode == 2 || row >= EH) return 0.0f;
    if (mode == 1) return ecomponent_whole(c, plane, x, (float)row);
    if (row < 22 || row >= 310 || x < 185.0f || x > 1106.0f) return 0.0f;
    return ecomponent(c, plane, x, (float)row, 185, 1107);
}
float ecf(__global const ushort *input, int row, int x) {
    if (row < 0 || row >= EH || x < 0 || x >= EW) return 0.0f;
    return (float)input[PLANE + row * EW + x] - 32767.0f;
}
// export_pal over every line and sample, for whole-raster colour.
__kernel void export_pal_full(__global const ushort *input, __global const float *burst,
                              __global const float *coeff, __global float *component) {
    int x = get_global_id(0), row = get_global_id(1);
    float pu=0,qu=0,pv=0,qv=0;
    for(int b=0;b<=7;++b) {
        float m0=0,m1=0,m2=0,m3=0,n0=0,n1=0,n2=0,n3=0;
        for(int side=-1;side<=1;side+=2) {
            int i=x+side*b;
            float a=ecf(input,row,i);
            float c=-ecf(input,row-2,i)-ecf(input,row+2,i);
            float d=ecf(input,row-1,i)-ecf(input,row+1,i);
            float e=-ecf(input,row-3,i)+ecf(input,row+3,i);
            float s=esin(i),t=ecos(i);
            m0+=a*s; m1+=c*s; m2+=d*s; m3+=e*s;
            n0+=a*t; n1+=c*t; n2+=d*t; n3+=e*t;
        }
        float a=coeff[b*4],c=coeff[b*4+1],d=coeff[b*4+2],e=coeff[b*4+3];
        pu+=m0*a+m1*c+n2*d+n3*e; qu+=n0*a+n1*c-m2*d-m3*e;
        pv+=m0*a+m1*c-n2*d-n3*e; qv+=n0*a+n1*c+m2*d+m3*e;
    }
    float bp=burst[row*3],bq=burst[row*3+1],sign=burst[row*3+2];
    int at=row*EW+x;
    component[at]=(float)input[at];
    component[PLANE+at]=-2*(pu*bp+qu*bq);
    component[2*PLANE+at]=-2*sign*(qv*bp-pv*bq);
}
__kernel void preview_full_nv12(__global const ushort *input, __global const float *c,
                                __global uchar *output, int width, int mode) {
    int x = get_global_id(0), y = get_global_id(1);
    float sx = (x + 0.5f) * EW / (float)width - 0.5f;
    output[y * width + x] = convert_uchar_sat_rte(
        clamp(16.0f + (full_luma(input, y, sx, mode) - 14745.6f) * 219.0f / 44672.0f, 1.0f, 254.75f));
    if ((x & 1) == 0 && (y & 1) == 0) {
        float u = 0.0f, v = 0.0f;
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            float px = (x + dx + 0.5f) * EW / (float)width - 0.5f;
            u += full_chroma(c, 1, y + dy, px, mode);
            v += full_chroma(c, 2, y + dy, px, mode);
        }
        int at = width * FULL_H + (y / 2) * width + x;
        output[at] = convert_uchar_sat_rte(clamp(128.0f + u * 0.25f * 112.0f / (0.886f * 0.492111041f * 44672.0f), 1.0f, 254.75f));
        output[at + 1] = convert_uchar_sat_rte(clamp(128.0f + v * 0.25f * 112.0f / (0.701f * 0.877283220f * 44672.0f), 1.0f, 254.75f));
    }
}
