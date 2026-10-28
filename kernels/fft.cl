// Single precision is deliberate: Arc A750 has no native FP64.
// Twiddles are calculated in f64 on the host and rounded once.
uint reverse_bits(uint x, uint bits) {
    uint r = 0;
    for (uint b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}
// Fuse bit reversal and the first nine stages in 4 KiB of local memory.
// One workgroup owns 512 samples, with two samples per work item.
__kernel void local_begin(__global const float2 *src, __global float2 *dst,
                          __global const float2 *twiddle, uint bits, uint n, int inverse) {
    __local float2 tile[512];
    uint batch = get_global_id(1) * n;
    src += batch; dst += batch;
    uint lid = get_local_id(0), base = get_group_id(0) * 512;
    tile[lid] = src[reverse_bits(base + lid, bits)];
    tile[lid+256] = src[reverse_bits(base + lid+256, bits)];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint width = 2; width <= 512; width <<= 1) {
        uint h = width >> 1, j = lid % h, a = (lid / h) * width + j;
        float2 w = twiddle[j * (n/width)];
        if (inverse) w.y = -w.y;
        float2 u = tile[a], v = tile[a+h];
        float2 t = (float2)(w.x*v.x - w.y*v.y, w.x*v.y + w.y*v.x);
        tile[a] = u+t; tile[a+h] = u-t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dst[base+lid] = tile[lid]; dst[base+lid+256] = tile[lid+256];
}
__kernel void reorder(__global const float2 *src, __global float2 *dst,
                      uint bits, uint n) {
    src += get_global_id(1) * n; dst += get_global_id(1) * n;
    uint i = get_global_id(0), r = 0, x = i;
    for (uint b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
    dst[r] = src[i];
}
__kernel void stage(__global float2 *data, __global const float2 *twiddle,
                    uint n, uint width, int inverse) {
    data += get_global_id(1) * n;
    uint i = get_global_id(0), halfwidth = width >> 1;
    uint j = i % halfwidth, a = (i / halfwidth) * width + j;
    float2 w = twiddle[j * (n / width)];
    if (inverse) w.y = -w.y;
    float2 u = data[a], v = data[a + halfwidth];
    float2 t = (float2)(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x);
    data[a] = u + t; data[a + halfwidth] = u - t;
}
// Fuse the remaining stages of 1K..64K transforms. Each group owns one
// residue modulo 512, so all cross-tile butterflies remain local here.
__kernel void local_end(__global float2 *data, __global const float2 *twiddle,
                        uint n, int inverse) {
    data += get_global_id(1) * n;
    __local float2 tile[128];
    uint count = n/512, lid = get_local_id(0), residue = get_group_id(0);
    tile[lid] = data[residue+lid*512];
    tile[lid+count/2] = data[residue+(lid+count/2)*512];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint width = 2; width <= count; width <<= 1) {
        uint h = width/2, j = lid%h, a = (lid/h)*width+j;
        float2 w = twiddle[(j*512+residue)*(n/(width*512))];
        if (inverse) w.y = -w.y;
        float2 u = tile[a], v = tile[a+h];
        float2 t = (float2)(w.x*v.x-w.y*v.y, w.x*v.y+w.y*v.x);
        tile[a] = u+t; tile[a+h] = u-t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    data[residue+lid*512] = tile[lid];
    data[residue+(lid+count/2)*512] = tile[lid+count/2];
}
