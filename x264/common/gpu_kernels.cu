// Device code for the CUDA candidate generator.
//
// The kernels do a coarse exhaustive search on a quarter-resolution luma copy
// and then refine the best four integer-pel positions at full resolution.
// They are *open-loop*: the host side never writes a coding decision, and
// me.c re-evaluates every returned vector with the real x264 cost function.
//
// This file contains no host code.  It is compiled with `nvcc -fatbin` and
// the resulting fat binary is embedded in x264.exe (see gpu_cuda.cpp), so the
// kernels can be built with a toolkit that still supports pre-Turing GPUs
// even though the host code is compiled by the (much newer) MSVC toolchain.
//
// The kernels are declared extern "C" so that cuModuleGetFunction() can find
// them by their plain name.

#include "gpu_kernels.h"

/* CUDA's device min()/max() are declared in the runtime headers; provide them
 * here so the kernels compile against the minimal cuda_runtime.h stub (see
 * cuda_stub/) that keeps the C++ standard library out of the device build. */
__device__ __forceinline__ int gpu_min( int a, int b ) { return a < b ? a : b; }
__device__ __forceinline__ int gpu_max( int a, int b ) { return a > b ? a : b; }

__device__ __forceinline__ uint32_t sad4x4( const uint8_t *a, int as,
                                            const uint8_t *b, int bs )
{
    uint32_t s = 0;
#pragma unroll
    for( int j = 0; j < 4; j++ )
#pragma unroll
        for( int i = 0; i < 4; i++ )
        {
            int d = (int)a[j*as+i] - (int)b[j*bs+i];
            s += (uint32_t)(d < 0 ? -d : d);
        }
    return s;
}

__device__ __forceinline__ void insert_topk( Cand *best, int x, int y, uint32_t sad )
{
    if( sad >= best[GK-1].sad )
        return;
    int i = GK - 1;
    while( i > 0 && sad < best[i-1].sad )
    {
        best[i] = best[i-1];
        i--;
    }
    best[i].mv[0] = (int16_t)x;
    best[i].mv[1] = (int16_t)y;
    best[i].sad = sad;
}

extern "C" __global__ void downsample4( const uint8_t *__restrict__ src, int sw, int sh,
                                        uint8_t *__restrict__ dst, int dw, int dh )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= dw || y >= dh )
        return;
    int sx = x * 4, sy = y * 4;
    int sum = 0;
#pragma unroll
    for( int j = 0; j < 4; j++ )
#pragma unroll
        for( int i = 0; i < 4; i++ )
        {
            int px = gpu_min( sx + i, sw - 1 );
            int py = gpu_min( sy + j, sh - 1 );
            sum += src[py * sw + px];
        }
    dst[y * dw + x] = (uint8_t)((sum + 8) >> 4);
}

extern "C" __global__ void coarse_search( const uint8_t *__restrict__ cur, int cw, int ch,
                                          const uint8_t *__restrict__ ref, int rw, int rh,
                                          int mbw, int mbh, int range, Cand *out )
{
    int mb = blockIdx.x * blockDim.x + threadIdx.x;
    if( mb >= mbw * mbh )
        return;
    int mx = mb % mbw, my = mb / mbw;
    int bx = mx * 4, by = my * 4;
    Cand best[GK];
#pragma unroll
    for( int i = 0; i < GK; i++ )
    {
        best[i].mv[0] = 0;
        best[i].mv[1] = 0;
        best[i].sad = 0xFFFFFFFFu;
    }
    /* The full-resolution range is expressed in luma pixels; the quarter
     * resolution grid covers it with range/4 steps.  The first pass runs on
     * a two-step grid, the second pass tests the eight neighbours of the
     * winner, so the grid is only ~1/4 as expensive with practically the
     * same result. */
    int r = range / 4;
    if( r < 1 )
        r = 1;
    for( int dy = -r; dy <= r; dy += 2 )
        for( int dx = -r; dx <= r; dx += 2 )
        {
            int x = gpu_max( 0, gpu_min( bx + dx, cw - 4 ) );
            int y = gpu_max( 0, gpu_min( by + dy, ch - 4 ) );
            uint32_t sad = sad4x4( cur + by * cw + bx, cw, ref + y * rw + x, rw );
            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
        }
    int fx = bx + best[0].mv[0] / 4;
    int fy = by + best[0].mv[1] / 4;
    for( int dy = -1; dy <= 1; dy++ )
        for( int dx = -1; dx <= 1; dx++ )
        {
            if( !dx && !dy )
                continue;
            int x = gpu_max( 0, gpu_min( fx + dx, cw - 4 ) );
            int y = gpu_max( 0, gpu_min( fy + dy, ch - 4 ) );
            uint32_t sad = sad4x4( cur + by * cw + bx, cw, ref + y * rw + x, rw );
            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
        }
    for( int i = 0; i < GK; i++ )
        out[mb * GK + i] = best[i];
}

extern "C" __global__ void refine_search( const uint8_t *__restrict__ cur, int cw, int ch,
                                          const uint8_t *__restrict__ ref, int rw, int rh,
                                          int mbw, int mbh, int radius,
                                          const Cand *__restrict__ coarse, Cand *out )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mbw * mbh * GK;
    if( idx >= total )
        return;
    int mb = idx / GK;
    int mx = mb % mbw, my = mb / mbw;
    int bx = mx * 16, by = my * 16;
    int cx = bx + coarse[idx].mv[0];
    int cy = by + coarse[idx].mv[1];
    const uint8_t *pc = cur + by * cw + bx;
    Cand best;
    best.mv[0] = 0;
    best.mv[1] = 0;
    best.sad = 0xFFFFFFFFu;
    for( int dy = -radius; dy <= radius; dy++ )
        for( int dx = -radius; dx <= radius; dx++ )
        {
            int x = gpu_max( 0, gpu_min( cx + dx, cw - 16 ) );
            int y = gpu_max( 0, gpu_min( cy + dy, ch - 16 ) );
            const uint8_t *pr = ref + y * rw + x;
            uint32_t sad = 0;
            /* Subsample the 16x16 block with a stride of two pixels.  The
             * candidates are re-scored on the CPU with the real x264 metric,
             * so this only needs to rank positions, not to be exact. */
#pragma unroll
            for( int j = 0; j < 16; j += 2 )
#pragma unroll
                for( int i = 0; i < 16; i += 2 )
                {
                    int d = (int)pc[j*cw+i] - (int)pr[j*rw+i];
                    sad += (uint32_t)(d < 0 ? -d : d);
                }
            if( sad < best.sad )
            {
                best.sad = sad;
                best.mv[0] = (int16_t)(x - bx);
                best.mv[1] = (int16_t)(y - by);
            }
        }
    out[idx] = best;
}
