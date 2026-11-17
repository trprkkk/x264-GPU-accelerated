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
// them by their plain name.  Each kernel exists in two pixel flavours: the
// plain names operate on 8-bit planes, the *_10 names on the two-byte pixels
// of the 10-bit (and higher) builds.  Both are always compiled into every
// fat binary; the host side looks up the one matching the encoder's depth.

#include "gpu_kernels.h"

/* CUDA's device min()/max() are declared in the runtime headers; provide them
 * here so the kernels compile against the minimal cuda_runtime.h stub (see
 * cuda_stub/) that keeps the C++ standard library out of the device build. */
__device__ __forceinline__ int gpu_min( int a, int b ) { return a < b ? a : b; }
__device__ __forceinline__ int gpu_max( int a, int b ) { return a > b ? a : b; }

/* --- pixel-type-generic implementations ---------------------------------- */

template<class P>
__device__ __forceinline__ uint32_t sad4x4_t( const P *a, int as, const P *b, int bs )
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

template<class P>
__device__ __forceinline__ void downsample4_impl( const P *__restrict__ src, int sw, int sh,
                                                  P *__restrict__ dst, int dw, int dh )
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
    dst[y * dw + x] = (P)((sum + 8) >> 4);
}

template<class P>
__device__ __forceinline__ void coarse_search_impl( const P *__restrict__ cur, int cw, int ch,
                                                    const P *__restrict__ ref, int rw, int rh,
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
            uint32_t sad = sad4x4_t( cur + by * cw + bx, cw, ref + y * rw + x, rw );
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
            uint32_t sad = sad4x4_t( cur + by * cw + bx, cw, ref + y * rw + x, rw );
            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
        }
    for( int i = 0; i < GK; i++ )
        out[mb * GK + i] = best[i];
}

template<class P>
__device__ __forceinline__ void refine_search_impl( const P *__restrict__ cur, int cw, int ch,
                                                    const P *__restrict__ ref, int rw, int rh,
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
    const P *pc = cur + by * cw + bx;
    Cand best;
    best.mv[0] = 0;
    best.mv[1] = 0;
    best.sad = 0xFFFFFFFFu;
    for( int dy = -radius; dy <= radius; dy++ )
        for( int dx = -radius; dx <= radius; dx++ )
        {
            int x = gpu_max( 0, gpu_min( cx + dx, cw - 16 ) );
            int y = gpu_max( 0, gpu_min( cy + dy, ch - 16 ) );
            const P *pr = ref + y * rw + x;
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

/* --- 8-bit entry points --------------------------------------------------- */

extern "C" __global__ void downsample4( const uint8_t *__restrict__ src, int sw, int sh,
                                        uint8_t *__restrict__ dst, int dw, int dh )
{
    downsample4_impl( src, sw, sh, dst, dw, dh );
}

extern "C" __global__ void coarse_search( const uint8_t *__restrict__ cur, int cw, int ch,
                                          const uint8_t *__restrict__ ref, int rw, int rh,
                                          int mbw, int mbh, int range, Cand *out )
{
    coarse_search_impl( cur, cw, ch, ref, rw, rh, mbw, mbh, range, out );
}

extern "C" __global__ void refine_search( const uint8_t *__restrict__ cur, int cw, int ch,
                                          const uint8_t *__restrict__ ref, int rw, int rh,
                                          int mbw, int mbh, int radius,
                                          const Cand *__restrict__ coarse, Cand *out )
{
    refine_search_impl( cur, cw, ch, ref, rw, rh, mbw, mbh, radius, coarse, out );
}

/* --- 10-bit (and higher) entry points ------------------------------------- */

extern "C" __global__ void downsample4_10( const uint16_t *__restrict__ src, int sw, int sh,
                                           uint16_t *__restrict__ dst, int dw, int dh )
{
    downsample4_impl( src, sw, sh, dst, dw, dh );
}

extern "C" __global__ void coarse_search_10( const uint16_t *__restrict__ cur, int cw, int ch,
                                             const uint16_t *__restrict__ ref, int rw, int rh,
                                             int mbw, int mbh, int range, Cand *out )
{
    coarse_search_impl( cur, cw, ch, ref, rw, rh, mbw, mbh, range, out );
}

extern "C" __global__ void refine_search_10( const uint16_t *__restrict__ cur, int cw, int ch,
                                             const uint16_t *__restrict__ ref, int rw, int rh,
                                             int mbw, int mbh, int radius,
                                             const Cand *__restrict__ coarse, Cand *out )
{
    refine_search_impl( cur, cw, ch, ref, rw, rh, mbw, mbh, radius, coarse, out );
}

/* ================== lookahead (lowres) kernels ============================
 * Exact transliterations of the C reference code the CPU lookahead executes
 * (common/pixel.c satd/sad, common/predict.c 8x8c and 8x8 predictors), so the
 * host can substitute the GPU results for CPU computation without changing
 * any decision.  Every operation, including pixel.c's packed-lane Hadamard
 * trick, is replicated verbatim; the results are bit-identical by
 * construction and are verified against the CPU by the fork's selftests.
 *
 * Conventions (mirroring slicetype.c's lowres analysis):
 *   - one macroblock is an 8x8 block on the half-resolution plane,
 *   - fenc blocks use FENC_STRIDE (16 pixels), the prediction scratch
 *     buffer uses FDEC_STRIDE (32 pixels), as in the CPU code.
 * ======================================================================== */

template<class S2, class ST, int BPS>
__device__ __forceinline__ S2 la_abs2( S2 a )
{
    S2 s = ((a >> (BPS - 1)) & (((S2)1 << BPS) + 1)) * (S2)(ST)-1;
    return (a + s) ^ s;
}

template<class S2>
__device__ __forceinline__ void la_had4( S2 &d0, S2 &d1, S2 &d2, S2 &d3,
                                         S2 s0, S2 s1, S2 s2, S2 s3 )
{
    S2 t0 = s0 + s1;
    S2 t1 = s0 - s1;
    S2 t2 = s2 + s3;
    S2 t3 = s2 - s3;
    d0 = t0 + t2;
    d2 = t0 - t2;
    d1 = t1 + t3;
    d3 = t1 - t3;
}

/* x264_pixel_satd_8x4_c.  fenc rows are 8 ints apart, candidate rows 32
 * (FDEC_STRIDE) ints apart, exactly like the CPU call. */
template<class S2, class ST, int BPS>
__device__ __forceinline__ int la_satd_8x4( const int *fenc, const int *cand )
{
    S2 tmp[4][4];
    S2 a0, a1, a2, a3;
    S2 sum = 0;
#pragma unroll
    for( int i = 0; i < 4; i++, fenc += 8, cand += 32 )
    {
        S2 d0 = (S2)(fenc[0] - cand[0]) + ((S2)(fenc[4] - cand[4]) << BPS);
        S2 d1 = (S2)(fenc[1] - cand[1]) + ((S2)(fenc[5] - cand[5]) << BPS);
        S2 d2 = (S2)(fenc[2] - cand[2]) + ((S2)(fenc[6] - cand[6]) << BPS);
        S2 d3 = (S2)(fenc[3] - cand[3]) + ((S2)(fenc[7] - cand[7]) << BPS);
        la_had4( tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], d0, d1, d2, d3 );
    }
#pragma unroll
    for( int i = 0; i < 4; i++ )
    {
        la_had4( a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i] );
        sum += la_abs2<S2, ST, BPS>( a0 ) + la_abs2<S2, ST, BPS>( a1 )
             + la_abs2<S2, ST, BPS>( a2 ) + la_abs2<S2, ST, BPS>( a3 );
    }
    return (int)((((ST)sum) + (sum >> BPS)) >> 1);
}

template<class S2, class ST, int BPS>
__device__ __forceinline__ int la_satd_8x8( const int *fenc, const int *cand )
{
    return la_satd_8x4<S2, ST, BPS>( fenc, cand )
         + la_satd_8x4<S2, ST, BPS>( fenc + 4 * 8, cand + 4 * 32 );
}

/* Same transform as la_satd_8x4 but reading straight from strided pixel
 * planes, so the lookahead table can produce SATD entries when the CPU
 * search's fpelcmp resolves to satd (me_method TESA). */
template<class S2, class ST, int BPS, class P>
__device__ __forceinline__ int la_satd_8x4_raw( const P *fenc, int fs, const P *cand, int cs )
{
    S2 tmp[4][4];
    S2 a0, a1, a2, a3;
    S2 sum = 0;
#pragma unroll
    for( int i = 0; i < 4; i++ )
    {
        const P *fr = fenc + i * fs;
        const P *cr = cand + i * cs;
        S2 d0 = (S2)((int)fr[0] - (int)cr[0]) + ((S2)((int)fr[4] - (int)cr[4]) << BPS);
        S2 d1 = (S2)((int)fr[1] - (int)cr[1]) + ((S2)((int)fr[5] - (int)cr[5]) << BPS);
        S2 d2 = (S2)((int)fr[2] - (int)cr[2]) + ((S2)((int)fr[6] - (int)cr[6]) << BPS);
        S2 d3 = (S2)((int)fr[3] - (int)cr[3]) + ((S2)((int)fr[7] - (int)cr[7]) << BPS);
        la_had4( tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], d0, d1, d2, d3 );
    }
#pragma unroll
    for( int i = 0; i < 4; i++ )
    {
        la_had4( a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i] );
        sum += la_abs2<S2, ST, BPS>( a0 ) + la_abs2<S2, ST, BPS>( a1 )
             + la_abs2<S2, ST, BPS>( a2 ) + la_abs2<S2, ST, BPS>( a3 );
    }
    return (int)((((ST)sum) + (sum >> BPS)) >> 1);
}

template<class S2, class ST, int BPS, class P>
__device__ __forceinline__ int la_satd_8x8_raw( const P *fenc, int fs, const P *cand, int cs )
{
    return la_satd_8x4_raw<S2, ST, BPS, P>( fenc, fs, cand, cs )
         + la_satd_8x4_raw<S2, ST, BPS, P>( fenc + 4 * fs, fs, cand + 4 * cs, cs );
}

__device__ __forceinline__ int la_sad_8x8( const int *fenc, const int *cand )
{
    int s = 0;
#pragma unroll
    for( int j = 0; j < 8; j++ )
#pragma unroll
        for( int i = 0; i < 8; i++ )
        {
            int d = fenc[j * 8 + i] - cand[j * 32 + i];
            s += d < 0 ? -d : d;
        }
    return s;
}

template<class P>
__device__ __forceinline__ int la_satd_8x8_p( const int *fenc, const int *cand )
{
    if( sizeof(P) == 1 )
        return la_satd_8x8<uint32_t, uint16_t, 16>( fenc, cand );
    return la_satd_8x8<uint64_t, uint32_t, 32>( fenc, cand );
}

/* --- predictions on an FDEC-style scratch buffer (stride 32 ints) --------- */

#define LA_F1( a, b )    (((a) + (b) + 1) >> 1)
#define LA_F2( a, b, c ) (((a) + 2 * (b) + (c) + 2) >> 2)

__device__ __forceinline__ void la_pred_8x8c_dc( int *src )
{
    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
    for( int i = 0; i < 4; i++ )
    {
        s0 += src[i - 32];
        s1 += src[i + 4 - 32];
        s2 += src[-1 + i * 32];
        s3 += src[-1 + (i + 4) * 32];
    }
    int dc0 = ( s0 + s2 + 4 ) >> 3;
    int dc1 = ( s1 + 2 ) >> 2;
    int dc2 = ( s3 + 2 ) >> 2;
    int dc3 = ( s1 + s3 + 4 ) >> 3;
#pragma unroll
    for( int y = 0; y < 4; y++, src += 32 )
#pragma unroll
        for( int x = 0; x < 8; x++ )
            src[x] = x < 4 ? dc0 : dc1;
#pragma unroll
    for( int y = 0; y < 4; y++, src += 32 )
#pragma unroll
        for( int x = 0; x < 8; x++ )
            src[x] = x < 4 ? dc2 : dc3;
}

__device__ __forceinline__ void la_pred_8x8c_h( int *src )
{
#pragma unroll
    for( int y = 0; y < 8; y++ )
#pragma unroll
        for( int x = 0; x < 8; x++ )
            src[y * 32 + x] = src[y * 32 - 1];
}

__device__ __forceinline__ void la_pred_8x8c_v( int *src )
{
#pragma unroll
    for( int y = 0; y < 8; y++ )
#pragma unroll
        for( int x = 0; x < 8; x++ )
            src[y * 32 + x] = src[x - 32];
}

__device__ __forceinline__ void la_pred_8x8c_p( int *src, int maxpx )
{
    int H = 0, V = 0;
#pragma unroll
    for( int i = 0; i < 4; i++ )
    {
        H += ( i + 1 ) * ( src[4 + i - 32] - src[2 - i - 32] );
        V += ( i + 1 ) * ( src[-1 + (i + 4) * 32] - src[-1 + (2 - i) * 32] );
    }

    int a = 16 * ( src[-1 + 7 * 32] + src[7 - 32] );
    int b = ( 17 * H + 16 ) >> 5;
    int c = ( 17 * V + 16 ) >> 5;
    int i00 = a - 3 * b - 3 * c + 16;

#pragma unroll
    for( int y = 0; y < 8; y++, src += 32 )
    {
        int pix = i00;
#pragma unroll
        for( int x = 0; x < 8; x++ )
        {
            int v = pix >> 5;
            src[x] = v < 0 ? 0 : (v > maxpx ? maxpx : v);
            pix += b;
        }
        i00 += c;
    }
}

/* predict_8x8_filter_c for ALL_NEIGHBORS (topleft and topright present),
 * which is what slicetype's intra pre-pass uses. */
__device__ __forceinline__ void la_pred_8x8_filter_all( int *src, int edge[36] )
{
#define LASRC( x, y ) src[(x) + (y) * 32]
    /* left */
    edge[15] = ( LASRC( 0, -1 ) + 2 * LASRC( -1, -1 ) + LASRC( -1, 0 ) + 2 ) >> 2;
    edge[14] = ( LASRC( -1, -1 ) + 2 * LASRC( -1, 0 ) + LASRC( -1, 1 ) + 2 ) >> 2;
#pragma unroll
    for( int y = 1; y <= 6; y++ )
        edge[14 - y] = LA_F2( LASRC( -1, y - 1 ), LASRC( -1, y ), LASRC( -1, y + 1 ) );
    edge[6] = edge[7] = ( LASRC( -1, 6 ) + 3 * LASRC( -1, 7 ) + 2 ) >> 2;
    /* top */
    edge[16] = ( LASRC( -1, -1 ) + 2 * LASRC( 0, -1 ) + LASRC( 1, -1 ) + 2 ) >> 2;
#pragma unroll
    for( int x = 1; x <= 6; x++ )
        edge[16 + x] = LA_F2( LASRC( x - 1, -1 ), LASRC( x, -1 ), LASRC( x + 1, -1 ) );
    edge[23] = ( LASRC( 6, -1 ) + 2 * LASRC( 7, -1 ) + LASRC( 8, -1 ) + 2 ) >> 2;
    /* topright */
#pragma unroll
    for( int x = 8; x <= 14; x++ )
        edge[16 + x] = LA_F2( LASRC( x - 1, -1 ), LASRC( x, -1 ), LASRC( x + 1, -1 ) );
    edge[31] = edge[32] = ( LASRC( 14, -1 ) + 3 * LASRC( 15, -1 ) + 2 ) >> 2;
#undef LASRC
}

__device__ __forceinline__ void la_pred_8x8_ddl( int *src, const int edge[36] )
{
    int t0 = edge[16], t1 = edge[17], t2 = edge[18], t3 = edge[19];
    int t4 = edge[20], t5 = edge[21], t6 = edge[22], t7 = edge[23];
    int t8 = edge[24], t9 = edge[25], t10 = edge[26], t11 = edge[27];
    int t12 = edge[28], t13 = edge[29], t14 = edge[30], t15 = edge[31];
#define LDDLSRC( x, y ) src[(x) + (y) * 32]
    LDDLSRC( 0, 0 ) = LA_F2( t0, t1, t2 );
    LDDLSRC( 0, 1 ) = LDDLSRC( 1, 0 ) = LA_F2( t1, t2, t3 );
    LDDLSRC( 0, 2 ) = LDDLSRC( 1, 1 ) = LDDLSRC( 2, 0 ) = LA_F2( t2, t3, t4 );
    LDDLSRC( 0, 3 ) = LDDLSRC( 1, 2 ) = LDDLSRC( 2, 1 ) = LDDLSRC( 3, 0 ) = LA_F2( t3, t4, t5 );
    LDDLSRC( 0, 4 ) = LDDLSRC( 1, 3 ) = LDDLSRC( 2, 2 ) = LDDLSRC( 3, 1 ) = LDDLSRC( 4, 0 ) = LA_F2( t4, t5, t6 );
    LDDLSRC( 0, 5 ) = LDDLSRC( 1, 4 ) = LDDLSRC( 2, 3 ) = LDDLSRC( 3, 2 ) = LDDLSRC( 4, 1 ) = LDDLSRC( 5, 0 ) = LA_F2( t5, t6, t7 );
    LDDLSRC( 0, 6 ) = LDDLSRC( 1, 5 ) = LDDLSRC( 2, 4 ) = LDDLSRC( 3, 3 ) = LDDLSRC( 4, 2 ) = LDDLSRC( 5, 1 ) = LDDLSRC( 6, 0 ) = LA_F2( t6, t7, t8 );
    LDDLSRC( 0, 7 ) = LDDLSRC( 1, 6 ) = LDDLSRC( 2, 5 ) = LDDLSRC( 3, 4 ) = LDDLSRC( 4, 3 ) = LDDLSRC( 5, 2 ) = LDDLSRC( 6, 1 ) = LDDLSRC( 7, 0 ) = LA_F2( t7, t8, t9 );
    LDDLSRC( 1, 7 ) = LDDLSRC( 2, 6 ) = LDDLSRC( 3, 5 ) = LDDLSRC( 4, 4 ) = LDDLSRC( 5, 3 ) = LDDLSRC( 6, 2 ) = LDDLSRC( 7, 1 ) = LA_F2( t8, t9, t10 );
    LDDLSRC( 2, 7 ) = LDDLSRC( 3, 6 ) = LDDLSRC( 4, 5 ) = LDDLSRC( 5, 4 ) = LDDLSRC( 6, 3 ) = LDDLSRC( 7, 2 ) = LA_F2( t9, t10, t11 );
    LDDLSRC( 3, 7 ) = LDDLSRC( 4, 6 ) = LDDLSRC( 5, 5 ) = LDDLSRC( 6, 4 ) = LDDLSRC( 7, 3 ) = LA_F2( t10, t11, t12 );
    LDDLSRC( 4, 7 ) = LDDLSRC( 5, 6 ) = LDDLSRC( 6, 5 ) = LDDLSRC( 7, 4 ) = LA_F2( t11, t12, t13 );
    LDDLSRC( 5, 7 ) = LDDLSRC( 6, 6 ) = LDDLSRC( 7, 5 ) = LA_F2( t12, t13, t14 );
    LDDLSRC( 6, 7 ) = LDDLSRC( 7, 6 ) = LA_F2( t13, t14, t15 );
    LDDLSRC( 7, 7 ) = LA_F2( t14, t15, t15 );
#undef LDDLSRC
}

__device__ __forceinline__ void la_pred_8x8_ddr( int *src, const int edge[36] )
{
    int l0 = edge[14], l1 = edge[13], l2 = edge[12], l3 = edge[11];
    int l4 = edge[10], l5 = edge[9], l6 = edge[8], l7 = edge[7];
    int lt = edge[15];
    int t0 = edge[16], t1 = edge[17], t2 = edge[18], t3 = edge[19];
    int t4 = edge[20], t5 = edge[21], t6 = edge[22], t7 = edge[23];
#define LDDRSRC( x, y ) src[(x) + (y) * 32]
    LDDRSRC( 0, 7 ) = LA_F2( l7, l6, l5 );
    LDDRSRC( 0, 6 ) = LDDRSRC( 1, 7 ) = LA_F2( l6, l5, l4 );
    LDDRSRC( 0, 5 ) = LDDRSRC( 1, 6 ) = LDDRSRC( 2, 7 ) = LA_F2( l5, l4, l3 );
    LDDRSRC( 0, 4 ) = LDDRSRC( 1, 5 ) = LDDRSRC( 2, 6 ) = LDDRSRC( 3, 7 ) = LA_F2( l4, l3, l2 );
    LDDRSRC( 0, 3 ) = LDDRSRC( 1, 4 ) = LDDRSRC( 2, 5 ) = LDDRSRC( 3, 6 ) = LDDRSRC( 4, 7 ) = LA_F2( l3, l2, l1 );
    LDDRSRC( 0, 2 ) = LDDRSRC( 1, 3 ) = LDDRSRC( 2, 4 ) = LDDRSRC( 3, 5 ) = LDDRSRC( 4, 6 ) = LDDRSRC( 5, 7 ) = LA_F2( l2, l1, l0 );
    LDDRSRC( 0, 1 ) = LDDRSRC( 1, 2 ) = LDDRSRC( 2, 3 ) = LDDRSRC( 3, 4 ) = LDDRSRC( 4, 5 ) = LDDRSRC( 5, 6 ) = LDDRSRC( 6, 7 ) = LA_F2( l1, l0, lt );
    LDDRSRC( 0, 0 ) = LDDRSRC( 1, 1 ) = LDDRSRC( 2, 2 ) = LDDRSRC( 3, 3 ) = LDDRSRC( 4, 4 ) = LDDRSRC( 5, 5 ) = LDDRSRC( 6, 6 ) = LDDRSRC( 7, 7 ) = LA_F2( l0, lt, t0 );
    LDDRSRC( 1, 0 ) = LDDRSRC( 2, 1 ) = LDDRSRC( 3, 2 ) = LDDRSRC( 4, 3 ) = LDDRSRC( 5, 4 ) = LDDRSRC( 6, 5 ) = LDDRSRC( 7, 6 ) = LA_F2( lt, t0, t1 );
    LDDRSRC( 2, 0 ) = LDDRSRC( 3, 1 ) = LDDRSRC( 4, 2 ) = LDDRSRC( 5, 3 ) = LDDRSRC( 6, 4 ) = LDDRSRC( 7, 5 ) = LA_F2( t0, t1, t2 );
    LDDRSRC( 3, 0 ) = LDDRSRC( 4, 1 ) = LDDRSRC( 5, 2 ) = LDDRSRC( 6, 3 ) = LDDRSRC( 7, 4 ) = LA_F2( t1, t2, t3 );
    LDDRSRC( 4, 0 ) = LDDRSRC( 5, 1 ) = LDDRSRC( 6, 2 ) = LDDRSRC( 7, 3 ) = LA_F2( t2, t3, t4 );
    LDDRSRC( 5, 0 ) = LDDRSRC( 6, 1 ) = LDDRSRC( 7, 2 ) = LA_F2( t3, t4, t5 );
    LDDRSRC( 6, 0 ) = LDDRSRC( 7, 1 ) = LA_F2( t4, t5, t6 );
    LDDRSRC( 7, 0 ) = LA_F2( t5, t6, t7 );
#undef LDDRSRC
}

__device__ __forceinline__ void la_pred_8x8_vr( int *src, const int edge[36] )
{
    int l0 = edge[14], l1 = edge[13], l2 = edge[12], l3 = edge[11];
    int l4 = edge[10], l5 = edge[9], l6 = edge[8], l7 = edge[7];
    int lt = edge[15];
    int t0 = edge[16], t1 = edge[17], t2 = edge[18], t3 = edge[19];
    int t4 = edge[20], t5 = edge[21], t6 = edge[22], t7 = edge[23];
#define LVRSRC( x, y ) src[(x) + (y) * 32]
    LVRSRC( 0, 6 ) = LA_F2( l5, l4, l3 );
    LVRSRC( 0, 7 ) = LA_F2( l6, l5, l4 );
    LVRSRC( 0, 4 ) = LVRSRC( 1, 6 ) = LA_F2( l3, l2, l1 );
    LVRSRC( 0, 5 ) = LVRSRC( 1, 7 ) = LA_F2( l4, l3, l2 );
    LVRSRC( 0, 2 ) = LVRSRC( 1, 4 ) = LVRSRC( 2, 6 ) = LA_F2( l1, l0, lt );
    LVRSRC( 0, 3 ) = LVRSRC( 1, 5 ) = LVRSRC( 2, 7 ) = LA_F2( l2, l1, l0 );
    LVRSRC( 0, 1 ) = LVRSRC( 1, 3 ) = LVRSRC( 2, 5 ) = LVRSRC( 3, 7 ) = LA_F2( l0, lt, t0 );
    LVRSRC( 0, 0 ) = LVRSRC( 1, 2 ) = LVRSRC( 2, 4 ) = LVRSRC( 3, 6 ) = LA_F1( lt, t0 );
    LVRSRC( 1, 1 ) = LVRSRC( 2, 3 ) = LVRSRC( 3, 5 ) = LVRSRC( 4, 7 ) = LA_F2( lt, t0, t1 );
    LVRSRC( 1, 0 ) = LVRSRC( 2, 2 ) = LVRSRC( 3, 4 ) = LVRSRC( 4, 6 ) = LA_F1( t0, t1 );
    LVRSRC( 2, 1 ) = LVRSRC( 3, 3 ) = LVRSRC( 4, 5 ) = LVRSRC( 5, 7 ) = LA_F2( t0, t1, t2 );
    LVRSRC( 2, 0 ) = LVRSRC( 3, 2 ) = LVRSRC( 4, 4 ) = LVRSRC( 5, 6 ) = LA_F1( t1, t2 );
    LVRSRC( 3, 1 ) = LVRSRC( 4, 3 ) = LVRSRC( 5, 5 ) = LVRSRC( 6, 7 ) = LA_F2( t1, t2, t3 );
    LVRSRC( 3, 0 ) = LVRSRC( 4, 2 ) = LVRSRC( 5, 4 ) = LVRSRC( 6, 6 ) = LA_F1( t2, t3 );
    LVRSRC( 4, 1 ) = LVRSRC( 5, 3 ) = LVRSRC( 6, 5 ) = LVRSRC( 7, 7 ) = LA_F2( t2, t3, t4 );
    LVRSRC( 4, 0 ) = LVRSRC( 5, 2 ) = LVRSRC( 6, 4 ) = LVRSRC( 7, 6 ) = LA_F1( t3, t4 );
    LVRSRC( 5, 1 ) = LVRSRC( 6, 3 ) = LVRSRC( 7, 5 ) = LA_F2( t3, t4, t5 );
    LVRSRC( 5, 0 ) = LVRSRC( 6, 2 ) = LVRSRC( 7, 4 ) = LA_F1( t4, t5 );
    LVRSRC( 6, 1 ) = LVRSRC( 7, 3 ) = LA_F2( t4, t5, t6 );
    LVRSRC( 6, 0 ) = LVRSRC( 7, 2 ) = LA_F1( t5, t6 );
    LVRSRC( 7, 1 ) = LA_F2( t5, t6, t7 );
    LVRSRC( 7, 0 ) = LA_F1( t6, t7 );
#undef LVRSRC
}

__device__ __forceinline__ void la_pred_8x8_vl( int *src, const int edge[36] )
{
    int t0 = edge[16], t1 = edge[17], t2 = edge[18], t3 = edge[19];
    int t4 = edge[20], t5 = edge[21], t6 = edge[22], t7 = edge[23];
    int t8 = edge[24], t9 = edge[25], t10 = edge[26], t11 = edge[27];
    int t12 = edge[28];
#define LVLSRC( x, y ) src[(x) + (y) * 32]
    LVLSRC( 0, 0 ) = LA_F1( t0, t1 );
    LVLSRC( 0, 1 ) = LA_F2( t0, t1, t2 );
    LVLSRC( 0, 2 ) = LVLSRC( 1, 0 ) = LA_F1( t1, t2 );
    LVLSRC( 0, 3 ) = LVLSRC( 1, 1 ) = LA_F2( t1, t2, t3 );
    LVLSRC( 0, 4 ) = LVLSRC( 1, 2 ) = LVLSRC( 2, 0 ) = LA_F1( t2, t3 );
    LVLSRC( 0, 5 ) = LVLSRC( 1, 3 ) = LVLSRC( 2, 1 ) = LA_F2( t2, t3, t4 );
    LVLSRC( 0, 6 ) = LVLSRC( 1, 4 ) = LVLSRC( 2, 2 ) = LVLSRC( 3, 0 ) = LA_F1( t3, t4 );
    LVLSRC( 0, 7 ) = LVLSRC( 1, 5 ) = LVLSRC( 2, 3 ) = LVLSRC( 3, 1 ) = LA_F2( t3, t4, t5 );
    LVLSRC( 1, 6 ) = LVLSRC( 2, 4 ) = LVLSRC( 3, 2 ) = LVLSRC( 4, 0 ) = LA_F1( t4, t5 );
    LVLSRC( 1, 7 ) = LVLSRC( 2, 5 ) = LVLSRC( 3, 3 ) = LVLSRC( 4, 1 ) = LA_F2( t4, t5, t6 );
    LVLSRC( 2, 6 ) = LVLSRC( 3, 4 ) = LVLSRC( 4, 2 ) = LVLSRC( 5, 0 ) = LA_F1( t5, t6 );
    LVLSRC( 2, 7 ) = LVLSRC( 3, 5 ) = LVLSRC( 4, 3 ) = LVLSRC( 5, 1 ) = LA_F2( t5, t6, t7 );
    LVLSRC( 3, 6 ) = LVLSRC( 4, 4 ) = LVLSRC( 5, 2 ) = LVLSRC( 6, 0 ) = LA_F1( t6, t7 );
    LVLSRC( 3, 7 ) = LVLSRC( 4, 5 ) = LVLSRC( 5, 3 ) = LVLSRC( 6, 1 ) = LA_F2( t6, t7, t8 );
    LVLSRC( 4, 6 ) = LVLSRC( 5, 4 ) = LVLSRC( 6, 2 ) = LVLSRC( 7, 0 ) = LA_F1( t7, t8 );
    LVLSRC( 4, 7 ) = LVLSRC( 5, 5 ) = LVLSRC( 6, 3 ) = LVLSRC( 7, 1 ) = LA_F2( t7, t8, t9 );
    LVLSRC( 5, 6 ) = LVLSRC( 6, 4 ) = LVLSRC( 7, 2 ) = LA_F1( t8, t9 );
    LVLSRC( 5, 7 ) = LVLSRC( 6, 5 ) = LVLSRC( 7, 3 ) = LA_F2( t8, t9, t10 );
    LVLSRC( 6, 6 ) = LVLSRC( 7, 4 ) = LA_F1( t9, t10 );
    LVLSRC( 6, 7 ) = LVLSRC( 7, 5 ) = LA_F2( t9, t10, t11 );
    LVLSRC( 7, 6 ) = LA_F1( t10, t11 );
    LVLSRC( 7, 7 ) = LA_F2( t10, t11, t12 );
#undef LVLSRC
}

__device__ __forceinline__ void la_pred_8x8_hd( int *src, const int edge[36] )
{
    int l0 = edge[14], l1 = edge[13], l2 = edge[12], l3 = edge[11];
    int l4 = edge[10], l5 = edge[9], l6 = edge[8], l7 = edge[7];
    int lt = edge[15];
    int t0 = edge[16], t1 = edge[17], t2 = edge[18], t3 = edge[19];
    int t4 = edge[20], t5 = edge[21], t6 = edge[22], t7 = edge[23];
    /* Each source row is stored as four adjacent pixels; the C reference
     * builds them from packed pairs, expanded here one pixel at a time. */
#define LHD2( x, y, a, b ) do { src[(x) + (y) * 32] = (a); src[(x) + 1 + (y) * 32] = (b); } while( 0 )
    LHD2( 0, 7, LA_F1( l6, l7 ), LA_F2( l5, l6, l7 ) );
    LHD2( 2, 7, LA_F1( l5, l6 ), LA_F2( l4, l5, l6 ) );
    LHD2( 4, 7, LA_F1( l4, l5 ), LA_F2( l3, l4, l5 ) );
    LHD2( 6, 7, LA_F1( l3, l4 ), LA_F2( l2, l3, l4 ) );
    LHD2( 0, 6, LA_F1( l5, l6 ), LA_F2( l4, l5, l6 ) );
    LHD2( 2, 6, LA_F1( l4, l5 ), LA_F2( l3, l4, l5 ) );
    LHD2( 4, 6, LA_F1( l3, l4 ), LA_F2( l2, l3, l4 ) );
    LHD2( 6, 6, LA_F1( l2, l3 ), LA_F2( l1, l2, l3 ) );
    LHD2( 0, 5, LA_F1( l4, l5 ), LA_F2( l3, l4, l5 ) );
    LHD2( 2, 5, LA_F1( l3, l4 ), LA_F2( l2, l3, l4 ) );
    LHD2( 4, 5, LA_F1( l2, l3 ), LA_F2( l1, l2, l3 ) );
    LHD2( 6, 5, LA_F1( l1, l2 ), LA_F2( l0, l1, l2 ) );
    LHD2( 0, 4, LA_F1( l3, l4 ), LA_F2( l2, l3, l4 ) );
    LHD2( 2, 4, LA_F1( l2, l3 ), LA_F2( l1, l2, l3 ) );
    LHD2( 4, 4, LA_F1( l1, l2 ), LA_F2( l0, l1, l2 ) );
    LHD2( 6, 4, LA_F1( l0, l1 ), LA_F2( lt, l0, l1 ) );
    LHD2( 0, 3, LA_F1( l2, l3 ), LA_F2( l1, l2, l3 ) );
    LHD2( 2, 3, LA_F1( l1, l2 ), LA_F2( l0, l1, l2 ) );
    LHD2( 4, 3, LA_F1( l0, l1 ), LA_F2( lt, l0, l1 ) );
    LHD2( 6, 3, LA_F1( lt, l0 ), LA_F2( l0, lt, t0 ) );
    LHD2( 0, 2, LA_F1( l1, l2 ), LA_F2( l0, l1, l2 ) );
    LHD2( 2, 2, LA_F1( l0, l1 ), LA_F2( lt, l0, l1 ) );
    LHD2( 4, 2, LA_F1( lt, l0 ), LA_F2( l0, lt, t0 ) );
    LHD2( 6, 2, LA_F2( t1, t0, lt ), LA_F2( t2, t1, t0 ) );
    LHD2( 0, 1, LA_F1( l0, l1 ), LA_F2( lt, l0, l1 ) );
    LHD2( 2, 1, LA_F1( lt, l0 ), LA_F2( l0, lt, t0 ) );
    LHD2( 4, 1, LA_F2( t1, t0, lt ), LA_F2( t2, t1, t0 ) );
    LHD2( 6, 1, LA_F2( t3, t2, t1 ), LA_F2( t4, t3, t2 ) );
    LHD2( 0, 0, LA_F1( lt, l0 ), LA_F2( l0, lt, t0 ) );
    LHD2( 2, 0, LA_F2( t1, t0, lt ), LA_F2( t2, t1, t0 ) );
    LHD2( 4, 0, LA_F2( t3, t2, t1 ), LA_F2( t4, t3, t2 ) );
    LHD2( 6, 0, LA_F2( t5, t4, t3 ), LA_F2( t6, t5, t4 ) );
#undef LHD2
}

__device__ __forceinline__ void la_pred_8x8_hu( int *src, const int edge[36] )
{
    int l0 = edge[14], l1 = edge[13], l2 = edge[12], l3 = edge[11];
    int l4 = edge[10], l5 = edge[9], l6 = edge[8], l7 = edge[7];
#define LHU2( x, y, a, b ) do { src[(x) + (y) * 32] = (a); src[(x) + 1 + (y) * 32] = (b); } while( 0 )
    LHU2( 0, 0, LA_F1( l0, l1 ), LA_F2( l0, l1, l2 ) );
    LHU2( 2, 0, LA_F1( l1, l2 ), LA_F2( l1, l2, l3 ) );
    LHU2( 4, 0, LA_F1( l2, l3 ), LA_F2( l2, l3, l4 ) );
    LHU2( 6, 0, LA_F1( l3, l4 ), LA_F2( l3, l4, l5 ) );
    LHU2( 0, 1, LA_F1( l1, l2 ), LA_F2( l1, l2, l3 ) );
    LHU2( 2, 1, LA_F1( l2, l3 ), LA_F2( l2, l3, l4 ) );
    LHU2( 4, 1, LA_F1( l3, l4 ), LA_F2( l3, l4, l5 ) );
    LHU2( 6, 1, LA_F1( l4, l5 ), LA_F2( l4, l5, l6 ) );
    LHU2( 0, 2, LA_F1( l2, l3 ), LA_F2( l2, l3, l4 ) );
    LHU2( 2, 2, LA_F1( l3, l4 ), LA_F2( l3, l4, l5 ) );
    LHU2( 4, 2, LA_F1( l4, l5 ), LA_F2( l4, l5, l6 ) );
    LHU2( 6, 2, LA_F1( l5, l6 ), LA_F2( l5, l6, l7 ) );
    LHU2( 0, 3, LA_F1( l3, l4 ), LA_F2( l3, l4, l5 ) );
    LHU2( 2, 3, LA_F1( l4, l5 ), LA_F2( l4, l5, l6 ) );
    LHU2( 4, 3, LA_F1( l5, l6 ), LA_F2( l5, l6, l7 ) );
    LHU2( 6, 3, LA_F1( l6, l7 ), LA_F2( l6, l7, l7 ) );
    LHU2( 0, 4, LA_F1( l4, l5 ), LA_F2( l4, l5, l6 ) );
    LHU2( 2, 4, LA_F1( l5, l6 ), LA_F2( l5, l6, l7 ) );
    LHU2( 4, 4, LA_F1( l6, l7 ), LA_F2( l6, l7, l7 ) );
    LHU2( 6, 4, l7, l7 );
    LHU2( 0, 5, LA_F1( l5, l6 ), LA_F2( l5, l6, l7 ) );
    LHU2( 2, 5, LA_F1( l6, l7 ), LA_F2( l6, l7, l7 ) );
    LHU2( 4, 5, l7, l7 );
    LHU2( 6, 5, l7, l7 );
    LHU2( 0, 6, LA_F1( l6, l7 ), LA_F2( l6, l7, l7 ) );
    LHU2( 2, 6, l7, l7 );
    LHU2( 4, 6, l7, l7 );
    LHU2( 6, 6, l7, l7 );
    LHU2( 0, 7, l7, l7 );
    LHU2( 2, 7, l7, l7 );
    LHU2( 4, 7, l7, l7 );
    LHU2( 6, 7, l7, l7 );
#undef LHU2
}

/* The full intra pre-pass for one macroblock, replicating slicetype_mb_cost's
 * lowres_intra_mb section: 8x8c DC/H/V, then (subme > 1) planar and the six
 * 8x8 directional modes with the 1-pixel-tap edge filter, all scored with
 * SATD (or SAD for subme <= 1).  Returns the raw minimum (before the depth
 * shift and the penalty the CPU adds). */
template<class P>
__device__ __forceinline__ int la_intra_mb( const P *sp, int stride, int b_satd, int maxpx )
{
    /* Mirror the CPU's local buffers: fenc block on an 8-int row and the
     * prediction scratch on FDEC_STRIDE (32) rows, edges at row -1 / col -1
     * (KPIXI maps exactly onto the CPU's pix1[8 + x + (y+1)*32] layout). */
    int fenc[64];
    int kpix[9 * 32];
#define KPIXI( x, y ) kpix[8 + (x) + ((y) + 1) * 32]
#pragma unroll
    for( int x = 0; x < 16; x++ )
        KPIXI( x, -1 ) = (int)sp[x - stride];
#pragma unroll
    for( int y = -1; y < 8; y++ )
#pragma unroll
        for( int c = 0; c < (int)(sizeof(P) == 1 ? 4 : 2); c++ )
            KPIXI( -1 - c, y ) = (int)sp[y * stride - 1 - c];
#pragma unroll
    for( int j = 0; j < 8; j++ )
#pragma unroll
        for( int i = 0; i < 8; i++ )
            fenc[j * 8 + i] = (int)sp[j * stride + i];

    int *blk = kpix + 8 + 32;
    int best;
    la_pred_8x8c_dc( blk );
    best = b_satd ? la_satd_8x8_p<P>( fenc, blk ) : la_sad_8x8( fenc, blk );
    la_pred_8x8c_h( blk );
    {
        int c = b_satd ? la_satd_8x8_p<P>( fenc, blk ) : la_sad_8x8( fenc, blk );
        if( c < best ) best = c;
    }
    la_pred_8x8c_v( blk );
    {
        int c = b_satd ? la_satd_8x8_p<P>( fenc, blk ) : la_sad_8x8( fenc, blk );
        if( c < best ) best = c;
    }

    if( b_satd )
    {
        la_pred_8x8c_p( blk, maxpx );
        {
            int c = la_satd_8x8_p<P>( fenc, blk );
            if( c < best ) best = c;
        }
        int edge[36];
        la_pred_8x8_filter_all( blk, edge );
#define LA_TRY( fn ) { fn( blk, edge ); int c = la_satd_8x8_p<P>( fenc, blk ); if( c < best ) best = c; }
        LA_TRY( la_pred_8x8_ddl )
        LA_TRY( la_pred_8x8_ddr )
        LA_TRY( la_pred_8x8_vr )
        LA_TRY( la_pred_8x8_vl )
        LA_TRY( la_pred_8x8_hd )
        LA_TRY( la_pred_8x8_hu )
#undef LA_TRY
    }
    return best;
}

template<class P>
__device__ __forceinline__ void la_intra_impl( const P *__restrict__ buf, int plane_off,
                                               int stride, int mbw, int mbh,
                                               int b_satd, int maxpx, int *__restrict__ out )
{
    int mb = blockIdx.x * blockDim.x + threadIdx.x;
    if( mb >= mbw * mbh )
        return;
    int mx = mb % mbw, my = mb / mbw;
    const P *sp = buf + plane_off + (size_t)my * 8 * stride + mx * 8;
    out[mb] = la_intra_mb<P>( sp, stride, b_satd, maxpx );
}

extern "C" __global__ void la_intra( const uint8_t *__restrict__ buf, int plane_off,
                                     int stride, int mbw, int mbh,
                                     int b_satd, int maxpx, int *__restrict__ out )
{
    la_intra_impl( buf, plane_off, stride, mbw, mbh, b_satd, maxpx, out );
}

extern "C" __global__ void la_intra_10( const uint16_t *__restrict__ buf, int plane_off,
                                        int stride, int mbw, int mbh,
                                        int b_satd, int maxpx, int *__restrict__ out )
{
    la_intra_impl( buf, plane_off, stride, mbw, mbh, b_satd, maxpx, out );
}

/* Exhaustive 8x8 SAD square for one (fenc, ref) pair: out[(mb*S + (dy+R))*S +
 * (dx+R)] holds SAD(fenc's MB, ref displaced by (dx,dy)) for all |dx|,|dy| <=
 * R, exactly the values h->pixf.fpelcmp[PIXEL_8x8] would return.  With
 * b_satd set, the same Hadamard SATD that mbcmp_init() puts into fpelcmp
 * (me_method TESA) is computed instead of the SAD. */
template<class P>
__device__ __forceinline__ void la_table_impl( const P *__restrict__ fbuf, int foff,
                                               const P *__restrict__ rbuf, int roff,
                                               int fstride, int rstride,
                                               int mbw, int mbh, int R, int b_satd,
                                               uint16_t *__restrict__ out )
{
    int S = 2 * R + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mbw * mbh * S;
    if( idx >= total )
        return;
    int mb = idx / S;
    int dy = idx % S - R;
    int mx = mb % mbw, my = mb / mbw;
    const P *f = fbuf + foff + (size_t)my * 8 * fstride + mx * 8;
    const P *r = rbuf + roff + (intptr_t)(my * 8 + dy) * rstride + mx * 8 - R;
    uint16_t *row = out + (size_t)mb * S * S + (size_t)(dy + R) * S;
    for( int dx = 0; dx < S; dx++ )
    {
        int val;
        const P *rr = r + dx;
        if( b_satd )
        {
            if( sizeof(P) == 1 )
                val = la_satd_8x8_raw<uint32_t, uint16_t, 16, P>( f, fstride, rr, rstride );
            else
                val = la_satd_8x8_raw<uint64_t, uint32_t, 32, P>( f, fstride, rr, rstride );
        }
        else
        {
            int sad = 0;
#pragma unroll
            for( int j = 0; j < 8; j++ )
#pragma unroll
                for( int i = 0; i < 8; i++ )
                {
                    int d = (int)f[j * fstride + i] - (int)rr[j * rstride + i];
                    sad += d < 0 ? -d : d;
                }
            val = sad;
        }
        row[dx] = (uint16_t)val;
    }
}

extern "C" __global__ void la_table( const uint8_t *__restrict__ fbuf, int foff,
                                     const uint8_t *__restrict__ rbuf, int roff,
                                     int fstride, int rstride,
                                     int mbw, int mbh, int R, int b_satd, uint16_t *__restrict__ out )
{
    la_table_impl( fbuf, foff, rbuf, roff, fstride, rstride, mbw, mbh, R, b_satd, out );
}

extern "C" __global__ void la_table_10( const uint16_t *__restrict__ fbuf, int foff,
                                        const uint16_t *__restrict__ rbuf, int roff,
                                        int fstride, int rstride,
                                        int mbw, int mbh, int R, int b_satd, uint16_t *__restrict__ out )
{
    la_table_impl( fbuf, foff, rbuf, roff, fstride, rstride, mbw, mbh, R, b_satd, out );
}

