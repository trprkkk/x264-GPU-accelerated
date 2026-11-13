// CUDA implementation of the hierarchical candidate generator.
//
// The GPU does a coarse exhaustive search on a quarter-resolution luma copy
// and then refines the best four integer-pel positions at full resolution.
// It is deliberately an *open-loop* pre-analysis: it uses the reference
// planes that x264 already reconstructed, but never writes any coding
// decision.  me.c evaluates every returned vector with the real x264 cost
// function, so rate control, RDO, entropy coding and the bitstream syntax stay
// exactly in the original CPU code.

#include "gpu.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <mutex>

#define GK X264_GPU_CANDS

static double gpu_now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>( steady_clock::now().time_since_epoch() ).count();
}

typedef struct
{
    int16_t  mv[2];
    uint32_t sad;
} Cand;

struct x264_gpu_ctx_t;

struct x264_gpu_frame_t
{
    int mbw, mbh;
    int nref[2];       /* allocated candidate slots per list */
    int nfilled[2];    /* references actually analysed and published */
    int pending[2];    /* analysed but not yet published (async prefetch) */
    int ncand;
    /* Reference identity of every analysed slot.  The analysis may run for a
     * predicted reference set before the final lists are known, so me.c
     * matches its (list, ref) against these identities. */
    const void *slot_ptr[2][X264_GPU_MAX_REFS];
    int64_t     slot_id[2][X264_GPU_MAX_REFS];
    Cand *cand[2][X264_GPU_MAX_REFS];
    struct x264_gpu_ctx_t *ctx;    /* owner, for sync-on-free */
};

struct x264_gpu_ctx_t
{
    int w, h, lw, lh, mbw, mbh, max_refs, device;
    uint8_t *d_cur;
    uint8_t *d_cur_low;
    Cand    *d_coarse;
    Cand    *d_out;
    cudaStream_t stream;
    std::mutex lock;
    int disabled;
    x264_gpu_frame_t *pending_frame;   /* analysis in flight (async prefetch) */
    uint8_t *h_ref;            /* pinned staging for reference uploads */
    Cand    *h_out;            /* pinned staging for candidate D2H copies */
    /* Optional per-stage timing (enabled with X264_GPU_TIMING=1). */
    int timing;
    long long nframes;
    double t_upload_cur, t_upload_ref, t_down, t_coarse, t_refine, t_d2h, t_sync;
    double t_finish;          /* wait for a prefetched analysis */
    long long n_finish;
    double t_call;            /* wall time inside x264_gpu_analyze*() */
    long long n_call;
    int prof;                 /* X264_GPU_PROF=1: print call-time stats */
    /* Reference plane cache.  Keyed by the source frame pointer and its
     * presentation number, because x264 recycles frame objects. */
    struct CacheEntry
    {
        const void *ptr;
        int64_t id;
        uint8_t *full;
        uint8_t *low;
    };
    CacheEntry cache[X264_GPU_CACHE];
    int cache_next;
};

static char gpu_name[256] = "CUDA device";

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

__global__ void downsample4( const uint8_t *__restrict__ src, int sw, int sh,
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
            int px = min( sx + i, sw - 1 );
            int py = min( sy + j, sh - 1 );
            sum += src[py * sw + px];
        }
    dst[y * dw + x] = (uint8_t)((sum + 8) >> 4);
}

__global__ void coarse_search( const uint8_t *__restrict__ cur, int cw, int ch,
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
            int x = max( 0, min( bx + dx, cw - 4 ) );
            int y = max( 0, min( by + dy, ch - 4 ) );
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
            int x = max( 0, min( fx + dx, cw - 4 ) );
            int y = max( 0, min( fy + dy, ch - 4 ) );
            uint32_t sad = sad4x4( cur + by * cw + bx, cw, ref + y * rw + x, rw );
            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
        }
    for( int i = 0; i < GK; i++ )
        out[mb * GK + i] = best[i];
}

__global__ void refine_search( const uint8_t *__restrict__ cur, int cw, int ch,
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
            int x = max( 0, min( cx + dx, cw - 16 ) );
            int y = max( 0, min( cy + dy, ch - 16 ) );
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

extern "C" void *x264_gpu_create( int width, int height, int max_refs, int device )
{
    /* The kernels work on whole macroblocks; the caller passes the frame size
     * rounded up to the macroblock grid (x264 pads its planes to that size). */
    if( width <= 0 || height <= 0 || (width & 15) || (height & 15) )
        return NULL;

    int ndev = 0;
    if( cudaGetDeviceCount( &ndev ) != cudaSuccess || ndev <= 0 )
        return NULL;
    if( device < 0 || device >= ndev )
        device = 0;
    if( cudaSetDevice( device ) != cudaSuccess )
        return NULL;

    cudaDeviceProp prop;
    if( cudaGetDeviceProperties( &prop, device ) == cudaSuccess )
        snprintf( gpu_name, sizeof(gpu_name), "%s (sm_%d%d)", prop.name,
                  prop.major, prop.minor );

    x264_gpu_ctx_t *c = new (std::nothrow) x264_gpu_ctx_t();
    if( !c )
        return NULL;
    c->w = width;
    c->h = height;
    c->lw = (width + 3) / 4;
    c->lh = (height + 3) / 4;
    c->mbw = (width + 15) / 16;
    c->mbh = (height + 15) / 16;
    c->max_refs = max_refs < 1 ? 1 : (max_refs > X264_GPU_MAX_REFS ? X264_GPU_MAX_REFS : max_refs);
    c->device = device;
    c->disabled = 0;
    c->timing = getenv( "X264_GPU_TIMING" ) != NULL;
    c->prof = getenv( "X264_GPU_PROF" ) != NULL;
    if( c->timing )
        printf( "x264 CUDA ME timing: %dx%d, %d refs, device %d\n", width, height, c->max_refs, device );

    size_t fsz = (size_t)width * height;
    size_t lsz = (size_t)c->lw * c->lh;
    size_t csz = (size_t)c->mbw * c->mbh * GK * sizeof(Cand);
    int fail = 0;
    if( cudaStreamCreate( &c->stream ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_cur, fsz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_cur_low, lsz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_coarse, csz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_out, csz ) != cudaSuccess ||
        cudaMallocHost( (void**)&c->h_ref, fsz ) != cudaSuccess ||
        cudaMallocHost( (void**)&c->h_out, 2 * X264_GPU_MAX_REFS * csz ) != cudaSuccess )
        fail = 1;
    for( int i = 0; i < X264_GPU_CACHE && !fail; i++ )
        if( cudaMalloc( (void**)&c->cache[i].full, fsz ) != cudaSuccess ||
            cudaMalloc( (void**)&c->cache[i].low, lsz ) != cudaSuccess )
            fail = 1;
    if( fail )
    {
        if( c->stream ) cudaStreamDestroy( c->stream );
        if( c->d_cur ) cudaFree( c->d_cur );
        if( c->d_cur_low ) cudaFree( c->d_cur_low );
        if( c->d_coarse ) cudaFree( c->d_coarse );
        if( c->d_out ) cudaFree( c->d_out );
        if( c->h_ref ) cudaFreeHost( c->h_ref );
        if( c->h_out ) cudaFreeHost( c->h_out );
        for( int i = 0; i < X264_GPU_CACHE; i++ )
        {
            if( c->cache[i].full ) cudaFree( c->cache[i].full );
            if( c->cache[i].low ) cudaFree( c->cache[i].low );
        }
        delete c;
        return NULL;
    }
    return c;
}

extern "C" void x264_gpu_destroy( void *ctx )
{
    x264_gpu_ctx_t *c = (x264_gpu_ctx_t*)ctx;
    if( !c )
        return;
    if( c->timing && c->nframes )
    {
        double n = (double)c->nframes;
        printf( "x264 CUDA ME timing (%lld frames analysed, ms/frame): "
                "cur-upload %.3f  ref-upload %.3f  downsample %.3f  coarse %.3f  "
                "refine %.3f  d2h %.3f  sync %.3f  total %.3f\n",
                c->nframes,
                c->t_upload_cur / n, c->t_upload_ref / n, c->t_down / n,
                c->t_coarse / n, c->t_refine / n, c->t_d2h / n,
                c->t_sync / n,
                (c->t_upload_cur + c->t_upload_ref + c->t_down + c->t_coarse
                 + c->t_refine + c->t_d2h + c->t_sync) / n );
        if( c->n_finish )
            printf( "x264 CUDA ME timing: prefetch finish wait %.3f ms over %lld finishes\n",
                    c->t_finish / n, c->n_finish );
    }
    if( c->prof && c->n_call )
        printf( "x264 CUDA ME profile: x264_gpu_analyze*() call %.3f ms on average over %lld calls\n",
                c->t_call / c->n_call, c->n_call );
    if( c->stream ) cudaStreamSynchronize( c->stream );
    if( c->stream ) cudaStreamDestroy( c->stream );
    if( c->d_cur ) cudaFree( c->d_cur );
    if( c->d_cur_low ) cudaFree( c->d_cur_low );
    if( c->d_coarse ) cudaFree( c->d_coarse );
    if( c->d_out ) cudaFree( c->d_out );
    if( c->h_ref ) cudaFreeHost( c->h_ref );
    if( c->h_out ) cudaFreeHost( c->h_out );
    for( int i = 0; i < X264_GPU_CACHE; i++ )
    {
        if( c->cache[i].full ) cudaFree( c->cache[i].full );
        if( c->cache[i].low ) cudaFree( c->cache[i].low );
    }
    delete c;
}

extern "C" const char *x264_gpu_name( void )
{
    return gpu_name;
}

extern "C" x264_gpu_frame_t *x264_gpu_frame_alloc( int mbw, int mbh, int nref0, int nref1 )
{
    x264_gpu_frame_t *f = (x264_gpu_frame_t*)calloc( 1, sizeof(*f) );
    if( !f )
        return NULL;
    f->mbw = mbw;
    f->mbh = mbh;
    f->ncand = GK;
    f->nref[0] = nref0 < 0 ? 0 : (nref0 > X264_GPU_MAX_REFS ? X264_GPU_MAX_REFS : nref0);
    f->nref[1] = nref1 < 0 ? 0 : (nref1 > X264_GPU_MAX_REFS ? X264_GPU_MAX_REFS : nref1);
    f->nfilled[0] = 0;
    f->nfilled[1] = 0;
    size_t n = (size_t)mbw * mbh * GK * sizeof(Cand);
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
        {
            f->cand[l][r] = (Cand*)malloc( n );
            if( !f->cand[l][r] )
            {
                x264_gpu_frame_free( f );
                return NULL;
            }
            memset( f->cand[l][r], 0, n );
        }
    return f;
}

extern "C" void x264_gpu_frame_free( x264_gpu_frame_t *f )
{
    if( !f )
        return;
    /* Never free candidate memory that an in-flight D2H copy may still be
     * writing to. */
    if( f->ctx && f->ctx->pending_frame == f )
        x264_gpu_finish( f->ctx );
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
            free( f->cand[l][r] );
    free( f );
}

static int upload_plane( cudaStream_t st, uint8_t *dst, const uint8_t *src,
                         int stride, int width, int height )
{
    if( !src || stride < width )
        return -1;
    if( stride == width )
        return cudaMemcpyAsync( dst, src, (size_t)width * height,
                                cudaMemcpyHostToDevice, st ) == cudaSuccess ? 0 : -1;
    return cudaMemcpy2DAsync( dst, (size_t)width, src, (size_t)stride,
                              (size_t)width, (size_t)height,
                              cudaMemcpyHostToDevice, st ) == cudaSuccess ? 0 : -1;
}

/* Upload through a pinned staging buffer: pageable sources would make the
 * asynchronous copy synchronise the stream, which serialises the whole
 * analysis onto the calling thread. */
static int upload_plane_pinned( cudaStream_t st, uint8_t *dst, const uint8_t *src,
                                int stride, int width, int height, uint8_t *stage )
{
    if( !src || !stage || stride < width )
        return -1;
    if( stride == width )
        memcpy( stage, src, (size_t)width * height );
    else
        for( int y = 0; y < height; y++ )
            memcpy( stage + (size_t)y * width, src + (size_t)y * stride, width );
    return cudaMemcpyAsync( dst, stage, (size_t)width * height,
                            cudaMemcpyHostToDevice, st ) == cudaSuccess ? 0 : -1;
}

/* Copy the staged candidates into the frame after the stream has been
 * synchronised. */
static void publish_candidates( x264_gpu_ctx_t *c, x264_gpu_frame_t *frame, int list, int nref )
{
    size_t n = (size_t)c->mbw * c->mbh * GK * sizeof(Cand);
    for( int r = 0; r < nref; r++ )
        memcpy( frame->cand[list][r],
                c->h_out + (size_t)(list * X264_GPU_MAX_REFS + r) * (c->mbw * c->mbh * GK),
                n );
}

static int gpu_analyze_common( void *ctx, x264_gpu_frame_t *frame,
                               const uint8_t *cur, int cur_stride,
                               uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                               uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                               int range, int radius, int b_sync )
{
    x264_gpu_ctx_t *c = (x264_gpu_ctx_t*)ctx;
    if( !c || !frame || c->disabled )
        return -1;
    if( frame->mbw != c->mbw || frame->mbh != c->mbh )
        return -1;
    if( range < 4 ) range = 4;
    if( range > 128 ) range = 128;
    if( radius < 0 ) radius = 0;
    if( radius > 8 ) radius = 8;

    std::lock_guard<std::mutex> guard( c->lock );

    /* Candidates are only valid for the references analysed in this call;
     * anything else must fall back to the CPU search.  In async mode nothing
     * is published until x264_gpu_finish() has synchronised the stream. */
    frame->nfilled[0] = 0;
    frame->nfilled[1] = 0;
    frame->pending[0] = 0;
    frame->pending[1] = 0;
    frame->ctx = c;
    if( !b_sync )
        c->pending_frame = frame;
    double t_call0 = c->prof ? gpu_now_ms() : 0;

    int lw = c->lw, lh = c->lh;
    dim3 blk( 16, 16 );
    dim3 grd( (lw + 15) / 16, (lh + 15) / 16 );
    int threads = 256;
    int mbcount = c->mbw * c->mbh;
    int total = mbcount * GK;

    double t_last = c->timing ? gpu_now_ms() : 0;
#define TIM_MARK(member) \
    if( c->timing ) { cudaStreamSynchronize( c->stream ); \
                      c->member += gpu_now_ms() - t_last; t_last = gpu_now_ms(); }

    if( upload_plane( c->stream, c->d_cur, cur, cur_stride, c->w, c->h ) < 0 )
        goto fail;
    TIM_MARK( t_upload_cur );
    downsample4<<<grd, blk, 0, c->stream>>>( c->d_cur, c->w, c->h,
                                             c->d_cur_low, lw, lh );
    TIM_MARK( t_down );

    for( int list = 0; list < 2; list++ )
    {
        uint8_t *const *refs = list ? refs1 : refs0;
        const int64_t *ids = list ? ref_ids1 : ref_ids0;
        const int *strides = list ? stride1 : stride0;
        int nref = list ? nref1 : nref0;
        if( nref > frame->nref[list] )
            nref = frame->nref[list];
        for( int r = 0; r < nref; r++ )
        {
            /* Remember which reference this slot belongs to. */
            frame->slot_ptr[list][r] = refs[r];
            frame->slot_id[list][r] = ids[r];
            /* Reuse the device copy of a reference plane while its content
             * stays the same.  References are typically re-analysed every
             * frame, so this removes most of the PCIe traffic. */
            uint8_t *d_ref = NULL, *d_ref_low = NULL;
            for( int i = 0; i < X264_GPU_CACHE; i++ )
                if( c->cache[i].ptr == refs[r] && c->cache[i].id == ids[r] )
                {
                    d_ref = c->cache[i].full;
                    d_ref_low = c->cache[i].low;
                    break;
                }
            if( !d_ref )
            {
                int slot = c->cache_next++ % X264_GPU_CACHE;
                d_ref = c->cache[slot].full;
                d_ref_low = c->cache[slot].low;
                if( upload_plane_pinned( c->stream, d_ref, refs[r], strides[r], c->w, c->h, c->h_ref ) < 0 )
                    goto fail;
                TIM_MARK( t_upload_ref );
                downsample4<<<grd, blk, 0, c->stream>>>( d_ref, c->w, c->h,
                                                         d_ref_low, lw, lh );
                TIM_MARK( t_down );
                c->cache[slot].ptr = refs[r];
                c->cache[slot].id = ids[r];
            }
            coarse_search<<<(mbcount + threads - 1) / threads, threads, 0, c->stream>>>(
                c->d_cur_low, lw, lh, d_ref_low, lw, lh,
                c->mbw, c->mbh, range, c->d_coarse );
            TIM_MARK( t_coarse );
            refine_search<<<(total + threads - 1) / threads, threads, 0, c->stream>>>(
                c->d_cur, c->w, c->h, d_ref, c->w, c->h,
                c->mbw, c->mbh, radius, c->d_coarse, c->d_out );
            TIM_MARK( t_refine );
            /* Stage the D2H through pinned memory: writing directly into the
             * frame's pageable candidate buffer would synchronise the stream
             * and serialise the analysis onto the calling thread. */
            if( cudaMemcpyAsync( c->h_out + (size_t)(list * X264_GPU_MAX_REFS + r) * total,
                                 c->d_out, (size_t)total * sizeof(Cand),
                                 cudaMemcpyDeviceToHost, c->stream ) != cudaSuccess )
                goto fail;
            TIM_MARK( t_d2h );
        }
        if( b_sync )
            frame->nfilled[list] = nref;
        else
            frame->pending[list] = nref;
    }
    if( b_sync )
    {
        if( cudaStreamSynchronize( c->stream ) != cudaSuccess )
            goto fail;
        TIM_MARK( t_sync );
        publish_candidates( c, frame, 0, frame->nfilled[0] );
        publish_candidates( c, frame, 1, frame->nfilled[1] );
    }
    if( c->prof )
    {
        c->t_call += gpu_now_ms() - t_call0;
        c->n_call++;
    }
    c->nframes++;
#undef TIM_MARK
    return 0;

fail:
    cudaStreamSynchronize( c->stream );
    c->disabled = 1;
    c->pending_frame = NULL;
    frame->nfilled[0] = frame->nfilled[1] = 0;
    frame->pending[0] = frame->pending[1] = 0;
    return -1;
}

extern "C" int x264_gpu_analyze( void *ctx, x264_gpu_frame_t *frame,
                                 const uint8_t *cur, int cur_stride,
                                 uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                                 uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                                 int range, int radius )
{
    return gpu_analyze_common( ctx, frame, cur, cur_stride,
                               refs0, ref_ids0, stride0, nref0,
                               refs1, ref_ids1, stride1, nref1,
                               range, radius, 1 );
}

extern "C" int x264_gpu_analyze_async( void *ctx, x264_gpu_frame_t *frame,
                                       const uint8_t *cur, int cur_stride,
                                       uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                                       uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                                       int range, int radius )
{
    return gpu_analyze_common( ctx, frame, cur, cur_stride,
                               refs0, ref_ids0, stride0, nref0,
                               refs1, ref_ids1, stride1, nref1,
                               range, radius, 0 );
}

/* Wait for an in-flight analysis and publish its results. */
extern "C" int x264_gpu_finish( void *ctx )
{
    x264_gpu_ctx_t *c = (x264_gpu_ctx_t*)ctx;
    if( !c || !c->pending_frame )
        return 0;
    std::lock_guard<std::mutex> guard( c->lock );
    x264_gpu_frame_t *f = c->pending_frame;
    double t0 = c->timing ? gpu_now_ms() : 0;
    if( cudaStreamSynchronize( c->stream ) != cudaSuccess )
    {
        c->disabled = 1;
        c->pending_frame = NULL;
        f->pending[0] = f->pending[1] = 0;
        return -1;
    }
    if( c->timing )
    {
        c->t_finish += gpu_now_ms() - t0;
        c->n_finish++;
    }
    publish_candidates( c, f, 0, f->pending[0] );
    publish_candidates( c, f, 1, f->pending[1] );
    f->nfilled[0] = f->pending[0];
    f->nfilled[1] = f->pending[1];
    f->pending[0] = f->pending[1] = 0;
    c->pending_frame = NULL;
    return 0;
}

extern "C" int x264_gpu_is_pending( void *ctx, const x264_gpu_frame_t *frame )
{
    x264_gpu_ctx_t *c = (x264_gpu_ctx_t*)ctx;
    return c && c->pending_frame && c->pending_frame == frame;
}

extern "C" int x264_gpu_pending_any( void *ctx )
{
    x264_gpu_ctx_t *c = (x264_gpu_ctx_t*)ctx;
    return c && c->pending_frame != NULL;
}

extern "C" const x264_gpu_cand_t *x264_gpu_frame_get( const x264_gpu_frame_t *f,
                                                      int list, const void *ref_plane, int64_t ref_id,
                                                      int mbx, int mby, int *count )
{
    /* Only references published by the most recent analysis have valid
     * candidates; everything else falls back to the normal CPU search.
     * Slots are matched by identity so candidates stay usable when the
     * analysis ran for a predicted reference set. */
    if( !f || !count || list < 0 || list > 1 )
        return NULL;
    if( mbx < 0 || mbx >= f->mbw || mby < 0 || mby >= f->mbh )
        return NULL;
    for( int r = 0; r < f->nfilled[list]; r++ )
        if( f->slot_ptr[list][r] == ref_plane && f->slot_id[list][r] == ref_id )
        {
            *count = f->ncand;
            return (const x264_gpu_cand_t *)( f->cand[list][r] +
                                             ((size_t)mby * f->mbw + mbx) * GK );
        }
    return NULL;
}

extern "C" int x264_gpu_frame_covers( const x264_gpu_frame_t *f,
                                      int list, const void *ref_plane, int64_t ref_id )
{
    if( !f || list < 0 || list > 1 )
        return 0;
    for( int r = 0; r < f->nfilled[list]; r++ )
        if( f->slot_ptr[list][r] == ref_plane && f->slot_id[list][r] == ref_id )
            return 1;
    return 0;
}
