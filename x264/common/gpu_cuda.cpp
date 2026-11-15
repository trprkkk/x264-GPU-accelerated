// CUDA host implementation of the hierarchical candidate generator.
//
// The GPU does a coarse exhaustive search on a quarter-resolution luma copy
// and then refines the best four integer-pel positions at full resolution.
// It is deliberately an *open-loop* pre-analysis: it uses the reference
// planes that x264 already reconstructed, but never writes any coding
// decision.  me.c evaluates every returned vector with the real x264 cost
// function, so rate control, RDO, entropy coding and the bitstream syntax stay
// exactly in the original CPU code.
//
// This file is the CUDA backend host side; the public API (dispatcher) is
// gpu.c and the symbols here carry the x264_gpu_cuda_ prefix.
//
// Device code lives in gpu_kernels.cu and is compiled to fat binaries by nvcc
// (CUDA 11.8 for sm_35..sm_90, CUDA 13.x for the newer architectures).  The
// fat binaries are embedded in x264.exe as RCDATA resources and loaded here
// through the CUDA driver API from nvcuda.dll, which is loaded dynamically:
// no CUDA runtime library has to be present, the kernels can come from two
// different toolkits, and the host code stays independent of the (much newer)
// nvcc that could no longer build for older GPUs.  The host API (memory,
// streams, copies) uses the CUDA runtime (cudart, statically linked) with
// CUDA 11.8, which is the last runtime that supports Kepler..Volta drivers.
//
// The fat binary is picked from the device's compute capability:
//   * cc < 10  -> legacy image (CUDA 11.8: sm_35..sm_90 + compute_75 PTX)
//   * cc >= 10 -> modern image (CUDA 13.x: sm_100/sm_120 + compute_90 PTX)
// When the preferred image cannot be loaded (missing resource, unsupported
// device, ...) the other one is tried; if that fails too, x264_gpu_create()
// returns NULL and the dispatcher falls back to OpenCL or the CPU.

/* x264 ships an extras/stdint.h fallback which, following C++98, only defines
 * the (U)INT*_MIN/MAX macros when these are requested; the C++ headers used
 * below (through <ratio>) need them, and the decision is latched when
 * <stdint.h> is first included. */
#define __STDC_LIMIT_MACROS
#define __STDC_CONSTANT_MACROS

#include "gpu.h"
#include "gpu_kernels.h"

/* Rename the backend entry points (definitions below only; the public
 * declarations from gpu.h keep their names and are implemented by gpu.c). */
#define x264_gpu_create       x264_gpu_cuda_create
#define x264_gpu_destroy      x264_gpu_cuda_destroy
#define x264_gpu_name         x264_gpu_cuda_name
#define x264_gpu_frame_alloc  x264_gpu_cuda_frame_alloc
#define x264_gpu_frame_free   x264_gpu_cuda_frame_free
#define x264_gpu_analyze      x264_gpu_cuda_analyze
#define x264_gpu_analyze_async x264_gpu_cuda_analyze_async
#define x264_gpu_finish       x264_gpu_cuda_finish
#define x264_gpu_is_pending   x264_gpu_cuda_is_pending
#define x264_gpu_pending_any  x264_gpu_cuda_pending_any
#define x264_gpu_frame_get    x264_gpu_cuda_frame_get
#define x264_gpu_frame_covers x264_gpu_cuda_frame_covers

/* Forward declarations for calls that appear before the definitions. */
extern "C" x264_gpu_frame_t *x264_gpu_frame_alloc( int mbw, int mbh, int nref0, int nref1 );
extern "C" void x264_gpu_frame_free( x264_gpu_frame_t *frame );
extern "C" int  x264_gpu_finish( void *ctx );

/* CUDA 11.8 predates the MSVC toolchain used here; the runtime API and the
 * headers work fine, only the version gates in crt/host_config.h refuse the
 * newer compiler. */
#define __NV_NO_HOST_COMPILER_CHECK
#include <cuda.h>
#include <cuda_runtime.h>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

/* Resource ids of the embedded kernel images (kept in sync with the generated
 * resource script, see the Makefile). */
#define GPU_RES_LEGACY 101
#define GPU_RES_MODERN 102

#define GK X264_GPU_CANDS

/* --- CUDA driver API, loaded dynamically from nvcuda.dll ------------------ */

namespace
{
struct GpuDriverApi
{
    decltype(&cuModuleLoadData)          cuModuleLoadData;
    decltype(&cuModuleGetFunction)       cuModuleGetFunction;
    decltype(&cuModuleUnload)            cuModuleUnload;
    decltype(&cuLaunchKernel)            cuLaunchKernel;
    decltype(&cuCtxSetCurrent)           cuCtxSetCurrent;
    decltype(&cuDeviceGet)               cuDeviceGet;
    decltype(&cuDevicePrimaryCtxRetain)  cuDevicePrimaryCtxRetain;
    decltype(&cuDevicePrimaryCtxRelease) cuDevicePrimaryCtxRelease;
};
}

static GpuDriverApi g_drv;
static int g_drv_state = -1;   /* -1: unknown, 0: unavailable, 1: ready */

static int gpu_drv_load( void )
{
    if( g_drv_state >= 0 )
        return g_drv_state;
    HMODULE lib = LoadLibraryW( L"nvcuda.dll" );
    if( !lib )
    {
        g_drv_state = 0;
        return 0;
    }
    #define GPU_DRV_LOAD( name ) (FARPROC&)g_drv.name = GetProcAddress( lib, #name )
    GPU_DRV_LOAD( cuModuleLoadData );
    GPU_DRV_LOAD( cuModuleGetFunction );
    GPU_DRV_LOAD( cuModuleUnload );
    GPU_DRV_LOAD( cuLaunchKernel );
    GPU_DRV_LOAD( cuCtxSetCurrent );
    GPU_DRV_LOAD( cuDeviceGet );
    GPU_DRV_LOAD( cuDevicePrimaryCtxRetain );
    GPU_DRV_LOAD( cuDevicePrimaryCtxRelease );
    #undef GPU_DRV_LOAD
    g_drv_state = g_drv.cuModuleLoadData && g_drv.cuModuleGetFunction &&
                  g_drv.cuModuleUnload && g_drv.cuLaunchKernel &&
                  g_drv.cuCtxSetCurrent &&
                  g_drv.cuDeviceGet && g_drv.cuDevicePrimaryCtxRetain &&
                  g_drv.cuDevicePrimaryCtxRelease;
    return g_drv_state;
}

/* Kernel images embedded as RCDATA resources in the executable. */
static const void *gpu_res_blob( int id, size_t *size )
{
    HRSRC res = FindResourceW( NULL, MAKEINTRESOURCEW( id ), (LPCWSTR)RT_RCDATA );
    if( !res )
        return NULL;
    DWORD sz = SizeofResource( NULL, res );
    if( !sz )
        return NULL;
    HGLOBAL dat = LoadResource( NULL, res );
    if( !dat )
        return NULL;
    const void *p = LockResource( dat );
    if( !p )
        return NULL;
    *size = (size_t)sz;
    return p;
}

static double gpu_now_ms()
{
    static LARGE_INTEGER freq = {};
    LARGE_INTEGER c;
    if( !freq.QuadPart )
        QueryPerformanceFrequency( &freq );
    QueryPerformanceCounter( &c );
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

struct x264_gpu_ctx_t;

struct x264_gpu_frame_t
{
    x264_gpu_frame_header_t hdr;   /* must be first: dispatcher reads it */
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

/* Pinned staging ring for reference uploads.  Two buffers are alternated and
 * each async copy out of a buffer is fenced with an event, so an upload that
 * starts while the previous copy is still reading its staging buffer cannot
 * corrupt it (the two are of comparable speed on old, slow PCIe links). */
#define X264_GPU_STAGE 2

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
    uint8_t *h_ref[X264_GPU_STAGE];    /* pinned staging ring for ref uploads */
    cudaEvent_t stage_done[X264_GPU_STAGE];
    int stage_next;
    Cand    *h_out;            /* pinned staging for candidate D2H copies */
    /* Kernel module (loaded from an embedded fat binary). */
    CUmodule module;
    CUfunction fn_ds;          /* downsample4 */
    CUfunction fn_coarse;      /* coarse_search */
    CUfunction fn_refine;      /* refine_search */
    CUcontext cu_ctx;          /* primary context (retained when we created it) */
    CUdevice  cu_dev;
    int cu_retained;
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

/* Launch one of the module's kernels on the analysis stream. */
static int gpu_launch( x264_gpu_ctx_t *c, CUfunction fn, dim3 grid, dim3 block, void **args )
{
    if( !fn || g_drv.cuLaunchKernel( fn, grid.x, grid.y, grid.z,
                                     block.x, block.y, block.z,
                                     0, (CUstream)c->stream, args, NULL ) != CUDA_SUCCESS )
        return -1;
    return 0;
}

extern "C" void *x264_gpu_create( int width, int height, int max_refs, int device )
{
    /* The kernels work on whole macroblocks; the caller passes the frame size
     * rounded up to the macroblock grid (x264 pads its planes to that size). */
    if( width <= 0 || height <= 0 || (width & 15) || (height & 15) )
        return NULL;

    int ndev = 0;
    cudaError_t ce = cudaGetDeviceCount( &ndev );
    if( ce != cudaSuccess || ndev <= 0 )
        return NULL;
    if( device < 0 || device >= ndev )
        device = 0;
    ce = cudaSetDevice( device );
    if( ce != cudaSuccess )
        return NULL;

    if( !gpu_drv_load() )
        return NULL;

    cudaDeviceProp prop;
    memset( &prop, 0, sizeof(prop) );
    if( cudaGetDeviceProperties( &prop, device ) == cudaSuccess )
        snprintf( gpu_name, sizeof(gpu_name), "%s (sm_%d%d)", prop.name,
                  prop.major, prop.minor );

    /* Retain the primary context explicitly instead of trusting whatever the
     * runtime left current: some cudart versions expose a destroyed handle
     * after their device queries, and the driver API needs a live current
     * context.  cudart shares the same primary context, so this only adds a
     * reference. */
    CUcontext cu_ctx = NULL;
    CUdevice  cu_dev = 0;
    int cu_retained = 0;
    if( g_drv.cuDeviceGet( &cu_dev, device ) != CUDA_SUCCESS ||
        g_drv.cuDevicePrimaryCtxRetain( &cu_ctx, cu_dev ) != CUDA_SUCCESS ||
        g_drv.cuCtxSetCurrent( cu_ctx ) != CUDA_SUCCESS )
    {
        fprintf( stderr, "x264 CUDA ME: cannot establish a CUDA context (device %d)\n", device );
        return NULL;
    }
    cu_retained = 1;

    x264_gpu_ctx_t *c = new (std::nothrow) x264_gpu_ctx_t();
    if( !c )
    {
        if( cu_retained )
            g_drv.cuDevicePrimaryCtxRelease( cu_dev );
        return NULL;
    }
    c->w = width;
    c->h = height;
    c->lw = (width + 3) / 4;
    c->lh = (height + 3) / 4;
    c->mbw = (width + 15) / 16;
    c->mbh = (height + 15) / 16;
    c->max_refs = max_refs < 1 ? 1 : (max_refs > X264_GPU_MAX_REFS ? X264_GPU_MAX_REFS : max_refs);
    c->device = device;
    c->disabled = 0;
    c->cu_ctx = cu_ctx;
    c->cu_dev = cu_dev;
    c->cu_retained = cu_retained;
    c->timing = getenv( "X264_GPU_TIMING" ) != NULL;
    c->prof = getenv( "X264_GPU_PROF" ) != NULL;
    if( c->timing )
        printf( "x264 CUDA ME timing: %dx%d, %d refs, device %d\n", width, height, c->max_refs, device );

    /* Pick and load the kernel image.  cc >= 10 (Blackwell and newer) uses
     * the image built by the modern toolkit, everything else the CUDA 11.8
     * image; both carry a PTX fallback for unknown devices. */
    {
        static const int ids[2] = { GPU_RES_LEGACY, GPU_RES_MODERN };
        int first = prop.major >= 10 ? 1 : 0;
        int ntry = 2;
        int order[2];
        order[0] = ids[first];
        order[1] = ids[1-first];
        const char *force = getenv( "X264_GPU_FATBIN" );   /* test hook */
        if( force )
        {
            if( !strcmp( force, "legacy" ) )
                order[0] = GPU_RES_LEGACY;
            else if( !strcmp( force, "modern" ) )
                order[0] = GPU_RES_MODERN;
            ntry = 1;   /* forced: keep the outcome observable */
        }
        for( int i = 0; i < ntry && !c->module; i++ )
        {
            size_t bsz = 0;
            const void *blob = gpu_res_blob( order[i], &bsz );
            if( !blob )
                continue;
            if( g_drv.cuModuleLoadData( &c->module, blob ) == CUDA_SUCCESS )
            {
                if( force )
                    fprintf( stderr, "x264 CUDA ME: using %s kernel image\n",
                             order[i] == GPU_RES_LEGACY ? "legacy (CUDA 11.8)"
                                                        : "modern (CUDA 13.x)" );
            }
            else
                c->module = NULL;
        }
    }
    int look_fail = 0;
    if( c->module )
    {
        if( g_drv.cuModuleGetFunction( &c->fn_ds, c->module, "downsample4" ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_coarse, c->module, "coarse_search" ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_refine, c->module, "refine_search" ) != CUDA_SUCCESS )
        {
            fprintf( stderr, "x264 CUDA ME: kernel entry points missing from the image\n" );
            look_fail = 1;
        }
    }
    else
        fprintf( stderr, "x264 CUDA ME: no kernel image for this device (sm_%d%d)\n",
                 prop.major, prop.minor );
    if( !c->module || look_fail )
    {
        if( c->module )
            g_drv.cuModuleUnload( c->module );
        if( cu_retained )
            g_drv.cuDevicePrimaryCtxRelease( cu_dev );
        delete c;
        return NULL;
    }

    size_t fsz = (size_t)width * height;
    size_t lsz = (size_t)c->lw * c->lh;
    size_t csz = (size_t)c->mbw * c->mbh * GK * sizeof(Cand);
    int fail = 0;
    if( cudaStreamCreate( &c->stream ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_cur, fsz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_cur_low, lsz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_coarse, csz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_out, csz ) != cudaSuccess ||
        cudaMallocHost( (void**)&c->h_out, 2 * X264_GPU_MAX_REFS * csz ) != cudaSuccess )
        fail = 1;
    for( int i = 0; i < X264_GPU_STAGE && !fail; i++ )
    {
        if( cudaMallocHost( (void**)&c->h_ref[i], fsz ) != cudaSuccess ||
            cudaEventCreateWithFlags( &c->stage_done[i], cudaEventDisableTiming ) != cudaSuccess )
            fail = 1;
        else
            cudaEventRecord( c->stage_done[i], c->stream );   /* start "done" */
    }
    for( int i = 0; i < X264_GPU_CACHE && !fail; i++ )
        if( cudaMalloc( (void**)&c->cache[i].full, fsz ) != cudaSuccess ||
            cudaMalloc( (void**)&c->cache[i].low, lsz ) != cudaSuccess )
            fail = 1;
    if( !fail )
    {
        /* Probe launch: verify that the loaded image can actually run on this
         * device (unsupported architecture, broken driver, ...).  On failure
         * this backend is rejected and the dispatcher can fall back. */
        const uint8_t *a_src = c->d_cur;   int a_sw = 16, a_sh = 16;
        uint8_t *a_dst = c->d_cur_low;     int a_dw = 4, a_dh = 4;
        void *args[] = { &a_src, &a_sw, &a_sh, &a_dst, &a_dw, &a_dh };
        if( gpu_launch( c, c->fn_ds, dim3( 1 ), dim3( 1 ), args ) < 0 ||
            cudaStreamSynchronize( c->stream ) != cudaSuccess )
        {
            cudaError_t pe = cudaGetLastError();   /* clear the sticky error */
            fprintf( stderr, "x264 CUDA ME: kernel launch failed on this device (%d: %s)\n",
                     (int)pe, cudaGetErrorString( pe ) );
            fail = 1;
        }
    }
    if( fail )
    {
        if( c->stream ) cudaStreamDestroy( c->stream );
        if( c->d_cur ) cudaFree( c->d_cur );
        if( c->d_cur_low ) cudaFree( c->d_cur_low );
        if( c->d_coarse ) cudaFree( c->d_coarse );
        if( c->d_out ) cudaFree( c->d_out );
        for( int i = 0; i < X264_GPU_STAGE; i++ )
        {
            if( c->h_ref[i] ) cudaFreeHost( c->h_ref[i] );
            if( c->stage_done[i] ) cudaEventDestroy( c->stage_done[i] );
        }
        if( c->h_out ) cudaFreeHost( c->h_out );
        for( int i = 0; i < X264_GPU_CACHE; i++ )
        {
            if( c->cache[i].full ) cudaFree( c->cache[i].full );
            if( c->cache[i].low ) cudaFree( c->cache[i].low );
        }
        if( c->module ) g_drv.cuModuleUnload( c->module );
        if( cu_retained ) g_drv.cuDevicePrimaryCtxRelease( cu_dev );
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
    for( int i = 0; i < X264_GPU_STAGE; i++ )
    {
        if( c->h_ref[i] ) cudaFreeHost( c->h_ref[i] );
        if( c->stage_done[i] ) cudaEventDestroy( c->stage_done[i] );
    }
    if( c->h_out ) cudaFreeHost( c->h_out );
    for( int i = 0; i < X264_GPU_CACHE; i++ )
    {
        if( c->cache[i].full ) cudaFree( c->cache[i].full );
        if( c->cache[i].low ) cudaFree( c->cache[i].low );
    }
    if( c->module ) g_drv.cuModuleUnload( c->module );
    if( c->cu_retained ) g_drv.cuDevicePrimaryCtxRelease( c->cu_dev );
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
    f->hdr.backend = X264_GPU_BACKEND_CUDA;
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

/* Upload through the pinned staging ring: pageable sources would make the
 * asynchronous copy synchronise the stream, which serialises the whole
 * analysis onto the calling thread.  Each buffer is reused only after the
 * copy that last read it has completed (see the ring comment). */
static int upload_plane_pinned( x264_gpu_ctx_t *c, uint8_t *dst, const uint8_t *src,
                                int stride, int width, int height )
{
    if( !src || stride < width )
        return -1;
    int slot = c->stage_next++ % X264_GPU_STAGE;
    if( cudaEventQuery( c->stage_done[slot] ) == cudaErrorNotReady )
        cudaEventSynchronize( c->stage_done[slot] );
    uint8_t *stage = c->h_ref[slot];
    if( stride == width )
        memcpy( stage, src, (size_t)width * height );
    else
        for( int y = 0; y < height; y++ )
            memcpy( stage + (size_t)y * width, src + (size_t)y * stride, width );
    if( cudaMemcpyAsync( dst, stage, (size_t)width * height,
                         cudaMemcpyHostToDevice, c->stream ) != cudaSuccess )
        return -1;
    if( cudaEventRecord( c->stage_done[slot], c->stream ) != cudaSuccess )
        return -1;
    return 0;
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

    /* Testing hook: X264_GPU_SIMULATE_FAIL=cuda makes this backend fail at
     * the first analysis (used to exercise the OpenCL fallback). */
    {
        static int sim = -1;
        if( sim < 0 )
        {
            const char *ev = getenv( "X264_GPU_SIMULATE_FAIL" );
            sim = ev && !strcmp( ev, "cuda" );
        }
        if( sim )
        {
            c->disabled = 1;
            c->pending_frame = NULL;
            return -1;
        }
    }

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
    {
        const uint8_t *a_src = c->d_cur;   int a_sw = c->w, a_sh = c->h;
        uint8_t *a_dst = c->d_cur_low;     int a_dw = lw, a_dh = lh;
        void *args[] = { &a_src, &a_sw, &a_sh, &a_dst, &a_dw, &a_dh };
        if( gpu_launch( c, c->fn_ds, grd, blk, args ) < 0 )
            goto fail;
    }
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
                if( upload_plane_pinned( c, d_ref, refs[r], strides[r], c->w, c->h ) < 0 )
                    goto fail;
                TIM_MARK( t_upload_ref );
                {
                    const uint8_t *a_src = d_ref;   int a_sw = c->w, a_sh = c->h;
                    uint8_t *a_dst = d_ref_low;     int a_dw = lw, a_dh = lh;
                    void *args[] = { &a_src, &a_sw, &a_sh, &a_dst, &a_dw, &a_dh };
                    if( gpu_launch( c, c->fn_ds, grd, blk, args ) < 0 )
                        goto fail;
                }
                TIM_MARK( t_down );
                c->cache[slot].ptr = refs[r];
                c->cache[slot].id = ids[r];
            }
            {
                const uint8_t *a_cur = c->d_cur_low; int a_cw = lw, a_ch = lh;
                const uint8_t *a_ref = d_ref_low;    int a_rw = lw, a_rh = lh;
                int a_mbw = c->mbw, a_mbh = c->mbh, a_range = range;
                Cand *a_out = c->d_coarse;
                void *args[] = { &a_cur, &a_cw, &a_ch, &a_ref, &a_rw, &a_rh,
                                 &a_mbw, &a_mbh, &a_range, &a_out };
                if( gpu_launch( c, c->fn_coarse, dim3( (mbcount + threads - 1) / threads ),
                                dim3( threads ), args ) < 0 )
                    goto fail;
            }
            TIM_MARK( t_coarse );
            {
                const uint8_t *a_cur = c->d_cur; int a_cw = c->w, a_ch = c->h;
                const uint8_t *a_ref = d_ref;    int a_rw = c->w, a_rh = c->h;
                int a_mbw = c->mbw, a_mbh = c->mbh, a_radius = radius;
                const Cand *a_coarse = c->d_coarse;
                Cand *a_out = c->d_out;
                void *args[] = { &a_cur, &a_cw, &a_ch, &a_ref, &a_rw, &a_rh,
                                 &a_mbw, &a_mbh, &a_radius, &a_coarse, &a_out };
                if( gpu_launch( c, c->fn_refine, dim3( (total + threads - 1) / threads ),
                                dim3( threads ), args ) < 0 )
                    goto fail;
            }
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

/* --- backend registration ------------------------------------------------- */

static int gpu_cuda_available( void )
{
    int ndev = 0;
    return cudaGetDeviceCount( &ndev ) == cudaSuccess && ndev > 0;
}

extern "C" const x264_gpu_backend_t x264_gpu_backend_cuda =
{
    X264_GPU_BACKEND_CUDA,
    "cuda",
    gpu_cuda_available,
    x264_gpu_cuda_create,
    x264_gpu_cuda_destroy,
    x264_gpu_cuda_name,
    x264_gpu_cuda_frame_alloc,
    x264_gpu_cuda_frame_free,
    x264_gpu_cuda_analyze,
    x264_gpu_cuda_analyze_async,
    x264_gpu_cuda_finish,
    x264_gpu_cuda_is_pending,
    x264_gpu_cuda_pending_any,
    x264_gpu_cuda_frame_get,
    x264_gpu_cuda_frame_covers,
};
