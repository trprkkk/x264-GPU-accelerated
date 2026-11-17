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
    int pxsz;          /* bytes per pixel: 1 (8-bit) or 2 (10-bit & higher) */
    const char *ks_ds; /* kernel entry names for the configured depth */
    const char *ks_coarse;
    const char *ks_refine;
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

/* Pick and load the embedded kernel image for the current device.  cc >= 10
 * (Blackwell and newer) prefers the modern toolkit's image, everything else
 * the CUDA 11.8 image; both carry a PTX fallback for unknown devices.
 * X264_GPU_FATBIN=legacy|modern forces one image (test hook, single try).
 * tag is used in the diagnostic messages ("ME" / "LA"). */
static CUmodule gpu_load_module( const cudaDeviceProp *prop, const char *tag )
{
    static const int ids[2] = { GPU_RES_LEGACY, GPU_RES_MODERN };
    int first = prop->major >= 10 ? 1 : 0;
    int ntry = 2;
    int order[2];
    order[0] = ids[first];
    order[1] = ids[1 - first];
    const char *force = getenv( "X264_GPU_FATBIN" );
    if( force )
    {
        if( !strcmp( force, "legacy" ) )
            order[0] = GPU_RES_LEGACY;
        else if( !strcmp( force, "modern" ) )
            order[0] = GPU_RES_MODERN;
        ntry = 1;   /* forced: keep the outcome observable */
    }
    for( int i = 0; i < ntry; i++ )
    {
        size_t bsz = 0;
        const void *blob = gpu_res_blob( order[i], &bsz );
        if( !blob )
            continue;
        CUmodule mod = NULL;
        if( g_drv.cuModuleLoadData( &mod, blob ) == CUDA_SUCCESS )
        {
            if( force )
                fprintf( stderr, "x264 CUDA %s: using %s kernel image\n", tag,
                         order[i] == GPU_RES_LEGACY ? "legacy (CUDA 11.8)"
                                                    : "modern (CUDA 13.x)" );
            return mod;
        }
    }
    return NULL;
}

extern "C" void *x264_gpu_create( int width, int height, int max_refs, int device, int bit_depth )
{
    /* The kernels work on whole macroblocks; the caller passes the frame size
     * rounded up to the macroblock grid (x264 pads its planes to that size). */
    if( width <= 0 || height <= 0 || (width & 15) || (height & 15) )
        return NULL;
    if( bit_depth != 8 && (bit_depth < 9 || bit_depth > 12) )
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
    c->pxsz = (bit_depth == 8) ? 1 : 2;
    c->ks_ds     = (bit_depth == 8) ? "downsample4"    : "downsample4_10";
    c->ks_coarse = (bit_depth == 8) ? "coarse_search"  : "coarse_search_10";
    c->ks_refine = (bit_depth == 8) ? "refine_search"  : "refine_search_10";
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
    c->module = gpu_load_module( &prop, "ME" );
    int look_fail = 0;
    if( c->module )
    {
        if( g_drv.cuModuleGetFunction( &c->fn_ds, c->module, c->ks_ds ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_coarse, c->module, c->ks_coarse ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_refine, c->module, c->ks_refine ) != CUDA_SUCCESS )
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

    size_t fsz = (size_t)width * height * c->pxsz;
    size_t lsz = (size_t)c->lw * c->lh * c->pxsz;
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
                         int stride, int width, int height, int pxsz )
{
    if( !src || stride < width )
        return -1;
    size_t rb = (size_t)width * pxsz, sb = (size_t)stride * pxsz;
    if( stride == width )
        return cudaMemcpyAsync( dst, src, rb * height,
                                cudaMemcpyHostToDevice, st ) == cudaSuccess ? 0 : -1;
    return cudaMemcpy2DAsync( dst, rb, src, sb, rb, height,
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
    size_t rb = (size_t)width * c->pxsz, sb = (size_t)stride * c->pxsz;
    if( stride == width )
        memcpy( stage, src, rb * height );
    else
        for( int y = 0; y < height; y++ )
            memcpy( stage + y * rb, src + (size_t)y * sb, rb );
    if( cudaMemcpyAsync( dst, stage, rb * height,
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
                               const void *cur, int cur_stride,
                               void *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                               void *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
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

    if( upload_plane( c->stream, c->d_cur, (const uint8_t*)cur, cur_stride, c->w, c->h, c->pxsz ) < 0 )
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
        uint8_t *const *refs = list ? (uint8_t *const *)refs1 : (uint8_t *const *)refs0;
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
                if( upload_plane_pinned( c, d_ref, (const uint8_t*)refs[r], strides[r], c->w, c->h ) < 0 )
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
                                 const void *cur, int cur_stride,
                                 void *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                                 void *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                                 int range, int radius )
{
    return gpu_analyze_common( ctx, frame, cur, cur_stride,
                               refs0, ref_ids0, stride0, nref0,
                               refs1, ref_ids1, stride1, nref1,
                               range, radius, 1 );
}

extern "C" int x264_gpu_analyze_async( void *ctx, x264_gpu_frame_t *frame,
                                       const void *cur, int cur_stride,
                                       void *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                                       void *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
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

/* ================== lookahead (lowres) accelerator ========================
 * Exact-mode companion of the ME backend: it computes the values the CPU
 * lookahead would otherwise compute pixel-by-pixel:
 *   - la_intra: the raw per-MB intra cost (minimum predictor SATD/SAD),
 *   - la_table: the exhaustive 8x8 SAD square for one (fenc, ref) pair,
 * every value bit-identical to the CPU reference.  A failure here just makes
 * the caller fall back to the CPU, which computes the same values itself.  */

#define GPU_LA_RMAX       32     /* largest supported table radius */
#define GPU_LA_TAB_SLOTS  2      /* pinned ring: list0 + list1 per pass */

struct gpu_la_ctx_t
{
    int mbw, mbh;              /* lowres MB grid == full-resolution grid */
    int pxsz, maxpx;           /* bytes per pixel, max pixel value */
    CUmodule module;
    CUfunction fn_intra, fn_table;
    cudaStream_t stream;
    CUcontext cu_ctx;
    CUdevice  cu_dev;
    int cu_retained;
    int disabled;
    uint8_t *d_f, *d_r;        /* uploaded fenc / ref planes */
    size_t plane_cap;          /* capacity of d_f/d_r in bytes */
    int32_t *d_intra;
    uint16_t *d_tab;
    uint16_t *h_tab[GPU_LA_TAB_SLOTS];
    int tab_S;                 /* allocated square side (0 = none) */
    int tab_next;
};

static char gpu_la_name_buf[256] = "CUDA device";

static void gpu_la_cleanup( gpu_la_ctx_t *c )
{
    if( c->d_f ) cudaFree( c->d_f );
    if( c->d_r ) cudaFree( c->d_r );
    if( c->d_intra ) cudaFree( c->d_intra );
    if( c->d_tab ) cudaFree( c->d_tab );
    for( int i = 0; i < GPU_LA_TAB_SLOTS; i++ )
        if( c->h_tab[i] ) cudaFreeHost( c->h_tab[i] );
    if( c->stream ) cudaStreamDestroy( c->stream );
    if( c->module ) g_drv.cuModuleUnload( c->module );
    if( c->cu_retained ) g_drv.cuDevicePrimaryCtxRelease( c->cu_dev );
}

static int gpu_la_upload( gpu_la_ctx_t *c, uint8_t **pdst, const void *base,
                          int stride_px, int rows )
{
    size_t bytes = (size_t)stride_px * c->pxsz * rows;
    if( c->plane_cap < bytes )
    {
        /* Grown lazily on the first use; stride and rows stay constant for a
         * whole encode, so this runs at most twice. */
        if( c->d_f ) cudaFree( c->d_f );
        if( c->d_r ) cudaFree( c->d_r );
        c->d_f = c->d_r = NULL;
        if( cudaMalloc( (void**)&c->d_f, bytes ) != cudaSuccess ||
            cudaMalloc( (void**)&c->d_r, bytes ) != cudaSuccess )
        {
            c->plane_cap = 0;
            return -1;
        }
        c->plane_cap = bytes;
    }
    return cudaMemcpyAsync( *pdst, base, bytes, cudaMemcpyHostToDevice, c->stream )
           == cudaSuccess ? 0 : -1;
}

extern "C" void *x264_gpu_cuda_la_create( int mbw, int mbh, int device, int bit_depth )
{
    if( mbw <= 0 || mbh <= 0 )
        return NULL;
    if( bit_depth != 8 && (bit_depth < 9 || bit_depth > 12) )
        return NULL;

    int ndev = 0;
    if( cudaGetDeviceCount( &ndev ) != cudaSuccess || ndev <= 0 )
        return NULL;
    if( device < 0 || device >= ndev )
        device = 0;
    if( cudaSetDevice( device ) != cudaSuccess )
        return NULL;
    if( !gpu_drv_load() )
        return NULL;

    cudaDeviceProp prop;
    memset( &prop, 0, sizeof(prop) );
    if( cudaGetDeviceProperties( &prop, device ) == cudaSuccess )
        snprintf( gpu_la_name_buf, sizeof(gpu_la_name_buf), "%s (sm_%d%d)", prop.name,
                  prop.major, prop.minor );

    CUcontext cu_ctx = NULL;
    CUdevice  cu_dev = 0;
    if( g_drv.cuDeviceGet( &cu_dev, device ) != CUDA_SUCCESS ||
        g_drv.cuDevicePrimaryCtxRetain( &cu_ctx, cu_dev ) != CUDA_SUCCESS ||
        g_drv.cuCtxSetCurrent( cu_ctx ) != CUDA_SUCCESS )
    {
        fprintf( stderr, "x264 CUDA LA: cannot establish a CUDA context (device %d)\n", device );
        return NULL;
    }

    gpu_la_ctx_t *c = new (std::nothrow) gpu_la_ctx_t();
    if( !c )
    {
        g_drv.cuDevicePrimaryCtxRelease( cu_dev );
        return NULL;
    }
    c->mbw = mbw;
    c->mbh = mbh;
    c->pxsz = (bit_depth == 8) ? 1 : 2;
    c->maxpx = (1 << bit_depth) - 1;
    c->cu_ctx = cu_ctx;
    c->cu_dev = cu_dev;
    c->cu_retained = 1;
    c->module = gpu_load_module( &prop, "LA" );
    if( !c->module )
    {
        fprintf( stderr, "x264 CUDA LA: no kernel image for this device (sm_%d%d)\n",
                 prop.major, prop.minor );
        gpu_la_cleanup( c );
        delete c;
        return NULL;
    }
    const char *ks_i = (bit_depth == 8) ? "la_intra"   : "la_intra_10";
    const char *ks_t = (bit_depth == 8) ? "la_table"   : "la_table_10";
    if( g_drv.cuModuleGetFunction( &c->fn_intra, c->module, ks_i ) != CUDA_SUCCESS ||
        g_drv.cuModuleGetFunction( &c->fn_table, c->module, ks_t ) != CUDA_SUCCESS )
    {
        fprintf( stderr, "x264 CUDA LA: kernel entry points missing from the image\n" );
        gpu_la_cleanup( c );
        delete c;
        return NULL;
    }
    if( cudaStreamCreate( &c->stream ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_intra, (size_t)mbw * mbh * sizeof(int32_t) ) != cudaSuccess )
    {
        gpu_la_cleanup( c );
        delete c;
        return NULL;
    }
    return c;
}

extern "C" void x264_gpu_cuda_la_destroy( void *ctx )
{
    gpu_la_ctx_t *c = (gpu_la_ctx_t*)ctx;
    if( !c )
        return;
    gpu_la_cleanup( c );
    delete c;
}

extern "C" const char *x264_gpu_cuda_la_name( void )
{
    return gpu_la_name_buf;
}

extern "C" int x264_gpu_cuda_la_intra( void *ctx, const void *base, int base_off,
                                       int stride, int rows, int b_satd, int *out )
{
    gpu_la_ctx_t *c = (gpu_la_ctx_t*)ctx;
    if( !c || c->disabled || !base || rows < c->mbh * 8 || stride < c->mbw * 8 + 16 )
        return -1;
    if( gpu_la_upload( c, &c->d_f, base, stride, rows ) < 0 )
        goto fail;
    {
        int total = c->mbw * c->mbh;
        void *args[8] =
        {
            (void*)&c->d_f, (void*)&base_off, (void*)&stride,
            (void*)&c->mbw, (void*)&c->mbh, (void*)&b_satd, (void*)&c->maxpx,
            (void*)&c->d_intra
        };
        dim3 grid( (total + 255) / 256 ), block( 256 );
        if( g_drv.cuLaunchKernel( c->fn_intra, grid.x, 1, 1, block.x, 1, 1,
                                  0, (CUstream)c->stream, args, NULL ) != CUDA_SUCCESS )
            goto fail;
        if( cudaMemcpyAsync( out, c->d_intra, (size_t)total * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, c->stream ) != cudaSuccess )
            goto fail;
        if( cudaStreamSynchronize( c->stream ) != cudaSuccess )
            goto fail;
    }
    return 0;

fail:
    cudaStreamSynchronize( c->stream );
    c->disabled = 1;
    return -1;
}

extern "C" uint16_t *x264_gpu_cuda_la_table( void *ctx, const void *fenc_base, int fenc_off,
                                             int fenc_stride, const void *ref_base, int ref_off,
                                             int ref_stride, int rows, int R, int b_satd )
{
    gpu_la_ctx_t *c = (gpu_la_ctx_t*)ctx;
    if( !c || c->disabled || !fenc_base || !ref_base || R < 1 || R > GPU_LA_RMAX ||
        rows < c->mbh * 8 || fenc_stride < c->mbw * 8 + 16 || ref_stride < c->mbw * 8 + 16 )
        return NULL;
    int S = 2 * R + 1;
    size_t bytes = (size_t)c->mbw * c->mbh * S * S * sizeof(uint16_t);
    if( c->tab_S < S )
    {
        for( int i = 0; i < GPU_LA_TAB_SLOTS; i++ )
            if( c->h_tab[i] ) cudaFreeHost( c->h_tab[i] );
        if( c->d_tab ) cudaFree( c->d_tab );
        c->d_tab = NULL;
        for( int i = 0; i < GPU_LA_TAB_SLOTS; i++ )
            c->h_tab[i] = NULL;
        if( cudaMalloc( (void**)&c->d_tab, bytes ) != cudaSuccess )
            goto fail;
        for( int i = 0; i < GPU_LA_TAB_SLOTS; i++ )
            if( cudaMallocHost( (void**)&c->h_tab[i], bytes ) != cudaSuccess )
                goto fail;
        c->tab_S = S;
        c->tab_next = 0;
    }
    if( gpu_la_upload( c, &c->d_f, fenc_base, fenc_stride, rows ) < 0 )
        goto fail;
    if( gpu_la_upload( c, &c->d_r, ref_base, ref_stride, rows ) < 0 )
        goto fail;
    {
        int total = c->mbw * c->mbh * S;
        void *args[11] =
        {
            (void*)&c->d_f, (void*)&fenc_off, (void*)&c->d_r, (void*)&ref_off,
            (void*)&fenc_stride, (void*)&ref_stride,
            (void*)&c->mbw, (void*)&c->mbh, (void*)&R, (void*)&b_satd, (void*)&c->d_tab
        };
        dim3 grid( (total + 255) / 256 ), block( 256 );
        if( g_drv.cuLaunchKernel( c->fn_table, grid.x, 1, 1, block.x, 1, 1,
                                  0, (CUstream)c->stream, args, NULL ) != CUDA_SUCCESS )
            goto fail;
        int slot = c->tab_next;
        if( cudaMemcpyAsync( c->h_tab[slot], c->d_tab, bytes,
                             cudaMemcpyDeviceToHost, c->stream ) != cudaSuccess )
            goto fail;
        if( cudaStreamSynchronize( c->stream ) != cudaSuccess )
            goto fail;
        c->tab_next = (slot + 1) % GPU_LA_TAB_SLOTS;
        return c->h_tab[slot];
    }

fail:
    cudaStreamSynchronize( c->stream );
    c->disabled = 1;
    return NULL;
}

extern "C" const x264_gpu_la_backend_t x264_gpu_la_backend_cuda =
{
    X264_GPU_BACKEND_CUDA,
    "cuda",
    gpu_cuda_available,
    x264_gpu_cuda_la_create,
    x264_gpu_cuda_la_destroy,
    x264_gpu_cuda_la_name,
    x264_gpu_cuda_la_intra,
    x264_gpu_cuda_la_table,
};


