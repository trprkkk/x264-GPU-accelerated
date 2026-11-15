/*****************************************************************************
 * gpu.c: dispatcher for the GPU candidate generator
 *****************************************************************************
 * The public API declared in gpu.h is implemented here by forwarding to the
 * selected backend: CUDA (gpu_cuda.cpp with the kernels in gpu_kernels.cu) or
 * OpenCL (gpu_cl.c).  Frames start with a backend header so lookups can be
 * routed without extra state.
 *
 * When an analysis fails at runtime, x264_gpu_fallback() retires the failing
 * backend (kept alive so that frames still holding its candidate buffers can
 * be freed safely) and continues on the other backend.
 *****************************************************************************/

#include "gpu.h"

#include <stdlib.h>

typedef struct
{
    const x264_gpu_backend_t *be;
    void *impl;
    const x264_gpu_backend_t *old_be;   /* retired backend, kept for frame frees */
    void *old_impl;
    /* creation parameters, needed to build a fallback backend */
    int w, h, max_refs, device;
} gpu_ctx_pub_t;

static const x264_gpu_backend_t *gpu_backend_by_id( int id )
{
    switch( id )
    {
        case X264_GPU_BACKEND_CUDA:   return &x264_gpu_backend_cuda;
        case X264_GPU_BACKEND_OPENCL: return &x264_gpu_backend_cl;
    }
    return NULL;
}

static const x264_gpu_backend_t *gpu_backend_of_frame( const x264_gpu_frame_t *frame )
{
    if( !frame )
        return NULL;
    return gpu_backend_by_id( ((const x264_gpu_frame_header_t*)frame)->backend );
}

void *x264_gpu_create( int width, int height, int max_refs, int device, int backend )
{
    const x264_gpu_backend_t *be = NULL;
    void *impl = NULL;
    if( backend == X264_GPU_BACKEND_AUTO )
    {
        static const int order[2] = { X264_GPU_BACKEND_CUDA, X264_GPU_BACKEND_OPENCL };
        for( int i = 0; i < 2 && !impl; i++ )
        {
            be = gpu_backend_by_id( order[i] );
            if( be && be->available() )
                impl = be->create( width, height, max_refs, device );
        }
    }
    else
    {
        be = gpu_backend_by_id( backend );
        if( be && be->available() )
            impl = be->create( width, height, max_refs, device );
    }
    if( !impl )
        return NULL;
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)calloc( 1, sizeof(*pub) );
    if( !pub )
    {
        be->destroy( impl );
        return NULL;
    }
    pub->be = be;
    pub->impl = impl;
    pub->w = width;
    pub->h = height;
    pub->max_refs = max_refs;
    pub->device = device;
    return pub;
}

void x264_gpu_destroy( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub )
        return;
    pub->be->destroy( pub->impl );
    if( pub->old_be )
        pub->old_be->destroy( pub->old_impl );
    free( pub );
}

/* After an analysis failure: drain and retire the active backend and start
 * the other one.  At most one switch is allowed, so a double failure ends in
 * the regular CPU fallback.  The caller must have released the frames it
 * still holds (their buffers belong to the failing backend). */
int x264_gpu_fallback( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub || pub->old_be )
        return -1;
    static const int order[2] = { X264_GPU_BACKEND_CUDA, X264_GPU_BACKEND_OPENCL };
    for( int i = 0; i < 2; i++ )
    {
        const x264_gpu_backend_t *be = gpu_backend_by_id( order[i] );
        if( !be || be == pub->be || !be->available() )
            continue;
        pub->be->finish( pub->impl );   /* drain anything in flight */
        void *impl = be->create( pub->w, pub->h, pub->max_refs, pub->device );
        if( !impl )
            continue;
        pub->old_be = pub->be;
        pub->old_impl = pub->impl;
        pub->be = be;
        pub->impl = impl;
        return 0;
    }
    return -1;
}

const char *x264_gpu_name( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->name() : "";
}

const char *x264_gpu_backend_id( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->name_id : "";
}

x264_gpu_frame_t *x264_gpu_frame_alloc( void *ctx, int mbw, int mbh, int nref0, int nref1 )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->frame_alloc( mbw, mbh, nref0, nref1 ) : NULL;
}

void x264_gpu_frame_free( x264_gpu_frame_t *frame )
{
    const x264_gpu_backend_t *be = gpu_backend_of_frame( frame );
    if( be )
        be->frame_free( frame );
}

/* The analysis can only run on frames allocated by the active backend. */
static int gpu_frame_is_current( const gpu_ctx_pub_t *pub, const x264_gpu_frame_t *frame )
{
    const x264_gpu_backend_t *fbe = gpu_backend_of_frame( frame );
    return fbe && fbe == pub->be;
}

#define GPU_ANALYZE_ARGS \
    x264_gpu_frame_t *frame, const uint8_t *cur, int cur_stride, \
    uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0, \
    uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1, \
    int range, int radius

int x264_gpu_analyze( void *ctx, GPU_ANALYZE_ARGS )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub || !gpu_frame_is_current( pub, frame ) )
        return -1;
    return pub->be->analyze( pub->impl, frame, cur, cur_stride,
                             refs0, ref_ids0, stride0, nref0,
                             refs1, ref_ids1, stride1, nref1, range, radius );
}

int x264_gpu_analyze_async( void *ctx, GPU_ANALYZE_ARGS )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub || !gpu_frame_is_current( pub, frame ) )
        return -1;
    return pub->be->analyze_async( pub->impl, frame, cur, cur_stride,
                                   refs0, ref_ids0, stride0, nref0,
                                   refs1, ref_ids1, stride1, nref1, range, radius );
}

int x264_gpu_finish( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->finish( pub->impl ) : 0;
}

int x264_gpu_is_pending( void *ctx, const x264_gpu_frame_t *frame )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->is_pending( pub->impl, frame ) : 0;
}

int x264_gpu_pending_any( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->pending_any( pub->impl ) : 0;
}

const x264_gpu_cand_t *x264_gpu_frame_get( const x264_gpu_frame_t *frame,
                                           int list, const void *ref_plane, int64_t ref_id,
                                           int mbx, int mby, int *count )
{
    const x264_gpu_backend_t *be = gpu_backend_of_frame( frame );
    return be ? be->frame_get( frame, list, ref_plane, ref_id, mbx, mby, count ) : NULL;
}

int x264_gpu_frame_covers( const x264_gpu_frame_t *frame,
                           int list, const void *ref_plane, int64_t ref_id )
{
    const x264_gpu_backend_t *be = gpu_backend_of_frame( frame );
    return be ? be->frame_covers( frame, list, ref_plane, ref_id ) : 0;
}
