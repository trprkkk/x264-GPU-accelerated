/*****************************************************************************
 * gpu.c: dispatcher for the GPU candidate generator
 *****************************************************************************
 * The public API declared in gpu.h is implemented here by forwarding to the
 * selected backend: CUDA (gpu.cu) or OpenCL (gpu_cl.c).  Frames start with a
 * backend header so lookups can be routed without extra state.
 *****************************************************************************/

#include "gpu.h"

#include <stdlib.h>

typedef struct
{
    const x264_gpu_backend_t *be;
    void *impl;
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
    return pub;
}

void x264_gpu_destroy( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub )
        return;
    pub->be->destroy( pub->impl );
    free( pub );
}

const char *x264_gpu_name( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->name() : "";
}

const char *x264_gpu_backend_id( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->id : "";
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

#define GPU_ANALYZE_ARGS \
    x264_gpu_frame_t *frame, const uint8_t *cur, int cur_stride, \
    uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0, \
    uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1, \
    int range, int radius

int x264_gpu_analyze( void *ctx, GPU_ANALYZE_ARGS )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->analyze( pub->impl, frame, cur, cur_stride,
                                   refs0, ref_ids0, stride0, nref0,
                                   refs1, ref_ids1, stride1, nref1, range, radius ) : -1;
}

int x264_gpu_analyze_async( void *ctx, GPU_ANALYZE_ARGS )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->analyze_async( pub->impl, frame, cur, cur_stride,
                                         refs0, ref_ids0, stride0, nref0,
                                         refs1, ref_ids1, stride1, nref1, range, radius ) : -1;
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
