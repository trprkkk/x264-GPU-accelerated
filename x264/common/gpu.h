/*****************************************************************************
 * gpu.h: GPU candidate generation for the hybrid x264 fork
 *****************************************************************************
 * This module never makes the final coding decision.  It returns a small
 * list of integer-pel motion vectors per macroblock; the unmodified x264
 * rate-distortion code evaluates them against the real reconstructed
 * reference and remains the only authority on the output bitstream.
 *
 * common/gpu.c is the dispatcher for the public API below; the actual
 * implementations live in common/gpu.cu (CUDA) and common/gpu_cl.c (OpenCL)
 * and are selected with x264_gpu_create()'s backend argument.
 *****************************************************************************/

#ifndef X264_GPU_H
#define X264_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define X264_GPU_MAX_REFS  4
#define X264_GPU_CANDS     4
#define X264_GPU_CACHE     8

/* Backend selection for x264_gpu_create(). */
#define X264_GPU_BACKEND_AUTO   0   /* CUDA first, then OpenCL */
#define X264_GPU_BACKEND_CUDA   1
#define X264_GPU_BACKEND_OPENCL 2

typedef struct
{
    int16_t  mv[2];   /* full-pel, relative to the block origin */
    uint32_t sad;     /* integer SAD that produced the candidate */
} x264_gpu_cand_t;

typedef struct x264_gpu_frame_t x264_gpu_frame_t;

/* Every backend's frame object starts with this header so that the
 * dispatcher can find the owning backend from an opaque frame pointer. */
typedef struct
{
    int backend;      /* X264_GPU_BACKEND_* */
} x264_gpu_frame_header_t;

typedef struct x264_gpu_backend_t
{
    const char *id;      /* "cuda" / "opencl" */
    int      (*available)( void );
    void *   (*create)( int width, int height, int max_refs, int device );
    void     (*destroy)( void *ctx );
    const char *(*name)( void );
    x264_gpu_frame_t *(*frame_alloc)( int mbw, int mbh, int nref0, int nref1 );
    void     (*frame_free)( x264_gpu_frame_t *frame );
    int      (*analyze)( void *ctx, x264_gpu_frame_t *frame,
                         const uint8_t *cur, int cur_stride,
                         uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                         uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                         int range, int radius );
    int      (*analyze_async)( void *ctx, x264_gpu_frame_t *frame,
                               const uint8_t *cur, int cur_stride,
                               uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                               uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                               int range, int radius );
    int      (*finish)( void *ctx );
    int      (*is_pending)( void *ctx, const x264_gpu_frame_t *frame );
    int      (*pending_any)( void *ctx );
    const x264_gpu_cand_t *(*frame_get)( const x264_gpu_frame_t *frame,
                                         int list, const void *ref_plane, int64_t ref_id,
                                         int mbx, int mby, int *count );
    int      (*frame_covers)( const x264_gpu_frame_t *frame,
                              int list, const void *ref_plane, int64_t ref_id );
} x264_gpu_backend_t;

/* Backend instances (defined in gpu.cu / gpu_cl.c). */
extern const x264_gpu_backend_t x264_gpu_backend_cuda;
extern const x264_gpu_backend_t x264_gpu_backend_cl;

/* Create a candidate generator on the requested backend (0 = auto).
 * Returns NULL when the selected backend is unavailable or init fails. */
void *x264_gpu_create( int width, int height, int max_refs, int device, int backend );
void  x264_gpu_destroy( void *ctx );
/* Human-readable name of the active backend/device ("" when none). */
const char *x264_gpu_name( void *ctx );
/* Backend id of a completed create() (for logging). */
const char *x264_gpu_backend_id( void *ctx );

x264_gpu_frame_t *x264_gpu_frame_alloc( void *ctx, int mbw, int mbh, int nref0, int nref1 );
void              x264_gpu_frame_free( x264_gpu_frame_t *frame );

/* Upload the current picture and the references, run the coarse-to-fine
 * search, and fill the frame's candidate arrays.  ref_ids are opaque keys
 * (identifying the reference content together with the plane pointer) used
 * to cache reference planes on the device.  Returns 0 on success. */
int x264_gpu_analyze( void *ctx, x264_gpu_frame_t *frame,
                      const uint8_t *cur, int cur_stride,
                      uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                      uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                      int range, int radius );

/* Asynchronous (prefetch) variant of x264_gpu_analyze: launches the same work
 * but returns before the results are ready.  The candidates become visible
 * only after x264_gpu_finish(), which also refuses to publish work whose
 * references no longer match.  At most one analysis may be in flight. */
int x264_gpu_analyze_async( void *ctx, x264_gpu_frame_t *frame,
                            const uint8_t *cur, int cur_stride,
                            uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                            uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                            int range, int radius );

/* Wait for an in-flight analysis and publish its results (nfilled).  Returns
 * 0 when there was nothing pending or the wait succeeded, -1 on failure. */
int x264_gpu_finish( void *ctx );
/* 1 when `frame` currently has an analysis in flight. */
int x264_gpu_is_pending( void *ctx, const x264_gpu_frame_t *frame );
/* 1 when any analysis is in flight. */
int x264_gpu_pending_any( void *ctx );

/* Candidate lookup used by me.c.  The reference is identified by the
 * reference frame's luma plane pointer and its i_frame, so candidates remain
 * usable even when the analysis was launched for a predicted reference set.
 * Returns NULL when this list/reference/MB has no candidates (10-bit encode,
 * unsupported size, reference not analysed, ...). */
const x264_gpu_cand_t *x264_gpu_frame_get( const x264_gpu_frame_t *frame,
                                           int list, const void *ref_plane, int64_t ref_id,
                                           int mbx, int mby, int *count );

/* 1 when a candidate slot for (list, ref_plane, ref_id) exists. */
int x264_gpu_frame_covers( const x264_gpu_frame_t *frame,
                           int list, const void *ref_plane, int64_t ref_id );

#ifdef __cplusplus
}
#endif

#endif
