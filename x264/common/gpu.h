/*****************************************************************************
 * gpu.h: CUDA candidate generation for the hybrid x264 fork
 *****************************************************************************
 * This module never makes the final coding decision.  It returns a small
 * list of integer-pel motion vectors per macroblock; the unmodified x264
 * rate-distortion code evaluates them against the real reconstructed
 * reference and remains the only authority on the output bitstream.
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

typedef struct
{
    int16_t  mv[2];   /* full-pel, relative to the block origin */
    uint32_t sad;     /* integer SAD that produced the candidate */
} x264_gpu_cand_t;

typedef struct x264_gpu_frame_t x264_gpu_frame_t;

/* Returns NULL when CUDA is unavailable or initialisation fails. */
void *x264_gpu_create( int width, int height, int max_refs, int device );
void  x264_gpu_destroy( void *ctx );
const char *x264_gpu_name( void );

x264_gpu_frame_t *x264_gpu_frame_alloc( int mbw, int mbh, int nref0, int nref1 );
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
