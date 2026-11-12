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

/* Candidate lookup used by me.c.  Returns NULL when this list/ref/MB has no
 * candidates (for example a 10-bit encode, an unsupported size, or a reference
 * index beyond the analysed set). */
const x264_gpu_cand_t *x264_gpu_frame_get( const x264_gpu_frame_t *frame,
                                           int list, int ref, int mbx, int mby,
                                           int *count );

#ifdef __cplusplus
}
#endif

#endif
