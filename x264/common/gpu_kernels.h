/*****************************************************************************
 * gpu_kernels.h: definitions shared by the CUDA kernels and their host
 *****************************************************************************
 * The kernels live in gpu_kernels.cu and are compiled to a fat binary by nvcc
 * (no host code).  gpu_cuda.cpp embeds those fat binaries, loads the right
 * one at runtime through the CUDA driver API and launches the kernels.
 *
 * Keeping the device code in its own translation unit decouples it from the
 * host compiler: the kernels can be built with a toolkit that still knows
 * pre-Turing architectures (CUDA 11.8) while the host side is compiled by the
 * regular C++ compiler of the build.
 *****************************************************************************/

#ifndef X264_GPU_KERNELS_H
#define X264_GPU_KERNELS_H

#include "gpu.h"

#define GK X264_GPU_CANDS

/* Same layout as the public candidate type. */
typedef x264_gpu_cand_t Cand;

#endif
