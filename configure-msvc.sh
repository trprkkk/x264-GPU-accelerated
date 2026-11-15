#!/bin/bash
# Configure an MSVC build tree (build-msvc/) for the x264 fork and enable the
# optional CUDA motion-search module.  Run from anywhere: all paths are
# resolved relative to this script.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Bundled NASM: place nasm under toolchain/ and it takes precedence over PATH.
if [ -d "$ROOT/toolchain/nasm-2.16.03" ]; then
    export PATH="$ROOT/toolchain/nasm-2.16.03:$PATH"
fi

mkdir -p "$ROOT/build-msvc"
cd "$ROOT/build-msvc"
CC=cl ../x264/configure --host=x86_64-w64-mingw32 --prefix="$ROOT/dist"

# Enable the optional CUDA motion-search module.
#
# The CUDA backend is built from two toolkits:
#   * CUDA 11.8 (CUDA_ROOT) compiles the kernels for Kepler..Hopper
#     (sm_35..sm_90) and is also the runtime (cudart) linked into x264.exe,
#     so all drivers from R451 up keep working.
#   * a newer toolkit (CUDA_MODERN_ROOT) compiles the kernels for Blackwell
#     and newer (sm_100+); its runtime is not linked.
# The kernel images are embedded as resources and loaded through the driver
# API at runtime, so no CUDA DLL has to be shipped.
if ! grep -q '^HAVE_CUDA' config.mak; then
CUDA_ROOT="${CUDA_ROOT:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8}"
CUDA_MODERN_ROOT="${CUDA_MODERN_ROOT:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3}"
CUDA_ARCHS_LEGACY="${CUDA_ARCHS_LEGACY:-35 37 50 52 53 60 61 62 70 72 75 80 86 87 89 90}"
CUDA_PTX_LEGACY="${CUDA_PTX_LEGACY:-75}"
CUDA_ARCHS_MODERN="${CUDA_ARCHS_MODERN:-100 120}"
CUDA_PTX_MODERN="${CUDA_PTX_MODERN:-90}"
GEN_LEGACY=""
for a in $CUDA_ARCHS_LEGACY; do
    GEN_LEGACY="$GEN_LEGACY -gencode arch=compute_$a,code=sm_$a"
done
for a in $CUDA_PTX_LEGACY; do
    GEN_LEGACY="$GEN_LEGACY -gencode arch=compute_$a,code=compute_$a"
done
GEN_MODERN=""
for a in $CUDA_ARCHS_MODERN; do
    GEN_MODERN="$GEN_MODERN -gencode arch=compute_$a,code=sm_$a"
done
for a in $CUDA_PTX_MODERN; do
    GEN_MODERN="$GEN_MODERN -gencode arch=compute_$a,code=compute_$a"
done
cat >> config.h <<EOF
#define HAVE_CUDA 1
EOF
cat >> config.mak <<EOF

HAVE_CUDA=1
NVCC_LEGACY=${CUDA_ROOT}/bin/nvcc.exe
NVCC_MODERN=${CUDA_MODERN_ROOT}/bin/nvcc.exe
CUDAFLAGS_LEGACY=-O3 -std=c++14 -Wno-deprecated-gpu-targets -allow-unsupported-compiler -I\$(SRCPATH)/common/cuda_stub -I\$(SRCPATH)${GEN_LEGACY}
CUDAFLAGS_MODERN=-O3 -std=c++14 -I\$(SRCPATH)/common/cuda_stub -I\$(SRCPATH)${GEN_MODERN}
CUDALIBS=cudart_static.lib
OPENCL_INC=-I"${CUDA_ROOT}/include"
EOF
# Note: the CUDA headers (cuda_runtime.h for gpu_cuda.cpp) are picked up from
# INCLUDE, which make-msvc.sh appends the toolkit's include directory to; the
# toolkit's stdint.h must not shadow the compiler's.
fi
