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
# CUDA_ROOT:  CUDA Toolkit location.
# CUDA_ARCHS: native SASS images to embed (default: Turing..Blackwell).
#             CUDA Toolkit 13 can only build sm_75 and newer; older GPUs
#             (Kepler..Volta) are served by the OpenCL backend instead.
# CUDA_PTX:   virtual architecture(s) for forward compatibility (JIT).
if ! grep -q '^HAVE_CUDA' config.mak; then
CUDA_ROOT="${CUDA_ROOT:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3}"
CUDA_ARCHS="${CUDA_ARCHS:-75 80 86 87 89 90 100 120}"
CUDA_PTX="${CUDA_PTX:-75}"
ARCH_FLAGS=""
for a in $CUDA_ARCHS; do
    ARCH_FLAGS="$ARCH_FLAGS -gencode arch=compute_$a,code=sm_$a"
done
for a in $CUDA_PTX; do
    ARCH_FLAGS="$ARCH_FLAGS -gencode arch=compute_$a,code=compute_$a"
done
cat >> config.h <<EOF
#define HAVE_CUDA 1
EOF
cat >> config.mak <<EOF

HAVE_CUDA=1
NVCC=${CUDA_ROOT}/bin/nvcc.exe
CUDAFLAGS=-O3 -std=c++14${ARCH_FLAGS} -I\$(SRCPATH)
CUDALIBS=cudart_static.lib
OPENCL_INC=-I"${CUDA_ROOT}/include"
EOF
fi
