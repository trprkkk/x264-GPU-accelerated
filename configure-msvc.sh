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
# CUDA_ROOT: CUDA Toolkit location.
# CUDA_ARCH: nvcc -arch value; set it to your GPU (sm_75 = Turing, sm_86 = Ampere...).
if ! grep -q '^HAVE_CUDA' config.mak; then
CUDA_ROOT="${CUDA_ROOT:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3}"
CUDA_ARCH="${CUDA_ARCH:-sm_75}"
cat >> config.h <<EOF
#define HAVE_CUDA 1
EOF
cat >> config.mak <<EOF

HAVE_CUDA=1
NVCC=${CUDA_ROOT}/bin/nvcc.exe
CUDAFLAGS=-O3 -std=c++14 -arch=${CUDA_ARCH} -I\$(SRCPATH)
CUDALIBS=cudart_static.lib
OPENCL_INC=-I"${CUDA_ROOT}/include"
OPENCL_LIB="${CUDA_ROOT}/lib/x64/OpenCL.lib"
EOF
fi
