#!/bin/bash
# Build the configured MSVC tree (run configure-msvc.sh first).
# Run from anywhere: all paths are resolved relative to this script.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Bundled NASM: place nasm under toolchain/ and it takes precedence over PATH.
if [ -d "$ROOT/toolchain/nasm-2.16.03" ]; then
    export PATH="$ROOT/toolchain/nasm-2.16.03:$PATH"
fi
# mingw32-make must be reachable; the common MinGW install location is added
# when it exists.
if [ -d /c/MinGW/bin ]; then
    export PATH="/c/MinGW/bin:$PATH"
fi

export CUDA_ROOT="${CUDA_ROOT:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8}"
export LIB="${CUDA_ROOT}/lib/x64;${LIB}"
# The CUDA headers must be searched *after* the system headers: the toolkit
# ships its own stdint.h, which would otherwise shadow the compiler's.
export INCLUDE="${INCLUDE};${CUDA_ROOT}/include"
cd "$ROOT/build-msvc"
mingw32-make -j8 "$@"
