#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: build.sh OUTPUT GGUF_SOURCE_ROOT' >&2
  echo 'GGUF_SOURCE_ROOT must contain src/gguf.{h,cpp}; NVCC must be CUDA 13.1' >&2
  exit 1
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$1
gguf_root=$2
nvcc=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
[[ -x "$nvcc" && -f "$gguf_root/src/gguf.h" && -f "$gguf_root/src/gguf.cpp" ]] || { echo 'missing CUDA 13.1 compiler or GGUF reader sources' >&2; exit 2; }
version=$("$nvcc" --version)
[[ "$version" == *'release 13.1,'* ]] || { echo 'compiler must be CUDA 13.1' >&2; exit 2; }
if [[ -n ${MMA_OBJECT+x} ]]; then
  [[ -f "$MMA_OBJECT" ]] || { echo 'MMA_OBJECT does not exist' >&2; exit 2; }
  mma="$MMA_OBJECT"
else
  mma="$here/batch_mma.cu"
  [[ -f "$mma" ]] || { echo 'batch_mma.cu is missing' >&2; exit 2; }
fi
exec "$nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v -I"$gguf_root/src" \
  "$here/bench.cu" "$mma" "$here/../binary-bitserial/bitserial.cu" \
  "$gguf_root/src/gguf.cpp" -o "$output"
