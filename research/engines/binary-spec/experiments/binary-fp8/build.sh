#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: build.sh OUTPUT GGUF_SOURCE_ROOT' >&2
  echo 'GGUF_SOURCE_ROOT must contain src/gguf.{h,cpp}; NVCC must be CUDA 13.1' >&2
  echo 'Optional FP8_OBJECT and BITSERIAL_OBJECT select existing compiled objects' >&2
  exit 1
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$1
gguf_root=$2
nvcc=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
[[ -x "$nvcc" && -f "$gguf_root/src/gguf.h" && -f "$gguf_root/src/gguf.cpp" ]] || { echo 'missing CUDA 13.1 compiler or GGUF reader sources' >&2; exit 2; }
version=$("$nvcc" --version)
[[ "$version" == *'release 13.1,'* ]] || { echo 'compiler must be CUDA 13.1' >&2; exit 2; }
if [[ -n ${FP8_OBJECT+x} ]]; then
  [[ -f "$FP8_OBJECT" ]] || { echo 'FP8_OBJECT does not exist' >&2; exit 2; }
  fp8="$FP8_OBJECT"
else
  fp8="$here/fp8.cu"
  [[ -f "$fp8" ]] || { echo 'fp8.cu is missing' >&2; exit 2; }
fi
if [[ -n ${BITSERIAL_OBJECT+x} ]]; then
  [[ -f "$BITSERIAL_OBJECT" ]] || { echo 'BITSERIAL_OBJECT does not exist' >&2; exit 2; }
  bitserial="$BITSERIAL_OBJECT"
else
  bitserial="$here/../binary-bitserial/bitserial.cu"
  [[ -f "$bitserial" ]] || { echo 'bitserial.cu is missing' >&2; exit 2; }
fi
exec "$nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v -I"$gguf_root/src" \
  "$here/bench.cu" "$fp8" "$bitserial" "$gguf_root/src/gguf.cpp" -o "$output"
