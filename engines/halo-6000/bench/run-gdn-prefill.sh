#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
"$NVCC" -std=c++17 -O3 -arch=sm_120a --fmad=true -Iinclude bench/gdn-prefill.cu -o bench/halo-gdn-prefill
exec timeout 90 bench/halo-gdn-prefill "$@"
