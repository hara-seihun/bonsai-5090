#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH
HALO=${F1_HALO_SRC:-$PWD}
# CUDA12.8+ with sm_120a FP4 support. Compiler resource report is part of the result.
nvcc -O3 -std=c++17 -arch=sm_120a --threads 2 -Xptxas=-v \
  -Isrc -I"$HALO/src" bench/f1_bench.cu src/f1.cu src/f1_prep.cu \
  "$HALO/src/fp4.cu" "$HALO/src/kernels.cu" \
  -o bench/f1-bench
