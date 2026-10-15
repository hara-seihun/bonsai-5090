#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
  -Iinclude bench/rotation.cu -o bench/rotation 2>bench/rotation-ptxas.log
exec bench/rotation "$@"
