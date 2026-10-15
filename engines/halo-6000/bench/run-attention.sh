#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v -Iinclude bench/attention.cu -o bench/attention 2>bench/attention-ptxas.log
exec bench/attention "$@"
