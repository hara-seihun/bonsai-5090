#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
  -Iinclude bench/head.cu -o bench/head 2>bench/head-ptxas.log
exec flock /workspace/halo-6000/gpu.lock bash -c '
  nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv
  set -e
  bench/head "$1"
  nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv
' -- "${1:-/workspace/halo-6000/bench/head-fixture}"
