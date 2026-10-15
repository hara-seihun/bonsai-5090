#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
sha256sum include/matvec.cuh include/model.hpp include/device.cuh include/q2k.cuh bench/row-exact.cu > bench/row-source.sha256
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
  -Iinclude bench/row-exact.cu -o bench/row-exact 2>bench/row-ptxas.log
exec flock /workspace/halo-6000/gpu.lock bash -c '
 set -e
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv
 bench/row-exact "$1"
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv
' -- "${1:-/workspace/halo-6000/bench/row-fixture}"
