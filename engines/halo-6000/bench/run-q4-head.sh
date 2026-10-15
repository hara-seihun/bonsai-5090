#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
F=/workspace/src/prism-llama
sha256sum include/q4_0.cuh include/q4_0-format.hpp include/device.cuh include/model.hpp \
  bench/q4-head.cu "$F/ggml/src/ggml-cuda/vecdotq.cuh" "$F/ggml/src/ggml-cuda/common.cuh" \
  "$F/ggml/src/ggml-cuda/quantize.cu" > bench/q4-source.sha256
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
  -Iinclude -I"$F/ggml/include" -I"$F/ggml/src" -I"$F/ggml/src/ggml-cuda" -I"$F/include" \
  bench/q4-head.cu -o bench/q4-head 2>bench/q4-ptxas.log
exec flock /workspace/halo-6000/gpu.lock bash -c '
 set -e
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv
 bench/q4-head "$1" "$2"
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv
' -- "${1:-/workspace/halo-6000/bench/q4-fixture}" "${2:-/workspace/halo-6000/bench/head-fixture/head-input.f32}"
