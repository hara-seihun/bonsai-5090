#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-docs/results/router-candidate}
mkdir -p "$OUT" build
for f in gguf cache; do
 if [[ ! -f build/router-$f.o || src/$f.cpp -nt build/router-$f.o || include/format.hpp -nt build/router-$f.o || include/model.hpp -nt build/router-$f.o ]]; then
  ${CXX:-c++} -std=c++17 -O2 -fopenmp -Iinclude -Isrc -c src/$f.cpp -o build/router-$f.o
 fi
done
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v -Iinclude -Isrc \
 bench/router-candidate.cu build/router-gguf.o build/router-cache.o -Xcompiler=-fopenmp -o build/router-candidate 2> "$OUT/ptxas.log"
sha256sum include/router.cuh include/matvec.cuh include/moe.cuh bench/router-candidate.cu bench/row-fixture/router.inputs.f32 > "$OUT/source.sha256"
exec flock /workspace/halo-6000/gpu.lock bash -c '
 set -euo pipefail
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv > "$1/gpu-before.csv"
 OMP_NUM_THREADS=8 build/router-candidate /workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf /workspace/models/gguf/Qwen3.6-35B-A3B-ablate-nonexpert-q8-down-q2-PTQ1_0.gguf bench/row-fixture/router.inputs.f32 | tee "$1/results.jsonl"
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv > "$1/gpu-after.csv"
' -- "$OUT"
