#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/dense5
mkdir -p "$OUT" build/dense5
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
MODEL=${H5_MODEL:?Set H5_MODEL to the original Bonsai27B PTQ1_0 GGUF}
python3 experiments/dense5/generate.py
g++ -O3 -std=c++17 -Iexperiments/dense5 experiments/dense5/host.cpp -o build/dense5/host
build/dense5/host | tee "$OUT/host.jsonl"
FLAGS=(-O3 -std=c++17 -arch=sm_120a -lineinfo -Xcompiler -fPIC,-O3 --expt-relaxed-constexpr -Isrc -Iexperiments/dense5 -Xptxas -v)
# Compilation is CPU-only. At most four compiler threads; sequential translation units.
"$NVCC" "${FLAGS[@]}" --threads 4 -c build/dense5/control.cu -o build/dense5/control.o 2>"$OUT/control-build.log"
"$NVCC" "${FLAGS[@]}" --threads 4 -c build/dense5/candidate.cu -o build/dense5/candidate.o 2>"$OUT/candidate-build.log"
"$NVCC" "${FLAGS[@]}" -c src/kernels.cu -o build/dense5/kernels.o 2>"$OUT/kernels-build.log"
"$NVCC" "${FLAGS[@]}" experiments/dense5/native.cu src/gguf.cpp build/dense5/control.o build/dense5/candidate.o build/dense5/kernels.o -o build/dense5/native 2>"$OUT/native-build.log"
"${NVCC%/nvcc}/cuobjdump" --dump-sass build/dense5/native >"$OUT/native.sass"
if [[ ${BUILD_ONLY:-0} == 1 ]]; then exit 0; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >"$OUT/hardware.csv"
sha256sum experiments/dense5/input.bin >"$OUT/input-sha256.txt"
timeout 55 build/dense5/native "$MODEL" experiments/dense5/input.bin | tee "$OUT/native.jsonl"
python3 experiments/dense5/summarize.py
