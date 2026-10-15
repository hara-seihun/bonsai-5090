#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
mkdir -p build/matmul-fixture bench/matmul-results
sha256sum include/{matmul.cuh,matmul.hpp,model.hpp,device.cuh} bench/{matmul.cu,matmul-fixture.py,matmul-pod.cpp,matmul-link-a.cu,matmul-link-b.cu} > bench/matmul-source.sha256
g++ -std=c++17 -O2 -Iinclude bench/matmul-pod.cpp -o build/matmul-pod
./build/matmul-pod > bench/matmul-pod.json
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120 -Iinclude bench/matmul-link-a.cu bench/matmul-link-b.cu -o build/matmul-link
./build/matmul-link
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -Iinclude -Xptxas=-v bench/matmul.cu -o build/matmul 2>bench/matmul-ptxas.log
python3 bench/matmul-fixture.py /workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf /workspace/models/gguf/Qwen3.6-35B-A3B-ablate-nonexpert-q8-down-q2-PTQ1_0.gguf /workspace/halo-6000/build/matmul-fixture
# The flock's process tree owns the GPU through the complete receipt-bearing run.
flock /workspace/halo-6000/gpu.lock bash -c '
 set -euo pipefail
 nvidia-smi -q > bench/matmul-context.txt
 ps -eo pid,ppid,pgid,args > bench/matmul-processes.txt
 while IFS= read -r key; do
  [ -n "$key" ] || continue
  ./build/matmul build/matmul-fixture "$key" > "bench/matmul-results/$key.jsonl"
 done < <(cut -f1 build/matmul-fixture/cases.tsv)
 nvidia-smi -q > bench/matmul-context-after.txt
'
