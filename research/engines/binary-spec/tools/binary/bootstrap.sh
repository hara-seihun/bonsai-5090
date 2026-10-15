#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?usage: bootstrap.sh REMOTE_ROOT}
mkdir -p "$ROOT/models" "$ROOT/results"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build git curl python3 python3-venv python3-numpy g++ libssl-dev
cd "$ROOT/llama"
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_ALL_QUANTS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-bench llama-cli llama-perplexity llama-server mtmd
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv > "$ROOT/results/hardware.csv"
nvcc --version > "$ROOT/results/nvcc.txt"
