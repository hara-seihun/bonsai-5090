#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf}
mkdir -p build
${CXX:-c++} -std=c++17 -O2 -fopenmp -Wall -Wextra -Iinclude -Isrc \
 src/gguf.cpp src/cache.cpp bench/metadata-guard.cpp -o build/metadata-guard
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} build/metadata-guard "$MODEL"
