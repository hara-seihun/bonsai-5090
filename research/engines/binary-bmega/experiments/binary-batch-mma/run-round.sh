#!/usr/bin/env bash
set -euo pipefail
cd /workspace/binary/batch-mma
mkdir -p results
model=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
./bench "$model" gu all 0 results/gu
./bench "$model" down all 0 results/down
./bench "$model" head all 65536 results/head65536
./bench "$model" head all 0 results/head248320
