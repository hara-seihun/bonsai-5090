#!/usr/bin/env bash
set -euo pipefail
cd /workspace/binary/batch-map
./binary-batch-map-witness | tee results/native-witness.log
for tensor in gu blk.0.ffn_down.weight output.weight; do
  for r in 2 4 5 8 16 32; do
    ./binary-batch-map-bench /workspace/binary/models/Bonsai-27B-Q1_0.gguf "$tensor" "$r" 0 "results/${tensor}-r${r}"
  done
done
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.sm,memory.total --format=csv > results/device.txt
sha256sum binary-batch-map-bench binary-batch-map-witness > results/binary-sha256.txt
