#!/usr/bin/env bash
set -euo pipefail
root=/workspace/halo-6000
out="$root/docs/hybrid-forward-b"
mkdir -p "$out"
model=/workspace/models/gguf/Qwen3.6-35B-A3B-ablate-nonexpert-q8-down-q2-PTQ1_0.gguf
"$root/build/halo-6000" forward-test --model "$model" --cache-dir /workspace/models/halo6000-cache --tokens-file /workspace/ternary-bench/oracle/teacher-v1/tokens.i32 --layer-output-dir "$out/layer0" --output-dir "$out/forward" --steps 16 --grid 188 --row-lanes 32
python3 /workspace/ternary-bench/eval/compare_debug.py /workspace/ternary-bench/oracle/hybrid-final-debug-step "$out/layer0" "$out/layer0-comparison.json" --phase-only > "$out/layer0-comparison.log"
python3 /workspace/ternary-bench/eval/compare_debug.py /workspace/ternary-bench/oracle/hybrid-final-debug-step "$out/forward" "$out/forward-comparison.json" > "$out/forward-comparison.log"
