#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
"$root/build/halo-6000" forward-test --model /workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf --cache-dir /workspace/models/halo6000-cache --tokens-file /workspace/ternary-bench/oracle/teacher-v1/tokens.i32 --layer-output-dir "$root/docs/layer0-v2" --output-dir "$root/docs/forward-v1" --steps 16 --grid 188
python3 /workspace/ternary-bench/eval/compare_debug.py /workspace/ternary-bench/oracle/v1-final-debug-step "$root/docs/layer0-v2" "$root/docs/layer0-v2/phase-compare.json" --phase-only > "$root/docs/layer0-v2/phase-compare.log"
python3 /workspace/ternary-bench/eval/compare_debug.py /workspace/ternary-bench/oracle/v1-final-debug-step "$root/docs/forward-v1" "$root/docs/forward-v1/comparison.json" > "$root/docs/forward-v1/comparison.log"
