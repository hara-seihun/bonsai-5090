#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=EAGER
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
out=${1:?output directory required}
mkdir -p "$out"
snapshot() { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; ps -eo pid,ppid,stat,pcpu,pmem,args; }
snapshot > "$out/context-before.txt"
sha256sum "$root/build/halo-6000" > "$out/binary.sha256"
model=/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf
common=(--model "$model" --cache-dir /workspace/models/halo6000-cache --warm 40)
"$root/build/halo-6000" profile "${common[@]}" --grid 188 > "$out/profile.json"
python3 "$root/bench/analyze-profile.py" "$out/profile.json" --output "$out/profile-analysis.json"
for grid in 188 376 188; do
 "$root/build/halo-6000" bench "${common[@]}" --grid "$grid" --steps 128 >> "$out/timing.jsonl"
 snapshot > "$out/context-after-grid-${grid}-$(date +%s).txt"
done
snapshot > "$out/context-after.txt"
