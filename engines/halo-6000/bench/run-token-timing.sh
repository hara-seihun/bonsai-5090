#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
snapshot() { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; ps -eo pid,ppid,stat,pcpu,pmem,args; }
snapshot > "$root/docs/token-timing-context-before.txt"
sha256sum "$root/build/halo-6000" > "$root/docs/token-timing-binary.sha256"
: > "$root/docs/token-timing.jsonl"
for grid in 188 94 256 188; do
 "$root/build/halo-6000" bench --model /workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf --cache-dir /workspace/models/halo6000-cache --context 1024 --warm 40 --steps 128 --grid "$grid" | tee -a "$root/docs/token-timing.jsonl"
done
snapshot > "$root/docs/token-timing-context-after.txt"
