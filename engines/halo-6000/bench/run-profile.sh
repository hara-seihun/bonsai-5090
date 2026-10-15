#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
snapshot() { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; ps -eo pid,ppid,stat,pcpu,pmem,args; }
snapshot > "$root/docs/profile-context-before.txt"
sha256sum "$root/build/halo-6000" > "$root/docs/profile-binary.sha256"
"$root/build/halo-6000" profile --model /workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf --cache-dir /workspace/models/halo6000-cache --warm 40 --grid 188 > "$root/docs/profile.json"
python3 "$root/bench/analyze-profile.py" "$root/docs/profile.json" --output "$root/docs/profile-analysis.json"
snapshot > "$root/docs/profile-context-after.txt"
