#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$root/docs"
snapshot() { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; ps -eo pid,ppid,stat,pcpu,pmem,args; }
snapshot > "$root/docs/barrier-context-before.txt"
"$root/bench/barrier" > "$root/docs/barrier-quiet.json.tmp"
mv "$root/docs/barrier-quiet.json.tmp" "$root/docs/barrier-quiet.json"
snapshot > "$root/docs/barrier-context-after.txt"
