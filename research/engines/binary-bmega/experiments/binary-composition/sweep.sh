#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then printf 'usage: sweep.sh MODEL OUTPUT_DIR\n' >&2; exit 2; fi
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$2"
printf 'started %s\n' "$(date --iso-8601=seconds)" > "$2/started"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.sm --format=csv > "$2/device.csv"
for r in 1 8 16; do
  "$here/bench" "$1" "$r" "$2/r$r" > "$2/r$r.log" 2>&1
done
printf 'finished %s\n' "$(date --iso-8601=seconds)" > "$2/finished"
