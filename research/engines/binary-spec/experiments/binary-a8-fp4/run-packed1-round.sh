#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
"${SSH[@]}" "flock -w 600 /tmp/binary-gpu.lock timeout 55 bash -s" <<'REMOTE'
set -euo pipefail
B=/workspace/binary/a8-fp4/packed1
M=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
export H5_PDL=1
for layout in wf8 packed1; do
  "$B/bench" "$M" witness 8 "$B/results/$layout-witness.jsonl" "$layout"
  for kind in gu down head; do
    for r in 8 32; do
      "$B/bench" "$M" "$kind" "$r" "$B/results/$layout-$kind-r$r.jsonl" "$layout"
    done
  done
done
REMOTE
mkdir -p results/packed1
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT \
  'root@$POD_HOST:/workspace/binary/a8-fp4/packed1/results/*.jsonl' results/packed1/
