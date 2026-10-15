#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
"${SSH[@]}" "flock -w 600 /tmp/binary-gpu.lock timeout 55 bash -s" <<'REMOTE'
set -euo pipefail
B=/workspace/binary/a8-fp4/ragged
M=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
export H5_PDL=1
for kind in gu down head; do
  for r in 16 24 31; do
    "$B/bench" "$M" "$kind" "$r" "$B/results/$kind-r$r.jsonl" packed1
  done
done
REMOTE
mkdir -p results/ragged
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT \
  'root@$POD_HOST:/workspace/binary/a8-fp4/ragged/results/*.jsonl' results/ragged/
