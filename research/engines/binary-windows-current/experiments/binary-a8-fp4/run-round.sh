#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
"${SSH[@]}" "flock -w 600 /tmp/binary-gpu.lock timeout 45 bash -s" <<'REMOTE'
set -euo pipefail
B=/workspace/binary/a8-fp4
M=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
export H5_PDL=1
"$B/bench" "$M" witness 8 "$B/results/wf8-witness.jsonl" wf8
for kind in gu down head; do
  for r in 8 32; do
    "$B/bench" "$M" "$kind" "$r" "$B/results/wf8-$kind-r$r.jsonl" wf8
  done
done
REMOTE
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT \
  'root@$POD_HOST:/workspace/binary/a8-fp4/results/*.jsonl' results/
