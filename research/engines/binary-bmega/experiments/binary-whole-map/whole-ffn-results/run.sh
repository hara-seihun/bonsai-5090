#!/usr/bin/env bash
set -euo pipefail
cd ~/work/clones/binary-spec-5090
E=experiments/binary-whole-map
R=/workspace/binary/digits-ragged/halo/$E
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
for rows in "$@"; do
  "${SSH[@]}" "mkdir -p '$R/whole-ffn-results'; flock -w 600 /tmp/binary-gpu.lock timeout 55 env H5_PDL=1 '$R/build/whole_ffn_bench' /workspace/binary/models/Bonsai-27B-Q1_0.gguf '$rows' '$R/whole-ffn-results/r$rows.jsonl'"
  scp -q -o BatchMode=yes -i ~/.ssh/id_ed25519 -P $POD_PORT "root@$POD_HOST:$R/whole-ffn-results/r$rows.jsonl" "$E/whole-ffn-results/"
done
