#!/usr/bin/env bash
set -euo pipefail
cd ~/work/clones/binary-spec-5090
E=experiments/binary-whole-map
R=/workspace/binary/digits-ragged/halo/$E
ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST "flock -w 600 /tmp/binary-gpu.lock timeout 55 env H5_PDL=1 '$R/build/whole_ffn_controls_r5' /workspace/binary/models/Bonsai-27B-Q1_0.gguf 5 '$R/whole-ffn-results/controls-r5.jsonl'"
scp -q -o BatchMode=yes -i ~/.ssh/id_ed25519 -P $POD_PORT "root@$POD_HOST:$R/whole-ffn-results/controls-r5.jsonl" "$E/whole-ffn-results/"
