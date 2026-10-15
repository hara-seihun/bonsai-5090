#!/usr/bin/env bash
set -euo pipefail
cd ~/work/clones/binary-spec-5090
E=experiments/binary-whole-map
R=/workspace/binary/digits-ragged/halo/$E
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
"${SSH[@]}" "flock -w 240 /tmp/binary-gpu.lock timeout 55 env H5_PDL=1 H5_PREP_CHUNK=0 H5_PREP_WIDE=0 '$R/build/whole_ffn_joint_control' /workspace/binary/models/Bonsai-27B-Q1_0.gguf '$R/whole-ffn-results/joint-r16-r32.jsonl'"
scp -q -o BatchMode=yes -i ~/.ssh/id_ed25519 -P $POD_PORT "root@$POD_HOST:$R/whole-ffn-results/joint-r16-r32.jsonl" "$E/whole-ffn-results/"
