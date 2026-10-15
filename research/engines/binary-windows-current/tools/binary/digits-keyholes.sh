#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:?usage: digits-keyholes.sh ENGINE_DIR RESULT_DIR}
RESULT=${2:?usage: digits-keyholes.sh ENGINE_DIR RESULT_DIR}
MODEL=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
REF=/workspace/binary/results
mkdir -p "$RESULT"
BASE=(env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=/root H5_WEIGHTS=packed1 H5_STATE=f16 H5_KV=f16 H5_DETERMINISTIC_GEMM=1 H5_ORDERED_PREP=off H5_A8_GU=off)
for act in a8 f32-r1-a8; do
  for project in dp4a fp4-digits; do
    for chunk in 32 64; do
      "${BASE[@]}" H5_ACT="$act" H5_A8_PROJECT="$project" H5_CHUNK="$chunk" "$ENGINE/h5" check "$MODEL" "$REF/wiki256.ref.bin" > "$RESULT/$act-$project-chunk$chunk-wiki.log" 2>&1
    done
    "${BASE[@]}" H5_ACT="$act" H5_A8_PROJECT="$project" H5_CHUNK=32 "$ENGINE/h5" check "$MODEL" "$REF/chat64.ref.bin" > "$RESULT/$act-$project-chat.log" 2>&1
  done
done
sha256sum "$ENGINE/h5" "$ENGINE/h5-server" > "$RESULT/executable-sha256.txt"
printf 'complete\n' > "$RESULT/status.txt"
