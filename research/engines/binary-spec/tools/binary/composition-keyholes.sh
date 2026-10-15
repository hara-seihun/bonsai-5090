#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:?usage: composition-keyholes.sh ENGINE_DIR RESULT_DIR}
RESULT=${2:?usage: composition-keyholes.sh ENGINE_DIR RESULT_DIR}
MODEL=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
REF=/workspace/binary/results
mkdir -p "$RESULT"
BASE=(env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=/root H5_WEIGHTS=packed1 H5_STATE=f16 H5_KV=f16 H5_DETERMINISTIC_GEMM=1 H5_ORDERED_PREP=off)
for act in a8 f32-r1-a8; do
  for fusion in off fused; do
    for chunk in 8 16; do
      "${BASE[@]}" H5_ACT="$act" H5_A8_GU="$fusion" H5_CHUNK="$chunk" "$ENGINE/h5" check "$MODEL" "$REF/wiki256.ref.bin" > "$RESULT/$act-$fusion-chunk$chunk-wiki.log" 2>&1
    done
    "${BASE[@]}" H5_ACT="$act" H5_A8_GU="$fusion" H5_CHUNK=16 "$ENGINE/h5" decodebench "$MODEL" "$REF/wiki256.ref.bin" 128 5 > "$RESULT/$act-$fusion-decodebench.json" 2> "$RESULT/$act-$fusion-decodebench.log"
  done
done
sha256sum "$ENGINE/h5" "$ENGINE/h5-server" > "$RESULT/executable-sha256.txt"
printf 'complete\n' > "$RESULT/status.txt"
