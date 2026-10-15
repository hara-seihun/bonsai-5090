#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:?usage: native-sf-keyholes.sh ENGINE_DIR RESULT_DIR}
RESULT=${2:?usage: native-sf-keyholes.sh ENGINE_DIR RESULT_DIR}
MODEL=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
REF=/workspace/binary/results
mkdir -p "$RESULT"
BASE=(env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=/root H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 H5_FP4_PASSES=2 H5_FP4_PAIR=joint-r16 H5_STATE=f16 H5_KV=f16 H5_DETERMINISTIC_GEMM=1 H5_ORDERED_PREP=off)
for native in off sf1-r32 sf2-r32; do
  for chunk in 32 64; do
    "${BASE[@]}" H5_NATIVE_SF="$native" H5_CHUNK="$chunk" "$ENGINE/h5" check "$MODEL" "$REF/wiki256.ref.bin" > "$RESULT/$native-chunk$chunk-wiki.log" 2>&1
  done
  "${BASE[@]}" H5_NATIVE_SF="$native" H5_CHUNK=32 "$ENGINE/h5" check "$MODEL" "$REF/chat64.ref.bin" > "$RESULT/$native-chat.log" 2>&1
done
sha256sum "$ENGINE/h5" "$ENGINE/h5-server" > "$RESULT/executable-sha256.txt"
printf 'complete\n' > "$RESULT/status.txt"
