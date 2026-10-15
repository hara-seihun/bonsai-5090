#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:?usage: joint-keyholes.sh ENGINE_DIR RESULT_DIR}
RESULT=${2:?usage: joint-keyholes.sh ENGINE_DIR RESULT_DIR}
MODEL=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
REF=/workspace/binary/results
mkdir -p "$RESULT"
BASE=(env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=/root H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 H5_FP4_PASSES=2 H5_STATE=f16 H5_KV=f16 H5_DETERMINISTIC_GEMM=1 H5_ORDERED_PREP=off)
for pair in off joint-r16; do
  for chunk in 8 16; do
    "${BASE[@]}" H5_FP4_PAIR="$pair" H5_CHUNK="$chunk" "$ENGINE/h5" check "$MODEL" "$REF/wiki256.ref.bin" > "$RESULT/$pair-chunk$chunk-wiki.log" 2>&1
  done
  "${BASE[@]}" H5_FP4_PAIR="$pair" H5_CHUNK=16 "$ENGINE/h5" check "$MODEL" "$REF/chat64.ref.bin" > "$RESULT/$pair-chat.log" 2>&1
  "${BASE[@]}" H5_FP4_PAIR="$pair" H5_CHUNK=16 "$ENGINE/h5" decodebench "$MODEL" "$REF/wiki256.ref.bin" 128 5 > "$RESULT/$pair-decodebench.json" 2> "$RESULT/$pair-decodebench.log"
  "${BASE[@]}" H5_FP4_PAIR="$pair" H5_CHUNK=16 "$ENGINE/h5" gen "$MODEL" 'Explain why a rainbow is a circle, in three concise sentences.' 128 --chat > "$RESULT/$pair-generated.txt" 2> "$RESULT/$pair-generated.log"
done
sha256sum "$ENGINE/h5" "$ENGINE/h5-server" > "$RESULT/executable-sha256.txt"
printf 'complete\n' > "$RESULT/status.txt"
