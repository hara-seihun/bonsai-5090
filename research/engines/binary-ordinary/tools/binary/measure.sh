#!/usr/bin/env bash
set -euo pipefail
if [ "$#" != 2 ]; then echo 'usage: measure.sh REMOTE_ROOT int8|fp4|binary1|packed1' >&2; exit 2; fi
ROOT=$1
MODE=$2
M="$ROOT/models/Bonsai-27B-Q1_0.gguf"
R="$ROOT/results"
export H5_STATE=f32 H5_KV=f16 H5_LOAD_THREADS=4
unset H5_COMPACT H5_WSCALE H5_MEGA H5_DRAFT_MARGIN H5_FP4_PASSES H5_CHUNK
case "$MODE" in
  int8) export H5_WEIGHTS=int8 H5_ACT=int8 ;;
  fp4) export H5_WEIGHTS=fp4 H5_ACT=fp4 ;;
  binary1) export H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 ;;
  fp4-two) export H5_WEIGHTS=fp4 H5_ACT=fp4 H5_FP4_PASSES=2 ;;
  binary1-two) export H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 H5_FP4_PASSES=2 ;;
  packed1) export H5_WEIGHTS=packed1 H5_ACT=f32 ;;
  packed1-f16) export H5_WEIGHTS=packed1 H5_ACT=f32 H5_STATE=f16 ;;
  binary1-two-f16) export H5_WEIGHTS=fp4 H5_ACT=fp4 H5_COMPACT=binary1 H5_FP4_PASSES=2 H5_STATE=f16 ;;
  *) echo "MODE_INVALID: $MODE" >&2; exit 2 ;;
esac
cd "$ROOT/halo"
printf 'H5_WEIGHTS=%s H5_ACT=%s H5_STATE=%s H5_KV=%s H5_COMPACT=%s\n' "$H5_WEIGHTS" "$H5_ACT" "$H5_STATE" "$H5_KV" "${H5_COMPACT-unset}" > "$R/$MODE-env.txt"
printf 'H5_FP4_PASSES=%s\n' "${H5_FP4_PASSES-unset}" >> "$R/$MODE-env.txt"
./h5 check "$M" "$R/chat64.ref.bin" > "$R/$MODE-chat-quality.log" 2>&1
./h5 check "$M" "$R/wiki256.ref.bin" > "$R/$MODE-wiki-quality.log" 2>&1
./h5 decodebench "$M" "$R/wiki256.ref.bin" 128 5 > "$R/$MODE-decodebench.json" 2> "$R/$MODE-decodebench.log"
./h5 gen "$M" 'Explain why a rainbow is a circle, in three concise sentences.' 128 --chat > "$R/$MODE-generated-chat.txt" 2> "$R/$MODE-generated-chat.log"
case "$MODE" in packed1|packed1-f16) STREAMS='1 8' ;; *) STREAMS='1 16 64' ;; esac
for S in $STREAMS; do
  ./h5 bench "$M" "$S" 128 30 > "$R/$MODE-synthetic-s$S-c128.log" 2>&1
done
./h5 bench "$M" 1 4096 30 > "$R/$MODE-synthetic-s1-c4096.log" 2>&1
