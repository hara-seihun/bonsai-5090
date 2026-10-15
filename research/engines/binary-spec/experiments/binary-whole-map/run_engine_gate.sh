#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/binary/whole-map
CONTROL=/workspace/binary/digits-ragged/halo/h5
MODEL=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
REF=/workspace/binary/results/wiki256.ref.bin
COMMON=(env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=/root H5_WEIGHTS=packed1 H5_ACT=f32-r1-a8 H5_STATE=f16 H5_KV=f16 H5_DETERMINISTIC_GEMM=1 H5_ORDERED_PREP=off H5_A8_GU=off H5_A8_PROJECT=fp4-digits-ragged H5_CHUNK=32)
case "${1:?quality or fixed}" in
quality)
  for arm in four three; do
    BIN=$CONTROL; [[ $arm == three ]] && BIN=$ROOT/three-h5
    "${COMMON[@]}" "$BIN" check "$MODEL" "$REF" > "$ROOT/engine-$arm-wiki.log" 2>&1
  done
  ;;
fixed)
  for rows in 16 32; do
    n=0
    for arm in four three three four; do
      "${COMMON[@]}" "$ROOT/fixed-$arm" "$MODEL" "$rows" > "$ROOT/fixed-$rows-$n-$arm.log" 2>&1
      n=$((n+1))
    done
  done
  ;;
*) echo 'unsupported gate' >&2; exit 2 ;;
esac
