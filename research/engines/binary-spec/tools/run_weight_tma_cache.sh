#!/usr/bin/env bash
set -euo pipefail
if (( $# != 7 )); then
  echo 'usage: run_weight_tma_cache.sh BUILD MODEL PLAN ORDINARY_EVENTS VERIFY_EVENTS NEW_OUTPUT none|normal|all' >&2
  exit 2
fi
build=$(realpath "$1"); model=$(realpath "$2"); plan=$(realpath "$3")
ordinary=$(realpath "$4"); verify=$(realpath "$5"); out=$(realpath -m "$6")
case "$7" in
  none) controls=(none);;
  normal) controls=(normal);;
  all) controls=(none normal);;
  *) echo 'invalid contrast' >&2; exit 2;;
esac
for control in "${controls[@]}" first; do
  [[ -x "$build/$control/profile-replay" ]] || { echo "missing binary: $control" >&2; exit 2; }
done
mkdir "$out"
for key in "${!H5_@}"; do unset "$key"; done
export H5_COMPACT=dense5 H5_STATE=f16 H5_KV=fp8 H5_WEIGHTS=fp4 H5_ACT=fp4 H5_PDL=1
export H5_STATE_FUSION=0 H5_GDN_CG=8 H5_PREP_G4=0.015625 H5_ATTN_DIRECT=0 H5_PREP_CHUNK=2 H5_PREP_CLEAR=1 H5_SCALE_CARRIER=1 H5_PLAN_READ="$plan"
env | grep '^H5_' | sort > "$out/environment.txt"
sha256sum "$build"/{none,normal,first}/profile-replay "$plan" "$ordinary" "$verify" > "$out/provenance.txt"
cp "$build/source-commit.txt" "$build/source-diff.patch" "$out/"
for work in ordinary verify; do
  events=$verify; extra=()
  if [[ "$work" == ordinary ]]; then events=$ordinary; extra=(--ordinary); fi
  for control in "${controls[@]}"; do
    slot=0
    for arm in "$control" first first "$control"; do
      slot=$((slot+1)); name="$out/$work-$control-$slot-$arm"
      command=("$build/$arm/profile-replay" "$model" "$events" "$name" "${extra[@]}" --graph-upload launch --graph-timing off --graph-nodes kernel-memory --passes 1)
      printf '%q ' "${command[@]}" > "$name.command"; printf '\n' >> "$name.command"
      timeout --signal=TERM --kill-after=5s 60s "${command[@]}" > "$name.log" 2>&1
    done
  done
done
printf 'complete\n' > "$out/state.txt"
