#!/usr/bin/env bash
set -euo pipefail
exec 9>/root/h5-gpu-measure.lock
flock -n 9 || { echo GPU_LOCK_BUSY >&2; exit 2; }
RUNTIME=/root/halo-cap7-runtime
OUT="$PWD/results/cap7-panel"
mkdir -p "$OUT"
exec > "$OUT/panel.log" 2>&1
ENGINE_SOURCE=$(<"$RUNTIME/engine-source.txt")
printf 'runner_commit=%s\nengine_source=%s\n' "$HALO_COMMIT" "$ENGINE_SOURCE"
date -u --iso-8601=seconds
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/apps-before.txt"
test ! -s "$OUT/apps-before.txt" || { echo GPU_NOT_EXCLUSIVE >&2; exit 3; }
sha256sum --check "$RUNTIME/server.sha256"
COMMON=(--corpus-manifest "$RUNTIME/corpus/manifest.json" --corpus-sha256 "$CAP7_CORPUS_SHA"
  --server "$RUNTIME/h5-server" --server-source-commit "$ENGINE_SOURCE"
  --model /root/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --drafter /root/models/model.safetensors
  --selection "$RUNTIME/corpus/selection32.json" --prompts "$RUNTIME/corpus/prompts32.json"
  --compact dense5 --draft-proposals 7 --port 18080 --start 0 --count 32
  --max-tokens 768 --warmup-tokens 128 --deadline-s 85 --request-timeout-s 45)
run_panel() {
  local name="$1"; shift
  set +e
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
    python3 tools/registered_commit_panel.py "${COMMON[@]}" --output "$OUT/$name" "$@" \
    > "$OUT/$name-exit.json" 2> "$OUT/$name-driver.stderr"
  local code=$?
  set -e
  printf '%s\t%s\n' "$name" "$code" >> "$OUT/exits.tsv"
  return "$code"
}
# Recovered demand route measured against its explicit full-head control.
if run_panel strict-native7-head-all --engine-environment "$RUNTIME/engine-all.json" --draft-margin 0 --role baseline --commit-policy fixed; then
  echo head_all_control_complete
else
  echo head_all_control_failed_retained
fi
# Distinct explicit serving-mode baselines, not interchangeable pairing fingerprints.
if run_panel strict-native7 --engine-environment "$RUNTIME/engine.json" --draft-margin 0 --role baseline --commit-policy fixed; then
  echo strict_native_complete
else
  echo strict_native_failed_retained
fi
if run_panel strict-tree7 --engine-environment "$RUNTIME/engine-tree.json" --draft-margin 0 --role baseline --commit-policy fixed; then
  echo strict_tree_complete
else
  echo strict_tree_failed_retained
fi
run_panel relaxed-fixed7 --engine-environment "$RUNTIME/engine.json" --draft-margin 16 --role baseline --commit-policy fixed
for entry in cap4-a:4 cap7-a:7 cap7-b:7 cap4-b:4; do
  name="${entry%:*}"; cap="${entry#*:}"
  if run_panel "$name" --engine-environment "$RUNTIME/engine.json" --draft-margin 16 \
    --role candidate --baseline "$OUT/relaxed-fixed7" --commit-policy confidence \
    --commit-min 0 --commit-max "$cap" --commit-confidence 1; then
    printf 'arm_complete:%s\n' "$name"
  else
    printf 'arm_failed_retained:%s\n' "$name"
  fi
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/apps-after.txt"
test ! -s "$OUT/apps-after.txt" || { echo GPU_CHILD_REMAINS >&2; exit 4; }
sha256sum "$RUNTIME/h5-server" "$RUNTIME"/engine*.json "$RUNTIME/gemm-plan.txt" > "$OUT/final-inputs.sha256"
date -u --iso-8601=seconds > "$OUT/finished-at.txt"
printf 'complete_with_all_arm_exit_receipts\n' > "$OUT/state.txt"
