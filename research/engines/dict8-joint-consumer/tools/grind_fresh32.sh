#!/usr/bin/env bash
# Fresh32 native panel: original 1935316 server versus the grind build (carrier load fix,
# carveout 100, GU pre-wait 5), strict native7 in ABBA order, plus relaxed presets on the grind build.
# Runs on an exclusively held RTX 5090 pod prepared by grind_pod_setup.sh. Usage:
#   grind_fresh32.sh OLD_SERVER NEW_SERVER OUTDIR
set -euo pipefail
OLD=$1 NEW=$2 OUT=$3
CORPUS=/root/corpus
CORPUS_SHA=c9b9265ee4884428c74fdee8211a333b65086026dd4e7267b8edab79218978fe
mkdir -p "$OUT"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/apps-before.txt"
test ! -s "$OUT/apps-before.txt" || { echo GPU_NOT_EXCLUSIVE >&2; exit 3; }
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
core = json.loads(Path('results/integrated/selected-preset.json').read_text())
def env(mode, extra=None, drop=()):
    v = {**core['environment'], **core['modes'][mode]['environment']}
    for k in ('H5_COMPACT', 'H5_DRAFT_MARGIN', *drop): v.pop(k, None)
    v['H5_PLAN_READ'] = '/root/gemm-plan.txt'; v['H5_HEAD_ROWS'] = 'demand'
    v.update(extra or {})
    return v
(out / 'engine-new.json').write_text(json.dumps(env('strict-native'), indent=2) + '\n')
(out / 'engine-old.json').write_text(json.dumps(env('strict-native', drop=('H5_CARVEOUT', 'H5_GEMM_PREWAIT')), indent=2) + '\n')
PY
COMMON=(--corpus-manifest "$CORPUS/manifest.json" --corpus-sha256 "$CORPUS_SHA"
  --model /root/models/bonsai.gguf --drafter /root/models/model.safetensors
  --selection "$CORPUS/selection32.json" --prompts "$CORPUS/prompts32.json"
  --compact dense5 --port 18080 --start 0 --count 32
  --max-tokens 768 --warmup-tokens 128 --deadline-s 85 --request-timeout-s 45)
run() {
  local name=$1 server=$2 envf=$3; shift 3
  set +e
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
    python3 tools/registered_commit_panel.py "${COMMON[@]}" --server "$server" \
      --server-source-commit "$(cat "$server.source")" --engine-environment "$envf" \
      --output "$OUT/$name" "$@" > "$OUT/$name-exit.json" 2> "$OUT/$name-driver.stderr"
  local code=$?
  set -e
  printf '%s\t%s\n' "$name" "$code" >> "$OUT/exits.tsv"
}
run strict-old-a "$OLD" "$OUT/engine-old.json" --draft-proposals 7 --draft-margin 0 --role baseline --commit-policy fixed
run strict-new-a "$NEW" "$OUT/engine-new.json" --draft-proposals 7 --draft-margin 0 --role baseline --commit-policy fixed
run strict-new-b "$NEW" "$OUT/engine-new.json" --draft-proposals 7 --draft-margin 0 --role baseline --commit-policy fixed
run strict-old-b "$OLD" "$OUT/engine-old.json" --draft-proposals 7 --draft-margin 0 --role baseline --commit-policy fixed
run relaxed7-d4-new "$NEW" "$OUT/engine-new.json" --draft-proposals 7 --draft-margin 4 --role baseline --commit-policy fixed
run relaxed16-d4-new "$NEW" "$OUT/engine-new.json" --draft-proposals 16 --draft-margin 4 --role baseline --commit-policy fixed
date -u --iso-8601=seconds > "$OUT/finished-at.txt"
