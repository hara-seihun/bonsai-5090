#!/usr/bin/env bash
# Fresh32 strict native7 server panel, ABBA: OLD server with the selected preset minus NEW_KEYS versus
# NEW server with the full selected preset. Run from the NEW source tree on an exclusively held 5090.
#   grind_fresh32_ab.sh OLD_SERVER NEW_SERVER OUTDIR NEW_KEY...
# A NEW_KEY written KEY=OLDVALUE keeps KEY in the old arm with OLDVALUE (for required keys).
# Each server has SERVER.source (commit). Corpus in /root/corpus, plan /root/gemm-plan.txt.
set -euo pipefail
OLD=$1 NEW=$2 OUT=$3; shift 3
CORPUS=/root/corpus
CORPUS_SHA=c9b9265ee4884428c74fdee8211a333b65086026dd4e7267b8edab79218978fe
mkdir -p "$OUT"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/apps-before.txt"
test ! -s "$OUT/apps-before.txt" || { echo GPU_NOT_EXCLUSIVE >&2; exit 3; }
python3 - "$OUT" "$@" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); new_keys = sys.argv[2:]
core = json.loads(Path('results/integrated/selected-preset.json').read_text())
def env(drop=()):
    v = {**core['environment'], **core['modes']['strict-native']['environment']}
    for k in ('H5_COMPACT', 'H5_DRAFT_MARGIN', *drop): v.pop(k, None)
    v['H5_PLAN_READ'] = '/root/gemm-plan.txt'; v['H5_HEAD_ROWS'] = 'demand'
    return v
keys = [k.split('=', 1)[0] for k in new_keys]
old_values = dict(k.split('=', 1) for k in new_keys if '=' in k)
for k in keys:
    if k not in core['environment']: raise SystemExit(f'{k} is not in the selected preset environment')
(out / 'engine-new.json').write_text(json.dumps(env(), indent=2) + '\n')
(out / 'engine-old.json').write_text(json.dumps({**env(keys), **old_values}, indent=2) + '\n')
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
S=(--draft-proposals 7 --draft-margin 0 --role baseline --commit-policy fixed)
run strict-old-a "$OLD" "$OUT/engine-old.json" "${S[@]}"
run strict-new-a "$NEW" "$OUT/engine-new.json" "${S[@]}"
run strict-new-b "$NEW" "$OUT/engine-new.json" "${S[@]}"
run strict-old-b "$OLD" "$OUT/engine-old.json" "${S[@]}"
date -u --iso-8601=seconds > "$OUT/finished-at.txt"
