#!/usr/bin/env bash
set -euo pipefail
if [ "$#" != 5 ]; then
  echo 'usage: launch.sh OFFER_ID PRISM_SOURCE_DIR MODEL_DATA_DIR PCR2_GPTQ2_FILE VOCAB_65536_FILE' >&2
  exit 2
fi
OFFER=$1
PRISM=$2
DATA=$3
DRAFTER=$4
VOCAB=$5
HERE=$(cd "$(dirname "$0")" && pwd)
HALO=$(cd "$HERE/../.." && pwd)
VAST=vastai
KEY=~/.ssh/id_ed25519
PIN=1a07bfa5f4144274c8f1c9963821dd9d9a51854b
MODEL=Bonsai-27B-Q1_0.gguf
SHA=17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0
[ "$(git -C "$PRISM" rev-parse HEAD)" = "$PIN" ] || { echo 'REFERENCE_PIN_MISMATCH' >&2; exit 1; }
[ -f "$DATA/$MODEL" ] || { echo 'MODEL_MISSING' >&2; exit 1; }
printf '%s  %s\n' "$SHA" "$DATA/$MODEL" | sha256sum -c -
DRAFT_SHA=bf5496a7153408c7ac4ed3e7213c6b1482f1fbed4b12ed81b6f1b4d398925692
printf '%s  %s\n' "$DRAFT_SHA" "$DRAFTER" | sha256sum -c -
[ -f "$VOCAB" ] || { echo 'VOCAB_MISSING' >&2; exit 1; }
python3 - "$VOCAB" <<'PY'
import sys
from pathlib import Path
values = [int(x) for x in Path(sys.argv[1]).read_text().split()]
if len(values) != 65536 or len(set(values)) != 65536 or min(values) < 0 or max(values) >= 248320:
    raise SystemExit('VOCAB_INVALID')
PY
SOURCE=$(git -C "$HALO" rev-parse HEAD)
CREATE=$($VAST create "$OFFER" --image nvidia/cuda:13.1.1-devel-ubuntu24.04 --disk 80 --label binary-bonsai-5090)
printf '%s\n' "$CREATE"
ID=$(printf '%s\n' "$CREATE" | awk '$1 == "instance" && $3 == "created" {print $2}')
[ -n "$ID" ] || { echo 'CREATE_RECEIPT_INVALID' >&2; exit 1; }
cleanup_error() { echo "launch failed; destroying $ID" >&2; "$VAST" destroy "$ID"; }
trap cleanup_error ERR
for attempt in $(seq 1 60); do
  read -r HOST PORT < <("$VAST" show "$ID" --json | python3 -c 'import json,sys; i=json.load(sys.stdin); p=i.get("ports",{}).get("22/tcp",[]); print(i.get("public_ipaddr","unset"),p[0]["HostPort"] if p else i.get("machine_dir_ssh_port",0))')
  if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$PORT" "root@$HOST" true 2>/dev/null; then break; fi
  if [ "$attempt" = 60 ]; then echo 'SSH_NOT_READY' >&2; false; fi
  sleep 5
done
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p "$PORT" "root@$HOST")
"${SSH[@]}" 'mkdir -p /workspace/binary/{llama,halo,models,results}'
tar -C "$PRISM" --exclude=./.git --exclude=./build-hip --exclude=./models --exclude=./hip-env.sh -cf - . | "${SSH[@]}" 'tar -C /workspace/binary/llama -xf -'
git -C "$HALO" archive "$SOURCE" | "${SSH[@]}" 'tar -C /workspace/binary/halo -xf -'
"${SSH[@]}" "printf '%s\\n' '$SOURCE' > /workspace/binary/source.txt"
scp -i "$KEY" -P "$PORT" "$DATA/$MODEL" "root@$HOST:/workspace/binary/models/"
scp -i "$KEY" -P "$PORT" "$DRAFTER" "root@$HOST:/workspace/binary/models/procreations-r2-gptq2.safetensors"
scp -i "$KEY" -P "$PORT" "$VOCAB" "root@$HOST:/workspace/binary/models/vocab-65536.txt"
"${SSH[@]}" "printf '%s  %s\\n' '$SHA' '/workspace/binary/models/$MODEL' '$DRAFT_SHA' '/workspace/binary/models/procreations-r2-gptq2.safetensors' | sha256sum -c -"
"${SSH[@]}" 'bash /workspace/binary/halo/tools/binary/bootstrap.sh /workspace/binary; cd /workspace/binary/halo; make -j1 PROFILE=release LLAMA=/workspace/binary/llama h5 h5-server refdump'
trap - ERR
printf 'READY instance=%s host=%s port=%s\n' "$ID" "$HOST" "$PORT"
printf 'Destroy when finished: %s destroy %s\n' "$VAST" "$ID"
