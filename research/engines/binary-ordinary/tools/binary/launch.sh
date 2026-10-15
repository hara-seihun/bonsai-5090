#!/usr/bin/env bash
set -euo pipefail
if [ "$#" != 3 ]; then
  echo 'usage: launch.sh OFFER_ID PRISM_SOURCE_DIR MODEL_DATA_DIR' >&2
  exit 2
fi
OFFER=$1
PRISM=$2
DATA=$3
HERE=$(cd "$(dirname "$0")" && pwd)
HALO=$(cd "$HERE/../.." && pwd)
VAST=vastai
KEY=~/.ssh/id_ed25519
PIN=1a07bfa5f4144274c8f1c9963821dd9d9a51854b
MODEL=Bonsai-27B-Q1_0.gguf
SHA=17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0
[ "$(git -C "$PRISM" rev-parse HEAD)" = "$PIN" ] || { echo 'REFERENCE_PIN_MISMATCH' >&2; exit 1; }
[ -f "$DATA/$MODEL" ] || { echo 'MODEL_MISSING' >&2; exit 1; }
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
tar -C "$HALO" --exclude=./.git --exclude=./build --exclude=./results --exclude=./tools/binary/endpoint.local -cf - . | "${SSH[@]}" 'tar -C /workspace/binary/halo -xf -'
scp -i "$KEY" -P "$PORT" "$DATA/$MODEL" "root@$HOST:/workspace/binary/models/"
"${SSH[@]}" "printf '%s  %s\\n' '$SHA' '/workspace/binary/models/$MODEL' | sha256sum -c -"
"${SSH[@]}" 'bash /workspace/binary/halo/tools/binary/bootstrap.sh /workspace/binary; cd /workspace/binary/halo; make -j4 LLAMA=/workspace/binary/llama h5 refdump'
trap - ERR
printf 'READY instance=%s host=%s port=%s\n' "$ID" "$HOST" "$PORT"
printf 'Destroy when finished: %s destroy %s\n' "$VAST" "$ID"
