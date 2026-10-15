#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
local_out="$root/experiments/binary-dspark-profile/live/evidence"
remote=/workspace/binary/dspark-live-profile-82ad352e
ssh_args=(-i ~/.ssh/id_ed25519 -p $POD_PORT -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2)
scp_args=(-i ~/.ssh/id_ed25519 -P $POD_PORT -o ConnectTimeout=10)
host=root@$POD_HOST
mkdir -p "$local_out"
ssh "${ssh_args[@]}" "$host" "mkdir -p '$remote'; nsys --version; sha256sum /workspace/binary/dspark-live/halo/h5-server" > "$local_out/remote-identity.txt"
scp "${scp_args[@]}" "$root/experiments/binary-dspark-profile/live/run_remote.py" "$root/experiments/binary-dspark-profile/live/launch.json" "$host:$remote/"
set +e
ssh "${ssh_args[@]}" "$host" "flock -w 180 /tmp/binary-gpu.lock timeout --kill-after=5 240 python3 '$remote/run_remote.py' '$remote/evidence' '$remote/launch.json'" > "$local_out/job.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$local_out/remote-exit.txt"
# CUDA work has ended and the exclusive lease is released before CPU-only export.
ssh "${ssh_args[@]}" "$host" "if test -f '$remote/evidence/trace.nsys-rep'; then timeout 45 nsys export --type sqlite --output '$remote/evidence/trace.sqlite' '$remote/evidence/trace.nsys-rep'; fi" > "$local_out/export.log" 2>&1
scp -r "${scp_args[@]}" "$host:$remote/evidence/." "$local_out/"
find "$local_out" -maxdepth 1 -type f -printf '%f %s bytes\n'
exit "$result"
