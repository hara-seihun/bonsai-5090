#!/usr/bin/env bash
# Own one explicit serial GPU measurement suite; panel.py owns each server.
set -euo pipefail
if (( $# < 2 )); then echo 'usage: panels.sh CONFIG_DIR NAME...' >&2; exit 2; fi
configs=$1; shift
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
python=/workspace/binary/panel/env/bin/python
failed=0
for name in "$@"; do
  config="$configs/$name.json"
  test -f "$config"
  printf '\n=== PANEL %s ===\n' "$name"
  if flock /tmp/binary-gpu.lock timeout --signal=TERM --kill-after=20 2100 env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=/root PYTHONPATH="$root/tools" "$python" "$root/tools/binary/panel.py" --config "$config"; then
    printf 'PANEL_COMPLETE %s\n' "$name"
  else
    status=$?
    printf 'PANEL_FAILED %s %s\n' "$name" "$status"
    failed=1
  fi
done
exit "$failed"
