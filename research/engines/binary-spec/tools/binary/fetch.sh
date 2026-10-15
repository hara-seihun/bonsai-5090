#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?usage: fetch.sh MODEL_DATA_DIR}
mkdir -p "$DATA"
M=Bonsai-27B-Q1_0.gguf
EXPECTED=17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0
if [ ! -f "$DATA/$M" ]; then
  curl -fL --connect-timeout 20 --max-time 1800 -o "$DATA/$M.partial" "https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main/$M"
  printf '%s  %s\n' "$EXPECTED" "$DATA/$M.partial" | sha256sum -c -
  mv "$DATA/$M.partial" "$DATA/$M"
fi
printf '%s  %s\n' "$EXPECTED" "$DATA/$M" | sha256sum -c -
