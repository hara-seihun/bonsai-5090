#!/usr/bin/env bash
set -euo pipefail
DIR=${1:?usage: fetch-dspark.sh DATA_DIR bf16|Q4_1}
FORMAT=${2:?usage: fetch-dspark.sh DATA_DIR bf16|Q4_1}
PIN=f10afb355f104535e3e3e98cf7ab7795c72bd292
case "$FORMAT" in
  bf16) SIZE=7291885792; SHA=93bd7f0326bcb39982051dd7742e5a2c9031358f94293a9c91f6aa98dcb6573b ;;
  Q4_1) SIZE=1787468768; SHA=25e73f9f7ab5d1f7f1336711496dbc12da674e639ec88d579dc8683045befb1b ;;
  *) echo 'DSPARK_FORMAT_INVALID' >&2; exit 2 ;;
esac
mkdir -p "$DIR"
FILE="$DIR/Bonsai-27B-dspark-$FORMAT.gguf"
if [ -e "$FILE" ]; then printf '%s  %s\n' "$SHA" "$FILE" | sha256sum -c -; exit; fi
FREE=$(df -PB1 "$DIR" | awk 'NR==2 {print $4}')
[ "$FREE" -ge "$((SIZE + 8589934592))" ] || { echo 'DSPARK_FETCH_HEADROOM_8G_REQUIRED' >&2; exit 1; }
curl --http1.1 -fL --retry 3 --retry-all-errors --connect-timeout 15 --max-time 1800 -C - \
  "https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/$PIN/Bonsai-27B-dspark-$FORMAT.gguf" -o "$FILE.download"
[ "$(stat -c %s "$FILE.download")" -eq "$SIZE" ] || { echo 'DSPARK_SIZE_MISMATCH' >&2; exit 1; }
printf '%s  %s\n' "$SHA" "$FILE.download" | sha256sum -c -
mv "$FILE.download" "$FILE"
printf 'DSPARK_READY %s %s\n' "$SHA" "$FILE"
