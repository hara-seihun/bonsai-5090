#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"${NVCC:-/usr/local/cuda/bin/nvcc}" -std=c++17 -O3 -arch=sm_120 --fmad=true -I"$ROOT/include" "$ROOT/bench/gdn.cu" -o "$ROOT/bench/halo-gdn"
"$ROOT/bench/halo-gdn" "$@"
