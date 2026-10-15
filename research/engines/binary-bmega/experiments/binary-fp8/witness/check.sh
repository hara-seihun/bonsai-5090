#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  printf 'Usage: %s MATHLIB_LEAN_PROJECT\n' "$0" >&2
  exit 2
fi
project=$(realpath "$1")
witness=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
{
  printf 'Binary FP8 semantic witness check\n'
  date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
  printf 'Lean project: %s\n' "$project"
  (cd "$project" && timeout 5s lake env lean --version)
  timeout 5s g++ --version
  printf '\nSource hashes:\n'
  (cd "$witness" && sha256sum BinaryFP8.lean finite.cpp README.md check.sh)
  printf '\nLean: lake env lean BinaryFP8.lean (45-second limit)\n'
  (cd "$project" && timeout 45s lake env lean "$witness/BinaryFP8.lean")
  printf '\nC++: g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic finite.cpp\n'
  timeout 15s g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic "$witness/finite.cpp" -o "$tmp/finite"
  timeout 5s "$tmp/finite"
  printf '\nPASS: Lean and C++ checks exited 0\n'
} 2>&1 | tee "$witness/check-receipt.txt"
