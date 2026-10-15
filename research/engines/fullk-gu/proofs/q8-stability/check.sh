#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! -d "$1/mathlib/.lake/build/lib/lean" ]]; then
  printf 'Usage: %s MATHLIB_LAKE_PACKAGES_DIRECTORY\n' "$0" >&2
  exit 2
fi
cache=$(realpath "$1")
cd "$(dirname "$(realpath "$0")")"
mkdir -p .build
export LEAN_PATH="$PWD/.build:$(find "$cache" -maxdepth 5 -type d -path '*/build/lib/lean' | sort | paste -sd:)"
{
  printf 'checked_at_utc='; date -u +%FT%TZ
  lean --version
  printf 'mathlib_commit='; git -C "$cache/mathlib" rev-parse HEAD
  sha256sum Stability.lean Runaway.lean Audit.lean
  lean -o .build/Stability.olean Stability.lean
  lean -o .build/Runaway.olean Runaway.lean
  lean Audit.lean
  printf 'PASS: all declarations compiled; theorem axioms printed above\n'
} 2>&1 | tee check-receipt.txt
