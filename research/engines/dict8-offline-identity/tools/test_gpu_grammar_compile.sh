#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# > 1 )); then
    printf 'usage: %s [GRAMMAR_CASES_JSON]\n' "$0" >&2
    exit 2
fi
cases=${1:-"$root/tools/grammar_cases.json"}
build=$(mktemp -d "${TMPDIR:-/tmp}/gpu-grammar-compile-test.XXXXXXXX")
trap 'rm -rf -- "$build"' EXIT

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Wpedantic -I"$root/src" \
    "$root/tools/gpu_grammar_compile_test.cpp" \
    "$root/src/grammar.cpp" "$root/src/gpu_grammar_compile.cpp" \
    -o "$build/gpu-grammar-compile-test"
"$build/gpu-grammar-compile-test" "$cases"
