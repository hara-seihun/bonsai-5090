#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! -d "$1" ]]; then
    printf 'usage: %s EXISTING_LEAN_MATHLIB_OWNER\n' "$0" >&2
    exit 2
fi
owner=$(realpath "$1")
here=$(cd -- "$(dirname -- "$0")" && pwd)
binary=$(mktemp /tmp/binary-bitserial-witness.XXXXXX)
lean_log=$(mktemp /tmp/binary-bitserial-lean.XXXXXX)
trap 'rm -f -- "$binary" "$lean_log"' EXIT
printf 'source_sha256:\n'
sha256sum "$here/witness.cpp" "$here/Bitserial.lean"
printf 'C++ compiler: '
c++ -dumpfullversion
printf 'C++ command: c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic witness.cpp -o TEMP\n'
timeout 20s c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic "$here/witness.cpp" -o "$binary"
timeout 10s /usr/bin/time -f 'C++ elapsed=%e seconds' "$binary"
printf 'Lean owner: %s\n' "$owner"
cd -- "$owner"
timeout 5s lake env lean --version
printf 'Lean command: cd OWNER && lake env lean ABSOLUTE_PATH/Bitserial.lean\n'
timeout 30s /usr/bin/time -f 'Lean elapsed=%e seconds' lake env lean "$here/Bitserial.lean" | tee "$lean_log"
if grep -Eq 'sorryAx|ofReduceBool|nativeDecide' "$lean_log"; then
    printf 'FAIL Lean proof contains an admitted or native-computation axiom\n' >&2
    exit 1
fi
printf 'PASS Lean kernel checked: no sorryAx, no native-decide axiom\n'
