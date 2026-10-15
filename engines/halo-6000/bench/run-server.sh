#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FORK=${FORK:-/workspace/src/prism-llama}
MODEL=${MODEL:-/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf}
FIXTURES=${FIXTURES:-bench/tokenizer-fixtures/fixtures.json}
CXX=${CXX:-c++}
mkdir -p build
HTTP="$FORK/vendor/cpp-httplib"
if [[ ! -f build/httplib-test.o || "$HTTP/httplib.cpp" -nt build/httplib-test.o || "$HTTP/httplib.h" -nt build/httplib-test.o ]]; then
 "$CXX" -std=c++17 -O1 -pthread -I"$HTTP" -c "$HTTP/httplib.cpp" -o build/httplib-test.o
fi
"$CXX" -std=c++17 -O2 -pthread -Wall -Wextra -Iinclude -Isrc \
 -isystem "$FORK/include" -isystem "$FORK/common" -isystem "$FORK/ggml/include" -isystem "$FORK/vendor" -isystem "$HTTP" \
 src/tokenizer.cpp src/server.cpp bench/server-test.cpp build/httplib-test.o \
 -L"$FORK/build/bin" -Wl,-rpath,"$FORK/build/bin" -lllama-common -lllama -o build/server-test
build/server-test "$MODEL" "$FIXTURES"
