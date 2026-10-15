#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FORK=${FORK:-/workspace/src/prism-llama}
MODEL=${MODEL:-/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf}
FIXTURES=${FIXTURES:-bench/tokenizer-fixtures/fixtures.json}
mkdir -p build
${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Iinclude -Isrc \
  -isystem "$FORK/include" -isystem "$FORK/common" -isystem "$FORK/ggml/include" -isystem "$FORK/vendor" \
  src/tokenizer.cpp bench/tokenizer-test.cpp -L"$FORK/build/bin" \
  -Wl,-rpath,"$FORK/build/bin" -lllama-common -lllama -o build/tokenizer-test
build/tokenizer-test "$MODEL" "$FIXTURES"
