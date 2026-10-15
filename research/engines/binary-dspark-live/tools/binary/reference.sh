#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?usage: reference.sh REMOTE_ROOT}
M="$ROOT/models/Bonsai-27B-Q1_0.gguf"
B="$ROOT/llama/build/bin"
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv > "$ROOT/results/hardware.csv"
printf '%s  %s\n' 17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0 "$M" | sha256sum -c - > "$ROOT/results/model-sha256.txt"
"$B/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 512 -n 128 -r 5 -o json > "$ROOT/results/reference-bench.json" 2> "$ROOT/results/reference-bench.log"
cd "$ROOT/halo"
make -j4 LLAMA="$ROOT/llama" refdump
./refdump "$M" "$ROOT/models/wiki.test.raw" 256 "$ROOT/results/wiki256.ref.bin" teacher > "$ROOT/results/reference-wiki.log" 2>&1
g++ -O2 -std=c++17 -I"$ROOT/llama/include" -I"$ROOT/llama/ggml/include" tools/binary/refbench.cpp -o refbench -L"$ROOT/llama/build/bin" -lllama -Wl,-rpath,"$ROOT/llama/build/bin"
./refbench "$M" "$ROOT/results/wiki256.ref.bin" 128 5 > "$ROOT/results/reference-decodebench.json" 2> "$ROOT/results/reference-decodebench.log"
printf '%s\n' 'Explain why a rainbow is a circle, in three concise sentences.' > "$ROOT/results/chat.txt"
./refdump "$M" "$ROOT/results/chat.txt" 64 "$ROOT/results/chat64.ref.bin" chat > "$ROOT/results/reference-chat.log" 2>&1
