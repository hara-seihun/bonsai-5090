#!/bin/bash
set -euo pipefail
ROOT=/workspace/binary/digits-ragged/halo
OUT=/workspace/binary/row-fibers
cd "$ROOT"
NVCC=/usr/local/cuda/bin/nvcc
"$NVCC" -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I. -Isrc -I/workspace/binary/llama/include -I/workspace/binary/llama/ggml/include -x cu -c "$OUT/row_fibers.cpp" -o "$OUT/row_fibers.o"
OBJECTS=(build/engine.o build/dspark_native.o build/dspark_kernels.o build/packed1.o build/bitserial.o build/digits.o build/composition.o build/fp4_pair.o build/graph_memory.o build/kernels.o build/fp4.o build/sk.o build/draft.o build/sampling_gpu.o build/f02-producer.o build/f02-continuation.o build/gguf.o)
"$NVCC" -o "$OUT/row_fibers" "$OUT/row_fibers.o" "${OBJECTS[@]}" -lcublas -L/workspace/binary/llama/build/bin -lllama -Xlinker -rpath,/workspace/binary/llama/build/bin -Xlinker --wrap=_ZN2h511launch_prepERKNS_8PrepArgsEiP11CUstream_st -Xlinker --wrap=_ZN9bitserial7produceENS_5RouteEPKfiiNS_7CarrierEP11CUstream_st -Xcompiler -pthread
{ printf 'source='; head -1 source.txt; sha256sum "$OUT/row_fibers.cpp" "$OUT/row_fibers.o" "$OUT/row_fibers" "${OBJECTS[@]}" src/engine.cu src/chat.h /workspace/binary/models/wiki.test.raw; } > "$OUT/sourcepins.txt"
python3 "$OUT/row_fibers_prepare.py"
