#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?Pass a new bounded BF16 experiment output directory}
ORACLE=${2:-/workspace/ternary-bench/oracle/v1-router40-r1-router-step/manifest.json}
MODEL=${MODEL:-/workspace/models/gguf/Qwen3.6-35B-A3B-ternary-v1-PTQ1_0.gguf}
FORK=${FORK:-/workspace/src/prism-llama}
mkdir -p "$OUT" build
python3 bench/router-bf16-fixture.py "$ORACLE" "$OUT/fixture" > "$OUT/fixture-extraction.json"
if [[ ! -f build/bf16-gguf.o || src/gguf.cpp -nt build/bf16-gguf.o || include/model.hpp -nt build/bf16-gguf.o || include/format.hpp -nt build/bf16-gguf.o ]]; then
 ${CXX:-c++} -std=c++17 -O2 -Iinclude -Isrc -c src/gguf.cpp -o build/bf16-gguf.o
fi
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
 -Iinclude -Isrc -I"$FORK/vendor" bench/router-bf16.cu build/bf16-gguf.o -o build/router-bf16 2> "$OUT/ptxas.log"
sha256sum include/router-bf16.cuh include/router.cuh include/moe.cuh include/model.hpp bench/router-bf16.cu "$ORACLE" "$OUT/fixture/inputs.f32" > "$OUT/source.sha256"
flock /workspace/halo-6000/gpu.lock bash -c '
 set -euo pipefail
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv > "$1/gpu-before.csv"
 build/router-bf16 "$2" "$1/fixture" "$1" | tee "$1/results.jsonl"
 nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv > "$1/gpu-after.csv"
' -- "$OUT" "$MODEL"
python3 - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);p=out/'image.json';m=json.loads(p.read_text());m['sha256']=hashlib.sha256((out/'router-bf16-rne-v1.bin').read_bytes()).hexdigest();p.write_text(json.dumps(m,indent=2)+'\n')
PY
PYTHONPATH="$FORK/gguf-py${PYTHONPATH:+:$PYTHONPATH}" python3 bench/router-bf16-source.py \
 "$MODEL" "$OUT/router-bf16-rne-v1.bin" "$OUT/source-exactness.json"
sha256sum bench/router-bf16-source.py > "$OUT/source-audit.sha256"
