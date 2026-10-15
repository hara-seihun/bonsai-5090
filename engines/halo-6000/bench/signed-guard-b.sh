#!/usr/bin/env bash
set -euo pipefail
cd /workspace/halo-6000
python3 - <<'PY'
import json,pathlib,time
p=pathlib.Path('/workspace/halo-6000/docs/online-head-b/cache-proof.json');build=pathlib.Path('/workspace/ternary-bench/jobs/halo-content-build-proof.json')
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    try:
        proof=json.loads(p.read_text());state=json.loads(build.read_text())
        if state['state']=='succeeded' and proof.get('source_weights_verified') is True:
            break
    except (FileNotFoundError,json.JSONDecodeError): pass
    time.sleep(2)
else: raise SystemExit('signed guard dependency deadline')
PY
/usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=sm_120a -Iinclude -Isrc -Xcompiler -fopenmp bench/signed-layer.cu build/runtime.o build/decode.o build/gguf.o build/cache.o -o bench/signed-layer
flock gpu.lock bash -lc 'set -euo pipefail; nvidia-smi > docs/online-head-b/signed-context-before.txt; CUDA_MODULE_LOADING=EAGER bench/signed-layer /workspace/models/gguf/Qwen3.6-35B-A3B-online-rot-head-native-PTQ1_0-online.gguf /workspace/models/halo6000-cache > docs/online-head-b/signed-phase-guard.jsonl; nvidia-smi > docs/online-head-b/signed-context-after.txt'
