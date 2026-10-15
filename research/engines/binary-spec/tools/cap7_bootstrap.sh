#!/usr/bin/env bash
set -euo pipefail
exec 9>/root/h5-gpu-measure.lock
flock -n 9 || { echo GPU_LOCK_BUSY >&2; exit 2; }
OUT="$PWD/results/cap7-bootstrap"
RUNTIME=/root/halo-cap7-runtime
mkdir -p "$OUT" "$RUNTIME" /root/models
exec > "$OUT/bootstrap.log" 2>&1
printf 'runner_commit=%s\n' "$HALO_COMMIT"
date -u --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,utilization.gpu --format=csv > "$OUT/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/apps-before.txt"
test ! -s "$OUT/apps-before.txt" || { echo GPU_NOT_EXCLUSIVE >&2; exit 3; }
for command in git cmake g++ curl sha256sum tar; do command -v "$command" >/dev/null; done
PRISM=1a07bfa5f4144274c8f1c9963821dd9d9a51854b
if [[ -e /root/llama ]]; then echo UNEXPECTED_LLAMA_DIRECTORY >&2; exit 4; fi
git clone --filter=blob:none --no-checkout https://github.com/PrismML-Eng/llama.cpp.git /root/llama
git -C /root/llama fetch origin "$PRISM"
git -C /root/llama checkout --detach "$PRISM"
cmake -S /root/llama -B /root/llama/build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build /root/llama/build --target llama mtmd -j4
fetch_model() {
  local file="$1" expected="$2" url="$3"
  test ! -e "$file"
  curl --fail --location --retry 0 --connect-timeout 10 --max-time 150 --output "$file.part" "$url"
  printf '%s  %s\n' "$expected" "$file.part" | sha256sum --check
  mv "$file.part" "$file"
}
fetch_model /root/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3 \
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/b072e1d3/Ternary-Bonsai-2-27B-PTQ1_0.gguf
fetch_model /root/models/model.safetensors 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c \
  https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/015e7956/model.safetensors
[[ $(g++ -dumpversion) == 13 ]] || { echo TARGET_GCC_MUST_BE_13 >&2; exit 5; }
NVCC=/usr/local/cuda/bin/nvcc
"$NVCC" --version | grep -F 'release 12.8'
make -j4 NVCC="$NVCC" NVCC_HOST_CXX=g++-13 LLAMA=/root/llama LLAMA_BUILD=/root/llama/build BUILD=build SERVER_BINARY="$RUNTIME/h5-server" integrated-single
sha256sum "$RUNTIME/h5-server" > "$RUNTIME/server.sha256"
printf '%s\n' "$HALO_COMMIT" > "$RUNTIME/engine-source.txt"
cp build/compile-config "$OUT/compile-config.txt"
"$NVCC" --version > "$OUT/nvcc-version.txt"
chmod 500 "$RUNTIME/h5-server"
ldd "$RUNTIME/h5-server" | tee "$OUT/server.ldd.txt"
if grep -F 'not found' "$OUT/server.ldd.txt"; then exit 5; fi
set +e
"$RUNTIME/h5-server" > "$OUT/server-start.stdout" 2> "$OUT/server-start.stderr"
code=$?
set -e
printf '%s\n' "$code" > "$OUT/server-start.exit"
[[ "$code" == 1 ]]
grep -F 'usage:' "$OUT/server-start.stderr"
cp -a "$INPUTS_DIR/capture/corpus" "$RUNTIME/corpus"
cp "$INPUTS_DIR/capture/gemm-plan.txt" "$RUNTIME/"
python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path('/root/halo-cap7-runtime')
features=json.loads(Path('results/integrated/build-features.json').read_text())
preset_files=['selected-preset.json','selected-batch-preset.json','selected-vision-preset.json','selected-degree8-preset.json']
for filename in preset_files:
 preset=json.loads((Path('results/integrated')/filename).read_text())
 preset['base_engine_source']=preset['engine_source']
 preset['engine_source']=os.environ['HALO_COMMIT']
 preset['server_sha256']=hashlib.sha256((root/'h5-server').read_bytes()).hexdigest()
 if preset['planning']=='frozen': preset['plan_file']='gemm-plan.txt'
 if 'serving-e03' in features['enabled_features']:
  if filename=='selected-batch-preset.json':
   preset['unsupported_modes']=[m for m in preset['unsupported_modes'] if m not in ['fair128','http-prefix']]
  elif filename in ['selected-preset.json','selected-vision-preset.json']:
   preset['environment'].update(H5_SERVING_FAIR='1',H5_SERVING_RESERVE='0',H5_SERVING_CHECK='0')
 if 'f02-h32' in features['enabled_features'] and filename=='selected-batch-preset.json':
  preset['unsupported_modes']=[m for m in preset['unsupported_modes'] if m!='f02-h32']
 if 'f07-degree8' in features['enabled_features'] and filename=='selected-degree8-preset.json':
  preset['unsupported_modes']=[]
 if 'head-demand' in features['enabled_features']:
  preset['environment']['H5_HEAD_ROWS']='demand'
 (root/filename).write_text(json.dumps(preset,indent=2)+'\n')
core=json.loads((root/'selected-preset.json').read_text())
for filename,mode in [('engine.json','strict-native'),('engine-tree.json','strict-tree')]:
 values={**core['environment'],**core['modes'][mode]['environment']}
 del values['H5_COMPACT'];del values['H5_DRAFT_MARGIN'];values['H5_PLAN_READ']='gemm-plan.txt'
 (root/filename).write_text(json.dumps(values,indent=2)+'\n')
all_values=json.loads((root/'engine.json').read_text());all_values['H5_HEAD_ROWS']='all'
(root/'engine-all.json').write_text(json.dumps(all_values,indent=2)+'\n')
PY
cp "$RUNTIME/h5-server" "$RUNTIME/selected-preset.json" "$RUNTIME/selected-batch-preset.json" "$RUNTIME/selected-vision-preset.json" "$RUNTIME/selected-degree8-preset.json" "$OUT/"
cp results/integrated/build-features.json "$OUT/"
python3 tools/registered_commit_panel.py --corpus-manifest "$RUNTIME/corpus/manifest.json" --corpus-sha256 "$CAP7_CORPUS_SHA" --validate-only
sha256sum /root/llama/build/bin/*.so* > "$OUT/runtime-libraries.sha256"
tar -C /root/llama/build/bin -cf "$OUT/runtime-libraries.tar" --wildcards '*.so*'
{
  printf 'state=ready\nengine_source=%s\nprism=%s\n' "$HALO_COMMIT" "$PRISM"
  sha256sum "$RUNTIME/h5-server" /root/models/*.gguf /root/models/model.safetensors "$RUNTIME/gemm-plan.txt" "$RUNTIME/engine.json"
  g++ --version
  git -C /root/llama rev-parse HEAD
  date -u --iso-8601=seconds
} > "$OUT/ready.txt"
