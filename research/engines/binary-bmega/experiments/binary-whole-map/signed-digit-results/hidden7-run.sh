#!/usr/bin/env bash
set -euo pipefail
cd ~/work/clones/binary-spec-5090
E=experiments/binary-whole-map
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
R=/workspace/binary/digits-ragged/halo/experiments/binary-whole-map
"${SSH[@]}" "mkdir -p '$R/build' '$R/signed-digit-results'"
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT "$E/signed_digits.cu" "$E/signed_digits.cuh" "$E/signed_digits_bench.cu" "root@$POD_HOST:$R/"
"${SSH[@]}" "timeout 55 bash -s" <<'REMOTE'
set -euo pipefail
cd /workspace/binary/digits-ragged/halo
E=experiments/binary-whole-map
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc)
/usr/local/cuda/bin/nvcc "${FLAGS[@]}" -c "$E/signed_digits.cu" -o "$E/build/signed_digits.o"
/usr/local/cuda/bin/nvcc "${FLAGS[@]}" -c "$E/signed_digits_bench.cu" -o "$E/build/signed_digits_bench.o"
/usr/local/cuda/bin/nvcc "${FLAGS[@]}" "$E/build/signed_digits.o" "$E/build/signed_digits_bench.o" build/bitserial.o build/digits.o build/kernels.o build/gguf.o -o "$E/build/signed_digits_bench"
sha256sum "$E"/signed_digits* "$E/build"/* build/{bitserial,digits,kernels,gguf}.o > "$E/signed-digit-results/provenance.sha256"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv > "$E/signed-digit-results/gpu.txt"
REMOTE
set +e
"${SSH[@]}" "flock -w 600 /tmp/binary-gpu.lock timeout 55 bash -s" <<'REMOTE'
set -euo pipefail
cd /workspace/binary/digits-ragged/halo
E=experiments/binary-whole-map
export H5_PDL=1
M=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
"$E/build/signed_digits_bench" "$M" witness 16 "$E/signed-digit-results/witness.jsonl"
"$E/build/signed_digits_bench" "$M" down 16 "$E/signed-digit-results/down-r16.jsonl"
"$E/build/signed_digits_bench" "$M" down 32 "$E/signed-digit-results/down-r32.jsonl"
REMOTE
status=$?
set -e
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT "root@$POD_HOST:$R/signed-digit-results/*" "$E/signed-digit-results/"
exit "$status"
