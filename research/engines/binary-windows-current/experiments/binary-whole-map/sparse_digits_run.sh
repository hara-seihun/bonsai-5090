#!/usr/bin/env bash
set -euo pipefail
if [ "$#" != 1 ]; then echo 'usage: sparse_digits_run.sh all|witness-only' >&2; exit 2; fi
case "$1" in all) rounds=(witness down16 down32);; witness-only) rounds=(witness);; *) echo 'invalid round selection' >&2; exit 2;; esac
cd ~/work/clones/binary-spec-5090
E=experiments/binary-whole-map
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST)
R=/workspace/binary/digits-ragged/halo/experiments/binary-whole-map
D="$E/sparse-digit-results"
export_results(){ scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT "root@$POD_HOST:$R/sparse-digit-results/*" "$D/"; }
trap export_results EXIT
"${SSH[@]}" "mkdir -p '$R/build' '$R/sparse-digit-results'"
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT "$E/sparse_digits.cu" "$E/sparse_digits.cuh" "$E/sparse_digits_bench.cu" "root@$POD_HOST:$R/"
scp -q -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P $POD_PORT "$D/captured-hidden.f32" "root@$POD_HOST:$R/sparse-digit-results/"
"${SSH[@]}" "timeout 55 bash -s -- '$1'" <<'REMOTE'
set -euo pipefail
round=$1
cd /workspace/binary/digits-ragged/halo
E=experiments/binary-whole-map
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc)
/usr/local/cuda/bin/nvcc "${FLAGS[@]}" -Xptxas=-v -c "$E/sparse_digits.cu" -o "$E/build/sparse_digits.o" 2> "$E/sparse-digit-results/build.log"
/usr/local/cuda/bin/nvcc "${FLAGS[@]}" -c "$E/sparse_digits_bench.cu" -o "$E/build/sparse_digits_bench.o"
/usr/local/cuda/bin/nvcc "${FLAGS[@]}" "$E/build/sparse_digits.o" "$E/build/sparse_digits_bench.o" "$E/build/signed_digits.o" build/bitserial.o build/digits.o build/kernels.o build/gguf.o -o "$E/build/sparse_digits_bench"
sha256sum "$E"/sparse_digits.{cu,cuh} "$E/sparse_digits_bench.cu" "$E/build"/sparse_digits* "$E/build/signed_digits.o" build/{bitserial,digits,kernels,gguf}.o > "$E/sparse-digit-results/provenance.sha256"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv > "$E/sparse-digit-results/gpu.txt"
/usr/local/cuda/bin/cuobjdump --dump-sass "$E/build/sparse_digits.o" | gzip -c > "$E/sparse-digit-results/sparse_digits.sass.gz"
cp "$E/build/sparse_digits.o" "$E/sparse-digit-results/sparse_digits.o"
for artifact in sparse_digits.o sparse_digits.sass.gz provenance.sha256; do
  dst="$E/sparse-digit-results/$round-$artifact"
  if [ -e "$dst" ]; then echo "round artifact already exists: $dst" >&2; exit 2; fi
  cp "$E/sparse-digit-results/$artifact" "$dst"
done
REMOTE
status=0
for kind in "${rounds[@]}"; do
  set +e
  "${SSH[@]}" "flock -w 600 /tmp/binary-gpu.lock timeout 55 bash -s -- '$kind'" <<'REMOTE'
set -euo pipefail
kind=$1
cd /workspace/binary/digits-ragged/halo
E=experiments/binary-whole-map
export H5_PDL=1
M=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
if [ "$kind" = witness ]; then r=16; else r=${kind#down};kind=down; fi
"$E/build/sparse_digits_bench" "$M" "$kind" "$r" "$E/sparse-digit-results/$kind-r$r.jsonl" "$E/sparse-digit-results/captured-hidden.f32"
REMOTE
  status=$?
  set -e
  if [ "$status" != 0 ]; then break; fi
done
exit "$status"
