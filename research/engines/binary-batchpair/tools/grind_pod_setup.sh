#!/usr/bin/env bash
# Prepare a fresh RTX 5090 RunPod pod for halo grind measurements.
# Expects /root/src-*.tgz (git archives) already copied; builds each into /root/srv/<name>.
#   grind_pod_setup.sh NAME... (each NAME has /root/src-NAME.tgz and /root/src-NAME.commit)
set -euo pipefail
mkdir -p /root/models /root/srv /root/boot
cd /root/boot
cat > cudacheck.cu <<'C'
#include <cstdio>
__global__ void k(float*x){x[threadIdx.x]+=1;}
int main(){int n=0;cudaGetDeviceCount(&n);float*d;cudaMalloc(&d,1024);k<<<1,256>>>(d);auto e=cudaDeviceSynchronize();printf("devs=%d sync=%s\n",n,cudaGetErrorString(e));return e!=0;}
C
/usr/local/cuda/bin/nvcc -Wno-deprecated-gpu-targets -o cudacheck cudacheck.cu && ./cudacheck
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv
fetch() {
  local file=$1 sha=$2 url=$3
  [[ -f $file ]] && return 0
  curl -fsSL --retry 3 -o "$file.part" "$url"
  echo "$sha  $file.part" | sha256sum -c --quiet
  mv "$file.part" "$file"
}
fetch /root/models/bonsai.gguf 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3 \
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/b072e1d3/Ternary-Bonsai-2-27B-PTQ1_0.gguf &
fetch /root/models/model.safetensors 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c \
  https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/015e7956/model.safetensors &
if [[ ! -f /root/llama/build/bin/libllama.so ]]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/PrismML-Eng/llama.cpp.git /root/llama
  git -C /root/llama fetch -q origin 1a07bfa5f4144274c8f1c9963821dd9d9a51854b
  git -C /root/llama checkout -q --detach 1a07bfa5f4144274c8f1c9963821dd9d9a51854b
  cmake -S /root/llama -B /root/llama/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF > /dev/null
  cmake --build /root/llama/build --target llama mtmd -j32 > /dev/null
fi
for name in "$@"; do
  d=/root/srv/$name; rm -rf "$d"; mkdir -p "$d"; tar xzf /root/src-$name.tgz -C "$d"
  ( cd "$d" && make -j24 NVCC=/usr/local/cuda/bin/nvcc NVCC_HOST_CXX=g++-13 LLAMA=/root/llama LLAMA_BUILD=/root/llama/build \
      BUILD=build SERVER_BINARY=/root/srv/$name.server integrated-single > /root/srv/$name.build.log 2>&1 ) &
done
wait
for name in "$@"; do
  test -x /root/srv/$name.server || { echo "BUILD_FAILED $name"; tail -20 /root/srv/$name.build.log; exit 1; }
  cp /root/src-$name.commit /root/srv/$name.server.source
  sha256sum /root/srv/$name.server
done
ls -la /root/models
echo SETUP_OK
