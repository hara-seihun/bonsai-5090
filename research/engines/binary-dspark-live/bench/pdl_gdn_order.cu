// Stress the real GDN kernel's ancestor-read/current-producer-write boundary.
// Run with H5_PDL=0 and H5_PDL=1, compare the binary snapshots byte for byte.
#include "kernels.cuh"
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace h5 { void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t); }
static void check(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); }
}
template <typename T> static T* alloc(size_t count) {
  T* p; check(cudaMalloc(&p, count * sizeof(T))); return p;
}
__device__ void delay(unsigned ns) {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}
__global__ void seed(__half* state, __half* pend, h5::Seg* seg) {
  h5::pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (int j = i; j < h5::HV * 128 * 128; j += stride)
    state[j] = __float2half_rn((j % 17 - 8) * .002f);
  for (int j = i; j < 2 * h5::PMAX * h5::PEND_GDN; j += stride)
    pend[j] = __float2half_rn((j % 11 - 5) * .004f);
  if (i == 0) *seg = {0, 0, 8, 0, 1, 0};
  // Deliberately skew ancestor CTAs. The intermediate producer must acquire all of them.
  if (threadIdx.x == 0) delay(blockIdx.x % 3 * 3000);
  __syncthreads(); h5::pdl_launch();
}
__global__ void metadata(h5::Seg* dst, h5::Seg src) {
  h5::pdl_wait();
  if (threadIdx.x == 0) *dst = src;
  __syncthreads(); h5::pdl_launch();
}
__global__ void produce(float* conv, float* gates, unsigned ns, int step) {
  // Same contract as k_conv: acquires all ancestors, releases, writes only conv/gates.
  h5::pdl_wait(); h5::pdl_launch();
  delay(ns + (blockIdx.x % 4) * 1000);
  const int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (int j = i; j < 8 * h5::CONV_CH; j += stride) conv[j] = (j % 23 - 11 + step) * .003f;
  for (int j = i; j < 8 * 96; j += stride) gates[j] = j % 96 < h5::HV ? .97f : .125f;
}
int main(int argc, char** argv) {
  if (argc != 2 || !getenv("H5_PDL") || (getenv("H5_PDL")[0] != '0' && getenv("H5_PDL")[0] != '1')) {
    fprintf(stderr, "usage: H5_PDL=0|1 pdl-gdn-order NEW_SNAPSHOT\n"); return 1;
  }
  FILE* f = fopen(argv[1], "wbx"); if (!f) { perror(argv[1]); return 1; }
  const size_t state_n = h5::HV * 128 * 128, pend_n = 2 * h5::PMAX * h5::PEND_GDN;
  auto* state = alloc<__half>(state_n); auto* pend = alloc<__half>(pend_n);
  auto* seg = alloc<h5::Seg>(1); auto* conv = alloc<float>(8 * h5::CONV_CH);
  auto* gates = alloc<float>(8 * 96); auto* y = alloc<float>(8 * h5::VDIM);
  cudaStream_t st; check(cudaStreamCreate(&st));
  h5::GdnArgs args{}; args.state = state; args.pend = pend; args.segs = seg;
  args.conv_out = conv; args.gates = gates; args.y = y;
  auto chain = [&](unsigned ns) {
    h5::pdl(seed, dim3(96), dim3(256), 0, st, state, pend, seg);
    for (int step = 0; step < 12; step++) {
      const h5::Seg visits[] = {{0,0,8,0,1,0}, {0,0,8,3,1,1}, {0,0,8,2,0,0}, {0,0,8,0,1,1}};
      h5::pdl(metadata, dim3(1), dim3(32), 0, st, seg, visits[step % 4]);
      h5::pdl(produce, dim3(96), dim3(128), 0, st, conv, gates, ns, step);
      h5::launch_gdn(args, 1, h5::StateFormat::F16, st);
    }
  };
  auto snapshot = [&](const void* src, size_t bytes) {
    std::vector<unsigned char> host(bytes); check(cudaMemcpy(host.data(), src, bytes, cudaMemcpyDeviceToHost));
    if (fwrite(host.data(), 1, bytes, f) != bytes) { perror("snapshot"); exit(1); }
  };
  for (unsigned ns : {0U, 2000U, 5000U, 20000U}) {
    check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); chain(ns);
    cudaGraph_t graph; check(cudaStreamEndCapture(st, &graph));
    cudaGraphExec_t exec; check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 4; repeat++) {
      check(cudaGraphLaunch(exec, st)); check(cudaStreamSynchronize(st));
      snapshot(state, state_n * sizeof(__half)); snapshot(pend, pend_n * sizeof(__half));
      snapshot(y, 8 * h5::VDIM * sizeof(float));
    }
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
  }
  if (fclose(f)) { perror("snapshot close"); return 1; }
  check(cudaFree(state)); check(cudaFree(pend)); check(cudaFree(seg));
  check(cudaFree(conv)); check(cudaFree(gates)); check(cudaFree(y));
  check(cudaStreamDestroy(st));
  puts("48 GDN visits x 4 graph repeats; snapshot saved");
}
