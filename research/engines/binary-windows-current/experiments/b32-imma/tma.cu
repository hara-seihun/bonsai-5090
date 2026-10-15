// TMA (cp.async.bulk) streaming throughput per SM: DRAM-sourced and L2-sourced, by copy size and ring depth.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
__device__ __forceinline__ unsigned su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su32(b)), "r"(c)); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned ph) { asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(su32(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void bulk(void* d, const void* s, unsigned n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(su32(d)), "l"(s), "r"(n), "r"(su32(b)) : "memory");
}
// One thread issues; ring of S stages of `bytes` each; `copies` copies per stage (bytes/copies each).
__global__ void stream(const uint8_t* src, size_t span, int iters, int bytes, int copies, int stages, int* sink) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + stages * bytes);
  if (threadIdx.x == 0) { for (int s = 0; s < stages; ++s) mbar_init(&full[s], 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  if (threadIdx.x != 0) return;
  const size_t base = (size_t)blockIdx.x * iters * bytes;
  int acc = 0;
  for (int i = 0; i < iters + stages; ++i) {
    if (i >= stages) { const int s = (i - stages) % stages; mbar_wait(&full[s], ((i - stages) / stages) & 1); acc += sm[s * bytes]; }
    if (i < iters) {
      const int s = i % stages;
      mbar_expect(&full[s], bytes);
      const size_t off = (base + (size_t)i * bytes) % span;
      for (int c = 0; c < copies; ++c) bulk(sm + s * bytes + c * (bytes / copies), src + off + c * (bytes / copies), bytes / copies, &full[s]);
    }
  }
  if (acc == 0x7fffffff) sink[0] = acc;
}
int main() {
  int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  size_t big = size_t(2) << 30; uint8_t* buf; int* sink; CK(cudaMalloc(&buf, big)); CK(cudaMalloc(&sink, 4)); CK(cudaMemset(buf, 1, big));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int l2 : {0, 1}) for (int bytes : {2048, 4608, 9216, 18432}) for (int copies : {1, 4}) for (int stages : {4, 8}) {
    const size_t span = l2 ? (size_t(24) << 20) : big;
    const int smem = stages * bytes + stages * 8 + 128;
    if (smem > 99 * 1024) continue;
    CK(cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const int iters = int(std::min<size_t>(l2 ? 2000 : big / sms / bytes, 2000));
    stream<<<sms, 32, smem>>>(buf, span, 16, bytes, copies, stages, sink); CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int r = 0; r < 5; ++r) { CK(cudaEventRecord(a)); stream<<<sms, 32, smem>>>(buf, span, iters, bytes, copies, stages, sink); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms); }
    std::sort(t.begin(), t.end());
    const double tot = double(sms) * iters * bytes;
    printf("{\"src\":\"%s\",\"bytes\":%d,\"copies\":%d,\"stages\":%d,\"total_gbps\":%.0f,\"per_sm_gbps\":%.1f}\n", l2 ? "L2" : "DRAM", bytes, copies, stages, tot / (t[2] * 1e-3) / 1e9, tot / sms / (t[2] * 1e-3) / 1e9);
  }
  return 0;
}
