// PDL release latency vs. in-flight dependent prefetch traffic.
// P: 1 CTA, triggers dependents at entry, spins ~SPIN ns, stamps its end, exits.
// D: CTAS CTAs launched with programmatic serialization; before griddepcontrol.wait each CTA
//    issues BYTES of L2 bulk prefetch (or bulk copies into smem with MODE=1); stamps wait return.
// Prints median (D first release - P end) per configuration.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstdint>
__device__ __forceinline__ uint64_t gt() { uint64_t v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v) :: "memory"); return v; }
__global__ void kP(uint64_t* stamp, unsigned spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint64_t t0 = gt(); while (gt() - t0 < spin_ns) {}
  stamp[0] = gt();
}
__global__ void kD(const uint8_t* buf, size_t bytes, uint64_t* stamp, int mode) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) uint64_t bar;
  if (threadIdx.x == 0) {
    const uint8_t* p = buf + (size_t)blockIdx.x * bytes;
    if (mode == 0) {
      for (size_t o = 0; o < bytes; o += 16384) {
        unsigned n = (unsigned)(bytes - o < 16384 ? bytes - o : 16384);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(n) : "memory");
      }
    } else if (bytes) {
      unsigned b = (unsigned)__cvta_generic_to_shared(&bar);
      asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(b));
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"((unsigned)bytes));
      for (size_t o = 0; o < bytes; o += 16384) {
        unsigned n = (unsigned)(bytes - o < 16384 ? bytes - o : 16384);
        unsigned d = (unsigned)__cvta_generic_to_shared(sm + o);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "l"(p + o), "r"(n), "r"(b) : "memory");
      }
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  uint64_t t = gt();
  if (threadIdx.x == 0) stamp[1 + blockIdx.x] = t;
  if (mode == 1 && bytes && threadIdx.x == 0) {
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar);
    asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra W; }" :: "r"(b) : "memory");
  }
}
int main() {
  const int CTAS = 136; const size_t TOTAL = 2ull << 30;
  uint8_t* buf; cudaMalloc(&buf, TOTAL); cudaMemset(buf, 1, TOTAL);
  uint64_t* st; cudaMalloc(&st, 8 * (CTAS + 1));
  cudaStream_t s; cudaStreamCreate(&s);
  cudaFuncSetAttribute(kD, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  std::vector<uint64_t> h(CTAS + 1);
  for (int mode = 0; mode < 2; mode++)
  for (unsigned spin : {2000u, 6000u})
  for (size_t bytes : {0ul, 16384ul, 65536ul, 98304ul, 196608ul}) {
    std::vector<double> gaps;
    size_t off = 0;
    for (int it = 0; it < 40; it++) {
      if (off + bytes * CTAS > TOTAL) off = 0;
      kP<<<1, 32, 0, s>>>(st, spin);
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = CTAS; cfg.blockDim = 128; cfg.dynamicSmemBytes = mode ? 200 * 1024 : 0; cfg.stream = s;
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = at; cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kD, (const uint8_t*)(buf + off), bytes, st, mode);
      off += bytes * CTAS;
      cudaStreamSynchronize(s);
      cudaMemcpy(h.data(), st, 8 * (CTAS + 1), cudaMemcpyDeviceToHost);
      uint64_t first = *std::min_element(h.begin() + 1, h.end());
      if (it >= 5) gaps.push_back(((double)first - (double)h[0]) / 1e3);
    }
    std::sort(gaps.begin(), gaps.end());
    printf("mode %s spin %u ns  per-CTA %6zu B (total %5.1f MB): release gap median %.3f us  p10 %.3f p90 %.3f\n", mode ? "tma-smem" : "l2-prefetch",
           spin, bytes, bytes * CTAS / 1e6, gaps[gaps.size() / 2], gaps[gaps.size() / 10], gaps[gaps.size() * 9 / 10]);
  }
  cudaError_t e = cudaDeviceSynchronize(); printf("%s\n", cudaGetErrorString(e));
}
