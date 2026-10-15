// PDL release latency with the dependent's pre-wait traffic shaped like the production GEMMs:
// cp.async.bulk global->shared copies into an mbarrier ring (what k_gemm_fp4's producer issues), versus
// cp.async.bulk.prefetch.L2 (bench/pdl_release.cu) and none. Also varies when the dependent's CTAs
// arrive (predecessor triggers at its start, or after `trig_ns`).
//   P: 136 CTAs x 128 threads; waits on its predecessor, spins trig_ns, triggers dependents, spins to
//      spin_ns total, writes 64 KB in total, exits; records its last exit.
//   D: 340 CTAs x 288 threads, 29 KB dynamic smem: thread 0 issues `stages` copies of 6656 B, then
//      griddepcontrol.wait; records entry, issue end, wait return and copy completion.
// Prints per configuration: D wait return minus P last exit (min / median / max over CTAs), median D entry
// and copy completion relative to P last exit, all in microseconds, median over 50 repetitions.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
__device__ __forceinline__ uint64_t gt() { uint64_t v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v) :: "memory"); return v; }
constexpr int STAGE = 6656, NP = 136, ND = 340, TD = 288, MAXST = 4;
__global__ void kP(uint64_t* pexit, float* sink, int trig_ns, int spin_ns) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint64_t t0 = gt();
  if (trig_ns == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  else { while (gt() - t0 < (uint64_t)trig_ns) {} asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
  while (gt() - t0 < (uint64_t)spin_ns) {}
  for (int i = threadIdx.x; i < 120; i += blockDim.x) sink[blockIdx.x * 120 + i] = (float)i;   // ~64 KB of stores
  __syncthreads();
  if (threadIdx.x == 0) atomicMax((unsigned long long*)pexit, (unsigned long long)gt());
}
__device__ __forceinline__ unsigned su(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__global__ void kD(uint64_t* rec, const uint8_t* w, int mode, int stages) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) uint64_t bar[MAXST];
  const uint64_t e = gt();
  uint64_t issued = e;
  if (threadIdx.x == 0) {
    for (int s = 0; s < MAXST; s++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(su(&bar[s])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const uint8_t* src = w + (size_t)blockIdx.x * MAXST * STAGE;
    for (int s = 0; s < stages; s++) {
      if (mode == 1) {
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(su(&bar[s])), "r"(STAGE) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(su(sm + s * STAGE)), "l"(src + s * STAGE), "r"(STAGE), "r"(su(&bar[s])) : "memory");
      } else if (mode == 2) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + s * STAGE), "r"(STAGE) : "memory");
      }
    }
    issued = gt();
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint64_t wr = gt();
  uint64_t done = wr;
  if (threadIdx.x == 0 && mode == 1) {
    for (int s = 0; s < stages; s++) {
      unsigned ok = 0;
      while (!ok) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(su(&bar[s])) : "memory");
    }
    done = gt();
  }
  if (threadIdx.x == 0) { rec[4 * blockIdx.x] = e; rec[4 * blockIdx.x + 1] = issued; rec[4 * blockIdx.x + 2] = wr; rec[4 * blockIdx.x + 3] = done; }
}
int main() {
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "st");
  uint64_t *pexit, *rec; float* sink; uint8_t* w;
  ck(cudaMalloc(&pexit, 8), "m"); ck(cudaMalloc(&rec, 4 * ND * 8), "m"); ck(cudaMalloc(&sink, NP * 120 * 4), "m");
  const size_t wbytes = (size_t)ND * MAXST * STAGE * 64;   // 64 disjoint regions so repetitions miss L2
  ck(cudaMalloc(&w, wbytes), "m"); ck(cudaMemset(w, 1, wbytes), "m");
  ck(cudaFuncSetAttribute(kD, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXST * STAGE), "attr");
  uint8_t* flush; ck(cudaMalloc(&flush, 256 << 20), "m");
  printf("mode stages MB trig_us | wait-ret min med max | D entry med | copy done med   (us rel. P last exit)\n");
  const char* mname[3] = {"none", "bulk-smem", "prefetch-L2"};
  for (int psm : {0, 16384})
  for (int trig : {0, 1500})
    for (int mode : {0, 1, 2})
      for (int stages : {1, 2, 4}) {
        if (mode == 0 && stages > 1) continue;
        std::vector<double> mn, md, mx, en, dn;
        for (int rep = 0; rep < 52; rep++) {
          ck(cudaMemsetAsync(flush, rep, 256 << 20, st), "flush");   // evict L2
          ck(cudaMemsetAsync(pexit, 0, 8, st), "z");
          cudaLaunchConfig_t c = {}; cudaLaunchAttribute at[1];
          at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
          c.attrs = at; c.numAttrs = 1; c.stream = st;
          c.gridDim = NP; c.blockDim = 128; c.dynamicSmemBytes = psm;
          ck(cudaLaunchKernelEx(&c, kP, pexit, sink, trig, 3000), "P");
          c.gridDim = ND; c.blockDim = TD; c.dynamicSmemBytes = MAXST * STAGE;
          ck(cudaLaunchKernelEx(&c, kD, rec, (const uint8_t*)(w + (size_t)(rep % 64) * ND * MAXST * STAGE), mode, stages), "D");
          ck(cudaStreamSynchronize(st), "sync");
          if (rep < 2) continue;
          uint64_t pe; std::vector<uint64_t> r(4 * ND);
          ck(cudaMemcpy(&pe, pexit, 8, cudaMemcpyDeviceToHost), "c"); ck(cudaMemcpy(r.data(), rec, r.size() * 8, cudaMemcpyDeviceToHost), "c");
          std::vector<double> wv, ev, dv;
          for (int i = 0; i < ND; i++) { wv.push_back((double)(int64_t)(r[4 * i + 2] - pe) / 1e3); ev.push_back((double)(int64_t)(r[4 * i] - pe) / 1e3); dv.push_back((double)(int64_t)(r[4 * i + 3] - pe) / 1e3); }
          std::sort(wv.begin(), wv.end()); std::sort(ev.begin(), ev.end()); std::sort(dv.begin(), dv.end());
          mn.push_back(wv.front()); md.push_back(wv[ND / 2]); mx.push_back(wv.back()); en.push_back(ev[ND / 2]); dn.push_back(dv[ND / 2]);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        printf("psm%5d %-11s %d %5.1f %5.1f | %6.2f %6.2f %6.2f | %6.2f | %6.2f\n", psm, mname[mode], stages, mode ? ND * stages * STAGE / 1e6 : 0.0, trig / 1e3,
               med(mn), med(md), med(mx), med(en), med(dn));
      }
  return 0;
}
