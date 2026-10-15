// Latency from a predecessor grid's last CTA exit to its PDL dependent's griddepcontrol.wait return,
// as a function of both grids' shapes (CTAs, threads, dynamic shared memory).
//   P: grid gp x tp threads, sp bytes smem; every CTA waits on its own predecessor, triggers, spins ~us, exits.
//   D: grid gd x td threads, sd bytes smem, launched with programmatic serialization after P; records the
//      wait return. Reported: D's wait-return (min/median/max) minus P's last exit, and D's CTA entry spread.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
__device__ __forceinline__ uint64_t gt() { uint64_t v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v) :: "memory"); return v; }
__global__ void kP(uint64_t* pexit, int spin_ns) {
  extern __shared__ uint8_t sm[];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint64_t t0 = gt();
  if (threadIdx.x == 0) sm[0] = 1;
  while (gt() - t0 < (uint64_t)spin_ns) {}
  __syncthreads();
  if (threadIdx.x == 0) atomicMax((unsigned long long*)pexit, (unsigned long long)gt());
}
__global__ void kD(uint64_t* rec, const uint8_t* w, unsigned long long pre) {
  extern __shared__ uint8_t sm[];
  const uint64_t e = gt();
  if (threadIdx.x == 0) sm[0] = 2;
  if (threadIdx.x == 0 && pre) {   // pre-wait weight traffic: this CTA's share of `pre` bytes
    const unsigned long long per = (pre / gridDim.x) & ~15ULL;
    for (unsigned long long o = 0; o < per; o += 16384)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w + per * blockIdx.x + o), "r"((unsigned)min(16384ULL, per - o)) : "memory");
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint64_t wr = gt();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) { rec[2 * blockIdx.x] = e; rec[2 * blockIdx.x + 1] = wr; }
}
int main() {
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "st");
  uint64_t *pexit, *rec; ck(cudaMalloc(&pexit, 8), "m"); ck(cudaMalloc(&rec, 2 * 4096 * 8), "m");
  uint8_t* wbuf; const size_t wcap = 1ULL << 30; ck(cudaMalloc(&wbuf, wcap), "m"); ck(cudaMemset(wbuf, 1, wcap), "m");
  size_t woff = 0;
  ck(cudaFuncSetAttribute(kP, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024), "a");
  ck(cudaFuncSetAttribute(kD, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024), "a");
  auto launch = [&](void* f, int g, int t, int s, bool pdl, void** args) {
    cudaLaunchConfig_t c = {}; c.gridDim = g; c.blockDim = t; c.dynamicSmemBytes = s; c.stream = st;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = at; c.numAttrs = pdl ? 1 : 0;
    ck(cudaLaunchKernelExC(&c, f, args), "launch");
  };
  printf("%4s %5s %4s %6s | %5s %4s %6s | %8s %8s %8s | %8s %8s\n", "MB", "gp", "tp", "sp", "gd", "td", "sd", "wait-min", "wait-med", "wait-max", "entry-min", "entry-max");
  const int gps[] = {8, 136}, sps[] = {16384}, gds[] = {136, 340}, tds[] = {288}, sds[] = {29 * 1024};
  for (unsigned long long pre : {0ULL, 2ULL << 20, 5ULL << 20, 12ULL << 20, 24ULL << 20})
  for (int gp : gps) for (int sp : sps) for (int gd : gds) for (int td : tds) for (int sd : sds) {
    std::vector<double> wmin, wmed, wmax, emin, emax;
    for (int rep = 0; rep < 12; rep++) {
      ck(cudaMemsetAsync(pexit, 0, 8, st), "ms");
      int spin = 2000; const uint8_t* wp = wbuf + woff; woff = (woff + (32ULL << 20)) % (wcap - (32ULL << 20));
      void* ap[] = {&pexit, &spin}; void* ad[] = {&rec, &wp, &pre};
      cudaGraph_t g; cudaGraphExec_t ge;
      ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap");
      launch((void*)kP, gp, gp == 8 ? 640 : 32, sp ? sp : 16, true, ap);
      launch((void*)kD, gd, td, sd, true, ad);
      ck(cudaStreamEndCapture(st, &g), "end"); ck(cudaGraphInstantiate(&ge, g, 0), "inst");
      ck(cudaGraphLaunch(ge, st), "run"); ck(cudaStreamSynchronize(st), "sync");
      uint64_t pe; std::vector<uint64_t> r(2 * gd);
      ck(cudaMemcpy(&pe, pexit, 8, cudaMemcpyDeviceToHost), "c"); ck(cudaMemcpy(r.data(), rec, 16 * gd, cudaMemcpyDeviceToHost), "c");
      cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
      if (rep < 2) continue;
      std::vector<double> w, e;
      for (int i = 0; i < gd; i++) { e.push_back(((double)r[2 * i] - (double)pe) / 1e3); w.push_back(((double)r[2 * i + 1] - (double)pe) / 1e3); }
      std::sort(w.begin(), w.end()); std::sort(e.begin(), e.end());
      wmin.push_back(w.front()); wmed.push_back(w[w.size() / 2]); wmax.push_back(w.back()); emin.push_back(e.front()); emax.push_back(e.back());
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    printf("%4llu %5d %4d %6d | %5d %4d %6d | %8.2f %8.2f %8.2f | %8.2f %8.2f\n", pre >> 20, gp, gp == 8 ? 640 : 32, sp, gd, td, sd, med(wmin), med(wmed), med(wmax), med(emin), med(emax));
  }
  return 0;
}
