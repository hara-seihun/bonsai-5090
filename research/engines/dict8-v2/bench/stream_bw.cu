// Read-bandwidth ceiling of weight-streaming patterns on this GPU (no compute):
//   ldg      : grid-stride LDG.128 over the buffer (reference)
//   tma-seq  : G CTAs, each streams its contiguous 1/G of the buffer with 1D bulk copies of C bytes
//              through a D-deep mbarrier ring (one consumer warp releases slots)
//   tma-kb   : the current weight order: tile-major units of C bytes at stride NGall*832 (GU shape)
// Each pattern runs as a graph of L launches over L distinct buffers of S bytes (PDL-chained, like the
// verifier's projections), reported as GB/s over all bytes.
#include "fp4_ops.cuh"
#include "kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace h5;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }

__global__ void k_ldg(const uint4* a, size_t n, unsigned* out) {
  pdl_wait(); pdl_launch();
  unsigned x = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const uint4 v = __ldcs(a + i); x ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (x == 0x12345678u) out[0] = x;
}
template<int READS> __global__ void k_rw(const uint4* a, const uint4* b, uint4* out, size_t n) {
  pdl_wait(); pdl_launch();
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) {
    uint4 v=__ldcs(a+i);
    if constexpr(READS>=2) { const uint4 w=__ldcs(b+i); v.x^=w.x;v.y^=w.y;v.z^=w.z;v.w^=w.w; }
    if constexpr(READS==4) {
      const uint4 c=__ldcs(b+n+i), d=__ldcs(b+2*n+i);
      v.x^=c.x^d.x;v.y^=c.y^d.y;v.z^=c.z^d.z;v.w^=c.w^d.w;
    }
    __stcs(out+i,v);
  }
}
// strided: unit u of this CTA reads C bytes at base + (u % per_tile) * stride + (u / per_tile) * C
template <bool STRIDED, bool ACT>
__global__ void k_tma(const uint8_t* buf, size_t total, int C, int D, size_t stride, int per_tile, int prefetch, unsigned* out, const uint8_t* act) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int SC = ACT ? C + 576 : C;
  uint64_t* full = (uint64_t*)(smem + (size_t)D * SC);
  uint64_t* empty = full + D;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t units = total / C;
  const size_t ub = units * blockIdx.x / gridDim.x, ue = units * (blockIdx.x + 1) / gridDim.x;
  const int n = (int)(ue - ub);
  if (threadIdx.x == 0) { for (int s = 0; s < D; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); } asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  auto addr = [&](size_t u) -> const uint8_t* {
    if (!STRIDED) return buf + u * C;
    return buf + (u % per_tile) * stride + (u / per_tile) * C;
  };
  if (warp == 1) {
    if (lane == 0) {
      const int npre = min(n, D);
      auto issue = [&](int i, int s) {
        mbar_expect(&full[s], ACT ? C + 576 : C);
        bulk_g2s(smem + (size_t)s * SC, addr(ub + i), C, &full[s]);
      };
      auto acts = [&](int i, int s) {
        const int kb = (int)((ub + i) % 40);
        bulk_g2s(smem + (size_t)s * SC + C, act + (size_t)kb * 4096, 512, &full[s]);
        bulk_g2s(smem + (size_t)s * SC + C + 512, act + 40 * 4096 + (size_t)kb * 512, 64, &full[s]);
      };
      for (int i = 0; i < npre; i++) issue(i, i);
      pdl_wait();
      for (int i = 0; i < n; i++) {
        const int s = i % D;
        if (i >= npre) {
          if (prefetch && i + prefetch < n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(addr(ub + i + prefetch)), "r"(C) : "memory");
          mbar_wait(&empty[s], ((i / D) & 1) ^ 1);
          issue(i, s);
        }
        if (ACT) acts(i, s);
      }
    }
    return;
  }
  pdl_wait(); pdl_launch();
  unsigned x = 0;
  for (int i = 0; i < n; i++) {
    const int s = i % D;
    mbar_wait(&full[s], (i / D) & 1);
    x ^= smem[(size_t)s * SC + lane];
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
  }
  if (x == 0x12345678u) out[0] = x;
}

int main() {
  int nsm = 0; ck(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0), "attr");
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "st");
  const size_t cap = 5ULL << 30;
  uint8_t* buf; ck(cudaMalloc(&buf, cap), "malloc"); ck(cudaMemset(buf, 1, cap), "memset");
  unsigned* out; ck(cudaMalloc(&out, 64), "malloc");
  auto time_graph = [&](auto body) {
    cudaGraph_t g; cudaGraphExec_t ge;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap"); body(); ck(cudaStreamEndCapture(st, &g), "end");
    ck(cudaGraphInstantiate(&ge, g, 0), "inst");
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 2; i++) ck(cudaGraphLaunch(ge, st), "w");
    std::vector<float> v;
    for (int i = 0; i < 7; i++) { cudaEventRecord(e0, st); ck(cudaGraphLaunch(ge, st), "r"); cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "s"); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms); }
    std::sort(v.begin(), v.end()); cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return v[3] * 1e-3;
  };
  struct Case { size_t S; int L; const char* name; };
  const Case cases[] = {{36ULL << 20, 64, "36MB x64 (GU)"}, {6ULL << 20, 128, "6MB x128 (out)"}, {256ULL << 20, 8, "256MB x8 (head)"}};
  for (const Case& cs : cases) {
    const double bytes = (double)cs.S * cs.L;
    printf("\n== %s ==\n", cs.name);
    for (int grid : {nsm * 16}) {
      const double t = time_graph([&] { for (int l = 0; l < cs.L; l++) pdl(k_ldg, dim3(grid), dim3(256), 0, st, (const uint4*)(buf + (size_t)l * cs.S), cs.S / 16, out); });
      printf("  ldg grid %d                         %7.0f GB/s\n", grid, bytes / t / 1e9);
    }
    for (int strided = 0; strided < 2; strided++)
     for (int actv = 0; actv < 2; actv++)
      for (int C : {6656, 8192, 16384, 32768})
        for (int D : {2, 4, 8})
          for (int per : {1, 2})
            for (int pf : {0, 4}) {
              const size_t smem = (size_t)D * (C + (actv ? 576 : 0)) + 16 * D;
              if (smem > 99 * 1024 || (per == 2 && smem > 48 * 1024)) continue;
              if (strided && C != 6656 && C != 8192) continue;
              if (!strided || C != 6656 || pf) continue;
              auto f = actv ? k_tma<true, true> : k_tma<true, false>;
              ck(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "attr");
              const size_t S = cs.S / C * C;
              const int per_tile = 40;   // K blocks per tile (K = 5120)
              const size_t stride = (S / per_tile) / 128 * 128;
              const double t = time_graph([&] { for (int l = 0; l < cs.L; l++) pdl(f, dim3(nsm * per), dim3(64), smem, st, (const uint8_t*)(buf + (size_t)l * cs.S), S, C, D, stride, per_tile, pf, out, buf + cap - (64ULL << 20)); });
              printf("  tma-%s%s C %5d D %d x%d pf %d  %7.0f GB/s\n", strided ? "kb " : "seq", actv ? "+act" : "    ", C, D, per, pf, (double)S * cs.L / t / 1e9);
            }
  }
  const size_t S=1ULL<<30;
  const uint4* a=(const uint4*)buf; const uint4* b=(const uint4*)(buf+S); uint4* dst=(uint4*)(buf+4*S);
  for(int reads:{1,2,4}) for(int grid:{nsm*2,nsm*8,nsm*16}) {
    auto f=reads==1?k_rw<1>:reads==2?k_rw<2>:k_rw<4>;
    const double t=time_graph([&] { for(int l=0;l<8;l++) pdl(f,dim3(grid),dim3(256),0,st,a,b,dst,S/16); });
    printf("rw reads=%d writes=1 grid=%d bytes_per_graph=%llu seconds=%.9g aggregate_GBs=%.3f\n",reads,grid,(unsigned long long)(S*(reads+1)*8),t,(double)S*(reads+1)*8/t/1e9);
  }
  ck(cudaFree(out),"free"); ck(cudaFree(buf),"free"); ck(cudaStreamDestroy(st),"stream destroy");
  return 0;
}
