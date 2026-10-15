// L2 -> SM bandwidth on this card, the feed the large-row FP4 GEMMs need (~3.7 TB/s at MMA peak with
// 256x128 tiles). (1) bulk copies (TMA) of an L2-resident buffer into a shared ring, one CTA per SM with a
// single issuing lane like the GEMM producer, and 2 CTAs/SM; (2) plain 16-byte loads; (3) a 2-CTA cluster
// where each CTA bulk-copies half the tile and reads the partner's half over DSMEM (distributed shared memory).
//   l2-bw [MB]
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdint>
namespace cg = cooperative_groups;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { printf("%s: %s\n", w, cudaGetErrorString(e)); exit(1); } }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(b)), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned tx) { asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], %1; @!p bra W; }" :: "r"((unsigned)__cvta_generic_to_shared(b)), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk(void* dst, const void* src, unsigned n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(n), "r"((unsigned)__cvta_generic_to_shared(b)) : "memory");
}
// Each CTA streams `iters` chunks of CH bytes (wrapping over an L2-resident buffer) through a 4-deep ring.
template <int CH>
__global__ void k_bulk(const uint8_t* buf, size_t bytes, int iters, unsigned* sink) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ uint64_t full[4];
  if (threadIdx.x == 0) { for (int i = 0; i < 4; i++) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  size_t off = ((size_t)blockIdx.x * 7919 * CH) % bytes;
  unsigned acc = 0;
  if (threadIdx.x == 0) for (int i = 0; i < 4 && i < iters; i++) { mbar_expect(&full[i], CH); bulk(sm + i * CH, buf + (off + (size_t)i * CH) % bytes, CH, &full[i]); }
  for (int i = 0; i < iters; i++) {
    const int s = i & 3;
    mbar_wait(&full[s], (i >> 2) & 1);
    acc += ((const unsigned*)(sm + s * CH))[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0 && i + 4 < iters) { mbar_expect(&full[s], CH); bulk(sm + s * CH, buf + (off + (size_t)(i + 4) * CH) % bytes, CH, &full[s]); }
  }
  if (acc == 0x12345678) *sink = acc;
}
__global__ void k_ld(const uint4* buf, size_t n, int iters, unsigned* sink) {
  unsigned acc = 0; size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) % n;
  for (int it = 0; it < iters; it++) { uint4 v = buf[i]; acc += v.x ^ v.w; i = (i + (size_t)gridDim.x * blockDim.x) % n; }
  if (acc == 0x12345678) *sink = acc;
}
// Cluster of 2: each CTA bulk-loads half (CH/2) of a CH chunk into its own ring, then both read the whole chunk:
// own half locally, partner's half through DSMEM (map_shared_rank). L2 traffic per CTA is half the chunk.
template <int CH>
__global__ void __cluster_dims__(2, 1, 1) k_dsmem(const uint8_t* buf, size_t bytes, int iters, unsigned* sink) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ uint64_t full[4];
  cg::cluster_group cl = cg::this_cluster();
  const unsigned rank = cl.block_rank();
  if (threadIdx.x == 0) { for (int i = 0; i < 4; i++) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;"); }
  cl.sync();
  size_t off = ((size_t)(blockIdx.x / 2) * 7919 * CH) % bytes + rank * (CH / 2);
  unsigned acc = 0;
  const uint8_t* peer = cl.map_shared_rank(sm, rank ^ 1);
  if (threadIdx.x == 0) for (int i = 0; i < 4 && i < iters; i++) { mbar_expect(&full[i], CH / 2); bulk(sm + i * CH, buf + (off + (size_t)i * CH) % bytes, CH / 2, &full[i]); }
  for (int i = 0; i < iters; i++) {
    const int s = i & 3;
    mbar_wait(&full[s], (i >> 2) & 1);
    cl.sync();   // partner's half of stage s has landed too
    const unsigned* mine = (const unsigned*)(sm + s * CH); const unsigned* theirs = (const unsigned*)(peer + s * CH);
    for (int w = threadIdx.x; w < CH / 8; w += blockDim.x) acc += mine[w] ^ theirs[w];   // read both halves (CH/2 each)
    cl.sync();
    if (threadIdx.x == 0 && i + 4 < iters) { mbar_expect(&full[s], CH / 2); bulk(sm + s * CH, buf + (off + (size_t)(i + 4) * CH) % bytes, CH / 2, &full[s]); }
  }
  if (acc == 0x12345678) *sink = acc;
}
int main(int argc, char** argv) {
  const size_t mb = argc > 1 ? atoi(argv[1]) : 24, bytes = mb << 20;
  uint8_t* buf; unsigned* sink; ck(cudaMalloc(&buf, bytes), "malloc"); ck(cudaMalloc(&sink, 4), "sink"); cudaMemset(buf, 1, bytes);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto timeit = [&](const char* name, double total_bytes, auto launch) {
    launch(); ck(cudaDeviceSynchronize(), name);
    cudaEventRecord(a); for (int r = 0; r < 5; r++) launch(); cudaEventRecord(b); ck(cudaEventSynchronize(b), name);
    float ms; cudaEventElapsedTime(&ms, a, b); printf("%-44s %8.1f GB/s\n", name, 5 * total_bytes / (ms * 1e-3) / 1e9);
  };
  const int iters = 2000;
  constexpr int CH = 16384;
  ck(cudaFuncSetAttribute(k_bulk<CH>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * CH), "attr");
  for (int per : {1, 2})
    timeit(per == 1 ? "bulk 16KB stages, 1 CTA/SM" : "bulk 16KB stages, 2 CTA/SM", (double)nsm * per * iters * CH,
           [&] { k_bulk<CH><<<nsm * per, 128, 4 * CH>>>(buf, bytes, iters, sink); });
  timeit("ld.global.v4, 8 warps x 2/SM", (double)nsm * 2 * 256 * 16 * iters, [&] { k_ld<<<nsm * 2, 256>>>((const uint4*)buf, bytes / 16, iters, sink); });
  ck(cudaFuncSetAttribute(k_dsmem<CH>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * CH), "attr2");
  {
    cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 4 * CH;
    int clusters = 0; cudaOccupancyMaxActiveClusters(&clusters, (void*)k_dsmem<CH>, &cfg);
    printf("max active 2-CTA clusters: %d\n", clusters);
    // effective bytes delivered to SMs: each CTA consumes CH per iteration, half from L2 and half over DSMEM
    timeit("cluster2: half L2 + half DSMEM (delivered)", (double)nsm * iters * CH,
           [&] { k_dsmem<CH><<<nsm, 256, 4 * CH>>>(buf, bytes, iters, sink); });
  }
  return 0;
}
