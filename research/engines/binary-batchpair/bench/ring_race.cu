// Ring-release race reproducer: the drafter's hidden projection (k_gemm_fp4<4,2,32,store,WF1>, N=256,
// K=5120, R=448: 14 CTAs that all read the same 655 KB of L2-resident weights) is launched repeatedly on
// immutable inputs while a co-resident "LSU pressure" grid (small shared memory, so it shares SMs with
// the GEMM CTAs) streams global loads with L1 misses through the same memory pipe. Every output is
// compared bit-for-bit with an isolated reference run.
//
// Without a proxy fence before the consumer's empty-barrier arrive, ptxas issues the arrive while the
// warp's last LDS.128 of the stage are still queued; under pipe pressure the producer's bulk refill of
// that stage (L2 hit) can land first, and the late loads read the next K-block's weights. Reported per
// mismatching run: changed elements and the 16-feature m-tile (column % 64 / 16) they fall in.
//
// Build (from the repository root): nvcc -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -I src
//   -DH5_PROFILE_GEMM -DH5_PDL_EARLY_GEMM_RELEASE -DH5_NO_L2PF -DH5_WEIGHT_TMA_CACHE_POLICY=2
//   -o ring-race bench/ring_race.cu
// Usage: ring-race ITERATIONS PRESSURE_CTAS_PER_SM   (PRESSURE 0 = isolated control, -1 = PDL successor
//   grid of 448 head-prep-shaped CTAs after every GEMM, the drafter's actual launch pair)
#define H5_WARP_MICROSCOPE_STANDALONE
#include "fp4.cu"
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace h5;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); exit(2); } }

__global__ void k_pressure(const float4* __restrict__ src, size_t n4, volatile int* stop, float* sink) {
  __shared__ float red[256];
  float acc = 0.f;
  size_t i = (size_t)blockIdx.x * blockDim.x * 97 + threadIdx.x * 61;
  while (!*stop) {
#pragma unroll 8
    for (int k = 0; k < 64; k++) {
      i = (i + 1048573u) % n4;               // strided: almost every load misses L1
      const float4 v = __ldcg(src + i);
      acc += v.x + v.y + v.z + v.w;
    }
    red[threadIdx.x] = acc; __syncthreads(); acc += red[(threadIdx.x + 1) & 255]; __syncthreads();
  }
  if (acc == 1.2345f) sink[0] = acc;
}

// The drafter's real successor: the candidate-head prep (k_prep2: one CTA of 5 warps per row, constants
// loaded before griddepcontrol.wait), launched with programmatic serialization so its CTAs become
// resident beside the still-running GEMM.
__global__ void __launch_bounds__(160) k_successor(const float* __restrict__ nw, const float* __restrict__ sg, float* out) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, e0 = w * 1024 + lane * 32;
  float4 cn[8], cs[8];
#pragma unroll
  for (int i = 0; i < 8; i++) { cn[i] = *(const float4*)(nw + e0 + 4 * i); cs[i] = *(const float4*)(sg + e0 + 4 * i); }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float a = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) a += cn[i].x * cs[i].y + cn[i].z * cs[i].w;
  if (a == 1.2345f) out[blockIdx.x] = a;
}

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: ring-race ITERATIONS PRESSURE_CTAS_PER_SM\n"); return 2; }
  const int iters = atoi(argv[1]), per_sm = atoi(argv[2]);
  constexpr int FW = 4, WG = 2, TOKW = 32, WF = 1, N = 256, K = 5120, R = 448;
  using G = WsGeom<FW, WG, TOKW, WF, 0>;
  auto fn = k_gemm_fp4<FW, WG, TOKW, 0, WF, 0>;
  ck(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM), "smem attr");
  const int nkb = K / 128, NG = N / 32, tiles = (R + 63) / 64;
  std::vector<uint8_t> hW((size_t)nkb * NG * 2048), hS((size_t)nkb * NG * 256), hA((size_t)tiles * nkb * 64 * 64), hAS((size_t)tiles * nkb * 64 * 8);
  srand(12345);
  for (auto& b : hW) b = rand() & 0xff;
  for (auto& b : hA) b = rand() & 0xff;
  for (auto& b : hS) b = 0x30 + (rand() % 16);    // UE4M3 0.25..0.5, finite
  for (auto& b : hAS) b = 0x30 + (rand() % 16);
  std::vector<float> hAG(R, 1.f), hRS(N, 1.f);
  uint8_t *W, *S, *A, *AS; float *AG, *RS, *out, *sink; int *stop, *dstop;
  ck(cudaMalloc(&W, hW.size()), "W"); ck(cudaMalloc(&S, hS.size()), "S"); ck(cudaMalloc(&A, hA.size()), "A"); ck(cudaMalloc(&AS, hAS.size()), "AS");
  ck(cudaMalloc(&AG, R * 4), "AG"); ck(cudaMalloc(&RS, N * 4), "RS"); ck(cudaMalloc(&out, (size_t)R * N * 4), "out"); ck(cudaMalloc(&sink, 4), "sink");
  ck(cudaHostAlloc(&stop, 4, cudaHostAllocMapped), "stop"); ck(cudaHostGetDevicePointer(&dstop, stop, 0), "stop ptr");
  cudaMemcpy(W, hW.data(), hW.size(), cudaMemcpyHostToDevice); cudaMemcpy(S, hS.data(), hS.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(A, hA.data(), hA.size(), cudaMemcpyHostToDevice); cudaMemcpy(AS, hAS.data(), hAS.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(AG, hAG.data(), R * 4, cudaMemcpyHostToDevice); cudaMemcpy(RS, hRS.data(), N * 4, cudaMemcpyHostToDevice);
  const size_t n4 = (size_t)64 << 20;   // 1 GiB of float4 pressure source
  float4* src; ck(cudaMalloc(&src, n4 * 16), "src"); cudaMemset(src, 0, n4 * 16);
  cudaStream_t sg, sp; cudaStreamCreateWithFlags(&sg, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&sp, cudaStreamNonBlocking);
  const dim3 grid((R + TOKW - 1) / TOKW, (N / 32 + FW * WG - 1) / (FW * WG), 1), block((FW + 1) * 32);
  float *nw, *sgn, *sout; ck(cudaMalloc(&nw, 5120 * 4), "nw"); ck(cudaMalloc(&sgn, 5120 * 4), "sg"); ck(cudaMalloc(&sout, R * 4), "sout");
  cudaMemset(nw, 0, 5120 * 4); cudaMemset(sgn, 0, 5120 * 4);
  auto gemm = [&] {
    if (per_sm >= 0) { fn<<<grid, block, G::SMEM, sg>>>(A, AS, AG, R, K, W, S, N, out, N, RS); return; }
    cudaLaunchAttribute at{}; at.id = cudaLaunchAttributeProgrammaticStreamSerialization; at.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t c{}; c.blockDim = block; c.dynamicSmemBytes = G::SMEM; c.stream = sg; c.attrs = &at; c.numAttrs = 1; c.gridDim = grid;
    ck(cudaLaunchKernelEx(&c, fn, (const uint8_t*)A, (const uint8_t*)AS, (const float*)AG, R, K, (const uint8_t*)W, (const uint8_t*)S, N, out, N, (const float*)RS), "gemm pdl");
    c.gridDim = dim3(R); c.blockDim = dim3(160); c.dynamicSmemBytes = 0;
    ck(cudaLaunchKernelEx(&c, k_successor, (const float*)nw, (const float*)sgn, sout), "successor");
  };
  std::vector<float> ref((size_t)R * N), got((size_t)R * N);
  gemm(); ck(cudaStreamSynchronize(sg), "reference");
  cudaMemcpy(ref.data(), out, ref.size() * 4, cudaMemcpyDeviceToHost);
  int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  *(volatile int*)stop = 0;
  if (per_sm > 0) { k_pressure<<<sms * per_sm, 256, 0, sp>>>(src, n4, dstop, sink); ck(cudaGetLastError(), "pressure launch"); }
  long bad_runs = 0, bad_elems = 0, tile_hits[4] = {0, 0, 0, 0};
  for (int it = 0; it < iters; it++) {
    cudaMemsetAsync(out, 0xff, (size_t)R * N * 4, sg);
    gemm();
    ck(cudaMemcpyAsync(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost, sg), "copy");
    ck(cudaStreamSynchronize(sg), "gemm");
    long b = 0;
    for (size_t e = 0; e < got.size(); e++)
      if (memcmp(&got[e], &ref[e], 4)) { b++; tile_hits[(e % N) % 64 / 16]++; }
    if (b) { bad_runs++; bad_elems += b; }
  }
  *(volatile int*)stop = 1; ck(cudaDeviceSynchronize(), "drain");
  printf("{\"iterations\":%d,\"pressure_ctas_per_sm\":%d,\"mismatching_runs\":%ld,\"changed_elements\":%ld,\"mtile_hits\":[%ld,%ld,%ld,%ld]}\n",
         iters, per_sm, bad_runs, bad_elems, tile_hits[0], tile_hits[1], tile_hits[2], tile_hits[3]);
  return bad_runs ? 3 : 0;
}
