// IMMA issue budget: m16n8k32 s8 throughput with ALU work interleaved per MMA and few warps per SM.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
__device__ __forceinline__ void mma(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// ALU: ALU ops per MMA spread over 4 independent float chains; CH independent MMA chains.
template <int ALU, int CH> __global__ void k(int iters, int* sink, float* fsink) {
  int c[CH][4] = {};
  unsigned a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x01020304u, a2 = a0 ^ 0x05060708u, a3 = a0 ^ 0x01000100u, b0 = a1 ^ 7, b1 = a2 ^ 9;
  float f0 = threadIdx.x, f1 = f0 + 1, f2 = f0 + 2, f3 = f0 + 3;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      mma(c[j], a0, a1, a2, a3, b0, b1);
#pragma unroll
      for (int u = 0; u < ALU; ++u) {
        if ((u & 3) == 0) f0 = fmaf(f0, 1.0001f, 0.5f);
        else if ((u & 3) == 1) f1 = fmaf(f1, 0.9999f, 0.25f);
        else if ((u & 3) == 2) f2 = fmaf(f2, 1.0002f, 0.125f);
        else f3 = fmaf(f3, 0.9998f, 0.0625f);
      }
    }
  }
  int s = 0;
#pragma unroll
  for (int j = 0; j < CH; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 0x12345678) sink[0] = s;
  if (f0 + f1 + f2 + f3 == 1.2345f) fsink[0] = f0;
}
template <int ALU, int CH> void run(int warps_per_sm, int sms) {
  int* sink; float* fs; CK(cudaMalloc(&sink, 4)); CK(cudaMalloc(&fs, 4));
  const int iters = 2048, threads = warps_per_sm * 32;
  k<ALU, CH><<<sms, threads>>>(8, sink, fs); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); std::vector<float> t;
  for (int rep = 0; rep < 5; ++rep) { CK(cudaEventRecord(a)); k<ALU, CH><<<sms, threads>>>(iters, sink, fs); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms); }
  std::sort(t.begin(), t.end());
  const double mmas = double(sms) * warps_per_sm * iters * CH;
  printf("{\"alu_per_mma\":%d,\"chains\":%d,\"warps_per_sm\":%d,\"mma_per_sm_per_ns\":%.4f,\"tops\":%.1f}\n", ALU, CH, warps_per_sm, mmas / sms / (t[2] * 1e6), mmas * 8192 / (t[2] * 1e-3) / 1e12);
  CK(cudaFree(sink)); CK(cudaFree(fs));
}
int main() {
  int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  for (int w : {4, 8, 16}) {
    run<0, 4>(w, sms); run<2, 4>(w, sms); run<4, 4>(w, sms); run<6, 4>(w, sms); run<8, 4>(w, sms); run<12, 4>(w, sms); run<16, 4>(w, sms);
  }
  run<6, 8>(8, sms); run<6, 16>(8, sms); run<8, 8>(8, sms);
  return 0;
}
