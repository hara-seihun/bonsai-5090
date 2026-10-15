// Register-resident tensor-core issue-rate probe on sm_120: INT8 m16n8k32, NVFP4 m16n8k64,
// and 1-bit m16n8k256 AND+POPC. Operands live in registers; 8 independent accumulator chains
// per warp. Reports raw instruction rate and "logical" rates.
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>

template <int KIND>
__global__ void k(int iters, int* sink) {
  unsigned a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x5a5a5a5au, a2 = a0 + 7, a3 = a0 * 3, b0 = a0 ^ 0x33333333u, b1 = a1 + 11;
  int c[8][4] = {};
  float f[8][4] = {};
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
      if (KIND == 0) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+r"(c[ch][0]), "+r"(c[ch][1]), "+r"(c[ch][2]), "+r"(c[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if (KIND == 1) {
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+r"(c[ch][0]), "+r"(c[ch][1]), "+r"(c[ch][2]), "+r"(c[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if (KIND == 2) {
        unsigned sa = 0x38383838, sb = 0x38383838; uint16_t z = 0;
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
                     : "+f"(f[ch][0]), "+f"(f[ch][1]), "+f"(f[ch][2]), "+f"(f[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "h"(z));
      } else if (KIND == 3) {
        asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+r"(c[ch][0]), "+r"(c[ch][1]), "+r"(c[ch][2]), "+r"(c[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  int s = 0;
#pragma unroll
  for (int ch = 0; ch < 8; ch++) s += c[ch][0] + c[ch][1] + c[ch][2] + c[ch][3] + (int)(f[ch][0] + f[ch][1] + f[ch][2] + f[ch][3]);
  if (s == 0x12345) sink[0] = s;
}

int main() {
  int* sink; cudaMalloc(&sink, 4);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  const int sms = p.multiProcessorCount;
  const char* names[] = {"int8 m16n8k32", "b1 and.popc m16n8k256", "nvfp4 m16n8k64", "int4 u4.s4 m16n8k64"};
  const double macs[] = {16 * 8 * 32, 16 * 8 * 256, 16 * 8 * 64, 16 * 8 * 64};
  for (int kind = 0; kind < 4; kind++) {
    for (int warps : {8, 16, 32}) {
      const int iters = 4000;
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      auto launch = [&]() {
        dim3 g(sms * 2), b(warps * 32 / 2);
        if (kind == 0) k<0><<<g, b>>>(iters, sink); else if (kind == 1) k<1><<<g, b>>>(iters, sink);
        else if (kind == 2) k<2><<<g, b>>>(iters, sink); else k<3><<<g, b>>>(iters, sink);
      };
      launch(); cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) { printf("%s: %s\n", names[kind], cudaGetErrorString(err)); break; }
      float best = 1e9;
      for (int r = 0; r < 5; r++) { cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms; }
      double instr = (double)sms * 2 * (warps / 2) * iters * 8;
      double rate = instr / (best * 1e-3);
      printf("%-24s warps/SM %2d  %.3f ms  %.2f Ginstr/s  %.1f T-ops/s (k-element MACs)\n", names[kind], warps, best, rate / 1e9, rate * macs[kind] / 1e12);
    }
  }
  return 0;
}
