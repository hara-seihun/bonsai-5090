// Register-resident issue rate: dense vs 2:4-sparse INT8 and dense vs sparse NVFP4 on sm_120.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
template <int KIND> __global__ void k(int iters, int* sink) {
  unsigned a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x5a5a5a5au, a2 = a0 + 7, a3 = a0 * 3, b0 = a0 ^ 0x33333333u, b1 = a1 + 11, b2 = b0 + 5, b3 = b1 * 7;
  unsigned meta = 0x44444444u;  // 2:4 index pairs (0,1) everywhere
  int c[8][4] = {}; float f[8][4] = {};
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
      if (KIND == 0) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(c[ch][0]), "+r"(c[ch][1]), "+r"(c[ch][2]), "+r"(c[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      if (KIND == 1) asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12,0x0;"
        : "+r"(c[ch][0]), "+r"(c[ch][1]), "+r"(c[ch][2]), "+r"(c[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));
      if (KIND == 2) { unsigned sa = 0x38383838, sb = 0x38383838; uint16_t z = 0;
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
        : "+f"(f[ch][0]), "+f"(f[ch][1]), "+f"(f[ch][2]), "+f"(f[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "h"(z)); }
      if (KIND == 3) { unsigned sa = 0x38383838, sb = 0x38383838; uint16_t z = 0;
        asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%0,%1,%2,%3},%12,0x0,%13,{%15,%15},%14,{%15,%15};"
        : "+f"(f[ch][0]), "+f"(f[ch][1]), "+f"(f[ch][2]), "+f"(f[ch][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta), "r"(sa), "r"(sb), "h"(z)); }
    }
  }
  int s = 0;
#pragma unroll
  for (int ch = 0; ch < 8; ch++) s += c[ch][0] + c[ch][1] + c[ch][2] + c[ch][3] + (int)(f[ch][0] + f[ch][1] + f[ch][2] + f[ch][3]);
  if (s == 0x12345) sink[0] = s;
}
int main() {
  int* sink; cudaMalloc(&sink, 4); cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); const int sms = p.multiProcessorCount;
  const char* names[] = {"int8 dense k32", "int8 2:4 sparse k64", "nvfp4 dense k64", "nvfp4 4:8 sparse k128"};
  const double macs[] = {16 * 8 * 32, 16 * 8 * 64, 16 * 8 * 64, 16 * 8 * 128};
  for (int kind = 0; kind < 4; kind++) {
    const int iters = 4000, warps = 16; float best = 1e9;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 6; r++) {
      cudaEventRecord(e0);
      dim3 g(sms * 2), b(warps * 16);
      if (kind == 0) k<0><<<g, b>>>(iters, sink); else if (kind == 1) k<1><<<g, b>>>(iters, sink); else if (kind == 2) k<2><<<g, b>>>(iters, sink); else k<3><<<g, b>>>(iters, sink);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (r && ms < best) best = ms;
    }
    cudaError_t err = cudaGetLastError(); if (err) { printf("%s: %s\n", names[kind], cudaGetErrorString(err)); continue; }
    double instr = (double)sms * 2 * (warps / 2) * iters * 8, rate = instr / (best * 1e-3);
    printf("%-24s %.2f Ginstr/s  %.0f T logical MAC/s\n", names[kind], rate / 1e9, rate * macs[kind] / 1e12);
  }
}
