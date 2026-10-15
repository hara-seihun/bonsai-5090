// Hardware keyholes for the B32 binary projection roof: tensor-pipe rates by MMA kind,
// DRAM streaming bandwidth and L2-resident read bandwidth on the rented RTX 5090.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

constexpr int CHAINS = 8;

template <int KIND> __device__ __forceinline__ void mma_op(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  if constexpr (KIND == 0) {  // s8 m16n8k32
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else if constexpr (KIND == 1) {  // u8 x s8 m16n8k32
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else if constexpr (KIND == 2) {  // e4m3 m16n8k32 f32 accumulate
    float* f = reinterpret_cast<float*>(c);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else if constexpr (KIND == 3) {  // f8f6f4 kind with e2m1 operands m16n8k32
    float* f = reinterpret_cast<float*>(c);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else if constexpr (KIND == 4) {  // native mxf4nvf4 m16n8k64 blockscale (ue4m3, 4X)
    float* f = reinterpret_cast<float*>(c);
    const unsigned sa = 0x38383838u, sb = 0x38383838u; const uint16_t z = 0;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
                 : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "h"(z));
  } else if constexpr (KIND == 5) {  // f16 m16n8k16 f32 accumulate (a2,a3 unused per spec size: 4 regs a, 2 regs b)
    float* f = reinterpret_cast<float*>(c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else if constexpr (KIND == 6) {  // f16 m16n8k16 f16 accumulate
    unsigned* h = reinterpret_cast<unsigned*>(c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(h[0]), "+r"(h[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}
template <int KIND> __global__ void mma_rate(int iters, int* sink) {
  int c[CHAINS][4] = {};
  unsigned a0 = threadIdx.x * 0x01010101u & 0x0f0f0f0fu, a1 = a0 ^ 0x01020304u, a2 = a0 ^ 0x05060708u, a3 = a0 ^ 0x01000100u;
  unsigned b0 = a1 ^ 0x02020202u, b1 = a2 ^ 0x03030303u;
  if constexpr (KIND >= 2 && KIND <= 4) { a0 &= 0x77777777u; a1 &= 0x77777777u; a2 &= 0x77777777u; a3 &= 0x77777777u; b0 &= 0x77777777u; b1 &= 0x77777777u; }
  if constexpr (KIND >= 5) { a0 = a1 = a2 = a3 = b0 = b1 = 0x3c003c00u; }
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) mma_op<KIND>(c[j], a0, a1, a2, a3, b0, b1);
  }
  int s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 0x12345678) sink[0] = s;
}

__global__ void stream_read(const int4* __restrict__ p, size_t n, int* sink) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    int4 v = __ldcs(p + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7f7f7f7f) sink[0] = 1;
}
__global__ void l2_read(const int4* __restrict__ p, size_t n, int reps, int* sink) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (int r = 0; r < reps; ++r)
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
      int4 v = __ldcg(p + ((i + size_t(r) * 7919) % n)); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7f7f7f7f) sink[0] = 1;
}

template <int KIND> static void run_mma(const char* name, int sms, double ops_per_mma) {
  int* sink; CK(cudaMalloc(&sink, 4));
  const int iters = 4096, blocks = sms * 4, threads = 256;  // 8 warps per block, 32 warps per SM
  mma_rate<KIND><<<blocks, threads>>>(16, sink); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> t;
  for (int rep = 0; rep < 5; ++rep) {
    CK(cudaEventRecord(a)); mma_rate<KIND><<<blocks, threads>>>(iters, sink); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  const double mmas = double(blocks) * (threads / 32) * iters * CHAINS;
  printf("{\"kind\":\"mma\",\"name\":\"%s\",\"ms\":%.4f,\"tops\":%.1f,\"mma_per_sm_per_ns\":%.4f}\n", name, t[2], mmas * ops_per_mma / (t[2] * 1e-3) / 1e12,
         mmas / sms / (t[2] * 1e6));
  CK(cudaFree(sink));
}

int main() {
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0, mem_khz = 0; CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0)); CK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
  printf("{\"kind\":\"device\",\"name\":\"%s\",\"sms\":%d,\"l2_bytes\":%d,\"smem_per_sm\":%zu,\"smem_per_block_optin\":%zu,\"regs_per_sm\":%d,\"clock_khz\":%d,\"mem_khz\":%d,\"bus_bits\":%d}\n",
         prop.name, prop.multiProcessorCount, prop.l2CacheSize, prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlockOptin, prop.regsPerMultiprocessor, clock_khz, mem_khz, prop.memoryBusWidth);
  const int sms = prop.multiProcessorCount;
  run_mma<0>("s8.s8.m16n8k32", sms, 2.0 * 16 * 8 * 32);
  run_mma<1>("u8.s8.m16n8k32", sms, 2.0 * 16 * 8 * 32);
  run_mma<2>("e4m3.m16n8k32.f32", sms, 2.0 * 16 * 8 * 32);
  run_mma<3>("f8f6f4.e2m1.m16n8k32.f32", sms, 2.0 * 16 * 8 * 32);
  run_mma<4>("mxf4nvf4.e2m1.m16n8k64.ue4m3x4", sms, 2.0 * 16 * 8 * 64);
  run_mma<5>("f16.m16n8k16.f32acc", sms, 2.0 * 16 * 8 * 16);
  run_mma<6>("f16.m16n8k16.f16acc", sms, 2.0 * 16 * 8 * 16);
  // DRAM stream
  {
    size_t bytes = size_t(2) << 30; int4* p; int* sink; CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&sink, 4)); CK(cudaMemset(p, 1, bytes));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int threads : {256, 512}) for (int per_sm : {2, 4, 8}) {
      std::vector<float> t;
      for (int rep = 0; rep < 5; ++rep) {
        CK(cudaEventRecord(a)); stream_read<<<sms * per_sm, threads>>>(p, bytes / 16, sink); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
      }
      std::sort(t.begin(), t.end());
      printf("{\"kind\":\"dram_read\",\"bytes\":%zu,\"blocks_per_sm\":%d,\"threads\":%d,\"ms\":%.4f,\"gbps\":%.1f}\n", bytes, per_sm, threads, t[2], bytes / (t[2] * 1e-3) / 1e9);
    }
    // L2-resident reads: 24 MiB buffer, many repetitions
    size_t l2b = size_t(24) << 20; int reps = 40;
    for (int per_sm : {4, 8}) {
      std::vector<float> t;
      for (int rep = 0; rep < 5; ++rep) {
        CK(cudaEventRecord(a)); l2_read<<<sms * per_sm, 256>>>(p, l2b / 16, reps, sink); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
      }
      std::sort(t.begin(), t.end());
      printf("{\"kind\":\"l2_read\",\"bytes\":%zu,\"reps\":%d,\"blocks_per_sm\":%d,\"ms\":%.4f,\"gbps\":%.1f}\n", l2b, reps, per_sm, t[2], double(l2b) * reps / (t[2] * 1e-3) / 1e9);
    }
    CK(cudaFree(p)); CK(cudaFree(sink));
  }
  return 0;
}
