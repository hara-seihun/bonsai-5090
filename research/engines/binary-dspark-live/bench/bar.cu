// grid barrier latency on this GPU: plain atomic counter vs red.release/ld.acquire variants
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ void bar_a(unsigned* bar, unsigned nb) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* gen = bar + 1; const unsigned g = *gen; __threadfence();
    if (atomicAdd(bar, 1u) == nb - 1) { atomicExch(bar, 0u); __threadfence(); atomicAdd(bar + 1, 1u); }
    else { while (*gen == g) __nanosleep(32); }
    __threadfence();
  }
  __syncthreads();
}
__device__ __forceinline__ void bar_b(unsigned* bar, unsigned nb) {   // monotonic counter, no reset; acquire spin, no sleep
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.u32 %0, [%1], 1;" : "=r"(old) : "l"(bar) : "memory");
    const unsigned target = (old / nb + 1) * nb;
    unsigned cur;
    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(bar) : "memory"); } while ((int)(cur - target) < 0);
  }
  __syncthreads();
}
template <int V> __global__ void k(unsigned* bar, int iters) { for (int i = 0; i < iters; i++) { if (V == 0) bar_a(bar, gridDim.x); else bar_b(bar + 4, gridDim.x); } }
int main() {
  unsigned* bar; cudaMalloc(&bar, 64); cudaMemset(bar, 0, 64);
  int n; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  for (int v = 0; v < 2; v++) for (int nb : {n, n / 2, 32}) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int iters = 2000;
    void* args[] = {&bar, (void*)&iters};
    if (v == 0) cudaLaunchCooperativeKernel((void*)k<0>, nb, 256, args, 0, 0); else cudaLaunchCooperativeKernel((void*)k<1>, nb, 256, args, 0, 0);
    cudaEventRecord(e0);
    if (v == 0) cudaLaunchCooperativeKernel((void*)k<0>, nb, 256, args, 0, 0); else cudaLaunchCooperativeKernel((void*)k<1>, nb, 256, args, 0, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("variant %d, %d CTAs: %.3f us per barrier (%s)\n", v, nb, ms * 1e3 / iters, cudaGetErrorString(cudaGetLastError()));
  }
}
