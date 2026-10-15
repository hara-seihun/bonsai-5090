// PTX helpers shared by the FP4 GEMM kernels: cp.async, ldmatrix, the block-scaled FP4 MMA,
// mbarriers and 1D bulk (TMA) copies with the selectable weight cache hint.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace h5 {
__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* smem, const void* gmem, bool pred) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(s), "l"(gmem), "r"(pred ? 8 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  // The pointer operand alone does not tell the C++ compiler this instruction reads
  // shared memory. Keep the read inside the acquired ring stage's lifetime.
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s) : "memory");
}
// Activation row scales are produced by preceding kernels, unlike immutable weight
// scales. A coherent volatile load keeps them out of ld.global.nc/read-only caching.
__device__ __forceinline__ float activation_scale(const float* p) {
  return *reinterpret_cast<const volatile float*>(p);
}
__device__ __forceinline__ void mma_fp4(float* d, const unsigned* a, unsigned b0, unsigned b1, unsigned sb, const float* c, unsigned sa = 0x38383838u) {
  const uint16_t z = 0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},%14,{%16,%16},%15,{%16,%16};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "r"(sb), "h"(z));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(count)); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, unsigned tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory"); }
// Consumer release of a ring stage that the producer refills with bulk (async-proxy) copies. The
// stage was read through the generic proxy (LDS/LDSM), so the release needs the proxy fence: without
// it ptxas issues the arrive while the warp's last shared loads of the stage are still in flight, and
// the refill can land under them. Every lane fences; one arrives per warp (arrival count FW).
__device__ __forceinline__ void ring_release(uint64_t* empty, bool arrive) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (arrive) mbar_arrive(empty);
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

// 0: unchanged instruction, 1: explicit normal hint, 2: evict-first weight hint.
#ifndef H5_WEIGHT_TMA_CACHE_POLICY
#define H5_WEIGHT_TMA_CACHE_POLICY 0
#endif
#if H5_WEIGHT_TMA_CACHE_POLICY < 0 || H5_WEIGHT_TMA_CACHE_POLICY > 2
#error "H5_WEIGHT_TMA_CACHE_POLICY must be 0 (none), 1 (normal), or 2 (evict_first)"
#endif
__device__ __forceinline__ uint64_t weight_tma_policy() {
#if H5_WEIGHT_TMA_CACHE_POLICY == 0
  return 0;
#else
  uint64_t policy;
#if H5_WEIGHT_TMA_CACHE_POLICY == 1
  asm volatile("createpolicy.fractional.L2::evict_normal.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(policy));
#else
  asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(policy));
#endif
  return policy;
#endif
}
__device__ __forceinline__ void bulk_weight_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar, uint64_t policy) {
#if H5_WEIGHT_TMA_CACHE_POLICY == 0
  bulk_g2s(dst, src, bytes, bar);
#else
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
#endif
}

}
