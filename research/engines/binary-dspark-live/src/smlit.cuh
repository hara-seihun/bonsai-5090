#pragma once
// SM-lit map: diagnostic per-warp residency stamps for every instrumented kernel.
// Compiled only with -DH5_SMLIT. Each translation unit that launches kernels defines
// H5_LIT_MODULE before including this header and exports h5_lit_set_<module>(buffer).
// A null buffer disables recording; the host toggles it between replay rounds.
// Records are written by lane 0 of every warp at kernel scope exit. Timer reads perturb
// scheduling slightly; compare against an uninstrumented build before using absolute times.
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace h5::lit {
// Named sites occupy the high-bit namespace, separate from module/line tags.
// Values are capture identities: keep them stable when a kernel moves or changes.
enum class SiteTag : uint32_t {
  GdnSingle = 0x80010001u,
  GdnVerify = 0x80010002u,
  ConvGdnVerify = 0x80010003u,
  Prep2wOrdered = 0x80010004u,
  AttnCombineTail = 0x80010005u,
};
struct Rec {                 // 48 bytes
  uint32_t tag, cta, grid, aux;
  uint16_t warp, sm; uint32_t block_threads;
  uint64_t t_entry_ns, t_exit_ns;
  uint32_t c_wait, c_exit;   // clock64 cycles after entry: pdl wait returned (0 = none), scope exit
  uint32_t mark[4];          // optional phase marks, cycles after entry (0 = unset)
};
struct Buf { unsigned long long count; unsigned long long cap; Rec recs[1]; };
}

#if defined(H5_SMLIT) && defined(H5_LIT_MODULE)
namespace h5::lit {
static __device__ Buf* buffer = nullptr;
__device__ __forceinline__ uint64_t gtime() { uint64_t v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v) :: "memory"); return v; }
__device__ __forceinline__ uint64_t clk() { uint64_t v; asm volatile("mov.u64 %0, %%clock64;" : "=l"(v) :: "memory"); return v; }
__device__ __forceinline__ uint64_t* wait_slots() { __shared__ uint64_t slots[32]; return slots; }
// Phase marks reachable from any device function of the kernel (H5_LIT_MARKX): absolute clock per warp.
__device__ __forceinline__ uint64_t* mark_slots() { __shared__ uint64_t slots[32][4]; return &slots[0][0]; }
__device__ __forceinline__ void mark_any(int i) {
  if ((threadIdx.x & 31) == 0) { unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z); mark_slots()[((t >> 5) & 31) * 4 + i] = clk(); }
}
__device__ __forceinline__ void note_wait() {
  if ((threadIdx.x & 31) == 0) {
    unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    wait_slots()[(t >> 5) & 31] = clk();
  }
}
struct Warp {
  Buf* b; uint64_t t0, c0; uint32_t tag, aux; unsigned warp; uint32_t m[4];
  __device__ __forceinline__ Warp(uint32_t tag_, uint32_t aux_ = 0) : b(nullptr), tag(tag_), aux(aux_) {
    unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    warp = t >> 5;
    if ((t & 31) == 0) {
      b = buffer;
      if (b) { t0 = gtime(); c0 = clk(); wait_slots()[warp & 31] = 0; m[0] = m[1] = m[2] = m[3] = 0;
               for (int k = 0; k < 4; k++) mark_slots()[(warp & 31) * 4 + k] = 0; }
    }
  }
  __device__ __forceinline__ ~Warp() {
    if (!b) return;
    uint64_t c1 = clk(), t1 = gtime();
    unsigned long long i = atomicAdd(&b->count, 1ULL);
    if (i >= b->cap) return;
    Rec& r = b->recs[i];
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    uint64_t cw = wait_slots()[warp & 31];
    r.tag = tag;
    r.cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    r.grid = gridDim.x * gridDim.y * gridDim.z;
    r.aux = aux; r.warp = (uint16_t)warp; r.sm = (uint16_t)sm;
    r.block_threads = blockDim.x * blockDim.y * blockDim.z;
    r.t_entry_ns = t0; r.t_exit_ns = t1;
    r.c_wait = cw ? (uint32_t)(cw - c0) : 0u; r.c_exit = (uint32_t)(c1 - c0);
    for (int k = 0; k < 4; k++) { const uint64_t x = mark_slots()[(warp & 31) * 4 + k]; if (!m[k] && x) m[k] = (uint32_t)(x - c0); }
    r.mark[0] = m[0]; r.mark[1] = m[1]; r.mark[2] = m[2]; r.mark[3] = m[3];
  }
  __device__ __forceinline__ void mark(int i) { if (b) m[i] = (uint32_t)(clk() - c0); }
};
}
#define H5_LIT_J0(a, b) a##b
#define H5_LIT_J(a, b) H5_LIT_J0(a, b)
#define H5_LIT_TAG (H5_LIT_MODULE_ID * 100000u + (uint32_t)__LINE__)
#define H5_LIT(aux) h5::lit::Warp h5_lit_scope(H5_LIT_TAG, (uint32_t)(aux))
#define H5_LIT_SITE(site, aux) h5::lit::Warp h5_lit_scope((uint32_t)h5::lit::SiteTag::site, (uint32_t)(aux))
#define H5_LIT_MARK(i) h5_lit_scope.mark(i)
#define H5_LIT_MARKX(i) h5::lit::mark_any(i)
#define H5_LIT_EXPORT \
  extern "C" void H5_LIT_J(h5_lit_set_, H5_LIT_MODULE)(h5::lit::Buf* b) { \
    if (cudaMemcpyToSymbol(h5::lit::buffer, &b, sizeof(b)) != cudaSuccess) { abort(); } }
#else
#define H5_LIT(aux) do {} while (0)
#define H5_LIT_SITE(site, aux) do {} while (0)
#define H5_LIT_MARK(i) do {} while (0)
#define H5_LIT_MARKX(i) do {} while (0)
#define H5_LIT_EXPORT
namespace h5::lit { __device__ __forceinline__ void note_wait() {} }
#endif
