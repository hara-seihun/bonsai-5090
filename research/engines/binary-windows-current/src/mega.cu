// Persistent megakernel for single-slot decode/verify steps (R <= 8 rows of one slot).
//
// One cooperative launch runs the whole 64-layer forward: one CTA (8 warps) per SM, phases separated
// by a software grid barrier. Matrix phases are split into warp work items (32 or 64 output features
// over a K range) that read weight fragments straight from global memory into MMA registers, against
// activations staged once per CTA in shared memory. Weight streaming for upcoming projections is
// started (bulk L2 prefetch) during the latency-bound phases (prep, conv, GDN, attention), so HBM
// never idles across a phase boundary. Activations are NVFP4 with one FP32 scale per (row, 1024-chunk),
// so every prep item is independent (no row-wide max). The head computes argmax in its epilogue.
//
// This file is #included into kernels.cu (one translation unit with the GDN/attention bodies).
#include <cooperative_groups.h>

namespace h5 {

constexpr int MG_WARPS = 8, MG_THREADS = MG_WARPS * 32;
__device__ __forceinline__ void mma_fp4(float* d, const unsigned* a, unsigned b0, unsigned b1, unsigned sb, const float* c, unsigned sa = 0x38383838u) {
  const uint16_t z = 0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},%14,{%16,%16},%15,{%16,%16};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sa), "r"(sb), "h"(z));
}
constexpr int MG_SMEM = A8_SMEM > (136 * (512 + 64) + 8 * 17 * 4) ? A8_SMEM : (136 * (512 + 64) + 8 * 17 * 4);
static_assert(MG_SMEM >= GDN_SMEM, "megakernel smem");

__device__ __forceinline__ unsigned long long mg_time() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
// Grid barrier: one monotonic counter, release on arrival, acquire spin (0.56 us on 170 CTAs, measured).
__device__ __forceinline__ void mg_sync(unsigned* bar, unsigned nb, unsigned long long* prof = nullptr, int code = 0) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old, cur;
    asm volatile("atom.add.release.gpu.u32 %0, [%1], 1;" : "=r"(old) : "l"(bar) : "memory");
    const unsigned target = (old / nb + 1) * nb;
    while (true) {
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(bar) : "memory");
      if ((int)(cur - target) >= 0) break;
#ifdef MG_SPIN_SLEEP
      __nanosleep(MG_SPIN_SLEEP);
#endif
    }
    if (prof && blockIdx.x == 0) { const unsigned i = (unsigned)prof[0]++; if (i < 2047) prof[1 + i] = (mg_time() << 8) | (unsigned)code; }
  }
  __syncthreads();
}

// Bulk-prefetch [p, p+bytes) into L2, split over all warps of the grid (lane 0 issues).
__device__ __forceinline__ void mg_prefetch(const void* p, size_t bytes, int gw, int nw, int on = 1) {
  if (!p || !on) return;
  constexpr size_t CH = 64 * 1024;
  const size_t n = (bytes + CH - 1) / CH;
  if ((threadIdx.x & 31) != 0) return;
  for (size_t i = gw; i < n; i += nw) {
    const size_t off = i * CH, len = min(CH, bytes - off) & ~(size_t)15;
    if (len) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const uint8_t*)p + off), "r"((unsigned)len) : "memory");
  }
}

// ---- prep: one warp = one (row, 1024-chunk). Writes NVFP4 activations for the megakernel GEMM:
// a4 [kb][8 rows][64 B] (16-B chunk c of row r at c ^ ((r >> 1) & 3)), s4 [kb][8 rows][8 B], gs [row][chunk].
enum { MP_NORM = 1, MP_SIGN = 2, MP_HAD = 4, MP_QUANT = 8, MP_STORE = 16, MP_SIGN_AFTER = 32, MP_GDN = 64, MP_OUTNORM = 128, MP_EMBED = 256, MP_SILU_IL = 512 };
__device__ __forceinline__ void mg_prep(const MegaArgs& a, int row, int chunk, int n, unsigned fl, const float* x, int ldx,
                                        const float* norm_w, const float* signs, const float* gnorm_w, const float* gz,
                                        float* store, float* out_norm, const float* inv_row = nullptr,
                                        uint8_t* da4 = nullptr, uint8_t* ds4 = nullptr, float* dgs = nullptr) {
  if (!da4) { da4 = a.a4; ds4 = a.s4; dgs = a.gs; }
  const int lane = threadIdx.x & 31, e0 = chunk * 1024 + lane * 32;
  float v[32];
  if (fl & MP_EMBED) {   // decode token row from the PTQ1_0 embedding (natural order), 32 elements per lane
    const int t = a.tok[row];
    const uint8_t* blk = a.emb + ((size_t)t * (D / 128) + (e0 >> 7)) * 28;
    const float sc = __half2float(*(const __half*)(blk + 26));
    const int ib = e0 & 127;   // element index within the 128-block
#pragma unroll
    for (int r = 0; r < 32; r++) {
      const int e = ib + r;   // PTQ1_0: stage 0 = bytes 0..15 (5 trits each, trit n of byte m at n*16+m), stage 1 = bytes 16..23 at 80 + n*8+m, qh at 120 + n*2 + h
      int byte, nn;
      if (e < 80) { nn = e / 16; byte = e % 16; } else if (e < 120) { nn = (e - 80) / 8; byte = 16 + (e - 80) % 8; } else { nn = (e - 120) / 2; byte = 24 + (e - 120) % 2; }
      unsigned q = blk[byte];
      for (int k = 0; k < nn; k++) q = (q * 3u) & 0xff;
      v[r] = (float)((int)((q * 3u) >> 8) - 1) * sc;
    }
  } else if (fl & MP_GDN) {
#pragma unroll
    for (int r = 0; r < 32; r++) { const int gi = e0 + r, hd = gi & 127, q = gi >> 7, rr = q % 3, kh = q / 3; v[r] = x[(size_t)row * ldx + hd + 128 * (kh + 16 * rr)]; }
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 32; r++) ss = fmaf(v[r], v[r], ss);
    ss += __shfl_xor_sync(0xffffffff, ss, 1); ss += __shfl_xor_sync(0xffffffff, ss, 2);
    const float inv = rsqrtf(ss / 128.f + NORM_EPS);
    const int q = e0 >> 7, rr = q % 3, kh = q / 3, head = kh + 16 * rr, hd0 = e0 & 127;
    const float* z = gz + (size_t)row * GDN_IN + head * 128 + hd0;
#pragma unroll
    for (int r = 0; r < 32; r++) v[r] = v[r] * inv * gnorm_w[hd0 + r] * silu(z[r]);
  } else if (fl & MP_SILU_IL) {   // x = [R][GU] with gate group g at columns 64g.., up group g at 64g+32..
    const float* gr = x + (size_t)row * ldx + (e0 >> 5) * 64;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 gv = *(const float4*)(gr + 4 * i), uv = *(const float4*)(gr + 32 + 4 * i);
      v[4 * i] = silu(gv.x) * uv.x; v[4 * i + 1] = silu(gv.y) * uv.y; v[4 * i + 2] = silu(gv.z) * uv.z; v[4 * i + 3] = silu(gv.w) * uv.w;
    }
  } else {
    const float* xr = x + (size_t)row * ldx;
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 t = *(const float4*)(xr + e0 + 4 * i); v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w; }
    if (fl & MP_NORM) {   // row-wide mean square: precomputed by the CTA, or recomputed by this warp (L2-resident)
      float ss = 0.f;
      if (inv_row) {
      } else if (n == D) {   // 40 float4 per lane, all loads in flight at once
        float4 t[40];
#pragma unroll
        for (int i = 0; i < 40; i++) t[i] = *(const float4*)(xr + lane * 4 + i * 128);
#pragma unroll
        for (int i = 0; i < 40; i++) ss += t[i].x * t[i].x + t[i].y * t[i].y + t[i].z * t[i].z + t[i].w * t[i].w;
      } else {
#pragma unroll 8
        for (int i = lane * 4; i < n; i += 128) { const float4 t = *(const float4*)(xr + i); ss += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w; }
      }
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float inv = inv_row ? inv_row[row] : rsqrtf(ss / (float)n + NORM_EPS);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const float4 nwt = *(const float4*)(norm_w + e0 + 4 * i);
        v[4 * i] *= inv * nwt.x; v[4 * i + 1] *= inv * nwt.y; v[4 * i + 2] *= inv * nwt.z; v[4 * i + 3] *= inv * nwt.w;
      }
      if (fl & MP_OUTNORM)
#pragma unroll
        for (int i = 0; i < 8; i++) *(float4*)(out_norm + (size_t)row * n + e0 + 4 * i) = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
    }
  }
  if (fl & MP_SIGN) {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 sg = *(const float4*)(signs + e0 + 4 * i); v[4 * i] *= sg.x; v[4 * i + 1] *= sg.y; v[4 * i + 2] *= sg.z; v[4 * i + 3] *= sg.w; }
  }
  if (fl & MP_HAD) {
#pragma unroll
    for (int len = 1; len < 32; len <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++)
        if (!(r & len)) { const float p = v[r], q = v[r + len]; v[r] = p + q; v[r + len] = p - q; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
#pragma unroll
      for (int r = 0; r < 32; r++) { const float q = __shfl_xor_sync(0xffffffff, v[r], o); v[r] = (lane & o) ? q - v[r] : v[r] + q; }
#pragma unroll
    for (int r = 0; r < 32; r++) v[r] *= 0.03125f;
  }
  if (fl & MP_SIGN_AFTER) {
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 sg = *(const float4*)(signs + e0 + 4 * i); v[4 * i] *= sg.x; v[4 * i + 1] *= sg.y; v[4 * i + 2] *= sg.z; v[4 * i + 3] *= sg.w; }
  }
  if (fl & MP_STORE)
#pragma unroll
    for (int i = 0; i < 8; i++) *(float4*)(store + (size_t)row * D + e0 + 4 * i) = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  if (!(fl & MP_QUANT)) return;
  float bm0 = 0.f, bm1 = 0.f;
#pragma unroll
  for (int r = 0; r < 16; r++) { bm0 = fmaxf(bm0, fabsf(v[r])); bm1 = fmaxf(bm1, fabsf(v[r + 16])); }
  float am = fmaxf(bm0, bm1);
  for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
  const float g = fmaxf(am / (6.f * 448.f), 1e-30f), ig = 1.f / g;
  if (lane == 0) dgs[row * 17 + chunk] = g;
  uint32_t words[4]; uint8_t sbs[2];
#pragma unroll
  for (int b = 0; b < 2; b++) {
    const float target = (b ? bm1 : bm0) / 6.f * ig;
    __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
    float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
    if (sv < target && sb < 0x7e) { sb++; sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))); }
    const float rs = sv > 0.f ? ig / sv : 0.f;
    sbs[b] = sb;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint32_t wd = 0;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int r = 16 * b + 8 * h + 2 * q;
        wd |= (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[r] * rs, v[r + 1] * rs), __NV_E2M1, cudaRoundNearest) << (8 * q);
      }
      words[2 * b + h] = wd;
    }
  }
  const int kb = e0 >> 7, c = lane & 3;
  const size_t base = (size_t)kb * 8 + row;
  *(uint4*)(da4 + base * 64 + ((c ^ ((row >> 1) & 3)) << 4)) = make_uint4(words[0], words[1], words[2], words[3]);
  *(uint16_t*)(ds4 + base * 8 + 2 * c) = (uint16_t)sbs[0] | ((uint16_t)sbs[1] << 8);
}

// Stage activation K blocks [k0, k0 + n) into shared memory (all threads of the CTA). Layout: n x 512 B
// of a4, then n x 64 B of s4, then the [8][17] chunk scales.
__device__ __forceinline__ void mg_stage(const MegaArgs& a, int k0, int n, uint8_t* smem) {
  __syncthreads();
  const uint4* src = (const uint4*)a.a4 + (size_t)k0 * 32; uint4* dst = (uint4*)smem;
  for (int i = threadIdx.x; i < n * 32; i += MG_THREADS) dst[i] = src[i];
  const uint4* ss = (const uint4*)a.s4 + (size_t)k0 * 4; uint4* ds = (uint4*)(smem + n * 512);
  for (int i = threadIdx.x; i < n * 4; i += MG_THREADS) ds[i] = ss[i];
  float* dg = (float*)(smem + n * 576);
  for (int i = threadIdx.x; i < 8 * 17; i += MG_THREADS) dg[i] = a.gs[i];
  __syncthreads();
}

enum { MG_STORE = 0, MG_ATOMIC = 1, MG_ARGMAX = 3 };
constexpr int MGP = 4;   // weight chunks (k blocks) a warp holds in registers ahead of use
struct MgPre { uint4 w[MGP][2]; uint2 s[MGP]; int it; };
// Item it of a projection split ks ways along K: group it / ks, K blocks [nkb*kp/ks, nkb*(kp+1)/ks).
// Adjacent warps take adjacent groups at the same K range, so a warp-row of loads is one contiguous run.
__device__ __forceinline__ void mg_item_range(int it, int ks, int nkb, int NG, int& g, int& kb0, int& kb1) {
  g = it % NG; const int kp = it / NG; kb0 = nkb * kp / ks; kb1 = nkb * (kp + 1) / ks;
}
__device__ __forceinline__ void mg_load(MgPre& p, const uint8_t* W, const uint16_t* S, int NGall, int g, int kb, int slot) {
  const int lane = threadIdx.x & 31, gid = lane >> 2;
  const size_t ch = (size_t)kb * NGall + g;
  const uint8_t* wc = W + ch * 1024 + lane * 16;
  p.w[slot][0] = __ldg((const uint4*)wc); p.w[slot][1] = __ldg((const uint4*)(wc + 512));
  p.s[slot] = __ldg((const uint2*)(S + ch * 32 + gid * 4));
}
// Start loading this warp's first item of the next projection (call before the barrier that precedes it).
__device__ __forceinline__ void mg_preload(MgPre& p, const uint8_t* W, const uint16_t* S, int N, int K, int ks) {
  const int gw = blockIdx.x * MG_WARPS + (threadIdx.x >> 5);
  p.it = -1;
  if (gw >= (N / 32) * ks) return;
  int g, kb0, kb1; mg_item_range(gw, ks, K / 128, N / 32, g, kb0, kb1);
#pragma unroll
  for (int q = 0; q < MGP; q++) if (kb0 + q < kb1) mg_load(p, W, S, N / 32, g, kb0 + q, q);
  p.it = gw;
}
template <int MODE>
__device__ __forceinline__ void mg_item(const MegaArgs& a, MgPre& pr, bool preloaded, const uint8_t* W, const uint16_t* S, int NGall, int K,
                                        int g, int kb0, int kb1, float* out, int ldo, const uint8_t* smem, int kbase, int nst) {
  const int lane = threadIdx.x & 31, gid = lane >> 2, tq = lane & 3;
  const uint8_t* as0 = smem - kbase * 512; const uint8_t* ss0 = smem + nst * 512 - kbase * 64; const float* gsm = (const float*)(smem + nst * 576);
  float acc[2][4];
#pragma unroll
  for (int m = 0; m < 2; m++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[m][i] = 0.f;
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  const int t0 = tq * 2;
  if (!preloaded)
#pragma unroll
    for (int q = 0; q < MGP; q++) if (kb0 + q < kb1) mg_load(pr, W, S, NGall, g, kb0 + q, q);
  for (int kb = kb0; kb < kb1; kb += MGP) {
#pragma unroll
    for (int q = 0; q < MGP; q++) {
      const int k = kb + q;
      if (k >= kb1) break;
      unsigned b0, b1, b2, b3;
      {
        const int row = lane & 7, c = lane >> 3;
        unsigned sa = (unsigned)__cvta_generic_to_shared(as0 + k * 512 + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(sa));
      }
      const uint2 bs = *(const uint2*)(ss0 + k * 64 + gid * 8);
      const float g0s = gsm[t0 * 17 + (k >> 3)], g1s = gsm[(t0 + 1) * 17 + (k >> 3)];
      const unsigned wv[8] = {pr.w[q][0].x, pr.w[q][0].y, pr.w[q][0].z, pr.w[q][0].w, pr.w[q][1].x, pr.w[q][1].y, pr.w[q][1].z, pr.w[q][1].w};
      const __half2* sh = (const __half2*)&pr.s[q];
      const float2 s01 = __half22float2(sh[0]), s23 = __half22float2(sh[1]);
      const float wsc[2][2] = {{s01.x, s01.y}, {s23.x, s23.y}};
      if (k + MGP < kb1) mg_load(pr, W, S, NGall, g, k + MGP, q);
#pragma unroll
      for (int mm = 0; mm < 2; mm++) {
        unsigned a0[4], a1[4];
        { const unsigned r0 = wv[mm * 4], r1 = wv[mm * 4 + 1]; a0[0] = r0 & 0xAAAAAAAAu; a0[2] = (r0 << 1) & 0xAAAAAAAAu; a0[1] = r1 & 0xAAAAAAAAu; a0[3] = (r1 << 1) & 0xAAAAAAAAu; }
        { const unsigned r0 = wv[mm * 4 + 2], r1 = wv[mm * 4 + 3]; a1[0] = r0 & 0xAAAAAAAAu; a1[2] = (r0 << 1) & 0xAAAAAAAAu; a1[1] = r1 & 0xAAAAAAAAu; a1[3] = (r1 << 1) & 0xAAAAAAAAu; }
        float d[4];
        mma_fp4(d, a0, b0, b1, bs.x, zero);
        mma_fp4(d, a1, b2, b3, bs.y, d);
        acc[mm][0] = fmaf(d[0], wsc[mm][0] * g0s, acc[mm][0]);
        acc[mm][1] = fmaf(d[1], wsc[mm][0] * g1s, acc[mm][1]);
        acc[mm][2] = fmaf(d[2], wsc[mm][1] * g0s, acc[mm][2]);
        acc[mm][3] = fmaf(d[3], wsc[mm][1] * g1s, acc[mm][3]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int t = t0 + j;
    if (MODE == MG_ARGMAX) {
      unsigned long long best = 0;
#pragma unroll
      for (int mm = 0; mm < 2; mm++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          const float v = acc[mm][r * 2 + j];
          const unsigned f = 32 * g + 16 * mm + 8 * r + gid;
          unsigned u = __float_as_uint(v); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
          const unsigned long long key = ((unsigned long long)u << 32) | (0xFFFFFFFFu - f);
          best = key > best ? key : best;
        }
      for (int o = 4; o < 32; o <<= 1) { const unsigned long long ob = __shfl_xor_sync(0xffffffff, best, o); best = ob > best ? ob : best; }
      if (gid == 0 && t < a.R) atomicMax(a.amax + t, best);
      continue;
    }
    if (t >= a.R) continue;
    float* o = out + (size_t)t * ldo + 32 * g;
#pragma unroll
    for (int mm = 0; mm < 2; mm++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        const int f = 16 * mm + 8 * r + gid;
        if (MODE == MG_ATOMIC) atomicAdd(o + f, acc[mm][r * 2 + j]); else o[f] = acc[mm][r * 2 + j];
      }
  }
}

// A projection phase: stage activations, then warp items over (32-feature groups x ks K splits).
// A projection phase: stage all activation K blocks once per CTA, then warp items (one per warp per wave).
// (Staging only each wave's K range was tried: the per-wave CTA barriers cost more than the copy saves.)
template <int MODE>
__device__ __forceinline__ void mg_gemm(const MegaArgs& a, MgPre& pr, const uint8_t* W, const uint16_t* S, int N, int K, int ks, float* out, int ldo, uint8_t* smem) {
  const int NG = N / 32, nkb = K / 128, nitems = NG * ks;
  mg_stage(a, 0, nkb, smem);
  for (int it = blockIdx.x * MG_WARPS + (threadIdx.x >> 5); it < nitems; it += gridDim.x * MG_WARPS) {
    int g, kb0, kb1; mg_item_range(it, ks, nkb, NG, g, kb0, kb1);
    mg_item<MODE>(a, pr, pr.it == it, W, S, NG, K, g, kb0, kb1, out, ldo, smem, 0, nkb);
    pr.it = -1;
  }
}
// What a GEMM phase quantises for itself (local prep): the activation source and its prep flags.
struct MgSrc { unsigned fl; const float* x; int ldx; const float* norm_w; const float* signs; const float* gnorm_w; const float* gz; };
// GEMM phase with local prep: each CTA quantises straight into its shared staging every (row, 1024-chunk)
// its own items read, so no grid-wide prep phase, barrier or global round trip precedes the matrix work.
// MP_NORM sources: warps 0..R-1 first compute the row rms inverses into inv_sh (also used by the caller).
template <int MODE>
__device__ __forceinline__ void mg_gemm_lp(const MegaArgs& a, MgPre& pr, const MgSrc& src, const uint8_t* W, const uint16_t* S, int N, int K, int ks,
                                           float* out, int ldo, uint8_t* smem, float* inv_sh, int pcode = 0) {
  const int NG = N / 32, nkb = K / 128, nitems = NG * ks, nw = gridDim.x * MG_WARPS, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned mask = 0;
  for (int base = blockIdx.x * MG_WARPS; base < nitems; base += nw)
    for (int w = 0; w < MG_WARPS && base + w < nitems; w++) {
      int g, k0, k1; mg_item_range(base + w, ks, nkb, NG, g, k0, k1);
      for (int c = k0 >> 3; c <= (k1 - 1) >> 3; c++) mask |= 1u << c;
    }
  if (src.fl & MP_NORM) {
    if (warp < a.R) {
      const float* xr = src.x + (size_t)warp * src.ldx;
      float ss = 0.f;
#pragma unroll 8
      for (int i = lane * 4; i < K; i += 128) { const float4 t = *(const float4*)(xr + i); ss += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w; }
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      if (lane == 0) inv_sh[warp] = rsqrtf(ss / (float)K + NORM_EPS);
    }
    __syncthreads();
    if (a.prof && pcode && blockIdx.x == 0 && threadIdx.x == 0) { const unsigned i = (unsigned)a.prof[0]++; if (i < 2047) a.prof[1 + i] = (mg_time() << 8) | 25u; }
  }
  const int nc = __popc(mask);
  for (int t = warp; t < a.R * nc; t += MG_WARPS) {
    const int row = t / nc;
    int ci = t % nc; unsigned m = mask;
    for (; ci; ci--) m &= m - 1;
    const int chunk = __ffs(m) - 1;
    mg_prep(a, row, chunk, K, src.fl | MP_QUANT, src.x, src.ldx, src.norm_w, src.signs, src.gnorm_w, src.gz, nullptr, nullptr, inv_sh,
            smem, smem + nkb * 512, (float*)(smem + nkb * 576));
  }
  __syncthreads();
  if (a.prof && pcode && blockIdx.x == 0 && threadIdx.x == 0) { const unsigned i = (unsigned)a.prof[0]++; if (i < 2047) a.prof[1 + i] = (mg_time() << 8) | (unsigned)pcode; }
  for (int it = blockIdx.x * MG_WARPS + warp; it < nitems; it += nw) {
    int g, kb0, kb1; mg_item_range(it, ks, nkb, NG, g, kb0, kb1);
    mg_item<MODE>(a, pr, pr.it == it, W, S, NG, K, g, kb0, kb1, out, ldo, smem, 0, nkb);
    pr.it = -1;
  }
}

// Zero rows [0, R) of a [R][n] buffer (all warps of the grid).
__device__ __forceinline__ void mg_zero(float* p, int n, int R) {
  for (int i = blockIdx.x * MG_THREADS + threadIdx.x; i < R * n / 4; i += gridDim.x * MG_THREADS) ((float4*)p)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}

// conv over one item of 256 channels (two heads), all threads; pending-cache semantics as k_conv.
__device__ __forceinline__ void mg_conv(const MegaArgs& a, const MLayer& L, int item, uint8_t* smem) {
  float* red = (float*)smem;   // [2][8]
  const Seg sg = a.seg;
  const int c = item * 256 + threadIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const bool qk = item < 16;
  float* st = a.conv_state + ((size_t)sg.slot * NGDN + L.gl) * 3 * CONV_CH;
  float s0 = st[c], s1 = st[CONV_CH + c], s2 = st[2 * CONV_CH + c];
  __nv_bfloat16* pend = (__nv_bfloat16*)a.pend_conv + ((size_t)sg.slot * NGDN + L.gl) * a.pending_rows * CONV_CH + c;
  for (int i = 0; i < sg.commit; i++) { s0 = s1; s1 = s2; s2 = __bfloat162float(pend[(size_t)i * CONV_CH]); }
  if (sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
  const float4 wk = *(const float4*)(L.conv_w + c * 4);
  for (int i = 0; i < sg.len; i++) {
    const int row = sg.start + i;
    const float xc = a.proj[(size_t)row * GDN_IN + c];
    float v = silu(wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * xc);
    if (qk) {
      float ss = v * v;
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      if (lane == 0) red[(i & 1) * 8 + w] = ss;
      __syncthreads();
      const int hb = (w >> 2) * 4;
      ss = red[(i & 1) * 8 + hb] + red[(i & 1) * 8 + hb + 1] + red[(i & 1) * 8 + hb + 2] + red[(i & 1) * 8 + hb + 3];
      v *= 1.f / fmaxf(sqrtf(ss), NORM_EPS);
    }
    a.conv_out[(size_t)row * CONV_CH + c] = v;
    if (sg.keep) pend[(size_t)i * CONV_CH] = __float2bfloat16(xc);
    if (item >= 16 && (threadIdx.x & 127) == 0) {   // v head: decay and beta gates
      const int hv = (c - 2 * KDIM) / 128;
      const float xg = a.abv[(size_t)row * 96 + hv] + L.ssm_dt[hv];
      const float sp = xg > 20.f ? xg : log1pf(__expf(xg));
      a.gates[(size_t)row * 96 + hv] = __expf(L.ssm_a[hv] * sp);
      a.gates[(size_t)row * 96 + HV + hv] = sigm(a.abv[(size_t)row * 96 + HV + hv]);
    }
    s0 = s1; s1 = s2; s2 = xc;
  }
  if (!sg.keep) { st[c] = s0; st[CONV_CH + c] = s1; st[2 * CONV_CH + c] = s2; }
  __syncthreads();
}

// K splits per projection (170 SMs x 8 warps = 1360 warps): item counts near whole multiples of the warp count.
constexpr int KS_IN = 10, KS_AIN = 12, KS_OUT = 16, KS_GU = 5, KS_DOWN = 34, KS_HEAD = 1;
template <int MT>
__global__ void __launch_bounds__(MG_THREADS, 1) k_mega(MegaArgs a) { H5_LIT(0);
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ float inv_sh[8];
  const unsigned nb = gridDim.x;
  const int warp = threadIdx.x >> 5, gw = blockIdx.x * MG_WARPS + warp, nw = gridDim.x * MG_WARPS;
  const int gp = warp * gridDim.x + blockIdx.x;   // spread small item sets one per SM first
  const int R = a.R;
  MgPre pr; pr.it = -1;
  if (a.prof && blockIdx.x == 0 && threadIdx.x == 0) { a.prof[0] = 0; }
  if (blockIdx.x == 0 && threadIdx.x < 8) a.amax[threadIdx.x] = 0ull;
  mg_preload(pr, a.L[0].w_in, a.L[0].s_in, a.L[0].n_in, D, a.L[0].attn ? KS_AIN : KS_IN);
  mg_zero(a.proj, GDN_IN > ATT_IN ? GDN_IN : ATT_IN, R);
  for (int it = gp; it < R * 5; it += nw) mg_prep(a, it / 5, it % 5, D, MP_EMBED | MP_HAD | MP_SIGN_AFTER | MP_STORE, nullptr, 0, nullptr, a.s5120, nullptr, nullptr, a.x, nullptr);
  mg_sync(a.bar, nb, a.prof, 1);
  for (int l = 0; l < NLAYER; l++) {
    const MLayer& L = a.L[l];
    const int nin = L.attn ? ATT_IN : GDN_IN, ksin = L.attn ? KS_AIN : KS_IN;
    // A. input projection, quantising the normed residual locally; zero the gate/up accumulator
    mg_zero(a.proj2, GU, R);
    mg_gemm_lp<MG_ATOMIC>(a, pr, MgSrc{MP_NORM | MP_SIGN | MP_HAD, a.x, D, L.attn_norm, a.s5120, nullptr, nullptr}, L.w_in, L.s_in, nin, D, ksin, a.proj, nin, smem, inv_sh, 20);
    if (!L.attn) {
      const int lane = threadIdx.x & 31;
      for (int it = gp; it < R * 96; it += nw) {   // alpha/beta bf16 matvec on the normed row (x * norm_w, times the row's rms inverse)
        const int row = it / 96, o = it % 96;
        const __nv_bfloat162* wr = (const __nv_bfloat162*)(L.ab + (size_t)o * D);
        const float2* xr = (const float2*)(a.x + (size_t)row * D);
        const float2* nr = (const float2*)L.attn_norm;
        float acc = 0.f;
#pragma unroll 8
        for (int i = lane; i < D / 2; i += 32) { const float2 wf = __bfloat1622float2(wr[i]); const float2 xv = xr[i], nv = nr[i]; acc = fmaf(wf.x, xv.x * nv.x, fmaf(wf.y, xv.y * nv.y, acc)); }
        for (int o2 = 16; o2; o2 >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o2);
        if (lane == 0) a.abv[(size_t)row * 96 + o] = acc * inv_sh[row];
      }
      mg_sync(a.bar, nb, a.prof, 3);
      mg_prefetch(L.w_out, (size_t)D * VDIM / 4, gw, nw, a.pf);
      for (int it = blockIdx.x; it < CONV_CH / 256; it += gridDim.x) mg_conv(a, L, it, smem);
      mg_sync(a.bar, nb, a.prof, 4);
      {
        GdnArgs ga{a.conv_out, a.proj, a.abv, L.ssm_a, L.ssm_dt, L.ssm_norm, a.gdn_state, L.gl, a.y, nullptr, a.pend_gdn, a.gates};
        ga.pending_rows = a.pending_rows;
        for (int it = blockIdx.x; it < 2 * HV; it += gridDim.x) { gdn_body<__nv_bfloat16>(ga, a.seg, it, smem); __syncthreads(); }
      }
      mg_preload(pr, L.w_out, L.s_out, D, VDIM, KS_OUT);
      mg_sync(a.bar, nb, a.prof, 5);
      // D. output projection, gated-norm prep of the GDN output done locally
      mg_gemm_lp<MG_ATOMIC>(a, pr, MgSrc{MP_GDN | MP_SIGN | MP_HAD, a.y, VDIM, nullptr, a.s6144, L.ssm_norm, a.proj + CONV_CH}, L.w_out, L.s_out, D, VDIM, KS_OUT, a.x, D, smem, inv_sh, 21);
    } else {
      mg_sync(a.bar, nb, a.prof, 7);
      AttnArgs aa{a.proj, L.q_norm, L.k_norm, a.kc, a.vc, a.ksc, a.vsc, L.al, a.ctx, a.slotv, a.pos, a.qrot, a.y};
      aa.vt = 1;
      for (int it = blockIdx.x; it < (NH + NKV) * R; it += gridDim.x) { attn_pre_body(aa, it % (NH + NKV), it / (NH + NKV)); __syncthreads(); }
      mg_sync(a.bar, nb, a.prof, 8);
      mg_prefetch(L.w_out, (size_t)D * ATTN_OUT / 4, gw, nw, a.pf);
      AttnTcArgs t{a.qrot, a.proj, a.kc, a.vc, a.ksc, a.vsc, L.al, a.ctx, a.pos, a.qbp, a.part_o, a.part_ml, a.y};
      t.tps = a.tps;
      for (int it = blockIdx.x; it < NKV * a.nsp; it += gridDim.x) { attn8_body<MT>(t, 0, it % NKV, it / NKV, a.nsp, smem); __syncthreads(); }
      mg_sync(a.bar, nb, a.prof, 9);
      for (int it = blockIdx.x; it < R * NH; it += gridDim.x) attn_combine_body(t, a.nsp, HD, it / NH, it % NH, NH);
      mg_preload(pr, L.w_out, L.s_out, D, ATTN_OUT, KS_OUT);
      mg_sync(a.bar, nb, a.prof, 10);
      mg_gemm_lp<MG_ATOMIC>(a, pr, MgSrc{MP_SIGN | MP_HAD, a.y, ATTN_OUT, nullptr, a.s6144, nullptr, nullptr}, L.w_out, L.s_out, D, ATTN_OUT, KS_OUT, a.x, D, smem, inv_sh, 21);
    }
    mg_preload(pr, L.w_gu, L.s_gu, GU, D, KS_GU);
    mg_sync(a.bar, nb, a.prof, 12);
    // E. gate/up (interleaved groups) from the post-norm residual, quantised locally; zero the input-projection accumulator
    mg_zero(a.proj, GDN_IN > ATT_IN ? GDN_IN : ATT_IN, R);
    mg_gemm_lp<MG_ATOMIC>(a, pr, MgSrc{MP_NORM | MP_SIGN | MP_HAD, a.x, D, L.post_norm, a.s5120, nullptr, nullptr}, L.w_gu, L.s_gu, GU, D, KS_GU, a.proj2, GU, smem, inv_sh, 22);
    mg_preload(pr, L.w_down, L.s_down, D, FF, KS_DOWN);
    mg_sync(a.bar, nb, a.prof, 14);
    // F. down projection of silu(gate) * up into the residual
    mg_gemm_lp<MG_ATOMIC>(a, pr, MgSrc{MP_SILU_IL | MP_SIGN | MP_HAD, a.proj2, GU, nullptr, a.s17408, nullptr, nullptr}, L.w_down, L.s_down, D, FF, KS_DOWN, a.x, D, smem, inv_sh, 23);
    if (l + 1 < NLAYER) {
      mg_prefetch(a.L[l + 1].w_in, (size_t)a.L[l + 1].n_in * D / 4, gw, nw, a.pf);
      mg_preload(pr, a.L[l + 1].w_in, a.L[l + 1].s_in, a.L[l + 1].n_in, D, a.L[l + 1].attn ? KS_AIN : KS_IN);
    } else mg_preload(pr, a.w_head, a.s_head, VOCAB, D, KS_HEAD);
    mg_sync(a.bar, nb, a.prof, 16);
    if (a.capture) {   // drafter features: residual after the capture layers
      int j = -1;
      if (l == 5) j = 0; else if (l == 19) j = 1; else if (l == 33) j = 2; else if (l == 47) j = 3; else if (l == 61) j = 4;
      if (j >= 0)
        for (int i = blockIdx.x * MG_THREADS + threadIdx.x; i < R * D / 4; i += gridDim.x * MG_THREADS) {
          const int row = (i * 4) / D, col = (i * 4) % D;
          *(float4*)(a.hcap + ((size_t)row * 5 + j) * D + col) = *(const float4*)(a.x + (size_t)row * D + col);
        }
    }
  }
  // final norm (local) and head with fused argmax
  mg_gemm_lp<MG_ARGMAX>(a, pr, MgSrc{MP_NORM | MP_SIGN | MP_HAD, a.x, D, a.out_norm, a.s5120, nullptr, nullptr}, a.w_head, a.s_head, VOCAB, D, KS_HEAD, nullptr, 0, smem, inv_sh, 24);
  mg_sync(a.bar, nb, a.prof, 18);
  if (blockIdx.x == 0 && threadIdx.x < R) a.out_tok[threadIdx.x] = (int)(0xFFFFFFFFu - (unsigned)(a.amax[threadIdx.x] & 0xFFFFFFFFull));
}

int mega_nsm() { int d; cudaGetDevice(&d); int n; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }
void launch_mega(const MegaArgs& a, int nblocks, cudaStream_t st) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(k_mega<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, MG_SMEM);
    cudaFuncSetAttribute(k_mega<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, MG_SMEM);
    cudaFuncSetAttribute(k_mega<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, MG_SMEM);
    init = true;
  }
  void* args[] = {(void*)&a};
  void (*k)(MegaArgs) = a.mt <= 1 ? k_mega<1> : a.mt == 2 ? k_mega<2> : k_mega<3>;
  cudaLaunchCooperativeKernel((void*)k, dim3(nblocks), dim3(MG_THREADS), args, MG_SMEM, st);
}

}  // namespace h5
