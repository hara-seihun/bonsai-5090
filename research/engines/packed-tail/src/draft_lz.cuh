#pragma once
// Drafter-side LZ adjustment (H5_DRAFT_LZ): device version of h5::lz::row (src/lz_penalty.h) for the selector's
// DF_TOPK candidates. Checked against the host definition by `make draft-lz-test` (bench/draft_lz_test.cu).
#include "kernels.cuh"
namespace h5 {
// LZ adjustment (src/lz_penalty.h, same definitions) for the DF_TOPK candidates cand[] at context tok[0..T).
// All 256 threads call it; adj[k] is valid after it returns.
__device__ inline void draft_lz_adjust(const DraftLz& z, float alpha, int T, const int* tok, const int* cand, float* adj, short* su, int* red) {
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  const int lo = max(0, T - z.w);
  const int src_lo = z.form == 1 ? lo + z.buffer : lo;
  if (t == 0) red[0] = 0;
  __syncthreads();
  int my_l = 0;
  for (int q = src_lo + t; q < T; q += 256) {
    int u = 0;
    while (u < z.buffer && q - 1 - u >= lo && tok[q - 1 - u] == tok[T - 1 - u]) ++u;
    su[q - lo] = (short)u; my_l = max(my_l, u);
  }
  if (my_l) atomicMax(red, my_l);
  __syncthreads();
  const int lmax = red[0];
  for (int k = w; k < DF_TOPK; k += 8) {
    const int c = cand[k];
    int near = -1, ext = -1, bx = 0, bq = -1;
    for (int q = src_lo + lane; q < T; q += 32) {
      if (tok[q] != c) continue;
      near = max(near, q);
      const int x = su[q - lo] + 1;
      if (lmax >= 1 && x == lmax + 1) ext = max(ext, q);
      if (x > bx || (x == bx && q > bq)) { bx = x; bq = q; }
    }
    for (int o = 16; o; o >>= 1) {
      near = max(near, __shfl_xor_sync(~0u, near, o));
      ext = max(ext, __shfl_xor_sync(~0u, ext, o));
      const int ox = __shfl_xor_sync(~0u, bx, o), oq = __shfl_xor_sync(~0u, bq, o);
      if (ox > bx || (ox == bx && oq > bq)) { bx = ox; bq = oq; }
    }
    if (lane == 0) {
      float a = 0.f;
      if (near >= 0) {
        double bits;
        if (z.form == 1) bits = fmin((double)z.log2v, fmax(0.0, log2((double)(T - bq))));
        else if (ext >= 0) { const double l = lmax, d = fmax((double)(T - ext), l + 2.0); bits = log2(1.0 + (d - l - 1.0) / (l * d)) - 1.0; }
        else bits = log2((double)(T - near));
        a = fminf(0.f, (float)(-(double)alpha * ((double)z.log2v - bits)));
      }
      adj[k] = a;
    }
  }
  __syncthreads();
}

}  // namespace h5
