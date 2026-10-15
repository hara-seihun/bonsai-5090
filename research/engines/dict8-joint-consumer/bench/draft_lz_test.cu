// GPU test: the drafter selector's LZ adjustment (src/draft_lz.cuh) equals h5::lz::row (src/lz_penalty.h) for the
// candidates, on random and periodic contexts, both forms. `make draft-lz-test` (needs a GPU).
#include "draft_lz.cuh"
#include "lz_penalty.h"
#include <cstdio>
#include <map>
#include <random>
#include <vector>
using namespace h5;

__global__ void k_test(DraftLz z, float alpha, int T, const int* tok_g, const int* cand, float* adj_out) {
  __shared__ int tok[DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS];
  __shared__ short su[DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS];
  __shared__ float adj[DF_TOPK];
  __shared__ int red[1];
  for (int i = threadIdx.x; i < T; i += 256) tok[i] = tok_g[i];
  __syncthreads();
  draft_lz_adjust(z, alpha, T, tok, cand, adj, su, red);
  if (threadIdx.x < DF_TOPK) adj_out[threadIdx.x] = adj[threadIdx.x];
}

int main() {
  std::mt19937 rng(11);
  int *d_tok, *d_cand; float* d_adj;
  cudaMalloc(&d_tok, 4 * (DRAFT_LZ_MAX_WINDOW + DRAFT_MAX_PROPOSALS)); cudaMalloc(&d_cand, 4 * DF_TOPK); cudaMalloc(&d_adj, 4 * DF_TOPK);
  lz::Scratch scratch(VOCAB);
  int fails = 0, checked = 0;
  for (int it = 0; it < 600; ++it) {
    const int W = (int[]){64, 256, 512, 1024}[it % 4], B = 1 + rng() % 32;
    const lz::Form form = it % 2 ? lz::Form::Paper : lz::Form::Dist;
    const int len = rng() % (W + 1), p = rng() % 8, T = len + p;
    if (T < 1 || B >= W) continue;
    std::vector<int> seq(T);
    const int alph = it % 3 == 0 ? 4 : 50, period = 1 + rng() % 40;
    for (int i = 0; i < T; ++i) seq[i] = (it % 5 < 2 && i > T / 3) ? seq[i - std::min(i, period)] : 1000 + (int)(rng() % alph);
    // host: row over committed seq[0..len) + proposals seq[len..T)
    lz::Config c{0.15f, W, B, form, lz::Scope::Output};
    std::map<int, float> want;
    lz::row(c, seq.data(), len, seq.data() + len, p, scratch, [&](int t, float v) { want[t] = v; });
    std::vector<int> cand(DF_TOPK);
    for (int k = 0; k < DF_TOPK; ++k) cand[k] = k < 12 && T ? seq[rng() % T] : 1000 + (int)(rng() % 60);
    cudaMemcpy(d_tok, seq.data(), 4 * T, cudaMemcpyHostToDevice); cudaMemcpy(d_cand, cand.data(), 4 * DF_TOPK, cudaMemcpyHostToDevice);
    DraftLz z; z.ctx = d_tok; z.w = W; z.buffer = B; z.form = form == lz::Form::Dist ? 1 : 0; z.log2v = log2f((float)VOCAB);
    k_test<<<1, 256>>>(z, 0.15f, T, d_tok, d_cand, d_adj);
    float got[DF_TOPK]; cudaMemcpy(got, d_adj, 4 * DF_TOPK, cudaMemcpyDeviceToHost);
    for (int k = 0; k < DF_TOPK; ++k) {
      const float w = want.count(cand[k]) ? want[cand[k]] : 0.f;
      if (fabsf(got[k] - w) > 1e-4f * fmaxf(1.f, fabsf(w))) { if (fails++ < 10) printf("it %d W %d B %d form %d T %d cand %d: got %g want %g\n", it, W, B, z.form, T, cand[k], got[k], w); }
      ++checked;
    }
  }
  if (cudaGetLastError() != cudaSuccess) { printf("CUDA error\n"); return 1; }
  printf("draft-lz-test: %d candidates, %d mismatches\n", checked, fails);
  return fails ? 1 : 0;
}
