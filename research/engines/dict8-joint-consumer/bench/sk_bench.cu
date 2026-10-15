// Verifier weight-GEMM sequence on synthetic weights: production kernels at the selected plan versus
// the stream-K kernel (src/sk.cu), per projection type and as the whole 64-layer + head chain.
// Weights are distinct per layer (5.2 GB), so nothing is served from L2 across launches.
//
//   sk-bench [R]           (R = rows, default 8)
// Prints: correctness (max relative difference to the production kernel), then per type and chain
// microseconds per pass and GB/s for the production plan, every stream-K shape x CTAs-per-SM, and the
// copy-only stream-K ceiling.
#include "kernels.cuh"
#include "sk.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace h5 {
cudaError_t launch_scale_carrier(int ci, int split, int mode, const uint8_t* A, const uint8_t* AS, const float* AG,
                                 int R, int K, const void* W, int N, float* out, int ldo, const float* mean, cudaStream_t st);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
void set_prewait_cap(int cap);
}
using namespace h5;

static void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}
__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, uint32_t mod) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = (uint8_t)(mod ? x % mod : x);
  }
}
__global__ void k_fillf(float* p, size_t n, uint32_t seed, float lo, float hi) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = lo + (hi - lo) * (x & 0xffffff) / 16777216.f;
  }
}
template <class T> static T* dev(size_t n) { T* p; ck(cudaMalloc(&p, n * sizeof(T)), "malloc"); ck(cudaMemset(p, 0, n * sizeof(T)), "memset"); ck(cudaDeviceSynchronize(), "memset sync"); return p; }

enum Type { T_IN_GDN, T_IN_ATT, T_AB, T_OUT, T_GU, T_DOWN, T_HEAD, NTYPE };
static const char* tname[NTYPE] = {"in_gdn", "in_att", "ab", "out", "gu", "down", "head"};
struct Mat { int type, N, K, wf; uint8_t* W; uint8_t* S; float* RS; };
struct Plan { int ci, ks; };   // production plan at R = 8 (plan-b973-ingest + selected overrides)
static Plan base_plan(int type) {
  switch (type) {
    case T_IN_GDN: return {26, 4};
    case T_IN_ATT: return {26, 6};
    case T_AB: return {2, 8};
    case T_OUT: return {26, 6};
    case T_GU: return {19, 1};
    case T_DOWN: return {26, 17};
    default: return {4, 1};   // head, H5_HEAD_KS=1
  }
}

int main(int argc, char** argv) {
  const int R = argc > 1 ? atoi(argv[1]) : 8;
  if (R < 1 || R > 8) { fprintf(stderr, "R must be 1..8\n"); return 2; }
  int nsm = 0; ck(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0), "attr");
  cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
  std::vector<Mat> mats;
  size_t bytes = 0;
  auto add = [&](int type, int N, int K, int wf) {
    Mat m{type, N, K, wf, nullptr, nullptr, nullptr};
    const size_t nkb = K / 128, ng = N / 32;
    const size_t wb = nkb * ng * (wf == 1 ? 2048 : 832);
    m.W = dev<uint8_t>(wb); k_fill<<<1024, 256, 0, st>>>(m.W, wb, (uint32_t)mats.size() * 7919u, wf == 1 ? 0 : 243);
    if (wf == 1) { const size_t sb = nkb * ng * 256; m.S = dev<uint8_t>(sb); k_fill<<<256, 256, 0, st>>>(m.S, sb, 99, 8); }   // UE4M3 scales 2^-7..2^0
    m.RS = dev<float>(N); k_fillf<<<64, 256, 0, st>>>(m.RS, N, (uint32_t)mats.size(), 0.5f, 1.5f);
    bytes += wb + (wf == 1 ? nkb * ng * 256 : 0);
    mats.push_back(m);
  };
  for (int l = 0; l < NLAYER; l++) {
    if (is_attn(l)) add(T_IN_ATT, ATT_IN, D, 3);
    else { add(T_IN_GDN, GDN_IN, D, 3); add(T_AB, 96, D, 1); }
    add(T_OUT, D, l % 4 == 3 ? ATTN_OUT : VDIM, 3);
    add(T_GU, GU, D, 3);
    add(T_DOWN, D, FF, 3);
  }
  add(T_HEAD, VOCAB, D, 3);
  ck(cudaStreamSynchronize(st), "fill");
  printf("weights %.3f GB in %zu matrices, %d SMs, R=%d\n", bytes / 1e9, mats.size(), nsm, R);
  // activations: [tile64][kb][64][64 B] + scales, sized for K = FF
  const int nkbmax = FF / 128;
  uint8_t* A = dev<uint8_t>((size_t)nkbmax * 64 * 64); k_fill<<<256, 256, 0, st>>>(A, (size_t)nkbmax * 64 * 64, 5, 0);
  uint8_t* AS = dev<uint8_t>((size_t)nkbmax * 64 * 8);
  { std::vector<uint8_t> h((size_t)nkbmax * 64 * 8, 0x30); ck(cudaMemcpy(AS, h.data(), h.size(), cudaMemcpyHostToDevice), "AS"); }
  float* AG = dev<float>(64); k_fillf<<<1, 64, 0, st>>>(AG, 64, 3, 0.5f, 1.f);
  float* proj = dev<float>((size_t)8 * GU);
  float* x = dev<float>((size_t)8 * D);
  float* ab = dev<float>((size_t)8 * 96);
  float* logits = dev<float>((size_t)8 * VOCAB);
  float* ws = dev<float>(sk_workspace_floats(VOCAB));
  int* cnt = dev<int>(sk_counter_ints(VOCAB));
  float* ref = dev<float>((size_t)8 * VOCAB);
  ck(cudaStreamSynchronize(st), "init");
  set_prewait_cap(5);

  struct Out { float* p; int ld; };
  auto out_of = [&](int type) -> Out {
    switch (type) {
      case T_IN_GDN: case T_IN_ATT: return {proj, GU};
      case T_AB: return {ab, 96};
      case T_OUT: case T_DOWN: return {x, D};
      case T_GU: return {proj, FF};
      default: return {logits, VOCAB};
    }
  };
  auto base_launch = [&](const Mat& m, float* o, int ld) {
    const Plan p = base_plan(m.type);
    if (m.type == T_AB) { launch_gemm_fp4_cfg(p.ci, p.ks, A, AS, AG, R, m.K, m.W, m.S, m.N, o, ld, true, st, m.RS, 1); return; }
    const int mode = m.type == T_GU ? 3 : p.ks > 1 ? 2 : (m.type == T_OUT || m.type == T_DOWN) ? 1 : 0;
    ck(launch_scale_carrier(p.ci, p.ks, mode, A, AS, AG, R, m.K, m.W, m.N, o, ld, m.RS, st), "base");
  };
  struct SkSel { int shape, per_sm; int var; int items = 4; };
  int* nextc = dev<int>(SK_SITES); int site = 0;
  auto counter = [&] { return nextc + (site++ % SK_SITES); };
  auto sk_launch_m = [&](const Mat& m, float* o, int ld, SkSel s) {
    const int grid = nsm * s.per_sm;
    SkArgs a{A, AS, AG, R, m.K, m.N, m.W, m.S, o, ld, m.RS, ws, cnt, 5, counter(), sk_chunk(s.shape, m.wf, m.N, m.K, 0, grid, s.items)};
    const int mode = m.type == T_GU ? SK_SILU : m.type == T_HEAD ? SK_STORE : SK_ATOMIC;
    ck(sk_launch(s.shape, m.wf, mode, m.wf == 1 ? 0 : s.var, a, grid, st), "sk");
  };

  // correctness: production vs stream-K on the first matrix of each type
  printf("\ncorrectness (first matrix of each type, max |sk-base| / max |base|)\n");
  for (int type = 0; type < NTYPE; type++) {
    const Mat* m = nullptr; for (auto& q : mats) if (q.type == type) { m = &q; break; }
    const int ld = type == T_GU ? FF : m->N;
    const size_t n = (size_t)R * ld;
    std::vector<float> hb(n), hs(n);
    ck(cudaMemsetAsync(ref, 0, n * 4, st), "clr");
    base_launch(*m, ref, ld);
    ck(cudaMemcpyAsync(hb.data(), ref, n * 4, cudaMemcpyDeviceToHost, st), "cp");
    for (int shape = 0; shape < sk_nshapes(); shape++) {
      const int mode = type == T_GU ? SK_SILU : type == T_HEAD ? SK_STORE : SK_ATOMIC;
      if (sk_occupancy(shape, m->wf, mode, 0) < 1) continue;
      ck(cudaMemsetAsync(ref, 0, n * 4, st), "clr");
      sk_launch_m(*m, ref, ld, {shape, 1, 0, 3});
      ck(cudaMemcpyAsync(hs.data(), ref, n * 4, cudaMemcpyDeviceToHost, st), "cp");
      ck(cudaStreamSynchronize(st), "corr");
      double mx = 0, md = 0;
      for (size_t i = 0; i < n; i++) { mx = std::max(mx, (double)fabsf(hb[i])); md = std::max(md, (double)fabsf(hb[i] - hs[i])); }
      const SkShape s = sk_shape(shape);
      printf("  %-7s fw%d wg%d st%-2d  %.2e%s\n", tname[type], s.fw, s.wg, s.stages, mx > 0 ? md / mx : md, md / std::max(mx, 1e-30) > 1e-4 ? "  MISMATCH" : "");
    }
  }

  // timing: graph of the selected matrices, replayed; median per replay
  auto time_graph = [&](const std::function<void()>& body) -> double {
    cudaGraph_t g; cudaGraphExec_t ge;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap");
    body();
    ck(cudaStreamEndCapture(st, &g), "endcap");
    ck(cudaGraphInstantiate(&ge, g, 0), "inst");
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) ck(cudaGraphLaunch(ge, st), "warm");
    std::vector<float> v;
    for (int i = 0; i < 15; i++) {
      cudaEventRecord(e0, st); ck(cudaGraphLaunch(ge, st), "run"); cudaEventRecord(e1, st);
      ck(cudaEventSynchronize(e1), "sync"); float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms * 1e3f);
    }
    std::sort(v.begin(), v.end());
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return v[v.size() / 2];
  };
  auto type_bytes = [&](int type) { double b = 0; for (auto& m : mats) if (type < 0 || m.type == type) b += (double)m.N / 32 * m.K / 128 * (m.wf == 1 ? 2304 : 832); return b; };

  printf("\nper type, all layers back to back (us per pass, GB/s)\n");
  std::vector<SkSel> best(NTYPE, SkSel{-1, 1, 0});
  for (int type = 0; type < NTYPE; type++) {
    const double b = type_bytes(type);
    auto run = [&](SkSel s) { return time_graph([&] { for (auto& m : mats) if (m.type == type) { auto o = out_of(type); sk_launch_m(m, o.p, o.ld, s); } }); };
    const double tb = time_graph([&] { for (auto& m : mats) if (m.type == type) { auto o = out_of(type); base_launch(m, o.p, o.ld); } });
    printf("  %-7s base cfg%-2d ks%-2d            %8.1f us %7.0f GB/s\n", tname[type], base_plan(type).ci, base_plan(type).ks, tb, b / tb / 1e3);
    double bt = 1e30;
    const int wf = type == T_AB ? 1 : 3, mode = type == T_GU ? SK_SILU : type == T_HEAD ? SK_STORE : SK_ATOMIC;
    for (int shape = 0; shape < sk_nshapes(); shape++)
      for (int items : {2, 4, 8}) {
        const int occ = sk_occupancy(shape, wf, mode, 0);
        for (int per = 1; per <= std::min(occ, 2); per++) {
          const SkSel s{shape, per, 0, items};
          const double t = run(s);
          const SkShape sh = sk_shape(shape);
          printf("  %-7s sk %2d fw%-2d wg%d st%-2d x%d it%d  %8.1f us %7.0f GB/s\n", tname[type], shape, sh.fw, sh.wg, sh.stages, per, items, t, b / t / 1e3);
          if (t < bt) { bt = t; best[type] = s; }
        }
      }
    if (wf == 3) {
      const SkSel bs = best[type];
      const double tc = run({bs.shape, bs.per_sm, 1, bs.items}), tk = run({bs.shape, bs.per_sm, 2, bs.items});
      printf("  %-7s best: copy-only %8.1f us %7.0f GB/s, compute-only %8.1f us %7.0f GB/s\n", tname[type], tc, b / tc / 1e3, tk, b / tk / 1e3);
    }
  }

  printf("\nwhole verifier GEMM chain (64 layers + head)\n");
  const double b = type_bytes(-1);
  const double tb = time_graph([&] { for (auto& m : mats) { auto o = out_of(m.type); base_launch(m, o.p, o.ld); } });
  printf("  base            %8.1f us %7.0f GB/s\n", tb, b / tb / 1e3);
  const double ts = time_graph([&] { for (auto& m : mats) { auto o = out_of(m.type); sk_launch_m(m, o.p, o.ld, best[m.type]); } });
  printf("  sk best/type    %8.1f us %7.0f GB/s\n", ts, b / ts / 1e3);
  const double tm = time_graph([&] { for (auto& m : mats) { auto o = out_of(m.type); if (m.type == T_AB) base_launch(m, o.p, o.ld); else sk_launch_m(m, o.p, o.ld, best[m.type]); } });
  printf("  sk, base ab     %8.1f us %7.0f GB/s\n", tm, b / tm / 1e3);
  auto merged = [&](SkSel in_sel) {
    for (size_t i = 0; i < mats.size(); i++) {
      const Mat& m = mats[i]; auto o = out_of(m.type);
      if (m.type == T_AB) continue;
      if (m.type == T_IN_GDN) {
        const Mat& b2 = mats[i + 1];
        SkArgs a{A, AS, AG, R, m.K, m.N, m.W, nullptr, o.p, o.ld, m.RS, ws, cnt, 5, counter(), sk_chunk(in_sel.shape, 3, m.N, m.K, 96, nsm * in_sel.per_sm, in_sel.items)};
        a.W2 = b2.W; a.S2 = b2.S; a.RS2 = b2.RS; a.N2 = b2.N; a.out2 = ab; a.ldo2 = 96;
        ck(sk_launch(in_sel.shape, 3, SK_ATOMIC, in_sel.var, a, nsm * in_sel.per_sm, st), "sk2");
      } else sk_launch_m(m, o.p, o.ld, best[m.type]);
    }
  };
  const double tg = time_graph([&] { merged(best[T_IN_GDN]); });
  printf("  sk, ab merged   %8.1f us %7.0f GB/s\n", tg, b / tg / 1e3);
  {  // correctness of the merged launch against separate stream-K launches
    const Mat& m = *std::find_if(mats.begin(), mats.end(), [](const Mat& q) { return q.type == T_IN_GDN; });
    const Mat& b2 = (&m)[1];
    std::vector<float> h1((size_t)R * GU), h2((size_t)R * 96), g1(h1.size()), g2(h2.size());
    ck(cudaMemsetAsync(proj, 0, (size_t)8 * GU * 4, st), "c"); ck(cudaMemsetAsync(ab, 0, 8 * 96 * 4, st), "c");
    sk_launch_m(m, proj, GU, best[T_IN_GDN]); sk_launch_m(b2, ab, 96, best[T_AB]);
    ck(cudaMemcpyAsync(h1.data(), proj, h1.size() * 4, cudaMemcpyDeviceToHost, st), "c"); ck(cudaMemcpyAsync(h2.data(), ab, h2.size() * 4, cudaMemcpyDeviceToHost, st), "c");
    ck(cudaMemsetAsync(proj, 0, (size_t)8 * GU * 4, st), "c"); ck(cudaMemsetAsync(ab, 0, 8 * 96 * 4, st), "c");
    SkArgs a{A, AS, AG, R, m.K, m.N, m.W, nullptr, proj, GU, m.RS, ws, cnt, 5, counter(), sk_chunk(best[T_IN_GDN].shape, 3, m.N, m.K, 96, nsm * best[T_IN_GDN].per_sm, best[T_IN_GDN].items)};
    a.W2 = b2.W; a.S2 = b2.S; a.RS2 = b2.RS; a.N2 = b2.N; a.out2 = ab; a.ldo2 = 96;
    ck(sk_launch(best[T_IN_GDN].shape, 3, SK_ATOMIC, best[T_IN_GDN].var, a, nsm * best[T_IN_GDN].per_sm, st), "sk2");
    ck(cudaMemcpyAsync(g1.data(), proj, g1.size() * 4, cudaMemcpyDeviceToHost, st), "c"); ck(cudaMemcpyAsync(g2.data(), ab, g2.size() * 4, cudaMemcpyDeviceToHost, st), "c");
    ck(cudaStreamSynchronize(st), "c");
    double d1 = 0, m1 = 0, d2 = 0, m2 = 0;
    for (int r = 0; r < R; r++) { for (int n = 0; n < m.N; n++) { size_t k = (size_t)r * GU + n; d1 = std::max(d1, (double)fabsf(h1[k] - g1[k])); m1 = std::max(m1, (double)fabsf(h1[k])); }
      for (int n = 0; n < 96; n++) { size_t k = (size_t)r * 96 + n; d2 = std::max(d2, (double)fabsf(h2[k] - g2[k])); m2 = std::max(m2, (double)fabsf(h2[k])); } }
    printf("  merged correctness: in %.2e, ab %.2e (max %.3g)\n", d1 / std::max(m1, 1e-30), d2 / std::max(m2, 1e-30), m2);
  }
  const double tc = time_graph([&] { for (auto& m : mats) { auto o = out_of(m.type); SkSel c = best[m.type]; c.var = 1; sk_launch_m(m, o.p, o.ld, c); } });
  printf("  copy-only       %8.1f us %7.0f GB/s\n", tc, b / tc / 1e3);
  for (int type = 0; type < NTYPE; type++) { const SkShape s = sk_shape(best[type].shape); printf("  best %-7s %d:%d:%d (fw%d wg%d st%d)\n", tname[type], best[type].shape, best[type].per_sm, best[type].items, s.fw, s.wg, s.stages); }
  return 0;
}
