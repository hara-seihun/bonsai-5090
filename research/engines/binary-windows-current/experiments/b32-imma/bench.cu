// Real-weight keyhole: b32 integer-tensor projection versus the engine's stable DP4A route.
// Both consume identical A8 integers (same max/127 round-to-nearest-even quantization), so the
// comparison isolates FP32 association. Every timed graph includes its own activation producer.
#include "b32.cuh"
#include "../binary-bitserial/bitserial.cuh"
#include "gguf.h"
#ifdef B32_CLOCKS
namespace b32 { void clock_stats(unsigned long long* out); void timeline(unsigned long long* out, int n); }
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
static unsigned mix(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16); }
struct Memory {
  std::vector<void*> p;
  template <class T> T* alloc(size_t n) { T* q = nullptr; CUDA(cudaMalloc(&q, std::max<size_t>(1, n) * sizeof(T))); CUDA(cudaMemset(q, 0, std::max<size_t>(1, n) * sizeof(T))); p.push_back(q); return q; }
  ~Memory() { for (auto q : p) cudaFree(q); }
};
struct Graph { cudaGraph_t g = nullptr; cudaGraphExec_t e = nullptr; ~Graph() { if (e) cudaGraphExecDestroy(e); if (g) cudaGraphDestroy(g); } };
template <class F> static Graph* capture(cudaStream_t st, F f) {
  Graph* g = new Graph; CUDA(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); f();
  CUDA(cudaStreamEndCapture(st, &g->g)); CUDA(cudaGraphInstantiate(&g->e, g->g, 0)); return g;
}
__global__ void evict(unsigned* p, size_t n) { size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) p[i] += 1; }
static float timeit(Graph* g, cudaStream_t st, unsigned* trash, size_t tn, bool cold, int reps = 11) {
  cudaEvent_t a, b; CUDA(cudaEventCreate(&a)); CUDA(cudaEventCreate(&b)); std::vector<float> s;
  for (int i = 0; i < reps + 2; ++i) {
    if (cold) { evict<<<unsigned((tn + 255) / 256), 256, 0, st>>>(trash, tn); }
    CUDA(cudaEventRecord(a, st)); CUDA(cudaGraphLaunch(g->e, st)); CUDA(cudaEventRecord(b, st)); CUDA(cudaEventSynchronize(b));
    float ms; CUDA(cudaEventElapsedTime(&ms, a, b)); if (i >= 2) s.push_back(ms * 1000.f);
  }
  cudaEventDestroy(a); cudaEventDestroy(b); std::sort(s.begin(), s.end()); return s[s.size() / 2];
}
struct Err { double maxabs, rel_rms, maxref; };
static Err compare(const std::vector<float>& a, const std::vector<float>& b, int n, int r, int lda, int ldb) {
  Err e{}; double sq = 0, rq = 0;
  for (int t = 0; t < r; ++t) for (int j = 0; j < n; ++j) {
    double x = a[size_t(t) * lda + j], y = b[size_t(t) * ldb + j];
    if (!std::isfinite(x) || !std::isfinite(y)) { fprintf(stderr, "nonfinite at t%d j%d: %g %g\n", t, j, x, y); std::exit(3); }
    double d = x - y; e.maxabs = std::max(e.maxabs, std::abs(d)); e.maxref = std::max(e.maxref, std::abs(y)); sq += d * d; rq += y * y;
  }
  e.rel_rms = std::sqrt(sq / std::max(rq, 1e-30)); return e;
}
// Exact A8 oracle on the CPU for sampled rows: raw Q1_0 blocks (FP16 scale + 16 LSB-first sign bytes) and producer bytes.
static double half_to_float(uint16_t h) { return (double)__half2float(*reinterpret_cast<const __half*>(&h)); }

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: bench MODEL TENSOR R[,R..] VARIANT[,..]|all [OUT.jsonl]\n"); return 1; }
  h5::Gguf model; model.open(argv[1]);
  std::string spec = argv[2]; std::vector<std::string> names;
  if (spec == "gu") names = {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"};
  else if (spec == "down") names = {"blk.0.ffn_down.weight"};
  else if (spec == "qkvz") names = {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"};
  else if (spec == "ab") names = {"blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"};
  else if (spec == "gdnout") names = {"blk.0.ssm_out.weight"};
  else if (spec == "attnin") names = {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"};
  else if (spec == "attnout") names = {"blk.3.attn_output.weight"};
  else if (spec == "head") names = {"output.weight"};
  else names = {spec};
  std::vector<uint8_t> raw; int k = 0, n = 0;
  for (auto& nm : names) {
    const auto& t = model.t(nm);
    if (t.type != 41 || t.ne.size() != 2) { fprintf(stderr, "%s not Q1_0 rank2\n", nm.c_str()); return 1; }
    if (k && int(t.ne[0]) != k) { fprintf(stderr, "K mismatch\n"); return 1; }
    k = int(t.ne[0]); n += int(t.ne[1]); raw.insert(raw.end(), t.data, t.data + t.bytes);
  }
  std::vector<int> rows; { std::string s = argv[3]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); rows.push_back(std::stoi(s.substr(p, q - p))); if (q == std::string::npos) break; p = q + 1; } }
  std::vector<int> vars; { std::string s = argv[4]; if (s == "all") for (int v = 0; v < b32::variants(); ++v) vars.push_back(v); else { size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); vars.push_back(std::stoi(s.substr(p, q - p))); if (q == std::string::npos) break; p = q + 1; } } }
  FILE* out = argc > 5 ? fopen(argv[5], "a") : stdout;
  cudaStream_t st; CUDA(cudaStreamCreate(&st));
  Memory m; const int kb = k / 128; const size_t blocks = size_t(n) * kb;
  auto draw = m.alloc<uint8_t>(raw.size()); CUDA(cudaMemcpy(draw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
  auto signs = m.alloc<uint32_t>(blocks * 4); auto wsc = m.alloc<__half>(blocks);
  CUDA(bitserial::prepare(draw, signs, wsc, n, k, st));
  const size_t trash_n = (256u << 20) / 4; auto trash = m.alloc<unsigned>(trash_n);
  const int R = 32;
  auto x = m.alloc<float>(size_t(R) * k);
  {
    std::vector<float> hx(size_t(R) * k);
    for (int t = 0; t < R; ++t) for (int i = 0; i < k; ++i) {
      unsigned h = mix(unsigned(t * 1000003 + i) + 17);
      float base = (int(h % 65536) - 32768) / 32768.f;
      float blockscale = 0.05f + 4.f * ((mix(unsigned(t * 7919 + i / 128)) % 1000) / 1000.f);  // per-K128 dynamic range
      float spike = (h >> 28) == 15 ? 6.f : 1.f;                                                // heavy tail
      hx[size_t(t) * k + i] = base * blockscale * spike;
    }
    CUDA(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
  }
  // Reference carrier (engine A8) and DP4A outputs
  bitserial::Carrier rc{}; rc.q = m.alloc<int8_t>(size_t(R) * k); rc.scales = m.alloc<float>(size_t(R) * kb);
  rc.dc = m.alloc<int32_t>(size_t(R) * kb); rc.status = m.alloc<bitserial::InputStatus>(size_t(R) * kb);
  b32::Carrier bc{}; bc.q = m.alloc<uint32_t>(b32::carrier_bytes(k) / 4); bc.error = m.alloc<uint32_t>(1);
  const int ldy = n + 64;  // explicit strided output
  auto yref = m.alloc<float>(size_t(R) * n); auto y = m.alloc<float>(size_t(R) * ldy);
  for (int r : rows) {
    // DP4A reference: tiles of <=16 rows, separate partial buffers
    std::vector<bitserial::Plan> plans; std::vector<float*> partials;
    for (int t = 0; t < r; t += 16) { bitserial::Plan p{}; CUDA(bitserial::select(n, k, std::min(16, r - t), &p)); plans.push_back(p); partials.push_back(m.alloc<float>(bitserial::partial_bytes(p) / 4 + 1)); }
    auto dp_consume = [&]() {
      int t = 0;
      for (size_t i = 0; i < plans.size(); ++i) {
        bitserial::Carrier v = rc; v.q += size_t(t) * k; v.scales += size_t(t) * kb; v.dc += size_t(t) * kb; v.status += size_t(t) * kb;
        CUDA(bitserial::project_carrier(plans[i], signs, wsc, v, yref + size_t(t) * n, partials[i], 1.f, 0.f, st)); t += plans[i].r;
      }
    };
    auto dp_all = [&]() { CUDA(bitserial::produce(bitserial::Route::Dp4a01, x, k, r, rc, st)); dp_consume(); };
    dp_all(); CUDA(cudaStreamSynchronize(st));
    std::vector<float> href(size_t(r) * n); CUDA(cudaMemcpy(href.data(), yref, href.size() * 4, cudaMemcpyDeviceToHost));
    // CPU exact-A8 oracle on sampled rows
    {
      std::vector<int8_t> hq(size_t(r) * k); std::vector<float> hs(size_t(r) * kb);
      CUDA(cudaMemcpy(hq.data(), rc.q, hq.size(), cudaMemcpyDeviceToHost)); CUDA(cudaMemcpy(hs.data(), rc.scales, hs.size() * 4, cudaMemcpyDeviceToHost));
      double worst = 0;
      for (int s = 0; s < 64; ++s) {
        int row = int(mix(unsigned(s) * 31 + 5) % unsigned(n)), t = int(mix(unsigned(s) * 17 + 3) % unsigned(r));
        double total = 0, mag = 0;
        for (int g = 0; g < kb; ++g) {
          const uint8_t* bl = raw.data() + (size_t(row) * kb + g) * 18; uint16_t hs16 = uint16_t(bl[0] | (bl[1] << 8));
          long dot = 0; for (int j = 0; j < 128; ++j) { int qv = hq[size_t(t) * k + g * 128 + j]; dot += ((bl[2 + j / 8] >> (j % 8)) & 1) ? qv : -qv; }
          total += double(dot) * half_to_float(hs16) * hs[size_t(t) * kb + g]; mag += std::abs(double(dot) * half_to_float(hs16) * hs[size_t(t) * kb + g]);
        }
        worst = std::max(worst, std::abs(total - href[size_t(t) * n + row]) / std::max(mag, 1e-30));
      }
      fprintf(out, "{\"kind\":\"oracle\",\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"dp4a_vs_exact_a8_max_rel_to_abs_sum\":%.3g}\n", spec.c_str(), n, k, r, worst);
    }
    Graph* gdp = capture(st, dp_all);
    Graph* gdpc = capture(st, dp_consume);
    const float dp_cold = timeit(gdp, st, trash, trash_n, true), dp_warm = timeit(gdp, st, trash, trash_n, false), dpc_cold = timeit(gdpc, st, trash, trash_n, true);
    fprintf(out, "{\"kind\":\"timing\",\"route\":\"dp4a\",\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"inclusive_cold_us\":%.3f,\"inclusive_warm_us\":%.3f,\"carrier_cold_us\":%.3f}\n", spec.c_str(), n, k, r, dp_cold, dp_warm, dpc_cold);
    fflush(out); delete gdp; delete gdpc;
    for (int v : vars) {
      b32::Plan p{}; if (b32::select(n, k, r, &p, v) != cudaSuccess) continue;
      void* ws = m.alloc<unsigned char>(b32::workspace_bytes(p));
      const int tn = b32::plan_tile_rows(v);
      uint8_t* tiled = m.alloc<uint8_t>(b32::tiled_bytes(n, k, tn));
      CUDA(b32::tile_from_packed1(signs, wsc, n, k, tn, tiled, st));
      auto mine_consume = [&](float beta) { CUDA(b32::project(p, tiled, bc, y, ldy, beta, ws, st)); };
      auto mine_all = [&]() { CUDA(b32::produce(x, k, k, r, bc, st)); mine_consume(0.f); };
      // correctness: beta 0, then beta 1 accumulates the same product again
      CUDA(cudaMemsetAsync(y, 0, size_t(R) * ldy * 4, st));
      mine_all(); CUDA(cudaStreamSynchronize(st));
      std::vector<float> hy(size_t(r) * ldy); CUDA(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
      Err e{}; if (!getenv("BENCH_NOCHECK")) e = compare(hy, href, n, r, ldy, n);
      mine_consume(1.f); CUDA(cudaStreamSynchronize(st));
      std::vector<float> hy2(size_t(r) * ldy); CUDA(cudaMemcpy(hy2.data(), y, hy2.size() * 4, cudaMemcpyDeviceToHost));
      double beta_err = 0; for (int t = 0; t < r; ++t) for (int j = 0; j < n; ++j) beta_err = std::max(beta_err, (double)std::abs(hy2[size_t(t) * ldy + j] - 2.f * hy[size_t(t) * ldy + j]));
      bool guard_ok = true; for (int t = 0; t < r; ++t) for (int j = n; j < ldy; ++j) guard_ok &= hy2[size_t(t) * ldy + j] == 0.f;
      // determinism: rerun and require bitwise equality
      mine_all(); CUDA(cudaStreamSynchronize(st)); std::vector<float> hy3(size_t(r) * ldy); CUDA(cudaMemcpy(hy3.data(), y, hy3.size() * 4, cudaMemcpyDeviceToHost));
      bool deterministic = true; for (int t = 0; t < r; ++t) for (int j = 0; j < n; ++j) deterministic &= hy3[size_t(t) * ldy + j] == hy[size_t(t) * ldy + j];
      Graph* ga = capture(st, mine_all);
      Graph* gc = capture(st, [&]() { mine_consume(0.f); });
#ifdef B32_CLOCKS
      { unsigned long long z[6]; b32::clock_stats(z); }
#endif
      const float cold = timeit(ga, st, trash, trash_n, true), warm = timeit(ga, st, trash, trash_n, false), ccold = timeit(gc, st, trash, trash_n, true), cwarm = timeit(gc, st, trash, trash_n, false);
#ifdef B32_TIMELINE
      {
        CUDA(cudaGraphLaunch(gc->e, st)); CUDA(cudaStreamSynchronize(st));
        std::vector<unsigned long long> tl(4 * p.ctas); b32::timeline(tl.data(), p.ctas);
        unsigned long long s0 = ~0ull, s1 = 0, e0 = ~0ull, e1 = 0, f0 = ~0ull, f1 = 0;
        for (int c = 0; c < p.ctas; ++c) { s0 = std::min(s0, tl[4*c]); s1 = std::max(s1, tl[4*c]); e0 = std::min(e0, tl[4*c+1]); e1 = std::max(e1, tl[4*c+1]); f0 = std::min(f0, tl[4*c+2]); f1 = std::max(f1, tl[4*c+2]); }
        printf("timeline %s %s: start spread %.2f us, first-stage ready %.2f..%.2f us, end %.2f..%.2f us (from first start)\n", spec.c_str(), b32::variant_name(v),
               (s1 - s0) / 1e3, (f0 - s0) / 1e3, (f1 - s0) / 1e3, (e0 - s0) / 1e3, (e1 - s0) / 1e3);
      }
#endif
#ifdef B32_CLOCKS
      { unsigned long long z[6]; b32::clock_stats(z); printf("clocks %s: per consumer-warp stage: wait %.0f compute %.0f total %.0f cycles; producer empty-wait per stage %.0f; stages/warp %.1f\n", b32::variant_name(v), double(z[0]) / z[4], double(z[1]) / z[4], double(z[2]) / z[4], double(z[5]) * (z[3] / (double)z[4]) / z[4] * 0 + double(z[5]) / (z[4] / (double)(z[3] / 8.0)) , double(z[4]) / z[3]); }
#endif
      const double wbytes = double(blocks) * 18.0;
      cudaFree(tiled); m.p.pop_back();
      fprintf(out, "{\"kind\":\"timing\",\"route\":\"b32\",\"variant\":\"%s\",\"ctas\":%d,\"tensor\":\"%s\",\"n\":%d,\"k\":%d,\"r\":%d,\"inclusive_cold_us\":%.3f,\"inclusive_warm_us\":%.3f,\"carrier_cold_us\":%.3f,\"carrier_warm_us\":%.3f,\"weight_gbps_cold\":%.1f,\"vs_dp4a_inclusive_cold\":%.3f,\"maxabs\":%.4g,\"rel_rms\":%.4g,\"maxref\":%.4g,\"beta_maxerr\":%.4g,\"guards\":%s,\"deterministic\":%s}\n",
              b32::variant_name(v), p.ctas, spec.c_str(), n, k, r, cold, warm, ccold, cwarm, wbytes / (ccold * 1e-6) / 1e9, dp_cold / cold, e.maxabs, e.rel_rms, e.maxref, beta_err,
              guard_ok ? "true" : "false", deterministic ? "true" : "false");
      fflush(out); delete ga; delete gc;
    }
  }
  // Fused GU (SiLU*up -> next carrier) and split-store contracts against the plain b32 store.
  if (spec == "gu") {
    for (int r : rows) {
      const int ff = n / 2;
      b32::Plan p{}, ps{};
      if (b32::select(n, k, r, &p, -1, 0) != cudaSuccess || b32::select(n, k, r, &ps, -1, 2) != cudaSuccess) { fprintf(stderr, "gu plans\n"); return 3; }
      const int tn = b32::plan_tile_rows(p.variant);
      uint8_t* tiled = m.alloc<uint8_t>(b32::tiled_bytes(n, k, tn)); CUDA(b32::tile_from_packed1(signs, wsc, n, k, tn, tiled, st));
      uint8_t* tiled_gu = m.alloc<uint8_t>(b32::tiled_bytes(n, k, 256)); CUDA(b32::tile_gu_from_packed1(signs, wsc, ff, k, tiled_gu, st));
      void* ws = m.alloc<unsigned char>(std::max(b32::workspace_bytes(p), b32::workspace_bytes(ps)));
      b32::Carrier hc{}; hc.q = m.alloc<uint32_t>(b32::carrier_bytes(ff) / 4); hc.error = m.alloc<uint32_t>(1);
      CUDA(b32::produce(x, k, k, r, bc, st));
      CUDA(b32::project(p, tiled, bc, y, ldy, 0.f, ws, st));
      CUDA(b32::project_silu(ps, tiled_gu, bc, hc, ws, st));
      CUDA(cudaStreamSynchronize(st));
      std::vector<float> hy(size_t(r) * ldy); CUDA(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<uint32_t> hq(b32::carrier_bytes(ff) / 4); CUDA(cudaMemcpy(hq.data(), hc.q, hq.size() * 4, cudaMemcpyDeviceToHost));
      uint32_t herr = 0; CUDA(cudaMemcpy(&herr, hc.error, 4, cudaMemcpyDeviceToHost));
      long mism = 0, big = 0, total = 0, dcbad = 0; double sigrel = 0;
      for (int t = 0; t < r; ++t) for (int blk = 0; blk < ff / 128; ++blk) {
        float hv[128]; float mx = 0;
        for (int f = 0; f < 128; ++f) { const int row = blk * 128 + f; const float g = hy[size_t(t) * ldy + row], u = hy[size_t(t) * ldy + ff + row]; hv[f] = g / (1.f + std::exp(-g)) * u; mx = std::max(mx, std::fabs(hv[f])); }
        const uint32_t* cb = hq.data() + (size_t)blk * 1152;
        const float sig = reinterpret_cast<const float*>(cb + 1024)[t];
        const int md = reinterpret_cast<const int*>(cb + 1024 + 32)[((t >> 3) * 4 + ((t & 7) >> 1)) * 4 + (t & 1)];
        const float s8 = b32::sigma8(mx);
        sigrel = std::max(sigrel, (double)std::fabs(sig - s8) / std::max(1e-30f, s8));
        int dc = 0;
        for (int f = 0; f < 128; ++f) {
          const int c = f >> 5, kp = f & 31, jw = kp & 7, i = kp >> 3, nt = t >> 3;
          const int word = ((nt * 2 + (c >> 1)) * 32 + 4 * (t & 7) + (jw & 3)) * 4 + (c & 1) * 2 + (jw >> 2);
          const int qv = int(int8_t((cb[word] >> (8 * i)) & 255)); dc += qv;
          const int qr = mx > 0 ? (int)std::nearbyint(hv[f] * (1.f / s8)) : 0;
          ++total; if (qv != qr) { ++mism; if (std::abs(qv - qr) > 1) ++big; }
        }
        if (md != b32::MAGIC - dc) ++dcbad;
      }
      fprintf(out, "{\"kind\":\"silu_contract\",\"r\":%d,\"bytes\":%ld,\"q_mismatch\":%ld,\"q_mismatch_gt1\":%ld,\"sigma_max_rel\":%.3g,\"dc_inconsistent\":%ld,\"error\":%u}\n", r, total, mism, big, sigrel, dcbad, herr);
      Graph* gs = capture(st, [&]() { CUDA(b32::project_silu(ps, tiled_gu, bc, hc, ws, st)); });
      Graph* gp = capture(st, [&]() { CUDA(b32::project(p, tiled, bc, y, ldy, 0.f, ws, st)); });
      fprintf(out, "{\"kind\":\"silu_timing\",\"r\":%d,\"silu_cold_us\":%.3f,\"silu_warm_us\":%.3f,\"plain_cold_us\":%.3f,\"plain_warm_us\":%.3f,\"ctas\":%d}\n", r,
              timeit(gs, st, trash, trash_n, true), timeit(gs, st, trash, trash_n, false), timeit(gp, st, trash, trash_n, true), timeit(gp, st, trash, trash_n, false), ps.ctas);
      fflush(out); delete gs; delete gp;
    }
  }
  if (spec == "qkvz") {
    // split store: rows >= 10240 (z) to a second buffer
    for (int r : rows) {
      b32::Plan p{}, ps{}; CUDA(b32::select(n, k, r, &p, -1, 0)); CUDA(b32::select(n, k, r, &ps, -1, 1));
      const int tn = b32::plan_tile_rows(p.variant);
      uint8_t* tiled = m.alloc<uint8_t>(b32::tiled_bytes(n, k, tn)); CUDA(b32::tile_from_packed1(signs, wsc, n, k, tn, tiled, st));
      void* ws = m.alloc<unsigned char>(std::max(b32::workspace_bytes(p), b32::workspace_bytes(ps)));
      const int split = 10240; float* y2 = m.alloc<float>(size_t(R) * (n - split));
      CUDA(b32::produce(x, k, k, r, bc, st)); CUDA(b32::project(p, tiled, bc, y, ldy, 0.f, ws, st)); CUDA(cudaStreamSynchronize(st));
      std::vector<float> a(size_t(r) * ldy); CUDA(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
      CUDA(cudaMemsetAsync(y, 0, size_t(R) * ldy * 4, st));
      CUDA(b32::project_split(ps, tiled, bc, y, ldy, split, y2, n - split, ws, st)); CUDA(cudaStreamSynchronize(st));
      std::vector<float> s1(size_t(r) * ldy), s2(size_t(r) * (n - split));
      CUDA(cudaMemcpy(s1.data(), y, s1.size() * 4, cudaMemcpyDeviceToHost)); CUDA(cudaMemcpy(s2.data(), y2, s2.size() * 4, cudaMemcpyDeviceToHost));
      long diff = 0; for (int t = 0; t < r; ++t) for (int j = 0; j < n; ++j) { const float v = j < split ? s1[size_t(t) * ldy + j] : s2[size_t(t) * (n - split) + j - split]; diff += v != a[size_t(t) * ldy + j]; }
      fprintf(out, "{\"kind\":\"split_contract\",\"r\":%d,\"bitwise_differences\":%ld}\n", r, diff); fflush(out);
    }
  }
  if (out != stdout) fclose(out);
  return 0;
}
