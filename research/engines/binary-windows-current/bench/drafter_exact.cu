// Exactness probe for the drafter latency changes (run on the GPU; prints one JSON line, exit 0 on pass):
//  1. split top-k (H5_TOPK_SPLIT=1) returns the same top-16 values and indices, in the same order,
//     as the one-CTA-per-row k_topk, on random logits and on logits with many exact ties;
//  2. chunk-route prep with output clearing (drafter precleared path) writes the same NVFP4 bytes
//     and row scales as without clearing, zeroes exactly the declared rows x stride, and nothing else;
//  3. the fused dynamic-conv prep (P_DCONV) quantizes exactly what k_dconv followed by the
//     quantize-only prep produces, across block boundaries and both conv halves;
//  4. fused ingest: one quantize-only chunk prep over the five concatenated capture layers writes the
//     same NVFP4 blocks as five per-layer preps, and one five-layer attention-prep launch writes the
//     same drafter K/V ring entries as five per-layer launches;
//  5. the interleaved gate/up SiLU prep (P_SILU_IL, verifier split-K GU) prepares exactly what the
//     de-interleaved P_SILU_MUL prep does, and with P_CONSUME_CLEAR also leaves its input all-zero.
// Build: make drafter-exact-test (links build/kernels.o build/draft.o).
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace h5 {
void launch_prep(const PrepArgs&, int, cudaStream_t);
void launch_topk(const float*, int, const int*, float*, int*, int, TopkScratch, cudaStream_t);
void launch_dconv(const float*, const float*, int, const float*, float*, int, int, int, cudaStream_t);
void launch_dattn_pre(const DAttnPreArgs&, int, cudaStream_t);
}
static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <class T> static T* dev(size_t n) { T* p; ck(cudaMalloc(&p, n * sizeof(T))); ck(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <class T> static std::vector<T> get(const T* p, size_t n) { std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
template <class T> static void same(const std::vector<T>& a, const std::vector<T>& b, const std::string& what) {
  if (a.size() != b.size() || memcmp(a.data(), b.data(), a.size() * sizeof(T))) throw std::runtime_error("mismatch: " + what);
}

static void topk_case(const char* name, int rows, bool ties, std::mt19937& rng, cudaStream_t st) {
  std::normal_distribution<float> nd(0.f, 3.f);
  std::vector<float> h((size_t)rows * h5::VOCAB);
  for (auto& x : h) { x = nd(rng); if (ties) x = std::round(x * 8.f) / 8.f; }
  float* l = dev<float>(h.size()); ck(cudaMemcpy(l, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  float* v0 = dev<float>(rows * h5::DF_TOPK); int* i0 = dev<int>(rows * h5::DF_TOPK);
  float* v1 = dev<float>(rows * h5::DF_TOPK); int* i1 = dev<int>(rows * h5::DF_TOPK);
  h5::TopkScratch scratch{dev<float>((size_t)rows * h5::TOPK_PARTS_PER_ROW),dev<int>((size_t)rows * h5::TOPK_PARTS_PER_ROW),rows};
  setenv("H5_TOPK_SPLIT", "0", 1); h5::launch_topk(l, h5::VOCAB, nullptr, v0, i0, rows, scratch, st);
  setenv("H5_TOPK_SPLIT", "1", 1); h5::launch_topk(l, h5::VOCAB, nullptr, v1, i1, rows, scratch, st);
  ck(cudaStreamSynchronize(st));
  same(get(v0, rows * h5::DF_TOPK), get(v1, rows * h5::DF_TOPK), std::string(name) + " values");
  same(get(i0, rows * h5::DF_TOPK), get(i1, rows * h5::DF_TOPK), std::string(name) + " indices");
  // host reference: value descending, index ascending
  auto iv = get(i1, rows * h5::DF_TOPK);
  for (int r = 0; r < rows; r++) {
    std::vector<int> idx(h5::VOCAB); for (int k = 0; k < h5::VOCAB; k++) idx[k] = k;
    const float* row = h.data() + (size_t)r * h5::VOCAB;
    std::partial_sort(idx.begin(), idx.begin() + h5::DF_TOPK, idx.end(), [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
    for (int k = 0; k < h5::DF_TOPK; k++) if (iv[r * h5::DF_TOPK + k] != idx[k]) throw std::runtime_error(std::string(name) + " differs from host reference");
  }
  cudaFree(l); cudaFree(v0); cudaFree(i0); cudaFree(v1); cudaFree(i1);cudaFree(scratch.values);cudaFree(scratch.indices);
}

static void topk_ownership_case() {
  constexpr int rows=128,n=65536;
  struct Work { float *logits,*values;int* indices;h5::TopkScratch scratch;cudaStream_t stream; } work[2];
  setenv("H5_TOPK_SPLIT","1",1);
  for (int id=0;id<2;id++) {
    auto& w=work[id];ck(cudaStreamCreateWithFlags(&w.stream,cudaStreamNonBlocking));
    std::vector<float> input((size_t)rows*n,-100.f);
    for (int r=0;r<rows;r++) for (int k=0;k<h5::DF_TOPK;k++) input[(size_t)r*n+((r%32)*37+k*31+id*16384)%n]=100.f*id+16-k;
    w.logits=dev<float>(input.size());ck(cudaMemcpy(w.logits,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    w.values=dev<float>(rows*h5::DF_TOPK);w.indices=dev<int>(rows*h5::DF_TOPK);
    w.scratch={dev<float>((size_t)rows*h5::TOPK_PARTS_PER_ROW),dev<int>((size_t)rows*h5::TOPK_PARTS_PER_ROW),rows};
  }
  for (int repeat=0;repeat<8;repeat++) for (auto& w:work)
    h5::launch_topk(w.logits,n,nullptr,w.values,w.indices,rows,w.scratch,w.stream);
  for (int id=0;id<2;id++) {
    auto& w=work[id];ck(cudaStreamSynchronize(w.stream));
    const auto values=get(w.values,rows*h5::DF_TOPK);const auto indices=get(w.indices,rows*h5::DF_TOPK);
    for (int r=0;r<rows;r++) for (int k=0;k<h5::DF_TOPK;k++)
      if (values[r*h5::DF_TOPK+k]!=100.f*id+16-k || indices[r*h5::DF_TOPK+k]!=((r%32)*37+k*31+id*16384)%n)
        throw std::runtime_error("independent top-k streams aliased scratch");
    ck(cudaFree(w.logits));ck(cudaFree(w.values));ck(cudaFree(w.indices));ck(cudaFree(w.scratch.values));ck(cudaFree(w.scratch.indices));ck(cudaStreamDestroy(w.stream));
  }
}

static void prep_case(const char* name, unsigned flags, int n, int ldx, int rows, int stride, std::mt19937& rng, cudaStream_t st) {
  constexpr int GUARD = 64;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hx((size_t)rows * ldx); for (auto& x : hx) x = nd(rng);
  float* x = dev<float>(hx.size()); ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
  struct Out { uint8_t *q4, *s4; float* g4; } o[2];
  for (auto& b : o) { b.q4 = dev<uint8_t>((size_t)64 * n / 2); b.s4 = dev<uint8_t>((size_t)64 * n / 16); b.g4 = dev<float>(64); }
  float* clear = dev<float>((size_t)rows * stride + 2 * GUARD);
  ck(cudaMemset(clear, 0xff, ((size_t)rows * stride + 2 * GUARD) * 4));
  for (int k = 0; k < 2; k++) {
    h5::PrepArgs a{}; a.x = x; a.ldx = ldx; a.n = n; a.flags = flags; a.q4 = o[k].q4; a.s4 = o[k].s4; a.g4 = o[k].g4;
    if (flags & h5::P_SILU_MUL) a.x2 = x + n;
    if (k) { a.clear_a = clear + GUARD; a.clear_a_stride = stride; }
    h5::launch_prep(a, rows, st);
  }
  ck(cudaStreamSynchronize(st));
  same(get(o[0].q4, (size_t)64 * n / 2), get(o[1].q4, (size_t)64 * n / 2), std::string(name) + " q4");
  same(get(o[0].s4, (size_t)64 * n / 16), get(o[1].s4, (size_t)64 * n / 16), std::string(name) + " s4");
  same(get(o[0].g4, 64), get(o[1].g4, 64), std::string(name) + " g4");
  auto c = get(reinterpret_cast<uint32_t*>(clear), (size_t)rows * stride + 2 * GUARD);
  for (size_t i = 0; i < c.size(); i++) {
    const bool inside = i >= GUARD && i < GUARD + (size_t)rows * stride;
    if (c[i] != (inside ? 0u : 0xffffffffu)) throw std::runtime_error(std::string(name) + (inside ? " left output uncleared" : " wrote outside clear region"));
  }
  cudaFree(x); cudaFree(clear); for (auto& b : o) { cudaFree(b.q4); cudaFree(b.s4); cudaFree(b.g4); }
}

static void dconv_case(const char* name, int rows, int block_rows, int which, std::mt19937& rng, cudaStream_t st) {
  std::normal_distribution<float> nd(0.f, 1.f);
  auto upload = [&](size_t n) { std::vector<float> h(n); for (auto& x : h) x = nd(rng); float* d = dev<float>(n); ck(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice)); return d; };
  float* src = upload((size_t)rows * h5::D); float* dyn = upload((size_t)rows * h5::DF_KPROJ); float* base = upload((size_t)4 * h5::D);
  float* cbuf = dev<float>((size_t)rows * h5::D);
  struct Out { uint8_t *q4, *s4; float* g4; } o[2];
  for (auto& b : o) { b.q4 = dev<uint8_t>((size_t)64 * h5::D / 2); b.s4 = dev<uint8_t>((size_t)64 * h5::D / 16); b.g4 = dev<float>(64); }
  h5::launch_dconv(src, dyn, which, base, cbuf, 0, rows, block_rows, st);
  h5::PrepArgs a{}; a.x = cbuf; a.ldx = h5::D; a.n = h5::D; a.flags = h5::P_QUANT; a.q4 = o[0].q4; a.s4 = o[0].s4; a.g4 = o[0].g4;
  h5::launch_prep(a, rows, st);
  h5::PrepArgs f{}; f.x = src; f.ldx = h5::D; f.n = h5::D; f.flags = h5::P_DCONV | h5::P_QUANT; f.q4 = o[1].q4; f.s4 = o[1].s4; f.g4 = o[1].g4;
  f.dc_dyn = dyn; f.dc_base = base; f.dc_which = which; f.dc_block_rows = block_rows;
  h5::launch_prep(f, rows, st);
  ck(cudaStreamSynchronize(st));
  same(get(o[0].q4, (size_t)64 * h5::D / 2), get(o[1].q4, (size_t)64 * h5::D / 2), std::string(name) + " q4");
  same(get(o[0].s4, (size_t)64 * h5::D / 16), get(o[1].s4, (size_t)64 * h5::D / 16), std::string(name) + " s4");
  same(get(o[0].g4, 64), get(o[1].g4, 64), std::string(name) + " g4");
  for (float* p : {src, dyn, base, cbuf}) cudaFree(p);
  for (auto& b : o) { cudaFree(b.q4); cudaFree(b.s4); cudaFree(b.g4); }
}

static void ingest_case(std::mt19937& rng, cudaStream_t st) {
  constexpr int NC = 5, W = NC * h5::D, ROWS = 3;
  std::normal_distribution<float> nd(0.f, 1.f);
  // capture rows live at hcap rows 4, 1, 6 (rowmap), width NC*D
  std::vector<float> hh((size_t)8 * W); for (auto& x : hh) x = nd(rng);
  float* hcap = dev<float>(hh.size()); ck(cudaMemcpy(hcap, hh.data(), hh.size() * 4, cudaMemcpyHostToDevice));
  std::vector<int> map = {4, 1, 6}; int* rm = dev<int>(ROWS); ck(cudaMemcpy(rm, map.data(), ROWS * 4, cudaMemcpyHostToDevice));
  uint8_t* fq = dev<uint8_t>((size_t)64 * W / 2); uint8_t* fs = dev<uint8_t>((size_t)64 * W / 16); float* fg = dev<float>(64);
  h5::PrepArgs f{}; f.x = hcap; f.ldx = W; f.n = W; f.flags = h5::P_QUANT; f.rowmap = rm; f.q4 = fq; f.s4 = fs; f.g4 = fg;
  h5::launch_prep(f, ROWS, st);
  std::vector<uint8_t*> q(NC), sc(NC);
  for (int j = 0; j < NC; j++) {
    q[j] = dev<uint8_t>((size_t)64 * h5::D / 2); sc[j] = dev<uint8_t>((size_t)64 * h5::D / 16); float* g = dev<float>(64);
    h5::PrepArgs a{}; a.x = hcap + (size_t)j * h5::D; a.ldx = W; a.n = h5::D; a.flags = h5::P_QUANT; a.rowmap = rm; a.q4 = q[j]; a.s4 = sc[j]; a.g4 = g;
    h5::launch_prep(a, ROWS, st);
  }
  ck(cudaStreamSynchronize(st));
  auto F = get(fq, (size_t)64 * W / 2); auto FS = get(fs, (size_t)64 * W / 16);
  for (int j = 0; j < NC; j++) {
    auto Q = get(q[j], (size_t)64 * h5::D / 2); auto S = get(sc[j], (size_t)64 * h5::D / 16);
    for (int kb = 0; kb < h5::D / 128; kb++) for (int r = 0; r < ROWS; r++) {
      const size_t fb = ((size_t)(j * (h5::D / 128) + kb) * 64 + r), sb = ((size_t)kb * 64 + r);
      if (memcmp(&F[fb * 64], &Q[sb * 64], 64) || memcmp(&FS[fb * 8], &S[sb * 8], 8)) throw std::runtime_error("fused ingest prep block differs");
    }
  }
  // five-layer attention prep
  const size_t ring = (size_t)h5::DF_LAYERS * h5::DF_NKV * h5::DF_RING * h5::DF_HD;
  std::vector<float> hk((size_t)ROWS * h5::DF_LAYERS * 2 * h5::DF_KV); for (auto& x : hk) x = nd(rng);
  float* kv = dev<float>(hk.size()); ck(cudaMemcpy(kv, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice));
  std::vector<float> hn((size_t)h5::DF_LAYERS * h5::DF_HD); for (auto& x : hn) x = 1.f + 0.1f * nd(rng);
  float* kn = dev<float>(hn.size()); ck(cudaMemcpy(kn, hn.data(), hn.size() * 4, cudaMemcpyHostToDevice));
  std::vector<int> pos = {1118, 1119, 1120}, slot = {0, 0, 0};
  int* dpos = dev<int>(ROWS); int* dslot = dev<int>(ROWS);
  ck(cudaMemcpy(dpos, pos.data(), ROWS * 4, cudaMemcpyHostToDevice)); ck(cudaMemcpy(dslot, slot.data(), ROWS * 4, cudaMemcpyHostToDevice));
  __half *kc[2], *vc[2];
  for (int k = 0; k < 2; k++) { kc[k] = dev<__half>(ring); vc[k] = dev<__half>(ring); }
  const int ld = h5::DF_LAYERS * 2 * h5::DF_KV;
  for (int l = 0; l < h5::DF_LAYERS; l++) {
    h5::DAttnPreArgs a{kv, ld, l * 2 * h5::DF_KV, 0, nullptr, kn + l * h5::DF_HD, dslot, dpos, nullptr, kc[0], vc[0], l};
    h5::launch_dattn_pre(a, ROWS, st);
  }
  h5::DAttnPreArgs a{kv, ld, 0, 0, nullptr, nullptr, dslot, dpos, nullptr, kc[1], vc[1], 0};
  a.layers = h5::DF_LAYERS; for (int l = 0; l < h5::DF_LAYERS; l++) a.kn[l] = kn + l * h5::DF_HD;
  h5::launch_dattn_pre(a, ROWS, st);
  ck(cudaStreamSynchronize(st));
  same(get(reinterpret_cast<uint16_t*>(kc[0]), ring), get(reinterpret_cast<uint16_t*>(kc[1]), ring), "fused ingest K ring");
  same(get(reinterpret_cast<uint16_t*>(vc[0]), ring), get(reinterpret_cast<uint16_t*>(vc[1]), ring), "fused ingest V ring");
}

static void silu_il_case(int rows, std::mt19937& rng, cudaStream_t st) {
  std::normal_distribution<float> nd(0.f, 2.f);
  const int FF = h5::FF, GU = 2 * h5::FF;
  std::vector<float> il((size_t)rows * GU), de((size_t)rows * GU), sg(FF);
  for (auto& x : il) x = nd(rng);
  for (auto& x : sg) x = (rng() & 1) ? 1.f : -1.f;
  for (int r = 0; r < rows; r++) for (int e = 0; e < FF; e++) {
    const size_t g = (size_t)r * GU + 64 * (e / 32) + e % 32;
    de[(size_t)r * GU + e] = il[g]; de[(size_t)r * GU + FF + e] = il[g + 32];
  }
  float* dil = dev<float>(il.size()); float* dde = dev<float>(de.size()); float* dsg = dev<float>(FF);
  ck(cudaMemcpy(dil, il.data(), il.size() * 4, cudaMemcpyHostToDevice)); ck(cudaMemcpy(dde, de.data(), de.size() * 4, cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dsg, sg.data(), FF * 4, cudaMemcpyHostToDevice));
  struct Out { uint8_t *q4, *s4; float* g4; } o[2];
  for (auto& b : o) { b.q4 = dev<uint8_t>((size_t)64 * FF / 2); b.s4 = dev<uint8_t>((size_t)64 * FF / 16); b.g4 = dev<float>(64); }
  h5::PrepArgs a{}; a.x = dde; a.x2 = dde + FF; a.ldx = GU; a.n = FF; a.signs = dsg; a.flags = h5::P_SILU_MUL | h5::P_SIGN | h5::P_HADAMARD | h5::P_QUANT;
  a.q4 = o[0].q4; a.s4 = o[0].s4; a.g4 = o[0].g4; h5::launch_prep(a, rows, st);
  h5::PrepArgs b{}; b.x = dil; b.ldx = GU; b.n = FF; b.signs = dsg; b.flags = h5::P_SILU_IL | h5::P_SIGN | h5::P_HADAMARD | h5::P_QUANT;
  b.q4 = o[1].q4; b.s4 = o[1].s4; b.g4 = o[1].g4; h5::launch_prep(b, rows, st);
  ck(cudaStreamSynchronize(st));
  same(get(o[0].q4, (size_t)64 * FF / 2), get(o[1].q4, (size_t)64 * FF / 2), "silu-il q4");
  same(get(o[0].s4, (size_t)64 * FF / 16), get(o[1].s4, (size_t)64 * FF / 16), "silu-il s4");
  same(get(o[0].g4, 64), get(o[1].g4, 64), "silu-il g4");
  // consume-and-clear variant: same bytes, and the interleaved accumulator rows end all-zero
  h5::PrepArgs c = b; c.flags |= h5::P_CONSUME_CLEAR; c.q4 = o[0].q4; c.s4 = o[0].s4; c.g4 = o[0].g4;
  h5::launch_prep(c, rows, st); ck(cudaStreamSynchronize(st));
  same(get(o[0].q4, (size_t)64 * FF / 2), get(o[1].q4, (size_t)64 * FF / 2), "silu-il consume q4");
  same(get(o[0].s4, (size_t)64 * FF / 16), get(o[1].s4, (size_t)64 * FF / 16), "silu-il consume s4");
  for (auto w : get(reinterpret_cast<uint32_t*>(dil), il.size())) if (w) throw std::runtime_error("silu-il consume left accumulator nonzero");
  for (float* q : {dil, dde, dsg}) cudaFree(q);
  for (auto& c : o) { cudaFree(c.q4); cudaFree(c.s4); cudaFree(c.g4); }
}

int main() {
  try {
    // Fixed-scale chunk route, as in the selected preset; read once per process by launch_prep.
    setenv("H5_PREP_CHUNK", "2", 1); setenv("H5_PREP_G4", "0.015625", 1);
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    std::mt19937 rng(20261007);
    topk_case("topk-random-7", 7, false, rng, st);
    topk_case("topk-ties-7", 7, true, rng, st);
    topk_case("topk-random-1", 1, false, rng, st);
    topk_ownership_case();
    prep_case("chunk-quant-qkv", h5::P_QUANT, h5::D, h5::D, 8, h5::DF_Q + 2 * h5::DF_KV, rng, st);
    prep_case("chunk-quant-gu", h5::P_QUANT, h5::D, h5::D, 8, h5::GU, rng, st);
    prep_case("chunk-quant-o", h5::P_QUANT, h5::DF_Q, h5::DF_Q, 8, h5::D, rng, st);
    prep_case("chunk-silu-down", h5::P_SILU_MUL | h5::P_QUANT, h5::FF, h5::GU, 8, h5::D, rng, st);
    prep_case("chunk-quant-1row", h5::P_QUANT, h5::D, h5::D, 1, h5::GU, rng, st);
    dconv_case("dconv-attn-8", 8, 8, 0, rng, st);
    dconv_case("dconv-mlp-16-two-blocks", 16, 8, 0, rng, st);
    dconv_case("dconv-second-half", 8, 8, 1, rng, st);
    ingest_case(rng, st);
    silu_il_case(8, rng, st);
    silu_il_case(1, rng, st);
    printf("{\"drafter_exact\":\"pass\"}\n");
    return 0;
  } catch (const std::exception& e) { printf("{\"drafter_exact\":\"fail\",\"error\":\"%s\"}\n", e.what()); return 1; }
}
