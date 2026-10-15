// halo-5090 engine: weights, state, one batched step. C API at the bottom (used by python/h5.py).
#include "gguf.h"
#include "compact.hpp"
#include <cuda_fp8.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <chrono>
#include <map>
#include <algorithm>
#include <tuple>

namespace h5 {
void launch_gemm(const int8_t*, const float*, int, int, const uint4*, const float*, int, float*, int, bool, cudaStream_t);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
int fp4_ncfg();
bool fp4_cfg_wg2(int);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, cudaStream_t, const float*, int);
void launch_dconv(const float*, const float*, int, const float*, float*, int, int, int, cudaStream_t);
void launch_dattn_pre(const DAttnPreArgs&, int, cudaStream_t);
void launch_topk(const float*, float*, int*, int, cudaStream_t);
void launch_selector(const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const int*, const int*, int*, float*, int, int, cudaStream_t);
void launch_selector_two(const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const int*, const int*, int*, float*, int*, float*, int*, int, int, cudaStream_t);
const char* fp4_cfg_name(int, int);
void launch_prep(const PrepArgs&, int, cudaStream_t);
void launch_embed(const uint8_t*, const int*, float*, int, cudaStream_t);
void launch_visual_input(const int*, const float*, float*, int, cudaStream_t);
void launch_ab(const __nv_bfloat16*, const float*, float*, int, cudaStream_t);
void launch_conv(const ConvArgs&, int, StateFormat, cudaStream_t);
void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t);
void launch_attn_pre(const AttnArgs&, int, cudaStream_t);
void launch_attn_tc(const AttnTcArgs&, int, int, int, int, cudaStream_t);
void launch_attn_combine(const AttnTcArgs&, int, int, cudaStream_t);
void launch_embed(const uint8_t*, const int*, float*, int, cudaStream_t);
void launch_argmax(const float*, int, int*, int, cudaStream_t);
void launch_relaxed_argmax(const float*, int, int*, int, cudaStream_t, const int*, const Seg*, int, int, float);
void launch_mega(const MegaArgs&, int, cudaStream_t);
int mega_nsm();

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// GGUF PTQ1_0 block -> 128 trits in natural order (values -1,0,1) + scale
static inline void decode_ptq1(const uint8_t* blk, int8_t* trit, float* scale) {
  int e = 0;
  const int stages[2][2] = {{0, 16}, {16, 8}};
  for (int s = 0; s < 2; s++) {
    int j0 = stages[s][0], c = stages[s][1];
    for (int n = 0; n < 5; n++)
      for (int m = 0; m < c; m++) {
        unsigned q = blk[j0 + m];
        for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
        trit[e++] = (int8_t)((int)((q * 3u) >> 8) - 1);
      }
  }
  for (int n = 0; n < 4; n++)
    for (int h = 0; h < 2; h++) {
      unsigned q = blk[24 + h];
      for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
      trit[e++] = (int8_t)((int)((q * 3u) >> 8) - 1);
    }
  uint16_t hs; memcpy(&hs, blk + 26, 2);
  __half_raw r; r.x = hs; *scale = __half2float(__half(r));
}

// frag2 layout. Chunk (g, kb) = 32 features x 128 K = 1024 bytes; lane l owns bytes [32l, 32l+32) = 8 words.
// lane: gid = l>>2, t = l&3. reg index r = (s*4 + j)*2 + h, word = r>>2, field f = r&3 (bits 2f..2f+1 of each byte).
// byte b of reg r is weight (n = 32g + 8j + gid, k = 128kb + 32s + 16h + 4t + b), stored as (trit & 3).
struct TMat { uint4* w = nullptr; float* s = nullptr; uint4* w4 = nullptr; __half* s4 = nullptr; int N = 0, K = 0; float* rs = nullptr; int wf = 0; bool il = false; };

// ---- safetensors (bf16 drafter) ----
struct SafeT {
  void* base = nullptr; size_t size = 0; size_t data0 = 0; std::string hdr;
  void open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY); if (fd < 0) throw std::runtime_error("open " + path);
    struct stat sb; fstat(fd, &sb); size = sb.st_size;
    base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); ::close(fd);
    uint64_t n; memcpy(&n, base, 8); hdr.assign((const char*)base + 8, n); data0 = 8 + n;
  }
  // returns pointer and shape of tensor `name` (bf16 only)
  const uint16_t* t(const std::string& name, std::vector<int64_t>& shape) const {
    const std::string key = "\"" + name + "\"";
    size_t p = hdr.find(key); if (p == std::string::npos) throw std::runtime_error("safetensors: missing " + name);
    size_t e = hdr.find('}', p);
    std::string obj = hdr.substr(p, e - p);
    if (obj.find("BF16") == std::string::npos) throw std::runtime_error("safetensors: not bf16 " + name);
    shape.clear();
    size_t a = obj.find('[', obj.find("shape")), b = obj.find(']', a);
    std::string sh = obj.substr(a + 1, b - a - 1);
    for (size_t i = 0; i < sh.size();) { size_t j = sh.find(',', i); if (j == std::string::npos) j = sh.size(); shape.push_back(atoll(sh.substr(i, j - i).c_str())); i = j + 1; }
    a = obj.find('[', obj.find("data_offsets"));
    const size_t off = atoll(obj.c_str() + a + 1);
    return (const uint16_t*)((const uint8_t*)base + data0 + off);
  }
};
static inline float bf2f(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }
static inline float e4m3_val(uint8_t b) { __half_raw r = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3); return __half2float(__half(r)); }
static inline uint32_t e2m1_code(float v) {
  const float a = fabsf(v);
  uint32_t c = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3 : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.f ? 6 : 7;
  return c | (v < 0 && c ? 8u : 0u);
}
static const float kE2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
// bf16 [N][ld] columns [col0, col0+K) -> NVFP4 chunks (see fp4.cu WF 1). Row scale = amax / (6 * 448).
static void repack_nvfp4(const uint16_t* W, size_t ld, int N, int K, int col0, std::vector<uint8_t>& Wc, std::vector<uint8_t>& Sc,
                         std::vector<float>& rs, int nthreads) {
  const int NG = N / 32, nkb = K / 128;
  Wc.assign((size_t)NG * nkb * 2048, 0); Sc.assign((size_t)NG * nkb * 256, 0); rs.assign(N, 0.f);
  std::atomic<int> next{0};
  auto worker = [&]() {
    std::vector<uint8_t> nib(32 * 128);
    for (;;) {
      const int g = next.fetch_add(1); if (g >= NG) break;
      for (int r = 0; r < 32; r++) {
        const uint16_t* row = W + (size_t)(32 * g + r) * ld + col0;
        float m = 0.f; for (int k = 0; k < K; k++) m = std::max(m, fabsf(bf2f(row[k])));
        rs[32 * g + r] = std::max(m / (6.f * 448.f), 1e-30f);
      }
      for (int kb = 0; kb < nkb; kb++) {
        uint8_t* sc = Sc.data() + ((size_t)kb * NG + g) * 256;
        for (int r = 0; r < 32; r++) {
          const uint16_t* row = W + (size_t)(32 * g + r) * ld + col0 + kb * 128;
          const float rsc = rs[32 * g + r];
          for (int b = 0; b < 8; b++) {
            float bm = 0.f; for (int i = 0; i < 16; i++) bm = std::max(bm, fabsf(bf2f(row[b * 16 + i])));
            const float target = bm / 6.f / rsc;
            uint8_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
            if (e4m3_val(sb) < target && sb < 0x7e) sb++;
            sc[r * 8 + b] = sb;
            const float scale = e4m3_val(sb) * rsc;
            for (int i = 0; i < 16; i++) nib[r * 128 + b * 16 + i] = scale > 0.f ? e2m1_code(bf2f(row[b * 16 + i]) / scale) : 0;
          }
        }
        uint8_t* ch = Wc.data() + ((size_t)kb * NG + g) * 2048;
        for (int l = 0; l < 32; l++) {
          const int gid = l >> 2, tq = l & 3;
          for (int w = 0; w < 16; w++) {
            const int m = w >> 3, h = (w >> 2) & 1, j = w & 3;
            const int row = 16 * m + 8 * (j & 1) + gid, k0 = 64 * h + 32 * (j >> 1) + 8 * tq;
            uint32_t word = 0;
            for (int i = 0; i < 8; i++) word |= (uint32_t)nib[row * 128 + k0 + i] << (4 * i);
            memcpy(ch + (w >> 2) * 512 + l * 16 + (w & 3) * 4, &word, 4);
          }
        }
      }
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& x : th) x.join();
  (void)kE2M1;
}

// fp4 layout (see fp4.cu): chunk (g, kb) 1024 B, lane l bytes [32l, 32l+32) = words w = m*4 + h*2 + r.
static void repack_rows_fp4(const GTensor& t, int K, int Nt, int row0, std::vector<uint8_t>& W, std::vector<uint16_t>& S, int NGall, int nthreads, int gmul = 1, int goff = -1, bool compact = false) {
  const int nkb = K / 128;
  std::atomic<int> next{0};
  auto worker = [&]() {
    int8_t trit[32][128]; float sc[32];
    for (;;) {
      int gl = next.fetch_add(1);
      if (gl >= Nt / 32) break;
      int g = goff >= 0 ? goff + gl * gmul : (row0 / 32) + gl;
      for (int kb = 0; kb < nkb; kb++) {
        for (int r = 0; r < 32; r++) decode_ptq1(t.data + ((size_t)(gl * 32 + r) * nkb + kb) * 28, trit[r], &sc[r]);
        uint8_t* chunk = W.data() + ((size_t)kb * NGall + g) * (compact ? 832 : 1024);
        for (int l = 0; l < 32; l++) {
          const int gid = l >> 2, tq = l & 3;
          if (compact) {
            int8_t stream[128];
            for (int w = 0; w < 8; w++) {
              const int m = w >> 2, h = (w >> 1) & 1, r = w & 1, row = 16 * m + 8 * r + gid;
              for (int j = 0; j < 8; j++) {
                stream[w * 16 + j] = trit[row][64 * h + 8 * tq + j];
                stream[w * 16 + 8 + j] = trit[row][64 * h + 32 + 8 * tq + j];
              }
            }
            dense5::pack_lane(chunk, l, stream);
          } else {
          for (int w = 0; w < 8; w++) {
            const int m = w >> 2, h = (w >> 1) & 1, r = w & 1, row = 16 * m + 8 * r + gid;
            uint32_t word = 0;
            for (int j = 0; j < 8; j++) {
              const int t0 = trit[row][64 * h + 8 * tq + j], t1 = trit[row][64 * h + 32 + 8 * tq + j];
              word |= (uint32_t)((t0 != 0) << 1 | (t0 < 0) << 3) << (4 * j);
              word |= (uint32_t)((t1 != 0) << 0 | (t1 < 0) << 2) << (4 * j);
            }
            memcpy(chunk + (w >> 2) * 512 + l * 16 + (w & 3) * 4, &word, 4);
          }
          }
        }
        uint16_t* sh = S.data() + ((size_t)kb * NGall + g) * 32;
        for (int gid = 0; gid < 8; gid++)
          for (int q = 0; q < 4; q++) {
            const int m = q >> 1, r = q & 1, row = 16 * m + 8 * r + gid;
            __half hv = __float2half(sc[row]); uint16_t u; memcpy(&u, &hv, 2); sh[gid * 4 + q] = u;
          }
      }
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& x : th) x.join();
}

static void repack_rows(const GTensor& t, int K, int Nt, int row0, std::vector<uint8_t>& W, std::vector<float>& S, int N, int nthreads) {
  const int nkb = K / 128;
  std::atomic<int> next{0};
  auto worker = [&]() {
    int8_t trit[32][128]; float sc[32];
    for (;;) {
      int gl = next.fetch_add(1);
      if (gl >= Nt / 32) break;
      int g = (row0 / 32) + gl;
      for (int kb = 0; kb < nkb; kb++) {
        for (int r = 0; r < 32; r++) decode_ptq1(t.data + ((size_t)(gl * 32 + r) * nkb + kb) * 28, trit[r], &sc[r]);
        uint8_t* chunk = W.data() + ((size_t)g * nkb + kb) * 1024;
        memset(chunk, 0, 1024);
        for (int l = 0; l < 32; l++) {
          const int gid = l >> 2, tq = l & 3;
          for (int ri = 0; ri < 32; ri++) {
            const int h = ri & 1, j = (ri >> 1) & 3, s = ri >> 3, wi = ri >> 2, f = ri & 3;
            for (int b = 0; b < 4; b++) {
              const int n = 8 * j + gid, k = 32 * s + 16 * h + 4 * tq + b;
              chunk[l * 32 + wi * 4 + b] |= (uint8_t)((trit[n][k] & 3) << (2 * f));
            }
          }
        }
        for (int r = 0; r < 32; r++) S[(size_t)kb * N + g * 32 + r] = sc[r] / 64.f;
      }
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& x : th) x.join();
}

struct Layer {
  bool attn = false; int gl = -1, al = -1;
  float *attn_norm = nullptr, *post_norm = nullptr;
  TMat in, out, gu, down;
  __nv_bfloat16* ab = nullptr; TMat ab4;
  float *conv_w = nullptr, *ssm_a = nullptr, *ssm_dt = nullptr, *ssm_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
};

struct Engine {
  Gguf g;
  int max_slots, ctx, max_rows, nthreads;
  cudaStream_t st;
  float *signs5120, *signs6144, *signs17408, *output_norm; std::vector<float> hsigns5120;
  uint8_t* tok_embd;
  TMat head;
  std::vector<Layer> L;
  // state
  void* gdn_state; float* conv_state; void *kc, *vc; float *ksc = nullptr, *vsc = nullptr;
  StateFormat state_format = StateFormat::F16;
  bool kv8 = true; size_t state_es = 2;
  // activations
  float *x, *tmp, *xn, *proj, *conv_out, *ab, *y, *qrot, *logits;
  int8_t* xq; float* xs; uint8_t *q4, *s4; float* g4;
  int *d_tok, *d_slot, *d_pos, *d_out, *d_rope; float* d_visual; Seg* d_segs;
  int rope_sections[4] = {0, 0, 0, 0};
  int *h_tok, *h_slot, *h_pos, *h_out; Seg* h_segs;
  // speculation: per-slot pending rows of the last keep step, rows to commit at the next step
  void *pend_gdn = nullptr, *pend_conv = nullptr;
  std::vector<int> pend_n, commit_next, pend_par; std::vector<char> keep_slot;
  // drafter features: residual after CAP layers for every row of a step
  static constexpr int NCAP = 5; const int cap_layers[NCAP] = {5, 19, 33, 47, 61};
  float* hcap = nullptr; bool capture = false; int captured_rows = 0;
  QBlk *d_qb, *h_qb; float* part_o; float2* part_ml; int part_cap = 4096;
  size_t weight_bytes = 0;
  double last_step_s = 0;

  template <class T> T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }
  template <class T> T* upload(const void* src, size_t n) { T* p = dalloc<T>(n); CK(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice)); weight_bytes += n * sizeof(T); return p; }
  float* f32(const std::string& n) { auto& t = g.t(n); return upload<float>(t.data, t.bytes / 4); }

  bool use_int8 = false, use_fp4 = true; bool wscale_e4m3 = getenv("H5_WSCALE") && std::string(getenv("H5_WSCALE")) == "e4m3";   // weight images to build; act_fp4 picks the path at run time
  bool compact = getenv("H5_COMPACT") && std::string(getenv("H5_COMPACT")) == "dense5";
  bool act_fp4 = true, force_fp4 = false; bool ab_bf16 = getenv("H5_AB_BF16") != nullptr;
  // interleave: (fp4 image only) group g of tensor i goes to group g * names.size() + i, so gate and up
  // of the same 32 features sit in adjacent groups for the fused silu(gate)*up epilogue.
  TMat tmat(const std::vector<std::string>& names, int K, bool interleave = false) {
    int N = 0; for (auto& n : names) N += (int)g.t(n).ne[1];
    TMat m; m.N = N; m.K = K;
    for (auto& n : names) { auto& t = g.t(n); if ((int)t.ne[0] != K || t.type != 143) throw std::runtime_error("tmat " + n); }
    if (use_int8) {
      std::vector<uint8_t> W((size_t)N * K / 4); std::vector<float> S((size_t)N * (K / 128));
      int row0 = 0;
      for (auto& n : names) { auto& t = g.t(n); repack_rows(t, K, (int)t.ne[1], row0, W, S, N, nthreads); row0 += (int)t.ne[1]; }
      m.w = upload<uint4>(W.data(), W.size() / 16); m.s = upload<float>(S.data(), S.size());
    }
    if (use_fp4) {
      std::vector<uint8_t> W((size_t)(N / 32) * (K / 128) * (compact ? 832 : 1024)); std::vector<uint16_t> S((size_t)N * (K / 128));
      int row0 = 0;
      int idx = 0;
      for (auto& n : names) {
        auto& t = g.t(n);
        if (interleave) repack_rows_fp4(t, K, (int)t.ne[1], row0, W, S, N / 32, nthreads, (int)names.size(), idx, compact);
        else repack_rows_fp4(t, K, (int)t.ne[1], row0, W, S, N / 32, nthreads, 1, -1, compact);
        row0 += (int)t.ne[1]; idx++;
      }
      m.il = interleave;
      if (wscale_e4m3) {   // S chunk [kb][g][gid][4] halves -> bytes [kb][g][gid][4] (64 B stride kept) + row scale
        const int nkb = K / 128, NG = N / 32;
        std::vector<float> rsv(N, 0.f);
        for (size_t c = 0; c < (size_t)nkb * NG; c++) {
          const int g = c % NG;
          for (int i = 0; i < 32; i++) { __half_raw hr; hr.x = S[c * 32 + i]; const int gid = i / 4, q = i % 4, row = 32 * g + 16 * (q >> 1) + 8 * (q & 1) + gid; rsv[row] = std::max(rsv[row], __half2float(__half(hr))); }
        }
        for (auto& v : rsv) v = std::max(v / 448.f, 1e-30f);
        std::vector<uint8_t> SB(S.size() * 2, 0);
        for (size_t c = 0; c < (size_t)nkb * NG; c++) {
          const int g = c % NG;
          for (int i = 0; i < 32; i++) { __half_raw hr; hr.x = S[c * 32 + i]; const int gid = i / 4, q = i % 4, row = 32 * g + 16 * (q >> 1) + 8 * (q & 1) + gid;
            SB[c * 64 + gid * 4 + q] = __nv_cvt_float_to_fp8(__half2float(__half(hr)) / rsv[row], __NV_SATFINITE, __NV_E4M3); }
        }
        m.w4 = upload<uint4>(W.data(), W.size() / 16); m.s4 = (__half*)upload<uint8_t>(SB.data(), SB.size()); m.rs = upload<float>(rsv.data(), rsv.size()); m.wf = 2;
      } else { m.w4 = upload<uint4>(W.data(), W.size() / 16); m.s4 = upload<__half>(S.data(), S.size()); m.wf = compact ? 3 : 0; }
    }
    return m;
  }

  int df_proposals = DRAFT_BASE_PROPOSALS, pending_rows = PMAX;
  Engine(const char* path, int slots, int ctx_, int rows, int proposals = DRAFT_BASE_PROPOSALS) : max_slots(slots), ctx(ctx_), max_rows(rows), df_proposals(proposals), pending_rows(proposals + 1) {
    if (proposals != 7 && proposals != 16 && proposals != 32 && proposals != 64) throw std::runtime_error("unsupported draft proposal count");
    if (proposals != 7 && slots != 1) throw std::runtime_error("long draft requires one slot");
    if (proposals != 7 && (rows < proposals + 1 || ctx_ <= proposals + 2)) throw std::runtime_error("draft capacity too small");
    if (df_window <= 0 || (long long)df_window + proposals + 1 + 8 > DF_RING) throw std::runtime_error("H5_DF_WINDOW exceeds ring capacity");
    if (const char* margin = getenv("H5_DRAFT_MARGIN")) if (!parse_draft_margin(margin, draft_margin)) throw std::runtime_error("H5_DRAFT_MARGIN must be a finite nonnegative number");
    nthreads = getenv("H5_THREADS") ? std::max(1, std::min(4, atoi(getenv("H5_THREADS")))) : 4;
    if (const char* f = getenv("H5_LOAD_THREADS")) nthreads = std::max(1, std::min(nthreads, atoi(f)));
    if (compact) {
      if (wscale_e4m3) throw std::runtime_error("H5_COMPACT=dense5 requires original FP16 W128 scales; unset H5_WSCALE=e4m3");
      if (getenv("H5_MEGA") && atoi(getenv("H5_MEGA"))) throw std::runtime_error("H5_COMPACT=dense5 is not supported by the megakernel");
    }
    if (const char* f = getenv("H5_WEIGHTS")) { std::string v = f; use_int8 = v == "int8" || v == "both"; use_fp4 = v == "fp4" || v == "both"; }
    act_fp4 = use_fp4;
    if (const char* f = getenv("H5_ACT")) act_fp4 = std::string(f) == "fp4";
    if (act_fp4 && !use_fp4) throw std::runtime_error("H5_ACT=fp4 needs fp4 weights");
    if (!act_fp4 && !use_int8) throw std::runtime_error("int8 activations need int8 weights");
    double t0 = now_s();
    g.open(path);
    auto rsec = g.kv.find("qwen35.rope.dimension_sections");
    if (rsec != g.kv.end() && rsec->second.ints.size() == 4)
      for (int i = 0; i < 4; i++) rope_sections[i] = rsec->second.ints[i];
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    {
      auto& w = g.kv.at("prism.hadamard.sign_widths").ints; auto& v = g.kv.at("prism.hadamard.sign_values").ints;
      size_t off = 0;
      for (int64_t wd : w) {
        std::vector<float> s(wd); for (int64_t i = 0; i < wd; i++) s[i] = (float)v[off + i]; off += wd;
        float* d = upload<float>(s.data(), wd);
        if (wd == 5120) { signs5120 = d; hsigns5120 = s; } else if (wd == 6144) signs6144 = d; else if (wd == 17408) signs17408 = d;
      }
    }
    { auto& t = g.t("token_embd.weight"); tok_embd = upload<uint8_t>(t.data, t.bytes); }
    output_norm = f32("output_norm.weight");
    head = tmat({"output.weight"}, D);
    L.resize(NLAYER);
    int gl = 0, al = 0;
    for (int l = 0; l < NLAYER; l++) {
      Layer& y = L[l]; std::string p = "blk." + std::to_string(l) + ".";
      y.attn = is_attn(l);
      y.attn_norm = f32(p + "attn_norm.weight"); y.post_norm = f32(p + "post_attention_norm.weight");
      if (y.attn) {
        y.al = al++;
        y.in = tmat({p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, D);
        y.out = tmat({p + "attn_output.weight"}, ATTN_OUT);
        y.q_norm = f32(p + "attn_q_norm.weight"); y.k_norm = f32(p + "attn_k_norm.weight");
      } else {
        y.gl = gl++;
        y.in = tmat({p + "attn_qkv.weight", p + "attn_gate.weight"}, D);
        y.out = tmat({p + "ssm_out.weight"}, VDIM);
        y.conv_w = f32(p + "ssm_conv1d.weight"); y.ssm_a = f32(p + "ssm_a"); y.ssm_dt = f32(p + "ssm_dt.bias"); y.ssm_norm = f32(p + "ssm_norm.weight");
        std::vector<uint16_t> ab(96 * D);
        memcpy(ab.data(), g.t(p + "ssm_alpha.weight").data, 48 * D * 2);
        memcpy(ab.data() + 48 * D, g.t(p + "ssm_beta.weight").data, 48 * D * 2);
        y.ab = upload<__nv_bfloat16>(ab.data(), ab.size());
        {   // rotated NVFP4 copy: W' = W S H, so it consumes the in-projection's activations
          std::vector<uint16_t> rot(ab.size());
          std::vector<float> v(1024);
          for (int r = 0; r < 96; r++)
            for (int b0 = 0; b0 < D; b0 += 1024) {
              for (int i = 0; i < 1024; i++) v[i] = bf2f(ab[(size_t)r * D + b0 + i]) * hsigns5120[b0 + i];
              for (int len = 1; len < 1024; len <<= 1)
                for (int i = 0; i < 1024; i++) if (!(i & len)) { float p0 = v[i], p1 = v[i + len]; v[i] = p0 + p1; v[i + len] = p0 - p1; }
              for (int i = 0; i < 1024; i++) { float f = v[i] * 0.03125f; uint32_t u; memcpy(&u, &f, 4); rot[(size_t)r * D + b0 + i] = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
            }
          std::vector<uint8_t> W, S; std::vector<float> RS;
          repack_nvfp4(rot.data(), D, 96, D, 0, W, S, RS, nthreads);
          y.ab4.N = 96; y.ab4.K = D; y.ab4.wf = 1;
          y.ab4.w4 = upload<uint4>(W.data(), W.size() / 16); y.ab4.s4 = (__half*)upload<uint8_t>(S.data(), S.size()); y.ab4.rs = upload<float>(RS.data(), RS.size());
        }
      }
      y.gu = tmat({p + "ffn_gate.weight", p + "ffn_up.weight"}, D, !getenv("H5_NO_GUFUSE"));
      y.down = tmat({p + "ffn_down.weight"}, FF);
    }
    fprintf(stderr, "\nweights %.2f GB in %.1f s\n", weight_bytes / 1e9, now_s() - t0);
    // state
    if (const char* f = getenv("H5_STATE")) {
      if (std::string(f) == "bf16") state_format = StateFormat::BF16;
      else if (std::string(f) == "f16" || std::string(f) == "fp16") state_format = StateFormat::F16;
      else if (std::string(f) == "f32") state_format = StateFormat::F32;
      else throw std::runtime_error("H5_STATE must be f16, bf16 or f32");
    }
    if (const char* f = getenv("H5_KV")) kv8 = std::string(f) == "fp8";
    ctx = (ctx + 31) / 32 * 32;   // fp8 V is stored in 32-key transposed blocks
    state_es = state_format == StateFormat::F32 ? 4 : 2;
    gdn_state = dalloc<uint8_t>((size_t)max_slots * NGDN * HV * 128 * 128 * state_es);
    conv_state = dalloc<float>((size_t)max_slots * NGDN * 3 * CONV_CH);
    const size_t kvn = (size_t)max_slots * NATTN * NKV * ctx;
    kc = dalloc<uint8_t>(kvn * HD * (kv8 ? 1 : 2)); vc = dalloc<uint8_t>(kvn * HD * (kv8 ? 1 : 2));
    if (kv8) { ksc = dalloc<float>(kvn); vsc = dalloc<float>(kvn); }
    fprintf(stderr, "state %s, KV %s: %.1f MB state + %.1f MB KV per slot\n", state_format == StateFormat::BF16 ? "bf16" : state_format == StateFormat::F16 ? "f16" : "f32", kv8 ? "fp8" : "f16",
            NGDN * HV * 128.0 * 128 * state_es / 1e6, (double)NATTN * NKV * ctx * (HD * (kv8 ? 2 : 4) + (kv8 ? 8 : 0)) / 1e6);
    // activations
    const size_t R = max_rows;
    x = dalloc<float>(R * D); tmp = dalloc<float>(R * D); xn = dalloc<float>(R * D);
    proj = dalloc<float>(R * GU); conv_out = dalloc<float>(R * CONV_CH); ab = dalloc<float>(R * 96);
    y = dalloc<float>(R * VDIM); qrot = dalloc<float>(R * NH * HD); logits = dalloc<float>((size_t)128 * VOCAB);
    xq = dalloc<int8_t>(R * FF); xs = dalloc<float>(R * (FF / 128));
    gates = dalloc<float>(R * 96);
    scratch = dalloc<float>(std::max(R * (size_t)GU, (size_t)128 * VOCAB));
    const size_t Rp = (R + 127) / 128 * 128;
    q4 = dalloc<uint8_t>(Rp * FF / 2); s4 = dalloc<uint8_t>(Rp * FF / 16); g4 = dalloc<float>(Rp);
    d_tok = dalloc<int>(R); d_slot = dalloc<int>(R); d_pos = dalloc<int>(R); d_out = dalloc<int>(R); d_segs = dalloc<Seg>(R);
    d_rope = dalloc<int>(4 * R); d_visual = dalloc<float>((size_t)R * D);
    pend_gdn = dalloc<uint8_t>((size_t)max_slots * NGDN * 2 * pending_rows * PEND_GDN * state_es);
    pend_par.assign(max_slots, 0);
    pend_conv = dalloc<uint8_t>((size_t)max_slots * NGDN * pending_rows * CONV_CH * state_es);
    pend_n.assign(max_slots, 0); commit_next.assign(max_slots, 0); keep_slot.assign(max_slots, 0);
    hcap = dalloc<float>(R * NCAP * D);
    d_qb = dalloc<QBlk>(R); CK(cudaMallocHost(&h_qb, R * sizeof(QBlk)));
    part_o = dalloc<float>((size_t)part_cap * NH * HD); part_ml = dalloc<float2>((size_t)part_cap * std::max(NH, DF_NH));   // drafter has 32 heads
    CK(cudaMallocHost(&h_tok, R * 4)); CK(cudaMallocHost(&h_slot, R * 4)); CK(cudaMallocHost(&h_pos, R * 4)); CK(cudaMallocHost(&h_out, R * 4)); CK(cudaMallocHost(&h_segs, R * sizeof(Seg)));
    { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d); }
    mega_init();
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "ready: %d slots x %d ctx, %d rows; GPU free %.2f / %.2f GB, load %.1f s\n", max_slots, ctx, max_rows, fr / 1e9, tot / 1e9, now_s() - t0);
  }

  // ---- megakernel (single-slot steps of <= 8 rows) ----
  bool use_mega = false; int mega_nb = 0; int nsm = 170;
  MLayer* d_ml = nullptr; uint8_t *ma4 = nullptr, *ms4 = nullptr; float* mgs = nullptr; float* mproj2 = nullptr;
  unsigned long long* mamax = nullptr; unsigned* mbar = nullptr; int* d_mout = nullptr;
  unsigned long long* mprof = nullptr; std::vector<unsigned long long> hprof; double mph[32] = {}; long mph_n[32] = {}; long mprof_steps = 0;
  long mega_steps = 0; int mega_pf = getenv("H5_MEGA_PF") ? atoi(getenv("H5_MEGA_PF")) : 1;
  void mega_init() {
    use_mega = use_fp4 && !wscale_e4m3 && state_format == StateFormat::BF16 && kv8 && getenv("H5_MEGA") && atoi(getenv("H5_MEGA"));
    for (auto& y : L) if (!y.gu.il) use_mega = false;
    if (!use_mega) return;
    std::vector<MLayer> h(NLAYER);
    for (int l = 0; l < NLAYER; l++) {
      Layer& y = L[l]; MLayer& m = h[l];
      m.attn = y.attn; m.gl = y.gl; m.al = y.al; m.n_in = y.in.N;
      m.attn_norm = y.attn_norm; m.post_norm = y.post_norm;
      m.w_in = (const uint8_t*)y.in.w4; m.w_out = (const uint8_t*)y.out.w4; m.w_gu = (const uint8_t*)y.gu.w4; m.w_down = (const uint8_t*)y.down.w4;
      m.s_in = (const uint16_t*)y.in.s4; m.s_out = (const uint16_t*)y.out.s4; m.s_gu = (const uint16_t*)y.gu.s4; m.s_down = (const uint16_t*)y.down.s4;
      m.ab = y.ab; m.conv_w = y.conv_w; m.ssm_a = y.ssm_a; m.ssm_dt = y.ssm_dt; m.ssm_norm = y.ssm_norm; m.q_norm = y.q_norm; m.k_norm = y.k_norm;
    }
    d_ml = upload<MLayer>(h.data(), NLAYER);
    ma4 = dalloc<uint8_t>((size_t)(FF / 128) * 8 * 64); ms4 = dalloc<uint8_t>((size_t)(FF / 128) * 8 * 8); mgs = dalloc<float>(8 * 17); mproj2 = dalloc<float>((size_t)8 * GU);
    mamax = dalloc<unsigned long long>(8); mbar = dalloc<unsigned>(2); d_mout = dalloc<int>(8);
    mega_nb = mega_nsm();
    if (getenv("H5_MEGA_PROF")) { mprof = dalloc<unsigned long long>(2048); hprof.resize(2048); }
    fprintf(stderr, "megakernel: %d CTAs x 256 threads for single-slot steps of <= 8 rows\n", mega_nb);
  }
  void run_mega(int R, int nsp, int tps, int max_mt) {
    MegaArgs a{};
    a.L = d_ml; a.s5120 = signs5120; a.s6144 = signs6144; a.s17408 = signs17408; a.out_norm = output_norm;
    a.emb = tok_embd; a.w_head = (const uint8_t*)head.w4; a.s_head = (const uint16_t*)head.s4;
    a.R = R; a.ctx = ctx; a.nsp = nsp; a.tps = tps; a.mt = max_mt; a.capture = capture; a.pf = mega_pf;
    a.pending_rows = pending_rows;
    a.seg = h_segs[0]; a.qbp = d_qb; a.tok = d_tok; a.pos = d_pos; a.slotv = d_slot;
    a.x = x; a.xn = xn; a.proj = proj; a.proj2 = mproj2; a.conv_out = conv_out; a.abv = ab; a.gates = gates; a.y = y; a.qrot = qrot;
    a.part_o = part_o; a.part_ml = part_ml; a.hcap = hcap;
    a.a4 = ma4; a.s4 = ms4; a.gs = mgs; a.amax = mamax; a.out_tok = d_mout;
    a.gdn_state = gdn_state; a.pend_gdn = pend_gdn; a.pend_conv = pend_conv; a.conv_state = conv_state;
    a.kc = kc; a.vc = vc; a.ksc = ksc; a.vsc = vsc; a.bar = mbar; a.prof = mprof;
    unsigned long long t0 = 0;
    if (mprof) { CK(cudaStreamSynchronize(st)); }
    launch_mega(a, mega_nb, st);
    mega_steps++;
    if (mprof) {
      CK(cudaMemcpyAsync(hprof.data(), mprof, 2048 * 8, cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
      const int n = std::min<long>(hprof[0], 2047);
      (void)t0;
      for (int i = 1; i < n; i++) { const int code = hprof[1 + i] & 0xff; mph[code] += ((hprof[1 + i] >> 8) - (hprof[i] >> 8)) * 1e-6; mph_n[code]++; }
      mprof_steps++;
    }
  }

  void prep(const float* in, const float* in2, int ldx, const float* nw, const float* signs, int n, unsigned flags, int R, float* out_f32 = nullptr, float* out_norm = nullptr, int ldo = 0, const int* rowmap = nullptr) {
    PrepArgs a{}; a.x = in; a.x2 = in2; a.ldx = ldx; a.norm_w = nw; a.signs = signs; a.n = n; a.flags = flags; a.rowmap = rowmap;
    a.xq = xq; a.xs = xs; a.out_f32 = out_f32; a.out_norm = out_norm; a.ldo = ldo;
    if (act_fp4 || force_fp4) { a.q4 = q4; a.s4 = s4; a.g4 = g4; }
    launch_prep(a, R, st);
  }
  // Autotuned FP4 launch: first use of (N, K, R, add) times every config x split-K on a scratch output.
  std::map<std::tuple<int, int, int, bool, int>, std::pair<int, int>> tuned;
  float* scratch = nullptr; float* gates = nullptr;
  bool tune_verbose = getenv("H5_TUNE_VERBOSE") != nullptr;
  // Tuning buckets: powers of two from 8 rows, so a serving run meets a handful of keys and pretune()
  // can cover all of them at startup.
  static int rbucket(int R) { int b = 8; while (b < R && b < 4096) b <<= 1; return b; }
  std::pair<int, int> tune(const TMat& m, int R, const uint8_t* A, const uint8_t* AS, const float* AG, bool add) {
    auto key = std::make_tuple(m.N, m.K, rbucket(R), add, m.wf);
    auto it = tuned.find(key);
    if (it != tuned.end()) return it->second;
    if (getenv("H5_PLAN_READ")) throw std::runtime_error("frozen launch plan missing GEMM shape");
    if (const char* f = getenv("H5_FORCE_CFG")) { int ci = 0, ks = 1; sscanf(f, "%d,%d", &ci, &ks); if (m.K / 128 / ks < 1) ks = 1; tuned[key] = {ci, ks}; return tuned[key]; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e9; std::pair<int, int> bc{0, 1};
    const int nkb = m.K / 128;
    for (int ci = 0; ci < fp4_ncfg(); ci++)
      for (int ks : {1, 2, 3, 4, 6, 8, 12}) {
        if (nkb / ks < 4) continue;
        float tot = 0;
        for (int rep = 0; rep < 4; rep++) {
          cudaEventRecord(e0, st);
          launch_gemm_fp4_cfg(ci, ks, A, AS, AG, std::min(rbucket(R), max_rows), m.K, m.w4, m.s4, m.N, scratch, m.N, add, st, m.rs, m.wf);
          cudaEventRecord(e1, st); cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms, e0, e1); if (rep) tot += ms;
        }
        if (tot < best) { best = tot; bc = {ci, ks}; }
      }
    CK(cudaGetLastError());
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (tune_verbose) fprintf(stderr, "tune N=%d K=%d R=%d add=%d fmt=%d: %s %.1f us\n", m.N, m.K, R, (int)add, m.rs != nullptr, fp4_cfg_name(bc.first, bc.second), best / 3 * 1e3); (void)0;
    tuned[key] = bc;
    return bc;
  }
  // Fused gate/up (interleaved image): out[R][FF] = silu(gate) * up. Tuned over the WG=2 configs, no split-K.
  void gemm_silu(const TMat& m, int R, float* out, int ldo) {
    const uint8_t* A = q4; const uint8_t* AS = s4; const float* AG = g4;
    auto key = std::make_tuple(m.N, m.K, rbucket(R), false, 100);
    auto it = tuned.find(key);
    int ci;
    if (it != tuned.end()) ci = it->second.first;
    else if (getenv("H5_PLAN_READ")) throw std::runtime_error("frozen launch plan missing fused GEMM shape");
    else if (getenv("H5_FORCE_CFG") && fp4_cfg_wg2(atoi(getenv("H5_FORCE_CFG")))) { ci = atoi(getenv("H5_FORCE_CFG")); tuned[key] = {ci, 1}; }
    else {
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      float best = 1e9; ci = -1;
      for (int c = 0; c < fp4_ncfg(); c++) {
        if (!fp4_cfg_wg2(c)) continue;
        float tot = 0;
        for (int rep = 0; rep < 4; rep++) {
          cudaEventRecord(e0, st);
          launch_gemm_fp4_silu(c, A, AS, AG, std::min(rbucket(R), max_rows), m.K, m.w4, m.s4, m.N, scratch, ldo, st, m.rs, m.wf);
          cudaEventRecord(e1, st); cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms, e0, e1); if (rep) tot += ms;
        }
        if (tot < best) { best = tot; ci = c; }
      }
      CK(cudaGetLastError());
      cudaEventDestroy(e0); cudaEventDestroy(e1);
      if (tune_verbose) fprintf(stderr, "tune silu N=%d K=%d R=%d: %s %.1f us\n", m.N, m.K, R, fp4_cfg_name(ci, 1), best / 3 * 1e3);
      tuned[key] = {ci, 1};
    }
    launch_gemm_fp4_silu(ci, A, AS, AG, R, m.K, m.w4, m.s4, m.N, out, ldo, st, m.rs, m.wf);
  }
  // Tune every projection shape at every row bucket up to max_rows (server startup).
  void pretune() {
    if (const char* path = getenv("H5_PLAN_READ")) {
      FILE* f=fopen(path,"r"); if(!f) throw std::runtime_error("cannot read frozen launch plan");
      int n,k,r,a,w,c,s,fields;
      while ((fields=fscanf(f,"%d %d %d %d %d %d %d",&n,&k,&r,&a,&w,&c,&s))==7) {
        if(n<=0 || k<=0 || r<8 || (a!=0 && a!=1) || c<0 || c>=fp4_ncfg() || s<1 || k/128/s<1 || (w==100 && (!fp4_cfg_wg2(c) || s!=1)) || !tuned.emplace(std::make_tuple(n,k,r,a!=0,w),std::make_pair(c,s)).second) {
          fclose(f); throw std::runtime_error("invalid or duplicate frozen launch plan row");
        }
      }
      fclose(f); if(fields!=EOF || tuned.empty()) throw std::runtime_error("truncated or empty frozen launch plan");
    }
    const double t0 = now_s();
    std::vector<std::pair<const TMat*, bool>> mats;
    for (auto& y : L) { mats.push_back({&y.in, false}); mats.push_back({&y.out, true}); mats.push_back({&y.gu, false}); mats.push_back({&y.down, true}); if (!y.attn) mats.push_back({&y.ab4, false}); }
    mats.push_back({&head, false});
    if (df.loaded) {
      for (int j = 0; j < NCAP; j++) mats.push_back({&df.fc[j], j > 0});
      mats.push_back({&df.hproj, false});
      for (auto& y : df.L) for (const TMat* m : {&y.qkv, &y.kv, &y.o, &y.gu, &y.down, &y.akp, &y.mkp}) mats.push_back({m, false});
    }
    int n = 0;
    for (int b = 8; b <= max_rows; b <<= 1)
      for (auto& [m, add] : mats) {
        if (!m->w4) continue;
        const bool hd = m == &head;
        if (hd && b > 128) continue;
        if (m->il) { gemm_silu_tune_only(*m, b); n++; continue; }
        tune(*m, b, q4, s4, g4, add); n++;
      }
    CK(cudaStreamSynchronize(st));
    fprintf(stderr, "pretune: %zu shapes x buckets <= %d rows in %.1f s\n", tuned.size(), max_rows, now_s() - t0);
    if (const char* path=getenv("H5_PLAN_WRITE")) {
      FILE*f=fopen(path,"w"); if(!f) throw std::runtime_error("cannot write frozen launch plan");
      for(auto&[key,value]:tuned) { auto[n,k,r,a,w]=key; if(fprintf(f,"%d %d %d %d %d %d %d\n",n,k,r,(int)a,w,value.first,value.second)<0) { fclose(f); throw std::runtime_error("launch plan write failed"); } }
      if(fclose(f)!=0) throw std::runtime_error("launch plan close failed");
    }
  }
  void gemm_silu_tune_only(const TMat& m, int R) {
    // the tuning branch of gemm_silu, launching into scratch only
    auto key = std::make_tuple(m.N, m.K, rbucket(R), false, 100);
    if (tuned.count(key)) return;
    gemm_silu(m, R, scratch, FF);
  }
  void gemm(const TMat& m, int R, float* out, int ldo, bool add, int r0 = 0) {
    if (act_fp4 || force_fp4) {
      if (!m.w4 || !m.s4) throw std::runtime_error("FP4 GEMM requested without FP4 weights (N=" + std::to_string(m.N) + ", K=" + std::to_string(m.K) + ")");
      const uint8_t* A = q4 + (size_t)r0 * m.K / 2; const uint8_t* AS = s4 + (size_t)r0 * m.K / 16; const float* AG = g4 + r0;
      auto c = tune(m, R, A, AS, AG, add);
      launch_gemm_fp4_cfg(c.first, c.second, A, AS, AG, R, m.K, m.w4, m.s4, m.N, out, ldo, add, st, m.rs, m.wf);
    }
    else {
      if (!m.w || !m.s) throw std::runtime_error("INT8 GEMM requested without INT8 weights (N=" + std::to_string(m.N) + ", K=" + std::to_string(m.K) + ")");
      launch_gemm(xq + (size_t)r0 * m.K, xs + (size_t)r0 * (m.K / 128), R, m.K, m.w, m.s, m.N, out, ldo, add, st);
    }
  }

  // time every projection of the model at R rows (inputs are whatever the buffers hold)
  void gemm_bench(int R, int reps) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    double macs = 0; size_t wb = 0;
    auto all = [&]() {
      for (auto& Ly : L) {
        gemm(Ly.in, R, proj, Ly.attn ? ATT_IN : GDN_IN, false);
        gemm(Ly.out, R, x, D, true);
        if (Ly.gu.il && act_fp4) gemm_silu(Ly.gu, R, proj, FF); else gemm(Ly.gu, R, proj, GU, false);
        gemm(Ly.down, R, x, D, true);
      }
      gemm(head, std::min(R, 128), logits, VOCAB, false);
    };
    for (auto& Ly : L) for (const TMat* m : {&Ly.in, &Ly.out, &Ly.gu, &Ly.down}) { macs += (double)m->N * m->K * R; wb += (size_t)m->N * m->K / 4; }
    macs += (double)head.N * head.K * std::min(R, 128); wb += (size_t)head.N * head.K / 4;
    all(); CK(cudaStreamSynchronize(st));
    float best = 1e9;
    for (int i = 0; i < reps; i++) { cudaEventRecord(e0, st); all(); cudaEventRecord(e1, st); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms); }
    printf("gemm R=%4d %s: %.3f ms  %.1f TMAC/s  weights %.0f GB/s  %.4f ms/row\n", R, act_fp4 ? "fp4" : "int8", best, macs / best / 1e9, wb / best / 1e6, best / R);
  }

  // Clone slot src into dst for the first npos positions (prefix cache): recurrent state, conv ring,
  // KV [0, npos) and the drafter's ring. Pending speculative rows are dropped.
  void copy_slot(int src, int dst, int npos, bool pending = false) {
    const size_t sb = (size_t)NGDN * HV * 128 * 128 * state_es;
    CK(cudaMemcpyAsync((uint8_t*)gdn_state + dst * sb, (uint8_t*)gdn_state + src * sb, sb, cudaMemcpyDeviceToDevice, st));
    const size_t cb = (size_t)NGDN * 3 * CONV_CH;
    CK(cudaMemcpyAsync(conv_state + dst * cb, conv_state + src * cb, cb * 4, cudaMemcpyDeviceToDevice, st));
    const size_t es = kv8 ? 1 : 2;
    const bool full_kv = pending && getenv("H5_CLONE_FULL_KV") && atoi(getenv("H5_CLONE_FULL_KV"));
    if (full_kv) {
      const size_t kn = (size_t)NATTN * NKV * ctx, kb = kn * HD * es;
      CK(cudaMemcpyAsync((uint8_t*)kc + dst * kb, (uint8_t*)kc + src * kb, kb, cudaMemcpyDeviceToDevice, st));
      CK(cudaMemcpyAsync((uint8_t*)vc + dst * kb, (uint8_t*)vc + src * kb, kb, cudaMemcpyDeviceToDevice, st));
      if (kv8) {
        CK(cudaMemcpyAsync(ksc + dst * kn, ksc + src * kn, kn * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(vsc + dst * kn, vsc + src * kn, kn * 4, cudaMemcpyDeviceToDevice, st));
      }
    } else for (int l = 0; l < NATTN; l++)
      for (int hk = 0; hk < NKV; hk++) {
        const size_t o_src = (((size_t)src * NATTN + l) * NKV + hk) * ctx, o_dst = (((size_t)dst * NATTN + l) * NKV + hk) * ctx;
        CK(cudaMemcpyAsync((uint8_t*)kc + o_dst * HD * es, (uint8_t*)kc + o_src * HD * es, (size_t)npos * HD * es, cudaMemcpyDeviceToDevice, st));
        const size_t vn = kv8 ? (size_t)(npos + 31) / 32 * 32 : (size_t)npos;
        CK(cudaMemcpyAsync((uint8_t*)vc + o_dst * HD * es, (uint8_t*)vc + o_src * HD * es, vn * HD * es, cudaMemcpyDeviceToDevice, st));
        if (kv8) {
          CK(cudaMemcpyAsync(ksc + o_dst, ksc + o_src, (size_t)npos * 4, cudaMemcpyDeviceToDevice, st));
          CK(cudaMemcpyAsync(vsc + o_dst, vsc + o_src, (size_t)npos * 4, cudaMemcpyDeviceToDevice, st));
        }
      }
    if (df.loaded) {
      const size_t rb = (size_t)DF_LAYERS * DF_NKV * DF_RING * DF_HD;
      CK(cudaMemcpyAsync(df.kc + dst * rb, df.kc + src * rb, rb * 2, cudaMemcpyDeviceToDevice, st));
      CK(cudaMemcpyAsync(df.vc + dst * rb, df.vc + src * rb, rb * 2, cudaMemcpyDeviceToDevice, st));
    }
    if (pending) {
      const size_t pg = (size_t)NGDN * 2 * pending_rows * PEND_GDN * state_es;
      const size_t pc = (size_t)NGDN * pending_rows * CONV_CH * state_es;
      CK(cudaMemcpyAsync((uint8_t*)pend_gdn + dst * pg, (uint8_t*)pend_gdn + src * pg, pg, cudaMemcpyDeviceToDevice, st));
      CK(cudaMemcpyAsync((uint8_t*)pend_conv + dst * pc, (uint8_t*)pend_conv + src * pc, pc, cudaMemcpyDeviceToDevice, st));
      pend_n[dst] = pend_n[src]; commit_next[dst] = commit_next[src]; pend_par[dst] = pend_par[src];
      keep_slot[dst] = keep_slot[src];
    } else { pend_n[dst] = 0; commit_next[dst] = 0; }
  }

  int clone_slot(int src, int dst, int npos) {
    if (src < 0 || dst < 0 || src >= max_slots || dst >= max_slots || npos < 0 || npos > ctx) return -1;
    if (src == dst) return 0;
    copy_slot(src, dst, npos, true);
    CK(cudaStreamSynchronize(st)); // The bounded two-slot discriminator charges the complete clone before verification.
    return 0;
  }

  void reset(int slot) {
    pend_n[slot] = 0; commit_next[slot] = 0;
    CK(cudaMemsetAsync((uint8_t*)gdn_state + (size_t)slot * NGDN * HV * 128 * 128 * state_es, 0, (size_t)NGDN * HV * 128 * 128 * state_es, st));
    CK(cudaMemsetAsync(conv_state + (size_t)slot * NGDN * 3 * CONV_CH, 0, (size_t)NGDN * 3 * CONV_CH * 4, st));
  }

  // Phase profiler (H5_PROF=1): an event before each phase, durations summed by name.
  bool prof = getenv("H5_PROF") != nullptr;
  std::vector<std::pair<const char*, cudaEvent_t>> pev;
  std::map<std::string, double> ptime; int psteps = 0;
  bool sync_debug = getenv("H5_SYNC_DEBUG") != nullptr;
  const char* last_ph = "";
  void ph(const char* name) {
    if (sync_debug) {
      cudaError_t e = cudaStreamSynchronize(st);
      if (e != cudaSuccess) { fprintf(stderr, "error before phase %s (after %s): %s\n", name, last_ph, cudaGetErrorString(e)); exit(1); }
      last_ph = name;
    }
    if (!prof) return;
    cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, st); pev.push_back({name, e});
  }
  void prof_flush() {
    if (!prof || pev.empty()) return;
    cudaEventSynchronize(pev.back().second);
    for (size_t i = 0; i + 1 < pev.size(); i++) { float ms; cudaEventElapsedTime(&ms, pev[i].second, pev[i + 1].second); ptime[pev[i].first] += ms; }
    for (auto& p : pev) cudaEventDestroy(p.second);
    pev.clear(); psteps++;
  }
  void prof_report() {
    if (mprof && mprof_steps) {
      static const char* names[] = {"", "embed", "-", "in+ab", "conv", "gdn", "-", "a:in", "a:pre", "a:attn", "a:comb", "-",
                                    "out", "-", "gate/up", "-", "down", "-", "head", "", "lp:in", "lp:out", "lp:gu", "lp:down", "lp:head", "lp:norm"};
      double tot = 0; for (int i = 0; i < 32; i++) tot += mph[i];
      fprintf(stderr, "megakernel phases over %ld steps (ms/step, barrier to barrier, CTA 0):\n", mprof_steps);
      for (int i = 1; i < 26; i++) if (mph_n[i]) fprintf(stderr, "  %-12s %7.3f  (%5.1f us each)\n", names[i], mph[i] / mprof_steps, mph[i] / mph_n[i] * 1e3);
      fprintf(stderr, "  %-12s %7.3f\n", "total", tot / mprof_steps);
      memset(mph, 0, sizeof mph); memset(mph_n, 0, sizeof mph_n); mprof_steps = 0;
    }
    if (!prof || !psteps) return;
    double tot = 0; for (auto& p : ptime) tot += p.second;
    std::vector<std::pair<double, std::string>> v; for (auto& p : ptime) v.push_back({p.second, p.first});
    std::sort(v.rbegin(), v.rend());
    fprintf(stderr, "phase breakdown over %d steps (ms/step):\n", psteps);
    for (auto& x : v) fprintf(stderr, "  %-14s %8.3f  %5.1f%%\n", x.second.c_str(), x.first / psteps, 100 * x.first / tot);
    fprintf(stderr, "  %-14s %8.3f\n", "total", tot / psteps);
    ptime.clear(); psteps = 0;
  }


  // ================= DFlash2 drafter =================
  int df_window = getenv("H5_DF_WINDOW") ? parse_proposals(getenv("H5_DF_WINDOW")) : DF_WINDOW;
  struct DLayer { TMat qkv, kv, o, gu, down, akp, mkp; float *in_norm, *post_norm, *q_norm, *k_norm, *abase, *mbase; };
  struct Draft {
    bool loaded = false;
    TMat fc[NCAP], hproj; float *hidden_norm, *norm; __nv_bfloat16 *pred, *succ;
    DLayer L[DF_LAYERS];
    __half *kc, *vc;
    float *th, *dkv, *xd, *xn, *cbuf, *tmpo, *dyn, *dqkv, *qr, *dy, *dgu, *hp, *topv; int *topi;
    int *meta; int* h_meta; int* d_drafts; int* h_drafts; float *d_conf, *h_conf;
    int *d_alt, *h_alt, *d_branch, *h_branch; float *d_alt_conf, *h_alt_conf;
  } df;
  TMat nvfp4_tmat(const SafeT& st, const std::vector<std::string>& names, int col0 = 0, int Kc = 0) {
    std::vector<int64_t> sh;
    int N = 0, K = 0;
    for (auto& n : names) { st.t(n, sh); N += (int)sh[0]; K = Kc ? Kc : (int)sh[1]; }
    std::vector<uint8_t> W, S; std::vector<float> RS;
    std::vector<uint8_t> Wall, Sall; std::vector<float> RSall;
    // concatenate along N: chunk layout is [kb][g], so build per name then interleave by kb
    std::vector<std::vector<uint8_t>> Ws, Ss; std::vector<int> NGs;
    for (auto& n : names) {
      const uint16_t* p = st.t(n, sh);
      repack_nvfp4(p, sh[1], (int)sh[0], K, col0, W, S, RS, nthreads);
      Ws.push_back(W); Ss.push_back(S); NGs.push_back((int)sh[0] / 32);
      RSall.insert(RSall.end(), RS.begin(), RS.end());
    }
    const int nkb = K / 128, NG = N / 32;
    Wall.resize((size_t)NG * nkb * 2048); Sall.resize((size_t)NG * nkb * 256);
    for (int kb = 0; kb < nkb; kb++) {
      int g0 = 0;
      for (size_t i = 0; i < Ws.size(); i++) {
        memcpy(Wall.data() + ((size_t)kb * NG + g0) * 2048, Ws[i].data() + (size_t)kb * NGs[i] * 2048, (size_t)NGs[i] * 2048);
        memcpy(Sall.data() + ((size_t)kb * NG + g0) * 256, Ss[i].data() + (size_t)kb * NGs[i] * 256, (size_t)NGs[i] * 256);
        g0 += NGs[i];
      }
    }
    TMat m; m.N = N; m.K = K; m.wf = 1;
    m.w4 = upload<uint4>(Wall.data(), Wall.size() / 16); m.s4 = (__half*)upload<uint8_t>(Sall.data(), Sall.size()); m.rs = upload<float>(RSall.data(), RSall.size());
    return m;
  }
  float* bf16_f32(const SafeT& st, const std::string& n) {
    std::vector<int64_t> sh; const uint16_t* p = st.t(n, sh);
    size_t cnt = 1; for (auto d : sh) cnt *= d;
    std::vector<float> v(cnt); for (size_t i = 0; i < cnt; i++) v[i] = bf2f(p[i]);
    return upload<float>(v.data(), cnt);
  }
  void draft_load(const std::string& path) {
    double t0 = now_s();
    SafeT st; st.open(path);
    for (int j = 0; j < NCAP; j++) df.fc[j] = nvfp4_tmat(st, {"fc.weight"}, j * D, D);
    df.hproj = nvfp4_tmat(st, {"candidate_selector.hidden_projection.weight"});
    df.hidden_norm = bf16_f32(st, "hidden_norm.weight"); df.norm = bf16_f32(st, "norm.weight");
    std::vector<int64_t> sh;
    df.pred = upload<__nv_bfloat16>(st.t("candidate_selector.predecessor_codebook", sh), (size_t)VOCAB * DF_RANK);
    df.succ = upload<__nv_bfloat16>(st.t("candidate_selector.successor_codebook", sh), (size_t)VOCAB * DF_RANK);
    for (int l = 0; l < DF_LAYERS; l++) {
      const std::string p = "layers." + std::to_string(l) + ".";
      DLayer& y = df.L[l];
      y.qkv = nvfp4_tmat(st, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
      y.kv = nvfp4_tmat(st, {p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
      y.o = nvfp4_tmat(st, {p + "self_attn.o_proj.weight"});
      y.gu = nvfp4_tmat(st, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
      y.down = nvfp4_tmat(st, {p + "mlp.down_proj.weight"});
      y.akp = nvfp4_tmat(st, {p + "attention_conv.kernel_projection.weight"});
      y.mkp = nvfp4_tmat(st, {p + "mlp_conv.kernel_projection.weight"});
      y.in_norm = bf16_f32(st, p + "input_layernorm.weight"); y.post_norm = bf16_f32(st, p + "post_attention_layernorm.weight");
      y.q_norm = bf16_f32(st, p + "self_attn.q_norm.weight"); y.k_norm = bf16_f32(st, p + "self_attn.k_norm.weight");
      y.abase = bf16_f32(st, p + "attention_conv.base_kernel"); y.mbase = bf16_f32(st, p + "mlp_conv.base_kernel");
    }
    const size_t R = max_rows;
    const size_t ring = (size_t)max_slots * DF_LAYERS * DF_NKV * DF_RING * DF_HD;
    df.kc = dalloc<__half>(ring); df.vc = dalloc<__half>(ring);
    df.th = dalloc<float>(R * D); df.dkv = dalloc<float>(R * 2 * DF_KV); df.xd = dalloc<float>(R * D); df.xn = dalloc<float>(R * D);
    df.cbuf = dalloc<float>(R * D); df.tmpo = dalloc<float>(R * D); df.dyn = dalloc<float>(R * DF_KPROJ); df.dqkv = dalloc<float>(R * (DF_Q + 2 * DF_KV));
    df.qr = dalloc<float>(R * DF_Q); df.dy = dalloc<float>(R * DF_Q); df.dgu = dalloc<float>(R * GU); df.hp = dalloc<float>(R * DF_RANK);
    df.topv = dalloc<float>(R * DF_TOPK); df.topi = dalloc<int>(R * DF_TOPK);
    df.meta = dalloc<int>(R * 16); CK(cudaMallocHost(&df.h_meta, R * 16 * 4)); df.d_drafts = dalloc<int>(R); CK(cudaMallocHost(&df.h_drafts, R * 4));
    df.d_conf = dalloc<float>(R); CK(cudaMallocHost(&df.h_conf, R * 4));
    df.d_alt = dalloc<int>(R); CK(cudaMallocHost(&df.h_alt, R * 4));
    df.d_branch = dalloc<int>(R); CK(cudaMallocHost(&df.h_branch, R * 4));
    df.d_alt_conf = dalloc<float>(R); CK(cudaMallocHost(&df.h_alt_conf, R * 4));
    df.loaded = true;
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "dflash2 drafter loaded in %.1f s (NVFP4), GPU free %.2f GB\n", now_s() - t0, fr / 1e9);
  }

  // One batched drafter call. Stream i ingests n_ing[i] captured rows starting at hcap row ing_row0[i]
  // (positions ing_pos0[i]...) and, if anchor[i] >= 0, drafts a block anchored at position start[i].
  // drafts: [streams with a block][df_proposals]; query rows include one anchor.
  int draft_step(int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0, const int* anchor, const int* start, int* drafts,
                 int* alternatives = nullptr, int* branches = nullptr, int fork_span = 3) {
    if (!df.loaded || ns < 1 || ns > max_slots || fork_span < 1 || fork_span > 7) return -1;
    if (alternatives && df_proposals != 7) return -6;
    const int block_rows = df_proposals + 1;
    if (alternatives && draft_margin > 0.f) {
      fprintf(stderr, "h5_draft_two requires H5_DRAFT_MARGIN=0 (exact argmax).\n"); return -5;
    }
    struct Force {
      bool& f; bool previous;
      Force(bool& x, bool value = true) : f(x), previous(x) { f = value; }
      ~Force() { f = previous; }
    } force(force_fp4);
    int Ni = 0, nb = 0;
    for (int i = 0; i < ns; i++) {
      if (slot[i] < 0 || slot[i] >= max_slots || n_ing[i] < 0 || ing_row0[i] < 0 ||
          ing_pos0[i] < 0 || (n_ing[i] && (long long)ing_row0[i] + n_ing[i] > captured_rows) ||
          (anchor[i] >= 0 && (anchor[i] >= VOCAB || start[i] < 0 || (long long)start[i] + block_rows > ctx))) return -3;
      for (int j = 0; j < i; j++) if (slot[j] == slot[i]) return -3;
      for (int j = 0; j < n_ing[i]; j++) {
        const int r = ing_row0[i] + j;
        if (h_slot[r] != slot[i] || h_pos[r] != ing_pos0[i] + j) return -4;
      }
      Ni += n_ing[i]; nb += anchor[i] >= 0;
    }
    const int B = nb * block_rows, R7 = nb * df_proposals;
    int nqb = 0;
    if (Ni > max_rows || B > max_rows) return -2;
    // meta layout (ints): ing rowmap | ing slot | ing pos | blk tok | blk slot | blk pos | rowmap7 | anchors | qblk
    int* h = df.h_meta; const size_t M = max_rows;
    int* h_ir = h; int* h_is = h + M; int* h_ip = h + 2 * M; int* h_bt = h + 3 * M; int* h_bs = h + 4 * M; int* h_bp = h + 5 * M; int* h_r7 = h + 6 * M; int* h_an = h + 7 * M;
    QBlk* h_qbd = (QBlk*)(h + 8 * M);
    for (int i = 0, k = 0, b = 0; i < ns; i++) {
      for (int j = 0; j < n_ing[i]; j++, k++) { h_ir[k] = ing_row0[i] + j; h_is[k] = slot[i]; h_ip[k] = ing_pos0[i] + j; }
      if (anchor[i] >= 0) {
        for (int j = 0; j < block_rows; j++) { h_bt[b * block_rows + j] = j ? DF_MASK : anchor[i]; h_bs[b * block_rows + j] = slot[i]; h_bp[b * block_rows + j] = start[i] + j; }
        for (int j = 0; j < df_proposals; j++) h_r7[b * df_proposals + j] = b * block_rows + 1 + j;
        h_an[b] = anchor[i];
        for (int j = 0; j < block_rows; j += QB_ROWS) {
          const int len = std::min(QB_ROWS, block_rows - j);
          h_qbd[nqb++] = QBlk{slot[i], b * block_rows + j, len, start[i] + j + len - 1};
        }
        b++;
      }
    }
    const int span = (df_window + block_rows + 31) / 32 + 1;
    int tps = 8, nsp = (span + tps - 1) / tps;
    if (nb > 0 && nsp * B > part_cap) { nsp = std::max(1, part_cap / B); tps = (span + nsp - 1) / nsp; nsp = (span + tps - 1) / tps; }
    auto body = [&]() {
    CK(cudaMemcpyAsync(df.meta, h, M * 16 * 4, cudaMemcpyHostToDevice, st));
    const int *d_ir = df.meta, *d_is = df.meta + M, *d_ip = df.meta + 2 * M, *d_bt = df.meta + 3 * M, *d_bs = df.meta + 4 * M, *d_bp = df.meta + 5 * M, *d_r7 = df.meta + 6 * M, *d_an = df.meta + 7 * M;
    const QBlk* d_qbd = (const QBlk*)(df.meta + 8 * M);
    ph("d_ingest");
    if (Ni > 0) {
      for (int j = 0; j < NCAP; j++) {
        prep(hcap + (size_t)j * D, nullptr, NCAP * D, nullptr, nullptr, D, P_QUANT, Ni, nullptr, nullptr, 0, d_ir);
        gemm(df.fc[j], Ni, df.th, D, j > 0);
      }
      prep(df.th, nullptr, D, df.hidden_norm, nullptr, D, P_NORM | P_QUANT, Ni);
      for (int l = 0; l < DF_LAYERS; l++) {
        gemm(df.L[l].kv, Ni, df.dkv, 2 * DF_KV, false);
        DAttnPreArgs a{df.dkv, 2 * DF_KV, 0, 0, df.L[l].q_norm, df.L[l].k_norm, d_is, d_ip, nullptr, df.kc, df.vc, l};
        launch_dattn_pre(a, Ni, st);
      }
    }
    if (nb == 0) return;
    ph("d_layers");
    launch_embed(tok_embd, d_bt, tmp, B, st);
    prep(tmp, nullptr, D, nullptr, signs5120, D, P_HADAMARD | P_SIGN_AFTER | P_STORE_F32, B, df.xd, nullptr, D);
    for (int l = 0; l < DF_LAYERS; l++) {
      DLayer& y = df.L[l];
      ph("d_prep");
      prep(df.xd, nullptr, D, y.in_norm, nullptr, D, P_NORM | P_OUT_NORM | P_QUANT, B, nullptr, df.xn, D);
      ph("d_gemm");
      gemm(y.akp, B, df.dyn, DF_KPROJ, false);
      ph("d_conv");
      launch_dconv(df.xn, df.dyn, 0, y.abase, df.cbuf, 0, B, block_rows, st);
      ph("d_prep");
      prep(df.cbuf, nullptr, D, nullptr, nullptr, D, P_QUANT, B);
      ph("d_gemm");
      gemm(y.qkv, B, df.dqkv, DF_Q + 2 * DF_KV, false);
      DAttnPreArgs a{df.dqkv, DF_Q + 2 * DF_KV, DF_Q, 1, y.q_norm, y.k_norm, d_bs, d_bp, df.qr, df.kc, df.vc, l};
      ph("d_attn");
      launch_dattn_pre(a, B, st);
      AttnTcArgs t{df.qr, nullptr, df.kc, df.vc, nullptr, nullptr, l, DF_RING, d_bp, d_qbd, part_o, part_ml, df.dy};
      t.nlayer = DF_LAYERS; t.nkv = DF_NKV; t.hd = DF_HD; t.noncausal = 1; t.window = df_window; t.ring = DF_RING; t.tps = tps;
      t.draft_block_rows = block_rows;
      launch_attn_tc(t, nqb, nsp, B, 2, st);
      ph("d_prep");
      prep(df.dy, nullptr, DF_Q, nullptr, nullptr, DF_Q, P_QUANT, B);
      ph("d_gemm");
      gemm(y.o, B, df.tmpo, D, false);
      ph("d_conv");
      launch_dconv(df.tmpo, df.dyn, 1, y.abase, df.xd, 1, B, block_rows, st);
      ph("d_prep");
      prep(df.xd, nullptr, D, y.post_norm, nullptr, D, P_NORM | P_OUT_NORM | P_QUANT, B, nullptr, df.xn, D);
      ph("d_gemm");
      gemm(y.mkp, B, df.dyn, DF_KPROJ, false);
      ph("d_conv");
      launch_dconv(df.xn, df.dyn, 0, y.mbase, df.cbuf, 0, B, block_rows, st);
      ph("d_prep");
      prep(df.cbuf, nullptr, D, nullptr, nullptr, D, P_QUANT, B);
      ph("d_gemm");
      gemm(y.gu, B, df.dgu, GU, false);
      ph("d_prep");
      prep(df.dgu, df.dgu + FF, GU, nullptr, nullptr, FF, P_SILU_MUL | P_QUANT, B);
      ph("d_gemm");
      gemm(y.down, B, df.tmpo, D, false);
      ph("d_conv");
      launch_dconv(df.tmpo, df.dyn, 1, y.mbase, df.xd, 1, B, block_rows, st);
    }
    ph("d_head");
    prep(df.xd, nullptr, D, df.norm, nullptr, D, P_NORM | P_QUANT, R7, nullptr, nullptr, 0, d_r7);
    gemm(df.hproj, R7, df.hp, DF_RANK, false);
    ph("d_headgemm");
    {
      // The drafter projections are NVFP4, but its candidate head is the shared
      // target head. Prepare activations in that head's configured precision.
      Force target_precision(force_fp4, false);
      prep(df.xd, nullptr, D, df.norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R7, nullptr, nullptr, 0, d_r7);
      for (int r0 = 0; r0 < R7; r0 += 128) {
        const int rr = std::min(128, R7 - r0);
        gemm(head, rr, logits, VOCAB, false, r0);
        ph("d_topk");
        launch_topk(logits, df.topv + (size_t)r0 * DF_TOPK, df.topi + (size_t)r0 * DF_TOPK, rr, st);
      }
    }
    if (sync_debug) {
      std::vector<int> ti((size_t)R7 * DF_TOPK); CK(cudaMemcpy(ti.data(), df.topi, ti.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<float> xs((size_t)B * D); CK(cudaMemcpy(xs.data(), df.xd, xs.size() * 4, cudaMemcpyDeviceToHost));
      int bad = 0;
      for (int r = 0; r < R7; r++) for (int k = 0; k < DF_TOPK; k++) if (ti[r * DF_TOPK + k] < 0 || ti[r * DF_TOPK + k] >= VOCAB) { if (bad++ < 5) fprintf(stderr, "bad topi row %d k %d = %d\n", r, k, ti[r * DF_TOPK + k]); }
      int nanrows = 0; for (int r = 0; r < B; r++) { bool nn = false; for (int i = 0; i < D; i++) if (!std::isfinite(xs[(size_t)r * D + i])) nn = true; if (nn && nanrows++ < 5) fprintf(stderr, "non-finite drafter hidden row %d (block %d)\n", r, r / block_rows); }
      if (bad || nanrows) fprintf(stderr, "B=%d R7=%d Ni=%d nb=%d bad=%d nanrows=%d\n", B, R7, Ni, nb, bad, nanrows);
    }
    ph("d_select");
    if (alternatives) launch_selector_two(df.pred, df.succ, df.hp, df.topv, df.topi, d_an, df.d_drafts, df.d_conf,
        df.d_alt, df.d_alt_conf, df.d_branch, fork_span, nb, st);
    else launch_selector(df.pred, df.succ, df.hp, df.topv, df.topi, d_an, df.d_drafts, df.d_conf, nb, df_proposals, st);
    ph("d_end");
    CK(cudaMemcpyAsync(df.h_drafts, df.d_drafts, R7 * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(df.h_conf, df.d_conf, R7 * 4, cudaMemcpyDeviceToHost, st));
    if (alternatives) {
      CK(cudaMemcpyAsync(df.h_alt, df.d_alt, R7 * 4, cudaMemcpyDeviceToHost, st));
      CK(cudaMemcpyAsync(df.h_alt_conf, df.d_alt_conf, R7 * 4, cudaMemcpyDeviceToHost, st));
      CK(cudaMemcpyAsync(df.h_branch, df.d_branch, nb * 4, cudaMemcpyDeviceToHost, st));
    }
    };
    if (use_graphs && !prof && !sync_debug) run_graph(GKey{alternatives ? 2 : 1, Ni, nb, nsp, tps, alternatives ? fork_span : 0, 0, 0}, body);
    else body();
    CK(cudaStreamSynchronize(st));
    if (nb > 0) {
      memcpy(drafts, df.h_drafts, R7 * 4);
      if (alternatives) memcpy(alternatives, df.h_alt, R7 * 4);
      if (alternatives && branches) memcpy(branches, df.h_branch, nb * 4);
    }
    CK(cudaGetLastError());
    prof_flush();
    return 0;
  }

  // CUDA graphs: a step's whole launch sequence is captured once per launch shape and replayed.
  // The shape (rows, segments, query blocks, attention splits, ...) fixes every grid and by-value
  // argument; per-step data (tokens, positions, segments) lives in pinned host buffers that the
  // captured copies read at replay. A recurring small shape is captured on its third sighting;
  // earlier sightings execute eagerly (and tune any GEMMs not covered at startup).
  struct GKey { int kind, a, b, c, d, e, f, g; bool operator<(const GKey& o) const { return std::tie(kind, a, b, c, d, e, f, g) < std::tie(o.kind, o.a, o.b, o.c, o.d, o.e, o.f, o.g); } };
  bool use_graphs = !getenv("H5_NOGRAPH");
  float draft_margin = 0.f;
  int graph_max_rows = getenv("H5_GRAPH_MAX_ROWS") ? atoi(getenv("H5_GRAPH_MAX_ROWS")) : 64;
  std::map<GKey, cudaGraphExec_t> graphs; std::map<GKey, int> gseen;
  long graph_hits = 0, graph_caps = 0;
  template <class F> void run_graph(const GKey& k, F&& body) {
    auto it = graphs.find(k);
    if (it != graphs.end()) { graph_hits++; CK(cudaGraphLaunch(it->second, st)); return; }
    // capture only shapes that recur (third sighting) and only for small steps, where launch overhead is a
    // real fraction of the step; large batched steps vary in shape and are dominated by kernel time
    if (k.a > graph_max_rows || gseen[k]++ < 2) { body(); return; }
    if (graphs.size() >= 1024) { for (auto& g : graphs) cudaGraphExecDestroy(g.second); graphs.clear(); }
    cudaGraph_t gr;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    body();
    CK(cudaStreamEndCapture(st, &gr));
    cudaGraphExec_t ex;
    CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
    CK(cudaGraphDestroy(gr));
    graphs[k] = ex; graph_caps++;
    CK(cudaGraphLaunch(ex, st));
  }

  int step(int R, const int* slot, const int* tok, const int* pos, int* out, float* host_logits,
           const int* rope = nullptr, const float* visual = nullptr) {
    if (R <= 0 || R > max_rows) return -1;
    if (rope && rope_sections[0] + rope_sections[1] + rope_sections[2] + rope_sections[3] != NROT / 2) return -4;
    int nseg = 0;
    for (int i = 0; i < R; i++) {
      if (slot[i] < 0 || slot[i] >= max_slots || pos[i] < 0 || pos[i] >= ctx) return -2;
      if (tok[i] < -1 || tok[i] >= VOCAB || (tok[i] == -1 && !visual)) return -5;
      h_tok[i] = tok[i]; h_slot[i] = slot[i]; h_pos[i] = pos[i];
      if (i == 0 || slot[i] != slot[i - 1]) {
        const int sl = slot[i];
        h_segs[nseg] = Seg{sl, i, 0, std::min(commit_next[sl], pend_n[sl]), keep_slot[sl], pend_par[sl]}; nseg++;
      }
      h_segs[nseg - 1].len++;
    }
    for (int g = 0; g < nseg; g++) {
      Seg& q = h_segs[g];
      if (q.keep && q.len > pending_rows) return -3;
      pend_n[q.slot] = q.keep ? q.len : 0; commit_next[q.slot] = 0;
      if (q.keep) pend_par[q.slot] ^= 1;
    }
    int nqb = 0, maxlen = 0, maxkend = 0;
    for (int g = 0; g < nseg; g++)
      for (int r = 0; r < h_segs[g].len; r += QB_ROWS) {
        const int len = std::min(QB_ROWS, h_segs[g].len - r), start = h_segs[g].start + r;
        h_qb[nqb++] = QBlk{h_segs[g].slot, start, len, h_pos[start + len - 1]};
        maxlen = std::max(maxlen, len); maxkend = std::max(maxkend, h_pos[start + len - 1] + 1);
      }
    // splits: enough CTAs to fill the GPU twice, >= 128 keys each, partials within capacity
    // splits of 8 tiles (256 keys) at absolute boundaries, so a row's attention does not depend on the batch
    // it rides in; wider splits only when the partial buffer would overflow
    const int span = (maxkend + 31) / 32;
    static const int tps0 = getenv("H5_ATTN_TPS") ? atoi(getenv("H5_ATTN_TPS")) : 16;
    int tps = tps0;
    // few query blocks: split the keys finer so the attention grid fills the GPU (~1 CTA per SM)
    if (!getenv("H5_ATTN_TPS") && nqb * NKV * ((span + tps - 1) / tps) < nsm) tps = std::max(1, std::min(tps0, (span * nqb * NKV + nsm - 1) / nsm));
    int nsp = (span + tps - 1) / tps;
    if (nsp * R > part_cap) { nsp = std::max(1, part_cap / R); tps = (span + nsp - 1) / nsp; nsp = (span + tps - 1) / tps; }
    const int max_mt = (maxlen * (NH / NKV) + 15) / 16;
    double t0 = now_s();
    if (rope) CK(cudaMemcpyAsync(d_rope, rope, (size_t)R * 4 * sizeof(int), cudaMemcpyHostToDevice, st));
    if (visual) CK(cudaMemcpyAsync(d_visual, visual, (size_t)R * D * sizeof(float), cudaMemcpyHostToDevice, st));
    auto body = [&]() {
    CK(cudaMemcpyAsync(d_qb, h_qb, nqb * sizeof(QBlk), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_tok, h_tok, R * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_slot, h_slot, R * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_pos, h_pos, R * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_segs, h_segs, nseg * sizeof(Seg), cudaMemcpyHostToDevice, st));
    ph("embed");
    launch_embed(tok_embd, d_tok, tmp, R, st);
    prep(tmp, nullptr, D, nullptr, signs5120, D, P_HADAMARD | P_SIGN_AFTER | P_STORE_F32, R, x, nullptr, D);
    // The table is Hadamard-latent; mmproj output is already in the primal basis.
    if (visual) launch_visual_input(d_tok, d_visual, x, R, st);
    for (int l = 0; l < NLAYER; l++) {
      Layer& Ly = L[l];
      if (!Ly.attn) {
        ph("prep");
        prep(x, nullptr, D, Ly.attn_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT | P_OUT_NORM, R, nullptr, xn, D);
        ph("gemm_in");
        gemm(Ly.in, R, proj, GDN_IN, false);
        ph("ab");
        if (ab_bf16 || !act_fp4) launch_ab(Ly.ab, xn, ab, R, st);
        else gemm(Ly.ab4, R, ab, 96, false);
        ph("conv");
        ConvArgs c{proj, Ly.conv_w, conv_state, Ly.gl, conv_out, d_segs, pend_conv, ab, Ly.ssm_a, Ly.ssm_dt, gates};
        c.pending_rows = pending_rows;
        launch_conv(c, nseg, state_format, st);
        GdnArgs ga{conv_out, proj, ab, Ly.ssm_a, Ly.ssm_dt, Ly.ssm_norm, gdn_state, Ly.gl, y, d_segs, pend_gdn, gates};
        static const bool replay_delta = getenv("H5_GDN_DELTA") && atoi(getenv("H5_GDN_DELTA"));
        ga.replay_delta = replay_delta; ga.pending_rows = pending_rows;
        ph("gdn");
        launch_gdn(ga, nseg, state_format, st);
        ph("prep");
        {
          PrepArgs pa{}; pa.x = y; pa.ldx = VDIM; pa.signs = signs6144; pa.n = VDIM; pa.flags = P_PERMUTE_GDN | P_GDN_NORM | P_SIGN | P_HADAMARD | P_QUANT;
          pa.xq = xq; pa.xs = xs; if (act_fp4 || force_fp4) { pa.q4 = q4; pa.s4 = s4; pa.g4 = g4; }
          pa.gnorm_w = Ly.ssm_norm; pa.gz = proj + CONV_CH; pa.gz_ld = GDN_IN;
          launch_prep(pa, R, st);
        }
        ph("gemm_out");
        gemm(Ly.out, R, x, D, true);
      } else {
        ph("prep");
        prep(x, nullptr, D, Ly.attn_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R);
        ph("gemm_in");
        gemm(Ly.in, R, proj, ATT_IN, false);
        ph("attn_pre");
        AttnArgs a{proj, Ly.q_norm, Ly.k_norm, kc, vc, ksc, vsc, Ly.al, ctx, d_slot, d_pos, qrot, y};
        a.vt = !getenv("H5_ATTN8") || atoi(getenv("H5_ATTN8"));
        a.rope_pos = rope ? d_rope : nullptr;
        for (int j = 0; j < 4; j++) a.rope_sections[j] = rope_sections[j];
        launch_attn_pre(a, R, st);
        ph("attn");
        AttnTcArgs t{qrot, proj, kc, vc, ksc, vsc, Ly.al, ctx, d_pos, d_qb, part_o, part_ml, y};
        t.tps = tps;
        if (kv8) { launch_attn_tc(t, nqb, nsp, 0, max_mt, st); ph("attn_comb"); launch_attn_combine(t, nsp, R, st); }
        else launch_attn_tc(t, nqb, nsp, R, max_mt, st);
        ph("prep");
        prep(y, nullptr, ATTN_OUT, nullptr, signs6144, ATTN_OUT, P_SIGN | P_HADAMARD | P_QUANT, R);
        ph("gemm_out");
        gemm(Ly.out, R, x, D, true);
      }
      ph("prep");
      prep(x, nullptr, D, Ly.post_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R);
      ph("gemm_gu");
      if (Ly.gu.il && (act_fp4 || force_fp4)) {
        gemm_silu(Ly.gu, R, proj, FF);
        ph("prep");
        prep(proj, nullptr, FF, nullptr, signs17408, FF, P_SIGN | P_HADAMARD | P_QUANT, R);
      } else {
        gemm(Ly.gu, R, proj, GU, false);
        ph("prep");
        prep(proj, proj + FF, GU, nullptr, signs17408, FF, P_SILU_MUL | P_SIGN | P_HADAMARD | P_QUANT, R);
      }
      ph("gemm_down");
      gemm(Ly.down, R, x, D, true);
      if (capture)
        for (int j = 0; j < NCAP; j++)
          if (l == cap_layers[j]) CK(cudaMemcpy2DAsync(hcap + (size_t)j * D, NCAP * D * 4, x, D * 4, D * 4, R, cudaMemcpyDeviceToDevice, st));
    }
    ph("prep");
    prep(x, nullptr, D, output_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R);
    ph("head");
    for (int r0 = 0; r0 < R; r0 += 128) {
      const int rr = std::min(128, R - r0);
      gemm(head, rr, logits, VOCAB, false, r0);
      if (draft_margin > 0.f)
        launch_relaxed_argmax(logits, VOCAB, d_out + r0, rr, st, d_tok, d_segs, nseg, r0, draft_margin);
      else launch_argmax(logits, VOCAB, d_out + r0, rr, st);
      if (host_logits) CK(cudaMemcpyAsync(host_logits + (size_t)r0 * VOCAB, logits, (size_t)rr * VOCAB * 4, cudaMemcpyDeviceToHost, st));
    }
    ph("end");
    CK(cudaMemcpyAsync(h_out, d_out, R * 4, cudaMemcpyDeviceToHost, st));
    };
    const bool graphable = use_graphs && !prof && !sync_debug && !host_logits && !rope && !visual;
    if (!rope && !visual && use_mega && !wscale_e4m3 && nseg == 1 && R <= 8 && act_fp4 && !host_logits && !prof) {
      CK(cudaMemcpyAsync(d_qb, h_qb, nqb * sizeof(QBlk), cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_tok, h_tok, R * 4, cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_slot, h_slot, R * 4, cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_pos, h_pos, R * 4, cudaMemcpyHostToDevice, st));
      run_mega(R, nsp, tps, max_mt);
      CK(cudaMemcpyAsync(h_out, d_mout, R * 4, cudaMemcpyDeviceToHost, st));
    }
    else if (graphable) run_graph(GKey{0, R, nseg, nqb, nsp, tps, max_mt, (int)capture}, body);
    else body();
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    prof_flush();
    last_step_s = now_s() - t0;
    captured_rows = capture ? R : 0;
    memcpy(out, h_out, R * 4);
    return 0;
  }
};

}  // namespace h5

using h5::Engine;
static std::atomic<FILE*> h5_record_ptr{nullptr};
extern "C" {
int h5_record_begin(const char* path) {
  FILE*f=fopen(path,"wx"); if(!f) return -1;
  FILE*expected=nullptr; if(!h5_record_ptr.compare_exchange_strong(expected,f)) { fclose(f); return -2; } return 0;
}
int h5_record_end() { FILE*f=h5_record_ptr.exchange(nullptr); return f ? fclose(f) : -1; }
void* h5_create(const char* path, int slots, int ctx, int rows) {
  try { return new Engine(path, slots, ctx, rows); } catch (std::exception& e) { fprintf(stderr, "h5_create: %s\n", e.what()); return nullptr; }
}
int h5_step(void* e, int R, const int* slot, const int* tok, const int* pos, int* out) {
  if(FILE*f=h5_record_ptr.load()) { fprintf(f,"STEP %d",R); for(int i=0;i<R;i++) fprintf(f," %d %d %d",slot[i],tok[i],pos[i]); fprintf(f,"\n"); }
  try { return ((Engine*)e)->step(R, slot, tok, pos, out, nullptr); } catch (std::exception& ex) { fprintf(stderr, "h5_step: %s\n", ex.what()); return -9; }
}
int h5_step_visual(void* e, int R, const int* slot, const int* tok, const int* pos,
                   const int* rope, const float* visual, int* out) {
  try { return ((Engine*)e)->step(R, slot, tok, pos, out, nullptr, rope, visual); }
  catch (std::exception& ex) { fprintf(stderr, "h5_step_visual: %s\n", ex.what()); return -9; }
}
int h5_step_visual_logits(void* e, int R, const int* slot, const int* tok, const int* pos,
                          const int* rope, const float* visual, int* out, float* logits) {
  try { return ((Engine*)e)->step(R, slot, tok, pos, out, logits, rope, visual); }
  catch (std::exception& ex) { fprintf(stderr, "h5_step_visual_logits: %s\n", ex.what()); return -9; }
}
int h5_step_logits(void* e, int R, const int* slot, const int* tok, const int* pos, int* out, float* logits) {
  try { return ((Engine*)e)->step(R, slot, tok, pos, out, logits); } catch (std::exception& ex) { fprintf(stderr, "h5_step: %s\n", ex.what()); return -9; }
}
void h5_gemm_bench(void* e, int R, int reps) { ((Engine*)e)->gemm_bench(R, reps); }
void* h5_create_draft(const char* path, int slots, int ctx, int rows, int proposals) {
  const auto cfg = h5::draft_config(proposals, slots, false, ctx, 1);
  if (cfg.error != h5::DraftError::None || rows < cfg.query_rows) {
    fprintf(stderr, "h5_create_draft: %s (max_rows=%d)\n", cfg.error == h5::DraftError::None ? "draft row capacity too small" : h5::draft_error(cfg.error), rows); return nullptr;
  }
  try { return new Engine(path, slots, ctx, rows, proposals); } catch (std::exception& e) { fprintf(stderr, "h5_create_draft: %s\n", e.what()); return nullptr; }
}
int h5_draft_load(void* e, const char* path) { try { ((Engine*)e)->draft_load(path); return 0; } catch (std::exception& ex) { fprintf(stderr, "draft_load: %s\n", ex.what()); return -1; } }
void h5_capture(void* e, int on) { ((Engine*)e)->capture = on; }
int h5_draft(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0, const int* anchor, const int* start, int* drafts) {
  try { return ((Engine*)e)->draft_step(ns, slot, n_ing, ing_row0, ing_pos0, anchor, start, drafts); } catch (std::exception& ex) { fprintf(stderr, "h5_draft: %s\n", ex.what()); return -9; }
}
int h5_draft_two(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0,
    const int* anchor, const int* start, int* drafts, int* alternatives, int* branches, int fork_span) {
  if (!alternatives) return -1;
  try { return ((Engine*)e)->draft_step(ns, slot, n_ing, ing_row0, ing_pos0, anchor, start, drafts, alternatives, branches, fork_span); }
  catch (std::exception& ex) { fprintf(stderr, "h5_draft_two: %s\n", ex.what()); return -9; }
}
int h5_clone_slot(void* e, int src, int dst, int npos) {
  try { return ((Engine*)e)->clone_slot(src, dst, npos); }
  catch (std::exception& ex) { fprintf(stderr, "h5_clone_slot: %s\n", ex.what()); return -9; }
}
const float* h5_draft_conf(void* e) { return ((Engine*)e)->df.h_conf; }
void h5_keep(void* e, int slot, int flag) { if(FILE*f=h5_record_ptr.load()) fprintf(f,"KEEP %d %d\n",slot,flag); ((Engine*)e)->keep_slot[slot] = flag; }
void h5_accept(void* e, int slot, int n) { if(FILE*f=h5_record_ptr.load()) fprintf(f,"ACCEPT %d %d\n",slot,n); ((Engine*)e)->commit_next[slot] = n; }
void h5_prof_report(void* e) { ((Engine*)e)->prof_report(); }
void h5_copy_slot(void* e, int src, int dst, int npos) { ((Engine*)e)->copy_slot(src, dst, npos); }
void h5_reset(void* e, int slot) { if(FILE*f=h5_record_ptr.load()) fprintf(f,"RESET %d\n",slot); ((Engine*)e)->reset(slot); }
void h5_pretune(void* e) { ((Engine*)e)->pretune(); }
double h5_last_step_s(void* e) { return ((Engine*)e)->last_step_s; }
long h5_graph_stats(void* e, int which) { return which ? ((Engine*)e)->graph_caps : ((Engine*)e)->graph_hits; }
// logits of the last head chunk (rows r0..r0+127 of the final chunk), for reference comparison
int h5_logits(void* e, int row, float* out) {
  Engine* en = (Engine*)e; cudaMemcpy(out, en->logits + (size_t)row * h5::VOCAB, h5::VOCAB * 4, cudaMemcpyDeviceToHost); return 0;
}
}
