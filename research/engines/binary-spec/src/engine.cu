// halo-5090 engine: weights, state, one batched step. C API at the bottom (used by python/h5.py).
#include "gguf.h"
#include "dspark_native.cuh"
#include "packed1.cuh"
#include "../experiments/binary-bitserial/bitserial.cuh"
#include "../experiments/binary-a8-fp4/digits.cuh"
#include "../experiments/binary-whole-map/signed_digits.cuh"
#include "../experiments/binary-whole-map/whole_ffn.cuh"
#include "../experiments/binary-native-sf/native.cuh"
#include "../experiments/binary-composition/composition.cuh"
#include "../experiments/binary-fp4-pair/pair.cuh"
#include "compact.hpp"
#include "ternary_operands.h"
#include "graph_probe.h"
#include "head_demand.h"
#include "draft_probe.h"
#include "gdn_diagnostic.h"
#include "graph_memory.h"
#include "scale_carrier.cuh"
#include "f02.cuh"
#include "f02_route.h"
#include "f07.cuh"
#include "f07_contract.h"
#include "sk.h"
#include "sampling_gpu.cuh"
#include "deterministic_gemm.cuh"
#include <time.h>
#include <cuda_fp8.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include "kernels.cuh"
#include "gdn_single.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include "serving_policy.h"
#include <stdexcept>
#include <chrono>
#include <map>
#include <algorithm>
#include <tuple>

namespace h5 {
void launch_gemm(const int8_t*, const float*, int, int, const uint4*, const float*, int, float*, int, bool, cudaStream_t);
void launch_gemm_fp4_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
void launch_gemm_fp4_precleared_cfg(int, int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, bool, cudaStream_t, const float*, int);
int fp4_ncfg();
bool fp4_cfg_wg2(int);
void launch_gemm_fp4_silu(int, const uint8_t*, const uint8_t*, const float*, int, int, const void*, const void*, int, float*, int, cudaStream_t, const float*, int);
void launch_dconv(const float*, const float*, int, const float*, float*, int, int, int, cudaStream_t);
void launch_dattn_pre(const DAttnPreArgs&, int, cudaStream_t);
void launch_topk(const float*, int, const int*, float*, int*, int, TopkScratch, cudaStream_t);
cudaError_t launch_draft_dependency_fence(cudaStream_t);
void launch_selector(const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const int*, const int*, int*, float*, int, int, const DraftLz&, cudaStream_t);
void launch_selector_two(const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const int*, const int*, int*, float*, int*, float*, int*, int, int, cudaStream_t);
const char* fp4_cfg_name(int, int);
void launch_prep(const PrepArgs&, int, cudaStream_t);
void launch_embed(const uint8_t*, const int*, float*, int, cudaStream_t);
void launch_embed_q1(const uint8_t*, const int*, float*, int, cudaStream_t);
void launch_visual_input(const int*, const float*, float*, int, cudaStream_t);
void launch_ab(const __nv_bfloat16*, const float*, float*, int, cudaStream_t);
void launch_conv(const ConvArgs&, int, StateFormat, cudaStream_t);
void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t);
void launch_conv_gdn(const ConvArgs&, const GdnArgs&, int, StateFormat, cudaStream_t);
void launch_attn_pre(const AttnArgs&, int, cudaStream_t);
void launch_attn_tc(const AttnTcArgs&, int, int, int, int, cudaStream_t);
void launch_attn_combine(const AttnTcArgs&, int, int, cudaStream_t);
void launch_embed(const uint8_t*, const int*, float*, int, cudaStream_t);
void launch_argmax(const float*, int, int*, int, cudaStream_t);
void launch_relaxed_argmax(const float*, int, int*, int, cudaStream_t, const int*, const Seg*, int, int, float, const int*, RowStat*, int);
void launch_logit_adj(float*, int, int, const int2*, int, int, const int*, cudaStream_t);
void launch_mega(const MegaArgs&, int, cudaStream_t);
int mega_nsm();

struct CudaFailure : std::runtime_error {
  cudaError_t status;
  CudaFailure(cudaError_t code, const char* operation)
      : std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code)), status(code) {}
};
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw h5::CudaFailure(e_, #x); } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct TuningEvents {
  cudaEvent_t start, end;
  std::string context, rejected;
  explicit TuningEvents(std::string where) : context(std::move(where)) {
    CK(cudaEventCreate(&start));
    const auto error = cudaEventCreate(&end);
    if (error != cudaSuccess) { cudaEventDestroy(start); CK(error); }
  }
  ~TuningEvents() { cudaEventDestroy(start); cudaEventDestroy(end); }
  void begin(cudaStream_t stream) { CK(cudaEventRecord(start, stream)); }
  bool accepted(int cfg, int split) {
    const auto error = cudaGetLastError();
    if (error == cudaSuccess) return true;
    rejected = context + " cfg=" + std::to_string(cfg) + " split=" + std::to_string(split) + ": " + cudaGetErrorString(error);
    if (error == cudaErrorInvalidConfiguration || error == cudaErrorLaunchOutOfResources || error == cudaErrorInvalidValue) return false;
    throw std::runtime_error(rejected);
  }
  float finish(cudaStream_t stream) {
    CK(cudaEventRecord(end, stream));
    const auto error = cudaEventSynchronize(end);
    if (error != cudaSuccess) throw std::runtime_error(context + ": asynchronous CUDA failure: " + cudaGetErrorString(error));
    float ms; CK(cudaEventElapsedTime(&ms, start, end)); return ms;
  }
  void require_candidate(bool found) const {
    if (!found) throw std::runtime_error("no launchable " + context + "; last rejection: " + rejected);
  }
};
struct GraphMemoryFailure : std::runtime_error {
  GraphMemoryError code;
  explicit GraphMemoryFailure(GraphMemoryResult result) :
    std::runtime_error("graph-memory rewrite error " + std::to_string((int)result.error) + ": " + cudaGetErrorString(result.cuda_error)), code(result.error) {}
};

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
extern "C" void h5_set_prewait_cap(int);
extern "C" void h5_set_l2pf_stages(int);
static int& fuse_gu_slot() {
  static int v = [] { const char* c = getenv("H5_FUSE_GU"); if (!c) return 0;
    if (strcmp(c, "0") && strcmp(c, "1")) throw std::runtime_error("H5_FUSE_GU must be 0 or 1"); return c[0] == '1' ? 1 : 0; }();
  return v;
}
static bool fuse_gu_selected() { return fuse_gu_slot() != 0; }
extern "C" void h5_set_fuse_gu(int v) { if (v != 0 && v != 1) abort(); fuse_gu_slot() = v; }
// Head screening probe (H5_HEAD_SCREEN_PROBE=1): rank of the exact argmax among partial logits computed
// from a strided subset of K blocks. Histogram bucket b counts ranks in [2^(b-1), 2^b) (bucket 0 = rank 0);
// one histogram per screen. Host-mapped so an exit handler can print it after the context is gone.
__global__ void k_screen_rank(const float* screen, int ld, const int* exact, int* hist) {
  pdl_wait(); pdl_launch();
  const int row = blockIdx.x, e = exact[row];
  const float* s = screen + (size_t)row * ld;
  const float se = s[e];
  int c = 0;
  for (int v = threadIdx.x; v < ld; v += blockDim.x) c += s[v] > se || (s[v] == se && v < e);
  for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
  __shared__ int part[32];
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    int t = 0; for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += part[w];
    int b = 0; while (b < 15 && (1 << b) <= t) b++;
    atomicAdd(hist + b, 1);
  }
}
// drafter_requant: a drafter projection requantised at load from bf16 (H5_DRAFT_WFMT: ternary dense5 or the
// 3/2-bit E2M1-subset formats); it takes the frozen plan's NVFP4 entry for its shape (plan format 1).
struct TMat { uint32_t* whole_gu = nullptr; __half* whole_gu_scales = nullptr; uint32_t* w1 = nullptr; __half* s1 = nullptr; uint4* w = nullptr; float* s = nullptr; uint4* w4 = nullptr; __half* s4 = nullptr; uint8_t* native_sf = nullptr; float* native_row_factor = nullptr; int N = 0, K = 0; float* rs = nullptr; int wf = 0; bool il = false; float* mean_scale = nullptr; bool drafter_requant = false;
  int plan_wf() const { return drafter_requant ? 1 : wf; } };

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

// One 32-row x 128-K chunk of trits with FP16 row scales into the WF0 (1024 B, two-bit) or dense5 (832 B)
// lane-fragment layout (see fp4.cu) and its [gid][4] FP16 scale block.
static void pack_trit_chunk(const int8_t (&trit)[32][128], const float (&sc)[32], uint8_t* chunk, uint16_t* sh, bool compact) {
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
        for (int gid = 0; gid < 8; gid++)
          for (int q = 0; q < 4; q++) {
            const int m = q >> 1, r = q & 1, row = 16 * m + 8 * r + gid;
            __half hv = __float2half(sc[row]); uint16_t u; memcpy(&u, &hv, 2); sh[gid * 4 + q] = u;
          }
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
        pack_trit_chunk(trit, sc, W.data() + ((size_t)kb * NGall + g) * (compact ? 832 : 1024), S.data() + ((size_t)kb * NGall + g) * 32, compact);
      }
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& x : th) x.join();
}

static void decode_q1(const uint8_t* raw, int8_t* values, float* scale) {
  __half_raw h; memcpy(&h.x, raw, 2); *scale = __half2float(__half(h));
  for (int i = 0; i < 128; ++i) values[i] = (raw[2 + i / 8] >> (i % 8)) & 1 ? 1 : -1;
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
        for (int r = 0; r < 32; r++) {
          const uint8_t* raw = t.data + ((size_t)(gl * 32 + r) * nkb + kb) * (t.type == 41 ? 18 : 28);
          if (t.type == 41) decode_q1(raw, trit[r], &sc[r]); else decode_ptq1(raw, trit[r], &sc[r]);
        }
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
  float *attn_norm_s = nullptr, *post_norm_s = nullptr;   // norm weight * Hadamard sign (producer-tail prep)
  TMat in, out, gu, down;
  __nv_bfloat16* ab = nullptr; TMat ab4;
  float *conv_w = nullptr, *ssm_a = nullptr, *ssm_dt = nullptr, *ssm_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
};

struct Engine {
  Gguf g;
  int max_slots, ctx, max_rows, nthreads;
  cudaStream_t st;
  // Side-stream L2 weight prefetch (H5_L2PF_SIDE_MB=cap per GEMM, unset = off). Prefetch of GEMM g
  // is forked after GEMM g-1 completes (or at step start) and joined before the step's capture ends.
  cudaStream_t ab_st = nullptr; cudaEvent_t ab_fork = nullptr, ab_join = nullptr;   // H5_AB_PARALLEL branch
  double l2pf_side_bytes = -1; cudaStream_t pf_st = nullptr; cudaEvent_t pf_fork = nullptr, pf_join = nullptr;
  bool pf_forked = false, pf_used = false, pf_capturing = false;
  void pf_begin_step() {
    if (l2pf_side_bytes <= 0) return;
    CK(cudaEventRecord(pf_fork, st)); pf_forked = true; pf_used = false; pf_capturing = true;
  }
  void pf_before_gemm(const void* w, int N, int K, int wf, int ks) {
    if (l2pf_side_bytes <= 0 || !pf_forked) return;
    CK(cudaStreamWaitEvent(pf_st, pf_fork, 0));
    CK(launch_l2pf(w, N, K, wf, ks, l2pf_side_bytes, pf_st));
    pf_used = true; pf_forked = false;
  }
  void pf_after_gemm() { if (l2pf_side_bytes > 0 && pf_capturing) { CK(cudaEventRecord(pf_fork, st)); pf_forked = true; } }
  void pf_end_step() {
    if (l2pf_side_bytes <= 0) return;
    if (pf_used) { CK(cudaEventRecord(pf_join, pf_st)); CK(cudaStreamWaitEvent(st, pf_join, 0)); }
    pf_forked = false; pf_used = false; pf_capturing = false;
  }
  float *signs5120 = nullptr, *signs6144 = nullptr, *signs17408 = nullptr, *output_norm; std::vector<float> hsigns5120;
  uint8_t* tok_embd;
  TMat head;
  std::vector<Layer> L;
  // state
  void* gdn_state; float* conv_state; void *kc, *vc; float *ksc = nullptr, *vsc = nullptr;
  StateFormat state_format = StateFormat::F16;
  const Q8ScaleMode q8_scale = gdn_q8_scale_mode();
  bool kv8 = true; size_t state_es = 2, state_slot_bytes = 0;
  bool gdn_diagnostic = false;
  void* diagnostic_state = nullptr;
  float* diagnostic_y = nullptr;
  StateFormat diagnostic_format = StateFormat::F16;
  size_t diagnostic_slot_bytes = 0;
  bool diagnostic_readout_enabled = false, diagnostic_readout_armed = false;
  bool diagnostic_readout_capturing = false, diagnostic_readout_ready = false;
  std::vector<int> diagnostic_readout_layers, diagnostic_readout_heads;
  std::vector<GdnReadoutSample> diagnostic_readout_samples;
  std::vector<float> diagnostic_readout_host;
  size_t readout_index(int source, int slot, size_t layer, size_t head) const {
    return (((size_t)source * max_slots + slot) * diagnostic_readout_layers.size() + layer) * diagnostic_readout_heads.size() + head;
  }
  void observe_gdn_readout(int layer, int nseg) {
    if (!diagnostic_readout_capturing) return;
    const auto found = std::lower_bound(diagnostic_readout_layers.begin(), diagnostic_readout_layers.end(), layer);
    if (found == diagnostic_readout_layers.end() || *found != layer) return;
    const size_t li = found - diagnostic_readout_layers.begin();
    for (int source = 0; source < (diagnostic_state ? 2 : 1); ++source) {
      const float* output = source == 0 ? y : diagnostic_y;
      for (int si = 0; si < nseg; ++si) {
        const Seg& segment = h_segs[si];
        const int row = segment.start + segment.len - 1;
        CK(cudaMemcpyAsync(diagnostic_readout_host.data(), output + (size_t)row * VDIM, VDIM * sizeof(float), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (size_t hi = 0; hi < diagnostic_readout_heads.size(); ++hi) {
          const int head = diagnostic_readout_heads[hi];
          diagnostic_readout_samples[readout_index(source, segment.slot, li, hi)] = diagnostic_readout_stats(
              diagnostic_readout_host.data() + head * SS, SS, row, h_tok[row], h_pos[row]);
        }
      }
    }
  }
  // activations
  float *x, *tmp, *xn, *proj, *conv_out, *ab, *y, *qrot, *logits;
  float* sk_ws = nullptr; int* sk_cnt = nullptr; int* sk_next = nullptr;   // stream-K split-tile workspace and arrival counters; all-zero between launches
  float* gu_acc = nullptr;   // [8][GU] split-K GU accumulator (H5_GU_SPLIT); all-zero between uses
  int8_t* xq; float* xs; uint8_t *q4, *s4; float* g4; float* prep_rms_inv;
  // Producer-tail prep (H5_TAIL_PREP, tail_prep.cuh): a tail writes the next GEMM's activations while
  // CTAs of its own GEMM may still read the current ones, so tails write the other buffer of a
  // 64-row ping-pong pair and q4/s4/g4 then point at it. Every step body starts and ends on the base.
  uint8_t *q4_base = nullptr, *s4_base = nullptr, *q4_alt = nullptr, *s4_alt = nullptr; float *g4_base = nullptr, *g4_alt = nullptr;
  unsigned *tail_cnt = nullptr, *tail_rdv = nullptr; float* tail_part = nullptr;
  unsigned* flag_sig = nullptr;   // H5_FLAG_WAIT / H5_GEMM_FLAG: [NLAYER][8] producer-done counters, zeroed at the start of every step body
  unsigned* prep_done_sig_next = nullptr;
  bool prep_early_next = false;   // H5_PREP_EARLY: the next prep() call triggers its dependents before its wait
  unsigned long long* argmax_keys = nullptr; unsigned* argmax_cnt = nullptr;   // H5_ARGMAX_SPLIT scratch [128], all-zero between steps
  // H5_TAIL_DBG=1: finaliser stamps of the last launch per call site: 0 GDN, 1 attention combine, 2 output, 3 gate/up, 4 down.
  unsigned long long* tail_dbg = nullptr;
  void act_flip() {
    const bool base = q4 == q4_base;
    q4 = base ? q4_alt : q4_base; s4 = base ? s4_alt : s4_base; g4 = base ? g4_alt : g4_base;
  }
  void act_reset() { q4 = q4_base; s4 = s4_base; g4 = g4_base; }
  TailArgs tail_args(const float* xin, int ldx, int n, int R, unsigned flags, int dbg_slot) {
    TailArgs t; t.x = xin; t.ldx = ldx; t.n = n; t.R = R; t.flags = flags;
    const bool base = q4 == q4_base;
    t.q4 = base ? q4_alt : q4_base; t.s4 = base ? s4_alt : s4_base; t.g4 = base ? g4_alt : g4_base;
    t.fixed_g = prep_fixed_g(); t.cnt = tail_cnt; t.rdv = tail_rdv; t.part = tail_part;
    if (tail_dbg) t.dbg = tail_dbg + (size_t)dbg_slot * TAIL_MAX_CHUNKS * TAIL_DBG_STAMPS;
    return t;
  }
  int *d_tok, *d_slot, *d_pos, *d_out, *d_rope; float* d_visual; Seg* d_segs;
  int rope_sections[4] = {0, 0, 0, 0};
  int *h_tok, *h_slot, *h_pos, *h_out; Seg* h_segs;
  int *d_head_rows, *h_head_rows;
  int last_logits_rows = 0;
  GpuSampler gpu_sampler;
  FILE* sampling_trace = nullptr;
  int sampling_trace_limit = 0, sampling_trace_stride = 0, sampling_trace_left = 0;
  uint64_t sampling_trace_calls = 0, draft_trace_seed = 0, draft_trace_first = UINT64_MAX;
  H5SamplingConfig draft_trace_config{};
  std::vector<int> draft_trace_ids, draft_trace_tokens;
  std::vector<double> draft_trace_scores;
  bool trace_due() const { return sampling_trace && sampling_trace_left > 0 && sampling_trace_calls % sampling_trace_stride == 0; }
  bool deterministic_gemm = [] {
    const char* value = getenv("H5_DETERMINISTIC_GEMM");
    if (!value || !strcmp(value, "0")) return false;
    if (!strcmp(value, "1")) return true;
    throw std::runtime_error("H5_DETERMINISTIC_GEMM must be 0 or 1");
  }();
  struct DeterministicScratch { float* pointer = nullptr; size_t capacity = 0; std::vector<float*> retained; };
  std::map<cudaStream_t, DeterministicScratch> deterministic_scratch;
  float* deterministic_partials(size_t required) {
    if (!required) return nullptr;
    auto& buffer = deterministic_scratch[st];
    if (required > buffer.capacity) {
      cudaStreamCaptureStatus status;
      CK(cudaStreamIsCapturing(st, &status));
      if (status != cudaStreamCaptureStatusNone) throw std::runtime_error("deterministic GEMM scratch must be sized before graph capture");
      float* next; CK(cudaMalloc(&next, required * sizeof(float)));
      if (buffer.pointer) buffer.retained.push_back(buffer.pointer);
      buffer.pointer = next; buffer.capacity = required;
    }
    return buffer.pointer;
  }
  // speculation: per-slot pending rows of the last keep step, rows to commit at the next step
  void *pend_gdn = nullptr, *pend_conv = nullptr;
  std::vector<int> pend_n, commit_next, pend_par, conv_par; std::vector<char> keep_slot;
  bool state_fusion = false, attn_direct = false;
  int conv_banks = 1;
  // drafter features: residual after CAP layers for every row of a step
  static constexpr int DF_INGEST_FUSED_ROWS = 16;   // fused ingest covers verifier rounds up to 15 proposals (q4 capacity: 64-row tile x NCAP*D)
  static constexpr int NCAP = 5; int cap_layers[NCAP] = {5, 19, 33, 47, 61};
  std::unique_ptr<NativeDspark> native_dspark;
  bool dspark_mode = false;
  float* hcap = nullptr; bool capture = false; int captured_rows = 0;
  QBlk *d_qb, *h_qb; float* part_o; float2* part_ml; int part_cap = 4096;
  size_t weight_bytes = 0;
  double last_step_s = 0;

  // The engine stream is nonblocking: default-stream initialization does not order its consumers.
  template <class T> T* dalloc(size_t n) {
    T* p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemsetAsync(p, 0, n * sizeof(T), st)); return p;
  }
  template <class T> T* upload(const void* src, size_t n) {
    T* p; CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemcpyAsync(p, src, n * sizeof(T), cudaMemcpyHostToDevice, st));
    CK(cudaStreamSynchronize(st));  // The caller may release its source storage on return.
    weight_bytes += n * sizeof(T); return p;
  }
  float* f32(const std::string& n) { auto& t = g.t(n); return upload<float>(t.data, t.bytes / 4); }
  float* f32_signed(const std::string& n) {   // D-wide norm weight times the 5120-wide Hadamard sign
    auto& t = g.t(n);
    if (binary) return f32(n);
    if (t.bytes != (size_t)D * 4 || hsigns5120.size() != (size_t)D) throw std::runtime_error("signed norm weight requires a 5120-wide norm and signs");
    std::vector<float> v(D); const float* h = (const float*)t.data;
    for (int i = 0; i < D; i++) v[i] = h[i] * hsigns5120[i];
    return upload<float>(v.data(), D);
  }

  bool use_packed = false, binary = false;
  enum class PackedActivation { F32, A8, F32R1A8 };
  PackedActivation packed_activation = PackedActivation::F32;
  bool a8_prepared = false, a8_gu_fused = false, a8_composed = false;
  enum class A8Projection { Dp4a, Fp4DigitsR32, Fp4DigitsRagged, Fp4ThreeDigitsRagged };
  A8Projection a8_projection = A8Projection::Dp4a;
  a8_fp4::Digits a8_digits{};
  signed_fp4::Digits signed_a8_digits{};
  bool whole_ffn_enabled = false;
  whole_ffn::Scratch whole_scratch{};
  size_t whole_weight_bytes = 0;
  size_t packed_partial_capacity = 0;
  int8_t* a8_gu_q = nullptr;
  float *a8_gu_scales = nullptr, *a8_gu_partials = nullptr;
  int32_t* a8_gu_dc = nullptr;
  bitserial::InputStatus* a8_gu_status = nullptr;
  int32_t* a8_dc = nullptr;
  bitserial::InputStatus *a8_status = nullptr, *a8_error = nullptr;
  bitserial::InputStatus a8_error_host = bitserial::InputStatus::Valid;
  bool binary1 = getenv("H5_COMPACT") && std::string(getenv("H5_COMPACT")) == "binary1";
  int fp4_passes = 1;
  bool joint_fp4_r16 = false;
  enum class NativeSf { Off, SF1R32, SF2R32 };
  NativeSf native_sf_mode = NativeSf::Off;
  float* native_sf_partials = nullptr; size_t native_sf_capacity = 0;
  float* pair_partials = nullptr; size_t pair_capacity = 0;
  float *xf = nullptr, *packed_tables = nullptr, *packed_partials = nullptr;
  uint8_t *q4_res = nullptr, *s4_res = nullptr; float* g4_res = nullptr;
  bool use_int8 = false, use_fp4 = true; bool wscale_e4m3 = getenv("H5_WSCALE") && std::string(getenv("H5_WSCALE")) == "e4m3";   // weight images to build; act_fp4 picks the path at run time
  bool compact = getenv("H5_COMPACT") && std::string(getenv("H5_COMPACT")) == "dense5";
  bool scale_carrier = false, carrier_wf0 = false;
  TernaryOperands ternary_operands = TernaryOperands::Packed;
  F02Mode f02_mode = F02Mode::Ordinary;
  F07Mode f07_mode = F07Mode::Ordinary;
  long f02_steps = 0, f02_rows = 0;
  enum class TargetProjection { Block, VocabularyHead };
  bool act_fp4 = true, force_fp4 = false; bool ab_bf16 = getenv("H5_AB_BF16") != nullptr;
  // interleave: (fp4 image only) group g of tensor i goes to group g * names.size() + i, so gate and up
  // of the same 32 features sit in adjacent groups for the fused silu(gate)*up epilogue.
  TMat tmat(const std::vector<std::string>& names, int K, TargetProjection role, bool interleave = false) {
    std::vector<const GTensor*> ts; for (auto& n : names) ts.push_back(&g.t(n));
    return tmat(ts, K, role, interleave);
  }
  TMat binary_tmat(const std::vector<const GTensor*>& ts, int K, bool interleave) {
    interleave = interleave && fp4_passes == 1 && !use_packed;
    int N = 0; bool aligned = true;
    std::vector<GTensor> tensors;
    for (auto* tp : ts) {
      if (tp->type != 41 || tp->ne.size() != 2 || (int)tp->ne[0] != K) throw std::runtime_error("invalid binary projection: " + tp->name);
      N += (int)tp->ne[1]; aligned &= tp->ne[1] % 32 == 0; tensors.push_back(*tp);
    }
    if (N <= 0 || N % 32 || K <= 0 || K % 128) throw std::runtime_error("binary projection requires N32/K128");
    if (interleave && (!aligned || ts.size() != 2 || ts[0]->ne[1] != ts[1]->ne[1])) throw std::runtime_error("invalid binary interleave");
    std::vector<uint8_t> joined;
    if (!aligned || (use_packed && tensors.size() > 1)) {
      size_t bytes = 0; for (auto& t : tensors) bytes += t.bytes;
      joined.resize(bytes); size_t off = 0;
      for (auto& t : tensors) { memcpy(joined.data() + off, t.data, t.bytes); off += t.bytes; }
      GTensor t = tensors.front(); t.ne[1] = N; t.bytes = bytes; t.data = joined.data(); tensors = {t};
    }
    TMat m; m.N = N; m.K = K; m.il = interleave;
    if (use_packed) {
      m.w1 = dalloc<uint32_t>(packed1::sign_bytes(N, K) / 4); m.s1 = dalloc<__half>(packed1::scale_bytes(N, K) / 2);
      weight_bytes += packed1::sign_bytes(N, K) + packed1::scale_bytes(N, K);
      uint8_t* raw = dalloc<uint8_t>(tensors.front().bytes);
      CK(cudaMemcpyAsync(raw, tensors.front().data, tensors.front().bytes, cudaMemcpyHostToDevice, st));
      CK(packed1::prepare(raw, m.w1, m.s1, N, K, st));
      if (whole_ffn_enabled && N == GU && K == D && ts.size() == 2 &&
          ts[0]->ne[1] == FF && ts[1]->ne[1] == FF) {
        m.whole_gu = dalloc<uint32_t>(packed1::sign_bytes(N, K) / 4);
        m.whole_gu_scales = dalloc<__half>(packed1::scale_bytes(N, K) / 2);
        const size_t bytes = packed1::sign_bytes(N, K) + packed1::scale_bytes(N, K);
        weight_bytes += bytes; whole_weight_bytes += bytes;
        CK(whole_ffn::prepare(raw, FF, K, m.whole_gu, m.whole_gu_scales, st));
      }
      CK(cudaStreamSynchronize(st)); CK(cudaFree(raw));
    }
    if (use_int8) {
      std::vector<uint8_t> W((size_t)N * K / 4); std::vector<float> S((size_t)N * (K / 128)); int row0 = 0;
      for (auto& t : tensors) { repack_rows(t, K, (int)t.ne[1], row0, W, S, N, nthreads); row0 += (int)t.ne[1]; }
      m.w = upload<uint4>(W.data(), W.size() / 16); m.s = upload<float>(S.data(), S.size());
    }
    if (use_fp4) {
      const size_t chunks = (size_t)(N / 32) * (K / 128), bytes = chunks * (binary1 ? 512 : 1024);
      m.w4 = dalloc<uint4>(bytes / 16); m.s4 = dalloc<__half>(chunks * 32); m.wf = binary1 ? 8 : 0;
      weight_bytes += bytes + chunks * 64;
      int row0 = 0, idx = 0;
      for (auto& t : tensors) {
        uint8_t* raw = dalloc<uint8_t>(t.bytes); CK(cudaMemcpyAsync(raw, t.data, t.bytes, cudaMemcpyHostToDevice, st));
        CK(launch_repack_q1_fp4(raw, K, (int)t.ne[1], (uint8_t*)m.w4, (uint16_t*)m.s4, N / 32, interleave ? (int)ts.size() : 1, interleave ? idx : row0 / 32, binary1, st));
        CK(cudaStreamSynchronize(st)); CK(cudaFree(raw)); row0 += (int)t.ne[1]; ++idx;
      }
      if (native_sf_mode != NativeSf::Off && N >= D) {
        const size_t bytes = binary_native_sf::scale_bytes(N, K, native_sf_planes());
        m.native_sf = dalloc<uint8_t>(bytes); m.native_row_factor = dalloc<float>(N);
        weight_bytes += bytes + (size_t)N * sizeof(float);
        binary_native_sf::ScaleStatus* status = nullptr;
        CK(cudaMalloc((void**)&status, (size_t)N * sizeof(*status)));
        std::unique_ptr<binary_native_sf::ScaleStatus, decltype(&cudaFree)> guard(status, &cudaFree);
        std::vector<binary_native_sf::ScaleStatus> values(N);
        CK(binary_native_sf::prepare((const uint16_t*)m.s4, m.native_sf, m.native_row_factor, status, N, K, native_sf_planes(), st));
        CK(cudaMemcpyAsync(values.data(), status, (size_t)N * sizeof(*status), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int row = 0; row < N; ++row) {
          switch (values[row]) {
            case binary_native_sf::ScaleStatus::Valid: break;
            case binary_native_sf::ScaleStatus::NonFinite:
              throw std::runtime_error("NativeSfScaleError: nonfinite original scale at row " + std::to_string(row));
            default: throw std::runtime_error("NativeSfScaleError: invalid preparation status");
          }
        }
      }
    }
    return m;
  }
  TMat tmat(const std::vector<const GTensor*>& ts, int K, TargetProjection role, bool interleave = false) {
    if (binary) return binary_tmat(ts, K, interleave);
    int N = 0; for (auto* t : ts) N += (int)t->ne[1];
    const bool wf0 = carrier_wf0 && role == TargetProjection::Block;
    const bool compact_weights = compact && !wf0;
    TMat m; m.N = N; m.K = K;
    if (scale_carrier && (N <= 0 || N % 32 || K <= 0 || K % 128))
      throw std::runtime_error("scale carrier requires N multiple of 32 and K multiple of 128");
    for (auto* tp : ts) {
      auto& t = *tp;
      if ((int)t.ne[0] != K || t.type != 143) throw std::runtime_error("tmat " + t.name);
      if (scale_carrier && (t.ne[1] <= 0 || t.ne[1] % 32 || (interleave && t.ne[1] != ts.front()->ne[1])))
        throw std::runtime_error("scale carrier requires complete, equally interleaved FP4 row groups: " + t.name);
    }
    if (use_int8) {
      std::vector<uint8_t> W((size_t)N * K / 4); std::vector<float> S((size_t)N * (K / 128));
      int row0 = 0;
      for (auto* tp : ts) { auto& t = *tp; repack_rows(t, K, (int)t.ne[1], row0, W, S, N, nthreads); row0 += (int)t.ne[1]; }
      m.w = upload<uint4>(W.data(), W.size() / 16); m.s = upload<float>(S.data(), S.size());
    }
    if (use_fp4) {
      std::vector<uint8_t> W((size_t)(N / 32) * (K / 128) * (compact_weights ? 832 : 1024)); std::vector<uint16_t> S((size_t)N * (K / 128));
      int row0 = 0;
      int idx = 0;
      for (auto* tp : ts) {
        auto& t = *tp;
        if (interleave) repack_rows_fp4(t, K, (int)t.ne[1], row0, W, S, N / 32, nthreads, (int)ts.size(), idx, compact_weights);
        else repack_rows_fp4(t, K, (int)t.ne[1], row0, W, S, N / 32, nthreads, 1, -1, compact_weights);
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
        if (ternary_operands != TernaryOperands::Packed) {
          const bool expanded = ternary_operands == TernaryOperands::Expanded;
          const size_t chunks = (size_t)nkb * NG;
          std::vector<uint8_t> EW(expanded ? chunks * TERNARY_EXPANDED_CW : 0), ES(chunks * TERNARY_EXPANDED_CS);
          std::atomic<size_t> next{0};
          auto worker = [&] {
            for (;;) {
              const size_t c = next.fetch_add(1);
              if (c >= chunks) break;
              if (expanded) expand_ternary_chunk(W.data() + c * 1024, EW.data() + c * TERNARY_EXPANDED_CW);
              expand_ternary_scales(SB.data() + c * 64, ES.data() + c * TERNARY_EXPANDED_CS);
            }
          };
          std::vector<std::thread> th;
          for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
          for (auto& t : th) t.join();
          m.w4 = expanded ? upload<uint4>(EW.data(), EW.size() / 16) : upload<uint4>(W.data(), W.size() / 16);
          m.s4 = (__half*)upload<uint8_t>(ES.data(), ES.size());
          m.wf = expanded ? TERNARY_EXPANDED_WF : TERNARY_SCALE_WORDS_WF;
        } else {
          m.w4 = upload<uint4>(W.data(), W.size() / 16);
          m.s4 = (__half*)upload<uint8_t>(SB.data(), SB.size());
          m.wf = 2;
        }
        m.rs = upload<float>(rsv.data(), rsv.size());
      } else { m.w4 = upload<uint4>(W.data(), W.size() / 16); m.s4 = upload<__half>(S.data(), S.size()); m.wf = compact_weights ? 3 : 0; }
    }
#ifdef H5_PROFILE_SCALE_CARRIER
    if (scale_carrier) {
      if (!m.w4 || !m.s4 || m.rs || m.wf != (wf0 ? 0 : 3))
        throw std::runtime_error("scale carrier payload does not match selected WF0/dense5 FP16-scale projection");
      m.mean_scale = dalloc<float>(m.N);
      // upload() uses cudaMemcpy from pageable memory, which may return before its DMA lands, and
      // st is a non-blocking stream that does not order after the legacy stream: without this
      // barrier the carrier means are computed from partially copied scales in some processes.
      CK(cudaDeviceSynchronize());
      CK(scale_carrier_prepare(m.s4, m.mean_scale, nullptr, m.N, m.K, st));
      std::vector<float> prepared(m.N);
      CK(cudaMemcpyAsync(prepared.data(), m.mean_scale, (size_t)m.N * sizeof(float), cudaMemcpyDeviceToHost, st));
      CK(cudaStreamSynchronize(st));
      for (float v : prepared) if (!std::isfinite(v) || v < 0.f)
        throw std::runtime_error("nonfinite or negative load-time scale carrier mean");
      weight_bytes += (size_t)m.N * sizeof(float);
    }
#endif
    return m;
  }

  int df_proposals = DRAFT_BASE_PROPOSALS, pending_rows = PMAX;
  Engine(const char* path, int slots, int ctx_, int rows, int proposals = DRAFT_BASE_PROPOSALS) : max_slots(slots), ctx(ctx_), max_rows(rows), df_proposals(proposals), pending_rows(proposals + 1) {
    if (const char* trace = getenv("H5_SAMPLE_TRACE_FILE")) {
      const char* limit = getenv("H5_SAMPLE_TRACE_LIMIT"), *stride = getenv("H5_SAMPLE_TRACE_STRIDE");
      sampling_trace_limit = limit ? parse_proposals(limit) : -1;
      sampling_trace_stride = stride ? parse_proposals(stride) : -1;
      if (!*trace || sampling_trace_limit < 1 || sampling_trace_limit > 512 || sampling_trace_stride < 1 || sampling_trace_stride > 1024)
        throw std::runtime_error("sampling trace requires explicit file, limit1..512 and stride1..1024");
      sampling_trace = fopen(trace, "a");
      if (!sampling_trace) throw std::runtime_error("cannot open sampling trace file");
      sampling_trace_left = sampling_trace_limit;
    }
    const char* drafter_kind = getenv("H5_DRAFTER");
    dspark_mode = drafter_kind && !strcmp(drafter_kind, "dspark-bf16");
    if (drafter_kind && !dspark_mode && strcmp(drafter_kind, "dflash2")) throw std::runtime_error("H5_DRAFTER must be dflash2 or dspark-bf16");
    if (dspark_mode && (proposals < 1 || proposals > 4 || slots < 1 || (long long)slots * 4 > rows)) throw std::runtime_error("DSpark requires proposals1..4 and four query rows per slot");
    if (proposals < 1 || proposals > DRAFT_MAX_PROPOSALS) throw std::runtime_error("unsupported draft proposal count");
    if (!dspark_mode && proposals != 7 && slots != 1) throw std::runtime_error("long DFlash draft requires one slot");
    if (proposals != 7 && (rows < proposals + 1 || ctx_ <= proposals + 2)) throw std::runtime_error("draft capacity too small");
    if (df_window <= 0 || (long long)df_window + proposals + 1 + 8 > DF_RING) throw std::runtime_error("H5_DF_WINDOW exceeds ring capacity");
    if (const char* margin = getenv("H5_DRAFT_MARGIN")) if (!parse_draft_margin(margin, draft_margin)) throw std::runtime_error("H5_DRAFT_MARGIN must be a finite nonnegative number");
    // H5_RELAX_STATS=1|2: the relaxed argmax also reports per-row greedy/best/proposed logits (2: plus log-sum-exp
    // and entropy) for a host-side acceptance policy (relax_policy.h). Requires H5_DRAFT_MARGIN > 0.
    if (const char* rs = getenv("H5_RELAX_STATS")) {
      if (!strcmp(rs, "0")) relax_stats = 0; else if (!strcmp(rs, "1")) relax_stats = 1; else if (!strcmp(rs, "2")) relax_stats = 2;
      else throw std::runtime_error("H5_RELAX_STATS must be 0, 1 or 2");
      if (relax_stats && !(draft_margin > 0.f)) throw std::runtime_error("H5_RELAX_STATS requires H5_DRAFT_MARGIN > 0");
    }
    // H5_LOGIT_ADJ=K: up to K host-supplied (token, delta) logit adjustments per row (h5_set_logit_adj), applied
    // before the argmax of every head row. 0 = off.
    if (const char* la = getenv("H5_LOGIT_ADJ")) {
      adj_k = parse_proposals(la);
      if (adj_k < 0 || adj_k > 8192) throw std::runtime_error("H5_LOGIT_ADJ must be 0..8192");
    }
    // H5_DRAFT_LZ=WINDOW:BUFFER:paper|dist: the drafter's selector adds the LZ penalty (set by h5-server from H5_LZ
    // when H5_LZ_DRAFT=on); per-step contexts and strengths come from h5_set_draft_lz.
    if (const char* dl = getenv("H5_DRAFT_LZ")) {
      char form[8] = {0}; char tail = 0;
      if (std::sscanf(dl, "%d:%d:%7[a-z]%c", &lzd_w, &lzd_b, form, &tail) != 3 || lzd_w < 16 || lzd_w > DRAFT_LZ_MAX_WINDOW ||
          lzd_b < 1 || lzd_b >= lzd_w || (strcmp(form, "paper") && strcmp(form, "dist")))
        throw std::runtime_error("H5_DRAFT_LZ must be WINDOW(16..1024):BUFFER:paper|dist");
      lzd_form = strcmp(form, "dist") ? 0 : 1;
    }
    // H5_DRAFT_NEST=K: the drafter's first K block rows (anchor + K-1 proposals) attend only to each other and the
    // context, so proposals 1..K-1 are those of a K-row block and the rows after them are extras.
    if (const char* nest = getenv("H5_DRAFT_NEST")) {
      draft_nest = parse_proposals(nest);
      if (draft_nest != 0 && (draft_nest < 2 || draft_nest >= proposals + 1)) throw std::runtime_error("H5_DRAFT_NEST must be 0 or 2..proposals");
    }
    auto experiment = [](const char* name) {
      const char* v = getenv(name);
      if (!v || std::string(v) == "0") return false;
      if (std::string(v) == "1") return true;
      throw std::runtime_error(std::string(name) + " must be 0 or 1");
    };
    if (const char* nodes = getenv("H5_GRAPH_NODES")) {
      if (std::string(nodes) == "native") graph_nodes = H5_GRAPH_NODES_NATIVE;
      else if (std::string(nodes) == "kernel-memory") graph_nodes = H5_GRAPH_NODES_KERNEL_MEMORY;
      else throw std::runtime_error("H5_GRAPH_NODES must be native or kernel-memory");
    }
    state_fusion = experiment("H5_STATE_FUSION"); attn_direct = experiment("H5_ATTN_DIRECT");
    conv_banks = state_fusion ? 2 : 1;
    if (state_fusion && getenv("H5_MEGA") && atoi(getenv("H5_MEGA")))
      throw std::runtime_error("H5_STATE_FUSION requires the normal graph path, not H5_MEGA");
    nthreads = getenv("H5_THREADS") ? std::max(1, std::min(4, atoi(getenv("H5_THREADS")))) : 4;
    if (const char* f = getenv("H5_LOAD_THREADS")) nthreads = std::max(1, std::min(nthreads, atoi(f)));
    if (compact) {
      if (wscale_e4m3) throw std::runtime_error("H5_COMPACT=dense5 requires original FP16 W128 scales; unset H5_WSCALE=e4m3");
      if (getenv("H5_MEGA") && atoi(getenv("H5_MEGA"))) throw std::runtime_error("H5_COMPACT=dense5 is not supported by the megakernel");
    }
    if (const char* value = getenv("H5_COMPACT")) if (strcmp(value, "dense5") && strcmp(value, "binary1")) throw std::runtime_error("H5_COMPACT must be dense5 or binary1");
    if (const char* f = getenv("H5_WEIGHTS")) {
      std::string v = f;
      if (v != "int8" && v != "fp4" && v != "both" && v != "packed1") throw std::runtime_error("H5_WEIGHTS must be int8, fp4, both or packed1");
      use_int8 = v == "int8" || v == "both"; use_fp4 = v == "fp4" || v == "both"; use_packed = v == "packed1";
    }
    act_fp4 = use_fp4;
    if (const char* f = getenv("H5_ACT")) {
      std::string v = f;
      const bool packed_act = v == "f32" || v == "a8" || v == "f32-r1-a8";
      if (!packed_act && v != "fp4" && v != "int8") throw std::runtime_error("H5_ACT must be fp4, int8, f32, a8 or f32-r1-a8");
      if (packed_act != use_packed) throw std::runtime_error("H5_ACT=f32|a8|f32-r1-a8 requires H5_WEIGHTS=packed1 exclusively");
      packed_activation = v == "a8" ? PackedActivation::A8 : v == "f32-r1-a8" ? PackedActivation::F32R1A8 : PackedActivation::F32;
      act_fp4 = v == "fp4";
    }
    if (const char* value = getenv("H5_A8_GU")) {
      if (!strcmp(value, "fused")) a8_gu_fused = true;
      else if (strcmp(value, "off")) throw std::runtime_error("H5_A8_GU must be off or fused");
    }
    if (a8_gu_fused && (!use_packed || packed_activation == PackedActivation::F32)) throw std::runtime_error("H5_A8_GU=fused requires packed1 a8 or f32-r1-a8 activations");
    if (const char* value = getenv("H5_A8_PROJECT")) {
      if (!strcmp(value, "fp4-digits")) a8_projection = A8Projection::Fp4DigitsR32;
      else if (!strcmp(value, "fp4-digits-ragged")) a8_projection = A8Projection::Fp4DigitsRagged;
      else if (!strcmp(value, "fp4-three-digits-ragged")) a8_projection = A8Projection::Fp4ThreeDigitsRagged;
      else if (strcmp(value, "dp4a")) throw std::runtime_error("H5_A8_PROJECT must be dp4a, fp4-digits, fp4-digits-ragged or fp4-three-digits-ragged");
    }
    if (digits_enabled() && (!use_packed || packed_activation == PackedActivation::F32))
      throw std::runtime_error("FP4 digit projection requires packed1 a8 or f32-r1-a8 activations");
    if (const char* value = getenv("H5_A8_FFN")) {
      if (!strcmp(value, "three-direct-r16")) whole_ffn_enabled = true;
      else if (strcmp(value, "off")) throw std::runtime_error("H5_A8_FFN must be off or three-direct-r16");
    }
    if (whole_ffn_enabled && (!use_packed || packed_activation == PackedActivation::F32 || a8_gu_fused))
      throw std::runtime_error("whole FFN requires packed1 a8 or hybrid with H5_A8_GU=off");
    if (act_fp4 && !use_fp4) throw std::runtime_error("H5_ACT=fp4 needs fp4 weights");
    if (!act_fp4 && !use_int8 && !use_packed) throw std::runtime_error("int8 activations need int8 weights");
    if (const char* v = getenv("H5_FP4_PASSES")) {
      if (!strcmp(v, "2")) fp4_passes = 2;
      else if (strcmp(v, "1")) throw std::runtime_error("H5_FP4_PASSES must be 1 or 2");
    }
    if (fp4_passes == 2 && !act_fp4) throw std::runtime_error("two FP4 passes require FP4 activations");
    if (!parse_ternary_operands(getenv("H5_TERNARY_OPERANDS"), ternary_operands))
      throw std::runtime_error("H5_TERNARY_OPERANDS must be packed, expanded or scale-words");
    if (ternary_operands != TernaryOperands::Packed && (!act_fp4 || !use_fp4 || !wscale_e4m3 || compact ||
        (getenv("H5_MEGA") && atoi(getenv("H5_MEGA")))))
      throw std::runtime_error("prepared ternary operands require FP4 target activations, H5_WSCALE=e4m3, no dense5, normal graph path");
    scale_carrier = experiment("H5_SCALE_CARRIER");
    carrier_wf0 = experiment("H5_CARRIER_WF0");
    if (carrier_wf0 && !scale_carrier)
      throw std::runtime_error("H5_CARRIER_WF0 requires H5_SCALE_CARRIER=1");
#ifndef H5_PROFILE_SCALE_CARRIER
    if (scale_carrier) throw std::runtime_error("H5_SCALE_CARRIER requested in a build without scale-carrier kernels");
    if (fuse_gu_selected() || env_mb("H5_PF_OUT_MB") > 0 || env_mb("H5_PF_GU_MB") > 0)
      throw std::runtime_error("GU fusion and chained prefetch require a scale-carrier build");
#endif
    if (scale_carrier && (!compact || !act_fp4 || !use_fp4 || wscale_e4m3 || (getenv("H5_MEGA") && atoi(getenv("H5_MEGA")))))
      throw std::runtime_error("H5_SCALE_CARRIER requires dense5 FP4 target activations and FP16 scales on normal graph path");
    if (!parse_f02_mode(getenv("H5_F02"), f02_mode)) throw std::runtime_error("H5_F02 must be 0 or 1");
    if (!parse_f07_mode(getenv("H5_F07"), f07_mode)) throw std::runtime_error("H5_F07 must be 0 or degree8");
    const bool mega_requested = getenv("H5_MEGA") && atoi(getenv("H5_MEGA"));
    const auto f02_error = validate_f02_config(f02_mode, act_fp4 && use_fp4, wscale_e4m3, compact,
      scale_carrier, mega_requested, !getenv("H5_NO_GUFUSE"), max_rows, f07_mode != F07Mode::Ordinary);
    if (f02_error != F02ConfigError::None) throw std::runtime_error(f02_config_error(f02_error));
    const auto f07_error = validate_f07_config(f07_mode, act_fp4 && use_fp4, compact, scale_carrier,
      wscale_e4m3, mega_requested, f02_mode != F02Mode::Ordinary, max_rows);
    if (f07_error != F07ConfigError::None) throw std::runtime_error(f07_config_error_name(f07_error));
    if (f07_mode != F07Mode::Ordinary && getenv("H5_NO_GUFUSE"))
      throw std::runtime_error("H5_F07=degree8 requires interleaved GU; unset H5_NO_GUFUSE");
    if (ternary_operands != TernaryOperands::Packed && f02_mode != F02Mode::Ordinary)
      throw std::runtime_error("prepared ternary operands do not support H5_F02");
    if (ternary_operands == TernaryOperands::Expanded)
      fprintf(stderr, "WF6 pre-expanded ternary MMA operands: identical WF2/E4M3 arithmetic; 2048-B weights + 128-B broadcast scales per chunk\n");
    if (ternary_operands == TernaryOperands::ScaleWords)
      fprintf(stderr, "WF7 packed ternary MMA operands: identical WF2/E4M3 arithmetic; 1024-B weights + 128-B broadcast scales per chunk\n");
    if (f02_mode == F02Mode::H32)
      fprintf(stderr, "F02 H32 enabled: exact R512 WF2/E4M3 interleaved GU; ordinary low rows, changed hidden quantization, current full-engine measurement pending\n");
    if (f07_mode == F07Mode::Degree8)
      fprintf(stderr, "F07 Degree8 enabled: exact R512 original WF0 FP16 scales, approximate half2 SiLU/product; changed live hidden codes/scale words, model quality/full-engine speed unmeasured\n");
    double t0 = now_s();
    g.open(path);
    const auto& emb = g.t("token_embd.weight");
    if (emb.type != 41 && emb.type != 143) throw std::runtime_error("unsupported target weight type");
    binary = emb.type == 41;
    if (use_packed && (!binary || compact || binary1)) throw std::runtime_error("packed1 requires Q1 weights and unset H5_COMPACT");
    if (binary1 && (!binary || !use_fp4)) throw std::runtime_error("binary1 requires Q1 FP4 target");
    if (fp4_passes == 2 && !binary) throw std::runtime_error("two-pass FP4 requires binary Q1 target");
    if (const char* value = getenv("H5_FP4_PAIR")) {
      if (!strcmp(value, "joint-r16")) joint_fp4_r16 = true;
      else if (strcmp(value, "off")) throw std::runtime_error("H5_FP4_PAIR must be off or joint-r16");
    }
    if (joint_fp4_r16 && (!binary || !binary1 || !act_fp4 || fp4_passes != 2))
      throw std::runtime_error("H5_FP4_PAIR=joint-r16 requires binary1 FP4 weights and two activation planes");
    if (joint_fp4_r16) fprintf(stderr, "FP4 pair: joint WF8 projection R1..16, ordered two-pass R>16; unchanged quantized planes, changed FP32 association\n");
    if (const char* value = getenv("H5_NATIVE_SF")) {
      if (!strcmp(value, "sf1-r32")) native_sf_mode = NativeSf::SF1R32;
      else if (!strcmp(value, "sf2-r32")) native_sf_mode = NativeSf::SF2R32;
      else if (strcmp(value, "off")) throw std::runtime_error("H5_NATIVE_SF must be off, sf1-r32 or sf2-r32");
    }
    if (native_sf_mode != NativeSf::Off) {
      if (!binary || !binary1 || !act_fp4 || fp4_passes != 2 || use_int8 || use_packed || wscale_e4m3 || scale_carrier || mega_requested)
        throw std::runtime_error("native SF requires Q1 binary1 FP4-only weights, two FP4 activation planes, original FP16 scale metadata and normal graph path");
      CK(binary_native_sf::configure(binary_native_sf::Schedule::F8G1T32, native_sf_planes()));
      if (native_sf_mode == NativeSf::SF2R32)
        CK(binary_native_sf::configure(binary_native_sf::Schedule::F4G2T16, native_sf_planes()));
      fprintf(stderr, "native SF%d: E4M3 weight scale planes + row factor, one WF8 sign payload; target N>=5120 at aligned R32 only; other rows/small matrices original FP16 reader, original scales retained\n", native_sf_planes());
    }
    if (binary && (compact || wscale_e4m3 || ab_bf16 || scale_carrier || ternary_operands != TernaryOperands::Packed || f02_mode != F02Mode::Ordinary || f07_mode != F07Mode::Ordinary || mega_requested))
      throw std::runtime_error("binary target requires original FP16 K128 scales, native Q1 alpha/beta, ordinary projection path");
    if (binary) {
      for (const char* key : {"H5_CONV_AB", "H5_AB_PARALLEL", "H5_SK_AB", "H5_TAIL_PREP", "H5_DOWN_CLEAR", "H5_PREP_CLEAR", "H5_GEMM_FLAG", "H5_PF_CHAIN_MB", "H5_L2PF_SIDE_MB", "H5_FUSE_GU"})
        if (const char* value = getenv(key)) if (strcmp(value, "0")) throw std::runtime_error(std::string("binary target does not support ") + key);
      if (const char* value = getenv("H5_ORDERED_PREP")) if (strcmp(value, "off")) throw std::runtime_error("binary target requires H5_ORDERED_PREP=off");
      if (env_mb("H5_PF_OUT_MB") > 0 || env_mb("H5_PF_GU_MB") > 0)
        throw std::runtime_error("binary target does not support chained packed-carrier prefetch");
    }
    auto require_int = [&](const char* key, int expected) {
      auto it = g.kv.find(key);
      if (it == g.kv.end() || it->second.ints.size() != 1 || it->second.ints[0] != expected) throw std::runtime_error(std::string("unsupported model geometry: ") + key);
    };
    if (g.kv.at("general.architecture").strs.at(0) != "qwen35") throw std::runtime_error("expected qwen35 model");
    require_int("qwen35.block_count", NLAYER); require_int("qwen35.embedding_length", D); require_int("qwen35.feed_forward_length", FF);
    require_int("qwen35.attention.head_count", NH); require_int("qwen35.attention.head_count_kv", NKV);
    require_int("qwen35.attention.key_length", HD); require_int("qwen35.rope.dimension_count", NROT);
    require_int("qwen35.ssm.group_count", HK); require_int("qwen35.ssm.time_step_rank", HV);
    require_int("qwen35.ssm.state_size", SS); require_int("qwen35.ssm.inner_size", VDIM); require_int("qwen35.full_attention_interval", 4);
    if (emb.ne != std::vector<uint64_t>{D, VOCAB}) throw std::runtime_error("unsupported embedding shape");
    fprintf(stderr, "target: %s, %s K128 scales\n", binary ? "binary Q1_0" : "ternary PTQ1_0", wscale_e4m3 ? "row-relative E4M3" : "original FP16");
    auto rsec = g.kv.find("qwen35.rope.dimension_sections");
    if (rsec != g.kv.end() && rsec->second.ints.size() == 4)
      for (int i = 0; i < 4; i++) rope_sections[i] = rsec->second.ints[i];
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    prof_stream = st;
    CK(cudaStreamCreateWithFlags(&ab_st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ab_fork, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&ab_join, cudaEventDisableTiming));
    if (const char* pw = getenv("H5_GEMM_PREWAIT")) {
      char* end = nullptr; const long cap = strtol(pw, &end, 10);
      if (!end || *end || end == pw || cap < 0 || cap > 64) throw std::runtime_error("H5_GEMM_PREWAIT must be an integer 0..64");
      h5_set_prewait_cap((int)cap);
      fprintf(stderr, "GEMM pre-wait ring stages capped at %ld\n", cap);
    }
    if (const char* lp = getenv("H5_GEMM_L2PF")) {
      char* end = nullptr; const long n = strtol(lp, &end, 10);
      if (!end || *end || end == lp || n < 0 || n > 64) throw std::runtime_error("H5_GEMM_L2PF must be an integer 0..64");
      h5_set_l2pf_stages((int)n);
      fprintf(stderr, "GEMM pre-wait L2 prefetch: %ld K-blocks after the ring\n", n);
    }
    if (const char* pf = getenv("H5_L2PF_SIDE_MB")) {
      char* end = nullptr; l2pf_side_bytes = strtod(pf, &end) * 1048576.0;
      if (!end || *end || !(l2pf_side_bytes > 0)) throw std::runtime_error("H5_L2PF_SIDE_MB must be a positive number");
      CK(cudaStreamCreateWithFlags(&pf_st, cudaStreamNonBlocking));
      CK(cudaEventCreateWithFlags(&pf_fork, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&pf_join, cudaEventDisableTiming));
      fprintf(stderr, "side L2 weight prefetch: %.1f MiB per GEMM\n", l2pf_side_bytes / 1048576.0);
    }
    if (!binary) {
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
    head = tmat({"output.weight"}, D, TargetProjection::VocabularyHead);
    L.resize(NLAYER);
    int gl = 0, al = 0;
    for (int l = 0; l < NLAYER; l++) {
      Layer& y = L[l]; std::string p = "blk." + std::to_string(l) + ".";
      y.attn = is_attn(l);
      y.attn_norm = f32(p + "attn_norm.weight"); y.post_norm = f32(p + "post_attention_norm.weight");
      y.attn_norm_s = f32_signed(p + "attn_norm.weight"); y.post_norm_s = f32_signed(p + "post_attention_norm.weight");
      if (y.attn) {
        y.al = al++;
        y.in = tmat({p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, D, TargetProjection::Block);
        y.out = tmat({p + "attn_output.weight"}, ATTN_OUT, TargetProjection::Block);
        y.q_norm = f32(p + "attn_q_norm.weight"); y.k_norm = f32(p + "attn_k_norm.weight");
      } else {
        y.gl = gl++;
        y.in = tmat({p + "attn_qkv.weight", p + "attn_gate.weight"}, D, TargetProjection::Block);
        y.out = tmat({p + "ssm_out.weight"}, VDIM, TargetProjection::Block);
        y.conv_w = f32(p + "ssm_conv1d.weight"); y.ssm_a = f32(p + "ssm_a"); y.ssm_dt = f32(p + "ssm_dt.bias"); y.ssm_norm = f32(p + "ssm_norm.weight");
        if (binary) y.ab4 = tmat({p + "ssm_alpha.weight", p + "ssm_beta.weight"}, D, TargetProjection::Block);
        else {
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
      }
      y.gu = tmat({p + "ffn_gate.weight", p + "ffn_up.weight"}, D, TargetProjection::Block, !getenv("H5_NO_GUFUSE"));
      y.down = tmat({p + "ffn_down.weight"}, FF, TargetProjection::Block);
    }
    if (joint_fp4_r16) {
      auto reserve = [&](const TMat& m) {
        for (int r = 1; r <= std::min(16, max_rows); ++r) pair_capacity = std::max(pair_capacity, binary_fp4_pair::partial_floats(pair_plan(m, r)));
      };
      for (auto& ly : L) for (const TMat* m : {&ly.in, &ly.out, &ly.gu, &ly.down, &ly.ab4}) if (m->w4) reserve(*m);
      reserve(head);
      if (pair_capacity) pair_partials = dalloc<float>(pair_capacity);
    }
    if (native_sf_mode != NativeSf::Off) {
      auto reserve = [&](const TMat& m) {
        if (m.native_sf) native_sf_capacity = std::max(native_sf_capacity, binary_native_sf::partial_floats(native_sf_plan(m)));
      };
      for (auto& ly : L) for (const TMat* m : {&ly.in, &ly.out, &ly.gu, &ly.down, &ly.ab4}) reserve(*m);
      reserve(head);
      if (native_sf_capacity) native_sf_partials = dalloc<float>(native_sf_capacity);
    }
    fprintf(stderr, "\nweights %.2f GB in %.1f s\n", weight_bytes / 1e9, now_s() - t0);
    // state
    if (const char* f = getenv("H5_STATE")) {
      if (std::string(f) == "bf16") state_format = StateFormat::BF16;
      else if (std::string(f) == "f16" || std::string(f) == "fp16") state_format = StateFormat::F16;
      else if (std::string(f) == "f32") state_format = StateFormat::F32;
      else if (std::string(f) == "q8") state_format = StateFormat::Q8;
      else throw std::runtime_error("H5_STATE must be f16, bf16, f32 or q8");
    }
    if (const char* f = getenv("H5_KV")) kv8 = std::string(f) == "fp8";
    ctx = (ctx + 31) / 32 * 32;   // fp8 V is stored in 32-key transposed blocks
    state_es = pending_state_element_bytes(state_format);
    state_slot_bytes = gdn_state_slot_bytes(state_format);
    gdn_state = dalloc<uint8_t>((size_t)max_slots * state_slot_bytes);
    conv_state = dalloc<float>((size_t)max_slots * NGDN * conv_banks * 3 * CONV_CH);
    const size_t kvn = (size_t)max_slots * NATTN * NKV * ctx;
    kc = dalloc<uint8_t>(kvn * HD * (kv8 ? 1 : 2)); vc = dalloc<uint8_t>(kvn * HD * (kv8 ? 1 : 2));
    if (kv8) { ksc = dalloc<float>(kvn); vsc = dalloc<float>(kvn); }
    fprintf(stderr, "state %s, KV %s: %.1f MB state + %.1f MB KV per slot\n", state_format_name(state_format), kv8 ? "fp8" : "f16",
            state_slot_bytes / 1e6, (double)NATTN * NKV * ctx * (HD * (kv8 ? 2 : 4) + (kv8 ? 8 : 0)) / 1e6);
    if (state_format == StateFormat::Q8 || getenv("H5_GDN_Q8_SCALE"))
      fprintf(stderr, "Q8 row scale: %s (fixed per engine)\n", q8_scale_name(q8_scale));
    // activations
    const size_t R = max_rows;
    x = dalloc<float>(R * D); tmp = dalloc<float>(R * D); xn = dalloc<float>(R * D);
    proj = dalloc<float>(R * GU); gu_acc = dalloc<float>((size_t)8 * GU);
    sk_ws = dalloc<float>(sk_workspace_floats(VOCAB)); sk_cnt = dalloc<int>(sk_counter_ints(VOCAB)); sk_next = dalloc<int>(SK_SITES); conv_out = dalloc<float>(R * CONV_CH); ab = dalloc<float>(R * 96);
    y = dalloc<float>(R * VDIM); qrot = dalloc<float>(R * NH * HD); logits = dalloc<float>((size_t)128 * VOCAB);
    xq = dalloc<int8_t>(R * FF); xs = dalloc<float>(R * (FF / 128));
    if (use_packed || fp4_passes == 2) xf = dalloc<float>(R * FF);
    if (use_packed) {
      if (packed_activation != PackedActivation::F32) {
        a8_dc = dalloc<int32_t>(R * (FF / 128)); a8_status = dalloc<bitserial::InputStatus>(R * (FF / 128)); a8_error = dalloc<bitserial::InputStatus>(1);
        if (a8_gu_fused) {
          a8_gu_q = dalloc<int8_t>(R * FF); a8_gu_scales = dalloc<float>(R * (FF / 128));
          a8_gu_dc = dalloc<int32_t>(R * (FF / 128)); a8_gu_status = dalloc<bitserial::InputStatus>(R * (FF / 128));
          size_t bytes = 0;
          for (int rows = 1; rows <= 16; ++rows) bytes = std::max(bytes, composition::partial_bytes(composition_plan(rows)));
          a8_gu_partials = dalloc<float>(bytes / sizeof(float));
          fprintf(stderr, "A8 GU: fused continuation to distinct next carrier; arbitrary rows tiled1..16, F32 routes unchanged\n");
        }
        if (whole_ffn_enabled) {
          CK(whole_ffn::configure(signed_fp4::Schedule::F4G2T8));
          CK(whole_ffn::configure(signed_fp4::Schedule::F4G2T16));
          CK(signed_fp4::configure(signed_fp4::Schedule::F4G2T16));
          auto allocate_digits = [&](int k) {
            signed_fp4::Digits d{}; d.encoding = signed_fp4::Encoding::ExactA8;
            d.capacity = signed_fp4::digit_bytes(k, 32); d.data = dalloc<uint8_t>(d.capacity); return d;
          };
          whole_scratch.input_digits = allocate_digits(D);
          whole_scratch.next_digits = allocate_digits(FF);
          whole_scratch.next = {nullptr, nullptr, dalloc<float>(32 * (FF / 128)),
            dalloc<int32_t>(32 * (FF / 128)), dalloc<bitserial::InputStatus>(32 * (FF / 128)), a8_error};
          for (int rows = 16; rows <= 32; ++rows)
            whole_scratch.down_partial_capacity = std::max(whole_scratch.down_partial_capacity,
              signed_fp4::partial_floats(whole_down_plan(rows)));
          whole_scratch.down_partials = dalloc<float>(whole_scratch.down_partial_capacity);
          fprintf(stderr, "whole FFN: exact3 direct GU/SFU/A8/down preparedR16..32; other rows original; added GU residency=%zu bytes, no global GU/hidden/q tensor on compound path\n", whole_weight_bytes);
        }
        fprintf(stderr, "packed target activation: %s; A8 uses K128 max/127 nearest-even codes, original FP16 weight scales\n", packed_activation == PackedActivation::A8 ? "a8" : "f32 for prepared R=1, a8 for prepared R>1");
        if (digits_enabled()) {
          if (three_digits_enabled()) {
            CK(signed_fp4::configure(signed_fp4::Schedule::F4G2T16));
            signed_a8_digits.encoding = signed_fp4::Encoding::ExactA8;
            signed_a8_digits.capacity = signed_fp4::digit_bytes(FF, 32);
            signed_a8_digits.data = dalloc<uint8_t>(signed_a8_digits.capacity);
          } else {
            CK(a8_fp4::configure(a8_fp4::Schedule::F4G2T16));
            a8_digits.capacity = a8_fp4::digit_bytes(FF, 32);
            a8_digits.data = dalloc<uint8_t>(a8_digits.capacity);
          }
          fprintf(stderr, "A8 projection: exact %d FP4 digit planes on original packed1 weights, GU/down/head tiles%s; other families and smaller tails DP4A, F32 unchanged\n",
            three_digits_enabled() ? 3 : 4, a8_projection == A8Projection::Fp4DigitsR32 ? "32" : "16..32");
        }
      }
      size_t tables = 0, partials = 0;
      auto reserve = [&](const TMat& m) {
        if (!m.w1) return;
        for (int r = 1; r <= 8; ++r) {
          packed1::Plan p = packed_plan(m, r);
          tables = std::max(tables, packed1::table_bytes(p)); partials = std::max(partials, packed1::partial_bytes(p));
        }
        if (packed_activation != PackedActivation::F32) for (int r = 1; r <= 16; ++r) {
          bitserial::Plan p; CK(bitserial::select(m.N, m.K, r, &p)); partials = std::max(partials, bitserial::partial_bytes(p));
        }
        if (digits_enabled() && digit_family(m))
          partials = std::max(partials, (three_digits_enabled()
            ? signed_fp4::partial_floats(three_digit_plan(m, 32))
            : a8_fp4::partial_floats(digit_plan(m, 32))) * sizeof(float));
      };
      for (auto& ly : L) for (const TMat* m : {&ly.in, &ly.out, &ly.gu, &ly.down, &ly.ab4}) reserve(*m);
      reserve(head);
      if (tables) packed_tables = dalloc<float>(tables / 4);
      packed_partial_capacity = partials / sizeof(float);
      if (partials) packed_partials = dalloc<float>(packed_partial_capacity);
    }
    gates = dalloc<float>(R * 96);
    scratch = dalloc<float>(std::max(R * (size_t)GU, (size_t)128 * VOCAB));
    const size_t Rp = (R + 127) / 128 * 128;
    q4 = dalloc<uint8_t>(Rp * FF / 2); s4 = dalloc<uint8_t>(Rp * FF / 16); g4 = dalloc<float>(Rp);
    if (fp4_passes == 2) { q4_res = dalloc<uint8_t>(Rp * FF / 2); s4_res = dalloc<uint8_t>(Rp * FF / 16); g4_res = dalloc<float>(Rp); }
    q4_base = q4; s4_base = s4; g4_base = g4;
    q4_alt = dalloc<uint8_t>((size_t)64 * FF / 2); s4_alt = dalloc<uint8_t>((size_t)64 * FF / 16); g4_alt = dalloc<float>(64);
    if (getenv("H5_TAIL_DBG")) tail_dbg = dalloc<unsigned long long>((size_t)8 * TAIL_MAX_CHUNKS * TAIL_DBG_STAMPS);
    tail_cnt = dalloc<unsigned>(TAIL_MAX_CHUNKS); flag_sig = dalloc<unsigned>((size_t)NLAYER * 8);
    argmax_keys = dalloc<unsigned long long>(128); argmax_cnt = dalloc<unsigned>(128); tail_rdv = dalloc<unsigned>(1); tail_part = dalloc<float>((size_t)TAIL_MAX_CHUNKS * TAIL_MAX_ROWS * 2);
    // Dedicated output-row RMS scratch; never aliases the concurrently written g4.
    prep_rms_inv = dalloc<float>(R);
    d_tok = dalloc<int>(R); d_slot = dalloc<int>(R); d_pos = dalloc<int>(R); d_out = dalloc<int>(R); d_segs = dalloc<Seg>(R);
    d_rope = dalloc<int>(4 * R); d_visual = dalloc<float>((size_t)R * D);
    d_head_rows = dalloc<int>(R); CK(cudaMallocHost(&h_head_rows, R * sizeof(int)));
    pend_gdn = dalloc<uint8_t>((size_t)max_slots * NGDN * 2 * pending_rows * PEND_GDN * state_es);
    pend_par.assign(max_slots, 0); conv_par.assign(max_slots, 0);
    pend_conv = dalloc<uint8_t>((size_t)max_slots * NGDN * conv_banks * pending_rows * CONV_CH * state_es);
    pend_n.assign(max_slots, 0); commit_next.assign(max_slots, 0); keep_slot.assign(max_slots, 0);
    hcap = dalloc<float>(R * NCAP * D);
    d_qb = dalloc<QBlk>(R); CK(cudaMallocHost(&h_qb, R * sizeof(QBlk)));
    part_o = dalloc<float>((size_t)part_cap * NH * HD); part_ml = dalloc<float2>((size_t)part_cap * std::max(NH, DF_NH));   // drafter has 32 heads
    CK(cudaMallocHost(&h_tok, R * 4)); CK(cudaMallocHost(&h_slot, R * 4)); CK(cudaMallocHost(&h_pos, R * 4)); CK(cudaMallocHost(&h_out, R * 4)); CK(cudaMallocHost(&h_segs, R * sizeof(Seg)));
    if (lzd_w) {
      d_lzctx = dalloc<int>((size_t)max_slots * lzd_w); CK(cudaMallocHost(&h_lzctx, (size_t)max_slots * lzd_w * sizeof(int)));
      d_lzlen = dalloc<int>(max_slots); CK(cudaMallocHost(&h_lzlen, max_slots * sizeof(int)));
      d_lzalpha = dalloc<float>(max_slots); CK(cudaMallocHost(&h_lzalpha, max_slots * sizeof(float)));
      for (int i = 0; i < max_slots; ++i) { h_lzlen[i] = 0; h_lzalpha[i] = 0.f; }
    }
    if (adj_k) {
      d_adj = dalloc<int2>((size_t)R * adj_k); CK(cudaMallocHost(&h_adj, (size_t)R * adj_k * sizeof(int2)));
      for (size_t i = 0; i < (size_t)R * adj_k; ++i) h_adj[i] = make_int2(-1, 0);
    }
    if (relax_stats) { d_rstat = dalloc<RowStat>(R); CK(cudaMallocHost(&h_rstat, R * sizeof(RowStat))); last_rstat.assign(R, RowStat{-1, 0.f, 0.f, 0.f, 0.f, -1, 0.f}); }
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

  packed1::Plan packed_plan(const TMat& m, int R) const {
    packed1::Plan p; CK(packed1::select(m.N, m.K, R, &p));
    auto integer = [](const char* key, int& dst) {
      if (const char* v = getenv(key)) {
        char* end = nullptr; long n = strtol(v, &end, 10);
        if (!*v || *end || n <= 0 || n > 65536) throw std::runtime_error(std::string("invalid packed1 integer: ") + key);
        dst = (int)n;
      }
    };
    integer("H5_PACKED_GROUPS", p.groups_per_split); integer("H5_PACKED_THREADS", p.threads);
    if (const char* v = getenv("H5_PACKED_ROUTE")) {
      if (!strcmp(v, "lut4")) p.route = packed1::Route::Lut4;
      else if (!strcmp(v, "sign")) p.route = packed1::Route::Sign;
      else if (!strcmp(v, "warp-sign")) p.route = packed1::Route::WarpSign;
      else throw std::runtime_error("H5_PACKED_ROUTE must be lut4, sign or warp-sign");
    }
    CK(packed1::validate(p)); return p;
  }
  bool uses_a8(int R) const { return use_packed && (packed_activation == PackedActivation::A8 || (packed_activation == PackedActivation::F32R1A8 && R > 1)); }
  bool digit_family(const TMat& m) const {
    return (m.N == GU && m.K == D) || (m.N == D && m.K == FF) || (&m == &head && m.N >= 65536 && m.K == D);
  }
  bool digits_enabled() const { return a8_projection != A8Projection::Dp4a; }
  bool three_digits_enabled() const { return a8_projection == A8Projection::Fp4ThreeDigitsRagged; }
  bool digit_rows_enabled(int rows) const {
    switch (a8_projection) {
      case A8Projection::Dp4a: return false;
      case A8Projection::Fp4DigitsR32: return rows >= 32;
      case A8Projection::Fp4DigitsRagged:
      case A8Projection::Fp4ThreeDigitsRagged: return rows >= 16;
    }
    throw std::runtime_error("invalid A8 projection route");
  }
  a8_fp4::Plan digit_plan(const TMat& m, int rows) const {
    if (!digit_family(m) || rows < 16 || rows > 32) throw std::runtime_error("FP4 digit projection outside explicit GU/down/head row scope");
    const bool down = m.N == D && m.K == FF;
    return {m.N, m.K, rows, down ? 8 : m.N == GU && rows == 32 ? 4 : 1,
      a8_fp4::Schedule::F4G2T16, a8_fp4::WeightLayout::Packed1};
  }
  signed_fp4::Plan three_digit_plan(const TMat& m, int rows) const {
    if (!three_digits_enabled() || !digit_family(m) || rows < 16 || rows > 32)
      throw std::runtime_error("three FP4 digit projection outside explicit GU/down/head row scope");
    const bool down = m.N == D && m.K == FF;
    return {m.N, m.K, rows, down ? 8 : m.N == GU && rows == 32 ? 4 : 1,
      signed_fp4::Schedule::F4G2T16, signed_fp4::WeightLayout::Packed1};
  }
  bitserial::Carrier composed_carrier(int row = 0) const {
    return {a8_gu_q + (size_t)row * FF, nullptr, a8_gu_scales + (size_t)row * (FF / 128), a8_gu_dc + (size_t)row * (FF / 128), a8_gu_status + (size_t)row * (FF / 128), a8_error};
  }
  bitserial::Carrier a8_carrier(int n, int row = 0) const {
    if (a8_composed) {
      if (n != FF) throw std::runtime_error("fused GU carrier must feed the down projection");
      return composed_carrier(row);
    }
    return bitserial::Carrier{xq + (size_t)row * n, nullptr, xs + (size_t)row * (n / 128), a8_dc + (size_t)row * (n / 128), a8_status + (size_t)row * (n / 128), a8_error};
  }
  static signed_fp4::Plan whole_down_plan(int rows) {
    return {D, FF, rows, rows == 16 ? 16 : 8, signed_fp4::Schedule::F4G2T16, signed_fp4::WeightLayout::Packed1};
  }
  bool whole_ffn_rows(int rows) const { return whole_ffn_enabled && a8_prepared && rows >= 16 && rows <= 32; }
  void compound_ffn(const Layer& layer, int rows) {
    if (!whole_ffn_rows(rows) || a8_composed || !layer.gu.whole_gu || !layer.gu.whole_gu_scales || !layer.down.w1 || !layer.down.s1)
      throw std::runtime_error("invalid whole FFN carrier or weight state");
    const whole_ffn::Plan plan{FF, D, rows, rows == 16 ? signed_fp4::Schedule::F4G2T8 : signed_fp4::Schedule::F4G2T16};
    CK(whole_ffn::project(plan, whole_down_plan(rows),
      {layer.gu.whole_gu, layer.gu.whole_gu_scales, layer.down.w1, layer.down.s1},
      a8_carrier(D), whole_scratch, x, D, 1.f, 1.f, st));
  }
  static composition::Plan composition_plan(int rows) {
    const int tile = rows % 8 == 0 ? 8 : rows % 4 == 0 ? 4 : 1;
    composition::Plan p{FF, D, rows, tile, rows <= 8 ? 4 : 16}; CK(composition::validate(p)); return p;
  }
  void compose_gu(const TMat& m, int R) {
    if (!a8_gu_fused || !a8_prepared || a8_composed || !m.w1 || !m.s1 || m.N != GU || m.K != D || R < 1 || R > max_rows)
      throw std::runtime_error("invalid A8 GU composition state");
    for (int row = 0; row < R; row += 16) {
      const auto p = composition_plan(std::min(16, R - row));
      CK(composition::fused(p, m.w1, m.s1, a8_carrier(D, row), composed_carrier(row), a8_gu_partials, st));
    }
    a8_composed = true;
  }
  void produce_a8(int n, int R) {
    a8_composed = false;
    for (int row = 0; row < R; row += 65535)
      CK(bitserial::produce(bitserial::Route::Dp4a01, xf + (size_t)row * n, n, std::min(65535, R - row), a8_carrier(n, row), st));
  }
  void clear_a8_error() {
    if (a8_error) CK(cudaMemsetAsync(a8_error, 0, sizeof(*a8_error), st));
  }
  void read_a8_error() {
    if (a8_error) CK(cudaMemcpyAsync(&a8_error_host, a8_error, sizeof(*a8_error), cudaMemcpyDeviceToHost, st));
  }
  void require_a8_valid() const {
    if (a8_error && a8_error_host != bitserial::InputStatus::Valid)
      throw std::runtime_error(a8_error_host == bitserial::InputStatus::NonFinite ? "A8 activation producer: nonfinite input" : "A8 activation producer: scale outside [1e-30,1e30]");
  }
  void finish_target_prep(int n, int R) {
    a8_composed = false;
    if (force_fp4) return;
    if (fp4_passes == 2) prep_residual(n, R);
    a8_prepared = uses_a8(R);
    if (a8_prepared) produce_a8(n, R);
  }
  void prep_residual(int n, int R) {
    PrepArgs a{}; a.x = xf; a.ldx = n; a.n = n; a.flags = P_QUANT | P_FP4_RESIDUAL;
    a.q4 = q4_res; a.s4 = s4_res; a.g4 = g4_res; a.ref_q4 = q4; a.ref_s4 = s4; a.ref_g4 = g4;
    launch_prep(a, R, st);
  }
  void target_prep(PrepArgs& a) {
    if (binary) a.flags &= ~(P_SIGN | P_HADAMARD | P_SIGN_AFTER | P_PERMUTE_GDN);
    if (!force_fp4 && (use_packed || fp4_passes == 2) && (a.flags & P_QUANT)) {
      if (use_packed) a.flags &= ~P_QUANT;
      a.flags |= P_STORE_F32; a.out_f32 = xf; a.ldo = a.n;
    }
  }
  void prep_dconv(const float* src, const float* dyn, int which, const float* base, int R, int block_rows, float* clear_a, int clear_a_stride) {
    PrepArgs a{}; a.x = src; a.ldx = D; a.n = D; a.flags = P_DCONV | P_QUANT;
    a.dc_dyn = dyn; a.dc_base = base; a.dc_which = which; a.dc_block_rows = block_rows;
    a.xq = xq; a.xs = xs; a.clear_a = clear_a; a.clear_a_stride = clear_a_stride;
    if (act_fp4 || force_fp4) { a.q4 = q4; a.s4 = s4; a.g4 = g4; }
    launch_prep(a, R, st);
  }
  void prep(const float* in, const float* in2, int ldx, const float* nw, const float* signs, int n, unsigned flags, int R, float* out_f32 = nullptr, float* out_norm = nullptr, int ldo = 0, const int* rowmap = nullptr, float* clear_a = nullptr, int clear_a_stride = 0, float* clear_b = nullptr, int clear_b_stride = 0, const unsigned* sig = nullptr, unsigned sig_expect = 0) {
    PrepArgs a{}; a.x = in; a.x2 = in2; a.ldx = ldx; a.norm_w = nw; a.signs = signs; a.n = n; a.flags = flags; a.rowmap = rowmap;
    a.sig = sig; a.sig_expect = sig_expect;
    a.done_sig = prep_done_sig_next; prep_done_sig_next = nullptr;
    a.early = prep_early_next; prep_early_next = false;
    a.xq = xq; a.xs = xs; a.out_f32 = out_f32; a.out_norm = out_norm; a.ldo = ldo;
    a.clear_a = clear_a; a.clear_a_stride = clear_a_stride; a.clear_b = clear_b; a.clear_b_stride = clear_b_stride;
    if (flags & P_NORM) { a.rms_inv = prep_rms_inv; a.rms_rows = max_rows; }
    if (act_fp4 || force_fp4) { a.q4 = q4; a.s4 = s4; a.g4 = g4; }
    target_prep(a);
    launch_prep(a, R, st);
    if (flags & P_QUANT) finish_target_prep(n, R);
  }
#ifdef H5_PROFILE_SCALE_CARRIER
  void launch_carrier_projection(const TMat& m, int ci, int ks, int mode, const uint8_t* A, const uint8_t* AS,
                                 const float* AG, int R, float* out, int ldo) {
    switch (m.wf) {
      case 0:
        if (!carrier_wf0) throw std::runtime_error("WF0 scale carrier payload was not admitted");
        CK(launch_scale_carrier_wf0(ci, ks, mode, A, AS, AG, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, st));
        return;
      case 3:
        pf_before_gemm(m.w4, m.N, m.K, 3, ks);
        CK(launch_scale_carrier(ci, ks, mode, A, AS, AG, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, st));
        pf_after_gemm();
        return;
      default:
        throw std::runtime_error("unsupported scale carrier physical payload");
    }
  }
#endif
  // H5_SK=SHAPE:GRID:ITEMS (read at capture): every FP4 GEMM of at most eight rows whose payload the stream-K
  // kernel reads (dense5 carrier, NVFP4 with row scales) runs as one CTA range per SM slot (src/sk.cu);
  // unset or "0" = the frozen-plan kernels. H5_SK_OVERRIDE=N:K:WF=SHAPE:GRID:ITEMS|off[,...] replaces the shape
  // for single projections. H5_GEMM_PREWAIT bounds the stages issued before the predecessor wait.
  // SHAPE:GRID:ITEMS: kernel shape index, CTAs per launch, target work items per CTA; "off" (overrides
  // only) keeps that projection on the frozen-plan kernels.
  struct SkSel { bool on; int shape, grid, items; };
  static bool sk_parse(const char* v, SkSel& s) {
    if (!strcmp(v, "off")) { s = {false, 0, 0, 0}; return true; }
    char tail = 0; s.on = true;
    if (sscanf(v, "%d:%d:%d%c", &s.shape, &s.grid, &s.items, &tail) != 3) return false;
    return s.shape >= 0 && s.shape < sk_nshapes() && s.grid >= 1 && s.grid <= 4096 && s.items >= 1 && s.items <= 64;
  }
  bool sk_select(const TMat& m, int R, SkSel& sel) const {
    const char* v = getenv("H5_SK");
    if (!v || !strcmp(v, "0") || R > 8) return false;
    if (!sk_parse(v, sel) || !sel.on) throw std::runtime_error("H5_SK must be 0 or SHAPE:GRID:ITEMS");
    // H5_SK covers the dense5 carrier payloads; NVFP4 (drafter) projections only through override entries.
    const bool carrier = m.mean_scale && m.wf == 3, nvfp4 = !m.mean_scale && m.wf == 1 && m.rs;
    if (!carrier && !nvfp4) return false;
    bool chosen = carrier;
    if (const char* ov = getenv("H5_SK_OVERRIDE")) {
      std::string all = ov; size_t p0 = 0;
      while (p0 < all.size()) {
        const size_t p1 = std::min(all.find(',', p0), all.size());
        const std::string item = all.substr(p0, p1 - p0);
        int n, k, w; SkSel s; char rest[32];
        if (sscanf(item.c_str(), "%d:%d:%d=%31s", &n, &k, &w, rest) != 4 || !sk_parse(rest, s))
          throw std::runtime_error("H5_SK_OVERRIDE items must be N:K:WF=SHAPE:GRID:ITEMS or N:K:WF=off");
        if (n == m.N && k == m.K && w == m.wf) { sel = s; chosen = true; }
        p0 = p1 + 1;
      }
    }
    return chosen && sel.on;
  }
  void launch_sk(const TMat& m, const SkSel& sel, int mode, const uint8_t* A, const uint8_t* AS, const float* AG, int R, float* out, int ldo) {
    static const int prewait = [] { const char* c = getenv("H5_GEMM_PREWAIT"); if (!c) return 1 << 30; const int v = atoi(c);   // unset = whole ring, as k_gemm_fp4
      if (v < 0) throw std::runtime_error("H5_GEMM_PREWAIT must be >= 0"); return v; }();
    const int grid = sel.grid, chunk = sk_chunk(sel.shape, m.wf, m.N, m.K, 0, grid, sel.items);
    const SkArgs a{A, AS, AG, R, m.K, m.N, (const uint8_t*)m.w4, m.wf == 1 ? (const uint8_t*)m.s4 : nullptr, out, ldo,
                   m.mean_scale ? m.mean_scale : m.rs, sk_ws, sk_cnt, prewait, sk_site_counter(), chunk};
    CK(sk_launch(sel.shape, m.wf, mode, 0, a, grid, st));
  }
  // Each launch (site) takes its own zeroed work counter; graphs replay sequentially, so a site's counter is
  // rearmed by its last CTA long before the site runs again.
  int sk_site = 0;
  int* sk_site_counter() { return sk_next + (sk_site++ % SK_SITES); }
  // H5_PF_OUT_MB / H5_PF_GU_MB (read at capture; 0 = off): after each in-projection, a chained prefetch kernel
  // L2-prefetches up to that many MB of the layer's out-projection / gate-up weights, so they stream
  // during the conv/GDN/attention kernels. Weights are [kb][group] chunks; the prefix is the first K blocks.
  static double env_mb(const char* name) {
    const char* v = getenv(name); if (!v) return 0;
    char* e = nullptr; const double x = strtod(v, &e);
    if (!e || *e || !(x >= 0) || x > 96) throw std::runtime_error(std::string(name) + " must be 0..96 (MB)");
    return x;
  }
  void prefetch_after_in(const Layer& Ly) {
    const double out_mb = env_mb("H5_PF_OUT_MB"), gu_mb = env_mb("H5_PF_GU_MB");
#ifdef H5_PROFILE_SCALE_CARRIER
    if (binary && (out_mb > 0 || gu_mb > 0)) throw std::runtime_error("binary target does not support chained packed-carrier prefetch");
    auto bytes_of = [](const TMat& m) { return (size_t)(m.N / 32) * (m.K / 128) * (m.wf == 3 ? 832 : m.wf == 1 || m.wf == 6 ? 2048 : m.wf == 4 ? 1536 : 1024); };
    if (out_mb > 0) CK(launch_l2pf_chain(Ly.out.w4, std::min(bytes_of(Ly.out), (size_t)(out_mb * 1048576.0)), nsm, st));
    if (gu_mb > 0) CK(launch_l2pf_chain(Ly.gu.w4, std::min(bytes_of(Ly.gu), (size_t)(gu_mb * 1048576.0)), nsm, st));
#else
    if (out_mb > 0 || gu_mb > 0) throw std::runtime_error("chained prefetch requires a scale-carrier build");
#endif
  }
  // Head screening probe: partial head logits over `keep` of every 8 K blocks (one per Hadamard 1024-block
  // stride), for keep = 1, 2, 4; ranks of the exact argmax accumulate into screen_hist[3][16].
  const bool screen_probe = getenv("H5_HEAD_SCREEN_PROBE") && !strcmp(getenv("H5_HEAD_SCREEN_PROBE"), "1");
  int* screen_hist = nullptr; float* screen_logits = nullptr;
  static int*& screen_hist_global() { static int* p = nullptr; return p; }
  void head_screen_probe(int r0, int rr) {
#ifdef H5_PROFILE_SCALE_CARRIER
    if (!head.mean_scale || head.wf != 3) throw std::runtime_error("head screening probe needs the dense5 carrier head");
    if (!screen_hist) {
      CK(cudaHostAlloc(&screen_hist, 3 * 16 * sizeof(int), cudaHostAllocMapped)); memset(screen_hist, 0, 3 * 16 * sizeof(int));
      screen_hist_global() = screen_hist;
      screen_logits = dalloc<float>((size_t)128 * VOCAB);
      atexit([] { const int* h = screen_hist_global(); for (int k = 0; k < 3; k++) { fprintf(stderr, "head screen keep %d/8: rank histogram", 1 << k);
        for (int b = 0; b < 16; b++) fprintf(stderr, " %d", h[k * 16 + b]); fprintf(stderr, "\n"); } });
    }
    int* dh = nullptr; CK(cudaHostGetDevicePointer((void**)&dh, screen_hist, 0));
    const uint8_t* A = q4 + (size_t)r0 * head.K / 2; const uint8_t* AS = s4 + (size_t)r0 * head.K / 16; const float* AG = g4 + r0;
    const size_t kbytes = (size_t)(head.N / 32) * 832;
    const int ci = tune(head, rr, A, AS, AG, false).first;
    for (int k = 0; k < 3; k++) {
      const int keep = 1 << k;
      for (int blk = 0; blk < head.K / 1024; blk++) {
        const int kb = blk * 8;
        CK(launch_scale_carrier(ci, 1, blk ? 1 : 0, A + (size_t)kb * 4096, AS + (size_t)kb * 512, AG, rr, keep * 128,
                                (const uint8_t*)head.w4 + kb * kbytes, head.N, screen_logits, VOCAB, head.mean_scale, st));
      }
      pdl(k_screen_rank, dim3(rr), dim3(1024), 0, st, (const float*)screen_logits, VOCAB, (const int*)(d_out + r0), dh + k * 16);
    }
#else
    throw std::runtime_error("head screening probe needs a scale-carrier build");
#endif
  }
  void launch_sk_in_ab(const TMat& m, const TMat& ab_m, const SkSel& sel, int R, float* out, int ldo, float* out2, int ldo2) {
    static const int prewait = [] { const char* c = getenv("H5_GEMM_PREWAIT"); if (!c) return 1 << 30; const int v = atoi(c);
      if (v < 0) throw std::runtime_error("H5_GEMM_PREWAIT must be >= 0"); return v; }();
    if (m.wf != 3 || !m.mean_scale || ab_m.wf != 1 || !ab_m.rs || !ab_m.s4) throw std::runtime_error("merged in/alpha-beta launch needs carrier dense5 and NVFP4 payloads");
    const int grid = sel.grid;
    SkArgs a{q4, s4, g4, R, m.K, m.N, (const uint8_t*)m.w4, nullptr, out, ldo, m.mean_scale, sk_ws, sk_cnt, prewait,
             sk_site_counter(), sk_chunk(sel.shape, 3, m.N, m.K, ab_m.N, grid, sel.items)};
    a.W2 = (const uint8_t*)ab_m.w4; a.S2 = (const uint8_t*)ab_m.s4; a.RS2 = ab_m.rs; a.N2 = ab_m.N; a.out2 = out2; a.ldo2 = ldo2;
    CK(sk_launch(sel.shape, 3, SK_ATOMIC, 0, a, grid, st));
  }
  void launch_projection(const TMat& m, int ci, int ks, const uint8_t* A, const uint8_t* AS, const float* AG,
                         int R, float* out, int ldo, bool add, bool precleared = false) {
    if (deterministic_gemm) {
      const size_t required = ks > 1 ? size_t(ks) * R * m.N : 0;
      float* partials = deterministic_partials(required);
      CK(launch_gemm_fp4_deterministic(ci, ks, A, AS, AG, R, m.K, m.w4, m.s4,
          m.N, out, ldo, add, st, m.mean_scale ? m.mean_scale : m.rs,
          m.wf, m.mean_scale != nullptr, partials, required));
      return;
    }
    if (SkSel sel; sk_select(m, R, sel)) { launch_sk(m, sel, add || precleared ? SK_ATOMIC : SK_STORE, A, AS, AG, R, out, ldo); return; }
#ifdef H5_PROFILE_SCALE_CARRIER
    if (m.mean_scale) {
      if (ks > 1 && !add && !precleared) CK(cudaMemsetAsync(out, 0, (size_t)R * ldo * sizeof(float), st));
      const int mode = ks > 1 ? 2 : add ? 1 : 0;
      launch_carrier_projection(m, ci, ks, mode, A, AS, AG, R, out, ldo);
    } else
#endif
    {
      const auto launch = precleared ? launch_gemm_fp4_precleared_cfg : launch_gemm_fp4_cfg;
      launch(ci, ks, A, AS, AG, R, m.K, m.w4, m.s4, m.N, out, ldo, add, st, m.rs, m.wf);
    }
  }
  void launch_projection_silu(const TMat& m, int ci, const uint8_t* A, const uint8_t* AS, const float* AG,
                              int R, float* out, int ldo) {
    if (SkSel sel; !deterministic_gemm && m.wf == 3 && sk_select(m, R, sel)) { launch_sk(m, sel, SK_SILU, A, AS, AG, R, out, ldo); return; }
#ifdef H5_PROFILE_SCALE_CARRIER
    if (m.mean_scale) launch_carrier_projection(m, ci, 1, 3, A, AS, AG, R, out, ldo);
    else
#endif
    launch_gemm_fp4_silu(ci, A, AS, AG, R, m.K, m.w4, m.s4, m.N, out, ldo, st, m.rs, m.wf);
  }
  void gemm_residual_prep(const TMat& m, int R, float* residual, const float* norm, bool early) {
    const auto c = tune(m, R, q4, s4, g4, true);
    if (c.second == 1) {
      launch_projection(m, c.first, 1, q4, s4, g4, R, residual, D, true);
      prep_early_next = early;
      prep(residual, nullptr, D, norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R);
      return;
    }
    const size_t required = deterministic_gemm_partial_floats(R, m.N, c.second);
    float* partials = deterministic_partials(required);
    CK(launch_gemm_fp4_partials(c.first, c.second, q4, s4, g4, R, m.K,
        m.w4, m.s4, m.N, st, m.mean_scale ? m.mean_scale : m.rs, m.wf,
        m.mean_scale != nullptr, partials, required));
    PrepArgs a{};
    a.x = residual; a.ldx = D; a.norm_w = norm; a.signs = signs5120; a.n = D;
    a.flags = P_NORM | P_SIGN | P_HADAMARD | P_QUANT;
    a.q4 = q4; a.s4 = s4; a.g4 = g4; a.early = early;
    CK(launch_prep_ordered(a, OrderedResidual{partials, residual, R, c.second}, st));
  }
  // Autotuned FP4 launch: first use of (N, K, R, add) times every config x split-K on a scratch output.
  std::map<std::tuple<int, int, int, bool, int>, std::pair<int, int>> tuned;
  void f02_plan(const TMat& m) {
    const auto key = std::make_tuple(m.N, m.K, F02_ROWS, false, F02_PLAN_FORMAT);
    const auto expected = std::make_pair(F02_PRODUCER_CONFIG, 1);
    const auto it = tuned.find(key);
    if (it != tuned.end()) {
      if (it->second != expected) throw std::runtime_error("frozen F02 plan must select producer FW4/WG2/TOK64 config2, split1");
      return;
    }
    if (getenv("H5_PLAN_READ")) throw std::runtime_error("frozen launch plan missing F02 H32 shape (format102)");
    tuned.emplace(key, expected);
  }
  void f02_launch(const TMat& m, int R, float* out) {
    f02_plan(m);
    PrepArgs pa{}; pa.x = out; pa.ldx = FF; pa.n = FF; pa.q4 = q4; pa.s4 = s4; pa.g4 = g4;
    CK(f02::launch_h32(q4, s4, g4, R, m.K, m.w4, m.s4, m.N, out, FF, st, m.rs, signs17408, pa));
  }
  float* scratch = nullptr; float* gates = nullptr;
  // H5_PLAN_FILL=1: shapes absent from the frozen H5_PLAN_READ plan (e.g. row buckets above 512) are autotuned
  // at first use instead of failing; the frozen entries stay as they are.
  static bool plan_fill() { static const bool v = getenv("H5_PLAN_FILL") && !strcmp(getenv("H5_PLAN_FILL"), "1"); return v; }
  bool tune_verbose = getenv("H5_TUNE_VERBOSE") != nullptr;
  // Tuning buckets: powers of two from 8 rows, so a serving run meets a handful of keys and pretune()
  // can cover all of them at startup.
  static int rbucket(int R) { int b = 8; while (b < R && b < 4096) b <<= 1; return b; }
  std::pair<int, int> tune(const TMat& m, int R, const uint8_t* A, const uint8_t* AS, const float* AG, bool add) {
    auto key = std::make_tuple(m.N, m.K, rbucket(R), add, m.plan_wf());
    // H5_PLAN_OVERRIDE=N:K:ADD:WF=CFG:KS[,...] or none (read per launch, so at capture): replaces the frozen plan
    // entry for that shape on steps of at most 8 rows.
    if (R <= 8) if (const char* ov = getenv("H5_PLAN_OVERRIDE"); ov && strcmp(ov, "none")) {   // "none" = frozen plan (explicit off)
      std::string all = ov; size_t p0 = 0;
      while (p0 <= all.size()) {
        const size_t p1 = std::min(all.find(',', p0), all.size());
        const std::string item = all.substr(p0, p1 - p0);
        int n, k, a, w, c, ks; char tail = 0;
        if (sscanf(item.c_str(), "%d:%d:%d:%d=%d:%d%c", &n, &k, &a, &w, &c, &ks, &tail) != 6 || (a != 0 && a != 1) || c < 0 || c >= fp4_ncfg() || ks < 1 || k / 128 / ks < 1)
          throw std::runtime_error("H5_PLAN_OVERRIDE items must be N:K:ADD:WF=CFG:KS");
        if (n == m.N && k == m.K && (a != 0) == add && w == m.plan_wf()) return {c, ks};
        p0 = p1 + 1;
      }
    }
    auto it = tuned.find(key);
    if (it != tuned.end()) return it->second;
    if (getenv("H5_PLAN_READ") && !plan_fill()) throw std::runtime_error("frozen launch plan missing GEMM shape");
    if (const char* f = getenv("H5_FORCE_CFG")) { int ci = 0, ks = 1; sscanf(f, "%d,%d", &ci, &ks); if (m.K / 128 / ks < 1) ks = 1; tuned[key] = {ci, ks}; return tuned[key]; }
    TuningEvents events("GEMM N=" + std::to_string(m.N) + " K=" + std::to_string(m.K) + " R=" + std::to_string(R) + " wf=" + std::to_string(m.wf));
    float best = 1e9; std::pair<int, int> bc{0, 1};
    const int nkb = m.K / 128;
    for (int ci = 0; ci < fp4_ncfg(); ci++)
      for (int ks : {1, 2, 3, 4, 6, 8, 12}) {
        if (nkb / ks < 4) continue;
        float tot = 0; bool launched = true;
        for (int rep = 0; rep < 4 && launched; rep++) {
          events.begin(st);
          launch_projection(m, ci, ks, A, AS, AG, std::min(rbucket(R), max_rows), scratch, m.N, add);
          if (!events.accepted(ci, ks)) { launched = false; break; }
          const float ms = events.finish(st); if (rep) tot += ms;
        }
        if (launched && tot < best) { best = tot; bc = {ci, ks}; }
      }
    CK(cudaGetLastError());
    events.require_candidate(best < 1e9);
    if (tune_verbose) fprintf(stderr, "tune N=%d K=%d R=%d add=%d fmt=%d: %s %.1f us\n", m.N, m.K, R, (int)add, m.rs != nullptr, fp4_cfg_name(bc.first, bc.second), best / 3 * 1e3); (void)0;
    tuned[key] = bc;
    return bc;
  }
  // Fused gate/up (interleaved image): out[R][FF] = silu(gate) * up. Tuned over the WG=2 configs, no split-K.
  // Fused post-norm prep + GU (H5_FUSE_GU=1, or h5_set_fuse_gu per engine before capture): returns the
  // frozen plan's configuration when it supports in-CTA preparation for this row count, else -1.
  struct GuSplit { bool on; int cfg, ks; };
  GuSplit gu_split(const TMat& m, int R, F02Route f02) {
    const char* e = getenv("H5_GU_SPLIT");   // "0" = fused GU (explicit off)
    if (!e || !strcmp(e, "0") || R > 8 || f02 == F02Route::H32) return {false, 0, 0};
    int cfg = -1, ks = -1; char tail = 0;
    if (sscanf(e, "%d,%d%c", &cfg, &ks, &tail) != 2 || cfg < 0 || cfg >= fp4_ncfg() || ks < 2 || m.K / 128 / ks < 1)
      throw std::runtime_error("H5_GU_SPLIT must be CFG,KS with a valid config and 2 <= KS <= K/128");
    if (!m.il || !m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4)) throw std::runtime_error("H5_GU_SPLIT requires the interleaved carrier dense5 GU with FP4 activations");
    return {true, cfg, ks};
  }
  bool gu_plain(const TMat& m, int R) const {
    static const int rows = [] { const char* c = getenv("H5_GU_PLAIN_ROWS"); if (!c) return 1 << 30; char* e = nullptr; const long v = strtol(c, &e, 10);
      if (!e || *e || e == c || v < 1) throw std::runtime_error("H5_GU_PLAIN_ROWS must be a positive row count"); return (int)v; }();
    return R >= rows && m.il && m.mean_scale && m.wf == 3 && (act_fp4 || force_fp4);
  }
  int fused_gu_config(const TMat& m, int R) {
#ifndef H5_PROFILE_SCALE_CARRIER
    if (fuse_gu_selected()) throw std::runtime_error("GU fusion requires a scale-carrier build");
    return -1;
#else
    if (!fuse_gu_selected() || R != 1 || !m.il || !m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4) || f07_mode != F07Mode::Ordinary) return -1;
    const auto it = tuned.find(std::make_tuple(m.N, m.K, rbucket(R), false, 100));
    if (it == tuned.end() || !scale_carrier_fuse_supported(it->second.first)) return -1;
    return it->second.first;
#endif
  }
  void gemm_silu(const TMat& m, int R, float* out, int ldo, bool allow_f07 = true) {
    const uint8_t* A = q4; const uint8_t* AS = s4; const float* AG = g4;
    const auto f07_route = select_f07_route(allow_f07 ? f07_mode : F07Mode::Ordinary, R, m.wf, m.il, m.mean_scale != nullptr);
    if (f07_route == F07Route::Incompatible) throw std::runtime_error("incompatible F07 GU payload");
    if (f07_route == F07Route::Degree8) {
      CK(launch_f07_degree8(A, AS, AG, R, m.K, m.w4, m.s4, m.N, out, ldo, st));
      return;
    }
    auto key = std::make_tuple(m.N, m.K, rbucket(R), false, m.wf == 6 || m.wf == 7 ? 100 + m.wf : m.mean_scale && m.wf == 0 ? 101 : 100);
    auto it = tuned.find(key);
    int ci;
    if (it != tuned.end()) ci = it->second.first;
    else if (getenv("H5_PLAN_READ") && !plan_fill()) throw std::runtime_error("frozen launch plan missing fused GEMM shape");
    else if (getenv("H5_FORCE_CFG") && fp4_cfg_wg2(atoi(getenv("H5_FORCE_CFG")))) { ci = atoi(getenv("H5_FORCE_CFG")); tuned[key] = {ci, 1}; }
    else {
      TuningEvents events("GU GEMM N=" + std::to_string(m.N) + " K=" + std::to_string(m.K) + " R=" + std::to_string(R) + " wf=" + std::to_string(m.wf));
      float best = 1e9; ci = -1;
      for (int c = 0; c < fp4_ncfg(); c++) {
        if (!fp4_cfg_wg2(c)) continue;
        float tot = 0; bool launched = true;
        for (int rep = 0; rep < 4 && launched; rep++) {
          events.begin(st);
          launch_projection_silu(m, c, A, AS, AG, std::min(rbucket(R), max_rows), scratch, ldo);
          if (!events.accepted(c, 1)) { launched = false; break; }
          const float ms = events.finish(st); if (rep) tot += ms;
        }
        if (launched && tot < best) { best = tot; ci = c; }
      }
      CK(cudaGetLastError());
      events.require_candidate(ci >= 0);
      if (tune_verbose) fprintf(stderr, "tune silu N=%d K=%d R=%d: %s %.1f us\n", m.N, m.K, R, fp4_cfg_name(ci, 1), best / 3 * 1e3);
      tuned[key] = {ci, 1};
    }
    launch_projection_silu(m, ci, A, AS, AG, R, out, ldo);
  }
  // Tune every projection shape at every row bucket up to max_rows (server startup).
  void pretune() {
    if (const char* path = getenv("H5_PLAN_READ")) {
      FILE* f=fopen(path,"r"); if(!f) throw std::runtime_error("cannot read frozen launch plan");
      int n,k,r,a,w,c,s,fields;
      while ((fields=fscanf(f,"%d %d %d %d %d %d %d",&n,&k,&r,&a,&w,&c,&s))==7) {
        if((w==F02_PLAN_FORMAT && (n!=GU || k!=D || r!=F02_ROWS || a!=0 || c!=F02_PRODUCER_CONFIG || s!=1)) || n<=0 || k<=0 || r<8 || (a!=0 && a!=1) || c<0 || c>=fp4_ncfg() || s<1 || k/128/s<1 || ((w==100 || w==101 || w==106 || w==107) && (!fp4_cfg_wg2(c) || s!=1)) || !tuned.emplace(std::make_tuple(n,k,r,a!=0,w),std::make_pair(c,s)).second) {
          fclose(f); throw std::runtime_error("invalid or duplicate frozen launch plan row");
        }
      }
      fclose(f); if(fields!=EOF || tuned.empty()) throw std::runtime_error("truncated or empty frozen launch plan");
      if (df.vhead.w4) for (int b = 8; b <= 128; b <<= 1) {
        const auto it = tuned.find(std::make_tuple(head.N, head.K, b, false, head.wf));
        if (it != tuned.end()) tuned.emplace(std::make_tuple(df.vhead.N, df.vhead.K, b, false, df.vhead.wf), std::make_pair(it->second.first, 1));
      }
    }
    const double t0 = now_s();
    std::vector<std::pair<const TMat*, bool>> mats;
    for (auto& y : L) { mats.push_back({&y.in, false}); mats.push_back({&y.out, true}); mats.push_back({&y.gu, false}); mats.push_back({&y.down, true}); if (!y.attn) mats.push_back({&y.ab4, false}); }
    mats.push_back({&head, false});
    if (df.vhead.w4) mats.push_back({&df.vhead, false});
    if (df.loaded) {
      for (int j = 0; j < NCAP; j++) mats.push_back({&df.fc[j], j > 0});
      mats.push_back({&df.hproj, false});
      for (auto& y : df.L) for (const TMat* m : {&y.qkv, &y.kv, &y.o, &y.gu, &y.down, &y.akp, &y.mkp}) mats.push_back({m, false});
    }
    int n = 0;
    for (int b = 8; b <= max_rows; b <<= 1)
      for (auto& [m, add] : mats) {
        if (!m->w4) continue;
        if (joint_fp4_r16 && m->wf == 8 && b <= 16) continue;
        const bool hd = m == &head || m == &df.vhead;
        if (hd && b > 128) continue;
        if (m->il) {
          gemm_silu_tune_only(*m, b);
          if (select_f02_route(f02_mode, b, m->wf, m->il, m->mean_scale != nullptr) == F02Route::H32) f02_launch(*m, b, scratch);
          n++; continue;
        }
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
    auto key = std::make_tuple(m.N, m.K, rbucket(R), false, m.wf == 6 || m.wf == 7 ? 100 + m.wf : m.mean_scale && m.wf == 0 ? 101 : 100);
    if (!tuned.count(key)) gemm_silu(m, R, scratch, FF, false);
    if (select_f07_route(f07_mode, R, m.wf, m.il, m.mean_scale != nullptr) == F07Route::Degree8)
      CK(launch_f07_degree8(q4, s4, g4, R, m.K, m.w4, m.s4, m.N, scratch, FF, st));
  }
  // Carrier dense5 add projection whose tail runs the next row-norm prep; activations flip afterwards.
  void gemm_tail_norm(const TMat& m, int R, float* out, int ldo, const TailArgs& ta) {
#ifdef H5_PROFILE_SCALE_CARRIER
    if (!m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4)) throw std::runtime_error("producer-tail prep requires the dense5 carrier projection with FP4 activations");
    const auto c = tune(m, R, q4, s4, g4, true);
    CK(launch_scale_carrier_tail(c.first, c.second, c.second > 1 ? 2 : 1, TAIL_NORM, q4, s4, g4, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, ta, st));
    act_flip();
#else
    throw std::runtime_error("producer-tail prep requires a scale-carrier build");
#endif
  }
  // Carrier dense5 add projection whose CTAs also zero the next split-K outputs (ta.clear_a/b) after their wait.
  void gemm_clear(const TMat& m, int R, float* out, int ldo, const TailArgs& ta) {
#ifdef H5_PROFILE_SCALE_CARRIER
    if (!m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4)) throw std::runtime_error("H5_DOWN_CLEAR requires the dense5 carrier projection with FP4 activations");
    const auto c = tune(m, R, q4, s4, g4, true);
    CK(launch_scale_carrier_tail(c.first, c.second, c.second > 1 ? 2 : 1, TAIL_SIDE, q4, s4, g4, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, ta, st));
#else
    throw std::runtime_error("H5_DOWN_CLEAR requires a scale-carrier build");
#endif
  }
  // Carrier dense5 projection launched with side duties only (TAIL_SIDE): clearing and/or the flag start.
  void gemm_side(const TMat& m, int R, float* out, int ldo, bool add, const TailArgs& ta) {
#ifdef H5_PROFILE_SCALE_CARRIER
    if (!m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4)) throw std::runtime_error("GEMM side duties require the dense5 carrier projection with FP4 activations");
    const auto c = tune(m, R, q4, s4, g4, add);
    const int mode = c.second > 1 ? 2 : add ? 1 : 0;
    if (mode == 0) throw std::runtime_error("GEMM side duties require a split-K or add projection");
    CK(launch_scale_carrier_tail(c.first, c.second, mode, TAIL_SIDE, q4, s4, g4, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, ta, st));
#else
    throw std::runtime_error("GEMM side duties require a scale-carrier build");
#endif
  }
  void gemm_silu_side(const TMat& m, int R, float* out, int ldo, const TailArgs& ta) {
#ifdef H5_PROFILE_SCALE_CARRIER
    if (!m.il || !m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4) || f07_mode != F07Mode::Ordinary)
      throw std::runtime_error("GU side duties require the interleaved dense5 carrier GU with FP4 activations");
    const auto it = tuned.find(std::make_tuple(m.N, m.K, rbucket(R), false, 100));
    if (it == tuned.end()) throw std::runtime_error("GU side duties require a frozen fused GU plan entry");
    CK(launch_scale_carrier_tail(it->second.first, 1, 3, TAIL_SIDE, q4, s4, g4, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, ta, st));
#else
    throw std::runtime_error("GU side duties require a scale-carrier build");
#endif
  }
  void gemm_silu_tail(const TMat& m, int R, float* out, int ldo, const TailArgs& ta) {
#ifdef H5_PROFILE_SCALE_CARRIER
    if (!m.il || !m.mean_scale || m.wf != 3 || !(act_fp4 || force_fp4) || f07_mode != F07Mode::Ordinary)
      throw std::runtime_error("producer-tail SiLU prep requires the interleaved dense5 carrier GU with FP4 activations");
    const auto it = tuned.find(std::make_tuple(m.N, m.K, rbucket(R), false, 100));
    if (it == tuned.end()) throw std::runtime_error("producer-tail SiLU prep requires a frozen fused GU plan entry");
    CK(launch_scale_carrier_tail(it->second.first, 1, 3, TAIL_CHUNK, q4, s4, g4, R, m.K, m.w4, m.N, out, ldo, m.mean_scale, ta, st));
    act_flip();
#else
    throw std::runtime_error("producer-tail prep requires a scale-carrier build");
#endif
  }
  static binary_fp4_pair::Plan pair_plan(const TMat& m, int R) {
    using S = binary_fp4_pair::Schedule;
    if (m.wf != 8 || R < 1 || R > 16) throw std::runtime_error("joint pair requires WF8 and R1..16");
    S schedule; int splits;
    if (m.N == VOCAB && m.K == D) {
      schedule = R == 1 ? S::F8G1T8 : R <= 8 ? S::F4G2T8 : S::F8G2T16;
      splits = R == 1 ? 8 : R <= 8 ? 4 : 1;
    } else if (m.N == D && m.K == FF) {
      schedule = R == 1 ? S::F8G1T8 : S::F4G2T8;
      splits = R <= 8 ? 16 : 8;
    } else if (m.N == GU && m.K == D) {
      schedule = R == 1 ? S::F8G1T8 : R <= 8 ? S::F4G2T8 : S::F4G2T16; splits = 1;
    } else {
      // Explicit structural plan for the other native projection families; timing is not inferred.
      schedule = R <= 8 ? S::F4G2T8 : S::F4G2T16; splits = 1;
    }
    return binary_fp4_pair::Plan{schedule, m.N, m.K, R, splits};
  }
  int native_sf_planes() const {
    switch (native_sf_mode) {
      case NativeSf::SF1R32: return 1;
      case NativeSf::SF2R32: return 2;
      case NativeSf::Off: throw std::runtime_error("native SF plane count requested while disabled");
    }
    throw std::runtime_error("invalid native SF mode");
  }
  binary_native_sf::Plan native_sf_plan(const TMat& m) const {
    const bool narrow = m.N <= D;
    const auto schedule = native_sf_mode == NativeSf::SF2R32 && narrow
      ? binary_native_sf::Schedule::F4G2T16 : binary_native_sf::Schedule::F8G1T32;
    return {schedule, m.N, m.K, 32, m.N <= 16384 ? 4 : 1, native_sf_planes()};
  }
  void gemm(const TMat& m, int R, float* out, int ldo, bool add, int r0 = 0, bool precleared = false) {
    if (precleared && (add || r0 != 0 || !(act_fp4 || force_fp4)))
      throw std::runtime_error("precleared GEMM requires FP4 nonadd full prepared rows");
    if (m.w1) {
      if (force_fp4 || !use_packed || ldo != m.N || precleared) throw std::runtime_error("invalid packed1 target projection");
      if (a8_prepared) {
        for (int row = 0; row < R;) {
          const auto carrier = a8_carrier(m.K, r0 + row);
          const int digit_rows = std::min(32, R - row);
          if (digit_rows_enabled(digit_rows) && digit_family(m)) {
            if (three_digits_enabled()) {
              const auto plan = three_digit_plan(m, digit_rows);
              CK(signed_fp4::encode(carrier, m.K, digit_rows, signed_a8_digits, st));
              CK(signed_fp4::project(plan, m.w1, m.s1, carrier, signed_a8_digits,
                out + (size_t)row * ldo, ldo, packed_partials, packed_partial_capacity, 1.f, add ? 1.f : 0.f, st));
            } else {
              const auto plan = digit_plan(m, digit_rows);
              CK(a8_fp4::encode(carrier, m.K, digit_rows, a8_digits, st));
              CK(a8_fp4::project(plan, m.w1, m.s1, carrier, a8_digits,
                out + (size_t)row * ldo, ldo, packed_partials, packed_partial_capacity, 1.f, add ? 1.f : 0.f, st));
            }
            row += digit_rows;
          } else {
            const int count = std::min(16, R - row);
            bitserial::Plan plan; CK(bitserial::select(m.N, m.K, count, &plan));
            CK(bitserial::project_carrier(plan, m.w1, m.s1, carrier, out + (size_t)row * ldo, packed_partials, 1.f, add ? 1.f : 0.f, st));
            row += count;
          }
        }
      } else {
        for (int row = 0; row < R; row += 8) {
          const int count = std::min(8, R - row); const packed1::Plan plan = packed_plan(m, count);
          CK(packed1::project(plan, m.w1, m.s1, xf + (size_t)(r0 + row) * m.K, out + (size_t)row * ldo, packed_tables, packed_partials, 1.f, add ? 1.f : 0.f, st));
        }
      }
    } else if (act_fp4 || force_fp4) {
      if (!m.w4 || !m.s4) throw std::runtime_error("FP4 GEMM requested without FP4 weights (N=" + std::to_string(m.N) + ", K=" + std::to_string(m.K) + ")");
      const uint8_t* A = q4 + (size_t)r0 * m.K / 2; const uint8_t* AS = s4 + (size_t)r0 * m.K / 16; const float* AG = g4 + r0;
      if (m.native_sf && !force_fp4 && R == 32 && r0 % 64 == 0) {
        if (m.il || m.wf != 8 || !m.native_row_factor) throw std::runtime_error("invalid native SF weight coordinate");
        CK(binary_native_sf::project(native_sf_plan(m), {A, AS, AG},
          {q4_res + (size_t)r0 * m.K / 2, s4_res + (size_t)r0 * m.K / 16, g4_res + r0},
          m.w4, m.native_sf, m.native_row_factor, out, ldo, native_sf_partials, native_sf_capacity, st, add));
        return;
      }
      if (joint_fp4_r16 && !force_fp4 && R <= 16) {
        if (precleared || r0 % 64 || m.il) throw std::runtime_error("joint pair requires aligned noninterleaved prepared rows");
        CK(binary_fp4_pair::project(pair_plan(m, R), {A, AS, AG},
          {q4_res + (size_t)r0 * m.K / 2, s4_res + (size_t)r0 * m.K / 16, g4_res + r0},
          m.w4, m.s4, out, ldo, pair_partials, pair_capacity, st, add));
        return;
      }
      auto c = tune(m, R, A, AS, AG, add);
      // H5_HEAD_KS=N (read at capture): split-K for the vocabulary head. Split-K > 1 needs the logits
      // cleared first (a graph clear kernel of R x 248320 floats that also breaks the PDL chain);
      // the head has 1940 N-tiles, so it fills the GPU without splitting K.
      if (&m == &head) if (const char* hk = getenv("H5_HEAD_KS")) {
        char* end = nullptr; const long ks = strtol(hk, &end, 10);
        if (!end || *end || end == hk || ks < 1 || ks > 12 || m.K / 128 / ks < 4) throw std::runtime_error("H5_HEAD_KS must be 1..12 with at least 4 K-blocks per split");
        c.second = (int)ks;
      }
      launch_projection(m, c.first, c.second, A, AS, AG, R, out, ldo, add, precleared);
      if (binary && !force_fp4 && fp4_passes == 2) {
        launch_projection(m, c.first, c.second, q4_res + (size_t)r0 * m.K / 2, s4_res + (size_t)r0 * m.K / 16, g4_res + r0, R, out, ldo, true);
      }
    }
    else {
      if (!m.w || !m.s) throw std::runtime_error("INT8 GEMM requested without INT8 weights (N=" + std::to_string(m.N) + ", K=" + std::to_string(m.K) + ")");
      launch_gemm(xq + (size_t)r0 * m.K, xs + (size_t)r0 * (m.K / 128), R, m.K, m.w, m.s, m.N, out, ldo, add, st);
    }
  }

  // time every projection of the model at R rows (inputs are whatever the buffers hold)
  void gemm_bench(int R, int reps) {
    last_logits_rows = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    if (R < 1 || R > max_rows || reps < 1) throw std::runtime_error("invalid projection benchmark capacity");
    double macs = 0; size_t wb = 0;
    auto project = [&](const TMat& m, int rows, float* out, int ld, bool add) {
      if (use_packed) { a8_prepared = uses_a8(rows); if (a8_prepared) produce_a8(m.K, rows); }
      gemm(m, rows, out, ld, add);
    };
    auto all = [&]() {
      clear_a8_error();
      for (auto& Ly : L) {
        project(Ly.in, R, proj, Ly.attn ? ATT_IN : GDN_IN, false);
        if (binary && !Ly.attn) project(Ly.ab4, R, ab, 96, false);
        project(Ly.out, R, x, D, true);
        if (Ly.gu.il && act_fp4) gemm_silu(Ly.gu, R, proj, FF); else project(Ly.gu, R, proj, GU, false);
        project(Ly.down, R, x, D, true);
      }
      project(head, std::min(R, 128), logits, VOCAB, false);
    };
    auto bytes = [&](const TMat& m) {
      const size_t nk = (size_t)m.N * m.K;
      return m.w1 ? nk / 8 + nk / 64 : act_fp4 ? (size_t)(m.N / 32) * (m.K / 128) * ((m.wf == 8 ? 512 : m.wf == 3 ? 832 : m.wf == 1 || m.wf == 6 ? 2048 : m.wf == 4 ? 1536 : 1024) + (m.wf == 1 ? 256 : m.wf >= 4 && m.wf != 8 ? 128 : 64)) : nk / 4 + nk / 32;
    };
    for (auto& Ly : L) for (const TMat* m : {&Ly.in, &Ly.out, &Ly.gu, &Ly.down, &Ly.ab4}) {
      if (m == &Ly.ab4 && (!binary || Ly.attn)) continue;
      macs += (double)m->N * m->K * R; wb += bytes(*m);
    }
    macs += (double)head.N * head.K * std::min(R, 128); wb += bytes(head);
    all(); CK(cudaStreamSynchronize(st));
    float best = 1e9;
    for (int i = 0; i < reps; i++) { cudaEventRecord(e0, st); all(); cudaEventRecord(e1, st); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms); }
    read_a8_error(); CK(cudaStreamSynchronize(st)); require_a8_valid();
    const char* mode = use_packed ? (uses_a8(R) ? "packed1-a8" : "packed1-f32") : act_fp4 ? "fp4" : "int8";
    printf("gemm R=%4d %s: %.3f ms  %.1f TMAC/s  weights %.0f GB/s  %.4f ms/row\n", R, mode, best, macs / best / 1e9, wb / best / 1e6, best / R);
  }

  // Clone slot src into dst for the first npos positions (prefix cache): recurrent state, conv ring,
  // KV [0, npos) and the drafter's ring. Pending speculative rows are dropped.
  void copy_slot(int src, int dst, int npos, bool pending = false) {
    if (native_dspark) native_dspark->copy_slot(src, dst);
    const size_t sb = state_slot_bytes;
    CK(cudaMemcpyAsync((uint8_t*)gdn_state + dst * sb, (uint8_t*)gdn_state + src * sb, sb, cudaMemcpyDeviceToDevice, st));
    const size_t cb = (size_t)NGDN * conv_banks * 3 * CONV_CH;
    CK(cudaMemcpyAsync(conv_state + dst * cb, conv_state + src * cb, cb * 4, cudaMemcpyDeviceToDevice, st));
    conv_par[dst] = conv_par[src];
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
      const size_t pc = (size_t)NGDN * conv_banks * pending_rows * CONV_CH * state_es;
      CK(cudaMemcpyAsync((uint8_t*)pend_gdn + dst * pg, (uint8_t*)pend_gdn + src * pg, pg, cudaMemcpyDeviceToDevice, st));
      CK(cudaMemcpyAsync((uint8_t*)pend_conv + dst * pc, (uint8_t*)pend_conv + src * pc, pc, cudaMemcpyDeviceToDevice, st));
      pend_n[dst] = pend_n[src]; commit_next[dst] = commit_next[src]; pend_par[dst] = pend_par[src];
      keep_slot[dst] = keep_slot[src];
    } else { pend_n[dst] = 0; commit_next[dst] = 0; pend_par[dst] = 0; keep_slot[dst] = 0; }
  }

  int clone_slot(int src, int dst, int npos) {
    if (src < 0 || dst < 0 || src >= max_slots || dst >= max_slots || npos < 0 || npos > ctx) return -1;
    if (src == dst) return 0;
    copy_slot(src, dst, npos, true);
    CK(cudaStreamSynchronize(st)); // The bounded two-slot discriminator charges the complete clone before verification.
    return 0;
  }

  void reset(int slot) {
    if (native_dspark) native_dspark->reset(slot);
    if (diagnostic_readout_enabled) {
      diagnostic_readout_armed = false; diagnostic_readout_capturing = false; diagnostic_readout_ready = false;
    }
    pend_n[slot] = 0; commit_next[slot] = 0; pend_par[slot] = 0; keep_slot[slot] = 0; conv_par[slot] = 0;
    CK(cudaMemsetAsync((uint8_t*)gdn_state + (size_t)slot * state_slot_bytes, 0, state_slot_bytes, st));
    if (diagnostic_state) CK(cudaMemsetAsync((uint8_t*)diagnostic_state + (size_t)slot * diagnostic_slot_bytes, 0, diagnostic_slot_bytes, st));
    CK(cudaMemsetAsync(conv_state + (size_t)slot * NGDN * conv_banks * 3 * CONV_CH, 0, (size_t)NGDN * conv_banks * 3 * CONV_CH * 4, st));
  }

  // Eager phase intervals have one stream owner; forked work contributes through its main-stream join.
  bool prof = getenv("H5_PROF") != nullptr;
  cudaStream_t prof_stream = nullptr;
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
    if (!prof || st != prof_stream) return;
    cudaEvent_t e; CK(cudaEventCreate(&e)); CK(cudaEventRecord(e, prof_stream)); pev.push_back({name, e});
  }
  void prof_flush() {
    if (!prof || pev.empty()) return;
    CK(cudaEventSynchronize(pev.back().second));
    for (size_t i = 0; i + 1 < pev.size(); i++) { float ms; CK(cudaEventElapsedTime(&ms, pev[i].second, pev[i + 1].second)); ptime[pev[i].first] += ms; }
    for (auto& p : pev) CK(cudaEventDestroy(p.second));
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
    fprintf(stderr, "eager/instrumented phase breakdown over %d steps (ms/step, ordered main-stream intervals; not graph-wall time):\n", psteps);
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
    TMat fc_all, kv_all; float* dkv_all;   // fused ingest: fc over all captures (K=NCAP*D); k/v of every layer (N=DF_LAYERS*2*DF_KV)
    DLayer L[DF_LAYERS];
    __half *kc, *vc;
    float *th, *dkv, *xd, *xn, *cbuf, *tmpo, *dyn, *dqkv, *qr, *dy, *dgu, *hp, *topv; int *topi;
    int *meta; int* h_meta; int* d_drafts; int* h_drafts; float *d_conf, *h_conf;
    int *d_alt, *h_alt, *d_branch, *h_branch; float *d_alt_conf, *h_alt_conf;
    TopkScratch topk_scratch{nullptr, nullptr, 0};
    int completed_block_rows = 0, completed_prediction_rows = 0;
    double* sample_scores = nullptr;
    // H5_DRAFT_VOCAB=FILE: candidate head over a pruned vocabulary (rows of the target head listed in
    // FILE, one token id per line); vmap[column] = token id. Unset: the full shared target head.
    TMat vhead; int* vmap = nullptr; int vocab_n = VOCAB; float* vlogits = nullptr;
  } df;
  void draft_vocab_load(const char* path) {
    FILE* f = fopen(path, "r"); if (!f) throw std::runtime_error("H5_DRAFT_VOCAB: cannot read file");
    std::vector<int> ids; std::vector<char> seen(VOCAB, 0); int id;
    while (fscanf(f, "%d", &id) == 1) {
      if (id < 0 || id >= VOCAB || seen[id]) { fclose(f); throw std::runtime_error("H5_DRAFT_VOCAB: token ids must be distinct and inside the vocabulary"); }
      seen[id] = 1; ids.push_back(id);
    }
    const bool clean_end = feof(f); fclose(f);
    if (!clean_end) throw std::runtime_error("H5_DRAFT_VOCAB: non-integer entry");
    if (ids.empty() || ids.size() % 64 || ids.size() > (size_t)VOCAB) throw std::runtime_error("H5_DRAFT_VOCAB: count must be a positive multiple of 64");
    const GTensor& h = g.t("output.weight");
    const size_t row_bytes = h.bytes / h.ne[1];
    if (row_bytes * h.ne[1] != h.bytes || h.ne[1] != (uint64_t)VOCAB) throw std::runtime_error("H5_DRAFT_VOCAB: unexpected head layout");
    vocab_rows.resize(ids.size() * row_bytes);
    for (size_t i = 0; i < ids.size(); i++) memcpy(vocab_rows.data() + i * row_bytes, h.data + (size_t)ids[i] * row_bytes, row_bytes);
    GTensor t = h; t.name = "draft_vocab_head"; t.ne[1] = ids.size(); t.data = vocab_rows.data(); t.bytes = vocab_rows.size();
    if (use_packed) { packed1::Plan plan; CK(packed1::select((int)ids.size(), D, 8, &plan)); }
    df.vhead = tmat(std::vector<const GTensor*>{&t}, D, TargetProjection::VocabularyHead);
    std::vector<uint8_t>().swap(vocab_rows);
    df.vmap = upload<int>(ids.data(), ids.size()); df.vocab_n = (int)ids.size();
    df.vlogits = dalloc<float>((size_t)128 * df.vocab_n);
    fprintf(stderr, "drafter candidate vocabulary: %d of %d tokens (%s)\n", df.vocab_n, VOCAB, path);
  }
  std::vector<uint8_t> vocab_rows;
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
  // Absmean ternarisation of bf16 drafter weights, per 32-row x 128-K block row: s0 = mean|w|,
  // t = clamp(round(w / s0), -1, 1), then the least-squares scale for those trits, s = sum_{t != 0} |w| / nnz.
  // Stored as dense5 labels + FP16 per-128 scales (WF 3, original-scale GEMM path).
  TMat ternary_tmat(const SafeT& st, const std::vector<std::string>& names, int col0 = 0, int Kc = 0) {
    std::vector<int64_t> sh;
    int N = 0, K = 0;
    for (auto& n : names) { st.t(n, sh); N += (int)sh[0]; K = Kc ? Kc : (int)sh[1]; }
    if (N % 32 || K % 128) throw std::runtime_error("ternary drafter projection needs N % 32 == 0 and K % 128 == 0");
    const int nkb = K / 128, NG = N / 32;
    std::vector<uint8_t> W((size_t)NG * nkb * 832); std::vector<uint16_t> S((size_t)N * nkb);
    int row0 = 0;
    for (auto& n : names) {
      const uint16_t* p = st.t(n, sh);
      const int Nt = (int)sh[0]; const size_t ld = sh[1];
      std::atomic<int> next{0};
      std::mutex err_mu; double err2 = 0, ref2 = 0;
      auto worker = [&]() {
        int8_t trit[32][128]; float sc[32];
        double e2 = 0, r2 = 0;
        for (;;) {
          const int gl = next.fetch_add(1); if (gl >= Nt / 32) { std::lock_guard<std::mutex> lk(err_mu); err2 += e2; ref2 += r2; break; }
          const int g = row0 / 32 + gl;
          for (int kb = 0; kb < nkb; kb++) {
            for (int r = 0; r < 32; r++) {
              const uint16_t* row = p + (size_t)(gl * 32 + r) * ld + col0 + kb * 128;
              float s0 = 0.f; for (int i = 0; i < 128; i++) s0 += fabsf(bf2f(row[i]));
              s0 /= 128.f;
              float sum = 0.f; int nnz = 0;
              for (int i = 0; i < 128; i++) {
                const float w = bf2f(row[i]);
                const int t = s0 > 0.f && fabsf(w) > 0.5f * s0 ? (w > 0.f ? 1 : -1) : 0;
                trit[r][i] = (int8_t)t; if (t) { sum += fabsf(w); nnz++; }
              }
              sc[r] = nnz ? sum / nnz : 0.f;
              for (int i = 0; i < 128; i++) { const double w = bf2f(row[i]), d = w - trit[r][i] * (double)sc[r]; e2 += d * d; r2 += w * w; }
            }
            pack_trit_chunk(trit, sc, W.data() + ((size_t)kb * NG + g) * 832, S.data() + ((size_t)kb * NG + g) * 32, true);
          }
        }
      };
      std::vector<std::thread> th;
      for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
      for (auto& x : th) x.join();
      fprintf(stderr, "ternary drafter %s [%d x %zu cols %d..%d]: relative error %.4f\n", n.c_str(), Nt, ld, col0, col0 + K, ref2 > 0 ? sqrt(err2 / ref2) : 0.0);
      row0 += Nt;
    }
    TMat m; m.N = N; m.K = K; m.wf = 3; m.drafter_requant = true;
    m.w4 = upload<uint4>(W.data(), W.size() / 16); m.s4 = upload<__half>(S.data(), S.size());
    return m;
  }
  // 3-bit ({±0.5,±1.5,±3,±6}, WF 4) or 2-bit ({±0.5,±1.5}, WF 5) E2M1-subset codes with a UE4M3 scale per 32 K
  // relative to an FP32 row scale. Per block, the scale minimising squared error over a grid of 15 candidates;
  // codes are nearest levels under the decoded UE4M3 scale. Layout in fp4.cu (WFmt).
  TMat lowbit_tmat(const SafeT& st, const std::vector<std::string>& names, int bits, int col0 = 0, int Kc = 0) {
    if (bits != 2 && bits != 3) throw std::runtime_error("low-bit drafter format must be 2 or 3 bits");
    std::vector<int64_t> sh;
    int N = 0, K = 0;
    for (auto& n : names) { st.t(n, sh); N += (int)sh[0]; K = Kc ? Kc : (int)sh[1]; }
    if (N % 32 || K % 128) throw std::runtime_error("low-bit drafter projection needs N % 32 == 0 and K % 128 == 0");
    const int nkb = K / 128, NG = N / 32, CW = bits == 3 ? 1536 : 1024;
    static const float L3[4] = {0.5f, 1.5f, 3.f, 6.f}, L2[2] = {0.5f, 1.5f};
    const float* L = bits == 3 ? L3 : L2; const int nl = bits == 3 ? 4 : 2;
    std::vector<uint8_t> W((size_t)NG * nkb * CW, 0), S((size_t)NG * nkb * 128, 0); std::vector<float> RS(N, 0.f);
    int row0 = 0;
    for (auto& n : names) {
      const uint16_t* p = st.t(n, sh);
      const int Nt = (int)sh[0]; const size_t ld = sh[1];
      std::atomic<int> next{0};
      std::mutex err_mu; double err2 = 0, ref2 = 0;
      // nearest level by midpoints: 3-bit 1, 2.25, 4.5; 2-bit 1
      auto nearest = [&](float a) { return nl == 4 ? (a > 1.f) + (a > 2.25f) + (a > 4.5f) : (int)(a > 1.f); };
      auto worker = [&]() {
        std::vector<float> w(K), bs(K / 32);
        double e2 = 0, r2 = 0;
        std::vector<std::vector<uint8_t>> codes(32, std::vector<uint8_t>(K)); std::vector<std::vector<uint8_t>> sb(32, std::vector<uint8_t>(K / 32));
        for (;;) {
          const int gl = next.fetch_add(1); if (gl >= Nt / 32) { std::lock_guard<std::mutex> lk(err_mu); err2 += e2; ref2 += r2; break; }
          const int g = row0 / 32 + gl;
          for (int r = 0; r < 32; r++) {
            const uint16_t* row = p + (size_t)(gl * 32 + r) * ld + col0;
            for (int k = 0; k < K; k++) w[k] = bf2f(row[k]);
            float smax = 0.f;
            for (int b = 0; b < K / 32; b++) {
              float am = 0.f; for (int i = 0; i < 32; i++) am = std::max(am, fabsf(w[32 * b + i]));
              float best_s = am / L[nl - 1], best_e = INFINITY;
              for (int c = 0; c < 15 && am > 0.f; c++) {
                const float sc = am * (0.5f + 0.05f * c) / L[nl - 1];
                float e = 0.f; for (int i = 0; i < 32; i++) { const float x = fabsf(w[32 * b + i]) / sc; const float d = (L[nearest(x)] - x) * sc; e += d * d; }
                if (e < best_e) { best_e = e; best_s = sc; }
              }
              bs[b] = best_s; smax = std::max(smax, best_s);
            }
            const float rs = std::max(smax / 448.f, 1e-30f);
            RS[32 * g + r] = rs;
            for (int b = 0; b < K / 32; b++) {
              uint8_t e8 = __nv_cvt_float_to_fp8(bs[b] / rs, __NV_SATFINITE, __NV_E4M3);
              sb[r][b] = e8;
              const float sc = e4m3_val(e8) * rs;
              for (int i = 0; i < 32; i++) {
                const float x = w[32 * b + i];
                const int m = sc > 0.f ? nearest(fabsf(x) / sc) : 0;
                codes[r][32 * b + i] = (uint8_t)((x < 0.f) << 2 | m);
                const double d = (double)(x < 0.f ? -L[m] : L[m]) * sc - x; e2 += d * d; r2 += (double)x * x;
              }
            }
          }
          for (int kb = 0; kb < nkb; kb++) {
            uint8_t* ch = W.data() + ((size_t)kb * NG + g) * CW;
            uint8_t* sc = S.data() + ((size_t)kb * NG + g) * 128;
            for (int r = 0; r < 32; r++) for (int b = 0; b < 4; b++) sc[r * 4 + b] = sb[r][kb * 4 + b];
            for (int l = 0; l < 32; l++) {
              const int gid = l >> 2, tq = l & 3;
              uint32_t words[12] = {};
              for (int n2 = 0; n2 < 16; n2++) {
                const int mm = n2 >> 3, h = (n2 >> 2) & 1, o = n2 & 3;
                const int row = 16 * mm + 8 * (o & 1) + gid, k0 = kb * 128 + 64 * h + 32 * (o >> 1) + 8 * tq;
                for (int j = 0; j < 8; j++) {
                  const uint8_t c = codes[row][k0 + j]; const uint32_t sg = c >> 2, mg = c & 3;
                  if (bits == 2) {
                    const int wi = n2 >> 1;
                    words[wi] |= (n2 & 1) ? (sg << (4 * j + 2) | mg << (4 * j)) : (sg << (4 * j + 3) | mg << (4 * j + 1));
                  } else {
                    const int q = n2 >> 2, idx = n2 & 3, m1 = mg >> 1, m0 = mg & 1;
                    uint32_t& XY = words[3 * q + (idx >> 1)];
                    XY |= (idx & 1) ? (sg << (4 * j + 1) | (uint32_t)m1 << (4 * j)) : (sg << (4 * j + 3) | (uint32_t)m1 << (4 * j + 2));
                    static const int zpos[4] = {1, 0, 3, 2};
                    words[3 * q + 2] |= (uint32_t)m0 << (4 * j + zpos[idx]);
                  }
                }
              }
              for (int wi = 0; wi < (bits == 3 ? 12 : 8); wi++) memcpy(ch + (wi >> 2) * 512 + l * 16 + (wi & 3) * 4, &words[wi], 4);
            }
          }
        }
      };
      std::vector<std::thread> th;
      for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
      for (auto& x : th) x.join();
      fprintf(stderr, "%d-bit drafter %s [%d x %zu cols %d..%d]: relative error %.4f\n", bits, n.c_str(), Nt, ld, col0, col0 + K, ref2 > 0 ? sqrt(err2 / ref2) : 0.0);
      row0 += Nt;
    }
    TMat m; m.N = N; m.K = K; m.wf = bits == 3 ? 4 : 5; m.drafter_requant = true;
    m.w4 = upload<uint4>(W.data(), W.size() / 16); m.s4 = (__half*)upload<uint8_t>(S.data(), S.size()); m.rs = upload<float>(RS.data(), RS.size());
    return m;
  }
  // H5_DRAFT_WFMT=nvfp4, or a comma list GROUP=FORMAT with FORMAT in nvfp4|tern|b3|b2 (unlisted groups stay NVFP4).
  // Groups: gu (gate/up), down, fc (capture fusion, all layouts), attn (q/k/v/o and the fused k/v), conv
  // (attention and MLP conv kernel projections), sel (selector hidden projection).
  enum class DFmt { NVFP4, Tern, B3, B2 };
  struct DraftFormats { DFmt gu = DFmt::NVFP4, down = DFmt::NVFP4, fc = DFmt::NVFP4, attn = DFmt::NVFP4, conv = DFmt::NVFP4, sel = DFmt::NVFP4; };
  static DraftFormats draft_formats() {
    DraftFormats f;
    const char* v = getenv("H5_DRAFT_WFMT");
    if (!v) throw std::runtime_error("H5_DRAFT_WFMT must be set: nvfp4 or GROUP=FORMAT[,...] (groups gu,down,fc,attn,conv,sel; formats nvfp4,tern,b3,b2)");
    if (!strcmp(v, "nvfp4")) return f;
    std::string all = v; size_t p0 = 0;
    while (p0 <= all.size()) {
      const size_t p1 = std::min(all.find(',', p0), all.size());
      const std::string item = all.substr(p0, p1 - p0); const size_t eq = item.find('=');
      if (eq == std::string::npos) throw std::runtime_error("H5_DRAFT_WFMT items must be GROUP=FORMAT, got '" + item + "'");
      const std::string g = item.substr(0, eq), fs = item.substr(eq + 1);
      DFmt fm;
      if (fs == "nvfp4") fm = DFmt::NVFP4; else if (fs == "tern") fm = DFmt::Tern; else if (fs == "b3") fm = DFmt::B3; else if (fs == "b2") fm = DFmt::B2;
      else throw std::runtime_error("H5_DRAFT_WFMT: unknown format '" + fs + "' (nvfp4, tern, b3, b2)");
      if (g == "gu") f.gu = fm; else if (g == "down") f.down = fm; else if (g == "fc") f.fc = fm;
      else if (g == "attn") f.attn = fm; else if (g == "conv") f.conv = fm; else if (g == "sel") f.sel = fm;
      else throw std::runtime_error("H5_DRAFT_WFMT: unknown group '" + g + "' (gu, down, fc, attn, conv, sel)");
      p0 = p1 + 1;
    }
    return f;
  }
  float* bf16_f32(const SafeT& st, const std::string& n) {
    std::vector<int64_t> sh; const uint16_t* p = st.t(n, sh);
    size_t cnt = 1; for (auto d : sh) cnt *= d;
    std::vector<float> v(cnt); for (size_t i = 0; i < cnt; i++) v[i] = bf2f(p[i]);
    return upload<float>(v.data(), cnt);
  }
  void draft_load(const std::string& path) {
    if (dspark_mode) {
      if (!binary || draft_nest || lzd_w) throw std::runtime_error("native DSpark BF16 requires binary target, no DFlash nesting or draft LZ");
      if (df.loaded || native_dspark) throw std::runtime_error("drafter already loaded");
      native_dspark = std::make_unique<NativeDspark>(path, max_slots, ctx, max_rows, df_proposals, st);
      std::copy(native_dspark->capture_layers().begin(), native_dspark->capture_layers().end(), cap_layers);
      return;
    }
    double t0 = now_s();
    SafeT st; st.open(path);
    const DraftFormats fmt = draft_formats();
    auto mat = [&](DFmt f, const std::vector<std::string>& names, int col0 = 0, int Kc = 0) {
      switch (f) {
        case DFmt::NVFP4: return nvfp4_tmat(st, names, col0, Kc);
        case DFmt::Tern: return ternary_tmat(st, names, col0, Kc);
        case DFmt::B3: return lowbit_tmat(st, names, 3, col0, Kc);
        case DFmt::B2: return lowbit_tmat(st, names, 2, col0, Kc);
      }
      throw std::runtime_error("unhandled drafter weight format");
    };
    for (int j = 0; j < NCAP; j++) df.fc[j] = mat(fmt.fc, {"fc.weight"}, j * D, D);
    df.fc_all = mat(fmt.fc, {"fc.weight"});
    if (df.fc_all.K != NCAP * D || df.fc_all.N != D) throw std::runtime_error("drafter fc.weight must be D x NCAP*D");
    {
      std::vector<std::string> kv_names;
      for (int l = 0; l < DF_LAYERS; l++) { const std::string p = "layers." + std::to_string(l) + ".self_attn."; kv_names.push_back(p + "k_proj.weight"); kv_names.push_back(p + "v_proj.weight"); }
      df.kv_all = mat(fmt.attn, kv_names);
    }
    df.hproj = mat(fmt.sel, {"candidate_selector.hidden_projection.weight"});
    df.hidden_norm = bf16_f32(st, "hidden_norm.weight"); df.norm = bf16_f32(st, "norm.weight");
    std::vector<int64_t> sh;
    df.pred = upload<__nv_bfloat16>(st.t("candidate_selector.predecessor_codebook", sh), (size_t)VOCAB * DF_RANK);
    df.succ = upload<__nv_bfloat16>(st.t("candidate_selector.successor_codebook", sh), (size_t)VOCAB * DF_RANK);
    for (int l = 0; l < DF_LAYERS; l++) {
      const std::string p = "layers." + std::to_string(l) + ".";
      DLayer& y = df.L[l];
      y.qkv = mat(fmt.attn, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
      y.kv = mat(fmt.attn, {p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
      y.o = mat(fmt.attn, {p + "self_attn.o_proj.weight"});
      y.gu = mat(fmt.gu, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
      y.down = mat(fmt.down, {p + "mlp.down_proj.weight"});
      y.akp = mat(fmt.conv, {p + "attention_conv.kernel_projection.weight"});
      y.mkp = mat(fmt.conv, {p + "mlp_conv.kernel_projection.weight"});
      y.in_norm = bf16_f32(st, p + "input_layernorm.weight"); y.post_norm = bf16_f32(st, p + "post_attention_layernorm.weight");
      y.q_norm = bf16_f32(st, p + "self_attn.q_norm.weight"); y.k_norm = bf16_f32(st, p + "self_attn.k_norm.weight");
      y.abase = bf16_f32(st, p + "attention_conv.base_kernel"); y.mbase = bf16_f32(st, p + "mlp_conv.base_kernel");
    }
    const size_t R = max_rows;
    const size_t ring = (size_t)max_slots * DF_LAYERS * DF_NKV * DF_RING * DF_HD;
    df.kc = dalloc<__half>(ring); df.vc = dalloc<__half>(ring);
    df.th = dalloc<float>(R * D); df.dkv = dalloc<float>(R * 2 * DF_KV); df.dkv_all = dalloc<float>((size_t)DF_INGEST_FUSED_ROWS * DF_LAYERS * 2 * DF_KV); df.xd = dalloc<float>(R * D); df.xn = dalloc<float>(R * D);
    df.cbuf = dalloc<float>(R * D); df.tmpo = dalloc<float>(R * D); df.dyn = dalloc<float>(R * DF_KPROJ); df.dqkv = dalloc<float>(R * (DF_Q + 2 * DF_KV));
    df.qr = dalloc<float>(R * DF_Q); df.dy = dalloc<float>(R * DF_Q); df.dgu = dalloc<float>(R * GU); df.hp = dalloc<float>(R * DF_RANK);
    df.topv = dalloc<float>(R * DF_TOPK); df.topi = dalloc<int>(R * DF_TOPK);
    df.topk_scratch = {dalloc<float>(R * TOPK_PARTS_PER_ROW), dalloc<int>(R * TOPK_PARTS_PER_ROW), max_rows};
    df.sample_scores = dalloc<double>(df_proposals * DF_TOPK);
    df.meta = dalloc<int>(R * 16); CK(cudaMallocHost(&df.h_meta, R * 16 * 4)); df.d_drafts = dalloc<int>(R); CK(cudaMallocHost(&df.h_drafts, R * 4));
    df.d_conf = dalloc<float>(R); CK(cudaMallocHost(&df.h_conf, R * 4));
    df.d_alt = dalloc<int>(R); CK(cudaMallocHost(&df.h_alt, R * 4));
    df.d_branch = dalloc<int>(R); CK(cudaMallocHost(&df.h_branch, R * 4));
    df.d_alt_conf = dalloc<float>(R); CK(cudaMallocHost(&df.h_alt_conf, R * 4));
    if (const char* vp = getenv("H5_DRAFT_VOCAB")) draft_vocab_load(vp);
    df.loaded = true;
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "dflash2 drafter loaded in %.1f s (NVFP4), GPU free %.2f GB\n", now_s() - t0, fr / 1e9);
  }

  // One batched drafter call. Stream i ingests n_ing[i] captured rows starting at hcap row ing_row0[i]
  // (positions ing_pos0[i]...) and, if anchor[i] >= 0, drafts a block anchored at position start[i].
  // drafts: [streams with a block][df_proposals]; query rows include one anchor.
  int draft_step(int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0, const int* anchor, const int* start, int* drafts,
                 int* alternatives = nullptr, int* branches = nullptr, int fork_span = 3) {
    last_logits_rows = 0;
    if (graph_memory_failed != GraphMemoryError::None) return (int)graph_memory_failed;
    if (native_dspark) {
      if (alternatives) throw std::runtime_error("native DSpark does not implement DFlash two-path proposals");
      if (ns < 1 || ns > max_slots || !slot || !n_ing || !ing_row0 || !ing_pos0 || !anchor || !start) return -1;
      for (int i = 0; i < ns; ++i) {
        if (n_ing[i] < 0 || ing_row0[i] < 0 || (long long)ing_row0[i] + n_ing[i] > captured_rows) return -3;
        for (int j = 0; j < n_ing[i]; ++j) if (h_slot[ing_row0[i]+j] != slot[i] || h_pos[ing_row0[i]+j] != ing_pos0[i]+j) return -4;
      }
      native_dspark->run(hcap, captured_rows, ns, slot, n_ing, ing_row0, ing_pos0, anchor, start, drafts);
      return 0;
    }
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
    df.completed_block_rows = 0; df.completed_prediction_rows = 0;
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
    clear_a8_error();
    CK(cudaMemcpyAsync(df.meta, h, M * 16 * 4, cudaMemcpyHostToDevice, st));
    const int *d_ir = df.meta, *d_is = df.meta + M, *d_ip = df.meta + 2 * M, *d_bt = df.meta + 3 * M, *d_bs = df.meta + 4 * M, *d_bp = df.meta + 5 * M, *d_r7 = df.meta + 6 * M, *d_an = df.meta + 7 * M;
    const QBlk* d_qbd = (const QBlk*)(df.meta + 8 * M);
    ph("d_ingest");
    // H5_DRAFT_INGEST_FUSE=1 (read at capture), at most DF_INGEST_FUSED_ROWS ingested rows: one prep and
    // one GEMM over all captured layers (K = NCAP*D) instead of five of each, then one GEMM for every
    // drafter layer's k/v and one attention-prep launch over all layers. Same math; the fp32
    // accumulation order of fc and k/v changes.
    const char* ifm = getenv("H5_DRAFT_INGEST_FUSE");
    if (ifm && strcmp(ifm, "0") && strcmp(ifm, "1")) throw std::runtime_error("H5_DRAFT_INGEST_FUSE must be 0 or 1");
    if (Ni > 0 && ifm && ifm[0] == '1' && Ni <= std::min(DF_INGEST_FUSED_ROWS, std::max(8, df_proposals + 1))) {   // P7 keeps the 8-row plan
      prep(hcap, nullptr, NCAP * D, nullptr, nullptr, NCAP * D, P_QUANT, Ni, nullptr, nullptr, 0, d_ir);
      gemm(df.fc_all, Ni, df.th, D, false);
      prep(df.th, nullptr, D, df.hidden_norm, nullptr, D, P_NORM | P_QUANT, Ni);
      gemm(df.kv_all, Ni, df.dkv_all, DF_LAYERS * 2 * DF_KV, false);
      DAttnPreArgs a{df.dkv_all, DF_LAYERS * 2 * DF_KV, 0, 0, nullptr, nullptr, d_is, d_ip, nullptr, df.kc, df.vc, 0};
      a.layers = DF_LAYERS; for (int l = 0; l < DF_LAYERS; l++) a.kn[l] = df.L[l].k_norm;
      launch_dattn_pre(a, Ni, st);
    }
    else if (Ni > 0) {
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
    // H5_DRAFT_PRECLEAR=1 (read at capture): each block-layer prep zeroes the next split-K GEMM's
    // output, replacing the graph's clear kernels, which also broke the PDL chain before every GEMM.
    const char* pcm = getenv("H5_DRAFT_PRECLEAR");
    if (pcm && strcmp(pcm, "0") && strcmp(pcm, "1")) throw std::runtime_error("H5_DRAFT_PRECLEAR must be 0 or 1");
    const bool pc = pcm && pcm[0] == '1';
    // H5_DRAFT_DCONV_PREP=1 (read at capture): the conv that feeds a quantize-only prep is computed inside
    // that prep (P_DCONV), removing the conv launch and its intermediate buffer; same arithmetic.
    const char* dcm = getenv("H5_DRAFT_DCONV_PREP");
    if (dcm && strcmp(dcm, "0") && strcmp(dcm, "1")) throw std::runtime_error("H5_DRAFT_DCONV_PREP must be 0 or 1");
    const bool dcp = dcm && dcm[0] == '1';
    ph("d_layers");
    if (binary) launch_embed_q1(tok_embd, d_bt, tmp, B, st); else launch_embed(tok_embd, d_bt, tmp, B, st);
    prep(tmp, nullptr, D, nullptr, signs5120, D, P_HADAMARD | P_SIGN_AFTER | P_STORE_F32, B, df.xd, nullptr, D);
    for (int l = 0; l < DF_LAYERS; l++) {
      DLayer& y = df.L[l];
      ph("d_prep");
      prep(df.xd, nullptr, D, y.in_norm, nullptr, D, P_NORM | P_OUT_NORM | P_QUANT, B, nullptr, df.xn, D, nullptr, pc ? df.dyn : nullptr, pc ? DF_KPROJ : 0);
      ph("d_gemm");
      gemm(y.akp, B, df.dyn, DF_KPROJ, false, 0, pc);
      ph("d_conv");
      if (dcp) { ph("d_prep"); prep_dconv(df.xn, df.dyn, 0, y.abase, B, block_rows, pc ? df.dqkv : nullptr, pc ? DF_Q + 2 * DF_KV : 0); }
      else {
        launch_dconv(df.xn, df.dyn, 0, y.abase, df.cbuf, 0, B, block_rows, st);
        ph("d_prep");
        prep(df.cbuf, nullptr, D, nullptr, nullptr, D, P_QUANT, B, nullptr, nullptr, 0, nullptr, pc ? df.dqkv : nullptr, pc ? DF_Q + 2 * DF_KV : 0);
      }
      ph("d_gemm");
      gemm(y.qkv, B, df.dqkv, DF_Q + 2 * DF_KV, false, 0, pc);
      DAttnPreArgs a{df.dqkv, DF_Q + 2 * DF_KV, DF_Q, 1, y.q_norm, y.k_norm, d_bs, d_bp, df.qr, df.kc, df.vc, l};
      ph("d_attn");
      launch_dattn_pre(a, B, st);
      AttnTcArgs t{df.qr, nullptr, df.kc, df.vc, nullptr, nullptr, l, DF_RING, d_bp, d_qbd, part_o, part_ml, df.dy};
      t.nlayer = DF_LAYERS; t.nkv = DF_NKV; t.hd = DF_HD; t.noncausal = 1; t.window = df_window; t.ring = DF_RING; t.tps = tps;
      t.draft_block_rows = block_rows; t.draft_nest = draft_nest;
      launch_attn_tc(t, nqb, nsp, B, 2, st);
      ph("d_prep");
      prep(df.dy, nullptr, DF_Q, nullptr, nullptr, DF_Q, P_QUANT, B, nullptr, nullptr, 0, nullptr, pc ? df.tmpo : nullptr, pc ? D : 0);
      ph("d_gemm");
      gemm(y.o, B, df.tmpo, D, false, 0, pc);
      ph("d_conv");
      launch_dconv(df.tmpo, df.dyn, 1, y.abase, df.xd, 1, B, block_rows, st);
      ph("d_prep");
      prep(df.xd, nullptr, D, y.post_norm, nullptr, D, P_NORM | P_OUT_NORM | P_QUANT, B, nullptr, df.xn, D, nullptr, pc ? df.dyn : nullptr, pc ? DF_KPROJ : 0);
      ph("d_gemm");
      gemm(y.mkp, B, df.dyn, DF_KPROJ, false, 0, pc);
      ph("d_conv");
      if (dcp) { ph("d_prep"); prep_dconv(df.xn, df.dyn, 0, y.mbase, B, block_rows, pc ? df.dgu : nullptr, pc ? GU : 0); }
      else {
        launch_dconv(df.xn, df.dyn, 0, y.mbase, df.cbuf, 0, B, block_rows, st);
        ph("d_prep");
        prep(df.cbuf, nullptr, D, nullptr, nullptr, D, P_QUANT, B, nullptr, nullptr, 0, nullptr, pc ? df.dgu : nullptr, pc ? GU : 0);
      }
      ph("d_gemm");
      gemm(y.gu, B, df.dgu, GU, false, 0, pc);
      ph("d_prep");
      prep(df.dgu, df.dgu + FF, GU, nullptr, nullptr, FF, P_SILU_MUL | P_QUANT, B, nullptr, nullptr, 0, nullptr, pc ? df.tmpo : nullptr, pc ? D : 0);
      ph("d_gemm");
      gemm(y.down, B, df.tmpo, D, false, 0, pc);
      ph("d_conv");
      launch_dconv(df.tmpo, df.dyn, 1, y.mbase, df.xd, 1, B, block_rows, st);
    }
    ph("d_head");
    // Capture-time diagnostic only: independently exclude the incoming prep/GEMM edge
    // and outgoing GEMM/shared-activation-reuse edge without disabling all drafter PDL.
    const char* hpf = getenv("H5_DRAFT_HPROJ_FENCE");
    const bool hp_before = hpf && (!strcmp(hpf, "before") || !strcmp(hpf, "both"));
    const bool hp_after = hpf && (!strcmp(hpf, "after") || !strcmp(hpf, "both"));
    if (hpf && strcmp(hpf, "none") && !hp_before && !hp_after)
      throw std::runtime_error("H5_DRAFT_HPROJ_FENCE must be none, before, after or both");
    prep(df.xd, nullptr, D, df.norm, nullptr, D, P_NORM | P_QUANT, R7, nullptr, nullptr, 0, d_r7);
    if (hp_before) CK(launch_draft_dependency_fence(st));
    gemm(df.hproj, R7, df.hp, DF_RANK, false);
    if (hp_after) CK(launch_draft_dependency_fence(st));
    ph("d_headgemm");
    {
      // The drafter projections are NVFP4, but its candidate head is the shared
      // target head. Prepare activations in that head's configured precision.
      Force target_precision(force_fp4, false);
      prep(df.xd, nullptr, D, df.norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R7, nullptr, nullptr, 0, d_r7);
      const bool pruned = df.vhead.w4 != nullptr || df.vhead.w1 != nullptr;
      for (int r0 = 0; r0 < R7; r0 += 128) {
        const int rr = std::min(128, R7 - r0);
        if (pruned) gemm(df.vhead, rr, df.vlogits, df.vocab_n, false, r0);
        else gemm(head, rr, logits, VOCAB, false, r0);
        ph("d_topk");
        const TopkScratch topk{df.topk_scratch.values + (size_t)r0 * TOPK_PARTS_PER_ROW,
                             df.topk_scratch.indices + (size_t)r0 * TOPK_PARTS_PER_ROW, max_rows - r0};
        launch_topk(pruned ? df.vlogits : logits, df.vocab_n, pruned ? df.vmap : nullptr, df.topv + (size_t)r0 * DF_TOPK, df.topi + (size_t)r0 * DF_TOPK, rr, topk, st);
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
    if (lzd_w) {
      if (alternatives) throw std::runtime_error("H5_DRAFT_LZ excludes the two-path drafter");
      CK(cudaMemcpyAsync(d_lzctx, h_lzctx, (size_t)nb * lzd_w * sizeof(int), cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_lzlen, h_lzlen, nb * sizeof(int), cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_lzalpha, h_lzalpha, nb * sizeof(float), cudaMemcpyHostToDevice, st));
    }
    if (alternatives) launch_selector_two(df.pred, df.succ, df.hp, df.topv, df.topi, d_an, df.d_drafts, df.d_conf,
        df.d_alt, df.d_alt_conf, df.d_branch, fork_span, nb, st);
    else launch_selector(df.pred, df.succ, df.hp, df.topv, df.topi, d_an, df.d_drafts, df.d_conf, nb, df_proposals, draft_lz(), st);
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
    read_a8_error();
    CK(cudaStreamSynchronize(st));
    require_a8_valid();
    if (nb > 0) {
      memcpy(drafts, df.h_drafts, R7 * 4);
      if (alternatives) memcpy(alternatives, df.h_alt, R7 * 4);
      if (alternatives && branches) memcpy(branches, df.h_branch, nb * 4);
    }
    CK(cudaGetLastError());
    if (nb > 0) { df.completed_block_rows = B; df.completed_prediction_rows = R7; }
    prof_flush();
    return 0;
  }

  // CUDA graphs: a step's whole launch sequence is captured once per launch shape and replayed.
  // The shape (rows, segments, query blocks, attention splits, ...) fixes every grid and by-value
  // argument; per-step data (tokens, positions, segments) lives in pinned host buffers that the
  // captured copies read at replay. A recurring small shape is captured on its third sighting;
  // earlier sightings execute eagerly (and tune any GEMMs not covered at startup).
  struct GKey { int kind, a, b, c, d, e, f, g; int single = 0; bool operator<(const GKey& o) const { return std::tie(kind, a, b, c, d, e, f, g, single) < std::tie(o.kind, o.a, o.b, o.c, o.d, o.e, o.f, o.g, o.single); } };
  bool use_graphs = !getenv("H5_NOGRAPH");
  float draft_margin = 0.f;
  int relax_stats = 0;
  int adj_k = 0;
  int lzd_w = 0, lzd_b = 0, lzd_form = 0;   // H5_DRAFT_LZ
  int *d_lzctx = nullptr, *h_lzctx = nullptr, *d_lzlen = nullptr, *h_lzlen = nullptr;
  float *d_lzalpha = nullptr, *h_lzalpha = nullptr;
  DraftLz draft_lz() const {
    DraftLz z;
    if (lzd_w) { z.ctx = d_lzctx; z.len = d_lzlen; z.alpha = d_lzalpha; z.w = lzd_w; z.buffer = lzd_b; z.form = lzd_form; z.log2v = log2f((float)VOCAB); }
    return z;
  }
  int2 *d_adj = nullptr, *h_adj = nullptr;
  RowStat *d_rstat = nullptr, *h_rstat = nullptr;
  std::vector<RowStat> last_rstat;   // published to batch rows after each step (greedy -1: no head row)
  int draft_nest = 0;
  int graph_max_rows = getenv("H5_GRAPH_MAX_ROWS") ? atoi(getenv("H5_GRAPH_MAX_ROWS")) : 64;
  struct GraphEntry { cudaGraphExec_t exec; size_t nodes, roots; GraphMemoryResult memory; uint64_t rewrite_ns, last_use; size_t observed_bytes; };
  enum class GraphRejection { MemoryAdmission, CaptureOutOfMemory };
  std::map<GKey, GraphEntry> graphs; std::map<GKey, int> gseen;
  std::map<GKey, GraphRejection> graph_rejections;
  uint64_t graph_clock = 0;
  static constexpr size_t graph_cache_entries = 32;
  static constexpr size_t graph_cache_bytes = 512ULL << 20;
  static constexpr size_t graph_free_reserve = 256ULL << 20;
  size_t graph_observed_bytes() const {
    size_t bytes = 0;
    for (const auto& item : graphs) bytes += item.second.observed_bytes;
    return bytes;
  }
  bool evict_graph() {
    if (graphs.empty()) return false;
    CK(cudaStreamSynchronize(st));
    auto victim = std::min_element(graphs.begin(), graphs.end(), [](const auto& a, const auto& b) {
      return a.second.last_use < b.second.last_use;
    });
    size_t before = 0, after = 0, total = 0;
    CK(cudaMemGetInfo(&before, &total));
    const size_t observed = victim->second.observed_bytes;
    CK(cudaGraphExecDestroy(victim->second.exec)); graphs.erase(victim);
    CK(cudaMemGetInfo(&after, &total));
    fprintf(stderr, "GraphResource stage=evicted cache=%zu observed_bytes=%zu free_before=%zu free_after=%zu reclaimed_bytes=%zu\n",
            graphs.size(), observed, before, after, after > before ? after - before : 0);
    return true;
  }
  void graph_resource_notice(const GKey& k, const char* stage, size_t before, size_t after, size_t total) {
    fprintf(stderr, "GraphResource stage=%s shape=%d,%d,%d,%d,%d,%d,%d,%d,%d cache=%zu observed_bytes=%zu free_before=%zu free_after=%zu total=%zu\n",
            stage, k.kind, k.a, k.b, k.c, k.d, k.e, k.f, k.g, k.single,
            graphs.size(), graph_observed_bytes(), before, after, total);
  }
  std::atomic<long> graph_hits{0}, graph_caps{0};
  H5GraphUpload graph_upload = H5_GRAPH_UPLOAD_AT_LAUNCH;
  H5GraphTiming graph_timing = H5_GRAPH_TIMING_OFF;
  H5GraphNodes graph_nodes = H5_GRAPH_NODES_NATIVE;
  GraphMemoryError graph_memory_failed = GraphMemoryError::None;
  H5GraphSample graph_sample{};
  bool graph_sample_ready = false, graph_probe_used = false;
  using ProbeClock = std::chrono::steady_clock;
  static uint64_t probe_ns(ProbeClock::time_point a, ProbeClock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  }
  static uint64_t thread_ns() {
    timespec t;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)) throw std::runtime_error("graph probe: thread clock failed");
    return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
  }
  void launch_graph(const GraphEntry& entry) {
    if (graph_timing == H5_GRAPH_TIMING_OFF) { CK(cudaGraphLaunch(entry.exec, st)); return; }
    graph_sample.nodes = entry.nodes; graph_sample.roots = entry.roots;
    graph_sample.rewrite_ns = entry.rewrite_ns;
    graph_sample.memory_copies = entry.memory.copies; graph_sample.memory_clears = entry.memory.clears;
    graph_sample.copy_bytes = entry.memory.copy_bytes; graph_sample.clear_bytes = entry.memory.clear_bytes;
    graph_sample.host_read_bytes = entry.memory.host_read_bytes; graph_sample.host_write_bytes = entry.memory.host_write_bytes;
    const auto cpu = thread_ns();
    const auto begin = ProbeClock::now();
    CK(cudaGraphLaunch(entry.exec, st));
    const auto end = ProbeClock::now();
    graph_sample.launch_thread_ns = thread_ns() - cpu;
    graph_sample.launch_wall_ns = probe_ns(begin, end);
  }
  template <class F> void run_graph(const GKey& k, F&& body) {
    graph_probe_used = true;
    if (graph_timing != H5_GRAPH_TIMING_OFF) {
      const int shape[] = {k.kind, k.a, k.b, k.c, k.d, k.e, k.f, k.g};
      std::copy(shape, shape + 8, graph_sample.shape);
    }
    auto it = graphs.find(k);
    if (it != graphs.end()) {
      it->second.last_use = ++graph_clock;
      graph_sample.route = H5_GRAPH_ROUTE_CACHED; graph_hits++; launch_graph(it->second); return;
    }
    if (k.a > graph_max_rows || graph_rejections.count(k) || gseen[k]++ < 2) { body(); return; }
    size_t free_before = 0, total = 0;
    CK(cudaMemGetInfo(&free_before, &total));
    while (graphs.size() >= graph_cache_entries || graph_observed_bytes() >= graph_cache_bytes || free_before < graph_free_reserve) {
      if (!evict_graph()) break;
      CK(cudaMemGetInfo(&free_before, &total));
    }
    if (free_before < graph_free_reserve) {
      graph_rejections.emplace(k, GraphRejection::MemoryAdmission);
      graph_resource_notice(k, "eager-memory-admission", free_before, free_before, total);
      body(); return;
    }
    struct GraphOwner {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      ~GraphOwner() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
    } owner;
    cudaGraph_t& gr = owner.graph;
    cudaGraphExec_t& ex = owner.exec;
    const cudaStream_t capture_origin = st;
    bool capturing = false;
    auto cleanup = [&]() {
      if (capturing) {
        cudaGraph_t abandoned = nullptr;
        const auto status = cudaStreamEndCapture(capture_origin, &abandoned);
        capturing = false;
        if (abandoned) cudaGraphDestroy(abandoned);
        if (status != cudaSuccess && status != cudaErrorStreamCaptureInvalidated)
          fprintf(stderr, "GraphResource capture-abort: %s\n", cudaGetErrorString(status));
      }
      pf_forked = false; pf_used = false; pf_capturing = false;
      if (ex) { cudaGraphExecDestroy(ex); ex = nullptr; }
      if (gr) { cudaGraphDestroy(gr); gr = nullptr; }
    };
    GraphMemoryResult memory{GraphMemoryError::None, cudaSuccess, 0, 0, 0, 0, 0, 0};
    uint64_t rewrite_ns = 0;
    size_t nodes = 0, roots = 0;
    const char* stage = "begin-capture";
    try {
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); capturing = true;
      stage = "capture-body";
      pf_begin_step(); body(); pf_end_step();
      stage = "end-capture";
      const auto end_status = cudaStreamEndCapture(st, &gr); capturing = false; CK(end_status);
      if (graph_nodes == H5_GRAPH_NODES_KERNEL_MEMORY) {
        const auto begin = ProbeClock::now();
        memory = graph_memory_rewrite(gr);
        rewrite_ns = probe_ns(begin, ProbeClock::now());
        if (memory.error != GraphMemoryError::None) { graph_memory_failed = memory.error; throw GraphMemoryFailure(memory); }
      }
      stage = "instantiate";
      for (;;) {
        const auto status = cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0);
        if (status != cudaErrorMemoryAllocation) { CK(status); break; }
        if (ex) { CK(cudaGraphExecDestroy(ex)); ex = nullptr; }
        if (!evict_graph()) throw CudaFailure(status, "cudaGraphInstantiate after cache eviction");
        CK(cudaMemGetInfo(&free_before, &total));
        cudaGetLastError();
      }
      CK(cudaGraphGetNodes(gr, nullptr, &nodes)); CK(cudaGraphGetRootNodes(gr, nullptr, &roots));
      // Admission must include device upload before the first effect, even with deferred-upload probe policy.
      stage = "admission-upload";
      const auto upload_begin = graph_timing != H5_GRAPH_TIMING_OFF ? ProbeClock::now() : ProbeClock::time_point{};
      CK(cudaGraphUpload(ex, st)); CK(cudaStreamSynchronize(st));
      if (graph_timing != H5_GRAPH_TIMING_OFF) graph_sample.upload_wall_ns = probe_ns(upload_begin, ProbeClock::now());
      CK(cudaGraphDestroy(gr)); gr = nullptr;
    } catch (const CudaFailure& failure) {
      cleanup();
      if (failure.status != cudaErrorMemoryAllocation) throw;
      cudaStreamCaptureStatus capture_status;
      CK(cudaStreamIsCapturing(st, &capture_status));
      if (capture_status != cudaStreamCaptureStatusNone)
        throw std::runtime_error("GraphResource capture cleanup left stream captured");
      cudaGetLastError();
      while (evict_graph()) {}
      CK(cudaStreamSynchronize(st));
      size_t free_after = 0; CK(cudaMemGetInfo(&free_after, &total));
      graph_rejections.emplace(k, GraphRejection::CaptureOutOfMemory);
      graph_resource_notice(k, stage, free_before, free_after, total);
      // Capture/instantiate/upload have not executed the body. Eager execution is its first effect.
      body(); return;
    } catch (...) { cleanup(); throw; }
    size_t free_after = 0; CK(cudaMemGetInfo(&free_after, &total));
    const size_t observed = free_before > free_after ? free_before - free_after : 0;
    if (observed > graph_cache_bytes || free_after < graph_free_reserve) {
      cleanup();
      graph_rejections.emplace(k, GraphRejection::MemoryAdmission);
      graph_resource_notice(k, "eager-after-instantiate", free_before, free_after, total);
      body(); return;
    }
    graphs.emplace(k, GraphEntry{ex, nodes, roots, memory, rewrite_ns, ++graph_clock, observed});
    ex = nullptr; graph_caps++;
    graph_resource_notice(k, "admitted", free_before, free_after, total);
    while (graph_observed_bytes() > graph_cache_bytes && graphs.size() > 1) evict_graph();
    graph_sample.route = H5_GRAPH_ROUTE_CAPTURED;
    launch_graph(graphs.at(k));
  }

  int step(int R, const int* slot, const int* tok, const int* pos, int* out, float* host_logits,
           const int* rope = nullptr, const float* visual = nullptr, HeadDemand demand = HeadDemand::all()) {
    last_logits_rows = 0;
    if (diagnostic_readout_enabled) {
      diagnostic_readout_ready = false;
      diagnostic_readout_capturing = diagnostic_readout_armed;
      diagnostic_readout_armed = false;
      if (diagnostic_readout_capturing)
        std::fill(diagnostic_readout_samples.begin(), diagnostic_readout_samples.end(), GdnReadoutSample{});
    }
    if (gdn_diagnostic) {
      if (diagnostic_environment() != GdnDiagnosticError::None || state_fusion || use_mega || gdn_single_mode() != GdnSingleMode::Off || use_graphs || rope || visual)
        return (int)GdnDiagnosticError::UnsupportedRoute;
      for (int sl = 0; sl < max_slots; ++sl)
        if (keep_slot[sl] || commit_next[sl] || pend_n[sl]) return (int)GdnDiagnosticError::UnsupportedSpeculation;
    }
    graph_probe_used = true;
    graph_sample_ready = false;
    if (graph_memory_failed != GraphMemoryError::None) return (int)graph_memory_failed;
    const bool probe = graph_timing != H5_GRAPH_TIMING_OFF;
    const auto probe_begin = probe ? ProbeClock::now() : ProbeClock::time_point{};
    if (probe) graph_sample = H5GraphSample{};
    if (R <= 0 || R > max_rows || !slot || !tok || !pos || !out) return -1;
    const auto head_error = demand.validate(R, host_logits != nullptr);
    if (head_error != H5_HEAD_OK) return head_error;
    // A sorted, unique selection of R rows is exactly All, including its graph/mega route.
    if (!demand.is_all() && demand.count(R) == R) demand = HeadDemand::all();
    const int head_rows = demand.count(R);
    if (!demand.is_all()) for (int j = 0; j < head_rows; ++j) h_head_rows[j] = demand.source(j);
    if (rope && rope_sections[0] + rope_sections[1] + rope_sections[2] + rope_sections[3] != NROT / 2) return -4;
    int nseg = 0;
    for (int i = 0; i < R; i++) {
      if (slot[i] < 0 || slot[i] >= max_slots || pos[i] < 0 || pos[i] >= ctx) return -2;
      if (tok[i] < -1 || tok[i] >= VOCAB || (tok[i] == -1 && !visual)) return -5;
      h_tok[i] = tok[i]; h_slot[i] = slot[i]; h_pos[i] = pos[i];
      if (i == 0 || slot[i] != slot[i - 1]) {
        const int sl = slot[i];
        h_segs[nseg] = Seg{sl, i, 0, std::min(commit_next[sl], pend_n[sl]), keep_slot[sl], pend_par[sl], conv_par[sl]}; nseg++;
      }
      h_segs[nseg - 1].len++;
    }
    const GdnSingleMode single_mode = gdn_single_mode();
    const char* sd = getenv("H5_GDN_SINGLE_DIAG");
    if (sd && strcmp(sd, "0") && strcmp(sd, "1")) throw std::runtime_error("H5_GDN_SINGLE_DIAG requires 0|1");
    const bool single_diag = sd && sd[0] == '1';
    validate_gdn_single(single_mode, state_format, state_fusion, getenv("H5_MEGA") && atoi(getenv("H5_MEGA")), 0);
    const bool single_gdn = single_mode != GdnSingleMode::Off && gdn_single_eligible(h_segs, nseg);
    const GdnVerifyMode verify_mode = gdn_verify_mode_from_env();
    const bool verify_gdn = verify_mode != GdnVerifyMode::Off && nseg == 1 &&
        h_segs[0].keep == 1 && (R == 8 || R == 16) && state_format == StateFormat::F16;
    if (verify_mode != GdnVerifyMode::Off && use_mega)
      throw std::runtime_error("H5_GDN_VERIFY requires the graph engine");
    if (verify_mode == GdnVerifyMode::Fused && !state_fusion)
      throw std::runtime_error("H5_GDN_VERIFY=fused requires H5_STATE_FUSION=1");
    if ((verify_mode == GdnVerifyMode::Serial || verify_mode == GdnVerifyMode::Chunk) && state_fusion)
      throw std::runtime_error("H5_GDN_VERIFY=serial|chunk requires H5_STATE_FUSION=0");
    for (int g = 0; g < nseg; g++) {
      Seg& q = h_segs[g];
      if (q.keep && q.len > pending_rows) return -3;
      pend_n[q.slot] = q.keep ? q.len : 0; commit_next[q.slot] = 0;
      if (q.keep) pend_par[q.slot] ^= 1;
      if (state_fusion) conv_par[q.slot] ^= 1;
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
    if (attn_direct) { tps = span; nsp = 1; }
    const int max_mt = (maxlen * (NH / NKV) + 15) / 16;
    double t0 = now_s();
    const auto probe_submit = probe ? ProbeClock::now() : ProbeClock::time_point{};
    if (rope) CK(cudaMemcpyAsync(d_rope, rope, (size_t)R * 4 * sizeof(int), cudaMemcpyHostToDevice, st));
    if (visual) CK(cudaMemcpyAsync(d_visual, visual, (size_t)R * D * sizeof(float), cudaMemcpyHostToDevice, st));
    auto body = [&]() {
    clear_a8_error();
    act_reset();
    CK(cudaMemcpyAsync(d_qb, h_qb, nqb * sizeof(QBlk), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_tok, h_tok, R * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_slot, h_slot, R * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_pos, h_pos, R * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d_segs, h_segs, nseg * sizeof(Seg), cudaMemcpyHostToDevice, st));
    if (!demand.is_all() && head_rows) CK(cudaMemcpyAsync(d_head_rows, h_head_rows, head_rows * sizeof(int), cudaMemcpyHostToDevice, st));
    if (adj_k) CK(cudaMemcpyAsync(d_adj, h_adj, (size_t)R * adj_k * sizeof(int2), cudaMemcpyHostToDevice, st));
    ph("embed");
    if (binary) launch_embed_q1(tok_embd, d_tok, tmp, R, st); else launch_embed(tok_embd, d_tok, tmp, R, st);
    prep(tmp, nullptr, D, nullptr, signs5120, D, P_HADAMARD | P_SIGN_AFTER | P_STORE_F32, R, x, nullptr, D);
    // The table is Hadamard-latent; mmproj output is already in the primal basis.
    if (visual) launch_visual_input(d_tok, d_visual, x, R, st);
    const char* clear_mode = getenv("H5_PREP_CLEAR");
    if (clear_mode && strcmp(clear_mode, "0") && strcmp(clear_mode, "1"))
      throw std::runtime_error("H5_PREP_CLEAR must be 0 or 1");
    const bool prep_clear = clear_mode && !strcmp(clear_mode, "1");
    if (prep_clear && !(act_fp4 || force_fp4)) throw std::runtime_error("H5_PREP_CLEAR requires FP4 activations");
    // H5_CONV_AB=1 (read per capture): on steps of at most 8 rows the conv kernel's v-head CTAs compute
    // alpha/beta from the bf16 weights before their wait, replacing the alpha/beta projection launch.
    const char* conv_ab_mode = getenv("H5_CONV_AB");
    if (conv_ab_mode && strcmp(conv_ab_mode, "0") && strcmp(conv_ab_mode, "1")) throw std::runtime_error("H5_CONV_AB must be 0 or 1");
    const bool conv_ab = conv_ab_mode && conv_ab_mode[0] == '1' && R <= 8 && !state_fusion;
    const char* ab_first_mode = getenv("H5_AB_FIRST");
    if (ab_first_mode && strcmp(ab_first_mode, "0") && strcmp(ab_first_mode, "1")) throw std::runtime_error("H5_AB_FIRST must be 0 or 1");
    const bool ab_first = ab_first_mode && ab_first_mode[0] == '1';
    // H5_TAIL_PREP=MASK (read per capture; steps of at most 8 rows): preparations run in the tail of the
    // producer instead of their own launch. Bit 1: gated-RMSNorm ssm_out prep in the GDN kernel; 2: post-norm
    // prep in the output projection; 4: SiLU-output prep in the fused gate/up; 8: the next layer's input-norm
    // prep (and the next in-projection's output clearing) in the down projection; 16: attention-output prep in
    // the attention combine. 0 = separate preps.
    const char* tail_env = getenv("H5_TAIL_PREP");
    unsigned tail_mask = 0;
    if (tail_env) {
      char* end = nullptr; const long v = strtol(tail_env, &end, 10);
      if (!end || *end || end == tail_env || v < 0 || v > 31) throw std::runtime_error("H5_TAIL_PREP must be a mask 0..31");
      tail_mask = (unsigned)v;
    }
    if (deterministic_gemm && tail_mask) throw std::runtime_error("deterministic GEMM requires H5_TAIL_PREP=0");
    validate_gdn_single(single_mode, state_format, state_fusion, false, tail_mask);
    if (R > TAIL_MAX_ROWS) tail_mask = 0;
    if (tail_mask && (!(act_fp4 || force_fp4) || f02_mode != F02Mode::Ordinary || fuse_gu_selected()))
      throw std::runtime_error("H5_TAIL_PREP requires FP4 activations, the ordinary GU route and no fused-prep GU");
    if ((tail_mask & 1) && state_fusion) throw std::runtime_error("H5_TAIL_PREP bit 1 requires the separate GDN kernel (H5_STATE_FUSION=0)");
    if ((tail_mask & 8) && (!prep_clear || ab_bf16 || conv_ab)) throw std::runtime_error("H5_TAIL_PREP bit 8 requires H5_PREP_CLEAR=1 and the FP4 alpha/beta projection");
    // H5_DOWN_CLEAR=1 (read per capture; steps of at most 8 rows, H5_PREP_CLEAR=1): the down projection's CTAs zero
    // the next in-projection (and alpha/beta) outputs after their wait, instead of the 8-CTA input-norm prep
    // doing it on the critical path. "0" = clearing in the prep.
    const char* dclear_env = getenv("H5_DOWN_CLEAR");
    if (dclear_env && strcmp(dclear_env, "0") && strcmp(dclear_env, "1")) throw std::runtime_error("H5_DOWN_CLEAR must be 0 or 1");
    const bool down_clear = dclear_env && dclear_env[0] == '1' && R <= TAIL_MAX_ROWS && prep_clear;
    // The normalised rows xn feed only the bf16/INT8 alpha/beta paths and the conv-fused alpha/beta.
    const bool need_xn = !binary && (conv_ab || ab_bf16 || !act_fp4);
    // H5_FLAG_WAIT=1 (read per capture): the GDN-norm prep starts when every GDN CTA has signalled its outputs
    // (before the state save), and the attention-output prep when every combine CTA has; both still wait for
    // their producer grid at exit. "0" = start at producer grid completion.
    const char* flag_env = getenv("H5_FLAG_WAIT");
    if (flag_env && strcmp(flag_env, "0") && strcmp(flag_env, "1")) throw std::runtime_error("H5_FLAG_WAIT must be 0 or 1");
    const bool flag_wait = flag_env && flag_env[0] == '1';
    if (flag_wait && (tail_mask & 17)) throw std::runtime_error("H5_FLAG_WAIT and H5_TAIL_PREP bits 1/16 replace the same preps");
    // H5_GEMM_FLAG=1 (read per capture; steps of 2..8 rows with the wide preps): the in/out/gate-up/down GEMMs
    // start their main loop when every CTA of the preceding prep has published (release counter), not at the
    // prep grid's completion, which waits behind the GEMM's own pre-wait weight reads; they wait for the prep
    // grid at exit instead. "0" = griddepcontrol.wait.
    const char* gflag_env = getenv("H5_GEMM_FLAG");
    if (gflag_env && strcmp(gflag_env, "0") && strcmp(gflag_env, "1")) throw std::runtime_error("H5_GEMM_FLAG must be 0 or 1");
    const bool gemm_flag_on = gflag_env && gflag_env[0] == '1' && R >= 2 && R <= TAIL_MAX_ROWS;
    if (deterministic_gemm && gemm_flag_on) throw std::runtime_error("deterministic GEMM requires H5_GEMM_FLAG=0");
    if (gemm_flag_on) {
      const char* cw = getenv("H5_PREP_CHUNK_WIDE"); const char* pw = getenv("H5_PREP_WIDE");
      if (!cw || strcmp(cw, "1") || !pw || (strcmp(pw, "1") && strcmp(pw, "multi")) || tail_mask || state_fusion || ab_first || conv_ab || ab_bf16 || !act_fp4)
        throw std::runtime_error("H5_GEMM_FLAG requires H5_PREP_CHUNK_WIDE=1, H5_PREP_WIDE=1|multi, no tail preps, separate GDN, FP4 alpha/beta after the in-projection");
    }
    if (flag_wait || gemm_flag_on) CK(cudaMemsetAsync(flag_sig, 0, (size_t)NLAYER * 8 * sizeof(unsigned), st));
    // Flag-start TailArgs for this layer's GEMM after a prep that signals slot k (expected CTA count).
    auto flag_ta = [&](int l, int k, unsigned expect) { TailArgs t{}; t.R = R; t.wait_sig = flag_sig + 8 * l + k; t.wait_expect = expect; return t; };
    // H5_PREP_EARLY=MASK (read per capture): preps that trigger the following GEMM before their own wait.
    // 1 input-norm prep (in-projection), 2 post-norm prep (gate/up), 4 SiLU prep (down), 8 GDN-norm and
    // attention-output preps (out-projection), 16 final norm (head).
    unsigned early_mask = 0;
    if (const char* em = getenv("H5_PREP_EARLY")) {
      char* end = nullptr; const long v = strtol(em, &end, 10);
      if (!end || *end || end == em || v < 0 || v > 31) throw std::runtime_error("H5_PREP_EARLY must be a mask 0..31");
      early_mask = (unsigned)v;
    }
    // H5_EARLY_K=MASK (read per capture): latency kernels that trigger their dependents before their own wait.
    // 1 conv, 2 GDN, 4 attention prep, 8 attention core, 16 attention combine.
    unsigned early_k = 0;
    if (const char* ek = getenv("H5_EARLY_K")) {
      char* end = nullptr; const long v = strtol(ek, &end, 10);
      if (!end || *end || end == ek || v < 0 || v > 31) throw std::runtime_error("H5_EARLY_K must be a mask 0..31");
      early_k = (unsigned)v;
    }
    // Capture-time best-effort L2 hints. Placement/granularity do not change the
    // target image, aligned cap, CTA geometry, or predecessor wait protocol.
    size_t pf_chain = 0;
    if (const char* pc = getenv("H5_PF_CHAIN_MB")) {
      char* end = nullptr; const double v = strtod(pc, &end);
      if (!end || *end || end == pc || !(v >= 0) || v > 64) throw std::runtime_error("H5_PF_CHAIN_MB must be 0..64");
      pf_chain = (size_t)(v * 1048576.0);
    }
    auto mat_bytes = [](const TMat& m) { return (size_t)(m.N / 32) * (m.K / 128) * (m.wf == 3 ? 832 : m.wf == 1 || m.wf == 6 ? 2048 : m.wf == 4 ? 1536 : 1024); };
    const char* pf_target = getenv("H5_PF_CHAIN_TARGET");
    if (pf_target && strcmp(pf_target, "out") && strcmp(pf_target, "gu") && strcmp(pf_target, "far-gu"))
      throw std::runtime_error("H5_PF_CHAIN_TARGET must be out, gu or far-gu");
    const bool pf_gu = pf_target && !strcmp(pf_target, "gu");
    const bool pf_far_gu = pf_target && !strcmp(pf_target, "far-gu");
    const char* pf_gran = getenv("H5_PF_CHAIN_GRANULARITY");
    if (pf_gran && strcmp(pf_gran, "pieces") && strcmp(pf_gran, "cta"))
      throw std::runtime_error("H5_PF_CHAIN_GRANULARITY must be pieces or cta");
    const h5::PfChainGranularity pf_granularity = pf_gran && !strcmp(pf_gran, "cta") ?
        h5::PfChainGranularity::CtaShare : h5::PfChainGranularity::Pieces16KiB;
    const char* pf_place = getenv("H5_PF_CHAIN_PLACEMENT");
    if (pf_place && strcmp(pf_place, "prologue") && strcmp(pf_place, "tail"))
      throw std::runtime_error("H5_PF_CHAIN_PLACEMENT must be prologue or tail");
    const h5::PfChainPlacement pf_placement = pf_place && !strcmp(pf_place, "tail") ?
        h5::PfChainPlacement::Tail : h5::PfChainPlacement::Prologue;
    if (pf_chain && state_fusion && !(single_gdn && single_mode == GdnSingleMode::Fused))
      throw std::runtime_error("H5_PF_CHAIN_MB requires separate conv or the eligible single-token fused route");
    size_t pf_intended_bytes = 0, pf_intended_requests = 0;
    auto record_pf = [&](const uint8_t* ptr, unsigned bytes, unsigned ncta) {
      if (!ptr || (uintptr_t(ptr) & 15u) || (bytes & 15u) || !ncta)
        throw std::runtime_error("prefetch chain requires aligned image, byte cap and nonempty grid");
      pf_intended_bytes += bytes;
      for (unsigned cta = 0; cta < ncta; ++cta) {
        const unsigned n = h5::pf_chain_slice(bytes, ncta, cta).bytes;
        pf_intended_requests += pf_granularity == h5::PfChainGranularity::CtaShare ? (n != 0) : (n + 16383u) / 16384u;
      }
    };
    unsigned ordered_prep = 0;
    bool ordered_r8_only = false;
    if (const char* mode = getenv("H5_ORDERED_PREP")) {
      if (!strcmp(mode, "out8")) { ordered_prep = 1; ordered_r8_only = true; }
      else if (!strcmp(mode, "out")) ordered_prep = 1;
      else if (!strcmp(mode, "down")) ordered_prep = 2;
      else if (!strcmp(mode, "both")) ordered_prep = 3;
      else if (strcmp(mode, "off")) throw std::runtime_error("H5_ORDERED_PREP requires off|out8|out|down|both");
    }
    if (ordered_prep && (!deterministic_gemm || need_xn || fuse_gu_selected() || tail_mask || gemm_flag_on || down_clear))
      throw std::runtime_error("ordered prep requires deterministic GEMM, ordinary row preps and FP4 alpha/beta");
    if (R < 2 || R > 16 || (ordered_r8_only && R != 8)) ordered_prep = 0;
    bool in_prepped = false;   // this layer's input-norm prep ran in the previous down projection's tail
    bool in_cleared = false;   // this layer's in-projection (and alpha/beta) outputs were zeroed by the previous down projection
    for (int l = 0; l < NLAYER; l++) {
      Layer& Ly = L[l];
      if (!Ly.attn) {
        ph("prep");
        const bool clear_ab = prep_clear && !ab_bf16 && act_fp4 && !conv_ab;
        const bool pc = prep_clear && !in_cleared;
        if (gemm_flag_on && !in_prepped) prep_done_sig_next = flag_sig + 8 * l + 4;
        prep_early_next = !in_prepped && (early_mask & 1);
        if (!in_prepped)
        prep(x, nullptr, D, Ly.attn_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT | (need_xn ? P_OUT_NORM : 0u), R, nullptr, need_xn ? xn : nullptr, need_xn ? D : 0, nullptr,
             pc ? proj : nullptr, pc ? GDN_IN : 0, pc && clear_ab ? ab : nullptr, pc && clear_ab ? 96 : 0);
        // H5_AB_FIRST=1 (read at capture): issue the small alpha/beta projection before the in-projection,
        // so its latency overlaps the in-projection's pre-wait weight ring instead of standing alone
        // between the in-projection and the conv. Both read the same prepared rows; conv needs both.
        auto ab_proj = [&]() {
          ph("ab");
          if (conv_ab) {}
          else if (!binary && (ab_bf16 || !act_fp4)) launch_ab(Ly.ab, xn, ab, R, st);
          else gemm(Ly.ab4, R, ab, 96, false, 0, prep_clear);
        };
        // H5_SK_AB=1 (read at capture, with H5_SK): the alpha/beta projection rides inside the stream-K
        // in-projection launch as its NVFP4 second segment; both outputs are precleared by the prep.
        const char* sk_ab_mode = getenv("H5_SK_AB");
        if (sk_ab_mode && strcmp(sk_ab_mode, "0") && strcmp(sk_ab_mode, "1")) throw std::runtime_error("H5_SK_AB must be 0 or 1");
        SkSel in_sel;
        const bool merge_ab = !deterministic_gemm && sk_ab_mode && sk_ab_mode[0] == '1' && clear_ab && !ab_first && sk_select(Ly.in, R, in_sel);
        // H5_AB_PARALLEL=1 (read at capture): the alpha/beta projection depends only on the prepared rows, so it
        // runs on a forked branch beside the in-projection and joins before the conv, off the critical path.
        const char* ab_par_mode = getenv("H5_AB_PARALLEL");
        if (ab_par_mode && strcmp(ab_par_mode, "0") && strcmp(ab_par_mode, "1")) throw std::runtime_error("H5_AB_PARALLEL must be 0 or 1");
        const bool ab_par = ab_par_mode && ab_par_mode[0] == '1' && !conv_ab && !ab_first && !merge_ab;
        if (ab_par) {
          CK(cudaEventRecord(ab_fork, st)); CK(cudaStreamWaitEvent(ab_st, ab_fork, 0));
          {
            struct StreamExchange {
              cudaStream_t& main; cudaStream_t& branch;
              StreamExchange(cudaStream_t& a, cudaStream_t& b) : main(a), branch(b) { std::swap(main, branch); }
              ~StreamExchange() { std::swap(main, branch); }
            } exchange(st, ab_st);
            ab_proj();
          }
          CK(cudaEventRecord(ab_join, ab_st));
          ph("gemm_in+ab_join");
          gemm(Ly.in, R, proj, GDN_IN, false, 0, prep_clear);
          prefetch_after_in(Ly);
          CK(cudaStreamWaitEvent(st, ab_join, 0));
        } else if (merge_ab) {
          ph("gemm_in");
          launch_sk_in_ab(Ly.in, Ly.ab4, in_sel, R, proj, GDN_IN, ab, 96);
        } else {
        if (ab_first) ab_proj();
        ph("gemm_in");
        if (gemm_flag_on && prep_clear) gemm_side(Ly.in, R, proj, GDN_IN, false, flag_ta(l, 4, R));
        else gemm(Ly.in, R, proj, GDN_IN, false, 0, prep_clear);
        prefetch_after_in(Ly);
        if (!ab_first) ab_proj();
        }
        ph("conv");
        ConvArgs c{proj, Ly.conv_w, conv_state, Ly.gl, conv_out, d_segs, pend_conv, ab, Ly.ssm_a, Ly.ssm_dt, gates};
        c.pending_rows = pending_rows; c.early = early_k & 1;
        if (pf_chain > 0) {
          const TMat& pf = pf_far_gu ? L[(l + NLAYER / 2) % NLAYER].gu : pf_gu ? Ly.gu : Ly.out;
          c.pf = (const uint8_t*)pf.w4; c.pf_bytes = (unsigned)std::min(pf_chain, mat_bytes(pf)) & ~15u;
          c.pf_granularity = pf_granularity; c.pf_placement = pf_placement;
          record_pf(c.pf, c.pf_bytes, state_fusion ? gdn_grid_ctas(nseg) : (CONV_CH / 128) * nseg);
        }
        if (conv_ab) { c.ab_w = Ly.ab; c.xn = xn; c.ab_out = ab; }
        if (!state_fusion) launch_conv(c, nseg, state_format, st);
        GdnArgs ga{conv_out, proj, ab, Ly.ssm_a, Ly.ssm_dt, Ly.ssm_norm, gdn_state, Ly.gl, y, d_segs, pend_gdn, gates};
        static const bool replay_delta = getenv("H5_GDN_DELTA") && atoi(getenv("H5_GDN_DELTA"));
        ga.replay_delta = replay_delta; ga.pending_rows = pending_rows; ga.early = early_k & 2;
        ga.q8_scale = q8_scale;
        if (flag_wait && !state_fusion) ga.sig = flag_sig + 8 * l;
        ph("gdn");
        if (diagnostic_state) {
          GdnArgs shadow = ga;
          shadow.state = diagnostic_state; shadow.y = diagnostic_y;
          shadow.sig = nullptr; shadow.early = false;
          launch_gdn(shadow, nseg, diagnostic_format, st);
        }
        if (tail_mask & 1) {
          TailArgs ta = tail_args(y, VDIM, VDIM, R, P_PERMUTE_GDN | P_GDN_NORM | P_SIGN | P_HADAMARD | P_QUANT, 0);
          ta.mul = signs6144; ta.gnorm_w = Ly.ssm_norm; ta.gz = proj + CONV_CH; ta.gz_ld = GDN_IN;
          launch_gdn_tail(ga, nseg, state_format, ta, st);
          act_flip();
        } else {
        if (single_gdn) {
          if (single_diag) {
            std::vector<Seg> observed(nseg);
            CK(cudaMemcpyAsync(observed.data(), d_segs, nseg * sizeof(Seg), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            for (int si = 0; si < nseg; ++si) {
              const Seg &host = h_segs[si], &device = observed[si];
              const bool match = !memcmp(&host, &device, sizeof(Seg));
              if (l == 0 || !match)
                fprintf(stderr, "single-diag layer=%d segment=%d ptr=%p host=[%d,%d,%d,%d,%d,%d,%d] device=[%d,%d,%d,%d,%d,%d,%d] match=%d\n",
                    l, si, (const void*)ga.segs, host.slot, host.start, host.len, host.commit, host.keep, host.par, host.conv_par,
                    device.slot, device.start, device.len, device.commit, device.keep, device.par, device.conv_par, match);
              if (!match) throw std::runtime_error("single GDN metadata differs between host selection and device launch");
            }
          }
          if (single_mode == GdnSingleMode::Fused) launch_conv_gdn_single(c, ga, nseg, state_format, st);
          else launch_gdn_single(ga, nseg, state_format, st);
        } else if (verify_gdn && verify_mode == GdnVerifyMode::Fused)
          launch_conv_gdn_verify(c, ga, nseg, R, state_format, st);
        else if (state_fusion) launch_conv_gdn(c, ga, nseg, state_format, st);
        else if (verify_gdn) launch_gdn_verify(ga, nseg, R, verify_mode, state_format, st);
        else launch_gdn(ga, nseg, state_format, st);
        observe_gdn_readout(l, nseg);
        ph("prep");
        {
          PrepArgs pa{}; pa.x = y; pa.ldx = VDIM; pa.signs = signs6144; pa.n = VDIM; pa.flags = P_PERMUTE_GDN | P_GDN_NORM | P_SIGN | P_HADAMARD | P_QUANT;
          pa.xq = xq; pa.xs = xs; if (act_fp4 || force_fp4) { pa.q4 = q4; pa.s4 = s4; pa.g4 = g4; }
          pa.gnorm_w = Ly.ssm_norm; pa.gz = proj + CONV_CH; pa.gz_ld = GDN_IN;
          if (ga.sig) { pa.sig = ga.sig; pa.sig_expect = (unsigned)gdn_grid_ctas(nseg); }
          if (gemm_flag_on) pa.done_sig = flag_sig + 8 * l + 5;
          pa.early = early_mask & 8;
          target_prep(pa);
          launch_prep(pa, R, st);
          finish_target_prep(VDIM, R);
        }
        }
      } else {
        ph("prep");
        const bool pc = prep_clear && !in_cleared;
        if (gemm_flag_on && !in_prepped) prep_done_sig_next = flag_sig + 8 * l + 4;
        prep_early_next = !in_prepped && (early_mask & 1);
        if (!in_prepped)
        prep(x, nullptr, D, Ly.attn_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R, nullptr, nullptr, 0, nullptr,
             pc ? proj : nullptr, pc ? ATT_IN : 0);
        ph("gemm_in");
        if (gemm_flag_on && prep_clear) gemm_side(Ly.in, R, proj, ATT_IN, false, flag_ta(l, 4, R));
        else gemm(Ly.in, R, proj, ATT_IN, false, 0, prep_clear);
        prefetch_after_in(Ly);
        ph("attn_pre");
        AttnArgs a{proj, Ly.q_norm, Ly.k_norm, kc, vc, ksc, vsc, Ly.al, ctx, d_slot, d_pos, qrot, y};
        a.vt = !getenv("H5_ATTN8") || atoi(getenv("H5_ATTN8"));
        a.early = early_k & 4;
        if (pf_chain > 0) {
          const TMat& pf = pf_far_gu ? L[(l + NLAYER / 2) % NLAYER].gu : pf_gu ? Ly.gu : Ly.out;
          a.pf = (const uint8_t*)pf.w4; a.pf_bytes = (unsigned)std::min(pf_chain, mat_bytes(pf)) & ~15u;
          a.pf_granularity = pf_granularity; a.pf_placement = pf_placement;
          record_pf(a.pf, a.pf_bytes, (NH + NKV) * R);
        }
        a.rope_pos = rope ? d_rope : nullptr;
        for (int j = 0; j < 4; j++) a.rope_sections[j] = rope_sections[j];
        launch_attn_pre(a, R, st);
        ph("attn");
        AttnTcArgs t{qrot, proj, kc, vc, ksc, vsc, Ly.al, ctx, d_pos, d_qb, part_o, part_ml, y};
        t.tps = tps; t.direct = attn_direct;
        if ((tail_mask & 16) && kv8 && !attn_direct) {
          launch_attn_tc(t, nqb, nsp, 0, max_mt, st); ph("attn_comb");
          TailArgs ta = tail_args(y, ATTN_OUT, ATTN_OUT, R, P_SIGN | P_HADAMARD | P_QUANT, 1);
          ta.mul = signs6144;
          launch_attn_combine_tail(t, nsp, R, ta, st);
          act_flip();
        } else {
        const bool comb_flag = flag_wait && kv8 && !attn_direct;
        if (comb_flag) t.sig = flag_sig + 8 * l + 1;
        if (kv8 && !attn_direct) {
          t.early = early_k & 8; launch_attn_tc(t, nqb, nsp, 0, max_mt, st); ph("attn_comb");
          t.early = early_k & 16; launch_attn_combine(t, nsp, R, st); t.early = false;
        }
        else launch_attn_tc(t, nqb, nsp, R, max_mt, st);
        ph("prep");
        if (gemm_flag_on) prep_done_sig_next = flag_sig + 8 * l + 5;
        prep_early_next = early_mask & 8;
        prep(y, nullptr, ATTN_OUT, nullptr, signs6144, ATTN_OUT, P_SIGN | P_HADAMARD | P_QUANT, R, nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, 0,
             comb_flag ? t.sig : nullptr, comb_flag ? (unsigned)(R * NH) : 0u);
        }
      }
      in_prepped = false; in_cleared = false;
      ph("gemm_out");
      const bool gu_prepped = (tail_mask & 2) || (ordered_prep & 1);
      if (ordered_prep & 1) {
        gemm_residual_prep(Ly.out, R, x, Ly.post_norm, early_mask & 2);
      } else if (gu_prepped) {
        TailArgs ta = tail_args(x, D, D, R, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, 2);
        ta.mul = Ly.post_norm_s;
        gemm_tail_norm(Ly.out, R, x, D, ta);
      } else if (gemm_flag_on) gemm_side(Ly.out, R, x, D, true, flag_ta(l, 5, (unsigned)(R * (Ly.attn ? ATTN_OUT : VDIM) / 1024)));
      else gemm(Ly.out, R, x, D, true);
      const auto f02_route = select_f02_route(f02_mode, R, Ly.gu.wf, Ly.gu.il, Ly.gu.mean_scale != nullptr);
      if (f02_route == F02Route::Incompatible) throw std::runtime_error("incompatible F02 GU payload");
      const int fuse_ci = f02_route == F02Route::H32 ? -1 : fused_gu_config(Ly.gu, R);
      if (fuse_ci >= 0 && gu_prepped) throw std::runtime_error("H5_TAIL_PREP bit 2 cannot feed the fused-prep GU");
      bool compound_done = false;
      if (fuse_ci >= 0) {
        ph("gemm_gu_fused");
#ifdef H5_PROFILE_SCALE_CARRIER
        CK(launch_scale_carrier_fused_gu(fuse_ci, R, Ly.gu.K, Ly.gu.w4, Ly.gu.N, proj, FF, Ly.gu.mean_scale, x, D, Ly.post_norm, signs5120, st));
#else
        throw std::runtime_error("GU fusion requires a scale-carrier build");
#endif
        ph("prep");
        prep(proj, nullptr, FF, nullptr, signs17408, FF, P_SIGN | P_HADAMARD | P_QUANT, R);
      } else {
      // H5_GU_SPLIT=CFG,KS (read at capture; steps of at most 8 rows, interleaved carrier GU): instead of
      // the fused SiLU GEMM (one CTA per 256 columns, no split-K: 136 CTAs on 170 SMs), run the plain
      // split-K carrier GEMM over the interleaved image, then SiLU*up inside the following chunk prep.
      const auto gs = gu_split(Ly.gu, R, f02_route);
      ph("prep");
      // The split-K GU accumulates into gu_acc, which is all-zero between uses: allocation zeroes it and the
      // SiLU prep zeroes every element it consumes (P_CONSUME_CLEAR), so no separate clear is needed.
      if (gemm_flag_on && !gu_prepped) prep_done_sig_next = flag_sig + 8 * l + 2;
      prep_early_next = !gu_prepped && (early_mask & 2);
      if (!gu_prepped) prep(x, nullptr, D, Ly.post_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, R);
      ph("gemm_gu");
      compound_done = whole_ffn_rows(R);
      if (compound_done) {
        compound_ffn(Ly, R);
      } else {
      if (a8_gu_fused && a8_prepared) {
        compose_gu(Ly.gu, R);
      } else if ((tail_mask & 4) && !gs.on && f02_route != F02Route::H32 && Ly.gu.il) {
        TailArgs ta = tail_args(proj, FF, FF, R, P_SIGN | P_HADAMARD | P_QUANT, 3);
        ta.mul = signs17408;
        gemm_silu_tail(Ly.gu, R, proj, FF, ta);
      } else if (gs.on) {
#ifdef H5_PROFILE_SCALE_CARRIER
        if (deterministic_gemm) launch_projection(Ly.gu, gs.cfg, gs.ks, q4, s4, g4, R, gu_acc, GU, false);
        else launch_carrier_projection(Ly.gu, gs.cfg, gs.ks, 2, q4, s4, g4, R, gu_acc, GU);
#else
        throw std::runtime_error("H5_GU_SPLIT requires a scale-carrier build");
#endif
        ph("prep");
        prep(gu_acc, nullptr, GU, nullptr, signs17408, FF, P_SILU_IL | P_CONSUME_CLEAR | P_SIGN | P_HADAMARD | P_QUANT, R);
      } else if (f02_route == F02Route::H32) {
        f02_launch(Ly.gu, R, proj);
      } else if (gu_plain(Ly.gu, R)) {
        // H5_GU_PLAIN_ROWS=N: at N or more rows the GU runs as the plain carrier GEMM over the interleaved image
        // (every tile shape eligible, not only the WG=2 SiLU-epilogue ones) and SiLU*up moves into the prep.
        gemm(Ly.gu, R, proj, GU, false);
        ph("prep");
        prep(proj, nullptr, GU, nullptr, signs17408, FF, P_SILU_IL | P_SIGN | P_HADAMARD | P_QUANT, R);
      } else if (Ly.gu.il && (act_fp4 || force_fp4)) {
        if (gemm_flag_on) gemm_silu_side(Ly.gu, R, proj, FF, flag_ta(l, 2, R));
        else gemm_silu(Ly.gu, R, proj, FF);
        ph("prep");
        if (gemm_flag_on) prep_done_sig_next = flag_sig + 8 * l + 3;
        prep_early_next = early_mask & 4;
        prep(proj, nullptr, FF, nullptr, signs17408, FF, P_SIGN | P_HADAMARD | P_QUANT, R);
      } else {
        gemm(Ly.gu, R, proj, GU, false);
        ph("prep");
        prep(proj, proj + FF, GU, nullptr, signs17408, FF, P_SILU_MUL | P_SIGN | P_HADAMARD | P_QUANT, R);
      }
      }
      }
      ph("gemm_down");
      if (!compound_done) {
      if ((ordered_prep & 2) && l + 1 < NLAYER) {
        gemm_residual_prep(Ly.down, R, x, L[l + 1].attn_norm, early_mask & 1);
        in_prepped = true; in_cleared = true;
      } else if ((tail_mask & 8) && l + 1 < NLAYER) {
        const Layer& Nx = L[l + 1];
        TailArgs ta = tail_args(x, D, D, R, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, 4);
        ta.mul = Nx.attn_norm_s;
        ta.clear_a = proj; ta.clear_a_n4 = R * (Nx.attn ? ATT_IN : GDN_IN) / 4;
        if (!Nx.attn) { ta.clear_b = ab; ta.clear_b_n4 = R * 96 / 4; }
        gemm_tail_norm(Ly.down, R, x, D, ta);
        in_prepped = true; in_cleared = true;
      } else if ((down_clear && l + 1 < NLAYER) || gemm_flag_on) {
        TailArgs ta = gemm_flag_on ? flag_ta(l, 3, (unsigned)(R * FF / 1024)) : TailArgs{};
        ta.R = R;
        if (down_clear && l + 1 < NLAYER) {
        const Layer& Nx = L[l + 1];
        ta.clear_a = proj; ta.clear_a_n4 = R * (Nx.attn ? ATT_IN : GDN_IN) / 4;
        if (!Nx.attn && prep_clear && !ab_bf16 && act_fp4 && !conv_ab) { ta.clear_b = ab; ta.clear_b_n4 = R * 96 / 4; }
        in_cleared = true;
        }
        gemm_clear(Ly.down, R, x, D, ta);
      } else gemm(Ly.down, R, x, D, true);
      }
      if (capture)
        for (int j = 0; j < NCAP; j++)
          if (l == cap_layers[j]) CK(cudaMemcpy2DAsync(hcap + (size_t)j * D, NCAP * D * 4, x, D * 4, D * 4, R, cudaMemcpyDeviceToDevice, st));
    }
    if (pf_chain) {
      cudaStreamCaptureStatus status;
      CK(cudaStreamIsCapturing(st, &status));
      if (status == cudaStreamCaptureStatusActive)
        fprintf(stderr, "prefetch-chain placement=%s granularity=%s target=%s cap_bytes=%zu intended_hint_bytes=%zu intended_requests=%zu traffic=unmeasured\n",
            pf_placement == h5::PfChainPlacement::Tail ? "tail" : "prologue",
            pf_granularity == h5::PfChainGranularity::CtaShare ? "cta" : "pieces",
            pf_far_gu ? "far-gu" : pf_gu ? "gu" : "out", pf_chain, pf_intended_bytes, pf_intended_requests);
    }
    if (head_rows) {
      ph("prep");
      prep_early_next = early_mask & 16;
      prep(x, nullptr, D, output_norm, signs5120, D, P_NORM | P_SIGN | P_HADAMARD | P_QUANT, head_rows,
           nullptr, nullptr, 0, demand.is_all() ? nullptr : d_head_rows);
      ph("head");
      for (int r0 = 0; r0 < head_rows; r0 += 128) {
        const int rr = std::min(128, head_rows - r0);
        gemm(head, rr, logits, VOCAB, false, r0);
        if (adj_k) launch_logit_adj(logits, VOCAB, rr, d_adj, adj_k, r0, demand.is_all() ? nullptr : d_head_rows, st);
        if (draft_margin > 0.f)
          launch_relaxed_argmax(logits, VOCAB, d_out + r0, rr, st, d_tok, d_segs, nseg, r0, draft_margin,
                                demand.is_all() ? nullptr : d_head_rows, d_rstat, relax_stats);
        else if (const char* as = getenv("H5_ARGMAX_SPLIT"); as && !strcmp(as, "1")) launch_argmax_split(logits, VOCAB, d_out + r0, rr, argmax_keys, argmax_cnt, st);
        else if (as && strcmp(as, "0")) throw std::runtime_error("H5_ARGMAX_SPLIT must be 0 or 1");
        else launch_argmax(logits, VOCAB, d_out + r0, rr, st);
        if (screen_probe) head_screen_probe(r0, rr);
        if (host_logits) CK(cudaMemcpyAsync(host_logits + (size_t)r0 * VOCAB, logits, (size_t)rr * VOCAB * 4, cudaMemcpyDeviceToHost, st));
      }
      CK(cudaMemcpyAsync(h_out, d_out, head_rows * sizeof(int), cudaMemcpyDeviceToHost, st));
      if (relax_stats) CK(cudaMemcpyAsync(h_rstat, d_rstat, head_rows * sizeof(RowStat), cudaMemcpyDeviceToHost, st));
    }
    ph("end");
    act_reset();
    };
    const bool graphable = use_graphs && !prof && !sync_debug && !single_diag && !host_logits && !rope && !visual;
    const bool mega = demand.is_all() && !rope && !visual && use_mega && !wscale_e4m3 && nseg == 1 && R <= 8 && act_fp4 && !host_logits && !prof;
    if (mega) {
      CK(cudaMemcpyAsync(d_qb, h_qb, nqb * sizeof(QBlk), cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_tok, h_tok, R * 4, cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_slot, h_slot, R * 4, cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(d_pos, h_pos, R * 4, cudaMemcpyHostToDevice, st));
      graph_sample.route = H5_GRAPH_ROUTE_MEGA;
      run_mega(R, nsp, tps, max_mt);
      CK(cudaMemcpyAsync(h_out, d_mout, R * 4, cudaMemcpyDeviceToHost, st));
    }
    else if (graphable) run_graph(GKey{demand.is_all() ? 0 : 3, R, nseg, nqb, nsp, tps, max_mt,
                                      demand.is_all() ? (int)capture : 2 * head_rows + (int)capture,
                                      2 * (int)single_mode + (int)single_gdn}, body);
    else body();
    const auto probe_sync = probe ? ProbeClock::now() : ProbeClock::time_point{};
    read_a8_error();
    CK(cudaStreamSynchronize(st));
    require_a8_valid();
    const auto probe_finish = probe ? ProbeClock::now() : ProbeClock::time_point{};
    CK(cudaGetLastError());
    prof_flush();
    last_step_s = now_s() - t0;
    if (f02_mode == F02Mode::H32 && R == F02_ROWS) { f02_steps++; f02_rows += R; }
    captured_rows = capture ? R : 0;
    demand.publish(R, h_out, out);
    if (adj_k) {
      if (mega) throw std::runtime_error("H5_LOGIT_ADJ is not supported on the megakernel route");
      for (size_t i = 0; i < (size_t)R * adj_k; ++i) h_adj[i].x = -1;   // adjustments apply to one step only
    }
    if (relax_stats) {
      if (mega) throw std::runtime_error("H5_RELAX_STATS is not supported on the megakernel route");
      std::fill_n(last_rstat.begin(), R, RowStat{-1, 0.f, 0.f, 0.f, 0.f, -1, 0.f});
      if (head_rows) for (int j = 0; j < demand.count(R); ++j) last_rstat[demand.source(j)] = h_rstat[j];
    }
    last_logits_rows = !mega && head_rows ? (head_rows - 1) % 128 + 1 : 0;
    if (diagnostic_readout_enabled) {
      diagnostic_readout_ready = diagnostic_readout_capturing;
      diagnostic_readout_capturing = false;
    }
    if (probe) {
      const auto end = ProbeClock::now();
      graph_sample.prepare_ns = probe_ns(probe_begin, probe_submit);
      graph_sample.submit_ns = probe_ns(probe_submit, probe_sync);
      graph_sample.sync_ns = probe_ns(probe_sync, probe_finish);
      graph_sample.finish_ns = probe_ns(probe_finish, end);
      graph_sample.total_ns = probe_ns(probe_begin, end);
      graph_sample_ready = true;
    }
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
h5::GdnDiagnosticResult h5_gdn_diagnostic_configure(void* e, h5::GdnDiagnosticMode mode) {
  using namespace h5;
  if (!e || (mode != GdnDiagnosticMode::Observe && mode != GdnDiagnosticMode::ShadowF16 && mode != GdnDiagnosticMode::ShadowQ8))
    return {GdnDiagnosticError::BadArgument, 0};
  if (diagnostic_environment() != GdnDiagnosticError::None) return {GdnDiagnosticError::UnsupportedRoute, 0};
#if defined(EXP_GDN_NOMEM) || defined(EXP_GDN_NOCOMP)
  return {GdnDiagnosticError::UnsupportedRoute, 0};
#endif
  auto* en = (Engine*)e;
  if (en->gdn_diagnostic || en->graph_probe_used) return {GdnDiagnosticError::AlreadyUsed, 0};
  if (en->state_fusion || en->use_mega || gdn_single_mode() != GdnSingleMode::Off ||
      (mode != GdnDiagnosticMode::Observe && !diagnostic_shadow_driver_format(en->state_format)))
    return {GdnDiagnosticError::UnsupportedRoute, 0};
  for (int sl = 0; sl < en->max_slots; ++sl)
    if (en->keep_slot[sl] || en->commit_next[sl] || en->pend_n[sl]) return {GdnDiagnosticError::UnsupportedSpeculation, 0};
  if (mode != GdnDiagnosticMode::Observe) {
    const StateFormat format = mode == GdnDiagnosticMode::ShadowQ8 ? StateFormat::Q8 : StateFormat::F16;
    const size_t bytes = gdn_state_slot_bytes(format);
    void* state = nullptr; float* output = nullptr;
    auto error = cudaMalloc(&state, (size_t)en->max_slots * bytes);
    if (error == cudaSuccess) error = cudaMalloc(&output, (size_t)en->max_rows * VDIM * sizeof(float));
    if (error == cudaSuccess) error = cudaMemsetAsync(state, 0, (size_t)en->max_slots * bytes, en->st);
    if (error == cudaSuccess) error = cudaStreamSynchronize(en->st);
    if (error != cudaSuccess) {
      if (output) cudaFree(output); if (state) cudaFree(state);
      return {GdnDiagnosticError::Cuda, (int)error};
    }
    en->diagnostic_format = format; en->diagnostic_slot_bytes = bytes;
    en->diagnostic_state = state; en->diagnostic_y = output;
  }
  en->use_graphs = false; en->gdn_diagnostic = true;
  return {GdnDiagnosticError::None, 0};
}
h5::GdnDiagnosticResult h5_gdn_diagnostic_info(void* e, h5::GdnDiagnosticSource source, h5::GdnDiagnosticInfo* out) {
  using namespace h5;
  if (!e || !out || (source != GdnDiagnosticSource::Driver && source != GdnDiagnosticSource::Shadow)) return {GdnDiagnosticError::BadArgument, 0};
  auto* en = (Engine*)e;
  if (source == GdnDiagnosticSource::Shadow && !en->diagnostic_state) return {GdnDiagnosticError::NoShadow, 0};
  *out = {source == GdnDiagnosticSource::Driver ? en->state_format : en->diagnostic_format, en->max_slots, en->ctx, en->max_rows};
  return {GdnDiagnosticError::None, 0};
}
h5::GdnDiagnosticResult h5_gdn_diagnostic_read(void* e, h5::GdnDiagnosticSource source, int slot, int layer, int head, float* out, size_t elements) {
  using namespace h5;
  if (!e || !out || slot < 0 || head < 0 || head >= HV || gdn_layer_index(layer) < 0) return {GdnDiagnosticError::BadArgument, 0};
  GdnDiagnosticInfo info{};
  const auto status = h5_gdn_diagnostic_info(e, source, &info);
  if (status.error != GdnDiagnosticError::None) return status;
  if (slot >= info.slots) return {GdnDiagnosticError::BadArgument, 0};
  if (elements < GDN_DIAGNOSTIC_HEAD_ELEMENTS) return {GdnDiagnosticError::BufferTooSmall, 0};
  auto* en = (Engine*)e;
  const void* base = source == GdnDiagnosticSource::Driver ? en->gdn_state : en->diagnostic_state;
  const size_t gl = (size_t)slot * NGDN + gdn_layer_index(layer);
  try {
    std::vector<uint32_t> raw(GDN_DIAGNOSTIC_HEAD_ELEMENTS);
    float scales[SS];
    auto error = cudaStreamSynchronize(en->st);
    if (error != cudaSuccess) return {GdnDiagnosticError::Cuda, (int)error};
    if (info.format == StateFormat::Q8) {
      const auto* layer_data = (const uint8_t*)base + gl * sizeof(GdnQ8Layer);
      error = cudaMemcpyAsync(raw.data(), layer_data + (size_t)head * SS * SS, GDN_DIAGNOSTIC_HEAD_ELEMENTS, cudaMemcpyDeviceToHost, en->st);
      if (error == cudaSuccess) error = cudaMemcpyAsync(scales, layer_data + offsetof(GdnQ8Layer, scales) + (size_t)head * SS * sizeof(float), sizeof(scales), cudaMemcpyDeviceToHost, en->st);
    } else {
      const size_t bytes = pending_state_element_bytes(info.format);
      const size_t index = (gl * HV + head) * GDN_DIAGNOSTIC_HEAD_ELEMENTS;
      error = cudaMemcpyAsync(raw.data(), (const uint8_t*)base + index * bytes, GDN_DIAGNOSTIC_HEAD_ELEMENTS * bytes, cudaMemcpyDeviceToHost, en->st);
    }
    // Even on a failed second copy, drain any first copy before destroying host buffers.
    const auto sync = cudaStreamSynchronize(en->st);
    if (error != cudaSuccess || sync != cudaSuccess) return {GdnDiagnosticError::Cuda, (int)(error != cudaSuccess ? error : sync)};
    return {diagnostic_decode_head(info.format, raw.data(), info.format == StateFormat::Q8 ? scales : nullptr, out), 0};
  } catch (const std::bad_alloc&) { return {GdnDiagnosticError::HostAllocation, 0}; }
}
h5::GdnDiagnosticResult h5_gdn_readout_configure(void* e, int layer_count, const int* layers, int head_count, const int* heads) {
  using namespace h5;
  if (!e || !layers || !heads || layer_count < 1 || layer_count > NGDN || head_count < 1 || head_count > HV)
    return {GdnDiagnosticError::BadArgument, 0};
  auto* en = (Engine*)e;
  if (!en->gdn_diagnostic) return {GdnDiagnosticError::BadArgument, 0};
  if (en->diagnostic_readout_enabled || en->graph_probe_used) return {GdnDiagnosticError::AlreadyUsed, 0};
  for (int i = 0; i < layer_count; ++i)
    if (gdn_layer_index(layers[i]) < 0 || (i && layers[i] <= layers[i - 1])) return {GdnDiagnosticError::BadArgument, 0};
  for (int i = 0; i < head_count; ++i)
    if (heads[i] < 0 || heads[i] >= HV || (i && heads[i] <= heads[i - 1])) return {GdnDiagnosticError::BadArgument, 0};
  try {
    std::vector<int> selected_layers(layers, layers + layer_count), selected_heads(heads, heads + head_count);
    std::vector<float> host(VDIM);
    std::vector<GdnReadoutSample> samples((size_t)(en->diagnostic_state ? 2 : 1) * en->max_slots * layer_count * head_count);
    en->diagnostic_readout_layers = std::move(selected_layers); en->diagnostic_readout_heads = std::move(selected_heads);
    en->diagnostic_readout_host = std::move(host); en->diagnostic_readout_samples = std::move(samples);
    en->diagnostic_readout_enabled = true;
    return {GdnDiagnosticError::None, 0};
  } catch (const std::bad_alloc&) { return {GdnDiagnosticError::HostAllocation, 0}; }
}
h5::GdnDiagnosticResult h5_gdn_readout_arm(void* e) {
  using namespace h5;
  if (!e) return {GdnDiagnosticError::BadArgument, 0};
  auto* en = (Engine*)e;
  if (!en->diagnostic_readout_enabled) return {GdnDiagnosticError::ObserverDisabled, 0};
  if (en->diagnostic_readout_armed || en->diagnostic_readout_capturing) return {GdnDiagnosticError::AlreadyUsed, 0};
  en->diagnostic_readout_ready = false; en->diagnostic_readout_armed = true;
  return {GdnDiagnosticError::None, 0};
}
h5::GdnDiagnosticResult h5_gdn_readout_read(void* e, h5::GdnDiagnosticSource source, int slot, int layer, int head, h5::GdnReadoutSample* out) {
  using namespace h5;
  if (!e || !out || (source != GdnDiagnosticSource::Driver && source != GdnDiagnosticSource::Shadow)) return {GdnDiagnosticError::BadArgument, 0};
  auto* en = (Engine*)e;
  if (slot < 0 || slot >= en->max_slots || gdn_layer_index(layer) < 0 || head < 0 || head >= HV) return {GdnDiagnosticError::BadArgument, 0};
  if (!en->diagnostic_readout_enabled) return {GdnDiagnosticError::ObserverDisabled, 0};
  if (source == GdnDiagnosticSource::Shadow && !en->diagnostic_state) return {GdnDiagnosticError::NoShadow, 0};
  const auto li = std::lower_bound(en->diagnostic_readout_layers.begin(), en->diagnostic_readout_layers.end(), layer);
  const auto hi = std::lower_bound(en->diagnostic_readout_heads.begin(), en->diagnostic_readout_heads.end(), head);
  if (li == en->diagnostic_readout_layers.end() || *li != layer || hi == en->diagnostic_readout_heads.end() || *hi != head)
    return {GdnDiagnosticError::NotSelected, 0};
  *out = en->diagnostic_readout_ready ? en->diagnostic_readout_samples[en->readout_index(
      source == GdnDiagnosticSource::Driver ? 0 : 1, slot, li - en->diagnostic_readout_layers.begin(), hi - en->diagnostic_readout_heads.begin())] : GdnReadoutSample{};
  return {GdnDiagnosticError::None, 0};
}
int h5_step(void*, int, const int*, const int*, const int*, int*);
h5::GdnDiagnosticResult h5_gdn_diagnostic_step(void* e, int rows, const int* slots, const int* tokens, const int* positions, int* output) {
  using namespace h5;
  if (!e || !((Engine*)e)->gdn_diagnostic) return {GdnDiagnosticError::BadArgument, 0};
  const int status = h5_step(e, rows, slots, tokens, positions, output);
  if (status == 0) return {GdnDiagnosticError::None, 0};
  if (status == (int)GdnDiagnosticError::UnsupportedRoute || status == (int)GdnDiagnosticError::UnsupportedSpeculation)
    return {(GdnDiagnosticError)status, 0};
  return {GdnDiagnosticError::EngineStep, status};
}
int h5_step(void* e, int R, const int* slot, const int* tok, const int* pos, int* out) {
  if(FILE*f=h5_record_ptr.load()) { fprintf(f,"STEP %d",R); for(int i=0;i<R;i++) fprintf(f," %d %d %d",slot[i],tok[i],pos[i]); fprintf(f,"\n"); }
  try { return ((Engine*)e)->step(R, slot, tok, pos, out, nullptr); }
  catch (h5::GraphMemoryFailure& ex) { fprintf(stderr, "h5_step: %s\n", ex.what()); return (int)ex.code; }
  catch (std::exception& ex) { fprintf(stderr, "h5_step: %s\n", ex.what()); return -9; }
}
int h5_step_head(void* e, int R, const int* slot, const int* tok, const int* pos, int* out,
                 H5HeadMode mode, int count, const int* rows) {
  return h5_step_visual_head(e, R, slot, tok, pos, nullptr, nullptr, out, mode, count, rows);
}
int h5_step_visual_head(void* e, int R, const int* slot, const int* tok, const int* pos,
                        const int* rope, const float* visual, int* out, H5HeadMode mode, int count, const int* rows) {
  h5::HeadDemand demand = h5::HeadDemand::all();
  const auto error = h5::HeadDemand::from_api(mode, count, rows, R, demand);
  if (error != H5_HEAD_OK) return error;
  if (!e || !slot || !tok || !pos || !out) return -1;
  if (!rope && !visual) if (FILE* f = h5_record_ptr.load()) {
    fprintf(f, "HEADSTEP %d", R);
    for (int i = 0; i < R; ++i) fprintf(f, " %d %d %d", slot[i], tok[i], pos[i]);
    fprintf(f, " %d", demand.count(R));
    for (int j = 0; j < demand.count(R); ++j) fprintf(f, " %d", demand.source(j));
    fprintf(f, "\n");
  }
  try { return ((Engine*)e)->step(R, slot, tok, pos, out, nullptr, rope, visual, demand); }
  catch (h5::GraphMemoryFailure& ex) { fprintf(stderr, "h5_step_head: %s\n", ex.what()); return (int)ex.code; }
  catch (std::exception& ex) { fprintf(stderr, "h5_step_head: %s\n", ex.what()); return -9; }
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
  const char* kind = getenv("H5_DRAFTER");
  const bool dspark = kind && !strcmp(kind, "dspark-bf16");
  const auto cfg = dspark ? h5::dspark_config(proposals, slots, false, ctx, 1) : h5::draft_config(proposals, slots, false, ctx, 1);
  if (cfg.error != h5::DraftError::None || rows < cfg.query_rows) {
    fprintf(stderr, "h5_create_draft: %s (max_rows=%d)\n", cfg.error == h5::DraftError::None ? "draft row capacity too small" : h5::draft_error(cfg.error), rows); return nullptr;
  }
  try { return new Engine(path, slots, ctx, rows, proposals); } catch (std::exception& e) { fprintf(stderr, "h5_create_draft: %s\n", e.what()); return nullptr; }
}
int h5_draft_load(void* e, const char* path) { try { ((Engine*)e)->draft_load(path); return 0; } catch (std::exception& ex) { fprintf(stderr, "draft_load: %s\n", ex.what()); return -1; } }
void h5_capture(void* e, int on) { ((Engine*)e)->capture = on; }
int h5_draft(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0, const int* anchor, const int* start, int* drafts) {
  try { return ((Engine*)e)->draft_step(ns, slot, n_ing, ing_row0, ing_pos0, anchor, start, drafts); }
  catch (h5::GraphMemoryFailure& ex) { fprintf(stderr, "h5_draft: %s\n", ex.what()); return (int)ex.code; }
  catch (std::exception& ex) { fprintf(stderr, "h5_draft: %s\n", ex.what()); return -9; }
}
int h5_sample_target(void* e, const H5SamplingConfig* c, uint64_t seed, uint64_t first,
    const unsigned char* seen, const int* prefix, int first_row, int rows, int* out) {
  if (!e || !c) return -1;
  auto* en = (Engine*)e;
  if (first_row < 0 || rows < 1 || first_row > en->last_logits_rows - rows) return H5_HEAD_LOGITS_UNAVAILABLE;
  const bool matched = en->draft_trace_first == first && en->draft_trace_seed == seed;
  en->gpu_sampler.trace = matched && en->trace_due() && c->temperature > 0 && c->top_k > 0 &&
      en->draft_trace_config.temperature > 0 && !en->draft_trace_scores.empty();
  const int result = h5::gpu_sample_rows(en->gpu_sampler, en->logits + size_t(first_row) * h5::VOCAB,
      h5::VOCAB, *c, seed, first, seen, prefix, rows, out, en->st);
  if (!result && en->gpu_sampler.trace) {
    FILE* file = en->sampling_trace;
    const auto& q = en->draft_trace_config;
    const int recorded = std::min(rows, en->df_proposals);
    int accepted = 0;
    while (accepted < recorded && out[accepted] == en->draft_trace_tokens[accepted]) ++accepted;
    fprintf(file, "{\"seed\":%llu,\"first_index\":%llu,\"temperature\":%.17g,\"top_p\":%.17g,\"min_p\":%.17g,\"top_k\":%d,\"draft_temperature\":%.17g,\"draft_top_p\":%.17g,\"draft_min_p\":%.17g,\"draft_top_k\":%d,\"accepted_prefix\":%d,\"rows\":[",
        (unsigned long long)seed, (unsigned long long)first, c->temperature, c->top_p, c->min_p, c->top_k,
        q.temperature, q.top_p, q.min_p, q.top_k, accepted);
    for (int r = 0; r < recorded; ++r) {
      fprintf(file, "%s{\"target_ids\":[", r ? "," : "");
      for (int k = 0; k < c->top_k; ++k) fprintf(file, "%s%d", k ? "," : "", en->gpu_sampler.trace_ids[size_t(r) * c->top_k + k]);
      fprintf(file, "],\"target_scores\":[");
      for (int k = 0; k < c->top_k; ++k) fprintf(file, "%s%.17g", k ? "," : "", en->gpu_sampler.trace_scores[size_t(r) * c->top_k + k]);
      fprintf(file, "],\"draft_ids\":[");
      for (int k = 0; k < h5::DF_TOPK; ++k) fprintf(file, "%s%d", k ? "," : "", en->draft_trace_ids[r * h5::DF_TOPK + k]);
      fprintf(file, "],\"draft_scores\":[");
      for (int k = 0; k < h5::DF_TOPK; ++k) fprintf(file, "%s%.17g", k ? "," : "", en->draft_trace_scores[r * h5::DF_TOPK + k]);
      fprintf(file, "],\"draft_token\":%d,\"target_token\":%d}", en->draft_trace_tokens[r], out[r]);
    }
    fprintf(file, "]}\n");
    if (fflush(file) || ferror(file)) { fprintf(stderr, "sampling trace write failed\n"); return -9; }
    --en->sampling_trace_left;
  }
  if (matched) ++en->sampling_trace_calls;
  return result;
}
int h5_draft_sampled(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0,
    const int* ing_pos0, const int* anchor, const int* start, int* drafts,
    const H5SamplingConfig* c, uint64_t seed, uint64_t first, const unsigned char* seen) {
  if (!e || !c || !seen || !drafts || ns != 1 || !h5::sampling_config_valid(*c)) return -1;
  auto* en = (Engine*)e;
  try {
    if (en->native_dspark) throw std::runtime_error("native DSpark BF16 currently supports greedy proposals only");
    const int rc = en->draft_step(ns, slot, n_ing, ing_row0, ing_pos0, anchor, start, drafts);
    if (rc) return rc;
    if (!en->df.completed_prediction_rows) return 0;
    if (first > UINT64_MAX - uint64_t(en->df_proposals - 1)) return -1;
    const int prepared = h5::gpu_sampler_seen(en->gpu_sampler, seen, h5::VOCAB, en->st);
    if (prepared) return prepared;
    CK(cudaMemsetAsync(en->gpu_sampler.error, 0, sizeof(int), en->st));
    CK(h5::launch_selector_sampled(en->df.pred, en->df.succ, en->df.hp, en->df.topv, en->df.topi,
        en->df.meta + 7 * en->max_rows, en->df.d_drafts, en->df.d_conf, en->df_proposals, *c, seed, first,
        en->gpu_sampler.seen, en->gpu_sampler.error, en->trace_due() ? en->df.sample_scores : nullptr, en->draft_lz(), en->st));
    CK(cudaMemcpyAsync(drafts, en->df.d_drafts, en->df_proposals * sizeof(int), cudaMemcpyDeviceToHost, en->st));
    CK(cudaMemcpyAsync(en->df.h_conf, en->df.d_conf, en->df_proposals * sizeof(float), cudaMemcpyDeviceToHost, en->st));
    int error = 0;
    CK(cudaMemcpyAsync(&error, en->gpu_sampler.error, sizeof(int), cudaMemcpyDeviceToHost, en->st));
    if (en->trace_due()) {
      en->draft_trace_ids.resize(en->df_proposals * h5::DF_TOPK);
      en->draft_trace_scores.resize(en->df_proposals * h5::DF_TOPK);
      CK(cudaMemcpyAsync(en->draft_trace_ids.data(), en->df.topi, en->draft_trace_ids.size() * sizeof(int), cudaMemcpyDeviceToHost, en->st));
      CK(cudaMemcpyAsync(en->draft_trace_scores.data(), en->df.sample_scores, en->draft_trace_scores.size() * sizeof(double), cudaMemcpyDeviceToHost, en->st));
    } else { en->draft_trace_ids.clear(); en->draft_trace_scores.clear(); }
    CK(cudaStreamSynchronize(en->st));
    if (en->sampling_trace) {
      en->draft_trace_tokens.assign(drafts, drafts + en->df_proposals);
      en->draft_trace_config = *c; en->draft_trace_seed = seed; en->draft_trace_first = first;
    }
    return error ? -2 : 0;
  } catch (std::exception& ex) { fprintf(stderr, "h5_draft_sampled: %s\n", ex.what()); return -9; }
}
int h5_draft_two(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0,
    const int* anchor, const int* start, int* drafts, int* alternatives, int* branches, int fork_span) {
  if (!alternatives) return -1;
  try { return ((Engine*)e)->draft_step(ns, slot, n_ing, ing_row0, ing_pos0, anchor, start, drafts, alternatives, branches, fork_span); }
  catch (h5::GraphMemoryFailure& ex) { fprintf(stderr, "h5_draft_two: %s\n", ex.what()); return (int)ex.code; }
  catch (std::exception& ex) { fprintf(stderr, "h5_draft_two: %s\n", ex.what()); return -9; }
}
int h5_clone_slot(void* e, int src, int dst, int npos) {
  try { return ((Engine*)e)->clone_slot(src, dst, npos); }
  catch (std::exception& ex) { fprintf(stderr, "h5_clone_slot: %s\n", ex.what()); return -9; }
}
const float* h5_draft_conf(void* e) { auto* en = (Engine*)e; return en->native_dspark ? en->native_dspark->confidence() : en->df.h_conf; }
h5::DraftProbeResult h5_draft_probe(void* e, h5::DraftProbeKind kind, void* output, size_t capacity) {
  using h5::DraftProbeError;
  static_assert(sizeof(int) == sizeof(float));
  if (!e || !output || (kind != h5::DraftProbeKind::TrunkResidual && kind != h5::DraftProbeKind::HiddenProjection &&
                       kind != h5::DraftProbeKind::TopValues && kind != h5::DraftProbeKind::TopIndices))
    return {DraftProbeError::BadArgument, 0, 0, 0, 0};
  const auto& df = ((Engine*)e)->df;
  if (!df.completed_prediction_rows) return {DraftProbeError::NoPrediction, 0, 0, 0, 0};
  const void* source = nullptr; int rows = 0, columns = 0;
  switch (kind) {
    case h5::DraftProbeKind::TrunkResidual: source=df.xd;rows=df.completed_block_rows;columns=h5::D;break;
    case h5::DraftProbeKind::HiddenProjection: source=df.hp;rows=df.completed_prediction_rows;columns=h5::DF_RANK;break;
    case h5::DraftProbeKind::TopValues: source=df.topv;rows=df.completed_prediction_rows;columns=h5::DF_TOPK;break;
    case h5::DraftProbeKind::TopIndices: source=df.topi;rows=df.completed_prediction_rows;columns=h5::DF_TOPK;break;
    default: return {DraftProbeError::BadArgument, 0, 0, 0, 0};
  }
  const size_t bytes = (size_t)rows * columns * sizeof(float);
  if (capacity < bytes) return {DraftProbeError::BufferTooSmall, 0, 0, 0, bytes};
  const auto error = cudaMemcpy(output, source, bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return {DraftProbeError::Cuda, (int)error, 0, 0, bytes};
  return {DraftProbeError::None, 0, rows, columns, bytes};
}
void h5_keep(void* e, int slot, int flag) { if(FILE*f=h5_record_ptr.load()) fprintf(f,"KEEP %d %d\n",slot,flag); ((Engine*)e)->keep_slot[slot] = flag; }
// Per-row relaxed-argmax statistics of the last step (H5_RELAX_STATS); rows without a head row have greedy -1.
int h5_row_stats(void* e, int R, h5::RowStat* out) {
  auto* E = (Engine*)e;
  if (!E->relax_stats || R < 0 || R > (int)E->last_rstat.size()) return -1;
  std::copy_n(E->last_rstat.begin(), R, out); return 0;
}
// Logit adjustments for the next step's rows 0..R-1: tok/val are R x K (K = H5_LOGIT_ADJ), token -1 = unused.
int h5_set_logit_adj(void* e, int R, const int* tok, const float* val) {
  auto* E = (Engine*)e;
  if (!E->adj_k || R < 0 || R > E->max_rows) return -1;
  for (size_t i = 0; i < (size_t)R * E->adj_k; ++i) { int bits; std::memcpy(&bits, &val[i], 4); E->h_adj[i] = make_int2(tok[i] >= 0 && tok[i] < h5::VOCAB ? tok[i] : -1, bits); }
  return 0;
}
int h5_logit_adj_k(void* e) { return ((Engine*)e)->adj_k; }
// Drafter LZ contexts for the next draft step's blocks 0..nb-1 (draft order): ctx is nb x WINDOW (oldest first),
// len[b] <= WINDOW tokens used, alpha[b] the strength (0 = off). Needs H5_DRAFT_LZ.
int h5_set_draft_lz(void* e, int nb, const int* ctx, const int* len, const float* alpha) {
  auto* E = (Engine*)e;
  if (!E->lzd_w || nb < 0 || nb > E->max_slots) return -1;
  for (int b = 0; b < nb; ++b) {
    if (len[b] < 0 || len[b] > E->lzd_w || !(alpha[b] >= 0.f)) return -1;
    std::memcpy(E->h_lzctx + (size_t)b * E->lzd_w, ctx + (size_t)b * E->lzd_w, len[b] * sizeof(int));
    E->h_lzlen[b] = len[b]; E->h_lzalpha[b] = alpha[b];
  }
  return 0;
}
int h5_draft_lz_window(void* e) { return ((Engine*)e)->lzd_w; }
void h5_accept(void* e, int slot, int n) { if(FILE*f=h5_record_ptr.load()) fprintf(f,"ACCEPT %d %d\n",slot,n); ((Engine*)e)->commit_next[slot] = n; }
void h5_prof_report(void* e) { ((Engine*)e)->prof_report(); }
void h5_copy_slot(void* e, int src, int dst, int npos) { ((Engine*)e)->copy_slot(src, dst, npos); }
int h5_copy_prefix(void* e, int src, int dst, int npos) {
  Engine* en = (Engine*)e;
  if (src < 0 || src >= en->max_slots) return (int)h5::serving::CopyError::InvalidRange;
  const auto status = h5::serving::copy_guard(src, dst, npos, en->max_slots, en->ctx, en->pend_n[src], en->commit_next[src]);
  if (status != h5::serving::CopyError::None) return (int)status;
  try { en->copy_slot(src, dst, npos); return 0; }
  catch (std::exception& ex) { fprintf(stderr, "h5_copy_prefix: %s\n", ex.what()); return -9; }
}
int h5_tail_dbg(void* e, unsigned long long* out) {   // [8][TAIL_MAX_CHUNKS][TAIL_DBG_STAMPS]; -1 without H5_TAIL_DBG
  auto* g = (Engine*)e; if (!g->tail_dbg) return -1;
  return cudaMemcpy(out, g->tail_dbg, (size_t)8 * h5::TAIL_MAX_CHUNKS * h5::TAIL_DBG_STAMPS * 8, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -2;
}
void h5_reset(void* e, int slot) {
  if(FILE*f=h5_record_ptr.load()) fprintf(f,"RESET %d\n",slot);
  auto* en = (Engine*)e;
  en->draft_trace_first = UINT64_MAX; en->draft_trace_ids.clear(); en->draft_trace_scores.clear();
  en->sampling_trace_calls = 0; en->sampling_trace_left = en->sampling_trace_limit;
  en->reset(slot);
}
void h5_pretune(void* e) { ((Engine*)e)->pretune(); }
double h5_last_step_s(void* e) { return ((Engine*)e)->last_step_s; }
long h5_graph_stats(void* e, int which) { return which ? ((Engine*)e)->graph_caps.load() : ((Engine*)e)->graph_hits.load(); }
void h5_f02_stats(void* e, long* steps, long* rows, long* graph_hits) {
  auto* x = (Engine*)e; *steps = x->f02_steps; *rows = x->f02_rows; *graph_hits = x->graph_hits.load();
}
H5GraphProbeError h5_graph_probe_configure(void* e, H5GraphUpload upload, H5GraphTiming timing) {
  if (!e || (upload != H5_GRAPH_UPLOAD_AT_LAUNCH && upload != H5_GRAPH_UPLOAD_AT_CAPTURE) ||
      (timing != H5_GRAPH_TIMING_OFF && timing != H5_GRAPH_TIMING_WALL_THREAD)) return H5_GRAPH_PROBE_BAD_ARGUMENT;
  Engine* en = (Engine*)e;
  if (en->graph_probe_used) return H5_GRAPH_PROBE_ALREADY_USED;
  if (!en->use_graphs) return H5_GRAPH_PROBE_GRAPHS_DISABLED;
  en->graph_upload = upload; en->graph_timing = timing;
  return H5_GRAPH_PROBE_OK;
}
H5GraphProbeError h5_graph_nodes_configure(void* e, H5GraphNodes nodes) {
  if (!e || (nodes != H5_GRAPH_NODES_NATIVE && nodes != H5_GRAPH_NODES_KERNEL_MEMORY)) return H5_GRAPH_PROBE_BAD_ARGUMENT;
  Engine* en = (Engine*)e;
  if (en->graph_probe_used) return H5_GRAPH_PROBE_ALREADY_USED;
  if (!en->use_graphs) return H5_GRAPH_PROBE_GRAPHS_DISABLED;
  en->graph_nodes = nodes;
  return H5_GRAPH_PROBE_OK;
}
H5GraphProbeError h5_graph_probe_read(void* e, H5GraphSample* out) {
  if (!e || !out) return H5_GRAPH_PROBE_BAD_ARGUMENT;
  Engine* en = (Engine*)e;
  if (en->graph_timing == H5_GRAPH_TIMING_OFF) return H5_GRAPH_PROBE_DISABLED;
  if (!en->graph_sample_ready) return H5_GRAPH_PROBE_NO_SAMPLE;
  *out = en->graph_sample;
  return H5_GRAPH_PROBE_OK;
}
// Logits of the final compact head chunk; invalidated by no-head steps, mega, or drafter calls.
int h5_logits(void* e, int row, float* out) {
  if (!e || !out) return -1;
  Engine* en = (Engine*)e;
  if (row < 0 || row >= en->last_logits_rows) return H5_HEAD_LOGITS_UNAVAILABLE;
  return cudaMemcpy(out, en->logits + (size_t)row * h5::VOCAB, h5::VOCAB * 4, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -9;
}
}
