#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <utility>
// halo-5090 kernels: batched decode of Ternary Bonsai 2 27B on sm_120 (RTX 5090).
// Every row is one token of one sequence slot. Rows of the same slot are consecutive and in
// position order, so one kernel can carry a slot's recurrent state across several of its rows.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "pf_chain.h"
#include "model.h"
#include "state_format.h"
#include "q8_scale.h"
#include "pdl_clock.cuh"
#include "smlit.cuh"
#include "draft_contract.h"

namespace h5 {
// Programmatic dependent launch: every kernel launched through pdl() starts with pdl_wait() (wait
// for the preceding grid to complete and flush) and pdl_launch() (let the next grid's CTAs start
// launching). The next kernel's launch latency and any work before its own wait (weight prefetch
// in the GEMM) overlap the tail of this one. H5_PDL=0 launches normally.
#if defined(H5_PDLCLOCK) && defined(H5_PDLCLOCK_MODULE)
#define pdl_wait() pdl_wait_impl(__LINE__)
#else
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); h5::lit::note_wait(); }
#endif
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ bool l2pf_enabled() {
#ifdef H5_NO_L2PF
  return false;
#else
  return true;
#endif
}
cudaError_t launch_l2pf(const void* W, int N, int K, int wf, int splits, double max_bytes, cudaStream_t st);
cudaError_t launch_l2pf_chain(const void* W, size_t bytes, int ctas, cudaStream_t st);
// Runtime experiment bits for in-process A/B (H5_AB_VARIANT). Each translation unit has its own copy;
// h5_set_variant() updates all modules. Bit meanings are listed in tools/profile_replay.cpp.
static __device__ unsigned g_variant = 0;
// Variant bit 1 (0x2): latency kernels (prep, conv, GDN, attention, embed, argmax) release their dependents
// before their own wait, so the next grids become resident and stage their independent work (GEMM weight
// rings) while this one still waits; bit 2 (0x4): GEMMs too. Safe because every pre-wait access reads only
// weights, constants or state written before the step began, and every write follows the wait.
__device__ __forceinline__ void pdl_enter() {
  if (g_variant & 2u) { pdl_launch(); pdl_wait(); } else { pdl_wait(); pdl_launch(); }
}
__device__ __forceinline__ void pdl_enter_e(bool early) {   // early: trigger dependents before the wait (H5_EARLY_K)
  if (early || (g_variant & 2u)) { pdl_launch(); pdl_wait(); } else { pdl_wait(); pdl_launch(); }
}
__device__ __forceinline__ void pdl_early_trigger(unsigned bit) { if (g_variant & bit) pdl_launch(); }
// H5_CARVEOUT=0..100|none (re-read at every launch, i.e. at graph capture, so engines of one process can
// differ; h5_set_launch_carveout overrides it) and H5_CARVEOUT_SCOPE=all|prep|prepall: every launch, preps of
// one-row steps only, or every prep.
inline int& launch_carveout_override() { static int v = -2; return v; }   // -2 = follow the environment
inline int launch_carveout_slot_value() {
  if (launch_carveout_override() != -2) return launch_carveout_override();
  const char* c = getenv("H5_CARVEOUT"); if (!c || !strcmp(c, "none")) return -1;
  char* e = nullptr; long x = strtol(c, &e, 10);
  if (!e || *e || e == c || x < 0 || x > 100) { fprintf(stderr, "H5_CARVEOUT must be 0..100 or none\n"); abort(); }
  return (int)x;
}
inline int& carveout_prep_depth() { static int d = 0; return d; }
inline int carveout_scope() {   // 0 all, 1 prep (one-row steps), 2 prepall
  const char* c = getenv("H5_CARVEOUT_SCOPE"); if (!c || !strcmp(c, "all")) return 0;
  if (!strcmp(c, "prep")) return 1; if (!strcmp(c, "prepall")) return 2;
  fprintf(stderr, "H5_CARVEOUT_SCOPE must be all, prep or prepall\n"); abort();
}
inline int launch_carveout() { return (carveout_scope() == 0 || carveout_prep_depth() > 0) ? launch_carveout_slot_value() : -1; }
inline bool pdl_enabled() { static int on = getenv("H5_PDL") ? atoi(getenv("H5_PDL")) : 1; return on; }
template <typename... P, typename... A>
inline void pdl(void (*k)(P...), dim3 g, dim3 b, size_t smem, cudaStream_t st, A&&... args) {
  // Preferred shared-memory carveout for every launch (H5_CARVEOUT=0..100, or set per engine via
  // h5_set_launch_carveout before its graphs are captured). An SM configured for a small-smem kernel
  // cannot accept a large-smem CTA until it drains, so early-launched GEMM CTAs otherwise double up
  // on other SMs and stretch their tail.
  const int carveout = launch_carveout();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[2]; unsigned na = 0;
  if (pdl_enabled()) { at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[na].val.programmaticStreamSerializationAllowed = 1; na++; }
  if (carveout >= 0) { at[na].id = cudaLaunchAttributePreferredSharedMemoryCarveout; at[na].val.sharedMemCarveout = (unsigned)carveout; na++; }
  cfg.attrs = at; cfg.numAttrs = na;
  cudaLaunchKernelEx(&cfg, k, std::forward<A>(args)...);
}


enum PrepFlags : unsigned {
  P_NORM = 1, P_SILU_MUL = 2, P_PERMUTE_GDN = 4, P_SIGN = 8, P_HADAMARD = 16,
  P_QUANT = 32, P_STORE_F32 = 64, P_SIGN_AFTER = 128, P_OUT_NORM = 256, P_GDN_NORM = 512,
  P_DCONV = 1024,   // chunk route only: input = drafter dynamic conv of x (dc_* fields), see k_dconv
  P_SILU_IL = 2048, // chunk route only: x row holds gate/up interleaved in 32-column groups (gate g at 64g, up at 64g+32); value = silu(gate) * up
  P_CONSUME_CLEAR = 4096,   // with P_SILU_IL: zero every input element after loading it (the input is a split-K accumulator kept all-zero between uses)
};

struct PrepArgs {
  const float* x; const float* x2; int ldx;     // row stride in floats
  const float* norm_w; const float* signs; int n; unsigned flags;
  int8_t* xq; float* xs;                        // [row][n], [row][n/128]
  float* out_f32; float* out_norm; int ldo;      // [row][ldo]
  uint8_t* q4; uint8_t* s4; float* g4;          // NVFP4 output when q4 != nullptr
  const int* rowmap = nullptr;                  // input row = rowmap[row] when set
  const float* gnorm_w = nullptr; const float* gz = nullptr; int gz_ld = 0;   // P_GDN_NORM: per-head RMSNorm * w * silu(z)
  float* rms_inv = nullptr; int rms_rows = 0;   // dedicated [output row] scratch for explicit RMS prepass
  float* clear_a = nullptr; int clear_a_stride = 0;
  float* clear_b = nullptr; int clear_b_stride = 0;
  // P_DCONV: x is the conv source [row][D]; the prepared value is the drafter's grouped dynamic conv
  // out[t][e] = dconv_elem(base[w][0][e], dyn[t][w*640 + e/16], x[t][e], base[w][1][e], dyn[t][w*640 + 320 + e/16], x[t-1][e])
  // with x[-1] = 0 at each block start; identical arithmetic to k_dconv without add.
  const float* dc_dyn = nullptr; const float* dc_base = nullptr; int dc_which = 0, dc_block_rows = 0;
  // H5_FLAG_WAIT: start when the producer's CTAs have all signalled *sig >= sig_expect (release/acquire)
  // instead of at producer grid completion; griddepcontrol.wait then runs at exit, so this grid still
  // completes only after its producer. Chunk route only.
  const unsigned* sig = nullptr; unsigned sig_expect = 0;
  // H5_GEMM_FLAG: every CTA adds 1 (release) after its last output store, so the consuming GEMM can start
  // on the count instead of this grid's completion. Wide routes only (k_prep2w, k_prep_chunkw).
  unsigned* done_sig = nullptr;
  // H5_PREP_EARLY: trigger dependents before this prep's own wait, so the next GEMM's CTAs stream their
  // pre-wait weight ring while the prep runs (the GEMM still waits for this grid before reading activations).
  bool early = false;
  float fixed_g = 0.f;   // k_prep2w: fixed NVFP4 row scale instead of the row max (H5_PREP_ROW_FIXEDG)
};
__device__ __forceinline__ void prep_signal(unsigned* sig) {
  if (!sig) return;
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(sig) : "memory");
}
// One element of the drafter's dynamic conv (shared by k_dconv and the fused prep; explicit rounding
// so both compile to the same operations).
__device__ __forceinline__ float dconv_elem(float b0, float d0, float x0, float b1, float d1, float x1) {
  return __fmaf_rn(__fadd_rn(b0, d0), x0, __fmul_rn(__fadd_rn(b1, d1), x1));
}
void launch_prep_rms(const PrepArgs&, int rows, float fixed_g, cudaStream_t);
struct OrderedResidual {
  const float* partials;
  float* residual;
  int rows, splits;
};
cudaError_t launch_prep_ordered(const PrepArgs&, const OrderedResidual&, cudaStream_t);

// One segment = the consecutive rows of one slot in this step.
// commit: rows of this slot's pending cache (previous keep step) to replay into the state first.
// keep: this step's rows are speculative; the state is written after the replay only and the new
// rows go to the pending cache (len <= configured pending_rows).
// par: pending buffer parity of this step's new rows (replayed rows come from par ^ 1).
struct Seg { int slot, start, len, commit, keep, par; int conv_par = 0; };
constexpr int PMAX = 8;
constexpr int PEND_GDN = KDIM + VDIM + 2 * HV;   // per row: normalised k (per key head), v, g, beta; two parity buffers

struct GdnArgs {
  const float* conv_out;   // [row][CONV_CH]
  const float* proj;       // [row][GDN_IN]: z at CONV_CH
  const float* ab;         // [row][96]
  const float* ssm_a; const float* dt_bias; const float* ssm_norm;
  void* state;             // F32/BF16/F16: [slot][NGDN][HV][128][128]; Q8: [slot][NGDN] GdnQ8Layer
  int gl;                  // gdn layer index
  float* y;                // [row][VDIM]
  const Seg* segs;
  void* pend;              // [slot][NGDN][2][PMAX][PEND_GDN] (state dtype; Q8 uses FP16)
  const float* gates;      // [row][2*HV] from k_conv
  bool replay_delta = false; // pending value rows hold innovations, avoiding replay dot products
  int pending_rows = PMAX;
  unsigned* sig = nullptr;   // H5_FLAG_WAIT: each CTA adds 1 (release) once its outputs y are written, before its state save
  bool early = false;         // H5_EARLY_K: trigger dependents before the wait
  Q8ScaleMode q8_scale = Q8ScaleMode::LeastSquares; // immutable per-engine selection, captured by value
};

struct ConvArgs {
  const float* proj;       // [row][GDN_IN]
  const float* w;          // [CONV_CH][4]
  float* state;            // [slot][NGDN][3][CONV_CH]
  int gl;
  float* out;              // [row][CONV_CH]
  const Seg* segs;
  void* pend;              // [slot][NGDN][PMAX][CONV_CH] pre-conv inputs (state dtype)
  const float* ab; const float* ssm_a; const float* dt_bias; float* gates;   // gates[row][2*HV]: g | beta
  int pending_rows = PMAX;
  // Fused alpha/beta (H5_CONV_AB=1, steps of at most 8 rows): when ab_w is set, each v-head CTA computes
  // its two ab outputs per row from the bf16 weights and the normalised rows xn before its PDL wait
  // (xn comes from the row prep two grids upstream), writes them to ab_out, and uses them for the gates.
  const __nv_bfloat16* ab_w = nullptr; const float* xn = nullptr; float* ab_out = nullptr;
  bool early = false;         // H5_EARLY_K: trigger dependents before the wait
  const uint8_t* pf = nullptr; unsigned pf_bytes = 0;   // aligned intended hint bytes, not measured traffic
  h5::PfChainGranularity pf_granularity = h5::PfChainGranularity::Pieces16KiB;
  h5::PfChainPlacement pf_placement = h5::PfChainPlacement::Prologue;
};

struct AttnArgs {
  const float* proj;       // [row][ATT_IN]: q_full | k | v
  const float* q_norm; const float* k_norm;
  void* kc; void* vc;      // [slot][NATTN][NKV][ctx][HD] fp16, or fp8 e4m3 with per-(pos) scales
  float* ksc; float* vsc;  // [slot][NATTN][NKV][ctx] (fp8 only)
  int al, ctx;
  const int* row_slot; const int* row_pos;
  float* qrot;             // [row][NH*HD]
  float* y;                // [row][ATTN_OUT]
  int vt = 1;              // fp8 V stored transposed per 32-key block
  const int* rope_pos = nullptr; // optional [row][t,h,w,z], separate from causal/cache positions
  int rope_sections[4] = {0, 0, 0, 0};
  bool early = false;         // H5_EARLY_K: trigger dependents before the wait
  const uint8_t* pf = nullptr; unsigned pf_bytes = 0;   // aligned intended hint bytes, not measured traffic
  h5::PfChainGranularity pf_granularity = h5::PfChainGranularity::Pieces16KiB;
  h5::PfChainPlacement pf_placement = h5::PfChainPlacement::Prologue;
};

constexpr int QB_ROWS = 8;     // rows per attention query block
struct QBlk { int slot, start, len, pos_last; };
struct AttnTcArgs {
  const float* qrot; const float* proj;       // proj (gate source) null = no output gate
  const void* kc; const void* vc; const float* ksc; const float* vsc;
  int al, ctx;
  const int* row_pos;
  const QBlk* qb;
  float* part_o; float2* part_ml;
  float* y;
  int nlayer = NATTN, nkv = NKV, hd = HD;     // cache geometry [slot][nlayer][nkv][ctx][hd]
  int noncausal = 0, window = 0, ring = 0;    // ring: cache position = key % ring
  int tps = 8;                                // 32-key tiles per split
  int draft_block_rows = 0;                   // full noncausal block extent, independent of query tiling
  int draft_nest = 0;                         // H5_DRAFT_NEST: block rows < nest see only block keys < nest (0 = whole block)
  bool direct = false;                        // one split: normalise/gate in the core epilogue
  unsigned* sig = nullptr;                    // H5_FLAG_WAIT (combine): each CTA adds 1 (release) after its outputs
  bool early = false;         // H5_EARLY_K: trigger dependents before the wait
};

// DFlash2 drafter geometry
constexpr int DF_LAYERS = 5, DF_NH = 32, DF_NKV = 8, DF_HD = 128, DF_Q = DF_NH * DF_HD, DF_KV = DF_NKV * DF_HD;
struct TopkScratch { float* values; int* indices; int row_capacity; };
// Drafter-side LZ penalty (H5_DRAFT_LZ, tools/ANTILOOP.md): per draft block b, the committed generated tail
// ctx[b * w .. b * w + len[b]) (oldest first) and strength alpha[b] (0 = off for that block). The selector adds the
// LZ adjustment of every candidate, with the block's earlier proposals appended to the context, so proposals follow
// the penalised target. ctx == nullptr: off. form 0 = paper, 1 = dist (src/lz_penalty.h).
constexpr int DRAFT_LZ_MAX_WINDOW = 1024;
struct DraftLz { const int* ctx = nullptr; const int* len = nullptr; const float* alpha = nullptr; int w = 0, buffer = 0, form = 0; float log2v = 0.f; };
constexpr int TOPK_SPLITS = 16, TOPK_PARTS_PER_ROW = TOPK_SPLITS * 16;
constexpr int DF_KPROJ = 1280, DF_RANK = 256, DF_TOPK = 16, DF_BLOCK = 8, DF_WINDOW = 1024, DF_RING = 1120, DF_MASK = 248070;   // trained window 2048; 1024 measured equal acceptance
struct DAttnPreArgs {
  const float* qkv; int ld, koff, do_q;          // row: [q | k | v] at koff for k (v follows k)
  const float* q_norm; const float* k_norm;
  const int* slot; const int* pos;
  float* qrot;                                   // [row][DF_Q]
  __half* kc; __half* vc;                        // [slot][DF_LAYERS][DF_NKV][DF_RING][DF_HD]
  int layer;
  // Fused ingest (layers > 1, k/v only): grid z = layer offset; layer z reads its k/v at koff + z*2*DF_KV
  // of the same row and normalises k with kn[z].
  int layers = 1; const float* kn[DF_LAYERS] = {};
};

// megakernel (mega.cu)
struct MLayer {
  int attn, gl, al, n_in;
  const float *attn_norm, *post_norm;
  const uint8_t *w_in, *w_out, *w_gu, *w_down;
  const uint16_t *s_in, *s_out, *s_gu, *s_down;
  const __nv_bfloat16* ab;
  const float *conv_w, *ssm_a, *ssm_dt, *ssm_norm, *q_norm, *k_norm;
};

struct MegaArgs {
  const MLayer* L;
  const float *s5120, *s6144, *s17408, *out_norm;
  const uint8_t* emb;
  const uint8_t* w_head; const uint16_t* s_head;
  int R, ctx, nsp, tps, mt, capture, pf;
  Seg seg; const QBlk* qbp;
  const int *tok, *pos, *slotv;
  float *x, *xn, *proj, *proj2, *conv_out, *abv, *gates, *y, *qrot, *part_o; float2* part_ml; float* hcap;
  uint8_t *a4, *s4; float* gs;
  unsigned long long* amax;     // [8] packed argmax keys
  int* out_tok;
  void *gdn_state, *pend_gdn, *pend_conv; float* conv_state;
  void *kc, *vc; float *ksc, *vsc;
  unsigned* bar;
  unsigned long long* prof;   // optional: [count, (globaltimer << 8 | phase code)...] recorded by CTA 0 after each barrier
  int pending_rows = PMAX;
};

#include "tail_prep.cuh"
// GDN whose tail runs the gated-RMSNorm ssm_out prep (TAIL_CHUNK with P_PERMUTE_GDN | P_GDN_NORM).
void launch_attn_combine_tail(const AttnTcArgs&, int nsp, int rows, const TailArgs&, cudaStream_t);
void launch_argmax_split(const float* logits, int ld, int* out, int rows, unsigned long long* keys, unsigned* cnt, cudaStream_t);   // keys/cnt [rows] all-zero between uses
float prep_fixed_g();   // H5_PREP_G4 of the chunk route; typed error without one
// H5_GDN_ARITH=fp32|fp16x2|bf16x2 (unset: fp32). Packed modes require matching state dtype;
// sequential packed local dot/update arithmetic, FP32 cross-lane sums/innovation/output. See tools/GDN_PACKED.md.
enum class GdnVerifyMode { Off, Serial, Chunk, Fused };
GdnVerifyMode gdn_verify_mode_from_env(); // H5_GDN_VERIFY=off|serial|chunk|fused; unset explicitly selects Off
// Single-stream F16 speculative visit: len=rows (8/16), keep=1, commit<=pending_rows<=16.
// Serial preserves the generic CG arithmetic; Chunk changes only new-row arithmetic.
void launch_gdn_verify(const GdnArgs&, int nseg, int rows, GdnVerifyMode, StateFormat, cudaStream_t);
// Fused mode requires explicit two-bank ConvArgs state/pending and matching segment/layer metadata.
void launch_conv_gdn_verify(const ConvArgs&, const GdnArgs&, int nseg, int rows, StateFormat, cudaStream_t);
int gdn_grid_ctas(int nseg);   // CTAs of launch_gdn for nseg segments (H5_GDN_CG)
void launch_gdn_tail(const GdnArgs&, int nseg, StateFormat, const TailArgs&, cudaStream_t);
}  // namespace h5
