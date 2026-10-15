#include <cstdlib>
#include <utility>
// halo-5090 kernels: batched decode of Ternary Bonsai 2 27B on sm_120 (RTX 5090).
// Every row is one token of one sequence slot. Rows of the same slot are consecutive and in
// position order, so one kernel can carry a slot's recurrent state across several of its rows.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "model.h"
#include "pdl_clock.cuh"
#include "draft_contract.h"

namespace h5 {
// Programmatic dependent launch: every kernel launched through pdl() starts with pdl_wait() (wait
// for the preceding grid to complete and flush) and pdl_launch() (let the next grid's CTAs start
// launching). The next kernel's launch latency and any work before its own wait (weight prefetch
// in the GEMM) overlap the tail of this one. H5_PDL=0 launches normally.
#if defined(H5_PDLCLOCK) && defined(H5_PDLCLOCK_MODULE)
#define pdl_wait() pdl_wait_impl(__LINE__)
#else
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
#endif
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ bool l2pf_enabled() {
#ifdef H5_NO_L2PF
  return false;
#else
  return true;
#endif
}
inline bool pdl_enabled() { static int on = getenv("H5_PDL") ? atoi(getenv("H5_PDL")) : 1; return on; }
template <typename... P, typename... A>
inline void pdl(void (*k)(P...), dim3 g, dim3 b, size_t smem, cudaStream_t st, A&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl_enabled() ? 1 : 0;
  cudaLaunchKernelEx(&cfg, k, std::forward<A>(args)...);
}


enum class StateFormat { F32, BF16, F16 };

enum PrepFlags : unsigned {
  P_NORM = 1, P_SILU_MUL = 2, P_PERMUTE_GDN = 4, P_SIGN = 8, P_HADAMARD = 16,
  P_QUANT = 32, P_STORE_F32 = 64, P_SIGN_AFTER = 128, P_OUT_NORM = 256, P_GDN_NORM = 512,
};

struct PrepArgs {
  const float* x; const float* x2; int ldx;     // row stride in floats
  const float* norm_w; const float* signs; int n; unsigned flags;
  int8_t* xq; float* xs;                        // [row][n], [row][n/128]
  float* out_f32; float* out_norm; int ldo;      // [row][ldo]
  uint8_t* q4; uint8_t* s4; float* g4;          // NVFP4 output when q4 != nullptr
  const int* rowmap = nullptr;                  // input row = rowmap[row] when set
  const float* gnorm_w = nullptr; const float* gz = nullptr; int gz_ld = 0;   // P_GDN_NORM: per-head RMSNorm * w * silu(z)
};

// One segment = the consecutive rows of one slot in this step.
// commit: rows of this slot's pending cache (previous keep step) to replay into the state first.
// keep: this step's rows are speculative; the state is written after the replay only and the new
// rows go to the pending cache (len <= configured pending_rows).
// par: pending buffer parity of this step's new rows (replayed rows come from par ^ 1).
struct Seg { int slot, start, len, commit, keep, par; };
constexpr int PMAX = 8;
constexpr int PEND_GDN = KDIM + VDIM + 2 * HV;   // per row: normalised k (per key head), v, g, beta; two parity buffers

struct GdnArgs {
  const float* conv_out;   // [row][CONV_CH]
  const float* proj;       // [row][GDN_IN]: z at CONV_CH
  const float* ab;         // [row][96]
  const float* ssm_a; const float* dt_bias; const float* ssm_norm;
  void* state;             // [slot][NGDN][HV][128][128], float, bf16 or fp16
  int gl;                  // gdn layer index
  float* y;                // [row][VDIM]
  const Seg* segs;
  void* pend;              // [slot][NGDN][2][PMAX][PEND_GDN] (state dtype)
  const float* gates;      // [row][2*HV] from k_conv
  bool replay_delta = false; // pending value rows hold innovations, avoiding replay dot products
  int pending_rows = PMAX;
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
};

// DFlash2 drafter geometry
constexpr int DF_LAYERS = 5, DF_NH = 32, DF_NKV = 8, DF_HD = 128, DF_Q = DF_NH * DF_HD, DF_KV = DF_NKV * DF_HD;
constexpr int DF_KPROJ = 1280, DF_RANK = 256, DF_TOPK = 16, DF_BLOCK = 8, DF_WINDOW = 1024, DF_RING = 1120, DF_MASK = 248070;   // trained window 2048; 1024 measured equal acceptance
struct DAttnPreArgs {
  const float* qkv; int ld, koff, do_q;          // row: [q | k | v] at koff for k (v follows k)
  const float* q_norm; const float* k_norm;
  const int* slot; const int* pos;
  float* qrot;                                   // [row][DF_Q]
  __half* kc; __half* vc;                        // [slot][DF_LAYERS][DF_NKV][DF_RING][DF_HD]
  int layer;
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

}  // namespace h5
