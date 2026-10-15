// NVFP4 tensor-core ternary GEMM and the FP4 activation quantiser.
//
//   out[t][n] (+)= g[t] * sum_kb ws[n][kb] * sum_{k in kb} trit[n][k] * a[t][k]
//
// Weights: trits stored at two bits each so that one 32-bit word becomes two E2M1 operand
// registers with one AND and one SHIFT+AND (bit 1 of a nibble = |trit|, bit 3 = sign; +1 = 0x2,
// -1 = 0xA, 0 = 0x0). The FP16 group scale (per 128 K) is applied exactly in FP32 after each
// 128-K block, so the only loss against the INT8 path is the FP4 activation.
// Activations: E2M1 values, UE4M3 scale per 16 K, FP32 scale per token (g).
//
// Weight chunk (g, kb) = 32 features x 128 K = 1024 bytes; lane l owns bytes [32l, 32l+32) = 8 words.
// lane: gid = l>>2, tq = l&3. word w = m*4 + h*2 + r  (m = m-tile, h = k64 half, r = row half)
//   row  = 32g + 16m + 8r + gid
//   nibble j of the word: bits {1,3} = trit(row, 128kb + 64h + 8tq + j),
//                         bits {0,2} = trit(row, 128kb + 64h + 32 + 8tq + j)
// Scales: half [g][kb][gid][4] = rows (m=0,r=0), (0,1), (1,0), (1,1).
#define H5_PDLCLOCK_MODULE fp4
#define H5_LIT_MODULE fp4
#define H5_LIT_MODULE_ID 2u
#include "kernels.cuh"
#include "compact.cuh"
#include "f07.cuh"
#include "warp_microscope.cuh"
#include "fp4_ops.cuh"
#include "deterministic_gemm.cuh"
#ifdef H5_PROFILE_SCALE_CARRIER
#include "scale_carrier.cuh"
#endif
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <stdexcept>

#if defined(H5_GEMM_MATCHED_RELEASE) && H5_GEMM_MATCHED_RELEASE != 0 && H5_GEMM_MATCHED_RELEASE != 1
#error "H5_GEMM_MATCHED_RELEASE must be 0 or 1"
#endif
#if defined(H5_GEMM_MATCHED_RELEASE) && H5_GEMM_MATCHED_RELEASE && !defined(H5_PDL_EARLY_GEMM_RELEASE)
#error "H5_GEMM_MATCHED_RELEASE requires H5_PDL_EARLY_GEMM_RELEASE"
#endif
#if defined(H5_GEMM_MATCHED_RELEASE) && H5_GEMM_MATCHED_RELEASE && (defined(EXP_NOWAIT) || defined(EXP_NOMMA))
#error "H5_GEMM_MATCHED_RELEASE requires the intact GEMM"
#endif

namespace h5 {

#ifdef H5_GEMM_MATCHED_RELEASE
constexpr bool gemm_matched_release_enabled = H5_GEMM_MATCHED_RELEASE != 0;
#else
constexpr bool gemm_matched_release_enabled = false;
#endif

// Warp-specialised GEMM. FW consumer warps (32*WG features each) x TOKW tokens; warp FW is the
// producer: lane 0 streams each 128-K stage with 1D bulk copies (TMA) into a STAGES-deep ring
// guarded by mbarriers, so no CTA-wide barrier runs in the K loop.
//
// Global layouts (all produced by this engine):
//   weights  [kb][g] chunks of 1024 B; inside, word (lane l, w) at (w>>2)*512 + l*16 + (w&3)*4
//   wscale   [kb][g][gid][4] halves (64 B per chunk)
//   acts     [tile64][kb][64 rows][64 B], 16-B chunk c of row r stored at chunk c ^ ((r>>1)&3)
//   ascale   [tile64][kb][64 rows][8 B] UE4M3, one per 16 K; ag [row] FP32
// WF 0: ternary trits (1024 B chunk + 64 B fp16 group scales); WF 1: NVFP4 weights (2048 B chunk of
// E2M1 nibbles, word (lane l, w) at (w>>2)*512 + l*16 + (w&3)*4 holding a[m=w>>3][h=(w>>2)&1][w&3],
// + 256 B UE4M3 scales [32 rows][8 k16-blocks]) with an FP32 scale per output row.
// WF 2: trits as WF 0, but the group scale is E4M3 [gid][4 rows (m,r)] in the MMA's scale-A operand
// (relative to an FP32 row scale RS), so the MMAs accumulate directly with no per-block epilogue.
// WF 3: dense5 lane-fragment labels, 832 B chunk + unchanged FP16 W128 scales.
// WF 4 / 5: E2M1-subset codes, 3 / 2 bits per weight ({±0.5,±1.5,±3,±6} / {±0.5,±1.5}), UE4M3 scale per 32 K
// ([32 rows][4] bytes = 128 B) and an FP32 row scale (RS). Lane l reads 12 / 8 words at g*512 + l*16 + 4i;
// register (mm, h, o) (row 16mm + 8(o&1) + gid, k 64h + 32(o>>1) + 8tq + j) decodes with masks and shifts:
// 2-bit word p holds registers 2p (sign at nibble bit 3, magnitude at bit 1) and 2p+1 (bits 2, 0);
// 3-bit words 3q, 3q+1 hold sign/high-magnitude bits of registers 4q..4q+3 the same way, word 3q+2 their low
// magnitude bits (registers 4q, 4q+1, 4q+2, 4q+3 at nibble bits 1, 0, 3, 2). Nibble = s<<3 | (2m+1).
#if defined(H5_WF2_SCALE_PRMT) && H5_WF2_SCALE_PRMT != 0 && H5_WF2_SCALE_PRMT != 1
#error "H5_WF2_SCALE_PRMT must be 0 or 1"
#endif
// WF6: pre-expanded WF2 MMA words (2048 B) + [gid][4] broadcast E4M3 scale words (128 B).
// WF7: packed WF2 trits (1024 B) + the same broadcast E4M3 scale words (128 B).
// WF8: Dense5 labels (832 B) + the WF2 E4M3 scale layout (64 B).
template <int WF> struct WFmt { static constexpr int CW = WF == 1 || WF == 6 ? 2048 : (WF == 3 || WF == 8 || WF == 9 || WF == 10) ? 832 : WF == 4 ? 1536 : 1024, CS = WF == 1 ? 256 : (WF >= 4 && WF != 8 && WF != 10) ? 128 : 64; };
__device__ __forceinline__ void lowbit_registers3(const uint8_t* chunk, int lane, unsigned (&r)[16]) {
  const uint4 v0 = *(const uint4*)(chunk + lane * 16), v1 = *(const uint4*)(chunk + 512 + lane * 16), v2 = *(const uint4*)(chunk + 1024 + lane * 16);
  const unsigned w[12] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, v2.x, v2.y, v2.z, v2.w};
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const unsigned X = w[3 * q], Y = w[3 * q + 1], Z = w[3 * q + 2];
    r[4 * q]     = (X & 0xCCCCCCCCu) | (Z & 0x22222222u) | 0x11111111u;
    r[4 * q + 1] = ((X << 2) & 0xCCCCCCCCu) | ((Z << 1) & 0x22222222u) | 0x11111111u;
    r[4 * q + 2] = (Y & 0xCCCCCCCCu) | ((Z >> 2) & 0x22222222u) | 0x11111111u;
    r[4 * q + 3] = ((Y << 2) & 0xCCCCCCCCu) | ((Z >> 1) & 0x22222222u) | 0x11111111u;
  }
}
__device__ __forceinline__ void lowbit_registers2(const uint8_t* chunk, int lane, unsigned (&r)[16]) {
  const uint4 v0 = *(const uint4*)(chunk + lane * 16), v1 = *(const uint4*)(chunk + 512 + lane * 16);
  const unsigned w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
  for (int p = 0; p < 8; p++) { r[2 * p] = (w[p] & 0xAAAAAAAAu) | 0x11111111u; r[2 * p + 1] = ((w[p] << 1) & 0xAAAAAAAAu) | 0x11111111u; }
}
template <int FW, int WG, int TOKW, int WF = 0, int PIPE = 0> struct WsGeom {
  static constexpr int NG = FW * WG;                        // 32-feature groups per CTA
  static constexpr int SW = NG * WFmt<WF>::CW, SWS = NG * WFmt<WF>::CS, SA = TOKW * 64, SAS = TOKW * 8;
  static constexpr int STAGE = SW + SWS + SA + SAS;
#ifndef H5_STAGE_KB
#define H5_STAGE_KB 96
#endif
  static constexpr int STAGES = PIPE ? PIPE : ((H5_STAGE_KB * 1024) / STAGE > 8 ? 8 : (H5_STAGE_KB * 1024) / STAGE);
  static_assert(STAGES > 0 && STAGES <= 24);
  static constexpr int SMEM = STAGES * STAGE + 2 * STAGES * 8 + (WF == 9 ? 2048 : (WF == 3 || WF == 8) ? 1024 : 0);
};

// Upper bound on ring stages issued before the predecessor wait (H5_GEMM_PREWAIT; unset = full ring).
static __device__ int g_prewait_cap = 1 << 30;
void set_prewait_cap(int cap) {
  if (cap < 0) abort();
  if (cudaMemcpyToSymbol(g_prewait_cap, &cap, sizeof(cap)) != cudaSuccess) abort();
}
// Bounded pre-wait L2 prefetch (H5_GEMM_L2PF=N, or h5_set_l2pf_stages): before the predecessor wait,
// the producer also issues cp.async.bulk.prefetch.L2 for the N K-blocks after its pre-wait ring, so
// DRAM streams this CTA's next weights while the predecessor (row prep, conv, GDN) still runs.
// 0 = off. Independent of the full-slice H5_NO_L2PF prefetch.
static __device__ int g_l2pf_stages = 0;
void set_l2pf_stages(int n) {
  if (n < 0 || n > 64) abort();
  if (cudaMemcpyToSymbol(g_l2pf_stages, &n, sizeof(n)) != cudaSuccess) abort();
}
// Fused row preparation (RMSNorm * weight, sign, 1024-point Hadamard, NVFP4 quantization with a
// per-row FP32 scale) executed by the consumer warps of the GEMM that reads it. The arithmetic is
// the k_prep2 sequence in the same order, so the quantized operand equals the separate prep's.
struct FuseArgs { const float* x; int ldx; const float* norm_w; const float* signs; int n; };
template <int TOKW> struct FuseGeom {
  static constexpr int MAXCH = 8;   // n <= 8192
  // one real row: [kb][64 B] operand + [kb][8 B] scales, one zero row (72 B), RMS/amax scratch, g
  static constexpr int bytes(int n) { return (n / 128) * 72 + 128 + 2 * MAXCH * 4 + 16; }
};
// One-row variant: each consumer warp owns at most TPW 1024-feature chunks held in registers for the
// whole preparation; norm weights and signs are loaded before the predecessor wait.
template <int FW, int TPW> struct FuseRegs { float4 nw[TPW][8], sg[TPW][8]; };
template <int FW, int TPW>
__device__ __forceinline__ void fused_preload(const FuseArgs& fa, FuseRegs<FW, TPW>& k) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nch = fa.n >> 10;
#pragma unroll
  for (int j = 0; j < TPW; j++) {
    const int c = warp + j * FW;
    if (c >= nch) continue;
    const int e0 = c * 1024 + lane * 32;
#pragma unroll
    for (int i = 0; i < 8; i++) { k.nw[j][i] = *(const float4*)(fa.norm_w + e0 + 4 * i); k.sg[j][i] = *(const float4*)(fa.signs + e0 + 4 * i); }
  }
}
template <int FW, int TOKW, int TPW>
__device__ __forceinline__ void fused_row_prep1(const FuseArgs& fa, const FuseRegs<FW, TPW>& k, uint8_t* abuf, uint8_t* sbuf, uint8_t* zrow, float* ssv, float* amv, float* gsm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nch = fa.n >> 10;
  if (threadIdx.x < 18) ((uint32_t*)zrow)[threadIdx.x] = 0u;
  float v[TPW][32];
#pragma unroll
  for (int j = 0; j < TPW; j++) {
    const int c = warp + j * FW;
    if (c >= nch) continue;
    const float* x = fa.x + c * 1024 + lane * 32;
#pragma unroll
    for (int i = 0; i < 8; i++) { const float4 t = *(const float4*)(x + 4 * i); v[j][4*i] = t.x; v[j][4*i+1] = t.y; v[j][4*i+2] = t.z; v[j][4*i+3] = t.w; }
  }
#pragma unroll
  for (int j = 0; j < TPW; j++) {
    const int c = warp + j * FW;
    if (c >= nch) continue;
    float ss = 0.f;
#pragma unroll
    for (int q = 0; q < 32; q++) ss = fmaf(v[j][q], v[j][q], ss);
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if (!lane) ssv[c] = ss;
  }
  asm volatile("bar.sync 1, %0;" :: "r"(FW * 32) : "memory");
  float ss = 0.f;
  for (int c = 0; c < nch; c++) ss += ssv[c];
  const float inv = rsqrtf(ss / (float)fa.n + NORM_EPS);
#pragma unroll
  for (int j = 0; j < TPW; j++) {
    const int c = warp + j * FW;
    if (c >= nch) continue;
    float* w = v[j];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 nw = k.nw[j][i], sg = k.sg[j][i];
      w[4*i] *= inv * nw.x; w[4*i+1] *= inv * nw.y; w[4*i+2] *= inv * nw.z; w[4*i+3] *= inv * nw.w;
      w[4*i] *= sg.x; w[4*i+1] *= sg.y; w[4*i+2] *= sg.z; w[4*i+3] *= sg.w;
    }
#pragma unroll
    for (int len = 1; len < 32; len <<= 1)
#pragma unroll
      for (int q = 0; q < 32; q++)
        if (!(q & len)) { const float a = w[q], b = w[q + len]; w[q] = a + b; w[q + len] = a - b; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1)
#pragma unroll
      for (int q = 0; q < 32; q++) { const float b = __shfl_xor_sync(0xffffffff, w[q], o); w[q] = (lane & o) ? b - w[q] : w[q] + b; }
    float am = 0.f;
#pragma unroll
    for (int q = 0; q < 32; q++) { w[q] *= 0.03125f; am = fmaxf(am, fabsf(w[q])); }
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, o));
    if (!lane) amv[c] = am;
  }
  asm volatile("bar.sync 1, %0;" :: "r"(FW * 32) : "memory");
  float am = 0.f;
  for (int c = 0; c < nch; c++) am = fmaxf(am, amv[c]);
  const float g = fmaxf(am / (6.f * 448.f), 1e-30f), ig = 1.f / g;
  if (threadIdx.x == 0) gsm[0] = g;
#pragma unroll
  for (int j = 0; j < TPW; j++) {
    const int c = warp + j * FW;
    if (c >= nch) continue;
    const float* w = v[j];
    float bm0 = 0.f, bm1 = 0.f;
#pragma unroll
    for (int q = 0; q < 16; q++) { bm0 = fmaxf(bm0, fabsf(w[q])); bm1 = fmaxf(bm1, fabsf(w[q + 16])); }
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
          const int i = 16 * b + 8 * h + 2 * q;
          wd |= (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(w[i] * rs, w[i + 1] * rs), __NV_E2M1, cudaRoundNearest) << (8 * q);
        }
        words[2 * b + h] = wd;
      }
    }
    const int kb = (c * 1024 + lane * 32) >> 7, cc = lane & 3;
    *(uint4*)(abuf + kb * 64 + (cc << 4)) = make_uint4(words[0], words[1], words[2], words[3]);
    *(uint16_t*)(sbuf + kb * 8 + 2 * cc) = (uint16_t)sbs[0] | ((uint16_t)sbs[1] << 8);
  }
  asm volatile("bar.sync 1, %0;" :: "r"(FW * 32) : "memory");
}

template <int FW, int WG, int TOKW, int MODE, int WF, int PIPE = 0, bool MICRO = false, bool CARRIER = false, bool DEGREE8 = false, bool FUSE = false, int TAIL = TAIL_NONE>   // MODE 0 store, 1 add, 2 atomic add, 3 silu(gate)*up (WG 2, interleaved groups)
__device__ __forceinline__ void gemm_fp4_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ AS, const float* __restrict__ AG,
                                                              int R, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int N,
                                                              float* __restrict__ out, int ldo, const float* __restrict__ RS, const FuseArgs& fa,
                                                              const TailArgs& ta, const uint32_t* global_book) {
  static_assert(TAIL == TAIL_NONE || (!FUSE && !MICRO && !DEGREE8 && TOKW == 8), "producer-tail prep: eight-row tiles of the plain or carrier GEMM");
#ifndef H5_PDL_EARLY_GEMM_RELEASE
  static_assert(TAIL == TAIL_NONE, "producer-tail prep clears after the consumer wait of the early-release GEMM");
#endif
  static_assert(!FUSE || (CARRIER && WF == 3 && MODE == 3 && TOKW == 8 && !MICRO), "fused prep supports the dense5 carrier GU, eight-row tiles");
#ifndef H5_PDL_EARLY_GEMM_RELEASE
  static_assert(!FUSE, "fused prep requires the early-release consumer wait");
#endif
  static_assert(!DEGREE8 || (MODE == 3 && WG == 2 && WF == 0 && !CARRIER && !MICRO), "F07 Degree8 requires original-scale interleaved GU");
  static_assert(!CARRIER || WF == 0 || WF == 3 || WF == 9 || WF == 10, "row scale carrier supports native two-bit or dense5 only");
#ifdef H5_PROFILE_GEMM_STREAM
  static_assert(!CARRIER, "scale carrier has a distinct tensor accumulation schedule");
#endif
  static_assert(!MICRO || (WF == 3 && TOKW == 8), "microscope supports dense5 token-eight shapes");
#if defined(EXP_NOWAIT) || defined(EXP_NOMMA) || defined(H5_PROFILE_GEMM_STREAM)
  static_assert(!MICRO, "microscope requires the intact integrated GEMM, not an ablation/stream variant");
  static_assert(!CARRIER, "scale carrier requires intact GEMM, not an ablation/stream variant");
#endif
  // Diagnostic control: flag and PDL arms release at the same post-K-loop landmark,
  // after a full-grid acquire. Fused row prep and producer-tail prep keep their own protocol.
  constexpr bool matched_release = gemm_matched_release_enabled && !FUSE && (TAIL == TAIL_NONE || TAIL == TAIL_SIDE);
  micro::Trace<MICRO> trace;
  trace.begin(micro::Setup);
  using G = WsGeom<FW, WG, TOKW, WF, PIPE>;
  constexpr int CW = WFmt<WF>::CW, CS = WFmt<WF>::CS;
  constexpr int NT = TOKW / 8, MT = 2 * WG, STAGES = G::STAGES;
constexpr int empty_arrivals = FW;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = (uint64_t*)(smem + STAGES * G::STAGE);
  uint64_t* empty = full + STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nkb_all = K / 128;
  const int kb0 = (int)((long)nkb_all * blockIdx.z / gridDim.z), kb1 = (int)((long)nkb_all * (blockIdx.z + 1) / gridDim.z);
  const int nkb = kb1 - kb0;
  const int NGall = N / 32;
  const int g0 = blockIdx.y * G::NG;
  const int t0 = blockIdx.x * TOKW;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], empty_arrivals); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  uint32_t* book = nullptr;
  if constexpr (WF == 9) {
    book = (uint32_t*)(smem + G::SMEM - 2048);
    for (int code=tid;code<512;code+=blockDim.x) {
      unsigned c=code&255,g=0,u=0,ng=0,nu=0;
      for (int j=0;j<5;j++,c/=3) {
        unsigned d=c%3,v=d==0?10u:d==2?2u:0u;
        if (((j+(code>>8))&1)==0) {g|=v<<(4*ng++);}else{u|=v<<(4*nu++);}
      }
      book[code]=g|(u<<16);
    }
  } else if constexpr (WF == 3 || WF == 8) {
    book = (uint32_t*)(smem + G::SMEM - 1024);
    for(int code=tid;code<256;code+=blockDim.x) {
      unsigned q=code, value=0;
#pragma unroll
      for(int j=0;j<5;j++,q/=3) value|=(q%3==0?10u:q%3==2?2u:0u)<<(4*j);
      book[code]=value;
    }
  }
  __syncthreads();
  trace.end(micro::Setup);

  if (warp == FW) {   // producer
#ifdef EXP_NOWAIT
    return;
#endif
    if (lane == 0) {
      const uint64_t weight_policy = weight_tma_policy();
      const int ng = min(G::NG, NGall - g0);
      const int tile = t0 / 64, toff = t0 % 64;
      const unsigned tx = ng * CW + (CARRIER ? 0 : ng * CS) + (FUSE ? 0 : TOKW * 64 + TOKW * 8);
      // weights do not depend on the previous kernel: start streaming this CTA's whole weight slice
      // into L2 and issue the first ring of it into shared memory, then wait for the producer kernel
      trace.begin(micro::WeightPrefetch);
      if (l2pf_enabled()) for (int kb = STAGES; kb < nkb; kb++) {
        const size_t kk = kb0 + kb;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(W + (kk * NGall + g0) * CW), "r"(ng * CW) : "memory");
      }
      const int npre = min(min(nkb, STAGES), g_prewait_cap);
      for (int kb = npre, kend = min(nkb, npre + g_l2pf_stages); kb < kend; kb++) {
        const size_t kk = kb0 + kb;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(W + (kk * NGall + g0) * CW), "r"(ng * CW) : "memory");
      }
      for (int kb = 0; kb < npre; kb++) {
        uint8_t* st = smem + kb * G::STAGE;
        mbar_expect(&full[kb], tx);
        const size_t kk = kb0 + kb;
        bulk_weight_g2s(st, W + (kk * NGall + g0) * CW, ng * CW, &full[kb], weight_policy);
        if constexpr (!CARRIER) bulk_g2s(st + G::SW, S + (kk * NGall + g0) * CS, ng * CS, &full[kb]);
      }
      trace.end(micro::WeightPrefetch);
      if constexpr (matched_release) H5_LIT_MARKX(0);
      trace.begin(micro::Predecessor);
      if constexpr (!matched_release) pdl_early_trigger(4u);
      if (TAIL != TAIL_NONE && ta.wait_sig) tail_flag_wait(ta.wait_sig, ta.wait_expect);
      else pdl_wait();
      trace.end(micro::Predecessor);
      if constexpr (matched_release) H5_LIT_MARKX(1);
#ifdef H5_PDL_EARLY_GEMM_RELEASE
      if constexpr (!matched_release) pdl_launch();
#endif
      for (int kb = 0; kb < nkb; kb++) {
        const int s = kb % STAGES;
        uint8_t* st = smem + s * G::STAGE;
        const size_t kk = kb0 + kb;
        if (kb >= npre) {
          trace.begin(micro::EmptyWait, kb);
          mbar_wait(&empty[s], ((kb / STAGES) & 1) ^ 1);
          trace.end(micro::EmptyWait, kb);
          trace.begin(micro::WeightIssue, kb);
          mbar_expect(&full[s], tx);
          bulk_weight_g2s(st, W + (kk * NGall + g0) * CW, ng * CW, &full[s], weight_policy);
          if constexpr (!CARRIER) bulk_g2s(st + G::SW, S + (kk * NGall + g0) * CS, ng * CS, &full[s]);
          trace.end(micro::WeightIssue, kb);
        }
        trace.begin(micro::ActivationIssue, kb);
        constexpr int SUB = TOKW < 64 ? TOKW : 64;
        if constexpr (!FUSE)
#pragma unroll
        for (int u = 0; u < TOKW / SUB; u++) {
          const size_t rb = ((size_t)(tile + u) * nkb_all + kk) * 64 + toff;
          bulk_g2s(st + G::SW + G::SWS + u * SUB * 64, A + rb * 64, SUB * 64, &full[s]);
          bulk_g2s(st + G::SW + G::SWS + G::SA + u * SUB * 8, AS + rb * 8, SUB * 8, &full[s]);
        }
        trace.end(micro::ActivationIssue, kb);
      }
      if constexpr (matched_release) {
        pdl_wait();
        H5_LIT_MARKX(2);
        pdl_launch();
        H5_LIT_MARKX(3);
      } else if (TAIL != TAIL_NONE && ta.wait_sig) pdl_wait();   // complete only after the predecessor grid
    }
    trace.finish();
    return;
  }

  const int gid = lane >> 2, tq = lane & 3;
#if defined(H5_WF2_SCALE_PRMT) && H5_WF2_SCALE_PRMT
  unsigned scale_sel0, scale_sel1;
  // C++ selectors are rematerialized in the K loop; opaque setup keeps them outside it.
  if constexpr (WF == 2 || WF == 8)
    asm volatile("and.b32 %0, %2, 1; mul.lo.u32 %0, %0, 0x1111; add.u32 %1, %0, 0x2222;"
                 : "=&r"(scale_sel0), "=&r"(scale_sel1) : "r"(tq));
#endif
  const int gw = warp * WG;                       // first group of this warp inside the CTA
  const bool live = g0 + gw < NGall;
  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[m][n][i] = 0.f;
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};

  constexpr int FTPW = (8 + FW - 1) / FW;   // chunks per warp at n <= 8192, one row
  FuseRegs<FW, FUSE ? FTPW : 1> fregs;
  if constexpr (FUSE) fused_preload<FW, FTPW>(fa, fregs);
#ifdef H5_PDL_EARLY_GEMM_RELEASE
  // Release only after acquiring the preceding grid, so successors may safely read ancestors.
  // Successors still acquire this grid before reading its outputs.
  trace.begin(micro::Predecessor);
  if constexpr (!matched_release) pdl_early_trigger(4u);
  // Flag start: only the producer lane polls; consumers read activations through the stage mbarriers (whose
  // producer-side arrive follows its acquire) and AG after the main loop, so they need no poll of their own.
  if constexpr (!matched_release)
    if (!(TAIL != TAIL_NONE && ta.wait_sig)) { pdl_wait(); pdl_launch(); }
  trace.end(micro::Predecessor);
  if constexpr (TAIL != TAIL_NONE && !matched_release)
    if (!ta.wait_sig) tail_clear_share(ta, (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x, gridDim.x * gridDim.y * gridDim.z, tid, FW * 32);
#endif
  uint8_t* const fbuf = smem + G::SMEM;
  uint8_t* const fsbuf = fbuf + nkb_all * 64;
  uint8_t* const fzero = fsbuf + nkb_all * 8;
  float* const fss = (float*)(fzero + 128);
  float* const fam = fss + FuseGeom<TOKW>::MAXCH;
  float* const fgsm = fam + FuseGeom<TOKW>::MAXCH;
  if constexpr (FUSE) fused_row_prep1<FW, TOKW, FTPW>(fa, fregs, fbuf, fsbuf, fzero, fss, fam, fgsm);
  for (int kb = 0; kb < nkb; kb++) {
    const int s = kb % STAGES;
#ifdef EXP_NOWAIT
    const uint8_t* st = smem + (kb & 1) * G::STAGE;   // compute on stale smem; no producer dependency
#else
    trace.begin(micro::FullWait, kb);
    mbar_wait(&full[s], (kb / STAGES) & 1);
    // Flag start: release dependents once the first stage (after the producer's acquire) has landed, as
    // the griddepcontrol path does after its wait, not at entry (early dependents crowd the SMs).
    if constexpr (!matched_release && TAIL != TAIL_NONE) if (kb == 0 && ta.wait_sig) pdl_launch();
    if constexpr (matched_release) if (kb == 0) H5_LIT_MARKX(0);
    trace.end(micro::FullWait, kb);
    const uint8_t* st = smem + s * G::STAGE;
#endif
#ifdef H5_PROFILE_GEMM_STREAM
    if constexpr (WF == 3 && PIPE > 0) {
      static_assert(TOKW == 8);
      const uint8_t* as = st + G::SW + G::SWS;
      unsigned b0, b1, b2, b3;
      const int row = lane & 7, c = lane >> 3;
      ldsm4(b0, b1, b2, b3, as + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
      const uint2 bs = *(const uint2*)(as + G::SA + gid * 8);
#pragma unroll
      for (int q = 0; q < WG; q++) {
        uint32_t labels[7];
        dense5_labels(st + (gw + q) * CW, lane, labels);
        const uint2 sc = *(const uint2*)(st + G::SW + (gw + q) * CS + gid * 8);
        const __half2* sh = (const __half2*)&sc;
#pragma unroll
        for (int mm = 0; mm < 2; mm++) {
          unsigned a0[4], a1[4];
          const int w = mm * 8;
          a0[0] = dense5_word(labels, book, w); a0[2] = dense5_word(labels, book, w + 1);
          a0[1] = dense5_word(labels, book, w + 2); a0[3] = dense5_word(labels, book, w + 3);
          a1[0] = dense5_word(labels, book, w + 4); a1[2] = dense5_word(labels, book, w + 5);
          a1[1] = dense5_word(labels, book, w + 6); a1[3] = dense5_word(labels, book, w + 7);
          float d[4];
          mma_fp4(d, a0, b0, b1, bs.x, zero);
          mma_fp4(d, a1, b2, b3, bs.y, d);
          const float2 ws = __half22float2(sh[mm]);
          acc[q * 2 + mm][0][0] = fmaf(d[0], ws.x, acc[q * 2 + mm][0][0]);
          acc[q * 2 + mm][0][1] = fmaf(d[1], ws.x, acc[q * 2 + mm][0][1]);
          acc[q * 2 + mm][0][2] = fmaf(d[2], ws.y, acc[q * 2 + mm][0][2]);
          acc[q * 2 + mm][0][3] = fmaf(d[3], ws.y, acc[q * 2 + mm][0][3]);
        }
      }
      ring_release(&empty[s], lane == 0);
      continue;
    }
#endif
    trace.begin(micro::Decode, kb);
    unsigned a[MT][2][4];
    float wsc[MT][2];
    unsigned sca[MT][2];
    if constexpr (WF == 10) {
      static_assert(WG==2 && MODE==3 && CARRIER);
      // Each original native word is eight trits -> a 13-bit base3 code.
      // Gate/up sibling codes are adjacent; 32 codes occupy exactly 52B/lane.
      uint32_t labels[13];
#pragma unroll
      for(int i=0;i<13;i++)labels[i]=*(const uint32_t*)(st+gw*832+i*128+lane*4);
#pragma unroll
      for(int w=0;w<16;w++) {
        constexpr unsigned mask=8191;
        int b=(26*w)/32,sh=(26*w)%32;
        unsigned v=__funnelshift_r(labels[b],b+1<13?labels[b+1]:0,sh);
        a[w/8][(w/4)%2][w%4]=global_book[v&mask];
        a[2+w/8][(w/4)%2][w%4]=global_book[(v>>13)&mask];
      }
    } else if constexpr (WF == 9) {
      static_assert(WG==2 && MODE==3 && CARRIER);
      uint32_t labels[13],g[26],u[26];
#pragma unroll
      for(int i=0;i<13;i++)labels[i]=*(const uint32_t*)(st+gw*832+i*128+lane*4);
#pragma unroll
      for(int i=0;i<26;i++) {
        unsigned word=labels[i/2],a0=book[(word>>(16*(i%2)))&255],a1=book[256+((word>>(16*(i%2)+8))&255)];
        g[i]=(a0&0xfff)|((a1&255)<<12);u[i]=(a0>>16)|((a1>>16)<<8);
      }
#pragma unroll
      for(int w=0;w<16;w++) {
        constexpr unsigned mask=0xffffffffu;
        int b=(8*w)/5,sh=((8*w)%5)*4;
        unsigned gv=(g[b]>>sh)|(g[b+1]<<(20-sh)),uv=(u[b]>>sh)|(u[b+1]<<(20-sh));
        if(sh>=12){gv|=g[b+2]<<(40-sh);uv|=u[b+2]<<(40-sh);}
        a[w/8][(w/4)%2][w%4]=gv&mask;
        a[2+w/8][(w/4)%2][w%4]=uv&mask;
      }
    } else {
#pragma unroll
    for (int q = 0; q < WG; q++) {
      if constexpr (WF == 4 || WF == 5) {
        unsigned r[16];
        if constexpr (WF == 4) lowbit_registers3(st + (gw + q) * CW, lane, r); else lowbit_registers2(st + (gw + q) * CW, lane, r);
#pragma unroll
        for (int n = 0; n < 16; n++) a[q * 2 + (n >> 3)][(n >> 2) & 1][n & 3] = r[n];
        const uint8_t* sc = st + G::SW + (gw + q) * CS;
#pragma unroll
        for (int mm = 0; mm < 2; mm++) {
          const unsigned v = *(const unsigned*)(sc + (16 * mm + gid + 8 * (tq & 1)) * 4);
          sca[q * 2 + mm][0] = __byte_perm(v, 0, 0x1100); sca[q * 2 + mm][1] = __byte_perm(v, 0, 0x3322);
        }
      } else if constexpr (WF != 1 && WF != 6) {
        if constexpr (WF == 3 || WF == 8) {
          unsigned native[16];
          dense5_registers(st + (gw + q) * CW, lane, book, native);
#pragma unroll
          for (int mm = 0; mm < 2; mm++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
              const int w = mm * 4 + h * 2;
              unsigned* o = a[q * 2 + mm][h];
              o[0]=native[w*2]; o[2]=native[w*2+1];
              o[1]=native[(w+1)*2]; o[3]=native[(w+1)*2+1];
            }
        } else {
        const uint8_t* wc = st + (gw + q) * CW + lane * 16;
        const uint4 w0 = *(const uint4*)wc, w1 = *(const uint4*)(wc + 512);
        const unsigned wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int mm = 0; mm < 2; mm++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const unsigned r0 = wv[mm * 4 + h * 2], r1 = wv[mm * 4 + h * 2 + 1];
            unsigned* o = a[q * 2 + mm][h];
            o[0] = r0 & 0xAAAAAAAAu; o[2] = (r0 << 1) & 0xAAAAAAAAu;
            o[1] = r1 & 0xAAAAAAAAu; o[3] = (r1 << 1) & 0xAAAAAAAAu;
          }
        }
        if constexpr (WF == 7) {
          const uint8_t* sc = st + G::SW + (gw + q) * CS;
#pragma unroll
          for (int mm = 0; mm < 2; mm++) {
            const unsigned v = *(const unsigned*)(sc + gid * 16 + (2 * mm + (tq & 1)) * 4);
            sca[q * 2 + mm][0] = sca[q * 2 + mm][1] = v;
          }
        } else if constexpr (WF == 2 || WF == 8) {
          const uint32_t sb4 = *(const uint32_t*)(st + G::SW + (gw + q) * CS + gid * 4);
#pragma unroll
          for (int mm = 0; mm < 2; mm++) {
#if defined(H5_WF2_SCALE_PRMT) && H5_WF2_SCALE_PRMT
            const unsigned scale = __byte_perm(sb4, 0, mm ? scale_sel1 : scale_sel0);
#else
            const uint32_t b = (sb4 >> (8 * (2 * mm + (tq & 1)))) & 0xff;
            const unsigned scale = b * 0x01010101u;
#endif
            sca[q * 2 + mm][0] = sca[q * 2 + mm][1] = scale;
          }
        } else if constexpr (!CARRIER) {
          const uint2 sc = *(const uint2*)(st + G::SW + (gw + q) * CS + gid * 8);
          const __half2* sh = (const __half2*)&sc;
          const float2 s01 = __half22float2(sh[0]), s23 = __half22float2(sh[1]);
          wsc[q * 2][0] = s01.x; wsc[q * 2][1] = s01.y; wsc[q * 2 + 1][0] = s23.x; wsc[q * 2 + 1][1] = s23.y;
        }
      } else {
        const uint8_t* wc = st + (gw + q) * CW + lane * 16;
#pragma unroll
        for (int g4 = 0; g4 < 4; g4++) {
          const uint4 wv = *(const uint4*)(wc + g4 * 512);
          unsigned* o = a[q * 2 + (g4 >> 1)][g4 & 1];
          o[0] = wv.x; o[1] = wv.y; o[2] = wv.z; o[3] = wv.w;
        }
        const uint8_t* sc = st + G::SW + (gw + q) * CS;
#pragma unroll
        for (int mm = 0; mm < 2; mm++) {
          if constexpr (WF == 6) {
            const unsigned v = *(const unsigned*)(sc + gid * 16 + (2 * mm + (tq & 1)) * 4);
            sca[q * 2 + mm][0] = sca[q * 2 + mm][1] = v;
          } else {
            const uint2 v = *(const uint2*)(sc + (16 * mm + gid + 8 * (tq & 1)) * 8);
            sca[q * 2 + mm][0] = v.x; sca[q * 2 + mm][1] = v.y;
          }
        }
      }
    }
    if constexpr (MICRO) {
#pragma unroll
      for (int m = 0; m < MT; m++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
          for (int i = 0; i < 4; i++) trace.consume(a[m][h][i]);
          if constexpr (WF == 0 || WF == 3) trace.consume(wsc[m][h]);
          else trace.consume(sca[m][h]);
        }
      }
    }
    } // original independent-weight decoder
    trace.end(micro::Decode, kb);
    trace.begin(micro::ActivationOperand, kb);
    const uint8_t* as = st + G::SW + G::SWS;
    const uint8_t* ssc = as + G::SA;
#ifdef EXP_NOMMA
    acc[0][0][0] += __int_as_float(a[0][0][0] ^ a[MT - 1][1][3]) + wsc[0][0] + (float)sca[0][0] + as[lane] + ssc[lane];
    ring_release(&empty[s], lane == 0);
    continue;
#endif
    // Large token tiles with direct accumulation: activation fragments are software-pipelined one token pair
    // ahead in distinct registers, and the MMAs of a pair are ordered so dependent accumulations sit four
    // instructions apart. (Loading every n into the same registers exposed the ldmatrix latency NT times per
    // K block with two consumer warps per scheduler.)
#ifndef H5_GEMM_PIPE
    constexpr bool PIPED = false;   // measured neutral (bench/batch_gemm.cu at 512 rows: within 1%); opt-in build define
#else
    constexpr bool PIPED = !FUSE && !MICRO && NT >= 2 && NT % 2 == 0 && (CARRIER || (WF != 0 && WF != 3));
#endif
    if constexpr (PIPED) {
      unsigned bb[2][2][4]; uint2 bsb[2][2];
      auto loadp = [&](int p, int slot) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int n = 2 * p + j, row = n * 8 + (lane & 7), c = lane >> 3;
          ldsm4(bb[slot][j][0], bb[slot][j][1], bb[slot][j][2], bb[slot][j][3], as + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
          bsb[slot][j] = *(const uint2*)(ssc + (n * 8 + gid) * 8);
        }
      };
      loadp(0, 0);
#pragma unroll
      for (int p = 0; p < NT / 2; p++) {
        const int cur = p & 1;
        if (p + 1 < NT / 2) loadp(p + 1, cur ^ 1);
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int m = 0; m < MT; m++)
              mma_fp4(acc[m][2 * p + j], a[m][h], bb[cur][j][2 * h], bb[cur][j][2 * h + 1], h ? bsb[cur][j].y : bsb[cur][j].x,
                      acc[m][2 * p + j], CARRIER ? 0x38383838u : sca[m][h]);
      }
    }
#pragma unroll
    for (int n = 0; n < (PIPED ? 0 : NT); n++) {
      const int row = n * 8 + (lane & 7), c = lane >> 3;
      unsigned b0, b1, b2, b3;
      if constexpr (FUSE) ldsm4(b0, b1, b2, b3, row ? fzero : fbuf + (kb0 + kb) * 64 + (c << 4));
      else ldsm4(b0, b1, b2, b3, as + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
      const uint2 bs = FUSE ? *(const uint2*)(gid ? fzero : fsbuf + (kb0 + kb) * 8) : *(const uint2*)(ssc + (n * 8 + gid) * 8);
      if constexpr (MICRO) { trace.consume(b0); trace.consume(b1); trace.consume(b2); trace.consume(b3); trace.consume(bs.x); trace.consume(bs.y); }
      trace.end(micro::ActivationOperand, kb);
      trace.begin(micro::MmaScale, kb);
#pragma unroll
      for (int m = 0; m < MT; m++) {
        if (CARRIER || (WF != 0 && WF != 3)) {
          mma_fp4(acc[m][n], a[m][0], b0, b1, bs.x, acc[m][n], CARRIER ? 0x38383838u : sca[m][0]);
          mma_fp4(acc[m][n], a[m][1], b2, b3, bs.y, acc[m][n], CARRIER ? 0x38383838u : sca[m][1]);
          continue;
        }
        float d[4];
        mma_fp4(d, a[m][0], b0, b1, bs.x, zero);
        mma_fp4(d, a[m][1], b2, b3, bs.y, d);
        acc[m][n][0] = fmaf(d[0], wsc[m][0], acc[m][n][0]);
        acc[m][n][1] = fmaf(d[1], wsc[m][0], acc[m][n][1]);
        acc[m][n][2] = fmaf(d[2], wsc[m][1], acc[m][n][2]);
        acc[m][n][3] = fmaf(d[3], wsc[m][1], acc[m][n][3]);
      }
    }
    if constexpr (MICRO) {
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
          for (int i = 0; i < 4; i++) trace.consume(acc[m][n][i]);
    }
    trace.end(micro::MmaScale, kb);
    trace.begin(micro::EmptyRelease, kb);
    ring_release(&empty[s], lane == 0);
    trace.end(micro::EmptyRelease, kb);
  }
  if constexpr (matched_release) {
    H5_LIT_MARKX(1);
    pdl_wait();
    H5_LIT_MARKX(2);
    pdl_launch();
    H5_LIT_MARKX(3);
  }
#ifndef H5_PDL_EARLY_GEMM_RELEASE
  trace.begin(micro::FinalPredecessor);
  pdl_wait();
  pdl_launch();
  trace.end(micro::FinalPredecessor);
#endif
  // Flag start: the clearing waits for the main loop, whose stages follow the producer's acquire of the
  // predecessor's done count (the predecessor read the cleared buffer before signalling).
  if constexpr (TAIL != TAIL_NONE)
    if (matched_release || ta.wait_sig) tail_clear_share(ta, (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x, gridDim.x * gridDim.y * gridDim.z, tid, FW * 32);
  if (!live && TAIL == TAIL_NONE) { trace.finish(); return; }
  trace.begin(micro::Epilogue);
  if (live) {
  if constexpr (MODE == 3) {   // gate group gw and up group gw+1 of the same 32 features: write silu(gate) * up
    if constexpr (WG == 2) {
      const int fg = (g0 + gw) >> 1;   // feature group in the hidden (FF) coordinate
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int t = t0 + n * 8 + tq * 2 + j;
          if (t >= R) continue;
          const float gt = FUSE ? fgsm[0] : activation_scale(AG + t);
          float* o = out + (size_t)t * ldo + fg * 32 + gid;
#pragma unroll
          for (int mm = 0; mm < 2; mm++) {
            if constexpr (DEGREE8) {
              const float2 h = f07_degree8_product(acc[mm][n][j] * gt, acc[mm][n][2 + j] * gt,
                                                   acc[2 + mm][n][j] * gt, acc[2 + mm][n][2 + j] * gt);
              o[mm * 16] = h.x; o[mm * 16 + 8] = h.y;
            } else {
#pragma unroll
              for (int r = 0; r < 2; r++) {
                float gv = acc[mm][n][r * 2 + j] * gt, uv = acc[2 + mm][n][r * 2 + j] * gt;
                if (CARRIER || (WF != 0 && WF != 3)) {   // row scales of the interleaved gate (group gw) and up (group gw + 1) features
                  gv *= RS[(g0 + gw) * 32 + mm * 16 + r * 8 + gid];
                  uv *= RS[(g0 + gw + 1) * 32 + mm * 16 + r * 8 + gid];
                }
                o[mm * 16 + r * 8] = gv / (1.f + __expf(-gv)) * uv;
              }
            }
          }
        }
    }
  } else {
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int t = t0 + n * 8 + tq * 2 + j;
      if (t >= R) continue;
      const float gt = activation_scale(AG + t);
      float* o = out + (size_t)t * ldo + (g0 + gw) * 32 + gid;
#pragma unroll
      for (int m = 0; m < MT; m++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          if (g0 + gw + (m >> 1) >= NGall) continue;   // second group of a WG=2 warp past N
          float v = acc[m][n][r * 2 + j] * gt;
          if (CARRIER || (WF != 0 && WF != 3)) v *= RS[(g0 + gw) * 32 + gid + m * 16 + r * 8];
          if (MODE == 2) atomicAdd(o + m * 16 + r * 8, v);
          else { if (MODE == 1) v += o[m * 16 + r * 8]; o[m * 16 + r * 8] = v; }
        }
    }
  }
  }
  }
  trace.end(micro::Epilogue);
  if (!matched_release && TAIL == TAIL_SIDE && ta.wait_sig) pdl_wait();
  if constexpr (TAIL == TAIL_CHUNK || TAIL == TAIL_NORM) {
    // Columns of this CTA in the prepared row's coordinate (MODE 3 writes FF features: half the groups).
    // The host admits only tiles that divide 1024, so a CTA lies inside exactly one chunk.
    constexpr int CB = MODE == 3 ? G::NG * 16 : G::NG * 32;
    static_assert(1024 % CB == 0, "producer-tail tiles must divide a 1024-feature chunk");
    constexpr int RPW = (TAIL_MAX_ROWS + FW - 1) / FW;
    __shared__ unsigned tail_word;
    const int c = (blockIdx.y * CB) >> 10;
    const unsigned expect = (unsigned)((1024 / CB) * gridDim.x * gridDim.z);
    const unsigned long long t_arr = ta.dbg ? tail_clock() : 0ull;
    TailPre<RPW> pre;
    tail_preload<RPW, FW>(ta, c, pre);
    if (tail_arrive(ta, c, expect, tid, FW * 32, &tail_word)) {
      if (ta.dbg && tid == 0) ta.dbg[c * TAIL_DBG_STAMPS] = t_arr;
      tail_finalise<TAIL, FW, RPW>(ta, c, tid, FW * 32, pre);
    }
  }
  trace.finish();
}

__global__ __launch_bounds__(288) void dict8_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means,const uint32_t*dict){
 gemm_fp4_body<8,2,32,3,10,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{},TailArgs{},dict);
}
__global__ __launch_bounds__(288) void native0_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means){
 gemm_fp4_body<8,2,32,3,0,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{},TailArgs{},nullptr);
}
} // h5
