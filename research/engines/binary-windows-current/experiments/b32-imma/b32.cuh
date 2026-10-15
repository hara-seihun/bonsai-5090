// B32 binary projection on the integer tensor pipe, original Q1 signs and FP16 K128 scales.
//
// Weights stay in the existing packed1 coordinates (signs [K/128][4][N] uint32, scales
// [K/128][N] half); nothing is duplicated or expanded in memory. A 32-bit sign word w covers
// 32 consecutive K. The tensor-core operand register for byte position i is taken directly as
// rotr(w, j-1) & 0x02020202, i.e. byte i <- 2*bit(j+8i): the MMA's K order inside a 32-chunk is a
// transpose of the natural order, so the activation producer writes the matching transpose.
// The integer MMA accumulator is seeded with MAGIC-DC, so it ends each K128 block holding the float
// bit pattern of M + (2P-DC) = M + signed dot, M = 2^23 + 2^14 (exact for |dot| <= 128*127).
// The activation scale sigma keeps 8 significant bits (max/127 rounded up), so M*sigma is exact and
// fma(f, sigma, -M*sigma) = dot*sigma exactly; one more FFMA applies the original FP16 weight scale.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace b32 {
constexpr int MAX_TOKENS = 32;          // carrier capacity per K128 block (zero-padded)
constexpr int MAGIC = 0x4B004000;       // bit pattern of 8404992.0f = 2^23 + 2^14
constexpr float MAGIC_F = 8404992.0f;
// sigma = max/127 rounded up to 8 significant bits (7 stored mantissa bits); q = rn(x / sigma) stays in [-127,127].
__host__ __device__ inline float sigma8(float max) {
  const float s = max / 127.f; uint32_t u; memcpy(&u, &s, 4); u = (u + 0xFFFFu) & 0xFFFF0000u; float r; memcpy(&r, &u, 4); return r;
}

enum class Status : uint32_t { Valid = 0, NonFinite = 1, ScaleRange = 2 };

// Activation carrier for a [R,K] input, R <= 32; one contiguous 4608 B block per K128:
//   [0,4096):    q words [4 n8-tiles][2 chunk pairs][32 lanes][4 words], MMA-transposed K order
//   [4096,4224): sigma[32] floats (sigma8(max); NaN marks an invalid group)
//   [4224,4480): seeded accumulators [4 n8-tiles][4 lane q][4] ints {MAGIC-DC[2q],MAGIC-DC[2q+1],same,same}
//   [4480,4608): -MAGIC_F*sigma[32] floats (exact)
struct Carrier {
  uint32_t* q;       // [K/128][1152] words
  uint32_t* error;   // optional aggregate status (atomicMax), cleared by the caller
};
constexpr int CARRIER_BLOCK = 4608;
size_t carrier_bytes(int k);

// Static tile-major weight coordinates: for tile t of TN rows and K128 block kb, one contiguous 18*TN byte
// record: signs [TN/16 m-tiles][8 g][rows g, g+8][4 packed1 words] then scales [TN/16][8 g][rows g, g+8] FP16;
// rows past N are zero. Each MMA lane reads its 8 sign words with two 16-byte loads and both scales with one.
// Same 18 B per (row, K128) as the checkpoint; this replaces packed1 for the tensors that use it.
size_t tiled_bytes(int n, int k, int tn);
cudaError_t tile_from_packed1(const uint32_t* signs, const __half* scales, int n, int k, int tn, uint8_t* out, cudaStream_t st);
int plan_tile_rows(int variant);

// F32 x[R,K] (row stride ldx) -> carrier. q = rn(x*127/max) per (row, K128), max==0 -> q=0,sigma=0.
// Rows R..rows_padded-1 (rows_padded = 8*ceil(R/8)) are zero-filled. Invalid groups set sigma=NaN.
cudaError_t produce(const float* x, int ldx, int k, int r, Carrier out, cudaStream_t st);

struct Plan {
  int n, k, r;        // N output features, K inputs, R tokens (1..32)
  int ctas;           // persistent grid size (all CTAs co-resident)
  int variant;        // kernel configuration index
  int epi;            // 0 store, 1 split store, 2 fused SiLU(gate)*up -> next carrier
};
// Workspace: a fixed COUNTER_BYTES region of per-tile arrival counters (zero-initialised once by the
// caller; every launch leaves it zero again) followed by stream-K partials. One workspace may be shared
// by every plan launched in order on one stream.
constexpr size_t COUNTER_BYTES = 65536;
size_t workspace_bytes(const Plan& p);
cudaError_t select(int n, int k, int r, Plan* out, int variant = -1, int epi = 0);
// y[t*ldy + n] = sum + beta*y   (beta must be 0 or 1)
cudaError_t project(const Plan& p, const uint8_t* tiled, Carrier in,
                    float* y, int ldy, float beta, void* workspace, cudaStream_t st);
// rows < split -> y[t*ldy + row]; rows >= split -> y2[t*ldy2 + row - split]
cudaError_t project_split(const Plan& p, const uint8_t* tiled, Carrier in, float* y, int ldy, int split,
                          float* y2, int ldy2, void* workspace, cudaStream_t st);
// Fused GU: tiled_gu from tile_gu_from_packed1 (N = 2*FF); writes the A8 carrier of silu(gate)*up (K = FF) to out.
cudaError_t project_silu(const Plan& p, const uint8_t* tiled_gu, Carrier in, Carrier out, void* workspace, cudaStream_t st);
// GU tile image: each 256-row tile holds 128 gate rows and their 128 up rows (per 32-row warp group:
// 16 gate rows then the same 16 features' up rows); packed1 source rows are [gate FF | up FF].
cudaError_t tile_gu_from_packed1(const uint32_t* signs, const __half* scales, int ff, int k, uint8_t* out, cudaStream_t st);
int variants();
const char* variant_name(int v);
}
