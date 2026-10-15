// Binary Bonsai 27B persistent single-stream decode step ("bmega").
//
// One cooperative launch runs the whole token: embedding, 64 layers, final norm, head and argmax.
// Every CTA owns a fixed slice of every projection. Its slices are laid out at load time as one
// contiguous stream of 2304-byte chunks (16 rows x 1024 K: FP4-MMA-native sign words + FP16 K128
// scales), in exactly the order the CTA consumes them. A dedicated producer warp streams the CTA's
// whole-token sequence through a shared-memory ring with TMA bulk copies and never waits for
// activations: weights do not depend on them. Consumer warps meet at software grid barriers between
// dependent phases; the ring keeps DRAM busy across those barriers and the latency-bound phases.
//
// Activations are original A8 (max/127 per K128 block, round to nearest even), carried as three
// exact FP4 digit planes (q = a + 4b + 16c); every integer K128 dot is exact before the original
// FP16 weight scale and the activation scale. State (F16 GDN, F32 conv) and F16 KV use the engine's
// own layouts, so prefill and every other route share them.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace bmega {
constexpr int NCTA = 160;                 // divides the 320 out/down tiles
constexpr int NCW = 8;                    // consumer warps per CTA
constexpr int THREADS = (NCW + 1) * 32;   // + one producer warp
constexpr int CHUNK = 2304;               // 16 rows x 1024 K
constexpr int NSLOT = 24;                 // ring slots
constexpr int NPH = 64 * 4 + 1;           // weight phases: per layer IN, OUT, GU, DOWN; then HEAD
constexpr int GDN_ITEMS = 96;             // 48 v-heads x 2 column halves
constexpr int MAX_SPLITS = 40;            // attention key splits per kv head

struct LayerP {
  const float* attn_norm; const float* post_norm;
  const float* conv_w; const float* ssm_a; const float* ssm_dt; const float* ssm_norm;
  const float* q_norm; const float* k_norm;
  int attn, gl, al, pad;
};

struct Args {
  const uint8_t* stream;            // concatenated CTA chunk streams
  const long long* stream_off;      // [NCTA + 1] chunk offsets
  const int4* tab;                  // [NPH][NCTA]: tile0, ntiles, chunk_base, 0
  const LayerP* L;                  // [64]
  const uint8_t* emb;               // Q1 token embedding, raw GGUF blocks
  const float* out_norm;
  __half* gdn_state; float* conv_state; __half* kc; __half* vc;
  int slot, pos, tok, ctx;
  int observer; // 0 original, 1 digit-N, 2 scale-field, 3 tensor-native TF32
  float* x; float* conv_out; float* zbuf; float* ab; float* att; float* y; float* h;
  float* part; unsigned* att_cnt;
  uint8_t* actbuf;                  // 3 rotating encoded-activation exchange buffers
  float* ubuf;                      // GDN gated output values before encoding
  int lookahead;                    // L2 prefetch distance in chunks while the ring drains
  int lookahead_hi;                 // L2 prefetch distance while consumers are stalled
  unsigned step;                    // launch sequence number: generations of every readiness flag
  int nowait;                       // diagnostic: skip every data dependency wait (garbage output, timing floor)
  int pf_policy;                    // 1: L2 prefetch with evict_last hint
  int attn_keys;                    // keys per attention split
  unsigned long long* indone;       // [NCTA] input-projection epilogue generations
  unsigned* head_cnt;               // head completions (modular)
  unsigned long long* bar; unsigned long long* amax;   // amax: two step-alternating slots
  float* logits;                    // optional full logits
  int* out_tok;
  unsigned long long* prof;         // optional phase timestamps
  unsigned long long* arrive;       // optional [NCTA][512] barrier arrival/exit times
};

struct Model {
  uint8_t* stream = nullptr; long long* stream_off = nullptr; int4* tab = nullptr; LayerP* L = nullptr;
  size_t stream_bytes = 0;
  float *x = nullptr, *conv_out = nullptr, *zbuf = nullptr, *ab = nullptr, *att = nullptr, *y = nullptr, *h = nullptr, *part = nullptr;
  uint8_t* actbuf = nullptr; float* ubuf = nullptr; unsigned long long* indone = nullptr; unsigned* head_cnt = nullptr; unsigned step = 0;
  unsigned* att_cnt = nullptr; unsigned long long* bar = nullptr; unsigned long long* amax = nullptr; int* out_tok = nullptr;
  unsigned long long* prof = nullptr; unsigned long long* arrive = nullptr;
};

// Raw Q1_0 rows (18-byte K128 blocks: FP16 scale + 16 sign bytes) of one projection's source tensors.
struct RowSource {
  std::vector<const uint8_t*> base;   // per source tensor
  std::vector<int> rows;              // rows per source tensor
  int K = 0;
  bool gu = false;                    // interleave tile: 8 rows of base[0], then the same 8 rows of base[1]
};
struct HostLayer { RowSource in, out, gu, down; };

// Build the per-CTA chunk streams and phase table; upload. Returns total stream bytes.
size_t build(Model& m, const std::vector<HostLayer>& layers, const RowSource& head, int nthreads);
int smem_bytes();
cudaError_t launch(const Args& a, cudaStream_t st);
// Accumulate CTA0 phase stamps of the last step (when Model::prof is set) and print every 64 steps.
void prof_collect(const Model& m);
void arrive_dump(const Model& m);
}
