// Ternary Bonsai 2 27B (qwen35) geometry. Semantics follow bonsai-halo/PLAN.md "Semantics pinned".
#pragma once
#include <cstdint>
#include <cstddef>
namespace h5 {
constexpr int D = 5120, FF = 17408, NLAYER = 64, VOCAB = 248320, HAD = 1024;
constexpr int GRAMMAR_WORDS = (VOCAB + 31) / 32;
constexpr int HV = 48, HK = 16, SS = 128, KDIM = HK * SS, VDIM = HV * SS, CONV_CH = 2 * KDIM + VDIM; // 2048, 6144, 10240
constexpr int NH = 24, NKV = 4, HD = 256, NROT = 64, Q_OUT = NH * HD * 2, KV_OUT = NKV * HD, ATTN_OUT = NH * HD;
constexpr int NATTN = 16, NGDN = 48;
constexpr float NORM_EPS = 1e-6f, ROPE_BASE = 1e7f;
constexpr int BLK = 128;   // ternary weight group / activation quant group
inline bool is_attn(int l) { return (l + 1) % 4 == 0; }
// fused projection widths
constexpr int GDN_IN = CONV_CH + VDIM;            // qkv | z           16384
constexpr int ATT_IN = Q_OUT + 2 * KV_OUT;        // q+gate | k | v    14336
constexpr int GU = 2 * FF;                        // gate | up          34816
}

// Callback advances a private proposal-prefix matcher by previous (-1 at position 0),
// then writes the full vocabulary mask. Returns allowed count, or a negative error.
using H5GrammarMask = int (*)(void*, int block, int position, int previous, uint32_t*, int words);
extern "C" {
int h5_set_grammar_masks(void*, int rows, const uint32_t* masks);
int h5_set_grammar_head_options(void*, int count, const int* thresholds, const int* bulk_select);
int h5_grammar_draft_timings(void*, double* body_s, double* select_s, double* token_wait_s, int* token_waits);
int h5_draft_grammar(void*, int, const int*, const int*, const int*, const int*, const int*, const int*, int*, H5GrammarMask, void*);
}
