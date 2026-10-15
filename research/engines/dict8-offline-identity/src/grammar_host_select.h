#pragma once
#include "model.h"
#include "draft_contract.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#if defined(__FAST_MATH__)
#error "grammar host selection requires IEEE FP32 arithmetic, not fast math"
#endif

namespace h5 {
constexpr int GRAMMAR_HOST_RANK = 256;
constexpr int GRAMMAR_HOST_TOPK = 16;

template<class T> struct GrammarHostSpan { T* data; size_t count; };
enum class GrammarHostError {
  None, InvalidGeometry, BufferTooSmall, InvalidAnchor, UnsupportedArithmetic,
  CallbackFailure, CallbackException, EmptyMask, InvalidMask, NoFiniteCandidate,
  NonfiniteTransition
};
struct GrammarHostResult {
  GrammarHostError error;
  size_t completed_rows;
  int block, position, callback_status;
  bool ok() const noexcept { return error == GrammarHostError::None; }
};

// Packed prediction rows, excluding the anchor query: row = block * proposals + position.
// All spans remain caller-owned; outputs before completed_rows are valid on error.
// The callback's matcher has advanced and must be discarded on error, not retried.
struct GrammarHostSelect {
  int blocks, proposals, vocab, rank;
  GrammarHostSpan<const float> logits;     // [blocks * proposals][vocab]
  GrammarHostSpan<const float> hproj;      // [blocks * proposals][rank]
  GrammarHostSpan<const uint16_t> pred;    // [vocab][rank], raw BF16, cached at load
  GrammarHostSpan<const uint16_t> succ;    // [vocab][rank], raw BF16, cached at load
  GrammarHostSpan<const int> anchors;      // [blocks]
  GrammarHostSpan<int> tokens;            // [blocks * proposals]
  GrammarHostSpan<float> confidence;      // [blocks * proposals]
  GrammarHostSpan<uint32_t> mask;          // scratch [(vocab + 31) / 32]
  H5GrammarMask callback;
  void* context;
};

namespace grammar_host_detail {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "selector requires binary32");
inline float bf16(uint16_t value) noexcept {
  const uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
// Stores force FP32 rounding between operations, preventing FMA/reassociation.
inline float mul(float a, float b) noexcept { volatile float r = a * b; return r; }
inline float add(float a, float b) noexcept { volatile float r = a + b; return r; }
inline float sub(float a, float b) noexcept { volatile float r = a - b; return r; }
inline float warp_sum(float* lanes) noexcept {
  // Only the lower half at each XOR stage can reach lane zero.
  for (int offset = 16; offset; offset >>= 1)
    for (int lane = 0; lane < offset; ++lane)
      lanes[lane] = add(lanes[lane], lanes[lane + offset]);
  return lanes[0];
}
template<class T> inline bool fits(GrammarHostSpan<T> span, size_t count) noexcept {
  return span.data && span.count >= count;
}
inline bool product(size_t a, size_t b, size_t& out) noexcept {
  if (b && a > std::numeric_limits<size_t>::max() / b) return false;
  out = a * b;
  return true;
}
}

inline GrammarHostResult grammar_host_select(const GrammarHostSelect& a) noexcept {
  using namespace grammar_host_detail;
  const auto fail = [](GrammarHostError e, size_t rows, int b, int p, int status = 0) {
    return GrammarHostResult{e, rows, b, p, status};
  };
  if (a.blocks <= 0 || a.proposals <= 0 || a.proposals > DRAFT_MAX_PROPOSALS ||
      a.vocab <= 0 || a.rank != GRAMMAR_HOST_RANK || !a.callback)
    return fail(GrammarHostError::InvalidGeometry, 0, -1, -1);
  size_t rows, logits_count, hp_count, cb_count;
  if (!product(size_t(a.blocks), size_t(a.proposals), rows) ||
      !product(rows, size_t(a.vocab), logits_count) ||
      !product(rows, size_t(a.rank), hp_count) ||
      !product(size_t(a.vocab), size_t(a.rank), cb_count))
    return fail(GrammarHostError::InvalidGeometry, 0, -1, -1);
  const size_t words = (size_t(a.vocab) + 31) / 32;
  if (!fits(a.logits, logits_count) || !fits(a.hproj, hp_count) ||
      !fits(a.pred, cb_count) || !fits(a.succ, cb_count) ||
      !fits(a.anchors, size_t(a.blocks)) || !fits(a.tokens, rows) ||
      !fits(a.confidence, rows) || !fits(a.mask, words))
    return fail(GrammarHostError::BufferTooSmall, 0, -1, -1);
  if (std::fegetround() != FE_TONEAREST)
    return fail(GrammarHostError::UnsupportedArithmetic, 0, -1, -1);
#if defined(__SSE__)
  if (_mm_getcsr() & 0x8040u) // CPU FTZ/DAZ would change CUDA's gradual-underflow arithmetic.
    return fail(GrammarHostError::UnsupportedArithmetic, 0, -1, -1);
#endif
  for (int b = 0; b < a.blocks; ++b)
    if (a.anchors.data[b] < 0 || a.anchors.data[b] >= a.vocab)
      return fail(GrammarHostError::InvalidAnchor, 0, b, -1);

  size_t completed = 0;
  for (int b = 0; b < a.blocks; ++b) {
    int previous = a.anchors.data[b];
    for (int p = 0; p < a.proposals; ++p) {
      const size_t r = size_t(b) * a.proposals + p;
      std::fill_n(a.mask.data, words, uint32_t(0));
      int allowed;
      try {
        allowed = a.callback(a.context, b, p, p ? previous : -1, a.mask.data, int(words));
      } catch (...) {
        return fail(GrammarHostError::CallbackException, completed, b, p);
      }
      if (allowed < 0)
        return fail(GrammarHostError::CallbackFailure, completed, b, p, allowed);
      if (allowed == 0)
        return fail(GrammarHostError::EmptyMask, completed, b, p);
      if (a.vocab % 32 && (a.mask.data[words - 1] >> (a.vocab % 32)))
        return fail(GrammarHostError::InvalidMask, completed, b, p, allowed);
      int actual = 0, first = -1;
      for (size_t w = 0; w < words; ++w) {
        const uint32_t bits = a.mask.data[w];
        actual += __builtin_popcount(bits);
        if (first < 0 && bits) first = int(w * 32 + __builtin_ctz(bits));
      }
      if (actual != allowed)
        return fail(GrammarHostError::InvalidMask, completed, b, p, allowed);
      if (allowed == 1) {
        previous = first;
        a.tokens.data[r] = previous;
        a.confidence.data[r] = FLT_MAX;
        ++completed;
        continue;
      }

      float topv[GRAMMAR_HOST_TOPK];
      int topi[GRAMMAR_HOST_TOPK];
      std::fill_n(topv, GRAMMAR_HOST_TOPK, -std::numeric_limits<float>::infinity());
      std::fill_n(topi, GRAMMAR_HOST_TOPK, std::numeric_limits<int>::max());
      const float* logits = a.logits.data + r * a.vocab;
      for (size_t w = 0; w < words; ++w) {
        uint32_t bits = a.mask.data[w];
        while (bits) {
          const int i = int(w * 32 + __builtin_ctz(bits));
          bits &= bits - 1;
          const float value = logits[i];
          if (!std::isfinite(value) || value < topv[GRAMMAR_HOST_TOPK - 1] ||
              (value == topv[GRAMMAR_HOST_TOPK - 1] && i >= topi[GRAMMAR_HOST_TOPK - 1])) continue;
          for (int k = 0; k < GRAMMAR_HOST_TOPK; ++k)
            if (value > topv[k] || (value == topv[k] && i < topi[k])) {
              for (int j = GRAMMAR_HOST_TOPK - 1; j > k; --j) {
                topv[j] = topv[j - 1]; topi[j] = topi[j - 1];
              }
              topv[k] = value; topi[k] = i;
              break;
            }
        }
      }
      if (topi[0] == std::numeric_limits<int>::max())
        return fail(GrammarHostError::NoFiniteCandidate, completed, b, p);
      float pr[GRAMMAR_HOST_RANK];
      for (int t = 0; t < GRAMMAR_HOST_RANK; ++t) {
        pr[t] = mul(bf16(a.pred.data[size_t(previous) * a.rank + t]),
                    a.hproj.data[r * a.rank + t]);
        if (!std::isfinite(pr[t]))
          return fail(GrammarHostError::NonfiniteTransition, completed, b, p);
      }
      float best = -std::numeric_limits<float>::infinity(), second = best;
      int winner = 0;
      for (int k = 0; k < GRAMMAR_HOST_TOPK; ++k) {
        if (topi[k] == std::numeric_limits<int>::max()) break;
        float score = topv[k];
        for (int warp = 0; warp < 8; ++warp) {
          float lanes[32];
          for (int lane = 0; lane < 32; ++lane) {
            const int t = warp * 32 + lane;
            lanes[lane] = mul(pr[t], bf16(a.succ.data[size_t(topi[k]) * a.rank + t]));
          }
          score = add(score, warp_sum(lanes));
        }
        if (!std::isfinite(score))
          return fail(GrammarHostError::NonfiniteTransition, completed, b, p);
        if (score > best) { second = best; best = score; winner = k; }
        else if (score > second) second = score;
      }
      previous = topi[winner];
      a.tokens.data[r] = previous;
      a.confidence.data[r] = sub(best, second);
      ++completed;
    }
  }
  return {GrammarHostError::None, completed, -1, -1, 0};
}
}
