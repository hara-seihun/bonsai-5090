#include "grammar_host_select.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace h5;
namespace {
constexpr int V = 64, R = 256, N = 6;
struct Masks {
  std::vector<std::vector<int>> legal;
  std::vector<int> previous;
  int calls = 0, fail_at = -1, invalid_at = -1, throw_at = -1;
};
int callback(void* context, int block, int position, int previous, uint32_t* mask, int words) {
  auto& state = *static_cast<Masks*>(context);
  const int r = block * 3 + position;
  assert(r == state.calls++ && words == 2 && previous == state.previous[r]);
  if (r == state.throw_at) throw std::runtime_error("fixture callback boundary");
  if (r == state.fail_at) return -23;
  for (int token : state.legal[r]) mask[token / 32] |= uint32_t(1) << (token % 32);
  return int(state.legal[r].size()) + (r == state.invalid_at);
}
struct Fixture {
  std::vector<float> logits = std::vector<float>(N * V, 0.f);
  std::vector<float> hp = std::vector<float>(N * R, 0.f);
  std::vector<uint16_t> pred = std::vector<uint16_t>(V * R, 0);
  std::vector<uint16_t> succ = std::vector<uint16_t>(V * R, 0);
  int anchors[2] = {32, 33}, tokens[N] = {};
  float conf[N] = {};
  uint32_t mask[2] = {};
  Masks state;
  Fixture() {
    state.legal = {{}, {20}, {1, 2}, {1, 2}, {3, 4}, {3, 4}};
    for (int i = 0; i < 20; ++i) state.legal[0].push_back(i);
    state.previous = {-1, 0, 20, -1, 2, 4};
    for (int t = 0; t < R; ++t) {
      pred[32 * R + t] = pred[20 * R + t] = pred[2 * R + t] = 0x3f80; // 1
      pred[4 * R + t] = 0xbf80; // -1
      succ[1 * R + t] = 0x3f80;
      succ[19 * R + t] = 0x4300; // 128: must not resurrect the excluded 17th+ logit.
    }
    // Candidate 19 would win via the selector, but is outside the top-16 head cutoff.
    hp[1] = 1.f;
    succ[1 * R + 1] = 0;
    // Singleton rows never evaluate their invalid head or hproj.
    logits[V + 20] = hp[R] = std::numeric_limits<float>::quiet_NaN();
    hp[2 * R] = 1e20f; hp[2 * R + 16] = -1e20f;
    hp[2 * R + 8] = hp[2 * R + 24] = 1.f;
    logits[3 * V + 1] = 3.f; logits[3 * V + 2] = 5.f;
    hp[4 * R] = hp[5 * R] = 1.f;
    succ[3 * R] = 0x3f80; succ[4 * R] = 0x4000; // 1, 2
  }
  GrammarHostSelect args() {
    return {2, 3, V, R, {logits.data(), logits.size()}, {hp.data(), hp.size()},
            {pred.data(), pred.size()}, {succ.data(), succ.size()}, {anchors, 2},
            {tokens, N}, {conf, N}, {mask, 2}, callback, &state};
  }
};
}
int main() {
  {
    Fixture f;
    const auto result = grammar_host_select(f.args());
    assert(result.ok() && result.completed_rows == N && f.state.calls == N);
    const int expected[] = {0, 20, 1, 2, 4, 3};
    const float margins[] = {0, FLT_MAX, 2, 2, 1, 1};
    for (int r = 0; r < N; ++r) assert(f.tokens[r] == expected[r] && f.conf[r] == margins[r]);
  }
  {
    Fixture f;
    // Transition ties retain the descending-head order, not ascending token id.
    f.logits[0] = 1.f; f.logits[1] = 2.f;
    f.hp[0] = -1.f; // candidate 1 gets -1, so both final scores are 1.
    f.state.previous[1] = 1;
    assert(grammar_host_select(f.args()).ok() && f.tokens[0] == 1 && f.conf[0] == 0.f);
  }
  {
    Fixture f;
    // Add the eight warp partials to the head logit in sequence, not to each other first.
    f.state.legal[2] = {1, 2};
    f.logits[2 * V + 1] = 1e20f;
    std::fill(f.hp.begin() + 2 * R, f.hp.begin() + 3 * R, 0.f);
    f.hp[2 * R] = -1e20f; f.hp[2 * R + 32] = 1.f;
    assert(grammar_host_select(f.args()).ok() && f.tokens[2] == 1 && f.conf[2] == 1.f);
  }
  {
    Fixture f; auto a = f.args(); --a.hproj.count;
    assert(grammar_host_select(a).error == GrammarHostError::BufferTooSmall && f.state.calls == 0);
  }
  {
    Fixture f; f.anchors[1] = V;
    assert(grammar_host_select(f.args()).error == GrammarHostError::InvalidAnchor && f.state.calls == 0);
  }
  {
    Fixture f; f.state.fail_at = 1;
    const auto r = grammar_host_select(f.args());
    assert(r.error == GrammarHostError::CallbackFailure && r.completed_rows == 1 && r.callback_status == -23);
  }
  {
    Fixture f; f.state.invalid_at = 1;
    assert(grammar_host_select(f.args()).error == GrammarHostError::InvalidMask);
  }
  {
    Fixture f; f.state.throw_at = 1;
    assert(grammar_host_select(f.args()).error == GrammarHostError::CallbackException);
  }
  {
    Fixture f; f.state.legal[0].clear();
    assert(grammar_host_select(f.args()).error == GrammarHostError::EmptyMask);
  }
  {
    Fixture f;
    std::fill(f.logits.begin(), f.logits.begin() + V, std::numeric_limits<float>::infinity());
    assert(grammar_host_select(f.args()).error == GrammarHostError::NoFiniteCandidate);
  }
  {
    Fixture f; f.hp[0] = std::numeric_limits<float>::quiet_NaN();
    assert(grammar_host_select(f.args()).error == GrammarHostError::NonfiniteTransition);
  }
  {
    Fixture f;
    f.state.legal[0] = {0, 1};
    f.logits[1] = std::numeric_limits<float>::infinity();
    assert(grammar_host_select(f.args()).ok() && f.tokens[0] == 0 && std::isinf(f.conf[0]));
  }
  {
    Fixture f;
    const int original = std::fegetround();
    assert(std::fesetround(FE_DOWNWARD) == 0);
    const auto r = grammar_host_select(f.args());
    assert(std::fesetround(original) == 0);
    assert(r.error == GrammarHostError::UnsupportedArithmetic && f.state.calls == 0);
  }
#if defined(__SSE__)
  {
    Fixture f;
    const auto original = _mm_getcsr();
    _mm_setcsr(original | 0x8040u);
    const auto r = grammar_host_select(f.args());
    _mm_setcsr(original);
    assert(r.error == GrammarHostError::UnsupportedArithmetic && f.state.calls == 0);
  }
#endif
  std::puts("grammar host selector: exact cutoff/ties/tree/prefix/singletons and typed errors passed");
}
