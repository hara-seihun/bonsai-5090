#include "../src/commit_policy.h"
#include "../src/draft_contract.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>
using namespace h5::commit;

int main() {
  Config c;
  assert(configure({"fixed", "", "", "", ""}, 64, c) == Error::None);
  assert(select(c, initial(c), nullptr, 64).selected == 64);
  assert(configure({"unknown", "", "", "", ""}, 64, c) == Error::InvalidMode);
  assert(configure({"fixed", "0", "", "", ""}, 64, c) == Error::UnusedParameter);
  assert(configure({"periodic", "", "", "", ""}, 64, c) == Error::MissingParameter);
  assert(configure({"periodic", "", "65", "", ""}, 64, c) == Error::InvalidRange);
  assert(configure({"periodic", "", "nan", "", ""}, 64, c) == Error::InvalidInteger);
  assert(configure({"confidence", "0", "64", "nan", ""}, 64, c) == Error::InvalidFloat);
  assert(configure({"confidence", "0", "64", "inf", ""}, 64, c) == Error::InvalidFloat);
  assert(configure({"confidence", "8", "4", "1", ""}, 64, c) == Error::InvalidRange);
  assert(configure({"confidence", "0", "64", "1", "2"}, 64, c) == Error::UnusedParameter);
  assert(configure({"feedback", "0", "64", "1", "0"}, 64, c) == Error::InvalidRange);

  for (int B : {7, 16, 32, 64}) {
    std::vector<float> gaps(B, 2.f);
    assert(configure({"confidence", "0", std::to_string(B), "1", ""}, B, c) == Error::None);
    auto state = initial(c);
    assert(select(c, state, gaps.data(), B).selected == B);
    for (int k = 0; k < B; ++k) {
      gaps[k] = .5f;
      auto pick = select(c, state, gaps.data(), B);
      assert(pick.error == Error::None && pick.selected == k && pick.low_position == k);
      assert(pick.trigger == Trigger::LowConfidence);
      gaps[k] = 2.f;
    }
    gaps[0] = std::numeric_limits<float>::quiet_NaN();
    assert(select(c, state, gaps.data(), B).error == Error::InvalidConfidence);
    gaps[0] = -1.f;
    assert(select(c, state, gaps.data(), B).error == Error::InvalidConfidence);
    gaps[0] = 2.f;
    assert(select(c, state, nullptr, B).error == Error::InvalidConfidence);
    for (int selected = 0; selected <= B; ++selected) {
      assert(configure({"periodic", "", std::to_string(selected), "", ""}, B, c) == Error::None);
      state = initial(c);
      assert(select(c, state, nullptr, B).selected == selected);
      for (int accepted = 0; accepted <= selected; ++accepted) {
        auto outcome = observe(c, state, selected, accepted, false);
        assert(outcome.error == Error::None && outcome.gate_rejection == (accepted < selected));
        const int paid_rows = selected + 1, commit_inputs = accepted + 1;
        assert(commit_inputs > 0 && commit_inputs <= paid_rows);
        // Prefix+anchor input rows are the only states and draft-capture rows retained;
        // the final target output becomes the next input anchor, never a skipped suffix token.
        std::vector<int> speculative, committed;
        for (int j = 0; j < paid_rows; ++j) speculative.push_back(1000 + j);
        committed.assign(speculative.begin(), speculative.begin() + commit_inputs);
        assert(committed.size() == static_cast<size_t>(commit_inputs));
        assert(committed.back() == 1000 + accepted);
        int replayed = 0, consumed = 0;
        while (replayed < commit_inputs || consumed < paid_rows) {
          const auto chunk = h5::replay_chunk(commit_inputs, paid_rows, replayed, consumed, 16);
          assert(chunk.replay_rows + chunk.new_rows <= 16);
          assert(chunk.replay_rows + chunk.new_rows > 0);
          replayed += chunk.replay_rows; consumed += chunk.new_rows;
        }
        assert(replayed == commit_inputs && consumed == paid_rows);
      }
    }
  }

  assert(configure({"feedback", "1", "16", "1", "2"}, 16, c) == Error::None);
  auto state = initial(c);
  auto outcome = observe(c, state, 16, 3, false);
  assert(outcome.gate_rejection && state.limit == 3);
  outcome = observe(c, state, 3, 3, false);
  assert(!outcome.gate_rejection && state.limit == 5);
  outcome = observe(c, state, 5, 0, false);
  assert(outcome.gate_rejection && state.limit == 1);
  auto before = state.limit;
  outcome = observe(c, state, 1, 1, true);
  assert(!outcome.gate_rejection && state.limit == before);
  assert(observe(c, state, 1, 0, true).error == Error::InvalidRound);
  assert(observe(c, state, 1, 2, false).error == Error::InvalidRound);

  assert(configure({"confidence", "2", "16", "1", ""}, 16, c) == Error::None);
  std::vector<float> gaps(16, .5f);
  auto pick = select(c, initial(c), gaps.data(), 16);
  assert(pick.selected == 2); // Deliberate minimum, not an invented confidence fallback.
  puts("commit policy contracts pass: explicit config/errors, all lengths/prefixes, EOS, feedback, bounded replay");
}
