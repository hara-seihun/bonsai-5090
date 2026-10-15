#include "../src/commit_policy.h"
#include <cassert>
#include <limits>
#include <cstdio>
#include <array>
using namespace h5::commit;

int main() {
  Config c;
  assert(configure({"scheduled", "", "", "", "", ""}, 7, c) == Error::MissingParameter);
  for (const char* seed : {"-1", "2147483648", "nan", "7junk"})
    assert(configure({"scheduled", "", "", "", "", seed}, 7, c) == Error::InvalidInteger);
  for (int B : {16, 32, 64})
    assert(configure({"scheduled", "", "", "", "", "20261007"}, B, c) == Error::InvalidSchedule);
  for (const char* mode : {"fixed", "periodic", "confidence", "feedback"})
    assert(configure({mode, "", "", "", "", "1"}, 7, c) == Error::UnusedParameter);
  assert(configure({"scheduled", "0", "", "", "", "1"}, 7, c) == Error::UnusedParameter);
  assert(configure({"scheduled", "", "4", "", "", "1"}, 7, c) == Error::UnusedParameter);
  assert(configure({"scheduled", "", "", "1", "", "1"}, 7, c) == Error::UnusedParameter);
  assert(configure({"scheduled", "", "", "", "2", "1"}, 7, c) == Error::UnusedParameter);

  for (const char* seed : {"0", "20261007", "2147483647"}) {
    assert(configure({"scheduled", "", "", "", "", seed}, 7, c) == Error::None);
    assert(c.minimum == 0 && c.maximum == 4 && c.schedule.size() == SCHEDULE_LENGTH);
    assert(c.schedule == make_schedule(c.schedule_seed));
    assert(c.schedule != make_schedule(c.schedule_seed == 0 ? 1 : 0));
    std::array<int, 5> histogram{};
    for (int x : c.schedule) { assert(x >= 0 && x <= 4); ++histogram[x]; }
    assert(histogram == SCHEDULE_COUNTS);
    assert(schedule_hex(c.schedule).size() == 2 * SCHEDULE_LENGTH);
    int prompt[] = {2, 3};
    int offset = prompt_schedule_offset(c, prompt, 2);
    assert(offset >= 0 && offset < SCHEDULE_LENGTH);
    assert(prompt_schedule_offset(c, prompt, 2) == offset);
    State s = initial(c, offset);
    float poison[7]; for (float& x : poison) x = std::numeric_limits<float>::quiet_NaN();
    std::array<int, 5> used{};
    for (int round = 0; round < 2 * SCHEDULE_LENGTH; ++round) {
      auto a = select(c, s, poison, 7), b = select(c, s, nullptr, 7);
      assert(a.error == Error::None && a.trigger == Trigger::FrozenSchedule);
      assert(a.selected == b.selected && a.selected == c.schedule[(offset + round) % SCHEDULE_LENGTH]);
      assert(a.low_position == -1 && a.limit == 4);
      ++used[a.selected];
      State wrong = s;
      assert(observe(c, wrong, (a.selected + 1) % 5, 0, false).error == Error::InvalidRound);
      assert(wrong.schedule_round == s.schedule_round);
      const bool matched = a.selected > 0 && round % 3 == 0;
      const int accepted = matched ? a.selected : round % (a.selected + 1);
      auto result = observe(c, s, a.selected, accepted, matched);
      assert(result.error == Error::None && s.schedule_round == static_cast<uint64_t>(round + 1));
      assert(s.limit == 4 && s.schedule_offset == offset);
    }
    for (int x = 0; x < 5; ++x) assert(used[x] == 2 * SCHEDULE_COUNTS[x]);
    s.schedule_offset = -1;
    assert(select(c, s, nullptr, 7).error == Error::InvalidRound);
    s = initial(c, SCHEDULE_LENGTH);
    assert(select(c, s, nullptr, 7).error == Error::InvalidRound);
    s = initial(c, 0); s.schedule_round = UINT64_MAX;
    assert(select(c, s, nullptr, 7).error == Error::InvalidRound);
  }
  assert(configure({"scheduled", "", "", "", "", "20261007"}, 7, c) == Error::None);
  int prompt[] = {2, 3};
  assert(prompt_schedule_offset(c, prompt, 2) == 2167);
  assert(schedule_hex(c.schedule).substr(0, 64) == "0400020200040404040103040404040404040300040004010404040404040404");
  int other_prompt[] = {20, 42, 0};
  for (const auto& witness : std::array<std::pair<const char*, int>, 3>{{{"0", 2494}, {"17", 1445}, {"2147483647", 2558}}}) {
    assert(configure({"scheduled", "", "", "", "", witness.first}, 7, c) == Error::None);
    assert(prompt_schedule_offset(c, other_prompt, 3) == witness.second);
  }
  assert(configure({"scheduled", "", "", "", "", "20261007"}, 7, c) == Error::None);
  uint64_t zero = 0;
  assert(schedule_random(zero) == UINT64_C(0xe220a8397b1dcdaf));
  std::printf("schedule seed20261007 prompt2,3 offset=%d prefix_hex=%s\n", prompt_schedule_offset(c, prompt, 2), schedule_hex(c.schedule).substr(0, 64).c_str());
  std::puts("scheduled contracts pass: explicit parameters, frozen histogram, deterministic seed/prompt offset, complete cycles, poisoned-gap independence, gate/EOS consumption, invalid states");
}
