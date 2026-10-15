#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cfloat>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstdio>

namespace h5::commit {
inline constexpr std::array<int, 5> SCHEDULE_COUNTS{394, 338, 232, 239, 1640};
inline constexpr int SCHEDULE_LENGTH = 2843;
inline constexpr const char* SCHEDULE_ID = "native7-g1-cap4-histogram-v1";
inline constexpr const char* SCHEDULE_ALGORITHM = "splitmix64-fisher-yates-rejection-v1";
inline constexpr const char* SCHEDULE_OFFSET = "fnv1a64-token-le32-xor-seed-splitmix64-mod-v1";
inline constexpr const char* SCHEDULE_CONSUMPTION = "prompt-rotated-cyclic-one-per-verifier-round";
inline uint64_t schedule_random(uint64_t& state) {
  uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
inline uint64_t schedule_bounded(uint64_t& state, uint64_t bound) {
  const uint64_t threshold = (uint64_t(0) - bound) % bound;
  uint64_t value;
  do { value = schedule_random(state); } while (value < threshold);
  return value % bound;
}
inline std::vector<int> make_schedule(int seed) {
  std::vector<int> values; values.reserve(SCHEDULE_LENGTH);
  for (int k = 0; k < 5; ++k) values.insert(values.end(), SCHEDULE_COUNTS[k], k);
  uint64_t state = static_cast<uint64_t>(seed);
  for (int i = SCHEDULE_LENGTH - 1; i > 0; --i)
    std::swap(values[i], values[schedule_bounded(state, static_cast<uint64_t>(i + 1))]);
  return values;
}
inline std::string schedule_hex(const std::vector<int>& values) {
  std::string result; result.reserve(values.size() * 2);
  for (int v : values) { result.push_back('0'); result.push_back(static_cast<char>('0' + v)); }
  return result;
}
enum class Mode { Fixed, Periodic, Confidence, Feedback, Scheduled, Expected, Nested };
enum class Error { None, InvalidMode, MissingParameter, UnusedParameter, InvalidInteger, InvalidFloat, InvalidRange, InvalidConfidence, InvalidRound, InvalidSchedule, InvalidCalibration, InvalidNested };
enum class Trigger { FullBlock, Periodic, LowConfidence, PrefixLimit, FeedbackLimit, FrozenSchedule, ExpectedRate, NestedShort, NestedLong };
struct Config {
  Mode mode = Mode::Fixed;
  int minimum = 0, maximum = 0, growth = 0;
  float threshold = 0.f;
  int schedule_seed = -1;
  std::vector<int> schedule;
  // Expected mode: p(accept position i | earlier accepted) = sigmoid(a + b * gap_i + c * i).
  float cal_a = 0.f, cal_b = 0.f, cal_c = 0.f;
};
// RTX 5090, selected preset, short context: verifier step ms by row count (h5 specrows, 2026-10-07)
// and the drafter round. Index = verifier rows (anchor + proposals).
inline constexpr std::array<float, 17> VERIFY_MS{0.f, 4.755f, 5.015f, 5.20f, 5.383f, 5.47f, 5.560f, 5.62f, 5.673f,
                                                 6.00f, 6.04f, 6.09f, 6.145f, 6.20f, 6.26f, 6.32f, 6.373f};
inline constexpr float DRAFT_MS = 1.11f;
inline float round_ms(int proposals) { return VERIFY_MS[std::min(proposals + 1, 16)] + DRAFT_MS; }
struct Options {
  std::string mode = "fixed", minimum, maximum, threshold, growth, schedule_seed, calibration;
};
struct State { int limit; uint64_t schedule_round = 0; int schedule_offset = 0; double rate = 0.6; };
struct Selection {
  Error error;
  int selected, limit, low_position;
  float low_gap;
  Trigger trigger;
};
struct Outcome { Error error; bool gate_rejection; int next_limit; };
inline const char* name(Mode m) {
  switch (m) {
    case Mode::Fixed: return "fixed";
    case Mode::Periodic: return "periodic";
    case Mode::Confidence: return "confidence";
    case Mode::Feedback: return "feedback";
    case Mode::Scheduled: return "scheduled";
    case Mode::Expected: return "expected";
    case Mode::Nested: return "nested";
  }
  std::abort();
}
inline const char* name(Trigger t) {
  switch (t) {
    case Trigger::FullBlock: return "full-block";
    case Trigger::Periodic: return "periodic";
    case Trigger::LowConfidence: return "low-confidence";
    case Trigger::PrefixLimit: return "prefix-limit";
    case Trigger::FeedbackLimit: return "feedback-limit";
    case Trigger::FrozenSchedule: return "frozen-schedule";
    case Trigger::ExpectedRate: return "expected-rate";
    case Trigger::NestedShort: return "nested-short";
    case Trigger::NestedLong: return "nested-long";
  }
  std::abort();
}
inline const char* error(Error e) {
  switch (e) {
    case Error::None: return "ok";
    case Error::InvalidMode: return "unknown commitment policy";
    case Error::MissingParameter: return "required commitment parameter unset";
    case Error::UnusedParameter: return "commitment parameter not used by selected policy";
    case Error::InvalidInteger: return "commitment integer must be finite, nonnegative and within int range";
    case Error::InvalidFloat: return "commitment confidence threshold must be finite and nonnegative";
    case Error::InvalidRange: return "require 0 <= minimum <= maximum <= proposals; feedback growth > 0";
    case Error::InvalidConfidence: return "observed selector gap must be finite and nonnegative";
    case Error::InvalidRound: return "invalid commitment round accounting";
    case Error::InvalidSchedule: return "scheduled policy requires native7 proposals and the frozen histogram";
    case Error::InvalidCalibration: return "expected/nested policy requires --commit-calibration A,B,C (finite floats)";
    case Error::InvalidNested: return "nested policy requires 1 <= --commit-min < --commit-max == proposals and --commit-confidence in [0,1]";
  }
  std::abort();
}
inline Error integer(const std::string& s, int& v) {
  if (s.empty()) return Error::MissingParameter;
  char* end = nullptr; errno = 0;
  long n = std::strtol(s.c_str(), &end, 10);
  if (errno || end == s.c_str() || *end || n < 0 || n > INT_MAX) return Error::InvalidInteger;
  v = static_cast<int>(n); return Error::None;
}
inline Error configure(const Options& o, int proposals, Config& c) {
  if (o.mode == "fixed") c.mode = Mode::Fixed;
  else if (o.mode == "periodic") c.mode = Mode::Periodic;
  else if (o.mode == "confidence") c.mode = Mode::Confidence;
  else if (o.mode == "feedback") c.mode = Mode::Feedback;
  else if (o.mode == "scheduled") c.mode = Mode::Scheduled;
  else if (o.mode == "expected") c.mode = Mode::Expected;
  else if (o.mode == "nested") c.mode = Mode::Nested;
  else return Error::InvalidMode;
  if (c.mode == Mode::Nested) {
    // Nested block: verify the short prefix (minimum) unless the calibrated probability that all of it is
    // accepted exceeds threshold; then verify the whole block (maximum == proposals). threshold 1 never extends.
    if (!o.growth.empty() || !o.schedule_seed.empty()) return Error::UnusedParameter;
    Error e = integer(o.minimum, c.minimum); if (e != Error::None) return e;
    e = integer(o.maximum, c.maximum); if (e != Error::None) return e;
    char tail = 0;
    if (o.calibration.empty() || std::sscanf(o.calibration.c_str(), "%f,%f,%f%c", &c.cal_a, &c.cal_b, &c.cal_c, &tail) != 3 ||
        !std::isfinite(c.cal_a) || !std::isfinite(c.cal_b) || !std::isfinite(c.cal_c)) return Error::InvalidCalibration;
    if (o.threshold.empty()) return Error::MissingParameter;
    char* end = nullptr; errno = 0; c.threshold = std::strtof(o.threshold.c_str(), &end);
    if (errno || end == o.threshold.c_str() || *end || !std::isfinite(c.threshold) || c.threshold < 0.f || c.threshold > 1.f ||
        c.minimum < 1 || c.minimum >= c.maximum || c.maximum != proposals) return Error::InvalidNested;
    c.schedule.clear(); c.schedule_seed = -1; c.growth = 0;
    return Error::None;
  }
  if (c.mode == Mode::Expected) {
    if (!o.threshold.empty() || !o.growth.empty() || !o.schedule_seed.empty()) return Error::UnusedParameter;
    Error e = integer(o.minimum, c.minimum); if (e != Error::None) return e;
    e = integer(o.maximum, c.maximum); if (e != Error::None) return e;
    char tail = 0;
    if (o.calibration.empty() || std::sscanf(o.calibration.c_str(), "%f,%f,%f%c", &c.cal_a, &c.cal_b, &c.cal_c, &tail) != 3 ||
        !std::isfinite(c.cal_a) || !std::isfinite(c.cal_b) || !std::isfinite(c.cal_c)) return Error::InvalidCalibration;
    if (c.minimum > c.maximum || c.maximum > proposals || c.maximum > 15) return Error::InvalidRange;
    c.schedule.clear(); c.schedule_seed = -1; c.growth = 0; c.threshold = 0.f;
    return Error::None;
  }
  if (!o.calibration.empty()) return Error::UnusedParameter;
  if (c.mode == Mode::Scheduled) {
    if (!o.minimum.empty() || !o.maximum.empty() || !o.threshold.empty() || !o.growth.empty()) return Error::UnusedParameter;
    Error e = integer(o.schedule_seed, c.schedule_seed); if (e != Error::None) return e;
    if (proposals != 7) return Error::InvalidSchedule;
    c.minimum = 0; c.maximum = 4; c.growth = 0; c.threshold = 0.f;
    c.schedule = make_schedule(c.schedule_seed); return Error::None;
  }
  if (!o.schedule_seed.empty()) return Error::UnusedParameter;
  c.schedule.clear(); c.schedule_seed = -1;
  if (c.mode == Mode::Fixed) {
    if (!o.minimum.empty() || !o.maximum.empty() || !o.threshold.empty() || !o.growth.empty()) return Error::UnusedParameter;
    c.minimum = c.maximum = proposals; c.growth = 0; c.threshold = 0.f; return Error::None;
  }
  Error e = integer(o.maximum, c.maximum); if (e != Error::None) return e;
  if (c.mode == Mode::Periodic) {
    if (!o.minimum.empty() || !o.threshold.empty() || !o.growth.empty()) return Error::UnusedParameter;
    c.minimum = c.maximum; c.growth = 0; c.threshold = 0.f;
  } else {
    e = integer(o.minimum, c.minimum); if (e != Error::None) return e;
    if (o.threshold.empty()) return Error::MissingParameter;
    char* end = nullptr; errno = 0; c.threshold = std::strtof(o.threshold.c_str(), &end);
    if (errno || end == o.threshold.c_str() || *end || !std::isfinite(c.threshold) || c.threshold < 0.f) return Error::InvalidFloat;
    if (c.mode == Mode::Feedback) {
      e = integer(o.growth, c.growth); if (e != Error::None) return e;
    } else if (!o.growth.empty()) return Error::UnusedParameter;
  }
  if (c.minimum > c.maximum || c.maximum > proposals || (c.mode == Mode::Feedback && c.growth == 0)) return Error::InvalidRange;
  return Error::None;
}
inline int prompt_schedule_offset(const Config& c, const int* tokens, size_t count) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < count; ++i) {
    uint32_t token = static_cast<uint32_t>(tokens[i]);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (token >> (byte * 8)) & 255u; hash *= UINT64_C(1099511628211);
    }
  }
  uint64_t state = hash ^ static_cast<uint64_t>(c.schedule_seed);
  return static_cast<int>(schedule_random(state) % SCHEDULE_LENGTH);
}
inline State initial(const Config& c, int offset = 0) { return {c.maximum, 0, offset}; }
inline Selection select(const Config& c, const State& s, const float* gaps, int proposals) {
  if (c.mode == Mode::Scheduled) {
    if (proposals != 7 || c.schedule.size() != SCHEDULE_LENGTH || s.schedule_offset < 0 || s.schedule_offset >= SCHEDULE_LENGTH || s.schedule_round > UINT64_MAX - SCHEDULE_LENGTH)
      return {Error::InvalidRound, 0, 4, -1, 0.f, Trigger::FrozenSchedule};
    const int selected = c.schedule[(s.schedule_round + s.schedule_offset) % SCHEDULE_LENGTH];
    if (selected < 0 || selected > 4) return {Error::InvalidRound, 0, 4, -1, 0.f, Trigger::FrozenSchedule};
    return {Error::None, selected, 4, -1, 0.f, Trigger::FrozenSchedule};
  }
  if (c.mode == Mode::Fixed) return {Error::None, proposals, proposals, -1, 0.f, Trigger::FullBlock};
  if (c.mode == Mode::Nested) {
    if (!gaps || proposals != c.maximum) return {Error::InvalidConfidence, 0, c.maximum, -1, 0.f, Trigger::NestedShort};
    double survive = 1.0;
    for (int k = 0; k < c.minimum; ++k) {
      const float g = gaps[k];
      if (!std::isfinite(g) || g < 0.f) return {Error::InvalidConfidence, 0, c.maximum, k, g, Trigger::NestedShort};
      survive *= 1.0 / (1.0 + std::exp(-(double)(c.cal_a + c.cal_b * g + c.cal_c * k)));
    }
    const bool extend = survive > (double)c.threshold;
    return {Error::None, extend ? c.maximum : c.minimum, c.maximum, -1, (float)survive, extend ? Trigger::NestedLong : Trigger::NestedShort};
  }
  if (c.mode == Mode::Expected) {
    // Maximise E[emitted tokens] - rate * round time over prefixes k in [minimum, maximum] (Dinkelbach step
    // with the running emitted-tokens-per-ms rate).
    if (!gaps) return {Error::InvalidConfidence, 0, c.maximum, -1, 0.f, Trigger::ExpectedRate};
    double survive = 1.0, expected = 1.0, best = -1e30; int bk = c.minimum;
    for (int k = 0; k <= c.maximum; ++k) {
      if (k > 0) {
        const float g = gaps[k - 1];
        if (!std::isfinite(g) || g < 0.f) return {Error::InvalidConfidence, 0, c.maximum, k - 1, g, Trigger::ExpectedRate};
        survive *= 1.0 / (1.0 + std::exp(-(double)(c.cal_a + c.cal_b * g + c.cal_c * (k - 1))));
        expected += survive;
      }
      const double value = expected - s.rate * round_ms(k);
      if (k >= c.minimum && value > best) { best = value; bk = k; }
    }
    return {Error::None, bk, c.maximum, bk < c.maximum ? bk : -1, bk < c.maximum ? gaps[bk] : 0.f, Trigger::ExpectedRate};
  }
  const int limit = c.mode == Mode::Feedback ? s.limit : c.maximum;
  if (limit < c.minimum || limit > c.maximum || limit > proposals) return {Error::InvalidRound, 0, limit, -1, 0.f, Trigger::PrefixLimit};
  Trigger trigger = c.mode == Mode::Periodic ? Trigger::Periodic :
    limit < c.maximum ? Trigger::FeedbackLimit : limit < proposals ? Trigger::PrefixLimit : Trigger::FullBlock;
  if (c.mode == Mode::Periodic) return {Error::None, limit, limit, -1, 0.f, trigger};
  if (!gaps && limit) return {Error::InvalidConfidence, 0, limit, -1, 0.f, trigger};
  for (int p = 0; p < limit; ++p) {
    if (!std::isfinite(gaps[p]) || gaps[p] < 0.f) return {Error::InvalidConfidence, 0, limit, p, gaps[p], trigger};
    if (p >= c.minimum && gaps[p] < c.threshold)
      return {Error::None, p, limit, p, gaps[p], Trigger::LowConfidence};
  }
  return {Error::None, limit, limit, -1, 0.f, trigger};
}
inline Outcome observe(const Config& c, State& s, int selected, int accepted, bool matched_eos) {
  if (selected < 0 || selected > c.maximum || accepted < 0 || accepted > selected || (matched_eos && accepted == 0))
    return {Error::InvalidRound, false, s.limit};
  const bool rejection = accepted < selected && !matched_eos;
  if (c.mode == Mode::Scheduled) {
    const auto selection = select(c, s, nullptr, 7);
    if (selection.error != Error::None || selection.selected != selected) return {Error::InvalidRound, false, s.limit};
    ++s.schedule_round;
  }
  if (c.mode == Mode::Expected) s.rate = 0.9 * s.rate + 0.1 * (accepted + 1) / round_ms(selected);
  if (c.mode == Mode::Feedback && !matched_eos) {
    // Gate-triggered shortening, followed by additive recovery after a full selected-prefix acceptance.
    // This controls the next target intervention, not the learned drafter query shape.
    s.limit = rejection ? std::max(c.minimum, std::min(selected / 2, accepted)) :
      std::min(c.maximum, s.limit + std::min(c.growth, c.maximum - s.limit));
  }
  return {Error::None, rejection, s.limit};
}
}
