#pragma once
#include <algorithm>
#include <cmath>
#include <climits>
#include <string>
#include <vector>

namespace h5::serving {
constexpr int QUANTUM = 128;
enum class Error { None, InvalidFlag, EmptyPrefix, MissingPrefix, NoRequestSlots, DiagnosticSlots, InvalidAllocation };
inline const char* error(Error e) {
  switch (e) {
    case Error::None: return "none";
    case Error::InvalidFlag: return "serving flags require exactly 0 or 1 (unset means disabled)";
    case Error::EmptyPrefix: return "H5_SERVING_PREFIX must name a nonempty file path";
    case Error::MissingPrefix: return "H5_SERVING_CHECK=1 requires H5_SERVING_PREFIX";
    case Error::NoRequestSlots: return "reserved prefix template requires at least two physical --slots";
    case Error::DiagnosticSlots: return "prefix diagnostic requires at least two request slots";
    case Error::InvalidAllocation: return "invalid prefill allocation state";
  }
  return "invalid serving error";
}
inline Error flag(const char* value, bool& out) {
  out = false;
  if (!value || std::string(value) == "0") return Error::None;
  if (std::string(value) != "1") return Error::InvalidFlag;
  out = true; return Error::None;
}
struct Config {
  bool fair = false, reserve = false, check = false;
  std::string prefix_file;
  int request_slots = 0, physical_slots = 0, template_slot = -1;
};
inline Error configure(const char* fair, const char* reserve, const char* prefix,
                       const char* check, int slots, bool tree, Config& c) {
  Config candidate;
  const std::pair<const char*, bool*> flags[] = {{fair, &candidate.fair},
    {reserve, &candidate.reserve}, {check, &candidate.check}};
  for (auto pair : flags) {
    auto e = flag(pair.first, *pair.second); if (e != Error::None) return e;
  }
  if (prefix) {
    if (!*prefix) return Error::EmptyPrefix;
    candidate.prefix_file = prefix; candidate.reserve = true;
  }
  if (candidate.check && !prefix) return Error::MissingPrefix;
  if (slots < 1 || (tree && slots == INT_MAX)) return Error::NoRequestSlots;
  candidate.request_slots = slots - (candidate.reserve ? 1 : 0);
  if (candidate.request_slots < 1) return Error::NoRequestSlots;
  if (candidate.check && candidate.request_slots < 2) return Error::DiagnosticSlots;
  candidate.physical_slots = slots + (tree ? 1 : 0);
  if (candidate.reserve) candidate.template_slot = candidate.request_slots + (tree ? 1 : 0);
  c = candidate; return Error::None;
}
struct Prefill { int slot, remaining; double age; };
struct Allocation { Error error; std::vector<int> rows; };
inline Allocation allocate(std::vector<Prefill> items, int slots, int budget, int cursor) {
  if (slots < 1 || budget < 1 || cursor < 0 || cursor >= slots) return {Error::InvalidAllocation, {}};
  std::vector<int> rows(slots, 0);
  for (const auto& p : items) {
    if (p.slot < 0 || p.slot >= slots || p.remaining < 1 || !std::isfinite(p.age) || p.age < 0 || rows[p.slot])
      return {Error::InvalidAllocation, {}};
    rows[p.slot] = 1;
  }
  std::fill(rows.begin(), rows.end(), 0);
  std::stable_sort(items.begin(), items.end(), [=](const Prefill& a, const Prefill& b) {
    bool aa = a.age >= 1., ba = b.age >= 1.;
    if (aa != ba) return aa;
    if (!aa && a.remaining != b.remaining) return a.remaining < b.remaining;
    return (a.slot + slots - cursor) % slots < (b.slot + slots - cursor) % slots;
  });
  for (const auto& p : items) {
    int n = std::min({p.remaining, QUANTUM, budget});
    rows[p.slot] = n; budget -= n;
  }
  return {Error::None, std::move(rows)};
}
inline bool prefix_match(const std::vector<int>& prefix, const std::vector<int>& prompt, bool image) {
  return !image && !prefix.empty() && prompt.size() > prefix.size() && std::equal(prefix.begin(), prefix.end(), prompt.begin());
}
enum class CopyError : int { None = 0, InvalidRange = -1, PendingDonor = -2 };
inline CopyError copy_guard(int src, int dst, int npos, int slots, int ctx, int pending, int commit) {
  if (src < 0 || dst < 0 || src >= slots || dst >= slots || src == dst || npos < 1 || npos > ctx)
    return CopyError::InvalidRange;
  if (pending || commit) return CopyError::PendingDonor;
  return CopyError::None;
}
}
