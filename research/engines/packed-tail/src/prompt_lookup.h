#pragma once
// Prompt lookup for the agreement extension (SPECULATION_IDEAS.md idea 2).
//
// Index holds one request's tokens (prompt, then every emitted token). continuation(K) finds the longest
// suffix of the sequence, among the lengths in LENGTHS, that occurred earlier with at least one following
// token, takes its most recent such occurrence, and returns up to K tokens that followed it (a view into the
// sequence, valid until the next append). No hit has length 0.
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace h5::lookup {
inline constexpr int LENGTHS[] = {24, 16, 12, 8, 6, 4, 3, 2};   // descending; longest match wins
inline constexpr int NLENGTHS = sizeof(LENGTHS) / sizeof(LENGTHS[0]);

struct Hit { const int* tokens; int length; };

class Index {
 public:
  // H5_LOOKUP_SCOPE=prompt: freeze() after the prompt, so only prompt n-grams are indexed (the model's own output
  // can then never be proposed back to it). H5_LOOKUP_MIN=N: ignore suffix matches shorter than N tokens.
  void configure(bool prompt_only, int min_length) { prompt_only_ = prompt_only; min_length_ = min_length; }
  void freeze() { frozen_ = prompt_only_; }
  void append(const int* tokens, size_t count) {
    seq_.insert(seq_.end(), tokens, tokens + count);
    if (frozen_) return;
    // An n-gram ending (exclusive) at e is indexable once seq_[e] exists.
    for (; indexed_ < seq_.size(); ++indexed_) {
      const size_t e = indexed_;   // continuation token seq_[e] exists
      for (int li = 0; li < NLENGTHS; ++li) {
        const int n = LENGTHS[li];
        if (e < (size_t)n) continue;
        maps_[li][hash(e - n, n)] = (int)e;
      }
    }
  }
  Hit continuation(int max_tokens) const {
    const size_t len = seq_.size();
    for (int li = 0; li < NLENGTHS; ++li) {
      const int n = LENGTHS[li];
      if (n < min_length_) break;
      if (len < (size_t)n + 1) continue;
      const auto it = maps_[li].find(hash(len - n, n));
      if (it == maps_[li].end()) continue;
      const size_t e = it->second;
      bool same = true;
      for (int k = 0; k < n && same; ++k) same = seq_[e - n + k] == seq_[len - n + k];
      if (!same) continue;   // hash collision: try the next shorter length
      const size_t avail = len - e;
      return {seq_.data() + e, (int)(avail < (size_t)max_tokens ? avail : (size_t)max_tokens)};
    }
    return {nullptr, 0};
  }
  size_t size() const { return seq_.size(); }

 private:
  uint64_t hash(size_t begin, int n) const {
    uint64_t h = UINT64_C(1469598103934665603) ^ (uint64_t)n;
    for (int k = 0; k < n; ++k) { h ^= (uint32_t)seq_[begin + k]; h *= UINT64_C(1099511628211); h ^= h >> 29; }
    return h;
  }
  std::vector<int> seq_;
  size_t indexed_ = 0;
  bool prompt_only_ = false, frozen_ = false;
  int min_length_ = 0;   // n-grams ending (exclusive) before indexed_ are in the maps
  std::unordered_map<uint64_t, int> maps_[NLENGTHS];
};
}  // namespace h5::lookup
