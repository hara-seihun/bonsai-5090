#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>
namespace h5::gpu_grammar {
constexpr uint64_t TOKEN_CACHE_BUDGET = uint64_t(2) << 30;
constexpr uint32_t TOKEN_COMPILE_CHUNK = 16;
enum class TokenCacheError : int { None = 0, Budget = -10, Overflow = -11 };
struct TokenLayout {
  std::vector<uint32_t> offsets;
  uint64_t sparse_edges = 0, wide_states = 0, bytes = 0;
};
template<class Entries> inline auto token_cache_victim(Entries& entries) {
  auto victim = entries.end();
  for (auto it = entries.begin(); it != entries.end(); ++it)
    if (!(*it)->refs && (victim == entries.end() || (*it)->used < (*victim)->used)) victim = it;
  return victim;
}
inline TokenCacheError token_layout(const std::vector<uint32_t>& counts, uint32_t vocabulary,
    uint64_t source_bytes, TokenLayout* output) {
  if (!output || !vocabulary || counts.empty()) return TokenCacheError::Overflow;
  TokenLayout layout;
  layout.offsets.resize(counts.size());
  for (size_t s = 0; s < counts.size(); ++s) {
    if (counts[s] > vocabulary) return TokenCacheError::Overflow;
    const uint64_t offset = counts[s] <= 512 ? layout.sparse_edges : layout.wide_states;
    if (offset > std::numeric_limits<uint32_t>::max()) return TokenCacheError::Overflow;
    layout.offsets[s] = uint32_t(offset);
    if (counts[s] <= 512) layout.sparse_edges += counts[s];
    else ++layout.wide_states;
  }
  const uint64_t words = (uint64_t(vocabulary) + 31) / 32;
  layout.bytes = source_bytes + counts.size() * 9ull + layout.sparse_edges * 8 +
      layout.wide_states * (uint64_t(vocabulary) * 4 + words * 4 + 512 * 4);
  if (layout.sparse_edges > std::numeric_limits<uint32_t>::max() ||
      layout.bytes > std::numeric_limits<size_t>::max()) return TokenCacheError::Overflow;
  *output = std::move(layout);
  return TokenCacheError::None;
}
}
