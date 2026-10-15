#pragma once
#include <climits>
#include <cfloat>
#include <cstdlib>

#ifdef __CUDACC__
#define H5_HD __host__ __device__
#else
#define H5_HD
#endif

namespace h5 {
constexpr int DRAFT_BASE_PROPOSALS = 7;
constexpr int DRAFT_MAX_PROPOSALS = 64;
enum class DraftError { None, UnsupportedLength, SingleStreamRequired, TreeLengthUnsupported, Capacity };
struct DraftConfig {
  DraftError error;
  int proposals, query_rows, verifier_rows, pending_rows, engine_rows;
};
inline DraftConfig draft_config(int proposals, int slots, bool tree, int ctx, int prefill) {
  if (proposals != 7 && proposals != 16 && proposals != 32 && proposals != 64)
    return {DraftError::UnsupportedLength, 0, 0, 0, 0, 0};
  if (tree && proposals != 7) return {DraftError::TreeLengthUnsupported, 0, 0, 0, 0, 0};
  if (proposals != 7 && slots != 1) return {DraftError::SingleStreamRequired, 0, 0, 0, 0, 0};
  const int rows = proposals + 1;
  if (slots <= 0 || prefill <= 0 || ctx <= rows + 1 ||
      (long long)slots * rows + prefill + (tree ? 8 : 0) > INT_MAX)
    return {DraftError::Capacity, 0, 0, 0, 0, 0};
  return {DraftError::None, proposals, rows, rows, rows, slots * rows + prefill + (tree ? 8 : 0)};
}
inline const char* draft_error(DraftError e) {
  switch (e) {
    case DraftError::None: return "ok";
    case DraftError::UnsupportedLength: return "unsupported proposal count: use 7, 16, 32 or 64";
    case DraftError::SingleStreamRequired: return "experimental long draft requires --slots 1";
    case DraftError::TreeLengthUnsupported: return "H5_TREE2 supports only 7 proposals";
    case DraftError::Capacity: return "invalid draft/context/prefill capacity";
  }
  return "invalid DraftError";
}

inline int parse_proposals(const char* value) {
  char* end = nullptr;
  const long n = std::strtol(value, &end, 10);
  return end == value || *end || n < 0 || n > INT_MAX ? -1 : (int)n;
}
inline bool parse_draft_margin(const char* value, float& margin) {
  char* end = nullptr;
  margin = std::strtof(value, &end);
  return end != value && !*end && margin >= 0.f && margin <= FLT_MAX;
}
H5_HD inline bool draft_eos(int token) { return token == 248044 || token == 248046; }
H5_HD inline bool relaxed_draft_allowed(int target, int proposed, float best, float score, float margin, int vocab) {
  return proposed >= 0 && proposed < vocab && !draft_eos(target) && !draft_eos(proposed) &&
    best <= FLT_MAX && best >= -FLT_MAX && score <= FLT_MAX && score >= -FLT_MAX && best - score <= margin;
}
inline int draft_accepted_prefix(const int* targets, const int* proposals, int count) {
  int matched = 0;
  while (matched < count && targets[matched] == proposals[matched]) {
    const int token = proposals[matched++];
    if (draft_eos(token)) break;
  }
  return matched;
}

struct ReplayChunk { int replay_offset, replay_rows, new_offset, new_rows; };
H5_HD inline ReplayChunk replay_chunk(int committed, int rows, int replayed, int done, int tile) {
  const int remaining = committed - replayed;
  const int nr = remaining < tile ? remaining : tile;
  const int available = tile - nr;
  const int nn = rows - done < available ? rows - done : available;
  return {replayed, nr, done, nn};
}
H5_HD inline int draft_key_end(int query_pos, int query_row, int block_rows) {
  return query_pos - query_row % block_rows + block_rows;
}
}
#undef H5_HD
