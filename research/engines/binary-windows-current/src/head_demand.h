#pragma once
#include <algorithm>
#include <cstring>
#include <vector>

// All preserves the full-row control contract. Selected names original batch rows.
enum H5HeadMode { H5_HEAD_ALL = 0, H5_HEAD_SELECTED = 1 };
enum H5HeadError { H5_HEAD_OK = 0, H5_HEAD_INVALID_DEMAND = -6, H5_HEAD_LOGITS_UNAVAILABLE = -7 };
extern "C" {
int h5_step_head(void*, int, const int*, const int*, const int*, int*, H5HeadMode, int, const int*);
int h5_step_visual_head(void*, int, const int*, const int*, const int*, const int*, const float*, int*, H5HeadMode, int, const int*);
}

namespace h5 {
inline H5HeadError parse_head_mode(const char* text, H5HeadMode& mode) {
  if (!text) return H5_HEAD_INVALID_DEMAND;
  if (!std::strcmp(text, "all")) { mode = H5_HEAD_ALL; return H5_HEAD_OK; }
  if (!std::strcmp(text, "demand")) { mode = H5_HEAD_SELECTED; return H5_HEAD_OK; }
  return H5_HEAD_INVALID_DEMAND;
}
inline int step_head_rows(void* e, int n, const int* slot, const int* tok, const int* pos, int* out,
                          H5HeadMode mode, const std::vector<int>& rows) {
  if (mode == H5_HEAD_ALL) return h5_step_head(e, n, slot, tok, pos, out, mode, 0, nullptr);
  if (mode == H5_HEAD_SELECTED) return h5_step_head(e, n, slot, tok, pos, out, mode, rows.size(), rows.empty() ? nullptr : rows.data());
  return H5_HEAD_INVALID_DEMAND;
}
class HeadDemand {
  H5HeadMode mode_;
  int count_;
  const int* rows_;
  HeadDemand(H5HeadMode mode, int count, const int* rows) : mode_(mode), count_(count), rows_(rows) {}
public:
  static HeadDemand all() { return HeadDemand(H5_HEAD_ALL, 0, nullptr); }
  static HeadDemand selected(int count, const int* rows) { return HeadDemand(H5_HEAD_SELECTED, count, rows); }
  H5HeadError validate(int batch_rows, bool full_logits) const {
    if (batch_rows <= 0) return H5_HEAD_INVALID_DEMAND;
    switch (mode_) {
      case H5_HEAD_ALL: return count_ == 0 && rows_ == nullptr ? H5_HEAD_OK : H5_HEAD_INVALID_DEMAND;
      case H5_HEAD_SELECTED:
        if (full_logits || count_ < 0 || count_ > batch_rows || (count_ == 0) != (rows_ == nullptr)) return H5_HEAD_INVALID_DEMAND;
        for (int j = 0; j < count_; ++j)
          if (rows_[j] < 0 || rows_[j] >= batch_rows || (j && rows_[j] <= rows_[j - 1])) return H5_HEAD_INVALID_DEMAND;
        return H5_HEAD_OK;
    }
    return H5_HEAD_INVALID_DEMAND;
  }
  bool is_all() const { return mode_ == H5_HEAD_ALL; }
  int count(int batch_rows) const { return is_all() ? batch_rows : count_; }
  int source(int compact_row) const { return is_all() ? compact_row : rows_[compact_row]; }
  void publish(int batch_rows, const int* compact, int* out) const {
    if (!is_all()) std::fill_n(out, batch_rows, -1);
    for (int j = 0; j < count(batch_rows); ++j) out[source(j)] = compact[j];
  }
  static H5HeadError from_api(H5HeadMode mode, int count, const int* rows, int batch_rows, HeadDemand& out) {
    HeadDemand candidate(mode, count, rows);
    const auto error = candidate.validate(batch_rows, false);
    if (error == H5_HEAD_OK) out = candidate;
    return error;
  }
};
enum class HeadUse { IntermediatePrefill, FinishingPrefill, Verify };
inline void append_head_rows(std::vector<int>& rows, int first, int count, HeadUse use) {
  switch (use) {
    case HeadUse::IntermediatePrefill: return;
    case HeadUse::FinishingPrefill: rows.push_back(first + count - 1); return;
    case HeadUse::Verify: for (int j = 0; j < count; ++j) rows.push_back(first + j); return;
  }
}
}
