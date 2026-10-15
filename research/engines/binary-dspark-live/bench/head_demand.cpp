#include "head_demand.h"
#include <cassert>
#include <iostream>
#include <numeric>

int main() {
  using h5::HeadDemand;
  using h5::HeadUse;
  const int sorted[] = {0, 2, 130, 255};
  const int duplicate[] = {2, 2}, reversed[] = {2, 1}, negative[] = {-1}, past[] = {256};
  assert(HeadDemand::all().validate(256, true) == H5_HEAD_OK);
  assert(HeadDemand::selected(4, sorted).validate(256, false) == H5_HEAD_OK);
  assert(HeadDemand::selected(0, nullptr).validate(256, false) == H5_HEAD_OK);
  for (const int* invalid : {duplicate, reversed}) assert(HeadDemand::selected(2, invalid).validate(256, false) == H5_HEAD_INVALID_DEMAND);
  for (const int* invalid : {negative, past}) assert(HeadDemand::selected(1, invalid).validate(256, false) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::selected(4, sorted).validate(256, true) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::selected(-1, nullptr).validate(256, false) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::selected(0, sorted).validate(256, false) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::selected(1, nullptr).validate(256, false) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::selected(4, sorted).validate(3, false) == H5_HEAD_INVALID_DEMAND);
  HeadDemand checked = HeadDemand::all();
  assert(HeadDemand::from_api(static_cast<H5HeadMode>(7), 0, nullptr, 256, checked) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::from_api(H5_HEAD_ALL, 4, sorted, 256, checked) == H5_HEAD_INVALID_DEMAND);
  assert(HeadDemand::from_api(H5_HEAD_SELECTED, 4, sorted, 256, checked) == H5_HEAD_OK);
  std::vector<int> out(256, 99);
  int compact[] = {11, 12, 13, 14};
  checked.publish(256, compact, out.data());
  for (int j = 0; j < 256; ++j) {
    auto it = std::find(std::begin(sorted), std::end(sorted), j);
    assert(out[j] == (it == std::end(sorted) ? -1 : compact[it - sorted]));
  }
  HeadDemand::selected(0, nullptr).publish(256, nullptr, out.data());
  assert(std::all_of(out.begin(), out.end(), [](int x) { return x == -1; }));
  std::vector<int> all(256); std::iota(all.begin(), all.end(), 10);
  HeadDemand::all().publish(256, all.data(), out.data()); assert(out == all);

  // Mixed step: unfinished prompt, finishing prompt, verifier, ordinary decode,
  // another finishing prompt, and both private tree paths. Source IDs never compact state.
  std::vector<int> rows;
  h5::append_head_rows(rows, 0, 128, HeadUse::IntermediatePrefill);
  h5::append_head_rows(rows, 128, 3, HeadUse::FinishingPrefill);
  h5::append_head_rows(rows, 131, 8, HeadUse::Verify);
  h5::append_head_rows(rows, 139, 1, HeadUse::Verify);
  h5::append_head_rows(rows, 140, 2, HeadUse::FinishingPrefill);
  h5::append_head_rows(rows, 142, 16, HeadUse::Verify);
  assert(rows.size() == 27 && rows[0] == 130 && rows[9] == 139 && rows[10] == 141 && rows.back() == 157);
  assert(HeadDemand::selected(rows.size(), rows.data()).validate(158, false) == H5_HEAD_OK);

  // Prompt crossing 128/256 chunk boundaries: only the completing call has a tail.
  for (int length : {1, 127, 128, 129, 255, 256, 257}) {
    int heads = 0;
    for (int start = 0; start < length; start += 128) {
      int count = std::min(128, length - start); rows.clear();
      h5::append_head_rows(rows, 0, count, start + count == length ? HeadUse::FinishingPrefill : HeadUse::IntermediatePrefill);
      assert(HeadDemand::selected(rows.size(), rows.empty() ? nullptr : rows.data()).validate(count, false) == H5_HEAD_OK);
      if (!rows.empty()) { ++heads; assert(rows[0] == count - 1); }
    }
    assert(heads == 1);
  }
  H5HeadMode mode = H5_HEAD_SELECTED;
  assert(h5::parse_head_mode("all", mode) == H5_HEAD_OK && mode == H5_HEAD_ALL);
  assert(h5::parse_head_mode("demand", mode) == H5_HEAD_OK && mode == H5_HEAD_SELECTED);
  for (const char* text : {"", "tail", "0", "ALL"}) assert(h5::parse_head_mode(text, mode) == H5_HEAD_INVALID_DEMAND);
  assert(h5::parse_head_mode(nullptr, mode) == H5_HEAD_INVALID_DEMAND);
  std::cout << "head demand contracts pass: validation, no-head, source IDs, mixed/tree rows, chunk boundaries, explicit policy\n";
}
