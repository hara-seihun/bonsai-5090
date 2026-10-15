#include "../src/serving_policy.h"
#include <cassert>
#include <cstdio>
#include <limits>
using namespace h5::serving;
int main() {
  Config c;
  assert(configure(nullptr, nullptr, nullptr, nullptr, 1, false, c) == Error::None);
  assert(!c.fair && !c.reserve && c.request_slots == 1 && c.physical_slots == 1 && c.template_slot == -1);
  assert(configure("0", "0", nullptr, "0", 64, true, c) == Error::None);
  assert(!c.fair && c.request_slots == 64 && c.physical_slots == 65);
  for (const char* bad : {"", "false", "2", "01", " 1"}) {
    assert(configure(bad, nullptr, nullptr, nullptr, 64, false, c) == Error::InvalidFlag);
    assert(configure(nullptr, bad, nullptr, nullptr, 64, false, c) == Error::InvalidFlag);
    assert(configure(nullptr, nullptr, nullptr, bad, 64, false, c) == Error::InvalidFlag);
  }
  assert(configure("1", nullptr, "prefix", "1", 64, false, c) == Error::None);
  assert(c.fair && c.check && c.reserve && c.request_slots == 63 && c.physical_slots == 64 && c.template_slot == 63);
  assert(configure("1", "0", "prefix", "0", 2, true, c) == Error::None);
  assert(c.request_slots == 1 && c.physical_slots == 3 && c.template_slot == 2);
  assert(configure(nullptr, nullptr, "", nullptr, 64, false, c) == Error::EmptyPrefix);
  assert(configure(nullptr, nullptr, nullptr, "1", 64, false, c) == Error::MissingPrefix);
  assert(configure(nullptr, nullptr, "prefix", nullptr, 1, false, c) == Error::NoRequestSlots);
  assert(configure(nullptr, nullptr, "prefix", "1", 2, false, c) == Error::DiagnosticSlots);
  assert(prefix_match({1,2}, {1,2,3}, false));
  assert(!prefix_match({1,2}, {1,2}, false) && !prefix_match({1,2}, {1,3,4}, false));
  assert(!prefix_match({1,2}, {1,2,3}, true) && !prefix_match({}, {1,2}, false));
  auto a = allocate({{0, 1000, 0}, {1, 20, 0}, {2, 30, 0}}, 3, 128, 0);
  assert(a.error == Error::None && a.rows == std::vector<int>({78,20,30}));
  a = allocate({{0, 1000, 1}, {1, 20, 0}}, 2, 128, 1);
  assert(a.rows == std::vector<int>({128,0}));
  for (int cursor = 0; cursor < 4; ++cursor) {
    a = allocate({{0,1000,2}, {1,1000,2}, {2,1000,2}, {3,1000,2}}, 4, 128, cursor);
    assert(a.rows[cursor] == 128);
    int sum = 0; for (int n : a.rows) sum += n; assert(sum == 128);
  }
  assert(allocate({{0,1,0}, {0,2,0}}, 1, 128, 0).error == Error::InvalidAllocation);
  assert(allocate({{1,1,0}}, 1, 128, 0).error == Error::InvalidAllocation);
  assert(allocate({{0,1,std::numeric_limits<double>::quiet_NaN()}}, 1, 128, 0).error == Error::InvalidAllocation);
  assert(allocate({}, 1, 128, 0).rows == std::vector<int>({0}));
  assert(copy_guard(2,0,3,3,4096,0,0) == CopyError::None);
  assert(copy_guard(2,2,3,3,4096,0,0) == CopyError::InvalidRange);
  assert(copy_guard(2,0,4097,3,4096,0,0) == CopyError::InvalidRange);
  assert(copy_guard(2,0,3,3,4096,1,0) == CopyError::PendingDonor);
  assert(copy_guard(2,0,3,3,4096,0,1) == CopyError::PendingDonor);
  puts("serving contracts pass: explicit flags, disjoint slot ownership, prefix eligibility, fair128 budget/aging/rotation, pending donor rejection");
}
