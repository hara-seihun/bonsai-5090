#include "f02_route.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
  using namespace h5;
  F02Mode mode = F02Mode::Ordinary;
  assert(parse_f02_mode(nullptr, mode) && mode == F02Mode::Ordinary);
  assert(parse_f02_mode("0", mode) && mode == F02Mode::Ordinary);
  assert(parse_f02_mode("1", mode) && mode == F02Mode::H32);
  for (const char* invalid : {"", "-1", "2", "true", "1junk"}) assert(!parse_f02_mode(invalid, mode));
  for (int bits = 0; bits < 256; bits++) {
    bool fp4 = bits & 1, e4m3 = bits & 2, compact = bits & 4, carrier = bits & 8;
    bool mega = bits & 16, il = bits & 32, capacity = bits & 64, nonlinear = bits & 128;
    auto error = validate_f02_config(F02Mode::H32, fp4, e4m3, compact, carrier, mega, il, capacity ? 512 : 511, nonlinear);
    assert((error == F02ConfigError::None) == (fp4 && e4m3 && !compact && !carrier && !mega && il && capacity && !nonlinear));
    assert(validate_f02_config(F02Mode::Ordinary, fp4, e4m3, compact, carrier, mega, il, 1, nonlinear) == F02ConfigError::None);
  }
  for (int rows : {1, 8, 64, 128, 256, 511, 512, 513, 1024})
    for (int wf : {0, 1, 2, 3}) for (bool il : {false, true}) for (bool carrier : {false, true}) {
      assert(select_f02_route(F02Mode::Ordinary, rows, wf, il, carrier) == F02Route::Ordinary);
      auto selected = select_f02_route(F02Mode::H32, rows, wf, il, carrier);
      if (rows != 512) assert(selected == F02Route::Ordinary);
      else assert(selected == (wf == 2 && il && !carrier ? F02Route::H32 : F02Route::Incompatible));
    }
  std::puts("F02 mode/compatibility/exact-row route contracts passed");
}
