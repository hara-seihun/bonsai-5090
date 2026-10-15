#include "f07_contract.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
using namespace h5;

int main() {
  F07Mode mode = F07Mode::Degree8;
  assert(parse_f07_mode(nullptr, mode) && mode == F07Mode::Ordinary);
  assert(parse_f07_mode("0", mode) && mode == F07Mode::Ordinary);
  assert(parse_f07_mode("degree8", mode) && mode == F07Mode::Degree8);
  for (const char* value : {"", "1", "Degree8", "degree8 ", "pade", "-1"}) {
    assert(!parse_f07_mode(value, mode));
    assert(mode == F07Mode::Degree8);
  }
  for (unsigned bits = 0; bits < 128; ++bits) {
    const bool fp4 = bits & 1, compact = bits & 2, carrier = bits & 4, e4m3 = bits & 8;
    const bool mega = bits & 16, f02 = bits & 32;
    const int capacity = bits & 64 ? 512 : 128;
    const auto error = validate_f07_config(F07Mode::Degree8, fp4, compact, carrier, e4m3, mega, f02, capacity);
    assert((error == F07ConfigError::None) == (bits == 65));
    assert(validate_f07_config(F07Mode::Ordinary, fp4, compact, carrier, e4m3, mega, f02, capacity) == F07ConfigError::None);
    assert(f07_config_error_name(error));
  }
  assert(validate_f07_config(static_cast<F07Mode>(99), true, false, false, false, false, false, 512) == F07ConfigError::UnknownMode);
  for (int rows : {1, 8, 16, 32, 64, 128, 256, 511, 512, 513, 1024}) {
    for (int wf = 0; wf <= 3; ++wf) {
      for (bool il : {false, true}) for (bool carrier : {false, true}) {
        assert(select_f07_route(F07Mode::Ordinary, rows, wf, il, carrier) == F07Route::Ordinary);
        const auto selected = select_f07_route(F07Mode::Degree8, rows, wf, il, carrier);
        if (wf != 0 || !il || carrier) assert(selected == F07Route::Incompatible);
        else assert(selected == (rows == 512 ? F07Route::Degree8 : F07Route::Ordinary));
      }
    }
  }
  assert(select_f07_route(F07Mode::Degree8, 0, 0, true, false) == F07Route::Incompatible);
  assert(select_f07_route(static_cast<F07Mode>(99), 512, 0, true, false) == F07Route::Incompatible);
  assert(f07_launch_shape_valid(512, 5120, 34816, 17408));
  assert(f07_launch_shape_valid(512, 128, 64, 32));
  assert(!f07_launch_shape_valid(511, 5120, 34816, 17408));
  assert(!f07_launch_shape_valid(512, 5121, 34816, 17408));
  assert(!f07_launch_shape_valid(512, 0, 34816, 17408));
  assert(!f07_launch_shape_valid(512, 5120, 34815, 17408));
  assert(!f07_launch_shape_valid(512, 5120, 32, 16));
  assert(!f07_launch_shape_valid(512, 5120, 34816, 17407));
  puts("F07 parser, compatibility, exact-row dispatch and launch-shape contracts passed");
}
