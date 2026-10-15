#pragma once
#include "model.h"
#include <cstdlib>

namespace h5 {
enum class StateFormat { F32, BF16, F16, Q8 };

// Each layer keeps the packed payload contiguous; row scales travel with it in
// reset, clone and snapshot operations. Q8 pending replay inputs remain FP16.
struct GdnQ8Layer {
  int8_t data[HV][SS][SS];
  float scales[HV][SS];
};
static_assert(sizeof(GdnQ8Layer) == (size_t)HV * SS * (SS + sizeof(float)));

inline size_t pending_state_element_bytes(StateFormat format) {
  switch (format) {
    case StateFormat::F32: return sizeof(float);
    case StateFormat::BF16: case StateFormat::F16: case StateFormat::Q8: return 2;
  }
  std::abort();
}
inline size_t gdn_state_slot_bytes(StateFormat format) {
  if (format == StateFormat::Q8) return (size_t)NGDN * sizeof(GdnQ8Layer);
  return (size_t)NGDN * HV * SS * SS * pending_state_element_bytes(format);
}
inline const char* state_format_name(StateFormat format) {
  switch (format) {
    case StateFormat::F32: return "f32";
    case StateFormat::BF16: return "bf16";
    case StateFormat::F16: return "f16";
    case StateFormat::Q8: return "q8 (FP32 recurrence, FP16 pending)";
  }
  std::abort();
}
} // namespace h5
