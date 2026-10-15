#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace h5 {
enum class Q8ScaleMode { Max, LeastSquares };
inline Q8ScaleMode gdn_q8_scale_mode() {
  const char* value = std::getenv("H5_GDN_Q8_SCALE");
  if (!value || !std::strcmp(value, "ls")) return Q8ScaleMode::LeastSquares;
  if (!std::strcmp(value, "max")) return Q8ScaleMode::Max;
  throw std::invalid_argument("H5_GDN_Q8_SCALE requires max|ls");
}
inline void validate_q8_scale_mode(Q8ScaleMode mode) {
  switch (mode) {
    case Q8ScaleMode::Max: case Q8ScaleMode::LeastSquares: return;
  }
  throw std::invalid_argument("invalid Q8ScaleMode");
}
inline const char* q8_scale_name(Q8ScaleMode mode) {
  switch (mode) {
    case Q8ScaleMode::Max: return "max";
    case Q8ScaleMode::LeastSquares: return "ls";
  }
  std::abort();
}
} // namespace h5
