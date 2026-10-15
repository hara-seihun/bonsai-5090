#pragma once
#include <cstring>

namespace h5 {
enum class F07Mode { Ordinary, Degree8 };
enum class F07ConfigError { None, UnknownMode, RequiresFp4, UnsupportedWeightRoute, Megakernel, F02Conflict, RowCapacity };
enum class F07Route { Ordinary, Degree8, Incompatible };

inline bool parse_f07_mode(const char* value, F07Mode& mode) {
  if (!value || std::strcmp(value, "0") == 0) { mode = F07Mode::Ordinary; return true; }
  if (std::strcmp(value, "degree8") == 0) { mode = F07Mode::Degree8; return true; }
  return false;
}
constexpr F07ConfigError validate_f07_config(F07Mode mode, bool fp4, bool compact, bool carrier,
                                            bool e4m3, bool mega, bool f02, int max_rows) {
  if (mode == F07Mode::Ordinary) return F07ConfigError::None;
  if (mode != F07Mode::Degree8) return F07ConfigError::UnknownMode;
  if (!fp4) return F07ConfigError::RequiresFp4;
  if (compact || carrier || e4m3) return F07ConfigError::UnsupportedWeightRoute;
  if (mega) return F07ConfigError::Megakernel;
  if (f02) return F07ConfigError::F02Conflict;
  if (max_rows < 512) return F07ConfigError::RowCapacity;
  return F07ConfigError::None;
}
constexpr const char* f07_config_error_name(F07ConfigError error) {
  switch (error) {
    case F07ConfigError::None: return "none";
    case F07ConfigError::UnknownMode: return "H5_F07 has unknown mode";
    case F07ConfigError::RequiresFp4: return "H5_F07=degree8 requires FP4 target activations";
    case F07ConfigError::UnsupportedWeightRoute: return "H5_F07=degree8 requires original WF0 FP16 scales, without dense5, scale carrier or E4M3";
    case F07ConfigError::Megakernel: return "H5_F07=degree8 requires the normal graph path";
    case F07ConfigError::F02Conflict: return "H5_F07=degree8 and H5_F02=1 are incompatible";
    case F07ConfigError::RowCapacity: return "H5_F07=degree8 requires at least 512 rows of capacity";
  }
  return "unknown F07 config error";
}
constexpr F07Route select_f07_route(F07Mode mode, int rows, int wf, bool interleaved, bool carrier) {
  if (mode == F07Mode::Ordinary) return F07Route::Ordinary;
  if (mode != F07Mode::Degree8 || wf != 0 || !interleaved || carrier || rows < 1) return F07Route::Incompatible;
  return rows == 512 ? F07Route::Degree8 : F07Route::Ordinary;
}
constexpr bool f07_launch_shape_valid(int rows, int k, int n, int ldo) {
  return rows == 512 && k >= 128 && k % 128 == 0 && n >= 64 && n % 64 == 0 && ldo >= n / 2;
}
} // namespace h5
