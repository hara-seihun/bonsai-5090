#pragma once
#include <cstring>

namespace h5 {
enum class F02Mode { Ordinary, H32 };
enum class F02ConfigError { None, Activations, WeightScales, Compact, Carrier, Megakernel, GateUp, Capacity, Nonlinear };
enum class F02Route { Ordinary, H32, Incompatible };
constexpr int F02_ROWS = 512, F02_PLAN_FORMAT = 102, F02_PRODUCER_CONFIG = 2;

inline bool parse_f02_mode(const char* value, F02Mode& mode) {
  if (!value || std::strcmp(value, "0") == 0) { mode = F02Mode::Ordinary; return true; }
  if (std::strcmp(value, "1") == 0) { mode = F02Mode::H32; return true; }
  return false;
}
inline F02ConfigError validate_f02_config(F02Mode mode, bool fp4, bool e4m3, bool compact,
                                        bool carrier, bool mega, bool interleaved, int max_rows, bool nonlinear) {
  if (mode == F02Mode::Ordinary) return F02ConfigError::None;
  if (!fp4) return F02ConfigError::Activations;
  if (!e4m3) return F02ConfigError::WeightScales;
  if (compact) return F02ConfigError::Compact;
  if (carrier) return F02ConfigError::Carrier;
  if (mega) return F02ConfigError::Megakernel;
  if (!interleaved) return F02ConfigError::GateUp;
  if (max_rows < F02_ROWS) return F02ConfigError::Capacity;
  if (nonlinear) return F02ConfigError::Nonlinear;
  return F02ConfigError::None;
}
inline const char* f02_config_error(F02ConfigError error) {
  switch (error) {
    case F02ConfigError::None: return "none";
    case F02ConfigError::Activations: return "H5_F02=1 requires FP4 activations and weights";
    case F02ConfigError::WeightScales: return "H5_F02=1 requires H5_WSCALE=e4m3 (WF2)";
    case F02ConfigError::Compact: return "H5_F02=1 cannot use dense5";
    case F02ConfigError::Carrier: return "H5_F02=1 cannot use scale carrier";
    case F02ConfigError::Megakernel: return "H5_F02=1 requires normal graph path";
    case F02ConfigError::GateUp: return "H5_F02=1 requires interleaved fused gate/up; unset H5_NO_GUFUSE";
    case F02ConfigError::Capacity: return "H5_F02=1 requires engine row capacity >=512";
    case F02ConfigError::Nonlinear: return "H5_F02=1 cannot be composed with F07";
  }
  return "invalid F02 configuration error";
}
inline F02Route select_f02_route(F02Mode mode, int rows, int wf, bool interleaved, bool carrier) {
  if (mode == F02Mode::Ordinary || rows != F02_ROWS) return F02Route::Ordinary;
  if (wf != 2 || !interleaved || carrier) return F02Route::Incompatible;
  return F02Route::H32;
}
}

extern "C" void h5_f02_stats(void* engine, long* steps, long* rows, long* graph_hits);
