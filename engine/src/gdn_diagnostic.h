#pragma once
#include "state_format.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace h5 {
enum class GdnDiagnosticMode { Observe, ShadowF16, ShadowQ8 };
enum class GdnDiagnosticSource { Driver, Shadow };
inline bool diagnostic_shadow_driver_format(StateFormat format) {
  return format == StateFormat::F16 || format == StateFormat::F32;
}
enum class GdnDiagnosticError {
  None = 0, BadArgument = -740, UnsupportedRoute = -741, AlreadyUsed = -742,
  NoShadow = -743, BufferTooSmall = -744, Cuda = -745,
  UnsupportedSpeculation = -746, HostAllocation = -747, EngineStep = -748,
  ObserverDisabled = -749, NotSelected = -750
};
struct GdnDiagnosticResult {
  GdnDiagnosticError error;
  int detail; // CUDA error code or original engine-step status, otherwise zero.
};
struct GdnDiagnosticInfo {
  StateFormat format;
  int slots, context, max_rows;
};
enum class GdnReadoutObservation { Unobserved, Observed };
struct GdnReadoutSample {
  GdnReadoutObservation observation = GdnReadoutObservation::Unobserved;
  int row = -1, token = -1, position = -1;
  size_t elements = 0, nonfinite = 0, nan = 0, infinity = 0;
  double max_abs = std::numeric_limits<double>::quiet_NaN();
};
inline GdnReadoutSample diagnostic_readout_stats(const float* values, size_t count, int row, int token, int position) {
  GdnReadoutSample out{};
  if (!values || !count || row < 0 || token < 0 || position < 0) return out;
  out.observation = GdnReadoutObservation::Observed; out.row = row; out.token = token; out.position = position;
  out.elements = count; out.max_abs = 0;
  for (size_t i = 0; i < count; ++i) {
    const double value = values[i];
    out.nan += std::isnan(value); out.infinity += std::isinf(value);
    if (!std::isnan(value)) out.max_abs = std::fmax(out.max_abs, std::fabs(value));
  }
  out.nonfinite = out.nan + out.infinity;
  if (out.nan) out.max_abs = std::numeric_limits<double>::quiet_NaN();
  return out;
}
constexpr size_t GDN_DIAGNOSTIC_HEAD_ELEMENTS = (size_t)SS * SS;
inline GdnDiagnosticError diagnostic_environment() {
  const char* names[] = {"H5_STATE_FUSION", "H5_MEGA", "H5_GDN_SINGLE", "H5_GDN_ARITH", "H5_GDN_SCALED", "H5_GDN_PIPE", "H5_GDN_OCC", "H5_TAIL_PREP", "H5_GDN_DELTA"};
  const char* values[] = {"0", "0", "off", "fp32", "0", "0", "2", "0", "0"};
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    const char* value = std::getenv(names[i]);
    if (value && std::strcmp(value, values[i])) return GdnDiagnosticError::UnsupportedRoute;
  }
  return GdnDiagnosticError::None;
}
inline int gdn_model_layer(int gl) { return gl >= 0 && gl < NGDN ? 4 * (gl / 3) + gl % 3 : -1; }
inline int gdn_layer_index(int model_layer) {
  return model_layer >= 0 && model_layer < NLAYER && !is_attn(model_layer) ? 3 * (model_layer / 4) + model_layer % 4 : -1;
}
inline float diagnostic_f16(uint16_t bits) {
  const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
  const float magnitude = exponent == 31 ? (fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
      exponent == 0 ? std::ldexp((float)fraction, -24) : std::ldexp((float)(1024 + fraction), exponent - 25);
  return bits & 32768 ? -magnitude : magnitude;
}
inline GdnDiagnosticError diagnostic_decode_head(StateFormat format, const void* payload, const float* scales, float* out) {
  if (!payload || !out) return GdnDiagnosticError::BadArgument;
  for (size_t i = 0; i < GDN_DIAGNOSTIC_HEAD_ELEMENTS; ++i) {
    switch (format) {
      case StateFormat::F32: out[i] = ((const float*)payload)[i]; break;
      case StateFormat::F16: out[i] = diagnostic_f16(((const uint16_t*)payload)[i]); break;
      case StateFormat::BF16: {
        const uint32_t bits = (uint32_t)((const uint16_t*)payload)[i] << 16;
        std::memcpy(out + i, &bits, sizeof(float)); break;
      }
      case StateFormat::Q8:
        if (!scales) return GdnDiagnosticError::BadArgument;
        out[i] = (float)((const int8_t*)payload)[i] * scales[i / SS]; break;
      default: return GdnDiagnosticError::BadArgument;
    }
  }
  return GdnDiagnosticError::None;
}
enum class GdnRelativeKind { Finite, ZeroReferenceEqual, ZeroReferenceDifferent, Nonfinite };
struct GdnHeadMetrics {
  size_t reference_nonfinite, candidate_nonfinite;
  double reference_l2, candidate_l2, reference_rms, candidate_rms;
  double absolute_l2, error_rms, error_max, relative_l2;
  GdnRelativeKind relative_kind;
};
inline GdnHeadMetrics diagnostic_metrics(const float* reference, const float* candidate, size_t count) {
  GdnHeadMetrics m{};
  double ref2 = 0, cand2 = 0, err2 = 0;
  for (size_t i = 0; i < count; ++i) {
    const double a = reference[i], b = candidate[i];
    m.reference_nonfinite += !std::isfinite(a); m.candidate_nonfinite += !std::isfinite(b);
    if (std::isfinite(a)) ref2 += a * a;
    if (std::isfinite(b)) cand2 += b * b;
    if (!std::isfinite(a) || !std::isfinite(b)) continue;
    const double d = b - a;
    err2 += d * d; m.error_max = std::fmax(m.error_max, std::fabs(d));
  }
  if (!count || m.reference_nonfinite || m.candidate_nonfinite) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    m.reference_l2 = !count || m.reference_nonfinite ? nan : std::sqrt(ref2);
    m.reference_rms = !count || m.reference_nonfinite ? nan : std::sqrt(ref2 / count);
    m.candidate_l2 = !count || m.candidate_nonfinite ? nan : std::sqrt(cand2);
    m.candidate_rms = !count || m.candidate_nonfinite ? nan : std::sqrt(cand2 / count);
    m.absolute_l2 = m.error_rms = m.error_max = m.relative_l2 = nan;
    m.relative_kind = GdnRelativeKind::Nonfinite;
    return m;
  }
  m.reference_l2 = std::sqrt(ref2); m.candidate_l2 = std::sqrt(cand2);
  m.reference_rms = std::sqrt(ref2 / count); m.candidate_rms = std::sqrt(cand2 / count);
  m.absolute_l2 = std::sqrt(err2); m.error_rms = std::sqrt(err2 / count);
  if (ref2 > 0) { m.relative_kind = GdnRelativeKind::Finite; m.relative_l2 = std::sqrt(err2 / ref2); }
  else if (err2 == 0) { m.relative_kind = GdnRelativeKind::ZeroReferenceEqual; m.relative_l2 = 0; }
  else { m.relative_kind = GdnRelativeKind::ZeroReferenceDifferent; m.relative_l2 = std::numeric_limits<double>::infinity(); }
  return m;
}
}
extern "C" {
// Exclusive engine ownership is required for all diagnostic calls. Configuration is one-shot,
// before the first step. Reads synchronize the engine stream and return caller-owned host floats.
h5::GdnDiagnosticResult h5_gdn_diagnostic_configure(void*, h5::GdnDiagnosticMode);
h5::GdnDiagnosticResult h5_gdn_diagnostic_info(void*, h5::GdnDiagnosticSource, h5::GdnDiagnosticInfo*);
h5::GdnDiagnosticResult h5_gdn_diagnostic_read(void*, h5::GdnDiagnosticSource, int slot, int model_layer, int head, float*, size_t elements);
h5::GdnDiagnosticResult h5_gdn_diagnostic_step(void*, int rows, const int* slots, const int* tokens, const int* positions, int* output);
// Sparse observer: configure selected physical layers/heads before the first step, then arm only
// selected STEP checkpoints. Samples describe the last executed row per touched slot, before norm.
h5::GdnDiagnosticResult h5_gdn_readout_configure(void*, int layer_count, const int* layers, int head_count, const int* heads);
h5::GdnDiagnosticResult h5_gdn_readout_arm(void*);
h5::GdnDiagnosticResult h5_gdn_readout_read(void*, h5::GdnDiagnosticSource, int slot, int model_layer, int head, h5::GdnReadoutSample*);
}
