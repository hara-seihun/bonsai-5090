#pragma once
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace h5::diagnostic {
enum class LogitMetricsKind { Finite, RejectedNonfinite, BadArgument };
struct LogitSummary {
  LogitMetricsKind kind = LogitMetricsKind::BadArgument;
  size_t finite = 0, nan = 0, infinity = 0;
  int top1 = -1, top2 = -1;
  double margin = std::numeric_limits<double>::quiet_NaN();
};
struct LogitMetrics {
  LogitMetricsKind kind = LogitMetricsKind::BadArgument;
  LogitSummary reference, candidate;
  double kl_reference_candidate = std::numeric_limits<double>::quiet_NaN();
};
inline LogitSummary logit_summary(const float* values, size_t count) {
  LogitSummary out;
  if (!values || count < 2 || count > static_cast<size_t>(std::numeric_limits<int>::max())) return out;
  for (size_t i = 0; i < count; ++i) {
    const double value = values[i];
    if (std::isnan(value)) ++out.nan;
    else if (std::isinf(value)) ++out.infinity;
    else ++out.finite;
  }
  if (out.nan || out.infinity) { out.kind = LogitMetricsKind::RejectedNonfinite; return out; }
  out.kind = LogitMetricsKind::Finite;
  out.top1 = 0; out.top2 = 1;
  if (values[1] > values[0]) { out.top1 = 1; out.top2 = 0; }
  for (size_t i = 2; i < count; ++i) {
    if (values[i] > values[out.top1]) { out.top2 = out.top1; out.top1 = static_cast<int>(i); }
    else if (values[i] > values[out.top2]) out.top2 = static_cast<int>(i);
  }
  out.margin = static_cast<double>(values[out.top1]) - static_cast<double>(values[out.top2]);
  return out;
}
inline LogitMetrics logit_metrics(const float* reference, const float* candidate, size_t count) {
  LogitMetrics out;
  out.reference = logit_summary(reference, count); out.candidate = logit_summary(candidate, count);
  if (out.reference.kind == LogitMetricsKind::BadArgument || out.candidate.kind == LogitMetricsKind::BadArgument) return out;
  if (out.reference.kind == LogitMetricsKind::RejectedNonfinite || out.candidate.kind == LogitMetricsKind::RejectedNonfinite) {
    out.kind = LogitMetricsKind::RejectedNonfinite; return out;
  }
  const double max_reference = reference[out.reference.top1], max_candidate = candidate[out.candidate.top1];
  double sum_reference = 0, sum_candidate = 0;
  for (size_t i = 0; i < count; ++i) {
    sum_reference += std::exp(static_cast<double>(reference[i]) - max_reference);
    sum_candidate += std::exp(static_cast<double>(candidate[i]) - max_candidate);
  }
  // Keep maxima separate: adding log(sum) to a 1e30 maximum loses the normalization.
  const double log_sum_reference = std::log(sum_reference), log_sum_candidate = std::log(sum_candidate);
  double kl = 0, correction = 0;
  for (size_t i = 0; i < count; ++i) {
    const double a = static_cast<double>(reference[i]) - max_reference;
    const double b = static_cast<double>(candidate[i]) - max_candidate;
    const double term = (std::exp(a) / sum_reference) * ((a - b) + (log_sum_candidate - log_sum_reference));
    const double adjusted = term - correction, next = kl + adjusted;
    correction = (next - kl) - adjusted; kl = next;
  }
  out.kind = LogitMetricsKind::Finite; out.kl_reference_candidate = kl;
  return out;
}
inline const char* logit_metrics_name(LogitMetricsKind kind) {
  switch (kind) {
    case LogitMetricsKind::Finite: return "finite";
    case LogitMetricsKind::RejectedNonfinite: return "rejected_nonfinite";
    case LogitMetricsKind::BadArgument: return "bad_argument";
  }
  std::abort();
}
}
