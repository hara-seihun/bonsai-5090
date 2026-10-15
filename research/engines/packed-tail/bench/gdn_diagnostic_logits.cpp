#include "../tools/gdn_diagnostic_logits.h"
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace h5::diagnostic;
namespace {
void require(bool value, const char* message) {
  if (!value) { std::cerr << "gdn diagnostic logits: " << message << '\n'; std::exit(1); }
}
bool near(double a, double b, double tolerance = 1e-12) { return std::fabs(a - b) <= tolerance * std::fmax(1., std::fabs(b)); }
}
int main() {
  const float uniform[] = {0, 0, 0}, large[] = {1e30f, 1e30f, 1e30f}, other[] = {0, 1, 2};
  const auto base = logit_metrics(uniform, other, 3), shifted = logit_metrics(large, other, 3);
  require(base.kind == LogitMetricsKind::Finite && near(base.kl_reference_candidate, std::log(1 + std::exp(1.) + std::exp(2.)) - 1 - std::log(3.)), "analytic KL");
  require(near(shifted.kl_reference_candidate, base.kl_reference_candidate), "large common reference offset must retain normalization");
  const auto reverse_shift = logit_metrics(other, large, 3), reverse_base = logit_metrics(other, uniform, 3);
  require(near(reverse_shift.kl_reference_candidate, reverse_base.kl_reference_candidate), "large candidate offset");
  const float gap[] = {1e30f, -1e30f}, opposite[] = {-1e30f, 1e30f};
  require(near(logit_metrics(gap, opposite, 2).kl_reference_candidate, 2 * static_cast<double>(gap[0])), "extreme finite gaps");
  const float moderate[] = {1000, 0}, swapped[] = {0, 1000};
  require(near(logit_metrics(moderate, swapped, 2).kl_reference_candidate, 1000), "underflowed softmax must not create infinity");
  require(logit_metrics(large, large, 3).kl_reference_candidate == 0, "identical large logits");
  const float ties[] = {2, 3, 3, 1};
  const auto tied = logit_summary(ties, 4);
  require(tied.top1 == 1 && tied.top2 == 2 && tied.margin == 0, "ties choose lowest vocabulary index");
  const float ranked[] = {-3, 5, 4};
  const auto rank = logit_summary(ranked, 3);
  require(rank.top1 == 1 && rank.top2 == 2 && rank.margin == 1, "top1/top2 margin");
  const float bad[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  const auto rejected = logit_metrics(bad, other, 3);
  require(rejected.kind == LogitMetricsKind::RejectedNonfinite && std::isnan(rejected.kl_reference_candidate), "nonfinite KL is rejected");
  require(rejected.reference.finite == 0 && rejected.reference.nan == 1 && rejected.reference.infinity == 2 && rejected.reference.top1 == -1 && std::isnan(rejected.reference.margin), "no partial nonfinite metrics");
  require(rejected.candidate.kind == LogitMetricsKind::Finite && rejected.candidate.finite == 3 && rejected.candidate.top1 == 2 && rejected.candidate.margin == 1, "independent finite candidate summary");
  const auto invalid = logit_metrics(nullptr, other, 3);
  require(invalid.kind == LogitMetricsKind::BadArgument, "null input explicit error");
  require(logit_metrics(other, other, 1).kind == LogitMetricsKind::BadArgument, "too-small vocabulary explicit error");
  std::vector<float> vocabulary(248320, 1e30f);
  const auto vocab = logit_metrics(vocabulary.data(), vocabulary.data(), vocabulary.size());
  require(vocab.kind == LogitMetricsKind::Finite && vocab.kl_reference_candidate == 0 && vocab.reference.finite == vocabulary.size(), "full vocabulary large-finite smoke");
  std::cout << "gdn diagnostic logit CPU tests passed\n";
}
