#pragma once
#include "kernels.cuh"
#include <stdexcept>

namespace h5 {
enum class GdnSingleMode { Off, Separate, Fused };
inline GdnSingleMode gdn_single_mode() {
  const char* s = getenv("H5_GDN_SINGLE");
  if (!s || !strcmp(s, "off")) return GdnSingleMode::Off;
  if (!strcmp(s, "separate")) return GdnSingleMode::Separate;
  if (!strcmp(s, "fused")) return GdnSingleMode::Fused;
  throw std::runtime_error("H5_GDN_SINGLE requires off|separate|fused");
}
__host__ __device__ constexpr bool gdn_single_eligible(const Seg& s) {
  return s.len == 1 && s.commit == 0 && s.keep == 0;
}
inline bool gdn_single_eligible(const Seg* segs, int nseg) {
  if (nseg < 1) return false;
  for (int i = 0; i < nseg; ++i) if (!gdn_single_eligible(segs[i])) return false;
  return true;
}
size_t gdn_single_pad_bytes();
void validate_gdn_single(GdnSingleMode, StateFormat, bool state_fusion, bool mega, unsigned tail_mask);
void launch_gdn_single(const GdnArgs&, int nseg, StateFormat, cudaStream_t);
void launch_conv_gdn_single(const ConvArgs&, const GdnArgs&, int nseg, StateFormat, cudaStream_t);
}
