#pragma once
#include <cstdint>

namespace h5 {
enum class PfChainGranularity : uint8_t { Pieces16KiB, CtaShare };
enum class PfChainPlacement : uint8_t { Prologue, Tail };
struct PfChainSlice { unsigned begin, bytes; };

#if defined(__CUDACC__)
#define H5_PF_HD __host__ __device__
#else
#define H5_PF_HD
#endif
// Callers supply a 16-byte-aligned byte cap and a valid grid/CTA coordinate.
H5_PF_HD constexpr PfChainSlice pf_chain_slice(unsigned bytes, unsigned ncta, unsigned cta) {
  const unsigned per = (unsigned)(((uint64_t(bytes) + ncta - 1) / ncta + 15u) & ~uint64_t(15u));
  const uint64_t begin = uint64_t(cta) * per;
  if (begin >= bytes) return {bytes, 0};
  const unsigned remaining = bytes - (unsigned)begin;
  return {(unsigned)begin, remaining < per ? remaining : per};
}
#undef H5_PF_HD
}
