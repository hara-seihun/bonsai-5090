#pragma once
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define H5_SAMPLE_HD __host__ __device__
#else
#define H5_SAMPLE_HD
#endif

namespace h5::sampling {
H5_SAMPLE_HD inline uint64_t noise_mix(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
H5_SAMPLE_HD inline double noise_uniform(uint64_t seed, uint64_t index, uint64_t token) {
  const uint64_t key = noise_mix(seed + UINT64_C(0x9e3779b97f4a7c15)) ^
                       noise_mix(index + UINT64_C(0xd1b54a32d192ed03)) ^
                       noise_mix(token + UINT64_C(0x8cb92baa3f3d8dd7));
  return (double(noise_mix(key) >> 12) + 0.5) * 0x1p-52;
}
H5_SAMPLE_HD inline double noise_gumbel(uint64_t seed, uint64_t index, uint64_t token) {
  return -::log(-::log(noise_uniform(seed, index, token)));
}
}
#undef H5_SAMPLE_HD
