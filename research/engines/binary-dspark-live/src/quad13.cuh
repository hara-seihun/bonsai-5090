#pragma once
#include <cstddef>
#include <cstdint>

namespace quad13 {
constexpr int bytes_per_block = 26;
constexpr int representatives = 41;
constexpr unsigned legal_pairs = 6561;

inline unsigned encode_quad(const int8_t* t) {
  return unsigned(t[0] + 1) + 3u * unsigned(t[1] + 1) +
         9u * unsigned(t[2] + 1) + 27u * unsigned(t[3] + 1);
}
inline void encode_block(const int8_t* t, uint8_t* dst) {
  for (int b = 0; b < bytes_per_block; ++b) dst[b] = 0;
  for (int p = 0; p < 16; ++p) {
    unsigned z = encode_quad(t + 8 * p) + 81 * encode_quad(t + 8 * p + 4);
    int bit = 13 * p;
    for (int j = 0; j < 13; ++j)
      dst[(bit + j) / 8] |= uint8_t(((z >> j) & 1) << ((bit + j) % 8));
  }
}
inline unsigned pair_at(const uint8_t* src, int p) {
  int bit = 13 * p, b = bit / 8, shift = bit % 8;
  unsigned z = unsigned(src[b]) | (unsigned(src[b + 1]) << 8);
  if (shift > 3) z |= unsigned(src[b + 2]) << 16;
  return (z >> shift) & 8191;
}
inline bool decode_pair(unsigned z, int8_t* t) {
  if (z >= legal_pairs) return false;
  unsigned c[2] = {z % 81, z / 81};
  for (int q = 0; q < 2; ++q)
    for (int j = 0; j < 4; ++j, c[q] /= 3) t[4 * q + j] = int(c[q] % 3) - 1;
  return true;
}
inline unsigned rep_index(unsigned c) { return c < 40 ? 40 - c : c - 40; }
inline int rep_sign(unsigned c) { return c < 40 ? -1 : 1; }

template<class T> inline T signed_component(int t, T x) {
  return t == 0 ? T(0) : t > 0 ? x : -x;
}
template<class T> inline T quad_sum(unsigned c, const T* x) {
  T a = signed_component(int(c % 3) - 1, x[0]); c /= 3;
  T b = signed_component(int(c % 3) - 1, x[1]); c /= 3;
  T d = signed_component(int(c % 3) - 1, x[2]); c /= 3;
  T e = signed_component(int(c % 3) - 1, x[3]);
  return ((a + b) + d) + e;
}

#ifdef __CUDACC__
} // namespace quad13
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
namespace quad13 {
enum class Residency { direct, staged, local };

__host__ __device__ inline float e2m1(unsigned c) {
  const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return c & 8 ? -v[c & 7] : v[c & 7];
}
__host__ __device__ inline float local_scale(uint8_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}
__host__ __device__ inline float activation(const uint8_t* a, const uint8_t* s,
                                           const float* g, int nkb, int r, int k) {
  const size_t base = (size_t(r / 64) * nkb + k / 128) * 64 + r % 64;
  const int kk = k % 128, byte = kk / 2;
  const int swizzled = (((byte / 16) ^ ((r % 64 >> 1) & 3)) * 16) + byte % 16;
  unsigned c = (a[base * 64 + swizzled] >> (4 * (kk & 1))) & 15;
  return (e2m1(c) * local_scale(s[base * 8 + kk / 16])) * g[r];
}
__device__ inline float make_rep(unsigned code, const float* x) {
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j, code /= 3) {
    const int t = int(code % 3) - 1;
    v[j] = t == 0 ? 0.f : t > 0 ? x[j] : -x[j];
  }
  return ((v[0] + v[1]) + v[2]) + v[3];
}
__device__ inline void build_quad(float* dst, int stride, const uint8_t* a,
                                  const uint8_t* s, const float* g,
                                  int nkb, int r, int kb, int q) {
  float x[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = activation(a, s, g, nkb, r, kb * 128 + 4 * q + j);
  dst[0] = 0.f;
  for (int i = 1; i < representatives; ++i) dst[i] = make_rep(40 + i, x);
  for (int i = representatives; i < stride; ++i) dst[i] = 0.f;
}
__global__ void prepare_tables(const uint8_t* a, const uint8_t* s, const float* g,
                               int nkb, int rows, int stride, float* table) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= rows * nkb * 32) return;
  int q = i % 32, kb = (i / 32) % nkb, r = i / (32 * nkb);
  build_quad(table + size_t(i) * stride, stride, a, s, g, nkb, r, kb, q);
}
__device__ inline float oriented(const float* table, unsigned c) {
  unsigned idx = c < 40 ? 40 - c : c - 40;
  float v = table[idx];
  return __uint_as_float(__float_as_uint(v) ^ (c < 40 ? 0x80000000u : 0u));
}

// Byte planes are [K/128][26][N]; every row still owns precisely 208 bits.
// Streaming the reservoir reads each of those 26 bytes once, not 2-3 bytes/pair.
template<int R, Residency MODE>
__global__ void contract(const uint8_t* w, const uint16_t* s, int N, int nkb,
                         const uint8_t* a, const uint8_t* as, const float* ag,
                         const float* global_table, int stride, float* partial) {
  extern __shared__ float shared[];
  int n = int(blockIdx.x * blockDim.x + threadIdx.x);
  int split = int(blockIdx.y);
  int kb0 = nkb * split / int(gridDim.y), kb1 = nkb * (split + 1) / int(gridDim.y);
  float acc[R] = {};
  for (int kb = kb0; kb < kb1; ++kb) {
    if constexpr (MODE == Residency::local) {
      for (int i = threadIdx.x; i < R * 32; i += blockDim.x)
        build_quad(shared + i * stride, stride, a, as, ag, nkb, i / 32, kb, i % 32);
      __syncthreads();
    } else if constexpr (MODE == Residency::staged) {
      for (int i = threadIdx.x; i < R * 32 * stride; i += blockDim.x) {
        int r = i / (32 * stride), offset = i % (32 * stride);
        shared[i] = global_table[(size_t(r) * nkb + kb) * 32 * stride + offset];
      }
      __syncthreads();
    }
    if (n < N) {
      float dot[R] = {};
      unsigned reservoir = 0;
      int bits = 0, byte = 0;
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        while (bits < 13) {
          reservoir |= unsigned(w[(size_t(kb) * 26 + byte) * N + n]) << bits;
          ++byte; bits += 8;
        }
        const unsigned z = reservoir & 8191;
        reservoir >>= 13; bits -= 13;
        const unsigned c0 = z % 81, c1 = z / 81;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const float* table;
          if constexpr (MODE == Residency::direct)
            table = global_table + ((size_t(r) * nkb + kb) * 32 + 2 * p) * stride;
          else table = shared + (r * 32 + 2 * p) * stride;
          dot[r] += oriented(table, c0);
          dot[r] += oriented(table + stride, c1);
        }
      }
      float ws = __half2float(__ushort_as_half(s[size_t(kb) * N + n]));
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r] = fmaf(dot[r], ws, acc[r]);
    }
    if constexpr (MODE != Residency::direct) __syncthreads();
  }
  if (n < N)
#pragma unroll
    for (int r = 0; r < R; ++r) partial[(size_t(split) * R + r) * N + n] = acc[r];
}
__global__ void reduce(const float* partial, float* out, int count, int splits) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  float sum = 0;
  for (int s = 0; s < splits; ++s) sum += partial[size_t(s) * count + i];
  out[i] = sum;
}
#endif
} // namespace quad13
