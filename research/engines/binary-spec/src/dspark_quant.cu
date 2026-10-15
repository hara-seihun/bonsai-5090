#include "dspark_quant.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <climits>
#include <cstddef>

namespace h5::dspark {
namespace {

__device__ __forceinline__ float bf16_value(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ float e4m3_value(uint8_t bits) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3)));
}

__device__ __forceinline__ uint32_t fp4_code(float value) {
    const float a = fabsf(value);
    const uint32_t c = a < .25f ? 0 : a < .75f ? 1 : a < 1.25f ? 2 :
                       a < 1.75f ? 3 : a < 2.5f ? 4 : a < 3.5f ? 5 :
                       a < 5.f ? 6 : 7;
    return c | (value < 0.f && c ? 8u : 0u);
}

__global__ void prepare_rows(const uint16_t* input, int K, float* factors,
                             QuantRowStatus* status) {
    __shared__ float warp_max[8];
    __shared__ unsigned warp_invalid[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const uint16_t* src = input + static_cast<size_t>(row) * K;
    float maximum = 0.f;
    unsigned invalid = 0;
    for (size_t k = threadIdx.x; k < static_cast<size_t>(K); k += 256) {
        const uint16_t bits = src[k];
        if ((bits & 0x7f80u) == 0x7f80u) invalid = 1;
        else maximum = fmaxf(maximum, fabsf(bf16_value(bits)));
    }
    for (int delta = 16; delta; delta >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
        invalid |= __shfl_down_sync(0xffffffffu, invalid, delta);
    }
    if (lane == 0) {
        warp_max[warp] = maximum;
        warp_invalid[warp] = invalid;
    }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 8 ? warp_max[lane] : 0.f;
        invalid = lane < 8 ? warp_invalid[lane] : 0;
        for (int delta = 16; delta; delta >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
            invalid |= __shfl_down_sync(0xffffffffu, invalid, delta);
        }
        if (lane == 0) {
            const float factor = fmaxf(__fdiv_rn(maximum, 2688.f), 1e-30f);
            if (invalid) {
                factors[row] = 0.f;
                status[row] = QuantRowStatus::non_finite_input;
            } else if (!(factor > 0.f) || !isfinite(factor)) {
                factors[row] = 0.f;
                status[row] = QuantRowStatus::range_error;
            } else {
                factors[row] = factor;
                status[row] = QuantRowStatus::ready;
            }
        }
    }
}

__device__ __forceinline__ size_t wf1_word_offset(int row, int k) {
    const int gid = row & 7;
    const int tq = (k >> 3) & 3;
    const int j = ((k >> 5) & 1) * 2 + ((row >> 3) & 1);
    const int w = (row >> 4) * 8 + (k >> 6) * 4 + j;
    const int lane = gid * 4 + tq;
    return (w >> 2) * 128 + lane * 4 + (w & 3);
}

__global__ void prepare_chunks(const uint16_t* input, int NG, int K,
                               uint8_t* weights, uint8_t* scales,
                               const float* factors, QuantRowStatus* status) {
    const int g = blockIdx.x % NG;
    const int kb = blockIdx.x / NG;
    const int local_row = threadIdx.x >> 3;
    const int block16 = threadIdx.x & 7;
    const int row = g * 32 + local_row;
    const size_t chunk = blockIdx.x;
    uint32_t* out = reinterpret_cast<uint32_t*>(weights + chunk * 2048);
    uint8_t* scale_out = scales + chunk * 256 + local_row * 8 + block16;
    const int k0 = block16 * 16;
    const float factor = factors[row];
    if (!(factor > 0.f)) {
        *scale_out = 0;
        out[wf1_word_offset(local_row, k0)] = 0;
        out[wf1_word_offset(local_row, k0 + 8)] = 0;
        return;
    }

    const uint4* src = reinterpret_cast<const uint4*>(
        input + static_cast<size_t>(row) * K + kb * 128 + k0);
    const uint4 lo = src[0], hi = src[1];
    const uint32_t pairs[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    float values[16];
    float maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        values[i * 2] = bf16_value(static_cast<uint16_t>(pairs[i]));
        values[i * 2 + 1] = bf16_value(static_cast<uint16_t>(pairs[i] >> 16));
        maximum = fmaxf(maximum, fabsf(values[i * 2]));
        maximum = fmaxf(maximum, fabsf(values[i * 2 + 1]));
    }
    const float target = __fdiv_rn(__fdiv_rn(maximum, 6.f), factor);
    uint8_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
    if (e4m3_value(sb) < target && sb < 0x7e) ++sb;
    const float scale = __fmul_rn(e4m3_value(sb), factor);
    bool bad = !isfinite(target) || !isfinite(scale) || target < 0.f || scale < 0.f;
    uint32_t words[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float normalized = scale > 0.f ? __fdiv_rn(values[i], scale) : 0.f;
        bad |= !isfinite(normalized);
        words[i >> 3] |= fp4_code(normalized) << ((i & 7) * 4);
    }
    if (bad) {
        atomicMax(reinterpret_cast<unsigned*>(status + row),
                  static_cast<unsigned>(QuantRowStatus::range_error));
        *scale_out = 0;
        words[0] = words[1] = 0;
    } else {
        *scale_out = sb;
    }
    out[wf1_word_offset(local_row, k0)] = words[0];
    out[wf1_word_offset(local_row, k0 + 8)] = words[1];
}

struct ActivationArgs {
    const float* input;
    uint8_t *q, *s;
    float* g;
    ActivationQuantError* error;
    const uint8_t *ref_q, *ref_s;
    const float* ref_g;
    int rows, width;
};

__device__ __forceinline__ float activation_value(const ActivationArgs& a, int row, int feature,
                                                 unsigned& error) {
    float value = a.input[static_cast<size_t>(row) * a.width + feature];
    if (!isfinite(value)) error = static_cast<unsigned>(ActivationQuantError::non_finite_input);
    if (a.ref_q) {
        const int rr = row & 63, o = feature & 127, seg = o >> 5;
        const size_t base = (static_cast<size_t>(row >> 6) * (a.width / 128) + (feature >> 7)) * 64 + rr;
        const uint8_t packed = a.ref_q[base * 64 + ((seg ^ ((rr >> 1) & 3)) << 4) + ((o & 31) >> 1)];
        const unsigned code = (packed >> (4 * (o & 1))) & 15, m = code & 7;
        float f = m < 4 ? .5f * m : m == 4 ? 2.f : m == 5 ? 3.f : m == 6 ? 4.f : 6.f;
        if (code & 8) f = -f;
        const uint8_t sb = a.ref_s[base * 8 + (o >> 4)];
        const float rg = a.ref_g[row];
        if (!(rg > 0.f) || !isfinite(rg) || sb > 0x7e)
            error = static_cast<unsigned>(ActivationQuantError::invalid_reference);
        const float scale = rg * e4m3_value(sb);
        value -= f * scale;
        if (!isfinite(value)) error = static_cast<unsigned>(ActivationQuantError::range_error);
    }
    return value;
}

__global__ void activation_rows(ActivationArgs a) {
    __shared__ float warp_max[8];
    __shared__ unsigned warp_error[8];
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float maximum = 0.f;
    unsigned error = 0;
    for (size_t k = threadIdx.x; k < static_cast<size_t>(a.width); k += 256) {
        unsigned element_error = 0;
        const float value = activation_value(a, row, static_cast<int>(k), element_error);
        error = max(error, element_error);
        maximum = fmaxf(maximum, fabsf(value));
    }
    for (int delta = 16; delta; delta >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
        error = max(error, __shfl_down_sync(0xffffffffu, error, delta));
    }
    if (!lane) { warp_max[warp] = maximum; warp_error[warp] = error; }
    __syncthreads();
    if (!warp) {
        maximum = lane < 8 ? warp_max[lane] : 0.f;
        error = lane < 8 ? warp_error[lane] : 0;
        for (int delta = 16; delta; delta >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
            error = max(error, __shfl_down_sync(0xffffffffu, error, delta));
        }
        if (!lane) {
            const float g = fmaxf(__fdiv_rn(maximum, 2688.f), 1e-30f);
            if (!isfinite(g) || !(g > 0.f)) error = static_cast<unsigned>(ActivationQuantError::range_error);
            a.g[row] = error ? 0.f : g;
            if (error) atomicMax(reinterpret_cast<unsigned*>(a.error), error);
        }
    }
}

__global__ void activation_blocks(ActivationArgs a, size_t blocks16) {
    const size_t block16 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block16 >= blocks16) return;
    const int row = block16 / (a.width / 16);
    const int feature = (block16 % (a.width / 16)) * 16;
    const int rr = row & 63, o = feature & 127;
    const size_t base = (static_cast<size_t>(row >> 6) * (a.width / 128) + (feature >> 7)) * 64 + rr;
    uint32_t* dst = reinterpret_cast<uint32_t*>(a.q + base * 64 +
        (((o >> 5) ^ ((rr >> 1) & 3)) << 4) + ((o & 31) >> 1));
    uint8_t* scale_out = a.s + base * 8 + (o >> 4);
    if (row >= a.rows || !(a.g[row] > 0.f)) {
        dst[0] = dst[1] = 0; *scale_out = 0; return;
    }
    float values[16], maximum = 0.f;
    unsigned error = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = activation_value(a, row, feature + i, error);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    const float ig = __fdiv_rn(1.f, a.g[row]);
    const float target = __fdiv_rn(maximum, 6.f) * ig;
    uint8_t sb = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
    float sv = e4m3_value(sb);
    if (sv < target && sb < 0x7e) { ++sb; sv = e4m3_value(sb); }
    const float rs = sv > 0.f ? __fdiv_rn(ig, sv) : 0.f;
    uint32_t words[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float lo = values[2*i] * rs, hi = values[2*i+1] * rs;
        if (!isfinite(lo) || !isfinite(hi)) error = static_cast<unsigned>(ActivationQuantError::range_error);
        words[i >> 2] |= static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(
            make_float2(lo, hi), __NV_E2M1, cudaRoundNearest)) << ((i & 3) * 8);
    }
    if (error) {
        atomicMax(reinterpret_cast<unsigned*>(a.error), error);
        dst[0] = dst[1] = 0; *scale_out = 0;
    } else {
        dst[0] = words[0]; dst[1] = words[1]; *scale_out = sb;
    }
}

bool aligned(const void* pointer, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace

cudaError_t launch_prepare_nvfp4(const uint16_t* bf16, int N, int K,
                                uint8_t* weights, uint8_t* scales,
                                float* row_factor, QuantRowStatus* row_status,
                                cudaStream_t stream) {
    static_assert(sizeof(QuantRowStatus) == sizeof(unsigned));
    if (!bf16 || !weights || !scales || !row_factor || !row_status ||
        N <= 0 || K <= 0 || N % 32 || K % 128 ||
        !aligned(bf16, 16) || !aligned(weights, 4) ||
        !aligned(row_factor, 4) || !aligned(row_status, 4))
        return cudaErrorInvalidValue;
    const size_t chunks = static_cast<size_t>(N / 32) * (K / 128);
    if (chunks > INT_MAX) return cudaErrorInvalidValue;
    prepare_rows<<<N, 256, 0, stream>>>(bf16, K, row_factor, row_status);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    prepare_chunks<<<static_cast<unsigned>(chunks), 256, 0, stream>>>(
        bf16, N / 32, K, weights, scales, row_factor, row_status);
    return cudaGetLastError();
}

cudaError_t launch_prepare_activations(const float* input, int R, int K,
                                      uint8_t* q4, uint8_t* s4, float* g4,
                                      ActivationQuantError* device_error,
                                      const uint8_t* ref_q4, const uint8_t* ref_s4,
                                      const float* ref_g4, cudaStream_t stream) {
    const bool any_ref = ref_q4 || ref_s4 || ref_g4;
    if (!input || !q4 || !s4 || !g4 || !device_error || R <= 0 || K <= 0 || K % 128 ||
        !aligned(input, 16) || !aligned(q4, 16) || !aligned(g4, 4) || !aligned(device_error, 4) ||
        (any_ref && (!ref_q4 || !ref_s4 || !ref_g4 || !aligned(ref_g4, 4))) ||
        q4 == ref_q4 || s4 == ref_s4 || g4 == ref_g4)
        return cudaErrorInvalidValue;
    const size_t padded_rows = (static_cast<size_t>(R) + 63) / 64 * 64;
    if (padded_rows > INT_MAX) return cudaErrorInvalidValue;
    const size_t blocks16 = padded_rows * (K / 16);
    const size_t grid = (blocks16 + 255) / 256;
    if (grid > INT_MAX) return cudaErrorInvalidValue;
    ActivationArgs a{input, q4, s4, g4, device_error, ref_q4, ref_s4, ref_g4, R, K};
    activation_rows<<<R, 256, 0, stream>>>(a);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    activation_blocks<<<static_cast<unsigned>(grid), 256, 0, stream>>>(a, blocks16);
    return cudaGetLastError();
}

} // namespace h5::dspark
