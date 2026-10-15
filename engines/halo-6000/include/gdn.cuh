#pragma once
#include "model.hpp"
#include "device.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace halo6000 {
struct GdnView {
    const float *qkv;
    const float *z;
    const float *alpha;
    const float *beta;
    const float *conv_weight;
    const float *a;
    const float *dt;
    const float *norm;
    float *state;
    float *history;
    float *out;
};
struct GdnScratch {
    float *qkv;
    float *decay;
    float *beta;
    float *core;
    float *log_decay = nullptr;
    float *softplus = nullptr;
};

__device__ __forceinline__ float gdn_warp_sum(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
    return x;
}
__device__ __forceinline__ float gdn_silu(float x) { return x / (1.f + expf(-x)); }

// All phases use a 1D grid and blocks whose thread count is a multiple of32.
// Caller owns a global barrier between phases; there are no hidden launches.
__device__ __forceinline__ void gdn_conv(const GdnView &v, const GdnScratch &s) {
    for (int c = int(blockIdx.x * blockDim.x + threadIdx.x); c < ConvChannels;
         c += int(gridDim.x * blockDim.x)) {
        float *h = v.history + c * (ConvTaps - 1);
        const float *w = v.conv_weight + c * ConvTaps;
        float x = v.qkv[c];
        float y = 0.f;
        y = fmaf(h[0], w[0], y);
        y = fmaf(h[1], w[1], y);
        y = fmaf(h[2], w[2], y);
        y = fmaf(x, w[3], y);
        s.qkv[c] = gdn_silu(y);
        h[0] = h[1]; h[1] = h[2]; h[2] = x;
    }
}

__device__ __forceinline__ void gdn_prepare(const GdnView &v, const GdnScratch &s) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    for (int h = warp; h < 2 * KeyHeads; h += stride) {
        float *x = s.qkv + h * StateDim;
        float val[StateDim / 32];
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) {
            val[r] = x[lane + r * 32];
            sum = fmaf(val[r], val[r], sum);
        }
        float gain = rsqrtf(fmaxf(gdn_warp_sum(sum), NormEpsilon * NormEpsilon));
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) x[lane + r * 32] = val[r] * gain;
    }
    for (int h = int(blockIdx.x * blockDim.x + threadIdx.x); h < ValueHeads;
         h += int(gridDim.x * blockDim.x)) {
        float x = v.alpha[h] + v.dt[h];
        float softplus = x > 20.f ? x : logf(1.f + expf(x));
        float log_decay = v.a[h] * softplus;
        s.decay[h] = expf(log_decay);
        s.beta[h] = 1.f / (1.f + expf(-v.beta[h]));
        if (s.log_decay) s.log_decay[h] = log_decay;
        if (s.softplus) s.softplus[h] = softplus;
    }
}

// A warp owns all128 channels of one Q/K/V head; Q/K norm consumes its producer
// registers directly, so no global conv→prepare barrier or scratch reload is needed.
__device__ __forceinline__ void gdn_conv_prepare(const GdnView &v, const GdnScratch &s) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    for (int unit = warp; unit < ConvChannels / StateDim; unit += stride) {
        float val[StateDim / 32];
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) {
            int c = unit * StateDim + lane + r * 32;
            float *h = v.history + c * 3;
            const float *w = v.conv_weight + c * ConvTaps;
            float x = v.qkv[c];
            float y = fmaf(h[0], w[0], 0.f);
            y = fmaf(h[1], w[1], y);
            y = fmaf(h[2], w[2], y);
            y = fmaf(x, w[3], y);
            val[r] = gdn_silu(y);
            sum = fmaf(val[r], val[r], sum);
            h[0] = h[1]; h[1] = h[2]; h[2] = x;
        }
        float gain = 1.f;
        if (unit < 2 * KeyHeads)
            gain = rsqrtf(fmaxf(gdn_warp_sum(sum), NormEpsilon * NormEpsilon));
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r)
            s.qkv[unit * StateDim + lane + r * 32] = val[r] * gain;
    }
    for (int h = int(blockIdx.x * blockDim.x + threadIdx.x); h < ValueHeads;
         h += int(gridDim.x * blockDim.x)) {
        float x = v.alpha[h] + v.dt[h];
        float softplus = x > 20.f ? x : logf(1.f + expf(x));
        float log_decay = v.a[h] * softplus;
        s.decay[h] = expf(log_decay);
        s.beta[h] = 1.f / (1.f + expf(-v.beta[h]));
        if (s.log_decay) s.log_decay[h] = log_decay;
        if (s.softplus) s.softplus[h] = softplus;
    }
}

__device__ __forceinline__ void gdn_recur(const GdnView &v, const GdnScratch &s) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    for (int unit = warp; unit < ValueHeads * ValueDim; unit += stride) {
        const int h = unit / ValueDim;
        const int kh = h % KeyHeads; // GGUF converter stores V heads in tiled order.
        const float *q = s.qkv + kh * StateDim;
        const float *k = s.qkv + KeyHeads * StateDim + kh * StateDim;
        float *state = v.state + unit * StateDim; // [value_head][value_col][key_row]
        float st[StateDim / 32], qr[StateDim / 32], kr[StateDim / 32];
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) {
            int i = lane + r * 32;
            st[r] = state[i]; qr[r] = q[i]; kr[r] = k[i];
            kv = fmaf(st[r], kr[r], kv);
        }
        const float decay = s.decay[h];
        float delta = fmaf(-decay, gdn_warp_sum(kv), s.qkv[2 * KeyHeads * StateDim + unit]) * s.beta[h];
        float out = 0.f;
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) {
            st[r] = fmaf(kr[r], delta, decay * st[r]);
            state[lane + r * 32] = st[r];
            out = fmaf(st[r], qr[r], out);
        }
        out = gdn_warp_sum(out);
        if (lane == 0) s.core[unit] = out * 0.08838834764831845f;
    }
}

template<bool Quantize>
__device__ __forceinline__ void gdn_finish_impl(const GdnView &v, const GdnScratch &s, ActivationBuffer a) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    for (int h = warp; h < ValueHeads; h += stride) {
        float core[ValueDim / 32];
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < ValueDim / 32; ++r) {
            core[r] = s.core[h * ValueDim + lane + r * 32];
            sum = fmaf(core[r], core[r], sum);
        }
        float gain = rsqrtf(gdn_warp_sum(sum) / ValueDim + NormEpsilon);
#pragma unroll
        for (int r = 0; r < ValueDim / 32; ++r) {
            int col = lane + r * 32;
            int i = h * ValueDim + col;
            float normalized = gain * core[r] * v.norm[col];
            float value = gdn_silu(v.z[i]) * normalized;
            v.out[i] = value;
            if constexpr (Quantize) {
                float amax = warp_max(fabsf(value));
                float d = amax / 127.f;
                a.qs[i] = amax == 0.f ? 0 : int8_t(roundf(value / d));
                if (lane == 0) a.scale[h * (ValueDim / 32) + r] = __half2float(__float2half(d));
            }
        }
    }
}
__device__ __forceinline__ void gdn_finish(const GdnView &v, const GdnScratch &s) {
    gdn_finish_impl<false>(v, s, ActivationBuffer{});
}
// One row, k=4096, group32: divisor remains unrounded; only effective scale is FP16-rounded.
__device__ __forceinline__ void gdn_finish_q8(const GdnView &v, const GdnScratch &s, ActivationBuffer a) {
    gdn_finish_impl<true>(v, s, a);
}
} // namespace halo6000
