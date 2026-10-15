#pragma once
#include "gdn.cuh"

namespace halo6000 {
// One slot, rows chronological timesteps. All row strides count F32 elements.
// Base weights/state/history have exactly the serial GdnView layout.
struct GdnPrefillView {
    GdnView base;
    int rows;
    int64_t qkv_stride = ConvChannels, z_stride = Inner;
    int64_t alpha_stride = ValueHeads, beta_stride = ValueHeads, out_stride = Inner;
};
struct GdnPrefillScratch {
    GdnScratch base;
    int64_t qkv_stride = ConvChannels, gate_stride = ValueHeads, core_stride = Inner;
};

// Raw time<0 denotes one of the three immutable initial history entries.
__device__ __forceinline__ float gdn_prefill_raw(const GdnPrefillView &v, int time, int channel) {
    return time < 0 ? v.base.history[channel * 3 + time + 3]
                    : v.base.qkv[int64_t(time) * v.qkv_stride + channel];
}

// Reads history only. It must not alias any destination, including scratch.
// Warp owns one128-element head at one timestep; exactly the serial FMA/XOR tree.
__device__ __forceinline__ void gdn_prefill_prepare(const GdnPrefillView &v, const GdnPrefillScratch &s) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    constexpr int Heads = ConvChannels / StateDim;
    for (int unit = warp; unit < v.rows * Heads; unit += stride) {
        const int time = unit / Heads, head = unit % Heads;
        float val[StateDim / 32];
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) {
            const int c = head * StateDim + lane + r * 32;
            const float *w = v.base.conv_weight + c * ConvTaps;
            float y = fmaf(gdn_prefill_raw(v,time-3,c), w[0], 0.f);
            y = fmaf(gdn_prefill_raw(v,time-2,c), w[1], y);
            y = fmaf(gdn_prefill_raw(v,time-1,c), w[2], y);
            y = fmaf(gdn_prefill_raw(v,time,c), w[3], y);
            val[r] = gdn_silu(y);
            sum = fmaf(val[r], val[r], sum);
        }
        float gain = 1.f;
        if (head < 2 * KeyHeads)
            gain = rsqrtf(fmaxf(gdn_warp_sum(sum), NormEpsilon * NormEpsilon));
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r)
            s.base.qkv[int64_t(time) * s.qkv_stride + head * StateDim + lane + r * 32] = val[r] * gain;
    }
    for (int unit = int(blockIdx.x * blockDim.x + threadIdx.x); unit < v.rows * ValueHeads;
         unit += int(gridDim.x * blockDim.x)) {
        const int time = unit / ValueHeads, head = unit % ValueHeads;
        const int64_t o = int64_t(time) * s.gate_stride + head;
        float x = v.base.alpha[int64_t(time) * v.alpha_stride + head] + v.base.dt[head];
        float softplus = x > 20.f ? x : logf(1.f + expf(x));
        float log_decay = v.base.a[head] * softplus;
        s.base.decay[o] = expf(log_decay);
        s.base.beta[o] = 1.f / (1.f + expf(-v.base.beta[int64_t(time) * v.beta_stride + head]));
        if (s.base.log_decay) s.base.log_decay[o] = log_decay;
        if (s.base.softplus) s.base.softplus[o] = softplus;
    }
}

// Call only after prepare's global barrier. No internal barriers/launches.
// History is independent of recurrence and can commit in the recurrence epoch.
// Handles rows1/2 by retaining the appropriate tail of initial history.
__device__ __forceinline__ void gdn_prefill_commit_history(const GdnPrefillView &v) {
    for (int c = int(blockIdx.x * blockDim.x + threadIdx.x); c < ConvChannels;
         c += int(gridDim.x * blockDim.x)) {
        float h0 = gdn_prefill_raw(v,v.rows-3,c);
        float h1 = gdn_prefill_raw(v,v.rows-2,c);
        float h2 = gdn_prefill_raw(v,v.rows-1,c);
        v.base.history[c * 3] = h0;
        v.base.history[c * 3 + 1] = h1;
        v.base.history[c * 3 + 2] = h2;
    }
}

// Warp owns a value column for all timesteps, loading/storing state only once.
// Keeping state in F32 registers changes storage traffic, NOT arithmetic order.
__device__ __forceinline__ void gdn_prefill_recur(const GdnPrefillView &v, const GdnPrefillScratch &s) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    for (int unit = warp; unit < ValueHeads * ValueDim; unit += stride) {
        const int head = unit / ValueDim, kh = head % KeyHeads;
        float *state = v.base.state + unit * StateDim;
        float st[StateDim / 32];
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) st[r] = state[lane + r * 32];
        for (int time = 0; time < v.rows; ++time) {
            const float *q = s.base.qkv + int64_t(time) * s.qkv_stride + kh * StateDim;
            const float *k = q + KeyHeads * StateDim;
            float qr[StateDim / 32], kr[StateDim / 32];
            float kv = 0.f;
#pragma unroll
            for (int r = 0; r < StateDim / 32; ++r) {
                qr[r] = q[lane + r * 32]; kr[r] = k[lane + r * 32];
                kv = fmaf(st[r], kr[r], kv);
            }
            const int64_t go = int64_t(time) * s.gate_stride + head;
            const float decay = s.base.decay[go];
            float delta = fmaf(-decay, gdn_warp_sum(kv),
                s.base.qkv[int64_t(time) * s.qkv_stride + 2 * KeyHeads * StateDim + unit]) * s.base.beta[go];
            float out = 0.f;
#pragma unroll
            for (int r = 0; r < StateDim / 32; ++r) {
                st[r] = fmaf(kr[r], delta, decay * st[r]);
                out = fmaf(st[r], qr[r], out);
            }
            out = gdn_warp_sum(out);
            if (lane == 0) s.base.core[int64_t(time) * s.core_stride + unit] = out * 0.08838834764831845f;
        }
#pragma unroll
        for (int r = 0; r < StateDim / 32; ++r) state[lane + r * 32] = st[r];
    }
}

// Q8 group32, unrounded divisor, half-rounded effective F32 scale. Output is
// unrotated; any output-writer rotation is a later caller-owned phase.
template<bool Quantize>
__device__ __forceinline__ void gdn_prefill_finish_impl(const GdnPrefillView &v, const GdnPrefillScratch &s, ActivationBuffer a) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int stride = int(gridDim.x * (blockDim.x / 32));
    for (int unit = warp; unit < v.rows * ValueHeads; unit += stride) {
        const int time = unit / ValueHeads, head = unit % ValueHeads;
        float core[ValueDim / 32];
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < ValueDim / 32; ++r) {
            core[r] = s.base.core[int64_t(time) * s.core_stride + head * ValueDim + lane + r * 32];
            sum = fmaf(core[r], core[r], sum);
        }
        float gain = rsqrtf(gdn_warp_sum(sum) / ValueDim + NormEpsilon);
#pragma unroll
        for (int r = 0; r < ValueDim / 32; ++r) {
            int col = lane + r * 32, i = head * ValueDim + col;
            float normalized = gain * core[r] * v.base.norm[col];
            float value = gdn_silu(v.base.z[int64_t(time) * v.z_stride + i]) * normalized;
            v.base.out[int64_t(time) * v.out_stride + i] = value;
            if constexpr (Quantize) {
                float amax = warp_max(fabsf(value));
                float d = amax / 127.f;
                a.qs[int64_t(time) * a.q_stride + i] = amax == 0.f ? 0 : int8_t(roundf(value / d));
                if (lane == 0) a.scale[int64_t(time) * a.scale_stride + head * (ValueDim / 32) + r] = __half2float(__float2half(d));
            }
        }
    }
}
__device__ __forceinline__ void gdn_prefill_finish(const GdnPrefillView &v, const GdnPrefillScratch &s) {
    gdn_prefill_finish_impl<false>(v,s,ActivationBuffer{});
}
__device__ __forceinline__ void gdn_prefill_finish_q8(const GdnPrefillView &v, const GdnPrefillScratch &s, ActivationBuffer a) {
    gdn_prefill_finish_impl<true>(v,s,a);
}
} // namespace halo6000
