#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace h5::dspark {

enum class QuantRowStatus : uint32_t {
    ready = 0,
    non_finite_input = 1,
    range_error = 2
};

// Tight device BF16[N,K] -> fp4.cu WF1 chunks, ordered [K/128,N/32].
// N and K must be positive multiples of 32 and 128 respectively. Required,
// mutually disjoint outputs: weights[N*K/2], scales[N*K/16], row_factor[N],
// row_status[N]. BF16 must be 16-byte aligned; weights, factors and status
// 4-byte aligned. No allocation, synchronization or status initialization is
// needed. Capture-safe. Launch errors return directly; inspect every row_status
// after stream completion before using weights. Only all-ready is a valid tensor.
// Nonfinite input sets non_finite_input and clears its row. Arithmetic outside
// the finite scale/code domain sets range_error; that row must not be consumed.
// Conversion uses the engine.cu repack_nvfp4 law, including its half-up FP4
// thresholds and E4M3 ceiling. Compile without FTZ/fast-math.
cudaError_t launch_prepare_nvfp4(const uint16_t* bf16, int N, int K,
                                uint8_t* weights, uint8_t* scales,
                                float* row_factor, QuantRowStatus* row_status,
                                cudaStream_t stream);

enum class ActivationQuantError : uint32_t {
    success = 0, non_finite_input = 1, invalid_reference = 2, range_error = 3
};

// Tight F32[R,K] -> existing q4/s4/g4 activation layout, with dynamic row g.
// R>0, K>0 and K%128==0. q4 and s4 hold ceil(R/64)*64*K/2 and /16
// bytes; padding rows are cleared. g4 holds R floats. input and q4 require
// 16-byte alignment, s4 byte alignment, g4/error 4-byte alignment. Buffers
// are disjoint, including an optional separate reference plane. Reference
// q/s/g must be all null or all set; set computes input - dequant(reference)
// before both row reduction and quantization, as P_FP4_RESIDUAL does.
// device_error is mandatory sticky state: caller clears it to success once
// per sequence and inspects after completion. Valid paths perform no atomics.
// No allocation/sync/global-config/Hadamard; capture-safe. Launch errors return.
cudaError_t launch_prepare_activations(const float* input, int R, int K,
                                      uint8_t* q4, uint8_t* s4, float* g4,
                                      ActivationQuantError* device_error,
                                      const uint8_t* ref_q4, const uint8_t* ref_s4,
                                      const float* ref_g4, cudaStream_t stream);

} // namespace h5::dspark
