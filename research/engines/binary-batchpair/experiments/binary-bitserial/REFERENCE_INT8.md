# Original native INT8 projection comparator

`reference_int8.cuh` / `reference_int8.cu` are standalone CUDA sources in namespace
`int8_reference`. They have no engine dependencies and no activation producer.

## Source custody

Origin: binary-5090 commit `524b2ab11af15e587cf0dc0042ad4e0e5502fb5a`.

- `src/kernels.cu` SHA256:
  `ff37c73550795b9f3f28dbfcb1745f23011ae4d57ba57f77416c3bf4530e4312`.
  The section from `constexpr int GM =` through the end of `k_gemm_t2`,
  immediately before `void launch_gemm`, is copied byte-for-byte, including
  cp.async, ldmatrix, and signed INT8 MMA helpers.
- `src/engine.cu` SHA256:
  `73d4dd3630dbd24c435003cf55ba257345335995d5730d25f65069476deaf306`.
  The Q1_0 branch of `decode_sign_block` and fragment layout from `repack_rows`
  supply preparation. The packing loops retain the original indexing and
  `scale / 64.f`. Preparation is now single-threaded, accepts raw host bytes,
  and returns CUDA errors instead of engine exceptions.

The launch wrapper uses the original store specialization, CTA shape, grid,
shared-memory size, and output stride N. It propagates attribute-setting and
launch errors. Per-thread/per-device attribute configuration avoids the original
unchecked global static initialization. N=96 uses the original inactive fourth
32-column group, not a replacement tail kernel.

## Contract

Every API returns `cudaError_t`; inspect it before using outputs.

`sizes(N,K,&requirements)` accepts positive N divisible by 32 and positive K
divisible by 128. With B = N*(K/128), the returned allocation sizes are:

| Field | Bytes | Layout |
|---|---:|---|
| `raw_q1_bytes` | 18*B | Host input, row-major Q1_0 blocks |
| `packed_weight_bytes` | 32*B | Device uint4 fragment-packed signed 2-bit weights |
| `scale_bytes` | 4*B | Device FP32 S[K/128,N], weight block scale / 64 |

Each raw Q1_0 block is two little-endian FP16 scale bytes followed by 16 sign
bytes, LSB-first. A set bit means +1 and a clear bit means -1. Nonfinite weight
scales are rejected. Zero and negative finite scales preserve their supplied
meaning.

`prepare(raw_host,raw_bytes,N,K,&W,&S)` is blocking host preparation plus device
upload. Pass the exact byte count and two distinct output slots initialized to
null. Success transfers independent cudaMalloc allocations W and S to the
caller. Free both after their final stream use. Failed preparation frees any
allocated device buffers; if cudaFree itself fails, the corresponding nonnull
slot remains available to the caller and the cleanup error is returned.

`project(q,sx,W,S,out,N,K,R,1.f,0.f,stream)` accepts R=1..16. q is device INT8
[R,K], sx is device FP32 [R,K/128], and out is device FP32 [R,N]. All other
alpha/beta pairs are explicitly rejected, including NaNs. q, sx, W, S and out
must not alias and must be correctly sized and on the current device; pointer
presence/alignment and scalar/dimension constraints are checked at launch.

For each row r and feature n, the result is the original kernel's accumulation
of signed integer 128-block dots, dequantized with sx[r,kb] and the raw Q1_0
weight scale. Signed 2-bit weight fields are lifted into INT8 ±64, canceled by
S's /64. SX belongs to the benchmark's activation producer; this source neither
quantizes activations nor invents their scaling convention.

Successful project return means the asynchronous launch was accepted, not that
execution completed. Inspect `cudaStreamSynchronize(stream)` or the benchmark's
CUDA event completion result for execution errors. Kernels and API validation
perform no timing, and one-time host downloads/preparation/uploads belong
outside recurring projection timing.

## Build

```sh
nvcc -std=c++17 -arch=sm_120a -O3 -c reference_int8.cu -o reference_int8.o
```

This standalone compile succeeded with CUDA 13.1.115 on a CPU build host. GPU
correctness and paired projection/boundary timings belong to the benchmark
coordinator; this extraction does not supply those results.
