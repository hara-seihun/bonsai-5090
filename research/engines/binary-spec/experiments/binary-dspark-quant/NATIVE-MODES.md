# Native DSpark projection precision

Source `5732320` adds explicit `H5_DSPARK_PRECISION`:

- Unset or `bf16`: unchanged live-attention BF16/cuBLAS checkpoint control.
- `nvfp4-a1`: direct-primal NVFP4 weights and one NVFP4 activation plane for tap FC, ingestion KV, six-layer QKV/O/gate/up/down and own candidate head.
- `nvfp4-a2`: the same weights with a second activation plane formed from original F32 input minus decoded first plane; projections add the two planes before downstream nonlinearities.
- `nvfp4-head-a2`: only the own output head uses NVFP4 weights/two activation planes; tap FC and trunk remain BF16.

Embeddings, Markov A/B, confidence projection, norms, SNR conditioning and mode embeddings remain in their control coordinates. No Hadamard is introduced. Unknown/empty precision values are rejected. Quantized projections require N multiple32/K multiple128; a candidate vocabulary whose count is not N32 is rejected for these modes, not silently rerouted.

The original BF16 projection buffers are temporary load inputs only. `dspark_quant` prepares the current WF1 coordinate on device; row statuses must all be ready before the original buffer is freed. Retained weight bytes include FP4 codes, K16 E4M3 scales and original row factors. Full embeddings and predecessor Markov domain remain resident.

Recurring activation preparation uses the private fixed256-thread DSpark producer, with the existing dynamic row-factor, E4M3 ceiling and native-nearest E2M1 law. It supports any K128 width, including tap FC K25600, without selecting the target's global preparation configuration. A2 subtracts decoded first-plane values into a separate output plane. Required sticky activation errors are cleared once per run and inspected after stream completion; invalid input/reference/range is an explicit failure. The existing WF1 consumer uses fixed cfg3 (F4G2/T32) with explicit split policy: head N>=32768 split1, small N<2048 split32, K>8192 split32, otherwise split8. These are initial coherent schedules, not claimed optima. The split path retains its existing FP32 atomic association. Input/output normalization and residual additions are unchanged.

Static/native method signatures and opaque class size remain unchanged. `NativeDspark::projection_precision()` and `dspark_precision_name()` expose the actual selected precision for serving provenance. Constructor logs also state it.

## Build custody

`build/dspark_native_repaired.o` (CUDA13.1.115, O3/C++17, compute120a→sm120a, PIC, relaxed constexpr) SHA256 `108e3c065045dbf998de317a5a50172c89793cebf2171df0044249df3ba36848`.

`build/dspark_quant.o`, including weight and activation preparation, SHA256 `d3e6584fa7a879ee39a91ea50cb24a58c791e490b8cdbf55bbb6e3c3c34874b2`.

Standalone contract links those two objects plus unchanged live `dspark_kernels.o`, `gguf.o`, cached `kernels.o`, cached `fp4.o`, CUDA runtime, cuBLAS and Prism llama. No giant translation unit rebuild is required. Compile/link the small host fixture on the rental host to avoid glibc version mismatch; repaired prepared input objects are at `/workspace/binary/dspark-fast-repaired-input`. The first failed consumer run and its manifests remain in `results/first-contract/`; weight byte parity and the BF16 control passed before the FC row-preparation launch-bound violation.

The bounded full-checkpoint contract reports synthetic/file tap quality, proposal agreement and warmed synchronous run timings per precision. Quantized fidelity and timing are separate results; nonfinite/geometry/runtime failures are errors, while finite approximation differences remain numerical reports for the campaign parent's acceptance/task gate. The BF16 control retains the actual-fork strict gate. No quantized default or macroquality conclusion is adopted by this document.
