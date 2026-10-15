# DSpark BF16 -> WF1 preparation

`src/dspark_quant.{cuh,cu}` converts tight GPU BF16 weights directly into the existing WF1 NVFP4 layout. This is a load-time preparation boundary, not a new inference kernel. The binary RTX5090 campaign owns it; no service or rental is created here.

`h5::dspark::launch_prepare_nvfp4` takes caller-owned input, output buffers, row factors and typed per-row status. All buffers remain alive until stream completion. Only all-ready statuses make the tensor consumable. The caller owns the load synchronization, status copy/inspection and freeing the original BF16 buffer. The operation itself performs two kernel launches, with no allocation or synchronization, and supports CUDA graph capture/replay.

The first kernel reduces each row's maximum and rejects nonfinite BF16. The second loads a K16 block per thread, computes the original CPU E4M3-ceiling/half-up-E2M1 law, and writes the inverse WF1 word mapping. Every finite BF16 magnitude is accepted, including signed zero, subnormals and the largest finite BF16. Explicit round-to-nearest division preserves the original separate divisions; compile with FTZ disabled and without fast-math.

## Bounded build and parity

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC,-O3 --expt-relaxed-constexpr --ftz=false -c src/dspark_quant.cu -o build/dspark_quant.o
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC,-O3 --expt-relaxed-constexpr --ftz=false experiments/binary-dspark-quant/parity.cu build/dspark_quant.o -o build/dspark-quant-parity
build/dspark-quant-parity --cpu-only
# On the campaign parent's admitted GPU keyhole:
build/dspark-quant-parity
```

The N64/K256 oracle retains the original engine's forward layout, independently checks the inverse write bijection, and covers generic signed BF16, zero, wide finite BF16 ranges (including minimum subnormal, maximum finite and signed zero), nonfinite input rows and exact FP4 threshold ties. The GPU invocation captures preparation once, then replays against all four input cases and compares every output weight byte, scale byte, row-factor bit and typed status.

CUDA13.1.115 release compilation and the CPU-only layout/tie checks passed locally on October10. The parent reported weight GPU parity passed in its admitted keyhole. Generated objects and executables are rebuildable under `build/`; delete them after campaign custody without losing source or authoritative GPU results.

## Activation producer

`launch_prepare_activations` owns DSpark's dynamic F32 activation preparation independently of the shared row kernel's thread-count bound. It accepts every positive K128 width, including FC width25600, using fixed256-thread row reduction plus K16 quantization. The layout is the existing padded64-row q4/s4 swizzle, with dynamic g4 per valid row, E4M3 ceiling and native nearest-even FP4 conversion. Padding rows are cleared. Its optional reference triple produces a separate residual plane after decoding/subtracting the first plane, including the existing CUDA fused multiply-subtract arithmetic.

The required `ActivationQuantError` is caller-cleared sticky state across a run. Nonfinite input, invalid reference scales and finite-domain arithmetic failures are explicit errors. Valid paths have no atomics. Buffer capacity, alignment and reference-triple requirements are in the header.

`activation_parity.cu` is a bounded GPU fixture: R4/K5120 first and residual planes compare bitwise with shared `launch_prep`, forced into dynamic row mode; K5120 and K25600 also compare with an independent scalar oracle and test zero/negative rows, padded output and sticky nonfinite errors. Its cached-object link is owned by the parent's admitted GPU gate:

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC,-O3 --expt-relaxed-constexpr --ftz=false -c experiments/binary-dspark-quant/activation_parity.cu -o build/dspark-activation-parity.o
# Link that object and dspark_quant.o with the gate's cached kernels.o/fp4.o.
```

Activation producer and fixture objects compiled locally; activation GPU results remain with the parent's gate.
