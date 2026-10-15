# Native DSpark precision gate

Source `0ffcaea` includes fixed-thread activation producer `382867b`. Native object SHA256 `108e3c065045dbf998de317a5a50172c89793cebf2171df0044249df3ba36848`; preparation object `d3e6584fa7a879ee39a91ea50cb24a58c791e490b8cdbf55bbb6e3c3c34874b2`. Both CUDA13.1.115/O3/C++17/sm120a. Same cached live-attention/kernel/FP4/cuBLAS/Prism link closure for every arm.

Raw custody: `results/binary-5090/dspark-fast-contract-repaired/`, including `contract-inputs.sha256`, activation parity and actual-fork comparison logs. Runtime input root `/workspace/binary/dspark-fast-repaired-input`. The first failed shared-row producer run remains under this experiment's `results/first-contract/`.

## Scope

Synthetic four raw post-block tap rows, anchor198, four-row ingestion and four-query block, candidate vocabulary65536. Native timings include the synchronous run and host proposals/confidence; reset excluded. Two warmups, eight measured runs; reference loaded once. This is a projection/drafter contract, not HTTP or target throughput.

K5120/R4 primary and residual activation preparation matches the existing shared producer byte-for-byte. K25600/R4 matches independent scalar witnesses. Padding, sticky errors and GPU weight-preparation/capture parity pass.

| Precision | Median run ms | Proposal agreement | Base relative RMS | Cosine |
|---|---:|---:|---:|---:|
| BF16 | 2.403822 | 4/4 | 0.002701 | 0.999996 |
| NVFP4 A1 | 1.231374 | 0/4 | 0.303666 | 0.968306 |
| NVFP4 A2 | 2.015746 | 3/4 | 0.262463 | 0.973891 |
| NVFP4 head A2 | 2.253051 | 4/4 | 0.043024 | 0.999925 |

BF16 passes its strict reference gate. Quantized arms have finite numerical reports, not a strict fidelity pass. Only parent-owned actual target verification, acceptance and task panels select a serving arm. No precision default changes.

## Execution-resource custody

The separate memory contract (`d97ab0a` fixture, unchanged `0ffcaea` serving objects) passed with explicit LAZY and EAGER module-loading modes. Raw JSONL, executable pins and selected-kernel resource records are in `results/memory/`.

LAZY A1 allocates3,657,433,088 bytes during construction and releases exactly3,657,433,088 bytes on destruction. Its first FP4 execution separately consumes3,623,878,656 bytes that remain with the CUDA context. Subsequent A2 and head-only modes release their complete construction allocations and add no execution footprint. There is no owned NativeDspark Mat allocation leak in this witness.

EAGER still retains3,550,478,336 bytes on the first FP4 execution, so eager code loading does not remove the persistent execution resource. The reachable owner is the cached FP4 function/execution family: `fp4_get` initializes every configuration/mode/weight-format function with `cudaFuncSetAttribute`; these are static-linked CUDA-context resources rather than NativeDspark allocations. The runtime does not expose their finer allocation decomposition in this fixture. Selected WF1 cfg3 SASS reports123 registers, zero stack and zero local bytes, excluding a spill/stack claim for those selected kernels. No unload or VRAM-reclamation mechanism is claimed; resources leave with the process/context.

Memory-probe numeric values can vary in full quantized split projections because the inherited split path uses FP32 atomic accumulation. The retained selection gate and macro task oracle remain separate.
