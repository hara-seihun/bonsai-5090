# Direct packed Q1_0 projection on RTX 5090

Owner: binary Bonsai27B campaign. Kernel/API commit **8964df6**; parent owns engine integration and complete-model PPL/timing. This directory contains standalone projection measurements, not a model throughput result. The [bounded LUT8/LUT4-ILP tuning round](TUNING.md) lost and leaves the original selection unchanged.

## Result

Full-precision FP32 activations and original FP16 K128 scales, with no dense weight decoding. Times below are **microseconds for the whole R-token projection**, including recurring table production, split-K reduction, output writes and alpha/beta continuation. CUDA graph, median9 after3 warmups; each cold trial first evicts L2 using a128MiB buffer outside the timed interval.

| N × K | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5120 × 5120 | 8.19 | 9.98 | 12.26 | 12.26 | 14.30 | 16.35 | 16.35 | 18.18 |
| 17408 × 5120 | 16.35 | 18.40 | 22.50 | 24.32 | 28.64 | 32.74 | 34.53 | 38.62 |
| 5120 × 17408 | 16.13 | 18.14 | 22.53 | 24.58 | 30.43 | 34.53 | 36.58 | 40.54 |
| 248320 × 5120 | 116.74 | 135.17 | 182.27 | 219.14 | 272.38 | 319.49 | 368.64 | 415.74 |
| 96 × 5120 | 4.06 | 6.11 | 5.86 | 5.86 | 6.11 | 5.89 | 6.11 | 5.86 |

Other native engine shapes, R1/R8: N16384 K5120 **14.34/36.83us**; N14336 K5120 **14.30/34.56us**; N34816 K5120 **24.54/67.30us**; N5120 K6144 **10.21/20.19us**. Their intermediate R selections use the measured neighboring LUT-family schedule, not a separate sweep for every R.

The head R8 route shares weight loads across all eight tokens:415.74us versus8×116.74=933.89us for eight R1 calls (2.25× relative to repeated projection calls). This comparison is between projection schedules, not a complete-model speedup or vendor baseline.

### Numerical checks

Weights are deterministic synthetic Q1_0 blocks with random signs and exactly representable positive scales between1/128 and127/128. Activations are deterministic generic FP32 values. They exercise the model's actual packing and dimensions; they are **not extracted trained-model weights**.

All output elements were compared with an independent GPU reference reading original Q1_0 blocks and doing FP32 sign-dot accumulation. Among selected configurations across the five shapes and R1..8:

- Maximum absolute difference: **1.06811523e-4**.
- Maximum relative RMS difference: **2.97128353e-7**.
- Maximum absolute error versus FP64 mathematical projection on64 distributed rows/token: **3.055487e-5**.
- Alpha=.75,beta=.25 with initial Y equal to reference projection was exercised for all shapes/R, including N96 and the head. Reported errors were near roundoff. The current harness additionally provides an independent initial-Y check for subsequent runs.

There is no activation/scale quantization in this route. FP32 addition order differs from the reference; bitwise equivalence is not the contract.

## Representation and recurring bill

Original block:18bytes, FP16 scale first,16 sign bytes, LSB-first; bit1 means +scale. Preparation transposes to signs `[K/128][4][N]` (32-bit words), scales `[K/128][N]` (FP16). Prepared payload stays exactly **1.125bits/weight**, with no padding or dense sign matrix. Preparation is a one-time load transform; its host-synchronized timing is included in each JSON receipt but excluded from recurring projection latency.

Three explicit projection routes:

- `Lut4`: producer creates16 FP32 signed sums for each input quartet. Nibble addresses in the original sign words select dot contributions directly. R2..8 projection CTAs share each loaded sign word and scale across all token accumulators. Each K128 group applies its own original scale.
- `Sign`: one thread/output row selects activation signs by XOR of the FP32 sign bit, using a shared activation group.
- `WarpSign`: one warp/output row accumulates four activations/lane per group, then reduces its final scaled sum. This wins the narrow N96 projection.

LUT storage is16K bytes/token:80KiB atK5120 and272KiB atK17408. Dynamic CTA shared memory is2KiB×R for LUT,512bytes for Sign,0 for WarpSign. Split-K scratch is4×R×N×split_count bytes and includes scale-preserving partials. With one split, the projection writes alpha/beta output directly; otherwise a final reduction performs the continuation. Recurring paths use2 or3 launches for LUT,1 or2 for sign routes. All these stages are inside the timed graph.

Final compiler receipt `results/final-build.log`: LUT R1 uses36 registers/thread; R2..8 use40. Sign39, WarpSign40; **zero spills**. No tensor instructions, activation bit-plane quantization, or full-weight decode is used.

## Engine interface

Header `packed1.cuh`, implementation `packed1.cu`, CUDA runtime errors throughout. No allocation or host synchronization in `project`; stream/capture-safe.

```cpp
packed1::Plan p;
CK(packed1::select(N, K, R, &p));
// Allocate caller-owned signs/scales once; prepare original device Q1 bytes once.
CK(packed1::prepare(raw, signs, scales, N, K, stream));
// X [R,K] and Y [R,N] are tightly packed FP32.
// Allocate scratch using table_bytes(p), partial_bytes(p) after successful select/validate.
CK(packed1::project(p, signs, scales, X, Y, tables, partials,
                    alpha, beta, stream));
```

`select` supports the dimension families above and R1..8; unsupported dimensions/row counts return `cudaErrorInvalidValue` without writing the output plan. Explicit custom plans may use any positive N,K multiple128,R1..8,threads64/128/256 and positive groups_per_split, after `validate`. Size helpers require validated dimensions/plans. Callers retain allocation ownership and propagate every CUDA error. Native worker consumes this as `H5_WEIGHTS=packed1 H5_ACT=f32`; engine/PPL custody is outside this directory.

## Reproduce

Build on CUDA13.1 / sm_120a:

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v packed1.cu tuning.cu bench.cu -o bench
flock /tmp/binary-gpu.lock ./bench 248320 5120 8 sweep head-r8
# Current production selection, with51 warm and51 cold trials:
flock /tmp/binary-gpu.lock ./bench 248320 5120 8 selected selected-head-r8
python3 summarize.py
```

Other benchmark modes: `lut`, `sign`, `warp`, `sweep`; explicit tuning modes `lut8`, `ilp`. Arguments are N K R mode output-prefix. Each JSON line prices one entire route, records warm/cold latency and numerical errors, and includes preparation timing.

Measurements ran October10,2026 around18:33–18:37UTC on the parent-provisioned RTX5090 at `/workspace/binary/packed1`. GPU UUID `GPU-00000000-0000-0000-0000-000000000000`,170SMs, driver595.58.03,CUDA13.1,500W power limit. A subsequent snapshot showed2550MHz; clocks were not locked. No additional rental was provisioned.

`summary.json` pins selected raw receipt filenames and plans. `*-r1-v2.jsonl` are generic-FP32 single-token sweeps; R2..8 receipts use shared-token LUTs. Earlier `*-r1.jsonl` receipts used dyadic activations and are not the headline. Warm timing is resident-matrix timing, not a claim that all model weights remain cached.

Measurements preceded final addition of `select` and removal of the unused templated LUT branch. The final kernel compiled successfully; the selected-plan rerun did not start because the parent held the shared GPU lock for full-model work (`results/final.log` is empty). The original sweep harness alpha/beta test used the reference projection as initial Y. Independent initial-Y alpha/beta subsequently passed in the paired tuning round. Parent now owns complete-model numerical/latency evidence.
