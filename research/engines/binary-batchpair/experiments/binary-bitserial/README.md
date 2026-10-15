# Packed binary × A8 ISA maps

Owner: binary Bonsai27B campaign. Kernel/API `3e79216`, complete-prefill producer `443e27f`, measured fused/pruned scheduling `c7f2736`. Parent owns modern-engine integration, complete-model quality/task deltas and whole-model timing. This directory owns the mathematical map, standalone CUDA implementation and inclusive projection measurements.

## Result and integration decision

**DP4A 0/1 relabeling plus shared DC correction wins at R8/R16. Preserve the original FP32 single-token route.** This is exact integer arithmetic for the given signed-A8 carrier, not an exact FP32 activation route.

Cold microseconds for the whole R-token projection, including recurring production and continuation:

| Trained geometry N×K | R | FP32 LUT4 | Original INT8 | Packed popcount | Packed DP4A |
|---|---:|---:|---:|---:|---:|
|FFN gate17408×5120|1|16.384|59.136|18.400|16.384|
|FFN gate17408×5120|8|38.912|57.344|40.960|28.384|
|FFN gate17408×5120|16|75.520|59.392|88.032|36.864|
|FFN down5120×17408|1|16.384|169.728|18.176|18.176|
|FFN down5120×17408|8|40.928|169.984|43.008|28.640|
|FFN down5120×17408|16|75.744|169.984|88.032|38.912|
|Full head248320×5120|1|116.736|589.568|126.688|118.784|
|Full head248320×5120|8|415.744|591.872|409.568|214.784|
|Full head248320×5120|16|831.488|595.968|991.200|360.448|
|Pruned-head geometry65536×5120|1|55.168|200.224|40.928|38.912|
|Pruned-head geometry65536×5120|8|133.120|200.704|120.800|69.632|
|Pruned-head geometry65536×5120|16|259.840|202.752|280.384|100.352|

Other actual fused geometry, cold FP32-LUT4 / packed-DP4A microseconds:

| Geometry | R1 | R8 | R16 |
|---|---:|---:|---:|
|GDN input16384×5120|16.384 /16.352|36.832 /26.592|73.440 /36.864|
|Attention input14336×5120|14.336 /16.352|34.784 /26.336|65.248 /32.768|
|Gate+up34816×5120|24.608 /26.592|67.552 /42.976|134.912 /63.488|
|Output5120×6144|10.240 /10.272|20.448 /16.096|36.576 /18.432|
|Alpha+beta96×5120|8.192 /8.192|16.384 /7.936|26.624 /7.936|

R15 alpha+beta also passes: DP4A8.192us, tiled FP32 LUT4 26.624us. The5120×5120 gate-row prefix control gives R1/R8/R16 DP4A10.208/14.304/18.400us versus LUT4 10.240/20.224/34.560us.

The R16 FP32 comparator is two existing R8 calls; R15 uses8+7. LUT4 is explicitly selected for the comparator, not the original narrow-matrix WarpSign winner. Original INT8 is the unchanged `524b2ab` GEMM with its fixed128-token tile and the same Q8_128 producer; it is not a newly tuned INT8 small-M kernel. [REFERENCE_INT8.md](REFERENCE_INT8.md) pins the exact extracted source.

The full head R8 gain is1.94× versus FP32 LUT4 and2.76× versus that original INT8 launcher. Full head R16 is2.31× versus tiled LUT4 and1.65× versus original INT8. These are projection gains, not model throughput ratios. The explicit modern-engine modes `H5_ACT=a8` and `H5_ACT=f32-r1-a8` are owned by the integration worker; default FP32 remains unchanged. Batched integration pays at the projection boundary; acceptance of its whole-model quality and timing belongs to the parent.

## The matched map and relabeling

Original weights have only signs: bit b=1 means+1 and b=0 means−1. For a fixed quantized activation block q, define the selected-positive label

```
P(b,q) = sum_i b_i q_i
DC(q)  = sum_i q_i
signedDot(b,q) = 2 P(b,q) - DC(q).
```

For each fixed q, this affine continuation is injective in P and therefore has exactly the signed-dot map's fibers on reachable labels. Across different q, the carrier includes DC; P alone is not claimed sufficient. DC is produced once and shared across output rows. It is applied before each row's original K128 scale, not globally moved past varying weight scales.

### DP4A composition

One sign nibble n is relabeled directly into four signed-byte **0/1** coefficients:

```
expand(n) = (n * 0x00204081u) & 0x01010101u
byte_i(expand(n)) = bit_i(n), i=0..3
selected += IDP.4A.S8.S8(expand(n), packed_signed_q_bytes)
dot = 2*selected - DC.
```

This is an IMAD/AND operand composition followed by native signed DP4A and an affine continuation. It never forms a ±1 weight tensor. Expanded register operands are ephemeral and their recurring work is timed. All tokens in a CTA share each loaded original sign word. Four independent selected-sum accumulators per token shorten the recurrence.

### Bitserial population-count composition

Represent signed q in two's-complement planes with coefficients `(1,2,4,8,16,32,64,-128)`:

```
P = sum_p coefficient[p] * popcount(weightSigns AND activationPlane[p])
dot = 2*P - DC.
```

The producer performs ballot-based plane transposition once. Native POPC executes directly; this route does not use compiler-expanded BMMA. It is a measured negative versus DP4A: at R16 the implementation uses255 registers and spills244B stores/240B loads with184B stack, and the complete head takes991us. DP4A R1/R8/R16 uses40/85/148 registers and no spills. The negative is for this priced schedule, not all possible bitserial schedules.

`results/sass-receipt.json` records native IDP.4A.S8.S8 and POPC counts and SASS block hashes for every instantiated row count; the candidate projection kernels contain no IMMA. This is native instruction evidence, not an operation-rate ceiling.

## Formal and numerical proof scope

[The witness](witness/README.md) supplies a kernel-checked Lean scalar byte bridge, arbitrary-width packed popcount dot law, K128 identity/bounds, finite nibble expansion, and affine DP4A identity. No admissions or native-computation axioms. C++ checks512 scalar cases,16 nibble expansions,16,384 lane cases and69,632 representative K128 vectors. The proof ends at exact **given-A8** integers; it does not identify native FP32 arithmetic with real-number arithmetic.

The independent GPU reference reads original GGUF sign bytes and accumulates exact integer K128 dots, then applies the same original scales and activation scales in FP32. All matrix outputs are compared. Initial four geometries/R1,8,16 selected-map maxima: absolute FP32-reference error4.76837158e-6, relative RMS2.43763258e-7, FP64 mathematical sample error6.53654951e-7 on64 distributed rows/token. Each raw receipt retains the corresponding numerical fields for the expanded geometries too. Unsplitted packed routes are bitwise equal to the independent A8 reference in the exercised cases.

`carrier_check.cu` executes all R1..16 for both native maps on a complete K128 projection, including original FP16 scales and alpha=.75,beta=.25 with independent initial Y: **8,704 outputs bit-exact** against the dense integer/FMA reference. It also checks R512 producer operation, persistent error aggregation surviving later valid preparation, typed nonfinite/scale-range outcomes, invalid-carrier NaN projection and the all-zero valid path. `results/carrier-check.log` is the device receipt.

## Declared activation-quality interface

The recurring producer takes finite FP32 X and quantizes independently per token/K128:

```
max = max_i abs(x_i)
sA = max / 127
q_i = round-to-nearest-even(x_i * (127 / max))
```

An all-zero group has sA=0,q=0. Reachable normal q is in[-127,127]; the integer witness covers the stronger full signed-byte range[-128,127]. Nonfinite input, or a nonzero maximum outside[1e-30,1e30], produces a typed device error instead of silently selecting another representation. q is set to0 for that invalid group and its projection is NaN; callers reject the recorded status. `Carrier.error`, when supplied, preserves an aggregate error across later producer calls with atomicMax only on invalid groups. The engine clears it once/body and checks it at normal synchronization, avoiding a per-prep host fence.

On the benchmark's generic FP32 activations and actual trained weights, mapped output relative RMS versus the unquantized FP32 route is around0.0038–0.0040. This is activation-quantization error, not packed-map error and not model KL/PPL. Parent owns the complete-model quality/task comparison and any decision to adopt the explicit hybrid.

## Complete recurring bill and storage

Preparation transposes original18B Q1_0 blocks into signs `[K/128,4,N]` and FP16 scales `[K/128,N]`. Prepared weights stay exactly1.125bits/weight, without padding or dense decode. Static preparation is excluded from recurring projection latency and documented separately; original INT8 uses its own2-bit frag2 layout plus FP32 scales.

DP4A carrier writes128 q bytes plus12 metadata bytes per K128/token (scale, DC, typed status). Popcount writes128 plane bytes plus the same metadata; no recurring q byte array is required by its projection. No dynamic dot LUT is built. Producer reads the FP32 activations, performs max reduction, rounding, DC reduction, and optional plane transposition. The carrier can be shared by matrices with the same producer. Standalone measurements charge its production on every projection; no reuse credit is taken.

Both candidates read original packed weight words, accumulate exact int32 selected labels, apply `original_FP16_weight_scale * FP32_activation_scale` per K128, and either write final alpha/beta output or spill scaled split-K partials. Partials cost `4*R*N*split_count` bytes. Producer, projection, scaling, partial writes/reads and output continuation all lie inside the timed graph. Static weight layout boundaries are paid once, not counted as free recurring work.

## API and reproduction

`bitserial.cuh`/`bitserial.cu`, CUDA error returns, caller-owned device allocations:

```cpp
bitserial::Plan p;
CK(bitserial::select(N,K,R,&p)); // recommended explicit DP4A schedule
CK(bitserial::produce(bitserial::Route::Dp4a01, X, K, wholeBatchR, carrier, stream));
CK(bitserial::project_carrier(p, signs, originalFp16Scales, carrier,
                             Y, partials, alpha, beta, stream));
```

Projection supports every R1..16. Producer supports R1..65535 so complete prefill batches prepare once before projection tiling. `select` supports all positive N,K divisible128 with a declared scheduling formula seeded by the measured geometries; unmeasured intermediate row counts/shapes are supported correctly but are not separate tuning receipts. Matrices and carrier/output views are tightly packed. `prepare` consumes original device Q1 bytes. Size helpers require validated plans/dimensions. All buffers remain owned by the caller, and asynchronous numerical status must be checked as documented.

Important measured schedules: N96 always chunk1/threads128; full head R8 chunk16/128 and R16 chunk40/128; pruned65536 head R8 chunk4/128 and R16 chunk16/128; fused GU R8 chunk16/128; FFN down R8 chunk4/256 and R16 chunk4/128. `summary.json` retains every chosen raw plan; `select` maps intermediate rows to explicit family schedules.

Build the standalone benchmark against the unchanged earlier packed1 source and binary GGUF reader:

```sh
# On the existing campaign GPU from /workspace/binary/bitserial:
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v -I../packed1 -I../halo/src \
  bitserial.cu reference_int8.cu bench.cu ../packed1/packed1.cu ../halo/src/gguf.cpp -o bench
flock /tmp/binary-gpu.lock ./bench /workspace/binary/models/Bonsai-27B-Q1_0.gguf \
  output.weight 8 0 head-r8
# Fused trained tensors: alpha_beta, gdn_in, attention_in, gu.
# Fifth argument limits rows: 65536 for pruned-head geometry; 0 uses all rows.
python3 summarize.py
```

Benchmark arguments: MODEL TENSOR R N_ROWS PREFIX. With only MODEL, list binary tensors. Fused groups concatenate the actual model tensors in engine order; square control uses the first5120 gate rows. Pruned-head geometry uses the first65536 trained output rows, not a claim about the engine's chosen vocabulary or its quality.

Measurements: October10,2026 around19:15–19:24UTC, parent-provisioned RTX5090/170SMs, CUDA13.1, driver595.58.03,500W limit, free boost. Median15 after3 initial trials, both warm and cold. Cold trials first evict L2 with128MiB outside the timed interval. Batches hold `/tmp/binary-gpu.lock`; no concurrent parent timing, rental or public push. Model source is the parent-uploaded official binary27B GGUF, recorded SHA25617ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0; input activations are deterministic generic values, not captured model hidden states.

Raw sources/results are retained by phase: initial R1/8/16 candidate kernels; `build-allrows.log` and expanded geometry after all-row instantiation; final `carrier-check.log` after persistent-status and scheduling integration. Later select changes affect only chosen schedules, not the measured map implementations. Whole-model modern-engine evidence stays with the parent.
