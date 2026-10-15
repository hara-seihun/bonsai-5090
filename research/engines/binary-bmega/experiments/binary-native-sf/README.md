# Native binary weight scale coordinates

Opt-in capturable consumer: [`native.cuh`](native.cuh), [`native.cu`](native.cu).
This changes the weight representation, not merely the scheduling of original FP16 scales.

## Decision: R32 inclusive cold latency

| Matrix | Ordered FP4 µs | Joint pair µs | DP4A µs | SF1 µs | SF1 / joint speedup | SF1 projection RMS |
|---|---:|---:|---:|---:|---:|---:|
| GU | 77.568 | 61.440 | 135.168 | 59.136 | 1.039× | 2.7836% |
| Down | 75.776 | 59.392 | 86.016 | 57.312 | 1.036× | 2.8549% |
| Full head | 368.640 | 235.520 | 739.328 | 202.752 | 1.162× | 2.6894% |

SF1 wins over the existing shared-weight pair, with additional weight-scale error.
Whole-model PPL/KL determines whether to adopt it. SF2 reconstructs every tested
original scale exactly (observed scale RMS/max-relative error zero), but its cold
GU/down/head latencies71.424/65.504/294.912µs lose against the joint pair. SF2
projection relative RMS is at most9.36123797e-8 across the tested configurations.
No full-model throughput or quality result is claimed here.

SF1 weight resident bytes GU/down/head:23,814,144 /11,857,920 /169,850,880.
Original controls:25,067,520 /12,533,760 /178,790,400.
SF2:25,206,784 /12,554,240 /179,783,680. No residual weight sign payload is stored.

Best SF1 cold plans: F8G1T32/split1 GU/head, F8G1T32/split4 down. SF2 cold plans:
F8G1T32/split1 GU/head, F4G2T16/split4 down. Every winner is selected independently
from the same finite schedule/split set. The147 R32 records and390 R1/R8 records in [`results/`](results/)
and [`summary.json`](summary.json) preserve the controls and losses.

## Decision: R1/R8 sidecar

| R | Matrix | Joint pair cold µs | DP4A cold µs | SF1 cold µs | SF2 cold µs | SF1/joint speedup | SF2/joint speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | GU | 51.200 | 45.056 | 52.992 | 53.248 | 0.966× | 0.962× |
| 1 | Down | 42.464 | 34.816 | 40.960 | 43.008 | 1.037× | 0.987× |
| 1 | Full head | 146.912 | 135.168 | 139.008 | 147.456 | 1.057× | 0.996× |
| 8 | GU | 49.152 | 61.472 | 51.200 | 51.200 | 0.960× | 0.960× |
| 8 | Down | 40.960 | 47.104 | 43.008 | 43.040 | 0.952× | 0.952× |
| 8 | Full head | 165.920 | 229.376 | 143.360 | 155.392 | 1.157× | 1.068× |

No broad R1 mode wins: DP4A beats both SF variants in every R1 geometry. No broad
R8 SF replacement wins against the joint pair: GU/down lose. Only an explicit
**full-head R8** SF1 or SF2 mode is supported by this price comparison, both at
**F8G1T8/split1**. SF1 needs the parent's whole-model PPL/KL decision; SF2 is a
scale-exact measured head alternative, not universally FP16 exact.

Best cold plans for explicit experimental use (not automatic dispatch):

| R | Matrix | SF1 cfg/splits | SF2 cfg/splits | SF1 projection RMS |
|---:|---|---|---|---:|
| 1 | GU | F8G1T8/8 | F8G1T8/1 | 2.7825% |
| 1 | Down | F8G1T8/16 | F8G1T8/8 | 2.9018% |
| 1 | Full head | F8G1T8/1 | F8G1T8/1 | 2.7216% |
| 8 | GU | F8G1T8/1 | F8G1T8/1 | 2.7828% |
| 8 | Down | F4G2T8/16 | F8G1T8/16 | 2.8439% |
| 8 | Full head | F8G1T8/1 | F8G1T8/1 | 2.6704% |

All prepared row statuses were Valid in all six cases. SF2 again has zero scale
error on every inspected trained K128 scale. Across all small-row configurations,
SF2 projection relative RMS<=9.39866592e-8; SF1<=2.90182241%. Independent decoded
scalar maximum errors SF1/SF2 are3.28740555e-7 /4.00564993e-7. Every native add
contract passed. Output errors are versus original-scale two-plane FP4 on the
same transformed synthetic inputs; no captured-model/token quality is claimed.
Resident bytes and activation/continuation boundaries are unchanged from R32.
[`results/small-provenance.json`](results/small-provenance.json) binds the final
status/configure API, unchanged consumer object, chosen warm/cold plans and job.
The six-case GPU round completed exit0 under the55-second exclusive lock guard.

## Representation

The prepared binary sign payload is unchanged WF8: 512 bytes per N32/K128 chunk.
Each row has one FP32 factor g, selected statically as a power of two at load:

```
g = 2^ceil(log2(max_K128(abs(original_d)) / 256))
SF1 = E4M3_RN(original_d / g)
SF2 = E4M3_RN(original_d / g - decode(SF1))
```

An all-zero weight row deliberately uses g=1 and zero SF bytes. Each SF byte
contains magnitude in its low seven bits and a sign in bit7. The native scale
operand receives magnitude; sign XORs the corresponding binary E2M1 register
fragment. Both scale planes reuse the same resident sign payload.

Each chunk stores `[plane][gid][m,r]`, 32 bytes per plane. The value is replicated
across the four native K16 scales of each K64 MMA. One or two weight planes is
an explicit plan choice. Both primary/residual activation operands retain their
existing q/SF/g layouts. Each activation plane accumulates across all assigned K
blocks directly in tensor cores, including both weight planes when selected.
Only the final epilogue combines activation row factors and multiplies by g:

```
y = (primary_acc * primary_g + residual_acc * residual_g) * weight_row_g
```

Thus there is no per-K128 FP32 weight-scale epilogue or accumulator reset. SF1
requires four MMAs/K128; SF2 eight, versus four plus per-block FP32 continuation
for the shared-weight joint-pair control. Split-K folds remain explicit and paid.

## API

```
configure(schedule, weight_planes) // outside graph capture, for every chosen pair
prepare(original_WF8_half_scales, native_SF, row_factor, row_status,
        N, K, weight_planes, stream)
project(plan, primary, residual, WF8_signs, native_SF, row_factor,
        output, output_stride, partials, partial_capacity, stream, add)
```

All memory belongs to the caller. `prepare` is a static load-time transform,
never part of recurring inference. `row_status` is a required N-element device
ScaleStatus array. After normal stream completion, inspect all rows before using
the representation. Valid finite half input writes Valid; NaN/Inf writes NonFinite,
a NaN row factor and zero SF bytes. Unknown/nonfinite input is not treated as a
successful representation. Negative finite scales are supported. Finite FP16
input yields a representable positive power-of-two row factor and normalized
magnitude<=256; no scale-range fallback is needed. The status buffer can be
released after checking load-time success and is not part of inference residency.
`configure` explicitly sets the chosen kernel's shared-memory attribute outside
capture. `project` then performs only capture-compatible launches and host argument
validation, with no allocation or attribute changes. `scale_bytes`
returns N*(K/128)*weight_planes for valid N/K and one/two planes; invalid dimensions
return zero. Row factors require N floats. Input activation pointers must start at an existing
tile64 boundary: `[tile64][K128][row64]` q bytes64/row/block, SF bytes8/row/block,
and g one float/token. Do not approximate this swizzled coordinate with an arbitrary
`row*K/2` q offset; R>32 is explicitly rejected. `partial_floats` provides the split
scratch requirement for a valid plan. Production needs only native.cu/native.cuh,
not benchmark/control objects. No allocation occurs in `project`.

Positive N divisible32, K divisible128, R1..32, stride>=N, splits1..K/128 are
structurally supported; this receipt times only R32 and the three trained matrix
geometries. q/SF/W/native-SF pointers must be16-byte aligned, row factors/g/output
and partials float-aligned. Scratch must not overlap output. Invalid arguments or
unknown schedules return cudaErrorInvalidValue; asynchronous CUDA status is
caller-owned. `add=true` adds once after the native projection/split fold.

## Measurement boundary

Weights/norms: Bonsai-27B-Q1_0.gguf, layer0 GU N34816/K5120, down N5120/K17408,
full head N248320/K5120, all R32. Inputs are deterministic synthetic activations,
not captured model activations. GU pays trained post-attention RMSNorm, existing
primary/residual FP4 producers, projection, SiLU*up and next primary/residual FP4
producers. Down pays SiLU-multiply, both producers, projection, residual add,
trained next-layer norm and next primary/residual FP4 producers. Head pays final
RMSNorm, both producers, projection and full-vocabulary GPU argmax.

The DP4A arm pays the same FP32 transform, A8 carrier production and original-scale
DP4A projection, then the same downstream FP4 continuation. It does not pay unused
input FP4 producers. Its output differs in activation quantization from two-plane
FP4, so its numerical error versus that reference is reported rather than treated
as exact parity.

Controls: ordered existing two-pass FP4, unchanged shared-weight pair kernel,
and original-scale A8/DP4A. [`pair-control.cu`](pair-control.cu) calls the existing
pair kernel directly at R32 because that experiment's public wrapper caps R16;
its math/schedule is unchanged. R32 native/baseline schedules3..6, splits1/4/16 are swept independently.
R1/R8 use schedules0..3 (TOK8 and TOK16), splits1/4/8/16; the compiled native,
control and producer objects are reused, with only benchmark enumeration rebuilt. DP4A uses its existing explicit measured selector, tiled R16.
All arm-owned activation production and all common downstream continuation recur
inside CUDA graphs. Static model mapping, repacking, native scale preparation,
allocations and capture are outside. Weight resident bytes include every native
SF plane and every weight row factor; original representations are charged their
own complete signs/scales, not simultaneous diagnostic copies.

Warm/cold: 3 discarded warmups plus median10, CUDA graph elapsed time. Cold evicts
128MiB immediately before each inclusive graph. One existing RTX5090,
`flock /tmp/binary-gpu.lock`; the whole GPU round is bounded at55s. No rental or
service is created. Raw records and the summary retain all measured configurations.

The benchmark checks each projection against original-scale ordered two-pass FP4,
checks sixteen sampled columns/token against independent FP64 decoded scalar sums
for both original and transformed weight scales, and checks exact add-contract
agreement with its own computed projection. Scale errors inspect every K128 scale
of all three matrices. Exact observed SF2 scale reconstruction is a measurement,
not a claim that arbitrary FP16 values decompose exactly into two E4M3 values.

## Build and reproduce

```
make -C experiments/binary-native-sf -j2
bash experiments/binary-native-sf/run-round.sh
bash experiments/binary-native-sf/run-small-round.sh
python3 experiments/binary-native-sf/summarize.py
```

The remote runner paths are explicit campaign paths in run-round.sh. Standalone
build reuses the modern engine producer object when KERNELS_OBJECT is supplied;
source/object hashes and compiler flags are in results/provenance.json. The final
standalone object is retained at `~/data/binary-5090/native-sf/native.o`,
SHA256 `2d06f884d4eca46fb418f4d23fa25c841564a80bc1d65957ed710638b0484037`.
It is rebuildable from this source and may be deleted after downstream integration
no longer needs it. Final API changes added explicit load-time status and moved
shared-memory configuration outside capture after the measured round; the valid
weight representation, tensor-core kernel body and device timing path are unchanged.
Remote sources/objects live at `/workspace/binary/native-sf/halo/`, exported local
results in this directory survive teardown. Whole-model PPL/KL, engine selection
and token-throughput integration remain with the campaign parent.
