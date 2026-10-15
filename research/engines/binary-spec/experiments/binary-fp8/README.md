# Native FP8 binary batch projection

Original binary weights and FP16 K128 weight scales stay exact. FP32 activations
use a new **approximate E4M3** carrier, not A8 and not exact original FP32.
This experiment is separate from the exact four-FP4-digit A8 route. Defaults and
engine source remain unchanged.

## Concrete family selection

CUDA13.1/sm120a rejects mixed E4M3×E2M1 `kind::f8f6f4` and blockscaled
`kind::mxf8f6f4` (minimal probes and rejection logs retained). The executable route
therefore relabels each binary sign **exactly** to E4M3 `+1=0x38`, `-1=0xb8`
register operands and uses ordinary native
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
No E2M1×FP8 hardware support is claimed. The sign-to-E4M3 boundary is priced in the
consumer; original prepared Q1 remains16 sign bytes plus2 scale bytes/K128/output,
with **zero persistent FP8 weights**. Nibble expansion is
`0xb8b8b8b8 XOR ((((nibble*0x00204081)&0x01010101)<<7))`.

The sm120a consumer lowers to four native `QMMA.16832.F32.E4M3.E4M3` per K128,
56 registers/thread, zero spills. Matrix orientation is M16 output features ×N8
tokens, so each lane reuses two original weight scales across its two token
columns. Tokens and N96 feature tails are masked. Activation words use a
shared-memory bank swizzle; all shared staging/register expansion is recurring
consumer work, not omitted setup.

## Activation and scale boundary

For each token/K128, producer computes `maximum=maxabs(x)`, FP32
`sA=maximum/448`, and RN/saturate-finite E4M3 codes of `x*(448/maximum)`.
Zero groups produce zeros. Nonfinite or nonzero max outside[1e-30,1e30] gets the
existing typed status; optional persistent error aggregates invalid production.
Native QMMA accumulates within each K128 only, then the original FP16 W scale
multiplied by full FP32 sA continues the dot into FP32 output. There is no E8M0 or
UE4M3 approximation of those scales. Split-K reduction and alpha/beta follow.

`fp8.cuh` owns `Plan{n,k,r,groups_per_split,n_tile}` and
`Carrier{q,scales,status,error}`. N32/K128, R1..32, feature tiles32/64/128; output
row stride `ldo >= N`. Producer supports R1..65535 for full-batch production before
row-view projections. Caller owns all buffers and `[R,split,N]` FP32 partials;
row views offset q by row*K and metadata by row*K/128. Launches allocate nothing,
synchronize no host and change no function attributes: capture-safe. Invalid
carrier status produces NaN output; the engine must inspect its aggregate error
at normal end synchronization. Invalid plans/buffers/strides return CUDA errors.

## Paired keyhole

`bench.cu` uses trained GU/down/output weights and generic deterministic FP32
activations. Full native outputs are checked against an independent raw-Q1
reference with decoded E4M3 bytes. DP4A has its own raw-Q1/A8 oracle. These are
correctness gates, unlike FP8-versus-A8 comparisons. Quantization error versus
original FP32 is reported separately on256 evenly distributed output columns
with FP64 accumulation; native-versus-DP4A is reported over full output.

N96 guards cover alpha.75/beta.25, stride128, prefix/padding/tail guards, R5
uniform and R32 Gaussian inputs. Main timing uses uniform nonconstant/distinct
inputs, R32 GU/down/head65536/fullhead, three fixed feature tiles, chunk16.
Head65536 is an explicitly labeled contiguous prefix, **not** the engine's
vocabulary-pruned row set or a pruning-quality result.

Inclusive graphs charge one full-R activation producer, transient exact weight
format conversion, shared-memory packing, native projection, original scales and
final split continuation. Carrier-only still charges all consumer boundaries.
DP4A also produces once fullR, then uses <=16-row views without re-quantization.
Median7 after2 discarded samples;128MiB eviction is before cold events. Static
packing/allocations/transfers/graph construction are excluded from recurring
projection and persistent bytes reported. One shared50-second harness budget
inside55-second post-lock timeout; `round_complete` plus zero exit required.

```sh
bash experiments/binary-fp8/build.sh /absolute/path/bench "$PWD"
flock /tmp/binary-gpu.lock timeout 55s /workspace/binary/fp8/bench \
  /workspace/binary/models/Bonsai-27B-Q1_0.gguf round /workspace/binary/fp8/results/round
```

## Result: no FP8 integration

The one locked round completed in0.708875 seconds. Every full native output
passes its decoded-FP8 oracle (largest relative RMS2.28e-7, maxabs5.72e-6), and
all56 strided guard cases pass. Correctness of the FP8 carrier is separate from
its quantization error versus the original input.

| R32 projection | DP4A cold µs | Best FP8 cold µs | FP8 tile |
|---|---:|---:|---:|
| GU34816×5120 | 112.640 | 139.264 | 32 |
| down5120×17408 | 65.504 | 79.616 | 32 |
| output prefix65536×5120 | 190.464 | 249.856 | 32 |
| full output248320×5120 | 725.024 | 944.128 | 64 |

All fixed FP8 controls use chunk16. Warm GU has a small gain106.816→104.608µs;
warm down58.464→59.488 and heads also lose. Sharing the producer cannot rescue
this consumer: GU producer-only is4.096µs for both carriers, while cold
carrier-only is116.736DP4A versus139.264FP8.

Uniform-input projection relative RMS versus independent FP64 accumulation of
original FP32 inputs (256 evenly distributed columns, all32 tokens):

| Geometry | A8/DP4A | FP8 |
|---|---:|---:|
| GU | 0.401% | 2.542% |
| down | 0.375% | 2.609% |
| output prefix65536 | 0.392% | 2.520% |
| full output | 0.390% | 2.560% |

These are synthetic activation distributions on trained weights, not full-model
PPL/KL. **Retain DP4A / the separately owned exact FP4 candidate; do not integrate
this FP8 consumer.** Quantization quality is worse and the complete cold route
also loses. Source/API and all candidate/quality/residency receipts are retained
in `results/`, with exact timed source/object/executable hashes and compile
rejections. The [finite/Lean witness](witness/README.md) owns exact binary-to-E4M3
relabeling and the linear error identity, not CUDA or full-model quality.
Parent owns any full-model PPL/KL/quality decision and engine integration.
