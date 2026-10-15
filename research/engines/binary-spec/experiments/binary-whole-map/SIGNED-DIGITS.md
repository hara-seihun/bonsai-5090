# Native three-plane FP4 digits

**Exact fullA8 wins on GU/down and R32 head.** The initial hidden7 signed-magnitude gate also passed. Parent supplied the subsequent arithmetic recode that extends three planes to every signed byte, without changing quantization. This experiment owns native prices/parity; engine dispatch and full-model integration belong to the parent.

## Exact fullA8 result

Cold complete boundary: originalA8 producer + encoding + original-scaled projection/split reduction + common continuation. Both candidate and controls consume identical original q/sx. OriginalDP4A includes its producer and continuation but has no digit encoding to pay.

| Trained shape | R | DP4A µs | Best four digits µs | Exact three µs | Three cfg/split | Matched-plan four µs |
|---|---:|---:|---:|---:|---:|---:|
| down5120×17408 | 16 | 42.720 | 34.784 | **30.688** | 0/8 | 34.784 |
| down5120×17408 | 32 | 69.376 | 49.120 | **42.976** | 1/8 | 49.120 |
| GU34816×5120 | 16 | 67.584 | 63.200 | **59.392** | 0/1 | 63.200 |
| GU34816×5120 | 32 | 118.560 | 83.936 | **77.792** | 1/1 | 83.936 |
| head248320×5120 | 16 | 374.528 | 339.968 | **339.968** | 0/1 | 349.952 |
| head248320×5120 | 32 | 733.184 | 425.984 | **403.456** | 1/1 | 425.984 |

HeadR16 is **neutral against the best four-plane alternative**, despite beating its matched four-plane plan. Down's exact3 incremental speedups over bestfour are1.133×/1.143×, GU1.064×/1.079×, headR32 1.056×. These measured prices do not constitute an end-to-end model-throughput result.

All480 native exact3 output arrays (six shapes×20plans×four boundaries) are **bit-exact against their same-plan four-digit outputs**. Original fullA8 quantization/scales are unchanged. Maximum projection error against independent rawQ1 across all arms is5.7220459e−6, maximum relativeRMS2.42138633e−7; split association accounts for the numerical difference. The60-plan rawQ1 witness covers all256 byte patterns and R5/16/32/65 tails: split1 is bit-exact; split4/8 maxerror2.38418579e−7 and maxrelativeRMS8.47087015e−8. Status/zero/guard cases passed. Head argmax is checked against the independent reference; GU/down continuations and next-producer statuses are checked.

`exact8-*.jsonl`, `exact8-round.log`, `exact8-provenance.sha256`, `gpu.txt` and `summary.json` retain all receipts and losses. Exported source hashes match the current local files. GPU: RTX5090, CUDA13.1.115,500W.

## ExactA8 recode

Use explicit `Encoding::ExactA8` in `Digits`. For q∈[-128,127]:

```
a = q & 3
b = (q >> 2) & 3
c = q >> 4                   // signed arithmetic shift
if abs(c) == 5 or abs(c) == 7:
    delta = b odd ? 1 : -1
    c += delta
    b -= 4*delta
q = a + 4*b + 16*c
```

Now a,b,c belong to2×E2M1={0,±1,±2,±3,±4,±6,±8,±12}. Three planes hold a/2,b/2,c/2. **No lookup table is loaded:** a's FP4 nibble is a; adjusted b's nibble is `(0x95b4>>(4*original_b))&15`; c's magnitude nibble is `m<=4?m:m/2+2`, plus its sign bit.

Each K128 dot uses six FP4 MMA instructions (two halves×three planes), with immediate unit UE4M3 scales. Recover `RN(2*d0)+4*RN(2*d1)+16*RN(2*d2)` as an integer, then apply **original FP16 weight-scale × original FP32 activation-scale** and RN FP32 FMA. No sx→UE4M3 conversion, dense persistent weight decode or quantization loss is introduced. Every signed byte, including−128, is representable.

`signed-digit-results/exact_a8.lean` proves the actual E2M1 nibble decoding identity on all256 byte patterns; Lean completed successfully. The bench independently exhausts host bitcodes; GPU witnesses and same-plan parity establish the measured native result.

## API and residency

Sources: `signed_digits.{cuh,cu}`; namespace `signed_fp4`. Existing four-plane consumer `binary-a8-fp4/digits.cu` is unchanged.

```
Digits d{buffer, capacity, Encoding::ExactA8};
Plan p{N, K, rows, splits, Schedule::F4G2T16, WeightLayout::Packed1};
configure(p.schedule); // once per device/context, before capture
encode(original_carrier, K, rows, d, stream);
project(p, original_signs, original_fp16_scales, original_carrier,
        d, y, stride, partials, partial_capacity, alpha, beta, stream);
```

The parent can link the targeted object at `/workspace/binary/digits-ragged/halo/experiments/binary-whole-map/build/signed_digits.o`. Original producer is `bitserial::produce(Route::Dp4a01,...)`; **`signed_fp4::produce()` is specifically the hidden7 producer, not the ExactA8 producer**.

Packed1 consumes existing positive signs `[K128,4,N]` and original FP16 scales `[K128,N]`, with no extra resident weights. Signs are expanded once and reused for all three planes. Four-stage ring traffic is512B signs+64B scales perN32×K128, plus three activation tiles. Three-plane scratch is1.5B per padded activation, padded to64 rows: K17408 at R16/R32 uses1,671,168B versus four planes2,228,224B. Original q/sx/DC/status/error remain live. Absolute-value/sign wiring in Signed7 and arithmetic recoding in ExactA8 are recurring encoder costs.

Valid shapes are positive N32/K128, rows1..65535, explicit split1..K/128, one of five schedules, explicit WF8/Packed1 and explicit Signed7/ExactA8 encoding. Bad structural inputs/capacity/enum return CUDA errors. Invalid carrier status propagates NaN; aggregate producer error remains caller-visible and sticky. Finite alpha/beta are required; beta0 does not read Y. Split-K folds partials in ascending order with RN arithmetic before alpha/beta continuation. No allocation or attribute mutation occurs inside capture-safe encode/project.

Cfg0=F4G2T8, cfg1=F4G2T16, cfg2=F4G2T32, cfg3=F8G1T32, cfg4=F8G2T32.

## Measurement boundaries

`signed_digits_bench.cu` reads actual original trained layer0 GU/down and fullhead bytes. It keeps four boundaries distinct:

1. `project`: given prepared planes, complete projection and split reduction.
2. `encode_project`: given q/sx, native encoding + projection.
3. `producer_encode_project`: given F32 in the projection's input coordinate, originalA8 producer + encoding + projection.
4. Complete producer/encode/project + common continuation: GU uses SiLU(gate)×up then originalA8 at17408; down uses residual add then originalA8 at5120; head uses existing argmax.

The stage4 receipt enum is named `producer_encode_project_residual_next_a8`; **its operation is determined by `kind`** as listed above. `summary.json` supplies the explicit `continuation` field. GU's continuation error is stored in the common `residual_*` metric fields.

Every schedule×split1/4/8/16 is retained. DP4A uses the original selector, with R32 tiled into2×R16 and one producer per graph. Four-plane/three-plane routes use identical q/sx. Inputs and residuals are deterministic synthetic F32, not captured model activations. GU starts in normalized-input coordinates, down starts after hidden SiLU×up, head starts in final-input coordinates; predecessors outside these boundaries are neither optimized nor credited as free.

Each graph reports median7 after two discarded warmups, both warm and following128MiB cache eviction. Static transpose, allocation, graph capture and host preparation are excluded. Recurring original exact scales, encoding, split continuation and stated producers are included. Three exclusive GPU windows ran under `/tmp/binary-gpu.lock`, timeout55 after acquisition, owned by `kjob@binary-exact-a8-three-plane`. All receipts were exported before its successful settlement.

The rawQ1 witness uses trained first96 down rows, independent varying F32 sx, q cycling all256 states, R5/16/32/65, stride128, alpha.75/beta.25, initialized guards and N/row tails. It also checks invalidstatus→NaN, undersized scratch/invalidlayout rejection, original NonFinite/ScaleRange errors, persistent error after validzero production, and zero→zero.

## Hidden7 result retained

`Encoding::Signed7` uses sign-magnitude radix4 digits for q∈[-63,63], all integer E2M1. Its recovery factor is1 rather than2. `signed_fp4::produce()` uses original K128 mechanics with divisor63: q=RNE(x*(63/maxabs)), sx=maxabs/63. Nonfinite/out-of-range maxima yield typed status/error; zero yields zero. Signed7 encode rejects q outside[-63,63] by asynchronous ScaleRange status/error and downstream NaN. `signed_digits.lean` proves its finite identity/bounds.

| Hidden7 down, cold complete boundary | R16 µs | R32 µs |
|---|---:|---:|
| Original DP4A, same hidden7 q | 42.752 | 69.056 |
| Best four digits, same hidden7 q | 34.784 | 49.120 |
| Signed3 | **32.736** | **45.056** |
| signed3 cfg/split | 0/8 | 1/8 |

The160 signed3 output arrays were same-plan-four-digit bit-exact;60 independent witness plans passed including q64 rejection. Bare projection minima24.288/34.560µs; q→encode→projection minima28.640/38.912µs. Separately named original fullA8 controls were42.976/67.552µs DP4A and34.784/49.152µs bestfour. Their q differs from hidden7; they are not same-map parity claims.

`down-r{16,32}.jsonl`, `witness.jsonl`, `hidden7-source.{cu,cuh}`, `hidden7-bench.cu`, `hidden7-provenance.sha256`, `hidden7-run.sh` and `hidden7-round.log` retain that gate. Current API adds explicit `Digits.encoding`; new integration must set it.

## Reproduce

From the repository root:

```
bash experiments/binary-whole-map/signed-digit-results/run.sh
python3 experiments/binary-whole-map/signed-digit-results/summarize.py
lean experiments/binary-whole-map/signed-digit-results/exact_a8.lean
```

The finite runner transfers independent sources to the existing rental mirror, compiles against pinned `bitserial.o`, `digits.o`, `kernels.o`, `gguf.o`, runs locked sub-minute windows and exports receipts before exit. Parent owns rental lifecycle/teardown. No commit, public push, new rental or recurring wake is part of this subtask.
