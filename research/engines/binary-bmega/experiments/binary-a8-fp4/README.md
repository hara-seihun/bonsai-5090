# Exact A8 on four FP4 digits

Bounded opt-in experiment. Source/API: `digits.cu`, `digits.cuh`. Current DP4A/default engine source is unchanged.

## Decision

**R32 wins; R8 is shape-selective.** On RTX5090 at500W/CUDA13.1.115, measured cold inclusive cost:

| Trained projection | R | Current DP4A µs | Four FP4 digits µs | Speedup | Chosen cfg/split |
|---|---:|---:|---:|---:|---:|
| GU34816×5120 | 8 | 46.848 | 57.344 | 0.817× (negative) | 0/1 |
| GU34816×5120 | 32 | 118.784 | 81.920 | 1.450× | 0/4 |
| down5120×17408 | 8 | 32.768 | 28.416 | 1.153× | 0/8 |
| down5120×17408 | 32 | 69.376 | 51.168 | 1.356× | 2/8 |
| full head248320×5120 | 8 | 227.328 | 311.296 | 0.730× (negative) | 0/1 |
| full head248320×5120 | 32 | 739.360 | 440.320 | 1.679× | 0/1 |

Cfg0=F4G2T8, cfg1=F4G2T16, cfg2=F4G2T32, cfg3=F8G1T32, cfg4=F8G2T32. Four ring stages each. Every sweep record, including losses, is preserved in `results/`; `summary.json` owns the chosen records. All126 timing records and60 native witness records completed within the one45-second GPU window and were exported by the durable job.

Maximum full-output error against DP4A across all candidate configurations is6.67572021e-6; max relative RMS2.414e-7. The independent rawQ1 witness is bit-exact for split1 at every tested schedule/R; split4/8 max absolute error is1.78813934e-7. Witness RMS is computed including initialized guards; use its explicit maximum error and separate exact guard check rather than that diluted RMS. Numerical-status/zero checks passed and their terminal receipt is `results/round.log`.

The first WF8 round recommended explicit opt-in R32 GU/down/head; the direct-Packed1 follow-up below resolves resident one-bit ownership and updates the chosen schedules. R8 GU/head should remain DP4A. This experiment does not claim model-throughput or full-model quality; parent/native worker own integration and its task/quality gate.

## Map

Given the **already produced** signed-byte A8 carrier, split its two's-complement byte into radix4 digits:

```
q = d0 + 4*d1 + 16*d2 + 64*d3
 d0,d1,d2 ∈ {0,1,2,3}; d3 ∈ {-2,-1,0,1}
```

All these digit values are exactly representable E2M1. `encode()` produces four engine-compatible swizzled FP4 activation planes, zero-padding unused rows to64. There are **no dynamic SF bytes**: the MMA uses immediate unit UE4M3 scales `0x38383838`. The tensor consumer reads/decompresses each packed WF8 sign fragment once for all four planes; four K128 digit dots are converted RN to integer, then combined by1/4/16/64. Only after recovering the signed integer dot does it apply `original_FP16_weight_scale * original_FP32_A8_sx`, followed by FP32 FMA. It never converts sx to UE4M3, never changes q quantization and never persistently decodes dense weights.

The four digit dots have small integral ranges (|dot|≤384 for the nonnegative digits and≤256 for the signed top digit); rounding their native FP4 F32 accumulator to integer removes the irrelevant native floating representation before exact radix recombination. Correctness is conditional on the supplied carrier's numerical status, just as the DP4A route is. Invalid status produces NaN projected output. Metadata/status/error remain owned by the original A8 producer.

## Inclusive boundary

The timing graph includes **one unchanged original F32→A8 quantization**, digit encoding for all four planes, complete projection, original exact scales, ordered split-K continuation, and common output continuation:

* Trained layer0 GU34816×5120: gate/up are joined contiguous halves; common SiLU(gate)×up, followed by original A8 producer for width17408.
* Trained down5120×17408: common independently seeded residual add, followed by original A8 producer for width5120.
* Full head248320×5120: common existing GPU argmax.

The timing starts at a synthetic F32 activation **already in the projection's input coordinate**, not at the model's normalization/gating predecessor. These common input transformations are not credited as improvements. Input/residual vectors are deterministic finite synthetic vectors, not captured model activations. Actual original trained Q1 sign/FP16scale bytes are used. Static repacking, allocations, graph capture and host preparation are excluded.

Each arm pays its own native packed format; baseline uses unchanged original DP4A selection, tiling R32 into2×R16 while quantizing once. Candidate streams WF8, a different one-bit coordinate with identical bit/scale storage size. Both incur128MiB cache eviction before the cold graph; two warmups discarded, median7 retained. Five consumer schedules, splits1/4/8/16. GPU work is exclusive under `/tmp/binary-gpu.lock`, once≤45s, submitted by `kjob@binary-a8-fp4-round`; source compilation is targeted locally.

## Packed1 follow-up

Commit `f53754a` adds explicit `WeightLayout::Packed1` alongside `WeightLayout::WF8`; targeted object pin `c783e12e68948e8eea0dce5776d29c3c079ab0eced3857a0bd899e572ab1653d`. The historical table above is the first WF8 round. The additional direct-Packed1/control gate **passed**, within one55-second exclusive window:120 trained candidate projections are bit-exact against their corresponding same-plan WF8 outputs, and both60-record generic rawQ1 witnesses pass. Split1 is bit-exact against rawQ1; split4/8 maxerr1.78813934e-7 and max active-output RMS3.25980372e-8. No guards are included in the follow-up witness's RMS denominator.

| Direct resident Packed1 | R | Current DP4A µs | Best digits µs | Speedup | Best cfg/split |
|---|---:|---:|---:|---:|---:|
| GU | 32 | 118.784 | 83.936 | 1.415× | 1/4 |
| down | 32 | 69.632 | 48.864 | 1.425× | 1/8 |
| full head | 32 | 741.376 | 430.080 | 1.724× | 1/1 |
| GU | 8 | 47.104 | 57.344 | 0.821× (negative) | 0/1 |
| down | 8 | 32.512 | 28.384 | 1.145× | 0/16 |
| full head | 8 | 227.360 | 313.344 | 0.726× (negative) | 0/1 |

All three R32 cold winners now use `F4G2T16` (cfg1); configure it once and use explicit split4/8/1 for GU/down/head. The native worker's first integrated cfg0/2/0 schedules also remain positive:85.760/51.168/471.072µs, respectively. R8 GU/head remain DP4A. These are the same inclusive boundaries defined above, not full-model throughput.

Matched WF8 costs at the **first integrated** schedules are81.920/49.120/444.416µs for R32 GU/down/head; direct-Packed1 wiring adds3.840/2.048/26.656µs there, explicitly priced rather than treated as free. Improved chosen schedules pay that wiring and still win. Full cold/warm sweeps, both layouts, direct parity flags and logs are under `results/packed1/`; the parent/native worker now own full-model acceptance. `run-packed1-round.sh` and `summarize-packed1.py` reproduce the follow-up.

Packed1 staging copies four contiguous K32 sign strips for the N-tile directly from existing positive signs `[K128,4,N]`. Consumer warps select the relevant eight-bit subword, invert positive-sign polarity, expand the native E2M1 fragment once, and reuse it for all four digits. Weight scales use existing `[K128,N]` linear FP16 rows with the appropriate MMA row mapping. Both physical layouts cost512B signs+64B originalscales per32×128 chunk. No static format change, additional resident weight copy, or persistent dense decode is needed. WF8 remains an explicit control route in the same consumer.

`configure(Schedule)` establishes each schedule's dynamic shared-memory limit once after CUDA device selection and **before graph capture**. `project()` does not set attributes or allocate; constructor/device initialization owns configuration. Configure again for a new CUDA device/context.

## Ragged occupancy price

One additional approved55-second exclusive gate completed R16/R24/R31 on trained GU/down/fullhead, full inclusive boundary unchanged. All180 candidate output arrays are bit-exact against their corresponding WF8 controls. `results/ragged/` retains all189 timing records, log and object/source provenance; `summarize-ragged.py` reports selected-schedule and best-alternative curves separately.

| Selected F4G2T16 | R | DP4A cold µs | Digits cold µs | Speedup | split |
|---|---:|---:|---:|---:|---:|
| GU | 16 | 67.584 | 63.488 | 1.065× | 1 |
| GU | 24 | 98.304 | 81.632 | 1.204× | 1 |
| GU | 31 | 118.528 | 92.160 | 1.286× | 1 |
| down | 16 | 43.008 | 34.784 | 1.236× | 8 |
| down | 24 | 57.344 | 46.816 | 1.225× | 8 |
| down | 31 | 67.552 | 48.864 | 1.382× | 8 |
| full head | 16 | 374.784 | 331.520 | 1.131× | 1 |
| full head | 24 | 583.424 | 415.744 | 1.403× | 1 |
| full head | 31 | 727.040 | 555.040 | 1.310× | 1 |

Recommended occupancy policy: GU/head digits at≥16 active rows, down at≥8 (the previousR8 down receipt is positive), with `F4G2T16` as the shared selected schedule. Use GU split1 for16..31; GU R32 remains split4. Down uses split8 at16..32; its measuredR8 winner is F4G2T8/split16, so applying F4G2T16/split8 atR8 is **not** priced by this table. Keep R8 GU/head DP4A. GU's R32 split4 schedule is essentially neutral atR16 (67.552 vs67.584µs); it is not the smaller-occupancy winner.

The **measured** positive lower tested rows are GU/headR16, downR8. Exact crossover inside9..15 forGU/head is unknown; rows17..23 and25..30 are inferred between tested occupancy points, not separate timing claims. No claim of monotonicity is made. Conservative measured-only dispatch can retain DP4A on untested rows; native owner decides the explicit selection rule.

Optional alternatives: atR24, F4G2T8 beats selectedF4G2T16 for all three families (GU73.472µs/split1,down42.976µs/split8,head392.992µs/split1); atR31, headF4G2T32/split1 is540.416µs vs selected555.040µs. All alternatives are preserved, not combined as additive improvements. Kernel/API/object remain unchanged from `f53754a`; only the benchmark's accepted rows and scripts changed (`d0430a2`). Engine rowcut selection remains the native worker's write scope.

## API/ownership

`encode(original_carrier,k,rows,digits,stream)` consumes original q/scales/status and returns a CUDA status. `project(plan,wf8,fp16_scales,original_carrier,digits,y,stride,partials,capacity,alpha,beta,stream)` consumes prepared planes and writes `alpha*projection + beta*y` (beta0 never reads y). Both are capture-safe, allocate nothing and require caller-owned live buffers. Numeric producer errors are asynchronous typed carrier metadata; callers inspect `status`/persistent `error` at their existing synchronization.

Structurally supported: positive N32/K128, rows1..65535 (four-plane buffers padded to64), output stride≥N, explicit split1..K/128, one of the five explicit schedules, and explicit `Plan.layout` (`WF8` or `Packed1`, sixth field after `Schedule`). No shape fallback. R8/16/32 are the target cases; witness also covers R5 tails. Unsupported dimensions/schedules/pointers/capacities return `cudaErrorInvalidValue`.

* `digit_bytes(k,rows)`: padded_rows×K×2B. Four planes cost2B/activation, plus the retained original1B A8 q and unchanged F32 sx/status/dc/error. Width17408 at R32 uses2.125MiB digit scratch because of64-row padding; no extra persistent weight expansion.
* `partial_floats(valid_plan)`: splits×rows×N when splits>1, otherwise0. Split reduction is ascending-order RN, then one alpha/beta continuation. K128 FMA association matches the original DP4A map within a split; different split partitioning changes FP32 association.
* WF8 chunks are `[K128][N32]`512B negative signs and64B swizzled FP16 group scales. Packed1 signs are positive bits `[K128,4,N]` and originalFP16 scales `[K128,N]`. q/SF scale fields are not applicable to digit planes.

**Integration residency:** the Packed1 option consumes the current engine's resident w1 and scales directly, leaving its F32R1 reader unchanged. The benchmark holds both layouts only for control comparisons; a native integration uses Packed1 and adds no resident weight copy. Engine/Makefile integration belongs to the native worker's `f122bbe`, separate from this experiment's write scope.

## Witness

`bench MODEL witness 8 OUTPUT` uses actual trained first96 output rows with supplied generic q cycling all256 signed bytes including−128/127, independent F32 K128 scales, R5/8/16/32, output stride128, N96 tails, initialized guards, alpha.75/beta.25. All schedules×split1/4/8 compare every output/guard against an independent rawQ1 integer-dot+FMA reference. Additional cases exercise supplied invalidstatus→NaN, original producer NonFinite/ScaleRange outcomes, persistent error surviving later validzero preparation, and allzero→zero projection.

## Reproduce

```
make -C experiments/binary-a8-fp4 -j1 KERNELS_OBJECT=/path/to/modern/kernels.o
kjob run binary-a8-fp4-round --owner-thread "$PI_THREAD_ID" --cwd "$PWD" -- \
  bash experiments/binary-a8-fp4/run-round.sh
python3 experiments/binary-a8-fp4/summarize.py
```

The reused producer object is needed only for the benchmark's staticWF8 control repack and full-head argmax. OriginalA8 producer/DP4A baseline are compiled from their owning source, not modified. Production of digits requires only `digits.o`, not the benchmark's full kernels object. Sources/receipts remain private in the campaign branch; remote ephemeral files `/workspace/binary/a8-fp4/` belong to the existing campaign rental, no new rental/service.
