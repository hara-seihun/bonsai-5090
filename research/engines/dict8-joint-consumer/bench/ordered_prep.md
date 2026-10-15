# Exact ordered fold preparation

**Keep the existing serial route for serving.** All three experimental modes are exact but
slower than the original serial fold on the native RTX5090 screen. The original serial fold remains selected
when `H5_PREP_ORDERED_PIPE` is unset or `0`.

`ordered-prep-bench` isolates the real D-wide down projection and its row-norm preparation.
It compares the separate deterministic reducer plus ordinary wide prep, the serial folded
prep, and compile-specialized four-plane folds with and without a source-level warp-sync
boundary. Every fold starts with plane zero, adds ascending split planes using `__fadd_rn`,
then adds the residual once. Normalization, Hadamard, scales and quantization are unchanged.

## Selection and reproduction

- `H5_PREP_ORDERED_PIPE=0` (or unset): existing serial fold.
- `H5_PREP_ORDERED_PIPE=1`: unfenced specialization for split6 or split17.
- `H5_PREP_ORDERED_PIPE=2`: specialization with `__syncwarp()` between each four-plane load
  batch and its adds. This is a source-level scheduling experiment, not a hardware barrier
  guarantee: CUDA12.8 removes the physical warp barrier.
- `H5_PREP_ORDERED_PIPE=3`: unchanged runtime serial fold body with only its kernel launch
  bound changed from 1024 to 640. This isolated spill-cap control also loses to the original.

Other values, or other split counts with a specialized route selected, return
`cudaErrorInvalidValue`. The choice is read at launch/graph capture. The specialized kernel
has a 640-thread launch bound matching the validated 5120-wide input; generic wide prep
retains its existing bound. Existing ordered-fold engine switches still decide whether this
preparation is reached; this switch does not enable down folding by itself.

Build with the integrated definitions (CUDA12.8, sm_120a in the native screen):

```sh
defs="$(python3 tools/selected_build_defines.py results/integrated/selected-preset.json)"
make DEFS="$defs" ordered-prep-bench
flock /root/gpu-round.lock ./ordered-prep-bench
# Isolated original-body launch-bound control, omitting the rejected specializations:
flock /root/gpu-round.lock ./ordered-prep-bench --bound640
```

The benchmark checks residual FP32 bytes, all allocated q4/s4 bytes (including untouched
padding), and live g4 bytes against the separate deterministic route before any timing.
Five distinct weight/residual inputs are checked for R8/R16 and K6144/K17408. Every check
passed in both screens. It uses actual compact ternary WF3/carrier GEMMs, selected integrated
build definitions, 32 distinct rotating weight allocations, and three paired ABBA blocks for
every arm pair. The rings are 204,472,320 and 579,338,240 bytes, both larger than the RTX5090's
96 MiB L2. The earlier eight-copy K6144 ring was only 51,118,080 bytes, so it did not enforce
cold weights.

## Native result, October 9 2026

Mean paired ABBA deltas in microseconds; positive is slower. Each table entry is reduced
from its own matched arm pair, not subtraction of unrelated global averages.

| Rows | Splits | Serial − separate | Unfenced − serial | Sync-boundary − serial |
|---:|---:|---:|---:|---:|
| 8 | 6 | +0.047 | +0.557 | +0.208 |
| 8 | 17 | +2.576 | +1.246 | +1.080 |
| 16 | 6 | +0.017 | +0.704 | +0.240 |
| 16 | 17 | +2.206 | +1.369 | +1.124 |

For R8/split17, the existing serial loss is reproducible: 18.912us separate versus 21.488us
serial. The specialized sync-boundary route loses another 1.080us against serial in all
three paired blocks (1.076–1.085us). The source-level sync boundary recovers 0.160us against
the unfenced specialization, not the original fold's loss. The first unfenced-only screen
independently measured +1.261us versus serial for this shape.

### Mechanism and corrected hypothesis

The runtime serial loop already auto-unrolls four iterations in SASS and overlaps multiple
plane loads. It is not simply one dependent load/add repeated seventeen times. Ptxas freely
reschedules the unfenced specialized C++ load groups into a near per-split load/add stream.
Adding `__syncwarp()` changes the generated load schedule and register lifetimes, but there
is no `WARPSYNC` or `BAR.WARP` instruction in either specialization; grouping still does not
enforce four complete plane loads before the first addition.

All kernels use 33,024B shared memory. Ptxas resource counts:

| Route | Registers | Spill stores | Spill loads |
|---|---:|---:|---:|
| Serial | 64 | 16B | 32B |
| Unfenced split6 | 48 | 4B | 4B |
| Unfenced split17 | 58 | 0 | 0 |
| Sync-boundary split6 | 62 | 0 | 0 |
| Sync-boundary split17 | 64 | 0 | 0 |
| Original runtime body, bound640 | 66 | 0 | 0 |

Removing the loop/address liveness and spills did not help latency. The measured schedule
change matters more than these resource reductions. This rejects these source regrouping,
compile-specialization and warp-sync-boundary routes; it does not establish a floor for
stronger memory-issue mechanisms.

### Isolated unchanged-body launch-bound control

A separate native screen finished at 02:27 UTC. Only the ordered kernel's launch bound
changed to 640; its runtime serial fold body, split count, accumulation order and prep
arithmetic stayed unchanged. Five distinct exactness cases per shape all passed.

| Rows | Splits | Bound640 − original serial | Three paired ABBA block range |
|---:|---:|---:|---:|
| 8 | 6 | +0.157us | +0.150–0.171us |
| 8 | 17 | +1.357us | +1.353–1.364us |
| 16 | 6 | +0.303us | +0.297–0.312us |
| 16 | 17 | +1.505us | +1.496–1.511us |

The relaxed cap eliminates all original spills at 66 registers, yet loses more than a
microsecond for split17. The compiler's launch-bound/resource constraint itself affects the
native schedule: removing spills is not independently a win. This control closes the
confounding question left by the specialized routes and does not support cap relaxation.

Measurements include residual initialization, GEMM and prep. They omit the following GU
GEMM and the rest of the round: the coordinator owns the coupled successor/full-round
result. The microbenchmark is a local mechanism screen, not full-round adoption.

## Artifact custody

Native source/build/run receipts and reduction tooling are under
`~/work/tune-5090/round/ordered-pipe/`. `summary.json` holds the first three-arm
screen; `v2/summary.json` holds the four-arm screen. All three numeric EXIT receipts are `0`; `bound640/summary.json` holds the isolated control.
`v2/ordered-pipe-native-artifacts.tgz` preserves the native binary, reused fp4 object,
compiled kernels, SASS, build flags, hashes, raw measurements and exit receipt. Immutable
source closures are `source.tgz`, `ordered-pipe-v2-source.tgz`, and
`ordered-pipe-bound640-source.tgz`. `bound640/ordered-pipe-bound640-native-artifacts.tgz`
preserves the isolated-control native binary, objects and receipts.

Remote source paths are `/root/ordered-pipe-source` and `/root/ordered-pipe-source-v2` on
existing VM54937744; the isolated control uses `/root/ordered-pipe-source-bound640`.
All three pod-owned units finished; no job was rerun. The durable local
observer was repaired to recognize loaded/inactive units whose start timestamp resets to
zero, consuming the pod-owned receipt rather than waiting forever.
