# Shared-weight binary residual FP4 pair

Opt-in native consumer for the near-quality **two-plane** route. Source/API:
[`pair.cuh`](pair.cuh), [`pair.cu`](pair.cu). The single-plane approximation is not substituted.

## Decision

On RTX5090, 500W, CUDA13.1.115, this bounded inclusive projection experiment wins all nine measured trained geometries. Engine integration/full-model quality remain with the campaign parent; these are not token/s measurements.

| Projection | R | Warm ordered → joint µs | Cold ordered → joint µs | Cold speedup | Joint cfg/split |
|---|---:|---:|---:|---:|---:|
| down | 1 | 40.224 → 33.120 | 46.848 → 38.912 | 1.204× | 2/16 |
| down | 8 | 44.096 → 36.928 | 48.608 → 40.960 | 1.187× | 1/16 |
| down | 16 | 50.432 → 41.856 | 59.136 → 51.200 | 1.155× | 1/8 |
| gu | 1 | 50.368 → 38.112 | 63.232 → 50.944 | 1.241× | 2/1 |
| gu | 8 | 58.304 → 42.976 | 67.328 → 49.120 | 1.371× | 1/1 |
| gu | 16 | 65.824 → 48.224 | 71.392 → 53.248 | 1.341× | 3/1 |
| full head | 1 | 252.800 → 140.608 | 262.144 → 147.200 | 1.781× | 2/8 |
| full head | 8 | 272.096 → 145.344 | 284.704 → 165.888 | 1.716× | 1/4 |
| full head | 16 | 294.560 → 164.192 | 313.344 → 182.272 | 1.719× | 6/1 |

Each cell compares the fastest independently swept schedule/split for that cache condition, not necessarily the same configuration as the other column. Raw 504 records are in [`results/`](results/); [`summary.json`](summary.json) records the winners and correctness envelopes. No losing tested configuration is deleted.

## Composition and numerical scope

One warp-specialized eight-stage TMA ring streams each prepared WF8 weight chunk and its original FP16 K128 scales **once**, alongside both existing activation operands and their existing SF bytes. Consumer warps expand each binary sign fragment once, then reuse it for both planes' two K64 MMAs. For each K128 block and output element:

```
primary_dot = MMA(primary half0, primary half1; existing primary SF)
residual_dot = MMA(residual half0, residual half1; existing residual SF)
pair = fma(residual_dot, residual_row_g, primary_dot * primary_row_g)
acc = fma(pair, original_FP16_weight_scale, acc)
```

The corresponding existing route accumulates each plane independently across K128, multiplies each finished sum by its row-g, then adds the two results. Distributing row-g and summing before the original weight scale changes **FP32 association**. No quantized operand, SF field, original weight scale, sign, or residual producer is changed.

Across all joint configurations, maximum projection error versus existing ordered two-pass output is 2.38418579e-6; maximum relative RMS is 2.20464537e-7. The benchmark checks every projected element, rejects nonfinite differences, and compares 16 sampled output columns per row against an independent FP64 sum of decoded existing planes and original Q1 blocks. It also checks `add=true` for every joint schedule/split, exactly against the same joint result plus independently initialized output. This establishes projection and additive contracts, not full-model token agreement.

## Inclusive boundary

Weights and normalization vectors are from `Bonsai-27B-Q1_0.gguf`, SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`. Inputs/residual vectors are deterministic synthetic finite activations; they are **not captured model activations**.

* GU: layer0 gate/up, N34816/K5120, original interleaved 32-column gate/up groups. Existing `launch_prep` applies trained post-attention RMSNorm, writes FP32 transformed input and produces primary NVFP4; existing residual prep produces plane2. Projection continues through SiLU×up and existing primary/residual quantization of the next activation (width17408).
* Down: layer0 down, N5120/K17408. Existing SiLU-multiply producer, FP32 transformed input, primary and residual producers; projection; residual add; trained next-layer attention RMSNorm and next primary/residual producers (width5120).
* Head: full trained output matrix, N248320/K5120. Trained final RMSNorm, primary and residual producers, projection, existing full-vocabulary GPU argmax.

The SiLU-interleave and residual-add continuations are explicit experiment kernels followed by the existing producer. Both arms pay these same kernels. The down benchmark deliberately times projection then common residual add; the API also has an additive epilogue for native integration. The original engine can associate its initial residual, primary, and second-plane additions differently. All scale/output work and both activation producers recur **inside** the timed graph. Static GGUF mapping, weight repacking, allocation, graph creation and host configuration are outside.

Timing is CUDA graph/device elapsed time, 3 discarded warmups + median10 samples. Cold runs evict128MiB immediately before the inclusive graph; warm and cold are reported separately. Timings use `/tmp/binary-gpu.lock`, with no simultaneous GPU timing. Seven schedules × splits1/4/8/16 are swept for each arm. Existing `src/fp4.cu` kernels are instantiated directly for the baseline; no baseline source/dispatch is modified. The finite baseline set includes production-shaped TOK16/TOK32 and diagnostic TOK8 schedules.

## Stable API and schedules

```
project(plan, primary, residual, prepared_WF8, original_FP16_scales,
        output, output_stride, partials, partial_capacity, stream, add)
```

* `Plane` is three caller-owned device pointers q/SF/g in the modern engine's `[tile64][K128][row64]` swizzled layout. Pad rows to64, including zero unused rows as in the existing engine. Immutable weights use WF8 512B chunks and64B scales per chunk. q/SF/W/S pointers must be16-byte aligned; g/output/partials float-aligned.
* Supported: R1..16, positive N divisible32, positive K divisible128, output stride≥N; splits1..K/128. Arbitrary valid N/K are structurally supported but only the three geometries above have timing receipts. R>16, other physical weight formats, altered weight scales and incompatible dimensions are unsupported and rejected, not silently routed elsewhere.
* `partial_floats(plan)` is the scratch requirement for a **valid** plan. Split1 requires no scratch. Split>1 requires caller-owned, non-overlapping `splits*R*N` FP32 scratch and folds ascending split order with RN additions. `add=false` overwrites; `add=true` adds after this deterministic projection/fold. No atomic split reduction.
* Async CUDA errors/status are returned as `cudaError_t`; caller owns stream completion and buffer lifetimes. The function neither allocates nor performs activation preparation.

Cfg IDs used in receipts:

| cfg | enum | consumer warps | weight groups/warp | token tile |
|---:|---|---:|---:|---:|
| 0 | F2G2T8 | 2 | 2 | 8 |
| 1 | F4G2T8 | 4 | 2 | 8 |
| 2 | F8G1T8 | 8 | 1 | 8 |
| 3 | F4G2T16 | 4 | 2 | 16 |
| 4 | F8G1T32 | 8 | 1 | 32 |
| 5 | F4G2T32 | 4 | 2 | 32 |
| 6 | F8G2T16 | 8 | 2 | 16 |

The cold winners in the headline table are the first schedules to try when the model streams weights. For other row counts/shapes, caller must choose an explicit schedule/split or run its own bounded tuning; this API invents no default.

## Build and reproduce

```
make -C experiments/binary-fp4-pair -j1
flock /tmp/binary-gpu.lock env H5_PDL=1 H5_PREP_CHUNK=0 H5_PREP_WIDE=0 \
  experiments/binary-fp4-pair/build/bench MODEL.gguf gu 1 gu-r1.jsonl
python3 experiments/binary-fp4-pair/summarize.py
```

Repeat `gu|down|head` at R1/8/16 with unique receipts. Full head only is measured; a pruned head is not separately qualified. The benchmark's argmax expects the complete248320 vocabulary.

Production needs only `pair.cu` as an additional object plus `pair.cuh`; it does not include or instantiate `src/fp4.cu`. Standalone build compiles bounded baseline FP4 schedules and uses the ordinary producer object. The measured executable reused `/workspace/binary/modern/halo/build/kernels.o` with identical producer source hash rather than recompiling all unrelated GDN kernels. [`results/provenance.json`](results/provenance.json) owns those hashes and exact flags. Root source and existing numerical receipts remain intact.

Remote ephemeral experiment files are `/workspace/binary/pair/`, on the campaign's existing GPU disk. Exported source/results in this directory survive campaign teardown; no additional rental or service was created.
