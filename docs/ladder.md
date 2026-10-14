# From PrismML's llama.cpp to this engine, one change at a time

Every number on this page is a measured median of three passes. Nothing is extrapolated, and
changes without their own measurement are marked as such. Raw per-pass values, the exact
runtime environment of each row and the model hash are in [`../receipts/ladder.json`](../receipts/ladder.json).

**Two machines, two windows.** Rows 0–6 were measured on one rented Linux RTX 5090 (driver
580.178.04, 600 W): 64 streams, 1,024 timed tokens per stream. The final row was measured later on a
different RTX 5090 under Windows with 128 timed tokens per stream (that receipt does not record
driver or power limit). That is the
build this repository reproduces. Compare rows within the ladder. Don't divide the final row by the
baseline; the two were never measured under the same conditions.

All rows use the same weights (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, sha256 `53107f53…`), the same
prompt fixture (the first 64 prompts of `data/video-gsm-prompts256.bin`), greedy decoding, no
speculation, and 8 untimed warm-up tokens per stream. Load, weight conversion, plan tuning and prefill
fall outside the timer.

## Scoreboard

| Row | Change | B64 tok/s | vs previous main row |
|---|---|---:|---:|
| 0 | PrismML llama.cpp fork, unmodified kernels (baseline) | 930.6 | — |
| 1 | GPU batch token selection, one token-only readback (01a) | 1,006.8 | +8.2% |
| 1b | Embedding table resident on the GPU (01b) | 1,009.3 | +0.2% |
| 1c | Packed ternary embedding reader (01c) | 1,013.3 | +0.4% |
| 2 | Direct recurrent state + 8-lane cooperative update (02a+02b) | 1,238.9 | +22.3% |
| 3 | FP16 recurrent state storage (03a–c) | 1,384.9 | +11.8% |
| (side) | FP4 input projections only, still inside llama.cpp (04-input) | 1,446.2 | +4.4% vs row 3; not on the main line |
| 4 | Complete fused FP4 engine (04a–k, measured only as a bundle) | 4,579.7 | +230.7% vs row 3 |
| 5 | Six-bit recurrent state + single-token update (05a+05b) | 6,118.4 | +33.6% |
| (5b) | FP4 KV cache, fragments still staged through shared memory (06a only) | 6,284.2 | +2.7% vs row 5 |
| 6 | FP4 KV read directly into attention fragments (06a–c) | 6,446.4 | +5.4% vs row 5 |
| **Final** | **Row 6 engine as built here, Windows, 128-token window** | **6,777.4** | different machine and window, see above |
| Final, deterministic | Same build, `H5_DETERMINISTIC_GEMM=1` | 6,583.1 | −2.87% vs final |

Row 6 against row 0, same machine and window: **6.93×**.

"vs previous main row" is the measured ratio between adjacent rows. It is not an independent
contribution of that change. Later changes alter what earlier ones cost, so these percentages don't
add up and can't be reordered.

## Why batch-64 decode is slow to begin with

One decode step at 64 streams reads every weight once and does 64 small dot products against it.
Bonsai 2 27B is a hybrid: 48 Gated-DeltaNet (linear-attention) layers carry a 128×128 recurrent
matrix per head per stream, and 16 ordinary attention layers carry a KV cache. A step therefore moves
three kinds of bytes: weights (shared by all streams), recurrent state (per stream, read and written
every step), and KV cache (per stream, growing). The baseline also pays fixed per-step costs: host
round trips, kernel launches, and format conversions between kernels. The ladder removes those costs
roughly in order of how cheap they were to remove.

## The changes

Each unit has a code view in [`steps/`](steps/): an excerpt patch from the PrismML source, or the
previous step, to the measured source. The excerpts are review views, not apply-able production
patches. Line counts describe the excerpts, not effort.

### Row 1 — sampling and readback (01a, 01b, 01c)

- **01a-sampling.** The baseline selects each stream's greedy token through the per-sequence sampler
  chain, then copies results back row by row. The change reduces all 64 argmaxes on the GPU in one
  batched kernel and copies back 64 int32 token IDs (256 bytes) in one asynchronous transfer. It's
  faster because per-row host work and synchronizations disappear from every step. Measured:
  930.6 → 1,006.8.
- **01b-embedding-placement.** Places the packed embedding table in GPU memory so the lookup output
  feeds the next GPU stage without a host hop. Measured: 1,009.3.
- **01c-embedding-reader.** Reads eight natural-order ternary coordinates per thread directly from
  the packed PTQ1 rows (28-byte blocks of 128 trits plus an FP16 scale). Measured: 1,013.3.

### Row 2 — recurrent update (02a, 02b; measured together)

- **02a-direct-state.** When each stream's state slot maps to itself, the update reads and writes
  state in place, with no gather into a temporary and scatter back. The guarded gather path remains
  for other layouts.
- **02b-cg8-update.** Eight lanes cooperate on one row of the 128×128 state, matching how the state
  is laid out in memory, so loads are coalesced and the per-row reduction is a short shuffle.
- Why faster (our reading, not separately measured): the recurrent state is the largest per-stream
  traffic at B64, and both changes cut redundant passes over it. Measured together: 1,013.3 → 1,238.9. The two units have no separate
  timings.

### Row 3 — FP16 state (03a, 03b, 03c; measured together)

Stores the recurrent matrices as FP16 (32 KB per head instead of 64 KB) while keeping FP32 arithmetic
inside the update. Half the state bytes cross memory every step. This is lossy: rounding happens
between steps. Measured: 1,238.9 → 1,384.9. Allocation, load/store and initialisation (03a–c) have no
separate timings.

### Side row — FP4 input projections inside llama.cpp (04-input)

A measured check on the modular branch: only the input projections use FP4 Tensor Core GEMMs while
everything else stays in llama.cpp. 1,446.2. It shows that FP4 arithmetic on its own, still paying
the framework's format conversions and launches, buys little. The big step comes from the engine.

### Row 4 — the fused FP4 engine (04a–k; one measurement for the whole bundle)

These eleven units only work together, so they were measured once, as a bundle:
1,384.9 → **4,579.7**. None has its own number.

- **04a weight layout.** At load time, ternary weights are repacked into exactly the per-lane order the
  matrix instruction consumes: two-bit trits ("WF0") for most matrices, five trits per byte
  ("Dense5") for the vocabulary head. Conversion happens once, outside the timer. Weights stay about
  as small as the GGUF; they are never expanded to FP16.
- **04b operand map.** Masks and shifts turn packed trits directly into FP4 (E2M1) operand registers.
  The baseline also decodes packed trits vectorised, but into INT8 lanes for an INT8 kernel.
- **04c activation prep.** A single kernel performs RMS normalisation, the sign flip, the Hadamard
  rotation and FP4 packing with block scales, then writes exactly the layout the GEMM reads. Prism
  already fused FWHT+Q8; this changes format, layout and integration. FP4 activations are lossy.
- **04d tensor core.** Uses sm_120a block-scaled FP4 MMA (`kind::mxf4nvf4.block_scale`). Per byte
  moved, FP4 operands carry twice as many products as INT8.
- **04e scale carrier.** Applies a per-row mean weight scale once in the epilogue instead of a scale
  per block inside the reduction. This is an approximation of the original block scales.
- **04f projection layout.** Projections that share an input are stored and computed as one matrix,
  with gate/up interleaved, so the activation is prepared once and read once.
- **04g output fusion.** The GEMM epilogue computes SiLU(gate)×up and adds residuals directly, so
  intermediates aren't written and re-read.
- **04h copy pipeline.** A shared-memory ring with asynchronous copies and completion barriers keeps the
  next weight tile in flight while the current one is multiplied.
- **04i recurrence integration.** Convolution and the recurrent update read and write the engine's own
  buffers. Convolution and recurrence remain separate kernels (`H5_STATE_FUSION=0`).
- **04j attention.** Query/key normalisation and RoPE are fused with the cache append, and a custom
  attention kernel reads an FP8 KV cache (FP16 Tensor Core products).
- **04k execution.** The engine owns one CUDA graph per shape, replays it each step, uses frozen GEMM
  geometry from `engine/frozen-production-plan.txt`, overlaps launches with programmatic dependent
  launch (PDL), and runs the small alpha/beta projections beside the large input projection.

Why it's faster: each step now moves roughly the minimum weight bytes, in a format the Tensor Cores
consume without conversion, and launch and conversion overhead between layers largely disappears.

### Row 5 — six-bit state (05a, 05b; measured together)

- **05a state codec.** Each recurrent row is stored as six-bit signed codes in three 32-bit word
  planes with a least-squares row scale. Each lane loads 16 codes from three words (funnel shifts at
  word boundaries), updates in FP32, and repacks. DP4A computes the integer sum of squares for the
  scale fit. The allocation still uses the Q8 container size; the bytes actually moved shrink.
- **05b single-token update.** At decode, each stream adds exactly one token, so a kernel specialised
  for one new token per segment is selected (CG8, FP32 math, occupancy policy `H5_GDN_OCC=2`). B64 means 64
  independent one-token updates per step.
- Why faster: after row 4, state traffic dominates again, and six-bit codes move fewer bytes than FP16.
  Lossy. Measured together: 4,579.7 → 6,118.4.

### Row 6 — FP4 KV cache (06a, 06b, 06c)

- **06a kv writer.** Writes K and V as packed E2M1 (two per byte) with one E4M3 scale per 16 channels,
  during attention preparation: 144 bytes per head vector instead of 260. Measured with the reader still
  staging through shared memory: 6,284.2.
- **06b kv reader.** Converts packed bytes and scales straight into FP16 MMA fragments in registers. No
  half-precision tile is materialised, and shared memory per CTA shrinks (40,960 bytes at MT1 against
  the FP8 reader's fixed 94,464).
- **06c wiring.** Switches both attention entry points together.
- Why faster: fewer KV bytes per position, less shared-memory staging, and more CTAs resident. Lossy.
  Measured: 6,118.4 → 6,446.4 (06b and 06c have no separate timings).

### Final row — this repository's build

The row-6 engine, packaged as a standalone library: the source in [`../engine`](../engine) and the
settings in [`../profiles/serving.json`](../profiles/serving.json). The only source change from the
row-6 configuration is that FP4 KV slot admission accepts up to 256 slots (`engine/dispatch.cu`,
`slots()`); arithmetic and plan are unchanged, and the measured run uses 64. Measured on the Windows
RTX 5090 with 128 timed tokens per stream: 6,778.4, 6,777.4 and 6,775.7, median **6,777.4 tok/s**.
A fresh build of this repository on the same GPU measured 6,731.7 tok/s (see
[reproduction.md](reproduction.md)).

`H5_DETERMINISTIC_GEMM=1` replaces split-K float-atomic folds with a fixed reduction order. Repeated
runs then produce bit-identical whole-model outputs. Ordinary mode varies across runs in the last
bits, because split-K atomics complete in whatever order the hardware schedules. Measured:
6,585.4, 6,583.1 and 6,579.8, median **6,583.1 tok/s** (−2.87%).
