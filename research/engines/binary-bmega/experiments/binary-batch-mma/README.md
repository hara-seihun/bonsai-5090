# Batch MMA standalone benchmark

`bench.cu` compares the original Q1_0 trained signs/FP16 K128 scales through `batch_mma` against the stable `bitserial::Route::Dp4a01` projection. There is no persistent expanded weight representation. Synthetic activations are deterministic, nonconstant, and distinct across token rows.

## Build

CUDA 13.1 and the campaign's existing sm_120a GPU are required. Only the benchmark, batch kernel, stable bitserial kernel, and GGUF reader are compiled; no engine build is involved.

```sh
# From the source checkout:
bash experiments/binary-batch-mma/build.sh /absolute/path/bench "$PWD"

# Existing campaign mirror; reuse the owner's freshly compiled batch kernel:
cd /workspace/binary/batch-mma
MMA_OBJECT=$PWD/batch_mma.o bash experiments/binary-batch-mma/build.sh \
  "$PWD/bench" /workspace/binary/halo
```

`NVCC` may explicitly name another CUDA 13.1 compiler. `MMA_OBJECT`, when set, must name an existing object; otherwise the script compiles `batch_mma.cu` in this directory.

## Run

Compilation does not run CUDA work. The campaign owner runs timings under its existing GPU lock:

```sh
cd /workspace/binary/batch-mma
model=/workspace/binary/models/Bonsai-27B-Q1_0.gguf
flock /tmp/binary-gpu.lock ./bench "$model" gu all 0 gu
flock /tmp/binary-gpu.lock ./bench "$model" down all 0 down
flock /tmp/binary-gpu.lock ./bench "$model" head all 65536 head65536
flock /tmp/binary-gpu.lock ./bench "$model" head all 0 head248320
```

Arguments: `MODEL TENSOR R N_ROWS PREFIX`. `R` is `5`, `8`, `16`, `32`, or `all`. `N_ROWS=0` uses all trained rows; a positive value uses the leading trained rows. Aliases: `gu` concatenates `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight` in that order; `down` selects `blk.0.ffn_down.weight`; `head` selects `output.weight`. Exact GGUF tensor names are also accepted. With only `MODEL`, the executable lists Q1_0 rank-two tensors. Expected campaign geometries are GU34816x5120, down5120x17408, head65536x5120 and head248320x5120.

Each invocation writes `PREFIX.jsonl`, flushing every completed record. Errors exit nonzero; partial files are not complete results. A 50-second wall budget bounds correctness and timing work after GGUF loading. Exit codes: 1 invalid arguments/geometry, 2 CUDA/model/file error, 3 numerical/guard/status failure, 4 elapsed budget.

## Measurement and correctness

- Nine fixed candidates: output tiles32/64/128 × K128 group chunks4/16/all.
- Inclusive CUDA graphs contain one full-R carrier producer plus projection, DC correction, original K128 scales, and split-K continuation. Carrier-only graphs omit the producer. Timing uses tightly packed output, alpha1/beta0, so graph replays do not accumulate previous outputs.
- The paired stable comparator also produces the full-R carrier once. It consumes row views of at most16 tokens using `bitserial::select` for each view, with separate DP4A partial allocations. It never re-quantizes individual row chunks.
- Warm and cold GPU-event timings are medians of7 samples after2 discarded samples. Each cold trial touches128MiB before the start event, outside the measured interval. Setup, packing, allocations, graph construction, checks, and host copies are excluded.
- Every full candidate output is checked against the stable DP4A exact-A8 output, reporting maximum absolute error, absolute RMS, and relative RMS. Association tolerance is relative RMS≤1e-5 and maxabs≤1e-4+3e-5×maxabs(reference). Nonfinite outputs or invalid producer statuses fail immediately.
- Before timings, each candidate runs N96 with R5 and R32, output stride128=N+32, alpha.75/beta.25, nonconstant initial outputs, and prefix/row-padding/trailing token guards. An independent GPU oracle reads raw Q1_0 blocks and quantized signed A8 bytes directly; the stable DP4A result is also checked against it. These records use `kind=correctness`; projection timing records use `kind=timing`.

## Native map and engine boundary

Four `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` instructions per K128
compute `P = sum_i w_i*q_i`, with original sign bits `w_i` relabeled to 0/1
operand bytes by `(nibble * 0x00204081u) & 0x01010101u`. Expanded weights exist
only in registers. The correction `2*P - DC`, `DC = sum_i q_i`, recovers the
signed binary dot exactly. Reset integer accumulators at each K128, then continue
with FP32 `fma(float(dot), original_half_weight_scale * activation_scale, sum)`.
The integer map and nibble relabeling are the existing kernel-checked
[Lean witness](../binary-bitserial/witness/README.md); native fragment addressing
is exercised separately by this benchmark. Exact A8 is not exact original FP32.

`project_carrier()` still contains per-CTA activation shared-memory staging,
register expansion, DC correction, original scale multiplication and continuation.
Only the carrier producer is omitted. This distinguishes actual producer sharing
from claiming a bare MMA time as a complete projection.

`batch_mma.cuh` owns `Plan{n,k,r,groups_per_split,n_tile}`, `select`, `validate`,
`partial_bytes`, `project`, and `project_carrier`. Accepted geometry is N32/K128,
R1..32; M16 token tails and N96 feature tails are masked. Output stride `ldo >= N`
is explicit. Caller owns the carrier and `[R,split,N]` FP32 partials. Launches
allocate nothing, synchronize no host thread and configure no function, so are
capture-safe. Invalid producer status produces NaN and retains the producer's
aggregate error for the engine's normal end synchronization. Invalid plans,
missing buffers and invalid output stride return `cudaErrorInvalidValue`.

sm120a lowering is native `IMMA.16832.S8.S8`: 52 registers for R<=16, 60 for
R>16, zero spills, 2240/4480 shared bytes. Stable DP4A and exact-FP32 sources stay
unchanged. The parent owns full-model quality and inclusive engine timing.

## Result: retain DP4A

The locked round completed successfully (55-second post-lock timeout). All full
outputs and 72 N96 alpha/beta/stride guard cases passed; largest observed absolute
error was 6.68e-6. `results/*.jsonl` retain every candidate and `summary.json` the
cold winners. Original trained Q1 weights, deterministic distinct FP32 inputs.

| Geometry, R32 | DP4A inclusive cold µs | Best MMA µs | tile/chunk |
|---|---:|---:|---|
| GU34816×5120 | 112.640 | 124.672 | 32/16 |
| down5120×17408 | 65.504 | 69.344 | 128/16 |
| head65536×5120 | 190.464 | 223.232 | 32/16 |
| head248320×5120 | 724.992 | 794.624 | 64/40 |

R8 GU/down also lose: 45.056→71.648 and 28.672→43.008 µs. Warm R32 GU/down show
small gains (109.568→105.824 and 58.496→54.816 µs), insufficient to recommend this
complete cold projection path. No engine integration. `select()` returns the
best measured cold MMA candidate for these four geometries at R5/8/16/32; other
valid geometries return `cudaErrorNotSupported`. Explicit plans still exercise
the generic consumer. This helper is not a recommendation to replace DP4A.

`device-source-receipt.txt` binds the actual timed consumer object/executable and
stable DP4A source; subsequent edits affect only host plan selection/validation.
The source remains a correct native-MMA negative, not a claimed acceleration.
