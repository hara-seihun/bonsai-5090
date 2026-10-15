# NVFP4 / FP32 / INT32 overlap probe

`mma_overlap.cu` is a standalone, register-resident sm_120a probe. No model files,
loader or engine changes. It measures equal amounts of work under six schedules,
not just whether two kernels can be launched together.

## Build and run

CUDA 13.x (built locally with 13.1). Explicit `gencode` is required: CUDA 13's
`-arch=sm_120a` also emits plain compute_120 PTX, which rejects block-scaled MMA.

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v bench/mma_overlap.cu -o /tmp/mma_overlap
/tmp/mma_overlap --iters 4000 --repeats 7 --warps 4 --blocks-per-sm 2 \
  --chains 8 --scalar-steps 1 --case all --trace /tmp/overlap-spans.csv \
  > /tmp/overlap.csv
cuobjdump --dump-sass /tmp/mma_overlap > /tmp/overlap.sass
```

All seven workload arguments are required; unknown/duplicate arguments and CUDA
errors fail explicitly. Device 0 must be sm_120. `--help` lists cases and bounds.
`--case interleave_fma`, for example, runs that case plus all four isolated
references. Exit 4 means nonfinite output or a mixed-work checksum mismatch.

Sweep `--chains 1` and `8`, `--warps 4/8/16`, `--blocks-per-sm 1/2/4`, and
`--scalar-steps 1/4/16`. Start with eight chains and one scalar step for peak
issue rates. Increase iterations if launch overhead is material; compare stable
rates at 4000 and 16000 iterations. Runs rotate case order between repetitions;
CSV reports median, minimum and maximum, not just the most favorable sample.

## Work and schedules

Tensor work is `mma.sync...m16n8k64...mxf4nvf4...scale_vec::4X`: 8192 MACs per
warp instruction (16384 FLOPs). Packed FP4 operands are all +1, scales all +1;
each accumulator element adds 64 per invocation. FP32 work is a bounded affine
recurrence `f = fma(f, 0.99999994f, lane_dependent_increment)`. INT work is the
modulo-2^32 LCG/hash recurrence `i = i * 1664525 + 1013904223`, one `mad.lo.u32`.
These consume independent state and produce observable outputs; there is no
conditional rare-value sink. INT MAD throughput does **not** establish throughput
of every integer opcode (in particular LOP3, shifts, or DP4A).

Each chain is dependent across iterations; `--chains 8` creates eight independent
chains of each active kind. MMA has four FP32 accumulator registers per chain.
Scalar passes visit all independent chains before repeating them. Interleaving
issues MMA then one scalar update for each chain, then any additional scalar
passes requested by `--scalar-steps`. Volatile PTX prevents elimination; SASS
scheduling is still the compiler's. Inspect `OMMA.SF.16864`, `FFMA`, `IMAD` and
loop-control instructions in the actual build before attributing an issue limit.
There are no data loads in the measured arithmetic loops.

For suffixes `fma`, `int`, `fma_int`:

| Case | Schedule, with identical tensor and scalar operation counts |
|---|---|
| `tensor`, `fma`, `int`, `fma_int` | isolated references |
| `serial_*` | tensor kernel then scalar kernel, one stream; actual serial timing |
| `phases_*` | same warps, all tensor iterations then all scalar iterations |
| `interleave_*` | same warps, instruction interleaving within each iteration |
| `warps_*` | tensor-only and scalar-only halves of each CTA |
| `ctas_*` | alternating tensor-only/scalar-only CTAs, one kernel |
| `streams_*` | separate pure kernels in nonblocking streams, common start event |

Let `B = SM_count * blocks_per_sm`, `W = warps`, `I = iters`, `C = chains`,
`S = scalar_steps`. Every mixed case does `B*W*I*C` warp MMAs and, for each
selected scalar kind, `32*B*W*I*C*S` scalar instructions. An FMA is two FP32
operations; one INT MAD is reported as one instruction (multiply+add would be two
arithmetic operations). Counts exclude initialization, reductions, stores,
loop control, barriers and span instrumentation; timings include them.

`warps_*` launches B CTAs of 2W warps; `ctas_*` launches 2B CTAs of W warps;
`streams_*` launches B CTAs of W warps in each kernel. Same-warp cases launch B
CTAs of W warps. **Blocks-per-SM is requested grid geometry, not measured
residency.** CSV includes compiled registers/thread and the occupancy API's
maximum active CTAs/SM. For serial/streams rows these fields describe the scalar
kernel; the tensor kernel's fields are on the `tensor` reference row. Mixed CTA
kernels reserve registers for both paths; streams use separately compiled pure
kernels. This distinction is part of the measurement.

## Reading the output

- `Gwarp_MMA_s`: warp collective MMA instructions/s, not multiplied by 32.
- `TMAC_s`: that instruction rate times 8192; FP4 FLOP/s is twice this value.
- `TFLOP_s`: selected FP32 FMA rate times two.
- `GINT_MAD_s`: lane-level INT MAD instructions/s.
- `speedup_vs_isolated_sum`: `(isolated_tensor_ms + isolated_scalar_ms)/mixed_ms`.
- `time_vs_isolated_max`: `mixed_ms/max(isolated_tensor_ms, isolated_scalar_ms)`;
  ideal overlap at unchanged rates approaches 1, serialization approaches the
  isolated sum. Compare the measured `serial_*` row as well: it includes launch
  and transition costs that an arithmetic sum does not.
- Tensor/FP32/INT checksums are unconditional sums of every thread's output;
  integer accumulator-chain sums wrap to uint32 within a thread, then host sums
  use uint64. Every mixed checksum must exactly match isolated work. `fma_int`
  itself is also compared against the individual scalar references.
- CUDA event time covers the full launch/join interval. Resource queries and
  output downloads are outside it. Two streams wait on the common start event;
  control waits on both completion events before recording stop.

CTA spans record `%smid` and `clock64` after setup and before return (including
output-store/barrier overhead). The trace contains the final repetition of each
case, role 1=tensor, 2=scalar, 3=mixed CTA. For CTAs/streams/serial, the overlap
column is summed **per-SM intersection / per-SM union** of tensor and scalar CTA
intervals. Thus zero means no measured same-SM co-residency; positive means real
co-residency, not necessarily simultaneous instruction issue. Only cycles from
the same SM are compared. For mixed CTAs the column is -1: a whole-CTA span cannot
distinguish its phases/warps. Use counters to resolve the pipe behavior.

## Hardware counters

On a profiling-enabled Vast VM (see `tools/STRUCTURE.md`, hardware counters), run
as root if the driver requires it:

```sh
bash bench/mma_overlap_ncu.sh /tmp/mma_overlap /tmp/ncu-interleave-fma \
  --iters 4000 --repeats 1 --warps 4 --blocks-per-sm 2 \
  --chains 8 --scalar-steps 1 --case interleave_fma
```

Set `NCU=/path/to/ncu` if not on PATH. The helper profiles the first matching
warm-up of the requested specialization and exports `.ncu-rep`, raw/details CSV
and a log. Repeat for isolated `tensor`, `fma`, `int`, and for `warps_*` and
`ctas_*`. Sections expose issue-active/eligible warps, instruction mix, tensor,
FMA/ALU pipe utilization, stalls and occupancy. Metric availability/names depend
on Nsight/driver; the raw CSV preserves the actual counters.

Do **not** infer concurrent-stream speed from kernel-replay ncu timings: profiling
can serialize kernels. Run event timings and CTA-span traces without ncu; profile
the two pure kernels separately for their pipe demand. Nsight Systems can further
show the two-stream timeline. No GPU has been rented or run by this subtask;
local compilation and SASS inspection are the delivered preparation.
