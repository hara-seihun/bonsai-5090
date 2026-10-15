# Packed arithmetic beside NVFP4 MMA (sm_120a)

`packed_ops.cu` measures register-resident FP32, FP16×2, BF16×2 and integer
instructions alone and in the same warps as FP4 tensor MMA. It does not change
production kernels. [The hotspot ledger](packed_ops_hotspots.md) identifies
where the instruction results can remove engine work.

## Build / run

CUDA 13.1 build and SASS inspection completed locally; **no GPU run or rental**.
All 104 specializations compile with zero spill loads/stores. Runtime results
are intentionally absent until the campaign's existing VM runs them.

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v bench/packed_ops.cu -o /tmp/packed_ops \
  2> /tmp/packed_ops-build.log
cuobjdump --dump-sass /tmp/packed_ops | c++filt > /tmp/packed_ops.sass
/tmp/packed_ops --iters 4000 --repeats 7 --warps 4 --blocks-per-sm 2 \
  --chains 8 --scalar-steps 1 --op all > /tmp/packed_ops.csv
```

Explicit `gencode` avoids CUDA 13's plain compute_120 PTX companion rejecting
block-scaled MMA. Device 0 must be sm_120. Every workload argument is required;
unknown/duplicate options, invalid ranges, CUDA errors and checksum mismatches
fail explicitly. `--help` is GPU-free. Exit 4 is a value/domain mismatch.

Start with all ops at eight chains, then sweep `--chains 1|8`,
`--warps 4|8|16`, `--blocks-per-sm 1|2|4`, `--scalar-steps 1|4|16`.
`--op bf16x2_fma`, for example, runs only that op's three schedules plus tensor.
Repeat at 16000 iterations to identify fixed launch/setup/store cost. Blocks per
SM is **requested grid size**, not measured residency. CSV includes compiled
registers/thread and the occupancy API's maximum active CTAs/SM for each kernel.

For counters on the profiling-enabled VM (root if required by its driver):

```sh
ncu --kernel-name-base mangled \
  --kernel-name 'regex:_Z13packed_kernelIL2Op4ELi8EL8Schedule3' \
  --launch-count 1 --section SchedulerStats --section WarpStateStats \
  --section InstructionStats --section ComputeWorkloadAnalysis \
  --section Occupancy --force-overwrite --export /tmp/packed-bf16x2 \
  /tmp/packed_ops --iters 4000 --repeats 1 --warps 4 --blocks-per-sm 2 \
  --chains 8 --scalar-steps 1 --op bf16x2_fma
```

The regex selects BF16 FMA, eight chains, interleave, including its first warmup.
Use Op IDs in the SASS table below and Schedule 1/2/3 for isolated/phases/interleave.
Use unprofiled CUDA-event CSV for speed; counters identify pipe conflicts and
issue/stall demand, not instruction-pipe independence merely from opcode names.

## Equal work, bounded values, independent state

- `tensor`: all tensor iterations, no scalar work.
- `isolated_OP`: all scalar iterations, no tensor work.
- `phases_OP`: tensor loop then scalar loop in the same kernel/warps. It retains
  both kinds' register state, unlike adding two isolated timings.
- `interleave_OP`: each iteration visits all chains, emitting one MMA and one
  scalar step per chain; then emits any additional scalar steps. No tensor result
  feeds a scalar accumulator, and no scalar result feeds MMA.

**Interleave is an iteration-level schedule, not enforced alternating SASS.**
CUDA 13.1 bundles the eight MMAs before the eight HFMA2s in the eight-chain FP16
build. Both groups repeat within one loop; phases executes the entire tensor
loop first. This measures useful mixed-loop issue, not an assumed dual-issue
mechanism. No artificial data dependency or per-instruction barrier forces order.
One-chain builds are the complementary tightly interleaved/latency case.

Each chain has its own loop-carried scalar register and four FP32 MMA accumulator
registers; eight chains provide eight independent chains of each kind. All 32
lanes execute every warp MMA and scalar instruction. Every chain's four tensor
words and scalar word reach unconditional stores; there is no rare-value sink.
Runtime loops explicitly disable unrolling: unchecked unrolling let ptxas combine
repeated integer adds, so counting PTX adds would have overstated native work.
The chain loop remains unrolled.

MMA is `m16n8k64` block-scaled `mxf4nvf4`, 8192 MACs/warp instruction. All FP4
values/scales are +1; each output adds 64. Iterations ≤65536 keep that sum exactly
representable in FP32.

Floating seeds are exact dyadics `lo=.5+chain/8+lane/16`, `hi=lo+1/8`:

| Op | One scalar step | Target instructions / element updates / arithmetic ops |
|---|---|---|
| `ffma` | `x=fma(x,-1,.25)` | 1 / 1 / 2 |
| `f16x2_fma`, `bf16x2_fma` | `x=fma(x,[-1,-1],[.25,.375])` | 1 / 2 / 4 |
| `f16x2_add`, `bf16x2_add` | add `[.125,.0625]`, then its negative | 2 / 4 / 4 |
| `f16x2_mul`, `bf16x2_mul` | multiply by `[-1,-1]` | 1 / 2 / 2 |
| `dp4a` | signed byte dot `[4,3,2,1]·[8,7,6,5] + acc` | 1 / 4 / 8 |
| `dp2a_lo`, `dp2a_hi` | signed halfword `[3,2]` dot low/high byte pair of `[1,2,3,4]` | 1 / 2 / 4 |
| `lop3` | XOR three inputs, LUT `0x96`; constants XOR to all ones | 1 / 1 / 0 |
| `prmt` | exchange the two 16-bit halves, selector `0x1032` | 1 / 1 / 0 |
| `imad` | `acc=acc*1664525+1013904223` modulo 2³² | 1 / 1 / 2 |
| `iadd`, `isub` | add/subtract `0x00010001` modulo 2³² | 1 / 1 / 1 |
| `swar15x2_add` | packed add of one in two unsigned 15-bit lanes, then mask | 1 / 2 / 2 |
| `swar15x2_sub` | set guard bits, packed subtract of one, then mask | 1 / 2 / 2 |

FMA/mul states alternate; each add in the balanced pair changes the value before
restoring it. All intermediate values are finite, nonzero and below 4 in absolute
value, exactly representable in FP16 **and** BF16. There is no saturating modifier,
overflow, underflow or long accumulator stagnation. The add pair's final value
alone cannot witness its two instructions; SASS retention is essential and was
checked. The floating tests intentionally measure native arithmetic, not conversion
cost or the numerical behavior of long approximate GDN reductions.

Integer states use unsigned host arithmetic for modulo-2³² predictions. DP2A
increments are 7/17; DP4A's is 70. PRMT and LOP3 are reversible, not maps that
quickly discard their changing state. SWAR lanes occupy bits 0–14 and 16–30.
The masks/guards implement modulo-2¹⁵ lane arithmetic, not unconstrained packed
INT16. [packed_ops_swar.lean](packed_ops_swar.lean) proves add/sub for every pair
of 15-bit lane values (`lean bench/packed_ops_swar.lean`, checked with Lean 4.33).

After timing, fresh untimed launches download every chain. Isolated outputs are
checked against exact host recurrence predictions; every mixed scalar/tensor word
must bit-match its isolated reference. CSV hashes all output words in order;
these hashes are observability receipts, not the equality test. A hash of all-zero
unused words is not numerically zero.

## Counts and interpretation

Let `B=SM_count*blocks_per_sm`, `W=warps`, `I=iters`, `C=chains`, `S=scalar_steps`.
Every tensor-containing case executes `M=B*W*I*C` warp MMAs. Every scalar case
executes `U=32*M*S` lane-level scalar steps. For each op, multiply U by the table's
three counts for target instructions, element updates and arithmetic operations.
Add steps contain two packed additions: four element updates, not two.

SWAR add has **two** PTX/native ALU instructions (IADD + mask), SWAR sub has
**three** (guard OR + IADD + mask). `target_lane_instructions` counts only their
one add/sub; `scalar_ptx_lane_instructions` counts all 2/3. Other ops' PTX count
matches their target count. FMA counts two arithmetic operations per element;
DP4A four MACs = eight arithmetic operations, DP2A two MACs = four. LOP3/PRMT have
register transforms but no numeric arithmetic operations. For active full warps,
scalar warp-instruction counts are lane-instruction counts divided by 32.

Counts exclude initialization, loop control, operand-constant materialization,
addressing and output stores; event time includes all of them. Packed PTX counts
are **not** a full scheduler-issue ledger. Inspect SASS/counters for that.
`Gtarget_lane_instructions_s`, `Gscalar_ptx_lane_instructions_s`,
`Gelement_updates_s` and `Garithmetic_ops_s` deliberately expose all denominators.
`FP4_TMAC_s` is warp MMA rate ×8192; its FLOP rate is twice that.

`speedup_vs_isolated_sum=(tensor_ms+isolated_scalar_ms)/mixed_ms`;
`time_vs_isolated_max=mixed_ms/max(tensor_ms,isolated_scalar_ms)`.
Compare interleave with phases too: both retain mixed state and use one launch,
whereas isolated sums have different register/residency budgets and two launches.
Times are medians/min/max across rotated case order, not best-only samples.

## Local SASS retention receipt

CUDA 13.1.115, `-O3`, sm_120a, eight chains, Schedule 3. Every specialization has
8 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` sites. Scalar sites occur once after MMA
and once in the extra-step loop (the latter is skipped at S=1). Counts below are
**static sites for the whole function**, not dynamic instructions per iteration:

| Op ID / name | Retained opcode / static sites | Registers/thread |
|---|---|---:|
| 0 `ffma` | FFMA 24: 16 scalar sites + 8 seed setup | 50 |
| 1 `f16x2_fma` | HFMA2 16 | 56 |
| 2 `f16x2_add` | HADD2 32 | 50 |
| 3 `f16x2_mul` | HMUL2 16 | 50 |
| 4 `bf16x2_fma` | HFMA2.BF16_V2 16 | 56 |
| 5 `bf16x2_add` | HADD2.BF16_V2 32 | 50 |
| 6 `bf16x2_mul` | HMUL2.BF16_V2 16 | 50 |
| 7 `dp4a` | IDP.4A.S8.S8 16 | 50 |
| 8 `dp2a_lo` | IDP.2A.LO.S16.S8 16 | 50 |
| 9 `dp2a_hi` | IDP.2A.HI.S16.S8 16 | 50 |
| 10 `lop3` | LOP3.LUT 17: 16 target + 1 setup | 50 |
| 11 `prmt` | PRMT 16 | 50 |
| 12 `imad` | IMAD 18: 16 target + 2 setup (wide address ops separate) | 50 |
| 13/14 `iadd`/`isub` | IADD 25: 16 target + 9 setup | 50 |
| 15 `swar15x2_add` | IADD 25, LOP3.LUT 25 (16 masks + 9 setup) | 50 |
| 16 `swar15x2_sub` | IADD 25, LOP3.LUT 41 (32 guards/masks + 9 setup) | 50 |

In the FP16 FMA function the eight MMA accumulator bases are
R24/R20/R16/R12/R8/R4/R28/R32; the scalar accumulators are
R0/R53/R51/R49/R47/R45/R43/R41. They are distinct, not duplicated instructions
on one chain. No loop-carried state shares tensor/scalar registers. Constants
can share physical registers across non-overlapping lifetimes; setup/loop-control
opcodes therefore still matter to measured issue demand.

## Saturated VM result and constant-hoist control

CUDA12.8 on the campaign's RTX5090,1000 iterations,8 chains,16 warps,3 requested blocks/SM,
64 scalar steps,7 repeats: isolated FFMA87.10TFLOP/s versus FP16x2/BF16x2 107.34/107.28.
This schedule gains1.23× useful arithmetic, not2×. The low-scalar-pressure8-warp/step1 case
is31.98 versus61.05/60.89TFLOP/s. Both statements describe measured schedules, not vendor peaks.

Isolated saturated-kernel NCU counters show FFMA97.97% issue,70.57% FMA-heavy and85.76% FMA-lite;
FP16x2 has65.71% issue but97.80/97.66% on both FMA subpipes. BF16x2 is97.82/97.68%.
Packed arithmetic frees issue slots while occupying both physical FMA subpipes. Their instruction
counters overlap: adding heavy+lite is not a count of distinct scheduler instructions.

The CUDA12.8 isolated FP16 kernel also rematerializes its constant multiplier in the inner loop:
`HADD2 R0,-RZ.H0_H0,1,1` followed by eight targetHFMA2. Counter evidence agrees:4,177,920,000
requested warp FMAs versus4,700,168,160 FP16-type warp instructions (including initialization).
This additional packed instruction is overhead, not useful arithmetic.

`-DPACKED_OPS_HOIST=1` passes the multiplier as an explicit invariant kernel parameter, preventing
that literal rematerialization. The unset/0 build keeps the original literal path. Raw CSV metadata
records `constants=literal|kernel-parameter`. CUDA12.8 native SASS confirms the hoisted build has
no recurring HADD2 in that loop. Alternating literal/parameter/parameter/literal processes,
otherwise identical flags and the saturated configuration above, give FP16x2 108.16/119.83/119.49/107.89
TFLOP/s; BF16x2 108.24/119.80/119.42/107.87. FFMA stays87.56–87.82. All exact checks pass.
Removing the instruction raises useful packed throughput to~119.7TFLOP/s (~1.36× FFMA here),
without inventing a second set of FMA pipes. Raw alternating runs are under `batch2/equivalence/packed-*`.

Raw evidence: `~/work/tune-5090/batch2/packed-pipes/` (including nativeSASS and counterCSV),
with the hoisted variant under `packed-hoist/`. The source-owned runner is
`tools/pod/batch2_packed_pipes.py`.
