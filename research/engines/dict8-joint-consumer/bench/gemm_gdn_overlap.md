# Actual GEMM + GDN overlap (sm_120a)

`gemm_gdn_overlap.cu` measures the engine's **unsplit WF2/WF6 projection GEMMs**
against its **bf16-state, FP32-arithmetic CG4 GDN recurrence**, at 64 slots × 8
rows = 512 rows. The two workloads have independent buffers: GEMM output is not
GDN input. This measures the opportunity for scheduling independent engine work,
not permission to overlap a layer's own producer and consumer.

## Build and run

From the repository root, CUDA 13.x:

```sh
DEFS="$(python3 tools/selected_build_defines.py results/integrated/selected-preset.json)"
make -f bench/gemm_gdn_overlap.mk -j2 DEFS="$DEFS"
# On this Ubuntu compiler host also pass NVCC_HOST_CXX=g++-14.

build-gemm-gdn-overlap/gemm-gdn-overlap \
  --wf wf2 --shape all --cfg all --repeats 7 --launches 64 --weights cold \
  > /tmp/gemm-gdn-wf2.log
build-gemm-gdn-overlap/gemm-gdn-overlap \
  --wf wf6 --shape all --cfg all --repeats 7 --launches 64 --weights cold \
  > /tmp/gemm-gdn-wf6.log
```

For a short single-configuration screen, `--shape in_gdn --cfg 7 --launches 16
--weights hot --repeats 7`. `--help` gives the complete interface. Every option
is required; unknown, duplicate and malformed options fail. Exit 2 is a contract
error, 3 a CUDA failure, 4 an output/state mismatch. Unlaunchable tiles and non-wg2
fused-GU configurations print `SKIP`; a selection with no runnable tile fails.

The private makefile produces independent objects and never alters the campaign's
engine binaries. It includes `src/kernels.cu` in the benchmark translation unit
only to query the exact GDN specialization's resources; the actual work goes
through `launch_gdn`. GEMMs link the unmodified `src/fp4.cu` and use its public
launchers and `fp4_profile_resources`. No model files, engine state, rental or
GPU service required to build. Building takes longer than 55 seconds on some
hosts; use the host's durable build job when necessary.

## Equal work and input contracts

Each paired schedule performs `--launches` GEMMs and `--launches` GDN visits;
isolated references perform the corresponding one workload. Every visit contains
all 512 rows. Each graph starts from identical seeded state and zero add-GEMM
output. Reset copies and output poisoning are **outside** timing.

Shapes are the production `in_gdn` (16384×5120), `in_att` (14336×5120), `out`
(5120×6144, add), `gu_silu` (34816×5120, fused SiLU(gate)·up), and `down`
(5120×17408, add). All use split-K=1. Activations are finite E2M1 nibbles, E4M3
block scales, and unit FP32 row scales in the engine's blocked layout. Weight
scales are finite E4M3 and FP32 row factors. WF6 expands **exactly the same**
WF2 packed words and scales via `src/ternary_operands.h`, including signed zero;
its larger representation changes no input arithmetic.

The recurrence has HK=16 key/query heads, HV=48 value heads, SS=128, 64 slots,
eight successive token rows per slot. Synthetic conv output contains normalized
queries/keys, bounded values, gates 0<g<1 and 0<beta<1, and nonzero bf16 state.
Each segment uses `commit=0, keep=0`, so all eight rows update and save committed
state. Pending storage exists for the real pointer layout but is not read or
written. This excludes speculative pending/replay, conv, gated normalization,
quantization and GEMM preps. The first layer is visited repeatedly using the same
eight input rows; resetting between samples keeps schedules comparable.

GDN addresses state with the engine's `[slot][NGDN=48][head][128][128]` stride;
the benchmark reserves that address span (~4.8 GB) but touches only one layer
(~100.7 MB of bf16 state). Pending's strided allocation is ~0.8 GB. Only touched
state slices are downloaded/compared. A 32 GB RTX5090 has ample capacity.

`--weights hot` reuses one weight copy. `cold` rotates independently seeded
copies with a total weight+scale footprint exceeding twice the device's L2 size;
launches must cover the entire rotation. Copies, working-set size and L2 size are
printed. This is a cache-capacity screen, not an assertion that every load
missed L2. The state reset itself exceeds L2 and affects both cases equally.

## Schedules and correctness

The six captured CUDA graphs are:

- `gemm`, `gdn`: isolated references.
- `serial_gemm_first`, `serial_gdn_first`: each iteration issues both kernels in
  one nonblocking stream in the named order.
- `concurrent_gemm_first`, `concurrent_gdn_first`: all GEMMs on one nonblocking
  stream and all GDN visits on another, common graph fork and final join, no
  cross-workload data dependencies. The suffix selects the control-stream branch
  and capture submission order, exposing scheduling bias.

Graph capture removes host enqueue gaps. Samples time one entire graph with
CUDA events after reset and before downloads, including graph dispatch, all
work, fork/join and the final completion. PDL is pinned off so serial really
means serial; GDN CG4/OCC2/unscaled/nonpipelined and shared-memory carveout `none`
and `H5_GDN_ARITH=fp32` are pinned explicitly. Other compile-time selected-preset settings still apply.

One initial execution of every graph serves as warmup and correctness sample.
Every GEMM element, every GDN output element, and every touched bf16 state element
must be **bit-identical** to its isolated reference, finite and not all zero.
There is no split-K rounding allowance because this benchmark never splits K.
`CHECK` prints element counts, unequal/nonfinite counts, maximum absolute error,
sum, sum of squares and an FNV-1a hash of all output bytes. The all-configurations
`PASS` line appears only after every selected runnable configuration passed.
These checks establish schedule equivalence, not a new mathematical GDN oracle.

Timed samples start from the same state but exclude correctness downloads. Case
order rotates across repetitions. `TIME` reports median/min/max whole-graph ms,
µs per launch set, and actual GEMM/GDN launch counts. A launch set is one visit
for isolated cases, one GEMM + one visit for paired cases.

## Reading the result

- `speedup_vs_isolated_sum = (GEMM_median + GDN_median) / paired_median`.
- `time_vs_isolated_max = paired_median / max(GEMM_median, GDN_median)`.
- `OVERLAP.speedup_vs_same_order_serial` compares concurrent with the measured
  serial case having the same suffix; above one means overlap helped.
- `saved_us_per_pair` is that serial/concurrent median difference divided by
  launch count. `helps` is the sign of this measured difference, not a statistical
  confidence claim. Inspect the min/max spread and both branch orders.

`RESOURCE` prints **real compiled** registers/thread, static and dynamic shared
memory, local bytes/thread, actual grid/block geometry, and occupancy API active
CTAs/SM in isolation. GEMM dynamic shared bytes come from the engine's registered
`WsGeom` allocation, not an inferred model. Independent occupancy ceilings do
not establish mixed-kernel co-residency. No kernel instrumentation is inserted;
Nsight Systems can inspect the concurrent intervals if the GPU owner wants a
physical overlap trace.

`GEMM_COST` reports logical `512*N*K` MACs/launch, unique packed activation bytes,
weight and scale bytes/copy, row factors, output writes and add-output reads.
`GDN_COST` reports state read+write bytes/launch, unique conv+gate input bytes,
output bytes, and recurrence counts. For each new token and state element there
are two state-dot FMAs (M·k and M·q) and one state-update FMA; the update also
forms delta·k with a multiply. Reductions, scalar gate/innovation arithmetic,
loads, stores and launch overhead are included in timing, not those FMA counts.
Byte numbers are **logical unique traffic**, not measured DRAM traffic: CTA
input duplication, cache reuse, weight reuse across row tiles and compiler local
traffic can change physical transactions. Multiply per-launch counts by the
printed launch counts to obtain total work for a timed sample.
