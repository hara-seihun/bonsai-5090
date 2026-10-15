# Two real engines, one fixed batch

Owner: batch2, `bench/multi_engine_batch.{cpp,mk,md}`. Completed-buffer probes and engine-owned
split-top-k scratch live in `src/{draft_probe.h,draft.cu,engine.cu,kernels.cuh}`.

The question is whether two independent 32-stream engines can finish the same total 64×8
verify-plus-draft work faster concurrently than serially. This harness keeps **the same two
engine objects** alive for serial AB, serial BA and concurrent arms, uses two persistent host
threads for all three arms, and joins the public synchronous engine calls. The concurrent
mutex release is a common-start barrier. Wall time includes worker wakeup, row metadata,
verification, forced acceptance, drafter work, output collection and final join; per-engine
verify/draft times are also emitted. No CUDA event timing sums are substituted for the wall time.

## Work and state

- Every round verifies eight rows per stream and drafts seven proposals. `ACCEPT` is the number
  of accepted proposals, 0–7; the anchor plus proposals advances by `ACCEPT+1`.
- All target inputs and drafter anchors are a deterministic function of **global** stream and
  absolute position. Outputs never choose subsequent work. Different arms therefore execute
  identical work even when an output differs. This is a fixed-work arithmetic experiment, not
  a natural generation throughput or quality measurement.
- Unlike the profiler's synthetic nonzero-context start, every arm replays a complete fixed
  prefix from position zero, with target verification and ingest-only drafter calls. `CTX`
  must be a positive multiple of eight. Prefix replay is excluded from timing.
- `h5_reset` clears recurrent/conv state, not target KV or drafter KV. Reset alone is **not**
  the restore operation. Full prefix replay overwrites every target prefix KV position and
  every drafter past-window position readable in a measured round. Pending rows are dropped
  by reset. Prefix calls use keep=0; measured calls use keep=1. A device fence now completes
  reset writes before any prefix kernel can preload state. Prefix replay is the intended
  restore, not a demonstrated snapshot: the first GPU run found large repeated-serial
  differences. The diagnostic modes below isolate target split-K replay drift and a separate
  PDL-sensitive hidden-projection failure. Unread
  cache bytes and scratch are not claimed bit-identical.
- Three complete restored sequences warm all measured graph shapes serially before arm
  timing, avoiding first-use graph capture in concurrent threads. `graphs` explicitly enables
  graphs up to 512 rows; `eager` explicitly disables them. The frozen plan is required,
  `H5_PLAN_FILL=0` is enforced, and write/launch-override/profiling knobs are rejected.
- Arm order rotates between repetitions. Serial AB repeat0 is the exact-output baseline.
  Checks compare **every** verifier token (512/round), drafter token (448/round) and selector
  confidence float's raw bits (448/round). A mismatch reports count and first flat index,
  retains full output files, continues the remaining comparisons, and finally exits3.
  Zero token differences but confidence differences is not reported as bit-exact arithmetic.
- This is an observable-output check over the selected rounds, not a raw recurrent/KV dump.
  Serial BA and repeated AB checks distinguish repeatability failure from a concurrency-only
  difference. The first GPU run failed repeated serial checks substantially; the cause is not
  established, and those differences must not be labelled split-K noise.

The single64 reference runs in a **separate process**: there is no public engine destroy,
so keeping a third engine resident would invent a capacity requirement. Single64 uses the same
inputs and context but a different numerical launch geometry. Its difference from pair serial
is a geometry result, never conflated with same-pair concurrency equivalence. Its timing does
not include the pair's worker dispatch overhead.

## Build

CPU-side compilation needs CUDA headers but no GPU and takes seconds:

```sh
make -f Makefile -f bench/multi_engine_batch.mk \
  BUILD=build-multi-engine multi-engine-batch-object
```

Normal full link (use the campaign's **matching** `BUILD`/`DEFS`, CUDA12.8 on the GPU VM):

```sh
make -f Makefile -f bench/multi_engine_batch.mk \
  BUILD=build-selected DEFS="$SELECTED_DEFS" multi-engine-batch
```

To link already-built VM objects without rebuilding/changing their compile configuration:

```sh
g++ -O2 -std=c++17 -Wall -Wextra -Werror -pthread \
  -I/usr/local/cuda/include -c bench/multi_engine_batch.cpp -o /tmp/multi_engine_batch.o
nvcc -gencode arch=compute_120a,code=sm_120a -o multi-engine-batch \
  /tmp/multi_engine_batch.o \
  build-selected/{engine,graph_memory,kernels,fp4,sk,draft,f02-producer,f02-continuation,gguf}.o \
  -Xcompiler -pthread
```

Replace `build-selected` with the existing candidate object's directory. Nothing links llama or
HTTP serving. Local CUDA13.1 host-object compilation succeeded with `-Wall -Wextra -Werror`.
A CPU mock of the public APIs/CUDA calls checked all five pair comparisons, pair/single64
output indexing, API failure records/exit1, and confidence-only mismatches completing all
arms before exit3. These check harness control flow, not GPU arithmetic.
The campaign owner ran the original harness on the existing GPU; this author has not
launched GPU work. The original evidence remains in
`~/work/tune-5090/batch2/diagnostics/engines-{pair,single64}.{stdout,stderr}`
and associated `.txt` files. The repaired source and concurrent scratch-ownership test built
successfully with CUDA12.8 on the owner's VM; `drafter-exact.repair` exited0. The completed
probe API also passed host-mock tests, including an injected hidden-projection-only difference.

## Bounded screen

Run exclusively on the campaign owner's existing GPU; no rental is needed. If the combined
load/prefix-replay run exceeds the attended shell limit, submit these bounded commands to the
owner's durable GPU worker rather than detaching a shell. Use the exact same candidate binary,
model, drafter, frozen plan and selected WF2/CG4 environment for both commands. A frozen plan
covering target R256/R512, all smaller prefix/head buckets, draft ingest and draft block shapes
is needed. Missing shapes are a failure, not an invitation to tune during a timed arm.

```sh
mkdir -p out/multi-engine
# Export the selected candidate's H5_* arithmetic/representation knobs first.
# Unset H5_PLAN_WRITE, H5_PLAN_OVERRIDE, H5_SK_OVERRIDE, H5_PROF,
# H5_SYNC_DEBUG, H5_AB_PLAN_B and H5_NO_SPEC.
timeout 240s ./multi-engine-batch pair "$MODEL" "$DRAFTER" \
  600 4 7 2 0 "$PLAN" out/multi-engine/pair graphs wf2-cg4 \
  >out/multi-engine/pair.jsonl 2>out/multi-engine/pair.stderr
PAIR_STATUS=$?
printf '%s\n' "$PAIR_STATUS" >out/multi-engine/pair.exit
# Continue to single64 after pair failure too: capacity negatives are useful evidence.
timeout 180s ./multi-engine-batch single64 "$MODEL" "$DRAFTER" \
  600 4 7 2 0 "$PLAN" out/multi-engine/reference graphs wf2-cg4 \
  >out/multi-engine/reference.jsonl 2>out/multi-engine/reference.stderr
REFERENCE_STATUS=$?
printf '%s\n' "$REFERENCE_STATUS" >out/multi-engine/reference.exit
```

CLI:

```text
multi-engine-batch MODE MODEL DRAFTER CTX ROUNDS ACCEPT REPEATS DEVICE PLAN OUTPUT_PREFIX graphs|eager LABEL
MODE = pair | single64 | diagnose-pair | diagnose-single64 |
       fresh-ab | fresh-ba | fresh-concurrent | fresh-single64
```

Exit0 = requested checks passed (fresh modes only collect observations and declare
`comparison=not-run`); exit1 = runtime/load/IO failure; exit2 = invalid configuration;
exit3 = measured-output or diagnostic prefix differences; timeout's exit124 = unfinished bounded screen,
not a capacity conclusion. Negative load/capacity evidence requires the engine/native error,
not merely an elapsed deadline. Exceptions from public void operations are converted at the
API boundary; integer API codes, native CUDA code/name and exception detail are retained in
JSON. Engine stderr retains native errors consumed inside the C wrappers. Memory events record
free/total/used bytes before load, after each target+drafter+pretune, and after warmed graphs.
Use differences between successive used-byte records for engine and graph memory cost.

Output `.txt` files contain one row per `(round, global_stream)`: those two fields, eight
verifier tokens, seven draft tokens, seven confidence float bit patterns. The header declares
format and round count. Compare pair geometry with single64 separately:

```sh
python3 - <<'PY'
from pathlib import Path
import json
paths = [Path('out/multi-engine/pair.serial_ab.0.txt'),
         Path('out/multi-engine/reference.single64.0.txt')]
if not all(p.exists() for p in paths):
    raise SystemExit('geometry comparison unavailable: inspect load errors/exit receipts')
a, b = [p.read_text().splitlines() for p in paths]
if a[0] != b[0] or len(a) != len(b):
    raise SystemExit('different work manifests')
counts = dict(verify=0, draft=0, confidence_bits=0)
first = {}
for ra, rb in zip(a[1:], b[1:]):
    aa, bb = [list(map(int, r.split())) for r in (ra, rb)]
    if len(aa) != 24 or len(bb) != 24 or aa[:2] != bb[:2]:
        raise SystemExit('malformed or reordered output')
    for label, lo, hi in [('verify', 2, 10), ('draft', 10, 17), ('confidence_bits', 17, 24)]:
        for k in range(lo, hi):
            if aa[k] != bb[k]:
                counts[label] += 1
                first.setdefault(label, dict(round=aa[0], stream=aa[1], lane=k-lo,
                                             pair=aa[k], single64=bb[k]))
print(json.dumps(dict(comparison='two-engine-vs-single64-geometry', differences=counts, first=first)))
PY
```

The final pair `result` event reports both serial means, the concurrent mean, both
serial/concurrent ratios and saved microseconds per round, alongside same-geometry exactness.
A useful result includes those equal-work timings, all same-geometry difference counts,
single64 timing/geometry differences if it fits, and memory cost. A pair OOM with a successful single64 is a valid capacity negative. A concurrent speedup
with repeated serial-equivalent outputs answers the real-model overlap question for this
capacity and pinned plan; no scheduler or HTTP changes are required to obtain that answer.

## First GPU result and restoration diagnosis

The original pair fits at 32.68GB resident after warmed graphs (0.986GB free). Both serial
orders averaged54.90/54.91ms, concurrent52.92ms: ~3.7% better than two serial engines but
slower than single64 (~51.5ms). There is no integration win in that result.

Repeated serial AB nevertheless changed519/2048 verifier tokens,1126/1792 draft tokens and
all1792 confidence float bit patterns. Confidence absolute change was median0.239, p951.298,
max3.591, with17/64 verifier lane0 changes already in round0. Single64 repeated serial also
changed518 verifier tokens. These are substantial repeatability failures, not an observed
concurrency-only defect. Original evidence is retained; use **new output prefixes** below.

Revised code adds a reset-completion fence and these explicit modes:

| Mode | State history | Observations |
|---|---|---|
| `diagnose-pair`, `diagnose-single64` | same three warmups and repeated restores as normal | every prefix chunk's CPU input hash, every prefix output token, and full-vocabulary logits of that chunk's last row compared with the **first** restore |
| `fresh-ab`, `fresh-ba`, `fresh-concurrent`, `fresh-single64` | one restore from a newly loaded engine, **no warmup**, `REPEATS=1` | prefix checkpoint hashes and ordinary output files; cross-process comparisons done by the caller |

Prefix checkpoints query existing public `h5_logits` **after** target verification and
**before** draft ingest, without changing the target's head/graph route. At32/64 streams,
row127 of the final compact head chunk is the last stream's final row. The event records that
logical stream/position, raw-bit logit hash, unequal-bit count, maximum absolute/RMS logit
delta and nonfinite count, plus all verifier-token differences for that prefix chunk. The
baseline retains about75MB host logits per engine atCTX600, no new GPU buffers. Diagnostic
D2H copies are outside round timing, but diagnostic/fresh timings are not a steady-state
performance reference. Fresh mode's graph creation can occur in its first measured round.

Per-round receipt hashes cover the CPU call arguments for both target and drafter. They
establish input agreement at the API boundary, not what a broken device metadata copy read.

Source narrowing found a concrete mechanism by which replay can fail to restore identical
state: `src/fp4.cu` selects MODE2 at split count>1 (around824–825) and atomically adds partials
(around683). The **actual** archived `out/curve-wf2-3072/plan.txt` has R256 target-down split2
and alpha/beta split8; R512 alpha/beta is split8 and the rank256 projection is split4. FP4
thresholding in `src/prep_chunk.cuh:203–215` and recurrent FP16 saves in
`src/kernels.cu:1027–1045` can amplify small unordered-reduction differences through the
600-position replay. This identifies a mechanism, not its measured responsibility for the
observed magnitude until the unsplit discriminator below. `H5_FORCE_CFG` does not replace
entries already read from a frozen plan.

The cheapest direct discriminator is a **new** frozen plan preserving all configurations and
setting ordinary-format split counts to1. Keep the original plan/evidence unchanged:

```sh
python3 - "$PLAN" out/multi-engine/unsplit.plan <<'PY'
from pathlib import Path
import hashlib, json, sys
source, dest = map(Path, sys.argv[1:])
original = source.read_text()
rows, changed = [], 0
for line in original.splitlines():
    if not line.strip():
        continue
    fields = list(map(int, line.split()))
    if len(fields) != 7 or fields[-1] < 1:
        raise SystemExit('invalid frozen plan row')
    if 0 <= fields[4] <= 7:
        changed += fields[-1] != 1
        fields[-1] = 1
    rows.append(' '.join(map(str, fields)))
if not rows:
    raise SystemExit('empty plan')
rewritten = '\n'.join(rows) + '\n'
with dest.open('x') as f:
    f.write(rewritten)
print(json.dumps(dict(source=str(source), dest=str(dest), changed=changed,
    source_sha256=hashlib.sha256(original.encode()).hexdigest(),
    dest_sha256=hashlib.sha256(rewritten.encode()).hexdigest())))
PY
timeout 180s ./multi-engine-batch diagnose-single64 "$MODEL" "$DRAFTER" \
  600 4 7 2 0 out/multi-engine/unsplit.plan out/multi-engine/diag-unsplit graphs wf2-cg4 \
  >out/multi-engine/diag-unsplit.jsonl 2>out/multi-engine/diag-unsplit.stderr
```

If this is repeatable and the original plan is not, isolate prefix replay's split reductions
before calling the original timed arms equivalent. This discriminator has now run: unsplit
plans made **all prefix token/logit-bit and measured verifier checks exact**, eager and graphs.
Original eager had300 bad prefix checkpoints. Drafter output still differed under PDL1;
unsplit plus PDL0 made prefix, verify, draft and confidence checks all exact. Unsplit timings are a different launch
plan and must be measured separately; they do not repair the original performance result.

For graph/state isolation, an eager single64 run with all other selected settings unchanged
is also bounded:

```sh
timeout 180s ./multi-engine-batch diagnose-single64 "$MODEL" "$DRAFTER" \
  600 4 7 2 0 "$PLAN" out/multi-engine/diag-eager eager wf2-cg4 \
  >out/multi-engine/diag-eager.jsonl 2>out/multi-engine/diag-eager.stderr
```

Then follow the first differing prefix checkpoint rather than only final tokens:

- Different CPU input hash: harness/work mismatch. Same hash but different target logits
  before any draft ingest at position0: reset, initial readable state, or target arithmetic;
  drafter ring history cannot explain the first prefix target step.
- Eager repeatable but graph mode not: compare `H5_GRAPH_NODES=native` with selected
  `kernel-memory` using `diagnose-single64 ... graphs`. This separates graph/cache-shape
  effects from the graph-memory rewrite. Do not change the plan.
- Eager already differs: repeat the eager diagnostic with `H5_PDL=0`. This disables
  programmatic launch overlap while retaining arithmetic/shape choices. A change isolates a
  scheduling dependency defect; it is not a new performance candidate.
- Prefix target tokens/logits repeat but measured rounds differ: concentrate on keep/pending
  commit transitions and drafter state. Prefix last-row logits are one stream's probe, not a
  raw dump of all recurrent/KV state.
- Fresh-process repeated runs agree but repeated restores do not: retained engine state or
  cached graph metadata is implicated. Fresh runs themselves disagree: initialization or
  arithmetic/kernel ordering remains in play, independent of cross-arm restore.

For a fresh-process test, invoke the same command twice with distinct output prefixes:

```sh
timeout 90s ./multi-engine-batch fresh-single64 "$MODEL" "$DRAFTER" \
  600 4 7 1 0 "$PLAN" out/multi-engine/fresh-1 eager wf2-cg4
# Repeat as fresh-2; compare fresh-{1,2}.single64.0.txt and prefix checkpoint hashes.
```

Fresh pair arm files retain names `.serial_ab.0.txt`, `.serial_ba.0.txt` and
`.concurrent.0.txt`. The geometry comparison script above also works for any two64-stream
files of the same round count; use a same-geometry comparison label for fresh AB/concurrent
rather than mislabelling it a one64 geometry test. Reset behavior is unchanged.

The source review found no prefix DF-ring coverage gap, stale pending-count replay, or
obvious graph-copy geometry/dependency loss for this workload. One secondary mechanism is
transposed FP8 attention's unconditional padding V reads (`src/attn.cu:382–386`). At our
four-key-aligned boundaries the outside-end V scales are zero, so finite stale V cannot
change the result; stale E4M3 NaNs can still make zero×NaN reach PV MMA. `H5_ATTN8=0` switches
both writer (`src/engine.cu`, `a.vt`) and reader (`src/attn.cu`, `a8`) to predicated row-major
FP8 loads, providing a discriminator without a production edit. Nonfinite intermediates can
be hidden by subsequent quantization, so finite final logits alone do not rule that out.

## Completed drafter probes and ownership repair

Diagnostic modes additionally query `h5_draft_probe` **after completed** `h5_draft` calls:
trunk residual `B×5120`, hidden projection `R7×256`, and all candidate top values/indices
`R7×16`. These retained buffers are not the reused per-chunk logits. Typed errors distinguish
no prediction, invalid argument, insufficient host capacity and native CUDA failure. The
probe adds no in-flight fence; it copies completed buffers and reports exact bits, row ranges,
absolute/RMS deltas and a first-changed-row proportional-fit residual. Diagnostic round
wall time includes these copies and must not replace normal-mode timing.

With repaired scratch ownership, eager unsplit PDL1 retained trunk and top values/indices
were exact, while hidden projection differed in32-row bands (the frozen hproj token tile):
4k–18k of114688 elements, maxabs0.56–1.50, with non-proportional changes. Public outputs
changed229 draft tokens and868 confidence bits. Evidence:
`~/work/tune-5090/batch2/draft-probe-repair/`.
Disabling preclear alone also failed (83 draft tokens/515 confidence bits); evidence:
`~/work/tune-5090/batch2/preclear-screen/`.

The definite two-engine bug was process-global split-top-k scratch. Each engine now owns
its scratch, with disjoint per-head-chunk views; simultaneous streams never share it.
`bench/drafter_exact.cu` exercises repeated concurrent128-row top-k calls with distinct
inputs. This repair is separate from the single-engine hidden-projection failure.

`H5_DRAFT_HPROJ_FENCE=none|before|after|both` is a capture-time diagnostic screen. Unset
means none. A conventional one-thread kernel provides a strict stream boundary, before
hproj (incoming prep/GEMM dependency) or after it (outgoing activation-reuse dependency),
while leaving all other PDL launches unchanged. Run each value with a **new prefix**:

```sh
H5_DRAFT_HPROJ_FENCE=before timeout 180s ./multi-engine-batch diagnose-single64 \
  "$MODEL" "$DRAFTER" 600 4 7 2 0 "$UNSPLIT_PLAN" \
  out/multi-engine/hproj-before eager wf2-cg4
# Repeat with after and both. Preserve selected PDL1 and every other environment value.
```

The AFTER fence passed every retained/public value; BEFORE still failed188 draft tokens
and896 confidence bits. Subsequent dedicated hproj q4/s4/g4 ownership **did not fix** the
failure (eager239/1185 draft/confidence differences; graphs131/1185), so that allocation
change was reverted. The fence identifies a relevant overlap boundary, not necessarily an
activation alias: eliminating downstream contention can mask a kernel-internal pipeline
failure. Evidence is retained under `batch2/{hproj-fence,hproj-lifetime}/`.

`src/fp4_ops.cuh::ldsm4` now declares its indirect shared-memory access with a compiler
`memory` clobber. Both `fp4.o` and `sk.o` consume this helper and need rebuilding. This is
a concrete inline-assembly contract correction. Actual CUDA12.8 cfg3/WF1 disassembly already
kept all shared reads between full wait and empty release, so no observed ring-read escape
establishes it as the failure's cause.

The same disassembly loads mutable activation row scales through `LDG.E.CONSTANT` (PTX
`ld.global.nc`, outside the memory-consistency model). `activation_scale` now performs a
coherent volatile global read in ordinary FP4 and stream-K epilogues; immutable weight
scales are unchanged. A tiny local CUDA13.1 compilation emits `ld.volatile.global.b32`.
Completed hproj probes additionally report proportional fits for each64-feature consumer-
warp group of the first changed row; a whole-row fit cannot exclude warp-local stale scales.
The host probe passed an injected-drift CPU mock.

Both compiler/cache-policy screens failed. With no hproj fence, the clobber-only eager
screen still changed159 draft tokens/525 confidence bits; coherent AG plus clobber changed
255/1091. Prefix and measured verifier checks remained exact; retained trunk and candidate
top buffers stayed exact, while hidden projection still differed. Evidence:
`~/work/tune-5090/batch2/{hproj-clobber,hproj-coherent}/`. These corrections are not
an established repair of the observed drafter failure.

The coherent screen's first-changed-row fits included35 changed64-feature groups over16
checkpoints. None was uniformly proportional (relative fit residual min0.079, median0.186,
max0.424); several groups changed only8/16/32 of64 features. This excludes a uniform
consumer-warp scale factor as a complete account. It does not prove the exact internal
cause; no further GPU discriminator was run before the full numerical evaluations.

The AFTER fence remains the known-exact **single-engine** control. Its exactness is an
observed scheduling-boundary result, not proof of an activation alias or a missing source
wait. The source PDL contract nominally orders the inputs, and dedicated activation ownership
failed. Underlying drafter non-equivalence remains unresolved.

Pair serial AB/BA are exact, but concurrent still differs even with PDL0 (verify
110/51, draft421/315, confidence662/535 across two repeats). Evidence:
`~/work/tune-5090/batch2/pair-pdl0/`. No remaining process-global mutable scratch
alias was found; attribution to an actual cross-engine alias is not established. The active
alpha/beta fork still uses a second engine-owned stream under PDL0. Its explicit off control
is `H5_AB_PARALLEL=0`; this changes the launch schedule, not GEMM arithmetic.

## Small native-kernel reproducer

[`hproj_pressure.md`](hproj_pressure.md) owns the model-free native WF1 hproj under independent
WF2 head pressure. It checks full outputs and guards under both serial/concurrent launch
orders and same-stream PDL chains, eager/graphs, with a matched sanitizer filter. Optional
CTA timestamps witness actual pressure residency. This tests the kernel pipeline, not the
model's state-restoration or activation-lifetime path.
