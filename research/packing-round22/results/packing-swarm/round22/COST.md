# Round22 — equal-boundary paired-reader invoice

CPU-only exact source requests. No GPU execution, spending or physical-time claim.

## Paired geometry

Both arms: eight32-feature groups/256features per CTA, TOKW32, K5120, forty ascendingK128 folds per independent pass. Each stage is4096W+512halfscales+2048q0+256s0+2048q1+256s1=9216B; eight stages plus sixteen8B mbarriers and256B shared global scales request74112B dynamic shared. Both arms load the two actual global-scale vectors once per CTA (8*rows*CTAs bytes), not per warp or output lane.

Candidate: four consumer warps/WG2 plus one producer (160threads),128 independent accumulator floats per consumer thread. Strongest: eight consumer warps/WG1 plus one producer (288threads),64 floats/thread. Both retain65536B aggregate accumulator payload/CTA. This is source liveness, not ptxas registers or occupancy.

| Family | Rows | CTAs | Paired W+halfscale B | Paired q+s B (each arm) | Each arm launches |
|---|---:|---:|---:|---:|---:|
| input3 | 1 | 65 | 11865600 | 11980800 | 1 |
| input3 | 8 | 65 | 11865600 | 11980800 | 1 |
| input3 | 32 | 65 | 11865600 | 11980800 | 1 |
| attention3 | 1 | 56 | 10321920 | 10321920 | 1 |
| attention3 | 8 | 56 | 10321920 | 10321920 | 1 |
| attention3 | 32 | 56 | 10321920 | 10321920 | 1 |
| ffn2 | 1 | 136 | 25067520 | 25067520 | 1 |
| ffn2 | 8 | 136 | 25067520 | 25067520 | 1 |
| ffn2 | 32 | 136 | 25067520 | 25067520 | 1 |

Both arms halve recurring weight/halfscale requests and reader launches against their unpaired origins. Equal tile staging removes the R4 q/scale duplication. Shared and aggregate accumulator payload match between paired arms; candidate doubles accumulator payload per consumer thread. Bytes and reader launches match against the equally strengthened control; fewer consumer barrier/shared-q/scale instructions are bought with doubled per-thread accumulator liveness. Compiled candidate spills while control does not. No nominee, time/residency/occupancy claim.

## Complete inherited invoice

Includes all source/norm factories and strides; E/D, half/global scales, quantizers and padded64 tapes; original19, materialized20 and direct21 continuations; ring/convpending/bothpending/checkpoint/F16KV/head/logits/argmax; source preservation, status/fault owners and reset; scans, metadata, borrow views, Frozen/old-new/deepclone and D2D/before-after images; retention on uncertain effects, request/evidence storage, serialization/rename/directoryfsync and whole release. None is a zero charge.

The source-bound JSON retains every regional and full64 inherited cost. Standalone source requests are compared separately from the complete retained validation harness. Both paired arms use identical source+Norm, one full Tape (physical64-row q/s padding, globals, transformed/Norm and producer statuses), one independent prepared weight image and16BF output/snapshot/two-owner bytes. Original candidate owned prep still pays its producer status D2H/checks; generic strongest prep does not. Direct preparation/down continuation charges remain exactly inherited, including the candidate co-reader and strongest generic control.

Both new arms are appended, not substituted for old controls. Each family adds four FP32 images, four owner images and a separately allocated/launched originalFW8 primary reference: nine allocations/36BF device bytes. Initial fills36BF plus explicit zeros/owner poison28BF; complete new observations/original primary/end-preservation add52BF D2H (13calls). New host retention40BF, original-primary transient4BF plus prolonged originalFW8 final-vector transient4BF, preservation transient16BF, Frozen40BF+4B. Both arms have complete primary/final and owner observations. Original/direct outputs and new primary snapshots survive to preservation, and uncertain backing stays owned.

Per family successful added CUDA calls before release:9allocations+9initialfills+7explicitfills+3attributes+3launches+3fences+13D2H=47. Four new pass manifest groups add12 start/launched/observed calls. Native full64 adds512 groups (38..45/layer); all complete manifest/progress/domain/evidence scans remain paid. ReaderProgress/control-block/vector/allocator capacities remain unset, not zero.

Full64 each paired arm requests2339020800B weight/halfscales,2344550400B q/scales,101760*rowsB globals,128reader launches and12720CTAs. Original independent/retained controls still execute: validation adds384launches including128primary references,7017062400B weight/halfscale and5861376000B q/scale requests; no cache reduction is assumed.

| Fresh rows | Retained-all generation B | Success-assumed layer-local peak B | Added device B | Added D2H B |
|---|---:|---:|---:|---:|
| 1 | 9358146200 | 553385600 | 116951040 | 168929280 |
| 8 | 11808378432 | 740601216 | 935608320 | 1351434240 |

Full64 adds1152 whole-array validation handles:14573 total (80B records plus8B pointer entries), at most248/layer under256. Host/Frozen deltas are129945600*rows and129945600*rows+512B. Old8/newR, reset, deepclone copies and enlarged host images are recomposed for all24 serialized independent-stream configurations; none is physical residency.

## Compiled liveness loss

sm120a candidate:255registers/thread,16B stack,32B ptxas spill stores/32B spill loads. Strongest:131registers/thread,zero stack/spills. Both request74112B dynamic shared; ptxas reports no user static shared, ELF resource shared1024B is separately retained. Actual host:O1 appended after nvcc injected O3; device:O3/defaultcontraction/no_fastmath.

SASS opcode/site counts are bound in JSON, including actual local spill sites. Dynamic spill bytes, barrier-spin iterations, residency and occupancy are unset. Source barrier arrivals/waits and shared q/scale fragment instructions are lower for four consumer warps; the matched bytes/launches and compiled spill regression prevent nomination. This is not an absolute Pareto claim across every instruction metric.

**Unset:** native words/faults/time, concreteCUDArefinement, allocator capacity/granularity/residency, cache-effective traffic, occupancy, tensor-wideB8/B32 controller, engine/model/task/HTTP.

Replay: `python3 method/packing-swarm/tools/round22/cost/invoice.py --check`; regenerate with no flag.
