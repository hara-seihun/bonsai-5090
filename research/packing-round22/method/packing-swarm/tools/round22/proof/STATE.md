# Round22 — paired shared-tile dictionary

Owner: packing-swarm proof sidecar. Start with `PairedTile.lean`; replay is `python3 method/packing-swarm/tools/round22/proof/replay.py`. The receipt is `results/packing-swarm/round22/proof/receipt.json`; fresh lossless compiler/source/olean/audit runs live under `archive/packing-swarm/round22/proof/run-*`. Round20 and Round21 sources are unchanged.

## Proven dictionary

The candidate uses FW4/WG2; the independently implemented strongest reader uses FW8/WG1. Both cover eight 32-feature groups and32 tokens per CTA. Group division/modulo reconstructs each group and proves unique warp/subgroup ownership. The full output coordinate dictionary reconstructs feature and token from CTA tile, warp, subgroup, MMA half, row component, lane quartet and output component. Bounded coordinates are injective, including the complete native-stride output address; the literal `m=q*2+mm` expression maps to the same feature.

All merged16480 GDN,14336 attention and34816 FFN features are included, not isolated slices. GDN has515 groups,64 complete CTAs and a three-group final CTA. B1/8/32 each occupies one32-token CTA; output guards crop padded tokens, not source features. The tail retains full16480 row stride. The byte dictionary includes prepared binary512-byte groups and original64-byte W128 scale groups, unchanged lane/word/scale coordinates and the activation row/chunk XOR involution. The64-row padded activation tapes retain all40 K128 stages; a32-row second half preserves the original swizzle.

| Shared component | Byte offset within stage | Bytes |
|---|---:|---:|
| Eight binary groups |0|4096|
| Original weight scales |4096|512|
| Primary q |4608|2048|
| Primary scales |6656|256|
| Innovation q |6912|2048|
| Innovation scales |8960|256|

The stage is9216 bytes. Eight stages occupy73728 bytes; full and empty barrier arrays use128 more, making the ring73856 bytes. The two once-staged32-row FP32 global scales use256 bytes after the ring, making actual shared allocation74112 bytes. Stage segments, stage indices, barrier slots and global-scale slots have injective, disjoint addresses; every stage byte has a segment reconstruction. This fixes the initial73984 arithmetic in the assignment. Neither73856 nor74112 is an allocator/occupancy result.

The producer event dictionary contains exactly one W, one WS, and one q/scale event per pass per K128 per CTA. Global scales have one event per pass per token slot, shared across all consumers; live slots read the original value and padded slots deliberately contain zero. On the GDN tail only1536 weight bytes and192 weight-scale bytes are staged; the mathematical staged-read contract covers exactly these live ranges. No equality is asserted for unstaged inactive weight slots. The fixed stage allocation still includes all eight groups.

## Ordered floating map

MMA, FMA, multiplication and addition are arbitrary opaque primitives. Each step performs low MMA from zero, high MMA from the low result, then the original W128-scale FMA into that pass's accumulator. The ascending fold recursion preserves the40 K128 order. A paired fold has exactly the projections of two independent folds; a payload dictionary equality carries through the complete fold, without distributivity or reassociation.

Primary is `add(mul(acc0,AG0),prior)`. Final is `add(mul(acc1,AG1),primary)`. The theorem equates this exact ordered pair to two sequential literal MODE1 calls. The complete conditional paired-reader theorem requires equality of each original/staged step, not an assumption that any arbitrary reader is numerically equivalent. No floating primitive equations or native tensor semantics are introduced.

## Complete observation and release contract

`PairedRelease` requires the inherited immutable whole-backing release, the exact positive source row domain, a complete merged feature domain, and every coordinate of six mandatory observations: primary output/status/owner and innovation output/status/owner. It also requires exact manifest identity and a drain for every manifest entry. A missing coordinate or pending entry excludes release. The inherited full source/norm/fold/continuation/control/host-layer observations and completed state/source images remain mandatory. Release applies only to the exact whole allocation; uncertainty retains its pointer and identity. This is a structural lifetime contract, not a controller implementation proof.

## Scope and custody

Pinned Lean4.33.0/Std compiles64 new theorems and rechecks70 inherited theorems. Replay audits all134 with `#print axioms` and rejects unchecked declarations and warnings. Permitted axioms are only the standard `propext`, `Quot.sound` and `Classical.choice`; finite swizzle proofs use kernel-checked `decide`, not native execution. Source/compiler/Std/native-contract/literal identities and raw logs are hash-bound in the receipt. Refresh replay after final native-contract edits or documentation edits to keep every source hash current.

Definitions-only reconstruction, exact shared geometry, producer-load event counts, opaque ordered-fold congruence and complete-release prerequisites are checked. The event and staging dictionaries specify required behavior; they do not certify CUDA execution or mbarrier scheduling. C++/CUDA/parser/controller refinement, native words/faults/time, physical/cache/occupancy/granularity/residency, engine/whole-model/task/HTTP remain unset. No new capacity bound, source nominee or symmetry-based search narrowing is claimed; shared128 achieved/129 open and CW13-only29 optimum remain unchanged.

Development failures (division/modulo substitution, unreduced address association/Boolean match expressions and unused-variable warnings) were repaired directly; failed source/log snapshots remain retained. No GPU, spending, engine mutation, separate commit or persistent job belongs to this sidecar. The main combines these paths into the Round22 publication.
