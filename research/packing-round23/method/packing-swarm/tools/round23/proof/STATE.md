# Round23 — independent-coordinate q serialization

Owner: packing-swarm proof sidecar. Source: `SerializedGroup.lean`. The three predecessor modules are exact copies, SHA256-pinned in `replay.py`: Round20 `NativeJoin`, Round21 `StridedPrep`, Round22 `PairedTile`. They retain the full source/Norm, complete shape, native-stride, shared/tape and whole-backing release dictionaries.

## Map and theorem

The original per-K consumer schedule is pass/n/m/i. The shortened schedule is q/pass/n/mm/i, with m=2q+mm. Both FW4/WG2 and FW8/WG1 enumerate exactly the same unique coordinates. The schedule-permutation proofs use kernel-checked `decide`, not native execution. The general theorem proves any permutation of independent coordinate updates preserves the state for arbitrary value types and arbitrary update functions. Only writes to different accumulator coordinates commute; no floating operation or differing K stages are commuted.

The outer K recursion is unchanged. Every valid coordinate receives exactly one update per stage, and its projection is the literal ascending fold. At K=40 both schedules are equal and each coordinate is precisely its forty-K128 fold. Low MMA from zero, then high MMA, then scale FMA remain ordered opaque functions. Primary output is add(mul(primary accumulator, primary global scale), prior); innovation output is add(mul(innovation accumulator, innovation global scale), primary). The full saved-payload dictionary carries through this exact ordered pair.

The local m quotient/remainder reconstructs q/mm uniquely. Original and q-local saved-weight fragments, byte component order 0/2/1/3, weight-scale addresses and full-stride feature coordinates coincide. Every natural feature/token/pass coordinate reconstructs a scheduled coordinate; imported Round22 supplies complete merged16480/14336/34816, B1/8/32, GDN three-group tail, padding and all output/lane ownership dictionaries. No feature domain is narrowed.

Declared fragment storage is candidate32→16 words and8→4 scales; FW8 remains16 words/4 scales. Accumulators remain128/64 values. Candidate activation-fragment load requests become16 rather than8 per K, while FW8 remains8. These are source structure counts, not measured register liveness, cache requests or occupancy.

## Replay and custody

Run `python3 method/packing-swarm/tools/round23/proof/replay.py --root "$PWD" --contract method/packing-swarm/tools/round23/native/paired.cuh --contract method/packing-swarm/tools/round23/native/generate.py`. Each stage has a bounded subprocess deadline; the complete replay has a53-second deadline. Results are `results/packing-swarm/round23/proof/receipt.json`; immutable sources, oleans, every compiler/audit log, stage receipts and the receipt copy are under `archive/packing-swarm/round23/proof/run-*`. The audit covers every theorem in all four modules, rejects unchecked declarations/warnings/unrecognized output, and permits only standard propext/Quot.sound/Classical.choice. Source/compiler/Std/native-contract hashes bind the result. Rerun after the last native or proof-source change.

## Scope

Definitions-only dictionary and schedule equality are checked. Float primitives remain opaque, including the proposed native `ordered_fold` binding to volatile PTX `fma.rn.f32` in both arms; the proof supplies no new concrete primitive semantics. C++/CUDA/controller refinement, native numerical/fault/time evidence, physical/cache/occupancy/residency and full-model/task/HTTP remain unset. The mathematical staging/load and release dictionaries are contracts, not GPU traces. No capacity/impossibility or symmetry-based search narrowing claim is made. Shared128 achieved/129 open and CW13-only29 optimum remain unchanged. No GPU, spending or engine/demo mutation belongs to this sidecar.
