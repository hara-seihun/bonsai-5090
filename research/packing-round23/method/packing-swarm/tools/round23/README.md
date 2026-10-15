# Round23 — retire local packed fragments

Private CPU/source-only. [Report](../../../../results/packing-swarm/ROUND-23.md), [receiver](../../../../results/packing-swarm/round23/RECEIVER.md), [price](../../../../results/packing-swarm/round23/COST.md), [next](../../../../ROUND24.md).

`native/generate.py` emits independent candidate FW4/WG2 and strongest FW8/WG1. It serializes q-group fragments and binds individual ordered folds to volatile PTX RN FMA without changing per-coordinate arithmetic. Every CTA still stages eight32-feature groups×two passes×32padded rows once; original scales, full output/owner images, direct consumer views, all retained controls and whole release lifecycle remain.

Source lifetime: candidate native A words32→16 and half-scale floats8→4; independent accumulator floats128/thread remain versus64 in control. Candidate activation fragment requests double, equal control in aggregate. Compiler1 q-serialization and compiler2 explicit fold-retirement both retain255 candidate registers with28B spill stores/loads; equally strengthened control131/zero. No source nominee.

Bounded phased build (fresh directory, indices0..10 once each):

```sh
python3 method/packing-swarm/tools/round23/native/generate.py
python3 method/packing-swarm/tools/round23/build.py plan archive/packing-swarm/round23/compiler-FRESH
python3 method/packing-swarm/tools/round23/build.py phase archive/packing-swarm/round23/compiler-FRESH 0
```

Final retained directory is `compiler2`; compiler1 includes a hash-bound complete39-file source snapshot. Each phase enforces52s plus immutable closure/predecessor receipts; deviceO3/sm120a/defaultcontraction/no-fastmath, actualhostO1 appended after injectedO3. `compiler_report.py DIRECTORY` consumes exact cubin/SASS and exercises five zero-GPU guards. `native23 --describe-native23` reports source ABI only; never run authorized selectors without new human GPU authority.

CPU and proof replay paths are in `cpu/README.md` and `proof/STATE.md`. Cost generator supports source-bound generation/check and expansion of the complete hash-pinned predecessor invoice. No saved producer is regenerated. `custody.py` seals exact current source/evidence, first-attempt snapshots, private tape hashes and unchanged canonical programme/engine144ref captures.

Publication: combined private commit/branch/bundle plus external `~/work/packing-mma-round23-delivery.json` and stable `~/work/packing-mma-round23-evidence/snapshot`. Reconstruct by exact commit or incremental bundle prerequisite, not by an assumption that native release retains the writer cwd. Borrowed absolute inputs and original sealed evidence remain separately pinned.

Native words/time/fault/allocation/physical/cache/occupancy, concrete CUDA/controller refinement and full-model/logit/task/HTTP acceptance remain unset. No GPU/spend/rental/service/demo/engine mutation, new packing bound or quality claim.
