# Round22 — paired complete reader

CPU/source authorization only. No GPU calls, spending, rental, engine/demo mutation or new packing bound. [Round22 brief](../../../../ROUND22.md), [receiver](../../../../results/packing-swarm/round22/RECEIVER.md), [report](../../../../results/packing-swarm/ROUND-22.md).

`native/generate.py` emits two concrete CUDA kernels: candidate FW4/WG2 and separately instantiated strongest FW8/WG1. Both cover eight32-feature groups, TOKW32, forty ascendingK128 blocks and two independent pass accumulators. Binary weights stay native packed coordinates: literal `binary_e2m1` register composition, no dense tensor. Each CTA requests packed weights and half scales once perK, both tapes once perK and both global scales once perlive row. The ring is9216B×8+128B barriers=73856B, plus256B global scales=74112B.

`native/family.cuh` retains original21 direct R4, independent originalFW8, current19 per-reader/shared/separated controls, materialized20 downstream controls and their original source/Norm factories. It adds paired candidate/strongest full primary/final/owner observations and a literal originalFW8 primary snapshot. Only paired output supplies the new borrowed full merged views. The original Binding/Protocol gains eight complete pass manifests38..45 perlayer; existing manifests remain. Complete source/Norm/all outputs are reobserved before whole release; failed/uncertain backing remains owned.

Build one phase per bounded command:

```sh
python3 method/packing-swarm/tools/round22/native/generate.py
python3 method/packing-swarm/tools/round22/build.py plan archive/packing-swarm/round22/compiler1
python3 method/packing-swarm/tools/round22/build.py phase archive/packing-swarm/round22/compiler1 0
# continue exact indices1..10; no retry into a landed phase
```

Each phase enforces52s, immutable source hashes and predecessor receipt. DeviceO3/defaultcontraction/no-fastmath, actual hostO1 appended last. `native22 --describe-native22` reads source ABI only. Invalid/missing authorization returns `GPU_calls:0` before initialization. `--serve-native22` is not authority: CUDA execution still needs the operator's new yes.

All new source/evidence is committed to the private programme remote and incremental bundle; delivery lives outside reclaimable writers. Exact clone paths are not promised after native release. Reconstruct source from the delivery commit; retained external original input/evidence paths remain pinned dependencies.
