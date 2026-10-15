# Complete shortened-q saved-tape CPU witness

`check.cpp` executes the q-local Round23 CPU map and both independent retained full-live Round22 CPU maps against the exact saved producer tapes and three actual saved output/trace controls (`candidate`, `mergedFW8`, `current19`). `scoped.hpp` loads one q's two M E2M1 fragments and their scales, consumes pass,n,mm in that lexical scope and drops those fragments before the next q. All primary/innovation accumulators remain independent and each output's K128 fold owner must advance exactly 0 through39. The raw and native K64 halves are compared individually before the K128 fold; no float reassociation or producer regeneration occurs.

`replay.py` reads exact private remote commit `9c4ed70089114706200215bf60910dbed2c35c95`, authenticates the predecessor receipt, custody pin, manifest, every raw member, source/norm/owner snapshots and saved outputs/traces. It creates and deletes one per-case compressed archive in the external sidecar scratch directory; no raw tape archive enters the managed writer. All preserved q/scales/global arrays and raw/native weights retain their identities.

Fresh build (under55s):

```sh
python3 method/packing-swarm/tools/round23/cpu/replay.py --build
```

Each case is separately bounded under45s inside the replay, with shell timeout55s. Run all combinations of `--case input0|input3|ffn0`, `--batch 1|8|32`, `--kind ordinary|ubsan`, then seal against the coordinator's actual final native source:

```sh
python3 method/packing-swarm/tools/round23/cpu/replay.py --case ffn0 --batch 32 --kind ubsan
python3 method/packing-swarm/tools/round23/cpu/replay.py --seal --native method/packing-swarm/tools/round23/native/paired.cuh
```

The ordinary and UBSan case receipts must match exactly. Results and receipts live in `results/packing-swarm/round23/cpu`; immutable compiled source/binaries/build/run logs live in `archive/packing-swarm/round23/cpu`.

Coverage: full GDN16480 with tail3 groups, attention14336, FFN34816; B1/8/32; all padded64 input tape bytes; every active primary/final output/status/owner and all40 fold trace digests. Per reader arm, per build across nine cases: 215,272,960 K128 dot/scale/fold checks, 430,545,920 individual K64 half checks, 5,381,824 complete pass outputs and owners. The same complete scope also executes for each inherited full-live Round22 map.

The CPU witness establishes byte/coordinate/orderedfold agreement. It does not execute CUDA MMA or measure register allocation, native numerical words, allocation/fault/time, cache or occupancy. Saved trace digests bind old observations but do not reconstruct each old intermediate word. Layer3 input and FFN input retain their inherited contract-only trajectories; no controller/full-model/task/HTTP promotion or packing-bound change is made.
