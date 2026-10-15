# Native23 serialized packed reader

`--serve-native23` is a source selector, not GPU authority. Invalid/missing authorization and `--describe-native23` are zero-GPU paths. This round executes CPU/source/Lean/compile only.

## Changed map

Both separately emitted kernels retain complete256-feature×32-padded-row CTA staging, forty ascendingK128 folds, both primary/innovation native tapes, unchanged packed binary weight coordinates, original unequal half scales and once-CTA global scales. Candidate FW4/WG2 now visits one32-feature group at a time: local `a[2][2][4]` and `wsc[2][2]` die after that group's two passes and all four N8 blocks; the next group does not require their values. Strongest FW8/WG1 gets the same local-fragment arrangement. All eight groups still stage together; no decoded dense weight tensor exists.

The first build preserved CUDA `fmaf` folds. The second binds each individual same RN fold to volatile inline PTX `fma.rn.f32`, retiring the MMA result before later volatile MMA operations. No K fold or two-half contraction is reassociated. Epilogue remains default contraction with literal primary=`add(mul(acc0,g0),prior)` then final=`add(mul(acc1,g1),primary)`. Primary and final snapshots and independent owner arrays are complete.

Candidate still retains128 independent accumulator floats per consumer thread versus64 for strongest. Moving q outside pass/N requires two LDSM+activation-scale observations per previous candidate observation; aggregate requests now equal strongest. Complete global staging bytes, layouts, output stores, forty fold updates, two-pass staging and CTA/reader launch counts are unchanged. Compiled resources are evidence distinct from this source payload.

## Unchanged boundary and controls

GDN16480, attention14336, FFN34816; B1/8/32; original full row strides; immutable RowViews into unchanged original FFN/GDN factories. Original source/Norm, current19 per-reader/shared/separated, materialized20, original21 direct candidate, independent literalFW8 primary/final and exactA8/INT8 endpoints remain. All original conv/GDN/attention/SiLU/down/residual/head/state/KV continuations remain.

Native23's family, stage, views, binding, session, worker and transport reproduce the Native22 complete lifecycle at the same boundaries, with selector/namespace/path names changed. Primary/final/status/owner observations enter complete Frozen images and manifests38..45. All source/Norm/direct/current19/paired observations after downstream consumption, manifest/source/consumer drains and fence precede identified whole-allocation release. Failed or uncertain effects retain original backing identities. Interior views cannot free an allocation.

## Evidence scope

CPU witnesses check the explicit serialized coordinate schedule on exact inherited saved native/raw weights, half/global scales, tapes, primary/final words and owners; producer regeneration is absent. CPU `std::fma`, multiplication/addition and dictionaries are not CUDA MMA/floating primitive acceptance. Lean independently models coordinate update schedules and opaque per-coordinate folds; no concrete C++/CUDA/controller refinement follows. Compiler RN instructions and resource counts are not native words, cache traffic, occupancy or timing. Full model/logit/task/HTTP acceptance remains unset. Shared128 achieved/129 open and CW13-only29 optimum remain unchanged.

Exact private source, all build attempts and stable artifact custody survive release; the writer's exact cwd does not.
