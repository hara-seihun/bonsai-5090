# Native22 complete paired reader

The new selector is `--serve-native22`; invocation is not GPU authority. This publication performs CPU/source/Lean/compile only. Invalid/missing authorization and `--describe-native22` are zero-GPU paths.

## Actual map

Two concrete kernels consume unchanged complete saved primary/innovation tapes and independently prepared merged weights. Candidate has four consumer warps and WG2; strongest control has eight consumer warps and WG1. Both cover256 features×32 padded rows perCTA. Forty ascendingK128 folds use literal two-half native FP4 MMA and original half-scale `fmaf`; each pass has its own accumulator. Binary sign bits compose directly into the literal E2M1 register operands. No decoded floating weight tensor is materialized.

EachK tile stages8×512B packed weights plus8×64B half scales, primary2048B q/256B scales and innovation2048B q/256B scales. Eight9216B stages plus128B full/empty barriers form73856B. Each global scale is read once perlive row/pass by the initialization warp into256B shared after the ring. Dynamicshared=74112B for both. Producer completion expects exactly the live tail weights/scales plus both full padded32-row tapes; empty barriers count4/8 consumers respectively. GDN16480 has515groups and a three-group final CTA; attention14336 has448groups, FFN34816 has1088groups. All complete outputs, B1/8/32 and original full row strides are covered.

The epilogue is exactly ordered: primary=`add(mul(acc0,g0),prior)` then innovation=`add(mul(acc1,g1),primary)`. It stores the complete primary snapshot and final output separately, plus one complete owner array perpass. No sum of accumulators replaces the two additions. Floating primitives stay opaque in Lean; default native contraction is retained and CUDA refinement remains unset.

## Controls and ownership

Original21 direct candidate and independently reloaded originalFW8 still run both literal separate passes. Current19 per-reader R4/sharedFW8/separatedFW8 and original preparation, paid20 materialized downstream controls, exactA8/INT8 endpoints and complete original conv/GDN/attention/SiLU/down/residual/head/state/KV consumers remain. One extra literal FW8 primary-only control is computed and compared to both paired primary snapshots; paired finals compare to both old direct finals. Strongest pairedFW8 receives identical staging, paired weights, global reuse, full output/owner observations and full-tile layout.

Paired final allocations supply immutable RowViews and unchanged full FFN/GDN borrowed factories. Whole allocations remain bound to source/Norm/domain/stream/owner/ticket. Interior pointers have no release authority. The complete original direct outputs, paired primary/final, every current19 output, original input and Norm are reobserved after downstream consumers. Primary/final/status/owner snapshots for each newarm are written to complete Frozen images; manifest38..45 adds two passes×two arms×two families perlayer, while original manifest groups remain. All manifest/source/consumer drains and fence precede identified whole release. Failed/uncertain acquisition/launch/copy/release retains backing until reconciliation.

## Boundaries

CPU saved-word replay demonstrates exact integer/halfscale/orderedfold/output dictionaries on retained independently computed tapes; it does not run CUDA MMA/quantizer/fma or trained post-attention trajectories. The host borrowing implementation is inherited unchanged21, except the explicitly extended manifest and new paired allocation/view routing. FullB8/B32 region coverage is not tensor-wide controller Seg/QBlk scheduling. Compiler resources/static sites do not determine occupancy, cache-effective traffic or native time. New native words/allocation/fault/time/physical/engine/full-model/logit/task/HTTP remain unset. Shared128 achieved/129 open and CW13-only29 optimum are unchanged; no new capacity claim or search restriction.
