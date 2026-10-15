# Native affine GU/down carrier and joint consumers

Owner: this continuation, linked from MIXED.md. Original provenance and native prime witnesses remain in README.md. Existing RTX 5090/mutex only; no engine, serving, driver or rental change.

## Whole route

Two independent full15-value FP4 activations, independent ternary gate/up weight pairs, ternary down weight:54,675 source cases. Source observation is saturating RNE Q8 of `SiLU(g)*u*down`, using explicit binary32 and CUDA `__expf`, output scale1.

Prepared weight coordinates are `(g0+64*u0,g1+64*u1)`; activation coordinates are `(2*a0,2*a1)`. A genuine nonuniform-K8 TF32 MMA contracts them into the opaque integer coordinate `2g+64*2u`. A second MMA consumes this accumulator as C and adds4096 times the down role. Neither conventional g/u nor SiLU exists between these instructions.

`native_affine_gu.cu` / `affine-gu-launch.log`:54,675 inputs,6,998,400 all-four/lane fragment checks,zero errors,2,115 physical states,zero Q8 collisions. Immutable observer SHA256 `fab9aff7e926c4402c12f3111796064dbb5b353080b93c8c56964f19b035af8d`; native image `4ff372a2d30ffaab016ab518450b77fcc1f9725ffd3d35bb28db0e93e66c6db0`. Producer2,448B. Build22registers/no spills or barriers.

`AffineContract.lean`, checked without sorry, proves integer injectivity of `g2+64*u2+4096*role` for doubled sums g2/u2 in[-24,24], even with arbitrary integer roles. It also proves preservation of any observer with a shared additional context. Native floating-point conformance, actual source/SFU semantics and numerical scale handling remain empirical. This is a finer carrier than the95-valued terminal Q8 map.

## Native readers and complete prices

The first final GPU reader uses a90,504B dense table indexed by a paid float-to-integer conversion. Each entry retains the full physical float word and target; empty/wrong keys fail explicitly. A sibling initializes the first MMA's C with16384*role, then normalizes by1/4, fusing the second MMA without changing the final observer. Both pass13,996,800 native readout checks, missing0/wrong0.

The full-tile sibling assigns128 distinct outputs per MMA:243 permuted weight/down rows ×225 permuted activation columns. Every54,675 endpoint passes. Row/column coordinate axes occupy936B. Cached-global preparation expands row labels tofloat4 (three fields plus padding) and activation columns tofloat2,5,688B total. All producer reads, TF32 conversion, C formation, MMA, normalization, index/key comparison, final observation and stores are timed.

| Matched harness | Native µs | Direct Q8 µs | Full-map lookup µs |
|---|---:|---:|---:|
| Warp-redundant two-MMA |5.300|1.605|1.613|
| Warp-redundant one-MMA+C |5.137|1.605|1.613|
| Full-tile constant producer |2.739|1.837|1.472|
| Full-tile cached-global producer |1.983|1.856|1.493|
| Full-tile packed32 final reader |1.982|1.860|1.491|

Every native realization loses to its matched controls. Cross-layout differences are progression, not matched speedups. Cached-global and packed kernels use18registers/no spills/local/shared/barriers; the constant tile uses20. Final output218,700B; lookup218,700B. Source-oracle FP4 table60B is separately resident.

Packed32 reader:45,252B. Exact source physical words have zero low9bits; remaining23bits and signed Q8 target8bits fit one word. Host checks this domain, and the GPU checks the reconstructed full physical key. The prepared operands are exactly TF32-representable, checked on the host; reinterpretation removes redundant RNA conversions. Half the observer bytes establishes no measured gain.

Cached-global preparation:91.723ms includes CUDA context startup, label/observer generation, allocation and transfer;0.335ms control table/validation;3.045ms graph capture/warmup. Packed sibling:89.308/0.362/3.019ms respectively. These are separate wall observations, never hidden inside or substituted for endpoint event timing. Graphs have100 complete kernels,7 rotated rounds.

## Context-parametrized analytic final observation

`native_affine_context.cu` removes the native target table. Its final observation alone unwraps g2/u2/down role and applies dynamic scale context plus SiLU/Q8. **This sibling recovers conventional g/u inside the final observer**; it is not a decoder-free result. No conventional values exist between carrier instructions.

Activation scale: positive finite E4M3 codes0…125 (code0 is zero), selected by `(id*37+11)%126`. Up scale: dyadic `1+j/16`, j=`(id*13)%16`. Checked domain is54,675 source/context pairs,covering1,008 distinct scale pairs; it is not all independent combinations of scales and sources. All endpoints pass. This common-scale two-term map does not establish independent16-slot scale-group or full-K model parity.

Native analytic3.007µs vs same-run direct2.139µs/lookup1.470µs;21registers/no spills/local/shared/barriers. Native observer table0B, scale table504B, labels5,688B, axes936B. Preparation85.501ms context/labels/scale allocation-transfer,0.307ms control table/validation,3.690ms graph capture/warmup. Source/binary hashes and actual SASS are in `affine-context-sass-custody.json`.

## Two serial groups with independent scales

`native_affine_groups.cu` composes two genuine nonuniform-K MMAs through shared C. The first carries one opaque doubled GU pair; the second's weight labels are scaled4096 and append another pair without recovering the first. Final normalization by1/4 yields `g0+64*u0+4096*g1+262144*u1`, each component a doubled two-term partial sum in[-24,24]. Range±6,391,320 fits exact binary32 integer representation. The final observer alone unwraps four sums, applies two independent E4M3 group scales, separate dyadic gate/up weight scales and down role, then SiLU/Q8.

`AffineContract.lean` also proves this four-component integer coordinate injective without sorry. Native producer/accumulation/SFU semantics remain outside that theorem.

Probe:1,048,576 deterministic sampled inputs,134,217,728 all-four/lane checks; physical-word/domain/Q8 errors0 and context-fibre collisions0. Joint physical/context image1,048,575 states. This is not exhaustive over source groups or scale combinations.36registers/no spills/barriers. Complete16MiB image, binary, source and receipts are retained through `affine-groups-custody.json`.

Matched endpoint:256 prepared weight-pair rows ×256 activation-pair columns,65,536 cases.128 distinct outputs per two serial MMAs; all physical words and endpoints pass. Native3.008µs vs same-run direct2.756µs/lookup0.997µs.24/25/10registers respectively;zero spills/local/shared/barriers. Labels8,192B; axes2,048B; scale table504B; source FP4 table60B; native observer table0B; output and control lookup262,144B each. Preparation86.839ms labels/context startup/allocation/transfers;0.300ms control table/validation;3.726ms graph capture/warmup. Every recurring producer/operand-placement, two MMAs, normalization, final interpretation/scaling/SFU/Q8 and store is timed. Actual SASS:probe888static instructions/2HMMA; timed native592/2HMMA. Counters are unset.

This supplies a new serial carrier with two independent scale contexts and a priced losing realization. Full-K/learned-model acceptance remains open. [K16.md](K16.md) and [K64.md](K64.md) now own independent16-term groups, packed producers, native C/RTZ observations and the collective consumer. Learned full-K acceptance and a cheaper complete route remain the target; expanding finite constants alone is not that target.

## Custody and scope

`affine-custody.json`, `affine-sass-custody.json`, `affine-context-sass-custody.json` bind recovered sources, images, observers, receipts, builds, executable hashes and actual SASS. Packed/context executables are also retained locally. Completed measurements were recovered, not replayed to retrieve them. Initial affine recovery export omitted its print and returned no body; the reader was repaired and existing artifacts fetched without GPU replay. Original warp-readout stdout omitted a final JSON quote; its outer receipt/raw stdout remain intact and any normalized projection must name that formatting repair. Later harness stdout is valid JSON.

Actual SASS timed entries: warp two-MMA80instructions/2HMMA; fused-C72/1; constant tile168/1; cached-global tile144/1. Static counts include edge/control paths and do not measure utilization. Hardware counters remain permission-unavailable; no driver change.

Native finite sufficiency, integer injectivity, complete pricing and actual-model performance are separate. Full-K/independent-scale reduction, learned weights, quality/TPS/HTTP and engine adoption remain open. Search continues through joint carriers/consumers rather than terminating at this reader's loss.
