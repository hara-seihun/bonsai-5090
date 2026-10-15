# GPU grammar engine integration

`src/gpu_grammar.h` owns the C ABI. `gpu_grammar_runtime.cuh` owns cached token legality/advance. [Compiled-token cache](GPU_GRAMMAR_TOKEN_CACHE.md) owns boundary compilation and persistent plan storage. `gpu_grammar_select.cuh` owns legal top-16/argmax, and `gpu_grammar_engine.cuh` contains Engine's upload/staging/head helpers (included inside Engine).

The vocabulary uploads once; plan handles belong to their engine. Upload/release are request-boundary operations. A pending target cannot release a referenced plan. Target staging translates host handles into device descriptors; it does not match token bytes on the CPU. Null plans are unconstrained, or use an existing CPU baseline mask in a mixed target batch.

All target rows initialize their state with the supplied token prefix. Selected head rows compute legality, choose, and advance on the GPU. The usual round copy/synchronization returns predictions and physical-row `RowResult`s. Unselected rows report initialized prefix state with allowed=0. `h5_gpu_grammar_results` only copies already-ready host results; it never synchronizes. Return -8 means device grammar error, not a failed round: successful predictions and all row results remain available. Other returns: -1 arguments, -2 missing vocabulary/results, -3 invalid/busy handle, -7 unsupported route, -9 CUDA/runtime failure, -10 token-cache budget, -11 token representation overflow.

Draft blocks begin at the request state after its emitted anchor. Each fixed-loop position enqueues legality, row-zero activation preparation, conditional head, top-16, ordinary codebook selection and advance. There is one normal end copy/synchronization for all positions, including errors and EOS continuation. Results contain the last position's before/after state and sticky error per block. There is no CPU matcher callback or intermediate D2H decision. CUDA events report body and selection seconds after that same sync; token-wait time/count are zero.

Head routing uses the device legal count: singleton bypass, 2..512 exact prepared sparse rows, >512 dense head. GPU-only all-head target batches up to128 rows classify wide states on-device, prepare their source rows together, and project the full head once. Fixed graph geometry pads unused activation rows; selectors address only real wide rows. Selected or mixed CPU-mask batches retain per-row projection. Sequential draft states retain per-position projection.

Wide top-1/top-16 selection uses128 parallel256-thread partitions plus a final exact merge. Compact sets and singleton/error rows bypass the wide scan; top-16 padding retains a safe original ID and negative-infinite score. Dense-row mapping is device data, so batched target selection and graph replay share the same selector. Both projection paths use the ordinary head's frozen configuration and validated `H5_HEAD_KS` override, preserving means and split arithmetic. FP4 sources are prepared into activation row zero, never addressed as arbitrary row-major slices. Rejected projection predicates acquire their predecessor before testing the count and never stream weights, clear output, or fold partials.

Draft graph kind 11 captures the complete fixed loop. Target kind 12 captures all-head, GPU-only batches; mixed CPU masks and selected-head batches execute eagerly. Plans/states/prefixes are copied data, not graph keys. Prefix transfers have fixed max_rows size. Vocabulary replacement destroys these graphs before releasing old vocabulary allocations.

The native route requires FP4 activation/head preparation and excludes synchronous profiling, sparse-logit checking, stream-K head overrides, relaxed target selection, host logits, target visual/rope rows, pruned drafter heads, draft nesting, alternatives and draft LZ. CPU callback routes remain measurement baselines.

## Device contracts

Build (on the device compiler host), then run on the assigned idle GPU:

```sh
make DEFS='-DH5_PROFILE_SCALE_CARRIER' gpu-grammar-runtime-probe predicated-gemm-probe
./gpu-grammar-runtime-probe
./predicated-gemm-probe
```

[Runtime probe](GPU_GRAMMAR_RUNTIME.md) covers boundary token compilation, prefixes, EOS, errors and compact-list boundaries. `gpu-grammar-select-probe` checks exact sorted selection across compact/wide sets and graph replay;328 executions pass on the campaign5090. [Predicated projection probe](PREDICATED_GEMM.md) covers graph replay, inaccessible skipped operands, output preservation and active arithmetic. These are contract probes; full-model quality/timing belongs to the [campaign ledger](../results/sparse-head-1010/README.md).
