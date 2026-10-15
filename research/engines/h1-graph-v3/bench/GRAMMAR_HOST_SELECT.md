# Host-owned batched grammar selector

`src/grammar_host_select.h` takes a complete CPU-resident draft prediction block,
then advances each private grammar matcher and selects every token without CUDA.
It implements the grammar route's exact top-16 head cutoff followed by the greedy
codebook selector. This is a full-head/bulk-transfer route, not a sparse-head
speed claim.

## Engine integration

1. Cache both predecessor and successor codebooks at model load as raw BF16
   `uint16_t` arrays, each `[VOCAB][256]`. Retain the exact bytes loaded for the
   device arrays; do not convert or quantize them. Together they occupy
   254,279,680 bytes (242.5 MiB) at vocabulary size 248320.
2. Allocate reusable host (preferably pinned) FP32 buffers for
   `[blocks * proposals][VOCAB]` logits and `[blocks * proposals][256]` hidden
   projection values. Prediction rows exclude the anchor query. Enqueue one bulk
   copy for each tensor on the drafter's stream after the full block is ready,
   then synchronize once. There are no subsequent per-token device uploads,
   kernels, token copies or waits in this selector.
3. Supply `GrammarHostSelect` with explicit geometry, all pointer/element-count
   spans, the block anchors, token/confidence output spans, one reusable mask
   scratch span of `(vocab + 31) / 32` words, and `H5GrammarMask` plus its context.
   `rank` must be 256; `proposals` is 1..64. Buffers must not overlap and the
   callback must write only the provided mask span. The selector clears the mask
   before each callback and verifies its returned count against its bits.
4. Call `grammar_host_select(args)` and handle its closed `GrammarHostError`.
   Only `None` means the complete block is available. On error, only the leading
   `completed_rows` outputs are valid; discard the private matcher because its
   callbacks have already advanced. Negative callback results are retained in
   `callback_status`, and exceptions are caught at that foreign boundary.
5. Add this header to the owning engine object's Makefile prerequisites. Assert
   the engine's `DF_RANK == GRAMMAR_HOST_RANK` and
   `DF_TOPK == GRAMMAR_HOST_TOPK` at integration.

The callback receives `previous == -1` at position zero of every block, then the
last chosen token. The initial anchor still supplies position zero's predecessor
codebook. Mask-singleton positions bypass logits/projection/codebooks and yield
`FLT_MAX` confidence, as the engine grammar route does.

For branching masks, nonfinite logits are excluded, head candidates order by
value descending then token ID ascending, and score ties retain that candidate
order. FP32 multiplication rounds twice, warp contributions use the XOR shuffle
association (16,8,4,2,1), and the eight warp partials are added sequentially to
the logit. Confidence is the best score minus the second best score; a branching
mask with just one finite logit yields positive infinity. No-finite-candidate
and nonfinite-transition cases are explicit errors rather than invented tokens.
Build without fast math, with round-to-nearest and gradual underflow (CPU
FTZ/DAZ disabled); unsupported arithmetic settings are rejected before callbacks.

## Seconds-long fixture

```sh
g++ -O3 -std=c++17 -Wall -Wextra -Werror -Isrc \
  bench/grammar_host_select_test.cpp -o /tmp/grammar-host-select-test
/tmp/grammar-host-select-test
```

The fixture covers cutoff versus transition score, two distinct tie rules,
FP32 reduction association, anchor/prefix dependence, forced-token bypass,
nonfinite-logit filtering and error boundaries. End-to-end latency and CUDA
parity measurement belong to the integrated engine experiment.
