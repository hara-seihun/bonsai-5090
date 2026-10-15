# GPU grammar runtime

[Engine integration](GPU_GRAMMAR_ENGINE.md) owns the ABI lifecycle, conditional head, graph routes and paired probe targets. [Token cache](GPU_GRAMMAR_TOKEN_CACHE.md) owns hybrid storage, cache budget/lifetime, cold-cost telemetry and cache probes.

`src/gpu_grammar_runtime.cuh` owns compiled-token lookup and row lifecycle. `src/gpu_grammar_token_compile.cuh` owns the boundary-only byte-DFA compiler. The engine owns descriptor upload, allocation, vocabulary validation, stream ordering and selection. `BytePlan` transitions are state-major, 256 entries per state, and reject with `INVALID`. Vocabulary offsets have `size + 1` entries, EOS flags have `size` entries, and `eos_ids` lists EOS IDs in ascending order. The engine validates uploaded array bounds and descriptor lifetimes before enqueueing work.

## Boundary token compilation

`launch_token_counts(BytePlan, vocabulary, counts, stream)` counts legal raw token transitions for every byte state. The caller zeros `counts` first. At this boundary the engine reads counts and allocates storage: states with at most 512 legal tokens use sorted `sparse_ids`/`sparse_next` arrays, and larger states use vocabulary-sized `wide_next`, `ceil(size/32)`-word `wide_mask`, and 512 sorted `wide_ids`. `offsets[state]` denotes a sparse entry offset or a wide row index according to `counts[state]`.

`launch_token_chunk` compiles and packs up to `TOKEN_COMPILE_CHUNK` (16) states at a time. Its caller-provided temporary holds `state_count * vocabulary.size` uint32 next states and can be reused by subsequent chunks on the same stream. Compilation uses generic byte semantics: accepting states retain whitespace continuations, EOS maps to the accepting state itself, and empty non-EOS entries reject. Byte walks occur only in these boundary kernels; prefix consumption, legality and advancement use compiled tokens thereafter. No compiler wrapper allocates, reads counts to the host or synchronizes. The source DFA and vocabulary must be boundary-validated, and output allocations must match the returned counts. Changing either source requires recompilation.

`launch_init` consumes each explicit input state and prefix into `RowResult.before/after`. Failed prefixes publish the original state and a typed error, not a partial prefix. `DeviceRow` carries the device plan pointer; `RowResult.after` carries the current state. A null plan is unconstrained and permits every vocabulary ID without changing state.

`launch_legal` follows the server termination law: an accepting constrained state offers only EOS, even when its byte DFA still accepts trailing whitespace. Prefix initialization and token advancement retain the generic DFA semantics. It works on one indexed row. Its output buffers are single-row: `ceil(size/32)` mask words, 512 signed IDs, and one uint32 count. The count is uncapped; the first 512 IDs are ascending, and unused IDs are -1. Any row error clears all legal outputs. Zero legal tokens sets `NoLegalTokens`.

`launch_advance` reads one device-selected ID and publishes before/after. Failure leaves the state unchanged. EOS is legal only in an accepting state and sets `ended=1`; ended rows permit EOS repeatedly for fixed-length draft loops and reject other tokens. Empty non-EOS entries reject for constrained rows. `complete` means the current byte state is accepting; `after.ended` separately records EOS.

All three wrappers return CUDA enqueue errors, allocate nothing and never synchronize. Init, legal, selection and advance must be enqueued in order on the same stream (or with explicit event dependencies). Draft positions enqueue legal → selection → advance serially; neither count nor token has to reach the CPU. Runtime errors are sticky until reinitialization. `NonfiniteLogits` is reserved for the engine selector when a nonsingleton legal set has no finite score.

## Standalone contract probe

`bench/gpu_grammar_runtime_probe.cu` compiles a synthetic 19-state byte DFA at the boundary, exercising multiple 16-state chunks, sparse and wide storage, and recompilation after a DFA mutation. It covers prefix success/atomic rejection, accepting-state EOS-only serving masks with legal trailing-whitespace prefixes, a 777-token set exceeding compact capacity, ascending IDs, partial-byte UTF-8 states, EOS and post-EOS continuation, unconstrained empty special tokens, invalid selection/state and a dead-end state. It is independent of the model and engine. Run only when the campaign's GPU timing owner releases the device:

```sh
nvcc -std=c++17 -Isrc -I/workspace/h5-llama/vendor \
  bench/gpu_grammar_runtime_probe.cu -o /tmp/gpu-grammar-runtime-probe
/tmp/gpu-grammar-runtime-probe
```

The token runtime probe passes on the campaign5090. Numerical/model panels are recorded in the [campaign ledger](../results/sparse-head-1010/README.md).
