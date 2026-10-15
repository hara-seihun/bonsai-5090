# Compiled-token grammar cache

`gpu_grammar_token_compile.cuh` converts a validated byte DFA and the engine vocabulary into a token automaton at upload. `gpu_grammar_runtime.cuh` never walks token bytes: prefixes and advancement use binary search in sparse next-state lists or direct indexing in wide maps. Legality copies stored counts, first512 IDs and bitsets in one CTA. Accepting states offer the vocabulary's cached EOS-only list; raw token maps still support whitespace prefixes.

## Footprint and custody

An engine owns a fixed 2 GiB cache (`TOKEN_CACHE_BUDGET`). Each entry retains its exact source Plan as an equality key plus its token device buffers. All requests using identical transitions, accepting bits and initial state share that entry. Active C ABI handles remain distinct and hold reference counts. Releasing a handle removes its reference, not its cached token data. The next equivalent upload hits the cache, including repetitions after all earlier requests finished.

Inactive entries are evicted least-recently-used. Active entries and pending target descriptors cannot be freed. A plan larger than the budget, or a cache pinned by active handles, returns -10 (`TokenCacheError::Budget`); representational/grid overflow returns -11 (`TokenCacheError::Overflow`). Neither condition selects a byte-walking runtime. Vocabulary replacement requires no active handles and destroys GPU grammar graphs before clearing cached token data.

For V vocabulary tokens, S byte states, E sparse edges and W wide states, device payload is `9*S + 8*E + W*(4*V + 4*ceil(V/32) + 2048) + sizeof(DevicePlan)`. Resident accounting adds the retained host source Plan bytes. The shared vocabulary is accounted separately from the token cache. Compilation temporarily uploads the byte transitions and allocates at most `16*V*4` bytes for token next-state chunks (15.16 MiB at V=248320), not an S-by-V temporary. Temporary byte transitions and chunk storage are freed before upload returns. Dense persistent maps are allocated only for raw states with more than512 legal tokens.

The boundary first enqueues parallel token counts, reads those counts once to size the hybrid buffers, then enqueues chunk compilation/packing and waits for the ready descriptor. GPU byte walking and host waits occur only here, never in draft positions or target matching. `CacheStats.compile_s` records cumulative completed cold-upload wall time, including final synchronization and temporary cleanup. Hits leave it unchanged. `token_plan_bytes`, `wide_states` and `sparse_edges` describe the handle's payload; `resident_bytes`, `hits`, `misses` and `evictions` describe the engine cache. The getter reads host counters only.

## Probes

- `bench/gpu_grammar_runtime_probe.cu`: synthetic boundary compilation, exact512/513 split, ascending IDs/bitsets, sparse/wide token advance, prefixes, EOS and typed errors.
- `bench/gpu_grammar_cache_probe.cpp`: actual layout and victim-selection functions, footprint arithmetic, overflow, LRU choice and active-reference fencing. Build/run: `g++ -O2 -std=c++17 -Isrc bench/gpu_grammar_cache_probe.cpp -o /tmp/cache-probe && /tmp/cache-probe`.
- `bench/gpu_grammar_cache_lifecycle.cpp`: public C APIs with a fresh engine and synthetic vocabulary. Tests equivalent distinct handles sharing data, pending release fencing, releasing one handle while decoding through another, retained data after all handles release, cache hit without extra compilation, and vocabulary lifetime boundaries. Link the engine objects and define `GPU_GRAMMAR_CACHE_PROBE_MAIN`, then pass the model path. It replaces the engine vocabulary: run on a fresh probe engine, not a serving engine.

CPU layout/LRU contracts pass. On the campaign5090, runtime token compilation/lookup and the public cached-handle/prefix/full-model singleton decode/lifetime probe pass. Full-model quality/timing and cold-versus-warm prices are recorded in the [campaign ledger](../results/sparse-head-1010/README.md).
