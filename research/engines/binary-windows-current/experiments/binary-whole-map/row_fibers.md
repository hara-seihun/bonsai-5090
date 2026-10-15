# Actual originalA8 FFN input row fibres — 2026-10-10

## Finding

Exact packet repetition exists in real mixed prompt batches, but every observed repeated packet had the **same complete token prefix**. There were no collisions joining different token prefixes, no antipodal pairs, and no repeated packets in Wiki chunk-prefill. This supports common-prefix reuse, not an observed additional whole-FFN activation quotient after prompts diverge.

All counts below apply separately and identically at layers **0, 30, 61**; they are not pooled across different weight maps.

| Regime | Native rows/call | Observed rows/layer | Distinct complete packets | Duplicate rows | Row-count compression |
|---|---:|---:|---:|---:|---:|
| Wiki first128 tokens, contiguous chunk-prefill |16|128|128|0|1.000×|
| Wiki first128 tokens, contiguous chunk-prefill |32|128|128|0|1.000×|
| Mixed actual panel prompts, first53 aligned token steps |16|848|665|183|1.275188×|
| Mixed actual panel prompts, first53 aligned token steps |32|1696|1301|395|1.303613×|
| Identical-prompt positive control, first64 steps |16|1024|64|960|16×|
| Identical-prompt positive control, first64 steps |32|2048|64|1984|32×|

Within-call compression equals the aggregate ratio: no extra reuse appears between different calls. Mixed R16 is 6 GSM8K + 6 HE+ + 4 chat prompts; R32 adds another disjoint set of 6+6+4. No prompt is duplicated in the mixed panel. All selected layers have 16 distinct rows from zero-based token position31 onward at R16, and 32 from position33 onward at R32. The initial three shared chat-template tokens collapse to one row; HE instruction/prompt scaffolding accounts for most subsequent repetition. The actual native prefix hashes and per-call counts are exported, rather than attributing collisions by inspection alone.

## Exact observation contract

Control: immutable `/workspace/binary/digits-ragged/halo`, source374c77d; original Q1 signs and original K128 FP16 model weight scales. Model `/workspace/binary/models/Bonsai-27B-Q1_0.gguf` has owning campaign SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`. No model weights, GU/down kernels, main source, or ordinary bench files were edited.

Host GNU `--wrap` diagnostics intercept the **native** `bitserial::produce(D)` after preparation and capture the last such packet before the real FFN SiLU preparation. That is the actual model-produced postnorm packet feeding joined gate/up, not a synthetic/recomputed quantization. The hook observes real `h5_step` calls at R16/R32. It retains q byte patterns, float32 sx bit patterns, DC, and uint32 typed status. DC sums were independently checked against q; all captured statuses were Valid. Complete-packet classification uses byte-key comparison, so hash collisions cannot manufacture equality. q-only and q+sx distinct counts happen to match complete counts here; this is an observation, not license to drop scales/status from a key.

Each stored packet is **5600B**: 5120 signedA8 q bytes +40×4B sx +40×4B DC +40×4B status. The minimally sufficient captured q/sx/status key is5440B; DC is deterministic given q but included in raw evidence. Antipode means q→−q, DC→−DC, with identical sx/status. No observed antipodal pair exists. Moreover SiLU×up is not an odd map, so an antipodal packet would not automatically supply a cheap whole-FFN output reuse law.

Environment explicitly selects packed1, pureA8, F16 state/KV, no graph, no PDL, no GU fusion, no whole-FFN selected path, and `fp4-digits-ragged`. This is engine374's real native R16/R32 path, **not** an R1 F32 substitution or a measurement of source952's fused FFN. Its native row-size, split, quantization and scale/reduction policy are part of the endpoint.

Wiki source is the campaign's existing `wiki.test.raw`. GSM/HE messages and their recorded target continuations come from existing `tasks-reference-b32/*/responses.jsonl`; chat uses existing `bench/two_path_prompts.json`. Native `chat.h` thinking-off template is reproduced exactly. Prompt continuations are included in fixture custody, though the mixed census stops at the shortest prompt's53 token steps. These are teacher-forced native observations, not an HTTP serving workload, generation-speed test, or task-quality result.

## Recurring price and minimum realization

The packet census closes this **input-row identity/antipode** coordinate on the observed diverged prefixes, not every possible FFN output fibre or broader carrier construction.

For the observed common-prefix hits, the cheapest plausible route is **token-prefix coalescing before model execution**, with each request's recurrent/KV state and native prefix ownership handled by the existing state-copy/prefix machinery. Prefix identity can be known before paying a5440B activation key at every FFN. No cache was integrated or benchmarked here.

If nevertheless realizing an FFN-local row quotient, the recurring path must include:

- GPU hash scan: at least5440R bytes of q/sx/status input, fingerprint writes, grouping and row-index map; hash matches require full packet comparisons (up to another10880B of reads per compared pair).
- Exact comparison and miss ownership: all unmatched rows execute the unchanged native map; table coverage is never assumed. Cross-call keys bind model/layer, native arithmetic/split policy and the full packet. Persistent cache storage/insertion/eviction adds cost absent from an intra-call quotient.
- If gathering U representative packets:5600U read +5600U write bytes, including native DC/status. Indirect native operand loads might avoid this physical gather, but they still incur indexing and noncontiguous loads.
- Representative native GU→SiLU×A8→down work under **the original R-specific native arithmetic/split policy**, not selecting a different lower-row/F32 endpoint merely because U is small. Padding/masking and tensor-tile occupancy may prevent row-count compression from becoming proportional work compression.
- A5120-F32 branch output is20480B per row: representative write20480U, broadcast read20480R and final write20480R. A residual epilogue also reads the original20480R residual bytes. These are a traffic budget for a materialized branch-output realization, not a measured lower bound for every fusion.
- Packet equality applies to the quantized **FFN branch input**, not necessarily the original residual x. Cache the pre-residual branch result and retain every row's native residual epilogue; caching the entire residual-added row from q/sx/status alone is not justified. Changing reduction/epilogue association must be priced and checked at its intended quality contract.

At R16/R32 mixed53-step scope only21.58%/23.29% of rows are duplicate, all due to known prefixes, while the real Wiki scope saves zero rows and would pay every lookup. The16×/32× identical-prompt positive controls establish that the diagnostic sees true exact repetition; they are not model speed claims.

## Custody and reproduction

Sources (disjoint `row_fibers*` scope, integrated in the private experiment branch):

- `row_fibers.cpp`: native hook and finite harness.
- `row_fibers_prepare.py`: exact existing prompt fixture selection and source hashes.
- `row_fibers_build.sh`: cached-object build and SHA256 source/object/executable pins.
- `row_fibers_analyze.py`: complete equality/antipode/prefix census.
- `row_fibers_results/report.json`, `fingerprints.jsonl`, `sourcepins.txt`, `prompts.json`, `prompts/`, `packets.bin.tokens`: exported counts, per-row fingerprints, exact token prefixes, prompts, pins.
- `row_fibers_results/samples.bin` + `samples.json`: small raw packet samples with offsets and hashes.
- `row_fibers_results/export.tgz`: small reproduction/export bundle. Full98,670,264B packets (17,616 rows; SHA256 `204e785caecae308fdbc6ba6690a0df81d953311133564616d1c6aff0c7e3df2`) were exported before teardown and are losslessly retained at `~/data/binary-5090/whole-map/row-fibers/packets.bin.zst` (35,538,403B compressed). Restored byte count/hash were checked before removing the uncompressed duplicate. `engine-results/data-custody.json` binds compressed and restored custody; `zstd -dc .../packets.bin.zst` recovers the exact dataset.

The final GPU run completed exit0 under `flock /tmp/binary-gpu.lock` and an after-lock55s timeout, owned by `kjob@binary-row-fibers-final-ddec2b21`. Initial short-chat fixture rejection occurred before engine creation; corrected finite53-step scope. An intermediate mixed fixture repeated two chat prompts and was replaced by the disjoint12+12+8 final selection. An analysis started before final job completion rejected incomplete packet bytes; rerun after the exit0 receipt produced the final exported census. No incomplete/earlier census is used for the table above.

No commit, public push, new rental, kernel integration, or wake was created. Parent owns shared-writer heartbeat, final private commit/custody, and rental destruction.
