# Two coherent candidates from one DFlash trunk

This is a bounded native discriminator with an opt-in single-active-text HTTP route, not a server default or a trained tree drafter. It compares the existing greedy candidate selector with a second coherent Markov path drawn from the same masked DFlash hidden rows/top16 candidates.

## Proposal and verification

The rank256 selector first reproduces the existing seven-token primary path. It retains the runner-up at the minimum-margin position among the first three visible draft positions (bounded by the verified depth). Subsequent alternate positions are scored with that alternate's own selected predecessor. The hidden trunk and top16 vocabulary candidates are computed once, not once per branch.

The two-slot baseline pays duplicate shared-prefix computation. Two four-row paths consume eight verifier rows; two eight-row paths consume16. Each path includes the same already-known anchor at row0. The primary owns the target argmax of the shared prefix and fork node. An alternate is eligible only when its fork token equals that canonical target exactly, after every preceding shared draft matched exactly. Its own suffix then matches its own causal verifier argmax, token for token. Predictions from duplicated alternate common-prefix rows are never used to obtain extra acceptance. This is one logical tree parent, not best-of-two numerically inconsistent parent outputs, and is not a relaxed verifier.

The selected slot becomes active by swapping its index, with no gratuitous copy-back. Only that slot's accepted captured feature rows are ingested for the next DFlash trunk. `draft_step` validates captured-row bounds, slot and positions before ingestion. The unused slot is overwritten by the next clone.

`h5_clone_slot` differs from prefix-only `h5_copy_slot`: it copies recurrent GDN state, conv ring, both GDN pending parities, conv pending rows, pending count, parity, commit-next count and keep flag, plus accepted-prefix KV and the full drafter ring. It synchronizes completion, and all copy/sync/commit/KV/selector work is inside paid decode wall time.

`H5_CLONE_FULL_KV=1` is the changed lighter route: replace256 per-head KV/scales prefix copies with four contiguous full-slot copies (default FP8 cache), reducing the complete clone from262 copy submissions to10. It copies unused future KVtail bytes as well; those bytes are charged in the report and remain outside causal attention. Source recurrence, pending state/metadata and drafter-ring copies are unchanged. F16 KV changes128 cache copies into two. Bulk clone passed the INT8/F32/f16-KV64-token corruption50 replay guard with19 rounds/eight slot switches. The normal prefix-cache API retains its prefix-only behavior.

## Runnable tests

```sh
make -j2 h5 selector-two-test LLAMA=/root/llama
make two-path-policy-test
./selector-two-test
# Exact arithmetic/state replay control (no DFlash network needed):
H5_WEIGHTS=int8 H5_ACT=int8 H5_STATE=f32 H5_KV=f16 \
  ./h5 pathguard MODEL 'Explain why the sky is blue.' 64 4 50
H5_WEIGHTS=int8 H5_ACT=int8 H5_STATE=f32 H5_KV=f16 \
  ./h5 pathguard MODEL 'Explain why the sky is blue.' 64 8 50
# One shared model/drafter load, eight mixed long-answer prompts; matched1x8,2x4,2x8:
./h5 pathmatrix MODEL DRAFTER bench/two_path_prompts.bin results/two-path/matrix 192
# One geometry only: PATHS, ROWS_PER_PATH, MAXNEW, optional scalar-greedy comparison:
./h5 pathbench MODEL DRAFTER bench/two_path_prompts.bin results/two-path/one.bin 2 4 192 0
```

CPU policy tests ensure an alternate cannot rescue a rejected canonical shared prefix, cannot select the duplicated parent's argmax, and emits common-prefix targets from the canonical parent. The synthetic selector test checks old primary identity, the intended uncertain fork, and predecessor-sensitive alternate continuation. Native corruption tests deliberately switch slots while accepted rows remain pending, exercise both parities and variable commit lengths, and compare the whole emitted stream against scalar greedy. A mismatch returns nonzero.

`pathmatrix` runs all geometries in one process to avoid repeated22-second source-weight repacks. It warms each geometry with32 generated tokens on the first request, then runs baseline1×8, candidate2×4, candidate2×8 and repeated baseline1×8. Warm-up records have explicit run labels and are excluded by `tools/two_path_summary.py`; the before/after baseline controls expose numeric trajectory variation and first-use bias. Prompt contexts, model/drafter, slot allocation, output cap and input requests are held fixed. Cases reset their slots and drafter history through actual prefill ingestion. Long mixed chat/code/math/reasoning/writing requests avoid claiming a win from shortened EOS outputs. Natural EOS remains respected and emitted counts are explicit. Startup/prefill time is separate; paid decode includes cloning, verification, exact host selection, selected feature ingestion and the next draft. No forced draft acceptance is used.

Output `.bin` files use the existing length-prefixed text format; sibling `.tokens` files retain token IDs. `python3 tools/two_path_summary.py LOG --prefix OUT_PREFIX --cap 192` compares complete runs, including output lengths, paid time against both baselines and token-prefix/position agreement. It refuses partial matrices. JSONL reports per-request and total paid time, emitted tokens/round, alternate rescues, clone bytes/time, verify/draft time and mean context. `H5_PATH_TRACE=PATH` records selected slot and ingestion rows per round (its I/O is paid). `H5_TWO_FORK_SPAN=1..7` changes the early uncertainty search, bounded by path depth. `H5_PATH_RAW=1` accepts already-chat-formatted prompt strings; otherwise the runner applies the model's chat template. `H5_LOAD_THREADS` is capped at4 in this branch.

## Numerical controls and status

CPU policy and synthetic GPU selector guards passed. Initial default-F16/FP4 oracle corruption checks exposed existing numerical differences: duplicate shared-prefix argmax can differ with split-K atomics, and scalar one-token state rounding differs from speculative keep/replay. Pinning no-split GEMM removes the duplicate-parent difference but does not make F16 pending rows lossless (initial64-token oracle matched36/64). The final verifier consequently assigns one canonical parent target, rather than exploiting duplicate rounding differences. INT8/F32/f16-KV corruption50 replay controls passed64/64 emitted tokens at both depths: depth4 used30 rounds and8 slot switches; depth8 used19 rounds and8 switches. Logs are in [`results/two-path`](../results/two-path/). The original paid192-token screen completed all24 requests at192 tokens (no EOS shortening): baseline383.857 tok/s and3.293 emitted/round; equal-eight-row budget2×4 gives336.703 tok/s and3.008 emitted/round (-12.3%);16-row2×8 gives390.250 tok/s and3.839 emitted/round (+1.7%). Prefix clone cost about0.87 ms/round. The16-row first request included new R16 tuning, and output divergence was substantial, including some different first tokens, so this is not an adoption result. Text excerpts remain coherent but192-token truncation is insufficient for full task scoring.

The bulk-clone warm192 matrix completed with all32 measured requests capped at192: baseline-before402.298 tok/s (3.300 emitted/round), equal-eight-node candidate371.623 (3.026),16-node candidate430.524 (3.664), baseline-after421.149 (3.426). Bulk clone costs **0.1396 ms/round**, down from about0.871 ms/round: a6.2× complete clone improvement, including full unused KVtail copies and stream synchronization. Equal-eight-node allocation still loses;16nodes is7.0% above the first baseline but only2.2% above the faster repeated control. The unchanged baseline itself shifts4.7% and its generated token streams diverge, including a different initial token on one code request. Equal lengths therefore do not establish equal trajectory or task quality. This remains an opt-in candidate, not a robust production win.

See [warm paired summary](../results/two-path/warm-bulk-summary192.json) and retained per-request JSONL/token dumps. The meaningful eliminated work is252 native copy submissions per clone, not fewer semantic cache bytes. Longer512-token mixed requests and a clean repeat are warranted before adopting the16-node policy. The parent owns attached runs exceeding the worker55-second command ceiling. Vision's scoped shared-head precision repair is incorporated (owning commit846f534) so INT8 target+DFlash no longer requests nonexistent FP4 target-head weights. No production/default-enable or overall task-quality claim yet.

## Combined compact+vision512 discriminator

This branch is based on parent `fb9cf3a` (compact dense5 + focused native vision), with the same native two-path implementation. Run:

```sh
H5_COMPACT=dense5 H5_CLONE_FULL_KV=1 ./h5 pathmatrix MODEL DRAFTER \
  bench/two_path_prompts.bin results/two-path/compact-bulk512 512 0
python3 tools/two_path_summary.py results/two-path/compact-bulk512.jsonl \
  --prefix results/two-path/compact-bulk512 --cap 512
python3 tools/two_path_code_check.py results/two-path/compact-bulk512-{base-before-p1-d8,sixteen-p2-d8,base-after-p1-d8}.bin
```

Warm controls are stable in total paid rate:431.639 /430.254 tok/s. The equal-eight-node route loses at378.441. Two eight-row coherent paths reach468.596 tok/s and3.826 emitted/round (baseline3.435 /3.393), with150 alternate rescues in1066 rounds and charged bulk clone0.1538 ms/round. One candidate sky answer ends naturally at502 generated tokens instead of512; the other seven requests reach512 in every geometry. Excluding that shorter request leaves **3577 timed tokens in each case**, candidate7.037639 s versus controls7.656241 /7.724889 s: **+8.79% /+9.77% paid CLI token rate**. This improvement is not explained by EOS shortening. It is not a matched-content comparison: unchanged baselines themselves share only5–105 initial tokens across the eight512-token answers, and candidate trajectories also differ. The verifier remains canonical exact argmax, but default FP4/split-K/F16 execution is not numerically reproducible.

Retained [paired summary](../results/two-path/compact-bulk-summary512.json), [raw JSONL](../results/two-path/compact-bulk512.jsonl), `.bin` texts and `.tokens` dumps permit reanalysis. [Readable outputs](../results/two-path/compact-bulk-output512.md) are coherent; several requested later examples/tests remain truncated even at512. The completed Python merge functions in both controls and candidate pass400 input pairs without mutation. Their JavaScript frequency counters pass six ordinary cases and all fail the inherited-object-key case (`constructor`, `__proto__`); this limitation is shared, not introduced by the candidate. These [small functional checks](../results/two-path/compact-bulk-code512.json) are not a broad task score.

## Opt-in HTTP single-active text route

`H5_TREE2=1` allocates exactly one private physical scratch slot after the advertised/admitted request slots. It never admits a request into that slot. A single active text request with a valid alternate proposal uses the16-row canonical verifier. Every image and multi-stream round keeps the ordinary scheduler. A transition back to one stream may spend an ordinary round generating fresh alternate proposals before using the tree. No physical-slot remapping or vision-path change is required.

On a rescue, accepted captured features are ingested from scratch into that scratch slot's DFlash ring, and its next two proposals are selected **before** cloning its complete pending causal state, KV and updated drafter ring back to the original request slot. Primary selection needs no copy-back. Clone-before and clone-back are both paid in scheduler `busy_s` and HTTP wall time; selectors are paid in `draft_s`. `/metrics` and per-response native metrics report tree rounds, rescues, fallback rounds and clone time separately. `H5_CLONE_FULL_KV=1` remains explicit; HTTP's larger configured context copies more unused KVtail bytes than the small CLI context and must be priced by the HTTP measurement, not inferred from the CLI win.

```sh
H5_TREE2=1 H5_COMPACT=dense5 H5_CLONE_FULL_KV=1 ./h5-server MODEL DRAFTER --slots 1 --ctx 4096 --port 8010
H5_PATH_CLONE_BACK=1 H5_CLONE_FULL_KV=1 H5_WEIGHTS=int8 H5_ACT=int8 H5_STATE=f32 H5_KV=f16 \
  ./h5 pathguard MODEL 'Explain why the sky is blue.' 64 8 50
```

The exact-arithmetic clone-back corruption guard passes64/64 tokens,19 rounds and eight rescues. Tree mode is off by default, and request slot count is unchanged. Tree CLI modes, HTTP tree startup and the native two-proposal API reject positive `H5_DRAFT_MARGIN`: the inherited relaxed-argmax option is incompatible with this exact policy. CLI/HTTP rejection was tested with CUDA hidden and deliberately missing model files, before loading a model or initializing CUDA.

### Paid HTTP and exploratory quality results

Parent measured the actual same server binary with compact dense5 and configured4096-token cache allocation, tree off/on, sequential mixed8 prompts ×2 repeats. Baseline **365.17 E2E /367.65 decode tok/s**, tree **398.72 /401.78** (+9.2% /9.3%). The measured tree has1879 tree rounds,260 rescues and about0.347 ms charged clone time per round, including rescue copy-back and the larger cache tail. Output totals differ (6350 versus6271 tokens), so this is a rate comparison, not matched-content latency. Excluding all varying-length outputs leaves eight paired512-token requests (4096 emitted tokens each): **369.22→399.25 E2E** and **371.03→401.41 decode**, +8.13% /+8.19%. The lead survives removing EOS-length differences. [Capped subset](../results/two-path/tree-http-capped-summary.json), full raw summaries/requests/logs in [`tree-http-base`](../results/two-path/tree-http-base/) and [`tree-http-two`](../results/two-path/tree-http-two/).

The HTTP outputs remain coherent. Completed prime and SQL code look sound; one candidate haiku sequence ends after one poem while the baseline and second candidate repeat give several. Wording, lengths and answers are not identical. Exploratory **reused GSM128** sample scores **121/128 in both modes**: baseline670.31 E2E /694.70 decode, tree677.96 /702.04 (about+1.1%, consistent with less headroom on already-high-acceptance math). There are two correctness gains and two losses, plus two changed wrong answers; identical aggregate accuracy is not itemwise equivalence. This is not a fresh held-out score or a general quality-equivalence guarantee. Full prompts, frozen gold selection, generated requests, per-item scores and summaries are retained in [`tree-gsm-base`](../results/two-path/tree-gsm-base/), [`tree-gsm-two`](../results/two-path/tree-gsm-two/) and [paired summary](../results/two-path/tree-gsm-paired-summary.json).

```sh
python3 tools/score_single_gsm.py results/two-path/tree-gsm-selected128.json \
  results/two-path/tree-gsm-two/requests.jsonl --output /tmp/tree-score.json
H5_COMPACT=dense5 H5_CLONE_FULL_KV=1 python3 tools/tree_http_guard.py \
  --server ./h5-server --model MODEL --drafter DRAFTER --mmproj MMPROJ \
  --output results/two-path/tree-http-guard
```

The lifecycle guard passed actual two-active request overlap,68 ordinary fallback rounds,28 rescues, streamed cancellation, subsequent request-slot reuse, and both ordinary/speculative image requests with tree rounds0. A later text request reenters the tree route. All request slots return idle; the private scratch slot is never admitted. [Guard results](../results/two-path/tree-http-guard/summary.json) and raw responses/server log are retained. These are lifecycle/counters/color guards, not scalar-greedy identity. No more unchanged GPU reruns are needed for this bounded route. Keep the tree opt-in; broader prompt quality and concurrency performance are future work, not claims of this result.
