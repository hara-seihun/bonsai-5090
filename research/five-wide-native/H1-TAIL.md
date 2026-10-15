# H1 terminal carrier

Sibling of exact8B/feature v3 at `d7926585e`, not a serving change.

## Map and matched control

Selected frozen producer: configuration13, FW8/WG2/TOK32/PIPE0, split1,288threads, two32-row grids. Keep the full-K MMA order and row-mean scaling. For adjacent even/odd features, compute the same rounded SiLU(gate)*up, then explicit sign multiplication, then the original first butterfly `(even+odd, even−odd)`. Producer lane XOR4 flips feature bit0 while preserving row because `gid=lane>>2`, `tq=lane&3`. Encode those two resulting F32 bit patterns jointly in one8-byte word. Consumer reads each pair once, optionally inverts the joint label, starts the remaining source Hadamard loop at len2, and keeps the1024-feature normalization, fixed global scale, block scale rounding and actual q4/s4/g4 layout.

`h1-plain` executes the same H1 producer, geometry, storage and consumer with all mixed-coordinate relabelings disabled. `h1-mixed` includes relabelings in the full contraction and at the terminal pair. This measures the whole mixed route, not just terminal-XOR cost. Parent `tail-plain`/`tail-mixed` controls remain available in this sibling to distinguish H1 from numerical-profile changes; preserved v3 source/DLL are unchanged.

H1 retains8B/two features:557,056 pairs /4,456,448 bytes per64-row layer. Write+read is8,912,896 bytes, half v3's17,825,792, and equal to the ordinary4B/feature SiLU temporary's traffic. Across64 layers and128 generated tokens, nominal carrier write+read falls146,028,888,064→73,014,444,032 bytes per batch. Sign and first-butterfly work move between kernels; they are not eliminated. Allocated carrier halves8,912,896→4,456,448 bytes. No measured speedup follows from those facts.

## CPU evidence and compiler invoice

`bench/fullk_h1_contract.cpp` exhausts1,114,112 feature writes and557,056 pair writes, exactlyonce, verifies XOR4 row/feature ownership and complete consumer pair coverage, and checks10 schedule parser cases. `make fullk-h1-contract` runs it. `H1.lean` compiles paired-bit inverse/terminal composition and lane-XOR4 identities. The sign/SiLU/add/sub functions in its continuation theorem are parameters: it does not prove CUDA floating semantics or FP4 conformance.

CPU-only native build succeeds17.3726s, DLL `398a9d2bb38ec8613af6663dd782a282727da04b5f255f43e4c10780f1c6d768`. Producer registers:131plain/142mixed, zero spills; consumer40plain/47mixed, zero spills. Ptxas reports4096B consumer shared storage; later linked-ELF cuobjdump reports5120B. Both observations are retained; no occupancy claim depends on resolving their accounting. Producer counts remain matched to actual cfg13 v3. These are compiler observations, not GPU timing. All six compiled source hashes must match `h1-build-receipt-v1.json`.

## Numerical-profile composition and repairs

- Strict reference and candidates explicitly set `H5_DETERMINISTIC_GEMM=1`, `H5_DOWN_CLEAR=0`, use the same immutable plan/config13/split1, and clear ambient experimental/QC variables. Ordinary throughput explicitly uses deterministic0. Baselines are6777.352952 ordinary and6583.070035 deterministic, kept separate.
- Ordinary non-GU projections already use ordered split-major partials/fold in deterministic mode; H1 GU is full-K split1, so there is no unordered GU fold to replace. Compiler contraction/source arithmetic remains a separate conformance obligation.
- Repaired the candidate owner's `gemm_clear` deterministic bypass: preserve dense5/FP4 and pointer/count/row contracts, reject flag waits, run ordered projection/add, then stream-ordered clear-range memsets. Original active down-clear could bypass the deterministic route on low rows. Strict fixtures pin DOWN_CLEAR0 so comparisons do not depend on this unmeasured repair.
- Fixed GU schedule validation to permit explicit0/off while rejecting malformed or active tail/ordered/flag schedules.
- Added `fullk_tail.cu`/object/dependencies to ordinary Makefile ownership. Clean link previously omitted directly referenced tail symbols; native sidecar builds separately included them. Dry-run confirms the object is compiled and linked; a full normal Makefile build was not run.

## GPU obligations and harness generations

CPU-generated host scripts and receipts live at `~/work/five-wide-native/` under `h1-*` / `run-h1-v1-*`; `h1-test-plan-v1.json` binds source/DLL, profiles and limits. Actual GPU execution requires explicit PrismML between-shard handback plus atomic mutex acquisition. Never infer slot authority from a free mutex.

Strict matrix first creates a fresh frozen-serving reference and independent repeat, then parent-tail and H1 plain/mixed comparisons on the same teacher-forced inputs. Graph-v2 warms three64-row decode steps (two eager visits then capture), then two cached-replay64-row steps observe all64×248320 logits, require finite bits, exact inputs/tokens/logit bytes and positive native decode graph hits. Use normal `qc_step` then `qc_logits`: inline step-logits disables graph execution. Candidate checks also require positive counts and zero first-butterfly/SiLU mismatches and zero final q4/s4/g4 mismatches. Reference repeat must pass before candidate interpretation.

Separate3-repeat B64×128 whole-model timing jobs include producer, carrier conversion, consumer and all other model costs; both ordinary and deterministic profiles have named plain/mixed receipts. No producer-only timing can nominate this map. Global4-minute budget and50s native subprocess limits retain partial completed evidence for a later explicitly offered gap. No serving binaries/plans, rentals or services are modified.

### Graph lifecycle repair

Windowv1 failed the frozen-serving reference prerequisite at nativeexit7: two observed steps produced finite logits on all64rows, with graph hits/captures0. No H1 candidate or TPS ran. Native subprocess exited; runner released the mutex; driver independently acquired/released it with wait0 after12.47s. The driver window, failure receipt, stdout/stderr and read-only native source extracts are retained in `graph-v1-failure/`. Native original logits remain under `fullk-h1-tail-v1/strict/serving-reference/logits.f32`, hash `a2e2fc1e8e2bf4868ea8040222371d8b9dc637da66776daa74d946c5783bc666`. No H1 numerical rejection follows.

Installed frozen-serving source `e2ae33c4` and H1 source `2c1cef9e` both use `gseen[k]++ < 2`: sightings one and two are eager, sighting three captures, later visits replay. The qc wrapper calls ordinary h5_step and reads logits afterward; it did not disable graphs. The v1 fixture ended before its first possible capture.

`graph-v2/` is a deliberately named CPU-only successor, reusing unchanged DLL398a9d2b. `make_h1_graph_v2.py` derives it from hash-bound v1 scripts. Three teacher-forced warmup steps precede the two observed steps; warmup inputs, output tokens and graph counters are retained and exactly compared. Every observed step must be cached graph replay: hits2/captures0, independent of preexisting counters. Finite full logits, reference-repeat stability and complete intermediate/final endpoint counters remain mandatory. CPU contract tests exercise the old two-visit refusal, third-visit capture, warmed replay, teacher forcing and incorrect counter refusals; they do not establish native acceptance. Ordinary/deterministic timing profiles remain separate. `graph-v2/driver.py` owns the one240s window and checks strict prerequisites before timing; each native subprocess is bounded50s or less remaining time. No GPU probe or v1 replay was made during repair.

### Isolated Python packaging repair

Graph-v2's reference child exited1 in0.146s before any native DLL/model loading: the RTX 5090 host's embedded Python runs isolated and cannot import a neighbouring helper merely because that file was staged. The original driver window/failure is retained in `graph-v2-failure/`; final mutex wait0, ReleaseMutex1 and CloseHandle1, with no candidate/TPS. Originalv1/v2 source and effects remain unchanged.

Named `graph-v3/` embeds the shared pure phase contract into the generated strict worker instead of requiring runtime helper discovery. `make_h1_graph_v3.py` owns all six consumers. Same DLL398a9d2b,3warmups→2cached observations, deterministic1/DOWN_CLEAR0, full64logit rows, warmup/token equality, complete endpoint counters and separate ordinary/deterministic TPS. A CPU-only test executed the actual generated worker's imports and contract functions under the RTX 5090 host's actual isolated interpreter, reproduced the lifecycle refusal/acceptance and teacher-forcing cases, parsed all ten runners, and confirmed six embedded workers byte-identical. It loaded no native DLL and created no GPU window. `graph-v3/cpu-stage-receipt.json` binds this test, all six compiled source hashes, unchanged DLL/bench and staged script bytes. Native acceptance and throughput were subsequently measured in the graph-v3 settlement below.

The v3 driver preserves typed failure receipts and also exits7 after a failed/setup prefix. Full-window240s/native50s caps are unchanged; only an explicit fresh PrismML handback permits its GPU invocation.


## Native graph-v3 settlement

Source `95fce9015b13e65954793c415c12153e8fa8bc0f`, driver `95491795dccfc766e68a6b1a3e04921412e695ccfdfa4f64eff6d221eac40292`, unchanged candidate DLL398a9d2b. One fresh authorized window:1792012005.804475→1792012188.6962101,182.891735s of240s. All ten native runner processes exited0, each bounded50s. Original durable Bash `00000000-0000-0000-0000-000000000000` completed0. Final independent mutex wait0, ReleaseMutex1/CloseHandle1; serving DLL and plan hashes unchanged. Explicit actual release sent to8614 under operation`00000000-0000-0000-0000-000000000000`; result sent to9769 under`00000000-0000-0000-0000-000000000000`. No further GPU window queued.

Six strict runners pass: serving reference/repeat have identical full logits, inputs, tokens and warmup inputs/tokens. Every runner observes two cached graph replays (hits2/captures0), following three warmups (hits0/captures1 separately retained). H1 and parent-tail plain/mixed each compare356,515,840 intermediate F32 words,178,257,920q4 bytes,22,282,240s4 bytes and20,480g4 words, all0 mismatches. H1's intermediate is the source-order first Hadamard butterfly; parent tail's is SiLU. Both observed64×248320-logit arrays are finite and byte-exact against the frozen deterministic reference; teacher-forced warmup inputs/output tokens match. Shared full-logit SHA256`47b9805a60b141d7d1ada84fe1bf5645605b72de32e4a96a01d5ac2751a9a777`. This is measured native/model conformance on this corpus; Lean's scope remains the conditional label/continuation theorem.

| Profile | Arm | Three-repeat median tok/s | Matched baseline | Throughput loss |
|---|---|---:|---:|---:|
| Ordinary | H1 plain |4022.795777|6777.352952|40.644%|
| Ordinary | H1 mixed |4001.247069|6777.352952|40.962%|
| Deterministic | H1 plain |3947.455855|6583.070035|40.036%|
| Deterministic | H1 mixed |3931.397121|6583.070035|40.280%|

Whole-model B64×128,8192generated tokens/pass,8warmup tokens/stream. Ordinary and deterministic profiles remain separate; strict numerical acceptance is for deterministic1/DOWN_CLEAR0. Halving the carrier traffic does not remove the~40% whole-model loss; these timings alone do not isolate which component dominates. **No adoption**, no serving mutation.

[Source-bound complete result](h1-window-v3-result.json) embeds all10 native receipts and timing samples. The unset-versus0 DOWN_CLEAR seam is closed from exact frozen enginee2ae33c4 lines2584–2586: both selectfalse; the old controls and H1 are matched on that behavior. [Original driver log](h1-window-v3-driver.log), SHA256`57c131e55c0f803a0f4011d95a5603fb421f142177a2dfdc0f82706b09290aa7`, preserves complete stdout and final receipt. All saved native arrays, logs and build files are also retained in [immutable readable custody](h1-custody-v1/README.md): six identical v3 logit files resolve to one SHA object47b9805a; the distinct v1 failure logits a2e2fc1e remain separate. V1/V2 failures, private source95fce901 and bundle47b2d027 remain unchanged; all source writers are released.

### Immutable CPU custody and next cost

[Complete package](h1-custody-v1.tar.zst):447,397,500B, SHA256`edcd9b3a226cc253b6de90d284c42c3456f4f381467e2e7341217d354bf94acc`; [manifest](h1-custody-v1/MANIFEST.json) SHA256`505635b9fa575285b3ea00510477de4b6e4c61504fcac3a24b9731978607d006`.2366 readable logical paths map to2094 objects,1,382,499,471→620,300,218 bytes. Full source snapshot95fce901, prior source bundles/failures, build objects/identities, conditional proof sources, all saved native outputs and seven exact SASS dumps are retained; [package receipt](h1-custody-v1-receipt.json) records boundaries. Previous custody is untouched.

Conformance is corpus-scoped: three teacher-forced warmups compare inputs/output tokens; two observed cached steps compare full finite logits; endpoint counters cover the five-decode run. Raw intermediate GU/first-butterfly/q4/s4/g4 tensors and full warmup logits were never serialized, so the terminated run supplies counters rather than recoverable dumps. Every saved file was recovered; no saved-output gap. Original Lean stdout is blank and retained; the earlier successful compile is documented without a separate persisted compiler-version/exit receipt. GGUF and shared runtime dependencies remain separately owned inputs; the package does not provision a fresh host.

[Source/SASS-bound diagnosis](H1-COST.md), [opcode/resource invoice](h1-cost-invoice.json): H1 plain's main40-block K loop contains128 static SHFL.IDX sites; frozen-serving cfg13/WF0 has no SHFL and the same32 static OMMA sites. This supports selecting repeated cooperative dense5 operand reconstruction/cross-lane synchronization as the cost a new map must remove. Its dynamic latency share is an unmeasured hypothesis. Consumer-traffic reduction left the~40% model deficit; another such refinement is not selected. No new GPU window/probe, serving mutation or experiment replay was used for custody or diagnosis.
