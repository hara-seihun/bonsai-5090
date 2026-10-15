# Original PrismML15 protocol: selected benchmark reproduction

This directory exports a portable text/code/tool evaluation harness and compact retained observations. It leaves the repository's top-level inference engine unchanged. `source-manifest.json` binds exported source to the original bytes and names adaptations; no Windows controller, SSH, machine mutex, account gateway, credentials, model weights or generated environment is included.

## Results

Canonical machine-readable summary: [`receipts/canonical-scores.json`](receipts/canonical-scores.json).

| Benchmark | Complete Halo result | Published July Bonsai | Same-Bonsai2-weight native screen |
|---|---:|---:|---:|
| GSM8K, native parse-failure-only trigger | **1276/1319 = 96.74%** | 96.06% | 8/8 |
| MATH500, Gemini grading | **494/500 = 98.8%** | 99.20% | 10/10 |
| IFEval, prompt-strict | **500/541 = 92.4%** | 85.03% | 7/8 |
| HumanEval+ (EvalPlus), pass@1 | **154/164 = 93.90%** | 93.90% | full pending |

The four complete results cover 21/25/9/6 disjoint seeded windows respectively. The native screens overlap the complete task sets and are never added to their denominators; full native arms remain unfinished. Halo screens are respectively 8/8, 10/10 and 8/8. These are variable-work quality measurements, not fixed-work speed comparisons.

July's original Ternary Bonsai27B is Qwen3.6-derived; the tested September Bonsai2 is Qwen3.8-derived. Nominal score deltas conflate checkpoints and execution. Same-weight native screens remove the checkpoint difference but matched seeds still do not imply matched stochastic draws.

### GSM8K

The canonical trigger follows whitepaper B.3/Table13: invoke the judge only when the native extractor fails to parse. Retained outputs classify **1276 parsed-correct, 34 parsed-but-wrong, 9 empty parse failures**. Existing Gemini adjudications score every parse failure zero, so the final count remains 1276.

The installed `auto`/`RULE_DEFAULT` route had skipped Gemini. Original rule-only coverage also totals 1276, but that alone is not a replay of the native trigger. [`gsm-native-trigger-result.json`](receipts/gsm-native-trigger-result.json) and the 1319 per-occurrence [`rows`](receipts/gsm-native-trigger-rows.json) preserve that distinction. Historical publisher extractor revision is unknown; this audit used the pinned installed Math-Verify-backed extractor.

**1283/1319 = 97.27% is diagnostic only.** Broad `llm_recall` judges all 43 rule misses, including parsed wrong answers: two semantic/format upgrades and five numeric-gold disagreements. All seven upgraded responses parsed successfully and receive no canonical fallback credit. Original gold answers remain unchanged. [`Diagnostic receipt`](receipts/gsm-broad-recall-diagnostic.json), [`arithmetic checks`](receipts/gsm-gold-disagreement-check.json). Fifty historical judge receipts include seven explicit judge-output parse recoveries; one used low reasoning effort. No new paid calls or generation occurred for this export.

## HumanEval+ — complete

[Full aggregate](receipts/humaneval-plus-full.json): **154/164 = 93.9024390% pass@1**.
Six disjoint windows score **31/32, 28/32, 29/32, 32/32, 30/32, 4/4**; the original32 and continuation132
are counted once, task-weighted. All164 remain: **159 EOS/reasoning-closed, five capped/unclosed**.
Rule/execution scoring,30,000 output cap,92,160 context/two slots, no paid judging.
The full same-weight native arm remains pending; published93.90% concerns the older Bonsai checkpoint.

Canonical programme-12 source **`eb6cd0356ca6176f27741b8948e2f08c4f582de7`** supplies the
[retained result bindings](receipts/retained-result-bindings.json): all164 generation-receipt hashes,
18 prediction/scoring-journal hashes, six status identities, exact partition and source adoption.
[Source provenance](receipts/humaneval-plus-source.json) records the original Git blob/SHA256 and
sanitized export identity. The original aggregate and partition bytes were checked against those
committed bindings. Raw streams remain outside this compact export. No inference or scoring was rerun.

## Offline receipt replay

From the repository root, Python standard library only, no GPU/network/paid services:

```sh
python3 benchmarks/replay.py
```

This derives GSM's count from each retained native extraction/judge decision, and MATH/IFEval/HumanEval+ from retained per-window scores. It checks complete disjoint source-index coverage against exported partitions and agreement with the canonical summary. It also validates partial coverage totals. It does **not** rerun the extractor, Gemini, instruction checks or code execution. The compact export omits the complete raw model streams and judge messages; their hashes/source coordinates remain evidence identities, not claims those bytes are bundled. Root `scripts/gsm8k.py` is a separate rule-only reproduction path.

## Partial observations

- [`Tau2 partial`](receipts/tau2-partial.json): retained full114 reducer snapshot, **66 correct among 82 scored / 114 telecom occurrences**, with 2 errored, 29 missing and 1 untouched. Full score is unset. Its disjoint partition and source-report hash are retained; this is a frozen observation, not live campaign monitoring. Simulator: non-reasoning Qwen3-235B-A22B-Instruct-2507, DeepInfra-only, temperature0; default dual-control, 100 steps, one pass. Historical Prism domain/simulator/revision is unknown; current AA's three repeats differ. The earlier programme12 excerpt's 45/114 is historical.


Tau2 remains a retained partial observation, with no full score. Its future completion run does not alter this frozen snapshot.

## Source identities and protocol

- Programme12 **`08dadac9f891c4ccdad2ed5db853d8001a1c68ac`**, recovered directly from its durable Git mirror: [`commit/tree/blob identity`](receipts/programme12-source.json) and selected, path-sanitized [`Prism15 section`](receipts/programme12-prism15-excerpt.md). No unrelated research history is exported.
- EvalScope **`78b061cb3d1e9cda9d86a001d65f58e2b19e95ab`**, upstream `modelscope/evalscope`. The historical environment installed that exact source archive.
- Bonsai2 GGUF SHA256 **`53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`**. Recorded production DLL SHA256 `a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c`; production plan SHA256 `713c060b627ef9955d68edfef17e12d75f07093c68b74c3fd8ecf4d19b3507bd`. These identify historical quality execution, not necessarily the current top-level engine build.
- Exact GGUF chat template: [`assets/chat_template.jinja`](assets/chat_template.jinja). Qwen non-streaming tool parser source URL and SHA: [`assets/tool-parser-source.json`](assets/tool-parser-source.json).
- Public dataset repository revisions and every selected file SHA256: [`datasets-original15.lock.json`](datasets-original15.lock.json). Publisher revisions remain unknown. Source map paths `campaign://`, `baseline://`, `installed://` are sanitized evidence coordinates, not filesystem paths.

All complete runs: thinking ON, temperature0.7/top-p0.95/top-k20; reasoning removed before scoring. Disclosed reproduction choices: zero-shot, xhigh, seed20261014, min-p0, presence0, repetition1. Quality leaves `H5_DETERMINISTIC_GEMM` unset. IFEval langdetect seed0; publisher language-check seed unknown. Caps: GSM16,384; MATH81,920; IFEval20,480; HumanEval30,000. Historical allocation: GSM20,480ctx/8slots, MATH92,160ctx/2slots, IFEval28,672ctx/8slots. Actual packed formats were Q6 planes/CG8 and fp4-direct KV; some allocation labels printed FP8/Q8.

Original sources: [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf), [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf), [announcement](https://prismml.com/news/bonsai-27b). Exact historical harness/template/dataset/scorer versions and shots were unpublished; other published option shuffles were unseeded. September's different-temperature Bonsai2 protocol is a distinct cohort.

## Portable harness: new evaluations

The exported scope is GSM8K, MATH500, IFEval, HumanEval+, MBPP+, IFBench and Tau2. Dataset availability/CPU qualification is not a quality score. BFCL, LCB, MCQ/AIME and vision operational integrations are deliberately omitted. No script here acquires a GPU, starts an engine or reads host credentials. Supply an owned raw-completion endpoint and, where necessary, an independently budgeted OpenAI-compatible judge/simulator gateway.

```sh
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install -r benchmarks/requirements.txt
# Fetch public data only; exact revisions/hashes, bounded total download.
benchmarks/.venv/bin/python benchmarks/fetch_assets.py \
  --benchmarks gsm8k math_500 ifeval humaneval_plus --max-bytes 67108864
# Optional disjoint MATH windows, preserving seed and source identities:
benchmarks/.venv/bin/python benchmarks/prepare_shards.py --benchmark math_500 --size 20
```

Some upstream files in the historical pin (e.g. prepared HumanEval JSONL) may require the publisher's local layout; a missing file/hash mismatch is an explicit data error, never permission to replace revisions. Provide that exact pinned local file before continuing. No data downloads or environment installation were performed during export.

First terminal: raw engine endpoint must implement `/v1/completions`, explicit sampler/seed/cap and return raw text including the reasoning terminator. Its context/slots must fit the requested output cap.

```sh
# Replace backend URL with your owned endpoint; this explicitly enables inference.
benchmarks/.venv/bin/python benchmarks/harness/template_proxy.py \
  --backend http://127.0.0.1:18330/v1 --backend-model bonsai --profile reproduction \
  --template benchmarks/assets/chat_template.jinja --receipts benchmarks/quality-campaign/receipts/reproduction \
  --seed 20261014 --backend-concurrency 8 --port 18331 --allow-inference
```

Second terminal, one full IFEval run (no paid judge):

```sh
benchmarks/.venv/bin/python benchmarks/harness/campaign.py run \
  --profile reproduction --api-url http://127.0.0.1:18331/v1 \
  --benchmark ifeval --concurrency 8 --inference-authorized
```

For GSM replace `--benchmark ifeval` with `--benchmark gsm8k --judge-trigger published_parse_failure_only --judge-proxy-url http://127.0.0.1:18341/v1`. The judge must serve `gemini-3-flash-preview`. `--rule-only` produces an explicitly preliminary rule score, not the full native-trigger claim. Broad diagnostic scoring requires the separate explicit `broad_rule_miss_diagnostic` trigger. For MATH use `--benchmark math_500 --concurrency 2 --judge-proxy-url http://127.0.0.1:18341/v1`; to run one prepared window also pass `--dataset-snapshot "$PWD/benchmarks/math_500-shards/000" --screen-count 20`. The screen limit is per dataset subset, never a full-score shortcut.

Portable GSM's new-run judge adapter preserves native trigger selection and official prompt/scoring, but does not copy the original owner's budget ledger or staged parse-recovery machinery. It uses one explicit 4096-token judge stage, no automatic recovery. Thus it is a transparent future-run adaptation, not byte-identical historical adjudication. The offline replay above is the retained historical result.

Code tasks require Docker on the caller's context, an immutable local image ID in `benchmarks/quality-campaign/assets/sandbox.lock.json`, and `--sandbox-ready`. [`assets/sandbox.lock.json`](assets/sandbox.lock.json) binds the measured image, base digest and numpy version; no image is bundled. Rebuild on your platform and record its actual ID; never reuse the recorded ID for different bytes. `harness/docker_backend.py` retains network-disabled, no-host-mount execution and durable per-run container ownership. MBPP uses base plus augmented tests; premature `sys.exit(0)` is a failure, not a pass. Tau2 additionally requires fetched pinned data, `--allow-tools` on the proxy, and `--simulator-proxy-url`, `--simulator-model`, `--simulator-domains telecom` on the campaign. No simulator or judge account plumbing is bundled.

Use a fresh profile/output directory for a new run. Completed proxy receipts can replay equal occurrence-indexed prompts with inference disabled, but a partial EvalScope cache must not be mixed with occurrence seeding. Retain uncertain requests and unfinished statuses rather than silently regenerating or reducing the denominator.

## Failures and exclusions

Empty capped finals remain wrong: GSM9, MATH1, IFEval14. Remaining1310/499/527 outputs were EOS/reasoning-closed; complete aggregate receipts report no generation transport or scoring errors. Native IFEval genuinely emitted two italic phrases where three were required. The first MATH92,160ctx/8slot pilot completed no response under memory pressure and was retired: interruption is unknown/unscored, not a model wrong answer. The successful2slot run preserves that failure. MATH019 transport recovery preserved000–018 and generated019–024 only.

[`Code controls before/after`](receipts/code-e2e-fixture-after.json) preserve the discovered base-only MBPP and premature-exit false positives and repaired execution behavior; CPU controls are not model scores. Genuine program errors/timeouts remain wrong/timeout; transport/scorer failures remain unscored infrastructure failures. Tau2 retained two upstream429 effects before any Bonsai request on the first provider, then explicitly changed provider without changing checkpoint. Full-suite affordability/equivalence of cheaper judges is unestablished.

Excluded: full raw generation/token streams, whole sandbox images, operational controller/cleanup process inventory, account routing/budget/credential files, weights, binaries, generated dependencies and unrelated private data. Paths/hardware owner identities were removed where they do not affect reproducibility. Final integration/scan/push belongs to the repository owner. See [`NOTICE`](NOTICE).
