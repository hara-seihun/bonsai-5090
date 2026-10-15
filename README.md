# Bonsai inference: packed weights, native kernels, measured results

Native inference engines for PrismML's Bonsai models, plus the benchmark harness, measurements,
and packed-computation experiments behind them. The primary runnable target here is
**Ternary Bonsai 2 27B on one RTX 5090**. The repository also preserves the original CUDA
speculative server, AMD HIP engine, related MoE engine, binary-model work and five-wide kernel research.

Start with the root build for the current batch-decode engine. [Other engines](engines/README.md),
[packed/binary research](research/README.md) and [quality benchmarks](benchmarks/README.md) have
separate source identities and reproduction instructions. Their experimental changes are **not all
combined into the root engine**.

## What we changed, and why

A ternary weight has three possible values: negative, zero or positive, multiplied by a scale.
Storing those choices in floating-point wastes bandwidth. Expanding them into a large floating-point
matrix before every multiplication wastes it again. We keep the weights packed and arrange their
bits so the GPU can turn them directly into the operands its instructions consume.

The model has both recurrent layers and conventional attention. Making its matrix multiplications
faster exposes the cost of recurrent-state traffic, the KV cache, launch overhead and CPU round trips.
The engine therefore optimizes the complete decode path:

- **Packed weight readers:** two-bit trits for most matrices; five trits per byte for the vocabulary
  head. Weight conversion is done at load time; no full floating-point weight expansion is needed.
- **Blackwell FP4 Tensor Cores:** prepare activations directly in the instruction's layout; fuse
  normalization, sign changes, Hadamard rotation and quantization. Fuse gate/up projections and
  nonlinear/residual output work where appropriate.
- **Less state traffic:** cooperative recurrent updates and six-bit recurrent-state storage, with
  FP32 update arithmetic. A direct FP4 KV reader avoids staging a half-precision cache tile.
- **Less orchestration:** GPU token selection, resident embeddings, CUDA graphs, asynchronous weight
  copies and programmatic dependent launch.

These are engineering approximations as well as layout changes. FP4 activations, six-bit recurrent
state, FP4 KV and mean weight scales are lossy relative to the reference. Quality is measured
separately. Deterministic reductions make repeated executions stable; they do not make this engine
numerically identical to PrismML's reference.

The research takes the same idea further: treat packed bits as a coordinate system to compute in.
Search for an instruction composition that preserves the final answer, including constructions whose
intermediate labels never resemble ordinary numbers. A proof of such a construction, a native kernel
check and a whole-engine speedup are separate results. The export includes unsuccessful consumers
and unintegrated candidates alongside the wins.

## Decode performance

Every figure below links to retained evidence. Rates count generated tokens across all streams during
**decode only**: load, conversion, tuning, tokenization and prefill are excluded. These runs are greedy,
without speculation. They are not HTTP service throughput or single-user token rates.

| Measurement | Result | Receipt / conditions |
|---|---:|---|
| PrismML llama.cpp, original baseline | 930.6 tok/s | [Ladder](receipts/ladder.json): Linux RTX 5090, batch 64, 1,024 timed tokens per stream |
| Fused FP4 engine | 4,579.7 tok/s | [Same ladder](receipts/ladder.json), same device/window; the FP4 bundle was measured together |
| + six-bit recurrence | 6,118.4 tok/s | [Same ladder](receipts/ladder.json) |
| + direct FP4 KV | **6,446.4 tok/s; 6.93× baseline** | [Same ladder](receipts/ladder.json), matched device/window |
| Original final engine measurement | **6,777.4 tok/s** | [Receipt](receipts/serving-b64x128.json): another RTX 5090, Windows, batch 64 × 128 timed tokens |
| Original deterministic measurement | **6,583.1 tok/s** | [Receipt](receipts/deterministic-b64x128.json), fixed-order split-K reduction |
| Clean checkout, ordinary | **6,731.7 tok/s** | [Receipt](receipts/clean-checkout/serving-b64x128.json) |
| Clean checkout, deterministic | **6,550.5 / 6,549.7 tok/s** | [Receipt](receipts/clean-checkout/deterministic-b64x128.json), repeated tokens identical |

**6.93× is the same-machine speedup.** Dividing the later Windows result by the Linux baseline
would mix devices and timing windows. The clean-build gap to the original final measurement remains
unexplained. [The full ladder](docs/ladder.md) explains each change, its measured boundary and the
changes that have no standalone ablation. [Clean-build status](docs/reproduction.md) records the
successful Windows run and the incomplete Linux build/run qualification.

### Single stream, PP512 / TG128

PrismML's community table uses llama-bench PP512/TG128, one stream, no speculation. We measured the same
workload definitions with this engine's own harness, on the same `PTQ1_0` model file as the published RTX 5090 row:
**PP512 8,986.8 ± 88.1 tok/s (2.18× the published 4,116.6)** and **TG128 135.8 ± 2.3 tok/s**. That is
1.02× the published PTQ1_0 row (133.0) and below the published PQ2_0 row (143.0). The figures are means
of five quiescent repeats. Execution precision, KV format, microbatching, OS and driver differ from the
published llama.cpp run. [Method, raw samples and the separate compile-overlapped repeat](benchmarks/pp512-tg128/).

## Quality against PrismML's published results

The model under test is **Ternary Bonsai 2 27B** (GGUF sha256 `53107f53…33ee3`) on our engine, one RTX 5090.
PrismML publishes two relevant score sets:

- **Bonsai 2 (same checkpoint):** [Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf),
  Table 10, xhigh thinking, and the [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
  Protocol: temperature **1.0**, top-p 0.95, top-k 20, min-p 0; output caps 81,920 (MATH500, AIME, GPQA,
  LiveCodeBench, IFBench), 49,152 (HumanEval+) and 32,768 (GSM8K, IFEval, AA-LCR and the rest); vLLM on H100.
- **Original Bonsai (different checkpoint):** [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf),
  temperature 0.7.

Our four complete runs used the **original-Bonsai protocol** (temperature 0.7 and the original caps), because
that was the published target when they started. Against Bonsai 2's own numbers they are therefore
cross-protocol: same weights, different temperature and caps. They do not isolate the engine's effect.
Full same-weight reference arms on our hardware remain unfinished; small overlapping reference screens are
documented in the [benchmark report](benchmarks/README.md).

| Benchmark | Bonsai 2 published (T 1.0) | Original Bonsai published (T 0.7) | Ours, Bonsai 2 weights (T 0.7, cap) | Evidence |
|---|---:|---:|---:|---|
| GSM8K | 96.66% | 96.06% | **96.74% — 1,276/1,319** (cap 16,384) | [Native-trigger grading and receipts](benchmarks/README.md#gsm8k) |
| MATH500 | 98.80% | 99.20% | **98.80% — 494/500** (cap 81,920) | [Full quality report](benchmarks/README.md) |
| IFEval, prompt-strict | 91.31% | 85.03% | **92.42% — 500/541** (cap 20,480) | [Full quality report](benchmarks/README.md) |
| HumanEval+ (EvalPlus), pass@1 | 95.12% | 93.90% | **93.90% — 154/164** (cap 30,000) | [Full aggregate](benchmarks/receipts/humaneval-plus-full.json), [source binding](benchmarks/receipts/humaneval-plus-source.json) |

In items, against Bonsai 2's published scores: GSM8K +1, MATH500 identical, IFEval +6, HumanEval+ −2.
Five of our HumanEval+ responses hit our 30,000 cap; PrismML's Bonsai 2 cap was 49,152.

**AA-LCR (long-context reasoning), Bonsai 2 protocol: 73.0% — 73/100.** Full split, all 100 questions
generated and scored. Temperature 1.0, output cap 32,768, 163,840-token context, prompts 87,891–122,152
tokens, rule-first grading with a Gemini 3 Flash judge fallback, seed 20261014, AA-LCR revision `9a77ef56`.
Published: Bonsai 2 **77.0**, FP16 parent 78.0.

- **Capped answers:** 12 generations hit the output cap with an empty final answer and are counted wrong.
- **Errors and exclusions:** 0 transport errors, 0 remaining exclusions.
- **Generation:** 98 outputs come from the original run, which was interrupted for the PP512/TG128
  measurement. A continuation generated only the 2 missing items; no completed answer was regenerated.
- **Item 64:** its original judge reply failed to parse. That failure is retained, and a distinct judge request
  over the same output (8,192 tokens, low thinking) graded it INCORRECT. No scoring rule changed.

[Settings, result, commands and every output with its verdict](benchmarks/aa-lcr/b2lc-20261015-halo-aa_lcr-full/).
Earlier running snapshots of this same run are kept in git history as historical states, not results:
52/73 at `b197222` and 62/85 at `5118ac2`. A separate full run with a raised output cap (40,960) is in
progress and will be reported as its own result. The capped failures above stay attributed to this run.

Canonical GSM8K uses rule-first grading with judge fallback **only on parse failure**. The retained
outputs contain 1,276 parsed correct answers, 34 parsed wrong answers and nine empty parse failures;
the judge gives those nine zero. A broader regrade of every rule miss produces
**97.27% — 1,283/1,319**, retained **only as a diagnostic**. It is not the canonical benchmark result.
Gold answers were not changed. [Canonical score receipts](benchmarks/receipts/canonical-scores.json)
and [programme-12 provenance](benchmarks/receipts/programme12-source.json) bind the grading to
`08dadac9f891c4ccdad2ed5db853d8001a1c68ac` and keep both policies explicit.

Thinking is on; temperature 0.7, top-p 0.95, top-k 20. Output caps are 16,384 (GSM8K),
81,920 (MATH500), 20,480 (IFEval) and 30,000 (HumanEval+). Our explicit choices for unpublished details include zero-shot,
xhigh reasoning and fixed generation/dataset seeds. Empty capped finals stay wrong. The publisher's
exact historical extractor, prompts, shots, harness/dataset revisions and instruction-check seed are
unknown. Judge-replay and programmatic scores have different reproduction requirements.

HumanEval+ is complete across six disjoint windows: **31/32, 28/32, 29/32, 32/32, 30/32, 4/4**.
All **164** tasks remain in the denominator, including **159 EOS/reasoning-closed** and **five capped/unclosed**
responses. The result is bound to programme-12 commit
`eb6cd0356ca6176f27741b8948e2f08c4f582de7` and its
[retained result bindings](benchmarks/receipts/retained-result-bindings.json). Rounded agreement with
PrismML's original-model score does not establish same-checkpoint engine equivalence.

**Tau2 partial snapshot — excluded from the full-result table:**

| Benchmark | Retained completed portion | Full benchmark state |
|---|---|---|
| Tau2 telecom | **66 correct / 82 scored**, out of 114 intended occurrences | [Partial receipt](benchmarks/receipts/tau2-partial.json): two errors, one untouched, 29 missing; full score unset |

These are frozen export observations of unfinished campaigns. No partial is presented as a completed
suite score, and prepared adapters for other benchmarks establish no quality result.

Reduce the exported quality receipts without GPU, dependencies or paid calls:

```sh
python benchmarks/replay.py
```

This checks GSM occurrence-level grading decisions and disjoint shard coverage; it does not rerun
the model, judge, instruction checker or code execution.

## Build and reproduce the primary engine

Requirements: RTX 5090 / `sm_120a`, CUDA Toolkit 13+, CMake 3.28+, Ninja, Python 3.10+;
Windows additionally needs Visual Studio 2022's x64 Native Tools prompt. Linux needs a compatible
host compiler; its complete clean-build GPU qualification remains pending. Allow model storage and
32 GB VRAM. Model and drafter weights are acquired separately under their original licenses.

```sh
python scripts/fetch_model.py
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
python scripts/bench.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
python scripts/bench.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --profile deterministic --check-determinism
```

The model fetch pins its revision and SHA256. CMake fetches PrismML's llama.cpp at
`6067dbea986ed1359db98c42ab06eb49426bbf16`, CPU-only, for tokenization/chat templates.
`bench.py` checks the prompt fixture and frozen GEMM plan and writes per-pass timings, environment,
input identities and binary hashes under `results/`. Generated results are local; retained evidence
lives in `receipts/`.

### Serve and run GSM8K

```sh
python -m pip install -r requirements.txt
python engine/serve.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --profile profiles/serving.json --slots 8 --ctx 20480 --port 18322
# In another terminal:
python scripts/gsm8k.py --url http://127.0.0.1:18322 --out results/gsm8k
```

The server accepts already-templated prompts at `/v1/completions` and batches at
`/v1/batch/completions`. Sampling runs on the GPU. The portable GSM script uses Math-Verify rule
scoring, which reproduces the retained canonical count on these outputs; it does not itself call a
fallback judge. Use the [full benchmark harness](benchmarks/README.md) for native-trigger replay,
MATH500, IFEval and partial tool/code tasks.

To rescore the original retained GSM8K outputs without GPU inference:

```sh
python -c "import gzip,pathlib; p=pathlib.Path('results/gsm8k'); p.mkdir(parents=True,exist_ok=True); (p/'outputs.jsonl').write_bytes(gzip.decompress(pathlib.Path('receipts/gsm8k/outputs.jsonl.gz').read_bytes()))"
python scripts/gsm8k.py --out results/gsm8k --score-only
```

The dataset fetch still needs network access. Fresh temperature-0.7 generation is not expected to
produce identical sampled answers or exactly the same score. Historical quality scores are retained
measurements, not a claim that this expanded repository was rerun end-to-end at publication.

## Repository map

| Path | Purpose |
|---|---|
| `engine/`, `CMakeLists.txt`, `profiles/` | Portable batch-decode engine, C/Python API, HTTP server, frozen plan |
| `scripts/`, `data/` | Model fetch, throughput harness, portable GSM scorer and prompt fixture |
| `receipts/` | Original ladder, final-engine/clean-build timings and retained GSM outputs |
| `docs/ladder.md`, `docs/steps/` | Measured optimization sequence and explanatory source excerpts |
| `docs/reproduction.md` | What has actually run from the clean portable build |
| `engines/` | Additional coherent engine snapshots and their own build/results guides |
| `research/` | Binary, packed/five-wide, full-K, Dict8 and related source/finite experiments |
| `benchmarks/` | Quality harness, source pins, complete and explicitly partial results |

The added source snapshots preserve their own configuration contracts. An experimental loss or
successful microkernel check does not select a new serving default. Hardware/runtime work was not
started by this publication.

## License and exclusions

See [LICENSE](LICENSE) and [NOTICE](NOTICE), plus component notices. Third-party dependencies,
benchmarks and datasets retain their original licenses. Weights, model-derived activation/cache
payloads, compiled binaries, generated dependencies, credentials and private personal data are
excluded. Only sanitized source snapshots and selected scientific evidence are exported; private
source Git history and machine-operation state remain outside this repository. Our own code is
Apache-2.0.
