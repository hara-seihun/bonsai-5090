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

## Quality against PrismML's published results

PrismML's original [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
reports the original Bonsai generation. We evaluated **Bonsai 2** with the recovered original
thinking-on protocol. The checkpoints differ: these nominal comparisons do **not** isolate the
engine's effect. Full same-weight reference arms remain unfinished; small overlapping reference
screens are documented in the [benchmark report](benchmarks/README.md).

| Benchmark | PrismML published | Our complete result | Evidence |
|---|---:|---:|---|
| GSM8K | 96.06% | **96.74% — 1,276/1,319** | [Native-trigger grading and receipts](benchmarks/README.md#gsm8k) |
| MATH500 | 99.20% | **98.8% — 494/500** | [Full quality report](benchmarks/README.md) |
| IFEval, prompt-strict | 85.03% | **92.4% — 500/541** | [Full quality report](benchmarks/README.md) |

Canonical GSM8K uses rule-first grading with judge fallback **only on parse failure**. The retained
outputs contain 1,276 parsed correct answers, 34 parsed wrong answers and nine empty parse failures;
the judge gives those nine zero. A broader regrade of every rule miss produces
**97.27% — 1,283/1,319**, retained **only as a diagnostic**. It is not the canonical benchmark result.
Gold answers were not changed. [Canonical score receipts](benchmarks/receipts/canonical-scores.json)
and [programme-12 provenance](benchmarks/receipts/programme12-source.json) bind the grading to
`08dadac9f891c4ccdad2ed5db853d8001a1c68ac` and keep both policies explicit.

Thinking is on; temperature 0.7, top-p 0.95, top-k 20. Output caps are 16,384 (GSM8K),
81,920 (MATH500) and 20,480 (IFEval). Our explicit choices for unpublished details include zero-shot,
xhigh reasoning and fixed generation/dataset seeds. Empty capped finals stay wrong. The publisher's
exact historical extractor, prompts, shots, harness/dataset revisions and instruction-check seed are
unknown. Judge-replay and programmatic scores have different reproduction requirements.

**Partial snapshots — excluded from the full-result table:**

| Benchmark | Retained completed portion | Full benchmark state |
|---|---|---|
| Tau2 telecom | **66 correct / 82 scored**, out of 114 intended occurrences | [Partial receipt](benchmarks/receipts/tau2-partial.json): two errors, one untouched, 29 missing; full score unset |
| HumanEval+ | **120 correct / 128 scored**, out of 164 intended tasks | [Partial receipt](benchmarks/receipts/humaneval-plus-partial.json): completed disjoint windows only; full score unset |

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
source Git history and machine-operation state remain outside this repository. The repository
remains private.
