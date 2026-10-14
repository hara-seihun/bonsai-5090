# Ternary Bonsai 2 27B on one RTX 5090

A standalone CUDA decode engine for PrismML's
[Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (PTQ1_0 GGUF), with
the scripts to build it, benchmark it, serve it and run GSM8K on it, plus a step-by-step account of how
it got from PrismML's llama.cpp fork to here.

## Results

| Measurement | Value | Conditions |
|---|---:|---|
| Batch-64 decode throughput | **6,777 tok/s** (median; 6,778.4 / 6,777.4 / 6,775.7) | RTX 5090, Windows, 64 streams × 128 timed tokens, 3 passes, `profiles/serving.json` |
| Same, deterministic mode | **6,583 tok/s** (median; 6,585.4 / 6,583.1 / 6,579.8) | as above + `H5_DETERMINISTIC_GEMM=1`; repeats are bit-identical |
| GSM8K, full test set | **1276 / 1319 (96.74%)** | thinking on, temperature 0.7, see below |
| PrismML llama.cpp baseline | 930.6 tok/s | different RTX 5090 (Linux), 64 × 1,024 tokens; see [docs/ladder.md](docs/ladder.md) |
| **Clean checkout of this repo** | **6,731.7 tok/s** ordinary; **6,550.5 / 6,549.7** deterministic (tokens identical) | same RTX 5090 as the first two rows, fresh build; [docs/reproduction.md](docs/reproduction.md) |

The first two rows are the original measurements of the engine. A fresh build of this repository on the
same GPU currently runs 0.67% (ordinary) and 0.50% (deterministic) slower; the cause has not been
isolated. See [docs/reproduction.md](docs/reproduction.md).

Throughput counts generated tokens across all 64 streams during the timed decode window only. Model
load, load-time weight conversion, plan tuning, tokenisation and prefill are excluded; 8 warm-up tokens
per stream precede the timer. Decoding is greedy with no speculation. The two machines were never
measured under identical conditions, so the ladder's same-machine ratio (6.93×, row 6 vs baseline) is
the honest speedup figure. See [docs/ladder.md](docs/ladder.md).

Precision: weights are PrismML's ternary values; activations are FP4 (E2M1, block-scaled), the
recurrent state is six-bit, and the KV cache is FP4. All three are lossy relative to the baseline. The
GSM8K score is measured on exactly this configuration.

## Requirements

- NVIDIA RTX 5090 (sm_120a; the kernels use Blackwell block-scaled FP4 MMA and won't run elsewhere)
- CUDA Toolkit 13.0 or newer, CMake ≥ 3.28, Ninja
- Linux: GCC 13+. Windows: Visual Studio 2022 (MSVC 14.4x), run from an x64 Native Tools prompt
- Python 3.10+; `pip install -r requirements.txt` for GSM8K only
- About 6 GB of disk for the model and 32 GB of VRAM

The build fetches PrismML's llama.cpp fork at a pinned commit and compiles it CPU-only. The engine uses
it only for the tokenizer and chat template.

## Reproduce

```bash
python scripts/fetch_model.py                        # 5.9 GB, pinned revision, sha256-verified
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build                                   # the CUDA kernels take a while to compile
python scripts/bench.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
python scripts/bench.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --profile deterministic --check-determinism
```

Each `bench.py` run writes a JSON receipt under `results/` with the per-pass numbers, the exact
environment, input hashes, binary hashes and `nvidia-smi` device line. GEMM launch geometry is read from the frozen plan in
`engine/frozen-production-plan.txt`, which `bench.py` verifies is unchanged before and after the run;
any plan output is written to a separate file in the run directory, never over the frozen plan.

### Serving

```bash
python engine/serve.py --model models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --profile profiles/serving.json \
    --slots 8 --ctx 20480 --port 18322
```

`POST /v1/completions` takes one raw, already-templated prompt; `POST /v1/batch/completions` takes
`{"requests": [...]}` with up to `--slots` entries and decodes them as one batch. Sampling
(temperature, top-p, top-k, min-p, presence and repetition penalty, per-request seed) runs on the GPU.

### GSM8K

```bash
python scripts/gsm8k.py --url http://127.0.0.1:18322 --out results/gsm8k
```

This sends the same request bytes as the measured run: the dataset revision, chat-template rendering,
system prompt, sampling settings and per-item seeds are fixed in the script (`--print-request N` shows
one). Scoring takes the text after `</think>` and compares it with the gold answer using Math-Verify.
Sampling at temperature 0.7 on a GPU is not bit-reproducible across runs (see deterministic mode), so
expect a score near 1276, not necessarily equal. The measured outputs are in
`receipts/gsm8k/outputs.jsonl.gz`; `--score-only` against them reproduces 1276 exactly.

## Layout

| Path | Contents |
|---|---|
| `engine/` | Engine source (`src/`), C API (`api.cpp`), FP4 KV dispatch, benchmark driver (`length_bench.cpp`), HTTP server, frozen GEMM plan |
| `profiles/` | Exact runtime settings for the measured runs |
| `scripts/` | `fetch_model.py`, `bench.py`, `gsm8k.py` |
| `data/` | Benchmark prompt fixture (GSM8K questions, see NOTICE) |
| `docs/ladder.md`, `docs/steps/` | The optimisation ladder, change by change, with code views |
| `docs/reproduction.md` | What was verified, where, and what is still pending |
| `receipts/` | Ladder measurements, headline receipts, GSM8K outputs |

## License

Engine source and scripts: Apache-2.0 ([LICENSE](LICENSE)). Third-party material and attributions:
[NOTICE](NOTICE). The model weights are not included; download them from PrismML under their license.
