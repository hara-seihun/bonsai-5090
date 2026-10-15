# Original CUDA Bonsai 2 27B engine

Export of `halo-5090` master `772d9060ec5b5eba5b7429833331843b93eca93e`. This is the original ternary model engine, separate from the root portable B64 variant.

## Build and use

CUDA 12.8+, RTX 5090 (`sm_120a`), C++17, OpenSSL development libraries. Build PrismML llama.cpp at `1a07bfa5f4144274c8f1c9963821dd9d9a51854b` as described in [dependencies](../DEPENDENCIES.md). Supply weights yourself: `prism-ml/Ternary-Bonsai-2-27B-gguf` PTQ1_0 language GGUF; `incoai/Qwen3.8-27B-DFlash2` model.safetensors; optional official Q8_0 mmproj GGUF.

```sh
make -j4 LLAMA=/path/to/prism-llama
./h5 bench MODEL.gguf 1 4000 10
./h5-server MODEL.gguf DRAFTER.safetensors --slots 64 --ctx 4096 --port 8000
python3 tools/loadtest.py http://127.0.0.1:8000 --concurrency 8 --requests 24 --max-tokens 512 --stream
python3 tools/gsm8k_server.py http://127.0.0.1:8000 gsm8k_test.jsonl 1319 64
```

`h5-server` supports greedy chat/completions, SSE, continuous batching, usage, cancellation, stop strings, health/metrics. Text uses libllama vocabulary only. Native CUDA owns language forward, GDN, attention and batched DFlash2. Normal activations are approximate NVFP4, recurrent/pending storage FP16, KV FP8. `H5_ACT=int8 H5_WEIGHTS=int8 H5_STATE=f32 H5_KV=f16` selects the higher-precision control. `H5_WSCALE=e4m3` is an explicit approximate group-scale option.

`H5_COMPACT=dense5` is an opt-in 26-byte/128-trit reader with original FP16 scales; incompatible with E4M3 scales/megakernel. `H5_TREE2=1 H5_COMPACT=dense5 H5_CLONE_FULL_KV=1` enables the single-active-text two-path HTTP experiment. `H5_DRAFT_MARGIN=1` changes acceptance and therefore output policy; it is neither strict speculation nor a quality-equivalent replacement. Tree and relaxed-margin modes are separate.

## Retained measurements

October 6, 2026, single RTX 5090. Each row is its own workload/precision configuration. `results/` contains raw summaries, per-request timings, public/synthetic prompts, selected generated outputs and score receipts. Historical result source identities differ from the export commit.

| Workload | Result | Evidence |
|---|---|---|
| Sequential single-slot HTTP, eight mixed-chat prompts twice, max512, temp0/thinking off; exclude 256-token warmup and startup, include request prefill/output | original 360.57; compact strict 376.60; compact margin1 443.92 output tok/s | `results/single-stream/{single-baseline,compact-margin-0,compact-margin-1}/summary.json` |
| Same reused exploratory GSM8K128 panel, max768, one stream | strict121/128; margin1 120/128 | `results/single-stream/quality128-margin-*/score.json`; this does not establish equivalent quality |
| Matched compact strict single-active-text HTTP two-path panel | 365.17→398.72 e2e tok/s; reused GSM128 121/128 both, different mistakes | `results/tree-http-{base,two}/`, `results/tree-gsm-{base,two}/`; one mixed-chat poem-count miss, no default enable |
| Full GSM8K1319 short-context HTTP, concurrency64, max768, thinking off; FP16 state + E4M3 scales | 5548.85 output tok/s,1241/1319 (94.09%) | `results/combined/fp16-scales-full.gsm.txt`, matching meta/metrics; request prefill and batch tail included |
| Shared-prefix CLI GSM8K1319,60 slots, ~3956–3961 decode context; FP16 state/original scales | 4162 e2e /5069 decode-only tok/s,1253/1319 | `results/combined/fp16-4k.log`; excludes one-time prefix preparation; **not HTTP** |

These short mixed-chat, math, prefix-cloned CLI and forced acceptance kernel tests are different benchmarks. No cross-row general speedup or cross-hardware claim follows. Dynamic approximate arithmetic may change output across batching/tuning choices.

## Vision

```sh
make -j4 h5-server h5-vision-check LLAMA=/path/to/prism-llama
./h5-server MODEL.gguf DRAFTER.safetensors --slots 2 --ctx 4096 \
  --mmproj MMPROJ.gguf --vision-gpu 0 --image-max-tokens 1024
```

Native IMRoPE and visual embeddings feed the CUDA language engine; `libmtmd` encodes images. CPU encoder is explicit default; `--vision-gpu 1` needs a CUDA-enabled ggml encoder build. `tools/prepare_vision_cuda.py` contains the pinned focused encoder build recipe. Images are OpenAI-style image_url parts; `vision_speculation: true` opts into strictly target-verified image proposals. Unsupported media/types are rejected.

Synthetic OCR/inventory/chart/spatial tasks scored10/10 ordinary and10/10 speculative with original FP16 scales and1024 image-token cap (`results/vision-tasks-fp16scale/`); generator/scorer is `tools/vision_tasks.py`. Three higher-precision INT8/F32/F16KV controls matched ordinary/speculative128-token outputs (`results/vision-gpu-spec-control-fixed/`). Production FP4 ordinary/speculative strings can differ; three simple fixtures measured ordinary156–158 versus speculative266–470 decode tok/s (`results/vision-gpu-spec/`). These are bounded functional checks, not broad visual accuracy or a general vision speedup.

## Reproduction boundary

`src/`, `Makefile`, `bench/`, `experiments/` and local `tools/` are retained. `tools/evals.py`, GSM8K clients, loadtest and vision harnesses require their Python dependencies/datasets. Model weights, full logits, token-output binaries, long-draft external audit captures and host deployment scripts are excluded. Historical6,105tok/s belongs to a different600W GPU run; the table above uses retained matched575W combined-run evidence. No new performance acceptance is claimed for the current long-draft code.
