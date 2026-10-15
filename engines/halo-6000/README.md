# Halo6000 — ternary MoE CUDA

Export of main `fec60de800f9a3a5a236a65270b79426f5e4fa5e`. Native fixed-shape Qwen3.6-35B-A3B engine for RTX PRO6000 Blackwell (`sm_120a`,188SM). It is a different model and engine from the two Bonsai27B trees and the root B64 reproduction.

## Build/run

CUDA12.8+, C++20, OpenMP, OpenSSL development libraries and PrismML llama.cpp at `1a07bfa5f4144274c8f1c9963821dd9d9a51854b` with common/tokenizer libraries. See [dependencies](../DEPENDENCIES.md).

```sh
make -j4 build/halo-6000 FORK=/path/to/prism-llama
./build/halo-6000 --help
./build/halo-6000 serve --model MODEL.gguf --host 127.0.0.1 --port 8080 \
  --slots 8 --context-per-slot 4096 --thinking off --raw-content
```

Input is the separately prepared ternary v1 PTQ1_0 Qwen3.6-35B-A3B GGUF, or the measured practical hybrid with Q8 nonexpert/Q2_K expert down tensors. These are custom model preparations; a stock Qwen checkpoint is not a substitute. No weights or complete preparation/training pipeline are supplied here. The versioned cache loader rejects unsupported shapes/types. Model/cache hashes and byte counts survive in selected receipts.

Native PANE2/Q8/Q2_K projections, router/top8, signed Hadamard preparation, recurrent and KV state and sampling are in `include/` and `src/`. Thirty GDN layers use FP32 state; ten attention layers use FP16KV. Libllama may tokenize but does not execute language forward.

## Entry points and limits

The ordinary CLI is the direct-launch correctness reference (default row-lanes32; `--row-lanes 8` chooses a measured faster arithmetic-equivalent reduction tree). `include/batch.hpp`/`src/batch.cu` expose independent-slot native batch decode; `bench/batch.cu` and `bench/run-batch.py` exercise it. `src/schedule*.cu`, `include/tune.hpp` and `tune/` retain the separate exact graph search/backend. They are **not the default CLI backend**. `tune/build.sh`, `tune/search.py`, `tune/driver.cu`, `tune/space.py`, selected best headers and JSON preserve that experiment; inspect arguments and pass explicit local paths.

HTTP serving remains greedy/nonstreaming, thinking-off and **serialized single-stream**, even with8 workers/slots. It resets full GDN/conv/KV state per request. Native batching is not integrated into HTTP scheduling. Temporal whole-model prefill, tuned32stream throughput, MTP and several Q4/FP4 routes remain unfinished. Helper code or cache acceptance does not establish integration.

## Measured evidence (October5,2026)

One RTX PRO6000, warm native kernels, weights/preparation excluded from timing. These are historical source-specific receipts, not new measurements of the export commit.

| Workload | Observed rate / scope | Local evidence |
|---|---|---|
| v1 direct single stream,grid188,40warm/128timed teacher-forced steps |239.74/302.20/309.19tok/s at row-lanes32/16/8 |`docs/launch-cache-b/timing.json`; lanes are **not stream counts** |
| v1 native independent-slot GPU-feedback batch,capacity/context128,8warm/16timedsteps |s8=856.25,s32=1019.14 aggregate tok/s |`bench/batch-results/halo-batch-v1-r8.json`, `halo-batch-v2-r32.json` |
| Practical hybrid,same batch schedule |s8=733.07,s32=893.53 aggregate tok/s |matching `halo-batch-hybrid-r*.json` |
| Exact graph search,40warm/128timed teacher-token replays,3reps/median |best s1=347.239tok/s |`tune/best.json`, `tune/results/{s1v6/best.json,summary.json}` |
| Exact graph search,s8 |~1056 aggregate tok/s final configuration; exploratory search best1058.78 |`tune/best-s8.json`, `tune/results/summary.json`; separate selected-vs-search measurements |
| Frozen native v1 intrinsic,4080scoredpositions |PPL18.1376563, KLD0.986707661 vs Q8 reference |`docs/results/native-intrinsic-v1.json`; measures model quality, not equivalence to Q8 |

Batch guards compare complete logits and all slot state with canonical serial-native math, including feedback/reset/ragged position cases. Exact graph guard compares16×248320logits, argmax and full GDN/conv/KV/positions bytewise before timing; it establishes equivalence to **native v1**, not to original Q8 weights. The near-exact same-weight fork comparison (`docs/hybrid-forward-b/forward-comparison.json`) is a different test:16teachersteps, argmax16/16,KLD≤0.00286,maxlogitabs0.439. Full logits/activations are excluded.

The frozen intrinsic reader expects the exact token fixture SHA256 recorded in `src/intrinsic.cpp`; the external dataset/token package and Q8 reference distributions are not included. Its retained JSON supports the historical reported score but cannot by itself rerun the complete quality evaluation.

No comparator-win claim follows from comparing these token schedules with llama-bench/fork baselines. Native row-parallel batching was slower than recorded fork PANE batching; the opt-in exact expert-bucketing candidate also lost (465.15 versus856.25aggregate s8). The graph search result has not been promoted into the ordinary CLI or HTTP backend. No generalized speedup is claimed.

Component oracle probes `bench/q2k.cu` and `bench/q4-head.cu` include upstream `vecdotq.cuh`; compile them with the pinned fork ggml CUDA include directories. They compare helpers against upstream math and are not dependencies of the native engine build.
