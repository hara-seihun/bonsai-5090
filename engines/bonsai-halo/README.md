# Bonsai Halo — AMD HIP

Export of master `1b71c6798d1d210e969d59a1513edf9c79b18447`. Fixed-shape Ternary Bonsai2 27B engine for Radeon8060S (`gfx1151`), with own HALO weight layout, persistent eight-row kernel, wide sequence execution, GDN/attention and MTP/DFlash2 paths. This export is separate from both NVIDIA engines.

## Build/run

ROCm/hipcc with gfx1151 support, C++17 and external PrismML llama.cpp vocabulary libraries. See [dependencies](../DEPENDENCIES.md). The Makefile accepts explicit compiler/architecture/library overrides. No original host pathname is needed.

```sh
make -j4 LLAMA=/path/to/prism-llama
./bonsai-halo -m MODEL.gguf -p 'Why is the sky blue?' -n 200
./bonsai-halo -m MODEL.gguf --bench -n 100
./bonsai-halo -m MODEL.gguf --bench --dflash DRAFTER.safetensors \
  --draft-weights q4 --prompts bench/drafter-prompts.txt -n 192 --sweep-rounds 3
make -j4 LLAMA=/path/to/prism-llama tools/batch_compare
./tools/batch_compare --help
```

Use the official Bonsai2 PTQ1_0 GGUF; loader also accepts PQ2_0. First use repacks into `.halo` beside the GGUF; this generated cache is not bundled. DFlash2 checkpoint is `incoai/Qwen3.8-27B-DFlash2`, MTP tensors come from Qwen3.8-27B shard18. `tools/draft_quant.cpp` prepares drafter caches. Libllama is vocabulary/chat-template only, not language inference.

The normal target uses FP32 recurrent state; optional A4 activations/INT8 state are approximate target modes. DFlash verification against an approximate target does not establish equivalence to an unquantized reference. Sampling options exist for plain decode; drafted paths are greedy. `src/server.cpp` and `src/serve_batch.cpp` are exported, but historical prototype concurrent speculative serving was not accepted as an equivalent faster server.

## Compact retained evidence

`receipts/full-tps/` and `receipts/generation-128/` include numeric JSON, benchmark logs and prompts from September21,2026. These are historical measurements, **not remeasurements of the export commit**.

| Source/workload | Observed output rate |
|---|---:|
| `83f0478307d166933fc39c2d5841c6792a3128e9`, single Q4 DFlash, FP32 state, wide-deployed prompt route; median of10 prompt medians,3 rounds, EOS/192tokens |69.45tok/s (per-prompt51.42–113.35) |
| Same panel, single DFlash + INT8state/A4 prompt/sequence input |62.08tok/s; output digests differ |
| Same panel, one-stream plain batch-driver decode/INT8state |33.70tok/s |
| Same panel,8 /32 streams,64 fixed greedy steps,3rounds,short distinct prompts,capacity1024 |132.55 /383.32 aggregate tok/s |
| `1c7d7f2dc4f9e5c3d9114ee75a8e37d5aaef6157`,64 /128 streams,mode19/INT8state/A4FFN/A4sequence-input,64fixedsteps,3rounds,capacity256,128allocatedslots |463.14 /517.73 aggregate tok/s;7.24 /4.04 per stream |

No drafter runs in those multi-stream rows. EOS is fed back until the fixed step count. Loading/static preparation/prompt processing are excluded from generation timing. Capacity is not starting context. At128streams explicit managed allocation and43GiB process budget were used (43.81GB tracked GPU-addressable allocation); the64stream row used the same memory-saving wide-only weight policy. These are approximate target routes, not evidence of task-quality parity. Later engine changes superseded some rates; do not advertise these historical maxima as the latest engine-wide ceiling.

The first panel also records768-token full-model prefill at862.22tok/s median (three rounds778.28/935.50/862.22), separate from generation. Ten single-stream starting prompts contain23–49tokens; context capacity4096. The two drafted configurations ran in separate processes and changed multiple options together; no isolated attribution follows.

## Replay

Run on a dedicated, sufficiently large gfx1151 system. The direct `tools/batch_compare` executable is the retained benchmark driver; `tools/batch_compare.py` analyzes its output. Inspect `--help` for source-version flags. Historical commands/precision/capacity are in the receipt JSON; local service-stop, clock-management and machine-admission wrappers are intentionally excluded. `HALO_GDN_STATE=f32 HALO_SEQ_QUANT=a8` with `--prefill-ffn wide-deployed` identifies the faster historical drafted configuration; `i8`, `a4`, `wide-commit-a4` identify the approximate alternative. Current code is not guaranteed to reproduce old timing or dispatch exactly.

All kernels/build dependencies owned by this repository are included, including vendored cpp-httplib and nlohmann/json with notices. Full logits, weights, generated dependencies and MoE research tools unrelated to this engine are excluded. No broad quality benchmark or updated GPU acceptance was run for this export.
