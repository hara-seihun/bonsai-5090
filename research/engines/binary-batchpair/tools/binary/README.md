# Binary Bonsai 27B on an RTX 5090

Private October10 campaign. Source owner: local `halo-5090` branches `binary-bonsai-5090` (ordinary ladder), `binary-modern-baseline` (isolated modern F32 control), `binary-spec-bonsai-5090` (modern speculative/ISA integration), and `binary-batchpair-bonsai-5090` (explicit batch-axis IMAD engine candidate). Public pushes are not part of this campaign.

Target: largest released binary checkpoint, `prism-ml/Bonsai-27B-gguf`, original Q1 signs and FP16 K128 scales, no Hadamard. This is not Ternary Bonsai2. Alpha/beta matrices are binary too. Presentation index: `/PRIVATE_PRESENTATION_NOT_EXPORTED`, **Binary**. Numerical/provider receipts: `results/binary-5090/`.

## Measured boundaries

Ordinary matched real-token cached decode: Prism195.79, initial INT838.56, old2bit FP4single165.91, packed1 F32activations200.01, packed1/F16state203.62, binary1 FP4single202.00, ordered two-plane FP4116.70, two-plane/F16state117.20 tok/s. Packed1/F16state interleaved repeats203.30–203.61 versus fresh Prism195.29–195.95 (+3.75–4.11%). Published binary27B RTX5090 rate162.6 has a different protocol; H100104.8 is not its same-card comparator.

Wiki256 Prism PPL19.1780; packed1/F16state19.1952, KL.0001502, top1252/256. Fixed chat64/64 top1. SingleFP4 loses quality (19.6712); two-plane restores it (19.1706) but ordered passes lose speed. Alternatives are not additive improvements.

Modern isolated baseline `f0ed506` gives203.38 tok/s, same F32 intrinsic quality. ISA source `4a81c7c` gives203.25 F32,192.92 pureA8,202.80 hybrid. Hybrid preparedR1 is F32; preparedR>1 is A8. Wiki256 hybrid chunk16 prefill/decode PPL19.2377/19.2036, KL.0005152/.0002187. Full HTTP/spec/batch task ladder is in progress; do not borrow ternary acceptance or token rates.

Reference task controls, each32GSM +32HE+, greedy/thinkingoff, ctx8192, cap4096: S1 GSM29/32 withonecap, HE+32/32,21304tokens/127.576generationseconds=166.99 HTTP/client-wall tok/s. B32 same scores/cap,21178/109.968=192.58tok/s. These are scoped panels, not full quality gates and not teacher-forced decode.

## Packed research owners

- [Exact F32 dynamic quartet LUT](../../experiments/binary-packed/README.md): known LUT/Four-Russians family implemented in packed coordinates; [LUT8/ILP negative round](../../experiments/binary-packed/TUNING.md).
- [A8 instruction composition](../../experiments/binary-bitserial/README.md): nibble IMAD/AND → 0/1-byte DP4A → shared `2P−DC`, original FP16 scales. Exact for given A8, not F32. Trained fullhead R8/R16 inclusive214.8/360.4µs versus F32 LUT415.7/831.5. Producer, operand wiring, DC and scale/reduction are timed. Popcount negative retained. Whole-model gains require the task ladder.
- [Batch-axis paired IMAD](../../experiments/binary-batch-map/README.md): standalone source/proof pin `700d50aaee9f34af2bc7cbfcf26dc824bf4e5172`; opt-in engine integration below. Projection results are mostly negative; whole-model results belong to the campaign parent.
- [Joint residual FP4 consumer](../../experiments/binary-fp4-pair/README.md): reuse each packed weight for both existing planes, original scales; inclusive trained geometry1.15–1.78× projection gains with synthetic activations. FP32 association changes, max relative RMS2.21e-7. Whole-model integration/quality is separate.
- GPU Q1→FP4 load transform `4660136`: twelve trained-layout fixtures matched every prepared byte/scale; `tools/binary/repack/check.cu` on ordinary branch. Startup is separate from tok/s.

## Explicit batchpair engine candidate

Set `H5_WEIGHTS=packed1 H5_ACT=a8-batchpair` for paired A8 at every prepared row count, including R1 with a charged ghost lane. Set `H5_ACT=f32-r1-a8-batchpair` for unchanged F32 at **prepared** R1 and paired A8 at prepared R>1. Existing `f32`, `a8`/DP4A and `f32-r1-a8` behavior is unchanged. Both paired modes require row capacity1..512 and original Q1/FP16 scales.

The engine prepares one immutable positive-sign-count byte/K128 block. Each activation preparation produces the whole batch once into `ceil(R/2)*K` packed uint32 words plus the existing per-real-token FP32 scale/int32 DC/status and persistent error word. Odd batches pay the complete ghost lane. Every projection tiles at most16 rows from an even first token; output and split-K scratch are tile-local. The error word clears once per body, never per producer, and is checked at the normal synchronized body fence. Nonfinite/out-of-range activation errors remain explicit.

The explicit structural plans use chunk16/128threads for full or restricted vocabulary heads; down uses chunk1/128 at B2, chunk4/256 at B4/B5, otherwise chunk4/128; GU uses chunk4/256 at B4, chunk16/256 at B32, otherwise chunk4/128. Other projections use chunk4/128. Plans use the whole prepared batch size, not its tile size. These are standalone projection recommendations, not claims about whole-model speed. Parent measures B2/4/5/8/16/32 and owns quality/actual throughput.

Build only the changed objects locally with CUDA13.1 release, empty `DEFS`, and `-gencode arch=compute_120a,code=sm_120a`:

```sh
make -j2 PROFILE=release NVCC=/usr/local/cuda-13.1/bin/nvcc build/engine.o build/batch_map.o
```

`Makefile` includes `batch_map.o` in every native engine link target. Giant kernels/FP4/draft objects remain unchanged. Retained immutable changed objects and toolchain/source hashes live under `~/data/binary-5090/batchpair/SOURCE_SHA/`; they are campaign build artifacts, owned here, reusable only with this exact source/configuration and the parent's validated unchanged-object manifest. Delete after campaign custody/export no longer needs them. No rental or GPU is touched by this integration.

## Inputs

- `Bonsai-27B-Q1_0.gguf`:3,803,452,480B, SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Prism fork `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`; local `~/work/clones/bonsai-hip`. Recover by cloning `https://github.com/PrismML-Eng/llama.cpp.git` and checking out that commit.
- PCr2 GPTQ2: `~/work/tune-5090/drafter-retrain/weights/r2-gptq2.safetensors`, SHA256 `bf5496a7153408c7ac4ed3e7213c6b1482f1fbed4b12ed81b6f1b4d398925692`. External ProCreations/DFlash checkpoint, not our binary-specific training.
- Drafter vocabulary `~/work/tune-5090/spec-ideas/vocab-65536.txt`. Restricts drafter head, not target's final greedy vocabulary.
- WikiText corpus `~/data/engine-shootout/corpus/wiki.test.raw`, SHA256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. First128tokens prefix, next256 scored, positions reset.

Fetch target: `bash tools/binary/fetch.sh MODEL_DATA_DIR`; hash is mandatory. No silent substitution of a changed upstream artifact.
