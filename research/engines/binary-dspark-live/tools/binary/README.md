# Binary Bonsai 27B on an RTX 5090

Private October10 campaign. Source owner: local `halo-5090` branches `binary-bonsai-5090` (ordinary ladder), `binary-modern-baseline` (isolated modern F32 control), and `binary-spec-bonsai-5090` (modern speculative/ISA integration). Public pushes are not part of this campaign.

Target: largest released binary checkpoint, `prism-ml/Bonsai-27B-gguf`, original Q1 signs and FP16 K128 scales, no Hadamard. This is not Ternary Bonsai2. Alpha/beta matrices are binary too. Presentation index: `/PRIVATE_PRESENTATION_NOT_EXPORTED`, **Binary**. Numerical/provider receipts: `results/binary-5090/`.

the operator's October10 rule: Prism's fork/reference is an **output-quality oracle, not a structural specification**. Measured PPL/KL/task quality is the contract. Every static encoding, kernel boundary, activation/state/KV representation and scheduler may be redesigned for speed; preserve no structure merely because the fork uses it. Retained original scales and explicit BF16 DSpark are measured controls, not restrictions on subsequent native representations.

## Measured boundaries

Ordinary matched real-token cached decode: Prism195.79, initial INT838.56, old2bit FP4single165.91, packed1 F32activations200.01, packed1/F16state203.62, binary1 FP4single202.00, ordered two-plane FP4116.70, two-plane/F16state117.20 tok/s. Packed1/F16state interleaved repeats203.30–203.61 versus fresh Prism195.29–195.95 (+3.75–4.11%). Published binary27B RTX5090 rate162.6 has a different protocol; H100104.8 is not its same-card comparator.

Wiki256 Prism PPL19.1780; packed1/F16state19.1952, KL.0001502, top1252/256. Fixed chat64/64 top1. SingleFP4 loses quality (19.6712); two-plane restores it (19.1706) but ordered passes lose speed. Alternatives are not additive improvements.

Modern isolated baseline `f0ed506` gives203.38 tok/s, same F32 intrinsic quality. ISA source `4a81c7c` gives203.25 F32,192.92 pureA8,202.80 hybrid. Hybrid preparedR1 is F32; preparedR>1 is A8. Wiki256 hybrid chunk16 prefill/decode PPL19.2377/19.2036, KL.0005152/.0002187. Full HTTP/spec/batch task ladder is in progress; do not borrow ternary acceptance or token rates.

Reference task controls, each32GSM +32HE+, greedy/thinkingoff, ctx8192, cap4096: S1 GSM29/32 withonecap, HE+32/32,21304tokens/127.576generationseconds=166.99 HTTP/client-wall tok/s. B32 same scores/cap,21178/109.968=192.58tok/s. These are scoped panels, not full quality gates and not teacher-forced decode.

## Packed research owners

- [Exact F32 dynamic quartet LUT](../../experiments/binary-packed/README.md): known LUT/Four-Russians family implemented in packed coordinates; [LUT8/ILP negative round](../../experiments/binary-packed/TUNING.md).
- [A8 instruction composition](../../experiments/binary-bitserial/README.md): nibble IMAD/AND → 0/1-byte DP4A → shared `2P−DC`, original FP16 scales. Exact for given A8, not F32. Trained fullhead R8/R16 inclusive214.8/360.4µs versus F32 LUT415.7/831.5. Producer, operand wiring, DC and scale/reduction are timed. Popcount negative retained. Whole-model gains require the task ladder.
- [Finite GU continuation fusion](../../experiments/binary-composition/README.md): retains integer dots and dynamic scales through SiLU×up→A8, eliminating FP32 roundtrips. Given the same split schedule, next-carrier bytes/scales/DC/status and trained down outputs matched bitwise. Inclusive FFN gain1.10/1.18/1.20× R1/8/16; engine integration separate.
- [Batch-axis packed map](../../experiments/binary-batch-map/README.md): lifted signedA8 pairs share two16-bit lanes in one32-bit IMAD accumulation, no carry at K128; odd B5 ghost lane paid. Native262144integer outputs exact. Fullhead mostly loses against DP4A (B8 286.75vs212.99µs); native batchtok/s ladder pending.
- [Joint residual FP4 consumer](../../experiments/binary-fp4-pair/README.md): reuse each packed weight for both existing planes, original scales; inclusive trained geometry1.15–1.78× projection gains with synthetic activations. FP32 association changes, max relative RMS2.21e-7. Whole-model integration/quality is separate.
- GPU Q1→FP4 load transform `4660136`: twelve trained-layout fixtures matched every prepared byte/scale; `tools/binary/repack/check.cu` on ordinary branch. Startup is separate from tok/s.

## Inputs

- `Bonsai-27B-Q1_0.gguf`:3,803,452,480B, SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Prism fork `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`; local `~/work/clones/bonsai-hip`. Recover by cloning `https://github.com/PrismML-Eng/llama.cpp.git` and checking out that commit.
- PCr2 GPTQ2: `~/work/tune-5090/drafter-retrain/weights/r2-gptq2.safetensors`, SHA256 `bf5496a7153408c7ac4ed3e7213c6b1482f1fbed4b12ed81b6f1b4d398925692`. External ProCreations/DFlash checkpoint, not our binary-specific training.
- Binary-trained DSpark BF16 checkpoint: `~/data/bonsai1/Bonsai-27B-dspark-bf16.gguf`,7,291,885,792B, SHA256 `93bd7f0326bcb39982051dd7742e5a2c9031358f94293a9c91f6aa98dcb6573b`, HF pin `f10afb355f104535e3e3e98cf7ab7795c72bd292`. Fetch with `bash tools/binary/fetch-dspark.sh DATA_DIR bf16` (or separately pinnedQ4_1). Six layers, own embedding/head, five taps1/16/31/46/61, KV4, block4 and SNR/Markov/correction machinery; native port in progress. `gguf-header.py` reads metadata/tensors without needing the entire payload in memory.
- Drafter vocabulary `~/work/tune-5090/spec-ideas/vocab-65536.txt`. Restricts drafter head, not target's final greedy vocabulary.
- WikiText corpus `~/data/engine-shootout/corpus/wiki.test.raw`, SHA256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. First128tokens prefix, next256 scored, positions reset.

Fetch target: `bash tools/binary/fetch.sh MODEL_DATA_DIR`; hash is mandatory. No silent substitution of a changed upstream artifact.
