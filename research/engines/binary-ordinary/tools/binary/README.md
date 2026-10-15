# Binary Bonsai 27B on an RTX 5090

Private campaign, October 10, 2026. Source owner: the local `halo-5090` repository's `binary-bonsai-5090` branch. Nothing from this campaign is pushed to a public remote.

The target is the largest published binary Bonsai: `prism-ml/Bonsai-27B-gguf`, original Q1_0_g128 signs and FP16 K128 scales. It is a different checkpoint from Ternary Bonsai 2 27B. Its qwen35 geometry matches Halo's engine, but binary weights have no Hadamard folding, and its alpha/beta matrices are binary too.

Campaign results and provider teardown receipts belong in `results/binary-5090/`; the operator's presentation index is `/PRIVATE_PRESENTATION_NOT_EXPORTED`, section **Binary**. The ordinary ladder is measured; a modern speculative port and ISA-map experiment are underway in managed writer `~/work/clones/binary-spec-5090`, branch `binary-spec-bonsai-5090` based on `tune/main@4a40eb1`. Final teardown remains pending.

## Ordinary result

Same-card real-token cached decode: reference195.79, native INT838.56, old2bit FP4single165.91, exact packed1 FP32activation200.01, packed1/F16state203.62, true1bit FP4single202.00, two-plane FP4116.70, two-plane/F16state117.20 tok/s. Packed1/F16state repeats203.30–203.61 against fresh reference195.29–195.95 (+3.75–4.11%). Wiki256 reference PPL19.1780; selected packed1/F16state19.1952 (+0.0893%), KL.0001502, top1252/256; fixed chat64/64 top1. SingleFP4 loses quality (PPL19.6712); two-plane restores it (19.1706 withF16state) but loses speed. These are alternatives, not additive gains. Published original binary27B RTX5090 rate162.6 uses its own protocol; H100104.8 is not the same-card comparison.

The direct packed dynamic quartet LUT is a known algorithm family implemented here with exact original scales; sign-byte-to-FP4 operand wiring is our implementation of vendor blockscaled MMA. GPU load-only transformation `4660136` matched every prepared byte and FP16scale across12 actual-weight layout cases; standalone fixture `tools/binary/repack/check.cu`, receipt `results/binary-5090/repack-check.log`. Negative LUT8/ILP round and initial inclusive kernel timing/error records belong to [packed projection research](../../experiments/binary-packed/README.md) and its [negative round](../../experiments/binary-packed/TUNING.md).

Reference task control: fresh32GSM29/32 withone4096-token cap; first32HE+32/32. Both greedy, thinkingoff, ctx8192, no speculation. Total21304 emittedtokens/127.576 generation seconds=167.0 HTTP/client-wall tok/s across64requests. This is a scoped task panel, not the full quality gate and not the ~196 teacher-forced protocol. `results/binary-5090/tasks-reference/` retains requests/responses/scores/source hashes.

## Inputs

- Language GGUF: `Bonsai-27B-Q1_0.gguf`, 3,803,452,480 bytes, SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Reference: PrismML llama.cpp source `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`; this host holds it in `~/work/clones/bonsai-hip`. Only its tracked vocab fixtures are absent; inference source is unmodified.
- Perplexity panel: first 128 tokenizer tokens of WikiText-2 test as prefix, following 256 tokens scored, positions reset. Corpus `~/data/engine-shootout/corpus/wiki.test.raw`, SHA256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
- Chat panel: one fixed rainbow explanation, 64 reference-greedy output tokens. Teacher forcing means each candidate sees the same reference continuation, not its own generated history.

Fetch the public model into a chosen directory with `bash tools/binary/fetch.sh MODEL_DATA_DIR`. Its byte hash is mandatory; a newer upstream file is not silently substituted. To recover reference source, clone `https://github.com/PrismML-Eng/llama.cpp.git` and checkout the exact commit above.
