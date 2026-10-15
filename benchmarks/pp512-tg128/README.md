# Single-stream PP512 / TG128 — Ternary Bonsai 2 27B, one RTX 5090

Measured 2026-10-15 with this repository's native engine, using llama-bench's PP512/TG128 workload
definitions as the protocol. The engine was **not** run through llama-bench, so the numbers come from a
different harness. They have not been submitted to any external leaderboard.

## Result (quiescent repeat, 22:40:53–22:41:13 UTC)

| Metric | Five-repeat mean ± sample SD | PrismML-community RTX 5090, PTQ1_0 | Ratio |
|---|---:|---:|---:|
| PP512 | **8,986.8 ± 88.1 tok/s** | 4,116.6 | 2.18× |
| TG128 (random-fed, depth 0) | **135.8 ± 2.3 tok/s** | 133.0 | 1.02× |

- **TG128 versus the other published rows:** 2.1% faster than the published PTQ1_0 row and 5.0% slower than
  the published PQ2_0 row (143.0 tok/s).
- **Same input file:** the published PTQ1_0 row used the same model file,
  `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes, sha256 `53107f53…33ee3`.
- **Arithmetic and settings differ:**
  - Our engine runs FP4 weights and activations, Q8 recurrent state with FP32 update arithmetic, and FP8 KV,
    with 256-row prefill microbatches.
  - The published llama.cpp fork uses its own PTQ arithmetic, F16 KV, batch 2048 and microbatch 512.
  - The hardware and OS also differ: our run used Windows 11 with driver 610.88 and CUDA 13.3; the published
    row used Linux, an Intel 265K, driver 595.91 and CUDA 12.8.
  - The input files are identical; the computation is not.

Raw sample timings, the reducer result and full engine environment: [quiescent-2241/](quiescent-2241/).

## Earlier repeat with CPU compile running

[compile-overlapped-2221/](compile-overlapped-2221/) is a five-repeat run from 22:21 UTC.

- **Result:** PP512 **9,256.95 ± 229.20**, TG128 **130.25 ± 3.87** tok/s.
- **Background load:** two CPU-only compiler processes, unrelated to the measurement, overlapped the whole
  window. The GPU had no other compute process.
- **Status:** kept as its own row and not averaged with the quiescent repeat. The quiescent repeat is the
  headline result.

The first urgent run is in [first-pp-only-2219/](first-pp-only-2219/). It took a single PP512 sample of
9,087.3 tok/s, with random-ID generation outside the timed region.

## Method

- **Single stream:** one slot, no speculative decoding (`H5_NO_SPEC=1`), speculative depth 0, no batching
  acceleration.
- **Fresh state per sample:** the engine is reset before each sample.
- **PP512:** 512 random prompt token IDs, with random-ID generation inside the clock.
- **TG128:** 128 random-fed decode steps, with the RNG and an explicit per-token device sync inside the clock.
  No 512-token prompt precedes TG128.
- **Random tokens:** vocabulary 248,320, no BOS. The token generator matches glibc `rand()` with seed 1.
- **Excluded from timing:** load, pre-tuning and warm-up.
- **Included in timing:** Python/FFI and host-copy overhead.
- **Statistics:** five samples per arm, reported as the arithmetic mean of the per-sample rates and the sample
  standard deviation. Raw nanosecond timings are retained.
- **Context:** configured context is 163,840.
- **Command:** `python bench-final.py native --output final-five.jsonl --arm all --repetitions 5`.
  [bench-final.py](bench-final.py) is the harness.

## Redaction

Local filesystem paths and the name of a host-local GPU lock were replaced with placeholders (`<WORKSPACE>`,
`<GPU_LOCK>`, `<HOST>`). Measurements, settings and hashes are otherwise unchanged.
[redaction-receipt.json](redaction-receipt.json) lists the sha256 of each published file and of its private
original. Embedded `source_sha256` values, such as `7d052e49…`, refer to the unredacted harness.
