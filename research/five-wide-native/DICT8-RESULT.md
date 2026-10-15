# Dict8v2 — exact numerical closure, whole-engine performance rejection

Private source `3fdbad4547e51455b82d5e7e3f5d6194081449cb`, DLL `4098ba03c9d8b53e0672e9f6800d1461c453bce46e80923c46e389edbc415dfa` ran once after FULL MATH500 completion and source-bound START. **No adoption. Production is unchanged.**

## Matched whole-model result

B64 ×128, 8 warmup tokens per stream, 3 passes per arm; ordinary and deterministic measured independently in the same bounded window. These fresh controls replace no historical evidence.

| Profile | Serving median tok/s | Dict8 median tok/s | Delta |
|---|---:|---:|---:|
| Ordinary |6764.610944|4236.037906|−37.379430%|
| Deterministic |6549.839344|4094.571609|−37.485923%|

Ordinary candidate samples:4236.037906/4238.149001/4235.882829. Deterministic:4102.205817/4094.116911/4094.571609. Timing retains the complete engine and observer; diagnostic endpoint checks are disabled. Ordinary generations are nondeterministic and do not establish token identity. Deterministic timing-token logs are preserved.

## Numerical contract

Unchanged frozen reference, independent stable repeat, then candidate: all pass. Exact initial/prefill and teacher-forced inputs/tokens, three warmups then two cached observations, all64×248320 finite binary32 logits per observation. Observed graph hits2/captures0; warmup hits0/captures1. Shared logit SHA256 `47b9805a60b141d7d1ada84fe1bf5645605b72de32e4a96a01d5ac2751a9a777`.

Candidate source-bound native-WF0 comparator checks separate rounded gate/up after activation scales and row means, SiLU, then unchanged sign/Hadamard1024/FP4 q4/s4/g4 observer. Warmup and observation counters are separate:

| Endpoint | Warmup checked | Observed checked | Mismatches |
|---|---:|---:|---:|
| gate/up words |427819008|285212672|0|
| SiLU words |213909504|142606336|0|
| q4 bytes |106954752|71303168|0|
| s4 bytes |13369344|8912896|0|
| g4 words |12288|8192|0|

All nonfinite counters zero. This closes the saved five-decode corpus, not every possible input or CUDA formal equivalence. Scoped Lean layout/opaque-replay statements remain distinct from empirical machine conformance.

## Setup and residency

The startup-order repair works natively: dictionary upload follows successful nonblocking stream creation; no null/default-stream masking. Candidate engine creation24.6293162s versus frozen9.6528904/9.3372661s. Preparation:13.6555725s CPU encoding,0.5616332s upload.

64 prepared layers:2317352960B plus26244B dictionary. Original GU2852126720B, scales178257920B and means8912896B remain resident. Added requested allocation2317379204B; total GU5356676740B. Runtime `total_weight_bytes=9425849988` covers all model weights, a wider scope than the GU-only invoice. Strict startup reported GPU free16.74/34.19GB versus serving19.19/34.19GB; timing candidate16.77GB free. This establishes fit for these runs, not a portable physical allocator peak or standalone allocation-granularity measurement.

The bit-exact18.75% packing survives this experiment. The measured loss rejects this consumer implementation. The [completed private single-scan offline-cache successor](dict8-identity-result/README.md), source `9387d08ae`, consumes immutable prepared bytes with no startup encode fallback. Its 2,317,355,136-byte real-model cache passes every original operand-word comparison and all64 payload checks; CPU/native compilation and failure tests pass. One lifetime-bound mapped-model hash supplies Engine and Reader identity. the RTX 5090 host local-NTFS admission retains the deny-write handle through unmap; Linux uses a protected private snapshot, with its full copy/allocation separately priced. Full host payload preload, read/hash and retained WF0 preparation remain recurring costs. The [two-scan predecessor](dict8-cache-result/README.md) remains in immutable custody. Source-bound timeline attribution and the joint ISA search are separate. Cached native startup and a faster consumer both require their own eventual measurements. Logical lookup bytes, static instruction counts and modeled bank geometry remain distinct from measured DRAM, cache misses, latency and occupancy. No causal hardware-counter attribution is established.

## One-run lifecycle and transport recovery

Authorization `00000000-0000-0000-0000-000000000000`; START delivery `00000000-0000-0000-0000-000000000000`. Original Bash job `00000000-0000-0000-0000-000000000000` acquired at1792035619.5985982, finished1792035771.9230685:152.3244703s within240s. All7 native children returned0, longest34.0978651s within50s. ReleaseMutex1/CloseHandle1; serving DLL/production plan exact. Explicit handback to8614 operation `00000000-0000-0000-0000-000000000000`; Independent physical check1792035852.2813988 confirms children absent, ports18461/18464 closed, mutex acquired/released and full model/serving DLL/benchmark/production plan exact; [receipt](dict8-engine-physical-handback-v2.json).

The local SSH job exited143 at its55s shell timeout. Native execution continued to completion. Read-only job `00000000-0000-0000-0000-000000000000` recovered the original final receipt; **no reset or replay**. The local failure remains recorded. Future multi-minute native drivers must use durable native execution plus separate bounded readback, or a sufficiently bounded detached transport; a55s transport timeout is not an adequate lifetime for a240s native operation.

## Custody

- [Original native result](dict8-engine-window-v2-result.json), all7 child receipts, exact environment/sample arrays, counters and setup.
- [Readback including stdout/stderr prefixes](dict8-engine-window-v2-readback.log).
- [Consumed request and original job identity](dict8-engine-run-request-v2.json); [exact handback](dict8-engine-handback-v2.json).
- [Immutable source/build/CPU lifecycle checkpoint](dict8-engine-result-v2/README.md), source bundle and native build ZIP linked there. Released writer is not needed.
- [Lossless runtime archive receipt](dict8-engine-window-v2-custody-receipt.json) binds all strict/timing runtime files and deduplicated full logits when exported.

V1 remains a separate consumed startup failure, losslessly preserved. V2 used a distinct authorization and attempt. No active GPU child, held mutex, queued successor, rental, download or production mutation belongs to this lane.
