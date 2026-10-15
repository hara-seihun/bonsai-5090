# Binary, packed and five-wide inference research

Source-only research snapshots, October 15, 2026. These are independent variants, **not a combined engine or promoted defaults**. The portable engine at repository root is the runnable starting point. [Build guide](BUILD.md) explains how to inspect/build each research family without the original private repository.

## Index

| Source | What it contains | Current result |
|---|---|---|
| [Current Windows binary](engines/binary-windows-current/) | `c7d236b4`; complete binary engine including joint-five, B32 IMMA and persistent kernel sources | Root portable engine derives from this line; inclusion confers no new performance acceptance |
| [Ordinary binary](engines/binary-ordinary/), [modern F32 control](engines/binary-f32-control/) | Original Q1 signs/scales, packed/FP4 projection ladder, distinct F32 arithmetic control | Distinct numerical controls; do not substitute a ternary model |
| [Binary speculative](engines/binary-spec/) | DSpark BF16/NVFP4 precision, exact A8 digits, whole FFN, graph/resource experiments and numerical/task summaries | Several finite/fixed-work wins; workload-dependent macro outcomes, including losses |
| [Batch-pair](engines/binary-batchpair/), [DSpark live](engines/binary-dspark-live/), [Bmega](engines/binary-bmega/) | Independent batch-axis IMMA integration, repaired live attention, one-kernel-per-token alternative | Source/build or measured scopes remain separate; no merged multiplier |
| [Full-K GU](engines/fullk-gu/), [packed terminal tail](engines/packed-tail/), [H1 graph-v3](engines/h1-graph-v3/) | Actual ternary-engine integrations, rounded gate/up carriers, SiLU/sign/Hadamard1024/FP4 continuation and strict graph harness | Full-K engine ~39.8% slower; packed tail ~40.8–41.4% slower; H1 strict logits/endpoints pass but whole-model TPS ~40–41% slower |
| [Dict8 v2](engines/dict8-v2/) | Eight-trit/13-bit dictionary, actual B64/config13/split1 engine integration and startup-order repair | Strict full logits/endpoints pass; whole-model TPS loses ~37.38% ordinary/~37.49% deterministic. No adoption |
| [Dict8 offline identity](engines/dict8-offline-identity/) | Latest opt-in Engine + Reader single model identity, offline encoder/cache producer, typed failures, Linux snapshot/Windows retained-NTFS mapping and tests | Full CPU 1,426,063,360 words/64 payloads match. Cache 2,317,355,136 bytes. Cached GPU startup/residency/TPS **unset** |
| [Dict8 joint consumer](engines/dict8-joint-consumer/) | Shared8 compact/dual, split5+3 and lookup-free six-LUT3 encodings, complete generated consumers, CPU and Lean source | Shared8 1,664-byte pair (18.75% below 2-bit), 242 K-loop sites/126 registers/88,896 dynamic shared bytes under Linux CUDA13.1. Literal3 1,728 bytes/361 sites/121 registers. Native endpoints/occupancy/TPS **unset** |
| [Five-wide native](five-wide-native/README.md) | Standalone CRT/affine/full-K/RZ/cooperative kernels, finite Lean contracts, synthetic and actual-model receipts/failures | Finite composition established; ordinary FP4 remains stronger on full-K complete-route timing |
| [FP4/TF32 chains](fp4-chain/README.md) | Uniform-K native chains, nonuniform-K collision/readouts, tile/affine/K16/K64 variants, CPU searches and Lean contracts | Native constructions and bounded negatives; every priced latest candidate loses to its strongest simultaneous control |
| [Packing Round23](packing-round23/README.md), [inherited Round22](packing-round22/) | Serialized q-group packed readers, complete source/proof/invoice dependencies | Shorter lexical weight lifetime, 255 registers/28 B spill diagnostics vs FW8 131/zero. No measured GPU timing/nominee |

## Read claims at their boundary

- **Representation:** Dict8 saves recurring prepared weight payload bytes, while retaining original weights/scales. It is extra prepared memory, not a net model-memory saving. Offline caching removes repeated encoding; it still pays model identity, payload allocation/read/hash and the retained original WF0 work.
- **Finite native:** all `243 × 15^5 = 184,528,125` mixed CRT checks pass. This is a common-scale five-term map; serial/raw successors lose to stronger packed controls. The finite radix128 two-group route is 5.225600 µs vs 5.440001 µs strongest folded control, separate from full K or an engine.
- **Full K:** rolling 8-byte state matches the declared synthetic K5120/B64/GU observer, including source scales/RZ boundaries and normalization. About 776 µs beats grouped SIMT (~1037 µs) and loses to ordinary FP4 (~450 µs). No B64 engine win follows.
- **Actual model:** H1 and Dict8 strict full-logit results require stable unchanged-serving repeats and deterministic fold controls. Earlier failed reference/graph/startup windows stay failures. Original packed-tail DLL full logits were not retroactively certified by H1's sibling build.
- **Formal:** Lean files prove their declared finite algebra/continuation contracts. They do not certify CUDA compilation, native floating-point semantics, real-model quality or throughput.
- **Binary task panels:** whole-FFN selected steady GSM256 B32 improves HTTP 1237.31→1286.15 tok/s, correct 240→239/256; quick64 loses 615.21→593.33 tok/s. DSpark A1 quick64 299.09→320.67 tok/s retains item outcomes; generated text/volume differs. These are separate workloads, not multiplicative or fixed-work gains. Source experiment docs and `results/binary-5090/` retain scope and controls.

## Provenance and omissions

[SOURCE-REFS.json](SOURCE-REFS.json) gives complete branch commit IDs. [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json) maps each retained file to its source and both original/export SHA256. Only host/account locators and private presentation references are sanitized; original hashes remain evidence, not claims that edited bytes match them. Generated source packs supplement their Git snapshots. No source history is exported.

[SCAN.md](SCAN.md) documents publication filtering and preliminary scanner classifications. [EXPORT-EXCLUSIONS.json](EXPORT-EXCLUSIONS.json) records filtered extraction paths; broad excluded classes include weights, model caches, real weight-row fixtures, activation/logit dumps, finite binary dictionaries, executables/objects/cubins, dependency trees, compiler expansions, profiling traces and private transport/provider state. Synthetic dictionaries can be regenerated by the retained source. Hash references to omitted artifacts are intentional.

Historical source READMEs and receipts retain their original point-in-time statuses and paths. This index states the latest combined observation; it does not grant hardware execution authority. **No GPU/paid run, source-campaign mutation, default promotion or build was performed for this export.**
