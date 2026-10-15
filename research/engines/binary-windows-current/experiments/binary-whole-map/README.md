# Binary Bonsai whole-map search

This experiment searches the **complete observed FFN map**, not just adjacent kernel launches. Its start is the packed original A8 input; its end is the original scaled down output and declared continuation. The programme12 [observer-search plan](canon/program-search/observer-search/PLAN.md) owns sufficiency, closure and cost. SiLU is part of the finite map, not a reason to reject a search.

## Findings and controls

[PROGRAMME12.md](PROGRAMME12.md) is the self-contained case report: real activation fibers, whole-model lossy carrier probes, complete small-map witnesses, trained weight structure, exact three-plane A8, whole-FFN native continuations, and whole-model disposition.

[WHOLE-FFN.md](WHOLE-FFN.md) owns the complete direct-carrier path, eliminating global GU/hidden/q-byte tensors. R32 complete cold98.048µs versus best four-plane120.832 and fusedDP4A155.392 in that A8 control window; R16 is not universally faster than DP4A, and R5 is rejected. [Final joint control](whole-ffn-results/joint-README.md) retains a different two-plane NVFP4 numerical point: R16 joint73.696µs cold beats paired A8 compound100.352; R32 ordinary wide two-pass112.384 beats paired compound126.944. No absolute cross-precision frontier is claimed. Actual R32 fixed work improves12.636%, steadyGSM256 HTTP3.9469% with239 vs240 correct; quick64 HTTP loses3.56% and one GSM answer. Extra1.494140625GiB GU fallback residency is explicit. No default promotion follows.

* [SIGNED-DIGITS.md](SIGNED-DIGITS.md): every full signed byte has a table-free three-FP4-plane coordinate. Six instead of eight MMA instructions/K128; original scales and quantization unchanged. Same-plan outputs bit-exact; best inclusive down13–14%, GU6–8% faster. Engine opt-in `H5_A8_PROJECT=fp4-three-digits-ragged` is integrated privately, not a new default.
* [EXACT.md](EXACT.md): complete64-state gated/down maps, structured one-dot+table positives,3,375-carrier collision witnesses, prepared SF/lane-wired positives, and a rejected additive endpoint-table lift. These bounded witnesses neither prove general impossibility nor trained-model speed.
* [README-weight-census.md](README-weight-census.md): actual signs/scales, exact local class carriers, and two native schedules that lose to DP4A. Lossless11-bit half-scale coordinates are a storage result, not measured throughput.
* `block_rank.py`, `block-rank-layer{0,30,61}.json`: all136 down K128 blocks/layer have rank128, witnessed by selected original128×128 sign submatrices with nonzero determinant modulo3. Nonzero original half row scales preserve real rank. This rejects local linear annihilation with fixed other blocks/scales, not native-rounded or nonlocal whole-map carriers.
* `ThreePlanes.lean` and `signed-digit-results/exact_a8.lean`: finite exact digit identity and arbitrary observer transport; `exact_search.lean`: factorization/rectangle witnesses. No custom axioms/sorry. Native evaluation is checked by the CUDA harness, not a real-arithmetic substitution.

Further exact-coordinate searches: [row_fibers.md](row_fibers.md) finds only same-prefix packet repetition in real mixed prompt batches, not post-divergence collisions; `sparse_third.py` and `sparse-third.json` expose an exact balanced carrier whose third digit is nonzero in5.7–8.7% of captured hidden entries. K128 and tested K64×rowtile8/16/32 zero-plane skipping have no hits. [SPARSE-DIGITS.md](SPARSE-DIGITS.md) prices its exact sparse integer continuation on real captured hidden inputs:188.416/345.856µs coldR16/R32 versus dense3 30.432/45.024. All native witnesses pass; the schedule loses and is not integrated.

## Real activation capture

`bench/whole_map_capture.cpp` links diagnostic `--wrap` hooks against the unchanged control engine. It rejects unsupported graphs/fused/interleaved/flagged observer routes and counts all64 FFN calls per step. True postnorm F32 input, its original A8 q/sx, joined GU and native post-SiLU product are captured for layers0/30/61. F16 recurrent state **and F16 KV** are explicit. `f16-activation-layer*.json` are the corrected current-route reports; earlier `activation-layer*.json` retain the separately labelled FP8-KV pilot.

```
bash experiments/binary-whole-map/build_engine_probes.sh \
  /workspace/binary/digits-ragged/halo /workspace/binary/whole-map /workspace/binary/llama
# Every GPU invocation runs under the parent's /tmp/binary-gpu.lock and timeout55.
/workspace/binary/whole-map/capture MODEL WIKI 128 32 OUTPUT capture PRISM_REF
python3 experiments/binary-whole-map/analyze_capture.py \
  --model MODEL --capture OUTPUT.capture --layer 0 --kv f16 --out REPORT.json
# Repeat separate bounded calls for layers30,61; no lossy grouping is called an exact fiber.
```

For whole-model hidden diagnostics use `128 256 OUTPUT levels:127 PRISM_REF`, levels:63/31/255, baseline, or drop:0.01. The wrapper changes hidden production at **all64 FFNs and every prefix/scored step**; levels255 matches the original producer. Full trained down receiver uses the complete5120 outputs. Dynamic K128 scales/status remain in the packet. Sample entropy is not a population cardinality or general minimum-bit assertion.

## Whole-engine pricing and quality

`engine_adapter.cu` maps only four-plane bytes/configure/encode/project to exact three-plane counterparts, preserving original four-plane Plan enums, geometry and split policy. Old source374c77d/control engine object SHA5b29212cb2ecab3ef748f4c5e7220b2e047bb88cbc8417f5c4e77f242baae342; new source anchor7961268. `engine_fixed_work.cpp` demands every head, token198 on16/32 independent slots, positions0–63, context allocation512; first32 warm, last32 timed. Fresh-process ABBA order is retained by `run_engine_gate.sh` and every per-step time survives in `engine-results/`.

Fixed work mean-median throughput rises **1.5775% R16 / .7602% R32**, not the projection speed ratio. Wiki256 prefill/decode printed metrics are identical. Parent actual-native resource-fixed sourceb328bf6 separately gives **−1.15%** on generated HTTP GSM32/HE32, with identical29/32 and32/32 outcomes/caps1/0; parent receipts own that serving result. No default promotion follows a microkernel win.

`engine-results/summary.json` records ordering/raw timing. `engine-results/{adapter-build,fixed-build,build-f16}.log` retain object/binary hashes. `engine-results/data-custody.json` locates and hashes large captures and measured binaries at `~/data/binary-5090/whole-map/`, bounded below256MiB. Those data are not source-Git blobs. The measured binaries `capture`, `three-h5`, `fixed-four`, `fixed-three` were exported before teardown and losslessly retained in `measured-binaries.tar.zst`, restoring and checking original hashes before removing uncompressed duplicates. Targeted objects remain in `bin/`; full17,616-row input-fiber packets are losslessly retained in `row-fibers/packets.bin.zst`. Runtime llama libraries belong to the campaign owner. Do not confuse a later recompiled control with these pinned receipts.

Parent owns native engine integration, final model task panels, shared rental55260842 lifecycle, existing writer heartbeat and private-local push. No new rental, public push, recurring wake or change to the deployment default belongs to this experiment.
