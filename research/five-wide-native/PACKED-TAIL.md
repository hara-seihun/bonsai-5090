# Packed GU terminal consumer

Private local source: `halo-5090`, branch `agent/fullk-fp4-tail`; current source commit `d7926585e`. Managed checkout `~/work/clones/fullk-fp4-tail` is working custody, not a permanent artifact path.

## Contract

On the actual model's64-row path, retain row-mean-scaled gate/up coordinates as a joint exact8-byte label per feature, consume that label directly through the existing SiLU, explicit sign, Hadamard1024, fixed-global-scale FP4 prep arithmetic, and produce the actual q4/s4/g4 layout. Mixed invertible labels and raw paired-coordinate controls are separate modes (`tail-mixed`, `tail-plain`). Other row shapes and unset serving remain unchanged.

This first carrier doubles global traffic relative to a4-byte SiLU temporary:8 bytes written and8 read per feature versus4+4. It establishes the complete observer boundary before shrinking the carrier. There is no speedup claim from label compression alone. `PackedTail.lean` proves exact paired32-bit label inverse/injectivity and arbitrary terminal-observer composition; it does not prove CUDA, F32 arithmetic or FP4 layout conformance.

## Current observations

- Initial native compilation failed on missing local SiLU helper; source repaired in `c656d634a`.
- Named v2 DLL built6.185s, SHA256 `9a5a2034b59bcf1576407b3a40e9dd083b1e6e546bf5a6ba93f8541fe8c9d0104`.
- First v2 native boundary exited1 before its first64-row producer launch. Packed dispatch mistakenly accepted configuration7 only (FW4/WG2/TOK64). Frozen serving chooses configuration13 (FW8/WG2/TOK32). This is a setup refusal, not mathematical evidence or a conformance result. `fp4-tail-v2-boundary-tail-mixed.log` retains the failure and observed `MUTEX RELEASED` in `finally`; no timing run followed.
- Source `d7926585e` repairs the producer to the actual frozen configuration13, including its two32-row CTAs and288-thread block. Named v3 CPU build succeeded15.998s; DLL SHA256 `3572a8e6aecd89bf96b26b755440ac1e6b5ded7a49b602fe710ff8279f2e3512`. `fp4-tail-v3-build-receipt.json` binds sources, inherited objects and fixed-serving identities; full compiler output is `build-fp4-tail-build-v4.log`.
- Both check-stat arrays initialize to zero through the existing `dalloc` stream-ordered `cudaMemsetAsync`; no additional initialization patch was needed.
- Fresh unchanged-serving B64×128 control:6778.428544,6777.352952,6775.657263tok/s; median6777.352952 (`VIDEO-SERVING-TPS.md` and its command/environment/sample receipts). Both v3 controls passed71,303,168 SiLU F32 words,35,651,584 q4 bytes,4,456,448 s4 bytes and4096 g4 F32 words each, with zero mismatches.3-repeat mixed TPS3970.455776/3973.086607/3977.466913, median3973.086607; plain4003.496021/4012.525034/4014.598987, median4012.525034. Losses against fresh serving are41.38% mixed and40.80% plain. No adoption. Whole-model logits acceptance independently remains unset because unchanged-serving process-repeat controls drift.

## Operations and custody

Only `~/bonsai/quality-campaign/gpu_host.py` operates the RTX 5090 host. Native directory is `C:/work/models/experiments/bonsai-quality-20261013/fullk-fp4-tail-v3`; no serving files, production plans, services or models are modified. Local `run-fp4-tail-v3-boundary-tail-{mixed,plain}.py` and `run-fp4-tail-v3-timing-tail-{mixed,plain}.py` acquire `Local\\BonsaiGPU` atomically and release in `finally`. Coordinate the slot explicitly with worker8614 and manager9769; a free mutex does not preempt an offered or assigned window. Total offered closure is at most4 minutes, native subprocesses at most55 seconds.

Boundary success requires positive compared counts, zero SiLU-word mismatches, and zero q4-byte/s4-byte/g4-word mismatches against the serving consumer on the same per-layer inputs. Only then run3-repeat B64×128-token model timing under the original manifest environment and compare a fresh unchanged-serving receipt. Inputs, DLL, plan and exact command/environment must stay bound.

Retain every failed named generation. Build v1 Python syntax failure (`build-fp4-tail-v1.log`) made no build effect; native missing-helper build v2 (`build-fp4-tail-build-v2.log`) left compiled engine/ring objects later hash-bound into the successful build. `build-fp4-tail-build-v3.py` owns named v2; `build-fp4-tail-build-v4.py` owns named v3. They return existing completed receipts rather than silently overwriting generations.

`prepare-fp4-tail.py` is already applied and is not idempotent. `prepare-fp4-h1-tail.py` is an unapplied proposal: move sign and the actual feature-bit0 Hadamard stage into producer, encode two transformed features in8 bytes, resume consumer at stage2. Its original FW4/TOK64 configuration needs correction before application. No empirical H1 result exists.

## Selected-configuration resource invoice and retention

For actual configuration13 FW8/WG2/TOK32, packed producer uses142 registers mixed,131 plain,288 threads, zero spills. The wide terminal consumer uses48 registers mixed,54 plain,4096B shared and zero spills. These are selected-configuration observations; prior FW4/TOK64 resources do not describe configuration13.

All4 native v3 processes exited0. The final plain timing process released the mutex and preserved serving DLL/plan hashes; explicit release was sent to9769 and8614. No queued GPU variants remain.17 losslessly retained native files live in `retained-engine-v2/fullk-fp4-tail-v3/`, with SHA256 inventory `retained-files.json`; `fp4-tail-v3-result.json` is the reduced result. Private source is pushed to `~/projects/halo-5090`, branch `agent/fullk-fp4-tail`; `fullk-packed-tail-v3.bundle` is independent stable source custody, SHA256 `c4bb515593c5f8e8b85c809aaa41d229b5deb2b778956cc7a49267ed44344ebc`.

[H1 CPU-ready sibling](H1-TAIL.md) owns the actual selected-config sign/first-butterfly8B/two-feature map, complete matched strict/ordinary profile harness, private ordered-clear repair and CPU compiler invoice. No H1 GPU evidence exists yet.
