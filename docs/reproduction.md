# What has been reproduced from this repository

Retained reproduction evidence for the initial portable engine export, including its recovered Linux
build completion. Each claim is listed with where it was checked; this is not a new build of current main.

## Clean-checkout run on an RTX 5090

A fresh copy of this repository (no prior build tree) was configured and built on an RTX 5090
workstation: Windows 11, MSVC 14.44, CUDA 13.0.2, CMake 4.1.2, Ninja 1.13.1, driver 610.47, 600 W.
`cmake --build build --parallel 2` finished all 228 steps in 706 s. Then:

| Command | Result | Receipt |
|---|---|---|
| `scripts/bench.py --profile serving` | **6,731.7 tok/s** (passes 6,731.7 / 6,733.0 / 6,730.8) | [`receipts/clean-checkout/serving-b64x128.json`](../receipts/clean-checkout/serving-b64x128.json) |
| `scripts/bench.py --profile deterministic --check-determinism` | **6,550.5** and **6,549.7 tok/s** in two runs; generated tokens identical across runs | [`receipts/clean-checkout/deterministic-b64x128.json`](../receipts/clean-checkout/deterministic-b64x128.json) |

Same GPU, same model hash, same prompt fixture, same frozen plan (hash unchanged after the run) and the
same runtime settings as the headline measurement.

**Gap to the headline.** The binaries from the clean build run 0.67% slower than the measured ones
(6,731.7 vs 6,777.4), and the deterministic build runs 0.50% slower (6,550.5 vs 6,583.1). The headline
binaries came from an incremental build tree that relinked objects compiled earlier from the same
source, so their hashes differ from a fresh build's. The tokenizer libraries also came from a different
llama.cpp build. That gap has not been decomposed. Treat ~6,730 tok/s as what this repository
reproduces today, and 6,777 tok/s as the original measurement.

## Linux compile and link

The retained Linux x86_64 Release build completed with **GCC 15.2.0 and CUDA 13.1**.
The recovered final log finishes the remaining CUDA objects and both links:
`bin/libquality_custom.so` and `bin/quality_length_bench`. Both ELF outputs were observed present.
This is **compile-only evidence**; no Linux GPU execution was performed.
[Build receipt](../receipts/clean-checkout/linux-build.json) and
[sanitized final log](../receipts/clean-checkout/linux-build.log) retain the result and output hashes.

## CPU-side checks

- **GSM8K requests:** for all 1,319 items, the request `scripts/gsm8k.py` sends (prompt bytes, sampling,
  seed, stop strings) is byte-identical to the request recorded in the measured run.
- **GSM8K score:** `scripts/gsm8k.py --score-only` on the measured outputs gives 1276/1319. It agrees
  item by item with the original EvalScope verdicts.
- **Fixture and plan:** the prompt fixture and the frozen plan are byte-identical to the measured
  inputs. `bench.py` checks both hashes before every run.
- **Model:** the pinned Hugging Face revision serves the measured GGUF (5,946,648,928 bytes, matching
  sha256).

## Not yet reproduced from this repository

- **The GSM8K run itself.** The 1276/1319 score comes from the original campaign using the same engine
  binary as the headline. A full rerun through `engine/serve.py` + `scripts/gsm8k.py` from the clean
  build has not been done. Temperature-0.7 sampling won't reproduce identical text across runs.
- **Linux GPU run.** Compilation and both links completed as recorded above; Linux GPU execution
  remains untested.
- **The ladder rows.** These are historical measurements on a rented Linux RTX 5090 that has since been
  destroyed. Only the final engine is buildable from this repository; [`steps/`](steps/) holds code views,
  not buildable intermediate states.
- **`engine/serve.py` from the clean build.** It is the same server used for the GSM8K run, with only
  path handling changed, but it hasn't been started from this checkout.
