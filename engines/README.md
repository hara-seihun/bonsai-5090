# Additional engines

These are separate source exports, **not interchangeable presets of `../engine/`**. The root engine remains the portable B64 RTX 5090 reproduction. Nothing here changes its benchmark claims.

| Export | Original ref and exact exported commit | Target/model |
|---|---|---|
| [halo-5090](halo-5090/README.md) | `master` / `772d9060ec5b5eba5b7429833331843b93eca93e` | RTX 5090, CUDA, Ternary Bonsai 2 27B; native DFlash2, HTTP and image input |
| [bonsai-halo](bonsai-halo/README.md) | `master` / `1b71c6798d1d210e969d59a1513edf9c79b18447` | Radeon 8060S `gfx1151`, HIP, Ternary Bonsai 2 27B |
| [halo-6000](halo-6000/README.md) | `main` / `fec60de800f9a3a5a236a65270b79426f5e4fa5e` | RTX PRO 6000 Blackwell, CUDA, ternary Qwen3.6-35B-A3B MoE |

`SOURCE.json` in each directory records original-file SHA256 and exported SHA256, including portability edits. Historical measurements are not remeasurements of these export commits. Their own recorded source/executable identity remains authoritative. Results from different models, hardware, task schedules, contexts, precision policies or timing boundaries establish no general speedup.

## What's included

Native source and Makefiles, component probes, local benchmark/evaluation harnesses, synthetic image fixtures, compact raw timing/quality receipts and experimental tuner source. All three use externally built PrismML llama.cpp libraries for vocabulary/chat templates; the optional 5090 image encoder also uses `libmtmd` and ggml. Language execution is native. See [DEPENDENCIES.md](DEPENDENCIES.md) and [NOTICE](NOTICE).

## Deliberate exclusions and gaps

No Git history, weights, prepared weight caches, activations, full logits, binaries, generated dependencies, rental/service-management scripts or host operating manuals. Upstream long-form READMEs are replaced by these portable guides. No GPU run or build was performed for this export. Benchmark output strings are generated responses to synthetic/public benchmark prompts.

The HIP tree had an uncommitted `tools/shootout_runner.py` edit; the export uses only committed files and excludes that host runner. The MoE model/cache and frozen intrinsic dataset acquisition pipeline are external prerequisites, not bundled assets. Historical caches and compiler environments are not recreated by this export. Exact old engine commits are identified where retained; their entire historical trees are not duplicated. Large historical proof/timing snapshots are excluded. These are reproducible source entry points with retained evidence, not a promise of bit-identical replay on another driver/GPU.
