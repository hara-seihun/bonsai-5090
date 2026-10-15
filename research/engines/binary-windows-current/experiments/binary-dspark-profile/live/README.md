# Live-ring attention: matched S1 follow-up

**The repair removes attention as the dominant drafter bottleneck.** On the exact previous warmed 48-input /128-output HTTPS prompt, untraced decode rises **173.113 →224.199 tok/s (+29.51%)**, client-wall HTTP **164.774 →210.114 tok/s (+27.52%)**. Text, token usage, accepted proposals and prefix histogram are unchanged.

This is cached isolated source `90d1577`: old `e4de708` plus only historical-ring kernel repair `93835c0`, cached b17 object, binary `/workspace/binary/dspark-live/halo/h5-server`. SHA256 `7ed6df7353320997b65ee6eea4d2fcdc76e68ba30cf104840b1f73926b4abbad`. No model, numerical policy, context or projection route was changed. Parent owns independent fork/task quality gates; this result is the matched prompt only.

## Matched request result

Old control/capture are immutable `../evidence/` from commit `3004518`; new receipts are `evidence/`.

| Quantity | Old untraced | New untraced | Old traced | New traced |
|---|---:|---:|---:|---:|
| Decode tok/s (127 after first token) | 173.113 | **224.199** | 168.571 | **216.717** |
| Client-wall HTTP tok/s (128 tokens) | 164.774 | **210.114** | 159.870 | **202.325** |
| Client wall ms | 776.820 | **609.192** | 800.652 | **632.645** |
| Native target verification ms | 415.216 | **413.505** | 429.874 | **427.997** |
| Native drafting ms | 318.203 | **152.746** | 323.275 | **157.788** |

All four arms: **74/220 accepted (33.64%)**, 55 verify rounds, 275 verifier rows, prefix histogram `[24,9,7,9,6]`, identical response choices and 48/128 token usage. This is not the panel's 73% acceptance. The preparation dimensions remain identical: target R5, drafter query R4, steady ingestion histogram `{1:24,2:9,3:6,4:9,5:6}`, initial ingestion R16. This trace does not activate an R16+ target projection consumer.

Each run has one64-token warmup followed by control128 and traced128, on the same server. Capture-disabled control is an nsys-armed server with NVTX enabled, not a separately launched profiler-free executable. The new trace's decode rate is 3.34% below its control. Old/new are successive bounded windows, not interleaved repetitions; the saved identical work and stable target timing locate the measured gain in the repaired drafter.

## Traced decode decomposition (prefill excluded)

| Stage | Old range wall ms | New range wall ms | New GPU-busy union ms |
|---|---:|---:|---:|
| Target verify, 55 R5 calls | 429.857 | **427.961** | 377.750 |
| Steady draft, 54 calls | 317.444 | **155.043** | 145.827 |
| Initial draft, ingests final16 prefill rows | 5.756 | **2.706** | 2.525 |
| Final ingestion-only | 0.300 | **0.307** | 0.265 |

Three target-prefill and two draft-prefill-ingestion calls are excluded. Kernel sums overlap inside the target and must not be added as elapsed walltime. Non-GPU target range time is 50.211ms versus old50.882ms; steady draft9.216ms versus old9.339ms. Copy activity is stable: target0.358ms, steady draft1.254ms. Host/copies did not create this gain.

**Attention kernel activity: 186.710 →21.499ms, 8.685× faster** across the same330 six-layer launches. Steady per-layer median564.836 →64.480 microseconds. It is now **13.6% of drafter wall /3.7% of decode wall**, rather than57.8% /24.8%. The repair changes ring-position traversal; valid-key attention remains serial.

New initial+steady projection families (including conversion/reduction):

- Drafter BF16 six-layer trunk: **68.680ms** (old68.752).
- Candidate head: **22.636ms** (old22.641).
- Tap FC: **9.398ms**; ingestion KV **3.622ms**; confidence **1.501ms**.
- Drafter norms **8.093ms**, candidate argmax **4.855ms**, Markov **3.072ms**.
- Target packed-A8 R5 projections: **229.986ms**; preparation **87.209ms**; continuation **25.804ms**; producer **14.467ms**; GDN **28.074ms**.

**Next lever:** actual target R5 projection/preparation remains the overall limit (target is73% of new control stage wall). Within the drafter, BF16 trunk projection is now larger than attention, followed by its pruned head; another attention-only optimization can no longer supply the previous leap. Serial valid-key attention may matter at longer actual context, which this short prompt does not measure. No additional engine patch is made here.

## Receipt and reproduction

- Durable job `binary-dspark-live-profile-82ad352e` exited0 and copied off raw nsys-rep/sqlite, request/control/capture responses, native metrics, commands and identity. `evidence/kjob-receipt.txt`, `kjob-log.txt`, `process.json` retain custody.
- Server cleanup completed **2.742 seconds after readiness**, within the one55-second exclusive `flock /tmp/binary-gpu.lock` window. No process remains. CPU-only SQLite export followed GPU release; no old evidence was overwritten.
- `launch.json` differs from previous configuration only in cached binary path; samectx8192, proposals4, slots1, prefill16 and launch environment.
- `job.sh` /`run_remote.py` own lifetime and capture. `analyze.py` reconstructs native stages and preparation geometry; `compare.py` compares against immutable old evidence and saves `evidence/comparison.json`.
- `dspark_kernels.cu` and `.cuh` snapshot exact repair93835c0. Source snapshots are reference evidence, not a new engine implementation.
- Rental55260842, budget and hard00:28Z October11 destroy guard stay with parent. No rental, build, public push or persistent service.
