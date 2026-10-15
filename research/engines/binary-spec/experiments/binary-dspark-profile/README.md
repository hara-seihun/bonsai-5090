# Binary DSpark S1: the next speed limit

**Next independent fix: stop scanning the entire DSpark KV ring, then replace its serial per-key attention.** Cached source `e4de708` spends 186.710 ms in six-layer drafter attention across 55 four-row blocks: 57.8% of drafter wall and 24.8% of traced decode wall. Native target projection remains the largest overall family, but its small-row native-SF integration has a separate owner.

This is one warmed actual greedy HTTPS-explanation prompt, 48 input / 128 output tokens, not a macroquality panel. Four proposals, one stream, exact original launch environment, F16 target state/KV, packed1 hybrid F32-R1/A8-R>1 target, native BF16 drafter, candidate vocabulary 65,536. No build or engine edit was performed.

## Real request control

| Quantity | Capture disabled | CUDA graph-node capture |
|---|---:|---:|
| Client wall | 776.820 ms | 800.652 ms |
| Client-wall rate, 128 output tokens | 164.77 tok/s | 159.87 tok/s |
| Native prefill | 41.830 ms | 45.211 ms |
| Decode wall after first token | 733.624 ms | 753.390 ms |
| Decode rate, remaining 127 tokens | 173.113 tok/s | 168.571 tok/s |
| Target verify | 415.216 ms | 429.874 ms |
| Drafter | 318.203 ms | 323.275 ms |
| Scheduler residual after first token | 0.206 ms | 0.241 ms |

Both requests produced identical text, 55 verification rounds, 275 verified rows, 74/220 accepted proposals (33.64%), and prefix histogram `[24,9,7,9,6]`. This prompt does **not** reproduce the aggregate panel's approximately 73% acceptance / 250 HTTP tok/s. Capturing adds 2.7% decode wall; the control is the same nsys-armed server with collection disabled, not a separately launched profiler-free executable. NVTX is enabled in both.

## Decode-only time split

Source labels wrap synchronous native calls. Three prefill target calls and two ingestion-only draft calls are excluded. Initial drafting (which ingests the final 16 prefill rows) is shown separately; the final ingestion-only call after completion is also separate.

| Traced region | Calls | CPU range wall | GPU busy interval union | Wall minus GPU busy |
|---|---:|---:|---:|---:|
| Target verification, R=5 | 55 | 429.857 ms | 378.975 ms | 50.882 ms |
| Steady drafter | 54 | 317.444 ms | 308.105 ms | 9.339 ms |
| Initial drafter | 1 | 5.756 ms | 5.571 ms | 0.185 ms |
| Final ingestion-only | 1 | 0.300 ms | 0.265 ms | 0.036 ms |

Wall minus GPU busy includes submission, synchronization, scheduler stalls and trace overhead; it is not an identified removable host computation. CUDA API duration includes waits and cannot be added to GPU duration. Target kernel sums overlap (412.456 ms versus kernel union 378.615 ms), so family sums below are activity costs, not additive wall fractions.

### Target verification kernel activity

- Packed-A8 R5 projections: **230.585 ms**, 16,775 launches.
- F32 preparation/RMS/continuations in `k_prep2`: **87.630 ms**, 14,190 launches.
- A8 scale/output continuation: **25.874 ms**; A8 producer **14.538 ms**.
- F16 GDN: **28.167 ms**; convolution **11.310 ms**.
- Target attention TC + combine + preparation: **13.277 ms**.
- Argmax **0.779 ms**.
- All target copy activity: **0.360 ms**, 47.32 MB including device-to-device taps. KV/copies are not the dominant short-context family.

### Drafter query activity, initial + steady

The native project sequence was reconstructed from conversion boundaries and pinned source, retaining cuBLAS split-K reductions where present.

| Family | GPU kernel activity |
|---|---:|
| Attention | **186.710 ms** |
| Six-layer trunk BF16 projections + converts | **68.752 ms** |
| BF16 candidate head + convert | **22.641 ms** |
| Tap FC + convert/reduction | **9.397 ms** |
| Six-layer ingestion KV projections + converts/reductions | **3.630 ms** |
| Norms | **8.102 ms** |
| Candidate argmax | **4.843 ms** |
| Markov rank256 | **3.080 ms** |
| Confidence projections + converts/reductions | **1.505 ms** |
| QKV/RoPE/cache | **1.424 ms** |

Steady draft copy activity is only **1.258 ms**, 73.96 MB. The candidate head is already pruned; further head/Markov work is not the largest current lever. SNR conditioning is prepared once at load, outside the capture; it is not a recurring decode stage.

## Actual prepared dimensions

No inferred R16 padding: trace contains `bitserial::projection<int5>`, producer gridY=5 and preparation gridX=5 for all 55 verification rounds. Target head has all five rows and full V=248,320. Dimensions below use `output N × input K`; R is the actual prepared activation row count.

- Target D=5,120, FF=17,408, 64 layers (48 GDN /16 attention), R=5.
- Target GDN input N=16,384 K=5,120; alpha/beta N=96 K=5,120; GDN output N=5,120 K=6,144.
- Target attention input N=14,336 K=5,120; output N=5,120 K=6,144.
- Target fused gate/up N=34,816 K=5,120; down N=5,120 K=17,408; head N=248,320 K=5,120.
- Drafter block query R=4, six layers, D=FF=5,120, H=40, KV=4, HD=128. Attention launch grid `[4,40,1]`, block `[128,1,1]`.
- Drafter ingestion R histogram across 54 steady calls: `{1:24,2:9,3:6,4:9,5:6}`. Confirmed by tap-FC converter's `100*R` CTAs, not guessed from requested proposals. Initial ingestion R=16.
- Tap FC N=5,120 K=25,600; ingestion KV N=1,024 K=5,120.
- Query QKV N=6,144 K=5,120; O, gate, up, down each N=K=5,120.
- Candidate head N=65,536 K=5,120; Markov output 65,536 × rank256 × R1 repeated four serial positions; confidence N=1 K=5,376 R1.
- BF16 query GEMM trace selects `cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8`, observed grids `[8,48,1]` QKV, `[8,40,1]` other trunk, `[8,512,1]` head, block32. These launch tiles are not additional logical query rows.

The already-integrated digit-ragged R16+ policy does not cover S1 R5. Small-row target projection improvements must price the actual R5 route, including preparation and continuation.

## Concrete fix and scope

`source/dspark_kernels.cu:190` scans **all context+4=8,196 ring slots for every query/head** regardless of actual populated context. Each layer launches 160 CTAs; all six layers scan about 7.87 million ring-position entries per four-row block before serial attention over valid keys. `attention_key` performs a 128-thread shared-memory reduction and online softmax for each key. Steady trace attention median is 564.836 microseconds per layer; exact inventory is in `evidence/analysis.json`.

1. Iterate logical positions `[begin,anchor)` and derive each ring address as `position % ring`, with exact tag equality and the existing typed missing-KV error. Keep four query rows' block visibility. This removes the empty-slot scan without restricting the model's context or changing its weights. Preserve validity checks currently performed for block and tag metadata.
2. If valid-key serialization remains dominant, use tiled parallel QK/softmax/V for the four-row block and shared grouped-query KV reads. The contract is the measured output/task quality; no obligation to preserve current per-key intermediate structure.

These are handed to integration owner `00000000-0000-0000-0000-000000000000`, not implemented here. No speedup is claimed. Even eliminating **all** measured drafter attention would change this control's decode rate only from 173 to about 232 tok/s at the same acceptance (counterfactual using traced attention time). Reaching 2,000 requires both much faster actual R5 target execution and more useful committed tokens per round, not just pruning the draft head.

## Reproduction and custody

- Input launch owner: `../../results/binary-5090/tasks-pruned/tasks-hybrid-dspark4-b1/launch.json`.
- `run_remote.py` owns readiness, warm/control/capture requests, a 45-second alarm with <=9 seconds cleanup, server process group, and typed failure artifact.
- `job.sh` uploads runner/config, uses `flock /tmp/binary-gpu.lock`, and owns foreground remote execution, CPU-only sqlite export after releasing GPU lock, and evidence copy-off.
- Durable job `binary-dspark-profile-82ad352e` completed with exit0, receipt preserved in `evidence/kjob-receipt.txt`. Remote directory `/workspace/binary/dspark-profile-82ad352e` is disposable after campaign custody. No process remains. Diagnostic/server exit completed **3.283 seconds after readiness**, within the 55-second lease; export completed in the same job after lease release.
- Cached executable `/workspace/binary/pruned/halo/h5-server`, SHA256 `78c94e429d1c6881a24a97e131e31636a59b5d81b180374ac6a1b254c554432a`; source owner pin e4de708 supplied by campaign. Remote checkout has no Git metadata. Selected source snapshots are exact `git show e4de708:...` exports.
- Nsight Systems 2025.4.1, CUDA graph-node /CUDA /NVTX /cuBLAS trace. cuBLAS domain is present but no per-call dimension NVTX payload was exported; dimensions are pinned source plus observed conversion and launch geometry, not hardware operand dumps.
- `evidence/trace.nsys-rep` and `trace.sqlite` retain raw capture; JSON requests/responses, native metrics, startup logs and commands retain the control. `python3 experiments/binary-dspark-profile/analyze.py` rebuilds decode-only analysis in seconds.
- Existing rental55260842 and parent destruction guard remain unchanged. No new rental, public push or persistent service. This experiment owns no macroquality conclusion and no engine patch.
