# Native F1 FFN discriminator — RTX5090, 2026-10-06

**Measured route negative, not a universal fusion negative.** F1 computes its intended label map correctly but does not improve the matched complete FFN chain. Keep it experimental; no engine/Makefile integration is recommended.

## Final matched result

Own NVFP4 tensor-core kernels; CUDA12.8, `sm_120a`, NVIDIA GeForce RTX5090. Current project FP4/prep kernels are the baseline, not vendor GEMM. Fixture: original Bonsai2 27B layer10, eight unique held normalized inputs from saved chat captures, repeated for512 timing rows. This is normalized-input→residual **contribution**, not full model/serving: common RMSNorm and residual addition are excluded from every arm. Weight scales are fp16 per128 (WF0); this is not a comparison against the newer optional E4M3-scale production path.

CUDA-graph full-chain timings, ten rotating-order rounds of30 executions/arm (warm). F1 names indicate GU ownership tile; down is selected independently as FW8/TOK64 at512 and FW8/TOK16/splitK16 at1–8. Baseline independently tunes its GU and down across installed configurations, with down splitK1/4/16.

| Rows | Baseline | Fixed-scale/fullH | Best F1 compact | Best F1 expanded |
|---|---:|---:|---:|---:|
|1|33.738us|32.810us|35.218us (GU16), +4.39%|33.505us (GU16), −0.69%|
|8|35.726us|34.825us|35.301us (GU16), −1.19%|34.689us (GU16), −2.90%|
|512|330.329us|338.201us|335.253us (GU32), **+1.49%**|337.364us (GU32), **+2.13%**|

Five rotating-order rounds of five512-row executions after128MB cache perturbation **before, outside** each event bracket: baseline324.243us, fixed329.517us, F1 compact331.213us (**+2.15%**), expanded330.349us (**+1.88%**). The perturbed test is a cache diagnostic, not a DRAM counter or guarantee of cold lines; neither it nor one repeated layer is a64-layer serving benchmark. The smaller8-row F1 difference is matched or beaten by the fixed-scale/fullH control without the basis change. There is no established useful whole-FFN gain.

The final native outputs differ from their CPU-map references by roughly1.5–3.7e-7 relative RMS. Compact/expanded outputs match exactly at512 and within floating reduction noise at low-row splitK. The code gates timing on finite outputs and<1.5% CPU relative error; actual error was far smaller. This proves correct lowering for these probes, not broad model quality. Earlier whole-model F1 CPU KL remains .005822 versus .005316 halo emulation across252 teacher-forced tokens; no new whole-model native quality benchmark was run.

## Where the local gain went

Paid component screen at512; these are direct component CUDA-event times, **not** summed as the headline:

| Component | Current baseline | Final F1 compact, GU32 |
|---|---:|---:|
| Input signed H1024/NVFP4 prep |8.35us|11.61us (fixed-scale custom prep)|
| Gate/up, SiLU |191.10us|207.99us (includes signs/H256/quant/direct code emission)|
| Separate hidden signed H1024/quant prep |32.77us|deleted|
| Down |79.28us|103.23us (SWAR H4 transformed operands)|

Producer-local fusion itself is useful in this screen: fusedGU208us costs less than originalGU191us plus hiddenprep32.8us. **Down operand formation/reuse and the input boundary erase that saving.** The complete chain, not this phase comparison, decides adoption. Expanded down is101.18us despite removing decode because it reads twice the coefficient bytes.

## Implemented maps/layouts

- `src/f1.cu`: specialization of our own `src/fp4.cu` streaming/mbarrier MMA kernel in a distinct namespace; no existing kernel edited. FusedGU FW8/WG2 owns256 hidden positions, fixing hidden bits5/6 and varying0–4/7–9. Gate/up groups are permuted offline by `f1_export.py`. Two CTA barriers drain the streaming producer and reuse its ring as a row×256 tile. The epilogue performs H256, fixed-scale NVFP4 quantization and stores in the original tiled/swizzled natural-K layout. No global fp32 h.
- `src/f1_epilogue.cuh`: H256 and block16 quantizer. GU16/32/64 variants; output row scales written by one CTA only.
- `src/f1_swar.cuh`: eight four-trit H4/2 groups directly become E2M1 operand labels through biased4-bit add/sub and magnitude/sign logic. Every nibble intermediate remains bounded to prevent cross-channel carry/borrow. No per-tuple LUT, shuffle or scatter on the default path. `F1_USE_LUT` selects the initial shared-LUT comparator. `f1_swar_test.cpp` executes209,952 alternating-neighbor checks.
- `src/f1_prep.cu`: fixed-scale or row-amax signed H1024 quantizer for normalized inputs. Exactly one input transform and one hidden transform in fixed/fullH; **no accidentally duplicated fullH**. Vectorized float4 loads and two scale conversions per lane, matching current quantization structure.

Down compact keeps original2-bit trits,22,282,240 coefficient bytes +1,392,640 scale bytes =**23,674,880B/layer**. Expanded uses44,564,480 coefficient bytes +same scales =**45,957,120B/layer**. Both gate/up images are47,349,760B including scales. The compiled all-variant benchmark executable is12,258,480B (SHA256 pinned); that includes the baseline configuration bank and unrelated kernels, not a selected F1-only binary. No selected device-program-text/total-static-byte dominance is claimed. The standalone benchmark retains both images/controls, but a selected F1 implementation needs only its permuted GU plus selected down representation. Offline permutations/repacking and CPU references are excluded from online timing; all recurring decode, MMA scales, activation preparation, producer communication, quantization and down are included.

Compiler resources, final variants:

| Kernel | Registers/thread | Spill loads/stores | Dynamic shared |
|---|---:|---:|---:|
| Fused GU16 |84|0|92,880B|
| Fused GU32 |136|0|78,912B|
| Fused GU64 |168|376B each;192B stack|88,128B|
| Compact down FW8/TOK64 |133|0|93,296B|
| Expanded down FW8/TOK64 |131|0|86,080B|

GPU opt-in per-block shared limit is101,376B; these shared footprints limit residence. The GU64 spilling route takes~581us complete-chain at512, far worse than GUI32. Full ptxas logs are preserved; these resources are compiled measurements, unlike the earlier modelled occupancy discussion.

## Reduction history / evidence

1. `9c8397d`: initial shared512B LUT decoder, forced FW4 down tied to GU32/64. Correct lowering,512 baseline325.46us versus compact755.32/738.78us and expanded421.30/595.98us. Initial null row-scale pointer read in the new down epilogue was caught by memcheck and fixed before any accepted timing. Not used as an approximation result.
2. `30e94ad`: SWAR decoder and vectorized fixed prep. Compact430.88us versus baseline329.99us. Paid phases revealed the independent down geometry mismatch (FW4/TOK32 versus baselineFW8/TOK64).
3. Final source: independent GU/down geometry, low-row splitK and GU16; final results above. This closes the configuration mismatch without turning the remaining loss into a theorem.

`f1_first*.log/jsonl`, `f1_swar*.log/jsonl`, `f1_aligned*.log/jsonl`, `f1_cold.jsonl`, and `f1_native_sha256.txt` preserve resource/timing/source/binary receipts. Initial and intermediate timings must not be promoted over the final matched configuration.

## Build and run

No model download needed on the GPU; exporter writes only one layer and the timing/reference fixture (~175MB). Host export/audit obey2-thread cap:

```sh
cd ~/work/clones/halo-5090-combined
~/work/eval-cap/run env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=~/work/ffn-label \
  python3 bench/f1_export.py ~/work/ffn-label/data/native-f1 10
~/work/eval-cap/run env OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=~/work/ffn-label \
  python3 bench/f1_audit.py ~/work/ffn-label/data/native-f1 10
c++ -O3 -std=c++17 bench/f1_swar_test.cpp -o /tmp/f1-swar-test
/tmp/f1-swar-test
# On the parent's existing5090, only during a granted exclusive window:
bash bench/f1_build.sh
bench/f1-bench /root/f1-fixture 30 10 1 0
bench/f1-bench /root/f1-fixture 5 5 1 1 512
```

The programme12 native bundle also builds independently: from its native/ directory, set `F1_HALO_SRC` to the halo source checkout for the pinned baseline headers/kernels and run `bash bench/f1_build.sh`. Original numerical tools/captures stay at the owning parent FFN record/work directory, as the exporter commands specify.

Build requires CUDA12.8+, `sm_120a` (plain sm_120 rejects block-scaled MMA). No rentals were created here; three brief GPU windows were scheduled by the parent. GPU tests are finished; **no integration edits** are proposed for this losing route. The information-theoretic question remains open to other cheaper maps; this experiment measures one complete implementation, not all possible fused FFNs.
