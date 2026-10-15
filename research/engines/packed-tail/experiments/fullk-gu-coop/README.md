# Full-K cooperative GU — opt-in development

Owner: programme12, reporting conversation00000000-0000-0000-0000-000000000000. Serving-source base049cd7db. Sources/receipts: `~/work/five-wide-native/FULL-K.md`; the RTX 5090 host sidecars under `models/experiments/bonsai-quality-20261013/fullk-gu-engine-v*`. Serving binaries and frozen plan remain unchanged.

`H5_GU_COOP=mixed|plain|ring-mixed|ring-plain` selects only R64 target GU; other row shapes explicitly retain serving behavior. Requires row-mean FP4 target arithmetic and ordinary sign/Hadamard/FP4 down preparation. Unknown settings and incompatible alternate schedules reject at creation. Static packing/extra weight residency are outside decode timing and reported separately.

- `mixed/plain`: coalesced64-K dense5 slabs, standalone128-feature/Tok8 launch geometry, SiLU fused. v2 built,40registers/no spills. B64x128 one-pass screen: mixed3021.25/plain3039.17tok/s; serving6782.90,flag-disabled candidate6780.93. No adoption.
- `ring-mixed/ring-plain`: compressed832-byte32-row/128K slabs and the actual selected serving TMA/warp/Tok geometry. Mixed state is encoded between native128-K transitions. `fullk_ring.cu` instantiates the shared `fp4.cu` body; frozen ordinary objects are reused for the private Windows sidecar. v4 built in16.95seconds. B64x128 one-pass: mixed4075.02/plain4087.28tok/s; still loses serving. Each arm separately matches71,303,168 same-input serving GU F32 words,zero errors. No adoption.
- Fixed-index/shared-book v5 loses further:3804.04/3813.17tok/s. Fixed-index/global-book v6:3701.36/3723.92. Both wiring formulas are Lean-proved, and neither price improvement follows. Live experimental source retains the best measured v4 generic/global producer; successor evidence is retained in native custody.
- `H5_GU_COOP_CHECK=1`: diagnostic same-input comparison to the serving GU at every selected boundary. It changes timing; omit from performance runs. `h5_fullk_gu_check_stats` returns compared words, differing words,max-absolute-error bits,nonfinite differences.

The synthetic prototype's128-feature Q8 normalization is not the engine's1024-Hadamard/FP4 consumer. No direct Q8 substitution is performed. Actual FP4 observer fusion remains separate work.

Model-check receipts retain8 teacher-forced B64 steps and full248320-logit heads at rows0/17/33/63. Declared max1e-3/mean1e-4 and zero top1-difference bounds failed for both candidate arms and the serving binary's own repeat (max2.2648/mean0.2003,15/512 top1 differences). These do not establish adapter-specific corruption or acceptance. Use same-input GU diagnostics to isolate it.

v1's table upload preceded stream creation and crashed at load; v2 repairs lifetime ordering. The failed executable and receipts remain retained. v3's all-template rebuild exceeded240seconds; the source-shared sidecar avoids rebuilding the ordinary catalogue. No successful v3 build is assumed.
