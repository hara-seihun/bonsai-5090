# Predicated head GEMM contract probe

`predicated_gemm_probe.cu` links with the selected preset's `fp4.o`; compile both with `H5_PROFILE_SCALE_CARRIER`, C++17, `-Isrc`, and the engine's sm_120a architecture flags. It needs no model, engine object, or timing harness. The campaign manager owns its Makefile target and GPU execution.

The finite probe covers 320 cases, each replayed twice through a CUDA graph:

- sparse `N=512`, bounds `[2,512]`; dense `N=1024` proxy, bounds `[513,UINT32_MAX]`;
- counts `0,1,2,512,513`, `R=1/8`, split-K `1/4`, store/residual-add;
- ordinary atomic and deterministic ordered sums;
- WF3 original W128 scales and WF3 carrier with row means prepared by `scale_carrier_prepare`.

Each graph produces count on the GPU, releasing PDL before publishing it. Before each replay, count is poisoned to the opposite predicate result. This exercises the head's acquire of the producer's completion rather than its early release.

Rejected launches receive aligned, nonnull, unmapped pointers for **all operands**, while output and partial workspace are valid sentinel-filled allocations. CUDA join must succeed, and both buffers must remain bitwise unchanged. An actual TMA weight copy or activation/scale read on the rejected path faults. This exercises the skipped clear and skipped reduction as well as the GEMM guard; it is not a hardware traffic-counter measurement.

Active launches are compared with the existing ordinary/deterministic launchers using identical operands, prepared means, split, and add policy. Ordered and split1 outputs and all partial planes are compared bitwise; atomic split4 outputs allow `1e-6 + 2e-6*abs(reference)` for scheduling-dependent FP32 summation. Row padding is compared bitwise, including the ordinary nonadd split clear's existing whole-row behavior. References must be finite and nondegenerate.

The executable forces `H5_PDL=1`, prints every case and a final failure count, and exits nonzero on any failure. It reports no latency or performance result. Source added without a build or GPU execution; the run receipt belongs to the campaign.
