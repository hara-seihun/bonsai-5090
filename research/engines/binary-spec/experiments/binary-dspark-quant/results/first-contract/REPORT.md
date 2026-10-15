# First native precision contract

Source native5732320/quant488175b, cached live-attention control, CUDA13.1 sm120a. Synthetic four raw-tap rows, anchor198, candidate vocabulary65536, H5_PDL=1; no HTTP or target throughput.

Weight producer GPU/CPU WF1 prepared-byte parity and graph replay passed, including zero, wide finite BF16 and invalid-row cases.

BF16 control agreed on all four proposals with the fork. Base logits relative RMS0.002700707149, cosine0.999996372749, maximum absolute error0.032507062. Warmed complete native run (four-row ingestion plus four-query proposal block) mean2.404606ms across eight samples.

The first nvfp4-a1 activation call failed before a projection result: shared row preparation selected800 threads for FC width25600, but its kernel has launch_bounds544. No quantized quality or speed claim follows from this run. The owning repair gives DSpark an independent fixed-thread activation producer rather than changing the target's global preparation configuration. Raw failed receipt and object manifests remain beside this report.
