# Round22 paired saved-tape witnesses

Owner: packing-swarm CPU sidecar. Result: [receipt](../../../../../results/packing-swarm/round22/cpu/receipt.json), [state/boundary](../../../../../results/packing-swarm/round22/cpu/STATE.md). Inherited source commit: `9c4ed70089114706200215bf60910dbed2c35c95`.

`check.cpp` consumes sealed Round20 primary/innovation q, scale and global-scale tapes; all three independently computed complete primary/final snapshots and ordered-trace fingerprints. It never runs a producer, norm factory, quantizer or source trajectory. Preserved GGUF raw weights are reloaded using the inherited model index. Raw and native packed-weight hashes must match those retained by Round20.

The candidate output enumeration is four warps × two groups; the independently enumerated strongest FW8 arm is eight warps × one group. Both own eight groups and 32 tokens per CTA. Each has two independent ordered accumulators. The eight-stage 9,216-byte ring uses weight4096, weightscale512, primaryq2048, primaryscale256, innovationq2048, innovationscale256. Global scales are staged separately once per CTA. Every live output coordinate has one owner in each pass. Padded token q/scales are copied and addressed completely; missing GDN groups are guarded. Native-negative-nibble versus independent raw-positive-byte dots, half scales and explicit ascending40 FMA folds compare exactly in-loop; both outputs and trace fingerprints compare with every inherited arm.

Arithmetic is CPU integer-grid/E2M1/E4M3 and F32, with `-ffp-contract=off` plus explicit `std::fma`. K64 half-dot sums are integer witnesses, not an executed CUDA MMA result. The numerical lookup tables are built from saved tapes; every corresponding staged q/scale coordinate is checked equal before consumption. The saved initial output is +0. Source/norm hashes and trajectories remain inherited, including contract-only layer3 and FFN probes.

## Read-only replay

```sh
python3 method/packing-swarm/tools/round22/cpu/verify.py
```

This checks all sealed source/native/artifact identities and inherited lossless member hashes, then reruns all18 ordinary/UBSan cases without overwriting receipts or logs. It requires the retained `~/work/clones/packing-mma-round20` evidence cwd, the index at `~/work/packing-mma-round13/archive/packing-swarm/round13/model-index.txt` and the preserved `~/data/bonsai1/Bonsai-27B-Q1_0.gguf`. If the Round20 cwd has been released, restore exactly the nine `*-final.raw.manifest.json` and `*-final.raw.tar.zst` blobs from the prerequisite commit into the admitted replay workspace, then set the evidence path explicitly in a new replay generation. Do not regenerate input tapes.

## Fresh construction

In a fresh workspace without the output `archive/packing-swarm/round22/cpu/run-final`:

```sh
python3 method/packing-swarm/tools/round22/cpu/replay.py --build
# Each case is independently bounded; run each family as a separate <=55s command.
for B in 1 8 32; do
  for KIND in ordinary ubsan; do
    python3 method/packing-swarm/tools/round22/cpu/replay.py --case input0 --batch "$B" --kind "$KIND"
  done
done
# Repeat separately for input3 and ffn0.
python3 method/packing-swarm/tools/round22/cpu/replay.py --seal
```

The build copies the complete local C++ include closure before compiling it. Seal requires the native paired-reader source identity. Two successful pre-global-scale/explicit-owner instrumentation pilots and their source/binaries are retained outside `run-final`, excluded from final totals. No failure or numerical discrepancy occurred; one pilot build's misleading-indentation warning was fixed in the final source.
