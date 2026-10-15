# Complete joint two-FP4 FFN control

`../whole_ffn_joint_control.cu` loads real layer0 gate/up/down Q1 weights and original FP16 K128 scales. It starts from the **same deterministic F32 post-normalization input** as the A8 controls; no RMSNorm operation is inside this boundary.

The complete joint map charges both original NVFP4 primary/residual producers, joint gate/up projection, actual native SFU SiLU×up, both original next NVFP4 producers, joint trained down projection, all original weight-scale arithmetic and split folds. Gate/up WF8 weights retain the engine's 32-gate/32-up interleave. The separate A8 comparison maps charge their original K128 A8 producer and digit encoding where required.

Current joint API supports R≤16. R16 uses the engine's F4G2T16 GU/split1 and F4G2T8 down/split8; down/split16 is also priced. R32 is explicitly two independent R16 joint tiles with independently prepared 64-padded operand arenas. This is a declared tiled route, not an unsupported native-R32 call. A normal wide R32 ordered two-pass reference is measured separately with ordinary F4G2T32 projections, GU/split1 and down/split8; it is not a frozen full-engine tuning claim.

Numerical comparisons are separate:

- Native joint complete output versus its own ordinary primary-then-residual two-pass full-FFN reference; both use identical original FP4 producers, weights/scales and native SFU, but floating association can alter subsequent quantized operands.
- Each complete output versus the original-K128-A8 compound on the common F32 input. These are **different dynamic quantization maps**, not a same-A8 exactness comparison or accuracy ordering.
- The campaign parent reported prior Prism/Wiki256 quality for the original two-FP4 route: PPL19.2028/KL0.00037166. This subtask does not rerun whole-model quality, HTTP or task panels.

One bounded GPU window, under `/tmp/binary-gpu.lock` after parent priority work, has a 55-second execution timeout and a bounded 240-second lock wait. Timings use CUDA graphs, two initial replays then median seven samples; cold evicts128MiB outside the timed complete map. Static preparation, allocation and graph capture are excluded; all dynamic producers/continuations/readers/scales/folds are included. Source and cached linked object hashes are exported in `joint-provenance.sha256`.

## Single-window result

The bounded run completed at2026-10-10T23:49:21Z, exit0. All12 complete-map records are exported in `joint-r16-r32.jsonl`; these timings are this window's paired points, not best-control sweeps and not interchangeable with earlier window minima.

| Route | R16 warm/cold µs | R32 warm/cold µs |
|---|---:|---:|
|Ordinary two-pass reference, GU32split1/down32split8|93.280 /102.144|101.376 /112.384|
|Joint GU16split1/down8split8, independent16-row tiles|63.552 /73.696|119.840 /131.072|
|Joint same GU/down8split16|65.632 /75.744|123.168 /134.880|
|Compound exact3 A8|95.744 /100.352|122.944 /126.944|
|Four-digit A8|107.680 /114.432|140.352 /145.376|
|Fused DP4A GU→A8→DP4A down|74.080 /79.872|146.688 /153.600|

The joint R16 route wins this matched window; explicit two-R16-tile joint R32 loses to the wide ordinary two-pass reference. A universal native-three-plane cost win against this two-FP4 quantization map is therefore rejected.

Full-FFN native joint versus its own ordinary two-pass reference has relative RMS4.16e-5 atR16 and3.37e-5 atR32, after the actual nonlinear and next quantization. This is larger than isolated projection association error because the changed association can change next FP4 rounding. Complete pair-versus-A8 differences have relative RMS about0.00817 atR16 and0.00795 atR32; neither route is asserted exact relative to the other's quantization map. No full-model quality or task conclusion is inferred from these deterministic-input numerical points.

The source and receipts are stable. No pinned kernel/header/benchmark source changed, and no further GPU work is part of this subtask.
