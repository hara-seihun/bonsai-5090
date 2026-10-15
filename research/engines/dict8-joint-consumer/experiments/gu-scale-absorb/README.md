# Fixed-row-mean absorption: a CPU falsifier

State: rejected **one-plane absorption**; no model-window nominee. No GPU/device query, native model/DLL execution, mutex acquisition or serving change occurred. The paired-WF0/K128/radix512 frontier is independent.

## Target and outcome

The target remains the complete real model above 6777.352952 ordinary / 6583.070035 deterministic tok/s, preserving the rounded K5120 gate/up contraction, row means and existing source-ordered SiLU/sign/Hadamard1024/FP4 observer. Fixed weights may be prepared once.

This route tries to put each fixed mean into prepared E2M1 weight operands and UE4M3 weight scales, removing its terminal multiplication. It fails two independent contracts:

1. **Actual weights do not fit one native product.** First 32 gate and 32 up rows of block0 in GGUF SHA256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` have zero exactly representable row means. Gate row0's reconstructed mean is binary32 `0x3c480333`, exactly 13108019 / 1073741824. A finite E2M1×UE4M3 coefficient is an integer divided by1024. Its odd significand is at most45; this mean's is13108019. A power-of-two residual cannot fix that significand. All64 sampled rows fail this necessary condition.
2. **Exact coefficient representation would still be insufficient.** With K5120,4096 terms of2048, zeros and a final0.5, source-order binary32 contraction followed by multiplication by an exactly representable1.5 gives12582912 (`0x4b400000`). Absorbing1.5 before contraction gives12582913 (`0x4b400001`). First-stage sums are exact under any grouping; the half-ulp term lies alone in the final128-K stage. This is a constructed interface witness, not a saved model activation.

The actual-row witness uses a real nonzero trit and a separate zero-trit maximum-amplitude guard in each row: q4=0.5/s4=1 at the contributing coordinate, q4=6/s4=448 at the zero-weight guard, g4=1, all other q4=0. Gate row0's original output is `0xbbc80333`; the nearest one-plane coefficient gives `0xbbc80000`,819 binary32 bit-pattern steps apart. This refutes preservation of the rounded GU boundary. No claim is made that an observed model tape or the final FP4 observer changes, or that this is universal native-MMA validation.

An arbitrary row-specific binary32 residual could retain the missing information, but then its multiplication remains. Multiple prepared planes require a new rounded reduction proof; real distributivity does not establish the original binary32 contraction.

## Complete-route bill

A single prepared FP4 plane for both17408-row matrices atK5120 needs89,128,960 q4 bytes plus11,141,120 scale bytes per layer:100,270,080B, versus44,564,480B of native WF0 two-bit weights (separate row means139,264B). At least two planes are necessary for these means **if no arbitrary binary32 residual remains**; two planes cost200,540,160B/layer and12,834,570,240B across64 layers, before any prepared-layout duplication. This is a representation lower bound, not a claim that two planes suffice or fit the RTX 5090 host's live resident state.

A direct independent-plane tensor contraction doubles the32 static OMMA instructions per K128 body to64;40 K128 stages recur per contraction. Other compositions may have different costs and require their own complete invoice. Preparing weights once does not remove recurring plane loads/reductions. The existing8B H1 carrier and its full consumer remain; retaining them contributes8,912,896B write+read per64-row layer and73,014,444,032B per64-layer×128-token batch. Folding the terminal scale alone earns no reduction in that bill. SiLU, sign, ordered Hadamard, q4/s4/g4 outputs and whole-model timing remain unmeasured for any new multi-plane route.

Thus this route supplies no complete-target nominee. It closes a specific mechanism, not packed-MMA GU research generally.

## Correct frozen-serving comparator

Exact frozen image `a836fd6818df9bc92a8df71d2f9e92fc97b9a64195fab4142d9101fb20785e7c`, caller `launch_scale_carrier_wf0 → carrier_launch<0> → CARRIER=true`, cfg13 FW8/WG2/TOK32/MODE3/split1:

```
_ZN2h510k_gemm_fp4ILi8ELi2ELi32ELi3ELi0ELi0ELb0ELb1ELb0EEEvPKhS2_PKfiiS2_S2_iPfiS4_
```

Correct selected function:2288 static instruction lines,0SHFL.IDX,0SHFL.BFLY,32OMMA;126registers,stack/local0,static shared1024B. This replaces the withdrawn CARRIER=false2427/no-SHFL/136register comparator. H1's own5604plain/5949mixed counts and128SHFL.IDX in its selected K loop remain. Static counts do not establish executed work, occupancy or reconstruction's latency share. No numerical/TPS conclusion changes.

`row-mean.sass` binds the actual frozen preparation kernel too: sequential FADD and reciprocal/refinement FFMA division sequence. The CPU row-mean reconstruction follows the frozen source's binary32 sum and division contract; device values were not read or newly executed. The scale swizzle only permutes rows: `pack_trit_chunk` preserves original FP16 scales, and `k_row_scale_mean` recovers each row in K order.

## Evidence and reproduction

- `collect_cpu.py`: CPU-only exact-function disassembly, whole-file GGUF hash and64 raw fixed-weight rows. Native generation refuses overwriting existing custody.
- `collect_row_mean.py`: separately preserved CPU-only preparation-kernel disassembly.
- `falsify.py` / `falsifier.json`: exact finite-format enumeration, original PTQ trit decode and all64 actual-row witnesses.
- `rounding.cpp`: separate binary32 witness implementation. Compile `g++ -std=c++17 -O3 -frounding-math -ffp-contract=off rounding.cpp -o rounding`; running it exits0 only when both mismatches occur.
- `Absorption.lean`: integer-grid obstruction for actual gate row0, finite format odd-capacity bound and explicit local round-to-even-bin witnesses. `lean Absorption.lean` passes Lean4.33.0. These statements do not formalize CUDA's complete floating semantics or the model observer.
- `evidence/`: hash-bound native SASS/source/raw rows and local receipts. No generated host binary is retained in source.

Native read-only collection archive SHA256 `b35c5cdcd2291a500b40edc31272ad5420f8703186fb6e8e7b6b219c36a3e199`,202221B. Original H1 invoices, packages, failures and serving files remain unchanged. Source is private; push/bundle custody precedes writer release.
