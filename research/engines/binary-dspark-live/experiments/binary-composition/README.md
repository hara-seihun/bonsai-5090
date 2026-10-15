# Binary GU → SiLU×up → next A8 composition

Owner: binary Bonsai27B campaign. Parent owns `src/` integration and full-model quality/throughput. This experiment owns a finite-map search, fused implementation, original trained-weight keyhole and inclusive recurring bill. No GPU rental, public push or persistent serving was added.

## Result: fuse the continuation

**The split-K finalizer should produce the next A8 carrier directly.** It saves the GU and hidden FP32 roundtrips and two launches; the complete trained FFN chain is 1.10–1.20× faster in this keyhole. No coarser integer label in the declared grammar survived the observed whole suffix.

RTX5090 cold median microseconds, `N=17408`, `K=5120`, original `blk.0` gate/up/down weights:

| R | input producer + GU + SiLU×up + next A8, unfused → fused | same + down, unfused → fused | winning tile/chunk |
|---:|---:|---:|:---:|
|1|30.720 → 24.576|42.976 → 38.880|1/4|
|8|47.104 → 34.560|67.552 → 57.088|8/4|
|16|67.328 → 50.688|98.336 → 81.888|8/16|

The unfused column uses existing `bitserial::select` and then explicit SiLU×up and existing A8 production. The same-chunk unfused controls are also in the raw ladder: including down, 42.752, 69.632, 102.400us versus the winning 38.880, 57.088, 81.888us. Same-chunk comparisons are **bitwise equal at the next carrier and down output**. Different chunk counts change FP32 summation association; selected-versus-best speed ratios do not assert bitwise equality across those schedules.

Warm inclusive whole-chain unfused/fused: R1 20.864/18.656us, R8 50.368/43.264us, R16 83.680/75.872us. `summary.json` selects the cold winner independently for each R and scope. Raw `results/r{1,8,16}.jsonl` retains every measured route, including losing routes.

## Start, target and middle

Start: one given signed-A8 token/K128 carrier and the original packed one-bit signs plus FP16 weight scales for gate and up. End: the **next K128 A8 carrier** consumed by the down projection, including dynamic block scales, DC and typed status. Neither an FP32 GU tensor nor an FP32 hidden tensor is part of that end state.

For each original K128 group, the paired projection computes labels

```
selectedG = DP4A(expand(signNibbleG), packedQ)
selectedU = DP4A(expand(signNibbleU), packedQ)
dotG = 2*selectedG - DC(Q)
dotU = 2*selectedU - DC(Q)
g = orderedFMA(dotG, originalHalfScaleG * activationScale, g)
u = orderedFMA(dotU, originalHalfScaleU * activationScale, u)
```

`expand(n)=((n*0x00204081)&0x01010101)` operates on the original nibble coordinate; no persistent ±1 byte weights exist. Gate and up share each activation word load. All tokens in a tile share each weight word load. Original scales remain separate for every row and K128 group.

No-split route keeps `g,u` in registers and quantizes `SiLU(g)*u` within the CTA's128 feature lanes. Split route writes scaled gate/up partials, then its finalizer folds those partials **in the same order as the unfused control**, applies native `g/(1+__expf(-g))*u`, reduces128 magnitudes and writes A8 directly. There is no final GU or hidden FP32 image.

Requant contract is unchanged from `binary-bitserial`: `s=max(abs(h))/127`, `q=roundToNearestEven(h*(127/max))`, all-zero block→zero scale/bytes. It writes q, scale, DC and status. Nonfinite h or a nonzero maximum outside `[1e-30,1e30]` is an explicit asynchronous error, with optional persistent error aggregation. Invalid input carriers yield nonfinite continuation rather than a fabricated successful output.

## Finite search and witnesses

`search.cpp` exhausts the **32513 integer labels in [-16256,16256]**, the signed-±127 K128 envelope. The existing [integer witness](../binary-bitserial/witness/README.md) proves the stronger full signed-byte bound and packed DP4A bridge. The endpoint search is an over-domain necessary test, not an enumeration of all correlated trained gate/up pairs.

Declared27-candidate grammar: bit masks8..16; ten residues255..32768; three affine `3*x+b` labels; native FP32 integer labels; four rounded shifts. Exact dot-fibre matches: masks15/16, residues32513/32768, all three affine labels and FP32 conversion. Smaller masks/residues and rounded shifts return concrete collisions. These injective matches do not imply a cheaper continuation: they still need decoding/scaling, so the CUDA route retains the prepared DP4A selected-positive labels instead.

The search also evaluates a complete finite suffix: variable gate dot, effective scale in `{1/1024,1/128,1/16,1}`, up=1, and127 other hidden lanes=1, observing the variable next-A8 byte and dynamic output scale. Its fixed-scale fibre counts are15110,16255,16292,16264. The same coarser candidate labels fail with distinguishing endpoints. These are **host FP32/libm** finite results, not a claim that CPU exp and GPU SFU exp have identical bits. Native fused/unfused equality is separately checked on the device.

A scale-blind full integer sum fails even at two groups: dots `(1,0)` and `(0,1)` both sum1; effective scales `(0.5,1)` give gate0.5 versus1, SiLU0.311229676 versus0.731058598, and next byte40 versus93 when the other hidden lanes are1. Dynamic activation scales and original per-row/group scales cannot be dropped. A table over one collapsed integer sum is not the original region's map.

`Composition.lean` proves the selected/DC affine identity, equality through an arbitrary ordered scale/state transition, and equality through an arbitrary **whole feature-group observer**. Scale and State are abstract types so the theorem does not substitute ideal real arithmetic for native scale/FMA/SFU semantics. This is a congruence/transport theorem; CUDA memory scheduling and the concrete native observer remain device-checked. `results/lean.log` contains kernel-check axioms, no admissions or native-decide axiom.

The native trained-weight keyhole compares every next q byte, FP32 scale bit, DC and status against the same-chunk existing projection→SiLU→producer path. It then compares every down-output FP32 bit. All candidates passed at R1/8/16, for chunks4/16/40 and tiles1/4/8 where divisible. Per comparison: R1 17408 bytes/136 blocks/5120 down outputs; R8 139264/1088/40960; R16 278528/2176/81920. Inputs are deterministic generic finite FP32 activations, **not captured model hidden states**. Original trained weights supply the requested real-model keyhole; full-model PPL/tasks/TPS remain the parent's work.

## Complete bill

All timed graphs include input FP32→A8 production, packed projection, every original scale multiplication, all partial writes/reads, SiLU×up, block max, round, DC/status and next carrier stores. The `down_included=true` graph also consumes that actual carrier in the trained down projection, including its own scaled split-K partials and output continuation. Weight preparation/device upload is measured separately in each row's `static_prepare_ms`, excluded from recurring latency, and reused. No static conversion is presented as a recurring free operation.

- Original prepared signs+scales remain1.125bits/weight. Gate/up together occupy25067520 bytes. No dynamic dot/nonlinear LUT is produced.
- Input carrier per token/K128:128 q bytes +12 metadata bytes; next carrier same. Optional aggregate error is one shared4-byte allocation.
- Winning GU partial bill is `4 * R * 2*N * ceil((K/128)/chunk)` bytes written and subsequently read: R1/chunk4 1392640 bytes; R8/chunk4 11141120 bytes; R16/chunk16 6684672 bytes. Same-chunk unfused controls incur the same partial bill.
- Removed roundtrip traffic is `24*R*N` bytes:417792 /3342336 /6684672 bytes at R1/8/16, plus two kernel launches. Input quantization and next-carrier stores remain charged.
- The down projection uses existing `bitserial::select`; no down gain or producer reuse credit is fabricated.
- No-split fusion avoids all GU partials but loses at R8/R16 in this schedule. Splitting remains worthwhile despite its fully charged memory bill.

`results/build.log` records paired T1/T4/T8:38/72/116 registers, no spills; finalizer34 registers. `results/sass.json` records native `IDP.4A.S8.S8` (64/256/512 static instructions), `MUFU.EX2`, reciprocal and scale FMA, no IMMA. Counts are instruction evidence, not throughput extrapolation.

## Reproduction and custody

From this directory:

```sh
make
./build/search > results/search.jsonl
cd /opt/lean-mathlib
lake env lean /ABS/experiments/binary-composition/Composition.lean
# Existing campaign GPU, within its granted window:
cp build/bench ./bench
flock /tmp/binary-gpu.lock timeout 55s ./sweep.sh MODEL results
python3 summarize.py
```

API: explicit `composition::Plan{N,K,R,tile,chunk}`, `fused(plan,signs,originalScales,inputCarrier,nextCarrier,partials,stream)`; `validate` returns CUDA's typed invalid-value error for unsupported dimensions/tile/chunk. Supports R1..16 divisible by tile1/4/8, N and K positive multiples128. Buffers are caller-owned; `partial_bytes` is meaningful after plan validation. Input/output/partials must be distinct tightly packed allocations. Check asynchronous numerical status after ordinary stream synchronization.

Measurement:2026-10-10 20:11:30–31UTC, existing RTX5090, driver595.58.03,500W, CUDA13.1, build from source4fb3647 plus this experiment. Median15 after3 warmups;128MiB L2 eviction lies outside each cold timed graph. kjob `binary-composition-20261010` waited for the parent's `/tmp/binary-gpu.lock`, executed the entire sweep under a55-second guard and exited0. Raw sources/receipts exported before handback. `results/timed-build.json` binds executable and uploaded source hashes; the later upper-dimension host validation guard changes no device kernel. No `src/`, parent `tools/` or parent `results/` files were edited here.
