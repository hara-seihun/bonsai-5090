# Composed mixed-FP4 gated map

Owner: this continuation; Programme12 owns eventual research integration, engine owns any adoption. Original ternary-only witnesses and all reader/profiling controls remain separate. Sources, complete native images, observers, negative witnesses, builds and hash receipts stay here and on the RTX 5090 host. No engine/service mutation, rental or model download.

## Fixed source and observation

Input: two independent activations from the full15-value E2M1 set `{-6,-4,-3,-2,-1.5,-1,-.5,0,.5,1,1.5,2,3,4,6}`, independent gate and up ternary weight pairs. Domain9×9×225=18,225. Observed output is exact GPU binary32 bits of

`(g / (1 + __expf(-g))) * u`,

where `g=a0*g0+a1*g1`, `u=a0*u0+a1*u1`. Explicit F32 operations, `--fmad=false`, CUDA `__expf`; this is the selected finite source oracle, not mathematical-real exp or an actual learned-model/scale contract. Signed zeros are retained.

## Coordinates and composition

Source activations and weights become index labels; no individual trit is recovered by the native carrier path. For positive magnitudes .5,1,1.5,2,3,4,6 use exponents0…6; negative adds7; zero uses21. Ternary weights+1/−1/0 use0/7/21.

- Gate-side weight/activation pair labels use primes2/3, each scaled2^-27.
- Up-side labels use primes5/7, each scaled2^-54.
- Producer tables:2×9 floats plus2×225 floats =1,872B.
- First two uniform-K8 TF32 MMAs compute **prime-coordinate labels**, not conventional gate/up values. The source oracle alone computes conventional g/u for checking.
- Native normal-mantissa/exponent extraction creates the next machine state. One candidate composes the mantissas with a native F32 multiply, another with a third TF32 MMA and1/8 normalization. Separate physical exponent roles remain in the carrier key.
- Only the final joint observer returns the source's output bits. It never restores conventional intermediates between the composed instructions.

This is an explicit ISA route on a declared finite domain. It establishes sufficient fibres, not equality with endpoint fibres: the source has630 distinct F32 output words, while candidate images below retain far more distinctions.

## Native evidence

`native_mixed_gu.cu`, `mixed-gu-launch.log`, `mixed-custody.json`, `native-mixed-image.bin` and `mixed-observer-mode{0,1,2}.bin` bind all18,225 inputs,6,998,400 all-four/lane fragment checks, zero fragment errors:

| Machine-state carrier | Reachable keys | Target collisions |
|---|---:|---:|
| Two-MMA joint pair |18,144|0|
| Two MMA→F32 mantissa product + exponent roles |18,144|0|
| Two MMA→third MMA mantissa product + exponent roles |17,125|0|

The third MMA safely merges1,019 states while preserving the final nonlinear observer. The image uses32B records: three64-bit keys,32-bit GPU target and source ID. Observer records are16B:64-bit key,32-bit target, representative input. They are not runtime intermediates of the source network.

## Complete resident readout price

`native_mixed_readout.cu` / `mixed-gu-readout-receipt.json` and `mixed-readout-launch.log` check every input through all four accumulator paths for each carrier:6,998,400 reads, missing0/wrong0. Each timed native path includes source-ID placement, producer reads, conversions, normalization, complete ISA composition, exact-key final reader and output. No intermediate source tensor is required.

| Route | Median | Registers | Resident observer |
|---|---:|---:|---:|
| Two-MMA pair |3.304µs|20|1MiB hash|
| Two MMA→F32 composition |3.520µs|22|1MiB hash|
| Three-MMA composition |3.547µs|18|1MiB hash|
| Direct source SiLU |1.450µs|17|none|
| Full source-map lookup |1.347µs|12|72,900B|

No spills/local/shared/barriers. Native successful-probe maxima12/11/14. Prepared producer1,872B; source inputs/output72,900B each. Static observer construction, table generation, allocations/transfers/validation/warmup are outside event timing; CUDA graphs contain100 complete kernels and7 rotated route rounds. All static state remains charged. This realization loses; its price does not invalidate the structural route or close alternate compositions/real consumers.

## Fourth instruction and terminal Q8

`native_mixed_q8.cu` changes the **declared final consumer**: append a ternary down weight and observe saturating RNE Q8 at scale1, clipping[-127,127]. Domain54,675; source F32-after-down remains a separate stronger control. First three native instructions remain opaque; fourth MMA consumes the third carrier with a signed/zero down-weight label. The final key retains exponent roles.

`mixed-q8-launch.log` records27,993,600 fragment checks, zero errors,41,942 reachable keys. Q8 has zero target collisions. The stronger F32 observer has2,365 collision boundaries; first exact witness key20530, targets+0/-0, source IDs17947/18070. `mixed-q8-F32-collisions.json` classifies all2,365 boundaries as signed-zero only; no nonzero witness occurred. This does not alter the preceding exact F32 witness, and no F32 closure is claimed for the fourth instruction. Final Q8 readout pricing and real-scale/learned-model closure remain open.

Prefix coarsening (`q8-prefix-results.jsonl`): dropping0–17 low mantissa bits while preserving sign/exponent and exponent roles is Q8-safe on the complete native image. Drop17 retains40,348 keys. Drop18 fails90 boundaries; exact witness has targets0/−36, source IDs32060/11453. Larger drops also fail. This is finite-domain observation, not a universal precision bound; no native prefix reader price is established.

## Affine alternate composition

[AFFINE.md](AFFINE.md) owns the native nonuniform-K contraction→C-continuation→Q8 route,2,115-state witness, fused-C/full-tile siblings, exact-key reader and context-parametrized analytic observation. It abandons the prime/two-carrier architecture rather than only polishing its reader. Complete native prices remain behind matched controls; no engine adoption follows. Integer context preservation is proved in `AffineContract.lean`; native numerical conformance remains empirical.

## Boundaries still open

- Native compound discovery is empirical; `Observer.lean` proves the generic fibre/prefix implication, not CUDA/IEEE/SFU semantics.
- Prime routes use uniform-K carrier instructions; the affine route uses nonuniform-K. Neither establishes arbitrary full-K model contraction.
- Scales are1 in the source oracle. Dynamic scale groups, actual trained FFN, CUDA reduction parity, quality, TPS and HTTP benefit are unset.
- Source indices/table preparation are paid coordinates. Successful finite readout is neither a cheap analytic observer nor reusable actual-model performance.
- Structural discovery and complete-region pricing remain separate. Further carriers and their consumers are searched jointly; individual conventional-stage closure is not required.
