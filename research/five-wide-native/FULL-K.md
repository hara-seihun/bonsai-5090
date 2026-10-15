# Full-K virtualized GU and block-Q8

## Latest model-bound closures

- H1 graph-v3 (`95fce9015`/DLL`398a9d2b`) closes the actual model-bound first-butterfly and q4/s4/g4 observer, all64-row logits, independent reference-repeat and warmed graph replay. Both mixed/plain arms have0 endpoint/logit mismatches. Ordinary medians4022.795777plain/4001.247069mixed (40.644%/40.962% loss); deterministic3947.455855/3931.397121 (40.036%/40.280% loss). Halving carrier traffic does not remove the whole-model deficit. No adoption. All ten native children exited0 in182.892s, mutex released, explicit GPU release toPrismML. [H1 settlement](H1-TAIL.md#native-graph-v3-settlement), [result](h1-window-v3-result.json).
- Packed-tail v3 (`d7926585e`): real-model mixed/plain each pass71,303,168 SiLU words,35,651,584q4 bytes,4,456,448s4 bytes and4096g4 words,0 mismatches. Three-repeat B64×128 medians3973.086607mixed/4012.525034plain versus6777.352952 serving:41.377%/40.795% loss, no adoption. Original DLL strict full-model logits remain unset; parent-tail arms in H1's sibling DLL now pass the strict profile. [Retained original settlement](PACKED-TAIL.md).
- First divergence: prefillstep0/layer0 `ab` GEMM cfg12/split-K6 float atomic fold, on identical operands/ordered partials. All720 fold orders cover every observed output word; phase-sync does not remove it. Private witness commit`baa1d8ee0` on `agent/serving-divergence-1014`.
- Unchanged serving with `H5_DETERMINISTIC_GEMM=1`:6585.409637/6583.070035/6579.811334tok/s; median6583.070035,2.866649% below video6777.352952. Fixed-order numerical profile is the future strict repeat-logit reference; ordinary serving stays the TPS baseline and original15 quality cohort stays detunset/distinct. [Exact report/receipt](DETERMINISTIC-SERVING-TPS.md). Nativeexit0, frozenplan unchanged, mutex released and PrismML resumed. No serving binary/plan mutation.

- [CPU cost evidence](H1-COST.md): H1 plain's main K loop contains128static SHFL.IDX sites and32OMMA sites. The caller-bound frozen-WF0 CARRIER=true successor has2288static instruction lines,0SHFL.IDX/0SHFL.BFLY,32OMMA and126registers/stack-local0. It replaces the wrong CARRIER=false2427/136register comparator; original invoice/package remain intact. H1 stack/local0 does not support a spill diagnosis. Recurring construction is search evidence, with no measured latency attribution or prescribed route. [Immutable H1 custody](h1-custody-v1/README.md) retains2366paths/2094objects; archive`h1-custody-v1.tar.zst` SHA`edcd9b3a226cc253b6de90d284c42c3456f4f381467e2e7341217d354bf94acc`. Raw intermediate tensors/warmup full logits were never serialized; saved counters/tokens and observed full logits are retained. The [CPU row-mean absorption successor](gu-scale-absorb-result-v1/README.md) closes one route:64 real fixed rows fail one-plane FP4/UE4M3 exact representation. Its separate RN binary32 K5120 witness still needs native accumulation-schedule binding; serving-loop noncommutation is not established by it. No future-window nominee; full observer/native/TPS acceptance remains unset. the PrismML quality campaign retains the GPU.

Owner: programme12, this five-wide continuation; reporting coordinator00000000-0000-0000-0000-000000000000. Existing RTX 5090 only, sm120a, shared nonblocking mutex. No serving/engine/production-plan mutation or rentals.

## Contract and current result

At the engine's GU geometry **17408 features ×64 tokens ×K5120**, opaque native block carriers and their RZ-aware observer reproduce an actual native FP4 reference on saved synthetic source data:

-356,515,84016-K block labels:0 mismatches;
-1,114,112 post-SiLU F32 words and Q8 bytes:0 mismatches;
-8704 columnwise128-feature Q8 scale words:0 mismatches.

The current rolling route retains one8B mixed register state per output through full K, without a history trace. Its1,114,112 final labels also match the actual native reference's full G/U state BEFORE SiLU, including cases saturation would otherwise hide. Latest complete prices: rolling776.016µs, matched plain streamed777.152, strongest grouped-SIMT1037.272 and ordinary FP4449.560. No FP4 speedup or engine adoption. These are standalone complete-region prices.

Source fields and order:

- Fixed weights are independent ternary G/U rows, five trits per byte in a logical row stream. This is not the engine's installed lane-interleaved dense5 image.
- Inputs retain all16 raw E2M1 encodings/15 numeric values, including both zeros. E4M3 activation scales are independently sampled from finite nonnegative codes0…126 per16 K.
- G/U positive F16 scales are independent per128 K, sampled from exponent fields10…19 and all1024 mantissas. Token FP32 scale follows the full K sum.
- Every128-K chunk performs native64-K accumulation from zero, another native64-K accumulation into that C, then source-ordered `fmaf(partial,weightScale,outerAccumulator)`.
- Final `g/(1+__expf(-g))*u` follows token scaling. Q8 normalizes128 contiguous FEATURES for each token: `amax=max(abs(y))`, inverse127/amax (zero for an all-zero block), round-nearest signed bytes and F32 scaleamax/127.

The Q8 observer implements that quantization suboperation. It does not include preceding Hadamard/RMS preparation, a down projection, a model invocation or installed engine acceptance. All data here is synthetic, including explicit zero blocks/token, at the real projection dimensions.

## Carrier maps

### Signed INT8, complete16-K block

`fullk_trace.cu` uses a full m16n8k32 tile:

- A's first16 slots are G trits; second16 are64·U trits.
- B's first16 slots are doubled FP4 valuesq; second16 are8q.
- One native MMA yields the joint integer `L=g+512u`, with−192≤g,u≤192 and−98496≤L≤98496.

128 independently positioned logical outputs per warp. Four-trit raw-source extraction crosses the five-trit-byte boundaries correctly; the small producer is243×8B and raw activation quartet lookup65536×4B. The physical trace stores one4B joint word per16-K block until the complete K loop ends. At full dimensions this is1,426,063,360B, before its final observer reads it again.

`fullk_rz.cu` repairs the observer's native arithmetic. `fullk_compose.cu` adds stronger fused controls and a compressed128-K trace. `fullk_compose_cold.cu` prices full dimensions, cold weights and the lookup control.

### FP4 with data-carrying A scales

`fullk_fp4scale.cu` produces the same L without raw-FP4→INT8 conversion. In a native m16n8k64:

- first16 slots contract G trits against original FP4 values;
- next16 contract **2·U trits** against the same values, with **A scale256**;
- remaining32 slots are zero, B scales of the active quarters are1.

The FP32 result is exactly L/2 in the sampled physical checks; conversion to integer L is exact. A scale values1 and256 (encoded bytes0x38 and0x78) are computational coordinates, separate from the logical activation-scale context deferred to the observer. The same full-shape physical/F32/Q8/scale checks pass. This supplies a different native instruction/scale map, not an FP4 adoption claim.

###128-K mixed-word trace

After the exact native128-K rounded partials are obtained, letgb/ub be their32-bit words:

```
lo = gb XOR rotl(ub,13)
hi = ub XOR rotl(lo,7)
C128 = hi:lo
```

Neither stored word is an ordinary G/U float. The inverse is exact. Only40 such8B labels per cell are needed through full K; the complete trace is356,515,840B, four times smaller than the16-K trace. The final observer reads those labels in original128-K order, applies independent weight scales/FMA and the block observer. All source and terminal witnesses remain identical.

### Packed dyadic/RZ consumer

`fullk_packed_rz.cu` replaces each group's two float scale products with one paired integer calculation. E4M3 scales have integer unitsS=512·scale, at most229376. The label's biased G/U digits are placed32 bits apart and the entire64-bit pair is multiplied byS. Each term's digit product is at most88,080,384; eight biased sums stay below2^30, so no low32-bit carry corrupts the high field. Bias is tracked explicitly.

After four terms, signed fields are truncated to24 significant bits toward zero and repacked; the next four terms accumulate into that carrier. The final conversion is round-toward-zero times2^-10. This reproduces the two native64-K boundaries, then the F16/FMA128-K boundary. Compressed producer56regs versus the preceding72; streamed native64regs versus96; no spills. The packed consumer is also tested in the grouped-SIMT control.

The original streamed variants maintain ordinary G/U accumulators across K and serve as strong controls.

### Rolling whole-K label

`fullk_rolling.cu` replaces the40-label history with a joint state:

```
lo = gb XOR ub
hi = ub XOR rotl(lo,7)
C = hi:lo
```

After each128-K rounded native prefix, an ordered FMA update is conjugated by that code. The update transiently decodes F32 operands for the two required FMA operations, then re-encodes their result. These operands exist; this is not an all-tensor/no-arithmetic-readout claim. The persistent state crossing subsequent native producers is mixed64-bit data, and only the final observer reads the completed state.

Volatile bitwise assembly retains the map at actual machine boundaries. Executable SASS around0x4b30…0x4c20 mixes/XORs/rotates register pairsR8:R9,R26:R27,R28:R29,R30:R31 before the loop backedge0x4c20→0x290. Decode occurs after full K as well as transiently inside updates.64regs, no spills; no intermediate global trace in the timed rolling route.

The native reference independently exports its final G/U F32 words under the same label. All1,114,112 complete labels agree, in addition to all terminal F32/Q8/scale words. `fullk-native-final-labels.bin` and `fullk-reference-final-labels.bin` share SHA256`ae497ef9fd8094e36776c46879f7c4eee8a763b3497064f42b803b20619a0446`.

`fullk_observer.cu` additionally explores the final observer's saturation fibres. For finite G/U, zero G or an exp result0 bypass division; exp infinity gives exact signed zero.477 of1,114,112 sampled states need ordinary division (zero G25472, exp zero544266, exp infinity543897). Every terminal bit still matches the original unmodified-division native oracle. Matched ordinary/fast FP4 prices449.560/449.632µs show no complete-route gain: this observer shortcut does not fix the producer bill.

## The actual rounding turn

The initial observers assumed ordinary nearest-rounded F32 sums. At M512, three terminal F32 mismatches remained despite matching Q8 bytes/scales. A diagnostic instrumented480 native64-K accumulation steps from the actual FP4 instruction. Exact sums rounded nearest failed3 times; exact sums rounded **toward zero** matched all480.

`fp4-rounding-invoice.json` retains the three failures. Example:

```
terms=[0.095703125,0.2548828125,0.203125,15456]
C=1784.421875
exact=17240.9755859375
nearest=17240.9765625
native=toward-zero=17240.974609375
```

The corrected observer rounds once after the first four16-K scaled sums, and again after adding the next four to that C. Scaling those exact dyadic terms by1024 gives signed integers below2^29; integer significant-bit truncation implements the observed boundary without FP64 arithmetic.

On full M17408, nearest/tree observers miss216–229 terminal F32 words and7–8 **Q8 scale words**, while every Q8 BYTE still matches. The full observer contract therefore includes scale words. Integer/RZ, exact-double/RZ and the actual FP4 reference match all retained terminal bits.

This is empirical conformance for the tested sm120a mxf4nvf4 schedule, not a universal rounding law for all MMA families.

## Same-K complete pricing

CUDA graphs, seven rotating arm rounds; full size uses four complete operations per graph. `fullk-prices.json` retains every event sample and median. Input/output transfers, initial validation, source-record generation, static table construction and weight preparation are excluded. Dynamic source conversion, all native instructions, carrier writes/reads, reduction, SiLU and Q8 normalization are timed.

Weight copies rotate across three equal source images: dense5 weights total106,954,752B (>96MiB L2). Lookup's prepared quartet-code images total133,693,440B; its complete81×65536 dot4 table is5,308,416B. All arms compute the same saved values. Inputs/scales and the small producer tables are resident. No hardware-counter utilization assertion follows.

| Route | Initial INT8 µs | Data-scale FP4 µs | Packed/RZ FP4 µs |
|---|---:|---:|---:|
| Complete16-K opaque trace + observer + Q8 |2565.616|2239.352|2244.320|
| Grouped-SIMT opaque trace + observer + Q8 |3142.376|3148.872|3144.944|
| Strong grouped-SIMT fused + Q8 |1127.512|1124.688|1040.032|
| Per-output prepared-weight DP4A + Q8 |22793.031|22791.281|22728.201|
|128-K mixed opaque trace + observer + Q8 |1665.648|1023.552|1017.048|
| Streamed native partial carrier + Q8 |896.728|875.456|774.600|
| Ordinary native FP4 reference + Q8 |451.632|451.192|451.040|
| Complete prepared dot4 lookup + Q8 |27247.434|27239.807|27213.080|

The full opaque history traces and the current streamed/rolling implementations are retired as FP4-speed candidates. The full-K/RZ structural closure and rolling shared-state witness survive those losses. The two small-shape gains are not engine evidence. [Executable-bound static invoice](fullk-sass-invoice.json): packed/RZ compressed producer changes200 FMUL to8, while both native coupled producers execute eight MMAs per128-K loop for16 features; ordinary FP4 uses two for8 features. This scopes the current carrier's extra tensor work, not arbitrary virtualization.

## A native scalar-carrier falsifier

`scaled_word_fixed_weights.cu` uses the **same fixed weight matrix and scale contexts** for both inputs. Only one raw activation changes0→0.5; a zero-weighted activation6 keeps the16-K scale unchanged. The first16-K block has U=43008 and G=0; the next contributes G=0 or1/1024. Logical scale codes are126 and1, weight scales1. All remaining K5120 positions and127 features are zero.

A native scalar FP32 `g+512u` map combining those two differently scaled blocks returns the same word1269301248, value22020096, for both inputs. Actual native reference plus128-feature block observer gives:

| | Input0 | Input1 |
|---|---:|---:|
| G |0|0.0009765625|
| U |43008|43008|
| SiLU(G)·U |0|21.0102539|
| Q8 block scale |0|0.165435076|
| First Q8 byte |0|127|

No observer of that scalar and the unchanged contexts can separate the required outputs. This kills that declared single-scalar linear-family shortcut across unequal scales. Multiple accumulators, residuals, other labelings and consumer-specific quotients remain open.

The first unrolled single-lane-output probe produced unreliable first-variant values. Its executable reveals the cause: the compiler sank the first two volatile MMAs beneath a lane!=0 branch, so only lane0 reached them. Unwritten output registers retained the scale-operand bits0x38383838/0x78387838. This violated full-warp MMA participation.

A memory-clobber successor still fails. Explicit `__syncwarp()` immediately before AND after the inline MMA repairs the unchanged original single-lane probe; SASS now places both MMAs before the lane0-only observer branch. The fixed-weight all-fragment witness also passes with those guards, with identical physical/observer binaries. `collective-convergence-invoice.json` binds failed/guarded executables and placements.

`fullk_collective.cu` applies the guards to all native helpers and repeats full-shape conformance: all saved terminal bytes/words remain identical. Its prices are compressed1023.064, plain streamed775.624, grouped-SIMT1045.656 and FP4444.152µs. Later rolling/observer sources keep these guards. All32 lanes must still call each collective; volatile assembly alone does not encode that convergence requirement.

## Proof and custody

`FullKCarrier.lean` builds in Lean4.33 without sorry. It proves signed label injectivity/bounds, operand bounds, E4 unit bounds, paired multiplication/no-cross-field-carry under its stated bound, reversible mixed words, and ordered observer composition conditional on primitive conformance. It also proves the generic consequence of a collision: equal producer labels plus distinct target outputs exclude an observer factoring through that producer. Native witness hypotheses remain empirical. The proof does not certify CUDA, floating rounding or the full engine. `fullk-lean-receipt.json` binds the compiler and exact source.

- `fullk-oracle-custody.zip`:1,542,429B, SHA256`cd2b1848fd1b3846a7be3feef28905819184fe652290fe2a1acdaf9865dff6c4`; initial observer failures and native rounding diagnostic. Readable `retained-fullk-oracle/`.
- `fullk-map-custody.zip`:52,526,170B, SHA256`c225ab1c6af1b03f9873484724af97fc80303ad586ab2de56890c3f31089017c`;10 variants,345 bound artifacts,105 deduplicated objects. Exact sources/executables, source inputs, all terminal bit witnesses, event samples, build/SASS, proof and rounding invoice. Readable `retained-fullk/` uses immutable hardlinked objects.
- `fullk-fixed-weight-custody.zip`:127,907B, SHA256`f3e54acf122c7db5af4e1301b0e6453cd7a92bfa9f0e00df899d72ffe83cf346`; fixed-weight native falsifier, all-fragment/block-Q8 witness, SASS, exact LF Lean source/receipt and the documentation snapshot. Readable `retained-fullk-fixed/` (13 artifacts). This preserves exact compiled Lean bytes independently of Windows text newline conversion in the full bundle.

- `fullk-collective-custody.zip`:49,516,379B, SHA256`854bdc0d93163db7b3b101586e697984830763fe2981a2bbf5893a8035fe18c0`; failed memory-clobber probe, guarded original/fixed-weight probes and guarded full-shape kernel;72 artifacts in `retained-fullk-collective/`.
- `fullk-rolling-custody.zip`:57,807,623B, SHA256`ab53850dfff75b435e392658d34a884168fa69864ab33f0f5146352c3798fd3a`;49 artifacts in `retained-fullk-rolling/`, including both complete1.114M-label witnesses.

All retained archive/artifact hashes were checked after export without replaying benchmarks. Working1.426GB/356MB GPU traces are temporary; successful receipts preserve full physical comparisons and the source/terminal witnesses needed to reproduce them. No GPU job or mutex remains held.

## Continuing target

Find a genuinely cheaper coupled carrier/consumer for the complete K/scale/observer region against ordinary native FP4 and the strongest matched controls. Full-K arithmetic closure is present; the cost is the open gate. Installed engine adapters, actual-model prompts/quality/TPS, full down-projection K and preceding normalization/Hadamard are separate acceptance obligations. No scalar-capacity argument here excludes arbitrary whole-ISA virtualization.

## Cooperative producer and fused observer — complete-region win

`fullk_coop.cu` closes the same M17408/N64/K5120 synthetic source/terminal contract and changes the price: **308.880 µs mixed-label fused versus449.744 µs ordinary FP4+observer**,31.3% lower latency. Matched plain-fused is308.312 µs; label mixing contributes no isolated speedup. Producer reuse and observer integration earn the gain. Naive fusion471 µs and occupancy-only fusion468 µs retained their losses.

Four lanes cooperate on each64-K dense5 row: four code bytes per lane become an80-bit nibble segment, then warp gathers emit the two native operand slices.40registers/no spills, three512-thread CTAs/SM. The timed kernel produces SiLU/Q8/scales directly; intermediate F32/label matrices are written only for validation. All1,114,112 mixed final labels and F32/Q8 outputs plus8704 scale words match. Native/reference/fused labels share SHA256`ae497ef9fd8094e36776c46879f7c4eee8a763b3497064f42b803b20619a0446`.

`Cooperative.lean` builds: all61 possible nibble starts over arbitrary320-bit input agree with the32-bit reference window. This is a bit-wiring contract, not CUDA load/shuffle/MMA or floating-point correctness. Native receipts carry those observations.

Custody: `fullk-coop-custody.zip`,2,069,599bytes,228artifacts,SHA256`49485dc8d87ada3d76959bc5f8fe294e186372c2b998aec864a06ee41f0ef30f`. This delta uses already hash-verified content-addressed objects from preserved earlier archives. `retained-fullk-coop/REQUIRES.json` names each required hash; every materialized artifact is locally verified.

coordinator assigned engine integration after export. Fresh writer `~/work/clones/fullk-gu-coop` is active on the current serving source049cd7db. The serving contract differs: row-mean scales with native C carried through all K, then SiLU/sign/1024-Hadamard/FP4. Initial mixed/plain opt-in engine arms preserve that arithmetic and FP4 consumer while importing cooperative dense5 production and fused SiLU. The prototype's128-feature Q8 normalization cannot replace the engine consumer without changing model arithmetic. Model TPS/logit acceptance and actual FP4-tail fusion remain in progress; serving binary and frozen plan are untouched.

## October14 actual-engine adapter outcome

Cooperative production plus SiLU observation is integrated behind `H5_GU_COOP` in private halo-5090 source commit `31ad74a9a8253e8b6016613994cde33bf18c5c64`, branch `agent/fullk-gu-coop`. Unset keeps serving. Production executable/DLL and frozen plan are unchanged. The existing sign/1024-Hadamard/FP4 down-preparation remains; the synthetic128-feature Q8 observer was not substituted. Full FP4-tail fusion remains unset.

Matched B64x128 one-pass model screens (tok/s):

| Route | Mixed | Plain |
|---|---:|---:|
| Standalone cooperative v2 |3021.248092|3039.172905|
| Actual TMA/ring/Tok geometry v4 |4075.016112|4087.284478|
| Fixed-window/shared-book v5 |3804.040288|3813.166568|
| Fixed-window/global-book v6 |3701.358058|3723.923613|

Serving:6782.901128 before,6788.274451 after; candidate flag-disabled:6780.934911. The best ring port is approximately39.8% below serving. No model-level speedup or adoption. V4 remains the live experimental source; v5/v6 are retained negative successors.

Same-input real-model GU checks: v4 mixed and plain each compare71,303,168 F32 words over all64layers,zero differing words/max error0/nonfinite differences0. This closes the observed GU arithmetic boundary. It does not establish full-model determinism.

Independent-process teacher-forced checks retain8steps/64rows and all248320 logits for rows0/17/33/63:7,946,240 words per arm. Predeclared max1e-3/mean1e-4/zero-top1-difference acceptance failed: serving-repeat max2.2648/mean0.2003/15of512 top1 differences; disabled2.0290/0.1977/15; mixed2.6567/0.1940/18; plain2.4059/0.1954/19. A one-row-per-slot, position-synchronous prefill control still fails on the serving binary repeat (max2.3275/mean0.1973/17). All are finite. Root cause and whole-model exact/bounded acceptance remain unset; the controls prevent attributing those differences to the adapter. Probe API state reports Q8 recurrence/FP8 KV; retain its environment and distinguish it from CLI-specific serving controls.

v1 startup access violation is retained; v2 moves table upload after stream creation. V3's full catalogue rebuild exceeded240seconds and is not a successful build. V4's source-shared sidecar reuses unchanged objects and builds in16.95seconds. The generic61-window and optimized eight-window wiring contracts compile in Lean for arbitrary320-bit input. This proves bit wiring, not CUDA scheduling or F32 behavior; optimized wiring lost in the observed engine.

Initial engine custody: `fullk-engine-custody-v1.zip`,299,757,431bytes,516artifacts/221content objects, SHA256`d2b6b74895d043ac563daa431540997614c49f547c36145375e2419da0ff91f3`. Every object/manifest reference is locally hash-verified under `retained-engine-v1/`; `engine-custody-verified.json` is the receipt. It contains v1/v2/v4/v5 sources, builds, outputs and failures. Later v6/synchronous-control/SASS custody is a separate delta, not a mutation of this archive.

### Final native custody and resource invoice

Delta `fullk-engine-custody-v2.zip`:134,148,104bytes,262artifacts/61newobjects/100priorobjects,SHA256`df0b25fbc07a672329908d227c33f952e0d3f0cee0523ebf108bceef811e0833`. Locally verified under `retained-engine-v2/`, with prior objects hardlinked from verified v1; receipt `engine-delta-verified.json`. It binds v3 timeout/source, v6 build/model output, synchronous-serving controls, final baseline confirmation and retained SASS failures/successors.

FW4/Tok64 compiler resources, plain/mixed: v4 224/237registers; v5 206/220; v6 206/218. All report zero stack/local bytes and1024static shared bytes (dynamic ring storage additional). Reduced register counts did not produce a model win. Native SASS and resource outputs are retained per version; first dump failed because `NVDISASM_PATH` named the executable rather than its directory. Named `sass-v2` uses the directory plus child-only PATH and extracts both mixed/plain functions successfully. Earlier failure receipts remain unchanged.

Finite native workers are complete; no GPU rental/service or serving rollout was created. Private source is retained in halo-5090 branch `agent/fullk-gu-coop`. Full FP4-tail fusion and whole-model strict logit acceptance remain open.

Selected-config scope correction: frozen GU R64 uses cfg13 FW8/WG2/Tok32. FW4/Tok64 resource/SASS extraction is a comparison, not the selected model-launch invoice. `engine-selected-resource-invoice.json` binds the actualcfg13 attributes from retained native resource files.
