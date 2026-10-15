# Binary Bonsai 27B: whole-map search, 2026-10-10

## Result and decision

**A native complete-FFN realization now crosses its hidden boundary directly in the next consumer's coded coordinates**, eliminating global GU/hidden/A8-byte tensors. Its largest measured benefit is atR32; small-row DP4A remains better, and serving outcomes depend on the declared workload and quality as recorded below. No general trained-FFN collapse was found.

A new exact coordinate system reduces every signed-A8 byte to **three native FP4 planes**, rather than four, with a table-free arithmetic encoder. Its original-weight continuation is bit-exact against the same-plan four-plane path. On RTX5090, complete producer/encoder/projection/continuation costs improve down by 13–14%, GU by 6–8%, and R32 head by 5.6%. This is a carrier improvement, not a collapse of the complete FFN.

Whole-model fixed work improves 1.58% at R16 and 0.76% at R32. A separate actual HTTP GSM/HE pair on the newer native engine is **1.15% slower**, with identical task outcomes. The mode remains experimental and opt-in. No multiplier from a projection is credited to model serving.

The complete gated-map search has retained positive small witnesses and bounded negative witnesses. A native compound full-FFN realization now eliminates global GU/hidden/A8-byte tensors, writing the exact three-plane continuation directly. Complete cold R32 cost falls110.336→98.048µs against the best already-three-plane materialized control. Actual-model R32 fixed work improves12.636%; the steadyGSM256 HTTP pair improves3.9469% with239 rather than240 correct answers, while the quick64 HTTP pair loses3.56%. The compound remains experimental, with each quality/workload result retained separately. There is no general impossibility finding.

## Declared start and observer

Checkpoint Bonsai-27B-Q1_0.gguf SHA256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`, 64 layers, D5120, FF17408. Fixed binary ±1 weights retain original FP16 per-output/K128 scales. Input is original K128 A8 with max/127 nearest-even quantization, floating activation scales and typed status/error. The map includes gate/up projection, native SiLU×up, hidden K128 quantization, original binary down, and the stated continuation. Intermediate numeric gate/up/hidden arrays are not requirements.

This follows observer-search PLAN: **sufficiency, closure and cost are separate questions**. Nonlinearity is not used as an obstruction. Small maps are searched as complete numeric truth vectors; trained activation observation is separate from synthetic native pricing. Obligation3052 is not discharged by a faster microkernel or by a change in quality.

## 1. Real activation fibers

Diagnostics observed true normalized input, its actual A8 carrier, gate/up and post-SiLU product on actual layers0/30/61 and 32 Wiki positions127–158. All64 FFN calls per step were counted; graphs/fused GU were disabled, F16 state and F16 KV were explicit. The receiver calculation used each complete trained 5120-output down matrix, not a selected output coordinate.

| Layer | Native hidden unique values/neuron over32 rows | Hidden A8 mean unique codes | Empirical code entropy bits | A8 zero fraction | Hidden energy from g<0 |
|---|---:|---:|---:|---:|---:|
| 0 | 32 | 23.105 | 4.348 | 8.853% | 23.577% |
| 30 | 32 | 25.136 | 4.525 | 4.997% | 44.841% |
| 61 | 31.99994 | 22.242 | 4.281 | 6.475% | 18.366% |

Those are finite-sample statistics, not population domain sizes or minimum-bit claims. The sufficient current down input is the hidden code **plus the shared dynamic K128 scale and status**; g and u themselves can be discarded after the product/producer is computed. A scalar q-only name loses information: actual neuron0 observations sharing q have different effective hidden values at every sampled layer. Frozen-coordinate down delta L2 is .012678/.055668/.273024 at layers0/30/61. This is a scale-blind scalar collision witness, not a collision of two complete reachable q-vectors.

Reduced-code and erasure candidates were measured against the complete down receiver. The 127-state hidden carrier has down normalized squared error .000513/.000942/.001685 across those layers. Lower code cardinality is not declared an exact quotient.

All-layer whole-model Wiki256 tests (one-row pureA8, eager teacher-forced prefix, F16 state/KV) gave:

| Hidden carrier | PPL | KL(Prism||candidate) | Top1 matches/256 |
|---|---:|---:|---:|
| Original255 states / bit-identical control | 19.19753712 | .000478712737 | 250 |
| 127 states | 19.24083290 | .001082910806 | 250 |
| 63 states | 19.04359085 | .003063655119 | 248 |
| 31 states | 19.17575671 | .009788591969 | 245 |
| Erase below .01×K128 maximum, retain original scale | 19.13743701 | .000850965737 | 252 |

Reference PPL19.17802760. No GSM/HE acceptance was run for the lossy candidates; exact three-plane A8 was selected instead. These one-row metrics are not mixed with production chunk32 metrics.

## 2. Complete small maps, with continuations

The bounded exhaustive family has three independently varying signed-A2 inputs, two active hidden features, two down outputs, independent original-style half group scales, complete native-style FP32 SiLU×up and K128 A8 hidden producer: **65,536 GU/down networks on64 states**. A producer-reachable binary-weight embedding at D5120/FF17408 is explicit; it is not asserted to occur in trained Bonsai.

All3,375 single-common-unit-SF E2M1 coefficient triples were searched, granting an arbitrary final decoder. 57,344 networks (87.5%) reject every such dot with endpoint collision witnesses. Fixed dense network16 has60 endpoint fibers and a witness for every carrier. The negative applies only to this grammar: per-K16 scale wiring is outside it.

The8,192 successful networks have shared relative signs on the first two input coordinates across all four GU rows. S=x±y and T=t suffice. A native-value dot (.5,.5,4) creates the compound label (S+8T)/2; a32-slot,256B float2 table completes the whole nonlinear region. It does not reconstruct the source stages. Prepared SF/lane wiring [1,4,16] can instead retain all64 states with one K64 FP4 MMA and a512B endpoint table. DP4A radix4 is a positive control for the same64-state label. No native time is claimed for these toy maps.

For three full A8 inputs a float2 endpoint table is128MiB. A full-width A2 input-preserving table has2^10240 entries. These are prices for those table families, not proof against hierarchical carriers. Exact-additive lifting of independently completed local tables is rejected by a retained nonzero dyadic mixed rectangle; that rejection does not apply to arbitrary hierarchical readers.

Lean kernel checks transport/collision obstruction, radix recovery and the explicit dyadic rectangle. C++ retains exact output bit vectors and checks the census in .56s; it does not claim CPU exp equals GPU SFU bit-for-bit.

## 3. Trained fixed-weight structure and native rejection

Exhaustive trained layers0/30/61 contain no GU duplicate/antipodal whole rows or K128 blocks, including the joint GU pool, and no complete5120-output down-column equality within each original K128. Global gate/up sign disagreement is near50%, but per-neuron correlations are retained rather than assuming independence.

All136 down K128 blocks in each of those layers have **column rank128**, certified by selected original128×128 sign submatrices with nonzero determinant modulo3. Nonzero original half row scales preserve real rank. Thus a local real-linear hidden quotient cannot annihilate a nonzero change in one K128 block while the other blocks and scales are fixed. This directs exact quotient search toward nonlocal/hierarchical carriers. It is not an FP32-rounding or reachable-whole-FFN impossibility claim. `block_rank.py` and `block-rank-layer{0,30,61}.json` retain all row indices, determinants, sign/scale hashes; each complete layer runs in2.1–2.3s.

Every four-output down/K128 tile has eight antipodal sign classes. Class aggregation needs148 signed adds versus508 direct in an abstract count. Two native realizations were priced through original A8 production, grouping/encoding, down projection and original-scale continuation. Best class realization costs28.67/159.74/307.20µs at R1/16/32 versus DP4A18.18/38.66/65.28µs. The shared-atomic grouping version is worse. Same-order original-weight outputs are exact. Reject those schedules, not the class carrier family.

All examined FFN FP16 scale words have low3 and high2 bits zero. Their11-bit code reconstructs the original half exactly by <<3. This is a storage-coordinate witness only; no speedup is claimed from it.

## 4. Exact three-plane A8 coordinate

For q∈[-128,127], take a=q&3, b=(q>>2)&3, c=q>>4 (arithmetic). When |c| is5 or7, set delta=+1 for odd b, otherwise−1; update c+=delta,b−=4delta. Then q=a+4b+16c, and a,b,c lie in2×E2M1. Native planes store a/2,b/2,c/2. No lookup table, no new quantization and no scale conversion is needed.

The reader computes RN(2dot0)+4RN(2dot1)+16RN(2dot2) as the exact integer dot before the original FP16 weight and floating activation scales. Six MMA instructions per K128 replace eight. Lean exhausts all256 signed-byte patterns, including−128; GPU tests cover all codes, errors/tails and480 complete candidate output arrays. Every same-plan three/four result is bit-identical.

Cold complete producer+encoder+projection+common continuation, trained layer0/full-head weights:

| Region | Rows | DP4A µs | Best4planes µs | Best3planes µs |
|---|---:|---:|---:|---:|
| Down | 16 | 42.720 | 34.784 | 30.688 |
| Down | 32 | 69.376 | 49.120 | 42.976 |
| GU | 16 | 67.584 | 63.200 | 59.392 |
| GU | 32 | 118.560 | 83.936 | 77.792 |
| Head | 16 | 374.528 | 339.968 | 339.968 |
| Head | 32 | 733.184 | 425.984 | 403.456 |

All original scales, status/error, split folding and recurrence are paid. Static weight preparation and allocation are separate. Inputs in these microbenchmarks are synthetic F32; real-capture evidence is in section1.

## 5. Compound full-FFN realization

Static original-weight relabeling gives each GU CTA a128-gate/128-up pair for one hidden K128 group. The CTA completes every input K128 accumulation, native SFU SiLU×up, original dynamic hidden max/127 producer and direct ExactA8 digits in the down reader's layout. The continuation uses original down weights/scales and complete split folding, with optional alpha/beta residual. **No global GU tensor, hidden tensor or hidden A8-byte tensor is required** (`next.q=nullptr`). Local shared/register numeric intermediates are an implementation choice, not an externally required boundary. The complete map still performs GU accumulation and native SiLU; no claim of a one-instruction trained-FFN replacement is made.

Latest paired trained layer0 receipts, starting from original input A8 q/sx, include input recoding, full GU, SFU, hidden quantization/direct carrier and complete down:

| Rows | Materialized Exact3 warm/cold µs | Compound warm/cold µs | Selected compound GU/down plan |
|---|---:|---:|---|
| 16 | 76.736 / 81.920 | 75.008 / 79.872 | GU F4G2T8 split1; down F4G2T16 split16 |
| 32 | 103.392 / 110.336 | 93.408 / 98.048 | GU F4G2T16 split1; down F4G2T16 split8 |

Charging the preceding dynamic F32→originalA8 producer as well gives83.936→81.888µs cold R16,110.336→100.096µs R32. R32 current GU16 split4 plus down16 split8 is separately114.688µs cold; its different GU association is not hidden inside a bit-parity assertion. Additional matched complete controls are now retained rather than adding independent stage minima:

| Rows, given original A8 | Four planes warm/cold µs | Best fused DP4A warm/cold µs | Compound warm/cold µs |
|---|---:|---:|---:|
| 16 | 85.152 / 91.904 | 74.848 / 81.664 | 75.840 / 79.872 |
| 32 | 113.920 / 120.832 | 146.528 / 155.392 | 95.136 / 98.048 |

Warm/cold minima may select different control plans; every record survives. R32 wins against the tested complete original-A8 controls in these windows. R16 is not universally faster than DP4A: F32-producer-inclusive compound76.544/81.920 versus fusedDP75.840/79.872. ActualR5 is rejected for promotion: compound60.544/63.232 versus materializedDP38.144/59.136 and fusedDP47.968/55.264, F32-producer-inclusive. No R5 extension is made.

A final complete **joint-two-FP4** control starts at common F32 postnorm input and includes both primary/residual input producers, joint GU, actual SFU, both next producers, complete down and all folds. In its one paired window:

| Region/route | Warm/cold µs |
|---|---:|
| R16 joint, GU16 split1/down8 split8 | 63.552 / 73.696 |
| R16 compound originalA8, selected plan | 95.744 / 100.352 |
| R16 ordered ordinary two-pass reference | 93.280 / 102.144 |
| R32 joint, explicitly two independent R16 tiles | 119.840 / 131.072 |
| R32 compound originalA8, selected plan | 122.944 / 126.944 |
| R32 ordered ordinary wide two-pass reference | 101.376 / 112.384 |

Current joint API supports≤16 rows; the tiled R32 call is not an unsupported joint32 instruction. Both numerical maps keep original weights/scales but dynamic two-plane NVFP4 differs from originalA8. Joint-versus-own-two-pass complete RMS differences4.16e−5/3.37e−5 reflect association plus requantization; A8-versus-twoFP4 RMS differs about.00817/.00795. Prior parent twoFP4 Wiki quality isPPL19.2028/KL.00037166, a separate model operating point. This paired window rejects an absolute compound advantage over joint/twoFP4 and is not mixed with earlier graph-window minima. `whole_ffn_joint_control.cu` and `whole-ffn-results/joint-*` retain source/object pins and every complete price.

Same-plan next scales/DC/status/digits and full trained down are bit-exact, including null q, all256 input-code witness, zero/invalid paths, padded digits, R5/16/17/31/32/65, strided guards and alpha1beta1 residual. Independent raw integer-dot/F32 oracle differences are ordinary association (max6.68e−6,relativeRMS2.25e−7 atR32); native split4 versus split1 can alter the hidden quantizer. Thus the actual-model gate is mandatory for that policy change.

Static GU signs+scales occupy25,067,520B/layer; relabeling itself took67.584µs for layer0. A replacement-only layout has no added static bytes, but the first engine mode retains original GU for small/F32 fallbacks: **added64-layer residency1,604,321,280B (1.494140625GiB)** is explicitly paid, plus separate64-padded input/next digit scratch and down partials. No extra down weight copy is made. Configure/preparation are outside capture; recurring input encoding, output folds and status/error remain inside.

Source anchor639f3e2, opt-in actual integration952efb3, mode `H5_A8_FFN=three-direct-r16`, only preparedR16..32; incompatible fused-GU mode rejects. Other rows retain original paths. HeaderSHA2561d22468bf8923b08206c0d777b97a5d3abf6439f6ba8ec032dab33b1fe2cf7b5; CUDASHA256ab2399c081aba97ad7feb8b37d1be45dc9a2b93c76f6c6a44138a7ed360a6a0e. `WHOLE-FFN.md`, `whole_ffn.{cuh,cu}`, `whole_ffn_bench.cu`, `whole-ffn-results/` own native source/SASS/object pins and every paired receipt. No defaults are promoted before model/task evidence.

## 6. Whole-model evidence and disposition

The unchanged-plan link adapter on source374c77d and actual opt-in sourceb328bf6 both retain the original four-plane split policy. Wiki256 chunk32 PPL/KL/top1 are identical four vs three: prefill19.199563025/.000530691585/251; decode19.174658451/.000224840615/252. Native chat64 prefill and decode remain64/64.

Fixed whole-engine work on374c77d, all heads demanded, F16 state/KV, 16 or32 slots, allocated context512, token198, positions0–63, first32 warm then32 timed steps, fresh-process **ABBA** order:

* R16: mean medians1335.29951→1356.364135 tokens/s (**+1.5775%**).
* R32:1358.67133→1368.99961 (**+0.7602%**).

Newer resource-fixed native engineb328bf6, actual quick64 HTTP generated panel, B32:
**589.6761→582.8942 tokens/s (−1.15%)**, identical29/32 GSM and32/32 HE, caps1/0. Different source and workload, so both results stand separately. The digit mode stays opt-in experimental; no default serving promotion.

Actual compound integration952efb3 executed Wiki256/chat64 off/on successfully, with the expected GU split-association policy difference rather than whole-model bit parity:

| Metric | Same-root off | Whole compound |
|---|---:|---:|
| Wiki prefill PPL | 19.199563 | 19.174918 |
| Wiki prefill KL | .000530692 | .000537197 |
| Wiki prefill top1/256 | 251 | 250 |
| Wiki decode PPL | 19.174658 | 19.177878 |
| Wiki decode KL | .000224841 | .000208546 |
| Wiki decode top1/256 | 252 | 251 |
| Chat prefill top1/64 | 64 | 63 |
| Chat decode top1/64 | 64 | 64 |

Same-binary compound model fixed work on source952efb3 (R32, independent slots, token198, prefix32, context512, F16 state/KV, PDL1,12 timed decode steps, no Dspark, prefill excluded) gives1674.60→1886.21 tokens/s (**+12.636%**),19.111→16.963ms. This is not HTTP.

Matched actual quick64 B32/context8192 HTTP gives materialized615.21→compound593.33 tokens/s (**−3.56%**), GSM30/32→29/32 with one loss/cap, HE32/32 unchanged. The compound is not promoted on that result. The longer selected GSM256 steady pair gives materialized1237.310→compound1286.146 HTTP tokens/s (**+3.9469%**), generation wall118.755→112.386s and146,937→144,545 generated tokens. Correct answers240→239/256 (4 gains,5 losses), caps11→12. This is a measured steady throughput gain at that slightly changed quality, not a fixed-work equivalence or quick-panel win. The mode remains experimental opt-in, not a silent default promotion. Original weights/scales and dynamic input producer are unchanged; source-coordinate grouping/split association changes hidden rounding.

## 7. Further exact-carrier structure

An actual native packet census on source374c77d observed full original postnorm A8 q/sx/DC/status with exact byte comparisons at layers0/30/61, R16/R32. Wiki128 has128 distinct packets/128 observations per layer. Mixed real6GSM+6HE+4chat prompts atR16 yield665 distinct/848 rows (1.275188× row-count compression); disjoint12+12+8 atR32 yield1301/1696 (1.303613×). Every duplicate has the same complete token prefix, and all rows become distinct after positions31/33. No cross-prefix or antipodal packet collision is observed. Identical-prompt positive controls give16×/32× row compression, not native speed measurements. These observations support upstream prefix reuse, not a newly discovered activation quotient after divergence. `row_fibers.md` prices complete keys, comparisons, misses, policy-specific representative execution, gather/broadcast and per-row residuals; none is credited free.

A balanced exact-three-digit census also tests a more sparse continuation: q=a+4b+16c with a,b,c∈2E2M1, choosing c=0 whenever q∈D+4D. All256 signed bytes remain exact; the two-plane set alone covers103 of255 A8 states. Actual captured hidden nonzero-third fractions are6.655%/8.676%/5.722% (8.52/11.11/7.32 entries perK128), compared with16.46%/22.62%/20.93% in GU input. R32 support unions still cover105.34/112.23/78.88 positions perK128; no observed K128 group is third-zero. Thus coarse whole-group zero-MMA skipping has no hits here; the observed K64 third-plane tiles also have zero skip hits for every tested rowtile8/16/32. A two-dense-plane plus sparse integer third continuation was priced natively for down and **loses clearly**: real layer0 captured-hidden cold complete188.416/345.856µs atR16/R32 versus dense exact3 30.432/45.024, best four34.784/49.152 andDP4A40.928/69.632. Bare projection already182.016/337.920µs. All480 timed same-plan arrays and240 strengthened witnesses (including isolated all256 codes through the sparse path, high-density fallback, errors/tails/alpha/beta/guards) pass; Lean covers all256 identities. Original q/sx are preserved. Gathered integer correction and compiler pressure (cfg0 96registers,44B/76B spill stores/loads) dominate; no full-model integration is made. `SPARSE-DIGITS.md` and `sparse-digit-results/` retain exact witnesses, native source/SASS/pins and the loss. `sparse_third.py`/`sparse-third.json` retain the exact256-state table and real-support census. Regrouping32 sequential capture observations measures support, not an actual simultaneous decode batch.

## Evidence custody

Private Halo source owner: `~/projects/halo-5090`, branch `binary-spec-bonsai-5090`; source anchor79612688689e5c9c07088b17df2a2bc23abd3d35, actual native integrationb328bf6. Complete sources/proofs/receipts live at `experiments/binary-whole-map/`:

* `README.md`, `SIGNED-DIGITS.md`, `EXACT.md`, `README-weight-census.md`.
* `ThreePlanes.lean`, `signed-digit-results/exact_a8.lean`, `exact_search.{cpp,lean,jsonl}`.
* `bench/whole_map_capture.cpp`, `analyze_capture.py`, `f16-activation-layer{0,30,61}.json`.
* `engine_adapter.cu`, `engine_fixed_work.cpp`, `run_engine_gate.sh`, `engine-results/` raw ABBA and quality logs.
* `signed_digits.{cuh,cu}`, `signed_digits_bench.cu`, `signed-digit-results/` all native receipts/source pins.
* `weight_census.{py,json}`, `block_rank.py`, `block-rank-layer{0,30,61}.json`, `down_classes.cu`, `down_classes_r{1,16,32}.jsonl` and build receipt.
* `WHOLE-FFN.md`, `whole_ffn.{cuh,cu}`, `whole_ffn_bench.cu`, `whole-ffn-results/` compound native map and full down/residual checks.

Large real captures stay in bounded private campaign data at `~/data/binary-5090/whole-map/`; measured probe binaries are losslessly retained in `measured-binaries.tar.zst` with their original hashes checked before removing the uncompressed duplicates. `engine-results/data-custody.json` records hashes/bytes. Parent owns native model/task-panel receipts at `results/binary-5090/three-digits-native-{keyholes,panels}/` and whole-FFN quality/profile/panels/steady scopes. The complete17,616-row native packet dataset is exported losslessly at `~/data/binary-5090/whole-map/row-fibers/packets.bin.zst`; restored98,670,264B SHA256204e785caecae308fdbc6ba6690a0df81d953311133564616d1c6aff0c7e3df2 was checked before removing the uncompressed duplicate. `row_fibers_results/` retains fingerprints, prompts, exact prefixes and small packet witnesses. RTX5090 sm120a,500W,CUDA13.1.115; all jobs use exclusive `/tmp/binary-gpu.lock` windows under55s. No new rental/public push/wake; shared rental owner exports and destroys at00:28Z.

This case extends the [observer-search plan](../../program-search/observer-search/PLAN.md), bounded whole-map table constructions, and composition transport. It does not close an ISA frontier or a trained-model full-block lower bound.
