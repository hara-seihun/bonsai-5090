# Packed-op / scale-fold hotspot ledger

Source snapshot inspected at assignment base `8640b89` on `tune/batch2`.
Concurrent production changes are owned by the campaign's other workers; this
subtask changes only benchmark/proof/docs. Counts are local to the stated loop,
row or CTA; there is no extrapolated whole-step speedup.

**Evidence levels:** native means a compile-only source copy's SASS was inspected;
core count means arithmetic instruction replacement before conversions, reductions,
loads and scheduling; algebra means a proposed representation change, not a
measured instruction saving. The benchmark's GPU timings are still to be run.

## Decode: actual native transformations

`packed_ops_decode_probe.cu` copies the relevant transforms and provides explicit
PRMT/LOP3 alternatives. Reproduce without a GPU:

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -cubin bench/packed_ops_decode_probe.cu -o /tmp/packed_ops_decode_probe.cubin
cuobjdump --dump-sass /tmp/packed_ops_decode_probe.cubin | c++filt \
  > /tmp/packed_ops_decode_probe.sass
```

| Engine site | Recurring change / local count | Status |
|---|---|---|
| `src/fp4.cu`, WF0/WF2 `gemm_fp4_body` trit decode | Each of 8 input words/lane/group/K128 emits 1 shift + 2 LOP3 masks = **24 native transforms**. Pre-expanded WF6 removes those transforms, but reads 16 operand words instead of 8. | WF6 already present; not a new LOP3 win. WF0/WF2 already use the cheap mask decode. |
| Same function, WF2 `sb4` scale byte broadcast | Two extracted/broadcast scale bytes: **2 SHF + 2 LOP3 + 2 IMAD → 2 PRMT**, saving **4 recurring instructions/lane/group/K128**, if selectors are kept outside K. | Native compile-copy candidate. Selector setup is 3 instructions in each tiny helper, excluded from recurring counts. WF6 already loads broadcast words, so no win there. |
| `lowbit_registers2`, WF5 | Per word, current `(r&mask)|ones` and shifted version emit **shift + 4 LOP3**. Explicit LUT `0xea` (`(a&b)|c`) emits **shift + 2 LOP3**. Saves **2/word**, **16/lane/group/K128** for 8 words, with constants hoisted. | Native compile-copy candidate. CUDA13.1 did **not** automatically fuse this expression. Proposed helper materializes `0x11111111` once with **HFMA2**, not MOV; a full-kernel build must keep that outside K. |
| `lowbit_registers3`, WF4 | Its multi-mask ORs are also ternary-logic candidates, but the three-source expressions have different shift/constant needs. | No numeric saving assigned without that expression's SASS. Do not reuse WF5's count. |
| WF0/WF3 K128 FP16 weight-scale epilogue | For each lane/m/n, 4 FP32 FFMAs after the two MMAs. WF2 puts the weight-group scale in MMA's scale-A operand and removes those **4 FFMA/lane/m/n/K128**. | Already present in WF2; not a new norm-scale fold. WF6 preserves WF2 numerics/layout intent. |

Native copies exclude common LDG/STG/addressing and do not prove that the
production register allocator has identical lowering. The proposed transforms
compute exactly the existing bit maps; packed integer *dot* instructions do not
perform these masks or bit permutations.

## Norm / quantise: real map changes

Let `r=rsqrt(mean(x²)+eps)>0`, `w` be the learned per-feature weights,
`u=H(sign*w*x)/sqrt(1024)`, and N the row width. H is linear; w is not uniform.
For homogeneous dynamic row quantization away from its floor, quantizing `r*u`
has the same codes and block scale bytes as quantizing u, with row scale multiplied
by r. The FP32 row scale is already consumed in the GEMM epilogue (`AG`); forcing
arbitrary r into UE4M3 block scales is unnecessary and would add scale rounding.

| Engine site | Proposed saving / scope | Already done / remaining work |
|---|---|---|
| `k_prep2` in `src/kernels.cu`, and `k_prep_chunk` in `src/prep_chunk.cuh`, `P_NORM` | Replace `v *= r*w` with `v *= w`, prepare/quantize u, publish `AG=r*g_u`. Removes **N per-feature multiplies/row**, adds one row-scale multiply conceptually. At D=5120 that is 5120 removed vs 1 added before accounting for replicated execution. | r is still multiplied into each feature before H in these source paths. The learned w multiplication and sum-of-squares reduction remain. |
| Fixed-g `k_prep_chunk` path | Keep `AG=g_fixed`; use **effective `ig=r/g_fixed`** in both `target=bm_u/6*ig` and `rs=ig/sv`, then quantize `u*rs`. Removes N per-feature r multiplies, adds evaluation of the effective ig at its existing scope. | No per-16-block r multiply is required. In the current warp implementation this scalar expression executes in each producer lane: up to **32 lane evaluations per 1024 chunk**, not one physical evaluation per row. Preserve the fixed-g output contract. |
| `TAIL_NORM` in `src/tail_prep.cuh` | It already computes H on unnormalized values and includes r in g, but `tail_quant` still multiplies all N values by r before multiplying by rs. Quantize u with g_u, publish r*g_u to remove **N feature multiplies/row**. | Commuting through H / row max is **already done** here; removing the feature multiply is not. This is a genuine remaining fold. |
| Norm weight × sign | Precompute their product at model load, replace two feature multiplications with one; **N arithmetic multiplies/row** removed, plus **N sign values** no longer loaded (N/4 float4 source loads, not N native load instructions). | `TailArgs.mul` already stores the combined norm-weight/sign multiplier for `TAIL_NORM`; ordinary `k_prep2`/`k_prep_chunk` still have separate norm and sign arrays. |
| `P_OUT_NORM`, including alpha/beta consumers | The FP32 normalized output consumer still needs its normalized values. | AG folding removes r only on the quantized branch, not on the independently materialized normed-row branch. Do not count its N output-normalizing multiplies as gone. |

Finite-precision Hadamard order, FP32 multiplication association, scale-ceiling
rounding, the **1e-30 row-scale floor**, and fixed-g UE4M3 saturation are parts of
the real map. They prevent declaring the algebraic fold bit-identical. Keep it
selectable and record changed codes/scales/quality. In particular `r*max(am/C,eps)`
and `max(r*am/C,eps)` are not the same at the floor.
`fold_homogeneous_scale` in [packed_ops_swar.lean](packed_ops_swar.lean) states the
conditional algebra explicitly; it makes no IEEE rounding assertion.
A nonuniform learned diagonal w cannot be moved through H as a scalar. Folding
w into weight columns would require transforming/repacking the actual weights,
and then paying for their changed quantization/scale layout.

## Packed floating/integer work: boundary-aware counts

| Site | Arithmetic-core replacement / scope | Boundary and current-state accounting |
|---|---|---|
| Plain `gdn_body`, CG4 (`MC=32`), new-token loop | Per thread/new token: Mk dot 32 FFMA, state update 32 FMUL + 32 FFMA, Mq dot 32 FFMA = **128 scalar arithmetic instructions → 64 packed** (16+16+16+16), saving **64 core instructions**. | Half/BF16 accumulation changes the recurrence and reduction. This count excludes packing/conversion and horizontal sums; packed state must stay packed across tokens. BF16 **storage** alone is already present, while this baseline's working m is FP32. Another worker owns the production packed-GDN option. |
| Same loop, replay token | Without replay-delta shortcut: **96 → 48** core instructions/thread. With replay-delta already supplied: **64 → 32** for state update only. | Replay omits Mq output; a retained delta can also omit Mk. Do not count new-token savings on every replay row. |
| `H5_GDN_SCALED`, `gdn_body` | Its scaled state already reduces state update from FMUL+FFMA to one FFMA/element. Packing the 32 updates gives **32 → 16** core instructions/thread/token. | Do not add both the removed decay multiply and the unscaled 64→32 update saving. Renormalization/materialization still has its own cost. |
| Hadamard in prep / `tail_hadamard` | At each of 10 stages, N scalar add/sub operations → N/2 packed add/sub operations, if matching elements remain in packed registers. For a 1024 chunk: **10240 → 5120 arithmetic instructions**, excluding shuffles/sign selection/conversions. | Cross-lane shuffle topology and packed-lane pairing must match each stage. New FP16/BF16 rounding boundaries. Not an instruction count for a production variant that does not exist yet. |
| NVFP4 value × rs in quantise | Two resident packed elements: **2 FMUL → 1 HMUL2**. Row arithmetic-core saving N/2 multiplies. | FP32→half2 packing costs at least one extra instruction/pair and adds rounding; converting just for this one multiply cancels the minimum local issue saving. Keep an entire region packed to make this meaningful. |
| RMS sum-of-squares | Two squares accumulated in packed lanes: **2 FFMA → 1 HFMA2**; arithmetic-core saving N/2, then horizontal reduction. | Packed partial sums change accumulation precision, and may overflow for large inputs. Keep FP32 reductions unless their numeric option is separately measured; half/BF16 *input* storage is not a packed FP32 accumulator. |
| `src/f07.cuh`, Degree8 SiLU/product | Existing approximation uses **4 HFMA2 + 3 HMUL2 per two features**, plus conversions and FP32 tail. | Already implemented. Benchmarking HFMA2 does not newly remove the exact SiLU's exp/divide or validate the approximation. |
| DP4A side work | A prepacked signed-int8 four-element dot needs one DP4A instead of four scalar IMADs: **4→1**, saving 3 arithmetic instructions/dot. DP2A gives **2→1** for signed-int16 × signed-int8 pairs. | The current `k_gemm_t2` uses **integer tensor MMA**, not a scalar IMAD dot. No 4→1 saving exists in that loop. FP4 operands / ternary two-bit words are not ready-made DP4A bytes; account for repacking, activation quantization and accumulation scale before choosing a side path. |
| Guarded 15×2 INT32 add/sub | Two scalar adds → IADD+mask (**2→2**, no instruction saving). Two scalar subtracts → OR+IADD+mask (**2→3**, one worse). | Meaningful as a register-packed baseline, not a claimed issue win. Existing `src/f1_swar.cuh` instead uses eight bounded biased nibbles with proven-by-domain no-carry intermediates; do not conflate that narrower map with generic integer SIMD. |
| `src/draft.cu`, top-k / bitonic | **No arithmetic instruction saving assigned** from H2/B2/DP4A/DP2A. Ordering uses FP32 value comparisons plus exact token-index tie breaks; FP16/BF16 collapses distinct logits. | `n=2^p` bitonic sorting has `n/2 * p(p+1)/2` comparator pairs (256:4608, 1024:28160, 2048:67584). The costly operation is compare/swap plus shared-memory/barrier work, not a dot or fused arithmetic. PRMT cannot fetch its cross-thread partner. Packing values+indices into an ordering key is a different algorithm with its own cost. |

The immediate implementation candidates with direct native evidence are WF2
PRMT scale broadcasts and WF5 explicit LOP3 decode. The largest algebraic row
saving is the global RMS factor fold. Packed GDN becomes an end-to-end candidate
only with its conversions/state format and measured model quality included.
