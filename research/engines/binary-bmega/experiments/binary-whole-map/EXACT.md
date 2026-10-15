# Exact binary whole-map search

**Decision:** do not measure a single unit-SF FP4-dot replacement for the general gated region. It fails an exhaustive, explicitly bounded whole-map grammar. Prepared block scales can repair the information loss, but the resulting full-input label plus table has no affordable full-width lift. The successful small construction is retained as a search seed, not a Bonsai speed result.

Start: one producer-reachable activation vector, fixed binary GU/down weights and original-style per-row/K128 half scales. End: the exact FP32 bit words of two down outputs after SiLU×up and the entire next K128 A8 producer. Neither GU nor the hidden values are required intermediate states.

## Domain and observer

Three independently varying signed-A2 coordinates `x,y,t ∈ {-2,-1,0,1}` give **64 reachable inputs**. `x,y` are in one input K128 group; `t` is in another. Two active hidden features have:

| Feature | Gate group scales | Up group scales |
|---|---|---|
| 0 | 1/8, 1/2 | 1/2, 1/4 |
| 1 | 1/4, 1/8 | 1/8, 1/2 |

All scales are exactly representable FP16. Dynamic input scales are one. Each of the four GU rows independently chooses all eight binary sign patterns on the three variables. Each of two down rows independently chooses all four binary sign patterns on the two active features. Down scales are 1/16 and 1/8. Thus the census contains **65,536 complete networks**, not independently chosen gate/up states.

The other 126 hidden features in the observed K128 group are zero. The observer performs `h=SiLU(g)*u`, `m=max(abs(h))`, `sx=m/127`, RN-even quantization `q=round(h*(127/m))`, then the binary down integer dot followed by its original scale multiplication/FMA. A zero block produces zero q/scale. All enumerated inputs and intermediates are finite; invalid-status paths are outside this search domain.

This small family can be embedded in **D5120/FF17408/K128**, without assuming unreachable A8 states: duplicate each varying input coordinate, put `+127,-127` anchors in each nonzero input block, and set paired variable signs equal on active GU rows. Halve the listed group scales to compensate the doubled integer dot. Both anchor signs are equal and cancel. On inactive hidden rows, opposite signs on each duplicated coordinate give gate zero; every other input is zero. The original A8 producer then has scale one and returns precisely these bytes. Remaining hidden groups are zero, so their quantizers/down contributions are zero. This is a constructed binary-weight family, **not the trained Bonsai matrices**.

`exact_search.cpp` evaluates host FP32 `fma`, float `exp` and RN-even `nearbyint`, with exact numeric bit-vector keys. It is not a claim of CPU/GPU exp bit parity. The semantic carrier is a half-integer dot, converted to its exact doubled integer label; it does not exploit undocumented native rounding fingerprints.

## Exhaustive one-dot grammar

Carrier `C(x)=a*x+b*y+c*t`, with one common unit scale and each coefficient in the native E2M1 value set:

```
{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```

The search uses doubled integer coefficients, preserving the actual numeric values rather than canonicalizing equal fibre partitions. It includes **all 3,375 triples**, with zero coefficients. It grants an arbitrary downstream decoder and any final output relabeling: a rejection is stronger than rejecting a particular lookup implementation. Post-dot shifts, masks or arbitrary unary readers cannot repair a retained collision. Multiple dots, per-K16 SFs, arbitrary input relabelings and a carrier that also retains the input are different grammars, not covered by this negative.

Results, in **0.55 seconds** on this host:

* **57,344 networks (87.5%) reject every carrier.**
* 8,192 networks admit 52 carriers each; 104 carriers admit 4,096 networks each, and 3,271 admit none.
* No unit-SF E2M1 coefficient triple is injective on all 64 inputs.
* 16,384 distinct complete numeric truth vectors occur.
* Successful networks are exactly those where all four GU rows have the **same relative sign on x and y**. The exhaustive characterization check has zero counterexamples.

Fixed network **16**, sign patterns `[g0,u0,g1,u1,d0,d1]=[0,0,0,1,0,0]`, has 60 endpoint fibres and rejects all 3,375 carriers. Pattern bit one means +1, zero means −1; bit0/1/2 correspond to x/y/t. `exact_search.jsonl` contains its complete 64-point output truth vector and a distinguishing input/output pair for **every** coefficient triple. Candidate records retain first failures over the whole census as well.

For example, carrier `(x+y+8t)/2` merges `(-1,-2,-2)` and `(-2,-1,-2)`. This network returns first-output bits `3191968630` and `3191171212` respectively (−0.189039081 versus −0.177156627). Their inputs share a gate/up producer, yet the final observed map still separates them.

## Reusable structured construction

When the four GU sign rows share the x/y relative sign, the whole region depends only on `S=x±y` and `T=t`. For the positive relative sign:

```
S ∈ [-4,2], T ∈ [-2,1]
P = S + 8*T + 20
S = (P % 8) - 4
T = (P / 8) - 2
```

One native-value dot with coefficients **(0.5,0.5,4)** produces `(S+8T)/2`. Its doubled result plus20 is the whole-region table index. There are 28 reachable states in 32 slots. First network0 has its complete 28-state float2 table retained in JSONL. The negative-relative-sign variant uses `x−y ∈ [-3,3]` and the corresponding bias19. This is a producer-relative identity: it does not reconstruct gate or up first.

### Priced boundary

* Preparation: compute one fixed 32-entry table per small network; **256 bytes** for two FP32 down outputs, including four unreachable slots. The input and output coordinates are defined above.
* Recurring producer: gather the three activation bytes, convert the three signed-A2 values to E2M1 codes, wire them into the MMA fragment and zero other K positions. This is not free; the existing A8→four-FP4-digit producer cannot be credited as a zero-cost input boundary.
* Body: one dot under the exact half-integer contract. Actual sm120a execution would require a native witness; no GPU measurement was performed.
* Reader: double/round the accumulator to integer, add20, index one float2 table load. Direct register placement may avoid a shared staging image but still pays lane wiring. A warp MMA tile and any unused lanes must be counted.

This table removes both SiLUs, hidden requantization and down work **for this small compartment**. It does not remove those operations from the general dense FFN. At arbitrary A8 amplitudes, `S` spans 511 states and `T` 256; the radix would need at least512 rather than8, exceeding the common-SF E2M1 coefficient grammar.

## Input-preserving positive controls

`DP4A([1,4,16,0], [x,y,t,0])+42` produces an injective index in `[0,63]`. Its complete network0 float2 truth table costs **512 bytes**. It requires gathering/packing the input bytes, one DP4A, bias/address arithmetic and a float2 table read. It is an information-positive control, not a surprising new kernel.

**Prepared SFs/lane wiring are allowed.** The same radix label is structurally realizable by one blockscaled K64 FP4 MMA: put x/y/t at K positions0/16/32, use weight digits +1 and prepared UE4M3 weight SFs `[1,4,16,1]`, common activation SF one. This escapes the unit-SF negative. The actual `src/fp4_ops.cuh` instruction has four K16 scale vectors. Its preparation, scattered input-fragment producer, tile occupancy and table read remain unmeasured; DP4A obtains the same label without the tensor/lane boundary. Neither is recommended for GPU timing as a replacement for the real FFN.

For three full A8 coordinates an input-preserving table has `2^24` entries: **128 MiB for two FP32 outputs**, before replication across compartments. The A2 full-width input-preserving table has `4^5120 = 2^10240` entries. Adding a second label word rather than a table changes the continuation problem; it does not finish it.

## Why completed local tables do not lift additively

Even network0 violates the necessary mixed-rectangle identity for a sum of independent completed local tables. At t=0, its first down output is:

```
F(0,0)=0
F(1,0)=F(0,1)=-0.00268193451
F(1,1)=-0.00980362669
```

`F(1,1)+F(0,0) ≠ F(1,0)+F(0,1)`. Exact output and rectangle-sum bits are retained. As exact dyadics in units of `2^-32`, the two sides are −42,106,256 and −23,037,642; Lean checks this inequality. This is a rejection of **exact-additive partitioned completed-FFN LUTs**, not arbitrary hierarchical carriers or a search over nonlinear floating-rounding encodings. A local table returning scaled GU partials still leaves the nonlinear continuation and the max over the full K128 hidden group; it is no longer a whole-map replacement.

At D5120 there are40 original input groups; at FF17408 there are136 hidden/down groups. Independent row/group scales, the nonlinear gate of accumulated projections, hidden max and down scales cannot be discarded by summing the toy endpoint table. The reusable exact compression mechanism is **equal sign columns within the same scale group**; with full A8, merged activation sums require wider digits and explicit pricing. The census supplies no trained-column equality or full-width throughput evidence.

## Formal statements and reproduction

`exact_search.lean` (Std only) kernel-checks:

* collisions prohibit every decoder, regardless of relabeling;
* packed S/T labels recover both coordinates on their reachable range;
* an arbitrary complete observer runs through that packed label;
* radix4 recovers the three input coordinates;
* equal integer dots preserve an arbitrary same-scale native continuation;
* the retained bit-word witness rejects the sum carrier, conditional on the evaluator's explicit bit words;
* a nonzero mixed rectangle rejects exact-additive independent tables, including the retained dyadic integer rectangle.

The exhaustive C++ census is a reproducible computation, not a Lean-certified evaluation of `exp`. No theorem substitutes real-number algebra for native FP32/SFU semantics. The Lean file has no `sorry`, extra axioms or native-decide admissions.

```sh
g++ -O3 -std=c++20 -ffp-contract=off -fno-fast-math \
  experiments/binary-whole-map/exact_search.cpp -o /tmp/binary-whole-map-exact-search
timeout 55 /tmp/binary-whole-map-exact-search \
  > experiments/binary-whole-map/exact_search.jsonl
timeout 55 lean experiments/binary-whole-map/exact_search.lean
```

Method owner: programme12 [observer-search PLAN](canon/program-search/observer-search/PLAN.md), with endpoint/factorization doctrine in [Composition.lean](canon/foundations/lean/Kelana/Composition.lean). Native controls remain [DP4A](../binary-bitserial/README.md) and [four FP4 A8 digits](../binary-a8-fp4/README.md). Trained activations, engine capture and actual model evaluation belong to the parent experiment, not this bounded census.
