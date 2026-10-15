# FP4/TF32 fibre continuation — October 13, 2026

Owner: packed-computation programme 12; reporting coordinator `00000000-0000-0000-0000-000000000000`. Private research custody, no engine adoption, rental, persistent service or model/HTTP performance claim. This folder retains sources, finite observers and native receipts. Rebuildable executables may be deleted; sources, observer witnesses and original observations are retained until integrated into programme custody.

Original agent session log (not exported). Its research source is `~/bonsai/research/fp4-fibre-search/`; its structural and whole-region workers are `00000000-0000-0000-0000-000000000000` and `00000000-0000-0000-0000-000000000000`. The source tree is not a Git repository. Use `sudo -n` for its root-owned files.

## What changed

1. **Native FP4→TF32 chain:** original compiler and separate `run-chain.py` passed 729 logical triples /93,312 fragment-output checks on the RTX 5090. Two independent ternary three-factor products are carried by uniform FP4 K64 followed directly by TF32 K8. No intermediate logical decoder. Original source hash `9a04b6c6b2ee9bf08f3d6077fbf71e96f4803489f56778ddc9c200118282902c`; binary `33246cf7642baa2f05d6ebb490a442ee53b649f5780148e67a6041a4788e1e17`.
2. **Three-chain native TF32:** `native_prime.cu` passed every 27⁴=531,441 input tuple, 17,006,112 lane checks, zero hardware/CPU bit mismatches and zero final-observer collisions. Three independent ternary four-factor chains use three TF32 MMAs. Each input names three trits. Prime encoder `(2,5,7)`, exponent 1 for −1, 5 for 0, 0 for +1, scale 2⁻¹⁵. Explicit `cvt.rna.tf32.f32` precedes each MMA, and exact 1/8 scaling compensates uniform K8 replication after each MMA. These conversions and normalizations are paid instructions, although neither decodes a logical intermediate. Observer has18,690 entries/149,520 bytes. The original probe compared accumulator0 in every lane. The final GPU readout below independently checks all four accumulator paths. `native-receipts.json` binds original source, binary and observer hashes.
3. **Five-chain CPU binary32:** `five_search.cpp` establishes collision-free final observation for all243⁴=3,486,784,401 tuples. Each input names five trits, and three scalar binary32 multiplications carry five independent four-factor products. Encoder `(2,3,5,7,11)`, exponents as above, scale2⁻²⁷. Deterministic prefix quotient evaluates52,259,580 final transitions from215,060 reachable three-input states rather than enumerate all raw tuples. Final observer has2,490,090 labels/19,920,720 bytes in `w5-q5-m0.bin`; 3.80seconds/1,285,516KiB maximum resident set measured. This is **five per binary32 scalar multiply**, not five per native TF32/MMA. The original12-million-transition search bound was expanded to60million for this one candidate; no timeout is called an obstruction.

## Why the prefix quotient covers the whole domain

Every source tuple is included. At each stage, enumerate every reachable physical FP32 bit pattern against every next source input. Two prefixes may share a representative only after their physical bits and logical target agree. Future physical transitions depend solely on those physical bits and the next source label; logical transitions depend solely on the ternary target and next tuple. Equal pairs therefore have identical continuations. The logical state exists only in the search oracle; no runtime logical intermediate is required.

`Observer.lean` proves the generic reachable-fibre observer theorem and deterministic-prefix continuation equality, with no `sorry` and no numerical FP32 model. `lean Observer.lean` completed successfully. It does **not** formalize the concrete exhaustive enumeration or CUDA semantics. Finite dictionaries and executable receipts carry those empirical checks.

`wider-results.jsonl` retains the bounded prime-family search at widths3–5 and q5–8 under FP32, TF32 RNA and TF32 RNE. Redundant width3/4 decoder files were deleted; the same executable regenerates them. `tf32_four_search.cpp` and `tf32-four-results.jsonl` additionally retain all1,536 four-wide TF32-RNA prime-family candidates (six prime tuples, independently chosen q5–8 per coordinate): every candidate collides,13.51seconds/8,352KiB peak. This is CPU rounded arithmetic, not new native rejection evidence; no universal bound follows. Collisions reject only that particular encoder/arithmetic composition. A prefix collision extends to the complete chain by padding with all+1 tuples, a power-of-two multiplication preserving equal physical states and unequal logical targets. No local collision excludes arbitrary whole-region carriers, extra lanes, scales, multi-accumulator paths or final consumer observers.

## Price and actual-target limits

- Native scalar witnesses use **uniform replicated K**, with identical operands throughout each native contraction. They establish composition, not independent K accumulation or a dense-projection replacement.
- Three-chain probe event time0.29686ms is the exhaustive test kernel only, including validation atomics and outputs. It is not a matched packing benchmark. Original chain0.143999s is process duration, not kernel timing.
- Three-chain source table108B, finite sorted observer149,520B; five-chain source table972B, observer19,920,720B. The three-chain observer now executes on GPU, with costs below. Five-wide GPU realization/readout remains unset.
- Complete resident invoices below include producer lookup, operand conversion, normalization, final observation and output. They exclude static preparation, host allocation/transfers and validation/warmup. Arbitrary actual-model K joins, lane ownership and resident transformed images still need their own invoice.
- Actual selected Bonsai WF0 sources have ternary **weights** and full15-value FP4 dynamic activations plus scales. These all-ternary operand witnesses do not establish an actual-model kernel. Whole FFN/head final observers may admit stronger mappings than a source-stage decoder.
- Original `whole-region/oracle.cpp` and its CPU full-layer smoke receipt are mathematical F64 oracles; CUDA reduction/SFU parity remains open. Actual-model complete-region/native timing, quality and TPS remain unset.

## Native readout and independent-K results

`native_readout.cu` / `native-readout-receipt.json`: shuffled exhaustive531,441 tuples, all four accumulator paths /68,024,448 lane checks, zero missing or incorrect outputs. Complete resident producer→three uniform-K TF32 MMAs→observer→output median **105.085µs**, against76.240µs warp-direct and8.874µs lane-direct. Timed native18registers, no local/shared bytes or compiler-reported spills/barriers. Correct construction, **no speedup**.

`native_k2.cu` / `native-k2-receipt.json`: a genuine nonuniform-K8 contraction sums four copies each of two independent pair products, then normalizes by1/4. Layout control `2·3+5·7=41` and68,024,448 fragment-agreement checks passed, but the one-float `(2,5,7)q5` carrier has3,094 target-collision boundaries among26,076 states. Exact witness: physical bits834666496, logical targets74/114, inputs `[8,8,8,26]` / `[26,26,26,20]`. This rejects that family only; no usable observer was written.

`native_k2_readout.cu` / `native-k2-readout-receipt.json`: retain two independent pair carriers from **two TF32 MMAs**, then query the existing immutable observer twice and join at final observation. All531,441 tuples /68,024,448 all-four-path lane checks pass with zero missing/wrong outputs. Complete resident path **147.901µs**, versus72.102µs warp-direct and7.450µs lane-direct.22registers/no local/shared bytes or spills/barriers. Padding RNA conversions and2⁻³⁰ relabels are charged. This is three two-term ternary dot sums; it establishes neither single-MMA K compression nor a speedup.

The recovery observation in `recovery-observation.jsonl` recovered these exact saved the RTX 5090 host receipts without replay. No fibre executable was running; a separate `quality_length_bench` remained active and untouched. Launches serialize with `Local\\BonsaiGPU`; mutex wait0/128 grants ownership,258 launches nothing. All bounded owners release in `finally`.

`native-ir-invoice.json`, `final-readout.ptx`, `independent-k2.ptx`, `independent-k2-joint.ptx` retain emitted PTX and native receipts. PTX static totals include validation specializations, not just the timed path. The profiling owner below binds actual SASS to saved executables using the installed matching tools; PTX does not substitute for it.

## Full-tile ownership, reachable-pair reader and matched pricing

`native_tile_join.cu` assigns128 distinct cells per full m16n8 tile to two pair MMAs. All531,441 Cartesian pair inputs pass, including timed-output comparison. Original binary reader:15.389µs versus7.885µs lane-direct. The earlier147.901µs warp-replicated layout is an engineering progression, never a matched speedup.

The GPU-generated729-pair image has240 exact physical keys. Each image entry is checked against the original immutable149,520B witness before constructing the runtime hash. `native-pair-image.bin`, `native-pair-hash.bin`, `native-pair-compact.bin` retain original/projected/packed readers; wrong or missing keys fail explicitly. The16KiB hash has at most3 probes. The8KiB hash uses one multiply/shift on the19-bit TF32 key and at most2 probes. Its target labels are three3-bit fields; final addition cannot carry between fields, and only the final observation converts to the required base5 output.

Complete graph invoices use100 unchanged kernels/graph,7 rotated rounds, one layout and source/output contract. All routes pass every531,441 output before pricing; static projection/label/table construction and allocations/transfers/validation/warmup are excluded from events and retained as paid resident state. Selected simultaneous observations:

| Harness | Native16KiB | Native8KiB | Native prepared | Native direct-PTX4KiB | Original binary | Arithmetic | Pair lookup729B |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph |5.218µs|—|—|—|13.282µs|3.254µs|—|
| compact |4.947µs|4.686µs|—|—|13.285µs|3.250µs|—|
| prepared |5.114µs|4.695µs|4.525µs|—|13.268µs|3.252µs|—|
| control |5.055µs|4.937µs|4.673µs|—|13.190µs|3.249µs|2.372µs|
| direct-PTX |5.064µs|4.950µs|4.684µs|4.401µs|13.282µs|3.252µs|2.137µs|

The16KiB reader is2.55× faster than the matched binary-reader native sibling in the graph harness. **Every native variant still loses to the strongest729B pair-lookup complete-endpoint control.** No model/HTTP gain follows. Prepared labels are108B pre-rounded TF32 values, cached global reads instead of divergent constant reads. The direct-PTX reader packs the exact19-bit physical key and8-bit target into one32-bit entry (4KiB total), has two explicit probes and uses `ld.global.nc.u32`. Its reachable positive/finite/normal domain permits the two power-of-two normalizations to commute with RNA; subtracting33 from the rounded exponent replaces two FP32 multiplies. The host rejects domain/probe-bound violations, and native whole-map validation remains authoritative. Native prepared/direct-PTX timed kernels use26registers; other native siblings24; pair control14. No spills, local/shared bytes or barriers are reported.

Each `tile-*-launch.log` retains build output, complete saved receipt, hashes and mutex observations. Corresponding remote folders are `full-tile-{joint,hash,graph,compact,prepared,control,ptx}`. Sources/drivers/failure logs remain beside the original native witnesses. Recovery inspected actual remote processes and read saved receipts before any new launch; no fibre executable was active.

## Profiling

[PROFILING.md](PROFILING.md) owns reliable side-by-side tool invocation, versions, locations, bounded counter/SASS extraction and timing interpretation. PATH did not expose the tools; read-only discovery found existing Nsight Compute2025.3.1, Systems2025.3.2, matching nvdisasm/cuobjdump13.0.85. The working compiler/global selection remains unchanged. `profiling-tools.json` retains discovery; `profile-native-ptx.py` sets only its child's tool environment. Earlier SASS-unset receipts remain truthful historical observations; newly captured actual SASS is separately bound to the executable.

## Reproduce and native custody

CPU (normally under55seconds):

```sh
cd ~/work/fp4-chain-continuation
g++ -std=c++20 -O3 five_search.cpp -o five_search
./five_search
g++ -std=c++20 -O3 wider_search.cpp -o wider_search
./wider_search
lean Observer.lean
```

Native source/build/receipt live on the RTX 5090 host under `C:/work/models/experiments/bonsai-quality-20261013/fp4-fibre-search/continuation-native`. `build-prime.py` performs build and numerical execution using the existing `quality-campaign/gpu_host.py` transport. Inspect existing receipts before repeating execution. The first compilation failed through a macro-local name shadowing the CUDA event `e`; the corrected source uses `ck_status`, and `build-macro-failure.log` retains that compiler evidence. No active quality binary was replaced.

Next real target: find five-wide native carriers (whole tile/loop alternatives remain admissible), or place the three-wide construction in a consumer where final readout/producer costs can win. Any proposed speedup needs complete matched-region timing first.

## Five-wide native continuation

[Five-wide native owner](../five-wide-native/README.md) retains the completed B64 dense5/WF0 A/B and native INT8-MMA shared CRT-label witnesses, including all243×15^5 mixed-domain pairs, final-observer pricing and scope limits. Its [FULL-K.md](../five-wide-native/FULL-K.md) closes synthetic full-K/RZ and retains fixed-weight scalar-carrier falsifiers. Installed/learned full-K/FFN acceptance remains open.

## Mixed and serial affine consumers

[MIXED.md](MIXED.md) owns full-FP4 GU→down/Q8 witnesses and prefix collisions; [AFFINE.md](AFFINE.md) owns affine C-continuations, prices, context proofs and independent scale groups. [K16.md](K16.md) owns packed16-slot producers, FP16 lowering, the joint collective observer, native scales-as-data, the PRMT/DP4A-conditioned32/36-bit join and the six-instruction C-consuming coalesced run. [K64.md](K64.md) adds native C/RTZ witnesses and the independent-scale integer-grid observer. Their Lean contracts are retained here; K16/K64 custody binds252 verified source/binary/input/output/SASS artifacts. Every priced candidate still loses to its strongest simultaneous native control. These replace no engine/service; learned full-K/reduction, quality/TPS and owner-study integration remain open.
