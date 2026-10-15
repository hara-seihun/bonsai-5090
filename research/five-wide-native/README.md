> Latest Dict8v2 (`3fdbad454` / DLL`4098ba03`) passes strict frozen-reference/repeat/full-logit and all gate/up→SiLU→q4/s4/g4 endpoints, but loses37.38% ordinary and37.49% deterministic whole-model TPS against fresh matched controls. No adoption; production unchanged. Native matrix completed once in152.324s; all7children0≤50s, mutex released/closed and explicit handback sent8614. Local SSH143 was reconciled by read-only retrieval of the completed native receipt, without replay. [Measured result and complete invoice](DICT8-RESULT.md), [native receipt](dict8-engine-window-v2-result.json), [lossless runtime custody](dict8-engine-window-v2-custody-receipt.json). V1 remains a separate consumed startup failure; [immutable CPU successor/source/build](dict8-engine-result-v2/README.md).

> Latest accepted model closure: H1 graph-v3 closes strict full-model logits and first-butterfly/Hadamard1024/FP4 endpoints. Whole-model TPS loses40–41%; no adoption. All ten native runners exited0, GPU explicitly returned to PrismML. [H1 settlement](H1-TAIL.md#native-graph-v3-settlement); [source-bound result](h1-window-v3-result.json).

## Joint compact-consumer CPU closure

[Joint consumer result](joint-consumer-result/README.md) binds private source `3dc21a76b` to eight complete `sm_120a` consumer builds, finite Lean contracts and 40,960 real-operand CPU comparisons. Lead Shared8compact retains the 1,664-byte/18.75%-smaller pair, replaces per-stage global gathers with a once-filled shared book, and compiles at242 K-loop sites/126 registers/88,896 dynamic shared bytes. The six-LUT no-lookup alternative costs1,728 bytes/361 K-loop sites/121 registers. These are same-build Linux counts, separate from historical timeline counts. Native startup/endpoints, occupancy, throughput and adoption remain unset; no GPU or serving/cache/profile-owner mutation occurred. [Delivery and immutable source/build custody](joint-consumer-result/DELIVERY.json).

## Latest model-bound closures

- H1 graph-v3 (`95fce9015`, unchanged DLL`398a9d2b`): independent serving-repeat stable; H1 and parent-tail plain/mixed each match all64×248320 logits across two cached observations after three teacher-forced warmups. Observed hits2/captures0; warmup inputs/tokens exact. Each arm checks356,515,840 intermediate F32 words,178,257,920q4 bytes,22,282,240s4 bytes,20,480g4 words, all0 mismatches. Ordinary medians4022.795777plain/4001.247069mixed vs6777.352952 (40.644%/40.962% loss); deterministic3947.455855/3931.397121 vs6583.070035 (40.036%/40.280% loss). Completed182.892s/240, native50s caps; all children stopped, final mutex acquired/released, serving unchanged, explicit release to8614. No successor queued. V1 reference-graph and v2 isolated-import failures remain retained; v3 was run once after fresh handback. [Full contract and evidence](H1-TAIL.md).
- Packed-tail v3 (`d7926585e`): real-model mixed/plain each pass71,303,168 SiLU words,35,651,584q4 bytes,4,456,448s4 bytes and4096g4 words,0 mismatches. Three-repeat B64×128 medians3973.086607mixed/4012.525034plain versus6777.352952 serving:41.377%/40.795% loss, no adoption. The strict parent-tail comparisons above use H1's sibling DLL/profile; they do not retroactively certify the preserved original DLL's full logits. [Retained original settlement](PACKED-TAIL.md);17 native files, private source bundle, writer released.
- First divergence: prefillstep0/layer0 `ab` GEMM cfg12/split-K6 float atomic fold, on identical operands/ordered partials. All720 fold orders cover every observed output word; phase-sync does not remove it. Private witness commit`baa1d8ee0` on `agent/serving-divergence-1014`.
- Unchanged serving with `H5_DETERMINISTIC_GEMM=1`:6585.409637/6583.070035/6579.811334tok/s; median6583.070035,2.866649% below video6777.352952. Fixed-order numerical profile is the future strict repeat-logit reference; ordinary serving stays the TPS baseline and original15 quality cohort stays detunset/distinct. [Exact report/receipt](DETERMINISTIC-SERVING-TPS.md). Nativeexit0, frozenplan unchanged, mutex released and PrismML resumed. No serving binary/plan mutation.

# Five-wide native carrier discovery (Oct 13–14, 2026)

Owner: programme 12 packed compute; reporting coordinator `00000000-0000-0000-0000-000000000000`. Scope: tensor-core (MMA) realizations and their complete engine/observer costs; non-tensor inference routes belong to other agents. Private opt-in builds are allowed; serving remains unchanged. GPU via shared mutex `Local\BonsaiGPU`.

## Dict8 consumer continuation

the operator's October14 direction: retain the tested-exact18.75%-smaller carrier; the measured−37% whole-model result rejects this consumer realization, not the packing idea. Search carrier labels and MMA consumer jointly, including direct operand encodings and few-op ISA conversions such as prmt/lop3; price complete conversion, shared/register traffic and tensor-core feed against saved weight traffic. Cache the13.656s encoding offline rather than repeating it at startup.

The matched ordinary gap is0.7228743s per8192 output tokens, or5.64746ms per64-token decode step; no measured breakdown is established yet. Timeline owner`e732787c` uses retained evidence first; one fresh≤240s timeline window after Tau2 is authorized in principle, with exact harness approval and8614 physical START still required. Joint code/consumer search owner`16e8e656`; offline cache owner`66909408`; coordinator`976997f4` reports the profile breakdown and best design to coordinator`24c1a319`. CPU work proceeds in parallel. Serving unchanged; no adoption.

[Private single-scan offline cache and CPU evidence](dict8-identity-result/README.md) binds a complete 2,317,355,136-byte cache to the exact model, ordered payload and encoder/consumer semantics. One hash-before-parse identity is shared by the actual opt-in Engine and Reader, with retained local-NTFS mapping on Windows and private snapshot on Linux. Full-model CPU comparison passes all1,426,063,360 words/64payloads; recurring model identity, platform backing, whole-payload allocation/read/hash and retained WF0 work are separately invoiced. No startup encoding fallback exists. Native cached-startup, GPU residency and throughput remain unset; the completed original v2 matrix was not replayed. [Predecessor custody](dict8-cache-result/README.md) is preserved.

[Joint carrier/consumer CPU/compiler closure](joint-consumer-result/README.md), private commit`3dc21a76`, supplies shared8 compact as the lead measurement candidate: unchanged18.75% payload reduction, zero stage global dictionary gathers,242 K-loop sites/126 registers under Linux nvcc13.1,88,896 dynamic+1,024 static shared bytes. The profiler owner's the RTX 5090 host nvcc13.0.88 rebuild gives259 sites/125 registers; compiler identity is part of the comparison. Its initial table fill and shared-bank address multiplicity remain charged. The distinct lookup-free six-LUT3 carrier saves15.625%,361 sites/121 registers; the whole decoder costs25 SWAR data operations/24trits, not six. Both new layouts match40,960 retained operand words and retain the source arithmetic suffix; native startup/endpoints/TPS remain unset. [Coordinator inspection and exact limits](joint-consumer-settlement.json). Same-build native137/global272 counts are separate from the historical Windows160/297 counts.

[Profiler retained-evidence audit and evolving finite harness](../dict8-profile-1014/FINDINGS.md): no retained timeline matches v2 or frozen serving. Fresh paired nsys evidence is needed for the GU/non-GU split; NCU counters are unavailable on the current the RTX 5090 host token. Driver teardown/cleanup bounds are being repaired before exact authorization and post-Tau2 physical START. No native profiler window has been consumed.

## Target as fixed

Five trits per byte (3^5=243<=256, 1.6 bit/trit) consumed by native `mma.sync m16n8k64 kind::mxf4nvf4 block_scale.scale_vec::4X ue4m3` (the actual WF0 activation format: E2M1 + E4M3 per 16 K) across a whole K loop, one readout at the boundary.

## Current frontier

[Full-K/block-Q8 continuation](FULL-K.md): a rolling8B mixed register label carries the complete17408-feature×B64×K5120 reduction without a history trace.356.5M native block labels,1.114M final full-K label words/F32/Q8 outputs and8704 scale words match an actual FP4 reference on saved synthetic data. Source16-K E4M3 scales,128-K F16 scales, observed native RZ boundaries and128-feature normalization are retained. Rolling route776.016µs versus strong grouped-SIMT1037.272 and ordinary FP4449.560; no engine adoption. The transient FMA operands, compiler-convergence repair, scoped scalar-carrier falsifier and continuing cheaper whole-route search are explicit there. Earlier finite K10/radix128 evidence remains below.

## Prior art found (do not re-derive)

- `H5_COMPACT=dense5` (WF3) already stores 26 B per 128 trits and decodes through a shared 256-entry codebook (LDS) plus shift/or assembly into ordinary e2m1 operands (`src/compact.cuh`). It is the single-stream configuration (strict S1 691 tok/s). The B64 quality configuration (`custom-gsm64/manifest.json`) sets `H5_CARRIER_WF0=1`: block matrices use two-bit WF0, head stays dense5.
- Same-binary single-stream ABBA: WF0 blocks lose 9.8% to dense5 (`ternary-results/halo-5090/local-profile-fixes/integrated/README.md`). A 32-replica dense5 LUT decoder lost 4.3% (`decoder-isa/9e1054d`).
- So an ALU decode of 5-trit bytes into standard operands is existing art. New work must make the tensor core carry part of the decode, or make five-wide pay at B64.

## Scoped negatives (each limited to its stated family)

1. **Per-block sum collision, uniform scales.** With A/B block scales not used as data, one 16-slot block sum of e2m1×e2m1 products takes <=4609 values. Carrying two rows' dot products with one shared full-15-value activation per slot needs ~148k block outcomes, so two inputs collide in that block; any observer sees identical lanes. Does not cover data-carrying scales, product chains, restricted activation domains or approximate contracts.
2. **Single FP32 lane, full K=5120.** Two independent full-domain dot products need ~1.5e10 distinguishable outcomes > 2^32. Chunked accumulators are outside this bound.
3. **5-window bitwise-affine family (design search, `grid5.cpp`).** Pass 1 raw nibbles, pass 2 and half of pass 3 from 544 distinct bitwise transforms (rotate/xor/and). Searched for an affine 3^5 grid among byte window images. Completed parts found none; incomplete parts are unknown, not negative.

## B64 pricing of the N-as-trit-position design

At batch 1 the eight N columns of m16n8 are mostly idle, so columns can be trit positions. At B64 the 64 streams already fill N: five columns per token = 40 n8 tiles per 64 tokens, times >=3 passes per 32 bytes = 120 MMAs per 16x160 trits versus 20 for plain FP4. About 6x tensor work: **single-stream only.**

With activations in N (K-layout), exact FP4 B forces each window slot to hold one scaled activation, so each window must be proportional to one trit for the minimum 2.5 passes. That is a full in-register decode, the same economics as dense5. At B64 ALU decode is amortized over 8 n-tiles while every extra tensor pass is not. This prices only the stated conventional-operand/N-as-position route. Whole instruction compositions, shared relabelings, data-carrying scales and jointly searched carriers/consumers remain open.

## Native R64 GU-geometry result (RTX 5090, Oct 14, 01:2x UTC)

`r64_carrier.cu` (sha256 `5dc048d1…3857`, exe `20646e6d…f9c3`). the RTX 5090 host path `models/experiments/bonsai-quality-20261013/five-wide-native/` holds the receipt. Runner: `gpu_run.py`, which wraps the shared mutex `Local\BonsaiGPU`, held ~1.5 s and released.

Geometry M=34816 (GU), K=5120, N=64. Every arm runs the same mxf4nvf4 4X ue4m3 MMA stream: 2 m-tiles × 8 n-tiles per warp, real-format E2M1 activations with random E4M3 scales. Weight copies rotate to exceed the 96 MiB L2.

| Arm | Bits/trit | Weight bytes | CPU exact (360 rows, unit scales) | Bitwise vs E4 (full M×N, random scales) | Median µs | Achieved weight GB/s |
|---|---:|---:|---:|---:|---:|---:|
| E4 pre-expanded e2m1 | 4 | 89,128,960 | 0 mismatches | — | 74.432 | 1197.5 |
| WF2 (WF0 mask/shift interleave) | 2 | 44,564,480 | 0 | 0 | 48.128 | 926.0 |
| D5 (5 trits/byte, 256-entry LDS LUT + funnel) | 1.6 | 35,651,584 | 0 | 0 | 38.240 | 932.3 |

- **D5 beats WF2 by 20.5%** at R64 in this microkernel. That equals the 0.8 byte ratio, and WF2 and D5 achieve the same GB/s, so the kernel is weight-traffic-bound. The LUT decode amortized over 8 n-tiles costs nothing measurable here.
- The tensor floor for these 1.39M MMAs is about 11 µs, so tensor work is not binding.
- The 0-mismatch CPU check also confirms the assumed m16n8k64 FP4 A/B/D fragment layouts.
- Scope: standalone kernel at about 930 GB/s achieved, not the engine's pipelined GEMM. A better-pipelined kernel moves toward peak bandwidth, where decode could become binding. The engine-level decider is a same-binary B64 A/B of the existing selector (`H5_CARRIER_WF0=1` vs dense5 blocks).

## Retained whole-engine B64 A/B

`retained-b64-custody.zip`, SHA256 `69f295b510eba328de51c2873bfa10eda7cfd47242bd1157464a79f6056f0a40`, preserves original failed v1 and completed v2 ABBA. Completed measurements were recovered without GPU replay. WF0 mean 6630.0729765 tok/s; dense5 mean6194.0271345: dense5 −6.5768%. Same wide256 executable/DLL, B64×512, three passes each, separate missing-key-filled plans; frozen production plan unchanged. This rejects the existing dense5→conventional FP4 integration at B64.

Saved GSM64 timing-token screen: both arms350/384 correct. The capture omits prefill prediction and eight warmup tokens and continues generation after EOG; this is reused timing output, not a published quality benchmark. Text/prediction identity is not established. Corrected decoder and outputs are separate from the original immutable ZIP: `b64-recovered-quality-summary.json`, `grade-retained-b64.py`, extracted v2 `decoded-screen.jsonl` files.

## Shared-label native CRT constructions

### All-ternary chains

`crt5.cu` maps five trits into243 reachable CRT labels modulo15015 (3·5·7·11·13). Each native INT8 m16n8k32 MMA forms low/high-byte2×2 product fragments; shift/add and modulo return the **same coordinate**, which feeds the next MMA without logical trits. Three MMAs/warp implement32 independent four-factor chains,160 logical products. All59,049 pair transitions pass and262,144 four-factor chains match the final observer. Producer486B, observer15015B,27regs/no spills.

Launch-priced first run7.281µs vs arithmetic7.627. Matched stronger packed-bit control (`crt5_control.cu`): native7.988, arithmetic8.311, packed7.893µs; all three agree. No speedup claim follows from the weaker-control lead. These are resident endpoint kernels, not B64 engine GEMMs.

### Actual full15-value FP4 domain

`crt5_fp4_global.cu` uses two CRT coordinates modulo15015, with prime assignmentsA=[3,5,7,11,13],B=[11,7,13,3,5]. Each logical doubled FP4 value is in{-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12}. Per-coordinate prime products[33,35,91,33,65] uniquely separate all integers−12…12. The joint label retains all759375 product states, allowing a final five-term dot readout.

**184,528,125 native checks** cover all243 ternary weight codes ×759375 independent full15 activation codes. Zero final integer-dot and binary32 suffix errors; deterministic native-label fingerprint44361825457853400. A262144-case matched kernel check also passes. The two MMA calls in this initial mixed-domain version are parallel carrier branches. The later six-MMA serial construction below supplies reusable mixed-label continuation.

The suffix is `((integerDot*0.5)*activationScale)*weightScale`. Activation scales are nonnegative finite E4M3 codes0…125; weight scales are the F16-compatible dyadic grid1+j/16, j0…15, selected deterministically by case index. This is a declared common-scale five-term map, not the full independent-scale/K-ordering or FFN contract.

| Reader | Graph-priced native µs | Matched arithmetic µs | Active observer |
|---|---:|---:|---:|
| joint hash (`crt5_fp4.cu`) |7.367680|3.904640|16MiB/max21 probes|
| small constant (`crt5_fp4_small.cu`) |5.057600|3.913280|257B|
| small read-only global (`crt5_fp4_global.cu`) |4.235200|3.906880|257B|

Pricing:100 complete kernels/graph,7 alternating rounds,262144 independent cases, median per-kernel event time. Input coordinate load, producer lookups, both MMAs, shift/add/mod, final observer and scale suffix are timed. Static table construction, allocations/transfers and raw five-FP4-nibble→base15-coordinate conversion are excluded. Weight producer972B; activation producer3037500B. Small-reader harnesses also allocate/export the unused16MiB hash as a comparison witness; it is not read by the measured small kernel. Thus total harness allocation includes it, while active resident image does not.

Executable-bound SASS (`retained-crt/crt5-fp4-global/sass.txt`):2IMMA.16832.U8.U8,12SHFL.IDX,14IMAD.HI.U32 and5LDG.E.S8.CONSTANT static instructions among the full producer/observer/suffix flow;26regs/no spills. Static counts are not cycle measurements. `crt5-global-sass-invoice.json` lists the complete histogram.

`CRTContract.lean` proves joint residue injectivity on25 consecutive integers at each coordinate and continuation preservation conditional on physical-step conformance. It builds without sorry. Native machine conformance, negative-value CRT dictionary and full engine arithmetic remain empirical/unformalized; the Lean file does not certify CUDA.

### Custody and frontier

`crt-native-custody.zip` SHA256 `dc18a748dedc479a8887c7911e12c0d90e0411a2d17f94b78ebade2c517ae021`,8211855B, contains exact executable/source/receipt/build/SASS and deduplicated binary witnesses. `MANIFEST.json` maps Windows relative paths to hashed objects; the readable extraction is `retained-crt/`. Every retained receipt/file hash was checked during recovery; no measurement was replayed. Toolhelp32 process enumeration observed no matching crt5.exe process after the runtime restart; each native runner released the shared mutex. PowerShell CIM/tasklist denial is retained separately.

### Exact mixed-label closure, serial continuation and raw source

`crt5_fp4_remainder.cu` compares both physical coordinates to the encoded logical product on all184,528,125 pairs: zero errors, unchanged fingerprint44361825457853400. Native4.314240µs, direct3.908480;28regs/no spills.

`crt5_fp4_serial.cu` starts from full15 FP4 labels and applies three ternary sign/zero transitions through six native MMAs, without intermediate logical products.262,144 four-factor cases pass. The initial8.325760µs packed control was weakened by divergent constant tables; its repaired global-table successor (`crt5_fp4_serial_control.cu`) measures native5.211200µs vs control4.254720,32/20regs, no spills. Each successor also retains the184,528,125-pair exact-label check.

`crt5_fp4_raw.cu` times actual raw E2M1-nibble→joint-label producer lookup: five raw nibbles and three dense5 bytes in an8B input record. Producer4,194,304B plus972B weight and257B final reader; six serial MMAs.262,144 raw composed cases match, with the exhaustive base15-domain closure also passing. Native5.312000µs vs packed-mask/DP4A2.872960;32/23regs/no spills. Creating the8B records, table construction and transfers remain outside timing. These losses concern this CRT byte-fragment route.

## Five-wide GU → nonlinear terminal observers

### Single five-term group

Two dense5 weight bytes independently select gate/up rows. Five raw FP4 nibbles carry the full15 numeric domain, including both zero encodings. Doubled partial sums satisfy−60≤g,u≤60. One signed INT8 MMA produces the opaque integer `g+125u`; a second unsigned MMA consumes that C and adds16320·down_role.128 independently positioned outputs per16×8 tile,1,048,576 sampled outputs total. Final target is rounded/clamped unit-step Q8 of `SiLU(g/2)*(u/2)*(down_role−1)`, using `__expf` and `--fmad=false`.

The original59,049-entry pair producer occupied472,392B. `affine5_gu_packed.cu` replaces it with a1,944B243-entry digit-plane producer. For all five byte positions, `G+125*U+0x0202020202 XOR0x8080808080` produces the signed coefficient `wg+125wu`; each biased byte is2…254, so no byte carries occur. All59,049 weight pairs ×5 coefficients were checked before launch.

Latest matched seven-arm pricing (`affine5_gu_folded.cu`,100 kernels/graph, seven rotated rounds):

| Complete route | µs |
|---|---:|
| Two native MMAs, tagged final reader |4.194240|
| One native MMA, final role multiplication |2.984640|
| Direct scalar GU + analytic SiLU/Q8 |7.047680|
| Prepared-weight DP4A + analytic observer |4.930240|
| Prepared-weight DP4A + dense final reader |4.685120|
| Folded SIMT per output + tagged reader |5.298240|
| Folded SIMT128-output tile + final role |3.251520|

Thus the two-MMA tag route loses to the strongest control. The one-MMA route has8.21% lower latency than that control in this finite resident map. All physical-word/Q8 and control errors are zero on the saved corpus. Native two/one kernels34/30regs; folded-tile28regs; no spills. Source96B/tile. Full observer allocation47,761B; one-MMA and untagged controls read only its15,121B positive-role subimage. Producer lookups, raw nibble conversion, MMA/DP4A, reader and output are timed; preparing Tile records and tables is excluded.

The first source failed to compile because a missing space named the pointer `__restrict__obs`; v2 then exposed incorrect B-fragment placement, with1,023,613 word errors. Both failures are retained. v3 repaired B column=lane_group and K=4·thread_in_group using the [PTX layout excerpt](ptx-m16n8k32-layout.txt); successful results use that placement.

### Two five-term groups with independent scale contexts

`affine5_groups.cu` produces `L1=g1+125u1`, multiplies all four s32 C registers by15625, then the second native MMA appends `L0=g0+125u0`. No g/u readout occurs between MMAs. The final observer reconstructs four partials and applies:

- independently selected nonnegative finite E4M3 activation codes0…126 for each group/column;
- independent G and U weight scales for each group/row on the F16-compatible grid1+j/16, j0…15;
- source-ordered F32 scaled sums, then `__expf` SiLU, signed down-role and unit-step Q8.

Q8 clamps the float before converting to integer, preserving positive saturation above the s32 conversion range. The exact pre-quantization F32 bits are checked too.1,048,576 sampled outputs have zero physical-word/F32/Q8 errors. Native6.077760µs loses to grouped folded-SIMT5.449280; direct scalar12.244801, DP4A7.477760.36regs/no spills; producer1,944B, source240B/tile.

`affine5_groups128.cu` changes the **whole carrier/observer map**: use the otherwise spare K slots to contract A=(wg,64wu), B=(q,2q), yielding `L=g+128u` in one MMA. Each group uses ten nonzero K slots within m16n8k32. The first C is multiplied by16384 before the second MMA. The four-digit final observer uses shifts/masks instead of division by125. Packed doubled activation bytes are `(X<<1)&0xfefefefefe`, so no cross-byte carry survives. Source inputs and exact terminal F32/Q8 witnesses are identical to the radix125 run.

| Radix128 complete route | µs |
|---|---:|
| Two native MMAs, analytic scaled observer |5.225600|
| Direct scalar |12.201281|
| Grouped folded-SIMT128-output tile |5.440001|
| Prepared signed-weight DP4A |7.441600|

Native has3.94% lower latency than the strongest control here.38 native/40 folded-tile regs, no spills; producer3,888B. Executable-bound SASS contains two`IMMA.16832.S8.S8` instructions; [static invoice](affine5-128-sass-invoice.json) is not a cycle profile. [CPU-only identity reconciliation](affine5-128-sass-reconciliation-v2.json) binds executable`3beed256…`, retained timing receipt and every static opcode count; the raw source hashes differ only by newline encoding. Its first NOP-omitting parser result is retained separately; no GPU measurement was replayed. The shared s32 word lies within±126,819,900. `AffineFive.lean` and `AffineFive128.lean` prove integer digit injectivity and s32 bounds; the former also proves the per-byte bound and bias identity. [Lean receipt](affine5-lean-receipt.json) binds the compiler, exact files and successful builds. CUDA conformance, intrinsic F32 semantics and source-layout correspondence remain empirical.

### Successor custody and remaining contract

`five-wide-native-successor-custody.zip`,26,349,837B, SHA256`b691cc01a07af37e30be4401a0a08076b8c997ee94c17226bef0a5be2a536623`, retains all four newer CRT variants and radix125 GU experiments, including compile and native-placement failures. Readable extraction: `retained-successor/` (123 bound artifacts).

`five-wide-affine128-custody.zip`,8,351,443B, SHA256`caf733d84b4502e5c2ae4913fc8ff8563d08183825b654009c1a8731871c16e9`, retains the exact radix128 source/executable, receipt, source inputs, physical words, F32/Q8 witnesses, producers, SASS and both Lean proofs/receipt. Readable extraction: `retained-affine128/` (16 bound artifacts). All archive/artifact hashes were checked locally after export, without replaying measurements. Original B64/CRT archives remain unchanged. Shared mutex was acquired nonblocking and released after every GPU run.

The [full-K continuation](FULL-K.md) supersedes the arithmetic/observer gap of this finite two-group result. Remaining target: a cheaper complete coupled route against ordinary native FP4, installed engine/source-layout adapters, actual-model acceptance, preceding normalization/Hadamard and down-projection K. No engine adoption, B64 speedup, complete installed FFN acceptance, rental or serving mutation is established.

### Actual-engine outcome

Private source `31ad74a9` integrates opt-in mixed/plain cooperative GU. Real-model GU boundary:71,303,168 F32 words/arm match serving exactly. Best engine screen4087tok/s vs serving6788:39.8% loss, no adoption. Full-model logit acceptance is unset because unchanged-serving repeat controls also fail. See `FULL-K.md` for retained controls, source and custody.

[Packed GU terminal consumer](PACKED-TAIL.md) owns the exact paired-label GU→SiLU→sign→Hadamard1024→FP4 successor, failed configuration7 setup, repaired selected configuration13 build, and bounded GPU closure.
