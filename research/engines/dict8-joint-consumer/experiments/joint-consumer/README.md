# Joint Dict8 carrier/consumer search — CPU and compiler closure

**Lead: `shared8_compact.cu`.** Keep the 1,664-byte pair and original 13-bit labels, fill a CTA-local 6,561×U32 book once, and read it in shared memory throughout K5120. Drop four unused 1,024-byte W-scale ring regions because CARRIER never copies or reads them. This removes per-stage global gathers without inventing different arithmetic. **No native execution, speedup, adoption or cached-startup result is established.**

## Same-build compiler comparison

CUDA13.1 V13.1.115, Linux `sm_120a`, ordinary capture=false, intact R64/K5120/N34816 consumer, same flags. Every row has 32 MMA sites per K128 stage, zero local/stack/spill bytes, 1,024 static shared bytes, unchanged separate rounded gate/up→means→SiLU suffix. Dynamic shared is additional.

| Route | Pair bytes | K-loop sites | Registers | Dynamic shared | What changes |
|---|---:|---:|---:|---:|---|
| Native two-bit control | 2048 | 137 | 124 | 78,912 | WF0 original native words |
| Current Dict8 global | 1664 | 272 | 130 | 66,624 | 32 dictionary LDG/stage |
| **Shared8 compact** | **1664** | **242** | **126** | **88,896** | 32 dictionary LDS/stage; zero stage LDG |
| Shared8 without ring compaction | 1664 | 242 | 126 | 92,992 | isolates the 4,096-byte reservation saving |
| Shared8 four-consumer-warp/two-stage | 1664 | 243 | 126 | 44,320 | 160 threads; doubled CTA/book setup count |
| Split5+3 shared tables | 1664 | 422 | 134 | 67,776 | two table LDS/native word |
| Split5+3 shared5/shuffle3 | 1664 | 387 | 147 | 67,648 | one LDS + one SHFL/native word, extra outlined collective path |
| **Literal six-LUT3 joint code** | **1728** | **361** | **121** | **64,576** | **no lookup at all** |

These are this build/extraction's counts. The historical timeline's 160/297 and Windows resource counts belong to different build/extraction identities. `invoice.json` pins each actual K-loop address/backedge and source/cubin hash. Whole-kernel static counts include setup/producer/nonlinear suffix and outlined code; K-loop counts exclude outlined wait/collective bodies. They are not dynamic latency or instruction-retirement measurements.

Shared8 compact remains a one-CTA shared-memory-budget design; observed occupancy is unset. The smaller dual geometry admits a different resource model, not an established two-resident-CTA result. Registers, allocation granularity, shared carveout, initial fill, scheduling and synchronization remain charged. Neither 242 versus272 sites nor 18.75% fewer payload bytes predicts whole-model TPS.

## Complete costs

All compact8/split53 routes retain 36,208,640 payload bytes/layer, versus 44,564,480 native; 2,317,352,960 extra prepared bytes across64 layers. At two token CTAs, logical weight TMA bytes are72,417,280 versus89,128,960/layer/decode. Original weights/scales and their consumers remain retained; this is extra memory, not net model-memory saving.

Shared8 uses the existing26,244-byte global book and adds26,368 bytes/CTA dynamic reservation (124 unused alignment bytes). Each of136 CTAs reads/copies6,561 words once: **3,569,184 logical global lane bytes/layer/decode**, before the K loop and before weight issue. The loop still performs **178,257,920 logical shared lane lookup bytes/layer/decode**. The dual variant has272 CTAs and doubles the initial book cost. No DRAM, cache hit, traffic reduction, cold-start or latency measurement follows from these logical counts. Setup may delay prefetch; it is not free.

On the retained block0/first32rows/K5120 fixture, full8's maximum distinct U32 words per bank averages3.5359375, worst7; split5's average3.15078125, worst5; split3 always1. This is an exact address-multiplicity model over1,280 warp-load sites with broadcasts merged, not hardware bank-conflict or latency measurement. More CPU rows would not establish a timing result.

Split5+3 chooses separate low-five and high-three coordinates, with low8-bit keys in8 U32 lane-striped planes and high5-bit keys in5 planes. Same13 planes/1,664 bytes; no regrouped gate/up intermediate arrays. Book values occupy972+108 bytes (three-trit U32 entries;54 bytes suffice abstractly), with1,152 reserved shared bytes in the actual shared implementation. The shuffle route reserves1,024 and retains one codebook word/lane; its collective handling is included separately in full-kernel SASS. Smaller tables did not give the smallest compiled route.

Literal3 is a distinct no-lookup candidate: 37,601,280 prepared bytes/layer,75,202,560 logical TMA bytes/layer/decode,2,406,481,920 bytes/64 layers; **15.625% below two-bit**, versus18.75% for Dict8. No weight-transfer padding:1,728 is16-byte aligned. Ten groups/lane each store L:U32 (four codebits in each nibble) plus H:U8 (fifth bits); the final two native words share one original-style U32. Physical pair offsets: L[group*128+lane*4], H[1280+group*32+lane], tail[1600+lane*4]. Groups cross gate/up storage coordinates only; the exact separate operands are reconstructed before the original MMA path.

## Six-gate code, with boundaries paid

For input bits a..e, low→high, NVIDIA LUT index4x+2y+z:

- M0=LUT(a,d,e,0xd5); S0=LUT(d,c,M0,0x82)
- M1=LUT(b,d,e,0x97); S1=c AND M1
- M2=b XOR M1; S2=LUT(a,b,M1,0x24)

All32 inputs yield native magnitude/sign pairs; image size27. The canonical inverse chooses least preimage and reserves codes3,7,22,23,28. Host parsing rejects aliases, out-of-range codes and nonnative nibbles with typed errors. Hardware aliases are completely specified; no signed-zero substitution.

Six is the one-result bitplane-gate lower bound in the declared grammar (six distinct nonconstant required output planes, none raw/complemented input), **not a general ISA lower bound**. `circuit/Carrier.lean` proves finite coverage, canonical injection/roundtrip and the boundary facts; it does not prove NVCC, native execution, or a globally minimal physical decoder.

Eight triplets produce three exact original native words, with no output shuffle: **25 specified data operations/24trits**, comprising10 shifts and15 LOP3 (input alignment, byte spreading, six gates and native assembly included). Ten groups plus the joint native two-word tail cost253 specified data operations/256trits before loads/addressing/scales/MMA. Compiled full-loop361 already prices the actual optimizer/schedule. `literal.cuh` is the exact device continuation used by that compiled kernel; the worker device alternative is retained for provenance only.

## Evidence and reproduction

`python3 reproduce.py` regenerates derivatives, builds cubins only, dumps SASS/resources, compiles/runs CPU witnesses, checks suffix identity, and checks both Lean files. It never opens CUDA, a GPU/model/DLL, or a serving process. Compiler/CPU commands have55-second child deadlines. Inspect retained jobs/receipts rather than resubmitting an uncertain command.

- `check.cpp`/`cpu.json`: all6,561 valid split labels,1,631 rejected13-bit labels, invalid native nibbles, shape errors,40,960 actual source operand words; zero mismatches.
- `check_literal.cpp`/`literal-cpu.json`: entire1,728-byte physical layout and exact original operand positions,40,960 actual words; zero mismatches.
- `circuit/witness.cpp`: exhaustive raw/canonical code map, slot isolation, byte spreading and two-slot tests. `circuit/Carrier.lean` proves the finite code; `JointCarrier.lean` proves split coordinate identities/injectivity and conditional opaque suffix preservation.
- `source-contract.json`: identical post-decode arithmetic/source suffix; source bindings for all inherited local headers, generated kernels and CPU fixtures. This is source evidence, never native floating execution proof.
- `build/`: exact cubins, compiler logs, SASS, loop slices and resource observations. `build-receipt.json` binds actual commands and returned status.

The fullshared book copy is identity on all6,561 entries; the existing13-bit payload remains byte-compatible with the separate offline cache owner5bdb08fa. Its source/semantic header binding must still be deliberately updated for any new integrated consumer. No cache files/owner artifacts were changed. Split53/literal3 require genuinely different payload semantics, preparation and cache identities. Their full-model preparation/startup is unset; the small CPU fixture does not claim a full real-model cache.

Original Bashf7ebf3d1 was reconciled completed/exit0; it was not replayed. Parent source3fdbad454 is private and remains frozen. This experiment changes no serving source, model, plan, DLL, profiler e732787c artifacts or cache66909408 artifacts. GPU use, adoption and profiling still require the separate owner's next authorized native window after Tau2. Whole-model speed, strict endpoints and original15 task quality are unset for every candidate here.
