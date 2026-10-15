# Sparse exact third-digit DOWN sidecar

Starting state: the measured original-A8 three-plane consumer is exact, but captured hidden packets need nonzero balanced third digits on only a small minority of coordinates.
Target state: the coordinator knows whether replacing the third dense MMA with a priced sparse integer continuation makes the complete trained DOWN boundary faster, while retaining every signed byte and original scale/status semantics.

This sidecar is separate from whole-FFN/serving gates and does not alter their engines or dispatch. Parent owns the GPU lifecycle and promotion. No positive performance result is assumed.

## Carrier and native path

The 256-entry balanced grammar in `sparse_third.py` chooses `(a,b,c)` with `q=a+4*b+16*c` and digits in twice-E2M1. It minimizes `(c!=0, abs(c), abs(a)+abs(b), a, b, c)`. This represents every byte exactly; 103 byte codes admit c=0. The generated native label is two FP4 nibbles for `a/2,b/2`, followed by signed c in its upper byte. The table is 512 bytes in CUDA device global read-only storage, not a uniform constant broadcast table.

The recurring single encoder packs two dense planes and compacts c!=0 coordinates per row×K128 into up to32 `(position:uint7,c:int8)` uint16 entries. Four adjacent encoder threads cover each row's128 coordinates; segmented shuffle prefix sums allocate their entries. A uint8 count retains the full0..128 count. List slots beyond count are never read.

If count<=32, the native consumer runs four FP4 MMA instructions per K128 (two halves×two planes), then computes the integer c-dot from the compact list and the original Packed1 positive-sign bits. Recovery is `RN(2*d0)+4*RN(2*d1)+16*cDot`. If count>32, the same row/output/group takes the explicitly defined dense integer route: all128 original q bytes and sign bits form the full integer dot. No list truncation, dropped values, or substituted quantization occurs. The two lower-plane MMA results are ignored on this dense route.

Decode the integer group dot **before** multiplying the original FP16 weight scale by original FP32 sx, then perform the original RN FP32 group FMA. Split-K folds ascending partials in the same plan as the dense controls. Original q, sx, DC, status and sticky error stay live. Invalid producer status propagates NaN. Alpha/beta, stride and tail semantics match the dense source. Only Packed1 layout is accepted.

Bounds: a,b magnitudes<=12 and c magnitude<=6. A native half-digit dot magnitude<=768, integer sparse c-dot<=768, and full signed-q group dot magnitude<=16384. Counts fit uint8 and capacity32 list entries fit their allocated slab. Signed integer arithmetic has ample headroom. The full256 label identity and c<=6 bound are stated in `sparse-digit-results/sparse_a8.lean`, emitted from the exact native labels by `sparse_digits_generate.py`.

## Residency and boundaries

Caller-owned scratch: two dense64-row-padded activation planes (1 byte per padded activation), one count byte per padded row×K128, and32 uint16 list entries per padded row×K128. K17408,R16/R32 each uses1,114,112 dense bytes +8,704 count bytes +557,056 list bytes =1,679,872 bytes. No extra resident weights. The list is globally read and cache-reused; dense activation TMA staging fetches two planes rather than three.

The bench reads actual trained layer0 DOWN5120×17408 weights and the original32-row F16 capture's hidden F32 values. Row stack positions127..158 are independent exported model observations, **not an actual simultaneous model batch**; R16 uses its first16rows. Both controls and candidate invoke the same original float32 A8 producer. Host checks compare q and sx against the exact original float32 producer formula before timing.

Cold and warm measurements retain four boundaries: prepared projection; encoder/list producer + projection; original A8 producer + encoder/list producer + projection; that complete path + common residual/next-A8 continuation. Static weight transposition, graph capture, allocation and host export are excluded. Recurring sparse lookup, compaction, correction, original scaling and split reduction are included. The fourth boundary's residual is deterministic synthetic, not a captured residual; the hidden input is actual captured F32 in every boundary.

DP4A, exact3 and four-digit controls use identical q/sx. Same-plan output bit equality is checked against exact3; exact3 itself is checked against same-plan four digits. Independent raw Q1 dot checks retain split-association tolerance. All five schedules and split1/4/8/16 are retained for both R16 and R32.

Native witnesses cover dense all256 byte patterns; a sparse-list case (every16th coordinate is127, other coordinates−3..3); high-density127 requiring the full raw-q dense route; and all256 byte codes **isolated one per K128** so the sparse continuation itself must recover each code. Each tests R5/16/32/65, all schedules, split1/4/8, bitwise same-plan exact3 output, guard/tail/stride/alpha/beta behavior, invalidstatus/nonfinite/range errors, sticky error after validzero, zero, unsupported layout and undersized plane/count/list storage rejection. Modes0/1/2/3 in `witness-r16.jsonl` are dense all-byte, sparse, dense127, isolated all-byte, each60rows. `first-witness-r16.jsonl` retains the initial180-plan round.

## Custody

Owned sources: `sparse_digits.{cu,cuh}`, `sparse_digits_bench.cu`, `sparse_digits_generate.py`, `sparse_digits_bench_generate.py`, `sparse_digits_run.sh`, `sparse_digits_summarize.py`, this document and `sparse-digit-results/`. No changes to `signed_digits*`, whole-FFN or engine source.

Remote source/object: `/workspace/binary/digits-ragged/halo/experiments/binary-whole-map/build/sparse_digits.o`. RTX5090 sm120a CUDA13.1.115, same pinned `bitserial.o`, `digits.o`, `kernels.o`, `gguf.o` and measured `signed_digits.o`. Native source/object hashes, ptxas resource report and compressed SASS are retained beside result JSONL. The exported strengthened-witness object is retained at `~/data/binary-5090/whole-map/bin/sparse_digits.o`, bound by `engine-results/data-custody.json`. Every GPU window uses `/tmp/binary-gpu.lock` and timeout55 **after** lock acquisition. The local finite kjob has this sidecar's thread owner; no polling, new rental, public push, commit or recurring wake is required.

Reproduce after the generator inputs are present:

```
python3 experiments/binary-whole-map/sparse_digits_generate.py
python3 experiments/binary-whole-map/sparse_digits_bench_generate.py
bash experiments/binary-whole-map/sparse_digits_run.sh all
python3 experiments/binary-whole-map/sparse_digits_summarize.py
```

## Native result: stop this route

**The sparse continuation loses by6.19×/7.68× to exact3 at R16/R32.** It is a valid full-byte carrier, not a faster native DOWN route. Keep dense exact3; do not promote or credit a whole-model gain.

Cold complete boundary (original A8 producer + native encoder/list producer + projection/correction/split reduction + common residual/next-A8):

| R | Sparse2 µs | Exact3 µs | Best four µs | DP4A µs | Sparse cfg/split |
|---:|---:|---:|---:|---:|---:|
|16|188.416|30.432|34.784|40.928|0/16|
|32|345.856|45.024|49.152|69.632|0/8|

Cold bare projection is182.016/337.920µs for the sparse route versus24.288/38.624µs best exact3. Encoder/list production is fully paid, but removing it does not rescue the result. Every schedule/split and both cache states remain in the raw JSONL and `summary.json`.

Actual captured hidden c density is6.3936%/6.6550%, mean8.184/8.518 per K128; maximum32 and **zero dense-fallback groups** at either row count. Native original q and sx agree bitwise with the exact source float32 formula. All240 timed candidate/control plans per R are bit-exact to same-plan exact3; DP4A has a different split policy and is independently raw-checked. Maximum independent raw projection error across all timed arms is1.43051147e−6.

The actual ISA does lower the two-plane arithmetic to `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. Sparse positions then require U16 global gathers, position/sign extraction, signed integer accumulation and variable-length loops for each output pair. The best cold schedule cfg0 uses96 registers and44B spill stores/76B spill loads per ptxas static report; cfg1 has150 registers and no spills but also loses. Larger schedules spill more. These are concrete properties of this lowered implementation, **not a nonlinearity or impossibility argument**. The bounded price rejects this route without claiming that all sparse implementations must lose.

The compiler first reported an ADL ambiguity because Plan aliases the signed namespace; explicit sparse-qualified validation/size calls fixed it. `build-first-adl-failure.log` retains that receipt. Timing uses `timing-bench.cu` and `timing-provenance.sha256`; the final bench additionally strengthens the native isolated-code witness. Candidate source is unchanged between the two rounds. Timing linked object SHA256`48defe087ef8bbf5410ab9323a17c848600407afde0806d8d55e505e6c1e52e2`; the stronger witness rebuilt and exported object SHA256`f02aaecaf2de0fd887613fcbc8720da260d686165280b7256004db2c447d10e6`. The exported object/SASS belong to the stronger witness build; timing's immutable source/bench/object hashes remain in `timing-provenance.sha256`. Both builds use identical candidate source and flags. Subsequent runner invocations preserve round-labelled object/SASS/provenance artifacts rather than replacing the prior round's evidence.

Final native witness:240plans, all bit-exact to same-plan exact3; max independent raw error5.96046448e−7, max relativeRMS1.25266875e−7. Isolated-all256 mode forces the sparse continuation through every code. The generated Lean identity and c-bound completed with `lean_exit=0`. All receipts were exported by23:41:29Z; the sidecar stopped well before the rental guard.
