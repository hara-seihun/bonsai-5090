# Native complete-FFN continuation

Start: original signed A8 K128 carrier for normalized input; original trained Q1 gate/up/down weights with their original FP16 K128 scales.
End: complete trained down-projection output. GU and SiLU×up tensors are not public boundaries.

`whole_ffn.cu` prepares gate/up rows into alternating 128-gate/128-up blocks. Each native three-plane tensor-core CTA owns both halves of one hidden K128 group, accumulates all original K128 input groups, and applies the actual native SFU SiLU×up continuation in shared memory. It computes original dynamic K128 A8 scales, RNE codes, DC/status/error, and writes exact three-plane digits directly in the down consumer's swizzled layout. `next.q=nullptr` removes the A8-byte tensor too. The downstream native `signed_fp4::project` consumes those digits and produces the complete original trained down output.

There is no approximation to SiLU or weight scales, no prepared activation table, no learned replacement, and no dropped down projection. Both controls and candidate use the same input A8 codes. Native scale accumulation association is split/schedule-specific; comparison to original row-major integer-dot/F32 accumulation is reported separately from identical-plan parity.

## Contracts

- Gate/up hidden width is divisible by 128; input K divisible by 128. Other dimensions are typed `cudaErrorInvalidValue`.
- Supported producer schedules: F4G2T8, F4G2T16, F4G2T32, F8G1T32. Gate/up uses one complete K split, not partial gate/up global tensors.
- Input and output digit encodings must be `ExactA8`. All 256 input A8 codes are represented exactly by the existing three-plane construction.
- Invalid input status propagates to nonfinite hidden-group status and NaN complete down output. Error aggregation remains sticky until the caller clears it.
- All-zero groups write zero scale/codes/DC with valid status. Padding in the last 64-row digit tile is explicitly zero every call. Partial row tiles are supported.
- `configure` is called once outside graph capture. Preparation is static; it changes no sign/scales. Replacing ordinary prepared GU adds zero static bytes. Retaining the original GU fallback alongside the candidate adds 25,067,520 bytes/layer, or **1,604,321,280 bytes across 64 layers** (~1.494 GiB). Original down signs/scales are reused.

## Measurement

`whole_ffn_bench.cu MODEL R OUTPUT` loads real layer0 D5120/FF17408 gate/up/down. It prices input digit encoding, full gate/up accumulation, native SFU, original A8 dynamic quantization and direct digit encoding, and complete trained down with all split folds. A second boundary also charges the dynamic F32→A8 producer. Warm and cache-evicted graph medians are separate. Control GU schedule/splits include GU32 split1 and the engine GU16 split4, with down16 split8; candidate down configurations are explicitly priced rather than omitted.

Inputs are deterministic F32 vectors quantized with the original K128 A8 producer, not real model activations. Parent owns real-activation and whole-model quality/task integration.

Preparation and numerical receipts, including same-plan carrier and complete-down bit-exact checks, live in `whole-ffn-results/`. GPU jobs run under `/tmp/binary-gpu.lock`, each computation with `timeout 55` after lock acquisition. The parent owns immutable source integration and full-model decisions.

`whole_ffn_controls.cu` additionally measures complete four-digit and DP4A FFNs, including the existing fused DP4A GU→A8 continuation. Its bounded native control space covers five GU schedules, GU splits 1/4/8, down schedules16/32 and splits4/8/16. DP4A down groups/split4/16/32/136 × threads64/128/256 are priced, with GU fused tiles4/8 and groups/split4/16/40. R5 uses DP tile1 and also prices materialized exact3 and native down8. Every quoted complete control includes the actual nonlinear/requantization/down path and all split reductions.

## Results

RTX5090, driver595.58.03, 500W limit. Timings are CUDA-graph medians of seven measured replays after two initial replays; cold runs evict128MiB before the timed map. Input normalization is the supplied State1; the F32 boundary additionally charges its original A8 quantization producer, not an unmeasured RMS normalization.

### Complete map versus materialized exact3

Final `r16.jsonl`/`r32.jsonl`, input A8→input digit encoding→complete trained down, microseconds:

| Rows | Best materialized warm/cold | Compound warm/cold | GU / down geometry |
|---|---:|---:|---|
|16|76.736 /81.920|75.008 /79.872|GU F4G2T8 split1 / down F4G2T16 split16|
|32|103.392 /110.336|93.408 /98.048|GU F4G2T16 split1 / down F4G2T16 split8|

R32 saves9.66% warm and11.14% cold in this complete region. The original engine GU split4/down split8 control remains a distinct record in each file; it is not relabeled as the best control.

Same-plan A8 bytes/scales/DC/status, every next digit including padding, and complete original trained down are **bit-exact**, for R5/16/17/31/32/65 and the four supported GU schedules. Complete `project` residual witnesses use alpha1/beta1, null next.q and guarded outputs. R17/31 explicitly pass both stride5120 and stride5152. All256 given-A8 codes appear in the tail witnesses. Zero output, invalid-status NaN propagation, input ScaleRange retention, sticky errors and undersized scratch errors pass. Against independent original row-major integer-dot GU→native SFU/A8→integer-dot down, the selected compound paths have relative RMS about1.7e-7 (R16) and2.3e-7 (R32); other split associations can change subsequent A8 codes, and are reported separately.

### Matched whole-block four-digit and DP4A controls

Latest `controls-r16.jsonl`/`controls-r32.jsonl` after the bounded DP down sweep, given-A8 boundary. Warm/cold columns select their respective best records, not necessarily one common plan:

| Rows | Four digits | Materialized DP4A | Fused DP4A GU→A8→down | Compound exact3 |
|---|---:|---:|---:|---:|
|16|85.152 /91.904|83.104 /100.352|74.848 /81.664|75.840 /79.872|
|32|113.920 /120.832|170.080 /182.272|146.528 /155.392|95.136 /98.048|

R32 is cheaper than both tested four-digit and DP4A whole-block controls. **R16 is not warm-fastest across DP4A controls.** With the F32→A8 producer included, R16 compound76.544/81.920 versus fused DP4A75.840/79.872 also rejects a blanket R16 win. Candidate/kernel pins are unchanged between these matched control windows and the exact3 comparison window.

### Actual R5 continuation: rejected for promotion

`controls-r5.jsonl`, full F32→original A8 producer→map→native SFU→requant→complete trained down, microseconds:

| Route | Best warm | Best cold |
|---|---:|---:|
|Compound exact3, GU8 / down8 split16|60.544|63.232|
|Materialized exact3|60.256|65.504|
|Materialized DP4A|38.144|59.136|
|Fused DP4A GU tile1→A8→down|47.968|55.264|

The native compound is not the cheaper R5 route. Keep the ordinary/fused DP4A option for that actual downstream state; do not extend the compound policy to R5 on these prices. This is a priced candidate rejection, not a general nonlinear obstruction.

### Complete joint-two-FP4 control

`whole_ffn_joint_control.cu` starts from common F32 postnorm input and charges both original primary/residual producers, joint GU, native SFU, both next producers and complete down. Its paired window measures:

| Route | Warm/cold µs |
|---|---:|
|R16 joint GU16 split1/down8 split8|63.552 /73.696|
|R16 compound originalA8, selected plan|95.744 /100.352|
|R16 ordinary ordered two-pass|93.280 /102.144|
|R32 joint as two independent R16 tiles|119.840 /131.072|
|R32 compound originalA8, selected plan|122.944 /126.944|
|R32 ordinary wide ordered two-pass|101.376 /112.384|

The joint API supports R≤16; tiled R32 is declared explicitly. NVFP4 primary/residual and originalA8 are different dynamic quantization maps. Joint versus its own two-pass reference has relative RMS4.16e−5/3.37e−5; twoFP4 versus A8 differs about.00817/.00795. Prior parent twoFP4 Wiki PPL19.2028/KL.00037166 is a separate operating point, not a quality gate rerun here. This rejects an absolute compound cost advantage over twoFP4. Window-specific timings are retained separately from earlier minima; the accompanying four/DP records are declared plans, not a new best-plan sweep. `whole-ffn-results/joint-README.md`, raw JSONL and provenance retain boundaries, source/linked-object pins and all controls.

## Integration conventions

`prepare` consumes original raw gate rows followed by original raw up rows. `project` consumes a tightly indexed current-row carrier view and its independently encoded digits, then writes `output <- alpha*full_down + beta*output`.

For a row slice starting at `offset`: input.q +=offset*D; input.scales/DC/status +=offset*(D/128); keep the same aggregate-error pointer. Each ≤32-row slice uses independent 64-padded arenas: input digits491,520B for D5120 and next digits1,671,168B for H17408. Do not slice a larger digit tensor using ordinary row-major offsets. Next scales/DC/status are locally indexed [sliceRows,136]; next.q and next.planes may be null. Down partial capacity is a count of floats, `rows*5120*splits`; R32/down8 requires1,310,720 floats. Output may point to residual+offset*D with strideD.

R16 uses GUF4G2T8/downF4G2T16split16; R17..32 uses GUF4G2T16/downF4G2T16split8. Configure both producer schedules and selected down schedule before graph capture. The parent owns the selective engine policy and actual quality/task/resource gates.

The compound removes global GU/hidden/A8-q storage of3,620,864B atR16 or7,241,728B atR32. The remaining downstream carrier is already in the native down consumer's coordinates. First static relabel price is about68us/layer in these runs; it is preparation, not recurring work.

## Reproduction and evidence

Build with `nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc`. Link whole_ffn.o, signed_digits.o, bitserial.o, digits.o, kernels.o, gguf.o; controls additionally link composition.o. Existing foreground kjob wrappers are `whole-ffn-results/run.sh`, `run-controls.sh`, and `run-r5-controls.sh`.

- `summary.json` is regenerated by `summarize.py` and preserves selected plans and separate boundaries.
- `price-provenance.sha256` binds the main priced benchmark snapshot `priced-whole_ffn_bench.cu`; `provenance.sha256` binds the added literal strideD tail witness benchmark. Native CU/CUH/object hashes are identical.
- `controls-provenance.sha256` binds the R16/32 DP down sweep snapshot `dp-down-whole_ffn_controls.cu`; `r5-controls-provenance.sha256` binds the standalone current controls source/R5 executable. Earlier selected-down controls retain their own source and manifests.
- The exported measured `whole_ffn.o` is retained at `~/data/binary-5090/whole-map/bin/whole_ffn.o`, SHA25621e4ba34c634c3024b44c55e79b99e5726f7ac9623398ceefdb03256bd955f23, bound by `engine-results/data-custody.json`.
- `whole_ffn.sass` contains actual native `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, native SFU and fused carrier stores. `resources.txt` records compiled resources. F4G2T32 spills56B/thread; F8G1T32 removes spills but loses at132/134.912us for R32/down16split8 in its trial, so it is not selected.
- `Relabel.lean` kernel-checks the actual layer0 row permutation inverse/range, while the existing `ThreePlanes.lean` owns exact signed-byte three-plane labels. Neither prices CUDA execution or certifies native accumulation.

