# LUT8 / LUT4-ILP bounded tuning round

October10,2026,18:52–18:53UTC, same RTX5090/CUDA13.1 as the original projection sweeps. Parent requested one short round after its full-model packed-F32 measurement. **Decision: keep the existing selected route. Neither candidate wins.**

`packed1.cu`, `packed1.cuh` and `packed1::select` are byte-for-byte unchanged from parent source2fe3332. Candidates are separately opt-in through `tuning.cuh`/`tuning.cu`, namespace `packed1::tuning`.

## Paired measured ladder

Cold microseconds for the **whole R-token projection**, each candidate's best configuration in the same sweep:

| N × K | R | Current LUT4 | LUT4-ILP | Compact LUT8 |
|---|---:|---:|---:|---:|
|5120 ×5120|1|10.208|10.208|10.208|
|17408 ×5120|1|16.352|16.352|16.352|
|5120 ×17408|1|16.128|16.128|18.176|
|248320 ×5120|1|118.784|118.784|122.880|
|5120 ×5120|8|18.176|20.192|32.736|
|17408 ×5120|8|38.624|40.672|81.888|
|5120 ×17408|8|40.672|40.960|86.016|
|248320 ×5120|8|417.792|428.032|1038.304|

LUT4 here is the paired current LUT implementation, not the default Sign winner for square R1 (original selected Sign8.192us). Thus neither new route improves the current overall selection there either.

Timing includes recurring activation-table generation, table staging, original K128 scale application, split-K partials, reduction and alpha/beta output continuation. CUDA graph, warm median15 and cold median9 after3 initial trials; each cold trial evicts L2 with128MiB outside the timed interval. GPU batch waited for the shared `/tmp/binary-gpu.lock`, held it once, then released it. No concurrent parent timing occurred. Weight and activation generators match the original generic-FP32 synthetic harness; these are format/dimension-matched projections, not trained-model quality scores.

`tuning-summary.json` records each chosen plan and links to raw JSONL in `results/tuning/`. All tested group chunks1,2,4,8,16,K/128 and thread counts64,128,256 are retained. No second tuning round or full-model candidate run was commissioned.

## What was priced

### Compact LUT8

One128-entry FP32 table per activation byte. Tables cover bit7=0. For bit7=1, complemented low-seven sign bits address the antipodal pattern and an FP32 sign-bit XOR negates its response. No activation quantization or weight expansion; each original FP16 K128 scale remains separate.

Lookup count halves from32 to16 per K128/output/token. Four independent subaccumulators shorten the addition chains. The recurring table is **4× larger**:64K bytes/token instead of16K;320KiB at K5120/R1,2.5MiB at R8. Dynamic shared memory rises from2KiB×R to8KiB×R:64KiB per R8 CTA. Every group stages this enlarged table, and every byte response requires complement/sign handling. These measured boundary costs outweigh the reduced lookup count. At R8, shared-memory capacity permits fewer resident CTAs; this is a resource consequence, not a measured counter attribution.

### LUT4-ILP

Same16-entry quartet tables and bytes as current LUT4. Split each K128 dot into four independent eight-lookup sums and combine them with a balanced tree. R1 yields no measured gain; R8 loses. Compiler register counts:40 at R1 versus current36;56 at R8 versus current40. Compact LUT8 uses40/48 at R1/R8. All candidates have **zero spills** (`results/tuning/tuning-build.log`).

No default map or plan was changed based on these local timings.

## Numerical checks

All outputs compared with the independent original-Q1 GPU FP32 reference. Across the selected entries in this round:

- Maximum absolute FP32-reference difference1.6784668e-4.
- Maximum relative RMS difference2.95281604e-7.
- FP64 mathematical oracle on64 distributed rows/token: maximum absolute error3.17044768e-5.
- Alpha=.75,beta=.25 with an **independent nonzero initial Y** passed for all paired shapes and R1/R8. Maximum reported absolute error1.2665987e-4 (`tuning-round.log`).

The route preserves original signs/scales and FP32 activations but changes accumulation order. No trained-model quality conclusion is drawn for either candidate.

## Explicit experimental interface

`packed1::tuning::Plan{n,k,r,groups_per_split,threads,Route::Lut8|Lut4Ilp}`. Device signs/scales use existing `packed1::prepare`. Call `tuning::configure()` once before capture; it opts LUT8 R7/R8 into64KiB dynamic shared memory. Allocate `tuning::table_bytes(p)` and `tuning::partial_bytes(p)` after successful `validate(p)`. Then call `tuning::project` with the same X/Y/alpha/beta/stream argument contract as the original interface. No automatic selection or engine integration is installed.

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xptxas=-v packed1.cu tuning.cu bench.cu -o bench-round
flock /tmp/binary-gpu.lock ./bench-round 248320 5120 8 lut paired-head-lut
flock /tmp/binary-gpu.lock ./bench-round 248320 5120 8 ilp paired-head-ilp
flock /tmp/binary-gpu.lock ./bench-round 248320 5120 8 lut8 paired-head-lut8
```

Measured source SHA256: tuning.cu `7fc8dc959c9349491cc79312212ce471627106e4656e453ae47ed8b8360f7085`; tuning.cuh `caf5215986b8e16effd61051eaa53f43543538f2dccbe9aa101d10d5539be1e3`; bench.cu `939c5f6bd37711e05bb5ef78dc961c1f2755e7c9db95094dc080ee02a9622bbb`.
