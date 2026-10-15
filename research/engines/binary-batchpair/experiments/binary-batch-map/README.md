# Binary batch-axis packed integer map

Owner: October10 binary Bonsai27B RTX5090 campaign. This directory owns an opt-in standalone candidate and its proof/measurements; the modern engine and whole-model batch timing belong to the integration worker/parent. No model throughput is inferred from these projection measurements.

## Measured result

**Mostly a bounded negative versus selected DP4A.** One small GU/B2 win does not offset the head/down losses. Keep this as an explicit candidate, not the default.

Inclusive cold microseconds for the complete batch, producer+consumer+continuation:

| Geometry | B | DP4A | Batch-pair IMAD | DP4A/pair |
|---|---:|---:|---:|---:|
| GU34816×5120 |2|36.864|32.768|1.125|
| GU |4|38.912|40.928|0.951|
| GU |5|43.008|43.008|1.000|
| GU |8|43.008|49.152|0.875|
| GU |16|63.520|88.064|0.721|
| GU |32|114.688|161.792|0.709|
| down5120×17408 |2|18.432|22.528|0.818|
| down |4|18.432|26.592|0.693|
| down |5|22.528|26.656|0.845|
| down |8|28.672|30.720|0.933|
| down |16|38.912|51.200|0.760|
| down |32|65.536|92.160|0.711|
| fullhead248320×5120 |2|129.024|145.152|0.889|
| fullhead |4|137.216|188.448|0.728|
| fullhead |5|165.888|231.424|0.717|
| fullhead |8|212.992|286.752|0.743|
| fullhead |16|362.496|540.672|0.670|
| fullhead |32|720.896|1093.632|0.659|

Best pair plans: head16/128 (chunk/threads) throughout; down B2=1/128, B4/B5=4/256, others4/128; GU B4=4/256, B32=16/256, others4/128. `summary.json` retains exact chosen raw records and separately measured producer/consumer cost. All162 projection-plan runs checked **173,818,368 outputs**; worst absolute A8-reference error4.76837158e-6, relativeRMS2.46981953e-7. Unsplit candidates match the exercised independent reference bitwise. Native K128 witness passes.

Device: RTX5090, driver595.58.03,500W power limit, CUDA13.1/sm_120a; timing October10 around20:16UTC. GGUF SHA256 remains campaign-pinned17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0. The later API boundary guard tightening does not change measured producer/projection kernels; `binary-sha256.txt` pins the actual timed binaries.

## Map, lane separation, and relabel inversion

One sign bit `b` is reused across two independent token streams, not two fragments of the same stream. For full signed-A8 `q∈[-128,127]`, lift `u=q+128∈[0,255]`. For a K128 block define

```
A_i = u_i(token0) + 65536*u_i(token1)
S   = sum_i b_i*A_i       // mad.lo.u32 directly in packed coordinates
count = sum_i b_i         // one static uint8 per original weight block
DC_t = sum_i q_i(token t) // recurring, once per producer and shared across rows
dot_t = 2*lane_t(S) - DC_t - 256*count
```

Each selected-positive lane sum is at most128*255=32640<65536. Every partial sum is nonnegative and bounded by the final sum. The packed total is below2^32: neither cross-lane carry nor native32-bit overflow occurs. Four independent K32 partial labels shorten recurrence; adding them preserves the same bound. Original FP16 weight scales and FP32 activation scales are applied **per K128**, before split-K/output continuation.

For fixed DC/count, the continuation `S_t→2*S_t-DC_t-256*count` is injective, with inverse `(dot_t+DC_t+256*count)/2`. The retained DC is essential across different activation inputs; the selected label alone is not claimed sufficient. No ideal-floating identity is substituted for native FP32 scale continuation.

`BatchMap.lean` is kernel-checked against installed Mathlib in `/opt/lean-mathlib`: universal two-lane separation, packed step identity, selected K128 bound, arbitrary full-block signed correction, and affine inverse. No admissions/native-computation axioms. The proof was checked **before** CUDA implementation. `results/lean-check.log` records dependencies.

The realization search here compares this one two-lane IMAD carrier/schedule family with the existing selected 0/1-byte DP4A map. It is not an exhaustive search over all ISA programs or static weight encodings. Full-A8 K128 selected sums require15 bits per lane; this construction uses16-bit lanes to make extraction cheap. Its smaller register footprint competes with DP4A's four-K-products/instruction.

## API and complete bill

`batch_map.cuh`/`batch_map.cu` expose explicit CUDA errors and caller-owned storage. Existing transposed signs/scales remain16+2 bytes/block. `prepare_counts` adds **one immutable byte/block**, making the prepared weight layout19 bytes/K128=1.1875bits/weight, versus DP4A18 bytes=1.125bits/weight. Raw→transposed preparation and count generation are one-time load operations, outside recurring measurements. No dense ±1 weight tensor is allocated.

```
Carrier c{packed, metadata, K, wholeR};
produce(X, c, stream);  // wholeR 1..512, once for the complete batch
for (int first=0; first<wholeR; first+=16) {
  Plan p{N,K,min(16,wholeR-first),first,chunk,threads};
  project(p,signs,originalFp16Scales,counts,c,Y+first*N,
          partials,alpha,beta,stream);
}
```

Each projection supports tileRows1..16; first_token must be even. Tiling16 guarantees that invariant for every wholeR1..512. All tails are compiled. The carrier stores `ceil(R/2)*K*4` bytes, plus12 bytes/K128/real-token for scale/DC/status. Even batches pay2 bytes/activation, twice DP4A's signed-A8 storage. **B5 pays three pairs**:12K bytes vs5K A8 bytes, including the full ghost lane and its arithmetic. The ghost has lifted value0, contributes nothing, and is never observed as an extra output.

The fused producer reads FP32 inputs, independently reduces max/DC, uses the same round-to-nearest-even A8 rule as bitserial, and writes paired labels directly; it does not produce then transpose an intermediate q array. All-zero groups remain valid with q=scale=0. Nonfinite input or a nonzero max outside[1e-30,1e30] sets the existing typed status and yields NaN projection; optional persistent metadata.error aggregates failures. Metadata.q/planes are unused by this route. Storage is caller-owned and the caller checks asynchronous statuses after its normal synchronization.

The consumer reads original sign words, extracts each0/1 multiplier into a register, and executes native32-bit multiply-add against paired activation labels. This recurring extraction, packed loads, selected-sum accumulation, lane extraction, DC/count correction, original scaling and final alpha/beta all lie inside the timed graph. Split partials cost4*tileRows*N*ceil(groups/chunk) bytes and their write/read/continuation is charged. B32 produces once then executes two16-row consumers, both included.

## Numerical and instruction receipts

`native_witness.cu` executes actual native `mad.lo.u32` through128 steps for every pair of signed-byte values and both uniform sign bits:131,072 cases /262,144 integer outputs, including extremal lane sums and signed dots. This is finite device evidence alongside the general Lean proof, not an exhaustive K128-vector census.

`bench.cu` loads actual trained GU, down and fullhead weights from the parent GGUF. Activations are deterministic generic FP32 values, not captured hidden states. The independent GPU reference reads original raw sign bytes, accumulates K128 int32 dots and applies original scales. Every output of every plan is checked; producer labels/scales/DC/status are also compared with the existing A8 producer. Floating split-K association errors are recorded as maximum absolute and relativeRMS, not silently described as bitwise equality.

`results/build.log` records registers/spills for allR1..16. The pair R5/R8/R16 kernels use46/56/92 registers with **no spills**, versus existing DP4A's larger R16 register footprint. `results/sass-receipt.json` records opcode counts and SASS block hashes for all16 row-count kernels; native witness SASS is retained separately. Native integer instruction evidence and projection timing are separate results.

## Reproduction

From this checkout:

```sh
cd /opt/lean-mathlib
lake env lean ~/work/clones/binary-spec-5090/experiments/binary-batch-map/BatchMap.lean
# From repository root; no dependency installation:
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas=-v -I./src \
  experiments/binary-batch-map/batch_map.cu experiments/binary-bitserial/bitserial.cu \
  experiments/binary-batch-map/bench.cu src/gguf.cpp -o /tmp/binary-batch-map-bench
nvcc -O3 -gencode arch=compute_120a,code=sm_120a \
  experiments/binary-batch-map/native_witness.cu -o /tmp/binary-batch-map-witness
```

Parent-granted suite runs on existing instance55260842, holding `/tmp/binary-gpu.lock` with55-second execution cap after lock acquisition. `run_gpu.sh` owns all18 tensor/batch cases and exports device/binary identities. Nine medians after3 warmups, warm and cold; cold evicts128MiB outside timed intervals. Candidate chunks1/4/16/all and threads128/256 are priced; comparator is unchanged `bitserial::select` DP4A (R32 tiled16), not retrospectively retuned. Production and consumption are separately measured for diagnosis, but selection uses **inclusive full graph cold latency**, not the sum of separate measurements.

`python3 experiments/binary-batch-map/summarize.py /tmp/batch-map-sass.txt` records SASS and summarizes complete receipts. Whole-model adoption/quality and batch tok/s remain with the parent/integrator.
