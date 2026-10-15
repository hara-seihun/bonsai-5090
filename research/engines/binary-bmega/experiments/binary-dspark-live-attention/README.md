# DSpark logical-history attention fixture

Source: [`../../bench/dspark_live_attention.cu`](../../bench/dspark_live_attention.cu). This standalone harness calls `h5::dspark::launch_attention`; it does not load a model. Local build below does not access a GPU.

## Measured keyhole

Existing rental55260842, CUDA13.1, one locked bounded window. Baseline3004518 versus live-position repair93835c0, same seeded inputs:

| Ring8196 live history | Physical scan median | Logical iteration median |
|---|---:|---:|
| 48 | 351.884 µs | 20.474 µs |
| 176 | 385.080 µs | 64.447 µs |

Both implementations passed all seven scalar-oracle cases, including missing-tag NaNs. These are isolated eight-row/8-Q-head/2-KV-head fixtures, not full-model or HTTP throughput. The live and baseline visitation orders differ on wrapped histories; maximum observed absolute scalar-oracle error was below 8e-7.

Raw logs and object/binary manifests are in `evidence/`. The first locally linked executable failed before CUDA because Ubuntu26 GLIBC2.43 is newer than the rental's Ubuntu24. The owning repair compiled the host fixture and linked it on the rental against pinned CUDA objects; no numerical source changed. The failed receipt is retained. Both remotely linked fixtures share the `impl=live-history` string; filenames and manifests identify which kernel object they actually link. Durable repaired job `binary-dspark-live-keyhole-host-repaired` exited0. Full-model acceptance remains with the campaign parent.

Compile the host fixture and link on the execution host when its glibc differs; transport the CUDA objects rather than a locally linked executable.

## Build

From the repository root, using CUDA 13 with sm_120a support:

```sh
mkdir -p build-dspark-live-attention
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc \
  -c bench/dspark_live_attention.cu -o build-dspark-live-attention/fixture.o
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc \
  -c src/dspark_kernels.cu -o build-dspark-live-attention/kernels-live.o
nvcc -gencode arch=compute_120a,code=sm_120a \
  build-dspark-live-attention/fixture.o build-dspark-live-attention/kernels-live.o \
  -o build-dspark-live-attention/live-attention
```

The exact physical-ring-scan baseline is commit `300451845798525f6171712210a3ef85a03f4d08`. Build the same fixture against that immutable source for paired event timings and the same scalar-oracle checks:

```sh
git show 300451845798525f6171712210a3ef85a03f4d08:src/dspark_kernels.cu \
  > build-dspark-live-attention/ring-scan-3004518.cu
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc \
  '-DDSPARK_FIXTURE_IMPL="ring-scan-3004518"' \
  -c bench/dspark_live_attention.cu -o build-dspark-live-attention/fixture-ring-scan.o
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Isrc \
  -c build-dspark-live-attention/ring-scan-3004518.cu \
  -o build-dspark-live-attention/kernels-ring-scan.o
nvcc -gencode arch=compute_120a,code=sm_120a \
  build-dspark-live-attention/fixture-ring-scan.o build-dspark-live-attention/kernels-ring-scan.o \
  -o build-dspark-live-attention/ring-scan-attention
```

Generated source, objects and executables under `build-dspark-live-attention/` are rebuildable and may be deleted as a unit.

## Run under parent's GPU custody

```sh
build-dspark-live-attention/live-attention --correctness
build-dspark-live-attention/ring-scan-attention --correctness
build-dspark-live-attention/live-attention --timing
build-dspark-live-attention/ring-scan-attention --timing
```

Both modes require CUDA device 0. Invalid CLI input and CUDA failures exit 2; status/output mismatches exit 1; only completed checks emit `PASS`. The timing mode runs all correctness cases and times only ring8196 histories48/176. Buffers are allocated/uploaded once per case outside timing; each case has eight warmups and five samples of 64 launches measured with CUDA events on a nonblocking stream. `TIMING` reports min/median/max microseconds per launch, including any stream scheduling/host submission gaps; it is an isolated API measurement, not full-model speed. Timing includes neither uploads nor the CPU oracle. Correctness is checked again after timing.

## Covered contract

Deterministic xorshift seed `0x31415926` supplies finite F32 Q and rounded F16 K/V. All fixtures use head_dim128, q_heads8, kv_heads2, slots2 and layers3; populated layers and slots have distinct values. The oracle computes double-precision scalar dot products and a separate max-subtracted softmax over the fixture's logical history list plus **every** explicitly selected current-block row, including future rows relative to each query. It does not replicate the GPU's online softmax or physical ring scan. Acceptance is `abs_error <= 2e-5 + 2e-5*abs(expected)` per finite output.

Cases cover:

- Ring8196 with 48 and 176 live historical positions and four current rows per slot, wrapping across the physical ring boundary.
- Ring64/window48 and ring256/window176, including a 16-row current block.
- `context_begin` clipping independently of `context_window` and anchor-zero clipping at position0.
- Layers0/1/2, distinct slots and grouped-query head mapping.
- Historical position `pos` in cache `pos % ring`, with exact tags.
- Current entries deliberately outside every natural current modulo index, disjoint from live history.
- Unlisted pending tags beyond the anchor with conspicuous finite poison K/V; expired history outside the context interval.
- One live historical tag replaced by `pos + ring` in layer2/slot0: expected sticky `missing_kv`, all outputs in that block NaN, other slot's outputs finite and oracle-matching.

The baseline and live executables use identical seeded inputs and oracle checks. Their physical/logical visitation orders differ, so the fixture compares numerical outputs to the common oracle rather than requiring bitwise equality.
