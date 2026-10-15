# Dense5 single-stream compact operand reader

Starting engine: `832ef1b`. Parent: single-stream thread `3cc763d4`.

A distinct reader from F03's scalar query contractions and F04's eight-trit/13-bit packets: concatenate a lane's sixteen native 8-nibble MMA operand registers into 128 trits, then store 26 five-trit radix3 bytes. Six uint32 planes and one uint16 plane give coalesced warp reads. A 1KiB per-CTA book maps each five-trit code directly to five E2M1 nibbles; constant shifts splice sixteen operand registers. No global/intermediate expanded weight image, FP32 query scratch, support assumption, or scale approximation. The final code has two zero padding trits. W128 and A16 boundaries remain unchanged because the compact stream only changes weight register wiring, not the consumer's multiplication/scaling order.

`generate.py` reads the owning `src/fp4.cu`, compiles the WF0 control and WF3 compact reader through an identical benchmark call interface. It retains the current pipeline, all18 geometries, FP16 scale arithmetic, MMAs, splitK, fused GU epilogue and output. It emits separately compiled control/candidate namespaces and removes unused formats/add modes equally. Opt-in engine integration uses `H5_COMPACT=dense5`; no permanent expanded or duplicate ternary image is kept. All ternary projections/head use WF3, including prefill; the drafter/alpha-beta NVFP4 format is unchanged. `H5_WSCALE=e4m3` and the megakernel are explicitly rejected. Default behavior is unchanged. Whole-model acceptance remains required.

Code plus original FP16 W128 scale: 1.750bpw versus2.125bpw. Shared scratch gains1024bytes for the book while staged weight scratch shrinks. Compiler register counts matter: all26 expanded book results are live near register assembly, so a smaller stored image is not itself a speed claim.

## Checks and run

CPU host guard (seconds, one thread):

```
python3 experiments/dense5/generate.py
g++ -O3 -std=c++17 -Wall -Wextra -Werror -Iexperiments/dense5 experiments/dense5/host.cpp -o build/dense5/host
build/dense5/host
```

Passed2,886,248 nibble checks, all59,049 adjacent label pairs at all five starting offsets and4,096 complete lane streams. GPU certificate checks the actual device reader's3,888 native operand registers over all243 five-trit labels. Full consumer guards compare every real GU/SiLU and down output bit-for-bit against baseline at the identical geometry, splitK1. Independent tuning considers all current WG2 GU geometries and all18 down geometries with splitK1/4/16. Timed splitK retains the engine's nondeterministic atomic FP32 ordering.

GPU run only after parent assigns the serial turn:

```
H5_MODEL=/path/to/original-model.gguf bash experiments/dense5/run.sh
```

CPU-only CUDA compilation: add `BUILD_ONLY=1` to that command. Compile the four units sequentially, <=4 nvcc threads. Use parent's durable service if a compile exceeds55seconds. After a CPU-only build, the bounded native check/timing can run directly:

```
timeout 55 build/dense5/native "$H5_MODEL" experiments/dense5/input.bin > results/dense5/native.jsonl
python3 experiments/dense5/summarize.py
```

Harness/fixture ancestry: F04 native benchmark from `experiments/f04-decoders` supplied the matrix loader, identical-geometry guard, graph timing and tuner structure. Retains the same eight real normalized layer10 input rows, with provenance in `input.json`. Only rows1/8, not512, are assigned here. GU5120→34816 and down17408→5120 are the actual model shapes. Both dynamic sign/Hadamard/FP4 producers, GU/SiLU, hidden producer, down, splitK zeroing and stores are in the graph; common upstream RMSNorm and final residual are omitted equally. Six rotated rounds × six repetitions; warm and128MiB cache-perturbed, eviction outside timing. This is not full-model quality/HTTP acceptance.

## Native result on the assigned dedicated5090

Matched bitwise guards passed every GU/SiLU and down element at rows1/8, plus3,888 native operand-register checks. Six rotated rounds × six samples:

| Rows | Cache | Current µs | Dense5 µs | Ratio |
|---|---|---:|---:|---:|
| 1 | warm | 38.856 | 43.315 | 1.11475 |
| 1 |128MiB perturbed |65.533 |62.624 |0.95561 |
| 8 | warm |39.544 |44.517 |1.12577 |
| 8 |128MiB perturbed |66.715 |63.816 |0.95655 |

Warm loses; cold wins4.4%. A full-model token streams severalGiB, so the cold result justified opt-in integration, not default adoption. Per-CTA table setup is included. Weight format is1.750bpw including scales versus2.125bpw. Current native GU winner: fw8/wg2/tok16; compact GU winner: fw4/wg2/tok32. Down uses splitK16. Compiler logs show some larger configurations spilling; tuning retains them in the search rather than assuming the compact format is universally cheaper.

Artifacts under `results/dense5`: host/native JSONL, final result JSON, hardware, compiler logs, compressed SASS and input hash. The original standalone build/run exit was0. A subsequent full production `fp4.o` CPU compile exceeded the worker's48second window; parent completed the foreground production build. Parent's first matched ordinary4k whole-model run: compact160.3tok/s versus153.6 baseline (+4.4%); resident weights5.95GB versus7.15GB. This is one end-to-end speed result, not completed task-quality/replay acceptance. Parent owns further integration and full-model acceptance; production build log is preserved at `~/work/halo-single/dense5/production-build.log`.
