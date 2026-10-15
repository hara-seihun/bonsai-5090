# Original binary-weight census and down-class carrier

This sidecar examines **Bonsai-27B-Q1_0.gguf**, original GGUF type41, K128 FP16 scales and signs, D5120/FF17408. It does not edit the engine. `weight_census.py` and `weight_census.json` own exhaustive layers0/30/61 observations. Each layer takes about4 seconds on a CPU build host.

## Census result

Across all three selected layers:

- No identical/antipodal gate/up whole rows, including their joint pool. No identical/antipodal K128 blocks in either matrix or their joint pool. No down-matrix row or K128 block repeats.
- GU matched-coordinate sign disagreement is49.908–49.931%. Aligned8-bit words have entropy7.99976–7.99998bits, all256 values occur; all65,536 aligned16-bit values occur, with entropy15.9908–15.9915bits. Aligned32-bit words are almost all unique. These scopes are not a claim about every layer or every possible factorization.
- No identical/antipodal down columns across **all5120 outputs**, when the128 columns are compared within each original K128. Thus full-output exact column merging contributes no reduction at those boundaries.
- Every original FP16 scale in these nine matrices has its low3 and high2 bits zero (`OR=0x3ff8`). The label `original_fp16_bits>>3` fits11bits and reconstructs the original FP16 exactly by `<<3`. Each matrix has333–666 distinct scales. The GPU carrier below deliberately retains ordinary FP16 scale loads.

## The useful local map

For each contiguous four-output down tile and original K128, normalize each column's four signs by its first sign. There are8 possible relative patterns (3-bit class) and one orientation bit. **All8 classes occur in every one of the522,240 examined tile/K128 pairs.**

At runtime, orient each input and accumulate it into its class sum. Produce four signed reductions of those8 sums, then apply each output's **own original FP16 weight scale**, the original activation K128 scale, and the ordinary split-K continuation. Four-output grouping never requires equal weight scales among the four rows.

The scalar signed-addition model is120 class-aggregation additions plus28 output additions =148, versus4×127=508 direct additions. Scale operations remain unchanged. Key lookup, aggregation scheduling, loads, initialization and continuation determine actual GPU price; the arithmetic ratio29.13% is not a speedup measurement. At8 outputs the mean class count is≈81/128 and the addition ratio≈67.6%; at16 outputs it is≈127.75 and the ratio≈99.82%.

The report contains local collisions and exact rejection witnesses outside the observer tile. Example, layer0 original K128#0: input columns0 and127 are antipodal on outputs0–7. They already cease being antipodal at output8, whose original weight scale bits are`1ff0`, with both sign bits1. The local merge cannot be promoted to a full-output merge.

The global≈50% GU rate hides neuron-specific sign correlation. Whole-row disagreement p05/p95 is37.81/62.13% in layer0,44.34/55.51% in layer30 and40.45/59.16% in layer61. The report retains this without turning correlation into an exact merge.

GU has a separate generic two-class linear map per neuron/K128: `P=sum(gsign*x where gsign=usign)` and `Q=sum(gsign*x where gsign!=usign)`, then `G=dG*(P+Q)`, `U=dU*(P-Q)`. Both classes are nonempty in every examined matched block. Its scalar addition model is128 versus254 direct additions. It preserves the independent original scales, not floating-point accumulation order; no SiLU/product continuation closure is asserted.

## GPU price experiment

`down_classes.cu` implements original layer0 down4-output grouping with exact A8 integer class sums. It uses native `match_any`, subgroup `reduce_add`, warp-private shared integer atomic accumulation, and four final integer reductions. A warp owns a four-output tile. One CUDA block has four warps. Packed keys have two class/orientation nibbles per byte in `[tile4,K128,64bytes]` layout.

Recurring memory and continuation:

- Key table11,141,120bytes, exactly the original sign-bit payload size; no expanded sign words.
- Original FP16 weight scale table1,392,640bytes, `[tile4,K128,4]`.
- Existing `bitserial::produce(Dp4a01)` creates the same A8 bytes and scale/DC/status arrays used by the baseline. Q bytes areR×17408; scale/DC/status bytes areR×136×12. Only classes consumeQ+scale+status, but complete producer cost remains paid.
- Each split-K partial table isR×5120×ceil(136/chunk)×4bytes. Its final FP32 continuation is included. Original scale multiplications are included.
- R32 baseline is two `project_carrier` calls atR16, sharing one producer. This is the baseline API's current limit, not a newR32 native implementation.
- Both projection-only and producer-inclusive paths are measured atR1/16/32, split chunks1/4/16/136, warm and after128MiB L2 eviction. Reference is raw original Q1 weights and the same A8 producer, integer K128 dots and original FP16/activation scale continuation. Synthetic uniform activations are **cost-only**, not quality evidence.

Both measured constructions are slower than the existing DP4A map. Best producer-inclusive cold times in microseconds, from `down_classes_r{1,16,32}.jsonl`:

| R | shared-bin class4 | best direct nibble DP4A | existing bitserial DP4A |
|---|---:|---:|---:|
|1|92.16|28.67 (T1)|18.18|
|16|1222.66|159.74 (T4)|38.66|
|32|2425.86|307.20 (T4)|65.28|

Projection-only also loses. The exact low-class-count structure is real; this runtime grouping intermediate is expensive. Split-K outputs agree with the same A8/original-scale reference to relative RMS≈2.4e-7, with maximum absolute error<4.8e-6. These are rounding differences from the split reduction, not a change to the integer K128 dots.

The direct nibble alternative is1.58×/4.13×/4.71× slower atR1/16/32. This rejects these two native schedules as replacements, not every possible local-class realization.

The source includes an alternative **without constructing class sums**: `nibble4_dp4a_t1`/`t4` reconstruct four signed-byte coefficient words from the packed class/orientation nibbles, computes four native `DP4A` maps per lane, and uses native warp reductions. T4 reuses the loaded prepared weight codes across four tokens. This still uses the same packed-key endpoint and original scales, but no runtime `match_any`, shared bins or atomic operation. For chunk136 the code requires every FP32 output to equal the raw original-Q1 A8 reference exactly, with the same order of original scale continuation; each integer dot is bounded by16256. All routes pass that exact same-order check on5120/81,920/163,840 outputs atR1/16/32, respectively, both projection-only and producer-inclusive. A separate exhaustive host check reconstructs all89,128,960 layer0 down coefficients from the packed keys with zero mismatches. `down_classes_build.txt` pins the CUDA13.1 build, RTX5090 identity, source/executable/imported-object hashes and SASS showing actual `MATCH.ANY`, `REDUX.SUM.S32`, `ATOMS.ADD` and `IDP.4A.S8.S8` instructions.

Compile on the existing GPU source/object mirror:

```sh
cd /workspace/binary/digits-ragged/halo
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  -Iexperiments/binary-bitserial -Isrc \
  /workspace/binary/down-classes/down_classes.cu \
  build/bitserial.o build/gguf.o \
  -o /workspace/binary/down-classes/down_classes
```

Run eachR under a55-second command ceiling and hold `/tmp/binary-gpu.lock` for GPU ownership. The source and receipts remain in this experiment directory; the parent owns whole-engine activation capture and quality evaluation.

## Census reproduction

```sh
for layer in 0 30 61; do
  OPENBLAS_NUM_THREADS=1 python3 experiments/binary-whole-map/weight_census.py \
    --model ~/data/bonsai1/Bonsai-27B-Q1_0.gguf \
    --layer "$layer" --report experiments/binary-whole-map/weight_census.json
done
```

Each stage atomically replaces its entry in the report and rejects a different file identity. Neither stage invokes a GPU nor creates synthetic matrices. All observed scales must be finite and positive; any other state is an explicit input error.
