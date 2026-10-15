# Native W4 × A8 representation witness

The original weight sign bit is 1 for +1, 0 for −1. A signed four-bit operand
encodes those weights exactly as `0001` (1) and `1111` (15); signed-nibble decode
maps 1 to +1 and 15 to −1. This is exact binary-weight expansion, not weight
quantization.

For every signed byte `q ∈ [−128,127]`, set

```
u = q + 128
low = u % 16       // unsigned nibble, 0…15
high = u / 16      // unsigned nibble, 0…15
q = low + 16*high − 128
```

The activation operands are **unsigned** four-bit digits; interpreting 8…15 as
signed changes the map. Weight operands are signed. For an arbitrary finite
projection (indeed any integer weights),

```
Σᵢ wᵢqᵢ = Σᵢ wᵢlowᵢ + 16 Σᵢ wᵢhighᵢ − 128 Σᵢ wᵢ.
```

For K128 binary weights, `Σw = 2(countPositive−64)`. Store
`H = countPositive−64 ∈ [−64,64]` in int8 and use

```
dot = dotLow + 16*dotHigh − 256*H.
```

The final integer dot is in `[−16384,16384]`, with both endpoints attained by
all `q=−128` and uniform weight signs. Each digit dot is in `[−1920,1920]`;
the loose independent bound for the correction is ±49024. Accumulate and apply
the correction in signed int32, not int8 or int16.

## Files and proof scope

- `Native4.lean`: kernel-checked exhaustive 256-code byte decomposition and digit
  bounds; arbitrary integer signed-byte bridge; s4 decoding and bit encoding;
  arbitrary finite projection algebra; K128 signed-weight projection;
  K128 half-DC identity and int8 bounds; final K128 integer bounds.
  Finite bridges use `decide +kernel`. Printed axiom dependencies contain no
  `sorryAx` or native-evaluation axiom.
- `witness.cpp`: exhausts all 512 sign × signed-byte scalar pairs and all
  256 eight-sign patterns (2048 packed s4 lanes); compares 69,632 K128 vectors
  with an independent dense integer reference. Those vectors exercise uniform
  bytes, offset ramps, isolated coordinates and packed-word boundaries. Every
  vector checks the int8 half-DC identity/bounds; explicit cases attain both
  final bounds. This is not an exhaustive K128 census.
- `checkreceipt`: actual local commands, timings, source hashes and successful
  C++/Lean outputs.

`Fin 256` in Lean indexes **offset-binary** values `u`, not raw two's-complement
bytes. `signed_q_bridge` and `signed_q_projection` state the same result directly
for integer `q ∈ [−128,127]`.

The C++ packing assigns coordinate i to nibble i%8 of word i/8. It is a host-side
representation witness, not a statement about CUDA tensor operand addresses,
ISA execution or compiler lowering. The mathematical endpoint is the integer
dot of the given A8 values, before scale multiplication and floating reduction.

## Reproduce

From the repository root:

```sh
timeout 50s c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  experiments/binary-native4/witness/witness.cpp -o /tmp/binary-native4-witness
timeout 50s /tmp/binary-native4-witness
rm /tmp/binary-native4-witness
cd /opt/lean-mathlib
timeout 50s lake env lean \
  ~/work/clones/binary-spec-5090/experiments/binary-native4/witness/Native4.lean
```

The existing `/opt/lean-mathlib` environment is Lean 4.33.0 with Mathlib revision
`db584cd6d46c92f209a44c0f1c829460d327499d`, owned by
`the Lean environment notes`. No dependency install is needed.
