# Exact binary × signed-A8 projection witness

`w=1` means weight `+1`; `w=0` means `−1`. Activation `q` is a signed byte,
with two's-complement bits `a₀,…,a₇` and coefficients
`c=(1,2,4,8,16,32,64,−128)`.

The scalar identity is

```
q = Σₚ cₚ aₚ
(2w−1)q = Σₚ cₚ (2(w AND aₚ)−aₚ).
```

For arbitrary coordinates, distribute sums and exchange their order:

```
Σᵢ (2wᵢ−1)qᵢ = 2 Σₚ cₚ popcount(wBits AND activationPlane[p]) − Σᵢ qᵢ.
```

The K128 witness stores original weight signs in four `uint32_t` words. Logical
coordinate `i=32j+b` uses bit `b` of word `j`, in both weight and activation planes.
If a physical weight format uses another coordinate permutation, transpose the
activation planes into that same order. The packed projection reads weight words
only through AND and population count; it never materializes dense signed weights.
Activation transposition and its DC correction `Σq` can be shared across output rows.

## Nibble / signed-DP4A composition

For each four-bit nibble `n`, the exact relabeling is

```
expand(n) = (n * 0x00204081u) & 0x01010101u
byteLane(expand(n), i) = bit(n,i),  i=0,…,3.
```

All four operand bytes are 0 or 1, so signed and unsigned interpretations of the
expanded weight operand agree. A signed DP4A with signed activation bytes computes
`selected = Σᵢ bit(n,i) qᵢ`. Apply the same affine correction:
`dot = 2*selected − Σq`. This expands one nibble directly into a register operand;
it does not allocate a dense weight array. The C++ witness emulates the lane sum,
not device instruction execution or compiler lowering.

## Arithmetic bounds and scale continuation

For K128, the final integer dot lies in `[-16384,16384]`; both endpoints are
attained with all activations `−128` and uniform weight signs. The masked signed
sum is in `[-16384,16256]`; doubling it stays in `[-32768,32512]`.
The activation sum is in `[-16384,16256]`. Even loose independent bounds on the
final correction fit signed int32, as do each population-count term and running
plane accumulation. Do the arithmetic in int32, not int8.

The proved map ends at the exact integer dot of the **given quantized activations**.
For a block with real scale factors `sW,sA`, its mathematical continuation is
`(sW*sA)*integerDot`; blocks may have different factors. The witness makes no claim
of FP32 bitwise identity after scale multiplication, split-K reduction, or a changed
floating evaluation order. Activation quantization error and full-model quality
are separate parent-owned questions.

## Runnable checks

From the repository root:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  experiments/binary-bitserial/witness/witness.cpp -o /tmp/binary-bitserial-witness
/tmp/binary-bitserial-witness

cd /opt/lean-mathlib
lake env lean ~/work/clones/binary-spec-5090/experiments/binary-bitserial/witness/Bitserial.lean
```

Or run both, with compiler timeouts and temporary-file cleanup:

```sh
experiments/binary-bitserial/witness/check.sh /opt/lean-mathlib
```

The existing Lean owner is documented in
`the Lean environment notes`: Lean4.33.0, Mathlib revision
`db584cd6d46c92f209a44c0f1c829460d327499d`. No dependency installation or library
modification is needed. `check-receipt.txt` records this host's successful run.

### C++ scope

- Exhausts all 512 weight-bit × signed-byte scalar inputs, including the byte bridge.
- Exhausts all 16 nibble expansions.
- Exhausts all 16×4×256 independent signed DP4A lane inputs (16,384 cases).
- Checks 69,632 full K128 vectors: all byte values in uniform and offset-ramp
  activations for eight weight patterns; every byte in every isolated coordinate
  for one-bit weight words and their complements.
- Checks the sharp ±16384 bounds and ±16256 positive-byte endpoints explicitly.
- Compares both packed maps against an independent dense integer reference.

The full vectors exercise word boundaries and the negative sign plane. They are
representative cases, not an exhaustive census of K128 inputs.

### Lean scope

`Bitserial.lean` proves:

- `decode_signed_byte`: kernel-checked finite bridge for all256 byte encodings.
- `scalar_identity`: algebraic scalar identity for every Boolean weight and every
  eight-bit function, with the signed plane coefficient explicit.
- `dot_identity` / `signed_dot_identity`: arbitrary finite-width summation and
  population-count law, where popcount is the cardinality of true coordinates.
- `k128_identity`: the requested exact signed-A8 packed dot statement for128 inputs.
- `k128_bounds`: the integer output bound for all weight/activation inputs.
- `nibble_expansion` / `nibble_lane_bridge`: all16 expansions and64 extracted lanes.
- `nibble_dp4a_identity`: affine correction of the lane sum for arbitrary integer
  activations, hence signed bytes as a special case.

Finite bridges use `decide +kernel`, not `native_decide`. Printed dependencies are
only ordinary Lean axioms (`propext`, `Classical.choice`, `Quot.sound`), with no
`sorryAx` or native-computation axiom. This is a proof of the mathematical packed
map; CUDA address selection and the native ISA implementation remain separate.
