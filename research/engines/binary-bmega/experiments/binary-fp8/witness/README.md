# Binary weights on an E4M3 carrier

`BinaryFP8.lean` proves the representation used by the native FP8 route:

- `expand(n) = 0xb8b8b8b8 XOR ((((n * 0x00204081) & 0x01010101) << 7))`, in 32-bit arithmetic.
- For every nibble `n : Fin 16` and lane `i : Fin 4`, low-to-high byte lane `i` is `0x38` when bit `i` is 1, otherwise `0xb8`.
- A genuine E4M3FN decoder uses bias 7, three fraction bits, subnormals and the two NaN bytes. Both carrier codes have exponent 7 and fraction 0 and decode to exactly +1/-1.
- For arbitrary finite index sets, real coefficients, decoded activation values and scale, projection with decoded weight carriers equals projection with the original binary signs. The coefficients may include weight scales. Each weight carrier carries a decoder equation, rather than an invented value for NaN.
- Quantization error is exactly `sum w * (qDecoded * sA - x)` in real arithmetic.

`finite.cpp` exhausts all 16 nibbles / 64 lane cases, with compile-time byte and rational-decoder checks and runtime E4M3 float-decoder checks. Its exact decoder represents all finite values as integer multiples of 1/512; NaN returns an unset optional.

## Reproduce

```sh
bash experiments/binary-fp8/witness/check.sh /opt/lean-mathlib
```

The supplied Lean project must already provide `Mathlib`; this host has Lean 4.33 in `/opt/lean-mathlib`. The script uses bounded foreground commands, builds C++ into a temporary directory, removes the binary on exit and replaces `check-receipt.txt` with tool versions, source hashes and check output. Lean prints theorem axioms: standard Mathlib axioms are allowed; no admission, `sorryAx` or native-decision axiom is used.

## Boundary

This witnesses exact relabeling of the original 1-bit **weights**. The projection theorem is relative to the decoded FP8 **activation carrier**, not the original activation. Quantization quality is measured separately. It makes no exact-A8 claim, floating-point accumulation claim, CUDA indexing/layout claim or performance claim.
