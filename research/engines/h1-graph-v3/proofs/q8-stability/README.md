# Q8 GDN: exact runaway and least-squares repair

Owner: Halo 5090. [GDN_Q8_STABILITY.md](../../tools/GDN_Q8_STABILITY.md) connects this exact-arithmetic contract to the CUDA selector and real-token observations. This proof uses no GPU.

**All 24 audited theorems compile in Lean 4.33.0.** `check-receipt.txt` records source hashes, Mathlib commit and axiom dependencies. Every audited theorem uses only `propext`, `Classical.choice`, and `Quot.sound`; none uses `sorryAx` or a custom axiom. There are no unproved statements substituted for these results.

## Exact counterexample — `Runaway.lean`

The file defines the actual two-coordinate row-amax quantizer, with scale `max(|x₁|,|x₂|)/127`, zero handling, and nearest-integer ties-to-even codes. Its rounding definition takes `floor(x+1/2)` and subtracts one exactly at odd upper ties. The relevant rounding facts `rne(127)=127` and `rne(127/2)=64` are checked.

For `a>0`, let `x(a)=a(1,64/127)` and

```
P = [[4/5, 2/5], [2/5, 1/5]],  r = 636/635 > 1.
```

Checked results:

- `P_idempotent`, `P_self_adjoint`, `P_energy_identity`, `P_energy_le`: P is an orthogonal projection for the coordinate Euclidean inner product; its removed squared energy is `(x₁−2x₂)²/5`.
- `P_ray`: `P x(a) = ar(1,1/2)`.
- `maxQuant_half_ray`: the quantizer sends `(b,b/2)` to `(b,64b/127)` for every `b>0`.
- `exact_expanding_step`: `Q(P x(a)) = x(ra)`.
- `exact_expanding_orbit`: `(Q∘P)^n x(a) = x(r^n a)` for every natural n.
- `quantized_projection_energy_increases`: squared Euclidean norm strictly increases each combined step, although P alone contracts it.

The paired coordinate type is not used with Lean's product norm (which is a max norm); `energy` explicitly means `x₁²+x₂²`. This avoids silently proving the counterexample in the wrong norm.

## Repair — `Stability.lean`

For any real inner-product space and any chosen code vector q,

```
LS(x,q) = (<x,q>/<q,q>) q
```

is checked to satisfy:

- `ls_zero`: zero codes reconstruct zero.
- `ls_residual_orthogonal`: `<x−LS(x,q),q>=0`.
- `ls_pythagoras`: `‖x−LS(x,q)‖² + ‖LS(x,q)‖² = ‖x‖²`.
- `ls_norm_le`, `adaptive_ls_norm_le`: `‖LS(x,q)‖≤‖x‖` even when q is an arbitrary function of x. No code-quality or rounding hypothesis is needed.

This is **origin nonexpansion**, not a pairwise Lipschitz assertion for an adaptive quantizer. In exact real arithmetic, division by zero is defined as zero; `ls_zero` proves the desired zero-code branch. A CUDA implementation must explicitly implement that branch.

For GDN `T(x)=g[x−β<k,x>k]`, `rankOne_energy` checks the identity

```
‖x−β<k,x>k‖² = ‖x‖² − β(2−β‖k‖²)<k,x>².
```

`gdn_norm_le` proves contraction under exactly `0≤g≤1` and `0≤β‖k‖²≤2`, including k=0. No separate β≥0 assumption is imposed.

`quantized_homogeneous_bound` proves `‖LS(Tx,q(Tx))‖≤‖x‖` for every contractive T. `quantized_step_bound` proves

```
‖LS(Tx+f,q(Tx+f))‖ ≤ ‖x‖ + ‖f‖.
```

`linear_forcing_bound` then proves `‖state n‖≤‖state 0‖+nB` whenever every step obeys that inequality and `‖forcing n‖≤B`. Thus LS rescaling removes this homogeneous quantized exponential mechanism; bounded additive forcing can give at most linear norm growth under these assumptions.

## Reproduce

```bash
cd proofs/q8-stability
./check.sh /opt/lean-mathlib/.lake/packages
```

The argument is the installed Mathlib Lake packages directory; the script imports its existing caches, writes local `.build/*.olean`, checks both source files, prints audited theorem axioms, and replaces `check-receipt.txt`. It does not modify the cache or fetch/build dependencies. To reproduce elsewhere use Lean 4.33.0 and cached Mathlib commit `db584cd6d46c92f209a44c0f1c829460d327499d` with its package dependencies.

These theorems are about exact real arithmetic. They do not bound CUDA floating-point reductions, division, reconstruction or update roundoff; they do not establish real-model quality or claim the actual runaway trace is fixed. Those are separate implementation and trace results owned by the kernel/CPU workers.
