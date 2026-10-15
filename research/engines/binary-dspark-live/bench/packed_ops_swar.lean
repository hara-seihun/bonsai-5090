import Std.Tactic.BVDecide

namespace PackedOps

def pack15 (lo hi : BitVec 15) : BitVec 32 :=
  lo.zeroExtend 32 ||| (hi.zeroExtend 32 <<< 16)

theorem guarded_add (lo hi : BitVec 15) :
    ((pack15 lo hi + 0x00010001) &&& 0x7fff7fff) =
      pack15 (lo + 1) (hi + 1) := by
  unfold pack15
  bv_decide

theorem guarded_sub (lo hi : BitVec 15) :
    (((pack15 lo hi ||| 0x80008000) - 0x00010001) &&& 0x7fff7fff) =
      pack15 (lo - 1) (hi - 1) := by
  unfold pack15
  bv_decide

-- The RMS-fold claim is conditional on homogeneity, not a claim about IEEE rounding.
theorem fold_homogeneous_scale {V S C : Type} (h : V → V) (scale : S → V → V)
    (quant : V → C × S) (mul : S → S → S) (r : S) (x : V)
    (h_commutes : h (scale r x) = scale r (h x))
    (quant_homogeneous : ∀ u, quant (scale r u) = ((quant u).1, mul r (quant u).2)) :
    quant (h (scale r x)) = ((quant (h x)).1, mul r (quant (h x)).2) := by
  rw [h_commutes, quant_homogeneous]

end PackedOps
