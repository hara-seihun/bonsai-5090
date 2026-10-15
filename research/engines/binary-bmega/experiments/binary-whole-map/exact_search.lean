import Std

namespace BinaryWholeMap

def FactorsThrough {X C O : Type} (observe : X → O) (carrier : X → C) : Prop :=
  ∃ finish : C → O, ∀ x, finish (carrier x) = observe x

theorem collision_rejects_decoder {X C O : Type}
    (observe : X → O) (carrier : X → C) (x y : X)
    (same : carrier x = carrier y) (different : observe x ≠ observe y) :
    ¬ FactorsThrough observe carrier := by
  rintro ⟨finish, correct⟩
  apply different
  rw [← correct x, ← correct y, same]

def packed (s t : Int) : Int := s + 8 * t + 20

theorem packed_recovers (s t : Int) (low : -4 ≤ s) (high : s ≤ 2) :
    packed s t % 8 - 4 = s ∧ packed s t / 8 - 2 = t := by
  unfold packed
  omega

theorem packed_fibres (s t s' t' : Int)
    (lo : -4 ≤ s) (hi : s ≤ 2) (lo' : -4 ≤ s') (hi' : s' ≤ 2) :
    packed s t = packed s' t' ↔ s = s' ∧ t = t' := by
  constructor
  · intro same
    have a := packed_recovers s t lo hi
    have b := packed_recovers s' t' lo' hi'
    rw [same] at a
    omega
  · rintro ⟨rfl,rfl⟩
    rfl

/-- The observer may contain ordered IEEE scale folds, SiLU, the whole-block
    quantizer and the down projection. No closure of a source stage is assumed. -/
theorem compound_observer {O : Type} (observe : Int → Int → O)
    (x y t : Int) (xlo : -2 ≤ x) (xhi : x ≤ 1)
    (ylo : -2 ≤ y) (yhi : y ≤ 1) :
    observe (packed (x+y) t % 8 - 4) (packed (x+y) t / 8 - 2) =
    observe (x+y) t := by
  have h := packed_recovers (x+y) t (by omega) (by omega)
  rw [h.1,h.2]

def radix4 (x y z : Int) : Int := x + 4*y + 16*z + 42

theorem radix4_recovers (x y z : Int)
    (xl : -2 ≤ x) (xh : x ≤ 1) (yl : -2 ≤ y) (yh : y ≤ 1) :
    radix4 x y z % 4 - 2 = x ∧
    radix4 x y z / 4 % 4 - 2 = y ∧
    radix4 x y z / 16 - 2 = z := by
  unfold radix4
  omega

/-- Equal integer group dots preserve any same-schedule native continuation.
    This does not assert that native FP4 is exact; that lowering is separate. -/
theorem native_continuation_transport {Dot Scale State Out : Type}
    (fold : Dot → Scale → State) (observe : State → Out)
    (a b : Dot) (scale : Scale) (same : a = b) :
    observe (fold a scale) = observe (fold b scale) := by
  rw [same]

def firstCollisionLeft : Int × Int × Int := (-2,-1,-2)
def firstCollisionRight : Int × Int × Int := (-2,-2,-1)
def sumCarrier (x : Int × Int × Int) : Int := x.1 + x.2.1 + x.2.2

theorem retained_sum_witness :
    sumCarrier firstCollisionLeft = sumCarrier firstCollisionRight ∧
    (3192693556 : Nat) ≠ 3189766944 := by
  decide

/-- Exact endpoint bit words are supplied by the host truth-vector evaluator.
    The proof separates the logical obstruction from that evaluator's semantics. -/
theorem retained_observer_rejection (observe : (Int × Int × Int) → Nat)
    (left : observe firstCollisionLeft = 3192693556)
    (right : observe firstCollisionRight = 3189766944) :
    ¬ FactorsThrough observe sumCarrier := by
  apply collision_rejects_decoder observe sumCarrier firstCollisionLeft firstCollisionRight
  · exact retained_sum_witness.1
  · rw [left,right]
    exact retained_sum_witness.2

/-- A nonzero mixed rectangle prevents summing independent local tables. -/
theorem additive_partition_rejected (observe : Int → Int → Int)
    (different : observe 1 1 + observe 0 0 ≠ observe 1 0 + observe 0 1) :
    ¬ ∃ a b : Int → Int, ∀ x y, observe x y = a x + b y := by
  rintro ⟨a,b,h⟩
  apply different
  rw [h 1 1,h 0 0,h 1 0,h 0 1]
  omega

theorem retained_exact_rectangle :
    (-42106256 : Int) + 0 ≠ (-11518821) + (-11518821) := by
  decide

#print axioms retained_exact_rectangle
#print axioms collision_rejects_decoder
#print axioms packed_recovers
#print axioms packed_fibres
#print axioms compound_observer
#print axioms radix4_recovers
#print axioms native_continuation_transport
#print axioms retained_observer_rejection
#print axioms additive_partition_rejected
end BinaryWholeMap
