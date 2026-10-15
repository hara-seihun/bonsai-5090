import Std
namespace SignedDigits

def q (i : Fin 127) : Int := Int.ofNat i.val - 63
def s (x : Int) : Int := if x < 0 then -1 else 1
def digit (x : Int) (p : Nat) : Int := s x * Int.ofNat ((x.natAbs / 4^p) % 4)

theorem exact_on_hidden7 : ∀ i : Fin 127,
    q i = digit (q i) 0 + 4 * digit (q i) 1 + 16 * digit (q i) 2 := by
  decide

theorem e2m1_integer_digits : ∀ i : Fin 127, ∀ p : Fin 3,
    -3 ≤ digit (q i) p.val ∧ digit (q i) p.val ≤ 3 := by
  decide

end SignedDigits
