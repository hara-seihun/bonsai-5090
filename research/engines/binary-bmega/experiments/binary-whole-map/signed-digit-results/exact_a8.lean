import Std
namespace ExactA8

def q (i : Fin 256) : Int := Int.ofNat i.val - 128
def a (i : Fin 256) : Nat := i.val % 4
def b (i : Fin 256) : Nat := (i.val / 4) % 4
def c (i : Fin 256) : Int := Int.ofNat (i.val / 16) - 8
def adjust (i : Fin 256) : Bool := (c i).natAbs == 5 || (c i).natAbs == 7
def delta (i : Fin 256) : Int := if b i % 2 == 1 then 1 else -1
def c' (i : Fin 256) : Int := if adjust i then c i + delta i else c i
def bcode (i : Fin 256) : Nat := if adjust i then
  match b i with | 0 => 4 | 1 => 11 | 2 => 5 | _ => 9
  else b i
def ccode (i : Fin 256) : Nat :=
  (if (c' i).natAbs ≤ 4 then (c' i).natAbs else (c' i).natAbs / 2 + 2) +
  (if c' i < 0 then 8 else 0)
def twiceE2M1 (code : Nat) : Int :=
  (if code ≥ 8 then -1 else 1) * Int.ofNat
    (match code % 8 with | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 4 | 5 => 6 | 6 => 8 | _ => 12)

set_option maxRecDepth 4096 in
set_option maxHeartbeats 1000000 in
theorem exact_all_byte_patterns : ∀ i : Fin 256,
    q i = twiceE2M1 (a i) + 4 * twiceE2M1 (bcode i) + 16 * twiceE2M1 (ccode i) := by
  decide

end ExactA8
