import Lean.Elab.Tactic.Omega
namespace GuScaleAbsorption
-- E2M1 magnitudes multiplied by 2, including zero.
def q2 (q : Fin 8) : Nat := #[0,1,2,3,4,6,8,12][q.val]!
-- Finite positive UE4M3, multiplied by 512. 0x7f is excluded (NaN).
def s512 (s : Fin 127) : Nat :=
  if s.val / 8 = 0 then s.val % 8
  else (8 + s.val % 8) * 2 ^ (s.val / 8 - 1)
def numerator1024 (q : Fin 8) (s : Fin 127) : Nat := q2 q * s512 s
-- Every one-plane coefficient is numerator1024/1024.
-- Actual model gate row zero's CPU-reconstructed mean is 13108019/2^30.
-- Their equality would require the following impossible integer equation.
theorem actual_row_zero_not_one_plane (n : Nat) :
    n * 1048576 ≠ 13108019 := by omega

def stripTwos : Nat → Nat → Nat
  | 0, n => n
  | f+1, n => if n != 0 && n % 2 == 0 then stripTwos f (n/2) else n
-- This is a finite format fact. It does not assert CUDA arithmetic semantics.
set_option maxRecDepth 8192 in
set_option maxHeartbeats 0 in
theorem finite_odd_capacity :
    ∀ (q : Fin 8) (s : Fin 127), stripTwos 32 (numerator1024 q s) ≤ 45 := by decide
-- The actual mean's odd significand exceeds the capacity of a single product.
theorem actual_odd_large : 45 < 13108019 ∧ 13108019 % 2 = 1 := by decide
-- Local round-to-nearest-even with input measured in half-ulp units.
-- This models the two normal, positive exponent bins used in the CPU witness;
-- correspondence to a GPU's complete contraction remains a separate obligation.
def nearestEven (halfUnits : Nat) : Nat :=
  let whole := halfUnits / 2
  if halfUnits % 2 == 0 || whole % 2 == 0 then whole else whole+1
-- At 2^23, ulp=1; 2^23+0.5 ties toward its even lower neighbor.
theorem contraction_tie : nearestEven 16777217 = 8388608 := by decide
-- Scaling the rounded contraction by 1.5 gives 12582912; absorbing scale first
-- adds 0.75, which rounds up in the ulp=1 bin.
theorem scaled_bin_half_tie : nearestEven 25165825 = 12582912 := by decide
-- 12582912.75 is represented here after multiplying by 4 rather than 2.
def nearestEvenQuarter (quarterUnits : Nat) : Nat :=
  let whole := quarterUnits / 4
  let rem := quarterUnits % 4
  if rem < 2 || (rem == 2 && whole % 2 == 0) then whole else whole+1
theorem explicit_rounding_difference :
    nearestEvenQuarter 50331651 = 12582913 ∧ 12582912 ≠ 12582913 := by decide
end GuScaleAbsorption
