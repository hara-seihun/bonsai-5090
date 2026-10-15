import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Lean.Elab.Tactic.Omega

namespace BinaryBitserial

open scoped BigOperators

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

def bit (b : Bool) : ℤ := if b then 1 else 0

def weight (w : Bool) : ℤ := 2 * bit w - 1

def coefficient (p : Fin 8) : ℤ := if p.val = 7 then -128 else (2 : ℤ) ^ p.val

def decode (a : Fin 8 → Bool) : ℤ := ∑ p, coefficient p * bit (a p)

def planes (a : Fin 256) (p : Fin 8) : Bool := a.val.testBit p.val

def signedByte (a : Fin 256) : ℤ := (a.val : ℤ) - if a.val < 128 then 0 else 256

def scalarPlanes (w : Bool) (a : Fin 8 → Bool) : ℤ :=
  ∑ p, coefficient p * (2 * bit (w && a p) - bit (a p))

theorem bit_and (w a : Bool) : bit (w && a) = bit w * bit a := by
  cases w <;> cases a <;> decide

theorem scalar_identity (w : Bool) (a : Fin 8 → Bool) :
    weight w * decode a = scalarPlanes w a := by
  unfold weight decode scalarPlanes
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [bit_and]
  ring

-- The finite bridge connects the coefficient representation to conventional signed int8.
theorem decode_signed_byte : ∀ a : Fin 256, decode (planes a) = signedByte a := by
  decide +kernel

def popcount {n : ℕ} (b : Fin n → Bool) : ℤ :=
  ((Finset.univ.filter fun i => b i).card : ℤ)

theorem popcount_eq_sum {n : ℕ} (b : Fin n → Bool) :
    popcount b = ∑ i, bit (b i) := by
  simp [popcount, bit]

def dot {n : ℕ} (w : Fin n → Bool) (a : Fin n → Fin 8 → Bool) : ℤ :=
  ∑ i, weight (w i) * decode (a i)

def packedDot {n : ℕ} (w : Fin n → Bool) (a : Fin n → Fin 8 → Bool) : ℤ :=
  2 * (∑ p, coefficient p * popcount (fun i => w i && a i p)) - ∑ i, decode (a i)

theorem dot_identity {n : ℕ} (w : Fin n → Bool) (a : Fin n → Fin 8 → Bool) :
    dot w a = packedDot w a := by
  unfold dot packedDot
  simp_rw [scalar_identity, scalarPlanes, mul_sub, Finset.sum_sub_distrib]
  congr 1
  rw [Finset.sum_comm, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [popcount_eq_sum, Finset.mul_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem signed_dot_identity {n : ℕ} (w : Fin n → Bool) (a : Fin n → Fin 256) :
    (∑ i, weight (w i) * signedByte (a i)) =
      2 * (∑ p, coefficient p * popcount (fun i => w i && planes (a i) p)) -
      ∑ i, signedByte (a i) := by
  simpa only [dot, packedDot, decode_signed_byte] using
    dot_identity w (fun i => planes (a i))

theorem k128_identity (w : Fin 128 → Bool) (a : Fin 128 → Fin 256) :
    (∑ i, weight (w i) * signedByte (a i)) =
      2 * (∑ p, coefficient p * popcount (fun i => w i && planes (a i) p)) -
      ∑ i, signedByte (a i) := signed_dot_identity w a

theorem signed_byte_bounds (a : Fin 256) :
    -128 ≤ signedByte a ∧ signedByte a ≤ 127 := by
  unfold signedByte
  split_ifs <;> have := a.isLt <;> omega

theorem signed_product_bounds (w : Bool) (a : Fin 256) :
    -128 ≤ weight w * signedByte a ∧ weight w * signedByte a ≤ 128 := by
  have h := signed_byte_bounds a
  cases w <;> simp [weight, bit] <;> omega

theorem k128_bounds (w : Fin 128 → Bool) (a : Fin 128 → Fin 256) :
    -16384 ≤ (∑ i, weight (w i) * signedByte (a i)) ∧
    (∑ i, weight (w i) * signedByte (a i)) ≤ 16384 := by
  constructor
  · calc
      -16384 = ∑ _i : Fin 128, (-128 : ℤ) := by norm_num
      _ ≤ ∑ i, weight (w i) * signedByte (a i) :=
        Finset.sum_le_sum fun i _ => (signed_product_bounds (w i) (a i)).1
  · calc
      (∑ i, weight (w i) * signedByte (a i)) ≤ ∑ _i : Fin 128, (128 : ℤ) :=
        Finset.sum_le_sum fun i _ => (signed_product_bounds (w i) (a i)).2
      _ = 16384 := by norm_num

def expandNibble (n : Fin 16) : ℕ := (n.val * 0x00204081) &&& 0x01010101

def nibbleLane (n : Fin 16) (i : Fin 4) : ℤ :=
  ((expandNibble n / 2 ^ (8 * i.val)) % 256 : ℕ)

theorem nibble_expansion : ∀ n : Fin 16,
    expandNibble n = ∑ i : Fin 4, (if n.val.testBit i.val then 1 else 0) * 2 ^ (8 * i.val) := by
  decide +kernel

theorem nibble_lane_bridge : ∀ (n : Fin 16) (i : Fin 4),
    nibbleLane n i = bit (n.val.testBit i.val) := by
  decide +kernel

theorem affine_identity (w : Bool) (q : ℤ) : weight w * q = 2 * bit w * q - q := by
  unfold weight
  ring

theorem nibble_dp4a_identity (n : Fin 16) (q : Fin 4 → ℤ) :
    (∑ i, weight (n.val.testBit i.val) * q i) =
      2 * (∑ i, nibbleLane n i * q i) - ∑ i, q i := by
  simp_rw [nibble_lane_bridge, affine_identity]
  rw [Finset.sum_sub_distrib, Finset.mul_sum]
  simp only [mul_assoc]

#print axioms nibble_expansion
#print axioms nibble_lane_bridge
#print axioms nibble_dp4a_identity
#print axioms decode_signed_byte
#print axioms scalar_identity
#print axioms k128_identity
#print axioms k128_bounds

end BinaryBitserial
