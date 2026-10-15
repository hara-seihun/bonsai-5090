import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Lean.Elab.Tactic.Omega

namespace BinaryNative4

open scoped BigOperators

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-- Offset-binary code: code 0 is -128, code 255 is 127. -/
def signedByte (u : Fin 256) : ℤ := (u.val : ℤ) - 128

def low (u : Fin 256) : ℤ := (u.val % 16 : ℕ)
def high (u : Fin 256) : ℤ := (u.val / 16 : ℕ)

theorem byte_bridge : ∀ u : Fin 256,
    signedByte u = low u + 16 * high u - 128 ∧
    0 ≤ low u ∧ low u ≤ 15 ∧ 0 ≤ high u ∧ high u ≤ 15 := by
  decide +kernel

/-- The same decomposition for an arbitrary integer signed byte q. -/
theorem signed_q_bridge (q : ℤ) (hq : -128 ≤ q ∧ q ≤ 127) :
    q = (q + 128) % 16 + 16 * ((q + 128) / 16) - 128 ∧
    0 ≤ (q + 128) % 16 ∧ (q + 128) % 16 ≤ 15 ∧
    0 ≤ (q + 128) / 16 ∧ (q + 128) / 16 ≤ 15 := by
  omega

def weight (positive : Bool) : ℤ := if positive then 1 else -1

def s4Code (positive : Bool) : Fin 16 := if positive then ⟨1, by decide⟩ else ⟨15, by decide⟩
def signedNibble (n : Fin 16) : ℤ := (n.val : ℤ) - if n.val < 8 then 0 else 16

theorem signed_nibble_one : signedNibble ⟨1, by decide⟩ = 1 := by decide
theorem signed_nibble_fifteen : signedNibble ⟨15, by decide⟩ = -1 := by decide

theorem weight_encoding (positive : Bool) : signedNibble (s4Code positive) = weight positive := by
  cases positive <;> decide

def s4Bit (positive : Bool) (b : Fin 4) : Bool := if b.val = 0 then true else !positive

theorem weight_bit_encoding : ∀ positive : Bool,
    (s4Code positive).val = ∑ b : Fin 4, (if s4Bit positive b then 1 else 0) * 2 ^ b.val := by
  decide +kernel

/-- General finite algebra; no binary-weight assumption is needed. -/
theorem projection_algebra {ι : Type*} (s : Finset ι) (w l h : ι → ℤ) :
    (∑ i ∈ s, w i * (l i + 16 * h i - 128)) =
      (∑ i ∈ s, w i * l i) + 16 * (∑ i ∈ s, w i * h i) -
        128 * (∑ i ∈ s, w i) := by
  calc
    _ = ∑ i ∈ s, (w i * l i + 16 * (w i * h i) - 128 * w i) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by
      rw [Finset.sum_sub_distrib, Finset.sum_add_distrib,
        ← Finset.mul_sum, ← Finset.mul_sum]

theorem projection_identity {ι : Type*} (s : Finset ι) (w : ι → ℤ) (u : ι → Fin 256) :
    (∑ i ∈ s, w i * signedByte (u i)) =
      (∑ i ∈ s, w i * low (u i)) + 16 * (∑ i ∈ s, w i * high (u i)) -
        128 * (∑ i ∈ s, w i) := by
  simp_rw [(byte_bridge _).1]
  exact projection_algebra s w (fun i => low (u i)) (fun i => high (u i))

theorem signed_q_projection {ι : Type*} (s : Finset ι) (w q : ι → ℤ)
    (hq : ∀ i ∈ s, -128 ≤ q i ∧ q i ≤ 127) :
    (∑ i ∈ s, w i * q i) =
      (∑ i ∈ s, w i * ((q i + 128) % 16)) +
      16 * (∑ i ∈ s, w i * ((q i + 128) / 16)) - 128 * (∑ i ∈ s, w i) := by
  calc
    _ = ∑ i ∈ s, w i * ((q i + 128) % 16 + 16 * ((q i + 128) / 16) - 128) := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [← (signed_q_bridge (q i) (hq i hi)).1]
    _ = _ := projection_algebra s w _ _

theorem k128_identity (w : Fin 128 → Bool) (u : Fin 128 → Fin 256) :
    (∑ i, weight (w i) * signedByte (u i)) =
      (∑ i, signedNibble (s4Code (w i)) * low (u i)) +
      16 * (∑ i, signedNibble (s4Code (w i)) * high (u i)) -
      128 * (∑ i, weight (w i)) := by
  simp_rw [weight_encoding]
  exact projection_identity Finset.univ (fun i => weight (w i)) u

def countPositive (w : Fin 128 → Bool) : ℤ :=
  ((Finset.univ.filter fun i => w i).card : ℤ)

def halfDC (w : Fin 128 → Bool) : ℤ := countPositive w - 64

theorem k128_dc_identity (w : Fin 128 → Bool) :
    (∑ i, weight (w i)) = 2 * (countPositive w - 64) := by
  have hweight : ∀ b : Bool, weight b = 2 * (if b then (1 : ℤ) else 0) - 1 := by
    intro b
    cases b <;> decide
  have hcount : countPositive w = ∑ i, (if w i then (1 : ℤ) else 0) := by
    simp [countPositive]
  simp_rw [hweight]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← hcount]
  norm_num
  ring

theorem half_dc_bounds (w : Fin 128 → Bool) :
    -64 ≤ halfDC w ∧ halfDC w ≤ 64 := by
  have hc := Finset.card_le_card (Finset.filter_subset (fun i => w i) (Finset.univ : Finset (Fin 128)))
  simp only [Finset.card_univ, Fintype.card_fin] at hc
  unfold halfDC countPositive
  omega

theorem half_dc_fits_int8 (w : Fin 128 → Bool) :
    -128 ≤ halfDC w ∧ halfDC w ≤ 127 := by
  have := half_dc_bounds w
  omega

theorem k128_half_dc_identity (w : Fin 128 → Bool) (u : Fin 128 → Fin 256) :
    (∑ i, weight (w i) * signedByte (u i)) =
      (∑ i, signedNibble (s4Code (w i)) * low (u i)) +
      16 * (∑ i, signedNibble (s4Code (w i)) * high (u i)) - 256 * halfDC w := by
  rw [k128_identity, k128_dc_identity]
  unfold halfDC
  ring

theorem signed_byte_bounds (u : Fin 256) :
    -128 ≤ signedByte u ∧ signedByte u ≤ 127 := by
  unfold signedByte
  have := u.isLt
  omega

theorem signed_product_bounds (w : Bool) (u : Fin 256) :
    -128 ≤ weight w * signedByte u ∧ weight w * signedByte u ≤ 128 := by
  have h := signed_byte_bounds u
  cases w <;> simp [weight] <;> omega

theorem k128_bounds (w : Fin 128 → Bool) (u : Fin 128 → Fin 256) :
    -16384 ≤ (∑ i, weight (w i) * signedByte (u i)) ∧
    (∑ i, weight (w i) * signedByte (u i)) ≤ 16384 := by
  constructor
  · calc
      -16384 = ∑ _i : Fin 128, (-128 : ℤ) := by norm_num
      _ ≤ ∑ i, weight (w i) * signedByte (u i) :=
        Finset.sum_le_sum fun i _ => (signed_product_bounds (w i) (u i)).1
  · calc
      (∑ i, weight (w i) * signedByte (u i)) ≤ ∑ _i : Fin 128, (128 : ℤ) :=
        Finset.sum_le_sum fun i _ => (signed_product_bounds (w i) (u i)).2
      _ = 16384 := by norm_num

#print axioms byte_bridge
#print axioms signed_q_bridge
#print axioms weight_encoding
#print axioms weight_bit_encoding
#print axioms projection_algebra
#print axioms projection_identity
#print axioms signed_q_projection
#print axioms k128_identity
#print axioms k128_bounds
#print axioms k128_dc_identity
#print axioms half_dc_bounds
#print axioms half_dc_fits_int8
#print axioms k128_half_dc_identity

end BinaryNative4
