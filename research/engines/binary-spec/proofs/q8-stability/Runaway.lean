import Mathlib.Algebra.Order.Round
import Mathlib.Tactic

noncomputable section
namespace Q8Runaway

abbrev V := ℝ × ℝ

def energy (x : V) : ℝ := x.1 ^ 2 + x.2 ^ 2
def dot (x y : V) : ℝ := x.1 * y.1 + x.2 * y.2
def scale (a : ℝ) (x : V) : V := (a * x.1, a * x.2)
def ray (a : ℝ) : V := (a, a * (64 / 127))
def P (x : V) : V := ((4 / 5) * x.1 + (2 / 5) * x.2,
                       (2 / 5) * x.1 + (1 / 5) * x.2)
def gain : ℝ := 636 / 635

/-- Nearest integer, ties to even: adjust half-up exactly at odd upper ties. -/
def rne (x : ℝ) : ℤ :=
  let hi : ℤ := ⌊x + 1 / 2⌋
  if x = (hi : ℝ) - 1 / 2 ∧ hi % 2 ≠ 0 then hi - 1 else hi

def amax (x : V) : ℝ := max |x.1| |x.2|

def maxQuant (x : V) : V :=
  let m := amax x
  if m = 0 then (0, 0)
  else let s := m / 127
       (s * (rne (x.1 / s) : ℝ), s * (rne (x.2 / s) : ℝ))

theorem rne_127 : rne (127 : ℝ) = 127 := by norm_num [rne]
theorem rne_half_127 : rne ((127 : ℝ) / 2) = 64 := by norm_num [rne]

theorem P_idempotent (x : V) : P (P x) = P x := by
  apply Prod.ext <;> simp only [P] <;> ring

theorem P_self_adjoint (x y : V) : dot (P x) y = dot x (P y) := by
  simp only [P, dot]
  ring

theorem P_energy_identity (x : V) :
    energy x - energy (P x) = (x.1 - 2 * x.2) ^ 2 / 5 := by
  simp only [energy, P]
  ring

theorem P_energy_le (x : V) : energy (P x) ≤ energy x := by
  have h := P_energy_identity x
  nlinarith [sq_nonneg (x.1 - 2 * x.2)]

theorem P_ray (a : ℝ) : P (ray a) = (a * gain, (a * gain) / 2) := by
  apply Prod.ext <;> simp only [P, ray, gain] <;> ring

theorem maxQuant_half_ray (b : ℝ) (hb : 0 < b) :
    maxQuant (b, b / 2) = (b, b * (64 / 127)) := by
  have hm : amax (b, b / 2) = b := by
    simp only [amax, abs_of_pos hb, abs_of_pos (half_pos hb)]
    exact max_eq_left (by linarith)
  have hb0 : b ≠ 0 := ne_of_gt hb
  have hn1 : b / (b / 127) = (127 : ℝ) := by field_simp
  have hn2 : (b / 2) / (b / 127) = (127 : ℝ) / 2 := by field_simp
  simp only [maxQuant, hm, hb0, ↓reduceIte, hn1, hn2, rne_127, rne_half_127]
  apply Prod.ext
  · norm_num
  · norm_num
    ring

theorem gain_gt_one : 1 < gain := by norm_num [gain]

theorem exact_expanding_step (a : ℝ) (ha : 0 < a) :
    maxQuant (P (ray a)) = ray (gain * a) := by
  rw [P_ray, maxQuant_half_ray (a * gain) (mul_pos ha (by norm_num [gain]))]
  apply Prod.ext <;> simp only [ray] <;> ring

/-- The same fixed projection and row-amax quantizer give an exponential orbit. -/
theorem exact_expanding_orbit (a : ℝ) (ha : 0 < a) (n : ℕ) :
    (maxQuant ∘ P)^[n] (ray a) = ray (gain ^ n * a) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih]
    change maxQuant (P (ray (gain ^ n * a))) = _
    rw [exact_expanding_step _ (mul_pos (pow_pos (by norm_num [gain]) _) ha), pow_succ]
    congr 1
    ring

theorem ray_energy (a : ℝ) : energy (ray a) = a ^ 2 * (1 + (64 / 127 : ℝ) ^ 2) := by
  simp only [energy, ray]
  ring

theorem quantized_projection_energy_increases (a : ℝ) (ha : 0 < a) :
    energy (ray a) < energy (maxQuant (P (ray a))) := by
  rw [exact_expanding_step a ha, ray_energy, ray_energy]
  have hsq : 0 < a ^ 2 := sq_pos_of_pos ha
  norm_num [gain]
  nlinarith

end Q8Runaway
