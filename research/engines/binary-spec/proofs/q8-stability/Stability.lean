import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.Order.Round
import Mathlib.Tactic

noncomputable section
namespace Q8Stability

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Least-squares dequantization along any chosen code vector, including zero. -/
def ls (x q : E) : E := (inner ℝ x q / inner ℝ q q) • q

theorem ls_zero (x : E) : ls x 0 = 0 := by simp [ls]

theorem ls_residual_orthogonal (x q : E) : inner ℝ (x - ls x q) q = 0 := by
  by_cases hq : q = 0
  · simp [hq]
  · rw [inner_sub_left, ls, real_inner_smul_left,
      div_mul_cancel₀ _ (inner_self_ne_zero.mpr hq)]
    ring

theorem ls_pythagoras (x q : E) :
    ‖x - ls x q‖ ^ 2 + ‖ls x q‖ ^ 2 = ‖x‖ ^ 2 := by
  have h : inner ℝ (x - ls x q) (ls x q) = 0 := by
    change inner ℝ (x - ls x q) ((inner ℝ x q / inner ℝ q q) • q) = 0
    rw [real_inner_smul_right, ls_residual_orthogonal, mul_zero]
  have hn := norm_add_sq_real (x - ls x q) (ls x q)
  rw [sub_add_cancel, h] at hn
  linarith

/-- Origin nonexpansion, not a pairwise Lipschitz claim for adaptive codes. -/
theorem ls_norm_le (x q : E) : ‖ls x q‖ ≤ ‖x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  have hp := ls_pythagoras x q
  nlinarith [sq_nonneg ‖x - ls x q‖]

theorem adaptive_ls_norm_le (codes : E → E) (x : E) :
    ‖ls x (codes x)‖ ≤ ‖x‖ := ls_norm_le x (codes x)

/-- GDN's rank-one homogeneous update before scalar decay. -/
def rankOne (β : ℝ) (k x : E) : E := x - (β * inner ℝ k x) • k

theorem rankOne_energy (β : ℝ) (k x : E) :
    ‖rankOne β k x‖ ^ 2 =
      ‖x‖ ^ 2 - β * (2 - β * ‖k‖ ^ 2) * (inner ℝ k x) ^ 2 := by
  unfold rankOne
  rw [norm_sub_sq_real, real_inner_smul_right, real_inner_comm x k]
  rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  ring

theorem rankOne_norm_le (β : ℝ) (k x : E)
    (hβ : 0 ≤ β * ‖k‖ ^ 2) (hβ2 : β * ‖k‖ ^ 2 ≤ 2) :
    ‖rankOne β k x‖ ≤ ‖x‖ := by
  by_cases hk : k = 0
  · simp [rankOne, hk]
  have hkpos : 0 < ‖k‖ ^ 2 := sq_pos_of_pos (norm_pos_iff.mpr hk)
  have hβpos : 0 ≤ β := by nlinarith
  have hc : 0 ≤ β * (2 - β * ‖k‖ ^ 2) * (inner ℝ k x) ^ 2 :=
    mul_nonneg (mul_nonneg hβpos (by linarith)) (sq_nonneg _)
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [rankOne_energy]
  linarith

def gdn (g β : ℝ) (k x : E) : E := g • rankOne β k x

theorem gdn_norm_le (g β : ℝ) (k x : E)
    (hg : 0 ≤ g) (hg1 : g ≤ 1)
    (hβ : 0 ≤ β * ‖k‖ ^ 2) (hβ2 : β * ‖k‖ ^ 2 ≤ 2) :
    ‖gdn g β k x‖ ≤ ‖x‖ := by
  rw [gdn, norm_smul, Real.norm_eq_abs, abs_of_nonneg hg]
  calc
    g * ‖rankOne β k x‖ ≤ ‖rankOne β k x‖ :=
      mul_le_of_le_one_left (norm_nonneg _) hg1
    _ ≤ ‖x‖ := rankOne_norm_le β k x hβ hβ2

theorem quantized_step_bound (T codes : E → E)
    (hT : ∀ x, ‖T x‖ ≤ ‖x‖) (x forcing : E) :
    ‖ls (T x + forcing) (codes (T x + forcing))‖ ≤ ‖x‖ + ‖forcing‖ := by
  calc
    _ ≤ ‖T x + forcing‖ := ls_norm_le _ _
    _ ≤ ‖T x‖ + ‖forcing‖ := norm_add_le _ _
    _ ≤ ‖x‖ + ‖forcing‖ := by gcongr; exact hT x

theorem quantized_homogeneous_bound (T codes : E → E)
    (hT : ∀ x, ‖T x‖ ≤ ‖x‖) (x : E) :
    ‖ls (T x) (codes (T x))‖ ≤ ‖x‖ :=
  (ls_norm_le _ _).trans (hT x)

omit [InnerProductSpace ℝ E] in
 theorem linear_forcing_bound (state forcing : ℕ → E) (B : ℝ)
    (hstep : ∀ n, ‖state (n + 1)‖ ≤ ‖state n‖ + ‖forcing n‖)
    (hforcing : ∀ n, ‖forcing n‖ ≤ B) (n : ℕ) :
    ‖state n‖ ≤ ‖state 0‖ + (n : ℝ) * B := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hs := hstep n
    have hf := hforcing n
    push_cast
    nlinarith

end Q8Stability
