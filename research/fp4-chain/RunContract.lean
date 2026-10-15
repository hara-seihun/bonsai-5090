import Lean.Elab.Tactic.Omega

namespace PackedRun

def carrier (g u : Int) : Int := g + 5376 * u

-- K192: doubled FP4 dot sums have absolute value at most 12 * 192.
theorem injective (g₀ g₁ u₀ u₁ : Int)
    (h₀ : -2304 ≤ g₀ ∧ g₀ ≤ 2304)
    (h₁ : -2304 ≤ g₁ ∧ g₁ ≤ 2304)
    (same : carrier g₀ u₀ = carrier g₁ u₁) :
    g₀ = g₁ ∧ u₀ = u₁ := by
  unfold carrier at same
  omega

theorem exact_integer_budget (g u : Int)
    (hg : -2304 ≤ g ∧ g ≤ 2304) (hu : -2304 ≤ u ∧ u ≤ 2304) :
    -12388608 ≤ carrier g u ∧ carrier g u ≤ 12388608 := by
  unfold carrier
  omega

theorem continuation (g u dg du : Int) :
    carrier g u + carrier dg du = carrier (g + dg) (u + du) := by
  unfold carrier
  omega

theorem observer_context {Context Target : Type}
    (f : Int → Int → Context → Target) (ctx : Context)
    (g₀ g₁ u₀ u₁ : Int)
    (h₀ : -2304 ≤ g₀ ∧ g₀ ≤ 2304)
    (h₁ : -2304 ≤ g₁ ∧ g₁ ≤ 2304)
    (same : carrier g₀ u₀ = carrier g₁ u₁) :
    f g₀ u₀ ctx = f g₁ u₁ ctx := by
  obtain ⟨rfl, rfl⟩ := injective g₀ g₁ u₀ u₁ h₀ h₁ same
  rfl

end PackedRun
