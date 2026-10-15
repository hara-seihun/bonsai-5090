import Lean.Elab.Tactic.Omega

namespace PackedAffine

def carrier (g u role : Int) : Int := g + 64 * u + 4096 * role

theorem carrier_injective
    (g₁ g₂ u₁ u₂ r₁ r₂ : Int)
    (hg₁ : -24 ≤ g₁ ∧ g₁ ≤ 24) (hg₂ : -24 ≤ g₂ ∧ g₂ ≤ 24)
    (hu₁ : -24 ≤ u₁ ∧ u₁ ≤ 24) (hu₂ : -24 ≤ u₂ ∧ u₂ ≤ 24)
    (h : carrier g₁ u₁ r₁ = carrier g₂ u₂ r₂) :
    g₁ = g₂ ∧ u₁ = u₂ ∧ r₁ = r₂ := by
  unfold carrier at h
  omega

theorem any_context_observer
    {Context Target : Type} (f : Int → Int → Int → Context → Target)
    (g₁ g₂ u₁ u₂ r₁ r₂ : Int) (context : Context)
    (hg₁ : -24 ≤ g₁ ∧ g₁ ≤ 24) (hg₂ : -24 ≤ g₂ ∧ g₂ ≤ 24)
    (hu₁ : -24 ≤ u₁ ∧ u₁ ≤ 24) (hu₂ : -24 ≤ u₂ ∧ u₂ ≤ 24)
    (h : carrier g₁ u₁ r₁ = carrier g₂ u₂ r₂) :
    f g₁ u₁ r₁ context = f g₂ u₂ r₂ context := by
  obtain ⟨rfl, rfl, rfl⟩ := carrier_injective g₁ g₂ u₁ u₂ r₁ r₂ hg₁ hg₂ hu₁ hu₂ h
  rfl

def fourCarrier (g₀ u₀ g₁ u₁ : Int) : Int :=
  g₀ + 64 * u₀ + 4096 * g₁ + 262144 * u₁

theorem four_carrier_injective
    (g₀ u₀ g₁ u₁ h₀ v₀ h₁ v₁ : Int)
    (bounds : -24 ≤ g₀ ∧ g₀ ≤ 24 ∧ -24 ≤ u₀ ∧ u₀ ≤ 24 ∧
      -24 ≤ g₁ ∧ g₁ ≤ 24 ∧ -24 ≤ u₁ ∧ u₁ ≤ 24 ∧
      -24 ≤ h₀ ∧ h₀ ≤ 24 ∧ -24 ≤ v₀ ∧ v₀ ≤ 24 ∧
      -24 ≤ h₁ ∧ h₁ ≤ 24 ∧ -24 ≤ v₁ ∧ v₁ ≤ 24)
    (h : fourCarrier g₀ u₀ g₁ u₁ = fourCarrier h₀ v₀ h₁ v₁) :
    g₀ = h₀ ∧ u₀ = v₀ ∧ g₁ = h₁ ∧ u₁ = v₁ := by
  unfold fourCarrier at h
  omega

def sixteenCarrier (g u : Int) : Int := g + 512 * u

theorem sixteen_carrier_injective
    (g₀ g₁ u₀ u₁ : Int)
    (bounds : -192 ≤ g₀ ∧ g₀ ≤ 192 ∧ -192 ≤ g₁ ∧ g₁ ≤ 192 ∧
      -192 ≤ u₀ ∧ u₀ ≤ 192 ∧ -192 ≤ u₁ ∧ u₁ ≤ 192)
    (h : sixteenCarrier g₀ u₀ = sixteenCarrier g₁ u₁) :
    g₀ = g₁ ∧ u₀ = u₁ := by
  unfold sixteenCarrier at h
  omega

end PackedAffine
