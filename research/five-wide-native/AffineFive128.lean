import Lean.Elab.Tactic.Omega
namespace AffineFive128
-- Scale-free integer carrier dictionary. No CUDA conformance is claimed.
theorem low_digit_injective (g h u v : Int)
    (hg : -60 ≤ g ∧ g ≤ 60) (hh : -60 ≤ h ∧ h ≤ 60)
    (eq : g + 128*u = h + 128*v) : g = h ∧ u = v := by omega

theorem gu_injective (g u h v : Int)
    (hg : -60 ≤ g ∧ g ≤ 60) (hh : -60 ≤ h ∧ h ≤ 60)
    (eq : g + 128*u = h + 128*v) : (g,u) = (h,v) := by
  obtain ⟨a,b⟩ := low_digit_injective g h u v hg hh eq
  subst h; subst v; rfl

theorem four_partials_injective (g0 u0 g1 u1 h0 v0 h1 v1 : Int)
    (hg0 : -60 ≤ g0 ∧ g0 ≤ 60) (hh0 : -60 ≤ h0 ∧ h0 ≤ 60)
    (hu0 : -60 ≤ u0 ∧ u0 ≤ 60) (hv0 : -60 ≤ v0 ∧ v0 ≤ 60)
    (hg1 : -60 ≤ g1 ∧ g1 ≤ 60) (hh1 : -60 ≤ h1 ∧ h1 ≤ 60)
    (eq : (g1 + 128*u1)*16384 + g0 + 128*u0 =
          (h1 + 128*v1)*16384 + h0 + 128*v0) :
    (g0,u0,g1,u1) = (h0,v0,h1,v1) := by
  have e0 : g0 + 128*(u0 + 128*g1 + 16384*u1) =
            h0 + 128*(v0 + 128*h1 + 16384*v1) := by omega
  obtain ⟨a,b⟩ := low_digit_injective _ _ _ _ hg0 hh0 e0
  have e1 : u0 + 128*(g1 + 128*u1) = v0 + 128*(h1 + 128*v1) := by omega
  obtain ⟨c,d⟩ := low_digit_injective _ _ _ _ hu0 hv0 e1
  obtain ⟨e,f⟩ := low_digit_injective _ _ _ _ hg1 hh1 d
  subst h0; subst v0; subst h1; subst v1; rfl

theorem two_group_int32_bound (g0 u0 g1 u1 : Int)
    (h0 : -60 ≤ g0 ∧ g0 ≤ 60) (h1 : -60 ≤ u0 ∧ u0 ≤ 60)
    (h2 : -60 ≤ g1 ∧ g1 ≤ 60) (h3 : -60 ≤ u1 ∧ u1 ≤ 60) :
    -126819900 ≤ (g1+128*u1)*16384+g0+128*u0 ∧
    (g1+128*u1)*16384+g0+128*u0 ≤ 126819900 := by omega
end AffineFive128
