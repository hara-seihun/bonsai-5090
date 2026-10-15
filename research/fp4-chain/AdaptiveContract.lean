import Lean.Elab.Tactic.Omega

namespace PackedAdaptive

def radix (r : Int) : Int := if r ≤ 127 then 256 else 448

def carrier (r g u : Int) : Int := g + radix r * u

theorem carrier_injective (r g₀ g₁ u₀ u₁ : Int)
    (hr : 0 ≤ r ∧ r ≤ 192)
    (hg₀ : -r ≤ g₀ ∧ g₀ ≤ r) (hg₁ : -r ≤ g₁ ∧ g₁ ≤ r)
    (h : carrier r g₀ u₀ = carrier r g₁ u₁) :
    g₀ = g₁ ∧ u₀ = u₁ := by
  by_cases narrow : r ≤ 127
  · simp [carrier, radix, narrow] at h
    omega
  · simp [carrier, radix, narrow] at h
    omega

theorem carrier_bounds (r g u : Int)
    (hr : 0 ≤ r ∧ r ≤ 192)
    (hg : -r ≤ g ∧ g ≤ r) (hu : -r ≤ u ∧ u ≤ r) :
    (r ≤ 127 → -32639 ≤ carrier r g u ∧ carrier r g u ≤ 32639) ∧
    (-86208 ≤ carrier r g u ∧ carrier r g u ≤ 86208) := by
  by_cases narrow : r ≤ 127
  · simp [carrier, radix, narrow]
    omega
  · simp [carrier, radix, narrow]
    omega

-- Residues are carrier data, not activation/scale context. Their widths follow the radius tag.
def low (n : Int) : Int := n % 65536
def high (n : Int) : Int := (n / 65536) % 4

theorem narrow_low_injective (a b : Int)
    (ha : -32639 ≤ a ∧ a ≤ 32639) (hb : -32639 ≤ b ∧ b ≤ 32639)
    (h : low a = low b) : a = b := by
  unfold low at h
  omega

theorem wide_limbs_injective (a b : Int)
    (ha : -86208 ≤ a ∧ a ≤ 86208) (hb : -86208 ≤ b ∧ b ≤ 86208)
    (hl : low a = low b) (hh : high a = high b) : a = b := by
  unfold low at hl
  unfold high at hh
  omega

theorem two_low_join_injective (a b c d : Int)
    (h : low a + 65536 * low b = low c + 65536 * low d) :
    low a = low c ∧ low b = low d := by
  unfold low at *
  omega

end PackedAdaptive
