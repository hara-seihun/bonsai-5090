import Lean.Elab.Tactic.Omega

namespace PackedGrid

-- Each scaled doubled partial is in units 2^-10.
-- This models the integer coordinate, not NVIDIA floating arithmetic.
def coefficient (code : Nat) : Nat :=
  if code < 8 then code else (8 + code % 8) * 2 ^ (code / 8 - 1)

theorem coefficient_bound (code : Fin 127) : coefficient code.val ≤ 229376 := by
  exact (by decide : ∀ c : Fin 127, coefficient c.val ≤ 229376) code

def contribution (g : Int) (code : Fin 127) : Int := g * coefficient code.val

theorem contribution_bound (g : Int) (code : Fin 127)
    (hg : -192 ≤ g ∧ g ≤ 192) :
    -44040192 ≤ contribution g code ∧ contribution g code ≤ 44040192 := by
  have hcNat := coefficient_bound code
  have hc : (coefficient code.val : Int) ≤ 229376 := by omega
  have hc0 : 0 ≤ (coefficient code.val : Int) := Int.ofNat_zero_le _
  have lower := Int.mul_le_mul_of_nonneg_right hg.1 hc0
  have upper := Int.mul_le_mul_of_nonneg_right hg.2 hc0
  have bound := Int.mul_le_mul_of_nonneg_left hc (by decide : (0 : Int) ≤ 192)
  unfold contribution
  omega

def trim (n quantum : Nat) : Nat := n / quantum * quantum

theorem trim_no_increase (n quantum : Nat) : trim n quantum ≤ n := by
  exact Nat.div_mul_le_self n quantum

theorem trim_loss (n quantum : Nat) (hq : 0 < quantum) :
    n - trim n quantum < quantum := by
  have hm := Nat.mod_lt n hq
  have hd := Nat.mod_add_div n quantum
  have hd' : n % quantum + n / quantum * quantum = n := by
    simpa [Nat.mul_comm] using hd
  unfold trim
  omega

theorem trim_idempotent (n quantum : Nat) (hq : 0 < quantum) :
    trim (trim n quantum) quantum = trim n quantum := by
  have hd : (n / quantum * quantum) / quantum = n / quantum := by
    simpa [Nat.mul_comm] using Nat.mul_div_cancel_left (n / quantum) hq
  unfold trim
  rw [hd]

theorem exact_lattice (n lattice quantum : Nat)
    (hq : quantum ∣ lattice) (hn : lattice ∣ n) : trim n quantum = n := by
  have hd := Nat.dvd_trans hq hn
  have hm := Nat.mod_eq_zero_of_dvd hd
  have he := Nat.mod_add_div n quantum
  simpa [trim, hm, Nat.mul_comm] using he

theorem shear_inverse (g u : Int) :
    ((g + u) + (g - u)) / 2 = g ∧
    ((g + u) - (g - u)) / 2 = u := by omega

theorem add_budget (a b ra rb : Int)
    (ha : -ra ≤ a ∧ a ≤ ra) (hb : -rb ≤ b ∧ b ≤ rb) :
    -(ra + rb) ≤ a + b ∧ a + b ≤ ra + rb := by omega

theorem k128_signed32_budget : 8 * 44040192 < (2147483648 : Nat) := by decide

end PackedGrid
