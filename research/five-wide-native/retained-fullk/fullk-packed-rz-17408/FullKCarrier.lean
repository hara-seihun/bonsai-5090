import Lean.Elab.Tactic.Omega
import Lean.Elab.Tactic.BVDecide
namespace FullKCarrier
-- These are integer/code contracts; native MMA and IEEE conformance remain empirical.
def sumInt : List Int → Int
  | [] => 0
  | a :: xs => a + sumInt xs

theorem bounded_sum (xs : List Int) (bound : Int)
    (hx : ∀ x ∈ xs, -bound ≤ x ∧ x ≤ bound) :
    -bound * xs.length ≤ sumInt xs ∧ sumInt xs ≤ bound * xs.length := by
  induction xs with
  | nil => simp [sumInt]
  | cons a xs ih =>
    have ha := hx a (by simp)
    have ht : ∀ x ∈ xs, -bound ≤ x ∧ x ≤ bound := by
      intro x h; exact hx x (by simp [h])
    have hi := ih ht
    simp only [sumInt, List.length_cons, Int.natCast_add, Int.natCast_one,
      Int.mul_add, Int.mul_one]
    omega

theorem label_injective (g u h v : Int)
    (hg : -192 ≤ g ∧ g ≤ 192) (hh : -192 ≤ h ∧ h ≤ 192)
    (eq : g + 512*u = h + 512*v) : g = h ∧ u = v := by omega

theorem label_bound (g u : Int)
    (hg : -192 ≤ g ∧ g ≤ 192) (hu : -192 ≤ u ∧ u ≤ 192) :
    -98496 ≤ g+512*u ∧ g+512*u ≤ 98496 := by omega

theorem scaled_up_operands (u x : Int)
    (hu : -1 ≤ u ∧ u ≤ 1) (hx : -12 ≤ x ∧ x ≤ 12) :
    (-64 ≤ 64*u ∧ 64*u ≤ 64) ∧ (-96 ≤ 8*x ∧ 8*x ≤ 96) := by omega

def e4Units (c : Nat) : Nat :=
  if c < 8 then c else (8+c%8)*2^(c/8-1)
theorem e4_units_bound (c : Fin 127) : e4Units c.val ≤ 229376 := by
  revert c
  decide

theorem eight_term_units_bound (v0 v1 v2 v3 v4 v5 v6 v7 : Int)
    (h0 : -44040192 ≤ v0 ∧ v0 ≤ 44040192)
    (h1 : -44040192 ≤ v1 ∧ v1 ≤ 44040192)
    (h2 : -44040192 ≤ v2 ∧ v2 ≤ 44040192)
    (h3 : -44040192 ≤ v3 ∧ v3 ≤ 44040192)
    (h4 : -44040192 ≤ v4 ∧ v4 ≤ 44040192)
    (h5 : -44040192 ≤ v5 ∧ v5 ≤ 44040192)
    (h6 : -44040192 ≤ v6 ∧ v6 ≤ 44040192)
    (h7 : -44040192 ≤ v7 ∧ v7 ≤ 44040192) :
    -352321536 ≤ v0+v1+v2+v3+v4+v5+v6+v7 ∧
    v0+v1+v2+v3+v4+v5+v6+v7 ≤ 352321536 := by omega

theorem positive_term_cap (a s : Nat) (ha : a ≤ 384) (hs : s ≤ 229376) :
    a*s ≤ 88080384 := by
  have h := Nat.mul_le_mul ha hs
  omega

theorem paired_multiply (a b s : Nat) :
    (a+4294967296*b)*s = a*s+4294967296*(b*s) := by
  simp [Nat.add_mul, Nat.mul_assoc]

theorem paired_fields (a b s : Nat) (h : a*s < 4294967296) :
    ((a+4294967296*b)*s)%4294967296 = a*s ∧
    ((a+4294967296*b)*s)/4294967296 = b*s := by
  rw [paired_multiply]
  omega

-- Arbitrary reversible masks include the implemented XOR/rotate mixed-word carrier.
def mix (f g : BitVec 32 → BitVec 32) (a b : BitVec 32) : BitVec 32 × BitVec 32 :=
  let lo := a ^^^ f b
  (lo, b ^^^ g lo)
def unmix (f g : BitVec 32 → BitVec 32) (lo hi : BitVec 32) : BitVec 32 × BitVec 32 :=
  let b := hi ^^^ g lo
  (lo ^^^ f b, b)
theorem mixed_words_recover (f g : BitVec 32 → BitVec 32) (a b : BitVec 32) :
    unmix f g (mix f g a b).1 (mix f g a b).2 = (a,b) := by
  simp [mix, unmix, BitVec.xor_assoc]

-- A prefix's declared observer-equivalence is sufficient for arbitrary downstream maps.
theorem observer_factor {Raw Label Partial Out : Type}
    (produce : Raw → Label) (decode : Label → Partial) (native : Raw → Partial)
    (next : Partial → Out) (conformance : ∀ r, decode (produce r) = native r) :
    ∀ r, next (decode (produce r)) = next (native r) := by
  intro r; rw [conformance]
-- Declared observer order is the list order, including each 128-K scale context.
theorem ordered_observer {Raw Label Ctx State : Type}
    (produce : Raw → Label)
    (physical : State → Label × Ctx → State)
    (native : State → Raw × Ctx → State)
    (conformance : ∀ s r c, physical s (produce r,c) = native s (r,c))
    (s : State) (inputs : List (Raw × Ctx)) :
    inputs.foldl (fun state rc => physical state (produce rc.1,rc.2)) s =
    inputs.foldl native s := by
  simp only [conformance]

end FullKCarrier
