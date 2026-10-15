import Mathlib

namespace BatchMap

def pack (lo hi : ℕ) : ℕ := lo + 65536 * hi

theorem lanes (lo hi : ℕ) (hlo : lo ≤ 32640) (hhi : hi ≤ 32640) :
    pack lo hi % 65536 = lo ∧ pack lo hi / 65536 = hi ∧ pack lo hi < 2^32 := by
  unfold pack
  omega

theorem step (lo hi u v b : ℕ) (hlo : lo + b*u ≤ 32640)
    (hhi : hi + b*v ≤ 32640) :
    pack lo hi + b * pack u v = pack (lo+b*u) (hi+b*v) ∧
    (pack lo hi + b * pack u v) % 65536 = lo+b*u ∧
    (pack lo hi + b * pack u v) / 65536 = hi+b*v := by
  have eq : pack lo hi + b * pack u v = pack (lo+b*u) (hi+b*v) := by
    unfold pack
    ring
  rw [eq]
  exact ⟨rfl, (lanes _ _ hlo hhi).1, (lanes _ _ hlo hhi).2.1⟩

theorem selected_bound (u : Fin 128 → ℕ) (b : Fin 128 → Bool)
    (hu : ∀ i, u i ≤ 255) :
    (∑ i, if b i then u i else 0) ≤ 32640 := by
  calc
    _ ≤ ∑ _i : Fin 128, (255 : ℕ) := Finset.sum_le_sum (fun i _ => by split <;> simp_all)
    _ = 32640 := by norm_num

theorem correction (q : Fin 128 → ℤ) (b : Fin 128 → Bool) :
    (∑ i, (if b i then q i else -q i)) =
    2 * (∑ i, if b i then q i + 128 else 0) - (∑ i, q i) -
    256 * (∑ i, if b i then (1 : ℤ) else 0) := by
  simp only [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  cases h : b i <;> simp [h] <;> ring

theorem inverse (s dc count : ℤ) :
    (2*s-dc-256*count + dc+256*count) / 2 = s := by omega

#print axioms lanes
#print axioms step
#print axioms selected_bound
#print axioms correction
#print axioms inverse
end BatchMap
