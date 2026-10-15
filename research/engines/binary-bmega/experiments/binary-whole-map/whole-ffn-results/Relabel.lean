import Std

namespace WholeFFN

def sourceRow (hidden prepared : Nat) : Nat :=
  (prepared / 256) * 128 + prepared % 128 +
    if prepared % 256 < 128 then 0 else hidden

def preparedRow (hidden source : Nat) : Nat :=
  ((source % hidden) / 128) * 256 + source % 128 +
    if source < hidden then 0 else 128

-- These are the actual layer0 coordinates used by the native complete-block witness.
theorem layer0_row_inverse :
    ∀ i : Fin 34816, preparedRow 17408 (sourceRow 17408 i.val) = i.val := by
  intro i
  unfold preparedRow sourceRow
  split <;> split <;> omega

theorem layer0_row_range :
    ∀ i : Fin 34816, sourceRow 17408 i.val < 34816 := by
  intro i
  unfold sourceRow
  split <;> omega

end WholeFFN
