import Mathlib

namespace BinaryComposition

def scaledFold {n : ℕ} {Scale State : Type} (step : ℤ → Scale → State → State)
    (initial : State) (dot : Fin n → ℤ) (scale : Fin n → Scale) : State :=
  (List.ofFn fun i => (dot i, scale i)).foldl (fun acc p => step p.1 p.2 acc) initial

def decode (selected dc : ℤ) : ℤ := 2 * selected - dc

theorem affine_dot (positive negative : ℤ) :
    decode positive (positive + negative) = positive - negative := by
  unfold decode
  ring

-- Scale may carry original half scale and dynamic float activation scale.
-- State and step may be native IEEE/SFU state transitions rather than real arithmetic.
theorem ordered_continuation {n : ℕ} {Scale State : Type}
    (step : ℤ → Scale → State → State) (initial : State)
    (positive negative : Fin n → ℤ) (scale : Fin n → Scale) :
    scaledFold step initial (fun i => decode (positive i) (positive i + negative i)) scale =
    scaledFold step initial (fun i => positive i - negative i) scale := by
  congr 1
  funext i
  exact affine_dot _ _

-- Observer consumes an entire feature group, so it may include max and requant.
theorem whole_observer {m n : ℕ} {Scale State Output : Type}
    (step : ℤ → Scale → State → State) (initial : State)
    (observe : (Fin m → State) → (Fin m → State) → Output)
    (gp gn up un : Fin m → Fin n → ℤ) (gs us : Fin m → Fin n → Scale) :
    observe
      (fun f => scaledFold step initial (fun i => decode (gp f i) (gp f i + gn f i)) (gs f))
      (fun f => scaledFold step initial (fun i => decode (up f i) (up f i + un f i)) (us f)) =
    observe
      (fun f => scaledFold step initial (fun i => gp f i - gn f i) (gs f))
      (fun f => scaledFold step initial (fun i => up f i - un f i) (us f)) := by
  congr 1 <;> funext f <;> exact ordered_continuation _ _ _ _ _

theorem scale_coordinates_required :
    (1 : ℤ) + 0 = 0 + 1 ∧ (1 : ℤ) * 1 + 0 * 2 ≠ 0 * 1 + 1 * 2 := by
  norm_num

#print axioms affine_dot
#print axioms whole_observer
#print axioms scale_coordinates_required
end BinaryComposition
