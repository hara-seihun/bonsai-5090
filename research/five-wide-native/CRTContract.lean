namespace CRTFive

theorem residue33 (a b : Fin 25)
    (h : (a.val % 3, a.val % 11) = (b.val % 3, b.val % 11)) : a = b := by
  revert a b
  decide

theorem residue35 (a b : Fin 25)
    (h : (a.val % 5, a.val % 7) = (b.val % 5, b.val % 7)) : a = b := by
  revert a b
  decide

theorem residue91 (a b : Fin 25)
    (h : (a.val % 7, a.val % 13) = (b.val % 7, b.val % 13)) : a = b := by
  revert a b
  decide

theorem residue65 (a b : Fin 25)
    (h : (a.val % 13, a.val % 5) = (b.val % 13, b.val % 5)) : a = b := by
  revert a b
  decide

def primesA (i : Fin 5) : Nat := [3,5,7,11,13][i.val]!
def primesB (i : Fin 5) : Nat := [11,7,13,3,5][i.val]!

theorem joint_residues_injective (a b : Fin 5 → Fin 25)
    (h : ∀ i, ((a i).val % primesA i, (a i).val % primesB i) =
               ((b i).val % primesA i, (b i).val % primesB i)) : a = b := by
  funext i
  have hi := h i
  match i with
  | ⟨0, _⟩ => exact residue33 _ _ hi
  | ⟨1, _⟩ => exact residue35 _ _ hi
  | ⟨2, _⟩ => exact residue91 _ _ hi
  | ⟨3, _⟩ =>
    apply residue33
    exact Prod.ext (congrArg Prod.snd hi) (congrArg Prod.fst hi)
  | ⟨4, _⟩ => exact residue65 _ _ hi

-- Machine conformance is empirical; this theorem exposes that exact assumption.
theorem shared_label_continuation
    {Logical Physical Input : Type}
    (encode : Logical → Physical)
    (logicalStep : Logical → Input → Logical)
    (physicalStep : Physical → Input → Physical)
    (conformance : ∀ s i, physicalStep (encode s) i = encode (logicalStep s i))
    (s : Logical) (inputs : List Input) :
    inputs.foldl physicalStep (encode s) = encode (inputs.foldl logicalStep s) := by
  induction inputs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, conformance]
    exact ih (logicalStep s i)

end CRTFive
