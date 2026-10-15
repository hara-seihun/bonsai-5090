namespace PackedObserver

-- An observer is required only on reachable physical labels.
theorem exists_reachable_observer
    {Input Carrier Target : Type}
    (encode : Input → Carrier) (target : Input → Target)
    (fibres : ∀ x y, encode x = encode y → target x = target y) :
    ∃ decode : {c : Carrier // ∃ x, encode x = c} → Target,
      ∀ x, decode ⟨encode x, ⟨x, rfl⟩⟩ = target x := by
  classical
  let decode : {c : Carrier // ∃ x, encode x = c} → Target :=
    fun c => target (Classical.choose c.property)
  refine ⟨decode, ?_⟩
  intro x
  apply fibres
  exact Classical.choose_spec (show ∃ y, encode y = encode x from ⟨x, rfl⟩)

-- Deterministic prefix compression preserves every physical continuation.
theorem equal_prefix_same_continuation
    {State Input : Type} (step : State → Input → State)
    (a b : State) (inputs : List Input) (equal : a = b) :
    inputs.foldl step a = inputs.foldl step b := by
  cases equal
  rfl

-- Logical target state can be carried with physical state in the proof oracle.
-- This pair is not a required runtime intermediate representation.
theorem equal_oracle_prefix_same_continuation
    {Physical Logical Input : Type}
    (physicalStep : Physical → Input → Physical)
    (logicalStep : Logical → Input → Logical)
    (a b : Physical × Logical) (inputs : List Input) (equal : a = b) :
    inputs.foldl (fun s i => (physicalStep s.1 i, logicalStep s.2 i)) a =
    inputs.foldl (fun s i => (physicalStep s.1 i, logicalStep s.2 i)) b := by
  cases equal
  rfl

end PackedObserver
