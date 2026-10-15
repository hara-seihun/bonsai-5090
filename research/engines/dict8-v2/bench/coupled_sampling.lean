def generateTokens (step : List Nat → Nat) (history : List Nat) : Nat → List Nat
  | 0 => []
  | n + 1 => step history :: generateTokens step (history ++ [step history]) n

def verifyTokens (step : List Nat → Nat) (history : List Nat) : List Nat → List Nat
  | [] => [step history]
  | proposal :: rest =>
      if step history = proposal then
        proposal :: verifyTokens step (history ++ [proposal]) rest
      else [step history]

-- `step` includes fixed request seed and position-keyed noise. For any draft,
-- verification emits precisely a prefix of that serial sampled realization.
-- The draft need not have the target distribution or use target logits.
theorem verified_prefix_is_serial (step : List Nat → Nat) (history draft : List Nat) :
    generateTokens step history (verifyTokens step history draft).length =
      verifyTokens step history draft := by
  induction draft generalizing history with
  | nil => simp [verifyTokens, generateTokens]
  | cons proposal rest ih =>
      by_cases h : step history = proposal
      · simp [verifyTokens, h, generateTokens, ih]
      · simp [verifyTokens, h, generateTokens]
