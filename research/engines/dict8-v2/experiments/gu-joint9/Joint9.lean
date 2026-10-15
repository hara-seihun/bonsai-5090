import Lean.Elab.Tactic.BVDecide
namespace Joint9
-- g,u here are shifted ternary digits 0,1,2. Their joint digit is 0..8.
def encode (g u : Fin 3) : Nat := g.val + 3*u.val
def readG (d : Nat) : Nat := d % 3
def readU (d : Nat) : Nat := d / 3
theorem pair_exact : ∀ g u : Fin 3, readG (encode g u) = g.val ∧ readU (encode g u) = u.val := by decide
-- Native E2M1 operand words for the original ternary -1/0/+1 values.
def tritCode (d : Nat) : Nat := if d=0 then 10 else if d=2 then 2 else 0
theorem operand_exact : ∀ g u : Fin 3,
    tritCode (readG (encode g u)) = tritCode g.val ∧
    tritCode (readU (encode g u)) = tritCode u.val := by
  intro g u
  rcases pair_exact g u with ⟨hg,hu⟩
  simp [hg,hu]
-- Native MMA is deliberately opaque. No RN/RTZ or distributivity assumption.
def replay {State Operand : Type} (mma : State → Operand → State)
    (start : State) (instructions : List Operand) : State := instructions.foldl mma start
theorem same_instruction_stream {State Operand : Type}
    (mma : State → Operand → State) (start : State) (old new : List Operand)
    (operands_match : old = new) : replay mma start old = replay mma start new := by
  rw [operands_match]
theorem same_full_observer {State Operand Output : Type}
    (mma : State → Operand → State) (observe : State → Output)
    (start : State) (old new : List Operand) (operands_match : old = new) :
    observe (replay mma start old) = observe (replay mma start new) := by
  rw [operands_match]
-- Actual CUDA word packing and ordered instruction-stream equality are still
-- checked against the source/layout/compiler evidence, not supplied by algebra.
end Joint9
