import Lean.Elab.Tactic.BVDecide
namespace PackedTail
abbrev Word := BitVec 32
abbrev Pair := Word × Word
def label (g u : Word) : Pair :=
  let lo := g ^^^ u
  (lo, u ^^^ lo.rotateLeft 7)
def read (l : Pair) : Pair :=
  let u := l.2 ^^^ l.1.rotateLeft 7
  (l.1 ^^^ u, u)
theorem exact_read (g u : Word) : read (label g u) = (g,u) := by
  simp [read,label,BitVec.xor_assoc]
theorem injective (g u h v : Word) (eq : label g u = label h v) : (g,u) = (h,v) := by
  have e := congrArg read eq
  simpa [exact_read] using e
def target {Operand : Type} (observe : Pair → Operand) (s : Pair) : Operand := observe s
def consumer {Operand : Type} (observe : Pair → Operand) (l : Pair) : Operand := observe (read l)
theorem terminal_exact {Operand : Type} (observe : Pair → Operand) (g u : Word) :
    consumer observe (label g u) = target observe (g,u) := by
  simp [consumer,target,exact_read]
-- The observer here may be the complete source-ordered SiLU/sign/Hadamard/quantizer.
-- That observer's CUDA/F32 correspondence and layout are separate empirical obligations.
theorem region_exact {Input Operand : Type} (produce : Input → Pair)
    (observe : Pair → Operand) (x : Input) :
    consumer observe (label (produce x).1 (produce x).2) = observe (produce x) := by
  simp [consumer,exact_read]
def firstPair (silu : Word → Word → Word) (sign add sub : Word → Word → Word)
    (g0 u0 g1 u1 s0 s1 : Word) : Pair :=
  let a := sign (silu g0 u0) s0
  let b := sign (silu g1 u1) s1
  (add a b, sub a b)
theorem h1_terminal_exact {Operand : Type} (observe : Pair → Operand)
    (silu : Word → Word → Word) (sign add sub : Word → Word → Word)
    (g0 u0 g1 u1 s0 s1 : Word) :
    let p := firstPair silu sign add sub g0 u0 g1 u1 s0 s1
    consumer observe (label p.1 p.2) = observe p := by
  simp [consumer, exact_read]
theorem peer_row (lane : BitVec 5) : ((lane ^^^ 4) &&& 3) = (lane &&& 3) := by bv_decide
theorem peer_feature (lane : BitVec 5) : ((lane ^^^ 4) >>> 2) = ((lane >>> 2) ^^^ 1) := by bv_decide
end PackedTail
