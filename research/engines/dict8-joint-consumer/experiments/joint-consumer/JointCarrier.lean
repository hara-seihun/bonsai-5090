import Std
namespace JointCarrier
-- The two coordinates may be renamed independently without changing any trit word.
def pack53 (low : Fin 243) (high : Fin 27) : Nat := low.val + 256 * high.val
def radix8 (low : Nat) (high : Nat) : Nat := low + 243 * high
 theorem split_fits (low : Fin 243) (high : Fin 27) : pack53 low high < 8192 := by
  simp [pack53]; omega
 theorem split_low (low : Fin 243) (high : Fin 27) : pack53 low high % 256 = low.val := by
  simp [pack53]; omega
 theorem split_high (low : Fin 243) (high : Fin 27) : pack53 low high / 256 = high.val := by
  simp [pack53]; omega
 theorem code_inverse (x : Fin 6561) : radix8 (x.val % 243) (x.val / 243) = x.val := by
  simp [radix8]; omega
 theorem split_injective (a c : Fin 243) (b d : Fin 27)
   (h : pack53 a b = pack53 c d) : a = c ∧ b = d := by
  have lo := congrArg (fun x => x % 256) h
  have hi := congrArg (fun x => x / 256) h
  rw [split_low, split_low] at lo
  rw [split_high, split_high] at hi
  exact ⟨Fin.ext lo, Fin.ext hi⟩
-- This applies to the whole opaque native suffix, including rounded separate gate/up,
-- means, SiLU and q4/s4/g4. The premise must be established at the native boundary.
 theorem same_observer {State Result : Type} (observe : State → Result)
   (source candidate : State) (same : source = candidate) :
   observe source = observe candidate := congrArg observe same
end JointCarrier
