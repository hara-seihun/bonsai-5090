import Std
namespace Dict8
-- The native nibble labels are exactly -1, zero, +1; no arithmetic rounding is modeled.
def nativeNibble (x : Fin 3) : Nat := if x.val = 0 then 10 else if x.val = 1 then 0 else 2
def digitOfNibble (x : Nat) : Nat := if x = 10 then 0 else if x = 0 then 1 else 2
theorem digit_roundtrip : ∀ x : Fin 3, digitOfNibble (nativeNibble x) = x.val := by decide
theorem dictionary_key_fits : 6561 < 8192 := by decide
theorem sibling_keys (g u : Nat) (hg : g < 8192) (_hu : u < 8192) :
    (g + 8192*u) % 8192 = g ∧ (g + 8192*u) / 8192 = u := by omega
-- Replaying the same operand/control stream through any native observer yields the same result.
theorem opaque_native_replay {Operands Result : Type} (observe : Operands → Result)
    (original candidate : Operands) (same : original = candidate) :
    observe original = observe candidate := congrArg observe same
end Dict8
