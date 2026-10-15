import Std
namespace Ternary5

def bit (c : Fin 32) (i : Nat) : Bool := c.val.testBit i

def lut (imm : Nat) (a b c : Bool) : Bool :=
  imm.testBit (4*a.toNat + 2*b.toNat + c.toNat)

def planes (c : Fin 32) : Fin 6 → Bool :=
  let a := bit c 0; let b := bit c 1; let k := bit c 2
  let d := bit c 3; let e := bit c 4
  let m0 := lut 0xd5 a d e
  let s0 := lut 0x82 d k m0
  let m1 := lut 0x97 b d e
  let s1 := lut 0xc0 k m1 false
  let m2 := lut 0x3c b m1 false
  let s2 := lut 0x24 a b m1
  fun i => if i.val=0 then m0 else if i.val=1 then s0 else
    if i.val=2 then m1 else if i.val=3 then s1 else if i.val=4 then m2 else s2

def nibble (m s : Bool) : Nat := 2*m.toNat+8*s.toNat

def decode (c : Fin 32) : Nat × Nat × Nat :=
  let p := planes c
  (nibble (p 0) (p 1), nibble (p 2) (p 3), nibble (p 4) (p 5))

def canonical (c : Fin 32) : Bool :=
  !(c.val==3 || c.val==7 || c.val==22 || c.val==23 || c.val==28)

def native (n : Nat) : Bool := n==0 || n==2 || n==10

def ofTrit (t : Fin 3) : Nat := if t.val==0 then 0 else if t.val==1 then 2 else 10

def encode (a b c : Fin 3) : Option (Fin 32) :=
  (List.finRange 32).find? fun x => canonical x && decode x == (ofTrit a,ofTrit b,ofTrit c)

set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

theorem capacity : 2^4 < 27 ∧ 27 ≤ 2^5 := by decide

theorem all_raw_outputs_native : ∀ c : Fin 32,
    let t := decode c
    native t.1 = true ∧ native t.2.1 = true ∧ native t.2.2 = true := by decide

theorem full_source_coverage : ∀ a b c : Fin 3, ∃ x : Fin 32,
    canonical x = true ∧ decode x = (ofTrit a,ofTrit b,ofTrit c) := by decide

theorem canonical_injective : ∀ x y : Fin 32,
    canonical x = true → canonical y = true → decode x = decode y → x = y := by decide

theorem encoder_roundtrip : ∀ a b c : Fin 3, ∃ x : Fin 32,
    encode a b c = some x ∧ canonical x = true ∧
    decode x = (ofTrit a,ofTrit b,ofTrit c) := by decide

theorem canonical_count : ((List.finRange 32).filter canonical).length = 27 := by decide

-- These discharge the output-count lower bound for one-result Boolean gates,
-- with five input planes and optional free input complements/constants.
theorem no_output_is_primary : ∀ o : Fin 6, ∀ j : Fin 5,
    (∃ c : Fin 32, canonical c = true ∧ planes c o ≠ bit c j.val) ∧
    (∃ c : Fin 32, canonical c = true ∧ planes c o ≠ !(bit c j.val)) := by decide

theorem no_output_is_constant : ∀ o : Fin 6,
    (∃ c : Fin 32, canonical c = true ∧ planes c o = false) ∧
    (∃ c : Fin 32, canonical c = true ∧ planes c o = true) := by decide

theorem outputs_distinct : ∀ o p : Fin 6, o ≠ p →
    ∃ c : Fin 32, canonical c = true ∧ planes c o ≠ planes c p := by decide

def rawPair (g u : Fin 3) : Nat := ofTrit g ||| (ofTrit u >>> 1)

theorem raw_pair_restores : ∀ g u : Fin 3,
    (rawPair g u &&& 10) = ofTrit g ∧
    ((rawPair g u <<< 1) &&& 10) = ofTrit u := by decide

#print axioms encoder_roundtrip
#print axioms canonical_injective
#print axioms no_output_is_primary
end Ternary5
