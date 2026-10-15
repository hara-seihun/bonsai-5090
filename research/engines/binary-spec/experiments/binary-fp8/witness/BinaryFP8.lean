import Mathlib

namespace BinaryFP8

/-- Exactly the unsigned 32-bit expansion, with byte lanes numbered low to high. -/
def expand (n : Fin 16) : BitVec 32 :=
  0xb8b8b8b8 ^^^ (((BitVec.ofNat 32 n.val * 0x00204081) &&& 0x01010101) <<< 7)

def positive (n : Fin 16) (lane : Fin 4) : Bool :=
  decide (n.val / 2 ^ lane.val % 2 = 1)

def carrierByte (p : Bool) : Nat := if p then 0x38 else 0xb8

def laneByte (n : Fin 16) (lane : Fin 4) : Nat :=
  (expand n).toNat / 2 ^ (8 * lane.val) % 256

/-- Kernel-reduced exhaustive statement: all 16 nibbles and all four lanes. -/
theorem all_lane_codes : ∀ n : Fin 16, ∀ lane : Fin 4,
    laneByte n lane = carrierByte (positive n lane) := by decide

/-- E4M3FN: bias 7, three fraction bits, subnormals at exponent zero;
    exponent 15 / fraction 7 is NaN, all other byte patterns are finite. -/
def decodeE4M3 (byte : Fin 256) : Option ℚ :=
  let e := byte.val / 8 % 16
  let m := byte.val % 8
  let sign : ℚ := if byte.val < 128 then 1 else -1
  if e = 15 ∧ m = 7 then none
  else some (sign * (if e = 0 then (m : ℚ) / 512
    else (1 + (m : ℚ) / 8) *
      (if e < 7 then 1 / (2 : ℚ) ^ (7 - e) else (2 : ℚ) ^ (e - 7))))

def code (p : Bool) : Fin 256 := ⟨carrierByte p, by cases p <;> decide⟩

def signValue (p : Bool) : ℚ := if p then 1 else -1

theorem carrier_fields (p : Bool) :
    (code p).val / 8 % 16 = 7 ∧ (code p).val % 8 = 0 := by
  cases p <;> decide

theorem carrier_decode (p : Bool) : decodeE4M3 (code p) = some (signValue p) := by
  cases p <;> norm_num [decodeE4M3, code, carrierByte, signValue]

def expandedCode (n : Fin 16) (lane : Fin 4) : Fin 256 :=
  ⟨laneByte n lane, Nat.mod_lt _ (by decide)⟩

theorem expanded_code_eq (n : Fin 16) (lane : Fin 4) :
    expandedCode n lane = code (positive n lane) := by
  apply Fin.ext
  exact all_lane_codes n lane

theorem expanded_decode (n : Fin 16) (lane : Fin 4) :
    decodeE4M3 (expandedCode n lane) = some (signValue (positive n lane)) := by
  rw [expanded_code_eq, carrier_decode]

/-- A carrier value is supplied together with its actual decoder equation,
    so no NaN or invented decoder fallback enters a projection. -/
def DecodedLane (n : Fin 16) (lane : Fin 4) :=
  { q : ℚ // decodeE4M3 (expandedCode n lane) = some q }

theorem decoded_lane_value (n : Fin 16) (lane : Fin 4) (q : DecodedLane n lane) :
    q.val = signValue (positive n lane) := by
  exact (Option.some.inj ((expanded_decode n lane).symm.trans q.property)).symm

/-- Arbitrary finite weighted projection over the decoded activation carrier.
    `coeff` can include per-weight scales; `sA` is the activation scale. -/
theorem weighted_projection {ι : Type*} (s : Finset ι)
    (n : ι → Fin 16) (lane : ι → Fin 4)
    (w : ∀ i, DecodedLane (n i) (lane i))
    (coeff qDecoded : ι → ℝ) (sA : ℝ) :
    (∑ i ∈ s, coeff i * (w i).val * (qDecoded i * sA)) =
      ∑ i ∈ s, coeff i * (signValue (positive (n i) (lane i)) : ℝ) *
        (qDecoded i * sA) := by
  apply Finset.sum_congr rfl
  intro i _
  rw [decoded_lane_value]

/-- Real arithmetic identity; it does not assert that quantization is exact. -/
theorem quantization_error {ι : Type*} (s : Finset ι)
    (w qDecoded x : ι → ℝ) (sA : ℝ) :
    (∑ i ∈ s, w i * (qDecoded i * sA)) - (∑ i ∈ s, w i * x i) =
      ∑ i ∈ s, w i * (qDecoded i * sA - x i) := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

#print axioms all_lane_codes
#print axioms carrier_decode
#print axioms expanded_decode
#print axioms weighted_projection
#print axioms quantization_error

end BinaryFP8
