import Std

namespace BinaryWholeMap.ThreePlanes

-- Digit values are twice the native E2M1 values, not IEEE int4 labels.
def alphabet : List Int := [0, 1, 2, 3, 4, 6, 8, 12, -1, -2, -3, -4, -6, -8, -12]
def signedByte (q : Fin 256) : Int := if q.val < 128 then q.val else (q.val : Int) - 256

def digits (q : Int) : Int × Int × Int :=
  let a := q % 4
  let b := q / 4 % 4
  let c := q / 16
  if c.natAbs = 5 ∨ c.natAbs = 7 then
    let δ : Int := if b % 2 = 1 then 1 else -1
    (a, b - 4 * δ, c + δ)
  else (a, b, c)

def magnitudeCode (q : Int) : Nat :=
  let m := q.natAbs
  if m ≤ 4 then m else m / 2 + 2

def fp4Code (q : Int) : Nat := magnitudeCode q + (if q < 0 then 8 else 0)
def twiceFp4 (code : Nat) : Int :=
  let magnitude : Int := match code % 8 with
    | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3
    | 4 => 4 | 5 => 6 | 6 => 8 | _ => 12
  if code ≥ 8 then -magnitude else magnitude

def recover (d : Int × Int × Int) : Int := d.1 + 4 * d.2.1 + 16 * d.2.2

def valid (d : Int × Int × Int) : Prop :=
  d.1 ∈ alphabet ∧ d.2.1 ∈ alphabet ∧ d.2.2 ∈ alphabet

set_option maxRecDepth 4096 in
set_option maxHeartbeats 4000000 in
theorem all_signed_bytes : ∀ q : Fin 256,
    recover (digits (signedByte q)) = signedByte q ∧ valid (digits (signedByte q)) := by
  unfold valid
  decide

set_option maxRecDepth 4096 in
set_option maxHeartbeats 4000000 in
theorem native_labels : ∀ q : Fin 256,
    let d := digits (signedByte q)
    fp4Code d.1 < 16 ∧ fp4Code d.2.1 < 16 ∧ fp4Code d.2.2 < 16 ∧
    twiceFp4 (fp4Code d.1) = d.1 ∧
    twiceFp4 (fp4Code d.2.1) = d.2.1 ∧
    twiceFp4 (fp4Code d.2.2) = d.2.2 := by
  decide

theorem whole_observer_transport {O : Type} (observe : Int → O) (q : Fin 256) :
    observe (recover (digits (signedByte q))) = observe (signedByte q) := by
  rw [(all_signed_bytes q).1]

-- Group scales, native accumulation and every actual suffix may be included in
-- observe. This theorem prices none of them and does not certify MMA rounding.
#print axioms all_signed_bytes
#print axioms native_labels
#print axioms whole_observer_transport
end BinaryWholeMap.ThreePlanes
