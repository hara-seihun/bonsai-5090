import Lean.Elab.Tactic.BVDecide
namespace Cooperative
-- Wiring contract only; native load/shuffle/MMA semantics remain empirical.
def segment (s : BitVec 320) (i : Nat) : BitVec 80 :=
  (s >>> (80*i)).truncate 80

def funnel (a b : BitVec 32) (n : Nat) : BitVec 32 :=
  ((a.zeroExtend 64 ||| (b.zeroExtend 64 <<< 32)) >>> n).truncate 32

def gather (s : BitVec 320) (nib : Nat) : BitVec 32 :=
  let owner := nib / 20
  let ofs := (nib % 20)*4
  let cur := segment s owner
  let l := cur.truncate 32
  let h := (cur >>> 32).truncate 32
  let top := (cur >>> 64).zeroExtend 32
  let next := (segment s ((owner+1)%4)).truncate 32
  let out := if ofs < 32 then funnel l h ofs
    else if ofs < 64 then funnel h top (ofs-32)
    else top >>> (ofs-64)
  if ofs > 48 then out ||| (next <<< (80-ofs)) else out

def expected (s : BitVec 320) (nib : Nat) : BitVec 32 :=
  (s >>> (4*nib)).truncate 32

def allMatch (s : BitVec 320) : Bool :=
  (gather s 0 == expected s 0) &&
  (gather s 1 == expected s 1) &&
  (gather s 2 == expected s 2) &&
  (gather s 3 == expected s 3) &&
  (gather s 4 == expected s 4) &&
  (gather s 5 == expected s 5) &&
  (gather s 6 == expected s 6) &&
  (gather s 7 == expected s 7) &&
  (gather s 8 == expected s 8) &&
  (gather s 9 == expected s 9) &&
  (gather s 10 == expected s 10) &&
  (gather s 11 == expected s 11) &&
  (gather s 12 == expected s 12) &&
  (gather s 13 == expected s 13) &&
  (gather s 14 == expected s 14) &&
  (gather s 15 == expected s 15) &&
  (gather s 16 == expected s 16) &&
  (gather s 17 == expected s 17) &&
  (gather s 18 == expected s 18) &&
  (gather s 19 == expected s 19) &&
  (gather s 20 == expected s 20) &&
  (gather s 21 == expected s 21) &&
  (gather s 22 == expected s 22) &&
  (gather s 23 == expected s 23) &&
  (gather s 24 == expected s 24) &&
  (gather s 25 == expected s 25) &&
  (gather s 26 == expected s 26) &&
  (gather s 27 == expected s 27) &&
  (gather s 28 == expected s 28) &&
  (gather s 29 == expected s 29) &&
  (gather s 30 == expected s 30) &&
  (gather s 31 == expected s 31) &&
  (gather s 32 == expected s 32) &&
  (gather s 33 == expected s 33) &&
  (gather s 34 == expected s 34) &&
  (gather s 35 == expected s 35) &&
  (gather s 36 == expected s 36) &&
  (gather s 37 == expected s 37) &&
  (gather s 38 == expected s 38) &&
  (gather s 39 == expected s 39) &&
  (gather s 40 == expected s 40) &&
  (gather s 41 == expected s 41) &&
  (gather s 42 == expected s 42) &&
  (gather s 43 == expected s 43) &&
  (gather s 44 == expected s 44) &&
  (gather s 45 == expected s 45) &&
  (gather s 46 == expected s 46) &&
  (gather s 47 == expected s 47) &&
  (gather s 48 == expected s 48) &&
  (gather s 49 == expected s 49) &&
  (gather s 50 == expected s 50) &&
  (gather s 51 == expected s 51) &&
  (gather s 52 == expected s 52) &&
  (gather s 53 == expected s 53) &&
  (gather s 54 == expected s 54) &&
  (gather s 55 == expected s 55) &&
  (gather s 56 == expected s 56) &&
  (gather s 57 == expected s 57) &&
  (gather s 58 == expected s 58) &&
  (gather s 59 == expected s 59) &&
  (gather s 60 == expected s 60)

theorem all_positions (s : BitVec 320) : allMatch s = true := by
  unfold allMatch gather expected segment funnel
  dsimp
  bv_decide

def fixed (s : BitVec 320) (t : Nat) (second : Bool) : BitVec 32 :=
  let owner := if second then (if t = 0 then 1 else 2) else (if t = 3 then 1 else 0)
  let cur := segment s owner
  let l := cur.truncate 32
  let h := (cur >>> 32).truncate 32
  let top := (cur >>> 64).truncate 32
  let next := (segment s (owner+1)).truncate 32
  if second then
    if t = 0 then (h >>> 16) ||| (top <<< 16)
    else if t = 1 then l else if t = 2 then h else top ||| (next <<< 16)
  else
    if t = 0 then l else if t = 1 then h else if t = 2 then top ||| (next <<< 16)
    else (l >>> 16) ||| (h <<< 16)

def fixedMatch (s : BitVec 320) : Bool :=
  (fixed s 0 false == expected s 0) &&
  (fixed s 1 false == expected s 8) &&
  (fixed s 2 false == expected s 16) &&
  (fixed s 3 false == expected s 24) &&
  (fixed s 0 true == expected s 32) &&
  (fixed s 1 true == expected s 40) &&
  (fixed s 2 true == expected s 48) &&
  (fixed s 3 true == expected s 56)

theorem fixed_lowering (s : BitVec 320) : fixedMatch s = true := by
  unfold fixedMatch fixed expected segment
  dsimp
  bv_decide

end Cooperative
