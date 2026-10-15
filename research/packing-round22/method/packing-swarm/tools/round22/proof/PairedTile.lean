import StridedPrep
namespace Packing22
open Packing20 Packing21

inductive Arm where
  | candidate | control
  deriving DecidableEq

def consumerWarps : Arm → Nat | .candidate => 4 | .control => 8
def groupsPerWarp : Arm → Nat | .candidate => 2 | .control => 1
def group (a : Arm) (warp q : Nat) := warp * groupsPerWarp a + q

theorem equal_complete_tile (a : Arm) : consumerWarps a * groupsPerWarp a = 8 := by
  cases a <;> decide

theorem group_bound (a : Arm) (warp q : Nat)
    (hw : warp < consumerWarps a) (hq : q < groupsPerWarp a) : group a warp q < 8 := by
  cases a <;> simp [group, consumerWarps, groupsPerWarp] at * <;> omega

theorem group_reconstruction (a : Arm) (g : Nat) (hg : g < 8) :
    g / groupsPerWarp a < consumerWarps a ∧ g % groupsPerWarp a < groupsPerWarp a ∧
      group a (g / groupsPerWarp a) (g % groupsPerWarp a) = g := by
  cases a <;> simp [group, consumerWarps, groupsPerWarp] <;> omega

theorem group_unique (a : Arm) (w q w' q' : Nat)
    (hq : q < groupsPerWarp a) (hq' : q' < groupsPerWarp a)
    (same : group a w q = group a w' q') : w = w' ∧ q = q' := by
  cases a <;> simp [group, groupsPerWarp] at * <;> omega

def feature (g mm r gid : Nat) := g * 32 + mm * 16 + r * 8 + gid
def token (n tq j : Nat) := n * 8 + tq * 2 + j

theorem feature_bound (g mm r gid : Nat) (hg : g < 8) (hm : mm < 2)
    (hr : r < 2) (hi : gid < 8) : feature g mm r gid < 256 := by
  unfold feature; omega

theorem feature_unique (g mm r gid g' mm' r' gid' : Nat)
    (hm : mm < 2) (hr : r < 2) (hi : gid < 8)
    (hm' : mm' < 2) (hr' : r' < 2) (hi' : gid' < 8)
    (same : feature g mm r gid = feature g' mm' r' gid') :
    g = g' ∧ mm = mm' ∧ r = r' ∧ gid = gid' := by
  unfold feature at same; omega

theorem feature_reconstruction (f : Nat) :
    f % 32 / 16 < 2 ∧ f % 16 / 8 < 2 ∧ f % 8 < 8 ∧
    feature (f / 32) (f % 32 / 16) (f % 16 / 8) (f % 8) = f := by
  unfold feature; omega

theorem token_bound (n tq j : Nat) (hn : n < 4) (ht : tq < 4) (hj : j < 2) :
    token n tq j < 32 := by unfold token; omega

theorem token_unique (n tq j n' tq' j' : Nat) (ht : tq < 4) (hj : j < 2)
    (ht' : tq' < 4) (hj' : j' < 2) (same : token n tq j = token n' tq' j') :
    n = n' ∧ tq = tq' ∧ j = j' := by unfold token at same; omega

theorem token_reconstruction (t : Nat) :
    t % 8 / 2 < 4 ∧ t % 2 < 2 ∧ token (t / 8) (t % 8 / 2) (t % 2) = t := by
  unfold token; omega

theorem lane_dictionary (lane : Fin 32) :
    lane.val / 4 < 8 ∧ lane.val % 4 < 4 ∧ (lane.val / 4) * 4 + lane.val % 4 = lane.val := by
  have := lane.isLt; omega

theorem literal_m_to_group_feature (gw q mm r gid : Nat) :
    gw * 32 + gid + (q * 2 + mm) * 16 + r * 8 = feature (gw + q) mm r gid := by
  unfold feature; omega

def outputAddress (stride t f : Nat) := t * stride + f

theorem output_address_unique (stride t f t' f' : Nat) (hf : f < stride) (hf' : f' < stride)
    (same : outputAddress stride t f = outputAddress stride t' f') : t = t' ∧ f = f' := by
  unfold outputAddress at same
  have h := compact_index_div_mod t f stride hf
  have h' := compact_index_div_mod t' f' stride hf'
  have hd := congrArg (fun x => x / stride) same
  have hm := congrArg (fun x => x % stride) same
  rw [h.1, h'.1] at hd
  rw [h.2, h'.2] at hm
  exact ⟨hd, hm⟩

theorem whole_feature_tile_reconstruction (F f : Nat) (hf : f < F) :
    f / 256 * 8 + f % 256 / 32 = f / 32 ∧
      feature (f / 256 * 8 + f % 256 / 32) (f % 32 / 16) (f % 16 / 8) (f % 8) = f ∧
      f / 32 < (F + 31) / 32 := by
  have := feature_reconstruction f
  unfold feature at *; omega

theorem whole_token_tile_reconstruction (B t : Nat) (_ht : t < B) :
    t / 32 * 32 + token (t % 32 / 8) (t % 8 / 2) (t % 2) = t := by
  unfold token; omega

theorem complete_reader_coordinate_reconstruction (a : Arm) (f t : Nat) :
    ∃ w q mm r gid n tq j : Nat,
      w < consumerWarps a ∧ q < groupsPerWarp a ∧ mm < 2 ∧ r < 2 ∧ gid < 8 ∧
      n < 4 ∧ tq < 4 ∧ j < 2 ∧
      feature (f / 256 * 8 + group a w q) mm r gid = f ∧
      t / 32 * 32 + token n tq j = t := by
  have hg : f % 256 / 32 < 8 := by omega
  have h := group_reconstruction a (f % 256 / 32) hg
  refine ⟨(f % 256 / 32) / groupsPerWarp a, (f % 256 / 32) % groupsPerWarp a,
    f % 32 / 16, f % 16 / 8, f % 8, t % 32 / 8, t % 8 / 2, t % 2,
    h.1, h.2.1, ?_⟩
  rw [h.2.2]
  unfold feature token
  omega

theorem concrete_whole_shapes :
    16480 / 32 = 515 ∧ 515 / 8 = 64 ∧ 515 % 8 = 3 ∧
    14336 / 32 = 448 ∧ 448 % 8 = 0 ∧ 34816 / 32 = 1088 ∧ 1088 % 8 = 0 := by decide

theorem merged_gdn_tail (g : Nat) (hg : g < 8) : 512 + g < 515 ↔ g < 3 := by omega

theorem concrete_batch_tiles :
    (1 + 31) / 32 = 1 ∧ (8 + 31) / 32 = 1 ∧ (32 + 31) / 32 = 1 := by decide

theorem full_stride_not_tail_stride (t f : Nat) :
    outputAddress 16480 t (512 * 32 + f) = t * 16480 + 16384 + f := by
  unfold outputAddress; omega

inductive Segment where
  | weight | weightScale | primaryQ | primaryScale | innovationQ | innovationScale
  deriving DecidableEq

def segmentOffset : Segment → Nat
  | .weight => 0 | .weightScale => 4096 | .primaryQ => 4608
  | .primaryScale => 6656 | .innovationQ => 6912 | .innovationScale => 8960

def segmentBytes : Segment → Nat
  | .weight => 4096 | .weightScale => 512 | .primaryQ => 2048
  | .primaryScale => 256 | .innovationQ => 2048 | .innovationScale => 256

def sharedAddress (s : Nat) (seg : Segment) (i : Nat) := s * 9216 + segmentOffset seg + i

theorem shared_geometry :
    8 * 512 = 4096 ∧ 8 * 64 = 512 ∧ 32 * 64 = 2048 ∧ 32 * 8 = 256 ∧
    4096 + 512 + 2 * (2048 + 256) = 9216 ∧ 8 * 9216 + 2 * 8 * 8 = 73856 := by decide

def barrierAddress (empty : Bool) (s : Nat) := 73728 + (if empty then 64 else 0) + s * 8
def globalScaleAddress (pass t : Nat) := 73856 + pass * 128 + t * 4

def globalScaleLoads (t : Fin 32) : List (Bool × Fin 32) := [(false, t), (true, t)]

def stagedGlobal {F : Type} (zero : F) (B : Nat) (source : Bool → Nat → F) (pass : Bool) (t : Nat) :=
  if t < B then source pass t else zero

theorem globals_once_per_cta (t : Fin 32) (pass : Bool) :
    (globalScaleLoads t).count (pass, t) = 1 := by cases pass <;> simp [globalScaleLoads]

theorem live_global_scale_dictionary {F : Type} (zero : F) (B : Nat)
    (source : Bool → Nat → F) (pass : Bool) (t : Nat) (ht : t < B) :
    stagedGlobal zero B source pass t = source pass t := by simp [stagedGlobal, ht]

theorem padded_global_scale_defined_zero {F : Type} (zero : F) (B : Nat)
    (source : Bool → Nat → F) (pass : Bool) (t : Nat) (ht : B ≤ t) :
    stagedGlobal zero B source pass t = zero := by simp [stagedGlobal, show ¬ t < B by omega]

theorem actual_shared_allocation : 73856 + 2 * 32 * 4 = 74112 := by decide

theorem barrier_dictionary_unique (empty empty' : Bool) (s s' : Nat)
    (hs : s < 8) (hs' : s' < 8)
    (same : barrierAddress empty s = barrierAddress empty' s') : empty = empty' ∧ s = s' := by
  cases empty <;> cases empty' <;> simp [barrierAddress] at * <;> omega

theorem globals_dictionary_unique (pass t pass' t' : Nat) (_hp : pass < 2) (ht : t < 32)
    (_hp' : pass' < 2) (ht' : t' < 32)
    (same : globalScaleAddress pass t = globalScaleAddress pass' t') : pass = pass' ∧ t = t' := by
  unfold globalScaleAddress at same; omega

theorem global_scale_bounds (pass t byte : Nat) (hp : pass < 2) (ht : t < 32) (hb : byte < 4) :
    73856 ≤ globalScaleAddress pass t + byte ∧ globalScaleAddress pass t + byte < 74112 := by
  unfold globalScaleAddress; omega

theorem shared_segment_bound (seg : Segment) (i : Nat) (hi : i < segmentBytes seg) :
    segmentOffset seg + i < 9216 := by cases seg <;> simp [segmentOffset, segmentBytes] at * <;> omega

theorem shared_segments_disjoint (seg seg' : Segment) (i i' : Nat)
    (hi : i < segmentBytes seg) (hi' : i' < segmentBytes seg')
    (same : segmentOffset seg + i = segmentOffset seg' + i') : seg = seg' ∧ i = i' := by
  cases seg <;> cases seg' <;> simp [segmentOffset, segmentBytes] at * <;> omega

theorem shared_stage_unique (s s' : Nat) (seg seg' : Segment) (i i' : Nat)
    (hi : i < segmentBytes seg) (hi' : i' < segmentBytes seg')
    (same : sharedAddress s seg i = sharedAddress s' seg' i') : s = s' ∧ seg = seg' ∧ i = i' := by
  have hb := shared_segment_bound seg i hi
  have hb' := shared_segment_bound seg' i' hi'
  have hs : outputAddress 9216 s (segmentOffset seg + i) =
      outputAddress 9216 s' (segmentOffset seg' + i') := by
    simpa only [outputAddress, sharedAddress, Nat.add_assoc] using same
  have ha := output_address_unique 9216 s (segmentOffset seg + i) s' (segmentOffset seg' + i') hb hb' hs
  have hh := shared_segments_disjoint seg seg' i i' hi hi' ha.2
  exact ⟨ha.1, hh.1, hh.2⟩

theorem shared_byte_reconstruction (i : Nat) (hi : i < 9216) :
    ∃ seg : Segment, ∃ j : Nat, j < segmentBytes seg ∧ segmentOffset seg + j = i := by
  by_cases h0 : i < 4096
  · exact ⟨.weight, i, h0, by simp [segmentOffset]⟩
  by_cases h1 : i < 4608
  · exact ⟨.weightScale, i - 4096, by simp [segmentBytes]; omega, by simp [segmentOffset]; omega⟩
  by_cases h2 : i < 6656
  · exact ⟨.primaryQ, i - 4608, by simp [segmentBytes]; omega, by simp [segmentOffset]; omega⟩
  by_cases h3 : i < 6912
  · exact ⟨.primaryScale, i - 6656, by simp [segmentBytes]; omega, by simp [segmentOffset]; omega⟩
  by_cases h4 : i < 8960
  · exact ⟨.innovationQ, i - 6912, by simp [segmentBytes]; omega, by simp [segmentOffset]; omega⟩
  exact ⟨.innovationScale, i - 8960, by simp [segmentBytes]; omega, by simp [segmentOffset]; omega⟩

theorem ring_barrier_globals_disjoint (s : Nat) (seg : Segment) (i : Nat)
    (hs : s < 8) (hi : i < segmentBytes seg) (empty : Bool) (b : Nat) (hb : b < 8)
    (pass t : Nat) (_hp : pass < 2) (_ht : t < 32) :
    sharedAddress s seg i < barrierAddress empty b ∧
      barrierAddress empty b + 8 ≤ globalScaleAddress pass t := by
  have := shared_segment_bound seg i hi
  cases empty <;> simp [sharedAddress, barrierAddress, globalScaleAddress] <;> omega

def swizzle (row chunk : Nat) := Nat.xor chunk ((row / 2) % 4)

theorem swizzle_complete_involution : ∀ row : Fin 32, ∀ chunk : Fin 4,
    swizzle row.val chunk.val < 4 ∧ swizzle row.val (swizzle row.val chunk.val) = chunk.val := by decide

theorem swizzled_row_chunk_unique (row row' : Fin 32) (chunk chunk' : Fin 4)
    (same : row.val * 64 + swizzle row.val chunk.val * 16 =
      row'.val * 64 + swizzle row'.val chunk'.val * 16) : row = row' ∧ chunk = chunk' := by
  have h := swizzle_complete_involution row chunk
  have h' := swizzle_complete_involution row' chunk'
  have hr : row = row' := by apply Fin.ext; omega
  subst row'
  have hc : swizzle row.val chunk.val = swizzle row.val chunk'.val := by omega
  have hh := congrArg (swizzle row.val) hc
  rw [h.2, h'.2] at hh
  exact ⟨rfl, Fin.ext hh⟩

def weightByte (g lane word byte : Nat) := g * 512 + lane * 16 + word * 4 + byte
def weightScaleByte (g gid mm r byte : Nat) := 4096 + g * 64 + gid * 8 + (mm * 2 + r) * 2 + byte

theorem binary_weight_byte_bound (g lane word byte : Nat)
    (hg : g < 8) (hl : lane < 32) (hw : word < 4) (hb : byte < 4) :
    weightByte g lane word byte < 4096 := by unfold weightByte; omega

theorem binary_weight_byte_unique (g lane word byte g' lane' word' byte' : Nat)
    (hl : lane < 32) (hw : word < 4) (hb : byte < 4)
    (hl' : lane' < 32) (hw' : word' < 4) (hb' : byte' < 4)
    (same : weightByte g lane word byte = weightByte g' lane' word' byte') :
    g = g' ∧ lane = lane' ∧ word = word' ∧ byte = byte' := by unfold weightByte at same; omega

theorem binary_weight_byte_reconstruction (i : Nat) :
    weightByte (i / 512) (i % 512 / 16) (i % 16 / 4) (i % 4) = i := by
  unfold weightByte; omega

theorem original_w128_scale_address (g gid mm r byte : Nat) (hg : g < 8)
    (hi : gid < 8) (hm : mm < 2) (hr : r < 2) (hb : byte < 2) :
    4096 ≤ weightScaleByte g gid mm r byte ∧ weightScaleByte g gid mm r byte < 4608 := by
  unfold weightScaleByte; omega

theorem original_w128_scale_unique (g gid mm r byte g' gid' mm' r' byte' : Nat)
    (hi : gid < 8) (hm : mm < 2) (hr : r < 2) (hb : byte < 2)
    (hi' : gid' < 8) (hm' : mm' < 2) (hr' : r' < 2) (hb' : byte' < 2)
    (same : weightScaleByte g gid mm r byte = weightScaleByte g' gid' mm' r' byte') :
    g = g' ∧ gid = gid' ∧ mm = mm' ∧ r = r' ∧ byte = byte' := by
  unfold weightScaleByte at same; omega

def qTapeAddress (tile64 kb toff row chunk : Nat) :=
  ((tile64 * 40 + kb) * 64 + toff + row) * 64 + swizzle row chunk * 16

def scaleTapeAddress (tile64 kb toff row k16 : Nat) :=
  ((tile64 * 40 + kb) * 64 + toff + row) * 8 + k16

theorem same_padded_q_dictionary (_a : Arm) (tile64 kb toff row chunk : Nat) :
    qTapeAddress tile64 kb toff row chunk =
      ((tile64 * (5120 / 128) + kb) * 64 + toff + row) * 64 + swizzle row chunk * 16 := rfl

theorem token_half_preserves_swizzle (row : Nat) :
    ((32 + row) / 2) % 4 = (row / 2) % 4 := by omega

theorem padded_q_complete_rows (t : Nat) :
    (t / 32 * 32) / 64 * 64 + (t / 32 * 32) % 64 + t % 32 = t := by omega

-- One producer event per segment per K128; consumers do not appear in this domain.
structure LoadEvent where
  kb : Fin 40
  segment : Segment
  deriving DecidableEq

def producerLoads (kb : Fin 40) : List LoadEvent :=
  [.mk kb .weight, .mk kb .weightScale, .mk kb .primaryQ,
    .mk kb .primaryScale, .mk kb .innovationQ, .mk kb .innovationScale]

theorem staged_once_per_cta (kb : Fin 40) (seg : Segment) :
    (producerLoads kb).count (.mk kb seg) = 1 := by cases seg <;> simp [producerLoads]

theorem equal_boundary_producer_dictionary (_a : Arm) (kb : Fin 40) :
    (producerLoads kb).length = 6 := rfl

theorem tail_transaction_bytes : 3 * 512 + 3 * 64 + 2 * (32 * 64 + 32 * 8) = 6336 := by decide

theorem full_transaction_bytes : 8 * 512 + 8 * 64 + 2 * (32 * 64 + 32 * 8) = 9216 := by decide

def stageRead (memory : Nat → Word) (seg : Segment) (i : Nat) := memory (segmentOffset seg + i)
def stagedBytes (ng : Nat) : Segment → Nat
  | .weight => ng * 512
  | .weightScale => ng * 64
  | seg => segmentBytes seg

def stagedDictionary (source : Segment → Nat → Word) (memory : Nat → Word) (ng : Nat) : Prop :=
  ∀ seg i, i < stagedBytes ng seg → stageRead memory seg i = source seg i

theorem valid_staged_bytes_fit_capacity (ng : Nat) (hng : ng ≤ 8) (seg : Segment) :
    stagedBytes ng seg ≤ segmentBytes seg := by
  cases seg <;> simp [stagedBytes, segmentBytes] <;> omega

theorem tail_only_three_weight_groups_staged :
    stagedBytes 3 .weight = 1536 ∧ stagedBytes 3 .weightScale = 192 := by decide

theorem live_weight_group_staged (ng g lane word byte : Nat) (hg : g < ng)
    (hl : lane < 32) (hw : word < 4) (hb : byte < 4) :
    weightByte g lane word byte < stagedBytes ng .weight := by
  simp only [weightByte, stagedBytes]; omega

theorem every_staged_read_same_word (source : Segment → Nat → Word) (memory : Nat → Word)
    (ng : Nat) (contract : stagedDictionary source memory ng) (seg : Segment) (i : Nat)
    (hi : i < stagedBytes ng seg) : stageRead memory seg i = source seg i := contract seg i hi

structure FloatOps (F : Type) where
  zero : F
  mma : Word → Word → Word → F → F
  fma : F → Word → F → F
  mul : F → Word → F
  add : F → F → F

structure FoldPayload where
  weightLow : Word
  weightHigh : Word
  qLow : Word
  qHigh : Word
  qScaleLow : Word
  qScaleHigh : Word
  weightScale : Word

def orderedStep {F : Type} (ops : FloatOps F) (p : FoldPayload) (acc : F) : F :=
  ops.fma (ops.mma p.weightHigh p.qHigh p.qScaleHigh
    (ops.mma p.weightLow p.qLow p.qScaleLow ops.zero)) p.weightScale acc

def orderedFold {F : Type} (step : Nat → F → F) (zero : F) : Nat → F
  | 0 => zero
  | k + 1 => step k (orderedFold step zero k)

theorem ascending_fold_successor {F : Type} (step : Nat → F → F) (zero : F) (k : Nat) :
    orderedFold step zero (k + 1) = step k (orderedFold step zero k) := rfl

theorem forty_preserved_k128 : 5120 / 128 = 40 ∧ 40 * 128 = 5120 := by decide

theorem fold_dictionary_congruence {F : Type} (step step' : Nat → F → F) (zero : F)
    (k : Nat) (same : ∀ i, i < k → step i = step' i) :
    orderedFold step zero k = orderedFold step' zero k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [orderedFold]
    rw [ih (fun i hi => same i (by omega)), same k (by omega)]

theorem payload_dictionary_preserves_ordered_fold {F : Type} (ops : FloatOps F)
    (staged source : Nat → FoldPayload) (same : ∀ kb, kb < 40 → staged kb = source kb) :
    orderedFold (fun kb => orderedStep ops (staged kb)) ops.zero 40 =
      orderedFold (fun kb => orderedStep ops (source kb)) ops.zero 40 := by
  apply fold_dictionary_congruence
  intro kb hk
  rw [same kb hk]

def pairedFold {F : Type} (step0 step1 : Nat → F → F) (zero : F) : Nat → F × F
  | 0 => (zero, zero)
  | k + 1 => let p := pairedFold step0 step1 zero k; (step0 k p.1, step1 k p.2)

theorem paired_fold_two_independent_ordered_accumulators {F : Type}
    (step0 step1 : Nat → F → F) (zero : F) (k : Nat) :
    pairedFold step0 step1 zero k = (orderedFold step0 zero k, orderedFold step1 zero k) := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [pairedFold, ih, orderedFold]

def literalMode1 {F : Type} (ops : FloatOps F) (acc : F) (ag : Word) (prior : F) :=
  ops.add (ops.mul acc ag) prior

def pairedOutput {F : Type} (ops : FloatOps F) (acc0 acc1 : F) (ag0 ag1 : Word) (prior : F) : F × F :=
  let primary := ops.add (ops.mul acc0 ag0) prior
  (primary, ops.add (ops.mul acc1 ag1) primary)

theorem primary_then_innovation_literal_order {F : Type} (ops : FloatOps F)
    (acc0 acc1 : F) (ag0 ag1 : Word) (prior : F) :
    pairedOutput ops acc0 acc1 ag0 ag1 prior =
      (literalMode1 ops acc0 ag0 prior, literalMode1 ops acc1 ag1 (literalMode1 ops acc0 ag0 prior)) := rfl

theorem complete_paired_fold_and_output_dictionary {F : Type} (ops : FloatOps F)
    (s0 s1 c0 c1 : Nat → F → F) (ag0 ag1 : Word) (prior : F)
    (h0 : ∀ i, i < 40 → s0 i = c0 i) (h1 : ∀ i, i < 40 → s1 i = c1 i) :
    let acc := pairedFold s0 s1 ops.zero 40
    pairedOutput ops acc.1 acc.2 ag0 ag1 prior =
      (literalMode1 ops (orderedFold c0 ops.zero 40) ag0 prior,
       literalMode1 ops (orderedFold c1 ops.zero 40) ag1
         (literalMode1 ops (orderedFold c0 ops.zero 40) ag0 prior)) := by
  rw [paired_fold_two_independent_ordered_accumulators]
  dsimp only
  rw [fold_dictionary_congruence s0 c0 ops.zero 40 h0,
      fold_dictionary_congruence s1 c1 ops.zero 40 h1]
  rfl

inductive PassObservation where
  | primaryOutput | primaryStatus | primaryOwner
  | innovationOutput | innovationStatus | innovationOwner
  deriving DecidableEq

structure PairedCustody where
  borrow : BorrowCustody
  passObserved : PassObservation → Nat → Nat → Bool
  rows : Nat
  features : Nat
  manifest : List Nat
  drained : Nat → Bool

structure PairedRelease (c : PairedCustody) where
  backingRelease : BorrowRelease c.borrow
  exactRows : c.rows = c.borrow.identity.source.domain.rows
  completeShape : c.features = 16480 ∨ c.features = 14336 ∨ c.features = 34816
  complete : ∀ o t f, t < c.rows → f < c.features → c.passObserved o t f = true
  exactManifest : c.manifest = c.borrow.identity.manifest
  drains : ∀ entry ∈ c.manifest, c.drained entry = true

theorem paired_release_observation_domain_nonempty (c : PairedCustody) (r : PairedRelease c) :
    0 < c.rows ∧ 0 < c.features := by
  have hr := r.backingRelease.exact.1.1
  rw [← r.exactRows] at hr
  have hf := r.completeShape
  exact ⟨hr, by omega⟩

theorem paired_release_needs_every_complete_pass_observation (c : PairedCustody)
    (r : PairedRelease c) (o : PassObservation) (t f : Nat) (ht : t < c.rows) (hf : f < c.features) :
    c.passObserved o t f = true := r.complete o t f ht hf

theorem missing_pass_coordinate_cannot_release (c : PairedCustody) (o : PassObservation)
    (t f : Nat) (ht : t < c.rows) (hf : f < c.features)
    (missing : c.passObserved o t f = false) : ¬ Nonempty (PairedRelease c) := by
  rintro ⟨r⟩
  have := r.complete o t f ht hf
  rw [missing] at this
  contradiction

theorem undrained_manifest_entry_cannot_release (c : PairedCustody) (entry : Nat)
    (member : entry ∈ c.manifest) (pending : c.drained entry = false) : ¬ Nonempty (PairedRelease c) := by
  rintro ⟨r⟩
  have := r.drains entry member
  rw [pending] at this
  contradiction

theorem paired_release_requires_exact_whole_backing (c : PairedCustody) (r : PairedRelease c) :
    c.borrow.original.identity.range.offset = 0 ∧ 0 < c.borrow.original.identity.range.bytes ∧
    c.borrow.original.identity.range.bytes = c.borrow.original.identity.range.allocationBytes :=
  borrow_release_only_exact_whole_allocation c.borrow r.backingRelease

theorem paired_uncertain_release_retains_backing (c : PairedCustody) (r : PairedRelease c)
    (uncertain : r.backingRelease.wholeRelease.outcome = .uncertain) :
    r.backingRelease.wholeRelease.after.pointer = c.borrow.original.pointer :=
  (uncertain_borrow_retains_original_and_interior_pointer c.borrow r.backingRelease uncertain).1

end Packing22
