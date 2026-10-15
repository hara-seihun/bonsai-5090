import Std
namespace Packing20

abbrev Word := UInt32
abbrev Tensor (rows features : Nat) := Fin rows → Fin features → Word
inductive Region where
  | input3 | ffn2
  deriving DecidableEq
structure Domain where
  layer : Nat
  generation : Nat
  boundary : Nat
  rows : Nat
  token : Nat
  stream : Nat
  region : Region
  normIdentity : Nat
  tensors : List Nat
  deriving DecidableEq
structure SourceImage where
  domain : Domain
  x : List Word
  norm : List Word
  deriving DecidableEq
def ValidSource (s : SourceImage) : Prop :=
  0 < s.domain.rows ∧ s.x.length = s.domain.rows * 5120 ∧ s.norm.length = 5120

structure Owner where
  layer : Nat
  generation : Nat
  boundary : Nat
  deriving DecidableEq
structure ByteRange where
  offset : Nat
  bytes : Nat
  allocationBytes : Nat
  deriving DecidableEq
structure HandleIdentity where
  ticket : Nat
  owner : Owner
  range : ByteRange
  native : Nat
  deriving DecidableEq
inductive Life where
  | live | failedRelease | uncertainRelease | released
  deriving DecidableEq
inductive Outcome where
  | success | failure | uncertain
  deriving DecidableEq
structure Handle where
  identity : HandleIdentity
  validation : Bool
  life : Life
  pointer : Option Nat
  deriving DecidableEq
structure CompletedLayer where
  owner : Owner
  stateImage : List UInt8
  sourceImage : List UInt8
  sourceDrained : Bool
  readersDrained : List Bool
  sealed : Bool

def Dead (c : CompletedLayer) (h : Handle) : Prop :=
  h.validation = true ∧ c.owner = h.identity.owner ∧
  c.sealed = true ∧ c.stateImage ≠ [] ∧ c.sourceImage ≠ [] ∧
  c.sourceDrained = true ∧ (∀ r ∈ c.readersDrained, r = true) ∧
  h.identity.range.offset = 0 ∧ 0 < h.identity.range.bytes ∧
  h.identity.range.bytes = h.identity.range.allocationBytes

def finishRelease (h : Handle) (outcome : Outcome) : Handle :=
  match outcome with
  | .success => { h with life := .released, pointer := none }
  | .failure => { h with life := .failedRelease }
  | .uncertain => { h with life := .uncertainRelease }

def reconcile (h : Handle) (outcome : Outcome) : Handle :=
  match outcome with
  | .success => { h with life := .released, pointer := none }
  | .failure => { h with life := .live }
  | .uncertain => h

structure ReleaseStep (c : CompletedLayer) (before : Handle) where
  outcome : Outcome
  dead : Dead c before
  live : before.life = .live
  after : Handle
  exactTransition : after = finishRelease before outcome

theorem admitted_release_needs_all_drains (c : CompletedLayer) (h : Handle)
    (step : ReleaseStep c h) :
    c.sourceDrained = true ∧ ∀ r ∈ c.readersDrained, r = true :=
  ⟨step.dead.2.2.2.2.2.1, step.dead.2.2.2.2.2.2.1⟩

theorem persistent_has_no_release_step (c : CompletedLayer) (h : Handle)
    (persistent : h.validation = false) : ¬ Nonempty (ReleaseStep c h) := by
  rintro ⟨step⟩
  have clash := step.dead.1
  rw [persistent] at clash
  contradiction

theorem admitted_release_retains_exact_identity (c : CompletedLayer) (h : Handle)
    (step : ReleaseStep c h) : step.after.identity = h.identity := by
  rw [step.exactTransition]
  cases step.outcome <;> rfl

theorem unreconciled_cannot_release (c : CompletedLayer) (h : Handle)
    (notLive : h.life ≠ .live) : ¬ Nonempty (ReleaseStep c h) := by
  rintro ⟨step⟩
  exact notLive step.live

theorem release_identity (h : Handle) (o : Outcome) :
    (finishRelease h o).identity = h.identity := by cases o <;> rfl

theorem failed_pointer_ticket_domain (h : Handle) :
    (finishRelease h .failure).pointer = h.pointer ∧
    (finishRelease h .failure).identity = h.identity := ⟨rfl, rfl⟩

theorem uncertain_pointer_ticket_domain (h : Handle) :
    (finishRelease h .uncertain).pointer = h.pointer ∧
    (finishRelease h .uncertain).identity = h.identity := ⟨rfl, rfl⟩

theorem successful_pointer_erased (h : Handle) :
    (finishRelease h .success).pointer = none ∧
    (finishRelease h .success).life = .released := ⟨rfl, rfl⟩

theorem persistent_not_dead (c : CompletedLayer) (h : Handle)
    (persistent : h.validation = false) : ¬ Dead c h := by
  intro dead
  have clash := dead.1
  rw [persistent] at clash
  contradiction

theorem dead_has_completed_images (c : CompletedLayer) (h : Handle) (d : Dead c h) :
    c.sealed = true ∧ c.stateImage ≠ [] ∧ c.sourceImage ≠ [] :=
  ⟨d.2.2.1, d.2.2.2.1, d.2.2.2.2.1⟩

theorem dead_all_dependencies_drained (c : CompletedLayer) (h : Handle) (d : Dead c h) :
    c.sourceDrained = true ∧ ∀ r ∈ c.readersDrained, r = true :=
  ⟨d.2.2.2.2.2.1, d.2.2.2.2.2.2.1⟩

theorem dead_releases_whole_allocation (c : CompletedLayer) (h : Handle) (d : Dead c h) :
    h.identity.range.offset = 0 ∧ 0 < h.identity.range.bytes ∧
    h.identity.range.bytes = h.identity.range.allocationBytes := d.2.2.2.2.2.2.2

theorem reconcile_identity (h : Handle) (o : Outcome) :
    (reconcile h o).identity = h.identity := by cases o <;> rfl

theorem reconcile_failed_keeps_pointer (h : Handle) :
    (reconcile h .failure).pointer = h.pointer := rfl

theorem bounded_step (performed limit : Nat) (bounded : performed ≤ limit)
    (limitBound : limit ≤ 256) : performed ≤ 256 := Nat.le_trans bounded limitBound

-- All primitives below are opaque word functions, including SiluMul and residual.
def slice {B F n : Nat} (whole : Tensor B F) (offset : Nat)
    (fits : offset + n ≤ F) : Tensor B n :=
  fun row i => whole row ⟨offset + i.val, by have := i.isLt; omega⟩

def concat3 {B q g a : Nat} (qkv : Tensor B q) (gate : Tensor B g)
    (ab : Tensor B a) : Tensor B (q + g + a) :=
  fun row i =>
    if hq : i.val < q then qkv row ⟨i.val, hq⟩
    else if hg : i.val < q + g then gate row ⟨i.val - q, by omega⟩
    else ab row ⟨i.val - (q + g), by have := i.isLt; omega⟩

theorem concat3_original_slices {B q g a : Nat} (merged : Tensor B (q + g + a)) :
    concat3 (slice merged 0 (by omega))
      (slice merged q (by omega)) (slice merged (q + g) (by omega)) = merged := by
  funext row i
  simp only [concat3]
  split
  · simp only [slice, Nat.zero_add]
  · split
    · simp only [slice]
      congr 1
      apply Fin.ext
      change q + (i.val - q) = i.val
      omega
    · simp only [slice]
      congr 1
      apply Fin.ext
      change q + g + (i.val - (q + g)) = i.val
      omega

theorem input3_merged_continuation {B : Nat} {Result : Type}
    (merged : Tensor B (10240 + 6144 + 96))
    (qkv : Tensor B 10240) (gate : Tensor B 6144) (ab : Tensor B 96)
    (hq : qkv = slice merged 0 (by omega))
    (hg : gate = slice merged 10240 (by omega))
    (ha : ab = slice merged (10240 + 6144) (by omega))
    (original : Tensor B (10240 + 6144 + 96) → Result) :
    original (concat3 qkv gate ab) = original merged := by
  rw [hq, hg, ha, concat3_original_slices]

def concat2 {B g u : Nat} (gate : Tensor B g) (up : Tensor B u) : Tensor B (g + u) :=
  fun row i => if h : i.val < g then gate row ⟨i.val, h⟩
    else up row ⟨i.val - g, by have := i.isLt; omega⟩

theorem concat2_original_slices {B g u : Nat} (merged : Tensor B (g + u)) :
    concat2 (slice merged 0 (by omega)) (slice merged g (by omega)) = merged := by
  funext row i
  simp only [concat2]
  split
  · simp only [slice, Nat.zero_add]
  · simp only [slice]
    congr 1
    apply Fin.ext
    change g + (i.val - g) = i.val
    omega

theorem ffn2_original_silu_down_residual {B : Nat} {Hidden Down Output : Type}
    (merged : Tensor B (17408 + 17408)) (gate up : Tensor B 17408)
    (hg : gate = slice merged 0 (by omega))
    (hu : up = slice merged 17408 (by omega))
    (siluMul : Tensor B (17408 + 17408) → Hidden)
    (down : Hidden → Down) (residual : Down → Output) :
    residual (down (siluMul (concat2 gate up))) = residual (down (siluMul merged)) := by
  rw [hg, hu, concat2_original_slices]

theorem attention3_merged_continuation {B : Nat} {Result : Type}
    (merged : Tensor B (12288 + 1024 + 1024))
    (q : Tensor B 12288) (k v : Tensor B 1024)
    (hq : q = slice merged 0 (by omega))
    (hk : k = slice merged 12288 (by omega))
    (hv : v = slice merged (12288 + 1024) (by omega))
    (original : Tensor B (12288 + 1024 + 1024) → Result) :
    original (concat3 q k v) = original merged := by
  rw [hq, hk, hv, concat3_original_slices]

theorem immutable_source_domain (before after : SourceImage) (preserved : after = before) :
    after.domain = before.domain ∧ after.x = before.x ∧ after.norm = before.norm := by
  subst after
  exact ⟨rfl, rfl, rfl⟩

theorem full_source_shape_preserved (before after : SourceImage)
    (preserved : after = before) (valid : ValidSource before) : ValidSource after := by
  rw [preserved]
  exact valid

theorem shared_source_observers {Tape Output : Type}
    (producer : SourceImage → Tape) (observer : Tape → Output)
    (actual original : SourceImage) (sameImage : actual = original) :
    observer (producer actual) = observer (producer original) := by rw [sameImage]

theorem projections_preserve_old_source {Projection : Type}
    (old : SourceImage) (project : SourceImage → Projection) :
    (old, project old).1 = old := rfl

structure RowView where
  rows : Nat
  fullFeatures : Nat
  offsetFeatures : Nat
  width : Nat
  deriving DecidableEq

def RowView.strideBytes (v : RowView) := 4 * v.fullFeatures
def RowView.offsetBytes (v : RowView) := 4 * v.offsetFeatures
def RowView.allocationBytes (v : RowView) := 4 * (v.rows * v.fullFeatures)
def RowView.address (v : RowView) (row col : Nat) :=
  4 * (row * v.fullFeatures + v.offsetFeatures + col)
def ValidView (v : RowView) : Prop :=
  0 < v.rows ∧ 0 < v.width ∧ v.offsetFeatures + v.width ≤ v.fullFeatures

theorem view_address_inside_whole_backing (v : RowView) (valid : ValidView v)
    (row col : Nat) (hr : row < v.rows) (hc : col < v.width) :
    v.address row col + 4 ≤ v.allocationBytes := by
  have fits := valid.2.2
  have hi : v.offsetFeatures + col + 1 ≤ v.fullFeatures := by omega
  have hm : (row + 1) * v.fullFeatures ≤ v.rows * v.fullFeatures :=
    Nat.mul_le_mul_right v.fullFeatures (by omega)
  have he : row * v.fullFeatures + v.offsetFeatures + col + 1 ≤
      v.rows * v.fullFeatures := by
    rw [Nat.add_mul, Nat.one_mul] at hm
    omega
  unfold RowView.address RowView.allocationBytes
  omega

theorem view_address_matches_full_row (v : RowView) (row col : Nat) :
    v.address row col = row * v.strideBytes + v.offsetBytes + 4 * col := by
  simp only [RowView.address, RowView.strideBytes, RowView.offsetBytes, Nat.mul_add]
  rw [Nat.mul_left_comm 4 row v.fullFeatures]

def rowTensor (memory : Nat → Word) (B F : Nat) : Tensor B F :=
  fun row col => memory (row.val * F + col.val)
def viewTensor (memory : Nat → Word) (v : RowView) : Tensor v.rows v.width :=
  fun row col => memory (row.val * v.fullFeatures + v.offsetFeatures + col.val)

theorem native_view_is_full_row_slice (memory : Nat → Word) (v : RowView)
    (fits : v.offsetFeatures + v.width ≤ v.fullFeatures) :
    viewTensor memory v = slice (rowTensor memory v.rows v.fullFeatures) v.offsetFeatures fits := by
  funext row col
  simp only [viewTensor, slice, rowTensor, Nat.add_assoc]

theorem byte_view_decodes_same_word (v : RowView) (row col : Nat) :
    v.address row col / 4 = row * v.fullFeatures + v.offsetFeatures + col := by
  unfold RowView.address
  omega

theorem full_candidate_control_reconstruction {B q g a : Nat}
    (candidate control : Tensor B (q + g + a)) (equalWords : candidate = control) :
    concat3 (slice candidate 0 (by omega)) (slice candidate q (by omega))
      (slice candidate (q + g) (by omega)) = control := by
  rw [concat3_original_slices, equalWords]

def gdnViews (B : Nat) : List RowView :=
  [⟨B, 16480, 0, 10240⟩, ⟨B, 16480, 10240, 6144⟩, ⟨B, 16480, 16384, 96⟩]
def attentionViews (B : Nat) : List RowView :=
  [⟨B, 14336, 0, 12288⟩, ⟨B, 14336, 12288, 1024⟩, ⟨B, 14336, 13312, 1024⟩]
def ffnViews (B : Nat) : List RowView :=
  [⟨B, 34816, 0, 17408⟩, ⟨B, 34816, 17408, 17408⟩]

theorem gdn_full_native_extents (B : Nat) :
    (gdnViews B).map RowView.strideBytes = [65920, 65920, 65920] ∧
    (gdnViews B).map RowView.offsetBytes = [0, 40960, 65536] ∧
    (gdnViews B).map RowView.width = [10240, 6144, 96] := by
  exact ⟨rfl, rfl, rfl⟩
theorem attention_full_native_extents (B : Nat) :
    (attentionViews B).map RowView.strideBytes = [57344, 57344, 57344] ∧
    (attentionViews B).map RowView.offsetBytes = [0, 49152, 53248] ∧
    (attentionViews B).map RowView.width = [12288, 1024, 1024] := by
  exact ⟨rfl, rfl, rfl⟩
theorem ffn_full_native_extents (B : Nat) :
    (ffnViews B).map RowView.strideBytes = [139264, 139264] ∧
    (ffnViews B).map RowView.offsetBytes = [0, 69632] ∧
    (ffnViews B).map RowView.width = [17408, 17408] := by
  exact ⟨rfl, rfl, rfl⟩

structure ViewIdentity where
  backing : HandleIdentity
  descriptor : RowView
  sourceIdentity : Nat
  outputIdentity : Nat
  deriving DecidableEq
inductive ObservationKind where
  | reader | continuation | source | status
  deriving DecidableEq
structure Observation where
  kind : ObservationKind
  id : Nat
  deriving DecidableEq
structure ViewCustody where
  identity : ViewIdentity
  declared : List Observation
  completed : List Observation
  life : Life
  pointer : Option Nat

def AllObserved (c : ViewCustody) : Prop :=
  c.declared ≠ [] ∧ ∀ o ∈ c.declared, o ∈ c.completed
structure ViewReleaseStep (c : ViewCustody) where
  fullDrain : AllObserved c
  live : c.life = .live
  handle : Handle
  sameBacking : handle.identity = c.identity.backing
  samePointer : handle.pointer = c.pointer
  completedLayer : CompletedLayer
  nativeRelease : ReleaseStep completedLayer handle

theorem view_release_requires_every_observation (c : ViewCustody)
    (s : ViewReleaseStep c) (o : Observation) (declared : o ∈ c.declared) :
    o ∈ c.completed := s.fullDrain.2 o declared

theorem undrained_view_cannot_release (c : ViewCustody) (o : Observation)
    (declared : o ∈ c.declared) (pending : o ∉ c.completed) :
    ¬ Nonempty (ViewReleaseStep c) := by
  rintro ⟨s⟩
  exact pending (view_release_requires_every_observation c s o declared)

theorem view_release_keeps_backing_identity (c : ViewCustody) (s : ViewReleaseStep c) :
    s.nativeRelease.after.identity = c.identity.backing := by
  rw [admitted_release_retains_exact_identity, s.sameBacking]

theorem failed_view_release_retains_pointer (c : ViewCustody) (s : ViewReleaseStep c)
    (failed : s.nativeRelease.outcome = .failure) :
    s.nativeRelease.after.pointer = c.pointer := by
  rw [s.nativeRelease.exactTransition, failed]
  exact s.samePointer

theorem uncertain_view_release_retains_pointer (c : ViewCustody) (s : ViewReleaseStep c)
    (uncertain : s.nativeRelease.outcome = .uncertain) :
    s.nativeRelease.after.pointer = c.pointer := by
  rw [s.nativeRelease.exactTransition, uncertain]
  exact s.samePointer

theorem immutable_view_observation {Output : Type} (identity : ViewIdentity)
    (whole : Tensor identity.descriptor.rows identity.descriptor.fullFeatures)
    (observer : ViewIdentity → Tensor identity.descriptor.rows identity.descriptor.fullFeatures → Output) :
    (identity, whole, observer identity whole).1 = identity := rfl

end Packing20
