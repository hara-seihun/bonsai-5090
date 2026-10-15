import NativeJoin
namespace Packing21
open Packing20

-- Pointer offsets/strides are in words. Materialization is the specified full row copy.
def readStrided (memory : Nat → Word) (stride offset B W : Nat) : Tensor B W :=
  fun row col => memory (row.val * stride + offset + col.val)
def copyRows (memory : Nat → Word) (stride offset width : Nat) : Nat → Word :=
  fun i => memory ((i / width) * stride + offset + i % width)
def readableSpan (rows stride width : Nat) := (rows - 1) * stride + width

theorem compact_index_div_mod (row col width : Nat) (hc : col < width) :
    (row * width + col) / width = row ∧ (row * width + col) % width = col := by
  have hw : 0 < width := by omega
  constructor
  · rw [Nat.mul_comm row width, Nat.mul_add_div hw, Nat.div_eq_of_lt hc, Nat.add_zero]
  · simp [Nat.add_mod, Nat.mod_eq_of_lt hc]

theorem copied_rows_same_complete_coordinates (memory : Nat → Word)
    (stride offset B W : Nat) :
    rowTensor (copyRows memory stride offset W) B W = readStrided memory stride offset B W := by
  funext row col
  have h := compact_index_div_mod row.val col.val W col.isLt
  simp only [rowTensor, copyRows, readStrided, h.1, h.2]

theorem copied_subview_same_complete_coordinates (memory : Nat → Word)
    (stride base oldWidth offset B W : Nat) (fits : offset + W ≤ oldWidth) :
    readStrided (copyRows memory stride base oldWidth) oldWidth offset B W =
      readStrided memory stride (base + offset) B W := by
  funext row col
  have hc : offset + col.val < oldWidth := by have := col.isLt; omega
  have h := compact_index_div_mod row.val (offset + col.val) oldWidth hc
  simp only [readStrided, copyRows, Nat.add_assoc, h.1, h.2]

theorem strided_input_is_whole_row_slice (memory : Nat → Word) (v : RowView)
    (fits : v.offsetFeatures + v.width ≤ v.fullFeatures) :
    readStrided memory v.fullFeatures v.offsetFeatures v.rows v.width =
      slice (rowTensor memory v.rows v.fullFeatures) v.offsetFeatures fits := by
  exact native_view_is_full_row_slice memory v fits

theorem every_read_inside_readable_span (rows stride width row col : Nat)
    (hr : row < rows) (hc : col < width) :
    row * stride + col < readableSpan rows stride width := by
  have hm : row * stride ≤ (rows - 1) * stride :=
    Nat.mul_le_mul_right stride (by omega)
  unfold readableSpan
  omega

theorem final_coordinate_requires_full_span (rows stride width : Nat)
    (_hb : 0 < rows) (hw : 0 < width) :
    (rows - 1) * stride + (width - 1) + 1 = readableSpan rows stride width := by
  unfold readableSpan
  omega

theorem readable_span_inside_whole_allocation (rows stride offset width : Nat)
    (hb : 0 < rows) (fits : offset + width ≤ stride) :
    offset + readableSpan rows stride width ≤ rows * stride := by
  have he : rows = (rows - 1) + 1 := by omega
  have hm : rows * stride = (rows - 1) * stride + stride := by
    calc rows * stride = ((rows - 1) + 1) * stride := congrArg (fun n => n * stride) he
         _ = (rows - 1) * stride + stride := by rw [Nat.add_mul, Nat.one_mul]
  unfold readableSpan
  rw [hm]
  omega

theorem readable_span_bytes (rows stride width : Nat) :
    4 * readableSpan rows stride width = (rows - 1) * (4 * stride) + 4 * width := by
  simp only [readableSpan, Nat.mul_add]
  rw [Nat.mul_left_comm]

def selectRows {B D W : Nat} (rowmap : Fin D → Fin B) (input : Tensor B W) : Tensor D W :=
  fun row col => input (rowmap row) col

theorem explicit_rowmap_preserves_dictionary {B D W : Nat}
    (rowmap : Fin D → Fin B) (a b : Tensor B W) (same : a = b) :
    selectRows rowmap a = selectRows rowmap b := by rw [same]

def ffnGate (memory : Nat → Word) (B : Nat) := readStrided memory 34816 0 B 17408
def ffnUp (memory : Nat → Word) (B : Nat) := readStrided memory 34816 17408 B 17408

theorem ffn_shared_ldx_complete_dictionary (memory : Nat → Word) (B : Nat) :
    ffnGate memory B = rowTensor (copyRows memory 34816 0 17408) B 17408 ∧
    ffnUp memory B = rowTensor (copyRows memory 34816 17408 17408) B 17408 := by
  constructor <;> symm <;> apply copied_rows_same_complete_coordinates

theorem ffn_shared_ldx_reconstructs_full_merged (memory : Nat → Word) (B : Nat) :
    concat2 (ffnGate memory B) (ffnUp memory B) = rowTensor memory B 34816 := by
  have hg := strided_input_is_whole_row_slice memory (RowView.mk B 34816 0 17408) (by change 0 + 17408 ≤ 34816; decide)
  have hu := strided_input_is_whole_row_slice memory (RowView.mk B 34816 17408 17408) (by change 17408 + 17408 ≤ 34816; decide)
  change concat2 (readStrided memory 34816 0 B 17408)
    (readStrided memory 34816 17408 B 17408) = _
  rw [hg, hu]
  exact concat2_original_slices _

theorem ffn_borrowed_complete_silu_quant_down_residual {B : Nat} {Q : Type}
    (memory : Nat → Word)
    (siluMul : Tensor B 17408 → Tensor B 17408 → Tensor B 17408)
    (quantize : Tensor B 17408 → Q) (down : Q → Tensor B 5120)
    (prior : Tensor B 5120) (residual : Tensor B 5120 → Tensor B 5120 → Tensor B 5120) :
    residual prior (down (quantize (siluMul (ffnGate memory B) (ffnUp memory B)))) =
    residual prior (down (quantize (siluMul
      (rowTensor (copyRows memory 34816 0 17408) B 17408)
      (rowTensor (copyRows memory 34816 17408 17408) B 17408)))) := by
  obtain ⟨hg, hu⟩ := ffn_shared_ldx_complete_dictionary memory B
  rw [hg, hu]

def gdnGate (memory : Nat → Word) (B : Nat) := readStrided memory 16480 10240 B 6144
def oldGdnProjection (memory : Nat → Word) := copyRows memory 16480 0 16384

theorem gdn_gz_ld_complete_dictionary (memory : Nat → Word) (B : Nat) :
    readStrided (oldGdnProjection memory) 16384 10240 B 6144 = gdnGate memory B := by
  exact copied_subview_same_complete_coordinates memory 16480 0 16384 10240 B 6144 (by decide)

theorem gdn_complete_head_coordinates (memory : Nat → Word) (row head hd : Nat)
    (hh : head < 48) (hdBound : hd < 128) :
    memory (row * 16480 + 10240 + head * 128 + hd) =
      oldGdnProjection memory (row * 16384 + 10240 + head * 128 + hd) := by
  have hm := Nat.mul_le_mul_right 128 (show head ≤ 47 by omega)
  have hc : 10240 + head * 128 + hd < 16384 := by omega
  have h := compact_index_div_mod row (10240 + head * 128 + hd) 16384 hc
  simp only [oldGdnProjection, copyRows, Nat.add_assoc] at *
  rw [h.1, h.2]
  simp

theorem gdn_borrowed_full_opaque_preparation {B : Nat} {Q : Type}
    (memory : Nat → Word) (y : Tensor B 6144) (norm : Fin 128 → Word)
    (gnorm : Tensor B 6144 → Tensor B 6144 → (Fin 128 → Word) → Tensor B 6144)
    (quantize : Tensor B 6144 → Q) (out : Q → Tensor B 5120)
    (prior : Tensor B 5120) (residual : Tensor B 5120 → Tensor B 5120 → Tensor B 5120) :
    residual prior (out (quantize (gnorm y (gdnGate memory B) norm))) =
    residual prior (out (quantize (gnorm y
      (readStrided (oldGdnProjection memory) 16384 10240 B 6144) norm))) := by
  rw [gdn_gz_ld_complete_dictionary]

theorem concrete_readable_spans (B : Nat) :
    readableSpan B 34816 17408 = (B - 1) * 34816 + 17408 ∧
    readableSpan B 16480 6144 = (B - 1) * 16480 + 6144 := ⟨rfl, rfl⟩

theorem full_ffn_literal_coordinate_coverage (col : Fin 17408) :
    ∃ warp lane component : Nat, warp < 17 ∧ lane < 32 ∧ component < 32 ∧
      warp * 1024 + lane * 32 + component = col.val := by
  refine ⟨col.val / 1024, (col.val % 1024) / 32, col.val % 32, ?_⟩
  have hcol := col.isLt
  have h1024 := Nat.mod_add_div col.val 1024
  have h32 := Nat.mod_add_div (col.val % 1024) 32
  have hmod := Nat.mod_mul_left_mod col.val 32 32
  have hsmall := Nat.mod_lt col.val (show 0 < 1024 by decide)
  have hcomponent := Nat.mod_lt col.val (show 0 < 32 by decide)
  omega

theorem full_gdn_head_coordinate_coverage (col : Fin 6144) :
    ∃ head hd : Nat, head < 48 ∧ hd < 128 ∧ head * 128 + hd = col.val := by
  refine ⟨col.val / 128, col.val % 128, ?_⟩
  have hcol := col.isLt
  have h := Nat.mod_add_div col.val 128
  have hmod := Nat.mod_lt col.val (show 0 < 128 by decide)
  omega

theorem gdn_new_stride_retains_row_gap (row col : Nat) :
    row * 16480 + 10240 + col = (row * 16384 + 10240 + col) + row * 96 := by omega

-- A borrowed pointer carries no release capability; only the exact whole handle does.
structure BorrowIdentity where
  backing : HandleIdentity
  descriptor : RowView
  source : SourceImage
  manifest : List Nat
  outputIdentity : Nat
  deriving DecidableEq

def ExactBorrow (i : BorrowIdentity) : Prop :=
  ValidSource i.source ∧ ValidView i.descriptor ∧
  i.descriptor.rows = i.source.domain.rows ∧ i.manifest = i.source.domain.tensors ∧
  i.manifest ≠ [] ∧ i.backing.owner.layer = i.source.domain.layer ∧
  i.backing.owner.generation = i.source.domain.generation ∧
  i.backing.owner.boundary = i.source.domain.boundary ∧
  i.backing.range.offset = 0 ∧
  i.backing.range.bytes = i.descriptor.allocationBytes ∧
  i.backing.range.bytes = i.backing.range.allocationBytes

def borrowedPointer (i : BorrowIdentity) (wholePointer : Option Nat) :=
  wholePointer.map (fun p => p + i.descriptor.offsetBytes)

inductive RequiredObservation where
  | primary | innovation | status | owner | control | source | norm | folds
  | continuation | hostLayer | drainsFence
  deriving DecidableEq

def requiredObservations : List RequiredObservation :=
  [.primary, .innovation, .status, .owner, .control, .source, .norm, .folds,
    .continuation, .hostLayer, .drainsFence]

structure BorrowCustody where
  identity : BorrowIdentity
  currentIdentity : BorrowIdentity
  original : Handle
  completed : List RequiredObservation
  sourceAfter : SourceImage

structure BorrowRelease (c : BorrowCustody) where
  exact : ExactBorrow c.identity
  immutable : c.currentIdentity = c.identity
  originalIdentity : c.original.identity = c.identity.backing
  sourcePreserved : c.sourceAfter = c.identity.source
  observations : ∀ o ∈ requiredObservations, o ∈ c.completed
  layer : CompletedLayer
  wholeRelease : ReleaseStep layer c.original

theorem all_required_observations_present (o : RequiredObservation) :
    o ∈ requiredObservations := by cases o <;> simp [requiredObservations]

theorem borrow_release_needs_every_required_observation (c : BorrowCustody)
    (s : BorrowRelease c) (o : RequiredObservation) : o ∈ c.completed :=
  s.observations o (all_required_observations_present o)

theorem pending_borrow_cannot_release (c : BorrowCustody) (o : RequiredObservation)
    (pending : o ∉ c.completed) : ¬ Nonempty (BorrowRelease c) := by
  rintro ⟨s⟩
  exact pending (borrow_release_needs_every_required_observation c s o)

theorem changed_identity_cannot_release (c : BorrowCustody)
    (changed : c.currentIdentity ≠ c.identity) : ¬ Nonempty (BorrowRelease c) := by
  rintro ⟨s⟩
  exact changed s.immutable

theorem changed_source_cannot_release (c : BorrowCustody)
    (changed : c.sourceAfter ≠ c.identity.source) : ¬ Nonempty (BorrowRelease c) := by
  rintro ⟨s⟩
  exact changed s.sourcePreserved

theorem borrow_release_needs_completed_images_and_drains (c : BorrowCustody)
    (s : BorrowRelease c) :
    s.layer.sealed = true ∧ s.layer.stateImage ≠ [] ∧ s.layer.sourceImage ≠ [] ∧
      s.layer.sourceDrained = true ∧ (∀ r ∈ s.layer.readersDrained, r = true) := by
  have hi := dead_has_completed_images s.layer c.original s.wholeRelease.dead
  have hd := dead_all_dependencies_drained s.layer c.original s.wholeRelease.dead
  exact ⟨hi.1, hi.2.1, hi.2.2, hd.1, hd.2⟩

theorem borrow_release_only_exact_whole_allocation (c : BorrowCustody)
    (s : BorrowRelease c) :
    c.original.identity.range.offset = 0 ∧ 0 < c.original.identity.range.bytes ∧
      c.original.identity.range.bytes = c.original.identity.range.allocationBytes :=
  dead_releases_whole_allocation s.layer c.original s.wholeRelease.dead

theorem interior_handle_cannot_release (layer : CompletedLayer) (interior : Handle)
    (offset : 0 < interior.identity.range.offset) : ¬ Nonempty (ReleaseStep layer interior) := by
  rintro ⟨s⟩
  have h := (dead_releases_whole_allocation layer interior s.dead).1
  omega

theorem borrow_release_retains_exact_backing (c : BorrowCustody) (s : BorrowRelease c) :
    s.wholeRelease.after.identity = c.identity.backing := by
  rw [admitted_release_retains_exact_identity, s.originalIdentity]

theorem failed_borrow_retains_original_and_interior_pointer (c : BorrowCustody)
    (s : BorrowRelease c) (failed : s.wholeRelease.outcome = .failure) :
    s.wholeRelease.after.pointer = c.original.pointer ∧
    borrowedPointer c.identity s.wholeRelease.after.pointer =
      borrowedPointer c.identity c.original.pointer := by
  rw [s.wholeRelease.exactTransition, failed]
  exact ⟨rfl, rfl⟩

theorem uncertain_borrow_retains_original_and_interior_pointer (c : BorrowCustody)
    (s : BorrowRelease c) (uncertain : s.wholeRelease.outcome = .uncertain) :
    s.wholeRelease.after.pointer = c.original.pointer ∧
    borrowedPointer c.identity s.wholeRelease.after.pointer =
      borrowedPointer c.identity c.original.pointer := by
  rw [s.wholeRelease.exactTransition, uncertain]
  exact ⟨rfl, rfl⟩

theorem successful_whole_release_unsets_borrow_pointer (c : BorrowCustody)
    (s : BorrowRelease c) (success : s.wholeRelease.outcome = .success) :
    borrowedPointer c.identity s.wholeRelease.after.pointer = none := by
  rw [s.wholeRelease.exactTransition, success]
  rfl

theorem immutable_borrow_preserves_full_source_identity (c : BorrowCustody)
    (s : BorrowRelease c) :
    c.currentIdentity.source.domain = c.identity.source.domain ∧
    c.currentIdentity.source.x = c.identity.source.x ∧
    c.currentIdentity.source.norm = c.identity.source.norm ∧
    c.currentIdentity.manifest = c.identity.manifest := by
  rw [s.immutable]
  exact ⟨rfl, rfl, rfl, rfl⟩

end Packing21
