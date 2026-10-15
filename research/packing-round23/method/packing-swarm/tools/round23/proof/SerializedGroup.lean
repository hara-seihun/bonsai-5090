import PairedTile
namespace Packing23
open Packing20 Packing21 Packing22

structure Coordinate where
  pass : Nat
  m : Nat
  n : Nat
  i : Nat
  deriving DecidableEq, BEq

def originalSchedule (a : Arm) : List Coordinate :=
  (List.range 2).flatMap fun pass =>
    (List.range 4).flatMap fun n =>
      (List.range (2 * groupsPerWarp a)).flatMap fun m =>
        (List.range 4).map fun i => ⟨pass, m, n, i⟩

def serializedSchedule (a : Arm) : List Coordinate :=
  (List.range (groupsPerWarp a)).flatMap fun q =>
    (List.range 2).flatMap fun pass =>
      (List.range 4).flatMap fun n =>
        (List.range 2).flatMap fun mm =>
          (List.range 4).map fun i => ⟨pass, q * 2 + mm, n, i⟩

def ValidCoordinate (a : Arm) (c : Coordinate) : Prop :=
  c.pass < 2 ∧ c.m < 2 * groupsPerWarp a ∧ c.n < 4 ∧ c.i < 4

theorem local_m_reconstruction (a : Arm) (m : Nat) (hm : m < 2 * groupsPerWarp a) :
    m / 2 < groupsPerWarp a ∧ m % 2 < 2 ∧ m / 2 * 2 + m % 2 = m := by
  cases a <;> simp [groupsPerWarp] at * <;> omega

theorem local_m_unique (q mm q' mm' : Nat) (hm : mm < 2) (hm' : mm' < 2)
    (same : q * 2 + mm = q' * 2 + mm') : q = q' ∧ mm = mm' := by omega

theorem local_m_bound (a : Arm) (q mm : Nat) (hq : q < groupsPerWarp a) (hm : mm < 2) :
    q * 2 + mm < 2 * groupsPerWarp a := by
  cases a <;> simp [groupsPerWarp] at * <;> omega

theorem original_schedule_membership (a : Arm) (c : Coordinate) :
    c ∈ originalSchedule a ↔ ValidCoordinate a c := by
  simp only [originalSchedule, List.mem_flatMap, List.mem_range, List.mem_map]
  constructor
  · rintro ⟨pass, hp, n, hn, m, hm, i, hi, he⟩
    cases he
    exact ⟨hp, hm, hn, hi⟩
  · intro h
    exact ⟨c.pass, h.1, c.n, h.2.2.1, c.m, h.2.1, c.i, h.2.2.2, rfl⟩

set_option maxRecDepth 4096 in
set_option maxHeartbeats 8000000 in
theorem schedule_interchange_permutation (a : Arm) :
    List.Perm (originalSchedule a) (serializedSchedule a) := by
  cases a <;> decide

set_option maxRecDepth 4096 in
set_option maxHeartbeats 8000000 in
theorem original_schedule_unique (a : Arm) : (originalSchedule a).Nodup := by
  cases a <;> decide

theorem serialized_schedule_unique (a : Arm) : (serializedSchedule a).Nodup :=
  (schedule_interchange_permutation a).nodup (original_schedule_unique a)

theorem serialized_schedule_membership (a : Arm) (c : Coordinate) :
    c ∈ serializedSchedule a ↔ ValidCoordinate a c := by
  rw [← (schedule_interchange_permutation a).mem_iff]
  exact original_schedule_membership a c

def updateAt {C F : Type} [DecidableEq C] (step : C → F → F)
    (state : C → F) (c : C) : C → F :=
  fun x => if x = c then step c (state c) else state x

theorem independent_coordinate_updates_commute {C F : Type} [DecidableEq C]
    (step : C → F → F) (state : C → F) (c d : C) :
    updateAt step (updateAt step state c) d = updateAt step (updateAt step state d) c := by
  funext x
  by_cases h : c = d
  · subst d; rfl
  · by_cases hc : x = c <;> by_cases hd : x = d <;> simp_all [updateAt]

def sweep {C F : Type} [DecidableEq C] (coords : List C)
    (step : C → F → F) (state : C → F) : C → F := coords.foldl (updateAt step) state

theorem independent_coordinate_loop_interchange {C F : Type} [DecidableEq C]
    (before after : List C) (order : List.Perm before after) (step : C → F → F) (state : C → F) :
    sweep before step state = sweep after step state := by
  apply order.foldl_eq'
  intro c _ d _ s
  exact independent_coordinate_updates_commute step s c d

theorem sweep_absent_coordinate {C F : Type} [DecidableEq C]
    (coords : List C) (step : C → F → F) (state : C → F) (c : C) (absent : c ∉ coords) :
    sweep coords step state c = state c := by
  induction coords generalizing state with
  | nil => rfl
  | cons d ds ih =>
    have hd : c ≠ d := by intro h; apply absent; simp [h]
    have ht : c ∉ ds := by intro h; exact absent (List.mem_cons_of_mem d h)
    change sweep ds step (updateAt step state d) c = state c
    rw [ih _ ht]
    simp [updateAt, hd]

theorem sweep_one_update_per_coordinate {C F : Type} [DecidableEq C]
    (coords : List C) (unique : coords.Nodup) (step : C → F → F) (state : C → F)
    (c : C) (present : c ∈ coords) : sweep coords step state c = step c (state c) := by
  induction coords generalizing state with
  | nil => simp at present
  | cons d ds ih =>
    have h := List.nodup_cons.mp unique
    change sweep ds step (updateAt step state d) c = step c (state c)
    by_cases eq : c = d
    · subst c
      rw [sweep_absent_coordinate ds step _ d h.1]
      simp [updateAt]
    · have ht : c ∈ ds := by simpa [eq] using present
      rw [ih h.2 _ ht]
      simp [updateAt, eq]

def scheduledFold {C F : Type} [DecidableEq C] (coords : List C)
    (step : Nat → C → F → F) (zero : F) : Nat → C → F
  | 0 => fun _ => zero
  | k + 1 => sweep coords (step k) (scheduledFold coords step zero k)

theorem scheduled_fold_independent_loop_interchange {C F : Type} [DecidableEq C]
    (before after : List C) (order : List.Perm before after) (step : Nat → C → F → F)
    (zero : F) (k : Nat) : scheduledFold before step zero k = scheduledFold after step zero k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [scheduledFold, ih]
    exact independent_coordinate_loop_interchange before after order (step k) _

theorem scheduled_fold_is_ascending_coordinate_fold {C F : Type} [DecidableEq C]
    (coords : List C) (unique : coords.Nodup) (step : Nat → C → F → F) (zero : F)
    (c : C) (present : c ∈ coords) (k : Nat) :
    scheduledFold coords step zero k c = orderedFold (fun kb => step kb c) zero k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    change sweep coords (step k) (scheduledFold coords step zero k) c =
      step k c (orderedFold (fun kb => step kb c) zero k)
    rw [sweep_one_update_per_coordinate coords unique (step k) _ c present, ih]

theorem serialized_preserves_forty_ascending_folds {F : Type} (a : Arm)
    (step : Nat → Coordinate → F → F) (zero : F) :
    scheduledFold (originalSchedule a) step zero 40 =
      scheduledFold (serializedSchedule a) step zero 40 :=
  scheduled_fold_independent_loop_interchange _ _ (schedule_interchange_permutation a) step zero 40

theorem serialized_coordinate_is_literal_forty_fold {F : Type} (a : Arm)
    (step : Nat → Coordinate → F → F) (zero : F) (c : Coordinate) (valid : ValidCoordinate a c) :
    scheduledFold (serializedSchedule a) step zero 40 c =
      orderedFold (fun kb => step kb c) zero 40 :=
  scheduled_fold_is_ascending_coordinate_fold _ (serialized_schedule_unique a) step zero c
    ((serialized_schedule_membership a c).mpr valid) 40

def componentByte (i : Nat) := (i % 2) * 2 + i / 2

theorem fragment_component_dictionary :
    componentByte 0 = 0 ∧ componentByte 1 = 2 ∧ componentByte 2 = 1 ∧ componentByte 3 = 3 := by decide

def retainedFragment (a : Arm) (packed : Nat → Nat → Nat → Word)
    (expand : Word → Nat → Word) (warp lane m h i : Nat) :=
  expand (packed (warp * groupsPerWarp a + m / 2) lane (m % 2 * 2 + h)) (componentByte i)

def localFragment (a : Arm) (packed : Nat → Nat → Nat → Word)
    (expand : Word → Nat → Word) (warp lane q mm h i : Nat) :=
  expand (packed (warp * groupsPerWarp a + q) lane (mm * 2 + h)) (componentByte i)

theorem shortened_fragment_same_saved_word (a : Arm) (packed : Nat → Nat → Nat → Word)
    (expand : Word → Nat → Word) (warp lane q mm h i : Nat) (hm : mm < 2) :
    retainedFragment a packed expand warp lane (q * 2 + mm) h i =
      localFragment a packed expand warp lane q mm h i := by
  have hd : (q * 2 + mm) / 2 = q := by omega
  have hr : (q * 2 + mm) % 2 = mm := by omega
  simp only [retainedFragment, localFragment, hd, hr]

theorem shortened_weight_scale_same_address (a : Arm) (warp q mm r gid byte : Nat) (hm : mm < 2) :
    weightScaleByte (warp * groupsPerWarp a + (q * 2 + mm) / 2) gid
      ((q * 2 + mm) % 2) r byte =
    weightScaleByte (group a warp q) gid mm r byte := by
  have hd : (q * 2 + mm) / 2 = q := by omega
  have hr : (q * 2 + mm) % 2 = mm := by omega
  simp only [hd, hr, group]

theorem shortened_feature_same_literal_address (a : Arm) (tile warp q mm r gid : Nat) :
    (tile * 8 + warp * groupsPerWarp a) * 32 + gid + (q * 2 + mm) * 16 + r * 8 =
      feature (tile * 8 + group a warp q) mm r gid := by
  simpa only [group, Nat.add_assoc] using
    literal_m_to_group_feature (tile * 8 + warp * groupsPerWarp a) q mm r gid

def serializedOutput {F : Type} (ops : FloatOps F) (a : Arm)
    (payload : Nat → Coordinate → FoldPayload) (ag0 ag1 : Word) (prior : F) (m n i : Nat) : F × F :=
  let acc := scheduledFold (serializedSchedule a) (fun kb c => orderedStep ops (payload kb c)) ops.zero 40
  pairedOutput ops (acc ⟨0, m, n, i⟩) (acc ⟨1, m, n, i⟩) ag0 ag1 prior

theorem serialized_output_exact_primary_first_ordered_folds {F : Type} (ops : FloatOps F) (a : Arm)
    (payload : Nat → Coordinate → FoldPayload) (ag0 ag1 : Word) (prior : F) (m n i : Nat)
    (hm : m < 2 * groupsPerWarp a) (hn : n < 4) (hi : i < 4) :
    serializedOutput ops a payload ag0 ag1 prior m n i =
      (literalMode1 ops (orderedFold (fun kb => orderedStep ops (payload kb ⟨0, m, n, i⟩)) ops.zero 40) ag0 prior,
       literalMode1 ops (orderedFold (fun kb => orderedStep ops (payload kb ⟨1, m, n, i⟩)) ops.zero 40) ag1
         (literalMode1 ops (orderedFold (fun kb => orderedStep ops (payload kb ⟨0, m, n, i⟩)) ops.zero 40) ag0 prior)) := by
  have h0 : ValidCoordinate a ⟨0, m, n, i⟩ := ⟨by change 0 < 2; decide, hm, hn, hi⟩
  have h1 : ValidCoordinate a ⟨1, m, n, i⟩ := ⟨by change 1 < 2; decide, hm, hn, hi⟩
  unfold serializedOutput
  dsimp only
  rw [serialized_coordinate_is_literal_forty_fold a _ ops.zero _ h0,
      serialized_coordinate_is_literal_forty_fold a _ ops.zero _ h1]
  rfl

theorem serialized_saved_payload_dictionary_complete_output {F : Type} (ops : FloatOps F) (a : Arm)
    (staged source : Nat → Coordinate → FoldPayload) (ag0 ag1 : Word) (prior : F) (m n i : Nat)
    (hm : m < 2 * groupsPerWarp a) (hn : n < 4) (hi : i < 4)
    (dictionary : ∀ kb c, kb < 40 → ValidCoordinate a c → staged kb c = source kb c) :
    serializedOutput ops a staged ag0 ag1 prior m n i =
      (literalMode1 ops (orderedFold (fun kb => orderedStep ops (source kb ⟨0, m, n, i⟩)) ops.zero 40) ag0 prior,
       literalMode1 ops (orderedFold (fun kb => orderedStep ops (source kb ⟨1, m, n, i⟩)) ops.zero 40) ag1
         (literalMode1 ops (orderedFold (fun kb => orderedStep ops (source kb ⟨0, m, n, i⟩)) ops.zero 40) ag0 prior)) := by
  rw [serialized_output_exact_primary_first_ordered_folds ops a staged ag0 ag1 prior m n i hm hn hi]
  rw [payload_dictionary_preserves_ordered_fold ops _ _
        (fun kb hk => dictionary kb ⟨0, m, n, i⟩ hk ⟨by change 0 < 2; decide, hm, hn, hi⟩),
      payload_dictionary_preserves_ordered_fold ops _ _
        (fun kb hk => dictionary kb ⟨1, m, n, i⟩ hk ⟨by change 1 < 2; decide, hm, hn, hi⟩)]

def fullDomainCoordinate (a : Arm) (f t pass : Nat) : Coordinate :=
  ⟨pass, (f % 256 / 32 % groupsPerWarp a) * 2 + f % 32 / 16,
    t % 32 / 8, (f % 16 / 8) * 2 + t % 2⟩

theorem full_domain_coordinate_in_serialized_schedule (a : Arm) (f t pass : Nat) (hp : pass < 2) :
    fullDomainCoordinate a f t pass ∈ serializedSchedule a := by
  apply (serialized_schedule_membership a _).mpr
  unfold ValidCoordinate fullDomainCoordinate
  dsimp only
  cases a <;> simp only [groupsPerWarp] <;> omega

theorem structural_declared_fragment_extents :
    2 * (2 * groupsPerWarp Arm.candidate) * 4 = 32 ∧
    2 * (2 * groupsPerWarp Arm.control) * 4 = 16 ∧
    2 * 2 * 4 = 16 ∧
    (2 * groupsPerWarp Arm.candidate) * 2 = 8 ∧
    (2 * groupsPerWarp Arm.control) * 2 = 4 ∧ 2 * 2 = 4 ∧
    2 * (2 * groupsPerWarp Arm.candidate) * 4 * 4 = 128 ∧
    2 * (2 * groupsPerWarp Arm.control) * 4 * 4 = 64 := by decide

theorem structural_activation_fragment_request_multiplicity :
    2 * 4 = 8 ∧ groupsPerWarp Arm.candidate * 2 * 4 = 16 ∧
    groupsPerWarp Arm.control * 2 * 4 = 8 := by decide

end Packing23
