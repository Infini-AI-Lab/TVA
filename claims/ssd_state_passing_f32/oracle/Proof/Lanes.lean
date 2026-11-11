import Claim
import TVA.Proof
import Proof.Meta

/-!
# Lanes and footprints

Program `(pm, h)` of the grid (`program_id(0) = pm`, `program_id(2) = h`) carries the state of the
`1024` lanes `m = pm·1024 + i` of head `h`. This file names the elements it owns (`lanes`, the
output slots `slotW`), what it may read (`Rin`), and proves the footprints of distinct programs
and slots disjoint.
-/
namespace Claims.SsdStatePassing

open TVA

/-- `BLOCK_SIZE` is `1024`. -/
theorem BL_eq : BL = 1024 := rfl

/-- `Nat` division: `a·K + r` with `r < K` determines `a` and `r`. -/
theorem div_unique {a a' r r' K : Nat} (h : a * K + r = a' * K + r') (hr : r < K) (hr' : r' < K) :
    a = a' ∧ r = r' := by
  have hK : 0 < K := by omega
  have h1 : (a * K + r) / K = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hr]; simp
  have h2 : (a' * K + r') / K = a' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hr']; simp
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

/-- The element `(0, c, h, m)` of `states` and `out` determines `c`, `h` and `m`. -/
theorem stIdx_inj {d : Inst} {c c' h h' m m' : Nat} (hh : h < d.nheads) (hh' : h' < d.nheads)
    (hm : m < d.dim) (hm' : m' < d.dim) (e : stIdx d c h m = stIdx d c' h' m') :
    c = c' ∧ h = h' ∧ m = m' := by
  unfold stIdx at e
  have hr : h * d.dim + m < d.nheads * d.dim := by
    have : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
    rw [Nat.succ_mul] at this; omega
  have hr' : h' * d.dim + m' < d.nheads * d.dim := by
    have : (h' + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh'
    rw [Nat.succ_mul] at this; omega
  obtain ⟨h1, h2⟩ := div_unique (a := c) (a' := c') (K := d.nheads * d.dim)
    (by rw [← add_assoc, ← add_assoc]; exact e) hr hr'
  obtain ⟨h3, h4⟩ := div_unique (K := d.dim) h2 hm hm'
  exact ⟨h1, h3, h4⟩

/-- The element `(0, h, m)` of `final_states` determines `h` and `m`. -/
theorem finIdx_inj {d : Inst} {h h' m m' : Nat} (hm : m < d.dim) (hm' : m' < d.dim)
    (e : finIdx d h m = finIdx d h' m') : h = h' ∧ m = m' :=
  div_unique (K := d.dim) e hm hm'

/-- The state elements program `(pm, ·)` owns: its block of `1024` lanes, cut at `dim`. -/
def lanes (d : Inst) (pm : Nat) : Set Nat := {m | pm * BL ≤ m ∧ m < pm * BL + BL ∧ m < d.dim}

/-- Slot `c < nchunks` of program `(pm, h)` in `out`: the elements `(0, c, h, m)` of its lanes. -/
def outIdxs (d : Inst) (h pm c : Nat) : Set Nat := {j | ∃ m ∈ lanes d pm, j = stIdx d c h m}

/-- The elements `(0, h, m)` of `final_states` of program `(pm, h)`'s lanes. -/
def finIdxs (d : Inst) (h pm : Nat) : Set Nat := {j | ∃ m ∈ lanes d pm, j = finIdx d h m}

/-- **The slots of the state**: slot `c < nchunks` is the state entering chunk `c`, held in `out`'s
chunk `c`; slot `nchunks` is the final state, held in `final_states`. -/
def slotW (d : Inst) (h pm c : Nat) : Set Int :=
  if c < d.nchunks then d.out.elems (outIdxs d h pm c) else d.finalst.elems (finIdxs d h pm)

/-- What program `(pm, h)` writes: all its slots. -/
def Wp (d : Inst) (h pm : Nat) : Set Int := iterW (fun c => slotW d h pm c) (d.nchunks + 1)

/-- What every program may read: the bytes of the five input buffers. -/
def Rin (d : Inst) : Set Int :=
  d.states.bytes ∪ d.dA.bytes ∪ d.initst.bytes ∪ d.seqIdx.bytes ∪ d.chunkOffs.bytes

/-- A slot below `nchunks` is the elements of `out`. -/
theorem slotW_out {d : Inst} {h pm c : Nat} (hc : c < d.nchunks) :
    slotW d h pm c = d.out.elems (outIdxs d h pm c) := by simp [slotW, hc]

/-- The last slot is the elements of `final_states`. -/
theorem slotW_fin {d : Inst} {h pm : Nat} : slotW d h pm d.nchunks = d.finalst.elems (finIdxs d h pm) := by
  simp [slotW]

/-- An element `(0, c, h, m)` of `states` or `out` lies below `nchunks · nheads · dim`. -/
theorem stIdx_lt {d : Inst} {c h m : Nat} (hc : c < d.nchunks) (hh : h < d.nheads) (hm : m < d.dim) :
    stIdx d c h m < d.nchunks * (d.nheads * d.dim) := by
  unfold stIdx
  have h1 : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
  have h2 : (c + 1) * (d.nheads * d.dim) ≤ d.nchunks * (d.nheads * d.dim) := Nat.mul_le_mul_right _ hc
  rw [Nat.succ_mul] at h1 h2
  omega

/-- An element `(0, h, m)` of `final_states` lies below `nheads · dim`. -/
theorem finIdx_lt {d : Inst} {h m : Nat} (hh : h < d.nheads) (hm : m < d.dim) :
    finIdx d h m < d.nheads * d.dim := by
  unfold finIdx
  have h1 : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
  rw [Nat.succ_mul] at h1
  omega

/-- The elements of a slot of `out` lie inside `out`. -/
theorem outIdxs_lt {d : Inst} (hd : d.ok) {h pm c : Nat} (hc : c < d.nchunks) (hh : h < d.nheads) :
    ∀ j ∈ outIdxs d h pm c, j < d.out.len := by
  rintro j ⟨m, ⟨-, -, hm⟩, rfl⟩
  exact lt_of_lt_of_le (stIdx_lt hc hh hm) hd.out_len

/-- The elements of the last slot lie inside `final_states`. -/
theorem finIdxs_lt {d : Inst} (hd : d.ok) {h pm : Nat} (hh : h < d.nheads) :
    ∀ j ∈ finIdxs d h pm, j < d.finalst.len := by
  rintro j ⟨m, ⟨-, -, hm⟩, rfl⟩
  exact lt_of_lt_of_le (finIdx_lt hh hm) hd.fin_len

/-- Distinct slots of one program are disjoint. -/
theorem slotW_disjoint {d : Inst} (hd : d.ok) {h pm c c' : Nat} (hh : h < d.nheads)
    (hc : c ≤ d.nchunks) (hc' : c' ≤ d.nchunks) (hne : c ≠ c') :
    Disjoint (slotW d h pm c) (slotW d h pm c') := by
  by_cases h1 : c < d.nchunks <;> by_cases h2 : c' < d.nchunks
  · rw [slotW_out h1, slotW_out h2]
    refine d.out.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro j ⟨m, ⟨-, -, hm⟩, rfl⟩ ⟨m', ⟨-, -, hm'⟩, e⟩
    exact hne (stIdx_inj hh hh hm hm' e).1
  · rw [slotW_out h1, show c' = d.nchunks by omega, slotW_fin]
    exact d.out.elems_disjoint_of d.finalst hd.disj.2.2.2.2.2.1 (outIdxs_lt hd h1 hh) (finIdxs_lt hd hh)
  · rw [slotW_out h2, show c = d.nchunks by omega, slotW_fin]
    exact (d.out.elems_disjoint_of d.finalst hd.disj.2.2.2.2.2.1 (outIdxs_lt hd h2 hh)
      (finIdxs_lt hd hh)).symm
  · exact absurd (by omega) hne

/-- A state element belongs to the lanes of one program only. -/
theorem lanes_unique {d : Inst} {m pm pm' : Nat} (h1 : m ∈ lanes d pm) (h2 : m ∈ lanes d pm') :
    pm = pm' := by
  obtain ⟨a1, a2, -⟩ := h1
  obtain ⟨b1, b2, -⟩ := h2
  rw [BL_eq] at *
  omega

/-- Slots of distinct programs are disjoint. -/
theorem slotW_prog_disjoint {d : Inst} (hd : d.ok) {h h' pm pm' c c' : Nat} (hh : h < d.nheads)
    (hh' : h' < d.nheads) (hc : c ≤ d.nchunks) (hc' : c' ≤ d.nchunks) (hne : (h, pm) ≠ (h', pm')) :
    Disjoint (slotW d h pm c) (slotW d h' pm' c') := by
  by_cases h1 : c < d.nchunks <;> by_cases h2 : c' < d.nchunks
  · rw [slotW_out h1, slotW_out h2]
    refine d.out.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro j ⟨m, ⟨a1, a2, hm⟩, rfl⟩ ⟨m', ⟨b1, b2, hm'⟩, e⟩
    obtain ⟨-, rfl, rfl⟩ := stIdx_inj hh hh' hm hm' e
    exact hne (by rw [lanes_unique (d := d) ⟨a1, a2, hm⟩ ⟨b1, b2, hm⟩])
  · rw [slotW_out h1, show c' = d.nchunks by omega, slotW_fin]
    exact d.out.elems_disjoint_of d.finalst hd.disj.2.2.2.2.2.1 (outIdxs_lt hd h1 hh) (finIdxs_lt hd hh')
  · rw [slotW_out h2, show c = d.nchunks by omega, slotW_fin]
    exact (d.out.elems_disjoint_of d.finalst hd.disj.2.2.2.2.2.1 (outIdxs_lt hd h2 hh')
      (finIdxs_lt hd hh)).symm
  · rw [show c = d.nchunks by omega, show c' = d.nchunks by omega, slotW_fin, slotW_fin]
    refine d.finalst.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro j ⟨m, ⟨a1, a2, hm⟩, rfl⟩ ⟨m', ⟨b1, b2, hm'⟩, e⟩
    obtain ⟨rfl, rfl⟩ := finIdx_inj hm hm' e
    exact hne (by rw [lanes_unique (d := d) ⟨a1, a2, hm⟩ ⟨b1, b2, hm⟩])

/-- A slot lies in the outputs' bytes. -/
theorem slotW_sub {d : Inst} (hd : d.ok) {h pm c : Nat} (hh : h < d.nheads) (hc : c ≤ d.nchunks) :
    slotW d h pm c ⊆ d.out.bytes ∨ slotW d h pm c ⊆ d.finalst.bytes := by
  by_cases h1 : c < d.nchunks
  · rw [slotW_out h1]; exact Or.inl (d.out.elems_sub_bytes (outIdxs_lt hd h1 hh))
  · rw [show c = d.nchunks by omega, slotW_fin]
    exact Or.inr (d.finalst.elems_sub_bytes (finIdxs_lt hd hh))

/-- No slot overlaps an input. -/
theorem slotW_disjoint_Rin {d : Inst} (hd : d.ok) {h pm c : Nat} (hh : h < d.nheads)
    (hc : c ≤ d.nchunks) : Disjoint (slotW d h pm c) (Rin d) := by
  obtain ⟨o1, o2, o3, o4, o5, -, f1, f2, f3, f4, f5⟩ := hd.disj
  unfold Rin
  rcases slotW_sub hd hh hc with hs | hs
  · exact ((((o1.mono_left hs).union_right (o2.mono_left hs)).union_right (o3.mono_left hs)).union_right
      (o4.mono_left hs)).union_right (o5.mono_left hs)
  · exact ((((f1.mono_left hs).union_right (f2.mono_left hs)).union_right (f3.mono_left hs)).union_right
      (f4.mono_left hs)).union_right (f5.mono_left hs)

end Claims.SsdStatePassing
