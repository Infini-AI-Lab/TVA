import Proof.Program

/-!
# `combine_draft_tokens_i32`: the footprints of distinct programs, and `Post`

Distinct programs write disjoint elements: their logits indices lie in disjoint ranges of
`cu_num_logits` (`log_range`), their `input_ids` slots in disjoint query ranges (`ids_range`).
No program writes an input. `post_of` assembles `Post` from the programs' post-conditions.
-/

namespace Claims.CombineDraftTokens

open TVA

/-- The write footprint of program `b`, as the two outputs' parts. -/
theorem W_eq (d : Inst) (init : Mem) (b : Nat) :
    W d init b = d.logits_indices.elems (logE d init b) ∪
      d.input_ids.elems (lastE d init b ∪ draftE d init b) := by
  rw [W, Buf.elems_union]

/-- No program writes an input. -/
theorem R_disj_W {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.B) :
    Disjoint (R d) (W d init b) := by
  rw [W_eq]
  have hl := log_sub hpre hb
  have hi : d.input_ids.elems (lastE d init b ∪ draftE d init b) ⊆ d.input_ids.bytes := by
    rw [Buf.elems_union]; exact ids_sub hd hpre hb
  obtain ⟨-, i1, i2, i3, i4, i5, i6, i7⟩ := hd.hids
  obtain ⟨l1, l2, l3, l4, l5, l6, l7⟩ := hd.hlog
  refine Disjoint.union_right ?_ ?_
  · refine (Disjoint.mono_right hl ?_)
    simp only [R]
    exact Disjoint.union_left (Disjoint.union_left (Disjoint.union_left (Disjoint.union_left
      (Disjoint.union_left (Disjoint.union_left l1.symm l2.symm) l3.symm) l4.symm) l5.symm)
      l6.symm) l7.symm
  · refine (Disjoint.mono_right hi ?_)
    simp only [R]
    exact Disjoint.union_left (Disjoint.union_left (Disjoint.union_left (Disjoint.union_left
      (Disjoint.union_left (Disjoint.union_left i1.symm i2.symm) i3.symm) i4.symm) i5.symm)
      i6.symm) i7.symm

/-- Distinct programs write disjoint bytes. -/
theorem W_disj {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b b' : Nat}
    (hb : b < d.B) (hb' : b' < d.B) (hne : b ≠ b') : Disjoint (W d init b) (W d init b') := by
  rw [W_eq, W_eq]
  have hl := log_sub hpre hb
  have hl' := log_sub hpre hb'
  have hi : d.input_ids.elems (lastE d init b ∪ draftE d init b) ⊆ d.input_ids.bytes := by
    rw [Buf.elems_union]; exact ids_sub hd hpre hb
  have hi' : d.input_ids.elems (lastE d init b' ∪ draftE d init b') ⊆ d.input_ids.bytes := by
    rw [Buf.elems_union]; exact ids_sub hd hpre hb'
  have hli := hd.hids.1
  -- the order of the two requests
  have hlog : Disjoint (logE d init b) (logE d init b') := by
    refine Set.disjoint_left.mpr fun t ht ht' => ?_
    have h1 := log_range ht
    have h2 := log_range ht'
    rcases Nat.lt_or_gt_of_ne hne with h | h
    · have := cu_mono hpre (show b + 1 ≤ b' by omega) (by omega); omega
    · have := cu_mono hpre (show b' + 1 ≤ b by omega) (by omega); omega
  have hids : Disjoint (lastE d init b ∪ draftE d init b) (lastE d init b' ∪ draftE d init b') := by
    refine Set.disjoint_left.mpr fun t ht ht' => ?_
    have h1 := ids_range hd hpre hb ht
    have h2 := ids_range hd hpre hb' ht'
    rcases Nat.lt_or_gt_of_ne hne with h | h
    · have := qsl_mono hpre (show b + 1 ≤ b' by omega) (by omega)
      simp only [qend] at h1 h2; omega
    · have := qsl_mono hpre (show b' + 1 ≤ b by omega) (by omega)
      simp only [qend] at h1 h2; omega
  refine Disjoint.union_left (Disjoint.union_right (d.logits_indices.elems_disjoint hlog) ?_)
    (Disjoint.union_right ?_ (d.input_ids.elems_disjoint hids))
  · exact hli.symm.mono hl hi'
  · exact (hli.mono hi hl')

/-- A read of an element whose bytes lie in a footprint on which `fin` and `m` agree. -/
theorem read_of_agree {dt : DType} {b : Buf dt} {fin m : Mem} {S : Set Nat} {T : Set Int}
    {t : Nat} (ht : t ∈ S) (hS : b.elems S ⊆ T) (h : ∀ x ∈ T, fin x = m x) :
    fin.read dt (b.ptr (t : Int)).addr = m.read dt (b.ptr (t : Int)).addr :=
  Mem.read_congr fun x hx => h x (hS (b.span_sub_elems ht hx))

/-- **`Post`** from the programs' post-conditions. -/
theorem post_of {d : Inst} {init : Mem} (hpre : Pre d init) (fin : Mem)
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init (i : Nat) m ∧
      ∀ x ∈ R d ∪ W d init (i : Nat), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init (i : Nat)) → fin x = init x) :
    Post d init fin := by
  have hloc' : ∀ b, b < d.B → ∃ m, Q d init b m ∧ ∀ x ∈ W d init b, fin x = m x := by
    intro b hb
    obtain ⟨m, hQ, h⟩ := hloc ⟨b, by rw [launch_size]; exact hb⟩
    exact ⟨m, hQ, fun x hx => h x (Or.inr hx)⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro b hb j hj
    obtain ⟨m, ⟨hQL, -, -⟩, h⟩ := hloc' b hb
    obtain ⟨w, hw, hv⟩ := hQL j hj
    have hc0 := (cu_bounds hpre hb).1
    have hk : ((cuAt d init b + j).toNat : Int) = cuAt d init b + j := Int.toNat_of_nonneg (by omega)
    have hmem : (cuAt d init b + j).toNat ∈ logE d init b := ⟨j, hj, hk⟩
    refine ⟨w, ?_, hv⟩
    rw [← hk] at hw ⊢
    rw [read_of_agree hmem (Set.subset_union_left) h, hw]
  · intro b hb hlast
    obtain ⟨m, ⟨-, hQA, -⟩, h⟩ := hloc' b hb
    obtain ⟨w, hw, hv⟩ := hQA hlast
    obtain ⟨-, -, -, -, hNl, -, hq, -⟩ := hpre.2.2.2.2.2.2.2.2.2.1 b hb
    have hq0 := (qsl_bounds hpre hb).1
    have hk : ((qend d init b - nlog d init b).toNat : Int) = qend d init b - nlog d init b :=
      Int.toNat_of_nonneg (by omega)
    have hmem : (qend d init b - nlog d init b).toNat ∈ lastE d init b := ⟨hlast, hk⟩
    refine ⟨w, ?_, hv⟩
    rw [← hk] at hw ⊢
    rw [read_of_agree hmem (Set.subset_union_left.trans Set.subset_union_right) h, hw]
  · intro b hb hdec j hj
    obtain ⟨m, ⟨-, -, hQD⟩, h⟩ := hloc' b hb
    obtain ⟨w, hw, hv⟩ := hQD hdec j hj
    obtain ⟨-, -, -, -, hNl, -, hq, -⟩ := hpre.2.2.2.2.2.2.2.2.2.1 b hb
    have hq0 := (qsl_bounds hpre hb).1
    have hk : ((qend d init b - ndraft d init b + j).toNat : Int) =
        qend d init b - ndraft d init b + j :=
      Int.toNat_of_nonneg (by simp only [ndraft]; omega)
    have hmem : (qend d init b - ndraft d init b + j).toNat ∈ draftE d init b := ⟨hdec, j, hj, hk⟩
    refine ⟨w, ?_, hv⟩
    rw [← hk] at hw ⊢
    rw [read_of_agree hmem (Set.subset_union_right.trans Set.subset_union_right) h, hw]
  · intro z hz1 hz2
    refine hfr z fun i hz => ?_
    have hi : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
    rcases hz with hz | hz | hz
    · exact hz1 (d.logits_indices.elems_mono
        (show logE d init i ⊆ logOut d init from fun t ht => ⟨(i : Nat), hi, ht⟩) hz)
    · exact hz2 (d.input_ids.elems_mono
        (show lastE d init i ⊆ idsOut d init from fun t ht => ⟨(i : Nat), hi, Or.inl ht⟩) hz)
    · exact hz2 (d.input_ids.elems_mono
        (show draftE d init i ⊆ idsOut d init from fun t ht => ⟨(i : Nat), hi, Or.inr ht⟩) hz)

end Claims.CombineDraftTokens
