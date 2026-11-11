import Claim
import TVA.Proof
import Proof.Safe

/-!
# Distinct programs write distinct elements

`tok_ne`: two programs of the same value head never write the same token: their chunks are
different 64-token blocks of the batch (padded: a different batch row or chunk; packed: a
different chunk of one sequence, or another sequence). Hence their footprints are disjoint.
-/

namespace Claims.RecomputeWU

open TVA

/-- Program `i` is `p0 + g0 · bh`. -/
theorem prog_eq {d : Inst} {i : Nat} : i = p0Of d i + g0 d * bhOf d i ∨ g0 d * (d.B * d.HV) ≤ i := by
  by_cases hi : i < g0 d * (d.B * d.HV)
  · left
    have hg : 0 < g0 d := by
      rcases Nat.eq_zero_or_pos (g0 d) with h | h
      · rw [h] at hi; simp at hi
      · exact h
    unfold p0Of bhOf
    have : i / g0 d < d.B * d.HV := (Nat.div_lt_iff_lt_mul hg).mpr (by rw [Nat.mul_comm]; exact hi)
    rw [Nat.mod_eq_of_lt this]
    have := Nat.mod_add_div i (g0 d)
    linarith [Nat.mul_comm (g0 d) (i / g0 d)]
  · right; omega

/-- **Two different programs of one value head write different tokens.** -/
theorem tok_ne {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i j : Nat}
    (hi : i < g0 d * (d.B * d.HV)) (hj : j < g0 d * (d.B * d.HV)) (hij : i ≠ j)
    (hh : hOf d i = hOf d j) {r r' : Nat} (hr : r < 64) (hr' : r' < 64)
    (hrow : itOf d init i * 64 + r < lenOf d init i)
    (hrow' : itOf d init j * 64 + r' < lenOf d init j) :
    s0Of d init i + itOf d init i * 64 + r ≠ s0Of d init j + itOf d init j * 64 + r' := by
  intro heq
  obtain ⟨hp, hb, hhv, hbB⟩ := coords hd hi
  obtain ⟨hp', hb', hhv', hbB'⟩ := coords hd hj
  have ei := prog_eq (d := d) (i := i)
  have ej := prog_eq (d := d) (i := j)
  simp only [show ¬ (g0 d * (d.B * d.HV) ≤ i) by omega, or_false] at ei
  simp only [show ¬ (g0 d * (d.B * d.HV) ≤ j) by omega, or_false] at ej
  have hbh_def : ∀ k, bhOf d k = d.HV * bOf d k + hOf d k := fun k => by
    unfold bOf hOf; exact (Nat.div_add_mod _ _).symm
  by_cases hv : d.isVarlen = true
  · -- packed: `B = 1`, so the head is the whole of `bh`
    have hB := hd.varlen_b hv
    have hb1 : bhOf d i < d.HV := by rw [hB] at hb; simpa using hb
    have hb2 : bhOf d j < d.HV := by rw [hB] at hb'; simpa using hb'
    have hbh : bhOf d i = bhOf d j := by
      unfold hOf at hh; rw [Nat.mod_eq_of_lt hb1, Nat.mod_eq_of_lt hb2] at hh; exact hh
    have hp0 : p0Of d i ≠ p0Of d j := by
      intro hp0; apply hij; rw [ei, ej, hp0, hbh]
    have hS := hpre.seqs hv
    have hNT : p0Of d i < d.NT := by rw [← g0_varlen hv]; exact hp
    have hNT' : p0Of d j < d.NT := by rw [← g0_varlen hv]; exact hp'
    obtain ⟨hn, hc⟩ := hS.ci_ok _ hNT
    obtain ⟨hn', hc'⟩ := hS.ci_ok _ hNT'
    simp only [itOf, s0Of, lenOf, hv, ite_true] at heq hrow hrow'
    by_cases hseq : chSeq d init (p0Of d i) = chSeq d init (p0Of d j)
    · rw [hseq] at heq
      have hch : chChunk d init (p0Of d i) ≠ chChunk d init (p0Of d j) := by
        intro hch
        exact hp0 (hS.ci_once _ _ hNT hNT' hseq hch)
      omega
    · rcases Nat.lt_or_gt_of_ne hseq with hlt | hgt
      · have h1 := cuN_mono hS (chSeq d init (p0Of d i) + 1) (chSeq d init (p0Of d j)) (by omega)
          (by omega)
        have h2 := seqLen_add hS hn
        unfold seqLen at hrow
        omega
      · have h1 := cuN_mono hS (chSeq d init (p0Of d j) + 1) (chSeq d init (p0Of d i)) (by omega)
          (by omega)
        have h2 := seqLen_add hS hn'
        unfold seqLen at hrow'
        omega
  · have hv' : d.isVarlen = false := by simpa using hv
    simp only [itOf, s0Of, lenOf, hv', Bool.false_eq_true, ite_false] at heq hrow hrow'
    by_cases hbo : bOf d i = bOf d j
    · have hbh : bhOf d i = bhOf d j := by rw [hbh_def i, hbh_def j, hbo, hh]
      have hp0 : p0Of d i ≠ p0Of d j := by
        intro hp0; apply hij; rw [ei, ej, hp0, hbh]
      rw [hbo] at heq
      omega
    · rcases Nat.lt_or_gt_of_ne hbo with hlt | hgt
      · have : (bOf d i + 1) * d.T ≤ bOf d j * d.T := Nat.mul_le_mul_right _ hlt
        nlinarith
      · have : (bOf d j + 1) * d.T ≤ bOf d i * d.T := Nat.mul_le_mul_right _ hgt
        nlinarith


/-- A program's value head is below `HV`. -/
theorem hOf_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : hOf d i < d.HV := by
  have hi : (i : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq i.isLt (launch_size_eq d); omega
  exact (coords hd hi).2.2.1

/-- `geo` for a program of the launch. -/
theorem geo' {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size) :
    itOf d init i * 64 < lenOf d init i ∧ s0Of d init i + lenOf d init i ≤ nTok d := by
  have hi : (i : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq i.isLt (launch_size_eq d); omega
  exact geo hd hpre hi

/-- Two different programs write different elements of `u`. -/
theorem UE_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (UE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i))
      (UE d (s0Of d init j) (lenOf d init j) (itOf d init j) (hOf d j)) := by
  rw [Set.disjoint_left]
  rintro x ⟨r, c, hr, hrow, hc, rfl⟩ ⟨r', c', hr', hrow', hc', e⟩
  have := idx_inj (hOf_lt hd i) (hOf_lt hd j) hc hc' e
  have hi : (i : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq i.isLt (launch_size_eq d); omega
  have hj : (j : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq j.isLt (launch_size_eq d); omega
  exact tok_ne hd hpre hi hj (fun h => hij (Fin.ext h)) this.2.1 hr hr' hrow hrow' this.1

/-- Two different programs write different elements of `w`. -/
theorem WE_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (WE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i))
      (WE d (s0Of d init j) (lenOf d init j) (itOf d init j) (hOf d j)) := by
  rw [Set.disjoint_left]
  rintro x ⟨r, c, hr, hrow, hc, rfl⟩ ⟨r', c', hr', hrow', hc', e⟩
  have := idx_inj (hOf_lt hd i) (hOf_lt hd j) hc hc' e
  have hi : (i : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq i.isLt (launch_size_eq d); omega
  have hj : (j : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq j.isLt (launch_size_eq d); omega
  exact tok_ne hd hpre hi hj (fun h => hij (Fin.ext h)) this.2.1 hr hr' hrow hrow' this.1

/-- The elements of `u` of one program and of `w` of another are disjoint bytes. -/
theorem UE_WE_disjoint' {d : Inst} (hd : d.ok) {init : Mem} (i j : Fin (launch d).size)
    (hpre : Pre d init) :
    Disjoint (d.u.elems (UE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i)))
      (d.w.elems (WE d (s0Of d init j) (lenOf d init j) (itOf d init j) (hOf d j))) :=
  d.u.elems_disjoint_of d.w hd.w_u.symm (UE_lt hd (geo' hd hpre i).2 (hOf_lt hd i))
    (WE_lt hd (geo' hd hpre j).2 (hOf_lt hd j))

/-- **Different programs write disjoint bytes.** -/
theorem Wf_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) : Disjoint (Wf d init i) (Wf d init j) := by
  unfold Wf
  rw [Set.disjoint_union_left, Set.disjoint_union_right, Set.disjoint_union_right]
  refine ⟨⟨d.u.elems_disjoint (UE_disjoint hd hpre hij), UE_WE_disjoint' hd i j hpre⟩,
    (UE_WE_disjoint' hd j i hpre).symm, d.w.elems_disjoint (WE_disjoint hd hpre hij)⟩

/-- A program writes none of the inputs. -/
theorem Wf_disjoint_R {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) : Disjoint (Wf d init i) (Rall d) := by
  unfold Wf
  rw [Set.disjoint_union_left]
  exact ⟨Set.disjoint_of_subset_left (d.u.elems_sub_bytes (UE_lt hd (geo' hd hpre i).2
      (hOf_lt hd i))) (u_disj_R hd),
    Set.disjoint_of_subset_left (d.w.elems_sub_bytes (WE_lt hd (geo' hd hpre i).2
      (hOf_lt hd i))) (w_disj_R hd)⟩

end Claims.RecomputeWU
