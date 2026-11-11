import Claim
import TVA.Proof
import Proof.Defs

/-!
# The programs' local results give `Post`

Each `(t, h)` is the token and head of exactly one program, and the bytes that program touches
contain its lse elements and its rows. So the local postconditions `Qm` give `MergeOk` on the
merged tokens, `CopyOk` after them, and the frame elsewhere.
-/

namespace Claims.MergeAttnStates

open TVA

/-- The program that handles token `t` and head `h`. -/
def pidx (d : Inst) (t h : Nat) (ht : t < d.T) (hh : h < d.H) : Fin (launch d).size :=
  ⟨h * d.T + t, by
    rw [launch_size]
    have h1 : h * d.T + t < h * d.T + d.T := Nat.add_lt_add_left ht _
    have h2 : h * d.T + d.T = (h + 1) * d.T := by rw [Nat.add_mul, Nat.one_mul]
    have h3 : (h + 1) * d.T ≤ d.H * d.T := Nat.mul_le_mul_right _ hh
    rw [Nat.mul_comm d.T d.H]
    exact lt_of_lt_of_le (h2 ▸ h1) h3⟩

/-- The program `pidx d t h` handles token `t`. -/
theorem pidx_pidT (d : Inst) (t h : Nat) (ht : t < d.T) (hh : h < d.H) :
    pidT d (pidx d t h ht hh) = t := by
  have hT : 0 < d.T := Nat.lt_of_le_of_lt (Nat.zero_le _) ht
  simp only [pidT, pidx]
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]

/-- The program `pidx d t h` handles head `h`. -/
theorem pidx_pidH (d : Inst) (t h : Nat) (ht : t < d.T) (hh : h < d.H) :
    pidH d (pidx d t h ht hh) = h := by
  have hT : 0 < d.T := Nat.lt_of_le_of_lt (Nat.zero_le _) ht
  simp only [pidH, pidx]
  rw [Nat.add_comm, Nat.add_mul_div_right t h hT, Nat.div_eq_of_lt ht, Nat.zero_add,
    Nat.mod_eq_of_lt hh]

/-! ## Reading through a pointwise agreement -/

/-- A `float32` read depends only on the bytes of its element's span. -/
theorem rdF_congr {m₁ m₂ : Mem} {b : Buf .f32} {i : Nat} {T : Set Int}
    (h : ∀ x ∈ T, m₁ x = m₂ x) (hsub : Mem.span .f32 (b.ptr (i : Int)).addr ⊆ T) :
    rdF m₁ b i = rdF m₂ b i := by
  unfold rdF
  exact Mem.read_congr (fun x hx => h x (hsub hx))

/-- A `bfloat16` read depends only on the bytes of its element's span. -/
theorem rdB_congr {m₁ m₂ : Mem} {b : Buf .bf16} {i : Nat} {T : Set Int}
    (h : ∀ x ∈ T, m₁ x = m₂ x) (hsub : Mem.span .bf16 (b.ptr (i : Int)).addr ⊆ T) :
    rdB m₁ b i = rdB m₂ b i := by
  unfold rdB
  exact Mem.read_congr (fun x hx => h x (hsub hx))

/-- Element `j < head_size` of the output row of `(t, h)` lies in that row. -/
theorem oIdx_mem_row (d : Inst) (t h j : Nat) (hj : j < d.head_size) :
    oIdx d t h j ∈ row d (oIdx d t h 0) := by
  have hidx : oIdx d t h j = oIdx d t h 0 + j := by
    simp only [oIdx]
    omega
  simp only [row, Set.mem_ofPred_eq]
  omega

/-- The output row of `(t, h)` is a set of elements the launch writes. -/
theorem row_subset_outSet (d : Inst) (t h : Nat) (ht : t < d.T) (hh : h < d.H) :
    row d (oIdx d t h 0) ⊆ outSet d := by
  intro j hj
  simp only [row, Set.mem_ofPred_eq] at hj
  obtain ⟨hj1, hj2⟩ := hj
  have hjlt : j - oIdx d t h 0 < d.head_size := by omega
  refine ⟨t, h, j - oIdx d t h 0, ht, hh, hjlt, ?_⟩
  have hgoal : oIdx d t h (j - oIdx d t h 0) = j := by
    simp only [oIdx] at hj1 ⊢
    omega
  exact hgoal.symm

/-- The lse element of `(t, h)` is an element the launch writes. -/
theorem olIdx_mem_outLseSet (d : Inst) (t h : Nat) (ht : t < d.T) (hh : h < d.H) :
    olIdx d t h ∈ outLseSet d :=
  ⟨t, h, ht, hh, rfl⟩

/-- **The local results give `Post`.** -/
theorem post_of_local {d : Inst} (hd : d.ok) {init : Mem} (_hpre : Pre d init) {fin : Mem}
    (hloc : ∀ i : Fin (launch d).size,
      ∃ m, Qq d init i m ∧ ∀ x ∈ Rq d init i ∪ Wq d init i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ Wq d init i) → fin x = init x) :
    Post d init fin := by
  -- the local result of the program of each `(t, h)`, in the footprints `Rf`/`Wf`
  have hmem : ∀ (t h : Nat) (ht : t < d.T) (hh : h < d.H),
      ∃ m, Qm d init t h m ∧ ∀ x ∈ Rf d t h ∪ Wf d t h, fin x = m x := by
    intro t h ht hh
    obtain ⟨m, hQ, hagr⟩ := hloc (pidx d t h ht hh)
    have h1 := pidx_pidT d t h ht hh
    have h2 := pidx_pidH d t h ht hh
    refine ⟨m, ?_, ?_⟩
    · simpa [Qq, h1, h2] using hQ
    · intro x hx
      exact hagr x (by simpa [Rq, Wq, h1, h2] using hx)
  refine ⟨?_, ?_, ?_⟩
  · -- the merged tokens
    intro t h ht hh
    have htT : t < d.T := lt_of_lt_of_le ht hd.ptwc_le
    obtain ⟨m, hQ, hagr⟩ := hmem t h htT hh
    have hM : MergeOk d init m t h := by
      have h := hQ
      simp only [Qm] at h
      rwa [ite_eq_left ht] at h
    obtain ⟨p, s, hp, hs, hlse, hout⟩ := hM
    refine ⟨p, s, hp, hs, ?_, ?_⟩
    · intro ho
      obtain ⟨cl, hcl, hbe, hval⟩ := hlse ho
      refine ⟨cl, ?_, hbe, hval⟩
      have hsub : Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr ⊆
          Rf d t h ∪ Wf d t h := by
        intro x hx
        have hRe : RlseOut d t h =
            Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr := by
          simp [RlseOut, ho]
        simp only [Wf, Set.mem_union]
        exact Or.inr (Or.inr (by rw [hRe]; exact hx))
      exact (rdF_congr hagr hsub).trans hcl
    · intro j hj
      obtain ⟨w, hw, hbe, hval⟩ := hout j hj
      refine ⟨w, ?_, hbe, hval⟩
      have hsub : Mem.span .bf16 (d.output.ptr (oIdx d t h j : Int)).addr ⊆
          Rf d t h ∪ Wf d t h := by
        intro x hx
        simp only [Wf, Set.mem_union]
        exact Or.inr (Or.inl (Buf.span_sub_elems d.output (oIdx_mem_row d t h j hj) hx))
      exact (rdB_congr hagr hsub).trans hw
  · -- the copied tokens
    intro t h hpt ht hh
    obtain ⟨m, hQ, hagr⟩ := hmem t h ht hh
    have hC : CopyOk d init m t h := by
      have h := hQ
      simp only [Qm] at h
      rwa [ite_eq_right (not_lt.mpr hpt)] at h
    obtain ⟨hlse, hout⟩ := hC
    refine ⟨?_, ?_⟩
    · intro ho
      obtain ⟨s, hs1, hs2⟩ := hlse ho
      refine ⟨s, hs1, ?_⟩
      have hsub : Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr ⊆
          Rf d t h ∪ Wf d t h := by
        intro x hx
        have hRe : RlseOut d t h =
            Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr := by
          simp [RlseOut, ho]
        simp only [Wf, Set.mem_union]
        exact Or.inr (Or.inr (by rw [hRe]; exact hx))
      exact (rdF_congr hagr hsub).trans hs2
    · intro j hj
      obtain ⟨v, hv1, hv2⟩ := hout j hj
      refine ⟨v, hv1, ?_⟩
      have hsub : Mem.span .bf16 (d.output.ptr (oIdx d t h j : Int)).addr ⊆
          Rf d t h ∪ Wf d t h := by
        intro x hx
        simp only [Wf, Set.mem_union]
        exact Or.inr (Or.inl (Buf.span_sub_elems d.output (oIdx_mem_row d t h j hj) hx))
      exact (rdB_congr hagr hsub).trans hv2
  · -- the frame
    intro z hz
    apply hfr z
    intro i hzi
    have hrow : row d (oIdx d (pidT d i) (pidH d i) 0) ⊆ outSet d :=
      row_subset_outSet d (pidT d i) (pidH d i) (pidT_lt d i) (pidH_lt d i)
    have hRout : Rout d (pidT d i) (pidH d i) ⊆ writeSet d := by
      intro x hx
      simp only [writeSet, Set.mem_union]
      exact Or.inl (Buf.elems_mono d.output hrow hx)
    have hRlse : RlseOut d (pidT d i) (pidH d i) ⊆ writeSet d := by
      intro x hx
      cases ho : d.outLse
      · exfalso
        simp [RlseOut, ho] at hx
      · refine Or.inr ?_
        have hx' : x ∈
            Mem.span .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
          simpa [RlseOut, ho] using hx
        have hw : x ∈ d.output_lse.elems (outLseSet d) :=
          Buf.span_sub_elems d.output_lse
            (olIdx_mem_outLseSet d (pidT d i) (pidH d i) (pidT_lt d i) (pidH_lt d i)) hx'
        simpa [ho] using hw
    have hW : z ∈ Wf d (pidT d i) (pidH d i) := hzi
    rcases hW with h | h
    · exact hz (hRout h)
    · exact hz (hRlse h)

end Claims.MergeAttnStates
