import Claim
import TVA.Proof
import Proof.Foot

/-!
# The programs' results give `Post`

`post_of_Q`: a final memory that agrees with each program's memory on its footprint, and with
`init` where no program writes, satisfies `Post`: every token of every sequence is a row of some
program's chunk, and the cells it wrote are read back in the final memory.
-/

namespace Claims.RecomputeWU

open TVA

/-- A `u` cell read in the final memory is the one program's memory holds, as the final memory agrees with it on the footprint. -/
theorem cellU_transfer {d : Inst} {init m fin : Mem} {S : Set Int}
    (hagr : ∀ x ∈ S, fin x = m x) {s0 len it h r c : Nat} (hc : c < d.V)
    (hr : r < 64) (hrow : it * 64 + r < len)
    (hS : d.u.elems (UE d s0 len it h) ⊆ S)
    (hcell : CellU d init m s0 len (it * 64 + r) h c) : CellU d init fin s0 len (it * 64 + r) h c := by
  obtain ⟨o, ho, hf, hb⟩ := hcell
  refine ⟨o, ?_, hf, hb⟩
  unfold bfAt at ho ⊢
  rw [Mem.read_congr (m₂ := m) fun x hx => hagr x (hS ?_)]
  · exact ho
  · exact d.u.span_sub_elems (S := UE d s0 len it h) (j := vIdx d (s0 + (it * 64 + r)) h c)
      ⟨r, c, hr, hrow, hc, by unfold vIdx; ring_nf⟩ hx

/-- A `w` cell read in the final memory is the one program's memory holds. -/
theorem cellW_transfer {d : Inst} {init m fin : Mem} {S : Set Int}
    (hagr : ∀ x ∈ S, fin x = m x) {s0 len it h r c : Nat} (hc : c < d.K)
    (hr : r < 64) (hrow : it * 64 + r < len)
    (hS : d.w.elems (WE d s0 len it h) ⊆ S)
    (hcell : CellW d init m s0 len (it * 64 + r) h c) : CellW d init fin s0 len (it * 64 + r) h c := by
  obtain ⟨o, ho, hf, hb⟩ := hcell
  refine ⟨o, ?_, hf, hb⟩
  unfold bfAt at ho ⊢
  rw [Mem.read_congr (m₂ := m) fun x hx => hagr x (hS ?_)]
  · exact ho
  · exact d.w.span_sub_elems (S := WE d s0 len it h) (j := wIdx d (s0 + (it * 64 + r)) h c)
      ⟨r, c, hr, hrow, hc, by unfold wIdx; ring_nf⟩ hx

/-- Program `i`'s results, in a memory that agrees with its own on its footprint. -/
theorem tokOk_of {d : Inst} {init m fin : Mem} {i : Fin (launch d).size}
    (hQ : Qf d init i m) (hagr : ∀ x ∈ Wf d init i, fin x = m x) {r : Nat} (hr : r < 64)
    (hrow : itOf d init i * 64 + r < lenOf d init i) :
    TokOk d init fin (s0Of d init i) (lenOf d init i) (itOf d init i * 64 + r) (hOf d i) := by
  obtain ⟨hu, hw⟩ := hQ r hr hrow
  exact ⟨fun c hc => cellU_transfer hagr hc hr hrow
      (fun x hx => Or.inl hx) (hu c hc),
    fun c hc => cellW_transfer hagr hc hr hrow (fun x hx => Or.inr hx) (hw c hc)⟩


/-- A program's footprint lies in the elements of `u` and `w` that make up the tensors. -/
theorem Wf_sub {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size) :
    Wf d init i ⊆ d.u.elems (uSet d) ∪ d.w.elems (wSet d) := by
  have hs := (geo' hd hpre i).2
  have hh := hOf_lt hd i
  unfold Wf
  refine Set.union_subset_union (d.u.elems_mono ?_) (d.w.elems_mono ?_)
  · rintro x ⟨r, c, hr, hrow, hc, rfl⟩
    exact idx_lt (n := nTok d) (by omega) hh hc
  · rintro x ⟨r, c, hr, hrow, hc, rfl⟩
    exact idx_lt (n := nTok d) (by omega) hh hc

/-- **The programs' results give `Post`.** -/
theorem post_of_Q {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Qf d init i m ∧ ∀ x ∈ Rall d ∪ Wf d init i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ Wf d init i) → fin x = init x) :
    Post d init fin := by
  refine ⟨fun hv b t h hb ht hh => ?_, fun hv n t h hn ht hh => ?_, fun z hu hw => hfr z fun i hz => ?_⟩
  · -- padded
    have hg : 0 < g0 d := by
      rw [g0_padded hv]; omega
    have hp : t / 64 < g0 d := by rw [g0_padded hv]; omega
    have hbh : b * d.HV + h < d.B * d.HV := by nlinarith
    have hi : t / 64 + g0 d * (b * d.HV + h) < g0 d * (d.B * d.HV) := by nlinarith
    let i : Fin (launch d).size := ⟨t / 64 + g0 d * (b * d.HV + h), by
      rw [launch_size_eq]; omega⟩
    have e0 : p0Of d i = t / 64 := by
      unfold p0Of
      show (t / 64 + g0 d * (b * d.HV + h)) % g0 d = _
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
    have e1 : bhOf d i = b * d.HV + h := by
      unfold bhOf
      show (t / 64 + g0 d * (b * d.HV + h)) / g0 d % (d.B * d.HV) = _
      rw [Nat.add_mul_div_left _ _ hg, Nat.div_eq_of_lt hp, zero_add, Nat.mod_eq_of_lt hbh]
    have eh : hOf d i = h := by
      unfold hOf; rw [e1]
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh]
    have eb : bOf d i = b := by
      unfold bOf; rw [e1, Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hh,
        zero_add]
    have eit : itOf d init i = t / 64 := by simp [itOf, hv, e0]
    have es0 : s0Of d init i = b * d.T := by simp [s0Of, hv, eb]
    have elen : lenOf d init i = d.T := by simp [lenOf, hv]
    obtain ⟨m, hQ, hagr⟩ := hloc i
    have := tokOk_of hQ (fun x hx => hagr x (Or.inr hx)) (r := t % 64) (Nat.mod_lt _ (by norm_num))
      (by rw [eit, elen]; omega)
    rw [eit, es0, elen, eh] at this
    have et : t / 64 * 64 + t % 64 = t := by omega
    rw [et] at this
    exact this
  · -- packed
    have hS := hpre.seqs hv
    obtain ⟨p, hpN, hps, hpc⟩ := hS.ci_all n (t / 64) hn (by unfold seqLen at ht ⊢; omega)
    have hg : g0 d = d.NT := g0_varlen hv
    have hB := hd.varlen_b hv
    have hp : p < g0 d := by rw [hg]; exact hpN
    have hbh : h < d.B * d.HV := by rw [hB]; simpa using hh
    have hi : p + g0 d * h < g0 d * (d.B * d.HV) := by nlinarith
    let i : Fin (launch d).size := ⟨p + g0 d * h, by rw [launch_size_eq]; omega⟩
    have hg0 : 0 < g0 d := by omega
    have e0 : p0Of d i = p := by
      unfold p0Of
      show (p + g0 d * h) % g0 d = _
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
    have e1 : bhOf d i = h := by
      unfold bhOf
      show (p + g0 d * h) / g0 d % (d.B * d.HV) = _
      rw [Nat.add_mul_div_left _ _ hg0, Nat.div_eq_of_lt hp, zero_add, Nat.mod_eq_of_lt hbh]
    have eh : hOf d i = h := by unfold hOf; rw [e1, Nat.mod_eq_of_lt hh]
    have eit : itOf d init i = t / 64 := by simp [itOf, hv, e0, hpc]
    have es0 : s0Of d init i = cuN d init n := by simp [s0Of, hv, e0, hps]
    have elen : lenOf d init i = seqLen d init n := by simp [lenOf, hv, e0, hps]
    obtain ⟨m, hQ, hagr⟩ := hloc i
    have := tokOk_of hQ (fun x hx => hagr x (Or.inr hx)) (r := t % 64) (Nat.mod_lt _ (by norm_num))
      (by rw [eit, elen]; omega)
    rw [eit, es0, elen, eh] at this
    have et : t / 64 * 64 + t % 64 = t := by omega
    rw [et] at this
    exact this
  · exact (Wf_sub hd hpre i hz).elim hu hw

end Claims.RecomputeWU
