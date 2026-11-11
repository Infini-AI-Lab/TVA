import Claim
import TVA.Proof
import Proof.Defs

/-!
# The programs' footprints are disjoint

Every program writes its own row of `output` and its own element of `output_lse`, and reads the
four input buffers plus the lse elements of its own `(t, h)`.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## Bounds: each footprint lies in the bytes of its buffers -/

/-- A row of `output` lies in the buffer's bytes. -/
private theorem out_elems_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) :
    d.output.elems (row d (oIdx d t h 0)) ⊆ d.output.bytes := by
  refine Buf.elems_sub_bytes d.output ?_
  intro j hj
  rw [row_iff] at hj
  obtain ⟨hj1, hj2⟩ := hj
  rcases Nat.eq_zero_or_pos d.head_size with h0 | hpos
  · omega
  · have hlen := hd.output_len (by omega) (by omega) hpos
    have h1 : t * d.ots ≤ (d.T - 1) * d.ots :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : h * d.ohs ≤ (d.H - 1) * d.ohs :=
      Nat.mul_le_mul_right _ (by omega)
    simp only [oIdx] at hj2
    omega

/-- A row of `prefix_output` lies in the buffer's bytes. -/
private theorem pfx_elems_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.ptwc) (hh : h < d.H) :
    d.prefix_output.elems (row d (poIdx d t h 0)) ⊆ d.prefix_output.bytes := by
  refine Buf.elems_sub_bytes d.prefix_output ?_
  intro j hj
  rw [row_iff] at hj
  obtain ⟨hj1, hj2⟩ := hj
  rcases Nat.eq_zero_or_pos d.head_size with h0 | hpos
  · omega
  · have hlen := hd.prefix_output_len (by omega) (by omega) hpos
    have h1 : t * d.pots ≤ (d.ptwc - 1) * d.pots :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : h * d.phs ≤ (d.H - 1) * d.phs :=
      Nat.mul_le_mul_right _ (by omega)
    simp only [poIdx] at hj2
    omega

/-- A row of `suffix_output` lies in the buffer's bytes. -/
private theorem sfx_elems_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) :
    d.suffix_output.elems (row d (soIdx d t h 0)) ⊆ d.suffix_output.bytes := by
  refine Buf.elems_sub_bytes d.suffix_output ?_
  intro j hj
  rw [row_iff] at hj
  obtain ⟨hj1, hj2⟩ := hj
  rcases Nat.eq_zero_or_pos d.head_size with h0 | hpos
  · omega
  · have hlen := hd.suffix_output_len (by omega) (by omega) hpos
    have h1 : t * d.sots ≤ (d.T - 1) * d.sots :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : h * d.shs ≤ (d.H - 1) * d.shs :=
      Nat.mul_le_mul_right _ (by omega)
    simp only [soIdx] at hj2
    omega

/-- An element of `prefix_lse` lies in the buffer's bytes. -/
private theorem pfx_lse_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.ptwc) (hh : h < d.H) :
    Mem.span .f32 (d.prefix_lse.ptr (plIdx d t h : Int)).addr ⊆ d.prefix_lse.bytes := by
  refine Buf.span_sub d.prefix_lse ?_
  have hlen := hd.prefix_lse_len (by omega) (by omega)
  have h1 : h * d.plhs ≤ (d.H - 1) * d.plhs :=
    Nat.mul_le_mul_right _ (by omega)
  have h2 : t * d.plts ≤ (d.ptwc - 1) * d.plts :=
    Nat.mul_le_mul_right _ (by omega)
  simp only [plIdx]
  omega

/-- An element of `suffix_lse` lies in the buffer's bytes. -/
private theorem sfx_lse_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) :
    Mem.span .f32 (d.suffix_lse.ptr (slIdx d t h : Int)).addr ⊆ d.suffix_lse.bytes := by
  refine Buf.span_sub d.suffix_lse ?_
  have hlen := hd.suffix_lse_len (by omega) (by omega)
  have h1 : h * d.slhs ≤ (d.H - 1) * d.slhs :=
    Nat.mul_le_mul_right _ (by omega)
  have h2 : t * d.slts ≤ (d.T - 1) * d.slts :=
    Nat.mul_le_mul_right _ (by omega)
  simp only [slIdx]
  omega

/-- An element of `output_lse` lies in the buffer's bytes. -/
private theorem olse_sub {d : Inst} (hd : d.ok) (ho : d.outLse) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) :
    Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr ⊆ d.output_lse.bytes := by
  refine Buf.span_sub d.output_lse ?_
  have hlen := hd.output_lse_len ho (by omega) (by omega)
  have h1 : h * d.olhs ≤ (d.H - 1) * d.olhs :=
    Nat.mul_le_mul_right _ (by omega)
  have h2 : t * d.olts ≤ (d.T - 1) * d.olts :=
    Nat.mul_le_mul_right _ (by omega)
  simp only [olIdx]
  omega

/-! ## The footprints are contained in the relevant buffers -/

/-- `Rout` lies in the bytes of `output`. -/
private theorem Rout_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) : Rout d t h ⊆ d.output.bytes :=
  out_elems_sub hd ht hh

/-- `RlseOut` lies in the bytes of `output_lse`. -/
private theorem RlseOut_sub {d : Inst} (hd : d.ok) (ho : d.outLse) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) : RlseOut d t h ⊆ d.output_lse.bytes := by
  unfold RlseOut
  rw [ite_eq_left ho]
  exact olse_sub hd ho ht hh

/-- `Rin` lies in the bytes of the two inputs. -/
private theorem Rin_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) :
    Rin d t h ⊆ d.prefix_output.bytes ∪ d.suffix_output.bytes := by
  unfold Rin
  refine Set.union_subset ?_ ?_
  · by_cases hp : t < d.ptwc
    · rw [ite_eq_left hp]
      exact (pfx_elems_sub hd hp hh).trans Set.subset_union_left
    · rw [ite_eq_right hp]
      exact Set.empty_subset _
  · exact (sfx_elems_sub hd ht hh).trans Set.subset_union_right

/-- `Rlse` lies in the bytes of the two lse inputs. -/
private theorem Rlse_sub {d : Inst} (hd : d.ok) {t h : Nat}
    (ht : t < d.T) (hh : h < d.H) :
    Rlse d t h ⊆ d.prefix_lse.bytes ∪ d.suffix_lse.bytes := by
  unfold Rlse
  refine Set.union_subset ?_ ?_
  · by_cases hp : t < d.ptwc
    · rw [ite_eq_left hp]
      exact (pfx_lse_sub hd hp hh).trans Set.subset_union_left
    · rw [ite_eq_right hp]
      exact Set.empty_subset _
  · exact (sfx_lse_sub hd ht hh).trans Set.subset_union_right

/-! ## Subset helpers for the four-way union in `output_disj` -/

/-- A set is inside a union of four. -/
private theorem four_a {A B C D : Set Int} : A ⊆ A ∪ B ∪ C ∪ D := by
  intro x hx; exact Or.inl (Or.inl (Or.inl hx))

/-- A set is inside a union of four. -/
private theorem four_b {A B C D : Set Int} : B ⊆ A ∪ B ∪ C ∪ D := by
  intro x hx; exact Or.inl (Or.inl (Or.inr hx))

/-- A set is inside a union of four. -/
private theorem four_c {A B C D : Set Int} : C ⊆ A ∪ B ∪ C ∪ D := by
  intro x hx; exact Or.inl (Or.inr hx)

/-- A set is inside a union of four. -/
private theorem four_d {A B C D : Set Int} : D ⊆ A ∪ B ∪ C ∪ D := by
  intro x hx; exact Or.inr hx

/-- Two sets are inside a union of four. -/
private theorem four_ab {A B C D : Set Int} : A ∪ B ⊆ A ∪ B ∪ C ∪ D :=
  Set.union_subset four_a four_b

/-- Two sets are inside a union of four. -/
private theorem four_cd {A B C D : Set Int} : C ∪ D ⊆ A ∪ B ∪ C ∪ D :=
  Set.union_subset four_c four_d

/-! ## Byte-level disjointness of the buffers -/

/-- `output` shares no byte with the two row inputs. -/
private theorem out_pfxsfx {d : Inst} (hd : d.ok) :
    Disjoint d.output.bytes (d.prefix_output.bytes ∪ d.suffix_output.bytes) :=
  hd.output_disj.mono_right four_ab

/-- `output` shares no byte with the two lse inputs. -/
private theorem out_plseslse {d : Inst} (hd : d.ok) :
    Disjoint d.output.bytes (d.prefix_lse.bytes ∪ d.suffix_lse.bytes) :=
  hd.output_disj.mono_right four_cd

/-- `output_lse` shares no byte with the two row inputs. -/
private theorem olse_pfxsfx {d : Inst} (hd : d.ok) (ho : d.outLse) :
    Disjoint d.output_lse.bytes (d.prefix_output.bytes ∪ d.suffix_output.bytes) :=
  (hd.output_lse_in_disj ho).mono_right four_ab

/-- `output_lse` shares no byte with the two lse inputs. -/
private theorem olse_plseslse {d : Inst} (hd : d.ok) (ho : d.outLse) :
    Disjoint d.output_lse.bytes (d.prefix_lse.bytes ∪ d.suffix_lse.bytes) :=
  (hd.output_lse_in_disj ho).mono_right four_cd

/-! ## Injectivity of the index maps -/

/-- Splitting `t * H + h` with `h < H`. -/
private theorem pair_of_mul_add {H t h t' h' : Nat} (hH : 0 < H) (hh : h < H) (hh' : h' < H)
    (he : t * H + h = t' * H + h') : t = t' ∧ h = h' := by
  have hdiv : ∀ {t h : Nat}, h < H → (t * H + h) / H = t := by
    intro t h hh
    rw [Nat.mul_comm t H, Nat.mul_add_div hH, Nat.div_eq_of_lt hh, Nat.add_zero]
  have hmod : ∀ {t h : Nat}, h < H → (t * H + h) % H = h := by
    intro t h hh
    rw [Nat.mul_comm t H, Nat.mul_add_mod, Nat.mod_eq_of_lt hh]
  constructor
  · have := congrArg (fun x => x / H) he
    rwa [hdiv hh, hdiv hh'] at this
  · have := congrArg (fun x => x % H) he
    rwa [hmod hh, hmod hh'] at this

/-- `oIdx` of element zero, as a multiple of the head stride. -/
private theorem oIdx_base {d : Inst} (hd : d.ok) (t h : Nat) :
    oIdx d t h 0 = (t * d.H + h) * d.ohs := by
  simp only [oIdx, hd.ots_eq]
  ring

/-- Two different `(t, h)` write `output` rows that do not overlap. -/
private theorem Rout_disj {d : Inst} (hd : d.ok) {t h t' h' : Nat}
    (_ht : t < d.T) (hh : h < d.H) (_ht' : t' < d.T) (hh' : h' < d.H)
    (hne : (t, h) ≠ (t', h')) :
    Disjoint (Rout d t h) (Rout d t' h') := by
  unfold Rout
  refine Buf.elems_disjoint d.output ?_
  rw [Set.disjoint_left]
  intro j hj hj'
  rw [row_iff] at hj hj'
  obtain ⟨hj1, hj2⟩ := hj
  obtain ⟨hj1', hj2'⟩ := hj'
  rcases Nat.eq_zero_or_pos d.head_size with h0 | hpos
  · omega
  · have hge := hd.ohs_ge
    have hohs : 0 < d.ohs := lt_of_lt_of_le hpos hge
    have hH : 0 < d.H := by omega
    rw [oIdx_base hd t h] at hj1 hj2
    rw [oIdx_base hd t' h'] at hj1' hj2'
    have hAA' : t * d.H + h ≠ t' * d.H + h' := by
      intro he
      obtain ⟨e1, e2⟩ := pair_of_mul_add hH hh hh' he
      exact hne (by rw [e1, e2])
    rcases lt_or_gt_of_ne hAA' with hlt | hlt
    · have hle : (t * d.H + h) * d.ohs + d.ohs ≤ (t' * d.H + h') * d.ohs := by
        have h1 : t * d.H + h + 1 ≤ t' * d.H + h' := hlt
        have h2 := Nat.mul_le_mul_right d.ohs h1
        have h3 : (t * d.H + h + 1) * d.ohs = (t * d.H + h) * d.ohs + d.ohs := by ring
        omega
      omega
    · have hle : (t' * d.H + h') * d.ohs + d.ohs ≤ (t * d.H + h) * d.ohs := by
        have h1 : t' * d.H + h' + 1 ≤ t * d.H + h := hlt
        have h2 := Nat.mul_le_mul_right d.ohs h1
        have h3 : (t' * d.H + h' + 1) * d.ohs = (t' * d.H + h') * d.ohs + d.ohs := by ring
        omega
      omega

/-- `olIdx` is injective on the grid when `output_lse` is passed. -/
private theorem olIdx_inj {d : Inst} (hd : d.ok) (ho : d.outLse) {t h t' h' : Nat}
    (ht : t < d.T) (hh : h < d.H) (ht' : t' < d.T) (hh' : h' < d.H)
    (he : olIdx d t h = olIdx d t' h') : t = t' ∧ h = h' := by
  have hT : 0 < d.T := by omega
  have hH : 0 < d.H := by omega
  rcases hd.olse_layout ho with ⟨holt, hole⟩ | ⟨holh, hole⟩
  · -- `ols` layout: token stride inside the head stride
    have hsmall : ∀ {t : Nat}, t < d.T → t * d.olts < d.olhs := by
      intro t ht
      have h1 := Nat.mul_lt_mul_of_pos_right ht holt
      omega
    have holh : 0 < d.olhs := by
      have := hsmall (t := 0) hT
      omega
    have hdiv : ∀ {h t : Nat}, t < d.T → (h * d.olhs + t * d.olts) / d.olhs = h := by
      intro h t ht
      rw [Nat.mul_comm h d.olhs, Nat.mul_add_div holh, Nat.div_eq_of_lt (hsmall ht),
        Nat.add_zero]
    have hmod : ∀ {h t : Nat}, t < d.T → (h * d.olhs + t * d.olts) % d.olhs = t * d.olts := by
      intro h t ht
      rw [Nat.mul_comm h d.olhs, Nat.mul_add_mod, Nat.mod_eq_of_lt (hsmall ht)]
    constructor
    · have := congrArg (fun x => x % d.olhs) he
      simp only [olIdx] at this
      rw [hmod ht, hmod ht'] at this
      exact Nat.mul_right_cancel holt this
    · have := congrArg (fun x => x / d.olhs) he
      simp only [olIdx] at this
      rwa [hdiv ht, hdiv ht'] at this
  · -- the other layout: head stride inside the token stride
    have hbig : ∀ {h : Nat}, h < d.H → h * d.olhs < d.olts := by
      intro h hh
      have h1 := Nat.mul_lt_mul_of_pos_right hh holh
      omega
    have holt : 0 < d.olts := by
      have := hbig (h := 0) hH
      omega
    have hdiv : ∀ {h t : Nat}, h < d.H → (h * d.olhs + t * d.olts) / d.olts = t := by
      intro h t hh
      rw [Nat.add_mul_div_right (h * d.olhs) t holt, Nat.div_eq_of_lt (hbig hh), Nat.zero_add]
    have hmod : ∀ {h t : Nat}, h < d.H → (h * d.olhs + t * d.olts) % d.olts = h * d.olhs := by
      intro h t hh
      rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (hbig hh)]
    constructor
    · have := congrArg (fun x => x / d.olts) he
      simp only [olIdx] at this
      rwa [hdiv hh, hdiv hh'] at this
    · have := congrArg (fun x => x % d.olts) he
      simp only [olIdx] at this
      rw [hmod hh, hmod hh'] at this
      exact Nat.mul_right_cancel holh this

/-- Two different `(t, h)` write different `output_lse` elements. -/
private theorem RlseOut_disj {d : Inst} (hd : d.ok) {t h t' h' : Nat}
    (ht : t < d.T) (hh : h < d.H) (ht' : t' < d.T) (hh' : h' < d.H)
    (hne : (t, h) ≠ (t', h')) :
    Disjoint (RlseOut d t h) (RlseOut d t' h') := by
  by_cases ho : d.outLse
  · unfold RlseOut
    rw [ite_eq_left ho, ite_eq_left ho]
    have hinj : olIdx d t h ≠ olIdx d t' h' := by
      intro he
      obtain ⟨e1, e2⟩ := olIdx_inj hd ho ht hh ht' hh' he
      exact hne (by rw [e1, e2])
    refine Buf.span_disjoint d.output_lse ?_
    intro hcast
    exact hinj (by exact_mod_cast hcast)
  · simp [RlseOut, ho]

/-! ## The two theorems -/

/-- No program writes a byte another program reads or writes. -/
theorem disj_write {d : Inst} (hd : d.ok) (init : Mem) :
    ∀ i j : Fin (launch d).size, i ≠ j →
      Disjoint (Wq d init i) (Rq d init j ∪ Wq d init j) := by
  intro i j hij
  have hti := pidT_lt d i
  have hhi := pidH_lt d i
  have htj := pidT_lt d j
  have hhj := pidH_lt d j
  have hth : (pidT d i, pidH d i) ≠ (pidT d j, pidH d j) := by
    intro he
    exact hij (pid_pair (congrArg Prod.fst he) (congrArg Prod.snd he))
  have hAC : Disjoint (Rout d (pidT d i) (pidH d i)) (Rlse d (pidT d j) (pidH d j)) :=
    (out_plseslse hd).mono (Rout_sub hd hti hhi) (Rlse_sub hd htj hhj)
  have hAD : Disjoint (Rout d (pidT d i) (pidH d i)) (Rin d (pidT d j) (pidH d j)) :=
    (out_pfxsfx hd).mono (Rout_sub hd hti hhi) (Rin_sub hd htj hhj)
  have hAE : Disjoint (Rout d (pidT d i) (pidH d i)) (Rout d (pidT d j) (pidH d j)) :=
    Rout_disj hd hti hhi htj hhj hth
  have hBC : Disjoint (RlseOut d (pidT d i) (pidH d i)) (Rlse d (pidT d j) (pidH d j)) := by
    by_cases ho : d.outLse
    · exact (olse_plseslse hd ho).mono (RlseOut_sub hd ho hti hhi) (Rlse_sub hd htj hhj)
    · simp [RlseOut, ho]
  have hBD : Disjoint (RlseOut d (pidT d i) (pidH d i)) (Rin d (pidT d j) (pidH d j)) := by
    by_cases ho : d.outLse
    · exact (olse_pfxsfx hd ho).mono (RlseOut_sub hd ho hti hhi) (Rin_sub hd htj hhj)
    · simp [RlseOut, ho]
  have hBE : Disjoint (RlseOut d (pidT d i) (pidH d i)) (Rout d (pidT d j) (pidH d j)) := by
    by_cases ho : d.outLse
    · exact (hd.output_lse_disj ho).symm.mono (RlseOut_sub hd ho hti hhi) (Rout_sub hd htj hhj)
    · simp [RlseOut, ho]
  have hBF : Disjoint (RlseOut d (pidT d i) (pidH d i)) (RlseOut d (pidT d j) (pidH d j)) :=
    RlseOut_disj hd hti hhi htj hhj hth
  have hAF : Disjoint (Rout d (pidT d i) (pidH d i)) (RlseOut d (pidT d j) (pidH d j)) := by
    by_cases ho : d.outLse
    · exact (hd.output_lse_disj ho).mono (Rout_sub hd hti hhi) (RlseOut_sub hd ho htj hhj)
    · simp [RlseOut, ho]
  unfold Wq Rq Wf Rf
  exact Disjoint.union_left
    (Disjoint.union_right (Disjoint.union_right hAC hAD) (Disjoint.union_right hAE hAF))
    (Disjoint.union_right (Disjoint.union_right hBC hBD) (Disjoint.union_right hBE hBF))

/-- No program reads a byte it writes. -/
theorem disj_read_write {d : Inst} (hd : d.ok) (init : Mem) :
    ∀ i : Fin (launch d).size, Disjoint (Rq d init i) (Wq d init i) := by
  intro i
  have hti := pidT_lt d i
  have hhi := pidH_lt d i
  have hCE : Disjoint (Rlse d (pidT d i) (pidH d i)) (Rout d (pidT d i) (pidH d i)) :=
    (out_plseslse hd).symm.mono (Rlse_sub hd hti hhi) (Rout_sub hd hti hhi)
  have hDE : Disjoint (Rin d (pidT d i) (pidH d i)) (Rout d (pidT d i) (pidH d i)) :=
    (out_pfxsfx hd).symm.mono (Rin_sub hd hti hhi) (Rout_sub hd hti hhi)
  have hCF : Disjoint (Rlse d (pidT d i) (pidH d i)) (RlseOut d (pidT d i) (pidH d i)) := by
    by_cases ho : d.outLse
    · exact (olse_plseslse hd ho).symm.mono (Rlse_sub hd hti hhi) (RlseOut_sub hd ho hti hhi)
    · simp [RlseOut, ho]
  have hDF : Disjoint (Rin d (pidT d i) (pidH d i)) (RlseOut d (pidT d i) (pidH d i)) := by
    by_cases ho : d.outLse
    · exact (olse_pfxsfx hd ho).symm.mono (Rin_sub hd hti hhi) (RlseOut_sub hd ho hti hhi)
    · simp [RlseOut, ho]
  unfold Rq Wq Rf Wf
  exact Disjoint.union_left
    (Disjoint.union_right hCE hCF)
    (Disjoint.union_right hDE hDF)

end Claims.MergeAttnStates
