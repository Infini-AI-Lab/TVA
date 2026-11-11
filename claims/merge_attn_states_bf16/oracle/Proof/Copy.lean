import Claim
import TVA.Proof
import Proof.Defs
import Proof.Prog
import Proof.Addr

/-!
# The copy branch

For a token the context does not cover (`ptwc ≤ t`), program `(t, h)` copies the suffix row (and
the suffix lse) to `output` (and `output_lse`). Nothing is merged, so the postcondition `CopyOk`
asks only that the values written are the values read.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## A `tl.store1` and its addrs -/

/-- The address of a scalar pointer. -/
theorem scalar_addrs (d : DType) (p : Ptr d) :
    tl.addrs (Tensor.scalar p) none = fun _ => some p.addr := rfl

/-! ## Indices of a row -/

/-! ## Bounds on the offsets -/

/-- The `suffix_lse` element is inside the buffer. -/
theorem copy_slIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    slIdx d (pidT d i) (pidH d i) < d.suffix_lse.len := by
  have hlen := hd.suffix_lse_len (T_pos i) (H_pos i)
  have e1 : pidH d i * d.slhs ≤ (d.H - 1) * d.slhs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  have e2 : pidT d i * d.slts ≤ (d.T - 1) * d.slts :=
    Nat.mul_le_mul_right _ (by have := pidT_lt d i; omega)
  simp only [slIdx]
  omega

/-- The `output_lse` element is inside the buffer (when passed). -/
theorem copy_olIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (ho : d.outLse) :
    olIdx d (pidT d i) (pidH d i) < d.output_lse.len := by
  have hlen := hd.output_lse_len ho (T_pos i) (H_pos i)
  have e1 : pidH d i * d.olhs ≤ (d.H - 1) * d.olhs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  have e2 : pidT d i * d.olts ≤ (d.T - 1) * d.olts :=
    Nat.mul_le_mul_right _ (by have := pidT_lt d i; omega)
  simp only [olIdx]
  omega

/-- A `suffix_output` row element is inside the buffer. -/
theorem copy_soIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) {j : Nat}
    (hj : j < d.head_size) : soIdx d (pidT d i) (pidH d i) j < d.suffix_output.len := by
  have hpos : 0 < d.head_size := by omega
  have hlen := hd.suffix_output_len (T_pos i) (H_pos i) hpos
  have e1 : pidT d i * d.sots ≤ (d.T - 1) * d.sots :=
    Nat.mul_le_mul_right _ (by have := pidT_lt d i; omega)
  have e2 : pidH d i * d.shs ≤ (d.H - 1) * d.shs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  simp only [soIdx]
  omega

/-- An `output` row element is inside the buffer. -/
theorem copy_oIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) {j : Nat}
    (hj : j < d.head_size) : oIdx d (pidT d i) (pidH d i) j < d.output.len := by
  have hpos : 0 < d.head_size := by omega
  have hlen := hd.output_len (T_pos i) (H_pos i) hpos
  have e1 : pidT d i * d.ots ≤ (d.T - 1) * d.ots :=
    Nat.mul_le_mul_right _ (by have := pidT_lt d i; omega)
  have e2 : pidH d i * d.ohs ≤ (d.H - 1) * d.ohs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  simp only [oIdx]
  omega

/-- Every element of the `output` row is inside the buffer. -/
theorem copy_oIdx_row_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {j : Nat}
    (hj : j ∈ row d (oIdx d (pidT d i) (pidH d i) 0)) : j < d.output.len := by
  rw [row_iff] at hj
  obtain ⟨h1, h2⟩ := hj
  simp only [oIdx] at h1 h2
  have hpos : 0 < d.head_size := by omega
  have hlen := hd.output_len (T_pos i) (H_pos i) hpos
  have e1 : pidT d i * d.ots ≤ (d.T - 1) * d.ots :=
    Nat.mul_le_mul_right _ (by have := pidT_lt d i; omega)
  have e2 : pidH d i * d.ohs ≤ (d.H - 1) * d.ohs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  omega

/-! ## Lanes of a row -/

/-- The row's lane `j` is the `suffix_output` element `soIdx j`. -/
theorem copy_rowG_soIdx (d : Inst) (i : Fin (launch d).size) (j : Idx [d.padded]) :
    rowG d (soIdx d (pidT d i) (pidH d i) 0) j = soIdx d (pidT d i) (pidH d i) j.1.val := by
  simp only [rowG, soIdx]
  omega

/-- The row's lane `j` is the `output` element `oIdx j`. -/
theorem copy_rowG_oIdx (d : Inst) (i : Fin (launch d).size) (j : Idx [d.padded]) :
    rowG d (oIdx d (pidT d i) (pidH d i) 0) j = oIdx d (pidT d i) (pidH d i) j.1.val := by
  simp only [rowG, oIdx]
  omega

/-- Distinct lanes of a row are distinct elements. -/
theorem copy_rowG_inj (d : Inst) {base : Nat} {a b : Idx [d.padded]}
    (h : rowG d base a = rowG d base b) : a = b := by
  have hv : a.1.val = b.1.val := by
    simp only [rowG] at h
    omega
  obtain ⟨a1, a2⟩ := a
  obtain ⟨b1, b2⟩ := b
  have h2 : a2 = b2 := by cases a2; cases b2; rfl
  exact Prod.ext (Fin.ext hv) h2

/-- A lane is enabled by the head mask iff it is below `head_size`. -/
theorem copy_rowEn_iff (d : Inst) (j : Idx [d.padded]) : rowEn d j = true ↔ j.1.val < d.head_size := by
  simp only [rowEn, decide_eq_true_eq]

/-! ## Reading through agreement -/

/-- Two memories that agree on a set holding the element's bytes read it alike. -/
theorem copy_rdB_agree {m₁ m₂ : Mem} {b : Buf .bf16} {i : Nat} {T : Set Int}
    (h : ∀ x ∈ T, m₁ x = m₂ x) (hsub : Mem.span .bf16 (b.ptr (i : Int)).addr ⊆ T) :
    rdB m₁ b i = rdB m₂ b i := by
  unfold rdB
  exact Mem.read_congr (fun x hx => h x (hsub hx))

/-! ## The row part of the program -/

/-- The row copy at the end of `copyProg`: read `head_size` lanes of the suffix row, write them
to `output`. -/
def copy_rowProg (d : Inst) (c : Ctx) : Prog Unit := do
  let head_arange := tl.arange 0 d.padded
  let head_mask := head_arange.ltS (IntV.ofNat d.head_size : I32)
  let s_out : Tensor [d.padded] BF16 ←
    tl.load (d.suffix_output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.phs
      + tl.program_id c 1 * IntV.ofNat d.phs + head_arange : Tensor [d.padded] (Ptr .bf16)) (some head_mask)
  tl.store (d.output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.ohs
    + tl.program_id c 1 * IntV.ofNat d.ohs + head_arange) s_out (some head_mask)

/-- **The row copy.** Reading one row of `suffix_output` and writing it to `output`. -/
theorem copy_safe_row_copy {d : Inst} (hd : d.ok) {E : Env} {m : Mem} (i : Fin (launch d).size)
    (hvs : d.suffix_output.Valid m) (hvo : d.output.Valid m)
    {R W : Set Int}
    (hR : d.suffix_output.elems (row d (soIdx d (pidT d i) (pidH d i) 0)) ⊆ R)
    (hW : d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) ⊆ W)
    {Q : Unit → Mem → Prop}
    (hk : ∀ (s : Tensor [d.padded] BF16) (m' : Mem),
      (∀ j : Idx [d.padded], rowEn d j = true →
        m.read .bf16 (d.suffix_output.ptr (rowG d (soIdx d (pidT d i) (pidH d i) 0) j : Int)).addr
          = some (s.get j)) →
      (∀ j : Idx [d.padded], rowEn d j = true →
        m'.read .bf16 (d.output.ptr (rowG d (oIdx d (pidT d i) (pidH d i) 0) j : Int)).addr
          = some (s.get j)) →
      (∀ x, x ∉ d.output.bytes → m' x = m x) →
      Q () m') :
    Safe E R W Q m (copy_rowProg d ((launch d).ctx i)) := by
  have hA := suffix_row_addrs d hd i
  have hin : ∀ j : Idx [d.padded], rowEn d j = true →
      rowG d (soIdx d (pidT d i) (pidH d i) 0) j < d.suffix_output.len := by
    intro j hj
    rw [copy_rowG_soIdx]
    exact copy_soIdx_lt d hd i ((copy_rowEn_iff d j).mp hj)
  have hR' : d.suffix_output.elems (Buf.img (rowG d (soIdx d (pidT d i) (pidH d i) 0)) (rowEn d)) ⊆ R := by
    rw [img_rowG d hd]
    exact hR
  refine safe_load_buf d.suffix_output hA hin hvs hR' ?_
  intro s hld _
  have hA2 := output_row_addrs d hd i
  have hin2 : ∀ j : Idx [d.padded], rowEn d j = true →
      rowG d (oIdx d (pidT d i) (pidH d i) 0) j < d.output.len := by
    intro j hj
    rw [copy_rowG_oIdx]
    exact copy_oIdx_lt d hd i ((copy_rowEn_iff d j).mp hj)
  have hinj : ∀ a b : Idx [d.padded], rowEn d a = true → rowEn d b = true →
      rowG d (oIdx d (pidT d i) (pidH d i) 0) a = rowG d (oIdx d (pidT d i) (pidH d i) 0) b → a = b :=
    fun a b _ _ h => copy_rowG_inj d h
  have hW' : d.output.elems (Buf.img (rowG d (oIdx d (pidT d i) (pidH d i) 0)) (rowEn d)) ⊆ W := by
    rw [img_rowG d hd]
    exact hW
  show Safe E R W Q m
    (tl.store (d.output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.arange 0 d.padded) s
      (some ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32) : Tensor [d.padded] Bool))
      >>= fun _ => pure ())
  refine safe_store_buf d.output hA2 hin2 hinj hvo hW' ?_
  intro m' hst hfra _
  refine safe_ret (hk s m' hld hst ?_)
  intro x hx
  exact hfra x (fun hmem => by
    rw [img_rowG d hd] at hmem
    exact hx (d.output.elems_sub_bytes (fun j hj => copy_oIdx_row_lt hd i hj) hmem))

/-! ## The branch -/

/-- **The copy branch.** -/
theorem safe_copy {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) (_hpt : ¬ pidT d i < d.ptwc) :
    Safe E (Rf d (pidT d i) (pidH d i)) (Wf d (pidT d i) (pidH d i))
      (fun _ m => CopyOk d init m (pidT d i) (pidH d i)) init
      (copyProg d ((launch d).ctx i) (olOf d)) := by
  obtain ⟨hvout, hvoutlse, _hvpo, _hvpl, hvso, hvsl, _hvals⟩ := hpre
  have hsl := copy_slIdx_lt d hd i
  have hR : Mem.span .f32 (d.suffix_lse.ptr (slIdx d (pidT d i) (pidH d i) : Int)).addr ⊆
      Rf d (pidT d i) (pidH d i) := by
    intro x hx
    show x ∈ Rlse d (pidT d i) (pidH d i) ∪ Rin d (pidT d i) (pidH d i)
    exact Or.inl (Or.inr hx)
  have hRrow : d.suffix_output.elems (row d (soIdx d (pidT d i) (pidH d i) 0)) ⊆
      Rf d (pidT d i) (pidH d i) := by
    intro x hx
    show x ∈ Rlse d (pidT d i) (pidH d i) ∪ Rin d (pidT d i) (pidH d i)
    refine Or.inr ?_
    show x ∈ (if pidT d i < d.ptwc then
        d.prefix_output.elems (row d (poIdx d (pidT d i) (pidH d i) 0)) else ∅) ∪
      d.suffix_output.elems (row d (soIdx d (pidT d i) (pidH d i) 0))
    exact Or.inr hx
  have hWrow : d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) ⊆
      Wf d (pidT d i) (pidH d i) := by
    intro x hx
    show x ∈ Rout d (pidT d i) (pidH d i) ∪ RlseOut d (pidT d i) (pidH d i)
    exact Or.inl hx
  simp only [copyProg]
  rw [suffix_lse_ptr d hd i]
  refine safe_load1_buf d.suffix_lse hsl hvsl hR ?_
  intro s_lse hread
  by_cases hOL : d.outLse
  · have hol : olOf d = some (d.output_lse.ptr 0) := by simp [olOf, hOL]
    rw [hol]
    dsimp only
    rw [output_lse_ptr d hd i]
    have holL := copy_olIdx_lt d hd i hOL
    have hRe : RlseOut d (pidT d i) (pidH d i) =
        Mem.span .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
      simp [RlseOut, hOL]
    have hWst : Mem.span .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i))).addr ⊆
        Wf d (pidT d i) (pidH d i) := by
      intro x hx
      show x ∈ Rout d (pidT d i) (pidH d i) ∪ RlseOut d (pidT d i) (pidH d i)
      exact Or.inr (by rw [hRe]; exact hx)
    refine safe_store1 (p := d.output_lse.ptr (olIdx d (pidT d i) (pidH d i))) (v := s_lse)
      (d.output_lse.writable (hvoutlse hOL) holL) hWst ?_
    · intro m' hres hfr
      have hframe : ∀ x, x ∉ Mem.span .f32
          (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr → m' x = init x := by
        intro x hx
        exact hfr x hx
      have hvs' : d.suffix_output.Valid m' := by
        refine Buf.valid_of_agree hvso ?_
        intro x hx
        have hx' : x ∉ Mem.span .f32
            (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
          intro hspan
          exact Set.disjoint_left.mp (hd.output_lse_in_disj hOL)
            (d.output_lse.span_sub holL hspan) (Or.inl (Or.inl (Or.inr hx)))
        rw [hframe x hx']
        exact hvso.2.2.2 x hx
      have hvo' : d.output.Valid m' := by
        refine Buf.valid_of_agree hvout ?_
        intro x hx
        have hx' : x ∉ Mem.span .f32
            (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
          intro hspan
          exact Set.disjoint_left.mp (hd.output_lse_disj hOL) hx
            (d.output_lse.span_sub holL hspan)
        rw [hframe x hx']
        exact hvout.2.2.2 x hx
      have hWs : d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) ⊆
          Wf d (pidT d i) (pidH d i) \ Mem.span .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i))).addr := by
        intro x hx
        refine ⟨hWrow hx, ?_⟩
        intro hspan
        exact Set.disjoint_left.mp (hd.output_lse_disj hOL)
          (d.output.elems_sub_bytes (fun j hj => copy_oIdx_row_lt hd i hj) hx)
          (d.output_lse.span_sub holL hspan)
      refine copy_safe_row_copy hd i hvs' hvo' hRrow hWs ?_
      intro s_out m'' hld hst hfr
      refine ⟨?_, ?_⟩
      · intro _
        have hone : m'.read .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr
            = some s_lse := by
          exact hres
        refine ⟨s_lse, hread, ?_⟩
        show m''.read .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr
          = some s_lse
        rw [show m''.read .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr
              = m'.read .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr from
            Mem.read_congr (fun x hx => hfr x (fun hmem =>
              Set.disjoint_left.mp (hd.output_lse_disj hOL) hmem
                (d.output_lse.span_sub holL hx)))]
        exact hone
      · intro j hj
        have hjpad : j < d.padded := lt_of_lt_of_le hj hd.head_le
        refine ⟨s_out.get ((⟨j, hjpad⟩ : Fin d.padded), ()), ?_, ?_⟩
        · have hen : rowEn d ((⟨j, hjpad⟩ : Fin d.padded), ()) = true := by
            exact (copy_rowEn_iff d ((⟨j, hjpad⟩ : Fin d.padded), ())).mpr hj
          have hg : rowG d (soIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
              = soIdx d (pidT d i) (pidH d i) j := by
            exact copy_rowG_soIdx d i ((⟨j, hjpad⟩ : Fin d.padded), ())
          have hldj := hld _ hen
          rw [hg] at hldj
          have hspan : Mem.span .bf16
              (d.suffix_output.ptr (soIdx d (pidT d i) (pidH d i) j : Int)).addr ⊆
              d.suffix_output.bytes :=
            d.suffix_output.span_sub (copy_soIdx_lt d hd i hj)
          have hframesub : ∀ x ∈ d.suffix_output.bytes, m' x = init x := by
            intro x hx
            exact hframe x (by
              intro hsp
              exact Set.disjoint_left.mp (hd.output_lse_in_disj hOL)
                (d.output_lse.span_sub holL hsp) (Or.inl (Or.inl (Or.inr hx))))
          rw [← copy_rdB_agree hframesub hspan]
          exact hldj
        · have hen : rowEn d ((⟨j, hjpad⟩ : Fin d.padded), ()) = true := by
            exact (copy_rowEn_iff d ((⟨j, hjpad⟩ : Fin d.padded), ())).mpr hj
          have hg : rowG d (oIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
              = oIdx d (pidT d i) (pidH d i) j := by
            exact copy_rowG_oIdx d i ((⟨j, hjpad⟩ : Fin d.padded), ())
          have hstj := hst _ hen
          rw [hg] at hstj
          exact hstj
  · have hol : olOf d = none := by simp [olOf, hOL]
    rw [hol]
    dsimp only
    refine copy_safe_row_copy hd i hvso hvout hRrow hWrow ?_
    intro s_out m'' hld hst _
    refine ⟨?_, ?_⟩
    · intro hc
      exact absurd hc hOL
    · intro j hj
      have hjpad : j < d.padded := lt_of_lt_of_le hj hd.head_le
      refine ⟨s_out.get ((⟨j, hjpad⟩ : Fin d.padded), ()), ?_, ?_⟩
      · have hen : rowEn d ((⟨j, hjpad⟩ : Fin d.padded), ()) = true := by
          exact (copy_rowEn_iff d ((⟨j, hjpad⟩ : Fin d.padded), ())).mpr hj
        have hg : rowG d (soIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
            = soIdx d (pidT d i) (pidH d i) j := by
          exact copy_rowG_soIdx d i ((⟨j, hjpad⟩ : Fin d.padded), ())
        have hldj := hld _ hen
        rw [hg] at hldj
        exact hldj
      · have hen : rowEn d ((⟨j, hjpad⟩ : Fin d.padded), ()) = true := by
          exact (copy_rowEn_iff d ((⟨j, hjpad⟩ : Fin d.padded), ())).mpr hj
        have hg : rowG d (oIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
            = oIdx d (pidT d i) (pidH d i) j := by
          exact copy_rowG_oIdx d i ((⟨j, hjpad⟩ : Fin d.padded), ())
        have hstj := hst _ hen
        rw [hg] at hstj
        exact hstj

end Claims.MergeAttnStates
