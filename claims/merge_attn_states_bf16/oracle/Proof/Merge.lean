import Claim
import TVA.Proof
import Proof.Defs
import Proof.Prog
import Proof.Addr
import Proof.Copy
import Proof.NumMid
import Proof.NumOut
import Proof.NumRow

/-!
# The merge branch

For a token the context covers (`t < ptwc`), program `(t, h)` reads the two lse scalars and the two
`bfloat16` rows, merges them with the online-softmax formula, and writes the merged row (and the
merged lse).
-/

set_option maxHeartbeats 1000000

namespace Claims.MergeAttnStates

open TVA

/-! ## The branch test -/

/-- The kernel's `prefix_mask`, read as the host's comparison. -/
theorem prefix_mask_eq (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((bc (tl.program_id ((launch d).ctx i) 0)).ltS (IntV.ofNat d.ptwc : I32)).item
      = decide (pidT d i < d.ptwc) := by
  have hp31 : d.ptwc < 2 ^ 31 := by
    have := hd.ptwc_le
    have := hd.T_lt
    omega
  show IntV.lt (tl.program_id ((launch d).ctx i) 0) (IntV.ofNat d.ptwc : I32)
      = decide (pidT d i < d.ptwc)
  rw [IntV.lt_eq, pidT_toInt d hd i, IntV.I32.toInt_ofNat hp31]
  simp

/-- Fusion is on in the launch's environment. -/
theorem env_fusion (d : Inst) (P : Platform) : ((launch d).env P).fusion = true := rfl

/-! ## The footprints -/

/-- The `prefix_lse` element is in the read footprint (merged token). -/
theorem Rf_prefix_lse (d : Inst) (t h : Nat) (hpt : t < d.ptwc) :
    Mem.span .f32 (d.prefix_lse.ptr (plIdx d t h : Int)).addr ⊆ Rf d t h := by
  intro x hx
  refine Or.inl (Or.inl ?_)
  show x ∈ (if t < d.ptwc then
    Mem.span .f32 (d.prefix_lse.ptr (plIdx d t h : Int)).addr else ∅)
  rw [ite_eq_left hpt]
  exact hx

/-- The `suffix_lse` element is in the read footprint. -/
theorem Rf_suffix_lse (d : Inst) (t h : Nat) :
    Mem.span .f32 (d.suffix_lse.ptr (slIdx d t h : Int)).addr ⊆ Rf d t h := by
  intro x hx
  refine Or.inl (Or.inr ?_)
  show x ∈ Mem.span .f32 (d.suffix_lse.ptr (slIdx d t h : Int)).addr
  exact hx

/-- The `prefix_output` row is in the read footprint (merged token). -/
theorem Rf_prefix_row (d : Inst) (t h : Nat) (hpt : t < d.ptwc) :
    d.prefix_output.elems (row d (poIdx d t h 0)) ⊆ Rf d t h := by
  intro x hx
  refine Or.inr (Or.inl ?_)
  show x ∈ (if t < d.ptwc then
    d.prefix_output.elems (row d (poIdx d t h 0)) else ∅)
  rw [ite_eq_left hpt]
  exact hx

/-- The `suffix_output` row is in the read footprint. -/
theorem Rf_suffix_row (d : Inst) (t h : Nat) :
    d.suffix_output.elems (row d (soIdx d t h 0)) ⊆ Rf d t h := by
  intro x hx
  refine Or.inr (Or.inr ?_)
  show x ∈ d.suffix_output.elems (row d (soIdx d t h 0))
  exact hx

/-- The `output` row is in the write footprint. -/
theorem Wf_output_row (d : Inst) (t h : Nat) :
    d.output.elems (row d (oIdx d t h 0)) ⊆ Wf d t h := by
  intro x hx
  exact Or.inl hx

/-- The `output_lse` element is in the write footprint (when passed). -/
theorem Wf_output_lse (d : Inst) (t h : Nat) (ho : d.outLse) :
    Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr ⊆ Wf d t h := by
  intro x hx
  refine Or.inr ?_
  show x ∈ (if d.outLse then
    Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr else ∅)
  rw [ite_eq_left ho]
  exact hx

/-! ## Scalar-index bounds -/

/-- The `prefix_lse` element is inside the buffer. -/
theorem merge_plIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size)
    (hpt : pidT d i < d.ptwc) :
    plIdx d (pidT d i) (pidH d i) < d.prefix_lse.len := by
  have hlen := hd.prefix_lse_len (by omega : 0 < d.ptwc) (H_pos i)
  have e1 : pidH d i * d.plhs ≤ (d.H - 1) * d.plhs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  have e2 : pidT d i * d.plts ≤ (d.ptwc - 1) * d.plts :=
    Nat.mul_le_mul_right _ (by omega)
  simp only [plIdx]
  omega

/-- A `prefix_output` row element is inside the buffer. -/
theorem merge_poIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size)
    (hpt : pidT d i < d.ptwc) {j : Nat} (hj : j < d.head_size) :
    poIdx d (pidT d i) (pidH d i) j < d.prefix_output.len := by
  have hpos : 0 < d.head_size := by omega
  have hlen := hd.prefix_output_len (by omega : 0 < d.ptwc) (H_pos i) hpos
  have e1 : pidT d i * d.pots ≤ (d.ptwc - 1) * d.pots :=
    Nat.mul_le_mul_right _ (by omega)
  have e2 : pidH d i * d.phs ≤ (d.H - 1) * d.phs :=
    Nat.mul_le_mul_right _ (by have := pidH_lt d i; omega)
  simp only [poIdx]
  omega

/-- The row's lane `j` is the `prefix_output` element `poIdx j`. -/
theorem merge_rowG_poIdx (d : Inst) (i : Fin (launch d).size) (j : Idx [d.padded]) :
    rowG d (poIdx d (pidT d i) (pidH d i) 0) j = poIdx d (pidT d i) (pidH d i) j.1.val := by
  simp only [rowG, poIdx]
  omega

/-! ## The clamps of the two comparisons -/

/-- A scalar `feq` is `FloatV.eq`. -/
theorem merge_feq_item (x y : F32) :
    (((bc x : Tensor [] F32)).feq (bc y)).item = FloatV.eq x y := rfl

/-- The merged lse's clamp, whose condition is the comparison tensor itself. -/
theorem merge_where_item (M cl0 : F32) :
    ((Tensor.where_ ((bc M).feq (bc negInfC)) (bc negInfC) (bc cl0) : Tensor [] F32)).item
      = (if FloatV.eq M negInfC then negInfC else cl0) := rfl

/-- The merged row's clamp, whose condition is a `Bool` scalar. -/
theorem merge_where_get {d : Inst} (M : F32) (out32 : Tensor [d.padded] F32)
    (lane : Idx [d.padded]) :
    (Tensor.where_ (bc ((bc M).feq (bc negInfC)).item) (bc zeroC) out32 :
        Tensor [d.padded] F32).get lane
      = (if FloatV.eq M negInfC then zeroC else out32.get lane) := rfl

/-- If a maximum of two non-NaN values is `−∞`, then both values are. -/
theorem maxRel_eq_negInf_both' {x y M : F32} (hM : FloatV.MaxRel true x y M)
    (hx : x.isNaN = false) (hy : y.isNaN = false) (hMneg : M = negInfC) :
    x = negInfC ∧ y = negInfC := by
  have key : ∀ z : F32, z.isNaN = false → FloatV.le z M = true → z = negInfC := by
    intro z hz hle
    rw [hMneg] at hle
    have h1 : z.toRat ≤ negInfC.toRat := (FloatV.le_iff hz negInfC_isNaN).mp hle
    have h2 : negInfC.toRat ≤ z.toRat := by
      have h := (FloatV.negInf_toRat_le (f := Format.binary32) Format.binary32_regular (by decide) hz).1
      rwa [negInfC_eq_negInf]
    exact (eq_negInfC_iff hz).mp
      ((FloatV.eq_iff hz negInfC_isNaN).mpr (le_antisymm h1 h2))
  obtain ⟨hlex, hley⟩ := hM.le hx hy
  exact ⟨key x hx hlex, key y hy hley⟩

/-! ## The row part of the branch -/

/-- Reading the two `bfloat16` rows, scaling them by the two weights, and storing the merged row. -/
def merge_rowProg (d : Inst) (c : Ctx) (pe se o M : F32) : Prog Unit := do
  let head_arange := tl.arange 0 d.padded
  let head_mask := head_arange.ltS (IntV.ofNat d.head_size : I32)
  let p_out : Tensor [d.padded] BF16 ←
    tl.load (d.prefix_output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.phs
      + tl.program_id c 1 * IntV.ofNat d.phs + head_arange : Tensor [d.padded] (Ptr .bf16)) (some head_mask)
  let s_out : Tensor [d.padded] BF16 ←
    tl.load (d.suffix_output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.phs
      + tl.program_id c 1 * IntV.ofNat d.phs + head_arange : Tensor [d.padded] (Ptr .bf16)) (some head_mask)
  let p_scale := (← tl.div (bc pe) (bc o)).item
  let s_scale := (← tl.div (bc se) (bc o)).item
  let p32 : Tensor [d.padded] F32 ← tl.cvt p_out
  let s32 : Tensor [d.padded] F32 ← tl.cvt s_out
  let prod_p ← tl.mul p32 (bc p_scale)
  let prod_s ← tl.mul s32 (bc s_scale)
  let out32 ← tl.addc prod_p prod_s
  let out32 := Tensor.where_ (bc ((bc M).feq (bc negInfC)).item) (bc zeroC) out32
  let out16 : Tensor [d.padded] BF16 ← tl.cvt out32
  tl.store (d.output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.ohs
    + tl.program_id c 1 * IntV.ofNat d.ohs + head_arange) out16 (some head_mask)

/-- **The row part of the merge branch.** -/
theorem merge_safe_row {d : Inst} (hd : d.ok) {E : Env} {init : Mem} {m : Mem}
    (i : Fin (launch d).size) (hpt : pidT d i < d.ptwc)
    {p_lse s_lse M u v pe se o : F32}
    (hM : FloatV.MaxRel true (corr p_lse) (corr s_lse) M)
    (hpnan : p_lse.isNaN = false) (hsnan : s_lse.isNaN = false)
    (hm : ¬ bothEmpty p_lse s_lse → LseMid (corr p_lse) (corr s_lse) M u v pe se o)
    (hdt : Device.DivTotal E.P .binary32) (hdiv : Device.DivFull E.P)
    (hfus : E.fusion = true)
    {R W : Set Int}
    (hR : d.prefix_output.elems (row d (poIdx d (pidT d i) (pidH d i) 0)) ∪
      d.suffix_output.elems (row d (soIdx d (pidT d i) (pidH d i) 0)) ⊆ R)
    (hW : d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) ⊆ W)
    (hvp : d.prefix_output.Valid m) (hvs : d.suffix_output.Valid m) (hvo : d.output.Valid m)
    (hagree : ∀ x ∈ d.prefix_output.bytes ∪ d.suffix_output.bytes, m x = init x)
    (hreadP : rdF init d.prefix_lse (plIdx d (pidT d i) (pidH d i)) = some p_lse)
    (hreadS : rdF init d.suffix_lse (slIdx d (pidT d i) (pidH d i)) = some s_lse)
    (hval : ¬ bothEmpty p_lse s_lse → ∀ j, j < d.head_size → ∃ pv sv : BF16,
      rdB init d.prefix_output (poIdx d (pidT d i) (pidH d i) j) = some pv ∧
      rdB init d.suffix_output (soIdx d (pidT d i) (pidH d i) j) = some sv ∧
      pv.isFinite = true ∧ sv.isFinite = true)
    (hlse : d.outLse → ∃ cl : F32,
      (∀ m'' : Mem, (∀ x, x ∉ d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) →
        m'' x = m x) →
        rdF m'' d.output_lse (olIdx d (pidT d i) (pidH d i)) = some cl) ∧
      (bothEmpty p_lse s_lse → cl = negInfC) ∧
      (¬ bothEmpty p_lse s_lse → cl.isFinite = true ∧
        |cl.toReal - refLse (corr p_lse) (corr s_lse)| ≤
          (lseTol : ℝ) * (1 + |refLse (corr p_lse) (corr s_lse)|))) :
    Safe E R W (fun _ m' => MergeOk d init m' (pidT d i) (pidH d i)) m
      (merge_rowProg d ((launch d).ctx i) pe se o M) := by
  have hin1 : ∀ j : Idx [d.padded], rowEn d j = true →
      rowG d (poIdx d (pidT d i) (pidH d i) 0) j < d.prefix_output.len := by
    intro j hj
    rw [merge_rowG_poIdx]
    exact merge_poIdx_lt d hd i hpt ((copy_rowEn_iff d j).mp hj)
  have hR1 : d.prefix_output.elems (Buf.img (rowG d (poIdx d (pidT d i) (pidH d i) 0)) (rowEn d))
      ⊆ R := by
    rw [img_rowG d hd]
    intro x hx
    exact hR (Or.inl hx)
  refine safe_load_buf d.prefix_output (prefix_row_addrs d hd i) hin1 hvp hR1 ?_
  intro p_out hpld _
  have hin2 : ∀ j : Idx [d.padded], rowEn d j = true →
      rowG d (soIdx d (pidT d i) (pidH d i) 0) j < d.suffix_output.len := by
    intro j hj
    rw [copy_rowG_soIdx]
    exact copy_soIdx_lt d hd i ((copy_rowEn_iff d j).mp hj)
  have hR2 : d.suffix_output.elems (Buf.img (rowG d (soIdx d (pidT d i) (pidH d i) 0)) (rowEn d))
      ⊆ R := by
    rw [img_rowG d hd]
    intro x hx
    exact hR (Or.inr hx)
  refine safe_load_buf d.suffix_output (suffix_row_addrs d hd i) hin2 hvs hR2 ?_
  intro s_out hsld _
  refine safe_div (hex := fun k => hdt ((bc pe).get k) ((bc o).get k)) ?_
  intro psc hpsc
  try dsimp only
  refine safe_div (hex := fun k => hdt ((bc se).get k) ((bc o).get k)) ?_
  intro ssc hssc
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro p32 hc1
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro s32 hc2
  refine safe_mul ?_
  intro prod_p hm1
  refine safe_mul ?_
  intro prod_s hm2
  refine safe_addc ?_
  intro out32a ha
  refine safe_cvt (g := .bfloat16)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ ?_
  intro out16 hc3
  have hin3 : ∀ j : Idx [d.padded], rowEn d j = true →
      rowG d (oIdx d (pidT d i) (pidH d i) 0) j < d.output.len := by
    intro j hj
    rw [copy_rowG_oIdx]
    exact copy_oIdx_lt d hd i ((copy_rowEn_iff d j).mp hj)
  have hinj : ∀ a b : Idx [d.padded], rowEn d a = true → rowEn d b = true →
      rowG d (oIdx d (pidT d i) (pidH d i) 0) a = rowG d (oIdx d (pidT d i) (pidH d i) 0) b →
      a = b :=
    fun a b _ _ h => copy_rowG_inj d h
  have hW3 : d.output.elems (Buf.img (rowG d (oIdx d (pidT d i) (pidH d i) 0)) (rowEn d))
      ⊆ W := by
    rw [img_rowG d hd]
    exact hW
  show Safe E R W (fun _ m' => MergeOk d init m' (pidT d i) (pidH d i)) m
    (tl.store (d.output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.arange 0 d.padded) out16
      (some ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32) : Tensor [d.padded] Bool))
      >>= fun _ => pure ())
  refine safe_store_buf d.output (output_row_addrs d hd i) hin3 hinj hvo hW3 ?_
  intro m'' hst hfra _
  refine safe_ret ?_
  refine ⟨p_lse, s_lse, hreadP, hreadS, ?_, ?_⟩
  · intro hOL
    obtain ⟨cl, hclread, hclboth, hclne⟩ := hlse hOL
    refine ⟨cl, ?_, hclboth, hclne⟩
    exact hclread m'' (fun x hx => hfra x (by rwa [← img_rowG d hd] at hx))
  · intro j hj
    have hjpad : j < d.padded := lt_of_lt_of_le hj hd.head_le
    refine ⟨out16.get ((⟨j, hjpad⟩ : Fin d.padded), ()), ?_, ?_, ?_⟩
    · have hen : rowEn d ((⟨j, hjpad⟩ : Fin d.padded), ()) = true :=
        (copy_rowEn_iff d _).mpr hj
      have hg : rowG d (oIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
          = oIdx d (pidT d i) (pidH d i) j := copy_rowG_oIdx d i _
      have hstj := hst _ hen
      rw [hg] at hstj
      exact hstj
    · intro hBE
      have hMeq : M = negInfC := by
        refine maxRel_eq_negInf_both ?_
        rwa [hBE.1, hBE.2] at hM
      have hMn : M.isNaN = false := by
        rw [hMeq]
        exact negInfC_isNaN
      have hEqT : FloatV.eq M negInfC = true := (eq_negInfC_iff hMn).mpr hMeq
      have hgl := merge_where_get (d := d) M out32a ((⟨j, hjpad⟩ : Fin d.padded), ())
      have h3 := hc3 ((⟨j, hjpad⟩ : Fin d.padded), ())
      rw [hgl, ite_eq_left hEqT] at h3
      have h1 : out16.get ((⟨j, hjpad⟩ : Fin d.padded), ()) =
          FloatV.cvtWith .rn (zeroC : F32) :=
        FloatV.NaNEq.eq h3 (by rw [cvtWith_zeroC]; exact zeroB_isNaN)
      rw [h1, cvtWith_zeroC]
    · intro hne
      obtain ⟨pv, sv, hpvr, hsvr, hpvf, hsvf⟩ := hval hne j hj
      have hen : rowEn d ((⟨j, hjpad⟩ : Fin d.padded), ()) = true :=
        (copy_rowEn_iff d _).mpr hj
      have hldp := hpld ((⟨j, hjpad⟩ : Fin d.padded), ()) hen
      have hgp : rowG d (poIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
          = poIdx d (pidT d i) (pidH d i) j := merge_rowG_poIdx d i _
      rw [hgp] at hldp
      have hspanP : Mem.span .bf16
          (d.prefix_output.ptr (poIdx d (pidT d i) (pidH d i) j : Int)).addr ⊆
          d.prefix_output.bytes ∪ d.suffix_output.bytes :=
        fun x hx => Or.inl (d.prefix_output.span_sub (merge_poIdx_lt d hd i hpt hj) hx)
      have hagP : m.read .bf16
            (d.prefix_output.ptr (poIdx d (pidT d i) (pidH d i) j : Int)).addr
          = init.read .bf16
            (d.prefix_output.ptr (poIdx d (pidT d i) (pidH d i) j : Int)).addr :=
        Mem.read_congr (fun x hx => hagree x (hspanP hx))
      have hpv_eq : p_out.get ((⟨j, hjpad⟩ : Fin d.padded), ()) = pv := by
        have h2 := hpvr
        unfold rdB at h2
        rw [← hagP] at h2
        rw [hldp] at h2
        exact Option.some.inj h2
      have hlds := hsld ((⟨j, hjpad⟩ : Fin d.padded), ()) hen
      have hgs : rowG d (soIdx d (pidT d i) (pidH d i) 0) ((⟨j, hjpad⟩ : Fin d.padded), ())
          = soIdx d (pidT d i) (pidH d i) j := copy_rowG_soIdx d i _
      rw [hgs] at hlds
      have hspanS : Mem.span .bf16
          (d.suffix_output.ptr (soIdx d (pidT d i) (pidH d i) j : Int)).addr ⊆
          d.prefix_output.bytes ∪ d.suffix_output.bytes :=
        fun x hx => Or.inr (d.suffix_output.span_sub (copy_soIdx_lt d hd i hj) hx)
      have hagS : m.read .bf16
            (d.suffix_output.ptr (soIdx d (pidT d i) (pidH d i) j : Int)).addr
          = init.read .bf16
            (d.suffix_output.ptr (soIdx d (pidT d i) (pidH d i) j : Int)).addr :=
        Mem.read_congr (fun x hx => hagree x (hspanS hx))
      have hsv_eq : s_out.get ((⟨j, hjpad⟩ : Fin d.padded), ()) = sv := by
        have h2 := hsvr
        unfold rdB at h2
        rw [← hagS] at h2
        rw [hlds] at h2
        exact Option.some.inj h2
      have hc1s : FloatV.NaNEq (FloatV.cvtWith .rn pv)
          (p32.get ((⟨j, hjpad⟩ : Fin d.padded), ())) := by
        have h := hc1 ((⟨j, hjpad⟩ : Fin d.padded), ())
        rwa [hpv_eq] at h
      have hc2s : FloatV.NaNEq (FloatV.cvtWith .rn sv)
          (s32.get ((⟨j, hjpad⟩ : Fin d.padded), ())) := by
        have h := hc2 ((⟨j, hjpad⟩ : Fin d.padded), ())
        rwa [hsv_eq] at h
      have has : AddRel true (prod_p.get ((⟨j, hjpad⟩ : Fin d.padded), ()))
          (prod_s.get ((⟨j, hjpad⟩ : Fin d.padded), ()))
          (out32a.get ((⟨j, hjpad⟩ : Fin d.padded), ())) := by
        have h := ha ((⟨j, hjpad⟩ : Fin d.padded), ())
        rwa [hfus] at h
      have hMeqF : ¬ (FloatV.eq M negInfC = true) := by
        intro h
        have hMneg : M = negInfC :=
          (eq_negInfC_iff (FloatV.isNaN_of_mem (corr_isNaN hpnan) (corr_isNaN hsnan)
            (hM.mem (corr_isNaN hpnan) (corr_isNaN hsnan)))).mp h
        exact hne (maxRel_eq_negInf_both' hM (corr_isNaN hpnan) (corr_isNaN hsnan) hMneg)
      have hclamp : (Tensor.where_ (bc ((bc M).feq (bc negInfC)).item) (bc zeroC) out32a :
            Tensor [d.padded] F32).get ((⟨j, hjpad⟩ : Fin d.padded), ())
          = out32a.get ((⟨j, hjpad⟩ : Fin d.padded), ()) := by
        rw [merge_where_get (d := d) M out32a ((⟨j, hjpad⟩ : Fin d.padded), ()), ite_eq_right hMeqF]
      have hc3s : FloatV.NaNEq (FloatV.cvtWith .rn (out32a.get ((⟨j, hjpad⟩ : Fin d.padded), ())))
          (out16.get ((⟨j, hjpad⟩ : Fin d.padded), ())) := by
        have h := hc3 ((⟨j, hjpad⟩ : Fin d.padded), ())
        rwa [hclamp] at h
      obtain ⟨hfin, hbound⟩ := merge_row_num hdiv (hm hne) hne hpvf hsvf
        (hpsc ()) (hssc ()) hc1s hc2s (hm1 _) (hm2 _) has hc3s
      exact ⟨pv, sv, hpvr, hsvr, hfin, hbound⟩

/-! ## The branch -/

/-- **The merge branch.** -/
theorem safe_merge {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size)
    (hpt : pidT d i < d.ptwc) :
    Safe ((launch d).env P) (Rf d (pidT d i) (pidH d i)) (Wf d (pidT d i) (pidH d i))
      (fun _ m => MergeOk d init m (pidT d i) (pidH d i)) init
      (mergeProg d ((launch d).ctx i) (olOf d)) := by
  obtain ⟨hvout, hvoutlse, hvpo, hvpl, hvso, hvsl, hvals⟩ := hpre
  simp only [mergeProg]
  rw [prefix_lse_ptr d hd i hpt]
  refine safe_load1_buf d.prefix_lse (merge_plIdx_lt d hd i hpt) hvpl
    (Rf_prefix_lse d (pidT d i) (pidH d i) hpt) ?_
  intro p_lse hreadP
  rw [suffix_lse_ptr d hd i]
  refine safe_load1_buf d.suffix_lse (copy_slIdx_lt d hd i) hvsl
    (Rf_suffix_lse d (pidT d i) (pidH d i)) ?_
  intro s_lse hreadS
  obtain ⟨p0, s0, hp0, hs0, hnanP, hnanS, hfin⟩ :=
    hvals (pidT d i) (pidH d i) hpt (pidH_lt d i)
  have hp_eq : p_lse = p0 :=
    Option.some.inj (hreadP.symm.trans hp0)
  have hs_eq : s_lse = s0 :=
    Option.some.inj (hreadS.symm.trans hs0)
  have hnanP' : p_lse.isNaN = false := by
    rw [hp_eq]
    exact hnanP
  have hnanS' : s_lse.isNaN = false := by
    rw [hs_eq]
    exact hnanS
  have hcorrP : (if ((bc p_lse).feq (bc posInfC)).item then negInfC else p_lse)
      = corr p_lse := by
    rw [merge_feq_item p_lse posInfC, ← ite_corr hnanP']
  have hcorrS : (if ((bc s_lse).feq (bc posInfC)).item then negInfC else s_lse)
      = corr s_lse := by
    rw [merge_feq_item s_lse posInfC, ← ite_corr hnanS']
  rw [hcorrP, hcorrS]
  refine safe_maximum ?_
  intro ml hmax
  refine safe_sub ?_
  intro ub hu
  refine safe_sub ?_
  intro vb hv
  refine safe_math (fn := .exp)
    (hex := fun k => hP.mathTotal .exp .binary32 ((bc (Tensor.item ub)).get k)) ?_
  intro pse hpe
  refine safe_math (fn := .exp)
    (hex := fun k => hP.mathTotal .exp .binary32 ((bc (Tensor.item vb)).get k)) ?_
  intro sse hse
  refine safe_add ?_
  intro ose ho
  have hval1 : ¬ bothEmpty p_lse s_lse → ∀ j, j < d.head_size → ∃ pv sv : BF16,
      rdB init d.prefix_output (poIdx d (pidT d i) (pidH d i) j) = some pv ∧
      rdB init d.suffix_output (soIdx d (pidT d i) (pidH d i) j) = some sv ∧
      pv.isFinite = true ∧ sv.isFinite = true := by
    intro hne
    exact hfin (by rwa [hp_eq, hs_eq] at hne)
  have hM' : FloatV.MaxRel true (corr p_lse) (corr s_lse) (Tensor.item ml) := hmax ()
  have hLM : ¬ bothEmpty p_lse s_lse →
      LseMid (corr p_lse) (corr s_lse) (Tensor.item ml) (Tensor.item ub) (Tensor.item vb)
        (Tensor.item pse) (Tensor.item sse) (Tensor.item ose) := by
    intro hne
    exact merge_lse_mid (P := P) hP.expF32 hP.expF32NegInf
      (corr_isNaN hnanP') (corr_isNaN hnanS')
      (corr_finite_or hnanP') (corr_finite_or hnanS') hne hM'
      (hu ()) (hv ()) (hpe ()) (hse ()) (ho ())
  by_cases hOL : d.outLse
  · have hol : olOf d = some (d.output_lse.ptr 0) := by simp [olOf, hOL]
    rw [hol]
    dsimp only
    rw [output_lse_ptr d hd i]
    have holL := copy_olIdx_lt d hd i hOL
    have hWst : Mem.span .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i))).addr ⊆
        Wf d (pidT d i) (pidH d i) := by
      exact Wf_output_lse d (pidT d i) (pidH d i) hOL
    refine safe_math (fn := .log)
      (hex := fun k => hP.mathTotal .log .binary32 ((bc (Tensor.item ose)).get k)) ?_
    intro lgt hlg
    refine safe_add ?_
    intro cl0t hcl0t
    rw [merge_where_item (Tensor.item ml) (Tensor.item cl0t)]
    refine safe_store1 (p := d.output_lse.ptr (olIdx d (pidT d i) (pidH d i)))
      (v := (if FloatV.eq (Tensor.item ml) negInfC then negInfC else Tensor.item cl0t))
      (d.output_lse.writable (hvoutlse hOL) holL) hWst ?_
    · intro m1 hres hfr
      have hframe : ∀ x, x ∉ Mem.span .f32
          (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr → m1 x = init x := by
        intro x hx
        exact hfr x hx
      have hvp' : d.prefix_output.Valid m1 := by
        refine Buf.valid_of_agree hvpo ?_
        intro x hx
        have hx' : x ∉ Mem.span .f32
            (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
          intro hspan
          exact Set.disjoint_left.mp (hd.output_lse_in_disj hOL)
            (d.output_lse.span_sub holL hspan) (Or.inl (Or.inl (Or.inl hx)))
        rw [hframe x hx']
        exact hvpo.2.2.2 x hx
      have hvs' : d.suffix_output.Valid m1 := by
        refine Buf.valid_of_agree hvso ?_
        intro x hx
        have hx' : x ∉ Mem.span .f32
            (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
          intro hspan
          exact Set.disjoint_left.mp (hd.output_lse_in_disj hOL)
            (d.output_lse.span_sub holL hspan) (Or.inl (Or.inl (Or.inr hx)))
        rw [hframe x hx']
        exact hvso.2.2.2 x hx
      have hvo' : d.output.Valid m1 := by
        refine Buf.valid_of_agree hvout ?_
        intro x hx
        have hx' : x ∉ Mem.span .f32
            (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr := by
          intro hspan
          exact Set.disjoint_left.mp (hd.output_lse_disj hOL) hx
            (d.output_lse.span_sub holL hspan)
        rw [hframe x hx']
        exact hvout.2.2.2 x hx
      have hagree1 : ∀ x ∈ d.prefix_output.bytes ∪ d.suffix_output.bytes, m1 x = init x := by
        intro x hx
        refine hframe x (fun hmem => ?_)
        rcases hx with hx | hx
        · exact Set.disjoint_left.mp (hd.output_lse_in_disj hOL)
            (d.output_lse.span_sub holL hmem) (Or.inl (Or.inl (Or.inl hx)))
        · exact Set.disjoint_left.mp (hd.output_lse_in_disj hOL)
            (d.output_lse.span_sub holL hmem) (Or.inl (Or.inl (Or.inr hx)))
      have hWs : d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) ⊆
          Wf d (pidT d i) (pidH d i) \ Mem.span .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i))).addr := by
        intro x hx
        refine ⟨Wf_output_row d (pidT d i) (pidH d i) hx, ?_⟩
        intro hspan
        exact Set.disjoint_left.mp (hd.output_lse_disj hOL)
          (d.output.elems_sub_bytes (fun j hj => copy_oIdx_row_lt hd i hj) hx)
          (d.output_lse.span_sub holL hspan)
      have hlse1 : d.outLse → ∃ cl : F32,
          (∀ m'' : Mem, (∀ x, x ∉ d.output.elems (row d (oIdx d (pidT d i) (pidH d i) 0)) →
            m'' x = m1 x) →
            rdF m'' d.output_lse (olIdx d (pidT d i) (pidH d i)) = some cl) ∧
          (bothEmpty p_lse s_lse → cl = negInfC) ∧
          (¬ bothEmpty p_lse s_lse → cl.isFinite = true ∧
            |cl.toReal - refLse (corr p_lse) (corr s_lse)| ≤
              (lseTol : ℝ) * (1 + |refLse (corr p_lse) (corr s_lse)|)) := by
        intro _
        refine ⟨(if FloatV.eq (Tensor.item ml) negInfC then negInfC else Tensor.item cl0t),
          ?_, ?_, ?_⟩
        · intro m'' hfr
          have hread' : m''.read .f32
                (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr
              = m1.read .f32
                (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr :=
            Mem.read_congr (fun x hx => hfr x (fun hmem =>
              Set.disjoint_left.mp (hd.output_lse_disj hOL)
                (d.output.elems_sub_bytes (fun j hj => copy_oIdx_row_lt hd i hj) hmem)
                (d.output_lse.span_sub holL hx)))
          show m''.read .f32 (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i) : Int)).addr
            = some (if FloatV.eq (Tensor.item ml) negInfC then negInfC
                else Tensor.item cl0t)
          rw [hread']
          exact hres
        · intro hBE
          have hMeq : Tensor.item ml = negInfC := by
            refine maxRel_eq_negInf_both ?_
            rwa [hBE.1, hBE.2] at hM'
          have hMn : (Tensor.item ml).isNaN = false := by
            rw [hMeq]
            exact negInfC_isNaN
          rw [ite_eq_left ((eq_negInfC_iff hMn).mpr hMeq)]
        · intro hne
          exact merge_lse_out (P := P) hP.logF32 (hLM hne) hne (hlg ()) (hcl0t ()) rfl
      refine merge_safe_row (d := d) (E := (launch d).env P) (init := init) (m := m1)
        (p_lse := p_lse) (s_lse := s_lse)
        (R := Rf d (pidT d i) (pidH d i))
        (W := Wf d (pidT d i) (pidH d i) \ Mem.span .f32
          (d.output_lse.ptr (olIdx d (pidT d i) (pidH d i))).addr)
        hd i hpt hM' hnanP' hnanS' hLM (hP.divTotal _) hP.divFull (env_fusion d P)
        (Set.union_subset (Rf_prefix_row d (pidT d i) (pidH d i) hpt)
          (Rf_suffix_row d (pidT d i) (pidH d i)))
        hWs hvp' hvs' hvo' hagree1 hreadP hreadS hval1 hlse1
  · have hol : olOf d = none := by simp [olOf, hOL]
    rw [hol]
    dsimp only
    refine merge_safe_row (d := d) (E := (launch d).env P) (init := init) (m := init)
      (p_lse := p_lse) (s_lse := s_lse)
      (R := Rf d (pidT d i) (pidH d i)) (W := Wf d (pidT d i) (pidH d i))
      hd i hpt hM' hnanP' hnanS' hLM (hP.divTotal _) hP.divFull (env_fusion d P)
      (Set.union_subset (Rf_prefix_row d (pidT d i) (pidH d i) hpt)
        (Rf_suffix_row d (pidT d i) (pidH d i)))
      (Wf_output_row d (pidT d i) (pidH d i)) hvpo hvso hvout
      (fun _ _ => rfl) hreadP hreadS hval1 (fun h => absurd h hOL)

end Claims.MergeAttnStates
