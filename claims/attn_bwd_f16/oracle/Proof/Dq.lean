import Proof.Pair
import Proof.Basic
import Proof.Lanes

/-!
# `attn_bwd_f16`: the loop over key rows (`_attn_bwd_dq`)
-/

namespace Claims.AttnBwd

open TVA

/-- `Σ_c Q[m, c]·K[n, c]` is `sQK`. -/
theorem sQK_swap (d : Inst) (hv : HeadV) (n m : Nat) :
    ∑ c ∈ Finset.range d.HEAD_DIM, (hv.q m c).toReal * (hv.k n c).toReal = sQK d hv n m :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

/-- `Σ_c lm Q[m, c]·lm K[n, c]` is `aQK`. -/
theorem aQK_swap (d : Inst) (hv : HeadV) (n m : Nat) :
    ∑ c ∈ Finset.range d.HEAD_DIM, lm (hv.q m c) * lm (hv.k n c) = aQK d hv n m :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

/-- `Σ_c dO[m, c]·V[n, c]` is `sVD`. -/
theorem sVD_swap (d : Inst) (hv : HeadV) (n m : Nat) :
    ∑ c ∈ Finset.range d.HEAD_DIM, (hv.dO m c).toReal * (hv.v n c).toReal = sVD d hv n m :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

/-- `Σ_c lm dO[m, c]·lm V[n, c]` is `aVD`. -/
theorem aVD_swap (d : Inst) (hv : HeadV) (n m : Nat) :
    ∑ c ∈ Finset.range d.HEAD_DIM, lm (hv.dO m c) * lm (hv.v n c) = aVD d hv n m :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

/-- **The `dS` tile of one trip is good** (the counterpart of `dkdv_good`): every lane `(i, j)` of
the `float16` tile `ds` (line 355) is `DSGood` for key row `a + j` and query row `q0 + i`. -/
theorem dq_good {E : Env} (hP : Device E.P) (d : Inst) (hd : d.ok) {init m0 : Mem}
    (hag : ∀ x ∈ inBytes d, m0 x = init x) {b : Nat} (hb : b < G d) {hv : HeadV}
    (hH : hv.Holds d init b) (hOk : hv.Ok d) {BN : Nat} (q0 a : Nat) (hq0 : q0 + 128 ≤ d.N_CTX)
    (ha : a + BN ≤ d.N_CTX) {q dO : Tensor [128, d.HEAD_DIM] F16}
    (hq : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.Q.ptr (eIdx d b (q0 + i) j)).addr = some (q.get2 i j))
    (hdO : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.DO.ptr (eIdx d b (q0 + i) j)).addr = some (dO.get2 i j))
    {mm : Tensor [128, 1] F32} {Di : Tensor [128] F32}
    (hmm : ∀ i : Fin 128, m0.read .f32 (d.M.ptr (rIdx d b (q0 + i))).addr = some (mm.get2 i 0))
    (hDi : ∀ i : Fin 128, m0.read .f32 (d.D.ptr (rIdx d b (q0 + i))).addr = some (Di.get1 i))
    {kT vT : Tensor [d.HEAD_DIM, BN] F16}
    (hkT : ∀ (j : Fin d.HEAD_DIM) (i : Fin BN),
      m0.read .f16 (d.K.ptr (eIdx d b (a + i) j)).addr = some (kT.get2 j i))
    (hvT : ∀ (j : Fin d.HEAD_DIM) (i : Fin BN),
      m0.read .f16 (d.V.ptr (eIdx d b (a + i) j)).addr = some (vT.get2 j i))
    {qk r1 pT pt' dp2 dpT r2 ds0 : Tensor [128, BN] F32} {ds : Tensor [128, BN] F16}
    (hqk : E.P.dot dotCfg q kT (tl.zeros [128, BN] .f32) qk)
    (hr1 : ∀ i : Idx [128, BN], AddRel E.fusion (qk.get i) ((mm.bcastCols BN).get i).neg (r1.get i))
    (hpT : ∀ i : Idx [128, BN], E.P.math .exp2 .binary32 (r1.get i) (pT.get i))
    (hlane : ∀ (i : Fin 128) (j : Fin BN),
      (Allowed d (a + j) (q0 + i) → pt'.get2 i j = pT.get2 i j) ∧
        (¬ Allowed d (a + j) (q0 + i) → pt'.get2 i j = FloatV.ofPat 0))
    (hdp2 : E.P.dot dotCfg dO vT (tl.zeros [128, BN] .f32) dp2)
    (hdpT : ∀ i : Idx [128, BN], ConvRel E.P .f32 .f32 (dp2.get i) (dpT.get i))
    (hr2 : ∀ i : Idx [128, BN], AddRel E.fusion (dpT.get i) ((Di.col BN).get i).neg (r2.get i))
    (hds0 : ∀ i : Idx [128, BN], ((pt'.get i).mul (r2.get i)).NaNEq (ds0.get i))
    (hds : ∀ i : Idx [128, BN], (FloatV.cvtWith .rn (ds0.get i)).NaNEq (ds.get i)) :
    ∀ (i : Fin 128) (j : Fin BN), DSGood d hv (a + j) (q0 + i) (ds.get2 i j) := by
  intro i j
  have hn : a + j < d.N_CTX := by have := j.isLt; omega
  have hm : q0 + i < d.N_CTX := by have := i.isLt; omega
  have hHD : d.HEAD_DIM ≤ 256 := by have := hd.hd; omega
  have hz : DType.val? .f32 ((tl.zeros [128, BN] .f32).get2 i j) = some 0 := DType.val?_zero_f32
  obtain ⟨vr1, hvr1, he1⟩ := dot_lane hP.dotF16 hHD hqk i j (fun c => hv.q (q0 + i) c)
    (fun c => hv.k (a + j) c)
    (fun c => q_eq hag hH hb hm c.isLt (hq i c)) (fun c => k_eq hag hH hb hn c.isLt (hkT c j))
    (fun c => hOk.q _ _ hm c.isLt) (fun c => hOk.k _ _ hn c.isLt) hz (by norm_num)
  obtain ⟨vr2, hvr2, he2⟩ := dot_lane hP.dotF16 hHD hdp2 i j (fun c => hv.dO (q0 + i) c)
    (fun c => hv.v (a + j) c)
    (fun c => do_eq hag hH hb hm c.isLt (hdO i c)) (fun c => v_eq hag hH hb hn c.isLt (hvT c j))
    (fun c => hOk.dO _ _ hm c.isLt) (fun c => hOk.v _ _ hn c.isLt) hz (by norm_num)
  obtain ⟨hf1, hv1⟩ := DType.val?_f32.mp hvr1
  obtain ⟨hf2, hv2⟩ := DType.val?_f32.mp hvr2
  have hdpTeq : dpT.get (i, j, ()) = dp2.get (i, j, ()) := by
    have h := hdpT (i, j, ())
    simp only [ConvRel, ofFloatRel, FloatRel, ite_true] at h
    exact FloatV.ext_pat h
  refine (pair_good (fusion := E.fusion) hP.exp2F32 d hd hv hOk hn hm (d1 := qk.get2 i j)
    (d2 := dpT.get2 i j) (r1 := r1.get2 i j) (r2 := r2.get2 i j) (pt := pT.get2 i j)
    (pt' := pt'.get2 i j) (ds := ds0.get2 i j) (p16 := FloatV.cvt (pt'.get2 i j))
    (q16 := ds.get2 i j) ?_ ?_ ?_ ?_ ?_ ?_ ?_ (FloatV.NaNEq.refl _) ?_).2
  · refine ⟨hf1, ?_⟩
    have e : (qk.get2 i j).toReal = (vr1 : ℝ) := by unfold FloatV.toReal; rw [hv1]
    rw [e]; simpa [sQK_swap, aQK_swap, rH, aH] using he1
  · refine ⟨by rw [show dpT.get2 i j = dp2.get2 i j from hdpTeq]; exact hf2, ?_⟩
    have e : (dpT.get2 i j).toReal = (vr2 : ℝ) := by
      rw [show dpT.get2 i j = dp2.get2 i j from hdpTeq]; unfold FloatV.toReal; rw [hv2]
    rw [e]; simpa [sVD_swap, aVD_swap, rH, aH] using he2
  · have e : (mm.bcastCols BN).get (i, j, ()) = hv.lse (q0 + i) := lse_eq hag hH hb hm (hmm i)
    have := hr1 (i, j, ())
    rwa [e] at this
  · exact hpT (i, j, ())
  · exact hlane i j
  · have e : (Di.col BN).get (i, j, ()) = hv.dl (q0 + i) := dl_eq hag hH hb hm (hDi i)
    have := hr2 (i, j, ())
    rwa [e] at this
  · exact hds0 (i, j, ())
  · exact hds (i, j, ())

/-- **The loop over key rows** (`_attn_bwd_dq`, lines 319–363) is safe and keeps the invariant:
for the `128` query rows `q0, …, q0 + 127` of head `b`, with an accumulator `dq` that satisfies
`DqInv` over the key rows `S0` after `t0` steps, the call over the key rows
`n0, …, n0 + T·BN - 1` (blocks of `BN = 16` or `32`, masked or not) leaves an accumulator that
satisfies `DqInv` over `S0 ∪ [n0, n0 + T·BN)` after `t0 + T` steps, and does not touch memory. -/
theorem dq_safe {E : Env} (hP : Device E.P) (d : Inst) (hd : d.ok) {init m0 : Mem}
    (hpre : Pre d m0) (hag : ∀ x ∈ inBytes d, m0 x = init x) {b : Nat} (hb : b < G d)
    (BN : Nat) (hBN : BN = 16 ∨ BN = 32) (MASK : Bool) (hmaskC : MASK = true → d.CAUSAL = true)
    (q0 n0 T t0 : Nat) (S0 : Finset Nat) (hq0 : q0 + 128 ≤ d.N_CTX) (hT : n0 + T * BN ≤ d.N_CTX)
    (ht : t0 + T ≤ 1024) (hS0 : ∀ s ∈ S0, s < d.N_CTX ∧ (s < n0 ∨ n0 + T * BN ≤ s))
    (hAl : MASK = false → d.CAUSAL = false ∨ n0 + T * BN ≤ q0)
    {Kp Vp : Ptr .f16} {Dp : Ptr .f32}
    (hK : Kp = d.K.ptr ((b * d.N_CTX * d.HEAD_DIM : Nat) : Int))
    (hV : Vp = d.V.ptr ((b * d.N_CTX * d.HEAD_DIM : Nat) : Int))
    (hD : Dp = d.D.ptr ((b * d.N_CTX : Nat) : Int))
    {stride_tok stride_d H N start_m start_n num_steps : I32}
    (hs1 : stride_tok.toInt = d.HEAD_DIM) (hs2 : stride_d.toInt = 1)
    (hsm : start_m.toInt = q0) (hsn : start_n.toInt = n0) (hns : num_steps.toInt = T)
    {dq : Tensor [128, d.HEAD_DIM] F32} {q dO : Tensor [128, d.HEAD_DIM] F16}
    {mm : Tensor [128, 1] F32}
    (hq : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.Q.ptr (eIdx d b (q0 + i) j)).addr = some (q.get2 i j))
    (hdO : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.DO.ptr (eIdx d b (q0 + i) j)).addr = some (dO.get2 i j))
    (hmm : ∀ i : Fin 128, m0.read .f32 (d.M.ptr (rIdx d b (q0 + i))).addr = some (mm.get2 i 0))
    (hinv : DqInv d init b q0 S0 t0 dq) :
    Safe E (inBytes d) ∅
      (fun r m' => m' = m0 ∧ DqInv d init b q0 (S0 ∪ Finset.Ico n0 (n0 + T * BN)) (t0 + T) r)
      m0 (attn_bwd_dq 128 BN d.HEAD_DIM dq q Kp Vp dO mm Dp stride_tok stride_d H N start_m
        start_n num_steps MASK) := by
  unfold attn_bwd_dq
  simp only []
  have hHDle : d.HEAD_DIM ≤ 256 := by have := hd.hd; omega
  have hN := hd.n_le
  have hrowq : ∀ i : Fin 128, q0 + i < d.N_CTX := fun i => by have := i.isLt; omega
  refine safe_load_vec d d.D hd.d_len (D_sub_in d) hpre.2.2.2.2.2.2.2.2 hb
    (r := fun i : Fin 128 => q0 + i) hrowq ?_ fun Di hDi => ?_
  · intro i
    have hl := IntV.I32.toInt_sadd_arange (n := 128) hsm (by omega) i
    show Dp.offset ((start_m + tl.arange 0 128).get1 i).toInt = _
    rw [hl, hD, Buf.ptr_offset]; congr 1
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have h1 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) num_steps (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [h0, h1, hns]
    simp only [IntV.maxInt]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    have : T * BN ≤ 2 ^ 14 := by omega
    have : T ≤ 2 ^ 14 := by rcases hBN with rfl | rfl <;> omega
    omega
  have htrip : tl.tripCount (IntV.ofInt 0 : I32).toInt num_steps.toInt (IntV.ofInt 1 : I32).toInt = T := by
    rw [h0, h1, hns]; simp [tl.tripCount]
  refine Safe.mono (W := iterW (fun _ => (∅ : Set Int))
    (tl.tripCount (IntV.ofInt 0 : I32).toInt num_steps.toInt (IntV.ofInt 1 : I32).toInt) ∪ ∅)
    ?_ (fun _ _ h => h) (by simp)
  refine safe_forRange hrange
    (fun t st m' => m' = m0 ∧ st.2.2.1.toInt = ((n0 + t * BN : Nat) : Int) ∧
      (∀ (c : Fin d.HEAD_DIM) (j : Fin BN), st.2.2.2.1.get2 c j =
        d.K.ptr ((b * d.N_CTX * d.HEAD_DIM + c + (n0 + t * BN + j) * d.HEAD_DIM : Nat) : Int)) ∧
      (∀ (c : Fin d.HEAD_DIM) (j : Fin BN), st.2.2.2.2.get2 c j =
        d.V.ptr ((b * d.N_CTX * d.HEAD_DIM + c + (n0 + t * BN + j) * d.HEAD_DIM : Nat) : Int)) ∧
      DqInv d init b q0 (S0 ∪ Finset.Ico n0 (n0 + t * BN)) (t0 + t) st.1)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by simp) ?_ ?_ ?_
  · refine ⟨rfl, ?_, ?_, ?_, ?_⟩
    · show start_n.toInt = _; rw [hsn]; simp
    · have hfitA : (n0 + BN - 1) * d.HEAD_DIM < 2 ^ 31 :=
        lt_of_le_of_lt (Nat.mul_le_mul (show n0 + BN - 1 ≤ 2 ^ 15 by omega) hHDle) (by norm_num)
      have hfitB : (d.HEAD_DIM - 1) * 1 < 2 ^ 31 := by omega
      intro c j
      show (Kp + (start_n + tl.arange 0 BN).row d.HEAD_DIM * stride_tok +
        (tl.arange 0 d.HEAD_DIM).col BN * stride_d).get2 c j = _
      rw [ptrT_lane d.K hK hsn hs1 hs2 (by omega) hfitA (by omega) hfitB c j]
      congr 2; simp
    · have hfitA : (n0 + BN - 1) * d.HEAD_DIM < 2 ^ 31 :=
        lt_of_le_of_lt (Nat.mul_le_mul (show n0 + BN - 1 ≤ 2 ^ 15 by omega) hHDle) (by norm_num)
      have hfitB : (d.HEAD_DIM - 1) * 1 < 2 ^ 31 := by omega
      intro c j
      show (Vp + (start_n + tl.arange 0 BN).row d.HEAD_DIM * stride_tok +
        (tl.arange 0 d.HEAD_DIM).col BN * stride_d).get2 c j = _
      rw [ptrT_lane d.V hV hsn hs1 hs2 (by omega) hfitA (by omega) hfitB c j]
      congr 2; simp
    · simpa using hinv
  · intro t ht st m1 hI
    obtain ⟨dq1, om, cn, kp, vp⟩ := st
    obtain ⟨hm1, hcn, hkp, hvp, hinv1⟩ := hI
    have hm1' := hm1.symm
    subst hm1'
    simp only [] at hcn hkp hvp hinv1 ⊢
    have htT : t < T := htrip ▸ ht
    have hTB : (t + 1) * BN = t * BN + BN := by ring
    have hTB' : (t + 1) * BN ≤ T * BN := Nat.mul_le_mul_right _ htT
    have hBN0 : 0 < BN := by omega
    have hfit : n0 + t * BN + BN ≤ 2 ^ 31 := by omega
    have hrow : ∀ j : Fin BN, n0 + t * BN + j < d.N_CTX := by
      intro j
      have := j.isLt
      omega
    refine safe_load_cr d d.K hd.k_len (K_sub_in d) hpre.2.1 hb (r := fun j : Fin BN => n0 + t * BN + j)
      (c := fun c : Fin d.HEAD_DIM => c.val) hrow (fun j => j.isLt) ?_ fun kT hkT => ?_
    · intro c j
      rw [hkp c j]; congr 2; unfold eIdx; ring
    refine safe_load_cr d d.V hd.v_len (V_sub_in d) hpre.2.2.1 hb (r := fun j : Fin BN => n0 + t * BN + j)
      (c := fun c : Fin d.HEAD_DIM => c.val) hrow (fun j => j.isLt) ?_ fun vT hvT => ?_
    · intro c j
      rw [hvp c j]; congr 2; unfold eIdx; ring
    refine safe_dot (hP.dotTotal _ _ _ _) fun qk hqk => ?_
    refine safe_subc fun r1 hr1 => ?_
    refine safe_math (fn := .exp2) (fun _ => hP.mathTotal _ _ _) fun pT hpT => ?_
    refine safe_dot (hP.dotTotal _ _ _ _) fun dp2 hdp2 => ?_
    refine safe_convert (fun i => ⟨dp2.get i, by simp [ConvRel, ofFloatRel, FloatRel]⟩) fun dpT hdpT => ?_
    refine safe_subc fun r2 hr2 => ?_
    obtain ⟨pt', hpt'eq, hlane⟩ : ∃ pt' : Tensor [128, BN] F32,
        (if MASK = true then
          tl.where (((if MASK = true then cn + tl.arange 0 BN else om).row 128).le
            ((start_m + tl.arange 0 128).col BN)) pT (tl.full [128, BN] (FloatV.ofPat 0).litF32)
        else pT) = pt' ∧
        ∀ (i : Fin 128) (j : Fin BN),
          (Allowed d (n0 + t * BN + j) (q0 + i) → pt'.get2 i j = pT.get2 i j) ∧
          (¬ Allowed d (n0 + t * BN + j) (q0 + i) → pt'.get2 i j = FloatV.ofPat 0) := by
      refine ⟨_, rfl, fun i j => ?_⟩
      have hnl : ∀ j' : Fin BN, ((cn + tl.arange 0 BN).get1 j').toInt =
          ((n0 + t * BN + j' : Nat) : Int) := fun j' => IntV.I32.toInt_sadd_arange hcn hfit j'
      have hml : ∀ i' : Fin 128, ((start_m + tl.arange 0 128).get1 i').toInt =
          ((q0 + i' : Nat) : Int) := fun i' => IntV.I32.toInt_sadd_arange (n := 128) hsm (by omega) i'
      have hjj := j.isLt
      have hii := i.isLt
      cases MASK with
      | false =>
        simp only [Bool.false_eq_true, ite_false]
        refine ⟨fun _ => trivial, fun hna => absurd ?_ hna⟩
        rcases hAl rfl with h | h
        · exact Or.inl h
        · exact Or.inr (by omega)
      | true =>
        simp only [ite_true]
        have hC := hmaskC rfl
        rw [where_lane_T hml hnl]
        refine ⟨fun hal => ?_, fun hna => ?_⟩
        · have h1 : n0 + t * BN + j ≤ q0 + i := by
            rcases hal with h | h
            · rw [hC] at h; cases h
            · exact h
          simp [h1]
        · have h1 : ¬ (n0 + t * BN + j ≤ q0 + i) := fun h => hna (Or.inr h)
          simp [h1]
    rw [hpt'eq]
    refine safe_mul fun ds0 hds0 => ?_
    refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
      fun ds hds => ?_
    refine safe_dot_add (hP.dotTotal _ _ _ _) fun dqn hdqn => ?_
    have hBNHD : BN * d.HEAD_DIM ≤ 8192 := by
      rcases hBN with rfl | rfl <;> omega
    have hstep : (IntV.ofNat BN * stride_tok : I32).toInt = ((BN * d.HEAD_DIM : Nat) : Int) :=
      IntV.I32.mul_nat (IntV.I32.toInt_ofNat (by omega)) hs1 (by omega)
    refine Safe.ret ⟨rfl, ?_, ?_, ?_, ?_⟩
    · show (cn + IntV.ofNat BN : I32).toInt = _
      have := IntV.I32.add_nat hcn (IntV.I32.toInt_ofNat (n := BN) (by omega)) (by omega)
      rw [this]; congr 1; rw [hTB]; ring
    · intro c j
      show ((kp.get2 c j).offset (IntV.ofNat BN * stride_tok : I32).toInt) = _
      rw [hkp c j, hstep, Buf.ptr_offset]; congr 1; push_cast; ring
    · intro c j
      show ((vp.get2 c j).offset (IntV.ofNat BN * stride_tok : I32).toInt) = _
      rw [hvp c j, hstep, Buf.ptr_offset]; congr 1; push_cast; ring
    · intro hv' hH hOk
      have hgood := dq_good hP d hd hag hb hH hOk q0 (n0 + t * BN) hq0 (by omega) hq hdO hmm hDi
        hkT hvT hqk hr1 hpT hlane hdp2 hdpT hr2 hds0 hds
      have hdq1 := hinv1 hv' hH hOk
      have hsN : ∀ s, (s ∈ S0 ∪ Finset.Ico n0 (n0 + t * BN) ∨
          (n0 + t * BN ≤ s ∧ s < n0 + t * BN + BN)) → s < d.N_CTX := by
        intro s hs
        rcases hs with hs | hs
        · rcases Finset.mem_union.mp hs with hs | hs
          · exact (hS0 s hs).1
          · have := (Finset.mem_Ico.mp hs).2; omega
        · omega
      have hdisj : ∀ s ∈ S0 ∪ Finset.Ico n0 (n0 + t * BN),
          s < n0 + t * BN ∨ n0 + t * BN + BN ≤ s := by
        intro s hs
        rcases Finset.mem_union.mp hs with hs | hs
        · rcases (hS0 s hs).2 with h | h
          · left; omega
          · right; omega
        · left; exact (Finset.mem_Ico.mp hs).2
      have hS : ∀ s ∈ S0 ∪ Finset.Ico n0 (n0 + t * BN), s < 2 ^ 14 :=
        fun s hs => lt_of_lt_of_le (hsN s (Or.inl hs)) hN
      have hset : (S0 ∪ Finset.Ico n0 (n0 + t * BN)) ∪ Finset.Ico (n0 + t * BN) (n0 + t * BN + BN) =
          S0 ∪ Finset.Ico n0 (n0 + (t + 1) * BN) := by
        rw [Finset.union_assoc, Finset.Ico_union_Ico_eq_Ico (by omega) (by omega), hTB]
        congr 2; ring
      have hz : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
          DType.val? .f32 ((Tensor.full (DType.f32.ofBits 0) : Tensor [128, d.HEAD_DIM] F32).get2 i j)
            = some 0 := fun _ _ => DType.val?_zero_f32
      have hrw : t0 + (t + 1) = t0 + t + 1 := by ring
      have := AccInv.step (fusion := E.fusion) hP.dotF16 (by omega) (famDQ d hv')
        (fun i : Fin 128 => q0 + i) (fun j : Fin d.HEAD_DIM => j.val) hdisj hS (by omega) hdqn hz
        hdq1 (fun i k => hgood i k)
        (fun k j => (k_eq hag hH hb (by omega) j.isLt (hkT j k)))
        (fun s hs j => hOk.k s _ (hsN s hs) j.isLt)
        (fun i s hs => ds16L_le hOk (hsN s hs) (by have := i.isLt; omega))
      rw [hset, ← hrw] at this
      exact this
  · intro st m1 hI
    obtain ⟨dq1, om, cn, kp, vp⟩ := st
    obtain ⟨hm1, -, -, -, hinv1⟩ := hI
    refine Safe.ret ⟨hm1, ?_⟩
    rw [htrip] at hinv1
    exact hinv1

end Claims.AttnBwd
