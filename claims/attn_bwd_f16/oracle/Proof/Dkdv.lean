import Proof.Pair
import Proof.Basic
import Proof.Lanes

/-!
# `attn_bwd_f16`: the loop over query rows (`_attn_bwd_dkdv`)
-/

namespace Claims.AttnBwd

open TVA

/-- **The `P` and `dS` tiles of one trip are good.** Every lane `(i, kk)` of the `float16` tiles
`ppT` and `dsT` that the trip computes (lines 302 and 309) is `PGood` / `DSGood` for the pair of key
row `n0 + i` and query row `a + kk`, given what the walk of the trip knows about the platform's
results at its lines 293–309. -/
theorem dkdv_good {E : Env} (hP : Device E.P) (d : Inst) (hd : d.ok) {init m0 : Mem}
    (hag : ∀ x ∈ inBytes d, m0 x = init x) {b : Nat} (hb : b < G d) {hv : HeadV}
    (hH : hv.Holds d init b) (hOk : hv.Ok d) {BM : Nat} (n0 a : Nat) (hn0 : n0 + 128 ≤ d.N_CTX)
    (ha : a + BM ≤ d.N_CTX) {k v : Tensor [128, d.HEAD_DIM] F16}
    (hk : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.K.ptr (eIdx d b (n0 + i) j)).addr = some (k.get2 i j))
    (hvk : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.V.ptr (eIdx d b (n0 + i) j)).addr = some (v.get2 i j))
    {qT : Tensor [d.HEAD_DIM, BM] F16}
    (hqT : ∀ (j : Fin d.HEAD_DIM) (i : Fin BM),
      m0.read .f16 (d.Q.ptr (eIdx d b (a + i) j)).addr = some (qT.get2 j i))
    {dO : Tensor [BM, d.HEAD_DIM] F16}
    (hdO : ∀ (i : Fin BM) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.DO.ptr (eIdx d b (a + i) j)).addr = some (dO.get2 i j))
    {mv Di : Tensor [BM] F32}
    (hmv : ∀ i : Fin BM, m0.read .f32 (d.M.ptr (rIdx d b (a + i))).addr = some (mv.get1 i))
    (hDi : ∀ i : Fin BM, m0.read .f32 (d.D.ptr (rIdx d b (a + i))).addr = some (Di.get1 i))
    {qkT r1 pT pt' dp2 dpT r2 dsT0 : Tensor [128, BM] F32} {ppT dsT : Tensor [128, BM] F16}
    (hqkT : E.P.dot dotCfg k qT (tl.zeros [128, BM] .f32) qkT)
    (hr1 : ∀ i : Idx [128, BM], AddRel E.fusion (qkT.get i) ((mv.row 128).get i).neg (r1.get i))
    (hpT : ∀ i : Idx [128, BM], E.P.math .exp2 .binary32 (r1.get i) (pT.get i))
    (hlane : ∀ (i : Fin 128) (kk : Fin BM),
      (Allowed d (n0 + i) (a + kk) → pt'.get2 i kk = pT.get2 i kk) ∧
        (¬ Allowed d (n0 + i) (a + kk) → pt'.get2 i kk = FloatV.ofPat 0))
    (hppT : ∀ i : Idx [128, BM], (FloatV.cvtWith .rn (pt'.get i)).NaNEq (ppT.get i))
    (hdp2 : E.P.dot dotCfg v dO.trans (tl.zeros [128, BM] .f32) dp2)
    (hdpT : ∀ i : Idx [128, BM], ConvRel E.P .f32 .f32 (dp2.get i) (dpT.get i))
    (hr2 : ∀ i : Idx [128, BM], AddRel E.fusion (dpT.get i) ((Di.row 128).get i).neg (r2.get i))
    (hdsT0 : ∀ i : Idx [128, BM], ((pt'.get i).mul (r2.get i)).NaNEq (dsT0.get i))
    (hdsT : ∀ i : Idx [128, BM], (FloatV.cvtWith .rn (dsT0.get i)).NaNEq (dsT.get i)) :
    ∀ (i : Fin 128) (kk : Fin BM),
      PGood d hv (n0 + i) (a + kk) (ppT.get2 i kk) ∧ DSGood d hv (n0 + i) (a + kk) (dsT.get2 i kk) := by
  intro i kk
  have hn : n0 + i < d.N_CTX := by have := i.isLt; omega
  have hm : a + kk < d.N_CTX := by have := kk.isLt; omega
  have hHD : d.HEAD_DIM ≤ 256 := by have := hd.hd; omega
  have hz : DType.val? .f32 ((tl.zeros [128, BM] .f32).get2 i kk) = some 0 := DType.val?_zero_f32
  -- the dot of `k` and `qT`
  obtain ⟨vr1, hvr1, he1⟩ := dot_lane hP.dotF16 hHD hqkT i kk (fun c => hv.k (n0 + i) c)
    (fun c => hv.q (a + kk) c)
    (fun c => k_eq hag hH hb hn c.isLt (hk i c)) (fun c => q_eq hag hH hb hm c.isLt (hqT c kk))
    (fun c => hOk.k _ _ hn c.isLt) (fun c => hOk.q _ _ hm c.isLt) hz (by norm_num)
  obtain ⟨vr2, hvr2, he2⟩ := dot_lane hP.dotF16 hHD hdp2 i kk (fun c => hv.v (n0 + i) c)
    (fun c => hv.dO (a + kk) c)
    (fun c => v_eq hag hH hb hn c.isLt (hvk i c)) (fun c => do_eq hag hH hb hm c.isLt (hdO kk c))
    (fun c => hOk.v _ _ hn c.isLt) (fun c => hOk.dO _ _ hm c.isLt) hz (by norm_num)
  obtain ⟨hf1, hv1⟩ := DType.val?_f32.mp hvr1
  obtain ⟨hf2, hv2⟩ := DType.val?_f32.mp hvr2
  have hdpTeq : dpT.get (i, kk, ()) = dp2.get (i, kk, ()) := by
    have h := hdpT (i, kk, ())
    simp only [ConvRel, ofFloatRel, FloatRel, ite_true] at h
    exact FloatV.ext_pat h
  refine pair_good (fusion := E.fusion) hP.exp2F32 d hd hv hOk hn hm (d1 := qkT.get2 i kk) (d2 := dpT.get2 i kk)
    (r1 := r1.get2 i kk) (r2 := r2.get2 i kk) (pt := pT.get2 i kk) (pt' := pt'.get2 i kk)
    (ds := dsT0.get2 i kk) (p16 := ppT.get2 i kk) (q16 := dsT.get2 i kk) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · refine ⟨hf1, ?_⟩
    have e : (qkT.get2 i kk).toReal = (vr1 : ℝ) := by unfold FloatV.toReal; rw [hv1]
    rw [e]; simpa [sQK, aQK, rH, aH] using he1
  · refine ⟨by rw [show dpT.get2 i kk = dp2.get2 i kk from hdpTeq]; exact hf2, ?_⟩
    have e : (dpT.get2 i kk).toReal = (vr2 : ℝ) := by
      rw [show dpT.get2 i kk = dp2.get2 i kk from hdpTeq]; unfold FloatV.toReal; rw [hv2]
    rw [e]; simpa [sVD, aVD, rH, aH] using he2
  · have e : (mv.row 128).get (i, kk, ()) = hv.lse (a + kk) := lse_eq hag hH hb hm (hmv kk)
    have := hr1 (i, kk, ())
    rwa [e] at this
  · exact hpT (i, kk, ())
  · exact hlane i kk
  · have e : (Di.row 128).get (i, kk, ()) = hv.dl (a + kk) := dl_eq hag hH hb hm (hDi kk)
    have := hr2 (i, kk, ())
    rwa [e] at this
  · exact hdsT0 (i, kk, ())
  · exact hppT (i, kk, ())
  · exact hdsT (i, kk, ())

/-- **The loop over query rows** (`_attn_bwd_dkdv`, lines 266–315) is safe and keeps the invariant:
for the `128` key rows `n0, …, n0 + 127` of head `b`, with accumulators `dk`, `dv` that satisfy
`DkdvInv` over the query rows `S0` after `t0` steps, the call over the query rows
`s0, …, s0 + T·BM - 1` (blocks of `BM = 16` or `32`, masked or not) leaves accumulators that satisfy
`DkdvInv` over `S0 ∪ [s0, s0 + T·BM)` after `t0 + T` steps, and does not touch memory. `m0` is the
memory (it agrees with `init` on the inputs). -/
theorem dkdv_safe {E : Env} (hP : Device E.P) (d : Inst) (hd : d.ok) {init m0 : Mem}
    (hpre : Pre d m0) (hag : ∀ x ∈ inBytes d, m0 x = init x) {b : Nat} (hb : b < G d)
    (BM : Nat) (hBM : BM = 16 ∨ BM = 32) (MASK : Bool) (hmaskC : MASK = true → d.CAUSAL = true)
    (n0 s0 T t0 : Nat) (S0 : Finset Nat) (hn0 : n0 + 128 ≤ d.N_CTX) (hT : s0 + T * BM ≤ d.N_CTX)
    (ht : t0 + T ≤ 1024) (hS0 : ∀ s ∈ S0, s < d.N_CTX ∧ (s < s0 ∨ s0 + T * BM ≤ s))
    (hAl : MASK = false → d.CAUSAL = false ∨ n0 + 128 ≤ s0)
    {Qp DOp : Ptr .f16} {Mp Dp : Ptr .f32}
    (hQ : Qp = d.Q.ptr ((b * d.N_CTX * d.HEAD_DIM : Nat) : Int))
    (hDO : DOp = d.DO.ptr ((b * d.N_CTX * d.HEAD_DIM : Nat) : Int))
    (hM : Mp = d.M.ptr ((b * d.N_CTX : Nat) : Int))
    (hD : Dp = d.D.ptr ((b * d.N_CTX : Nat) : Int))
    {stride_tok stride_d H N start_n start_m num_steps : I32} {sm : F32}
    (hs1 : stride_tok.toInt = d.HEAD_DIM) (hs2 : stride_d.toInt = 1)
    (hsn : start_n.toInt = n0) (hsm : start_m.toInt = s0) (hns : num_steps.toInt = T)
    {dk dv : Tensor [128, d.HEAD_DIM] F32} {k v : Tensor [128, d.HEAD_DIM] F16}
    (hk : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.K.ptr (eIdx d b (n0 + i) j)).addr = some (k.get2 i j))
    (hv : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      m0.read .f16 (d.V.ptr (eIdx d b (n0 + i) j)).addr = some (v.get2 i j))
    (hinv : DkdvInv d init b n0 S0 t0 dk dv) :
    Safe E (inBytes d) ∅
      (fun r m' => m' = m0 ∧ DkdvInv d init b n0 (S0 ∪ Finset.Ico s0 (s0 + T * BM)) (t0 + T) r.1 r.2)
      m0 (attn_bwd_dkdv BM 128 d.HEAD_DIM dk dv Qp k v sm DOp Mp Dp stride_tok stride_d H N
        start_n start_m num_steps MASK) := by
  unfold attn_bwd_dkdv
  simp only []
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have h1 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hHDle : d.HEAD_DIM ≤ 256 := by have := hd.hd; omega
  have hN := hd.n_le
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) num_steps (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [h0, h1, hns]
    simp only [IntV.maxInt]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    have : T * BM ≤ 2 ^ 14 := by omega
    have : T ≤ 2 ^ 14 := by rcases hBM with rfl | rfl <;> omega
    omega
  have htrip : tl.tripCount (IntV.ofInt 0 : I32).toInt num_steps.toInt (IntV.ofInt 1 : I32).toInt = T := by
    rw [h0, h1, hns]; simp [tl.tripCount]
  refine Safe.mono (W := iterW (fun _ => (∅ : Set Int))
    (tl.tripCount (IntV.ofInt 0 : I32).toInt num_steps.toInt (IntV.ofInt 1 : I32).toInt) ∪ ∅)
    ?_ (fun _ _ h => h) (by simp)
  refine safe_forRange hrange
    (fun t st m' => m' = m0 ∧ st.2.2.2.1.toInt = ((s0 + t * BM : Nat) : Int) ∧
      (∀ (c : Fin d.HEAD_DIM) (i : Fin BM), st.2.2.2.2.1.get2 c i =
        d.Q.ptr ((b * d.N_CTX * d.HEAD_DIM + c + (s0 + t * BM + i) * d.HEAD_DIM : Nat) : Int)) ∧
      (∀ (i : Fin BM) (c : Fin d.HEAD_DIM), st.2.2.2.2.2.get2 i c =
        d.DO.ptr ((b * d.N_CTX * d.HEAD_DIM + (s0 + t * BM + i) * d.HEAD_DIM + c : Nat) : Int)) ∧
      DkdvInv d init b n0 (S0 ∪ Finset.Ico s0 (s0 + t * BM)) (t0 + t) st.1 st.2.1)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by simp) ?_ ?_ ?_
  · refine ⟨rfl, ?_, ?_, ?_, ?_⟩
    · show start_m.toInt = _; rw [hsm]; simp
    · have hfitA : (s0 + BM - 1) * d.HEAD_DIM < 2 ^ 31 :=
        lt_of_le_of_lt (Nat.mul_le_mul (show s0 + BM - 1 ≤ 2 ^ 15 by omega) hHDle) (by norm_num)
      have hfitB : (d.HEAD_DIM - 1) * 1 < 2 ^ 31 := by omega
      intro c i
      show (Qp + (start_m + tl.arange 0 BM).row d.HEAD_DIM * stride_tok +
        (tl.arange 0 d.HEAD_DIM).col BM * stride_d).get2 c i = _
      rw [ptrT_lane d.Q hQ hsm hs1 hs2 (by omega) hfitA (by omega) hfitB c i]
      congr 2; simp
    · have hfitA : (s0 + BM - 1) * d.HEAD_DIM < 2 ^ 31 :=
        lt_of_le_of_lt (Nat.mul_le_mul (show s0 + BM - 1 ≤ 2 ^ 15 by omega) hHDle) (by norm_num)
      have hfitB : (d.HEAD_DIM - 1) * 1 < 2 ^ 31 := by omega
      intro i c
      show (DOp + (start_m + tl.arange 0 BM).col d.HEAD_DIM * stride_tok +
        (tl.arange 0 d.HEAD_DIM).row BM * stride_d).get2 i c = _
      rw [ptrN_lane d.DO hDO hsm hs1 hs2 (by omega) hfitA (by omega) hfitB i c]
      congr 2; ring
    · simpa using hinv
  · intro t ht st m1 hI
    obtain ⟨dk1, dv1, om, cm, qp, dp⟩ := st
    obtain ⟨hm1, hcm, hqp, hdp, hinv1⟩ := hI
    have hm1' := hm1.symm
    subst hm1'
    simp only [] at hcm hqp hdp hinv1 ⊢
    have htT : t < T := htrip ▸ ht
    have hHD := hd.hd
    have hBM0 : 0 < BM := by omega
    have hrow : ∀ i : Fin BM, s0 + t * BM + i < d.N_CTX := by
      intro i
      have h1 : (t + 1) * BM ≤ T * BM := Nat.mul_le_mul_right _ htT
      have := i.isLt
      nlinarith
    have hN := hd.n_le
    have hTB : (t + 1) * BM = t * BM + BM := by ring
    have hTB' : (t + 1) * BM ≤ T * BM := Nat.mul_le_mul_right _ htT
    have hfit : s0 + t * BM + BM ≤ 2 ^ 31 := by omega
    refine safe_load_cr d d.Q hd.q_len (Q_sub_in d) hpre.1 hb (r := fun i : Fin BM => s0 + t * BM + i)
      (c := fun c : Fin d.HEAD_DIM => c.val) hrow (fun j => j.isLt) ?_ fun qT hqT => ?_
    · intro j i
      rw [hqp j i]; congr 2; unfold eIdx; ring
    refine safe_load_vec d d.M hd.m_len (M_sub_in d) hpre.2.2.2.2.2.2.2.1 hb
      (r := fun i : Fin BM => s0 + t * BM + i) hrow ?_ fun mv hmv => ?_
    · intro i
      have hl := IntV.I32.toInt_sadd_arange (n := BM) hcm hfit i
      show Mp.offset ((cm + tl.arange 0 BM).get1 i).toInt = _
      rw [hl, hM, Buf.ptr_offset]; congr 1
    refine safe_dot (hP.dotTotal _ _ _ _) fun qkT hqkT => ?_
    refine safe_subc fun r1 hr1 => ?_
    refine safe_math (fn := .exp2) (fun _ => hP.mathTotal _ _ _) fun pT hpT => ?_
    refine safe_load_rc d d.DO hd.do_len (DO_sub_in d) hpre.2.2.2.1 hb
      (r := fun i : Fin BM => s0 + t * BM + i) (c := fun c : Fin d.HEAD_DIM => c.val) hrow
      (fun j => j.isLt) ?_ fun dO hdO => ?_
    · intro i j
      rw [hdp i j]; congr 2; unfold eIdx; ring
    obtain ⟨pt', hpt'eq, hlane⟩ : ∃ pt' : Tensor [128, BM] F32,
        (if MASK = true then
          tl.where (((start_n + tl.arange 0 128).col BM).le ((cm + tl.arange 0 BM).row 128)) pT
            (tl.full [128, BM] (FloatV.ofPat 0).litF32)
        else pT) = pt' ∧
        ∀ (i : Fin 128) (kk : Fin BM), 
          (Allowed d (n0 + i) (s0 + t * BM + kk) → pt'.get2 i kk = pT.get2 i kk) ∧
          (¬ Allowed d (n0 + i) (s0 + t * BM + kk) → pt'.get2 i kk = FloatV.ofPat 0) :=
      by
        refine ⟨_, rfl, fun i kk => ?_⟩
        have hnl : ∀ i' : Fin 128, ((start_n + tl.arange 0 128).get1 i').toInt = ((n0 + i' : Nat) : Int) :=
          fun i' => IntV.I32.toInt_sadd_arange (n := 128) hsn (by omega) i'
        have hml : ∀ k' : Fin BM, ((cm + tl.arange 0 BM).get1 k').toInt =
            ((s0 + t * BM + k' : Nat) : Int) := fun k' => IntV.I32.toInt_sadd_arange hcm hfit k'
        have hkk := kk.isLt
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
          rw [where_lane_N hnl hml]
          refine ⟨fun hal => ?_, fun hna => ?_⟩
          · have h1 : n0 + i ≤ s0 + t * BM + kk := by
              rcases hal with h | h
              · rw [hC] at h; cases h
              · exact h
            simp [h1]
          · have h1 : ¬ (n0 + i ≤ s0 + t * BM + kk) := fun h => hna (Or.inr h)
            simp [h1]
    rw [hpt'eq]
    refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
      fun ppT hppT => ?_
    refine safe_dot_add (hP.dotTotal _ _ _ _) fun dvn hdvn => ?_
    refine safe_load_vec d d.D hd.d_len (D_sub_in d) hpre.2.2.2.2.2.2.2.2 hb
      (r := fun i : Fin BM => s0 + t * BM + i) hrow ?_ fun Di hDi => ?_
    · intro i
      have hl := IntV.I32.toInt_sadd_arange (n := BM) hcm hfit i
      show Dp.offset ((cm + tl.arange 0 BM).get1 i).toInt = _
      rw [hl, hD, Buf.ptr_offset]; congr 1
    refine safe_dot (hP.dotTotal _ _ _ _) fun dp2 hdp2 => ?_
    refine safe_convert (fun i => ⟨dp2.get i, by simp [ConvRel, ofFloatRel, FloatRel]⟩) fun dpT hdpT => ?_
    refine safe_subc fun r2 hr2 => ?_
    refine safe_mul fun dsT0 hdsT0 => ?_
    refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
      fun dsT hdsT => ?_
    refine safe_dot_add (hP.dotTotal _ _ _ _) fun dkn hdkn => ?_
    have hBMHD : BM * d.HEAD_DIM ≤ 8192 := by
      rcases hBM with rfl | rfl <;> omega
    have hstep : (IntV.ofNat BM * stride_tok : I32).toInt = ((BM * d.HEAD_DIM : Nat) : Int) :=
      IntV.I32.mul_nat (IntV.I32.toInt_ofNat (by omega)) hs1 (by omega)
    refine Safe.ret ⟨rfl, ?_, ?_, ?_, ?_⟩
    · show (cm + IntV.ofNat BM : I32).toInt = _
      have := IntV.I32.add_nat hcm (IntV.I32.toInt_ofNat (n := BM) (by omega)) (by omega)
      rw [this]; congr 1; rw [hTB]; ring
    · intro c i
      show ((qp.get2 c i).offset (IntV.ofNat BM * stride_tok : I32).toInt) = _
      rw [hqp c i, hstep, Buf.ptr_offset]; congr 1; push_cast; ring
    · intro i c
      show ((dp.get2 i c).offset (IntV.ofNat BM * stride_tok : I32).toInt) = _
      rw [hdp i c, hstep, Buf.ptr_offset]; congr 1; push_cast; ring
    · intro hv' hH hOk
      have hgood := dkdv_good hP d hd hag hb hH hOk n0 (s0 + t * BM) hn0 (by omega) hk hv hqT hdO
        hmv hDi hqkT hr1 hpT hlane hppT hdp2 hdpT hr2 hdsT0 hdsT
      obtain ⟨hdv1, hdk1⟩ := hinv1 hv' hH hOk
      have hsN : ∀ s, (s ∈ S0 ∪ Finset.Ico s0 (s0 + t * BM) ∨
          (s0 + t * BM ≤ s ∧ s < s0 + t * BM + BM)) → s < d.N_CTX := by
        intro s hs
        rcases hs with hs | hs
        · rcases Finset.mem_union.mp hs with hs | hs
          · exact (hS0 s hs).1
          · have := (Finset.mem_Ico.mp hs).2; omega
        · omega
      have hdisj : ∀ s ∈ S0 ∪ Finset.Ico s0 (s0 + t * BM),
          s < s0 + t * BM ∨ s0 + t * BM + BM ≤ s := by
        intro s hs
        rcases Finset.mem_union.mp hs with hs | hs
        · rcases (hS0 s hs).2 with h | h
          · left; omega
          · right; omega
        · left; exact (Finset.mem_Ico.mp hs).2
      have hS : ∀ s ∈ S0 ∪ Finset.Ico s0 (s0 + t * BM), s < 2 ^ 14 :=
        fun s hs => lt_of_lt_of_le (hsN s (Or.inl hs)) hN
      have hset : (S0 ∪ Finset.Ico s0 (s0 + t * BM)) ∪ Finset.Ico (s0 + t * BM) (s0 + t * BM + BM) =
          S0 ∪ Finset.Ico s0 (s0 + (t + 1) * BM) := by
        rw [Finset.union_assoc, Finset.Ico_union_Ico_eq_Ico (by omega) (by omega), hTB]
        congr 2; ring
      have hz : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
          DType.val? .f32 ((Tensor.full (DType.f32.ofBits 0) : Tensor [128, d.HEAD_DIM] F32).get2 i j)
            = some 0 := fun _ _ => DType.val?_zero_f32
      have hrw : t0 + (t + 1) = t0 + t + 1 := by ring
      constructor
      · have := AccInv.step (fusion := E.fusion) hP.dotF16 (by omega) (famDV d hv')
          (fun i : Fin 128 => n0 + i) (fun j : Fin d.HEAD_DIM => j.val) hdisj hS (by omega) hdvn hz
          hdv1 (fun i k => (hgood i k).1)
          (fun k j => (do_eq hag hH hb (by omega) j.isLt (hdO k j)))
          (fun s hs j => hOk.dO s _ (hsN s hs) j.isLt)
          (fun i s hs => p16L_le hOk (by have := i.isLt; omega) (hsN s hs))
        rw [hset, ← hrw] at this
        exact this
      · have := AccInv.step (fusion := E.fusion) hP.dotF16 (by omega) (famDK d hv')
          (fun i : Fin 128 => n0 + i) (fun j : Fin d.HEAD_DIM => j.val) hdisj hS (by omega) hdkn hz
          hdk1 (fun i k => (hgood i k).2)
          (fun k j => (q_eq hag hH hb (by omega) j.isLt (hqT j k)))
          (fun s hs j => hOk.q s _ (hsN s hs) j.isLt)
          (fun i s hs => ds16L_le hOk (by have := i.isLt; omega) (hsN s hs))
        rw [hset, ← hrw] at this
        exact this
  · intro st m1 hI
    obtain ⟨dk1, dv1, om, cm, qp, dp⟩ := st
    obtain ⟨hm1, -, -, -, hinv1⟩ := hI
    refine Safe.ret ⟨hm1, ?_⟩
    rw [htrip] at hinv1
    exact hinv1

end Claims.AttnBwd
