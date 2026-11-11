import Proof.PrefTiles
import Proof.PrefGlue
import Proof.RowFacts
import Proof.Load
import Proof.Lanes

/-!
# One trip of the cached-part loop

`pref_trip_safe` walks `prefBody` for one trip: the tile skip, the loads of the slots, `k` and
`v`, the score `tl.dot`, the online-softmax update and the second `tl.dot`; the collected
relations (`PrefData`) carry the invariant (`pref_trip`).
-/

namespace Claims.ExtendAttn

open TVA

/-- **One cached-part trip** keeps the invariant and writes nothing. -/
theorem pref_trip_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (_hm : mOf d i * d.BM < extOf d init (bOf d i)) {q : Tensor [d.BM, d.BD] BF16}
    (hq : QFacts d init i q) {n : Nat} (hn : n < tripsPre d init (bOf d i)) {sn : I32}
    (hsn : sn.toInt = ((BN * n : Nat) : Int)) {st : LoopSt d} {m : Mem}
    (hI : LoopInvP d init i n st m) :
    Safe ((launch d).env P) (Rf d) ∅ (fun st' m' => LoopInvP d init i (n + 1) st' m') m
      (prefBody d s (maskM d s) q sn st) := by
  obtain ⟨hm0, hI2⟩ := hI
  have hm0' := hm0.symm
  subst hm0'
  have hIfull : LoopInvP d init i n st init := ⟨rfl, hI2⟩
  obtain ⟨acc, deno, e_max⟩ := st
  unfold prefBody
  simp only []
  by_cases hsk : PrefSkip d s sn
  · simp only [hsk.1, ite_true]
    refine safe_convB.2 ?_
    simp only [pure_bind]
    have hc : (Tensor.scalar ((Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 (Tensor.map b2i (finalMaskP d s (maskM d s) sn)))).eq (IntV.ofInt 0))).not.item = false := by
      show (!((Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 (Tensor.map b2i (finalMaskP d s (maskM d s) sn)))).eq (IntV.ofInt 0))) = false
      simpa using hsk.2
    simp only [hc, Bool.false_eq_true, ite_false]
    exact safe_ret (pref_skip_inv hd hpre hi hs hn hsn hsk hIfull)
  · by_cases hsw : 0 < d.sw
    on_goal 1 =>
      simp only [hsw, ite_true]
      refine safe_convB.2 ?_
      simp only [pure_bind]
      have hc : (Tensor.scalar ((Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 (Tensor.map b2i (finalMaskP d s (maskM d s) sn)))).eq (IntV.ofInt 0))).not.item = true := by
        have h := fun h2 => hsk ⟨hsw, h2⟩
        show (!((Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 (Tensor.map b2i (finalMaskP d s (maskM d s) sn)))).eq (IntV.ofInt 0))) = true
        simpa using h
      simp only [hc, ite_true]
    on_goal 2 =>
      simp only [hsw, ite_false, pure_bind]
      simp only [Tensor.not, Tensor.map, Tensor.item, Tensor.scalar, ite_true, Bool.not_false]

    all_goals
      refine safe_conv64.2 ?_
      have hb := (prog_range hi).1
      obtain ⟨-, -, hrs3, -⟩ := req_sizes hpre hb
      refine safe_load_en d.kvi (fun c : Idx [BN] => decide (BN * n + c.1.val < preOf d init (bOf d i)))
        (fun c : Idx [BN] => ksOf d init (bOf d i) + (BN * n + c.1.val)) ?_ ?_ ?_ hpre.2.2.2.2.2.2.2.2.1 ?_
        fun kvloc hk1 hk2 => ?_
      · rintro ⟨c, ⟨⟩⟩; exact maskNP_lane hd hpre hi hs hn hsn c
      · rintro ⟨c, ⟨⟩⟩ _; exact kvi_ptr_lane hs hsn c
      · rintro ⟨c, ⟨⟩⟩ hc
        have := hd.kvi_len
        simp only [decide_eq_true_eq] at hc
        dsimp only; omega
      · exact rf_kvi
      have hKV := kvloc_facts hpre hi hk1 hk2
      refine safe_conv64.2 ?_
      refine safe_conv64.2 ?_
      refine safe_conv64.2 ?_
      refine convert_zero_safe fun ok hok => ?_
      refine safe_load_en d.kb
        (fun x : Idx [d.BD, BN] => decide (BN * n + x.2.1.val < preOf d init (bOf d i) ∧ x.1.val < d.D))
        (fun x : Idx [d.BD, BN] => kbIdx d (slotOf d init (bOf d i) (BN * n + x.2.1.val)) (kvh d (hOf d i)) x.1.val)
        ?_ ?_ ?_ hpre.2.2.2.2.1 rf_kb fun k hkk1 hkk2 => ?_
      · rintro ⟨e, c, ⟨⟩⟩
        show ((maskNP d s sn).get1 c && ((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).get1 e) = _
        rw [maskNP_lane hd hpre hi hs hn hsn c, maskD_lane' hd e]
        simp
      · rintro ⟨e, c, ⟨⟩⟩ hen
        simp only [decide_eq_true_eq] at hen
        have hsl := hKV c
        simp only [hen.1, ite_true] at hsl
        obtain ⟨sl, -, -, hlt, hel⟩ := slot_read hpre hb hen.1
        have hS : slotOf d init (bOf d i) (BN * n + c.val) < d.S := by omega
        refine ptr0_add d.kb _ _ ?_
        have := gather_lane_toInt hd hi hs (kv := kvloc.get1 c) hsl hS (st := d.sbkt) (sh := d.sbkh)
          (len := d.kb.len) hd.strides_lt.2.2.2.2.2.2.2.2.1 hd.strides_lt.2.2.2.2.2.2.2.2.2.1
          hd.heads_i32.2.2.2.2.1 (hd.kb_len (by omega)) (buf_len_le hpre.2.2.2.2.1)
          (ev := (tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e) hen.2
          (IntV.I32.toInt_tl_arange (n := d.BD) (by have := hd.bd_le; omega) e)
        exact this
      · rintro ⟨e, c, ⟨⟩⟩ hen
        simp only [decide_eq_true_eq] at hen
        have hsl := hKV c
        simp only [hen.1, ite_true] at hsl
        obtain ⟨sl, -, -, hlt, hel⟩ := slot_read hpre hb hen.1
        have hS : slotOf d init (bOf d i) (BN * n + c.val) < d.S := by omega
        exact gather_idx_lt hS (kvh_lt' hd (prog_range hi).2.1) hen.2 (hd.kb_len (by omega))
      have hKF := kFacts_of_load hd hpre hi hkk1 hkk2 hok
      refine safe_cvt ⟨Format.bfloat16_std, Format.bfloat16_std, Or.inl rfl⟩ fun q16 hq16 => ?_
      have hQ16 := qFacts_cvt hq hq16
      refine safe_dot (hP.dotTotal _ _ _ _) fun qk0 hqk0 => ?_
      refine safe_mul fun ss hss => ?_
      refine safe_mul fun qk1 hqk1 => ?_
      refine safe_max1 (by decide) true fun row_max hrm => ?_
      refine safe_maximum fun n_e_max hnm => ?_
      refine safe_sub fun x1 hx1 => ?_
      refine safe_math (fun _ => hP.mathTotal _ _ _) fun re hre => ?_
      refine safe_sub fun x2 hx2 => ?_
      refine safe_math (fun _ => hP.mathTotal _ _ _) fun p hp => ?_
      refine safe_sum1 (by decide) fun sm hsm => ?_
      refine safe_mul_add fun deno' hden => ?_
      refine safe_conv64.2 ?_
      refine safe_conv64.2 ?_
      refine safe_conv64.2 ?_
      refine convert_zero_safe fun okv hokv => ?_
      refine safe_load_en d.vb
        (fun x : Idx [BN, d.BD] => decide (BN * n + x.1.val < preOf d init (bOf d i) ∧ x.2.1.val < d.D))
        (fun x : Idx [BN, d.BD] => vbIdx d (slotOf d init (bOf d i) (BN * n + x.1.val)) (kvh d (hOf d i)) x.2.1.val)
        ?_ ?_ ?_ hpre.2.2.2.2.2.1 rf_vb fun v hvv1 hvv2 => ?_
      · rintro ⟨c, e, ⟨⟩⟩
        show ((maskNP d s sn).get1 c && ((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).get1 e) = _
        rw [maskNP_lane hd hpre hi hs hn hsn c, maskD_lane' hd e]
        simp
      · rintro ⟨c, e, ⟨⟩⟩ hen
        simp only [decide_eq_true_eq] at hen
        have hsl := hKV c
        simp only [hen.1, ite_true] at hsl
        obtain ⟨sl, -, -, hlt, hel⟩ := slot_read hpre hb hen.1
        have hS : slotOf d init (bOf d i) (BN * n + c.val) < d.S := by omega
        refine ptr0_add d.vb _ _ ?_
        have := gather_lane_toInt hd hi hs (kv := kvloc.get1 c) hsl hS (st := d.sbvt) (sh := d.sbvh)
          (len := d.vb.len) hd.strides_lt.2.2.2.2.2.2.2.2.2.2.1 hd.strides_lt.2.2.2.2.2.2.2.2.2.2.2
          hd.heads_i32.2.2.2.2.2 (hd.vb_len (by omega)) (buf_len_le hpre.2.2.2.2.2.1)
          (ev := (tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e) hen.2
          (IntV.I32.toInt_tl_arange (n := d.BD) (by have := hd.bd_le; omega) e)
        exact this
      · rintro ⟨c, e, ⟨⟩⟩ hen
        simp only [decide_eq_true_eq] at hen
        have hsl := hKV c
        simp only [hen.1, ite_true] at hsl
        obtain ⟨sl, -, -, hlt, hel⟩ := slot_read hpre hb hen.1
        have hS : slotOf d init (bOf d i) (BN * n + c.val) < d.S := by omega
        exact gather_idx_lt hS (kvh_lt' hd (prog_range hi).2.1) hen.2 (hd.vb_len (by omega))
      have hVF := vFacts_of_load hd hpre hi hvv1 hvv2 hokv
      refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun p16 hp16 => ?_
      refine safe_dot (hP.dotTotal _ _ _ _) fun dotv hdotv => ?_
      refine safe_mul fun a1 ha1 => ?_
      refine safe_mul fun a2 ha2 => ?_
      refine safe_addc fun acc' hadd => ?_
      let R : PrefData P true d.BM d.BD (FloatV.hostF32 d.sm_scale) :=
        { q16 := q16, k := k, v := v, FM := finalMaskP d s (maskM d s) sn, acc := acc, deno := deno,
          e_max := e_max, ss := ss.item, qk0 := qk0, qk1 := qk1, row_max := row_max,
          n_e_max := n_e_max, x1 := x1, re := re, x2 := x2, p := p, sm := sm, deno' := deno',
          p16 := p16, dotv := dotv, a1 := a1, a2 := a2, acc' := acc',
          hss := hss (), hdot1 := hqk0, hmul1 := hqk1, hmax := hrm, hnm := hnm, hsub1 := hx1,
          hexp1 := hre, hsub2 := hx2, hexp2 := hp, hsum := hsm, hden := hden, hcvt := hp16,
          hdot2 := hdotv, hmul2 := ha1, hmul3 := ha2, hadd := hadd }
      have hFM : ∀ (r : Fin d.BM) (c : Fin BN),
          R.FM.get2 r c = (decide (validRow d init i r) && vP d init i r n c) :=
        fun r c => finalMaskP_lane hd hpre hi hs hn hsn r c
      have key := pref_trip hP hd hpre hi hn rfl R hQ16 hKF hVF hFM hI2
      exact safe_ret ⟨rfl, key⟩

end Claims.ExtendAttn
