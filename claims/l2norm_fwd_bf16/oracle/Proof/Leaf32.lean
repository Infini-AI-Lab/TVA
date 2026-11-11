import Proof.Defs
import Proof.Leaf16

/-!
# `l2norm_fwd_bf16`: the result of one program, `y` of `float32`

Same as `Leaf16.qn16` but the tile of `y` is stored as `float32` directly, so there is no second
conversion: the stored value is `b_y` itself (`y_err_f32`).
-/

set_option maxHeartbeats 2000000

namespace Claims.L2NormFwd

open TVA
open TVA TVA.Format


/-- **One program, `float32` `y`**: as `qn16`, with the stored value `b_y` itself. -/
theorem qn32 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {yc : Buf .f32}
    (hy : d.y = .inr yc) (p : Nat) (m : Mem)
    (P : Platform) (hsqrt_acc : Device.SqrtF32 P) (hdiv_full : Device.DivFull P)
    (b_x16 : Tensor [d.BT, d.BD] BF16) (b_x b_xx : Tensor [d.BT, d.BD] F32)
    (s1 s2 sq b_rstd : Tensor [d.BT] F32) (b_y : Tensor [d.BT, d.BD] F32)
    (hx16 : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      init.read .bf16 (d.x.ptr (gx d p j)).addr = some (b_x16.get j))
    (hz : ∀ j : Idx [d.BT, d.BD], enx d p j = false →
      b_x16.get j = FloatV.litBF16 (FloatV.ofPat 0))
    (hcvt : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.cvtWith .rn (b_x16.get j)) (b_x.get j))
    (hmul : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.mul (b_x.get j) (b_x.get j)) (b_xx.get j))
    (htree : ∀ r : Fin d.BT, ∃ t : Tree, t.Covers d.BD ∧
      t.Eval (AddRel ((launch d).env P).fusion)
        (fun j => if h : j < d.BD then b_xx.get2 r ⟨j, h⟩ else default) (s1.get1 r))
    (haddc : ∀ j : Idx [d.BT],
      AddRel ((launch d).env P).fusion (s1.get j) ((bc (FloatV.hostF32 d.eps)).get j) (s2.get j))
    (hsqrt : ∀ j : Idx [d.BT], P.math .sqrt .binary32 (s2.get j) (sq.get j))
    (hdiv : ∀ j : Idx [d.BT],
      P.div .binary32 ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))).get j) (sq.get j)
        (b_rstd.get j))
    (hmul2 : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.mul (b_x.get j) ((b_rstd.col d.BD).get j)) (b_y.get j))
    (hyread : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      m.read .f32 (yc.ptr (gx d p j)).addr = some (b_y.get j))
    (hrread : ∀ j : Idx [d.BT], enT d p j = true →
      m.read .f32 (d.rstd.ptr (gT d p j)).addr = some (b_rstd.get j)) :
    Qn d init p m := by
  intro t ht
  have htT : t < d.T := ht.1
  have hlo : p * d.BT ≤ t := ht.2.1
  have hhi : t < p * d.BT + d.BT := ht.2.2
  have hsub : t - p * d.BT < d.BT := by omega
  let i0 : Fin d.BT := ⟨t - p * d.BT, hsub⟩
  let a : Nat → BF16 := fun j =>
    if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else FloatV.litBF16 (FloatV.ofPat 0)
  have htm : t ∈ tileRow d p := ⟨htT, hlo, hhi⟩
  have hi0 : (i0 : Fin d.BT).val = t - p * d.BT := rfl
  -- ## the row's inputs
  have hreadA : ∀ j, j < d.D → init.read .bf16 (d.x.ptr (xIdx d t j)).addr = some (a j) := by
    intro j hj
    have hav : a j = b_x16.get2 i0 ⟨j, col_lt_bd hd hj⟩ := by
      show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
        FloatV.litBF16 (FloatV.ofPat 0)) = _
      rw [dite_eq_left_of_eq_true (eq_true hj)]
    rw [hav]
    have h := hx16 (i0, (⟨j, col_lt_bd hd hj⟩, ())) (lane_enx htm hj)
    rwa [lane_gx htm] at h
  have hAe := hpre.2.2.2.2 t htT a hreadA
  have hAeZ : rowSum a d.D + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by simpa using hAe
  have hA0 : (0 : ℝ) ≤ rowSum a d.D := Finset.sum_nonneg (fun j _ => sq_nonneg _)
  have he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal := by
    have h := Rat.cast_le (K := ℝ).mpr hd.eps_lo
    push_cast at h
    simpa only [FloatV.toReal] using h
  -- ## the per-column facts
  have hcol : ∀ j (hj : j < d.BD),
      (b_x16.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      (b_x.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      (b_x.get2 i0 ⟨j, hj⟩).toReal = (a j).toReal ∧
      |(b_x.get2 i0 ⟨j, hj⟩).toRat| ≤ (2 : ℚ) ^ (63 : ℤ) ∧
      (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
    intro j hj
    by_cases hjd : j < d.D
    · have hreadj := hx16 (i0, (⟨j, hj⟩, ())) (lane_enx htm hjd)
      rw [lane_gx htm] at hreadj
      obtain ⟨a', ha', hfin'⟩ := hpre.2.2.2.1 t htT j hjd
      have hb16eq : b_x16.get2 i0 ⟨j, hj⟩ = a' := by
        have hs : some (b_x16.get (i0, (⟨j, hj⟩, ()))) = some a' := by rw [← hreadj]; exact ha'
        exact Option.some.inj hs
      have hfin16 : (b_x16.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hb16eq]; exact hfin'
      have hcvtw := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
        Format.bfloat16_regular Format.binary32_regular wider_bfloat16_binary32 .rn hfin16
      obtain ⟨hcvtfin, hcvts⟩ := hcvtw
      have hbxeq : b_x.get2 i0 ⟨j, hj⟩ =
          (FloatV.cvtWith .rn (b_x16.get2 i0 ⟨j, hj⟩) : F32) :=
        (hcvt (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hcvtfin).1
      have hbxfin : (b_x.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hbxeq]; exact hcvtfin
      have hbxs : (b_x.get2 i0 ⟨j, hj⟩).toRat = (b_x16.get2 i0 ⟨j, hj⟩).toRat := by
        rw [hbxeq, hcvts]
      have haj : a j = b_x16.get2 i0 ⟨j, hj⟩ := by
        show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
          FloatV.litBF16 (FloatV.ofPat 0)) = _
        rw [dite_eq_left_of_eq_true (eq_true hjd)]
      have hbxR : (b_x.get2 i0 ⟨j, hj⟩).toReal = (a j).toReal := by
        simp only [FloatV.toReal]
        rw [hbxs, ← haj]
      have hbxRu : ((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ) = ((a j).toRat : ℝ) := by
        simpa only [FloatV.toReal] using hbxR
      have hsq_le : (a j).toReal ^ 2 ≤ (2 : ℝ) ^ (126 : ℤ) := by
        have hle : (a j).toReal ^ 2 ≤ rowSum a d.D :=
          Finset.single_le_sum (f := fun k => (a k).toReal ^ 2) (fun k _ => sq_nonneg _)
            (Finset.mem_range.mpr hjd)
        have he0' : (0 : ℝ) ≤ d.eps.toReal := le_trans (by norm_num [zpow_neg]) he
        linarith
      have hq63 : |(b_x.get2 i0 ⟨j, hj⟩).toRat| ≤ (2 : ℚ) ^ (63 : ℤ) := by
        refine (Rat.cast_le (K := ℝ)).mp ?_
        push_cast
        have h2 : (2 : ℝ) ^ (63 : ℤ) * (2 : ℝ) ^ (63 : ℤ) = (2 : ℝ) ^ (126 : ℤ) := by norm_num
        have hsqq : |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| *
            |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| = (a j).toReal ^ 2 := by
          rw [← abs_mul, abs_of_nonneg (mul_self_nonneg (((b_x.get2 i0 ⟨j, hj⟩).toRat : ℚ) : ℝ)),
            hbxRu]
          simp only [FloatV.toReal, pow_two]
        have h3 : |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| *
            |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)|
            ≤ (2 : ℝ) ^ (63 : ℤ) * (2 : ℝ) ^ (63 : ℤ) := by
          rw [hsqq, h2]; exact hsq_le
        have hc : (0 : ℝ) ≤ (2 : ℝ) ^ (63 : ℤ) := by norm_num
        nlinarith [h3, abs_nonneg ((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ), hc,
          sq_nonneg (|((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| - (2 : ℝ) ^ (63 : ℤ))]
      have hmaxq : |(b_x.get2 i0 ⟨j, hj⟩).toRat * (b_x.get2 i0 ⟨j, hj⟩).toRat|
          ≤ Format.binary32.maxVal := by
        have hqq : (b_x.get2 i0 ⟨j, hj⟩).toRat * (b_x.get2 i0 ⟨j, hj⟩).toRat
            = |(b_x.get2 i0 ⟨j, hj⟩).toRat| * |(b_x.get2 i0 ⟨j, hj⟩).toRat| := by
          rw [← abs_mul, abs_of_nonneg (mul_self_nonneg _)]
        rw [abs_of_nonneg (mul_self_nonneg _), hqq, Format.binary32_maxVal]
        have h := mul_le_mul hq63 hq63 (abs_nonneg _) (by norm_num : (0:ℚ) ≤ (2:ℚ)^(63:ℤ))
        refine h.trans ?_
        norm_num [zpow_neg]
      obtain ⟨hxxfin, hxxerr⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
        hbxfin hbxfin hmaxq
      have hxxeq : b_xx.get2 i0 ⟨j, hj⟩ =
          FloatV.mul (b_x.get2 i0 ⟨j, hj⟩) (b_x.get2 i0 ⟨j, hj⟩) :=
        (hmul (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hxxfin).1
      have hxxfin' : (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hxxeq]; exact hxxfin
      have hxxerrR : |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
          ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
        have h := Rat.cast_le (K := ℝ).mpr hxxerr
        rw [Format.binary32_u, Format.binary32_eta] at h
        push_cast at h
        simp only [FloatV.toReal] at h ⊢
        rw [← hxxeq, hbxRu] at h
        simpa [pow_two, mul_comm, mul_left_comm, mul_assoc] using h
      exact ⟨hfin16, hbxfin, hbxR, hq63, hxxfin', hxxerrR⟩
    · have henx : enx d p (i0, (⟨j, hj⟩, ())) = false := by
        have hrow : p * d.BT + (i0 : Fin d.BT).val < d.T := by rw [hi0]; omega
        simp only [enx]
        rw [decide_eq_true_eq.mpr hrow, decide_eq_false_iff_not.mpr hjd, Bool.and_false]
      have hz16 : b_x16.get2 i0 ⟨j, hj⟩ = FloatV.litBF16 (FloatV.ofPat 0) := hz _ henx
      have hfin16 : (b_x16.get2 i0 ⟨j, hj⟩).isFinite = true := by
        rw [hz16]; exact litBF16_zero_finite
      have haj : a j = FloatV.litBF16 (FloatV.ofPat 0) := by
        show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
          FloatV.litBF16 (FloatV.ofPat 0)) = _
        rw [dite_eq_right_of_eq_false (eq_false hjd)]
      have hzeroR : (a j).toReal = 0 := by rw [haj]; exact litBF16_zero_toReal
      have hzeroRu : ((a j).toRat : ℝ) = 0 := by simpa only [FloatV.toReal] using hzeroR
      have hcvtw := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
        Format.bfloat16_regular Format.binary32_regular wider_bfloat16_binary32 .rn hfin16
      obtain ⟨hcvtfin, hcvts⟩ := hcvtw
      have hbxeq : b_x.get2 i0 ⟨j, hj⟩ =
          (FloatV.cvtWith .rn (b_x16.get2 i0 ⟨j, hj⟩) : F32) :=
        (hcvt (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hcvtfin).1
      have hbxfin : (b_x.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hbxeq]; exact hcvtfin
      have hbxzero : (b_x.get2 i0 ⟨j, hj⟩).toRat = 0 := by
        rw [hbxeq, hcvts, hz16]; exact litBF16_zero_toRat
      have hbxR : (b_x.get2 i0 ⟨j, hj⟩).toReal = (a j).toReal := by
        simp only [FloatV.toReal]
        rw [hbxzero, hzeroRu]
        norm_num
      have hq63 : |(b_x.get2 i0 ⟨j, hj⟩).toRat| ≤ (2 : ℚ) ^ (63 : ℤ) := by
        rw [hbxzero, abs_zero]; norm_num
      have hmaxq : |(b_x.get2 i0 ⟨j, hj⟩).toRat * (b_x.get2 i0 ⟨j, hj⟩).toRat|
          ≤ Format.binary32.maxVal := by
        rw [hbxzero, mul_zero, abs_zero, Format.binary32_maxVal]; norm_num
      obtain ⟨hxxfin, hxxerr⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
        hbxfin hbxfin hmaxq
      have hxxeq : b_xx.get2 i0 ⟨j, hj⟩ =
          FloatV.mul (b_x.get2 i0 ⟨j, hj⟩) (b_x.get2 i0 ⟨j, hj⟩) :=
        (hmul (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hxxfin).1
      have hxxfin' : (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hxxeq]; exact hxxfin
      have hxxerrR : |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
          ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
        have h := Rat.cast_le (K := ℝ).mpr hxxerr
        rw [Format.binary32_u, Format.binary32_eta] at h
        push_cast at h
        have hbxz : ((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ) = 0 := by rw [hbxzero]; norm_num
        simp only [FloatV.toReal, hbxz, hzeroRu] at h ⊢
        rw [← hxxeq] at h
        simpa [mul_zero, mul_comm, mul_left_comm, mul_assoc, pow_two] using h
      exact ⟨hfin16, hbxfin, hbxR, hq63, hxxfin', hxxerrR⟩
  -- ## the row sum
  have hsum_eq : rowSum a d.BD = rowSum a d.D := by
    unfold rowSum
    refine (Finset.sum_subset (Finset.range_subset_range.mpr hd.d_bd) ?_).symm
    intro x _ hx
    have hxd : ¬ x < d.D := by
      intro h'; exact hx (Finset.mem_range.mpr h')
    have hax : a x = FloatV.litBF16 (FloatV.ofPat 0) := by
      show (if h : x < d.D then b_x16.get2 i0 ⟨x, col_lt_bd hd h⟩ else
        FloatV.litBF16 (FloatV.ofPat 0)) = _
      rw [dite_eq_right_of_eq_false (eq_false hxd)]
    rw [hax, litBF16_zero_toReal]; norm_num
  have hAeBD : rowSum a d.BD + d.eps.toReal ≤ 2 ^ 126 := by
    rw [hsum_eq]; exact hAe
  obtain ⟨hs1fin, hs1err⟩ := sum_err_bd (d := d) (fusion := ((launch d).env P).fusion)
    (a := a) (bxx := fun j => if h : j < d.BD then b_xx.get2 i0 ⟨j, h⟩ else default)
    (s1 := s1.get1 i0) hd
    (fun j hj => by rw [dite_eq_left_of_eq_true (eq_true hj)]; exact (hcol j hj).2.2.2.2.1)
    (fun j hj => by rw [dite_eq_left_of_eq_true (eq_true hj)]; exact (hcol j hj).2.2.2.2.2) hAeBD he (htree i0)
  have hs1 : |(s1.get1 i0).toReal - rowSum a d.D|
      ≤ (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
    have h := hs1err
    rw [hsum_eq] at h
    exact h
  -- ## the `eps` as `float32`
  have hehi : d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hd.eps_hi
    have h2 : (((2 : ℚ) ^ (126 : ℤ) : ℚ) : ℝ) = (2 : ℝ) ^ (126 : ℤ) := by
      push_cast
      norm_num
    simpa only [FloatV.toReal, h2] using h
  have he0 : (0 : ℝ) ≤ d.eps.toReal := le_trans (by norm_num) he
  have hepos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_lo
  obtain ⟨heps32_fin, heps32_err⟩ := hostF32_err hd.eps_fin hepos.le hd.eps_hi
  have heps32_errR : |(FloatV.hostF32 d.eps).toReal - d.eps.toReal|
      ≤ (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr heps32_err
    push_cast at h
    simpa only [FloatV.toReal] using h
  have he32_lo : (2 : ℝ) ^ (-125 : ℤ) ≤ (FloatV.hostF32 d.eps).toReal :=
    hostF32_ge hd.eps_fin hd.eps_lo hd.eps_hi
  have he32_nn : (0 : ℝ) ≤ (FloatV.hostF32 d.eps).toReal := le_trans (by norm_num) he32_lo
  have heps32_abs : |(((FloatV.hostF32 d.eps).toRat : ℚ) : ℝ)|
      = (((FloatV.hostF32 d.eps).toRat : ℚ) : ℝ) :=
    abs_of_nonneg (by simpa only [FloatV.toReal] using he32_nn)
  have hVhi : rowSum a d.D + (FloatV.hostF32 d.eps).toReal ≤ (2 : ℝ) ^ (127 : ℤ) := by
    have hup : (FloatV.hostF32 d.eps).toReal
        ≤ d.eps.toReal + ((2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ)) := by
      have hb := (abs_le.1 heps32_errR).2
      linarith
    have h1e : (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (126 : ℤ) :=
      mul_le_mul_of_nonneg_left hehi (by norm_num)
    have hkey : (2 : ℝ) ^ (126 : ℤ)
        + ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (126 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))
        ≤ (2 : ℝ) ^ (127 : ℤ) := by
      norm_num [zpow_neg]
    linarith [hAeZ, hup, hkey, h1e]
  -- ## the exact `s1` bound and the sum with `eps` as `float32`
  -- ## the sharp joint bound for `s1` and `eps` as `float32`
  have hs1' : |(s1.get1 i0).toReal|
      ≤ rowSum a d.D + ((2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ)) := by
    have h2 : |(s1.get1 i0).toReal|
        ≤ |(s1.get1 i0).toReal - rowSum a d.D| + |rowSum a d.D| := by
      calc |(s1.get1 i0).toReal|
          = |((s1.get1 i0).toReal - rowSum a d.D) + rowSum a d.D| := by congr 1; ring
        _ ≤ _ := abs_add_le _ _
    have hAa : |rowSum a d.D| = rowSum a d.D := abs_of_nonneg hA0
    linarith [hs1, h2, hAa]
  have he32' : |(FloatV.hostF32 d.eps).toReal|
      ≤ d.eps.toReal + ((2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ)) := by
    have hb := abs_le.1 heps32_errR
    rw [abs_le]
    constructor <;> linarith [hb.1, hb.2, he0]
  have hMR : |(s1.get1 i0).toReal| + |(FloatV.hostF32 d.eps).toReal|
      ≤ (2 : ℝ) ^ (126 : ℤ) + ((2 : ℝ) ^ (-13 : ℤ) * (2 : ℝ) ^ (126 : ℤ) + 2) := by
    have h24 : (2 : ℝ) ^ (-24 : ℤ) ≤ (2 : ℝ) ^ (-13 : ℤ) := by norm_num [zpow_neg]
    have h2e : (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal ≤ (2 : ℝ) ^ (-13 : ℤ) * d.eps.toReal :=
      mul_le_mul_of_nonneg_right h24 he0
    have hd : (2 : ℝ) ^ (-13 : ℤ) * (rowSum a d.D + d.eps.toReal)
        = (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (2 : ℝ) ^ (-13 : ℤ) * d.eps.toReal := by ring
    have h3 : (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal
        ≤ (2 : ℝ) ^ (-13 : ℤ) * (rowSum a d.D + d.eps.toReal) := by
      rw [hd]
      linarith [h2e]
    have h4 : (2 : ℝ) ^ (-13 : ℤ) * (rowSum a d.D + d.eps.toReal)
        ≤ (2 : ℝ) ^ (-13 : ℤ) * (2 : ℝ) ^ (126 : ℤ) :=
      mul_le_mul_of_nonneg_left hAeZ (by positivity)
    have h5 : (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal
        ≤ (2 : ℝ) ^ (-13 : ℤ) * (2 : ℝ) ^ (126 : ℤ) := h3.trans h4
    have hc1 : (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) ≤ 1 := by norm_num [zpow_neg]
    have hc2 : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 := by norm_num [zpow_neg]
    have hc : (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) ≤ 2 := by
      linarith [hc1, hc2]
    refine (add_le_add hs1' he32').trans ?_
    have e1 : (rowSum a d.D
          + ((2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ)))
        + (d.eps.toReal + ((2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ)))
        = (rowSum a d.D + d.eps.toReal)
          + ((2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal)
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := by ring
    rw [e1]
    exact (add_le_add (add_le_add hAeZ h5) hc).trans (by linarith)
  have hkeyM : (2 : ℝ) ^ (126 : ℤ) + ((2 : ℝ) ^ (-13 : ℤ) * (2 : ℝ) ^ (126 : ℤ) + 2)
      ≤ ((Format.binary32.maxVal : ℚ) : ℝ) / 2 - 2 * ((Format.binary32.eta : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal, Format.binary32_eta]
    push_cast
    norm_num [zpow_neg]
  have hMq : |(s1.get1 i0).toRat| + |(FloatV.hostF32 d.eps).toRat|
      ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h := hMR.trans hkeyM
    refine (Rat.cast_le (K := ℝ)).mp ?_
    have h2 : (((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ)) : ℝ)
        = ((Format.binary32.maxVal : ℚ) : ℝ) / 2 - 2 * ((Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      ring
    rw [h2]
    simpa only [Rat.cast_abs, Rat.cast_add, FloatV.toReal] using h
  obtain ⟨hs2fin, hs2err⟩ := FloatV.AddRel.err (f := Format.binary32) Format.binary32_regular
    (by rw [Format.Std.ieee Format.binary32_std]; decide) (by rw [Format.binary32_u]; norm_num [zpow_neg])
    (haddc (i0, ())) hs1fin
    heps32_fin hMq
  have hs2R : |(s2.get1 i0).toReal
        - ((s1.get1 i0).toReal + (FloatV.hostF32 d.eps).toReal)|
      ≤ 3 * (2 : ℝ) ^ (-24 : ℤ)
          * (|(s1.get1 i0).toReal| + (FloatV.hostF32 d.eps).toReal)
        + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hs2err
    rw [Format.binary32_u, Format.binary32_eta] at h
    push_cast at h
    have hbc : ((bc (FloatV.hostF32 d.eps) : Tensor [d.BT] F32).get (i0, ()))
        = FloatV.hostF32 d.eps := rfl
    simpa only [FloatV.toReal, Tensor.get1, Rat.cast_abs, hbc, heps32_abs] using h
  -- ## the row's `rstd` and `y` in `float32`
  have hrstd_tol : (rstdTol : ℝ) = (2 : ℝ) ^ (-12 : ℤ) := by rw [rstdTol]; norm_num
  have hAe_pos : (0 : ℝ) < rowSum a d.D + d.eps.toReal := by
    have h0 : (0 : ℝ) < (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    linarith [hA0, he]
  have hR0 : 0 < rstdRef a d.D d.eps.toReal := by
    simp only [rstdRef]
    exact inv_pos.2 (Real.sqrt_pos.2 hAe_pos)
  obtain ⟨hs2lo, hs2hi⟩ := s2_range (A := rowSum a d.D) (e := d.eps.toReal)
    (s1 := (s1.get1 i0).toReal) (e32 := (FloatV.hostF32 d.eps).toReal)
    (s2 := (s2.get1 i0).toReal) hA0 he0 he hAeZ he32_nn hs1 heps32_errR hs2R
  obtain ⟨sref, hsref, hacc⟩ := hsqrt_acc
  have hsref' : sref = Real.sqrt := by
    have h : MathFn.ref MathFn.sqrt = some Real.sqrt := rfl
    rw [h] at hsref
    exact (Option.some.inj hsref).symm
  have hs2dom : (s2.get1 i0).toReal ∈ Device.sqrtDom := Or.inr hs2lo
  have hs2nat : (s2.get1 i0).toReal ∈ MathFn.natDom MathFn.sqrt := by
    have h : MathFn.natDom MathFn.sqrt = Set.Ici (0 : ℝ) := rfl
    rw [h]
    exact le_trans (by norm_num) hs2lo
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |sref (s2.get1 i0).toReal|
      + (2 : ℝ) ^ (-149 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [hsref']
    exact sqrt_room (le_trans (by norm_num) hs2lo) hs2hi
  obtain ⟨hsqfin, hsqerr0⟩ :=
    hacc (s2.get1 i0) (sq.get1 i0) (hsqrt (i0, ())) hs2fin hs2dom hs2nat hroom
  have hsqerr : |(sq.get1 i0).toReal - Real.sqrt (s2.get1 i0).toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt (s2.get1 i0).toReal + (2 : ℝ) ^ (-149 : ℤ) := by
    have h := hsqerr0
    rw [hsref'] at h
    simpa only [abs_of_nonneg (Real.sqrt_nonneg (s2.get1 i0).toReal)] using h
  obtain ⟨hqlow, hqinv⟩ := q_low hs2lo hsqerr
  have hqposR : (0 : ℝ) < (sq.get1 i0).toReal := lt_of_lt_of_le (by norm_num) hqlow
  have hqpos : (0 : ℚ) < (sq.get1 i0).toRat := by
    refine (Rat.cast_lt (K := ℝ)).mp ?_
    simpa only [FloatV.toReal, Rat.cast_zero] using hqposR
  have hqne : (sq.get1 i0).toRat ≠ 0 := ne_of_gt hqpos
  have honeq : ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
      Tensor [d.BT] F32).get (i0, ())).toRat = 1 := by
    have h : (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
        Tensor [d.BT] F32).get (i0, ()) = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := rfl
    rw [h]
    exact litF32_one_toRat
  have honefin : ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
      Tensor [d.BT] F32).get (i0, ())).isFinite = true := by
    have h : (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
        Tensor [d.BT] F32).get (i0, ()) = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := rfl
    rw [h]
    exact litF32_one_finite
  have hqbound : |((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
      Tensor [d.BT] F32).get (i0, ())).toRat / (sq.get1 i0).toRat| ≤ 2 ^ 126 := by
    rw [honeq]
    have hR : |(1 : ℝ) / (sq.get1 i0).toReal| ≤ (2 : ℝ) ^ (126 : ℕ) := by
      rw [abs_of_pos (one_div_pos.mpr hqposR)]
      have h65 : (1 : ℝ) / (sq.get1 i0).toReal ≤ (2 : ℝ) ^ (65 : ℕ) := by
        rw [← zpow_natCast (2 : ℝ) 65]
        exact hqinv
      have h2 : (2 : ℝ) ^ (65 : ℕ) ≤ (2 : ℝ) ^ (126 : ℕ) := by norm_num
      linarith [h65, h2]
    refine rat_le_of_cast ?_
    push_cast
    simpa only [Rat.cast_abs, FloatV.toReal] using hR
  obtain ⟨hbrfin, hbrerr⟩ := hdiv_full
    ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) : Tensor [d.BT] F32).get (i0, ()))
    (sq.get1 i0) (b_rstd.get1 i0) (hdiv (i0, ())) honefin hsqfin hqne hqbound
  have hbrerr' : |(b_rstd.get1 i0).toRat - 1 / (sq.get1 i0).toRat|
      ≤ (2 : ℚ) ^ (-22 : ℤ) * |1 / (sq.get1 i0).toRat| + (2 : ℚ) ^ (-148 : ℤ) := by
    simpa only [honeq] using hbrerr
  have hbrerrR : |(b_rstd.get1 i0).toReal - (1 : ℝ) / (sq.get1 i0).toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * |(1 : ℝ) / (sq.get1 i0).toReal| + (2 : ℝ) ^ (-148 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hbrerr'
    push_cast at h
    simpa only [FloatV.toReal] using h
  have hwide := rstd_err_wide (A := rowSum a d.D) (e := (FloatV.hostF32 d.eps).toReal)
    (s1 := (s1.get1 i0).toReal) (s2 := (s2.get1 i0).toReal) (q := (sq.get1 i0).toReal)
    (r := (b_rstd.get1 i0).toReal) he32_nn hA0 he32_lo hVhi hs1 hs2R
    (lt_of_lt_of_le (by norm_num) hqlow) hsqerr hbrerrR
  have hfinal := eps_transfer (A := rowSum a d.D) (e := d.eps.toReal)
    (e32 := (FloatV.hostF32 d.eps).toReal) (r := (b_rstd.get1 i0).toReal)
    hA0 he heps32_errR hwide
  refine ⟨a, hreadA, ?_, ?_⟩
  · have hread := hrread (i0, ()) (lane_enT (j := 0) htm (h1 := hsub) (_h2 := bd_pos hd))
    rw [lane_gT (j := 0) htm (h1 := hsub) (_h2 := bd_pos hd)] at hread
    refine ⟨b_rstd.get1 i0, ?_, hbrfin, ?_⟩
    · show m.read .f32 (d.rstd.ptr t).addr = some (b_rstd.get (i0, ()))
      exact hread
    · rw [hrstd_tol]
      exact hfinal
  · intro j hj
    have hjbd : j < d.BD := col_lt_bd hd hj
    obtain ⟨hx16fin, hbxfin, hbxR, hq63, hxxfin, hxxerr⟩ := hcol j hjbd
    have hyv : (b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())) = b_rstd.get1 i0 := by
      simp only [Tensor.get_col, Tensor.get1]
    have hbrfin' : ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).isFinite = true := by
      rw [hyv]
      exact hbrfin
    have hx2 : (a j).toReal ^ 2 ≤ rowSum a d.D :=
      Finset.single_le_sum (f := fun k => (a k).toReal ^ 2) (fun k _ => sq_nonneg _)
        (Finset.mem_range.mpr hj)
    have hmaxy : |(b_x.get2 i0 ⟨j, hjbd⟩).toRat * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toRat|
        ≤ Format.binary32.maxVal := by
      refine rat_le_of_cast ?_
      rw [Rat.cast_abs]
      have hcast : (((b_x.get2 i0 ⟨j, hjbd⟩).toRat
          * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toRat : ℚ) : ℝ)
          = (b_x.get2 i0 ⟨j, hjbd⟩).toReal
            * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal := by
        push_cast
        rfl
      rw [hcast, hyv, hbxR]
      exact mul_abs_le hAe_pos he0 hx2 hfinal
    obtain ⟨hmulfin, hmulq⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
      hbxfin hbrfin' hmaxy
    have hbyeq : b_y.get2 i0 ⟨j, hjbd⟩ =
        FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩) ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))) :=
      (hmul2 (i0, (⟨j, hjbd⟩, ()))).eq (FloatV.isFinite_iff.mp hmulfin).1
    have hyfin : (b_y.get2 i0 ⟨j, hjbd⟩).isFinite = true := by
      rw [hbyeq]
      exact hmulfin
    have hmulqR : |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
        ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
        - (b_x.get2 i0 ⟨j, hjbd⟩).toReal
          * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal|
        ≤ (2 : ℝ) ^ (-24 : ℤ)
            * |(b_x.get2 i0 ⟨j, hjbd⟩).toReal
              * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal|
          + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := Rat.cast_le (K := ℝ).mpr hmulq
      rw [Format.binary32_u, Format.binary32_eta] at h
      push_cast at h
      simpa only [FloatV.toReal, mul_comm ((2 : ℝ) ^ (-24 : ℤ))] using h
    have hy_err : |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
        ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
        - (a j).toReal * (b_rstd.get1 i0).toReal|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * |(a j).toReal * (b_rstd.get1 i0).toReal|
          + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := hmulqR
      simp only [hyv, hbxR] at h
      exact h
    have hyfinal : |(b_y.get2 i0 ⟨j, hjbd⟩).toReal
        - (a j).toReal * rstdRef a d.D d.eps.toReal|
        ≤ (2 : ℝ) ^ (-11 : ℤ) * |(a j).toReal * rstdRef a d.D d.eps.toReal|
          + (2 : ℝ) ^ (-149 : ℤ) :=
      y_err_f32 (x := (a j).toReal) (R := rstdRef a d.D d.eps.toReal)
        (r := (b_rstd.get1 i0).toReal)
        (y := (FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
          ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal)
        (c := (b_y.get2 i0 ⟨j, hjbd⟩).toReal) hR0 hfinal hy_err (by rw [hbyeq])
    have hreadj := hyread (i0, (⟨j, hjbd⟩, ())) (lane_enx htm hj)
    rw [lane_gx htm] at hreadj
    unfold YOk
    rw [hy]
    refine ⟨b_y.get2 i0 ⟨j, hjbd⟩, ?_, hyfin, ?_⟩
    · show m.read .f32 (yc.ptr (xIdx d t j)).addr = some (b_y.get (i0, (⟨j, hjbd⟩, ())))
      exact hreadj
    · have htol : (yTol32 : ℝ) = (2 : ℝ) ^ (-11 : ℤ) := by rw [yTol32]; norm_num
      have hfloor : (yFloor32 : ℝ) = (2 : ℝ) ^ (-149 : ℤ) := by rw [yFloor32]; norm_num
      have hscale : (yTol32 : ℝ) * (|(a j).toReal| * rstdRef a d.D d.eps.toReal)
          = (yTol32 : ℝ) * |(a j).toReal * rstdRef a d.D d.eps.toReal| := by
        rw [abs_mul, abs_of_pos hR0]
      rw [hscale, htol, hfloor]
      exact hyfinal

end Claims.L2NormFwd
