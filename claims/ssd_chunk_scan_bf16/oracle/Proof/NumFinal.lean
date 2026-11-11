import Proof.Zero

/-!
# The epilogue: numeric facts

The `D·x` term, the store's rounding and the relaxation of the derivation's bound to `tol`
(README, Tolerances steps 5, 6 and 8).
-/

namespace Claims.SsdChunkScan

open TVA

/-! ## The real inequality of step 8 -/

set_option maxHeartbeats 1600000 in
/-- **README step 8** over the reals (`fine ≤ tol`), the atoms being the claim's quantities. -/
theorem numfinal_fine_le {Y th ls ms cs es lw mw Pw Sw K xD rel ab rel32 a32
    eSig aSig errSig errW errDot aAcc errMA : ℝ}
    (hY : 0 ≤ Y) (hth : 0 ≤ th) (hls : 0 ≤ ls) (hms : 0 ≤ ms) (hcs : 0 ≤ cs) (hes : 0 ≤ es)
    (hlw : 0 ≤ lw) (hPw : 0 ≤ Pw) (hK0 : 0 ≤ K) (hK : K ≤ 512)
    (hrel : rel ≤ 130 * 2 ^ (-22 : ℤ)) (hab0 : 0 ≤ ab) (hab : ab ≤ 2 ^ (-118 : ℤ))
    (hr32 : rel32 = 34 * 2 ^ (-22 : ℤ)) (ha32 : a32 = 34 * 2 ^ (-126 : ℤ))
    (hlift : (1 + 2 ^ (-8 : ℤ)) * ms + cs * (2 ^ (-134 : ℤ) + 2 ^ (-126 : ℤ)) ≤ ls)
    (hmw : mw ≤ lw) (hSw : Sw ≤ lw) (hxD : xD ≤ Y + th * ls + lw)
    (hxD0 : 0 ≤ xD)
    (hE : eSig = 2 ^ (-8 : ℤ) * ms + 2 ^ (-134 : ℤ) * cs + rel * ls + ab)
    (hA : aSig = (1 + 2 ^ (-24 : ℤ)) * (ms + eSig) * th + 2 ^ (-150 : ℤ))
    (hSg : errSig = (1 + 2 ^ (-24 : ℤ)) * eSig * th + 2 ^ (-24 : ℤ) * ms * th + ms * es +
      2 ^ (-150 : ℤ))
    (hW : errW = Pw + (2 ^ (-8 : ℤ) + 3 * 2 ^ (-24 : ℤ)) * Sw)
    (hD : errDot = 2 * K * ((rel32 + 2 ^ (-21 : ℤ)) * (aSig + lw) + a32 + 2 ^ (-140 : ℤ)))
    (hAc : aAcc = aSig + mw + errDot)
    (hM : errMA = (2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ)) * (xD + aAcc) + 3 * 2 ^ (-150 : ℤ)) :
    (1 + 2 ^ (-8 : ℤ)) * (errSig + errW + errDot + errMA) + 2 ^ (-8 : ℤ) * Y + 2 ^ (-134 : ℤ) ≤
      (K + 256) * 2 ^ (-15 : ℤ) * (Y + th * ls + lw) + 2 * (ms * es + Pw) +
        2 ^ (-110 : ℤ) * (1 + th) := by
  -- the tiny constants, as atoms with their sizes
  have c110 : (2 : ℝ) ^ (-110 : ℤ) = 256 * 2 ^ (-118 : ℤ) := by norm_num
  have hF : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-118 : ℤ) / 1000 := by norm_num
  have hB : (2 : ℝ) ^ (-134 : ℤ) ≤ 2 ^ (-118 : ℤ) / 1000 := by norm_num
  have hZ : a32 + 2 ^ (-140 : ℤ) ≤ 3 * 2 ^ (-118 : ℤ) / 16 := by rw [ha32]; norm_num
  have hF0 : (0 : ℝ) ≤ 2 ^ (-150 : ℤ) := by positivity
  have hB0 : (0 : ℝ) ≤ 2 ^ (-134 : ℤ) := by positivity
  have hZ0 : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) := by positivity
  have ha320 : 0 ≤ a32 := by rw [ha32]; positivity
  have hmB : (2 : ℝ) ^ (-134 : ℤ) = 2 ^ (-8 : ℤ) * 2 ^ (-126 : ℤ) := by norm_num
  have hmB0 : (0 : ℝ) ≤ 2 ^ (-126 : ℤ) := by positivity
  have hrel32 : rel32 + 2 ^ (-21 : ℤ) = 36 * 2 ^ (-22 : ℤ) := by rw [hr32]; norm_num
  rw [hrel32] at hD
  have n8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have n24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have n22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have n23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
  have n47 : (2 : ℝ) ^ (-47 : ℤ) = 1 / 140737488355328 := by norm_num
  have n15 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
  generalize (2 : ℝ) ^ (-150 : ℤ) = eF at *
  generalize (2 : ℝ) ^ (-140 : ℤ) = z at *
  generalize (2 : ℝ) ^ (-126 : ℤ) = mB at *
  generalize (2 : ℝ) ^ (-134 : ℤ) = eB at *
  generalize (2 : ℝ) ^ (-118 : ℤ) = δ at *
  simp only [n8, n24, n22, n23, n47, n15] at *
  subst hmB
  have hδ0 : 0 ≤ δ := by linarith
  -- lifts
  have hmsls : ms ≤ ls := by linarith [mul_nonneg hcs hmB0]
  have s1 : eSig ≤ (1 / 256 + 130 / 4194304) * ls + ab := by
    have h1 := mul_le_mul_of_nonneg_right hrel hls
    linarith [mul_nonneg hcs hmB0]
  have s2 : ms + eSig ≤ (1 + 130 / 4194304) * ls + ab := by
    have h1 := mul_le_mul_of_nonneg_right hrel hls
    linarith [mul_nonneg hcs hmB0]
  have e1 := mul_le_mul_of_nonneg_right s1 hth
  have e2 := mul_le_mul_of_nonneg_right hmsls hth
  have e3 := mul_le_mul_of_nonneg_right s2 hth
  have e4 : 0 ≤ ab * th := mul_nonneg hab0 hth
  have e5 : ab * th ≤ δ * th := mul_le_mul_of_nonneg_right hab hth
  have hA0 : 0 ≤ th * ls := mul_nonneg hth hls
  -- the state term
  have s3 : errSig ≤ 394 / 100000 * (th * ls) + (1 + 1 / 16777216) * (ab * th) + ms * es + eF := by
    rw [hSg]; linarith
  have s4 : aSig ≤ 100004 / 100000 * (th * ls) + (1 + 1 / 16777216) * (ab * th) + eF := by
    rw [hA]; linarith
  -- the loop
  have hT0 : 0 ≤ (1 + 1 / 16777216) * (ab * th) + eF := by positivity
  have s5 : aSig + lw ≤ 100004 / 100000 * (th * ls) + lw + ((1 + 1 / 16777216) * (ab * th) + eF) := by
    linarith
  have k1 := mul_le_mul_of_nonneg_left s5 hK0
  have k2 := mul_le_mul_of_nonneg_left hZ hK0
  have k3 := mul_le_mul_of_nonneg_right hK hδ0
  have k4 := mul_le_mul_of_nonneg_right hK hT0
  have k5 := mul_le_mul_of_nonneg_right hK hA0
  have k6 := mul_le_mul_of_nonneg_right hK hlw
  have kA : 0 ≤ K * (th * ls) := mul_nonneg hK0 hA0
  have kL : 0 ≤ K * lw := mul_nonneg hK0 hlw
  have kY : 0 ≤ K * Y := mul_nonneg hK0 hY
  have d1 : errDot ≤ 72 / 4194304 * (100004 / 100000 * (K * (th * ls)) + K * lw + 512 * ((1 + 1 / 16777216) * (ab * th) + eF)) + 192 * δ := by
    rw [hD]; nlinarith
  have d2 : errDot ≤ 72 / 4194304 * (100004 / 100000 * (512 * (th * ls)) + 512 * lw + 512 * ((1 + 1 / 16777216) * (ab * th) + eF)) + 192 * δ := by
    linarith
  have a1 : aAcc ≤ 1009 / 1000 * (th * ls) + 1009 / 1000 * lw + 1009 / 1000 * ((1 + 1 / 16777216) * (ab * th) + eF) + 192 * δ := by
    rw [hAc]; linarith
  have m1 := mul_le_mul_of_nonneg_left (add_le_add hxD a1) (by norm_num : (0:ℝ) ≤ 1 / 8388608 + 1 / 140737488355328)
  have hbr : 0 ≤ Y + 2009 / 1000 * (th * ls) + 2009 / 1000 * lw + 1009 / 1000 * ((1 + 1 / 16777216) * (ab * th) + eF) + 192 * δ := by positivity
  have m2 : errMA ≤ 12 / 100000000 * (Y + 2009 / 1000 * (th * ls) + 2009 / 1000 * lw + 1009 / 1000 * ((1 + 1 / 16777216) * (ab * th) + eF) + 192 * δ) + 3 * eF := by
    have := mul_nonneg (by norm_num : (0:ℝ) ≤ 12 / 100000000 - (1 / 8388608 + 1 / 140737488355328)) hbr
    rw [hM]; linarith
  rw [c110]
  rw [hW]
  linarith [mul_nonneg hes hms]

/-! ## The claim's quantities -/

theorem numfinal_errE_nonneg {x δ : ℝ} (hδ : 0 ≤ δ) : 0 ≤ errE x δ := by
  unfold errE expRel
  have h1 : Real.exp x ≤ Real.exp (x + δ) := Real.exp_le_exp.mpr (by linarith)
  have h2 : 0 ≤ (2 : ℝ) ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|x| + δ) := by positivity
  have h3 := mul_nonneg h2 (Real.exp_pos (x + δ)).le
  have h4 : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) := by positivity
  nlinarith

/-- `eHat` bounds the exponential it estimates. -/
theorem numfinal_exp_le_eHat {x δ : ℝ} (hδ : 0 ≤ δ) : Real.exp x ≤ eHat x δ := by
  unfold eHat; linarith [numfinal_errE_nonneg (x := x) hδ]

/-- `eHat` is nonnegative. -/
theorem numfinal_eHat_nonneg {x δ : ℝ} (hδ : 0 ≤ δ) : 0 ≤ eHat x δ :=
  (Real.exp_pos x).le.trans (numfinal_exp_le_eHat hδ)

/-- The entering state's exponent error is nonnegative. -/
theorem numfinal_dErrSt_nonneg (m₀ : Mem) (d : Inst) (t h : Nat) : 0 ≤ dErrSt m₀ d t h := by
  unfold dErrSt uF etaF
  split_ifs
  · positivity
  · exact le_rfl

/-- A weight's exponent error is nonnegative. -/
theorem numfinal_dErrS_nonneg (m₀ : Mem) (d : Inst) (t h s : Nat) : 0 ≤ dErrS m₀ d t h s := by
  unfold dErrS uF etaF; positivity

/-- `hSt` is nonnegative. -/
theorem numfinal_hSt_nonneg (m₀ : Mem) (d : Inst) (t h : Nat) : 0 ≤ hSt m₀ d t h :=
  numfinal_eHat_nonneg (numfinal_dErrSt_nonneg m₀ d t h)

/-- `eSt` is nonnegative. -/
theorem numfinal_eSt_nonneg (m₀ : Mem) (d : Inst) (t h : Nat) : 0 ≤ eSt m₀ d t h :=
  numfinal_errE_nonneg (numfinal_dErrSt_nonneg m₀ d t h)

/-- `hSt` bounds the state's exact decay. -/
theorem numfinal_exp_dSt_le (m₀ : Mem) (d : Inst) (t h : Nat) :
    Real.exp (dSt m₀ d t h) ≤ hSt m₀ d t h :=
  numfinal_exp_le_eHat (numfinal_dErrSt_nonneg m₀ d t h)

/-- `wMag` is nonnegative. -/
theorem numfinal_wMag_nonneg (m₀ : Mem) (d : Inst) (t h s : Nat) : 0 ≤ wMag m₀ d t h s :=
  mul_nonneg (mul_nonneg (abs_nonneg _) (numfinal_eHat_nonneg (numfinal_dErrS_nonneg m₀ d t h s)))
    (abs_nonneg _)

/-- `wAbs` is nonnegative. -/
theorem numfinal_wAbs_nonneg (m₀ : Mem) (d : Inst) (t h s : Nat) : 0 ≤ wAbs m₀ d t h s := by
  unfold wAbs; positivity

/-- `wHat` bounds `wMag`. -/
theorem numfinal_wMag_le_wHat (m₀ : Mem) (d : Inst) (t h s : Nat) :
    wMag m₀ d t h s ≤ wHat m₀ d t h s := by
  unfold wHat uB uF
  have := numfinal_wMag_nonneg m₀ d t h s
  have := numfinal_wAbs_nonneg m₀ d t h s
  have h1 : (0 : ℝ) ≤ 2 ^ (-8 : ℤ) := by positivity
  have h2 : (0 : ℝ) ≤ 2 ^ (-24 : ℤ) := by positivity
  nlinarith

/-- `wMag` bounds the exact weight. -/
theorem numfinal_wTS_le_wMag (m₀ : Mem) (d : Inst) (t h s : Nat) :
    |wTS m₀ d t h s| ≤ wMag m₀ d t h s := by
  unfold wTS wMag
  rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _)]
  have := numfinal_exp_le_eHat (x := dS m₀ d t h s) (numfinal_dErrS_nonneg m₀ d t h s)
  have h1 := mul_le_mul_of_nonneg_left this (abs_nonneg (cbTS m₀ d t h s))
  have h2 := mul_le_mul_of_nonneg_right h1 (abs_nonneg (dtS m₀ d t h s))
  linarith

/-- `wHat` is nonnegative. -/
theorem numfinal_wHat_nonneg (m₀ : Mem) (d : Inst) (t h s : Nat) : 0 ≤ wHat m₀ d t h s :=
  (numfinal_wMag_nonneg m₀ d t h s).trans (numfinal_wMag_le_wHat m₀ d t h s)

/-- `mW ≤ lW`. -/
theorem numfinal_mW_le_lW (m₀ : Mem) (d : Inst) (t h p : Nat) :
    mW m₀ d t h p ≤ lW m₀ d t h p := by
  unfold mW lW
  refine Finset.sum_le_sum fun s _ => ?_
  have := numfinal_wHat_nonneg m₀ d t h s
  have : 0 ≤ mB := by unfold mB; positivity
  nlinarith [abs_nonneg (xv m₀ d s h p)]

/-- `Σ wMag·|x| ≤ lW`. -/
theorem numfinal_wMagx_le_lW (m₀ : Mem) (d : Inst) (t h p : Nat) :
    ∑ s ∈ toks d t, wMag m₀ d t h s * |xv m₀ d s h p| ≤ lW m₀ d t h p := by
  unfold lW
  refine Finset.sum_le_sum fun s _ => ?_
  have := numfinal_wMag_le_wHat m₀ d t h s
  have := numfinal_wMag_nonneg m₀ d t h s
  have : 0 ≤ mB := by unfold mB; positivity
  nlinarith [abs_nonneg (xv m₀ d s h p)]

/-- `lW` is nonnegative. -/
theorem numfinal_lW_nonneg (m₀ : Mem) (d : Inst) (t h p : Nat) : 0 ≤ lW m₀ d t h p := by
  unfold lW
  refine Finset.sum_nonneg fun s _ => mul_nonneg ?_ ?_
  · have : 0 ≤ mB := by unfold mB; positivity
    linarith [numfinal_wHat_nonneg m₀ d t h s]
  · have : 0 ≤ mB := by unfold mB; positivity
    linarith [abs_nonneg (xv m₀ d s h p)]

/-- **The lift**: `lSig ≥ (1 + u)·mSig + cSum·(2^-134 + 2^-126)`. -/
theorem numfinal_lift (m₀ : Mem) (d : Inst) (t h p : Nat) :
    (1 + uB) * mSig m₀ d t h p + cSum m₀ d t h * (etaB + mB) ≤ lSig m₀ d t h p := by
  unfold mSig cSum lSig
  rw [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun n _ => ?_
  have h1 : 0 ≤ mB := by unfold mB; positivity
  have h2 : 0 ≤ etaB := by unfold etaB; positivity
  have h3 : 0 ≤ uB := by unfold uB; positivity
  have := abs_nonneg (Cv m₀ d t (grp d h) n)
  have := abs_nonneg (Sv m₀ d t h p n)
  nlinarith [mul_nonneg h1 (abs_nonneg (Sv m₀ d t h p n)), mul_nonneg this h1,
    mul_nonneg this h2, mul_nonneg h2 h1]

/-- `mSig` is nonnegative. -/
theorem numfinal_mSig_nonneg (m₀ : Mem) (d : Inst) (t h p : Nat) : 0 ≤ mSig m₀ d t h p :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (abs_nonneg _) (abs_nonneg _)

/-- `cSum` is nonnegative. -/
theorem numfinal_cSum_nonneg (m₀ : Mem) (d : Inst) (t h : Nat) : 0 ≤ cSum m₀ d t h :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- `mSig ≤ lSig`. -/
theorem numfinal_mSig_le_lSig (m₀ : Mem) (d : Inst) (t h p : Nat) :
    mSig m₀ d t h p ≤ lSig m₀ d t h p := by
  have := numfinal_lift m₀ d t h p
  have h1 : 0 ≤ cSum m₀ d t h * (etaB + mB) :=
    mul_nonneg (numfinal_cSum_nonneg m₀ d t h) (by unfold etaB mB; positivity)
  have h2 := numfinal_mSig_nonneg m₀ d t h p
  have h3 : 0 ≤ uB := by unfold uB; positivity
  nlinarith

/-- `lSig` is nonnegative. -/
theorem numfinal_lSig_nonneg (m₀ : Mem) (d : Inst) (t h p : Nat) : 0 ≤ lSig m₀ d t h p :=
  (numfinal_mSig_nonneg m₀ d t h p).trans (numfinal_mSig_le_lSig m₀ d t h p)

/-- The `D·x` term is bounded by `|yRef|` and the two lifted magnitudes. -/
theorem numfinal_xD_le (m₀ : Mem) (d : Inst) (t h p : Nat) :
    |xv m₀ d t h p * Dv m₀ d h| ≤ |yRef m₀ d t h p| + hSt m₀ d t h * lSig m₀ d t h p +
      lW m₀ d t h p := by
  have hy : xv m₀ d t h p * Dv m₀ d h = yRef m₀ d t h p -
      (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p) := by
    unfold yRef; ring
  have h1 : |sig m₀ d t h p| ≤ mSig m₀ d t h p := by
    unfold sig mSig
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun n _ => ?_)
    rw [abs_mul]
  have h2 : |Real.exp (dSt m₀ d t h) * sig m₀ d t h p| ≤ hSt m₀ d t h * lSig m₀ d t h p := by
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul (numfinal_exp_dSt_le m₀ d t h) (h1.trans (numfinal_mSig_le_lSig m₀ d t h p))
      (abs_nonneg _) (numfinal_hSt_nonneg m₀ d t h)
  have h3 : |∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p| ≤ lW m₀ d t h p := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun s _ => ?_).trans
      (numfinal_mW_le_lW m₀ d t h p))
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right
      ((numfinal_wTS_le_wMag m₀ d t h s).trans (numfinal_wMag_le_wHat m₀ d t h s)) (abs_nonneg _)
  rw [hy]
  calc _ ≤ |yRef m₀ d t h p| + |Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p| := abs_sub _ _
    _ ≤ _ := by
      have := abs_add_le (Real.exp (dSt m₀ d t h) * sig m₀ d t h p)
        (∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p)
      linarith


/-- `relD n = (n + 2)·2^-22`. -/
theorem numfinal_relD (n : Nat) : relD n = ((n : ℝ) + 2) * 2 ^ (-22 : ℤ) := by
  unfold relD Device.relDot16; push_cast; ring

/-- `absD n = (n + 2)·2^-126`. -/
theorem numfinal_absD (n : Nat) : absD n = ((n : ℝ) + 2) * 2 ^ (-126 : ℤ) := by
  unfold absD Device.absDot16; push_cast; ring

/-- **README step 8: `fine ≤ tol`.** -/
theorem numfinal_fine_le_tol {d : Inst} (hd : d.ok) (m₀ : Mem) (t h p : Nat) :
    fine m₀ d t h p ≤ tol m₀ d t h p := by
  have hK : (nIt d : ℝ) ≤ 512 := by
    have := hd.cs_le
    unfold nIt BK
    norm_num at this ⊢
    exact_mod_cast (by omega : (d.cs + 32 - 1) / 32 ≤ 512)
  have hB : (d.BSD : ℝ) ≤ 128 := by exact_mod_cast hd.bsd_le
  have hB0 : (0 : ℝ) ≤ d.BSD := Nat.cast_nonneg _
  have hrel : relD d.BSD ≤ 130 * 2 ^ (-22 : ℤ) := by
    rw [numfinal_relD]
    have : (0 : ℝ) < 2 ^ (-22 : ℤ) := by positivity
    nlinarith
  have hab0 : 0 ≤ absD d.BSD := by rw [numfinal_absD]; positivity
  have hab : absD d.BSD ≤ 2 ^ (-118 : ℤ) := by
    rw [numfinal_absD]
    have h1 : ((d.BSD : ℝ) + 2) * 2 ^ (-126 : ℤ) ≤ 130 * 2 ^ (-126 : ℤ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    have h2 : (130 : ℝ) * 2 ^ (-126 : ℤ) ≤ 2 ^ (-118 : ℤ) := by norm_num
    linarith
  have key := numfinal_fine_le (Y := |yRef m₀ d t h p|) (th := hSt m₀ d t h)
    (ls := lSig m₀ d t h p) (ms := mSig m₀ d t h p) (cs := cSum m₀ d t h) (es := eSt m₀ d t h)
    (lw := lW m₀ d t h p) (mw := mW m₀ d t h p)
    (Pw := ∑ s ∈ toks d t, (|cbTS m₀ d t h s| * |dtS m₀ d t h s| *
      errE (dS m₀ d t h s) (dErrS m₀ d t h s) + wAbs m₀ d t h s) * |xv m₀ d s h p|)
    (Sw := ∑ s ∈ toks d t, wMag m₀ d t h s * |xv m₀ d s h p|) (K := (nIt d : ℝ))
    (xD := |xv m₀ d t h p * Dv m₀ d h|) (rel := relD d.BSD) (ab := absD d.BSD)
    (rel32 := relD BK) (a32 := absD BK) (eSig := eSig m₀ d t h p) (aSig := aSig m₀ d t h p)
    (errSig := errSig m₀ d t h p) (errW := errW m₀ d t h p) (errDot := errDot m₀ d t h p)
    (aAcc := aAcc m₀ d t h p) (errMA := errMA m₀ d t h p)
    (abs_nonneg _) (numfinal_hSt_nonneg m₀ d t h) (numfinal_lSig_nonneg m₀ d t h p)
    (numfinal_mSig_nonneg m₀ d t h p) (numfinal_cSum_nonneg m₀ d t h)
    (numfinal_eSt_nonneg m₀ d t h) (numfinal_lW_nonneg m₀ d t h p)
    (Finset.sum_nonneg fun s _ => mul_nonneg (add_nonneg (mul_nonneg (mul_nonneg (abs_nonneg _)
      (abs_nonneg _)) (numfinal_errE_nonneg (numfinal_dErrS_nonneg m₀ d t h s)))
      (numfinal_wAbs_nonneg m₀ d t h s)) (abs_nonneg _))
    (Nat.cast_nonneg _) hK hrel hab0 hab (by rw [numfinal_relD]; norm_num [BK])
    (by rw [numfinal_absD]; norm_num [BK])
    (by simpa only [uB, etaB, mB] using numfinal_lift m₀ d t h p) (numfinal_mW_le_lW m₀ d t h p)
    (numfinal_wMagx_le_lW m₀ d t h p) (numfinal_xD_le m₀ d t h p) (abs_nonneg _)
    (by simp only [eSig, uB, etaB])
    (by simp only [aSig, uF, etaF])
    (by simp only [errSig, uF, etaF])
    (by
      simp only [errW, ewS, uB, uF, ← Finset.sum_add_distrib, Finset.mul_sum]
      exact Finset.sum_congr rfl fun s _ => by ring)
    (by simp only [errDot])
    (by simp only [aAcc])
    (by simp only [errMA])
  unfold fine errPre tol mag expErr uB
  linarith

/-! ## The float operations of the epilogue -/

/-- Reading a finite `bfloat16` value and widening it to `float32` is exact. -/
theorem numfinal_read_exact {b : BF16} {v : F32} (hb : b.isFinite = true)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn b) v) : v.isFinite = true ∧ v.toRat = b.toRat := by
  have hw := FloatV.toRat_cvt_of_wider (g := .binary32) Format.bfloat16_regular
    Format.binary32_regular Format.wider_bfloat16_binary32 RoundMode.rn hb
  have hnan : (FloatV.cvtWith .rn b : F32).isNaN = false := (FloatV.isFinite_iff.mp hw.1).1
  have heq : v = FloatV.cvtWith .rn b := FloatV.NaNEq.eq h hnan
  rw [heq]; exact hw

/-- The arithmetic of two roundings: a product with error `d ≤ a·u + η`, then a sum of it with
`acc` (`|acc| = b`) with error `e ≤ S·u + η`, `S ≤ d + a + b`. -/
theorem numfinal_two_roundings {a b d S e u η : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (hη : 0 ≤ η) (hd : d ≤ a * u + η) (hS : S ≤ d + a + b)
    (he : e ≤ S * u + η) : e + d ≤ (2 * u + 2 * u * u) * (a + b) + 3 * η := by
  have h1 : S * u ≤ (d + a + b) * u := mul_le_mul_of_nonneg_right hS hu0
  have h2 : (d + a + b) * u ≤ (a * u + η + a + b) * u :=
    mul_le_mul_of_nonneg_right (by linarith) hu0
  have h3 : η * u ≤ η := mul_le_of_le_one_right hη hu1
  have h4 : 0 ≤ a * (u * u) := by positivity
  have h5 : 0 ≤ b * (u * u) := by positivity
  have h6 : b * u ≤ 2 * u * b := by nlinarith
  nlinarith

/-- **`tl.mul_add` on `float32`** (a rounded product added, or an `fma`) of finite operands with
room: finite, within `(2^-23 + 2^-47)·(|x·D| + |acc|) + 3·2^-150` of `x·D + acc`. -/
theorem numfinal_mulAdd {x D acc r : F32} (hx : x.isFinite = true) (hD : D.isFinite = true)
    (hacc : acc.isFinite = true) (hroom : |x.toReal * D.toReal| + |acc.toReal| ≤ 2 ^ 120)
    (hr : FloatV.NaNEq (FloatV.add (FloatV.mul x D) acc) r ∨ FloatV.NaNEq (FloatV.fma x D acc) r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal * D.toReal + acc.toReal)| ≤
      (2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ)) * (|x.toReal * D.toReal| + |acc.toReal|) +
        3 * 2 ^ (-150 : ℤ) := by
  have hmax : (2 : ℝ) ^ 122 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  have hxD : 0 ≤ |x.toReal * D.toReal| := abs_nonneg _
  have hac : 0 ≤ |acc.toReal| := abs_nonneg _
  have hu : (0 : ℝ) < 2 ^ (-24 : ℤ) := by positivity
  have hu1 : (2 : ℝ) ^ (-24 : ℤ) ≤ 1 / 2 ^ 24 := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 := by norm_num
  have h23 : (2 : ℝ) ^ (-23 : ℤ) = 2 * 2 ^ (-24 : ℤ) := by norm_num
  have h47 : (2 : ℝ) ^ (-47 : ℤ) = 2 ^ (-23 : ℤ) * 2 ^ (-24 : ℤ) := by norm_num
  rcases hr with hr | hr
  · have hm := FloatV.mul_err_real Format.binary32_regular hx hD
      (by
        have h3 : (2 : ℝ) ^ 122 = 2 * 2 ^ 120 + 2 * 2 ^ 120 := by norm_num
        linarith)
    rw [Format.binary32_u_real, Format.binary32_eta_real] at hm
    obtain ⟨hmf, hme⟩ := hm
    have hab := FloatV.add_err_real Format.binary32_regular hmf hacc
      (by
        have h1 : |(FloatV.mul x D).toReal + acc.toReal| ≤
            |(FloatV.mul x D).toReal - x.toReal * D.toReal| + |x.toReal * D.toReal| + |acc.toReal| := by
          calc _ = |((FloatV.mul x D).toReal - x.toReal * D.toReal) + x.toReal * D.toReal + acc.toReal| := by ring_nf
            _ ≤ _ := abs_add_three _ _ _
        have h2 : |x.toReal * D.toReal| * 2 ^ (-24 : ℤ) ≤ |x.toReal * D.toReal| :=
          mul_le_of_le_one_right hxD (by norm_num)
        have h3 : (2 : ℝ) ^ 122 = 2 * 2 ^ 120 + 2 * 2 ^ 120 := by norm_num
        linarith)
    rw [Format.binary32_u_real, Format.binary32_eta_real] at hab
    obtain ⟨haf, hae⟩ := hab
    have hnan : (FloatV.add (FloatV.mul x D) acc).isNaN = false := (FloatV.isFinite_iff.mp haf).1
    have heq : r = FloatV.add (FloatV.mul x D) acc := FloatV.NaNEq.eq hr hnan
    subst heq
    refine ⟨haf, ?_⟩
    have hd := numfinal_two_roundings (a := |x.toReal * D.toReal|) (b := |acc.toReal|)
      (d := |(FloatV.mul x D).toReal - x.toReal * D.toReal|)
      (S := |(FloatV.mul x D).toReal + acc.toReal|)
      (e := |(FloatV.add (FloatV.mul x D) acc).toReal - ((FloatV.mul x D).toReal + acc.toReal)|)
      (u := 2 ^ (-24 : ℤ)) (η := 2 ^ (-150 : ℤ)) hxD hac hu.le (by norm_num)
      (by positivity) hme (by
        calc _ = |((FloatV.mul x D).toReal - x.toReal * D.toReal) + x.toReal * D.toReal + acc.toReal| := by ring_nf
          _ ≤ _ := abs_add_three _ _ _) hae
    have htri : |(FloatV.add (FloatV.mul x D) acc).toReal - (x.toReal * D.toReal + acc.toReal)| ≤
        |(FloatV.add (FloatV.mul x D) acc).toReal - ((FloatV.mul x D).toReal + acc.toReal)| +
          |(FloatV.mul x D).toReal - x.toReal * D.toReal| := by
      calc _ = |((FloatV.add (FloatV.mul x D) acc).toReal - ((FloatV.mul x D).toReal + acc.toReal)) +
            ((FloatV.mul x D).toReal - x.toReal * D.toReal)| := by ring_nf
        _ ≤ _ := abs_add_le _ _
    have h47' : (2 : ℝ) ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) = 2 * 2 ^ (-24 : ℤ) + 2 * 2 ^ (-24 : ℤ) * 2 ^ (-24 : ℤ) := by
      norm_num
    rw [h47']
    linarith
  · have hf := FloatV.fma_err_real Format.binary32_regular hx hD hacc (by
      have h3 : (2 : ℝ) ^ 122 = 2 * 2 ^ 120 + 2 * 2 ^ 120 := by norm_num
      have := abs_add_le (x.toReal * D.toReal) acc.toReal
      rw [Format.binary32_maxVal_real] at hmax ⊢; linarith)
    rw [Format.binary32_u_real, Format.binary32_eta_real] at hf
    obtain ⟨hff, hfe⟩ := hf
    have hnan : (FloatV.fma x D acc).isNaN = false := (FloatV.isFinite_iff.mp hff).1
    have heq : r = FloatV.fma x D acc := FloatV.NaNEq.eq hr hnan
    subst heq
    refine ⟨hff, ?_⟩
    have h1 : |x.toReal * D.toReal + acc.toReal| ≤ |x.toReal * D.toReal| + |acc.toReal| :=
      abs_add_le _ _
    have h2 := mul_le_mul_of_nonneg_right h1 hu.le
    have h47' : (2 : ℝ) ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) = 2 * 2 ^ (-24 : ℤ) + 2 * 2 ^ (-24 : ℤ) * 2 ^ (-24 : ℤ) := by
      norm_num
    have h6 : 0 ≤ (|x.toReal * D.toReal| + |acc.toReal|) * (2 ^ (-24 : ℤ) * 2 ^ (-24 : ℤ)) := by positivity
    have h7 : 0 ≤ (|x.toReal * D.toReal| + |acc.toReal|) * 2 ^ (-24 : ℤ) := by positivity
    have h8 : (0 : ℝ) < 2 ^ (-150 : ℤ) := by positivity
    rw [h47']
    linarith


/-- **The store's `bfloat16` rounding** of a `float32` value of magnitude at most `2^121`: finite,
within `2^-8·|r| + 2^-134`. -/
theorem numfinal_store {r : F32} {out : BF16} (hr : r.isFinite = true)
    (hroom : |r.toReal| ≤ 2 ^ 121) (ho : FloatV.NaNEq (FloatV.cvtWith .rn r) out) :
    out.isFinite = true ∧ |out.toReal - r.toReal| ≤ 2 ^ (-8 : ℤ) * |r.toReal| + 2 ^ (-134 : ℤ) := by
  have hmax : |r.toReal| ≤ (Format.bfloat16.maxVal : ℝ) := by
    rw [Format.bfloat16_maxVal_real]; norm_num; linarith
  have h := FloatV.cvt_err_real (g := .bfloat16) Format.bfloat16_regular hr hmax
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h
  have hnan : (FloatV.cvtWith .rn r : BF16).isNaN = false := (FloatV.isFinite_iff.mp h.1).1
  have heq : out = FloatV.cvtWith .rn r := FloatV.NaNEq.eq ho hnan
  subst heq
  have h2 : |(FloatV.cvtWith .rn r : BF16).toReal - r.toReal| ≤
      |r.toReal| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := h.2
  exact ⟨h.1, by rw [mul_comm] at h2; exact h2⟩

/-- **The output element.** `acc` is the loop's result; `D`, `x` are read and converted to
`float32` exactly, `r = acc + x·D` is a `tl.mul_add` (two roundings, or one `fma`), and `out` is
its `bfloat16` rounding. Then `out` is finite and within `tol` of `yRef`. -/
theorem final_fact {m₀ : Mem} {d : Inst} (hd : d.ok) (hV : Vals m₀ d) {t h p : Nat}
    (ht : t < d.T) (hh : h < d.H) (hp : p < d.P) {acc D xr r : F32} {D0 xr0 out : BF16}
    (hacc : acc.isFinite = true ∧
      |(acc.toRat : ℝ) - (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p)| ≤
        errSig m₀ d t h p + errW m₀ d t h p + errDot m₀ d t h p ∧
      |(acc.toRat : ℝ)| ≤ aAcc m₀ d t h p)
    (hD0 : m₀.read .bf16 (d.D.ptr (h : Int)).addr = some D0)
    (hD : FloatV.NaNEq (FloatV.cvtWith .rn D0) D)
    (hx0 : m₀.read .bf16 (d.x.ptr (xIdx d t h p : Int)).addr = some xr0)
    (hxr : FloatV.NaNEq (FloatV.cvtWith .rn xr0) xr)
    (hr : FloatV.NaNEq (FloatV.add (FloatV.mul xr D) acc) r ∨
      (FloatV.NaNEq (FloatV.fma xr D acc) r))
    (ho : FloatV.NaNEq (FloatV.cvtWith .rn r) out) :
    out.isFinite = true ∧ |out.toReal - yRef m₀ d t h p| ≤ tol m₀ d t h p := by
  obtain ⟨-, hVx, hVD, -, -, -, -, -, hVl⟩ := hV
  obtain ⟨-, -, -, -, -, hroom⟩ := hVl t h p ht hh hp
  -- the operands, read exactly
  obtain ⟨vx, hvx, hvxf⟩ := hVx t h p ht hh hp
  have hxx : xr0 = vx := Option.some.inj (hx0.symm.trans hvx)
  subst hxx
  obtain ⟨hxf, hxrat⟩ := numfinal_read_exact hvxf hxr
  have hxv : xv m₀ d t h p = xr.toReal := by
    unfold xv valB
    rw [hvx]
    simp only [FloatV.toReal, hxrat]
  obtain ⟨vD, hvD, hvDf⟩ := hVD h hh
  have hDD : D0 = vD := Option.some.inj (hD0.symm.trans hvD)
  subst hDD
  obtain ⟨hDf, hDrat⟩ := numfinal_read_exact hvDf hD
  have hDv : Dv m₀ d h = D.toReal := by
    unfold Dv valB
    rw [hvD]
    simp only [FloatV.toReal, hDrat]
  -- the `tl.mul_add`
  obtain ⟨haf, hZ, haA⟩ := hacc
  have haA' : |acc.toReal| ≤ aAcc m₀ d t h p := haA
  have hroom' : |xr.toReal * D.toReal| + |acc.toReal| ≤ 2 ^ 120 := by
    rw [← hxv, ← hDv]; linarith
  obtain ⟨hrf, hre⟩ := numfinal_mulAdd hxf hDf haf hroom' hr
  rw [← hxv, ← hDv] at hre
  have hcoef : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) := by positivity
  have hre' : |r.toReal - (xv m₀ d t h p * Dv m₀ d h + acc.toReal)| ≤ errMA m₀ d t h p := by
    unfold errMA
    have := mul_le_mul_of_nonneg_left (add_le_add_left haA' |xv m₀ d t h p * Dv m₀ d h|) hcoef
    linarith
  -- the store
  have hr121 : |r.toReal| ≤ 2 ^ 121 := by
    have h1 : (2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) : ℝ) * (|xv m₀ d t h p * Dv m₀ d h| + |acc.toReal|) ≤
        (2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ)) * 2 ^ 120 :=
      mul_le_mul_of_nonneg_left (by linarith) hcoef
    have h2 : (2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) : ℝ) * 2 ^ 120 + 3 * 2 ^ (-150 : ℤ) ≤ 2 ^ 120 := by
      norm_num
    have h3 := abs_add_le (xv m₀ d t h p * Dv m₀ d h) acc.toReal
    have h4 := abs_sub_abs_le_abs_sub r.toReal (xv m₀ d t h p * Dv m₀ d h + acc.toReal)
    have : (2 : ℝ) ^ 121 = 2 * 2 ^ 120 := by norm_num
    linarith
  obtain ⟨hof, hoe⟩ := numfinal_store hrf hr121 ho
  refine ⟨hof, ?_⟩
  -- composition
  have hy : yRef m₀ d t h p = (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
      ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p) + xv m₀ d t h p * Dv m₀ d h := by
    unfold yRef; ring
  have hE : |r.toReal - yRef m₀ d t h p| ≤ errPre m₀ d t h p := by
    rw [hy]
    have h1 := abs_add_le (r.toReal - (xv m₀ d t h p * Dv m₀ d h + acc.toReal))
      (acc.toReal - (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p))
    have h2 : r.toReal - ((Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p) + xv m₀ d t h p * Dv m₀ d h) =
        (r.toReal - (xv m₀ d t h p * Dv m₀ d h + acc.toReal)) +
          (acc.toReal - (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
            ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p)) := by ring
    unfold errPre
    rw [h2]
    have hZ' : |acc.toReal - (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p)| ≤ _ := hZ
    linarith
  have hfine : |out.toReal - yRef m₀ d t h p| ≤ fine m₀ d t h p := by
    unfold fine uB
    have h1 := abs_add_le (out.toReal - r.toReal) (r.toReal - yRef m₀ d t h p)
    have h2 : out.toReal - yRef m₀ d t h p = (out.toReal - r.toReal) + (r.toReal - yRef m₀ d t h p) := by
      ring
    have h3 := abs_sub_abs_le_abs_sub r.toReal (yRef m₀ d t h p)
    have h4 : (0 : ℝ) ≤ 2 ^ (-8 : ℤ) := by positivity
    have h5 : |r.toReal| ≤ |yRef m₀ d t h p| + errPre m₀ d t h p := by
      have := abs_sub_abs_le_abs_sub r.toReal (yRef m₀ d t h p)
      linarith
    have h6 := mul_le_mul_of_nonneg_left h5 h4
    rw [h2]
    linarith
  exact hfine.trans (numfinal_fine_le_tol hd m₀ t h p)

end Claims.SsdChunkScan
