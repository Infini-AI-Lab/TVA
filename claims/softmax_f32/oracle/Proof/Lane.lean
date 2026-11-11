import Proof.Max

/-!
# One lane

The rounded difference `v_i - m` (`b32_sub_facts`) and the platform's `exp` of it: `lane_bound`
bounds `num_i` against `e^(x_i - m)` by `(4u|t_i| + 11u)·e^(t_i) + 2^-137`.
-/
namespace Claims.Softmax

open TVA

/-! ## The rounded difference `v_i − m` and the platform's `exp` at `0` -/

/-- `e^{-x} ≤ 2^{-k}` for `x ≥ k`, since `e ≥ 2`. -/
lemma exp_neg_le_pow2 {x : ℝ} {k : Nat} (hx : (k : ℝ) ≤ x) :
    Real.exp (-x) ≤ (2 : ℝ) ^ (-(k : ℤ)) := by
  have h1 : Real.exp (k : ℝ) ≤ Real.exp x := Real.exp_le_exp.mpr hx
  have h2 : Real.exp (k : ℝ) = (Real.exp 1) ^ k := by
    rw [← Real.exp_nat_mul]; norm_num
  have h3 : (2 : ℝ) ≤ Real.exp 1 := le_of_lt Real.exp_one_gt_two
  have h4 : (2 : ℝ) ^ k ≤ (Real.exp 1) ^ k := pow_le_pow_left₀ (by norm_num) h3 k
  have h5 : (2 : ℝ) ^ k ≤ Real.exp x := by rw [h2] at h1; linarith
  have h6 : Real.exp (-x) = (Real.exp x)⁻¹ := by rw [← Real.exp_neg]
  have h9 : (2 : ℝ) ^ (-(k : ℤ)) = ((2 : ℝ) ^ k)⁻¹ := by
    rw [zpow_neg, zpow_natCast]
  rw [h6, h9]
  exact (inv_le_inv₀ (Real.exp_pos x) (by positivity)).mpr h5

/-- `|e^x − 1| ≤ 2|x|` for `|x| ≤ 1/2`. -/
lemma abs_exp_sub_one_le {x : ℝ} (hx : |x| ≤ 1 / 2) : |Real.exp x - 1| ≤ 2 * |x| := by
  have hlo : -(1 / 2 : ℝ) ≤ x := by linarith [neg_abs_le x]
  have hhi : x ≤ 1 / 2 := by linarith [le_abs_self x]
  rcases lt_or_ge x 0 with hx0 | hx0
  · have h1 : Real.exp x ≤ 1 := by
      have h3 := Real.exp_le_exp.mpr (le_of_lt hx0)
      simpa using h3
    have h2 : x + 1 ≤ Real.exp x := Real.add_one_le_exp x
    have h3 : Real.exp x - 1 ≤ 0 := by linarith
    rw [abs_of_nonpos h3, abs_of_neg hx0]
    linarith
  · have h1 : Real.exp x ≤ 1 + 2 * x := exp_le_one_add_two_mul hx0 hhi
    have h2 : 0 ≤ Real.exp x - 1 := by
      have h3 : 1 ≤ Real.exp x := by simpa using Real.exp_le_exp.mpr hx0
      linarith
    rw [abs_of_nonneg h2, abs_of_nonneg hx0]
    linarith

/-- A constant of `float32` as a real. -/
lemma b32_u_cast : ((Format.binary32.u : ℚ) : ℝ) = uR := by
  unfold uR
  rw [Format.binary32_u]
  norm_num

/-- A constant of `float32` as a real. -/
lemma b32_eta_cast : ((Format.binary32.eta : ℚ) : ℝ) = etaR := by
  unfold etaR
  rw [Format.binary32_eta]
  norm_num

/-- A constant of `float32` as a real. -/
lemma b32_maxVal_cast :
    ((Format.binary32.maxVal : ℚ) : ℝ) = ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]
  norm_num

/-- A round-to-nearest magnitude is below the sign bit. -/
lemma roundMag_rn_lt_signBit {f : Format} (hf : f.Regular) (hs : f.specials = .ieee)
    {a : ℚ} (ha : 0 < a) : f.roundMag .rn false a < f.signBit := by
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := f) .rn ha
  by_cases hle : (Format.roundSig .rn a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) ≤ f.maxVal
  · have hpack : f.roundMag .rn false a =
        f.pack (Format.roundSig .rn a (f.qexp a)) (f.qexp a) := by
      unfold Format.roundMag Format.encode
      rw [Format.pow2_eq, ite_eq_left hle]
    rw [hpack]
    exact lt_of_le_of_lt (Format.pack_le_maxMag hf hN hQ hsub hle) (Format.maxMag_lt hf)
  · unfold Format.roundMag Format.encode
    rw [Format.pow2_eq, ite_eq_right hle]
    simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
    rw [hs]
    exact FloatV.infMag_lt_signBit

/-- The pattern a round-to-nearest produces is a pattern. -/
lemma roundRat_rn_lt {f : Format} (hf : f.Regular) (hs : f.specials = .ieee)
    {zs : Bool} {q : ℚ} : f.roundRat .rn false zs q < 2 ^ f.width := by
  by_cases hq : q = 0
  · subst hq
    exact (Format.roundRat_zero_spec (f := f) hf).1
  · rw [Format.roundRat_of_ne (f := f) hq]
    have hlt : f.roundMag .rn false |q| < f.signBit :=
      roundMag_rn_lt_signBit hf hs (abs_pos.mpr hq)
    by_cases hneg : q < 0
    · rw [ite_eq_left hneg]
      have h2 : f.signBit + f.roundMag .rn false |q| < f.signBit + f.signBit :=
        Nat.add_lt_add_left hlt f.signBit
      have h3 : f.signBit + f.signBit = 2 ^ f.width := by
        rw [← Format.two_signBit (f := f)]; ring
      rw [h3] at h2
      exact h2
    · rw [ite_eq_right hneg, Nat.zero_add]
      exact lt_of_lt_of_le hlt (le_of_lt (Format.signBit_lt (f := f)))

/-- A finite `x ≤ y` gives `x − y` either finite with one rounding's error, or exactly `−∞`. -/
lemma b32_sub_facts {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hxy : x.toRat ≤ y.toRat) :
    ((FloatV.sub x y).isFinite = true ∧
      |(FloatV.sub x y).toRat - (x.toRat - y.toRat)| ≤
        |(FloatV.sub x y).toRat| * Format.binary32.u + Format.binary32.eta) ∨
    FloatV.sub x y = FloatV.negInf := by
  have hsub : FloatV.sub x y = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)) := by
    unfold FloatV.sub
    rw [Format.sub_of_finite (f := Format.binary32) .rn hx hy]
    rfl
  rw [hsub]
  by_cases hQ : x.toRat - y.toRat = 0
  · rw [hQ]
    obtain ⟨hlt, hfin, hval⟩ := Format.roundRat_zero_spec (f := Format.binary32) b32_reg
      (rnd := .rn) (sat := false) (zs := Format.binary32.sign x.pat && !Format.binary32.sign y.pat)
    left
    refine ⟨?_, ?_⟩
    · rw [FloatV.isFinite_ofPat hlt]; exact hfin
    · rw [FloatV.toRat_ofPat hlt, hval]
      simp only [sub_zero, abs_zero, zero_mul, zero_add]
      exact Format.eta_nonneg
  · have hQle : x.toRat - y.toRat ≤ 0 := by linarith
    have hQneg : x.toRat - y.toRat < 0 := lt_of_le_of_ne hQle hQ
    have hQa : 0 < |x.toRat - y.toRat| := abs_pos.mpr hQ
    by_cases hfin : Format.binary32.isFinite (Format.binary32.roundRat .rn false
        (Format.binary32.sign x.pat && !Format.binary32.sign y.pat)
        (x.toRat - y.toRat)) = true
    · left
      have hlt := roundRat_rn_lt (f := Format.binary32) b32_reg b32_ieee
        (zs := Format.binary32.sign x.pat && !Format.binary32.sign y.pat)
        (q := x.toRat - y.toRat)
      refine ⟨?_, ?_⟩
      · rw [FloatV.isFinite_ofPat hlt]; exact hfin
      · rw [FloatV.toRat_ofPat hlt]
        exact Format.roundRat_rn_err_of_finite (f := Format.binary32) b32_reg
          (by rw [b32_ieee]; decide) _ hfin
    · right
      have hne : ¬ ((Format.roundSig .rn |x.toRat - y.toRat|
            (Format.binary32.qexp |x.toRat - y.toRat|) : ℚ) *
            (2 : ℚ) ^ (Format.binary32.qexp |x.toRat - y.toRat|) ≤ Format.binary32.maxVal) := by
        intro hle
        obtain ⟨hN, hQr, hsubr⟩ := Format.roundSig_bounds (f := Format.binary32) .rn hQa
        have hpack : Format.binary32.roundMag .rn false |x.toRat - y.toRat| =
            Format.binary32.pack (Format.roundSig .rn |x.toRat - y.toRat|
              (Format.binary32.qexp |x.toRat - y.toRat|))
              (Format.binary32.qexp |x.toRat - y.toRat|) := by
          unfold Format.roundMag Format.encode
          rw [Format.pow2_eq, ite_eq_left hle]
        have hb := Format.pack_le_maxMag (f := Format.binary32) b32_reg hN hQr hsubr hle
        have hlt := lt_of_le_of_lt hb (Format.maxMag_lt b32_reg)
        have hrat : Format.binary32.roundRat .rn false
            (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat) =
            Format.binary32.signBit +
              Format.binary32.roundMag .rn false |x.toRat - y.toRat| := by
          rw [Format.roundRat_of_ne (f := Format.binary32) hQ, ite_eq_left hQneg]
        have htrue : Format.binary32.isFinite (Format.binary32.pack
            (Format.roundSig .rn |x.toRat - y.toRat|
              (Format.binary32.qexp |x.toRat - y.toRat|))
            (Format.binary32.qexp |x.toRat - y.toRat|)) = true :=
          Format.isFinite_of_mag_le b32_reg (by rw [Format.mag_of_lt hlt]; exact hb)
        have hsign : Format.binary32.signBit + Format.binary32.pack
            (Format.roundSig .rn |x.toRat - y.toRat|
              (Format.binary32.qexp |x.toRat - y.toRat|))
            (Format.binary32.qexp |x.toRat - y.toRat|) =
            Format.binary32.setSign true (Format.binary32.pack
              (Format.roundSig .rn |x.toRat - y.toRat|
                (Format.binary32.qexp |x.toRat - y.toRat|))
              (Format.binary32.qexp |x.toRat - y.toRat|)) := by
          rw [Format.setSign, Format.mag_of_lt hlt]
          simp
        rw [hrat, hpack, hsign, Format.isFinite_setSign] at hfin
        exact hfin htrue
      have hround : Format.binary32.roundMag .rn false |x.toRat - y.toRat| =
          Format.binary32.infMag := by
        unfold Format.roundMag Format.encode
        rw [Format.pow2_eq, ite_eq_right hne]
        simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
        rw [b32_ieee]
      have hsign : Format.binary32.signBit + Format.binary32.infMag =
          Format.binary32.setSign true Format.binary32.infMag := by
        rw [Format.setSign, Format.mag_of_lt FloatV.infMag_lt_signBit]
        simp
      have hrat : Format.binary32.roundRat .rn false
          (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat) =
          Format.binary32.setSign true Format.binary32.infMag := by
        rw [Format.roundRat_of_ne (f := Format.binary32) hQ, ite_eq_left hQneg, hround, hsign]
      rw [hrat]
      rfl

/-! ## One lane

A lane's `v_i − m` is one rounding of `x_i − m`; the platform's `exp` then lands within
`(2^-22 + 2^-23·|s_i|)·e^{s_i} + 2^-140` of `e^{s_i}`. This section pushes that through to
`|num_i − e^{x_i − m}| ≤ (4u|t_i| + 11u)·e^{t_i} + 2^-137`, where `t_i = x_i − m`. -/

/-- A numeric fact about powers of two. -/
lemma e137_ge : (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-137 : ℤ) := by norm_num

/-- A numeric fact about the constant. -/
lemma etaR_small : etaR ≤ 1 / 16 := by unfold etaR; norm_num

/-- `e^x ≤ 1 + 2|x|` for `|x| ≤ 1/2`. -/
lemma exp_le_one_add_two_abs {x : ℝ} (hx : |x| ≤ 1 / 2) : Real.exp x ≤ 1 + 2 * |x| := by
  rcases lt_or_ge x 0 with h | h
  · have h1 : Real.exp x ≤ 1 := by
      have h3 := Real.exp_le_exp.mpr (le_of_lt h)
      simpa using h3
    have h2 : 0 ≤ 2 * |x| := by positivity
    linarith
  · have h2 : x ≤ 1 / 2 := by linarith [le_abs_self x]
    have h3 : Real.exp x ≤ 1 + 2 * x := exp_le_one_add_two_mul h h2
    rw [abs_of_nonneg h]
    exact h3

/-- `uR · |t| ≤ 1/16` for `|t| ≤ 2^20`. -/
lemma q_le {t : ℝ} (ht2 : |t| ≤ 2 ^ 20) : uR * |t| ≤ 1 / 16 := by
  have h1 : uR * |t| ≤ uR * 2 ^ 20 := by nlinarith [uR_pos, abs_nonneg t]
  have h2 : uR * 2 ^ 20 = 1 / 16 := by unfold uR; norm_num
  linarith

/-- `|e^d - 1| ≤ |d| · e^|d|`. -/
lemma abs_exp_sub_one_le_mul (d : ℝ) : |Real.exp d - 1| ≤ |d| * Real.exp |d| := by
  rcases lt_or_ge d 0 with h | h
  · have hd : |d| = -d := abs_of_neg h
    have he : Real.exp |d| = Real.exp (-d) := by rw [hd]
    have h2 : Real.exp d - 1 ≤ 0 := by
      have h3 : Real.exp d ≤ 1 := by simpa using Real.exp_le_exp.mpr (le_of_lt h)
      linarith
    rw [abs_of_nonpos h2, he, hd, neg_sub]
    have h3 : 1 - Real.exp d ≤ -d := by
      have := Real.add_one_le_exp d; linarith
    have h4 : -d ≤ -d * Real.exp (-d) := by
      have h5 : (1 : ℝ) ≤ Real.exp (-d) := Real.one_le_exp (by linarith : (0 : ℝ) ≤ -d)
      nlinarith
    linarith
  · have hd : |d| = d := abs_of_nonneg h
    have he : Real.exp |d| = Real.exp d := by rw [hd]
    have h2 : 0 ≤ Real.exp d - 1 := by
      have h3 : 1 ≤ Real.exp d := Real.one_le_exp h
      linarith
    rw [abs_of_nonneg h2, he, hd]
    have h3 : 1 - d ≤ Real.exp (-d) := Real.one_sub_le_exp_neg d
    have h4 : Real.exp d * (1 - d) ≤ 1 := by
      have h5 : Real.exp d * Real.exp (-d) = 1 := by rw [← Real.exp_add]; simp
      have h6 : (0 : ℝ) < Real.exp d := Real.exp_pos d
      nlinarith
    nlinarith [Real.exp_pos d]

/-- The rounded difference `s` of `t` has `|s - t| ≤ uR|s| + etaR`, bounding the gap between their exponents. -/
lemma sub_gap_le {s t : ℝ} (hst : |s - t| ≤ uR * |s| + etaR) :
    |s - t| ≤ uR * |t| + 2 * uR ^ 2 * |t| + 2 * etaR := by
  have h1 : |s| ≤ |t| + |s - t| := by
    have h := abs_add_le t (s - t)
    rwa [show t + (s - t) = s by ring] at h
  have h2 : |s - t| ≤ uR * (|t| + |s - t|) + etaR := by
    nlinarith [mul_le_mul_of_nonneg_left h1 uR_pos.le]
  have h3 : (0 : ℝ) < 1 - uR := by linarith [uR_le]
  have h4 : |s - t| * (1 - uR) ≤ uR * |t| + etaR := by nlinarith [uR_pos]
  have h5 : |s - t| ≤ (uR * |t| + etaR) / (1 - uR) := by
    rw [le_div_iff₀ h3]; linarith
  have h6 : (uR * |t| + etaR) / (1 - uR) ≤ uR * |t| + 2 * uR ^ 2 * |t| + 2 * etaR := by
    rw [div_le_iff₀ h3]
    have h8 : (0 : ℝ) ≤ (uR ^ 2 * |t| + etaR) * (1 - 2 * uR) := by
      have h9 : (0 : ℝ) ≤ 1 - 2 * uR := by linarith [uR_le]
      have h10 : (0 : ℝ) ≤ uR ^ 2 * |t| + etaR := by
        have h1 := etaR_pos; have h2 := abs_nonneg t; have h3 := uR_pos
        nlinarith [sq_nonneg uR]
      exact mul_nonneg h10 h9
    nlinarith [h8]
  linarith

set_option maxHeartbeats 400000 in
/-- The lane bound for `|t| ≤ 2^20`. -/
lemma num_lane_small {t s ns : ℝ} (ht2 : |t| ≤ 2 ^ 20)
    (hst : |s - t| ≤ uR * |s| + etaR)
    (hrule : |ns - Real.exp s| ≤ (4 * uR + 2 * uR * |s|) * Real.exp s + (2 : ℝ) ^ (-140 : ℤ)) :
    |ns - Real.exp t| ≤ (4 * uR * |t| + 11 * uR) * Real.exp t + (2 : ℝ) ^ (-137 : ℤ) := by
  set d := s - t with hd
  have hsd : s = t + d := by rw [hd]; ring
  have hdle : |d| ≤ uR * |t| + 2 * uR ^ 2 * |t| + 2 * etaR := by rw [hd]; exact sub_gap_le hst
  have hq : uR * |t| ≤ 1 / 16 := q_le ht2
  have hq0 : 0 ≤ uR * |t| := by have := uR_pos; positivity
  have hu2 : uR ^ 2 * |t| ≤ uR / 16 := by nlinarith [hq, uR_pos]
  have hd2 : |d| ≤ 1 / 2 := by
    have h1 := etaR_small; have h2 := uR_pos; have h3 := uR_le
    linarith [hdle, hu2]
  have hsabs : |s| ≤ |t| + |d| := by rw [hsd]; exact abs_add_le _ _
  set es := Real.exp s with hesdef
  set et := Real.exp t with hetdef
  have hes : 0 < es := by rw [hesdef]; exact Real.exp_pos s
  have het : 0 < et := by rw [hetdef]; exact Real.exp_pos t
  have hEs : es ≤ (1 + 2 * |d|) * et := by
    have h1 : es = et * Real.exp d := by rw [hesdef, hetdef, hsd, Real.exp_add]
    have h2 : Real.exp d ≤ 1 + 2 * |d| := exp_le_one_add_two_abs hd2
    calc es = et * Real.exp d := h1
      _ ≤ et * (1 + 2 * |d|) := mul_le_mul_of_nonneg_left h2 het.le
      _ = (1 + 2 * |d|) * et := by ring
  have hEd : |es - et| ≤ |d| * (1 + 2 * |d|) * et := by
    have h1 : es - et = et * (Real.exp d - 1) := by
      rw [hesdef, hetdef, hsd, Real.exp_add]; ring
    have h2 : |Real.exp d - 1| ≤ |d| * (1 + 2 * |d|) := by
      have h3 := abs_exp_sub_one_le_mul d
      have h5 : (0 : ℝ) ≤ |d| := abs_nonneg d
      have h4' : Real.exp |d| ≤ 1 + 2 * |d| := by
        simpa [abs_abs] using exp_le_one_add_two_abs (x := |d|) (by simpa [abs_abs] using hd2)
      have h6 : |d| * Real.exp |d| ≤ |d| * (1 + 2 * |d|) :=
        mul_le_mul_of_nonneg_left h4' h5
      linarith
    calc |es - et| = et * |Real.exp d - 1| := by rw [h1, abs_mul, abs_of_pos het]
      _ ≤ et * (|d| * (1 + 2 * |d|)) := mul_le_mul_of_nonneg_left h2 het.le
      _ = |d| * (1 + 2 * |d|) * et := by ring
  have hsplit : |ns - et| ≤ |ns - es| + |es - et| := by
    have := abs_add_le (ns - es) (es - et)
    rwa [show ns - es + (es - et) = ns - et by ring] at this
  have hu16 : uR ≤ 1 / 16 := by unfold uR; norm_num
  have hu1024 : uR ≤ 1 / 1024 := by unfold uR; norm_num
  have heu : 2 * etaR ≤ uR := by unfold etaR uR; norm_num
  have he4u : 4 * uR * etaR ≤ uR := by unfold etaR uR; norm_num
  have hCM : (4 * uR + 2 * uR * |s| + |d|) * (1 + 2 * |d|) ≤ 4 * (uR * |t|) + 11 * uR := by
    have hqsq : (uR * |t|) * (uR * |t|) ≤ (1 / 16) * (uR * |t|) := by nlinarith [hq0, hq]
    have huq : uR * (uR * |t|) ≤ uR * (1 / 16) := mul_le_mul_of_nonneg_left hq uR_pos.le
    have huqq : uR * ((uR * |t|) * (uR * |t|)) ≤ uR * ((1 / 16) * (uR * |t|)) :=
      mul_le_mul_of_nonneg_left hqsq uR_pos.le
    have huu : uR * uR ≤ uR * (1 / 1024) := mul_le_mul_of_nonneg_left hu1024 uR_pos.le
    have huuq : uR * (uR * (uR * |t|)) ≤ uR * (uR * (1 / 16)) :=
      mul_le_mul_of_nonneg_left huq uR_pos.le
    have h1 : 2 * uR * |s| ≤ 2 * uR * (|t| + |d|) :=
      mul_le_mul_of_nonneg_left hsabs (by have := uR_pos; positivity)
    have h2 : (1 + 2 * uR) * |d| ≤ (1 + 2 * uR) * ((uR * |t|) + 2 * uR ^ 2 * |t| + 2 * etaR) :=
      mul_le_mul_of_nonneg_left hdle (by linarith [uR_pos])
    have h3 : uR * (uR * |t|) = uR ^ 2 * |t| := by ring
    have h4 : 2 * |d| ≤ 2 * ((uR * |t|) + 2 * uR ^ 2 * |t| + 2 * etaR) := by linarith [hdle]
    have hC : (4 * uR + 2 * uR * |s| + |d|) ≤ 3 * (uR * |t|) + (17 / 4) * uR * (uR * |t|) + 7 * uR := by
      nlinarith [h1, h2, h3, heu, he4u, hu16, hq0, uR_pos, etaR_pos]
    have hM : (1 + 2 * |d|) ≤ 1 + 2 * (uR * |t|) + (17 / 4) * uR := by
      nlinarith [h4, h3, huq, hq0, uR_pos, etaR_pos, heu]
    have hC0 : (0 : ℝ) ≤ 3 * (uR * |t|) + (17 / 4) * uR * (uR * |t|) + 7 * uR := by
      nlinarith [hq0, uR_pos]
    have hstep : (3 * (uR * |t|) + (17 / 4) * uR * (uR * |t|) + 7 * uR) *
        (1 + 2 * (uR * |t|) + (17 / 4) * uR) ≤ 4 * (uR * |t|) + 11 * uR := by
      nlinarith [hqsq, huq, huqq, huu, huuq, hq0, uR_pos]
    calc (4 * uR + 2 * uR * |s| + |d|) * (1 + 2 * |d|)
        ≤ (3 * (uR * |t|) + (17 / 4) * uR * (uR * |t|) + 7 * uR) *
            (1 + 2 * (uR * |t|) + (17 / 4) * uR) := mul_le_mul hC hM (by positivity) hC0
      _ ≤ 4 * (uR * |t|) + 11 * uR := hstep
  have hmain : (4 * uR + 2 * uR * |s|) * es + |d| * (1 + 2 * |d|) * et ≤
      (4 * (uR * |t|) + 11 * uR) * et := by
    have h1 : (4 * uR + 2 * uR * |s|) * es ≤ (4 * uR + 2 * uR * |s|) * ((1 + 2 * |d|) * et) :=
      mul_le_mul_of_nonneg_left hEs (by have := uR_pos; positivity)
    have h2 : (4 * uR + 2 * uR * |s|) * ((1 + 2 * |d|) * et) + |d| * (1 + 2 * |d|) * et
        = (4 * uR + 2 * uR * |s| + |d|) * (1 + 2 * |d|) * et := by ring
    have h3 : (4 * uR + 2 * uR * |s| + |d|) * (1 + 2 * |d|) * et ≤
        (4 * (uR * |t|) + 11 * uR) * et := mul_le_mul_of_nonneg_right hCM het.le
    linarith
  calc |ns - et| ≤ |ns - es| + |es - et| := hsplit
    _ ≤ ((4 * uR + 2 * uR * |s|) * es + (2 : ℝ) ^ (-140 : ℤ)) + |d| * (1 + 2 * |d|) * et := by
        linarith [hrule, hEd]
    _ ≤ (4 * uR * |t| + 11 * uR) * et + (2 : ℝ) ^ (-137 : ℤ) := by
        have h1 := hmain
        have h2 := e137_ge
        linarith [h1, h2]

/-- `e^-14 ≤ 2^-15`. -/
lemma exp_neg14_le : Real.exp (-(14 : ℝ)) ≤ (2 : ℝ) ^ (-15 : ℤ) := by
  have h2 : (2.7 : ℝ) ≤ Real.exp 1 := le_of_lt (lt_trans (by norm_num) Real.exp_one_gt_d9)
  have h3 : (2.7 : ℝ) ^ 14 ≤ (Real.exp 1) ^ 14 := pow_le_pow_left₀ (by norm_num) h2 14
  have h4 : (32768 : ℝ) ≤ (2.7 : ℝ) ^ 14 := by norm_num
  have h1 : Real.exp (14 : ℝ) = (Real.exp 1) ^ 14 := by
    rw [← Real.exp_nat_mul]; norm_num
  have h5 : (32768 : ℝ) ≤ Real.exp (14 : ℝ) := by rw [h1]; linarith
  have h8 : Real.exp (-(14 : ℝ)) = (Real.exp (14 : ℝ))⁻¹ := by
    rw [← Real.exp_neg]
  have h7 : (2 : ℝ) ^ (-15 : ℤ) = ((32768 : ℝ))⁻¹ := by norm_num
  rw [h8, h7]
  exact (inv_le_inv₀ (Real.exp_pos 14) (by norm_num)).mpr h5

/-- `w · e^-w` is bounded. -/
lemma mul_exp_neg_le {w : ℝ} (_hw : 0 ≤ w) :
    w * Real.exp (-w) ≤ 14 * Real.exp (-w) + (2 : ℝ) ^ (-15 : ℤ) := by
  rcases le_or_gt w 14 with h | h
  · have h1 : w * Real.exp (-w) ≤ 14 * Real.exp (-w) :=
      mul_le_mul_of_nonneg_right h (Real.exp_pos (-w)).le
    have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-15 : ℤ) := by norm_num
    linarith
  · have hpos : 0 < w - 14 := by linarith
    have h1 : w - 14 ≤ Real.exp (w - 14) := by
      have h2 := Real.add_one_le_exp (w - 14)
      linarith
    have h3 : Real.exp (-w) = Real.exp (-(w - 14)) * Real.exp (-14) := by
      rw [← Real.exp_add]; ring_nf
    have h4 : (w - 14) * Real.exp (-(w - 14)) ≤ 1 := by
      have h5 := (Real.exp_pos (-(w - 14))).le
      have h6 : (w - 14) * Real.exp (-(w - 14)) ≤ Real.exp (w - 14) * Real.exp (-(w - 14)) :=
        mul_le_mul_of_nonneg_right h1 h5
      have h7 : Real.exp (w - 14) * Real.exp (-(w - 14)) = 1 := by
        rw [← Real.exp_add]; ring_nf; simp
      linarith
    have h9 : (w - 14) * Real.exp (-w) ≤ Real.exp (-14) := by
      rw [h3]
      have heq : (w - 14) * (Real.exp (-(w - 14)) * Real.exp (-14))
          = ((w - 14) * Real.exp (-(w - 14))) * Real.exp (-14) := by ring
      rw [heq]
      calc ((w - 14) * Real.exp (-(w - 14))) * Real.exp (-14) ≤ 1 * Real.exp (-14) :=
            mul_le_mul_of_nonneg_right h4 (Real.exp_pos (-14)).le
        _ = Real.exp (-14) := by ring
    have h10 : Real.exp (-14) ≤ (2 : ℝ) ^ (-15 : ℤ) := exp_neg14_le
    have hsplit : w * Real.exp (-w) = 14 * Real.exp (-w) + (w - 14) * Real.exp (-w) := by ring
    linarith

/-- The pattern of a finite `float32` difference. -/
lemma b32_sub_pattern {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    FloatV.sub x y = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign x.pat && ! Format.binary32.sign y.pat)
      (x.toRat - y.toRat)) := by
  unfold FloatV.sub
  rw [Format.sub_of_finite (f := Format.binary32) .rn hx hy]
  rfl

/-- A round-to-nearest of a non-positive rational is non-positive. -/
lemma b32_roundRat_rn_toRat_nonpos {zs : Bool} {q : ℚ} (hq : q ≤ 0) :
    Format.binary32.toRat (Format.binary32.roundRat .rn false zs q) ≤ 0 := by
  by_cases h0 : q = 0
  · rw [h0]
    exact le_of_eq (Format.roundRat_zero_spec (f := Format.binary32) b32_reg).2.2
  · have hqneg : q < 0 := lt_of_le_of_ne hq h0
    rw [Format.roundRat_of_ne (f := Format.binary32) h0, ite_eq_left hqneg]
    set M := Format.binary32.roundMag .rn false |q| with hM
    have hlt : M < Format.binary32.signBit :=
      roundMag_rn_lt_signBit b32_reg b32_ieee (abs_pos.mpr h0)
    have hset : Format.binary32.setSign true M = Format.binary32.signBit + M := by
      rw [Format.setSign_of_lt (f := Format.binary32) hlt]; simp
    rw [← hset, Format.toRat_setSign]
    simp only [ite_true]
    have hpos := Format.magVal_nonneg (f := Format.binary32) (Format.binary32.mag M)
    linarith

/-- A finite `float32` is not `-∞`. -/
lemma finite_ne_negInf {x : F32} (hx : x.isFinite = true) : x ≠ FloatV.negInf := by
  intro h
  rw [h] at hx
  rw [FloatV.isFinite_iff] at hx
  rw [FloatV.negInf_isInf b32_ieee] at hx
  exact absurd hx.2 (by simp)

/-- A subtraction that overflows to `−∞` had a gap below `−137`. -/
lemma sub_negInf_gap {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hxy : x.toRat ≤ y.toRat) (hinf : FloatV.sub x y = FloatV.negInf) :
    x.toRat - y.toRat < -(137 : ℚ) := by
  by_contra hcon
  have hge : -(137 : ℚ) ≤ x.toRat - y.toRat := not_lt.mp hcon
  have hq0 : x.toRat - y.toRat ≤ 0 := by linarith
  have hmax : (137 : ℚ) ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]; norm_num
  have hq : |x.toRat - y.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_of_nonpos hq0]; linarith
  obtain ⟨hfin, -⟩ := FloatV.ofPat_roundRat_err (f := Format.binary32) b32_reg false
    (Format.binary32.sign x.pat && ! Format.binary32.sign y.pat) hq
  have hp := b32_sub_pattern hx hy
  rw [hinf] at hp
  rw [← hp] at hfin
  rw [FloatV.isFinite_iff] at hfin
  rw [FloatV.negInf_isInf b32_ieee] at hfin
  exact absurd hfin.2 (by simp)

set_option maxHeartbeats 400000 in
/-- The lane bound for `t ≤ -2^20`, where the result is `0` or tiny. -/
lemma num_lane_big {t s ns : ℝ} (ht : t ≤ -(2 ^ 20 : ℝ))
    (hst : |s - t| ≤ uR * |s| + etaR) (hsc : |s| ≤ (2 : ℝ) ^ 130)
    (hrule : |ns - Real.exp s| ≤ (4 * uR + 2 * uR * |s|) * Real.exp s + (2 : ℝ) ^ (-140 : ℤ)) :
    |ns - Real.exp t| ≤ (4 * uR * |t| + 11 * uR) * Real.exp t + (2 : ℝ) ^ (-137 : ℤ) := by
  have ht0 : t ≤ 0 := by linarith
  have hta : (2 : ℝ) ^ 20 ≤ |t| := by
    rw [abs_of_nonpos ht0]; linarith
  have htabs : |t| = -t := abs_of_nonpos ht0
  have h1 : |s| ≤ |t| + |s - t| := by
    have h := abs_add_le t (s - t)
    rwa [show t + (s - t) = s by ring] at h
  have h2 : |s| ≤ |t| + (uR * |s| + etaR) := by linarith [h1, hst]
  have h3 : (1 - uR) * |s| ≤ |t| + etaR := by nlinarith [h2, uR_pos]
  have h4 : (0 : ℝ) < 1 - uR := by linarith [uR_le]
  have h5 : |s| ≤ 2 * |t| := by
    have h6 : |s| ≤ (|t| + etaR) / (1 - uR) := by
      rw [le_div_iff₀ h4]; linarith
    have h7 : (|t| + etaR) / (1 - uR) ≤ 2 * |t| := by
      rw [div_le_iff₀ h4]
      have heta : etaR ≤ (2 : ℝ) ^ (-23 : ℤ) * |t| := by
        have h20 : etaR ≤ (2 : ℝ) ^ (-23 : ℤ) * (2 : ℝ) ^ 20 := by unfold etaR; norm_num
        have hm := mul_le_mul_of_nonneg_left hta (by norm_num : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ))
        linarith
      have hu : 2 * uR = (2 : ℝ) ^ (-23 : ℤ) := by unfold uR; norm_num
      nlinarith [heta, hu, uR_pos, uR_le, abs_nonneg t]
    linarith
  have hs20 : s ≤ -((2 : ℝ) ^ 19) := by
    have h9 : |s - t| ≤ (2 : ℝ) ^ (-23 : ℤ) * |t| + etaR := by
      have hm := mul_le_mul_of_nonneg_left h5 uR_pos.le
      have hu : uR * (2 * |t|) = (2 : ℝ) ^ (-23 : ℤ) * |t| := by unfold uR; ring
      linarith [hst, hm, hu]
    have h10 : s ≤ t + |s - t| := by linarith [le_abs_self (s - t)]
    have h11 : t + (2 : ℝ) ^ (-23 : ℤ) * |t| + etaR ≤ -((2 : ℝ) ^ 19) := by
      have hc : (0 : ℝ) ≤ 1 - (2 : ℝ) ^ (-23 : ℤ) := by norm_num
      have hmul : (2 : ℝ) ^ 20 * (1 - (2 : ℝ) ^ (-23 : ℤ)) ≤
          |t| * (1 - (2 : ℝ) ^ (-23 : ℤ)) := mul_le_mul_of_nonneg_right hta hc
      have heta19 : etaR ≤ (2 : ℝ) ^ (-3 : ℤ) := by unfold etaR; norm_num
      have hexp : |t| * (1 - (2 : ℝ) ^ (-23 : ℤ)) = |t| - (2 : ℝ) ^ (-23 : ℤ) * |t| := by ring
      have hlow : (2 : ℝ) ^ 20 - (2 : ℝ) ^ (-3 : ℤ) ≤ |t| - (2 : ℝ) ^ (-23 : ℤ) * |t| := by
        have he : (2 : ℝ) ^ 20 * (1 - (2 : ℝ) ^ (-23 : ℤ)) = (2 : ℝ) ^ 20 - (2 : ℝ) ^ (-3 : ℤ) := by
          norm_num
        linarith [hmul, hexp, he]
      have ht' : -|t| = t := by linarith [htabs]
      have hsum1 : t + (2 : ℝ) ^ (-23 : ℤ) * |t| ≤ -((2 : ℝ) ^ 20) + (2 : ℝ) ^ (-3 : ℤ) := by
        linarith [hlow, ht']
      have hsum2 : -((2 : ℝ) ^ 20) + (2 : ℝ) ^ (-3 : ℤ) + etaR ≤ -((2 : ℝ) ^ 19) := by
        have h1 : (2 : ℝ) ^ (-3 : ℤ) + (2 : ℝ) ^ (-3 : ℤ) ≤ (2 : ℝ) ^ 19 := by norm_num
        linarith [heta19, h1]
      linarith [hsum1, hsum2]
    linarith
  have hk19 : ((2 ^ 19 : Nat) : ℤ) = (524288 : ℤ) := by norm_num
  have hes19 : Real.exp s ≤ (2 : ℝ) ^ (-((2 ^ 19 : Nat) : ℤ)) := by
    have h2 := exp_neg_le_pow2 (x := -s) (k := 2 ^ 19) (by
      have : ((2 ^ 19 : Nat) : ℝ) ≤ -s := by push_cast; linarith
      exact this)
    simpa using h2
  have het19 : Real.exp t ≤ (2 : ℝ) ^ (-((2 ^ 20 : Nat) : ℤ)) := by
    have h2 := exp_neg_le_pow2 (x := -t) (k := 2 ^ 20) (by
      have : ((2 ^ 20 : Nat) : ℝ) ≤ -t := by push_cast; linarith
      exact this)
    simpa using h2
  have hes140 : Real.exp s ≤ (2 : ℝ) ^ (-140 : ℤ) := by
    have h3 : (2 : ℝ) ^ (-((2 ^ 19 : Nat) : ℤ)) ≤ (2 : ℝ) ^ (-140 : ℤ) := by
      apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
      have h4 : (140 : ℤ) ≤ ((2 ^ 19 : Nat) : ℤ) := by rw [hk19]; norm_num
      linarith
    linarith
  have het140 : Real.exp t ≤ (2 : ℝ) ^ (-140 : ℤ) := by
    have h3 : (2 : ℝ) ^ (-((2 ^ 20 : Nat) : ℤ)) ≤ (2 : ℝ) ^ (-140 : ℤ) := by
      apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
      have h4 : (140 : ℤ) ≤ ((2 ^ 20 : Nat) : ℤ) := by norm_num
      linarith
    linarith
  have hcoef : (4 * uR + 2 * uR * |s|) ≤ (2 : ℝ) ^ (108 : ℤ) := by
    have hterm : 2 * uR * |s| ≤ (2 : ℝ) ^ (107 : ℤ) := by
      have hm := mul_le_mul_of_nonneg_left hsc (by have := uR_pos; linarith : (0 : ℝ) ≤ 2 * uR)
      have he : 2 * uR * (2 : ℝ) ^ 130 = (2 : ℝ) ^ (107 : ℤ) := by unfold uR; norm_num
      linarith [hm, he]
    have h4u : 4 * uR ≤ (2 : ℝ) ^ (107 : ℤ) := by unfold uR; norm_num
    have hsum : (2 : ℝ) ^ (107 : ℤ) + (2 : ℝ) ^ (107 : ℤ) = (2 : ℝ) ^ (108 : ℤ) := by norm_num
    linarith
  have hprod : (4 * uR + 2 * uR * |s|) * Real.exp s ≤ (2 : ℝ) ^ (-140 : ℤ) := by
    have hc0 : (0 : ℝ) ≤ 4 * uR + 2 * uR * |s| := by
      have h := uR_pos; have h' := abs_nonneg s; nlinarith
    have h1 : (4 * uR + 2 * uR * |s|) * Real.exp s ≤
        (2 : ℝ) ^ (108 : ℤ) * (2 : ℝ) ^ (-((2 ^ 19 : Nat) : ℤ)) :=
      mul_le_mul hcoef hes19 (Real.exp_pos s).le (by positivity)
    have h2 : (2 : ℝ) ^ (108 : ℤ) * (2 : ℝ) ^ (-((2 ^ 19 : Nat) : ℤ)) ≤
        (2 : ℝ) ^ (-140 : ℤ) := by
      rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
      have h4 : (108 : ℤ) + -((2 ^ 19 : Nat) : ℤ) ≤ -140 := by rw [hk19]; norm_num
      linarith
    linarith
  have hstep1 : |ns - Real.exp t| ≤ |ns - Real.exp s| + |Real.exp s - Real.exp t| := by
    have h := abs_add_le (ns - Real.exp s) (Real.exp s - Real.exp t)
    rwa [show ns - Real.exp s + (Real.exp s - Real.exp t) = ns - Real.exp t by ring] at h
  have hstep2 : |Real.exp s - Real.exp t| ≤ Real.exp s + Real.exp t := by
    rw [abs_le]
    constructor <;> linarith [Real.exp_pos s, Real.exp_pos t]
  have hRHS : (0 : ℝ) ≤ (4 * uR * |t| + 11 * uR) * Real.exp t := by
    have h1 : (0 : ℝ) ≤ 4 * uR * |t| + 11 * uR := by
      have h := uR_pos; have h' := abs_nonneg t; nlinarith
    exact mul_nonneg h1 (Real.exp_pos t).le
  have h4e : (2 : ℝ) ^ (-140 : ℤ) * 4 ≤ (2 : ℝ) ^ (-137 : ℤ) := by norm_num
  linarith [hstep1, hstep2, hrule, hprod, hes140, het140, hRHS, h4e]

/-- The bound for one lane: `s = fl(v_i − m)` is either `−∞` (a masked lane, or an overflow), or a
finite rounding of `t = v_i − m ≤ 0`; then the platform's `exp` is within the stated error. -/
lemma lane_bound {P : Platform} (hexp : Device.ExpF32 P)
    (hninf : Device.ExpF32NegInf P) {mv x rsm num : F32}
    (hmv : mv.isFinite = true) (hxle : x.toRat ≤ mv.toRat)
    (hxok : x.isFinite = true ∨ x = FloatV.negInf)
    (hrsm : FloatV.NaNEq (FloatV.sub x mv) rsm)
    (hnum : P.math .exp .binary32 rsm num) :
    num.isFinite = true ∧
      |num.toReal - ER mv x| ≤
        (4 * uR * |tR mv x| + 11 * uR) * ER mv x + (2 : ℝ) ^ (-137 : ℤ) := by
  have hninf' : ∀ r : F32, P.math .exp .binary32 (FloatV.negInf : F32) r →
      r = FloatV.ofPat 0 := by
    intro r hr
    exact hninf r hr
  have hpat0 : (0 : Nat) < 2 ^ Format.binary32.width := by positivity
  have hzero_fin : (FloatV.ofPat 0 : F32).isFinite = true := by
    rw [FloatV.isFinite_ofPat hpat0]
    exact Format.isFinite_zero b32_reg
  have hzero_rat : (FloatV.ofPat 0 : F32).toRat = 0 := by
    rw [FloatV.toRat_ofPat hpat0]
    exact Format.toRat_of_isZero Format.isZero_zero
  have hnegInf_nan : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN b32_ieee
  have hnegInf_inf : (FloatV.negInf : F32).isInf = true := FloatV.negInf_isInf b32_ieee
  rcases hxok with hxfin | hxneg
  · by_cases hsubinf : FloatV.sub x mv = FloatV.negInf
    · have hnan : (FloatV.sub x mv).isNaN = false := by rw [hsubinf]; exact hnegInf_nan
      have hrsm_eq : rsm = FloatV.negInf := by
        have h := FloatV.NaNEq.eq hrsm hnan
        rw [hsubinf] at h
        exact h
      have hnum0 : num = FloatV.ofPat 0 := by
        have h := hnum
        rw [hrsm_eq] at h
        exact hninf' num h
      subst hnum0
      have hxne : x ≠ FloatV.negInf := finite_ne_negInf hxfin
      have hgap2 : x.toRat - mv.toRat < -(137 : ℚ) := sub_negInf_gap hxfin hmv hxle hsubinf
      have ht137 : tR mv x ≤ -(137 : ℝ) := by
        have h1 : ((x.toRat - mv.toRat : ℚ) : ℝ) < -(137 : ℝ) := by exact_mod_cast hgap2
        have h2 : ((x.toRat - mv.toRat : ℚ) : ℝ) = tR mv x := by
          rw [Rat.cast_sub]; simp only [tR, mR]
        linarith
      have hER : ER mv x ≤ (2 : ℝ) ^ (-137 : ℤ) := by
        rw [ER_eq_exp hxne]
        have h := exp_neg_le_pow2 (x := -tR mv x) (k := 137) (by push_cast; linarith)
        simpa using h
      refine ⟨hzero_fin, ?_⟩
      unfold FloatV.toReal
      rw [hzero_rat]
      have hERnn : (0 : ℝ) ≤ ER mv x := ER_nonneg mv x
      have hcoef0 : (0 : ℝ) ≤ (4 * uR * |tR mv x| + 11 * uR) * ER mv x := by
        refine mul_nonneg ?_ hERnn
        have h := uR_pos; have h' := abs_nonneg (tR mv x)
        nlinarith [mul_nonneg (le_of_lt uR_pos) h']
      rw [Rat.cast_zero, zero_sub, abs_neg, abs_of_nonneg hERnn]
      linarith
    · have hsubfin : (FloatV.sub x mv).isFinite = true := by
        rcases (b32_sub_facts hxfin hmv hxle) with ⟨h, -⟩ | h
        · exact h
        · exact absurd h hsubinf
      have hnan : (FloatV.sub x mv).isNaN = false := (FloatV.isFinite_iff.mp hsubfin).1
      have hrsm_eq : rsm = FloatV.sub x mv := FloatV.NaNEq.eq hrsm hnan
      have hxne : x ≠ FloatV.negInf := finite_ne_negInf hxfin
      have hrsm_fin : rsm.isFinite = true := by rw [hrsm_eq]; exact hsubfin
      have hqle : x.toRat - mv.toRat ≤ 0 := by linarith
      have hsubpat := b32_sub_pattern (x := x) (y := mv) hxfin hmv
      have hsnp : rsm.toReal ≤ 0 := by
        have hpat : rsm = FloatV.ofPat (Format.binary32.roundRat .rn false
            (Format.binary32.sign x.pat && ! Format.binary32.sign mv.pat)
            (x.toRat - mv.toRat)) := by
          rw [hrsm_eq]; exact hsubpat
        unfold FloatV.toReal
        rw [hpat]
        have hlt := roundRat_rn_lt (f := Format.binary32) b32_reg b32_ieee
          (zs := Format.binary32.sign x.pat && ! Format.binary32.sign mv.pat)
          (q := x.toRat - mv.toRat)
        rw [FloatV.toRat_ofPat hlt]
        have h := b32_roundRat_rn_toRat_nonpos
          (zs := Format.binary32.sign x.pat && ! Format.binary32.sign mv.pat) hqle
        exact_mod_cast h
      obtain ⟨g, hg, hacc⟩ := hexp |rsm.toReal| (abs_nonneg _)
      have hgr : g = Real.exp := by
        simpa only [MathFn.ref, Option.some.injEq] using hg.symm
      rw [hgr] at hacc
      have hdom : rsm.toReal ∈ Set.Icc (-|rsm.toReal|) |rsm.toReal| :=
        ⟨by linarith [neg_abs_le rsm.toReal], le_abs_self rsm.toReal⟩
      have hnat : rsm.toReal ∈ MathFn.natDom .exp := by
        simp only [MathFn.natDom]
        exact Set.mem_univ _
      have hepos : (0 : ℝ) < Real.exp rsm.toReal := Real.exp_pos _
      have heabs : |Real.exp rsm.toReal| = Real.exp rsm.toReal := abs_of_pos hepos
      have he1 : Real.exp rsm.toReal ≤ 1 := by
        rw [← Real.exp_zero]
        exact Real.exp_le_exp.mpr hsnp
      have hsc_max : |rsm.toReal| ≤ (2 : ℝ) ^ 128 := by
        have h1 : |rsm.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
          have h := FloatV.abs_toRat_le_maxVal (f := Format.binary32) b32_reg hrsm_fin
          have h2 : ((|rsm.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
            exact_mod_cast h
          unfold FloatV.toReal
          rw [Rat.cast_abs] at h2
          exact h2
        have h2 : ((Format.binary32.maxVal : ℚ) : ℝ) < (2 : ℝ) ^ 128 := by
          have h : ((Format.binary32.maxVal : ℚ) : ℝ) < (((2 : ℚ) ^ 128 : ℚ) : ℝ) := by
            exact_mod_cast maxVal_lt_two_pow
          simpa using h
        linarith
      have hfac : (2 : ℝ) ^ 127 = (2 : ℝ) ^ 104 * (2 : ℝ) ^ 23 := by rw [← pow_add]
      have hbig : (2 : ℝ) ^ 23 ≤ ((2 : ℝ) ^ 24 - 1) := by norm_num
      have hmaxlb : (2 : ℝ) ^ 127 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
        rw [b32_maxVal_cast, hfac]
        have h0 : (0 : ℝ) < (2 : ℝ) ^ 104 := by norm_num
        calc (2 : ℝ) ^ 104 * (2 : ℝ) ^ 23 ≤ (2 : ℝ) ^ 104 * ((2 : ℝ) ^ 24 - 1) :=
              mul_le_mul_of_nonneg_left hbig h0.le
          _ = ((2 : ℝ) ^ 24 - 1) * (2 : ℝ) ^ 104 := by ring
      have h2a : (2 : ℝ) ^ (-23 : ℤ) * |rsm.toReal| ≤ (2 : ℝ) ^ 105 := by
        have hm := mul_le_mul_of_nonneg_left hsc_max (by norm_num : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ))
        have he : (2 : ℝ) ^ (-23 : ℤ) * (2 : ℝ) ^ 128 = (2 : ℝ) ^ 105 := by norm_num
        linarith [hm, he]
      have h2b : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
      have hroom : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |rsm.toReal|)) *
          |Real.exp rsm.toReal| + (2 : ℝ) ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
        have hrel : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |rsm.toReal| ≤
            (2 : ℝ) ^ 105 + 1 := by linarith [h2a, h2b]
        have hc106 : (2 : ℝ) ^ 105 + 1 + 1 ≤ (2 : ℝ) ^ 106 := by norm_num
        have h1 : 1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |rsm.toReal|) ≤
            (2 : ℝ) ^ 106 := by linarith [hrel, hc106]
        have h2 : |Real.exp rsm.toReal| ≤ 1 := by rw [heabs]; exact he1
        have h4 : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |rsm.toReal|)) *
            |Real.exp rsm.toReal| ≤ (2 : ℝ) ^ 106 := by
          calc _ ≤ (2 : ℝ) ^ 106 * 1 := mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
            _ = (2 : ℝ) ^ 106 := by ring
        have h5 : (2 : ℝ) ^ 106 + (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ 127 := by norm_num
        linarith [h4, h5, hmaxlb]
      have hacc' := hacc rsm num hnum hrsm_fin hdom hnat hroom
      have hrule : |num.toReal - Real.exp rsm.toReal| ≤
          (4 * uR + 2 * uR * |rsm.toReal|) * Real.exp rsm.toReal + (2 : ℝ) ^ (-140 : ℤ) := by
        have h1 := hacc'.2
        rw [heabs] at h1
        have h2 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |rsm.toReal| =
            4 * uR + 2 * uR * |rsm.toReal| := by
          rw [e22, e23]
        rw [h2] at h1
        exact h1
      have hst : |rsm.toReal - tR mv x| ≤ uR * |rsm.toReal| + etaR := by
        have hgap := b32_sub_facts hxfin hmv hxle
        rcases hgap with ⟨-, h⟩ | h
        · have h1 : ((|(FloatV.sub x mv).toRat - (x.toRat - mv.toRat)| : ℚ) : ℝ) ≤
              (((|(FloatV.sub x mv).toRat| * Format.binary32.u + Format.binary32.eta : ℚ)) : ℝ) := by
            exact_mod_cast h
          simp only [Rat.cast_abs, Rat.cast_sub, Rat.cast_mul, Rat.cast_add, b32_u_cast,
            b32_eta_cast] at h1
          have hsubR : ((FloatV.sub x mv).toRat : ℝ) = rsm.toReal := by
            rw [← hrsm_eq]
            rfl
          have htR : ((x.toRat : ℝ) - (mv.toRat : ℝ)) = tR mv x := by
            simp only [tR, mR]
          rw [hsubR, htR] at h1
          simpa only [mul_comm] using h1
        · exact absurd h hsubinf
      have hsc : |rsm.toReal| ≤ (2 : ℝ) ^ 130 := by
        have h1 : |rsm.toReal| ≤ (2 : ℝ) ^ 128 := hsc_max
        have h3 : (2 : ℝ) ^ 128 ≤ (2 : ℝ) ^ 130 := by norm_num
        linarith
      by_cases hsmall : |tR mv x| ≤ (2 : ℝ) ^ 20
      · have h := num_lane_small (t := tR mv x) (s := rsm.toReal) (ns := num.toReal)
          hsmall hst hrule
        refine ⟨hacc'.1, ?_⟩
        rw [ER_eq_exp hxne]
        exact h
      · have htle : tR mv x ≤ 0 := by
          have h1 : (x.toRat : ℝ) ≤ (mv.toRat : ℝ) := by exact_mod_cast hxle
          simp only [tR, mR]
          linarith
        have ht : tR mv x ≤ -((2 : ℝ) ^ 20) := by
          have hlt : (2 : ℝ) ^ 20 < |tR mv x| := lt_of_not_ge hsmall
          have habs : |tR mv x| = -tR mv x := abs_of_nonpos htle
          rw [habs] at hlt
          linarith
        have h := num_lane_big (t := tR mv x) (s := rsm.toReal) (ns := num.toReal)
          ht hst hsc hrule
        refine ⟨hacc'.1, ?_⟩
        rw [ER_eq_exp hxne]
        exact h
  · have hsub : FloatV.sub x mv = FloatV.negInf := by
      rw [hxneg]
      exact FloatV.sub_negInf_left (f := Format.binary32) b32_ieee hmv
    have hnan : (FloatV.sub x mv).isNaN = false := by rw [hsub]; exact hnegInf_nan
    have hrsm_eq : rsm = FloatV.negInf := by
      have h := FloatV.NaNEq.eq hrsm hnan
      rw [hsub] at h
      exact h
    have hnum0 : num = FloatV.ofPat 0 := by
      have h := hnum
      rw [hrsm_eq] at h
      exact hninf' num h
    subst hnum0
    refine ⟨hzero_fin, ?_⟩
    unfold FloatV.toReal
    rw [hzero_rat, hxneg, ER_eq_zero rfl]
    have hprod0 : (4 * uR * |tR mv FloatV.negInf| + 11 * uR) * (0 : ℝ) = 0 := by ring
    rw [hprod0]
    simp only [Rat.cast_zero, zero_sub, zero_add]
    norm_num

end Claims.Softmax
