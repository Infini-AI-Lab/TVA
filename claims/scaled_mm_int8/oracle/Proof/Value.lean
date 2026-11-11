import Claim
import TVA.Proof

/-!
# The value bound of `scaled_mm_int8`

The `float32` chain of the epilogue: the conversion of the exact integer accumulator, the two
scale products, the optional bias, and the rounding to the output format; and its relaxation to
`tol` and `outTol`.
-/

namespace Claims.ScaledMmInt8

open TVA


/-- `e3` is at most the linear bound in `|sa·sb·S|`. -/
theorem e3_le (sa sb S : ℚ) :
    e3 sa sb S ≤ (3 * u32 + 3 * u32 ^ 2 + u32 ^ 3) * |sa * sb * S| +
      (1 + u32) ^ 2 * |sa * sb| * eta32 + (1 + u32) * |sb| * eta32 + eta32 := by
  simp only [e3, e2, e1, abs_mul]
  exact le_of_eq (by ring)

/-- `u32 = 2^-24`. -/
theorem u32num : u32 = 1 / 16777216 := by
  rw [u32, Format.binary32_u]
  norm_num

/-- `u32 ≥ 0`. -/
theorem u32_nonneg : (0:ℚ) ≤ u32 := by unfold u32; exact Format.u_nonneg

/-- `eta32 ≥ 0`. -/
theorem eta32_nonneg : (0:ℚ) ≤ eta32 := by unfold eta32; exact Format.eta_nonneg

/-- `e3 ≥ 0`. -/
theorem e3_nonneg (sa sb S : ℚ) : 0 ≤ e3 sa sb S := by
  unfold e3 e2 e1
  have hu := u32_nonneg
  have he := eta32_nonneg
  positivity

/-- Relaxes the error of the chain to the closed form with the `1.000001` slack. -/
theorem relax_le (sa sb S β F : ℚ)
    (hF : F ≤ e3 sa sb S + 3 * u32 * (|sa * sb * S| + e3 sa sb S + |β|) + 4 * eta32) :
    F ≤ 6.000001 * u32 * |sa * sb * S| + 3 * u32 * |β| + 1.000001 * |sa * sb| * eta32 +
      1.000001 * |sb| * eta32 + 5.000001 * eta32 := by
  have hE := e3_le sa sb S
  have hu : u32 = 1 / 16777216 := u32num
  rw [hu] at hF hE ⊢
  have he0 : 0 ≤ eta32 := Format.eta_nonneg
  have hA0 : 0 ≤ |sa * sb * S| := abs_nonneg _
  have hB0 : 0 ≤ |sa * sb| := abs_nonneg _
  have hb0 : 0 ≤ |sb| := abs_nonneg _
  have hbeta0 : 0 ≤ |β| := abs_nonneg _
  nlinarith

/-- The relaxed bound is at most `tol` when `|sb|` is within `float32`. -/
theorem relax_tol (sa sb S β F : ℚ) (hsb : |sb| ≤ max32)
    (hF : F ≤ 6.000001 * u32 * |sa * sb * S| + 3 * u32 * |β| + 1.000001 * |sa * sb| * eta32 +
      1.000001 * |sb| * eta32 + 5.000001 * eta32) :
    F ≤ tol sa sb S β := by
  have hu : u32 = 1 / 16777216 := u32num
  have heta : eta32 = (2:ℚ) ^ (-150 : ℤ) := Format.binary32_eta
  have hmax : max32 = ((2:ℚ) ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have h21 : (2:ℚ) ^ (-21 : ℤ) = 1 / 2097152 := by norm_num
  unfold tol mag
  simp only [abs_mul] at hF ⊢
  rw [hu] at hF
  rw [h21]
  have hS0 : (0:ℚ) ≤ |S| := abs_nonneg S
  have hB0 : (0:ℚ) ≤ |sa| * |sb| := by positivity
  have he0 : (0:ℚ) ≤ eta32 := Format.eta_nonneg
  have hbe0 : (0:ℚ) ≤ |β| := abs_nonneg β
  have hsum : 6.000001 * (1/16777216) * |S| + 1.000001 * eta32 ≤ (1/2097152) * (|S| + 1) := by
    have hc1 : (6.000001 : ℚ) * (1/16777216) ≤ 1/2097152 := by norm_num
    have hc3 : (1.000001 : ℚ) * eta32 ≤ 1/2097152 := by rw [heta]; norm_num
    have e1 : 6.000001 * (1/16777216) * |S| ≤ (1/2097152) * |S| :=
      mul_le_mul_of_nonneg_right hc1 hS0
    linarith
  have h13 : 6.000001 * (1/16777216) * (|sa| * |sb| * |S|) + 1.000001 * (|sa| * |sb|) * eta32
      ≤ (1/2097152) * (|sa| * |sb| * (|S| + 1)) := by
    calc 6.000001 * (1/16777216) * (|sa| * |sb| * |S|) + 1.000001 * (|sa| * |sb|) * eta32
        = (|sa| * |sb|) * (6.000001 * (1/16777216) * |S| + 1.000001 * eta32) := by ring
      _ ≤ (|sa| * |sb|) * ((1/2097152) * (|S| + 1)) := mul_le_mul_of_nonneg_left hsum hB0
      _ = (1/2097152) * (|sa| * |sb| * (|S| + 1)) := by ring
  have h2 : 3 * (1/16777216) * |β| ≤ (1/2097152) * |β| := by
    have hc : (3 : ℚ) * (1/16777216) ≤ 1/2097152 := by norm_num
    exact mul_le_mul_of_nonneg_right hc hbe0
  have hb128 : |sb| ≤ ((2:ℚ)^24 - 1) * 2^104 := by rw [← hmax]; exact hsb
  have h4 : 1.000001 * |sb| * eta32 + 5.000001 * eta32 ≤ 1/2097152 := by
    have hb1 : |sb| * eta32 ≤ 1/4194304 := by
      have h1 : |sb| * eta32 ≤ (((2:ℚ)^24 - 1) * 2^104) * eta32 :=
        mul_le_mul_of_nonneg_right hb128 he0
      have h2' : (((2:ℚ)^24 - 1) * 2^104) * eta32 ≤ 1/4194304 := by
        rw [heta]; norm_num
      exact h1.trans h2'
    have h5 : (5.000001:ℚ) * eta32 ≤ 1/8388608 := by rw [heta]; norm_num
    have h7 : (1.000001:ℚ) * (|sb| * eta32) ≤ 1.000001 * (1/4194304) :=
      mul_le_mul_of_nonneg_left hb1 (by norm_num)
    nlinarith
  linarith

/-- `mag ≥ 0`. -/
theorem mag_nonneg (sa sb S β : ℚ) : 0 ≤ mag sa sb S β := by
  unfold mag
  positivity

/-- `|R| ≤ mag`. -/
theorem abs_R_le_mag (sa sb S β : ℚ) : |sa * sb * S + β| ≤ mag sa sb S β := by
  unfold mag
  have e1 : |sa * sb * S| = |sa * sb| * |S| := abs_mul _ _
  have e2 : |sa * sb * S + β| ≤ |sa * sb * S| + |β| := abs_add_le _ _
  have h3 : |sa * sb| * |S| ≤ |sa * sb| * (|S| + 1) := by
    have := abs_nonneg (sa * sb)
    nlinarith
  linarith

/-- A value within `tol` of `R` is within `mag + tol` in magnitude. -/
theorem out_le (sa sb S β F y : ℚ) (hF : F ≤ tol sa sb S β)
    (hy : |y - (sa * sb * S + β)| ≤ F) :
    |y| ≤ mag sa sb S β + tol sa sb S β := by
  have h2 : |sa * sb * S + β| ≤ mag sa sb S β := abs_R_le_mag sa sb S β
  have h1 : |y| - |sa * sb * S + β| ≤ |y - (sa * sb * S + β)| := abs_sub_abs_le_abs_sub _ _
  linarith

/-- The output rounding's error, with the format's constants abstracted. -/
theorem out_concl_aux {g : Format} (hg : g.Regular) {c : FloatV g} {x3 : F32}
    {sa sb S β u0 e0 : ℚ} (hu : g.u = u0) (he : g.eta ≤ e0)
    (hnum1 : (2:ℚ)^(-21:ℤ) + (2:ℚ)^(-21:ℤ) * u0 + u0 ≤ 2 * u0)
    (hnum2 : (2:ℚ)^(-21:ℤ) * u0 + e0 + (2:ℚ)^(-21:ℤ) ≤ (2:ℚ)^(-20:ℤ))
    (hfin : x3.isFinite = true)
    (hx3 : |x3.toRat - (sa * sb * S + β)| ≤ tol sa sb S β)
    (hne : FloatV.NaNEq (FloatV.cvt x3) c)
    (hmax : |sa * sb * S + β| + tol sa sb S β ≤ g.maxVal) :
    c.isFinite = true ∧
      |c.toRat - (sa * sb * S + β)| ≤ 2 * u0 * mag sa sb S β + (2:ℚ)^(-20:ℤ) := by
  have hu0 : (0:ℚ) ≤ u0 := by rw [← hu]; exact Format.u_nonneg
  have hRmag : |sa * sb * S + β| ≤ mag sa sb S β := abs_R_le_mag sa sb S β
  have hmag0 : (0:ℚ) ≤ mag sa sb S β := mag_nonneg sa sb S β
  have hx3abs : |x3.toRat| ≤ g.maxVal := by
    have h1 : |x3.toRat| - |sa * sb * S + β| ≤ |x3.toRat - (sa * sb * S + β)| :=
      abs_sub_abs_le_abs_sub _ _
    linarith
  have hcvt := FloatV.cvt_err (g := g) hg hfin hx3abs
  obtain ⟨hcvtfin, hcvterr⟩ := hcvt
  have hnan : (FloatV.cvt x3 : FloatV g).isNaN = false := (FloatV.isFinite_iff.mp hcvtfin).1
  have hceq : c = FloatV.cvt x3 := FloatV.NaNEq.eq hne hnan
  have hcfin : c.isFinite = true := by rw [hceq]; exact hcvtfin
  refine ⟨hcfin, ?_⟩
  rw [hceq]
  have hcvterr' : |(FloatV.cvt x3 : FloatV g).toRat - x3.toRat| ≤ |x3.toRat| * u0 + e0 := by
    rw [hu] at hcvterr
    linarith
  have hx3R : |x3.toRat| ≤ |sa * sb * S + β| + tol sa sb S β := by
    have h1 : |x3.toRat| - |sa * sb * S + β| ≤ |x3.toRat - (sa * sb * S + β)| :=
      abs_sub_abs_le_abs_sub _ _
    linarith
  have hstep : |(FloatV.cvt x3 : FloatV g).toRat - (sa * sb * S + β)| ≤
      (|x3.toRat| * u0 + e0) + tol sa sb S β := by
    have h1 := abs_add_le ((FloatV.cvt x3 : FloatV g).toRat - x3.toRat)
      (x3.toRat - (sa * sb * S + β))
    have e : (FloatV.cvt x3 : FloatV g).toRat - (sa * sb * S + β) =
        ((FloatV.cvt x3 : FloatV g).toRat - x3.toRat) + (x3.toRat - (sa * sb * S + β)) := by ring
    rw [e]
    linarith
  have e1 : |x3.toRat| * u0 ≤ (mag sa sb S β + tol sa sb S β) * u0 := by
    refine mul_le_mul_of_nonneg_right ?_ hu0
    linarith
  have hb : |(FloatV.cvt x3 : FloatV g).toRat - (sa * sb * S + β)| ≤
      mag sa sb S β * ((2:ℚ)^(-21:ℤ) + (2:ℚ)^(-21:ℤ) * u0 + u0) +
        ((2:ℚ)^(-21:ℤ) * u0 + e0 + (2:ℚ)^(-21:ℤ)) := by
    refine hstep.trans ?_
    have h2 : (|x3.toRat| * u0 + e0) + tol sa sb S β ≤
        ((mag sa sb S β + tol sa sb S β) * u0 + e0) + tol sa sb S β := by linarith [e1]
    refine h2.trans (le_of_eq ?_)
    unfold tol
    ring
  have hc1 : mag sa sb S β * ((2:ℚ)^(-21:ℤ) + (2:ℚ)^(-21:ℤ) * u0 + u0) ≤
      mag sa sb S β * (2 * u0) := by
    exact mul_le_mul_of_nonneg_left (by linarith [hnum1]) hmag0
  have hc2 : (2:ℚ)^(-21:ℤ) * u0 + e0 + (2:ℚ)^(-21:ℤ) ≤ (2:ℚ)^(-20:ℤ) := hnum2
  linarith

/-- The final conversion to `bfloat16` or `float16` gives the `OutSpec` conclusion. -/
theorem out_conclusion (P : Platform) {od : DType} (hod : od = .bf16 ∨ od = .f16)
    {sa sb S β : ℚ} {x3 : F32} {c : od.Val}
    (hfin : x3.isFinite = true)
    (hx3 : |x3.toRat - (sa * sb * S + β)| ≤ tol sa sb S β)
    (hc : ConvRel P .f32 od x3 c) :
    OutIs od x3 c ∧
      (|sa * sb * S + β| + tol sa sb S β ≤ outMax od →
        ∃ q : ℚ, od.val? c = some q ∧ |q - (sa * sb * S + β)| ≤ outTol od sa sb S β) := by
  have h1 : ¬ (Format.binary32 = Format.bfloat16) := by decide
  have h2 : ¬ (Format.binary32 = Format.binary16) := by decide
  have h3 : ¬ Format.binary32.F8 := by
    unfold Format.F8
    rintro (h | h) <;> exact absurd h (by decide)
  have h4 : ¬ Format.bfloat16.F8 := by
    unfold Format.F8
    rintro (h | h) <;> exact absurd h (by decide)
  have h5 : ¬ Format.binary16.F8 := by
    unfold Format.F8
    rintro (h | h) <;> exact absurd h (by decide)
  rcases hod with rfl | rfl
  · have hne : FloatV.NaNEq (FloatV.cvt x3 : BF16) c := by
      have h := hc
      unfold ConvRel at h
      simp only [ofFloatRel, FloatRel] at h
      simp only [h1, h3, h4, ite_false, false_and, and_false, or_false] at h
      exact h.2.2
    refine ⟨hne, ?_⟩
    intro hmax
    have hu : Format.bfloat16.u = 1/256 := by rw [Format.bfloat16_u]; norm_num
    have he : Format.bfloat16.eta ≤ 1/33554432 := by rw [Format.bfloat16_eta]; norm_num
    obtain ⟨hcfin, hbd⟩ := out_concl_aux (g := Format.bfloat16) Format.bfloat16_regular
      hu he (by norm_num) (by norm_num) hfin hx3 hne hmax
    refine ⟨c.toRat, ?_, ?_⟩
    · show DType.val? .bf16 c = some c.toRat
      simp only [DType.val?, hcfin, ite_true]
    · show |c.toRat - (sa * sb * S + β)| ≤ outTol .bf16 sa sb S β
      rw [show outTol .bf16 sa sb S β =
        2 * Format.bfloat16.u * mag sa sb S β + (2:ℚ)^(-20:ℤ) from rfl, hu]
      exact hbd
  · have hne : FloatV.NaNEq (FloatV.cvt x3 : F16) c := by
      have h := hc
      unfold ConvRel at h
      simp only [ofFloatRel, FloatRel] at h
      simp only [h2, h3, h5, ite_false, false_and, and_false, or_false] at h
      exact h.2.2
    refine ⟨hne, ?_⟩
    intro hmax
    have hu : Format.binary16.u = 1/2048 := by rw [Format.binary16_u]; norm_num
    have he : Format.binary16.eta ≤ 1/33554432 := by rw [Format.binary16_eta]; norm_num
    obtain ⟨hcfin, hbd⟩ := out_concl_aux (g := Format.binary16) Format.binary16_regular
      hu he (by norm_num) (by norm_num) hfin hx3 hne hmax
    refine ⟨c.toRat, ?_, ?_⟩
    · show DType.val? .f16 c = some c.toRat
      simp only [DType.val?, hcfin, ite_true]
    · show |c.toRat - (sa * sb * S + β)| ≤ outTol .f16 sa sb S β
      rw [show outTol .f16 sa sb S β =
        2 * Format.binary16.u * mag sa sb S β + (2:ℚ)^(-20:ℤ) from rfl, hu]
      exact hbd

/-- An integer within `2^31` is below `2^53`. -/
theorem itof_bound (S : ℤ) (hS : |((S : ℚ))| ≤ 2 ^ 31) :
    |S| < 2 ^ 53 := by
  have hS53 : |((S:ℚ))| < (2:ℚ)^53 := by
    have h2 : (2:ℚ)^31 < 2^53 := by norm_num
    linarith [abs_nonneg ((S:ℚ))]
  have h2 : ((|S| : ℤ) : ℚ) = |((S:ℚ))| := by rw [Int.cast_abs]
  have h3 : ((|S| : ℤ) : ℚ) < ((2:ℤ)^53 : ℚ) := by
    rw [h2]; push_cast; linarith
  have h4 : (|S| : ℤ) < (2:ℤ)^53 := by exact_mod_cast h3
  simpa using h4

/-- An integer within `2^31` is within `float32`'s range. -/
theorem S_le_max (S : ℤ) (hS : |((S : ℚ))| ≤ 2 ^ 31) :
    |((S : ℚ))| ≤ Format.binary32.maxVal := by
  have h1 : ((2:ℚ)^31) ≤ ((2:ℚ)^24 - 1) * 2^104 := by norm_num
  rw [Format.binary32_maxVal]
  linarith [abs_nonneg ((S:ℚ))]

/-- The `int32` accumulator converts to `float32` as the nearest value, finite, within `e1`. -/
theorem x0_prep {S : ℤ} {x0 : F32} (hS : |((S : ℚ))| ≤ 2 ^ 31)
    (h0 : tl.ItofRel .binary32 S x0) :
    x0 = FloatV.ofRat ((S : ℚ)) ∧ x0.isFinite = true ∧
      |x0.toRat - ((S : ℚ))| ≤ e1 ((S : ℚ)) := by
  have hx0 : x0 = FloatV.ofRat ((S : ℚ)) := by
    have h := (tl.itofRel_binary32 (itof_bound S hS)).mp h0
    simpa using h
  obtain ⟨hfin, herr⟩ := FloatV.ofRat_err (f := Format.binary32) Format.binary32_regular
    (q := ((S : ℚ))) (S_le_max S hS)
  refine ⟨hx0, ?_, ?_⟩
  · rw [hx0]; exact hfin
  · rw [hx0]
    simpa [e1, u32, eta32] using herr

/-- Without a bias, the `float32` chain stays within `tol` of `R`. -/
theorem tol_of_nobias {S : ℤ} {sa sb x0 x1 x2 : F32}
    (hS : |((S : ℚ))| ≤ 2 ^ 31)
    (h0 : tl.ItofRel .binary32 S x0)
    (h1 : FloatV.NaNEq (FloatV.mul sa x0) x1)
    (h2 : FloatV.NaNEq (FloatV.mul sb x1) x2)
    (hsa : sa.isFinite = true) (hsb : sb.isFinite = true)
    (hroom : Room32 sa.toRat sb.toRat ((S : ℚ)) 0) :
    x2.isFinite = true ∧
      |x2.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + 0)| ≤ tol sa.toRat sb.toRat ((S : ℚ)) 0 := by
  have hu0 : (0:ℚ) ≤ u32 := by unfold u32; exact Format.u_nonneg
  have he0 : (0:ℚ) ≤ eta32 := by unfold eta32; exact Format.eta_nonneg
  obtain ⟨hx0, hx0fin, hx0err⟩ := x0_prep hS h0
  have hx0abs : |x0.toRat| ≤ |((S : ℚ))| + e1 ((S : ℚ)) := by
    have h := abs_sub_abs_le_abs_sub x0.toRat ((S : ℚ))
    linarith
  have hm1max : |sa.toRat * x0.toRat| ≤ Format.binary32.maxVal := by
    have hr := hroom.1
    rw [show max32 = Format.binary32.maxVal from rfl] at hr
    rw [abs_mul]
    nlinarith [abs_nonneg sa.toRat]
  obtain ⟨hm1fin, hm1err⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
    hsa hx0fin hm1max
  have hx1eq : x1 = FloatV.mul sa x0 := FloatV.NaNEq.eq h1 (FloatV.isFinite_iff.mp hm1fin).1
  have hx1fin : x1.isFinite = true := by rw [hx1eq]; exact hm1fin
  have hx1err : |x1.toRat - sa.toRat * x0.toRat| ≤ |sa.toRat * x0.toRat| * u32 + eta32 := by
    rw [hx1eq]
    simpa [u32, eta32] using hm1err
  have hx1e2 : |x1.toRat - sa.toRat * ((S : ℚ))| ≤ e2 sa.toRat ((S : ℚ)) := by
    have e : x1.toRat - sa.toRat * ((S : ℚ)) =
        (x1.toRat - sa.toRat * x0.toRat) + sa.toRat * (x0.toRat - ((S : ℚ))) := by ring
    rw [e]
    have t := abs_add_le (x1.toRat - sa.toRat * x0.toRat)
      (sa.toRat * (x0.toRat - ((S : ℚ))))
    have h2r : |sa.toRat * (x0.toRat - ((S : ℚ)))| ≤ |sa.toRat| * e1 ((S : ℚ)) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hx0err (abs_nonneg _)
    have h3r : |sa.toRat * x0.toRat| ≤ |sa.toRat * ((S : ℚ))| + |sa.toRat| * e1 ((S : ℚ)) := by
      have e2' : sa.toRat * x0.toRat = sa.toRat * ((S : ℚ)) + sa.toRat * (x0.toRat - ((S : ℚ))) := by
        ring
      rw [e2']
      have t2 := abs_add_le (sa.toRat * ((S : ℚ))) (sa.toRat * (x0.toRat - ((S : ℚ))))
      linarith
    have h4r : |sa.toRat * x0.toRat| * u32 ≤
        (|sa.toRat * ((S : ℚ))| + |sa.toRat| * e1 ((S : ℚ))) * u32 :=
      mul_le_mul_of_nonneg_right h3r hu0
    unfold e2
    linarith
  have hx1abs : |x1.toRat| ≤ |sa.toRat * ((S : ℚ))| + e2 sa.toRat ((S : ℚ)) := by
    have h := abs_sub_abs_le_abs_sub x1.toRat (sa.toRat * ((S : ℚ)))
    linarith
  have hm2max : |sb.toRat * x1.toRat| ≤ Format.binary32.maxVal := by
    have hr := hroom.2.1
    rw [show max32 = Format.binary32.maxVal from rfl] at hr
    rw [abs_mul]
    nlinarith [abs_nonneg sb.toRat]
  obtain ⟨hm2fin, hm2err⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
    hsb hx1fin hm2max
  have hx2eq : x2 = FloatV.mul sb x1 := FloatV.NaNEq.eq h2 (FloatV.isFinite_iff.mp hm2fin).1
  have hx2fin : x2.isFinite = true := by rw [hx2eq]; exact hm2fin
  have hx2err : |x2.toRat - sb.toRat * x1.toRat| ≤ |sb.toRat * x1.toRat| * u32 + eta32 := by
    rw [hx2eq]
    simpa [u32, eta32] using hm2err
  have hx2e3 : |x2.toRat - sa.toRat * sb.toRat * ((S : ℚ))| ≤
      e3 sa.toRat sb.toRat ((S : ℚ)) := by
    have e : x2.toRat - sa.toRat * sb.toRat * ((S : ℚ)) =
        (x2.toRat - sb.toRat * x1.toRat) +
          sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))) := by ring
    rw [e]
    have t := abs_add_le (x2.toRat - sb.toRat * x1.toRat)
      (sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))))
    have h2r : |sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ)))| ≤ |sb.toRat| * e2 sa.toRat ((S : ℚ)) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hx1e2 (abs_nonneg _)
    have h3r : |sb.toRat * x1.toRat| ≤ |sa.toRat * sb.toRat * ((S : ℚ))| +
        |sb.toRat| * e2 sa.toRat ((S : ℚ)) := by
      have eqt : sb.toRat * x1.toRat =
          sb.toRat * (sa.toRat * ((S : ℚ))) + sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))) := by
        ring
      rw [eqt]
      have t2 := abs_add_le (sb.toRat * (sa.toRat * ((S : ℚ))))
        (sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))))
      have e4 : |sb.toRat * (sa.toRat * ((S : ℚ)))| = |sa.toRat * sb.toRat * ((S : ℚ))| := by
        congr 1
        ring
      linarith
    have h4r : |sb.toRat * x1.toRat| * u32 ≤
        (|sa.toRat * sb.toRat * ((S : ℚ))| + |sb.toRat| * e2 sa.toRat ((S : ℚ))) * u32 :=
      mul_le_mul_of_nonneg_right h3r hu0
    unfold e3
    linarith
  refine ⟨hx2fin, ?_⟩
  have hE3 : |x2.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + 0)| ≤ e3 sa.toRat sb.toRat ((S : ℚ)) := by
    simpa using hx2e3
  have hFbig : |x2.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + 0)| ≤
      e3 sa.toRat sb.toRat ((S : ℚ)) +
        3 * u32 * (|sa.toRat * sb.toRat * ((S : ℚ))| + e3 sa.toRat sb.toRat ((S : ℚ)) + |(0:ℚ)|) +
        4 * eta32 := by
    have hnn : (0:ℚ) ≤ 3 * u32 *
        (|sa.toRat * sb.toRat * ((S : ℚ))| + e3 sa.toRat sb.toRat ((S : ℚ)) + |(0:ℚ)|) + 4 * eta32 := by
      have h1 : (0:ℚ) ≤ 3 * u32 := by linarith
      have h2 : (0:ℚ) ≤ |sa.toRat * sb.toRat * ((S : ℚ))| + e3 sa.toRat sb.toRat ((S : ℚ)) + |(0:ℚ)| := by
        have h1 := abs_nonneg (sa.toRat * sb.toRat * ((S : ℚ)))
        have h3 := e3_nonneg sa.toRat sb.toRat ((S : ℚ))
        have h4 := abs_nonneg (0:ℚ)
        linarith
      nlinarith
    linarith
  have hrel := relax_le sa.toRat sb.toRat ((S : ℚ)) 0 _ hFbig
  have hsbmax : |sb.toRat| ≤ max32 := by
    have h := FloatV.abs_toRat_le_maxVal (f := Format.binary32) Format.binary32_regular hsb
    simpa [max32] using h
  exact relax_tol sa.toRat sb.toRat ((S : ℚ)) 0 _ hsbmax hrel

/-- With a bias, the `float32` chain stays within `tol` of `R`. -/
theorem tol_of_bias {S : ℤ} {sa sb x0 x1 x2 x3 bv : F32} {β : ℚ}
    (hS : |((S : ℚ))| ≤ 2 ^ 31)
    (h0 : tl.ItofRel .binary32 S x0)
    (h1 : FloatV.NaNEq (FloatV.mul sa x0) x1)
    (h2 : FloatV.NaNEq (FloatV.mul sb x1) x2)
    (hbv : bv.isFinite = true) (hbvq : bv.toRat = β)
    (h3 : AddRel true x2 bv x3)
    (hsa : sa.isFinite = true) (hsb : sb.isFinite = true)
    (hroom : Room32 sa.toRat sb.toRat ((S : ℚ)) β) :
    x3.isFinite = true ∧
      |x3.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + β)| ≤ tol sa.toRat sb.toRat ((S : ℚ)) β := by
  have hu0 : (0:ℚ) ≤ u32 := by unfold u32; exact Format.u_nonneg
  have he0 : (0:ℚ) ≤ eta32 := by unfold eta32; exact Format.eta_nonneg
  obtain ⟨hx0, hx0fin, hx0err⟩ := x0_prep hS h0
  have hx0abs : |x0.toRat| ≤ |((S : ℚ))| + e1 ((S : ℚ)) := by
    have h := abs_sub_abs_le_abs_sub x0.toRat ((S : ℚ))
    linarith
  have hm1max : |sa.toRat * x0.toRat| ≤ Format.binary32.maxVal := by
    have hr := hroom.1
    rw [show max32 = Format.binary32.maxVal from rfl] at hr
    rw [abs_mul]
    nlinarith [abs_nonneg sa.toRat]
  obtain ⟨hm1fin, hm1err⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
    hsa hx0fin hm1max
  have hx1eq : x1 = FloatV.mul sa x0 := FloatV.NaNEq.eq h1 (FloatV.isFinite_iff.mp hm1fin).1
  have hx1fin : x1.isFinite = true := by rw [hx1eq]; exact hm1fin
  have hx1err : |x1.toRat - sa.toRat * x0.toRat| ≤ |sa.toRat * x0.toRat| * u32 + eta32 := by
    rw [hx1eq]
    simpa [u32, eta32] using hm1err
  have hx1e2 : |x1.toRat - sa.toRat * ((S : ℚ))| ≤ e2 sa.toRat ((S : ℚ)) := by
    have e : x1.toRat - sa.toRat * ((S : ℚ)) =
        (x1.toRat - sa.toRat * x0.toRat) + sa.toRat * (x0.toRat - ((S : ℚ))) := by ring
    rw [e]
    have t := abs_add_le (x1.toRat - sa.toRat * x0.toRat)
      (sa.toRat * (x0.toRat - ((S : ℚ))))
    have h2r : |sa.toRat * (x0.toRat - ((S : ℚ)))| ≤ |sa.toRat| * e1 ((S : ℚ)) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hx0err (abs_nonneg _)
    have h3r : |sa.toRat * x0.toRat| ≤ |sa.toRat * ((S : ℚ))| + |sa.toRat| * e1 ((S : ℚ)) := by
      have e2' : sa.toRat * x0.toRat = sa.toRat * ((S : ℚ)) + sa.toRat * (x0.toRat - ((S : ℚ))) := by
        ring
      rw [e2']
      have t2 := abs_add_le (sa.toRat * ((S : ℚ))) (sa.toRat * (x0.toRat - ((S : ℚ))))
      linarith
    have h4r : |sa.toRat * x0.toRat| * u32 ≤
        (|sa.toRat * ((S : ℚ))| + |sa.toRat| * e1 ((S : ℚ))) * u32 :=
      mul_le_mul_of_nonneg_right h3r hu0
    unfold e2
    linarith
  have hx1abs : |x1.toRat| ≤ |sa.toRat * ((S : ℚ))| + e2 sa.toRat ((S : ℚ)) := by
    have h := abs_sub_abs_le_abs_sub x1.toRat (sa.toRat * ((S : ℚ)))
    linarith
  have hm2max : |sb.toRat * x1.toRat| ≤ Format.binary32.maxVal := by
    have hr := hroom.2.1
    rw [show max32 = Format.binary32.maxVal from rfl] at hr
    rw [abs_mul]
    nlinarith [abs_nonneg sb.toRat]
  obtain ⟨hm2fin, hm2err⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
    hsb hx1fin hm2max
  have hx2eq : x2 = FloatV.mul sb x1 := FloatV.NaNEq.eq h2 (FloatV.isFinite_iff.mp hm2fin).1
  have hx2fin : x2.isFinite = true := by rw [hx2eq]; exact hm2fin
  have hx2err : |x2.toRat - sb.toRat * x1.toRat| ≤ |sb.toRat * x1.toRat| * u32 + eta32 := by
    rw [hx2eq]
    simpa [u32, eta32] using hm2err
  have hx2e3 : |x2.toRat - sa.toRat * sb.toRat * ((S : ℚ))| ≤
      e3 sa.toRat sb.toRat ((S : ℚ)) := by
    have e : x2.toRat - sa.toRat * sb.toRat * ((S : ℚ)) =
        (x2.toRat - sb.toRat * x1.toRat) +
          sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))) := by ring
    rw [e]
    have t := abs_add_le (x2.toRat - sb.toRat * x1.toRat)
      (sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))))
    have h2r : |sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ)))| ≤ |sb.toRat| * e2 sa.toRat ((S : ℚ)) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hx1e2 (abs_nonneg _)
    have h3r : |sb.toRat * x1.toRat| ≤ |sa.toRat * sb.toRat * ((S : ℚ))| +
        |sb.toRat| * e2 sa.toRat ((S : ℚ)) := by
      have eqt : sb.toRat * x1.toRat =
          sb.toRat * (sa.toRat * ((S : ℚ))) + sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))) := by
        ring
      rw [eqt]
      have t2 := abs_add_le (sb.toRat * (sa.toRat * ((S : ℚ))))
        (sb.toRat * (x1.toRat - sa.toRat * ((S : ℚ))))
      have e4 : |sb.toRat * (sa.toRat * ((S : ℚ)))| = |sa.toRat * sb.toRat * ((S : ℚ))| := by
        congr 1
        ring
      linarith
    have h4r : |sb.toRat * x1.toRat| * u32 ≤
        (|sa.toRat * sb.toRat * ((S : ℚ))| + |sb.toRat| * e2 sa.toRat ((S : ℚ))) * u32 :=
      mul_le_mul_of_nonneg_right h3r hu0
    unfold e3
    linarith
  have hx2abs : |x2.toRat| ≤ |sa.toRat * sb.toRat * ((S : ℚ))| + e3 sa.toRat sb.toRat ((S : ℚ)) := by
    have h := abs_sub_abs_le_abs_sub x2.toRat (sa.toRat * sb.toRat * ((S : ℚ)))
    linarith
  have hM : |x2.toRat| + |bv.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hroom3 := hroom.2.2
    rw [show max32 = Format.binary32.maxVal from rfl,
      show eta32 = Format.binary32.eta from rfl] at hroom3
    rw [hbvq]
    linarith
  obtain ⟨hx3fin, hx3err⟩ := FloatV.AddRel.err Format.binary32_regular
    (by decide) (by rw [Format.binary32_u]; norm_num) h3 hx2fin hbv hM
  have hx3err' : |x3.toRat - (x2.toRat + bv.toRat)| ≤ 3 * u32 * (|x2.toRat| + |β|) + 4 * eta32 := by
    rw [hbvq] at hx3err ⊢
    simpa [u32, eta32] using hx3err
  refine ⟨hx3fin, ?_⟩
  have htri : |x3.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + β)| ≤
      (3 * u32 * (|x2.toRat| + |β|) + 4 * eta32) + e3 sa.toRat sb.toRat ((S : ℚ)) := by
    have e : x3.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + β) =
        (x3.toRat - (x2.toRat + bv.toRat)) +
          ((x2.toRat - sa.toRat * sb.toRat * ((S : ℚ))) + (bv.toRat - β)) := by ring
    rw [e]
    have t1 := abs_add_le (x3.toRat - (x2.toRat + bv.toRat))
      ((x2.toRat - sa.toRat * sb.toRat * ((S : ℚ))) + (bv.toRat - β))
    have t2 := abs_add_le (x2.toRat - sa.toRat * sb.toRat * ((S : ℚ))) (bv.toRat - β)
    have hz : |bv.toRat - β| = 0 := by rw [hbvq]; simp
    linarith
  have hFbig : |x3.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + β)| ≤
      e3 sa.toRat sb.toRat ((S : ℚ)) +
        3 * u32 * (|sa.toRat * sb.toRat * ((S : ℚ))| + e3 sa.toRat sb.toRat ((S : ℚ)) + |β|) +
        4 * eta32 := by
    have hmono : 3 * u32 * (|x2.toRat| + |β|) ≤
        3 * u32 * (|sa.toRat * sb.toRat * ((S : ℚ))| + e3 sa.toRat sb.toRat ((S : ℚ)) + |β|) := by
      have h3u : (0:ℚ) ≤ 3 * u32 := by linarith
      exact mul_le_mul_of_nonneg_left (by linarith) h3u
    linarith
  have hrel := relax_le sa.toRat sb.toRat ((S : ℚ)) β _ hFbig
  have hsbmax : |sb.toRat| ≤ max32 := by
    have h := FloatV.abs_toRat_le_maxVal (f := Format.binary32) Format.binary32_regular hsb
    simpa [max32] using h
  exact relax_tol sa.toRat sb.toRat ((S : ℚ)) β _ hsbmax hrel

/-- Without a bias: the `float32` chain from the integer sum `S` to the stored value `c` gives a finite `float32` `v` within `tol` of `R` that rounds to `c`, and `c` meets the output bound. -/
theorem value_nobias (P : Platform) {od : DType} (hod : od = .bf16 ∨ od = .f16)
    {S : ℤ} {sa sb x0 x1 x2 : F32} {c : od.Val}
    (hS : |((S : ℚ))| ≤ 2 ^ 31)
    (h0 : tl.ItofRel .binary32 S x0)
    (h1 : FloatV.NaNEq (FloatV.mul sa x0) x1)
    (h2 : FloatV.NaNEq (FloatV.mul sb x1) x2)
    (hc : ConvRel P .f32 od x2 c)
    (hsa : sa.isFinite = true) (hsb : sb.isFinite = true)
    (hroom : Room32 sa.toRat sb.toRat ((S : ℚ)) 0) :
    ∃ v : F32, v.isFinite = true ∧
      |v.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + 0)| ≤
        tol sa.toRat sb.toRat ((S : ℚ)) 0 ∧
      OutIs od v c ∧
      (|sa.toRat * sb.toRat * ((S : ℚ)) + 0| + tol sa.toRat sb.toRat ((S : ℚ)) 0 ≤ outMax od →
        ∃ q : ℚ, od.val? c = some q ∧
          |q - (sa.toRat * sb.toRat * ((S : ℚ)) + 0)| ≤
            outTol od sa.toRat sb.toRat ((S : ℚ)) 0) := by
  obtain ⟨hfin, htol⟩ := tol_of_nobias hS h0 h1 h2 hsa hsb hroom
  obtain ⟨his, hout⟩ := out_conclusion P hod hfin htol hc
  exact ⟨x2, hfin, htol, his, hout⟩

/-- The stored value with a bias satisfies the lane conclusion. -/
theorem value_bias (P : Platform) {od : DType} (hod : od = .bf16 ∨ od = .f16)
    {S : ℤ} {sa sb x0 x1 x2 x3 bv : F32} {β : ℚ} {c : od.Val}
    (hS : |((S : ℚ))| ≤ 2 ^ 31)
    (h0 : tl.ItofRel .binary32 S x0)
    (h1 : FloatV.NaNEq (FloatV.mul sa x0) x1)
    (h2 : FloatV.NaNEq (FloatV.mul sb x1) x2)
    (hbv : bv.isFinite = true) (hbvq : bv.toRat = β)
    (h3 : AddRel true x2 bv x3)
    (hc : ConvRel P .f32 od x3 c)
    (hsa : sa.isFinite = true) (hsb : sb.isFinite = true)
    (hroom : Room32 sa.toRat sb.toRat ((S : ℚ)) β) :
    ∃ v : F32, v.isFinite = true ∧
      |v.toRat - (sa.toRat * sb.toRat * ((S : ℚ)) + β)| ≤
        tol sa.toRat sb.toRat ((S : ℚ)) β ∧
      OutIs od v c ∧
      (|sa.toRat * sb.toRat * ((S : ℚ)) + β| + tol sa.toRat sb.toRat ((S : ℚ)) β ≤ outMax od →
        ∃ q : ℚ, od.val? c = some q ∧
          |q - (sa.toRat * sb.toRat * ((S : ℚ)) + β)| ≤
            outTol od sa.toRat sb.toRat ((S : ℚ)) β) := by
  obtain ⟨hfin, htol⟩ := tol_of_bias hS h0 h1 h2 hbv hbvq h3 hsa hsb hroom
  obtain ⟨his, hout⟩ := out_conclusion P hod hfin htol hc
  exact ⟨x3, hfin, htol, his, hout⟩

end Claims.ScaledMmInt8
