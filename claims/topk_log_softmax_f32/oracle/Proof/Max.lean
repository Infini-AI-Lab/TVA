import Proof.Exp

/-!
# Format facts and the exact quantities of a row

Format facts about `float32`, and the exact quantities `tR`, `ER`, `ZR` of a row relative to its
max `mv` (`expIn` written in them).
-/
namespace Claims.TopkLogSoftmax

open TVA

/-! ## `float32` facts -/

/-- A format fact about `float32`. -/
lemma b32_reg : Format.binary32.Regular := Format.binary32_regular
/-- A format fact about `float32`. -/
lemma b32_ieee : Format.binary32.specials = .ieee := rfl

/-- The rational value of `-∞` in `float32` is `-2^128`. -/
lemma negInf_toRat : (FloatV.negInf : F32).toRat = -((2 : ℚ) ^ 128) := by
  show Format.binary32.toRat (FloatV.negInf : F32).pat = _
  rw [FloatV.negInf_pat]
  have hisinf : Format.binary32.isInf
      (Format.binary32.setSign true Format.binary32.infMag) = true := by
    rw [Format.isInf_setSign]; exact Format.isInf_infMag b32_ieee
  rw [Format.toRat_of_isInf b32_reg hisinf, Format.sign_setSign]
  simp only [ite_true]
  have hb : Format.binary32.bias + 1 = (128 : Int) := by decide
  rw [hb]
  norm_num

/-- The largest finite `float32` is below `2^128`. -/
lemma maxVal_lt_two_pow : Format.binary32.maxVal < (2 : ℚ) ^ 128 := by
  have h := Format.maxVal_lt_inf b32_reg b32_ieee
  have hb : Format.binary32.bias = 127 := rfl
  rw [hb] at h
  simpa using h

/-! ## The exact quantities of one row

`mR` is the max `mv` as a real, `tR mv x = x - mv`, `ER mv x` is `e^(x - mv)` (`0` for `-∞`), and
`ZR` is the exact sum of the `ER` over the row. -/

/-- The block maximum `mv` as a real. -/
noncomputable def mR (mv : F32) : ℝ := (mv.toRat : ℝ)

/-- `x - mv` as a real: the exact exponent of lane `x`. -/
noncomputable def tR (mv x : F32) : ℝ := (x.toRat : ℝ) - mR mv

/-- `e^(x - mv)`, and `0` for `x = -∞`. -/
noncomputable def ER (mv x : F32) : ℝ :=
  if x = FloatV.negInf then 0 else Real.exp (tR mv x)

/-- The exact sum of `ER mv (xs k)` over the row. -/
noncomputable def ZR (mv : F32) (xs : Nat → F32) (n : Nat) : ℝ :=
  ∑ k ∈ Finset.range n, ER mv (xs k)

/-- `e^a = e^b · e^(a - b)`. -/
lemma exp_eq_mul_exp_sub (a b : ℝ) : Real.exp a = Real.exp b * Real.exp (a - b) := by
  rw [← Real.exp_add]; ring_nf

/-- A property of `ER`. -/
lemma ER_eq_exp {mv x : F32} (h : x ≠ FloatV.negInf) : ER mv x = Real.exp (tR mv x) := by
  unfold ER; rw [ite_eq_right h]

/-- A property of `ER`. -/
lemma ER_eq_zero {mv x : F32} (h : x = FloatV.negInf) : ER mv x = 0 := by
  unfold ER; rw [ite_eq_left h]

/-- A property of `ER`. -/
lemma ER_nonneg (mv x : F32) : 0 ≤ ER mv x := by
  unfold ER
  by_cases h : x = FloatV.negInf
  · rw [ite_eq_left h]
  · rw [ite_eq_right h]; exact (Real.exp_pos _).le

/-- A property of `ER`. -/
lemma ER_pos {mv x : F32} (h : x ≠ FloatV.negInf) : 0 < ER mv x := by
  rw [ER_eq_exp h]; exact Real.exp_pos _

/-- A property of `ER`. -/
lemma ER_le_one {mv x : F32} (h : x ≠ FloatV.negInf)
    (hle : (x.toRat : ℝ) ≤ mR mv) : ER mv x ≤ 1 := by
  rw [ER_eq_exp h, ← Real.exp_zero]
  apply Real.exp_le_exp.mpr
  show (x.toRat : ℝ) - mR mv ≤ 0
  linarith

/-- A property of `ER`. -/
lemma ER_self {mv : F32} (h : mv ≠ FloatV.negInf) : ER mv mv = 1 := by
  rw [ER_eq_exp h]
  have h0 : tR mv mv = 0 := by
    show (mv.toRat : ℝ) - mR mv = 0
    unfold mR; ring
  rw [h0, Real.exp_zero]

/-- The claim's `expIn x` is `e^mv · ER mv x`. -/
lemma expIn_eq (mv x : F32) : expIn x = Real.exp (mR mv) * ER mv x := by
  unfold expIn ER
  by_cases h : x = FloatV.negInf
  · rw [ite_eq_left h, ite_eq_left h]; ring
  · rw [ite_eq_right h, ite_eq_right h]
    exact exp_eq_mul_exp_sub _ _

/-- The exact denominator is `e^mv · ZR`. -/
lemma expIn_sum (mv : F32) (xs : Nat → F32) (n : Nat) :
    (∑ k ∈ Finset.range n, expIn (xs k)) = Real.exp (mR mv) * ZR mv xs n := by
  unfold ZR
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun k _ => expIn_eq mv (xs k))

/-- The sum `ZR` is at least `1` (the max lane contributes `1`). -/
lemma one_le_ZR {mv : F32} {xs : Nat → F32} {n i0 : Nat} (hi0 : i0 < n)
    (hmax : ER mv (xs i0) = 1) : 1 ≤ ZR mv xs n := by
  unfold ZR
  have h := Finset.single_le_sum (f := fun k => ER mv (xs k))
    (fun i _ => ER_nonneg mv (xs i)) (Finset.mem_range.mpr hi0)
  rwa [hmax] at h

/-- The sum `ZR` is at most `n`. -/
lemma ZR_le_n {mv : F32} {xs : Nat → F32} {n : Nat}
    (hle1 : ∀ k, k < n → ER mv (xs k) ≤ 1) : ZR mv xs n ≤ (n : ℝ) := by
  unfold ZR
  have h := Finset.sum_le_sum (fun k hk => hle1 k (Finset.mem_range.mp hk))
  simpa using h

end Claims.TopkLogSoftmax
