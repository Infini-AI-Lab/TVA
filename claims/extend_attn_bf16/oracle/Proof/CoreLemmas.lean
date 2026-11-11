import Proof.CoreDefs

/-!
# Small facts on `expm1` and the factors of the error analysis
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- One plus `expm1 x` is `exp x`. -/
theorem one_add_expm1 (x : ℝ) : 1 + expm1 x = Real.exp x := by unfold expm1; ring

/-- `expm1` is nonnegative on nonnegative arguments. -/
theorem expm1_nonneg {x : ℝ} (h : 0 ≤ x) : 0 ≤ expm1 x := by
  unfold expm1; linarith [Real.one_le_exp h]

/-- `expm1` is monotone. -/
theorem expm1_mono {x y : ℝ} (h : x ≤ y) : expm1 x ≤ expm1 y := by
  unfold expm1; linarith [Real.exp_le_exp.mpr h]

/-- Every real `x` is at most `expm1 x`. -/
theorem le_expm1 (x : ℝ) : x ≤ expm1 x := by unfold expm1; linarith [Real.add_one_le_exp x]

/-- `exp x ≤ 1 + x + 2 x²` for `0 ≤ x ≤ 1/2`. -/
theorem exp_le_quad {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1 / 2) : Real.exp x ≤ 1 + x + 2 * x ^ 2 := by
  have h := Real.exp_bound_div_one_sub_of_interval h0 (by linarith)
  have hx : 0 < 1 - x := by linarith
  refine h.trans ?_
  rw [div_le_iff₀ hx]
  nlinarith [mul_nonneg h0 h0, mul_nonneg (mul_nonneg h0 h0) h0, mul_nonneg h0 (sub_nonneg.mpr h1),
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.mpr h1)]

/-- The product of two factors close to `1`. -/
theorem fac_mul {F g A B : ℝ} (hA : 0 ≤ A) (hB : 0 ≤ B) (hF : |F - 1| ≤ expm1 A)
    (hg : |g - 1| ≤ expm1 B) : |F * g - 1| ≤ expm1 (A + B) := by
  have hA' := expm1_nonneg hA
  have hB' := expm1_nonneg hB
  have e : F * g - 1 = (F - 1) * (g - 1) + (F - 1) + (g - 1) := by ring
  have h1 : |F * g - 1| ≤ |F - 1| * |g - 1| + |F - 1| + |g - 1| := by
    rw [e]
    have a1 := abs_add_le ((F - 1) * (g - 1) + (F - 1)) (g - 1)
    have a2 := abs_add_le ((F - 1) * (g - 1)) (F - 1)
    rw [abs_mul] at a2
    linarith
  have h2 : expm1 (A + B) = expm1 A * expm1 B + expm1 A + expm1 B := by
    unfold expm1; rw [Real.exp_add]; ring
  rw [h2]
  have := mul_le_mul hF hg (abs_nonneg _) hA'
  linarith

/-- A factor close to `1` is at least `2/3` when its exponent is at most `1/4`. -/
theorem fac_ge {F A : ℝ} (hA : A ≤ 1 / 4) (hF : |F - 1| ≤ expm1 A) : 2 / 3 ≤ F := by
  have h1 : Real.exp A ≤ 4 / 3 := by
    rcases le_or_gt 0 A with h0 | h0
    · have := Real.exp_bound_div_one_sub_of_interval h0 (by linarith)
      refine this.trans ?_
      rw [div_le_iff₀ (by linarith)]; linarith
    · have := Real.exp_le_one_iff.mpr h0.le; linarith
  have := (abs_le.mp hF).1
  unfold expm1 at this
  linarith

/-- A factor close to `1`, at most `exp A`. -/
theorem fac_le {F A : ℝ} (hF : |F - 1| ≤ expm1 A) : F ≤ Real.exp A := by
  have := (abs_le.mp hF).2
  unfold expm1 at this
  linarith

/-- The weight `exp (ŝ − m) · F` is at most `exp (t e1)` when `m ≥ ŝ`: the factor's `e2` part
is less than the exponent's decay. -/
theorem weight_le {F : ℝ} {t : ℕ} {sj m : ℝ} (hm : sj ≤ m)
    (hF : |F - 1| ≤ expm1 (t * e1 + e2 * (m - sj))) : Real.exp (sj - m) * F ≤ Real.exp (t * e1) := by
  have h1 := fac_le hF
  have h0 : 0 ≤ Real.exp (sj - m) := (Real.exp_pos _).le
  calc Real.exp (sj - m) * F ≤ Real.exp (sj - m) * Real.exp (t * e1 + e2 * (m - sj)) :=
        mul_le_mul_of_nonneg_left h1 h0
    _ = Real.exp (t * e1 - (1 - e2) * (m - sj)) := by rw [← Real.exp_add]; congr 1; ring
    _ ≤ Real.exp (t * e1) := by
        apply Real.exp_le_exp.mpr
        have : 0 ≤ (1 - e2) * (m - sj) := mul_nonneg (by unfold e2; norm_num) (by linarith)
        linarith

/-- A trip count within `tmax` is at most `2^13` as a real. -/
theorem t_le_real {t : ℕ} (ht : t ≤ tmax) : (t : ℝ) ≤ 2 ^ 13 := by
  have h : t ≤ 8192 := by simpa [tmax] using ht
  have : (t : ℝ) ≤ 8192 := by exact_mod_cast h
  linarith

/-- The constant `e1` is at most 2^-21. -/
theorem e1_le : e1 ≤ 1 / 2 ^ 21 := by unfold e1; norm_num
/-- The constant `e1` is positive. -/
theorem e1_pos : 0 < e1 := by unfold e1; norm_num
/-- The constant `e2` is positive. -/
theorem e2_pos : 0 < e2 := by unfold e2; norm_num
/-- The constant `e2` is at most 2^-21. -/
theorem e2_le : e2 ≤ 1 / 2 ^ 21 := by unfold e2; norm_num
/-- The constant `uf` is positive. -/
theorem uf_pos : 0 < uf := by unfold uf; norm_num

/-- `exp e1 ≤ 1 + 2^-20`. -/
theorem exp_e1_le : Real.exp e1 ≤ 1 + 1 / 2 ^ 20 := by
  have := exp_le_quad e1_pos.le (e1_le.trans (by norm_num))
  unfold e1 at this ⊢
  norm_num at this ⊢
  linarith

/-- The exponent of a factor stays below `1/100`: `t ≤ 2^13` trips and a gap of `y ≤ 2^15`. -/
theorem expo_small {t y : ℝ} (ht : t ≤ 2 ^ 13) (_ht0 : 0 ≤ t) (hy : y ≤ 2 ^ 15) (_hy0 : 0 ≤ y) :
    t * e1 + e2 * y ≤ 1 / 100 := by
  have h1 : t * e1 ≤ 2 ^ 13 * e1 := mul_le_mul_of_nonneg_right ht e1_pos.le
  have h2 : e2 * y ≤ e2 * 2 ^ 15 := mul_le_mul_of_nonneg_left hy e2_pos.le
  unfold e1 at h1
  unfold e2 at h2
  norm_num at h1 h2
  unfold e1 e2
  linarith

/-- The weight bound `exp (t e1) ≤ 1 + 1/2^7`. -/
theorem exp_te1_le {t : ℝ} (ht : t ≤ 2 ^ 13) (ht0 : 0 ≤ t) : Real.exp (t * e1) ≤ 1 + 1 / 2 ^ 7 := by
  have h := expo_small ht ht0 (y := 0) (by norm_num) le_rfl
  have h1 : t * e1 ≤ 1 / 2 ^ 8 := by
    have h1 : t * e1 ≤ 2 ^ 13 * e1 := mul_le_mul_of_nonneg_right ht e1_pos.le
    unfold e1 at h1 ⊢; norm_num at h1 ⊢; linarith
  have := exp_le_quad (mul_nonneg ht0 e1_pos.le) (by linarith)
  nlinarith [mul_nonneg ht0 e1_pos.le]

end

end Claims.ExtendAttn
