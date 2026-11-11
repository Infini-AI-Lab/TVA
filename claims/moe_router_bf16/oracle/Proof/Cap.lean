import Claim
import TVA.Proof

/-!
# Real lemmas about `tanh` and the soft cap

`tanh` is `1`-Lipschitz, `eᵈ/(eᵈ+1)² ≤ e^{−|d|}`, and the change of `(a−1)/(a+1)`.
-/

namespace Claims.MoeRouter

open TVA

open Real

/-- The change of `(a−1)/(a+1)` between two nonnegative arguments: `2|a−b| / ((a+1)(b+1))`. -/
theorem cap_abs_diff (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    |(a - 1) / (a + 1) - (b - 1) / (b + 1)| ≤ 2 * |a - b| / ((a + 1) * (b + 1)) := by
  have ha1 : (0 : ℝ) < a + 1 := by linarith
  have hb1 : (0 : ℝ) < b + 1 := by linarith
  have hkey : (a - 1) / (a + 1) - (b - 1) / (b + 1)
      = 2 * (a - b) / ((a + 1) * (b + 1)) := by
    have h1 : a + 1 ≠ 0 := ha1.ne'
    have h2 : b + 1 ≠ 0 := hb1.ne'
    field_simp
    ring
  rw [hkey]
  simp only [abs_div, abs_mul, abs_of_pos ha1, abs_of_pos hb1,
    abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  exact le_rfl

/-- `eᵈ/(eᵈ+1)² ≤ e^{−|d|}`. -/
theorem exp_ratio_le (d : ℝ) :
    Real.exp d / (Real.exp d + 1) ^ 2 ≤ Real.exp (-(|d|)) := by
  have he : 0 < Real.exp d := Real.exp_pos d
  have hpos : (0 : ℝ) < (Real.exp d + 1) ^ 2 := by positivity
  rcases lt_or_ge d 0 with hd | hd
  · rw [abs_of_neg hd, neg_neg, div_le_iff₀ hpos]
    nlinarith [sq_nonneg (Real.exp d), he]
  · rw [abs_of_nonneg hd, Real.exp_neg, div_le_iff₀ hpos, inv_mul_eq_div,
      le_div_iff₀ he]
    nlinarith [sq_nonneg (Real.exp d), he]

/-- `tanh` is `1`-Lipschitz. -/
theorem tanh_lipschitz (a b : ℝ) : |Real.tanh a - Real.tanh b| ≤ |a - b| := by
  have hderiv : ∀ x : ℝ, HasDerivAt Real.tanh (1 - Real.tanh x ^ 2) x := by
    intro x
    have hexp : HasDerivAt (fun y : ℝ => Real.exp y) (Real.exp x) x := Real.hasDerivAt_exp x
    have hexpneg : HasDerivAt (fun y : ℝ => Real.exp (-y)) (-(Real.exp (-x))) x := by
      have h := (Real.hasDerivAt_exp (-x)).comp x (hasDerivAt_neg x)
      convert h using 1
      · funext y; simp
      · simp
    have hu : HasDerivAt (fun y : ℝ => Real.exp y - Real.exp (-y))
        (Real.exp x + Real.exp (-x)) x := by
      have h := hexp.sub hexpneg
      convert h using 1
      simp
    have hv : HasDerivAt (fun y : ℝ => Real.exp y + Real.exp (-y))
        (Real.exp x - Real.exp (-x)) x := by
      have h := hexp.add hexpneg
      convert h using 1
    have hvpos : (0 : ℝ) < Real.exp x + Real.exp (-x) := by positivity
    have hq := hu.div hv hvpos.ne'
    have hfun : Real.tanh
        = fun y => (Real.exp y - Real.exp (-y)) / (Real.exp y + Real.exp (-y)) :=
      funext Real.tanh_eq
    rw [hfun]
    convert hq using 1
    have hne : Real.exp x + Real.exp (-x) ≠ 0 := hvpos.ne'
    field_simp
  have hbound : ∀ x ∈ (Set.univ : Set ℝ), ‖1 - Real.tanh x ^ 2‖ ≤ (1 : ℝ) := by
    intro x _
    have hsq : (Real.tanh x) ^ 2 < 1 := Real.tanh_sq_lt_one x
    have hnn : (0 : ℝ) ≤ 1 - Real.tanh x ^ 2 := by linarith
    rw [Real.norm_eq_abs, abs_of_nonneg hnn]
    linarith [sq_nonneg (Real.tanh x)]
  have hmvt := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := Real.tanh) (f' := fun x => 1 - Real.tanh x ^ 2) (C := 1)
    (s := Set.univ) (fun x _ => (hderiv x).hasDerivWithinAt) hbound convex_univ
    (Set.mem_univ a) (Set.mem_univ b)
  rw [Real.norm_eq_abs, Real.norm_eq_abs, one_mul] at hmvt
  calc |Real.tanh a - Real.tanh b| = |Real.tanh b - Real.tanh a| := abs_sub_comm _ _
    _ ≤ |b - a| := hmvt
    _ = |a - b| := abs_sub_comm _ _

end Claims.MoeRouter
