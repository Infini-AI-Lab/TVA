import TVA.Proof

/-!
# The `y` chain: `mul`, then `cvt`

`y = fl(x · rstd)` in `f32`, then rounded to `bf16` (or stored as `f32`); we want it within
`2⁻⁷·|x·R| + 2⁻¹³²` (resp. `2⁻¹¹·|x·R| + 2⁻¹⁴⁹`) of `x · R`, where `R` is the exact `rstd`.
-/

namespace Claims.L2NormFwd

open TVA

/-- **The `bfloat16` `y`**: the product with the computed `rstd`, rounded to `bfloat16`, is within `2^-7·|x R| + 2^-132` of `x R`. -/
theorem y_err_bf16 {x R r y c : ℝ} (hR0 : 0 < R)
    (hr : |r - R| ≤ (2 : ℝ) ^ (-12 : ℤ) * R)
    (hy : |y - x * r| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x * r| + (2 : ℝ) ^ (-150 : ℤ))
    (hc : |c - y| ≤ (2 : ℝ) ^ (-8 : ℤ) * |y| + (2 : ℝ) ^ (-134 : ℤ)) :
    |c - x * R| ≤ (2 : ℝ) ^ (-7 : ℤ) * |x * R| + (2 : ℝ) ^ (-132 : ℤ) := by
  -- triangle inequality: `c - x*R` through `y` and `x*r`
  have h1 : |c - x * R| ≤ |c - y| + |y - x * r| + |x * r - x * R| := by
    have e : c - x * R = (c - y) + (y - x * r) + (x * r - x * R) := by ring
    rw [e]
    calc |(c - y) + (y - x * r) + (x * r - x * R)|
        ≤ |(c - y) + (y - x * r)| + |x * r - x * R| := abs_add_le _ _
      _ ≤ (|c - y| + |y - x * r|) + |x * r - x * R| := by
            linarith [abs_add_le (c - y) (y - x * r)]
  -- the `rstd` error: `|x*r - x*R| ≤ 2⁻¹²·|x*R|`
  have h2 : |x * r - x * R| ≤ (2 : ℝ) ^ (-12 : ℤ) * |x * R| := by
    have e : x * r - x * R = x * (r - R) := by ring
    rw [e, abs_mul]
    have hxR : |x * R| = |x| * R := by rw [abs_mul, abs_of_pos hR0]
    have h := mul_le_mul_of_nonneg_left hr (abs_nonneg x)
    calc |x| * |r - R| ≤ |x| * ((2 : ℝ) ^ (-12 : ℤ) * R) := h
      _ = (2 : ℝ) ^ (-12 : ℤ) * (|x| * R) := by ring
      _ = (2 : ℝ) ^ (-12 : ℤ) * |x * R| := by rw [hxR]
  -- the `mul` error: `|y| ≤ (1 + 2⁻²⁴)·|x*r| + 2⁻¹⁵⁰`
  have h3 : |y| ≤ |x * r| + (2 : ℝ) ^ (-24 : ℤ) * |x * r| + (2 : ℝ) ^ (-150 : ℤ) := by
    have e : y = (y - x * r) + x * r := by ring
    have h := abs_add_le (y - x * r) (x * r)
    calc |y| = |(y - x * r) + x * r| := by conv_lhs => rw [e]
      _ ≤ |y - x * r| + |x * r| := h
      _ ≤ ((2 : ℝ) ^ (-24 : ℤ) * |x * r| + (2 : ℝ) ^ (-150 : ℤ)) + |x * r| := by linarith
      _ = |x * r| + (2 : ℝ) ^ (-24 : ℤ) * |x * r| + (2 : ℝ) ^ (-150 : ℤ) := by ring
  -- `|x*r| ≤ (1 + 2⁻¹²)·|x*R|`
  have h4 : |x * r| ≤ |x * R| * (1 + (2 : ℝ) ^ (-12 : ℤ)) := by
    have e : x * r = (x * r - x * R) + x * R := by ring
    have h := abs_add_le (x * r - x * R) (x * R)
    calc |x * r| = |(x * r - x * R) + x * R| := by conv_lhs => rw [e]
      _ ≤ |x * r - x * R| + |x * R| := h
      _ ≤ (2 : ℝ) ^ (-12 : ℤ) * |x * R| + |x * R| := by linarith
      _ = |x * R| * (1 + (2 : ℝ) ^ (-12 : ℤ)) := by ring
  -- numerical facts
  have hA : ((2 : ℝ) ^ (-8 : ℤ) + (2 : ℝ) ^ (-32 : ℤ) + (2 : ℝ) ^ (-24 : ℤ)) *
      (1 + (2 : ℝ) ^ (-12 : ℤ)) + (2 : ℝ) ^ (-12 : ℤ) ≤ (2 : ℝ) ^ (-7 : ℤ) := by
    norm_num [zpow_neg]
  have hB : (2 : ℝ) ^ (-8 : ℤ) * (2 : ℝ) ^ (-150 : ℤ) + (2 : ℝ) ^ (-134 : ℤ) +
      (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-132 : ℤ) := by
    norm_num [zpow_neg]
  have hAP : ((2 : ℝ) ^ (-8 : ℤ) + (2 : ℝ) ^ (-32 : ℤ) + (2 : ℝ) ^ (-24 : ℤ)) *
      (1 + (2 : ℝ) ^ (-12 : ℤ)) * |x * R| + (2 : ℝ) ^ (-12 : ℤ) * |x * R| ≤
      (2 : ℝ) ^ (-7 : ℤ) * |x * R| := by
    have hp : (0 : ℝ) ≤ |x * R| := abs_nonneg _
    nlinarith [hA, hp, mul_nonneg hp hp]
  norm_num [zpow_neg] at h1 h2 h3 h4 hc hy hA hB hAP ⊢
  nlinarith [h1, h2, h3, h4, hc, hy, hA, hB]

/-- The `f32` `y` is stored without a further conversion, so `c = y`. -/
theorem y_err_f32 {x R r y c : ℝ} (hR0 : 0 < R)
    (hr : |r - R| ≤ (2 : ℝ) ^ (-12 : ℤ) * R)
    (hy : |y - x * r| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x * r| + (2 : ℝ) ^ (-150 : ℤ))
    (hc : c = y) :
    |c - x * R| ≤ (2 : ℝ) ^ (-11 : ℤ) * |x * R| + (2 : ℝ) ^ (-149 : ℤ) := by
  rw [hc]
  have h1 : |y - x * R| ≤ |y - x * r| + |x * r - x * R| := by
    have e : y - x * R = (y - x * r) + (x * r - x * R) := by ring
    rw [e]
    exact abs_add_le _ _
  have h2 : |x * r - x * R| ≤ (2 : ℝ) ^ (-12 : ℤ) * |x * R| := by
    have e : x * r - x * R = x * (r - R) := by ring
    rw [e, abs_mul]
    have hxR : |x * R| = |x| * R := by rw [abs_mul, abs_of_pos hR0]
    have h := mul_le_mul_of_nonneg_left hr (abs_nonneg x)
    calc |x| * |r - R| ≤ |x| * ((2 : ℝ) ^ (-12 : ℤ) * R) := h
      _ = (2 : ℝ) ^ (-12 : ℤ) * (|x| * R) := by ring
      _ = (2 : ℝ) ^ (-12 : ℤ) * |x * R| := by rw [hxR]
  have h4 : |x * r| ≤ |x * R| * (1 + (2 : ℝ) ^ (-12 : ℤ)) := by
    have e : x * r = (x * r - x * R) + x * R := by ring
    have h := abs_add_le (x * r - x * R) (x * R)
    calc |x * r| = |(x * r - x * R) + x * R| := by conv_lhs => rw [e]
      _ ≤ |x * r - x * R| + |x * R| := h
      _ ≤ (2 : ℝ) ^ (-12 : ℤ) * |x * R| + |x * R| := by linarith
      _ = |x * R| * (1 + (2 : ℝ) ^ (-12 : ℤ)) := by ring
  have hA : (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-12 : ℤ)) + (2 : ℝ) ^ (-12 : ℤ)
      ≤ (2 : ℝ) ^ (-11 : ℤ) := by norm_num [zpow_neg]
  have hB : (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-149 : ℤ) := by norm_num [zpow_neg]
  have hp : (0 : ℝ) ≤ |x * R| := abs_nonneg _
  have hm := mul_le_mul_of_nonneg_left h4 (by positivity : (0:ℝ) ≤ (2 : ℝ) ^ (-24 : ℤ))
  have hm' : (2 : ℝ) ^ (-24 : ℤ) * |x * r| ≤
      ((2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-12 : ℤ))) * |x * R| := by
    calc (2 : ℝ) ^ (-24 : ℤ) * |x * r|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (|x * R| * (1 + (2 : ℝ) ^ (-12 : ℤ))) := hm
      _ = ((2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-12 : ℤ))) * |x * R| := by ring
  have hstep : ((2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-12 : ℤ))) * |x * R| +
      (2 : ℝ) ^ (-12 : ℤ) * |x * R| ≤ (2 : ℝ) ^ (-11 : ℤ) * |x * R| := by
    have h := mul_le_mul_of_nonneg_right hA (abs_nonneg (x * R))
    rwa [add_mul] at h
  have key : (2 : ℝ) ^ (-24 : ℤ) * |x * r| + (2 : ℝ) ^ (-150 : ℤ) +
      (2 : ℝ) ^ (-12 : ℤ) * |x * R| ≤ (2 : ℝ) ^ (-11 : ℤ) * |x * R| + (2 : ℝ) ^ (-149 : ℤ) := by
    linarith [hm', hstep, hB]
  linarith [h1, h2, hy, key]

end Claims.L2NormFwd
