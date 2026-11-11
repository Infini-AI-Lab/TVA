import Proof.CoreDefs

/-!
# The division and the store
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **The output**: `x = acc / deno` is within `E · wabs + A` of `wref`; the `float32` division
gives `q` (`2^-22` relative, `2^-148` absolute), the store `c` (`bfloat16`: `2^-8` relative,
`2^-134` absolute). If `E (1 + 2^-22) + 2^-22 ≤ E1`, then `c` is within the claim's
`ub + (1 + ub) · E1` of `wref`, relative to `wabs`, plus `2^-80 (1 + vs)` with `vs` the sum of
`|v|`, when `A ≤ 2^-85 (1 + vs)`. -/
theorem store_bound {E E1 A wabs wr x q c vs : ℝ} (hwabs : 0 ≤ wabs)
    (hwr : |wr| ≤ wabs) (hvs : 0 ≤ vs) (hA : A ≤ 1 / 2 ^ 85 * (1 + vs))
    (hx : |x - wr| ≤ E * wabs + A) (hq : |q - x| ≤ 1 / 2 ^ 22 * |x| + 1 / 2 ^ 148)
    (hc : |c - q| ≤ ub * |q| + 1 / 2 ^ 134) (hE1 : E * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22 ≤ E1) :
    |c - wr| ≤ (1 / 2 ^ 8 + (1 + 1 / 2 ^ 8) * E1) * wabs + 1 / 2 ^ 80 * (1 + vs) := by
  have hub : ub = 1 / 2 ^ 8 := rfl
  set D := |x - wr| with hD
  have hD0 : 0 ≤ D := abs_nonneg _
  have hxw : |x| ≤ wabs + D := by
    have := abs_add_le wr (x - wr)
    rw [add_sub_cancel] at this
    linarith
  have hqw : |q - wr| ≤ 1 / 2 ^ 22 * wabs + (1 + 1 / 2 ^ 22) * D + 1 / 2 ^ 148 := by
    have := abs_add_le (q - x) (x - wr)
    rw [sub_add_sub_cancel] at this
    nlinarith
  have hqabs : |q| ≤ wabs + |q - wr| := by
    have := abs_add_le wr (q - wr)
    rw [add_sub_cancel] at this
    linarith
  have hcw : |c - wr| ≤ ub * wabs + (1 + ub) * |q - wr| + 1 / 2 ^ 134 := by
    have := abs_add_le (c - q) (q - wr)
    rw [sub_add_sub_cancel] at this
    rw [hub] at hc ⊢
    nlinarith [abs_nonneg (q - wr)]
  have hDb : D ≤ E * wabs + A := hx
  have hq2 : |q - wr| ≤ E1 * wabs + ((1 + 1 / 2 ^ 22) * A + 1 / 2 ^ 148) := by
    have h1 : (1 + 1 / 2 ^ 22) * D ≤ (1 + 1 / 2 ^ 22) * (E * wabs + A) :=
      mul_le_mul_of_nonneg_left hDb (by norm_num)
    have h2 : (E * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22) * wabs ≤ E1 * wabs :=
      mul_le_mul_of_nonneg_right hE1 hwabs
    nlinarith
  have hub0 : (0 : ℝ) ≤ 1 + ub := by rw [hub]; norm_num
  have h3 : (1 + ub) * |q - wr| ≤ (1 + ub) * (E1 * wabs + ((1 + 1 / 2 ^ 22) * A + 1 / 2 ^ 148)) :=
    mul_le_mul_of_nonneg_left hq2 hub0
  have hA0 : (1 + ub) * ((1 + 1 / 2 ^ 22) * A + 1 / 2 ^ 148) + 1 / 2 ^ 134 ≤ 1 / 2 ^ 80 * (1 + vs) := by
    have h4 : (1 + ub) * (1 + 1 / 2 ^ 22) * A ≤ (1 + ub) * (1 + 1 / 2 ^ 22) * (1 / 2 ^ 85 * (1 + vs)) :=
      mul_le_mul_of_nonneg_left hA (by rw [hub]; norm_num)
    rw [hub] at h4 ⊢
    nlinarith
  rw [hub] at h3
  nlinarith [h3, hcw, hA0]

end

end Claims.ExtendAttn
