import Proof.CoreLemmas

/-!
# The score of a lane, over the reals

`score_real`: the computed score `fl(D · ss)`, `D` the `tl.dot` of `bfloat16` operands and `ss` the
`float32` scale, is within the claim's `dlt BD X` of the exact `sig · Σ q·k`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **The computed score against the exact one.** `L = Σ liftMag q · liftMag k`, `sig · L ≤ X`;
`D` the dot (`BD` products), `ss` the scale within `2^-22` (relative) of `sig`, `qk1` the product
`fl (D · ss)`. -/
theorem score_real {BD : ℕ} {sig L S D ss qk1 X : ℝ} (hBD : BD ≤ 256)
    (hsig1 : 1 / 2 ^ 64 ≤ sig) (hsig2 : sig ≤ 2 ^ 64) (hL0 : 0 ≤ L) (hLX : sig * L ≤ X)
    (_hX : X ≤ 2 ^ 12) (hS : |S| ≤ L)
    (hD : |D - S| ≤ ((BD : ℝ) + 2) / 2 ^ 22 * L + ((BD : ℝ) + 2) / 2 ^ 126)
    (hss : |ss - sig| ≤ sig / 2 ^ 20)
    (hq : |qk1 - D * ss| ≤ 1 / 2 ^ 24 * |D * ss| + 1 / 2 ^ 150) :
    |qk1 - sig * S| ≤ ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X + 1 / 2 ^ 40 := by
  have hBD' : (BD : ℝ) ≤ 256 := by exact_mod_cast hBD
  have hBD0 : (0 : ℝ) ≤ BD := Nat.cast_nonneg BD
  have hsig0 : 0 < sig := lt_of_lt_of_le (by positivity) hsig1
  set aD : ℝ := ((BD : ℝ) + 2) / 2 ^ 126 with haD
  set rD : ℝ := ((BD : ℝ) + 2) / 2 ^ 22 with hrD
  have haD0 : 0 ≤ aD := by rw [haD]; positivity
  have hrD0 : 0 ≤ rD := by rw [hrD]; positivity
  have hrD1 : rD ≤ 1 / 2 ^ 13 := by rw [hrD]; rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith
  have hss1 : |ss| ≤ sig * (1 + 1 / 2 ^ 20) := by
    have := abs_sub_abs_le_abs_sub ss sig
    rw [abs_of_pos hsig0] at this
    have e : sig / 2 ^ 20 = sig * (1 / 2 ^ 20) := by ring
    nlinarith
  have hDb : |D| ≤ L * (1 + rD) + aD := by
    have := abs_sub_abs_le_abs_sub D S
    nlinarith
  have hDss : |D * ss| ≤ (L * (1 + rD) + aD) * (sig * (1 + 1 / 2 ^ 20)) := by
    rw [abs_mul]
    exact mul_le_mul hDb hss1 (abs_nonneg _) (by positivity)
  have hsplit : qk1 - sig * S = S * (ss - sig) + (D - S) * ss + (qk1 - D * ss) := by ring
  have h1 : |S * (ss - sig)| ≤ L * (sig / 2 ^ 20) := by
    rw [abs_mul]; exact mul_le_mul hS hss (abs_nonneg _) hL0
  have h2 : |(D - S) * ss| ≤ (rD * L + aD) * (sig * (1 + 1 / 2 ^ 20)) := by
    rw [abs_mul]; exact mul_le_mul hD hss1 (abs_nonneg _) (by positivity)
  have h3 : |qk1 - D * ss| ≤ 1 / 2 ^ 24 * ((L * (1 + rD) + aD) * (sig * (1 + 1 / 2 ^ 20))) + 1 / 2 ^ 150 := by
    have := mul_le_mul_of_nonneg_left hDss (by positivity : (0 : ℝ) ≤ 1 / 2 ^ 24)
    linarith
  have htri : |qk1 - sig * S| ≤ |S * (ss - sig)| + |(D - S) * ss| + |qk1 - D * ss| := by
    rw [hsplit]
    have a1 := abs_add_le (S * (ss - sig) + (D - S) * ss) (qk1 - D * ss)
    have a2 := abs_add_le (S * (ss - sig)) ((D - S) * ss)
    linarith
  -- collect: `sig * L ≤ X`, `aD * sig ≤ 2^-53`
  have hsL : sig * L ≤ X := hLX
  have hab : aD * sig ≤ 1 / 2 ^ 50 := by
    have : aD ≤ 258 / 2 ^ 126 := by rw [haD]; apply div_le_div_of_nonneg_right _ (by positivity); linarith
    calc aD * sig ≤ 258 / 2 ^ 126 * 2 ^ 64 := mul_le_mul this hsig2 hsig0.le (by positivity)
      _ ≤ 1 / 2 ^ 50 := by norm_num
  have hX0 : 0 ≤ X := le_trans (mul_nonneg hsig0.le hL0) hLX
  have hcoef : 1 / 2 ^ 20 + rD * (1 + 1 / 2 ^ 20) + 1 / 2 ^ 24 * (1 + rD) * (1 + 1 / 2 ^ 20) ≤
      ((BD : ℝ) + 8) * (1 / 2 ^ 22) := by
    rw [hrD]; nlinarith
  have hmain : |qk1 - sig * S| ≤
      (1 / 2 ^ 20 + rD * (1 + 1 / 2 ^ 20) + 1 / 2 ^ 24 * (1 + rD) * (1 + 1 / 2 ^ 20)) * (sig * L) +
        (aD * sig * (1 + 1 / 2 ^ 20) * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150) := by
    nlinarith [htri, h1, h2, h3]
  have hc0 : 0 ≤ 1 / 2 ^ 20 + rD * (1 + 1 / 2 ^ 20) + 1 / 2 ^ 24 * (1 + rD) * (1 + 1 / 2 ^ 20) := by positivity
  have hfin1 := mul_le_mul hcoef hsL (mul_nonneg hsig0.le hL0) (by positivity)
  have hfin2 : aD * sig * (1 + 1 / 2 ^ 20) * (1 + 1 / 2 ^ 24) ≤ 1 / 2 ^ 49 := by
    have : aD * sig * (1 + 1 / 2 ^ 20) * (1 + 1 / 2 ^ 24) ≤ 1 / 2 ^ 50 * 2 := by
      have hab0 : 0 ≤ aD * sig := mul_nonneg haD0 hsig0.le
      nlinarith
    norm_num at this ⊢; linarith
  have hnum : (1 : ℝ) / 2 ^ 49 + 1 / 2 ^ 150 ≤ 1 / 2 ^ 40 := by norm_num
  have hcoef0 : (0 : ℝ) ≤ ((BD : ℝ) + 8) * (1 / 2 ^ 22) := by positivity
  nlinarith [mul_le_mul_of_nonneg_left hLX hcoef0]

end

end Claims.ExtendAttn
