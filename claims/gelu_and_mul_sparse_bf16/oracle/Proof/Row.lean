import Proof.RowCut

/-!
# One row's statistics: the kernel's mean, standard deviation and cutoff

Each stage takes the bounds that the model's operations give (as reals: the `float32` sums along
trees of depth `L`, the divisions, the fusable subtractions, the squares, `tl.sqrt`, the `bfloat16`
roundings) and bounds the stage's result against the row's exact value.
-/

namespace Claims.GeluAndMulSparse

open TVA

namespace RowP

variable (r : RowP)

/-- `aMean ≥ 0`. -/
theorem aMean_nonneg : 0 ≤ r.aMean := r.aMean_nonneg'
/-- `aMean ≤ 2^32`. -/
theorem aMean_le : r.aMean ≤ 2 ^ 32 := r.aMean_le'
/-- `|mu| ≤ aMean`. -/
theorem mu_le : |r.mu| ≤ r.aMean := r.mu_le'
/-- `sd ≥ 0`. -/
theorem sd_nonneg : 0 ≤ r.sd := r.sd_nonneg'
/-- `sd ≤ 2^33`. -/
theorem sd_le : r.sd ≤ 2 ^ 33 := r.sd_le'
/-- `cutTol ≥ 2^-130`. -/
theorem cutTol_ge : 2 ^ (-130 : ℤ) ≤ r.cutTol := r.cutTol_ge'
/-- The block sum of `gl` is `d·mu`. -/
theorem sum_gl : ∑ j ∈ Finset.range (2 ^ r.L), r.gl j = r.d * r.mu := r.sum_gl'
/-- The block sum of `|gl|` is `d·aMean`. -/
theorem sum_abs_gl : ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| = r.d * r.aMean := r.sum_abs_gl'

/-- The sum of the block (`tl.sum` along a tree of depth `L`) is at most `2^48`. -/
theorem sum_room {S : ℝ}
    (hS : |S - ∑ j ∈ Finset.range (2 ^ r.L), r.gl j| ≤
      ((1 + u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| +
        eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L) :
    |S| ≤ 2 ^ 48 :=
  r.sum_room' hS

/-- **The mean** (line 75): the sum along a tree of depth `L`, then `/ d` (`Device.DivFull`). -/
theorem mean_stage {S mean0 : ℝ}
    (hS : |S - ∑ j ∈ Finset.range (2 ^ r.L), r.gl j| ≤
      ((1 + u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| +
        eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L)
    (hm : |mean0 - S / r.d| ≤ 2 ^ (-22 : ℤ) * |S / r.d| + 2 ^ (-148 : ℤ)) :
    |mean0 - r.mu| ≤ r.meanErr :=
  r.mean_stage' hS hm

/-- The centered lanes (line 76) are at most `2^34`. -/
theorem c_le {mean0 : ℝ} (hm0 : |mean0 - r.mu| ≤ r.meanErr) {c : ℕ → ℝ}
    (hc : ∀ j, j < r.d → |c j - (r.g j - mean0)| ≤ 3 * u32 * (|r.g j| + |mean0|) + 4 * eta32)
    (hc0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → c j = 0) :
    ∀ j, j < 2 ^ r.L → |c j| ≤ 2 ^ 34 :=
  r.c_le' hm0 hc hc0

/-- The sum of squares (line 77, a fusable tree of depth `L`) is at most `2^90`. -/
theorem Q_le {c q : ℕ → ℝ} {Q : ℝ} (hcb : ∀ j, j < 2 ^ r.L → |c j| ≤ 2 ^ 34)
    (hq : ∀ j, j < 2 ^ r.L → |q j - c j * c j| ≤ u32 * |c j * c j| + eta32)
    (hQ : |Q - ∑ j ∈ Finset.range (2 ^ r.L), q j| ≤
      ((1 + 3 * u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |q j| +
        4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L) :
    |Q| ≤ 2 ^ 90 :=
  r.Q_le' hcb hq hQ

/-- **The variance** (line 77): in `tl.sqrt`'s domain, and its square root within a bound of `sd`
(the lanes' errors enter through the triangle inequality of the Euclidean norm). -/
theorem var_stage (hok : r.Ok) {mean0 : ℝ} (hm0 : |mean0 - r.mu| ≤ r.meanErr) {c q : ℕ → ℝ}
    {Q var : ℝ}
    (hc : ∀ j, j < r.d → |c j - (r.g j - mean0)| ≤ 3 * u32 * (|r.g j| + |mean0|) + 4 * eta32)
    (hc0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → c j = 0)
    (hq : ∀ j, j < 2 ^ r.L → |q j - c j * c j| ≤ u32 * |c j * c j| + eta32)
    (hQ : |Q - ∑ j ∈ Finset.range (2 ^ r.L), q j| ≤
      ((1 + 3 * u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |q j| +
        4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L)
    (hv : |var - Q / r.d| ≤ 2 ^ (-22 : ℤ) * |Q / r.d| + 2 ^ (-148 : ℤ)) :
    2 ^ (-126 : ℤ) ≤ var ∧ var ≤ 2 ^ 80 ∧
      |Real.sqrt var - r.sd| ≤ ((r.L : ℝ) + 11) * u32 * r.aMean + 36 * u32 * r.sd :=
  r.var_stage' hok hm0 hc hc0 hq hQ hv

/-- **The standard deviation** (line 80, `tl.sqrt`, `Device.SqrtF32`). -/
theorem std_stage (hok : r.Ok) {var std32 : ℝ} (hvlo : 2 ^ (-126 : ℤ) ≤ var)
    (hsv : |Real.sqrt var - r.sd| ≤ ((r.L : ℝ) + 11) * u32 * r.aMean + 36 * u32 * r.sd)
    (hstd : |std32 - Real.sqrt var| ≤ 2 ^ (-22 : ℤ) * Real.sqrt var + 2 ^ (-149 : ℤ)) :
    |std32 - r.sd| ≤ r.stdErr :=
  r.std_stage' hok hvlo hsv hstd

/-- **The cutoff** (lines 79–82): the `bfloat16` roundings of the mean, the standard deviation,
their product with the multiplier and the sum. -/
theorem cut_stage (hok : r.Ok) {mean0 std32 m32 mb sb P scb Cs ck : ℝ}
    (hm0 : |mean0 - r.mu| ≤ r.meanErr) (hsd : |std32 - r.sd| ≤ r.stdErr)
    (hm32 : |m32 - r.m| ≤ u32 * |r.m| + eta32)
    (hmb : |mb - mean0| ≤ ub * |mean0| + etab)
    (hsb : |sb - std32| ≤ ub * |std32| + etab)
    (hP : |P - sb * m32| ≤ u32 * |sb * m32| + eta32)
    (hscb : |scb - P| ≤ ub * |P| + etab)
    (hCs : |Cs - (mb + scb)| ≤ u32 * |mb + scb| + eta32)
    (hck : |ck - Cs| ≤ ub * |Cs| + etab) :
    |ck - r.cRef| ≤ 7 / 8 * r.cutTol - 2 ^ (-131 : ℤ) :=
  r.cut_stage' hok hm0 hsd hm32 hmb hsb hP hscb hCs hck

end RowP

end Claims.GeluAndMulSparse
