import Claim
import TVA.Proof

/-! The real-valued constants and small facts shared by the tolerance lemmas of the layer-norm
row (`meanTol`, `rstdTol`, `yTol`). -/

namespace Claims.LayerNormFwd

open TVA
open scoped BigOperators

noncomputable section

/-- `float32`'s unit roundoff, `u = 2^-24`. -/
def u32 : ℝ := 2 ^ (-24 : ℤ)

/-- `float32`'s smallest subnormal step, `η = 2^-150`. -/
def eta32 : ℝ := 2 ^ (-150 : ℤ)

/-- `bfloat16`'s unit roundoff, `u16 = 2^-8`. -/
def u16 : ℝ := 2 ^ (-8 : ℤ)

/-- `bfloat16`'s smallest subnormal step, `η16 = 2^-134`. -/
def eta16 : ℝ := 2 ^ (-134 : ℤ)

/-- The float32 unit roundoff `u32` is positive. -/
theorem u32_pos : 0 < u32 := by unfold u32; norm_num

/-- The float32 underflow constant `eta32` is positive. -/
theorem eta32_pos : 0 < eta32 := by unfold eta32; norm_num

/-- The 16-bit unit roundoff `u16` is positive. -/
theorem u16_pos : 0 < u16 := by unfold u16; norm_num

/-- The 16-bit underflow constant `eta16` is positive. -/
theorem eta16_pos : 0 < eta16 := by unfold eta16; norm_num

/-- `u32` is at most 1/16. -/
theorem u32_le_small : u32 ≤ 1 / 16 := by unfold u32; norm_num

/-- `eta32` is at most `u32`. -/
theorem eta32_le_u32 : eta32 ≤ u32 := by unfold eta32 u32; norm_num

/-- `u32` is at most `u16`. -/
theorem u32_le_u16 : u32 ≤ u16 := by unfold u32 u16; norm_num

/-- For an accepted instance the real epsilon `epsR` is positive. -/
theorem epsR_pos {d : Inst} (hd : d.ok) : 0 < epsR d := by
  have h1 : ((2 : ℚ) ^ (-125 : ℤ) : ℝ) ≤ (d.eps.toRat : ℝ) := by
    exact_mod_cast hd.eps_ge
  have h2 : (0 : ℝ) < ((2 : ℚ) ^ (-125 : ℤ) : ℝ) := by norm_num
  unfold epsR FloatV.toReal FloatV.toRat
  exact lt_of_lt_of_le h2 h1

/-- For an accepted instance `epsR` is at least `2^(-125)`. -/
theorem epsR_ge {d : Inst} (hd : d.ok) : (2 : ℝ) ^ (-125 : ℤ) ≤ epsR d := by
  have h1 : ((2 : ℚ) ^ (-125 : ℤ) : ℝ) ≤ (d.eps.toRat : ℝ) := by
    exact_mod_cast hd.eps_ge
  exact h1

/-- For an accepted instance `epsR` is at most `2^126`. -/
theorem epsR_le {d : Inst} (hd : d.ok) : epsR d ≤ (2 : ℝ) ^ (126 : ℤ) := by
  have h1 : (d.eps.toRat : ℝ) ≤ ((2 : ℚ) ^ (126 : ℤ) : ℝ) := by
    exact_mod_cast hd.eps_le
  exact h1

/-- The exact mean is no larger in magnitude than the mean magnitude. -/
theorem abs_meanRef_le_absMean {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    |meanRef d init t| ≤ absMean d init t := by
  have hD' : (0 : ℝ) < d.D := by exact_mod_cast hD
  unfold meanRef absMean
  rw [abs_div, abs_of_pos hD']
  gcongr
  exact Finset.abs_sum_le_sum_abs _ _

/-- The deviations `s j − meanRef` sum to zero. -/
theorem sum_dev_zero {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t)) = 0 := by
  have hD' : ((d.D : ℝ)) ≠ 0 := by
    have : (0 : ℝ) < d.D := by exact_mod_cast hD
    exact ne_of_gt this
  have hcancel : (d.D : ℝ) * ((∑ j ∈ Finset.range d.D, sR d init t j) / (d.D : ℝ))
      = ∑ j ∈ Finset.range d.D, sR d init t j := by
    field_simp
  have hconst : (∑ _j ∈ Finset.range d.D, meanRef d init t)
      = (d.D : ℝ) * meanRef d init t := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hmean : meanRef d init t = (∑ j ∈ Finset.range d.D, sR d init t j) / (d.D : ℝ) := rfl
  rw [Finset.sum_sub_distrib, hconst, hmean, hcancel, sub_self]

end

end Claims.LayerNormFwd
