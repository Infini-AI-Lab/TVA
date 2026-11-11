import Proof.NumericE

/-!
# `lora_shrink_bf16`: the error terms of an element, summed

The element is within the atomic additions' error, the scaled parts' errors with their roundings,
and the rounding of `scaling` of `scaling · dotSum`.
-/

namespace Claims.LoraShrink

open TVA

/-- `DotLoop.bound` grows with the magnitude. -/
theorem bound_mono_B {ρ α : ℚ} (hρ : 0 ≤ ρ) (t : ℕ) {B B' : ℚ} (hBB' : B ≤ B') :
    DotLoop.bound ρ α t B ≤ DotLoop.bound ρ α t B' := by
  unfold DotLoop.bound
  have h1 : 1 ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have := mul_le_mul_of_nonneg_left hBB' (by linarith : (0 : ℚ) ≤ (1 + ρ) ^ t - 1)
  linarith

/-- `vMag` in terms of `s32`. -/
theorem vMag_eq (d : Inst) (va vb : Nat → BF16) :
    vMag d va vb = |s32 d| * partMag d va vb +
      (Format.binary32.u * |s32 d| * partMag d va vb + (sk d : ℚ) * Format.binary32.eta) := rfl

/-- A sum of per-part magnitudes with their roundings. -/
theorem sum_scaled (n : ℕ) (a : ℕ → ℚ) (c u η : ℚ) :
    ∑ p ∈ Finset.range n, (c * a p + (u * c * a p + η)) =
      c * ∑ p ∈ Finset.range n, a p + (u * c * ∑ p ∈ Finset.range n, a p + n * η) := by
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
    nsmul_eq_mul]

/-- The magnitudes of the operands add up to at most `vMag`. -/
theorem sum_abs_le_vMag (d : Inst) (va vb : Nat → BF16) (v : Nat → ℚ)
    (hv : ∀ p, p < sk d →
      |v p| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p) :
    ∑ p ∈ Finset.range (sk d), |v p| ≤ vMag d va vb := by
  calc ∑ p ∈ Finset.range (sk d), |v p|
      ≤ ∑ p ∈ Finset.range (sk d),
        (|s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p) :=
        Finset.sum_le_sum fun p hp => hv p (Finset.mem_range.mp hp)
    _ = vMag d va vb := by
      rw [vMag_eq]
      exact sum_scaled _ (fun p => |partSum d va vb p| + errPart d va vb p) _ _ _

/-- A sum of per-part roundings. -/
theorem sum_mulErr_aux (n : ℕ) (a : ℕ → ℚ) (k η : ℚ) :
    ∑ p ∈ Finset.range n, (k * a p + η) = k * ∑ p ∈ Finset.range n, a p + n * η := by
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
    nsmul_eq_mul]

set_option maxHeartbeats 4000000 in
/-- **The error terms, summed** (before the numeric estimate): the element is within the atomic
additions' error (`A1·vMag + A2`), the parts' errors scaled (`σ·Σ errPart`) with their roundings,
and the rounding of `scaling`, of `scaling · dotSum`. -/
theorem error_sum (d : Inst) (hd : d.ok) (va vb : Nat → BF16) (v : Nat → ℚ)
    (hv : ∀ p, p < sk d →
      |v p - s32 d * partSum d va vb p| ≤ |s32 d| * errPart d va vb p + mulErr d va vb p ∧
      |v p| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p)
    (y : ℚ)
    (hy : |y - ∑ p ∈ Finset.range (sk d), v p| ≤
      DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (sk d) (∑ p ∈ Finset.range (sk d), |v p|)) :
    |y - d.scaling.toRat * dotSum d va vb| ≤
      102 / 100 * (sk d : ℚ) * 2 ^ (-23 : ℤ) * vMag d va vb +
        102 / 100 * (sk d : ℚ) * 2 ^ (-124 : ℤ) +
      (|s32 d| * ∑ p ∈ Finset.range (sk d), errPart d va vb p + mulTol d va vb) +
      (|s32 d| * Format.binary32.u + Format.binary32.eta) * dotMag d va vb := by
  have hAle := sum_abs_le_vMag d va vb v fun p hp => (hv p hp).2
  have hskle : (sk d : ℚ) ≤ 64 := by
    rcases cfg_cases d with h | h <;> rw [h.2.2.2.2] <;> norm_num
  have hx : (0 : ℚ) ≤ 2 ^ (-23 : ℤ) := by positivity
  have hn : (sk d : ℚ) * 2 ^ (-23 : ℤ) ≤ 1 / 100 := by
    have : (sk d : ℚ) * 2 ^ (-23 : ℤ) ≤ 64 * 2 ^ (-23 : ℤ) :=
      mul_le_mul_of_nonneg_right hskle hx
    refine this.trans ?_; norm_num
  obtain ⟨hp1, hp2⟩ := pow_close hx (sk d) hn
  have t1 : |y - ∑ p ∈ Finset.range (sk d), v p| ≤
      102 / 100 * (sk d : ℚ) * 2 ^ (-23 : ℤ) * vMag d va vb +
        102 / 100 * (sk d : ℚ) * 2 ^ (-124 : ℤ) := by
    refine hy.trans ((bound_mono_B (by positivity) (sk d) hAle).trans ?_)
    unfold DotLoop.bound
    have h1 : ((1 + 2 ^ (-23 : ℤ)) ^ sk d - 1) * vMag d va vb ≤
        102 / 100 * ((sk d : ℚ) * 2 ^ (-23 : ℤ)) * vMag d va vb :=
      mul_le_mul_of_nonneg_right hp2 (vMag_nonneg d va vb)
    have h2 : (sk d : ℚ) * (1 + 2 ^ (-23 : ℤ)) ^ sk d * 2 ^ (-124 : ℤ) ≤
        (sk d : ℚ) * (102 / 100) * 2 ^ (-124 : ℤ) :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hp1 (Nat.cast_nonneg _)) (by positivity)
    linarith
  have t2 : |∑ p ∈ Finset.range (sk d), v p - s32 d * dotSum d va vb| ≤
      |s32 d| * ∑ p ∈ Finset.range (sk d), errPart d va vb p + mulTol d va vb := by
    rw [dotSum_eq_parts d hd va vb, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ p ∈ Finset.range (sk d), |v p - s32 d * partSum d va vb p|
        ≤ ∑ p ∈ Finset.range (sk d), (|s32 d| * errPart d va vb p + mulErr d va vb p) :=
          Finset.sum_le_sum fun p hp => (hv p (Finset.mem_range.mp hp)).1
      _ = _ := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum]
        congr 1
        unfold mulErr mulTol partMag
        rw [sum_mulErr_aux]
        rfl
  have t4 : |s32 d * dotSum d va vb - d.scaling.toRat * dotSum d va vb| ≤
      (|s32 d| * Format.binary32.u + Format.binary32.eta) * dotMag d va vb := by
    rw [← sub_mul, abs_mul]
    exact mul_le_mul (scaling32_err d hd) (abs_dotSum_le d va vb) (abs_nonneg _)
      (by have := Format.u_nonneg (f := Format.binary32)
          have := Format.eta_nonneg (f := Format.binary32)
          positivity)
  calc |y - d.scaling.toRat * dotSum d va vb|
      ≤ |y - ∑ p ∈ Finset.range (sk d), v p| +
        |∑ p ∈ Finset.range (sk d), v p - s32 d * dotSum d va vb| +
        |s32 d * dotSum d va vb - d.scaling.toRat * dotSum d va vb| := by
        refine (abs_sub_le _ (∑ p ∈ Finset.range (sk d), v p) _).trans ?_
        have := abs_sub_le (∑ p ∈ Finset.range (sk d), v p) (s32 d * dotSum d va vb)
          (d.scaling.toRat * dotSum d va vb)
        linarith
    _ ≤ _ := by linarith

end Claims.LoraShrink
