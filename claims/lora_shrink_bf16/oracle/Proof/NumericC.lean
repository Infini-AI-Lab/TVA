import Proof.NumericB

/-!
# `lora_shrink_bf16`: closed forms of the step-by-step bound

`(1 + x)^n` for small `n·x`, the constants `rho` and `alpha` in the two launch configurations, the
number of steps of a part, and the sum of the parts' errors.
-/

namespace Claims.LoraShrink

open TVA

/-- `(1 + x)^n` stays within 2% of `1` and `(1 + x)^n - 1` within 2% of `n·x`, when `n·x ≤ 1/100`. -/
theorem pow_close {x : ℚ} (hx : 0 ≤ x) (n : ℕ) (h : (n : ℚ) * x ≤ 1 / 100) :
    (1 + x) ^ n ≤ 102 / 100 ∧ (1 + x) ^ n - 1 ≤ 102 / 100 * ((n : ℚ) * x) := by
  have h1 := DotLoop.pow_mul_le hx n (by linarith)
  have hP : 0 ≤ (1 + x) ^ n := by positivity
  have hn : 0 ≤ (n : ℚ) * x := mul_nonneg (Nat.cast_nonneg _) hx
  have h2 : (1 + x) ^ n ≤ 102 / 100 := by
    nlinarith [mul_nonneg hP (by linarith : (0 : ℚ) ≤ 1 / 100 - (n : ℚ) * x)]
  refine ⟨h2, ?_⟩
  nlinarith [mul_nonneg (by linarith : (0 : ℚ) ≤ 102 / 100 - (1 + x) ^ n) hn]

/-- `rho` in the two configurations. -/
theorem rho_le_num (d : Inst) :
    (d.M < 128 → rho d ≤ 259 * 2 ^ (-22 : ℤ)) ∧ (128 ≤ d.M → rho d ≤ 35 * 2 ^ (-22 : ℤ)) := by
  constructor
  · intro h
    have h1 : bk d = 256 := by simp [bk, h]
    unfold rho DotLoop.stepRel Device.relDot16
    rw [h1, Format.binary32_u]; norm_num
  · intro h
    have h1 : bk d = 32 := by simp [bk]; omega
    unfold rho DotLoop.stepRel Device.relDot16
    rw [h1, Format.binary32_u]; norm_num

/-- `alpha` is tiny in both configurations. -/
theorem alpha_le_num (d : Inst) : alpha d ≤ 2 ^ (-117 : ℤ) := by
  rcases cfg_cases d with h | h
  · unfold alpha DotLoop.stepAbs Device.absDot16
    rw [h.2.1, Format.binary32_u, Format.binary32_eta]; norm_num
  · unfold alpha DotLoop.stepAbs Device.absDot16
    rw [h.2.1, Format.binary32_u, Format.binary32_eta]; norm_num

/-- A part has at most as many steps as part `0`, which has at least one, and at most `8`
(`BLOCK_K = 256`) or `512` (`BLOCK_K = 32`) when `K ≤ 2^17`. -/
theorem steps_le_zero (d : Inst) (p : Nat) : stepsOf d p ≤ stepsOf d 0 := by
  unfold stepsOf
  exact Nat.div_le_div_right (by omega)

/-- Part `0` has at least one trip when `K > 0`. -/
theorem steps_pos (d : Inst) (hK : 0 < d.K) : 0 < stepsOf d 0 := by
  have := bk_pos d
  unfold stepsOf blocks
  have h : 1 ≤ (d.K + bk d - 1) / bk d := by
    rw [Nat.le_div_iff_mul_le (bk_pos d)]; omega
  have := sk_pos d
  rw [Nat.pos_iff_ne_zero]
  intro h0
  have := Nat.div_eq_zero_iff.mp h0
  omega

/-- Part `0` has at most `512` trips when `K ≤ 2^17`. -/
theorem steps_zero_le (d : Inst) (hK : d.K ≤ 2 ^ 17) :
    (d.M < 128 → stepsOf d 0 ≤ 8) ∧ (128 ≤ d.M → stepsOf d 0 ≤ 512) := by
  constructor
  · intro h
    have h1 : bk d = 256 := by simp [bk, h]
    have h2 : sk d = 64 := by simp [sk, h]
    unfold stepsOf blocks; rw [h1, h2]; omega
  · intro h
    have h1 : bk d = 32 := by simp [bk]; omega
    have h2 : sk d = 8 := by simp [sk]; omega
    unfold stepsOf blocks; rw [h1, h2]; omega

/-- The step-by-step bound of a part is at most `1.02·T₀·ρ·L + 1.02·T₀·α`, `T₀` the steps of part
`0`, when `T₀·ρ ≤ 1/100`. -/
theorem errStep_le (d : Inst) (va vb : Nat → BF16) (p : Nat)
    (hT : (stepsOf d 0 : ℚ) * rho d ≤ 1 / 100) :
    errStep d va vb p ≤ 102 / 100 * (stepsOf d 0 : ℚ) * rho d * partLift d va vb p +
      102 / 100 * (stepsOf d 0 : ℚ) * alpha d := by
  have hρ := rho_nonneg d
  have hα := alpha_nonneg d
  have hL := partLift_nonneg d va vb p
  have hle : (stepsOf d p : ℚ) ≤ stepsOf d 0 := by exact_mod_cast steps_le_zero d p
  have hTp : (stepsOf d p : ℚ) * rho d ≤ 1 / 100 :=
    (mul_le_mul_of_nonneg_right hle hρ).trans hT
  obtain ⟨h1, h2⟩ := pow_close hρ (stepsOf d p) hTp
  unfold errStep DotLoop.bound
  have hT0 : (0 : ℚ) ≤ stepsOf d 0 := Nat.cast_nonneg _
  have hTp0 : (0 : ℚ) ≤ stepsOf d p := Nat.cast_nonneg _
  have e1 : ((1 + rho d) ^ stepsOf d p - 1) * partLift d va vb p ≤
      102 / 100 * (stepsOf d 0 : ℚ) * rho d * partLift d va vb p := by
    have : (102 / 100 : ℚ) * ((stepsOf d p : ℚ) * rho d) ≤ 102 / 100 * ((stepsOf d 0 : ℚ) * rho d) := by
      have := mul_le_mul_of_nonneg_right hle hρ; linarith
    calc ((1 + rho d) ^ stepsOf d p - 1) * partLift d va vb p
        ≤ 102 / 100 * ((stepsOf d p : ℚ) * rho d) * partLift d va vb p :=
          mul_le_mul_of_nonneg_right h2 hL
      _ ≤ 102 / 100 * ((stepsOf d 0 : ℚ) * rho d) * partLift d va vb p :=
          mul_le_mul_of_nonneg_right this hL
      _ = _ := by ring
  have e2 : (stepsOf d p : ℚ) * (1 + rho d) ^ stepsOf d p * alpha d ≤
      102 / 100 * (stepsOf d 0 : ℚ) * alpha d := by
    have : (stepsOf d p : ℚ) * (1 + rho d) ^ stepsOf d p ≤ (stepsOf d 0 : ℚ) * (102 / 100) :=
      mul_le_mul hle h1 (by positivity) hT0
    calc (stepsOf d p : ℚ) * (1 + rho d) ^ stepsOf d p * alpha d
        ≤ (stepsOf d 0 : ℚ) * (102 / 100) * alpha d := mul_le_mul_of_nonneg_right this hα
      _ = _ := by ring
  linarith

/-- **The sum of the parts' errors**: at most `1.02·T₀·ρ·dotMag + 1.02·SPLIT_K·T₀·α`. -/
theorem sum_errPart_le (d : Inst) (hd : d.ok) (va vb : Nat → BF16)
    (hT : (stepsOf d 0 : ℚ) * rho d ≤ 1 / 100) :
    ∑ p ∈ Finset.range (sk d), errPart d va vb p ≤
      102 / 100 * (stepsOf d 0 : ℚ) * rho d * dotMag d va vb +
        102 / 100 * (sk d : ℚ) * (stepsOf d 0 : ℚ) * alpha d := by
  calc ∑ p ∈ Finset.range (sk d), errPart d va vb p
      ≤ ∑ p ∈ Finset.range (sk d), (102 / 100 * (stepsOf d 0 : ℚ) * rho d * partLift d va vb p +
        102 / 100 * (stepsOf d 0 : ℚ) * alpha d) :=
        Finset.sum_le_sum fun p _ => (min_le_right _ _).trans (errStep_le d va vb p hT)
    _ = _ := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_partLift d hd]
      simp [Finset.sum_const]; ring

end Claims.LoraShrink
