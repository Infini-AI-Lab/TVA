import Claim
import TVA.Proof
import Proof.Base

/-!
# Numeric facts for the `ssd_bmm_chunk_bf16` proof

The tolerance relaxation of `DotLoop.bound`, the room estimate for `DotLift.dotAdd_step`, and the
fact that `zf32` is the `float32` `+0`.
-/

namespace Claims.SsdBmm

open TVA

/-- `zf32` is the `float32` `+0` pattern. -/
theorem zf32_eq : zf32 = (FloatV.ofPat 0 : F32) := by
  unfold zf32 FloatV.litF32
  have h : (FloatV.ofPat 0 : F64).isZero = true := by
    rw [FloatV.isZero_ofPat (Nat.two_pow_pos _)]
    exact Format.isZero_zero
  simp [h]

/-- The loop's trip count is at most `32` (`K ≤ 2^10`). -/
theorem steps_le {K : Nat} (h : K ≤ 2 ^ 10) : steps K ≤ 32 := by
  unfold steps BK
  rw [Nat.div_le_iff_le_mul_add_pred (by norm_num)]
  omega

/-- `ρ = stepRel relDot16 32` is nonnegative. -/
theorem rho_nonneg : 0 ≤ DotLoop.stepRel Device.relDot16 32 := by
  unfold DotLoop.stepRel Device.relDot16
  norm_num [Format.binary32_u]

/-- `α = stepAbs absDot16 32` is at most `2^-119`. -/
theorem alpha_le : DotLoop.stepAbs Device.absDot16 32 ≤ 2 ^ (-119 : ℤ) := by
  unfold DotLoop.stepAbs Device.absDot16
  norm_num [Format.binary32_u, Format.binary32_eta]

/-- The fine bound relaxes to the tolerance: for `T ≤ 32` steps and `B ≥ 0`,
`bound ρ α T B ≤ T/2^16·B + 2^-112`. -/
theorem bound_le_tol {T : Nat} (hT : T ≤ 32) {B : ℚ} (hB : 0 ≤ B) :
    DotLoop.bound (DotLoop.stepRel Device.relDot16 32) (DotLoop.stepAbs Device.absDot16 32) T B
      ≤ (T : ℚ) / 2 ^ 16 * B + 1 / 2 ^ 112 := by
  have hρ0 : 0 ≤ DotLoop.stepRel Device.relDot16 32 := rho_nonneg
  have hαle : DotLoop.stepAbs Device.absDot16 32 ≤ 2 ^ (-119 : ℤ) := alpha_le
  have hstep : ∀ n : ℕ, n ≤ 32 →
      (1 + DotLoop.stepRel Device.relDot16 32) ^ n ≤ 1 + (n : ℚ) * 2 ^ (-16 : ℤ) := by
    intro n
    induction n with
    | zero => intro _; norm_num
    | succ n ih =>
      intro hn
      have hn31 : n ≤ 31 := by omega
      have ihn : (1 + DotLoop.stepRel Device.relDot16 32) ^ n ≤ 1 + (n : ℚ) * 2 ^ (-16 : ℤ) :=
        ih (by omega)
      have hb : DotLoop.stepRel Device.relDot16 32 * (1 + (n : ℚ) * 2 ^ (-16 : ℤ)) ≤ 2 ^ (-16 : ℤ) := by
        have hnle : (n : ℚ) * 2 ^ (-16 : ℤ) ≤ 31 * 2 ^ (-16 : ℤ) := by
          have : (n : ℚ) ≤ 31 := by exact_mod_cast hn31
          exact mul_le_mul_of_nonneg_right this (by norm_num)
        have h1 : 1 + (n : ℚ) * 2 ^ (-16 : ℤ) ≤ 1 + 31 * 2 ^ (-16 : ℤ) := by linarith
        have h2 : DotLoop.stepRel Device.relDot16 32 * (1 + 31 * 2 ^ (-16 : ℤ)) ≤ 2 ^ (-16 : ℤ) := by
          unfold DotLoop.stepRel Device.relDot16
          norm_num [Format.binary32_u]
        have := mul_le_mul_of_nonneg_left h1 hρ0
        linarith
      have hmain : (1 + DotLoop.stepRel Device.relDot16 32) ^ (n + 1)
          ≤ 1 + ((n : ℚ) + 1) * 2 ^ (-16 : ℤ) := by
        rw [pow_succ]
        calc (1 + DotLoop.stepRel Device.relDot16 32) ^ n * (1 + DotLoop.stepRel Device.relDot16 32)
            ≤ (1 + (n : ℚ) * 2 ^ (-16 : ℤ)) * (1 + DotLoop.stepRel Device.relDot16 32) :=
              mul_le_mul_of_nonneg_right ihn (by linarith)
          _ = 1 + ((n : ℚ) + 1) * 2 ^ (-16 : ℤ) +
                (DotLoop.stepRel Device.relDot16 32 * (1 + (n : ℚ) * 2 ^ (-16 : ℤ)) - 2 ^ (-16 : ℤ)) := by
              ring
          _ ≤ 1 + ((n : ℚ) + 1) * 2 ^ (-16 : ℤ) := by linarith
      push_cast
      exact hmain
  have hpowT := hstep T hT
  have hTq : (T : ℚ) ≤ 32 := by exact_mod_cast hT
  have hpow2 : (1 + DotLoop.stepRel Device.relDot16 32) ^ T ≤ 2 := by
    have hc : (32 : ℚ) * 2 ^ (-16 : ℤ) ≤ 1 := by norm_num
    have hTle : (T : ℚ) * 2 ^ (-16 : ℤ) ≤ 32 * 2 ^ (-16 : ℤ) :=
      mul_le_mul_of_nonneg_right hTq (by norm_num)
    linarith
  have h1 : ((1 + DotLoop.stepRel Device.relDot16 32) ^ T - 1) * B
      ≤ (T : ℚ) * 2 ^ (-16 : ℤ) * B := by
    have : (1 + DotLoop.stepRel Device.relDot16 32) ^ T - 1 ≤ (T : ℚ) * 2 ^ (-16 : ℤ) := by
      linarith
    exact mul_le_mul_of_nonneg_right this hB
  have h2 : (T : ℚ) * (1 + DotLoop.stepRel Device.relDot16 32) ^ T *
      DotLoop.stepAbs Device.absDot16 32 ≤ 2 ^ (-112 : ℤ) := by
    have hfac0 : (0 : ℚ) ≤ (T : ℚ) * (1 + DotLoop.stepRel Device.relDot16 32) ^ T :=
      mul_nonneg (Nat.cast_nonneg T) (pow_nonneg (by linarith) T)
    have hp0 : (0:ℚ) ≤ (1 + DotLoop.stepRel Device.relDot16 32) ^ T :=
      pow_nonneg (by linarith) T
    have hfac2 : (T : ℚ) * (1 + DotLoop.stepRel Device.relDot16 32) ^ T ≤ 64 := by
      have h := mul_le_mul hTq hpow2 hp0 (by norm_num : (0:ℚ) ≤ 32)
      norm_num at h ⊢
      exact h
    calc (T : ℚ) * (1 + DotLoop.stepRel Device.relDot16 32) ^ T * DotLoop.stepAbs Device.absDot16 32
        ≤ (T : ℚ) * (1 + DotLoop.stepRel Device.relDot16 32) ^ T * 2 ^ (-119 : ℤ) :=
          mul_le_mul_of_nonneg_left hαle hfac0
      _ ≤ 64 * 2 ^ (-119 : ℤ) := mul_le_mul_of_nonneg_right hfac2 (by norm_num)
      _ = 2 ^ (-113 : ℤ) := by norm_num
      _ ≤ 2 ^ (-112 : ℤ) := by norm_num
  have heq : (T : ℚ) / 2 ^ 16 * B = (T : ℚ) * 2 ^ (-16 : ℤ) * B := by
    rw [show (2:ℚ) ^ (-16 : ℤ) = 1 / 2 ^ 16 by norm_num]
    ring
  have h112 : (1 : ℚ) / 2 ^ 112 = 2 ^ (-112 : ℤ) := by norm_num
  unfold DotLoop.bound
  rw [heq, h112]
  linarith

/-- The room estimate `dotAdd_step` asks for, from `|acc| ≤ 2^121` and `A ≤ 2^120`. -/
theorem room_le {vc A : ℚ} (hvc : |vc| ≤ 2 ^ 121) (hA : A ≤ 2 ^ 120) :
    2 * ((1 + Device.relDot16 32) * (|vc| + A) + Device.absDot16 32 +
        2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
  have hsum : |vc| + A ≤ 2 ^ 121 + 2 ^ 120 := by linarith
  have hρ0 : (0:ℚ) ≤ 1 + Device.relDot16 32 := by unfold Device.relDot16; norm_num
  have hρ2 : 1 + Device.relDot16 32 ≤ 2 := by unfold Device.relDot16; norm_num
  have hS : (0:ℚ) ≤ 2 ^ 121 + 2 ^ 120 := by norm_num
  have h1 : (1 + Device.relDot16 32) * (|vc| + A) ≤ 2 * (2 ^ 121 + 2 ^ 120) := by
    calc (1 + Device.relDot16 32) * (|vc| + A)
        ≤ (1 + Device.relDot16 32) * (2 ^ 121 + 2 ^ 120) := mul_le_mul_of_nonneg_left hsum hρ0
      _ ≤ 2 * (2 ^ 121 + 2 ^ 120) := mul_le_mul_of_nonneg_right hρ2 hS
  have h2 : 2 * ((1 + Device.relDot16 32) * (|vc| + A)) ≤ 2 * (2 * (2^121+2^120)) := by
    linarith
  have hgoal : 2 * ((1 + Device.relDot16 32) * (|vc| + A) + Device.absDot16 32 +
        2 * Format.binary32.eta)
      ≤ (2:ℚ) * (2 * (2^121+2^120) + 34*2^(-126:ℤ) + 2*2^(-150:ℤ)) := by
    rw [show Device.absDot16 32 = 34 * 2^(-126:ℤ) from by unfold Device.absDot16; norm_num,
        show (2:ℚ) * Format.binary32.eta = 2 * 2^(-150:ℤ) from by rw [Format.binary32_eta]]
    nlinarith [h2]
  have hfin : (2:ℚ) * (2 * (2^121+2^120) + 34*2^(-126:ℤ) + 2*2^(-150:ℤ))
      ≤ ((2:ℚ)^24-1)*2^104 := by norm_num
  rw [Format.binary32_maxVal]
  linarith [hgoal, hfin]

end Claims.SsdBmm



