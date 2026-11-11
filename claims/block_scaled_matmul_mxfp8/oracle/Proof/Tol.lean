import Claim
import TVA.Proof
import Proof.Basic

/-!
# The tolerance of the loop: the step recursion, its closed form and its relaxation

`tol p T` is the accumulator's error after `T` steps of `tl.dot_scaled` of `128` products each;
the loop has `T = K / 128` steps and the exact result is `partSum p (K / 128)`.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The sum of the products of step `t` (the elements `128 t ≤ e < 128 (t + 1)`). -/
def chunkSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range 128, p (128 * t + e)

/-- The chunk sum of `0` is the sum over the first `128` terms. -/
theorem chunkSum_zero (p : Nat → ℚ) : chunkSum p 0 = ∑ e ∈ Finset.range 128, p e := by
  simp [chunkSum]

/-- `partSum` after one more step. -/
theorem partSum_succ (p : Nat → ℚ) (t : Nat) :
    partSum p (t + 1) = partSum p t + chunkSum p t := by
  simp only [partSum, chunkSum, Nat.mul_succ, Finset.sum_range_add]

/-- The magnitude of a chunk is at least the magnitude of its sum. -/
theorem abs_chunkSum_le (p : Nat → ℚ) (t : Nat) : |chunkSum p t| ≤ chunkAbs p t := by
  simp only [chunkSum, chunkAbs]
  exact Finset.abs_sum_le_sum_abs _ _

/-- `stepMag` after one more step. -/
theorem stepMag_succ (p : Nat → ℚ) (t : Nat) :
    stepMag p (t + 1) = stepMag p t + (|partSum p t| + chunkAbs p t) := by
  simp only [stepMag, Finset.sum_range_succ]

/-- `stepMag` is nonnegative. -/
theorem stepMag_nonneg (p : Nat → ℚ) (T : Nat) : 0 ≤ stepMag p T := by
  simp only [stepMag]
  refine Finset.sum_nonneg (fun t _ => ?_)
  have h1 : (0:ℚ) ≤ |partSum p t| := abs_nonneg _
  have h2 : (0:ℚ) ≤ chunkAbs p t := by
    simp only [chunkAbs]
    exact Finset.sum_nonneg (fun e _ => abs_nonneg _)
  linarith

/-- `rho` is positive. -/
theorem rho_pos : 0 < rho := by
  unfold rho Device.relDot16
  norm_num

/-- `alpha` is nonnegative. -/
theorem alpha_nonneg : 0 ≤ alpha := by
  unfold alpha Device.absDot16
  positivity

/-- The closed form of the step recursion: each step's error is relative, so the error is bounded
by `(1 + rho)^T` times the total magnitude. -/
theorem tol_bound (p : Nat → ℚ) (T : Nat) :
    tol p T ≤ (1 + rho) ^ T * (rho * stepMag p T + (T : ℚ) * alpha) := by
  induction T with
  | zero => simp [tol, stepMag]
  | succ t ih =>
    have hstep : tol p (t + 1) =
        (1 + rho) * tol p t + rho * (|partSum p t| + chunkAbs p t) + alpha := by
      simp only [tol]
      ring
    have hA : 0 ≤ |partSum p t| + chunkAbs p t := by
      have h1 : (0:ℚ) ≤ |partSum p t| := abs_nonneg _
      have h2 : (0:ℚ) ≤ chunkAbs p t := by simp only [chunkAbs]; positivity
      linarith
    have hSMA : stepMag p (t + 1) = stepMag p t + (|partSum p t| + chunkAbs p t) :=
      stepMag_succ p t
    have hρ : (0:ℚ) ≤ rho := le_of_lt rho_pos
    have hα : (0:ℚ) ≤ alpha := alpha_nonneg
    have h1ρ : (0:ℚ) ≤ 1 + rho := by linarith
    have hmul : (1 + rho) * tol p t ≤
        (1 + rho) * ((1 + rho) ^ t * (rho * stepMag p t + (t : ℚ) * alpha)) :=
      mul_le_mul_of_nonneg_left ih h1ρ
    have hkey : (1 + rho) * ((1 + rho) ^ t * (rho * stepMag p t + (t : ℚ) * alpha)) =
        (1 + rho) ^ (t + 1) * (rho * stepMag p t + (t : ℚ) * alpha) := by
      rw [pow_succ]
      ring
    have hP1 : (0:ℚ) ≤ (1 + rho) ^ (t + 1) - 1 := by
      have h1 : (1:ℚ) ≤ 1 + rho := by linarith
      have h2 : (1:ℚ) ≤ (1 + rho) ^ (t + 1) := one_le_pow₀ h1
      linarith
    have hPA : (0:ℚ) ≤ rho * (|partSum p t| + chunkAbs p t) := mul_nonneg hρ hA
    have hsum : (0:ℚ) ≤ rho * (|partSum p t| + chunkAbs p t) + alpha := by linarith
    have hprod : (0:ℚ) ≤ ((1 + rho) ^ (t + 1) - 1) *
        (rho * (|partSum p t| + chunkAbs p t) + alpha) := mul_nonneg hP1 hsum
    rw [hstep, hSMA]
    push_cast
    nlinarith [hmul, hkey, hprod]

/-- **The relaxation to `vTol`**: for at most `2^10` steps the error is at most
`2^-14 · mu + 2^-108`. -/
theorem tol_le_vTol (p : Nat → ℚ) {T : Nat} (hT : T ≤ 2 ^ 10) :
    tol p T ≤ 2 ^ (-14 : ℤ) * stepMag p T + 2 ^ (-108 : ℤ) := by
  have hT1024 : T ≤ 1024 := by
    have h : (2:Nat) ^ 10 = 1024 := by norm_num
    simpa only [h] using hT
  have hTq : (T:ℚ) ≤ 1024 := by exact_mod_cast hT1024
  have hfac : 1 + 2 * (T:ℚ) * rho ≤ 1089/1024 := by
    unfold rho Device.relDot16
    norm_num
    nlinarith
  have h1024 : (1024:ℚ) * rho ≤ 1/2 := by
    unfold rho Device.relDot16
    norm_num
  have hTrho : (T:ℚ) * rho ≤ 1/2 :=
    le_trans (mul_le_mul_of_nonneg_right hTq (le_of_lt rho_pos)) h1024
  have hpow := DotLoop.pow_le (le_of_lt rho_pos) T hTrho
  have hX : 0 ≤ rho * stepMag p T + (T:ℚ) * alpha := by
    have h1 : (0:ℚ) ≤ rho * stepMag p T := mul_nonneg (le_of_lt rho_pos) (stepMag_nonneg p T)
    have h2 : (0:ℚ) ≤ (T:ℚ) * alpha := mul_nonneg (Nat.cast_nonneg T) alpha_nonneg
    linarith
  have hA : (1089/1024) * rho ≤ 2^(-14:ℤ) := by
    unfold rho Device.relDot16
    norm_num
  have hB : (1089/1024) * ((1024:ℚ) * alpha) ≤ 2^(-108:ℤ) := by
    unfold alpha Device.absDot16
    norm_num
  have hTa : (T:ℚ) * alpha ≤ 1024 * alpha := mul_le_mul_of_nonneg_right hTq alpha_nonneg
  have hSM : 0 ≤ stepMag p T := stepMag_nonneg p T
  have s3 : (1089/1024) * rho * stepMag p T ≤ 2^(-14:ℤ) * stepMag p T :=
    mul_le_mul_of_nonneg_right hA hSM
  have s4 : (1089/1024) * ((T:ℚ) * alpha) ≤ 2^(-108:ℤ) := by
    have h1 : (1089/1024) * ((T:ℚ) * alpha) ≤ (1089/1024) * ((1024:ℚ) * alpha) :=
      mul_le_mul_of_nonneg_left hTa (by norm_num)
    linarith
  calc tol p T
      ≤ (1 + rho) ^ T * (rho * stepMag p T + (T:ℚ) * alpha) := tol_bound p T
    _ ≤ (1 + 2 * (T:ℚ) * rho) * (rho * stepMag p T + (T:ℚ) * alpha) :=
        mul_le_mul_of_nonneg_right hpow hX
    _ ≤ (1089/1024) * (rho * stepMag p T + (T:ℚ) * alpha) :=
        mul_le_mul_of_nonneg_right hfac hX
    _ = (1089/1024) * rho * stepMag p T + (1089/1024) * ((T:ℚ) * alpha) := by ring
    _ ≤ 2^(-14:ℤ) * stepMag p T + 2^(-108:ℤ) := by linarith [s3, s4]

/-- **The relaxation for the store**: for at most `2^10` steps, `(1 + 2^-11)` times the error is
still below `2^-14 · mu + 2^-25`. -/
theorem tol_le_cTol (p : Nat → ℚ) {T : Nat} (hT : T ≤ 2 ^ 10) :
    (1 + 2 ^ (-11 : ℤ)) * tol p T ≤ 2 ^ (-14 : ℤ) * stepMag p T + 2 ^ (-25 : ℤ) := by
  have hT1024 : T ≤ 1024 := by
    have h : (2:Nat) ^ 10 = 1024 := by norm_num
    simpa only [h] using hT
  have hTq : (T:ℚ) ≤ 1024 := by exact_mod_cast hT1024
  have hfac : 1 + 2 * (T:ℚ) * rho ≤ 1089/1024 := by
    unfold rho Device.relDot16
    norm_num
    nlinarith
  have h1024 : (1024:ℚ) * rho ≤ 1/2 := by
    unfold rho Device.relDot16
    norm_num
  have hTrho : (T:ℚ) * rho ≤ 1/2 :=
    le_trans (mul_le_mul_of_nonneg_right hTq (le_of_lt rho_pos)) h1024
  have hpow := DotLoop.pow_le (le_of_lt rho_pos) T hTrho
  have hX : 0 ≤ rho * stepMag p T + (T:ℚ) * alpha := by
    have h1 : (0:ℚ) ≤ rho * stepMag p T := mul_nonneg (le_of_lt rho_pos) (stepMag_nonneg p T)
    have h2 : (0:ℚ) ≤ (T:ℚ) * alpha := mul_nonneg (Nat.cast_nonneg T) alpha_nonneg
    linarith
  have hc : (0:ℚ) ≤ 1 + 2 ^ (-11:ℤ) := by norm_num
  have hK : (0:ℚ) ≤ (1 + 2 ^ (-11:ℤ)) * (1089/1024) := by norm_num
  have hA : (1 + 2 ^ (-11:ℤ)) * (1089/1024) * rho ≤ 2^(-14:ℤ) := by
    unfold rho Device.relDot16
    norm_num
  have hB : (1 + 2 ^ (-11:ℤ)) * (1089/1024) * ((1024:ℚ) * alpha) ≤ 2^(-25:ℤ) := by
    unfold alpha Device.absDot16
    norm_num
  have hTa : (T:ℚ) * alpha ≤ 1024 * alpha := mul_le_mul_of_nonneg_right hTq alpha_nonneg
  have hSM : 0 ≤ stepMag p T := stepMag_nonneg p T
  have s3 : (1 + 2 ^ (-11:ℤ)) * (1089/1024) * (rho * stepMag p T) ≤
      2^(-14:ℤ) * stepMag p T := by
    have h := mul_le_mul_of_nonneg_right hA hSM
    linarith [h]
  have s4 : (1 + 2 ^ (-11:ℤ)) * (1089/1024) * ((T:ℚ) * alpha) ≤ 2^(-25:ℤ) := by
    have h1 : (1 + 2 ^ (-11:ℤ)) * (1089/1024) * ((T:ℚ) * alpha) ≤
        (1 + 2 ^ (-11:ℤ)) * (1089/1024) * ((1024:ℚ) * alpha) :=
      mul_le_mul_of_nonneg_left hTa hK
    linarith
  calc (1 + 2 ^ (-11:ℤ)) * tol p T
      ≤ (1 + 2 ^ (-11:ℤ)) *
          ((1 + rho) ^ T * (rho * stepMag p T + (T:ℚ) * alpha)) :=
        mul_le_mul_of_nonneg_left (tol_bound p T) hc
    _ ≤ (1 + 2 ^ (-11:ℤ)) *
          ((1 + 2 * (T:ℚ) * rho) * (rho * stepMag p T + (T:ℚ) * alpha)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hpow hX) hc
    _ ≤ (1 + 2 ^ (-11:ℤ)) *
          ((1089/1024) * (rho * stepMag p T + (T:ℚ) * alpha)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hfac hX) hc
    _ = (1 + 2 ^ (-11:ℤ)) * (1089/1024) * (rho * stepMag p T) +
          (1 + 2 ^ (-11:ℤ)) * (1089/1024) * ((T:ℚ) * alpha) := by ring
    _ ≤ 2^(-14:ℤ) * stepMag p T + 2^(-25:ℤ) := by linarith [s3, s4]

/-- The exact result is the partial sum after `K / 128` steps. -/
theorem partSum_ref {d : Inst} (hd : d.ok) (p : Nat → ℚ) :
    partSum p (d.K / 128) = dotSum p d.K := by
  have h := K_div d hd
  simp only [partSum, dotSum]
  rw [← h]

/-- **The store's rounding**: a finite `v` within `tol p T` of the exact result `ord`, rounded to
`float16`, is finite and within `cTol = 2^-11·|ord| + 2^-14·mu + 2^-24` of it, when `|ord| + tol`
leaves room in `float16`. -/
theorem cvt_bound (p : Nat → ℚ) {ord : ℚ} {T : Nat} {v : F32} {c : F16}
    (hT : T ≤ 2 ^ 10) (hvfin : v.isFinite = true) (hv : |v.toRat - ord| ≤ tol p T)
    (hc : FloatV.NaNEq (FloatV.cvt v) c)
    (hroom : |ord| + tol p T ≤ Format.binary16.maxVal) :
    c.isFinite = true ∧
      |c.toRat - ord| ≤ 2 ^ (-11 : ℤ) * |ord| + 2 ^ (-14 : ℤ) * stepMag p T + 2 ^ (-24 : ℤ) := by
  have hvabs : |v.toRat| ≤ |ord| + tol p T := by
    have e : (v.toRat - ord) + ord = v.toRat := by ring
    have h := abs_add_le (v.toRat - ord) ord
    rw [e] at h
    linarith
  have hmax : |v.toRat| ≤ Format.binary16.maxVal := le_trans hvabs hroom
  have hcv : (FloatV.cvt v : F16).isFinite = true ∧
      |(FloatV.cvt v : F16).toRat - v.toRat| ≤
        |v.toRat| * Format.binary16.u + Format.binary16.eta :=
    FloatV.cvt_err Format.binary16_regular hvfin hmax
  have hnan : (FloatV.cvt v : F16).isNaN = false :=
    Format.not_isNaN_of_isFinite hcv.1
  have hceq : c = (FloatV.cvt v : F16) := FloatV.NaNEq.eq hc hnan
  have hcfin : c.isFinite = true := by rw [hceq]; exact hcv.1
  have herr : |c.toRat - v.toRat| ≤ |v.toRat| * 2 ^ (-11 : ℤ) + 2 ^ (-25 : ℤ) := by
    rw [hceq]
    have h2 := hcv.2
    rw [Format.binary16_u, Format.binary16_eta] at h2
    exact h2
  refine ⟨hcfin, ?_⟩
  have h1 : |c.toRat - ord| ≤ |c.toRat - v.toRat| + |v.toRat - ord| := by
    have e : (c.toRat - v.toRat) + (v.toRat - ord) = c.toRat - ord := by ring
    have h := abs_add_le (c.toRat - v.toRat) (v.toRat - ord)
    rw [e] at h
    exact h
  have hcTol := tol_le_cTol p hT
  calc |c.toRat - ord|
      ≤ |c.toRat - v.toRat| + |v.toRat - ord| := h1
    _ ≤ (|v.toRat| * 2 ^ (-11:ℤ) + 2 ^ (-25:ℤ)) + tol p T := by linarith
    _ ≤ ((|ord| + tol p T) * 2 ^ (-11:ℤ) + 2 ^ (-25:ℤ)) + tol p T := by
        have h2 : |v.toRat| * 2 ^ (-11:ℤ) ≤ (|ord| + tol p T) * 2 ^ (-11:ℤ) :=
          mul_le_mul_of_nonneg_right hvabs (by norm_num)
        linarith
    _ = 2 ^ (-11:ℤ) * |ord| + (1 + 2 ^ (-11:ℤ)) * tol p T + 2 ^ (-25:ℤ) := by ring
    _ ≤ 2 ^ (-11:ℤ) * |ord| + (2 ^ (-14:ℤ) * stepMag p T + 2 ^ (-25:ℤ)) + 2 ^ (-25:ℤ) := by
        linarith
    _ = 2 ^ (-11:ℤ) * |ord| + 2 ^ (-14:ℤ) * stepMag p T + 2 ^ (-24:ℤ) := by
        rw [show (2:ℚ)^(-24:ℤ) = 2^(-25:ℤ) + 2^(-25:ℤ) from by norm_num]
        ring

end Claims.BlockScaledMxfp8
