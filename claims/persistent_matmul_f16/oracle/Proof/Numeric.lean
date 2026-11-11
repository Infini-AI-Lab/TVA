import Claim
import TVA.Proof

/-!
# The numbers of the inner loop: float constants, the error bound `errB` of `acc += tl.dot(...)`
after `t` steps, and the tolerance `tol` it stays within.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- The per-step relative constant of one inner step: `DotLoop.stepRel relDot16 64`. -/
def rho' : ℚ := DotLoop.stepRel Device.relDot16 64

/-- The per-step absolute constant of one inner step: `DotLoop.stepAbs absDot16 64`. -/
def alpha' : ℚ := DotLoop.stepAbs Device.absDot16 64

/-- A finite `float16` value is at most `65504` in magnitude. -/
theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have h := FloatV.abs_toRat_le_maxVal (f := Format.binary16) Format.binary16_regular hx
  rwa [Format.binary16_maxVal] at h

/-- A `float32` value of magnitude below `65504` is finite. -/
theorem f32_finite_of_abs_toRat_le {x : F32} (h : |x.toRat| ≤ 65504) : x.isFinite = true := by
  have h1 : Format.binary32.magVal (Format.binary32.mag x.pat) ≤ 65504 := by
    rw [← Format.abs_toRat]
    exact h
  have h2 : Format.binary32.mag x.pat ≤ Format.binary32.maxMag := by
    by_contra hlt
    have hlt' : Format.binary32.maxMag < Format.binary32.mag x.pat := not_le.mp hlt
    have h3 : Format.binary32.maxVal < Format.binary32.magVal (Format.binary32.mag x.pat) := by
      rw [← Format.magVal_maxMag Format.binary32_regular]
      exact Format.magVal_lt_magVal hlt'
    have h4 : (65504 : ℚ) ≤ Format.binary32.maxVal := by
      rw [Format.binary32_maxVal]; norm_num
    linarith
  exact Format.isFinite_of_mag_le Format.binary32_regular h2

/-- The lift floor of `float16` is below `65504`. -/
theorem liftFloor_f16_le : Device.liftFloor .f16 ≤ 65504 := by
  norm_num [Device.liftFloor]

/-- The lifted magnitude of a `float16` value below `65504` is below `65504`. -/
theorem liftMag_f16_le {q : ℚ} (hq : |q| ≤ 65504) : Device.liftMag .f16 q ≤ 65504 := by
  unfold Device.liftMag
  split_ifs with h
  · norm_num
  · exact max_le hq liftFloor_f16_le

/-- `ρ'` is nonnegative. -/
theorem rho'_nonneg : 0 ≤ rho' := by
  unfold rho' DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  norm_num

/-- `α'` is nonnegative. -/
theorem alpha'_nonneg : 0 ≤ alpha' := by
  unfold alpha' DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_eta, Format.binary32_u]
  norm_num

/-- The growth factor over the inner loop is below `1.0021`. -/
theorem rho'_pow_le (T : Nat) (hT : T ≤ 128) : (1 + rho') ^ T ≤ 10021 / 10000 := by
  have h1 : (1 : ℚ) ≤ 1 + rho' := by have := rho'_nonneg; linarith
  refine (pow_le_pow_right₀ h1 hT).trans ?_
  unfold rho' DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  norm_num

/-- One step's relative error is below `2^-15`. -/
theorem rho'_le : (10021 / 10000) * rho' ≤ 2 ^ (-15 : ℤ) := by
  unfold rho' DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  norm_num

/-- All the inner loop's absolute errors together are below `2^-112`. -/
theorem alpha'_le : (10021 / 10000) * ((128 : ℚ) * alpha') ≤ 2 ^ (-112 : ℤ) := by
  unfold alpha' DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_eta, Format.binary32_u]
  norm_num

/-- The number of inner steps is at most `128`. -/
theorem steps_le {K : Nat} (hK : K ≤ 2 ^ 13) : steps K ≤ 128 := by
  unfold steps
  omega

/-- A `K` that is a multiple of `64` fills `cdiv(K, 64)` steps of `64`. -/
theorem steps_mul {K : Nat} (hK : 64 ∣ K) : 64 * steps K = K := by
  unfold steps
  omega

/-- One step of the reduction: `partSum` grows by the step's terms. -/
theorem partSum_succ (p : Nat → ℚ) (t : Nat) :
    partSum p (t + 1) = partSum p t + ∑ k : Fin 64, p (64 * t + (k : Nat)) := by
  unfold partSum
  rw [show 64 * (t + 1) = 64 * t + 64 by ring, Finset.sum_range_add]
  congr 1

/-- The step's terms of `prodV` are the input values' products. -/
theorem stepMag_prodV {K t : Nat} (ht : 64 * (t + 1) ≤ K) (va vb : Nat → F16) :
    stepMag (prodV va vb K) t =
      ∑ k : Fin 64, (va (64 * t + (k : Nat))).toRat * (vb (64 * t + (k : Nat))).toRat := by
  unfold stepMag prodV
  refine Finset.sum_congr rfl fun k _ => ?_
  have hk : 64 * t + (k : Nat) < K := by
    have := k.isLt
    omega
  simp only [hk, ite_true]

/-- The step's lifted magnitudes of `liftV` are the input values' lifted magnitudes. -/
theorem stepMag_liftV {K t : Nat} (ht : 64 * (t + 1) ≤ K) (va vb : Nat → F16) :
    stepMag (liftV va vb K) t =
      ∑ k : Fin 64, Device.liftMag .f16 (va (64 * t + (k : Nat))).toRat *
        Device.liftMag .f16 (vb (64 * t + (k : Nat))).toRat := by
  unfold stepMag liftV
  refine Finset.sum_congr rfl fun k _ => ?_
  have hk : 64 * t + (k : Nat) < K := by
    have := k.isLt
    omega
  simp only [hk, ite_true]

/-- `partSum` over the whole reduction is the exact dot product. -/
theorem partSum_steps {K : Nat} (hK : 64 ∣ K) (va vb : Nat → F16) :
    partSum (prodV va vb K) (steps K) = dotSum va vb K := by
  unfold partSum dotSum
  rw [steps_mul hK]
  refine Finset.sum_congr rfl fun e he => ?_
  have heK : e < K := Finset.mem_range.mp he
  unfold prodV
  simp only [heK, ite_true]

/-- `accMag` is the sum of the per-step magnitudes. -/
theorem accMag_eq (va vb : Nat → F16) (K : Nat) :
    accMag va vb K =
      ∑ t ∈ Finset.range (steps K),
        (|partSum (prodV va vb K) t| + stepMag (liftV va vb K) t) := rfl

/-- One step of the error recursion, as arithmetic. -/
theorem err_step {ρ α c S D L r : ℚ} (hρ : 0 ≤ ρ) (_hL : 0 ≤ L)
    (h : |r - (c + D)| ≤ ρ * (|c| + L) + α) :
    |r - (S + D)| ≤ (1 + ρ) * |c - S| + ρ * (|S| + L) + α := by
  have h1 : |r - (S + D)| ≤ |r - (c + D)| + |c - S| := by
    have e : r - (S + D) = (r - (c + D)) + (c - S) := by ring
    rw [e]
    exact abs_add_le _ _
  have h2 : |c| ≤ |c - S| + |S| := by
    have h := abs_add_le (c - S) S
    rwa [sub_add_cancel] at h
  have h3 := mul_le_mul_of_nonneg_left h2 hρ
  linarith

/-- **The error recursion.** `T` steps of relative error `ρ` and absolute error `α`, from `0`,
leave an error below `(1+ρ)^T·(ρ·Σ_{t<T} A_t + T·α)`. -/
theorem err_le_upto (ρ α : ℚ) (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (A : Nat → ℚ) (hA : ∀ t, 0 ≤ A t)
    (e : Nat → ℚ) (h0 : e 0 = 0) :
    ∀ T, (∀ t, t < T → |e (t + 1)| ≤ (1 + ρ) * |e t| + ρ * A t + α) →
      |e T| ≤ (1 + ρ) ^ T * (ρ * ∑ t ∈ Finset.range T, A t + (T : ℚ) * α) := by
  intro T
  induction T with
  | zero => intro _; simp [h0]
  | succ T ih =>
    intro h
    have hT := ih fun t ht => h t (by omega)
    have hstep := h T (by omega)
    rw [Finset.sum_range_succ]
    have hp : (0 : ℚ) ≤ (1 + ρ) ^ T := by positivity
    have hP1 : (1 : ℚ) ≤ (1 + ρ) ^ T := one_le_pow₀ (by linarith)
    have hA' : 0 ≤ ρ * A T + α := by
      have := hA T
      nlinarith
    have hfac : (1 : ℚ) ≤ (1 + ρ) * (1 + ρ) ^ T := by nlinarith
    have hmul : ρ * A T + α ≤ ((1 + ρ) * (1 + ρ) ^ T) * (ρ * A T + α) := by
      have h := mul_le_mul_of_nonneg_right hfac hA'
      rwa [one_mul] at h
    have hArith :
        (1 + ρ) * |e T| + ρ * A T + α ≤
          (1 + ρ) * ((1 + ρ) ^ T * (ρ * ∑ t ∈ Finset.range T, A t + (T : ℚ) * α)) +
            (ρ * A T + α) := by nlinarith
    refine hstep.trans (hArith.trans ?_)
    rw [pow_succ]
    have hx : ρ * A T + α ≤ (1 + ρ) * ((1 + ρ) ^ T * (ρ * A T + α)) := by
      calc ρ * A T + α ≤ ((1 + ρ) * (1 + ρ) ^ T) * (ρ * A T + α) := hmul
        _ = (1 + ρ) * ((1 + ρ) ^ T * (ρ * A T + α)) := by ring
    push_cast
    nlinarith [hx]

/-- The relaxed tolerance bound: the fine bound is below `2^-15·A + 2^-112`. -/
theorem tol_bound (A : ℚ) (hA : 0 ≤ A) (T : Nat) (hT : T ≤ 128) :
    (1 + rho') ^ T * (rho' * A + (T : ℚ) * alpha') ≤ 2 ^ (-15 : ℤ) * A + 2 ^ (-112 : ℤ) := by
  have hpow := rho'_pow_le T hT
  have hAn : 0 ≤ rho' * A + (T : ℚ) * alpha' := by
    have := rho'_nonneg
    have := alpha'_nonneg
    have hT0 : (0 : ℚ) ≤ (T : ℚ) := by positivity
    nlinarith
  have h1 : (1 + rho') ^ T * (rho' * A + (T : ℚ) * alpha') ≤
      (10021 / 10000) * (rho' * A + (128 : ℚ) * alpha') := by
    refine mul_le_mul hpow ?_ hAn (by norm_num)
    have := alpha'_nonneg
    have hT' : (T : ℚ) ≤ 128 := by exact_mod_cast hT
    nlinarith
  have h3 := rho'_le
  have h4 := alpha'_le
  have hm1 : (10021 / 10000) * (rho' * A) ≤ (2 : ℚ) ^ (-15 : ℤ) * A := by
    have h := mul_le_mul_of_nonneg_right h3 hA
    calc (10021 / 10000) * (rho' * A) = ((10021 / 10000) * rho') * A := by ring
      _ ≤ (2 : ℚ) ^ (-15 : ℤ) * A := h
  have hm2 : (10021 / 10000) * ((128 : ℚ) * alpha') ≤ 2 ^ (-112 : ℤ) := h4
  have hsum : (10021 / 10000) * (rho' * A + (128 : ℚ) * alpha') =
      (10021 / 10000) * (rho' * A) + (10021 / 10000) * ((128 : ℚ) * alpha') := by ring
  have hfin : (10021 / 10000) * (rho' * A) + (10021 / 10000) * ((128 : ℚ) * alpha') ≤
      2 ^ (-15 : ℤ) * A + 2 ^ (-112 : ℤ) := by linarith [hm1, hm2]
  calc (1 + rho') ^ T * (rho' * A + (T : ℚ) * alpha')
      ≤ 10021 / 10000 * (rho' * A + 128 * alpha') := h1
    _ = (10021 / 10000) * (rho' * A) + (10021 / 10000) * ((128 : ℚ) * alpha') := hsum
    _ ≤ 2 ^ (-15 : ℤ) * A + 2 ^ (-112 : ℤ) := hfin

/-- A product of two finite `float16` values is at most `65504^2`. -/
theorem prodV_abs_le {K : Nat} {va vb : Nat → F16}
    (hva : ∀ k, k < K → (va k).isFinite = true) (hvb : ∀ k, k < K → (vb k).isFinite = true)
    (e : Nat) : |prodV va vb K e| ≤ 65504 ^ 2 := by
  unfold prodV
  split_ifs with h
  · have ha := f16_abs_le (hva e h)
    have hb := f16_abs_le (hvb e h)
    have ha' : 0 ≤ |(va e).toRat| := abs_nonneg _
    have hb' : 0 ≤ |(vb e).toRat| := abs_nonneg _
    rw [abs_mul]
    nlinarith
  · simp

/-- A partial sum of `t` steps of products is at most `64·t·65504^2`. -/
theorem partSum_abs_le {K : Nat} {va vb : Nat → F16}
    (hva : ∀ k, k < K → (va k).isFinite = true) (hvb : ∀ k, k < K → (vb k).isFinite = true)
    (t : Nat) : |partSum (prodV va vb K) t| ≤ (64 * t : Nat) * 65504 ^ 2 := by
  unfold partSum
  have h1 : |∑ e ∈ Finset.range (64 * t), prodV va vb K e| ≤
      ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| :=
    Finset.abs_sum_le_sum_abs _ _
  have h2 : ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| ≤
      ∑ _e ∈ Finset.range (64 * t), (65504 : ℚ) ^ 2 := by
    refine Finset.sum_le_sum fun e _ => prodV_abs_le hva hvb e
  have h3 : ∑ _e ∈ Finset.range (64 * t), (65504 : ℚ) ^ 2 =
      ((64 * t : Nat) : ℚ) * 65504 ^ 2 := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  calc |∑ e ∈ Finset.range (64 * t), prodV va vb K e|
      ≤ ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| := h1
    _ ≤ ∑ _e ∈ Finset.range (64 * t), (65504 : ℚ) ^ 2 := h2
    _ = ((64 * t : Nat) : ℚ) * 65504 ^ 2 := h3

/-- A step's lifted magnitude is at most `64·65504^2`. -/
theorem stepMag_liftV_le {K : Nat} {va vb : Nat → F16}
    (hva : ∀ k, k < K → (va k).isFinite = true) (hvb : ∀ k, k < K → (vb k).isFinite = true)
    (t : Nat) : stepMag (liftV va vb K) t ≤ 64 * 65504 ^ 2 := by
  unfold stepMag
  have h2 : ∑ k : Fin 64, liftV va vb K (64 * t + (k : Nat)) ≤
      ∑ _k : Fin 64, (65504 : ℚ) ^ 2 := by
    refine Finset.sum_le_sum fun k _ => ?_
    unfold liftV
    split_ifs with h
    · have ha := f16_abs_le (hva _ h)
      have hb := f16_abs_le (hvb _ h)
      have h1 := liftMag_f16_le ha
      have h2 := liftMag_f16_le hb
      have h1' : 0 ≤ Device.liftMag .f16 (va (64 * t + (k : Nat))).toRat :=
        Device.liftMag_nonneg _ _
      nlinarith
    · simp
  have h3 : ∑ _k : Fin 64, (65504 : ℚ) ^ 2 = 64 * 65504 ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    norm_num
  exact h2.trans_eq h3

/-- `accMag` is at most `2^53`. -/
theorem accMag_le {K : Nat} (hK : K ≤ 2 ^ 13) {va vb : Nat → F16}
    (hva : ∀ k, k < K → (va k).isFinite = true) (hvb : ∀ k, k < K → (vb k).isFinite = true) :
    accMag va vb K ≤ 2 ^ 53 := by
  rw [accMag_eq]
  have hT := steps_le hK
  have h2 : ∑ t ∈ Finset.range (steps K),
      (|partSum (prodV va vb K) t| + stepMag (liftV va vb K) t) ≤
      ∑ t ∈ Finset.range (steps K), (((64 * t : Nat) : ℚ) * 65504 ^ 2 + 64 * 65504 ^ 2) := by
    refine Finset.sum_le_sum fun t _ => ?_
    have h1 := partSum_abs_le (K := K) hva hvb t
    have h3 := stepMag_liftV_le (K := K) hva hvb t
    linarith
  have h3 : ∑ t ∈ Finset.range (steps K),
      (((64 * t : Nat) : ℚ) * 65504 ^ 2 + 64 * 65504 ^ 2) ≤
      ∑ t ∈ Finset.range (128),
        (((64 * t : Nat) : ℚ) * 65504 ^ 2 + 64 * 65504 ^ 2) := by
    refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 hT) ?_
    intro x _ _
    have : (0 : ℚ) ≤ ((64 * x : Nat) : ℚ) * 65504 ^ 2 + 64 * 65504 ^ 2 := by positivity
    linarith
  have h4 : ∑ t ∈ Finset.range (128),
      (((64 * t : Nat) : ℚ) * 65504 ^ 2 + 64 * 65504 ^ 2) ≤ 2 ^ 53 := by
    have hcast : ∀ t : Nat, ((64 * t : Nat) : ℚ) = 64 * (t : ℚ) := by
      intro t; push_cast; ring
    rw [Finset.sum_add_distrib]
    simp only [hcast, ← Finset.sum_mul, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul]
    norm_num
  exact (h2.trans h3).trans h4

/-- `tol` plus `accMag` is at most `2^54`. -/
theorem tol_add_accMag_le {K : Nat} (hK : K ≤ 2 ^ 13) {va vb : Nat → F16}
    (hva : ∀ k, k < K → (va k).isFinite = true) (hvb : ∀ k, k < K → (vb k).isFinite = true) :
    tol va vb K + accMag va vb K ≤ 2 ^ 54 := by
  have h := accMag_le hK hva hvb
  have h0 : 0 ≤ accMag va vb K := by
    rw [accMag_eq]
    refine Finset.sum_nonneg fun t _ => ?_
    have h1 : 0 ≤ |partSum (prodV va vb K) t| := abs_nonneg _
    have h2 : 0 ≤ stepMag (liftV va vb K) t := by
      unfold stepMag
      exact Finset.sum_nonneg fun k _ => by
        unfold liftV
        split_ifs with hh
        · have h1' := Device.liftMag_nonneg .f16 (va (64 * t + (k : Nat))).toRat
          have h2' := Device.liftMag_nonneg .f16 (vb (64 * t + (k : Nat))).toRat
          nlinarith
        · simp
    linarith
  unfold tol
  have hy : ((2 : ℚ) ^ (-15 : ℤ)) * accMag va vb K ≤ 2 ^ 38 := by
    have h1 : (2 : ℚ) ^ (-15 : ℤ) * accMag va vb K ≤ (2 : ℚ) ^ (-15 : ℤ) * 2 ^ 53 :=
      mul_le_mul_of_nonneg_left h (by norm_num)
    have e : (2 : ℚ) ^ (-15 : ℤ) * 2 ^ 53 = 2 ^ 38 := by
      rw [← zpow_natCast, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      norm_num
    linarith
  have hy2 : (0 : ℚ) ≤ (2 : ℚ) ^ (-112 : ℤ) := by norm_num
  have hnum : (2 : ℚ) ^ 38 + 2 ^ 53 + 2 ^ (-112 : ℤ) < 2 ^ 54 := by norm_num
  linarith

/-- The room condition of one `tl.dot` step, from the magnitude bounds. -/
theorem room {va vb : Fin 64 → ℚ} (hva : ∀ k, |va k| ≤ 65504) (hvb : ∀ k, |vb k| ≤ 65504)
    {c : ℚ} (hc : |c| ≤ 2 ^ 54) :
    2 * ((1 + Device.relDot16 64) *
        (|c| + ∑ k, Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k)) +
      Device.absDot16 64 + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
  have hsum : ∑ k, Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k) ≤
      64 * 65504 ^ 2 := by
    have h2 : ∑ k : Fin 64, Device.liftMag .f16 (va k) * Device.liftMag .f16 (vb k) ≤
        ∑ _k : Fin 64, (65504 : ℚ) ^ 2 := by
      refine Finset.sum_le_sum fun k _ => ?_
      have h1 := liftMag_f16_le (hva k)
      have h2 := liftMag_f16_le (hvb k)
      have h1' : 0 ≤ Device.liftMag .f16 (va k) := Device.liftMag_nonneg _ _
      nlinarith
    have h3 : ∑ _k : Fin 64, (65504 : ℚ) ^ 2 = 64 * 65504 ^ 2 := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      norm_num
    exact h2.trans_eq h3
  rw [Format.binary32_maxVal]
  unfold Device.relDot16 Device.absDot16
  rw [Format.binary32_eta]
  norm_num at *
  nlinarith

/-- **The final rounding.** Rounding a `float32` within `tol` of the sum to `float16` stays within
`tol` plus one `float16` rounding of the sum. -/
theorem cvt_out {x : F32} {K : Nat} {va vb : Nat → F16}
    (hxd : |x.toRat - dotSum va vb K| ≤ tol va vb K)
    (hroom16 : |dotSum va vb K| + tol va vb K ≤ Format.binary16.maxVal) :
    (FloatV.cvt x : F16).isFinite = true ∧
      |(FloatV.cvt x : F16).toRat - dotSum va vb K| ≤
        tol va vb K + (|dotSum va vb K| + tol va vb K) * Format.binary16.u +
          Format.binary16.eta := by
  have hxle : |x.toRat| ≤ Format.binary16.maxVal := by
    have h1 : |x.toRat| ≤ |dotSum va vb K| + tol va vb K := by
      have e : |x.toRat| ≤ |x.toRat - dotSum va vb K| + |dotSum va vb K| := by
        have h := abs_add_le (x.toRat - dotSum va vb K) (dotSum va vb K)
        rwa [sub_add_cancel] at h
      linarith
    linarith
  have hx : x.isFinite = true := by
    refine f32_finite_of_abs_toRat_le ?_
    rw [Format.binary16_maxVal] at hxle
    exact hxle
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err (g := Format.binary16) Format.binary16_regular hx hxle
  refine ⟨hfin, ?_⟩
  have h1 : |(FloatV.cvt x : F16).toRat - dotSum va vb K| ≤
      |(FloatV.cvt x : F16).toRat - x.toRat| + |x.toRat - dotSum va vb K| := by
    have h := abs_add_le ((FloatV.cvt x : F16).toRat - x.toRat) (x.toRat - dotSum va vb K)
    rwa [sub_add_sub_cancel] at h
  have h2 : |x.toRat| * Format.binary16.u ≤
      (|dotSum va vb K| + tol va vb K) * Format.binary16.u := by
    refine mul_le_mul_of_nonneg_right ?_ (by rw [Format.binary16_u]; norm_num)
    have h3 : |x.toRat| ≤ |x.toRat - dotSum va vb K| + |dotSum va vb K| := by
      have h := abs_add_le (x.toRat - dotSum va vb K) (dotSum va vb K)
      rwa [sub_add_cancel] at h
    linarith
  linarith

/-- **The inner loop's error bound.** The accumulator is within `tol` of the exact dot product. -/
theorem acc_err_le {d : Inst} (hd : d.ok) {va vb : Nat → F16} {e : Nat → ℚ}
    (h0 : e 0 = 0)
    (hstep : ∀ t, t < steps d.K →
      |e (t + 1)| ≤ (1 + rho') * |e t| +
        rho' * (|partSum (prodV va vb d.K) t| + stepMag (liftV va vb d.K) t) + alpha') :
    |e (steps d.K)| ≤ tol va vb d.K := by
  have hK := hd.K_le
  have hA : ∀ t, 0 ≤ |partSum (prodV va vb d.K) t| + stepMag (liftV va vb d.K) t := by
    intro t
    have h1 : 0 ≤ |partSum (prodV va vb d.K) t| := abs_nonneg _
    have h2 : 0 ≤ stepMag (liftV va vb d.K) t := by
      unfold stepMag
      exact Finset.sum_nonneg fun k _ => by
        unfold liftV
        split_ifs with hh
        · have h1' := Device.liftMag_nonneg .f16 (va (64 * t + (k : Nat))).toRat
          have h2' := Device.liftMag_nonneg .f16 (vb (64 * t + (k : Nat))).toRat
          nlinarith
        · simp
    linarith
  have hacc0 : 0 ≤ accMag va vb d.K := by
    rw [accMag_eq]
    exact Finset.sum_nonneg fun t _ => hA t
  have h := err_le_upto rho' alpha' rho'_nonneg alpha'_nonneg
    (fun t => |partSum (prodV va vb d.K) t| + stepMag (liftV va vb d.K) t) hA e h0
    (steps d.K) hstep
  have htb := tol_bound (accMag va vb d.K) hacc0 (steps d.K) (steps_le hK)
  unfold tol
  exact h.trans htb








/-- `tl.zeros` is a splat of the zero. -/
theorem zeros_eq_full (M N : Nat) :
    (tl.zeros [M, N] .f32) = (Tensor.full (FloatV.ofPat 0) : Tensor [M, N] F32) := by rfl

/-- A lane of a `128 × 256` splat. -/
theorem full_get2_128_256 (a : F32) (i : Fin 128) (n : Fin 256) :
    ((Tensor.full a : Tensor [128, 256] F32).get2 i n) = a := by rfl
/-- A lane of a `128 × 64` splat. -/
theorem full_get2_128_64 (a : F32) (i : Fin 128) (k : Fin 64) :
    ((Tensor.full a : Tensor [128, 64] F32).get2 i k) = a := by rfl
/-- A lane of a `64 × 256` splat. -/
theorem full_get2_64_256 (a : F32) (k : Fin 64) (n : Fin 256) :
    ((Tensor.full a : Tensor [64, 256] F32).get2 k n) = a := by rfl

/-- A `float32` to `float16` conversion has a result on every platform. -/
theorem convRel_f32_f16 (P : Platform) (x : F32) : ∃ r : F16, ConvRel P .f32 .f16 x r := by
  refine ⟨FloatV.cvt x, ?_⟩
  show FloatRel P .binary32 .binary16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide : ¬ (Format.binary32 = Format.binary16))]
  exact Or.inl ⟨by unfold Format.Std; exact Or.inr (Or.inr (Or.inl rfl)),
    by unfold Format.Std; exact Or.inl rfl, FloatV.NaNEq.refl _⟩

/-- `float32` to `float16` round-to-nearest-even is a covered conversion. -/
theorem roundOk_f32_f16 : tl.RoundOk .binary32 .binary16 .rn := Or.inl rfl


/-- The error bound of `t` accumulation steps with constants `ρ`, `α` over magnitude `B`: `(1+ρ)^t (ρ B + t α)`. -/
def errB (ρ α B : ℚ) (t : Nat) : ℚ := (1 + ρ) ^ t * (ρ * B + (t : ℚ) * α)

/-- The true value at `t = 0`; the requested `= 0` holds only when `ρ·B = 0`. -/
theorem errB_zero (ρ α B : ℚ) : errB ρ α B 0 = ρ * B := by
  unfold errB
  simp

/-- The base case of the recursion, with `B = 0` (as in the induction it is used in). -/
theorem errB_zero' (ρ α : ℚ) : errB ρ α 0 0 = 0 := by
  rw [errB_zero]
  ring

/-- `errB` grows with the magnitude `B`. -/
theorem errB_mono {ρ α B B' : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (hB : B ≤ B') (hB' : 0 ≤ B')
    {t t' : Nat} (ht : t ≤ t') : errB ρ α B t ≤ errB ρ α B' t' := by
  unfold errB
  have hp : (0 : ℚ) ≤ 1 + ρ := by linarith
  have hpow : (1 + ρ) ^ t ≤ (1 + ρ) ^ t' := pow_le_pow_right₀ (by linarith) ht
  have harg : ρ * B + (t : ℚ) * α ≤ ρ * B' + (t' : ℚ) * α := by
    have ha : ρ * B ≤ ρ * B' := mul_le_mul_of_nonneg_left hB hρ
    have hb : (t : ℚ) * α ≤ (t' : ℚ) * α :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast ht) hα
    linarith
  have hd : (0 : ℚ) ≤ ρ * B' + (t' : ℚ) * α := by
    have h1 : 0 ≤ ρ * B' := mul_nonneg hρ hB'
    have h2 : 0 ≤ (t' : ℚ) * α := mul_nonneg (by positivity) hα
    linarith
  have ha0 : (0 : ℚ) ≤ (1 + ρ) ^ t := pow_nonneg hp t
  calc (1 + ρ) ^ t * (ρ * B + (t : ℚ) * α)
      ≤ (1 + ρ) ^ t * (ρ * B' + (t' : ℚ) * α) := mul_le_mul_of_nonneg_left harg ha0
    _ ≤ (1 + ρ) ^ t' * (ρ * B' + (t' : ℚ) * α) := mul_le_mul_of_nonneg_right hpow hd

/-- One step of the recursion; the hypothesis `(hL : 0 ≤ L)` is necessary, see the module note. -/
theorem errB_step {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t : Nat} {c S D L r : ℚ}
    (hL : 0 ≤ L)
    (he : |c - S| ≤ errB ρ α B t) (hs : |r - (c + D)| ≤ ρ * (|c| + L) + α) :
    |r - (S + D)| ≤ errB ρ α (B + (|S| + L)) (t + 1) := by
  have hp : (0 : ℚ) ≤ 1 + ρ := by linarith
  have hbase := err_step (S := S) hρ hL hs
  have hmul : (1 + ρ) * |c - S| ≤ (1 + ρ) * errB ρ α B t :=
    mul_le_mul_of_nonneg_left he hp
  have hstep : |r - (S + D)| ≤ (1 + ρ) * errB ρ α B t + ρ * (|S| + L) + α := by
    linarith
  refine hstep.trans ?_
  have hA : 0 ≤ |S| + L := by
    have h := abs_nonneg S
    linarith
  have hQ : 0 ≤ ρ * (|S| + L) + α := by
    have h1 : 0 ≤ ρ * (|S| + L) := mul_nonneg hρ hA
    linarith
  have hP1 : (1 : ℚ) ≤ (1 + ρ) ^ (t + 1) := one_le_pow₀ (by linarith : (1 : ℚ) ≤ 1 + ρ)
  have hPQ : ρ * (|S| + L) + α ≤ (1 + ρ) ^ (t + 1) * (ρ * (|S| + L) + α) := by
    calc ρ * (|S| + L) + α = 1 * (ρ * (|S| + L) + α) := by ring
      _ ≤ (1 + ρ) ^ (t + 1) * (ρ * (|S| + L) + α) := mul_le_mul_of_nonneg_right hP1 hQ
  have e1 : (1 + ρ) * errB ρ α B t = (1 + ρ) ^ (t + 1) * (ρ * B + (t : ℚ) * α) := by
    unfold errB
    rw [pow_succ]
    ring
  have e2 : errB ρ α (B + (|S| + L)) (t + 1) =
      (1 + ρ) ^ (t + 1) * ((ρ * B + (t : ℚ) * α) + (ρ * (|S| + L) + α)) := by
    unfold errB
    push_cast
    ring
  have e3 : (1 + ρ) ^ (t + 1) * ((ρ * B + (t : ℚ) * α) + (ρ * (|S| + L) + α)) =
      (1 + ρ) ^ (t + 1) * (ρ * B + (t : ℚ) * α) +
        (1 + ρ) ^ (t + 1) * (ρ * (|S| + L) + α) := by ring
  rw [e1, e2, e3]
  linarith

/-- After all steps, `errB` is within the claim's `tol`. -/
theorem errB_le_tol {K : Nat} (hK : K ≤ 2 ^ 13) {va vb : Nat → F16}
    (_hva : ∀ k, k < K → (va k).isFinite = true) (_hvb : ∀ k, k < K → (vb k).isFinite = true) :
    errB rho' alpha' (accMag va vb K) (steps K) ≤ tol va vb K := by
  have h0 : 0 ≤ accMag va vb K := by
    rw [accMag_eq]
    refine Finset.sum_nonneg fun t _ => ?_
    have h1 : 0 ≤ |partSum (prodV va vb K) t| := abs_nonneg _
    have h2 : 0 ≤ stepMag (liftV va vb K) t := by
      unfold stepMag
      exact Finset.sum_nonneg fun k _ => by
        unfold liftV
        split_ifs with hh
        · have h1' := Device.liftMag_nonneg .f16 (va (64 * t + (k : Nat))).toRat
          have h2' := Device.liftMag_nonneg .f16 (vb (64 * t + (k : Nat))).toRat
          nlinarith
        · simp
    linarith
  have h := tol_bound (accMag va vb K) h0 (steps K) (steps_le hK)
  unfold errB tol
  exact h

end Claims.PersistentMatmul
