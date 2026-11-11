import TVA.Proof
import Claim

/-!
# The error of the reduction loop

The error of the reduction loop: the scalar recurrence of the accumulator, the relaxation of its bound to the claim's tolerance, and the final rounding to `float16`.
-/


/-!
# The error of the reduction loop

Constants, the partial-sum identity, the scalar recurrence, and the final relaxation.
-/

namespace Claims.Matmul

open TVA

/-- The relative constant of one accumulation step, `DotLoop.stepRel relDot16 64`, rounded up to
`268·2^-24`. -/
def rho : ℚ := 268 * 2 ^ (-24 : ℤ)

/-- The absolute constant of one accumulation step, `DotLoop.stepAbs absDot16 64`, rounded up to
`2^-112`. -/
def alp : ℚ := 2 ^ (-112 : ℤ)

/-- `rho` is nonnegative. -/
theorem rho_nonneg : 0 ≤ rho := by unfold rho; norm_num

/-- `alp` is nonnegative. -/
theorem alp_nonneg : 0 ≤ alp := by unfold alp; norm_num

/-- The per-step relative error of the claim's dot at `K = 64` is at most `rho`. -/
theorem stepRel_le_rho : DotLoop.stepRel Device.relDot16 64 ≤ rho := by
  unfold DotLoop.stepRel Device.relDot16 rho
  rw [Format.binary32_u]
  norm_num

/-- The per-step absolute error of the claim's dot at `K = 64` is at most `alp`. -/
theorem stepAbs_le_alp : DotLoop.stepAbs Device.absDot16 64 ≤ alp := by
  unfold DotLoop.stepAbs Device.absDot16 alp
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

/-- The partial sum before step `0` is empty. -/
theorem partSum_zero (p : Nat → ℚ) : partSum p 0 = 0 := by
  unfold partSum
  simp

/-- One step adds its `64` products to the partial sum. -/
theorem partSum_succ (p : Nat → ℚ) (t : Nat) :
    partSum p (t + 1) = partSum p t + stepMag p t := by
  unfold partSum stepMag
  rw [show 64 * (t + 1) = 64 * t + 64 by ring]
  rw [Finset.sum_range_add]
  congr 1

/-- When `64 ∣ K` the loop has `K / 64` steps. -/
theorem steps_eq {K : Nat} (hK : K % 64 = 0) : steps K = K / 64 := by
  obtain ⟨q, rfl⟩ := Nat.dvd_of_mod_eq_zero hK
  unfold steps
  rw [Nat.mul_add_div (by norm_num : 0 < 64), Nat.div_eq_of_lt (by norm_num : 63 < 64),
    Nat.add_zero, Nat.mul_div_cancel_left _ (by norm_num : 0 < 64)]

/-- When `64 ∣ K` the partial sum after all steps is the sum of all `K` products. -/
theorem partSum_full {p : Nat → ℚ} {K : Nat} (hK : K % 64 = 0) :
    partSum p (K / 64) = ∑ e ∈ Finset.range K, p e := by
  unfold partSum
  rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hK)]

/-- When `64 ∣ K` the exact dot product is the partial sum after all steps. -/
theorem dotSum_eq {va vb : Nat → F16} {K : Nat} (hK : K % 64 = 0) :
    dotSum va vb K = partSum (prodV va vb K) (K / 64) := by
  unfold dotSum partSum prodV
  rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hK)]
  exact Finset.sum_congr rfl (fun e he => by simp [Finset.mem_range.mp he])

/-- A step's magnitude of nonnegative terms is nonnegative. -/
theorem stepMag_nonneg {q : Nat → ℚ} (h : ∀ e, 0 ≤ q e) (t : Nat) : 0 ≤ stepMag q t := by
  unfold stepMag
  exact Finset.sum_nonneg (fun k _ => h _)

/-- Every lifted product magnitude is nonnegative. -/
theorem liftV_nonneg (va vb : Nat → F16) (K : Nat) : ∀ e, 0 ≤ liftV va vb K e := by
  intro e
  unfold liftV
  split_ifs with h
  · exact mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  · exact le_refl 0

/-- The products are bounded by their lifted magnitudes: `|va e · vb e| ≤ liftMag·liftMag`. -/
theorem prodV_abs_le_liftV (va vb : Nat → F16) (K e : Nat) :
    |prodV va vb K e| ≤ liftV va vb K e := by
  unfold prodV liftV
  split_ifs with h
  · rw [abs_mul]
    exact mul_le_mul (Device.abs_le_liftMag _ _) (Device.abs_le_liftMag _ _)
      (abs_nonneg _) (Device.liftMag_nonneg _ _)
  · exact le_refl 0

/-- `pathMag` is nonnegative. -/
theorem pathMag_nonneg (va vb : Nat → F16) (K : Nat) : 0 ≤ pathMag va vb K := by
  unfold pathMag
  exact Finset.sum_nonneg
    (fun t _ => add_nonneg (abs_nonneg _) (stepMag_nonneg (liftV_nonneg va vb K) t))

/-- The scalar recurrence: `T` steps that each add the exact increment up to
`ρ·(|acc| + L t) + α` compound to `(1 + ρ)^t·(ρ·Σ(|S s| + L s) + t·α)`. The magnitudes `L`
and the absolute constant `α` are nonnegative (in the use below: `L = stepMag (liftV …)`, `α = alp`). -/
theorem err_chain {ρ α : ℚ} (hρ : 0 ≤ ρ) {acc S L : Nat → ℚ} {T : Nat}
    (hacc : acc 0 = 0) (hS : S 0 = 0) (hL : ∀ s, 0 ≤ L s) (hα : 0 ≤ α)
    (hstep : ∀ t, t < T → |acc (t + 1) - (acc t + (S (t + 1) - S t))| ≤ ρ * (|acc t| + L t) + α) :
    ∀ t, t ≤ T →
      |acc t - S t| ≤ (1 + ρ) ^ t * (ρ * (∑ s ∈ Finset.range t, (|S s| + L s)) + (t : ℚ) * α) := by
  have hB : ∀ t, 0 ≤ ρ * (|S t| + L t) + α := by
    intro t
    have h1 : 0 ≤ ρ * (|S t| + L t) := mul_nonneg hρ (by linarith [abs_nonneg (S t), hL t])
    linarith
  intro t
  induction t with
  | zero =>
      intro _
      rw [hacc, hS]
      simp
  | succ n ih =>
      intro hnT
      have ih' : |acc n - S n| ≤
          (1 + ρ) ^ n * (ρ * (∑ s ∈ Finset.range n, (|S s| + L s)) + (n : ℚ) * α) :=
        ih (by omega)
      have hstepn := hstep n (by omega)
      have htri : |acc (n + 1) - S (n + 1)| ≤
          |acc (n + 1) - (acc n + (S (n + 1) - S n))| + |acc n - S n| := by
        have e : acc (n + 1) - S (n + 1) =
            (acc (n + 1) - (acc n + (S (n + 1) - S n))) + (acc n - S n) := by ring
        rw [e]
        exact abs_add_le _ _
      have habs : |acc n| ≤ |S n| + |acc n - S n| := by
        have e : acc n = S n + (acc n - S n) := by ring
        calc |acc n| = |S n + (acc n - S n)| := by conv_lhs => rw [e]
          _ ≤ |S n| + |acc n - S n| := abs_add_le _ _
      have hmul : ρ * (|acc n| + L n) ≤ ρ * (|S n| + |acc n - S n| + L n) := by
        apply mul_le_mul_of_nonneg_left _ hρ
        linarith
      have hE : |acc (n + 1) - S (n + 1)| ≤
          (ρ * (|S n| + L n) + α) + (1 + ρ) * |acc n - S n| := by
        linarith
      have hA : (ρ * (|S n| + L n) + α) + (1 + ρ) * |acc n - S n| ≤
          (ρ * (|S n| + L n) + α) + (1 + ρ) *
            ((1 + ρ) ^ n * (ρ * (∑ s ∈ Finset.range n, (|S s| + L s)) + (n : ℚ) * α)) := by
        have h1ρ : (0 : ℚ) ≤ 1 + ρ := by linarith
        linarith [mul_le_mul_of_nonneg_left ih' h1ρ]
      have hp1 : (1 : ℚ) ≤ (1 + ρ) ^ (n + 1) := one_le_pow₀ (by linarith)
      have hBble : ρ * (|S n| + L n) + α ≤ (1 + ρ) ^ (n + 1) * (ρ * (|S n| + L n) + α) := by
        calc ρ * (|S n| + L n) + α = 1 * (ρ * (|S n| + L n) + α) := by ring
          _ ≤ (1 + ρ) ^ (n + 1) * (ρ * (|S n| + L n) + α) :=
              mul_le_mul_of_nonneg_right hp1 (hB n)
      have hsum : (∑ s ∈ Finset.range (n + 1), (|S s| + L s)) =
          (∑ s ∈ Finset.range n, (|S s| + L s)) + (|S n| + L n) :=
        Finset.sum_range_succ (fun s => |S s| + L s) n
      have hcast : ((n + 1 : Nat) : ℚ) * α = (n : ℚ) * α + α := by push_cast; ring
      rw [hsum, hcast]
      calc |acc (n + 1) - S (n + 1)|
          ≤ (ρ * (|S n| + L n) + α) + (1 + ρ) * |acc n - S n| := hE
        _ ≤ (ρ * (|S n| + L n) + α) + (1 + ρ) *
              ((1 + ρ) ^ n * (ρ * (∑ s ∈ Finset.range n, (|S s| + L s)) + (n : ℚ) * α)) := hA
        _ = (1 + ρ) ^ (n + 1) * (ρ * (∑ s ∈ Finset.range n, (|S s| + L s)) + (n : ℚ) * α) +
              (ρ * (|S n| + L n) + α) := by rw [pow_succ]; ring
        _ ≤ (1 + ρ) ^ (n + 1) * (ρ * (∑ s ∈ Finset.range n, (|S s| + L s)) + (n : ℚ) * α) +
              (1 + ρ) ^ (n + 1) * (ρ * (|S n| + L n) + α) := by linarith
        _ = (1 + ρ) ^ (n + 1) *
              (ρ * ((∑ s ∈ Finset.range n, (|S s| + L s)) + (|S n| + L n)) +
                ((n : ℚ) * α + α)) := by ring

/-- `(1 + rho)^T ≤ 3/2` for at most `128` steps. -/
theorem pow_le {T : Nat} (hT : T ≤ 128) : (1 + rho) ^ T ≤ 3 / 2 := by
  have hρ : 0 ≤ rho := by unfold rho; norm_num
  have hTq : (T : ℚ) ≤ 128 := by exact_mod_cast hT
  have h1 : (T : ℚ) * rho ≤ 1 / 2 := by
    have h2 : (128 : ℚ) * rho ≤ 1 / 2 := by unfold rho; norm_num
    nlinarith
  have h1' : (T : ℚ) * rho ≤ 1 / 4 := by
    have h2 : (128 : ℚ) * rho ≤ 1 / 4 := by unfold rho; norm_num
    nlinarith
  have hp := DotLoop.pow_le hρ T h1
  have h3 : (1 : ℚ) + 2 * (T : ℚ) * rho ≤ 3 / 2 := by linarith
  linarith

/-- The fine bound, relaxed: `(1 + ρ)^T·(ρ·P + T·α) ≤ (3/2)·ρ·P + (3/2)·T·α`. -/
theorem fine_bound {ρ α P : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (hP : 0 ≤ P) {T : Nat}
    (hT : (1 + ρ) ^ T ≤ 3 / 2) :
    (1 + ρ) ^ T * (ρ * P + (T : ℚ) * α) ≤ (3 / 2) * ρ * P + (3 / 2) * (T : ℚ) * α := by
  have hT0 : (0 : ℚ) ≤ (T : ℚ) := by positivity
  have h1 : 0 ≤ ρ * P + (T : ℚ) * α := by nlinarith [mul_nonneg hρ hP]
  have h2 := mul_le_mul_of_nonneg_right hT h1
  nlinarith [h2]

/-- The accumulation's tolerance against the claim's `tol`. -/
theorem relax_tol {P : ℚ} (hP : 0 ≤ P) (T : Nat) (hT : T ≤ 128) :
    (3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp ≤ 2 ^ (-15 : ℤ) * P + 2 ^ (-100 : ℤ) := by
  have hTq : (T : ℚ) ≤ 128 := by exact_mod_cast hT
  have halp0 : 0 ≤ alp := by unfold alp; norm_num
  have h1 : (3 / 2) * rho ≤ 2 ^ (-15 : ℤ) := by unfold rho; norm_num
  have h2 : (3 / 2) * (T : ℚ) * alp ≤ 2 ^ (-100 : ℤ) := by
    have hs : (3 / 2) * (T : ℚ) * alp ≤ (3 / 2) * 128 * alp := by
      apply mul_le_mul_of_nonneg_right _ halp0
      nlinarith
    have hc : (3 / 2) * (128 : ℚ) * alp ≤ 2 ^ (-100 : ℤ) := by unfold alp; norm_num
    linarith
  have h3 : (3 / 2) * rho * P ≤ 2 ^ (-15 : ℤ) * P := mul_le_mul_of_nonneg_right h1 hP
  linarith

/-- The second clause's tolerance: one `float16` rounding on top of the accumulation's. -/
theorem relax_f16 {P : ℚ} (hP : 0 ≤ P) (T : Nat) (hT : T ≤ 128) :
    (1 + 2 ^ (-11 : ℤ)) * ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) + 2 ^ (-25 : ℤ) ≤
      2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ) := by
  have hTq : (T : ℚ) ≤ 128 := by exact_mod_cast hT
  have halp0 : 0 ≤ alp := by unfold alp; norm_num
  have hB : (3 / 2) * (T : ℚ) * alp ≤ 2 ^ (-100 : ℤ) := by
    have hs : (3 / 2) * (T : ℚ) * alp ≤ (3 / 2) * 128 * alp := by
      apply mul_le_mul_of_nonneg_right _ halp0
      nlinarith
    have hc : (3 / 2) * (128 : ℚ) * alp ≤ 2 ^ (-100 : ℤ) := by unfold alp; norm_num
    linarith
  have hc0 : (0 : ℚ) ≤ 1 + 2 ^ (-11 : ℤ) := by norm_num
  have h2 : (1 + 2 ^ (-11 : ℤ)) * ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) ≤
      (1 + 2 ^ (-11 : ℤ)) * ((3 / 2) * rho * P + 2 ^ (-100 : ℤ)) := by
    have hm := mul_le_mul_of_nonneg_left hB hc0
    linarith [hm]
  have h1 : (1 + 2 ^ (-11 : ℤ)) * ((3 / 2) * rho) ≤ 2 ^ (-15 : ℤ) := by unfold rho; norm_num
  have h3 : (1 + 2 ^ (-11 : ℤ)) * ((3 / 2) * rho * P + 2 ^ (-100 : ℤ)) + 2 ^ (-25 : ℤ) ≤
      2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ) := by
    have hm := mul_le_mul_of_nonneg_right h1 hP
    have hc : ((1 : ℚ) + 2 ^ (-11 : ℤ)) * 2 ^ (-100 : ℤ) + 2 ^ (-25 : ℤ) ≤ 2 ^ (-24 : ℤ) := by
      norm_num
    linarith [hm, hc]
  linarith

/-- The rounded value against the exact sum, given the fine bound on `v`. -/
theorem cvt_bound {S v cv P : ℚ} {T : Nat} (hP : 0 ≤ P) (hT : T ≤ 128)
    (hcv : |cv - v| ≤ |v| * 2 ^ (-11 : ℤ) + 2 ^ (-25 : ℤ))
    (hv : |v - S| ≤ (3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) :
    |cv - S| ≤ 2 ^ (-11 : ℤ) * |S| + 2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ) := by
  have hc0 : (0 : ℚ) ≤ 2 ^ (-11 : ℤ) := by norm_num
  have hcpos : (0 : ℚ) < 2 ^ (-11 : ℤ) := by norm_num
  have halp0 : 0 ≤ alp := by unfold alp; norm_num
  have htri : |cv - S| ≤ |cv - v| + |v - S| := by
    have e : cv - S = (cv - v) + (v - S) := by ring
    rw [e]
    exact abs_add_le _ _
  have hvabs : |v| ≤ |S| + |v - S| := by
    have e : v = S + (v - S) := by ring
    calc |v| = |S + (v - S)| := by conv_lhs => rw [e]
      _ ≤ |S| + |v - S| := abs_add_le _ _
  have h1 : 2 ^ (-11 : ℤ) * |v| ≤ 2 ^ (-11 : ℤ) * (|S| + |v - S|) :=
    mul_le_mul_of_nonneg_left hvabs hc0
  have hY : |v - S| ≤ (3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp := by
    have h2 : 2 ^ (-11 : ℤ) * |v - S| ≤
        2 ^ (-11 : ℤ) * ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) :=
      mul_le_mul_of_nonneg_left hv hc0
    exact le_of_mul_le_mul_left h2 hcpos
  have hc1 : (0 : ℚ) ≤ 1 + 2 ^ (-11 : ℤ) := by norm_num
  have key1 : |cv - v| ≤
      2 ^ (-11 : ℤ) * |S| + 2 ^ (-11 : ℤ) * |v - S| + 2 ^ (-25 : ℤ) := by
    calc |cv - v| ≤ |v| * 2 ^ (-11 : ℤ) + 2 ^ (-25 : ℤ) := hcv
      _ = 2 ^ (-11 : ℤ) * |v| + 2 ^ (-25 : ℤ) := by ring
      _ ≤ 2 ^ (-11 : ℤ) * (|S| + |v - S|) + 2 ^ (-25 : ℤ) :=
            add_le_add_left h1 (2 ^ (-25 : ℤ))
      _ = 2 ^ (-11 : ℤ) * |S| + 2 ^ (-11 : ℤ) * |v - S| + 2 ^ (-25 : ℤ) := by ring
  have key2 : |cv - S| ≤ 2 ^ (-11 : ℤ) * |S| +
      (1 + 2 ^ (-11 : ℤ)) * |v - S| + 2 ^ (-25 : ℤ) := by
    calc |cv - S| ≤ |cv - v| + |v - S| := htri
      _ ≤ (2 ^ (-11 : ℤ) * |S| + 2 ^ (-11 : ℤ) * |v - S| + 2 ^ (-25 : ℤ)) + |v - S| :=
            add_le_add_left key1 (|v - S|)
      _ = 2 ^ (-11 : ℤ) * |S| + (1 + 2 ^ (-11 : ℤ)) * |v - S| + 2 ^ (-25 : ℤ) := by ring
  have key3 : 2 ^ (-11 : ℤ) * |S| + (1 + 2 ^ (-11 : ℤ)) * |v - S| + 2 ^ (-25 : ℤ) ≤
      2 ^ (-11 : ℤ) * |S| + (1 + 2 ^ (-11 : ℤ)) *
        ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) + 2 ^ (-25 : ℤ) := by
    have hm := mul_le_mul_of_nonneg_left hY hc1
    exact add_le_add (add_le_add_right hm (2 ^ (-11 : ℤ) * |S|)) (le_refl (2 ^ (-25 : ℤ)))
  have hrelax := relax_f16 hP T hT
  have key4 : 2 ^ (-11 : ℤ) * |S| + (1 + 2 ^ (-11 : ℤ)) *
        ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) + 2 ^ (-25 : ℤ) ≤
      2 ^ (-11 : ℤ) * |S| + (2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ)) := by
    calc 2 ^ (-11 : ℤ) * |S| + (1 + 2 ^ (-11 : ℤ)) *
            ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) + 2 ^ (-25 : ℤ)
        = 2 ^ (-11 : ℤ) * |S| + ((1 + 2 ^ (-11 : ℤ)) *
            ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) + 2 ^ (-25 : ℤ)) := by ring
      _ ≤ 2 ^ (-11 : ℤ) * |S| + (2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ)) :=
            add_le_add_right hrelax (2 ^ (-11 : ℤ) * |S|)
  calc |cv - S| ≤ 2 ^ (-11 : ℤ) * |S| + (1 + 2 ^ (-11 : ℤ)) *
        ((3 / 2) * rho * P + (3 / 2) * (T : ℚ) * alp) + 2 ^ (-25 : ℤ) := key2.trans key3
    _ ≤ 2 ^ (-11 : ℤ) * |S| + (2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ)) := key4
    _ = 2 ^ (-11 : ℤ) * |S| + 2 ^ (-15 : ℤ) * P + 2 ^ (-24 : ℤ) := by ring

end Claims.Matmul
