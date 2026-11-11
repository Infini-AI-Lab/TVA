import Mathlib

/-!
# The error of a rescaled accumulation, one trip at a time

An online softmax keeps, for one query row, running sums that are rescaled by `α ≤ 1` whenever the
reference maximum grows: `X' = α·X + Y`. The same shape serves the row sum `l` (`Y = Σ p`) and each
output column of the accumulator (`Y = Σ p·v`). The computed state `x̂` follows it up to a rounding
per trip. `online_step` is the induction step of the closed-form bound

  `|x̂ₜ − Xₜ| ≤ exp Θ · ((Θ + ρ̄)·Mₜ + κ·Lₜ + ηv·Vₜ + ηA·t)`,

where `Θ` is the sum of the per-trip rates `ε` (the rounding of the rescaling, of the accumulation,
and the relative error of the computed `α`), `ρ̄` the relative error of the weights (made once, not
once per trip), and `κ`, `ηv`, `ηA` absolute errors proportional to the weight sum `Lₜ`, to
`Σ|v|` (`Vₜ`), and constant.
-/

namespace Claims.AttnFwd

/-- **One trip of a rescaled accumulation.** The exact state follows `X' = α·X + Y` with magnitude
`M' = α·M + Z` (`|Y| ≤ Z`), weight sum `L' = α·L + R` and `V' = V + W`; the computed state's error
so far is within the closed form with rate sum `Θ` after `t` trips; the trip is within
`ε·(α·|x̂| + Z) + ρ̄·Z + κ·R + ηv·W + ηA` of `α·x̂ + Y`. Then the error after the trip is within the
closed form with rate sum `Θ + ε`, and `|X'| ≤ M'`. -/
theorem online_step {α ε Θ ρb κ ηv ηA t x x' X Y M Z L R V W : ℝ}
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hε : 0 ≤ ε) (hΘ : 0 ≤ Θ) (hρ : 0 ≤ ρb) (hκ : 0 ≤ κ)
    (hηv : 0 ≤ ηv) (hηA : 0 ≤ ηA) (ht : 0 ≤ t) (hM : 0 ≤ M) (hZ : 0 ≤ Z) (hL : 0 ≤ L)
    (hR : 0 ≤ R) (hV : 0 ≤ V) (hW : 0 ≤ W)
    (hXM : |X| ≤ M) (hYZ : |Y| ≤ Z)
    (herr : |x - X| ≤ Real.exp Θ * ((Θ + ρb) * M + κ * L + ηv * V + ηA * t))
    (hstep : |x' - (α * x + Y)| ≤ ε * (α * |x| + Z) + ρb * Z + κ * R + ηv * W + ηA) :
    |x' - (α * X + Y)| ≤ Real.exp (Θ + ε) *
        ((Θ + ε + ρb) * (α * M + Z) + κ * (α * L + R) + ηv * (V + W) + ηA * (t + 1)) ∧
      |α * X + Y| ≤ α * M + Z := by
  refine ⟨?_, ?_⟩
  · set E := Real.exp Θ with hE
    set F := Real.exp ε with hF
    have hE1 : 1 ≤ E := Real.one_le_exp hΘ
    have hF1 : 1 + ε ≤ F := by linarith [Real.add_one_le_exp ε]
    have hEF : Real.exp (Θ + ε) = E * F := Real.exp_add Θ ε
    rw [hEF]
    set e := |x - X| with he
    have hxM : |x| ≤ M + e := by
      have := abs_sub_abs_le_abs_sub x X
      linarith
    have h1 : |x' - (α * X + Y)| ≤ α * e + |x' - (α * x + Y)| := by
      have : x' - (α * X + Y) = (x' - (α * x + Y)) + α * (x - X) := by ring
      rw [this]
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul, abs_of_nonneg hα0]
      linarith
    have h2 : ε * (α * |x|) ≤ ε * (α * (M + e)) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hxM hα0) hε
    have h3 : |x' - (α * X + Y)| ≤ (1 + ε) * α * e + ε * (α * M + Z) + ρb * Z + κ * R
        + ηv * W + ηA := by nlinarith
    set B := (Θ + ρb) * M + κ * L + ηv * V + ηA * t with hB
    have hB0 : 0 ≤ B := by positivity
    have h4 : (1 + ε) * α * e ≤ (1 + ε) * α * (E * B) :=
      mul_le_mul_of_nonneg_left herr (by positivity)
    have hEB : (1 + ε) * α * (E * B) ≤ E * F * (α * B) := by
      have : (1 + ε) * α * (E * B) = (1 + ε) * (E * (α * B)) := by ring
      rw [this]
      have hEαB : 0 ≤ E * (α * B) := by positivity
      calc (1 + ε) * (E * (α * B)) ≤ F * (E * (α * B)) := mul_le_mul_of_nonneg_right hF1 hEαB
        _ = E * F * (α * B) := by ring
    have hEF1 : 1 ≤ E * F := by nlinarith
    have hEF0 : 0 ≤ E * F := by positivity
    -- the new closed form, term by term
    have hαM : 0 ≤ α * M := by positivity
    have hαL : 0 ≤ α * L := by positivity
    have hαV : α * V ≤ V := mul_le_of_le_one_left hV hα1
    have hαt : α * t ≤ t := mul_le_of_le_one_left ht hα1
    have hαB : α * B ≤ (Θ + ρb) * (α * M) + κ * (α * L) + ηv * V + ηA * t := by
      have : α * B = (Θ + ρb) * (α * M) + κ * (α * L) + ηv * (α * V) + ηA * (α * t) := by
        rw [hB]; ring
      rw [this]
      nlinarith [mul_le_mul_of_nonneg_left hαV hηv, mul_le_mul_of_nonneg_left hαt hηA]
    have h5 : E * F * (α * B) ≤ E * F * ((Θ + ρb) * (α * M) + κ * (α * L) + ηv * V + ηA * t) :=
      mul_le_mul_of_nonneg_left hαB hEF0
    -- the remaining terms are covered by `E * F ≥ 1`
    set A := (Θ + ρb) * (α * M) + κ * (α * L) + ηv * V + ηA * t with hA
    set K := ε * (α * M + Z) + ρb * Z + κ * R + ηv * W + ηA with hK
    set K' := ε * (α * M) + (ε + ρb) * Z + κ * R + ηv * W + ηA with hK'
    have hKK : K = K' := by rw [hK, hK']; ring
    have hK'0 : 0 ≤ K' := by positivity
    have h5 : E * F * (α * B) ≤ E * F * A := mul_le_mul_of_nonneg_left hαB hEF0
    have h6 : K' ≤ E * F * K' := by
      have := mul_le_mul_of_nonneg_right hEF1 hK'0
      linarith
    have h7 : E * F * (A + K') ≤ E * F * ((Θ + ε + ρb) * (α * M + Z) + κ * (α * L + R) +
        ηv * (V + W) + ηA * (t + 1)) := by
      refine mul_le_mul_of_nonneg_left ?_ hEF0
      have : 0 ≤ Θ * Z := by positivity
      have e : (Θ + ε + ρb) * (α * M + Z) + κ * (α * L + R) + ηv * (V + W) + ηA * (t + 1) =
          (A + K') + Θ * Z := by rw [hA, hK']; ring
      linarith
    have h8 := h4.trans hEB
    have h9 := h8.trans h5
    have h3' : |x' - (α * X + Y)| ≤ (1 + ε) * α * e + K := by rw [hK]; linarith
    linarith [h3', hKK, h9, h6, h7]
  · calc |α * X + Y| ≤ |α * X| + |Y| := abs_add_le _ _
      _ ≤ α * M + Z := by
        rw [abs_mul, abs_of_nonneg hα0]
        exact add_le_add (mul_le_mul_of_nonneg_left hXM hα0) hYZ

end Claims.AttnFwd
