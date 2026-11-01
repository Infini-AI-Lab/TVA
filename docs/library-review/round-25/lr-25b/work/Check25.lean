import TVA

/-! lr-25b: round 25's lemmas and facts. Axioms of the new and changed theorems, and kernel-style uses
of the restated `DotF16`: on zero and normal `float16` values it gives the pre-round-25 `DotAcc`
bound; `liftMag` at `float16` and `bfloat16` is what the page says; and the `float16` lift adds at most
`2^-14·Σ|a| + 2^-14·Σ|b| + K·2^-28` to the magnitudes. -/

open TVA TVA.Rules.Num TVA.Math.Float

#print axioms TVA.Rules.Num.DotLiftAcc.mono
#print axioms TVA.Rules.Num.DotAcc.liftAcc
#print axioms TVA.DotLoop.dotAdd_step_of
#print axioms TVA.DotLoop.dotAdd_step
#print axioms TVA.DotLift.dotAdd_step
#print axioms TVA.DotLift.liftMag_mul_le
#print axioms TVA.DotLift.sum_liftMag_mul_le
#print axioms TVA.DotLift.DotLiftAcc.dotAcc
#print axioms TVA.Platform.ref.dotLiftAcc
#print axioms TVA.Platform.sm90ref.dotF16
#print axioms TVA.Platform.sm90ref.dotBF16
#print axioms TVA.Dot8Chunk.DotF8Chunk.liftAcc

/-- On zero and normal `float16` values (`|q| ≥ 2^-14`), `DotF16` is the old `DotAcc` bound. -/
example (P : Platform) (h : DotF16 P) (p : InputPrecision) :
    DotAcc P ⟨.f16, .f16, .f32, p, 0⟩ (fun q => q = 0 ∨ (2 : ℚ) ^ (-14 : ℤ) ≤ |q|) relDot16 absDot16 := by
  have hl : ∀ q : ℚ, (q = 0 ∨ (2 : ℚ) ^ (-14 : ℤ) ≤ |q|) → liftMag .f16 q = |q| := by
    intro q hq
    rcases hq with rfl | hq
    · simp [liftMag]
    · exact Dot8Chunk.liftMag_of_le (by simpa [minNormal8] using hq)
  exact DotLift.DotLiftAcc.dotAcc (h p) hl hl

/-- A subnormal `float16` value counts as `2^-14`, a subnormal `bfloat16` one as `2^-126`. -/
example : liftMag .f16 (2 ^ (-24 : ℤ)) = 2 ^ (-14 : ℤ) := by
  unfold liftMag minNormal8
  rw [if_neg (by positivity), abs_of_pos (by positivity)]
  exact max_eq_right (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

example : liftMag .bf16 (2 ^ (-133 : ℤ)) = 2 ^ (-126 : ℤ) := by
  unfold liftMag minNormal8
  rw [if_neg (by positivity), abs_of_pos (by positivity)]
  exact max_eq_right (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

/-- The `float16` lift against the plain magnitudes. -/
example {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ k, liftMag .f16 (va k) * liftMag .f16 (vb k) ≤
      ∑ k, |va k * vb k| + 2 ^ (-14 : ℤ) * ∑ k, |va k| + 2 ^ (-14 : ℤ) * ∑ k, |vb k| +
        K * (2 ^ (-14 : ℤ) * 2 ^ (-14 : ℤ)) := by
  simpa [DotLift.minNormal8_f16] using DotLift.sum_liftMag_mul_le .f16 .f16 va vb
