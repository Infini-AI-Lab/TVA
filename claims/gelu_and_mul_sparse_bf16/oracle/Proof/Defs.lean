import Claim
import TVA.Proof

/-!
# Shared definitions of the proof

The rounding constants as reals, the logistic function and the GELU's argument (`Proof.Gelu` proves
`geluRef x = x · sig (Xof x)`), and `RowP`: one row's exact data, on which `Proof.Row` bounds the
kernel's mean, standard deviation and cutoff.
-/

namespace Claims.GeluAndMulSparse

open TVA

/-- `u` of `float32`, `2^-24`. -/
noncomputable def u32 : ℝ := 2 ^ (-24 : ℤ)
/-- `η` of `float32`, `2^-150`. -/
noncomputable def eta32 : ℝ := 2 ^ (-150 : ℤ)
/-- `u` of `bfloat16`, `2^-8`. -/
noncomputable def ub : ℝ := 2 ^ (-8 : ℤ)
/-- `η` of `bfloat16`, `2^-134`. -/
noncomputable def etab : ℝ := 2 ^ (-134 : ℤ)

/-- The logistic function `1 / (1 + e^(-z))`. -/
noncomputable def sig (z : ℝ) : ℝ := 1 / (1 + Real.exp (-z))

/-- The source's `0.7978845608028654`, as a real. -/
noncomputable def kR : ℝ := 0.7978845608028654
/-- The source's `0.044715`, as a real. -/
noncomputable def aR : ℝ := 0.044715

/-- The logistic's argument in `geluRef x = x · sig (Xof x)`: `2·k·(x + a·x³)`. -/
noncomputable def Xof (x : ℝ) : ℝ := 2 * kR * (x + aR * x ^ 3)

/-- One row's exact data: `d` gate values `g j`, the block `2^L` (`L ≤ 15`), the multiplier `m`. -/
structure RowP where
  d : ℕ
  L : ℕ
  g : ℕ → ℝ
  m : ℝ
  d_pos : 1 ≤ d
  d_le : d ≤ 2 ^ L
  b_lt : 2 ^ L < 2 * d
  L_le : L ≤ 15
  g_le : ∀ j, j < d → |g j| ≤ 2 ^ 32
  m_le : |m| ≤ 64

namespace RowP

variable (r : RowP)

/-- The exact mean. -/
noncomputable def mu : ℝ := (∑ j ∈ Finset.range r.d, r.g j) / r.d
/-- The exact mean magnitude. -/
noncomputable def aMean : ℝ := (∑ j ∈ Finset.range r.d, |r.g j|) / r.d
/-- The exact (biased) standard deviation. -/
noncomputable def sd : ℝ := Real.sqrt ((∑ j ∈ Finset.range r.d, (r.g j - r.mu) ^ 2) / r.d)
/-- The exact cutoff. -/
noncomputable def cRef : ℝ := r.mu + r.m * r.sd
/-- The claim's tolerance of the cutoff. -/
noncomputable def cutTol : ℝ :=
  2 ^ (-6 : ℤ) * (|r.mu| + |r.m| * r.sd) + 2 ^ (-17 : ℤ) * (1 + |r.m|) * r.aMean + 2 ^ (-130 : ℤ)
/-- The block's lanes: `g j` below `d`, `0` beyond. -/
noncomputable def gl (j : ℕ) : ℝ := if j < r.d then r.g j else 0
/-- The row is not nearly constant (the claim's `RowOk`). -/
def Ok : Prop := 2 ^ (-18 : ℤ) * r.aMean + 2 ^ (-62 : ℤ) ≤ r.sd

/-- The bound on the kernel's `float32` mean. -/
noncomputable def meanErr : ℝ := ((r.L : ℝ) + 5) * u32 * r.aMean + 2 ^ (-134 : ℤ)
/-- The bound on the kernel's `float32` standard deviation. -/
noncomputable def stdErr : ℝ := ((r.L : ℝ) + 12) * u32 * r.aMean + 40 * u32 * r.sd

end RowP

end Claims.GeluAndMulSparse
