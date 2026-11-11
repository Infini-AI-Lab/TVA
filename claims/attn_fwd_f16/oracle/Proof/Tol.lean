import Mathlib

/-!
# The constants of the error analysis

The fine bound of the proof, as functions of the head size `D`, the number of key blocks `T`
(trips of the key loop) and `X`, the bound on the row's scaled score magnitudes. They are kept
as plain real expressions: the lemmas of `Proof.Lane`, `Proof.Trip` and `Proof.Row` produce
them, and `Proof.Relax` shows them below the claim's `epsO` and `epsM`.

* `xi D X`: the error of a lane's exponent `y ≈ score − max` (base 2), from the score's `tl.dot`,
  the scale, the products and subtractions;
* `rhoP`, `rho16`, `rbarA`, `rbarL`: the relative errors of the weights `p` (in `float32`, in
  `float16`), as they enter the accumulator and the row sum;
* `ThA`, `ThL`: the sums of the per-trip rates (rounding of the rescaling, of the accumulation,
  and the relative error of the computed rescaling factor, whose exponent errors telescope);
* `lamL`: the relative error of the row sum; `tolO`, `tolS`, `tolF0`: the error of `acc / l`
  as `tolO·attnAbs + tolS·Σ|v| + tolF0`; `tolO2`, `tolS2`, `tolF2`: after the division and the
  rounding to `float16`; `tolM`: the error of `M`.
-/

namespace Claims.AttnFwd

noncomputable section

/-- The unit roundoff of `float32`. -/
def u32 : ℝ := 1 / 2 ^ 24
/-- The relative error of `exp2`, `log2` and `/` on `float32` (the device facts). -/
def q22 : ℝ := 1 / 2 ^ 22
/-- The unit roundoff of `float16`. -/
def u16 : ℝ := 1 / 2 ^ 11
/-- Half the smallest subnormal of `float16`. -/
def e16 : ℝ := 1 / 2 ^ 25
/-- `relDot16 64`, the relative error of the `64`-term `tl.dot` of `p · v`. -/
def r64 : ℝ := 66 / 2 ^ 22
/-- `relDot16 D`, the relative error of the score's `tl.dot`. -/
def rD (D : ℕ) : ℝ := ((D : ℝ) + 2) / 2 ^ 22

/-- The error of a lane's exponent (base 2): the score `tl.dot`, the scale, the product and the
subtraction of the maximum, for scaled score magnitudes at most `X`. -/
def xi (D : ℕ) (X : ℝ) : ℝ := X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40

/-- A `float32` weight against the exact one: relative error. -/
def rhoP (D : ℕ) (X : ℝ) : ℝ := xi D X * (1 + xi D X) + q22 * (1 + xi D X * (1 + xi D X))
/-- The relative error of a `float32` sum along a tree of depth `6`. -/
def rhoS : ℝ := (1 + u32) ^ 6 - 1
/-- The relative error of a weight rounded to `float16`. -/
def rho16 (D : ℕ) (X : ℝ) : ℝ := rhoP D X + u16 * (1 + rhoP D X)
/-- The weights' relative error as it enters the accumulator. -/
def rbarA (D : ℕ) (X : ℝ) : ℝ := rho16 D X * (1 + r64)
/-- The weights' relative error as it enters the row sum. -/
def rbarL (D : ℕ) (X : ℝ) : ℝ := (rhoP D X + rhoS * (1 + rhoP D X)) * (1 + 3 * u32)

/-- The absolute error of a weight in `float16`. -/
def eta16 : ℝ := 1 / 2 ^ 25 + 1 / 2 ^ 124
/-- The accumulator's absolute error per unit of `Σ|v|`. -/
def etavA : ℝ := eta16 * (1 + r64) + r64 * (1 / 2 ^ 14)
/-- The accumulator's absolute error per unit of the weight sum (the lift of subnormal weights). -/
def kapA (D : ℕ) (X : ℝ) : ℝ := r64 * (1 / 2 ^ 14) * (1 + rho16 D X)
/-- The accumulator's absolute error per trip. -/
def etaAA : ℝ := 1 / 2 ^ 37
/-- The row sum's absolute error per trip. -/
def etaAL : ℝ := 1 / 2 ^ 100

/-- The accumulator's per-trip rate. -/
def EPSA : ℝ := 17 / 2 ^ 20
/-- The row sum's per-trip rate. -/
def EPSL : ℝ := 15 / 2 ^ 25
/-- The relative error of a computed rescaling factor per unit of growth of the reference
(`ln 2 · u32`, with room): the rounding of the subtraction `m_old − m_new` in its exponent. -/
def CA : ℝ := 6936 / 10000 * u32
/-- `CA` as it enters the rate sums (the rates are affine in the rescaling's error). -/
def CD : ℝ := 7 / 10 * u32

/-- A bound on the magnitude of a reference (base 2): `1/ln 2` times `X`, with room. -/
def Bm (X : ℝ) : ℝ := 1443 / 1000 * X + 1 / 2 ^ 40

/-- The accumulator's rate sum after `T` trips: the per-trip rates, plus the drift of the
rescaling factors' exponents (the references never decrease, so these errors telescope to at most
`CD` times the range `2·Bm X` of the references). -/
def ThA (T X : ℝ) : ℝ := T * EPSA + CD * (2 * Bm X)
/-- The row sum's rate sum after `T` trips. -/
def ThL (T X : ℝ) : ℝ := T * EPSL + CD * (2 * Bm X)
/-- A bound on `exp Θ` for `0 ≤ Θ ≤ 1`. -/
def EA (t : ℝ) : ℝ := 1 + t + t * t

/-- A lower bound on the row sum at the final reference (`2^(−ξ) ≥ 1 − ξ`). -/
def wlow (D : ℕ) (X : ℝ) : ℝ := 1 - xi D X
/-- The relative error of the row sum (its absolute part counted against `wlow`). -/
def lamL (D : ℕ) (T X : ℝ) : ℝ :=
  EA (ThL T X) * (ThL T X + rbarL D X) + EA (ThL T X) * etaAL * T / wlow D X

/-- The coefficient of `attnAbs` in the error of `acc / l`. -/
def tolO (D : ℕ) (T X : ℝ) : ℝ :=
  (EA (ThA T X) * (ThA T X + rbarA D X) + lamL D T X) / (1 - lamL D T X)
/-- The coefficient of `Σ|v|` in the error of `acc / l`. -/
def tolS (D : ℕ) (T X : ℝ) : ℝ :=
  EA (ThA T X) * etavA / (wlow D X * (1 - lamL D T X))
/-- The constant part of the error of `acc / l`. -/
def tolF0 (D : ℕ) (T X : ℝ) : ℝ :=
  EA (ThA T X) * kapA D X / (1 - lamL D T X) +
    EA (ThA T X) * etaAA * T / (wlow D X * (1 - lamL D T X))

/-- The relative error of the quotient rounded to `float16`: `/` then the conversion. -/
def lastRel : ℝ := u16 * (1 + q22) + q22
/-- The coefficient of `attnAbs` in the error of the output. -/
def tolO2 (D : ℕ) (T X : ℝ) : ℝ := lastRel * (1 + tolO D T X) + tolO D T X
/-- The coefficient of `Σ|v|` in the error of the output. -/
def tolS2 (D : ℕ) (T X : ℝ) : ℝ := tolS D T X * (1 + lastRel)
/-- The constant part of the error of the output. -/
def tolF2 (D : ℕ) (T X : ℝ) : ℝ :=
  tolF0 D T X * (1 + lastRel) + u16 * (1 / 2 ^ 148) + e16 + 1 / 2 ^ 148

/-- The error of `M = m + log₂ l`. -/
def tolM (D : ℕ) (T X : ℝ) : ℝ :=
  14427 / 10000 * (lamL D T X / (1 - lamL D T X)) + 18 * q22 + 1 / 2 ^ 149 +
    u32 * (Bm X + 19) + 1 / 2 ^ 150

/-- The claim's tolerance of the output, with `N` the context length. -/
def epsO' (D : ℕ) (N X : ℝ) : ℝ := (D * X + N + 2 ^ 10) / 2 ^ 20
/-- The claim's tolerance of `M`. -/
def epsM' (D : ℕ) (X : ℝ) : ℝ := (D * X + 2 ^ 10) / 2 ^ 20

end

end Claims.AttnFwd
