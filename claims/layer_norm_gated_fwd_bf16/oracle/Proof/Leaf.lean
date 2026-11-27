import Proof.Defs
import Proof.RowSum
import Proof.RstdRow
import Proof.Gate
import Proof.Elem

/-!
# `layer_norm_gated_fwd_bf16`: the result of one program

After the walk (`Proof.Walk`), the program has stored its `rstd` rows and its tile of `y`. This is
the leaf: from the values the walk computed and the facts the `Safe` rules gave about them, prove
`Qn d init p m` for the final memory `m` (`qn`).

1. Per lane of a row, the `float32` square of `b_xbar` is within `u·a² + η` of the exact `a²` of
   the row's input `a` (`+0` on the padded lanes `j ≥ D`), so the row's sum is within
   `(7/8)·2^-13·A + (11/16)·2^-137` of `A = Σ x_j²` (`sum_err_bd`), and `rstd` within `2^-12`
   (relative) of `rstdRef` (`rstd_row`).
2. Per element `j < D`: `|x·rstdRef| ≤ 23` (`z_bound`), the gate within `2^-15·σ + 2^-128`
   (`gate_err`), and the stored value within `yTol` (`elem_swish`, `elem_sigmoid`).
-/

set_option maxHeartbeats 4000000

namespace Claims.LayerNormGatedFwd

open TVA
open TVA TVA.Format

/-- What lines 121–124 compute on one lane from `y0 = b_x_hat·w` and the gate `g`: `tl.sigmoid(g)`
(the negation, `tl.exp`, `1 + …`, `1 / …`) and, for swish, `fl(fl(y0·g)·s)`, for sigmoid,
`fl(y0·s)`. -/
def GateOut (sw : Bool) (P : Platform) (y0 g y : F32) : Prop :=
  ∃ nx ex den s : F32, FloatV.NaNEq (FloatV.neg g) nx ∧ P.math .exp .binary32 nx ex ∧
    FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den ∧ P.div .binary32 (FloatV.ofRat 1) den s ∧
    (if sw then ∃ yg : F32, FloatV.NaNEq (FloatV.mul y0 g) yg ∧ FloatV.NaNEq (FloatV.mul yg s) y
     else FloatV.NaNEq (FloatV.mul y0 s) y)

/-- `|x·(√(A/D + e))⁻¹| ≤ 23` when `x² ≤ A`, `1 ≤ D ≤ 512`, `e > 0`: `(x·R)² ≤ D`. -/
theorem z_bound {x A D e : ℝ} (hx2 : x ^ 2 ≤ A) (hD1 : 1 ≤ D) (hD : D ≤ 512) (he : 0 < e) :
    |x * (Real.sqrt (A / D + e))⁻¹| ≤ 23 := by
  have hA0 : 0 ≤ A := le_trans (sq_nonneg x) hx2
  have hDpos : 0 < D := by linarith
  have hV : 0 < A / D + e := by have := div_nonneg hA0 hDpos.le; linarith
  have hs := Real.sq_sqrt hV.le
  have hspos := Real.sqrt_pos.mpr hV
  set R := (Real.sqrt (A / D + e))⁻¹ with hR
  have hRpos : 0 < R := inv_pos.mpr hspos
  have hR2 : R ^ 2 * (A / D + e) = 1 := by
    rw [hR, inv_pow, hs, inv_mul_cancel₀ (ne_of_gt hV)]
  have hz2 : (x * R) ^ 2 ≤ D := by
    have h1 : (x * R) ^ 2 * (A / D + e) = x ^ 2 := by
      calc (x * R) ^ 2 * (A / D + e) = x ^ 2 * (R ^ 2 * (A / D + e)) := by ring
        _ = x ^ 2 := by rw [hR2, mul_one]
    have h2 : (x * R) ^ 2 * (A / D) ≤ A := by
      have := mul_le_mul_of_nonneg_left he.le (sq_nonneg (x * R))
      nlinarith
    rcases eq_or_lt_of_le hA0 with h0 | h0
    · have hx0 : x = 0 := by
        have : x ^ 2 ≤ 0 := by rw [h0]; exact hx2
        exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp (le_antisymm this (sq_nonneg x))
      rw [hx0]; simp; linarith
    · have h3 : (x * R) ^ 2 * A ≤ D * A := by
        have e1 : (x * R) ^ 2 * (A / D) * D = (x * R) ^ 2 * A := by field_simp
        have := mul_le_mul_of_nonneg_right h2 hDpos.le
        nlinarith
      exact le_of_mul_le_mul_right h3 h0
  have h529 : (x * R) ^ 2 ≤ 23 ^ 2 := by nlinarith
  nlinarith [abs_nonneg (x * R), sq_abs (x * R)]

end Claims.LayerNormGatedFwd
