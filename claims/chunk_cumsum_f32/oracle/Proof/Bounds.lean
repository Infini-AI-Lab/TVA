import Claim
import TVA.Proof

/-!
# Elementary arithmetic of the numeric bounds

`one_add_pow_le`: the first-order bound on `(1 + u)^m`, the only inequality the relaxations of
`Tree.errBound` need.
-/

namespace Claims.ChunkCumsum

open TVA

/-- `(1 + u)^m ≤ 1 + 2·m·u` whenever `2·m·u ≤ 1`. -/
theorem one_add_pow_le (u : ℚ) (hu0 : 0 ≤ u) (m : Nat) (hm : 2 * (m : ℚ) * u ≤ 1) :
    (1 + u) ^ m ≤ 1 + 2 * (m : ℚ) * u := by
  induction m with
  | zero => simp
  | succ k ih =>
    have hm' : 2 * ((k : ℚ) + 1) * u ≤ 1 := by push_cast at hm; linarith
    have hk : 2 * (k : ℚ) * u ≤ 1 := by
      have h1 : 2 * (k : ℚ) * u ≤ 2 * ((k : ℚ) + 1) * u := by nlinarith
      linarith
    have ih' := ih hk
    have hku : 2 * (k : ℚ) * u * u ≤ u := by
      have h1 := mul_le_mul_of_nonneg_right hk hu0
      simpa using h1
    have hmul := mul_le_mul_of_nonneg_right ih' (show (0 : ℚ) ≤ 1 + u by linarith)
    have hstep : (1 + 2 * (k : ℚ) * u) * (1 + u) ≤ 1 + 2 * ((k : ℚ) + 1) * u := by
      nlinarith
    rw [pow_succ]
    push_cast
    linarith

end Claims.ChunkCumsum
