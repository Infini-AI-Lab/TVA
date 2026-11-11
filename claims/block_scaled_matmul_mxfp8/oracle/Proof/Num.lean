import Claim
import TVA.Proof
import Proof.Basic
import Proof.Tol

/-!
# Numeric auxiliaries: the magnitude bound over the steps, monotonicity of `tol`, the room of the
accumulator in `float32`
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- `|partSum p t|` is at most the sum of the absolute values of its terms. -/
theorem abs_partSum_le (p : Nat → ℚ) (t : Nat) :
    |partSum p t| ≤ ∑ e ∈ Finset.range (128 * t), |p e| := by
  simp only [partSum]
  exact Finset.abs_sum_le_sum_abs _ _

/-- **The magnitude of a step**: `|partSum p t| + chunkAbs p t` is what one `tl.dot_scaled` step
sees, and it is at most the total `dotAbs p K` when the first `128 (t+1)` elements are inside. -/
theorem mag_le_dotAbs (p : Nat → ℚ) {K t : Nat} (h : 128 * (t + 1) ≤ K) :
    |partSum p t| + chunkAbs p t ≤ dotAbs p K := by
  have h1 : |partSum p t| ≤ ∑ e ∈ Finset.range (128 * t), |p e| := abs_partSum_le p t
  have hsplit : (∑ e ∈ Finset.range (128 * (t + 1)), |p e|) =
      (∑ e ∈ Finset.range (128 * t), |p e|) +
        ∑ e ∈ Finset.range 128, |p (128 * t + e)| := by
    rw [show 128 * (t + 1) = 128 * t + 128 from by ring, Finset.sum_range_add]
  have hmono : (∑ e ∈ Finset.range (128 * (t + 1)), |p e|) ≤ dotAbs p K := by
    simp only [dotAbs]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr h)
      (fun e _ _ => abs_nonneg _)
  have h2 : chunkAbs p t = ∑ e ∈ Finset.range 128, |p (128 * t + e)| := rfl
  linarith

/-- `tol` is nonnegative. -/
theorem tol_nonneg (p : Nat → ℚ) (t : Nat) : 0 ≤ tol p t := by
  induction t with
  | zero => simp [tol]
  | succ t ih =>
    have h1 : (0:ℚ) ≤ |partSum p t| := abs_nonneg _
    have h2 : (0:ℚ) ≤ chunkAbs p t := by
      simp only [chunkAbs]
      exact Finset.sum_nonneg (fun e _ => abs_nonneg _)
    have h3 : (0:ℚ) ≤ rho * (|partSum p t| + tol p t + chunkAbs p t) :=
      mul_nonneg (le_of_lt rho_pos) (by linarith)
    have h4 : (0:ℚ) ≤ alpha := alpha_nonneg
    simp only [tol]
    linarith

/-- `tol` does not decrease. -/
theorem tol_le_succ (p : Nat → ℚ) (t : Nat) : tol p t ≤ tol p (t + 1) := by
  have h1 : (0:ℚ) ≤ |partSum p t| := abs_nonneg _
  have h2 : (0:ℚ) ≤ chunkAbs p t := by
    simp only [chunkAbs]
    exact Finset.sum_nonneg (fun e _ => abs_nonneg _)
  have h3 : (0:ℚ) ≤ rho * (|partSum p t| + tol p t + chunkAbs p t) :=
    mul_nonneg (le_of_lt rho_pos) (by linarith [tol_nonneg p t])
  have h4 : (0:ℚ) ≤ alpha := alpha_nonneg
  simp only [tol]
  linarith

/-- **`tol` is monotone**: a longer run has the larger tolerance. -/
theorem tol_mono (p : Nat → ℚ) {t T : Nat} (h : t ≤ T) : tol p t ≤ tol p T :=
  Nat.le_induction (P := fun T _ => tol p t ≤ tol p T) le_rfl
    (fun m _ ih => le_trans ih (tol_le_succ p m)) T h

/-- From an error bound on a difference, a bound on the magnitude. -/
theorem abs_le_of_abs_sub_le {x y e : ℚ} (h : |x - y| ≤ e) : |x| ≤ |y| + e := by
  have h2 : |x| - |y| ≤ |x - y| := by
    have := abs_sub_abs_le_abs_sub x y
    linarith
  linarith

/-- **The room in `float32`**: an accumulator magnitude of at most `2^127` leaves room for one
`tl.dot_scaled` step of `128` products (relative error `rho`, absolute error `alpha`). -/
theorem room_le {x : ℚ} (hx : x ≤ 2 ^ (127 : ℤ)) :
    x + (rho * x + alpha) ≤ Format.binary32.maxVal := by
  have hrho : rho = 130 * 2 ^ (-22 : ℤ) := by
    unfold rho Device.relDot16
    norm_num
  have halpha : alpha = 130 * 2 ^ (-126 : ℤ) := by
    unfold alpha Device.absDot16
    norm_num
  rw [Format.binary32_maxVal, hrho, halpha]
  have hpos : (0:ℚ) ≤ 1 + 130 * 2 ^ (-22 : ℤ) := by norm_num
  have hx1 : x * (1 + 130 * 2 ^ (-22 : ℤ)) ≤ 2 ^ (127 : ℤ) * (1 + 130 * 2 ^ (-22 : ℤ)) :=
    mul_le_mul_of_nonneg_right hx hpos
  have hnum : 2 ^ (127 : ℤ) * (1 + 130 * 2 ^ (-22 : ℤ)) + 130 * 2 ^ (-126 : ℤ) ≤
      ((2:ℚ) ^ 24 - 1) * 2 ^ 104 := by norm_num
  have he : x + (x * (130 * 2 ^ (-22 : ℤ)) + 130 * 2 ^ (-126 : ℤ)) =
      x * (1 + 130 * 2 ^ (-22 : ℤ)) + 130 * 2 ^ (-126 : ℤ) := by ring
  linarith

end Claims.BlockScaledMxfp8
