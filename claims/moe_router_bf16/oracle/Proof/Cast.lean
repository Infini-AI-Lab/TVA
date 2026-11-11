import Claim
import TVA.Proof

/-!
# Casts: `binary32`'s rounding unit and smallest-error constants, in `ℝ`

The library's error lemmas are stated in `ℚ`; the claim's tolerances are in `ℝ`. These are the
identities that move between them, plus the basic positivity facts.
-/

namespace Claims.MoeRouter

open TVA

/-- `binary32` is a regular format. -/
theorem bin32_regular : Format.binary32.Regular :=
  Format.binary32_regular

/-- `binary32` has infinities and NaNs. -/
theorem bin32_specials : Format.binary32.specials ≠ .finite := by
  simp [Format.binary32]

/-- `u ≤ 1/16` for `binary32`. -/
theorem bin32_u_le : Format.binary32.u ≤ 1 / 16 := by
  rw [Format.binary32_u]
  norm_num

/-- `0 < u` for `binary32`. -/
theorem bin32_u_pos : (0 : ℚ) < Format.binary32.u := by
  rw [Format.binary32_u]
  norm_num

/-- The library's `u`, as the claim's `u32`. -/
theorem u32_eq : ((Format.binary32.u : ℚ) : ℝ) = u32 := by
  rw [Format.binary32_u]
  norm_num [u32]

/-- The library's `eta`, as the claim's `eta32`. -/
theorem eta32_eq : ((Format.binary32.eta : ℚ) : ℝ) = eta32 := by
  rw [Format.binary32_eta]
  norm_num [eta32]

/-- `0 < u32`. -/
theorem u32_pos : (0 : ℝ) < u32 := by
  norm_num [u32]

/-- `0 < eta32`. -/
theorem eta32_pos : (0 : ℝ) < eta32 := by
  norm_num [eta32]

/-- `u32 ≤ 1/16`. -/
theorem u32_le : u32 ≤ 1 / 16 := by
  norm_num [u32]

/-- The largest finite `binary32`, in `ℝ`. -/
theorem maxVal32_eq :
    ((Format.binary32.maxVal : ℚ) : ℝ) = (2 : ℝ) ^ (128 : ℤ) - (2 : ℝ) ^ (104 : ℤ) := by
  rw [Format.binary32_maxVal]
  norm_num


end Claims.MoeRouter
