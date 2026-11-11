import Claim
import TVA.Proof

/-!
# Facts about the constants and the random numbers of `seeded_dropout`

The `float32` constants of the claim (`one32`, `Philox.scale32`), the grid of `float32`
values next to `1`, the divisor `dRef`, and the integer bound on `rint`.
-/

namespace Claims.SeededDropout

open TVA
open TVA.Format

/-! ## The constants -/

/-- `1 : float32` is finite. -/
theorem one32_isFinite : (one32 : F32).isFinite = true := by decide +kernel

/-- The value of `1 : float32` is `1`. -/
theorem one32_toRat : (one32 : F32).toRat = 1 := by decide +kernel

/-- `Philox.scale32` is finite. -/
theorem scale32_isFinite : (Philox.scale32 : F32).isFinite = true := by decide +kernel

/-- `Philox.scale32` is `2^-31 · (1 - 2^-24)` (pattern `0x2FFFFFFF`). -/
theorem scale32_toRat :
    (Philox.scale32 : F32).toRat = 2 ^ (-31 : ℤ) * (1 - 2 ^ (-24 : ℤ)) := by decide +kernel

/-! ## The grid of `float32` next to `1` -/

/-- The value of the pattern `0x3F7FFFFF`, the `float32` just below `1`. -/
theorem toRat_3F7FFFFF :
    Format.binary32.toRat (0x3F7FFFFF : Nat) = 1 - 2 ^ (-24 : ℤ) := by decide +kernel

/-- The value of the pattern `0x3F800000` is `1`. -/
theorem toRat_3F800000 : Format.binary32.toRat (0x3F800000 : Nat) = 1 := by decide +kernel

/-- The order key of `0x3F7FFFFF`. -/
theorem orderKey_3F7FFFFF : Format.binary32.orderKey (0x3F7FFFFF : Nat) = 1065353215 := by
  decide +kernel

/-- The order key of `0x3F800000`. -/
theorem orderKey_3F800000 : Format.binary32.orderKey (0x3F800000 : Nat) = 1065353216 := by
  decide +kernel

/-- **No `float32` lies strictly between `1 - 2^-24` and `1`**: a value below `1` is at most
`1 - 2^-24`. -/
theorem toRat_le_of_lt_one {x : F32} (h : x.toRat < 1) : x.toRat ≤ 1 - 2 ^ (-24 : ℤ) := by
  have hkey : Format.binary32.orderKey x.pat < Format.binary32.orderKey (0x3F800000 : Nat) := by
    rw [← Format.toRat_lt_iff_orderKey x.pat (0x3F800000 : Nat), toRat_3F800000]
    exact h
  have hle : Format.binary32.orderKey x.pat ≤ 1065353215 := by
    rw [orderKey_3F800000] at hkey
    omega
  have hnot : ¬ (Format.binary32.toRat (0x3F7FFFFF : Nat) < x.toRat) := by
    intro hc
    have hlt := (Format.toRat_lt_iff_orderKey (0x3F7FFFFF : Nat) x.pat).mp hc
    rw [orderKey_3F7FFFFF] at hlt
    omega
  have hle' : x.toRat ≤ Format.binary32.toRat (0x3F7FFFFF : Nat) := not_lt.mp hnot
  rwa [toRat_3F7FFFFF] at hle'

/-! ## The integer bounds -/

/-- `rint` is non-negative, below `2^31`. -/
theorem rint_lt (seed : Int) (j : Nat) : rint seed j < 2 ^ 31 := by
  have hr : (rbits seed j).toNat < 2 ^ 32 := (rbits seed j).isLt
  unfold rint
  dsimp only
  split_ifs with h
  · exact h
  · omega

/-! ## The comparison -/

/-! ## The divisor `1 - p` -/

/-- The divisor is the IEEE difference `1 - p`. -/
theorem dRef_eq (d : Inst) : dRef d = FloatV.sub one32 (p32 d) := rfl

end Claims.SeededDropout
