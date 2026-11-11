import Claim
import TVA.Proof

/-!
# The constant `zf32`

The kernel's `+0.0` (`other=0.0`, the `tl.where` value `0.0`) is the `float32` `+0`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The constant `zf32` is the `float32` `+0`. -/
theorem zf32_eq : zf32 = (FloatV.ofPat 0 : F32) := by
  unfold zf32 FloatV.litF32
  have h : (FloatV.ofPat 0 : F64).isZero = true := by
    rw [FloatV.isZero_ofPat (Nat.two_pow_pos _)]
    exact Format.isZero_zero
  simp [h]

end Claims.ChunkBwdDvLocal
