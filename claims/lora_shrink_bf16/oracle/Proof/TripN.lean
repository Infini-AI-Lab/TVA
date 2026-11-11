import Proof.TripC

/-!
# `lora_shrink_bf16`: the numeric room of a trip

The constants of the device facts are small against `float32`'s range, and the loop's error stays
a fraction of its magnitude, so every fact of the device has room when `dotMag ≤ 2^120`.
-/

namespace Claims.LoraShrink

open TVA

/-- `float32` holds `2^127`. -/
theorem two127_le_maxVal : (2 : ℚ) ^ 127 ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

/-- `Device.relDot16` of a block is at most `1`. -/
theorem relDot16_le (d : Inst) : Device.relDot16 (bk d) ≤ 1 := by
  unfold Device.relDot16
  rcases cfg4 d with ⟨h1, -⟩ | ⟨h1, -⟩ <;> rw [h1] <;> norm_num

/-- `Device.absDot16` of a block is at most `1`. -/
theorem absDot16_le (d : Inst) : Device.absDot16 (bk d) ≤ 1 := by
  unfold Device.absDot16
  rcases cfg4 d with ⟨h1, -⟩ | ⟨h1, -⟩ <;> rw [h1] <;> norm_num

/-- `Device.relDot16` of a block is nonnegative. -/
theorem relDot16_nonneg (d : Inst) : 0 ≤ Device.relDot16 (bk d) := by
  unfold Device.relDot16; positivity

/-- `Device.absDot16` of a block is nonnegative. -/
theorem absDot16_nonneg (d : Inst) : 0 ≤ Device.absDot16 (bk d) := by
  unfold Device.absDot16; positivity

/-- The chunk facts' room constants of a block are at most `1`. -/
theorem chunk_room_le (d : Inst) :
    Device.relChunk16 (bk d) ≤ 1 ∧ Device.absChunk16 (bk d) ≤ 1 := by
  unfold Device.relChunk16 Device.absChunk16 Device.rhoChunk16 Device.alphaChunk16
  rw [chunks16_bk]
  rcases cfg4 d with ⟨-, -, h3, -⟩ | ⟨-, -, h3, -⟩ <;> rw [h3] <;> norm_num

/-- The step-by-step bound stays within a quarter of the magnitude, plus `1`. -/
theorem bound_small (d : Inst) {t : Nat} (ht : t ≤ 512) {B : ℚ} (hB : 0 ≤ B) :
    DotLoop.bound (rho d) (alpha d) t B ≤ B / 4 + 1 := by
  have hr := rho_le d
  have ha := alpha_le d
  have hr0 := rho_nonneg d
  have ha0 := alpha_nonneg d
  have ht' : (t : ℚ) ≤ 512 := by exact_mod_cast ht
  have ht0 : (0 : ℚ) ≤ t := Nat.cast_nonneg t
  have hr' : rho d ≤ 1 / 4096 := by
    refine hr.trans (le_of_eq ?_); norm_num
  have ha' : alpha d ≤ 1 / 2 ^ 100 := by
    refine ha.trans (le_of_eq ?_); norm_num
  have h1 : (t : ℚ) * rho d ≤ 1 / 2 := by nlinarith
  have := DotLoop.bound_le hr0 ha0 h1 hB
  have h2 : 2 * (t : ℚ) * (rho d * B) ≤ B / 4 := by nlinarith [mul_nonneg ht0 hB]
  have h3 : 2 * (t : ℚ) * alpha d ≤ 1 := by nlinarith
  nlinarith

end Claims.LoraShrink
