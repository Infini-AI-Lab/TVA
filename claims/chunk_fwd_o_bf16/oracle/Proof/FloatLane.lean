import Claim
import TVA.Proof

/-!
# Float operations on one lane

The error bounds of the kernel's IEEE operations on a single lane, in `ℝ`, with the room stated
as a bound the proof's magnitudes meet: product, difference and fusable sum of `float32`, the
`float32 → bfloat16` conversion, and the host's `float32` scale.
-/

open TVA
namespace Claims.ChunkFwdO

/-- The unit roundoff of `binary32`, as a real, is `2^(-24)`. -/
theorem u32 : ((Format.binary32.u : ℚ) : ℝ) = 2 ^ (-24 : ℤ) := Format.binary32_u_real
/-- The smallest subnormal step of `binary32`, as a real, is `2^(-150)`. -/
theorem eta32 : ((Format.binary32.eta : ℚ) : ℝ) = 2 ^ (-150 : ℤ) := Format.binary32_eta_real

/-- The room: `2^105` is below `float32`'s largest value. -/
theorem two105_le_max32 : (2 : ℝ) ^ 105 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  rw [Format.binary32_maxVal_real]; norm_num

/-- The room: `2^105` is below `bfloat16`'s largest value. -/
theorem two105_le_maxbf : (2 : ℝ) ^ 105 ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
  rw [Format.bfloat16_maxVal_real]; norm_num

/-- `x * y` on `float32`, finite operands with a product below `2^105`. -/
theorem mul32_lane {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal * y.toReal| ≤ 2 ^ 105) (hr : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧
      |r.toReal - x.toReal * y.toReal| ≤ 2 ^ (-24 : ℤ) * |x.toReal * y.toReal| + 2 ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.mul_err_real Format.binary32_regular hx hy (hmax.trans two105_le_max32)
  obtain ⟨hf, -, hv⟩ := hr.exact_of_isFinite h1
  rw [Format.binary32_u_real] at h2
  rw [Format.binary32_eta_real] at h2
  refine ⟨hf, ?_⟩
  rw [hv]
  linarith [h2]

/-- `x - y` on `float32`, finite operands with a difference below `2^105`. -/
theorem sub32_lane {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal - y.toReal| ≤ 2 ^ 105) (hr : FloatV.NaNEq (FloatV.sub x y) r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal - y.toReal)| ≤ 2 ^ (-24 : ℤ) * |x.toReal - y.toReal| + 2 ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.sub_err_real Format.binary32_regular hx hy (hmax.trans two105_le_max32)
  obtain ⟨hf, -, hv⟩ := hr.exact_of_isFinite h1
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  refine ⟨hf, ?_⟩
  rw [hv]
  linarith [h2]

/-- `x + y` that a product may feed (`AddRel`, fused or not) on `float32`, with
`|x| + |y| ≤ 2^105`. -/
theorem addc32_lane {x y r : F32} {fusion : Bool} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hmax : |x.toReal| + |y.toReal| ≤ 2 ^ 105)
    (hr : AddRel fusion x y r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤
      3 * 2 ^ (-24 : ℤ) * (|x.toReal| + |y.toReal|) + 4 * 2 ^ (-150 : ℤ) := by
  have hsp : Format.binary32.specials ≠ .finite := by decide
  have hM : |x.toRat| + |y.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤ (((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ)) : ℝ) := by
      push_cast
      rw [Format.binary32_maxVal_real, Format.binary32_eta_real]
      have : |x.toReal| + |y.toReal| ≤ 2 ^ 105 := hmax
      unfold FloatV.toReal at this
      linarith [this, (by norm_num : (2:ℝ) ^ 105 + 2 * 2 ^ (-150 : ℤ) ≤ ((2 ^ 24 - 1) * 2 ^ 104) / 2)]
    exact_mod_cast h
  obtain ⟨hf, he⟩ := FloatV.AddRel.err Format.binary32_regular hsp
    (by rw [Format.binary32_u]; norm_num) hr hx hy hM
  refine ⟨hf, ?_⟩
  have he' : ((|r.toRat - (x.toRat + y.toRat)| : ℚ) : ℝ) ≤
      ((3 * Format.binary32.u * (|x.toRat| + |y.toRat|) + 4 * Format.binary32.eta : ℚ) : ℝ) := by
    exact_mod_cast he
  push_cast at he'
  rw [Format.binary32_u_real, Format.binary32_eta_real] at he'
  simpa [FloatV.toReal] using he'

/-- `float32 → bfloat16` (to nearest even) of a value below `2^105`. -/
theorem cvtbf_lane {x : F32} {r : BF16} (hx : x.isFinite = true) (hmax : |x.toReal| ≤ 2 ^ 105)
    (hr : FloatV.NaNEq (FloatV.cvtWith .rn x) r) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ 2 ^ (-8 : ℤ) * |x.toReal| + 2 ^ (-134 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hx
    (hmax.trans two105_le_maxbf)
  have hc : (FloatV.cvtWith .rn x : BF16) = FloatV.cvt x := rfl
  rw [hc] at hr
  obtain ⟨hf, -, hv⟩ := hr.exact_of_isFinite h1
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h2
  refine ⟨hf, ?_⟩
  rw [hv]
  linarith [h2]

/-- The host's `float32` scale (`FloatV.hostF32` of a finite double below `2^64`): finite and
within `|c|·2^-24 + 2^-150` of the double. -/
theorem scale_lane {c : F64} (hfin : c.isFinite = true) (h64 : |c.toReal| ≤ 2 ^ 64) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toReal - c.toReal| ≤ 2 ^ (-24 : ℤ) * |c.toReal| + 2 ^ (-150 : ℤ) := by
  have hnan : c.isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  have h : FloatV.hostF32 c = FloatV.cvt c := by
    unfold FloatV.hostF32; rw [hnan]; rfl
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.binary32) Format.binary32_regular hfin
    (h64.trans (by rw [Format.binary32_maxVal_real]; norm_num))
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  rw [h]
  exact ⟨h1, by linarith [h2]⟩

end Claims.ChunkFwdO
