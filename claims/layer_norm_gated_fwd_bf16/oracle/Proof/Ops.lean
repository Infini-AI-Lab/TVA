import TVA.Proof

/-!
# `layer_norm_gated_fwd_bf16`: one `float32` operation at a time

The IEEE operations the walk meets, as the `Safe` rules give them (`NaNEq` to the library's
operation), with the error of one rounding on real values: `float32` products and sums with room
below `2^127` (`mul32`, `add32`), the exact widening of a `bfloat16` (`widen16`) and the rounding
of a `float32` to `bfloat16` (`cvt16`).
-/

namespace Claims.LayerNormGatedFwd

open TVA

/-- `2^127 ≤ maxVal` of `float32`. -/
theorem maxVal32_ge : (2 : ℝ) ^ (127 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  rw [Format.binary32_maxVal_real]; norm_num

/-- `2^127 ≤ maxVal` of `bfloat16`. -/
theorem maxVal16_ge : (2 : ℝ) ^ (127 : ℤ) ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
  rw [Format.bfloat16_maxVal]; push_cast; norm_num

/-- A `float32` product with room: finite, within `2^-24·|x·y| + 2^-150`. -/
theorem mul32 {x y r : F32} (h : FloatV.NaNEq (FloatV.mul x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hmax : |x.toReal * y.toReal| ≤ (2 : ℝ) ^ (127 : ℤ)) :
    r.isFinite = true ∧
      |r.toReal - x.toReal * y.toReal| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x.toReal * y.toReal| +
        (2 : ℝ) ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.mul_err_real Format.binary32_regular hx hy (hmax.trans maxVal32_ge)
  obtain ⟨h3, -, h5⟩ := h.exact_of_isFinite h1
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  refine ⟨h3, ?_⟩
  rw [h5]; linarith

/-- A `float32` sum with room: finite, within `2^-24·|x + y| + 2^-150`. -/
theorem add32 {x y r : F32} (h : FloatV.NaNEq (FloatV.add x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hmax : |x.toReal + y.toReal| ≤ (2 : ℝ) ^ (127 : ℤ)) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x.toReal + y.toReal| +
        (2 : ℝ) ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.add_err_real Format.binary32_regular hx hy (hmax.trans maxVal32_ge)
  obtain ⟨h3, -, h5⟩ := h.exact_of_isFinite h1
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  refine ⟨h3, ?_⟩
  rw [h5]; linarith

/-- The widening of a finite `bfloat16` to `float32` is exact. -/
theorem widen16 {b : BF16} {x : F32} (h : FloatV.NaNEq (FloatV.cvtWith .rn b) x)
    (hb : b.isFinite = true) :
    x.isFinite = true ∧ x.toReal = b.toReal ∧ x.toRat = b.toRat := by
  obtain ⟨h1, h2, h3⟩ := FloatV.cvt_bf16_f32_exact hb
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨h4, h5, h6⟩ := h.exact_of_isFinite h1
  exact ⟨h4, by rw [h6, h3], by rw [h5, h2]⟩

/-- The rounding of a `float32` with room to `bfloat16`: finite, within `2^-8·|x| + 2^-134`. -/
theorem cvt16 {x : F32} {c : BF16} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) c)
    (hx : x.isFinite = true) (hmax : |x.toReal| ≤ (2 : ℝ) ^ (127 : ℤ)) :
    c.isFinite = true ∧ |c.toReal - x.toReal| ≤ (2 : ℝ) ^ (-8 : ℤ) * |x.toReal| + (2 : ℝ) ^ (-134 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hx
    (hmax.trans maxVal16_ge)
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨h3, -, h5⟩ := h.exact_of_isFinite h1
  have hu : ((Format.bfloat16.u : ℚ) : ℝ) = (2 : ℝ) ^ (-8 : ℤ) := by
    rw [Format.bfloat16_u]; push_cast; rfl
  have he : ((Format.bfloat16.eta : ℚ) : ℝ) = (2 : ℝ) ^ (-134 : ℤ) := by
    rw [Format.bfloat16_eta]; push_cast; rfl
  rw [hu, he] at h2
  refine ⟨h3, ?_⟩
  rw [h5]; linarith

end Claims.LayerNormGatedFwd
