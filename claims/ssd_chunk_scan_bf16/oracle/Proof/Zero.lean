import Proof.Defs

/-!
# The zero constants

The loop and the epilogue fill masked-off lanes from `zf32`, the `float32` `+0`, or from it
converted to `bfloat16`. This file collects what the proof needs of them: their value, finiteness,
and the conversions and products with zero.
-/

namespace Claims.SsdChunkScan

open TVA

/-- The `float32` zero pattern has value `0`. -/
theorem F32_zero_toRat : (FloatV.ofPat (0 : Nat) : F32).toRat = 0 := by
  rw [FloatV.toRat_ofPat (by decide)]
  exact (Format.toRat_eq_zero_iff).mpr Format.isZero_zero

/-- The `float32` zero pattern is finite. -/
theorem F32_zero_finite : (FloatV.ofPat (0 : Nat) : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  exact Format.isFinite_zero Format.binary32_regular

/-- The `bfloat16` zero pattern has value `0`. -/
theorem BF16_zero_toRat : (FloatV.ofPat (0 : Nat) : BF16).toRat = 0 := by
  rw [FloatV.toRat_ofPat (by decide)]
  exact (Format.toRat_eq_zero_iff).mpr Format.isZero_zero

/-- The `bfloat16` zero pattern is finite. -/
theorem BF16_zero_finite : (FloatV.ofPat (0 : Nat) : BF16).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  exact Format.isFinite_zero Format.bfloat16_regular

/-- The `bfloat16` zero pattern is not a NaN. -/
theorem BF16_zero_notNaN : (FloatV.ofPat (0 : Nat) : BF16).isNaN = false := by
  rw [FloatV.isNaN_ofPat (by decide)]
  decide

/-- The host literal `0.0` of `other=0.0` is the `float32` zero pattern. -/
theorem zf32_eq : zf32 = (FloatV.ofPat (0 : Nat) : F32) := by
  unfold zf32
  rw [FloatV.litF32, FloatV.isZero_ofPat (by decide), Format.isZero_zero]
  rfl

/-- `zf32` has value `0`. -/
theorem zf32_toRat : zf32.toRat = 0 := by rw [zf32_eq]; exact F32_zero_toRat

/-- `zf32` is finite. -/
theorem zf32_finite : zf32.isFinite = true := by rw [zf32_eq]; exact F32_zero_finite

/-- Converting the `float32` zero to `bfloat16` gives the `bfloat16` zero pattern. -/
theorem cvt_f32_bf16_zero :
    (FloatV.cvt (FloatV.ofPat (0 : Nat) : F32) : BF16) = (FloatV.ofPat (0 : Nat) : BF16) := by
  have h : Format.binary32.cvt Format.bfloat16 .rn false 0 = 0 := by
    rw [Format.cvt_of_finite (g := Format.bfloat16) (rnd := .rn) (sat := false)
      (by rw [Format.isFinite_zero Format.binary32_regular])]
    rw [show Format.binary32.toRat 0 = 0 from F32_zero_toRat,
      Format.roundRat_zero]
    simp [Format.setSign, Format.sign, Format.mag]
  unfold FloatV.cvt
  rw [FloatV.pat_ofPat (by decide), h]

/-- Every `float32` converts to `bfloat16` on the platform. -/
theorem exists_conv_f32_bf16 (P : Platform) (x : F32) : ∃ r, ConvRel P .f32 .bf16 x r := by
  refine ⟨FloatV.cvt x, ?_⟩
  show FloatRel P .binary32 .bfloat16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩

/-- A platform conversion `float32 → bfloat16` of `zf32` is the `bfloat16` zero pattern. -/
theorem convRel_zf32 {P : Platform} {r : BF16} (h : ConvRel P .f32 .bf16 zf32 r) :
    r = (FloatV.ofPat (0 : Nat) : BF16) := by
  rw [zf32_eq] at h
  simp only [ConvRel, ofFloatRel, FloatRel] at h
  simp only [show Format.binary32 ≠ Format.bfloat16 from by decide] at h
  rcases h with ⟨-, -, h | h⟩ | ⟨-, hg, -⟩ | ⟨hf, -⟩
  · rw [cvt_f32_bf16_zero] at h
    exact h.symm
  · rw [cvt_f32_bf16_zero, BF16_zero_notNaN] at h
    exact absurd h.1 (by simp)
  · rw [Format.F8] at hg
    rcases hg with h' | h' <;> exact absurd h' (by decide)
  · rw [Format.F8] at hf
    rcases hf with h' | h' <;> exact absurd h' (by decide)

end Claims.SsdChunkScan
