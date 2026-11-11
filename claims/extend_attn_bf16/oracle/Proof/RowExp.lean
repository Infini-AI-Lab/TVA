import Proof.RowNmax

/-!
# The exponentials of a row's trip

`re_scale = tl.exp (e_max − n_e_max)` and `p = tl.exp (qk − n_e_max)`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The `float32` difference of two finite values `a ≤ b` is at most `2^-100`. -/
theorem sub_le_small {a b d : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : a.toReal ≤ b.toReal) (hgap : b.toReal - a.toReal ≤ 2 ^ 100)
    (hd : FloatV.NaNEq (FloatV.sub a b) d) : d.isFinite = true ∧ d.toReal ≤ 1 / 2 ^ 100 := by
  have hxa : |a.toReal - b.toReal| ≤ 2 ^ 100 := by
    rw [abs_of_nonpos (by linarith)]; linarith
  obtain ⟨hf, he⟩ := f32_sub ha hb hxa hd
  refine ⟨hf, ?_⟩
  have h1 := (abs_le.mp he).2
  rw [abs_of_nonpos (by linarith : a.toReal - b.toReal ≤ 0)] at h1
  have h2 : (1 : ℝ) / 2 ^ 24 * (-(a.toReal - b.toReal)) ≤ 1 / 2 ^ 24 * 2 ^ 100 := by
    apply mul_le_mul_of_nonneg_left _ (by norm_num); linarith
  have : (1 : ℝ) / 2 ^ 150 ≤ 1 / 2 ^ 100 := by norm_num
  nlinarith

/-- **`tl.exp (a − b)`** for finite `a ≤ b` (gap at most `2^100`): finite and of magnitude at most
`1 + 2^-20`. -/
theorem exp_sub_le {P : Platform} (hP : Device P) {a b d r : F32} (ha : a.isFinite = true)
    (hb : b.isFinite = true) (hab : a.toReal ≤ b.toReal) (hgap : b.toReal - a.toReal ≤ 2 ^ 100)
    (hd : FloatV.NaNEq (FloatV.sub a b) d) (hr : P.math .exp .binary32 d r) :
    r.isFinite = true ∧ |r.toReal| ≤ 1 + 1 / 2 ^ 20 := by
  obtain ⟨hdf, hd0⟩ := sub_le_small ha hb hab hgap hd
  exact exp_le_one hP hdf hd0 hr

/-- **A masked lane's `p`**: `tl.exp (−∞ − b)` is `0`. -/
theorem p_masked {P : Platform} (hP : Device P) {b d r : F32} (hb : b.isFinite = true)
    (hd : FloatV.NaNEq (FloatV.sub FloatV.negInf b) d) (hr : P.math .exp .binary32 d r) :
    r.isFinite = true ∧ r.toReal = 0 := exp_negInf hP hb hd hr

end

end Claims.ExtendAttn
