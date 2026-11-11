import Proof.YBound

/-! The two products of the output chain: `bhat` (the deviation times the reciprocal standard
deviation) and `by0` (that times the weight).  Their magnitudes are what makes the bias addition
finite. -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section

set_option maxHeartbeats 1000000

/-- **The output chain's magnitudes**: the two products are finite, `bhat` is below `2^111` and
`by0` is below `2^127 - 2^110`, which leaves the bias addition inside the `float32` range. -/
theorem y_hat_by0_bd {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    {bx0 bmean bx brstd bw bhat by0 : F32}
    (hval : bx0.toReal = sR d init t j) (hbx0fin : bx0.isFinite = true)
    (hbdd : |sR d init t j| ≤ 2 ^ 57)
    (hmeanfin : bmean.isFinite = true)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hrstdfin : brstd.isFinite = true)
    (hrstd : |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t)
    (hdev : AddRel true bx0 (FloatV.neg bmean) bx) (hbxfin : bx.isFinite = true)
    (hmul1 : FloatV.NaNEq (FloatV.mul bx brstd) bhat)
    (hbwfin : bw.isFinite = true) (hbwbd : |bw.toReal| ≤ 2 ^ 16)
    (hmul2 : FloatV.NaNEq (FloatV.mul bhat bw) by0) :
    bhat.isFinite = true ∧ by0.isFinite = true ∧
      |bhat.toReal| ≤ 2 ^ 111 ∧ |by0.toReal| ≤ 2 ^ 127 - 2 ^ 110 := by
  obtain ⟨hbhatmulfin, hbmag⟩ :=
    y_bhat_bd hd hj hval hbx0fin hbdd hmeanfin hmean hrstdfin hrstd hdev hbxfin
  -- the first product is exactly the rounded deviation times the stored reciprocal std
  have hbhat_eq : bhat = FloatV.mul bx brstd :=
    hmul1.eq (FloatV.isFinite_iff.mp hbhatmulfin).1
  have hbhatfin : bhat.isFinite = true := by rw [hbhat_eq]; exact hbhatmulfin
  have hu32 : (0 : ℝ) ≤ u32 := le_of_lt u32_pos
  have heta0 : (0 : ℝ) ≤ eta32 := le_of_lt eta32_pos
  -- the exact product of the two operands is inside the `float32` range
  have hmaxR1 : |(bx.toRat : ℝ) * (brstd.toRat : ℝ)| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have h1 : |(bx.toRat : ℝ) * (brstd.toRat : ℝ)| = |bx.toReal| * |brstd.toReal| := by
      show |(bx.toRat : ℝ) * (brstd.toRat : ℝ)| = |(bx.toRat : ℝ)| * |(brstd.toRat : ℝ)|
      rw [abs_mul]
    rw [h1]
    have h2 : (2 : ℝ) ^ (110 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [Format.binary32_maxVal]
      norm_num
    have h3 : |bx.toReal| * |brstd.toReal| ≤ (2 : ℝ) ^ (110 : ℤ) := by simpa using hbmag
    linarith [h3, h2]
  have hmax1 : |bx.toRat * brstd.toRat| ≤ binary32.maxVal := by
    have hcast : ((|bx.toRat * brstd.toRat| : ℚ) : ℝ) ≤ ((binary32.maxVal : ℚ) : ℝ) := by
      push_cast
      exact hmaxR1
    exact_mod_cast hcast
  obtain ⟨-, herr1⟩ :=
    FloatV.mul_err (f := binary32) binary32_regular hbxfin hrstdfin hmax1
  have hcast1 : |((FloatV.mul bx brstd).toRat : ℝ) - ((bx.toRat * brstd.toRat : ℚ) : ℝ)|
      ≤ (((|bx.toRat * brstd.toRat| * Format.binary32.u + Format.binary32.eta : ℚ)) : ℝ) := by
    exact_mod_cast herr1
  push_cast at hcast1
  rw [rstd_tol_u_cast, rstd_tol_eta_cast] at hcast1
  have herr1R : |(FloatV.mul bx brstd).toReal - bx.toReal * brstd.toReal|
      ≤ |bx.toReal * brstd.toReal| * u32 + eta32 := by
    simpa [FloatV.toReal] using hcast1
  have hbmagR : |bx.toReal * brstd.toReal| ≤ (2 : ℝ) ^ (110 : ℤ) := by
    rw [abs_mul]
    simpa using hbmag
  have htri1 : |(FloatV.mul bx brstd).toReal|
      ≤ |bx.toReal * brstd.toReal|
        + |(FloatV.mul bx brstd).toReal - bx.toReal * brstd.toReal| := by
    have h := abs_add_le (bx.toReal * brstd.toReal)
      ((FloatV.mul bx brstd).toReal - bx.toReal * brstd.toReal)
    have he : bx.toReal * brstd.toReal
        + ((FloatV.mul bx brstd).toReal - bx.toReal * brstd.toReal)
        = (FloatV.mul bx brstd).toReal := by ring
    rw [he] at h
    exact h
  -- the tight magnitude bound on the first product
  have hbhat_tight : |bhat.toReal| ≤ (2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32 := by
    rw [hbhat_eq]
    have h1 : |bx.toReal * brstd.toReal| * u32 ≤ (2 : ℝ) ^ (110 : ℤ) * u32 :=
      mul_le_mul_of_nonneg_right hbmagR hu32
    linarith [htri1, herr1R, hbmagR, h1]
  -- the rational form of that bound, for the second multiplication's range side condition
  have hbhat_rat : |bhat.toRat|
      ≤ (2 : ℚ) ^ 110 * (1 + Format.binary32.u) + Format.binary32.eta := by
    have h1 : (((|bhat.toRat| : ℚ)) : ℝ)
        ≤ (((2 : ℚ) ^ 110 * (1 + Format.binary32.u) + Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      rw [rstd_tol_u_cast, rstd_tol_eta_cast]
      simpa [FloatV.toReal] using hbhat_tight
    exact_mod_cast h1
  have hbw_rat : |bw.toRat| ≤ (2 : ℚ) ^ 16 := by
    have h1 : |(bw.toRat : ℝ)| ≤ (2 : ℝ) ^ 16 := by simpa [FloatV.toReal] using hbwbd
    exact_mod_cast h1
  have hmax2 : |bhat.toRat * bw.toRat| ≤ binary32.maxVal := by
    rw [abs_mul]
    have hle : |bhat.toRat| * |bw.toRat|
        ≤ ((2 : ℚ) ^ 110 * (1 + Format.binary32.u) + Format.binary32.eta)
          * (2 : ℚ) ^ 16 :=
      mul_le_mul hbhat_rat hbw_rat (abs_nonneg _) (by
        rw [Format.binary32_u, Format.binary32_eta]
        positivity)
    have h2 : ((2 : ℚ) ^ 110 * (1 + Format.binary32.u) + Format.binary32.eta)
          * (2 : ℚ) ^ 16 ≤ (Format.binary32.maxVal : ℚ) := by
      rw [Format.binary32_maxVal, Format.binary32_u, Format.binary32_eta]
      norm_num
    linarith
  obtain ⟨hby0mulfin, herr2⟩ :=
    FloatV.mul_err (f := binary32) binary32_regular hbhatfin hbwfin hmax2
  have hby0_eq : by0 = FloatV.mul bhat bw :=
    hmul2.eq (FloatV.isFinite_iff.mp hby0mulfin).1
  have hby0fin : by0.isFinite = true := by rw [hby0_eq]; exact hby0mulfin
  have hcast2 : |((FloatV.mul bhat bw).toRat : ℝ) - ((bhat.toRat * bw.toRat : ℚ) : ℝ)|
      ≤ (((|bhat.toRat * bw.toRat| * Format.binary32.u + Format.binary32.eta : ℚ)) : ℝ) := by
    exact_mod_cast herr2
  push_cast at hcast2
  rw [rstd_tol_u_cast, rstd_tol_eta_cast] at hcast2
  have herr2R : |(FloatV.mul bhat bw).toReal - bhat.toReal * bw.toReal|
      ≤ |bhat.toReal * bw.toReal| * u32 + eta32 := by
    simpa [FloatV.toReal] using hcast2
  have hbwR : |bw.toReal| ≤ (2 : ℝ) ^ 16 := hbwbd
  have htri2 : |(FloatV.mul bhat bw).toReal|
      ≤ |bhat.toReal * bw.toReal|
        + |(FloatV.mul bhat bw).toReal - bhat.toReal * bw.toReal| := by
    have h := abs_add_le (bhat.toReal * bw.toReal)
      ((FloatV.mul bhat bw).toReal - bhat.toReal * bw.toReal)
    have he : bhat.toReal * bw.toReal
        + ((FloatV.mul bhat bw).toReal - bhat.toReal * bw.toReal)
        = (FloatV.mul bhat bw).toReal := by ring
    rw [he] at h
    exact h
  have hprod2 : |bhat.toReal * bw.toReal|
      ≤ ((2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32) * (2 : ℝ) ^ 16 := by
    rw [abs_mul]
    exact mul_le_mul hbhat_tight hbwR (abs_nonneg _) (by positivity)
  -- the second product stays below `2^127 - 2^110`
  have hby0_bound : |by0.toReal| ≤ (2 : ℝ) ^ 127 - (2 : ℝ) ^ 110 := by
    rw [hby0_eq]
    have hupper : |(FloatV.mul bhat bw).toReal|
        ≤ (((2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32) * (2 : ℝ) ^ 16)
          + (((2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32) * (2 : ℝ) ^ 16) * u32 + eta32 := by
      have hmul_u : |bhat.toReal * bw.toReal| * u32
          ≤ (((2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32) * (2 : ℝ) ^ 16) * u32 :=
        mul_le_mul_of_nonneg_right hprod2 hu32
      linarith [htri2, herr2R, hprod2, hmul_u]
    have hnum : (((2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32) * (2 : ℝ) ^ 16)
          + (((2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32) * (2 : ℝ) ^ 16) * u32 + eta32
          ≤ (2 : ℝ) ^ 127 - (2 : ℝ) ^ 110 := by
      unfold u32 eta32
      norm_num
    linarith [hupper, hnum]
  have hbhat_le : |bhat.toReal| ≤ (2 : ℝ) ^ 111 := by
    have hnum : (2 : ℝ) ^ (110 : ℤ) * (1 + u32) + eta32 ≤ (2 : ℝ) ^ 111 := by
      unfold u32 eta32
      norm_num
    linarith [hbhat_tight, hnum]
  exact ⟨hbhatfin, hby0fin, hbhat_le, hby0_bound⟩

end

end Claims.LayerNormFwd
