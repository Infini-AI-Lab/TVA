import Proof.Glue

/-!
# One lane's sparse gate (lines 83–84)

`gate - cutoff` (the IEEE `-`), its `bfloat16` rounding widened back, and `tl.where(… < 0.0, 0.0,
…)`: the lane's sparse gate `sgSel sg0` is a finite value in `[0, 2^41]` within
`(2^-8 + 2^-20)·max(g − c, 0) + 2^-131` of `max(g − c, 0)`, and `+0` when `g − c ≤ −2^-131`. The
exact-zero lemmas say that a product or conversion of a zero is a zero.
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 800000

/-- The lane of `tl.where(sg0 < 0.0, 0.0, sg0)` (line 84). -/
def sgSel (x : F32) : F32 := if FloatV.lt x (c32 0) then c32 0 else x

/-- A product with a zero left operand is a zero (exactly, finite). -/
theorem mul_zero_left {x y r : F32} (hx : x.isFinite = true) (hx0 : x.toRat = 0)
    (hy : y.isFinite = true) (h : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧ r.toRat = 0 := by
  have hf := Format.binary32_regular
  have he : FloatV.mul x y = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign x.pat != Format.binary32.sign y.pat) 0) := by
    unfold FloatV.mul
    rw [Format.mul_of_finite .rn hx hy]
    have : Format.binary32.toRat x.pat = 0 := hx0
    rw [this, zero_mul]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn) (sat := false)
    (zs := Format.binary32.sign x.pat != Format.binary32.sign y.pat) hf
  have hfin : (FloatV.mul x y).isFinite = true := by
    rw [he]; unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [he, FloatV.toRat_ofPat h1, h3]

/-- A product with a zero right operand is a zero (exactly, finite). -/
theorem mul_zero_right {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hy0 : y.toRat = 0) (h : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧ r.toRat = 0 := by
  have hf := Format.binary32_regular
  have he : FloatV.mul x y = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign x.pat != Format.binary32.sign y.pat) 0) := by
    unfold FloatV.mul
    rw [Format.mul_of_finite .rn hx hy]
    have : Format.binary32.toRat y.pat = 0 := hy0
    rw [this, mul_zero]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn) (sat := false)
    (zs := Format.binary32.sign x.pat != Format.binary32.sign y.pat) hf
  have hfin : (FloatV.mul x y).isFinite = true := by
    rw [he]; unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [he, FloatV.toRat_ofPat h1, h3]

/-- The `bfloat16` rounding of a zero is a zero. -/
theorem cvt16_zero {x : F32} {r : BF16} (hx : x.isFinite = true) (hx0 : x.toRat = 0)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) : r.isFinite = true ∧ r.toRat = 0 := by
  obtain ⟨hfin, herr⟩ := cvt16_lane hx (by rw [FloatV.toReal_eq, hx0]; norm_num) h
  refine ⟨hfin, ?_⟩
  have hf := Format.bfloat16_regular
  have he : (FloatV.cvtWith .rn x : BF16) = FloatV.ofPat (Format.bfloat16.roundRat .rn false
      (Format.binary32.sign x.pat) 0) := by
    unfold FloatV.cvtWith
    rw [Format.cvt_of_finite _ .rn false hx]
    have : Format.binary32.toRat x.pat = 0 := hx0
    rw [this]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.bfloat16) (rnd := .rn) (sat := false)
    (zs := Format.binary32.sign x.pat) hf
  have hfin' : (FloatV.cvtWith .rn x : BF16).isFinite = true := by
    rw [he]; unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [h.eq (FloatV.isFinite_iff.mp hfin').1, he, FloatV.toRat_ofPat h1, h3]

/-- `+0` of `float32` has value `0`. -/
theorem c32_zero_val : (c32 0).isFinite = true ∧ (c32 0).toRat = 0 ∧ (c32 0).isNaN = false := by
  rw [c_zero]
  exact ⟨FloatV.isFinite_ofPat_zero Format.binary32_regular, by decide +kernel,
    FloatV.isNaN_ofPat_zero Format.binary32_regular⟩

/-- `x < 0.0` on a finite `float32` is `x < 0`. -/
theorem lt_zero_iff {x : F32} (hx : x.isFinite = true) :
    FloatV.lt x (c32 0) = true ↔ x.toReal < 0 := by
  rw [FloatV.lt_iff (FloatV.isFinite_iff.mp hx).1 c32_zero_val.2.2, c32_zero_val.2.1,
    FloatV.toReal_eq]
  exact_mod_cast Iff.rfl

/-- The real arithmetic of the sparse gate: `x` the `float32` difference of `Δ`, `y` its
`bfloat16` rounding, and the selection `max`-like. -/
theorem sg_core {Δ x y s : ℝ} (hΔ : |Δ| ≤ 2 ^ 40 + 2 ^ 32)
    (hx : |x - Δ| ≤ u32 * |Δ| + eta32) (hy : |y - x| ≤ ub * |x| + etab)
    (hs : (y < 0 ∧ s = 0) ∨ (0 ≤ y ∧ s = y)) :
    0 ≤ s ∧ s ≤ 2 ^ 41 ∧ |s - max Δ 0| ≤ (2 ^ (-8 : ℤ) + 2 ^ (-20 : ℤ)) * max Δ 0 + 2 ^ (-131 : ℤ) ∧
      (Δ ≤ -2 ^ (-131 : ℤ) → y < 0) := by
  unfold u32 eta32 ub etab at *
  have hu : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-131 : ℤ) / 500000 := by norm_num
  have hb : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have he8 : (2 : ℝ) ^ (-134 : ℤ) = 2 ^ (-131 : ℤ) / 8 := by norm_num
  have h20 : (2 : ℝ) ^ (-20 : ℤ) = 1 / 1048576 := by norm_num
  have hp : (0 : ℝ) < 2 ^ (-131 : ℤ) := by positivity
  have hp40 : (2 : ℝ) ^ (-131 : ℤ) ≤ 1 := by norm_num
  rw [hu] at hx; rw [hb, he8] at hy; rw [hb, h20]
  -- bounds on x and y around Δ
  have hx1 : |x - Δ| ≤ |Δ| / 16777216 + 2 ^ (-131 : ℤ) / 500000 := by
    have := hx; linarith
  have hxa : |x| ≤ |Δ| + |Δ| / 16777216 + 2 ^ (-131 : ℤ) / 500000 := by
    have := abs_sub_abs_le_abs_sub x Δ; linarith
  have hy1 : |y - Δ| ≤ |Δ| * (1 / 256 + 1 / 16777216 + 1 / (256 * 16777216)) +
      2 ^ (-131 : ℤ) * (1 / 8 + 1 / 500000 + 1 / 128000000) := by
    have h3 : |y - Δ| ≤ |y - x| + |x - Δ| := by
      have := abs_sub_le y x Δ; linarith
    have h4 : 1 / 256 * |x| ≤ 1 / 256 * (|Δ| + |Δ| / 16777216 + 2 ^ (-131 : ℤ) / 500000) := by
      gcongr
    nlinarith
  have hΔ0 := abs_nonneg Δ
  refine ⟨?_, ?_, ?_, ?_⟩
  · rcases hs with ⟨-, hs0⟩ | ⟨h, hsy⟩
    · rw [hs0]
    · rw [hsy]; exact h
  · rcases hs with ⟨-, hs0⟩ | ⟨-, hsy⟩
    · rw [hs0]; norm_num
    · rw [hsy]
      have := abs_sub_abs_le_abs_sub y Δ
      have : |y| ≤ 2 ^ 41 := by nlinarith
      exact (le_abs_self y).trans this
  · rcases le_or_gt 0 Δ with hΔp | hΔn
    · rw [max_eq_left hΔp]
      rw [abs_of_nonneg hΔp] at hy1
      rcases hs with ⟨hyn, hs0⟩ | ⟨-, hsy⟩
      · -- y < 0 ≤ Δ: Δ is tiny
        rw [abs_le] at hy1
        rw [hs0, zero_sub, abs_neg, abs_of_nonneg hΔp]
        nlinarith
      · rw [hsy]; nlinarith [abs_le.mp hy1]
    · rw [max_eq_right hΔn.le]
      rw [abs_of_neg hΔn] at hy1
      rcases hs with ⟨-, hs0⟩ | ⟨hy0, hsy⟩
      · rw [hs0]; simp
      · rw [hsy, sub_zero, abs_of_nonneg hy0]
        rw [abs_le] at hy1
        nlinarith
  · intro hle
    have hΔn : Δ < 0 := by linarith
    rw [abs_of_neg hΔn] at hy1
    have := (abs_le.mp hy1).2
    generalize (2 : ℝ) ^ (-131 : ℤ) = P at hle this hp
    linarith

/-- **The lane's sparse gate** (lines 83–84): from the finite gate `g` and cutoff `c`, the lane
`sgSel sg0` is finite, in `[0, 2^41]`, within `(2^-8 + 2^-20)·max(g − c, 0) + 2^-131` of
`max(g − c, 0)`, and `+0` when `g − c ≤ −2^-131`. -/
theorem lane_s {g c s12 sg0 : F32} {sg16 : BF16} (hg : g.isFinite = true) (hc : c.isFinite = true)
    (hgb : |g.toReal| ≤ 2 ^ 32) (hcb : |c.toReal| ≤ 2 ^ 40)
    (h1 : FloatV.NaNEq (FloatV.sub g c) s12) (h2 : FloatV.NaNEq (FloatV.cvtWith .rn s12) sg16)
    (h3 : FloatV.NaNEq (FloatV.cvtWith .rn sg16) sg0) :
    (sgSel sg0).isFinite = true ∧ 0 ≤ (sgSel sg0).toReal ∧ (sgSel sg0).toReal ≤ 2 ^ 41 ∧
      |(sgSel sg0).toReal - max (g.toReal - c.toReal) 0| ≤
        (2 ^ (-8 : ℤ) + 2 ^ (-20 : ℤ)) * max (g.toReal - c.toReal) 0 + 2 ^ (-131 : ℤ) ∧
      (g.toReal - c.toReal ≤ -2 ^ (-131 : ℤ) → sgSel sg0 = c32 0) := by
  have hΔ : |g.toReal - c.toReal| ≤ 2 ^ 40 + 2 ^ 32 := by
    have := abs_sub g.toReal c.toReal; linarith
  obtain ⟨hf1, he1⟩ := sub_lane hg hc (hΔ.trans (by norm_num)) h1
  have hx1 : |s12.toReal| ≤ 2 ^ 120 := by
    have := abs_sub_abs_le_abs_sub s12.toReal (g.toReal - c.toReal)
    unfold u32 eta32 at he1
    have h4 : (2 : ℝ) ^ (-24 : ℤ) * |g.toReal - c.toReal| ≤ 2 ^ (-24 : ℤ) * (2 ^ 40 + 2 ^ 32) := by
      gcongr
    have h5 : (2 : ℝ) ^ (-24 : ℤ) * (2 ^ 40 + 2 ^ 32) + 2 ^ (-150 : ℤ) ≤ 2 ^ 20 := by norm_num
    linarith
  obtain ⟨hf2, he2⟩ := cvt16_lane hf1 hx1 h2
  obtain ⟨hf3, he3⟩ := cvt32_lane hf2 h3
  have hsel : ((sg0.toReal < 0 ∧ (sgSel sg0).toReal = 0) ∨
      (0 ≤ sg0.toReal ∧ (sgSel sg0).toReal = sg0.toReal)) := by
    unfold sgSel
    by_cases hl : FloatV.lt sg0 (c32 0) = true
    · left
      rw [if_pos hl]
      exact ⟨(lt_zero_iff hf3).mp hl, by rw [FloatV.toReal_eq, c32_zero_val.2.1]; norm_num⟩
    · right
      rw [if_neg hl]
      exact ⟨not_lt.mp fun h => hl ((lt_zero_iff hf3).mpr h), rfl⟩
  rw [← he3] at he2
  obtain ⟨r1, r2, r3, r4⟩ := sg_core hΔ he1 he2 hsel
  refine ⟨?_, r1, r2, r3, fun hle => ?_⟩
  · unfold sgSel; split_ifs
    · exact c32_zero_val.1
    · exact hf3
  · unfold sgSel
    rw [if_pos ((lt_zero_iff hf3).mpr (r4 hle))]

end Claims.GeluAndMulSparse
