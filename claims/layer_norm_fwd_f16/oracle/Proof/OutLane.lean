import Proof.Walk2
import Proof.ZStage

/-!
# One output element (lines 92–95)

`out_lane`: the roundings of `(x − mean) * rstd * w + b` on one lane, from the lane's loads to the
`float16` result, against the tolerance `yTol`.
-/

namespace Claims.LayerNormFwdF16

open TVA

set_option maxHeartbeats 1600000 in
/-- **One output element**: given the stored mean and `rstd` (`RstdOK`), the lane's value `x` of
the row, the fusable subtraction `dd`, the product `xh`, the exact conversions of `w` and `b`, the
`mul_add` `y` and its rounding `y16` to `float16`, `y16` is finite and within `yTol` of
`zRef · w + b`. -/
theorem out_lane (r : RowP) {j : ℕ} (hj : j < r.N) {fusion : Bool} {mf rs x dd xh w32 b32 y : F32}
    {y16 : F16} {w b : F16} (hmf : mf.isFinite = true)
    (hmf' : |mf.toReal - r.mu| ≤ 11 / 20 * r.Delta) (hrs : RstdOK r rs)
    (hx : x.isFinite = true) (hxv : x.toReal = r.x j) (hdd : AddRel fusion x (FloatV.neg mf) dd)
    (hxh : FloatV.NaNEq (FloatV.mul dd rs) xh) (hw : w.isFinite = true) (hb : b.isFinite = true)
    (hw32 : FloatV.NaNEq (FloatV.cvtWith .rn w) w32) (hb32 : FloatV.NaNEq (FloatV.cvtWith .rn b) b32)
    (hy : FloatV.NaNEq (FloatV.add (FloatV.mul xh w32) b32) y ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma xh w32 b32) y))
    (hy16 : FloatV.NaNEq (FloatV.cvtWith .rn y) y16)
    (hroom : |w.toReal| * (|r.zRef j| + r.zTol j) + |b.toReal| ≤ 2 ^ 15) :
    y16.isFinite = true ∧ |y16.toReal - (r.zRef j * w.toReal + b.toReal)| ≤
      2 ^ (-9 : ℤ) * (|w.toReal| * |r.zRef j| + |b.toReal|) + 2 * |w.toReal| * r.zTol j +
        2 ^ (-24 : ℤ) := by
  obtain ⟨hrsfin, hrspos, -, hsmall, hbig, hrs65⟩ := hrs
  have hmb : |mf.toReal| ≤ 65505 := by
    have h1 := r.mu_le
    have h2 := r.Delta_le
    have h3 : |mf.toReal| ≤ |r.mu| + |mf.toReal - r.mu| := by
      have := abs_add_le r.mu (mf.toReal - r.mu)
      simpa using this
    linarith
  have hxb : |x.toReal| ≤ 65504 := by rw [hxv]; exact r.x_le j hj
  -- the subtraction
  obtain ⟨hddfin, hlane⟩ := lane_sub hx hmf (hxb.trans (by norm_num)) (hmb.trans (by norm_num)) hdd
  rw [hxv] at hlane
  have hddb : |dd.toReal| ≤ 2 ^ 18 := by
    have hx' := r.x_le j hj
    unfold LaneErr u32 eta32 at hlane
    have h1 : |dd.toReal| ≤ |r.x j - mf.toReal| + (2 ^ (-24 : ℤ) * |r.x j - mf.toReal| +
        2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) * (|r.x j| + |mf.toReal|) + 3 * 2 ^ (-150 : ℤ)) := by
      have := abs_sub_abs_le_abs_sub dd.toReal (r.x j - mf.toReal)
      linarith
    have h2 : |r.x j - mf.toReal| ≤ 131009 := by
      have := abs_sub (r.x j) mf.toReal
      linarith
    have h3 : 2 ^ (-24 : ℤ) * |r.x j - mf.toReal| + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) *
        (|r.x j| + |mf.toReal|) + 3 * 2 ^ (-150 : ℤ) ≤ 1 := by
      have : (2 : ℝ) ^ (-24 : ℤ) * |r.x j - mf.toReal| ≤ 2 ^ (-24 : ℤ) * 131009 := by gcongr
      have h4 : 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) * (|r.x j| + |mf.toReal|) ≤
          2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) * (65504 + 65505) := by gcongr
      have h5 : (2 : ℝ) ^ (-24 : ℤ) * 131009 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) *
          (65504 + 65505) + 3 * 2 ^ (-150 : ℤ) ≤ 1 := by norm_num
      linarith
    have h6 : (131009 : ℝ) + 1 ≤ 2 ^ 18 := by norm_num
    linarith
  -- the product with `rstd`
  have hprod : |dd.toReal * rs.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul, abs_of_pos hrspos]
    calc |dd.toReal| * rs.toReal ≤ 2 ^ 18 * 2 ^ 65 :=
          mul_le_mul hddb hrs65 hrspos.le (by positivity)
      _ ≤ 2 ^ 120 := by norm_num
  obtain ⟨hxhfin, hz⟩ := mul_lane hddfin hrsfin hprod hxh
  -- the standardised value
  have hzb : |xh.toReal - r.zRef j| ≤ r.zTol j := by
    by_cases hc : r.cond ≤ 1 / 2
    · exact r.z_small j hj hc mf.toReal rs.toReal dd.toReal xh.toReal hmf' (hsmall hc) hlane hz
    · exact r.z_big j hj hc mf.toReal rs.toReal dd.toReal xh.toReal hmf' hrspos (hbig hc) hlane hz
  -- the weights, exactly
  have hw32e : w32.isFinite = true ∧ w32.toReal = w.toReal := by
    obtain ⟨h1, -, h3⟩ := FloatV.cvt_f16_f32_exact hw
    rw [FloatV.cvtWith_rn] at hw32
    rw [hw32.eq_of_isFinite h1]; exact ⟨h1, h3⟩
  have hb32e : b32.isFinite = true ∧ b32.toReal = b.toReal := by
    obtain ⟨h1, -, h3⟩ := FloatV.cvt_f16_f32_exact hb
    rw [FloatV.cvtWith_rn] at hb32
    rw [hb32.eq_of_isFinite h1]; exact ⟨h1, h3⟩
  have hzabs : |xh.toReal| ≤ |r.zRef j| + r.zTol j := by
    have := abs_sub_abs_le_abs_sub xh.toReal (r.zRef j)
    linarith
  have hzw : |xh.toReal * w32.toReal| + |b32.toReal| ≤ 2 ^ 15 := by
    rw [hw32e.2, hb32e.2, abs_mul]
    have : |xh.toReal| * |w.toReal| ≤ (|r.zRef j| + r.zTol j) * |w.toReal| :=
      mul_le_mul_of_nonneg_right hzabs (abs_nonneg _)
    nlinarith
  obtain ⟨hyfin, hy32⟩ := muladd_lane hxhfin hw32e.1 hb32e.1 (hzw.trans (by norm_num)) hy
  rw [hw32e.2, hb32e.2] at hy32
  have hroom32 := y32_room (r.zRef j) (r.zTol j) w.toReal b.toReal xh.toReal y.toReal hzb hroom hy32
  rw [FloatV.cvtWith_rn] at hy16
  obtain ⟨hy16fin, hy16err⟩ := cvt_f32_f16_lane hyfin hroom32 hy16
  exact ⟨hy16fin, y_stage (r.zRef j) (r.zTol j) w.toReal b.toReal xh.toReal y.toReal y16.toReal hzb
    hroom hy32 hy16err⟩

end Claims.LayerNormFwdF16
