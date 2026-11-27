import Proof.Chain
import Proof.Ops
import Proof.SigReal

/-!
# `layer_norm_gated_fwd_bf16`: one output element

Lane `(t, j)` computes `h1 = fl(x·rstd)`, `h2 = fl(h1·w)`, then `h3 = fl(h2·g)` and
`h4 = fl(h3·s)` (swish) or `h3 = fl(h2·s)` (sigmoid), `s` the kernel's `sigmoid(g)`, and stores
`bf16(h4)` (or `bf16(h3)`). With `rstd` within `2^-12` (relative) of `R`, `s` within
`2^-15·σ(g) + 2^-128` of `σ(g)` (`Gate.gate_err`), `|x·R| ≤ 23` and `|w|, |g| ≤ 2^56`, the stored
value is finite and within `2^-7·|yRef| + 2^-126·(1 + |Z|)·(1 + |w|)·gMag` of
`yRef = Z·w·gate` (`Z = x·R`): `elem_swish`, `elem_sigmoid`. Each step is a `Chain` step, its
constants weakened to round ones (`track_mono`).
-/

namespace Claims.LayerNormGatedFwd

open TVA

set_option maxHeartbeats 1000000 in
/-- The first two products, `h1 = fl(x·r)` and `h2 = fl(h1·w)`: finite, `h2` within
`(2^-12 + 2^-22)·|Z·w| + 2^-148·(1 + |w|)` of `Z·w`, and `|h2| ≤ 2^62`. -/
theorem elem_xw {xf wf r h1 h2 : F32} {x w R : ℝ} (hR : 0 < R)
    (hr : |r.toReal - R| ≤ (2 : ℝ) ^ (-12 : ℤ) * R) (hrfin : r.isFinite = true)
    (hZ : |x * R| ≤ 23) (hxf : xf.isFinite = true) (hxR : xf.toReal = x)
    (hwf : wf.isFinite = true) (hwR : wf.toReal = w) (hw56 : |w| ≤ (2 : ℝ) ^ (56 : ℤ))
    (hm1 : FloatV.NaNEq (FloatV.mul xf r) h1) (hm2 : FloatV.NaNEq (FloatV.mul h1 wf) h2) :
    h2.isFinite = true ∧
      |h2.toReal - x * R * w| ≤
        ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)) * |x * R * w| +
          (2 : ℝ) ^ (-148 : ℤ) * (1 * (1 + |w|)) ∧
      |h2.toReal| ≤ (2 : ℝ) ^ (62 : ℤ) := by
  -- `x·r` against `Z = x·R`
  have hxr : |x * r.toReal - x * R| ≤ (2 : ℝ) ^ (-12 : ℤ) * |x * R| + 0 * 1 := by
    rw [show x * r.toReal - x * R = x * (r.toReal - R) by ring, abs_mul, abs_mul, abs_of_pos hR]
    have := mul_le_mul_of_nonneg_left hr (abs_nonneg x)
    linarith
  have hxr_abs : |x * r.toReal| ≤ 24 := by
    have h := abs_le_of_track hxr
    have e : (1 + (2 : ℝ) ^ (-12 : ℤ)) * 23 + 0 * 1 ≤ 24 := by norm_num
    have := mul_le_mul_of_nonneg_left hZ (by positivity : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-12 : ℤ))
    linarith
  obtain ⟨h1fin, h1err⟩ := mul32 hm1 hxf hrfin (by rw [hxR]; exact hxr_abs.trans (by norm_num))
  rw [hxR] at h1err
  have h1t := round_step (by positivity) (by positivity) le_rfl le_rfl h1err hxr
  have h1t' : |h1.toReal - x * R| ≤ ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-23 : ℤ)) * |x * R| +
      (2 : ℝ) ^ (-149 : ℤ) * 1 :=
    track_mono h1t (by norm_num) (by norm_num) (by norm_num)
  have h1abs : |h1.toReal| ≤ 25 := by
    have h := abs_le_of_track h1t'
    have := mul_le_mul_of_nonneg_left hZ
      (by positivity : (0 : ℝ) ≤ 1 + ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-23 : ℤ)))
    have e : (1 + ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-23 : ℤ))) * 23 + (2 : ℝ) ^ (-149 : ℤ) * 1 ≤ 25 := by
      norm_num
    linarith
  -- `h2 = fl(h1·w)`
  have hroom2 : |h1.toReal * wf.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) := by
    rw [hwR]
    refine (abs_mul_le_of h1abs hw56).trans ?_
    norm_num
  obtain ⟨h2fin, h2err⟩ := mul32 hm2 h1fin hwf hroom2
  rw [hwR] at h2err
  have h2t := exact_step (by positivity) (by positivity) le_rfl (by positivity) h1t' h2err
  have h2t' := track_mono h2t (ε' := (2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-22 : ℤ))
    (κ' := (2 : ℝ) ^ (-148 : ℤ)) (by norm_num) (by norm_num) (by positivity)
  refine ⟨h2fin, h2t', ?_⟩
  have h := abs_le_of_track h2t'
  have hZw : |x * R * w| ≤ 23 * (2 : ℝ) ^ (56 : ℤ) := abs_mul_le_of hZ hw56
  have hm : 1 * (1 + |w|) ≤ 1 + (2 : ℝ) ^ (56 : ℤ) := by linarith
  have e1 := mul_le_mul_of_nonneg_left hZw
    (by positivity : (0 : ℝ) ≤ 1 + ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)))
  have e2 := mul_le_mul_of_nonneg_left hm (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-148 : ℤ))
  have e : (1 + ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-22 : ℤ))) * (23 * (2 : ℝ) ^ (56 : ℤ)) +
      (2 : ℝ) ^ (-148 : ℤ) * (1 + (2 : ℝ) ^ (56 : ℤ)) ≤ (2 : ℝ) ^ (62 : ℤ) := by norm_num
  linarith

set_option maxHeartbeats 2000000 in
/-- **An output element, swish**: `bf16(fl(fl(h2·g)·s))` against `Z·w·(g·σ(g))`. -/
theorem elem_swish {xf wf gf r h1 h2 h3 sv h4 : F32} {c : BF16} {x w g R : ℝ} (hR : 0 < R)
    (hr : |r.toReal - R| ≤ (2 : ℝ) ^ (-12 : ℤ) * R) (hrfin : r.isFinite = true)
    (hZ : |x * R| ≤ 23) (hxf : xf.isFinite = true) (hxR : xf.toReal = x)
    (hwf : wf.isFinite = true) (hwR : wf.toReal = w) (hw56 : |w| ≤ (2 : ℝ) ^ (56 : ℤ))
    (hgf : gf.isFinite = true) (hgR : gf.toReal = g) (hg56 : |g| ≤ (2 : ℝ) ^ (56 : ℤ))
    (hsfin : sv.isFinite = true)
    (hs : |sv.toReal - sig g| ≤ (2 : ℝ) ^ (-15 : ℤ) * sig g + (2 : ℝ) ^ (-128 : ℤ))
    (hm1 : FloatV.NaNEq (FloatV.mul xf r) h1) (hm2 : FloatV.NaNEq (FloatV.mul h1 wf) h2)
    (hm3 : FloatV.NaNEq (FloatV.mul h2 gf) h3) (hm4 : FloatV.NaNEq (FloatV.mul h3 sv) h4)
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn h4) c) :
    c.isFinite = true ∧
      |c.toReal - x * R * w * (g * sig g)| ≤
        (2 : ℝ) ^ (-7 : ℤ) * |x * R * w * (g * sig g)| +
          (2 : ℝ) ^ (-126 : ℤ) * (1 + |x * R|) * (1 + |w|) * (1 + |g|) := by
  obtain ⟨h2fin, h2t, h2abs⟩ := elem_xw hR hr hrfin hZ hxf hxR hwf hwR hw56 hm1 hm2
  -- `h3 = fl(h2·g)`
  have hroom3 : |h2.toReal * gf.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) := by
    rw [hgR]; refine (abs_mul_le_of h2abs hg56).trans ?_; norm_num
  obtain ⟨h3fin, h3err⟩ := mul32 hm3 h2fin hgf hroom3
  rw [hgR] at h3err
  have h3t := exact_step (by positivity) (by positivity) (by linarith [abs_nonneg w])
    (by positivity) h2t h3err
  have h3t' := track_mono h3t (ε' := (2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-21 : ℤ))
    (κ' := (2 : ℝ) ^ (-147 : ℤ)) (by norm_num) (by norm_num) (by positivity)
  have hm3 : 1 ≤ 1 * (1 + |w|) * (1 + |g|) := by nlinarith [abs_nonneg w, abs_nonneg g]
  have hZwg : |x * R * w * g| ≤ 23 * (2 : ℝ) ^ (56 : ℤ) * (2 : ℝ) ^ (56 : ℤ) :=
    abs_mul_le_of (abs_mul_le_of hZ hw56) hg56
  have h3abs : |h3.toReal| ≤ (2 : ℝ) ^ (118 : ℤ) := by
    have h := abs_le_of_track h3t'
    have hmb : 1 * (1 + |w|) * (1 + |g|) ≤ (1 + (2 : ℝ) ^ (56 : ℤ)) * (1 + (2 : ℝ) ^ (56 : ℤ)) := by
      rw [one_mul]
      exact mul_le_mul (by linarith) (by linarith) (by positivity) (by positivity)
    have e1 := mul_le_mul_of_nonneg_left hZwg
      (by positivity : (0 : ℝ) ≤ 1 + ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-21 : ℤ)))
    have e2 := mul_le_mul_of_nonneg_left hmb (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-147 : ℤ))
    have e : (1 + ((2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-21 : ℤ))) *
        (23 * (2 : ℝ) ^ (56 : ℤ) * (2 : ℝ) ^ (56 : ℤ)) +
        (2 : ℝ) ^ (-147 : ℤ) * ((1 + (2 : ℝ) ^ (56 : ℤ)) * (1 + (2 : ℝ) ^ (56 : ℤ))) ≤
        (2 : ℝ) ^ (118 : ℤ) := by norm_num
    linarith
  -- `h4 = fl(h3·s)`
  have hσ0 := sig_pos g
  have hσ1 := sig_le_one g
  have hsabs : |sv.toReal| ≤ 2 := by
    have h := abs_add_le (sig g) (sv.toReal - sig g)
    rw [show sig g + (sv.toReal - sig g) = sv.toReal by ring, abs_of_pos hσ0] at h
    have e : (2 : ℝ) ^ (-15 : ℤ) * sig g ≤ (2 : ℝ) ^ (-15 : ℤ) :=
      mul_le_of_le_one_right (by positivity) hσ1
    have e2 : (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-128 : ℤ) ≤ 1 := by norm_num
    linarith
  have hroom4 : |h3.toReal * sv.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) :=
    (abs_mul_le_of h3abs hsabs).trans (by norm_num)
  obtain ⟨h4fin, h4err⟩ := mul32 hm4 h3fin hsfin hroom4
  have hK : (1 : ℝ) ≤ 1 + |x * R| := by linarith [abs_nonneg (x * R)]
  have hAK : |x * R * w * g| ≤ (1 + |x * R|) * (1 * (1 + |w|) * (1 + |g|)) := by
    rw [abs_mul, abs_mul]
    have h1 : |x * R| ≤ 1 + |x * R| := by linarith
    have h2 : |w| ≤ 1 + |w| := by linarith
    have h3 : |g| ≤ 1 + |g| := by linarith
    have := mul_le_mul (mul_le_mul h1 h2 (abs_nonneg _) (by positivity)) h3 (abs_nonneg _)
      (by positivity)
    nlinarith [abs_nonneg (x * R), abs_nonneg w, abs_nonneg g]
  have h4t := approx_step (by positivity) (by positivity) hm3 hK (by positivity) (by positivity)
    (by positivity) (by positivity) hσ0.le hσ1 hAK h3t' hs h4err
  have h4t' := track_mono h4t (ε' := (2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-20 : ℤ))
    (κ' := (2 : ℝ) ^ (-127 : ℤ)) (by norm_num) (by norm_num) (by positivity)
  -- the `bfloat16` rounding
  have h4abs : |h4.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) := by
    have h := abs_add_le (h3.toReal * sv.toReal) (h4.toReal - h3.toReal * sv.toReal)
    rw [show h3.toReal * sv.toReal + (h4.toReal - h3.toReal * sv.toReal) = h4.toReal by ring] at h
    have hp := (abs_mul_le_of h3abs hsabs)
    have e : (2 : ℝ) ^ (118 : ℤ) * 2 + ((2 : ℝ) ^ (-24 : ℤ) * ((2 : ℝ) ^ (118 : ℤ) * 2) +
        (2 : ℝ) ^ (-150 : ℤ)) ≤ (2 : ℝ) ^ (127 : ℤ) := by norm_num
    have := mul_le_mul_of_nonneg_left hp (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-24 : ℤ))
    linarith
  obtain ⟨hcfin, hcerr⟩ := cvt16 hc h4fin h4abs
  have hct := round_step (by positivity) (by positivity) (by nlinarith) (by positivity) hcerr h4t'
  refine ⟨hcfin, ?_⟩
  have hY : x * R * w * (g * sig g) = x * R * w * g * sig g := by ring
  have hM : (2 : ℝ) ^ (-126 : ℤ) * ((1 + |x * R|) * (1 * (1 + |w|) * (1 + |g|))) =
      (2 : ℝ) ^ (-126 : ℤ) * (1 + |x * R|) * (1 + |w|) * (1 + |g|) := by ring
  rw [hY, ← hM]
  exact track_mono hct (by norm_num) (by norm_num) (by positivity)

set_option maxHeartbeats 2000000 in
/-- **An output element, sigmoid**: `bf16(fl(h2·s))` against `Z·w·σ(g)`. -/
theorem elem_sigmoid {xf wf r h1 h2 sv h3 : F32} {c : BF16} {x w g R : ℝ} (hR : 0 < R)
    (hr : |r.toReal - R| ≤ (2 : ℝ) ^ (-12 : ℤ) * R) (hrfin : r.isFinite = true)
    (hZ : |x * R| ≤ 23) (hxf : xf.isFinite = true) (hxR : xf.toReal = x)
    (hwf : wf.isFinite = true) (hwR : wf.toReal = w) (hw56 : |w| ≤ (2 : ℝ) ^ (56 : ℤ))
    (hsfin : sv.isFinite = true)
    (hs : |sv.toReal - sig g| ≤ (2 : ℝ) ^ (-15 : ℤ) * sig g + (2 : ℝ) ^ (-128 : ℤ))
    (hm1 : FloatV.NaNEq (FloatV.mul xf r) h1) (hm2 : FloatV.NaNEq (FloatV.mul h1 wf) h2)
    (hm3 : FloatV.NaNEq (FloatV.mul h2 sv) h3)
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn h3) c) :
    c.isFinite = true ∧
      |c.toReal - x * R * w * sig g| ≤
        (2 : ℝ) ^ (-7 : ℤ) * |x * R * w * sig g| +
          (2 : ℝ) ^ (-126 : ℤ) * (1 + |x * R|) * (1 + |w|) * 1 := by
  obtain ⟨h2fin, h2t, h2abs⟩ := elem_xw hR hr hrfin hZ hxf hxR hwf hwR hw56 hm1 hm2
  have hσ0 := sig_pos g
  have hσ1 := sig_le_one g
  have hsabs : |sv.toReal| ≤ 2 := by
    have h := abs_add_le (sig g) (sv.toReal - sig g)
    rw [show sig g + (sv.toReal - sig g) = sv.toReal by ring, abs_of_pos hσ0] at h
    have e : (2 : ℝ) ^ (-15 : ℤ) * sig g ≤ (2 : ℝ) ^ (-15 : ℤ) :=
      mul_le_of_le_one_right (by positivity) hσ1
    have e2 : (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-128 : ℤ) ≤ 1 := by norm_num
    linarith
  have hroom3 : |h2.toReal * sv.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) :=
    (abs_mul_le_of h2abs hsabs).trans (by norm_num)
  obtain ⟨h3fin, h3err⟩ := mul32 hm3 h2fin hsfin hroom3
  have hm : 1 ≤ 1 * (1 + |w|) := by linarith [abs_nonneg w]
  have hK : (1 : ℝ) ≤ 1 + |x * R| := by linarith [abs_nonneg (x * R)]
  have hAK : |x * R * w| ≤ (1 + |x * R|) * (1 * (1 + |w|)) := by
    rw [abs_mul]
    have := mul_le_mul (by linarith : |x * R| ≤ 1 + |x * R|) (by linarith : |w| ≤ 1 + |w|)
      (abs_nonneg _) (by positivity)
    linarith
  have h3t := approx_step (by positivity) (by positivity) hm hK (by positivity) (by positivity)
    (by positivity) (by positivity) hσ0.le hσ1 hAK h2t hs h3err
  have h3t' := track_mono h3t (ε' := (2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-20 : ℤ))
    (κ' := (2 : ℝ) ^ (-127 : ℤ)) (by norm_num) (by norm_num) (by positivity)
  have h3abs : |h3.toReal| ≤ (2 : ℝ) ^ (127 : ℤ) := by
    have h := abs_add_le (h2.toReal * sv.toReal) (h3.toReal - h2.toReal * sv.toReal)
    rw [show h2.toReal * sv.toReal + (h3.toReal - h2.toReal * sv.toReal) = h3.toReal by ring] at h
    have hp := (abs_mul_le_of h2abs hsabs)
    have e : (2 : ℝ) ^ (62 : ℤ) * 2 + ((2 : ℝ) ^ (-24 : ℤ) * ((2 : ℝ) ^ (62 : ℤ) * 2) +
        (2 : ℝ) ^ (-150 : ℤ)) ≤ (2 : ℝ) ^ (127 : ℤ) := by norm_num
    have := mul_le_mul_of_nonneg_left hp (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-24 : ℤ))
    linarith
  obtain ⟨hcfin, hcerr⟩ := cvt16 hc h3fin h3abs
  have hct := round_step (by positivity) (by positivity) (by nlinarith) (by positivity) hcerr h3t'
  refine ⟨hcfin, ?_⟩
  have hM : (2 : ℝ) ^ (-126 : ℤ) * ((1 + |x * R|) * (1 * (1 + |w|))) =
      (2 : ℝ) ^ (-126 : ℤ) * (1 + |x * R|) * (1 + |w|) * 1 := by ring
  rw [← hM]
  exact track_mono hct (by norm_num) (by norm_num) (by positivity)

end Claims.LayerNormGatedFwd
