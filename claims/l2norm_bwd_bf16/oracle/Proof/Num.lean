import TVA.Proof

/-!
# `l2norm_bwd_bf16`: the error of one element, in the reals

The kernel computes, per element, `t1 = fl(g·r)`, the row sum `s ≈ S = Σ g_k·a_k`,
`t2 = fl(s·a)`, `t3 = fl(t2·r)` and the fusable `dx = t1 − t3`, then (for a `bfloat16` `dx`) one
more rounding. This file has the one-operation facts in the reals (`mulR`, `subcR`, `cvt16R`), the
room the conditions give (`room`), and the composition (`dx_err32`, `dx_err16`): with
`M = |r|·(|g| + A·|a|)` and `F = (1 + |r|)·(1 + |a|)`,
`|dx − (g·r − S·a·r)| ≤ 2^-18·M + 2^-138·F`, and after the `bfloat16` rounding
`≤ 2^-8·|ref| + 2^-17·M + 2^-133·F`.

Constants are written as rationals (`1/2^24`, `1/2^150`) so that `linarith` reads them as numbers.
-/

namespace Claims.L2NormBwd

open TVA

/-! ## One operation, in the reals -/

/-- `u` and `η` of `float32` in the form used here. -/
theorem u32R : ((Format.binary32.u : ℚ) : ℝ) = 1 / 2 ^ 24 := by
  rw [Format.binary32_u]; push_cast; norm_num [zpow_neg]

theorem eta32R : ((Format.binary32.eta : ℚ) : ℝ) = 1 / 2 ^ 150 := by
  rw [Format.binary32_eta]; push_cast; norm_num [zpow_neg]

theorem max32R : ((Format.binary32.maxVal : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]; push_cast; norm_num

/-- **An IEEE `float32` product**, up to NaN payload, of finite operands with `|x·y| ≤ 2^127`: finite
and within `2^-24·|x·y| + 2^-150`. -/
theorem mulR {x y r : F32} (h : FloatV.NaNEq (FloatV.mul x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hm : |x.toReal * y.toReal| ≤ 2 ^ 127) :
    r.isFinite = true ∧
      |r.toReal - x.toReal * y.toReal| ≤ 1 / 2 ^ 24 * |x.toReal * y.toReal| + 1 / 2 ^ 150 := by
  have hm' : |x.toReal * y.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [max32R]; refine hm.trans ?_; norm_num
  obtain ⟨hf, he⟩ := FloatV.mul_err_real Format.binary32_regular hx hy hm'
  have heq : r = FloatV.mul x y := h.eq (FloatV.isFinite_iff.mp hf).1
  rw [heq]
  refine ⟨hf, ?_⟩
  rw [u32R, eta32R] at he
  linarith

/-- **A fusable `float32` subtraction** `x − y` (`AddRel` of `x` and `−y`) of finite operands with
`|x| + |y| ≤ 2^126`: finite and within `3·2^-24·(|x| + |y|) + 4·2^-150`. -/
theorem subcR {fusion : Bool} {x y r : F32} (h : AddRel fusion x (FloatV.neg y) r)
    (hx : x.isFinite = true) (hy : y.isFinite = true) (hm : |x.toReal| + |y.toReal| ≤ 2 ^ 126) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal - y.toReal)| ≤ 3 / 2 ^ 24 * (|x.toReal| + |y.toReal|) + 4 / 2 ^ 150 := by
  have hny : (FloatV.neg y).isFinite = true := by rw [FloatV.isFinite_neg]; exact hy
  have hnr : (FloatV.neg y).toRat = -y.toRat := FloatV.toRat_neg y
  have hmq : |x.toRat| + |(FloatV.neg y).toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [hnr, abs_neg]
    have hR : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by
      push_cast; simpa only [FloatV.toReal] using hm
    have hq : |x.toRat| + |y.toRat| ≤ 2 ^ 126 := by exact_mod_cast hR
    refine hq.trans ?_
    rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num [zpow_neg]
  obtain ⟨hf, he⟩ := FloatV.AddRel.err Format.binary32_regular (by decide)
    (by rw [Format.binary32_u]; norm_num [zpow_neg]) h hx hny hmq
  refine ⟨hf, ?_⟩
  rw [hnr, abs_neg] at he
  have hR := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at hR
  rw [u32R, eta32R] at hR
  simp only [FloatV.toReal]
  have e : (r.toRat : ℝ) - ((x.toRat : ℝ) - (y.toRat : ℝ)) = (r.toRat : ℝ) - ((x.toRat : ℝ) + -(y.toRat : ℝ)) := by ring
  rw [e]
  linarith

/-- **The rounding of a `float32` to `bfloat16`**, up to NaN payload, of a finite value with
`|x| ≤ 2^127`: finite and within `2^-8·|x| + 2^-134`. -/
theorem cvt16R {x : F32} {c : BF16} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) c)
    (hx : x.isFinite = true) (hm : |x.toReal| ≤ 2 ^ 127) :
    c.isFinite = true ∧ |c.toReal - x.toReal| ≤ 1 / 2 ^ 8 * |x.toReal| + 1 / 2 ^ 134 := by
  have hm' : |x.toReal| ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Format.bfloat16_maxVal]; push_cast; refine hm.trans ?_; norm_num
  obtain ⟨hf, he⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hx hm'
  have heq : c = (FloatV.cvt x : BF16) := by
    rw [FloatV.cvtWith_rn] at h; exact h.eq (FloatV.isFinite_iff.mp hf).1
  rw [heq]
  refine ⟨hf, ?_⟩
  rw [Format.bfloat16_u, Format.bfloat16_eta] at he
  push_cast at he
  norm_num [zpow_neg] at he ⊢
  linarith

/-! ## Room -/

/-- **The room the conditions give.** From `(1 + |r|)·(1 + |a|)·(1 + |g| + A) ≤ 2^124` with `A ≥ 0`,
every product the kernel forms is at most `2^124`. -/
theorem room {g a r A : ℝ} (hA : 0 ≤ A) (h : (1 + |r|) * (1 + |a|) * (1 + |g| + A) ≤ 2 ^ 124) :
    A ≤ 2 ^ 124 ∧ |a| ≤ 2 ^ 124 ∧ |r| ≤ 2 ^ 124 ∧ A * |a| ≤ 2 ^ 124 ∧ |g| * |r| ≤ 2 ^ 124 ∧
      A * |a| * |r| ≤ 2 ^ 124 ∧ |a| * |r| ≤ 2 ^ 124 ∧ |g| * |r| + A * |a| * |r| ≤ 2 ^ 124 := by
  have hr := abs_nonneg r
  have ha := abs_nonneg a
  have hg := abs_nonneg g
  have e : (1 + |r|) * (1 + |a|) * (1 + |g| + A) =
      1 + |r| + |a| + |a| * |r| + |g| + A + |g| * |r| + A * |r| + |a| * |g| + A * |a| +
        |a| * |g| * |r| + A * |a| * |r| := by ring
  rw [e] at h
  have h1 : 0 ≤ |a| * |r| := mul_nonneg ha hr
  have h2 : 0 ≤ |g| * |r| := mul_nonneg hg hr
  have h3 : 0 ≤ A * |a| * |r| := mul_nonneg (mul_nonneg hA ha) hr
  have h4 : 0 ≤ A * |r| := mul_nonneg hA hr
  have h5 : 0 ≤ |a| * |g| := mul_nonneg ha hg
  have h6 : 0 ≤ A * |a| := mul_nonneg hA ha
  have h7 : 0 ≤ |a| * |g| * |r| := mul_nonneg h5 hr
  refine ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith,
    by linarith⟩

/-! ## The composition -/

/-- **The error of one `float32` element.** With the row sum `s` within `29u·A + 2600η` of
`S`, `|S| ≤ A`, and the four roundings of the kernel (`t1 = fl(g·r)`, `t2 = fl(s·a)`,
`t3 = fl(t2·r)`, the fusable `dx = t1 − t3`), `dx` is within `2^-18·M + 2^-138·F` of
`g·r − S·a·r`. -/
theorem dx_err32 {g a r S A s t1 t2 t3 dx : ℝ} (hA : 0 ≤ A) (hS : |S| ≤ A)
    (hs : |s - S| ≤ 29 / 2 ^ 24 * A + 2600 / 2 ^ 150)
    (ht1 : |t1 - g * r| ≤ 1 / 2 ^ 24 * |g * r| + 1 / 2 ^ 150)
    (ht2 : |t2 - s * a| ≤ 1 / 2 ^ 24 * |s * a| + 1 / 2 ^ 150)
    (ht3 : |t3 - t2 * r| ≤ 1 / 2 ^ 24 * |t2 * r| + 1 / 2 ^ 150)
    (hdx : |dx - (t1 - t3)| ≤ 3 / 2 ^ 24 * (|t1| + |t3|) + 4 / 2 ^ 150) :
    |dx - (g * r - S * a * r)| ≤
      1 / 2 ^ 18 * (|r| * (|g| + A * |a|)) + 1 / 2 ^ 138 * ((1 + |r|) * (1 + |a|)) := by
  have hr := abs_nonneg r
  have ha := abs_nonneg a
  have hg := abs_nonneg g
  -- the atoms
  set X := |g| * |r| with hX
  set Y := A * |a| * |r| with hY
  set Z := |a| * |r| with hZ
  have hX0 : 0 ≤ X := mul_nonneg hg hr
  have hY0 : 0 ≤ Y := mul_nonneg (mul_nonneg hA ha) hr
  have hZ0 : 0 ≤ Z := mul_nonneg ha hr
  have hgr : |g * r| = X := abs_mul g r
  -- `t1`
  have ht1a : |t1| ≤ X + (1 / 2 ^ 24 * X + 1 / 2 ^ 150) := by
    have := abs_sub_abs_le_abs_sub t1 (g * r); rw [hgr] at this ht1; linarith
  rw [hgr] at ht1
  -- `s`
  have hsa : |s| ≤ A + (29 / 2 ^ 24 * A + 2600 / 2 ^ 150) := by
    have := abs_sub_abs_le_abs_sub s S; linarith
  -- `s·a` and `s·a·r`
  have hsa_a : |s * a| ≤ (1 + 29 / 2 ^ 24) * (A * |a|) + 2600 / 2 ^ 150 * |a| := by
    rw [abs_mul]
    have := mul_le_mul_of_nonneg_right hsa ha; nlinarith
  have hsa_ar : |s * a| * |r| ≤ (1 + 29 / 2 ^ 24) * Y + 2600 / 2 ^ 150 * Z := by
    have := mul_le_mul_of_nonneg_right hsa_a hr
    have e1 : (1 + 29 / 2 ^ 24) * (A * |a|) * |r| = (1 + 29 / 2 ^ 24) * Y := by rw [hY]; ring
    have e2 : 2600 / 2 ^ 150 * |a| * |r| = 2600 / 2 ^ 150 * Z := by rw [hZ]; ring
    nlinarith
  -- `|s − S|·|a|·|r|`
  have hsS : |s - S| * |a| * |r| ≤ 29 / 2 ^ 24 * Y + 2600 / 2 ^ 150 * Z := by
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hs ha) hr
    have e : (29 / 2 ^ 24 * A + 2600 / 2 ^ 150) * |a| * |r| = 29 / 2 ^ 24 * Y + 2600 / 2 ^ 150 * Z := by
      rw [hY, hZ]; ring
    linarith
  -- `t2·r`
  have ht2r : |t2 * r| ≤ (1 + 1 / 2 ^ 24) * ((1 + 29 / 2 ^ 24) * Y + 2600 / 2 ^ 150 * Z) +
      1 / 2 ^ 150 * |r| := by
    rw [abs_mul]
    have h1 : |t2| ≤ |s * a| + (1 / 2 ^ 24 * |s * a| + 1 / 2 ^ 150) := by
      have := abs_sub_abs_le_abs_sub t2 (s * a); linarith
    have h2 := mul_le_mul_of_nonneg_right h1 hr
    have e : (|s * a| + (1 / 2 ^ 24 * |s * a| + 1 / 2 ^ 150)) * |r| =
        (1 + 1 / 2 ^ 24) * (|s * a| * |r|) + 1 / 2 ^ 150 * |r| := by ring
    rw [e] at h2
    have h3 := mul_le_mul_of_nonneg_left hsa_ar (by norm_num : (0 : ℝ) ≤ 1 + 1 / 2 ^ 24)
    linarith
  -- `t3 − S·a·r`
  have ht3e : |t3 - S * a * r| ≤ 1 / 2 ^ 24 * |t2 * r| + 1 / 2 ^ 150 +
      (1 / 2 ^ 24 * ((1 + 29 / 2 ^ 24) * Y + 2600 / 2 ^ 150 * Z) + 1 / 2 ^ 150 * |r| +
        (29 / 2 ^ 24 * Y + 2600 / 2 ^ 150 * Z)) := by
    have e : t3 - S * a * r = (t3 - t2 * r) + ((t2 - s * a) * r + (s - S) * a * r) := by ring
    rw [e]
    have hb : |(t2 - s * a) * r + (s - S) * a * r| ≤ |t2 - s * a| * |r| + |s - S| * |a| * |r| := by
      refine (abs_add_le _ _).trans (le_of_eq ?_)
      rw [abs_mul, abs_mul, abs_mul]
    have hc := mul_le_mul_of_nonneg_right ht2 hr
    have e2 : (1 / 2 ^ 24 * |s * a| + 1 / 2 ^ 150) * |r| = 1 / 2 ^ 24 * (|s * a| * |r|) + 1 / 2 ^ 150 * |r| := by ring
    rw [e2] at hc
    have hd := mul_le_mul_of_nonneg_left hsa_ar (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 24)
    have := abs_add_le (t3 - t2 * r) ((t2 - s * a) * r + (s - S) * a * r)
    linarith
  -- `|t3|`
  have hSar : |S * a * r| ≤ Y := by
    rw [abs_mul, abs_mul, hY]
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hS ha) hr
  have ht3a : |t3| ≤ Y + |t3 - S * a * r| := by
    have := abs_sub_abs_le_abs_sub t3 (S * a * r); linarith
  -- the total
  have htot : |dx - (g * r - S * a * r)| ≤ |dx - (t1 - t3)| + |t1 - g * r| + |t3 - S * a * r| := by
    have e : dx - (g * r - S * a * r) = (dx - (t1 - t3)) + (t1 - g * r) - (t3 - S * a * r) := by ring
    rw [e]
    refine (abs_sub _ _).trans ?_
    linarith [abs_add_le (dx - (t1 - t3)) (t1 - g * r)]
  have hM : |r| * (|g| + A * |a|) = X + Y := by rw [hX, hY]; ring
  have hF : (1 + |r|) * (1 + |a|) = 1 + |r| + |a| + Z := by rw [hZ]; ring
  rw [hM, hF]
  linarith

/-- **The error of one `bfloat16` element**: the `float32` bound `2^-18·M + 2^-138·F` and one
rounding to `bfloat16` give `2^-8·|ref| + 2^-17·M + 2^-133·F`. -/
theorem dx_err16 {ref dx c M F : ℝ} (hM : 0 ≤ M) (hF : 1 ≤ F)
    (hE : |dx - ref| ≤ 1 / 2 ^ 18 * M + 1 / 2 ^ 138 * F)
    (hc : |c - dx| ≤ 1 / 2 ^ 8 * |dx| + 1 / 2 ^ 134) :
    |c - ref| ≤ 1 / 2 ^ 8 * |ref| + 1 / 2 ^ 17 * M + 1 / 2 ^ 133 * F := by
  have h1 : |dx| ≤ |ref| + |dx - ref| := by
    have := abs_sub_abs_le_abs_sub dx ref; linarith
  have h2 : |c - ref| ≤ |c - dx| + |dx - ref| := by
    have e : c - ref = (c - dx) + (dx - ref) := by ring
    rw [e]; exact abs_add_le _ _
  linarith

/-! ## Room for each operation -/

/-- `|s| ≤ (1 + 29u)·A + 2600η`. -/
theorem s_abs {S A s : ℝ} (hS : |S| ≤ A) (hs : |s - S| ≤ 29 / 2 ^ 24 * A + 2600 / 2 ^ 150) :
    |s| ≤ A + (29 / 2 ^ 24 * A + 2600 / 2 ^ 150) := by
  have := abs_sub_abs_le_abs_sub s S; linarith

/-- `|s·a|·|r| ≤ (1 + 29u)·A·|a|·|r| + 2600η·|a|·|r|`. -/
theorem sa_r {S A s a r : ℝ} (hS : |S| ≤ A) (hs : |s - S| ≤ 29 / 2 ^ 24 * A + 2600 / 2 ^ 150) :
    |s * a| * |r| ≤ (1 + 29 / 2 ^ 24) * (A * |a| * |r|) + 2600 / 2 ^ 150 * (|a| * |r|) := by
  have h := s_abs hS hs
  rw [abs_mul]
  have h2 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right h (abs_nonneg a)) (abs_nonneg r)
  have e : (A + (29 / 2 ^ 24 * A + 2600 / 2 ^ 150)) * |a| * |r| =
      (1 + 29 / 2 ^ 24) * (A * |a| * |r|) + 2600 / 2 ^ 150 * (|a| * |r|) := by ring
  linarith

/-- Room for `t2 = fl(s·a)`: `|s·a| ≤ 2^126`. -/
theorem room_sa {S A s a : ℝ} (hS : |S| ≤ A) (hs : |s - S| ≤ 29 / 2 ^ 24 * A + 2600 / 2 ^ 150)
    (hAa : A * |a| ≤ 2 ^ 124) (ha : |a| ≤ 2 ^ 124) : |s * a| ≤ 2 ^ 126 := by
  have h := sa_r (a := a) (r := 1) hS hs
  simp only [abs_one, mul_one] at h
  linarith

/-- Room for `t3 = fl(t2·r)`: `|t2·r| ≤ 2^126`. -/
theorem room_t2r {S A s a r t2 : ℝ} (hS : |S| ≤ A)
    (hs : |s - S| ≤ 29 / 2 ^ 24 * A + 2600 / 2 ^ 150)
    (ht2 : |t2 - s * a| ≤ 1 / 2 ^ 24 * |s * a| + 1 / 2 ^ 150)
    (hY : A * |a| * |r| ≤ 2 ^ 124) (hZ : |a| * |r| ≤ 2 ^ 124) (hr : |r| ≤ 2 ^ 124) :
    |t2 * r| ≤ (1 + 1 / 2 ^ 24) * ((1 + 29 / 2 ^ 24) * (A * |a| * |r|) +
      2600 / 2 ^ 150 * (|a| * |r|)) + 1 / 2 ^ 150 * |r| ∧ |t2 * r| ≤ 2 ^ 126 := by
  have h := sa_r (a := a) (r := r) hS hs
  have h1 : |t2| ≤ (1 + 1 / 2 ^ 24) * |s * a| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub t2 (s * a); linarith
  have h2 := mul_le_mul_of_nonneg_right h1 (abs_nonneg r)
  have e : ((1 + 1 / 2 ^ 24) * |s * a| + 1 / 2 ^ 150) * |r| =
      (1 + 1 / 2 ^ 24) * (|s * a| * |r|) + 1 / 2 ^ 150 * |r| := by ring
  have h3 := mul_le_mul_of_nonneg_left h (by norm_num : (0 : ℝ) ≤ 1 + 1 / 2 ^ 24)
  rw [abs_mul]
  refine ⟨by linarith, by linarith⟩

/-- Room for the fusable `t1 − t3`: `|t1| + |t3| ≤ 2^126`. -/
theorem room_sub {g r t1 t2 t3 B X : ℝ} (hX : |g| * |r| ≤ X)
    (ht1 : |t1 - g * r| ≤ 1 / 2 ^ 24 * |g * r| + 1 / 2 ^ 150)
    (ht3 : |t3 - t2 * r| ≤ 1 / 2 ^ 24 * |t2 * r| + 1 / 2 ^ 150)
    (hB : |t2 * r| ≤ B) (hXB : X + B ≤ 2 ^ 125) : |t1| + |t3| ≤ 2 ^ 126 := by
  have h1 := abs_sub_abs_le_abs_sub t1 (g * r)
  have h3 := abs_sub_abs_le_abs_sub t3 (t2 * r)
  rw [abs_mul] at h1 ht1
  have hX0 : 0 ≤ |g| * |r| := mul_nonneg (abs_nonneg _) (abs_nonneg _)
  have hB0 : 0 ≤ |t2 * r| := abs_nonneg _
  linarith

/-- Room for the conversion to `bfloat16`: `|dx| ≤ 2^127`. -/
theorem room_dx {g a r S A dx : ℝ} (hA : 0 ≤ A) (hS : |S| ≤ A)
    (hE : |dx - (g * r - S * a * r)| ≤
      1 / 2 ^ 18 * (|r| * (|g| + A * |a|)) + 1 / 2 ^ 138 * ((1 + |r|) * (1 + |a|)))
    (hP : (1 + |r|) * (1 + |a|) * (1 + |g| + A) ≤ 2 ^ 124) : |dx| ≤ 2 ^ 127 := by
  obtain ⟨-, -, -, -, -, -, -, hXY⟩ := room hA hP
  have hF : (1 + |r|) * (1 + |a|) ≤ 2 ^ 124 := by
    have h1 : 0 ≤ (1 + |r|) * (1 + |a|) := by positivity
    have h2 : (1 + |r|) * (1 + |a|) * 1 ≤ (1 + |r|) * (1 + |a|) * (1 + |g| + A) :=
      mul_le_mul_of_nonneg_left (by have := abs_nonneg g; linarith) h1
    linarith
  have href : |g * r - S * a * r| ≤ |g| * |r| + A * |a| * |r| := by
    refine (abs_sub _ _).trans ?_
    rw [abs_mul, abs_mul, abs_mul]
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hS (abs_nonneg a)) (abs_nonneg r)
    linarith
  have hM : |r| * (|g| + A * |a|) = |g| * |r| + A * |a| * |r| := by ring
  rw [hM] at hE
  have := abs_sub_abs_le_abs_sub dx (g * r - S * a * r)
  have h0 : 0 ≤ |g| * |r| + A * |a| * |r| := by positivity
  linarith

end Claims.L2NormBwd
