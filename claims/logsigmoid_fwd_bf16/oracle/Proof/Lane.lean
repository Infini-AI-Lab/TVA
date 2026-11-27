import Claim
import TVA.Proof
import Proof.Real

/-!
# One element: from the operations' results to `yTol`

`lane` takes the results the `Safe` rules hand the walk for one enabled lane (each `tl.*` result
up to NaN payload, each platform result in the platform's relation) and shows that the stored
`bfloat16` is finite and within `yTol` of `lsRef v / t`. Each step turns its operation's rule into
a real error bound (the device facts for `tl.exp`, `tl.log`, `/`; one rounding for `+`, `-`, the
conversions; exactness for `abs`, `neg`, `minimum` and the widening), and `Proof.Real` does the
arithmetic.
-/

set_option maxHeartbeats 1000000

namespace Claims.LogsigmoidFwd

open TVA

/-! ## Constants and exact operations -/

theorem zeroF32_isNaN : zeroF32.isNaN = false := by decide

theorem zeroF32_toReal : zeroF32.toReal = 0 := by
  have : zeroF32.toRat = 0 := by decide +kernel
  simp [FloatV.toReal, this]

theorem oneF32_isFinite : oneF32.isFinite = true := by decide +kernel

theorem oneF32_toReal : oneF32.toReal = 1 := by
  have : oneF32.toRat = 1 := by decide +kernel
  simp [FloatV.toReal, this]

theorem maxVal_bf16_ge : (2 : ℝ) ^ 127 ≤ (Format.bfloat16.maxVal : ℝ) := by
  have : (2 : ℚ) ^ 127 ≤ Format.bfloat16.maxVal := by decide +kernel
  exact_mod_cast this

theorem maxVal_bf16_lt : (Format.bfloat16.maxVal : ℝ) + 3 ≤ (Format.binary32.maxVal : ℝ) := by
  have : Format.bfloat16.maxVal + 3 ≤ Format.binary32.maxVal := by decide +kernel
  exact_mod_cast this

theorem maxVal32_ge_two : (2 : ℝ) ^ 127 ≤ (Format.binary32.maxVal : ℝ) := by
  have : (2 : ℚ) ^ 127 ≤ Format.binary32.maxVal := by decide +kernel
  exact_mod_cast this

/-- `tl.abs` of a finite value is exact. -/
theorem abs_exact {x r : F32} (hx : x.isFinite = true) (h : FloatV.NaNEq (FloatV.abs x) r) :
    r.isFinite = true ∧ r.toReal = |x.toReal| := by
  have hpat : Format.binary32.absPat x.pat = Format.binary32.setSign false x.pat := by
    simp [Format.absPat, Format.setSign]
  have hlt := Format.setSign_lt (f := Format.binary32) false x.pat
  have hfin : (FloatV.abs x).isFinite = true := by
    unfold FloatV.abs
    rw [hpat, FloatV.isFinite_ofPat hlt, Format.isFinite_setSign]
    exact hx
  have hval : (FloatV.abs x).toRat = |x.toRat| := by
    unfold FloatV.abs
    rw [hpat, FloatV.toRat_ofPat hlt, Format.toRat_setSign]
    simp only [Bool.false_eq_true, ↓reduceIte]
    exact (Format.abs_toRat x.pat).symm
  obtain ⟨hf, -, hv⟩ := FloatV.NaNEq.exact_of_isFinite h hfin
  refine ⟨hf, ?_⟩
  rw [hv, FloatV.toReal_eq, hval, Rat.cast_abs]
  rfl

/-- `tl.neg` of a finite value is exact. -/
theorem neg_exact {x r : F32} (hx : x.isFinite = true)
    (h : FloatV.NaNEq (FloatV.neg x) r) : r.isFinite = true ∧ r.toReal = -x.toReal := by
  have hfin : (FloatV.neg x).isFinite = true := by rw [FloatV.isFinite_neg]; exact hx
  obtain ⟨hf, -, hv⟩ := FloatV.NaNEq.exact_of_isFinite h hfin
  exact ⟨hf, by rw [hv, FloatV.toReal_neg]⟩

/-- The widening `bfloat16 → float32` of a finite value is exact. -/
theorem cvt_up {x : BF16} {r : F32} (hx : x.isFinite = true)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn (g := Format.binary32) x) r) :
    r.isFinite = true ∧ r.toReal = x.toReal := by
  obtain ⟨h1, -, h3⟩ := FloatV.cvt_bf16_f32_exact hx
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨hf, -, hv⟩ := FloatV.NaNEq.exact_of_isFinite h h1
  exact ⟨hf, by rw [hv, h3]⟩

/-- One IEEE rounding of a finite sum in range. -/
theorem add_step {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal + y.toReal| ≤ (Format.binary32.maxVal : ℝ))
    (h : FloatV.NaNEq (FloatV.add x y) r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤
      |x.toReal + y.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.add_err_real Format.binary32_regular hx hy hmax
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  obtain ⟨hf, -, hv⟩ := FloatV.NaNEq.exact_of_isFinite h h1
  exact ⟨hf, by rw [hv]; exact h2⟩

/-- One IEEE rounding of a finite difference in range. -/
theorem sub_step {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal - y.toReal| ≤ (Format.binary32.maxVal : ℝ))
    (h : FloatV.NaNEq (FloatV.sub x y) r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal - y.toReal)| ≤
      |x.toReal - y.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.sub_err_real Format.binary32_regular hx hy hmax
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  obtain ⟨hf, -, hv⟩ := FloatV.NaNEq.exact_of_isFinite h h1
  exact ⟨hf, by rw [hv]; exact h2⟩

/-- The rounding `float32 → bfloat16` of a finite value in range. -/
theorem cvt_down {q : F32} {b : BF16} (hq : q.isFinite = true)
    (hmax : |q.toReal| ≤ (Format.bfloat16.maxVal : ℝ))
    (h : FloatV.NaNEq (FloatV.cvtWith .rn (g := Format.bfloat16) q) b) :
    b.isFinite = true ∧ |b.toReal - q.toReal| ≤ |q.toReal| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hq hmax
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h2
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨hf, -, hv⟩ := FloatV.NaNEq.exact_of_isFinite h h1
  exact ⟨hf, by rw [hv]; exact h2⟩

/-- The host's `float32` of the temperature: finite, within `2^-23·|t|` of `t`. -/
theorem t32_err {t : F64} (hf : t.isFinite = true) (hlo : (2 : ℝ) ^ (-126 : ℤ) ≤ |t.toReal|)
    (hhi : |t.toReal| ≤ 2 ^ 127) :
    (FloatV.hostF32 t).isFinite = true ∧
      |(FloatV.hostF32 t).toReal - t.toReal| ≤ 2 ^ (-23 : ℤ) * |t.toReal| := by
  have hn : t.isNaN = false := Format.not_isNaN_of_isFinite hf
  have he : FloatV.hostF32 t = (FloatV.cvt t : F32) := by
    unfold FloatV.hostF32; rw [hn]; rfl
  rw [he]
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.binary32) Format.binary32_regular hf
    (hhi.trans maxVal32_ge_two)
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  refine ⟨h1, h2.trans ?_⟩
  have : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-24 : ℤ) * |t.toReal| := by
    calc (2 : ℝ) ^ (-150 : ℤ) = 2 ^ (-24 : ℤ) * 2 ^ (-126 : ℤ) := by norm_num
      _ ≤ 2 ^ (-24 : ℤ) * |t.toReal| := by gcongr
  have h3 : (2 : ℝ) ^ (-23 : ℤ) = 2 * 2 ^ (-24 : ℤ) := by norm_num
  rw [h3]
  linarith

/-! ## The platform's operations -/

/-- `tl.exp` of `-a` (`a ≥ 0`), from `Device.ExpF32` at `X = a`. -/
theorem exp_step {P : Platform} (hacc : Device.ExpF32 P) {x r : F32} {a : ℝ} (ha : 0 ≤ a)
    (hm : P.math .exp .binary32 x r) (hx : x.isFinite = true) (hxa : x.toReal = -a) :
    r.isFinite = true ∧
      |r.toReal - Real.exp (-a)| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * a) * Real.exp (-a) + 2 ^ (-140 : ℤ) := by
  obtain ⟨ref, href, H⟩ := hacc a ha
  have href' : ref = Real.exp := by
    have h := href
    rw [show MathFn.ref .exp = some Real.exp from rfl] at h
    exact (Option.some.inj h).symm
  subst href'
  have hmem : x.toReal ∈ Set.Icc (-a) a := by rw [hxa, Set.mem_Icc]; constructor <;> linarith
  have hE1 : Real.exp (-a) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have haE := mul_exp_neg_le ha
  have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * a)) * |Real.exp x.toReal| + 2 ^ (-140 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    rw [hxa, abs_of_pos (Real.exp_pos _)]
    have h2 := maxVal32_ge_two
    have hE0 := Real.exp_pos (-a)
    have : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * a)) * Real.exp (-a) =
        Real.exp (-a) + 2 ^ (-22 : ℤ) * Real.exp (-a) + 2 ^ (-23 : ℤ) * (a * Real.exp (-a)) := by ring
    rw [this]
    have h23 : (2 : ℝ) ^ (-23 : ℤ) * (a * Real.exp (-a)) ≤ 2 ^ (-23 : ℤ) * (3 / 8) :=
      mul_le_mul_of_nonneg_left haE (by positivity)
    have h22 : (2 : ℝ) ^ (-22 : ℤ) * Real.exp (-a) ≤ 2 ^ (-22 : ℤ) :=
      mul_le_of_le_one_right (by positivity) hE1
    norm_num at h2 h23 h22 ⊢
    linarith
  obtain ⟨hfin, herr⟩ := H x r hm hx hmem (Set.mem_univ _) hroom
  refine ⟨hfin, ?_⟩
  rw [hxa, abs_of_pos (Real.exp_pos _)] at herr
  exact herr

/-- `tl.log` of a finite value in `[1/2, 3]`, from `Device.LogF32`. -/
theorem log_step {P : Platform} (hacc : Device.LogF32 P) {x r : F32}
    (hm : P.math .log .binary32 x r) (hx : x.isFinite = true) (h1 : 1 / 2 ≤ x.toReal)
    (h2 : x.toReal ≤ 3) :
    r.isFinite = true ∧
      |r.toReal - Real.log x.toReal| ≤ 2 ^ (-22 : ℤ) * |Real.log x.toReal| + 2 ^ (-149 : ℤ) ∧
      |Real.log x.toReal| ≤ 2 := by
  obtain ⟨ref, href, H⟩ := hacc
  have href' : ref = Real.log := by
    have h := href
    rw [show MathFn.ref .log = some Real.log from rfl] at h
    exact (Option.some.inj h).symm
  subst href'
  have hpos : 0 < x.toReal := by linarith
  have hlog : |Real.log x.toReal| ≤ 2 := by
    rw [abs_le]
    constructor
    · have := Real.one_sub_inv_le_log_of_pos hpos
      have : x.toReal⁻¹ ≤ 2 := by
        rw [inv_le_comm₀ hpos (by norm_num)]; linarith
      linarith
    · have := Real.log_le_sub_one_of_pos hpos
      linarith
  have hroom : (1 + 2 ^ (-22 : ℤ)) * |Real.log x.toReal| + 2 ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    have := maxVal32_ge_two
    have h3 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log x.toReal| ≤ (1 + 2 ^ (-22 : ℤ)) * 2 :=
      mul_le_mul_of_nonneg_left hlog (by positivity)
    norm_num at this h3 ⊢
    linarith
  obtain ⟨hf, herr⟩ := H x r hm hx (Set.mem_univ _) (Set.mem_Ioi.mpr hpos) hroom
  exact ⟨hf, herr, hlog⟩

/-- `/` on `float32`, from `Device.DivFull`. -/
theorem div_step {P : Platform} (hfull : Device.DivFull P) {x y r : F32}
    (hm : P.div .binary32 x y r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hy0 : y.toReal ≠ 0) (hq : |x.toReal / y.toReal| ≤ (2 : ℝ) ^ (126 : ℤ)) :
    r.isFinite = true ∧
      |r.toReal - x.toReal / y.toReal| ≤ 2 ^ (-22 : ℤ) * |x.toReal / y.toReal| + 2 ^ (-148 : ℤ) := by
  have hyq : y.toRat ≠ 0 := by
    intro h0
    apply hy0
    simp only [FloatV.toReal, h0]
    norm_num
  have hqq : |x.toRat / y.toRat| ≤ (2 : ℚ) ^ (126 : ℤ) := by
    have h := hq
    simp only [FloatV.toReal] at h
    rw [← Rat.cast_div, ← Rat.cast_abs] at h
    exact_mod_cast h
  obtain ⟨hfin, herr⟩ := hfull x y r hm hx hy hyq hqq
  refine ⟨hfin, ?_⟩
  have h1 : ((|r.toRat - x.toRat / y.toRat| : ℚ) : ℝ) ≤
      ((2 ^ (-22 : ℤ) * |x.toRat / y.toRat| + 2 ^ (-148 : ℤ) : ℚ) : ℝ) := by
    exact_mod_cast herr
  simp only [FloatV.toReal]
  push_cast at h1
  exact h1

/-! ## One lane -/

/-- **One enabled lane.** The input `v` (a finite `bfloat16`) and the temperature's double `t`
(finite, `2^-126 ≤ |t| ≤ 2^127`, `|v| + 1 ≤ 2^125·|t|`); the results of the lane's operations as
the rules give them. The stored `c` is a finite `bfloat16` within `yTol` of `lsRef v / t`. -/
theorem lane {P : Platform} (hexp : Device.ExpF32 P) (hlog : Device.LogF32 P)
    (hfull : Device.DivFull P) {v : BF16} {t : F64}
    {bx bm ba bna be bz bl bs by' : F32} {c : BF16}
    (hv : v.isFinite = true) (htf : t.isFinite = true) (htlo : (2 : ℝ) ^ (-126 : ℤ) ≤ |t.toReal|)
    (hthi : |t.toReal| ≤ 2 ^ 127) (hroom : |v.toReal| + 1 ≤ 2 ^ 125 * |t.toReal|)
    (hbx : FloatV.NaNEq (FloatV.cvtWith .rn (g := Format.binary32) v) bx)
    (hbm : FloatV.MaxRel false zeroF32 bx bm)
    (hba : FloatV.NaNEq (FloatV.abs bx) ba)
    (hbna : FloatV.NaNEq (FloatV.neg ba) bna)
    (hbe : P.math .exp .binary32 bna be)
    (hbz : FloatV.NaNEq (FloatV.add oneF32 be) bz)
    (hbl : P.math .log .binary32 bz bl)
    (hbs : FloatV.NaNEq (FloatV.sub bm bl) bs)
    (hby : P.div .binary32 bs (FloatV.hostF32 t) by')
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn (g := Format.bfloat16) by') c) :
    c.isFinite = true ∧ |c.toReal - lsRef v.toReal / t.toReal| ≤ yTol v.toReal t.toReal := by
  obtain ⟨hbxf, hbxv⟩ := cvt_up hv hbx
  have hbxn : bx.isNaN = false := Format.not_isNaN_of_isFinite hbxf
  have hbmf : bm.isFinite = true := by
    rcases hbm.mem zeroF32_isNaN hbxn with h | h
    · rw [h]; decide
    · rw [h]; exact hbxf
  have hbmv : bm.toReal = min 0 v.toReal := by
    rw [hbm.toReal_eq_min zeroF32_isNaN hbxn, zeroF32_toReal, hbxv]
  obtain ⟨hbaf, hbav⟩ := abs_exact hbxf hba
  obtain ⟨hbnaf, hbnav⟩ := neg_exact hbaf hbna
  rw [hbav, hbxv] at hbnav
  obtain ⟨hbef, hbee⟩ := exp_step hexp (abs_nonneg v.toReal) hbe hbnaf hbnav
  obtain ⟨-, he1, he2⟩ := e_bounds (abs_nonneg v.toReal) hbee
  have h1e : |oneF32.toReal + be.toReal| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [oneF32_toReal]
    have := maxVal32_ge_two
    rw [abs_le]
    norm_num at he1 he2 this ⊢
    constructor <;> linarith
  obtain ⟨hbzf, hbzz⟩ := add_step oneF32_isFinite hbef h1e hbz
  rw [oneF32_toReal] at hbzz
  obtain ⟨hz1, hz2⟩ := z_bounds he1 he2 hbzz
  obtain ⟨hblf, hbll, hlz⟩ := log_step hlog hbl hbzf hz1 hz2
  have hLerr := L_err (abs_nonneg v.toReal) hbee hbzz hbll
  have hsub : |bm.toReal - bl.toReal| ≤ (Format.binary32.maxVal : ℝ) := by
    have hvb := FloatV.abs_toReal_le_maxVal Format.bfloat16_regular hv
    have hbm1 : |bm.toReal| ≤ |v.toReal| := by
      rw [hbmv]
      rcases le_total 0 v.toReal with h | h
      · rw [min_eq_left h, abs_zero]; exact abs_nonneg _
      · rw [min_eq_right h]
    have hbl1 : |bl.toReal| ≤ 3 := by
      have := abs_sub_abs_le_abs_sub bl.toReal (Real.log bz.toReal)
      norm_num at hbll this hlz ⊢
      nlinarith [abs_nonneg (Real.log bz.toReal)]
    have := maxVal_bf16_lt
    calc |bm.toReal - bl.toReal| ≤ |bm.toReal| + |bl.toReal| := abs_sub _ _
      _ ≤ (Format.bfloat16.maxVal : ℝ) + 3 := by linarith
      _ ≤ _ := this
  obtain ⟨hbsf, hbss⟩ := sub_step hbmf hblf hsub hbs
  rw [hbmv] at hbss
  obtain ⟨hS, -, hls⟩ := s_err hLerr hbss
  obtain ⟨ht32f, ht32⟩ := t32_err htf htlo hthi
  have ht0 : 0 < |t.toReal| := lt_of_lt_of_le (by positivity) htlo
  have hwr := div_room ht0 ht32 hS hls hroom
  have ht32ne : (FloatV.hostF32 t).toReal ≠ 0 := abs_pos.mp (t32_bounds ht0 ht32).2
  obtain ⟨hbyf, hbyq⟩ := div_step hfull hby hbsf ht32f ht32ne hwr
  have hqr := q_room hwr hbyq
  obtain ⟨hcf, hcq⟩ := cvt_down hbyf (hqr.trans maxVal_bf16_ge) hc
  exact ⟨hcf, final ht0 ht32 hS hbyq hcq⟩

end Claims.LogsigmoidFwd
