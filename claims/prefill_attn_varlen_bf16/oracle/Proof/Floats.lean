import Claim
import TVA.Proof
import Proof.NumStep
import Proof.Setup

section
/-!
# The float operations of one trip, as inequalities over `ℝ`

One lemma per operation of the kernel's loop body, from the device facts (`Device P`) and the
kit's error lemmas: the value the platform gives is finite and within the stated error of the exact
one. The loop's analysis (`Proof.Num*`) consumes them.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- `x - y` on `float32` (up to NaN payload): finite and within one rounding. -/
theorem f32_sub {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal - y.toReal| ≤ 2 ^ 100) (h : FloatV.NaNEq (FloatV.sub x y) r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal - y.toReal)| ≤ 2 ^ (-24 : ℤ) * |x.toReal - y.toReal| + 2 ^ (-150 : ℤ) := by
  have hm : (2 : ℝ) ^ 100 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.sub_err_real Format.binary32_regular hx hy (hb.trans hm)
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  refine ⟨hf, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  rw [hr]; linarith


/-- `tl.exp` of an argument `a` of size at most `X`, `a ≤ 2^-100`: the device fact in `ℝ`. -/
theorem f32_exp_acc {P : Platform} (hP : Device P) {a p : F32} (ha : a.isFinite = true)
    {X : ℝ} (hX0 : 0 ≤ X) (hX : |a.toReal| ≤ X) (hX1 : X ≤ 20000)
    (ha0 : a.toReal ≤ 2 ^ (-100 : ℤ)) (hp : P.math .exp .binary32 a p) :
    p.isFinite = true ∧ |p.toReal - Real.exp a.toReal| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X) * Real.exp a.toReal + 2 ^ (-140 : ℤ) := by
  obtain ⟨ref, href, hacc⟩ := hP.expF32 X hX0
  have href' : ref = Real.exp := by
    have h := href
    simp only [TVA.MathFn.ref] at h
    exact (Option.some.inj h).symm
  rw [href'] at hacc
  have hmem : a.toReal ∈ Set.Icc (-X) X :=
    ⟨by linarith [neg_abs_le a.toReal], le_trans (le_abs_self _) hX⟩
  have hexp : Real.exp a.toReal ≤ 2 := by
    by_cases h0 : a.toReal ≤ 0
    · linarith [Real.exp_le_one_iff.mpr h0]
    · have h1 : a.toReal ≤ 1 := ha0.trans (by norm_num)
      have := TVA.Softmax.exp_le_one_add_two_mul (not_le.mp h0).le h1
      have h3 : a.toReal ≤ 1 / 2 := ha0.trans (by norm_num)
      linarith
  have hexp0 : 0 < Real.exp a.toReal := Real.exp_pos _
  have hmax : (2 : ℝ) ^ 100 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X)) * |Real.exp a.toReal| + 2 ^ (-140 : ℤ) ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hexp0]
    have h1 : (2 : ℝ) ^ (-23 : ℤ) * X ≤ 1 := by
      have : (2 : ℝ) ^ (-23 : ℤ) * 20000 ≤ 1 := by norm_num
      nlinarith [(by norm_num : (0 : ℝ) ≤ 2 ^ (-23 : ℤ))]
    have h2 : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
    have h3 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X)) ≤ 3 := by linarith
    have h4 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X)) * Real.exp a.toReal ≤ 3 * 2 := by
      have : 0 ≤ 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X) := by
        have : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) * X := mul_nonneg (by norm_num) hX0
        positivity
      nlinarith
    have h5 : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 := by norm_num
    have h6 : (2 : ℝ) ^ 100 ≥ 8 := by norm_num
    linarith
  obtain ⟨hf, herr⟩ := hacc a p hp ha hmem (Set.mem_univ _) hroom
  refine ⟨hf, ?_⟩
  rw [abs_of_pos hexp0] at herr
  exact herr


/-- `tl.exp(s - m)` for `s ≤ m` (`Z` bounds `m - s`, `Z ≤ 10000`): the subtraction is rounded, the
exponential is the device's. Finite, within `2^-22·(8/5 + 4/5·Z)·exp(s - m) + 2^-140`. -/
theorem f32_exp_sub {P : Platform} (hP : Device P) {s m a p : F32}
    (hs : s.isFinite = true) (hm : m.isFinite = true) (hsm : s.toReal ≤ m.toReal)
    {Z : ℝ} (hZ : m.toReal - s.toReal ≤ Z) (hZ1 : Z ≤ 10000)
    (ha : FloatV.NaNEq (FloatV.sub s m) a) (hp : P.math .exp .binary32 a p) :
    p.isFinite = true ∧ |p.toReal - Real.exp (s.toReal - m.toReal)| ≤
      2 ^ (-22 : ℤ) * (8 / 5 + 4 / 5 * Z) * Real.exp (s.toReal - m.toReal) + 2 ^ (-140 : ℤ) := by
  set d := s.toReal - m.toReal with hd
  have hd0 : d ≤ 0 := by rw [hd]; linarith
  have hdZ : |d| ≤ Z := by rw [abs_of_nonpos hd0]; linarith
  have hZ0 : 0 ≤ Z := (abs_nonneg _).trans hdZ
  obtain ⟨haf, hae⟩ := f32_sub hs hm (by
    have : |d| ≤ 2 ^ 100 := hdZ.trans (hZ1.trans (by norm_num))
    exact this) ha
  set ε := 2 ^ (-24 : ℤ) * |d| + 2 ^ (-150 : ℤ) with hε
  have hε1 : ε ≤ 2 ^ (-24 : ℤ) * Z + 2 ^ (-150 : ℤ) := by
    rw [hε]; have : (0 : ℝ) ≤ 2 ^ (-24 : ℤ) := by norm_num
    nlinarith
  have hεs : ε ≤ 7 / 10000 := by
    have h1 : (2 : ℝ) ^ (-24 : ℤ) * 10000 + 2 ^ (-150 : ℤ) ≤ 7 / 10000 := by norm_num
    have h2 : (2 : ℝ) ^ (-24 : ℤ) * Z ≤ 2 ^ (-24 : ℤ) * 10000 :=
      mul_le_mul_of_nonneg_left hZ1 (by norm_num)
    exact hε1.trans ((add_le_add h2 le_rfl).trans h1)
  have hε0 : 0 ≤ ε := by rw [hε]; positivity
  have hadiff : |a.toReal - d| ≤ ε := hae
  have ha_le : a.toReal ≤ d + ε := by linarith [(abs_le.mp hadiff).2]
  have ha_ge : d - ε ≤ a.toReal := by linarith [(abs_le.mp hadiff).1]
  have ha0 : a.toReal ≤ 2 ^ (-100 : ℤ) := by
    have : d + ε ≤ 2 ^ (-150 : ℤ) := by
      rw [hε, abs_of_nonpos hd0]
      have h24 : (2 : ℝ) ^ (-24 : ℤ) ≤ 1 := by norm_num
      nlinarith [neg_nonneg.mpr hd0]
    have h2 : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-100 : ℤ) := by norm_num
    linarith
  have haX : |a.toReal| ≤ Z + 1 := by
    rw [abs_le]; constructor
    · have : -Z ≤ d := by linarith [(abs_le.mp hdZ).1]
      linarith
    · linarith
  obtain ⟨hpf, hpe⟩ := f32_exp_acc hP haf (by linarith) haX (by linarith) ha0 hp
  refine ⟨hpf, ?_⟩
  -- exp a against exp d
  have hea : Real.exp a.toReal ≤ Real.exp d * (1 + 2 * ε) := by
    have h1 : Real.exp a.toReal ≤ Real.exp (d + ε) := Real.exp_le_exp.mpr ha_le
    have h2 : Real.exp (d + ε) = Real.exp d * Real.exp ε := Real.exp_add _ _
    have h3 : Real.exp ε ≤ 1 + 2 * ε := TVA.Softmax.exp_le_one_add_two_mul hε0 (by linarith)
    rw [h2] at h1
    exact h1.trans (mul_le_mul_of_nonneg_left h3 (Real.exp_pos _).le)
  have hed : |Real.exp a.toReal - Real.exp d| ≤ ε * (Real.exp d * (1 + 2 * ε)) := by
    refine (TVA.Softmax.abs_exp_sub_exp_le _ _).trans ?_
    have hm' : max a.toReal d ≤ d + ε := max_le ha_le (by linarith)
    have h1 : Real.exp (max a.toReal d) ≤ Real.exp d * (1 + 2 * ε) := by
      refine (Real.exp_le_exp.mpr hm').trans ?_
      rw [Real.exp_add]
      exact mul_le_mul_of_nonneg_left (TVA.Softmax.exp_le_one_add_two_mul hε0 (by linarith))
        (Real.exp_pos _).le
    exact mul_le_mul hadiff h1 (Real.exp_pos _).le hε0
  have hed0 : 0 < Real.exp d := Real.exp_pos _
  have hrel : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1)) * Real.exp a.toReal ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1)) * (Real.exp d * (1 + 2 * ε)) :=
    mul_le_mul_of_nonneg_left hea (by positivity)
  have key : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1)) * (Real.exp d * (1 + 2 * ε)) +
      ε * (Real.exp d * (1 + 2 * ε)) ≤ 2 ^ (-22 : ℤ) * (8 / 5 + 4 / 5 * Z) * Real.exp d := by
    have e1 : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1)) * (Real.exp d * (1 + 2 * ε)) +
        ε * (Real.exp d * (1 + 2 * ε)) =
        Real.exp d * ((1 + 2 * ε) * (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1) + ε)) := by ring
    rw [e1]
    have h2 : (1 + 2 * ε) * (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1) + ε) ≤
        2 ^ (-22 : ℤ) * (8 / 5 + 4 / 5 * Z) := by
      have hε2 : ε ≤ 2 ^ (-24 : ℤ) * Z + 2 ^ (-150 : ℤ) := hε1
      have c1 : (1 + 2 * ε) ≤ 1 + 14 / 10000 := by linarith
      have c2 : 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1) + ε ≤
          2 ^ (-22 : ℤ) * (3 / 2 + 3 / 4 * Z) + 2 ^ (-150 : ℤ) := by
        have : (2 : ℝ) ^ (-23 : ℤ) = 2 ^ (-22 : ℤ) / 2 := by norm_num
        have h24 : (2 : ℝ) ^ (-24 : ℤ) = 2 ^ (-22 : ℤ) / 4 := by norm_num
        rw [this]; rw [h24] at hε2; linarith
      have hc2pos : 0 ≤ 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1) + ε := by positivity
      calc (1 + 2 * ε) * (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1) + ε)
          ≤ (1 + 14 / 10000) * (2 ^ (-22 : ℤ) * (3 / 2 + 3 / 4 * Z) + 2 ^ (-150 : ℤ)) :=
            mul_le_mul c1 c2 hc2pos (by norm_num)
        _ ≤ 2 ^ (-22 : ℤ) * (8 / 5 + 4 / 5 * Z) := by
            have h150 : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-22 : ℤ) / 10000 := by norm_num
            nlinarith [(by norm_num : (0 : ℝ) < 2 ^ (-22 : ℤ))]
    nlinarith [hed0, h2]
  calc |p.toReal - Real.exp d| ≤ |p.toReal - Real.exp a.toReal| + |Real.exp a.toReal - Real.exp d| :=
        abs_sub_le _ _ _
    _ ≤ ((2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1)) * Real.exp a.toReal + 2 ^ (-140 : ℤ)) +
        ε * (Real.exp d * (1 + 2 * ε)) := add_le_add hpe hed
    _ ≤ ((2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (Z + 1)) * (Real.exp d * (1 + 2 * ε)) +
          2 ^ (-140 : ℤ)) + ε * (Real.exp d * (1 + 2 * ε)) :=
        add_le_add (add_le_add hrel le_rfl) le_rfl
    _ ≤ _ := by linarith [key]


end Claims.PrefillAttnVarlen

end

section
/-!
# The score lane and the final conversion

`score_lane` and `score_lane_masked`: one lane `(r, c)` of lines 131-145 of the kernel, the
`bfloat16` dot product `qk₁ = tl.dot(q, k)` scaled by `sm_scale` and added to the mask (`+0` where
the key is attended, `-∞` otherwise). `cvt_store`: the conversion of line 183 of the accumulator
to `bfloat16`.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- The `float32` `+0` is finite and has value `0`. -/
theorem zf32_val : zf32.isFinite = true ∧ zf32.toRat = 0 := by
  have h : (FloatV.ofPat 0 : F32).isFinite = true := by
    rw [FloatV.isFinite_ofPat (Nat.two_pow_pos 32)]
    exact Format.isFinite_zero Format.binary32_regular
  have hz : (FloatV.ofPat 0 : F32).toRat = 0 :=
    FloatV.toRat_of_isZero (x := (FloatV.ofPat 0 : F32)) Format.isZero_zero
  simp only [zf32, FloatV.litF32_ofPat_zero]
  exact ⟨h, hz⟩

/-- The `float32` `-∞` of the claim is `FloatV.negInf`. -/
theorem ninf32_eq : ninf32 = (FloatV.negInf : F32) := by
  unfold ninf32 FloatV.litF32
  decide

/-- **The host's `float32` scale** `(float)c` of a finite double `c` in `[2^-64, 2^64]`: finite and
within `2^-24·c + 2^-150` of `c`. -/
theorem hostF32_err {c : F64} (hc : c.isFinite = true) (hlo : (2 : ℝ) ^ (-64 : ℤ) ≤ c.toReal)
    (hhi : c.toReal ≤ 2 ^ 64) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toReal - c.toReal| ≤ 2 ^ (-24 : ℤ) * c.toReal + 2 ^ (-150 : ℤ) := by
  have hn : c.isNaN = false := (FloatV.isFinite_iff.mp hc).1
  have hpos : 0 < c.toReal := lt_of_lt_of_le (by positivity) hlo
  have hmax : |c.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hpos, Format.binary32_maxVal_real]
    exact hhi.trans (by norm_num)
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.binary32) Format.binary32_regular hc hmax
  simp only [FloatV.hostF32, hn, Bool.false_eq_true, ite_false]
  rw [Format.binary32_u_real, Format.binary32_eta_real, abs_of_pos hpos] at h2
  exact ⟨h1, by linarith⟩

/-- **One lane of the score dot product** `qk₁ = tl.dot(q, k)` of `bfloat16` operands from a zero
accumulator: the lane `(r, c)` is finite and within `(BD + 3)·2^-22·Σ liftMag q·liftMag k + 2^-110`
of the exact `Σ q·k` (`Q`, `K` the loaded values), given room (`Σ liftMag q·liftMag k ≤ 2^120`). -/
theorem dot_lane {P : Platform} (hP : Device P) {fusion : Bool} {BD : ℕ} (hBD : BD ≤ 256)
    {q : Tensor [128, BD] BF16} {k : Tensor [BD, 128] BF16} {qk₀ z qk₁ : Tensor [128, 128] F32}
    (hr : P.dot cfgBF16 q k qk₀ qk₁ ∨
      ∃ d', P.dot cfgBF16 q k z d' ∧ ∀ i, AccAdd fusion .f32 (d'.get i) (qk₀.get i) (qk₁.get i))
    (r c : Fin 128) (hz : DType.val? .f32 (z.get2 r c) = some 0)
    (h0 : DType.val? .f32 (qk₀.get2 r c) = some 0) {Q K : Fin BD → ℚ}
    (hq : ∀ e, DType.val? .bf16 (q.get2 r e) = some (Q e))
    (hk : ∀ e, DType.val? .bf16 (k.get2 e c) = some (K e))
    (hL : ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) ≤ 2 ^ 120) :
    (qk₁.get2 r c).isFinite = true ∧
      |(qk₁.get2 r c).toReal - ∑ e, ((Q e : ℚ) : ℝ) * ((K e : ℚ) : ℝ)| ≤
        ((BD : ℝ) + 3) * 2 ^ (-22 : ℤ) *
            ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) +
          2 ^ (-110 : ℤ) := by
  obtain ⟨hc0, hc1⟩ := DType.val?_f32.mp h0
  have hLq : ∑ e, Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) ≤ 2 ^ 120 := by
    exact_mod_cast (by simpa using hL : (((∑ e, Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ)) : ℝ) ≤ ((2 ^ 120 : ℚ) : ℝ))
  have hL0 : 0 ≤ ∑ e, Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) :=
    Finset.sum_nonneg fun e _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hrel : 0 ≤ Device.relDot16 BD := by unfold Device.relDot16; positivity
  have habs : 0 ≤ Device.absDot16 BD := by unfold Device.absDot16; positivity
  have hrel' : Device.relDot16 BD ≤ 2 ^ (-13 : ℤ) := by
    unfold Device.relDot16
    have : (BD : ℚ) ≤ 256 := by exact_mod_cast hBD
    have h : ((BD : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ (256 + 2) * 2 ^ (-22 : ℤ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
    exact h.trans (by norm_num)
  have habs' : Device.absDot16 BD ≤ 2 ^ (-117 : ℤ) := by
    unfold Device.absDot16
    have : (BD : ℚ) ≤ 256 := by exact_mod_cast hBD
    have h : ((BD : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ (256 + 2) * 2 ^ (-126 : ℤ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
    exact h.trans (by norm_num)
  have hroom : 2 * ((1 + Device.relDot16 BD) *
      (|(qk₀.get2 r c).toRat| + ∑ e, Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e)) +
      Device.absDot16 BD + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    rw [hc1, abs_zero, zero_add, Format.binary32_maxVal, Format.binary32_eta]
    have h1 : (1 + Device.relDot16 BD) * (∑ e, Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e)) ≤
        (1 + 2 ^ (-13 : ℤ)) * 2 ^ 120 :=
      mul_le_mul (by linarith) hLq hL0 (by norm_num)
    have : (2 : ℚ) * ((1 + 2 ^ (-13 : ℤ)) * 2 ^ 120 + 2 ^ (-117 : ℤ) + 2 * 2 ^ (-150 : ℤ)) ≤
        (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  have hva : ∀ e, DType.val? .bf16 (q.get2 r e) = some (Q e) := hq
  have hstep := DotLift.dotAdd_step (fusion := fusion) (hP.dotBF16 .tf32) hrel habs hr r c hz hva hk
    (DType.val?_f32.mp h0).1 hroom
  refine ⟨hstep.1, ?_⟩
  have h2 := hstep.2
  rw [hc1, abs_zero, zero_add, zero_add] at h2
  have hsr : DotLoop.stepRel Device.relDot16 BD ≤ ((BD : ℚ) + 3) * 2 ^ (-22 : ℤ) := by
    unfold DotLoop.stepRel
    rw [Format.binary32_u]
    unfold Device.relDot16 at hrel' ⊢
    have ht : (0 : ℚ) < 2 ^ (-22 : ℤ) := by norm_num
    rw [show (2 : ℚ) ^ (-24 : ℤ) = 2 ^ (-22 : ℤ) / 4 by norm_num]
    have h13 : (2 : ℚ) ^ (-13 : ℤ) ≤ 1 / 8 := by norm_num
    have h4 : 2 ^ (-22 : ℤ) * (((BD : ℚ) + 2) * 2 ^ (-22 : ℤ)) ≤ 2 ^ (-22 : ℤ) * (1 / 8) :=
      mul_le_mul_of_nonneg_left (hrel'.trans h13) ht.le
    have e : (BD + 2 : ℚ) * 2 ^ (-22 : ℤ) + 3 * (2 ^ (-22 : ℤ) / 4) *
        (1 + (BD + 2 : ℚ) * 2 ^ (-22 : ℤ)) = ((BD : ℚ) + 2) * 2 ^ (-22 : ℤ) +
        3 / 4 * 2 ^ (-22 : ℤ) + 3 / 4 * (2 ^ (-22 : ℤ) * (((BD : ℚ) + 2) * 2 ^ (-22 : ℤ))) := by ring
    rw [e]
    nlinarith
  have hsa : DotLoop.stepAbs Device.absDot16 BD ≤ 2 ^ (-110 : ℤ) := by
    unfold DotLoop.stepAbs
    rw [Format.binary32_u, Format.binary32_eta]
    have : Device.absDot16 BD * (1 + 3 * 2 ^ (-24 : ℤ)) ≤ 2 ^ (-117 : ℤ) * 2 := by
      have := mul_le_mul habs' (by norm_num : (1 + 3 * 2 ^ (-24 : ℤ) : ℚ) ≤ 2) (by norm_num) (by norm_num)
      linarith
    have h2 : (2 : ℚ) ^ (-117 : ℤ) * 2 + 4 * 2 ^ (-150 : ℤ) ≤ 2 ^ (-110 : ℤ) := by norm_num
    linarith
  have h3 : |(qk₁.get2 r c).toRat - ∑ e, Q e * K e| ≤
      ((BD : ℚ) + 3) * 2 ^ (-22 : ℤ) * ∑ e, Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) +
        2 ^ (-110 : ℤ) := by
    refine h2.trans (add_le_add ?_ hsa)
    exact mul_le_mul_of_nonneg_right hsr hL0
  have h5 := (Rat.cast_le (K := ℝ)).mpr h3
  unfold FloatV.toReal
  push_cast at h5 ⊢
  exact h5


/-- **The arithmetic of a score lane.** `s` is the exact scale, `L` the lifted magnitude of the dot
(`s·L ≤ X ≤ 2^12`), `x` the computed dot (within `(N + 3)·2^-22·L + 2^-110` of the exact `D`),
`σ` the `float32` scale (within `2^-24·s + 2^-150` of `s`) and `y` the rounded `x·σ`. The product is
at most `2^13` and `y` is within `(N + 8)·2^-22·X + 2^-40` of `s·D`. -/
theorem score_arith {N s L X x σ y D : ℝ} (hN0 : 0 ≤ N) (hN : N ≤ 256)
    (hs0 : 2 ^ (-64 : ℤ) ≤ s) (hs1 : s ≤ 2 ^ 64) (hL0 : 0 ≤ L) (hX : s * L ≤ X)
    (hX1 : X ≤ 2 ^ 12) (hD : |D| ≤ L)
    (hx : |x - D| ≤ (N + 3) * 2 ^ (-22 : ℤ) * L + 2 ^ (-110 : ℤ))
    (hσ : |σ - s| ≤ 2 ^ (-24 : ℤ) * s + 2 ^ (-150 : ℤ))
    (hy : |y - x * σ| ≤ 3 * 2 ^ (-24 : ℤ) * |x * σ| + 4 * 2 ^ (-150 : ℤ)) :
    |x * σ| ≤ 2 ^ 13 ∧ |y - s * D| ≤ (N + 8) * 2 ^ (-22 : ℤ) * X + 2 ^ (-40 : ℤ) := by
  generalize ht : (2 : ℝ) ^ (-22 : ℤ) = t at hx ⊢
  generalize hu : (2 : ℝ) ^ (-24 : ℤ) = u at hσ hy
  generalize he : (2 : ℝ) ^ (-150 : ℤ) = e at hσ hy
  generalize ha : (2 : ℝ) ^ (-110 : ℤ) = a at hx
  have ht0 : 0 < t := by rw [← ht]; positivity
  have hu0 : 0 < u := by rw [← hu]; positivity
  have he0 : 0 < e := by rw [← he]; positivity
  have ha0 : 0 < a := by rw [← ha]; positivity
  have hut : u = t / 4 := by rw [← hu, ← ht]; norm_num
  obtain ⟨R, hR⟩ : ∃ R : ℝ, R = (N + 3) * t := ⟨_, rfl⟩
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hR1 : R ≤ 1 / 8192 := by
    rw [hR]
    have : (N + 3) * t ≤ (256 + 3) * t := mul_le_mul_of_nonneg_right (by linarith) ht0.le
    refine this.trans ?_
    rw [← ht]; norm_num
  have hu1 : u ≤ 1 / 1000000 := by rw [← hu]; norm_num
  have he1 : e = 2 ^ (-150 : ℤ) := he.symm
  have hL : L ≤ 2 ^ 76 := by
    have h1 : 2 ^ (-64 : ℤ) * L ≤ s * L := mul_le_mul_of_nonneg_right hs0 hL0
    have h2 : (2 : ℝ) ^ (-64 : ℤ) * L ≤ 2 ^ 12 := by linarith
    have h3 : L = 2 ^ 64 * (2 ^ (-64 : ℤ) * L) := by
      rw [← mul_assoc]; norm_num
    rw [h3]
    calc (2 : ℝ) ^ 64 * (2 ^ (-64 : ℤ) * L) ≤ 2 ^ 64 * 2 ^ 12 :=
          mul_le_mul_of_nonneg_left h2 (by norm_num)
      _ = 2 ^ 76 := by norm_num
  have hx2 : |x| ≤ L * (1 + R) + a := by
    have h1 : |x| ≤ |D| + |x - D| := by
      have := abs_add_le D (x - D); simpa using this
    have h2 : |x - D| ≤ R * L + a := by
      have : (N + 3) * t * L = R * L := by rw [hR]
      linarith
    nlinarith
  obtain ⟨sb, hsb⟩ : ∃ sb : ℝ, sb = s * (1 + u) + e := ⟨_, rfl⟩
  have hσ2 : |σ| ≤ sb := by
    have h1 : |σ| ≤ |s| + |σ - s| := by
      have := abs_add_le s (σ - s); simpa using this
    rw [abs_of_pos (lt_of_lt_of_le (by norm_num) hs0)] at h1
    rw [hsb]; nlinarith
  have hsb0 : 0 < sb := by rw [hsb]; positivity
  have hxσ : |x * σ| ≤ (L * (1 + R) + a) * sb := by
    rw [abs_mul]
    exact mul_le_mul hx2 hσ2 (abs_nonneg _) (by positivity)
  obtain ⟨sL, hsL⟩ : ∃ sL : ℝ, sL = s * L := ⟨_, rfl⟩
  have hsL0 : 0 ≤ sL := by rw [hsL]; positivity
  have hexp : (L * (1 + R) + a) * sb = sL * ((1 + R) * (1 + u)) + (L * (1 + R) * e + a * sb) := by
    rw [hsb, hsL]; ring
  rw [← hsL] at hX
  have hsb65 : sb ≤ 2 ^ 66 := by
    rw [hsb]
    have h1 : s * (1 + u) ≤ 2 ^ 64 * 2 := mul_le_mul hs1 (by linarith) (by linarith) (by norm_num)
    have h2 : e ≤ 1 := by rw [he1]; norm_num
    have h3 : (2 : ℝ) ^ 64 * 2 + 1 ≤ 2 ^ 66 := by norm_num
    linarith
  have hLe : L * (1 + R) * e ≤ 2 ^ (-73 : ℤ) := by
    have h1 : L * (1 + R) ≤ 2 ^ 76 * 2 := mul_le_mul hL (by linarith) (by linarith) (by norm_num)
    have h2 : L * (1 + R) * e ≤ 2 ^ 76 * 2 * e := mul_le_mul_of_nonneg_right h1 he0.le
    rw [he1] at h2 ⊢
    exact h2.trans (by norm_num : (2 : ℝ) ^ 76 * 2 * 2 ^ (-150 : ℤ) ≤ 2 ^ (-73 : ℤ))
  have hasb : a * sb ≤ 2 ^ (-44 : ℤ) := by
    have h4 : a * sb ≤ a * 2 ^ 66 := mul_le_mul_of_nonneg_left hsb65 ha0.le
    rw [← ha] at h4 ⊢
    exact h4.trans (by norm_num : (2 : ℝ) ^ (-110 : ℤ) * 2 ^ 66 ≤ 2 ^ (-44 : ℤ))
  have hrem1 : L * (1 + R) * e + a * sb ≤ 2 ^ (-43 : ℤ) := by
    have : (2 : ℝ) ^ (-73 : ℤ) + 2 ^ (-44 : ℤ) ≤ 2 ^ (-43 : ℤ) := by norm_num
    linarith
  have hcf : (1 + R) * (1 + u) ≤ 3 / 2 := by
    have : R * u ≤ 1 / 8192 * (1 / 1000000) := mul_le_mul hR1 hu1 hu0.le (by norm_num)
    nlinarith
  have hB : (L * (1 + R) + a) * sb ≤ 2 ^ 13 := by
    rw [hexp]
    have h1 : sL * ((1 + R) * (1 + u)) ≤ 2 ^ 12 * (3 / 2) :=
      mul_le_mul (hX.trans hX1) hcf (by positivity) (by norm_num)
    have : (2 : ℝ) ^ (-43 : ℤ) ≤ 1 := by norm_num
    linarith
  refine ⟨hxσ.trans hB, ?_⟩
  have e1 : |y - s * D| ≤ |y - x * σ| + |σ| * |x - D| + |D| * |σ - s| := by
    have h : y - s * D = (y - x * σ) + σ * (x - D) + D * (σ - s) := by ring
    rw [h]
    calc |(y - x * σ) + σ * (x - D) + D * (σ - s)|
        ≤ |(y - x * σ) + σ * (x - D)| + |D * (σ - s)| := abs_add_le _ _
      _ ≤ |y - x * σ| + |σ * (x - D)| + |D * (σ - s)| := by
          gcongr; exact abs_add_le _ _
      _ = _ := by rw [abs_mul, abs_mul]
  have t1 : |y - x * σ| ≤ 3 * u * ((L * (1 + R) + a) * sb) + 4 * e := by
    have : 3 * u * |x * σ| ≤ 3 * u * ((L * (1 + R) + a) * sb) :=
      mul_le_mul_of_nonneg_left hxσ (by positivity)
    linarith
  have t2 : |σ| * |x - D| ≤ sb * (R * L + a) := by
    have h2 : |x - D| ≤ R * L + a := by
      have : (N + 3) * t * L = R * L := by rw [hR]
      linarith
    exact mul_le_mul hσ2 h2 (abs_nonneg _) hsb0.le
  have t3 : |D| * |σ - s| ≤ L * (u * s + e) := mul_le_mul hD hσ (abs_nonneg _) hL0
  have hsum : |y - s * D| ≤ sL * ((1 + R) * (1 + u) * (3 * u) + R * (1 + u) + u) +
      (3 * u * (L * (1 + R) * e + a * sb) + 4 * e + (R * e * L + sb * a) + L * e) := by
    calc |y - s * D| ≤ |y - x * σ| + |σ| * |x - D| + |D| * |σ - s| := e1
      _ ≤ (3 * u * ((L * (1 + R) + a) * sb) + 4 * e) + sb * (R * L + a) + L * (u * s + e) :=
          add_le_add (add_le_add t1 t2) t3
      _ = _ := by subst hsb hsL; ring
  have hcoef : (1 + R) * (1 + u) * (3 * u) + R * (1 + u) + u ≤ (N + 8) * t := by
    have hut' : u ≤ t := by rw [hut]; linarith
    have h1 : (1 + R) * (1 + u) * (3 * u) ≤ 3 / 2 * (3 * t) :=
      mul_le_mul hcf (by linarith) (by positivity) (by norm_num)
    have h2 : R * u ≤ 1 / 8192 * t := mul_le_mul hR1 hut' hu0.le (by norm_num)
    have h3 : R = (N + 3) * t := hR
    have h4 : R * (1 + u) = R + R * u := by ring
    linarith [h1, h2, h3, h4, hut]
  have hcoef0 : 0 ≤ (1 + R) * (1 + u) * (3 * u) + R * (1 + u) + u := by positivity
  have hrem : 3 * u * (L * (1 + R) * e + a * sb) + 4 * e + (R * e * L + sb * a) + L * e ≤
      2 ^ (-40 : ℤ) := by
    have h1 : 3 * u * (L * (1 + R) * e + a * sb) ≤ 1 * 2 ^ (-43 : ℤ) :=
      mul_le_mul (by linarith) hrem1 (by positivity) (by norm_num)
    have h2 : R * e * L ≤ 2 ^ (-60 : ℤ) := by
      have : R * e ≤ 1 / 8192 * e := mul_le_mul_of_nonneg_right hR1 he0.le
      have h3 : R * e * L ≤ 1 / 8192 * e * L := mul_le_mul_of_nonneg_right this hL0
      rw [he1] at h3 ⊢
      have h4 : (1 / 8192 : ℝ) * 2 ^ (-150 : ℤ) * L ≤ 1 / 8192 * 2 ^ (-150 : ℤ) * 2 ^ 76 :=
        mul_le_mul_of_nonneg_left hL (by norm_num)
      exact h3.trans (h4.trans (by norm_num : (1 / 8192 : ℝ) * 2 ^ (-150 : ℤ) * 2 ^ 76 ≤ 2 ^ (-60 : ℤ)))
    have h5 : sb * a ≤ 2 ^ (-44 : ℤ) := by rw [mul_comm]; exact hasb
    have h6 : L * e ≤ 2 ^ (-73 : ℤ) := by
      have : L * e ≤ 2 ^ 76 * e := mul_le_mul_of_nonneg_right hL he0.le
      rw [he1] at this ⊢
      exact this.trans (by norm_num : (2 : ℝ) ^ 76 * 2 ^ (-150 : ℤ) ≤ 2 ^ (-73 : ℤ))
    have h7 : 4 * e ≤ 2 ^ (-100 : ℤ) := by rw [he1]; norm_num
    have : (1 : ℝ) * 2 ^ (-43 : ℤ) + 2 ^ (-100 : ℤ) + (2 ^ (-60 : ℤ) + 2 ^ (-44 : ℤ)) +
        2 ^ (-73 : ℤ) ≤ 2 ^ (-40 : ℤ) := by norm_num
    linarith
  have hfin : sL * ((1 + R) * (1 + u) * (3 * u) + R * (1 + u) + u) ≤ X * ((N + 8) * t) :=
    mul_le_mul hX hcoef hcoef0 (hsL0.trans hX)
  calc |y - s * D| ≤ _ := hsum
    _ ≤ X * ((N + 8) * t) + 2 ^ (-40 : ℤ) := add_le_add hfin hrem
    _ = _ := by ring


/-- A finite `float32` plus `-∞` is `-∞`. -/
theorem add_negInf {x : F32} (hx : x.isFinite = true) :
    FloatV.add x FloatV.negInf = FloatV.negInf := by
  have hxn := FloatV.isFinite_iff.mp hx
  have hs : Format.binary32.specials = .ieee := rfl
  unfold FloatV.add Format.add
  have h1 : Format.binary32.isNaN (FloatV.negInf : F32).pat = false := FloatV.negInf_isNaN hs
  have h2 : Format.binary32.isInf (FloatV.negInf : F32).pat = true := FloatV.negInf_isInf hs
  have h3 : Format.binary32.isNaN x.pat = false := hxn.1
  have h4 : Format.binary32.isInf x.pat = false := hxn.2
  simp only [h1, h2, h3, h4, Bool.or_false, Bool.false_eq_true, ite_false, ite_true]
  unfold Format.infPat; rw [hs]
  have := FloatV.negInf_sign (f := Format.binary32); unfold FloatV.sign at this; rw [this]
  rfl

/-- `fma x y -∞` of finite `x`, `y` is `-∞`. -/
theorem fma_negInf {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    FloatV.fma x y FloatV.negInf = FloatV.negInf := by
  have hxn := FloatV.isFinite_iff.mp hx
  have hyn := FloatV.isFinite_iff.mp hy
  have hs : Format.binary32.specials = .ieee := rfl
  unfold FloatV.fma Format.fma
  have h1 : Format.binary32.isNaN (FloatV.negInf : F32).pat = false := FloatV.negInf_isNaN hs
  have h2 : Format.binary32.isInf (FloatV.negInf : F32).pat = true := FloatV.negInf_isInf hs
  have h3 : Format.binary32.isNaN x.pat = false := hxn.1
  have h4 : Format.binary32.isInf x.pat = false := hxn.2
  have h5 : Format.binary32.isNaN y.pat = false := hyn.1
  have h6 : Format.binary32.isInf y.pat = false := hyn.2
  simp only [h3, h4, h5, h6, h1, h2, Bool.or_false, Bool.false_eq_true, ite_false, ite_true]
  unfold Format.infPat; rw [hs]
  have := FloatV.negInf_sign (f := Format.binary32); unfold FloatV.sign at this; rw [this]
  rfl


/-- A product of finite `float32` values below `2^13` in magnitude is finite and within one rounding
of the exact product. -/
private theorem mul_lane {x σ : F32} (hx : x.isFinite = true) (hσ : σ.isFinite = true)
    (hb : |x.toReal * σ.toReal| ≤ 2 ^ 13) :
    (FloatV.mul x σ).isFinite = true := by
  refine (FloatV.mul_err_real Format.binary32_regular hx hσ ?_).1
  rw [Format.binary32_maxVal_real]
  exact hb.trans (by norm_num)

/-- The facts every lane of the score shares: the dot is finite, within the dot bound of the exact
`Σ q·k`, the scale is finite within its rounding bound, and the exact scale is positive. -/
private theorem lane_facts {P : Platform} (hP : Device P) {fusion : Bool} {BD : ℕ} (hBD : BD ≤ 256)
    {q : Tensor [128, BD] BF16} {k : Tensor [BD, 128] BF16} {qk₀ z qk₁ : Tensor [128, 128] F32}
    (hr : P.dot cfgBF16 q k qk₀ qk₁ ∨
      ∃ d', P.dot cfgBF16 q k z d' ∧ ∀ i, AccAdd fusion .f32 (d'.get i) (qk₀.get i) (qk₁.get i))
    (r c : Fin 128) (hz : DType.val? .f32 (z.get2 r c) = some 0)
    (h0 : DType.val? .f32 (qk₀.get2 r c) = some 0) {Q K : Fin BD → ℚ}
    (hq : ∀ e, DType.val? .bf16 (q.get2 r e) = some (Q e))
    (hk : ∀ e, DType.val? .bf16 (k.get2 e c) = some (K e)) {sg X : ℝ}
    (hs0 : (2 : ℝ) ^ (-64 : ℤ) ≤ sg)
    (hX : sg * ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) ≤ X)
    (hX1 : X ≤ 2 ^ 12) :
    (qk₁.get2 r c).isFinite = true ∧
      |(qk₁.get2 r c).toReal - ∑ e, ((Q e : ℚ) : ℝ) * ((K e : ℚ) : ℝ)| ≤
        ((BD : ℝ) + 3) * 2 ^ (-22 : ℤ) *
            ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) +
          2 ^ (-110 : ℤ) := by
  refine dot_lane hP hBD hr r c hz h0 hq hk ?_
  have hL0 : 0 ≤ ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) :=
    Finset.sum_nonneg fun e _ => by
      exact_mod_cast mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have h1 : (2 : ℝ) ^ (-64 : ℤ) * ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ)
      ≤ 2 ^ 12 := (mul_le_mul_of_nonneg_right hs0 hL0).trans (hX.trans hX1)
  have h2 : ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) =
      2 ^ 64 * (2 ^ (-64 : ℤ) * ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ)) := by
    rw [← mul_assoc]; norm_num
  rw [h2]
  calc (2 : ℝ) ^ 64 * (2 ^ (-64 : ℤ) * ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ))
      ≤ 2 ^ 64 * 2 ^ 12 := mul_le_mul_of_nonneg_left h1 (by norm_num)
    _ ≤ 2 ^ 120 := by norm_num

/-- The exact dot is at most its lifted magnitude. -/
private theorem abs_dot_le {BD : ℕ} (Q K : Fin BD → ℚ) :
    |∑ e, ((Q e : ℚ) : ℝ) * ((K e : ℚ) : ℝ)| ≤
      ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) := by
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e _ => ?_)
  rw [← Rat.cast_mul, ← Rat.cast_abs]
  exact_mod_cast Device.abs_mul_le_liftMag _ _ _ _

/-- **An attended lane of the score** (`qk = qk₁·sm_scale + 0`, lines 131-145): with `sm_scale` the
host's `float32` of a double `c` in `[2^-64, 2^64]` and `X` a bound on the scale times the lifted
magnitude of the dot (`X ≤ 2^12`), the lane `y` is finite and within
`(BD + 8)·2^-22·X + 2^-40` of the exact `c·Σ Q·K`. -/
theorem score_lane {P : Platform} (hP : Device P) {fusion : Bool} {BD : ℕ} (hBD : BD ≤ 256)
    {q : Tensor [128, BD] BF16} {k : Tensor [BD, 128] BF16} {qk₀ z qk₁ : Tensor [128, 128] F32}
    (hr : P.dot cfgBF16 q k qk₀ qk₁ ∨
      ∃ d', P.dot cfgBF16 q k z d' ∧ ∀ i, AccAdd fusion .f32 (d'.get i) (qk₀.get i) (qk₁.get i))
    (r c : Fin 128) (hz : DType.val? .f32 (z.get2 r c) = some 0)
    (h0 : DType.val? .f32 (qk₀.get2 r c) = some 0) {Q K : Fin BD → ℚ}
    (hq : ∀ e, DType.val? .bf16 (q.get2 r e) = some (Q e))
    (hk : ∀ e, DType.val? .bf16 (k.get2 e c) = some (K e)) {sm : F64}
    (hsm : sm.isFinite = true) (hlo : (1 / 2 ^ 64 : ℚ) ≤ sm.toRat) (hhi : sm.toRat ≤ 2 ^ 64) {X : ℝ}
    (hX : (sm.toRat : ℝ) *
      ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) ≤ X)
    (hX1 : X ≤ 2 ^ 12) {y : F32}
    (hy : FloatV.NaNEq (FloatV.add (FloatV.mul (qk₁.get2 r c) (FloatV.hostF32 sm)) zf32) y ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (qk₁.get2 r c) (FloatV.hostF32 sm) zf32) y)) :
    y.isFinite = true ∧
      |y.toReal - (sm.toRat : ℝ) * ∑ e, ((Q e : ℚ) : ℝ) * ((K e : ℚ) : ℝ)| ≤
        ((BD : ℝ) + 8) * 2 ^ (-22 : ℤ) * X + 2 ^ (-40 : ℤ) := by
  have hlo' : (2 : ℝ) ^ (-64 : ℤ) ≤ (sm.toRat : ℝ) := by
    have : ((1 / 2 ^ 64 : ℚ) : ℝ) ≤ (sm.toRat : ℝ) := by exact_mod_cast hlo
    refine le_trans (le_of_eq ?_) this
    norm_num
  have hhi' : (sm.toRat : ℝ) ≤ 2 ^ 64 := by exact_mod_cast hhi
  have hsmr : sm.toReal = (sm.toRat : ℝ) := rfl
  obtain ⟨hσf, hσe⟩ := hostF32_err hsm (hsmr ▸ hlo') (hsmr ▸ hhi')
  rw [hsmr] at hσe
  obtain ⟨hxf, hxe⟩ := lane_facts hP hBD hr r c hz h0 hq hk hlo' hX hX1
  have hL0 : 0 ≤ ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) :=
    Finset.sum_nonneg fun e _ => by
      exact_mod_cast mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hpb : |(qk₁.get2 r c).toReal * (FloatV.hostF32 sm).toReal| ≤ 2 ^ 13 :=
    (score_arith (N := BD) (by positivity) (by exact_mod_cast hBD) hlo' hhi' hL0 hX hX1
      (abs_dot_le Q K) hxe hσe (y := (qk₁.get2 r c).toReal * (FloatV.hostF32 sm).toReal)
      (by rw [sub_self, abs_zero]; positivity)).1
  have hM : |(qk₁.get2 r c).toRat * (FloatV.hostF32 sm).toRat| + |zf32.toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h : |(qk₁.get2 r c).toRat * (FloatV.hostF32 sm).toRat| ≤ 2 ^ 13 := by
      have : ((|(qk₁.get2 r c).toRat * (FloatV.hostF32 sm).toRat| : ℚ) : ℝ) ≤ ((2 ^ 13 : ℚ) : ℝ) := by
        push_cast
        exact hpb
      exact_mod_cast this
    rw [zf32_val.2, abs_zero, Format.binary32_maxVal, Format.binary32_eta]
    have : (2 : ℚ) ^ 13 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * 2 ^ (-150 : ℤ) := by norm_num
    linarith
  obtain ⟨hyf, hye⟩ := FloatV.mulAdd_err (f := Format.binary32) Format.binary32_regular
    (by rw [Format.binary32_u]; norm_num) hy hxf hσf zf32_val.1 hM
  refine ⟨hyf, ?_⟩
  have hye' : |y.toReal - (qk₁.get2 r c).toReal * (FloatV.hostF32 sm).toReal| ≤
      3 * 2 ^ (-24 : ℤ) * |(qk₁.get2 r c).toReal * (FloatV.hostF32 sm).toReal| +
        4 * 2 ^ (-150 : ℤ) := by
    have h := (Rat.cast_le (K := ℝ)).mpr hye
    rw [zf32_val.2, Format.binary32_u, Format.binary32_eta] at h
    unfold FloatV.toReal
    push_cast at h ⊢
    simpa using h
  exact (score_arith (N := BD) (by positivity) (by exact_mod_cast hBD) hlo' hhi' hL0 hX hX1
    (abs_dot_le Q K) hxe hσe hye').2

/-- **A masked lane of the score** (`qk = qk₁·sm_scale + (-∞)`): when the scale times the lifted
magnitude of the dot is at most `2^12`, the lane `y` is `-∞`. -/
theorem score_lane_masked {P : Platform} (hP : Device P) {fusion : Bool} {BD : ℕ} (hBD : BD ≤ 256)
    {q : Tensor [128, BD] BF16} {k : Tensor [BD, 128] BF16} {qk₀ z qk₁ : Tensor [128, 128] F32}
    (hr : P.dot cfgBF16 q k qk₀ qk₁ ∨
      ∃ d', P.dot cfgBF16 q k z d' ∧ ∀ i, AccAdd fusion .f32 (d'.get i) (qk₀.get i) (qk₁.get i))
    (r c : Fin 128) (hz : DType.val? .f32 (z.get2 r c) = some 0)
    (h0 : DType.val? .f32 (qk₀.get2 r c) = some 0) {Q K : Fin BD → ℚ}
    (hq : ∀ e, DType.val? .bf16 (q.get2 r e) = some (Q e))
    (hk : ∀ e, DType.val? .bf16 (k.get2 e c) = some (K e)) {sm : F64}
    (hsm : sm.isFinite = true) (hlo : (1 / 2 ^ 64 : ℚ) ≤ sm.toRat) (hhi : sm.toRat ≤ 2 ^ 64)
    (hX : (sm.toRat : ℝ) *
      ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) ≤ 2 ^ 12) {y : F32}
    (hy : FloatV.NaNEq (FloatV.add (FloatV.mul (qk₁.get2 r c) (FloatV.hostF32 sm)) ninf32) y ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (qk₁.get2 r c) (FloatV.hostF32 sm) ninf32) y)) :
    y = FloatV.negInf := by
  have hlo' : (2 : ℝ) ^ (-64 : ℤ) ≤ (sm.toRat : ℝ) := by
    have : ((1 / 2 ^ 64 : ℚ) : ℝ) ≤ (sm.toRat : ℝ) := by exact_mod_cast hlo
    refine le_trans (le_of_eq ?_) this
    norm_num
  have hhi' : (sm.toRat : ℝ) ≤ 2 ^ 64 := by exact_mod_cast hhi
  have hsmr : sm.toReal = (sm.toRat : ℝ) := rfl
  obtain ⟨hσf, hσe⟩ := hostF32_err hsm (hsmr ▸ hlo') (hsmr ▸ hhi')
  rw [hsmr] at hσe
  obtain ⟨hxf, hxe⟩ := lane_facts hP hBD hr r c hz h0 hq hk hlo' hX le_rfl
  have hL0 : 0 ≤ ∑ e, ((Device.liftMag .bf16 (Q e) * Device.liftMag .bf16 (K e) : ℚ) : ℝ) :=
    Finset.sum_nonneg fun e _ => by
      exact_mod_cast mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hpb : |(qk₁.get2 r c).toReal * (FloatV.hostF32 sm).toReal| ≤ 2 ^ 13 :=
    (score_arith (N := BD) (by positivity) (by exact_mod_cast hBD) hlo' hhi' hL0 hX le_rfl
      (abs_dot_le Q K) hxe hσe (y := (qk₁.get2 r c).toReal * (FloatV.hostF32 sm).toReal)
      (by rw [sub_self, abs_zero]; positivity)).1
  have hmf := mul_lane hxf hσf hpb
  have hs : Format.binary32.specials = .ieee := rfl
  have hnn : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN hs
  rw [ninf32_eq] at hy
  rcases hy with h | ⟨-, h⟩
  · rw [add_negInf hmf] at h
    exact h.eq hnn
  · rw [fma_negInf hxf hσf] at h
    exact h.eq hnn

/-- **The final conversion** `o16 = tl.cvt(acc)` (line 183, to `bfloat16`, rounding to nearest even):
a finite `float32` `a` with `|a| ≤ 2^121` converts to a finite `bfloat16` `c` within
`2^-8·|a| + 2^-134` of `a`. -/
theorem cvt_store {a : F32} {c : BF16} (ha : a.isFinite = true) (hb : |a.toReal| ≤ 2 ^ 121)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn a) c) :
    c.isFinite = true ∧ |c.toReal - a.toReal| ≤ 2 ^ (-8 : ℤ) * |a.toReal| + 2 ^ (-134 : ℤ) := by
  have hmax : |a.toReal| ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Format.bfloat16_maxVal_real]
    exact hb.trans (by norm_num)
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular ha hmax
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h2
  refine ⟨hf, ?_⟩
  rw [hr]
  linarith

end Claims.PrefillAttnVarlen

end

section
/-!
# One trip of the loop: the float facts of the column side

The device facts for the weights' roundings, the scale `p_scale = beta / l_i_new`, the rescaling
factor, the `bfloat16` dot into the accumulator and the sizes of the running sums, as inequalities
over `ℝ` (`Proof/Trip.lean` assembles them into one trip).
-/

open TVA

namespace Claims.PrefillAttnVarlen

namespace Trip

/-- `x * y` on `float32` (up to NaN payload): finite and within one rounding. -/
theorem f32_mul {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal * y.toReal| ≤ 2 ^ 127) (h : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧
      |r.toReal - x.toReal * y.toReal| ≤ 2 ^ (-24 : ℤ) * |x.toReal * y.toReal| + 2 ^ (-150 : ℤ) := by
  have hm : (2 : ℝ) ^ 127 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.mul_err_real Format.binary32_regular hx hy (hb.trans hm)
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  refine ⟨hf, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  rw [hr]; linarith

/-- `x.to(bfloat16)` of a finite `float32` (up to NaN payload): finite and within one rounding. -/
theorem f32_cvt_bf16 {x : F32} {r : BF16} (hx : x.isFinite = true) (hb : |x.toReal| ≤ 2 ^ 100)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ 2 ^ (-8 : ℤ) * |x.toReal| + 2 ^ (-134 : ℤ) := by
  have hm : (2 : ℝ) ^ 100 ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Format.bfloat16_maxVal]; norm_num
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hx (hb.trans hm)
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  refine ⟨hf, ?_⟩
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h2
  rw [hr]; linarith

/-- `x / y` on `float32` (the platform's `div.full`): finite and within `2^-22·|x/y| + 2^-148`
for a positive divisor. -/
theorem f32_div {P : Platform} (hP : Device P) {a l q : F32} (ha : a.isFinite = true)
    (hl : l.isFinite = true) (hl0 : 0 < l.toReal) (hq : |a.toReal / l.toReal| ≤ 2 ^ 100)
    (hdiv : P.div .binary32 a l q) :
    q.isFinite = true ∧
      |q.toReal - a.toReal / l.toReal| ≤
        2 ^ (-22 : ℤ) * |a.toReal / l.toReal| + 2 ^ (-148 : ℤ) := by
  have hl0' : l.toRat ≠ 0 := by
    intro h; unfold FloatV.toReal at hl0; rw [h] at hl0; simp at hl0
  have hq' : |a.toRat / l.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat / l.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast
      unfold FloatV.toReal at hq
      have : (2 : ℝ) ^ 100 ≤ 2 ^ 126 := by norm_num
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := hP.divFull a l q hdiv ha hl hl0' hq'
  refine ⟨hf, ?_⟩
  have : ((|q.toRat - a.toRat / l.toRat| : ℚ) : ℝ) ≤
      (((2 : ℚ) ^ (-22 : ℤ) * |a.toRat / l.toRat| + (2 : ℚ) ^ (-148 : ℤ) : ℚ) : ℝ) := by
    exact_mod_cast he
  push_cast at this
  unfold FloatV.toReal
  exact this

/-- The `float32` `+0` has value `0`. -/
theorem zero32_rat : (FloatV.ofPat 0 : F32).toRat = 0 := by decide +kernel

/-- The `float32` `+0` is finite. -/
theorem zero32_fin : (FloatV.ofPat 0 : F32).isFinite = true := by decide +kernel

/-- The `float32` `+0` is `0` as a real. -/
theorem zero32_real : (FloatV.ofPat 0 : F32).toReal = 0 := by
  unfold FloatV.toReal; rw [zero32_rat]; simp

/-- A quotient `ps` of `x` (within `2^-22·|x| + 2^-148`) with `x` not below `-2^-139`: clamped at
`0` it is within `2^-22·max x 0 + 2^-148` of `max x 0`, and the clamp moves it by at most `2^-138`. -/
theorem clamp_quot {ps x : ℝ} (hpe : |ps - x| ≤ 2 ^ (-22 : ℤ) * |x| + 2 ^ (-148 : ℤ))
    (hlow : -(2 : ℝ) ^ (-139 : ℤ) ≤ x) :
    |Max.max ps 0 - Max.max x 0| ≤ 2 ^ (-22 : ℤ) * Max.max x 0 + 2 ^ (-148 : ℤ) ∧
      |ps - Max.max ps 0| ≤ 2 ^ (-138 : ℤ) := by
  have h22 : (0 : ℝ) < 2 ^ (-22 : ℤ) := by positivity
  have h22' : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
  have h148 : (2 : ℝ) ^ (-148 : ℤ) ≤ 2 ^ (-138 : ℤ) / 2 := by norm_num
  have h139 : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ (-139 : ℤ) ≤ 2 ^ (-148 : ℤ) := by norm_num
  have h139' : (2 : ℝ) ^ (-139 : ℤ) ≤ 2 ^ (-138 : ℤ) / 2 := by norm_num
  have h148' : (0 : ℝ) < 2 ^ (-148 : ℤ) := by positivity
  obtain ⟨hp1, hp2⟩ := abs_le.mp hpe
  rcases le_or_gt 0 x with hx | hx
  · rw [abs_of_nonneg hx] at hp1 hp2
    rw [max_eq_left hx]
    have hxa : 2 ^ (-22 : ℤ) * x ≤ x := by nlinarith
    refine ⟨?_, ?_⟩
    · rcases le_or_gt 0 ps with h | h
      · rw [max_eq_left h]; exact abs_le.mpr ⟨hp1, hp2⟩
      · rw [max_eq_right h.le]
        rw [abs_of_nonpos (by linarith)]
        nlinarith [mul_nonneg h22.le hx]
    · rcases le_or_gt 0 ps with h | h
      · rw [max_eq_left h]; simp
      · rw [max_eq_right h.le, sub_zero, abs_of_neg h]
        norm_num at hp1 hxa h148 ⊢
        linarith
  · rw [abs_of_neg hx] at hp1 hp2
    rw [max_eq_right hx.le]
    have hxb : 2 ^ (-22 : ℤ) * -x ≤ -x := by nlinarith
    have hxc : 2 ^ (-22 : ℤ) * -x ≤ 2 ^ (-22 : ℤ) * 2 ^ (-139 : ℤ) :=
      mul_le_mul_of_nonneg_left (by linarith) h22.le
    have hps : ps ≤ 2 ^ (-148 : ℤ) := by linarith
    refine ⟨?_, ?_⟩
    · rcases le_or_gt 0 ps with h | h
      · rw [max_eq_left h, sub_zero, abs_of_nonneg h]
        nlinarith [mul_nonneg h22.le (neg_nonneg.mpr hx.le)]
      · rw [max_eq_right h.le]; simp
    · rcases le_or_gt 0 ps with h | h
      · rw [max_eq_left h, sub_self, abs_zero]; positivity
      · rw [max_eq_right h.le, sub_zero, abs_of_neg h]
        norm_num at hp1 hxc h139 h139' h148 hlow ⊢
        linarith

/-- **The scale `p_scale = beta / l_i_new`.** With `β̃ = max β' 0` and `ps̃ = max ps 0` (the platform's
`exp` may give a tiny negative value where the exact one is tiny and positive): `ps` is finite, small,
`ps̃` is within `2^-22·(β̃/l') + 2^-148` of `β̃/l'`, and `ps̃` differs from `ps` by at most `2^-138`. -/
theorem ps_facts {P : Platform} (hP : Device P) {beta lnew psc : F32} {Y β : ℝ} (hY0 : 0 ≤ Y)
    (hbf : beta.isFinite = true) (hlf : lnew.isFinite = true) (hl1 : 1 / 2 ≤ lnew.toReal)
    (hbb : |beta.toReal| ≤ 3) (hβ0 : 0 ≤ β) (hrE : Num.rE Y ≤ 1 / 100)
    (hβ' : |beta.toReal - β| ≤ Num.rE Y * β + Num.ηE) (hdiv : P.div .binary32 beta lnew psc) :
    psc.isFinite = true ∧ |psc.toReal| ≤ 8 ∧
      |Max.max psc.toReal 0 - Max.max beta.toReal 0 / lnew.toReal| ≤
        2 ^ (-22 : ℤ) * (Max.max beta.toReal 0 / lnew.toReal) + 2 ^ (-148 : ℤ) ∧
      |psc.toReal - Max.max psc.toReal 0| ≤ 2 ^ (-138 : ℤ) := by
  have hl0 : 0 < lnew.toReal := by linarith
  have hx : |beta.toReal / lnew.toReal| ≤ 6 := by
    rw [abs_div, abs_of_pos hl0, div_le_iff₀ hl0]
    nlinarith
  obtain ⟨hpf, hpe⟩ := f32_div hP hbf hlf hl0 (hx.trans (by norm_num)) hdiv
  have hp8 : |psc.toReal| ≤ 8 := by
    have h1 := abs_sub_abs_le_abs_sub psc.toReal (beta.toReal / lnew.toReal)
    have h2 : (2 : ℝ) ^ (-22 : ℤ) * |beta.toReal / lnew.toReal| ≤ 2 ^ (-22 : ℤ) * 6 :=
      mul_le_mul_of_nonneg_left hx (by positivity)
    have h3 : (2 : ℝ) ^ (-22 : ℤ) * 6 + 2 ^ (-148 : ℤ) ≤ 1 := by norm_num
    linarith
  have hβlow : -(2 : ℝ) ^ (-140 : ℤ) ≤ beta.toReal := by
    have h1 := (abs_le.mp hβ').1
    have h2 : 0 ≤ Num.rE Y * β := mul_nonneg (by unfold Num.rE; positivity) hβ0
    have h3 : Num.rE Y * β ≤ 1 / 100 * β := mul_le_mul_of_nonneg_right hrE hβ0
    unfold Num.ηE at h1
    linarith
  have hlow : -(2 : ℝ) ^ (-139 : ℤ) ≤ beta.toReal / lnew.toReal := by
    rw [le_div_iff₀ hl0]
    have h2 : (2 : ℝ) ^ (-139 : ℤ) = 2 * 2 ^ (-140 : ℤ) := by norm_num
    have h3 : (0 : ℝ) < 2 ^ (-140 : ℤ) := by positivity
    rw [h2]
    nlinarith
  obtain ⟨hc1, hc2⟩ := clamp_quot hpe hlow
  have hmx : Max.max beta.toReal 0 / lnew.toReal = Max.max (beta.toReal / lnew.toReal) 0 := by
    rcases le_or_gt 0 beta.toReal with hb | hb
    · rw [max_eq_left hb, max_eq_left (div_nonneg hb hl0.le)]
    · rw [max_eq_right hb.le, max_eq_right (div_neg_of_neg_of_pos hb hl0).le, zero_div]
  rw [hmx]
  exact ⟨hpf, hp8, hc1, hc2⟩

/-- A product with a `+0` factor and a finite one is a finite zero. -/
theorem mul_zero_f32 {y r : F32} (hy : y.isFinite = true)
    (h : FloatV.NaNEq (FloatV.mul (FloatV.ofPat 0 : F32) y) r) :
    r.isFinite = true ∧ r.toRat = 0 := by
  have hx := zero32_fin
  have hm : (FloatV.mul (FloatV.ofPat 0 : F32) y).isFinite = true ∧
      (FloatV.mul (FloatV.ofPat 0 : F32) y).toRat = 0 := by
    have hfx : Format.binary32.isFinite (FloatV.ofPat 0 : F32).pat = true := hx
    have hfy : Format.binary32.isFinite y.pat = true := hy
    unfold FloatV.mul
    rw [Format.mul_of_finite .rn hfx hfy]
    have hz : Format.binary32.toRat (FloatV.ofPat 0 : F32).pat = 0 := zero32_rat
    rw [hz, zero_mul]
    obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32)
      Format.binary32_regular (rnd := .rn) (sat := false)
      (zs := (Format.binary32.sign (FloatV.ofPat 0 : F32).pat != Format.binary32.sign y.pat))
    exact ⟨(FloatV.isFinite_ofPat h1).trans h2, by rw [FloatV.toRat_ofPat h1]; exact h3⟩
  obtain ⟨hf, hr, -⟩ := h.exact_of_isFinite hm.1
  exact ⟨hf, hr.trans hm.2⟩

/-- Converting a finite `float32` zero to `bfloat16` gives a finite zero. -/
theorem cvt_zero_bf16 {x : F32} {r : BF16} (hx : x.isFinite = true) (h0 : x.toRat = 0)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) : r.isFinite = true ∧ r.toRat = 0 := by
  have hm : (FloatV.cvtWith .rn x : BF16).isFinite = true ∧ (FloatV.cvtWith .rn x : BF16).toRat = 0 := by
    have hfx : Format.binary32.isFinite x.pat = true := hx
    unfold FloatV.cvtWith
    rw [Format.cvt_of_finite Format.bfloat16 .rn false hfx]
    have hz : Format.binary32.toRat x.pat = 0 := h0
    rw [hz]
    obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.bfloat16)
      Format.bfloat16_regular (rnd := .rn) (sat := false) (zs := Format.binary32.sign x.pat)
    exact ⟨(FloatV.isFinite_ofPat h1).trans h2, by rw [FloatV.toRat_ofPat h1]; exact h3⟩
  obtain ⟨hf, hr, -⟩ := h.exact_of_isFinite hm.1
  exact ⟨hf, hr.trans hm.2⟩

/-- The two roundings of a weight lane (`p · ps` in `float32`, then `bfloat16`), against the exact
product with the clamped scale. -/
theorem round_chain {p ps ps' p1 p2 : ℝ} (hp : |p| ≤ 2) (hps : |ps - ps'| ≤ 2 ^ (-138 : ℤ))
    (h1 : |p1 - p * ps| ≤ 2 ^ (-24 : ℤ) * |p * ps| + 2 ^ (-150 : ℤ))
    (h2 : |p2 - p1| ≤ 2 ^ (-8 : ℤ) * |p1| + 2 ^ (-134 : ℤ)) :
    |p2 - p * ps'| ≤ (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ)) * |p * ps'| + 2 ^ (-130 : ℤ) := by
  have e1 : |p * ps - p * ps'| ≤ 2 * 2 ^ (-138 : ℤ) := by
    rw [← mul_sub, abs_mul]
    exact mul_le_mul hp hps (abs_nonneg _) (by norm_num)
  have e2 : |p * ps| ≤ |p * ps'| + 2 * 2 ^ (-138 : ℤ) := by
    have := abs_sub_abs_le_abs_sub (p * ps) (p * ps')
    linarith
  have e3 : |p1| ≤ |p * ps| + (2 ^ (-24 : ℤ) * |p * ps| + 2 ^ (-150 : ℤ)) := by
    have := abs_sub_abs_le_abs_sub p1 (p * ps)
    linarith
  have e4 := abs_sub_le p2 p1 (p * ps')
  have e5 : |p1 - p * ps'| ≤ |p1 - p * ps| + |p * ps - p * ps'| := abs_sub_le p1 (p * ps) (p * ps')
  have hW := abs_nonneg (p * ps')
  generalize |p * ps'| = W at *
  generalize |p * ps| = Q at *
  generalize |p1| = P1 at *
  norm_num at *
  linarith

/-- **A weight lane** `p2 = bf16(p · ps)`: finite, and within the rounding bounds of `p · ps̃`. -/
theorem p2_lane {pc psc p1c : F32} {p2c : BF16} (hpf : pc.isFinite = true)
    (hsf : psc.isFinite = true) (hp : |pc.toReal| ≤ 2) (hs : |psc.toReal| ≤ 8)
    (hs' : |psc.toReal - Max.max psc.toReal 0| ≤ 2 ^ (-138 : ℤ))
    (hm : FloatV.NaNEq (FloatV.mul pc psc) p1c)
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn p1c) p2c) :
    p2c.isFinite = true ∧
      |p2c.toReal - pc.toReal * Max.max psc.toReal 0| ≤
        (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ)) * |pc.toReal * Max.max psc.toReal 0| + 2 ^ (-130 : ℤ) := by
  have hb : |pc.toReal * psc.toReal| ≤ 16 := by
    rw [abs_mul]; nlinarith [abs_nonneg pc.toReal, abs_nonneg psc.toReal]
  obtain ⟨hf1, he1⟩ := f32_mul hpf hsf (hb.trans (by norm_num)) hm
  have hp1 : |p1c.toReal| ≤ 2 ^ 100 := by
    have h1 := abs_sub_abs_le_abs_sub p1c.toReal (pc.toReal * psc.toReal)
    have h2 : (2 : ℝ) ^ (-24 : ℤ) * |pc.toReal * psc.toReal| ≤ 2 ^ (-24 : ℤ) * 16 :=
      mul_le_mul_of_nonneg_left hb (by positivity)
    have h3 : (2 : ℝ) ^ (-24 : ℤ) * 16 + 2 ^ (-150 : ℤ) + 16 ≤ 2 ^ 100 := by norm_num
    linarith
  obtain ⟨hf2, he2⟩ := f32_cvt_bf16 hf1 hp1 hc
  exact ⟨hf2, round_chain hp hs' he1 he2⟩

/-- **The rescaling factor** `S = (l / l') · α'` (a quotient, then a product), against the exact
product: within `(2^-22 + 2^-23)·x + 2^-140`, `x = (l/l')·α'`. -/
theorem scale_real {l l' α' q S : ℝ} (hl0 : 0 ≤ l) (hl2 : l ≤ 2 ^ 18) (hl' : 1 / 2 ≤ l')
    (hα1 : |α'| ≤ 3) (hα2 : -(2 : ℝ) ^ (-140 : ℤ) ≤ α')
    (hq : |q - l / l'| ≤ 2 ^ (-22 : ℤ) * |l / l'| + 2 ^ (-148 : ℤ))
    (hS : |S - q * α'| ≤ 2 ^ (-24 : ℤ) * |q * α'| + 2 ^ (-150 : ℤ)) :
    |S - (l / l') * α'| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ)) * ((l / l') * α') + 2 ^ (-140 : ℤ) := by
  have hl0' : 0 < l' := by linarith
  have hx0 : 0 ≤ l / l' := div_nonneg hl0 hl0'.le
  have hx1 : l / l' ≤ 2 ^ 19 := by
    rw [div_le_iff₀ hl0']
    nlinarith
  set x := l / l' with hx
  rw [abs_of_nonneg hx0] at hq
  have f1 : |q - x| * |α'| ≤ (2 ^ (-22 : ℤ) * x + 2 ^ (-148 : ℤ)) * |α'| :=
    mul_le_mul_of_nonneg_right hq (abs_nonneg _)
  have hqx : |q| ≤ x + (2 ^ (-22 : ℤ) * x + 2 ^ (-148 : ℤ)) := by
    have := abs_sub_abs_le_abs_sub q x
    rw [abs_of_nonneg hx0] at this
    linarith
  have f2 : |q| * |α'| ≤ (x + (2 ^ (-22 : ℤ) * x + 2 ^ (-148 : ℤ))) * |α'| :=
    mul_le_mul_of_nonneg_right hqx (abs_nonneg _)
  have e1 : |q * α' - x * α'| = |q - x| * |α'| := by rw [← sub_mul, abs_mul]
  have e2 : |q * α'| = |q| * |α'| := abs_mul _ _
  have e3 := abs_sub_le S (q * α') (x * α')
  rw [e1] at e3
  rw [e2] at hS
  have hxa : 0 ≤ x * |α'| := mul_nonneg hx0 (abs_nonneg _)
  have hxα : |α'| ≤ 3 := hα1
  have hab : |α'| * 2 ^ (-148 : ℤ) ≤ 3 * 2 ^ (-148 : ℤ) :=
    mul_le_mul_of_nonneg_right hα1 (by positivity)
  have hx2 : x * (2 ^ (-22 : ℤ) * |α'|) = 2 ^ (-22 : ℤ) * (x * |α'|) := by ring
  rcases le_or_gt 0 α' with ha | ha
  · rw [abs_of_nonneg ha] at f1 f2 hab e3 hS
    have hxn : 0 ≤ x * α' := mul_nonneg hx0 ha
    norm_num at f1 f2 hab hS e3 hxn ⊢
    nlinarith
  · have ha' : |α'| ≤ 2 ^ (-140 : ℤ) := by
      rw [abs_of_neg ha]; linarith
    have ha0 : 0 ≤ |α'| := abs_nonneg _
    have g1 : (2 ^ (-22 : ℤ) * x + 2 ^ (-148 : ℤ)) * |α'| ≤
        (2 ^ (-22 : ℤ) * 2 ^ 19 + 2 ^ (-148 : ℤ)) * 2 ^ (-140 : ℤ) :=
      mul_le_mul (by
        have := mul_le_mul_of_nonneg_left hx1 (by positivity : (0 : ℝ) ≤ 2 ^ (-22 : ℤ))
        linarith) ha' ha0 (by positivity)
    have g2 : (x + (2 ^ (-22 : ℤ) * x + 2 ^ (-148 : ℤ))) * |α'| ≤
        (2 ^ 19 + (2 ^ (-22 : ℤ) * 2 ^ 19 + 2 ^ (-148 : ℤ))) * 2 ^ (-140 : ℤ) :=
      mul_le_mul (by
        have := mul_le_mul_of_nonneg_left hx1 (by positivity : (0 : ℝ) ≤ 2 ^ (-22 : ℤ))
        linarith) ha' ha0 (by positivity)
    have g3 : x * |α'| ≤ 2 ^ 19 * 2 ^ (-140 : ℤ) := mul_le_mul hx1 ha' ha0 (by positivity)
    rw [abs_of_neg ha] at g1 g2 g3 f1 f2 e3 hS
    norm_num at g1 g2 g3 f1 f2 hS e3 ⊢
    nlinarith

/-- `exp x ≤ 1 + 2x` on `[0, 1/2]`. -/
theorem exp_le_lin' {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1 / 2) : Real.exp x ≤ 1 + 2 * x := by
  rcases h0.eq_or_lt with h | h
  · subst h; simp
  · have h3 := Real.exp_bound_div_one_sub_of_interval' h (by linarith)
    have h2 : 1 / (1 - x) ≤ 1 + 2 * x := by
      rw [div_le_iff₀ (by linarith)]; nlinarith
    linarith

open Finset in
/-- **The sum `l` is within `12%` of the exact sum**, up to `2^-80` (a row that has run). -/
theorem l_rel' {L t : ℕ} {sc sh : ℕ → ℝ} {δ Y m l : ℝ} (inv : Num.RowInv L t sc sh δ Y m l)
    (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000) (ht0 : 0 < t)
    (ht : t ≤ 1024) :
    22 / 25 * Num.Lam sh (Num.keysN L t) m - 2 ^ (-80 : ℤ) ≤ l ∧
      l ≤ 28 / 25 * Num.Lam sh (Num.keysN L t) m + 2 ^ (-80 : ℤ) := by
  obtain ⟨-, -, hm1, hm2⟩ := inv.bounds hδ hY hY5 ht0 ht
  have hY0 : 0 ≤ Y := by linarith
  have hmax : max (m + Y) 0 ≤ 10000 := max_le (by linarith) (by norm_num)
  have h0 : 0 ≤ max (m + Y) 0 := le_max_right _ _
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have ht0' : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  have hw0 : 0 ≤ Num.omL Y t m := by
    unfold Num.omL Num.cL0 Num.cL1; positivity
  have hw1 : Num.omL Y t m ≤ 1 / 20 := by
    unfold Num.omL Num.cL0 Num.cL1; norm_num; nlinarith
  have hB0 : 0 ≤ Num.bL Y := by unfold Num.bL; positivity
  have hB1 : Num.bL Y ≤ 1 / 70 := by unfold Num.bL; norm_num; linarith
  have g1 : Real.exp (Num.omL Y t m) ≤ 11 / 10 := by
    have := exp_le_lin' hw0 (by linarith)
    linarith
  have g2 : Real.exp (Num.omL Y t m) * (1 + Num.bL Y) - 1 ≤ 3 / 25 := by
    have := exp_le_lin' hw0 (by linarith)
    nlinarith [mul_nonneg (sub_nonneg.mpr g1) (show (0 : ℝ) ≤ 1 + Num.bL Y by linarith)]
  have hη : (0 : ℝ) ≤ Num.ηL := by unfold Num.ηL; positivity
  have tiny : Real.exp (Num.omL Y t m) * (t * Num.ηL) ≤ 2 ^ (-80 : ℤ) := by
    calc Real.exp (Num.omL Y t m) * (t * Num.ηL) ≤ 11 / 10 * (1024 * Num.ηL) :=
          mul_le_mul g1 (mul_le_mul_of_nonneg_right ht' hη) (by positivity) (by norm_num)
      _ ≤ 2 ^ (-80 : ℤ) := by unfold Num.ηL; norm_num
  have hLam : 0 ≤ Num.Lam sh (Num.keysN L t) m :=
    sum_nonneg fun j _ => (Real.exp_pos _).le
  have cl := inv.l_close
  unfold Num.Close at cl
  have hG := mul_le_mul_of_nonneg_right g2 hLam
  obtain ⟨c1, c2⟩ := abs_le.mp cl
  generalize (2 : ℝ) ^ (-80 : ℤ) = e at tiny ⊢
  constructor <;> linarith

/-- **The rescaling is at most `3`**: `l·α' ≤ 3·l'` when `l`, `l'` are within `12%` of the exact
sums `Λ`, `Λ' ≥ α·Λ` and `α'` is within `1%` of `α` (up to `2^-140`). -/
theorem scale_le {l l' Lam Lam' α α' : ℝ} (hl0 : 0 ≤ l) (hl2 : l ≤ 2 ^ 18) (hl' : 1 / 2 ≤ l')
    (hA : l ≤ 28 / 25 * Lam + 2 ^ (-80 : ℤ))
    (hA' : 22 / 25 * Lam - 2 ^ (-80 : ℤ) ≤ l) (hB : 22 / 25 * Lam' - 2 ^ (-80 : ℤ) ≤ l')
    (hD : α * Lam ≤ Lam') (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hα' : α' ≤ 101 / 100 * α + 2 ^ (-140 : ℤ)) : l * α' ≤ 3 * l' := by
  have hη : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) := by positivity
  have h1 : l * α' ≤ l * (101 / 100 * α + 2 ^ (-140 : ℤ)) := mul_le_mul_of_nonneg_left hα' hl0
  have h2 : l * (101 / 100 * α + 2 ^ (-140 : ℤ)) ≤
      (28 / 25 * Lam + 2 ^ (-80 : ℤ)) * (101 / 100 * α + 2 ^ (-140 : ℤ)) :=
    mul_le_mul_of_nonneg_right hA (by positivity)
  have hLam2 : Lam ≤ 2 ^ 19 := by
    have : (2 : ℝ) ^ (-80 : ℤ) ≤ 1 := by norm_num
    nlinarith
  norm_num at h2 hA hA' hB hLam2
  rcases le_or_gt (α * Lam) (13 / 10) with hc | hc
  · nlinarith
  · nlinarith

/-- Lane `c` of row `r` of a rank-2 tensor (`default` beyond the row): the lane function the
reductions and proof rules use. -/
def lane {f : Format} {n : ℕ} (x : Tensor [128, n] (FloatV f)) (r : Fin 128) (c : ℕ) : FloatV f :=
  if h : c < n then x.get2 r ⟨c, h⟩ else default

/-- Inside the row, the lane function is the tensor's element. -/
theorem lane_of_lt {f : Format} {n : ℕ} (x : Tensor [128, n] (FloatV f)) (r : Fin 128) (c : Fin n) :
    lane x r c.val = x.get2 r c := by
  simp [lane]

/-- The arithmetic of one dot trip: the step's relative and absolute errors against the lifted
magnitudes, bounded by the claim's `rD`, `adc0`, `adc1` against the plain magnitudes. -/
theorem dot_alg {C A B SL sr sa : ℝ} (hC : 0 ≤ C) (hA : 0 ≤ A) (hB : 0 ≤ B)
    (hSL : SL ≤ A + 2 ^ (-124 : ℤ) + 2 ^ (-126 : ℤ) * B) (hsr0 : 0 ≤ sr) (hsr : sr ≤ Num.rD)
    (hsa : sa ≤ 3 * 2 ^ (-120 : ℤ)) :
    sr * (C + SL) + sa ≤ Num.rD * (C + A) + (Num.adc0 + Num.adc1 * B) := by
  have h1 : sr * (C + SL) ≤ sr * (C + (A + 2 ^ (-124 : ℤ) + 2 ^ (-126 : ℤ) * B)) :=
    mul_le_mul_of_nonneg_left (by linarith) hsr0
  have h2 : sr * (C + A) ≤ Num.rD * (C + A) := mul_le_mul_of_nonneg_right hsr (by linarith)
  have h3 : sr * 2 ^ (-124 : ℤ) ≤ Num.rD * 2 ^ (-124 : ℤ) :=
    mul_le_mul_of_nonneg_right hsr (by positivity)
  have h4 : sr * (2 ^ (-126 : ℤ) * B) ≤ Num.rD * (2 ^ (-126 : ℤ) * B) :=
    mul_le_mul_of_nonneg_right hsr (by positivity)
  have e : sr * (C + (A + 2 ^ (-124 : ℤ) + 2 ^ (-126 : ℤ) * B)) =
      sr * (C + A) + sr * 2 ^ (-124 : ℤ) + sr * (2 ^ (-126 : ℤ) * B) := by ring
  have h5 : Num.rD * 2 ^ (-124 : ℤ) + 3 * 2 ^ (-120 : ℤ) ≤ Num.adc0 := by
    unfold Num.rD Num.adc0; norm_num
  have h6 : Num.rD * (2 ^ (-126 : ℤ) * B) ≤ Num.adc1 * B := by
    have : Num.rD * 2 ^ (-126 : ℤ) ≤ Num.adc1 := by unfold Num.rD Num.adc1; norm_num
    nlinarith
  linarith

/-- A sum over the attended lanes of a function that vanishes on the others is the sum over all
`128` lanes. -/
theorem blk_sum_fin {L t : ℕ} {g : ℕ → ℝ} (h0 : ∀ k, k < 128 → k ∉ Num.blk L t → g k = 0) :
    ∑ k ∈ Num.blk L t, g k = ∑ k : Fin 128, g k.val := by
  rw [Fin.sum_univ_eq_sum_range g 128]
  refine Finset.sum_subset (fun k hk => ?_) (fun k hk hb => h0 k (Finset.mem_range.mp hk) hb)
  simp only [Num.blk, Finset.mem_filter] at hk
  exact hk.1

/-- A block has at most `128` attended lanes. -/
theorem blk_card_le (L t : ℕ) : (Num.blk L t).card ≤ 128 := by
  unfold Num.blk
  exact (Finset.card_filter_le _ _).trans (by simp)

/-- **One dot trip** `acc2 = acc1 + p2 @ v` at output element `(i, n)`: finite, and within the
claim's `rD`, `adc0`, `adc1` of the sum over the attended lanes (the weights vanish elsewhere). -/
theorem dot_ok {P : Platform} (hP : Device P) {fusion : Bool} {BD L t : ℕ} {vv : ℕ → ℝ}
    {a : Tensor [128, 128] BF16} {b : Tensor [128, BD] BF16}
    {c z r : Tensor [128, BD] F32} (i : Fin 128) (n : Fin BD)
    (hr : P.dot cfgBF16 a b c r ∨ ∃ d, P.dot cfgBF16 a b z d ∧
      ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (hz : z.get2 i n = FloatV.ofPat 0)
    (hafin : ∀ k : Fin 128, (a.get2 i k).isFinite = true)
    (hbfin : ∀ k : Fin 128, (b.get2 k n).isFinite = true)
    (haz : ∀ k : Fin 128, ¬ (128 * t + k.val ≤ L) → (a.get2 i k).toRat = 0)
    (hbv : ∀ k : Fin 128, 128 * t + k.val ≤ L → (b.get2 k n).toReal = vv (128 * t + k.val))
    (hvv : ∀ j, j ≤ L → |vv j| ≤ 2 ^ 120)
    (hcfin : (c.get2 i n).isFinite = true) (hc : |(c.get2 i n).toReal| ≤ 2 ^ 124)
    (hsum : ∑ k ∈ Num.blk L t, |(lane a i k).toReal| ≤ 2) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toReal - ((c.get2 i n).toReal +
          ∑ k ∈ Num.blk L t, (lane a i k).toReal * vv (128 * t + k))| ≤
        Num.rD * (|(c.get2 i n).toReal| +
          ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)|) +
        (Num.adc0 + Num.adc1 * ∑ k ∈ Num.blk L t, |vv (128 * t + k)|) := by
  classical
  have hblk : ∀ k, k ∈ Num.blk L t ↔ k < 128 ∧ 128 * t + k ≤ L := fun k => by simp [Num.blk]
  set ga : ℕ → ℚ := fun k => (lane a i k).toRat with hga
  set gb : ℕ → ℚ := fun k => if h : k < 128 then (b.get2 ⟨k, h⟩ n).toRat else 0 with hgb
  have hga' : ∀ k : Fin 128, ga k.val = (a.get2 i k).toRat := fun k => by
    simp [hga, lane_of_lt]
  have hgb' : ∀ k : Fin 128, gb k.val = (b.get2 k n).toRat := fun k => by simp [hgb]
  have hva : ∀ k : Fin 128, DType.val? .bf16 (a.get2 i k) = some (ga k.val) := fun k =>
    DType.val?_bf16.mpr ⟨hafin k, (hga' k).symm⟩
  have hvb : ∀ k : Fin 128, DType.val? .bf16 (b.get2 k n) = some (gb k.val) := fun k =>
    DType.val?_bf16.mpr ⟨hbfin k, (hgb' k).symm⟩
  have hz0 : DType.val? .f32 (z.get2 i n) = some 0 :=
    DType.val?_f32.mpr ⟨by rw [hz]; exact zero32_fin, by rw [hz]; exact zero32_rat⟩
  have hga0 : ∀ k, k < 128 → k ∉ Num.blk L t → ga k = 0 := by
    intro k hk hb
    have hn : ¬ (128 * t + k ≤ L) := fun h => hb ((hblk k).mpr ⟨hk, h⟩)
    have h2 : lane a i k = a.get2 i ⟨k, hk⟩ := lane_of_lt a i ⟨k, hk⟩
    show (lane a i k).toRat = 0
    rw [h2]
    exact haz ⟨k, hk⟩ hn
  have hgbv : ∀ k ∈ Num.blk L t, (gb k : ℝ) = vv (128 * t + k) := by
    intro k hk
    obtain ⟨hk1, hk2⟩ := (hblk k).mp hk
    have h1 : (gb k : ℝ) = (b.get2 ⟨k, hk1⟩ n).toReal := by simp [hgb, hk1, FloatV.toReal_eq]
    exact h1.trans (hbv ⟨k, hk1⟩ hk2)
  have hgav : ∀ k, (ga k : ℝ) = (lane a i k).toReal := fun k => rfl
  have hgbabs : ∀ k ∈ Num.blk L t, |(gb k : ℝ)| ≤ 2 ^ 120 := by
    intro k hk
    rw [hgbv k hk]
    exact hvv _ (by have := ((hblk k).mp hk).2; omega)
  -- the sums
  have hS1 : (∑ k : Fin 128, (ga k.val : ℝ) * (gb k.val : ℝ)) =
      ∑ k ∈ Num.blk L t, (lane a i k).toReal * vv (128 * t + k) := by
    refine (blk_sum_fin (L := L) (t := t) (g := fun k => (ga k : ℝ) * (gb k : ℝ))
      (fun k hk hb => by simp only [hga0 k hk hb, Rat.cast_zero, zero_mul])).symm.trans ?_
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [hgav, hgbv k hk]
  set SLq : ℚ := ∑ k : Fin 128, Device.liftMag .bf16 (ga k.val) * Device.liftMag .bf16 (gb k.val)
    with hSLq
  have hLift : (SLq : ℝ) ≤
      ∑ k ∈ Num.blk L t, (|(lane a i k).toReal * vv (128 * t + k)| +
        2 ^ (-126 : ℤ) * |(lane a i k).toReal| + 2 ^ (-126 : ℤ) * |vv (128 * t + k)| +
        2 ^ (-252 : ℤ)) := by
    rw [hSLq]
    push_cast
    refine (blk_sum_fin (L := L) (t := t) (g := fun k => ((Device.liftMag .bf16 (ga k) : ℚ) : ℝ) *
        ((Device.liftMag .bf16 (gb k) : ℚ) : ℝ)) (fun k hk hb => by
      simp only [hga0 k hk hb, Device.liftMag_zero, Rat.cast_zero, zero_mul])).symm.le.trans ?_
    refine Finset.sum_le_sum fun k hk => ?_
    have h := DotLift.liftMag_mul_le .bf16 .bf16 (ga k) (gb k)
    rw [Device.liftFloor_bf16] at h
    have h' : ((Device.liftMag .bf16 (ga k) * Device.liftMag .bf16 (gb k) : ℚ) : ℝ) ≤
        ((|ga k * gb k| + 2 ^ (-126 : ℤ) * |ga k| + 2 ^ (-126 : ℤ) * |gb k| +
          2 ^ (-126 : ℤ) * 2 ^ (-126 : ℤ) : ℚ) : ℝ) := by exact_mod_cast h
    push_cast at h'
    rw [hgav k, hgbv k hk] at h'
    have e : (2 : ℝ) ^ (-126 : ℤ) * 2 ^ (-126 : ℤ) = 2 ^ (-252 : ℤ) := by norm_num
    rw [e] at h'
    exact h'
  -- the sizes of the sums
  have hA : ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)| ≤ 2 ^ 121 := by
    calc ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)|
        ≤ ∑ k ∈ Num.blk L t, |(lane a i k).toReal| * 2 ^ 120 :=
          Finset.sum_le_sum fun k hk => by
            rw [abs_mul]
            exact mul_le_mul_of_nonneg_left
              (hvv _ (by have := ((hblk k).mp hk).2; omega)) (abs_nonneg _)
      _ = (∑ k ∈ Num.blk L t, |(lane a i k).toReal|) * 2 ^ 120 := by rw [Finset.sum_mul]
      _ ≤ 2 * 2 ^ 120 := mul_le_mul_of_nonneg_right hsum (by positivity)
      _ = 2 ^ 121 := by norm_num
  have hB : ∑ k ∈ Num.blk L t, |vv (128 * t + k)| ≤ 128 * 2 ^ 120 := by
    calc ∑ k ∈ Num.blk L t, |vv (128 * t + k)| ≤ ∑ _k ∈ Num.blk L t, (2 : ℝ) ^ 120 :=
          Finset.sum_le_sum fun k hk => hvv _ (by have := ((hblk k).mp hk).2; omega)
      _ = (Num.blk L t).card * 2 ^ 120 := by simp
      _ ≤ 128 * 2 ^ 120 := by
          have : ((Num.blk L t).card : ℝ) ≤ 128 := by exact_mod_cast blk_card_le L t
          nlinarith [(by positivity : (0 : ℝ) ≤ 2 ^ 120)]
  have hB0 : 0 ≤ ∑ k ∈ Num.blk L t, |vv (128 * t + k)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hA0 : 0 ≤ ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)| :=
    Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hexp : ∑ k ∈ Num.blk L t, (|(lane a i k).toReal * vv (128 * t + k)| +
        2 ^ (-126 : ℤ) * |(lane a i k).toReal| + 2 ^ (-126 : ℤ) * |vv (128 * t + k)| +
        2 ^ (-252 : ℤ)) ≤
      ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)| + 2 ^ (-124 : ℤ) +
        2 ^ (-126 : ℤ) * ∑ k ∈ Num.blk L t, |vv (128 * t + k)| := by
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul]
    have hc : ((Num.blk L t).card : ℝ) ≤ 128 := by exact_mod_cast blk_card_le L t
    have h1 : 2 ^ (-126 : ℤ) * ∑ k ∈ Num.blk L t, |(lane a i k).toReal| ≤ 2 ^ (-126 : ℤ) * 2 :=
      mul_le_mul_of_nonneg_left hsum (by positivity)
    have h2 : ((Num.blk L t).card : ℝ) * 2 ^ (-252 : ℤ) ≤ 128 * 2 ^ (-252 : ℤ) :=
      mul_le_mul_of_nonneg_right hc (by positivity)
    have h3 : (2 : ℝ) ^ (-126 : ℤ) * 2 + 128 * 2 ^ (-252 : ℤ) ≤ 2 ^ (-124 : ℤ) := by norm_num
    linarith
  have hSL : (SLq : ℝ) ≤ ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)| + 2 ^ (-124 : ℤ) +
      2 ^ (-126 : ℤ) * ∑ k ∈ Num.blk L t, |vv (128 * t + k)| := hLift.trans hexp
  have hSL122 : (SLq : ℝ) ≤ 2 ^ 122 := by
    have h1 : 2 ^ (-126 : ℤ) * ∑ k ∈ Num.blk L t, |vv (128 * t + k)| ≤ 2 ^ (-126 : ℤ) * (128 * 2 ^ 120) :=
      mul_le_mul_of_nonneg_left hB (by positivity)
    have h2 : (2 : ℝ) ^ 121 + 2 ^ (-124 : ℤ) + 2 ^ (-126 : ℤ) * (128 * 2 ^ 120) ≤ 2 ^ 122 := by norm_num
    linarith
  have hrel : 0 ≤ Device.relDot16 128 := by unfold Device.relDot16; positivity
  have habs : 0 ≤ Device.absDot16 128 := by unfold Device.absDot16; positivity
  have hcq : |(c.get2 i n).toRat| ≤ 2 ^ 124 := by
    have hc' : |((c.get2 i n).toRat : ℝ)| ≤ 2 ^ 124 := hc
    exact_mod_cast hc'
  have hSLq122 : SLq ≤ 2 ^ 122 := by exact_mod_cast hSL122
  have hroom : 2 * ((1 + Device.relDot16 128) * (|(c.get2 i n).toRat| + SLq) +
      Device.absDot16 128 + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    have hX : (1 + Device.relDot16 128) * (|(c.get2 i n).toRat| + SLq) ≤
        (1 + Device.relDot16 128) * (2 ^ 124 + 2 ^ 122) :=
      mul_le_mul_of_nonneg_left (add_le_add hcq hSLq122) (by unfold Device.relDot16; positivity)
    have key : 2 * ((1 + Device.relDot16 128) * (2 ^ 124 + 2 ^ 122) + Device.absDot16 128 +
        2 * (2 : ℚ) ^ (-150 : ℤ)) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
      unfold Device.relDot16 Device.absDot16; norm_num
    rw [Format.binary32_maxVal, Format.binary32_eta]
    linarith
  obtain ⟨hf, he⟩ := DotLift.dotAdd_step (hP.dotBF16 .tf32) hrel habs hr i n hz0 hva hvb hcfin hroom
  refine ⟨hf, ?_⟩
  have h : ((|(r.get2 i n).toRat - ((c.get2 i n).toRat + ∑ k : Fin 128, ga k.val * gb k.val)| : ℚ) : ℝ) ≤
      ((DotLoop.stepRel Device.relDot16 128 * (|(c.get2 i n).toRat| + SLq) +
        DotLoop.stepAbs Device.absDot16 128 : ℚ) : ℝ) := by exact_mod_cast he
  have hsr : DotLoop.stepRel Device.relDot16 128 ≤ 136 * 2 ^ (-22 : ℤ) := by
    unfold DotLoop.stepRel Device.relDot16; rw [Format.binary32_u]; norm_num
  have hsr0 : 0 ≤ DotLoop.stepRel Device.relDot16 128 := by
    unfold DotLoop.stepRel Device.relDot16; rw [Format.binary32_u]; norm_num
  have hsa : DotLoop.stepAbs Device.absDot16 128 ≤ 3 * 2 ^ (-120 : ℤ) := by
    unfold DotLoop.stepAbs Device.absDot16; rw [Format.binary32_u, Format.binary32_eta]; norm_num
  have hsr' : ((DotLoop.stepRel Device.relDot16 128 : ℚ) : ℝ) ≤ Num.rD := by
    have := (Rat.cast_le (K := ℝ)).mpr hsr
    push_cast at this
    unfold Num.rD; exact this
  have hsa' : ((DotLoop.stepAbs Device.absDot16 128 : ℚ) : ℝ) ≤ 3 * 2 ^ (-120 : ℤ) := by
    have := (Rat.cast_le (K := ℝ)).mpr hsa
    push_cast at this
    exact this
  have halg := dot_alg (C := |(c.get2 i n).toReal|) (A := ∑ k ∈ Num.blk L t, |(lane a i k).toReal * vv (128 * t + k)|)
    (B := ∑ k ∈ Num.blk L t, |vv (128 * t + k)|) (SL := (SLq : ℝ)) (abs_nonneg _) hA0 hB0 hSL
    (by exact_mod_cast hsr0) hsr' hsa'
  push_cast at h
  rw [hS1] at h
  exact h.trans halg

/-- **The weights sum to at most `2`**: each `l'·p2` is within `1%` of the exact weight `e·β`
(plus `2^-110`), and the exact weights sum to at most `Λ' ≤ 25/22·(l' + 2^-80)`. -/
theorem weights_sum {C : Finset ℕ} (hC : C.card ≤ 128) {e p2 : ℕ → ℝ} {β l' Lam' : ℝ}
    (hl1 : 1 / 2 ≤ l') (hΛ : 22 / 25 * Lam' - 2 ^ (-80 : ℤ) ≤ l')
    (hZ : ∑ c ∈ C, e c * β ≤ Lam') (he0 : ∀ c ∈ C, 0 ≤ e c * β)
    (hw : ∀ c ∈ C, |l' * p2 c - e c * β| ≤ 1 / 100 * (e c * β) + 2 ^ (-110 : ℤ)) :
    ∑ c ∈ C, |p2 c| ≤ 2 := by
  have hl0 : 0 < l' := by linarith
  have h1 : ∀ c ∈ C, l' * |p2 c| ≤ 101 / 100 * (e c * β) + 2 ^ (-110 : ℤ) := by
    intro c hc
    have h := hw c hc
    have h2 := abs_sub_abs_le_abs_sub (l' * p2 c) (e c * β)
    rw [abs_of_nonneg (he0 c hc), abs_mul, abs_of_pos hl0] at h2
    linarith
  have h2 : l' * ∑ c ∈ C, |p2 c| ≤ 101 / 100 * Lam' + C.card * 2 ^ (-110 : ℤ) := by
    rw [Finset.mul_sum]
    calc ∑ c ∈ C, l' * |p2 c| ≤ ∑ c ∈ C, (101 / 100 * (e c * β) + 2 ^ (-110 : ℤ)) :=
          Finset.sum_le_sum h1
      _ = 101 / 100 * ∑ c ∈ C, e c * β + C.card * 2 ^ (-110 : ℤ) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
      _ ≤ _ := by linarith
  have hc : (C.card : ℝ) ≤ 128 := by exact_mod_cast hC
  have h3 : (C.card : ℝ) * 2 ^ (-110 : ℤ) ≤ 128 * 2 ^ (-110 : ℤ) :=
    mul_le_mul_of_nonneg_right hc (by positivity)
  have h4 : l' * ∑ c ∈ C, |p2 c| ≤ 2 * l' := by
    have hε : (2 : ℝ) ^ (-80 : ℤ) ≤ 1 / 1000000 := by norm_num
    have hε2 : (128 : ℝ) * 2 ^ (-110 : ℤ) ≤ 1 / 1000000 := by norm_num
    nlinarith
  nlinarith [Finset.sum_nonneg (fun c (_ : c ∈ C) => abs_nonneg (p2 c))]

/-- Clamping at `0` moves a value no farther from a nonnegative target. -/
theorem clamp_lip {b β : ℝ} (hβ : 0 ≤ β) : |Max.max b 0 - β| ≤ |b - β| := by
  rcases le_total 0 b with h | h
  · rw [max_eq_left h]
  · rw [max_eq_right h, abs_of_nonpos (by linarith : 0 - β ≤ 0),
      abs_of_nonpos (by linarith : b - β ≤ 0)]
    linarith

/-- The rescaling factor is bounded by `4`. -/
theorem scale_abs {S x : ℝ} (hS : |S - x| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ)) * x + 2 ^ (-140 : ℤ))
    (hx1 : x ≤ 3) (hx0 : -1 ≤ x) : |S| ≤ 4 := by
  have hc : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) := by positivity
  have h1 := mul_le_mul_of_nonneg_left hx1 hc
  have h2 : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ)) * 3 + 2 ^ (-140 : ℤ) ≤ (1 : ℝ) := by norm_num
  have h3 := abs_sub_abs_le_abs_sub S x
  have h4 : |x| ≤ 3 := abs_le.mpr ⟨by linarith, hx1⟩
  linarith

/-- The relative error of a weight lane is at most `1%` when the exponential's is at most `1/400`. -/
theorem rhoq_le {ρ : ℝ} (h0 : 0 ≤ ρ) (h1 : ρ ≤ 1 / 400) :
    (1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + ρ) ^ 2 - 1 ≤ 1 / 100 := by
  have h2 : (1 + ρ) ^ 2 ≤ (1 + 1 / 400) ^ 2 := pow_le_pow_left₀ (by linarith) (by linarith) 2
  have h3 : (0 : ℝ) ≤ (1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) := by positivity
  have h4 := mul_le_mul_of_nonneg_left h2 h3
  have h5 : (1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + 1 / 400) ^ 2 - 1 ≤
      (1 : ℝ) / 100 := by norm_num
  linarith

end Trip

end Claims.PrefillAttnVarlen

end

section
/-!
# One trip of the loop on one query row

`Trip.trip`: the state of a row after a trip of the online-softmax loop (`RowState` at `t + 1`)
from its state before (`RowState` at `t`), the trip's scores and values, and the facts of the loop
body's operations (`Trip.Data`, the hypotheses the proof rules `safe_*` hand over). The row's
operations give the hypotheses of `Num.rowstep` (`Trip.row_hyps`), the column's give those of
`Num.colstep`.
-/

open TVA

namespace Claims.PrefillAttnVarlen

namespace Trip

/-- `tl.addc(tl.mul(a1, b1), tl.mul(a2, b2))` on `float32`: two rounded products then a fusable
addition, finite and within `5·2^-24` of the magnitudes plus `2^-140`. -/
theorem f32_addc2 {fusion : Bool} {a1 b1 a2 b2 p1 p2 r : F32}
    (ha1 : a1.isFinite = true) (hb1 : b1.isFinite = true)
    (ha2 : a2.isFinite = true) (hb2 : b2.isFinite = true)
    (hM : |a1.toReal * b1.toReal| + |a2.toReal * b2.toReal| ≤ 2 ^ 100)
    (hm1 : FloatV.NaNEq (FloatV.mul a1 b1) p1) (hm2 : FloatV.NaNEq (FloatV.mul a2 b2) p2)
    (hadd : AddRel fusion p1 p2 r) :
    r.isFinite = true ∧
      |r.toReal - (a1.toReal * b1.toReal + a2.toReal * b2.toReal)| ≤
        5 * 2 ^ (-24 : ℤ) * (|a1.toReal * b1.toReal| + |a2.toReal * b2.toReal|) +
          2 ^ (-140 : ℤ) := by
  have h1 := abs_nonneg (a1.toReal * b1.toReal)
  have h2 := abs_nonneg (a2.toReal * b2.toReal)
  have h127 : (2 : ℝ) ^ 100 ≤ 2 ^ 127 := by norm_num
  obtain ⟨hf1, he1⟩ := f32_mul ha1 hb1 (by linarith) hm1
  obtain ⟨hf2, he2⟩ := f32_mul ha2 hb2 (by linarith) hm2
  have hs : Format.binary32.specials ≠ .finite := by decide
  have hu32 : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hp1 : |p1.toReal| ≤ (1 + 2 ^ (-24 : ℤ)) * |a1.toReal * b1.toReal| + 2 ^ (-150 : ℤ) := by
    have := abs_sub_abs_le_abs_sub p1.toReal (a1.toReal * b1.toReal)
    nlinarith
  have hp2 : |p2.toReal| ≤ (1 + 2 ^ (-24 : ℤ)) * |a2.toReal * b2.toReal| + 2 ^ (-150 : ℤ) := by
    have := abs_sub_abs_le_abs_sub p2.toReal (a2.toReal * b2.toReal)
    nlinarith
  have hsum : |p1.toReal| + |p2.toReal| ≤ 2 ^ 102 := by
    have hu : (2 : ℝ) ^ (-24 : ℤ) ≤ 1 := by norm_num
    have e1 := mul_le_mul_of_nonneg_right hu h1
    have e2 := mul_le_mul_of_nonneg_right hu h2
    have : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 := by norm_num
    have h101 : (2 : ℝ) ^ 102 = 4 * 2 ^ 100 := by norm_num
    linarith
  have hM' : |p1.toRat| + |p2.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [← Rat.cast_le (K := ℝ)]
    push_cast
    rw [Format.binary32_maxVal_real, Format.binary32_eta_real]
    unfold FloatV.toReal at hsum
    have : (2 : ℝ) ^ 102 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * 2 ^ (-150 : ℤ) := by norm_num
    linarith
  obtain ⟨hrf, hre⟩ := FloatV.AddRel.err Format.binary32_regular hs hu32 hadd hf1 hf2 hM'
  refine ⟨hrf, ?_⟩
  have hre' : |r.toReal - (p1.toReal + p2.toReal)| ≤
      3 * 2 ^ (-24 : ℤ) * (|p1.toReal| + |p2.toReal|) + 4 * 2 ^ (-150 : ℤ) := by
    have : ((|r.toRat - (p1.toRat + p2.toRat)| : ℚ) : ℝ) ≤
        ((3 * Format.binary32.u * (|p1.toRat| + |p2.toRat|) + 4 * Format.binary32.eta : ℚ) : ℝ) := by
      exact_mod_cast hre
    push_cast at this
    rw [Format.binary32_u_real, Format.binary32_eta_real] at this
    unfold FloatV.toReal
    exact this
  have e : r.toReal - (a1.toReal * b1.toReal + a2.toReal * b2.toReal) =
      (r.toReal - (p1.toReal + p2.toReal)) + (p1.toReal - a1.toReal * b1.toReal) +
        (p2.toReal - a2.toReal * b2.toReal) := by ring
  rw [e]
  set A := |a1.toReal * b1.toReal| + |a2.toReal * b2.toReal| with hA
  have hA0 : 0 ≤ A := by linarith
  have hu0 : (0 : ℝ) < 2 ^ (-24 : ℤ) := by positivity
  have hu1 : (2 : ℝ) ^ (-24 : ℤ) ≤ 1 / 1000 := by norm_num
  have hsumP : |p1.toReal| + |p2.toReal| ≤ (1 + 2 ^ (-24 : ℤ)) * A + 2 * 2 ^ (-150 : ℤ) := by
    rw [hA]; linarith
  have k1 := mul_le_mul_of_nonneg_left hsumP (by positivity : (0 : ℝ) ≤ 3 * 2 ^ (-24 : ℤ))
  have k2 : 2 ^ (-24 : ℤ) * (2 ^ (-24 : ℤ) * A) ≤ 2 ^ (-24 : ℤ) * (1 / 1000 * A) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hu1 hA0) hu0.le
  have h3 := abs_add_le (r.toReal - (p1.toReal + p2.toReal) + (p1.toReal - a1.toReal * b1.toReal))
    (p2.toReal - a2.toReal * b2.toReal)
  have h4 := abs_add_le (r.toReal - (p1.toReal + p2.toReal)) (p1.toReal - a1.toReal * b1.toReal)
  have hc : (2 : ℝ) ^ (-150 : ℤ) * 20 ≤ 2 ^ (-140 : ℤ) := by norm_num
  have hxy : (2 : ℝ) ^ (-24 : ℤ) * 2 ^ (-150 : ℤ) ≤ 2 ^ (-150 : ℤ) := by norm_num
  have hAe : 2 ^ (-24 : ℤ) * |a1.toReal * b1.toReal| + 2 ^ (-24 : ℤ) * |a2.toReal * b2.toReal| =
      2 ^ (-24 : ℤ) * A := by rw [hA]; ring
  have hX : |r.toReal - (p1.toReal + p2.toReal) + (p1.toReal - a1.toReal * b1.toReal) +
      (p2.toReal - a2.toReal * b2.toReal)| ≤ (3 * 2 ^ (-24 : ℤ) * (|p1.toReal| + |p2.toReal|) +
        4 * 2 ^ (-150 : ℤ)) + (2 ^ (-24 : ℤ) * |a1.toReal * b1.toReal| + 2 ^ (-150 : ℤ)) +
        (2 ^ (-24 : ℤ) * |a2.toReal * b2.toReal| + 2 ^ (-150 : ℤ)) := by
    linarith
  have hAu : 0 ≤ 2 ^ (-24 : ℤ) * A := mul_nonneg hu0.le hA0
  have k3 : 3 * 2 ^ (-24 : ℤ) * ((1 + 2 ^ (-24 : ℤ)) * A + 2 * 2 ^ (-150 : ℤ)) =
      3 * 2 ^ (-24 : ℤ) * A + 3 * (2 ^ (-24 : ℤ) * (2 ^ (-24 : ℤ) * A)) +
        6 * (2 ^ (-24 : ℤ) * 2 ^ (-150 : ℤ)) := by ring
  have hmain : 3 * 2 ^ (-24 : ℤ) * (|p1.toReal| + |p2.toReal|) +
        4 * 2 ^ (-150 : ℤ) + (2 ^ (-24 : ℤ) * |a1.toReal * b1.toReal| + 2 ^ (-150 : ℤ)) +
        (2 ^ (-24 : ℤ) * |a2.toReal * b2.toReal| + 2 ^ (-150 : ℤ)) ≤
        5 * 2 ^ (-24 : ℤ) * A + 2 ^ (-140 : ℤ) := by
    generalize (2 : ℝ) ^ (-24 : ℤ) = u at *
    generalize (2 : ℝ) ^ (-150 : ℤ) = y at *
    generalize (2 : ℝ) ^ (-140 : ℤ) = z at *
    nlinarith
  exact hX.trans hmain

/-- A list sum over `range n` is the finset sum. -/
theorem list_range_sum (f : ℕ → ℚ) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]; simp

/-- The depth of every admissible tree over `128` lanes is at most `7`. -/
theorem depth_le_seven {P : Platform} (hP : Device P) {t : Tree}
    (hd : t.depth ≤ tl.sumBound P 128) : t.depth ≤ 7 := by
  have h1 := hP.sumDepth 128
  have h2 : Nat.clog 2 128 = 7 := by decide
  unfold tl.sumBound at hd
  omega

/-- **The row sum** `tl.sum1` of `128` lanes of magnitude at most `2` (IEEE additions along a tree
of depth at most `tl.sumBound`): finite and within `2^-21·Σ|x| + 2^-140` of `Σ x`. -/
theorem f32_sum128 {P : Platform} (hP : Device P) {x : ℕ → F32} {r : F32}
    (hfin : ∀ i, i < 128 → (x i).isFinite = true ∧ |(x i).toReal| ≤ 2)
    (hr : ∃ t : Tree, t.Covers 128 ∧ t.depth ≤ tl.sumBound P 128 ∧
      t.Eval (AddRel false) x r) :
    r.isFinite = true ∧
      |r.toReal - ∑ j ∈ Finset.range 128, (x j).toReal| ≤
        2 ^ (-21 : ℤ) * ∑ j ∈ Finset.range 128, |(x j).toReal| + 2 ^ (-140 : ℤ) := by
  obtain ⟨t, hcov, hdep, heval⟩ := hr
  have hd7 := depth_le_seven hP hdep
  have hxfin : ∀ i, i < 128 → (x i).isFinite = true := fun i hi => (hfin i hi).1
  have habs : ∑ i ∈ Finset.range 128, |(x i).toReal| ≤ 256 := by
    calc ∑ i ∈ Finset.range 128, |(x i).toReal| ≤ ∑ _i ∈ Finset.range 128, (2 : ℝ) :=
          Finset.sum_le_sum fun i hi => (hfin i (Finset.mem_range.mp hi)).2
      _ = 256 := by simp; norm_num
  have habs0 : 0 ≤ ∑ i ∈ Finset.range 128, |(x i).toReal| :=
    Finset.sum_nonneg fun i _ => abs_nonneg _
  have hEB : ((Tree.errBound Format.binary32.u Format.binary32.eta 7 127
      (∑ i ∈ Finset.range 128, |(x i).toRat|) : ℚ) : ℝ) ≤
      (2 : ℝ) ^ (-21 : ℤ) * ∑ i ∈ Finset.range 128, |(x i).toReal| + 2 ^ (-140 : ℤ) := by
    unfold Tree.errBound
    push_cast
    rw [Format.binary32_u_real, Format.binary32_eta_real]
    simp only [← FloatV.toReal_eq]
    have h1 : ((1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 7 - 1) ≤ 2 ^ (-21 : ℤ) := by norm_num
    have h2 : (127 : ℝ) * 2 ^ (-150 : ℤ) * (1 + 2 ^ (-24 : ℤ)) ^ 7 ≤ 2 ^ (-140 : ℤ) := by norm_num
    have h3 := mul_le_mul_of_nonneg_right h1 habs0
    linarith
  have hM : ((List.range 128).map fun i => |(x i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta 7 127
        ((List.range 128).map fun i => |(x i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [list_range_sum]
    rw [← Rat.cast_le (K := ℝ)]
    have h5 := hEB
    push_cast at h5 ⊢
    simp only [← FloatV.toReal_eq] at h5 ⊢
    rw [Format.binary32_maxVal_real]
    have : (2 : ℝ) ^ (-21 : ℤ) * 256 + 2 ^ (-140 : ℤ) ≤ 1 := by norm_num
    have h6 := mul_le_mul_of_nonneg_left habs (by norm_num : (0 : ℝ) ≤ 2 ^ (-21 : ℤ))
    have : (256 : ℝ) + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hrfin, herr⟩ := sum_err_depth Format.binary32_regular x hcov hd7 heval hxfin hM
  refine ⟨hrfin, ?_⟩
  rw [list_range_sum, list_range_sum] at herr
  have h1 : ((|r.toRat - ∑ i ∈ Finset.range 128, (x i).toRat| : ℚ) : ℝ) ≤ ((Tree.errBound Format.binary32.u Format.binary32.eta 7 127
      (∑ i ∈ Finset.range 128, |(x i).toRat|) : ℚ) : ℝ) := by exact_mod_cast herr
  refine le_trans ?_ hEB
  convert h1 using 2
  push_cast
  rfl

/-- The `float32` `-∞` is not finite. -/
theorem negInf_not_finite : (FloatV.negInf : F32).isFinite = false := by
  have h := FloatV.negInf_isInf (f := Format.binary32) (by decide)
  cases hf : (FloatV.negInf : F32).isFinite
  · rfl
  · have := (FloatV.isFinite_iff.mp hf).2
    rw [h] at this; exact absurd this (by decide)

/-- **The row maximum** `tl.max` of a row whose attended lanes are finite and whose other lanes
are `-∞` (the attended set `blk L t` is nonempty): a finite attended lane, no attended lane above. -/
theorem max_row {x : ℕ → F32} {mij : F32} {L t : ℕ} (hL : 128 * t ≤ L)
    (hin : ∀ c ∈ Num.blk L t, (x c).isFinite = true)
    (hout : ∀ c, c < 128 → c ∉ Num.blk L t → x c = FloatV.negInf)
    (htree : ∃ tr : Tree, tr.Covers 128 ∧ tr.Eval (FloatV.MaxRel true) x mij) :
    mij.isFinite = true ∧ (∃ c ∈ Num.blk L t, (x c).toReal = mij.toReal) ∧
      ∀ c ∈ Num.blk L t, (x c).toReal ≤ mij.toReal := by
  obtain ⟨tr, hc, hev⟩ := htree
  have hmem : ∀ i, i ∈ tr.leaves ↔ i < 128 := fun i => by rw [hc.mem_iff, List.mem_range]
  have hs : Format.binary32.specials = .ieee := by decide
  have hblk : ∀ c, c ∈ Num.blk L t ↔ c < 128 ∧ 128 * t + c ≤ L := fun c => by
    simp [Num.blk]
  have hnan : ∀ i ∈ tr.leaves, (x i).isNaN = false := by
    intro i hi
    have hi' := (hmem i).mp hi
    by_cases hb : i ∈ Num.blk L t
    · exact (FloatV.isFinite_iff.mp (hin i hb)).1
    · rw [hout i hi' hb]; exact FloatV.negInf_isNaN hs
  obtain ⟨⟨i0, hi0, hr⟩, hle⟩ := Tree.eval_maxRel_toReal hev hnan
  have h0 : 0 ∈ Num.blk L t := (hblk 0).mpr ⟨by norm_num, by omega⟩
  have hi0b : i0 ∈ Num.blk L t := by
    by_contra hb
    have hneg : x i0 = FloatV.negInf := hout i0 ((hmem i0).mp hi0) hb
    have h1 := hle 0 ((hmem 0).mpr (by norm_num))
    rw [hr, hneg] at h1
    have h2 := FloatV.negInf_toRat_le (f := Format.binary32) Format.binary32_regular hs
      (FloatV.isFinite_iff.mp (hin 0 h0)).1
    have h3 : (FloatV.negInf : F32).toRat = (x 0).toRat := by
      apply le_antisymm h2.1
      have : (x 0).toReal ≤ (FloatV.negInf : F32).toReal := h1
      unfold FloatV.toReal at this
      exact_mod_cast this
    have := h2.2 h3
    have h5 := hin 0 h0
    rw [this, negInf_not_finite] at h5
    exact absurd h5 (by decide)
  exact ⟨by rw [hr]; exact hin i0 hi0b, ⟨i0, hi0b, by rw [hr]⟩,
    fun c hc' => hle c ((hmem c).mpr ((hblk c).mp hc').1)⟩

/-- **The new running maximum** `tl.maximum(m_i, m_ij)`: from `-∞` it is `m_ij`; from a finite
`m_i` it is finite with value `max`. -/
theorem max_new {mi mij mnew : F32} (hmx : FloatV.MaxRel true mi mij mnew)
    (hmij : mij.isFinite = true) (hmi : mi = FloatV.negInf ∨ mi.isFinite = true) :
    mnew.isFinite = true ∧ (mi = FloatV.negInf → mnew = mij) ∧
      (mi.isFinite = true → mnew.toReal = Max.max mi.toReal mij.toReal) := by
  have hs : Format.binary32.specials = .ieee := by decide
  have hy := (FloatV.isFinite_iff.mp hmij).1
  have hx : mi.isNaN = false := by
    rcases hmi with h | h
    · rw [h]; exact FloatV.negInf_isNaN hs
    · exact (FloatV.isFinite_iff.mp h).1
  have hmem := hmx.mem hx hy
  refine ⟨?_, fun h => ?_, fun h => hmx.toReal_eq_max hx hy⟩
  · rcases hmi with h' | h'
    · have := FloatV.MaxRel.negInf_left Format.binary32_regular hs (h' ▸ hmx) hy
      rw [this]; exact hmij
    · rcases hmem with h | h
      · rw [h]; exact h'
      · rw [h]; exact hmij
  · rw [h] at hmx
    exact FloatV.MaxRel.negInf_left Format.binary32_regular hs hmx hy

/-- `tl.exp(s - m)` for `s = -∞` and a finite `m` is `+0`. -/
theorem exp_sub_negInf {P : Platform} (hP : Device P) {s m a p : F32} (hs : s = FloatV.negInf)
    (hm : m.isFinite = true) (ha : FloatV.NaNEq (FloatV.sub s m) a)
    (hp : P.math .exp .binary32 a p) : p = FloatV.ofPat 0 := by
  subst hs
  have h1 := FloatV.sub_negInf_left (f := Format.binary32) (by decide) hm
  have h2 : (FloatV.sub (FloatV.negInf : F32) m).isNaN = false := by
    rw [h1]; exact FloatV.negInf_isNaN (by decide)
  have h3 : a = FloatV.negInf := by rw [ha.eq h2, h1]
  rw [h3] at hp
  exact hP.expF32NegInf p hp

/-- `tl.exp(s - m)` for finite `s ≤ m` with `m - s ≤ 2Y`: within `rE Y · e^(s-m) + ηE`. -/
theorem exp_sub_rE {P : Platform} (hP : Device P) {s m a p : F32} {Y : ℝ}
    (hs : s.isFinite = true) (hm : m.isFinite = true) (hsm : s.toReal ≤ m.toReal)
    (hY0 : 0 ≤ Y) (hY5 : Y ≤ 5000) (hZ : m.toReal - s.toReal ≤ 2 * Y)
    (ha : FloatV.NaNEq (FloatV.sub s m) a) (hp : P.math .exp .binary32 a p) :
    p.isFinite = true ∧
      |p.toReal - Real.exp (s.toReal - m.toReal)| ≤
        Num.rE Y * Real.exp (s.toReal - m.toReal) + Num.ηE := by
  obtain ⟨hf, he⟩ := f32_exp_sub hP hs hm hsm (Z := 2 * Y) hZ (by linarith) ha hp
  refine ⟨hf, he.trans ?_⟩
  have h0 : 0 < Real.exp (s.toReal - m.toReal) := Real.exp_pos _
  have h1 : (2 : ℝ) ^ (-22 : ℤ) * (8 / 5 + 4 / 5 * (2 * Y)) ≤ Num.rE Y := by
    unfold Num.rE
    have : (2 : ℝ) ^ (-21 : ℤ) = 2 * 2 ^ (-22 : ℤ) := by norm_num
    rw [this]
    have : (0 : ℝ) < 2 ^ (-22 : ℤ) := by positivity
    nlinarith
  have := mul_le_mul_of_nonneg_right h1 h0.le
  unfold Num.ηE
  linarith

/-- The same with the exponent size `Δ` of the update `exp(m_old - m_new)`: within
`rA Δ · e^(s-m) + ηE`. -/
theorem exp_sub_rA {P : Platform} (hP : Device P) {s m a p : F32} {Δ : ℝ}
    (hs : s.isFinite = true) (hm : m.isFinite = true) (hsm : s.toReal ≤ m.toReal)
    (hΔ : m.toReal - s.toReal ≤ Δ) (hΔ1 : Δ ≤ 10000)
    (ha : FloatV.NaNEq (FloatV.sub s m) a) (hp : P.math .exp .binary32 a p) :
    p.isFinite = true ∧
      |p.toReal - Real.exp (s.toReal - m.toReal)| ≤
        Num.rA Δ * Real.exp (s.toReal - m.toReal) + Num.ηE := by
  obtain ⟨hf, he⟩ := f32_exp_sub hP hs hm hsm hΔ hΔ1 ha hp
  refine ⟨hf, he.trans ?_⟩
  have h0 : 0 < Real.exp (s.toReal - m.toReal) := Real.exp_pos _
  have hΔ0 : 0 ≤ Δ := by linarith
  have h1 : (2 : ℝ) ^ (-22 : ℤ) * (8 / 5 + 4 / 5 * Δ) ≤ Num.rA Δ := by
    unfold Num.rA
    have : (2 : ℝ) ^ (-21 : ℤ) = 2 * 2 ^ (-22 : ℤ) := by norm_num
    rw [this]
    have : (0 : ℝ) < 2 ^ (-22 : ℤ) := by positivity
    nlinarith
  have := mul_le_mul_of_nonneg_right h1 h0.le
  unfold Num.ηE
  linarith

/-- **The update factor `alpha` and the new maximum.** From `m_i` (`-∞` on the first trip, else
finite and within `Y`) and `m_ij` (finite, within `Y`): `m_new` is finite within `Y`, at least
`m_ij`, and `alpha` is finite and within `rA Δ·α + ηE` of the exact `α`. -/
theorem alpha_new {P : Platform} (hP : Device P) {t : ℕ} {Y : ℝ} {mi mij mnew e1 alpha : F32}
    (hY5 : Y ≤ 5000) (hmij : mij.isFinite = true)
    (hmijY : -Y ≤ mij.toReal ∧ mij.toReal ≤ Y)
    (hmi : (t = 0 ∧ mi = FloatV.negInf) ∨
      (0 < t ∧ mi.isFinite = true ∧ -Y ≤ mi.toReal ∧ mi.toReal ≤ Y))
    (hmx : FloatV.MaxRel true mi mij mnew)
    (hsub : FloatV.NaNEq (FloatV.sub mi mnew) e1) (hexp : P.math .exp .binary32 e1 alpha) :
    mnew.isFinite = true ∧ mij.toReal ≤ mnew.toReal ∧ -Y ≤ mnew.toReal ∧ mnew.toReal ≤ Y ∧
      alpha.isFinite = true ∧
      ∃ Δ α : ℝ, ((t = 0 ∧ mnew.toReal = mij.toReal ∧ Δ = 0 ∧ α = 0) ∨
        (0 < t ∧ mnew.toReal = Max.max mi.toReal mij.toReal ∧ Δ = mnew.toReal - mi.toReal ∧
          α = Real.exp (mi.toReal - mnew.toReal))) ∧
        |alpha.toReal - α| ≤ Num.rA Δ * α + Num.ηE := by
  rcases hmi with ⟨ht, hmi⟩ | ⟨ht, hmif, hmi1, hmi2⟩
  · obtain ⟨hf, hnew, -⟩ := max_new hmx hmij (Or.inl hmi)
    have hnew' := hnew hmi
    have hal := exp_sub_negInf hP hmi hf hsub hexp
    rw [hnew'] at hf ⊢
    refine ⟨hf, le_rfl, hmijY.1, hmijY.2, ?_, 0, 0, Or.inl ⟨ht, rfl, rfl, rfl⟩, ?_⟩
    · rw [hal]; exact zero32_fin
    · rw [hal, zero32_real]
      unfold Num.rA Num.ηE
      norm_num
  · obtain ⟨hf, -, hmax⟩ := max_new hmx hmij (Or.inr hmif)
    have hm := hmax hmif
    have hsm : mi.toReal ≤ mnew.toReal := by rw [hm]; exact le_max_left _ _
    have hnY : mnew.toReal ≤ Y := by rw [hm]; exact max_le hmi2 hmijY.2
    have hmn : mij.toReal ≤ mnew.toReal := by rw [hm]; exact le_max_right _ _
    obtain ⟨haf, hae⟩ := exp_sub_rA hP hmif hf hsm (Δ := mnew.toReal - mi.toReal) le_rfl
      (by linarith) hsub hexp
    exact ⟨hf, hmn, by linarith, hnY, haf, _, _, Or.inr ⟨ht, hm, rfl, rfl⟩, hae⟩

/-- The facts one trip's row operations give, as `Num.rowstep` and `Num.colstep` take them:
`qs` the lane scores, `p` the lane weights (`tl.exp`), `mij` the lane maximum, `m'` the new
maximum, `α'`, `β'` the computed update factors, `l'` the new sum, `lij` the lane sum. -/
structure RowHyps (L t : ℕ) (sc qs p : ℕ → ℝ) (δ Y m l m' mij Δ α α' β β' l' lij : ℝ) :
    Prop where
  hqs : ∀ c ∈ Num.blk L t, |qs c - sc (128 * t + c)| ≤ δ
  hmij : (∃ c ∈ Num.blk L t, qs c = mij) ∧ ∀ c ∈ Num.blk L t, qs c ≤ mij
  hm' : (t = 0 ∧ m' = mij ∧ Δ = 0 ∧ α = 0) ∨
    (0 < t ∧ m' = Max.max m mij ∧ Δ = m' - m ∧ α = Real.exp (m - m'))
  hα' : |α' - α| ≤ Num.rA Δ * α + Num.ηE
  hβ : β = Real.exp (mij - m')
  hβ' : |β' - β| ≤ Num.rE Y * β + Num.ηE
  hp0 : ∀ c ∈ Finset.range 128, c ∉ Num.blk L t → p c = 0
  hp : ∀ c ∈ Num.blk L t, |p c - Real.exp (qs c - mij)| ≤ Num.rE Y * Real.exp (qs c - mij) + Num.ηE
  hlij : |lij - ∑ c ∈ Finset.range 128, p c| ≤ Num.rS * ∑ c ∈ Finset.range 128, |p c| + Num.ηS
  hl' : |l' - (α' * l + β' * lij)| ≤ Num.rO * (|α' * l| + |β' * lij|) + Num.ηO

/-- The state a trip starts from, as floats: the first trip has `m_i = -∞`, `l_i = +0`; later ones
have finite values meeting the invariant. -/
def Start (L t : ℕ) (sc sh : ℕ → ℝ) (δ Y : ℝ) (mi li : F32) : Prop :=
  (t = 0 ∧ mi = FloatV.negInf ∧ li = FloatV.ofPat 0) ∨
  (0 < t ∧ Num.RowInv L t sc sh δ Y mi.toReal li.toReal ∧ mi.isFinite = true ∧
    li.isFinite = true)

/-- **The row operations of one trip**, from the device facts: the new maximum and sum are
finite, and the quantities meet the hypotheses of `Num.rowstep`. -/
theorem row_hyps {P : Platform} (hP : Device P) {fusion : Bool} {L t : ℕ} {sc sh : ℕ → ℝ}
    {δ Y : ℝ} (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000)
    (hL : 128 * t ≤ L) (ht : t < 1024)
    {mi li mij mnew lij lnew alpha beta e1 e2 u1 u2 : F32} {x x1 pl : ℕ → F32}
    (hst : Start L t sc sh δ Y mi li)
    (hxin : ∀ c ∈ Num.blk L t, (x c).isFinite = true ∧ |(x c).toReal - sc (128 * t + c)| ≤ δ)
    (hxout : ∀ c, c < 128 → c ∉ Num.blk L t → x c = FloatV.negInf)
    (hmax : ∃ tr : Tree, tr.Covers 128 ∧ tr.Eval (FloatV.MaxRel true) x mij)
    (hsub1 : ∀ c, c < 128 → FloatV.NaNEq (FloatV.sub (x c) mij) (x1 c))
    (hexp1 : ∀ c, c < 128 → P.math .exp .binary32 (x1 c) (pl c))
    (hsum : ∃ tr : Tree, tr.Covers 128 ∧ tr.depth ≤ tl.sumBound P 128 ∧
      tr.Eval (AddRel false) pl lij)
    (hmaximum : FloatV.MaxRel true mi mij mnew)
    (hsub2 : FloatV.NaNEq (FloatV.sub mi mnew) e1) (hexp2 : P.math .exp .binary32 e1 alpha)
    (hsub3 : FloatV.NaNEq (FloatV.sub mij mnew) e2) (hexp3 : P.math .exp .binary32 e2 beta)
    (hmul1 : FloatV.NaNEq (FloatV.mul alpha li) u1) (hmul2 : FloatV.NaNEq (FloatV.mul beta lij) u2)
    (haddc : AddRel fusion u1 u2 lnew) :
    mnew.isFinite = true ∧ lnew.isFinite = true ∧ alpha.isFinite = true ∧ beta.isFinite = true ∧
      (∀ c, c < 128 → (pl c).isFinite = true ∧ |(pl c).toReal| ≤ 2) ∧
      (∀ c, c < 128 → c ∉ Num.blk L t → pl c = FloatV.ofPat 0) ∧
      |alpha.toReal| ≤ 3 ∧ -(2 : ℝ) ^ (-140 : ℤ) ≤ alpha.toReal ∧ |beta.toReal| ≤ 3 ∧
      ∃ Δ α : ℝ, (0 ≤ α ∧ α ≤ 1 ∧ 0 ≤ Δ ∧ Δ ≤ 2 * Y) ∧
        RowHyps L t sc (fun c => (x c).toReal) (fun c => (pl c).toReal) δ Y mi.toReal
          li.toReal mnew.toReal mij.toReal Δ α alpha.toReal (Real.exp (mij.toReal - mnew.toReal))
          beta.toReal lnew.toReal lij.toReal := by
  have hY0 : 0 ≤ Y := by
    have := hY 0 (Nat.zero_le _)
    have := abs_nonneg (sc 0)
    linarith
  have hbl : ∀ c, c ∈ Num.blk L t ↔ c < 128 ∧ 128 * t + c ≤ L := fun c => by simp [Num.blk]
  obtain ⟨hmijf, ⟨c0, hc0, hc0e⟩, hle⟩ := max_row hL (fun c hc => (hxin c hc).1) hxout hmax
  have hqsY : ∀ c ∈ Num.blk L t, |(x c).toReal| ≤ Y := by
    intro c hc
    have h1 := (hxin c hc).2
    have h2 := hY (128 * t + c) ((hbl c).mp hc).2
    have h3 : |(x c).toReal| ≤ |sc (128 * t + c)| + |(x c).toReal - sc (128 * t + c)| := by
      calc |(x c).toReal| = |sc (128 * t + c) + ((x c).toReal - sc (128 * t + c))| := by ring_nf
        _ ≤ _ := abs_add_le _ _
    linarith
  have hmijY : -Y ≤ mij.toReal ∧ mij.toReal ≤ Y := by
    rw [← hc0e]; exact abs_le.mp (hqsY c0 hc0)
  have hmi : (t = 0 ∧ mi = FloatV.negInf) ∨
      (0 < t ∧ mi.isFinite = true ∧ -Y ≤ mi.toReal ∧ mi.toReal ≤ Y) := by
    rcases hst with ⟨h0, h1, -⟩ | ⟨h0, inv, hf, -⟩
    · exact Or.inl ⟨h0, h1⟩
    · obtain ⟨-, -, h3, h4⟩ := inv.bounds hδ hY hY5 h0 ht.le
      exact Or.inr ⟨h0, hf, h3, h4⟩
  have hlbound : li.isFinite = true ∧ |li.toReal| ≤ 2 ^ 18 := by
    rcases hst with ⟨-, -, h1⟩ | ⟨h0, inv, -, hf⟩
    · rw [h1]
      exact ⟨zero32_fin, by rw [zero32_real]; norm_num⟩
    · obtain ⟨h3, h4, -, -⟩ := inv.bounds hδ hY hY5 h0 ht.le
      exact ⟨hf, by rw [abs_of_nonneg (by linarith)]; exact h4⟩
  obtain ⟨hnf, hmn, hnY1, hnY2, haf, Δ, α, hm', hα'⟩ :=
    alpha_new hP hY5 hmijf hmijY hmi hmaximum hsub2 hexp2
  obtain ⟨hbf, hbe⟩ := exp_sub_rE hP hmijf hnf hmn hY0 hY5 (by linarith) hsub3 hexp3
  have hrE : Num.rE Y ≤ 1 / 100 := by unfold Num.rE; norm_num; nlinarith
  -- the weights of the lanes
  have hpe : ∀ c ∈ Num.blk L t, (pl c).isFinite = true ∧
      |(pl c).toReal - Real.exp ((x c).toReal - mij.toReal)| ≤
        Num.rE Y * Real.exp ((x c).toReal - mij.toReal) + Num.ηE := by
    intro c hc
    have hc' := ((hbl c).mp hc).1
    exact exp_sub_rE hP (hxin c hc).1 hmijf (hle c hc) hY0 hY5
      (by linarith [(abs_le.mp (hqsY c hc)).1]) (hsub1 c hc') (hexp1 c hc')
  have hp0 : ∀ c, c < 128 → c ∉ Num.blk L t → pl c = FloatV.ofPat 0 := fun c hc hb =>
    exp_sub_negInf hP (hxout c hc hb) hmijf (hsub1 c hc) (hexp1 c hc)
  have hpl : ∀ c, c < 128 → (pl c).isFinite = true ∧ |(pl c).toReal| ≤ 2 := by
    intro c hc
    by_cases hb : c ∈ Num.blk L t
    · obtain ⟨hf, he⟩ := hpe c hb
      refine ⟨hf, ?_⟩
      have h1 : Real.exp ((x c).toReal - mij.toReal) ≤ 1 :=
        Real.exp_le_one_iff.mpr (by linarith [hle c hb])
      have h2 : 0 < Real.exp ((x c).toReal - mij.toReal) := Real.exp_pos _
      have h3 := abs_sub_abs_le_abs_sub (pl c).toReal (Real.exp ((x c).toReal - mij.toReal))
      rw [abs_of_pos h2] at h3
      have h4 : Num.rE Y * Real.exp ((x c).toReal - mij.toReal) ≤ 1 / 100 * 1 :=
        mul_le_mul hrE h1 h2.le (by norm_num)
      have h5 : Num.ηE ≤ 1 / 100 := by unfold Num.ηE; norm_num
      linarith
    · rw [hp0 c hc hb]; exact ⟨zero32_fin, by rw [zero32_real]; norm_num⟩
  -- the row sum
  obtain ⟨hlijf, hlije⟩ := f32_sum128 hP hpl hsum
  have hS256 : ∑ j ∈ Finset.range 128, |(pl j).toReal| ≤ 256 := by
    calc ∑ j ∈ Finset.range 128, |(pl j).toReal| ≤ ∑ _j ∈ Finset.range 128, (2 : ℝ) :=
          Finset.sum_le_sum fun j hj => (hpl j (Finset.mem_range.mp hj)).2
      _ = 256 := by simp; norm_num
  have hS0 : 0 ≤ ∑ j ∈ Finset.range 128, |(pl j).toReal| :=
    Finset.sum_nonneg fun j _ => abs_nonneg _
  have hlij_abs : |lij.toReal| ≤ 600 := by
    have h1 := abs_sub_abs_le_abs_sub lij.toReal (∑ j ∈ Finset.range 128, (pl j).toReal)
    have h2 : |∑ j ∈ Finset.range 128, (pl j).toReal| ≤ ∑ j ∈ Finset.range 128, |(pl j).toReal| :=
      Finset.abs_sum_le_sum_abs _ _
    have h3 : (2 : ℝ) ^ (-21 : ℤ) * ∑ j ∈ Finset.range 128, |(pl j).toReal| ≤ 1 := by
      have := mul_le_mul_of_nonneg_left hS256 (by norm_num : (0 : ℝ) ≤ 2 ^ (-21 : ℤ))
      have : (2 : ℝ) ^ (-21 : ℤ) * 256 ≤ 1 := by norm_num
      linarith
    have h4 : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 := by norm_num
    linarith
  -- the sizes of the factors
  have hαb : 0 ≤ α ∧ α ≤ 1 ∧ 0 ≤ Δ ∧ Δ ≤ 2 * Y := by
    rcases hm' with ⟨-, -, hΔ, hα⟩ | ⟨h0, hm, hΔ, hα⟩
    · exact ⟨by rw [hα], by rw [hα]; norm_num, by rw [hΔ], by rw [hΔ]; linarith⟩
    · have hmi' : mi.isFinite = true ∧ -Y ≤ mi.toReal ∧ mi.toReal ≤ Y := by
        rcases hmi with ⟨h, -⟩ | ⟨-, h⟩
        · omega
        · exact h
      have hmm : mi.toReal ≤ mnew.toReal := by rw [hm]; exact le_max_left _ _
      exact ⟨by rw [hα]; exact (Real.exp_pos _).le,
        by rw [hα]; exact Real.exp_le_one_iff.mpr (by linarith), by rw [hΔ]; linarith [hmi'.2.1, hmi'.2.2],
        by rw [hΔ]; linarith⟩
  obtain ⟨hα0, hα1, hΔ0, hΔ2⟩ := hαb
  have hrA : Num.rA Δ ≤ 1 := by unfold Num.rA; norm_num; nlinarith
  have hα'b : |alpha.toReal| ≤ 3 := by
    have h1 := abs_sub_abs_le_abs_sub alpha.toReal α
    rw [abs_of_nonneg hα0] at h1
    have h2 : Num.rA Δ * α ≤ 1 * 1 := mul_le_mul hrA hα1 hα0 (by norm_num)
    have h3 : Num.ηE ≤ 1 := by unfold Num.ηE; norm_num
    linarith
  have hβ0 : 0 < Real.exp (mij.toReal - mnew.toReal) := Real.exp_pos _
  have hβ1 : Real.exp (mij.toReal - mnew.toReal) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hβ'b : |beta.toReal| ≤ 3 := by
    have h1 := abs_sub_abs_le_abs_sub beta.toReal (Real.exp (mij.toReal - mnew.toReal))
    rw [abs_of_pos hβ0] at h1
    have h2 : Num.rE Y * Real.exp (mij.toReal - mnew.toReal) ≤ 1 / 100 * 1 :=
      mul_le_mul hrE hβ1 hβ0.le (by norm_num)
    have h3 : Num.ηE ≤ 1 := by unfold Num.ηE; norm_num
    linarith
  -- the new sum
  have hM : |alpha.toReal * li.toReal| + |beta.toReal * lij.toReal| ≤ 2 ^ 100 := by
    rw [abs_mul, abs_mul]
    have h1 : |alpha.toReal| * |li.toReal| ≤ 3 * 2 ^ 18 :=
      mul_le_mul hα'b hlbound.2 (abs_nonneg _) (by norm_num)
    have h2 : |beta.toReal| * |lij.toReal| ≤ 3 * 600 :=
      mul_le_mul hβ'b hlij_abs (abs_nonneg _) (by norm_num)
    have : (3 : ℝ) * 2 ^ 18 + 3 * 600 ≤ 2 ^ 100 := by norm_num
    linarith
  obtain ⟨hlf, hle'⟩ := f32_addc2 haf hlbound.1 hbf hlijf hM hmul1 hmul2 haddc
  have hαlow : -(2 : ℝ) ^ (-140 : ℤ) ≤ alpha.toReal := by
    have h1 := (abs_le.mp hα').1
    have h2 : Num.rA Δ * α ≤ 1 * α := mul_le_mul_of_nonneg_right hrA hα0
    unfold Num.ηE at h1
    linarith
  refine ⟨hnf, hlf, haf, hbf, hpl, hp0, hα'b, hαlow, hβ'b, Δ, α, ⟨hα0, hα1, hΔ0, hΔ2⟩, ?_⟩
  refine ⟨fun c hc => (hxin c hc).2, ⟨⟨c0, hc0, hc0e⟩, hle⟩, hm', hα', rfl, hbe, ?_,
    fun c hc => (hpe c hc).2, ?_, ?_⟩
  · intro c hc hb
    rw [hp0 c (Finset.mem_range.mp hc) hb, zero32_real]
  · unfold Num.rS Num.ηS; exact hlije
  · unfold Num.rO Num.ηO; exact hle'

/-- **The tensors of one trip of the loop body and the facts the proof rules hand over** (lines
148-171 of the model): `qk` the masked scores, `mi`, `li`, `acc` the loop state, `v` the loaded
values; each other tensor is the result of the operation its fact describes. -/
structure Data (BD : ℕ) (P : Platform) (fusion : Bool) where
  qk : Tensor [128, 128] F32
  mi : Tensor [128] F32
  li : Tensor [128] F32
  acc : Tensor [128, BD] F32
  v : Tensor [128, BD] BF16
  mij : Tensor [128] F32
  x1 : Tensor [128, 128] F32
  p : Tensor [128, 128] F32
  lij : Tensor [128] F32
  mnew : Tensor [128] F32
  e1 : Tensor [128] F32
  alpha : Tensor [128] F32
  e2 : Tensor [128] F32
  beta : Tensor [128] F32
  u1 : Tensor [128] F32
  u2 : Tensor [128] F32
  lnew : Tensor [128] F32
  psc : Tensor [128] F32
  p1 : Tensor [128, 128] F32
  dq : Tensor [128] F32
  asc : Tensor [128] F32
  acc1 : Tensor [128, BD] F32
  p2 : Tensor [128, 128] BF16
  acc2 : Tensor [128, BD] F32
  hmax : ∀ i : Fin 128, ∃ tr : Tree, tr.Covers 128 ∧ tr.Eval (FloatV.MaxRel true)
    (fun j => if h : j < 128 then qk.get2 i ⟨j, h⟩ else default) (mij.get1 i)
  hsub1 : ∀ i, FloatV.NaNEq (FloatV.sub (qk.get i) ((mij.col 128).get i)) (x1.get i)
  hexp1 : ∀ i, P.math .exp .binary32 (x1.get i) (p.get i)
  hsum : ∀ i : Fin 128, ∃ tr : Tree, tr.Covers 128 ∧ tr.depth ≤ tl.sumBound P 128 ∧
    tr.Eval (AddRel false) (fun j => if h : j < 128 then p.get2 i ⟨j, h⟩ else default) (lij.get1 i)
  hmaximum : ∀ i, FloatV.MaxRel true (mi.get i) (mij.get i) (mnew.get i)
  hsub2 : ∀ i, FloatV.NaNEq (FloatV.sub (mi.get i) (mnew.get i)) (e1.get i)
  hexp2 : ∀ i, P.math .exp .binary32 (e1.get i) (alpha.get i)
  hsub3 : ∀ i, FloatV.NaNEq (FloatV.sub (mij.get i) (mnew.get i)) (e2.get i)
  hexp3 : ∀ i, P.math .exp .binary32 (e2.get i) (beta.get i)
  hmul1 : ∀ i, FloatV.NaNEq (FloatV.mul (alpha.get i) (li.get i)) (u1.get i)
  hmul2 : ∀ i, FloatV.NaNEq (FloatV.mul (beta.get i) (lij.get i)) (u2.get i)
  haddc : ∀ i, AddRel fusion (u1.get i) (u2.get i) (lnew.get i)
  hdiv1 : ∀ i, P.div .binary32 (beta.get i) (lnew.get i) (psc.get i)
  hmul3 : ∀ i, FloatV.NaNEq (FloatV.mul (p.get i) ((psc.col 128).get i)) (p1.get i)
  hdiv2 : ∀ i, P.div .binary32 (li.get i) (lnew.get i) (dq.get i)
  hmul4 : ∀ i, FloatV.NaNEq (FloatV.mul (dq.get i) (alpha.get i)) (asc.get i)
  hmul5 : ∀ i, FloatV.NaNEq (FloatV.mul (acc.get i) ((asc.col BD).get i)) (acc1.get i)
  hcvt : ∀ i, FloatV.NaNEq (FloatV.cvtWith .rn (p1.get i)) (p2.get i)
  hdot : P.dot cfgBF16 p2 v acc1 acc2 ∨
    ∃ d, P.dot cfgBF16 p2 v (Tensor.full (cfgBF16.acc.ofBits 0)) d ∧
      ∀ i, AccAdd fusion cfgBF16.acc (d.get i) (acc1.get i) (acc2.get i)

section Lanes

variable {BD : ℕ} {P : Platform} {fusion : Bool} (D : Data BD P fusion) (r : Fin 128)

/-- Lane `c` of `qk - m_ij`. -/
theorem Data.lane_sub1 {c : ℕ} (hc : c < 128) :
    FloatV.NaNEq (FloatV.sub (lane D.qk r c) (D.mij.get1 r)) (lane D.x1 r c) := by
  have h := D.hsub1 (r, ⟨c, hc⟩, ())
  simp only [Tensor.get_col] at h
  rw [lane_of_lt D.qk r ⟨c, hc⟩, lane_of_lt D.x1 r ⟨c, hc⟩]
  exact h

/-- Lane `c` of `tl.exp(qk - m_ij)`. -/
theorem Data.lane_exp1 {c : ℕ} (hc : c < 128) :
    P.math .exp .binary32 (lane D.x1 r c) (lane D.p r c) := by
  have h := D.hexp1 (r, ⟨c, hc⟩, ())
  rw [lane_of_lt D.x1 r ⟨c, hc⟩, lane_of_lt D.p r ⟨c, hc⟩]
  exact h

/-- Lane `c` of `p * p_scale`. -/
theorem Data.lane_mul3 {c : ℕ} (hc : c < 128) :
    FloatV.NaNEq (FloatV.mul (lane D.p r c) (D.psc.get1 r)) (lane D.p1 r c) := by
  have h := D.hmul3 (r, ⟨c, hc⟩, ())
  simp only [Tensor.get_col] at h
  rw [lane_of_lt D.p r ⟨c, hc⟩, lane_of_lt D.p1 r ⟨c, hc⟩]
  exact h

/-- Lane `c` of `p.to(bfloat16)`. -/
theorem Data.lane_cvt {c : ℕ} (hc : c < 128) :
    FloatV.NaNEq (FloatV.cvtWith .rn (lane D.p1 r c)) (lane D.p2 r c) := by
  have h := D.hcvt (r, ⟨c, hc⟩, ())
  rw [lane_of_lt D.p1 r ⟨c, hc⟩, lane_of_lt D.p2 r ⟨c, hc⟩]
  exact h

end Lanes

/-- The state a trip starts from, in the shape `Num.rowstep` and `Num.colstep` use. -/
theorem state_cases {BD : ℕ} {R : RowSpec BD} {δ Y : ℝ} {V : Fin BD → ℝ} {t : ℕ} {m l : F32}
    {a : Fin BD → F32} (hst : RowState R δ Y V t m l a) :
    ∃ sh : ℕ → ℝ, Start R.L t R.sc sh δ Y m l ∧
      Num.RowInv R.L t R.sc sh δ Y m.toReal l.toReal ∧
      ∀ e, (a e).isFinite = true ∧ (t = 0 → l.toReal = 0 ∧ (a e).toReal = 0) ∧
        Num.Close (l.toReal * (a e).toReal) (Num.Asum sh (R.vv e) (Num.keysN R.L t) m.toReal)
          (Num.Aabs sh (R.vv e) (Num.keysN R.L t) m.toReal) (Num.omA Y t m.toReal) (Num.bA Y)
          (Num.ηA * (1 + V e)) t := by
  rcases hst with ⟨h0, hm, hl, ha⟩ | ⟨h0, sh, hinv, hmf, hlf, hacc⟩
  · subst h0
    have hl0 : l.toReal = 0 := by rw [hl, zero32_real]
    refine ⟨fun _ => 0, Or.inl ⟨rfl, hm, hl⟩, ?_, fun e => ?_⟩
    · rw [hl0]; exact Num.RowInv.zero
    · have ha0 : (a e).toReal = 0 := by rw [ha e, zero32_real]
      refine ⟨by rw [ha e]; exact zero32_fin, fun _ => ⟨hl0, ha0⟩, ?_⟩
      rw [hl0, ha0]
      exact Num.Close.zero_col
  · exact ⟨sh, Or.inr ⟨h0, hinv, hmf, hlf⟩, hinv, fun e =>
      ⟨(hacc e).1, fun h => absurd h h0.ne', (hacc e).2⟩⟩

/-- **One trip of the online-softmax loop on one query row.** From the row's state before the
trip (`RowState` at `t`), the scores and values of the trip, and the facts of the loop body's
operations, the state after the trip (`RowState` at `t + 1`). -/
theorem trip {BD : ℕ} {P : Platform} (hP : Device P) {fusion : Bool} (D : Data BD P fusion)
    (R : RowSpec BD) {δ Y : ℝ} {V : Fin BD → ℝ} {t : ℕ} (r : Fin 128)
    (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ R.L → |R.sc j| + δ ≤ Y) (hY5 : Y ≤ 5000)
    (hV0 : ∀ e, 0 ≤ V e) (hV : ∀ e, ∑ j ∈ Finset.range (R.L + 1), |R.vv e j| ≤ V e)
    (hVu : ∀ e, V e ≤ 2 ^ 138) (hvv : ∀ e j, j ≤ R.L → |R.vv e j| ≤ 2 ^ 120)
    (hL : 128 * t ≤ R.L) (ht : t < 1024)
    (hst : RowState R δ Y V t (D.mi.get1 r) (D.li.get1 r) (fun e => D.acc.get2 r e))
    (hqk : ∀ c : Fin 128, 128 * t + c.val ≤ R.L →
      (D.qk.get2 r c).isFinite = true ∧ |(D.qk.get2 r c).toReal - R.sc (128 * t + c.val)| ≤ δ)
    (hmask : ∀ c : Fin 128, ¬ (128 * t + c.val ≤ R.L) → D.qk.get2 r c = FloatV.negInf)
    (hv : ∀ (c : Fin 128) (e : Fin BD), (D.v.get2 c e).isFinite = true)
    (hvl : ∀ c : Fin 128, 128 * t + c.val ≤ R.L → ∀ e,
      (D.v.get2 c e).toReal = R.vv e (128 * t + c.val)) :
    RowState R δ Y V (t + 1) (D.mnew.get1 r) (D.lnew.get1 r) (fun e => D.acc2.get2 r e) := by
  classical
  have hY0 : 0 ≤ Y := by
    have := hY 0 (Nat.zero_le _)
    have := abs_nonneg (R.sc 0)
    linarith
  have hblk : ∀ c, c ∈ Num.blk R.L t ↔ c < 128 ∧ 128 * t + c ≤ R.L := fun c => by
    simp [Num.blk]
  obtain ⟨sh, hstart, inv, hcol⟩ := state_cases hst
  obtain ⟨hmf, hlf, haf, hbf, hpl, hpz, hα3, hαlow, hβ3, Δ, α, ⟨hα0, hα1, hΔ0, hΔ2⟩, H⟩ :=
    row_hyps (fusion := fusion) hP hδ hY hY5 hL ht (mi := D.mi.get1 r) (li := D.li.get1 r)
      (mij := D.mij.get1 r) (mnew := D.mnew.get1 r) (lij := D.lij.get1 r) (lnew := D.lnew.get1 r)
      (alpha := D.alpha.get1 r) (beta := D.beta.get1 r) (e1 := D.e1.get1 r) (e2 := D.e2.get1 r)
      (u1 := D.u1.get1 r) (u2 := D.u2.get1 r) (x := lane D.qk r) (x1 := lane D.x1 r)
      (pl := lane D.p r) hstart
      (fun c hc => by
        obtain ⟨hc1, hc2⟩ := (hblk c).mp hc
        rw [lane_of_lt D.qk r ⟨c, hc1⟩]
        exact hqk ⟨c, hc1⟩ hc2)
      (fun c hc hb => by
        have hn : ¬ (128 * t + c ≤ R.L) := fun h => hb ((hblk c).mpr ⟨hc, h⟩)
        rw [lane_of_lt D.qk r ⟨c, hc⟩]
        exact hmask ⟨c, hc⟩ hn)
      (D.hmax r) (fun c hc => D.lane_sub1 r hc) (fun c hc => D.lane_exp1 r hc) (D.hsum r)
      (D.hmaximum (r, ())) (D.hsub2 (r, ())) (D.hexp2 (r, ())) (D.hsub3 (r, ()))
      (D.hexp3 (r, ())) (D.hmul1 (r, ())) (D.hmul2 (r, ())) (D.haddc (r, ()))
  have inv' := Num.rowstep (qs := fun c => (lane D.qk r c).toReal)
    (p := fun c => (lane D.p r c).toReal) hδ hY hY5 hL ht inv H.hqs H.hmij H.hm' H.hα' H.hβ H.hβ'
    H.hp0 H.hp H.hlij H.hl'
  obtain ⟨hl1, hl2, hm1, hm2⟩ := inv'.bounds hδ hY hY5 (Nat.succ_pos t) ht
  -- the old sum, `m_i`, and the new quantities
  have hlib : (D.li.get1 r).isFinite = true ∧ 0 ≤ (D.li.get1 r).toReal ∧
      (D.li.get1 r).toReal ≤ 2 ^ 18 := by
    rcases hstart with ⟨-, -, h⟩ | ⟨h0, hinv, -, hf⟩
    · rw [h]; exact ⟨zero32_fin, by rw [zero32_real], by rw [zero32_real]; norm_num⟩
    · obtain ⟨h3, h4, -, -⟩ := hinv.bounds hδ hY hY5 h0 ht.le
      exact ⟨hf, by linarith, h4⟩
  have hrE : Num.rE Y ≤ 1 / 400 := by
    unfold Num.rE; norm_num; nlinarith
  have hrE0 : 0 ≤ Num.rE Y := by unfold Num.rE; positivity
  have hβ0 : 0 ≤ Real.exp ((D.mij.get1 r).toReal - (D.mnew.get1 r).toReal) := (Real.exp_pos _).le
  have hmijm : (D.mij.get1 r).toReal ≤ (D.mnew.get1 r).toReal := by
    rcases H.hm' with ⟨-, h, -, -⟩ | ⟨-, h, -, -⟩
    · exact h.ge
    · rw [h]; exact le_max_right _ _
  have hβ1 : Real.exp ((D.mij.get1 r).toReal - (D.mnew.get1 r).toReal) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by linarith)
  obtain ⟨hpsf, hps8, hps, hpsc⟩ := ps_facts hP hY0 hbf hlf hl1 hβ3 hβ0
    (hrE.trans (by norm_num)) H.hβ' (D.hdiv1 (r, ()))
  have hβt : |Max.max (D.beta.get1 r).toReal 0 - Real.exp ((D.mij.get1 r).toReal - (D.mnew.get1 r).toReal)| ≤
      Num.rE Y * Real.exp ((D.mij.get1 r).toReal - (D.mnew.get1 r).toReal) + Num.ηE :=
    (clamp_lip hβ0).trans H.hβ'
  have hp2 : ∀ c ∈ Num.blk R.L t, (lane D.p2 r c).isFinite = true ∧
      |(lane D.p2 r c).toReal - (lane D.p r c).toReal * Max.max (D.psc.get1 r).toReal 0| ≤
        (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ)) *
          |(lane D.p r c).toReal * Max.max (D.psc.get1 r).toReal 0| + 2 ^ (-130 : ℤ) := by
    intro c hc
    have hc1 := ((hblk c).mp hc).1
    exact p2_lane (hpl c hc1).1 hpsf (hpl c hc1).2 hps8 hpsc (D.lane_mul3 r hc1) (D.lane_cvt r hc1)
  have hp2z : ∀ c, c < 128 → c ∉ Num.blk R.L t →
      (lane D.p2 r c).isFinite = true ∧ (lane D.p2 r c).toRat = 0 := by
    intro c hc hb
    have hm := D.lane_mul3 r hc
    rw [hpz c hc hb] at hm
    obtain ⟨hf1, h1⟩ := mul_zero_f32 hpsf hm
    exact cvt_zero_bf16 hf1 h1 (D.lane_cvt r hc)
  have hp2fin : ∀ c, c < 128 → (lane D.p2 r c).isFinite = true := fun c hc => by
    by_cases hb : c ∈ Num.blk R.L t
    · exact (hp2 c hb).1
    · exact (hp2z c hc hb).1
  -- the weights sum to at most `2`
  set l' := (D.lnew.get1 r).toReal with hl'def
  set l := (D.li.get1 r).toReal with hldef
  set m' := (D.mnew.get1 r).toReal with hm'def
  set mij := (D.mij.get1 r).toReal with hmijdef
  set β := Real.exp (mij - m') with hβdef
  set α' := (D.alpha.get1 r).toReal with hα'def
  have hLamsucc := Num.Lam_succ hL sh (fun c => (lane D.qk r c).toReal) (D.mi.get1 r).toReal m'
  have hLam0 : 0 ≤ Num.Lam sh (Num.keysN R.L t) (D.mi.get1 r).toReal :=
    Finset.sum_nonneg fun j _ => (Real.exp_pos _).le
  have hZ0 : 0 ≤ ∑ c ∈ Num.blk R.L t, Real.exp ((lane D.qk r c).toReal - m') :=
    Finset.sum_nonneg fun c _ => (Real.exp_pos _).le
  have hΛ := (l_rel' inv' hδ hY hY5 (Nat.succ_pos t) ht).1
  have hZ : ∑ c ∈ Num.blk R.L t, Real.exp ((lane D.qk r c).toReal - mij) * β ≤
      Num.Lam (Num.shNext t sh (fun c => (lane D.qk r c).toReal)) (Num.keysN R.L (t + 1)) m' := by
    have e : ∀ c ∈ Num.blk R.L t,
        Real.exp ((lane D.qk r c).toReal - mij) * β = Real.exp ((lane D.qk r c).toReal - m') := by
      intro c _
      rw [hβdef, ← Real.exp_add]; congr 1; ring
    rw [Finset.sum_congr rfl e, hLamsucc]
    have h0 : 0 ≤ Real.exp ((D.mi.get1 r).toReal - m') *
        Num.Lam sh (Num.keysN R.L t) (D.mi.get1 r).toReal := mul_nonneg (Real.exp_pos _).le hLam0
    linarith
  have hw : ∀ c ∈ Num.blk R.L t,
      |l' * (lane D.p2 r c).toReal - Real.exp ((lane D.qk r c).toReal - mij) * β| ≤
        1 / 100 * (Real.exp ((lane D.qk r c).toReal - mij) * β) + 2 ^ (-110 : ℤ) := by
    intro c hc
    have he0 : 0 < Real.exp ((lane D.qk r c).toReal - mij) := Real.exp_pos _
    have he1 : Real.exp ((lane D.qk r c).toReal - mij) ≤ 1 :=
      Real.exp_le_one_iff.mpr (by linarith [H.hmij.2 c hc])
    have h := Num.ta_weight hrE0 hrE he0.le he1 hβ0 hβ1 (H.hp c hc) hβt hl1 hl2 hps (hp2 c hc).2
    have hq := rhoq_le hrE0 hrE
    have hnn : 0 ≤ Real.exp ((lane D.qk r c).toReal - mij) * β := mul_nonneg he0.le hβ0
    exact h.trans (add_le_add_left (mul_le_mul_of_nonneg_right hq hnn) _)
  have hsum2 : ∑ c ∈ Num.blk R.L t, |(lane D.p2 r c).toReal| ≤ 2 :=
    weights_sum (blk_card_le R.L t) hl1 hΛ hZ
      (fun c _ => mul_nonneg (Real.exp_pos _).le hβ0) hw
  -- the rescaling factor
  have hl'pos : 0 < l' := by linarith
  have hl0 : 0 ≤ l := hlib.2.1
  have hl2' : l ≤ 2 ^ 18 := hlib.2.2
  have hx0 : 0 ≤ l / l' := div_nonneg hl0 hl'pos.le
  have hx19 : l / l' ≤ 2 ^ 19 := by
    rw [div_le_iff₀ hl'pos]; nlinarith
  obtain ⟨hdqf, hdqe⟩ := f32_div hP hlib.1 hlf hl'pos
    (by rw [abs_of_nonneg hx0]; linarith [(by norm_num : (2 : ℝ) ^ 19 ≤ 2 ^ 100)])
    (D.hdiv2 (r, ()))
  have hdqe' : |(D.dq.get1 r).toReal - l / l'| ≤ 2 ^ (-22 : ℤ) * |l / l'| + 2 ^ (-148 : ℤ) := hdqe
  have hq20 : |(D.dq.get1 r).toReal| ≤ 2 ^ 20 := by
    have h1 := abs_sub_abs_le_abs_sub (D.dq.get1 r).toReal (l / l')
    have h2 := hdqe'
    rw [abs_of_nonneg hx0] at h1 h2
    have h3 : (2 : ℝ) ^ (-22 : ℤ) * (l / l') ≤ 2 ^ (-22 : ℤ) * 2 ^ 19 :=
      mul_le_mul_of_nonneg_left hx19 (by positivity)
    have h4 : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 19 + 2 ^ (-148 : ℤ) + 2 ^ 19 ≤ 2 ^ 20 := by norm_num
    linarith
  have hqα : |(D.dq.get1 r).toReal * α'| ≤ 2 ^ 127 := by
    rw [abs_mul]
    have : |(D.dq.get1 r).toReal| * |α'| ≤ 2 ^ 20 * 3 :=
      mul_le_mul hq20 hα3 (abs_nonneg _) (by positivity)
    have h2 : (2 : ℝ) ^ 20 * 3 ≤ 2 ^ 127 := by norm_num
    linarith
  obtain ⟨hascf, hasce⟩ := f32_mul hdqf haf hqα (D.hmul4 (r, ()))
  have hS := scale_real hl0 hl2' hl1 hα3 hαlow hdqe' hasce
  have hxlow : -1 ≤ l / l' * α' := by
    have h1 : l / l' * -(2 : ℝ) ^ (-140 : ℤ) ≤ l / l' * α' :=
      mul_le_mul_of_nonneg_left hαlow hx0
    have h2 : l / l' * (2 : ℝ) ^ (-140 : ℤ) ≤ 2 ^ 19 * 2 ^ (-140 : ℤ) :=
      mul_le_mul_of_nonneg_right hx19 (by positivity)
    have h3 : (2 : ℝ) ^ 19 * 2 ^ (-140 : ℤ) ≤ 1 := by norm_num
    linarith
  have hx3 : l / l' * α' ≤ 3 := by
    rcases Nat.eq_zero_or_pos t with h0 | h0
    · have hl00 : l = 0 := by
        rcases hstart with ⟨-, -, h⟩ | ⟨h1, -⟩
        · rw [hldef, h, zero32_real]
        · omega
      rw [hl00]; norm_num
    · have hLr := l_rel' inv hδ hY hY5 h0 ht.le
      have hLr' := l_rel' inv' hδ hY hY5 (Nat.succ_pos t) ht
      obtain ⟨hαe⟩ : ∃ _ : α = Real.exp ((D.mi.get1 r).toReal - m'), True := by
        rcases H.hm' with ⟨h1, -⟩ | ⟨-, -, -, h⟩
        · omega
        · exact ⟨h, trivial⟩
      have hD : α * Num.Lam sh (Num.keysN R.L t) (D.mi.get1 r).toReal ≤
          Num.Lam (Num.shNext t sh (fun c => (lane D.qk r c).toReal)) (Num.keysN R.L (t + 1)) m' := by
        rw [hLamsucc, hαe]; linarith
      have hrA : Num.rA Δ ≤ 1 / 100 := by
        unfold Num.rA; norm_num; nlinarith
      have hα'up : α' ≤ 101 / 100 * α + 2 ^ (-140 : ℤ) := by
        have h1 := (abs_le.mp H.hα').2
        have h2 : Num.rA Δ * α ≤ 1 / 100 * α := mul_le_mul_of_nonneg_right hrA hα0
        have h3 : Num.ηE = 2 ^ (-140 : ℤ) := rfl
        rw [h3] at h1
        generalize (2 : ℝ) ^ (-140 : ℤ) = z at h1 ⊢
        generalize Num.rA Δ * α = w at h1 h2
        linarith
      have := scale_le hl0 hl2' hl1 hLr.2 hLr.1 hLr'.1 hD hα0 hα1 hα'up
      rw [div_mul_eq_mul_div, div_le_iff₀ hl'pos]
      linarith
  have hSabs : |(D.asc.get1 r).toReal| ≤ 4 := scale_abs hS hx3 hxlow
  -- the columns
  have hafin : ∀ k : Fin 128, (D.p2.get2 r k).isFinite = true := fun k => by
    have := hp2fin k.val k.isLt
    rwa [lane_of_lt D.p2 r k] at this
  have haz : ∀ k : Fin 128, ¬ (128 * t + k.val ≤ R.L) → (D.p2.get2 r k).toRat = 0 := fun k hk => by
    have := (hp2z k.val k.isLt (fun hb => hk ((hblk _).mp hb).2)).2
    rwa [lane_of_lt D.p2 r k] at this
  have hz : ∀ e : Fin BD, (Tensor.full (cfgBF16.acc.ofBits 0) : Tensor [128, BD] F32).get2 r e = FloatV.ofPat 0 := by
    intro e
    simp only [Tensor.get2, cfgBF16, DType.ofBits_f32]
    rfl
  have hres : ∀ e : Fin BD, (D.acc2.get2 r e).isFinite = true ∧
      Num.Close (l' * (D.acc2.get2 r e).toReal)
        (Num.Asum (Num.shNext t sh (fun c => (lane D.qk r c).toReal)) (R.vv e)
          (Num.keysN R.L (t + 1)) m')
        (Num.Aabs (Num.shNext t sh (fun c => (lane D.qk r c).toReal)) (R.vv e)
          (Num.keysN R.L (t + 1)) m')
        (Num.omA Y (t + 1) m') (Num.bA Y) (Num.ηA * (1 + V e)) (t + 1) := by
    intro e
    obtain ⟨hafe, ha0, hcinv⟩ := hcol e
    have ha121 : |(D.acc.get2 r e).toReal| ≤ 2 ^ 121 := by
      rcases Nat.eq_zero_or_pos t with h0 | h0
      · rw [(ha0 h0).2]; norm_num
      · exact (Num.acc_le hδ hY hY5 h0 ht.le inv (hvv e) (hV0 e) (hV e) (hVu e) hcinv).1
    have hm5 := D.hmul5 (r, e, ())
    simp only [Tensor.get_col] at hm5
    have haS : |(D.acc.get2 r e).toReal * (D.asc.get1 r).toReal| ≤ 2 ^ 121 * 4 := by
      rw [abs_mul]
      exact mul_le_mul ha121 hSabs (abs_nonneg _) (by positivity)
    obtain ⟨hf1, he1⟩ := f32_mul (x := D.acc.get2 r e) (y := D.asc.get1 r) hafe hascf
      (haS.trans (by norm_num)) hm5
    have ha1 : |(D.acc1.get2 r e).toReal - (D.acc.get2 r e).toReal * (D.asc.get1 r).toReal| ≤
        2 ^ (-24 : ℤ) * |(D.acc.get2 r e).toReal * (D.asc.get1 r).toReal| + 2 ^ (-149 : ℤ) := by
      have h5 : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-149 : ℤ) := by norm_num
      have he1' : |(D.acc1.get2 r e).toReal - (D.acc.get2 r e).toReal * (D.asc.get1 r).toReal| ≤
          2 ^ (-24 : ℤ) * |(D.acc.get2 r e).toReal * (D.asc.get1 r).toReal| + 2 ^ (-150 : ℤ) := he1
      linarith
    have hacc1 : |(D.acc1.get2 r e).toReal| ≤ 2 ^ 124 := by
      have h1 := abs_sub_abs_le_abs_sub (D.acc1.get2 r e).toReal
        ((D.acc.get2 r e).toReal * (D.asc.get1 r).toReal)
      have h3 : (2 : ℝ) ^ (-24 : ℤ) * |(D.acc.get2 r e).toReal * (D.asc.get1 r).toReal| ≤
          2 ^ (-24 : ℤ) * (2 ^ 121 * 4) := mul_le_mul_of_nonneg_left haS (by positivity)
      have h4 : (2 : ℝ) ^ 121 * 4 + 2 ^ (-24 : ℤ) * (2 ^ 121 * 4) + 2 ^ (-149 : ℤ) ≤ 2 ^ 124 := by
        norm_num
      linarith
    have hdot := dot_ok hP (fusion := fusion) (L := R.L) (t := t) (vv := R.vv e) (a := D.p2)
      (b := D.v) (c := D.acc1) (z := Tensor.full (cfgBF16.acc.ofBits 0)) (r := D.acc2) r e
      D.hdot (hz e) hafin (fun k => hv k e) haz (fun k hk => hvl k hk e) (hvv e) hf1 hacc1 hsum2
    refine ⟨hdot.1, ?_⟩
    exact Num.colstep (qs := fun c => (lane D.qk r c).toReal) (p := fun c => (lane D.p r c).toReal)
      (p2 := fun c => (lane D.p2 r c).toReal) (vv := R.vv e) hδ hY hY5 hL ht inv inv' (hvv e)
      (hV0 e) (hV e) (hVu e) hcinv ha0 H.hqs H.hmij H.hm' H.hα' H.hβ hβt H.hp hps
      (fun c hc => (hp2 c hc).2) hS ha1 hdot.2
  exact Or.inr ⟨Nat.succ_pos t, Num.shNext t sh (fun c => (lane D.qk r c).toReal), inv', hmf, hlf,
    fun e => hres e⟩

end Trip

end Claims.PrefillAttnVarlen

end
