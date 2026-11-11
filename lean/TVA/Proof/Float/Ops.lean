import TVA.Proof.Float.Round

/-!
# The IEEE operations on patterns: finite operands, negation, comparison, exact conversions

The pattern operations of `TVA.Base.Float.Arith` test for NaN and infinities first, then round
the exact result once. On finite operands only the rounding is left:

* `add_of_finite`, `sub_of_finite`, `mul_of_finite`, `fma_of_finite`, `div_of_finite`,
  `cvt_of_finite`: each operation is `roundRat` of the exact result. Combine with
  `TVA.Proof.Float.Round` to bound the error; `TVA.Proof.Float.Error` does it for `FloatV`.
* `cvt_rz_sat`: toward zero, saturation changes only the result for `±∞`.
* Negation flips the sign bit only: `toRat_neg`, `isFinite_neg`, `isNaN_neg`, ….
* `cmp_lt_iff`, `cmp_eq_iff`: the ordered comparison of non-NaN patterns is the order of their
  keys and the equality of their values.
* `Wider f g` (`g` has at least `f`'s precision, subnormal quantum and range) makes a conversion
  exact (`roundRat_of_wider`, `cvt_of_wider`); `wider_binary16_binary32`, … give it for every
  widening Triton performs (`float16`/`bfloat16` into `float32` or `float64`, `float32` into
  `float64`, fp8 into `float16`, `bfloat16`, `float32`, fp4 into `float16` and `bfloat16`).
-/
namespace TVA.Format

variable {f : Format}

/-! ## Finite operands -/

/-- On finite operands, pattern addition is `roundRat` of the exact sum. -/
theorem add_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true) :
    f.add rnd x y = f.roundRat rnd false (f.sign x && f.sign y) (f.toRat x + f.toRat y) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  unfold add; simp [hxn, hyn, hxi, hyi]

/-- On finite operands, pattern multiplication is `roundRat` of the exact product. -/
theorem mul_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true) :
    f.mul rnd x y = f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x * f.toRat y) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  unfold mul; simp [hxn, hyn, hxi, hyi]

/-- On finite operands, pattern `fma` is `roundRat` of the exact `x·y + z`. -/
theorem fma_of_finite (rnd : RoundMode) {x y z : ℕ} (hx : f.isFinite x = true)
    (hy : f.isFinite y = true) (hz : f.isFinite z = true) :
    f.fma rnd x y z = f.roundRat rnd false ((f.sign x != f.sign y) && f.sign z)
      (f.toRat x * f.toRat y + f.toRat z) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  obtain ⟨hzn, hzi⟩ := (isFinite_iff z).mp hz
  unfold fma; simp [hxn, hyn, hzn, hxi, hyi, hzi]

/-- On finite operands with a nonzero divisor, pattern division is `roundRat` of the exact
quotient. -/
theorem div_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true)
    (hy0 : f.isZero y = false) :
    f.div rnd x y = f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x / f.toRat y) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  unfold div; simp [hxn, hyn, hxi, hyi, hy0]

/-- On a finite operand, a conversion is `roundRat` of its value in the target format. -/
theorem cvt_of_finite (g : Format) (rnd : RoundMode) (sat : Bool) {x : ℕ} (hx : f.isFinite x = true) :
    f.cvt g rnd sat x = g.roundRat rnd sat (f.sign x) (f.toRat x) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  unfold cvt; simp [hxn, hxi]

/-- **Toward zero, saturation changes only `±∞`**: on every other pattern `cvt … .rz true` and
`cvt … .rz false` agree, since an overflow toward zero gives `±maxVal` either way (`encode`). -/
theorem cvt_rz_sat (g : Format) {x : ℕ} (hx : f.isInf x = false) :
    f.cvt g .rz true x = f.cvt g .rz false x := by
  unfold cvt roundRat roundMag encode
  simp [hx]

/-! ## Negation -/

/-- Negation flips the sign. -/
theorem sign_neg (x : ℕ) : f.sign (f.neg x) = !f.sign x := sign_setSign _ _

/-- Negation keeps the magnitude. -/
theorem mag_neg (x : ℕ) : f.mag (f.neg x) = f.mag x := mag_setSign _ _

/-- A negated pattern fits in the format's width. -/
theorem neg_lt (x : ℕ) : f.neg x < 2 ^ f.width := setSign_lt _ _

/-- Negation negates the value (of every pattern). -/
theorem toRat_neg (x : ℕ) : f.toRat (f.neg x) = -f.toRat x := by
  unfold neg; rw [toRat_setSign, toRat_eq]
  cases f.sign x <;> simp

/-- Negation keeps finiteness. -/
theorem isFinite_neg (x : ℕ) : f.isFinite (f.neg x) = f.isFinite x := isFinite_setSign _ _
/-- Negation keeps NaN-ness. -/
theorem isNaN_neg (x : ℕ) : f.isNaN (f.neg x) = f.isNaN x := isNaN_setSign _ _
/-- Negation keeps infiniteness. -/
theorem isInf_neg (x : ℕ) : f.isInf (f.neg x) = f.isInf x := isInf_setSign _ _
/-- Negation keeps zeroness. -/
theorem isZero_neg (x : ℕ) : f.isZero (f.neg x) = f.isZero x := isZero_setSign _ _

/-- On finite operands, pattern subtraction is `roundRat` of the exact difference. -/
theorem sub_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true) :
    f.sub rnd x y = f.roundRat rnd false (f.sign x && !f.sign y) (f.toRat x - f.toRat y) := by
  unfold sub
  rw [add_of_finite rnd hx (by rw [isFinite_neg]; exact hy), sign_neg, toRat_neg, sub_eq_add_neg]

/-! ## Comparison -/

/-- The comparison of two non-NaN patterns, by their values. -/
theorem cmp_of_not_nan {x y : ℕ} (hx : f.isNaN x = false) (hy : f.isNaN y = false) :
    f.cmp x y = some (if f.toRat x < f.toRat y then .lt else if f.toRat x = f.toRat y then .eq else .gt) := by
  unfold cmp; simp [hx, hy]

/-- **The ordered `<` is the order of the order keys**, on patterns that are not NaN. -/
theorem cmp_lt_iff {x y : ℕ} (hx : f.isNaN x = false) (hy : f.isNaN y = false) :
    f.cmp x y = some .lt ↔ f.orderKey x < f.orderKey y := by
  rw [cmp_of_not_nan hx hy, ← toRat_lt_iff_orderKey]
  constructor
  · intro h; by_contra h'; split_ifs at h <;> simp_all
  · intro h; simp [h]

/-- The ordered `==` of non-NaN patterns is the equality of their values (`-0 == +0`). -/
theorem cmp_eq_iff {x y : ℕ} (hx : f.isNaN x = false) (hy : f.isNaN y = false) :
    f.cmp x y = some .eq ↔ f.toRat x = f.toRat y := by
  rw [cmp_of_not_nan hx hy]
  constructor
  · intro h; split_ifs at h with h1 h2 <;> simp_all
  · intro h; simp [h]

/-! ## Exact conversions -/

/-- **A finite value of a narrower format rounds into a wider one exactly**, in either mode, a
zero keeping its sign. -/
theorem roundRat_of_wider {g : Format} (hf : f.Regular) (hg : g.Regular) (hw : Wider f g)
    (rnd : RoundMode) (sat : Bool) {n : ℕ} (hn : f.isFinite n = true) :
    g.isFinite (g.roundRat rnd sat (f.sign n) (f.toRat n)) = true ∧
      g.toRat (g.roundRat rnd sat (f.sign n) (f.toRat n)) = f.toRat n := by
  obtain ⟨hm, hq, hmax⟩ := hw
  have hrange : |f.toRat n| ≤ g.maxVal := (abs_toRat_le_maxVal hf hn).trans hmax
  by_cases h0 : f.toRat n = 0
  · rw [h0]
    obtain ⟨-, h2, h3⟩ := roundRat_zero_spec (f := g) (rnd := rnd) (sat := sat) (zs := f.sign n) hg
    exact ⟨h2, h3⟩
  obtain ⟨-, hfin, hval⟩ := roundRat_spec hg rnd sat (f.sign n) h0 hrange
  refine ⟨hfin, ?_⟩
  rw [hval]
  have ha : 0 < |f.toRat n| := abs_pos.mpr h0
  obtain ⟨S, E, hSE, hS, hE⟩ := magVal_sig_exp (f := f) (f.mag n)
  have habs : |f.toRat n| = S * (2 : ℚ) ^ E := by rw [abs_toRat, hSE]
  -- the quantum of `g` divides `2^E`
  have hQ : g.qexp |f.toRat n| ≤ E := by
    have hlt : |f.toRat n| < (2 : ℚ) ^ (E + f.m + 1) := by
      rw [habs]
      have : (S : ℚ) < (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
        rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]
        exact_mod_cast hS
      calc (S : ℚ) * 2 ^ E < (2 : ℚ) ^ ((f.m : ℤ) + 1) * 2 ^ E :=
            mul_lt_mul_of_pos_right this (by positivity)
        _ = (2 : ℚ) ^ (E + f.m + 1) := by rw [← zpow_add₀ (by norm_num)]; ring_nf
    have hL := log_lt_of_lt ha hlt
    rw [qexp_eq ha]
    have : (f.m : ℤ) ≤ g.m := by exact_mod_cast hm
    omega
  set Qg := g.qexp |f.toRat n| with hQg
  have hK : |f.toRat n| * (2 : ℚ) ^ (-Qg) = ((S * 2 ^ (E - Qg).toNat : ℕ) : ℚ) := by
    rw [habs]; push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ (by norm_num)]
    congr 2
  rw [roundSig_exact rnd hK]
  by_cases hneg : f.toRat n < 0
  · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
  · simp only [hneg, ite_false, abs_of_nonneg (not_lt.mp hneg), one_mul]

/-- **A conversion into a wider format is exact**: finite, with the same value. -/
theorem cvt_of_wider {g : Format} (hf : f.Regular) (hg : g.Regular) (hw : Wider f g)
    (rnd : RoundMode) (sat : Bool) {n : ℕ} (hn : f.isFinite n = true) :
    g.isFinite (f.cvt g rnd sat n) = true ∧ g.toRat (f.cvt g rnd sat n) = f.toRat n := by
  rw [cvt_of_finite g rnd sat hn]; exact roundRat_of_wider hf hg hw rnd sat hn

/-- `maxVal` of `f` is below `maxVal` of `g` when `f`'s values stay below `g`'s top binade. -/
theorem maxVal_le_of_top {g : Format} (hf : f.Regular) (hg : g.Regular)
    (h : (f.topExp : ℤ) + 1 - f.bias ≤ (g.topExp : ℤ) - g.bias) : f.maxVal ≤ g.maxVal := by
  have h1 := maxVal_lt hf
  have h2 := magVal_ge_of_exp (f := g) (M := g.maxMag) (by rw [maxMag_div hg]; exact topExp_pos hg)
  rw [magVal_maxMag hg, maxMag_div hg] at h2
  exact h1.le.trans ((zpow_le_zpow_right₀ (by norm_num) h).trans h2)

/-- `float32` holds every `float16` value. -/
theorem wider_binary16_binary32 : Wider binary16 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, binary16, binary32]⟩
/-- `float32` holds every `bfloat16` value. -/
theorem wider_bfloat16_binary32 : Wider bfloat16 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, bfloat16, binary32]⟩
/-- `float64` holds every `float32` value. -/
theorem wider_binary32_binary64 : Wider binary32 binary64 :=
  ⟨by decide, by decide, maxVal_le_of_top (by decide) (by decide) (by decide)⟩
/-- `float64` holds every `float16` value. -/
theorem wider_binary16_binary64 : Wider binary16 binary64 :=
  ⟨by decide, by decide, maxVal_le_of_top (by decide) (by decide) (by decide)⟩
/-- `float64` holds every `bfloat16` value. -/
theorem wider_bfloat16_binary64 : Wider bfloat16 binary64 :=
  ⟨by decide, by decide, maxVal_le_of_top (by decide) (by decide) (by decide)⟩
/-- `float16` holds every `float8e5` value. -/
theorem wider_e5m2_binary16 : Wider e5m2 binary16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e5m2, binary16]⟩
/-- `float16` holds every `float8e4nv` value. -/
theorem wider_e4m3_binary16 : Wider e4m3 binary16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e4m3, binary16]⟩
/-- `bfloat16` holds every `float8e5` value. -/
theorem wider_e5m2_bfloat16 : Wider e5m2 bfloat16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e5m2, bfloat16]⟩
/-- `bfloat16` holds every `float8e4nv` value. -/
theorem wider_e4m3_bfloat16 : Wider e4m3 bfloat16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e4m3, bfloat16]⟩
/-- `float32` holds every `float8e5` value. -/
theorem wider_e5m2_binary32 : Wider e5m2 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e5m2, binary32]⟩
/-- `float32` holds every `float8e4nv` value. -/
theorem wider_e4m3_binary32 : Wider e4m3 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e4m3, binary32]⟩
/-- `bfloat16` holds every fp4 (`e2m1`) value. -/
theorem wider_e2m1_bfloat16 : Wider e2m1 bfloat16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e2m1, bfloat16]⟩
/-- `float16` holds every fp4 (`e2m1`) value. -/
theorem wider_e2m1_binary16 : Wider e2m1 binary16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e2m1, binary16]⟩

end TVA.Format
