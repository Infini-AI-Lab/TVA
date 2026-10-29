import TVA.Math.Rounding

/-!
# The IEEE operations on patterns

Each operation takes and returns patterns of one format (IEEE 754-2019 §5.4, §6, §9.6):
- `add`, `sub`, `mul`, `fma`, `div`: one rounding of the exact result (`roundRat`), after the
  special cases (a NaN operand gives the library's canonical NaN `nanMag`; `∞ − ∞`, `0 · ∞`, `0 / 0` and
  `∞ / ∞` give it too; an infinite operand gives the infinity the sign rules say). The sign of an
  exact zero result follows §6.3: `x + y` is `-0` only when both are `-0`; a product or quotient
  has the exclusive or of the signs; `fma` treats the product as an addend with that sign.
- `sqrt`: `√x` rounded (`sqrtMag`); `sqrt(-0) = -0`; a negative operand gives NaN. To nearest it is
  finite and within `√x·u + η` of `√x` (`sqrt_rn_err`).
- `neg`, `absPat`: flip or clear the sign bit (a NaN stays a NaN).
- `maxNum`, `minNum`: IEEE 754-2008 `maxNum`/`minNum` (LLVM `maxnum`): a NaN operand gives the
  other operand, `max(-0, +0) = +0`, `min(-0, +0) = -0`. `maxNaN`, `minNaN`: a NaN operand gives
  NaN (LLVM `maximum`).
- `cmp`: the ordered comparison, `none` when an operand is NaN; `-0` equals `+0`.
- `cvt f g rnd sat`: from format `f` into `g`, one rounding; NaN gives `g`'s canonical NaN, `±∞`
  gives `±∞` (`±maxVal` when saturating, NaN into `e4m3`, which has no infinity); a finite value
  beyond `g`'s range overflows as `roundRat` says (`±maxVal` when saturating). Toward zero,
  saturation changes only the result for `±∞` (`cvt_rz_sat`: an overflow toward zero gives
  `±maxVal` either way).

In a format without infinities an infinite result (`x / 0`) is what an overflow gives there
(`infPat`). The definitions are in `TVA.Math.Defs`; the device agrees with them where
`evidence/float-layer` says. `Wider f g` (`g` holds every finite value of `f`) makes a
conversion exact (`cvt_of_wider`).

`add_of_finite`, `mul_of_finite`, `fma_of_finite`, `div_of_finite`, `cvt_of_finite`: on finite
operands each operation is `roundRat` of the exact result.
-/
namespace TVA.Math.Float.Format

/-! ## Finite operands -/

variable {f : Format}

theorem add_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true) :
    f.add rnd x y = f.roundRat rnd false (f.sign x && f.sign y) (f.toRat x + f.toRat y) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  unfold add; simp [hxn, hyn, hxi, hyi]

theorem mul_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true) :
    f.mul rnd x y = f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x * f.toRat y) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  unfold mul; simp [hxn, hyn, hxi, hyi]

theorem fma_of_finite (rnd : RoundMode) {x y z : ℕ} (hx : f.isFinite x = true)
    (hy : f.isFinite y = true) (hz : f.isFinite z = true) :
    f.fma rnd x y z = f.roundRat rnd false ((f.sign x != f.sign y) && f.sign z)
      (f.toRat x * f.toRat y + f.toRat z) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  obtain ⟨hzn, hzi⟩ := (isFinite_iff z).mp hz
  unfold fma; simp [hxn, hyn, hzn, hxi, hyi, hzi]

theorem div_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true)
    (hy0 : f.isZero y = false) :
    f.div rnd x y = f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x / f.toRat y) := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  obtain ⟨hyn, hyi⟩ := (isFinite_iff y).mp hy
  unfold div; simp [hxn, hyn, hxi, hyi, hy0]

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

theorem sign_neg (x : ℕ) : f.sign (f.neg x) = !f.sign x := sign_setSign _ _

theorem mag_neg (x : ℕ) : f.mag (f.neg x) = f.mag x := mag_setSign _ _

theorem neg_lt (x : ℕ) : f.neg x < 2 ^ f.width := setSign_lt _ _

theorem toRat_neg (x : ℕ) : f.toRat (f.neg x) = -f.toRat x := by
  unfold neg; rw [toRat_setSign, toRat_eq]
  cases f.sign x <;> simp

theorem isFinite_neg (x : ℕ) : f.isFinite (f.neg x) = f.isFinite x := isFinite_setSign _ _
theorem isNaN_neg (x : ℕ) : f.isNaN (f.neg x) = f.isNaN x := isNaN_setSign _ _
theorem isInf_neg (x : ℕ) : f.isInf (f.neg x) = f.isInf x := isInf_setSign _ _
theorem isZero_neg (x : ℕ) : f.isZero (f.neg x) = f.isZero x := isZero_setSign _ _

theorem sub_of_finite (rnd : RoundMode) {x y : ℕ} (hx : f.isFinite x = true) (hy : f.isFinite y = true) :
    f.sub rnd x y = f.roundRat rnd false (f.sign x && !f.sign y) (f.toRat x - f.toRat y) := by
  unfold sub
  rw [add_of_finite rnd hx (by rw [isFinite_neg]; exact hy), sign_neg, toRat_neg, sub_eq_add_neg]

/-! ## Comparison -/

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

theorem wider_binary16_binary32 : Wider binary16 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, binary16, binary32]⟩
theorem wider_bfloat16_binary32 : Wider bfloat16 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, bfloat16, binary32]⟩
theorem wider_binary32_binary64 : Wider binary32 binary64 :=
  ⟨by decide, by decide, maxVal_le_of_top (by decide) (by decide) (by decide)⟩
theorem wider_binary16_binary64 : Wider binary16 binary64 :=
  ⟨by decide, by decide, maxVal_le_of_top (by decide) (by decide) (by decide)⟩
theorem wider_bfloat16_binary64 : Wider bfloat16 binary64 :=
  ⟨by decide, by decide, maxVal_le_of_top (by decide) (by decide) (by decide)⟩
theorem wider_e5m2_binary16 : Wider e5m2 binary16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e5m2, binary16]⟩
theorem wider_e4m3_binary16 : Wider e4m3 binary16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e4m3, binary16]⟩
theorem wider_e5m2_bfloat16 : Wider e5m2 bfloat16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e5m2, bfloat16]⟩
theorem wider_e4m3_bfloat16 : Wider e4m3 bfloat16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e4m3, bfloat16]⟩
theorem wider_e5m2_binary32 : Wider e5m2 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e5m2, binary32]⟩
theorem wider_e4m3_binary32 : Wider e4m3 binary32 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e4m3, binary32]⟩
theorem wider_e2m1_bfloat16 : Wider e2m1 bfloat16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e2m1, bfloat16]⟩
theorem wider_e2m1_binary16 : Wider e2m1 binary16 :=
  ⟨by decide, by decide, by rw [maxVal_eq, maxVal_eq]; norm_num [topSig, topExp, bias, e2m1, binary16]⟩


/-! ## Square root -/

/-- **`√x` to nearest** (`sqrt .rn`) of a finite pattern with a nonnegative value: a finite pattern,
with a nonnegative value, within `√x·u + η` of `√x` (`sqrt(−0) = −0`, value `0`). -/
theorem sqrt_rn_err (hf : f.Regular) {x : ℕ} (hx : f.isFinite x = true) (h0 : 0 ≤ f.toRat x) :
    f.sqrt .rn x < 2 ^ f.width ∧ f.isFinite (f.sqrt .rn x) = true ∧ 0 ≤ f.toRat (f.sqrt .rn x) ∧
      |(f.toRat (f.sqrt .rn x) : ℝ) - Real.sqrt (f.toRat x)| ≤ Real.sqrt (f.toRat x) * f.u + f.eta := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  have hu0 : (0 : ℝ) ≤ (f.u : ℝ) := by exact_mod_cast u_nonneg
  have he0 : (0 : ℝ) ≤ (f.eta : ℝ) := by exact_mod_cast eta_nonneg
  by_cases hz : f.isZero x = true
  · have hv : f.toRat x = 0 := toRat_of_isZero hz
    have hs : f.sqrt .rn x = f.zeroPat (f.sign x) := by unfold sqrt; simp [hxn, hz]
    have hzv : f.toRat (f.zeroPat (f.sign x)) = 0 := by
      unfold zeroPat; rw [toRat_setSign]; unfold mag; simp [magVal_zero]
    rw [hs, hzv, hv]
    refine ⟨setSign_lt _ _, by unfold zeroPat; rw [isFinite_setSign]; exact isFinite_zero hf, le_rfl, ?_⟩
    simp only [Rat.cast_zero, Real.sqrt_zero, sub_zero, abs_zero, zero_mul, zero_add]
    exact he0
  · have hz' : f.isZero x = false := by simpa using hz
    have hpos : 0 < f.toRat x := lt_of_le_of_ne h0 (fun h => hz (toRat_eq_zero_iff.mp h.symm))
    have hsx : f.sign x = false := by
      by_contra hs
      have hs' : f.sign x = true := by simpa using hs
      have hm : 0 < f.mag x := by
        rcases Nat.eq_zero_or_pos (f.mag x) with h | h
        · exfalso; apply hz; unfold isZero; simp [h]
        · exact h
      have : f.toRat x < 0 := by
        rw [toRat_eq, ite_eq_left hs']; exact neg_neg_of_pos (magVal_pos hm)
      linarith
    have hs : f.sqrt .rn x = f.sqrtMag .rn (f.toRat x) := by unfold sqrt; simp [hxn, hz', hsx, hxi]
    have hmax : f.toRat x ≤ f.maxVal := (le_abs_self _).trans (abs_toRat_le_maxVal hf hx)
    obtain ⟨hle, herr⟩ := sqrtMag_rn_spec hf hpos hmax
    have hlt : f.sqrtMag .rn (f.toRat x) < f.signBit := hle.trans_lt (maxMag_lt hf)
    have hmag : f.mag (f.sqrtMag .rn (f.toRat x)) = f.sqrtMag .rn (f.toRat x) := mag_of_lt hlt
    have hval : f.toRat (f.sqrtMag .rn (f.toRat x)) = f.magVal (f.sqrtMag .rn (f.toRat x)) := by
      have := toRat_mag (f := f) (f.sqrtMag .rn (f.toRat x)); rwa [hmag] at this
    rw [hs, hval]
    exact ⟨hlt.trans signBit_lt, isFinite_of_mag_le hf (by rw [hmag]; exact hle), magVal_nonneg _, herr⟩

end TVA.Math.Float.Format
