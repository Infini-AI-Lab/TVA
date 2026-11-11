import Claim
import TVA.Proof
import TVA

/-!
# Facts about the constants, `corr`, and the reference values

The kernel's constants are `float32` bit patterns; the claim's `refMax`/`side`/`refSum` are real
functions with a special case at `−∞`. This file relates them to the library's `FloatV` facts.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## The constants -/

/-- The claim's `-∞` constant is `FloatV.negInf`. -/
theorem negInfC_eq_negInf : (negInfC : F32) = FloatV.negInf := by decide

/-- The claim's `+∞` constant is `FloatV.posInf`. -/
theorem posInfC_eq_posInf : (posInfC : F32) = FloatV.posInf := by decide

/-- The claim's `0.0` constant is `ofPat 0`. -/
theorem zeroC_eq : (zeroC : F32) = FloatV.ofPat 0 := by decide

/-- The `bfloat16` zero is `ofPat 0`. -/
theorem zeroB_eq : (zeroB : BF16) = FloatV.ofPat (0 : Nat) := rfl

/-- `-∞` is not finite. -/
theorem negInfC_isFinite : (negInfC : F32).isFinite = false := by decide

/-- `-∞` is not NaN. -/
theorem negInfC_isNaN : (negInfC : F32).isNaN = false := by decide

/-- `+∞` is not NaN. -/
theorem posInfC_isNaN : (posInfC : F32).isNaN = false := by decide

/-- `0.0` is finite. -/
theorem zeroC_isFinite : (zeroC : F32).isFinite = true := by decide

/-- `0.0` is the rational `0`. -/
theorem zeroC_toRat : (zeroC : F32).toRat = 0 := by
  rw [zeroC_eq]
  exact Format.toRat_of_isZero (f := Format.binary32) (by decide)

/-- `0.0` is the real `0`. -/
theorem zeroC_toReal : (zeroC : F32).toReal = 0 := by
  rw [FloatV.toReal, zeroC_toRat]; norm_num

/-- The `bfloat16` zero is not NaN. -/
theorem zeroB_isNaN : (zeroB : BF16).isNaN = false := by decide

/-- The `bfloat16` zero is the rational `0`. -/
theorem zeroB_toRat : (zeroB : BF16).toRat = 0 := by
  rw [zeroB_eq]
  exact Format.toRat_of_isZero (f := Format.bfloat16) (by decide)

/-- `-∞` is not `+∞`. -/
theorem negInfC_ne_posInfC : (negInfC : F32) ≠ posInfC := by decide

/-! ## Helper facts -/

/-- `-∞` is not zero. -/
theorem negInfC_isZero : (negInfC : F32).isZero = false := by decide

/-- The magnitude of an infinite pattern is `infMag`. -/
theorem mag_eq_infMag_of_isInf {f : Format} (hs : f.specials = .ieee) {n : ℕ}
    (h : f.isInf n = true) : f.mag n = f.infMag := by
  unfold Format.isInf at h
  rw [hs] at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨h1, h2⟩ := h
  simp only [Format.expField, Format.fracField] at h1 h2
  have hdm := Nat.div_add_mod (f.mag n) (2 ^ f.m)
  rw [h1, h2] at hdm
  unfold Format.infMag
  rw [← hdm, Nat.add_zero]
  exact Nat.mul_comm _ _

/-- An infinity with the sign bit set is `-∞`. -/
theorem eq_negInf_of_isInf_sign {f : Format} (hs : f.specials = .ieee) {p : FloatV f}
    (hinf : p.isInf = true) (hsign : p.sign = true) : p = FloatV.negInf := by
  apply FloatV.ext_pat
  rw [FloatV.negInf_pat]
  have hmag := mag_eq_infMag_of_isInf hs hinf
  have hsign' : f.sign p.pat = true := hsign
  have h1 : f.setSign (f.sign p.pat) (f.mag p.pat) = p.pat := by
    have hset : f.setSign (f.sign p.pat) (f.mag p.pat) = f.setSign (f.sign p.pat) p.pat := by
      simp [Format.setSign, Format.mag_mag]
    rw [hset, Format.eq_setSign (FloatV.pat_lt p)]
  calc p.pat = f.setSign (f.sign p.pat) (f.mag p.pat) := h1.symm
    _ = f.setSign true f.infMag := by rw [hsign', hmag]

/-- An infinity without the sign bit is `+∞`. -/
theorem eq_posInf_of_isInf_sign {f : Format} (hs : f.specials = .ieee) {p : FloatV f}
    (hinf : p.isInf = true) (hsign : p.sign = false) : p = FloatV.posInf := by
  apply FloatV.ext_pat
  rw [FloatV.posInf_pat]
  have hmag := mag_eq_infMag_of_isInf hs hinf
  have hsign' : f.sign p.pat = false := hsign
  have h1 : f.setSign (f.sign p.pat) (f.mag p.pat) = p.pat := by
    have hset : f.setSign (f.sign p.pat) (f.mag p.pat) = f.setSign (f.sign p.pat) p.pat := by
      simp [Format.setSign, Format.mag_mag]
    rw [hset, Format.eq_setSign (FloatV.pat_lt p)]
  calc p.pat = f.setSign (f.sign p.pat) (f.mag p.pat) := h1.symm
    _ = f.setSign false f.infMag := by rw [hsign', hmag]

/-- An infinite, non-NaN value that is not `+∞` is `−∞`. -/
theorem eq_negInfC_of_isInf {p : F32} (hinf : p.isInf = true) (hne : p ≠ posInfC) :
    p = negInfC := by
  cases hs : p.sign
  · have h := eq_posInf_of_isInf_sign (f := Format.binary32) (by decide) hinf hs
    exact absurd (h.trans posInfC_eq_posInf.symm) hne
  · exact (eq_negInf_of_isInf_sign (f := Format.binary32) (by decide) hinf hs).trans
      negInfC_eq_negInf.symm

/-- The value `-∞` stands for in the rational reading: `-2^(bias+1)`. -/
theorem negInf_toRat_general {f : Format} (hf : f.Regular) (hs : f.specials = .ieee) :
    (FloatV.negInf : FloatV f).toRat = -(2 : ℚ) ^ (f.bias + 1) := by
  have hinf : f.isInf (f.setSign true f.infMag) = true := by
    rw [Format.isInf_setSign, Format.isInf_infMag hs]
  unfold FloatV.toRat
  rw [FloatV.negInf_pat, Format.toRat_of_isInf hf hinf, Format.sign_setSign]
  simp

/-- The value `+∞` stands for in the rational reading: `2^(bias+1)`. -/
theorem posInf_toRat_general {f : Format} (hf : f.Regular) (hs : f.specials = .ieee) :
    (FloatV.posInf : FloatV f).toRat = (2 : ℚ) ^ (f.bias + 1) := by
  have hinf : f.isInf (f.setSign false f.infMag) = true := by
    rw [Format.isInf_setSign, Format.isInf_infMag hs]
  unfold FloatV.toRat
  rw [FloatV.posInf_pat, Format.toRat_of_isInf hf hinf, Format.sign_setSign]
  simp

/-- The rational reading of `-∞` in `float32`. -/
theorem negInfC_toRat : (negInfC : F32).toRat = -(2 : ℚ) ^ (Format.binary32.bias + 1) := by
  rw [negInfC_eq_negInf]
  exact negInf_toRat_general Format.binary32_regular (by decide)

/-- The rational reading of `+∞` in `float32`. -/
theorem posInfC_toRat : (posInfC : F32).toRat = (2 : ℚ) ^ (Format.binary32.bias + 1) := by
  rw [posInfC_eq_posInf]
  exact posInf_toRat_general Format.binary32_regular (by decide)

/-- A non-NaN value that is not finite is infinite. -/
theorem isInf_of_not_isFinite {f : Format} {x : FloatV f} (hn : x.isNaN = false)
    (hf : x.isFinite = false) : x.isInf = true := by
  by_contra h
  have h' : x.isInf = false := by cases hx : x.isInf <;> simp_all
  have hfin : x.isFinite = true := by
    rw [FloatV.isFinite, Format.isFinite_iff]
    exact ⟨hn, h'⟩
  rw [hfin] at hf
  exact Bool.false_ne_true hf.symm

/-- `max(x, -∞) = x` for `x` not NaN and not `-∞`. -/
theorem max_negInf_right' {f : Format} (hf : f.Regular) (hs : f.specials = .ieee)
    {x : FloatV f} (hx : x.isNaN = false) (hne : x ≠ FloatV.negInf) :
    FloatV.max x FloatV.negInf = x := by
  have hni : (FloatV.negInf : FloatV f).isNaN = false := FloatV.negInf_isNaN hs
  rcases FloatV.max_mem hx hni with h | h
  · exact h
  · exfalso
    have h1 : (FloatV.max x FloatV.negInf).toRat = x.toRat := by
      rw [FloatV.toRat_max hx hni]
      exact max_eq_left (FloatV.negInf_toRat_le hf hs hx).1
    rw [h] at h1
    exact hne ((FloatV.negInf_toRat_le hf hs hx).2 h1)

/-- `max(-∞, y) = y` for `y` not NaN. -/
theorem max_negInfC_left {y : F32} (hy : y.isNaN = false) : FloatV.max negInfC y = y := by
  rw [negInfC_eq_negInf]
  exact FloatV.max_negInf_left (f := Format.binary32) Format.binary32_regular (by decide) hy

/-- `max(x, -∞) = x` for `x` not NaN and not `-∞`. -/
theorem max_negInfC_right {x : F32} (hx : x.isNaN = false) (hne : x ≠ negInfC) :
    FloatV.max x negInfC = x := by
  rw [negInfC_eq_negInf] at hne ⊢
  exact max_negInf_right' Format.binary32_regular (by decide) hx hne

/-! ## The conversion of `zeroC` into `bfloat16` -/

/-- `0.0` converts to the `bfloat16` zero. -/
theorem cvtWith_zeroC : (FloatV.cvtWith .rn (zeroC : F32) : BF16) = zeroB := by
  rw [zeroC_eq]
  unfold FloatV.cvtWith
  rw [FloatV.pat_ofPat (by decide : (0 : Nat) < 2 ^ Format.binary32.width)]
  rw [Format.cvt_of_finite (g := Format.bfloat16) (rnd := .rn) (sat := false)
    (by decide : Format.binary32.isFinite 0 = true)]
  have hz : Format.binary32.toRat 0 = 0 :=
    Format.toRat_of_isZero (f := Format.binary32) (by decide)
  have hs : Format.binary32.sign 0 = false := by decide
  rw [hz, hs, Format.roundRat_zero]
  rw [zeroB_eq]
  congr 1

/-! ## `corr` -/

/-- `corr` leaves a value other than `+∞` alone. -/
theorem corr_of_ne {p : F32} (h : p ≠ posInfC) : corr p = p := by simp [corr, h]

/-- `corr` of a non-NaN value is not NaN. -/
theorem corr_isNaN {p : F32} (hp : p.isNaN = false) : (corr p).isNaN = false := by
  unfold corr
  split_ifs with h
  · exact negInfC_isNaN
  · exact hp

/-- `corr x` is never `+∞`. -/
theorem corr_ne_posInf (x : F32) : corr x ≠ posInfC := by
  unfold corr
  split_ifs with h
  · exact negInfC_ne_posInfC
  · exact h

/-- A non-NaN `corr p` is either finite or exactly `−∞`. -/
theorem corr_finite_or {p : F32} (hp : p.isNaN = false) :
    (corr p).isFinite = true ∨ corr p = negInfC := by
  unfold corr
  split_ifs with h
  · exact Or.inr rfl
  · by_cases hf : p.isFinite = true
    · exact Or.inl hf
    · refine Or.inr (eq_negInfC_of_isInf ?_ h)
      exact isInf_of_not_isFinite hp (by
        cases hh : p.isFinite
        · rfl
        · exact absurd hh hf)

/-- `FloatV.eq` against `+∞` is equality, for a non-NaN operand. -/
theorem eq_posInfC_iff {p : F32} (hp : p.isNaN = false) :
    FloatV.eq p posInfC = true ↔ p = posInfC := by
  constructor
  · intro h
    have hk : (2 : ℚ) ^ (Format.binary32.bias + 1) = Format.binary32.toRat posInfC.pat :=
      posInfC_toRat.symm
    have hrat : Format.binary32.toRat p.pat = (2 : ℚ) ^ (Format.binary32.bias + 1) := by
      rw [hk]
      exact (FloatV.eq_iff hp posInfC_isNaN).mp h
    have hpos : (0 : ℚ) < (2 : ℚ) ^ (Format.binary32.bias + 1) := by positivity
    have hinf : p.isInf = true := by
      by_contra h'
      have hf' : p.isInf = false := by cases hh : p.isInf <;> simp_all
      have hfin : p.isFinite = true := by
        rw [FloatV.isFinite, Format.isFinite_iff]
        exact ⟨hp, hf'⟩
      have hle := Format.abs_toRat_le_maxVal (f := Format.binary32) (by decide) hfin
      rw [hrat] at hle
      rw [abs_of_nonneg hpos.le] at hle
      have hlt := Format.maxVal_lt_inf (f := Format.binary32) (by decide) (by decide)
      linarith
    have hval : Format.binary32.toRat p.pat =
        if Format.binary32.sign p.pat then -(2 : ℚ) ^ (Format.binary32.bias + 1)
        else (2 : ℚ) ^ (Format.binary32.bias + 1) :=
      Format.toRat_of_isInf (f := Format.binary32) (by decide) hinf
    have hsign : Format.binary32.sign p.pat = false := by
      cases hs : Format.binary32.sign p.pat
      · rfl
      · exfalso
        rw [hval, hs] at hrat
        simp at hrat
        linarith
    exact (eq_posInf_of_isInf_sign (f := Format.binary32) (by decide) hinf hsign).trans
      posInfC_eq_posInf.symm
  · intro h
    rw [h]
    exact (FloatV.eq_iff posInfC_isNaN posInfC_isNaN).mpr rfl

/-- `FloatV.eq` against `−∞` is equality, for a non-NaN operand. -/
theorem eq_negInfC_iff {p : F32} (hp : p.isNaN = false) :
    FloatV.eq p negInfC = true ↔ p = negInfC := by
  constructor
  · intro h
    have hk : -(2 : ℚ) ^ (Format.binary32.bias + 1) = Format.binary32.toRat negInfC.pat :=
      negInfC_toRat.symm
    have hrat : Format.binary32.toRat p.pat = -(2 : ℚ) ^ (Format.binary32.bias + 1) := by
      rw [hk]
      exact (FloatV.eq_iff hp negInfC_isNaN).mp h
    have hpos : (0 : ℚ) < (2 : ℚ) ^ (Format.binary32.bias + 1) := by positivity
    have hinf : p.isInf = true := by
      by_contra h'
      have hf' : p.isInf = false := by cases hh : p.isInf <;> simp_all
      have hfin : p.isFinite = true := by
        rw [FloatV.isFinite, Format.isFinite_iff]
        exact ⟨hp, hf'⟩
      have hle := Format.abs_toRat_le_maxVal (f := Format.binary32) (by decide) hfin
      rw [hrat, abs_neg, abs_of_nonneg hpos.le] at hle
      have hlt := Format.maxVal_lt_inf (f := Format.binary32) (by decide) (by decide)
      linarith
    have hval : Format.binary32.toRat p.pat =
        if Format.binary32.sign p.pat then -(2 : ℚ) ^ (Format.binary32.bias + 1)
        else (2 : ℚ) ^ (Format.binary32.bias + 1) :=
      Format.toRat_of_isInf (f := Format.binary32) (by decide) hinf
    have hsign : Format.binary32.sign p.pat = true := by
      cases hs : Format.binary32.sign p.pat
      · exfalso
        rw [hval, hs] at hrat
        simp at hrat
        linarith
      · rfl
    exact (eq_negInf_of_isInf_sign (f := Format.binary32) (by decide) hinf hsign).trans
      negInfC_eq_negInf.symm
  · intro h
    rw [h]
    exact (FloatV.eq_iff negInfC_isNaN negInfC_isNaN).mpr rfl

/-- The kernel's clamp of a non-NaN `+∞` to `−∞` is `corr`. -/
theorem ite_corr {p : F32} (hp : p.isNaN = false) :
    (if FloatV.eq p posInfC then negInfC else p) = corr p := by
  by_cases h : p = posInfC
  · have heq : FloatV.eq p posInfC = true := (eq_posInfC_iff hp).mpr h
    unfold corr
    rw [heq, ite_eq_left h]
    simp
  · have heq : FloatV.eq p posInfC = false := by
      cases hh : FloatV.eq p posInfC
      · rfl
      · exact absurd ((eq_posInfC_iff hp).mp hh) h
    unfold corr
    rw [heq, ite_eq_right h]
    simp

/-! ## The reference values -/

/-- An empty side (`-∞`) weighs `0`. -/
theorem side_negInf (m : ℝ) : side negInfC m = 0 := by
  simp [side]

/-- A side other than `-∞` weighs `exp(x - m)`. -/
theorem side_of_ne {x : F32} (h : x ≠ negInfC) (m : ℝ) : side x m = Real.exp (x.toReal - m) := by
  simp [side, h]

/-- The reference maximum with an empty first side is the second. -/
theorem refMax_negInf_left (s : F32) : refMax negInfC s = s.toReal := by
  simp [refMax]

/-- The reference maximum with an empty second side is the first. -/
theorem refMax_negInf_right {p : F32} (h : p ≠ negInfC) : refMax p negInfC = p.toReal := by
  simp [refMax, h]

/-- With no side empty, the reference maximum is the real maximum. -/
theorem refMax_eq_max {p s : F32} (hp : p ≠ negInfC) (hs : s ≠ negInfC) :
    refMax p s = max p.toReal s.toReal := by
  simp [refMax, hp, hs]

/-- **The maximum of the two corrected values, as a real.** -/
theorem maxRel_toReal_refMax {x y M : F32} (hM : FloatV.MaxRel true x y M) (hx : x.isNaN = false)
    (hy : y.isNaN = false) (_hxf : x.isFinite = true ∨ x = negInfC)
    (_hyf : y.isFinite = true ∨ y = negInfC) (_hne : ¬ (x = negInfC ∧ y = negInfC)) :
    M.toReal = refMax x y := by
  rcases hM with hM | ⟨hzx, hzy, hMxy⟩
  · simp only [ite_true] at hM
    have hmem : FloatV.max x y = x ∨ FloatV.max x y = y := FloatV.max_mem hx hy
    have hnan : (FloatV.max x y).isNaN = false := FloatV.isNaN_of_mem hx hy hmem
    have hM' : M = FloatV.max x y := hM.eq hnan
    rw [hM']
    by_cases hxinf : x = negInfC
    · rw [hxinf]
      rw [refMax_negInf_left]
      rw [max_negInfC_left hy]
    · by_cases hyinf : y = negInfC
      · rw [hyinf]
        rw [refMax_negInf_right hxinf]
        rw [max_negInfC_right hx hxinf]
      · rw [refMax_eq_max hxinf hyinf]
        rw [FloatV.toReal, FloatV.toReal, FloatV.toReal, FloatV.toRat_max hx hy]
        exact Rat.cast_max _ _
  · have hx0 : x.toRat = 0 := Format.toRat_of_isZero (f := Format.binary32) hzx
    have hy0 : y.toRat = 0 := Format.toRat_of_isZero (f := Format.binary32) hzy
    have hxr : x.toReal = 0 := by rw [FloatV.toReal, hx0]; norm_num
    have hyr : y.toReal = 0 := by rw [FloatV.toReal, hy0]; norm_num
    have hxne : x ≠ negInfC := by
      intro h
      rw [h, negInfC_isZero] at hzx
      exact Bool.false_ne_true hzx
    have hyne : y ≠ negInfC := by
      intro h
      rw [h, negInfC_isZero] at hzy
      exact Bool.false_ne_true hzy
    rw [refMax_eq_max hxne hyne, hxr, hyr]
    rcases hMxy with hMx | hMy
    · rw [hMx, hxr]
      norm_num
    · rw [hMy, hyr]
      norm_num

/-- **Both empty.** -/
theorem maxRel_eq_negInf_both {M : F32} (hM : FloatV.MaxRel true negInfC negInfC M) :
    M = negInfC := by
  rcases hM with hM | ⟨hz, -, -⟩
  · simp only [ite_true] at hM
    have hm : FloatV.max negInfC negInfC = negInfC := max_negInfC_left negInfC_isNaN
    have hn : (FloatV.max negInfC negInfC).isNaN = false := by rw [hm]; exact negInfC_isNaN
    exact (hM.eq hn).trans hm
  · rw [negInfC_isZero] at hz
    exact absurd hz Bool.false_ne_true

/-- **One side empty.** -/
theorem maxRel_eq_of_negInf_left {y M : F32} (hM : FloatV.MaxRel true negInfC y M)
    (hy : y.isNaN = false) (_hy' : y ≠ negInfC) : M = y := by
  rcases hM with hM | ⟨hz, -, -⟩
  · simp only [ite_true] at hM
    have hm : FloatV.max negInfC y = y := max_negInfC_left hy
    have hn : (FloatV.max negInfC y).isNaN = false := by rw [hm]; exact hy
    exact (hM.eq hn).trans hm
  · rw [negInfC_isZero] at hz
    exact absurd hz Bool.false_ne_true

/-- A `tl.maximum` against `-∞` of a value not NaN and not `-∞` is that value. -/
theorem maxRel_eq_of_negInf_right {x M : F32} (hM : FloatV.MaxRel true x negInfC M)
    (hx : x.isNaN = false) (hx' : x ≠ negInfC) : M = x := by
  rcases hM with hM | ⟨-, hz, -⟩
  · simp only [ite_true] at hM
    have hm : FloatV.max x negInfC = x := max_negInfC_right hx hx'
    have hn : (FloatV.max x negInfC).isNaN = false := by rw [hm]; exact hx
    exact (hM.eq hn).trans hm
  · rw [negInfC_isZero] at hz
    exact absurd hz Bool.false_ne_true

/-- A finite `FloatV` is bounded by its format's `maxVal`. -/
theorem finite_abs_le_maxVal {x : F32} (hx : x.isFinite = true) :
    |x.toRat| ≤ Format.binary32.maxVal :=
  Format.abs_toRat_le_maxVal (f := Format.binary32) (by decide) hx

end Claims.MergeAttnStates
