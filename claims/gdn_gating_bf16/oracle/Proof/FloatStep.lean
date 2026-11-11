import Claim
import TVA.Proof
import Proof.Bridge

/-!
# Float-level stepping lemmas

Each `Safe` rule hands a continuation the result of one `tl.*` operation up to `NaNEq` (equal, or
both NaN).  These lemmas turn such a `NaNEq` fact into the shape the real-arithmetic chains of
`Proof.BetaChain`, `Proof.Softplus`, `Proof.GMain` and `Proof.GLow` want: the result is that
operation's exact rounding, so it carries the library's error bound, and an overflowing result is
exactly `±∞`.
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-! ## NaN-or-equal, and finiteness -/

/-- Finiteness transfers across a `NaNEq` fact whose left side is finite. -/
theorem fin_of_naneq_fin {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hx : x.isFinite = true) : y.isFinite = true :=
  (FloatV.NaNEq.exact_of_isFinite h hx).1

/-! ## Casting a rational error bound to `ℝ` -/

/-- A rational error bound `|a − b| ≤ |b|·u + e` casts to the same bound over `ℝ`. -/
private theorem rat_abs_le_mul_add {a b u e : ℚ} (h : |a - b| ≤ |b| * u + e) :
    |(a : ℝ) - (b : ℝ)| ≤ |(b : ℝ)| * (u : ℝ) + (e : ℝ) := by
  have h' : ((|a - b| : ℚ) : ℝ) ≤ ((|b| * u + e : ℚ) : ℝ) := by
    exact_mod_cast h
  rwa [Rat.cast_abs, Rat.cast_sub, Rat.cast_add, Rat.cast_mul, Rat.cast_abs] at h'

/-! ## Helper lemmas for the overflow steps -/

/-- Non-finiteness transfers backwards across a `NaNEq` fact. -/
private theorem isFinite_false_of_naneq {f : Format} {x y : FloatV f}
    (h : FloatV.NaNEq x y) (hy : y.isFinite = false) : x.isFinite = false := by
  rcases h with h | ⟨h1, h2⟩
  · rw [h]; exact hy
  · cases hxt : x.isFinite with
    | false => rfl
    | true =>
      exfalso
      have hh := (FloatV.isFinite_iff (x := x)).mp hxt
      rw [h1] at hh
      exact Bool.noConfusion hh.1

/-- The overflow pattern, for any sign of a zero result. -/
private theorem roundRat_overflow_neg_any {q : ℚ} (zs : Bool) (hq : q < 0)
    (hover : ¬ (Format.roundSig RoundMode.rn (-q)
      (Format.binary32.qexp (-q)) : ℚ) * Format.pow2 (Format.binary32.qexp (-q)) ≤
      Format.binary32.maxVal) :
    Format.binary32.roundRat RoundMode.rn false zs q =
      Format.binary32.setSign true Format.binary32.infMag := by
  have hq0 : q ≠ 0 := ne_of_lt hq
  rw [Format.roundRat_of_ne hq0, ite_eq_left hq, abs_of_neg hq]
  unfold Format.roundMag Format.encode
  rw [ite_eq_right hover]
  simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
  rw [show Format.binary32.specials = Format.Specials.ieee from rfl]
  simp only
  decide +kernel

/-- A nonzero rational whose rounded value is in range rounds to a finite signed pattern. -/
private theorem roundRat_rn_of_no_over_any {q : ℚ} (zs : Bool) (hq0 : q ≠ 0)
    (hle : (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|) : ℚ) *
      Format.pow2 (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal) :
    Format.binary32.roundRat RoundMode.rn false zs q =
        Format.binary32.setSign (decide (q < 0))
          (Format.binary32.pack (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|))
            (Format.binary32.qexp |q|)) ∧
      Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false zs q) = true := by
  have hf : Format.binary32.Regular := Format.binary32_regular
  have ha : (0 : ℚ) < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.binary32) RoundMode.rn ha
  have hle' : (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal := by
    rwa [Format.pow2_eq] at hle
  have hmag : Format.binary32.roundMag RoundMode.rn false |q| =
      Format.binary32.pack (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|))
        (Format.binary32.qexp |q|) := by
    unfold Format.roundMag Format.encode
    exact ite_eq_left hle
  have hple : Format.binary32.pack (Format.roundSig RoundMode.rn |q|
        (Format.binary32.qexp |q|)) (Format.binary32.qexp |q|) ≤
      Format.binary32.maxMag := Format.pack_le_maxMag hf hN hQ hsub hle'
  have hltp : Format.binary32.pack (Format.roundSig RoundMode.rn |q|
        (Format.binary32.qexp |q|)) (Format.binary32.qexp |q|) <
      Format.binary32.signBit := hple.trans_lt (Format.maxMag_lt hf)
  refine ⟨?_, ?_⟩
  · rw [Format.roundRat_of_ne hq0, hmag, Format.setSign_of_lt hltp]
    by_cases h : q < 0 <;> simp [h]
  · have hs : (if q < 0 then Format.binary32.signBit else 0) +
        Format.binary32.pack (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|))
          (Format.binary32.qexp |q|) =
        Format.binary32.setSign (decide (q < 0))
          (Format.binary32.pack (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|))
            (Format.binary32.qexp |q|)) := by
      unfold Format.setSign
      rw [Format.mag_of_lt hltp]
      by_cases h : q < 0 <;> simp [h]
    rw [Format.roundRat_of_ne hq0, hmag, hs, Format.isFinite_setSign]
    exact Format.isFinite_of_mag_le hf (by rw [Format.mag_of_lt hltp]; exact hple)

/-- A `NaNEq` partner of `-∞` is `-∞`. -/
private theorem eq_negInf_of_naneq {z r : F32} (hz : z = (FloatV.negInf : F32))
    (h : FloatV.NaNEq z r) : r = (FloatV.negInf : F32) := by
  subst hz
  have hnan : (FloatV.negInf : F32).isNaN = false :=
    FloatV.negInf_isNaN (show Format.binary32.specials = Format.Specials.ieee from rfl)
  rcases h with h | ⟨h1, h2⟩
  · exact h.symm
  · rw [hnan] at h1
    exact Bool.noConfusion h1

/-! ## `tl.add`, `tl.mul`

Every lemma below carries finiteness hypotheses on both operands: without them the statements
are false, since a non-finite operand can overflow or produce a NaN. -/

/-- A finite `x + y` (both operands finite) is within one rounding of `x + y`. -/
theorem add_exact_err_fin {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hroom : |x.toRat + y.toRat| ≤ Format.binary32.maxVal)
    (h : FloatV.NaNEq (FloatV.add x y) r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤
        |x.toReal + y.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  obtain ⟨hfin, herr⟩ := FloatV.add_err Format.binary32_regular hx hy hroom
  rw [Format.binary32_u, Format.binary32_eta] at herr
  have hr : r = FloatV.add x y := FloatV.NaNEq.eq_of_isFinite h hfin
  subst hr
  refine ⟨hfin, ?_⟩
  simp only [FloatV.toReal]
  have key := rat_abs_le_mul_add (a := (FloatV.add x y).toRat) (b := x.toRat + y.toRat)
    (u := 2 ^ (-24 : ℤ)) (e := 2 ^ (-150 : ℤ)) herr
  simpa using key

/-- A finite `x * y` (both operands finite) is within one rounding of `x * y`. -/
theorem mul_exact_err_fin {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hroom : |x.toRat * y.toRat| ≤ Format.binary32.maxVal)
    (h : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal * y.toReal)| ≤
        |x.toReal * y.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  obtain ⟨hfin, herr⟩ := FloatV.mul_err Format.binary32_regular hx hy hroom
  rw [Format.binary32_u, Format.binary32_eta] at herr
  have hr : r = FloatV.mul x y := FloatV.NaNEq.eq_of_isFinite h hfin
  subst hr
  refine ⟨hfin, ?_⟩
  simp only [FloatV.toReal]
  have key := rat_abs_le_mul_add (a := (FloatV.mul x y).toRat) (b := x.toRat * y.toRat)
    (u := 2 ^ (-24 : ℤ)) (e := 2 ^ (-150 : ℤ)) herr
  simpa using key

/-- An overflowing `add` of a negative sum, both operands finite, is `-∞`. -/
theorem add_negInf_fin {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (h : FloatV.NaNEq (FloatV.add x y) r) (hr : r.isFinite = false)
    (hneg : x.toReal + y.toReal < 0) : r = (FloatV.negInf : F32) := by
  have hq : Format.binary32.toRat x.pat + Format.binary32.toRat y.pat < 0 := by
    have h' := hneg
    simp only [FloatV.toReal, FloatV.toRat] at h'
    exact_mod_cast h'
  have hpat : FloatV.add x y = FloatV.ofPat (Format.binary32.roundRat RoundMode.rn false
      (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
      (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat)) := by
    unfold FloatV.add
    rw [Format.add_of_finite RoundMode.rn hx hy]
  have hover : ¬ (Format.roundSig RoundMode.rn
        (-(Format.binary32.toRat x.pat + Format.binary32.toRat y.pat))
        (Format.binary32.qexp (-(Format.binary32.toRat x.pat + Format.binary32.toRat y.pat))) : ℚ) *
        Format.pow2 (Format.binary32.qexp (-(Format.binary32.toRat x.pat + Format.binary32.toRat y.pat)))
        ≤ Format.binary32.maxVal := by
    intro hle
    have hfin : (FloatV.add x y).isFinite = false := isFinite_false_of_naneq h hr
    have hle' : (Format.roundSig RoundMode.rn
          |Format.binary32.toRat x.pat + Format.binary32.toRat y.pat|
          (Format.binary32.qexp |Format.binary32.toRat x.pat + Format.binary32.toRat y.pat|) : ℚ) *
          Format.pow2 (Format.binary32.qexp |Format.binary32.toRat x.pat + Format.binary32.toRat y.pat|)
          ≤ Format.binary32.maxVal := by
      rw [abs_of_neg hq]
      exact hle
    obtain ⟨hrat, hratfin⟩ := roundRat_rn_of_no_over_any
      (zs := Format.binary32.sign x.pat && Format.binary32.sign y.pat)
      (ne_of_lt hq) hle'
    have hltn : Format.binary32.roundRat RoundMode.rn false
        (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat) <
        2 ^ Format.binary32.width := by
      rw [hrat]
      exact Format.setSign_lt _ _
    have hp : (FloatV.add x y).isFinite = Format.binary32.isFinite
        (Format.binary32.roundRat RoundMode.rn false
          (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
          (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat)) := by
      unfold FloatV.add
      rw [Format.add_of_finite RoundMode.rn hx hy, FloatV.isFinite_ofPat hltn]
    rw [hp, hratfin] at hfin
    exact Bool.noConfusion hfin
  have hz : FloatV.add x y = (FloatV.negInf : F32) := by
    rw [hpat, roundRat_overflow_neg_any
      (zs := Format.binary32.sign x.pat && Format.binary32.sign y.pat) hq hover]
    rfl
  exact eq_negInf_of_naneq hz h

/-- An overflowing `mul` of a negative product, both operands finite, is `-∞`. -/
theorem mul_negInf_fin {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (h : FloatV.NaNEq (FloatV.mul x y) r) (hr : r.isFinite = false)
    (hneg : x.toReal * y.toReal < 0) : r = (FloatV.negInf : F32) := by
  have hq : Format.binary32.toRat x.pat * Format.binary32.toRat y.pat < 0 := by
    have h' := hneg
    simp only [FloatV.toReal, FloatV.toRat] at h'
    exact_mod_cast h'
  have hpat : FloatV.mul x y = FloatV.ofPat (Format.binary32.roundRat RoundMode.rn false
      (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
      (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)) := by
    unfold FloatV.mul
    rw [Format.mul_of_finite RoundMode.rn hx hy]
  have hover : ¬ (Format.roundSig RoundMode.rn
        (-(Format.binary32.toRat x.pat * Format.binary32.toRat y.pat))
        (Format.binary32.qexp (-(Format.binary32.toRat x.pat * Format.binary32.toRat y.pat))) : ℚ) *
        Format.pow2 (Format.binary32.qexp (-(Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)))
        ≤ Format.binary32.maxVal := by
    intro hle
    have hfin : (FloatV.mul x y).isFinite = false := isFinite_false_of_naneq h hr
    have hle' : (Format.roundSig RoundMode.rn
          |Format.binary32.toRat x.pat * Format.binary32.toRat y.pat|
          (Format.binary32.qexp |Format.binary32.toRat x.pat * Format.binary32.toRat y.pat|) : ℚ) *
          Format.pow2 (Format.binary32.qexp |Format.binary32.toRat x.pat * Format.binary32.toRat y.pat|)
          ≤ Format.binary32.maxVal := by
      rw [abs_of_neg hq]
      exact hle
    obtain ⟨hrat, hratfin⟩ := roundRat_rn_of_no_over_any
      (zs := Format.binary32.sign x.pat != Format.binary32.sign y.pat)
      (ne_of_lt hq) hle'
    have hltn : Format.binary32.roundRat RoundMode.rn false
        (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat) <
        2 ^ Format.binary32.width := by
      rw [hrat]
      exact Format.setSign_lt _ _
    have hp : (FloatV.mul x y).isFinite = Format.binary32.isFinite
        (Format.binary32.roundRat RoundMode.rn false
          (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
          (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)) := by
      unfold FloatV.mul
      rw [Format.mul_of_finite RoundMode.rn hx hy, FloatV.isFinite_ofPat hltn]
    rw [hp, hratfin] at hfin
    exact Bool.noConfusion hfin
  have hz : FloatV.mul x y = (FloatV.negInf : F32) := by
    rw [hpat, roundRat_overflow_neg_any
      (zs := Format.binary32.sign x.pat != Format.binary32.sign y.pat) hq hover]
    rfl
  exact eq_negInf_of_naneq hz h

/-- An overflowing `mul` of two finite numbers has an exact product past the edge of the format. -/
theorem mul_big_fin {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hr : (FloatV.mul x y).isFinite = false) :
    ((1 : ℝ) - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤ |x.toReal * y.toReal| := by
  have hpat : FloatV.mul x y = FloatV.ofPat (Format.binary32.roundRat RoundMode.rn false
      (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
      (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)) := by
    unfold FloatV.mul
    rw [Format.mul_of_finite RoundMode.rn hx hy]
  have hq : ((1 : ℚ) - 2 ^ (-22 : ℤ)) * Format.binary32.maxVal ≤
      |Format.binary32.toRat x.pat * Format.binary32.toRat y.pat| := by
    by_contra hcon
    rw [not_le] at hcon
    have hmax : |Format.binary32.toRat x.pat * Format.binary32.toRat y.pat| ≤
        Format.binary32.maxVal := by
      have hlt : ((1 : ℚ) - 2 ^ (-22 : ℤ)) * Format.binary32.maxVal <
          Format.binary32.maxVal := by
        have h1 : (0 : ℚ) < 2 ^ (-22 : ℤ) := by norm_num
        have h2 : (0 : ℚ) < Format.binary32.maxVal := Format.maxVal_pos Format.binary32_regular
        nlinarith
      exact (hcon.trans hlt).le
    have hfin := (FloatV.ofPat_roundRat_err (f := Format.binary32) Format.binary32_regular
      false (Format.binary32.sign x.pat != Format.binary32.sign y.pat) hmax).1
    rw [← hpat] at hfin
    rw [hfin] at hr
    exact Bool.noConfusion hr
  have hreal : ((1 : ℝ) - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤
      |(Format.binary32.toRat x.pat : ℝ) * (Format.binary32.toRat y.pat : ℝ)| := by
    have h1 : ((((1 : ℚ) - 2 ^ (-22 : ℤ)) * Format.binary32.maxVal : ℚ) : ℝ) ≤
        ((|Format.binary32.toRat x.pat * Format.binary32.toRat y.pat| : ℚ) : ℝ) := by
      exact_mod_cast hq
    calc ((1 : ℝ) - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ)
        = (((1 : ℚ) - 2 ^ (-22 : ℤ)) * Format.binary32.maxVal : ℚ) := by push_cast; ring
      _ ≤ ((|Format.binary32.toRat x.pat * Format.binary32.toRat y.pat| : ℚ) : ℝ) := h1
      _ = |(Format.binary32.toRat x.pat : ℝ) * (Format.binary32.toRat y.pat : ℝ)| := by
          rw [Rat.cast_abs, Rat.cast_mul]
  have hxr : x.toReal = (Format.binary32.toRat x.pat : ℝ) := by
    simp only [FloatV.toReal, FloatV.toRat]
  have hyr : y.toReal = (Format.binary32.toRat y.pat : ℝ) := by
    simp only [FloatV.toReal, FloatV.toRat]
  rw [hxr, hyr]
  exact hreal

/-- The pattern an overflow rounds to, in a sign-carrying form. -/
theorem roundRat_overflow_neg {q : ℚ} (hq : q < 0)
    (hover : ¬ (Format.roundSig RoundMode.rn (-q)
      (Format.binary32.qexp (-q)) : ℚ) * Format.pow2 (Format.binary32.qexp (-q)) ≤
      Format.binary32.maxVal) :
    Format.binary32.roundRat RoundMode.rn false false q =
      Format.binary32.setSign true Format.binary32.infMag := by
  have hq0 : q ≠ 0 := ne_of_lt hq
  rw [Format.roundRat_of_ne hq0, ite_eq_left hq, abs_of_neg hq]
  unfold Format.roundMag Format.encode
  rw [ite_eq_right hover]
  simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
  rw [show Format.binary32.specials = Format.Specials.ieee from rfl]
  simp only
  decide +kernel

/-! ## `tl.cvt` -/

/-- A widening conversion of a finite `bfloat16` is exact. -/
theorem cvt_up {x : BF16} (hx : x.isFinite = true) :
    (FloatV.cvt (g := Format.binary32) x).isFinite = true ∧
      (FloatV.cvt (g := Format.binary32) x).toReal = x.toReal := by
  obtain ⟨hfin, hv⟩ := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
    Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32
    RoundMode.rn hx
  rw [FloatV.cvtWith_rn] at hfin hv
  refine ⟨hfin, ?_⟩
  simp only [FloatV.toReal]
  rw [hv]

/-- A narrowing conversion to `bfloat16` is within one `bfloat16` rounding. -/
theorem cvt_down {q : F32} {b : BF16} (hq : q.isFinite = true)
    (hmax : |q.toRat| ≤ Format.bfloat16.maxVal)
    (h : FloatV.NaNEq (FloatV.cvtWith RoundMode.rn (g := Format.bfloat16) q) b) :
    b.isFinite = true ∧
      |b.toReal - q.toReal| ≤ |q.toReal| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := by
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular hq hmax
  rw [← FloatV.cvtWith_rn] at hfin herr
  rw [Format.bfloat16_u, Format.bfloat16_eta] at herr
  have hb : b = FloatV.cvtWith RoundMode.rn (g := Format.bfloat16) q := FloatV.NaNEq.eq_of_isFinite h hfin
  subst hb
  refine ⟨hfin, ?_⟩
  simp only [FloatV.toReal]
  have key := rat_abs_le_mul_add
    (a := (FloatV.cvtWith RoundMode.rn (g := Format.bfloat16) q).toRat) (b := q.toRat)
    (u := 2 ^ (-8 : ℤ)) (e := 2 ^ (-134 : ℤ)) herr
  simpa using key

/-! ## `tl.neg` -/

/-- Negation is exact. -/
theorem neg_exact {x r : F32} (hx : x.isFinite = true)
    (h : FloatV.NaNEq (FloatV.neg x) r) : r.isFinite = true ∧ r.toReal = -x.toReal := by
  have hfin : (FloatV.neg x).isFinite = true := by
    rw [FloatV.isFinite_neg]
    exact hx
  have hr : r = FloatV.neg x := FloatV.NaNEq.eq_of_isFinite h hfin
  subst hr
  refine ⟨hfin, ?_⟩
  simp only [FloatV.toReal, FloatV.toRat_neg]
  push_cast
  ring

/-! ## The platform's approximate operations -/

/-- `tl.exp` on a finite input with room. -/
theorem exp_apply {P : Platform} (hacc : Device.ExpF32 P) {x r : F32}
    (hm : P.math .exp .binary32 x r) (hx : x.isFinite = true) (hroom : ExpRoom x.toReal) :
    r.isFinite = true ∧
      |r.toReal - Real.exp x.toReal| ≤
        (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x.toReal|) * Real.exp x.toReal + 2 ^ (-140 : ℤ) := by
  obtain ⟨ref, href, H⟩ := hacc |x.toReal| (abs_nonneg _)
  have href' : ref = Real.exp := by
    have h := href
    rw [show MathFn.ref .exp = some Real.exp from rfl] at h
    exact (Option.some.inj h).symm
  subst href'
  have hmem : x.toReal ∈ Set.Icc (-|x.toReal|) |x.toReal| := by
    rw [Set.mem_Icc]
    exact ⟨neg_abs_le _, le_abs_self _⟩
  have hnd : x.toReal ∈ (MathFn.exp).natDom := Set.mem_univ _
  have hroom' : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x.toReal|)) *
        |Real.exp x.toReal| + 2 ^ (-140 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [abs_of_pos (Real.exp_pos _)]
    exact hroom
  obtain ⟨hfin, herr⟩ := H x r hm hx hmem hnd hroom'
  refine ⟨hfin, ?_⟩
  rw [abs_of_pos (Real.exp_pos _)] at herr
  exact herr

/-- `tl.log` on a finite positive input with room. -/
theorem log_apply {P : Platform} (hacc : Device.LogF32 P) {x r : F32}
    (hm : P.math .log .binary32 x r) (hx : x.isFinite = true) (hpos : 0 < x.toReal)
    (hroom : (1 + 2 ^ (-22 : ℤ)) * |Real.log x.toReal| + 2 ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ)) :
    r.isFinite = true ∧
      |r.toReal - Real.log x.toReal| ≤ 2 ^ (-22 : ℤ) * |Real.log x.toReal| + 2 ^ (-149 : ℤ) := by
  obtain ⟨ref, href, H⟩ := hacc
  have href' : ref = Real.log := by
    have h := href
    rw [show MathFn.ref .log = some Real.log from rfl] at h
    exact (Option.some.inj h).symm
  subst href'
  exact H x r hm hx (Set.mem_univ _) (Set.mem_Ioi.mpr hpos) hroom

/-- `tl.div` with the full-range rule. -/
theorem div_apply {P : Platform} (hfull : Device.DivFull P) {x y r : F32}
    (hm : P.div .binary32 x y r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hy0 : y.toReal ≠ 0) (hq : |x.toReal / y.toReal| ≤ (2 : ℝ) ^ (126 : ℤ)) :
    r.isFinite = true ∧
      |r.toReal - x.toReal / y.toReal| ≤
        2 ^ (-22 : ℤ) * |x.toReal / y.toReal| + 2 ^ (-148 : ℤ) := by
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
  have hreal : |(r.toRat : ℝ) - (x.toRat : ℝ) / (y.toRat : ℝ)| ≤
      2 ^ (-22 : ℤ) * |(x.toRat : ℝ) / (y.toRat : ℝ)| + 2 ^ (-148 : ℤ) := by
    have h1 : ((|r.toRat - x.toRat / y.toRat| : ℚ) : ℝ) ≤
        ((2 ^ (-22 : ℤ) * |x.toRat / y.toRat| + 2 ^ (-148 : ℤ) : ℚ) : ℝ) := by
      exact_mod_cast herr
    calc |(r.toRat : ℝ) - (x.toRat : ℝ) / (y.toRat : ℝ)|
        = ((|r.toRat - x.toRat / y.toRat| : ℚ) : ℝ) := by
          rw [Rat.cast_abs, Rat.cast_sub, Rat.cast_div]
      _ ≤ ((2 ^ (-22 : ℤ) * |x.toRat / y.toRat| + 2 ^ (-148 : ℤ) : ℚ) : ℝ) := h1
      _ = 2 ^ (-22 : ℤ) * |(x.toRat : ℝ) / (y.toRat : ℝ)| + 2 ^ (-148 : ℤ) := by
          push_cast [Rat.cast_abs]
          ring
  exact ⟨hfin, by simpa only [FloatV.toReal] using hreal⟩

end Claims.GdnGating
