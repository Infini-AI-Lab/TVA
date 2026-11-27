import Proof.Consts

/-!
# One lane: the float facts

* `scale_lane`: the product `x · scale₃₂` of a finite input with the scale's `float32` value is
  within `3u·|x·σ| + 5u` of `x·σ` (`σ` the host's double);
* `sub_valid_ieee`, `sub_valid_fused`: a lane's `v − m`, IEEE or fusable;
* `sub_masked_ieee`, `sub_masked_fused`: a masked lane's `−∞ − m` is `−∞`, or (fused) a finite
  value at most `−2^126`;
* `exp_valid`, `exp_masked`: the platform's `exp` of a lane;
* `max_facts`: the block's max is a finite valid lane, at least every valid lane.
-/

namespace Claims.LogsumexpFwd

open TVA

/-! ## Patterns -/

/-- `x + (−y)` is `x − y`. -/
lemma add_neg_eq_sub {f : Format} (x y : FloatV f) : FloatV.add x (FloatV.neg y) = FloatV.sub x y := by
  unfold FloatV.add FloatV.sub FloatV.neg Format.sub
  rw [FloatV.pat_ofPat (Format.neg_lt _)]

/-- `−∞` is not NaN. -/
lemma negInf_isNaN32 : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN b32_ieee

/-- `−∞` is not finite. -/
lemma negInf_not_finite : (FloatV.negInf : F32).isFinite = false := by decide

/-- A finite value is not `−∞`. -/
lemma ne_negInf_of_finite {x : F32} (hx : x.isFinite = true) : x ≠ FloatV.negInf := by
  intro h; rw [h, negInf_not_finite] at hx; cases hx

/-- The rational value of `−∞` in `float32` is `−2^128`. -/
lemma negInf_toRat : (FloatV.negInf : F32).toRat = -((2 : ℚ) ^ 128) := by
  show Format.binary32.toRat (FloatV.negInf : F32).pat = _
  rw [FloatV.negInf_pat]
  have hisinf : Format.binary32.isInf
      (Format.binary32.setSign true Format.binary32.infMag) = true := by
    rw [Format.isInf_setSign]; exact Format.isInf_infMag b32_ieee
  rw [Format.toRat_of_isInf b32_reg hisinf, Format.sign_setSign]
  simp only [ite_true]
  have hb : Format.binary32.bias + 1 = (128 : Int) := by decide
  rw [hb]
  norm_num

/-- The largest finite `float32` is below `2^128`. -/
lemma maxVal_lt_two_pow : Format.binary32.maxVal < (2 : ℚ) ^ 128 := by
  have h := Format.maxVal_lt_inf b32_reg b32_ieee
  have hb : Format.binary32.bias = 127 := rfl
  rw [hb] at h
  simpa using h

/-- A finite `float32` is at most `2^128` in magnitude, as a real. -/
lemma abs_toReal_le {x : F32} (hx : x.isFinite = true) : |x.toReal| ≤ 2 ^ 128 := by
  have h := FloatV.abs_toRat_le_maxVal b32_reg hx
  have h2 : |x.toRat| ≤ (2 : ℚ) ^ 128 := le_trans h maxVal_lt_two_pow.le
  rw [FloatV.toReal_eq, ← Rat.cast_abs]
  exact_mod_cast h2

/-- A round-to-nearest magnitude is below the sign bit. -/
lemma roundMag_rn_lt_signBit {a : ℚ} (ha : 0 < a) :
    Format.binary32.roundMag .rn false a < Format.binary32.signBit := by
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.binary32) .rn ha
  by_cases hle : (Format.roundSig .rn a (Format.binary32.qexp a) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp a) ≤ Format.binary32.maxVal
  · have hpack : Format.binary32.roundMag .rn false a =
        Format.binary32.pack (Format.roundSig .rn a (Format.binary32.qexp a))
          (Format.binary32.qexp a) := by
      unfold Format.roundMag Format.encode
      rw [Format.pow2_eq, ite_eq_left hle]
    rw [hpack]
    exact lt_of_le_of_lt (Format.pack_le_maxMag b32_reg hN hQ hsub hle) (Format.maxMag_lt b32_reg)
  · unfold Format.roundMag Format.encode
    rw [Format.pow2_eq, ite_eq_right hle]
    simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
    rw [b32_ieee]
    exact FloatV.infMag_lt_signBit

/-- The overflow pattern of a negative exact value. -/
lemma roundRat_overflow_neg {q : ℚ} (zs : Bool) (hq : q < 0)
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

/-- A nonzero exact value whose rounded magnitude is in range rounds to a finite pattern. -/
lemma roundRat_rn_finite_of_no_over {q : ℚ} (zs : Bool) (hq0 : q ≠ 0)
    (hle : (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|) : ℚ) *
      Format.pow2 (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal) :
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
  have hs : (if q < 0 then Format.binary32.signBit else 0) +
      Format.binary32.pack (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|))
        (Format.binary32.qexp |q|) =
      Format.binary32.setSign (decide (q < 0))
        (Format.binary32.pack (Format.roundSig RoundMode.rn |q| (Format.binary32.qexp |q|))
          (Format.binary32.qexp |q|)) := by
    rw [Format.setSign_of_lt hltp]
    by_cases h : q < 0 <;> simp [h]
  rw [Format.roundRat_of_ne hq0, hmag, hs, Format.isFinite_setSign]
  exact Format.isFinite_of_mag_le hf (by
    unfold Format.mag
    rw [Nat.mod_eq_of_lt hltp]
    exact hple)

/-- The pattern a round-to-nearest produces is a pattern. -/
lemma roundRat_rn_lt {zs : Bool} {q : ℚ} :
    Format.binary32.roundRat .rn false zs q < 2 ^ Format.binary32.width := by
  by_cases hq : q = 0
  · subst hq
    exact (Format.roundRat_zero_spec (f := Format.binary32) b32_reg).1
  · rw [Format.roundRat_of_ne (f := Format.binary32) hq]
    have hlt : Format.binary32.roundMag .rn false |q| < Format.binary32.signBit :=
      roundMag_rn_lt_signBit (abs_pos.mpr hq)
    have h3 : Format.binary32.signBit + Format.binary32.signBit = 2 ^ Format.binary32.width := by
      rw [← Format.two_signBit (f := Format.binary32)]; ring
    have h4 := Format.signBit_lt (f := Format.binary32)
    split_ifs <;> omega

/-- **A negative exact value rounds to a finite value or to `−∞`.** -/
lemma round_neg_cases {q : ℚ} (zs : Bool) (hq : q < 0) :
    (FloatV.ofPat (Format.binary32.roundRat .rn false zs q) : F32).isFinite = true ∨
      (FloatV.ofPat (Format.binary32.roundRat .rn false zs q) : F32) = FloatV.negInf := by
  by_cases hover : (Format.roundSig RoundMode.rn (-q)
      (Format.binary32.qexp (-q)) : ℚ) * Format.pow2 (Format.binary32.qexp (-q)) ≤
      Format.binary32.maxVal
  · left
    have hlt : Format.binary32.roundRat .rn false zs q < 2 ^ Format.binary32.width :=
      roundRat_rn_lt
    rw [FloatV.isFinite_ofPat hlt]
    exact roundRat_rn_finite_of_no_over zs (ne_of_lt hq) (by rwa [abs_of_neg hq])
  · right
    rw [roundRat_overflow_neg zs hq hover]
    rfl

/-! ## The scale -/

/-- **A lane times the scale.** For a finite input `x` and a finite double `σ` with
`|σ| ≤ 2^126` and `|x·σ| ≤ 2^20`, the IEEE product with `hostF32 σ` is finite and within
`3u·|x·σ| + 5u` of `x·σ`. -/
lemma scale_lane {x : F32} {σ : F64} {v : F32} (hx : x.isFinite = true) (hσ : σ.isFinite = true)
    (hσ1 : |σ.toReal| ≤ 2 ^ 126) (hy : |x.toReal * σ.toReal| ≤ 2 ^ 20)
    (hv : FloatV.NaNEq (FloatV.mul x (FloatV.hostF32 σ)) v) :
    v.isFinite = true ∧ |v.toReal - x.toReal * σ.toReal| ≤
      3 * uR * |x.toReal * σ.toReal| + 5 * uR := by
  have hu := uR_pos
  have he := etaR_pos
  have hσn : σ.isNaN = false := (FloatV.isFinite_iff.mp hσ).1
  have hs32 : FloatV.hostF32 σ = (FloatV.cvt σ : F32) := by
    unfold FloatV.hostF32; rw [hσn]; rfl
  have hσmax : |σ.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have := b32_maxVal_ge; linarith [show (2 : ℝ) ^ 126 ≤ 2 ^ 127 by norm_num]
  obtain ⟨hcf, hce⟩ := FloatV.cvt_err_real b32_reg hσ hσmax
  rw [b32_u_cast, b32_eta_cast] at hce
  rw [← hs32] at hcf hce
  set s := (FloatV.hostF32 σ : F32) with hsdef
  have hxabs := abs_toReal_le hx
  -- `|x·s − x·σ| ≤ u·|x·σ| + 4u`
  have hd1 : |x.toReal * s.toReal - x.toReal * σ.toReal| ≤ uR * |x.toReal * σ.toReal| + 4 * uR := by
    have e1 : x.toReal * s.toReal - x.toReal * σ.toReal = x.toReal * (s.toReal - σ.toReal) := by ring
    rw [e1, abs_mul]
    have h2 : |x.toReal| * |s.toReal - σ.toReal| ≤ |x.toReal| * (|σ.toReal| * uR + etaR) :=
      mul_le_mul_of_nonneg_left hce (abs_nonneg _)
    have h3 : |x.toReal| * etaR ≤ 2 ^ 128 * etaR := mul_le_mul_of_nonneg_right hxabs he.le
    have h4 : (2 : ℝ) ^ 128 * etaR = 4 * uR := by unfold etaR uR; norm_num
    have h5 : |x.toReal| * (|σ.toReal| * uR) = uR * |x.toReal * σ.toReal| := by rw [abs_mul]; ring
    nlinarith
  have hxs : |x.toReal * s.toReal| ≤ 2 ^ 21 := by
    have h1 : |x.toReal * s.toReal| ≤ |x.toReal * σ.toReal| +
        |x.toReal * s.toReal - x.toReal * σ.toReal| := by
      have := abs_add_le (x.toReal * σ.toReal) (x.toReal * s.toReal - x.toReal * σ.toReal)
      rwa [add_sub_cancel] at this
    have h2 : uR * |x.toReal * σ.toReal| ≤ uR * 2 ^ 20 := mul_le_mul_of_nonneg_left hy hu.le
    have h3 : uR * 2 ^ 20 + 4 * uR ≤ 1 := by unfold uR; norm_num
    nlinarith
  have hxsmax : |x.toReal * s.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have := b32_maxVal_ge; linarith [show (2 : ℝ) ^ 21 ≤ 2 ^ 127 by norm_num]
  obtain ⟨hmf, hme⟩ := FloatV.mul_err_real b32_reg hx hcf hxsmax
  rw [b32_u_cast, b32_eta_cast] at hme
  obtain ⟨hvf, -, hvr⟩ := FloatV.NaNEq.exact_of_isFinite hv hmf
  refine ⟨hvf, ?_⟩
  rw [hvr]
  have hsplit : |(FloatV.mul x s).toReal - x.toReal * σ.toReal| ≤
      |(FloatV.mul x s).toReal - x.toReal * s.toReal| +
        |x.toReal * s.toReal - x.toReal * σ.toReal| := by
    have := abs_add_le ((FloatV.mul x s).toReal - x.toReal * s.toReal)
      (x.toReal * s.toReal - x.toReal * σ.toReal)
    rwa [show (FloatV.mul x s).toReal - x.toReal * s.toReal +
      (x.toReal * s.toReal - x.toReal * σ.toReal) =
      (FloatV.mul x s).toReal - x.toReal * σ.toReal by ring] at this
  have hxs2 : |x.toReal * s.toReal| ≤ |x.toReal * σ.toReal| + uR * |x.toReal * σ.toReal| + 4 * uR := by
    have := abs_add_le (x.toReal * σ.toReal) (x.toReal * s.toReal - x.toReal * σ.toReal)
    rw [add_sub_cancel] at this
    linarith
  have hy0 := abs_nonneg (x.toReal * σ.toReal)
  have hq1 : |x.toReal * s.toReal| * uR ≤
      (|x.toReal * σ.toReal| + uR * |x.toReal * σ.toReal| + 4 * uR) * uR :=
    mul_le_mul_of_nonneg_right hxs2 hu.le
  have hsmall : uR * uR * |x.toReal * σ.toReal| ≤ uR * |x.toReal * σ.toReal| := by
    have : uR * uR ≤ uR := by nlinarith [uR_le]
    exact mul_le_mul_of_nonneg_right this hy0
  have hsmall2 : 4 * uR * uR + etaR ≤ uR := by unfold uR etaR; norm_num
  nlinarith

/-! ## The subtraction of a valid lane -/

/-- `v − m` with IEEE rounding, finite operands. -/
lemma sub_valid_ieee {v mx s : F32} (hv : v.isFinite = true) (hm : mx.isFinite = true)
    (hroom : |v.toReal - mx.toReal| ≤ 2 ^ 127) (hs : FloatV.NaNEq (FloatV.sub v mx) s) :
    s.isFinite = true ∧ |s.toReal - (v.toReal - mx.toReal)| ≤
      uR * |v.toReal - mx.toReal| + etaR := by
  have hmax : |v.toReal - mx.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) :=
    le_trans hroom b32_maxVal_ge
  obtain ⟨hf, he⟩ := FloatV.sub_err_real b32_reg hv hm hmax
  rw [b32_u_cast, b32_eta_cast] at he
  obtain ⟨hsf, -, hsr⟩ := FloatV.NaNEq.exact_of_isFinite hs hf
  refine ⟨hsf, ?_⟩
  rw [hsr]; linarith

/-- `v − m` that a product may feed (`AddRel`), finite operands. -/
lemma sub_valid_fused {b : Bool} {v mx s : F32} (hv : v.isFinite = true) (hm : mx.isFinite = true)
    (hroom : |v.toReal| + |mx.toReal| ≤ 2 ^ 126) (hs : AddRel b v (FloatV.neg mx) s) :
    s.isFinite = true ∧ |s.toReal - (v.toReal - mx.toReal)| ≤
      3 * uR * (|v.toReal| + |mx.toReal|) + 4 * etaR := by
  have hnf : (FloatV.neg mx).isFinite = true := by rw [FloatV.isFinite_neg]; exact hm
  have hroomQ : |v.toRat| + |(FloatV.neg mx).toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [FloatV.toRat_neg, abs_neg]
    have h1 : ((|v.toRat| + |mx.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by
      push_cast; rw [← FloatV.toReal_eq, ← FloatV.toReal_eq]; exact hroom
    have h2 : |v.toRat| + |mx.toRat| ≤ 2 ^ 126 := by exact_mod_cast h1
    rw [Format.binary32_maxVal, Format.binary32_eta]
    norm_num at h2 ⊢
    linarith
  obtain ⟨hf, he⟩ := FloatV.AddRel.err b32_reg b32_nf (by rw [Format.binary32_u]; norm_num)
    hs hv hnf hroomQ
  refine ⟨hf, ?_⟩
  rw [FloatV.toRat_neg, abs_neg] at he
  have he' : ((|s.toRat - (v.toRat + -mx.toRat)| : ℚ) : ℝ) ≤
      ((3 * Format.binary32.u * (|v.toRat| + |mx.toRat|) + 4 * Format.binary32.eta : ℚ) : ℝ) := by
    exact_mod_cast he
  push_cast at he'
  rw [b32_u_cast, b32_eta_cast] at he'
  simp only [FloatV.toReal_eq]
  have e : (s.toRat : ℝ) - ((v.toRat : ℝ) + -(mx.toRat : ℝ)) = (s.toRat : ℝ) - ((v.toRat : ℝ) - (mx.toRat : ℝ)) := by ring
  rw [e] at he'
  exact he'

/-! ## The subtraction of a masked lane -/

/-- `−∞ − m` with IEEE rounding is `−∞` (`m` finite). -/
lemma sub_masked_ieee {mx s : F32} (hm : mx.isFinite = true)
    (hs : FloatV.NaNEq (FloatV.sub FloatV.negInf mx) s) : s = FloatV.negInf := by
  rw [FloatV.sub_negInf_left b32_ieee hm] at hs
  exact hs.eq negInf_isNaN32

/-- An exact value that rounds to `−∞` is below `−maxVal`. -/
lemma roundsTo_negInf {p : ℚ} (h : RoundsTo p (FloatV.negInf : F32)) :
    p < -Format.binary32.maxVal := by
  rcases h with ⟨-, hfin, -⟩ | ⟨hinf, hpat⟩
  · rw [negInf_not_finite] at hfin; cases hfin
  · by_contra hn
    push Not at hn
    by_cases hp : |p| ≤ Format.binary32.maxVal
    · by_cases hp0 : p = 0
      · subst hp0
        obtain ⟨-, hf0, -⟩ := Format.roundRat_zero_spec (f := Format.binary32) b32_reg
          (rnd := .rn) (sat := false) (zs := false)
        rw [Format.isFinite_iff] at hf0
        rw [hf0.2] at hinf; cases hinf
      · obtain ⟨-, hf0, -⟩ := Format.roundRat_spec b32_reg .rn false false hp0 hp
        rw [Format.isFinite_iff] at hf0
        rw [hf0.2] at hinf; cases hinf
    · -- `p > maxVal ≥ 0`: the pattern has no sign bit, `−∞`'s has
      have hpos : 0 < p := by
        rcases le_or_gt 0 p with h | h
        · rcases eq_or_lt_of_le h with h' | h'
          · subst h'; simp at hp; exact absurd (Format.maxVal_pos b32_reg).le (not_le.mpr hp)
          · exact h'
        · exfalso; rw [abs_of_neg h] at hp; linarith
      rw [Format.roundRat_of_ne (ne_of_gt hpos), ite_eq_right (not_lt.mpr hpos.le), Nat.zero_add,
        FloatV.negInf_pat, Format.setSign_of_lt FloatV.infMag_lt_signBit] at hpat
      have h1 := roundMag_rn_lt_signBit (abs_pos.mpr (ne_of_gt hpos))
      simp only [ite_true] at hpat
      omega

/-- `−∞ − m` that a product may feed (`AddRel`): `−∞`, or a finite value at most `−2^126`
(`m` finite, `|m| ≤ 2^126`). -/
lemma sub_masked_fused {b : Bool} {mx s : F32} (hm : mx.isFinite = true)
    (hm1 : |mx.toReal| ≤ 2 ^ 126) (hs : AddRel b FloatV.negInf (FloatV.neg mx) s) :
    s = FloatV.negInf ∨ (s.isFinite = true ∧ s.toReal ≤ -2 ^ 126) := by
  have hnf : (FloatV.neg mx).isFinite = true := by rw [FloatV.isFinite_neg]; exact hm
  rcases hs with hne | ⟨-, hfu⟩
  · left
    rw [add_neg_eq_sub, FloatV.sub_negInf_left b32_ieee hm] at hne
    exact hne.eq negInf_isNaN32
  · rcases hfu with ⟨p, q, hp, hq, -, z, hne⟩ | ⟨p, -, hyi, -⟩ | ⟨q, -, -, hr⟩
    · have hp1 := roundsTo_negInf hp
      have hq1 := RoundsTo.abs_le b32_reg b32_nf hq hnf
      rw [FloatV.toRat_neg, abs_neg] at hq1
      have hmQ : |mx.toRat| ≤ 2 ^ 126 := by
        have : ((|mx.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by
          rw [Rat.cast_abs, ← FloatV.toReal_eq]; exact hm1
        exact_mod_cast this
      have hmv := maxVal_lt_two_pow
      have hmv2 : (2 : ℚ) ^ 127 + 2 ^ 126 + 2 ^ 104 ≤ Format.binary32.maxVal := by
        rw [Format.binary32_maxVal]; norm_num
      have hu : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u
      have he : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta
      have hq2 : |q| ≤ 2 ^ 126 + 2 ^ 104 := by
        rw [hu, he] at hq1
        have : (1 + (2 : ℚ) ^ (-24 : ℤ)) * |mx.toRat| ≤ (1 + 2 ^ (-24 : ℤ)) * 2 ^ 126 :=
          mul_le_mul_of_nonneg_left hmQ (by positivity)
        norm_num at this hq1 ⊢
        linarith
      have hpq : p + q < -(2 : ℚ) ^ 127 := by
        have := (abs_le.mp hq2).2
        linarith
      rcases round_neg_cases z (by linarith : p + q < 0) with hfin | hinf
      · right
        obtain ⟨hsf, hsq, -⟩ := hne.exact_of_isFinite hfin
        refine ⟨hsf, ?_⟩
        have hlt := roundRat_rn_lt (zs := z) (q := p + q)
        rw [FloatV.isFinite_ofPat hlt] at hfin
        have herr := Format.roundRat_rn_err_of_finite b32_reg b32_nf z hfin
        rw [FloatV.toRat_ofPat hlt] at hsq
        rw [← hsq] at herr
        have hsabs := FloatV.abs_toRat_le_maxVal b32_reg hsf
        rw [hu, he] at herr
        have hb : |s.toRat| * (2 : ℚ) ^ (-24 : ℤ) ≤ Format.binary32.maxVal * 2 ^ (-24 : ℤ) :=
          mul_le_mul_of_nonneg_right hsabs (by positivity)
        have hmv3 : Format.binary32.maxVal * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) ≤ (2 : ℚ) ^ 126 := by
          rw [Format.binary32_maxVal]; norm_num
        have hsQ : s.toRat ≤ -(2 : ℚ) ^ 126 := by
          have := (abs_le.mp herr).2
          linarith
        rw [FloatV.toReal_eq]
        exact_mod_cast hsQ
      · left
        rw [hinf] at hne
        exact hne.eq negInf_isNaN32
    · exfalso
      rw [(FloatV.isFinite_iff.mp hnf).2] at hyi; cases hyi
    · left; exact hr

end Claims.LogsumexpFwd
