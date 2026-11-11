import Claim
import TVA.Proof
import TVA
import Proof.NumFacts
import Proof.NumWide
import Proof.NumSlack
import Proof.NumMerge

/-!
# The middle of the merge branch

`tl.maximum`, the two subtractions, the two exponentials and their sum, turned into the reference
maximum, weights and sum.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## Elementary bounds -/

/-- `exp(-x) ≤ 2^-22` for `x ≥ 16`. -/
theorem exp_neg_le_of_ge {x : ℝ} (hx : 16 ≤ x) :
    Real.exp (-x) ≤ (2 : ℝ) ^ (-22 : ℤ) := by
  have h1 : Real.exp (-x) ≤ Real.exp (-16) := Real.exp_le_exp.mpr (by linarith)
  have h2 : Real.exp (-16 : ℝ) = (Real.exp 16)⁻¹ := Real.exp_neg 16
  have h3 : Real.exp (16 : ℝ) = (Real.exp 1) ^ 16 := by
    simp
  have h4 : (2 : ℝ) ^ 22 ≤ Real.exp 16 := by
    rw [h3]
    have h5 : (2 : ℝ) ^ 22 ≤ (2.7 : ℝ) ^ 16 := by norm_num
    have h5' : (2.7 : ℝ) ^ 16 ≤ (2.7182818283 : ℝ) ^ 16 := by norm_num
    have h6 : (2.7182818283 : ℝ) ^ 16 ≤ (Real.exp 1) ^ 16 :=
      pow_le_pow_left₀ (by norm_num) (le_of_lt Real.exp_one_gt_d9) 16
    linarith
  have h7 : (Real.exp 16)⁻¹ ≤ ((2 : ℝ) ^ 22)⁻¹ := by
    simpa [one_div] using (one_div_le_one_div_of_le (show (0 : ℝ) < (2 : ℝ) ^ 22 by positivity) h4)
  have h8 : (2 : ℝ) ^ (-22 : ℤ) = ((2 : ℝ) ^ 22)⁻¹ := by norm_num
  calc Real.exp (-x) ≤ Real.exp (-16) := h1
    _ = (Real.exp 16)⁻¹ := h2
    _ ≤ ((2 : ℝ) ^ 22)⁻¹ := h7
    _ = (2 : ℝ) ^ (-22 : ℤ) := h8.symm

/-- A nonpositive `q` rounded with relative error `2^-24` has `exp` within `2^-20` of `exp q`. -/
theorem exp_err_of_round {q : ℚ} {w : ℝ} (hq : (q : ℝ) ≤ 0)
    (hw : |w - (q : ℝ)| ≤ |(q : ℝ)| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) :
    |Real.exp w - Real.exp (q : ℝ)| ≤ (2 : ℝ) ^ (-20 : ℤ) := by
  refine exp_close hq ?_
  calc |w - (q : ℝ)| ≤ |(q : ℝ)| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := hw
    _ = (2 : ℝ) ^ (-24 : ℤ) * |(q : ℝ)| + (2 : ℝ) ^ (-150 : ℤ) := by rw [mul_comm]
    _ ≤ (2 : ℝ) ^ (-24 : ℤ) * |(q : ℝ)| + (2 : ℝ) ^ (-134 : ℤ) := by
        have h150 : (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-134 : ℤ) := by norm_num
        linarith

/-- The exponential of a finite nonpositive value: finite, accurate, and at most one. -/
theorem exp_acc_bound {P : Platform} (hE : Device.ExpF32 P) {x e : F32}
    (hxf : x.isFinite = true) (hx0 : x.toReal ≤ 0)
    (he : P.math .exp .binary32 x e) :
    e.isFinite = true ∧
      |e.toReal - Real.exp x.toReal| ≤
        (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-140 : ℤ) ∧
      Real.exp x.toReal ≤ 1 := by
  have hmul : |x.toReal| * Real.exp x.toReal ≤ 1 := by
    rw [abs_of_nonpos hx0]
    simpa using mul_exp_neg_le (-x.toReal)
  have hexp1 : Real.exp x.toReal ≤ 1 := Real.exp_le_one_iff.mpr hx0
  have hexpnn : 0 ≤ Real.exp x.toReal := (Real.exp_pos _).le
  obtain ⟨ref, href, hacc⟩ := hE |x.toReal| (abs_nonneg _)
  have href' : ref = Real.exp := by
    have h := href
    simp only [TVA.MathFn.ref] at h
    exact (Option.some.inj h).symm
  rw [href'] at hacc
  have hmem : x.toReal ∈ Set.Icc (-|x.toReal|) |x.toReal| := ⟨neg_abs_le _, le_abs_self _⟩
  have hnat : x.toReal ∈ TVA.MathFn.natDom .exp := Set.mem_univ _
  have hmax3q : (3 : ℚ) ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  have hmax3 : (3 : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast hmax3q
  have hrel : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x.toReal|) * Real.exp x.toReal
      ≤ (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) := by
    have h2 : (2 : ℝ) ^ (-23 : ℤ) * (|x.toReal| * Real.exp x.toReal) ≤ (2 : ℝ) ^ (-23 : ℤ) := by
      have hpos : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) := by norm_num
      calc (2 : ℝ) ^ (-23 : ℤ) * (|x.toReal| * Real.exp x.toReal)
          ≤ (2 : ℝ) ^ (-23 : ℤ) * 1 := mul_le_mul_of_nonneg_left hmul hpos
        _ = (2 : ℝ) ^ (-23 : ℤ) := by ring
    have h3 : (2 : ℝ) ^ (-22 : ℤ) * Real.exp x.toReal ≤ (2 : ℝ) ^ (-22 : ℤ) := by
      have hpos : (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) := by norm_num
      calc (2 : ℝ) ^ (-22 : ℤ) * Real.exp x.toReal ≤ (2 : ℝ) ^ (-22 : ℤ) * 1 :=
            mul_le_mul_of_nonneg_left hexp1 hpos
        _ = (2 : ℝ) ^ (-22 : ℤ) := by ring
    nlinarith [h2, h3]
  have hmain : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x.toReal|)) *
      Real.exp x.toReal ≤ 1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ)) := by
    nlinarith [hrel, hexp1, hexpnn]
  have hroom : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x.toReal|)) *
      |Real.exp x.toReal| + (2 : ℝ) ^ (-140 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [abs_of_nonneg hexpnn]
    have hconst : (1 : ℝ) + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ)) +
        (2 : ℝ) ^ (-140 : ℤ) ≤ 3 := by norm_num
    linarith [hmain, hconst, hmax3]
  obtain ⟨hefin, herr⟩ := hacc x e he hxf hmem hnat hroom
  refine ⟨hefin, ?_, hexp1⟩
  rw [abs_of_nonneg hexpnn] at herr
  linarith

/-! ## Rounding facts for the subtraction -/

/-- A nonpositive rounded value is nonpositive, in range. -/
theorem roundRat_rn_toRat_nonpos {zs : Bool} {q : ℚ}
    (hA : |q| ≤ Format.binary32.maxVal) (hq : q ≤ 0) :
    Format.binary32.toRat (Format.binary32.roundRat .rn false zs q) ≤ 0 := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨-, -, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn)
      (sat := false) (zs := zs) Format.binary32_regular
    rw [h3]
  · have hqneg : q < 0 := lt_of_le_of_ne hq hq0
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec (f := Format.binary32) Format.binary32_regular
      .rn false zs hq0 hA
    rw [hval, ite_eq_left hqneg]
    have h1 : (0 : ℚ) ≤ (Format.roundSig .rn |q| (Format.binary32.qexp |q|) : ℚ) :=
      Nat.cast_nonneg _
    have h2 : (0 : ℚ) ≤ (2 : ℚ) ^ (Format.binary32.qexp |q|) := by positivity
    have h3 : (0 : ℚ) ≤ (Format.roundSig .rn |q| (Format.binary32.qexp |q|) : ℚ) *
        (2 : ℚ) ^ (Format.binary32.qexp |q|) := mul_nonneg h1 h2
    have h4 := mul_nonpos_of_nonpos_of_nonneg (show (-1 : ℚ) ≤ 0 by norm_num) h3
    simpa using h4

/-- `setSign true` of any magnitude is nonpositive. -/
theorem toRat_setSign_true_nonpos {M : ℕ} :
    Format.binary32.toRat (Format.binary32.setSign true M) ≤ 0 := by
  rw [Format.toRat_setSign]
  split_ifs with h
  · exact neg_nonpos.mpr (Format.magVal_nonneg _)
  · exact absurd rfl h

/-- An overflowing rounding to nearest gives the infinite magnitude. -/
theorem roundMag_rn_eq_infMag {q : ℚ}
    (hno : ¬ (Format.roundSig .rn |q| (Format.binary32.qexp |q|) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal) :
    Format.binary32.roundMag .rn false |q| = Format.binary32.infMag := by
  unfold Format.roundMag Format.encode
  rw [Format.pow2_eq, ite_eq_right hno]
  simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
  cases hsp : Format.binary32.specials with
  | ieee => rfl
  | nanOnly => exact absurd hsp (by decide)
  | finite => exact absurd hsp (by decide)

/-! ## The exponential of the subtracted value -/

/-- The core estimate: a rounded pattern `n` close to a nonpositive `q` gives an exponential
within `2^-20 + 2^-21` of `exp q`. -/
theorem exp_sub_core {P : Platform} (hE : Device.ExpF32 P) {x M u e : F32} {n : ℕ} {q : ℚ}
    (hnlt : n < 2 ^ Format.binary32.width)
    (hfin : Format.binary32.isFinite n = true)
    (hnonpos : Format.binary32.toRat n ≤ 0)
    (herr : |(Format.binary32.toRat n : ℚ) - q| ≤
      |q| * Format.binary32.u + Format.binary32.eta)
    (hq0R : (q : ℝ) ≤ 0)
    (hqR : ((q : ℚ) : ℝ) = x.toReal - M.toReal)
    (hsub : FloatV.sub x M = FloatV.ofPat n)
    (hu : FloatV.NaNEq (FloatV.sub x M) u)
    (he : P.math .exp .binary32 u e) :
    e.isFinite = true ∧
      |e.toReal - Real.exp (x.toReal - M.toReal)| ≤
        (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) ∧
      e.toReal ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
  have hnan : (FloatV.sub x M).isNaN = false := by
    rw [hsub, FloatV.isNaN_ofPat hnlt]
    exact (Format.isFinite_iff (f := Format.binary32) n).mp hfin |>.1
  have hueq : u = FloatV.ofPat n := (hu.eq hnan).trans hsub
  have hufin : u.isFinite = true := by rw [hueq, FloatV.isFinite_ofPat hnlt]; exact hfin
  have huval : u.toReal = ((Format.binary32.toRat n : ℚ) : ℝ) := by
    rw [hueq, FloatV.toReal, FloatV.toRat_ofPat hnlt]
  have hu0 : u.toReal ≤ 0 := by rw [huval]; exact_mod_cast hnonpos
  have u_val : (Format.binary32.u : ℝ) = (2 : ℝ) ^ (-24 : ℤ) := by
    rw [Format.binary32_u]
    norm_num
  have eta_val : (Format.binary32.eta : ℝ) = (2 : ℝ) ^ (-150 : ℤ) := by
    rw [Format.binary32_eta]
    norm_num
  have herrR : |u.toReal - (q : ℝ)| ≤
      |(q : ℝ)| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have hcast : ((|(Format.binary32.toRat n : ℚ) - q| : ℚ) : ℝ) ≤
        ((|q| * Format.binary32.u + Format.binary32.eta : ℚ) : ℝ) := Rat.cast_le.mpr herr
    push_cast at hcast
    rw [u_val, eta_val] at hcast
    rw [huval]
    exact hcast
  obtain ⟨hefin, herrE, hexp1⟩ := exp_acc_bound hE hufin hu0 he
  have herrExp := exp_err_of_round hq0R herrR
  have hsplit : e.toReal - Real.exp (x.toReal - M.toReal) =
      (e.toReal - Real.exp u.toReal) + (Real.exp u.toReal - Real.exp (q : ℝ)) := by
    rw [← hqR]
    ring
  have htri := abs_add_le (e.toReal - Real.exp u.toReal)
    (Real.exp u.toReal - Real.exp (q : ℝ))
  refine ⟨hefin, ?_, ?_⟩
  · rw [hsplit]
    have hnum : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) +
        (2 : ℝ) ^ (-20 : ℤ) ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) := by norm_num
    linarith [htri, herrE, herrExp]
  · have h2 := (abs_le.mp herrE).2
    have hnum : (1 : ℝ) + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) ≤
        1 + (2 : ℝ) ^ (-18 : ℤ) := by norm_num
    linarith [hexp1, h2]

/-- **The exponential of a far-away subtracted value**: when `q` is at least `16` below zero and
beyond `maxVal`, both `exp u` and `exp q` are below `2^-22`. -/
theorem exp_sub_far {P : Platform} (hE : Device.ExpF32 P) {x M u e : F32} {n : ℕ} {q : ℚ}
    (hnlt : n < 2 ^ Format.binary32.width)
    (hfin : Format.binary32.isFinite n = true)
    (hnonpos : Format.binary32.toRat n ≤ 0)
    (herr : |(Format.binary32.toRat n : ℚ) - q| ≤
      |(Format.binary32.toRat n : ℚ)| * Format.binary32.u + Format.binary32.eta)
    (hqmax : (q : ℝ) ≤ -(Format.binary32.maxVal : ℝ))
    (hmax32 : (32 : ℝ) ≤ (Format.binary32.maxVal : ℝ))
    (hqR : ((q : ℚ) : ℝ) = x.toReal - M.toReal)
    (hsub : FloatV.sub x M = FloatV.ofPat n)
    (hu : FloatV.NaNEq (FloatV.sub x M) u)
    (he : P.math .exp .binary32 u e) :
    e.isFinite = true ∧
      |e.toReal - Real.exp (x.toReal - M.toReal)| ≤
        (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) ∧
      e.toReal ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
  have hnan : (FloatV.sub x M).isNaN = false := by
    rw [hsub, FloatV.isNaN_ofPat hnlt]
    exact (Format.isFinite_iff (f := Format.binary32) n).mp hfin |>.1
  have hueq : u = FloatV.ofPat n := (hu.eq hnan).trans hsub
  have hufin : u.isFinite = true := by rw [hueq, FloatV.isFinite_ofPat hnlt]; exact hfin
  have huval : u.toReal = ((Format.binary32.toRat n : ℚ) : ℝ) := by
    rw [hueq, FloatV.toReal, FloatV.toRat_ofPat hnlt]
  have hu0 : u.toReal ≤ 0 := by rw [huval]; exact_mod_cast hnonpos
  have u_val : (Format.binary32.u : ℝ) = (2 : ℝ) ^ (-24 : ℤ) := by
    rw [Format.binary32_u]
    norm_num
  have eta_val : (Format.binary32.eta : ℝ) = (2 : ℝ) ^ (-150 : ℤ) := by
    rw [Format.binary32_eta]
    norm_num
  have herrR : |u.toReal - (q : ℝ)| ≤
      |u.toReal| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have hcast : ((|(Format.binary32.toRat n : ℚ) - q| : ℚ) : ℝ) ≤
        ((|(Format.binary32.toRat n : ℚ)| * Format.binary32.u + Format.binary32.eta : ℚ) : ℝ) :=
      Rat.cast_le.mpr herr
    push_cast at hcast
    rw [u_val, eta_val] at hcast
    rw [huval]
    exact hcast
  have hstep : u.toReal - (q : ℝ) ≤
      (-u.toReal) * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have h1 := (le_abs_self (u.toReal - (q : ℝ))).trans herrR
    rwa [abs_of_nonpos hu0] at h1
  have hu24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have hetsmall : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 := by norm_num
  have hw16 : (16 : ℝ) ≤ -u.toReal := by
    rw [hu24] at hstep
    linarith
  have hq16 : (16 : ℝ) ≤ -(q : ℝ) := by linarith
  have hwu : Real.exp u.toReal ≤ (2 : ℝ) ^ (-22 : ℤ) := by
    simpa using exp_neg_le_of_ge hw16
  have hqexp : Real.exp (q : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) := by
    simpa using exp_neg_le_of_ge hq16
  obtain ⟨hefin, herrE, hexp1⟩ := exp_acc_bound hE hufin hu0 he
  have hboth : |Real.exp u.toReal - Real.exp (q : ℝ)| ≤
      (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-22 : ℤ) := by
    have h1 : |Real.exp u.toReal| ≤ (2 : ℝ) ^ (-22 : ℤ) := by
      rw [abs_of_nonneg (Real.exp_pos _).le]; exact hwu
    have h2 : |Real.exp (q : ℝ)| ≤ (2 : ℝ) ^ (-22 : ℤ) := by
      rw [abs_of_nonneg (Real.exp_pos _).le]; exact hqexp
    calc |Real.exp u.toReal - Real.exp (q : ℝ)|
        = |Real.exp u.toReal + -Real.exp (q : ℝ)| := by ring_nf
      _ ≤ |Real.exp u.toReal| + |-Real.exp (q : ℝ)| := abs_add_le _ _
      _ = |Real.exp u.toReal| + |Real.exp (q : ℝ)| := by rw [abs_neg]
      _ ≤ (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-22 : ℤ) := add_le_add h1 h2
  have hsplit : e.toReal - Real.exp (x.toReal - M.toReal) =
      (e.toReal - Real.exp u.toReal) + (Real.exp u.toReal - Real.exp (q : ℝ)) := by
    rw [← hqR]
    ring
  have htri := abs_add_le (e.toReal - Real.exp u.toReal)
    (Real.exp u.toReal - Real.exp (q : ℝ))
  refine ⟨hefin, ?_, ?_⟩
  · rw [hsplit]
    have hnum : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) +
        ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)) ≤
        (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) := by norm_num
    linarith [htri, herrE, hboth]
  · have h2 := (abs_le.mp herrE).2
    have hnum : (1 : ℝ) + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) ≤
        1 + (2 : ℝ) ^ (-18 : ℤ) := by norm_num
    linarith [hexp1, h2]

/-- **The exponential of the subtracted value is accurate**, for finite `x` and `M`. -/
theorem exp_sub_mid {P : Platform} (hE : Device.ExpF32 P)
    (hN : Device.ExpF32NegInf P) {x M u e : F32}
    (hxf : x.isFinite = true) (hMfin : M.isFinite = true)
    (hle : x.toReal ≤ M.toReal)
    (hu : FloatV.NaNEq (FloatV.sub x M) u)
    (he : P.math .exp .binary32 u e) :
    e.isFinite = true ∧
      |e.toReal - Real.exp (x.toReal - M.toReal)| ≤
        (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) ∧
      e.toReal ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
  set q : ℚ := Format.binary32.toRat x.pat - Format.binary32.toRat M.pat with hqdef
  set zs : Bool := Format.binary32.sign x.pat && !Format.binary32.sign M.pat with hzsdef
  set n : ℕ := Format.binary32.roundRat .rn false zs q with hn
  have hsub : FloatV.sub x M = FloatV.ofPat n := by
    unfold FloatV.sub
    rw [Format.sub_of_finite .rn hxf hMfin, ← hn]
  have hqR : ((q : ℚ) : ℝ) = x.toReal - M.toReal := by
    rw [hqdef]
    push_cast
    simp [FloatV.toReal, FloatV.toRat]
  have hq0R : (q : ℝ) ≤ 0 := by
    rw [hqR]
    linarith
  have hq0 : q ≤ 0 := by
    have h : ((q : ℚ) : ℝ) ≤ 0 := by rw [hqR]; linarith
    exact_mod_cast h
  by_cases hA : |q| ≤ Format.binary32.maxVal
  · obtain ⟨hlt, hfin', herr⟩ := Format.roundRat_rn_err (f := Format.binary32)
      Format.binary32_regular false zs (q := q) hA
    rw [← hn] at hlt hfin' herr
    have hnonpos : Format.binary32.toRat n ≤ 0 := by
      rw [hn]
      exact roundRat_rn_toRat_nonpos hA hq0
    exact exp_sub_core hE hlt hfin' hnonpos herr hq0R hqR hsub hu he
  · replace hA : Format.binary32.maxVal < |q| := not_le.mp hA
    have hmax0 : (0 : ℚ) ≤ Format.binary32.maxVal := by
      rw [Format.binary32_maxVal]
      norm_num
    have hqne : q ≠ 0 := by
      intro h
      rw [h, abs_zero] at hA
      exact absurd hA (not_lt.mpr hmax0)
    have hqneg : q < 0 := lt_of_le_of_ne hq0 hqne
    have hqnegR : (q : ℝ) ≤ -(Format.binary32.maxVal : ℝ) := by
      have h1 : ((Format.binary32.maxVal : ℚ) : ℝ) < ((|q| : ℚ) : ℝ) := Rat.cast_lt.mpr hA
      rw [Rat.cast_abs] at h1
      have h2 : |(q : ℝ)| = -(q : ℝ) := abs_of_nonpos hq0R
      linarith
    have hn_eq : n = (if q < 0 then Format.binary32.signBit else 0) +
        Format.binary32.roundMag .rn false |q| := by
      rw [hn]
      exact Format.roundRat_of_ne (f := Format.binary32) hqne
    obtain ⟨hNb, hQb, hsubm⟩ := Format.roundSig_bounds (f := Format.binary32) .rn
      (abs_pos.mpr hqne)
    by_cases hno : (Format.roundSig .rn |q| (Format.binary32.qexp |q|) : ℚ) *
        (2 : ℚ) ^ (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal
    · have hmag : Format.binary32.roundMag .rn false |q| =
          Format.binary32.pack (Format.roundSig .rn |q| (Format.binary32.qexp |q|))
            (Format.binary32.qexp |q|) := by
        unfold Format.roundMag Format.encode
        rw [Format.pow2_eq, ite_eq_left hno]
      have hpack := Format.pack_le_maxMag (f := Format.binary32) Format.binary32_regular
        hNb hQb hsubm hno
      have hplt : Format.binary32.pack (Format.roundSig .rn |q| (Format.binary32.qexp |q|))
          (Format.binary32.qexp |q|) < Format.binary32.signBit :=
        lt_of_le_of_lt hpack (Format.maxMag_lt Format.binary32_regular)
      have hnset : n = Format.binary32.setSign true
          (Format.binary32.pack (Format.roundSig .rn |q| (Format.binary32.qexp |q|))
            (Format.binary32.qexp |q|)) := by
        rw [hn_eq, ite_eq_left hqneg, hmag, Format.setSign_of_lt hplt]
        simp
      have hnlt : n < 2 ^ Format.binary32.width := by
        rw [hnset]
        exact Format.setSign_lt _ _
      have hfin' : Format.binary32.isFinite
          (Format.binary32.roundRat .rn false zs q) = true := by
        rw [← hn, hnset, Format.isFinite_setSign]
        exact Format.isFinite_of_mag_le (f := Format.binary32) Format.binary32_regular
          ((Format.mag_le (f := Format.binary32)
            (Format.binary32.pack (Format.roundSig .rn |q| (Format.binary32.qexp |q|))
              (Format.binary32.qexp |q|))).trans hpack)
      have hfin : Format.binary32.isFinite n = true := by rw [hn]; exact hfin'
      have herr := Format.roundRat_rn_err_of_finite (f := Format.binary32)
        Format.binary32_regular (by decide : Format.binary32.specials ≠ .finite) zs hfin'
      rw [← hn] at herr
      have hnonpos : Format.binary32.toRat n ≤ 0 := by
        rw [hnset]
        exact toRat_setSign_true_nonpos
      have hmax32q : (32 : ℚ) ≤ Format.binary32.maxVal := by
        rw [Format.binary32_maxVal]
        norm_num
      have hmax32 : (32 : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast hmax32q
      exact exp_sub_far hE hnlt hfin hnonpos herr hqnegR hmax32 hqR hsub hu he
    · have hmag : Format.binary32.roundMag .rn false |q| = Format.binary32.infMag :=
        roundMag_rn_eq_infMag hno
      have hnset : n = Format.binary32.setSign true Format.binary32.infMag := by
        rw [hn_eq, ite_eq_left hqneg, hmag,
          Format.setSign_of_lt (FloatV.infMag_lt_signBit (f := Format.binary32))]
        simp
      have hnlt : n < 2 ^ Format.binary32.width := by
        rw [hnset]
        exact Format.setSign_lt _ _
      have hnpat : n = (FloatV.negInf : F32).pat := by
        rw [hnset, FloatV.negInf_pat]
      have hni : Format.binary32.isNaN Format.binary32.infMag = false := by
        have h := FloatV.negInf_isNaN (f := Format.binary32) (by decide)
        unfold FloatV.isNaN at h
        rwa [FloatV.negInf_pat, Format.isNaN_setSign] at h
      have hnan : (FloatV.sub x M).isNaN = false := by
        rw [hsub, FloatV.isNaN_ofPat hnlt, hnset, Format.isNaN_setSign]
        exact hni
      have hueq : u = FloatV.negInf := by
        have h1 : u = FloatV.ofPat n := (hu.eq hnan).trans hsub
        rw [h1, hnpat]
        exact FloatV.ofPat_pat _
      have hmath : P.math .exp .binary32 FloatV.negInf e := by
        rw [← hueq]
        exact he
      have hez : e = FloatV.ofPat 0 := hN e hmath
      have hzero : e = zeroC := by
        rw [hez]
        show FloatV.ofPat 0 = (zeroC : F32)
        exact zeroC_eq.symm
      have hbound : Real.exp (x.toReal - M.toReal) ≤ (2 : ℝ) ^ (-22 : ℤ) := by
        have hge : (16 : ℝ) ≤ -(x.toReal - M.toReal) := by
          have hmax32q : (32 : ℚ) ≤ Format.binary32.maxVal := by
            rw [Format.binary32_maxVal]
            norm_num
          have hmax32 : (32 : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast hmax32q
          linarith
        simpa using exp_neg_le_of_ge hge
      have hnonneg : 0 ≤ Real.exp (x.toReal - M.toReal) := (Real.exp_pos _).le
      have hnum : (2 : ℝ) ^ (-22 : ℤ) ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) := by
        norm_num
      refine ⟨?_, ?_, ?_⟩
      · rw [hzero]
        exact zeroC_isFinite
      · rw [hzero, zeroC_toReal, zero_sub, abs_neg, abs_of_nonneg hnonneg]
        linarith
      · rw [hzero, zeroC_toReal]
        have hpos : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by positivity
        linarith

/-- **The weight of one side**: the exponential of the subtracted value, with the `−∞` cases. -/
theorem exp_side_mid {P : Platform} (hE : Device.ExpF32 P)
    (hN : Device.ExpF32NegInf P) {x M u e : F32} {m : ℝ}
    (hxf : x.isFinite = true ∨ x = negInfC) (hMfin : M.isFinite = true)
    (hm : m = M.toReal) (hle : x.toReal ≤ m)
    (hu : FloatV.NaNEq (FloatV.sub x M) u)
    (he : P.math .exp .binary32 u e) :
    e.isFinite = true ∧
      |e.toReal - side x m| ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) ∧
      e.toReal ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
  rcases eq_or_ne x negInfC with hx | hx
  · have hsub : FloatV.sub x M = FloatV.negInf := by
      rw [hx]
      exact FloatV.sub_negInf_left (by decide) hMfin
    have hni : (FloatV.negInf : F32).isNaN = false :=
      FloatV.negInf_isNaN (f := Format.binary32) (by decide)
    have hnan : (FloatV.sub x M).isNaN = false := by rw [hsub]; exact hni
    have hueq : u = FloatV.negInf := (hu.eq hnan).trans hsub
    have hmath : P.math .exp .binary32 FloatV.negInf e := by
      rw [← hueq]
      exact he
    have hez : e = FloatV.ofPat 0 := hN e hmath
    have hzero : e = zeroC := by
      rw [hez]
      show FloatV.ofPat 0 = (zeroC : F32)
      exact zeroC_eq.symm
    have hside : side x m = 0 := by rw [hx]; exact side_negInf m
    have hnum : (2 : ℝ) ^ (-22 : ℤ) ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) := by
      norm_num
    refine ⟨?_, ?_, ?_⟩
    · rw [hzero]
      exact zeroC_isFinite
    · rw [hzero, zeroC_toReal, hside, sub_zero, abs_zero]
      norm_num
    · rw [hzero, zeroC_toReal]
      have hpos : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by positivity
      linarith
  · have hxf' : x.isFinite = true := by rcases hxf with h | h; exact h; exact absurd h hx
    have hle' : x.toReal ≤ M.toReal := by rw [← hm]; exact hle
    obtain ⟨h1, h2, h3⟩ := exp_sub_mid hE hN hxf' hMfin hle' hu he
    have hside : side x m = Real.exp (x.toReal - M.toReal) := by
      rw [side_of_ne hx, hm]
    exact ⟨h1, by rw [hside]; exact h2, h3⟩

/-- The reference weight is nonnegative. -/
theorem side_nonneg_aux (x : F32) (m : ℝ) : 0 ≤ side x m := by
  unfold side
  split_ifs with h
  · norm_num
  · exact (Real.exp_pos _).le

/-- The reference sum is at least one. -/
theorem one_le_refSum_aux {p s : F32} (hne : ¬ (p = negInfC ∧ s = negInfC)) :
    1 ≤ refSum p s := by
  have hpn : 0 ≤ side p (refMax p s) := side_nonneg_aux p _
  have hsn : 0 ≤ side s (refMax p s) := side_nonneg_aux s _
  rcases eq_or_ne p negInfC with hp | hp
  · have hs : s ≠ negInfC := fun h => hne ⟨hp, h⟩
    have hm : refMax p s = s.toReal := by rw [hp]; exact refMax_negInf_left s
    have hside : side s (refMax p s) = 1 := by
      rw [side_of_ne hs, hm]; simp
    have hzero : side p (refMax p s) = 0 := by rw [hp]; exact side_negInf _
    unfold refSum
    rw [hzero, hside]
    norm_num
  · rcases eq_or_ne s negInfC with hs | hs
    · have hm : refMax p s = p.toReal := by
        simp [refMax, hp, hs]
      have hside : side p (refMax p s) = 1 := by
        rw [side_of_ne hp, hm]; simp
      have hzero : side s (refMax p s) = 0 := by rw [hs]; exact side_negInf _
      unfold refSum
      rw [hzero, hside]
      norm_num
    · have hm : refMax p s = max p.toReal s.toReal := refMax_eq_max hp hs
      rcases le_total p.toReal s.toReal with h | h
      · have hside : side s (refMax p s) = 1 := by
          rw [side_of_ne hs, hm, max_eq_right h]; simp
        unfold refSum
        rw [hside]
        linarith
      · have hside : side p (refMax p s) = 1 := by
          rw [side_of_ne hp, hm, max_eq_left h]; simp
        unfold refSum
        rw [hside]
        linarith

/-- **The lse of the merge branch.** -/
theorem merge_lse_mid {P : Platform} (hE : Device.ExpF32 P)
    (hN : Device.ExpF32NegInf P) {p s M u v pe se o : F32}
    (hp : p.isNaN = false) (hs : s.isNaN = false)
    (hpf : p.isFinite = true ∨ p = negInfC) (hsf : s.isFinite = true ∨ s = negInfC)
    (hne : ¬ (p = negInfC ∧ s = negInfC))
    (hM : FloatV.MaxRel true p s M)
    (hu : FloatV.NaNEq (FloatV.sub p M) u)
    (hv : FloatV.NaNEq (FloatV.sub s M) v)
    (he1 : P.math .exp .binary32 u pe)
    (he2 : P.math .exp .binary32 v se)
    (ho : FloatV.NaNEq (FloatV.add pe se) o) :
    LseMid p s M u v pe se o := by
  have hMn : M.isNaN = false :=
    FloatV.isNaN_of_mem hp hs (FloatV.MaxRel.mem hM hp hs)
  have hMfin : M.isFinite = true := by
    rcases eq_or_ne p negInfC with hpe | hpe
    · have hse : s ≠ negInfC := fun h => hne ⟨hpe, h⟩
      have hMeq : M = s := maxRel_eq_of_negInf_left (hpe ▸ hM) hs hse
      rw [hMeq]
      rcases hsf with hf | hf
      · exact hf
      · exact absurd hf hse
    · rcases eq_or_ne s negInfC with hse | hse
      · have hMeq : M = p := maxRel_eq_of_negInf_right (hse ▸ hM) hp hpe
        rw [hMeq]
        rcases hpf with hf | hf
        · exact hf
        · exact absurd hf hpe
      · rcases hM with h | ⟨-, -, hmem⟩
        · simp only [ite_true] at h
          have hmem2 : FloatV.max p s = p ∨ FloatV.max p s = s := FloatV.max_mem hp hs
          have hnan : (FloatV.max p s).isNaN = false := FloatV.isNaN_of_mem hp hs hmem2
          have hMeq : M = FloatV.max p s := h.eq hnan
          rw [hMeq]
          rcases hmem2 with h' | h'
          · rw [h']
            rcases hpf with hf | hf
            · exact hf
            · exact absurd hf hpe
          · rw [h']
            rcases hsf with hf | hf
            · exact hf
            · exact absurd hf hse
        · rcases hmem with h' | h'
          · rw [h']
            rcases hpf with hf | hf
            · exact hf
            · exact absurd hf hpe
          · rw [h']
            rcases hsf with hf | hf
            · exact hf
            · exact absurd hf hse
  have hMval : M.toReal = refMax p s :=
    maxRel_toReal_refMax hM hp hs hpf hsf hne
  have hinf_le : ∀ {x : F32}, x.isNaN = false → (negInfC : F32).toReal ≤ x.toReal := by
    intro x hx
    have h2 := FloatV.negInf_toRat_le (f := Format.binary32) Format.binary32_regular
      (by decide) hx
    have h3 : ((FloatV.negInf.toRat : ℚ) : ℝ) ≤ ((x.toRat : ℚ) : ℝ) := Rat.cast_le.mpr h2.1
    simpa [FloatV.toReal, negInfC_eq_negInf] using h3
  have hple : p.toReal ≤ refMax p s := by
    rcases eq_or_ne p negInfC with hpe | hpe
    · rw [hpe, refMax_negInf_left]
      exact hinf_le hs
    · rcases eq_or_ne s negInfC with hse | hse
      · rw [hse, refMax_negInf_right hpe]
      · rw [refMax_eq_max hpe hse]
        exact le_max_left _ _
  have hsle : s.toReal ≤ refMax p s := by
    rcases eq_or_ne p negInfC with hpe | hpe
    · rw [hpe, refMax_negInf_left]
    · rcases eq_or_ne s negInfC with hse | hse
      · rw [hse, refMax_negInf_right hpe]
        exact hinf_le hp
      · rw [refMax_eq_max hpe hse]
        exact le_max_right _ _
  obtain ⟨hpefin, hpe, hpe18⟩ :=
    exp_side_mid hE hN hpf hMfin hMval.symm hple hu he1
  obtain ⟨hsefin, hse, hse18⟩ :=
    exp_side_mid hE hN hsf hMfin hMval.symm hsle hv he2
  have hpen : |pe.toReal| ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
    rw [abs_le]
    have hE20 : (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
      norm_num
    exact ⟨by linarith [side_nonneg_aux p (refMax p s), (abs_le.mp hpe).1, hE20], hpe18⟩
  have hsen : |se.toReal| ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
    rw [abs_le]
    have hE20 : (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) ≤ 1 + (2 : ℝ) ^ (-18 : ℤ) := by
      norm_num
    exact ⟨by linarith [side_nonneg_aux s (refMax p s), (abs_le.mp hse).1, hE20], hse18⟩
  have hsum3R : |pe.toReal + se.toReal| ≤ 3 := by
    have h1 := abs_add_le pe.toReal se.toReal
    have h2 : (1 + (2 : ℝ) ^ (-18 : ℤ)) + (1 + (2 : ℝ) ^ (-18 : ℤ)) ≤ 3 := by norm_num
    linarith [h1, hpen, hsen, h2]
  have hsum3 : |(pe.toRat + se.toRat : ℚ)| ≤ 3 := by
    have hcast : ((|(pe.toRat + se.toRat : ℚ)| : ℚ) : ℝ) ≤ ((3 : ℚ) : ℝ) := by
      rw [Rat.cast_abs]
      push_cast
      simpa [FloatV.toReal] using hsum3R
    exact_mod_cast hcast
  have hmax3q : (3 : ℚ) ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  have hsum_abs : |(pe.toRat + se.toRat : ℚ)| ≤ Format.binary32.maxVal :=
    le_trans hsum3 hmax3q
  obtain ⟨haddfin, hadderr⟩ :=
    FloatV.add_err (f := Format.binary32) Format.binary32_regular hpefin hsefin hsum_abs
  have hnan_add : (FloatV.add pe se).isNaN = false := (FloatV.isFinite_iff.mp haddfin).1
  have hoeq : o = FloatV.add pe se := ho.eq hnan_add
  have hOfin : o.isFinite = true := by rw [hoeq]; exact haddfin
  have u_val : (Format.binary32.u : ℝ) = (2 : ℝ) ^ (-24 : ℤ) := by
    rw [Format.binary32_u]
    norm_num
  have eta_val : (Format.binary32.eta : ℝ) = (2 : ℝ) ^ (-150 : ℤ) := by
    rw [Format.binary32_eta]
    norm_num
  have hOerr : |o.toReal - (pe.toReal + se.toReal)| ≤
      (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have hcast : ((|(FloatV.add pe se).toRat - (pe.toRat + se.toRat)| : ℚ) : ℝ) ≤
        ((|(pe.toRat + se.toRat : ℚ)| * Format.binary32.u + Format.binary32.eta : ℚ) : ℝ) :=
      Rat.cast_le.mpr hadderr
    push_cast at hcast
    rw [u_val, eta_val] at hcast
    have h1 : |(FloatV.add pe se).toReal - (pe.toReal + se.toReal)| ≤
        |(pe.toReal + se.toReal)| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
      simpa [FloatV.toReal] using hcast
    have h2 : |(pe.toReal + se.toReal)| * (2 : ℝ) ^ (-24 : ℤ) ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hsum3R (by norm_num)
    have h3 : 3 * (2 : ℝ) ^ (-24 : ℤ) ≤ (2 : ℝ) ^ (-22 : ℤ) := by norm_num
    rw [hoeq]
    linarith
  have hside_abs : |(pe.toReal + se.toReal) - refSum p s| ≤
      (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) +
        ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ)) := by
    have he : (pe.toReal + se.toReal) - refSum p s =
        (pe.toReal - side p (refMax p s)) + (se.toReal - side s (refMax p s)) := by
      unfold refSum
      ring
    have htri := abs_add_le (pe.toReal - side p (refMax p s))
      (se.toReal - side s (refMax p s))
    rw [he]
    linarith [htri, hpe, hse]
  have hpo : |o.toReal - refSum p s| ≤ (2 : ℝ) ^ (-18 : ℤ) := by
    have he : o.toReal - refSum p s = (o.toReal - (pe.toReal + se.toReal)) +
        ((pe.toReal + se.toReal) - refSum p s) := by ring
    have htri := abs_add_le (o.toReal - (pe.toReal + se.toReal))
      ((pe.toReal + se.toReal) - refSum p s)
    have hnum : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) +
        ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) +
          ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ))) ≤ (2 : ℝ) ^ (-18 : ℤ) := by norm_num
    rw [he]
    linarith [htri, hOerr, hside_abs, hnum]
  have ho0 : 1 - (2 : ℝ) ^ (-18 : ℤ) ≤ o.toReal := by
    have h1 := (abs_le.mp hside_abs).1
    have h2 := (abs_le.mp hOerr).1
    have h3 := one_le_refSum_aux hne
    have hnum : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) +
        ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) +
          ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-21 : ℤ))) ≤ (2 : ℝ) ^ (-18 : ℤ) := by norm_num
    linarith
  have ho1 : o.toReal ≤ 2 + (2 : ℝ) ^ (-16 : ℤ) := by
    have h2 := (abs_le.mp hOerr).2
    have h4 : pe.toReal + se.toReal ≤ 2 + (2 : ℝ) ^ (-17 : ℤ) := by
      have h5 : (1 + (2 : ℝ) ^ (-18 : ℤ)) + (1 + (2 : ℝ) ^ (-18 : ℤ)) ≤
          2 + (2 : ℝ) ^ (-17 : ℤ) := by norm_num
      linarith [hpen, hsen, h5]
    have hnum : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) +
        (2 : ℝ) ^ (-17 : ℤ) ≤ (2 : ℝ) ^ (-16 : ℤ) := by norm_num
    linarith
  have hpe1' : pe.toReal ≤ 1 + (2 : ℝ) ^ (-16 : ℤ) := by
    have h5 : (1 : ℝ) + (2 : ℝ) ^ (-18 : ℤ) ≤ 1 + (2 : ℝ) ^ (-16 : ℤ) := by norm_num
    linarith [hpe18, h5]
  have hse1' : se.toReal ≤ 1 + (2 : ℝ) ^ (-16 : ℤ) := by
    have h5 : (1 : ℝ) + (2 : ℝ) ^ (-18 : ℤ) ≤ 1 + (2 : ℝ) ^ (-16 : ℤ) := by norm_num
    linarith [hse18, h5]
  exact ⟨hMn, Or.inl hMfin, hMfin, hMval, hpefin, hpe1', hsefin, hse1', hpe, hse,
    ho0, ho1, hOfin, hpo⟩

end Claims.MergeAttnStates
