import Claim
import TVA.Proof
import Proof.Bridge
import Proof.BetaChain
import Proof.FloatStep

/-!
# The value stored in `beta_output`, per lane

The `sigmoid` chain of one lane: widen `b` (exact), negate (exact), `tl.exp`, the IEEE `1 + ·`, the
default `float32` `/`, round to `bfloat16`, widen back (exact).
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-! ## Numeric facts about `float32` near its maximum -/

theorem b32_maxVal_eq : (Format.binary32.maxVal : ℚ) = (2:ℚ)^128 - (2:ℚ)^104 := by
  have h : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
  rw [h, Format.maxVal_eq]
  norm_num [Format.topSig, Format.topExp, Format.bias]

/-- `maxVal` of `float32` is at most `2^128`, in `ℝ`. -/
theorem b32_maxVal_le : (Format.binary32.maxVal : ℝ) ≤ (2:ℝ)^(128:ℤ) := by
  have h : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
  rw [h]
  norm_num [Format.maxVal, Format.topSig, Format.topExp, Format.bias, Format.pow2_eq]

/-- `2^128 < exp 90`. -/
theorem pow128_lt_exp90 : (2:ℝ)^(128:ℤ) < Real.exp 90 := by
  have h1 : (2.7:ℝ) < Real.exp 1 := lt_trans (by norm_num) Real.exp_one_gt_d9
  have h2 : Real.exp 90 = (Real.exp 1)^90 := by
    rw [show (90:ℝ) = ((90:ℕ) : ℝ) * 1 by norm_num, Real.exp_nat_mul]
  rw [h2]
  have h3 : (2 : ℝ) ^ (128:ℤ) < (2.7:ℝ)^90 := by norm_num
  exact lt_trans h3 (pow_lt_pow_left₀ h1 (by norm_num) (by norm_num))

/-- `2^-104 = 1 / 2^104`. -/
theorem b32_pow_neg104 : (2:ℚ)^(-104:ℤ) = 1/(2:ℚ)^104 := by norm_num

/-- The value of the largest finite magnitude `maxMag` is `maxVal`. -/
theorem b32_toRat_maxMag : (Format.binary32.toRat Format.binary32.maxMag : ℚ) =
    Format.binary32.maxVal := by
  have hreg : Format.binary32.Regular := by decide
  have hlt : Format.binary32.maxMag < Format.binary32.signBit := Format.maxMag_lt hreg
  have hsign : Format.binary32.sign Format.binary32.maxMag = false := by
    unfold Format.sign
    rw [Nat.div_eq_of_lt hlt]
    rfl
  unfold Format.toRat
  rw [hsign]
  simp only [Bool.false_eq_true, ite_false]
  rw [Format.mag_of_lt hlt]
  exact Format.magVal_maxMag hreg

/-- `maxMag` fits the `float32` width. -/
theorem b32_maxMag_lt_width : Format.binary32.maxMag < 2 ^ Format.binary32.width := by
  have hreg : Format.binary32.Regular := by decide
  have hlt : Format.binary32.maxMag < Format.binary32.signBit := Format.maxMag_lt hreg
  have hw : Format.binary32.signBit ≤ 2 ^ Format.binary32.width := by
    unfold Format.signBit Format.width
    have h : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
    rw [h]; norm_num
  omega

/-- `maxMag` is a finite pattern. -/
theorem b32_maxMag_finite : Format.binary32.isFinite Format.binary32.maxMag = true := by
  have hreg : Format.binary32.Regular := by decide
  refine Format.isFinite_of_mag_le hreg ?_
  rw [Format.mag_of_lt (Format.maxMag_lt hreg)]

/-- `maxMag` is not a NaN. -/
theorem b32_maxMag_notNaN : Format.binary32.isNaN Format.binary32.maxMag = false := by
  decide

/-- Just above `maxVal` (up to `maxVal + 1`) the quantum exponent is `104`. -/
theorem b32_qexp_top {q : ℚ} (h1 : (Format.binary32.maxVal : ℚ) < q)
    (h2 : q ≤ (Format.binary32.maxVal : ℚ) + 1) : Format.binary32.qexp q = 104 := by
  have hM := b32_maxVal_eq
  have hreg : Format.binary32.Regular := by decide
  have hqpos : 0 < q := by
    have hp := Format.maxVal_pos hreg
    linarith
  have hq2 : q < (2:ℚ)^(128:ℤ) := by
    rw [show (2:ℚ)^(128:ℤ) = (2:ℚ)^128 by norm_num]
    rw [hM] at h2
    linarith
  have hq1 : (2:ℚ)^(127:ℤ) ≤ q := by
    rw [show (2:ℚ)^(127:ℤ) = (2:ℚ)^127 by norm_num]
    rw [hM] at h1
    have : (2:ℚ)^127 ≤ (2:ℚ)^128 - (2:ℚ)^104 := by norm_num
    linarith
  have hlog : Int.log 2 q = 127 := by
    have hl1 := Format.log_lt_of_lt hqpos hq2
    have hl2 := Format.le_log_of_le hqpos hq1
    omega
  rw [Format.qexp_eq hqpos, hlog]
  have hb : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
  rw [hb]
  norm_num [Format.emin, Format.bias]

/-- Just above `maxVal` the rounded significand is `2^24 - 1`. -/
theorem b32_roundSig_top {q : ℚ} (h1 : (Format.binary32.maxVal : ℚ) < q)
    (h2 : q ≤ (Format.binary32.maxVal : ℚ) + 1) :
    Format.roundSig RoundMode.rn q 104 = 2^24 - 1 := by
  rw [Format.roundSig_eq]
  have hM := b32_maxVal_eq
  have hp := b32_pow_neg104
  have hpos : (0:ℚ) < (2:ℚ)^(-104:ℤ) := by rw [hp]; positivity
  have hMp : (Format.binary32.maxVal : ℚ) * (2:ℚ)^(-104:ℤ) = ((2^24 - 1 : ℕ) : ℚ) := by
    rw [hM, hp]
    field_simp
    norm_num
  have hMp1 : ((Format.binary32.maxVal : ℚ) + 1) * (2:ℚ)^(-104:ℤ) =
      ((2^24 - 1 : ℕ) : ℚ) + 1/(2:ℚ)^104 := by
    rw [hM, hp]
    field_simp
    norm_num
  have hxlo : ((2^24 - 1 : ℕ) : ℚ) < q * (2:ℚ)^(-104:ℤ) := by
    rw [← hMp]
    exact mul_lt_mul_of_pos_right h1 hpos
  have hxhi : q * (2:ℚ)^(-104:ℤ) ≤ ((2^24 - 1 : ℕ) : ℚ) + 1/(2:ℚ)^104 := by
    rw [← hMp1]
    exact mul_le_mul_of_nonneg_right h2 hpos.le
  have hqpos : 0 < q := by
    have hreg : Format.binary32.Regular := by decide
    have hpv := Format.maxVal_pos hreg
    linarith
  have hx0 : (0:ℚ) ≤ q * (2:ℚ)^(-104:ℤ) := mul_nonneg hqpos.le hpos.le
  have hfloor : ⌊q * (2:ℚ)^(-104:ℤ)⌋₊ = 2^24 - 1 := by
    refine (Nat.floor_eq_iff hx0).mpr ⟨le_of_lt hxlo, ?_⟩
    have h2' : ((2^24 - 1 : ℕ) : ℚ) + 1/(2:ℚ)^104 < ((2^24 - 1 : ℕ) : ℚ) + 1 := by
      norm_num
    exact lt_of_le_of_lt hxhi h2'
  have hc : Format.cmpq (q * (2:ℚ)^(-104:ℤ) - (((2^24 - 1 : ℕ)) : ℚ)) (1/2) = .lt := by
    unfold Format.cmpq
    have hsmall : (1:ℚ)/(2:ℚ)^104 < 1/2 := by norm_num
    have hlt : q * (2:ℚ)^(-104:ℤ) - (((2^24 - 1 : ℕ)) : ℚ) < 1/2 := by linarith
    rw [ite_eq_left hlt]
  rw [Format.rnat_eq, hfloor, hc]
  simp [Format.rint]

/-- A value in `(maxVal, maxVal + 1]` rounds to the largest finite magnitude, not to an overflow. -/
theorem b32_roundMag_top {q : ℚ} (h1 : (Format.binary32.maxVal : ℚ) < q)
    (h2 : q ≤ (Format.binary32.maxVal : ℚ) + 1) :
    Format.binary32.roundMag RoundMode.rn false q = Format.binary32.maxMag := by
  have hqexp := b32_qexp_top h1 h2
  have hsig := b32_roundSig_top h1 h2
  unfold Format.roundMag
  rw [hqexp, hsig]
  have hcond : (((2 ^ 24 - 1 : ℕ) : ℚ)) * Format.pow2 (104:ℤ) ≤ Format.binary32.maxVal := by
    rw [b32_maxVal_eq]
    norm_num [Format.pow2_eq]
  unfold Format.encode
  rw [ite_eq_left hcond]
  have hb : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
  rw [hb]
  norm_num [Format.pack, Format.maxMag, Format.topExp, Format.topSig, Format.bias]

/-- A rational just above the largest finite `float32` rounds, to nearest, to `maxMag`. -/
theorem b32_roundRat_top {zs : Bool} {q : ℚ} (h1 : (Format.binary32.maxVal : ℚ) < q)
    (h2 : q ≤ (Format.binary32.maxVal : ℚ) + 1) :
    Format.binary32.roundRat RoundMode.rn false zs q = Format.binary32.maxMag := by
  have hq0 : q ≠ 0 := by
    have hreg : Format.binary32.Regular := by decide
    have hp := Format.maxVal_pos hreg
    intro h; rw [h] at h1; linarith
  rw [Format.roundRat_of_ne (f := Format.binary32) hq0]
  have hqpos : 0 < q := by
    have hreg : Format.binary32.Regular := by decide
    have hp := Format.maxVal_pos hreg
    linarith
  rw [ite_eq_right (by linarith : ¬ q < 0)]
  rw [abs_of_pos hqpos]
  rw [b32_roundMag_top h1 h2]
  rfl

/-- The `exp` rule's lower bound: `tl.exp`'s result is at least `-1/2`. -/
theorem eb_lower {eb nb : ℝ} (hnb_ub : nb ≤ 90)
    (heberr : |eb - Real.exp nb| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb|) * Real.exp nb + 2 ^ (-140 : ℤ)) :
    -(1/2 : ℝ) ≤ eb := by
  set e : ℝ := Real.exp nb with hedef
  have hepos : 0 < e := by rw [hedef]; exact Real.exp_pos nb
  set X : ℝ := (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb|) * e + 2 ^ (-140 : ℤ) with hX
  have hkey : X ≤ e + 1/2 := by
    rw [hX]
    by_cases hnb : 0 ≤ nb
    · have habs : |nb| = nb := abs_of_nonneg hnb
      have hc : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb|) ≤ (1/2 : ℝ) := by
        rw [habs]
        nlinarith [hnb_ub]
      have h1 : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb|) * e ≤ (1/2) * e :=
        mul_le_mul_of_nonneg_right hc hepos.le
      have h2 : (2:ℝ) ^ (-140 : ℤ) ≤ (1/2:ℝ) := by norm_num
      nlinarith
    · have habs : |nb| = -nb := abs_of_neg (not_le.mp hnb)
      have hmul : |nb| * e ≤ 1 := by
        rw [habs, hedef]
        have heq : Real.exp nb = Real.exp (-(-nb)) := by ring_nf
        rw [heq, Real.exp_neg]
        rw [← div_eq_mul_inv]
        rw [div_le_one (Real.exp_pos (-nb))]
        linarith [Real.add_one_le_exp (-nb)]
      have h1 : 2 ^ (-22 : ℤ) * e ≤ (1/4:ℝ) * e := by
        have h4 : (2:ℝ) ^ (-22 : ℤ) ≤ 1/4 := by norm_num
        exact mul_le_mul_of_nonneg_right h4 hepos.le
      have h2 : 2 ^ (-23 : ℤ) * (|nb| * e) ≤ (1/2:ℝ) := by
        have h3 : (2:ℝ) ^ (-23 : ℤ) ≤ 1/2 := by norm_num
        have h4 := mul_le_mul h3 hmul (mul_nonneg (abs_nonneg nb) hepos.le)
          (by norm_num : (0:ℝ) ≤ 1/2)
        simpa using h4
      have hexp : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb|) * e
          = 2 ^ (-22 : ℤ) * e + 2 ^ (-23 : ℤ) * (|nb| * e) := by ring
      rw [hexp]
      nlinarith
  have hlow : e - X ≤ eb := by
    have h := (abs_le.mp heberr).1
    linarith
  linarith

/-- The denominator of the sigmoid is at least `1/4`: its rounding error is far smaller than
`1 + eb ≥ 1/2`. -/
theorem den_quarter {d eb c eps : ℝ} (hc : c ≤ 1/4) (hge : (1:ℝ)/2 ≤ 1 + eb)
    (heps : eps ≤ 1/8) (h : |d - (1 + eb)| ≤ c * (1 + eb) + eps) : (1:ℝ)/4 ≤ d := by
  set D : ℝ := 1 + eb with hD
  set K : ℝ := c * D with hK
  have hDpos : (0:ℝ) ≤ D := by rw [hD]; linarith
  have hDge : (1:ℝ)/2 ≤ D := by rw [hD]; exact hge
  have hlow := (abs_le.mp h).1
  have hcd : K ≤ (1/4) * D := by
    rw [hK]
    exact mul_le_mul_of_nonneg_right hc hDpos
  linarith

/-- `bfloat16`'s `maxVal` is at least `8`. -/
theorem bf16_le_maxVal : (8:ℚ) ≤ Format.bfloat16.maxVal := by
  have h : Format.bfloat16 = ⟨8, 7, Format.Specials.ieee⟩ := rfl
  rw [h]
  norm_num [Format.maxVal, Format.topSig, Format.topExp, Format.bias, Format.pow2_eq]

/-- `1 ≤ 2^-24 · maxVal` of `float32`. -/
theorem b32_one_le_maxVal : (1:ℝ) ≤ (2:ℝ)^(-24:ℤ) * ((Format.binary32.maxVal : ℚ) : ℝ) := by
  have h' : ((Format.binary32.maxVal : ℚ) : ℝ) = (2:ℝ)^128 - (2:ℝ)^104 := by
    rw [b32_maxVal_eq]
    push_cast
    norm_num
  rw [h']
  norm_num

/-- **One lane's `beta_output`.** -/
theorem lane_beta {P : Platform} (hexp : Device.ExpF32 P) (hfull : Device.DivFull P)
    {bv : BF16} (hbv : bv.isFinite = true) (hroom : ExpRoom (-bv.toReal))
    {b32 nb eb den q : F32} {b16 : BF16} {r : F32}
    (hb32 : FloatV.NaNEq (FloatV.cvt (g := Format.binary32) bv) b32)
    (hnb : FloatV.NaNEq (FloatV.neg b32) nb)
    (heb : P.math .exp .binary32 nb eb)
    (hden : FloatV.NaNEq (FloatV.add oneF32 eb) den)
    (hq : P.div .binary32 oneF32 den q)
    (hb16 : FloatV.NaNEq (FloatV.cvtWith RoundMode.rn (g := Format.bfloat16) q) b16)
    (hr : FloatV.NaNEq (FloatV.cvt (g := Format.binary32) b16) r) :
    r.isFinite = true ∧ |r.toReal - sigRef bv.toReal| ≤ bTol bv.toReal := by
  -- 1. the widening of `bv` is exact
  obtain ⟨hcvt_fin, hcvt_val⟩ := cvt_up hbv
  have hb32eq : b32 = FloatV.cvt (g := Format.binary32) bv := FloatV.NaNEq.eq_of_isFinite hb32 hcvt_fin
  have hb32fin : b32.isFinite = true := by rw [hb32eq]; exact hcvt_fin
  have hb32val : b32.toReal = bv.toReal := by rw [hb32eq]; exact hcvt_val
  -- 2. the negation is exact
  obtain ⟨hnbfin, hnbval⟩ := neg_exact hb32fin hnb
  have hnbv : nb.toReal = -bv.toReal := by rw [hnbval, hb32val]
  -- 3. `tl.exp`
  have hroom_nb : ExpRoom nb.toReal := by rw [hnbv]; exact hroom
  obtain ⟨hebfin, heberr⟩ := exp_apply hexp heb hnbfin hroom_nb
  have hnb_ub : nb.toReal ≤ 90 := by
    have h := hroom_nb
    simp only [ExpRoom] at h
    set E : ℝ := Real.exp nb.toReal with hE
    set C : ℝ := 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb.toReal| with hC
    have hCnn : (0:ℝ) ≤ C := by rw [hC]; positivity
    have hEpos : 0 < E := by rw [hE]; exact Real.exp_pos _
    have h140 : (0:ℝ) ≤ 2 ^ (-140 : ℤ) := by norm_num
    have hexp_le : E ≤ (Format.binary32.maxVal : ℝ) := by nlinarith
    have hlt : E < Real.exp 90 :=
      lt_of_le_of_lt (le_trans hexp_le b32_maxVal_le) pow128_lt_exp90
    have hlt' : Real.exp nb.toReal < Real.exp 90 := by rw [← hE]; exact hlt
    exact (Real.exp_lt_exp.mp hlt').le
  have heb_lb : -(1/2 : ℝ) ≤ eb.toReal := eb_lower hnb_ub heberr
  have hone : oneF32.toReal = 1 := by simp only [FloatV.toReal, oneF32_toRat]; norm_num
  have hq_ge : (1:ℝ)/2 ≤ 1 + eb.toReal := by linarith
  -- 4. the IEEE `1 + ·`
  have hdenfacts : den.isFinite = true ∧
      |den.toReal - (1 + eb.toReal)| ≤
        2 ^ (-24 : ℤ) * (1 + eb.toReal) + 2 ^ (-150 : ℤ) := by
    by_cases hroom2 : |oneF32.toRat + eb.toRat| ≤ Format.binary32.maxVal
    · obtain ⟨hf, he⟩ := add_exact_err_fin oneF32_isFinite hebfin hroom2 hden
      refine ⟨hf, ?_⟩
      rw [show oneF32.toReal = 1 from hone] at he
      have hpos : (0:ℝ) ≤ 1 + eb.toReal := by linarith
      rw [abs_of_nonneg hpos] at he
      calc |den.toReal - (1 + eb.toReal)|
          ≤ (1 + eb.toReal) * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := he
        _ = 2 ^ (-24 : ℤ) * (1 + eb.toReal) + 2 ^ (-150 : ℤ) := by ring
    · push Not at hroom2
      rw [oneF32_toRat] at hroom2
      have hqposR : (0:ℝ) < 1 + eb.toReal := by linarith
      have hqposQ : (0:ℚ) < 1 + eb.toRat := by
        have h : (0:ℝ) < 1 + ((eb.toRat : ℚ) : ℝ) := by
          simpa only [FloatV.toReal] using hqposR
        exact_mod_cast h
      have h1q : (Format.binary32.maxVal : ℚ) < 1 + eb.toRat := by
        have h := hroom2
        rw [abs_of_pos hqposQ] at h
        exact h
      have heb_ub : eb.toRat ≤ Format.binary32.maxVal := by
        have h : |eb.toRat| ≤ Format.binary32.maxVal :=
          FloatV.abs_toRat_le_maxVal (f := Format.binary32) (by decide) hebfin
        exact le_trans (le_abs_self _) h
      have h2q : (1:ℚ) + eb.toRat ≤ (Format.binary32.maxVal : ℚ) + 1 := by linarith
      have hround : Format.binary32.roundRat RoundMode.rn false
          (Format.binary32.sign oneF32.pat && Format.binary32.sign eb.pat)
          (oneF32.toRat + eb.toRat) = Format.binary32.maxMag := by
        rw [oneF32_toRat]
        exact b32_roundRat_top h1q h2q
      have hadd : FloatV.add oneF32 eb = FloatV.ofPat Format.binary32.maxMag := by
        have h := hround
        simp only [FloatV.toRat] at h
        unfold FloatV.add
        rw [Format.add_of_finite RoundMode.rn oneF32_isFinite hebfin, h]
      have hnan : (FloatV.add oneF32 eb).isNaN = false := by
        rw [hadd, FloatV.isNaN_ofPat b32_maxMag_lt_width]
        exact b32_maxMag_notNaN
      have hdeneq : den = FloatV.add oneF32 eb := FloatV.NaNEq.eq hden hnan
      have hdenfin : den.isFinite = true := by
        rw [hdeneq, hadd, FloatV.isFinite_ofPat b32_maxMag_lt_width]
        exact b32_maxMag_finite
      have hdenval : den.toRat = Format.binary32.maxVal := by
        rw [hdeneq, hadd, FloatV.toRat_ofPat b32_maxMag_lt_width, b32_toRat_maxMag]
      have hdenReal : den.toReal = ((Format.binary32.maxVal : ℚ) : ℝ) := by
        simp only [FloatV.toReal, hdenval]
      have hqRgt : ((Format.binary32.maxVal : ℚ) : ℝ) < 1 + ((eb.toRat : ℚ) : ℝ) := by
        exact_mod_cast h1q
      have hqRle : (1:ℝ) + ((eb.toRat : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) + 1 := by
        exact_mod_cast h2q
      refine ⟨hdenfin, ?_⟩
      rw [hdenReal]
      simp only [FloatV.toReal]
      have hdiff : |((Format.binary32.maxVal : ℚ) : ℝ) - (1 + ((eb.toRat : ℚ) : ℝ))| ≤ 1 := by
        have hneg : ((Format.binary32.maxVal : ℚ) : ℝ) - (1 + ((eb.toRat : ℚ) : ℝ)) < 0 := by
          linarith [hqRgt]
        rw [abs_of_neg hneg]
        linarith
      have hrhs : (1:ℝ) ≤ 2 ^ (-24 : ℤ) * (1 + ((eb.toRat : ℚ) : ℝ)) + 2 ^ (-150 : ℤ) := by
        have h := b32_one_le_maxVal
        have h24 : (0:ℝ) ≤ 2 ^ (-24 : ℤ) := by positivity
        have hstep : (2:ℝ) ^ (-24 : ℤ) * ((Format.binary32.maxVal : ℚ) : ℝ) ≤
            2 ^ (-24 : ℤ) * (1 + ((eb.toRat : ℚ) : ℝ)) :=
          mul_le_mul_of_nonneg_left (le_of_lt hqRgt) h24
        have h3 : (0:ℝ) ≤ 2 ^ (-150 : ℤ) := by positivity
        linarith
      exact le_trans hdiff hrhs
  have hdenpos : (1:ℝ)/4 ≤ den.toReal :=
    den_quarter (c := 2 ^ (-24 : ℤ)) (eps := 2 ^ (-150 : ℤ))
      (by norm_num) hq_ge (by norm_num) hdenfacts.2
  have hdenne : den.toReal ≠ 0 := by linarith
  -- 5. the `float32` division
  have hx_abs : |(1:ℝ) / den.toReal| ≤ 4 := by
    have hqpos : (0:ℝ) < den.toReal := by linarith
    have h1 : 1 / den.toReal ≤ 4 := by
      rw [div_le_iff₀ hqpos]
      linarith
    rw [abs_of_nonneg (by positivity)]
    exact h1
  have hq_bound : |oneF32.toReal / den.toReal| ≤ (2:ℝ)^(126:ℤ) := by
    rw [hone]
    have hx_le : (1:ℝ) / den.toReal ≤ 4 := (abs_le.mp hx_abs).2
    have hnn : (0:ℝ) ≤ 1 / den.toReal := by positivity
    rw [abs_of_nonneg hnn]
    have h3 : (4:ℝ) ≤ (2:ℝ)^(126:ℤ) := by norm_num
    linarith
  obtain ⟨hqfin, hqerr⟩ := div_apply hfull hq oneF32_isFinite hdenfacts.1 hdenne hq_bound
  have hqerr' : |q.toReal - 1 / den.toReal| ≤
      2 ^ (-22 : ℤ) * |1 / den.toReal| + 2 ^ (-148 : ℤ) := by
    rw [show oneF32.toReal = 1 from hone] at hqerr
    exact hqerr
  -- 6. the `bfloat16` rounding
  have hq4 : |q.toReal| ≤ 8 := by
    have hrle : q.toReal - 1 / den.toReal ≤
        2 ^ (-22 : ℤ) * |1 / den.toReal| + 2 ^ (-148 : ℤ) := (abs_le.mp hqerr').2
    have hrge : -(2 ^ (-22 : ℤ) * |1 / den.toReal| + 2 ^ (-148 : ℤ)) ≤
        q.toReal - 1 / den.toReal := (abs_le.mp hqerr').1
    have hx_le : (1:ℝ) / den.toReal ≤ 4 := (abs_le.mp hx_abs).2
    have hx_ge : -(4:ℝ) ≤ 1 / den.toReal := (abs_le.mp hx_abs).1
    have hmul : 2 ^ (-22 : ℤ) * |1 / den.toReal| ≤ 1 := by
      have h22 : (2:ℝ) ^ (-22 : ℤ) ≤ 1/4 := by norm_num
      have h4 : (0:ℝ) ≤ |1 / den.toReal| := abs_nonneg _
      calc 2 ^ (-22 : ℤ) * |1 / den.toReal| ≤ (1/4) * 4 := mul_le_mul h22 hx_abs h4 (by norm_num)
        _ = 1 := by norm_num
    have h148 : (2:ℝ) ^ (-148 : ℤ) ≤ 1 := by norm_num
    have hqle : q.toReal ≤ 8 := by linarith
    have hqge : -(8:ℝ) ≤ q.toReal := by linarith
    exact abs_le.mpr ⟨hqge, hqle⟩
  have hqmax : |q.toRat| ≤ Format.bfloat16.maxVal := by
    have h4 : |((q.toRat : ℚ) : ℝ)| ≤ 8 := by
      simpa only [FloatV.toReal] using hq4
    have h4' : |q.toRat| ≤ (8:ℚ) := by exact_mod_cast h4
    exact le_trans h4' bf16_le_maxVal
  obtain ⟨hb16fin, hb16err⟩ := cvt_down hqfin hqmax hb16
  have hq16 : |r.toReal - q.toReal| ≤
      (2:ℝ) ^ (-8 : ℤ) * |q.toReal| + (2:ℝ) ^ (-134 : ℤ) := by
    have hbv' : r.toReal = b16.toReal := by
      obtain ⟨hrfin_cvt, hrval_cvt⟩ := cvt_up hb16fin
      have hreq : r = FloatV.cvt (g := Format.binary32) b16 :=
        FloatV.NaNEq.eq_of_isFinite hr hrfin_cvt
      rw [hreq]
      exact hrval_cvt
    rw [hbv']
    calc |b16.toReal - q.toReal| ≤ |q.toReal| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := hb16err
      _ = 2 ^ (-8 : ℤ) * |q.toReal| + 2 ^ (-134 : ℤ) := by ring
  -- 7. the real-arithmetic chain
  have hmain := beta_bound (b0 := bv.toReal) (eb := eb.toReal) (nb := nb.toReal)
    (den := den.toReal) (q := q.toReal) (q16 := r.toReal)
    hnbv hnb_ub heberr hdenfacts.2 hqerr' hq16
  have hrfin : r.isFinite = true := by
    obtain ⟨hrfin_cvt, -⟩ := cvt_up hb16fin
    have hreq : r = FloatV.cvt (g := Format.binary32) b16 := FloatV.NaNEq.eq_of_isFinite hr hrfin_cvt
    rw [hreq]
    exact hrfin_cvt
  exact ⟨hrfin, hmain⟩

end Claims.GdnGating
