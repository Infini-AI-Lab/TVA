import AttnBwd.Spec

/-!
# `attn_bwd_f16`: the error of one lane and of one accumulation step

* The constants' values (`u32_eq`, …) and the library's error lemmas over the reals
  (`addRel_errR`, `mul_errR`, `cvt16_errR`).
* `two_rpow_sub_le`: `|2^a − 2^b| ≤ 2^b·(2^e − 1)` when `|a − b| ≤ e`.
* `lane`: one pair `(n, m)` of a block, from the two `tl.dot` results to the `float16` `P` and
  `dS` (lines 293–309 and 345–355): the errors `p16Err`, `ds16Err` and bounds `p16B`, `ds16B` of
  `Spec`, from `Rules.Num.Exp2F32`; a masked lane is exactly `0`.
* `dot0`: a `tl.dot` with the zero accumulator (lines 293, 307, 345, 353), from
  `Rules.Num.DotF16`.
* `AccI`, `accI_dotAdd`: the accumulator invariant and one step `acc += tl.dot(a, b)` (lines
  303, 310, 358), from `Rules.Num.DotF16` (`DotLoop.dotAdd_step`).
-/

namespace Kernels.AttnBwd

open TVA
open TVA.Math.Float

/-! ## Constants -/

theorem u32_eq : u32 = 2 ^ (-24 : ℤ) := by unfold u32; rw [Format.binary32_u]; push_cast; rfl

theorem eta32_eq : eta32 = 2 ^ (-150 : ℤ) := by unfold eta32; rw [Format.binary32_eta]; push_cast; rfl

theorem u16_eq : u16 = 2 ^ (-11 : ℤ) := by unfold u16; rw [Format.binary16_u]; push_cast; rfl

theorem eta16_eq : eta16 = 2 ^ (-25 : ℤ) := by unfold eta16; rw [Format.binary16_eta]; push_cast; rfl

theorem rH_le {d : Inst} (hd : d.ok) : 0 ≤ rH d ∧ rH d ≤ 2 ^ (-13 : ℤ) := by
  have h : d.HEAD_DIM ≤ 256 := by rcases hd.hd with h | h | h | h | h <;> omega
  have h' : (d.HEAD_DIM : ℝ) ≤ 256 := by exact_mod_cast h
  unfold rH Rules.Num.relDot16
  push_cast
  constructor
  · positivity
  · have : (2 : ℝ) ^ (-13 : ℤ) = 512 * 2 ^ (-22 : ℤ) := by norm_num
    rw [this]
    have h2 : (0 : ℝ) < 2 ^ (-22 : ℤ) := by positivity
    nlinarith

theorem aH_le {d : Inst} (hd : d.ok) : 0 ≤ aH d ∧ aH d ≤ 2 ^ (-117 : ℤ) := by
  have h : d.HEAD_DIM ≤ 256 := by rcases hd.hd with h | h | h | h | h <;> omega
  have h' : (d.HEAD_DIM : ℝ) ≤ 256 := by exact_mod_cast h
  unfold aH Rules.Num.absDot16
  push_cast
  constructor
  · positivity
  · have : (2 : ℝ) ^ (-117 : ℤ) = 512 * 2 ^ (-126 : ℤ) := by norm_num
    rw [this]
    have h2 : (0 : ℝ) < 2 ^ (-126 : ℤ) := by positivity
    nlinarith

theorem rhoA_eq : rhoA = 34 * 2 ^ (-22 : ℤ) + 3 * 2 ^ (-24 : ℤ) * (1 + 34 * 2 ^ (-22 : ℤ)) := by
  unfold rhoA DotLoop.stepRel Rules.Num.relDot16
  rw [Format.binary32_u]; push_cast; norm_num

theorem rhoA_nn : 0 ≤ rhoA := by rw [rhoA_eq]; positivity

theorem rhoA_le : rhoA ≤ 2 ^ (-16 : ℤ) := by rw [rhoA_eq]; norm_num

theorem alphaA_eq : alphaA = 34 * 2 ^ (-126 : ℤ) * (1 + 3 * 2 ^ (-24 : ℤ)) + 4 * 2 ^ (-150 : ℤ) := by
  unfold alphaA DotLoop.stepAbs Rules.Num.absDot16
  rw [Format.binary32_u, Format.binary32_eta]; push_cast; norm_num

theorem alphaA_nn : 0 ≤ alphaA := by rw [alphaA_eq]; positivity

theorem alphaA_le : alphaA ≤ 2 ^ (-100 : ℤ) := by rw [alphaA_eq]; norm_num

theorem maxVal32_ge : (2 : ℚ) ^ 127 ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

/-! ## The library's error lemmas over the reals -/

theorem toReal_def {f : Format} (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

/-- A result that is equal up to NaN to a finite value is that value. -/
theorem nanEq_fin {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y) (hx : x.isFinite = true) :
    y = x := h.eq (FloatV.isFinite_iff.mp hx).1

/-- A contractible `a − b` (`tl.subc`): finite, within `3u32·(|a| + |b|) + 4eta32` of `a − b`. -/
theorem addRel_errR {fusion : Bool} {a b r : F32} (h : AddRel fusion a (FloatV.neg b) r)
    (ha : a.isFinite = true) (hb : b.isFinite = true) (hM : |a.toReal| + |b.toReal| ≤ 2 ^ 126 + 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - (a.toReal - b.toReal)| ≤ 3 * u32 * (|a.toReal| + |b.toReal|) + 4 * eta32 := by
  have hM' : |a.toRat| + |b.toRat| ≤ 2 ^ 126 + 2 ^ 100 := by
    have : ((|a.toRat| + |b.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 + 2 ^ 100 : ℚ) : ℝ) := by
      push_cast; simpa [toReal_def] using hM
    exact_mod_cast this
  have hroom : |a.toRat| + |(FloatV.neg b).toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [FloatV.toRat_neg, abs_neg, Format.binary32_maxVal, Format.binary32_eta]
    refine hM'.trans ?_
    norm_num
  obtain ⟨hf, he⟩ := FloatV.AddRel.err Format.binary32_regular DotLoop.binary32_specials
    (by rw [Format.binary32_u]; norm_num) h ha (by rw [FloatV.isFinite_neg]; exact hb) hroom
  refine ⟨hf, ?_⟩
  rw [FloatV.toRat_neg, abs_neg] at he
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  simp only [toReal_def, u32, eta32]
  convert this using 2

/-- An IEEE product of finite operands with room: finite, within `u32·|a·b| + eta32`. -/
theorem mul_errR {a b : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hM : |a.toReal * b.toReal| ≤ 2 ^ 126) :
    (FloatV.mul a b).isFinite = true ∧
      |(FloatV.mul a b).toReal - a.toReal * b.toReal| ≤ u32 * |a.toReal * b.toReal| + eta32 := by
  have hM' : |a.toRat * b.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat * b.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast; simpa [toReal_def] using hM
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.mul_err Format.binary32_regular ha hb (hM'.trans (by
    rw [Format.binary32_maxVal]; norm_num))
  refine ⟨hf, ?_⟩
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  simp only [toReal_def, u32, eta32]
  linarith

/-- A rounding to `float16` with room: finite, within `u16·|a| + eta16`. -/
theorem cvt16_errR {a : F32} (ha : a.isFinite = true) (hM : |a.toReal| ≤ 65504) :
    (FloatV.cvt a : F16).isFinite = true ∧
      |(FloatV.cvt a : F16).toReal - a.toReal| ≤ u16 * |a.toReal| + eta16 := by
  have hM' : |a.toRat| ≤ Format.binary16.maxVal := by
    rw [Format.binary16_maxVal]
    have : ((|a.toRat| : ℚ) : ℝ) ≤ ((65504 : ℚ) : ℝ) := by push_cast; simpa [toReal_def] using hM
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.cvt_err (g := Format.binary16) Format.binary16_regular ha hM'
  refine ⟨hf, ?_⟩
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  simp only [toReal_def, u16, eta16]
  linarith

/-! ## Zeros -/

theorem f32_zero_lit : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by
  unfold FloatV.litF32
  rw [ite_eq_left]
  rw [FloatV.isZero_ofPat (by norm_num)]
  decide

theorem zero32 : (FloatV.ofPat 0 : F32).isFinite = true ∧ (FloatV.ofPat 0 : F32).toRat = 0 :=
  DotLoop.val?_f32.mp DotLoop.zero_val

/-- The product of a finite zero and a finite value is a finite zero. -/
theorem mul_zero_l {x y : F32} (hx : x.isFinite = true) (h0 : x.toRat = 0) (hy : y.isFinite = true) :
    (FloatV.mul x y).isFinite = true ∧ (FloatV.mul x y).toRat = 0 := by
  unfold FloatV.mul
  rw [Format.mul_of_finite .rn hx hy]
  have e : Format.binary32.toRat x.pat * Format.binary32.toRat y.pat = 0 := by
    show x.toRat * y.toRat = 0
    rw [h0, zero_mul]
  rw [e]
  obtain ⟨hlt, hf, hr⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn) (sat := false)
    (zs := (Format.binary32.sign x.pat != Format.binary32.sign y.pat)) Format.binary32_regular
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hf, hr⟩

/-- A finite zero rounds to a finite zero in `float16`. -/
theorem cvt_zero {x : F32} (hx : x.isFinite = true) (h0 : x.toRat = 0) :
    (FloatV.cvt x : F16).isFinite = true ∧ (FloatV.cvt x : F16).toRat = 0 := by
  unfold FloatV.cvt
  rw [Format.cvt_of_finite _ _ _ hx]
  have e : Format.binary32.toRat x.pat = 0 := h0
  rw [e]
  obtain ⟨hlt, hf, hr⟩ := Format.roundRat_zero_spec (f := Format.binary16) (rnd := .rn) (sat := false)
    (zs := Format.binary32.sign x.pat) Format.binary16_regular
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hf, hr⟩

/-! ## Powers of two -/

theorem two_rpow_pos (x : ℝ) : 0 < (2 : ℝ) ^ x := Real.rpow_pos_of_pos (by norm_num) x

theorem two_rpow_le {a b : ℝ} (h : a ≤ b) : (2 : ℝ) ^ a ≤ 2 ^ b :=
  Real.rpow_le_rpow_of_exponent_le (by norm_num) h

/-- `2^a` within `2^b·(2^e − 1)` of `2^b` when `|a − b| ≤ e`. -/
theorem two_rpow_sub_le {a b e : ℝ} (h : |a - b| ≤ e) :
    |(2 : ℝ) ^ a - 2 ^ b| ≤ 2 ^ b * (2 ^ e - 1) := by
  have hab : (2 : ℝ) ^ a = 2 ^ b * 2 ^ (a - b) := by
    rw [← Real.rpow_add (by norm_num)]; congr 1; ring
  have hb := two_rpow_pos b
  have he0 : 0 ≤ e := (abs_nonneg _).trans h
  rw [hab, show (2 : ℝ) ^ b * 2 ^ (a - b) - 2 ^ b = 2 ^ b * (2 ^ (a - b) - 1) by ring, abs_mul,
    abs_of_pos hb]
  refine mul_le_mul_of_nonneg_left ?_ hb.le
  have h1 := abs_le.mp h
  rcases le_total 0 (a - b) with ht | ht
  · rw [abs_of_nonneg (by linarith [Real.one_le_rpow (x := 2) (by norm_num) ht])]
    linarith [two_rpow_le h1.2]
  · have hlt : (2 : ℝ) ^ (a - b) ≤ 1 := by
      have := two_rpow_le ht
      rwa [Real.rpow_zero] at this
    rw [abs_of_nonpos (by linarith)]
    have hneg : (2 : ℝ) ^ (-e) ≤ 2 ^ (a - b) := two_rpow_le (by linarith)
    rw [Real.rpow_neg (by norm_num)] at hneg
    have hpe := two_rpow_pos e
    have hge : (1 : ℝ) ≤ 2 ^ e := Real.one_le_rpow (by norm_num) he0
    -- `1 − 1/2^e ≤ 2^e − 1`, as `(2^e − 1)^2 ≥ 0`
    have key : 1 - (2 ^ e)⁻¹ ≤ (2 : ℝ) ^ e - 1 := by
      have hinv : (2 : ℝ) ^ e * (2 ^ e)⁻¹ = 1 := mul_inv_cancel₀ hpe.ne'
      have hi0 : 0 < ((2 : ℝ) ^ e)⁻¹ := inv_pos.mpr hpe
      nlinarith [mul_nonneg (sq_nonneg ((2 : ℝ) ^ e - 1)) hi0.le]
    linarith

theorem two_rpow_15 : (2 : ℝ) ^ (15 : ℝ) = 2 ^ 15 := by
  rw [show (15 : ℝ) = ((15 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]

/-! ## One lane -/

/-- The error of one pair, from the two `tl.dot` results `s ≈ Σ K·Q` and `dp ≈ Σ V·dO` to the
`float16` `P` and `dS`. `al` says whether the lane is kept (`Allowed`): a masked lane holds
`+0` (`tl.where(mask, pT, 0.0)`), and its `P` and `dS` are exactly `0`. -/
theorem lane {P : Platform} (hexp : Rules.Num.Exp2F32 P) {fusion : Bool} {d : Inst} (hd : d.ok)
    {s L dp Dl x p0 y ds : F32} {p16 ds16 : F16} (al : Bool) {S A DP AD X EX Y EY : ℝ}
    (hA : A ≤ 2 ^ 40) (hSA : |S| ≤ A) (hAD : AD ≤ 2 ^ 40) (hDPA : |DP| ≤ AD)
    (hsf : s.isFinite = true) (hs : |s.toReal - S| ≤ rH d * A + aH d)
    (hLf : L.isFinite = true) (hL : |L.toRat| ≤ 2 ^ 126)
    (hdpf : dp.isFinite = true) (hdp : |dp.toReal - DP| ≤ rH d * AD + aH d)
    (hDlf : Dl.isFinite = true) (hDl : |Dl.toRat| ≤ 2 ^ 126)
    (hX : X = S - L.toReal) (hEX : EX = eSub d S A L.toReal)
    (hY : Y = DP - Dl.toReal) (hEY : EY = eSub d DP AD Dl.toReal)
    (hx : AddRel fusion s (FloatV.neg L) x) (hp0 : P.math .exp2 .binary32 x p0)
    (hy : AddRel fusion dp (FloatV.neg Dl) y)
    (hp16 : FloatV.NaNEq (FloatV.cvt (if al then p0 else FloatV.ofPat 0)) p16)
    (hds : FloatV.NaNEq (FloatV.mul (if al then p0 else FloatV.ofPat 0) y) ds)
    (hds16 : FloatV.NaNEq (FloatV.cvt ds) ds16)
    (hfit : al = true → X + EX ≤ 15 ∧
      ((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) * (|Y| + EY) ≤ 2 ^ 15) :
    p16.isFinite = true ∧ ds16.isFinite = true ∧
    (al = false → p16.toReal = 0 ∧ ds16.toReal = 0) ∧
    (al = true →
      |p16.toReal - 2 ^ X| ≤ ((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) - 2 ^ X +
        u16 * ((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) + eta16 ∧
      |p16.toReal| ≤ (1 + u16) * ((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) + eta16 ∧
      |ds16.toReal - 2 ^ X * Y| ≤
        ((((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) - 2 ^ X) * (|Y| + EY) +
          2 ^ X * EY + u32 * (((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) * (|Y| + EY)) + eta32) +
        u16 * ((1 + u32) * (((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) * (|Y| + EY)) + eta32) +
          eta16 ∧
      |ds16.toReal| ≤ (1 + u16) * ((1 + u32) *
        (((1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ)) * (|Y| + EY)) + eta32) + eta16) := by
  obtain ⟨hrH0, hrH⟩ := rH_le hd
  obtain ⟨haH0, haH⟩ := aH_le hd
  have hu32 : u32 = 2 ^ (-24 : ℤ) := u32_eq
  have heta32 : eta32 = 2 ^ (-150 : ℤ) := eta32_eq
  have hu16 : u16 = 2 ^ (-11 : ℤ) := u16_eq
  have heta16 : eta16 = 2 ^ (-25 : ℤ) := eta16_eq
  have hu32n : (0 : ℝ) ≤ u32 := by rw [hu32]; positivity
  have heta32n : (0 : ℝ) ≤ eta32 := by rw [heta32]; positivity
  have hu16n : (0 : ℝ) ≤ u16 := by rw [hu16]; positivity
  have heta16n : (0 : ℝ) ≤ eta16 := by rw [heta16]; positivity
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  have hAD0 : 0 ≤ AD := (abs_nonneg _).trans hDPA
  -- sizes of the dot results
  have hrHA : rH d * A ≤ 2 ^ 27 := by
    have := mul_le_mul hrH hA hA0 (by norm_num)
    refine this.trans (le_of_eq ?_); norm_num
  have hrHAD : rH d * AD ≤ 2 ^ 27 := by
    have := mul_le_mul hrH hAD hAD0 (by norm_num)
    refine this.trans (le_of_eq ?_); norm_num
  have haH1 : aH d ≤ 1 := haH.trans (by norm_num)
  have hsabs : |s.toReal| ≤ |S| + rH d * A + aH d := by
    have := abs_sub_abs_le_abs_sub s.toReal S; linarith
  have hdpabs : |dp.toReal| ≤ |DP| + rH d * AD + aH d := by
    have := abs_sub_abs_le_abs_sub dp.toReal DP; linarith
  have hLR : |L.toReal| ≤ 2 ^ 126 := by
    rw [toReal_def]; have : ((|L.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by exact_mod_cast hL
    push_cast at this; exact this
  have hDlR : |Dl.toReal| ≤ 2 ^ 126 := by
    rw [toReal_def]; have : ((|Dl.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by exact_mod_cast hDl
    push_cast at this; exact this
  -- `dp − Δ` (line 308): finite, within `EY` of `Y`
  obtain ⟨hyf, hye⟩ := addRel_errR hy hdpf hDlf (by
    have : |dp.toReal| ≤ 2 ^ 42 := by linarith
    have : (2 : ℝ) ^ 42 ≤ 2 ^ 100 := by norm_num
    linarith)
  have hyY : |y.toReal - Y| ≤ EY := by
    rw [hY, hEY]; unfold eSub
    have e : y.toReal - (DP - Dl.toReal) = (y.toReal - (dp.toReal - Dl.toReal)) + (dp.toReal - DP) := by ring
    rw [e]
    have t := abs_add_le (y.toReal - (dp.toReal - Dl.toReal)) (dp.toReal - DP)
    have : 3 * u32 * (|dp.toReal| + |Dl.toReal|) ≤ 3 * u32 * (|DP| + rH d * AD + aH d + |Dl.toReal|) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  have hEY0 : 0 ≤ EY := (abs_nonneg _).trans hyY
  have hyabs : |y.toReal| ≤ |Y| + EY := by
    have := abs_sub_abs_le_abs_sub y.toReal Y; linarith
  cases al with
  | false =>
    simp only [Bool.false_eq_true, ite_false] at hp16 hds
    obtain ⟨hz0, hz1⟩ := zero32
    obtain ⟨hcf, hcv⟩ := cvt_zero hz0 hz1
    rw [nanEq_fin hp16 hcf]
    obtain ⟨hmf, hmv⟩ := mul_zero_l hz0 hz1 hyf
    rw [nanEq_fin hds hmf] at hds16
    obtain ⟨hcf', hcv'⟩ := cvt_zero hmf hmv
    rw [nanEq_fin hds16 hcf']
    refine ⟨hcf, hcf', fun _ => ⟨by rw [toReal_def, hcv]; simp, by rw [toReal_def, hcv']; simp⟩,
      fun h => absurd h (by decide)⟩
  | true =>
    simp only [ite_true] at hp16 hds
    obtain ⟨hfit1, hfit2⟩ := hfit rfl
    -- the exponent (line 294): finite, within `EX` of `X`
    obtain ⟨hxf, hxe⟩ := addRel_errR hx hsf hLf (by
      have : |s.toReal| ≤ 2 ^ 42 := by linarith
      have : (2 : ℝ) ^ 42 ≤ 2 ^ 100 := by norm_num
      linarith)
    have hxX : |x.toReal - X| ≤ EX := by
      rw [hX, hEX]; unfold eSub
      have e : x.toReal - (S - L.toReal) = (x.toReal - (s.toReal - L.toReal)) + (s.toReal - S) := by ring
      rw [e]
      have t := abs_add_le (x.toReal - (s.toReal - L.toReal)) (s.toReal - S)
      have : 3 * u32 * (|s.toReal| + |L.toReal|) ≤ 3 * u32 * (|S| + rH d * A + aH d + |L.toReal|) :=
        mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      linarith
    have hEX0 : 0 ≤ EX := (abs_nonneg _).trans hxX
    -- `tl.math.exp2` (line 294), by `Exp2F32`
    obtain ⟨ref, href, hacc⟩ := hexp
    have hr : ref = fun t => (2 : ℝ) ^ t := by
      have href' : (some (fun t => (2 : ℝ) ^ t) : Option (ℝ → ℝ)) = some ref := href
      exact (Option.some.inj href').symm
    subst hr
    have hxup : x.toReal ≤ X + EX := by linarith [(abs_le.mp hxX).2]
    have h2x : (2 : ℝ) ^ x.toReal ≤ 2 ^ (X + EX) := two_rpow_le hxup
    have h2XE : (2 : ℝ) ^ (X + EX) ≤ 2 ^ 15 := by
      have := two_rpow_le hfit1; rwa [two_rpow_15] at this
    have h2xp := two_rpow_pos x.toReal
    have h2Xp := two_rpow_pos X
    have hc22 : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) := by positivity
    have h22x : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ x.toReal ≤ 2 ^ (-22 : ℤ) * 2 ^ (X + EX) :=
      mul_le_mul_of_nonneg_left h2x hc22
    have h22E : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ (X + EX) ≤ 2 ^ (-22 : ℤ) * 2 ^ 15 :=
      mul_le_mul_of_nonneg_left h2XE hc22
    have hc15 : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 15 + 2 ^ (-126 : ℤ) ≤ 1 := by norm_num
    obtain ⟨hp0f, hp0e⟩ := hacc x p0 hp0 hxf (Set.mem_univ _) (Set.mem_univ _) (by
      rw [abs_of_pos h2xp]
      have e : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ x.toReal = 2 ^ x.toReal + 2 ^ (-22 : ℤ) * 2 ^ x.toReal := by ring
      rw [e]
      have : ((Format.binary32.maxVal : ℚ) : ℝ) ≥ 2 ^ 127 := by
        have := maxVal32_ge
        exact_mod_cast this
      have : (2 : ℝ) ^ 15 + 1 ≤ 2 ^ 127 := by norm_num
      linarith)
    simp only at hp0e
    rw [abs_of_pos h2xp] at hp0e
    have hsub := two_rpow_sub_le hxX
    have hmulE : (2 : ℝ) ^ X * 2 ^ EX = 2 ^ (X + EX) := (Real.rpow_add (by norm_num) X EX).symm
    rw [mul_sub, mul_one, hmulE] at hsub
    set pB := (1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (X + EX) + 2 ^ (-126 : ℤ) with hpB
    have hpBe : pB = 2 ^ (X + EX) + 2 ^ (-22 : ℤ) * 2 ^ (X + EX) + 2 ^ (-126 : ℤ) := by rw [hpB]; ring
    have hp0X : |p0.toReal - 2 ^ X| ≤ pB - 2 ^ X := by
      have e : p0.toReal - 2 ^ X = (p0.toReal - 2 ^ x.toReal) + (2 ^ x.toReal - 2 ^ X) := by ring
      rw [e]
      refine (abs_add_le _ _).trans ?_
      rw [hpBe]; linarith
    have hp0abs : |p0.toReal| ≤ pB := by
      have := abs_sub_abs_le_abs_sub p0.toReal (2 ^ x.toReal)
      rw [abs_of_pos h2xp] at this
      rw [hpBe]; linarith
    have hpB15 : pB ≤ 2 ^ 16 - 1 := by
      rw [hpBe]; linarith
    have hpB0 : 0 ≤ pB := (abs_nonneg _).trans hp0abs
    -- `ppT.to(tl.float16)` (line 302)
    obtain ⟨hp16f, hp16e⟩ := cvt16_errR hp0f (by linarith)
    rw [nanEq_fin hp16 hp16f]
    have hp16X : |(FloatV.cvt p0 : F16).toReal - 2 ^ X| ≤ pB - 2 ^ X + u16 * pB + eta16 := by
      have e : (FloatV.cvt p0 : F16).toReal - 2 ^ X = ((FloatV.cvt p0 : F16).toReal - p0.toReal) + (p0.toReal - 2 ^ X) := by ring
      rw [e]
      refine (abs_add_le _ _).trans ?_
      have : u16 * |p0.toReal| ≤ u16 * pB := mul_le_mul_of_nonneg_left hp0abs hu16n
      linarith
    have hp16abs : |(FloatV.cvt p0 : F16).toReal| ≤ (1 + u16) * pB + eta16 := by
      have := abs_sub_abs_le_abs_sub (FloatV.cvt p0 : F16).toReal p0.toReal
      have : u16 * |p0.toReal| ≤ u16 * pB := mul_le_mul_of_nonneg_left hp0abs hu16n
      linarith
    -- `pT * (dpT − Di)` (line 308)
    have hYB0 : 0 ≤ |Y| + EY := by positivity
    have hprod : |p0.toReal * y.toReal| ≤ pB * (|Y| + EY) := by
      rw [abs_mul]; exact mul_le_mul hp0abs hyabs (abs_nonneg _) hpB0
    obtain ⟨hdsf, hdse⟩ := mul_errR hp0f hyf (hprod.trans (hfit2.trans (by norm_num)))
    rw [nanEq_fin hds hdsf] at hds16
    set ds' := FloatV.mul p0 y with hds'
    have hdsX : |ds'.toReal - 2 ^ X * Y| ≤
        (pB - 2 ^ X) * (|Y| + EY) + 2 ^ X * EY + u32 * (pB * (|Y| + EY)) + eta32 := by
      have e : ds'.toReal - 2 ^ X * Y = (ds'.toReal - p0.toReal * y.toReal) +
          ((p0.toReal - 2 ^ X) * y.toReal + 2 ^ X * (y.toReal - Y)) := by ring
      rw [e]
      refine (abs_add_le _ _).trans ?_
      have h1 : |(p0.toReal - 2 ^ X) * y.toReal + 2 ^ X * (y.toReal - Y)| ≤
          (pB - 2 ^ X) * (|Y| + EY) + 2 ^ X * EY := by
        refine (abs_add_le _ _).trans ?_
        rw [abs_mul, abs_mul, abs_of_pos h2Xp]
        have := mul_le_mul hp0X hyabs (abs_nonneg _) (by linarith [abs_nonneg (p0.toReal - 2 ^ X)])
        have := mul_le_mul_of_nonneg_left hyY h2Xp.le
        linarith
      have h2 : u32 * |p0.toReal * y.toReal| ≤ u32 * (pB * (|Y| + EY)) :=
        mul_le_mul_of_nonneg_left hprod hu32n
      linarith
    have hdsabs : |ds'.toReal| ≤ (1 + u32) * (pB * (|Y| + EY)) + eta32 := by
      have := abs_sub_abs_le_abs_sub ds'.toReal (p0.toReal * y.toReal)
      have h2 : u32 * |p0.toReal * y.toReal| ≤ u32 * (pB * (|Y| + EY)) :=
        mul_le_mul_of_nonneg_left hprod hu32n
      linarith
    -- `dsT.to(tl.float16)` (line 309)
    have hds65 : |ds'.toReal| ≤ 65504 := by
      refine hdsabs.trans ?_
      rw [hu32, heta32]
      have : (1 + (2 : ℝ) ^ (-24 : ℤ)) * (pB * (|Y| + EY)) ≤ (1 + 2 ^ (-24 : ℤ)) * 2 ^ 15 :=
        mul_le_mul_of_nonneg_left hfit2 (by positivity)
      have : (1 + (2 : ℝ) ^ (-24 : ℤ)) * 2 ^ 15 + 2 ^ (-150 : ℤ) ≤ 65504 := by norm_num
      linarith
    obtain ⟨hds16f, hds16e⟩ := cvt16_errR hdsf hds65
    rw [nanEq_fin hds16 hds16f]
    refine ⟨hp16f, hds16f, fun h => absurd h (by decide), fun _ => ⟨hp16X, hp16abs, ?_, ?_⟩⟩
    · have e : (FloatV.cvt ds' : F16).toReal - 2 ^ X * Y =
          ((FloatV.cvt ds' : F16).toReal - ds'.toReal) + (ds'.toReal - 2 ^ X * Y) := by ring
      rw [e]
      refine (abs_add_le _ _).trans ?_
      have : u16 * |ds'.toReal| ≤ u16 * ((1 + u32) * (pB * (|Y| + EY)) + eta32) :=
        mul_le_mul_of_nonneg_left hdsabs hu16n
      linarith
    · have := abs_sub_abs_le_abs_sub (FloatV.cvt ds' : F16).toReal ds'.toReal
      have : u16 * |ds'.toReal| ≤ u16 * ((1 + u32) * (pB * (|Y| + EY)) + eta32) :=
        mul_le_mul_of_nonneg_left hdsabs hu16n
      linarith

end Kernels.AttnBwd
