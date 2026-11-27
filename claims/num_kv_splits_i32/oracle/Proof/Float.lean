import Proof.Consts

/-!
# The `float32` lines: `ext_device_core_count`

Source lines 85-88 compute `trunc(fl(C · max(log₂(div(L, 64)), 1)))` in `float32`. With the
device facts for `/` (`DivFull`) and `tl.math.log2` (`Log2F32`), the product before truncation is
finite, at least `C`, and within `C·2^-17` of `v = C · max(log₂(L / 64), 1)` (`core_chain`), so
its truncation satisfies `CoreOk` (`core_trunc`):

* `L` and `C` convert exactly (`itof_exact`), `64.0` and `1.0` are exact (`c64_val`, `c1_val`);
* `|div(L, 64) − L/64| ≤ (2^-22 + 2^-142)·L/64`, so `log₂` moves by at most `3·(2^-22 + 2^-142)`
  (`logb_close`);
* `Log2F32` adds `2^-22·17 + 2^-149`; in all `|lg − log₂(L/64)| ≤ 21·2^-22`, and `tl.maximum`
  with `1` keeps the bound;
* the product rounds once: `|p − C·m| ≤ 17·C·2^-24 + 2^-150`, and it is at least `C` because `C`
  is a `float32` value below the product (`mul_ge_nat`).
-/

namespace Claims.NumKvSplits

open TVA

/-! ## Rounding keeps a lower bound that is a `float32` value -/

/-- **A product at least a natural `C` rounds to at least `C`** when it is below `2^24` (so `C` is
a multiple of the quantum). -/
theorem mul_ge_nat {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true) {C : ℕ}
    (hC : (C : ℚ) ≤ x.toRat * y.toRat) (hC1 : 1 ≤ C) (hlt : x.toRat * y.toRat < 2 ^ 24) :
    (C : ℚ) ≤ (FloatV.mul x y).toRat := by
  set q := x.toRat * y.toRat with hq
  have hq0 : 0 < q := lt_of_lt_of_le (by exact_mod_cast hC1) hC
  have hqm : |q| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hq0]; exact hlt.le.trans two24_le_maxVal32
  obtain ⟨hb, -, hv⟩ := Format.roundRat_spec (f := Format.binary32) Format.binary32_regular .rn
    false (x.sign != y.sign) hq0.ne' hqm
  unfold FloatV.mul
  rw [Format.mul_of_finite .rn hx hy]
  change (C : ℚ) ≤ (FloatV.ofPat (Format.binary32.roundRat .rn false (x.sign != y.sign) q)).toRat
  rw [FloatV.toRat_ofPat hb, hv, ite_eq_right (not_lt.mpr hq0.le), one_mul, abs_of_pos hq0]
  set Q := Format.binary32.qexp q with hQ
  have hlog : Int.log 2 q ≤ 23 := by
    have : q < (2 : ℚ) ^ (24 : ℤ) := by
      have : (2 : ℚ) ^ (24 : ℤ) = 2 ^ 24 := by norm_num
      rw [this]; exact hlt
    have := Format.log_lt_of_lt hq0 this
    omega
  have hQ0 : Q ≤ 0 := by
    rw [hQ, Format.qexp_eq hq0]
    have : Format.binary32.emin = -126 := by decide
    have hm : (Format.binary32.m : ℤ) = 23 := rfl
    rw [this, hm]; omega
  have h2 : (2 : ℚ) ^ (-Q) * (2 : ℚ) ^ Q = 1 := by
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]; simp
  have hK : ((C * 2 ^ (-Q).toNat : ℕ) : ℚ) = (C : ℚ) * (2 : ℚ) ^ (-Q) := by
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
  have hle : ((C * 2 ^ (-Q).toNat : ℕ) : ℚ) ≤ q * (2 : ℚ) ^ (-Q) := by
    rw [hK]; exact mul_le_mul_of_nonneg_right hC (by positivity)
  have hr := Format.le_rnat RoundMode.rn hle
  rw [Format.roundSig_eq]
  have hr' : ((C * 2 ^ (-Q).toNat : ℕ) : ℚ) ≤ (Format.rnat RoundMode.rn (q * (2 : ℚ) ^ (-Q)) : ℚ) := by
    exact_mod_cast hr
  calc (C : ℚ) = (C : ℚ) * (2 : ℚ) ^ (-Q) * (2 : ℚ) ^ Q := by rw [mul_assoc, h2, mul_one]
    _ = ((C * 2 ^ (-Q).toNat : ℕ) : ℚ) * (2 : ℚ) ^ Q := by rw [hK]
    _ ≤ _ := mul_le_mul_of_nonneg_right hr' (by positivity)

/-! ## `log₂` of a relatively close value -/

/-- `log 2 > 0.69`. -/
theorem log_two_gt : (0.69 : ℝ) < Real.log 2 := lt_trans (by norm_num) Real.log_two_gt_d9

/-- **`log₂` of a close value**: `|q − a| ≤ ε·a` with `ε ≤ 1/2` moves `log₂` by at most `3ε`. -/
theorem logb_close {a q ε : ℝ} (ha : 0 < a) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2)
    (hq : |q - a| ≤ ε * a) : 0 < q ∧ |Real.logb 2 q - Real.logb 2 a| ≤ 3 * ε := by
  have hqa := abs_le.mp hq
  have hq0 : 0 < q := by nlinarith
  refine ⟨hq0, ?_⟩
  set x := q / a with hx
  have hx0 : 0 < x := div_pos hq0 ha
  have hx1 : |x - 1| ≤ ε := by
    rw [hx, show q / a - 1 = (q - a) / a by field_simp, abs_div, abs_of_pos ha, div_le_iff₀ ha]
    exact hq
  have hx1' := abs_le.mp hx1
  have hup : Real.log x ≤ ε := (Real.log_le_sub_one_of_pos hx0).trans (by linarith)
  have hlo : -(2 * ε) ≤ Real.log x := by
    have h := Real.one_sub_inv_le_log_of_pos hx0
    have hxge : 1 / 2 ≤ x := by linarith
    have : 1 - x⁻¹ = (x - 1) / x := by field_simp
    rw [this] at h
    have : -(2 * ε) ≤ (x - 1) / x := by
      rw [le_div_iff₀ hx0]; nlinarith
    linarith
  have hlog : |Real.log x| ≤ 2 * ε := abs_le.mpr ⟨hlo, by linarith⟩
  have heq : Real.logb 2 q - Real.logb 2 a = Real.log x / Real.log 2 := by
    rw [hx, Real.log_div hq0.ne' ha.ne', Real.logb, Real.logb, sub_div]
  have hl2 := log_two_gt
  rw [heq, abs_div, abs_of_pos (by linarith : (0 : ℝ) < Real.log 2), div_le_iff₀ (by linarith)]
  nlinarith

/-- `log₂` of a value in `[2^-6, 2^16]` is in `[-6, 16]`. -/
theorem logb_bounds {a : ℝ} (ha : (2 : ℝ) ^ (-6 : ℤ) ≤ a) (ha' : a ≤ 2 ^ (16 : ℤ)) :
    -6 ≤ Real.logb 2 a ∧ Real.logb 2 a ≤ 16 := by
  have ha0 : 0 < a := lt_of_lt_of_le (by positivity) ha
  constructor
  · rw [Real.le_logb_iff_rpow_le (by norm_num) ha0]
    rw [show (-6 : ℝ) = ((-6 : ℤ) : ℝ) by norm_num, Real.rpow_intCast]; exact ha
  · rw [Real.logb_le_iff_le_rpow (by norm_num) ha0]
    rw [show (16 : ℝ) = ((16 : ℤ) : ℝ) by norm_num, Real.rpow_intCast]; exact ha'

/-! ## The chain, in pieces -/

/-- The quotient's bound in `ℝ`, relative: `2^-148 ≤ 2^-142 · L/64` for `L ≥ 1`. -/
theorem ext_rel {e : ℚ} {L : ℕ} (hL1 : 1 ≤ L)
    (h : |e - (L : ℚ) / 64| ≤ 2 ^ (-22 : ℤ) * |(L : ℚ) / 64| + 2 ^ (-148 : ℤ)) :
    |(e : ℝ) - (L : ℝ) / 64| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-142 : ℤ)) * ((L : ℝ) / 64) := by
  have hL : (1 : ℚ) ≤ L := by exact_mod_cast hL1
  rw [abs_of_nonneg (show (0 : ℚ) ≤ (L : ℚ) / 64 by positivity)] at h
  have hs : (2 : ℚ) ^ (-148 : ℤ) ≤ 2 ^ (-142 : ℤ) * ((L : ℚ) / 64) := by
    have e1 : (2 : ℚ) ^ (-148 : ℤ) = 2 ^ (-142 : ℤ) * (1 / 64) := by norm_num [zpow_neg]
    rw [e1]; exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have hq : |e - (L : ℚ) / 64| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-142 : ℤ)) * ((L : ℚ) / 64) := by
    rw [add_mul]; linarith
  have := (Rat.cast_le (K := ℝ)).mpr hq
  push_cast at this
  exact this

/-- The quotient is positive, its `log₂` is at most `17` in magnitude, and a `log₂` within
`2^-22·|log₂ e| + 2^-149` of it is within `21·2^-22` of `log₂(L/64)`, which is at most `16`. -/
theorem lg_close {a e : ℝ} (ha1 : 1 / 64 ≤ a) (ha2 : a ≤ 2 ^ 16)
    (he : |e - a| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-142 : ℤ)) * a) :
    0 < e ∧ |Real.logb 2 e| ≤ 17 ∧ Real.logb 2 a ≤ 16 ∧ ∀ l : ℝ,
      |l - Real.logb 2 e| ≤ 2 ^ (-22 : ℤ) * |Real.logb 2 e| + 2 ^ (-149 : ℤ) →
      |l - Real.logb 2 a| ≤ 21 * 2 ^ (-22 : ℤ) := by
  set ε : ℝ := 2 ^ (-22 : ℤ) + 2 ^ (-142 : ℤ) with hε
  have hε0 : 0 ≤ ε := by positivity
  have hεs : ε ≤ 1 / 2 := by rw [hε]; norm_num [zpow_neg]
  obtain ⟨he0, hd⟩ := logb_close (by linarith) hε0 hεs he
  obtain ⟨hb1, hb2⟩ := logb_bounds (a := a) (le_trans (by norm_num [zpow_neg]) ha1)
    (le_trans ha2 (by norm_num))
  have h3 : 3 * ε ≤ 1 := by rw [hε]; norm_num [zpow_neg]
  have hle : |Real.logb 2 e| ≤ 17 := by
    have := abs_le.mp hd
    rw [abs_le]; constructor <;> linarith
  refine ⟨he0, hle, hb2, fun l h1 => ?_⟩
  have h2 : (2 : ℝ) ^ (-22 : ℤ) * |Real.logb 2 e| ≤ 2 ^ (-22 : ℤ) * 17 :=
    mul_le_mul_of_nonneg_left hle (by positivity)
  have htri := abs_sub_le l (Real.logb 2 e) (Real.logb 2 a)
  have h4 : 3 * ε + (2 ^ (-22 : ℤ) * 17 + 2 ^ (-149 : ℤ)) ≤ 21 * (2 : ℝ) ^ (-22 : ℤ) := by
    rw [hε]; norm_num [zpow_neg]
  linarith

/-- The product's bound: from `|m − M| ≤ 21·2^-22` and `M ≤ 16` and one rounding of `C·m`,
`p` is within `C·2^-17` of `C·M`. -/
theorem prod_close {C m M p : ℝ} (hC : 1 ≤ C) (hM : M ≤ 16)
    (hmM : |m - M| ≤ 21 * 2 ^ (-22 : ℤ))
    (hp : |p - C * m| ≤ C * m * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) :
    |p - C * M| ≤ C * 2 ^ (-17 : ℤ) := by
  have hm17 : m ≤ 17 := by
    have := (abs_le.mp hmM).2
    have : (21 : ℝ) * 2 ^ (-22 : ℤ) ≤ 1 := by norm_num [zpow_neg]
    linarith
  have hcm : |C * m - C * M| ≤ C * (21 * 2 ^ (-22 : ℤ)) := by
    rw [← mul_sub, abs_mul, abs_of_nonneg (by linarith)]
    exact mul_le_mul_of_nonneg_left hmM (by linarith)
  have hmc : C * m * 2 ^ (-24 : ℤ) ≤ C * 17 * 2 ^ (-24 : ℤ) := by
    have : C * m ≤ C * 17 := mul_le_mul_of_nonneg_left hm17 (by linarith)
    exact mul_le_mul_of_nonneg_right this (by positivity)
  have htri := abs_sub_le p (C * m) (C * M)
  have h1 : (2 : ℝ) ^ (-150 : ℤ) ≤ 27 * 2 ^ (-24 : ℤ) := by norm_num [zpow_neg]
  have h2 : (27 : ℝ) * 2 ^ (-24 : ℤ) ≤ C * (27 * 2 ^ (-24 : ℤ)) := by
    have : (0 : ℝ) ≤ 27 * 2 ^ (-24 : ℤ) := by positivity
    nlinarith
  have e : C * 2 ^ (-17 : ℤ) =
      C * 17 * 2 ^ (-24 : ℤ) + C * (21 * 2 ^ (-22 : ℤ)) + C * (27 * 2 ^ (-24 : ℤ)) := by
    norm_num [zpow_neg]; ring
  linarith

/-- **The `float32` value before truncation** (source lines 85-87): for `1 ≤ L ≤ 2^22` and
`1 ≤ C ≤ 2^16`, every result the device facts allow is finite, at least `C`, and within `C·2^-17`
of `C · max(log₂(L / 64), 1)`. -/
theorem core_chain {P : Platform} (hP : Device P) {L C : ℕ} (hL1 : 1 ≤ L) (hL : L ≤ 2 ^ 22)
    (hC1 : 1 ≤ C) (hC : C ≤ 2 ^ 16) {msl ext lg mx dcc prod : F32}
    (hmsl : tl.ItofRel .binary32 (L : ℤ) msl) (hext : P.div .binary32 msl c64 ext)
    (hlg : P.math .log2 .binary32 ext lg) (hmx : FloatV.MaxRel true lg c1 mx)
    (hdcc : tl.ItofRel .binary32 (C : ℤ) dcc) (hprod : FloatV.NaNEq (FloatV.mul dcc mx) prod) :
    prod.isFinite = true ∧ (C : ℝ) ≤ prod.toReal ∧
      |prod.toReal - (C : ℝ) * max (Real.logb 2 ((L : ℝ) / 64)) 1| ≤ C * 2 ^ (-17 : ℤ) := by
  have hreg := Format.binary32_regular
  have hmax32 : ((Format.binary32.maxVal : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 :=
    Format.binary32_maxVal_real
  -- the exact conversions and constants
  obtain ⟨hmslf, hmslv⟩ := itof_exact (z := (L : ℤ)) (by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast hL.trans (by norm_num)) hmsl
  obtain ⟨hdccf, hdccv⟩ := itof_exact (z := (C : ℤ)) (by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast hC.trans (by norm_num)) hdcc
  obtain ⟨h64f, h64v⟩ := c64_val
  obtain ⟨h1f, h1v⟩ := c1_val
  push_cast at hmslv hdccv
  have hLr1 : (1 : ℝ) ≤ L := by exact_mod_cast hL1
  have hLr : (L : ℝ) ≤ 2 ^ 22 := by exact_mod_cast hL
  have hCr1 : (1 : ℝ) ≤ C := by exact_mod_cast hC1
  have hCr : (C : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hC
  -- the division
  obtain ⟨hextf, hexterr⟩ := hP.divFull msl c64 ext hext hmslf h64f (by rw [h64v]; norm_num) (by
    rw [hmslv, h64v, abs_of_nonneg (by positivity)]
    have : (L : ℚ) ≤ 2 ^ 22 := by exact_mod_cast hL
    linarith)
  rw [hmslv, h64v] at hexterr
  have hext_r := ext_rel hL1 hexterr
  rw [← FloatV.toReal_eq] at hext_r
  -- `tl.math.log2`
  obtain ⟨ref, href, hacc⟩ := hP.log2F32
  simp only [MathFn.ref, Option.some.injEq] at href
  subst href
  obtain ⟨he0, hle, hlb2, hfin⟩ := lg_close (a := (L : ℝ) / 64) (by linarith) (by linarith)
    hext_r
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.logb 2 ext.toReal| + 2 ^ (-149 : ℤ) ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [hmax32]
    have h1 : (1 + (2 : ℝ) ^ (-22 : ℤ)) ≤ 2 := by norm_num [zpow_neg]
    have h2 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.logb 2 ext.toReal| ≤ 2 * 17 :=
      mul_le_mul h1 hle (abs_nonneg _) (by norm_num)
    have h3 : (2 : ℝ) ^ (-149 : ℤ) ≤ 1 := by norm_num [zpow_neg]
    have h4 : (2 * 17 + 1 : ℝ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hlgf, hlgerr⟩ := hacc ext lg hlg hextf (Set.mem_univ _) he0 hroom
  have hD := hfin _ hlgerr
  -- `tl.maximum(…, 1.0)`
  have hlgn : lg.isNaN = false := (FloatV.isFinite_iff.mp hlgf).1
  have h1n : c1.isNaN = false := (FloatV.isFinite_iff.mp h1f).1
  have hc1r : c1.toReal = 1 := by rw [FloatV.toReal_eq, h1v]; norm_num
  have hmxr : mx.toReal = max lg.toReal 1 := by rw [hmx.toReal_eq_max hlgn h1n, hc1r]
  have hmxf : mx.isFinite = true := by
    rcases hmx.mem hlgn h1n with h | h <;> rw [h] <;> assumption
  have hmD : |mx.toReal - max (Real.logb 2 ((L : ℝ) / 64)) 1| ≤ 21 * 2 ^ (-22 : ℤ) := by
    rw [hmxr]; exact abs_max_sub_max_le_of hD (by norm_num [zpow_neg])
  have hm1 : 1 ≤ mx.toReal := by rw [hmxr]; exact le_max_right _ _
  have hm17 : mx.toReal ≤ 17 := by
    have := (abs_le.mp hmD).2
    have : max (Real.logb 2 ((L : ℝ) / 64)) 1 ≤ 16 := max_le hlb2 (by norm_num)
    have : (21 : ℝ) * 2 ^ (-22 : ℤ) ≤ 1 := by norm_num [zpow_neg]
    linarith
  -- the product
  have hdccr : dcc.toReal = C := by rw [FloatV.toReal_eq, hdccv]; push_cast; rfl
  have hCm0 : (0 : ℝ) ≤ C * mx.toReal := by positivity
  have hCm : (C : ℝ) * mx.toReal ≤ 2 ^ 16 * 17 :=
    mul_le_mul hCr hm17 (by linarith) (by norm_num)
  have hmulroom : |dcc.toReal * mx.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [hmax32, hdccr, abs_of_nonneg hCm0]
    have : (2 ^ 16 * 17 : ℝ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hmulf, hmulerr⟩ := FloatV.mul_err_real hreg hdccf hmxf hmulroom
  obtain ⟨hpf, -, hpr⟩ := hprod.exact_of_isFinite hmulf
  rw [Format.binary32_u_real, Format.binary32_eta_real, hdccr, abs_of_nonneg hCm0] at hmulerr
  refine ⟨hpf, ?_, ?_⟩
  · -- at least `C`
    have hq : (C : ℚ) ≤ dcc.toRat * mx.toRat := by
      have : ((C : ℚ) : ℝ) ≤ ((dcc.toRat * mx.toRat : ℚ) : ℝ) := by
        push_cast
        rw [← FloatV.toReal_eq, ← FloatV.toReal_eq, hdccr]; nlinarith
      exact_mod_cast this
    have hlt : dcc.toRat * mx.toRat < 2 ^ 24 := by
      have : ((dcc.toRat * mx.toRat : ℚ) : ℝ) < ((2 ^ 24 : ℚ) : ℝ) := by
        push_cast
        rw [← FloatV.toReal_eq, ← FloatV.toReal_eq, hdccr]; linarith
      exact_mod_cast this
    have hge := mul_ge_nat hdccf hmxf hq hC1 hlt
    rw [hpr, FloatV.toReal_eq]; exact_mod_cast hge
  · -- within `C·2^-17`
    rw [hpr]
    exact prod_close hCr1 (max_le hlb2 (by norm_num)) hmD hmulerr

/-- **The truncation** (source line 86's `tl.cast(…, tl.int32)`): a finite `float32` value `p` at
least `1` and below `2^31` truncates to `⌊p⌋`, and some result exists. -/
theorem trunc_spec {p : F32} (hpf : p.isFinite = true) (hp1 : (1 : ℝ) ≤ p.toReal)
    (hp2 : p.toReal < 2 ^ 31) :
    (∃ r : I32, FtoiRel true 32 p r) ∧ ∀ r : I32, FtoiRel true 32 p r →
      (r.toInt : ℝ) ≤ p.toReal ∧ p.toReal < r.toInt + 1 := by
  have hq1 : (1 : ℚ) ≤ p.toRat := by rw [FloatV.toReal_eq] at hp1; exact_mod_cast hp1
  have hq2 : p.toRat < 2 ^ 31 := by rw [FloatV.toReal_eq] at hp2; exact_mod_cast hp2
  have htd : p.toRat.num.tdiv p.toRat.den = ⌊p.toRat⌋ := by
    rw [Rat.floor_def', Int.tdiv_eq_ediv_of_nonneg (Rat.num_nonneg.mpr (by linarith))]
  have hfl1 : (1 : ℤ) ≤ ⌊p.toRat⌋ := Int.le_floor.mpr (by exact_mod_cast hq1)
  have hfl2 : ⌊p.toRat⌋ < 2 ^ 31 := Int.floor_lt.mpr (by exact_mod_cast hq2)
  have hstd : Format.binary32.Std := by decide
  have hmin : IntV.minInt true 32 ≤ p.toRat.num.tdiv p.toRat.den := by
    rw [htd]; simp [IntV.minInt]; omega
  have hmax : p.toRat.num.tdiv p.toRat.den ≤ IntV.maxInt true 32 := by
    rw [htd]; simp [IntV.maxInt]; omega
  refine ⟨⟨_, hstd, hpf, hmin, hmax, rfl⟩, ?_⟩
  rintro r ⟨-, -, -, -, hr⟩
  have hv : r.toInt = ⌊p.toRat⌋ := by
    rw [hr, htd]
    exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
  rw [hv, FloatV.toReal_eq]
  constructor
  · exact_mod_cast Int.floor_le p.toRat
  · exact_mod_cast Int.lt_floor_add_one p.toRat

end Claims.NumKvSplits
