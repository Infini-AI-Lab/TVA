import Proof.Numeric

/-!
# `attn_bwd_f16`: the `float16` `P` and `dS` of one pair

For a key row `n` and a query row `m`: from the two dot results and the platform's `exp2`, the
computed `float16` `P[n, m]` and `dS[n, m]` are within the claim's bounds (`pair_good`).
-/

namespace Claims.AttnBwd

open TVA

private theorem binary32_specials : Format.binary32.specials ≠ .finite := by decide

/-- **`exp2` sandwich.** If `x` is between `w/A` and `w·A` (`A ≥ 1`) and `pt` is within `ε·x + δ` of
`x`, then `|pt| ≤ (1 + ε)·w·A + δ` and `|pt − w| ≤ (1 + ε)·w·A + δ − w`. -/
theorem exp2_sandwich {w A ε δ x pt : ℝ} (hw : 0 < w) (hA : 1 ≤ A) (hε : 0 ≤ ε) (hε1 : ε ≤ 1)
    (hx1 : x ≤ w * A) (hx2 : w ≤ A * x) (hx0 : 0 ≤ x) (hpt : |pt - x| ≤ ε * x + δ) :
    |pt| ≤ (1 + ε) * w * A + δ ∧ |pt - w| ≤ (1 + ε) * w * A + δ - w := by
  have hwA : 0 ≤ w * A := by positivity
  obtain ⟨hp1, hp2⟩ := abs_le.mp hpt
  have hup : pt ≤ (1 + ε) * (w * A) + δ := by nlinarith
  have hlow : pt ≥ (1 - ε) * x - δ := by nlinarith
  have hpos : (1 + ε) * (w * A) + δ - w ≥ 0 := by nlinarith
  refine ⟨abs_le.mpr ⟨?_, ?_⟩, abs_le.mpr ⟨?_, ?_⟩⟩
  · nlinarith [mul_nonneg hε hx0]
  · nlinarith
  · -- w - pt ≤ (1 + ε) w A + δ - w
    have h1 : (1 - ε) * w ≤ (1 - ε) * (A * x) := mul_le_mul_of_nonneg_left hx2 (by linarith)
    have h2 : 0 ≤ (A - 1) * ((1 + ε) * A - (1 - ε)) := by
      apply mul_nonneg <;> nlinarith
    have h3 : A * (2 * w - (1 - ε) * x) ≤ (1 + ε) * w * A * A := by nlinarith
    have hApos : 0 < A := by linarith
    have h4 : 2 * w - (1 - ε) * x ≤ (1 + ε) * w * A := by
      by_contra hcon
      have hcon := not_le.mp hcon
      have := mul_lt_mul_of_pos_left hcon hApos
      nlinarith
    nlinarith
  · nlinarith

/-- The relative constant of the dot over `HEAD_DIM ≤ 256` products is at most `1`. -/
theorem rH_le {d : Inst} (hd : d.ok) : rH d ≤ 1 := by
  have h := hd.hd
  have hK : (d.HEAD_DIM : ℚ) ≤ 256 := by exact_mod_cast (by omega : d.HEAD_DIM ≤ 256)
  unfold rH Device.relDot16
  have : (2 : ℚ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  rw [this]
  have : (((d.HEAD_DIM : ℚ) + 2) * (1 / 4194304) : ℚ) ≤ 1 := by linarith
  exact_mod_cast this

/-- The absolute constant of the dot over `HEAD_DIM ≤ 256` products is at most `1`. -/
theorem aH_le {d : Inst} (hd : d.ok) : aH d ≤ 1 := by
  have h := hd.hd
  have hK : (d.HEAD_DIM : ℚ) ≤ 256 := by exact_mod_cast (by omega : d.HEAD_DIM ≤ 256)
  unfold aH Device.absDot16
  have h2 : (2 : ℚ) ^ (-126 : ℤ) ≤ 1 / 4194304 := by norm_num
  have : (((d.HEAD_DIM : ℚ) + 2) * 2 ^ (-126 : ℤ) : ℚ) ≤ 1 :=
    calc ((d.HEAD_DIM : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 258 * (1 / 4194304) :=
          mul_le_mul (by linarith) h2 (by positivity) (by norm_num)
      _ ≤ 1 := by norm_num
  exact_mod_cast this

/-- **A dot result minus a finite `L`** (`tl.subc`): finite, and within `eSub` of `s - L`. -/
theorem sub_chain {d : Inst} (hd : d.ok) {fusion : Bool} {d1 L r : F32} {s a : ℝ} (ha0 : 0 ≤ a)
    (ha : a ≤ 2 ^ 42) (hsa : |s| ≤ a)
    (h1 : d1.isFinite = true ∧ |d1.toReal - s| ≤ rH d * a + aH d)
    (hL : L.isFinite = true ∧ |L.toRat| ≤ 2 ^ 126) (hr : AddRel fusion d1 (FloatV.neg L) r) :
    r.isFinite = true ∧ |r.toReal - (s - L.toReal)| ≤ eSub d s a L.toReal := by
  have hrH := rH_le hd
  have haH := aH_le hd
  have hrH0 := rH_nonneg d
  have haH0 := aH_nonneg d
  have hB0 : rH d * a + aH d ≤ 2 ^ 42 + 1 := by
    have : rH d * a ≤ 1 * 2 ^ 42 := mul_le_mul hrH ha ha0 (by norm_num)
    linarith
  have hd1 : |d1.toReal| ≤ 2 ^ 44 := by
    have := abs_sub_abs_le_abs_sub d1.toReal s
    have h2 : (2 : ℝ) ^ 42 + (2 ^ 42 + 1) ≤ 2 ^ 44 := by norm_num
    linarith [h1.2]
  have hd1q : |d1.toRat| ≤ 2 ^ 44 := by
    have : ((|d1.toRat| : ℚ) : ℝ) ≤ ((2 ^ 44 : ℚ) : ℝ) := by
      rw [Rat.cast_abs]; push_cast; exact hd1
    exact_mod_cast this
  obtain ⟨hrfin, herr⟩ := FloatV.AddRel.err Format.binary32_regular binary32_specials
    (by rw [Format.binary32_u]; norm_num) hr h1.1 (by rw [FloatV.isFinite_neg]; exact hL.1)
    (by
      rw [FloatV.toRat_neg, abs_neg, Format.binary32_maxVal, Format.binary32_eta]
      have h5 : (2 : ℚ) ^ 126 + 2 ^ 44 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * 2 ^ (-150 : ℤ) := by norm_num
      linarith [hL.2])
  refine ⟨hrfin, ?_⟩
  have herr' := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at herr'
  rw [FloatV.toRat_neg] at herr'
  push_cast at herr'
  simp only [abs_neg] at herr'
  have herr2 : |r.toReal - (d1.toReal + -L.toReal)| ≤
      3 * ((Format.binary32.u : ℚ) : ℝ) * (|d1.toReal| + |L.toReal|) + 4 * ((Format.binary32.eta : ℚ) : ℝ) := herr'
  have e1 : r.toReal - (s - L.toReal) = (r.toReal - (d1.toReal + -L.toReal)) + (d1.toReal - s) := by
    ring
  have hu : ((Format.binary32.u : ℚ) : ℝ) = u32 := rfl
  have he : ((Format.binary32.eta : ℚ) : ℝ) = eta32 := rfl
  rw [hu, he] at herr2
  have hd1s : |d1.toReal| ≤ |s| + (rH d * a + aH d) := by
    have := abs_sub_abs_le_abs_sub d1.toReal s
    linarith [h1.2]
  have hu0 := u32_nonneg
  have heta0 := eta32_nonneg
  unfold eSub
  have e2 : |r.toReal - (s - L.toReal)| ≤
      |r.toReal - (d1.toReal + -L.toReal)| + |d1.toReal - s| := by
    rw [e1]; exact abs_add_le _ _
  have h3 : 3 * u32 * (|d1.toReal| + |L.toReal|) ≤
      3 * u32 * (|s| + rH d * a + aH d + |L.toReal|) := by
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    linarith
  linarith [h1.2, herr2]

/-- **`tl.math.exp2` of a computed exponent**: for `r1` within `e` of the exact exponent `x`
(`x + e ≤ 15`), every platform result `pt` is finite, of magnitude at most
`(1 + 2^-22)·2^(x+e) + 2^-126`, and within that minus `2^x` of `2^x`. -/
theorem exp2_step {P : Platform} (hexp : Device.Exp2F32 P) {r1 pt : F32} {x e : ℝ} (he0 : 0 ≤ e)
    (hx : x + e ≤ 15) (hr : r1.isFinite = true) (hrx : |r1.toReal - x| ≤ e)
    (hm : P.math .exp2 .binary32 r1 pt) :
    pt.isFinite = true ∧ |pt.toReal| ≤ (1 + 2 ^ (-22 : ℤ)) * 2 ^ (x + e) + 2 ^ (-126 : ℤ) ∧
      |pt.toReal - 2 ^ x| ≤ (1 + 2 ^ (-22 : ℤ)) * 2 ^ (x + e) + 2 ^ (-126 : ℤ) - 2 ^ x := by
  obtain ⟨ref, href, hacc⟩ := hexp
  have href' : ref = fun y : ℝ => (2 : ℝ) ^ y := by
    simp only [MathFn.ref, Option.some.injEq] at href
    exact href.symm
  subst href'
  obtain ⟨hr1, hr2⟩ := abs_le.mp hrx
  have h2pos : (0 : ℝ) < 2 := by norm_num
  have hxr : (2 : ℝ) ^ r1.toReal ≤ 2 ^ (x + e) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have hxr2 : (2 : ℝ) ^ x ≤ 2 ^ (e + r1.toReal) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have hxe : (2 : ℝ) ^ (x + e) = 2 ^ x * 2 ^ e := Real.rpow_add h2pos _ _
  have hxe2 : (2 : ℝ) ^ (e + r1.toReal) = 2 ^ e * 2 ^ r1.toReal := Real.rpow_add h2pos _ _
  have h15 : (2 : ℝ) ^ (x + e) ≤ 2 ^ (15 : ℝ) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) hx
  have h15' : (2 : ℝ) ^ (15 : ℝ) = 2 ^ 15 := by norm_num
  have hpos : 0 < (2 : ℝ) ^ r1.toReal := Real.rpow_pos_of_pos h2pos _
  have hroom : (1 + 2 ^ (-22 : ℤ)) * |(2 : ℝ) ^ r1.toReal| + 2 ^ (-126 : ℤ) ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hpos, Format.binary32_maxVal]
    push_cast
    have h3 : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 / 65536 := by norm_num
    have h4 : (2 : ℝ) ^ (-126 : ℤ) ≤ 1 := by norm_num
    have h5 : (2 : ℝ) ^ r1.toReal ≤ 2 ^ 15 := by rw [← h15']; exact hxr.trans h15
    have h6 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ r1.toReal ≤ (1 + 1 / 65536) * 2 ^ 15 :=
      mul_le_mul (by linarith) h5 hpos.le (by norm_num)
    have : (1 + 1 / 65536 : ℝ) * 2 ^ 15 + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    exact le_trans (add_le_add h6 h4) this
  obtain ⟨hfin, herr⟩ := hacc r1 pt hm hr (Set.mem_univ _) (by simp [MathFn.natDom]) hroom
  rw [abs_of_pos hpos] at herr
  have hw : (0 : ℝ) < 2 ^ x := Real.rpow_pos_of_pos h2pos _
  have hA : (1 : ℝ) ≤ 2 ^ e := Real.one_le_rpow (by norm_num) he0
  have hsand := exp2_sandwich (x := (2 : ℝ) ^ r1.toReal) (pt := pt.toReal) hw hA
    (by positivity) (by norm_num) (by rw [← hxe]; exact hxr) (by rw [← hxe2]; exact hxr2) hpos.le herr
  rw [hxe]
  refine ⟨hfin, ?_, ?_⟩
  · have := hsand.1; nlinarith
  · have := hsand.2; nlinarith

/-- **`float32` to `float16` rounding** of a finite value of magnitude at most `65504`. -/
theorem cvt_good {x : F32} {p : F16} (hx : x.isFinite = true) (hmax : |x.toReal| ≤ 65504)
    (hp : FloatV.NaNEq (FloatV.cvt x) p) :
    p.isFinite = true ∧ |p.toReal - x.toReal| ≤ |x.toReal| * u16 + eta16 := by
  have hm : |x.toReal| ≤ ((Format.binary16.maxVal : ℚ) : ℝ) := by
    rw [Format.binary16_maxVal_real]; exact hmax
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err_real (g := Format.binary16) Format.binary16_regular hx hm
  have hpe : p = FloatV.cvt x := hp.eq_of_isFinite hfin
  rw [hpe]
  exact ⟨hfin, herr⟩

/-- A finite `float16` of magnitude at most `η16` is zero. -/
theorem f16_tiny_zero {p : F16} (hp : |p.toReal| ≤ eta16) : p.toReal = 0 := by
  by_contra h
  have h2 := FloatV.two_eta_le_abs_toReal Format.binary16_regular h
  have : ((Format.binary16.eta : ℚ) : ℝ) = eta16 := rfl
  have hpos : (0 : ℝ) < eta16 := by unfold eta16; rw [Format.binary16_eta_real]; positivity
  rw [this] at h2
  linarith

/-- The product of `+0` and a finite `float32` is a zero. -/
theorem mul_zero_f32 {r ds : F32} (hr : r.isFinite = true)
    (hds : FloatV.NaNEq (FloatV.mul (FloatV.ofPat 0) r) ds) :
    ds.isFinite = true ∧ ds.toReal = 0 := by
  have h0 : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero Format.binary32_regular
  have hmax : |(FloatV.ofPat 0 : F32).toReal * r.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    simp only [FloatV.toReal_ofPat_zero, zero_mul, abs_zero]
    exact_mod_cast (by rw [Format.binary32_maxVal]; norm_num : (0 : ℚ) ≤ Format.binary32.maxVal)
  obtain ⟨hfin, herr⟩ := FloatV.mul_err_real Format.binary32_regular h0 hr hmax
  have hde : ds = FloatV.mul (FloatV.ofPat 0) r := hds.eq_of_isFinite hfin
  rw [hde]
  refine ⟨hfin, ?_⟩
  simp only [FloatV.toReal_ofPat_zero, zero_mul, abs_zero, sub_zero, zero_mul, zero_add] at herr
  by_contra h
  have h2 := FloatV.two_eta_le_abs_toReal Format.binary32_regular h
  have hpos : (0 : ℝ) < ((Format.binary32.eta : ℚ) : ℝ) := by
    rw [Format.binary32_eta_real]; positivity
  linarith

/-- A dot over at most `256` finite `float16` pairs has lifted magnitude sum at most `2^42`. -/
theorem sum_lm_le {K : Nat} (hK : K ≤ 256) (f g : Nat → F16) (hf : ∀ c < K, (f c).isFinite = true)
    (hg : ∀ c < K, (g c).isFinite = true) : ∑ c ∈ Finset.range K, lm (f c) * lm (g c) ≤ 2 ^ 42 := by
  calc ∑ c ∈ Finset.range K, lm (f c) * lm (g c) ≤ ∑ _c ∈ Finset.range K, (65504 * 65504 : ℝ) := by
        refine Finset.sum_le_sum fun c hc => ?_
        have hc' := Finset.mem_range.mp hc
        exact mul_le_mul (lm_le_65504 (hf c hc')) (lm_le_65504 (hg c hc')) (lm_nonneg _)
          (by norm_num)
    _ = K * (65504 * 65504) := by simp
    _ ≤ 256 * (65504 * 65504) := by
        have : (K : ℝ) ≤ 256 := by exact_mod_cast hK
        exact mul_le_mul_of_nonneg_right this (by norm_num)
    _ ≤ 2 ^ 42 := by norm_num

/-- A sum of products is bounded by the sum of the lifted magnitudes' products. -/
theorem abs_sum_le_sum_lm (K : Nat) (f g : Nat → F16) :
    |∑ c ∈ Finset.range K, (f c).toReal * (g c).toReal| ≤ ∑ c ∈ Finset.range K, lm (f c) * lm (g c) :=
  (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun _ _ => abs_mul_le_lm _ _)

/-- **The computed `P` and `dS` of one pair** (lines 293–309) are `PGood` and `DSGood`. -/
theorem pair_good {P : Platform} {fusion : Bool} (hexp : Device.Exp2F32 P) (d : Inst) (hd : d.ok)
    (hv : HeadV) (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX)
    {d1 d2 r1 r2 pt pt' ds : F32} {p16 q16 : F16}
    (h1 : d1.isFinite = true ∧ |d1.toReal - sQK d hv n m| ≤ rH d * aQK d hv n m + aH d)
    (h2 : d2.isFinite = true ∧ |d2.toReal - sVD d hv n m| ≤ rH d * aVD d hv n m + aH d)
    (hr1 : AddRel fusion d1 (FloatV.neg (hv.lse m)) r1) (hex : P.math .exp2 .binary32 r1 pt)
    (hpt : (Allowed d n m → pt' = pt) ∧ (¬ Allowed d n m → pt' = FloatV.ofPat 0))
    (hr2 : AddRel fusion d2 (FloatV.neg (hv.dl m)) r2) (hds : FloatV.NaNEq (FloatV.mul pt' r2) ds)
    (hp : FloatV.NaNEq (FloatV.cvt pt') p16) (hq : FloatV.NaNEq (FloatV.cvt ds) q16) :
    PGood d hv n m p16 ∧ DSGood d hv n m q16 := by
  have hHD : d.HEAD_DIM ≤ 256 := by have := hd.hd; omega
  have hkf : ∀ c < d.HEAD_DIM, (hv.k n c).isFinite = true := fun c hc => hok.k n c hn hc
  have hqf : ∀ c < d.HEAD_DIM, (hv.q m c).isFinite = true := fun c hc => hok.q m c hm hc
  have hvf : ∀ c < d.HEAD_DIM, (hv.v n c).isFinite = true := fun c hc => hok.v n c hn hc
  have hdf : ∀ c < d.HEAD_DIM, (hv.dO m c).isFinite = true := fun c hc => hok.dO m c hm hc
  have haQK : aQK d hv n m ≤ 2 ^ 42 := sum_lm_le hHD (hv.k n) (hv.q m) hkf hqf
  have haVD : aVD d hv n m ≤ 2 ^ 42 := sum_lm_le hHD (hv.v n) (hv.dO m) hvf hdf
  have hsQK : |sQK d hv n m| ≤ aQK d hv n m := abs_sum_le_sum_lm _ (hv.k n) (hv.q m)
  have hsVD : |sVD d hv n m| ≤ aVD d hv n m := abs_sum_le_sum_lm _ (hv.v n) (hv.dO m)
  obtain ⟨hr1f, hr1e⟩ := sub_chain hd (aQK_nonneg d hv n m) haQK hsQK h1 (hok.lse m hm) hr1
  obtain ⟨hr2f, hr2e⟩ := sub_chain hd (aVD_nonneg d hv n m) haVD hsVD h2 (hok.dl m hm) hr2
  have hr2y : |r2.toReal| ≤ yB d hv n m := by
    have := abs_sub_abs_le_abs_sub r2.toReal (yE d hv n m)
    unfold yB eY
    have e : r2.toReal - yE d hv n m = r2.toReal - (sVD d hv n m - (hv.dl m).toReal) := rfl
    rw [e] at this
    linarith
  have hyB := yB_nonneg d hv n m
  have hu16 := u16_nonneg
  have hu32 := u32_nonneg
  have he16 := eta16_nonneg
  have he32 := eta32_nonneg
  by_cases hal : Allowed d n m
  · have hpt1 := hpt.1 hal
    subst hpt1
    obtain ⟨hfit1, hfit2⟩ := hok.fits n m hn hm hal
    have he0 : 0 ≤ eX d hv n m := eSub_nonneg d (aQK_nonneg d hv n m)
    have hrx : |r1.toReal - xE d hv n m| ≤ eX d hv n m := hr1e
    obtain ⟨hptf, hptm, hptd⟩ := exp2_step hexp he0 hfit1 hr1f hrx hex
    have hpBr : pBr d hv n m = (1 + 2 ^ (-22 : ℤ)) * 2 ^ (xE d hv n m + eX d hv n m) + 2 ^ (-126 : ℤ) :=
      rfl
    rw [← hpBr] at hptm hptd
    have hpBr1 := pBr_le hfit1
    have hpBr0 := pBr_nonneg d hv n m
    have hw : 0 < (2 : ℝ) ^ xE d hv n m := Real.rpow_pos_of_pos (by norm_num) _
    have hpE : pE d hv n m = 2 ^ xE d hv n m := by unfold pE; rw [ite_eq_left hal]
    -- the `float16` `P`
    obtain ⟨hp16, hp16e⟩ := cvt_good hptf (by linarith) hp
    have hpt_u : |pt'.toReal| * u16 ≤ pBr d hv n m * u16 := mul_le_mul_of_nonneg_right hptm hu16
    have hP1 : |p16.toReal - pE d hv n m| ≤ p16Err d hv n m := by
      rw [hpE]; unfold p16Err; rw [ite_eq_left hal]
      have := abs_sub_le p16.toReal pt'.toReal (2 ^ xE d hv n m)
      linarith
    have hP2 : |p16.toReal| ≤ p16B d hv n m := by
      unfold p16B; rw [ite_eq_left hal]
      have h3 := abs_sub_abs_le_abs_sub p16.toReal pt'.toReal
      linarith
    -- the `float32` `dS`
    have hyE : |r2.toReal - yE d hv n m| ≤ eY d hv n m := hr2e
    have hprod : |pt'.toReal * r2.toReal| ≤ pBr d hv n m * yB d hv n m := by
      rw [abs_mul]; exact mul_le_mul hptm hr2y (abs_nonneg _) hpBr0
    have hprod2 : |pt'.toReal * r2.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [Format.binary32_maxVal]; push_cast
      have : (2 : ℝ) ^ 15 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith
    obtain ⟨hdsf, hdse⟩ := FloatV.mul_err_real Format.binary32_regular hptf hr2f hprod2
    have hdsE : ds = FloatV.mul pt' r2 := hds.eq_of_isFinite hdsf
    have hu32' : ((Format.binary32.u : ℚ) : ℝ) = u32 := rfl
    have he32' : ((Format.binary32.eta : ℚ) : ℝ) = eta32 := rfl
    rw [← hdsE, hu32', he32'] at hdse
    have hprodU : |pt'.toReal * r2.toReal| * u32 ≤ pBr d hv n m * yB d hv n m * u32 :=
      mul_le_mul_of_nonneg_right hprod hu32
    have hsw : |pt'.toReal - 2 ^ xE d hv n m| ≤ pBr d hv n m - 2 ^ xE d hv n m := hptd
    have hdecomp : pt'.toReal * r2.toReal - 2 ^ xE d hv n m * yE d hv n m =
        (pt'.toReal - 2 ^ xE d hv n m) * r2.toReal + 2 ^ xE d hv n m * (r2.toReal - yE d hv n m) := by
      ring
    have hd1 : |(pt'.toReal - 2 ^ xE d hv n m) * r2.toReal| ≤ (pBr d hv n m - 2 ^ xE d hv n m) * yB d hv n m := by
      rw [abs_mul]; exact mul_le_mul hsw hr2y (abs_nonneg _) ((abs_nonneg _).trans hsw)
    have hd2 : |2 ^ xE d hv n m * (r2.toReal - yE d hv n m)| ≤ 2 ^ xE d hv n m * eY d hv n m := by
      rw [abs_mul, abs_of_pos hw]; exact mul_le_mul_of_nonneg_left hyE hw.le
    have hD1 : |ds.toReal - dsE d hv n m| ≤ dsErr d hv n m := by
      have e : dsE d hv n m = 2 ^ xE d hv n m * yE d hv n m := by unfold dsE; rw [hpE]
      rw [e]
      have h3 := abs_sub_le ds.toReal (pt'.toReal * r2.toReal) (2 ^ xE d hv n m * yE d hv n m)
      have h4 := abs_add_le ((pt'.toReal - 2 ^ xE d hv n m) * r2.toReal)
        (2 ^ xE d hv n m * (r2.toReal - yE d hv n m))
      rw [← hdecomp] at h4
      unfold dsErr
      linarith
    have hD2 : |ds.toReal| ≤ dsB d hv n m := by
      unfold dsB
      have h3 := abs_sub_abs_le_abs_sub ds.toReal (pt'.toReal * r2.toReal)
      linarith
    have hdsB : dsB d hv n m ≤ 2 ^ 15 + 32 := by
      unfold dsB
      have h5 := u32_le
      have h6 := eta32_le
      have : (1 + u32) * (pBr d hv n m * yB d hv n m) ≤ (1 + 1 / 2048) * 2 ^ 15 :=
        mul_le_mul (by linarith) hfit2 (mul_nonneg hpBr0 hyB) (by norm_num)
      have h7 : (1 + 1 / 2048 : ℝ) * 2 ^ 15 + 1 / 2048 ≤ 2 ^ 15 + 32 := by norm_num
      linarith
    have hdsf' : ds.isFinite = true := by rw [hdsE]; exact hdsf
    obtain ⟨hq16, hq16e⟩ := cvt_good hdsf' (by
      have : (2 : ℝ) ^ 15 + 32 ≤ 65504 := by norm_num
      linarith) hq
    have hdsu : |ds.toReal| * u16 ≤ dsB d hv n m * u16 := mul_le_mul_of_nonneg_right hD2 hu16
    refine ⟨⟨hp16, hP1, lm_le_of_abs_le ?_⟩, ⟨hq16, ?_, lm_le_of_abs_le ?_⟩⟩
    · exact hP2
    · unfold ds16Err; rw [ite_eq_left hal]
      have := abs_sub_le q16.toReal ds.toReal (dsE d hv n m)
      linarith
    · unfold ds16B; rw [ite_eq_left hal]
      have h3 := abs_sub_abs_le_abs_sub q16.toReal ds.toReal
      linarith
  · have hpt0 := hpt.2 hal
    subst hpt0
    have hzero : (FloatV.ofPat 0 : F32).isFinite = true :=
      FloatV.isFinite_ofPat_zero Format.binary32_regular
    have hzr : (FloatV.ofPat 0 : F32).toReal = 0 := FloatV.toReal_ofPat_zero
    obtain ⟨hp16, hp16e⟩ := cvt_good hzero (by rw [hzr]; norm_num) hp
    rw [hzr] at hp16e
    have hp0 : p16.toReal = 0 := f16_tiny_zero (by simpa using hp16e)
    obtain ⟨hdsf, hds0⟩ := mul_zero_f32 hr2f hds
    obtain ⟨hq16, hq16e⟩ := cvt_good hdsf (by rw [hds0]; norm_num) hq
    have hq0 : q16.toReal = 0 := f16_tiny_zero (by rw [hds0] at hq16e; simpa using hq16e)
    have hpE : pE d hv n m = 0 := by unfold pE; rw [ite_eq_right hal]
    have hpL : p16L d hv n m = lift16 0 := by unfold p16L p16B; rw [ite_eq_right hal]
    have hsL : ds16L d hv n m = lift16 0 := by unfold ds16L ds16B; rw [ite_eq_right hal]
    refine ⟨⟨hp16, ?_, ?_⟩, ⟨hq16, ?_, ?_⟩⟩
    · rw [hpE, hp0]; unfold p16Err; rw [ite_eq_right hal]; simp
    · rw [hpL]; exact lm_le_of_abs_le (by rw [hp0]; simp)
    · have : dsE d hv n m = 0 := by unfold dsE; rw [hpE]; simp
      rw [this, hq0]; unfold ds16Err; rw [ite_eq_right hal]; simp
    · rw [hsL]; exact lm_le_of_abs_le (by rw [hq0]; simp)

/-- `LN2`, as the `float32` constant, is finite. -/
theorem ln2F_finite : ln2F.isFinite = true := by
  unfold ln2F ln2Pat
  decide +kernel

/-- The value of `LN2` as a `float32`: `0x3f317218 = 1453635/2097152`. -/
theorem ln2F_toRat : ln2F.toRat = 1453635 / 2097152 := by
  unfold ln2F ln2Pat
  decide +kernel

/-- `LN2` is nonnegative. -/
theorem ln2F_nonneg : 0 ≤ ln2F.toReal := by
  unfold FloatV.toReal; rw [ln2F_toRat]; norm_num

/-- `LN2` is at most `2^16` in magnitude. -/
theorem ln2F_abs_le : |ln2F.toReal| ≤ 2 ^ 16 := by
  unfold FloatV.toReal; rw [ln2F_toRat]; norm_num [abs_of_pos]

end Claims.AttnBwd
