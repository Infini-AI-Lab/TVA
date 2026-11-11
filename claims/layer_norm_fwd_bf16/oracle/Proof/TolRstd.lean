import Proof.TolBase

set_option maxHeartbeats 1000000

/-! The stored `rstd`'s tolerance: the variances's fused reduction (README step 3), the division
by `D`, the fused addition of `eps`, `tl.sqrt` and `1/·` (step 4), and the relaxation of their
roundings into `gam` (`rstdTol`). -/

namespace Claims.LayerNormFwd

open TVA
open Format
open scoped BigOperators

noncomputable section

/-- **Turning a bound `|r − t| ≤ |t|·u + η` into the product form** used by the variance chain.
If `t` is within `M·u + η` of `s`, then `r = (s + e)·(1 + c)` with `|c| ≤ 2u` and
`|e| ≤ M·u + 4η`.  (Either the relative error of `r` against `t` is within `2u`, or it is so small
that `|t|·u ≤ η` and everything is absolute.) -/
theorem rstd_tol_clip {u η M s t r : ℚ} (hu : 0 ≤ u) (hη : 0 ≤ η) (_hM : 0 ≤ M)
    (hrt : |r - t| ≤ |t| * u + η) (hts : |t - s| ≤ M * u + η) :
    ∃ c e : ℚ, |c| ≤ 2 * u ∧ |e| ≤ M * u + 4 * η ∧ r = (s + e) * (1 + c) := by
  by_cases ht : t = 0
  · refine ⟨0, r - s, by simpa using hu, ?_, by simp⟩
    rw [ht] at hrt hts
    simp only [abs_zero, zero_mul, zero_add, sub_zero] at hrt
    have h1 : |r - s| ≤ |r| + |s| := by
      have h2 := abs_add_le r (-s)
      simpa [sub_eq_add_neg] using h2
    have h3 : |s| ≤ M * u + η := by simpa using hts
    nlinarith [abs_nonneg s]
  · have ht' : 0 < |t| := abs_pos.mpr ht
    by_cases hc : |r - t| ≤ 2 * u * |t|
    · refine ⟨(r - t) / t, t - s, ?_, ?_, ?_⟩
      · rw [abs_div]
        exact (div_le_iff₀ ht').mpr (by linarith)
      · linarith
      · field_simp
        ring
    · push Not at hc
      refine ⟨0, r - s, by simpa using hu, ?_, by simp⟩
      have hle : |r - t| ≤ 2 * η := by nlinarith [abs_nonneg t]
      have h1 : |r - s| ≤ |r - t| + |t - s| := by
        have h2 := abs_add_le (r - t) (t - s)
        have e : r - t + (t - s) = r - s := by ring
        rwa [e] at h2
      linarith

/-- **The key split of a `tl.addc` node.** `AddRel true a b res` (IEEE or fused) puts `res` within
a *multiplicative* `(1 + c)`, `|c| ≤ 2u`, of `a + b + e`, where the absolute part `e` is at most
`u·max(|a|,|b|) + 4η`.  This is what the variance chain needs and what `FloatV.AddRel.err`'s
`3u(|a|+|b|) + 4η` cannot supply. -/
theorem rstd_tol_addrel_split {a b res : F32} (h : AddRel true a b res)
    (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hM : |a.toRat| + |b.toRat| ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta) :
    res.isFinite = true ∧ ∃ c e : ℚ, |c| ≤ 2 * Format.binary32.u ∧
      |e| ≤ Format.binary32.u * max |a.toRat| |b.toRat| + 4 * Format.binary32.eta ∧
      res.toRat = (a.toRat + b.toRat + e) * (1 + c) := by
  have hf : Format.binary32.Regular := Format.binary32_regular
  have hu0 : (0 : ℚ) ≤ Format.binary32.u := Format.u_nonneg
  have he0 : (0 : ℚ) ≤ Format.binary32.eta := Format.eta_nonneg
  have hmv0 : (0 : ℚ) < Format.binary32.maxVal := Format.maxVal_pos hf
  have hu1 : Format.binary32.u ≤ 1 := by rw [Format.binary32_u]; norm_num
  have hmax : |a.toRat + b.toRat| ≤ Format.binary32.maxVal := by
    have h1 : |a.toRat + b.toRat| ≤ |a.toRat| + |b.toRat| := abs_add_le _ _
    nlinarith [hmv0, he0]
  obtain ⟨haddfin, hadderr⟩ := FloatV.add_err hf ha hb hmax
  have hnan : (FloatV.add a b).isNaN = false := (FloatV.isFinite_iff.mp haddfin).1
  have hM0 : (0 : ℚ) ≤ max |a.toRat| |b.toRat| := le_trans (abs_nonneg _) (le_max_left _ _)
  rcases h with hne | ⟨-, hfu⟩
  · have hr : res = FloatV.add a b := hne.eq hnan
    rw [hr]
    refine ⟨haddfin, ?_⟩
    have hts : |(a.toRat + b.toRat) - (a.toRat + b.toRat)| ≤
        max |a.toRat| |b.toRat| * Format.binary32.u + Format.binary32.eta := by
      have : (0 : ℚ) ≤ max |a.toRat| |b.toRat| * Format.binary32.u + Format.binary32.eta := by
        nlinarith
      simpa using this
    obtain ⟨c, e, hc, he, hre⟩ := rstd_tol_clip hu0 he0 hM0 hadderr hts
    exact ⟨c, e, hc, by nlinarith, hre⟩
  · rcases hfu with ⟨p, q, hp, hq, hpq, z, hne⟩ | ⟨p, hp, hbi, hr⟩ | ⟨q, hq, hai, hr⟩
    · have hpabs := RoundsTo.abs_le hf (by decide) hp ha
      have hqabs := RoundsTo.abs_le hf (by decide) hq hb
      have hpqM : |p + q| ≤ Format.binary32.maxVal := by
        have h1 : |p + q| ≤ |p| + |q| := abs_add_le _ _
        have h3 : Format.binary32.u * (|a.toRat| + |b.toRat|) ≤
            Format.binary32.u * (Format.binary32.maxVal / 2 - 2 * Format.binary32.eta) :=
          mul_le_mul_of_nonneg_left hM hu0
        have h5 : Format.binary32.u * (Format.binary32.maxVal / 2) ≤
            1 * (Format.binary32.maxVal / 2) :=
          mul_le_mul_of_nonneg_right hu1 (by linarith)
        have h4 : Format.binary32.u * (Format.binary32.maxVal / 2 - 2 * Format.binary32.eta) ≤
            Format.binary32.maxVal / 2 := by
          nlinarith [mul_nonneg hu0 he0]
        nlinarith
      obtain ⟨hrfin, hrerr⟩ := FloatV.ofPat_roundRat_err hf false z hpqM
      have hrnan : (FloatV.ofPat (Format.binary32.roundRat .rn false z (p + q)) : F32).isNaN = false :=
        (FloatV.isFinite_iff.mp hrfin).1
      have hr : res = FloatV.ofPat (Format.binary32.roundRat .rn false z (p + q)) := hne.eq hrnan
      rw [hr]
      refine ⟨hrfin, ?_⟩
      have hts : |(p + q) - (a.toRat + b.toRat)| ≤
          max |a.toRat| |b.toRat| * Format.binary32.u + Format.binary32.eta := by
        rcases hpq with hpaeq | hqbeq
        · have hqerr := RoundsTo.err hf (by decide) hq hb
          have hbM : |b.toRat| ≤ max |a.toRat| |b.toRat| := le_max_right _ _
          have hbm : Format.binary32.u * |b.toRat| ≤
              Format.binary32.u * max |a.toRat| |b.toRat| :=
            mul_le_mul_of_nonneg_left hbM hu0
          rw [hpaeq]
          have heq : a.toRat + q - (a.toRat + b.toRat) = q - b.toRat := by ring
          rw [heq]
          linarith
        · have hperr := RoundsTo.err hf (by decide) hp ha
          have haM : |a.toRat| ≤ max |a.toRat| |b.toRat| := le_max_left _ _
          have ham : Format.binary32.u * |a.toRat| ≤
              Format.binary32.u * max |a.toRat| |b.toRat| :=
            mul_le_mul_of_nonneg_left haM hu0
          rw [hqbeq]
          have heq : p + b.toRat - (a.toRat + b.toRat) = p - a.toRat := by ring
          rw [heq]
          linarith
      obtain ⟨c, e, hc, he, hre⟩ := rstd_tol_clip hu0 he0 hM0 hrerr hts
      exact ⟨c, e, hc, by nlinarith [he], hre⟩
    · exfalso
      have hbf : b.isInf = false := (FloatV.isFinite_iff.mp hb).2
      rw [hbi] at hbf
      cases hbf
    · exfalso
      have haf : a.isInf = false := (FloatV.isFinite_iff.mp ha).2
      rw [hai] at haf
      cases haf

/-- The rational unit roundoff of binary32, cast to the reals, equals `u32`. -/
theorem rstd_tol_u_cast : ((Format.binary32.u : ℚ) : ℝ) = u32 := by
  rw [Format.binary32_u]; unfold u32; norm_num

/-- The rational underflow constant of binary32, cast to the reals, equals `eta32`. -/
theorem rstd_tol_eta_cast : ((Format.binary32.eta : ℚ) : ℝ) = eta32 := by
  rw [Format.binary32_eta]; unfold eta32; norm_num

/-- A `float32` addition away from overflow is `(a + b + e)(1 + c)` with `|c| ≤ 2u` and a small `e`. -/
theorem rstd_tol_addrel_split_real {a b res : F32} (h : AddRel true a b res)
    (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hM : |a.toRat| + |b.toRat| ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta) :
    res.isFinite = true ∧ ∃ c e : ℝ, |c| ≤ 2 * u32 ∧
      |e| ≤ u32 * max |a.toReal| |b.toReal| + 4 * eta32 ∧
      res.toReal = (a.toReal + b.toReal + e) * (1 + c) := by
  obtain ⟨hfin, c, e, hc, he, hre⟩ := rstd_tol_addrel_split h ha hb hM
  refine ⟨hfin, c, e, ?_, ?_, ?_⟩
  · have hc' : |(c : ℝ)| ≤ 2 * ((Format.binary32.u : ℚ) : ℝ) := by exact_mod_cast hc
    rwa [rstd_tol_u_cast] at hc'
  · have he' : |(e : ℝ)| ≤ ((Format.binary32.u : ℚ) : ℝ) *
        max |(a.toRat : ℝ)| |(b.toRat : ℝ)| + 4 * ((Format.binary32.eta : ℚ) : ℝ) := by
      exact_mod_cast he
    rw [rstd_tol_u_cast, rstd_tol_eta_cast] at he'
    simpa [FloatV.toReal] using he'
  · have hre' : (res.toRat : ℝ) =
        (((a.toRat : ℝ) + (b.toRat : ℝ) + (e : ℝ)) * (1 + (c : ℝ))) := by
      exact_mod_cast hre
    simpa [FloatV.toReal] using hre'

/-- A sum over `List.range n` is the `Finset.range n` sum. -/
theorem rstd_tol_list_sum_range {β : Type} [AddCommMonoid β] (f : Nat → β) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]
    simp

/-- For an accepted instance the row length `D`, as a real, is positive. -/
theorem rstd_tol_D_pos {d : Inst} (hd : d.ok) : (0 : ℝ) < d.D := by
  exact_mod_cast hd.d_pos

/-- The reduction depth `log₂ BD` is at most `9`, hence at most `10`. -/
theorem rstd_tol_depth_le {d : Inst} (hd : d.ok) : depth d ≤ 10 := by
  unfold depth
  refine Nat.clog_le_of_le_pow ?_
  have h1 : d.BD < 2 * d.D := hd.bd_lt
  have h2 : d.D ≤ 512 := hd.d_le
  calc d.BD ≤ 2 * 512 := by omega
    _ = 2 ^ 10 := by norm_num

/-- The mean of absolute values is at most the bound `2^57` on each value. -/
theorem rstd_tol_absMean_le {d : Inst} {init : Mem} {t : Nat}
    (hbdd : ∀ j, j < d.D → |sR d init t j| ≤ 2 ^ 57) (hD : 0 < d.D) :
    absMean d init t ≤ 2 ^ 57 := by
  have hD' : (0 : ℝ) < d.D := by exact_mod_cast hD
  unfold absMean
  rw [div_le_iff₀ hD']
  have h1 : (∑ j ∈ Finset.range d.D, |sR d init t j|)
      ≤ ∑ _j ∈ Finset.range d.D, (2 ^ 57 : ℝ) := by
    refine Finset.sum_le_sum fun j hj => ?_
    exact hbdd j (Finset.mem_range.mp hj)
  have h2 : (∑ _j ∈ Finset.range d.D, (2 ^ 57 : ℝ)) = 2 ^ 57 * d.D := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    ring
  linarith

/-- `meanTol` is at most `2^38`. -/
theorem rstd_tol_meanTol_le {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (hbdd : ∀ j, j < d.D → |sR d init t j| ≤ 2 ^ 57) :
    meanTol d init t ≤ 2 ^ 38 := by
  have hD : 0 < d.D := hd.d_pos
  have hA := rstd_tol_absMean_le hbdd hD
  have hL := rstd_tol_depth_le hd
  have hA0 : 0 ≤ absMean d init t := by
    unfold absMean
    positivity
  have h1 : ((depth d : ℝ) + 4) * 2 ^ (-23 : ℤ) * absMean d init t
      ≤ (10 + 4) * 2 ^ (-23 : ℤ) * 2 ^ 57 := by
    have h23 : (0 : ℝ) < 2 ^ (-23 : ℤ) := by norm_num
    have h2 : (depth d : ℝ) + 4 ≤ 10 + 4 := by
      have : (depth d : ℝ) ≤ 10 := by exact_mod_cast hL
      linarith
    have h3 : ((depth d : ℝ) + 4) * 2 ^ (-23 : ℤ) ≤ (10 + 4) * 2 ^ (-23 : ℤ) :=
      mul_le_mul_of_nonneg_right h2 (le_of_lt h23)
    exact mul_le_mul h3 hA hA0 (by positivity)
  have h5 : (14 : ℝ) * 2 ^ (-23 : ℤ) * 2 ^ 57 + 2 ^ (-140 : ℤ) ≤ 2 ^ 38 := by
    norm_num
  unfold meanTol
  linarith [h1]

/-- The lane mean error: `x j` is `(d j + ε j)·(1 + c j)` with `|c j| ≤ 2u` and `ε j` the lane's
mean error, at most `meanTol` plus the subtraction's own rounding. -/
theorem rstd_tol_dev {d : Inst} {init : Mem} {t j : Nat} (_hj : j < d.D)
    {bx bmean x : F32}
    (hbxfin : bx.isFinite = true) (hbmfin : bmean.isFinite = true)
    (hval : bx.toReal = sR d init t j)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hbddj : |sR d init t j| ≤ 2 ^ 57)
    (habsMean : absMean d init t ≤ 2 ^ 57)
    (hmeanTol : meanTol d init t ≤ 2 ^ 38)
    (hmeanRef : |meanRef d init t| ≤ absMean d init t)
    (hdev : AddRel true bx (FloatV.neg bmean) x) :
    x.isFinite = true ∧ ∃ c ε : ℝ, |c| ≤ 2 * u32 ∧
      |ε| ≤ meanTol d init t + u32 * (|sR d init t j| + absMean d init t + meanTol d init t)
        + 4 * eta32 ∧
      x.toReal = (sR d init t j - meanRef d init t + ε) * (1 + c) := by
  -- the magnitude of `bmean`
  have hβ : |bmean.toReal| ≤ absMean d init t + meanTol d init t := by
    have h1 : |bmean.toReal| ≤ |bmean.toReal - meanRef d init t| + |meanRef d init t| := by
      have h2 := abs_add_le (bmean.toReal - meanRef d init t) (meanRef d init t)
      have e : bmean.toReal - meanRef d init t + meanRef d init t = bmean.toReal := by ring
      rwa [e] at h2
    linarith
  -- the premise of the split, in `ℚ`
  have hM : |bx.toRat| + |(FloatV.neg bmean).toRat|
      ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
    have h1 : |bx.toRat| ≤ (2 : ℚ) ^ 57 := by
      have h2 : |(bx.toRat : ℝ)| ≤ (2 : ℝ) ^ 57 := by
        have h3 : |bx.toReal| ≤ (2 : ℝ) ^ 57 := by rw [hval]; exact hbddj
        simpa [FloatV.toReal] using h3
      exact_mod_cast h2
    have h2 : |(FloatV.neg bmean).toRat| ≤ (2 : ℚ) ^ 57 + (2 : ℚ) ^ 38 := by
      rw [FloatV.toRat_neg, abs_neg]
      have h3 : |(bmean.toRat : ℝ)| ≤ (2 : ℝ) ^ 57 + 2 ^ 38 := by
        have hb : (bmean.toRat : ℝ) = bmean.toReal := rfl
        rw [hb]
        linarith
      exact_mod_cast h3
    have hnum : (2 : ℚ) ^ 57 + ((2 : ℚ) ^ 57 + (2 : ℚ) ^ 38)
        ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
      rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
    linarith
  have hnegfin : (FloatV.neg bmean).isFinite = true := by
    rw [FloatV.isFinite_neg]; exact hbmfin
  obtain ⟨hxfin, c, e, hc, he, hxeq⟩ := rstd_tol_addrel_split_real hdev hbxfin hnegfin hM
  have hneg : (FloatV.neg bmean).toReal = -bmean.toReal := by
    simp [FloatV.toReal, FloatV.toRat_neg]
  refine ⟨hxfin, c, (meanRef d init t - bmean.toReal) + e, hc, ?_, ?_⟩
  · -- size of the lane mean error
    have h1 : |(meanRef d init t - bmean.toReal) + e|
        ≤ meanTol d init t + (u32 * max |bx.toReal| |(FloatV.neg bmean).toReal| + 4 * eta32) := by
      have h2 : |meanRef d init t - bmean.toReal| ≤ meanTol d init t := by
        rw [abs_sub_comm]; exact hmean
      calc |(meanRef d init t - bmean.toReal) + e|
          ≤ |meanRef d init t - bmean.toReal| + |e| := abs_add_le _ _
        _ ≤ meanTol d init t + (u32 * max |bx.toReal| |(FloatV.neg bmean).toReal| + 4 * eta32) := by
            linarith
    have h2 : max |bx.toReal| |(FloatV.neg bmean).toReal| ≤
        |sR d init t j| + (absMean d init t + meanTol d init t) := by
      rw [hneg, abs_neg, hval]
      have := le_abs_self (sR d init t j)
      have h3 : max |sR d init t j| |bmean.toReal| ≤ |sR d init t j| + |bmean.toReal| :=
        max_le (le_add_of_nonneg_right (abs_nonneg _)) (le_add_of_nonneg_left (abs_nonneg _))
      linarith
    have hu0 : (0 : ℝ) ≤ u32 := le_of_lt u32_pos
    have h4 : u32 * max |bx.toReal| |(FloatV.neg bmean).toReal|
        ≤ u32 * (|sR d init t j| + (absMean d init t + meanTol d init t)) :=
      mul_le_mul_of_nonneg_left h2 hu0
    linarith
  · rw [hxeq, hneg, hval]
    ring

/-- One rounded square: a finite lane of magnitude at most `2^58` squares to a finite value within
`u·x² + η`. -/
theorem rstd_tol_sq {x sq : F32} (hxfin : x.isFinite = true)
    (hxbd : |x.toReal| ≤ 2 ^ 58)
    (hsq : FloatV.NaNEq (FloatV.mul x x) sq) :
    sq.isFinite = true ∧ |sq.toReal - x.toReal ^ 2| ≤ u32 * x.toReal ^ 2 + eta32 := by
  have hf : Format.binary32.Regular := Format.binary32_regular
  have hxrat : |x.toRat| ≤ (2 : ℚ) ^ 58 := by
    have h1 : |(x.toRat : ℝ)| ≤ (2 : ℝ) ^ 58 := by simpa [FloatV.toReal] using hxbd
    exact_mod_cast h1
  have hmax : |x.toRat * x.toRat| ≤ (Format.binary32.maxVal : ℚ) := by
    rw [abs_mul]
    have h2 : |x.toRat| * |x.toRat| ≤ (2 : ℚ) ^ 58 * (2 : ℚ) ^ 58 :=
      mul_le_mul hxrat hxrat (abs_nonneg _) (by norm_num)
    have h3 : (2 : ℚ) ^ 58 * (2 : ℚ) ^ 58 = 2 ^ 116 := by norm_num
    have h4 : (2 : ℚ) ^ 116 ≤ (Format.binary32.maxVal : ℚ) := by
      rw [Format.binary32_maxVal]; norm_num
    linarith
  obtain ⟨hmulfin, hmulerr⟩ := FloatV.mul_err hf hxfin hxfin hmax
  have hmulnan : (FloatV.mul x x).isNaN = false := (FloatV.isFinite_iff.mp hmulfin).1
  have hq : sq = FloatV.mul x x := hsq.eq hmulnan
  refine ⟨?_, ?_⟩
  · rw [hq]; exact hmulfin
  · rw [hq]
    have hcast : |((FloatV.mul x x).toRat : ℝ) - ((x.toRat * x.toRat : ℚ) : ℝ)|
        ≤ (((|x.toRat * x.toRat| * Format.binary32.u + Format.binary32.eta : ℚ)) : ℝ) := by
      exact_mod_cast hmulerr
    push_cast at hcast
    rw [rstd_tol_u_cast, rstd_tol_eta_cast] at hcast
    have h1 : |((FloatV.mul x x).toRat : ℝ) - (x.toReal * x.toReal)|
        ≤ |x.toReal * x.toReal| * u32 + eta32 := by
      simpa [FloatV.toReal] using hcast
    have h3 : |x.toReal * x.toReal| = x.toReal ^ 2 := by
      rw [abs_mul, abs_mul_abs_self, pow_two]
    rw [h3] at h1
    simpa [FloatV.toReal, pow_two, mul_comm] using h1

/-- A masked lane (`x = 0`) squares to something at most `η` in magnitude. -/
theorem rstd_tol_sq_zero {x sq : F32} (hxfin : x.isFinite = true) (hx0 : x.toReal = 0)
    (hsq : FloatV.NaNEq (FloatV.mul x x) sq) :
    sq.isFinite = true ∧ |sq.toReal| ≤ eta32 := by
  have hxbd : |x.toReal| ≤ 2 ^ 58 := by rw [hx0]; norm_num
  obtain ⟨hfin, h⟩ := rstd_tol_sq hxfin hxbd hsq
  refine ⟨hfin, ?_⟩
  rw [hx0] at h
  simpa using h

/-- The reference variance of a row is nonnegative. -/
theorem rstd_tol_varRef_nonneg {d : Inst} {init : Mem} {t : Nat} : 0 ≤ varRef d init t := by
  unfold varRef; positivity

/-- The reference root-mean-square of a row is nonnegative. -/
theorem rstd_tol_rmsS_nonneg {d : Inst} {init : Mem} {t : Nat} : 0 ≤ rmsS d init t := by
  unfold rmsS; positivity

/-- The mean absolute value of a row is nonnegative. -/
theorem rstd_tol_absMean_nonneg {d : Inst} {init : Mem} {t : Nat} : 0 ≤ absMean d init t := by
  unfold absMean; positivity

/-- The mean tolerance of a row is nonnegative. -/
theorem rstd_tol_meanTol_nonneg {d : Inst} {init : Mem} {t : Nat} : 0 ≤ meanTol d init t := by
  have h1 : (0 : ℝ) ≤ absMean d init t := rstd_tol_absMean_nonneg
  have h2 : (0 : ℝ) ≤ ((depth d : ℝ) + 4) := by positivity
  have h3 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) := by norm_num
  have h4 : (0 : ℝ) ≤ (2 : ℝ) ^ (-140 : ℤ) := by norm_num
  unfold meanTol
  nlinarith [mul_nonneg h2 h3, mul_nonneg (mul_nonneg h2 h3) h1]

/-- The mean's magnitude `A` is at most the root mean square. -/
theorem rstd_tol_absMean_le_rmsS {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    absMean d init t ≤ rmsS d init t := by
  have hD' : (0 : ℝ) < (d.D : ℝ) := by exact_mod_cast hD
  have hA0 : 0 ≤ absMean d init t := rstd_tol_absMean_nonneg
  have hsum : (∑ j ∈ Finset.range d.D, |sR d init t j|)
      = (d.D : ℝ) * absMean d init t := by
    unfold absMean
    field_simp
  have hkey : (d.D : ℝ) * absMean d init t ^ 2
      ≤ ∑ j ∈ Finset.range d.D, sR d init t j ^ 2 := by
    have hnn : 0 ≤ ∑ j ∈ Finset.range d.D,
        (|sR d init t j| - absMean d init t) ^ 2 :=
      Finset.sum_nonneg fun j _ => sq_nonneg _
    have hexp : (∑ j ∈ Finset.range d.D, (|sR d init t j| - absMean d init t) ^ 2)
        = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2)
          - (d.D : ℝ) * absMean d init t ^ 2 := by
      have h1 : ∀ j, (|sR d init t j| - absMean d init t) ^ 2
          = sR d init t j ^ 2 - 2 * absMean d init t * |sR d init t j|
            + absMean d init t ^ 2 := by
        intro j; rw [sub_sq, sq_abs]; ring
      calc (∑ j ∈ Finset.range d.D, (|sR d init t j| - absMean d init t) ^ 2)
          = ∑ j ∈ Finset.range d.D, (sR d init t j ^ 2 - 2 * absMean d init t * |sR d init t j|
              + absMean d init t ^ 2) := Finset.sum_congr rfl fun j _ => h1 j
        _ = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2)
              - 2 * absMean d init t * (∑ j ∈ Finset.range d.D, |sR d init t j|)
              + (d.D : ℝ) * absMean d init t ^ 2 := by
            rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum,
              Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        _ = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) - (d.D : ℝ) * absMean d init t ^ 2 := by
            rw [hsum]; ring
    linarith
  have h4 : absMean d init t ^ 2 ≤ (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) / (d.D : ℝ) := by
    rw [le_div_iff₀ hD']; linarith
  unfold rmsS
  exact Real.le_sqrt_of_sq_le h4

/-- The square of the row root-mean-square is the average of the squared entries. -/
theorem rstd_tol_rmsS_sq {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    rmsS d init t ^ 2 = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) / (d.D : ℝ) := by
  have hD' : (0 : ℝ) < (d.D : ℝ) := by exact_mod_cast hD
  unfold rmsS
  exact Real.sq_sqrt (by positivity)

/-- The sum of squared entries of a row equals `D` times the squared root-mean-square. -/
theorem rstd_tol_sum_sq_eq {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    (∑ j ∈ Finset.range d.D, sR d init t j ^ 2)
      = (d.D : ℝ) * rmsS d init t ^ 2 := by
  rw [rstd_tol_rmsS_sq hD]; field_simp

/-- The sum of absolute entries of a row equals `D` times the mean absolute value. -/
theorem rstd_tol_sum_abs_eq {d : Inst} {init : Mem} {t : Nat} (hD : (d.D : ℝ) ≠ 0) :
    (∑ j ∈ Finset.range d.D, |sR d init t j|) = (d.D : ℝ) * absMean d init t := by
  unfold absMean
  field_simp

/-- For an accepted instance, reference variance plus `epsR` is positive. -/
theorem rstd_tol_Ws_pos {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    (0 : ℝ) < varRef d init t + epsR d := by
  have h1 := rstd_tol_varRef_nonneg (d := d) (init := init) (t := t)
  have h2 := epsR_pos hd
  linarith

/-- For an accepted instance the reference rstd of a row is positive. -/
theorem rstd_tol_R_pos {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    0 < rstdRef d init t := by
  unfold rstdRef
  exact one_div_pos.mpr (Real.sqrt_pos.mpr (rstd_tol_Ws_pos (init := init) (t := t) hd))

/-- `rstdRef` is at most `2^63` (`eps ≥ 2^-125`). -/
theorem rstd_tol_R_le {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    rstdRef d init t ≤ 2 ^ (63 : ℤ) := by
  have hW := rstd_tol_Ws_pos (init := init) (t := t) hd
  have heps := epsR_ge hd
  have h1 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 ≤ varRef d init t + epsR d := by
    have h2 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = (2 : ℝ) ^ (-126 : ℤ) := by
      rw [pow_two, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]; norm_num
    have h3 : (2 : ℝ) ^ (-126 : ℤ) ≤ (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    have h4 : 0 ≤ varRef d init t := rstd_tol_varRef_nonneg
    linarith
  have h5 : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt (varRef d init t + epsR d) :=
    Real.le_sqrt_of_sq_le h1
  unfold rstdRef
  rw [one_div]
  have h6 : (0 : ℝ) < (2 : ℝ) ^ (-63 : ℤ) := by norm_num
  calc (Real.sqrt (varRef d init t + epsR d))⁻¹ ≤ ((2 : ℝ) ^ (-63 : ℤ))⁻¹ := by
        simpa [one_div] using one_div_le_one_div_of_le h6 h5
    _ = 2 ^ (63 : ℤ) := by
        rw [← zpow_neg]
        norm_num

/-- `rstdRef · (varRef + epsR) = sqrt (varRef + epsR)`. -/
theorem rstd_tol_R_mul {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    rstdRef d init t * (varRef d init t + epsR d)
      = Real.sqrt (varRef d init t + epsR d) := by
  have hW := rstd_tol_Ws_pos (init := init) (t := t) hd
  have hsq : Real.sqrt (varRef d init t + epsR d) ^ 2 = varRef d init t + epsR d :=
    Real.sq_sqrt (le_of_lt hW)
  unfold rstdRef
  rw [one_div]
  have hs : Real.sqrt (varRef d init t + epsR d) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hW)
  field_simp
  linarith [hsq]

/-- `1 ≤ rstdRef · sqrt (varRef + epsR)` and similar; we record the product form. -/
theorem rstd_tol_R_sq_mul {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    rstdRef d init t ^ 2 * (varRef d init t + epsR d) = 1 := by
  have hW := rstd_tol_Ws_pos (init := init) (t := t) hd
  have hsq : Real.sqrt (varRef d init t + epsR d) ^ 2 = varRef d init t + epsR d :=
    Real.sq_sqrt (le_of_lt hW)
  have hs : Real.sqrt (varRef d init t + epsR d) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hW)
  unfold rstdRef
  rw [one_div]
  field_simp
  exact hsq.symm


/-- Minkowski for root-mean-squares of sums of nonnegative sequences. -/
theorem rstd_tol_sqrt_sum_add {n : ℕ} (hn : 0 < n) {a b : ℕ → ℝ}
    (ha : ∀ j, j < n → 0 ≤ a j) (hb : ∀ j, j < n → 0 ≤ b j) :
    Real.sqrt ((∑ j ∈ Finset.range n, (a j + b j) ^ 2) / n)
      ≤ Real.sqrt ((∑ j ∈ Finset.range n, (a j) ^ 2) / n)
        + Real.sqrt ((∑ j ∈ Finset.range n, (b j) ^ 2) / n) := by
  have hnR : (0:ℝ) < n := by exact_mod_cast hn
  set A := ∑ j ∈ Finset.range n, (a j) ^ 2 with hA
  set B := ∑ j ∈ Finset.range n, (b j) ^ 2 with hB
  have hA0 : 0 ≤ A := Finset.sum_nonneg fun j _ => sq_nonneg _
  have hB0 : 0 ≤ B := Finset.sum_nonneg fun j _ => sq_nonneg _
  set S := ∑ j ∈ Finset.range n, a j * b j with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun j hj =>
    mul_nonneg (ha j (Finset.mem_range.mp hj)) (hb j (Finset.mem_range.mp hj))
  have hAB : S ^ 2 ≤ A * B := by
    have := Finset.sum_mul_sq_le_sq_mul_sq (Finset.range n) a b
    simpa [hA, hB, hS] using this
  have hSle : S ≤ Real.sqrt A * Real.sqrt B := by
    rw [← Real.sqrt_mul hA0]
    exact Real.le_sqrt_of_sq_le hAB
  have hAn : (0:ℝ) ≤ A / n := by positivity
  have hBn : (0:ℝ) ≤ B / n := by positivity
  have hcross : 2 * Real.sqrt (A/n) * Real.sqrt (B/n) = 2 * (Real.sqrt A * Real.sqrt B) / n := by
    rw [Real.sqrt_div hA0, Real.sqrt_div hB0]
    have hsn : Real.sqrt (n:ℝ) ≠ 0 := by
      have : (0:ℝ) < n := hnR
      positivity
    field_simp
    rw [Real.sq_sqrt (le_of_lt hnR)]
    ring
  have hkey : (∑ j ∈ Finset.range n, (a j + b j) ^ 2) / n
      ≤ (Real.sqrt (A/n) + Real.sqrt (B/n)) ^ 2 := by
    have hexp : ∑ j ∈ Finset.range n, (a j + b j) ^ 2 = A + 2 * S + B := by
      rw [hA, hB, hS, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hexp, add_div, add_div, add_sq, Real.sq_sqrt hAn, Real.sq_sqrt hBn, hcross]
    have hSn : 2 * S / n ≤ 2 * (Real.sqrt A * Real.sqrt B) / n :=
      div_le_div_of_nonneg_right (by linarith) (le_of_lt hnR)
    linarith
  have hsq : Real.sqrt ((Real.sqrt (A/n) + Real.sqrt (B/n)) ^ 2)
      = Real.sqrt (A/n) + Real.sqrt (B/n) := Real.sqrt_sq (by positivity)
  have h2 := Real.sqrt_le_sqrt hkey
  rw [hsq] at h2
  exact h2

/-- `R` from `R²(V+eps) = 1` is `1/√(V+eps)`. -/
theorem rstd_tol_R_eq {V eps R : ℝ} (hVe : 0 < V + eps) (hR : 0 < R) (hRW : R ^ 2 * (V + eps) = 1) :
    R = 1 / Real.sqrt (V + eps) := by
  have hs : Real.sqrt (V + eps) ^ 2 = V + eps := Real.sq_sqrt (le_of_lt hVe)
  have hs0 : 0 < Real.sqrt (V + eps) := Real.sqrt_pos.2 hVe
  have h1 : R * Real.sqrt (V + eps) = 1 := by
    have h2 : (R * Real.sqrt (V + eps)) ^ 2 = 1 := by rw [mul_pow, hs, hRW]
    nlinarith [h2, mul_pos hR hs0]
  field_simp
  linarith [h1]

/-- Lower bound on `V0 + eps` from a bound on `|√V0 - √V|`. -/
theorem rstd_tol_V_lo {V0 V eps E R : ℝ} (hV0 : 0 ≤ V0) (hV : 0 ≤ V) (heps : 0 < eps)
    (hE : 0 ≤ E) (hR : 0 < R) (hRW : R ^ 2 * (V + eps) = 1)
    (h : |Real.sqrt V0 - Real.sqrt V| ≤ E) :
    (V + eps) - 2 * (E * R) * (V + eps) ≤ V0 + eps := by
  have hVe : 0 < V + eps := by linarith
  have hsV : 0 ≤ Real.sqrt V := Real.sqrt_nonneg V
  have hRV : R * (V + eps) = Real.sqrt (V + eps) := by
    rw [rstd_tol_R_eq hVe hR hRW]; field_simp
    rw [Real.sq_sqrt (le_of_lt hVe)]
  have hsqRV : Real.sqrt V ≤ Real.sqrt (V + eps) := Real.sqrt_le_sqrt (by linarith)
  have h1 : Real.sqrt V - E ≤ Real.sqrt V0 := by
    have := abs_le.mp h; linarith [this.1]
  have hsV2 : Real.sqrt V ^ 2 = V := Real.sq_sqrt hV
  have h4 : V - 2 * E * Real.sqrt V ≤ (Real.sqrt V - E) ^ 2 := by nlinarith [sq_nonneg E]
  have h5 : 2 * E * Real.sqrt V ≤ 2 * (E * R) * (V + eps) := by
    have h6 : Real.sqrt V ≤ R * (V + eps) := by rw [hRV]; exact hsqRV
    nlinarith [hE, h6, hV, heps]
  have h6 : V - 2 * E * Real.sqrt V ≤ V0 := by
    by_cases hc : 0 ≤ Real.sqrt V - E
    · have hs := Real.sq_sqrt hV0
      nlinarith [h1, hc, hs]
    · push Not at hc
      have h7 : Real.sqrt V * (Real.sqrt V - 2 * E) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos (Real.sqrt_nonneg V) (by linarith)
      nlinarith [h7]
  linarith

/-- Upper bound on `V0 + eps` from a bound on `|√V0 - √V|`. -/
theorem rstd_tol_V_hi {V0 V eps E R : ℝ} (hV0 : 0 ≤ V0) (hV : 0 ≤ V) (heps : 0 < eps)
    (hE : 0 ≤ E) (hR : 0 < R) (hRW : R ^ 2 * (V + eps) = 1)
    (h : |Real.sqrt V0 - Real.sqrt V| ≤ E) :
    V0 + eps ≤ (V + eps) * (1 + E * R) ^ 2 := by
  have hVe : 0 < V + eps := by linarith
  have hRV : R * (V + eps) = Real.sqrt (V + eps) := by
    rw [rstd_tol_R_eq hVe hR hRW]; field_simp
    rw [Real.sq_sqrt (le_of_lt hVe)]
  have hsqRV : Real.sqrt V ≤ Real.sqrt (V + eps) := Real.sqrt_le_sqrt (by linarith)
  have h1 : Real.sqrt V0 ≤ Real.sqrt V + E := by
    have := abs_le.mp h; linarith [this.2]
  have h3 : V0 ≤ (Real.sqrt V + E) ^ 2 := by
    have hs := Real.sq_sqrt hV0
    have h0 : 0 ≤ Real.sqrt V + E := by linarith [Real.sqrt_nonneg V]
    nlinarith [hs, h1, h0, Real.sqrt_nonneg V0]
  have h5 : 2 * E * Real.sqrt V ≤ 2 * (E * R) * (V + eps) := by
    have h6 : Real.sqrt V ≤ R * (V + eps) := by rw [hRV]; exact hsqRV
    nlinarith [hE, h6, hV, heps]
  have h6 : E ^ 2 ≤ (E * R) ^ 2 * (V + eps) := by
    have h7 : E ^ 2 * (R ^ 2 * (V + eps)) = E ^ 2 := by rw [hRW]; ring
    nlinarith [h7, sq_nonneg E, hE]
  have h8 : (Real.sqrt V + E) ^ 2 = V + 2 * E * Real.sqrt V + E ^ 2 := by
    rw [add_sq, Real.sq_sqrt hV]; ring
  rw [h8] at h3
  nlinarith [h3, h5, h6, hV, heps, hE]

/-- `1/√(1+δ) ≤ 1 - δ/2 + δ²` for `|δ| ≤ 1/2`. -/
theorem rstd_tol_inv_sqrt_upper {δ : ℝ} (h1 : -(1/2:ℝ) ≤ δ) (h2 : δ ≤ 1/2) :
    1 / Real.sqrt (1 + δ) ≤ 1 - δ/2 + δ^2 := by
  have hpos : 0 < 1 + δ := by linarith
  have hs0 : 0 < Real.sqrt (1+δ) := Real.sqrt_pos.2 hpos
  set C := 1 - δ/2 + δ^2 with hC
  have hC0 : 0 < C := by nlinarith [sq_nonneg δ]
  have hsq : Real.sqrt (1+δ)^2 = 1+δ := Real.sq_sqrt (le_of_lt hpos)
  have hC2 : 1 ≤ C^2 * (1+δ) := by
    rw [hC]; nlinarith [sq_nonneg δ, sq_nonneg (δ + δ^2), sq_nonneg (δ^2 - δ/2),
      mul_nonneg (sq_nonneg δ) (by linarith : (0:ℝ) ≤ 1/2 + δ)]
  have hle : 1/C ≤ Real.sqrt (1+δ) := by
    apply Real.le_sqrt_of_sq_le
    rw [div_pow, div_le_iff₀ (by positivity : (0:ℝ) < C^2)]
    nlinarith [hC2]
  rw [div_le_iff₀ hs0]
  have hmul : C * (1/C) ≤ C * Real.sqrt (1+δ) := mul_le_mul_of_nonneg_left hle (le_of_lt hC0)
  rw [mul_one_div, div_self (ne_of_gt hC0)] at hmul
  linarith

/-- `1 - δ/2 - δ² ≤ 1/√(1+δ)` for `|δ| ≤ 1/2`. -/
theorem rstd_tol_inv_sqrt_lower {δ : ℝ} (h1 : -(1/2:ℝ) ≤ δ) (h2 : δ ≤ 1/2) :
    1 - δ/2 - δ^2 ≤ 1 / Real.sqrt (1 + δ) := by
  have hpos : 0 < 1 + δ := by linarith
  have hs0 : 0 < Real.sqrt (1+δ) := Real.sqrt_pos.2 hpos
  set X := 1 - δ/2 - δ^2 with hX
  have hX0 : 0 < X := by nlinarith [sq_nonneg δ]
  have hsq : Real.sqrt (1+δ)^2 = 1+δ := Real.sq_sqrt (le_of_lt hpos)
  have hX2 : X^2 * (1+δ) ≤ 1 := by
    rw [hX]; nlinarith [sq_nonneg δ, sq_nonneg (δ^2 + δ/2), mul_nonneg (sq_nonneg δ) (by linarith : (0:ℝ) ≤ 1/2 - δ)]
  rw [le_div_iff₀ hs0]
  have hsq2 : (X * Real.sqrt (1+δ))^2 ≤ 1 := by
    rw [mul_pow, hsq]; linarith
  have hnn : 0 ≤ X * Real.sqrt (1+δ) := mul_nonneg (le_of_lt hX0) (le_of_lt hs0)
  nlinarith [hsq2, hnn]

/-- The chain from the float variance, `eps`, `sqrt` and the final division to `R`. -/
theorem rstd_tol_chain {W be bsq brstd R Th : ℝ} {u : ℝ}
    (hW : 0 < W) (hR : 0 < R) (hRWs : R ^ 2 * W = 1)
    (hu : 0 < u) (hu1 : 1000 * u ≤ 1)
    (hTh0 : 0 ≤ Th) (hTh : Th ≤ 1/2)
    (hbe : |be - W| ≤ Th * W)
    (hbsq : |bsq - Real.sqrt be| ≤ 4 * u * Real.sqrt be + (2:ℝ)^(-149:ℤ))
    (hbrstd : |brstd - 1/bsq| ≤ 4 * u * |1/bsq| + (2:ℝ)^(-148:ℤ))
    (hj1 : (2:ℝ)^(-149:ℤ) ≤ u * Real.sqrt be)
    (hj2 : (2:ℝ)^(-148:ℤ) ≤ u * R) :
    |brstd - R| ≤ R * (Th / 2 + Th ^ 2 + 22 * u) := by
  have hW0 : 0 < Real.sqrt W := Real.sqrt_pos.2 hW
  have hRW : R * Real.sqrt W = 1 := by
    have h2 : (R * Real.sqrt W) ^ 2 = 1 := by
      rw [mul_pow, Real.sq_sqrt (le_of_lt hW), hRWs]
    nlinarith [h2, mul_pos hR hW0]
  have hRinv : R = 1 / Real.sqrt W := by
    field_simp
    linarith [hRW]
  set δ := (be - W) / W with hδdef
  have hδ : |δ| ≤ Th := by
    rw [hδdef, abs_div, abs_of_pos hW, div_le_iff₀ hW]
    have := abs_le.mp hbe
    linarith [this.1, this.2]
  have hδm := abs_le.mp hδ
  have hδlo : -(1/2:ℝ) ≤ δ := by linarith [hδm.1, hTh]
  have hδhi : δ ≤ 1/2 := by linarith [hδm.2, hTh]
  have hδabs : |δ| ≤ Th := hδ
  have h1δ : 0 < 1 + δ := by linarith
  have hs1δ : 0 < Real.sqrt (1 + δ) := Real.sqrt_pos.2 h1δ
  have hbe_eq : be = W * (1 + δ) := by
    rw [hδdef]; field_simp; ring
  have hbe_pos : 0 < be := by rw [hbe_eq]; positivity
  have hsbe_pos : 0 < Real.sqrt be := Real.sqrt_pos.2 hbe_pos
  have hsbe : Real.sqrt be = Real.sqrt W * Real.sqrt (1+δ) := by
    rw [hbe_eq, Real.sqrt_mul (le_of_lt hW)]
  have hinv_be : 1 / Real.sqrt be = R * (1 / Real.sqrt (1+δ)) := by
    rw [hsbe, hRinv]
    field_simp
  have hU := rstd_tol_inv_sqrt_upper hδlo hδhi
  have hL := rstd_tol_inv_sqrt_lower hδlo hδhi
  -- reciprocal bound
  have hrec : |1 / Real.sqrt (1+δ) - 1| ≤ Th/2 + Th^2 := by
    rw [abs_le]
    constructor
    · nlinarith [hL, hδm.1, hδm.2]
    · nlinarith [hU, hδm.1, hδm.2]
  have hkey1 : |1 / Real.sqrt be - R| ≤ R * (Th/2 + Th^2) := by
    rw [hinv_be, show R * (1 / Real.sqrt (1 + δ)) - R = R * (1 / Real.sqrt (1 + δ) - 1) by ring,
      abs_mul, abs_of_pos hR]
    exact mul_le_mul_of_nonneg_left hrec (le_of_lt hR)
  have hinv_be_le : 1 / Real.sqrt be ≤ R * (1 + Th/2 + Th^2) := by
    have := abs_le.mp hkey1
    have h2 : 1 / Real.sqrt be - R ≤ R * (Th/2 + Th^2) := this.2
    nlinarith [h2]
  -- the square root's rounding
  have hnum : |Real.sqrt be - bsq| ≤ 4 * u * Real.sqrt be + (2:ℝ)^(-149:ℤ) := by
    rw [abs_sub_comm]; exact hbsq
  have heps1 : |bsq / Real.sqrt be - 1| ≤ 5 * u := by
    have h1 : bsq / Real.sqrt be - 1 = (bsq - Real.sqrt be) / Real.sqrt be := by
      field_simp
    rw [h1, abs_div, abs_of_pos hsbe_pos, div_le_iff₀ hsbe_pos]
    nlinarith [hnum, hj1]
  have hr1pos : 0 < bsq / Real.sqrt be := by
    have h := abs_le.mp heps1
    have hu5 : 5 * u ≤ 1/2 := by linarith
    linarith [h.1, hu5]
  have hbsq_pos : 0 < bsq := by
    have h := mul_pos hr1pos hsbe_pos
    have h2 : bsq / Real.sqrt be * Real.sqrt be = bsq := div_mul_cancel₀ bsq (ne_of_gt hsbe_pos)
    rwa [h2] at h
  have hA0 : (0:ℝ) ≤ Th/2 + Th^2 := by positivity
  have hA_le : Th/2 + Th^2 ≤ 1/2 := by nlinarith [hTh, hTh0]
  have hstep1 : |1/bsq - 1/Real.sqrt be| ≤ 6 * u * (1/Real.sqrt be) := by
    have hone : 1/bsq = (1/Real.sqrt be) / (bsq/Real.sqrt be) := by
      field_simp
    rw [hone, show (1/Real.sqrt be)/(bsq/Real.sqrt be) - 1/Real.sqrt be
        = (1/Real.sqrt be) * (1/(bsq/Real.sqrt be) - 1) by ring]
    rw [abs_mul, abs_of_pos (by positivity : (0:ℝ) < 1/Real.sqrt be)]
    have hr1 : 0 < bsq/Real.sqrt be := hr1pos
    have hb : |1/(bsq/Real.sqrt be) - 1| ≤ 6*u := by
      have h1 : 1/(bsq/Real.sqrt be) - 1 = (1 - bsq/Real.sqrt be)/(bsq/Real.sqrt be) := by
        field_simp
      rw [h1, abs_div, abs_of_pos hr1, div_le_iff₀ hr1]
      have h2 : |1 - bsq/Real.sqrt be| = |bsq/Real.sqrt be - 1| := abs_sub_comm _ _
      rw [h2]
      have h4 : 1 - 5*u ≤ bsq/Real.sqrt be := by
        have := abs_le.mp heps1; linarith [this.1]
      have hm : 6*u*(1-5*u) ≤ 6*u*(bsq/Real.sqrt be) :=
        mul_le_mul_of_nonneg_left h4 (by positivity)
      nlinarith [heps1, hm, hu, hu1]
    have hbe1 := mul_le_mul_of_nonneg_left hb (by positivity : (0:ℝ) ≤ 1/Real.sqrt be)
    linarith [hbe1]
  have hone_over_be : 1/Real.sqrt be ≤ R * (1 + (Th/2 + Th^2)) := by
    have h := abs_le.mp hkey1
    nlinarith [h.2]
  have hstep1' : |1/bsq - 1/Real.sqrt be| ≤ 11 * u * R := by
    have h3 : 6 * u * (1/Real.sqrt be) ≤ 6 * u * (R * (1 + (Th/2 + Th^2))) :=
      mul_le_mul_of_nonneg_left hone_over_be (by positivity)
    have hprod : u * R * (Th/2 + Th^2) ≤ u * R * (1/2) :=
      mul_le_mul_of_nonneg_left hA_le (by positivity)
    have h4 : 6 * u * (R * (1 + (Th/2 + Th^2))) ≤ 11 * u * R := by
      nlinarith [hprod, hR, hu]
    linarith [hstep1, h3, h4]
  have hinv_bsq_le : |1/bsq| ≤ 2 * R := by
    have h1 : |1/bsq| ≤ 1/Real.sqrt be + |1/bsq - 1/Real.sqrt be| := by
      have h3 := abs_add_le (1/Real.sqrt be) (1/bsq - 1/Real.sqrt be)
      have e : 1/Real.sqrt be + (1/bsq - 1/Real.sqrt be) = 1/bsq := by ring
      rw [e] at h3
      rwa [abs_of_pos (by positivity : (0:ℝ) < 1/Real.sqrt be)] at h3
    have h2 : 1/Real.sqrt be ≤ 2*R := by nlinarith [hone_over_be, hA_le, hR]
    nlinarith [h1, h2, hstep1', hu, hR]
  have hfinal : |brstd - R| ≤ 4 * u * |1/bsq| + (2:ℝ)^(-148:ℤ) + |1/bsq - R| := by
    have h2 : |brstd - R| ≤ |brstd - 1/bsq| + |1/bsq - R| := by
      have h3 := abs_add_le (brstd - 1/bsq) (1/bsq - R)
      have e : brstd - 1/bsq + (1/bsq - R) = brstd - R := by ring
      rwa [e] at h3
    linarith [h2]
  have hsum : |1/bsq - R| ≤ R*(Th/2 + Th^2) + 11*u*R := by
    have h3 : |1/bsq - R| ≤ |1/bsq - 1/Real.sqrt be| + |1/Real.sqrt be - R| := by
      have h4 := abs_add_le (1/bsq - 1/Real.sqrt be) (1/Real.sqrt be - R)
      have e : 1/bsq - 1/Real.sqrt be + (1/Real.sqrt be - R) = 1/bsq - R := by ring
      rwa [e] at h4
    linarith [h3, hstep1', hkey1]
  have h4u : 4 * u * |1/bsq| ≤ 8 * u * R := by
    have h := mul_le_mul_of_nonneg_left hinv_bsq_le (by positivity : (0:ℝ) ≤ 4*u)
    linarith [h]
  set a := |1/bsq| with ha
  set c := |1/bsq - R| with hc
  set d := |brstd - R| with hd
  have h4u' : 4*u*a ≤ 8*u*R := h4u
  have hsum' : c ≤ R*(Th/2 + Th^2) + 11*u*R := hsum
  have hfinal' : d ≤ 4*u*a + (2:ℝ)^(-148:ℤ) + c := hfinal
  have hf2 : d ≤ 4*u*a + u*R + c := by linarith [hfinal', hj2]
  have hbnd : 4*u*a + u*R + c ≤ R*(Th/2 + Th^2 + 22*u) := by
    have hR0 : 0 < u * R := mul_pos hu hR
    have h19 : R*(Th/2 + Th^2) + 20*u*R ≤ R*(Th/2 + Th^2 + 22*u) := by
      have e : R*(Th/2 + Th^2 + 22*u) = R*(Th/2 + Th^2) + 22*(u*R) := by ring
      rw [e]
      nlinarith [hR0]
    linarith [h4u', hsum', h19]
  exact le_trans hf2 hbnd

/-- One rounded square, for lanes of magnitude at most `2^58 + 2^40`. -/
theorem rstd_tol_sq_bd {x sq : F32} (hxfin : x.isFinite = true)
    (hxbd : |x.toReal| ≤ 2 ^ 58 + 2 ^ 40)
    (hsq : FloatV.NaNEq (FloatV.mul x x) sq) :
    sq.isFinite = true ∧ |sq.toReal - x.toReal ^ 2| ≤ u32 * x.toReal ^ 2 + eta32 := by
  have hf : Format.binary32.Regular := Format.binary32_regular
  have hxrat : |x.toRat| ≤ ((2 : ℚ) ^ 58 + 2 ^ 40) := by
    have h1 : |(x.toRat : ℝ)| ≤ ((2 : ℝ) ^ 58 + 2 ^ 40) := by
      simpa [FloatV.toReal] using hxbd
    exact_mod_cast h1
  have hmax : |x.toRat * x.toRat| ≤ (Format.binary32.maxVal : ℚ) := by
    rw [abs_mul]
    have h2 : |x.toRat| * |x.toRat| ≤ ((2 : ℚ) ^ 58 + 2 ^ 40) * ((2 : ℚ) ^ 58 + 2 ^ 40) :=
      mul_le_mul hxrat hxrat (abs_nonneg _) (by positivity)
    have h4 : ((2 : ℚ) ^ 58 + 2 ^ 40) * ((2 : ℚ) ^ 58 + 2 ^ 40)
        ≤ (Format.binary32.maxVal : ℚ) := by
      rw [Format.binary32_maxVal]; norm_num
    linarith
  obtain ⟨hmulfin, hmulerr⟩ := FloatV.mul_err hf hxfin hxfin hmax
  have hmulnan : (FloatV.mul x x).isNaN = false := (FloatV.isFinite_iff.mp hmulfin).1
  have hq : sq = FloatV.mul x x := hsq.eq hmulnan
  refine ⟨?_, ?_⟩
  · rw [hq]; exact hmulfin
  · rw [hq]
    have hcast : |((FloatV.mul x x).toRat : ℝ) - ((x.toRat * x.toRat : ℚ) : ℝ)|
        ≤ (((|x.toRat * x.toRat| * Format.binary32.u + Format.binary32.eta : ℚ)) : ℝ) := by
      exact_mod_cast hmulerr
    push_cast at hcast
    rw [rstd_tol_u_cast, rstd_tol_eta_cast] at hcast
    have h1 : |((FloatV.mul x x).toRat : ℝ) - (x.toReal * x.toReal)|
        ≤ |x.toReal * x.toReal| * u32 + eta32 := by
      simpa [FloatV.toReal] using hcast
    have h3 : |x.toReal * x.toReal| = x.toReal ^ 2 := by
      rw [abs_mul, abs_mul_abs_self, pow_two]
    rw [h3] at h1
    simpa [FloatV.toReal, pow_two, mul_comm] using h1

/-- Every masked or unmasked lane of `x` is finite and at most `2^58 + 2^40`. -/
theorem rstd_tol_x_bd {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    {bx : Nat → F32} {bmean : F32} {x : Nat → F32}
    (hmeanfin : bmean.isFinite = true)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hfinbx : ∀ j, j < d.D → (bx j).isFinite = true)
    (hval : ∀ j, j < d.D → (bx j).toReal = sR d init t j)
    (hbdd : ∀ j, j < d.D → |sR d init t j| ≤ 2 ^ 57)
    (hfinx : ∀ j, j < d.BD → (x j).isFinite = true)
    (hdev : ∀ j, j < d.D → AddRel true (bx j) (FloatV.neg bmean) (x j))
    (hzero : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0) :
    ∀ j, j < d.BD → (x j).isFinite = true ∧ |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40 := by
  intro j hj
  by_cases hjD : j < d.D
  · have hA := rstd_tol_absMean_le hbdd hd.d_pos
    have hM := rstd_tol_meanTol_le hd hbdd
    have hA0 : 0 ≤ absMean d init t := rstd_tol_absMean_nonneg
    have hM0 : 0 ≤ meanTol d init t := rstd_tol_meanTol_nonneg
    have hmeanRef := abs_meanRef_le_absMean (d := d) (init := init) (t := t) hd.d_pos
    obtain ⟨hfin, c, e, hc, he, hxeq⟩ :=
      rstd_tol_dev hjD (hfinbx j hjD) hmeanfin (hval j hjD) hmean (hbdd j hjD) hA hM hmeanRef
        (hdev j hjD)
    refine ⟨hfin, ?_⟩
    have hd_bd : |sR d init t j - meanRef d init t| ≤ 2 ^ 58 := by
      have h1 : |sR d init t j - meanRef d init t|
          ≤ |sR d init t j| + |meanRef d init t| := by
        simpa [sub_eq_add_neg] using abs_add_le (sR d init t j) (-(meanRef d init t))
      have h2 := hbdd j hjD
      have h3 : |meanRef d init t| ≤ absMean d init t := hmeanRef
      linarith
    have he_bd : |e| ≤ 2 ^ 39 := by
      have hs := hbdd j hjD
      have hu : 0 < u32 := u32_pos
      have h1 : u32 * (|sR d init t j| + absMean d init t + meanTol d init t)
          ≤ u32 * (2 ^ 57 + 2 ^ 57 + 2 ^ 38) := by
        apply mul_le_mul_of_nonneg_left _ (le_of_lt hu)
        linarith
      have h2 : u32 * (2 ^ 57 + 2 ^ 57 + 2 ^ 38) + 4 * eta32 ≤ 2 ^ 39 - 2 ^ 38 := by
        unfold u32 eta32; norm_num
      linarith [he]
    have hb1 : 0 ≤ |sR d init t j - meanRef d init t| + |e| := by positivity
    have hc1 : |1 + c| ≤ 1 + 2 * u32 := by
      have h := abs_add_le (1 : ℝ) c
      have h1 : |(1 : ℝ)| = 1 := abs_one
      have h2 : |c| ≤ 2 * u32 := hc
      linarith
    calc |(x j).toReal| = |(sR d init t j - meanRef d init t + e) * (1 + c)| := by rw [hxeq]
      _ = |sR d init t j - meanRef d init t + e| * |1 + c| := abs_mul _ _
      _ ≤ (|sR d init t j - meanRef d init t| + |e|) * (1 + 2 * u32) := by
          apply mul_le_mul (abs_add_le _ _) hc1 (abs_nonneg _) hb1
      _ ≤ (2 ^ 58 + 2 ^ 39) * (1 + 2 * u32) := by
          apply mul_le_mul_of_nonneg_right _ (by have hp := u32_pos; linarith)
          linarith
      _ ≤ 2 ^ 58 + 2 ^ 40 := by unfold u32; norm_num
  · have hle : d.D ≤ j := Nat.le_of_not_lt hjD
    refine ⟨hfinx j hj, ?_⟩
    rw [hzero j hj hle]
    norm_num

/-- Sum of the rounded squares over the padded block against the true sum of squares. -/
theorem rstd_tol_sq_sum {d : Inst} (hd : d.ok) {_init : Mem} {_t : Nat}
    {x sq : Nat → F32}
    (hx : ∀ j, j < d.BD → (x j).isFinite = true ∧ |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40)
    (hzero : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0)
    (hsq : ∀ j, j < d.BD → FloatV.NaNEq (FloatV.mul (x j) (x j)) (sq j))
    (_hfsq : ∀ j, j < d.BD → (sq j).isFinite = true) :
    |(∑ j ∈ Finset.range d.BD, (sq j).toReal) - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2)|
      ≤ u32 * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) + (d.BD : ℝ) * eta32 := by
  have hDB : d.D ≤ d.BD := hd.d_le_bd
  have hBD : d.BD = d.D + (d.BD - d.D) := (Nat.add_sub_of_le hDB).symm
  have hsplit : (∑ j ∈ Finset.range d.BD, (sq j).toReal)
      = (∑ j ∈ Finset.range d.D, (sq j).toReal)
        + (∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal) := by
    conv_lhs => rw [hBD]
    rw [Finset.sum_range_add]
    simp only [Nat.add_comm]
  have h1 : |(∑ j ∈ Finset.range d.D, (sq j).toReal)
      - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2)|
      ≤ u32 * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) + (d.D : ℝ) * eta32 := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ j ∈ Finset.range d.D, ((sq j).toReal - (x j).toReal ^ 2)|
        ≤ ∑ j ∈ Finset.range d.D, |(sq j).toReal - (x j).toReal ^ 2| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ Finset.range d.D, (u32 * (x j).toReal ^ 2 + eta32) := by
          apply Finset.sum_le_sum
          intro j hj
          have hjbd : j < d.BD := lt_of_lt_of_le (Finset.mem_range.mp hj) hDB
          exact (rstd_tol_sq_bd (hx j hjbd).1 (hx j hjbd).2 (hsq j hjbd)).2
      _ = u32 * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) + (d.D : ℝ) * eta32 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
            nsmul_eq_mul]
  have h2 : |∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal|
      ≤ ((d.BD - d.D : ℕ) : ℝ) * eta32 := by
    calc |∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal|
        ≤ ∑ i ∈ Finset.range (d.BD - d.D), |(sq (i + d.D)).toReal| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i ∈ Finset.range (d.BD - d.D), eta32 := by
          apply Finset.sum_le_sum
          intro i hi
          have hi' := Finset.mem_range.mp hi
          have hj : i + d.D < d.BD := by omega
          have hle : d.D ≤ i + d.D := Nat.le_add_left _ _
          have hx0 : (x (i + d.D)).toReal = 0 := hzero _ hj hle
          have := (rstd_tol_sq_zero (hx _ hj).1 hx0 (hsq _ hj)).2
          exact this
      _ = ((d.BD - d.D : ℕ) : ℝ) * eta32 := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hcast : ((d.BD - d.D : ℕ) : ℝ) + (d.D : ℝ) = (d.BD : ℝ) := by
    rw [Nat.cast_sub hDB]; ring
  have hsum : |(∑ j ∈ Finset.range d.D, (sq j).toReal)
      + (∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal)
      - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2)|
      ≤ (u32 * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) + (d.D : ℝ) * eta32)
        + ((d.BD - d.D : ℕ) : ℝ) * eta32 := by
    have he : (∑ j ∈ Finset.range d.D, (sq j).toReal)
        + (∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal)
        - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2)
        = ((∑ j ∈ Finset.range d.D, (sq j).toReal)
            - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2))
          + (∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal) := by ring
    rw [he]
    calc |((∑ j ∈ Finset.range d.D, (sq j).toReal)
          - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2))
          + (∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal)|
        ≤ |(∑ j ∈ Finset.range d.D, (sq j).toReal)
            - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2)|
          + |∑ i ∈ Finset.range (d.BD - d.D), (sq (i + d.D)).toReal| := abs_add_le _ _
      _ ≤ _ := by linarith [h1, h2]
  rw [hsplit]
  calc _ ≤ (u32 * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) + (d.D : ℝ) * eta32)
        + ((d.BD - d.D : ℕ) : ℝ) * eta32 := hsum
    _ = u32 * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) + (d.BD : ℝ) * eta32 := by
        rw [← hcast]; ring

/-- The sum of the true squares of the unmasked lanes is nonnegative and at most `D·2^117`,
hence at most `2^126`. -/
theorem rstd_tol_sum_x_sq_bd {d : Inst} (_hd : d.ok) {x : Nat → F32}
    (hxbd : ∀ j, j < d.D → |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40) :
    0 ≤ ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2 ∧
    ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2 ≤ (d.D : ℝ) * 2 ^ 117 := by
  have hnn : 0 ≤ ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  refine ⟨hnn, ?_⟩
  have hterm : ∀ j ∈ Finset.range d.D, (x j).toReal ^ 2 ≤ 2 ^ 117 := by
    intro j hj
    have h := hxbd j (Finset.mem_range.mp hj)
    have h2 : (x j).toReal ^ 2 ≤ (2 ^ 58 + 2 ^ 40) ^ 2 := by
      rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) h 2
    have h3 : ((2:ℝ) ^ 58 + 2 ^ 40) ^ 2 ≤ 2 ^ 117 := by norm_num
    linarith
  calc ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2 ≤ ∑ j ∈ Finset.range d.D, (2:ℝ) ^ 117 :=
        Finset.sum_le_sum hterm
    _ = (d.D : ℝ) * 2 ^ 117 := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]

/-- `(1+x)^n - 1 ≤ n·x·(1+x)^n` for `x ≥ 0`. -/
theorem rstd_tol_pow_sub_le {x : ℝ} (hx : 0 ≤ x) (n : ℕ) :
    (1 + x) ^ n - 1 ≤ (n : ℝ) * x * (1 + x) ^ n := by
  have h := geom_sum_mul_add x n
  have h1 : (∑ i ∈ Finset.range n, (1 + x) ^ i) * x + 1 = (1 + x) ^ n := by
    simpa only [add_comm x 1] using h
  have hsum : (∑ i ∈ Finset.range n, (1 + x) ^ i) * x = (1 + x) ^ n - 1 := by linarith
  rw [← hsum]
  have hx1 : 1 ≤ 1 + x := by linarith
  have hle : ∑ i ∈ Finset.range n, (1 + x) ^ i ≤ n * (1 + x) ^ n := by
    calc ∑ i ∈ Finset.range n, (1 + x) ^ i ≤ ∑ _i ∈ Finset.range n, (1 + x) ^ n := by
          apply Finset.sum_le_sum
          intro i hi
          exact pow_le_pow_right₀ hx1 (Nat.le_of_lt (Finset.mem_range.mp hi))
      _ = n * (1 + x) ^ n := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  calc (∑ i ∈ Finset.range n, (1 + x) ^ i) * x ≤ (n * (1 + x) ^ n) * x :=
        mul_le_mul_of_nonneg_right hle hx
    _ = (n : ℝ) * x * (1 + x) ^ n := by ring

/-- `BD`, a power of two below `2·D ≤ 1024`, is at most `512`. -/
theorem rstd_tol_BD_le {d : Inst} (hd : d.ok) : d.BD ≤ 512 := by
  obtain ⟨e, he⟩ := hd.bd_pow2
  have h2 : 2 ^ e < 2 ^ 10 := by
    have h3 : 2 * d.D ≤ 2 * 512 := by have := hd.d_le; omega
    calc 2 ^ e = d.BD := he.symm
      _ < 2 * d.D := hd.bd_lt
      _ ≤ 2 * 512 := h3
      _ = 2 ^ 10 := by norm_num
  have he10 : e < 10 := (Nat.pow_lt_pow_iff_right (by norm_num)).mp h2
  calc d.BD = 2 ^ e := he
    _ ≤ 2 ^ 9 := Nat.pow_le_pow_right (by norm_num) (by omega)
    _ = 512 := by norm_num

/-- The float sum of the rounded squares, against the true sum of squares. -/
theorem rstd_tol_bvar {d : Inst} (hd : d.ok) {_init : Mem} {_t : Nat}
    {x sq : Nat → F32} {bvar : F32} {tr : Tree}
    (hx : ∀ j, j < d.BD → (x j).isFinite = true ∧ |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40)
    (hzero : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0)
    (hfsq : ∀ j, j < d.BD → (sq j).isFinite = true)
    (hsq : ∀ j, j < d.BD → FloatV.NaNEq (FloatV.mul (x j) (x j)) (sq j))
    (hcov : Tree.Covers d.BD tr) (hdepth : tr.depth ≤ depth d)
    (heval : Tree.Eval (AddRel true) sq tr bvar) :
    bvar.isFinite = true ∧
    |bvar.toReal - (∑ j ∈ Finset.range d.BD, (sq j).toReal)|
      ≤ 3.1 * u32 * (depth d : ℝ) * (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2)
        + (2 : ℝ) ^ (-138 : ℤ) := by
  have hf : Format.binary32.Regular := Format.binary32_regular
  have hspec : Format.binary32.specials ≠ .finite := by decide
  have hu16 : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  set XR : ℝ := ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2 with hXR
  have hX0 : 0 ≤ XR := by rw [hXR]; exact Finset.sum_nonneg fun j _ => sq_nonneg _
  have hXbd : XR ≤ (d.D : ℝ) * 2 ^ 117 := by
    rw [hXR]
    exact (rstd_tol_sum_x_sq_bd hd
      (fun j hj => (hx j (lt_of_lt_of_le hj hd.d_le_bd)).2)).2
  have hXbd' : XR ≤ 2 ^ 126 := by
    have h512 : (d.D : ℝ) ≤ 512 := by exact_mod_cast hd.d_le
    nlinarith [hXbd]
  -- per-lane bound on the rounded square
  have hlane : ∀ j, j < d.BD → |(sq j).toReal| ≤ (1 + u32) * (x j).toReal ^ 2 + eta32 := by
    intro j hj
    by_cases hjD : j < d.D
    · have h := (rstd_tol_sq_bd (hx j hj).1 (hx j hj).2 (hsq j hj)).2
      have hx2 : 0 ≤ (x j).toReal ^ 2 := sq_nonneg _
      have h2 : |(sq j).toReal| ≤ |(sq j).toReal - (x j).toReal ^ 2| + |(x j).toReal ^ 2| := by
        have h3 := abs_add_le ((sq j).toReal - (x j).toReal ^ 2) ((x j).toReal ^ 2)
        have e : (sq j).toReal - (x j).toReal ^ 2 + (x j).toReal ^ 2 = (sq j).toReal := by ring
        rwa [e] at h3
      rw [abs_of_nonneg hx2] at h2
      nlinarith [h2, h]
    · have hle : d.D ≤ j := Nat.le_of_not_lt hjD
      have h := (rstd_tol_sq_zero (hx j hj).1 (hzero j hj hle) (hsq j hj)).2
      have hx0 : (x j).toReal = 0 := hzero j hj hle
      rw [hx0]; simpa using h
  -- the sum of the true squares over the padded block is `XR`
  have hXs : ∑ j ∈ Finset.range d.BD, (x j).toReal ^ 2 = XR := by
    rw [hXR]
    have hDB : d.D ≤ d.BD := hd.d_le_bd
    conv_lhs => rw [show d.BD = d.D + (d.BD - d.D) from (Nat.add_sub_of_le hDB).symm]
    rw [Finset.sum_range_add]
    have hz : ∀ i ∈ Finset.range (d.BD - d.D), (x (d.D + i)).toReal ^ 2 = 0 := by
      intro i hi
      have hj : d.D + i < d.BD := by
        have := Finset.mem_range.mp hi; omega
      rw [hzero _ hj (Nat.le_add_right _ _)]; ring
    rw [Finset.sum_eq_zero hz, add_zero]
  -- the absolute sum `AC` of the tree's leaves
  set AC : ℚ := ((List.range d.BD).map fun i => |(sq i).toRat|).sum with hAC
  have hACR : ((AC : ℚ) : ℝ) = ∑ j ∈ Finset.range d.BD, |(sq j).toReal| := by
    rw [hAC, rstd_tol_list_sum_range]
    push_cast
    rfl
  have hAq : ((AC : ℚ) : ℝ) ≤ (1 + u32) * XR + (d.BD : ℝ) * eta32 := by
    rw [hACR]
    have h1 : ∑ j ∈ Finset.range d.BD, |(sq j).toReal|
        ≤ ∑ j ∈ Finset.range d.BD, ((1 + u32) * (x j).toReal ^ 2 + eta32) :=
      Finset.sum_le_sum fun j hj => hlane j (Finset.mem_range.mp hj)
    have h2 : ∑ j ∈ Finset.range d.BD, ((1 + u32) * (x j).toReal ^ 2 + eta32)
        = (1 + u32) * XR + (d.BD : ℝ) * eta32 := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, hXs, Finset.sum_const, Finset.card_range,
        nsmul_eq_mul]
    linarith
  have hAqQ : AC ≤ (2 : ℚ) ^ 126 + 2 ^ 103 := by
    have h1 : ((AC : ℚ) : ℝ) ≤ (2 : ℝ) ^ 126 + 2 ^ 103 := by
      have h2 : (1 + u32) * XR ≤ (1 + u32) * 2 ^ 126 :=
        mul_le_mul_of_nonneg_left hXbd' (by have hp := u32_pos; linarith)
      have h3 : (d.BD : ℝ) * eta32 ≤ 512 * eta32 := by
        have hb : (d.BD : ℝ) ≤ 512 := by exact_mod_cast rstd_tol_BD_le hd
        nlinarith [eta32_pos]
      have h4 : (1 + u32) * 2 ^ 126 + 512 * eta32 ≤ (2 : ℝ) ^ 126 + 2 ^ 103 := by
        unfold u32 eta32; norm_num
      nlinarith [hAq, h2, h3, h4]
    exact_mod_cast h1
  have hAC0 : (0 : ℚ) ≤ AC := by
    rw [hAC]
    exact List.sum_nonneg (by
      intro z hz
      obtain ⟨i, -, rfl⟩ := List.mem_map.mp hz
      exact abs_nonneg _)
  -- room for the tree reduction
  have hδ0 : (0 : ℚ) ≤ 3 * Format.binary32.u := by
    rw [Format.binary32_u]; norm_num
  have hη0 : (0 : ℚ) ≤ 4 * Format.binary32.eta := by
    rw [Format.binary32_eta]; norm_num
  have hM : AC + Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (depth d)
      (d.BD - 1) AC ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
    have hBD1 : d.BD - 1 ≤ 511 := by have := rstd_tol_BD_le hd; omega
    have hmono := Tree.errBound_mono hδ0 hη0 (rstd_tol_depth_le hd) hBD1 hAC0 hAqQ
    have hstep : AC + Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (depth d)
        (d.BD - 1) AC
        ≤ ((2 : ℚ) ^ 126 + 2 ^ 103) +
          Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 10 511
            ((2 : ℚ) ^ 126 + 2 ^ 103) := by linarith [hmono, hAqQ]
    have hnum : ((2 : ℚ) ^ 126 + 2 ^ 103) +
        Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 10 511
          ((2 : ℚ) ^ 126 + 2 ^ 103)
        ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
      unfold Tree.errBound
      rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
      norm_num
    exact le_trans hstep hnum
  -- apply the tree reduction bound
  obtain ⟨hfin, herr⟩ := TVA.sumc_err_depth hf hspec hu16 true sq hcov hdepth heval hfsq hM
  refine ⟨hfin, ?_⟩
  have hcast : ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (depth d)
      (d.BD - 1) AC : ℚ) : ℝ)
      = ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
        + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d := by
    unfold Tree.errBound
    push_cast
    rw [rstd_tol_u_cast, rstd_tol_eta_cast]
  have herrQ : |bvar.toRat - (∑ j ∈ Finset.range d.BD, (sq j).toRat)|
      ≤ Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (depth d)
          (d.BD - 1) AC := by
    simpa only [← hAC, rstd_tol_list_sum_range] using herr
  have herrR : |bvar.toReal - (∑ j ∈ Finset.range d.BD, (sq j).toReal)|
      ≤ ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
        + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d := by
    have h4 : ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (depth d)
        (d.BD - 1) AC : ℚ) : ℝ)
        = ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
          + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d := hcast
    have h3 : ((|bvar.toRat - (∑ j ∈ Finset.range d.BD, (sq j).toRat)| : ℚ) : ℝ)
        ≤ ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
          + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d := by
      have h5 : ((|bvar.toRat - (∑ j ∈ Finset.range d.BD, (sq j).toRat)| : ℚ) : ℝ)
          ≤ ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (depth d)
              (d.BD - 1) AC : ℚ) : ℝ) := by
        exact_mod_cast herrQ
      rw [h4] at h5
      exact h5
    have h6 : ((|bvar.toRat - (∑ j ∈ Finset.range d.BD, (sq j).toRat)| : ℚ) : ℝ)
        = |bvar.toReal - (∑ j ∈ Finset.range d.BD, (sq j).toReal)| := by
      rw [Rat.cast_abs]
      congr 1
      push_cast
      rfl
    rwa [h6] at h3
  refine herrR.trans ?_
  -- final numeric bound on the tree error
  have hA0' : (0:ℝ) ≤ (depth d : ℝ) := Nat.cast_nonneg _
  have hu'' : (0:ℝ) ≤ u32 := le_of_lt u32_pos
  have hη0' : (0:ℝ) ≤ eta32 := le_of_lt eta32_pos
  have hb1 : (1:ℝ) ≤ 1 + 3 * u32 := by nlinarith
  have hq : (1 + 3 * u32) ^ depth d ≤ 1 + (2 : ℝ) ^ (-17 : ℤ) := by
    have h1 : (1 + 3 * u32) ^ depth d ≤ (1 + 3 * u32) ^ 10 :=
      pow_le_pow_right₀ hb1 (rstd_tol_depth_le hd)
    have h2 : (1 + 3 * u32) ^ 10 ≤ 1 + (2 : ℝ) ^ (-17 : ℤ) := by
      unfold u32; norm_num
    linarith
  have hrel : ((1 + 3 * u32) ^ depth d - 1) ≤ 3.01 * u32 * (depth d : ℝ) := by
    have h4 := rstd_tol_pow_sub_le (x := 3 * u32) (by linarith) (depth d)
    have hB : (depth d : ℝ) * (3 * u32) * (1 + 3 * u32) ^ depth d
        ≤ (depth d : ℝ) * (3 * u32) * (1 + 2 ^ (-17 : ℤ)) := by
      apply mul_le_mul_of_nonneg_left hq
      exact mul_nonneg hA0' (by linarith)
    have hε : 3 * (1 + (2:ℝ) ^ (-17 : ℤ)) ≤ 3.01 := by norm_num
    have hC : (depth d : ℝ) * (3 * u32) * (1 + 2 ^ (-17 : ℤ)) ≤ 3.01 * u32 * (depth d : ℝ) := by
      have h0 : (0:ℝ) ≤ (depth d : ℝ) * u32 := mul_nonneg hA0' hu''
      have h5 := mul_le_mul_of_nonneg_left hε h0
      calc (depth d : ℝ) * (3 * u32) * (1 + 2 ^ (-17 : ℤ))
          = ((depth d : ℝ) * u32) * (3 * (1 + 2 ^ (-17 : ℤ))) := by ring
        _ ≤ ((depth d : ℝ) * u32) * 3.01 := h5
        _ = 3.01 * u32 * (depth d : ℝ) := by ring
    calc ((1 + 3 * u32) ^ depth d - 1)
        ≤ (depth d : ℝ) * (3 * u32) * (1 + 3 * u32) ^ depth d := h4
      _ ≤ (depth d : ℝ) * (3 * u32) * (1 + 2 ^ (-17 : ℤ)) := hB
      _ ≤ 3.01 * u32 * (depth d : ℝ) := hC
  have heta : 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d
      ≤ (2 : ℝ) ^ (-139 : ℤ) := by
    have h1 : ((d.BD - 1 : ℕ) : ℝ) ≤ 511 := by
      have h3 : ((d.BD - 1 : ℕ) : ℝ) ≤ ((512 - 1 : ℕ) : ℝ) := by
        apply Nat.cast_le.mpr; have := rstd_tol_BD_le hd; omega
      norm_num at h3 ⊢; linarith
    have hP0 : (0:ℝ) ≤ (1 + 3 * u32) ^ depth d := pow_nonneg (by linarith) _
    have h0 : (0:ℝ) ≤ 4 * eta32 * (1 + 3 * u32) ^ depth d :=
      mul_nonneg (by linarith) hP0
    have hb0 : (0:ℝ) ≤ 4 * eta32 * 511 := by nlinarith [hη0']
    calc 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d
        = ((d.BD - 1 : ℕ) : ℝ) * (4 * eta32 * (1 + 3 * u32) ^ depth d) := by ring
      _ ≤ 511 * (4 * eta32 * (1 + 3 * u32) ^ depth d) := mul_le_mul_of_nonneg_right h1 h0
      _ = 4 * eta32 * 511 * (1 + 3 * u32) ^ depth d := by ring
      _ ≤ 4 * eta32 * 511 * (1 + (2:ℝ) ^ (-17 : ℤ)) := mul_le_mul_of_nonneg_left hq hb0
      _ ≤ (2 : ℝ) ^ (-139 : ℤ) := by unfold eta32; norm_num
  have hAq2 : ((AC : ℚ) : ℝ) ≤ (1 + u32) * XR + (d.BD : ℝ) * eta32 := hAq
  have hbdBD : (d.BD : ℝ) * eta32 ≤ 512 * eta32 := by
    have hb : (d.BD : ℝ) ≤ 512 := by exact_mod_cast rstd_tol_BD_le hd
    nlinarith [eta32_pos]
  have hcross : 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR + (d.BD : ℝ) * eta32)
      ≤ 3.1 * u32 * (depth d : ℝ) * XR + (2 : ℝ) ^ (-140 : ℤ) := by
    have hq7 : 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR)
        ≤ 3.1 * u32 * (depth d : ℝ) * XR := by
      have h0 : (0:ℝ) ≤ u32 * (depth d : ℝ) * XR :=
        mul_nonneg (mul_nonneg hu'' hA0') hX0
      have h2 : 3.01 * (1 + u32) ≤ 3.1 := by unfold u32; norm_num
      have h5 := mul_le_mul_of_nonneg_left h2 h0
      calc 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR)
          = (u32 * (depth d : ℝ) * XR) * (3.01 * (1 + u32)) := by ring
        _ ≤ (u32 * (depth d : ℝ) * XR) * 3.1 := h5
        _ = 3.1 * u32 * (depth d : ℝ) * XR := by ring
    have hq8 : 3.01 * u32 * (depth d : ℝ) * ((d.BD : ℝ) * eta32)
        ≤ (2 : ℝ) ^ (-140 : ℤ) := by
      have hL : ((depth d : ℝ)) ≤ 10 := by exact_mod_cast rstd_tol_depth_le hd
      have h0a : (0:ℝ) ≤ 3.01 * u32 * (depth d : ℝ) := by nlinarith [hA0', hu'']
      have hB : 3.01 * u32 * (depth d : ℝ) * ((d.BD : ℝ) * eta32)
          ≤ 3.01 * u32 * (depth d : ℝ) * (512 * eta32) := mul_le_mul_of_nonneg_left hbdBD h0a
      have h0b : (0:ℝ) ≤ 3.01 * u32 * 512 * eta32 := by nlinarith [hu'', hη0']
      have hC : 3.01 * u32 * (depth d : ℝ) * (512 * eta32)
          ≤ 3.01 * u32 * 10 * (512 * eta32) := by
        calc 3.01 * u32 * (depth d : ℝ) * (512 * eta32)
            = (depth d : ℝ) * (3.01 * u32 * 512 * eta32) := by ring
          _ ≤ 10 * (3.01 * u32 * 512 * eta32) := mul_le_mul_of_nonneg_right hL h0b
          _ = 3.01 * u32 * 10 * (512 * eta32) := by ring
      have hD : 3.01 * u32 * 10 * (512 * eta32) ≤ (2 : ℝ) ^ (-140 : ℤ) := by
        unfold u32 eta32; norm_num
      linarith [hB, hC, hD]
    calc 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR + (d.BD : ℝ) * eta32)
        = 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR)
          + 3.01 * u32 * (depth d : ℝ) * ((d.BD : ℝ) * eta32) := by ring
      _ ≤ 3.1 * u32 * (depth d : ℝ) * XR + (2 : ℝ) ^ (-140 : ℤ) := by linarith [hq7, hq8]
  have hb1' : ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
      ≤ 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR + (d.BD : ℝ) * eta32) := by
    have h2 : (0:ℝ) ≤ ((AC : ℚ) : ℝ) := by exact_mod_cast hAC0
    have h0 : (0:ℝ) ≤ 3.01 * u32 * (depth d : ℝ) := by nlinarith [hA0', hu'']
    calc ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
        ≤ (3.01 * u32 * (depth d : ℝ)) * ((AC : ℚ) : ℝ) :=
          mul_le_mul_of_nonneg_right hrel h2
      _ = (3.01 * u32 * (depth d : ℝ)) * ((AC : ℚ) : ℝ) := rfl
      _ ≤ (3.01 * u32 * (depth d : ℝ)) * ((1 + u32) * XR + (d.BD : ℝ) * eta32) :=
          mul_le_mul_of_nonneg_left hAq2 h0
      _ = 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR + (d.BD : ℝ) * eta32) := rfl
  have hfin_bd : ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
      + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d
      ≤ 3.1 * u32 * (depth d : ℝ) * XR + (2 : ℝ) ^ (-140 : ℤ) + (2 : ℝ) ^ (-139 : ℤ) := by
    have hstep1 : ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
        + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d
        ≤ 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR + (d.BD : ℝ) * eta32)
          + (2 : ℝ) ^ (-139 : ℤ) := add_le_add hb1' heta
    calc ((1 + 3 * u32) ^ depth d - 1) * ((AC : ℚ) : ℝ)
        + 4 * eta32 * ((d.BD - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ depth d
        ≤ 3.01 * u32 * (depth d : ℝ) * ((1 + u32) * XR + (d.BD : ℝ) * eta32)
          + (2 : ℝ) ^ (-139 : ℤ) := hstep1
      _ ≤ (3.1 * u32 * (depth d : ℝ) * XR + (2 : ℝ) ^ (-140 : ℤ)) + (2 : ℝ) ^ (-139 : ℤ) :=
          add_le_add hcross (le_refl ((2 : ℝ) ^ (-139 : ℤ)))
      _ = 3.1 * u32 * (depth d : ℝ) * XR + (2 : ℝ) ^ (-140 : ℤ) + (2 : ℝ) ^ (-139 : ℤ) := by ring
  have hsum2 : (2 : ℝ) ^ (-140 : ℤ) + (2 : ℝ) ^ (-139 : ℤ) ≤ (2 : ℝ) ^ (-138 : ℤ) := by
    norm_num
  linarith [hfin_bd, hsum2]

/-- Real-arithmetic combination: the stated side inequalities give the final bound on the accumulated error terms. -/
theorem rstd_tol_bva_fin {uu vv mm cc dd ee ff tt : ℝ}
    (h4 : 4 * uu * vv + 4 * uu * mm ≤ 5 * uu * vv)
    (h2 : 4 * uu * cc ≤ ee)
    (h3 : dd + cc + ee ≤ ff)
    (h5 : mm + 5 * uu * vv = tt) :
    4 * uu * vv + 4 * uu * mm + 4 * uu * cc + dd + (mm + cc) ≤ tt + ff := by
  linarith [h4, h2, h3, h5]

/-- The float variance `bvar`, divided by `D`, is within `(3.1·L + 5)u` of the true mean square
`Σ x²/D`, plus `2^-137`. -/
theorem rstd_tol_bva {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    {P : Platform} (hdivf : Device.DivFull P)
    {x sq : Nat → F32} {bvar bva dv : F32} {tr : Tree}
    (hx : ∀ j, j < d.BD → (x j).isFinite = true ∧ |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40)
    (hzero : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0)
    (hfsq : ∀ j, j < d.BD → (sq j).isFinite = true)
    (hsq : ∀ j, j < d.BD → FloatV.NaNEq (FloatV.mul (x j) (x j)) (sq j))
    (hcov : Tree.Covers d.BD tr) (hdepth : tr.depth ≤ depth d)
    (heval : Tree.Eval (AddRel true) sq tr bvar)
    (hdv : dv.toReal = (d.D : ℝ)) (hdvfin : dv.isFinite = true)
    (hdiv : P.div .binary32 bvar dv bva) :
    bva.isFinite = true ∧
    |bva.toReal - (∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) / (d.D : ℝ)|
      ≤ (3.1 * (depth d : ℝ) + 6) * u32 *
          ((∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) / (d.D : ℝ)) + (2 : ℝ) ^ (-136 : ℤ) := by
  have hu : 0 < u32 := u32_pos
  have hD : 0 < d.D := hd.d_pos
  have hDR : (0 : ℝ) < (d.D : ℝ) := by exact_mod_cast hD
  set XR : ℝ := ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2 with hXR
  set V : ℝ := XR / (d.D : ℝ) with hV
  have hXR0 : 0 ≤ XR := Finset.sum_nonneg fun j _ => sq_nonneg _
  have hV0 : 0 ≤ V := by rw [hV]; positivity
  -- per-lane bound on `x²`
  have hper : ∀ j ∈ Finset.range d.D, (x j).toReal ^ 2 ≤ (2 ^ 58 + 2 ^ 40) ^ 2 := by
    intro j hj
    have h1 := (hx j (lt_of_lt_of_le (Finset.mem_range.mp hj) hd.d_le_bd)).2
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) h1 2
  have hXRle : XR ≤ (d.D : ℝ) * ((2 : ℝ) ^ 58 + 2 ^ 40) ^ 2 := by
    rw [hXR]
    calc ∑ j ∈ Finset.range d.D, (x j).toReal ^ 2
        ≤ ∑ _j ∈ Finset.range d.D, ((2 : ℝ) ^ 58 + 2 ^ 40) ^ 2 := Finset.sum_le_sum hper
      _ = (d.D : ℝ) * ((2 : ℝ) ^ 58 + 2 ^ 40) ^ 2 := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hsqbd : ((2 : ℝ) ^ 58 + 2 ^ 40) ^ 2 ≤ 2 ^ 117 := by norm_num
  have hVle : V ≤ (2 : ℝ) ^ 117 := by
    rw [hV, div_le_iff₀ hDR]
    calc XR ≤ (d.D : ℝ) * ((2 : ℝ) ^ 58 + 2 ^ 40) ^ 2 := hXRle
      _ ≤ (d.D : ℝ) * 2 ^ 117 := mul_le_mul_of_nonneg_left hsqbd (le_of_lt hDR)
      _ = 2 ^ 117 * (d.D : ℝ) := by ring
  -- the sum of the rounded squares and its error
  obtain ⟨hbvarfin, hbvarbd⟩ := rstd_tol_bvar (_init := init) (_t := t) hd hx hzero hfsq hsq hcov hdepth heval
  have hsqsum := rstd_tol_sq_sum (_init := init) (_t := t) hd hx hzero hsq hfsq
  have hstep1 : |bvar.toReal - XR| ≤ (3.1 * (depth d : ℝ) + 1) * u32 * XR
      + ((2 : ℝ) ^ (-138 : ℤ) + (d.BD : ℝ) * eta32) := by
    have h1 : |bvar.toReal - XR| ≤ |bvar.toReal - (∑ j ∈ Finset.range d.BD, (sq j).toReal)|
        + |(∑ j ∈ Finset.range d.BD, (sq j).toReal) - XR| := by
      have h3 := abs_add_le (bvar.toReal - (∑ j ∈ Finset.range d.BD, (sq j).toReal))
        ((∑ j ∈ Finset.range d.BD, (sq j).toReal) - XR)
      have e : bvar.toReal - (∑ j ∈ Finset.range d.BD, (sq j).toReal)
          + ((∑ j ∈ Finset.range d.BD, (sq j).toReal) - XR) = bvar.toReal - XR := by ring
      rwa [e] at h3
    have h2 : |(∑ j ∈ Finset.range d.BD, (sq j).toReal) - XR|
        ≤ u32 * XR + (d.BD : ℝ) * eta32 := by
      simpa [hXR] using hsqsum
    have h4 : (3.1 * (depth d : ℝ) * u32) * XR + (2 : ℝ) ^ (-138 : ℤ)
        ≤ (3.1 * (depth d : ℝ) + 1) * u32 * XR + (2 : ℝ) ^ (-138 : ℤ) := by
      have h5 : (0 : ℝ) ≤ u32 * XR := mul_nonneg (le_of_lt hu) hXR0
      nlinarith [h5]
    linarith [h1, h2, h4]
  have hdel0 : (0 : ℝ) ≤ (2 : ℝ) ^ (-138 : ℤ) + (d.BD : ℝ) * eta32 := by
    have h1 : (0 : ℝ) ≤ (d.BD : ℝ) := by exact_mod_cast Nat.zero_le d.BD
    have h2 : (0 : ℝ) < eta32 := eta32_pos
    have h3 : (0 : ℝ) < (2 : ℝ) ^ (-138 : ℤ) := by norm_num
    nlinarith
  have hdelta : (2 : ℝ) ^ (-138 : ℤ) + (d.BD : ℝ) * eta32 ≤ (2 : ℝ) ^ (-137 : ℤ) := by
    have hb : (d.BD : ℝ) ≤ 512 := by exact_mod_cast rstd_tol_BD_le hd
    have h2 : 512 * eta32 ≤ (2 : ℝ) ^ (-138 : ℤ) := by unfold eta32; norm_num
    have h3 : (2 : ℝ) ^ (-138 : ℤ) + (2 : ℝ) ^ (-138 : ℤ) = (2 : ℝ) ^ (-137 : ℤ) := by
      norm_num
    nlinarith [eta32_pos]
  -- `|bvar/D - V|` is small
  have hstep2 : |bvar.toReal / (d.D : ℝ) - V| ≤ (3.1 * (depth d : ℝ) + 1) * u32 * V
      + (2 : ℝ) ^ (-137 : ℤ) := by
    have h1 : bvar.toReal / (d.D : ℝ) - V = (bvar.toReal - XR) / (d.D : ℝ) := by
      rw [hV]; field_simp
    rw [h1, abs_div, abs_of_pos hDR]
    rw [div_le_iff₀ hDR]
    have h2 : |bvar.toReal - XR| ≤ (3.1 * (depth d : ℝ) + 1) * u32 * XR
        + ((2 : ℝ) ^ (-138 : ℤ) + (d.BD : ℝ) * eta32) := hstep1
    have hXRe : XR = V * (d.D : ℝ) := by rw [hV]; field_simp
    have h5 : (3.1 * (depth d : ℝ) + 1) * u32 * XR
        = (3.1 * (depth d : ℝ) + 1) * u32 * V * (d.D : ℝ) := by rw [hXRe]; ring
    rw [h5] at h2
    have h6 : ((3.1 * (depth d : ℝ) + 1) * u32 * V + (2 : ℝ) ^ (-137 : ℤ)) * (d.D : ℝ)
        = (3.1 * (depth d : ℝ) + 1) * u32 * V * (d.D : ℝ)
          + (2 : ℝ) ^ (-137 : ℤ) * (d.D : ℝ) := by ring
    rw [h6]
    have h7 : (2 : ℝ) ^ (-137 : ℤ) ≤ (2 : ℝ) ^ (-137 : ℤ) * (d.D : ℝ) := by
      have h8 : (1 : ℝ) ≤ (d.D : ℝ) := by exact_mod_cast hD
      nlinarith [show (0:ℝ) < (2:ℝ) ^ (-137 : ℤ) by norm_num]
    have h8 : (2 : ℝ) ^ (-138 : ℤ) + (d.BD : ℝ) * eta32
        ≤ (2 : ℝ) ^ (-137 : ℤ) * (d.D : ℝ) := hdelta.trans h7
    exact h2.trans (add_le_add_right h8 _)
  -- the quotient's magnitude, and the division bound
  have hqmax : |bvar.toRat / dv.toRat| ≤ (2 : ℚ) ^ 126 := by
    have hbq : |bvar.toReal / (d.D : ℝ)| ≤ (2 : ℝ) ^ 118 := by
      have h4 : |bvar.toReal / (d.D : ℝ)| ≤ V + ((3.1 * (depth d : ℝ) + 1) * u32 * V
          + (2 : ℝ) ^ (-137 : ℤ)) := by
        have h5 := abs_add_le V (bvar.toReal / (d.D : ℝ) - V)
        have e : V + (bvar.toReal / (d.D : ℝ) - V) = bvar.toReal / (d.D : ℝ) := by ring
        rw [e, abs_of_nonneg hV0] at h5
        linarith [abs_le.mp hstep2]
      have hL : (depth d : ℝ) ≤ 10 := by exact_mod_cast rstd_tol_depth_le hd
      have h7 : (3.1 * (depth d : ℝ) + 1) * u32 ≤ 1 / 2 := by unfold u32; nlinarith
      have h8 : (3.1 * (depth d : ℝ) + 1) * u32 * V ≤ (1 / 2) * V :=
        mul_le_mul_of_nonneg_right h7 hV0
      have h9 : (2 : ℝ) ^ (-137 : ℤ) ≤ 1 := by norm_num
      nlinarith [hVle, h8, h9]
    have h2 : |bvar.toReal / dv.toReal| ≤ (2 : ℝ) ^ 126 := by
      rw [hdv]
      refine hbq.trans ?_
      norm_num
    have h6 : |(bvar.toRat : ℝ) / (dv.toRat : ℝ)| ≤ (2 : ℝ) ^ 126 := by
      simpa [FloatV.toReal] using h2
    have h7 : ((|bvar.toRat / dv.toRat| : ℚ) : ℝ) ≤ ((2 : ℚ) ^ 126 : ℝ) := by
      push_cast
      simpa using h6
    exact_mod_cast h7
  have hdvne : dv.toRat ≠ 0 := by
    intro h0
    have h1 : dv.toReal = 0 := by simp [FloatV.toReal, h0]
    rw [hdv] at h1
    have h2 : (d.D : ℝ) = 0 := h1
    exact (ne_of_gt hDR) h2
  obtain ⟨hbvafin, hbvaerrQ⟩ := hdivf bvar dv bva hdiv hbvarfin hdvfin hdvne hqmax
  have hbvaerr : |bva.toReal - bvar.toReal / dv.toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * |bvar.toReal / dv.toReal| + (2 : ℝ) ^ (-148 : ℤ) := by
    have h1 : ((|bva.toRat - bvar.toRat / dv.toRat| : ℚ) : ℝ)
        ≤ ((((2 : ℚ) ^ (-22 : ℤ)) * |bvar.toRat / dv.toRat| + (2 : ℚ) ^ (-148 : ℤ) : ℚ) : ℝ) := by
      exact_mod_cast hbvaerrQ
    push_cast at h1
    simpa [FloatV.toReal] using h1
  refine ⟨hbvafin, ?_⟩
  have hbvad : bvar.toReal / dv.toReal = bvar.toReal / (d.D : ℝ) := by rw [hdv]
  rw [hbvad] at hbvaerr
  have h4u : (2 : ℝ) ^ (-22 : ℤ) = 4 * u32 := by unfold u32; norm_num
  rw [h4u] at hbvaerr
  -- final assembly
  have hV1 : |bva.toReal - V| ≤ |bva.toReal - bvar.toReal / (d.D : ℝ)|
      + |bvar.toReal / (d.D : ℝ) - V| := by
    have h3 := abs_add_le (bva.toReal - bvar.toReal / (d.D : ℝ))
      (bvar.toReal / (d.D : ℝ) - V)
    have e : bva.toReal - bvar.toReal / (d.D : ℝ) + (bvar.toReal / (d.D : ℝ) - V)
        = bva.toReal - V := by ring
    rwa [e] at h3
  have hbq2 : |bvar.toReal / (d.D : ℝ)| ≤ V + ((3.1 * (depth d : ℝ) + 1) * u32 * V
      + (2 : ℝ) ^ (-137 : ℤ)) := by
    have h5 := abs_add_le V (bvar.toReal / (d.D : ℝ) - V)
    have e : V + (bvar.toReal / (d.D : ℝ) - V) = bvar.toReal / (d.D : ℝ) := by ring
    rw [e, abs_of_nonneg hV0] at h5
    linarith [abs_le.mp hstep2]
  have hmain : |bva.toReal - V| ≤ 4 * u32 * (V + ((3.1 * (depth d : ℝ) + 1) * u32 * V
      + (2 : ℝ) ^ (-137 : ℤ))) + (2 : ℝ) ^ (-148 : ℤ)
      + ((3.1 * (depth d : ℝ) + 1) * u32 * V + (2 : ℝ) ^ (-137 : ℤ)) := by
    have h1 : 4 * u32 * |bvar.toReal / (d.D : ℝ)| ≤ 4 * u32 * (V + ((3.1 * (depth d : ℝ) + 1)
        * u32 * V + (2 : ℝ) ^ (-137 : ℤ))) := by
      apply mul_le_mul_of_nonneg_left hbq2 (by positivity)
    linarith [hbvaerr, hstep2, h1, hV1]
  have h1 : 4 * u32 * ((3.1 * (depth d : ℝ) + 1) * u32 * V) ≤ 1 * u32 * V := by
    have h2 : 4 * u32 * ((3.1 * (depth d : ℝ) + 1) * u32) ≤ 1 * u32 := by
      have hL : (depth d : ℝ) ≤ 10 := by exact_mod_cast rstd_tol_depth_le hd
      unfold u32
      nlinarith
    nlinarith [h2, hV0, hu]
  have h2 : 4 * u32 * (2 : ℝ) ^ (-137 : ℤ) ≤ (2 : ℝ) ^ (-143 : ℤ) := by
    unfold u32; norm_num
  have h3 : (2 : ℝ) ^ (-148 : ℤ) + (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-143 : ℤ)
      ≤ (2 : ℝ) ^ (-136 : ℤ) := by norm_num
  have h4 : 4 * u32 * V + 4 * u32 * ((3.1 * (depth d : ℝ) + 1) * u32 * V)
      ≤ 5 * u32 * V := by linarith [h1]
  have h5 : (3.1 * (depth d : ℝ) + 1) * u32 * V + 5 * u32 * V
      = (3.1 * (depth d : ℝ) + 6) * u32 * V := by ring
  refine hmain.trans ?_
  have e2 : 4 * u32 * (V + ((3.1 * (depth d : ℝ) + 1) * u32 * V + (2 : ℝ) ^ (-137 : ℤ)))
      + (2 : ℝ) ^ (-148 : ℤ)
      + ((3.1 * (depth d : ℝ) + 1) * u32 * V + (2 : ℝ) ^ (-137 : ℤ))
      = 4 * u32 * V + 4 * u32 * ((3.1 * (depth d : ℝ) + 1) * u32 * V)
        + 4 * u32 * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-148 : ℤ)
        + (((3.1 * (depth d : ℝ) + 1) * u32 * V) + (2 : ℝ) ^ (-137 : ℤ)) := by ring
  rw [e2]
  exact rstd_tol_bva_fin h4 h2 h3 h5

/-- The float chain `be → bsq → brstd` against `R = 1/√W`, with the tighter constant `13u`. -/
theorem rstd_tol_chain13 {W be bsq brstd R Th e149 e148 : ℝ}
    (he149 : e149 = (2:ℝ)^(-149:ℤ)) (_he148 : e148 = (2:ℝ)^(-148:ℤ))
    (hW : 0 < W) (hR : 0 < R) (hRWs : R ^ 2 * W = 1)
    (_hTh0 : 0 ≤ Th) (hTh : Th ≤ 1/2)
    (hbe : |be - W| ≤ Th * W)
    (hbsq : |bsq - Real.sqrt be| ≤ 4 * u32 * Real.sqrt be + e149)
    (hbrstd : |brstd - 1/bsq| ≤ 4 * u32 * |1/bsq| + e148)
    (hs : (2:ℝ)^(-63:ℤ) ≤ Real.sqrt be)
    (hj2 : e148 ≤ u32 * R) :
    |brstd - R| ≤ R * (Th / 2 + Th ^ 2 + 13 * u32) := by
  have hu : 0 < u32 := u32_pos
  have hW0 : 0 < Real.sqrt W := Real.sqrt_pos.2 hW
  have hRW : R * Real.sqrt W = 1 := by
    have h2 : (R * Real.sqrt W) ^ 2 = 1 := by
      rw [mul_pow, Real.sq_sqrt (le_of_lt hW), hRWs]
    nlinarith only [h2, mul_pos hR hW0]
  have hRinv : R = 1 / Real.sqrt W := by
    field_simp
    linarith only [hRW]
  set δ := (be - W) / W with hδdef
  have hbem := abs_le.mp hbe
  have hδ : |δ| ≤ Th := by
    rw [hδdef, abs_div, abs_of_pos hW, div_le_iff₀ hW]
    exact abs_le.mpr ⟨by linarith only [hbem.1], by linarith only [hbem.2]⟩
  have hδm := abs_le.mp hδ
  have hδlo : -(1/2:ℝ) ≤ δ := by linarith only [hδm.1, hTh]
  have hδhi : δ ≤ 1/2 := by linarith only [hδm.2, hTh]
  have h1δ : 0 < 1 + δ := by linarith only [hδlo]
  have hbe_eq : be = W * (1 + δ) := by rw [hδdef]; field_simp; ring
  have hbe_pos : 0 < be := by rw [hbe_eq]; positivity
  have hsbe_pos : 0 < Real.sqrt be := Real.sqrt_pos.2 hbe_pos
  have hsbe : Real.sqrt be = Real.sqrt W * Real.sqrt (1+δ) := by
    rw [hbe_eq, Real.sqrt_mul (le_of_lt hW)]
  have hinv_be : 1 / Real.sqrt be = R * (1 / Real.sqrt (1+δ)) := by
    rw [hsbe, hRinv]
    field_simp
  have hA_le : 1 / Real.sqrt (1 + δ) ≤ (71/50 : ℝ) := by
    have h1 : Real.sqrt (1/2 : ℝ) ≤ Real.sqrt (1 + δ) := Real.sqrt_le_sqrt (by linarith only [hδlo])
    have h2 : (0:ℝ) < Real.sqrt (1/2 : ℝ) := Real.sqrt_pos.2 (by norm_num)
    have h3 : 1 / Real.sqrt (1 + δ) ≤ 1 / Real.sqrt (1/2 : ℝ) :=
      one_div_le_one_div_of_le h2 h1
    have h4 : 1 / Real.sqrt (1/2 : ℝ) ≤ (71/50 : ℝ) := by
      rw [div_le_iff₀ h2]
      have h5 : Real.sqrt (1/2 : ℝ) * Real.sqrt (1/2 : ℝ) = 1/2 :=
        Real.mul_self_sqrt (by norm_num)
      nlinarith only [h5, Real.sqrt_nonneg (1/2 : ℝ)]
    linarith only [h3, h4]
  have hU := rstd_tol_inv_sqrt_upper hδlo hδhi
  have hL := rstd_tol_inv_sqrt_lower hδlo hδhi
  have hrec : |1 / Real.sqrt (1+δ) - 1| ≤ Th/2 + Th^2 := by
    rw [abs_le]
    constructor
    · nlinarith only [hL, hδm.1, hδm.2, ‹0 ≤ Th›]
    · nlinarith only [hU, hδm.1, hδm.2, ‹0 ≤ Th›]
  have hkey1 : |1 / Real.sqrt be - R| ≤ R * (Th/2 + Th^2) := by
    rw [hinv_be, show R * (1 / Real.sqrt (1 + δ)) - R = R * (1 / Real.sqrt (1 + δ) - 1) by ring,
      abs_mul, abs_of_pos hR]
    exact mul_le_mul_of_nonneg_left hrec (le_of_lt hR)
  have hinv_be_le : 1 / Real.sqrt be ≤ (71/50 : ℝ) * R := by
    rw [hinv_be]
    nlinarith only [hA_le, hR]
  have heps1 : |bsq / Real.sqrt be - 1| ≤ (401/100 : ℝ) * u32 := by
    have h2 : e149 ≤ (1/100 : ℝ) * u32 * Real.sqrt be := by
      have h6 : e149 ≤ (1/100 : ℝ) * u32 * (2:ℝ)^(-63:ℤ) := by
        rw [he149]; unfold u32; norm_num
      have h7 : (0:ℝ) ≤ (1/100 : ℝ) * u32 := by positivity
      have h8 : (1/100 : ℝ) * u32 * (2:ℝ)^(-63:ℤ) ≤ (1/100 : ℝ) * u32 * Real.sqrt be :=
        mul_le_mul_of_nonneg_left hs h7
      exact h6.trans h8
    have h1 : bsq / Real.sqrt be - 1 = (bsq - Real.sqrt be) / Real.sqrt be := by
      field_simp
    rw [h1, abs_div, abs_of_pos hsbe_pos, div_le_iff₀ hsbe_pos]
    linarith only [hbsq, h2]
  have hr1pos : 0 < bsq / Real.sqrt be := by
    have h := abs_le.mp heps1
    have h2 : (401/100 : ℝ) * u32 ≤ 1/2 := by unfold u32; norm_num
    linarith only [h.1, h2]
  have hbsq_pos : 0 < bsq := by
    have h := mul_pos hr1pos hsbe_pos
    have h2 : bsq / Real.sqrt be * Real.sqrt be = bsq := div_mul_cancel₀ bsq (ne_of_gt hsbe_pos)
    rwa [h2] at h
  have hbsq_ne : bsq ≠ 0 := ne_of_gt hbsq_pos
  have hb2 : |1/(bsq/Real.sqrt be) - 1| ≤ (402/100 : ℝ) * u32 := by
    have hq1 : |1 - bsq/Real.sqrt be| = |bsq/Real.sqrt be - 1| := abs_sub_comm _ _
    have hqlo : 1 - (401/100 : ℝ) * u32 ≤ bsq/Real.sqrt be := by
      have h := abs_le.mp heps1; linarith only [h.1]
    have hqpos : 0 < bsq/Real.sqrt be := hr1pos
    rw [show 1/(bsq/Real.sqrt be) - 1 = (1 - bsq/Real.sqrt be)/(bsq/Real.sqrt be) by
      field_simp]
    rw [abs_div, abs_of_pos hqpos, div_le_iff₀ hqpos, hq1]
    refine heps1.trans ?_
    have h1 : (402/100 : ℝ) * u32 * (1 - (401/100 : ℝ) * u32)
        ≤ (402/100 : ℝ) * u32 * (bsq/Real.sqrt be) :=
      mul_le_mul_of_nonneg_left hqlo (by positivity)
    have h2 : (401/100 : ℝ) * u32 ≤ (402/100 : ℝ) * u32 * (1 - (401/100 : ℝ) * u32) := by
      unfold u32; norm_num
    linarith only [h1, h2]
  have hstep1 : |1/bsq - 1/Real.sqrt be| ≤ (571/100 : ℝ) * u32 * R := by
    have hone : 1/bsq = (1/Real.sqrt be) / (bsq/Real.sqrt be) := by field_simp
    rw [hone, show (1/Real.sqrt be)/(bsq/Real.sqrt be) - 1/Real.sqrt be
        = (1/Real.sqrt be) * (1/(bsq/Real.sqrt be) - 1) by ring]
    rw [abs_mul, abs_of_pos (by positivity : (0:ℝ) < 1/Real.sqrt be)]
    have hmul := mul_le_mul hinv_be_le hb2 (abs_nonneg _)
      (by positivity : (0:ℝ) ≤ (71/50 : ℝ) * R)
    have hfin : (71/50 : ℝ) * R * ((402/100 : ℝ) * u32) ≤ (571/100 : ℝ) * u32 * R := by
      have h0 : (0:ℝ) ≤ u32 * R := by positivity
      nlinarith only [h0]
    linarith only [hmul, hfin]
  have hinv_bsq_le : |1/bsq| ≤ (144/100 : ℝ) * R := by
    have h3 : |1/bsq| = (1/Real.sqrt be) * |1/(bsq/Real.sqrt be)| := by
      have e : (1:ℝ)/bsq = (1/Real.sqrt be) * (1/(bsq/Real.sqrt be)) := by field_simp
      rw [e, abs_mul, abs_of_pos (by positivity : (0:ℝ) < 1/Real.sqrt be)]
    have h4 : |1/(bsq/Real.sqrt be)| ≤ 1 + (402/100 : ℝ) * u32 := by
      have hb := abs_le.mp hb2
      have h0 : (0:ℝ) ≤ (402/100 : ℝ) * u32 := by positivity
      exact abs_le.mpr ⟨by linarith only [hb.1, h0], by linarith only [hb.2]⟩
    have h5 : (1 + (402/100 : ℝ) * u32) ≤ (100/99 : ℝ) := by unfold u32; norm_num
    rw [h3]
    have h6 : (1/Real.sqrt be) * |1/(bsq/Real.sqrt be)| ≤ ((71/50 : ℝ) * R) * (100/99 : ℝ) :=
      mul_le_mul hinv_be_le (le_trans h4 h5) (abs_nonneg _) (by positivity)
    have h7 : ((71/50 : ℝ) * R) * (100/99 : ℝ) ≤ (144/100 : ℝ) * R := by
      nlinarith only [hR]
    linarith only [h6, h7]
  have hsum : |1/bsq - R| ≤ R*(Th/2 + Th^2) + (571/100 : ℝ)*u32*R := by
    have h3 : |1/bsq - R| ≤ |1/bsq - 1/Real.sqrt be| + |1/Real.sqrt be - R| := by
      have h4 := abs_add_le (1/bsq - 1/Real.sqrt be) (1/Real.sqrt be - R)
      have e : 1/bsq - 1/Real.sqrt be + (1/Real.sqrt be - R) = 1/bsq - R := by ring
      rwa [e] at h4
    linarith only [h3, hstep1, hkey1]
  have hfinal : |brstd - R| ≤ 4 * u32 * |1/bsq| + e148 + |1/bsq - R| := by
    have h2 : |brstd - R| ≤ |brstd - 1/bsq| + |1/bsq - R| := by
      have h3 := abs_add_le (brstd - 1/bsq) (1/bsq - R)
      have e : brstd - 1/bsq + (1/bsq - R) = brstd - R := by ring
      rwa [e] at h3
    linarith only [h2, hbrstd]
  have ht : 4 * u32 * |1/bsq| + e148 + |1/bsq - R|
      ≤ R * (Th/2 + Th^2 + 13 * u32) := by
    have h4u : 4 * u32 * |1/bsq| ≤ (576/100 : ℝ) * u32 * R := by
      have h := mul_le_mul_of_nonneg_left hinv_bsq_le (by positivity : (0:ℝ) ≤ 4*u32)
      have e : 4 * u32 * ((144/100 : ℝ) * R) = (576/100 : ℝ) * u32 * R := by ring
      linarith only [h, e]
    have hm : 4 * u32 * |1/bsq| + e148 + |1/bsq - R|
        ≤ R*(Th/2+Th^2) + (1247/100 : ℝ) * (u32*R) := by
      linarith only [h4u, hj2, hsum]
    have hM0 : (0:ℝ) ≤ u32 * R := by positivity
    have hm2 : (1247/100 : ℝ) * (u32*R) ≤ 13 * (u32*R) := by nlinarith only [hM0]
    have e : R * (Th/2 + Th^2 + 13 * u32) = R*(Th/2+Th^2) + 13*(u32*R) := by ring
    linarith only [hm, hm2, e]
  exact hfinal.trans ht

/-- Reverse triangle (Minkowski) for the root mean square: moving every lane by `x - d` moves
`√(Σ·²/n)` by at most `√(Σ(x-d)²/n)`. -/
theorem rstd_tol_sqrt_sum_sub {n : ℕ} (hn : 0 < n) {x d : ℕ → ℝ} :
    |Real.sqrt ((∑ j ∈ Finset.range n, (x j) ^ 2) / n)
      - Real.sqrt ((∑ j ∈ Finset.range n, (d j) ^ 2) / n)|
      ≤ Real.sqrt ((∑ j ∈ Finset.range n, (x j - d j) ^ 2) / n) := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  have hsn : (0:ℝ) < Real.sqrt (n:ℝ) := Real.sqrt_pos.2 hnR
  have step : ∀ a b : ℕ → ℝ,
      Real.sqrt (∑ j ∈ Finset.range n, (a j) ^ 2)
        ≤ Real.sqrt (∑ j ∈ Finset.range n, (b j) ^ 2)
          + Real.sqrt (∑ j ∈ Finset.range n, (a j - b j) ^ 2) := by
    intro a b
    have hb0 : 0 ≤ ∑ j ∈ Finset.range n, (b j) ^ 2 :=
      Finset.sum_nonneg fun j _ => sq_nonneg _
    have he0 : 0 ≤ ∑ j ∈ Finset.range n, (a j - b j) ^ 2 :=
      Finset.sum_nonneg fun j _ => sq_nonneg _
    have h1 : ∑ j ∈ Finset.range n, (a j) ^ 2
        = (∑ j ∈ Finset.range n, (b j) ^ 2)
          + 2 * (∑ j ∈ Finset.range n, b j * (a j - b j))
          + (∑ j ∈ Finset.range n, (a j - b j) ^ 2) := by
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro j hj
      ring
    have h2 : ∑ j ∈ Finset.range n, b j * (a j - b j)
        ≤ Real.sqrt (∑ j ∈ Finset.range n, (b j) ^ 2)
          * Real.sqrt (∑ j ∈ Finset.range n, (a j - b j) ^ 2) :=
      Real.sum_mul_le_sqrt_mul_sqrt _ _ _
    have h3 : (∑ j ∈ Finset.range n, (a j) ^ 2)
        ≤ (Real.sqrt (∑ j ∈ Finset.range n, (b j) ^ 2)
          + Real.sqrt (∑ j ∈ Finset.range n, (a j - b j) ^ 2)) ^ 2 := by
      rw [h1]
      have e : (Real.sqrt (∑ j ∈ Finset.range n, (b j) ^ 2)
            + Real.sqrt (∑ j ∈ Finset.range n, (a j - b j) ^ 2)) ^ 2
          = (∑ j ∈ Finset.range n, (b j) ^ 2)
            + 2 * (Real.sqrt (∑ j ∈ Finset.range n, (b j) ^ 2)
              * Real.sqrt (∑ j ∈ Finset.range n, (a j - b j) ^ 2))
            + (∑ j ∈ Finset.range n, (a j - b j) ^ 2) := by
        rw [add_sq, Real.sq_sqrt hb0, Real.sq_sqrt he0]
        ring
      nlinarith [h2]
    calc Real.sqrt (∑ j ∈ Finset.range n, (a j) ^ 2)
        ≤ Real.sqrt ((Real.sqrt (∑ j ∈ Finset.range n, (b j) ^ 2)
          + Real.sqrt (∑ j ∈ Finset.range n, (a j - b j) ^ 2)) ^ 2) := Real.sqrt_le_sqrt h3
      _ = _ := Real.sqrt_sq (by positivity)
  have hSe : (∑ j ∈ Finset.range n, (d j - x j) ^ 2) = (∑ j ∈ Finset.range n, (x j - d j) ^ 2) := by
    apply Finset.sum_congr rfl
    intro j hj
    ring
  have hA := step x d
  have hB := step d x
  rw [hSe] at hB
  have hSx0 : 0 ≤ ∑ j ∈ Finset.range n, (x j) ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  have hSd0 : 0 ≤ ∑ j ∈ Finset.range n, (d j) ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  have hSe0 : 0 ≤ ∑ j ∈ Finset.range n, (x j - d j) ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  rw [Real.sqrt_div hSx0, Real.sqrt_div hSd0, Real.sqrt_div hSe0]
  rw [abs_le]
  constructor
  · have hB' : -(Real.sqrt (∑ j ∈ Finset.range n, (x j - d j) ^ 2))
        ≤ Real.sqrt (∑ j ∈ Finset.range n, (x j) ^ 2)
          - Real.sqrt (∑ j ∈ Finset.range n, (d j) ^ 2) := by
      linarith only [hB]
    rw [← neg_div, ← sub_div]
    exact div_le_div_of_nonneg_right hB' (le_of_lt hsn)
  · have hA' : Real.sqrt (∑ j ∈ Finset.range n, (x j) ^ 2)
        - Real.sqrt (∑ j ∈ Finset.range n, (d j) ^ 2)
        ≤ Real.sqrt (∑ j ∈ Finset.range n, (x j - d j) ^ 2) := by
      linarith only [hA]
    rw [← sub_div]
    exact div_le_div_of_nonneg_right hA' (le_of_lt hsn)

/-- The root-mean-square distance between the computed deviation lanes `x` and the exact ones
`sR - meanRef`, from the per-lane decomposition of `rstd_tol_dev`. -/
theorem rstd_tol_rms_dev {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    {x : Nat → F32}
    (hdecomp : ∀ j, j < d.D → ∃ c ε : ℝ, |c| ≤ 2 * u32 ∧
      |ε| ≤ meanTol d init t + u32 * (|sR d init t j| + absMean d init t + meanTol d init t)
        + 4 * eta32 ∧
      (x j).toReal = (sR d init t j - meanRef d init t + ε) * (1 + c)) :
    |Real.sqrt ((∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) / (d.D : ℝ))
      - Real.sqrt (varRef d init t)|
      ≤ 2 * u32 * Real.sqrt (varRef d init t)
        + (1 + 2 * u32) * (meanTol d init t
            + u32 * (rmsS d init t + absMean d init t + meanTol d init t) + 4 * eta32) := by
  have hu : 0 ≤ u32 := le_of_lt u32_pos
  have hη : 0 ≤ eta32 := le_of_lt eta32_pos
  have hD0 : 0 < d.D := hd.d_pos
  set EB : ℝ := meanTol d init t + u32 * (rmsS d init t + absMean d init t + meanTol d init t)
    + 4 * eta32 with hEB
  set EBj : ℕ → ℝ := fun j => meanTol d init t
    + u32 * (|sR d init t j| + absMean d init t + meanTol d init t) + 4 * eta32 with hEBj
  have hmT0 : 0 ≤ meanTol d init t := rstd_tol_meanTol_nonneg
  have haM0 : 0 ≤ absMean d init t := rstd_tol_absMean_nonneg
  have hrms0 : 0 ≤ rmsS d init t := rstd_tol_rmsS_nonneg
  have hEBj0 : ∀ j, 0 ≤ EBj j := by
    intro j
    have h1 : (0:ℝ) ≤ |sR d init t j| := abs_nonneg _
    rw [hEBj]; nlinarith [hu, hη, hmT0, haM0, h1]
  have hEB0 : 0 ≤ EB := by
    rw [hEB]; nlinarith [hu, hη, hmT0, haM0, hrms0]
  -- generic scaling of the quadratic mean
  have hscale : ∀ (c : ℝ), 0 ≤ c → ∀ F : ℕ → ℝ,
      Real.sqrt ((∑ j ∈ Finset.range d.D, (c * F j) ^ 2) / (d.D : ℝ))
        = c * Real.sqrt ((∑ j ∈ Finset.range d.D, (F j) ^ 2) / (d.D : ℝ)) := by
    intro c hc F
    have hsum : (∑ j ∈ Finset.range d.D, (c * F j) ^ 2)
        = c ^ 2 * (∑ j ∈ Finset.range d.D, (F j) ^ 2) := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hsum, mul_div_assoc, Real.sqrt_mul (by positivity), Real.sqrt_sq hc]
  have hddef : Real.sqrt ((∑ j ∈ Finset.range d.D,
      (sR d init t j - meanRef d init t) ^ 2) / (d.D : ℝ))
      = Real.sqrt (varRef d init t) := by
    simp only [varRef]
  -- the per-lane bound
  have hpt : ∀ j, j < d.D → |(x j).toReal - (sR d init t j - meanRef d init t)|
      ≤ 2 * u32 * |sR d init t j - meanRef d init t| + (1 + 2 * u32) * EBj j := by
    intro j hj
    obtain ⟨c, ε, hc, hε, hxeq⟩ := hdecomp j hj
    have h1c : |1 + c| ≤ 1 + 2 * u32 := by
      have h := abs_le.mp hc
      rw [abs_le]
      constructor <;> linarith [h.1, h.2]
    have hε' : |ε| ≤ EBj j := by
      rw [hEBj]
      exact hε
    have e : (x j).toReal - (sR d init t j - meanRef d init t)
        = c * (sR d init t j - meanRef d init t) + ε * (1 + c) := by
      rw [hxeq]; ring
    rw [e]
    have h1 : |c * (sR d init t j - meanRef d init t)|
        ≤ 2 * u32 * |sR d init t j - meanRef d init t| := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right hc (abs_nonneg _)
    have h2 : |ε * (1 + c)| ≤ EBj j * (1 + 2 * u32) := by
      rw [abs_mul]
      exact mul_le_mul hε' h1c (abs_nonneg _) (by nlinarith [hu, hEBj0 j])
    calc |c * (sR d init t j - meanRef d init t) + ε * (1 + c)|
        ≤ |c * (sR d init t j - meanRef d init t)| + |ε * (1 + c)| := abs_add_le _ _
      _ ≤ 2 * u32 * |sR d init t j - meanRef d init t| + EBj j * (1 + 2 * u32) := by linarith
      _ = 2 * u32 * |sR d init t j - meanRef d init t| + (1 + 2 * u32) * EBj j := by ring
  -- monotonicity in the lanes
  have hmono : Real.sqrt ((∑ j ∈ Finset.range d.D,
        ((x j).toReal - (sR d init t j - meanRef d init t)) ^ 2) / (d.D : ℝ))
      ≤ Real.sqrt ((∑ j ∈ Finset.range d.D,
        (2 * u32 * |sR d init t j - meanRef d init t| + (1 + 2 * u32) * EBj j) ^ 2)
          / (d.D : ℝ)) := by
    apply Real.sqrt_le_sqrt
    apply div_le_div_of_nonneg_right _ (by positivity)
    apply Finset.sum_le_sum
    intro j hj
    have hj' : j < d.D := Finset.mem_range.mp hj
    have h1 : |(x j).toReal - (sR d init t j - meanRef d init t)|
        ≤ 2 * u32 * |sR d init t j - meanRef d init t| + (1 + 2 * u32) * EBj j := hpt j hj'
    have h2 : (0:ℝ) ≤ 2 * u32 * |sR d init t j - meanRef d init t| + (1 + 2 * u32) * EBj j := by
      have h3 : (0:ℝ) ≤ 2 * u32 * |sR d init t j - meanRef d init t| := by positivity
      have h4 : (0:ℝ) ≤ (1 + 2 * u32) * EBj j := by nlinarith [hu, hEBj0 j]
      linarith
    rw [show ((x j).toReal - (sR d init t j - meanRef d init t)) ^ 2
        = |(x j).toReal - (sR d init t j - meanRef d init t)| ^ 2 from by rw [sq_abs]]
    exact pow_le_pow_left₀ (abs_nonneg _) h1 2
  -- split into the two error sources
  have hB : Real.sqrt ((∑ j ∈ Finset.range d.D,
        (2 * u32 * |sR d init t j - meanRef d init t| + (1 + 2 * u32) * EBj j) ^ 2)
          / (d.D : ℝ))
      ≤ 2 * u32 * Real.sqrt (varRef d init t)
        + (1 + 2 * u32) * Real.sqrt ((∑ j ∈ Finset.range d.D, (EBj j) ^ 2)
            / (d.D : ℝ)) := by
    have hsa := rstd_tol_sqrt_sum_add (n := d.D) hD0
      (a := fun j => 2 * u32 * |sR d init t j - meanRef d init t|)
      (b := fun j => (1 + 2 * u32) * EBj j)
      (fun j _ => by positivity) (fun j _ => by nlinarith [hu, hEBj0 j])
    have eA := hscale (2 * u32) (by positivity)
      (fun j => |sR d init t j - meanRef d init t|)
    have eA' : Real.sqrt ((∑ j ∈ Finset.range d.D,
        (|sR d init t j - meanRef d init t|) ^ 2) / (d.D : ℝ))
        = Real.sqrt (varRef d init t) := by
      rw [← hddef]
      congr 1
      congr 1
      exact Finset.sum_congr rfl fun j _ => by rw [sq_abs]
    have eB := hscale (1 + 2 * u32) (by nlinarith [hu]) (fun j => EBj j)
    refine hsa.trans ?_
    rw [eA, eA', eB]
  -- the `EBj` root mean square is at most `EB`
  have hEBS : Real.sqrt ((∑ j ∈ Finset.range d.D, (EBj j) ^ 2) / (d.D : ℝ)) ≤ EB := by
    set c1 : ℝ := u32 * (absMean d init t + meanTol d init t) + 4 * eta32 with hc1
    have hc10 : 0 ≤ c1 := by rw [hc1]; nlinarith [hu, hη, hmT0, haM0]
    have hsa1 := rstd_tol_sqrt_sum_add (n := d.D) hD0
      (a := fun j => meanTol d init t + u32 * |sR d init t j|)
      (b := fun _ => c1)
      (fun j _ => by
        have h1 : (0:ℝ) ≤ |sR d init t j| := abs_nonneg _
        nlinarith [hu, hmT0, h1]) (fun j _ => hc10)
    have e1 : (∑ j ∈ Finset.range d.D, (EBj j) ^ 2)
        = ∑ j ∈ Finset.range d.D,
            ((meanTol d init t + u32 * |sR d init t j|) + c1) ^ 2 := by
      apply Finset.sum_congr rfl
      intro j hj
      rw [hEBj, hc1]; ring
    have e2 : Real.sqrt ((∑ j ∈ Finset.range d.D,
        (meanTol d init t + u32 * |sR d init t j|) ^ 2) / (d.D : ℝ))
        ≤ meanTol d init t + u32 * rmsS d init t := by
      have hsa2 := rstd_tol_sqrt_sum_add (n := d.D) hD0
        (a := fun _ => meanTol d init t) (b := fun j => u32 * |sR d init t j|)
        (fun j _ => hmT0) (fun j _ => by positivity)
      have e2a : Real.sqrt ((∑ j ∈ Finset.range d.D, (meanTol d init t) ^ 2) / (d.D : ℝ))
          = meanTol d init t := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        rw [show ((d.D : ℝ) * meanTol d init t ^ 2) / (d.D : ℝ)
            = meanTol d init t ^ 2 from by field_simp]
        exact Real.sqrt_sq hmT0
      have e2b := hscale u32 hu (fun j => |sR d init t j|)
      have e2c : Real.sqrt ((∑ j ∈ Finset.range d.D, (|sR d init t j|) ^ 2) / (d.D : ℝ))
          = rmsS d init t := by
        rw [rmsS]
        congr 1
        congr 1
        exact Finset.sum_congr rfl fun j _ => by rw [sq_abs]
      rw [e2a, e2b, e2c] at hsa2
      exact hsa2
    have e3 : Real.sqrt ((∑ j ∈ Finset.range d.D, c1 ^ 2) / (d.D : ℝ)) = c1 := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      rw [show ((d.D : ℝ) * c1 ^ 2) / (d.D : ℝ) = c1 ^ 2 from by field_simp]
      exact Real.sqrt_sq hc10
    have h2 : Real.sqrt ((∑ j ∈ Finset.range d.D,
        ((meanTol d init t + u32 * |sR d init t j|) + c1) ^ 2) / (d.D : ℝ))
        ≤ (meanTol d init t + u32 * rmsS d init t) + c1 := by
      refine hsa1.trans ?_
      linarith [e2, e3]
    rw [e1]
    refine h2.trans ?_
    rw [hEB, hc1]
    ring_nf
    exact le_refl _
  -- assemble
  have hsub := rstd_tol_sqrt_sum_sub (n := d.D) hD0
    (x := fun j => (x j).toReal) (d := fun j => sR d init t j - meanRef d init t)
  rw [hddef] at hsub
  refine hsub.trans ?_
  refine hmono.trans ?_
  refine hB.trans ?_
  have h1 : (0:ℝ) ≤ 1 + 2 * u32 := by linarith
  have h2 := mul_le_mul_of_nonneg_left hEBS h1
  linarith


/-- The real arithmetic of the error of `var + eps`. -/
theorem rstd_tol_be_arith (L W u η b0 : ℝ) (hW : 0 < W) (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hb0 : 0 ≤ b0)
    (hcoef : 4 * ((3.1*L+6)*u) ≤ 3) (hub : 4*u*b0 + 12*η ≤ b0)
    (h7 : 7 + (3.1*L+6) ≤ 4*L+16) (hu2 : 2 * u * u ≤ u) (_hu3 : u ≤ 1)
    (hLc : 0 ≤ 3.1*L+6) :
    (1 + 2*u) * (u*W + (1+u)*((3.1*L+6)*u*W + b0) + 4*η) + 2*u*W
      ≤ (4*L+16)*u*W + 2*b0 := by
  have huW : 0 ≤ u * W := mul_nonneg hu (le_of_lt hW)
  have hu1u : 2*u*(1+u) ≤ 3*u := by linarith [hu2]
  have hB : 2*u*(u*W) ≤ u*W := by
    have e : 2*u*(u*W) = (2*u*u)*W := by ring
    rw [e]; exact mul_le_mul_of_nonneg_right hu2 (le_of_lt hW)
  have hX0 : 0 ≤ (3.1*L+6)*u*W + b0 := by
    have h1 : 0 ≤ (3.1*L+6)*u := mul_nonneg hLc hu
    have h2 : 0 ≤ ((3.1*L+6)*u)*W := mul_nonneg h1 (le_of_lt hW)
    have h3 : ((3.1*L+6)*u)*W = (3.1*L+6)*u*W := by ring
    linarith
  have hC : 2*u*((1+u)*((3.1*L+6)*u*W + b0)) ≤ 3*u*((3.1*L+6)*u*W + b0) := by
    have e : 2*u*((1+u)*((3.1*L+6)*u*W + b0)) = (2*u*(1+u))*((3.1*L+6)*u*W + b0) := by ring
    rw [e]; exact mul_le_mul_of_nonneg_right hu1u hX0
  have hD : 8*u*η ≤ 8*η := by nlinarith [‹u ≤ 1›, hη]
  have hstep1 : (1+2*u)*(u*W + (1+u)*((3.1*L+6)*u*W + b0) + 4*η) + 2*u*W
      ≤ 4*(u*W) + ((3.1*L+6)*u*W + b0) + 4*u*((3.1*L+6)*u*W + b0) + 12*η := by
    have hexp : (1+2*u)*(u*W + (1+u)*((3.1*L+6)*u*W + b0) + 4*η)
        = (u*W + (1+u)*((3.1*L+6)*u*W + b0) + 4*η)
          + 2*u*(u*W) + 2*u*((1+u)*((3.1*L+6)*u*W + b0)) + 8*u*η := by ring
    linarith [hexp, hB, hC, hD]
  have hA : 4*u*((3.1*L+6)*u*W) ≤ 3*(u*W) := by
    have e : 4*u*((3.1*L+6)*u*W) = (4*((3.1*L+6)*u))*(u*W) := by ring
    rw [e]; exact mul_le_mul_of_nonneg_right hcoef huW
  have h4u : 3*u*b0 ≤ 4*u*b0 := by nlinarith [hb0, hu]
  have h7' : 7*(u*W) + (3.1*L+6)*u*W ≤ (4*L+16)*u*W := by
    have e : 7*(u*W) + (3.1*L+6)*u*W = (7 + (3.1*L+6)) * (u*W) := by ring
    have e2 : (4*L+16)*u*W = (4*L+16)*(u*W) := by ring
    rw [e, e2]
    exact mul_le_mul_of_nonneg_right h7 huW
  linarith [hstep1, hA, h4u, hub, h7']


/-- The `tl.add` of the variance and `eps`: relative to `W = V + eps`, within `(4L+16)u` and
`2^-135`. -/
theorem rstd_tol_be {d : Inst} (hd : d.ok) {V : ℝ} {bva epsV be : F32}
    (hV0 : 0 ≤ V)
    (heps0 : 0 ≤ epsV.toReal) (hepspos : 0 < epsV.toReal)
    (hepsle : epsV.toReal ≤ (2 : ℝ) ^ (126 : ℤ))
    (hbva : |bva.toReal - V|
      ≤ (3.1 * (depth d : ℝ) + 6) * u32 * V + (2 : ℝ) ^ (-136 : ℤ))
    (hbvabd : |bva.toReal| ≤ (2 : ℝ) ^ 118)
    (hbvafin : bva.isFinite = true) (hepsfin : epsV.isFinite = true)
    (hadd : AddRel true bva epsV be) :
    be.isFinite = true ∧
    |be.toReal - (V + epsV.toReal)|
      ≤ (4 * (depth d : ℝ) + 16) * u32 * (V + epsV.toReal) + (2 : ℝ) ^ (-135 : ℤ) := by
  have hu : 0 ≤ u32 := le_of_lt u32_pos
  have hη : 0 ≤ eta32 := le_of_lt eta32_pos
  have hL : (depth d : ℝ) ≤ 10 := by exact_mod_cast rstd_tol_depth_le hd
  set W : ℝ := V + epsV.toReal with hW
  have hW0 : 0 < W := by rw [hW]; linarith
  have hVW : V ≤ W := by rw [hW]; linarith
  -- the `ℚ` magnitude condition of the split
  have hM : |bva.toRat| + |epsV.toRat|
      ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
    have h1 : |bva.toRat| ≤ (2 : ℚ) ^ 118 := by
      have h2 : |(bva.toRat : ℝ)| ≤ (2 : ℝ) ^ 118 := by simpa [FloatV.toReal] using hbvabd
      exact_mod_cast h2
    have h2 : |epsV.toRat| ≤ (2 : ℚ) ^ 126 := by
      have h3 : |(epsV.toRat : ℝ)| ≤ (2 : ℝ) ^ 126 := by
        have h4 : |epsV.toReal| ≤ (2 : ℝ) ^ 126 := by
          rw [abs_of_nonneg heps0]; exact hepsle
        simpa [FloatV.toReal] using h4
      exact_mod_cast h3
    have hnum : (2 : ℚ) ^ 118 + (2 : ℚ) ^ 126
        ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
      rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
    linarith
  obtain ⟨hfin, c, e, hc, he, heq⟩ := rstd_tol_addrel_split_real hadd hbvafin hepsfin hM
  refine ⟨hfin, ?_⟩
  set d1 : ℝ := bva.toReal - V with hd1
  have hd1W : |d1| ≤ (3.1 * (depth d : ℝ) + 6) * u32 * W + (2 : ℝ) ^ (-136 : ℤ) := by
    have hb : |d1| ≤ (3.1 * (depth d : ℝ) + 6) * u32 * V + (2 : ℝ) ^ (-136 : ℤ) := by
      rw [hd1]; exact hbva
    have h1 : (3.1 * (depth d : ℝ) + 6) * u32 * V ≤ (3.1 * (depth d : ℝ) + 6) * u32 * W := by
      apply mul_le_mul_of_nonneg_left hVW
      have h2 : (0:ℝ) ≤ 3.1 * (depth d : ℝ) + 6 := by linarith
      exact mul_nonneg h2 hu
    linarith
  -- the magnitude of `e`
  have hmax : max |bva.toReal| |epsV.toReal| ≤ W + |d1| := by
    have h1 : |bva.toReal| ≤ W + |d1| := by
      have h3 := abs_add_le V d1
      have h4 : V + d1 = bva.toReal := by rw [hd1]; ring
      rw [h4, abs_of_nonneg hV0] at h3
      linarith
    have h2 : |epsV.toReal| ≤ W + |d1| := by
      have h5 : epsV.toReal ≤ W := by rw [hW]; linarith
      rw [abs_of_nonneg heps0]; linarith [abs_nonneg d1]
    exact max_le h1 h2
  have hebd : |e| ≤ u32 * (W + |d1|) + 4 * eta32 := by
    refine he.trans ?_
    have h1 : u32 * max |bva.toReal| |epsV.toReal| ≤ u32 * (W + |d1|) :=
      mul_le_mul_of_nonneg_left hmax hu
    linarith
  -- assemble the size of `be`
  have heq' : be.toReal = (W + d1 + e) * (1 + c) := by
    rw [heq, hW]; ring
  have hmain : |be.toReal - W| ≤ (1 + 2 * u32) * (|d1| + |e|) + 2 * u32 * W := by
    have e1 : be.toReal - W = (d1 + e) + c * (W + d1 + e) := by rw [heq']; ring
    rw [e1]
    have h1 : |(d1 + e) + c * (W + d1 + e)|
        ≤ |d1 + e| + |c| * |W + d1 + e| := by
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul]
    have h2 : |d1 + e| ≤ |d1| + |e| := abs_add_le _ _
    have h3 : |W + d1 + e| ≤ W + (|d1| + |e|) := by
      have h4 : W + d1 + e = W + (d1 + e) := by ring
      rw [h4]
      have h5 := abs_add_le W (d1 + e)
      rw [abs_of_pos hW0] at h5
      linarith [h5, abs_add_le d1 e]
    have h5 : |c| * |W + d1 + e| ≤ (2 * u32) * (W + (|d1| + |e|)) := by
      apply mul_le_mul hc h3 (abs_nonneg _) (by linarith)
    have h6 : (0:ℝ) ≤ |d1| + |e| := by positivity
    nlinarith [h1, h2, h5, hu, hW0, h6, abs_nonneg d1, abs_nonneg e]
  -- numeric conclusion
  set b0 : ℝ := (2 : ℝ) ^ (-136 : ℤ) with hb0def
  have hb0nn : 0 ≤ b0 := by rw [hb0def]; norm_num
  set X : ℝ := (3.1 * (depth d : ℝ) + 6) * u32 * W + b0 with hXdef
  have hX0 : 0 ≤ X := by
    have h3 : (0:ℝ) ≤ 3.1 * (depth d : ℝ) + 6 := by linarith
    have h4 : (0:ℝ) ≤ u32 * W := mul_nonneg hu (le_of_lt hW0)
    have h5 : (0:ℝ) ≤ (3.1 * (depth d : ℝ) + 6) * (u32 * W) := mul_nonneg h3 h4
    have h6 : (3.1 * (depth d : ℝ) + 6) * (u32 * W) = X - b0 := by rw [hXdef]; ring
    linarith
  have hE1 : |d1| + |e| ≤ u32 * W + (1 + u32) * X + 4 * eta32 := by
    have h1 : |d1| ≤ X := by
      have h2 : |d1| ≤ (3.1 * (depth d : ℝ) + 6) * u32 * W + b0 := by
        rw [hb0def]; exact hd1W
      rw [hXdef]; convert h2 using 2
    have h2 : |e| ≤ u32 * (W + |d1|) + 4 * eta32 := hebd
    have h3 : u32 * |d1| ≤ u32 * X := mul_le_mul_of_nonneg_left h1 hu
    nlinarith [h1, h2, h3, hu, abs_nonneg d1, abs_nonneg e]
  have hscaled : (1 + 2 * u32) * (|d1| + |e|) + 2 * u32 * W
      ≤ (1 + 2 * u32) * (u32 * W + (1 + u32) * X + 4 * eta32) + 2 * u32 * W := by
    have hnn : (0:ℝ) ≤ 1 + 2 * u32 := by linarith
    linarith [mul_le_mul_of_nonneg_left hE1 hnn]
  -- the abstract arithmetic
  have hcoef : 4 * ((3.1 * (depth d : ℝ) + 6) * u32) ≤ 3 := by unfold u32; nlinarith [hL]
  have hub : 4 * u32 * b0 + 12 * eta32 ≤ b0 := by
    rw [hb0def]; unfold u32 eta32; norm_num
  have h7 : 7 + (3.1 * (depth d : ℝ) + 6) ≤ 4 * (depth d : ℝ) + 16 := by linarith
  have hu2 : 2 * u32 * u32 ≤ u32 := by unfold u32; norm_num
  have hu3 : u32 ≤ 1 := by unfold u32; norm_num
  have hLc : 0 ≤ 3.1 * (depth d : ℝ) + 6 := by linarith
  have hnum := rstd_tol_be_arith (depth d : ℝ) W u32 eta32 b0 hW0 hu hη hb0nn
    hcoef hub h7 hu2 hu3 hLc
  have hB : (2 : ℝ) ^ (-135 : ℤ) = 2 * b0 := by rw [hb0def]; norm_num
  rw [hB]
  exact hmain.trans (hscaled.trans hnum)


/-- `(1 + 2^-22)√s + 2^-149` stays in `float32` range for `s ≤ 2^128`. -/
theorem rstd_tol_room (s : ℝ) (_h0 : 0 ≤ s) (hs : s ≤ (2:ℝ)^(128:ℤ)) :
    (1 + (2:ℝ)^(-22:ℤ)) * |Real.sqrt s| + (2:ℝ)^(-149:ℤ)
      ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  have hs0 : 0 ≤ Real.sqrt s := Real.sqrt_nonneg s
  rw [abs_of_nonneg hs0]
  have hge1 : (1:ℝ) ≤ (2:ℝ)^(64:ℤ) := by norm_num
  have hsq : (2:ℝ)^(128:ℤ) = (2:ℝ)^(64:ℤ) * (2:ℝ)^(64:ℤ) := by
    rw [← zpow_add₀ (by norm_num : (2:ℝ) ≠ 0)]
    norm_num
  have hs' : s ≤ (2:ℝ)^(64:ℤ) * (2:ℝ)^(64:ℤ) := by
    rw [← hsq]; exact hs
  have h1 : Real.sqrt s ≤ (2:ℝ)^(64:ℤ) := by
    calc Real.sqrt s ≤ Real.sqrt ((2:ℝ)^(64:ℤ) * (2:ℝ)^(64:ℤ)) := Real.sqrt_le_sqrt hs'
      _ = (2:ℝ)^(64:ℤ) := Real.sqrt_mul_self (by norm_num)
  have hc : (1 + (2:ℝ)^(-22:ℤ)) ≤ 2 := by norm_num
  have hc2 : (1 + (2:ℝ)^(-22:ℤ)) * Real.sqrt s ≤ 2 * (2:ℝ)^(64:ℤ) := by
    have h := mul_le_mul hc h1 hs0 (by norm_num)
    linarith
  have h2 : 2 * (2:ℝ)^(64:ℤ) = (2:ℝ)^(65:ℤ) := by norm_num
  have h3 : (2:ℝ)^(65:ℤ) + (2:ℝ)^(-149:ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]
    push_cast
    norm_num
  linarith

/-- `tl.sqrt` at `be`: finite and within `4u·√be + 2^-149`. -/
theorem rstd_tol_sqrt_acc {P : Platform} (hsqrt : Device.SqrtF32 P)
    {be bsq : F32} (hfin : be.isFinite = true)
    (hdom : be.toReal = 0 ∨ (2:ℝ)^(-126:ℤ) ≤ be.toReal)
    (hroom : (1 + (2:ℝ)^(-22:ℤ)) * |Real.sqrt be.toReal| + (2:ℝ)^(-149:ℤ)
      ≤ ((Format.binary32.maxVal : ℚ) : ℝ))
    (hmath : P.math .sqrt .binary32 be bsq) :
    bsq.isFinite = true ∧
      |bsq.toReal - Real.sqrt be.toReal| ≤ 4 * u32 * Real.sqrt be.toReal + (2:ℝ)^(-149:ℤ) := by
  obtain ⟨ref, href, H⟩ := hsqrt
  have hsome : MathFn.ref .sqrt = some Real.sqrt := rfl
  rw [hsome] at href
  obtain rfl := Option.some.inj href
  have hnn : (0:ℝ) ≤ be.toReal := by
    rcases hdom with h | h
    · rw [h]
    · have hz : (0:ℝ) ≤ (2:ℝ)^(-126:ℤ) := by norm_num
      linarith
  have hnat : be.toReal ∈ MathFn.natDom .sqrt := hnn
  have hmem : be.toReal ∈ Device.sqrtDom := by
    show be.toReal = 0 ∨ (2:ℝ)^(-126:ℤ) ≤ be.toReal
    exact hdom
  obtain ⟨hfinr, herr⟩ := H be bsq hmath hfin hmem hnat hroom
  refine ⟨hfinr, ?_⟩
  have h4 : (2:ℝ)^(-22:ℤ) = 4 * u32 := by unfold u32; norm_num
  rw [abs_of_nonneg (Real.sqrt_nonneg _)] at herr
  rw [h4] at herr
  exact herr

/-- `1/·` (the default division) at `1` and `bsq`. -/
theorem rstd_tol_div_acc {P : Platform} (hdivf : Device.DivFull P)
    {one bsq brstd : F32} (honefin : one.isFinite = true) (hone : one.toRat = 1)
    (hbsqfin : bsq.isFinite = true) (hbsq0 : bsq.toRat ≠ 0)
    (hq : |one.toRat / bsq.toRat| ≤ (2:ℚ)^(126:ℤ))
    (hdivr : P.div .binary32 one bsq brstd) :
    brstd.isFinite = true ∧
      |brstd.toReal - 1 / bsq.toReal| ≤ 4 * u32 * |1 / bsq.toReal| + (2:ℝ)^(-148:ℤ) := by
  have hacc : Device.DivAcc P .binary32 (2^(126:ℤ)) (2^(-22:ℤ)) (2^(-148:ℤ)) := hdivf
  obtain ⟨hfin, herr⟩ := hacc one bsq brstd hdivr honefin hbsqfin hbsq0 hq
  refine ⟨hfin, ?_⟩
  have h' : ((|brstd.toRat - one.toRat / bsq.toRat| : ℚ) : ℝ)
      ≤ ((2^(-22:ℤ) * |one.toRat / bsq.toRat| + 2^(-148:ℤ) : ℚ) : ℝ) := by
    exact_mod_cast herr
  push_cast at h'
  simp only [FloatV.toReal] at h' ⊢
  have honeR : ((one.toRat : ℚ) : ℝ) = 1 := by rw [hone]; norm_num
  rw [honeR] at h'
  have h4 : (2:ℝ)^(-22:ℤ) = 4 * u32 := by unfold u32; norm_num
  rw [h4] at h'
  exact h'

/-- Real-arithmetic step: a lower bound on `epsV` of the form `epsR - (u*epsR + η)` gives `epsR * (1 - 4u) ≤ epsV`. -/
theorem rstd_tol_step {epsR epsV u η c : ℝ} (hu : 0 ≤ u) (heR : 0 ≤ epsR)
    (hle : epsR - (u*epsR + η) ≤ epsV)
    (he : η ≤ u*epsR) (hc : 4*u = c) : epsR * (1 - c) ≤ epsV := by
  have h4 : u*epsR + η ≤ 4*(u*epsR) := by nlinarith [he, hu, heR]
  have e : epsR * (1 - c) = epsR - 4*(u*epsR) := by rw [← hc]; ring
  rw [e]; linarith

/-- `2^-135 / W0 ≤ (2 + 2^-19)·(2^-136/epsR)` when `W0 ≥ epsV` is within the `eps` rounding of
`epsR`. -/
theorem rstd_tol_rho {epsR epsV W0 : ℝ} (hepsR : 0 < epsR)
    (hge : (2:ℝ)^(-125:ℤ) ≤ epsR)
    (heps : |epsV - epsR| ≤ u32 * epsR + eta32) (hW0 : epsV ≤ W0) :
    (2:ℝ)^(-135:ℤ) / W0 ≤ (2 + (2:ℝ)^(-19:ℤ)) * ((2:ℝ)^(-136:ℤ) / epsR) := by
  have hc : (0:ℝ) < 1 - (2:ℝ)^(-22:ℤ) := by norm_num
  have he : eta32 ≤ u32 * epsR := by
    have h1 : u32 * (2:ℝ)^(-125:ℤ) ≤ u32 * epsR :=
      mul_le_mul_of_nonneg_left hge (by unfold u32; norm_num)
    have h2 : eta32 ≤ u32 * (2:ℝ)^(-125:ℤ) := by unfold u32 eta32; norm_num
    exact h2.trans h1
  have hle : epsR - (u32*epsR + eta32) ≤ epsV := by
    have h1 : -(u32*epsR + eta32) ≤ epsV - epsR := (abs_le.mp heps).1
    have h2 : -(u32*epsR + eta32) + epsR ≤ epsV := by
      have h := add_le_add_right h1 epsR
      have h4 : epsR + (epsV - epsR) = epsV := by ring
      rw [h4] at h
      have h5 : epsR + -(u32*epsR + eta32) = -(u32*epsR + eta32) + epsR := by ring
      rw [h5] at h
      exact h
    have h3 : -(u32*epsR + eta32) + epsR = epsR - (u32*epsR + eta32) := by ring
    rw [h3] at h2
    exact h2
  have hstep : epsR * (1 - (2:ℝ)^(-22:ℤ)) ≤ epsV :=
    rstd_tol_step (le_of_lt u32_pos) (le_of_lt hepsR) hle he (by unfold u32; norm_num)
  have hpos : 0 < epsR * (1 - (2:ℝ)^(-22:ℤ)) := mul_pos hepsR hc
  have hleW : epsR * (1 - (2:ℝ)^(-22:ℤ)) ≤ W0 := le_trans hstep hW0
  have h1 : (2:ℝ)^(-135:ℤ) / W0 ≤ (2:ℝ)^(-135:ℤ) / (epsR * (1 - (2:ℝ)^(-22:ℤ))) :=
    div_le_div_of_nonneg_left (by norm_num) hpos hleW
  have h2 : (2:ℝ)^(-135:ℤ) / (epsR * (1 - (2:ℝ)^(-22:ℤ)))
      = (2 / (1 - (2:ℝ)^(-22:ℤ))) * ((2:ℝ)^(-136:ℤ) / epsR) := by
    field_simp
  have hcoef : 2 / (1 - (2:ℝ)^(-22:ℤ)) ≤ 2 + (2:ℝ)^(-19:ℤ) := by norm_num
  have hnn : (0:ℝ) ≤ (2:ℝ)^(-136:ℤ) / epsR := by positivity
  rw [h2] at h1
  exact h1.trans (mul_le_mul_of_nonneg_right hcoef hnn)

/-- The whole numeric content of the well-conditioned branch. -/
theorem rstd_tol_A2 {L u g2 ρ y c : ℝ} (hL0 : 0 ≤ L) (hL10 : L ≤ 10) (hu0 : 0 ≤ u)
    (hu24 : u^2 ≤ u * (1/16777216)) (hg0 : 0 ≤ g2) (hgg : g2 ≤ 1/2048)
    (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ (2 + (2:ℝ)^(-19:ℤ)) * g2)
    (hy0 : 0 ≤ y) (hyc : y ≤ c) (hc4 : 4*c ≤ 1) :
    (1 + 2*y) * (((4*L+16)*u + ρ)/2 + ((4*L+16)*u + ρ)^2 + 13*u) + 2*y + 6*u
      ≤ 2*(c + ((2*L+20)*u + g2)) := by
  set c0 : ℝ := 4*L+16 with hc0def
  set A : ℝ := (c0*u + ρ)/2 + (c0*u + ρ)^2 + 13*u with hAdef
  have hc00 : 0 ≤ c0 := by rw [hc0def]; linarith
  have hc056 : c0 ≤ 56 := by rw [hc0def]; linarith
  have hcu0 : 0 ≤ c0*u := mul_nonneg hc00 hu0
  have hcuρ : 0 ≤ c0*u + ρ := by linarith
  have hA0 : 0 ≤ A := by
    rw [hAdef]
    have h1 : 0 ≤ (c0*u + ρ)/2 := by linarith
    have h2 : 0 ≤ (c0*u + ρ)^2 := sq_nonneg _
    linarith
  -- the three bounds on the pieces of `A`
  have hρ2 : (3/2)*(ρ/2) ≤ (151/100)*g2 := by
    have h1 : (3/2)*(ρ/2) = (3/4)*ρ := by ring
    rw [h1]
    have h2 : (3/4)*ρ ≤ (3/4)*((2 + (2:ℝ)^(-19:ℤ)) * g2) :=
      mul_le_mul_of_nonneg_left hρ (by norm_num)
    have h3 : (3/4)*((2 + (2:ℝ)^(-19:ℤ)) * g2) = ((3/4)*(2 + (2:ℝ)^(-19:ℤ))) * g2 := by ring
    rw [h3] at h2
    have h4 : (3/4)*(2 + (2:ℝ)^(-19:ℤ)) ≤ 151/100 := by norm_num
    exact le_trans h2 (le_trans (mul_le_mul_of_nonneg_right h4 hg0) (by linarith))
  have hsq2 : (3/2)*((c0*u)^2) ≤ (2/10)*u := by
    have h1 : (c0*u)^2 ≤ u/8 := by
      have e1 : (c0*u)^2 = c0^2 * u^2 := by ring
      have h2 : c0^2 ≤ 56^2 := by nlinarith
      have h3 : c0^2 * u^2 ≤ 56^2 * u^2 := mul_le_mul_of_nonneg_right h2 (sq_nonneg u)
      have h4 : 56^2 * u^2 ≤ 56^2 * (u * (1/16777216)) :=
        mul_le_mul_of_nonneg_left hu24 (by norm_num)
      have h5 : 56^2 * (u * (1/16777216)) ≤ u/8 := by
        have e : 56^2 * (u * (1/16777216)) = (56^2 * (1/16777216)) * u := by ring
        rw [e]
        have hc2 : (56:ℝ)^2 * (1/16777216) ≤ 1/8 := by norm_num
        have h6 := mul_le_mul_of_nonneg_right hc2 hu0
        linarith
      rw [e1]
      linarith
    have e2 : (3/2)*((c0*u)^2) ≤ (3/2)*(u/8) := mul_le_mul_of_nonneg_left h1 (by norm_num)
    linarith
  have hcross2 : (3/2)*(2*(c0*u)*ρ) ≤ (2/10)*u := by
    have h1 : c0*u ≤ 56*u := mul_le_mul_of_nonneg_right hc056 hu0
    have h2 : 2*(c0*u)*ρ ≤ 2*(56*u)*((2 + (2:ℝ)^(-19:ℤ)) * g2) := by
      have h3 : 2*(c0*u) ≤ 2*(56*u) := by linarith
      have h4 : (0:ℝ) ≤ 2*(56*u) := by linarith
      exact mul_le_mul h3 hρ hρ0 h4
    have h5 : 2*(56*u)*((2 + (2:ℝ)^(-19:ℤ)) * g2) = (112*(2 + (2:ℝ)^(-19:ℤ)))*(u*g2) := by ring
    rw [h5] at h2
    have h6 : 112*(2 + (2:ℝ)^(-19:ℤ)) ≤ 225 := by norm_num
    have hug : 0 ≤ u*g2 := mul_nonneg hu0 hg0
    have h7 := mul_le_mul_of_nonneg_right h6 hug
    have h8 : 225*(u*g2) ≤ (1/8)*u := by
      have h9 : 225*g2 ≤ 1/8 := by nlinarith [hgg]
      calc 225*(u*g2) = u*(225*g2) := by ring
        _ ≤ u*(1/8) := mul_le_mul_of_nonneg_left h9 hu0
        _ = (1/8)*u := by ring
    nlinarith [h2, h7, h8]
  have hρsq2 : (3/2)*ρ^2 ≤ (2/100)*g2 := by
    have h2 : (0:ℝ) ≤ (2 + (2:ℝ)^(-19:ℤ)) * g2 := by positivity
    have h1 : ρ^2 ≤ ((2 + (2:ℝ)^(-19:ℤ)) * g2)^2 := by nlinarith [hρ0, hρ, h2]
    have h3 : ((2 + (2:ℝ)^(-19:ℤ)) * g2)^2 = (2 + (2:ℝ)^(-19:ℤ))^2 * g2^2 := by ring
    rw [h3] at h1
    have h4 : (2 + (2:ℝ)^(-19:ℤ))^2 ≤ 5 := by norm_num
    have h5 : (2 + (2:ℝ)^(-19:ℤ))^2 * g2^2 ≤ 5 * g2^2 :=
      mul_le_mul_of_nonneg_right h4 (sq_nonneg g2)
    have h6 : 5 * g2^2 ≤ (1/75)*g2 := by
      have h7 : 5 * g2 ≤ 1/75 := by nlinarith [hgg]
      calc 5 * g2^2 = (5*g2) * g2 := by ring
        _ ≤ (1/75) * g2 := mul_le_mul_of_nonneg_right h7 hg0
    have h8 : ρ^2 ≤ (1/75)*g2 := le_trans h1 (le_trans h5 h6)
    have h9 := mul_le_mul_of_nonneg_left h8 (by norm_num : (0:ℝ) ≤ 3/2)
    linarith
  -- assemble
  have hrest : (3/2)*(ρ/2 + (c0*u)^2 + 2*(c0*u)*ρ + ρ^2) ≤ (25/10)*u + 2*g2 := by
    have e : (3/2)*(ρ/2 + (c0*u)^2 + 2*(c0*u)*ρ + ρ^2)
        = (3/2)*(ρ/2) + (3/2)*((c0*u)^2) + (3/2)*(2*(c0*u)*ρ) + (3/2)*ρ^2 := by ring
    rw [e]
    linarith
  have hexp : (3/2)*A + 6*u
      = (3*L + 75/2)*u + (3/2)*(ρ/2 + (c0*u)^2 + 2*(c0*u)*ρ + ρ^2) := by
    rw [hAdef, hc0def]
    ring
  have hA3 : (3/2)*A + 6*u ≤ 2*((2*L+20)*u + g2) := by
    rw [hexp]
    nlinarith [hrest, hL0, hu0, hg0]
  have h1 : (1 + 2*y) * A ≤ (3/2)*A := by nlinarith [hA0, hy0, hyc, hc4]
  change (1 + 2*y) * A + 2*y + 6*u ≤ 2*(c + ((2*L+20)*u + g2))
  linarith

/-- `1/√(1-δ) ≤ 1+δ` for `0 ≤ δ` with `δ + δ² ≤ 1`. -/
theorem rstd_tol_one_div_sqrt_one_sub {δ : ℝ} (h0 : 0 ≤ δ) (h1 : δ + δ ^ 2 ≤ 1) :
    1 / Real.sqrt (1 - δ) ≤ 1 + δ := by
  have hpos : 0 < 1 - δ := by nlinarith [sq_nonneg δ]
  have hs : Real.sqrt (1 - δ) * Real.sqrt (1 - δ) = 1 - δ :=
    Real.mul_self_sqrt (le_of_lt hpos)
  have hs0 : 0 < Real.sqrt (1 - δ) := Real.sqrt_pos.2 hpos
  have h3 : 1 ≤ (1 + δ)^2 * (1 - δ) := by nlinarith [h1]
  have hsq : Real.sqrt (1 - δ)^2 = 1 - δ := by rw [sq]; exact hs
  have h4 : ((1 + δ) * Real.sqrt (1 - δ))^2 = (1 + δ)^2 * (1 - δ) := by
    rw [mul_pow, hsq]
  have h5 : 0 ≤ (1 + δ) * Real.sqrt (1 - δ) := by nlinarith [h0, hs0]
  have h2 : 1 ≤ (1 + δ) * Real.sqrt (1 - δ) := by nlinarith [h3, h4, h5]
  rw [div_le_iff₀ hs0]
  linarith

/-- The bound `y = E·rstdRef ≤ cond + 3u` of the well-conditioned branch. -/
theorem rstd_tol_y_le {u mT rms aM sqrtV R c η : ℝ}
    (hu0 : 0 ≤ u) (hu16 : u ≤ 1/16)
    (hmT : 0 ≤ mT) (hrms : 0 ≤ rms) (haM : 0 ≤ aM) (_hsv : 0 ≤ sqrtV)
    (hη : 0 ≤ η) (hR : 0 ≤ R) (hsvR : sqrtV * R ≤ 1)
    (hc : c = (mT + 2*u*(aM + rms)) * R) (heta : 5*η*R ≤ u/8)
    (hc4 : 4*c ≤ 1) :
    (2*u*sqrtV + (1+2*u)*(mT + u*(rms + aM + mT) + 4*η)) * R ≤ c + 3*u := by
  have h1 : 2*u*sqrtV*R ≤ 2*u := by
    have h2 : 2*u*(sqrtV*R) ≤ 2*u*1 := mul_le_mul_of_nonneg_left hsvR (by linarith)
    nlinarith [h2]
  have hx : 0 ≤ u*(aM+rms)*R := by
    have h3 : 0 ≤ aM + rms := by linarith
    have h4 : 0 ≤ u*(aM+rms) := mul_nonneg hu0 h3
    exact mul_nonneg h4 hR
  have hMR : (mT + u*(rms + aM + mT) + 4*η)*R
      ≤ c + u*(mT*R) + 4*(η*R) := by
    rw [hc]; nlinarith [hx]
  have hmTR : mT*R ≤ c := by rw [hc]; nlinarith [hx]
  have hu_mTR : u*(mT*R) ≤ u*c := mul_le_mul_of_nonneg_left hmTR hu0
  have huc : u*c ≤ u*(1/4) := mul_le_mul_of_nonneg_left (by linarith) hu0
  have hηR : 0 ≤ η*R := mul_nonneg hη hR
  have heta4 : 4*(η*R) ≤ u/8 := by nlinarith [heta, hηR]
  have hMR2 : (mT + u*(rms + aM + mT) + 4*η)*R ≤ c + u*c + u/8 := by linarith
  have hMR0 : 0 ≤ (mT + u*(rms + aM + mT) + 4*η)*R := by
    have h3 : 0 ≤ rms + aM + mT := by linarith
    have h4 : 0 ≤ u*(rms + aM + mT) := mul_nonneg hu0 h3
    have h5 : 0 ≤ 4*η := by linarith
    have h6 : 0 ≤ mT + u*(rms + aM + mT) + 4*η := by linarith
    exact mul_nonneg h6 hR
  have hstep : (1+2*u)*((mT + u*(rms + aM + mT) + 4*η)*R) ≤ c + u := by
    have h2u : 2*u*((mT + u*(rms + aM + mT) + 4*η)*R) ≤ 2*u*(c + u*c + u/8) :=
      mul_le_mul_of_nonneg_left hMR2 (by linarith)
    have hc0 : 0 ≤ c := by nlinarith [hc4]
    have e1 : 2*u*c ≤ u/2 := by nlinarith [hc4, hu0]
    have e2 : 2*u*(u*c) ≤ u/16 := by nlinarith [hc4, hu0, hu16, hc0]
    have e3 : 2*u*(u/8) = u^2/4 := by ring
    have e4 : u^2/4 ≤ u/16 := by nlinarith [hu0, hu16]
    have h2ub : 2*u*(c + u*c + u/8) ≤ (5/8)*u := by nlinarith [e1, e2, e3, e4]
    have hsum : (1+2*u)*((mT + u*(rms + aM + mT) + 4*η)*R)
        = (mT + u*(rms + aM + mT) + 4*η)*R
          + 2*u*((mT + u*(rms + aM + mT) + 4*η)*R) := by ring
    rw [hsum]
    nlinarith [hMR2, h2u, h2ub, huc, hu_mTR, heta4]
  nlinarith [h1, hstep, hMR0]

/-- The final numeric inequality of the well-conditioned branch. -/
theorem rstd_tol_final_num {L u g2 c Θ ρ : ℝ}
    (hL0 : 0 ≤ L) (hL10 : L ≤ 10) (hu0 : 0 ≤ u) (hu6 : u ≤ 1/1000000)
    (hg0 : 0 ≤ g2) (hgg : g2 ≤ 1/2048)
    (hΘ0 : 0 ≤ Θ) (hΘ : Θ ≤ (4*L+16)*u*(1+3*u/2) + 3*u/2 + 2*g2)
    (hρ : ρ = Θ/2 + Θ^2 + 13*u)
    (hc0 : 0 ≤ c) (hc4 : 4*c ≤ 1) :
    2*c + 6*u + (2*c + 6*u)*ρ + ρ ≤ 2*(c + ((2*L+20)*u + g2)) := by
  have huL : (4*L+16)*u*(1+3*u/2) ≤ (4*L+16)*u + (1/10)*u := by
    have h5 : (4*L+16) ≤ 56 := by linarith
    have h6 : 0 ≤ u*(3*u/2) := by nlinarith [hu0]
    have h4 : (4*L+16)*(u*(3*u/2)) ≤ 56*(u*(3*u/2)) :=
      mul_le_mul_of_nonneg_right h5 h6
    have h7 : 56*(u*(3*u/2)) = 84*u^2 := by ring
    have h8 : 84*u^2 ≤ (1/10)*u := by nlinarith [hu0, hu6]
    have h9 : (4*L+16)*u*(1+3*u/2) = (4*L+16)*u + (4*L+16)*(u*(3*u/2)) := by ring
    rw [h9]
    linarith
  have hΘ1 : Θ ≤ (4*L+17.6)*u + 2*g2 := by linarith [hΘ, huL]
  have hΘ2 : Θ ≤ 58*u + 2*g2 := by nlinarith [hΘ1, hL0, hL10, hu0]
  have hP0 : 0 ≤ 58*u + 2*g2 := by nlinarith [hu0, hg0]
  have hΘsq : Θ^2 ≤ (58*u + 2*g2)^2 := by nlinarith [hΘ0, hΘ2, hP0]
  have hρ0 : 0 ≤ ρ := by rw [hρ]; nlinarith [hΘ0, hu0]
  have hsqexp : (58*u + 2*g2)^2 ≤ 17*u + g2/2 := by
    nlinarith [hu0, hu6, hg0, hgg]
  have hρb : ρ ≤ 60*u + 2*g2 := by rw [hρ]; nlinarith [hΘ2, hsqexp, hu0, hg0]
  -- the three small pieces
  have hA : 5046*u^2 ≤ (1/10)*u := by nlinarith [hu0, hu6]
  have hB : 348*u*g2 ≤ (1/10)*g2 := by nlinarith [hg0, hu0, hu6]
  have hC : 6*g2^2 ≤ (3/10)*g2 := by nlinarith [hg0, hgg]
  have hD : 6*u*ρ ≤ (11/10)*u := by nlinarith [hu0, hρb, hg0, hgg]
  have hkey : (3/2)*Θ^2 + 6*u*ρ ≤ (13/10)*u + (1/2)*g2 := by
    nlinarith [hΘsq, hA, hB, hC, hD]
  have hmain : (3/2)*ρ + 6*u ≤ (4*L+40)*u + 2*g2 := by
    rw [hρ]
    nlinarith [hΘ1, hkey, hL0, hu0, hg0]
  have h2c : 2*c*ρ ≤ ρ/2 := by nlinarith [hc4, hρ0]
  have hexp : 2*c + 6*u + (2*c + 6*u)*ρ + ρ
      = 2*c + (6*u + (3/2)*ρ + 6*u*ρ + (2*c*ρ - ρ/2)) := by ring
  rw [hexp]
  nlinarith [hmain, h2c]


/-- The ill-conditioned branch: `brstd` lies in `[0, 2/√eps]` and so does `rstdRef`. -/
theorem rstd_tol_bad {be bsq brstd R eps : ℝ}
    (heps0 : 0 < eps) (hge : (2:ℝ)^(-125:ℤ) ≤ eps) (hle_eps : eps ≤ (2:ℝ)^(126:ℤ))
    (hbe_lo : eps/2 ≤ be) (hbe_hi : be ≤ (2:ℝ)^(128:ℤ))
    (hbsq : |bsq - Real.sqrt be| ≤ 4*u32*Real.sqrt be + (2:ℝ)^(-149:ℤ))
    (hbrstd : |brstd - 1/bsq| ≤ 4*u32*|1/bsq| + (2:ℝ)^(-148:ℤ))
    (hR0 : 0 < R) (hRle : R ≤ 2/Real.sqrt eps) :
    |brstd - R| ≤ 2/Real.sqrt eps := by
  set t := Real.sqrt be with htdef
  have htpos : 0 < t := by rw [htdef]; exact Real.sqrt_pos.2 (by linarith)
  have hspe : 0 < Real.sqrt eps := Real.sqrt_pos.2 heps0
  have hu0 : (0:ℝ) ≤ u32 := le_of_lt u32_pos
  have hu5 : 5*u32 ≤ 1/2 := by unfold u32; norm_num
  have h5pos : 0 < 1 - 5*u32 := by linarith
  -- t ≥ 2^-63
  have ht63 : (2:ℝ)^(-63:ℤ) ≤ t := by
    have h1 : Real.sqrt ((2:ℝ)^(-126:ℤ)) ≤ t := by
      rw [htdef]
      apply Real.sqrt_le_sqrt
      have h2 : (2:ℝ)^(-126:ℤ) = (2:ℝ)^(-125:ℤ) * (1/2) := by norm_num
      have h3 := mul_le_mul_of_nonneg_right hge (by norm_num : (0:ℝ) ≤ 1/2)
      rw [h2]
      linarith
    have h2 : Real.sqrt ((2:ℝ)^(-126:ℤ)) = (2:ℝ)^(-63:ℤ) := by
      rw [show (2:ℝ)^(-126:ℤ) = ((2:ℝ)^(-63:ℤ))^2 by norm_num]
      exact Real.sqrt_sq (by norm_num)
    linarith [h1, h2]
  have h149 : (2:ℝ)^(-149:ℤ) ≤ u32*t := by
    calc (2:ℝ)^(-149:ℤ) ≤ u32*(2:ℝ)^(-63:ℤ) := by unfold u32; norm_num
      _ ≤ u32*t := mul_le_mul_of_nonneg_left ht63 hu0
  -- bsq vs t
  have hbsqle : bsq ≤ t*(1+5*u32) := by
    have h := (abs_le.mp hbsq).2
    nlinarith [h, h149]
  have hbsqge : t*(1-5*u32) ≤ bsq := by
    have h := (abs_le.mp hbsq).1
    nlinarith [h, h149]
  have hbsqpos : 0 < bsq := by nlinarith [hbsqge, htpos, h5pos]
  have h1bpos : 0 < 1/bsq := one_div_pos.2 hbsqpos
  have h1bnn : 0 ≤ 1/bsq := le_of_lt h1bpos
  have ht1 : 0 ≤ 1/t := by positivity
  -- 1/bsq ≤ (1+6u)/t
  have h1ble : 1/bsq ≤ (1+6*u32)*(1/t) := by
    have hpos : 0 < t*(1-5*u32) := mul_pos htpos h5pos
    have h1 : 1/bsq ≤ 1/(t*(1-5*u32)) := one_div_le_one_div_of_le hpos hbsqge
    have h2 : 1/(t*(1-5*u32)) = (1/(1-5*u32))*(1/t) := by field_simp
    have h3 : 1/(1-5*u32) ≤ 1+6*u32 := by
      rw [div_le_iff₀ h5pos]
      unfold u32
      norm_num
    calc 1/bsq ≤ (1/(1-5*u32))*(1/t) := by rw [← h2]; exact h1
      _ ≤ (1+6*u32)*(1/t) := mul_le_mul_of_nonneg_right h3 ht1
  -- 1/bsq ≥ (1/2)/t
  have h1bge : (1/2)*(1/t) ≤ 1/bsq := by
    have h5u : 0 < 1+5*u32 := by linarith
    have hpos : 0 < t*(1+5*u32) := mul_pos htpos h5u
    have h1 : 1/(t*(1+5*u32)) ≤ 1/bsq := one_div_le_one_div_of_le hbsqpos hbsqle
    have h2 : 1/(t*(1+5*u32)) = (1/(1+5*u32))*(1/t) := by field_simp
    have h3 : (1/2) ≤ 1/(1+5*u32) := by
      rw [le_div_iff₀ h5u]; nlinarith [hu0]
    calc (1/2)*(1/t) ≤ (1/(1+5*u32))*(1/t) := mul_le_mul_of_nonneg_right h3 ht1
      _ = 1/(t*(1+5*u32)) := by rw [h2]
      _ ≤ 1/bsq := h1
  -- t ≤ 2^64
  have htle : t ≤ (2:ℝ)^(64:ℤ) := by
    rw [htdef]
    have h2 : Real.sqrt ((2:ℝ)^(128:ℤ)) = (2:ℝ)^(64:ℤ) := by
      rw [show (2:ℝ)^(128:ℤ) = ((2:ℝ)^(64:ℤ))^2 by norm_num]
      exact Real.sqrt_sq (by norm_num)
    calc Real.sqrt be ≤ Real.sqrt ((2:ℝ)^(128:ℤ)) := Real.sqrt_le_sqrt hbe_hi
      _ = (2:ℝ)^(64:ℤ) := h2
  have h1tge : (2:ℝ)^(-64:ℤ) ≤ 1/t := by
    have h := one_div_le_one_div_of_le htpos htle
    have e : 1/((2:ℝ)^(64:ℤ)) = (2:ℝ)^(-64:ℤ) := by norm_num
    rwa [e] at h
  -- 1/√eps bounds
  have hsqle : Real.sqrt eps ≤ (2:ℝ)^(63:ℤ) := by
    have h2 : Real.sqrt ((2:ℝ)^(126:ℤ)) = (2:ℝ)^(63:ℤ) := by
      rw [show (2:ℝ)^(126:ℤ) = ((2:ℝ)^(63:ℤ))^2 by norm_num]
      exact Real.sqrt_sq (by norm_num)
    calc Real.sqrt eps ≤ Real.sqrt ((2:ℝ)^(126:ℤ)) := Real.sqrt_le_sqrt hle_eps
      _ = (2:ℝ)^(63:ℤ) := h2
  have h1sge : (2:ℝ)^(-63:ℤ) ≤ 1/Real.sqrt eps := by
    have h := one_div_le_one_div_of_le hspe hsqle
    have e : 1/((2:ℝ)^(63:ℤ)) = (2:ℝ)^(-63:ℤ) := by norm_num
    rwa [e] at h
  -- 1/t ≤ (3/2)/√eps
  have h1t_le : 1/t ≤ (3/2)*(1/Real.sqrt eps) := by
    have h1 : 1/t ≤ 1/Real.sqrt (eps/2) := by
      apply one_div_le_one_div_of_le (Real.sqrt_pos.2 (by linarith))
      rw [htdef]
      exact Real.sqrt_le_sqrt hbe_lo
    have h2 : 1/Real.sqrt (eps/2) = Real.sqrt 2 * (1/Real.sqrt eps) := by
      rw [Real.sqrt_div heps0.le]
      field_simp
    have h3 : Real.sqrt 2 ≤ 3/2 := by
      nlinarith [Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2), Real.sqrt_nonneg 2]
    have h4 : 0 ≤ 1/Real.sqrt eps := by positivity
    calc 1/t ≤ Real.sqrt 2 * (1/Real.sqrt eps) := by rw [← h2]; exact h1
      _ ≤ (3/2)*(1/Real.sqrt eps) := mul_le_mul_of_nonneg_right h3 h4
  -- brstd bounds
  have hbrle : brstd ≤ (1+4*u32)*((1+6*u32)*(1/t)) + (2:ℝ)^(-148:ℤ) := by
    have h := (abs_le.mp hbrstd).2
    have habs : |1/bsq| = 1/bsq := abs_of_pos h1bpos
    rw [habs] at h
    have h1 : (1+4*u32)*(1/bsq) ≤ (1+4*u32)*((1+6*u32)*(1/t)) :=
      mul_le_mul_of_nonneg_left h1ble (by linarith)
    nlinarith [h, h1]
  have hbrge : 0 ≤ brstd := by
    have h := (abs_le.mp hbrstd).1
    have habs : |1/bsq| = 1/bsq := abs_of_pos h1bpos
    rw [habs] at h
    have h4u : 4*u32 ≤ 1/2 := by unfold u32; norm_num
    have hstep1 : (1/2)*(1/bsq) ≤ (1-4*u32)*(1/bsq) := by nlinarith [h4u, h1bnn]
    have hstep2 : (1/bsq) - 4*u32*(1/bsq) = (1-4*u32)*(1/bsq) := by ring
    have h1b65 : (2:ℝ)^(-65:ℤ) ≤ 1/bsq := by
      have hh : (1/2)*(2:ℝ)^(-64:ℤ) ≤ (1/2)*(1/t) :=
        mul_le_mul_of_nonneg_left h1tge (by norm_num)
      have e : (1/2)*(2:ℝ)^(-64:ℤ) = (2:ℝ)^(-65:ℤ) := by norm_num
      linarith [hh, h1bge, e ▸ hh]
    nlinarith [h, h1b65, hstep1, hstep2]
  have hb1 : brstd ≤ 2/Real.sqrt eps := by
    have hA : (1+4*u32)*(1+6*u32) ≤ 101/100 := by unfold u32; norm_num
    have hC : (1+4*u32)*((1+6*u32)*(1/t)) = ((1+4*u32)*(1+6*u32))*(1/t) := by ring
    have hD : ((1+4*u32)*(1+6*u32))*(1/t) ≤ (101/100)*(1/t) :=
      mul_le_mul_of_nonneg_right hA ht1
    have hE : (101/100)*(1/t) ≤ (101/100)*((3/2)*(1/Real.sqrt eps)) :=
      mul_le_mul_of_nonneg_left h1t_le (by norm_num)
    have hF : (101/100)*((3/2)*(1/Real.sqrt eps)) ≤ 2*(1/Real.sqrt eps) - (1/100)*(1/Real.sqrt eps) := by
      nlinarith [h1sge, hspe]
    have hG : (2:ℝ)^(-148:ℤ) ≤ (1/100)*(1/Real.sqrt eps) := by
      have h1 : (2:ℝ)^(-148:ℤ) ≤ (1/100)*(2:ℝ)^(-63:ℤ) := by norm_num
      have h2 : (1/100)*(2:ℝ)^(-63:ℤ) ≤ (1/100)*(1/Real.sqrt eps) :=
        mul_le_mul_of_nonneg_left h1sge (by norm_num)
      linarith
    rw [hC] at hbrle
    have e : 2/Real.sqrt eps = 2*(1/Real.sqrt eps) := by ring
    rw [e]
    linarith
  have hK : |brstd - R| ≤ 2/Real.sqrt eps := by
    rw [abs_le]
    constructor <;> linarith [hbrge, hb1, hRle, hR0]
  exact hK

/-- Real-arithmetic step: from `S - B ≤ be`, `B ≤ S/1000 + eps/500` and `7/8 eps ≤ S` we get `eps/2 ≤ be`. -/
theorem rstd_tol_belo_num {eps S B be : ℝ} (heps0 : 0 < eps)
    (hbe : S - B ≤ be) (hB : B ≤ S/1000 + eps/500) (hS : eps*(7/8) ≤ S) :
    eps/2 ≤ be := by
  nlinarith [hbe, hB, hS, heps0]

/-- Clean-context numeric core of `2^-126 ≤ be`. -/
theorem rstd_tol_ge126_num {eps be : ℝ} (hbe : eps/2 ≤ be)
    (hge : (2:ℝ)^(-125:ℤ) ≤ eps) : (2:ℝ)^(-126:ℤ) ≤ be := by
  have h2 : (2:ℝ)^(-126:ℤ) = (2:ℝ)^(-125:ℤ)/2 := by norm_num
  rw [h2]
  exact le_trans (div_le_div_of_nonneg_right hge (by norm_num)) hbe

/-- Clean-context numeric core of `0 < bsq`. -/
theorem rstd_tol_bsq_pos_num {u s bs e149 : ℝ} (hu4 : 4*u ≤ 1/2)
    (hs : 0 < s) (he : e149 ≤ (1/4)*s)
    (h1 : -(4*u*s + e149) ≤ bs - s) : 0 < bs := by
  have hmul : 4*u*s ≤ (1/2)*s := mul_le_mul_of_nonneg_right hu4 (le_of_lt hs)
  linarith [h1, hmul, he, hs]

/-- Clean-context numeric core of `2^-65 ≤ bsq`. -/
theorem rstd_tol_bsq_ge_num {u s bs e149 : ℝ} (hu4 : 4*u ≤ 1/2)
    (hs : (2:ℝ)^(-63:ℤ) ≤ s) (he : e149 ≤ (1/4)*s)
    (h1 : -(4*u*s + e149) ≤ bs - s) : (2:ℝ)^(-65:ℤ) ≤ bs := by
  have hs0 : 0 < s := lt_of_lt_of_le (by norm_num) hs
  have hmul : 4*u*s ≤ (1/2)*s := mul_le_mul_of_nonneg_right hu4 (le_of_lt hs0)
  have hb : (1/4)*s ≤ bs := by linarith [h1, hmul, he]
  have h2 : (2:ℝ)^(-65:ℤ) = (1/4)*(2:ℝ)^(-63:ℤ) := by norm_num
  rw [h2]
  exact le_trans (mul_le_mul_of_nonneg_left hs (by norm_num)) hb

/-- Clean-context numeric core: `2^-135 ≤ 2·g·W` when `g = 2^-136/e` and `e ≤ W`. -/
theorem rstd_tol_t135_le {e g W : ℝ} (he : 0 < e)
    (hg : g = (2:ℝ)^(-136:ℤ)/e) (hW : e ≤ W) :
    (2:ℝ)^(-135:ℤ) ≤ 2*g*W := by
  have hWe : 1 ≤ W/e := by
    rw [le_div_iff₀ he]
    linarith
  have e1 : (2:ℝ)^(-135:ℤ) = 2*((2:ℝ)^(-136:ℤ)) := by norm_num
  rw [e1, hg]
  have h2 : 2*((2:ℝ)^(-136:ℤ)/e)*W = 2*((2:ℝ)^(-136:ℤ))*(W/e) := by
    rw [div_eq_mul_inv, div_eq_mul_inv]
    ring
  rw [h2]
  have h3 := mul_le_mul_of_nonneg_left hWe
    (by norm_num : (0:ℝ) ≤ 2*((2:ℝ)^(-136:ℤ)))
  simpa using h3

/-- Clean-context numeric core of the relative error of `be`. -/
theorem rstd_tol_comb {A u g W t : ℝ} (ht : t ≤ 2*g*W) :
    A*u*(1+3*u/2)*W + t + 3*u/2*W ≤ (A*u*(1+3*u/2) + 3*u/2 + 2*g)*W := by
  have e : (A*u*(1+3*u/2) + 3*u/2 + 2*g)*W
      = A*u*(1+3*u/2)*W + 3*u/2*W + 2*g*W := by ring
  rw [e]
  linarith

/-- Clean-context numeric core of `|epsV - epsR| ≤ (3/2)u·W`. -/
theorem rstd_tol_epsdev {u e epsV W η : ℝ} (h : |epsV - e| ≤ u*e + η)
    (hη : η ≤ (1/2)*(u*e)) (hu0 : 0 ≤ u) (_he0 : 0 ≤ e) (hW : e ≤ W) :
    |epsV - e| ≤ 3*u/2*W := by
  have h1 : u*e + η ≤ 3*u/2*e := by
    have h2 : u*e + (1/2)*(u*e) = 3*u/2*e := by ring
    linarith
  have h2 : 3*u/2*e ≤ 3*u/2*W := by
    apply mul_le_mul_of_nonneg_left hW
    linarith
  linarith

/-- Clean-context numeric core: the bracketed deviation bound is nonnegative. -/
theorem rstd_tol_E_nonneg {u mT aM rms η sV : ℝ} (hu0 : 0 ≤ u) (hmT : 0 ≤ mT)
    (haM : 0 ≤ aM) (hrms : 0 ≤ rms) (hη : 0 ≤ η) (hsV : 0 ≤ sV) :
    0 ≤ 2*u*sV + (1+2*u)*(mT + u*(rms + aM + mT) + 4*η) := by
  have h3 : 0 ≤ u*(rms + aM + mT) := mul_nonneg hu0 (by linarith)
  have h4 : 0 ≤ 4*η := by linarith
  have h2 : 0 ≤ mT + u*(rms + aM + mT) + 4*η := by linarith
  have h5 : 0 ≤ 1 + 2*u := by linarith
  have h6 : 0 ≤ (1+2*u)*(mT + u*(rms + aM + mT) + 4*η) := mul_nonneg h5 h2
  have h7 : 0 ≤ u*sV := mul_nonneg hu0 hsV
  have h8 : 0 ≤ 2*(u*sV) := by linarith
  have e : 2*u*sV = 2*(u*sV) := by ring
  rw [e]
  linarith

/-- Two-sided bound on a ratio of square roots. -/
theorem rstd_tol_sqrt_ratio {sW s0 y : ℝ} (hsW : 0 < sW) (hs0 : 0 < s0)
    (hy0 : 0 ≤ y) (hy : 2*y + (2*y)^2 ≤ 1)
    (hlo : sW * Real.sqrt (1 - 2*y) ≤ s0)
    (hhi : s0 ≤ sW * (1 + y)) :
    |sW / s0 - 1| ≤ 2*y := by
  have h2pos : 0 < 1 - 2*y := by nlinarith [hy, hy0]
  have hr : 0 < Real.sqrt (1 - 2*y) := Real.sqrt_pos.2 h2pos
  have hle : sW / s0 ≤ 1 + 2*y := by
    rw [div_le_iff₀ hs0]
    have h1 : sW ≤ s0 / Real.sqrt (1 - 2*y) := by
      rw [le_div_iff₀ hr]
      exact hlo
    have h3' : 1 / Real.sqrt (1 - 2*y) ≤ 1 + 2*y :=
      rstd_tol_one_div_sqrt_one_sub (by linarith : (0:ℝ) ≤ 2 * y) hy
    have h2 : s0 / Real.sqrt (1 - 2*y) ≤ s0 * (1 + 2*y) := by
      rw [div_eq_mul_one_div]
      exact mul_le_mul_of_nonneg_left h3' (le_of_lt hs0)
    have e : s0 * (1 + 2*y) = (1 + 2*y) * s0 := by ring
    linarith [h1, h2, e]
  have hge : 1 - y ≤ sW / s0 := by
    rw [le_div_iff₀ hs0]
    have hy1 : 0 ≤ 1 - y := by nlinarith [hy, hy0]
    have h2 : (1-y)*s0 ≤ (1-y)*(sW*(1+y)) :=
      mul_le_mul_of_nonneg_left hhi hy1
    have h3 : (1-y)*(sW*(1+y)) = sW*(1 - y^2) := by ring
    have h4 : sW*(1 - y^2) ≤ sW := by
      have h5 : 1 - y^2 ≤ 1 := by nlinarith [sq_nonneg y]
      simpa using mul_le_mul_of_nonneg_left h5 (le_of_lt hsW)
    linarith [h2, h3, h4]
  rw [abs_le]
  constructor <;> linarith

/-- One-sided form of `rstd_tol_sqrt_ratio`. -/
theorem rstd_tol_sqrt_ratio_le {sW s0 y : ℝ} (hs0 : 0 < s0)
    (hy0 : 0 ≤ y) (hy : 2*y + (2*y)^2 ≤ 1)
    (hlo : sW * Real.sqrt (1 - 2*y) ≤ s0) :
    sW / s0 ≤ 1 + 2*y := by
  have h2pos : 0 < 1 - 2*y := by nlinarith [hy, hy0]
  have hr : 0 < Real.sqrt (1 - 2*y) := Real.sqrt_pos.2 h2pos
  rw [div_le_iff₀ hs0]
  have h1 : sW ≤ s0 / Real.sqrt (1 - 2*y) := by
    rw [le_div_iff₀ hr]
    exact hlo
  have h3' : 1 / Real.sqrt (1 - 2*y) ≤ 1 + 2*y :=
    rstd_tol_one_div_sqrt_one_sub (by linarith : (0:ℝ) ≤ 2 * y) hy
  have h2 : s0 / Real.sqrt (1 - 2*y) ≤ s0 * (1 + 2*y) := by
    rw [div_eq_mul_one_div]
    exact mul_le_mul_of_nonneg_left h3' (le_of_lt hs0)
  have e : s0 * (1 + 2*y) = (1 + 2*y) * s0 := by ring
  linarith [h1, h2, e]

/-- **The `rstd` tolerance**: the stored `rstd = 1/√(var + eps)` is within `rstdTol` of `rstdRef`, from the mean, the variance sum and the device facts for `/` and `sqrt`. -/
theorem rstd_tol {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    {P : Platform} (hdivf : Device.DivFull P) (hsqrt : Device.SqrtF32 P)
    {bx : Nat → F32} {bmean dv epsV one : F32}
    {x sq : Nat → F32} {bvar bva be bsq brstd : F32}
    (hdv : dv.toReal = (d.D : ℝ)) (hdvfin : dv.isFinite = true)
    (hone : one.toReal = 1) (honefin : one.isFinite = true)
    (hmeanfin : bmean.isFinite = true)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hfinbx : ∀ j, j < d.D → (bx j).isFinite = true)
    (hval : ∀ j, j < d.D → (bx j).toReal = sR d init t j)
    (hbdd : ∀ j, j < d.D → |sR d init t j| ≤ 2 ^ 57)
    (hfinx : ∀ j, j < d.BD → (x j).isFinite = true)
    (hdev : ∀ j, j < d.D → AddRel true (bx j) (FloatV.neg bmean) (x j))
    (hzero : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0)
    (hfsq : ∀ j, j < d.BD → (sq j).isFinite = true)
    (hsq : ∀ j, j < d.BD → FloatV.NaNEq (FloatV.mul (x j) (x j)) (sq j))
    (hcover : ∃ tr : Tree, Tree.Covers d.BD tr ∧ tr.depth ≤ Nat.clog 2 d.BD ∧
      Tree.Eval (AddRel true) sq tr bvar)
    (hdivvar : P.div .binary32 bvar dv bva)
    (hadd : AddRel true bva epsV be)
    (hepsfin : epsV.isFinite = true)
    (hepsge : (2 : ℝ) ^ (-125 : ℤ) ≤ epsV.toReal)
    (hepsle : epsV.toReal ≤ (2 : ℝ) ^ (126 : ℤ))
    (heps : |epsV.toReal - epsR d| ≤ u32 * epsR d + eta32)
    (hmath : P.math .sqrt .binary32 be bsq)
    (hdivr : P.div .binary32 one bsq brstd) :
    brstd.isFinite = true ∧ |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t := by
  -- ## numeric facts about the format constants
  have hu32 : u32 = 1/16777216 := by unfold u32; norm_num
  have hu_pos : (0:ℝ) < u32 := u32_pos
  have hu0 : (0:ℝ) ≤ u32 := le_of_lt u32_pos
  have hu16 : u32 ≤ 1/16 := u32_le_small
  have hη0 : (0:ℝ) ≤ eta32 := le_of_lt eta32_pos
  have hηhalf : eta32 ≤ (1/2) * u32 * (2:ℝ)^(-125:ℤ) := by unfold u32 eta32; norm_num
  have h23 : (2:ℝ)^(-23:ℤ) = 2*u32 := by unfold u32; norm_num
  have h4u : (2:ℝ)^(-22:ℤ) = 4*u32 := by unfold u32; norm_num
  -- ## facts about the instance
  have hD0 : 0 < d.D := hd.d_pos
  have hDR : (0:ℝ) < (d.D:ℝ) := rstd_tol_D_pos hd
  have hL0 : (0:ℝ) ≤ (depth d:ℝ) := by positivity
  have hL10 : (depth d:ℝ) ≤ 10 := by exact_mod_cast rstd_tol_depth_le hd
  have hmT0 : 0 ≤ meanTol d init t := rstd_tol_meanTol_nonneg
  have haM0 : 0 ≤ absMean d init t := rstd_tol_absMean_nonneg
  have hrms0 : 0 ≤ rmsS d init t := rstd_tol_rmsS_nonneg
  have hV0 : 0 ≤ varRef d init t := rstd_tol_varRef_nonneg
  have heps0 : 0 < epsR d := epsR_pos hd
  have hepsgeR : (2:ℝ)^(-125:ℤ) ≤ epsR d := epsR_ge hd
  have hepsleR : epsR d ≤ (2:ℝ)^(126:ℤ) := epsR_le hd
  have hR0 : 0 < rstdRef d init t := rstd_tol_R_pos hd
  have hRW : rstdRef d init t ^ 2 * (varRef d init t + epsR d) = 1 := rstd_tol_R_sq_mul hd
  have hWpos : 0 < varRef d init t + epsR d := rstd_tol_Ws_pos hd
  have hepsV0 : 0 ≤ epsV.toReal := le_trans (by norm_num : (0:ℝ) ≤ (2:ℝ)^(-125:ℤ)) hepsge
  have hepsVpos : 0 < epsV.toReal := lt_of_lt_of_le (by norm_num : (0:ℝ) < (2:ℝ)^(-125:ℤ)) hepsge
  -- ## the per-lane bound on `x`
  have hxbd := rstd_tol_x_bd hd hmeanfin hmean hfinbx hval hbdd hfinx hdev hzero
  have hDb : d.D ≤ d.BD := hd.d_le_bd
  have hxbdD : ∀ j, j < d.D → |(x j).toReal| ≤ 2^58 + 2^40 :=
    fun j hj => (hxbd j (lt_of_lt_of_le hj hDb)).2
  -- ## sums of squares
  have hsumBb := rstd_tol_sum_x_sq_bd hd hxbdD
  set X : ℝ := ∑ j ∈ Finset.range d.D, (x j).toReal^2 with hXdef
  have hX0 : 0 ≤ X := hsumBb.1
  have hXle : X ≤ (d.D:ℝ)*2^117 := hsumBb.2
  have hXle2 : X ≤ (2:ℝ)^126 := by
    have h512 : (d.D:ℝ) ≤ 512 := by exact_mod_cast hd.d_le
    nlinarith [hXle]
  set V0 : ℝ := X / (d.D:ℝ) with hV0def
  have hV0nn : 0 ≤ V0 := by rw [hV0def]; positivity
  have hV0le : V0 ≤ (2:ℝ)^117 := by
    rw [hV0def, div_le_iff₀ hDR]
    nlinarith [hXle]
  -- ## the tree sum
  obtain ⟨tr, hcov, hdepth0, heval⟩ := hcover
  have hdepth : tr.depth ≤ depth d := by simpa [depth] using hdepth0
  have hbvar := rstd_tol_bvar (_init := init) (_t := t) hd hxbd hzero hfsq hsq hcov hdepth heval
  -- ## the division
  have hbva := rstd_tol_bva (init := init) (t := t) hd hdivf hxbd hzero hfsq hsq hcov hdepth heval hdv hdvfin hdivvar
  have hbva' : |bva.toReal - V0| ≤ (3.1 * (depth d : ℝ) + 6) * u32 * V0 + (2 : ℝ) ^ (-136 : ℤ) := by
    simpa [hXdef, hV0def] using hbva.2
  have hcoef3 : (3.1*(depth d:ℝ)+6)*u32 ≤ 1/1000 := by
    have h1 : 3.1*(depth d:ℝ)+6 ≤ 37 := by linarith
    calc (3.1*(depth d:ℝ)+6)*u32 ≤ 37*u32 := mul_le_mul_of_nonneg_right h1 hu0
      _ = 37/16777216 := by rw [hu32]; norm_num
      _ ≤ 1/1000 := by norm_num
  have hcoef4 : (4*(depth d:ℝ)+16)*u32 ≤ 1/1000 := by
    have h1 : 4*(depth d:ℝ)+16 ≤ 56 := by linarith
    calc (4*(depth d:ℝ)+16)*u32 ≤ 56*u32 := mul_le_mul_of_nonneg_right h1 hu0
      _ = 56/16777216 := by rw [hu32]; norm_num
      _ ≤ 1/1000 := by norm_num
  have hbva_abs : |bva.toReal| ≤ (2:ℝ)^118 := by
    have h1 := hbva'
    have h2 : (3.1*(depth d:ℝ)+6)*u32*V0 ≤ (1/1000)*V0 :=
      mul_le_mul_of_nonneg_right hcoef3 hV0nn
    have h3 : |bva.toReal| ≤ |bva.toReal - V0| + V0 := by
      have h4 := abs_add_le (bva.toReal - V0) V0
      have h5 : |V0| = V0 := abs_of_nonneg hV0nn
      have e : bva.toReal - V0 + V0 = bva.toReal := by ring
      rw [e, h5] at h4
      exact h4
    have h4 : (2:ℝ)^(-136:ℤ) ≤ (2:ℝ)^117 := by norm_num
    nlinarith [h1, h2, h3, hV0le, h4]
  -- ## `epsV` against `epsR`
  have heta_eps : eta32 ≤ u32 * epsR d := by
    have h1 : u32 * (2:ℝ)^(-125:ℤ) ≤ u32*epsR d := mul_le_mul_of_nonneg_left hepsgeR hu0
    have h2 : eta32 ≤ u32*(2:ℝ)^(-125:ℤ) := by unfold u32 eta32; norm_num
    linarith
  have hepsV_lo : epsR d * (1 - 2*u32) ≤ epsV.toReal := by
    have h1 := (abs_le.mp heps).1
    nlinarith [h1, heta_eps]
  have hepsV_le : epsV.toReal ≤ epsR d * (1 + 2*u32) := by
    have h1 := (abs_le.mp heps).2
    nlinarith [h1, heta_eps]
  -- ## the added `eps`
  set W0 : ℝ := V0 + epsR d with hW0def
  set g2 : ℝ := (2:ℝ)^(-136:ℤ)/epsR d with hg2def
  have hW0pos : 0 < W0 := by rw [hW0def]; linarith
  have hW0ge_eps : epsR d ≤ W0 := by rw [hW0def]; linarith
  have hg2nn : 0 ≤ g2 := by rw [hg2def]; positivity
  have hg2le : g2 ≤ 1/2048 := by
    rw [hg2def, div_le_iff₀ heps0]
    have h2 : (2:ℝ)^(-136:ℤ) = (1/2048) * (2:ℝ)^(-125:ℤ) := by norm_num
    rw [h2]
    have h := mul_le_mul_of_nonneg_left hepsgeR (by norm_num : (0:ℝ) ≤ 1/2048)
    linarith
  have hetaW : eta32 ≤ (1/2)*u32*W0 := by
    have h1a : (1/2)*u32*(2:ℝ)^(-125:ℤ) ≤ (1/2)*u32*epsR d :=
      mul_le_mul_of_nonneg_left hepsgeR (by positivity)
    have h1b : (1/2)*u32*epsR d ≤ (1/2)*u32*W0 :=
      mul_le_mul_of_nonneg_left hW0ge_eps (by positivity)
    linarith [hηhalf, h1a, h1b]
  have hVeps : V0 + epsV.toReal ≤ W0*(1+2*u32) := by
    have h2 : 2*u32*epsR d ≤ 2*u32*W0 := mul_le_mul_of_nonneg_left hW0ge_eps (by positivity)
    nlinarith [hepsV_le, h2, hW0def]
  have hu1000 : 2*u32 ≤ 1/1000 := by rw [hu32]; norm_num
  have hVeps2 : V0 + epsV.toReal ≤ W0*(1+1/1000) := by
    have h2 : W0*(1+2*u32) ≤ W0*(1+1/1000) := by
      apply mul_le_mul_of_nonneg_left _ (le_of_lt hW0pos)
      linarith [hu1000]
    linarith [hVeps, h2]
  have hW0le : W0 ≤ (2:ℝ)^127 := by
    rw [hW0def]
    nlinarith [hV0le, hepsleR]
  set t135 : ℝ := (2:ℝ)^(-135:ℤ) with ht135
  have h135 : t135 ≤ epsR d / 500 := by
    have h1 : t135 ≤ (2:ℝ)^(-125:ℤ)/500 := by rw [ht135]; norm_num
    have h2 : (2:ℝ)^(-125:ℤ)/500 ≤ epsR d/500 :=
      div_le_div_of_nonneg_right hepsgeR (by norm_num)
    exact le_trans h1 h2
  -- ## `be`
  have hbe := rstd_tol_be hd (V := V0) hV0nn hepsV0 hepsVpos hepsle hbva'
    hbva_abs hbva.1 hepsfin hadd
  have hbe_hi : be.toReal ≤ (2:ℝ)^128 := by
    set P : ℝ := (2:ℝ)^127 with hPdef
    have hW0le' : W0 ≤ P := by rw [hPdef]; exact hW0le
    have h2 := (abs_le.mp hbe.2).2
    have hS0 : 0 ≤ V0 + epsV.toReal := by linarith
    have hm1 : (4*(depth d:ℝ)+16)*u32*(V0+epsV.toReal) ≤ (1/1000)*(V0+epsV.toReal) :=
      mul_le_mul_of_nonneg_right hcoef4 hS0
    have h135' : (2:ℝ)^(-135:ℤ) ≤ 1 := by norm_num
    have e1 : (V0+epsV.toReal) + (1/1000)*(V0+epsV.toReal) = (V0+epsV.toReal)*(1+1/1000) := by ring
    have hm2 : (V0+epsV.toReal)*(1+1/1000) ≤ (W0*(1+1/1000))*(1+1/1000) :=
      mul_le_mul_of_nonneg_right hVeps2 (by norm_num)
    have hm3 : (W0*(1+1/1000))*(1+1/1000) ≤ (P*(1+1/1000))*(1+1/1000) :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hW0le' (by norm_num)) (by norm_num)
    have hP2 : (2:ℝ) ≤ P := by
      have h := pow_le_pow_right₀ (by norm_num : (1:ℝ) ≤ 2) (by norm_num : (1:ℕ) ≤ 127)
      simpa [hPdef] using h
    have hlast : (P*(1+1/1000))*(1+1/1000) + 1 ≤ (2:ℝ)^128 := by
      have e : (2:ℝ)^128 = P * 2 := by
        rw [hPdef]
        rw [show (128:ℕ) = 127 + 1 by norm_num, pow_succ]
      rw [e]
      nlinarith [hP2]
    linarith [h2, hm1, h135', e1, hm2, hm3, hlast]
  have hbe_lo : epsR d / 2 ≤ be.toReal := by
    have h2 := (abs_le.mp hbe.2).1
    rw [← ht135] at h2
    have hVE0 : 0 ≤ V0 + epsV.toReal := by linarith
    have hb1 : (4*(depth d:ℝ)+16)*u32*(V0+epsV.toReal) ≤ (1/1000)*(V0+epsV.toReal) :=
      mul_le_mul_of_nonneg_right hcoef4 hVE0
    have h2u8 : 2*u32 ≤ 1/8 := by rw [hu32]; norm_num
    have h7 : epsR d * (7/8) ≤ epsR d * (1 - 2*u32) := by
      apply mul_le_mul_of_nonneg_left _ (le_of_lt heps0)
      linarith
    have hepsV78 : epsR d * (7/8) ≤ epsV.toReal := le_trans h7 hepsV_lo
    have hS : epsR d * (7/8) ≤ V0 + epsV.toReal := by linarith
    have hB : (4*(depth d:ℝ)+16)*u32*(V0+epsV.toReal) + t135
        ≤ (V0+epsV.toReal)/1000 + epsR d/500 := by linarith
    have hbeS : (V0+epsV.toReal) - ((4*(depth d:ℝ)+16)*u32*(V0+epsV.toReal) + t135)
        ≤ be.toReal := by linarith
    exact rstd_tol_belo_num heps0 hbeS hB hS
  -- ## `sqrt` at `be`
  have hbe_ge126 : (2:ℝ)^(-126:ℤ) ≤ be.toReal := rstd_tol_ge126_num hbe_lo hepsgeR
  have hbe_nn : 0 ≤ be.toReal := le_trans (by norm_num : (0:ℝ) ≤ (2:ℝ)^(-126:ℤ)) hbe_ge126
  have hdom : be.toReal = 0 ∨ (2:ℝ)^(-126:ℤ) ≤ be.toReal := Or.inr hbe_ge126
  have hroom : (1 + (2:ℝ)^(-22:ℤ))*|Real.sqrt be.toReal| + (2:ℝ)^(-149:ℤ)
      ≤ ((Format.binary32.maxVal : ℚ):ℝ) := rstd_tol_room be.toReal hbe_nn hbe_hi
  have hbsq_acc := rstd_tol_sqrt_acc hsqrt hbe.1 hdom hroom hmath
  have hs63 : (2:ℝ)^(-63:ℤ) ≤ Real.sqrt be.toReal := by
    have h := Real.sqrt_le_sqrt hbe_ge126
    have e : Real.sqrt ((2:ℝ)^(-126:ℤ)) = (2:ℝ)^(-63:ℤ) := by
      rw [show (2:ℝ)^(-126:ℤ) = ((2:ℝ)^(-63:ℤ))^2 by norm_num]
      exact Real.sqrt_sq (by norm_num)
    rwa [e] at h
  have h149le : (2:ℝ)^(-149:ℤ) ≤ (1/4)*Real.sqrt be.toReal := by
    have h1 : (2:ℝ)^(-149:ℤ) ≤ (1/4)*(2:ℝ)^(-63:ℤ) := by norm_num
    have h2 : (1/4)*(2:ℝ)^(-63:ℤ) ≤ (1/4)*Real.sqrt be.toReal :=
      mul_le_mul_of_nonneg_left hs63 (by norm_num)
    exact le_trans h1 h2
  have hs63pos : 0 < Real.sqrt be.toReal := lt_of_lt_of_le (by norm_num) hs63
  have hbsq_pos : 0 < bsq.toReal := by
    have h1 := (abs_le.mp hbsq_acc.2).1
    have h2 : 4*u32 ≤ 1/2 := by rw [hu32]; norm_num
    exact rstd_tol_bsq_pos_num h2 hs63pos h149le h1
  -- ## `1/·` at `bsq`
  have honeRat : one.toRat = 1 := by
    have h : ((one.toRat : ℚ) : ℝ) = 1 := by simpa [FloatV.toReal] using hone
    exact_mod_cast h
  have hbsqRat_pos : (0:ℚ) < bsq.toRat := by
    have h : (0:ℝ) < ((bsq.toRat : ℚ) : ℝ) := by simpa [FloatV.toReal] using hbsq_pos
    exact_mod_cast h
  have hbsqRat_ne : bsq.toRat ≠ 0 := ne_of_gt hbsqRat_pos
  have hbsqRge : (2:ℝ)^(-65:ℤ) ≤ bsq.toReal := by
    have h1 := (abs_le.mp hbsq_acc.2).1
    have h2 : 4*u32 ≤ 1/2 := by rw [hu32]; norm_num
    exact rstd_tol_bsq_ge_num h2 hs63 h149le h1
  have hbsqRat_ge : (2:ℚ)^(-65:ℤ) ≤ bsq.toRat := by
    have h : (((2:ℚ)^(-65:ℤ) : ℚ) : ℝ) ≤ (bsq.toRat : ℝ) := by
      push_cast
      simpa [FloatV.toReal] using hbsqRge
    exact_mod_cast h
  have hq : |one.toRat / bsq.toRat| ≤ (2:ℚ)^(126:ℤ) := by
    have hpos : (0:ℚ) < (2:ℚ)^(-65:ℤ) := by norm_num
    have h1 : 1/bsq.toRat ≤ 1/(2:ℚ)^(-65:ℤ) := one_div_le_one_div_of_le hpos hbsqRat_ge
    have h2 : (1:ℚ)/(2:ℚ)^(-65:ℤ) = (2:ℚ)^(65:ℤ) := by norm_num
    have h3 : (2:ℚ)^(65:ℤ) ≤ (2:ℚ)^(126:ℤ) := by norm_num
    calc |one.toRat / bsq.toRat| = 1/bsq.toRat := by
          rw [honeRat, abs_div, abs_one, abs_of_pos hbsqRat_pos, one_div]
      _ ≤ (2:ℚ)^(65:ℤ) := by rw [← h2]; exact h1
      _ ≤ (2:ℚ)^(126:ℤ) := h3
  have hbrstd_acc := rstd_tol_div_acc hdivf honefin honeRat hbsq_acc.1 hbsqRat_ne hq hdivr
  refine ⟨hbrstd_acc.1, ?_⟩
  by_cases hgood : 4 * cond d init t ≤ 1
  · -- ### the well-conditioned branch
    have hdecomp : ∀ j, j < d.D → ∃ c ε : ℝ, |c| ≤ 2 * u32 ∧
        |ε| ≤ meanTol d init t + u32 * (|sR d init t j| + absMean d init t + meanTol d init t)
          + 4 * eta32 ∧
        (x j).toReal = (sR d init t j - meanRef d init t + ε) * (1 + c) := by
      intro j hj
      exact (rstd_tol_dev hj (hfinbx j hj) hmeanfin (hval j hj) hmean (hbdd j hj)
        (rstd_tol_absMean_le hbdd hD0) (rstd_tol_meanTol_le hd hbdd)
        (abs_meanRef_le_absMean hD0) (hdev j hj)).2
    have hEabs0 := rstd_tol_rms_dev hd hdecomp
    obtain ⟨E, hEdef⟩ : ∃ E : ℝ, E = 2 * u32 * Real.sqrt (varRef d init t)
        + (1 + 2 * u32) * (meanTol d init t
            + u32 * (rmsS d init t + absMean d init t + meanTol d init t) + 4 * eta32) :=
      ⟨_, rfl⟩
    have hEabs : |Real.sqrt V0 - Real.sqrt (varRef d init t)| ≤ E := by
      rw [hEdef]
      have e : Real.sqrt V0
          = Real.sqrt ((∑ j ∈ Finset.range d.D, (x j).toReal ^ 2) / (d.D : ℝ)) := by
        rw [hV0def, hXdef]
      rw [e]
      exact hEabs0
    have hE0 : 0 ≤ E := by
      rw [hEdef]
      exact rstd_tol_E_nonneg hu0 hmT0 haM0 hrms0 hη0 (Real.sqrt_nonneg _)
    obtain ⟨y, hydef⟩ : ∃ y : ℝ, y = E * rstdRef d init t := ⟨_, rfl⟩
    have hy0 : 0 ≤ y := by
      rw [hydef]
      exact mul_nonneg hE0 (le_of_lt hR0)
    -- ## the condition number
    have hcdef : cond d init t
        = (meanTol d init t + 2*u32*(absMean d init t + rmsS d init t)) * rstdRef d init t := by
      unfold cond
      rw [h23]
    have hcond0 : 0 ≤ cond d init t := by
      rw [hcdef]
      have h1 : 0 ≤ meanTol d init t + 2*u32*(absMean d init t + rmsS d init t) := by
        have h2 : 0 ≤ 2*u32*(absMean d init t + rmsS d init t) := by
          have h3 : 0 ≤ absMean d init t + rmsS d init t := by linarith
          exact mul_nonneg (by linarith) h3
        linarith
      exact mul_nonneg h1 (le_of_lt hR0)
    have hsvR : Real.sqrt (varRef d init t) * rstdRef d init t ≤ 1 := by
      have h1 : Real.sqrt (varRef d init t)
          ≤ Real.sqrt (varRef d init t + epsR d) :=
        Real.sqrt_le_sqrt (by linarith)
      have h2 : 0 < Real.sqrt (varRef d init t + epsR d) := Real.sqrt_pos.2 hWpos
      have h3 : Real.sqrt (varRef d init t) * rstdRef d init t
          = Real.sqrt (varRef d init t) / Real.sqrt (varRef d init t + epsR d) := by
        unfold rstdRef
        rw [mul_one_div]
      rw [h3, div_le_one h2]
      exact h1
    have hetaR : 5 * eta32 * rstdRef d init t ≤ u32 / 8 := by
      have hRle63 := rstd_tol_R_le (init := init) (t := t) hd
      have hnn : 0 ≤ 5 * eta32 := by linarith
      have h1 : 5 * eta32 * rstdRef d init t ≤ 5 * eta32 * (2:ℝ)^(63:ℤ) :=
        mul_le_mul_of_nonneg_left hRle63 hnn
      have h2 : 5 * eta32 * (2:ℝ)^(63:ℤ) ≤ u32 / 8 := by
        unfold eta32 u32
        norm_num
      linarith
    have hu6 : u32 ≤ 1/1000000 := by rw [hu32]; norm_num
    have hy_le : y ≤ cond d init t + 3 * u32 := by
      rw [hydef, hEdef]
      exact rstd_tol_y_le (u := u32) (mT := meanTol d init t) (rms := rmsS d init t)
        (aM := absMean d init t) (sqrtV := Real.sqrt (varRef d init t))
        (R := rstdRef d init t) (c := cond d init t) (η := eta32)
        hu0 hu16 hmT0 hrms0 haM0 (Real.sqrt_nonneg _) hη0 (le_of_lt hR0) hsvR hcdef hetaR hgood
    have hy_quarter : y ≤ 1/4 + 3*u32 := by linarith [hy_le, hgood]
    have h3u8 : 3*u32 ≤ 1/8 := by rw [hu32]; norm_num
    have hy_half : y ≤ 1/2 := by linarith
    have h3u1000 : 3*u32 ≤ 1/1000 := by rw [hu32]; norm_num
    have h2y4 : 2*y + (2*y)^2 ≤ 1 := by
      have h1 : y^2 ≤ (1/4 + 3*u32)*y := by
        rw [pow_two]
        exact mul_le_mul_of_nonneg_right hy_quarter hy0
      have h34 : (3 + 12*u32)*(1/4 + 3*u32) ≤ 1 := by
        rw [hu32]; norm_num
      have h6 : 4*y^2 ≤ 4*((1/4 + 3*u32)*y) := by linarith [h1]
      have h7 : 2*y + 4*((1/4 + 3*u32)*y) = (3 + 12*u32)*y := by ring
      have h8 : (3 + 12*u32)*y ≤ (3 + 12*u32)*(1/4 + 3*u32) :=
        mul_le_mul_of_nonneg_left hy_quarter (by linarith)
      have h5 : 2*y + (2*y)^2 = 2*y + 4*y^2 := by ring
      rw [h5]
      linarith [h6, h7, h8, h34]
    -- ## the two variances `W0` and `W`
    set W : ℝ := varRef d init t + epsR d with hWdef
    have hW0lo : W * (1 - 2*y) ≤ W0 := by
      have h := rstd_tol_V_lo (V0 := V0) (V := varRef d init t) (eps := epsR d)
        (E := E) (R := rstdRef d init t) hV0nn hV0 heps0 hE0 hR0 hRW hEabs
      have e : W * (1 - 2*y)
          = (varRef d init t + epsR d)
            - 2 * (E * rstdRef d init t) * (varRef d init t + epsR d) := by
        rw [hWdef, hydef]
        ring
      linarith [h, e, hW0def]
    have hW0hi : W0 ≤ W * (1 + y)^2 := by
      rw [hW0def, hWdef, hydef]
      exact rstd_tol_V_hi hV0nn hV0 heps0 hE0 hR0 hRW hEabs
    set sW : ℝ := Real.sqrt W with hsWdef
    set s0 : ℝ := Real.sqrt W0 with hs0def
    have hsWpos : 0 < sW := by rw [hsWdef]; exact Real.sqrt_pos.2 hWpos
    have hs0pos : 0 < s0 := by rw [hs0def]; exact Real.sqrt_pos.2 hW0pos
    have hs0_hi : s0 ≤ sW * (1 + y) := by
      have hy1 : 0 ≤ 1 + y := by linarith
      calc s0 = Real.sqrt W0 := hs0def
        _ ≤ Real.sqrt (W * (1+y)^2) := Real.sqrt_le_sqrt hW0hi
        _ = Real.sqrt W * Real.sqrt ((1+y)^2) := Real.sqrt_mul (le_of_lt hWpos) ((1+y)^2)
        _ = Real.sqrt W * (1+y) := by rw [Real.sqrt_sq hy1]
        _ = sW * (1+y) := by rw [hsWdef]
    have hs0_lo : sW * Real.sqrt (1 - 2*y) ≤ s0 := by
      have h1 : Real.sqrt (W * (1-2*y)) ≤ s0 := by
        rw [hs0def]
        exact Real.sqrt_le_sqrt hW0lo
      have h2 : Real.sqrt (W * (1-2*y)) = sW * Real.sqrt (1-2*y) := by
        rw [Real.sqrt_mul (le_of_lt hWpos) (1-2*y), hsWdef]
      linarith [h1, h2]
    have hsq_le : sW / s0 ≤ 1 + 2*y :=
      rstd_tol_sqrt_ratio_le hs0pos hy0 h2y4 hs0_lo
    have hq_abs : |sW / s0 - 1| ≤ 2*y :=
      rstd_tol_sqrt_ratio hsWpos hs0pos hy0 h2y4 hs0_lo hs0_hi
    -- ## `1/sqrt W0` against `rstdRef`
    set Rs : ℝ := 1 / Real.sqrt W0 with hRsdef
    have hRs0 : 0 < Rs := by
      rw [hRsdef]
      exact one_div_pos.2 (Real.sqrt_pos.2 hW0pos)
    have hRsW : Rs ^ 2 * W0 = 1 := by
      rw [hRsdef]
      have h1 : (1 / Real.sqrt W0)^2 = 1/W0 := by
        rw [div_pow, one_pow, Real.sq_sqrt (le_of_lt hW0pos)]
      rw [h1, one_div, inv_mul_cancel₀ (ne_of_gt hW0pos)]
    have hRinv : rstdRef d init t = 1 / Real.sqrt W := by
      unfold rstdRef
      rfl
    have hRsRel : Rs = rstdRef d init t * (sW / s0) := by
      rw [hRsdef, hRinv, hsWdef, hs0def]
      have hne : Real.sqrt W ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hWpos)
      rw [div_eq_mul_inv, div_eq_mul_inv, div_eq_mul_inv]
      simp only [one_mul]
      rw [← mul_assoc, inv_mul_cancel₀ hne, one_mul]
    have hRs_le : Rs ≤ rstdRef d init t * (1 + 2*y) := by
      rw [hRsRel]
      exact mul_le_mul_of_nonneg_left hsq_le (le_of_lt hR0)
    have hdiff : |Rs - rstdRef d init t| ≤ 2*y*rstdRef d init t := by
      rw [hRsRel]
      have e : rstdRef d init t * (sW / s0) - rstdRef d init t
          = rstdRef d init t * (sW / s0 - 1) := by ring
      rw [e, abs_mul, abs_of_pos hR0]
      calc rstdRef d init t * |sW / s0 - 1| ≤ rstdRef d init t * (2*y) :=
            mul_le_mul_of_nonneg_left hq_abs (le_of_lt hR0)
        _ = 2*y*rstdRef d init t := by ring
    -- ## `epsV` against `epsR`
    have hη2 : eta32 ≤ (1/2)*(u32*epsR d) := by
      have h1 : (1/2)*u32*(2:ℝ)^(-125:ℤ) ≤ (1/2)*(u32*epsR d) := by
        have h2 : u32*((2:ℝ)^(-125:ℤ)) ≤ u32*epsR d :=
          mul_le_mul_of_nonneg_left hepsgeR (by linarith [hu0])
        have h3 : (1/2)*(u32*((2:ℝ)^(-125:ℤ))) ≤ (1/2)*(u32*epsR d) :=
          mul_le_mul_of_nonneg_left h2 (by norm_num)
        have e : (1/2)*u32*(2:ℝ)^(-125:ℤ) = (1/2)*(u32*((2:ℝ)^(-125:ℤ))) := by ring
        rw [e]; exact h3
      linarith [hηhalf, h1]
    have hEpsDev : |epsV.toReal - epsR d| ≤ 3*u32/2*W0 :=
      rstd_tol_epsdev (u := u32) (e := epsR d) (epsV := epsV.toReal) (W := W0)
        (η := eta32) heps hη2 hu0 (le_of_lt heps0) hW0ge_eps
    have hSbd : V0 + epsV.toReal ≤ W0*(1 + 3*u32/2) := by
      have h := (abs_le.mp hEpsDev).2
      have e : V0 + epsR d + 3*u32/2*W0 = W0*(1 + 3*u32/2) := by
        rw [hW0def]
        ring
      linarith [h, e]
    -- ## the relative error of `be`
    set Θ : ℝ := (4*(depth d:ℝ)+16)*u32*(1+3*u32/2) + 3*u32/2 + 2*g2 with hΘdef
    have hbe1 : |be.toReal - W0|
        ≤ |be.toReal - (V0 + epsV.toReal)| + |(V0 + epsV.toReal) - W0| := by
      have h := abs_add_le (be.toReal - (V0 + epsV.toReal)) ((V0 + epsV.toReal) - W0)
      have e : (be.toReal - (V0 + epsV.toReal)) + ((V0 + epsV.toReal) - W0)
          = be.toReal - W0 := by ring
      rwa [e] at h
    have hbe2 : |(V0 + epsV.toReal) - W0| ≤ 3*u32/2*W0 := by
      have e : (V0 + epsV.toReal) - W0 = epsV.toReal - epsR d := by
        rw [hW0def]
        ring
      rw [e]
      exact hEpsDev
    have hbe3 : |be.toReal - (V0 + epsV.toReal)|
        ≤ (4*(depth d:ℝ)+16)*u32*(V0 + epsV.toReal) + t135 := by
      rw [ht135]
      exact hbe.2
    have hbe4 : (4*(depth d:ℝ)+16)*u32*(V0 + epsV.toReal)
        ≤ (4*(depth d:ℝ)+16)*u32*(1+3*u32/2)*W0 := by
      have h0L : (0:ℝ) ≤ 4*(depth d:ℝ)+16 := by positivity
      have h := mul_le_mul_of_nonneg_left hSbd (mul_nonneg h0L hu0)
      have e : (4*(depth d:ℝ)+16)*u32*(W0*(1+3*u32/2))
          = (4*(depth d:ℝ)+16)*u32*(1+3*u32/2)*W0 := by ring
      linarith [h, e]
    have hbe5 : t135 ≤ 2*g2*W0 :=
      rstd_tol_t135_le (e := epsR d) (g := g2) (W := W0) heps0 hg2def hW0ge_eps
    have hbeΘ : |be.toReal - W0| ≤ Θ*W0 := by
      rw [hΘdef]
      have h6 := rstd_tol_comb (A := 4*(depth d:ℝ)+16) (u := u32) (g := g2)
        (W := W0) (t := t135) hbe5
      linarith [hbe1, hbe2, hbe3, hbe4, h6]
    have hΘ0 : 0 ≤ Θ := by
      rw [hΘdef]
      have ha : 0 ≤ (4*(depth d:ℝ)+16)*u32 := mul_nonneg (by linarith) hu0
      have hb : 0 ≤ 1 + 3*u32/2 := by linarith
      have hc : 0 ≤ 3*u32/2 := by linarith
      have hd1 : 0 ≤ (4*(depth d:ℝ)+16)*u32*(1+3*u32/2) := mul_nonneg ha hb
      have he : 0 ≤ 2*g2 := by linarith [hg2nn]
      linarith
    have hΘhalf : Θ ≤ 1/2 := by
      rw [hΘdef]
      have hb : 0 ≤ 1 + 3*u32/2 := by linarith
      have h1 : (4*(depth d:ℝ)+16)*u32*(1+3*u32/2) ≤ (1/1000)*(1+3*u32/2) :=
        mul_le_mul_of_nonneg_right hcoef4 hb
      have h3u : 3*u32/2 ≤ 1/1000 := by rw [hu32]; norm_num
      have h2 : (1/1000)*(1+3*u32/2) ≤ 1/500 := by linarith
      have h3 : 2*g2 ≤ 2*(1/2048) := by linarith [hg2le]
      linarith
    -- ## `2^-148 ≤ u·Rs`
    have hs0le : Real.sqrt W0 ≤ (2:ℝ)^(64:ℤ) := by
      have h1 : Real.sqrt W0 ≤ Real.sqrt ((2:ℝ)^(127:ℤ)) := Real.sqrt_le_sqrt hW0le
      have h3 : ((2:ℝ)^(64:ℤ))^2 = (2:ℝ)^(128:ℤ) := by norm_num
      have h3' : (2:ℝ)^(127:ℤ) ≤ ((2:ℝ)^(64:ℤ))^2 := by rw [h3]; norm_num
      have h2 : Real.sqrt ((2:ℝ)^(127:ℤ)) ≤ (2:ℝ)^(64:ℤ) := by
        calc Real.sqrt ((2:ℝ)^(127:ℤ)) ≤ Real.sqrt (((2:ℝ)^(64:ℤ))^2) :=
              Real.sqrt_le_sqrt h3'
          _ = (2:ℝ)^(64:ℤ) := Real.sqrt_sq (by norm_num)
      linarith
    have hRs_ge : (2:ℝ)^(-64:ℤ) ≤ Rs := by
      rw [hRsdef]
      have h1 : 1/(2:ℝ)^(64:ℤ) ≤ 1/Real.sqrt W0 :=
        one_div_le_one_div_of_le (Real.sqrt_pos.2 hW0pos) hs0le
      have e : (1:ℝ)/(2:ℝ)^(64:ℤ) = (2:ℝ)^(-64:ℤ) := by norm_num
      rwa [e] at h1
    have hj2 : (2:ℝ)^(-148:ℤ) ≤ u32 * Rs := by
      have hmul : u32 * (2:ℝ)^(-64:ℤ) ≤ u32 * Rs := mul_le_mul_of_nonneg_left hRs_ge hu0
      have h2 : (2:ℝ)^(-148:ℤ) ≤ u32*(2:ℝ)^(-64:ℤ) := by unfold u32; norm_num
      linarith
    -- ## the chain
    have hρ0 : 0 ≤ Θ/2 + Θ^2 + 13*u32 := by
      have h1 : 0 ≤ Θ/2 := by linarith
      have h2 : 0 ≤ Θ^2 := sq_nonneg Θ
      have h3 : 0 ≤ 13*u32 := by linarith
      linarith
    have hchain := rstd_tol_chain13 (W := W0) (R := Rs) (Th := Θ)
      (e149 := (2:ℝ)^(-149:ℤ)) (e148 := (2:ℝ)^(-148:ℤ)) rfl rfl
      hW0pos hRs0 hRsW hΘ0 hΘhalf hbeΘ hbsq_acc.2 hbrstd_acc.2 hs63 hj2
    have hfin : |brstd.toReal - rstdRef d init t|
        ≤ rstdRef d init t * (2*y + (1+2*y)*(Θ/2+Θ^2+13*u32)) := by
      have hab : |brstd.toReal - rstdRef d init t|
          ≤ |brstd.toReal - Rs| + |Rs - rstdRef d init t| := by
        have h := abs_add_le (brstd.toReal - Rs) (Rs - rstdRef d init t)
        have e : (brstd.toReal - Rs) + (Rs - rstdRef d init t)
            = brstd.toReal - rstdRef d init t := by ring
        rwa [e] at h
      have h2 : Rs * (Θ/2+Θ^2+13*u32)
          ≤ rstdRef d init t * (1+2*y) * (Θ/2+Θ^2+13*u32) :=
        mul_le_mul_of_nonneg_right hRs_le hρ0
      have e : rstdRef d init t * (1+2*y) * (Θ/2+Θ^2+13*u32) + 2*y*rstdRef d init t
          = rstdRef d init t * (2*y + (1+2*y)*(Θ/2+Θ^2+13*u32)) := by ring
      linarith [hab, hchain, h2, hdiff, e]
    have hfin2 : rstdRef d init t * (2*y + (1+2*y)*(Θ/2+Θ^2+13*u32))
        ≤ rstdRef d init t * (2*(cond d init t + gam d)) := by
      apply mul_le_mul_of_nonneg_left _ (le_of_lt hR0)
      have hcc : 2*y ≤ 2*cond d init t + 6*u32 := by linarith [hy_le]
      have hu2 : 1 + 2*y ≤ 1 + (2*cond d init t + 6*u32) := by linarith [hy_le]
      have h3 : (1+2*y)*(Θ/2+Θ^2+13*u32)
          ≤ (1 + (2*cond d init t + 6*u32))*(Θ/2+Θ^2+13*u32) :=
        mul_le_mul_of_nonneg_right hu2 hρ0
      have hgam : gam d = (2*(depth d:ℝ)+20)*u32 + g2 := by
        unfold gam
        rw [hg2def, h23]
        ring
      have hnum := rstd_tol_final_num (L := (depth d:ℝ)) (u := u32) (g2 := g2)
        (c := cond d init t) (Θ := Θ) (ρ := Θ/2+Θ^2+13*u32)
        hL0 hL10 hu0 hu6 hg2nn hg2le hΘ0 (le_of_eq hΘdef) rfl hcond0 hgood
      rw [hgam]
      linarith [hcc, h3, hnum]
    have hTol : rstdTol d init t = 2*(cond d init t + gam d)*rstdRef d init t := by
      unfold rstdTol
      rw [ite_eq_left hgood]
    rw [hTol]
    calc |brstd.toReal - rstdRef d init t|
        ≤ rstdRef d init t * (2*y + (1+2*y)*(Θ/2+Θ^2+13*u32)) := hfin
      _ ≤ rstdRef d init t * (2*(cond d init t + gam d)) := hfin2
      _ = 2*(cond d init t + gam d)*rstdRef d init t := by ring
  · -- ### the ill-conditioned branch
    have hRle : rstdRef d init t ≤ 2 / Real.sqrt (epsR d) := by
      have h1 : rstdRef d init t ≤ 1 / Real.sqrt (epsR d) := by
        rw [rstd_tol_R_eq hWpos hR0 hRW]
        apply one_div_le_one_div_of_le (Real.sqrt_pos.2 heps0)
        exact Real.sqrt_le_sqrt (by linarith [hV0])
      have h2 : 1 / Real.sqrt (epsR d) ≤ 2 / Real.sqrt (epsR d) := by
        have h3 : (0:ℝ) ≤ Real.sqrt (epsR d) := le_of_lt (Real.sqrt_pos.2 heps0)
        have h4 := div_le_div_of_nonneg_right (by norm_num : (1:ℝ) ≤ 2) h3
        linarith
      linarith
    have hbad := rstd_tol_bad (be := be.toReal) (bsq := bsq.toReal)
      (brstd := brstd.toReal) (R := rstdRef d init t) (eps := epsR d)
      heps0 hepsgeR hepsleR hbe_lo hbe_hi hbsq_acc.2 hbrstd_acc.2 hR0 hRle
    have hTol : rstdTol d init t = 2 / Real.sqrt (epsR d) := by
      unfold rstdTol
      rw [ite_eq_right hgood]
    rw [hTol]
    exact hbad


end

end Claims.LayerNormFwd
