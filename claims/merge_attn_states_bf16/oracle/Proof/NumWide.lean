import Claim
import TVA.Proof
import TVA

/-!
# Numerical tools

Elementary bounds on `Real.exp` / `Real.log`, a wide variant of `AddRel.err`, and the
finiteness of a rounding to nearest into `bfloat16` up to the overflow threshold.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## Real analysis -/

/-- `x · exp (-x) ≤ 1`. -/
theorem mul_exp_neg_le (x : ℝ) : x * Real.exp (-x) ≤ 1 := by
  have h := Real.mul_exp_neg_le_exp_neg_one x
  have h1 : Real.exp (-1 : ℝ) ≤ 1 := by
    rw [Real.exp_le_one_iff]; norm_num
  linarith

/-- For `t ≤ 0`, `|exp t - 1| ≤ |t|`. -/
theorem abs_exp_sub_one_le_of_nonpos {t : ℝ} (ht : t ≤ 0) : |Real.exp t - 1| ≤ |t| := by
  have hle : Real.exp t ≤ 1 := Real.exp_le_one_iff.mpr ht
  have hge : t + 1 ≤ Real.exp t := Real.add_one_le_exp t
  rw [abs_of_nonpos (by linarith), abs_of_nonpos ht]
  linarith

/-- **The mean value bound for `exp`.** -/
theorem exp_sub_le (a b : ℝ) : |Real.exp a - Real.exp b| ≤ |a - b| * Real.exp (max a b) := by
  rcases le_total a b with h | h
  · have hmax : max a b = b := max_eq_right h
    have e1 : Real.exp b * Real.exp (a - b) = Real.exp a := by
      rw [← Real.exp_add]; congr 1; ring
    have hfac : Real.exp a - Real.exp b = Real.exp b * (Real.exp (a - b) - 1) := by
      rw [mul_sub, mul_one, e1]
    have habs : |Real.exp (a - b) - 1| ≤ |a - b| := abs_exp_sub_one_le_of_nonpos (by linarith)
    rw [hmax, hfac, abs_mul, abs_of_nonneg (Real.exp_pos b).le]
    calc Real.exp b * |Real.exp (a - b) - 1| ≤ Real.exp b * |a - b| :=
          mul_le_mul_of_nonneg_left habs (Real.exp_pos b).le
      _ = |a - b| * Real.exp b := by ring
  · have hmax : max a b = a := max_eq_left h
    have e1 : Real.exp a * Real.exp (b - a) = Real.exp b := by
      rw [← Real.exp_add]; congr 1; ring
    have hfac : Real.exp a - Real.exp b = Real.exp a * (1 - Real.exp (b - a)) := by
      rw [mul_sub, mul_one, e1]
    have habs : |1 - Real.exp (b - a)| ≤ |b - a| := by
      rw [abs_sub_comm]; exact abs_exp_sub_one_le_of_nonpos (by linarith)
    rw [hmax, hfac, abs_mul, abs_of_nonneg (Real.exp_pos a).le]
    calc Real.exp a * |1 - Real.exp (b - a)| ≤ Real.exp a * |b - a| :=
          mul_le_mul_of_nonneg_left habs (Real.exp_pos a).le
      _ = |a - b| * Real.exp a := by rw [abs_sub_comm b a]; ring

/-- Two nearby non-positive arguments have nearby exponentials. -/
theorem exp_close {a b : ℝ} (hb : b ≤ 0)
    (h : |a - b| ≤ 2 ^ (-24 : ℤ) * |b| + 2 ^ (-134 : ℤ)) :
    |Real.exp a - Real.exp b| ≤ 2 ^ (-20 : ℤ) := by
  set t := -b with ht
  have ht0 : 0 ≤ t := by rw [ht]; linarith
  have hbt : b = -t := by rw [ht]; ring
  have habs : |b| = t := by rw [ht]; exact abs_of_nonpos hb
  have h' : |a - b| ≤ 2 ^ (-24 : ℤ) * t + 2 ^ (-134 : ℤ) := by rw [habs] at h; exact h
  have hle : a - b ≤ 2 ^ (-24 : ℤ) * t + 2 ^ (-134 : ℤ) := le_trans (le_abs_self _) h'
  have hcpos : (0 : ℝ) < 2 ^ (-24 : ℤ) := by positivity
  have hdpos : (0 : ℝ) < 2 ^ (-134 : ℤ) := by positivity
  have hkpos : (0 : ℝ) < 1 - 2 ^ (-24 : ℤ) := by norm_num
  have haC : a ≤ -(1 - 2 ^ (-24 : ℤ)) * t + 2 ^ (-134 : ℤ) := by
    rw [hbt] at hle; nlinarith
  have hbC : b ≤ -(1 - 2 ^ (-24 : ℤ)) * t + 2 ^ (-134 : ℤ) := by
    rw [hbt]; nlinarith
  have hmaxC : max a b ≤ -(1 - 2 ^ (-24 : ℤ)) * t + 2 ^ (-134 : ℤ) := max_le haC hbC
  have hexp : Real.exp (max a b) ≤
      Real.exp (2 ^ (-134 : ℤ) + -(1 - 2 ^ (-24 : ℤ)) * t) :=
    Real.exp_le_exp.mpr (by rw [add_comm]; exact hmaxC)
  rw [Real.exp_add] at hexp
  have hstep0 : 0 ≤ 2 ^ (-24 : ℤ) * t + 2 ^ (-134 : ℤ) := by
    have := mul_nonneg hcpos.le ht0; linarith
  have step1 : |Real.exp a - Real.exp b| ≤
      (2 ^ (-24 : ℤ) * t + 2 ^ (-134 : ℤ)) *
        (Real.exp (2 ^ (-134 : ℤ)) * Real.exp (-(1 - 2 ^ (-24 : ℤ)) * t)) :=
    (exp_sub_le a b).trans (mul_le_mul h' hexp (Real.exp_nonneg _) hstep0)
  set A := Real.exp (-(1 - 2 ^ (-24 : ℤ)) * t) with hA
  have hA0 : 0 ≤ A := Real.exp_nonneg _
  have hA1 : A ≤ 1 := by
    rw [hA, Real.exp_le_one_iff]
    nlinarith [mul_nonneg hkpos.le ht0]
  have hm : ((1 - 2 ^ (-24 : ℤ)) * t) * Real.exp (-((1 - 2 ^ (-24 : ℤ)) * t)) ≤ 1 :=
    mul_exp_neg_le ((1 - 2 ^ (-24 : ℤ)) * t)
  have hme : ((1 - 2 ^ (-24 : ℤ)) * t) * Real.exp (-((1 - 2 ^ (-24 : ℤ)) * t)) =
      (1 - 2 ^ (-24 : ℤ)) * (t * A) := by
    rw [hA]; ring
  rw [hme] at hm
  have hdiv2 : (1 : ℝ) / (1 - 2 ^ (-24 : ℤ)) ≤ 2 := by
    rw [div_le_iff₀ hkpos]; norm_num
  have hAk2 : t * A ≤ 2 := by
    have heq : t * A = (1 / (1 - 2 ^ (-24 : ℤ))) * ((1 - 2 ^ (-24 : ℤ)) * (t * A)) := by
      field_simp
    rw [heq]
    calc (1 / (1 - 2 ^ (-24 : ℤ))) * ((1 - 2 ^ (-24 : ℤ)) * (t * A)) ≤ 2 * 1 :=
          mul_le_mul hdiv2 hm (mul_nonneg hkpos.le (mul_nonneg ht0 hA0)) (by norm_num)
      _ = 2 := by norm_num
  have hexp3 : Real.exp (2 ^ (-134 : ℤ)) ≤ 3 := by
    have hc : (2 : ℝ) ^ (-134 : ℤ) ≤ 1 := by norm_num
    calc Real.exp (2 ^ (-134 : ℤ)) ≤ Real.exp 1 := Real.exp_le_exp.mpr hc
      _ ≤ 3 := le_of_lt Real.exp_one_lt_three
  have hexpand : (2 ^ (-24 : ℤ) * t + 2 ^ (-134 : ℤ)) *
      (Real.exp (2 ^ (-134 : ℤ)) * A) =
      Real.exp (2 ^ (-134 : ℤ)) *
        (2 ^ (-24 : ℤ) * (t * A) + 2 ^ (-134 : ℤ) * A) := by ring
  have hbound : Real.exp (2 ^ (-134 : ℤ)) *
      (2 ^ (-24 : ℤ) * (t * A) + 2 ^ (-134 : ℤ) * A) ≤
      (3 : ℝ) * (2 ^ (-23 : ℤ) + 2 ^ (-134 : ℤ)) := by
    have h1 : 2 ^ (-24 : ℤ) * (t * A) ≤ 2 ^ (-23 : ℤ) := by
      have hh := mul_le_mul_of_nonneg_left hAk2 hcpos.le
      calc (2 : ℝ) ^ (-24 : ℤ) * (t * A) ≤ 2 ^ (-24 : ℤ) * 2 := hh
        _ = 2 ^ (-23 : ℤ) := by norm_num
    have h2 : 2 ^ (-134 : ℤ) * A ≤ 2 ^ (-134 : ℤ) := by
      have hh := mul_le_mul_of_nonneg_left hA1 hdpos.le
      simpa using hh
    have hsum : 2 ^ (-24 : ℤ) * (t * A) + 2 ^ (-134 : ℤ) * A ≤
        2 ^ (-23 : ℤ) + 2 ^ (-134 : ℤ) := by linarith
    have hsum0 : 0 ≤ 2 ^ (-24 : ℤ) * (t * A) + 2 ^ (-134 : ℤ) * A := by
      have hh := mul_nonneg hcpos.le (mul_nonneg ht0 hA0)
      have hh2 := mul_nonneg hdpos.le hA0
      linarith
    exact mul_le_mul hexp3 hsum hsum0 (by norm_num)
  have hnum : (3 : ℝ) * (2 ^ (-23 : ℤ) + 2 ^ (-134 : ℤ)) ≤ 2 ^ (-20 : ℤ) := by norm_num
  calc |Real.exp a - Real.exp b|
      ≤ (2 ^ (-24 : ℤ) * t + 2 ^ (-134 : ℤ)) *
          (Real.exp (2 ^ (-134 : ℤ)) * A) := step1
    _ = Real.exp (2 ^ (-134 : ℤ)) *
          (2 ^ (-24 : ℤ) * (t * A) + 2 ^ (-134 : ℤ) * A) := hexpand
    _ ≤ (3 : ℝ) * (2 ^ (-23 : ℤ) + 2 ^ (-134 : ℤ)) := hbound
    _ ≤ 2 ^ (-20 : ℤ) := hnum

/-- **The mean value bound for `log`.** -/
theorem log_sub_le {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    |Real.log a - Real.log b| ≤ |a - b| / min a b := by
  rcases le_total a b with h | h
  · have hmin : min a b = a := min_eq_left h
    have hl : Real.log b - Real.log a ≤ (b - a) / a := by
      have h1 := Real.log_le_sub_one_of_pos (div_pos hb ha)
      rw [Real.log_div hb.ne' ha.ne'] at h1
      have he : b / a - 1 = (b - a) / a := by field_simp
      rwa [he] at h1
    have h1 : |Real.log a - Real.log b| = Real.log b - Real.log a := by
      rw [abs_of_nonpos (show Real.log a - Real.log b ≤ 0 by linarith [Real.log_le_log ha h])]
      ring
    have h2 : |a - b| = b - a := by
      rw [abs_of_nonpos (show a - b ≤ 0 by linarith)]; ring
    rw [hmin, h1, h2]
    exact hl
  · have hmin : min a b = b := min_eq_right h
    have hl : Real.log a - Real.log b ≤ (a - b) / b := by
      have h1 := Real.log_le_sub_one_of_pos (div_pos ha hb)
      rw [Real.log_div ha.ne' hb.ne'] at h1
      have he : a / b - 1 = (a - b) / b := by field_simp
      rwa [he] at h1
    have h1 : |Real.log a - Real.log b| = Real.log a - Real.log b := by
      rw [abs_of_nonneg (show 0 ≤ Real.log a - Real.log b by linarith [Real.log_le_log hb h])]
    have h2 : |a - b| = a - b := by
      rw [abs_of_nonneg (show 0 ≤ a - b by linarith)]
    rw [hmin, h1, h2]
    exact hl

/-- Two nearby positive arguments, the second not too small, have nearby logs. -/
theorem log_close {a b : ℝ} (ha : 0 < a) (hb : 1 - 2 ^ (-18 : ℤ) ≤ b)
    (h : |a - b| ≤ 2 ^ (-18 : ℤ)) : |Real.log a - Real.log b| ≤ 2 ^ (-17 : ℤ) := by
  have hbpos : (0 : ℝ) < b := by
    have : (0 : ℝ) < 1 - 2 ^ (-18 : ℤ) := by norm_num
    linarith
  have hamin : 1 - 2 ^ (-17 : ℤ) ≤ a := by
    have h2 := (abs_sub_le_iff.mp h).2
    have h1 : (1 : ℝ) - 2 ^ (-18 : ℤ) - 2 ^ (-18 : ℤ) = 1 - 2 ^ (-17 : ℤ) := by norm_num
    linarith
  have hpow : (2 : ℝ) ^ (-18 : ℤ) ≤ 2 ^ (-17 : ℤ) := by norm_num
  have hmin : 1 - 2 ^ (-17 : ℤ) ≤ min a b := le_min hamin (by linarith)
  have hminpos : (0 : ℝ) < min a b := by
    have : (0 : ℝ) < 1 - 2 ^ (-17 : ℤ) := by norm_num
    linarith
  have hnum : (2 : ℝ) ^ (-18 : ℤ) / (1 - 2 ^ (-17 : ℤ)) ≤ 2 ^ (-17 : ℤ) := by norm_num
  have hle : |a - b| / min a b ≤ (2 : ℝ) ^ (-18 : ℤ) / (1 - 2 ^ (-17 : ℤ)) := by
    have h1 : |a - b| / min a b ≤ (2 : ℝ) ^ (-18 : ℤ) / min a b :=
      (div_le_div_iff_of_pos_right hminpos).mpr h
    have h2 : (2 : ℝ) ^ (-18 : ℤ) / min a b ≤ (2 : ℝ) ^ (-18 : ℤ) / (1 - 2 ^ (-17 : ℤ)) :=
      (div_le_div_iff_of_pos_left (by positivity) hminpos (by norm_num)).mpr hmin
    linarith
  exact (log_sub_le ha hbpos).trans (hle.trans hnum)

/-- A convex combination is bounded by the largest element. -/
theorem abs_le_max_of_weights {w₁ w₂ x y : ℝ} (h₁ : 0 ≤ w₁) (h₂ : 0 ≤ w₂) (h : w₁ + w₂ = 1) :
    |w₁ * x + w₂ * y| ≤ max |x| |y| := by
  have hx : |x| ≤ max |x| |y| := le_max_left _ _
  have hy : |y| ≤ max |x| |y| := le_max_right _ _
  have h1 : |w₁ * x| = w₁ * |x| := by rw [abs_mul, abs_of_nonneg h₁]
  have h2 : |w₂ * y| = w₂ * |y| := by rw [abs_mul, abs_of_nonneg h₂]
  have h3 : w₁ * |x| ≤ w₁ * max |x| |y| := mul_le_mul_of_nonneg_left hx h₁
  have h4 : w₂ * |y| ≤ w₂ * max |x| |y| := mul_le_mul_of_nonneg_left hy h₂
  have h5 : w₁ * max |x| |y| + w₂ * max |x| |y| = max |x| |y| := by
    rw [← add_mul, h, one_mul]
  calc |w₁ * x + w₂ * y| ≤ |w₁ * x| + |w₂ * y| := abs_add_le _ _
    _ = w₁ * |x| + w₂ * |y| := by rw [h1, h2]
    _ ≤ max |x| |y| := by linarith

/-! ## The wide addition bound -/

/-- **A contractible addition of finite operands is within three roundings**, under a hypothesis
that only bounds the exact sum plus the rounding allowance. -/
theorem addRel_err_w {f : Format} (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16)
    {fusion : Bool} {x y r : FloatV f} (h : AddRel fusion x y r)
    (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hM : |x.toRat + y.toRat| + f.u * (|x.toRat| + |y.toRat|) + 2 * f.eta ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - (x.toRat + y.toRat)| ≤ 3 * f.u * (|x.toRat| + |y.toRat|) + 4 * f.eta := by
  have hu0 := Format.u_nonneg (f := f)
  have he0 := Format.eta_nonneg (f := f)
  have hS0 : 0 ≤ |x.toRat| + |y.toRat| := by positivity
  have hxy : |x.toRat + y.toRat| ≤ |x.toRat| + |y.toRat| := abs_add_le _ _
  have hxyM : |x.toRat + y.toRat| ≤ f.maxVal := by
    have h1 : 0 ≤ f.u * (|x.toRat| + |y.toRat|) := mul_nonneg hu0 hS0
    linarith
  rcases h with hne | ⟨-, ⟨p, q, hp, hq, hpq_eq, z, hr⟩ | ⟨p, hp, hyi, -⟩ | ⟨q, hq, hxi, -⟩⟩
  · -- the IEEE sum
    obtain ⟨hfin, herr⟩ := FloatV.add_err hf hx hy hxyM
    rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
    refine ⟨hfin, herr.trans ?_⟩
    nlinarith [mul_le_mul_of_nonneg_right hxy hu0]
  · -- the fused sum
    have ep := RoundsTo.err hf hs hp hx
    have eq := RoundsTo.err hf hs hq hy
    have hpq : |p + q| ≤ (1 + f.u) * (|x.toRat| + |y.toRat|) + 2 * f.eta := by
      have h1 := abs_sub_abs_le_abs_sub p x.toRat
      have h2 := abs_sub_abs_le_abs_sub q y.toRat
      have := abs_add_le p q
      nlinarith
    have hpqM : |p + q| ≤ f.maxVal := by
      have hqy : |q - y.toRat| ≤ |y.toRat| * f.u + f.eta := eq
      have hpx : |p - x.toRat| ≤ |x.toRat| * f.u + f.eta := ep
      rcases hpq_eq with hpx' | hqy'
      · have h1 : |p + q| ≤ |x.toRat + y.toRat| + |q - y.toRat| := by
          have e : p + q = (x.toRat + y.toRat) + (q - y.toRat) := by rw [hpx']; ring
          rw [e]; exact abs_add_le _ _
        have h2 : f.u * |y.toRat| ≤ f.u * (|x.toRat| + |y.toRat|) := by
          have : |y.toRat| ≤ |x.toRat| + |y.toRat| := by linarith [abs_nonneg x.toRat]
          exact mul_le_mul_of_nonneg_left this hu0
        linarith
      · have h1 : |p + q| ≤ |x.toRat + y.toRat| + |p - x.toRat| := by
          have e : p + q = (x.toRat + y.toRat) + (p - x.toRat) := by rw [hqy']; ring
          rw [e]; exact abs_add_le _ _
        have h2 : f.u * |x.toRat| ≤ f.u * (|x.toRat| + |y.toRat|) := by
          have : |x.toRat| ≤ |x.toRat| + |y.toRat| := by linarith [abs_nonneg y.toRat]
          exact mul_le_mul_of_nonneg_left this hu0
        linarith
    obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hf false z hpqM
    rw [hr.eq (FloatV.isFinite_iff.mp hfin).1]
    refine ⟨hfin, ?_⟩
    have e : (FloatV.ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (x.toRat + y.toRat) =
        ((FloatV.ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q)) +
          (p - x.toRat) + (q - y.toRat) := by ring
    rw [e]
    have t1 := abs_add_le (((FloatV.ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q)) +
      (p - x.toRat)) (q - y.toRat)
    have t2 := abs_add_le ((FloatV.ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q))
      (p - x.toRat)
    have hb : |p + q| * f.u ≤ ((1 + f.u) * (|x.toRat| + |y.toRat|) + 2 * f.eta) * f.u :=
      mul_le_mul_of_nonneg_right hpq hu0
    nlinarith [mul_nonneg hu0 hS0, mul_nonneg (mul_nonneg hu0 hu0) hS0, mul_nonneg hu0 he0]
  · exfalso; rw [(FloatV.isFinite_iff.mp hy).2] at hyi; cases hyi
  · exfalso; rw [(FloatV.isFinite_iff.mp hx).2] at hxi; cases hxi

/-! ## Rounding into `bfloat16` -/

/-- Rounding to nearest of `x` below `k + 1/2` is at most `k`. -/
theorem rnat_rn_le {x : ℚ} {k : ℕ} (hx : 0 ≤ x) (h : x < (k : ℚ) + 1 / 2) :
    Format.rnat .rn x ≤ k := by
  have h1 := Format.rnat_rn_err (x := x) hx
  have h2 : (Format.rnat .rn x : ℚ) < (k : ℚ) + 1 := by
    have := (abs_le.mp h1).2
    linarith
  have h3 : Format.rnat .rn x < k + 1 := by exact_mod_cast h2
  omega

/-- Rounding a magnitude below `(255 + 1/2) · 2^120` to nearest is a valid, finite pattern. -/
theorem roundRat_bf16_rn_spec (zs : Bool) {q : ℚ}
    (hq : |q| < (255 + 1 / 2) * 2 ^ (120 : ℤ)) :
    Format.bfloat16.roundRat .rn false zs q < 2 ^ Format.bfloat16.width ∧
      Format.bfloat16.isFinite (Format.bfloat16.roundRat .rn false zs q) = true := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨h1, h2, -⟩ := Format.roundRat_zero_spec (f := Format.bfloat16) (rnd := .rn)
      (sat := false) (zs := zs) Format.bfloat16_regular
    exact ⟨h1, h2⟩
  have ha : 0 < |q| := abs_pos.mpr hq0
  have hmax : ((Format.roundSig .rn |q| (Format.bfloat16.qexp |q|) : ℕ) : ℚ) *
      (2 : ℚ) ^ (Format.bfloat16.qexp |q|) ≤ Format.bfloat16.maxVal := by
    set Q : ℤ := Format.bfloat16.qexp |q| with hQdef
    obtain ⟨hNle, hQge, hQsub⟩ := Format.roundSig_bounds (f := Format.bfloat16) .rn ha
    rw [← hQdef] at hNle hQge hQsub
    set N : ℕ := Format.roundSig .rn |q| Q with hNdef
    have hm : Format.bfloat16.m = 7 := rfl
    have hem : Format.bfloat16.emin = -126 := rfl
    have hmaxv : Format.bfloat16.maxVal = 255 * (2 : ℚ) ^ (120 : ℤ) := by
      rw [Format.bfloat16_maxVal]; norm_num
    have hN256 : N ≤ 256 := by
      have h := hNle
      rw [hm] at h
      norm_num at h
      exact h
    by_cases hQle120 : Q ≤ 120
    · by_cases hQlt120 : Q < 120
      · have hQ119 : Q ≤ 119 := by omega
        have hNq : (N : ℚ) ≤ 256 := by exact_mod_cast hN256
        have hQpow : (2 : ℚ) ^ Q ≤ (2 : ℚ) ^ (119 : ℤ) :=
          zpow_le_zpow_right₀ (by norm_num) hQ119
        have h1 : (N : ℚ) * (2 : ℚ) ^ Q ≤ 256 * (2 : ℚ) ^ Q :=
          mul_le_mul_of_nonneg_right hNq (by positivity)
        have h2 : (256 : ℚ) * (2 : ℚ) ^ Q ≤ 255 * (2 : ℚ) ^ (120 : ℤ) := by
          have h3 : (256 : ℚ) * (2 : ℚ) ^ Q ≤ 256 * (2 : ℚ) ^ (119 : ℤ) :=
            mul_le_mul_of_nonneg_left hQpow (by norm_num)
          have h4 : (256 : ℚ) * (2 : ℚ) ^ (119 : ℤ) ≤ 255 * (2 : ℚ) ^ (120 : ℤ) := by
            norm_num
          linarith
        rw [hmaxv]
        linarith
      · have hQeq : Q = 120 := by omega
        have hx0 : (0 : ℚ) ≤ |q| * (2 : ℚ) ^ (-(120 : ℤ)) := by positivity
        have hxlt : |q| * (2 : ℚ) ^ (-(120 : ℤ)) < 255 + 1 / 2 := by
          have hpos : (0 : ℚ) < (2 : ℚ) ^ (-(120 : ℤ)) := zpow_pos (by norm_num) _
          have h1 := mul_lt_mul_of_pos_right hq hpos
          have e : ((255 + 1 / 2) * (2 : ℚ) ^ (120 : ℤ)) * (2 : ℚ) ^ (-(120 : ℤ)) =
              255 + 1 / 2 := by
            rw [mul_assoc, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]; norm_num
          rw [e] at h1
          exact h1
        have hN255 : Format.rnat .rn (|q| * (2 : ℚ) ^ (-(120 : ℤ))) ≤ 255 := rnat_rn_le hx0 hxlt
        have hNq : ((Format.rnat .rn (|q| * (2 : ℚ) ^ (-(120 : ℤ))) : ℕ) : ℚ) ≤ 255 := by
          exact_mod_cast hN255
        have hp : (0 : ℚ) < (2 : ℚ) ^ (120 : ℤ) := zpow_pos (by norm_num) _
        rw [hNdef, Format.roundSig_eq, hQeq, hmaxv]
        calc ((Format.rnat .rn (|q| * (2 : ℚ) ^ (-(120 : ℤ))) : ℕ) : ℚ) * (2 : ℚ) ^ (120 : ℤ)
            ≤ 255 * (2 : ℚ) ^ (120 : ℤ) := mul_le_mul_of_nonneg_right hNq hp.le
          _ ≤ 255 * (2 : ℚ) ^ (120 : ℤ) := le_rfl
    · exfalso
      have hQge121 : (121 : ℤ) ≤ Q := by omega
      have hQunfold : Q = max (Format.qlog2 |q|) (-126) - 7 := by
        rw [hQdef]; unfold Format.qexp; rw [hem, hm]; norm_num
      have hmaxge : (128 : ℤ) ≤ max (Format.qlog2 |q|) (-126) := by omega
      have hqlog : (128 : ℤ) ≤ Format.qlog2 |q| := by
        have := le_max_left (Format.qlog2 |q|) (-126); omega
      obtain ⟨hspec, -⟩ := Format.qlog2_spec (a := |q|) ha
      have h2pow : (2 : ℚ) ^ (128 : ℤ) ≤ |q| := by
        calc (2 : ℚ) ^ (128 : ℤ) ≤ (2 : ℚ) ^ (Format.qlog2 |q|) :=
              zpow_le_zpow_right₀ (by norm_num) hqlog
          _ ≤ |q| := hspec
      have hlt : |q| < (2 : ℚ) ^ (128 : ℤ) := by
        calc |q| < (255 + 1 / 2) * (2 : ℚ) ^ (120 : ℤ) := hq
          _ < (2 : ℚ) ^ (128 : ℤ) := by norm_num
      linarith
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.bfloat16) .rn ha
  have hle := Format.pack_le_maxMag Format.bfloat16_regular hN hQ hsub hmax
  have hlt : Format.bfloat16.pack (Format.roundSig .rn |q| (Format.bfloat16.qexp |q|))
      (Format.bfloat16.qexp |q|) < Format.bfloat16.signBit :=
    hle.trans_lt (Format.maxMag_lt Format.bfloat16_regular)
  have hpat : Format.bfloat16.roundRat .rn false zs q =
      Format.bfloat16.setSign (decide (q < 0))
        (Format.bfloat16.pack (Format.roundSig .rn |q| (Format.bfloat16.qexp |q|))
          (Format.bfloat16.qexp |q|)) := by
    rw [Format.roundRat_of_ne hq0]
    unfold Format.roundMag Format.encode
    rw [Format.pow2_eq, ite_eq_left hmax, Format.setSign_of_lt hlt]
    by_cases h : q < 0 <;> simp [h]
  refine ⟨?_, ?_⟩
  · rw [hpat]; exact Format.setSign_lt _ _
  · rw [hpat, Format.isFinite_setSign]
    exact Format.isFinite_of_mag_le Format.bfloat16_regular
      (by rw [Format.mag_of_lt hlt]; exact hle)

/-- **The `float32` to `bfloat16` conversion of an in-range finite value**: it is finite and
within one `bfloat16` rounding of the input. -/
theorem cvtWith_bf16_rn {x : F32} (hx : x.isFinite = true)
    (hq : |x.toRat| < (255 + 1 / 2) * 2 ^ (120 : ℤ)) :
    (FloatV.cvtWith .rn x : BF16).isFinite = true ∧
      |(FloatV.cvtWith .rn x : BF16).toRat - x.toRat| ≤
        |(FloatV.cvtWith .rn x : BF16).toRat| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := by
  have hcvt : (FloatV.cvtWith .rn x : BF16) =
      FloatV.ofPat (Format.bfloat16.roundRat .rn false (Format.binary32.sign x.pat)
        (Format.binary32.toRat x.pat)) := by
    unfold FloatV.cvtWith
    rw [Format.cvt_of_finite (f := Format.binary32) (g := Format.bfloat16) .rn false hx]
  have hq' : |Format.binary32.toRat x.pat| < (255 + 1 / 2) * 2 ^ (120 : ℤ) := hq
  obtain ⟨hlt, hfin⟩ := roundRat_bf16_rn_spec (Format.binary32.sign x.pat) hq'
  have herr := Format.roundRat_rn_err_of_finite (f := Format.bfloat16) Format.bfloat16_regular
    (by decide : Format.bfloat16.specials ≠ .finite) (Format.binary32.sign x.pat) hfin
  rw [hcvt, FloatV.isFinite_ofPat (f := Format.bfloat16) hlt,
    FloatV.toRat_ofPat (f := Format.bfloat16) hlt]
  refine ⟨hfin, ?_⟩
  rw [Format.bfloat16_u, Format.bfloat16_eta] at herr
  show |Format.bfloat16.toRat (Format.bfloat16.roundRat .rn false (Format.binary32.sign x.pat)
      (Format.binary32.toRat x.pat)) - Format.binary32.toRat x.pat| ≤
    |Format.bfloat16.toRat (Format.bfloat16.roundRat .rn false (Format.binary32.sign x.pat)
      (Format.binary32.toRat x.pat))| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)
  exact herr

end Claims.MergeAttnStates
