import Claim
import TVA.Proof

/-!
# The softplus, in the reals

The kernel computes `S = tl.where (1*xf <= 20) (1*log(1 + exp(1*xf))) xf` lane by lane.  These two
lemmas bound `|S - spRef x0|` by `52*2^-23*spRef x0 + 2^-24*(1 + 2^-19)` in each branch, where `x0`
is the exact sum `a + dt_bias` in the reals and the hypotheses are the rounding errors of the
steps that produced `xf = fl(x0)`, `t1 = fl(1*xf)`, `t2 = fl(1*xf)`, `e ~ exp t2`, `s = fl(1 + e)`,
`l ~ log s`, `l2 = fl(1*l)`.

The README (`Tolerances`) gets `(5.2 + 1.5*m)U` with `U = 2^-23`, `m = min(max(-x0,0),20)`; the
constant `52` here is a uniform, generous version of it (`m <= 20`).
-/

namespace Claims.GdnGating

open TVA

/-- On the log branch `spRef` is the logarithm. -/
lemma spRef_eq_log {x : ℝ} (h : x ≤ 20) : spRef x = Real.log (1 + Real.exp x) := by
  simp only [spRef, ite_eq_left h]

/-- On the identity branch `spRef x = x`. -/
lemma spRef_eq_self {x : ℝ} (h : 20 < x) : spRef x = x := by
  simp only [spRef, ite_eq_right (not_le.mpr h)]

/-- `spRef x` is positive. -/
lemma spRef_pos (x : ℝ) : 0 < spRef x := by
  rcases le_or_gt x 20 with h | h
  · rw [spRef_eq_log h]
    apply Real.log_pos
    linarith [Real.exp_pos x]
  · rw [spRef_eq_self h]
    linarith

/-- The logarithm splits off the exponential of `x`. -/
lemma log_one_add_exp_eq (x : ℝ) :
    Real.log (1 + Real.exp x) = x + Real.log (1 + Real.exp (-x)) := by
  have hx : Real.exp x ≠ 0 := ne_of_gt (Real.exp_pos x)
  have h1 : (0 : ℝ) < 1 + Real.exp (-x) := by linarith [Real.exp_pos (-x)]
  have hexp : Real.exp x * Real.exp (-x) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hfac : 1 + Real.exp x = Real.exp x * (1 + Real.exp (-x)) := by
    rw [mul_add, mul_one, hexp]
    ring
  rw [hfac, Real.log_mul hx (ne_of_gt h1), Real.log_exp]

/-- `log (1 + exp x)` dominates `spRef x`. -/
lemma spRef_le_log (x : ℝ) : spRef x ≤ Real.log (1 + Real.exp x) := by
  rcases le_or_gt x 20 with h | h
  · rw [spRef_eq_log h]
  · rw [spRef_eq_self h, log_one_add_exp_eq]
    have h0 : 0 ≤ Real.log (1 + Real.exp (-x)) := by
      apply Real.log_nonneg
      linarith [Real.exp_pos (-x)]
    linarith

/-- `log (1 + t) ≤ t` for `t ≥ 0`. -/
lemma log_one_add_le_self {t : ℝ} (h : 0 ≤ t) : Real.log (1 + t) ≤ t := by
  have := Real.log_le_sub_one_of_pos (by linarith : (0 : ℝ) < 1 + t)
  linarith

/-- A numerical bound: `exp (-20) ≤ 2^(-20)`. -/
lemma exp_neg20_le : Real.exp (-20) ≤ (2 : ℝ) ^ (-20 : ℤ) := by
  have h2 : (2 : ℝ) ≤ Real.exp 1 := le_of_lt Real.exp_one_gt_two
  have hp2 : (0 : ℝ) < (2 : ℝ) ^ 20 := by norm_num
  have hpow : (2 : ℝ) ^ 20 ≤ (Real.exp 1) ^ 20 := pow_le_pow_left₀ (by norm_num) h2 20
  have he : (Real.exp 1) ^ 20 = Real.exp 20 := by
    rw [← Real.exp_nat_mul]
    norm_num
  have hpow' : (2 : ℝ) ^ 20 ≤ Real.exp 20 := by
    rw [← he]
    exact hpow
  have hpos : (0 : ℝ) < Real.exp 20 := Real.exp_pos 20
  have hz : (2 : ℝ) ^ (-20 : ℤ) = ((2 : ℝ) ^ 20)⁻¹ := by
    rw [zpow_neg]
    rfl
  rw [Real.exp_neg, hz]
  exact (inv_le_inv₀ hpos hp2).mpr hpow'

/-- The gap between the two softplus forms is at most `2^(-20)`. -/
lemma log_sub_spRef_le (x : ℝ) : Real.log (1 + Real.exp x) - spRef x ≤ 2 ^ (-20 : ℤ) := by
  have h0 : (0 : ℝ) ≤ 2 ^ (-20 : ℤ) := by norm_num
  rcases le_or_gt x 20 with h | h
  · rw [spRef_eq_log h]
    linarith
  · rw [spRef_eq_self h, log_one_add_exp_eq]
    have hx20 : Real.exp (-x) ≤ Real.exp (-20) := by
      apply Real.exp_le_exp.mpr
      linarith
    have hb : Real.exp (-x) ≤ 2 ^ (-20 : ℤ) := le_trans hx20 exp_neg20_le
    have hlog : Real.log (1 + Real.exp (-x)) ≤ Real.exp (-x) :=
      log_one_add_le_self (le_of_lt (Real.exp_pos _))
    linarith

/-- `exp d ≤ 1 + 2 d` for `0 ≤ d ≤ 1/2`. -/
lemma exp_le_one_add_two_mul {d : ℝ} (h0 : 0 ≤ d) (h1 : d ≤ 1 / 2) :
    Real.exp d ≤ 1 + 2 * d := by
  have hp : (0 : ℝ) < Real.exp (-d) := Real.exp_pos _
  have hge : 1 - d ≤ Real.exp (-d) := by
    have := Real.add_one_le_exp (-d)
    linarith
  have hpos : (0 : ℝ) < 1 - d := by linarith
  have he : Real.exp d = (Real.exp (-d))⁻¹ := by
    rw [Real.exp_neg, inv_inv]
  rw [he]
  have hstep : (Real.exp (-d))⁻¹ ≤ (1 - d)⁻¹ := (inv_le_inv₀ hp hpos).mpr hge
  have h2 : (1 - d)⁻¹ ≤ 1 + 2 * d := by
    rw [← one_div, div_le_iff₀ hpos]
    nlinarith
  linarith

/-- `|exp d - 1| ≤ 2 |d|` for `|d| ≤ 1/2`. -/
lemma abs_exp_sub_one_le {d : ℝ} (h : |d| ≤ 1 / 2) : |Real.exp d - 1| ≤ 2 * |d| := by
  rcases le_or_gt 0 d with hd | hd
  · have hpos : (0 : ℝ) ≤ Real.exp d - 1 := by
      have := Real.one_le_exp hd
      linarith
    rw [abs_of_nonneg hpos, abs_of_nonneg hd]
    have := exp_le_one_add_two_mul hd (abs_le.mp h).2
    linarith
  · have hd0 : d < 0 := hd
    have hneg : Real.exp d - 1 < 0 := by
      have hlt : Real.exp d < 1 := by
        rw [← Real.exp_zero]
        exact Real.exp_lt_exp.mpr hd0
      linarith
    rw [abs_of_neg hneg, abs_of_neg hd0]
    have hnd : (0 : ℝ) ≤ -d := by linarith
    have hd' : -d ≤ 1 / 2 := by
      have := (abs_le.mp h).1
      linarith
    have h2 := exp_le_one_add_two_mul hnd hd'
    have hmul : Real.exp d * Real.exp (-d) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    have hp : (0 : ℝ) < Real.exp d := Real.exp_pos d
    have hle1 : Real.exp d ≤ 1 := by
      rw [← Real.exp_zero]
      exact Real.exp_le_exp.mpr (le_of_lt hd0)
    have hge0 : (0 : ℝ) ≤ Real.exp (-d) - 1 := by
      have := Real.add_one_le_exp (-d)
      linarith
    nlinarith [hmul, h2, hp, hle1, hge0, hnd]

/-- `|log (1 + z)| ≤ |z| (1 + 2 |z|)` for `|z| ≤ 1/2`. -/
lemma abs_log_one_add_le {z : ℝ} (h : |z| ≤ 1 / 2) :
    |Real.log (1 + z)| ≤ |z| * (1 + 2 * |z|) := by
  have hlow : -(1 / 2 : ℝ) ≤ z := (abs_le.mp h).1
  have hup : z ≤ 1 / 2 := (abs_le.mp h).2
  have hz1 : (0 : ℝ) < 1 + z := by linarith
  rcases le_or_gt 0 z with hz | hz
  · have hlog : Real.log (1 + z) ≤ z := log_one_add_le_self hz
    have h0 : 0 ≤ Real.log (1 + z) := Real.log_nonneg (by linarith)
    rw [abs_of_nonneg h0, abs_of_nonneg hz]
    nlinarith [abs_nonneg z]
  · have hz0 : z < 0 := hz
    have hneg : Real.log (1 + z) < 0 := by
      rw [← Real.log_one]
      exact Real.log_lt_log hz1 (by linarith)
    rw [abs_of_neg hneg, abs_of_neg hz0]
    have hw : (0 : ℝ) < -z := by linarith
    have hw2 : -z ≤ 1 / 2 := by linarith
    have hfac : (1 + z)⁻¹ = 1 + (-z) * (1 + z)⁻¹ := by
      field_simp
      ring
    have hinv : (1 + z)⁻¹ ≤ 1 + 2 * (-z) := by
      rw [← one_div, div_le_iff₀ hz1]
      nlinarith
    have hchain : -Real.log (1 + z) ≤ (-z) * (1 + 2 * (-z)) := by
      calc -Real.log (1 + z) = Real.log ((1 + z)⁻¹) := by rw [Real.log_inv]
        _ = Real.log (1 + (-z) * (1 + z)⁻¹) := congrArg Real.log hfac
        _ ≤ (-z) * (1 + z)⁻¹ := log_one_add_le_self (by positivity)
        _ ≤ (-z) * (1 + 2 * (-z)) := mul_le_mul_of_nonneg_left hinv (le_of_lt hw)
    linarith


/-- `t / (1 + t) ≤ log (1 + t)` for `t ≥ 0`. -/
lemma div_self_le_log_one_add {t : ℝ} (h : 0 ≤ t) :
    t / (1 + t) ≤ Real.log (1 + t) := by
  have hu : (0 : ℝ) < 1 + t := by linarith
  have hinv : Real.log ((1 + t)⁻¹) ≤ (1 + t)⁻¹ - 1 :=
    Real.log_le_sub_one_of_pos (by positivity)
  rw [Real.log_inv] at hinv
  have heq : t / (1 + t) = 1 - (1 + t)⁻¹ := by
    field_simp
    ring
  rw [heq]
  linarith

/-- The logistic factor is dominated by the softplus value. -/
lemma log_one_add_exp_ge_ratio (x : ℝ) :
    Real.exp x / (1 + Real.exp x) ≤ Real.log (1 + Real.exp x) :=
  div_self_le_log_one_add (le_of_lt (Real.exp_pos x))

/-- `t ↦ exp t / (1 + exp t)` is monotone. -/
lemma exp_div_one_add_exp_le_of_le {a b : ℝ} (h : a ≤ b) :
    Real.exp a / (1 + Real.exp a) ≤ Real.exp b / (1 + Real.exp b) := by
  have ha : (0 : ℝ) < Real.exp a := Real.exp_pos a
  have hb : (0 : ℝ) < Real.exp b := Real.exp_pos b
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith [Real.exp_le_exp.mpr h]

/-- Shifting the logistic factor by `d ≥ 0` costs a factor `1 + 2 d`. -/
lemma exp_div_shift_le {v d : ℝ} (hv : 0 < v) (hd : 0 ≤ d) (hd2 : d ≤ 1 / 2) :
    Real.exp d * v / (1 + Real.exp d * v) ≤ v / (1 + v) * (1 + 2 * d) := by
  have hs : Real.exp d ≤ 1 + 2 * d := exp_le_one_add_two_mul hd hd2
  have h1 : (0 : ℝ) < 1 + Real.exp d * v := by positivity
  have h2 : (0 : ℝ) < 1 + v := by linarith
  have hmul : Real.exp d * v * (1 + v) ≤ v * (1 + 2 * d) * (1 + Real.exp d * v) := by
    have hstep : Real.exp d * v ≤ (1 + 2 * d) * v := mul_le_mul_of_nonneg_right hs (le_of_lt hv)
    nlinarith [hstep, hv, Real.exp_pos d, sq_nonneg v, sq_nonneg (Real.exp d * v)]
  rw [div_le_iff₀ h1]
  have hrw : v / (1 + v) * (1 + 2 * d) * (1 + Real.exp d * v)
      = v * (1 + 2 * d) * (1 + Real.exp d * v) / (1 + v) := by ring
  rw [hrw, le_div_iff₀ h2]
  linarith

/-- The softplus logarithm dominates its argument. -/
lemma log_one_add_exp_ge_self (x : ℝ) : x ≤ Real.log (1 + Real.exp x) := by
  have h : Real.exp x ≤ 1 + Real.exp x := by linarith [Real.exp_pos x]
  have hlog : Real.log (Real.exp x) ≤ Real.log (1 + Real.exp x) :=
    Real.log_le_log (Real.exp_pos x) h
  rwa [Real.log_exp] at hlog

/-- `log 2 + x/2 ≤ log (1 + exp x)`. -/
lemma log_two_add_half_le (x : ℝ) :
    Real.log 2 + x / 2 ≤ Real.log (1 + Real.exp x) := by
  have ht : (0 : ℝ) < Real.exp (x / 2) := Real.exp_pos _
  have hmul : Real.exp (x / 2) * Real.exp (x / 2) = Real.exp x := by
    rw [← Real.exp_add]
    ring_nf
  have hone : Real.exp (x / 2) * Real.exp (-(x / 2)) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hinv : Real.exp (-(x / 2)) = (Real.exp (x / 2))⁻¹ := Real.exp_neg _
  have hsq : (0 : ℝ) ≤ (Real.exp (x / 2) - 1) ^ 2 := sq_nonneg _
  have hgoal : 2 * Real.exp (x / 2) ≤ Real.exp (x / 2) ^ 2 + 1 := by nlinarith [hsq]
  have hdiv : 2 ≤ (Real.exp (x / 2) ^ 2 + 1) / Real.exp (x / 2) :=
    (le_div_iff₀ ht).mpr hgoal
  have heq : (Real.exp (x / 2) ^ 2 + 1) / Real.exp (x / 2)
      = Real.exp (x / 2) + (Real.exp (x / 2))⁻¹ := by
    field_simp
  have hsum : 2 ≤ Real.exp (x / 2) + Real.exp (-(x / 2)) := by
    rw [hinv, ← heq]
    exact hdiv
  have hp2 : (0 : ℝ) < 2 := by norm_num
  have hpos : (0 : ℝ) < Real.exp (x / 2) + Real.exp (-(x / 2)) := by linarith
  have hlogineq : Real.log 2 ≤ Real.log (Real.exp (x / 2) + Real.exp (-(x / 2))) :=
    Real.log_le_log hp2 hsum
  have hfac : 1 + Real.exp x
      = Real.exp (x / 2) * (Real.exp (x / 2) + Real.exp (-(x / 2))) := by
    rw [mul_add, hmul, hone]
    ring
  have hlogfac : Real.log (Real.exp (x / 2) * (Real.exp (x / 2) + Real.exp (-(x / 2))))
      = x / 2 + Real.log (Real.exp (x / 2) + Real.exp (-(x / 2))) := by
    rw [Real.log_mul (ne_of_gt ht) (ne_of_gt hpos), Real.log_exp]
  rw [hfac, hlogfac]
  linarith

/-- `log 2 ≤ log (1 + exp x)` for `x ≥ 0`. -/
lemma log_two_le_log_one_add_exp {x : ℝ} (h : 0 ≤ x) :
    Real.log 2 ≤ Real.log (1 + Real.exp x) := by
  have := log_two_add_half_le x
  linarith

/-- For `x ≤ 0` the log branch value is at most `1`. -/
lemma log_one_add_exp_le_one {x : ℝ} (h : x ≤ 0) :
    Real.log (1 + Real.exp x) ≤ 1 := by
  have hx : Real.exp x ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr h
  have h2 : Real.log 2 ≤ 1 := by
    have := Real.log_le_log (by norm_num : (0 : ℝ) < 2) (le_of_lt Real.exp_one_gt_two)
    rwa [Real.log_exp] at this
  have hlog : Real.log (1 + Real.exp x) ≤ Real.log 2 :=
    Real.log_le_log (by linarith [Real.exp_pos x]) (by linarith)
  linarith


/-- For `|z| ≤ 2^(-15)` the log increment is at most `|z| (1 + 2^(-14))`. -/
lemma abs_log_one_add_le_small {z : ℝ} (h : |z| ≤ 2 ^ (-15 : ℤ)) :
    |Real.log (1 + z)| ≤ |z| * (1 + 2 ^ (-14 : ℤ)) := by
  have h2 : (2 : ℝ) ^ (-15 : ℤ) ≤ 1 / 2 := by norm_num [zpow_neg]
  have hb := abs_log_one_add_le (le_trans h h2)
  have hfac : 1 + 2 * |z| ≤ 1 + 2 ^ (-14 : ℤ) := by
    have h3 : 2 * (2 : ℝ) ^ (-15 : ℤ) = 2 ^ (-14 : ℤ) := by norm_num [zpow_neg]
    nlinarith [abs_nonneg z]
  nlinarith [abs_nonneg z, hb, hfac]

/-- A bound on the log-difference caused by the error in `e`. -/
lemma T1_bound {t2 e u : ℝ} (hu : u = 2 ^ (-24 : ℤ))
    (he : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + 2 ^ (-140 : ℤ))
    (ht2 : |t2| ≤ 22) :
    |Real.log (1 + e) - Real.log (1 + Real.exp t2)| ≤
      (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) * (1 + 2 ^ (-14 : ℤ))
        + 2 ^ (-138 : ℤ) := by
  have hu0 : 0 < u := by rw [hu]; norm_num
  have hpos : (0 : ℝ) < Real.exp t2 := Real.exp_pos t2
  have hden : (0 : ℝ) < 1 + Real.exp t2 := by linarith
  have hr1 : Real.exp t2 / (1 + Real.exp t2) ≤ 1 := by
    rw [div_le_one (by linarith)]
    linarith
  have hs1le : 4 * u + 2 * u * |t2| ≤ 2 ^ (-16 : ℤ) := by
    have h : 2 * u * |t2| ≤ 2 * u * 22 := by
      apply mul_le_mul_of_nonneg_left ht2
      linarith
    have h2 : (4 : ℝ) * u + 2 * u * 22 ≤ 2 ^ (-16 : ℤ) := by
      rw [hu]
      norm_num [zpow_neg]
    linarith
  set z : ℝ := (e - Real.exp t2) / (1 + Real.exp t2) with hzdef
  have hzb : |z| ≤ (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) + 2 ^ (-140 : ℤ) := by
    rw [hzdef, abs_div, abs_of_pos hden]
    have h1 : |e - Real.exp t2| / (1 + Real.exp t2)
        ≤ ((4 * u + 2 * u * |t2|) * Real.exp t2 + 2 ^ (-140 : ℤ)) / (1 + Real.exp t2) :=
      div_le_div_of_nonneg_right he (le_of_lt hden)
    have h2 : ((4 * u + 2 * u * |t2|) * Real.exp t2 + 2 ^ (-140 : ℤ)) / (1 + Real.exp t2)
        = (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2))
          + 2 ^ (-140 : ℤ) / (1 + Real.exp t2) := by
      field_simp
    have h4 : (2 : ℝ) ^ (-140 : ℤ) / (1 + Real.exp t2) ≤ 2 ^ (-140 : ℤ) := by
      rw [div_le_iff₀ hden]
      have hq : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) := by norm_num [zpow_neg]
      nlinarith [Real.exp_pos t2, hq]
    linarith
  have hs1r : (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) ≤ 2 ^ (-16 : ℤ) := by
    have hr0 : (0 : ℝ) ≤ Real.exp t2 / (1 + Real.exp t2) :=
      div_nonneg (le_of_lt hpos) (le_of_lt hden)
    calc (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2))
        ≤ 2 ^ (-16 : ℤ) * (Real.exp t2 / (1 + Real.exp t2)) :=
          mul_le_mul_of_nonneg_right hs1le hr0
      _ ≤ 2 ^ (-16 : ℤ) * 1 := by
          apply mul_le_mul_of_nonneg_left hr1
          norm_num [zpow_neg]
      _ = 2 ^ (-16 : ℤ) := by ring
  have hzabs : |z| ≤ 2 ^ (-15 : ℤ) := by
    have h2 : (2 : ℝ) ^ (-140 : ℤ) ≤ 2 ^ (-16 : ℤ) := by norm_num [zpow_neg]
    have h3 : (2 : ℝ) ^ (-16 : ℤ) + 2 ^ (-16 : ℤ) ≤ 2 ^ (-15 : ℤ) := by norm_num [zpow_neg]
    have hsum : (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) + 2 ^ (-140 : ℤ)
        ≤ 2 ^ (-15 : ℤ) := by linarith
    exact le_trans hzb hsum
  have hzpos : 0 < 1 + z := by
    have h1 := (abs_le.mp hzabs).1
    have h2 : (2 : ℝ) ^ (-15 : ℤ) < 1 := by norm_num [zpow_neg]
    linarith
  have hz1 : 1 + e = (1 + Real.exp t2) * (1 + z) := by
    rw [hzdef]
    field_simp
    ring
  have hkey : Real.log (1 + e) - Real.log (1 + Real.exp t2) = Real.log (1 + z) := by
    rw [hz1, Real.log_mul (ne_of_gt hden) (ne_of_gt hzpos)]
    ring
  rw [hkey]
  have hsmall := abs_log_one_add_le_small hzabs
  have hstep : |z| * (1 + 2 ^ (-14 : ℤ))
      ≤ ((4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) + 2 ^ (-140 : ℤ))
        * (1 + 2 ^ (-14 : ℤ)) := by
    apply mul_le_mul_of_nonneg_right hzb
    norm_num
  have hk2 : (2 : ℝ) ^ (-140 : ℤ) * (1 + 2 ^ (-14 : ℤ)) ≤ 2 ^ (-138 : ℤ) := by
    norm_num [zpow_neg]
  have hexpand : ((4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) + 2 ^ (-140 : ℤ))
        * (1 + 2 ^ (-14 : ℤ))
      = (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) * (1 + 2 ^ (-14 : ℤ))
        + 2 ^ (-140 : ℤ) * (1 + 2 ^ (-14 : ℤ)) := by ring
  calc |Real.log (1 + z)| ≤ |z| * (1 + 2 ^ (-14 : ℤ)) := hsmall
    _ ≤ ((4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) + 2 ^ (-140 : ℤ))
          * (1 + 2 ^ (-14 : ℤ)) := hstep
    _ = (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) * (1 + 2 ^ (-14 : ℤ))
        + 2 ^ (-140 : ℤ) * (1 + 2 ^ (-14 : ℤ)) := hexpand
    _ ≤ (4 * u + 2 * u * |t2|) * (Real.exp t2 / (1 + Real.exp t2)) * (1 + 2 ^ (-14 : ℤ))
        + 2 ^ (-138 : ℤ) := by linarith [hk2]

/-- For `t ≤ 0`, `|t| * exp t ≤ exp (t/2)`. -/
lemma abs_mul_exp_le_exp_half {t : ℝ} (h : t ≤ 0) :
    |t| * Real.exp t ≤ Real.exp (t / 2) := by
  have ht : |t| = -t := abs_of_nonpos h
  have ht0 : (0 : ℝ) ≤ -t := by linarith
  have h1 : |t| ≤ Real.exp (|t| / 2) := by
    rw [ht]
    have h3 : 1 + (-t) / 4 ≤ Real.exp ((-t) / 4) := by
      linarith [Real.add_one_le_exp ((-t) / 4)]
    have h4 : (0 : ℝ) ≤ Real.exp ((-t) / 4) := le_of_lt (Real.exp_pos _)
    have h5 : Real.exp ((-t) / 2) = Real.exp ((-t) / 4) * Real.exp ((-t) / 4) := by
      rw [← Real.exp_add]; ring_nf
    have h6 : (1 + (-t) / 4) ^ 2 ≤ (Real.exp ((-t) / 4)) ^ 2 := by
      apply pow_le_pow_left₀ (by linarith : (0 : ℝ) ≤ 1 + (-t) / 4) h3
    have h7 : -t ≤ (1 + (-t) / 4) ^ 2 := by
      nlinarith [sq_nonneg ((-t) / 4 - 1)]
    have h8 : (Real.exp ((-t) / 4)) ^ 2 = Real.exp ((-t) / 2) := by
      rw [h5]; ring
    linarith
  have h8 : |t| * Real.exp t ≤ Real.exp (|t| / 2) * Real.exp t :=
    mul_le_mul_of_nonneg_right h1 (le_of_lt (Real.exp_pos t))
  have he : Real.exp (|t| / 2) * Real.exp t = Real.exp (t / 2) := by
    rw [← Real.exp_add]
    congr 1
    rw [ht]; ring
  linarith

/-- `|exp d - 1| ≤ |d| (1 + 2|d|)` for `|d| ≤ 1/2`. -/
lemma abs_exp_sub_one_le_mul {d : ℝ} (hd : |d| ≤ 1 / 2) :
    |Real.exp d - 1| ≤ |d| * (1 + 2 * |d|) := by
  rcases le_or_gt 0 d with h | h
  · have hd0 : 0 ≤ d := h
    have hd1 : d ≤ 1 / 2 := (abs_le.mp hd).2
    have hm : Real.exp d * Real.exp (-d) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    have hexp : Real.exp d - 1 = Real.exp d * (1 - Real.exp (-d)) := by
      rw [mul_sub, mul_one, hm]
    have h1 : 0 ≤ 1 - Real.exp (-d) := by
      have hz : Real.exp (-d) ≤ Real.exp 0 := Real.exp_le_exp.mpr (by linarith)
      rw [Real.exp_zero] at hz
      linarith
    have h2 : 1 - Real.exp (-d) ≤ d := by
      have := Real.add_one_le_exp (-d); linarith
    have h3 : Real.exp d ≤ 1 + 2 * d := exp_le_one_add_two_mul hd0 hd1
    have hpos1 : 0 ≤ Real.exp d - 1 := by
      rw [hexp]
      exact mul_nonneg (le_of_lt (Real.exp_pos d)) h1
    rw [abs_of_nonneg hpos1, abs_of_nonneg hd0, hexp]
    nlinarith [h2, h3, h1, Real.exp_pos d]
  · have hd0 : d < 0 := h
    have hneg : Real.exp d - 1 < 0 := by
      have hlt : Real.exp d < Real.exp 0 := Real.exp_lt_exp.mpr hd0
      rw [Real.exp_zero] at hlt
      linarith
    have h2 : 1 - Real.exp d ≤ -d := by
      have := Real.add_one_le_exp d; linarith
    have h3 : (0 : ℝ) ≤ -d := by linarith
    rw [abs_of_neg hneg, abs_of_neg hd0]
    have h4 : (-d) ≤ (-d) * (1 + 2 * (-d)) := by nlinarith
    linarith

/-- The log-softplus difference is bounded by the logistic factor times `|b - a|`. -/
lemma abs_log_one_add_exp_sub_le {a b : ℝ} (hd : |b - a| ≤ 1 / 4) :
    |Real.log (1 + Real.exp b) - Real.log (1 + Real.exp a)| ≤
      |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 8 * |b - a|) := by
  have ha : 0 < Real.exp a := Real.exp_pos a
  have hden : 0 < 1 + Real.exp a := by linarith
  have hd2 : |b - a| ≤ 1 / 2 := by linarith
  have hd0 : 0 ≤ |b - a| := abs_nonneg _
  have hsig0 : 0 ≤ Real.exp a / (1 + Real.exp a) := div_nonneg (le_of_lt ha) (le_of_lt hden)
  have hsig1 : Real.exp a / (1 + Real.exp a) ≤ 1 := by
    rw [div_le_one hden]; linarith
  have hfac : Real.exp b = Real.exp a * Real.exp (b - a) := by
    rw [← Real.exp_add]; ring_nf
  have hnum : |Real.exp b - Real.exp a| ≤
      Real.exp a * (|b - a| * (1 + 2 * |b - a|)) := by
    have hsub : Real.exp a * Real.exp (b - a) - Real.exp a
        = Real.exp a * (Real.exp (b - a) - 1) := by ring
    rw [hfac, hsub, abs_mul, abs_of_pos ha]
    exact mul_le_mul_of_nonneg_left (abs_exp_sub_one_le_mul hd2) (le_of_lt ha)
  set z : ℝ := (Real.exp b - Real.exp a) / (1 + Real.exp a) with hzdef
  have hz1 : 1 + Real.exp b = (1 + Real.exp a) * (1 + z) := by
    rw [hzdef]; field_simp; ring
  have hzb : |z| ≤ |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|) := by
    rw [hzdef, abs_div, abs_of_pos hden, div_le_iff₀ hden]
    calc |Real.exp b - Real.exp a|
        ≤ Real.exp a * (|b - a| * (1 + 2 * |b - a|)) := hnum
      _ = |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|) * (1 + Real.exp a) := by
          field_simp
  have hm0 : 0 ≤ |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|) := by positivity
  have hzsmall : |z| ≤ 1 / 2 := by
    have hA : |b - a| * (Real.exp a / (1 + Real.exp a)) ≤ 1 / 4 := by
      have h := mul_le_mul hd hsig1 hsig0 (by norm_num : (0 : ℝ) ≤ 1 / 4)
      simpa using h
    have hB : 1 + 2 * |b - a| ≤ 3 / 2 := by linarith
    have h2 : |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|) ≤ 3 / 8 := by
      have h := mul_le_mul hA hB (by positivity) (by norm_num : (0 : ℝ) ≤ 1 / 4)
      linarith
    linarith
  have hzpos : 0 < 1 + z := by
    have := (abs_le.mp hzsmall).1
    linarith
  have hkey : Real.log (1 + Real.exp b) - Real.log (1 + Real.exp a) = Real.log (1 + z) := by
    rw [hz1, Real.log_mul (ne_of_gt hden) (ne_of_gt hzpos)]
    ring
  rw [hkey]
  have hsmall := abs_log_one_add_le hzsmall
  have hz0 : 0 ≤ |z| := abs_nonneg _
  have hstep1 : |z| * (1 + 2 * |z|) ≤
      (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)) *
        (1 + 2 * (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|))) := by
    apply mul_le_mul hzb
    · nlinarith [hzb]
    · positivity
    · exact hm0
  have hstep2 :
      (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)) *
        (1 + 2 * (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)))
      ≤ |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 8 * |b - a|) := by
    have hm_le : |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)
        ≤ |b - a| * (1 + 2 * |b - a|) := by
      have h := mul_le_mul_of_nonneg_left hsig1
        (by positivity : (0 : ℝ) ≤ |b - a| * (1 + 2 * |b - a|))
      nlinarith [h]
    have h1 : 1 + 2 * (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|))
        ≤ 1 + 2 * (|b - a| * (1 + 2 * |b - a|)) := by linarith
    have h2 : (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)) *
        (1 + 2 * (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)))
        ≤ (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)) *
          (1 + 2 * (|b - a| * (1 + 2 * |b - a|))) :=
      mul_le_mul_of_nonneg_left h1 hm0
    have hsc : (1 + 2 * |b - a|) * (1 + 2 * (|b - a| * (1 + 2 * |b - a|)))
        ≤ 1 + 8 * |b - a| := by
      nlinarith [hd, hd0, sq_nonneg (|b - a|)]
    have hds : 0 ≤ |b - a| * (Real.exp a / (1 + Real.exp a)) := mul_nonneg hd0 hsig0
    have h3 : (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)) *
          (1 + 2 * (|b - a| * (1 + 2 * |b - a|)))
        ≤ |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 8 * |b - a|) := by
      calc (|b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 2 * |b - a|)) *
            (1 + 2 * (|b - a| * (1 + 2 * |b - a|)))
          = (|b - a| * (Real.exp a / (1 + Real.exp a))) *
            ((1 + 2 * |b - a|) * (1 + 2 * (|b - a| * (1 + 2 * |b - a|)))) := by ring
        _ ≤ (|b - a| * (Real.exp a / (1 + Real.exp a))) * (1 + 8 * |b - a|) :=
            mul_le_mul_of_nonneg_left hsc hds
        _ = |b - a| * (Real.exp a / (1 + Real.exp a)) * (1 + 8 * |b - a|) := by ring
    linarith
  linarith


/-- `1 + e` stays above `1/2`. -/
lemma sp_one_add_e {e t2 u c140 : ℝ} (hu0 : 0 < u) (hu48 : 48 * u ≤ 1 / 2)
    (hc140 : c140 ≤ 1 / 2)
    (he : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + c140)
    (ht2 : |t2| ≤ 22) :
    1 / 2 ≤ 1 + e := by
  have hexp0 : (0 : ℝ) < Real.exp t2 := Real.exp_pos t2
  have hcoef : 4 * u + 2 * u * |t2| ≤ 48 * u := by
    have h : 2 * u * |t2| ≤ 2 * u * 22 := mul_le_mul_of_nonneg_left ht2 (by linarith)
    linarith
  have hfac : (0 : ℝ) ≤ 1 - (4 * u + 2 * u * |t2|) := by linarith
  have hmul : (0 : ℝ) ≤ (1 - (4 * u + 2 * u * |t2|)) * Real.exp t2 :=
    mul_nonneg hfac (le_of_lt hexp0)
  have h1 := (abs_le.mp he).1
  have h2 : -c140 ≤ e := by nlinarith [h1, hmul, hexp0]
  linarith

/-- Rounding of `s = fl(1+e)`: the induced log error is at most `u + 2^(-46)`. -/
lemma sp_A3 {s e u c150 c46 : ℝ} (hu0 : 0 < u) (hc150 : 0 ≤ c150)
    (hwu : u + 2 * c150 ≤ 1 / 2)
    (hge : 1 / 2 ≤ 1 + e)
    (hs : |s - (1 + e)| ≤ u * (1 + e) + c150)
    (hlf : (u + 2 * c150) * (1 + 2 * (u + 2 * c150)) ≤ u + c46) :
    |Real.log s - Real.log (1 + e)| ≤ u + c46 := by
  have hgepos : (0 : ℝ) < 1 + e := by linarith
  set w : ℝ := s / (1 + e) - 1 with hw
  have hweq : s = (1 + e) * (1 + w) := by
    rw [hw]; field_simp; ring
  have hwb : |w| ≤ u + 2 * c150 := by
    have h4 : |w| = |s - (1 + e)| / (1 + e) := by
      rw [hw, show s / (1 + e) - 1 = (s - (1 + e)) / (1 + e) by field_simp]
      rw [abs_div, abs_of_pos hgepos]
    rw [h4]
    have h1 : |s - (1 + e)| / (1 + e) ≤ (u * (1 + e) + c150) / (1 + e) :=
      div_le_div_of_nonneg_right hs (le_of_lt hgepos)
    have h2 : (u * (1 + e) + c150) / (1 + e) = u + c150 / (1 + e) := by
      field_simp
    have h3 : c150 / (1 + e) ≤ 2 * c150 := by
      rw [div_le_iff₀ hgepos]; nlinarith
    linarith
  have hw12 : |w| ≤ 1 / 2 := by linarith [hwb, hwu]
  have hwne : 1 + w ≠ 0 := by
    have h := (abs_le.mp hw12).1
    linarith
  have hlog : |Real.log s - Real.log (1 + e)| = |Real.log (1 + w)| := by
    rw [hweq, Real.log_mul (ne_of_gt hgepos) hwne]
    congr 1; ring
  rw [hlog]
  have hsmall := abs_log_one_add_le hw12
  have hmono : |w| * (1 + 2 * |w|) ≤ (u + 2 * c150) * (1 + 2 * (u + 2 * c150)) := by
    nlinarith [hwb, abs_nonneg w, hu0, hc150]
  linarith

/-- The logistic factor at `t2` versus at `x0`. -/
lemma sp_hST {t2 x0 : ℝ} (hd14 : |t2 - x0| ≤ 1 / 4) :
    Real.exp t2 / (1 + Real.exp t2) ≤
      Real.exp x0 / (1 + Real.exp x0) * (1 + 2 * |t2 - x0|) := by
  rcases le_or_gt t2 x0 with h | h
  · have h1 := exp_div_one_add_exp_le_of_le h
    have h2 : (0 : ℝ) ≤ 2 * |t2 - x0| := by positivity
    have hs0 : (0 : ℝ) ≤ Real.exp x0 / (1 + Real.exp x0) :=
      div_nonneg (le_of_lt (Real.exp_pos x0)) (by positivity)
    nlinarith [h1, h2, hs0]
  · have hd0 : 0 < t2 - x0 := by linarith
    have hd2 : t2 - x0 ≤ 1 / 2 := by
      have := (abs_le.mp hd14).2
      linarith
    have hv : 0 < Real.exp x0 := Real.exp_pos x0
    have hkey : Real.exp t2 = Real.exp (t2 - x0) * Real.exp x0 := by
      rw [← Real.exp_add]; ring_nf
    have habs : |t2 - x0| = t2 - x0 := abs_of_pos hd0
    rw [hkey, habs]
    exact exp_div_shift_le hv (le_of_lt hd0) hd2

set_option maxHeartbeats 1000000 in
/-- **Collecting the softplus errors**: from the bounds `A1 … A5` on the individual steps (argument rounding, `exp`, `1 + ·`, `log`, product), the total error is at most `100 u L + u + c43`. -/
lemma sp_core {u L s0 sT D A1 A2 A3 A4 A5 S labs c14 c46 c138 c149 c150 c43 : ℝ}
    (hu0 : 0 < u) (hu1e : u ≤ 1 / 1000000)
    (hL0 : 0 ≤ L) (hs0 : 0 ≤ s0) (hs0L : s0 ≤ L) (hS0 : 0 ≤ S)
    (hD0 : 0 ≤ D) (hD44 : D ≤ 44 * u)
    (hc14_0 : 0 ≤ c14) (hc14le : c14 ≤ 1) (hc14s : c14 ≤ 1 / 10000)
    (htiny : c46 + c138 + c150 + c149 + 5.0001 * u * (c46 + c138) + u * c149
      + 5.0001 * (u * u) ≤ c43)
    (hA1 : A1 ≤ u * labs + c150)
    (hA2 : A2 ≤ 4 * u * S + c149)
    (hA3 : A3 ≤ u + c46)
    (hA4 : A4 ≤ 48 * u * sT * (1 + c14) + c138)
    (hA5 : A5 ≤ D * s0 * (1 + 8 * D))
    (hST : sT ≤ s0 * (1 + 2 * D))
    (hS : S ≤ L + A3 + A4 + A5)
    (hlabs : labs ≤ S + A2) :
    A1 + A2 + A3 + A4 + A5 ≤ 100 * u * L + u + c43 := by
  have hD2 : 2 * D ≤ 1 / 10000 := by nlinarith [hD44, hu1e]
  -- A4
  have h4a : 48 * u * sT * (1 + c14) ≤ 48 * u * (s0 * (1 + 2 * D)) * (1 + c14) := by
    have h := mul_le_mul_of_nonneg_left hST
      (show (0:ℝ) ≤ 48 * u * (1 + c14) by nlinarith [hu0, hc14_0])
    nlinarith [h]
  have hprod4 : (1 + 2 * D) * (1 + c14) ≤ 1.002 := by
    have h1 : 2 * D * c14 ≤ 2 * D * 1 :=
      mul_le_mul_of_nonneg_left hc14le (by linarith)
    nlinarith [hD2, h1, hc14s]
  have h4b : 48 * u * (s0 * (1 + 2 * D)) * (1 + c14) ≤ 48.1 * u * s0 := by
    have hcoef : (0:ℝ) ≤ 48 * u * s0 := by nlinarith [hu0, hs0]
    have h := mul_le_mul_of_nonneg_left hprod4 hcoef
    nlinarith [h]
  have hA4' : A4 ≤ 48.1 * u * s0 + c138 := by linarith [hA4, h4a, h4b]
  -- A5
  have hDq : D * D ≤ D * (1 / 10000) := by
    have h := mul_le_mul_of_nonneg_left hD2 hD0
    nlinarith [h]
  have h5a : D * (1 + 8 * D) ≤ 44.1 * u := by nlinarith [hD44, hDq, hD0]
  have h5b : D * s0 * (1 + 8 * D) ≤ 44.1 * u * s0 := by
    have h := mul_le_mul_of_nonneg_right h5a hs0
    nlinarith [h]
  -- K
  have hK : A3 + A4 + A5 ≤ u + 92.2 * u * s0 + (c46 + c138) := by
    linarith [hA3, hA4', h5b]
  -- S
  have hsL : u * s0 ≤ u * L := mul_le_mul_of_nonneg_left hs0L (le_of_lt hu0)
  have hSB : S ≤ L + u + 92.2 * u * L + c46 + c138 := by
    linarith [hS, hK, hsL]
  -- labs
  have hlabs2 : u * labs ≤ u * (S + (4 * u * S + c149)) := by
    apply mul_le_mul_of_nonneg_left _ (le_of_lt hu0)
    linarith [hlabs, hA2]
  have hA12 : A1 + A2 ≤ u * (S + (4 * u * S + c149)) + c150 + (4 * u * S + c149) := by
    linarith [hA1, hA2, hlabs2]
  have hcoefS : 5 * u + 4 * u * u ≤ 5.0001 * u := by
    have h : u * u ≤ u * (1 / 1000000) := mul_le_mul_of_nonneg_left hu1e (le_of_lt hu0)
    nlinarith [h, hu0]
  have hA12' : A1 + A2 ≤ (5.0001 * u) * (L + u + 92.2 * u * L + c46 + c138)
      + u * c149 + c150 + c149 := by
    have h1 : (5 * u + 4 * u * u) * S ≤ (5.0001 * u) * S :=
      mul_le_mul_of_nonneg_right hcoefS hS0
    have h2 : (5.0001 * u) * S ≤ (5.0001 * u) * (L + u + 92.2 * u * L + c46 + c138) :=
      mul_le_mul_of_nonneg_left hSB (by nlinarith [hu0])
    have h3 : (5 * u + 4 * u * u) * S = u * (S + 4 * u * S) + 4 * u * S := by ring
    have h4 : u * (S + (4 * u * S + c149)) + c150 + (4 * u * S + c149)
        = u * (S + 4 * u * S) + 4 * u * S + u * c149 + c150 + c149 := by ring
    linarith [hA12, h1, h2, h3, h4]
  -- final
  have h92 : 92.2 * u * s0 ≤ 92.2 * u * L := mul_le_mul_of_nonneg_left hs0L (by nlinarith [hu0])
  have h461 : 462 * u ≤ 0.5 := by nlinarith [hu1e]
  have hLu : (462 * u) * (u * L) ≤ 0.5 * (u * L) :=
    mul_le_mul_of_nonneg_right h461 (by nlinarith [hu0, hL0])
  have hbig : A1 + A2 + A3 + A4 + A5 ≤ 97.8 * u * L + u + c43 := by
    nlinarith [hA12', hK, h92, hLu, htiny, hL0, hu0]
  nlinarith [hbig, hu0, hL0]

/-- **Sizes in the middle range** (`-20 ≤ x0`, `t1 ≤ 20`): `|x0| ≤ 42`, `|xf| ≤ 22`, `|t2| ≤ 22`, and `t2` is within `44 u` (so `1/4`) of `x0`. -/
lemma sp_caseA_bounds {x0 xf t1 t2 u c150 : ℝ}
    (hu0 : 0 < u) (hu44 : 44 * u ≤ 1 / 4)
    (hxm : -20 ≤ x0) (hle : t1 ≤ 20)
    (hxf : |xf - x0| ≤ u * |x0| + c150)
    (ht1 : |t1 - xf| ≤ u * |xf| + c150)
    (ht2 : |t2 - xf| ≤ u * |xf| + c150)
    (hc : 2 * c150 ≤ u)
    (hcoef : 2 * u + u * u ≤ 1 / 2)
    (hcsmall : u * c150 + 2 * c150 ≤ 1)
    (_hnumf : 42 * u + c150 ≤ 1)
    (_hnum : 20 + 42 * u + 43 * u + 2 * c150 ≤ 21)
    (_hnum2 : 21 * u + c150 ≤ 1) :
    |x0| ≤ 42 ∧ |xf| ≤ 22 ∧ |t2| ≤ 22 ∧ |t2 - x0| ≤ 44 * u ∧ |t2 - x0| ≤ 1 / 4 := by
  have hx0_le : x0 ≤ 20 + u * |x0| + u * |xf| + 2 * c150 := by
    have h1 : |x0 - xf| ≤ u * |x0| + c150 := by
      simpa [abs_sub_comm] using hxf
    have h2 : |xf - t1| ≤ u * |xf| + c150 := by
      simpa [abs_sub_comm] using ht1
    have h3 : x0 - t1 ≤ |x0 - xf| + |xf - t1| := by
      have h := add_le_add (le_abs_self (x0 - xf)) (le_abs_self (xf - t1))
      linarith
    linarith
  have hxf_abs : |xf| ≤ |x0| + u * |x0| + c150 := by
    have h := abs_add_le (xf - x0) x0
    have hx : xf - x0 + x0 = xf := by ring
    rw [hx] at h
    linarith [h, hxf]
  have hx0abs42 : |x0| ≤ 42 := by
    rcases le_or_gt 0 x0 with hp | hn
    · rw [abs_of_nonneg hp] at hx0_le hxf_abs ⊢
      have hu2 : u * |xf| ≤ u * (x0 + u * x0 + c150) :=
        mul_le_mul_of_nonneg_left hxf_abs (le_of_lt hu0)
      have hexp : u * (x0 + u * x0 + c150)
          = u * x0 + (u * u) * x0 + u * c150 := by ring
      have hu3 : u * |xf| ≤ u * x0 + (u * u) * x0 + u * c150 := by
        rw [hexp] at hu2; exact hu2
      have h6 : (2 * u + u * u) * x0 ≤ (1 / 2) * x0 :=
        mul_le_mul_of_nonneg_right hcoef hp
      have h6' : 2 * u * x0 + (u * u) * x0 ≤ (1 / 2) * x0 := by
        rw [add_mul] at h6; exact h6
      have h7 : x0 ≤ 20 + 2 * u * x0 + (u * u) * x0
          + (u * c150 + 2 * c150) := by
        linarith [hx0_le, hu3]
      linarith [h7, h6', hcsmall]
    · rw [abs_of_nonpos (le_of_lt hn)]
      linarith [hxm]
  have hx0abs : |x0| ≤ 21 := by
    rcases le_or_gt 0 x0 with hp | hn
    · rw [abs_of_nonneg hp] at hx0_le hxf_abs hx0abs42 ⊢
      have hu2 : u * x0 ≤ u * 42 := mul_le_mul_of_nonneg_left hx0abs42 (le_of_lt hu0)
      have hxf43 : |xf| ≤ 43 := by linarith [hxf_abs, hx0abs42, hu2]
      have hu3 : u * |xf| ≤ u * 43 := mul_le_mul_of_nonneg_left hxf43 (le_of_lt hu0)
      linarith [hx0_le, hu2, hu3]
    · rw [abs_of_nonpos (le_of_lt hn)]
      linarith [hxm]
  have hxf22 : |xf| ≤ 22 := by
    have hu21 : u * |x0| ≤ u * 21 := mul_le_mul_of_nonneg_left hx0abs (le_of_lt hu0)
    linarith [hxf_abs, hu21]
  have hd44 : |t2 - x0| ≤ 44 * u := by
    have h := abs_sub_le t2 xf x0
    have hu : u * |xf| ≤ u * 22 := mul_le_mul_of_nonneg_left hxf22 (le_of_lt hu0)
    have hu2 : u * |x0| ≤ u * 21 := mul_le_mul_of_nonneg_left hx0abs (le_of_lt hu0)
    linarith [h, ht2, hxf, hu, hu2, hc]
  have ht2_22 : |t2| ≤ 22 := by
    have h := abs_add_le (t2 - x0) x0
    have hx : t2 - x0 + x0 = t2 := by ring
    rw [hx] at h
    linarith [h, hd44, hx0abs, hu0]
  have hd14 : |t2 - x0| ≤ 1 / 4 := by linarith [hd44, hu44]
  exact ⟨hx0abs42, hxf22, ht2_22, hd44, hd14⟩

set_option maxHeartbeats 1000000 in
/-- Shared main estimate: from the individual error bounds, `|l2 - L| ≤ 100 u L + u + c43`. -/
lemma sp_main_aux {x0 t2 e s l l2 u c14 c46 c138 c140 c149 c150 c43 : ℝ}
    (hu0 : 0 < u) (_hu1 : u ≤ 1 / 2) (hu24 : u = 2 ^ (-24 : ℤ))
    (hc : 2 * c150 ≤ u) (hc150nn : 0 ≤ c150)
    (hc14_0 : 0 ≤ c14) (hc14le : c14 ≤ 1) (hc14s : c14 ≤ 1 / 10000)
    (hc14 : c14 = 2 ^ (-14 : ℤ)) (hc138 : c138 = 2 ^ (-138 : ℤ))
    (hc140 : c140 = 2 ^ (-140 : ℤ)) (hc140le : c140 ≤ 1 / 2)
    (htiny : c46 + c138 + c150 + c149 + 5.0001 * u * (c46 + c138) + u * c149
      + 5.0001 * (u * u) ≤ c43)
    (hlf : (u + 2 * c150) * (1 + 2 * (u + 2 * c150)) ≤ u + c46)
    (he : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + c140)
    (hs : |s - (1 + e)| ≤ u * (1 + e) + c150)
    (hl : |l - Real.log s| ≤ 4 * u * |Real.log s| + c149)
    (hl2 : |l2 - l| ≤ u * |l| + c150)
    (ht2_22 : |t2| ≤ 22) (hd44 : |t2 - x0| ≤ 44 * u) (hd14 : |t2 - x0| ≤ 1 / 4) :
    |l2 - Real.log (1 + Real.exp x0)| ≤ 100 * u * Real.log (1 + Real.exp x0) + u + c43 := by
  set D : ℝ := |t2 - x0| with hDdef
  set L : ℝ := Real.log (1 + Real.exp x0) with hLdef
  set s0 : ℝ := Real.exp x0 / (1 + Real.exp x0) with hs0def
  set sT : ℝ := Real.exp t2 / (1 + Real.exp t2) with hsTdef
  set S : ℝ := |Real.log s| with hSdef
  set labs : ℝ := |l| with hlabsdef
  set A1 : ℝ := |l2 - l| with hA1def
  set A2 : ℝ := |l - Real.log s| with hA2def
  set A3 : ℝ := |Real.log s - Real.log (1 + e)| with hA3def
  set A4 : ℝ := |Real.log (1 + e) - Real.log (1 + Real.exp t2)| with hA4def
  set A5 : ℝ := |Real.log (1 + Real.exp t2) - L| with hA5def
  have hL0 : 0 ≤ L := by
    rw [hLdef]
    exact Real.log_nonneg (by linarith [Real.exp_pos x0])
  have hs0 : 0 ≤ s0 := by
    rw [hs0def]
    exact div_nonneg (le_of_lt (Real.exp_pos x0)) (by positivity)
  have hs0L : s0 ≤ L := by
    rw [hs0def, hLdef]
    exact log_one_add_exp_ge_ratio x0
  have hS0 : 0 ≤ S := by rw [hSdef]; exact abs_nonneg _
  have hD0 : 0 ≤ D := by rw [hDdef]; exact abs_nonneg _
  have hD44 : D ≤ 44 * u := by rw [hDdef]; exact hd44
  have hA1 : A1 ≤ u * labs + c150 := by rw [hA1def]; exact hl2
  have hA2 : A2 ≤ 4 * u * S + c149 := by rw [hA2def]; exact hl
  have hu_small : u ≤ 1 / 4 := by rw [hu24]; norm_num [zpow_neg]
  have h1e : 1 / 2 ≤ 1 + e := by
    have h48 : 48 * u ≤ 1 / 2 := by rw [hu24]; norm_num [zpow_neg]
    exact sp_one_add_e hu0 h48 hc140le he ht2_22
  have hwu : u + 2 * c150 ≤ 1 / 2 := by linarith only [hc, hu_small]
  have hA3 : A3 ≤ u + c46 := by
    rw [hA3def]
    exact sp_A3 hu0 hc150nn hwu h1e hs hlf
  have hA4 : A4 ≤ 48 * u * sT * (1 + c14) + c138 := by
    rw [hA4def]
    have he' : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + (2 : ℝ) ^ (-140 : ℤ) := by
      rw [← hc140]; exact he
    have hT := T1_bound hu24 he' ht2_22
    rw [← hc14, ← hc138, ← hsTdef] at hT
    have hcoef4 : 4 * u + 2 * u * |t2| ≤ 48 * u := by
      have h : 2 * u * |t2| ≤ 2 * u * 22 := mul_le_mul_of_nonneg_left ht2_22 (by linarith)
      linarith
    have hsT0 : 0 ≤ sT := by
      rw [hsTdef]; exact div_nonneg (le_of_lt (Real.exp_pos t2)) (by positivity)
    have hfac0 : 0 ≤ 1 + c14 := by linarith
    have hmul1 : (4 * u + 2 * u * |t2|) * sT ≤ 48 * u * sT :=
      mul_le_mul_of_nonneg_right hcoef4 hsT0
    have hstep : (4 * u + 2 * u * |t2|) * sT * (1 + c14) ≤ 48 * u * sT * (1 + c14) :=
      mul_le_mul_of_nonneg_right hmul1 hfac0
    linarith [hT, hstep]
  have hA5 : A5 ≤ D * s0 * (1 + 8 * D) := by
    rw [hA5def]
    have h5 := abs_log_one_add_exp_sub_le (a := x0) (b := t2) hd14
    rw [← hLdef, ← hDdef, ← hs0def] at h5
    exact h5
  have hST : sT ≤ s0 * (1 + 2 * D) := by
    have h := sp_hST (t2 := t2) (x0 := x0) hd14
    rw [← hsTdef, ← hs0def, ← hDdef] at h
    exact h
  have hS : S ≤ L + A3 + A4 + A5 := by
    rw [hSdef, hA3def, hA4def, hA5def]
    have h3 := abs_sub_le (Real.log s) (Real.log (1 + e)) L
    have h4 := abs_sub_le (Real.log (1 + e)) (Real.log (1 + Real.exp t2)) L
    have h1 := abs_sub_le (Real.log s) L 0
    simp only [sub_zero] at h1
    rw [abs_of_nonneg hL0] at h1
    linarith [h1, h3, h4]
  have hlabs : labs ≤ S + A2 := by
    rw [hlabsdef, hSdef, hA2def]
    have h1 := abs_sub_le l (Real.log s) 0
    simp only [sub_zero] at h1
    linarith [h1]
  have htri : |l2 - L| ≤ A1 + A2 + A3 + A4 + A5 := by
    rw [hA1def, hA2def, hA3def, hA4def, hA5def]
    have h1 := abs_sub_le l2 l L
    have h2 := abs_sub_le l (Real.log s) L
    have h3 := abs_sub_le (Real.log s) (Real.log (1 + e)) L
    have h4 := abs_sub_le (Real.log (1 + e)) (Real.log (1 + Real.exp t2)) L
    linarith [h1, h2, h3, h4]
  have hu1e : u ≤ 1 / 1000000 := by rw [hu24]; norm_num [zpow_neg]
  have hcore := sp_core hu0 hu1e hL0 hs0 hs0L hS0 hD0 hD44 hc14_0 hc14le hc14s
    htiny hA1 hA2 hA3 hA4 hA5 hST hS hlabs
  linarith [htri, hcore]

/-- **The softplus branch** (`t1 = fl(1*xf) <= 20`). -/
lemma exp_neg20_le_28 : Real.exp (-20) ≤ (2 : ℝ) ^ (-28 : ℤ) := by
  have h2 : (2.7 : ℝ) ≤ Real.exp 1 := by
    have h := Real.exp_one_gt_d9
    linarith
  have hp28 : (0:ℝ) < (2:ℝ)^28 := by norm_num
  have hpow : (2:ℝ)^28 ≤ (Real.exp 1)^20 := by
    have h1 : (2:ℝ)^28 ≤ (2.7:ℝ)^20 := by norm_num
    have h2' : (2.7:ℝ)^20 ≤ (Real.exp 1)^20 := pow_le_pow_left₀ (by norm_num) h2 20
    linarith
  have he : (Real.exp 1)^20 = Real.exp 20 := by
    rw [← Real.exp_nat_mul]; norm_num
  have hpow' : (2:ℝ)^28 ≤ Real.exp 20 := by rw [← he]; exact hpow
  have hpos : (0:ℝ) < Real.exp 20 := Real.exp_pos 20
  have hz : (2:ℝ)^(-28:ℤ) = ((2:ℝ)^28)⁻¹ := by rw [zpow_neg]; rfl
  rw [Real.exp_neg, hz]
  exact (inv_le_inv₀ hpos hp28).mpr hpow'

/-- For `y ≥ 20`, `y · exp(-y) ≤ 20 · 2^-28`. -/
lemma mul_exp_neg_le_of_ge20 {y : ℝ} (h : 20 ≤ y) :
    y * Real.exp (-y) ≤ 20 * 2 ^ (-28 : ℤ) := by
  have hexp : y / 20 ≤ Real.exp (y - 20) := by
    have h1 : y - 20 + 1 ≤ Real.exp (y - 20) := Real.add_one_le_exp _
    nlinarith
  have h2 : (0:ℝ) < Real.exp (-(y-20)) := Real.exp_pos _
  have h3 : (y / 20) * Real.exp (-(y-20)) ≤ 1 := by
    have hstep := mul_le_mul_of_nonneg_right hexp (le_of_lt h2)
    have he1 : Real.exp (y-20) * Real.exp (-(y-20)) = 1 := by
      rw [← Real.exp_add]
      ring_nf
      exact Real.exp_zero
    rw [he1] at hstep
    exact hstep
  have heq : y * Real.exp (-y) = 20 * (Real.exp (-20) * ((y/20) * Real.exp (-(y-20)))) := by
    rw [show -y = -20 + (-(y-20)) by ring, Real.exp_add]
    field_simp
  have h7 : Real.exp (-20) ≤ (2:ℝ)^(-28:ℤ) := exp_neg20_le_28
  have h8 : (0:ℝ) ≤ 2^(-28:ℤ) := by norm_num
  have hpos20 : (0:ℝ) < Real.exp (-20) := Real.exp_pos _
  nlinarith [h3, h7, h8, hpos20]

/-- For `a, b ≥ 1`, `|log a - log b| ≤ |a - b|`. -/
lemma abs_log_sub_log_le {a b : ℝ} (ha : 1 ≤ a) (hb : 1 ≤ b) :
    |Real.log a - Real.log b| ≤ |a - b| := by
  have key : ∀ x y : ℝ, 1 ≤ y → y ≤ x → Real.log x - Real.log y ≤ x - y := by
    intro x y hy hxy
    have hy0 : 0 < y := by linarith
    have hd : 0 ≤ (x - y) / y := div_nonneg (by linarith) (le_of_lt hy0)
    have hkey : Real.log x - Real.log y = Real.log (1 + (x - y) / y) := by
      have hx : x = y * (1 + (x - y) / y) := by field_simp; ring
      conv_lhs => rw [hx]
      rw [Real.log_mul (ne_of_gt hy0) (by positivity)]
      ring
    have h1 : Real.log (1 + (x - y) / y) ≤ (x - y) / y := log_one_add_le_self hd
    have h2 : (x - y) / y ≤ x - y := by
      rw [div_le_iff₀ hy0]
      nlinarith
    rw [hkey]
    linarith
  rcases le_or_gt b a with h | h
  · have h4 := key a b hb h
    have h5 : 0 ≤ Real.log a - Real.log b := by
      have h6 := Real.log_le_log (by linarith : (0:ℝ) < b) h
      linarith
    have h6 : 0 ≤ a - b := by linarith
    rw [abs_of_nonneg h5, abs_of_nonneg h6]
    linarith
  · have h5 : Real.log a - Real.log b ≤ 0 := by
      have h6 := Real.log_le_log (by linarith : (0:ℝ) < a) (le_of_lt h)
      linarith
    have h6 : a - b ≤ 0 := by linarith
    rw [abs_of_nonpos h5, abs_of_nonpos h6]
    have h4 := key b a ha (le_of_lt h)
    linarith

/-- The log-softplus difference is bounded by the difference of the exponentials. -/
lemma abs_log_one_add_exp_sub_le' (a b : ℝ) :
    |Real.log (1 + Real.exp b) - Real.log (1 + Real.exp a)| ≤ |Real.exp b - Real.exp a| := by
  have ha : 1 ≤ 1 + Real.exp a := by linarith [Real.exp_pos a]
  have hb : 1 ≤ 1 + Real.exp b := by linarith [Real.exp_pos b]
  have h := abs_log_sub_log_le (a := 1 + Real.exp b) (b := 1 + Real.exp a) hb ha
  have heq : (1 + Real.exp b) - (1 + Real.exp a) = Real.exp b - Real.exp a := by ring
  rwa [heq] at h

/-- `exp t ≤ 2^(-46)` for `t ≤ -46`. -/
lemma exp_le_two_neg46 {t : ℝ} (h : t ≤ -46) : Real.exp t ≤ (2 : ℝ) ^ (-46 : ℤ) := by
  have h2 : (2 : ℝ) ≤ Real.exp 1 := le_of_lt Real.exp_one_gt_two
  have hp : (0 : ℝ) < (2 : ℝ) ^ 46 := by norm_num
  have hpow : (2 : ℝ) ^ 46 ≤ (Real.exp 1) ^ 46 := pow_le_pow_left₀ (by norm_num) h2 46
  have he : (Real.exp 1) ^ 46 = Real.exp 46 := by
    rw [← Real.exp_nat_mul]
    norm_num
  have hpos : (0 : ℝ) < Real.exp 46 := Real.exp_pos 46
  have h46 : Real.exp (-46) ≤ (2 : ℝ) ^ (-46 : ℤ) := by
    have hz : (2 : ℝ) ^ (-46 : ℤ) = ((2 : ℝ) ^ 46)⁻¹ := by rw [zpow_neg]; rfl
    rw [Real.exp_neg, hz]
    exact (inv_le_inv₀ hpos hp).mpr (by rw [← he]; exact hpow)
  exact le_trans (Real.exp_le_exp.mpr h) h46

/-- `1 + e` stays above `1/2` when `t2 ≤ 0`. -/
lemma sp_one_add_e_neg {e t2 u c140 : ℝ} (hu0 : 0 < u) (hu6 : 6 * u ≤ 1 / 4)
    (hc140 : c140 ≤ 1 / 4) (ht2 : t2 ≤ 0)
    (he : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + c140) :
    1 / 2 ≤ 1 + e := by
  have hexp0 : (0 : ℝ) < Real.exp t2 := Real.exp_pos t2
  have hexp1 : Real.exp t2 ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr ht2
  have hmul := abs_mul_exp_le_exp_half ht2
  have hhalf : Real.exp (t2 / 2) ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr (by linarith)
  have hbound : (4 * u + 2 * u * |t2|) * Real.exp t2 + c140 ≤ 1 / 2 := by
    have h1 : (4 * u + 2 * u * |t2|) * Real.exp t2
        = 4 * u * Real.exp t2 + 2 * u * (|t2| * Real.exp t2) := by ring
    have h2 : 4 * u * Real.exp t2 ≤ 4 * u := by nlinarith [hexp1, hu0]
    have h3 : 2 * u * (|t2| * Real.exp t2) ≤ 2 * u := by nlinarith [hmul, hhalf, hu0]
    linarith
  have h1 := (abs_le.mp he).1
  linarith

/-- If `|e - E| ≤ (1+E)/4` then the log-increment is at most `(3/2)|e - E|`. -/
lemma abs_log_one_add_sub_le_ratio {e E : ℝ} (hE : 0 < E) (hz : |e - E| ≤ (1 + E) / 4) :
    |Real.log (1 + e) - Real.log (1 + E)| ≤ (3 / 2) * |e - E| := by
  have hden : (0 : ℝ) < 1 + E := by linarith
  set z : ℝ := (e - E) / (1 + E) with hzdef
  have hz1 : 1 + e = (1 + E) * (1 + z) := by
    rw [hzdef]; field_simp; ring
  have hzabs : |z| ≤ 1 / 4 := by
    rw [hzdef, abs_div, abs_of_pos hden, div_le_iff₀ hden]
    linarith
  have hz0 : 1 + z ≠ 0 := by
    have h := (abs_le.mp hzabs).1
    intro hcontra
    linarith
  have hlog : Real.log (1 + e) - Real.log (1 + E) = Real.log (1 + z) := by
    rw [hz1, Real.log_mul (ne_of_gt hden) hz0]
    ring
  rw [hlog]
  have hb := abs_log_one_add_le (le_trans hzabs (by norm_num : (1:ℝ)/4 ≤ 1/2))
  have hze : |z| ≤ |e - E| := by
    have h0 : (0:ℝ) ≤ |e - E| := abs_nonneg _
    rw [hzdef, abs_div, abs_of_pos hden, div_le_iff₀ hden]
    nlinarith [h0, hden]
  have hfac : 1 + 2 * |z| ≤ 3 / 2 := by
    have h := abs_nonneg z
    linarith
  calc |Real.log (1 + z)| ≤ |z| * (1 + 2 * |z|) := hb
    _ ≤ |e - E| * (3 / 2) := by
        apply mul_le_mul hze hfac (by positivity) (abs_nonneg _)
    _ = (3 / 2) * |e - E| := by ring

/-- For `-2^20 ≤ x0 < -20`, the `log (1 + ·)` of the computed exponential differs from that of `exp t2` by at most `2^-45`. -/
lemma sp_A4_b1 {x0 t2 e u c140 c150 : ℝ}
    (hu24 : u = 2 ^ (-24 : ℤ)) (hc140 : c140 = 2 ^ (-140 : ℤ))
    (hc150 : c150 = 2 ^ (-150 : ℤ))
    (he : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + c140)
    (hx20 : x0 < -20) (hx0lo : -(2 : ℝ) ^ 20 ≤ x0)
    (hD : |t2 - x0| ≤ 3 * u * |x0| + 3 * c150) :
    |Real.log (1 + e) - Real.log (1 + Real.exp t2)| ≤ (2 : ℝ) ^ (-45 : ℤ) := by
  have hu0 : 0 < u := by rw [hu24]; norm_num
  have hy : |x0| = -x0 := abs_of_nonpos (by linarith)
  have hy20 : 20 ≤ |x0| := by rw [hy]; linarith
  have hyB : |x0| ≤ (2 : ℝ) ^ 20 := by rw [hy]; linarith
  have hDq : |t2 - x0| ≤ 1 / 4 := by
    have h1 : 3 * u * |x0| ≤ 3 * u * (2 : ℝ) ^ 20 :=
      mul_le_mul_of_nonneg_left hyB (by positivity)
    have h3 : (3 : ℝ) * u * (2 : ℝ) ^ 20 = 3 / 16 := by rw [hu24]; norm_num [zpow_neg]
    have h4 : 3 * c150 ≤ 1 / 16 := by rw [hc150]; norm_num [zpow_neg]
    linarith
  have hD124 : |t2 - x0| ≤ 1 / 2 := by linarith
  have ht2le : |t2| ≤ 51 / 50 * |x0| := by
    have h := abs_add_le (t2 - x0) x0
    have hx : t2 - x0 + x0 = t2 := by ring
    rw [hx] at h
    nlinarith [h, hDq, hy20, abs_nonneg x0]
  have hx0e : x0 = -|x0| := by rw [hy]; ring
  have hexpd : Real.exp (t2 - x0) ≤ 1 + 2 * |t2 - x0| := by
    have h := (abs_le.mp (abs_exp_sub_one_le hD124)).2
    linarith
  have hEbound : Real.exp t2 ≤ 3 / 2 * Real.exp (-|x0|) := by
    have hfac : Real.exp t2 = Real.exp (-|x0|) * Real.exp (t2 - x0) := by
      rw [← Real.exp_add]
      have : -|x0| + (t2 - x0) = t2 := by rw [hy]; ring
      rw [this]
    rw [hfac]
    have h2 : 1 + 2 * |t2 - x0| ≤ 1.5 := by linarith
    have hp : (0 : ℝ) < Real.exp (-|x0|) := Real.exp_pos _
    have hq : (0 : ℝ) < Real.exp (t2 - x0) := Real.exp_pos _
    nlinarith [hexpd, h2, hp, hq]
  have hE2 : Real.exp (-|x0|) ≤ (2 : ℝ) ^ (-28 : ℤ) := by
    have h1 : Real.exp (-|x0|) ≤ Real.exp (-20) := Real.exp_le_exp.mpr (by linarith)
    exact le_trans h1 exp_neg20_le_28
  have hE1'' : Real.exp t2 ≤ 3 / 2 * (2 : ℝ) ^ (-28 : ℤ) := by
    have h1 := mul_le_mul_of_nonneg_left hE2 (by norm_num : (0 : ℝ) ≤ 1.5)
    linarith
  have hyexp : |x0| * Real.exp (-|x0|) ≤ 20 * (2 : ℝ) ^ (-28 : ℤ) := mul_exp_neg_le_of_ge20 hy20
  have hpr : |t2| * Real.exp t2 ≤ 153 / 5 * (2 : ℝ) ^ (-28 : ℤ) := by
    have h1 : |t2| * Real.exp t2 ≤ (51 / 50 * |x0|) * (3 / 2 * Real.exp (-|x0|)) :=
      mul_le_mul ht2le hEbound (le_of_lt (Real.exp_pos t2)) (by positivity)
    have h2 : (51 / 50 * |x0|) * (3 / 2 * Real.exp (-|x0|))
        = 153 / 100 * (|x0| * Real.exp (-|x0|)) := by ring
    have h2' : 153 / 100 * (|x0| * Real.exp (-|x0|))
        ≤ 153 / 100 * (20 * (2 : ℝ) ^ (-28 : ℤ)) :=
      mul_le_mul_of_nonneg_left hyexp (by norm_num)
    have h3 : 153 / 100 * (20 * (2 : ℝ) ^ (-28 : ℤ)) = 153 / 5 * (2 : ℝ) ^ (-28 : ℤ) := by
      norm_num
    calc |t2| * Real.exp t2 ≤ (51 / 50 * |x0|) * (3 / 2 * Real.exp (-|x0|)) := h1
      _ = 153 / 100 * (|x0| * Real.exp (-|x0|)) := h2
      _ ≤ 153 / 100 * (20 * (2 : ℝ) ^ (-28 : ℤ)) := h2'
      _ = 153 / 5 * (2 : ℝ) ^ (-28 : ℤ) := h3
  have huE : u * Real.exp t2 ≤ 3 / 2 * (2 : ℝ) ^ (-52 : ℤ) := by
    have h1 := mul_le_mul_of_nonneg_left hE1'' (le_of_lt hu0)
    have h2 : u * (3 / 2 * (2 : ℝ) ^ (-28 : ℤ)) = 3 / 2 * (2 : ℝ) ^ (-52 : ℤ) := by
      rw [hu24]; norm_num [zpow_neg]
    linarith
  have huprod : u * (|t2| * Real.exp t2) ≤ 153 / 5 * (2 : ℝ) ^ (-52 : ℤ) := by
    have h1 := mul_le_mul_of_nonneg_left hpr (le_of_lt hu0)
    have h2 : u * (153 / 5 * (2 : ℝ) ^ (-28 : ℤ)) = 153 / 5 * (2 : ℝ) ^ (-52 : ℤ) := by
      rw [hu24]; norm_num [zpow_neg]
    linarith
  have huE : u * Real.exp t2 ≤ 3 / 2 * (2 : ℝ) ^ (-52 : ℤ) := by
    have h1 := mul_le_mul_of_nonneg_left hE1'' (le_of_lt hu0)
    have h2 : u * (3 / 2 * (2 : ℝ) ^ (-28 : ℤ)) = 3 / 2 * (2 : ℝ) ^ (-52 : ℤ) := by
      rw [hu24]; norm_num [zpow_neg]
    rw [h2] at h1
    exact h1
  have huprod : u * (|t2| * Real.exp t2) ≤ 153 / 5 * (2 : ℝ) ^ (-52 : ℤ) := by
    have h1 := mul_le_mul_of_nonneg_left hpr (le_of_lt hu0)
    have h2 : u * (153 / 5 * (2 : ℝ) ^ (-28 : ℤ)) = 153 / 5 * (2 : ℝ) ^ (-52 : ℤ) := by
      rw [hu24]; norm_num [zpow_neg]
    rw [h2] at h1
    exact h1
  have hexpand : (4 * u + 2 * u * |t2|) * Real.exp t2
      = 4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) := by ring
  rw [hexpand] at he
  have hbase : 4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140
      ≤ 4 * (3 / 2 * (2 : ℝ) ^ (-52 : ℤ)) + 2 * (153 / 5 * (2 : ℝ) ^ (-52 : ℤ))
        + (2 : ℝ) ^ (-140 : ℤ) := by
    have h1 : 4 * (u * Real.exp t2) ≤ 4 * (3 / 2 * (2 : ℝ) ^ (-52 : ℤ)) :=
      mul_le_mul_of_nonneg_left huE (by norm_num)
    have h2 : 2 * (u * (|t2| * Real.exp t2)) ≤ 2 * (153 / 5 * (2 : ℝ) ^ (-52 : ℤ)) :=
      mul_le_mul_of_nonneg_left huprod (by norm_num)
    have h3 : c140 ≤ (2 : ℝ) ^ (-140 : ℤ) := le_of_eq hc140
    exact add_le_add (add_le_add h1 h2) h3
  have hR : 4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140 ≤ 1 / 4 := by
    refine le_trans hbase ?_
    norm_num [zpow_neg]
  have hfin : (3 / 2) * (4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140)
      ≤ (2 : ℝ) ^ (-45 : ℤ) := by
    have hs : (3 / 2) * (4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140)
        ≤ (3 / 2) * (4 * (3 / 2 * (2 : ℝ) ^ (-52 : ℤ))
          + 2 * (153 / 5 * (2 : ℝ) ^ (-52 : ℤ)) + (2 : ℝ) ^ (-140 : ℤ)) :=
      mul_le_mul_of_nonneg_left hbase (by norm_num)
    have hnum : (3 / 2) * (4 * (3 / 2 * (2 : ℝ) ^ (-52 : ℤ))
        + 2 * (153 / 5 * (2 : ℝ) ^ (-52 : ℤ)) + (2 : ℝ) ^ (-140 : ℤ))
        ≤ (2 : ℝ) ^ (-45 : ℤ) := by norm_num [zpow_neg]
    exact le_trans hs hnum
  have h1 : |e - Real.exp t2| ≤ 1 / 4 := le_trans he hR
  have he_small : |e - Real.exp t2| ≤ (1 + Real.exp t2) / 4 := by
    have hE0 : (0 : ℝ) ≤ Real.exp t2 := le_of_lt (Real.exp_pos t2)
    linarith only [h1, hE0]
  have hA4 := abs_log_one_add_sub_le_ratio (Real.exp_pos t2) he_small
  have hmul : (3 / 2) * |e - Real.exp t2|
      ≤ (3 / 2) * (4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140) :=
    mul_le_mul_of_nonneg_left he (by norm_num)
  exact le_trans hA4 (le_trans hmul hfin)

/-- Case `-2^20 ≤ x0 < -20`: moving from `t2` back to `x0` costs at most `2^-45`. -/
lemma sp_A5_b1 {x0 t2 u c150 : ℝ}
    (hu24 : u = 2 ^ (-24 : ℤ)) (hc150 : c150 = 2 ^ (-150 : ℤ))
    (hx20 : x0 < -20) (hx0lo : -(2 : ℝ) ^ 20 ≤ x0)
    (hD : |t2 - x0| ≤ 3 * u * |x0| + 3 * c150) :
    |Real.log (1 + Real.exp t2) - Real.log (1 + Real.exp x0)| ≤ (2 : ℝ) ^ (-45 : ℤ) := by
  have hlog := abs_log_one_add_exp_sub_le' (a := x0) (b := t2)
  have hu0 : 0 < u := by rw [hu24]; norm_num
  have hc150pos : (0 : ℝ) < c150 := by rw [hc150]; positivity
  have hy : |x0| = -x0 := abs_of_nonpos (by linarith)
  have hy20 : 20 ≤ |x0| := by rw [hy]; linarith
  have hyB : |x0| ≤ (2 : ℝ) ^ 20 := by rw [hy]; linarith
  have hx0e : x0 = -|x0| := by rw [hy]; ring
  have hDq : |t2 - x0| ≤ 1 / 4 := by
    have h1 : 3 * u * |x0| ≤ 3 * u * (2 : ℝ) ^ 20 :=
      mul_le_mul_of_nonneg_left hyB (by positivity)
    have h3 : (3 : ℝ) * u * (2 : ℝ) ^ 20 = 3 / 16 := by rw [hu24]; norm_num [zpow_neg]
    have h4 : 3 * c150 ≤ 1 / 16 := by rw [hc150]; norm_num [zpow_neg]
    linarith
  have hD124 : |t2 - x0| ≤ 1 / 2 := by linarith
  have hE0pos : (0 : ℝ) < Real.exp x0 := Real.exp_pos _
  have hE0le : Real.exp x0 ≤ (2 : ℝ) ^ (-28 : ℤ) := by
    have h1 : Real.exp x0 = Real.exp (-|x0|) := congrArg Real.exp hx0e
    rw [h1]
    have h2 : Real.exp (-|x0|) ≤ Real.exp (-20) := Real.exp_le_exp.mpr (by linarith)
    exact le_trans h2 exp_neg20_le_28
  have hfac : Real.exp t2 = Real.exp x0 * Real.exp (t2 - x0) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have habs : |Real.exp t2 - Real.exp x0| = Real.exp x0 * |Real.exp (t2 - x0) - 1| := by
    rw [hfac]
    have h : Real.exp x0 * Real.exp (t2 - x0) - Real.exp x0
        = Real.exp x0 * (Real.exp (t2 - x0) - 1) := by ring
    rw [h, abs_mul, abs_of_pos hE0pos]
  have hd1 : |Real.exp (t2 - x0) - 1| ≤ 2 * |t2 - x0| := abs_exp_sub_one_le hD124
  have hstep1 : Real.exp x0 * |Real.exp (t2 - x0) - 1| ≤ Real.exp x0 * (2 * |t2 - x0|) :=
    mul_le_mul_of_nonneg_left hd1 (le_of_lt hE0pos)
  have hstep2 : Real.exp x0 * (2 * |t2 - x0|)
      ≤ Real.exp x0 * (2 * (3 * u * |x0| + 3 * c150)) := by
    apply mul_le_mul_of_nonneg_left _ (le_of_lt hE0pos)
    linarith [hD]
  have hexp : Real.exp x0 * (2 * (3 * u * |x0| + 3 * c150))
      = 6 * u * (|x0| * Real.exp x0) + 6 * c150 * Real.exp x0 := by ring
  have hprod : |x0| * Real.exp x0 ≤ 20 * (2 : ℝ) ^ (-28 : ℤ) := by
    have h1 : Real.exp x0 = Real.exp (-|x0|) := congrArg Real.exp hx0e
    rw [h1]
    exact mul_exp_neg_le_of_ge20 hy20
  have hterm1 : 6 * u * (|x0| * Real.exp x0) ≤ 6 * u * (20 * (2 : ℝ) ^ (-28 : ℤ)) :=
    mul_le_mul_of_nonneg_left hprod (by linarith)
  have hterm1' : 6 * u * (20 * (2 : ℝ) ^ (-28 : ℤ)) = 120 * (2 : ℝ) ^ (-52 : ℤ) := by
    rw [hu24]; norm_num [zpow_neg]
  have hterm2 : 6 * c150 * Real.exp x0 ≤ 6 * c150 * (2 : ℝ) ^ (-28 : ℤ) :=
    mul_le_mul_of_nonneg_left hE0le (by linarith)
  have hterm2' : 6 * c150 * (2 : ℝ) ^ (-28 : ℤ)
      = 6 * (2 : ℝ) ^ (-150 : ℤ) * (2 : ℝ) ^ (-28 : ℤ) := by rw [hc150]
  have hnum : 120 * (2 : ℝ) ^ (-52 : ℤ)
      + 6 * (2 : ℝ) ^ (-150 : ℤ) * (2 : ℝ) ^ (-28 : ℤ) ≤ (2 : ℝ) ^ (-45 : ℤ) := by
    norm_num [zpow_neg]
  calc |Real.log (1 + Real.exp t2) - Real.log (1 + Real.exp x0)|
      ≤ |Real.exp t2 - Real.exp x0| := hlog
    _ = Real.exp x0 * |Real.exp (t2 - x0) - 1| := habs
    _ ≤ Real.exp x0 * (2 * |t2 - x0|) := hstep1
    _ ≤ Real.exp x0 * (2 * (3 * u * |x0| + 3 * c150)) := hstep2
    _ = 6 * u * (|x0| * Real.exp x0) + 6 * c150 * Real.exp x0 := hexp
    _ ≤ 6 * u * (20 * (2 : ℝ) ^ (-28 : ℤ)) + 6 * (2 : ℝ) ^ (-150 : ℤ) * (2 : ℝ) ^ (-28 : ℤ) := by
        have h2 := hterm2
        rw [hterm2'] at h2
        exact add_le_add hterm1 h2
    _ = 120 * (2 : ℝ) ^ (-52 : ℤ)
          + 6 * (2 : ℝ) ^ (-150 : ℤ) * (2 : ℝ) ^ (-28 : ℤ) := by rw [hterm1']
    _ ≤ (2 : ℝ) ^ (-45 : ℤ) := hnum

/-- Case `x0 < -2^20`: `t2` is far to the left. -/

lemma t2_le_neg46 {x0 t2 u c150 : ℝ}
    (hu24 : u = 2 ^ (-24 : ℤ)) (hc150 : c150 = 2 ^ (-150 : ℤ))
    (hx0lo : x0 < -(2 : ℝ) ^ 20)
    (hD : |t2 - x0| ≤ 3 * u * |x0| + 3 * c150) : t2 ≤ -((2 : ℝ) ^ 18) := by
  have hy : |x0| = -x0 := abs_of_nonpos (by linarith)
  have hy20 : (2 : ℝ) ^ 20 ≤ |x0| := by rw [hy]; linarith
  have hstep : t2 ≤ x0 + (3 * u * |x0| + 3 * c150) := by
    have h := le_abs_self (t2 - x0)
    linarith only [h, hD]
  have h2u : 3 * u ≤ 1 / 2 := by rw [hu24]; norm_num [zpow_neg]
  have hle : 3 * u * |x0| ≤ (1 / 2) * |x0| :=
    mul_le_mul_of_nonneg_right h2u (abs_nonneg x0)
  have hc : 3 * c150 ≤ 2 ^ 18 := by rw [hc150]; norm_num [zpow_neg]
  linarith only [hstep, hle, hc, hy, hy20]

/-- Case `x0 < -2^20`: the `e`-log error is at most `2^-45`. -/
lemma sp_A4_b2 {x0 t2 e u c140 c150 : ℝ}
    (hu24 : u = 2 ^ (-24 : ℤ)) (hc140 : c140 = 2 ^ (-140 : ℤ))
    (hc150 : c150 = 2 ^ (-150 : ℤ))
    (he : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + c140)
    (hx0lo : x0 < -(2 : ℝ) ^ 20)
    (hD : |t2 - x0| ≤ 3 * u * |x0| + 3 * c150) :
    |Real.log (1 + e) - Real.log (1 + Real.exp t2)| ≤ (2 : ℝ) ^ (-45 : ℤ) := by
  have hu24pos : (0 : ℝ) < u := by rw [hu24]; positivity
  have ht2 : t2 ≤ -((2 : ℝ) ^ 18) := t2_le_neg46 hu24 hc150 hx0lo hD
  have h46 : (46 : ℝ) ≤ (2 : ℝ) ^ 18 := by norm_num
  have ht46 : t2 ≤ -46 := by linarith only [ht2, h46]
  have ht20 : t2 ≤ 0 := by linarith only [ht46]
  have hthalf : t2 / 2 ≤ -46 := by
    have h2t : t2 / 2 ≤ -((2 : ℝ) ^ 18) / 2 := by linarith only [ht2]
    have h2n : -((2 : ℝ) ^ 18) / 2 ≤ -46 := by norm_num
    exact le_trans h2t h2n
  set C : ℝ := (2 : ℝ) ^ (-46 : ℤ) with hC
  have hE : Real.exp t2 ≤ C := by rw [hC]; exact exp_le_two_neg46 ht46
  have hprod : |t2| * Real.exp t2 ≤ C := by
    rw [hC]
    exact le_trans (abs_mul_exp_le_exp_half ht20) (exp_le_two_neg46 hthalf)
  have huE : u * Real.exp t2 ≤ u * C := mul_le_mul_of_nonneg_left hE (le_of_lt hu24pos)
  have huprod : u * (|t2| * Real.exp t2) ≤ u * C :=
    mul_le_mul_of_nonneg_left hprod (le_of_lt hu24pos)
  have hexpand : (4 * u + 2 * u * |t2|) * Real.exp t2
      = 4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) := by ring
  rw [hexpand] at he
  have h6 : 6 * u * C = 6 * (2 : ℝ) ^ (-70 : ℤ) := by
    rw [hC, hu24, mul_assoc]
    congr 1
    rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    norm_num
  have hbase : 4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140
      ≤ 6 * (2 : ℝ) ^ (-70 : ℤ) + (2 : ℝ) ^ (-140 : ℤ) := by
    have h1 : 4 * (u * Real.exp t2) ≤ 4 * (u * C) :=
      mul_le_mul_of_nonneg_left huE (by norm_num)
    have h2 : 2 * (u * (|t2| * Real.exp t2)) ≤ 2 * (u * C) :=
      mul_le_mul_of_nonneg_left huprod (by norm_num)
    have h3 : c140 ≤ (2 : ℝ) ^ (-140 : ℤ) := le_of_eq hc140
    nlinarith only [h1, h2, h3, h6]
  have hfinal : 6 * (2 : ℝ) ^ (-70 : ℤ) + (2 : ℝ) ^ (-140 : ℤ) ≤ 1 / 4 := by
    norm_num [zpow_neg]
  have hR : 4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140 ≤ 1 / 4 :=
    le_trans hbase hfinal
  have he_small : |e - Real.exp t2| ≤ (1 + Real.exp t2) / 4 := by
    have h1 : |e - Real.exp t2| ≤ 1 / 4 := le_trans he hR
    have hE0 : (0 : ℝ) ≤ Real.exp t2 := le_of_lt (Real.exp_pos t2)
    linarith only [h1, hE0]
  have hA4 := abs_log_one_add_sub_le_ratio (Real.exp_pos t2) he_small
  have hnum : (3 / 2) * (6 * (2 : ℝ) ^ (-70 : ℤ) + (2 : ℝ) ^ (-140 : ℤ))
      ≤ (2 : ℝ) ^ (-45 : ℤ) := by norm_num [zpow_neg]
  have hm : (3 / 2) * |e - Real.exp t2|
      ≤ (3 / 2) * (4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140) :=
    mul_le_mul_of_nonneg_left he (by norm_num)
  have h2' : (3 / 2) * (4 * (u * Real.exp t2) + 2 * (u * (|t2| * Real.exp t2)) + c140)
      ≤ (3 / 2) * (6 * (2 : ℝ) ^ (-70 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) :=
    mul_le_mul_of_nonneg_left hbase (by norm_num)
  exact le_trans hA4 (le_trans hm (le_trans h2' hnum))

/-- Case `x0 < -2^20`: moving from `t2` back to `x0` costs at most `2^-45`. -/
lemma sp_A5_b2 {x0 t2 u c150 : ℝ}
    (hu24 : u = 2 ^ (-24 : ℤ)) (hc150 : c150 = 2 ^ (-150 : ℤ))
    (hx0lo : x0 < -(2 : ℝ) ^ 20)
    (hD : |t2 - x0| ≤ 3 * u * |x0| + 3 * c150) :
    |Real.log (1 + Real.exp t2) - Real.log (1 + Real.exp x0)| ≤ (2 : ℝ) ^ (-45 : ℤ) := by
  have hlog := abs_log_one_add_exp_sub_le' (a := x0) (b := t2)
  have ht2 : t2 ≤ -((2 : ℝ) ^ 18) := t2_le_neg46 hu24 hc150 hx0lo hD
  have h46 : (46 : ℝ) ≤ (2 : ℝ) ^ 18 := by norm_num
  have ht46 : t2 ≤ -46 := by linarith only [ht2, h46]
  have hx46 : x0 ≤ -46 := by
    have h20 : (46 : ℝ) ≤ (2 : ℝ) ^ 20 := by norm_num
    linarith only [hx0lo, h20]
  have hEt : Real.exp t2 ≤ (2 : ℝ) ^ (-46 : ℤ) := exp_le_two_neg46 ht46
  have hEx : Real.exp x0 ≤ (2 : ℝ) ^ (-46 : ℤ) := exp_le_two_neg46 hx46
  have hstep : |Real.exp t2 - Real.exp x0| ≤ (2 : ℝ) ^ (-45 : ℤ) := by
    have h2 : |Real.exp t2 - Real.exp x0| ≤ Real.exp t2 + Real.exp x0 := by
      rw [abs_le]
      constructor <;> linarith [Real.exp_pos t2, Real.exp_pos x0]
    have hnum : (2 : ℝ) ^ (-46 : ℤ) + (2 : ℝ) ^ (-46 : ℤ) = (2 : ℝ) ^ (-45 : ℤ) := by
      norm_num [zpow_neg]
    linarith only [h2, hEt, hEx, hnum]
  exact le_trans hlog hstep

/-- Assembly for the `x0 < -20` case: constants are covered by `c43`, the two log-differences
by `2^-45` each. -/
lemma sp_core_neg {u L A1 A2 A3 A4 A5 S labs c46 c149 c150 c43 X : ℝ}
    (hu0 : 0 < u) (hu1e : u ≤ 1 / 1000000)
    (hL0 : 0 ≤ L) (hS0 : 0 ≤ S) (_hc149nn : 0 ≤ c149) (_hc150nn : 0 ≤ c150)
    (hconst : c46 + X + X
      + (50001 / 10000) * u * (X + X)
      + (50001 / 10000) * u * (u + c46) + u * c149 + c150 + c149 ≤ c43)
    (hA1 : A1 ≤ u * labs + c150)
    (hA2 : A2 ≤ 4 * u * S + c149)
    (hA3 : A3 ≤ u + c46)
    (hA4 : A4 ≤ X)
    (hA5 : A5 ≤ X)
    (hS : S ≤ L + A3 + A4 + A5)
    (hlabs : labs ≤ S + A2) :
    A1 + A2 + A3 + A4 + A5 ≤ 100 * u * L + u + c43 := by
  have hlabs2 : u * labs ≤ u * (S + (4 * u * S + c149)) := by
    apply mul_le_mul_of_nonneg_left _ (le_of_lt hu0)
    linarith only [hlabs, hA2]
  have hA12 : A1 + A2 ≤ u * (S + (4 * u * S + c149)) + c150 + (4 * u * S + c149) := by
    linarith only [hA1, hA2, hlabs2]
  have hcoefS : 5 * u + 4 * u * u ≤ 50001 / 10000 * u := by nlinarith only [hu1e, hu0]
  have h1 : (5 * u + 4 * u * u) * S ≤ 50001 / 10000 * u * S :=
    mul_le_mul_of_nonneg_right hcoefS hS0
  have hSB : S ≤ L + (u + c46) + A4 + A5 := by linarith only [hS, hA3]
  have h2 : 50001 / 10000 * u * S ≤ 50001 / 10000 * u * (L + (u + c46) + A4 + A5) :=
    mul_le_mul_of_nonneg_left hSB (by nlinarith only [hu0])
  have h3 : (5 * u + 4 * u * u) * S = u * (S + 4 * u * S) + 4 * u * S := by ring
  have h4 : u * (S + (4 * u * S + c149)) + c150 + (4 * u * S + c149)
      = (5 * u + 4 * u * u) * S + u * c149 + c150 + c149 := by ring
  have hA12' : A1 + A2 ≤ 50001 / 10000 * u * (L + (u + c46) + A4 + A5)
      + u * c149 + c150 + c149 := by
    linarith only [hA12, h1, h2, h3, h4]
  have hr : 50001 / 10000 * u * (L + (u + c46) + A4 + A5)
      = 50001 / 10000 * u * L + 50001 / 10000 * u * u + 50001 / 10000 * u * c46
        + 50001 / 10000 * u * (A4 + A5) := by ring
  have hA12'' : A1 + A2 ≤ 50001 / 10000 * u * L + 50001 / 10000 * u * u
      + 50001 / 10000 * u * c46 + 50001 / 10000 * u * (A4 + A5) + u * c149 + c150 + c149 := by
    linarith only [hA12', hr]
  have h5 : 50001 / 10000 * u * (A4 + A5)
      ≤ 50001 / 10000 * u * (X + X) :=
    mul_le_mul_of_nonneg_left (by linarith only [hA4, hA5]) (by nlinarith only [hu0])
  have h6 : A4 + A5 ≤ X + X := by
    linarith only [hA4, hA5]
  have h7 : 50001 / 10000 * u * L ≤ 100 * u * L :=
    mul_le_mul_of_nonneg_right (by nlinarith only [hu0]) hL0
  linarith only [hA12'', hA3, h5, h6, h7, hconst]

/-- **Softplus when `xf ≤ 20`**: the value `l2` of the `log(1 + exp(·))` branch is within `52 · 2^-23 · spRef x0 + 2^-24 (1 + 2^-19)` of `spRef x0`. -/
theorem softplus_true {x0 xf t1 t2 e s l l2 : ℝ}
    (hxf : |xf - x0| ≤ 2 ^ (-24 : ℤ) * |x0| + 2 ^ (-150 : ℤ))
    (ht1 : |t1 - xf| ≤ 2 ^ (-24 : ℤ) * |xf| + 2 ^ (-150 : ℤ))
    (ht2 : |t2 - xf| ≤ 2 ^ (-24 : ℤ) * |xf| + 2 ^ (-150 : ℤ))
    (hle : t1 ≤ 20)
    (he : |e - Real.exp t2| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t2|) * Real.exp t2 + 2 ^ (-140 : ℤ))
    (hs : |s - (1 + e)| ≤ 2 ^ (-24 : ℤ) * (1 + e) + 2 ^ (-150 : ℤ))
    (hl : |l - Real.log s| ≤ 2 ^ (-22 : ℤ) * |Real.log s| + 2 ^ (-149 : ℤ))
    (hl2 : |l2 - l| ≤ 2 ^ (-24 : ℤ) * |l| + 2 ^ (-150 : ℤ)) :
    |l2 - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) := by
  set u : ℝ := (2 : ℝ) ^ (-24 : ℤ) with hu_def
  set c14 : ℝ := (2 : ℝ) ^ (-14 : ℤ) with hc14
  set c19 : ℝ := (2 : ℝ) ^ (-19 : ℤ) with hc19
  set c20 : ℝ := (2 : ℝ) ^ (-20 : ℤ) with hc20
  set c22 : ℝ := (2 : ℝ) ^ (-22 : ℤ) with hc22
  set c23 : ℝ := (2 : ℝ) ^ (-23 : ℤ) with hc23
  set c43 : ℝ := (2 : ℝ) ^ (-43 : ℤ) with hc43
  set c46 : ℝ := (2 : ℝ) ^ (-46 : ℤ) with hc46
  set c138 : ℝ := (2 : ℝ) ^ (-138 : ℤ) with hc138
  set c140 : ℝ := (2 : ℝ) ^ (-140 : ℤ) with hc140
  set c149 : ℝ := (2 : ℝ) ^ (-149 : ℤ) with hc149
  set c150 : ℝ := (2 : ℝ) ^ (-150 : ℤ) with hc150
  have hu0 : 0 < u := by rw [hu_def]; norm_num [zpow_neg]
  have hu1 : u ≤ 1 / 2 := by rw [hu_def]; norm_num [zpow_neg]
  have hu44 : 44 * u ≤ 1 / 4 := by rw [hu_def]; norm_num [zpow_neg]
  have h22 : c22 = 4 * u := by rw [hc22, hu_def]; norm_num [zpow_neg]
  have h23 : c23 = 2 * u := by rw [hc23, hu_def]; norm_num [zpow_neg]
  have h43 : u * c19 = c43 := by
    rw [hu_def, hc19, hc43, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    norm_num
  rw [h22, h23] at he
  rw [h23]
  have hc : 2 * c150 ≤ u := by rw [hc150, hu_def]; norm_num [zpow_neg]
  have hc150nn : 0 ≤ c150 := by rw [hc150]; norm_num [zpow_neg]
  have hc14_0 : 0 ≤ c14 := by rw [hc14]; norm_num [zpow_neg]
  have hc14le : c14 ≤ 1 := by rw [hc14]; norm_num [zpow_neg]
  have hc14s : c14 ≤ 1 / 10000 := by rw [hc14]; norm_num [zpow_neg]
  have hc140le : c140 ≤ 1 / 2 := by rw [hc140]; norm_num [zpow_neg]
  have htiny : c46 + c138 + c150 + c149 + 5.0001 * u * (c46 + c138) + u * c149
      + 5.0001 * (u * u) ≤ c43 := by
    rw [hu_def, hc46, hc138, hc150, hc149, hc43]; norm_num [zpow_neg]
  have hlf : (u + 2 * c150) * (1 + 2 * (u + 2 * c150)) ≤ u + c46 := by
    rw [hu_def, hc150, hc46]; norm_num [zpow_neg]
  have hmain : |l2 - Real.log (1 + Real.exp x0)| ≤
      100 * u * Real.log (1 + Real.exp x0) + u + c43 := by
    rcases le_or_gt (-20) x0 with hxm | hxm
    · -- main case: `-20 ≤ x0`
      have hcoef : 2 * u + u * u ≤ 1 / 2 := by
        have h2u : 2 * u ≤ 1 / 4 := by linarith [hu44]
        have huq : u * u ≤ 1 / 4 := by nlinarith [hu0, hu1]
        linarith
      have hcsmall : u * c150 + 2 * c150 ≤ 1 := by
        rw [hu_def, hc150]; norm_num [zpow_neg]
      have hnumf : 42 * u + c150 ≤ 1 := by rw [hu_def, hc150]; norm_num [zpow_neg]
      have hnum : 20 + 42 * u + 43 * u + 2 * c150 ≤ 21 := by
        rw [hu_def, hc150]; norm_num [zpow_neg]
      have hnum2 : 21 * u + c150 ≤ 1 := by rw [hu_def, hc150]; norm_num [zpow_neg]
      obtain ⟨hx0abs42, hxf22, ht2_22, hd44, hd14⟩ :=
        sp_caseA_bounds hu0 hu44 hxm hle hxf ht1 ht2 hc hcoef hcsmall hnumf hnum hnum2
      set D : ℝ := |t2 - x0| with hDdef
      set L : ℝ := Real.log (1 + Real.exp x0) with hLdef
      set s0 : ℝ := Real.exp x0 / (1 + Real.exp x0) with hs0def
      set sT : ℝ := Real.exp t2 / (1 + Real.exp t2) with hsTdef
      set S : ℝ := |Real.log s| with hSdef
      set labs : ℝ := |l| with hlabsdef
      set A1 : ℝ := |l2 - l| with hA1def
      set A2 : ℝ := |l - Real.log s| with hA2def
      set A3 : ℝ := |Real.log s - Real.log (1 + e)| with hA3def
      set A4 : ℝ := |Real.log (1 + e) - Real.log (1 + Real.exp t2)| with hA4def
      set A5 : ℝ := |Real.log (1 + Real.exp t2) - L| with hA5def
      have hL0 : 0 ≤ L := by
        rw [hLdef]
        exact Real.log_nonneg (by linarith [Real.exp_pos x0])
      have hs0 : 0 ≤ s0 := by
        rw [hs0def]
        exact div_nonneg (le_of_lt (Real.exp_pos x0)) (by positivity)
      have hs0L : s0 ≤ L := by
        rw [hs0def, hLdef]
        exact log_one_add_exp_ge_ratio x0
      have hS0 : 0 ≤ S := by rw [hSdef]; exact abs_nonneg _
      have hD0 : 0 ≤ D := by rw [hDdef]; exact abs_nonneg _
      have hD44 : D ≤ 44 * u := by rw [hDdef]; exact hd44
      have hc14_0 : 0 ≤ c14 := by rw [hc14]; norm_num [zpow_neg]
      have hc14le : c14 ≤ 1 := by rw [hc14]; norm_num [zpow_neg]
      have hc14s : c14 ≤ 1 / 10000 := by rw [hc14]; norm_num [zpow_neg]
      have htiny : c46 + c138 + c150 + c149 + 5.0001 * u * (c46 + c138) + u * c149
          + 5.0001 * (u * u) ≤ c43 := by
        rw [hu_def, hc46, hc138, hc150, hc149, hc43]; norm_num [zpow_neg]
      have hA1 : A1 ≤ u * labs + c150 := by rw [hA1def]; exact hl2
      have hA2 : A2 ≤ 4 * u * S + c149 := by
        rw [hA2def]
        rw [h22] at hl
        exact hl
      have h1e : 1 / 2 ≤ 1 + e := by
        have h48 : 48 * u ≤ 1 / 2 := by rw [hu_def]; norm_num [zpow_neg]
        have h140 : c140 ≤ 1 / 2 := by rw [hc140]; norm_num [zpow_neg]
        exact sp_one_add_e hu0 h48 h140 he ht2_22
      have hwu : u + 2 * c150 ≤ 1 / 2 := by linarith [hc, hu1]
      have hlf : (u + 2 * c150) * (1 + 2 * (u + 2 * c150)) ≤ u + c46 := by
        rw [hu_def, hc150, hc46]; norm_num [zpow_neg]
      have hA3 : A3 ≤ u + c46 := by
        rw [hA3def]
        exact sp_A3 hu0 hc150nn hwu h1e hs hlf
      have hA4 : A4 ≤ 48 * u * sT * (1 + c14) + c138 := by
        rw [hA4def]
        have he' : |e - Real.exp t2| ≤ (4 * u + 2 * u * |t2|) * Real.exp t2 + (2 : ℝ) ^ (-140 : ℤ) := by
          rw [← hc140]; exact he
        have hT := T1_bound hu_def he' ht2_22
        rw [← hc14, ← hc138, ← hsTdef] at hT
        have hcoef4 : 4 * u + 2 * u * |t2| ≤ 48 * u := by
          have h : 2 * u * |t2| ≤ 2 * u * 22 := mul_le_mul_of_nonneg_left ht2_22 (by linarith)
          linarith
        have hsT0 : 0 ≤ sT := by
          rw [hsTdef]; exact div_nonneg (le_of_lt (Real.exp_pos t2)) (by positivity)
        have hfac0 : 0 ≤ 1 + c14 := by linarith
        have hmul1 : (4 * u + 2 * u * |t2|) * sT ≤ 48 * u * sT :=
          mul_le_mul_of_nonneg_right hcoef4 hsT0
        have hstep : (4 * u + 2 * u * |t2|) * sT * (1 + c14) ≤ 48 * u * sT * (1 + c14) :=
          mul_le_mul_of_nonneg_right hmul1 hfac0
        linarith [hT, hstep]
      have hA5 : A5 ≤ D * s0 * (1 + 8 * D) := by
        rw [hA5def]
        have h5 := abs_log_one_add_exp_sub_le (a := x0) (b := t2) hd14
        rw [← hLdef, ← hDdef, ← hs0def] at h5
        exact h5
      have hST : sT ≤ s0 * (1 + 2 * D) := by
        have h := sp_hST (t2 := t2) (x0 := x0) hd14
        rw [← hsTdef, ← hs0def, ← hDdef] at h
        exact h
      have hS : S ≤ L + A3 + A4 + A5 := by
        rw [hSdef, hA3def, hA4def, hA5def]
        have h3 := abs_sub_le (Real.log s) (Real.log (1 + e)) L
        have h4 := abs_sub_le (Real.log (1 + e)) (Real.log (1 + Real.exp t2)) L
        have h1 := abs_sub_le (Real.log s) L 0
        simp only [sub_zero] at h1
        rw [abs_of_nonneg hL0] at h1
        linarith [h1, h3, h4]
      have hlabs : labs ≤ S + A2 := by
        rw [hlabsdef, hSdef, hA2def]
        have h1 := abs_sub_le l (Real.log s) 0
        simp only [sub_zero] at h1
        linarith [h1]
      have htri : |l2 - L| ≤ A1 + A2 + A3 + A4 + A5 := by
        rw [hA1def, hA2def, hA3def, hA4def, hA5def]
        have h1 := abs_sub_le l2 l L
        have h2 := abs_sub_le l (Real.log s) L
        have h3 := abs_sub_le (Real.log s) (Real.log (1 + e)) L
        have h4 := abs_sub_le (Real.log (1 + e)) (Real.log (1 + Real.exp t2)) L
        linarith [h1, h2, h3, h4]
      have hu1e : u ≤ 1 / 1000000 := by rw [hu_def]; norm_num [zpow_neg]
      have hcore := sp_core hu0 hu1e hL0 hs0 hs0L hS0 hD0 hD44 hc14_0 hc14le hc14s
        htiny hA1 hA2 hA3 hA4 hA5 hST hS hlabs
      linarith [htri, hcore]
    · -- case `x0 < -20`
      have hD3 : |t2 - x0| ≤ 3 * u * |x0| + 3 * c150 := by
        have hxf_abs : |xf| ≤ |x0| + u * |x0| + c150 := by
          have h := abs_add_le (xf - x0) x0
          have hx : xf - x0 + x0 = xf := by ring
          rw [hx] at h
          linarith only [h, hxf]
        have h1 := abs_sub_le t2 xf x0
        have hu1le : u ≤ 1 := le_trans hu1 (by norm_num)
        have huxf : u * |xf| ≤ u * |x0| + (u * u) * |x0| + u * c150 := by
          have h2 := mul_le_mul_of_nonneg_left hxf_abs (le_of_lt hu0)
          have he : u * (|x0| + u * |x0| + c150)
              = u * |x0| + (u * u) * |x0| + u * c150 := by ring
          linarith only [h2, he]
        have huu : u * u ≤ u := by
          have h1 : u * u ≤ (1 / 2) * u := mul_le_mul_of_nonneg_right hu1 (le_of_lt hu0)
          have h2 : (1 / 2) * u ≤ u := mul_le_of_le_one_left (le_of_lt hu0) (by norm_num)
          exact le_trans h1 h2
        have h2 : (u * u) * |x0| ≤ u * |x0| :=
          mul_le_mul_of_nonneg_right huu (abs_nonneg x0)
        have huc : u * c150 ≤ c150 := mul_le_of_le_one_left hc150nn hu1le
        linarith only [h1, ht2, hxf, huxf, h2, huc]
      have ht2neg : t2 ≤ 0 := by
        have hy : |x0| = -x0 := abs_of_nonpos (by linarith only [hxm])
        have hy20 : 20 ≤ |x0| := by rw [hy]; linarith only [hxm]
        have hstep : t2 ≤ x0 + (3 * u * |x0| + 3 * c150) := by
          have h := le_abs_self (t2 - x0)
          linarith only [h, hD3]
        have h3u : 3 * u ≤ 1 / 2 := by rw [hu_def]; norm_num [zpow_neg]
        have hle : 3 * u * |x0| ≤ (1 / 2) * |x0| :=
          mul_le_mul_of_nonneg_right h3u (abs_nonneg x0)
        have hc3 : 3 * c150 ≤ 10 := by rw [hc150]; norm_num [zpow_neg]
        have hc3' : 3 * c150 ≤ (1 / 2) * |x0| := le_trans hc3 (by linarith only [hy20])
        have hstep2 : x0 + (3 * u * |x0| + 3 * c150)
            ≤ x0 + ((1 / 2) * |x0| + (1 / 2) * |x0|) :=
          add_le_add le_rfl (add_le_add hle hc3')
        have hstep3 : x0 + ((1 / 2) * |x0| + (1 / 2) * |x0|) ≤ 0 := by
          have h : x0 + |x0| = 0 := by linarith only [hy]
          linarith only [h]
        exact le_trans hstep (le_trans hstep2 hstep3)
      set D : ℝ := |t2 - x0| with hDdef
      set L : ℝ := Real.log (1 + Real.exp x0) with hLdef
      set S : ℝ := |Real.log s| with hSdef
      set labs : ℝ := |l| with hlabsdef
      set A1 : ℝ := |l2 - l| with hA1def
      set A2 : ℝ := |l - Real.log s| with hA2def
      set A3 : ℝ := |Real.log s - Real.log (1 + e)| with hA3def
      set A4 : ℝ := |Real.log (1 + e) - Real.log (1 + Real.exp t2)| with hA4def
      set A5 : ℝ := |Real.log (1 + Real.exp t2) - L| with hA5def
      set X : ℝ := (2 : ℝ) ^ (-45 : ℤ) with hX
      have hL0 : 0 ≤ L := by
        rw [hLdef]
        exact Real.log_nonneg (by linarith [Real.exp_pos x0])
      have hS0 : 0 ≤ S := by rw [hSdef]; exact abs_nonneg _
      have hc149nn : 0 ≤ c149 := by rw [hc149]; norm_num [zpow_neg]
      have hA1 : A1 ≤ u * labs + c150 := by rw [hA1def]; exact hl2
      have hA2 : A2 ≤ 4 * u * S + c149 := by
        rw [hA2def]
        rw [h22] at hl
        exact hl
      have h1e : 1 / 2 ≤ 1 + e := by
        have h6u : 6 * u ≤ 1 / 4 := by rw [hu_def]; norm_num [zpow_neg]
        have h140 : c140 ≤ 1 / 4 := by rw [hc140]; norm_num [zpow_neg]
        exact sp_one_add_e_neg hu0 h6u h140 ht2neg he
      have hwu : u + 2 * c150 ≤ 1 / 2 := by linarith [hc, hu1]
      have hA3 : A3 ≤ u + c46 := by
        rw [hA3def]
        exact sp_A3 hu0 hc150nn hwu h1e hs hlf
      have hA4 : A4 ≤ X := by
        rw [hA4def, hX]
        rcases le_or_gt (-(2 : ℝ) ^ 20) x0 with hlo | hlo
        · exact sp_A4_b1 hu_def hc140 hc150 he hxm hlo hD3
        · exact sp_A4_b2 hu_def hc140 hc150 he hlo hD3
      have hA5 : A5 ≤ X := by
        rw [hA5def, hLdef, hX]
        rcases le_or_gt (-(2 : ℝ) ^ 20) x0 with hlo | hlo
        · exact sp_A5_b1 hu_def hc150 hxm hlo hD3
        · exact sp_A5_b2 hu_def hc150 hlo hD3
      have hS : S ≤ L + A3 + A4 + A5 := by
        rw [hSdef, hA3def, hA4def, hA5def]
        have h3 := abs_sub_le (Real.log s) (Real.log (1 + e)) L
        have h4 := abs_sub_le (Real.log (1 + e)) (Real.log (1 + Real.exp t2)) L
        have h1 := abs_sub_le (Real.log s) L 0
        simp only [sub_zero] at h1
        rw [abs_of_nonneg hL0] at h1
        linarith [h1, h3, h4]
      have hlabs : labs ≤ S + A2 := by
        rw [hlabsdef, hSdef, hA2def]
        have h1 := abs_sub_le l (Real.log s) 0
        simp only [sub_zero] at h1
        linarith [h1]
      have htri : |l2 - L| ≤ A1 + A2 + A3 + A4 + A5 := by
        rw [hA1def, hA2def, hA3def, hA4def, hA5def]
        have h1 := abs_sub_le l2 l L
        have h2 := abs_sub_le l (Real.log s) L
        have h3 := abs_sub_le (Real.log s) (Real.log (1 + e)) L
        have h4 := abs_sub_le (Real.log (1 + e)) (Real.log (1 + Real.exp t2)) L
        linarith [h1, h2, h3, h4]
      have hconst : c46 + X + X + (50001 / 10000) * u * (X + X)
          + (50001 / 10000) * u * (u + c46) + u * c149 + c150 + c149 ≤ c43 := by
        rw [hX, hu_def, hc46, hc149, hc150, hc43]
        norm_num [zpow_neg]
      have hu1e : u ≤ 1 / 1000000 := by rw [hu_def]; norm_num [zpow_neg]
      have hcore := sp_core_neg hu0 hu1e hL0 hS0 hc149nn hc150nn hconst hA1 hA2 hA3 hA4
        hA5 hS hlabs
      linarith [htri, hcore]
  rcases le_or_gt x0 20 with hx | hx
  · rw [spRef_eq_log hx]
    have hle : 100 * u * Real.log (1 + Real.exp x0) + u + c43
        ≤ 104 * u * Real.log (1 + Real.exp x0) + u * (1 + c19) := by
      have h104 : u * (1 + c19) = u + c43 := by
        rw [mul_add, mul_one, h43]
      have huL : 0 ≤ u * Real.log (1 + Real.exp x0) := by
        have : 0 < Real.log (1 + Real.exp x0) := by
          apply Real.log_pos; linarith [Real.exp_pos x0]
        positivity
      nlinarith [huL, h104]
    linarith [hmain, hle]
  · rw [spRef_eq_self hx]
    have hx20 : (20 : ℝ) ≤ x0 := le_of_lt hx
    have hLle : Real.log (1 + Real.exp x0) ≤ x0 + c20 := by
      have h := log_sub_spRef_le x0
      rw [spRef_eq_self hx] at h
      linarith only [h]
    have hLge : x0 ≤ Real.log (1 + Real.exp x0) := log_one_add_exp_ge_self x0
    have hc20le : c20 ≤ 20 * u := by rw [hc20, hu_def]; norm_num [zpow_neg]
    have hu2 : u * u ≤ u * (1 / 1000000) :=
      mul_le_mul_of_nonneg_left (by rw [hu_def]; norm_num [zpow_neg]) (le_of_lt hu0)
    have hc20u : c20 + 100 * u * c20 ≤ 4 * u * x0 := by
      have hprod : c20 * u ≤ (20 * u) * u :=
        mul_le_mul_of_nonneg_right hc20le (le_of_lt hu0)
      have h2000 : 2000 * (u * u) ≤ 2000 * (u * (1 / 1000000)) :=
        mul_le_mul_of_nonneg_left hu2 (by norm_num)
      have hsmall : 2000 * (u * (1 / 1000000)) ≤ 20 * u := by
        have h := mul_le_mul_of_nonneg_right
          (by norm_num : (2000 : ℝ) * (1 / 1000000) ≤ 20) (le_of_lt hu0)
        calc 2000 * (u * (1 / 1000000)) = (2000 * (1 / 1000000)) * u := by ring
          _ ≤ 20 * u := h
      have hx4 : 4 * u * 20 ≤ 4 * u * x0 :=
        mul_le_mul_of_nonneg_left hx20 (by positivity)
      have hmul : 40 * u ≤ 4 * u * 20 := by
        have h := mul_le_mul_of_nonneg_left (by norm_num : (40 : ℝ) ≤ 4 * 20) (le_of_lt hu0)
        calc 40 * u = u * 40 := by ring
          _ ≤ u * (4 * 20) := h
          _ = 4 * u * 20 := by ring
      calc c20 + 100 * u * c20 = c20 + 100 * (c20 * u) := by ring
        _ ≤ (20 * u) + 100 * ((20 * u) * u) :=
            add_le_add hc20le (mul_le_mul_of_nonneg_left hprod (by norm_num))
        _ = 20 * u + 2000 * (u * u) := by ring
        _ ≤ 20 * u + 2000 * (u * (1 / 1000000)) := add_le_add le_rfl h2000
        _ ≤ 20 * u + 20 * u := add_le_add le_rfl hsmall
        _ = 40 * u := by ring
        _ ≤ 4 * u * x0 := le_trans hmul hx4
    have hmain2 := abs_le.mp hmain
    have hup : l2 - x0 ≤ 104 * u * x0 + u + c43 := by
      nlinarith only [hmain2.2, hLle, hc20u, hu0]
    have hlow : x0 - l2 ≤ 104 * u * x0 + u + c43 := by
      nlinarith only [hmain2.1, hLge, hLle, hc20u, hu0]
    have h52 : 52 * (2 * u) = 104 * u := by ring
    have htl : u * (1 + c19) = u + c43 := by
      rw [mul_add, mul_one, h43]
    rw [h52, htl, abs_le]
    exact ⟨by linarith only [hlow], by linarith only [hup]⟩

end Claims.GdnGating
