import Claim
import TVA.Proof
import Proof.Cast

/-!
# Real-number lemmas for the cap bound

Facts about `exp` and about the growth of the logit tolerance `tolL`: `2 + t ≤ 2·eᵗ`, bounds on
`|x|·eˣ` and on `exp(±2^-23·…)`, and `tolL d S ≤ 40.0002·u·S + 2^-127`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

/-! ## Generic real lemmas -/

/-- `2 + t ≤ 2·eᵗ` for `t ≥ 0`. -/
theorem two_add_le_two_exp {t : ℝ} (ht : 0 ≤ t) : 2 + t ≤ 2 * Real.exp t := by
  have h1 : 1 + t / 2 ≤ Real.exp (t / 2) := by
    have := Real.add_one_le_exp (t / 2)
    linarith
  have hp : (0 : ℝ) < 1 + t / 2 := by linarith
  have h2 : Real.exp t = Real.exp (t / 2) * Real.exp (t / 2) := by
    rw [← Real.exp_add]
    ring_nf
  have hsq : (1 + t / 2) * (1 + t / 2) ≤ Real.exp (t / 2) * Real.exp (t / 2) :=
    mul_le_mul h1 h1 hp.le (Real.exp_pos (t / 2)).le
  rw [h2]
  nlinarith [hsq, hp]

/-- `(2+L)·e^{−L} ≤ 2` for `L ≥ 0`. -/
theorem two_add_mul_exp_neg_le {L : ℝ} (hL : 0 ≤ L) :
    (2 + L) * Real.exp (-L) ≤ 2 := by
  have h := two_add_le_two_exp hL
  have he : (0 : ℝ) < Real.exp L := Real.exp_pos L
  rw [Real.exp_neg]
  have hstep : (2 + L) * (Real.exp L)⁻¹ ≤ (2 * Real.exp L) * (Real.exp L)⁻¹ :=
    mul_le_mul_of_nonneg_right h (by positivity)
  calc (2 + L) * (Real.exp L)⁻¹ ≤ (2 * Real.exp L) * (Real.exp L)⁻¹ := hstep
    _ = 2 := by rw [mul_assoc, mul_inv_cancel₀ he.ne', mul_one]

/-- `(2^-22 + 2^-23·L)·e^{−L} ≤ 2^-22` for `L ≥ 0`. -/
theorem exp_smul_le {L : ℝ} (hL : 0 ≤ L) :
    ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * L) * Real.exp (-L) ≤
      (2 : ℝ) ^ (-22 : ℤ) := by
  have hkey : 2 + L ≤ 2 * Real.exp L := two_add_le_two_exp hL
  have hfac : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * L
      = (2 : ℝ) ^ (-23 : ℤ) * (2 + L) := by
    have h2 : (2 : ℝ) ^ (-22 : ℤ) = (2 : ℝ) ^ (-23 : ℤ) * 2 := by norm_num
    rw [h2]; ring
  rw [hfac, Real.exp_neg]
  have he : (0 : ℝ) < Real.exp L := Real.exp_pos L
  have hstep : (2 : ℝ) ^ (-23 : ℤ) * (2 + L) * (Real.exp L)⁻¹
      ≤ (2 : ℝ) ^ (-23 : ℤ) * (2 * Real.exp L) * (Real.exp L)⁻¹ := by
    apply mul_le_mul_of_nonneg_right _ (by positivity)
    exact mul_le_mul_of_nonneg_left hkey (by norm_num)
  calc (2 : ℝ) ^ (-23 : ℤ) * (2 + L) * (Real.exp L)⁻¹
      ≤ (2 : ℝ) ^ (-23 : ℤ) * (2 * Real.exp L) * (Real.exp L)⁻¹ := hstep
    _ = (2 : ℝ) ^ (-22 : ℤ) := by
        have h4 : (2 : ℝ) ^ (-23 : ℤ) * (2 * Real.exp L) * (Real.exp L)⁻¹
            = ((2 : ℝ) ^ (-23 : ℤ) * 2) * (Real.exp L * (Real.exp L)⁻¹) := by ring
        rw [h4, mul_inv_cancel₀ he.ne', mul_one]
        norm_num

/-- `|x|·eˣ ≤ 2` for `x ≤ 0`. -/
theorem exp_mul_abs_le {x : ℝ} (hx : x ≤ 0) : |x| * Real.exp x ≤ 2 := by
  have hL : 0 ≤ -x := by linarith
  have h := two_add_mul_exp_neg_le hL
  rw [show -(-x) = x by ring] at h
  rw [abs_of_nonpos hx]
  have he : (0 : ℝ) < Real.exp x := Real.exp_pos x
  nlinarith [h, he]

/-- `exp(−2^23) ≤ 2^-150`. -/
theorem exp_neg_two_pow23_le :
    Real.exp (-(((2 ^ 23 : ℕ)) : ℝ)) ≤ (2 : ℝ) ^ (-150 : ℤ) := by
  have h2 : (2 : ℝ) ≤ Real.exp 1 := by
    have := Real.add_one_le_exp (1 : ℝ); linarith
  have hpow : (2 : ℝ) ^ (150 : ℕ) ≤ (Real.exp 1) ^ ((2 ^ 23 : ℕ)) := by
    calc (2 : ℝ) ^ (150 : ℕ) ≤ (2 : ℝ) ^ ((2 ^ 23 : ℕ)) :=
          pow_le_pow_right₀ (by norm_num) (by norm_num)
      _ ≤ (Real.exp 1) ^ ((2 ^ 23 : ℕ)) := pow_le_pow_left₀ (by norm_num) h2 _
  have hE : Real.exp (((2 ^ 23 : ℕ)) : ℝ) = (Real.exp 1) ^ ((2 ^ 23 : ℕ)) := by
    simp
  have hEge : (2 : ℝ) ^ (150 : ℕ) ≤ Real.exp (((2 ^ 23 : ℕ)) : ℝ) := by
    rw [hE]; exact hpow
  rw [Real.exp_neg]
  have h150 : (0 : ℝ) < (2 : ℝ) ^ (150 : ℕ) := by positivity
  calc (Real.exp (((2 ^ 23 : ℕ)) : ℝ))⁻¹ ≤ ((2 : ℝ) ^ (150 : ℕ))⁻¹ :=
        inv_anti₀ h150 hEge
    _ = (2 : ℝ) ^ (-150 : ℤ) := by norm_num

/-- An approximation of `eᵈ` within the `ExpF32` error is above `−2^-139`. -/
theorem exp_low_of_err {a d : ℝ} (hd : d ≤ 88.001)
    (h : |a - Real.exp d| ≤
      ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d + (2 : ℝ) ^ (-140 : ℤ)) :
    -((2 : ℝ) ^ (-139 : ℤ)) ≤ a := by
  have hEr : (0 : ℝ) < Real.exp d := Real.exp_pos d
  have hlow : Real.exp d -
      (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d + (2 : ℝ) ^ (-140 : ℤ)) ≤ a := by
    have h1 := (abs_le.mp h).1
    linarith only [h1]
  have hc1 : -((2 : ℝ) ^ (-139 : ℤ)) ≤ -((2 : ℝ) ^ (-140 : ℤ)) := by norm_num
  by_cases hd0 : d ≤ 0
  · set L : ℝ := -d with hLdef
    have hdabs : |d| = L := by rw [hLdef, abs_of_nonpos hd0]
    have hEexp : Real.exp d = Real.exp (-L) := by rw [hLdef]; ring_nf
    have hfac : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)
        = (2 : ℝ) ^ (-23 : ℤ) * (2 + L) := by
      rw [hdabs]
      have h2 : (2 : ℝ) ^ (-22 : ℤ) = (2 : ℝ) ^ (-23 : ℤ) * 2 := by norm_num
      rw [h2]; ring
    have hEL : (0 : ℝ) < Real.exp (-L) := Real.exp_pos _
    have key : -((2 : ℝ) ^ (-139 : ℤ)) ≤
        Real.exp d - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
          + (2 : ℝ) ^ (-140 : ℤ)) := by
      rw [hfac, hEexp]
      by_cases hLc : L ≤ (((2 ^ 23 : ℕ) : ℝ)) - 2
      · have h1 : (2 : ℝ) ^ (-23 : ℤ) * (2 + L) ≤ 1 := by
          have h2 : (2 : ℝ) ^ (-23 : ℤ) * (((2 ^ 23 : ℕ) : ℝ)) = 1 := by norm_num
          nlinarith only [h2, hLc, show (0 : ℝ) < (2 : ℝ) ^ (-23 : ℤ) by norm_num]
        have hAE : (2 : ℝ) ^ (-23 : ℤ) * (2 + L) * Real.exp (-L) ≤ Real.exp (-L) := by
          have hm := mul_le_mul_of_nonneg_right h1 hEL.le
          simpa using hm
        have step : -((2 : ℝ) ^ (-140 : ℤ)) ≤
            Real.exp (-L) - ((2 : ℝ) ^ (-23 : ℤ) * (2 + L) * Real.exp (-L)
              + (2 : ℝ) ^ (-140 : ℤ)) := by linarith only [hAE]
        exact le_trans hc1 step
      · have hLc' : (((2 ^ 23 : ℕ) : ℝ)) - 2 < L := not_le.mp hLc
        have hM0 : 0 ≤ L - (((2 ^ 23 : ℕ) : ℝ) - 2) := by linarith only [hLc']
        have hp : (0 : ℝ) < 2 + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)) := by linarith only [hLc']
        have hstep : Real.exp (-(L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
            ≤ 2 / (2 + (L - (((2 ^ 23 : ℕ) : ℝ) - 2))) := by
          rw [le_div_iff₀ hp, mul_comm]
          exact two_add_mul_exp_neg_le hM0
        have hT2 : (2 : ℝ) ≤ (((2 ^ 23 : ℕ) : ℝ)) := by norm_num
        have hkey : (2 + L) * Real.exp (-(L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
            ≤ (((2 ^ 23 : ℕ) : ℝ)) := by
          have h2 : 2 + L = (((2 ^ 23 : ℕ) : ℝ)) + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)) := by ring
          rw [h2]
          calc ((((2 ^ 23 : ℕ) : ℝ)) + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
                * Real.exp (-(L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
              ≤ ((((2 ^ 23 : ℕ) : ℝ)) + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
                * (2 / (2 + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)))) := by
                apply mul_le_mul_of_nonneg_left hstep
                linarith only [hT2, hM0]
            _ ≤ (((2 ^ 23 : ℕ) : ℝ)) := by
                have heq : ((((2 ^ 23 : ℕ) : ℝ)) + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
                    * (2 / (2 + (L - (((2 ^ 23 : ℕ) : ℝ) - 2))))
                    = 2 * ((((2 ^ 23 : ℕ) : ℝ)) + (L - (((2 ^ 23 : ℕ) : ℝ) - 2)))
                      / (2 + (L - (((2 ^ 23 : ℕ) : ℝ) - 2))) := by ring
                rw [heq, div_le_iff₀ hp]
                nlinarith only [hT2, hM0]
        have hsplit : Real.exp (-L)
            = Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2)) * Real.exp (-(L - (((2 ^ 23 : ℕ) : ℝ) - 2))) := by
          rw [← Real.exp_add]; ring_nf
        have hdec : (2 + L) * Real.exp (-L)
            ≤ (((2 ^ 23 : ℕ) : ℝ)) * Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2)) := by
          rw [hsplit]
          calc (2 + L) * (Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2))
                * Real.exp (-(L - (((2 ^ 23 : ℕ) : ℝ) - 2))))
              = ((2 + L) * Real.exp (-(L - (((2 ^ 23 : ℕ) : ℝ) - 2))))
                * Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2)) := by ring
            _ ≤ (((2 ^ 23 : ℕ) : ℝ)) * Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2)) :=
                mul_le_mul_of_nonneg_right hkey (Real.exp_pos _).le
        have he : Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2))
            = Real.exp 2 * Real.exp (-(((2 ^ 23 : ℕ) : ℝ))) := by
          rw [← Real.exp_add]; ring_nf
        have hT23 : (2 : ℝ) ^ (-23 : ℤ) * (((2 ^ 23 : ℕ) : ℝ)) = 1 := by norm_num
        have he2 : Real.exp 2 ≤ 9 := by
          have h1 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
            rw [← Real.exp_add]; norm_num
          nlinarith only [h1, Real.exp_one_lt_three, Real.exp_pos (1 : ℝ)]
        have hsmall : Real.exp (-(((2 ^ 23 : ℕ) : ℝ))) ≤ (2 : ℝ) ^ (-150 : ℤ) :=
          exp_neg_two_pow23_le
        have hnum : (2 : ℝ) ^ (-23 : ℤ)
            * ((((2 ^ 23 : ℕ) : ℝ)) * Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2)))
            ≤ (2 : ℝ) ^ (-140 : ℤ) := by
          calc (2 : ℝ) ^ (-23 : ℤ)
                * ((((2 ^ 23 : ℕ) : ℝ)) * Real.exp (-((((2 ^ 23 : ℕ) : ℝ)) - 2)))
              = ((2 : ℝ) ^ (-23 : ℤ) * (((2 ^ 23 : ℕ) : ℝ)))
                * (Real.exp 2 * Real.exp (-(((2 ^ 23 : ℕ) : ℝ)))) := by rw [he]; ring
            _ = Real.exp 2 * Real.exp (-(((2 ^ 23 : ℕ) : ℝ))) := by rw [hT23, one_mul]
            _ ≤ 9 * (2 : ℝ) ^ (-150 : ℤ) := by
                exact mul_le_mul he2 hsmall (by positivity) (by norm_num)
            _ ≤ (2 : ℝ) ^ (-140 : ℤ) := by norm_num
        have hAE : (2 : ℝ) ^ (-23 : ℤ) * (2 + L) * Real.exp (-L) ≤ (2 : ℝ) ^ (-140 : ℤ) := by
          have hm := mul_le_mul_of_nonneg_left hdec (show (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) by norm_num)
          linarith only [hm, hnum]
        have hc2 : -((2 : ℝ) ^ (-139 : ℤ)) ≤
            -((2 : ℝ) ^ (-140 : ℤ)) - (2 : ℝ) ^ (-140 : ℤ) := by norm_num
        have step : -((2 : ℝ) ^ (-140 : ℤ)) - (2 : ℝ) ^ (-140 : ℤ) ≤
            Real.exp (-L) - ((2 : ℝ) ^ (-23 : ℤ) * (2 + L) * Real.exp (-L)
              + (2 : ℝ) ^ (-140 : ℤ)) := by linarith only [hAE, hEL]
        exact le_trans hc2 step
    linarith only [hlow, key]
  · have hdpos' : 0 < d := not_le.mp hd0
    have habs : |d| = d := abs_of_pos hdpos'
    have hrel : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d| ≤ 1 := by
      rw [habs]
      have hm := mul_le_mul_of_nonneg_left hd (show (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) by norm_num)
      have hkey : (2 : ℝ) ^ (-23 : ℤ) * d + (2 : ℝ) ^ (-22 : ℤ)
          ≤ (2 : ℝ) ^ (-23 : ℤ) * (88.001 : ℝ) + (2 : ℝ) ^ (-22 : ℤ) :=
        add_le_add_left hm _
      have hb : (2 : ℝ) ^ (-23 : ℤ) * (88.001 : ℝ) + (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
      have hgoal : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * d
          = (2 : ℝ) ^ (-23 : ℤ) * d + (2 : ℝ) ^ (-22 : ℤ) := by ring
      rw [hgoal]
      exact le_trans hkey hb
    have hAE : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
        ≤ Real.exp d := by
      have hm := mul_le_mul_of_nonneg_right hrel hEr.le
      simpa using hm
    have step : -((2 : ℝ) ^ (-140 : ℤ)) ≤
        Real.exp d - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
          + (2 : ℝ) ^ (-140 : ℤ)) := by linarith only [hAE]
    have key : -((2 : ℝ) ^ (-139 : ℤ)) ≤
        Real.exp d - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
          + (2 : ℝ) ^ (-140 : ℤ)) := le_trans hc1 step
    linarith only [hlow, key]

/-- `exp 88.001 ≤ 0.99·2^127`. -/
theorem exp_88_le : Real.exp (88.001) ≤ (2 : ℝ) ^ 127 * (99 / 100) := by
  have hpos : (0 : ℝ) < (2 : ℝ) ^ 127 * (99 / 100) := by positivity
  have hlog : Real.log ((2 : ℝ) ^ 127 * (99 / 100))
      = 127 * Real.log 2 + Real.log (99 / 100) := by
    rw [Real.log_mul (by positivity) (by norm_num), Real.log_pow]
    norm_num
  have hl2 : 0.6931471803 < Real.log 2 := Real.log_two_gt_d9
  have hl99 : -(1 : ℝ) / 99 ≤ Real.log (99 / 100) := by
    have h1 : Real.log (100 / 99) ≤ 100 / 99 - 1 :=
      Real.log_le_sub_one_of_pos (by norm_num)
    have h2 : Real.log (99 / 100) = -Real.log (100 / 99) := by
      rw [show (99 / 100 : ℝ) = (100 / 99)⁻¹ by norm_num, Real.log_inv]
    rw [h2]; linarith
  have hgt : (88.001 : ℝ) < Real.log ((2 : ℝ) ^ 127 * (99 / 100)) := by
    rw [hlog]; nlinarith
  calc Real.exp 88.001 ≤ Real.exp (Real.log ((2 : ℝ) ^ 127 * (99 / 100))) :=
        Real.exp_le_exp.mpr hgt.le
    _ = (2 : ℝ) ^ 127 * (99 / 100) := Real.exp_log hpos

/-- The `ExpF32` result bound stays below the largest finite `float32` for `x ≤ 88.001`. -/
theorem room_exp {x : ℝ} (hx : x ≤ 88.001) :
    (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x|)) * Real.exp x +
        (2 : ℝ) ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  rw [maxVal32_eq]
  have h23p : (0 : ℝ) < (2 : ℝ) ^ (-23 : ℤ) := by norm_num
  by_cases hx0 : 0 ≤ x
  · have hexp : Real.exp x ≤ Real.exp (88.001) := Real.exp_le_exp.mpr hx
    have habs : |x| ≤ 88.001 := by rw [abs_of_nonneg hx0]; exact hx
    have hbx : (2 : ℝ) ^ (-23 : ℤ) * |x| ≤ (2 : ℝ) ^ (-23 : ℤ) * 88.001 :=
      mul_le_mul_of_nonneg_left habs h23p.le
    have hn : (2 : ℝ) ^ (-23 : ℤ) * 88.001 ≤ 1 - (2 : ℝ) ^ (-22 : ℤ) := by norm_num
    have hA1 : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x|)) ≤ 2 := by
      have hb : (2 : ℝ) ^ (-23 : ℤ) * |x| ≤ 1 - (2 : ℝ) ^ (-22 : ℤ) := le_trans hbx hn
      linarith
    have hE0 : (0 : ℝ) ≤ Real.exp x := (Real.exp_pos x).le
    have hmul : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x|)) * Real.exp x
        ≤ 2 * ((2 : ℝ) ^ 127 * (99 / 100)) := by
      calc (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x|)) * Real.exp x
          ≤ 2 * Real.exp x := mul_le_mul_of_nonneg_right hA1 hE0
        _ ≤ 2 * ((2 : ℝ) ^ 127 * (99 / 100)) :=
            mul_le_mul_of_nonneg_left (le_trans hexp exp_88_le) (by norm_num)
    have hfin : 2 * ((2 : ℝ) ^ 127 * (99 / 100)) + (2 : ℝ) ^ (-140 : ℤ)
        ≤ (2 : ℝ) ^ 128 - (2 : ℝ) ^ 104 := by norm_num
    calc (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x|)) * Real.exp x
          + (2 : ℝ) ^ (-140 : ℤ)
        ≤ 2 * ((2 : ℝ) ^ 127 * (99 / 100)) + (2 : ℝ) ^ (-140 : ℤ) := add_le_add_left hmul _
      _ ≤ (2 : ℝ) ^ 128 - (2 : ℝ) ^ 104 := hfin
  · have hxn : x ≤ 0 := le_of_lt (not_le.mp hx0)
    have habs : |x| * Real.exp x ≤ 2 := exp_mul_abs_le hxn
    have hexp1 : Real.exp x ≤ 1 := by
      rw [← Real.exp_zero]
      exact Real.exp_le_exp.mpr hxn
    have h1 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * Real.exp x ≤ 1 + (2 : ℝ) ^ (-22 : ℤ) := by
      calc (1 + (2 : ℝ) ^ (-22 : ℤ)) * Real.exp x
          ≤ (1 + (2 : ℝ) ^ (-22 : ℤ)) * 1 :=
            mul_le_mul_of_nonneg_left hexp1 (by norm_num)
        _ = 1 + (2 : ℝ) ^ (-22 : ℤ) := by ring
    have h2 : (2 : ℝ) ^ (-23 : ℤ) * |x| * Real.exp x ≤ 2 * (2 : ℝ) ^ (-23 : ℤ) := by
      nlinarith [habs, h23p, Real.exp_pos x]
    have hsplit : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |x|)) * Real.exp x
        = (1 + (2 : ℝ) ^ (-22 : ℤ)) * Real.exp x
          + (2 : ℝ) ^ (-23 : ℤ) * |x| * Real.exp x := by ring
    rw [hsplit]
    have hb : (1 + (2 : ℝ) ^ (-22 : ℤ)) + 2 * (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)
        ≤ (2 : ℝ) ^ 128 - (2 : ℝ) ^ 104 := by norm_num
    linarith

/-! ## The growth of the logit tolerance -/

/-- `(1+3u)ⁿ ≤ 1 + 39.0001u` for `n ≤ 13`. -/
theorem growth_sharp {n : Nat} (hn : n ≤ 13) :
    (1 + 3 * u32) ^ n ≤ 1 + 39.0001 * u32 := by
  have hu : u32 = 2 ^ (-24 : ℤ) := rfl
  have h1 : 1 + 3 * u32 ≤ Real.exp (3 * u32) := by
    linarith [Real.add_one_le_exp (3 * u32)]
  have h0 : (0 : ℝ) ≤ 1 + 3 * u32 := by rw [hu]; norm_num
  have hu' : (0:ℝ) < u32 := u32_pos
  have hn13 : (n : ℝ) ≤ 13 := by exact_mod_cast hn
  have harg : (n : ℝ) * (3 * u32) ≤ 39 * u32 := by nlinarith
  have hx0 : (0 : ℝ) ≤ 39 * u32 := by linarith
  have hlt : 39 * u32 < 1 := by rw [hu]; norm_num
  have hfinal : 1 / (1 - 39 * u32) ≤ 1 + 39.0001 * u32 := by
    rw [div_le_iff₀ (by linarith)]
    rw [hu]
    norm_num
  calc (1 + 3 * u32) ^ n ≤ Real.exp (3 * u32) ^ n := pow_le_pow_left₀ h0 h1 n
    _ = Real.exp ((n : ℝ) * (3 * u32)) := (Real.exp_nat_mul _ n).symm
    _ ≤ Real.exp (39 * u32) := Real.exp_le_exp.mpr harg
    _ ≤ 1 / (1 - 39 * u32) := Real.exp_bound_div_one_sub_of_interval hx0 hlt
    _ ≤ 1 + 39.0001 * u32 := hfinal

/-- The logit tolerance is at most `40.0002·u·S + 2^-127`. -/
theorem tolL_le (d : Inst) (hd : d.ok) {S : ℝ} (hS : 0 ≤ S) :
    tolL d S ≤ 40.0002 * u32 * S + 2 ^ (-127 : ℤ) := by
  have hu : (0:ℝ) < u32 := u32_pos
  have he : (0:ℝ) < eta32 := eta32_pos
  have hD : Nat.clog 2 d.BLOCK ≤ 13 := by
    have := Nat.clog_mono_right 2 hd.block_le
    rwa [Nat.clog_pow 2 13 (by norm_num)] at this
  have hG := growth_sharp hD
  have hB' : ((d.BLOCK - 1 : ℕ) : ℝ) ≤ 8192 := by
    exact_mod_cast (by have := hd.block_le; omega : d.BLOCK - 1 ≤ 8192)
  have hh' : (d.hidden : ℝ) ≤ 131072 := by
    exact_mod_cast (by have := hd.hidden_le17; omega : d.hidden ≤ 131072)
  unfold tolL
  generalize hGdef : (1 + 3 * u32) ^ Nat.clog 2 d.BLOCK = G at hG ⊢
  have hG1 : 1 ≤ G := hGdef ▸ one_le_pow₀ (by linarith : (1:ℝ) ≤ 1 + 3*u32)
  have hhη : (d.hidden : ℝ) * eta32 ≤ 2 ^ (-130 : ℤ) := by
    have h1 : (d.hidden : ℝ) * eta32 ≤ 131072 * eta32 := mul_le_mul_of_nonneg_right hh' he.le
    have h2 : (131072 : ℝ) * eta32 ≤ 2 ^ (-130 : ℤ) := by
      rw [show (131072:ℝ) = 2^17 by norm_num, show eta32 = 2^(-150:ℤ) from rfl]
      norm_num
    exact le_trans h1 h2
  have hη1 : 39.0001 * u32 * ((d.hidden : ℝ) * eta32) ≤ 2 ^ (-130 : ℤ) := by
    have h1 : 39.0001 * u32 * ((d.hidden : ℝ) * eta32) ≤ 39.0001 * u32 * (2 ^ (-130 : ℤ)) := by
      apply mul_le_mul_of_nonneg_left hhη
      positivity
    have h2 : 39.0001 * u32 * (2 ^ (-130 : ℤ)) ≤ 2 ^ (-130 : ℤ) := by
      rw [show u32 = 2^(-24:ℤ) from rfl]
      have h3 : (39.0001 : ℝ) * 2^(-24:ℤ) ≤ 1 := by norm_num
      nlinarith [show (0:ℝ) < 2^(-130:ℤ) by norm_num]
    exact le_trans h1 h2
  have hη2 : 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G ≤ 2 ^ (-130 : ℤ) := by
    have h1 : 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G ≤
        4 * eta32 * 8192 * (1 + 39.0001 * u32) := by
      apply mul_le_mul _ hG _ _
      · apply mul_le_mul_of_nonneg_left hB' (by positivity)
      · linarith
      · positivity
    have h2 : 4 * eta32 * 8192 * (1 + 39.0001 * u32) ≤ 2 ^ (-130 : ℤ) := by
      rw [show eta32 = 2^(-150:ℤ) from rfl, show u32 = 2^(-24:ℤ) from rfl]
      norm_num
    exact le_trans h1 h2
  have hS' : 39.0001 * u32 * (1 + u32) * S + u32 * S ≤ 40.0002 * u32 * S := by
    have hc : 39.0001 * u32 * (1 + u32) + u32 ≤ 40.0002 * u32 := by
      rw [show u32 = 2^(-24:ℤ) from rfl]; norm_num
    have h := mul_le_mul_of_nonneg_right hc hS
    nlinarith
  have hX : (G - 1) * ((1 + u32) * S + (d.hidden : ℝ) * eta32) ≤
      39.0001 * u32 * (1 + u32) * S + 39.0001 * u32 * ((d.hidden : ℝ) * eta32) := by
    have h1 : (G - 1) * ((1 + u32) * S + (d.hidden : ℝ) * eta32) ≤
        39.0001 * u32 * ((1 + u32) * S + (d.hidden : ℝ) * eta32) := by
      apply mul_le_mul_of_nonneg_right _ (by positivity)
      linarith
    have h2 : 39.0001 * u32 * ((1 + u32) * S + (d.hidden : ℝ) * eta32) =
        39.0001 * u32 * (1 + u32) * S + 39.0001 * u32 * ((d.hidden : ℝ) * eta32) := by
      ring
    linarith
  have hsumη : 39.0001 * u32 * ((d.hidden : ℝ) * eta32)
      + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + (d.hidden : ℝ) * eta32
      ≤ 2 ^ (-127 : ℤ) := by
    have hb : 39.0001 * u32 * ((d.hidden : ℝ) * eta32)
        + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + (d.hidden : ℝ) * eta32
        ≤ (2:ℝ) ^ (-130 : ℤ) + (2:ℝ) ^ (-130 : ℤ) + (2:ℝ) ^ (-130 : ℤ) :=
      add_le_add (add_le_add hη1 hη2) hhη
    have hc : (2:ℝ) ^ (-130 : ℤ) + (2:ℝ) ^ (-130 : ℤ) + (2:ℝ) ^ (-130 : ℤ)
        ≤ (2:ℝ) ^ (-127 : ℤ) := by norm_num
    exact le_trans hb hc
  have hstep1 : (G - 1) * ((1 + u32) * S + (d.hidden : ℝ) * eta32)
      + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + u32 * S + (d.hidden : ℝ) * eta32
      ≤ (39.0001 * u32 * (1 + u32) * S + 39.0001 * u32 * ((d.hidden : ℝ) * eta32))
        + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + u32 * S + (d.hidden : ℝ) * eta32 := by
    linarith [hX]
  have hstep2 : (39.0001 * u32 * (1 + u32) * S + 39.0001 * u32 * ((d.hidden : ℝ) * eta32))
        + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + u32 * S + (d.hidden : ℝ) * eta32
      = (39.0001 * u32 * (1 + u32) * S + u32 * S)
        + (39.0001 * u32 * ((d.hidden : ℝ) * eta32)
          + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + (d.hidden : ℝ) * eta32) := by
    ring
  have hstep3 : (39.0001 * u32 * (1 + u32) * S + u32 * S)
      + (39.0001 * u32 * ((d.hidden : ℝ) * eta32)
        + 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G + (d.hidden : ℝ) * eta32)
      ≤ 40.0002 * u32 * S + 2 ^ (-127 : ℤ) := add_le_add hS' hsumη
  exact le_trans (le_trans hstep1 hstep2.le) hstep3

end Claims.MoeRouter
