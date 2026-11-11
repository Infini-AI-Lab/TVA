import Claim
import Proof.Defs
import TVA.Proof

/-!
# Num

The numeric core: the per-token error of the scaled `B` (`hb_chain`) and the final tolerance (`value_bound`).
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000

set_option maxHeartbeats 4000000

open TVA

/-- The relative error of one `tl.dot` step of `64` products. -/
noncomputable def rho : ℚ := DotLoop.stepRel Device.relDot16 BK
/-- The absolute error of one `tl.dot` step of `64` products. -/
noncomputable def alp : ℚ := DotLoop.stepAbs Device.absDot16 BK
private lemma exp_factor_le {s : ℝ} (hs0 : 0 ≤ s) (hs80 : s ≤ 80) :
    (1 + ((2:ℝ) ^ (-22 : ℤ) + (2:ℝ) ^ (-23 : ℤ) * (s + (s * (2:ℝ) ^ (-24 : ℤ) + (2:ℝ) ^ (-150 : ℤ)))))
      * Real.exp (s * (2:ℝ) ^ (-24 : ℤ) + (2:ℝ) ^ (-150 : ℤ))
      ≤ 1 + (2:ℝ) ^ (-16 : ℤ) := by
  set d : ℝ := s * (2:ℝ) ^ (-24 : ℤ) + (2:ℝ) ^ (-150 : ℤ) with hd
  set r : ℝ := (2:ℝ) ^ (-22 : ℤ) + (2:ℝ) ^ (-23 : ℤ) * (s + d) with hr
  set Dm : ℝ := 80 * (2:ℝ) ^ (-24 : ℤ) + (2:ℝ) ^ (-150 : ℤ) with hDm
  set B : ℝ := (2:ℝ) ^ (-22 : ℤ) + (2:ℝ) ^ (-23 : ℤ) * (80 + Dm) with hB
  have hd0 : 0 ≤ d := by rw [hd]; positivity
  have hd1 : d ≤ 1 := by
    rw [hd]
    have h1 : s * (2:ℝ) ^ (-24 : ℤ) ≤ 80 * (2:ℝ) ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hs80 (by positivity)
    have h2 : (80 : ℝ) * (2:ℝ) ^ (-24 : ℤ) + (2:ℝ) ^ (-150 : ℤ) ≤ 1 := by norm_num
    linarith
  have hdabs : |d| ≤ 1 := by rw [abs_of_nonneg hd0]; exact hd1
  have hb := Real.exp_bound (x := d) (n := 2) hdabs (by norm_num)
  have hseries : (∑ m ∈ Finset.range 2, d ^ m / (m.factorial : ℝ)) = 1 + d := by
    simp [Finset.sum_range_succ]
  rw [hseries] at hb
  have hexp : Real.exp d ≤ 1 + d + d ^ 2 := by
    have hb2 : |Real.exp d - (1 + d)| ≤ d ^ 2 * ((3 : ℝ) / (2 * 2)) := by
      simpa [abs_of_nonneg hd0, Nat.factorial] using hb
    have h4 : Real.exp d - (1 + d) ≤ d ^ 2 * ((3 : ℝ) / (2 * 2)) := (le_abs_self _).trans hb2
    nlinarith [sq_nonneg d, show ((3:ℝ)/(2*2)) ≤ 1 by norm_num]
  have hr0 : 0 ≤ r := by rw [hr]; positivity
  have hd_le : d ≤ Dm := by
    rw [hd, hDm]
    nlinarith [mul_le_mul_of_nonneg_right hs80 (show (0:ℝ) ≤ (2:ℝ) ^ (-24 : ℤ) by positivity)]
  have hr_le : r ≤ B := by
    rw [hr, hB, hDm] at *
    have h1 : s + d ≤ 80 + (80 * (2:ℝ) ^ (-24 : ℤ) + (2:ℝ) ^ (-150 : ℤ)) := by
      linarith
    nlinarith [show (0:ℝ) ≤ (2:ℝ) ^ (-23 : ℤ) by positivity]
  have hnum : (1 + B) * (1 + Dm + Dm ^ 2) ≤ 1 + (2:ℝ) ^ (-16 : ℤ) := by
    rw [hB, hDm]
    norm_num
  have hmono : (1 + r) * (1 + d + d ^ 2) ≤ (1 + B) * (1 + Dm + Dm ^ 2) := by
    have h1 : 1 + r ≤ 1 + B := by linarith
    have h2 : 1 + d + d ^ 2 ≤ 1 + Dm + Dm ^ 2 := by
      nlinarith [hd0, hd_le]
    exact mul_le_mul h1 h2 (by positivity) (by positivity)
  have hstep : (1 + r) * Real.exp d ≤ (1 + r) * (1 + d + d ^ 2) :=
    mul_le_mul_of_nonneg_left hexp (by linarith)
  linarith

-- decay of the polynomial factor against the exponential
private lemma t_decay {t : ℝ} (ht : 0 ≤ t) :
    (1 + 81 * (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-22:ℤ) * t)
      * Real.exp (-(t * (1 - (2:ℝ)^(-24:ℤ))))
      ≤ 1 + 81 * (2:ℝ)^(-22:ℤ) := by
  set A0 : ℝ := 1 + 81 * (2:ℝ)^(-22:ℤ) with hA0
  set c : ℝ := (2:ℝ)^(-22:ℤ) with hc
  set u : ℝ := (2:ℝ)^(-24:ℤ) with hu
  have hu2 : u ≤ 1/2 := by rw [hu]; norm_num
  have hA0pos : 0 < A0 := by rw [hA0]; norm_num
  have hcpos : 0 < c := by rw [hc]; positivity
  have hcA : 2 * c ≤ A0 := by rw [hA0, hc]; norm_num
  have hle1 : -(t * (1 - u)) ≤ -(t/2) := by nlinarith
  have he1 : Real.exp (-(t * (1 - u))) ≤ Real.exp (-(t/2)) := Real.exp_le_exp.mpr hle1
  have he2 : Real.exp (-(t/2)) ≤ 2/(2+t) := by
    have h1 : 1 + t/2 ≤ Real.exp (t/2) := by linarith [Real.add_one_le_exp (t/2)]
    have hpos : (0:ℝ) < 1 + t/2 := by linarith
    have hmono := one_div_le_one_div_of_le hpos h1
    have e1 : Real.exp (-(t/2)) = 1 / Real.exp (t/2) := by
      rw [Real.exp_neg]; ring
    have e2 : 1 / (1 + t/2) = 2/(2+t) := by
      field_simp
    rw [e1, ← e2]
    exact hmono
  have he : Real.exp (-(t * (1 - u))) ≤ 2/(2+t) := he1.trans he2
  have h2t : (0:ℝ) < 2 + t := by linarith
  have hfac0 : 0 ≤ A0 + c * t := by positivity
  have hmul : (A0 + c*t) * Real.exp (-(t * (1 - u))) ≤ (A0 + c*t) * (2/(2+t)) :=
    mul_le_mul_of_nonneg_left he hfac0
  have hfin : (A0 + c*t) * (2/(2+t)) ≤ A0 := by
    have h : (A0 + c*t) * 2 / (2 + t) ≤ A0 := by
      rw [div_le_iff₀ h2t]
      nlinarith
    calc (A0 + c*t) * (2/(2+t)) = (A0 + c*t) * 2 / (2+t) := by ring
      _ ≤ A0 := h
  exact le_trans hmul hfin

private lemma exp_abs_le {D : ℝ} (hD : D ≤ 64) (hs : 80 < |D|) :
    (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
      * Real.exp (D + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))
      ≤ (2:ℝ)^(-106:ℤ) := by
  set s : ℝ := |D| with hsd
  set u : ℝ := (2:ℝ)^(-24:ℤ) with hu
  set e : ℝ := (2:ℝ)^(-150:ℤ) with he
  set d : ℝ := s*u + e with hd
  set t : ℝ := s - 80 with ht
  set R : ℝ := (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (s + d) with hR
  have hs0 : 0 ≤ s := by rw [hsd]; exact abs_nonneg D
  have hs80 : 80 < s := by rw [hsd]; exact hs
  have ht0 : 0 ≤ t := by rw [ht]; linarith
  have hu1 : u ≤ 1 := by rw [hu]; norm_num
  have h80u : 80 * u ≤ 1 := by rw [hu]; norm_num
  have he0 : 0 ≤ e := by rw [he]; positivity
  have he1 : e ≤ 1 := by rw [he]; norm_num
  have hd0 : 0 ≤ d := by rw [hd]; positivity
  have hdle : d ≤ s := by
    have hu2 : u ≤ 1/2 := by rw [hu]; norm_num
    have hsu2 : s * u ≤ s/2 := by nlinarith
    have hesmall : e ≤ s/2 := by linarith
    rw [hd]
    exact le_trans (add_le_add hsu2 hesmall) (by linarith)
  have hDneg : D < 0 := by
    rcases lt_or_ge D 0 with h | h
    · exact h
    · have habs : |D| = D := abs_of_nonneg h
      rw [hsd, habs] at hs
      linarith
  have hDabs : D = -s := by
    rw [hsd, abs_of_neg hDneg]
    ring
  have hrA : 1 + R ≤ (1 + 81*(2:ℝ)^(-22:ℤ)) + (2:ℝ)^(-22:ℤ)*t := by
    rw [hR, ht]
    have h1 : s + d ≤ 2*s := by linarith
    have h2 : (2:ℝ)^(-23:ℤ) * (s+d) ≤ (2:ℝ)^(-23:ℤ) * (2*s) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have h3 : (2:ℝ)^(-23:ℤ) * (2*s) = (2:ℝ)^(-22:ℤ) * s := by
      have h4 : (2:ℝ)^(-22:ℤ) = (2:ℝ)^(-23:ℤ) * 2 := by norm_num
      rw [h4]; ring
    nlinarith [h2, h3]
  have hsplit : Real.exp (D + d) =
      Real.exp (-(80*(1-u)) + e) * Real.exp (-(t*(1-u))) := by
    rw [← Real.exp_add]
    have harg : D + d = (-(80*(1-u)) + e) + (-(t*(1-u))) := by
      rw [hd, hDabs, ht]; ring
    rw [harg]
  have hexp80 : Real.exp (-(80*(1-u))) ≤ (2:ℝ)^(-110:ℤ) := by
    have hle : (79:ℝ) ≤ 80*(1-u) := by nlinarith
    have hmono : Real.exp (79:ℝ) ≤ Real.exp (80*(1-u)) := Real.exp_le_exp.mpr hle
    have hexp79 : Real.exp (79:ℝ) = (Real.exp 1)^79 := by
      rw [show (79:ℝ) = (79:ℕ) * 1 by norm_num, Real.exp_nat_mul]
    have hbase : (2.7:ℝ)^79 ≤ (Real.exp 1)^79 :=
      pow_le_pow_left₀ (by norm_num) (by linarith [Real.exp_one_gt_d9]) 79
    have hnum : (2:ℝ)^110 ≤ (2.7:ℝ)^79 := by norm_num
    have h2 : (2:ℝ)^110 ≤ Real.exp (80*(1-u)) := by
      calc (2:ℝ)^110 ≤ (2.7:ℝ)^79 := hnum
        _ ≤ (Real.exp 1)^79 := hbase
        _ = Real.exp (79:ℝ) := hexp79.symm
        _ ≤ Real.exp (80*(1-u)) := hmono
    have hpos : (0:ℝ) < (2:ℝ)^110 := by positivity
    have hinv : (Real.exp (80*(1-u)))⁻¹ ≤ ((2:ℝ)^110)⁻¹ := by
      simpa [one_div] using one_div_le_one_div_of_le hpos h2
    have e1 : Real.exp (-(80*(1-u))) = (Real.exp (80*(1-u)))⁻¹ := Real.exp_neg _
    have e2 : ((2:ℝ)^110)⁻¹ = (2:ℝ)^(-110:ℤ) := by norm_num
    rw [e1, e2] at *
    exact hinv
  have hee : Real.exp e ≤ 3 := by
    have h1 : Real.exp e ≤ Real.exp 1 := Real.exp_le_exp.mpr he1
    linarith [Real.exp_one_lt_three]
  have hA0le : 1 + 81*(2:ℝ)^(-22:ℤ) ≤ 2 := by norm_num
  have hdec := t_decay (t := t) ht0
  have hdec' : ((1 + 81*(2:ℝ)^(-22:ℤ)) + (2:ℝ)^(-22:ℤ)*t)
      * Real.exp (-(t*(1-u))) ≤ 1 + 81*(2:ℝ)^(-22:ℤ) := by
    simpa [hu] using hdec
  have hE0 : (0:ℝ) ≤ Real.exp (-(80*(1-u)) + e) := (Real.exp_pos _).le
  have hT0 : (0:ℝ) ≤ Real.exp (-(t*(1-u))) := (Real.exp_pos _).le
  have hF0 : (0:ℝ) ≤ (1 + 81*(2:ℝ)^(-22:ℤ)) + (2:ℝ)^(-22:ℤ)*t := by positivity
  have hexpE : Real.exp (-(80*(1-u)) + e) ≤ (2:ℝ)^(-110:ℤ) * 3 := by
    rw [Real.exp_add]
    exact mul_le_mul hexp80 hee (Real.exp_pos _).le (by positivity)
  calc (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (s + d))) * Real.exp (D + d)
      = (1 + R) * (Real.exp (-(80*(1-u)) + e) * Real.exp (-(t*(1-u)))) := by rw [hR, hsplit]
    _ = ((1 + R) * Real.exp (-(t*(1-u)))) * Real.exp (-(80*(1-u)) + e) := by ring
    _ ≤ (((1 + 81*(2:ℝ)^(-22:ℤ)) + (2:ℝ)^(-22:ℤ)*t) * Real.exp (-(t*(1-u))))
          * Real.exp (-(80*(1-u)) + e) := by
      apply mul_le_mul_of_nonneg_right _ hE0
      exact mul_le_mul_of_nonneg_right hrA hT0
    _ ≤ ((1 + 81*(2:ℝ)^(-22:ℤ)) * Real.exp (-(80*(1-u)) + e)) := by
      apply mul_le_mul_of_nonneg_right hdec' hE0
    _ ≤ 2 * ((2:ℝ)^(-110:ℤ) * 3) := by
      apply mul_le_mul hA0le hexpE (by positivity) (by norm_num)
    _ ≤ (2:ℝ)^(-106:ℤ) := by norm_num

private lemma errE_le {D : ℝ} (hD : D ≤ 64) :
    (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
      * Real.exp (D + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ))) - Real.exp D + (2:ℝ)^(-140:ℤ)
      ≤ (2:ℝ)^(-16:ℤ) * Real.exp D + (2:ℝ)^(-105:ℤ) := by
  have hD0 : (0:ℝ) ≤ Real.exp D := (Real.exp_pos D).le
  by_cases hs : |D| ≤ 80
  · have hfac := exp_factor_le (s := |D|) (abs_nonneg D) hs
    have hsplit : Real.exp (D + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))
        = Real.exp D * Real.exp (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)) := Real.exp_add ..
    rw [hsplit]
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
        * (Real.exp D * Real.exp (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))
        ≤ (1 + (2:ℝ)^(-16:ℤ)) * Real.exp D := by
      calc (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
            * (Real.exp D * Real.exp (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))
          = ((1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
              * Real.exp (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ))) * Real.exp D := by ring
        _ ≤ (1 + (2:ℝ)^(-16:ℤ)) * Real.exp D := mul_le_mul_of_nonneg_right hfac hD0
    have h140 : (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^(-105:ℤ) := by norm_num
    linarith
  · have hs' : 80 < |D| := lt_of_not_ge hs
    have habs := exp_abs_le hD hs'
    have h140 : (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^(-105:ℤ) := by norm_num
    have h106 : (2:ℝ)^(-106:ℤ) + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^(-105:ℤ) := by norm_num
    have hc16 : (0:ℝ) ≤ (2:ℝ)^(-16:ℤ) * Real.exp D := by positivity
    linarith
/-! ## Mirror definitions and constants for the per-token error `errB` -/

/-- Real mirror of the IEEE subtraction's error `dErr` at exponent `D` (README, Tolerances). -/
private noncomputable def dErrF (D : ℝ) : ℝ := |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)
/-- Real mirror of `Device.ExpF32`'s relative constant at `X = |D| + dErr` (README, Tolerances). -/
private noncomputable def expRelF (D : ℝ) : ℝ := (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + dErrF D)
/-- Real mirror of the error of `tl.exp` (README, Tolerances). -/
private noncomputable def errEF (D : ℝ) : ℝ := (1 + expRelF D) * Real.exp (D + dErrF D) - Real.exp D + (2:ℝ)^(-140:ℤ)
/-- Real mirror of the computed exponential's magnitude bound (README, Tolerances). -/
private noncomputable def eHatF (D : ℝ) : ℝ := Real.exp D + errEF D
/-- Real mirror of the error of the scale `exp · dt` (README, Tolerances). -/
private noncomputable def errSF (D dv : ℝ) : ℝ := |dv| * errEF D + (2:ℝ)^(-24:ℤ) * (eHatF D * |dv|) + (2:ℝ)^(-150:ℤ)
/-- Real mirror of the scale's magnitude bound (README, Tolerances). -/
private noncomputable def sHatF (D dv : ℝ) : ℝ := |Real.exp D * dv| + errSF D dv
/-- Real mirror of the error of the product with `B` (README, Tolerances). -/
private noncomputable def errPF (D dv bv : ℝ) : ℝ := |bv| * errSF D dv + (2:ℝ)^(-24:ℤ) * (|bv| * sHatF D dv) + (2:ℝ)^(-150:ℤ)
/-- Real mirror of the product's magnitude bound (README, Tolerances). -/
private noncomputable def pHatF (D dv bv : ℝ) : ℝ := |bv * (Real.exp D * dv)| + errPF D dv bv
/-- Real mirror of the error of the `bfloat16` scaled `B` (README, Tolerances). -/
private noncomputable def errBF (D dv bv : ℝ) : ℝ := errPF D dv bv + (2:ℝ)^(-8:ℤ) * pHatF D dv bv + (2:ℝ)^(-134:ℤ)

/-- Constant: relative bound of `tl.exp`'s error, `2^-16`. -/
private noncomputable def cE : ℝ := (2:ℝ)^(-16:ℤ)
/-- Constant: absolute floor of `tl.exp`'s error, `2^-105`. -/
private noncomputable def cAbsE : ℝ := (2:ℝ)^(-105:ℤ)
/-- Constant: `float32` unit roundoff `2^-24`. -/
private noncomputable def uu : ℝ := (2:ℝ)^(-24:ℤ)
/-- Constant: `float32` smallest subnormal `2^-150`. -/
private noncomputable def et : ℝ := (2:ℝ)^(-150:ℤ)
/-- Constant: `1 + cE`. -/
private noncomputable def e1F : ℝ := 1 + cE
/-- Constant: relative bound after the scale `exp · dt`. -/
private noncomputable def c2F : ℝ := cE + uu * e1F
/-- Constant: relative bound after the product with `B`. -/
private noncomputable def c3F : ℝ := c2F + uu * (1 + c2F)
/-- Constant: absolute bound after the product with `B`. -/
private noncomputable def cA3F : ℝ := cAbsE * (1 + uu) ^ 2
/-- Constant: `bfloat16` unit roundoff `2^-8`. -/
private noncomputable def b8 : ℝ := (2:ℝ)^(-8:ℤ)
/-- Constant: `bfloat16` rounding floor `2^-134`. -/
private noncomputable def c134 : ℝ := (2:ℝ)^(-134:ℤ)
/-- Constant: the relative constant of `errB`. -/
private noncomputable def Krel : ℝ := b8 + (1 + b8) * c3F
/-- Constant: first term of the absolute constant of `errB`. -/
private noncomputable def CfA : ℝ := (1 + b8) * cA3F
/-- Constant: second term of the absolute constant of `errB`. -/
private noncomputable def CfBv : ℝ := (1 + b8) * et * (1 + uu)
/-- Constant: third term of the absolute constant of `errB`. -/
private noncomputable def CfB0 : ℝ := (1 + b8) * et + c134
/-- Constant: the absolute constant of `errB`. -/
private noncomputable def Cabs : ℝ := CfA + CfBv + CfB0

private lemma errB_eq (m₀ : Mem) (d : Inst) (c h t n : Nat) :
    errB m₀ d c h t n = errBF (dRef m₀ d c h t) (dtV m₀ d c h t) (bV m₀ d t h n) := rfl

private lemma errE_leF {D : ℝ} (hD : D ≤ 64) : errEF D ≤ cE * Real.exp D + cAbsE := by
  have h := errE_le hD
  simpa only [errEF, expRelF, dErrF, cE, cAbsE] using h

/-- The per-token error `errB` is at most `Krel·|b·e^D·dt| + CfloorA·|b·dt| + CfloorB`. -/
private lemma errB_chain {D dv bv : ℝ} (hD : D ≤ 64) :
    errBF D dv bv ≤ Krel * (|bv| * |Real.exp D * dv|) + CfA * (|bv| * |dv|)
      + CfBv * |bv| + CfB0 := by
  have hE0 : (0:ℝ) ≤ Real.exp D := (Real.exp_pos D).le
  have hA0 : (0:ℝ) ≤ |dv| := abs_nonneg dv
  have hB0 : (0:ℝ) ≤ |bv| := abs_nonneg bv
  have hu0 : (0:ℝ) ≤ uu := by unfold uu; positivity
  have het0 : (0:ℝ) ≤ et := by unfold et; positivity
  have hb80 : (0:ℝ) ≤ b8 := by unfold b8; positivity
  have h0 : errEF D ≤ cE * Real.exp D + cAbsE := errE_leF hD
  have h1 : eHatF D ≤ e1F * Real.exp D + cAbsE := by
    have h := h0
    unfold eHatF e1F
    linarith
  have h2 : errSF D dv ≤ |dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et := by
    have hm0 : |dv| * errEF D ≤ |dv| * (cE * Real.exp D + cAbsE) :=
      mul_le_mul_of_nonneg_left h0 hA0
    have hm1 : eHatF D * |dv| ≤ (e1F * Real.exp D + cAbsE) * |dv| :=
      mul_le_mul_of_nonneg_right h1 hA0
    have hm1' : uu * (eHatF D * |dv|) ≤ uu * ((e1F * Real.exp D + cAbsE) * |dv|) :=
      mul_le_mul_of_nonneg_left hm1 hu0
    have herrSF : errSF D dv = |dv| * errEF D + uu * (eHatF D * |dv|) + et := rfl
    rw [herrSF, show c2F = cE + uu * e1F from rfl]
    nlinarith [hm0, hm1']
  have h3 : sHatF D dv ≤
      |Real.exp D * dv| + (|dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et) := by
    have hs : sHatF D dv = |Real.exp D * dv| + errSF D dv := rfl
    rw [hs]
    linarith [h2]
  have h4 : errPF D dv bv ≤
      |bv| * (|dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et)
      + uu * (|bv| * (|Real.exp D * dv|
          + (|dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et))) + et := by
    have hm2 : |bv| * errSF D dv ≤
        |bv| * (|dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et) :=
      mul_le_mul_of_nonneg_left h2 hB0
    have hm3 : |bv| * sHatF D dv ≤
        |bv| * (|Real.exp D * dv| + (|dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et)) :=
      mul_le_mul_of_nonneg_left h3 hB0
    have hm3' : uu * (|bv| * sHatF D dv) ≤
        uu * (|bv| * (|Real.exp D * dv|
          + (|dv| * Real.exp D * c2F + |dv| * cAbsE * (1 + uu) + et))) :=
      mul_le_mul_of_nonneg_left hm3 hu0
    have hp : errPF D dv bv = |bv| * errSF D dv + uu * (|bv| * sHatF D dv) + et := rfl
    rw [hp]
    linarith [hm2, hm3']
  have h4' : errPF D dv bv ≤
      |bv| * |dv| * Real.exp D * c3F + |bv| * |dv| * cA3F + |bv| * et * (1 + uu) + et := by
    have hEA : |Real.exp D * dv| = Real.exp D * |dv| := by
      rw [abs_mul, abs_of_nonneg hE0]
    have h := h4
    rw [hEA] at h
    rw [show c3F = c2F + uu * (1 + c2F) from rfl,
        show cA3F = cAbsE * (1 + uu) ^ 2 from rfl]
    nlinarith [h]
  have h5 : pHatF D dv bv ≤
      |bv * (Real.exp D * dv)|
      + (|bv| * |dv| * Real.exp D * c3F + |bv| * |dv| * cA3F + |bv| * et * (1 + uu) + et) := by
    have hp : pHatF D dv bv = |bv * (Real.exp D * dv)| + errPF D dv bv := rfl
    rw [hp]
    linarith [h4']
  have h6 : errBF D dv bv ≤ Krel * (|bv| * |Real.exp D * dv|) + CfA * (|bv| * |dv|)
      + CfBv * |bv| + CfB0 := by
    have hEA : |Real.exp D * dv| = Real.exp D * |dv| := by
      rw [abs_mul, abs_of_nonneg hE0]
    have hbv : |bv * (Real.exp D * dv)| = |bv| * |Real.exp D * dv| := abs_mul bv (Real.exp D * dv)
    have hm5 : b8 * pHatF D dv bv ≤
        b8 * (|bv * (Real.exp D * dv)|
          + (|bv| * |dv| * Real.exp D * c3F + |bv| * |dv| * cA3F + |bv| * et * (1 + uu) + et)) :=
      mul_le_mul_of_nonneg_left h5 hb80
    rw [hbv] at hm5
    have hstep : errBF D dv bv ≤
        b8 * (|bv| * |Real.exp D * dv|)
        + (1 + b8) * (|bv| * |dv| * Real.exp D * c3F + |bv| * |dv| * cA3F
            + |bv| * et * (1 + uu) + et) + c134 := by
      have he : errBF D dv bv = errPF D dv bv + b8 * pHatF D dv bv + c134 := rfl
      rw [he]
      nlinarith [h4', hm5]
    rw [hEA] at hstep ⊢
    rw [show Krel = b8 + (1 + b8) * c3F from rfl,
        show CfA = (1 + b8) * cA3F from rfl,
        show CfBv = (1 + b8) * et * (1 + uu) from rfl,
        show CfB0 = (1 + b8) * et + c134 from rfl]
    nlinarith [hstep]
  exact h6
private lemma rho_nonneg : 0 ≤ rho := by
  unfold rho DotLoop.stepRel Device.relDot16 BK
  rw [Format.binary32_u]
  norm_num

private lemma alp_nonneg : 0 ≤ alp := by
  unfold alp DotLoop.stepAbs Device.absDot16 BK
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

private lemma rho_le : rho ≤ 2 ^ (-15 : ℤ) := by
  unfold rho DotLoop.stepRel Device.relDot16 BK
  rw [Format.binary32_u]
  norm_num

private lemma condA : Krel + 2 * (rho : ℝ) ≤ (2:ℝ)^(-8:ℤ) + (2:ℝ)^(-14:ℤ) := by
  unfold Krel c3F c2F e1F cE uu b8
  unfold rho DotLoop.stepRel Device.relDot16 BK
  rw [Format.binary32_u]
  push_cast
  norm_num

private lemma condB : Cabs ≤ (2:ℝ)^(-101:ℤ) := by
  unfold Cabs CfA CfBv CfB0 cA3F cAbsE uu et b8 c134
  norm_num

private lemma condC : (2:ℝ)^11 * (alp : ℝ) ≤ (2:ℝ)^(-101:ℤ) := by
  unfold alp DotLoop.stepAbs Device.absDot16 BK
  rw [Format.binary32_u, Format.binary32_eta]
  push_cast
  norm_num



private lemma nbChunkPos {d : Inst} (hd : d.ok) : 0 < d.chunk := by
  obtain ⟨k, hk⟩ := hd.chunk_pow2
  rw [hk]
  exact Nat.pow_pos (by norm_num : (0:ℕ) < 2)

private lemma nbCmulLt {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    c * d.chunk < d.seqlen := by
  have hch : 0 < d.chunk := nbChunkPos hd
  have h := (Nat.lt_div_iff_mul_lt hch).mp (by simpa [nc] using hc)
  have he : (d.seqlen + d.chunk - 1) - (d.chunk - 1) = d.seqlen := by omega
  omega

private lemma nbEndTokGe {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    c * d.chunk ≤ endTok d c := by
  have hch : 0 < d.chunk := nbChunkPos hd
  have hcm : c * d.chunk < d.seqlen := nbCmulLt hd hc
  unfold endTok
  have h1 : c * d.chunk + 1 ≤ (c + 1) * d.chunk := by
    have : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
    omega
  have h2 : c * d.chunk + 1 ≤ d.seqlen := hcm
  have h3 : c * d.chunk + 1 ≤ min ((c + 1) * d.chunk) d.seqlen := le_min h1 h2
  exact Nat.le_pred_of_lt (lt_of_lt_of_le (Nat.lt_succ_self _) h3)

private lemma nbEndTokMem {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    endTok d c ∈ toks d c := by
  have hge := nbEndTokGe hd hc
  unfold toks
  rw [Finset.mem_filter, Finset.mem_Icc]
  refine ⟨⟨hge, ?_⟩, rfl⟩
  exact le_rfl

private lemma nbEndTokLe {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    endTok d c + 1 ≤ (c + 1) * d.chunk := by
  have hch : 0 < d.chunk := nbChunkPos hd
  have hcm : c * d.chunk < d.seqlen := nbCmulLt hd hc
  unfold endTok
  have h1 : c * d.chunk + 1 ≤ (c + 1) * d.chunk := by
    have : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
    omega
  have h3 : c * d.chunk + 1 ≤ min ((c + 1) * d.chunk) d.seqlen :=
    le_min h1 hcm
  have hmin : min ((c + 1) * d.chunk) d.seqlen ≤ (c + 1) * d.chunk := min_le_left _ _
  have hpos : 0 < min ((c + 1) * d.chunk) d.seqlen := by omega
  have hEq : min ((c + 1) * d.chunk) d.seqlen - 1 + 1 = min ((c + 1) * d.chunk) d.seqlen := by
    omega
  rw [hEq]
  exact hmin

private lemma nbStepsGe {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    1 ≤ steps d c := by
  have hge := nbEndTokGe hd hc
  unfold steps BK
  rw [Nat.one_le_div_iff (by norm_num : 0 < 64)]
  omega

private lemma nbStepsLe {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    steps d c ≤ 1024 := by
  have hch : 0 < d.chunk := nbChunkPos hd
  have hL : endTok d c + 1 - c * d.chunk ≤ d.chunk := by
    have h1 : endTok d c + 1 ≤ (c + 1) * d.chunk := nbEndTokLe hd hc
    have h2 : endTok d c + 1 - c * d.chunk ≤ (c + 1) * d.chunk - c * d.chunk :=
      Nat.sub_le_sub_right h1 _
    have h3 : (c + 1) * d.chunk - c * d.chunk = d.chunk := by
      have : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
      omega
    omega
  have hcle : d.chunk ≤ 2 ^ 16 := hd.chunk_le
  unfold steps BK
  have h2 : (endTok d c + 1 - c * d.chunk + 64 - 1) / 64 ≤ (2 ^ 16 + 64 - 1) / 64 :=
    Nat.div_le_div_right (by omega)
  have h3 : (2 ^ 16 + 64 - 1) / 64 = 1024 := by decide
  omega



private lemma nbMagB_nonneg (q : ℚ) : 0 ≤ magB q := by
  unfold magB
  exact_mod_cast Device.liftMag_nonneg .bf16 q

private lemma nbErrE_nonneg {m₀ : Mem} {d : Inst} {c h t : Nat} : 0 ≤ errE m₀ d c h t := by
  unfold errE expRel dErr
  set D := dRef m₀ d c h t with hD
  have hrel : 0 ≤ (2:ℝ) ^ (-22:ℤ) + (2:ℝ) ^ (-23:ℤ) * (|D| + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))) := by
    positivity
  have hexp : Real.exp D ≤ Real.exp (D + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))) := by
    apply Real.exp_le_exp.mpr
    have h1 : (0:ℝ) ≤ |D| * (2:ℝ) ^ (-24:ℤ) := by positivity
    have h2 : (0:ℝ) ≤ (2:ℝ) ^ (-150:ℤ) := by positivity
    linarith
  have hmul : (1 + 0) * Real.exp D
      ≤ (1 + ((2:ℝ) ^ (-22:ℤ) + (2:ℝ) ^ (-23:ℤ) * (|D| + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ)))))
        * Real.exp (D + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))) := by
    apply mul_le_mul (by linarith) hexp (Real.exp_pos D).le (by linarith)
  have h140 : (0:ℝ) ≤ (2:ℝ) ^ (-140:ℤ) := by positivity
  linarith

private lemma nbErrS_nonneg {m₀ : Mem} {d : Inst} {c h t : Nat} : 0 ≤ errS m₀ d c h t := by
  have hE : 0 ≤ errE m₀ d c h t := nbErrE_nonneg
  have hdt : 0 ≤ |dtV m₀ d c h t| := abs_nonneg _
  have hH : 0 ≤ eHat m₀ d c h t := by
    have := Real.exp_pos (dRef m₀ d c h t)
    unfold eHat
    linarith
  have h1 : 0 ≤ |dtV m₀ d c h t| * errE m₀ d c h t := mul_nonneg hdt hE
  have h2 : 0 ≤ (2:ℝ) ^ (-24:ℤ) * (eHat m₀ d c h t * |dtV m₀ d c h t|) :=
    mul_nonneg (by positivity) (mul_nonneg hH hdt)
  have h3 : (0:ℝ) ≤ (2:ℝ) ^ (-150:ℤ) := by positivity
  unfold errS
  linarith

private lemma nbSHat_nonneg {m₀ : Mem} {d : Inst} {c h t : Nat} : 0 ≤ sHat m₀ d c h t := by
  have h1 : 0 ≤ errS m₀ d c h t := nbErrS_nonneg
  have h2 : 0 ≤ |wRef m₀ d c h t| := abs_nonneg _
  unfold sHat
  linarith

private lemma nbErrP_nonneg {m₀ : Mem} {d : Inst} {c h t n : Nat} : 0 ≤ errP m₀ d c h t n := by
  have hE : 0 ≤ errS m₀ d c h t := nbErrS_nonneg
  have hH : 0 ≤ sHat m₀ d c h t := nbSHat_nonneg
  have hb : 0 ≤ |bV m₀ d t h n| := abs_nonneg _
  have h1 : 0 ≤ |bV m₀ d t h n| * errS m₀ d c h t := mul_nonneg hb hE
  have h2 : 0 ≤ (2:ℝ) ^ (-24:ℤ) * (|bV m₀ d t h n| * sHat m₀ d c h t) :=
    mul_nonneg (by positivity) (mul_nonneg hb hH)
  have h3 : (0:ℝ) ≤ (2:ℝ) ^ (-150:ℤ) := by positivity
  unfold errP
  linarith

private lemma nbPHat_nonneg {m₀ : Mem} {d : Inst} {c h t n : Nat} : 0 ≤ pHat m₀ d c h t n := by
  have h1 : 0 ≤ errP m₀ d c h t n := nbErrP_nonneg
  have h2 : 0 ≤ |bV m₀ d t h n * wRef m₀ d c h t| := abs_nonneg _
  unfold pHat
  linarith

private lemma nbErrB_nonneg {m₀ : Mem} {d : Inst} {c h t n : Nat} : 0 ≤ errB m₀ d c h t n := by
  have h1 : 0 ≤ errP m₀ d c h t n := nbErrP_nonneg
  have h2 : 0 ≤ pHat m₀ d c h t n := nbPHat_nonneg
  have h3 : (0:ℝ) ≤ (2:ℝ) ^ (-134:ℤ) := by positivity
  unfold errB
  have h4 : 0 ≤ (2:ℝ) ^ (-8:ℤ) * pHat m₀ d c h t n := mul_nonneg (by positivity) h2
  linarith

private lemma nbInMag_ge_one {m₀ : Mem} {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hc : c < nc d) : 1 ≤ inMag m₀ d c h p n := by
  have hmem : endTok d c ∈ toks d c := nbEndTokMem hd hc
  have hterm : ∀ t ∈ toks d c,
      0 ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|) := by
    intro t _
    positivity
  have hsingle := Finset.single_le_sum hterm hmem
  have h1 : (1:ℝ) ≤ (1 + |xV m₀ d (endTok d c) h p|)
      * (1 + |bV m₀ d (endTok d c) h n|) * (1 + |dtV m₀ d c h (endTok d c)|) := by
    have ha := abs_nonneg (xV m₀ d (endTok d c) h p)
    have hb := abs_nonneg (bV m₀ d (endTok d c) h n)
    have hcc := abs_nonneg (dtV m₀ d c h (endTok d c))
    have hab : 0 ≤ |xV m₀ d (endTok d c) h p| * |bV m₀ d (endTok d c) h n| := mul_nonneg ha hb
    have hbc : 0 ≤ |bV m₀ d (endTok d c) h n| * |dtV m₀ d c h (endTok d c)| := mul_nonneg hb hcc
    have hac : 0 ≤ |xV m₀ d (endTok d c) h p| * |dtV m₀ d c h (endTok d c)| := mul_nonneg ha hcc
    have habc : 0 ≤ |xV m₀ d (endTok d c) h p| * |bV m₀ d (endTok d c) h n|
        * |dtV m₀ d c h (endTok d c)| := mul_nonneg hab hcc
    nlinarith [hab, hbc, hac, habc]
  unfold inMag
  linarith

private lemma nbX_le_term {m₀ : Mem} {d : Inst} {c h p n t : Nat} :
    |xV m₀ d t h p| ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|) := by
  have ha := abs_nonneg (xV m₀ d t h p)
  have hb := abs_nonneg (bV m₀ d t h n)
  have hcc := abs_nonneg (dtV m₀ d c h t)
  have hb1 : (1:ℝ) ≤ 1 + |bV m₀ d t h n| := by linarith
  have hc1 : (1:ℝ) ≤ 1 + |dtV m₀ d c h t| := by linarith
  have hmul : (1 + |xV m₀ d t h p|) * 1 * 1
      ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|) := by
    apply mul_le_mul
    · exact mul_le_mul (le_refl _) hb1 (by norm_num) (by positivity)
    · exact hc1
    · norm_num
    · positivity
  nlinarith [hmul]

private lemma nbXB_le_term {m₀ : Mem} {d : Inst} {c h p n t : Nat} :
    |xV m₀ d t h p| * |bV m₀ d t h n|
      ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|) := by
  have ha := abs_nonneg (xV m₀ d t h p)
  have hb := abs_nonneg (bV m₀ d t h n)
  have hcc := abs_nonneg (dtV m₀ d c h t)
  have hc1 : (1:ℝ) ≤ 1 + |dtV m₀ d c h t| := by linarith
  have h2 : |xV m₀ d t h p| * |bV m₀ d t h n|
      ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) := by nlinarith
  have hmul : (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * 1
      ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|) := by
    apply mul_le_mul (le_refl _) hc1 (by norm_num) (by positivity)
  nlinarith [h2, hmul]

private lemma nbXBD_le_term {m₀ : Mem} {d : Inst} {c h p n t : Nat} :
    |xV m₀ d t h p| * |bV m₀ d t h n| * |dtV m₀ d c h t|
      ≤ (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|) := by
  have ha := abs_nonneg (xV m₀ d t h p)
  have hb := abs_nonneg (bV m₀ d t h n)
  have hcc := abs_nonneg (dtV m₀ d c h t)
  nlinarith

private lemma nbKrel_nonneg : 0 ≤ Krel := by
  unfold Krel c3F c2F e1F cE uu b8
  positivity

private lemma nbCfA_nonneg : 0 ≤ CfA := by
  unfold CfA cA3F cAbsE uu b8
  positivity

private lemma nbCfBv_nonneg : 0 ≤ CfBv := by
  unfold CfBv et uu b8
  positivity

private lemma nbCfB0_nonneg : 0 ≤ CfB0 := by
  unfold CfB0 et c134 b8
  positivity

private lemma nbPerToken {m₀ : Mem} {d : Inst} (_hd : d.ok) {c h p n : Nat}
    (hc : c < nc d) (hh : h < d.nheads) (_hp : p < d.headdim) (_hn : n < d.dstate)
    (hV : Vals m₀ d) {bh : Nat → ℚ} {t : Nat} (ht : t ∈ toks d c)
    (_hbt : |(bh t : ℝ) - bV m₀ d t h n * wRef m₀ d c h t| ≤ errB m₀ d c h t n) :
    |xV m₀ d t h p| * errB m₀ d c h t n ≤
      Krel * (magB (valB m₀ d.x (xIdx d t h p)) * max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ)))
        + Cabs * ((1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|)) := by
  have hD : dRef m₀ d c h t ≤ 64 := (hV.2.2.2.1 c h t hc hh ht).1
  have hchain : errB m₀ d c h t n ≤
      Krel * |bV m₀ d t h n * wRef m₀ d c h t| + CfA * (|bV m₀ d t h n| * |dtV m₀ d c h t|)
        + CfBv * |bV m₀ d t h n| + CfB0 := by
    rw [errB_eq]
    have hch := errB_chain (D := dRef m₀ d c h t) (dv := dtV m₀ d c h t) (bv := bV m₀ d t h n) hD
    have hW : |bV m₀ d t h n| * |Real.exp (dRef m₀ d c h t) * dtV m₀ d c h t|
        = |bV m₀ d t h n * wRef m₀ d c h t| := by
      unfold wRef
      simp only [abs_mul]
    rw [hW] at hch
    exact hch
  have hX0 : (0:ℝ) ≤ |xV m₀ d t h p| := abs_nonneg _
  have herrB0 : (0:ℝ) ≤ errB m₀ d c h t n := nbErrB_nonneg
  have hK0 : (0:ℝ) ≤ Krel := nbKrel_nonneg
  have hA0 : (0:ℝ) ≤ CfA := nbCfA_nonneg
  have hBv0 : (0:ℝ) ≤ CfBv := nbCfBv_nonneg
  have hB00 : (0:ℝ) ≤ CfB0 := nbCfB0_nonneg
  have hQ0 : (0:ℝ) ≤ magB (valB m₀ d.x (xIdx d t h p)) := nbMagB_nonneg _
  have hW0 : (0:ℝ) ≤ |bV m₀ d t h n * wRef m₀ d c h t| := abs_nonneg _
  have hBH0 : (0:ℝ) ≤ bHat m₀ d c h t n := by
    unfold bHat
    linarith
  have hM0 : (0:ℝ) ≤ max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ)) :=
    le_trans hBH0 (le_max_left _ _)
  have hXQ : |xV m₀ d t h p| ≤ magB (valB m₀ d.x (xIdx d t h p)) := by
    have hlift := Device.abs_le_liftMag .bf16 (valB m₀ d.x (xIdx d t h p))
    have h' : ((|valB m₀ d.x (xIdx d t h p)| : ℚ) : ℝ) ≤ magB (valB m₀ d.x (xIdx d t h p)) := by
      unfold magB
      exact_mod_cast hlift
    have e : |xV m₀ d t h p| = ((|valB m₀ d.x (xIdx d t h p)| : ℚ) : ℝ) := by
      unfold xV
      rw [← Rat.cast_abs]
    rwa [e]
  have hWM : |bV m₀ d t h n * wRef m₀ d c h t| ≤ max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ)) := by
    have h1 : |bV m₀ d t h n * wRef m₀ d c h t| ≤ bHat m₀ d c h t n := by
      unfold bHat
      linarith
    exact le_trans h1 (le_max_left _ _)
  have hprod : |xV m₀ d t h p| * |bV m₀ d t h n * wRef m₀ d c h t|
      ≤ magB (valB m₀ d.x (xIdx d t h p)) * max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ)) :=
    mul_le_mul hXQ hWM hW0 hQ0
  have e1 : |xV m₀ d t h p| * (Krel * |bV m₀ d t h n * wRef m₀ d c h t|)
      ≤ Krel * (magB (valB m₀ d.x (xIdx d t h p)) * max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ))) := by
    have h := mul_le_mul_of_nonneg_left hprod hK0
    nlinarith [h]
  have e2 : |xV m₀ d t h p| * (CfA * (|bV m₀ d t h n| * |dtV m₀ d c h t|))
      ≤ CfA * ((1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|)) := by
    have h := nbXBD_le_term (m₀ := m₀) (d := d) (c := c) (h := h) (p := p) (n := n) (t := t)
    have h2 := mul_le_mul_of_nonneg_left h hA0
    nlinarith [h2]
  have e3 : |xV m₀ d t h p| * (CfBv * |bV m₀ d t h n|)
      ≤ CfBv * ((1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|)) := by
    have h := nbXB_le_term (m₀ := m₀) (d := d) (c := c) (h := h) (p := p) (n := n) (t := t)
    have h2 := mul_le_mul_of_nonneg_left h hBv0
    nlinarith [h2]
  have e4 : |xV m₀ d t h p| * CfB0
      ≤ CfB0 * ((1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|)) := by
    have h := nbX_le_term (m₀ := m₀) (d := d) (c := c) (h := h) (p := p) (n := n) (t := t)
    have h2 := mul_le_mul_of_nonneg_left h hB00
    nlinarith [h2]
  have hmul := mul_le_mul_of_nonneg_left hchain hX0
  unfold Cabs
  nlinarith [hmul, e1, e2, e3, e4]

private lemma nbSumBound {m₀ : Mem} {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hc : c < nc d) (hh : h < d.nheads) (hp : p < d.headdim) (hn : n < d.dstate)
    (hV : Vals m₀ d) {bh : Nat → ℚ}
    (hb : ∀ t ∈ toks d c, |(bh t : ℝ) - bV m₀ d t h n * wRef m₀ d c h t| ≤ errB m₀ d c h t n) :
    |(∑ t ∈ toks d c, xV m₀ d t h p * (bh t : ℝ)) - sRef m₀ d c h p n|
      ≤ Krel * dotMag m₀ d c h p n + Cabs * inMag m₀ d c h p n := by
  have hsub : (∑ t ∈ toks d c, xV m₀ d t h p * (bh t : ℝ)) - sRef m₀ d c h p n
      = ∑ t ∈ toks d c, xV m₀ d t h p * ((bh t : ℝ) - bV m₀ d t h n * wRef m₀ d c h t) := by
    unfold sRef
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro t _
    ring
  rw [hsub]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  have hstep : ∀ t ∈ toks d c,
      |xV m₀ d t h p * ((bh t : ℝ) - bV m₀ d t h n * wRef m₀ d c h t)|
      ≤ Krel * (magB (valB m₀ d.x (xIdx d t h p)) * max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ)))
        + Cabs * ((1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|)) := by
    intro t ht
    rw [abs_mul]
    refine le_trans (mul_le_mul_of_nonneg_left (hb t ht) (abs_nonneg _)) ?_
    exact nbPerToken hd hc hh hp hn hV ht (hb t ht)
  refine le_trans (Finset.sum_le_sum hstep) ?_
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  unfold dotMag inMag
  rfl

private lemma nbBoundCast (T : ℕ) (B : ℚ) :
    ((DotLoop.bound rho alp T B : ℚ) : ℝ)
      = ((1 + (rho:ℝ)) ^ T - 1) * (B:ℝ) + (T:ℝ) * (1 + (rho:ℝ)) ^ T * (alp:ℝ) := by
  unfold DotLoop.bound
  push_cast
  ring

private lemma nbPowLe {T : ℕ} (hT : T ≤ 1024) :
    (1 + (rho:ℝ)) ^ T ≤ 1 + 2 * (T:ℝ) * (rho:ℝ) := by
  have hρ : 0 ≤ rho := rho_nonneg
  have hTR : (T:ℚ) * rho ≤ 1 / 2 := by
    have h1 : (T:ℚ) ≤ 1024 := by exact_mod_cast hT
    have h2 : rho ≤ (2:ℚ) ^ (-15:ℤ) := rho_le
    have h3 : (T:ℚ) * rho ≤ 1024 * (2:ℚ) ^ (-15:ℤ) :=
      mul_le_mul h1 h2 hρ (by norm_num)
    have h4 : (1024:ℚ) * (2:ℚ) ^ (-15:ℤ) ≤ 1 / 2 := by norm_num
    exact le_trans h3 h4
  have h := DotLoop.pow_le hρ T hTR
  have h' : ((1 + rho) ^ T : ℚ) ≤ ((1 + 2 * (T:ℚ) * rho : ℚ)) := h
  exact_mod_cast h'

private lemma nbTolBound {T : ℕ} {dot inM bnd : ℝ}
    (hT1 : (1:ℝ) ≤ (T:ℝ)) (hT1024 : (T:ℝ) ≤ 1024)
    (hd0 : 0 ≤ dot) (hin1 : 1 ≤ inM)
    (hbnd : bnd ≤ 2 * (T:ℝ) * (rho:ℝ) * dot + 2 * (T:ℝ) * (alp:ℝ)) :
    bnd + (Krel * dot + Cabs * inM)
      ≤ 2 ^ (-8:ℤ) * (1 + (T:ℝ) * 2 ^ (-6:ℤ)) * dot + 2 ^ (-100:ℤ) * inM := by
  have hρ0 : (0:ℝ) ≤ (rho:ℝ) := by exact_mod_cast rho_nonneg
  have hα0 : (0:ℝ) ≤ (alp:ℝ) := by exact_mod_cast alp_nonneg
  have hT0 : (0:ℝ) ≤ (T:ℝ) := by linarith
  have hcoef : Krel + 2 * (T:ℝ) * (rho:ℝ) ≤ 2 ^ (-8:ℤ) + (T:ℝ) * 2 ^ (-14:ℤ) := by
    have h2ρ : 2 * (rho:ℝ) ≤ (2:ℝ) ^ (-14:ℤ) := by
      have hρleR : (rho:ℝ) ≤ (2:ℝ) ^ (-15:ℤ) := by
        have h := rho_le
        have hc : ((2:ℚ)^(-15:ℤ) : ℝ) = (2:ℝ)^(-15:ℤ) := by push_cast; norm_num
        calc (rho:ℝ) ≤ (((2:ℚ)^(-15:ℤ)) : ℝ) := by exact_mod_cast h
          _ = (2:ℝ)^(-15:ℤ) := hc
      have h5 : (2:ℝ) * (2:ℝ)^(-15:ℤ) = (2:ℝ)^(-14:ℤ) := by norm_num
      linarith
    have hTm1 : (0:ℝ) ≤ (T:ℝ) - 1 := by linarith
    have hstep : 2 * ((T:ℝ) - 1) * (rho:ℝ) ≤ ((T:ℝ) - 1) * (2:ℝ) ^ (-14:ℤ) := by
      have h := mul_le_mul_of_nonneg_left h2ρ hTm1
      nlinarith [h]
    have hK : Krel + 2 * (rho:ℝ) ≤ (2:ℝ) ^ (-8:ℤ) + (2:ℝ) ^ (-14:ℤ) := condA
    nlinarith [hstep, hK]
  have hcoefdot : (Krel + 2 * (T:ℝ) * (rho:ℝ)) * dot
      ≤ (2 ^ (-8:ℤ) + (T:ℝ) * 2 ^ (-14:ℤ)) * dot :=
    mul_le_mul_of_nonneg_right hcoef hd0
  have hCabs : Cabs ≤ (2:ℝ) ^ (-101:ℤ) := condB
  have hα2 : 2 * (T:ℝ) * (alp:ℝ) ≤ (2:ℝ) ^ (-101:ℤ) := by
    have h1 : 2 * (T:ℝ) * (alp:ℝ) ≤ (2:ℝ) ^ 11 * (alp:ℝ) := by
      have h2 : 2 * (T:ℝ) ≤ (2:ℝ) ^ 11 := by
        have : (2:ℝ) ^ 11 = 2048 := by norm_num
        linarith
      exact mul_le_mul_of_nonneg_right h2 hα0
    have h3 : (2:ℝ) ^ 11 * (alp:ℝ) ≤ (2:ℝ) ^ (-101:ℤ) := condC
    linarith
  have hsum : Cabs + 2 * (T:ℝ) * (alp:ℝ) ≤ (2:ℝ) ^ (-100:ℤ) := by
    have h : (2:ℝ) ^ (-101:ℤ) + (2:ℝ) ^ (-101:ℤ) = (2:ℝ) ^ (-100:ℤ) := by norm_num
    linarith
  have h2Tα0 : 0 ≤ 2 * (T:ℝ) * (alp:ℝ) := by positivity
  have hin0 : 0 ≤ inM := by linarith
  have hIm : Cabs * inM + 2 * (T:ℝ) * (alp:ℝ) ≤ (2:ℝ) ^ (-100:ℤ) * inM := by
    have h1 : Cabs * inM ≤ ((2:ℝ) ^ (-100:ℤ) - 2 * (T:ℝ) * (alp:ℝ)) * inM :=
      mul_le_mul_of_nonneg_right (by linarith) hin0
    have h2 : 0 ≤ 2 * (T:ℝ) * (alp:ℝ) * (inM - 1) :=
      mul_nonneg h2Tα0 (by linarith)
    nlinarith [h1, h2]
  have hA : (2:ℝ)^(-8:ℤ) * (1 + (T:ℝ) * (2:ℝ)^(-6:ℤ)) * dot + (2:ℝ)^(-100:ℤ) * inM
      = ((2:ℝ)^(-8:ℤ) + (T:ℝ) * (2:ℝ)^(-14:ℤ)) * dot + (2:ℝ)^(-100:ℤ) * inM := by
    have h1 : (2:ℝ)^(-8:ℤ) * (2:ℝ)^(-6:ℤ) = (2:ℝ)^(-14:ℤ) := by norm_num
    rw [show (2:ℝ)^(-8:ℤ) * (1 + (T:ℝ) * (2:ℝ)^(-6:ℤ))
        = (2:ℝ)^(-8:ℤ) + (T:ℝ) * ((2:ℝ)^(-8:ℤ) * (2:ℝ)^(-6:ℤ)) from by ring, h1]
  rw [hA]
  linarith [hbnd, hcoefdot, hIm]

private lemma nbDotMag_nonneg {m₀ : Mem} {d : Inst} {c h p n : Nat} :
    0 ≤ dotMag m₀ d c h p n := by
  unfold dotMag
  apply Finset.sum_nonneg
  intro t _
  apply mul_nonneg (nbMagB_nonneg _)
  exact le_trans (by positivity) (le_max_right _ _)

/-- **The tolerance.** From the per-token error of the scaled `B`, the sums of the loop and the accumulator's bound after all steps, the accumulator is within `tol` of `sRef`. -/
theorem value_bound {d : Inst} (hd : d.ok) {m₀ : Mem} (hV : Vals m₀ d) {c h p n : Nat}
    (hc : c < nc d) (hh : h < d.nheads) (hp : p < d.headdim) (hn : n < d.dstate)
    {aT : ℝ} (bh : Nat → ℚ) (S A : ℕ → ℚ)
    (hS : ((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ)
      = ∑ t ∈ toks d c, xV m₀ d t h p * (bh t : ℝ))
    (hA : ((∑ t ∈ Finset.range (steps d c), A t : ℚ) : ℝ) ≤ dotMag m₀ d c h p n)
    (hb : ∀ t ∈ toks d c, |(bh t : ℝ) - bV m₀ d t h n * wRef m₀ d c h t| ≤ errB m₀ d c h t n)
    (hacc : |aT - ((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ)|
      ≤ ((DotLoop.bound rho alp (steps d c) (∑ t ∈ Finset.range (steps d c), A t) : ℚ) : ℝ)) :
    |aT - sRef m₀ d c h p n| ≤ tol m₀ d c h p n := by
  have hTge : 1 ≤ steps d c := nbStepsGe hd hc
  have hTle : steps d c ≤ 1024 := nbStepsLe hd hc
  have hρ0 : (0:ℝ) ≤ (rho:ℝ) := by exact_mod_cast rho_nonneg
  have hα0 : (0:ℝ) ≤ (alp:ℝ) := by exact_mod_cast alp_nonneg
  have hpow : (1 + (rho:ℝ)) ^ (steps d c) ≤ 1 + 2 * ((steps d c : ℕ) : ℝ) * (rho:ℝ) :=
    nbPowLe hTle
  have hρleR : (rho:ℝ) ≤ (2:ℝ) ^ (-15:ℤ) := by
    have h := rho_le
    have hc : ((2:ℚ)^(-15:ℤ) : ℝ) = (2:ℝ)^(-15:ℤ) := by push_cast; norm_num
    calc (rho:ℝ) ≤ (((2:ℚ)^(-15:ℤ)) : ℝ) := by exact_mod_cast h
      _ = (2:ℝ)^(-15:ℤ) := hc
  have hTρ : ((steps d c : ℕ) : ℝ) * (rho:ℝ) ≤ 1 / 2 := by
    have h1 : ((steps d c : ℕ) : ℝ) ≤ 1024 := by exact_mod_cast hTle
    have h3 : ((steps d c : ℕ) : ℝ) * (rho:ℝ) ≤ 1024 * (2:ℝ) ^ (-15:ℤ) :=
      mul_le_mul h1 hρleR hρ0 (by norm_num)
    have h4 : (1024:ℝ) * (2:ℝ) ^ (-15:ℤ) ≤ 1 / 2 := by norm_num
    exact le_trans h3 h4
  have hpow2 : (1 + (rho:ℝ)) ^ (steps d c) ≤ 2 := by nlinarith [hpow, hTρ]
  have hcast : ((DotLoop.bound rho alp (steps d c) (∑ t ∈ Finset.range (steps d c), A t) : ℚ) : ℝ)
      = ((1 + (rho:ℝ)) ^ (steps d c) - 1) * (((∑ t ∈ Finset.range (steps d c), A t) : ℚ) : ℝ)
        + ((steps d c : ℕ) : ℝ) * (1 + (rho:ℝ)) ^ (steps d c) * (alp:ℝ) :=
    nbBoundCast (steps d c) (∑ t ∈ Finset.range (steps d c), A t)
  have hfac : 0 ≤ (1 + (rho:ℝ)) ^ (steps d c) - 1 := by
    have h1 : (1:ℝ) ≤ (1 + (rho:ℝ)) ^ (steps d c) := by
      apply one_le_pow₀
      linarith
    linarith
  have hdot0 : 0 ≤ dotMag m₀ d c h p n := nbDotMag_nonneg
  have hb1 : ((1 + (rho:ℝ)) ^ (steps d c) - 1) * (((∑ t ∈ Finset.range (steps d c), A t) : ℚ) : ℝ)
      ≤ ((1 + (rho:ℝ)) ^ (steps d c) - 1) * dotMag m₀ d c h p n :=
    mul_le_mul_of_nonneg_left hA hfac
  have hb2 : ((1 + (rho:ℝ)) ^ (steps d c) - 1) * dotMag m₀ d c h p n
      ≤ 2 * ((steps d c : ℕ) : ℝ) * (rho:ℝ) * dotMag m₀ d c h p n := by
    have h1 : (1 + (rho:ℝ)) ^ (steps d c) - 1 ≤ 2 * ((steps d c : ℕ) : ℝ) * (rho:ℝ) := by
      linarith [hpow]
    exact mul_le_mul_of_nonneg_right h1 hdot0
  have hb3 : ((steps d c : ℕ) : ℝ) * (1 + (rho:ℝ)) ^ (steps d c) * (alp:ℝ)
      ≤ 2 * ((steps d c : ℕ) : ℝ) * (alp:ℝ) := by
    have hT0 : (0:ℝ) ≤ ((steps d c : ℕ) : ℝ) := by positivity
    have h1 : ((steps d c : ℕ) : ℝ) * (1 + (rho:ℝ)) ^ (steps d c) ≤ ((steps d c : ℕ) : ℝ) * 2 :=
      mul_le_mul_of_nonneg_left hpow2 hT0
    nlinarith [h1, hα0]
  have hbnd : ((DotLoop.bound rho alp (steps d c) (∑ t ∈ Finset.range (steps d c), A t) : ℚ) : ℝ)
      ≤ 2 * ((steps d c : ℕ) : ℝ) * (rho:ℝ) * dotMag m₀ d c h p n
        + 2 * ((steps d c : ℕ) : ℝ) * (alp:ℝ) := by
    rw [hcast]
    linarith [hb1, hb2, hb3]
  have htol := nbTolBound (T := steps d c) (dot := dotMag m₀ d c h p n)
    (inM := inMag m₀ d c h p n)
    (bnd := ((DotLoop.bound rho alp (steps d c) (∑ t ∈ Finset.range (steps d c), A t) : ℚ) : ℝ))
    (by exact_mod_cast hTge) (by exact_mod_cast hTle) hdot0 (nbInMag_ge_one hd hc) hbnd
  have hsum := nbSumBound hd hc hh hp hn hV hb
  have hsplit : aT - sRef m₀ d c h p n
      = (aT - ((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ))
        + (((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ) - sRef m₀ d c h p n) := by ring
  have h2 : |((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ) - sRef m₀ d c h p n|
      ≤ Krel * dotMag m₀ d c h p n + Cabs * inMag m₀ d c h p n := by
    rw [hS]
    exact hsum
  have htri := abs_add_le (aT - ((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ))
    (((∑ t ∈ Finset.range (steps d c), S t : ℚ) : ℝ) - sRef m₀ d c h p n)
  rw [hsplit]
  unfold tol
  linarith [htri, hacc, h2, htol]

/-! ## Bound120 -/

/-!
# Room for the accumulation bound

The per-step constants of the `tl.dot_add` accumulation are small enough that, over at most 1024
iterations of magnitude at most `2^120`, `DotLoop.bound rho alp t B` stays below `2^120`, and an
accumulator within that bound of a partial sum of magnitude `B` is below `2^121`.
-/

open TVA

set_option maxHeartbeats 800000

private lemma rho_nonneg' : 0 ≤ rho := by
  unfold rho DotLoop.stepRel Device.relDot16 BK
  rw [Format.binary32_u]
  norm_num

private lemma alp_nonneg' : 0 ≤ alp := by
  unfold alp DotLoop.stepAbs Device.absDot16 BK
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

private lemma rho_le' : rho ≤ 2 ^ (-15 : ℤ) := by
  unfold rho DotLoop.stepRel Device.relDot16 BK
  rw [Format.binary32_u]
  norm_num

private lemma alp_le_one : alp ≤ 1 := by
  unfold alp DotLoop.stepAbs Device.absDot16 BK
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

/-- Up to `1024` steps, `T · rho ≤ 1/2`. -/
theorem t_mul_rho_le_half {t : Nat} (ht : t ≤ 1024) : (t : ℚ) * rho ≤ 1 / 2 := by
  have htq : (t : ℚ) ≤ 1024 := by exact_mod_cast ht
  have ht0 : (0 : ℚ) ≤ t := Nat.cast_nonneg t
  have hmul : (t : ℚ) * rho ≤ 1024 * (2 ^ (-15 : ℤ) : ℚ) :=
    mul_le_mul htq rho_le' rho_nonneg' (by norm_num)
  have h32 : (1024 : ℚ) * (2 ^ (-15 : ℤ) : ℚ) = 1 / 32 := by norm_num
  have hmul' : (t : ℚ) * rho ≤ 1 / 32 := hmul.trans_eq h32
  linarith

/-- The accumulation bound stays below `2^120` when the magnitudes do. -/
theorem bound_le_pow120 {t : Nat} (ht : t ≤ 1024) {B : ℚ} (hB0 : 0 ≤ B)
    (hB : B ≤ 2 ^ (120 : ℤ)) : DotLoop.bound rho alp t B ≤ 2 ^ (120 : ℤ) := by
  have hstep := DotLoop.bound_le rho_nonneg' alp_nonneg' (t_mul_rho_le_half ht) hB0
  have htq : (t : ℚ) ≤ 1024 := by exact_mod_cast ht
  have hrB : rho * B ≤ 2 ^ (105 : ℤ) := by
    have h := mul_le_mul rho_le' hB hB0 (show (0 : ℚ) ≤ 2 ^ (-15 : ℤ) by norm_num)
    have he : (2 ^ (-15 : ℤ) : ℚ) * 2 ^ (120 : ℤ) = 2 ^ (105 : ℤ) := by norm_num
    linarith
  have hsum : rho * B + alp ≤ 2 ^ (105 : ℤ) + 1 := by linarith [alp_le_one]
  have h2 : 2 * (rho * B + alp) ≤ 2 * (2 ^ (105 : ℤ) + 1) := by linarith
  have h2nn : (0 : ℚ) ≤ 2 * (rho * B + alp) := by
    have h0 : 0 ≤ rho * B := mul_nonneg rho_nonneg' hB0
    linarith [alp_nonneg']
  have hmul : (t : ℚ) * (2 * (rho * B + alp)) ≤ 1024 * (2 * (2 ^ (105 : ℤ) + 1)) :=
    mul_le_mul htq h2 h2nn (by norm_num)
  have hnum : (1024 : ℚ) * (2 * (2 ^ (105 : ℤ) + 1)) ≤ 2 ^ (120 : ℤ) := by norm_num
  calc DotLoop.bound rho alp t B ≤ 2 * (t : ℚ) * (rho * B + alp) := hstep
    _ = (t : ℚ) * (2 * (rho * B + alp)) := by ring
    _ ≤ 2 ^ (120 : ℤ) := le_trans hmul hnum

/-- An accumulator within the bound of the partial sums, plus one more step's magnitude, stays below `2^121`. -/
theorem abs_add_le_pow121 {acc St Bt A bnd : ℚ} (h : |acc - St| ≤ bnd) (hSt : |St| ≤ Bt)
    (hBA : Bt + A ≤ 2 ^ (120 : ℤ)) (hbnd : bnd ≤ 2 ^ (120 : ℤ)) (hA0 : 0 ≤ A) :
    |acc| + A ≤ 2 ^ (121 : ℤ) := by
  have := hA0
  have hacc : |acc| ≤ Bt + bnd := by
    have h1 := abs_sub_abs_le_abs_sub acc St
    linarith [abs_nonneg St]
  have hBt0 : 0 ≤ Bt := (abs_nonneg St).trans hSt
  have hAle : A ≤ 2 ^ (120 : ℤ) := by linarith
  have hsplit : (2 : ℚ) ^ (120 : ℤ) + 2 ^ (120 : ℤ) = 2 ^ (121 : ℤ) := by norm_num
  linarith


/-! ## FChain -/

/-!
# The per-token error chain for the `b` operand

For one token `t` of chunk `c`, head `h` and column `n`, the kernel computes

  `dA_diff = dA[last] - dA[k]`, `decay = exp dA_diff`, `w = decay * dt[k]`,
  `b = (bf16 -> f32) b0`, `b = b * w`, `r = (f32 -> bf16) b`

and the claim's `errB` is the accumulated error of exactly that chain against
`bV * (exp dRef * dtV)`.  This file states the chain as one lemma, in the claim's own
vocabulary, so that the loop-body proof only has to supply the `FloatV.NaNEq` / `P.math`
relations that the `Safe` rules hand it.
-/

open TVA

set_option maxHeartbeats 800000

/-! ### Casting rational error bounds to real ones -/

private lemma rat_abs_sub_le_real {a b c : ℚ} {A B C : ℝ}
    (ha : ((a : ℚ) : ℝ) = A) (hb : ((b : ℚ) : ℝ) = B) (hc : ((c : ℚ) : ℝ) = C)
    (h : |a - b| ≤ c) : |A - B| ≤ C := by
  have h1 : ((|a - b| : ℚ) : ℝ) ≤ ((c : ℚ) : ℝ) := by exact_mod_cast h
  rw [Rat.cast_abs] at h1
  push_cast at h1
  rwa [ha, hb, hc] at h1

private lemma u32_cast : ((Format.binary32.u : ℚ) : ℝ) = (2:ℝ)^(-24:ℤ) := by
  rw [Format.binary32_u]; push_cast; norm_num

private lemma eta32_cast : ((Format.binary32.eta : ℚ) : ℝ) = (2:ℝ)^(-150:ℤ) := by
  rw [Format.binary32_eta]; push_cast; norm_num

private lemma ubf_cast : ((Format.bfloat16.u : ℚ) : ℝ) = (2:ℝ)^(-8:ℤ) := by
  rw [Format.bfloat16_u]; push_cast; norm_num

private lemma etabf_cast : ((Format.bfloat16.eta : ℚ) : ℝ) = (2:ℝ)^(-134:ℤ) := by
  rw [Format.bfloat16_eta]; push_cast; norm_num

private lemma sub32_real {x y : FloatV .binary32}
    (h : |(FloatV.sub x y).toRat - (x.toRat - y.toRat)| ≤
      |x.toRat - y.toRat| * Format.binary32.u + Format.binary32.eta) :
    |(FloatV.sub x y).toReal - ((x.toRat : ℝ) - (y.toRat : ℝ))| ≤
      |(x.toRat : ℝ) - (y.toRat : ℝ)| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
  refine rat_abs_sub_le_real ?_ ?_ ?_ h
  · simp [FloatV.toReal]
  · push_cast; ring
  · push_cast; rw [u32_cast, eta32_cast]

private lemma mul32_real {x y : FloatV .binary32}
    (h : |(FloatV.mul x y).toRat - x.toRat * y.toRat| ≤
      |x.toRat * y.toRat| * Format.binary32.u + Format.binary32.eta) :
    |(FloatV.mul x y).toReal - (x.toRat : ℝ) * (y.toRat : ℝ)| ≤
      |(x.toRat : ℝ) * (y.toRat : ℝ)| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
  refine rat_abs_sub_le_real ?_ ?_ ?_ h
  · simp [FloatV.toReal]
  · push_cast; ring
  · push_cast; rw [u32_cast, eta32_cast]

private lemma cvt32bf_real {x : FloatV .binary32}
    (h : |(FloatV.cvtWith .rn x : FloatV .bfloat16).toRat - x.toRat| ≤
      |x.toRat| * Format.bfloat16.u + Format.bfloat16.eta) :
    |(FloatV.cvtWith .rn x : FloatV .bfloat16).toReal - (x.toRat : ℝ)| ≤
      |(x.toRat : ℝ)| * (2:ℝ)^(-8:ℤ) + (2:ℝ)^(-134:ℤ) := by
  refine rat_abs_sub_le_real ?_ ?_ ?_ h
  · simp [FloatV.toReal]
  · ring
  · push_cast; rw [ubf_cast, etabf_cast]


/-! ### Numeric facts about the formats -/

private lemma maxVal32_eq : (Format.binary32.maxVal : ℝ) = ((2:ℝ)^24 - 1) * (2:ℝ)^104 := by
  rw [Format.binary32_maxVal]; push_cast; norm_num

private lemma maxVal32_le_pow128 : (Format.binary32.maxVal : ℝ) ≤ (2:ℝ)^128 := by
  rw [maxVal32_eq]
  have h : ((2:ℝ)^24 - 1) * (2:ℝ)^104 ≤ (2:ℝ)^24 * (2:ℝ)^104 :=
    mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  have h2 : (2:ℝ)^24 * (2:ℝ)^104 = (2:ℝ)^128 := by
    rw [show (128:ℕ) = 24 + 104 by norm_num, pow_add]
  linarith

private lemma pow109_le_maxVal32 : (2:ℝ)^109 ≤ (Format.binary32.maxVal : ℝ) := by
  rw [maxVal32_eq]
  have h32 : (32:ℝ) ≤ (2:ℝ)^24 - 1 := by norm_num
  have h := mul_le_mul_of_nonneg_right h32 (show (0:ℝ) ≤ (2:ℝ)^104 by positivity)
  have h2 : (32:ℝ) * (2:ℝ)^104 = (2:ℝ)^109 := by
    rw [show (109:ℕ) = 5 + 104 by norm_num, pow_add]; norm_num
  linarith

private lemma pow126_le_maxVal32 : (2:ℝ)^126 ≤ (Format.binary32.maxVal : ℝ) := by
  rw [maxVal32_eq]
  have h32 : (2:ℝ)^22 ≤ (2:ℝ)^24 - 1 := by norm_num
  have h := mul_le_mul_of_nonneg_right h32 (show (0:ℝ) ≤ (2:ℝ)^104 by positivity)
  have h2 : (2:ℝ)^22 * (2:ℝ)^104 = (2:ℝ)^126 := by
    rw [show (126:ℕ) = 22 + 104 by norm_num, pow_add]
  linarith

private lemma maxValbf_eq : (Format.bfloat16.maxVal : ℝ) = ((2:ℝ)^8 - 1) * (2:ℝ)^120 := by
  rw [Format.bfloat16_maxVal]; push_cast; norm_num

private lemma three_pow126_le_maxValbf : 3 * (2:ℝ)^126 ≤ (Format.bfloat16.maxVal : ℝ) := by
  rw [maxValbf_eq]
  have h : (3:ℝ) * (2:ℝ)^126 = (192:ℝ) * (2:ℝ)^120 := by
    have h1 : (2:ℝ)^126 = (2:ℝ)^6 * (2:ℝ)^120 := by
      rw [show (126:ℕ) = 6 + 120 by norm_num, pow_add]
    rw [h1]; ring
  rw [h]
  have h2 : (192:ℝ) ≤ (2:ℝ)^8 - 1 := by norm_num
  exact mul_le_mul_of_nonneg_right h2 (by positivity)

private lemma exp65_le_pow104 : Real.exp 65 ≤ (2:ℝ)^104 := by
  have h : Real.exp 65 = (Real.exp 1)^65 := by
    rw [show (65:ℝ) = (65:ℕ) * 1 by norm_num, Real.exp_nat_mul]
  rw [h]
  have h1 : (Real.exp 1)^65 ≤ (3:ℝ)^65 :=
    pow_le_pow_left₀ (Real.exp_pos 1).le Real.exp_one_lt_three.le 65
  have h2 : (3:ℝ)^65 ≤ (2:ℝ)^104 := by norm_num
  linarith

/-! ### Nonnegativity of the claim's error bounds -/

private lemma errE_nonneg {m₀ : Mem} {d : Inst} {c h t : Nat} : 0 ≤ errE m₀ d c h t := by
  unfold errE expRel dErr
  set D := dRef m₀ d c h t with hD
  have hrel : 0 ≤ (2:ℝ) ^ (-22:ℤ) + (2:ℝ) ^ (-23:ℤ) *
      (|D| + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))) := by positivity
  have hexp : Real.exp D ≤
      Real.exp (D + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))) := by
    apply Real.exp_le_exp.mpr
    have h1 : (0:ℝ) ≤ |D| * (2:ℝ) ^ (-24:ℤ) := by positivity
    have h2 : (0:ℝ) ≤ (2:ℝ) ^ (-150:ℤ) := by positivity
    linarith
  have hmul : (1 + 0) * Real.exp D ≤
      (1 + ((2:ℝ) ^ (-22:ℤ) + (2:ℝ) ^ (-23:ℤ) *
        (|D| + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))))) *
      Real.exp (D + (|D| * (2:ℝ) ^ (-24:ℤ) + (2:ℝ) ^ (-150:ℤ))) := by
    apply mul_le_mul (by linarith) hexp (Real.exp_pos D).le (by linarith)
  have h140 : (0:ℝ) ≤ (2:ℝ) ^ (-140:ℤ) := by positivity
  linarith

private lemma eHat_nonneg {m₀ : Mem} {d : Inst} {c h t : Nat} : 0 ≤ eHat m₀ d c h t := by
  have h1 := Real.exp_pos (dRef m₀ d c h t)
  have h2 := errE_nonneg (m₀ := m₀) (d := d) (c := c) (h := h) (t := t)
  unfold eHat
  linarith

private lemma errS_nonneg {m₀ : Mem} {d : Inst} {c h t : Nat} : 0 ≤ errS m₀ d c h t := by
  have hE : 0 ≤ errE m₀ d c h t := errE_nonneg
  have hdt : 0 ≤ |dtV m₀ d c h t| := abs_nonneg _
  have hH : 0 ≤ eHat m₀ d c h t := eHat_nonneg
  have h1 : 0 ≤ |dtV m₀ d c h t| * errE m₀ d c h t := mul_nonneg hdt hE
  have h2 : 0 ≤ (2:ℝ) ^ (-24:ℤ) * (eHat m₀ d c h t * |dtV m₀ d c h t|) :=
    mul_nonneg (by positivity) (mul_nonneg hH hdt)
  have h3 : (0:ℝ) ≤ (2:ℝ) ^ (-150:ℤ) := by positivity
  unfold errS
  linarith

private lemma wRef_abs_le_sHat {m₀ : Mem} {d : Inst} {c h t : Nat} :
    |wRef m₀ d c h t| ≤ sHat m₀ d c h t := by
  have h1 : 0 ≤ errS m₀ d c h t := errS_nonneg
  unfold sHat
  linarith

/-! ### Two exponential estimates -/

private lemma exp_sub_le {D t e : ℝ} (_he0 : 0 ≤ e) (ht : |t - D| ≤ e) :
    |Real.exp t - Real.exp D| ≤ Real.exp (D + e) - Real.exp D := by
  have ht1 : t ≤ D + e := by have := (abs_le.mp ht).2; linarith
  have ht2 : D - e ≤ t := by have := (abs_le.mp ht).1; linarith
  have hA : (0:ℝ) < Real.exp D := Real.exp_pos D
  have hx : (0:ℝ) < Real.exp e := Real.exp_pos e
  have hkey : 2 ≤ Real.exp e + (Real.exp e)⁻¹ := by
    have hsq : 0 ≤ (Real.exp e - 1)^2 := sq_nonneg _
    have hmul : (Real.exp e) * (Real.exp e)⁻¹ = 1 := mul_inv_cancel₀ (ne_of_gt hx)
    nlinarith [hsq, hx, hmul]
  have h3 : Real.exp D - Real.exp (D - e) ≤ Real.exp (D + e) - Real.exp D := by
    have e1 : Real.exp (D - e) = Real.exp D / Real.exp e := by rw [Real.exp_sub]
    have e2 : Real.exp (D + e) = Real.exp D * Real.exp e := Real.exp_add ..
    rw [e1, e2, div_eq_mul_inv]
    have h2 : Real.exp D * 2 ≤ Real.exp D * (Real.exp e + (Real.exp e)⁻¹) :=
      mul_le_mul_of_nonneg_left hkey hA.le
    nlinarith [h2]
  rw [abs_le]
  constructor
  · have h2 : Real.exp (D - e) ≤ Real.exp t := Real.exp_le_exp.mpr ht2
    linarith
  · have h2 : Real.exp t ≤ Real.exp (D + e) := Real.exp_le_exp.mpr ht1
    linarith

private lemma exp_room {D t : ℝ} (hD : D ≤ 64)
    (hDm : |D| ≤ (Format.binary32.maxVal : ℝ))
    (ht : |t - D| ≤ |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)) :
    (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) *
      (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
      * |Real.exp t| + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
  set e : ℝ := |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) with he
  have he0 : 0 ≤ e := by rw [he]; positivity
  have htle : t ≤ D + e := by have := (abs_le.mp ht).2; linarith
  rw [abs_of_pos (Real.exp_pos t)]
  by_cases hcase : |D| ≤ 1024
  · have he1 : e ≤ 1 := by
      rw [he]
      have h1 : |D| * (2:ℝ)^(-24:ℤ) ≤ 1024 * (2:ℝ)^(-24:ℤ) :=
        mul_le_mul_of_nonneg_right hcase (by positivity)
      have h2 : (1024:ℝ) * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) ≤ 1 := by norm_num
      linarith
    have ht65 : t ≤ 65 := by linarith
    have hE1 : Real.exp t ≤ (2:ℝ)^104 :=
      le_trans (Real.exp_le_exp.mpr ht65) exp65_le_pow104
    have hK1 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ 1 := by
      have h1 : |D| + e ≤ (1025:ℝ) := by linarith
      have h2 : (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^(-23:ℤ) * (1025:ℝ) :=
        mul_le_mul_of_nonneg_left h1 (by positivity)
      have h3 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (1025:ℝ) ≤ 1 := by norm_num
      linarith
    have h2 : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) ≤ 2 := by linarith
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp t
        ≤ 2 * (2:ℝ)^104 :=
      mul_le_mul h2 hE1 (Real.exp_pos t).le (by norm_num)
    have hfin : 2 * (2:ℝ)^104 + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
      have h2 : (2:ℝ)^105 + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^109 := by norm_num
      have h3 : (2:ℝ) * (2:ℝ)^104 = (2:ℝ)^105 := by
        rw [show (105:ℕ) = 1 + 104 by norm_num, pow_add]; norm_num
      linarith [pow109_le_maxVal32]
    linarith
  · have h1024 : (1024:ℝ) < |D| := lt_of_not_ge hcase
    have hDneg : D < 0 := by
      rcases lt_or_ge D 0 with h | h
      · exact h
      · rw [abs_of_nonneg h] at h1024; linarith
    have hDle : D ≤ -1024 := by
      rw [abs_of_neg hDneg] at h1024; linarith
    have hDabs : |D| = -D := abs_of_neg hDneg
    have hee : e = (-D) * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by rw [he, hDabs]
    have hDp : D + e < 0 := by
      rw [hee]
      have h1 : D + (-D) * (2:ℝ)^(-24:ℤ) = D * (1 - (2:ℝ)^(-24:ℤ)) := by ring
      have h2 : D * (1 - (2:ℝ)^(-24:ℤ)) ≤ (-1024) * (1 - (2:ℝ)^(-24:ℤ)) :=
        mul_le_mul_of_nonneg_right hDle (by norm_num)
      have h3 : (-1024:ℝ) * (1 - (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ) < 0 := by norm_num
      linarith
    have ht0 : t < 0 := by linarith
    have hexpt1 : Real.exp t ≤ 1 := by
      have h := Real.exp_le_exp.mpr ht0.le
      simpa using h
    have heM : e ≤ (Format.binary32.maxVal : ℝ) := by
      rw [he]
      have h1 : |D| * (2:ℝ)^(-24:ℤ) ≤ (2:ℝ)^128 * (2:ℝ)^(-24:ℤ) :=
        mul_le_mul_of_nonneg_right (le_trans hDm maxVal32_le_pow128) (by positivity)
      have h2 : (2:ℝ)^128 * (2:ℝ)^(-24:ℤ) = (2:ℝ)^104 := by norm_num
      have h3 : (2:ℝ)^104 + (2:ℝ)^(-150:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
        rw [maxVal32_eq]; norm_num
      linarith
    have hK107 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^107 := by
      have hsum : |D| + e ≤ 2 * (2:ℝ)^128 := by
        have h := maxVal32_le_pow128
        linarith
      have h2 : (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^(-23:ℤ) * (2 * (2:ℝ)^128) :=
        mul_le_mul_of_nonneg_left hsum (by positivity)
      have h3 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (2 * (2:ℝ)^128) ≤ (2:ℝ)^107 := by
        norm_num
      linarith
    have h1K : 1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) ≤ (2:ℝ)^108 := by
      have h : (2:ℝ)^107 + 1 ≤ (2:ℝ)^108 := by norm_num
      linarith
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp t
        ≤ (2:ℝ)^108 * 1 :=
      mul_le_mul h1K hexpt1 (Real.exp_pos t).le (by positivity)
    have hfin : (2:ℝ)^108 + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
      have h2 : (2:ℝ)^108 + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^109 := by norm_num
      linarith [pow109_le_maxVal32]
    linarith

/-! ### The three rounding steps of the chain, abstractly -/

private lemma w_chain {w dexp dt D errE eHat errS sHat : ℝ}
    (hW : |w - dexp * dt| ≤ |dexp * dt| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ))
    (hda : |dexp| ≤ eHat)
    (hde : |dexp - Real.exp D| ≤ errE)
    (heE : eHat = Real.exp D + errE)
    (heS : errS = |dt| * errE + (2:ℝ)^(-24:ℤ) * (eHat * |dt|) + (2:ℝ)^(-150:ℤ))
    (hsS : sHat = |Real.exp D * dt| + errS) :
    |w - Real.exp D * dt| ≤ errS ∧ |w| ≤ sHat := by
  have he0 : 0 ≤ errE := le_trans (abs_nonneg _) hde
  have hdt0 : 0 ≤ |dt| := abs_nonneg _
  have heE0 : 0 ≤ eHat := by rw [heE]; linarith [Real.exp_pos D]
  have h1 : |dexp * dt| ≤ eHat * |dt| := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right hda hdt0
  have h2 : |dexp * dt - Real.exp D * dt| = |dt| * |dexp - Real.exp D| := by
    rw [show dexp * dt - Real.exp D * dt = (dexp - Real.exp D) * dt by ring, abs_mul]
    ring
  have h3 : |dexp * dt - Real.exp D * dt| ≤ |dt| * errE := by
    rw [h2]
    exact mul_le_mul_of_nonneg_left hde hdt0
  have h4 : w - Real.exp D * dt = (w - dexp * dt) + (dexp * dt - Real.exp D * dt) := by ring
  have h5 : |w - Real.exp D * dt| ≤ |w - dexp * dt| + |dexp * dt - Real.exp D * dt| := by
    rw [h4]; exact abs_add_le _ _
  have h6 : |dexp * dt| * (2:ℝ)^(-24:ℤ) ≤ (eHat * |dt|) * (2:ℝ)^(-24:ℤ) :=
    mul_le_mul_of_nonneg_right h1 (by positivity)
  have h7 : (eHat * |dt|) * (2:ℝ)^(-24:ℤ) = (2:ℝ)^(-24:ℤ) * (eHat * |dt|) := by ring
  have h8 : |w| ≤ |Real.exp D * dt| + |w - Real.exp D * dt| := by
    have h := abs_add_le (Real.exp D * dt) (w - Real.exp D * dt)
    rw [show Real.exp D * dt + (w - Real.exp D * dt) = w by ring] at h
    exact h
  constructor
  · rw [heS]; linarith [h5, hW, h6, h3]
  · rw [hsS]; linarith

private lemma b2_chain {b2 bv w wRef errS sHat errP pHat : ℝ}
    (hB : |b2 - bv * w| ≤ |bv * w| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ))
    (hw : |w - wRef| ≤ errS) (hws : |w| ≤ sHat)
    (_hsH : sHat = |wRef| + errS)
    (heP : errP = |bv| * errS + (2:ℝ)^(-24:ℤ) * (|bv| * sHat) + (2:ℝ)^(-150:ℤ))
    (hpH : pHat = |bv * wRef| + errP) :
    |b2 - bv * wRef| ≤ errP ∧ |b2| ≤ pHat := by
  have hbv0 : 0 ≤ |bv| := abs_nonneg _
  have h1 : |bv * w| ≤ |bv| * sHat := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hws hbv0
  have h2 : |bv * w - bv * wRef| = |bv| * |w - wRef| := by
    rw [show bv * w - bv * wRef = bv * (w - wRef) by ring, abs_mul]
  have h3 : |bv * w - bv * wRef| ≤ |bv| * errS := by
    rw [h2]; exact mul_le_mul_of_nonneg_left hw hbv0
  have h4 : b2 - bv * wRef = (b2 - bv * w) + (bv * w - bv * wRef) := by ring
  have h5 : |b2 - bv * wRef| ≤ |b2 - bv * w| + |bv * w - bv * wRef| := by
    rw [h4]; exact abs_add_le _ _
  have h6 : |bv * w| * (2:ℝ)^(-24:ℤ) ≤ (|bv| * sHat) * (2:ℝ)^(-24:ℤ) :=
    mul_le_mul_of_nonneg_right h1 (by positivity)
  have h7 : (|bv| * sHat) * (2:ℝ)^(-24:ℤ) = (2:ℝ)^(-24:ℤ) * (|bv| * sHat) := by ring
  have h8 : |b2| ≤ |bv * wRef| + |b2 - bv * wRef| := by
    have h := abs_add_le (bv * wRef) (b2 - bv * wRef)
    rw [show bv * wRef + (b2 - bv * wRef) = b2 by ring] at h
    exact h
  constructor
  · rw [heP]; linarith [h5, hB, h6, h3]
  · rw [hpH]; linarith

private lemma r_chain {r b2 tgt errP pHat errB bHat : ℝ}
    (hR : |r - b2| ≤ |b2| * (2:ℝ)^(-8:ℤ) + (2:ℝ)^(-134:ℤ))
    (hb2 : |b2 - tgt| ≤ errP) (hb2s : |b2| ≤ pHat)
    (heB : errB = errP + (2:ℝ)^(-8:ℤ) * pHat + (2:ℝ)^(-134:ℤ))
    (hbH : bHat = |tgt| + errB) :
    |r - tgt| ≤ errB ∧ |r| ≤ bHat := by
  have h1 : |b2| * (2:ℝ)^(-8:ℤ) ≤ pHat * (2:ℝ)^(-8:ℤ) :=
    mul_le_mul_of_nonneg_right hb2s (by positivity)
  have h2 : pHat * (2:ℝ)^(-8:ℤ) = (2:ℝ)^(-8:ℤ) * pHat := by ring
  have h3 : r - tgt = (r - b2) + (b2 - tgt) := by ring
  have h4 : |r - tgt| ≤ |r - b2| + |b2 - tgt| := by rw [h3]; exact abs_add_le _ _
  have h5 : |r| ≤ |tgt| + |r - tgt| := by
    have h := abs_add_le tgt (r - tgt)
    rw [show tgt + (r - tgt) = r by ring] at h
    exact h
  constructor
  · rw [heB]; linarith
  · rw [hbH]; linarith

private lemma rat_abs_mul_le {x y : FloatV .binary32} {C : ℝ}
    (h : |(x.toRat : ℝ) * (y.toRat : ℝ)| ≤ C)
    (hC : C ≤ (Format.binary32.maxVal : ℝ)) :
    |x.toRat * y.toRat| ≤ Format.binary32.maxVal := by
  have h2 : ((|x.toRat * y.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Rat.cast_abs]
    push_cast
    linarith
  exact_mod_cast h2

private lemma rat_abs_le_bf {x : FloatV .binary32} {C : ℝ}
    (h : |(x.toRat : ℝ)| ≤ C)
    (hC : C ≤ (Format.bfloat16.maxVal : ℝ)) :
    |x.toRat| ≤ Format.bfloat16.maxVal := by
  have h2 : ((|x.toRat| : ℚ) : ℝ) ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Rat.cast_abs]
    linarith
  exact_mod_cast h2

private lemma exp_combined {r t D e : ℝ} (he0 : 0 ≤ e) (ht : |t - D| ≤ e)
    (hr : |r - Real.exp t| ≤
      ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * |Real.exp t| + (2:ℝ)^(-140:ℤ)) :
    |r - Real.exp D| ≤
      (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp (D + e)
        - Real.exp D + (2:ℝ)^(-140:ℤ) := by
  have h2 := exp_sub_le he0 ht
  have htle : t ≤ D + e := by have h := (abs_le.mp ht).2; linarith
  have hexpt : Real.exp t ≤ Real.exp (D + e) := Real.exp_le_exp.mpr htle
  have hK0 : (0:ℝ) ≤ (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) := by positivity
  have h3 : ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * |Real.exp t|
      ≤ ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * Real.exp (D + e) :=
    mul_le_mul_of_nonneg_left (by rwa [abs_of_pos (Real.exp_pos t)]) hK0
  have htri : |r - Real.exp D| ≤ |r - Real.exp t| + |Real.exp t - Real.exp D| := by
    have h := abs_add_le (r - Real.exp t) (Real.exp t - Real.exp D)
    rw [show (r - Real.exp t) + (Real.exp t - Real.exp D) = r - Real.exp D by ring] at h
    exact h
  have hid : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp (D + e)
      - Real.exp D + (2:ℝ)^(-140:ℤ)
      = ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * Real.exp (D + e)
        + (Real.exp (D + e) - Real.exp D) + (2:ℝ)^(-140:ℤ) := by ring
  linarith [htri, hr, h3, h2]

private lemma pHat_le {bv wRef errS sHat errP pHat : ℝ}
    (hr2 : |bv| * sHat ≤ (2:ℝ)^(126:ℤ))
    (hws : |wRef| ≤ sHat)
    (hsH : sHat = |wRef| + errS)
    (heP : errP = |bv| * errS + (2:ℝ)^(-24:ℤ) * (|bv| * sHat) + (2:ℝ)^(-150:ℤ))
    (hpH : pHat = |bv * wRef| + errP) :
    pHat ≤ 3 * (2:ℝ)^(126:ℤ) := by
  have h1 : |bv * wRef| = |bv| * |wRef| := abs_mul ..
  have h2 : |bv| * |wRef| ≤ |bv| * sHat :=
    mul_le_mul_of_nonneg_left hws (abs_nonneg _)
  have h3 : |bv * wRef| ≤ (2:ℝ)^(126:ℤ) := by rw [h1]; linarith
  have heSle : errS ≤ sHat := by rw [hsH]; linarith [abs_nonneg wRef]
  have h4 : |bv| * errS ≤ |bv| * sHat :=
    mul_le_mul_of_nonneg_left heSle (abs_nonneg _)
  have h5 : (2:ℝ)^(-24:ℤ) * (|bv| * sHat) ≤ (2:ℝ)^(-24:ℤ) * (2:ℝ)^(126:ℤ) :=
    mul_le_mul_of_nonneg_left hr2 (by positivity)
  have h6 : (2:ℝ)^(102:ℤ) + (2:ℝ)^(-150:ℤ) ≤ (2:ℝ)^(126:ℤ) := by
    have h1 : (2:ℝ)^(102:ℤ) ≤ (2:ℝ)^(126:ℤ) := by norm_num
    have h2 : (2:ℝ)^(-150:ℤ) ≤ (2:ℝ)^(102:ℤ) := by norm_num
    linarith
  have h7 : (2:ℝ)^(-24:ℤ) * (2:ℝ)^(126:ℤ) = (2:ℝ)^(102:ℤ) := by norm_num
  rw [hpH, heP]
  linarith [h3, h4, h5, h6, h7, hr2]

/-- **The error of the scaled `B`.** The `bfloat16` value of `B · exp(dA_last − dA_t) · dt` is within `errB` of its exact value. -/
theorem hb_chain {E : Env} {d : Inst} {m₀ : Mem} {c h t n : Nat}
    (_hcd : c * d.chunk ≤ t)
    {v0 dk dtv : FloatV .binary32} {b0 : FloatV .bfloat16}
    (hv0 : m₀.read .f32 (d.dA.ptr ↑(dIdx d h c (d.chunk - 1))).addr = some v0)
    (hdk : m₀.read .f32 (d.dA.ptr ↑(dIdx d h c (t - c * d.chunk))).addr = some dk)
    (hdt : m₀.read .f32 (d.dt.ptr ↑(dIdx d h c (t - c * d.chunk))).addr = some dtv)
    (hb0 : m₀.read .bf16 (d.b.ptr ↑(bIdx d t (grp d h) n)).addr = some b0)
    (hv0f : v0.isFinite = true) (hdkf : dk.isFinite = true)
    (hdtf : dtv.isFinite = true) (hb0f : b0.isFinite = true)
    -- both `dA` entries are non-positive (this is what `Vals` gives); together with the
    -- finiteness of the two reads it makes `|dRef| ≤ maxVal`, so the subtraction cannot overflow
    (hv0n : v0.toRat ≤ 0) (hdkn : dk.toRat ≤ 0)
    (hD64 : dRef m₀ d c h t ≤ 64)
    (hr1 : eHat m₀ d c h t * |dtV m₀ d c h t| ≤ (2 : ℝ) ^ (126 : ℤ))
    (hr2 : |bV m₀ d t h n| * sHat m₀ d c h t ≤ (2 : ℝ) ^ (126 : ℤ))
    (hexpacc : Device.ExpF32 E.P)
    {dsub dexp w b1 b2 : FloatV .binary32} {r : FloatV .bfloat16}
    (hsub : FloatV.NaNEq (FloatV.sub v0 dk) dsub)
    (hexp' : E.P.math .exp .binary32 dsub dexp)
    (hw : FloatV.NaNEq (FloatV.mul dexp dtv) w)
    (hb1 : FloatV.NaNEq (FloatV.cvtWith .rn b0) b1)
    (hb2 : FloatV.NaNEq (FloatV.mul b1 w) b2)
    (hr : FloatV.NaNEq (FloatV.cvtWith .rn b2) r) :
    dsub.isFinite = true ∧ dexp.isFinite = true ∧
      |dexp.toReal - Real.exp (dRef m₀ d c h t)| ≤ errE m₀ d c h t ∧
      |dexp.toReal| ≤ eHat m₀ d c h t ∧
      w.isFinite = true ∧
      |w.toReal - wRef m₀ d c h t| ≤ errS m₀ d c h t ∧
      |w.toReal| ≤ sHat m₀ d c h t ∧
      b1.isFinite = true ∧ b1.toReal = bV m₀ d t h n ∧
      b2.isFinite = true ∧
      |b2.toReal - bV m₀ d t h n * wRef m₀ d c h t| ≤ errP m₀ d c h t n ∧
      |b2.toReal| ≤ pHat m₀ d c h t n ∧
      r.isFinite = true ∧
      |r.toReal - bV m₀ d t h n * wRef m₀ d c h t| ≤ errB m₀ d c h t n ∧
      |r.toReal| ≤ bHat m₀ d c h t n := by
  -- readings of the four operands
  have hD : dRef m₀ d c h t = (v0.toRat : ℝ) - (dk.toRat : ℝ) := by
    simp only [dRef, valF, hv0, hdk]
  have hDt : dtV m₀ d c h t = (dtv.toRat : ℝ) := by
    simp only [dtV, valF, hdt]
  have hBv : bV m₀ d t h n = (b0.toRat : ℝ) := by
    simp only [bV, valB, hb0]
  -- magnitudes of the reads
  have hv0m : |v0.toRat| ≤ Format.binary32.maxVal :=
    FloatV.abs_toRat_le_maxVal Format.binary32_regular hv0f
  have hdkm : |dk.toRat| ≤ Format.binary32.maxVal :=
    FloatV.abs_toRat_le_maxVal Format.binary32_regular hdkf
  have hv0abs : |v0.toRat| = -v0.toRat := abs_of_nonpos hv0n
  have hdkabs : |dk.toRat| = -dk.toRat := abs_of_nonpos hdkn
  have hv0m' : -v0.toRat ≤ Format.binary32.maxVal := by rw [← hv0abs]; exact hv0m
  have hdkm' : -dk.toRat ≤ Format.binary32.maxVal := by rw [← hdkabs]; exact hdkm
  have hDm_rat : |v0.toRat - dk.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_le]
    constructor
    · linarith
    · linarith
  have hDm : |dRef m₀ d c h t| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [hD]
    have h1 : ((|v0.toRat - dk.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      exact_mod_cast hDm_rat
    rw [Rat.cast_abs] at h1
    push_cast at h1
    exact h1
  -- the subtraction
  have hsuberr := FloatV.sub_err Format.binary32_regular hv0f hdkf hDm_rat
  have hsubnan : (FloatV.sub v0 dk).isNaN = false :=
    (FloatV.isFinite_iff.mp hsuberr.1).1
  have hdsubeq : dsub = FloatV.sub v0 dk := FloatV.NaNEq.eq hsub hsubnan
  have hdsfin : dsub.isFinite = true := by rw [hdsubeq]; exact hsuberr.1
  have hdErr0 : 0 ≤ dErr m₀ d c h t := by unfold dErr; positivity
  have hdsubD : |dsub.toReal - dRef m₀ d c h t| ≤ dErr m₀ d c h t := by
    have h := sub32_real hsuberr.2
    simp only [FloatV.toReal] at h ⊢
    simp only [dErr]
    rw [hdsubeq, hD]
    exact h
  -- the exponential
  obtain ⟨ref, href, hprop⟩ := hexpacc (|dRef m₀ d c h t| + dErr m₀ d c h t) (by positivity)
  have href' : ref = Real.exp := by
    have h1 : MathFn.ref .exp = some Real.exp := rfl
    rw [h1] at href
    cases href
    rfl
  rw [href'] at hprop
  have hmem : dsub.toReal ∈ Set.Icc (-(|dRef m₀ d c h t| + dErr m₀ d c h t))
      (|dRef m₀ d c h t| + dErr m₀ d c h t) := by
    rw [Set.mem_Icc]
    constructor
    · have h2 := (abs_le.mp hdsubD).1
      have h3 : -|dRef m₀ d c h t| ≤ dRef m₀ d c h t := neg_abs_le _
      linarith
    · have h2 := (abs_le.mp hdsubD).2
      have h3 : dRef m₀ d c h t ≤ |dRef m₀ d c h t| := le_abs_self _
      linarith
  have hroom : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) *
      (|dRef m₀ d c h t| + dErr m₀ d c h t))) * |Real.exp dsub.toReal|
      + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) :=
    exp_room hD64 hDm hdsubD
  have hdexp_res := hprop dsub dexp hexp' hdsfin hmem (by simp [MathFn.natDom]) hroom
  have hdexpfin : dexp.isFinite = true := hdexp_res.1
  have hdexp_err : |dexp.toReal - Real.exp (dRef m₀ d c h t)| ≤ errE m₀ d c h t := by
    have h := exp_combined (D := dRef m₀ d c h t) (e := dErr m₀ d c h t)
      (t := dsub.toReal) hdErr0 hdsubD hdexp_res.2
    simpa only [errE, expRel] using h
  have hdexp_abs : |dexp.toReal| ≤ eHat m₀ d c h t := by
    have h1 : |dexp.toReal| ≤ |Real.exp (dRef m₀ d c h t)| +
        |dexp.toReal - Real.exp (dRef m₀ d c h t)| := by
      have hadd := abs_add_le (Real.exp (dRef m₀ d c h t))
        (dexp.toReal - Real.exp (dRef m₀ d c h t))
      rw [show Real.exp (dRef m₀ d c h t) + (dexp.toReal - Real.exp (dRef m₀ d c h t))
        = dexp.toReal by ring] at hadd
      exact hadd
    rw [abs_of_pos (Real.exp_pos _)] at h1
    unfold eHat
    linarith
  -- the multiplication by `dt`
  have hroom_w_real : |(dexp.toRat : ℝ) * (dtv.toRat : ℝ)| ≤ (2:ℝ)^(126:ℤ) := by
    calc |(dexp.toRat : ℝ) * (dtv.toRat : ℝ)| = |dexp.toReal| * |dtV m₀ d c h t| := by
          rw [abs_mul, ← hDt]; rfl
      _ ≤ eHat m₀ d c h t * |dtV m₀ d c h t| :=
          mul_le_mul_of_nonneg_right hdexp_abs (abs_nonneg _)
      _ ≤ (2:ℝ)^(126:ℤ) := hr1
  have hroom_w : |dexp.toRat * dtv.toRat| ≤ Format.binary32.maxVal :=
    rat_abs_mul_le hroom_w_real pow126_le_maxVal32
  have hmulerr := FloatV.mul_err Format.binary32_regular hdexpfin hdtf hroom_w
  have hwnan : (FloatV.mul dexp dtv).isNaN = false :=
    (FloatV.isFinite_iff.mp hmulerr.1).1
  have hweq : w = FloatV.mul dexp dtv := FloatV.NaNEq.eq hw hwnan
  have hwfin : w.isFinite = true := by rw [hweq]; exact hmulerr.1
  have hw_err : |w.toReal - dexp.toReal * dtV m₀ d c h t| ≤
      |dexp.toReal * dtV m₀ d c h t| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
    have h := mul32_real hmulerr.2
    simp only [FloatV.toReal] at h ⊢
    rw [hweq, hDt]
    exact h
  have hwS := w_chain (D := dRef m₀ d c h t) (errE := errE m₀ d c h t)
    (eHat := eHat m₀ d c h t) (errS := errS m₀ d c h t) (sHat := sHat m₀ d c h t)
    (dt := dtV m₀ d c h t) hw_err hdexp_abs hdexp_err (by rfl) (by rfl) (by rfl)
  -- the widening of `b0`
  have hb1w := FloatV.toRat_cvt_of_wider (g := .binary32) Format.bfloat16_regular
    Format.binary32_regular Format.wider_bfloat16_binary32 RoundMode.rn hb0f
  have hb1nan : (FloatV.cvtWith .rn b0 : FloatV .binary32).isNaN = false :=
    (FloatV.isFinite_iff.mp hb1w.1).1
  have hb1eq : b1 = (FloatV.cvtWith .rn b0 : FloatV .binary32) :=
    FloatV.NaNEq.eq hb1 hb1nan
  have hb1fin : b1.isFinite = true := by rw [hb1eq]; exact hb1w.1
  have hb1val : b1.toReal = bV m₀ d t h n := by
    simp only [FloatV.toReal]
    rw [hb1eq, hb1w.2, hBv]
  -- the multiplication by `b`
  have hroom_b2_real : |(b1.toRat : ℝ) * (w.toRat : ℝ)| ≤ (2:ℝ)^(126:ℤ) := by
    calc |(b1.toRat : ℝ) * (w.toRat : ℝ)| = |b1.toReal| * |w.toReal| := by
          rw [abs_mul]; rfl
      _ = |bV m₀ d t h n| * |w.toReal| := by rw [hb1val]
      _ ≤ |bV m₀ d t h n| * sHat m₀ d c h t :=
          mul_le_mul_of_nonneg_left hwS.2 (abs_nonneg _)
      _ ≤ (2:ℝ)^(126:ℤ) := hr2
  have hroom_b2 : |b1.toRat * w.toRat| ≤ Format.binary32.maxVal :=
    rat_abs_mul_le hroom_b2_real pow126_le_maxVal32
  have hb2err := FloatV.mul_err Format.binary32_regular hb1fin hwfin hroom_b2
  have hb2nan : (FloatV.mul b1 w).isNaN = false :=
    (FloatV.isFinite_iff.mp hb2err.1).1
  have hb2eq : b2 = FloatV.mul b1 w := FloatV.NaNEq.eq hb2 hb2nan
  have hb2fin : b2.isFinite = true := by rw [hb2eq]; exact hb2err.1
  have hb2_err : |b2.toReal - b1.toReal * w.toReal| ≤
      |b1.toReal * w.toReal| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
    have h := mul32_real hb2err.2
    simp only [FloatV.toReal] at h ⊢
    rw [hb2eq]
    exact h
  rw [hb1val] at hb2_err
  have hb2S := b2_chain (bv := bV m₀ d t h n) (wRef := wRef m₀ d c h t)
    (errS := errS m₀ d c h t) (sHat := sHat m₀ d c h t)
    (errP := errP m₀ d c h t n) (pHat := pHat m₀ d c h t n)
    hb2_err hwS.1 hwS.2 (by rfl) (by rfl) (by rfl)
  -- the narrowing to bfloat16
  have hp_le : pHat m₀ d c h t n ≤ 3 * (2:ℝ)^(126:ℤ) :=
    pHat_le hr2 (wRef_abs_le_sHat) (by rfl) (by rfl) (by rfl)
  have hroom_r_real : |(b2.toRat : ℝ)| ≤ (Format.bfloat16.maxVal : ℝ) := by
    have h1 := le_trans hb2S.2 hp_le
    simp only [FloatV.toReal] at h1 ⊢
    linarith [three_pow126_le_maxValbf]
  have hroom_r : |b2.toRat| ≤ Format.bfloat16.maxVal :=
    rat_abs_le_bf hroom_r_real le_rfl
  have hrerr := FloatV.cvt_err (f := .binary32) (g := .bfloat16)
    Format.bfloat16_regular hb2fin hroom_r
  have hrnan : (FloatV.cvtWith .rn b2 : FloatV .bfloat16).isNaN = false :=
    (FloatV.isFinite_iff.mp hrerr.1).1
  have hreq : r = (FloatV.cvtWith .rn b2 : FloatV .bfloat16) :=
    FloatV.NaNEq.eq hr hrnan
  have hrfin : r.isFinite = true := by rw [hreq]; exact hrerr.1
  have hr_err : |r.toReal - b2.toReal| ≤
      |b2.toReal| * (2:ℝ)^(-8:ℤ) + (2:ℝ)^(-134:ℤ) := by
    have h := cvt32bf_real hrerr.2
    simp only [FloatV.toReal] at h ⊢
    rw [hreq]
    exact h
  have hrS := r_chain (tgt := bV m₀ d t h n * wRef m₀ d c h t)
    (errP := errP m₀ d c h t n) (pHat := pHat m₀ d c h t n)
    (errB := errB m₀ d c h t n) (bHat := bHat m₀ d c h t n)
    hr_err hb2S.1 hb2S.2 (by rfl) (by rfl)
  exact ⟨hdsfin, hdexpfin, hdexp_err, hdexp_abs, hwfin, hwS.1, hwS.2,
    hb1fin, hb1val, hb2fin, hb2S.1, hb2S.2, hrfin, hrS.1, hrS.2⟩

/-- The scale `exp(dA_last − dA_t) · dt` is finite. -/
theorem wFinite {E : Env} {d : Inst} {m₀ : Mem} {c h t : Nat}
    (_hcd : c * d.chunk ≤ t)
    {v0 dk dtv : FloatV .binary32}
    (hv0 : m₀.read .f32 (d.dA.ptr ↑(dIdx d h c (d.chunk - 1))).addr = some v0)
    (hdk : m₀.read .f32 (d.dA.ptr ↑(dIdx d h c (t - c * d.chunk))).addr = some dk)
    (hdt : m₀.read .f32 (d.dt.ptr ↑(dIdx d h c (t - c * d.chunk))).addr = some dtv)
    (hv0f : v0.isFinite = true) (hdkf : dk.isFinite = true) (hdtf : dtv.isFinite = true)
    (hv0n : v0.toRat ≤ 0) (hdkn : dk.toRat ≤ 0)
    (hD64 : dRef m₀ d c h t ≤ 64)
    (hr1 : eHat m₀ d c h t * |dtV m₀ d c h t| ≤ (2 : ℝ) ^ (126 : ℤ))
    (hexpacc : Device.ExpF32 E.P)
    {dsub dexp w : FloatV .binary32}
    (hsub : FloatV.NaNEq (FloatV.sub v0 dk) dsub)
    (hexp' : E.P.math .exp .binary32 dsub dexp)
    (hw : FloatV.NaNEq (FloatV.mul dexp dtv) w) :
    dexp.isFinite = true ∧ |dexp.toReal| ≤ eHat m₀ d c h t ∧ w.isFinite = true ∧
      |w.toReal| ≤ sHat m₀ d c h t := by
  -- readings of the two `dA` entries and of `dt`
  have hD : dRef m₀ d c h t = (v0.toRat : ℝ) - (dk.toRat : ℝ) := by
    simp only [dRef, valF, hv0, hdk]
  have hDt : dtV m₀ d c h t = (dtv.toRat : ℝ) := by
    simp only [dtV, valF, hdt]
  -- magnitudes of the reads
  have hv0m : |v0.toRat| ≤ Format.binary32.maxVal :=
    FloatV.abs_toRat_le_maxVal Format.binary32_regular hv0f
  have hdkm : |dk.toRat| ≤ Format.binary32.maxVal :=
    FloatV.abs_toRat_le_maxVal Format.binary32_regular hdkf
  have hv0abs : |v0.toRat| = -v0.toRat := abs_of_nonpos hv0n
  have hdkabs : |dk.toRat| = -dk.toRat := abs_of_nonpos hdkn
  have hv0m' : -v0.toRat ≤ Format.binary32.maxVal := by rw [← hv0abs]; exact hv0m
  have hdkm' : -dk.toRat ≤ Format.binary32.maxVal := by rw [← hdkabs]; exact hdkm
  have hDm_rat : |v0.toRat - dk.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_le]
    constructor
    · linarith
    · linarith
  have hDm : |dRef m₀ d c h t| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [hD]
    have h1 : ((|v0.toRat - dk.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      exact_mod_cast hDm_rat
    rw [Rat.cast_abs] at h1
    push_cast at h1
    exact h1
  -- the subtraction
  have hsuberr := FloatV.sub_err Format.binary32_regular hv0f hdkf hDm_rat
  have hsubnan : (FloatV.sub v0 dk).isNaN = false :=
    (FloatV.isFinite_iff.mp hsuberr.1).1
  have hdsubeq : dsub = FloatV.sub v0 dk := FloatV.NaNEq.eq hsub hsubnan
  have hdsfin : dsub.isFinite = true := by rw [hdsubeq]; exact hsuberr.1
  have hdErr0 : 0 ≤ dErr m₀ d c h t := by unfold dErr; positivity
  have hdsubD : |dsub.toReal - dRef m₀ d c h t| ≤ dErr m₀ d c h t := by
    have h := sub32_real hsuberr.2
    simp only [FloatV.toReal] at h ⊢
    simp only [dErr]
    rw [hdsubeq, hD]
    exact h
  -- the exponential
  obtain ⟨ref, href, hprop⟩ := hexpacc (|dRef m₀ d c h t| + dErr m₀ d c h t) (by positivity)
  have href' : ref = Real.exp := by
    have h1 : MathFn.ref .exp = some Real.exp := rfl
    rw [h1] at href
    cases href
    rfl
  rw [href'] at hprop
  have hmem : dsub.toReal ∈ Set.Icc (-(|dRef m₀ d c h t| + dErr m₀ d c h t))
      (|dRef m₀ d c h t| + dErr m₀ d c h t) := by
    rw [Set.mem_Icc]
    constructor
    · have h2 := (abs_le.mp hdsubD).1
      have h3 : -|dRef m₀ d c h t| ≤ dRef m₀ d c h t := neg_abs_le _
      linarith
    · have h2 := (abs_le.mp hdsubD).2
      have h3 : dRef m₀ d c h t ≤ |dRef m₀ d c h t| := le_abs_self _
      linarith
  have hroom : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) *
      (|dRef m₀ d c h t| + dErr m₀ d c h t))) * |Real.exp dsub.toReal|
      + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) :=
    exp_room hD64 hDm hdsubD
  have hdexp_res := hprop dsub dexp hexp' hdsfin hmem (by simp [MathFn.natDom]) hroom
  have hdexpfin : dexp.isFinite = true := hdexp_res.1
  have hdexp_err : |dexp.toReal - Real.exp (dRef m₀ d c h t)| ≤ errE m₀ d c h t := by
    have h := exp_combined (D := dRef m₀ d c h t) (e := dErr m₀ d c h t)
      (t := dsub.toReal) hdErr0 hdsubD hdexp_res.2
    simpa only [errE, expRel] using h
  have hdexp_abs : |dexp.toReal| ≤ eHat m₀ d c h t := by
    have h1 : |dexp.toReal| ≤ |Real.exp (dRef m₀ d c h t)| +
        |dexp.toReal - Real.exp (dRef m₀ d c h t)| := by
      have hadd := abs_add_le (Real.exp (dRef m₀ d c h t))
        (dexp.toReal - Real.exp (dRef m₀ d c h t))
      rw [show Real.exp (dRef m₀ d c h t) + (dexp.toReal - Real.exp (dRef m₀ d c h t))
        = dexp.toReal by ring] at hadd
      exact hadd
    rw [abs_of_pos (Real.exp_pos _)] at h1
    unfold eHat
    linarith
  -- the multiplication by `dt`
  have hroom_w_real : |(dexp.toRat : ℝ) * (dtv.toRat : ℝ)| ≤ (2:ℝ)^(126:ℤ) := by
    calc |(dexp.toRat : ℝ) * (dtv.toRat : ℝ)| = |dexp.toReal| * |dtV m₀ d c h t| := by
          rw [abs_mul, ← hDt]; rfl
      _ ≤ eHat m₀ d c h t * |dtV m₀ d c h t| :=
          mul_le_mul_of_nonneg_right hdexp_abs (abs_nonneg _)
      _ ≤ (2:ℝ)^(126:ℤ) := hr1
  have hroom_w : |dexp.toRat * dtv.toRat| ≤ Format.binary32.maxVal :=
    rat_abs_mul_le hroom_w_real pow126_le_maxVal32
  have hmulerr := FloatV.mul_err Format.binary32_regular hdexpfin hdtf hroom_w
  have hwnan : (FloatV.mul dexp dtv).isNaN = false :=
    (FloatV.isFinite_iff.mp hmulerr.1).1
  have hweq : w = FloatV.mul dexp dtv := FloatV.NaNEq.eq hw hwnan
  have hwfin : w.isFinite = true := by rw [hweq]; exact hmulerr.1
  have hw_err : |w.toReal - dexp.toReal * dtV m₀ d c h t| ≤
      |dexp.toReal * dtV m₀ d c h t| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
    have h := mul32_real hmulerr.2
    simp only [FloatV.toReal] at h ⊢
    rw [hweq, hDt]
    exact h
  have hwS := w_chain (D := dRef m₀ d c h t) (errE := errE m₀ d c h t)
    (eHat := eHat m₀ d c h t) (errS := errS m₀ d c h t) (sHat := sHat m₀ d c h t)
    (dt := dtV m₀ d c h t) hw_err hdexp_abs hdexp_err (by rfl) (by rfl) (by rfl)
  exact ⟨hdexpfin, hdexp_abs, hwfin, hwS.2⟩

end Claims.SsdChunkState
