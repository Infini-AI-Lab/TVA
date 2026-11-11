import Claim
import TVA.Proof
import Proof.Softplus

set_option maxHeartbeats 4000000

/-!
# The value stored in `g`, in the reals (`A_log` very negative)

The same two bounds as `Proof/GMain.lean`, for `A ≤ -89`, where the group
`2^-23*(64 + 128) * 2^-121 * (spRef x0 + 2^-6)` of `gTol` dominates everything (`(54 + |A|)*e^A`
decreases in `A` on that range, and `e^-89 * 143 < 2^-121 * 192`).
-/

namespace Claims.GdnGating

open TVA

/-- `exp A ≤ 2^k` when `A ≤ k · log 2`. -/
theorem glow_exp_le_pow {A : ℝ} {k : ℤ} (h : A ≤ (k:ℝ) * Real.log 2) :
    Real.exp A ≤ (2:ℝ)^k := by
  have hpos : (0:ℝ) < (2:ℝ)^k := zpow_pos (by norm_num) k
  rw [← Real.le_log_iff_exp_le hpos]
  rw [Real.log_zpow]
  exact h

/-- For `A ≤ -89`, `(2 + |A|) · exp A` is below the floor term `(64 + min |A| 128) · 2^-121 / 64` of `gTol`. -/
theorem glow_key2 {A : ℝ} (hA : A ≤ -89) :
    (2 + |A|) * Real.exp A ≤ (64 + min |A| 128) * (2:ℝ)^(-121:ℤ) / 64 := by
  have ha0 : (0:ℝ) ≤ |A| := abs_nonneg A
  rcases le_or_gt |A| 128 with hle | hgt
  · have hmin : min |A| 128 = |A| := min_eq_left hle
    rw [hmin]
    have hrr : (64 + |A|) * (2:ℝ)^(-121:ℤ) / 64 = (64 + |A|) * (2:ℝ)^(-127:ℤ) := by
      rw [show (2:ℝ)^(-121:ℤ) = (2:ℝ)^(-127:ℤ) * 64 by norm_num]
      ring
    rw [hrr]
    have he127 : Real.exp A ≤ (2:ℝ)^(-127:ℤ) := by
      refine glow_exp_le_pow (k := (-127:ℤ)) ?_
      have hk : ((-127:ℤ):ℝ) = (-127:ℝ) := by norm_num
      have h2 := Real.log_two_lt_d9
      rw [hk]
      nlinarith
    have h1 : (2 + |A|) ≤ (64 + |A|) := by linarith
    calc (2 + |A|) * Real.exp A ≤ (64 + |A|) * Real.exp A :=
          mul_le_mul_of_nonneg_right h1 (Real.exp_pos A).le
      _ ≤ (64 + |A|) * (2:ℝ)^(-127:ℤ) := mul_le_mul_of_nonneg_left he127 (by linarith)
  · have hmin : min |A| 128 = 128 := min_eq_right hgt.le
    rw [hmin]
    have hAneg : A < 0 := by linarith
    have hAn : |A| = -A := abs_of_neg hAneg
    have h184 : Real.exp (-128:ℝ) ≤ (2:ℝ)^(-184:ℤ) := by
      refine glow_exp_le_pow (k := (-184:ℤ)) ?_
      have hk : ((-184:ℤ):ℝ) = (-184:ℝ) := by norm_num
      have h2 := Real.log_two_lt_d9
      rw [hk]
      nlinarith
    have hstep2 : (|A| - 127) * Real.exp (-|A|) ≤ Real.exp (-128:ℝ) := by
      have h1 : |A| - 127 ≤ Real.exp (|A| - 128) := by
        have h3 := Real.add_one_le_exp (|A| - 128)
        linarith
      have h2 : Real.exp (|A| - 128) * Real.exp (-|A|) = Real.exp (-128:ℝ) := by
        rw [← Real.exp_add, show |A| - 128 + -|A| = (-128:ℝ) by ring]
      calc (|A| - 127) * Real.exp (-|A|) ≤ Real.exp (|A| - 128) * Real.exp (-|A|) :=
            mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
        _ = Real.exp (-128:ℝ) := h2
    have hAA : (2 + |A|) * Real.exp (-|A|) ≤ 130 * Real.exp (-128:ℝ) := by
      have hmono : (2 + |A|) ≤ 130 * (|A| - 127) := by linarith
      calc (2 + |A|) * Real.exp (-|A|) ≤ (130 * (|A| - 127)) * Real.exp (-|A|) :=
            mul_le_mul_of_nonneg_right hmono (Real.exp_pos _).le
        _ = 130 * ((|A| - 127) * Real.exp (-|A|)) := by ring
        _ ≤ 130 * Real.exp (-128:ℝ) := mul_le_mul_of_nonneg_left hstep2 (by norm_num)
    have hea : Real.exp A ≤ Real.exp (-|A|) := by
      rw [Real.exp_le_exp]
      linarith
    have hnum : 130 * (2:ℝ)^(-184:ℤ) ≤ (64 + 128) * (2:ℝ)^(-121:ℤ) / 64 := by norm_num
    have hexp130 : 130 * Real.exp (-128:ℝ) ≤ 130 * (2:ℝ)^(-184:ℤ) :=
      mul_le_mul_of_nonneg_left h184 (by norm_num)
    calc (2 + |A|) * Real.exp A ≤ (2 + |A|) * Real.exp (-|A|) :=
          mul_le_mul_of_nonneg_left hea (by linarith)
      _ ≤ 130 * Real.exp (-128:ℝ) := hAA
      _ ≤ (64 + 128) * (2:ℝ)^(-121:ℤ) / 64 := le_trans hexp130 hnum

/-- **The arithmetic of `g` for `A ≤ -89`**: with `e = exp A` below `2^-121 / 128` and the exponential's error `de`, the collected error of the product `-E · S` (rounded) is at most `2^-23 · c · (e + 2^-121) · (p + 2^-6)`, the `gTol` shape, for `c ≥ 153`. -/
theorem glow_arith_bound {e p de c : ℝ}
    (he : 0 < e) (hp : 0 < p) (hc : 153 ≤ c)
    (_hde0 : 0 ≤ de)
    (hde : de ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 + (2:ℝ)^(-140:ℤ))
    (heE : e ≤ (2:ℝ)^(-121:ℤ)/128) :
    (2:ℝ)^(-24:ℤ)*(e+de)*(p + (52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))))
      + (e+de)*(52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))
      + de*p + (2:ℝ)^(-150:ℤ)
    ≤ (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by
  set dp := 52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)) with hdp
  set D := (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 + (2:ℝ)^(-140:ℤ) with hD
  set E := (2:ℝ)^(-121:ℤ)/128 with hE
  have hu : (0:ℝ) < (2:ℝ)^(-24:ℤ) := by norm_num
  have hU : (0:ℝ) < (2:ℝ)^(-23:ℤ) := by norm_num
  have hkap : (0:ℝ) < (2:ℝ)^(-121:ℤ) := by norm_num
  have hc0 : (0:ℝ) < c := by linarith
  have hdp0 : (0:ℝ) ≤ dp := by rw [hdp]; positivity
  have hpp : (0:ℝ) ≤ p + dp := by linarith
  have hD0 : (0:ℝ) ≤ D := by rw [hD]; positivity
  have hE0 : (0:ℝ) ≤ E := by rw [hE]; positivity
  have hsum : e + de ≤ E + D := by linarith
  have k1 : de*(p+dp) ≤ D*(p+dp) := mul_le_mul_of_nonneg_right hde hpp
  have k2 : (e+de)*dp ≤ (E+D)*dp := mul_le_mul_of_nonneg_right hsum hdp0
  have k3 : de*p ≤ D*p := mul_le_mul_of_nonneg_right hde hp.le
  have k4 : (2:ℝ)^(-24:ℤ)*(e+de)*(p+dp) ≤ (2:ℝ)^(-24:ℤ)*(E+D)*(p+dp) := by
    have h1 : (e+de)*(p+dp) ≤ (E+D)*(p+dp) := mul_le_mul_of_nonneg_right hsum hpp
    have h2 := mul_le_mul_of_nonneg_left h1 hu.le
    calc (2:ℝ)^(-24:ℤ)*(e+de)*(p+dp) = (2:ℝ)^(-24:ℤ)*((e+de)*(p+dp)) := by ring
      _ ≤ (2:ℝ)^(-24:ℤ)*((E+D)*(p+dp)) := h2
      _ = (2:ℝ)^(-24:ℤ)*(E+D)*(p+dp) := by ring
  have step1 : (2:ℝ)^(-24:ℤ)*(e+de)*(p+dp) + (e+de)*dp + de*p + (2:ℝ)^(-150:ℤ)
      ≤ (2:ℝ)^(-24:ℤ)*(E+D)*(p+dp) + (E+D)*dp + D*p + (2:ℝ)^(-150:ℤ) := by
    linarith [k1, k2, k3, k4]
  have hid : (2:ℝ)^(-24:ℤ)*(E+D)*(p+dp) + (E+D)*dp + D*p + (2:ℝ)^(-150:ℤ)
      = p*(D + (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ)))
        + ((E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*(1+(2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ)) := by
    rw [hdp]; ring
  have h23 : (2:ℝ)^(-23:ℤ) = 1/8388608 := by norm_num
  have h24 : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
  have h19 : (2:ℝ)^(-19:ℤ) = 1/524288 := by norm_num
  have h140 : (2:ℝ)^(-140:ℤ) = 1/1393796574908163946345982392040522594123776 := by norm_num
  have h150 : (2:ℝ)^(-150:ℤ) = 1/1427247692705959881058285969449495136382746624 := by norm_num
  have h121 : (2:ℝ)^(-121:ℤ) = 1/2658455991569831745807614120560689152 := by norm_num
  have hal : D + (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ))
      ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ) := by
    rw [hD, hE, h23, h24, h140, h121]; linarith
  have hbe : (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*(1+(2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ)
      ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := by
    rw [hD, hE, h23, h24, h19, h140, h150, h121]; linarith
  have hstep2 : p*(D + (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ)))
        + ((E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*(1+(2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ))
      ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := by
    have h1 : p * (D + (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ))) ≤ p * ((2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)) :=
      mul_le_mul_of_nonneg_left hal hp.le
    have e1 : p * ((2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)) = (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p := by ring
    rw [e1] at h1
    exact add_le_add h1 hbe
  have h6 : (2:ℝ)^(-6:ℤ) = 1/64 := by norm_num
  have hfinal : (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64
      ≤ (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by
    have hinner : (2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-121:ℤ)*(1/64) ≤ (e+(2:ℝ)^(-121:ℤ))*(p+(1/64)) := by
      nlinarith [he, hp, hkap]
    have hcU : (0:ℝ) ≤ (2:ℝ)^(-23:ℤ)*c := mul_nonneg hU.le hc0.le
    calc (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64
        = (2:ℝ)^(-23:ℤ)*c*((2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-121:ℤ)*(1/64)) := by ring
      _ ≤ (2:ℝ)^(-23:ℤ)*c*((e+(2:ℝ)^(-121:ℤ))*(p+(1/64))) := mul_le_mul_of_nonneg_left hinner hcU
      _ = (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by rw [h6]; ring
  calc (2:ℝ)^(-24:ℤ)*(e+de)*(p + (52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))))
      + (e+de)*(52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))
      + de*p + (2:ℝ)^(-150:ℤ)
      = (2:ℝ)^(-24:ℤ)*(e+de)*(p+dp) + (e+de)*dp + de*p + (2:ℝ)^(-150:ℤ) := by rw [hdp]
    _ ≤ (2:ℝ)^(-24:ℤ)*(E+D)*(p+dp) + (E+D)*dp + D*p + (2:ℝ)^(-150:ℤ) := step1
    _ = p*(D + (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ)))
        + ((E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*(1+(2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ)) := hid
    _ ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := hstep2
    _ ≤ (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := hfinal

/-- **The same arithmetic when the product overflows**: the exact `E · S` exceeds `|gRef|` by at most `gTol`'s shape (with the `1 + 2^-21` factor of the overflow clause). -/
theorem glow_arith_bound_inf {e p de c : ℝ}
    (he : 0 < e) (hp : 0 < p) (hc : 153 ≤ c)
    (hde : de ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 + (2:ℝ)^(-140:ℤ))
    (heE : e ≤ (2:ℝ)^(-121:ℤ)/128) :
    (1+(2:ℝ)^(-21:ℤ))*((e+de)*(52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))) + de*p)
      + (2:ℝ)^(-21:ℤ)*e*p
    ≤ (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by
  set dp := 52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)) with hdp
  set D := (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 + (2:ℝ)^(-140:ℤ) with hD
  set E := (2:ℝ)^(-121:ℤ)/128 with hE
  set al := D + (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ)) with hal
  set be := (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*(1+(2:ℝ)^(-24:ℤ)) with hbe
  have hu : (0:ℝ) < (2:ℝ)^(-24:ℤ) := by norm_num
  have hU : (0:ℝ) < (2:ℝ)^(-23:ℤ) := by norm_num
  have hkap : (0:ℝ) < (2:ℝ)^(-121:ℤ) := by norm_num
  have hr : (0:ℝ) < (2:ℝ)^(-21:ℤ) := by norm_num
  have hc0 : (0:ℝ) < c := by linarith
  have hdp0 : (0:ℝ) ≤ dp := by rw [hdp]; positivity
  have hD0 : (0:ℝ) ≤ D := by rw [hD]; positivity
  have hE0 : (0:ℝ) ≤ E := by rw [hE]; positivity
  have hED : (0:ℝ) ≤ E + D := by linarith
  have hsum : e + de ≤ E + D := by linarith
  have k1 : de*p ≤ D*p := mul_le_mul_of_nonneg_right hde hp.le
  have k2 : (e+de)*dp ≤ (E+D)*dp := mul_le_mul_of_nonneg_right hsum hdp0
  have hX : (e+de)*dp + de*p ≤ (E+D)*dp + D*p := by linarith [k1, k2]
  have hA1 : 52*(2:ℝ)^(-23:ℤ) ≤ (2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ) := by nlinarith [hu]
  have hA2 : (E+D)*(52*(2:ℝ)^(-23:ℤ)) ≤ (E+D)*((2:ℝ)^(-24:ℤ)*(1+52*(2:ℝ)^(-23:ℤ)) + 52*(2:ℝ)^(-23:ℤ)) :=
    mul_le_mul_of_nonneg_left hA1 hED
  have hA3 : 52*(2:ℝ)^(-23:ℤ)*(E+D) = (E+D)*(52*(2:ℝ)^(-23:ℤ)) := by ring
  have hA : D + 52*(2:ℝ)^(-23:ℤ)*(E+D) ≤ al := by
    rw [hal, hA3]; linarith [hA2]
  have hB : (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))) ≤ be := by
    rw [hbe]
    have h1 : (1:ℝ) ≤ 1 + (2:ℝ)^(-24:ℤ) := by linarith [hu]
    have h0 : (0:ℝ) ≤ (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))) := by positivity
    calc (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))) = (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*1 := by ring
      _ ≤ (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)))*(1+(2:ℝ)^(-24:ℤ)) := mul_le_mul_of_nonneg_left h1 h0
  have hX2 : (E+D)*dp + D*p ≤ p*al + be := by
    have hid0 : (E+D)*dp + D*p = p*(D + 52*(2:ℝ)^(-23:ℤ)*(E+D)) + (E+D)*((2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))) := by
      rw [hdp]; ring
    rw [hid0]
    have h1 : p*(D + 52*(2:ℝ)^(-23:ℤ)*(E+D)) ≤ p*al := mul_le_mul_of_nonneg_left hA hp.le
    exact add_le_add h1 hB
  have hX3 : (e+de)*dp + de*p ≤ p*al + be := le_trans hX hX2
  have hstep1 : (1+(2:ℝ)^(-21:ℤ))*((e+de)*dp + de*p) + (2:ℝ)^(-21:ℤ)*e*p
      ≤ (1+(2:ℝ)^(-21:ℤ))*(p*al + be) + (2:ℝ)^(-21:ℤ)*E*p := by
    have h2 : (1+(2:ℝ)^(-21:ℤ))*((e+de)*dp + de*p) ≤ (1+(2:ℝ)^(-21:ℤ))*(p*al + be) :=
      mul_le_mul_of_nonneg_left hX3 (by linarith [hr])
    have h3 : (2:ℝ)^(-21:ℤ)*e*p ≤ (2:ℝ)^(-21:ℤ)*E*p := by
      have h4 : (2:ℝ)^(-21:ℤ)*e ≤ (2:ℝ)^(-21:ℤ)*E := mul_le_mul_of_nonneg_left heE hr.le
      exact mul_le_mul_of_nonneg_right h4 hp.le
    exact add_le_add h2 h3
  have hid2 : (1+(2:ℝ)^(-21:ℤ))*(p*al + be) + (2:ℝ)^(-21:ℤ)*E*p
      = p*((1+(2:ℝ)^(-21:ℤ))*al + (2:ℝ)^(-21:ℤ)*E) + (1+(2:ℝ)^(-21:ℤ))*be := by ring
  have h21 : (2:ℝ)^(-21:ℤ) = 1/2097152 := by norm_num
  have h23 : (2:ℝ)^(-23:ℤ) = 1/8388608 := by norm_num
  have h24 : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
  have h19 : (2:ℝ)^(-19:ℤ) = 1/524288 := by norm_num
  have h140 : (2:ℝ)^(-140:ℤ) = 1/1393796574908163946345982392040522594123776 := by norm_num
  have h150 : (2:ℝ)^(-150:ℤ) = 1/1427247692705959881058285969449495136382746624 := by norm_num
  have h121 : (2:ℝ)^(-121:ℤ) = 1/2658455991569831745807614120560689152 := by norm_num
  have halpha : (1+(2:ℝ)^(-21:ℤ))*al + (2:ℝ)^(-21:ℤ)*E ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ) := by
    rw [hal, hD, hE, h21, h23, h24, h140, h121]; linarith only [hc]
  have hbeta : (1+(2:ℝ)^(-21:ℤ))*be + (2:ℝ)^(-150:ℤ) ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := by
    rw [hbe, hD, hE, h21, h23, h24, h19, h140, h150, h121]; linarith only [hc]
  have hstep2 : p*((1+(2:ℝ)^(-21:ℤ))*al + (2:ℝ)^(-21:ℤ)*E) + (1+(2:ℝ)^(-21:ℤ))*be
      ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := by
    have h1 : p*((1+(2:ℝ)^(-21:ℤ))*al + (2:ℝ)^(-21:ℤ)*E) ≤ p*((2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)) :=
      mul_le_mul_of_nonneg_left halpha hp.le
    have e1 : p*((2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)) = (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p := by ring
    rw [e1] at h1
    have h2 : (1+(2:ℝ)^(-21:ℤ))*be ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := by linarith [hbeta]
    exact add_le_add h1 h2
  have h6 : (2:ℝ)^(-6:ℤ) = 1/64 := by norm_num
  have hfinal : (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64
      ≤ (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by
    have hinner : (2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-121:ℤ)*(1/64) ≤ (e+(2:ℝ)^(-121:ℤ))*(p+(1/64)) := by
      nlinarith [he, hp, hkap]
    have hcU : (0:ℝ) ≤ (2:ℝ)^(-23:ℤ)*c := mul_nonneg hU.le hc0.le
    calc (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64
        = (2:ℝ)^(-23:ℤ)*c*((2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-121:ℤ)*(1/64)) := by ring
      _ ≤ (2:ℝ)^(-23:ℤ)*c*((e+(2:ℝ)^(-121:ℤ))*(p+(1/64))) := mul_le_mul_of_nonneg_left hinner hcU
      _ = (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by rw [h6]; ring
  calc (1+(2:ℝ)^(-21:ℤ))*((e+de)*(52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ))) + de*p) + (2:ℝ)^(-21:ℤ)*e*p
      = (1+(2:ℝ)^(-21:ℤ))*((e+de)*dp + de*p) + (2:ℝ)^(-21:ℤ)*e*p := by rw [hdp]
    _ ≤ (1+(2:ℝ)^(-21:ℤ))*(p*al + be) + (2:ℝ)^(-21:ℤ)*E*p := hstep1
    _ = p*((1+(2:ℝ)^(-21:ℤ))*al + (2:ℝ)^(-21:ℤ)*E) + (1+(2:ℝ)^(-21:ℤ))*be := hid2
    _ ≤ (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)*p + (2:ℝ)^(-23:ℤ)*c*(2:ℝ)^(-121:ℤ)/64 := hstep2
    _ ≤ (2:ℝ)^(-23:ℤ)*c*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := hfinal

/-- **`g`, finite case, `A` very negative.** -/
theorem g_bound_low {A x0 E S gv : ℝ}
    (hA : A ≤ -89)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ))
    (hS : |S - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)))
    (hgv : |gv + E * S| ≤ 2 ^ (-24 : ℤ) * |E * S| + 2 ^ (-150 : ℤ)) :
    |gv - gRef A x0| ≤ gTol A x0 := by
  have hAneg : A < 0 := by linarith
  have ha89 : (89:ℝ) ≤ |A| := by
    have h := abs_of_neg hAneg
    linarith
  have hc : (153:ℝ) ≤ 64 + min |A| 128 := by
    have h1 : (89:ℝ) ≤ min |A| 128 := le_min ha89 (by norm_num)
    linarith
  have hepos : 0 < Real.exp A := Real.exp_pos A
  have hppos : 0 < spRef x0 := spRef_pos x0
  set dp := 52*(2:ℝ)^(-23:ℤ)*spRef x0 + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)) with hdp_def
  set de := Real.exp A*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) + (2:ℝ)^(-140:ℤ) with hde_def
  have hEe : |E - Real.exp A| ≤ de := by
    rw [hde_def]
    have h22 : (2:ℝ)^(-22:ℤ) = 2*(2:ℝ)^(-23:ℤ) := by norm_num
    have hE' := hE
    rw [h22] at hE'
    have e2 : (2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) * Real.exp A
        = Real.exp A*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) := by ring
    rw [← e2]
    exact hE'
  have hEabs : |E| ≤ Real.exp A + de := by
    rw [abs_le]
    constructor
    · have h2 := (abs_le.mp hEe).1
      linarith
    · have h2 := (abs_le.mp hEe).2
      linarith
  have hSp : |S - spRef x0| ≤ dp := by
    rw [hdp_def]
    exact hS
  have hSabs : |S| ≤ spRef x0 + dp := by
    have h2 : |S| = |spRef x0 + (S - spRef x0)| :=
      congrArg abs (by ring : S = spRef x0 + (S - spRef x0))
    rw [h2]
    calc |spRef x0 + (S - spRef x0)| ≤ |spRef x0| + |S - spRef x0| := abs_add_le _ _
      _ = spRef x0 + |S - spRef x0| := by rw [abs_of_pos hppos]
      _ ≤ spRef x0 + dp := by linarith [hSp]
  have hde0 : (0:ℝ) ≤ de := by rw [hde_def]; positivity
  have hde : de ≤ (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(2:ℝ)^(-121:ℤ)/64 + (2:ℝ)^(-140:ℤ) := by
    rw [hde_def]
    have hkey := glow_key2 hA
    have hU : (0:ℝ) < (2:ℝ)^(-23:ℤ) := by norm_num
    have h1 := mul_le_mul_of_nonneg_left hkey hU.le
    have e1 : (2:ℝ)^(-23:ℤ)*((2+|A|)*Real.exp A)
        = Real.exp A*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) := by ring
    have e2 : (2:ℝ)^(-23:ℤ)*((64 + min |A| 128)*(2:ℝ)^(-121:ℤ)/64)
        = (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(2:ℝ)^(-121:ℤ)/64 := by ring
    rw [e1, e2] at h1
    linarith [h1]
  have heE : Real.exp A ≤ (2:ℝ)^(-121:ℤ)/128 := by
    have h : Real.exp A ≤ (2:ℝ)^(-128:ℤ) := by
      refine glow_exp_le_pow (k := (-128:ℤ)) ?_
      have hk : ((-128:ℤ):ℝ) = (-128:ℝ) := by norm_num
      have h2 := Real.log_two_lt_d9
      rw [hk]
      nlinarith
    have h3 : (2:ℝ)^(-121:ℤ)/128 = (2:ℝ)^(-128:ℤ) := by norm_num
    rw [h3]
    exact h
  have htri : |gv - gRef A x0| ≤
      (2:ℝ)^(-24:ℤ) * |E| * |S| + (2:ℝ)^(-150:ℤ)
        + |E| * |S - spRef x0| + |E - Real.exp A| * spRef x0 := by
    have habs : gv - gRef A x0 = gv + Real.exp A * spRef x0 := by unfold gRef; ring
    rw [habs]
    have h1 : |gv + Real.exp A * spRef x0| ≤ |gv + E*S| + |E*S - Real.exp A*spRef x0| := by
      have h : gv + Real.exp A * spRef x0 = (gv + E*S) - (E*S - Real.exp A*spRef x0) := by ring
      rw [h]; exact abs_sub _ _
    have h2 : |gv + E*S| ≤ (2:ℝ)^(-24:ℤ) * |E*S| + (2:ℝ)^(-150:ℤ) := hgv
    have h3 : |E*S - Real.exp A*spRef x0| ≤ |E| * |S - spRef x0| + |E - Real.exp A| * spRef x0 := by
      have h : E*S - Real.exp A*spRef x0 = E*(S - spRef x0) + (E - Real.exp A)*spRef x0 := by ring
      rw [h]
      have h4 := abs_add_le (E*(S - spRef x0)) ((E - Real.exp A)*spRef x0)
      rw [abs_mul, abs_mul, abs_of_nonneg hppos.le] at h4
      exact h4
    have h5 : |E*S| = |E| * |S| := abs_mul E S
    rw [h5] at h2
    linarith [h1, h2, h3]
  have hde0 : (0:ℝ) ≤ de := by rw [hde_def]; positivity
  have hK1 : (0:ℝ) ≤ Real.exp A + de := by linarith
  have hp1 : |E| * |S| ≤ (Real.exp A + de) * (spRef x0 + dp) :=
    mul_le_mul hEabs hSabs (abs_nonneg S) hK1
  have e1 : (2:ℝ)^(-24:ℤ) * |E| * |S| ≤ (2:ℝ)^(-24:ℤ) * (Real.exp A + de) * (spRef x0 + dp) := by
    calc (2:ℝ)^(-24:ℤ) * |E| * |S| = (2:ℝ)^(-24:ℤ) * (|E| * |S|) := by ring
      _ ≤ (2:ℝ)^(-24:ℤ) * ((Real.exp A + de) * (spRef x0 + dp)) :=
            mul_le_mul_of_nonneg_left hp1 (by norm_num)
      _ = (2:ℝ)^(-24:ℤ) * (Real.exp A + de) * (spRef x0 + dp) := by ring
  have e2 : |E| * |S - spRef x0| ≤ (Real.exp A + de) * dp :=
    mul_le_mul hEabs hSp (abs_nonneg _) hK1
  have e3 : |E - Real.exp A| * spRef x0 ≤ de * spRef x0 :=
    mul_le_mul hEe le_rfl hppos.le hde0
  have harith := glow_arith_bound hepos hppos hc hde0 hde heE
  have harith' : (2:ℝ)^(-24:ℤ)*(Real.exp A + de)*(spRef x0 + dp)
      + (Real.exp A + de)*dp + de*spRef x0 + (2:ℝ)^(-150:ℤ)
      ≤ (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(Real.exp A + (2:ℝ)^(-121:ℤ))*(spRef x0 + (2:ℝ)^(-6:ℤ)) := by
    rw [hdp_def]
    exact harith
  have hgoal : gTol A x0 = (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(Real.exp A + (2:ℝ)^(-121:ℤ))*(spRef x0 + (2:ℝ)^(-6:ℤ)) := by
    unfold gTol gMag gEps
    ring
  rw [hgoal]
  linarith only [htri, e1, e2, e3, harith']

/-- **`g`, overflow, `A` very negative.** (Unreachable in practice, kept for a total case split.) -/
theorem g_inf_low {A x0 E S : ℝ}
    (hA : A ≤ -89)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ))
    (hS : |S - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)))
    (hbig : (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤ |E * S|) :
    (Format.binary32.maxVal : ℝ) ≤ |gRef A x0| + gTol A x0 := by
  have hAneg : A < 0 := by linarith
  have ha89 : (89:ℝ) ≤ |A| := by
    have h := abs_of_neg hAneg
    linarith
  have hc : (153:ℝ) ≤ 64 + min |A| 128 := by
    have h1 : (89:ℝ) ≤ min |A| 128 := le_min ha89 (by norm_num)
    linarith
  have hepos : 0 < Real.exp A := Real.exp_pos A
  have hppos : 0 < spRef x0 := spRef_pos x0
  set dp := 52*(2:ℝ)^(-23:ℤ)*spRef x0 + (2:ℝ)^(-24:ℤ)*(1+(2:ℝ)^(-19:ℤ)) with hdp_def
  set de := Real.exp A*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) + (2:ℝ)^(-140:ℤ) with hde_def
  have hEe : |E - Real.exp A| ≤ de := by
    rw [hde_def]
    have h22 : (2:ℝ)^(-22:ℤ) = 2*(2:ℝ)^(-23:ℤ) := by norm_num
    have hE' := hE
    rw [h22] at hE'
    have e2 : (2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) * Real.exp A
        = Real.exp A*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) := by ring
    rw [← e2]
    exact hE'
  have hEabs : |E| ≤ Real.exp A + de := by
    rw [abs_le]
    constructor
    · have h2 := (abs_le.mp hEe).1
      linarith
    · have h2 := (abs_le.mp hEe).2
      linarith
  have hSp : |S - spRef x0| ≤ dp := by
    rw [hdp_def]
    exact hS
  have hSabs : |S| ≤ spRef x0 + dp := by
    have h2 : |S| = |spRef x0 + (S - spRef x0)| :=
      congrArg abs (by ring : S = spRef x0 + (S - spRef x0))
    rw [h2]
    calc |spRef x0 + (S - spRef x0)| ≤ |spRef x0| + |S - spRef x0| := abs_add_le _ _
      _ = spRef x0 + |S - spRef x0| := by rw [abs_of_pos hppos]
      _ ≤ spRef x0 + dp := by linarith [hSp]
  have hde : de ≤ (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(2:ℝ)^(-121:ℤ)/64 + (2:ℝ)^(-140:ℤ) := by
    rw [hde_def]
    have hkey := glow_key2 hA
    have hU : (0:ℝ) < (2:ℝ)^(-23:ℤ) := by norm_num
    have h1 := mul_le_mul_of_nonneg_left hkey hU.le
    have e1 : (2:ℝ)^(-23:ℤ)*((2+|A|)*Real.exp A)
        = Real.exp A*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*|A|) := by ring
    have e2 : (2:ℝ)^(-23:ℤ)*((64 + min |A| 128)*(2:ℝ)^(-121:ℤ)/64)
        = (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(2:ℝ)^(-121:ℤ)/64 := by ring
    rw [e1, e2] at h1
    linarith [h1]
  have heE : Real.exp A ≤ (2:ℝ)^(-121:ℤ)/128 := by
    have h : Real.exp A ≤ (2:ℝ)^(-128:ℤ) := by
      refine glow_exp_le_pow (k := (-128:ℤ)) ?_
      have hk : ((-128:ℤ):ℝ) = (-128:ℝ) := by norm_num
      have h2 := Real.log_two_lt_d9
      rw [hk]
      nlinarith
    have h3 : (2:ℝ)^(-121:ℤ)/128 = (2:ℝ)^(-128:ℤ) := by norm_num
    rw [h3]
    exact h
  have hde0 : (0:ℝ) ≤ de := by rw [hde_def]; positivity
  have hK1 : (0:ℝ) ≤ Real.exp A + de := by linarith
  have hp1 : |E| * |S| ≤ (Real.exp A + de) * (spRef x0 + dp) :=
    mul_le_mul hEabs hSabs (abs_nonneg S) hK1
  have hgr : |gRef A x0| = Real.exp A * spRef x0 := by
    unfold gRef
    rw [abs_neg, abs_of_pos (mul_pos hepos hppos)]
  have hgoal : gTol A x0 = (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(Real.exp A + (2:ℝ)^(-121:ℤ))*(spRef x0 + (2:ℝ)^(-6:ℤ)) := by
    unfold gTol gMag gEps
    ring
  rw [hgr, hgoal]
  set M := (Format.binary32.maxVal : ℝ) with hM_def
  have h1 : (1 - (2:ℝ)^(-22:ℤ)) * M ≤ (Real.exp A + de) * (spRef x0 + dp) := by
    have h2 : |E*S| = |E| * |S| := abs_mul E S
    have hb := hbig
    rw [h2] at hb
    exact le_trans hb hp1
  have hr : (0:ℝ) < (2:ℝ)^(-21:ℤ) := by norm_num
  have hX : (Real.exp A + de) * (spRef x0 + dp)
      = Real.exp A * spRef x0 + ((Real.exp A + de)*dp + de*spRef x0) := by ring
  have hmul : (1 + (2:ℝ)^(-21:ℤ))*((1 - (2:ℝ)^(-22:ℤ))*M)
      ≤ (1 + (2:ℝ)^(-21:ℤ))*(Real.exp A * spRef x0 + ((Real.exp A + de)*dp + de*spRef x0)) := by
    have h2 := h1
    rw [hX] at h2
    exact mul_le_mul_of_nonneg_left h2 (by linarith [hr])
  have hcf : (1:ℝ) ≤ (1 + (2:ℝ)^(-21:ℤ))*(1 - (2:ℝ)^(-22:ℤ)) := by norm_num
  have hMposQ : (0:ℚ) < Format.binary32.maxVal :=
    Format.maxVal_pos (by decide)
  have hMpos : (0:ℝ) < M := by
    rw [hM_def]
    exact_mod_cast hMposQ
  have hMle : M ≤ (1 + (2:ℝ)^(-21:ℤ))*((1 - (2:ℝ)^(-22:ℤ))*M) := by
    have h2 := mul_le_mul_of_nonneg_right hcf hMpos.le
    calc M = 1*M := by ring
      _ ≤ ((1 + (2:ℝ)^(-21:ℤ))*(1 - (2:ℝ)^(-22:ℤ)))*M := h2
      _ = (1 + (2:ℝ)^(-21:ℤ))*((1 - (2:ℝ)^(-22:ℤ))*M) := by ring
  have harith := glow_arith_bound_inf hepos hppos hc hde heE
  have harith' : (1 + (2:ℝ)^(-21:ℤ))*((Real.exp A + de)*dp + de*spRef x0) + (2:ℝ)^(-21:ℤ)*Real.exp A*spRef x0
      ≤ (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(Real.exp A + (2:ℝ)^(-121:ℤ))*(spRef x0 + (2:ℝ)^(-6:ℤ)) := by
    rw [hdp_def]
    exact harith
  have hfin : (1 + (2:ℝ)^(-21:ℤ))*(Real.exp A * spRef x0 + ((Real.exp A + de)*dp + de*spRef x0))
      ≤ Real.exp A * spRef x0 + (2:ℝ)^(-23:ℤ)*(64 + min |A| 128)*(Real.exp A + (2:ℝ)^(-121:ℤ))*(spRef x0 + (2:ℝ)^(-6:ℤ)) := by
    have hid : (1 + (2:ℝ)^(-21:ℤ))*(Real.exp A * spRef x0 + ((Real.exp A + de)*dp + de*spRef x0))
        = Real.exp A * spRef x0 + ((1 + (2:ℝ)^(-21:ℤ))*((Real.exp A + de)*dp + de*spRef x0) + (2:ℝ)^(-21:ℤ)*Real.exp A*spRef x0) := by ring
    rw [hid]
    linarith only [harith']
  linarith only [hMle, hmul, hfin]

end Claims.GdnGating
