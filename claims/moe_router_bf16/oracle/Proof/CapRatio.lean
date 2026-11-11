import Claim
import TVA.Proof
import Proof.Cap
import Proof.CapReal

/-!
# Real-number lemmas for the cap bound: the ratio `(E − 1)/(E + 1)`

How far the computed `(Ẽ − 1)/(Ẽ + 1)` is from `(e^t − 1)/(e^t + 1)` when `Ẽ` is within the
`ExpF32` accuracy of `e^t`, for `t` in the covered range.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

/-- `cap_abs_diff` for arguments above `−1`. -/
theorem cap_abs_diff' (a b : ℝ) (ha : 0 < a + 1) (hb : 0 < b + 1) :
    |(a - 1) / (a + 1) - (b - 1) / (b + 1)| ≤ 2 * |a - b| / ((a + 1) * (b + 1)) := by
  have hkey : (a - 1) / (a + 1) - (b - 1) / (b + 1)
      = 2 * (a - b) / ((a + 1) * (b + 1)) := by
    have h1 : a + 1 ≠ 0 := ha.ne'
    have h2 : b + 1 ≠ 0 := hb.ne'
    field_simp
    ring
  rw [hkey]
  simp only [abs_div, abs_mul, abs_of_pos ha, abs_of_pos hb,
    abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  exact le_rfl

/-- The triangle inequality along a chain of six points. -/
theorem abs_sub_le_sum5 (a b c d e f : ℝ) :
    |a - f| ≤ |a - b| + |b - c| + |c - d| + |d - e| + |e - f| := by
  have h1 : |a - f| ≤ |a - e| + |e - f| := abs_sub_le a e f
  have h2 : |a - e| ≤ |a - d| + |d - e| := abs_sub_le a d e
  have h3 : |a - d| ≤ |a - c| + |c - d| := abs_sub_le a c d
  have h4 : |a - c| ≤ |a - b| + |b - c| := abs_sub_le a b c
  linarith

-- Sharp exp-error through the ratio
/-- `(a−1)/(a+1)` is within `2^-21 + 2^-39` of `(E−1)/(E+1)` when `a` is within relative `rel ≤ 2^-21` of `E`. -/
theorem ratio_exp_bound {a E rel : ℝ} (ha1 : (0 : ℝ) < a + 1) (hE : 0 < E)
    (hE1 : (0 : ℝ) < E + 1)
    (herr : |a - E| ≤ rel * E + (2 : ℝ) ^ (-140 : ℤ))
    (hrel0 : 0 ≤ rel) (hrel : rel ≤ (2 : ℝ) ^ (-21 : ℤ))
    (hratio : rel * E ≤ (2 : ℝ) ^ (-22 : ℤ) * (E + 1) ^ 2)
    (hlow : (E + 1) * (1 - rel - (2 : ℝ) ^ (-140 : ℤ)) ≤ a + 1) :
    |(a - 1) / (a + 1) - (E - 1) / (E + 1)| ≤ (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) := by
  have hstep := cap_abs_diff' a E ha1 hE1
  set c : ℝ := (2 : ℝ) ^ (-140 : ℤ) with hcdef
  have hcpos : (0 : ℝ) < c := by rw [hcdef]; norm_num
  have hx : rel + c ≤ (2 : ℝ) ^ (-20 : ℤ) := by
    rw [hcdef]
    have h : (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-20 : ℤ) := by norm_num
    linarith
  have hden1 : (0 : ℝ) < 1 - rel - c := by
    have h2' : (2 : ℝ) ^ (-20 : ℤ) < (1 : ℝ) := by norm_num
    linarith
  have hP : (0 : ℝ) < (E + 1) ^ 2 * (1 - rel - c) := by positivity
  have hQ : (0 : ℝ) < (a + 1) * (E + 1) := mul_pos ha1 hE1
  have hdenle : (E + 1) ^ 2 * (1 - rel - c) ≤ (a + 1) * (E + 1) := by
    nlinarith [hlow, hE1]
  have hA : 2 * |a - E| ≤ 2 * (rel * E + c) := by linarith [herr]
  have hnum0 : (0 : ℝ) ≤ 2 * (rel * E + c) := by
    have : (0 : ℝ) ≤ rel * E := mul_nonneg hrel0 hE.le
    nlinarith
  have hstep2 : 2 * |a - E| / ((a + 1) * (E + 1)) ≤
      2 * (rel * E + c) / ((E + 1) ^ 2 * (1 - rel - c)) := by
    rw [div_le_div_iff₀ hQ hP]
    have s1 : 2 * |a - E| * ((E + 1) ^ 2 * (1 - rel - c))
        ≤ 2 * (rel * E + c) * ((E + 1) ^ 2 * (1 - rel - c)) :=
      mul_le_mul_of_nonneg_right hA hP.le
    have s2 : 2 * (rel * E + c) * ((E + 1) ^ 2 * (1 - rel - c))
        ≤ 2 * (rel * E + c) * ((a + 1) * (E + 1)) :=
      mul_le_mul_of_nonneg_left hdenle hnum0
    linarith [s1, s2]
  have hP2 : (0 : ℝ) < (E + 1) ^ 2 := by positivity
  have hsq : (1 : ℝ) ≤ (E + 1) ^ 2 := by nlinarith [hE]
  have hT3 : 2 * (rel * E + c) / ((E + 1) ^ 2 * (1 - rel - c)) ≤
      (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) := by
    rw [div_le_iff₀ hP]
    have h1 : 2 * (rel * E) ≤ (2 : ℝ) ^ (-21 : ℤ) * (E + 1) ^ 2 := by
      have h : (2 : ℝ) ^ (-21 : ℤ) = 2 * (2 : ℝ) ^ (-22 : ℤ) := by norm_num
      nlinarith [hratio]
    have h2 : (rel + c) * (E + 1) ^ 2 ≤ (2 : ℝ) ^ (-20 : ℤ) * (E + 1) ^ 2 :=
      mul_le_mul_of_nonneg_right hx hP2.le
    have hnum1 : (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) -
        ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * (2 : ℝ) ^ (-20 : ℤ) ≥
        (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-40 : ℤ) := by norm_num
    have hnum2 : 2 * c ≤ (2 : ℝ) ^ (-40 : ℤ) := by rw [hcdef]; norm_num
    have hR : ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * ((E + 1) ^ 2 * (1 - rel - c))
        = ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * (E + 1) ^ 2
          - ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * ((rel + c) * (E + 1) ^ 2) := by ring
    have h3 : ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * ((rel + c) * (E + 1) ^ 2)
        ≤ ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * ((2 : ℝ) ^ (-20 : ℤ) * (E + 1) ^ 2) :=
      mul_le_mul_of_nonneg_left h2 (by norm_num)
    have hG : (2 : ℝ) ^ (-21 : ℤ) * (E + 1) ^ 2 + (2 : ℝ) ^ (-40 : ℤ) * (E + 1) ^ 2 ≤
        ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * (E + 1) ^ 2
          - ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * ((2 : ℝ) ^ (-20 : ℤ) * (E + 1) ^ 2) := by
      have hrw : ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * (E + 1) ^ 2
          - ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * ((2 : ℝ) ^ (-20 : ℤ) * (E + 1) ^ 2)
          = (((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) -
              ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) * (2 : ℝ) ^ (-20 : ℤ)) * (E + 1) ^ 2 := by
        ring
      rw [hrw]
      have hmm := mul_le_mul_of_nonneg_right hnum1 (le_of_lt hP2)
      have hsplit : ((2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-40 : ℤ)) * (E + 1) ^ 2
          = (2 : ℝ) ^ (-21 : ℤ) * (E + 1) ^ 2 + (2 : ℝ) ^ (-40 : ℤ) * (E + 1) ^ 2 := by ring
      linarith [hmm, hsplit]
    have hc2 : 2 * c ≤ (2 : ℝ) ^ (-40 : ℤ) * (E + 1) ^ 2 := by
      have h1 : (2 : ℝ) ^ (-40 : ℤ) * 1 ≤ (2 : ℝ) ^ (-40 : ℤ) * (E + 1) ^ 2 :=
        mul_le_mul_of_nonneg_left hsq (by norm_num)
      linarith [hnum2, h1]
    linarith [h1, hG, h3, hR, hc2]
  linarith [hstep, hstep2, hT3]

/-- Rounding the numerator `a−1`, the denominator `a+1` and the quotient moves `(a−1)/(a+1)` by at most `5u`. -/
theorem ratio_op_bound {a T B : ℝ}
    (halow0 : -((2 : ℝ) ^ (-139 : ℤ)) ≤ a)
    (hT : |T - (a - 1)| ≤ |a - 1| * u32 + eta32)
    (hB : |B - (a + 1)| ≤ |a + 1| * u32 + eta32) :
    |T / B - (a - 1) / (a + 1)| ≤ 5 * u32 := by
  have hu : (0 : ℝ) < u32 := u32_pos
  have he : (0 : ℝ) < eta32 := eta32_pos
  have hu0 : (0 : ℝ) ≤ u32 := hu.le
  have he0 : (0 : ℝ) ≤ eta32 := he.le
  have hu16 : u32 ≤ 1 / 16 := u32_le
  have heu : eta32 ≤ u32 * (1 / 1000) := by unfold eta32 u32; norm_num
  have hEpos : (0 : ℝ) < a + 1 := by
    have hlt : -((2 : ℝ) ^ (-139 : ℤ)) > -1 := by norm_num
    linarith
  have hEhalf : (2 : ℝ) ^ (-1 : ℤ) ≤ a + 1 := by
    have h : -((2 : ℝ) ^ (-139 : ℤ)) + 1 ≥ (2 : ℝ) ^ (-1 : ℤ) := by norm_num
    linarith
  have habs : |a + 1| = a + 1 := abs_of_pos hEpos
  have hB2 : (a + 1) * (1 - u32) - eta32 ≤ B := by
    have h := (abs_le.mp hB).1
    rw [habs] at h
    linarith
  have hB0 : (0 : ℝ) < B := by
    have hsub : 1 - 1 / 16 ≤ 1 - u32 := by linarith [hu16]
    have h0 : (0 : ℝ) ≤ 1 - 1 / 16 := by norm_num
    have hprod : (2 : ℝ) ^ (-1 : ℤ) * (1 - 1 / 16) ≤ (a + 1) * (1 - u32) :=
      mul_le_mul hEhalf hsub h0 hEpos.le
    have hnum0 : (0 : ℝ) < (2 : ℝ) ^ (-1 : ℤ) * (1 - 1 / 16) - (1 / 1000) := by norm_num
    have he1 : eta32 ≤ 1 / 1000 := by unfold eta32; norm_num
    linarith [hprod, hnum0, he1, hB2]
  -- uniform |a-1| <= 2(a+1)
  have ha1 : |a - 1| ≤ 2 * (a + 1) := by
    by_cases hle : 1 ≤ a
    · rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ a - 1)]
      linarith
    · have hlt : a < 1 := not_le.mp hle
      rw [abs_of_neg (by linarith : a - 1 < 0)]
      have h3 : 3 * (2 : ℝ) ^ (-139 : ℤ) ≤ 1 := by norm_num
      linarith [halow0, h3]
  -- numerator bound
  have hnum : |T * (a + 1) - (a - 1) * B| ≤
      4 * u32 * ((a + 1) * (a + 1)) + 3 * eta32 * (a + 1) := by
    have h1 : T * (a + 1) - (a - 1) * B
        = (T - (a - 1)) * (a + 1) - (a - 1) * (B - (a + 1)) := by ring
    rw [h1]
    have h2' : |(T - (a - 1)) * (a + 1) - (a - 1) * (B - (a + 1))|
        ≤ |(T - (a - 1)) * (a + 1)| + |(a - 1) * (B - (a + 1))| := by
      have h := abs_sub_le ((T - (a - 1)) * (a + 1)) 0 ((a - 1) * (B - (a + 1)))
      simpa using h
    have h3 : |(T - (a - 1)) * (a + 1)| ≤ (|a - 1| * u32 + eta32) * (a + 1) := by
      rw [abs_mul, habs]
      exact mul_le_mul_of_nonneg_right hT hEpos.le
    have hB' : |B - (a + 1)| ≤ (a + 1) * u32 + eta32 := by rw [habs] at hB; exact hB
    have h4 : |(a - 1) * (B - (a + 1))| ≤ |a - 1| * ((a + 1) * u32 + eta32) := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hB' (abs_nonneg _)
    have h5 : (|a - 1| * u32 + eta32) * (a + 1) ≤ (2 * (a + 1) * u32 + eta32) * (a + 1) := by
      apply mul_le_mul_of_nonneg_right _ hEpos.le
      have hm : |a - 1| * u32 ≤ 2 * (a + 1) * u32 := mul_le_mul_of_nonneg_right ha1 hu0
      linarith [hm]
    have h4' : |a - 1| * ((a + 1) * u32 + eta32) ≤ 2 * (a + 1) * ((a + 1) * u32 + eta32) :=
      mul_le_mul_of_nonneg_right ha1 (by positivity)
    have hexp1 : (2 * (a + 1) * u32 + eta32) * (a + 1)
        = 2 * u32 * ((a + 1) * (a + 1)) + eta32 * (a + 1) := by ring
    have hexp2 : 2 * (a + 1) * ((a + 1) * u32 + eta32)
        = 2 * u32 * ((a + 1) * (a + 1)) + 2 * eta32 * (a + 1) := by ring
    linarith [h2', h3, h5, h4', hexp1, hexp2]
  have hkey : T / B - (a - 1) / (a + 1)
      = (T * (a + 1) - (a - 1) * B) / (B * (a + 1)) := by
    field_simp
  have hdenpos : (0 : ℝ) < B * (a + 1) := mul_pos hB0 hEpos
  rw [hkey, abs_div, abs_of_pos hdenpos, div_le_iff₀ hdenpos]
  -- denominator lower bound
  have hBE : ((a + 1) * (a + 1)) * (1 - u32) - eta32 * (a + 1) ≤ B * (a + 1) := by
    have h := mul_le_mul_of_nonneg_right hB2 hEpos.le
    have hee : ((a + 1) * (1 - u32) - eta32) * (a + 1)
        = ((a + 1) * (a + 1)) * (1 - u32) - eta32 * (a + 1) := by ring
    linarith [h]
  -- numeric comparison
  have hnum2 : 4 * u32 * ((a + 1) * (a + 1)) + 3 * eta32 * (a + 1)
      ≤ 5 * u32 * (((a + 1) * (a + 1)) * (1 - u32) - eta32 * (a + 1)) := by
    have hA0 : (0 : ℝ) ≤ (a + 1) * (a + 1) := by positivity
    have huA0 : (0 : ℝ) ≤ u32 * ((a + 1) * (a + 1)) := by positivity
    -- E <= 2 A
    have h2a : 2 * (2 : ℝ) ^ (-1 : ℤ) = 1 := by norm_num
    have h2E : (0 : ℝ) ≤ 2 * (a + 1) - 1 := by linarith [hEhalf, h2a]
    have hE2 : (a + 1) ≤ 2 * ((a + 1) * (a + 1)) := by
      have hpr := mul_nonneg hEpos.le h2E
      have he : (a + 1) * (2 * (a + 1) - 1) = 2 * ((a + 1) * (a + 1)) - (a + 1) := by ring
      linarith [hpr, he]
    -- u * (A*u) <= (1/16) * (u*A)
    have hsq0 := mul_le_mul_of_nonneg_left hu16 huA0
    have hsq : u32 * (((a + 1) * (a + 1)) * u32) ≤ (1 / 16) * (u32 * ((a + 1) * (a + 1))) := by
      have e1 : u32 * (((a + 1) * (a + 1)) * u32) = (u32 * ((a + 1) * (a + 1))) * u32 := by ring
      have e2 : (1 / 16) * (u32 * ((a + 1) * (a + 1))) = (u32 * ((a + 1) * (a + 1))) * (1 / 16) := by ring
      rw [e1, e2]
      exact hsq0
    -- eta * E <= (2/1000) * (u*A)
    have hEU : eta32 * (a + 1) ≤ (2 / 1000) * (u32 * ((a + 1) * (a + 1))) := by
      have h := mul_le_mul heu hE2 hEpos.le (by positivity)
      have e : (u32 * (1 / 1000)) * (2 * ((a + 1) * (a + 1)))
          = (2 / 1000) * (u32 * ((a + 1) * (a + 1))) := by ring
      linarith [h]
    -- 3 eta E <= (6/1000) u A
    have h3E : 3 * (eta32 * (a + 1)) ≤ (6 / 1000) * (u32 * ((a + 1) * (a + 1))) := by
      have h := mul_le_mul_of_nonneg_left hEU (by norm_num : (0 : ℝ) ≤ 3)
      have e : 3 * ((2 / 1000) * (u32 * ((a + 1) * (a + 1))))
          = (6 / 1000) * (u32 * ((a + 1) * (a + 1))) := by ring
      linarith [h]
    -- 5 u (eta E) <= (10/1000)*(1/16) u A
    have h5uE : 5 * u32 * (eta32 * (a + 1))
        ≤ (10 / 1000 * (1 / 16)) * (u32 * ((a + 1) * (a + 1))) := by
      have h1 : 5 * u32 * (eta32 * (a + 1)) ≤ 5 * u32 * ((2 / 1000) * (u32 * ((a + 1) * (a + 1)))) :=
        mul_le_mul_of_nonneg_left hEU (by positivity)
      have e1 : 5 * u32 * ((2 / 1000) * (u32 * ((a + 1) * (a + 1))))
          = (10 / 1000) * (u32 * (((a + 1) * (a + 1)) * u32)) := by ring
      have h2 : (10 / 1000) * (u32 * (((a + 1) * (a + 1)) * u32))
          ≤ (10 / 1000) * ((1 / 16) * (u32 * ((a + 1) * (a + 1)))) :=
        mul_le_mul_of_nonneg_left hsq (by norm_num)
      have e2 : (10 / 1000) * ((1 / 16) * (u32 * ((a + 1) * (a + 1))))
          = (10 / 1000 * (1 / 16)) * (u32 * ((a + 1) * (a + 1))) := by ring
      linarith [h1, h2]
    -- 5 u^2 A <= (5/16) u A
    have h5u2 : 5 * (u32 * (((a + 1) * (a + 1)) * u32))
        ≤ 5 * ((1 / 16) * (u32 * ((a + 1) * (a + 1)))) :=
      mul_le_mul_of_nonneg_left hsq (by norm_num)
    -- constant comparison
    have hconst : 4 * (u32 * ((a + 1) * (a + 1))) + (6 / 1000) * (u32 * ((a + 1) * (a + 1)))
        + (10 / 1000 * (1 / 16)) * (u32 * ((a + 1) * (a + 1)))
        ≤ (75 / 16) * (u32 * ((a + 1) * (a + 1))) := by
      have hc : (4 : ℝ) + 6 / 1000 + 10 / 1000 * (1 / 16) ≤ 75 / 16 := by norm_num
      have h := mul_le_mul_of_nonneg_right hc huA0
      have e : (4 + 6 / 1000 + 10 / 1000 * (1 / 16)) * (u32 * ((a + 1) * (a + 1)))
          = 4 * (u32 * ((a + 1) * (a + 1))) + (6 / 1000) * (u32 * ((a + 1) * (a + 1)))
            + (10 / 1000 * (1 / 16)) * (u32 * ((a + 1) * (a + 1))) := by ring
      linarith [h]
    have eL : 5 * u32 * (((a + 1) * (a + 1)) * (1 - u32) - eta32 * (a + 1))
        = 5 * (u32 * ((a + 1) * (a + 1))) - 5 * (u32 * (((a + 1) * (a + 1)) * u32))
          - 5 * u32 * (eta32 * (a + 1)) := by ring
    have eT : 4 * u32 * ((a + 1) * (a + 1)) + 3 * eta32 * (a + 1)
        = 4 * (u32 * ((a + 1) * (a + 1))) + 3 * (eta32 * (a + 1)) := by ring
    linarith [h3E, h5uE, h5u2, hconst, eL, eT]
  have h5u : (0 : ℝ) ≤ 5 * u32 := by positivity
  have hfin : 5 * u32 * (((a + 1) * (a + 1)) * (1 - u32) - eta32 * (a + 1))
      ≤ 5 * u32 * (B * (a + 1)) := mul_le_mul_of_nonneg_left hBE h5u
  linarith [hnum, hnum2, hfin]

-- 15 tau <= 2^-18 S + 2^-20
/-- The input error of the cap, `15τ` (the scaled logit tolerance and the rounding of `l/30` and `2s`), is at most `2^-18·S + 2^-20`. -/
theorem tau_bound {M L S : ℝ} (hS0 : 0 ≤ S) (hMS : |M| ≤ S)
    (hLle : L ≤ 40.0002 * u32 * S + (2 : ℝ) ^ (-127 : ℤ)) :
    15 * (L / 15 + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * ((|M| + L) / 15) +
      (2 : ℝ) ^ (-140 : ℤ)) ≤ (2 : ℝ) ^ (-18 : ℤ) * S + (2 : ℝ) ^ (-20 : ℤ) := by
  have hu : (0 : ℝ) < u32 := u32_pos
  have hu24 : u32 = (2 : ℝ) ^ (-24 : ℤ) := rfl
  have hc : (0 : ℝ) ≤ u32 + (2 : ℝ) ^ (-21 : ℤ) := by positivity
  have hexp : 15 * (L / 15 + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * ((|M| + L) / 15) +
      (2 : ℝ) ^ (-140 : ℤ)) = L + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (|M| + L) + 15 * (2 : ℝ) ^ (-140 : ℤ) := by
    ring
  rw [hexp]
  have h1 : |M| + L ≤ S + L := by linarith [hMS]
  have h2 : (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (|M| + L) ≤ (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (S + L) :=
    mul_le_mul_of_nonneg_left h1 hc
  have hLb : (u32 + (2 : ℝ) ^ (-21 : ℤ)) * L
      ≤ (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ)) * S
        + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ)) := by
    have h := mul_le_mul_of_nonneg_left hLle hc
    have hee : (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (40.0002 * u32 * S + (2 : ℝ) ^ (-127 : ℤ))
        = (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ)) * S
          + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ)) := by ring
    linarith [h]
  have hcoef : 40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
      + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ)) ≤ (2 : ℝ) ^ (-18 : ℤ) := by
    rw [hu24]; norm_num
  have hconst : (2 : ℝ) ^ (-127 : ℤ) + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ))
      + 15 * (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-20 : ℤ) := by
    rw [hu24]; norm_num
  have hstep1 : (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (|M| + L)
      ≤ (u32 + (2 : ℝ) ^ (-21 : ℤ)) * S + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * L := by
    have e : (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (S + L)
        = (u32 + (2 : ℝ) ^ (-21 : ℤ)) * S + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * L := by ring
    rw [← e]
    exact h2
  have hcomb : L + ((u32 + (2 : ℝ) ^ (-21 : ℤ)) * S + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * L)
      + 15 * (2 : ℝ) ^ (-140 : ℤ)
      ≤ (40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
          + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ))) * S
        + ((2 : ℝ) ^ (-127 : ℤ)
          + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ)))
        + 15 * (2 : ℝ) ^ (-140 : ℤ) := by
    have e1 : (40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
          + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ))) * S
        = 40.0002 * u32 * S + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * S
          + ((40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ)) * S) := by ring
    rw [e1]
    linarith [hLle, hLb]
  have hcoefS : (40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
      + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ))) * S
      ≤ (2 : ℝ) ^ (-18 : ℤ) * S := mul_le_mul_of_nonneg_right hcoef hS0
  clear hexp h1 h2 hc hu hu24 hLb hcoef hLle hMS hS0
  have hstep1' : L + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * (|M| + L) + 15 * (2 : ℝ) ^ (-140 : ℤ)
      ≤ L + ((u32 + (2 : ℝ) ^ (-21 : ℤ)) * S + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * L)
        + 15 * (2 : ℝ) ^ (-140 : ℤ) := by
    linarith [hstep1]
  have htail : (40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
          + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ))) * S
        + ((2 : ℝ) ^ (-127 : ℤ)
          + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ)))
        + 15 * (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-18 : ℤ) * S + (2 : ℝ) ^ (-20 : ℤ) := by
    have hsum := add_le_add hcoefS hconst
    have e : (40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
          + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ))) * S
        + ((2 : ℝ) ^ (-127 : ℤ)
          + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ)))
        + 15 * (2 : ℝ) ^ (-140 : ℤ)
        = ((40.0002 * u32 + (u32 + (2 : ℝ) ^ (-21 : ℤ))
            + (40.0002 * u32 * u32 + 40.0002 * u32 * (2 : ℝ) ^ (-21 : ℤ))) * S)
          + (((2 : ℝ) ^ (-127 : ℤ)
            + (u32 * (2 : ℝ) ^ (-127 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) * (2 : ℝ) ^ (-127 : ℤ)))
            + 15 * (2 : ℝ) ^ (-140 : ℤ)) := by ring
    rw [e]
    exact hsum
  exact le_trans (le_trans hstep1' hcomb) htail

-- intermediate: D <= 88.001
/-- The doubled scaled logit is at most `88.001` when the capped logit bound `μ+capTol ≤ 1320` holds. -/
theorem db_le {Al M L D : ℝ} (hA : |Al - M| ≤ L)
    (hD : |D - Al / 15| ≤ (2 : ℝ) ^ (-21 : ℤ) * |Al| / 15 + (2 : ℝ) ^ (-145 : ℤ))
    (hcapML : (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L) ≤ 1320) : D ≤ 88.001 := by
  have hAl : Al ≤ M + L := by
    have h := (abs_le.mp hA).2
    linarith
  have hD' : D ≤ Al / 15 + (2 : ℝ) ^ (-21 : ℤ) * |Al| / 15 + (2 : ℝ) ^ (-145 : ℤ) := by
    have h := le_abs_self (D - Al / 15)
    linarith
  have htiny : (2 : ℝ) ^ (-145 : ℤ) ≤ 1 / 1000 := by norm_num
  by_cases h0 : 0 ≤ Al
  · have habs : |Al| = Al := abs_of_nonneg h0
    rw [habs] at hD'
    have h1 : Al / 15 + (2 : ℝ) ^ (-21 : ℤ) * Al / 15
        = (1 + (2 : ℝ) ^ (-21 : ℤ)) * Al / 15 := by ring
    have h2 : (1 + (2 : ℝ) ^ (-21 : ℤ)) * Al / 15
        ≤ (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L) / 15 :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hAl (by norm_num)) (by norm_num)
    have h13 : (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L) / 15 ≤ 88 := by
      have h : (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L) / 15 ≤ 1320 / 15 :=
        div_le_div_of_nonneg_right hcapML (by norm_num)
      linarith
    linarith [hD', h1, h2, h13, htiny]
  · have hneg : Al < 0 := not_le.mp h0
    have habs : |Al| = -Al := abs_of_neg hneg
    rw [habs] at hD'
    have hneg2 : Al / 15 + (2 : ℝ) ^ (-21 : ℤ) * (-Al) / 15 ≤ 0 := by
      have h : (0 : ℝ) ≤ (2 : ℝ) ^ (-21 : ℤ) := by norm_num
      nlinarith
    linarith [hD', hneg2, htiny]

/-- The computed ratio `T/B` has magnitude at most `1 + 5u + 2^-136`. -/
theorem ratio_abs_bound {a T B : ℝ} (halow0 : -((2 : ℝ) ^ (-139 : ℤ)) ≤ a)
    (hT : |T - (a - 1)| ≤ |a - 1| * u32 + eta32)
    (hB : |B - (a + 1)| ≤ |a + 1| * u32 + eta32) :
    |T / B| ≤ 1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ) := by
  have hu : u32 = (2 : ℝ) ^ (-24 : ℤ) := rfl
  have he : eta32 = (2 : ℝ) ^ (-150 : ℤ) := rfl
  have hA : (1 / 2 : ℝ) ≤ a + 1 := by
    have h140 : ((2 : ℝ) ^ (-139 : ℤ)) ≤ 1 / 2 := by norm_num
    have h2 : -(1 / 2 : ℝ) ≤ -((2 : ℝ) ^ (-139 : ℤ)) := neg_le_neg h140
    have h3 : -(1 / 2 : ℝ) ≤ a := le_trans h2 halow0
    have h4 : -(1 / 2 : ℝ) + 1 ≤ a + 1 := add_le_add_left h3 1
    have h5 : (-(1 / 2 : ℝ) + 1) = (1 / 2 : ℝ) := by norm_num
    rwa [h5] at h4
  have hA0 : (0 : ℝ) < a + 1 := by linarith
  have habs : |a - 1| ≤ (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) := by
    have h3 : (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ))
        = a + 1 + (a + 1) * (2 : ℝ) ^ (-137 : ℤ) := by ring
    rw [h3]
    rcases le_or_gt 1 a with h1 | h1
    · have h2 : |a - 1| = a - 1 := abs_of_nonneg (by linarith)
      rw [h2]
      have h4 : (0 : ℝ) ≤ (a + 1) * (2 : ℝ) ^ (-137 : ℤ) := by positivity
      linarith
    · have h2 : |a - 1| = 1 - a := by rw [abs_of_neg (by linarith)]; ring
      rw [h2]
      have hneg : -a ≤ (2 : ℝ) ^ (-139 : ℤ) := by
        simpa using neg_le_neg halow0
      have h8 : -2 * a ≤ (2 : ℝ) ^ (-138 : ℤ) := by
        have h9 := mul_le_mul_of_nonneg_left hneg (by norm_num : (0 : ℝ) ≤ 2)
        have h9' : (2 : ℝ) * -a ≤ (2 : ℝ) * (2 : ℝ) ^ (-139 : ℤ) := h9
        have h10 : (2 : ℝ) * (2 : ℝ) ^ (-139 : ℤ) = (2 : ℝ) ^ (-138 : ℤ) := by norm_num
        have h11 : (2 : ℝ) * -a = -2 * a := by ring
        linarith only [h9', h10, h11]
      have h10 : (2 : ℝ) ^ (-138 : ℤ) ≤ (a + 1) * (2 : ℝ) ^ (-137 : ℤ) := by
        have h11 := mul_le_mul_of_nonneg_right hA (by norm_num : (0 : ℝ) ≤ (2 : ℝ) ^ (-137 : ℤ))
        have h12 : (1 / 2 : ℝ) * (2 : ℝ) ^ (-137 : ℤ) = (2 : ℝ) ^ (-138 : ℤ) := by norm_num
        rw [h12] at h11
        exact h11
      have h13 : -2 * a ≤ (a + 1) * (2 : ℝ) ^ (-137 : ℤ) := le_trans h8 h10
      linarith only [h13]
  have hBpos : (0 : ℝ) < B := by
    have h1 := (abs_le.mp hB).1
    have h2 : |a + 1| = a + 1 := abs_of_pos hA0
    rw [h2] at h1
    have h3 : (a + 1) * u32 ≤ (a + 1) * (1 / 16) := mul_le_mul_of_nonneg_left u32_le hA0.le
    have hη : eta32 ≤ 1 / 16 := by rw [he]; norm_num
    linarith
  have hTabs : |T| ≤ (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + eta32 := by
    have h1 : |T| ≤ |T - (a - 1)| + |a - 1| := by
      simpa only [sub_add_cancel] using abs_add_le (T - (a - 1)) (a - 1)
    have h4 : |a - 1| * u32 + eta32 + |a - 1| ≤
        (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + eta32 := by
      have h5 : |a - 1| * (1 + u32)
          ≤ (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) := by
        apply mul_le_mul_of_nonneg_right habs
        have : (0 : ℝ) < u32 := u32_pos
        linarith
      have h6 : |a - 1| * u32 + eta32 + |a - 1| = |a - 1| * (1 + u32) + eta32 := by ring
      rw [h6]; linarith
    linarith
  have h3eta : eta32 ≤ (a + 1) * (2 : ℝ) ^ (-149 : ℤ) := by
    rw [he]
    have h4 : (0 : ℝ) < (2 : ℝ) ^ (-149 : ℤ) := by norm_num
    have h5 := mul_le_mul_of_nonneg_right hA h4.le
    have h6 : (1 / 2 : ℝ) * (2 : ℝ) ^ (-149 : ℤ) = (2 : ℝ) ^ (-150 : ℤ) := by norm_num
    calc (2 : ℝ) ^ (-150 : ℤ) = (1 / 2 : ℝ) * (2 : ℝ) ^ (-149 : ℤ) := h6.symm
      _ ≤ (a + 1) * (2 : ℝ) ^ (-149 : ℤ) := h5
  have hD : (a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ)) ≤ B := by
    have h1 := (abs_le.mp hB).1
    have h2 : |a + 1| = a + 1 := abs_of_pos hA0
    rw [h2] at h1
    have h8 : (a + 1) - ((a + 1) * u32 + eta32) ≤ B := by linarith only [h1]
    have h9 : (a + 1) - ((a + 1) * u32 + (a + 1) * (2 : ℝ) ^ (-149 : ℤ))
        ≤ (a + 1) - ((a + 1) * u32 + eta32) :=
      sub_le_sub_left (add_le_add_right h3eta _) (a + 1)
    have h7 : (a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ))
        = (a + 1) - (a + 1) * u32 - (a + 1) * (2 : ℝ) ^ (-149 : ℤ) := by ring
    rw [h7]
    calc (a + 1) - (a + 1) * u32 - (a + 1) * (2 : ℝ) ^ (-149 : ℤ)
        = (a + 1) - ((a + 1) * u32 + (a + 1) * (2 : ℝ) ^ (-149 : ℤ)) := by ring
      _ ≤ (a + 1) - ((a + 1) * u32 + eta32) := h9
      _ ≤ B := h8
  have hDpos : (0 : ℝ) < (a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ)) := by
    have h3 : (0 : ℝ) < 1 - u32 - (2 : ℝ) ^ (-149 : ℤ) := by
      have h6 : (2 : ℝ) ^ (-149 : ℤ) ≤ 1 / 16 := by norm_num
      have hsum : u32 + (2 : ℝ) ^ (-149 : ℤ) ≤ 1 / 16 + 1 / 16 := add_le_add u32_le h6
      have h8 : (1 / 16 : ℝ) + 1 / 16 = 1 / 8 := by norm_num
      rw [h8] at hsum
      have e : (1 : ℝ) - u32 - (2 : ℝ) ^ (-149 : ℤ) = 1 - (u32 + (2 : ℝ) ^ (-149 : ℤ)) := by ring
      rw [e]
      exact sub_pos.mpr (lt_of_le_of_lt hsum (by norm_num))
    exact mul_pos hA0 h3
  have hN : (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + eta32
      ≤ (a + 1) * ((1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + (2 : ℝ) ^ (-149 : ℤ)) := by
    have h1 : (a + 1) * ((1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + (2 : ℝ) ^ (-149 : ℤ))
        = (a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32)
          + (a + 1) * (2 : ℝ) ^ (-149 : ℤ) := by ring
    rw [h1]
    exact add_le_add_right h3eta _
  have hstep2 : ((a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + eta32) /
      ((a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ)))
      ≤ ((a + 1) * ((1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + (2 : ℝ) ^ (-149 : ℤ))) /
        ((a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ))) :=
    div_le_div_of_nonneg_right hN hDpos.le
  have hstep3 : ((a + 1) * ((1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + (2 : ℝ) ^ (-149 : ℤ))) /
      ((a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ)))
      = ((1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + (2 : ℝ) ^ (-149 : ℤ)) /
        (1 - u32 - (2 : ℝ) ^ (-149 : ℤ)) := by
    rw [mul_div_mul_left _ _ hA0.ne']
  have hfinal : ((1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + (2 : ℝ) ^ (-149 : ℤ)) /
      (1 - u32 - (2 : ℝ) ^ (-149 : ℤ)) ≤ 1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ) := by
    rw [hu]
    rw [div_le_iff₀ (by norm_num)]
    norm_num
  have hstep1 : |T| / B ≤ ((a + 1) * (1 + (2 : ℝ) ^ (-137 : ℤ)) * (1 + u32) + eta32) /
      ((a + 1) * (1 - u32 - (2 : ℝ) ^ (-149 : ℤ))) :=
    div_le_div₀ (by positivity) hTabs hDpos hD
  rw [abs_div, abs_of_pos hBpos]
  exact le_trans (le_trans hstep1 hstep2) (le_trans (le_of_eq hstep3) hfinal)

/-- `(a−1)/(a+1)` is within the `ExpF32` slope bound of `(eᵈ−1)/(eᵈ+1)` when `a` approximates `eᵈ`. -/
theorem exp_ratio_err {a d : ℝ} (halow0 : -((2 : ℝ) ^ (-139 : ℤ)) ≤ a) (hd : d ≤ 88.001)
    (herr : |a - Real.exp d| ≤
      ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d + (2 : ℝ) ^ (-140 : ℤ)) :
    |(a - 1) / (a + 1) - (Real.exp d - 1) / (Real.exp d + 1)|
      ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) := by
  set E := Real.exp d with hE
  have hEpos : 0 < E := by rw [hE]; exact Real.exp_pos d
  have hE1 : 0 < E + 1 := by linarith
  have hrel0 : (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d| := by
    have h1 : (0 : ℝ) < (2 : ℝ) ^ (-22 : ℤ) := by norm_num
    have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) * |d| := by positivity
    linarith
  have herr' : |a - E| ≤ ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
      + (2 : ℝ) ^ (-140 : ℤ) := by rw [hE]; exact herr
  have hA0 : (0 : ℝ) < a + 1 := by
    have h140 : ((2 : ℝ) ^ (-139 : ℤ)) < 1 := by norm_num
    have h2 : -(1 : ℝ) < -((2 : ℝ) ^ (-139 : ℤ)) := neg_lt_neg h140
    have h3 : -(1 : ℝ) < a := lt_of_lt_of_le h2 halow0
    have h4 : -(1 : ℝ) + 1 < a + 1 := add_lt_add_left h3 1
    have h5 : (-(1 : ℝ) + 1) = (0 : ℝ) := by norm_num
    rwa [h5] at h4
  have htwo : (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)
      ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) := by norm_num
  rcases le_or_gt |d| 2 with hd2 | hd2
  · have hrel21 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d| ≤ (2 : ℝ) ^ (-21 : ℤ) := by
      have h1 : (2 : ℝ) ^ (-23 : ℤ) * |d| ≤ (2 : ℝ) ^ (-23 : ℤ) * 2 :=
        mul_le_mul_of_nonneg_left hd2 (by norm_num)
      have h2 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 2 = (2 : ℝ) ^ (-21 : ℤ) := by norm_num
      linarith
    have hratio : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
        ≤ (2 : ℝ) ^ (-22 : ℤ) * (E + 1) ^ 2 := by
      have h1 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          ≤ (2 : ℝ) ^ (-21 : ℤ) * E := mul_le_mul_of_nonneg_right hrel21 hEpos.le
      have h2 : (2 : ℝ) ^ (-21 : ℤ) * E ≤ (2 : ℝ) ^ (-20 : ℤ) * E :=
        mul_le_mul_of_nonneg_right (by norm_num : (2 : ℝ) ^ (-21 : ℤ) ≤ (2 : ℝ) ^ (-20 : ℤ)) hEpos.le
      have h4 : (2 : ℝ) ^ (-22 : ℤ) * (E + 1) ^ 2
          = (2 : ℝ) ^ (-20 : ℤ) * E + (2 : ℝ) ^ (-22 : ℤ) * (E - 1) ^ 2 := by ring
      have h5 : (2 : ℝ) ^ (-20 : ℤ) * E ≤ (2 : ℝ) ^ (-22 : ℤ) * (E + 1) ^ 2 := by
        rw [h4]
        exact le_add_of_nonneg_right (by positivity)
      exact le_trans h1 (le_trans h2 h5)
    have hlowd : (E + 1) * (1 - ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)
        - (2 : ℝ) ^ (-140 : ℤ)) ≤ a + 1 := by
      have h1 : E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ)) ≤ a := by
        have h := (abs_le.mp herr').1
        linarith only [h]
      have h2 : (E + 1) * (1 - ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)
          - (2 : ℝ) ^ (-140 : ℤ))
          = (E + 1) - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E + 1)
            + (2 : ℝ) ^ (-140 : ℤ) * (E + 1)) := by ring
      have h3 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)
          ≤ ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E + 1)
            + (2 : ℝ) ^ (-140 : ℤ) * (E + 1) := by
        have he : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E + 1)
            + (2 : ℝ) ^ (-140 : ℤ) * (E + 1)
            = (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
              + (2 : ℝ) ^ (-140 : ℤ))
              + (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)
                + (2 : ℝ) ^ (-140 : ℤ) * E) := by ring
        rw [he]
        exact le_add_of_nonneg_right
          (add_nonneg hrel0 (mul_nonneg (by norm_num) hEpos.le))
      have h4 : (E + 1) - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E + 1)
            + (2 : ℝ) ^ (-140 : ℤ) * (E + 1))
          ≤ (E + 1) - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) := sub_le_sub_left h3 (E + 1)
      calc (E + 1) * (1 - ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)
            - (2 : ℝ) ^ (-140 : ℤ))
          = (E + 1) - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E + 1)
            + (2 : ℝ) ^ (-140 : ℤ) * (E + 1)) := h2
        _ ≤ (E + 1) - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) := h4
        _ = (E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ))) + 1 := by ring
        _ ≤ a + 1 := add_le_add_left h1 1
    exact le_trans (ratio_exp_bound hA0 hEpos hE1 herr' hrel0 hrel21 hratio hlowd) htwo
  · rcases le_or_gt 0 d with hd0 | hdneg
    · have hdge : 2 < d := by rw [abs_of_nonneg hd0] at hd2; exact hd2
      have hrel_half : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d| ≤ 1 / 2 := by
        have h1 : (2 : ℝ) ^ (-23 : ℤ) * |d| ≤ (2 : ℝ) ^ (-23 : ℤ) * 88.001 :=
          mul_le_mul_of_nonneg_left (by rw [abs_of_nonneg hd0]; exact hd) (by norm_num)
        have h3 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 88.001 ≤ 1 / 2 := by norm_num
        norm_num at h1 h3 ⊢
        linarith
      have hN : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E + (2 : ℝ) ^ (-140 : ℤ)
          ≤ (E + 1) / 2 := by
        have h2 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E ≤ (1 / 2 : ℝ) * E :=
          mul_le_mul_of_nonneg_right hrel_half hEpos.le
        have h3 : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 / 2 := by norm_num
        have h4 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E + (2 : ℝ) ^ (-140 : ℤ)
            ≤ (1 / 2 : ℝ) * E + 1 / 2 := add_le_add h2 h3
        have h5 : (1 / 2 : ℝ) * E + 1 / 2 = (E + 1) / 2 := by ring
        rw [h5] at h4
        exact h4
      have hA2 : (E + 1) / 2 ≤ a + 1 := by
        have h1 : E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) ≤ a := by
          have h := (abs_le.mp herr').1
          linarith only [h]
        have h2 : (E + 1) / 2 ≤ E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) + 1 := by linarith only [hN]
        exact le_trans h2 (add_le_add_left h1 1)
      have hfrac : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E / (E + 1) ^ 2)
          ≤ (2 : ℝ) ^ (-22 : ℤ) := by
        have h1 : E / (E + 1) ^ 2 ≤ Real.exp (-|d|) := by rw [hE]; exact exp_ratio_le d
        have h2 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E / (E + 1) ^ 2)
            ≤ ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp (-|d|) :=
          mul_le_mul_of_nonneg_left h1 hrel0
        have h3 := exp_smul_le (abs_nonneg d)
        exact le_trans h2 h3
      have hN0 : (0 : ℝ) ≤ ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ) := by
        have h1 : (0 : ℝ) ≤ ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E :=
          mul_nonneg hrel0 hEpos.le
        have h2 : (0 : ℝ) < (2 : ℝ) ^ (-140 : ℤ) := by norm_num
        linarith
      have hD'pos : (0 : ℝ) < (E + 1) ^ 2 / 2 := by positivity
      have hD'le : (E + 1) ^ 2 / 2 ≤ (a + 1) * (E + 1) := by
        nlinarith [hA2, hE1.le]
      have hstep2 : 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ)) / ((a + 1) * (E + 1))
          ≤ 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) / ((E + 1) ^ 2 / 2) :=
        div_le_div₀ (by positivity) le_rfl hD'pos hD'le
      have hstep3 : 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ)) / ((E + 1) ^ 2 / 2)
          = 4 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E / (E + 1) ^ 2)
            + (2 : ℝ) ^ (-140 : ℤ) / (E + 1) ^ 2) := by
        field_simp
        ring
      have hstep4 : 4 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * (E / (E + 1) ^ 2)
          + (2 : ℝ) ^ (-140 : ℤ) / (E + 1) ^ 2)
          ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) := by
        have h1 : (2 : ℝ) ^ (-140 : ℤ) / (E + 1) ^ 2 ≤ (2 : ℝ) ^ (-140 : ℤ) := by
          rw [div_le_iff₀ (by positivity)]
          have hsq : (1 : ℝ) ≤ (E + 1) ^ 2 := by nlinarith [hEpos.le]
          calc (2 : ℝ) ^ (-140 : ℤ) = (2 : ℝ) ^ (-140 : ℤ) * 1 := (mul_one _).symm
            _ ≤ (2 : ℝ) ^ (-140 : ℤ) * (E + 1) ^ 2 :=
                mul_le_mul_of_nonneg_left hsq (by norm_num)
        have h2 : 4 * (2 : ℝ) ^ (-22 : ℤ) = (2 : ℝ) ^ (-20 : ℤ) := by norm_num
        have h3 : 4 * (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-39 : ℤ) := by norm_num
        linarith [hfrac, h1, h2, h3]
      have hstep1 : 2 * |a - E| / ((a + 1) * (E + 1))
          ≤ 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) / ((a + 1) * (E + 1)) := by
        apply div_le_div_of_nonneg_right _ (by positivity)
        linarith [herr']
      exact le_trans (cap_abs_diff' a E hA0 hE1)
        (le_trans hstep1 (le_trans hstep2 (le_trans (le_of_eq hstep3) hstep4)))
    · have hdle : d < -2 := by
        have h1 := hd2
        rw [abs_of_neg hdneg] at h1
        linarith
      have hrelE : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E ≤ (2 : ℝ) ^ (-22 : ℤ) := by
        have h1 : Real.exp (-|d|) = E := by
          rw [abs_of_neg hdneg, neg_neg, hE]
        have h2 := exp_smul_le (abs_nonneg d)
        rwa [h1] at h2
      have hA2 : (1 : ℝ) / 2 ≤ a + 1 := by
        have h1 : E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) ≤ a := by
          have h := (abs_le.mp herr').1
          linarith only [h]
        have h5 : -(((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E)
            ≤ E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E) := by
          linarith only [hEpos.le]
        have h6 : -(((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E)
              - (2 : ℝ) ^ (-140 : ℤ)
            ≤ (E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E))
              - (2 : ℝ) ^ (-140 : ℤ) := sub_le_sub_right h5 _
        have h7 : -((2 : ℝ) ^ (-22 : ℤ)) - (2 : ℝ) ^ (-140 : ℤ)
            ≤ -(((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E)
              - (2 : ℝ) ^ (-140 : ℤ) := sub_le_sub_right (neg_le_neg hrelE) _
        have h8 : -((2 : ℝ) ^ (-22 : ℤ)) - (2 : ℝ) ^ (-140 : ℤ)
            ≤ (E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E))
              - (2 : ℝ) ^ (-140 : ℤ) := le_trans h7 h6
        have he : E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
              + (2 : ℝ) ^ (-140 : ℤ))
            = (E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E))
              - (2 : ℝ) ^ (-140 : ℤ) := by ring
        have h1' : (E - (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E))
              - (2 : ℝ) ^ (-140 : ℤ) ≤ a := by rw [← he]; exact h1
        have h9 : -((2 : ℝ) ^ (-22 : ℤ)) - (2 : ℝ) ^ (-140 : ℤ) ≥ -(1 / 4 : ℝ) := by norm_num
        have h10 : -(1 / 4 : ℝ) ≤ a := le_trans h9 (le_trans h8 h1')
        linarith only [h10]
      have hstep1 : 2 * |a - E| / ((a + 1) * (E + 1))
          ≤ 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) / ((a + 1) * (E + 1)) := by
        apply div_le_div_of_nonneg_right _ (by positivity)
        linarith [herr']
      have hstep2 : 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ)) / ((a + 1) * (E + 1))
          ≤ 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) / (1 / 2) := by
        apply div_le_div₀ (by positivity) le_rfl (by norm_num)
        nlinarith [hA2, hE1.le]
      have hstep3 : 2 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ)) / (1 / 2)
          = 4 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
            + (2 : ℝ) ^ (-140 : ℤ)) := by ring
      have hstep4 : 4 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
          + (2 : ℝ) ^ (-140 : ℤ)) ≤ (2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ) := by
        have h4 : 4 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E)
            ≤ (2 : ℝ) ^ (-20 : ℤ) := by
          have h5 := mul_le_mul_of_nonneg_left hrelE (by norm_num : (0 : ℝ) ≤ 4)
          have h6 : (4 : ℝ) * (2 : ℝ) ^ (-22 : ℤ) = (2 : ℝ) ^ (-20 : ℤ) := by norm_num
          rwa [h6] at h5
        have h7 : 4 * (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-39 : ℤ) := by norm_num
        have h8 : (4 : ℝ) * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E
              + (2 : ℝ) ^ (-140 : ℤ))
            = 4 * (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * E)
              + 4 * (2 : ℝ) ^ (-140 : ℤ) := by ring
        rw [h8]
        exact add_le_add h4 h7
      exact le_trans (cap_abs_diff' a E hA0 hE1)
        (le_trans hstep1 (le_trans hstep2 (le_trans (le_of_eq hstep3) hstep4)))

/-- The `exp` result stays well below the largest finite float. -/
theorem exp_err_abs_le {a d : ℝ} (hd : d ≤ 88.001)
    (herr : |a - Real.exp d| ≤
      ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d + (2 : ℝ) ^ (-140 : ℤ)) :
    |a| ≤ (2 : ℝ) ^ 127 := by
  have hEpos : (0 : ℝ) < Real.exp d := Real.exp_pos d
  have ha : a ≤ Real.exp d + (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
      + (2 : ℝ) ^ (-140 : ℤ)) := by
    have := (abs_le.mp herr).2; linarith
  have hb : -a ≤ Real.exp d + (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
      + (2 : ℝ) ^ (-140 : ℤ)) := by
    have := (abs_le.mp herr).1; linarith
  have hsum : Real.exp d + (((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d
      + (2 : ℝ) ^ (-140 : ℤ))
      = (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)) * Real.exp d
        + (2 : ℝ) ^ (-140 : ℤ) := by ring
  have hbound : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)) * Real.exp d
      + (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ 127 := by
    by_cases h0 : 0 ≤ d
    · have habs : |d| ≤ 88.001 := by rw [abs_of_nonneg h0]; exact hd
      have hrel2 : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)) ≤ 101 / 100 := by
        have h2 : (2 : ℝ) ^ (-23 : ℤ) * |d| ≤ (2 : ℝ) ^ (-23 : ℤ) * 88.001 :=
          mul_le_mul_of_nonneg_left habs (by norm_num)
        have h3 : (1 : ℝ) + (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 88.001 ≤ 101 / 100 := by
          norm_num
        linarith
      have hE : Real.exp d ≤ (2 : ℝ) ^ 127 * (99 / 100) :=
        le_trans (Real.exp_le_exp.mpr hd) exp_88_le
      have hmul : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)) * Real.exp d
          ≤ (101 / 100) * ((2 : ℝ) ^ 127 * (99 / 100)) :=
        mul_le_mul hrel2 hE hEpos.le (by norm_num)
      have hnum : (101 / 100) * ((2 : ℝ) ^ 127 * (99 / 100))
          + (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ 127 := by norm_num
      exact le_trans (add_le_add_left hmul _) hnum
    · have hneg : d ≤ 0 := le_of_lt (not_le.mp h0)
      have hE1 : Real.exp d ≤ 1 := by
        rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr hneg
      have hsm := exp_smul_le (L := -d) (by linarith)
      rw [show -(-d) = d by ring, ← abs_of_nonpos hneg] at hsm
      have h1 : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)) * Real.exp d
          ≤ 1 + (2 : ℝ) ^ (-22 : ℤ) := by
        have e : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|)) * Real.exp d
            = Real.exp d + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d|) * Real.exp d := by
          ring
        rw [e]; linarith
      have hnum : (1 : ℝ) + (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ 127 := by
        norm_num
      exact le_trans (add_le_add_left h1 _) hnum
  have ha' : a ≤ (2 : ℝ) ^ 127 := by
    rw [hsum] at ha
    linarith
  have hb' : -((2 : ℝ) ^ 127) ≤ a := by
    rw [hsum] at hb
    linarith
  exact abs_le.mpr ⟨hb', ha'⟩

end Claims.MoeRouter
