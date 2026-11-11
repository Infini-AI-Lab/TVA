import Claim
import TVA.Proof

/-!
# The value stored in `beta_output`, in the reals

`nb = -b0` (the negation is exact), `eb ~ exp nb`, `den = fl(1 + eb)`, `q ~ 1 / den` (the `/` of
`tl.sigmoid`, `Num.DivFull`), then the `bfloat16` rounding `q16 ~ q` and the exact widening back
to `float32`.  The stored value is bounded against `sigRef b0 = 1/(1 + e^-b0)` by
`bTol b0 = 2^-7*|sigRef b0| + 2^-133`.

`hnb_ub : nb <= 90` is what `ArgRoom`-style finiteness of `exp nb` gives us: `e^nb` must fit in
`float32`, so `nb` is at most `128*log 2 < 89`.  (For `nb <= 0` the relative error of `exp` is
harmless even for huge `|nb|`, because `t*e^-t <= 1`; only the positive side needs the bound.)
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-- Two positive reals within `c` of each other have reciprocals within `c / (a * b)`. -/
lemma abs_inv_sub_inv_le {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (h : |a - b| ≤ c) :
    |1 / a - 1 / b| ≤ c / (a * b) := by
  have hpos : 0 < a * b := mul_pos ha hb
  have heq : 1 / a - 1 / b = (b - a) / (a * b) := by field_simp
  rw [heq, abs_div, abs_of_pos hpos, abs_sub_comm b a]
  rw [div_le_iff₀ hpos]
  rw [show c / (a * b) * (a * b) = c from by rw [div_mul_cancel₀ c (ne_of_gt hpos)]]
  exact h

/-- **The sigmoid chain**: from the bounds on `tl.exp(-b)`, the sum `1 + ·`, the full-range division of `1` and the `bfloat16` rounding, the stored `q16` is within `bTol b0` of `sigRef b0`. -/
theorem beta_bound {b0 eb nb den q q16 : ℝ}
    (hnb : nb = -b0)
    (hnb_ub : nb ≤ 90)
    (heb : |eb - Real.exp nb| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |nb|) * Real.exp nb + 2 ^ (-140 : ℤ))
    (hden : |den - (1 + eb)| ≤ 2 ^ (-24 : ℤ) * (1 + eb) + 2 ^ (-150 : ℤ))
    (hq : |q - 1 / den| ≤ 2 ^ (-22 : ℤ) * |1 / den| + 2 ^ (-148 : ℤ))
    (hq16 : |q16 - q| ≤ 2 ^ (-8 : ℤ) * |q| + 2 ^ (-134 : ℤ)) :
    |q16 - sigRef b0| ≤ bTol b0 := by
  have hb0 : b0 = -nb := by linarith
  set e : ℝ := Real.exp nb with hedef
  have hepos : 0 < e := by rw [hedef]; exact Real.exp_pos nb
  have henonneg : 0 ≤ e := le_of_lt hepos
  set c22 : ℝ := (2:ℝ)^(-22:ℤ) with hc22def
  set c23 : ℝ := (2:ℝ)^(-23:ℤ) with hc23def
  set c24 : ℝ := (2:ℝ)^(-24:ℤ) with hc24def
  set c16 : ℝ := (2:ℝ)^(-16:ℤ) with hc16def
  set c8 : ℝ := (2:ℝ)^(-8:ℤ) with hc8def
  set c7 : ℝ := (2:ℝ)^(-7:ℤ) with hc7def
  set c140 : ℝ := (2:ℝ)^(-140:ℤ) with hc140def
  set c150 : ℝ := (2:ℝ)^(-150:ℤ) with hc150def
  set c148 : ℝ := (2:ℝ)^(-148:ℤ) with hc148def
  set c146 : ℝ := (2:ℝ)^(-146:ℤ) with hc146def
  set c134 : ℝ := (2:ℝ)^(-134:ℤ) with hc134def
  set c139 : ℝ := (2:ℝ)^(-139:ℤ) with hc139def
  set c133 : ℝ := (2:ℝ)^(-133:ℤ) with hc133def
  clear_value c22 c23 c24 c16 c8 c7 c140 c150 c148 c146 c134 c139 c133
  set σ : ℝ := 1 / (1 + e) with hσdef
  have hden1 : 0 < 1 + e := by linarith
  have hden1one : (1:ℝ) ≤ 1 + e := by linarith
  have hden1half : (1:ℝ) / 2 ≤ 1 + e := by linarith
  have hσpos : 0 < σ := one_div_pos.mpr hden1
  have hσnonneg : 0 ≤ σ := le_of_lt hσpos
  have hs : sigRef b0 = σ := by
    unfold sigRef
    rw [hb0, neg_neg, ← hedef]
  have hkey : nb ≤ 0 → |nb| * e ≤ 1 := by
    intro h
    have h1 : -nb + 1 ≤ Real.exp (-nb) := Real.add_one_le_exp (-nb)
    have hexp : Real.exp (-nb) = e⁻¹ := by rw [Real.exp_neg, ← hedef]
    rw [hexp] at h1
    have hm : e * (-nb + 1) ≤ e * e⁻¹ := mul_le_mul_of_nonneg_left h1 henonneg
    rw [mul_inv_cancel₀ (ne_of_gt hepos)] at hm
    rw [abs_of_nonpos h]
    nlinarith [hm]
  have hc_nonpos : nb ≤ 0 → (c22 + c23 * |nb|) * e ≤ c16 := by
    intro h
    have he1 : e ≤ 1 := by rw [hedef]; exact Real.exp_le_one_iff.mpr h
    have hk : |nb| * e ≤ 1 := hkey h
    have h22 : (0:ℝ) ≤ c22 := by rw [hc22def]; norm_num
    have h23 : (0:ℝ) ≤ c23 := by rw [hc23def]; norm_num
    have h3 : c22 * e ≤ c22 := by
      simpa using mul_le_mul_of_nonneg_left he1 h22
    have h4 : c23 * (|nb| * e) ≤ c23 := by
      simpa using mul_le_mul_of_nonneg_left hk h23
    have hexp : (c22 + c23 * |nb|) * e = c22 * e + c23 * (|nb| * e) := by ring
    have hnum : c22 + c23 ≤ c16 := by rw [hc22def, hc23def, hc16def]; norm_num
    linarith [hexp, h3, h4, hnum]
  have hc_le_pos : 0 < nb → (c22 + c23 * |nb|) ≤ c16 := by
    intro h
    have habs : |nb| = nb := abs_of_nonneg (le_of_lt h)
    rw [habs]
    have h23 : (0:ℝ) ≤ c23 := by rw [hc23def]; norm_num
    have h9 : c23 * nb ≤ c23 * 90 := mul_le_mul_of_nonneg_left hnb_ub h23
    have hnum : c22 + c23 * 90 ≤ c16 := by rw [hc22def, hc23def, hc16def]; norm_num
    linarith
  have hce : (c22 + c23 * |nb|) * e / (1 + e) ≤ c16 := by
    rw [div_le_iff₀ hden1]
    have h16pos : (0:ℝ) < c16 := by rw [hc16def]; norm_num
    have hNe : (0:ℝ) ≤ c16 * e := mul_nonneg (le_of_lt h16pos) henonneg
    have hgoal : c16 * (1 + e) = c16 + c16 * e := by ring
    by_cases h : nb ≤ 0
    · have h1 := hc_nonpos h
      linarith [h1, hNe, hgoal]
    · have h1 := hc_le_pos (lt_of_not_ge h)
      have h2 : (c22 + c23 * |nb|) * e ≤ c16 * e :=
        mul_le_mul_of_nonneg_right h1 henonneg
      linarith [h2, hNe, hgoal]
  have hce2 : (c22 + c23 * |nb|) * e ≤ e / 2 + 1 / 4 := by
    by_cases h : nb ≤ 0
    · have h1 := hc_nonpos h
      have hnum : c16 ≤ 1 / 4 := by rw [hc16def]; norm_num
      have h1' : (c22 + c23 * |nb|) * e ≤ 1 / 4 := le_trans h1 hnum
      linarith [h1', henonneg]
    · have h1 := hc_le_pos (lt_of_not_ge h)
      have h2 : (c22 + c23 * |nb|) * e ≤ c16 * e :=
        mul_le_mul_of_nonneg_right h1 henonneg
      have hN : c16 ≤ 1 / 2 := by rw [hc16def]; norm_num
      have h2' : (c22 + c23 * |nb|) * e ≤ e / 2 := by
        have hm := mul_le_mul_of_nonneg_right hN henonneg
        linarith [h2]
      linarith [h2']
  have h_eb_lb : e - ((c22 + c23 * |nb|) * e + c140) ≤ eb := by
    have h := (abs_le.mp heb).1
    linarith
  have h1peb : (1 + e) / 2 ≤ 1 + eb := by
    have hlam : c140 ≤ 1 / 4 := by rw [hc140def]; norm_num
    linarith [h_eb_lb, hce2, hlam]
  have h1peb_pos : 0 < 1 + eb := by linarith
  have h1peb_half : (1:ℝ) / 2 ≤ 1 + eb := by linarith
  have hden_lb : (1 + eb) / 4 ≤ den := by
    have h := (abs_le.mp hden).1
    have hu : c24 ≤ 1 / 2 := by rw [hc24def]; norm_num
    have hu2 : c24 * (1 + eb) ≤ (1 + eb) / 2 := by
      have hm := mul_le_mul_of_nonneg_right hu (le_of_lt h1peb_pos)
      linarith
    have h8 : c150 ≤ 1 / 8 := by rw [hc150def]; norm_num
    have heta : c150 ≤ (1 + eb) / 4 := by linarith
    linarith
  have hdenpos : 0 < den := by linarith
  have hdd_raw : |1 / den - 1 / (1 + eb)| ≤
      (c24 * (1 + eb) + c150) / (den * (1 + eb)) :=
    abs_inv_sub_inv_le hdenpos h1peb_pos hden
  have hden2_lb : (1 + eb)^2 / 4 ≤ den * (1 + eb) := by
    have hm := mul_le_mul_of_nonneg_right hden_lb (le_of_lt h1peb_pos)
    nlinarith [hm]
  have hnum_nonneg : (0:ℝ) ≤ c24 * (1 + eb) + c150 := by
    have h1 : (0:ℝ) < c24 := by rw [hc24def]; norm_num
    have h2 : (0:ℝ) < c150 := by rw [hc150def]; norm_num
    nlinarith [h1peb_pos, h1, h2]
  have hstep1 : (c24 * (1 + eb) + c150) / (den * (1 + eb))
      ≤ (c24 * (1 + eb) + c150) / ((1 + eb)^2 / 4) := by
    apply div_le_div_of_nonneg_left hnum_nonneg
    · exact div_pos (pow_pos h1peb_pos 2) (by norm_num)
    · exact hden2_lb
  have hstep2 : (c24 * (1 + eb) + c150) / ((1 + eb)^2 / 4)
      = 4 * c24 / (1 + eb) + 4 * c150 / (1 + eb)^2 := by
    field_simp
  have hstep3 : 4 * c24 / (1 + eb) = c22 * (1 / (1 + eb)) := by
    have h4 : (4:ℝ) * c24 = c22 := by rw [hc24def, hc22def]; norm_num
    rw [h4]
    ring
  have hstep4 : 4 * c150 / (1 + eb)^2 ≤ c146 := by
    have hden2 : (1:ℝ) / 4 ≤ (1 + eb)^2 := by
      have hp := pow_le_pow_left₀ (by norm_num : (0:ℝ) ≤ 1/2) h1peb_half 2
      norm_num at hp ⊢
      linarith
    have hnum2 : (0:ℝ) ≤ 4 * c150 := by
      have h : (0:ℝ) < c150 := by rw [hc150def]; norm_num
      nlinarith
    have h := div_le_div_of_nonneg_left hnum2 (by norm_num : (0:ℝ) < 1/4) hden2
    have heq : 4 * c150 / (1/4) = c146 := by rw [hc150def, hc146def]; norm_num
    linarith
  have hdd_bound : |1 / den - 1 / (1 + eb)| ≤ c22 * (1 / (1 + eb)) + c146 := by
    calc |1 / den - 1 / (1 + eb)|
        ≤ (c24 * (1 + eb) + c150) / (den * (1 + eb)) := hdd_raw
      _ ≤ (c24 * (1 + eb) + c150) / ((1 + eb)^2 / 4) := hstep1
      _ = 4 * c24 / (1 + eb) + 4 * c150 / (1 + eb)^2 := hstep2
      _ = c22 * (1 / (1 + eb)) + 4 * c150 / (1 + eb)^2 := by rw [hstep3]
      _ ≤ c22 * (1 / (1 + eb)) + c146 := by linarith [hstep4]
  have hd1_raw : |1 / (1 + eb) - 1 / (1 + e)| ≤
      ((c22 + c23 * |nb|) * e + c140) / ((1 + eb) * (1 + e)) := by
    apply abs_inv_sub_inv_le h1peb_pos hden1
    have hsub : (1 + eb) - (1 + e) = eb - e := by ring
    rw [hsub]
    exact heb
  have hprod_lb : (1 + e)^2 / 2 ≤ (1 + eb) * (1 + e) := by
    have hm := mul_le_mul_of_nonneg_right h1peb (le_of_lt hden1)
    nlinarith [hm]
  have hnum2_nonneg : (0:ℝ) ≤ (c22 + c23 * |nb|) * e + c140 := by
    have hc : (0:ℝ) ≤ c22 + c23 * |nb| := by
      have h1 : (0:ℝ) < c22 := by rw [hc22def]; norm_num
      have h2 : (0:ℝ) ≤ c23 * |nb| := by
        have h3 : (0:ℝ) < c23 := by rw [hc23def]; norm_num
        exact mul_nonneg (le_of_lt h3) (abs_nonneg nb)
      linarith
    have hm := mul_nonneg hc henonneg
    have h2 : (0:ℝ) < c140 := by rw [hc140def]; norm_num
    linarith
  have hstepA : ((c22 + c23 * |nb|) * e + c140) / ((1 + eb) * (1 + e))
      ≤ ((c22 + c23 * |nb|) * e + c140) / ((1 + e)^2 / 2) := by
    apply div_le_div_of_nonneg_left hnum2_nonneg
    · exact div_pos (pow_pos hden1 2) (by norm_num)
    · exact hprod_lb
  have hstepB : ((c22 + c23 * |nb|) * e + c140) / ((1 + e)^2 / 2)
      = 2 * ((c22 + c23 * |nb|) * e) / (1 + e)^2
        + 2 * c140 / (1 + e)^2 := by
    field_simp
  have hstepC : 2 * ((c22 + c23 * |nb|) * e) / (1 + e)^2
      ≤ 2 * c16 * σ := by
    have hp : (0:ℝ) ≤ 2 / (1 + e) :=
      div_nonneg (by norm_num) (le_of_lt hden1)
    have hm := mul_le_mul_of_nonneg_right hce hp
    have hl : ((c22 + c23 * |nb|) * e / (1 + e)) * (2 / (1 + e))
        = 2 * ((c22 + c23 * |nb|) * e) / (1 + e)^2 := by
      field_simp
    have hr : c16 * (2 / (1 + e)) = 2 * c16 * σ := by
      rw [hσdef]
      ring
    linarith [hm, hl, hr]
  have hstepD : 2 * c140 / (1 + e)^2 ≤ c139 := by
    have hden2 : (1:ℝ) ≤ (1 + e)^2 := by
      have hp := pow_le_pow_left₀ (by norm_num : (0:ℝ) ≤ 1) hden1one 2
      simpa using hp
    have hnum2 : (0:ℝ) ≤ 2 * c140 := by
      have h : (0:ℝ) < c140 := by rw [hc140def]; norm_num
      nlinarith
    have h := div_le_div_of_nonneg_left hnum2 (by norm_num : (0:ℝ) < 1) hden2
    have heq : 2 * c140 / 1 = c139 := by rw [hc140def, hc139def]; norm_num
    linarith
  have hd1_sigma : |1 / (1 + eb) - σ| ≤ 2 * c16 * σ + c139 := by
    calc |1 / (1 + eb) - σ|
        = |1 / (1 + eb) - 1 / (1 + e)| := by rw [hσdef]
      _ ≤ ((c22 + c23 * |nb|) * e + c140) / ((1 + eb) * (1 + e)) := hd1_raw
      _ ≤ ((c22 + c23 * |nb|) * e + c140) / ((1 + e)^2 / 2) := hstepA
      _ = 2 * ((c22 + c23 * |nb|) * e) / (1 + e)^2
          + 2 * c140 / (1 + e)^2 := hstepB
      _ ≤ 2 * c16 * σ + c139 := by linarith [hstepC, hstepD]
  have hd1pos : 0 < 1 / (1 + eb) := one_div_pos.mpr h1peb_pos
  have hq_abs : |q| ≤ (1 + c22) * |1 / den| + c148 := by
    have heqq : q = 1 / den + (q - 1 / den) := by ring
    calc |q| = |1 / den + (q - 1 / den)| := congrArg abs heqq
      _ ≤ |1 / den| + |q - 1 / den| := abs_add_le _ _
      _ ≤ |1 / den| + (c22 * |1 / den| + c148) := by linarith [hq]
      _ = (1 + c22) * |1 / den| + c148 := by ring
  have hdd_abs : |1 / den| ≤ (1 + c22) * (1 / (1 + eb)) + c146 := by
    have h1 := (abs_le.mp hdd_bound).1
    have h2 := (abs_le.mp hdd_bound).2
    rw [abs_le']
    constructor <;> linarith
  have hd1_sig : 1 / (1 + eb) ≤ (1 + 2 * c16) * σ + c139 := by
    have h := (abs_le.mp hd1_sigma).2
    linarith
  have hc22nn : (0:ℝ) ≤ 1 + c22 := by
    have h : (0:ℝ) < c22 := by rw [hc22def]; norm_num
    linarith
  have hq_abs2 : |q| ≤ (1 + c22) * ((1 + c22) * (1 / (1 + eb)) + c146) + c148 := by
    have hm := mul_le_mul_of_nonneg_left hdd_abs hc22nn
    linarith
  have hq_abs3 : |q| ≤ (1 + c22) * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148 := by
    have hinner : (1 + c22) * (1 / (1 + eb)) + c146
        ≤ (1 + c22) * ((1 + 2 * c16) * σ + c139) + c146 := by
      have hm := mul_le_mul_of_nonneg_left hd1_sig hc22nn
      linarith
    have hm := mul_le_mul_of_nonneg_left hinner hc22nn
    linarith
  have hdd_abs2 : |1 / den| ≤ (1 + c22) * ((1 + 2 * c16) * σ + c139) + c146 := by
    have hm := mul_le_mul_of_nonneg_left hd1_sig hc22nn
    linarith
  have htri : |q16 - σ| ≤ |q16 - q| + |q - 1 / den| + |1 / den - 1 / (1 + eb)| + |1 / (1 + eb) - σ| := by
    have h1 := abs_sub_le q16 q σ
    have h2 := abs_sub_le q (1 / den) σ
    have h3 := abs_sub_le (1 / den) (1 / (1 + eb)) σ
    linarith
  have hb1 : |q16 - q| ≤ c8 * ((1 + c22) * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148) + c134 := by
    have hc : (0:ℝ) ≤ c8 := by rw [hc8def]; norm_num
    have hm := mul_le_mul_of_nonneg_left hq_abs3 hc
    linarith
  have hb2 : |q - 1 / den| ≤ c22 * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148 := by
    have hc : (0:ℝ) ≤ c22 := by rw [hc22def]; norm_num
    have hm := mul_le_mul_of_nonneg_left hdd_abs2 hc
    linarith
  have hb3 : |1 / den - 1 / (1 + eb)| ≤ c22 * ((1 + 2 * c16) * σ + c139) + c146 := by
    have hc : (0:ℝ) ≤ c22 := by rw [hc22def]; norm_num
    have hm := mul_le_mul_of_nonneg_left hd1_sig hc
    linarith
  have hsum : |q16 - σ| ≤
      (c8 * ((1 + c22) * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148) + c134)
      + (c22 * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148)
      + (c22 * ((1 + 2 * c16) * σ + c139) + c146)
      + (2 * c16 * σ + c139) := by
    linarith [htri, hb1, hb2, hb3, hd1_sigma]
  have hK : c8 * ((1 + c22) * ((1 + c22) * (1 + 2 * c16)))
      + c22 * ((1 + c22) * (1 + 2 * c16))
      + c22 * (1 + 2 * c16) + 2 * c16 ≤ c7 := by
    rw [hc8def, hc22def, hc16def, hc7def]
    norm_num
  have hA : c8 * ((1 + c22) * ((1 + c22) * c139 + c146) + c148) + c134
      + (c22 * ((1 + c22) * c139 + c146) + c148)
      + (c22 * c139 + c146) + c139 ≤ c133 := by
    rw [hc8def, hc22def, hc139def, hc146def, hc148def, hc134def, hc133def]
    norm_num
  have hKσ : (c8 * ((1 + c22) * ((1 + c22) * (1 + 2 * c16)))
      + c22 * ((1 + c22) * (1 + 2 * c16))
      + c22 * (1 + 2 * c16) + 2 * c16) * σ ≤ c7 * σ :=
    mul_le_mul_of_nonneg_right hK hσnonneg
  have hexp :
      (c8 * ((1 + c22) * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148) + c134)
      + (c22 * ((1 + c22) * ((1 + 2 * c16) * σ + c139) + c146) + c148)
      + (c22 * ((1 + 2 * c16) * σ + c139) + c146)
      + (2 * c16 * σ + c139)
      = (c8 * ((1 + c22) * ((1 + c22) * (1 + 2 * c16)))
          + c22 * ((1 + c22) * (1 + 2 * c16))
          + c22 * (1 + 2 * c16) + 2 * c16) * σ
        + (c8 * ((1 + c22) * ((1 + c22) * c139 + c146) + c148) + c134
          + (c22 * ((1 + c22) * c139 + c146) + c148)
          + (c22 * c139 + c146) + c139) := by
    ring
  rw [hexp] at hsum
  have hfin : |q16 - σ| ≤ c7 * σ + c133 := by
    exact le_trans hsum (by linarith [hKσ, hA])
  unfold bTol
  rw [← hc7def, ← hc133def, hs, abs_of_pos hσpos]
  exact hfin

end Claims.GdnGating
