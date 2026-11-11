import Claim
import Proof.NumBase

section
/-!
# The numeric core of the tolerance

The growth bounds over a whole loop (`GLmax`, `GAmax`) and the final relative error they give
(`final_ineq`) stay within the claim's `tolRel` on every covered row: `T ≤ 1024` trips, scores
`X ≤ 2^12`, head dimension `D ≤ 256`.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

/-- `exp a ≤ 1 + (1 + a)·a` for `0 ≤ a ≤ 1`. -/
private lemma exp_le_quad {a : ℝ} (h0 : 0 ≤ a) (h1 : a ≤ 1) : Real.exp a ≤ 1 + (1 + a) * a := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := a) (by rw [abs_of_nonneg h0]; exact h1)
  have := (abs_le.mp h).2
  nlinarith

/-- `l`'s growth is at most a linear function of the trips and `Y`. -/
private lemma GLmax_lin {Y T : ℝ} (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) (hT0 : 0 ≤ T) (hT : T ≤ 1024) :
    GLmax Y T ≤ 1.012 * (T + Y) / 524288 + 1.012 * (1 / 131072 + Y / 524288) := by
  have e1 : (2 : ℝ) ^ (-19 : ℤ) = 1 / 524288 := by norm_num
  have e2 : (2 : ℝ) ^ (-20 : ℤ) = 1 / 1048576 := by norm_num
  have e3 : (2 : ℝ) ^ (-17 : ℤ) = 1 / 131072 := by norm_num
  unfold GLmax bL cL0 cL1
  rw [e1, e2, e3]
  set a : ℝ := 1 / 524288 * T + 1 / 1048576 * (2 * Y) with ha
  set b : ℝ := 1 / 131072 + 1 / 524288 * Y with hb
  have ha0 : 0 ≤ a := by rw [ha]; positivity
  have ha1 : a ≤ 0.0115 := by rw [ha]; linarith
  have hb0 : 0 ≤ b := by rw [hb]; positivity
  have hb1 : b ≤ 0.0096 := by rw [hb]; linarith
  have := exp_le_quad ha0 (by linarith)
  have h2 : Real.exp a * (1 + b) ≤ (1 + (1 + a) * a) * (1 + b) :=
    mul_le_mul_of_nonneg_right this (by linarith)
  have hT' : a = (T + Y) / 524288 := by rw [ha]; ring
  nlinarith [mul_nonneg ha0 hb0, mul_nonneg ha0 ha0]

/-- The accumulator's growth is at most a linear function of the trips and `Y`. -/
private lemma GAmax_lin {Y T : ℝ} (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) (hT0 : 0 ≤ T) (hT : T ≤ 1024) :
    GAmax Y T ≤ 1.052 * (160 * T / 4194304 + Y / 524288) +
      1.052 * (2 ^ (-8 : ℤ) + 2 ^ (-12 : ℤ) + Y / 524288) := by
  have e1 : (2 : ℝ) ^ (-19 : ℤ) = 1 / 524288 := by norm_num
  have e2 : (2 : ℝ) ^ (-20 : ℤ) = 1 / 1048576 := by norm_num
  have e3 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have e4 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have e5 : (2 : ℝ) ^ (-12 : ℤ) = 1 / 4096 := by norm_num
  unfold GAmax bA cA0 cA1
  rw [e1, e2, e3, e4, e5]
  set a : ℝ := 160 * (1 / 4194304) * T + 1 / 1048576 * (2 * Y) with ha
  set b : ℝ := 1 / 256 + 1 / 4096 + 1 / 524288 * Y with hb
  have ha0 : 0 ≤ a := by rw [ha]; positivity
  have ha1 : a ≤ 0.0487 := by rw [ha]; linarith
  have hb0 : 0 ≤ b := by rw [hb]; positivity
  have hb1 : b ≤ 0.0139 := by rw [hb]; linarith
  have := exp_le_quad ha0 (by linarith)
  have h2 : Real.exp a * (1 + b) ≤ (1 + (1 + a) * a) * (1 + b) :=
    mul_le_mul_of_nonneg_right this (by linarith)
  nlinarith [mul_nonneg ha0 hb0, mul_nonneg ha0 ha0]

/-- The growths are nonnegative. -/
private lemma growth_nonneg {Y T : ℝ} (hY0 : 0 ≤ Y) (hT0 : 0 ≤ T) :
    0 ≤ GLmax Y T ∧ 0 ≤ GAmax Y T := by
  have h1 : (0 : ℝ) ≤ 2 ^ (-19 : ℤ) := by positivity
  have h2 : (0 : ℝ) ≤ 2 ^ (-20 : ℤ) := by positivity
  have h3 : (0 : ℝ) ≤ 2 ^ (-17 : ℤ) := by positivity
  have h4 : (0 : ℝ) ≤ 2 ^ (-8 : ℤ) := by positivity
  have h5 : (0 : ℝ) ≤ 2 ^ (-12 : ℤ) := by positivity
  have h6 : (0 : ℝ) ≤ 160 * 2 ^ (-22 : ℤ) := by positivity
  constructor
  · unfold GLmax bL cL0 cL1
    have : 1 ≤ Real.exp (2 ^ (-19 : ℤ) * T + 2 ^ (-20 : ℤ) * (2 * Y)) :=
      Real.one_le_exp (by positivity)
    have : 0 ≤ 2 ^ (-17 : ℤ) + 2 ^ (-19 : ℤ) * Y := by positivity
    nlinarith
  · unfold GAmax bA cA0 cA1
    have : 1 ≤ Real.exp (160 * 2 ^ (-22 : ℤ) * T + 2 ^ (-20 : ℤ) * (2 * Y)) :=
      Real.one_le_exp (by positivity)
    have : 0 ≤ 2 ^ (-8 : ℤ) + 2 ^ (-12 : ℤ) + 2 ^ (-19 : ℤ) * Y := by positivity
    nlinarith

/-- Over a loop of at most `1024` trips with `Y ≤ 5000`, `l` grows by at most `1/8`. -/
theorem GLmax_le {Y T : ℝ} (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) (hT0 : 0 ≤ T) (hT : T ≤ 1024) :
    GLmax Y T ≤ 1 / 8 := by
  have := GLmax_lin hY0 hY hT0 hT
  have h1 : 1.012 * (T + Y) / 524288 + 1.012 * (1 / 131072 + Y / 524288) ≤ 1 / 8 := by linarith
  linarith

/-- ...and the accumulator by at most `1/8`. -/
theorem GAmax_le {Y T : ℝ} (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) (hT0 : 0 ≤ T) (hT : T ≤ 1024) :
    GAmax Y T ≤ 1 / 8 := by
  have := GAmax_lin hY0 hY hT0 hT
  have e4 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have e5 : (2 : ℝ) ^ (-12 : ℤ) = 1 / 4096 := by norm_num
  rw [e4, e5] at this
  linarith

/-- **The tolerance holds.** The relative error the loop's bounds give, rounded once more to
`bfloat16`, is within `tolRel`: `fine = 2^-8 + (1 + 2^-8)·(e^{2δ}·(G_A + G_L)/(1 − G_L − 2^-60)
+ e^{2δ} − 1)`. -/
theorem final_ineq {T D : ℕ} {X δ Y : ℝ} (hT1 : 1 ≤ T) (hT : T ≤ 1024) (hD1 : 1 ≤ D)
    (hD2 : D ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 4096)
    (hδ : δ = ((D : ℝ) + 8) * 2 ^ (-22 : ℤ) * X + 2 ^ (-40 : ℤ)) (hY : Y = X + δ) :
    2 ^ (-8 : ℤ) + (1 + 2 ^ (-8 : ℤ)) *
        (Real.exp (2 * δ) * ((GAmax Y T + GLmax Y T) / (1 - GLmax Y T - 2 ^ (-60 : ℤ))) +
          (Real.exp (2 * δ) - 1)) ≤ tolRel D T X := by
  have c40 : (0 : ℝ) ≤ 2 ^ (-40 : ℤ) := by positivity
  have c40' : (2 : ℝ) ^ (-40 : ℤ) ≤ 1e-11 := by norm_num
  have c60 : (0 : ℝ) ≤ 2 ^ (-60 : ℤ) := by positivity
  have c60' : (2 : ℝ) ^ (-60 : ℤ) ≤ 1e-11 := by norm_num
  have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have c8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have c6 : (2 : ℝ) ^ (-6 : ℤ) = 1 / 64 := by norm_num
  have c12 : (2 : ℝ) ^ (-12 : ℤ) = 1 / 4096 := by norm_num
  have c14 : (2 : ℝ) ^ (-14 : ℤ) = 1 / 16384 := by norm_num
  have c20 : (2 : ℝ) ^ (-20 : ℤ) = 1 / 1048576 := by norm_num
  have hDr1 : (1 : ℝ) ≤ D := by exact_mod_cast hD1
  have hDr2 : (D : ℝ) ≤ 256 := by exact_mod_cast hD2
  have hTr0 : (0 : ℝ) ≤ T := by positivity
  have hTr : (T : ℝ) ≤ 1024 := by exact_mod_cast hT
  have hw0 : 0 ≤ (D : ℝ) * X := by positivity
  have hw : (D : ℝ) * X ≤ 1048576 := by nlinarith
  have hδe : δ = ((D : ℝ) * X + 8 * X) / 4194304 + 2 ^ (-40 : ℤ) := by
    rw [hδ, c22]; ring
  have hδ0 : 0 ≤ δ := by rw [hδe]; positivity
  have hδ1 : δ ≤ 0.26 := by rw [hδe]; linarith
  have hY0 : 0 ≤ Y := by rw [hY]; linarith
  have hY5 : Y ≤ 5000 := by rw [hY]; linarith
  have hL := GLmax_lin hY0 hY5 hTr0 hTr
  have hA := GAmax_lin hY0 hY5 hTr0 hTr
  rw [c8, c12] at hA
  obtain ⟨hL0, hA0⟩ := growth_nonneg hY0 hTr0
  set L := GLmax Y T with hLdef
  set A := GAmax Y T with hAdef
  have hL1 : L ≤ 0.0214 := by linarith
  have hA1 : A ≤ 0.066 := by linarith
  generalize (2 : ℝ) ^ (-60 : ℤ) = c60v at *
  have hden : 0.978 ≤ 1 - L - c60v := by linarith
  have hS : (A + L) / (1 - L - c60v) ≤ 1.023 * (A + L) := by
    rw [div_le_iff₀ (by linarith)]
    have := mul_nonneg (add_nonneg hA0 hL0) (show 0 ≤ 1.023 * (1 - L - c60v) - 1 by linarith)
    linarith
  have hS0 : 0 ≤ (A + L) / (1 - L - c60v) := div_nonneg (by linarith) (by linarith)
  set S := (A + L) / (1 - L - c60v) with hSdef
  have hS1 : S ≤ 0.1 := by linarith
  set E := Real.exp (2 * δ) with hEdef
  have hE0 : 1 ≤ E := Real.one_le_exp (by linarith)
  have hE1 : E ≤ 1 + (1 + 2 * δ) * (2 * δ) := exp_le_quad (by linarith) (by linarith)
  have hE2 : E - 1 ≤ 3.04 * δ := by nlinarith
  have hES : E * S ≤ S + (E - 1) * 0.1 := by nlinarith
  unfold tolRel
  rw [c8, c6, c14, c20]
  rw [hY] at hL hA
  linarith

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# The end of the loop: the output against the exact attention

After the last trip the row's state is within the running bounds of the exact sums at the computed
scores `sh`. `final_bound` turns that into the claim's tolerance: the stored `bfloat16` `c` is
within `tolRel·attnAbs + tolAbs` of the exact attention over the exact scores `sc`
(`attnRef`, `attnAbs` of the claim, written as their sums). The weights `exp(sh)/Σ exp(sh)` and
`exp(sc)/Σ exp(sc)` differ by the factor `e^{±2δ}` termwise (`weights_close`); the running bounds
of `l` and of `l·a` give `a` against `Asum/Lam` (`ratio_step`); `final_ineq` compares the result
with `tolRel`.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- Weights proportional to `a` and to `b`, termwise within `p` of each other, differ by `p² − 1`
of the weight: the weighted means of any `v` differ by `(p² − 1)` times that of `|v|`, and the
mean of `|v|` changes by at most the factor `p²`. -/
private lemma weights_close {K : Finset ℕ} {a b v : ℕ → ℝ} {κ p : ℝ} (hK : K.Nonempty)
    (hp : 1 ≤ p) (hκ : 0 < κ) (ha : ∀ j ∈ K, 0 < a j)
    (hlo : ∀ j ∈ K, κ * a j ≤ p * b j) (hhi : ∀ j ∈ K, b j ≤ p * (κ * a j)) :
    |(∑ j ∈ K, b j * v j) / ∑ j ∈ K, b j - (∑ j ∈ K, a j * v j) / ∑ j ∈ K, a j| ≤
        (p ^ 2 - 1) * ((∑ j ∈ K, a j * |v j|) / ∑ j ∈ K, a j) ∧
      (∑ j ∈ K, b j * |v j|) / ∑ j ∈ K, b j ≤ p ^ 2 * ((∑ j ∈ K, a j * |v j|) / ∑ j ∈ K, a j) := by
  have hb : ∀ j ∈ K, 0 < b j := fun j hj => by
    have h1 := hlo j hj
    have h2 := mul_pos hκ (ha j hj)
    by_contra h3
    have h3' := not_lt.mp h3
    nlinarith
  have hA : 0 < ∑ j ∈ K, a j := Finset.sum_pos ha hK
  have hB : 0 < ∑ j ∈ K, b j := Finset.sum_pos hb hK
  set A := ∑ j ∈ K, a j with hAdef
  set B := ∑ j ∈ K, b j with hBdef
  have h1 : κ * A ≤ p * B := by
    rw [hAdef, hBdef, Finset.mul_sum, Finset.mul_sum]; exact Finset.sum_le_sum hlo
  have h2 : B ≤ p * (κ * A) := by
    rw [hAdef, hBdef, Finset.mul_sum, Finset.mul_sum]
    have : ∀ j ∈ K, b j ≤ p * (κ * a j) := hhi
    simpa [Finset.mul_sum] using Finset.sum_le_sum this
  have pt : ∀ j ∈ K, b j / B ≤ p ^ 2 * (a j / A) ∧ a j / A ≤ p ^ 2 * (b j / B) := by
    intro j hj
    have hj1 := hlo j hj
    have hj2 := hhi j hj
    have haj := ha j hj
    have hp0 : 0 < p := by linarith
    constructor
    · rw [div_le_iff₀ hB, show p ^ 2 * (a j / A) * B = p ^ 2 * a j * B / A by ring,
        le_div_iff₀ hA]
      have e1 : b j * A ≤ p * (κ * a j) * A := mul_le_mul_of_nonneg_right hj2 hA.le
      have e2 : p * (κ * A) * a j ≤ p * (p * B) * a j :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h1 hp0.le) haj.le
      nlinarith
    · rw [div_le_iff₀ hA, show p ^ 2 * (b j / B) * A = p ^ 2 * b j * A / B by ring,
        le_div_iff₀ hB]
      have e1 : a j * B ≤ a j * (p * (κ * A)) := mul_le_mul_of_nonneg_left h2 haj.le
      have e2 : κ * a j * (p * A) ≤ p * b j * (p * A) :=
        mul_le_mul_of_nonneg_right hj1 (by positivity)
      nlinarith
  have pt2 : ∀ j ∈ K, |b j / B - a j / A| ≤ (p ^ 2 - 1) * (a j / A) := by
    intro j hj
    obtain ⟨u1, u2⟩ := pt j hj
    have hy : 0 < a j / A := div_pos (ha j hj) hA
    have hx : 0 < b j / B := div_pos (hb j hj) hB
    have hp2 : 1 ≤ p ^ 2 := by nlinarith
    rw [abs_le]
    constructor
    · by_cases h : a j / A ≤ b j / B
      · nlinarith [mul_nonneg (sub_nonneg.2 hp2) hy.le]
      · nlinarith [mul_nonneg (sub_nonneg.2 hp2) (sub_nonneg.2 (not_le.mp h).le)]
    · nlinarith
  have hsum : ∀ w : ℕ → ℝ, (∑ j ∈ K, b j * w j) / B = ∑ j ∈ K, (b j / B) * w j := fun w => by
    rw [Finset.sum_div]; exact Finset.sum_congr rfl fun j _ => by ring
  have hsum' : ∀ w : ℕ → ℝ, (∑ j ∈ K, a j * w j) / A = ∑ j ∈ K, (a j / A) * w j := fun w => by
    rw [Finset.sum_div]; exact Finset.sum_congr rfl fun j _ => by ring
  constructor
  · rw [hsum, hsum', ← Finset.sum_sub_distrib, hsum' (fun j => |v j|), Finset.mul_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j hj => ?_)
    rw [← sub_mul, abs_mul]
    calc |b j / B - a j / A| * |v j| ≤ (p ^ 2 - 1) * (a j / A) * |v j| :=
          mul_le_mul_of_nonneg_right (pt2 j hj) (abs_nonneg _)
      _ = (p ^ 2 - 1) * (a j / A * |v j|) := by ring
  · rw [hsum, hsum', Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      rw [← mul_assoc]; exact mul_le_mul_of_nonneg_right (pt j hj).1 (abs_nonneg _)

/-- The comparison `(gA + gL + ε)/(1 − gL − ε) ≤ (gA + gL)/(1 − gL − c)`, `ε = 2^-80`,
`c = 2^-60`, for `gL ≥ 2^-17`. -/
private lemma rate_cmp {gA gL : ℝ} (hA : 0 ≤ gA) (hL : 2 ^ (-17 : ℤ) ≤ gL) (hL' : gL ≤ 1 / 8) :
    (gA + gL + 2 ^ (-80 : ℤ)) / (1 - gL - 2 ^ (-80 : ℤ)) ≤
      (gA + gL) / (1 - gL - 2 ^ (-60 : ℤ)) := by
  have e1 : (2 : ℝ) ^ (-17 : ℤ) = 1 / 131072 := by norm_num
  have e2 : (2 : ℝ) ^ (-80 : ℤ) ≤ 1e-24 := by norm_num
  have e3 : (8e-19 : ℝ) ≤ 2 ^ (-60 : ℤ) := by norm_num
  have e4 : (2 : ℝ) ^ (-60 : ℤ) ≤ 1e-11 := by norm_num
  have e5 : (0 : ℝ) ≤ 2 ^ (-80 : ℤ) := by positivity
  generalize (2 : ℝ) ^ (-80 : ℤ) = e at *
  generalize (2 : ℝ) ^ (-60 : ℤ) = c at *
  rw [e1] at hL
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_nonneg (add_nonneg hA (by linarith : (0 : ℝ) ≤ gL)) (by linarith : (0 : ℝ) ≤ c - e),
    mul_nonneg (by linarith : (0 : ℝ) ≤ gL - 1 / 131072) (by linarith : (0 : ℝ) ≤ c - e),
    mul_nonneg hA (by linarith : (0 : ℝ) ≤ c - e)]

/-- **One ratio.** From `|l − S| ≤ (gL + 2^-80)·S` and `|l·a − P| ≤ gA·Q + ab` (with `|P| ≤ Q`,
`S ≥ 1`), `a` is within `(Q/S)·(gA + gL)/(1 − gL − 2^-60) + 2·ab` of `P/S`. -/
private lemma ratio_step {l S P Q a gA gL ab : ℝ} (hS : 1 ≤ S) (hP : |P| ≤ Q) (hgA : 0 ≤ gA)
    (hgL : 2 ^ (-17 : ℤ) ≤ gL) (hgL' : gL ≤ 1 / 8) (hab : 0 ≤ ab)
    (hl : |l - S| ≤ (gL + 2 ^ (-80 : ℤ)) * S) (hx : |l * a - P| ≤ gA * Q + ab) :
    |a - P / S| ≤ Q / S * ((gA + gL) / (1 - gL - 2 ^ (-60 : ℤ))) + 2 * ab := by
  have e2 : (2 : ℝ) ^ (-80 : ℤ) ≤ 1e-24 := by norm_num
  have e5 : (0 : ℝ) ≤ 2 ^ (-80 : ℤ) := by positivity
  have e6 : (0 : ℝ) < 2 ^ (-17 : ℤ) := by positivity
  have hcmp := rate_cmp hgA hgL hgL'
  generalize (2 : ℝ) ^ (-60 : ℤ) = c at *
  generalize (2 : ℝ) ^ (-80 : ℤ) = e at *
  have hQ : 0 ≤ Q := (abs_nonneg _).trans hP
  have hd : 0.87 ≤ 1 - gL - e := by linarith
  have hS0 : 0 < S := by linarith
  have hlS : S * (1 - gL - e) ≤ l := by
    have := (abs_le.mp hl).1
    nlinarith
  have hl0 : 0 < l := by nlinarith
  have key : a - P / S = (S * (l * a - P) + P * (S - l)) / (l * S) := by
    field_simp; ring
  have n1 : |S * (l * a - P) + P * (S - l)| ≤ S * (gA * Q + ab) + Q * ((gL + e) * S) := by
    refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [abs_mul, abs_of_pos hS0]; exact mul_le_mul_of_nonneg_left hx hS0.le
    · rw [abs_mul, abs_sub_comm]
      exact mul_le_mul hP hl (abs_nonneg _) hQ
  have h3 : 0 ≤ S * (gA * Q + ab) + Q * ((gL + e) * S) := by
    have : 0 ≤ gL + e := by linarith
    positivity
  have hSd : 0 < S * (1 - gL - e) := by positivity
  have n2 : |a - P / S| ≤ (Q * (gA + gL + e) + ab) / (S * (1 - gL - e)) := by
    rw [key, abs_div, abs_of_pos (mul_pos hl0 hS0), div_le_div_iff₀ (mul_pos hl0 hS0) hSd]
    calc |S * (l * a - P) + P * (S - l)| * (S * (1 - gL - e))
        ≤ (S * (gA * Q + ab) + Q * ((gL + e) * S)) * (S * (1 - gL - e)) :=
          mul_le_mul_of_nonneg_right n1 hSd.le
      _ ≤ (S * (gA * Q + ab) + Q * ((gL + e) * S)) * l := mul_le_mul_of_nonneg_left hlS h3
      _ = (Q * (gA + gL + e) + ab) * (l * S) := by ring
  have n3 : (Q * (gA + gL + e) + ab) / (S * (1 - gL - e)) =
      Q / S * ((gA + gL + e) / (1 - gL - e)) + ab / (S * (1 - gL - e)) := by
    have : 1 - gL - e ≠ 0 := by linarith
    field_simp
  have n4 : ab / (S * (1 - gL - e)) ≤ 2 * ab := by
    rw [div_le_iff₀ hSd]
    nlinarith [mul_nonneg hab (by linarith : (0 : ℝ) ≤ S - 1),
      mul_nonneg hab (by linarith : (0 : ℝ) ≤ 1 - gL - e - 0.87), mul_nonneg hab hS0.le,
      mul_nonneg (mul_nonneg hab (by linarith : (0 : ℝ) ≤ S - 1))
        (by linarith : (0 : ℝ) ≤ 1 - gL - e)]
  have n5 : Q / S * ((gA + gL + e) / (1 - gL - e)) ≤ Q / S * ((gA + gL) / (1 - gL - c)) :=
    mul_le_mul_of_nonneg_left hcmp (by positivity)
  linarith

/-- `GLmax ≥ 2^-17` and `GAmax ≥ 0` when `Y ≥ 0`. -/
private lemma growth_low {Y T : ℝ} (hY0 : 0 ≤ Y) (hT0 : 0 ≤ T) :
    2 ^ (-17 : ℤ) ≤ GLmax Y T ∧ 0 ≤ GAmax Y T := by
  have h1 : (0 : ℝ) ≤ 2 ^ (-19 : ℤ) := by positivity
  have h2 : (0 : ℝ) ≤ 2 ^ (-20 : ℤ) := by positivity
  have h3 : (0 : ℝ) ≤ 2 ^ (-17 : ℤ) := by positivity
  have h4 : (0 : ℝ) ≤ 2 ^ (-8 : ℤ) := by positivity
  have h5 : (0 : ℝ) ≤ 2 ^ (-12 : ℤ) := by positivity
  have h6 : (0 : ℝ) ≤ 160 * 2 ^ (-22 : ℤ) := by positivity
  constructor
  · unfold GLmax bL cL0 cL1
    have : 1 ≤ Real.exp (2 ^ (-19 : ℤ) * T + 2 ^ (-20 : ℤ) * (2 * Y)) :=
      Real.one_le_exp (by positivity)
    have : 0 ≤ 2 ^ (-19 : ℤ) * Y := mul_nonneg h1 hY0
    nlinarith
  · unfold GAmax bA cA0 cA1
    have : 1 ≤ Real.exp (160 * 2 ^ (-22 : ℤ) * T + 2 ^ (-20 : ℤ) * (2 * Y)) :=
      Real.one_le_exp (by positivity)
    have : 0 ≤ 2 ^ (-19 : ℤ) * Y := mul_nonneg h1 hY0
    nlinarith

/-- A `Close` bound whose growth factor is at most `1 + g`, with `g ≤ 1/8`, is `g·M` plus
`9/8·n·η`. -/
private lemma close_final {x X M ω B η g : ℝ} {n : ℕ} (h : Close x X M ω B η n)
    (hM : 0 ≤ M) (hη : 0 ≤ n * η) (hg : Real.exp ω * (1 + B) ≤ 1 + g) (hB : 0 ≤ B)
    (hg' : g ≤ 1 / 8) : |x - X| ≤ g * M + 9 / 8 * (n * η) := by
  have he : Real.exp ω ≤ 9 / 8 := by
    have := Real.exp_pos ω
    nlinarith
  unfold Close at h
  have h1 : (Real.exp ω * (1 + B) - 1) * M ≤ g * M := mul_le_mul_of_nonneg_right (by linarith) hM
  have h2 : Real.exp ω * (n * η) ≤ 9 / 8 * (n * η) := mul_le_mul_of_nonneg_right he hη
  linarith

set_option maxHeartbeats 1600000 in
/-- **The output.** After the `T = L/128 + 1` trips of a row that attends the keys `0, …, L`
(`|sc j| ≤ X`, computed scores within `δ = (D + 8)·2^-22·X + 2^-40`), the stored `c` is within the
claim's tolerance of the exact attention. -/
theorem final_bound {L T D : ℕ} {sc sh vv : ℕ → ℝ} {X δ Y m l a c V : ℝ}
    (hT : T = L / 128 + 1) (hT1 : T ≤ 1024) (hD1 : 1 ≤ D) (hD2 : D ≤ 256)
    (hX0 : 0 ≤ X) (hX : X ≤ 4096) (hscX : ∀ j, j ≤ L → |sc j| ≤ X)
    (hδ : δ = ((D : ℝ) + 8) * 2 ^ (-22 : ℤ) * X + 2 ^ (-40 : ℤ)) (hY : Y = X + δ)
    (inv : RowInv L T sc sh δ Y m l)
    (hvv : ∀ j, j ≤ L → |vv j| ≤ 2 ^ 120) (hV : V = ∑ j ∈ range (L + 1), |vv j|)
    (cinv : Close (l * a) (Asum sh vv (keysN L T) m) (Aabs sh vv (keysN L T) m) (omA Y T m)
      (bA Y) (ηA * (1 + V)) T)
    (hc : |c - a| ≤ 2 ^ (-8 : ℤ) * |a| + 2 ^ (-134 : ℤ)) :
    |c - (∑ j ∈ range (L + 1), Real.exp (sc j) * vv j) / ∑ j ∈ range (L + 1), Real.exp (sc j)| ≤
      tolRel D T X *
          ((∑ j ∈ range (L + 1), Real.exp (sc j) * |vv j|) / ∑ j ∈ range (L + 1), Real.exp (sc j)) +
        tolAbs V := by
  have _ := hvv
  have hKe : keysN L T = range (L + 1) := by
    rw [keysN, min_eq_right (by omega)]
  have hT1' : 1 ≤ T := by omega
  have hD2r : (D : ℝ) ≤ 256 := by exact_mod_cast hD2
  have hTr : (T : ℝ) ≤ 1024 := by exact_mod_cast hT1
  have hTr0 : (0 : ℝ) ≤ T := by positivity
  have hδ0 : 0 ≤ δ := by rw [hδ]; positivity
  have hδ1 : δ ≤ 0.26 := by
    have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    have c40 : (2 : ℝ) ^ (-40 : ℤ) ≤ 1e-11 := by norm_num
    have h1 : ((D : ℝ) + 8) * X ≤ 264 * 4096 := by
      nlinarith [mul_nonneg (show (0 : ℝ) ≤ 256 - D by linarith) hX0]
    rw [hδ, c22]; nlinarith
  have hY0 : 0 ≤ Y := by rw [hY]; linarith
  have hY5 : Y ≤ 5000 := by rw [hY]; linarith
  obtain ⟨hgL, hgA⟩ := growth_low hY0 hTr0
  have hGL := GLmax_le hY0 hY5 hTr0 hTr
  have hGA := GAmax_le hY0 hY5 hTr0 hTr
  have hV0 : 0 ≤ V := by rw [hV]; exact Finset.sum_nonneg fun j _ => abs_nonneg _
  have hshK : ∀ j ∈ range (L + 1), |sh j - sc j| ≤ δ := by
    rw [← hKe]; exact inv.close_sh
  obtain ⟨j0, hj0, hm⟩ := inv.attain (by omega)
  rw [hKe] at hj0
  have hj0L : j0 ≤ L := by simpa [Nat.lt_succ_iff] using hj0
  have hmY : m ≤ Y := by
    have h1 := (abs_le.mp (hshK j0 hj0)).2
    have h2 := (abs_le.mp (hscX j0 hj0L)).2
    rw [← hm]; linarith
  have hS1 : 1 ≤ Lam sh (range (L + 1)) m := by
    unfold Lam
    calc (1 : ℝ) = Real.exp (sh j0 - m) := by rw [hm, sub_self, Real.exp_zero]
      _ ≤ ∑ j ∈ range (L + 1), Real.exp (sh j - m) :=
          Finset.single_le_sum (f := fun j => Real.exp (sh j - m))
            (fun j _ => (Real.exp_pos _).le) hj0
  have hl := inv.l_close
  rw [hKe] at hl cinv
  have hmax : max (m + Y) 0 ≤ 2 * Y := max_le (by linarith) (by linarith)
  have hbL : 0 ≤ bL Y := by
    unfold bL
    have : (0 : ℝ) ≤ 2 ^ (-19 : ℤ) := by positivity
    have : (0 : ℝ) ≤ 2 ^ (-17 : ℤ) := by positivity
    nlinarith
  have hbA : 0 ≤ bA Y := by
    unfold bA
    have : (0 : ℝ) ≤ 2 ^ (-19 : ℤ) := by positivity
    have : (0 : ℝ) ≤ 2 ^ (-8 : ℤ) := by positivity
    have : (0 : ℝ) ≤ 2 ^ (-12 : ℤ) := by positivity
    nlinarith
  have hexL : Real.exp (omL Y T m) * (1 + bL Y) ≤ 1 + GLmax Y T := by
    have h1 : omL Y T m ≤ cL0 * T + cL1 * (2 * Y) := by
      unfold omL
      have : (0 : ℝ) ≤ cL1 := by unfold cL1; positivity
      nlinarith
    have := mul_le_mul_of_nonneg_right (Real.exp_le_exp.2 h1) (by linarith : (0 : ℝ) ≤ 1 + bL Y)
    unfold GLmax; linarith
  have hexA : Real.exp (omA Y T m) * (1 + bA Y) ≤ 1 + GAmax Y T := by
    have h1 : omA Y T m ≤ cA0 * T + cA1 * (2 * Y) := by
      unfold omA
      have : (0 : ℝ) ≤ cA1 := by unfold cA1; positivity
      nlinarith
    have := mul_le_mul_of_nonneg_right (Real.exp_le_exp.2 h1) (by linarith : (0 : ℝ) ≤ 1 + bA Y)
    unfold GAmax; linarith
  have hQ0 : 0 ≤ Aabs sh vv (range (L + 1)) m :=
    Finset.sum_nonneg fun j _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _)
  have hηL : (T : ℝ) * ηL ≤ 1024 * 2 ^ (-100 : ℤ) :=
    mul_le_mul hTr (le_refl _) (by unfold ηL; positivity) (by norm_num)
  have hηA : (T : ℝ) * (ηA * (1 + V)) ≤ 1024 * 2 ^ (-96 : ℤ) * (1 + V) := by
    have : (T : ℝ) * ηA ≤ 1024 * 2 ^ (-96 : ℤ) :=
      mul_le_mul hTr (le_refl _) (by unfold ηA; positivity) (by norm_num)
    nlinarith
  have hl' : |l - Lam sh (range (L + 1)) m| ≤
      (GLmax Y T + 2 ^ (-80 : ℤ)) * Lam sh (range (L + 1)) m := by
    have h0 := close_final hl (by linarith) (mul_nonneg hTr0 (by unfold ηL; positivity))
      hexL hbL hGL
    have e : (9 / 8 : ℝ) * (1024 * 2 ^ (-100 : ℤ)) ≤ 2 ^ (-80 : ℤ) := by norm_num
    have e0 : (0 : ℝ) ≤ 2 ^ (-80 : ℤ) := by positivity
    have f2 : (2 : ℝ) ^ (-80 : ℤ) ≤ 2 ^ (-80 : ℤ) * Lam sh (range (L + 1)) m := by
      nlinarith
    have f3 : (GLmax Y T + 2 ^ (-80 : ℤ)) * Lam sh (range (L + 1)) m =
        GLmax Y T * Lam sh (range (L + 1)) m + 2 ^ (-80 : ℤ) * Lam sh (range (L + 1)) m := by ring
    generalize (2 : ℝ) ^ (-80 : ℤ) = c80 at *
    generalize (2 : ℝ) ^ (-100 : ℤ) = c100 at *
    linarith
  have hx : |l * a - Asum sh vv (range (L + 1)) m| ≤
      GAmax Y T * Aabs sh vv (range (L + 1)) m + 2 ^ (-84 : ℤ) * (1 + V) := by
    have h0 := close_final cinv hQ0
      (mul_nonneg hTr0 (mul_nonneg (by unfold ηA; positivity) (by linarith))) hexA hbA hGA
    have e : (9 / 8 : ℝ) * (1024 * 2 ^ (-96 : ℤ)) ≤ 2 ^ (-84 : ℤ) := by norm_num
    have f1 := mul_le_mul_of_nonneg_right e (by linarith : (0 : ℝ) ≤ 1 + V)
    generalize (2 : ℝ) ^ (-84 : ℤ) = c84 at *
    generalize (2 : ℝ) ^ (-96 : ℤ) = c96 at *
    linarith
  have hP : |Asum sh vv (range (L + 1)) m| ≤ Aabs sh vv (range (L + 1)) m := by
    unfold Asum Aabs
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_))
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have hR := ratio_step hS1 hP hgA hgL hGL (by positivity) hl' hx
  have hKne : (range (L + 1)).Nonempty := ⟨0, by simp⟩
  have hW := weights_close (K := range (L + 1)) (a := fun j => Real.exp (sc j))
    (b := fun j => Real.exp (sh j - m)) (v := vv) (κ := Real.exp (-m)) (p := Real.exp δ) hKne
    (Real.one_le_exp hδ0) (Real.exp_pos _) (fun j _ => Real.exp_pos _)
    (fun j hj => by
      have := (abs_le.mp (hshK j hj)).1
      rw [← Real.exp_add, ← Real.exp_add]
      exact Real.exp_le_exp.2 (by linarith))
    (fun j hj => by
      have := (abs_le.mp (hshK j hj)).2
      rw [← Real.exp_add, ← Real.exp_add]
      exact Real.exp_le_exp.2 (by linarith))
  beta_reduce at hW
  obtain ⟨hW1, hW2⟩ := hW
  have hp2 : Real.exp δ ^ 2 = Real.exp (2 * δ) := by
    rw [← Real.exp_nat_mul]; norm_num
  rw [hp2] at hW1 hW2
  have hZ : 0 < ∑ j ∈ range (L + 1), Real.exp (sc j) := Finset.sum_pos (fun j _ => Real.exp_pos _) hKne
  have hNN : |(∑ j ∈ range (L + 1), Real.exp (sc j) * vv j) / ∑ j ∈ range (L + 1), Real.exp (sc j)| ≤
      (∑ j ∈ range (L + 1), Real.exp (sc j) * |vv j|) / ∑ j ∈ range (L + 1), Real.exp (sc j) := by
    rw [abs_div, abs_of_pos hZ]
    refine div_le_div_of_nonneg_right ?_ hZ.le
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_))
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have hfin := final_ineq (T := T) (D := D) hT1' hT1 hD1 hD2 hX0 hX hδ hY
  unfold Asum Aabs Lam at hR
  have hab : (1 + 2 ^ (-8 : ℤ)) * (2 * (2 ^ (-84 : ℤ) * (1 + V))) + 2 ^ (-134 : ℤ) ≤ tolAbs V := by
    unfold tolAbs
    have e1 : (1 + (2 : ℝ) ^ (-8 : ℤ)) * (2 * 2 ^ (-84 : ℤ)) ≤ 2 ^ (-82 : ℤ) := by norm_num
    have e2 : (2 : ℝ) ^ (-82 : ℤ) + 2 ^ (-134 : ℤ) ≤ 1 / 2 ^ 80 := by norm_num
    have e3 : (0 : ℝ) ≤ 2 ^ (-134 : ℤ) := by positivity
    have f1 := mul_le_mul_of_nonneg_right e1 (by linarith : (0 : ℝ) ≤ 1 + V)
    have f2 := mul_le_mul_of_nonneg_right e2 (by linarith : (0 : ℝ) ≤ 1 + V)
    have f3 : (1 + (2 : ℝ) ^ (-8 : ℤ)) * (2 * (2 ^ (-84 : ℤ) * (1 + V))) =
        (1 + 2 ^ (-8 : ℤ)) * (2 * 2 ^ (-84 : ℤ)) * (1 + V) := by ring
    generalize (2 : ℝ) ^ (-82 : ℤ) = c82 at *
    generalize (2 : ℝ) ^ (-134 : ℤ) = c134 at *
    generalize (2 : ℝ) ^ (-84 : ℤ) = c84 at *
    generalize (2 : ℝ) ^ (-8 : ℤ) = c8 at *
    generalize (1 : ℝ) / 2 ^ 80 = c80 at *
    linarith [mul_nonneg e3 hV0]
  have hS'' : 0 ≤ (GAmax Y T + GLmax Y T) / (1 - GLmax Y T - 2 ^ (-60 : ℤ)) := by
    have h1 : (2 : ℝ) ^ (-60 : ℤ) ≤ 1e-11 := by norm_num
    have h2 : (0 : ℝ) < 2 ^ (-17 : ℤ) := by positivity
    generalize (2 : ℝ) ^ (-60 : ℤ) = c60 at *
    exact div_nonneg (by linarith) (by linarith)
  have hqs : 0 ≤ (∑ j ∈ range (L + 1), Real.exp (sc j) * |vv j|) / ∑ j ∈ range (L + 1), Real.exp (sc j) :=
    div_nonneg (Finset.sum_nonneg fun j _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _)) hZ.le
  have h1 := mul_le_mul_of_nonneg_right hW2 hS''
  have h2 := mul_le_mul_of_nonneg_left hfin hqs
  have h3 := abs_sub_le c a ((∑ j ∈ range (L + 1), Real.exp (sc j) * vv j) /
    ∑ j ∈ range (L + 1), Real.exp (sc j))
  have h4 := abs_sub_le a ((∑ j ∈ range (L + 1), Real.exp (sh j - m) * vv j) /
    ∑ j ∈ range (L + 1), Real.exp (sh j - m)) ((∑ j ∈ range (L + 1), Real.exp (sc j) * vv j) /
    ∑ j ∈ range (L + 1), Real.exp (sc j))
  have h5 := abs_sub_abs_le_abs_sub a ((∑ j ∈ range (L + 1), Real.exp (sc j) * vv j) /
    ∑ j ∈ range (L + 1), Real.exp (sc j))
  have hw0 : (0 : ℝ) ≤ 2 ^ (-8 : ℤ) := by positivity
  generalize tolRel D T X = tr at *
  generalize (2 : ℝ) ^ (-8 : ℤ) = w at *
  generalize (2 : ℝ) ^ (-134 : ℤ) = c134 at *
  generalize (2 : ℝ) ^ (-84 : ℤ) = c84 at *
  generalize Real.exp (2 * δ) = E at *
  generalize |a - (∑ j ∈ range (L + 1), Real.exp (sc j) * vv j) / ∑ j ∈ range (L + 1), Real.exp (sc j)| = D1
    at *
  have hD : D1 ≤ E * ((∑ j ∈ range (L + 1), Real.exp (sc j) * |vv j|) /
        ∑ j ∈ range (L + 1), Real.exp (sc j)) *
        ((GAmax Y T + GLmax Y T) / (1 - GLmax Y T - 2 ^ (-60 : ℤ))) + 2 * (c84 * (1 + V)) +
      (E - 1) * ((∑ j ∈ range (L + 1), Real.exp (sc j) * |vv j|) /
        ∑ j ∈ range (L + 1), Real.exp (sc j)) := by linarith
  have hA : |a| ≤ (∑ j ∈ range (L + 1), Real.exp (sc j) * |vv j|) /
        ∑ j ∈ range (L + 1), Real.exp (sc j) + D1 := by linarith
  have m1 := mul_le_mul_of_nonneg_left hD (by linarith : 0 ≤ 1 + w)
  have m2 := mul_le_mul_of_nonneg_left hA hw0
  linarith

end Num

end Claims.PrefillAttnVarlen

end
