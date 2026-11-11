import Claim
import TVA.Proof

/-!
# Rounding chains and the last step of the tolerances

Pure facts about floats and reals, with no program in sight:

* `chain_core`: the chain `bf16(fl32(p1 · e))` approximates `x · 2^g` when `p1` approximates `x`
  and `e` is `exp2`'s result;
* `w_arith`, `u_arith`: from the bounds of the sum of products, of the dot's result and of its
  rounding to the claim's tolerances `tolW`, `tolU`.
-/

namespace Claims.RecomputeWU

open TVA

/-- The chain `bf16(fl32(p1 · e))` approximates `x · G`: `p1` approximates `x` within `|x|·ε + δ` (`ε ≤ 2^-8`, `δ ≤ 2^-134`), `e` approximates `G = 2^g ≤ 2^8` within `2^-22·G + 2^-126`; then one `float32` product and one `bfloat16` rounding. The result is within `(2^-7 + 2^-10)·|x| G + 2^-125 (|x| + 1)`. -/
theorem chain_core {x G e p1 p2 kb ε δ : ℝ} (hG0 : 0 < G) (hG : G ≤ 2 ^ 8)
    (hε : ε ≤ 2 ^ (-8 : ℤ)) (hδ : δ ≤ 2 ^ (-134 : ℤ))
    (he : |e - G| ≤ 2 ^ (-22 : ℤ) * G + 2 ^ (-126 : ℤ))
    (h1 : |p1 - x| ≤ |x| * ε + δ)
    (h2 : |p2 - p1 * e| ≤ |p1 * e| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ))
    (h3 : |kb - p2| ≤ |p2| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) :
    |kb - x * G| ≤ (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (|x| * G) + 2 ^ (-125 : ℤ) * (|x| + 1) := by
  set X := |x| with hX
  have hX0 : 0 ≤ X := abs_nonneg x
  have h1' : |p1 - x| ≤ X * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := by
    refine h1.trans ?_
    have := mul_le_mul_of_nonneg_left hε hX0
    linarith
  have hE : |e| ≤ G * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ) := by
    have := abs_sub_abs_le_abs_sub e G
    have := abs_of_pos hG0
    nlinarith [abs_sub_abs_le_abs_sub e G]
  have hp1 : |p1| ≤ X * (1 + 2 ^ (-8 : ℤ)) + 2 ^ (-134 : ℤ) := by
    have := abs_sub_abs_le_abs_sub p1 x
    nlinarith [abs_sub_abs_le_abs_sub p1 x]
  have hE0 : 0 ≤ G * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ) := by positivity
  have hT : |p1 * e| ≤ (X * (1 + 2 ^ (-8 : ℤ)) + 2 ^ (-134 : ℤ)) *
      (G * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ)) := by
    rw [abs_mul]
    exact mul_le_mul hp1 hE (abs_nonneg _) (by positivity)
  have hexp : (X * (1 + 2 ^ (-8 : ℤ)) + 2 ^ (-134 : ℤ)) *
      (G * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ)) =
      (X * G) * ((1 + 2 ^ (-8 : ℤ)) * (1 + 2 ^ (-22 : ℤ))) + X * ((1 + 2 ^ (-8 : ℤ)) * 2 ^ (-126 : ℤ))
        + G * (2 ^ (-134 : ℤ) * (1 + 2 ^ (-22 : ℤ))) + 2 ^ (-134 : ℤ) * 2 ^ (-126 : ℤ) := by ring
  have hdiff : |p1 * e - x * G| ≤ (X * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) *
      (G * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ)) + X * (2 ^ (-22 : ℤ) * G + 2 ^ (-126 : ℤ)) := by
    have : p1 * e - x * G = (p1 - x) * e + x * (e - G) := by ring
    rw [this]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
    exact add_le_add (mul_le_mul h1' hE (abs_nonneg _) (by positivity))
      (mul_le_mul_of_nonneg_left he hX0)
  have hdexp : (X * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) *
      (G * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ)) + X * (2 ^ (-22 : ℤ) * G + 2 ^ (-126 : ℤ)) =
      (X * G) * (2 ^ (-8 : ℤ) * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-22 : ℤ)) +
        X * (2 ^ (-8 : ℤ) * 2 ^ (-126 : ℤ) + 2 ^ (-126 : ℤ)) +
        G * (2 ^ (-134 : ℤ) * (1 + 2 ^ (-22 : ℤ))) + 2 ^ (-134 : ℤ) * 2 ^ (-126 : ℤ) := by ring
  have hp2 : |p2| ≤ |p1 * e| + |p2 - p1 * e| := by
    have := abs_sub_abs_le_abs_sub p2 (p1 * e)
    linarith
  have hfin : |kb - x * G| ≤ |kb - p2| + |p2 - p1 * e| + |p1 * e - x * G| := by
    have h := abs_sub_le kb p2 (x * G)
    have h' := abs_sub_le p2 (p1 * e) (x * G)
    linarith
  have hY : 0 ≤ X * G := by positivity
  have hG' : G * 1 ≤ 2 ^ 8 := by linarith
  rw [hexp] at hT
  rw [hdexp] at hdiff
  have hbound : |p1 * e| * 2 ^ (-24 : ℤ) ≥ 0 := by positivity
  norm_num at *
  nlinarith [abs_nonneg (p1 * e), abs_nonneg (p2 - p1 * e), abs_nonneg p2]


/-- The last step of the `w` bound: from the bounds of the sum `S` of the products of the dot, of
the dot's result `r` and of its rounding `o`, to the claim's tolerance. -/
theorem w_arith {M F wref S r o L : ℝ} (hM : 0 ≤ M) (hF : 0 ≤ F) (hw : |wref| ≤ M)
    (hS : |S - wref| ≤ (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * M + 2 ^ (-125 : ℤ) * F)
    (hL : L ≤ (1 + (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ))) * M + 3 / 2 * 2 ^ (-125 : ℤ) * F)
    (hr : |r - S| ≤ 66 * 2 ^ (-22 : ℤ) * L + 66 * 2 ^ (-126 : ℤ))
    (ho : |o - r| ≤ |r| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) :
    |o - wref| ≤ 2 ^ (-6 : ℤ) * M + 2 ^ (-124 : ℤ) * F + 2 ^ (-119 : ℤ) := by
  have h1 : |r| ≤ |wref| + |S - wref| + |r - S| := by
    have := abs_sub_abs_le_abs_sub r wref
    have h2 := abs_sub_le r S wref
    linarith [abs_sub_abs_le_abs_sub r wref]
  have h3 : |o - wref| ≤ |o - r| + |r - S| + |S - wref| := by
    have h := abs_sub_le o r wref
    have h' := abs_sub_le r S wref
    linarith
  norm_num at *
  nlinarith [abs_nonneg r, abs_nonneg (r - S), abs_nonneg (S - wref)]

/-- The last step of the `u` bound. -/
theorem u_arith {ref r o L : ℚ} (hL : 0 ≤ L)
    (hr : |r - ref| ≤ 66 * 2 ^ (-22 : ℤ) * L + 66 * 2 ^ (-126 : ℤ))
    (ho : |o - r| ≤ |r| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) :
    |o - ref| ≤ 2 ^ (-8 : ℤ) * |ref| + 2 ^ (-15 : ℤ) * L + 2 ^ (-119 : ℤ) := by
  have h1 : |r| ≤ |ref| + |r - ref| := by
    have := abs_sub_abs_le_abs_sub r ref
    linarith
  have h3 : |o - ref| ≤ |o - r| + |r - ref| := abs_sub_le o r ref
  norm_num at *
  nlinarith [abs_nonneg r, abs_nonneg (r - ref)]


/-- `2^g > 0`. -/
theorem two_rpow_pos (g : ℝ) : 0 < (2 : ℝ) ^ g := Real.rpow_pos_of_pos (by norm_num) g

/-- `2^g ≤ 2^8` for `g ≤ 8`. -/
theorem two_rpow_le {g : ℝ} (hg : g ≤ 8) : (2 : ℝ) ^ g ≤ 2 ^ 8 := by
  have := Real.rpow_le_rpow_of_exponent_le (x := (2 : ℝ)) (by norm_num) hg
  simpa using this

/-- A bound `|x| ≤ 2^40` on the product of a covered `k` or `v` and a covered `beta`. -/
theorem abs_kb_le {k β : ℝ} (hk : |k| ≤ 2 ^ 32) (hβ : |β| ≤ 2 ^ 8) : |k * β| ≤ 2 ^ 40 := by
  rw [abs_mul]
  calc |k| * |β| ≤ 2 ^ 32 * 2 ^ 8 :=
        mul_le_mul hk hβ (abs_nonneg _) (by positivity)
    _ = 2 ^ 40 := by norm_num

end Claims.RecomputeWU
