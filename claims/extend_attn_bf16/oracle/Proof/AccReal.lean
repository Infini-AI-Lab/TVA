import Proof.CoreLemmas

/-!
# The accumulator update of a trip, over the reals

`acc_pref_real` (cached-part trips: `acc * re_scale + dot * v_scale`) and `acc_ext_real` (extend trips:
`tl.dot(p, v, acc * re_scale)`): from the errors of the float operations, the new accumulator is
`re_scale · acc + Σ p16 · v` up to `ε1 · |re_scale · acc| + ε2 · Σ |p16||v| + eps1 · (1 + Σ|v|)`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The accumulator's rate on the old accumulator in a cached-part trip. -/
def epsP1 : ℝ := 101 / (100 * 2 ^ 22)
/-- The accumulator's rate on the old accumulator in an extend trip. -/
def epsE1 : ℝ := 671 / (10 * 2 ^ 22)
/-- The accumulator's rate on the new weights (both kinds of trip). -/
def eps2 : ℝ := 135 / 2 ^ 23

/-- The constant `epsP1` is nonnegative, at most 2^-15, and satisfies the exp lower bounds against `g1p`. -/
theorem epsP1_ok : 0 ≤ epsP1 ∧ epsP1 ≤ 1 / 2 ^ 15 ∧ 1 + epsP1 ≤ Real.exp g1p ∧
    epsP1 * (1 + ub) ≤ expm1 g1p ∧ 0 ≤ g1p := by
  have h := Real.add_one_le_exp g1p
  unfold epsP1 ub g1p expm1 at *
  refine ⟨by norm_num, by norm_num, by linarith [h, (by norm_num : (101 : ℝ) / (100 * 2 ^ 22) ≤ 9 / 2 ^ 25)], ?_, by norm_num⟩
  linarith [h, (by norm_num : (101 : ℝ) / (100 * 2 ^ 22) * (1 + 1 / 2 ^ 8) ≤ 9 / 2 ^ 25)]

/-- The constant `epsE1` is nonnegative, at most 2^-15, and satisfies the exp lower bounds against `g1e`. -/
theorem epsE1_ok : 0 ≤ epsE1 ∧ epsE1 ≤ 1 / 2 ^ 15 ∧ 1 + epsE1 ≤ Real.exp g1e ∧
    epsE1 * (1 + ub) ≤ expm1 g1e ∧ 0 ≤ g1e := by
  have h := Real.add_one_le_exp g1e
  unfold epsE1 ub g1e expm1 at *
  refine ⟨by norm_num, by norm_num, by linarith [h, (by norm_num : (671 : ℝ) / (10 * 2 ^ 22) ≤ 68 / 2 ^ 22)], ?_, by norm_num⟩
  linarith [h, (by norm_num : (671 : ℝ) / (10 * 2 ^ 22) * (1 + 1 / 2 ^ 8) ≤ 68 / 2 ^ 22)]

/-- The constant `eps2` is nonnegative, at most 2^-15, and is dominated by `expm1 g2` at scale `1 + ub`. -/
theorem eps2_ok : 0 ≤ eps2 ∧ eps2 ≤ 1 / 2 ^ 15 ∧ eps2 * (1 + ub) ≤ expm1 g2 := by
  have h := Real.add_one_le_exp g2
  unfold eps2 ub g2 expm1 at *
  refine ⟨by norm_num, by norm_num, ?_⟩
  linarith [h, (by norm_num : (135 : ℝ) / 2 ^ 23 * (1 + 1 / 2 ^ 8) ≤ 68 / 2 ^ 22)]

/-- **An extend trip's accumulator** `tl.dot(p16, v, acc * re_scale)`: `c = fl(re · acc)`, then the
device's `dot_add` step. -/
theorem acc_ext_real {ra c a' Sg A Lp V1 : ℝ}
    (hc : |c - ra| ≤ 1 / 2 ^ 24 * |ra| + 1 / 2 ^ 150)
    (hd : |a' - (c + Sg)| ≤
      (66 / 2 ^ 22 + 3 * (1 / 2 ^ 24) * (1 + 66 / 2 ^ 22)) * (|c| + A) +
        (66 / 2 ^ 126 * (1 + 3 * (1 / 2 ^ 24)) + 4 * (1 / 2 ^ 150)))
    (hA : A ≤ Lp + 1 / 2 ^ 118 * (1 + V1)) (_hA0 : 0 ≤ A) (hV1 : 0 ≤ V1) (hLp : 0 ≤ Lp) :
    |a' - (ra + Sg)| ≤ epsE1 * |ra| + eps2 * Lp + eps1 * (1 + V1) := by
  have hcb : |c| ≤ (1 + 1 / 2 ^ 24) * |ra| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub c ra
    linarith
  have htri : |a' - (ra + Sg)| ≤ |a' - (c + Sg)| + |c - ra| := by
    have := abs_add_le (a' - (c + Sg)) (c - ra)
    rw [show a' - (c + Sg) + (c - ra) = a' - (ra + Sg) by ring] at this
    exact this
  have hsr : (0 : ℝ) ≤ 66 / 2 ^ 22 + 3 * (1 / 2 ^ 24) * (1 + 66 / 2 ^ 22) := by norm_num
  have h1 := mul_le_mul_of_nonneg_left (add_le_add hcb hA) hsr
  have hra := abs_nonneg ra
  unfold epsE1 eps2 eps1
  nlinarith [h1, hd, hc, htri, mul_nonneg hV1 hra]

/-- **A cached-part trip's accumulator** `acc * re_scale + dot * v_scale`: `x = fl(re · acc)`,
`D` the dot from a zero accumulator, `y = fl(D · ks)`, `a' = x ⊕ y`. -/
theorem acc_pref_real {ra x Dd ks y a' Sg A Lp V1 : ℝ}
    (hx : |x - ra| ≤ 1 / 2 ^ 24 * |ra| + 1 / 2 ^ 150)
    (hD : |Dd - Sg| ≤ 66 / 2 ^ 22 * A + 66 / 2 ^ 126)
    (hks : |ks - 1| ≤ 1 / 2 ^ 24 + 1 / 2 ^ 150)
    (hy : |y - Dd * ks| ≤ 1 / 2 ^ 24 * |Dd * ks| + 1 / 2 ^ 150)
    (ha : |a' - (x + y)| ≤ 3 * (1 / 2 ^ 24) * (|x| + |y|) + 4 * (1 / 2 ^ 150))
    (hA : A ≤ Lp + 1 / 2 ^ 118 * (1 + V1)) (_hA0 : 0 ≤ A) (hV1 : 0 ≤ V1)
    (hSg : |Sg| ≤ Lp) (hLp : 0 ≤ Lp) :
    |a' - (ra + Sg)| ≤ epsP1 * |ra| + eps2 * Lp + eps1 * (1 + V1) := by
  have hxb : |x| ≤ (1 + 1 / 2 ^ 24) * |ra| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub x ra
    linarith
  have hDb : |Dd| ≤ (1 + 66 / 2 ^ 22) * Lp + (66 / 2 ^ 22 * (1 / 2 ^ 118) * (1 + V1) + 66 / 2 ^ 126) := by
    have := abs_sub_abs_le_abs_sub Dd Sg
    nlinarith [mul_le_mul_of_nonneg_left hA (by norm_num : (0 : ℝ) ≤ 66 / 2 ^ 22)]
  have hksb : |ks| ≤ 1 + 1 / 2 ^ 24 + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub ks 1
    simp only [abs_one] at this
    linarith
  have hDk : |Dd * ks| ≤ (1 + 1 / 2 ^ 24 + 1 / 2 ^ 150) * |Dd| := by
    have := mul_le_mul_of_nonneg_left hksb (abs_nonneg Dd)
    rw [abs_mul]
    linarith
  have hyb : |y| ≤ (1 + 1 / 2 ^ 24) * |Dd * ks| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub y (Dd * ks)
    linarith
  have hDks : |Dd * ks - Dd| ≤ |Dd| * (1 / 2 ^ 24 + 1 / 2 ^ 150) := by
    rw [show Dd * ks - Dd = Dd * (ks - 1) by ring, abs_mul]
    exact mul_le_mul_of_nonneg_left hks (abs_nonneg _)
  have hySg : |y - Sg| ≤ |y - Dd * ks| + |Dd * ks - Dd| + |Dd - Sg| := by
    have h1 : |y - Dd| ≤ |y - Dd * ks| + |Dd * ks - Dd| := by
      have := abs_add_le (y - Dd * ks) (Dd * ks - Dd)
      rwa [show y - Dd * ks + (Dd * ks - Dd) = y - Dd by ring] at this
    have h2 : |y - Sg| ≤ |y - Dd| + |Dd - Sg| := by
      have := abs_add_le (y - Dd) (Dd - Sg)
      rwa [show y - Dd + (Dd - Sg) = y - Sg by ring] at this
    linarith
  have htri : |a' - (ra + Sg)| ≤ |a' - (x + y)| + |x - ra| + |y - Sg| := by
    have h1 := abs_add_le (a' - (x + y)) (x - ra)
    have h2 := abs_add_le (a' - (x + y) + (x - ra)) (y - Sg)
    rw [show a' - (x + y) + (x - ra) + (y - Sg) = a' - (ra + Sg) by ring] at h2
    linarith
  have hra := abs_nonneg ra
  have hDm := abs_nonneg Dd
  unfold epsP1 eps2 eps1
  have hA' := mul_le_mul_of_nonneg_left hA (by norm_num : (0 : ℝ) ≤ 66 / 2 ^ 22)
  linarith [hxb, hyb, hDk, hDb, hDks, hySg, htri, ha, hx, hy, hD, hA']

end

end Claims.ExtendAttn
