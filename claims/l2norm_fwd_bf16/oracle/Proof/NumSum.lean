import TVA.Proof

/-!
# The squared-sum error bound

`tl.sum1c` computes, on each row, a tree-shaped sum of `n ≤ 512` `f32` values. `sumc_err_covers`
bounds its error by `Tree.errBound (3u) (4η) (n-1) (n-1) B`. Here we turn that into the concrete
constant used downstream, and compose it with the error of the per-element squares `fl(a_j²)`.
-/

namespace Claims.L2NormFwd

open TVA

/-- The geometric series bound: for `c ≥ 0`, `(1 + c)^k · (1 − k·c) ≤ 1`. -/
theorem geom_mul_le_one (c : ℚ) (hc : 0 ≤ c) (k : Nat) :
    (1 + c) ^ k * (1 - (k : ℚ) * c) ≤ 1 := by
  induction k with
  | zero => norm_num
  | succ k ih =>
    have h1 : (0:ℚ) ≤ 1 + c := by linarith
    have hge : (1:ℚ) ≤ (1 + c) ^ (k + 1) := one_le_pow₀ (by linarith)
    have hmul : (1 + c) * ((1 + c) ^ k * (1 - (k : ℚ) * c)) ≤ (1 + c) * 1 :=
      mul_le_mul_of_nonneg_left ih h1
    have hpow : (1 + c) ^ (k + 1) = (1 + c) ^ k * (1 + c) := pow_succ _ _
    have hcast : ((k + 1 : Nat) : ℚ) = (k : ℚ) + 1 := by push_cast; ring
    rw [hpow, hcast]
    nlinarith [hmul, hge, hc, mul_nonneg hc (pow_nonneg h1 k)]

/-- `(1 + 3·2⁻²⁴)^511 ≤ 1 + (3/4)·2⁻¹³`, the sharper form of `one_add_pow_le`. -/
theorem one_add_pow_511_le_quarter :
    ((1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 511 : ℚ) ≤ 1 + (3/4) * (2 : ℚ) ^ (-13 : ℤ) := by
  have hc : (0:ℚ) ≤ 3 * (2 : ℚ) ^ (-24 : ℤ) := by positivity
  have key : (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 511 *
      (1 - (511 : ℚ) * (3 * (2 : ℚ) ^ (-24 : ℤ))) ≤ 1 :=
    geom_mul_le_one _ hc 511
  have hpos : (0:ℚ) < 1 - (511 : ℚ) * (3 * (2 : ℚ) ^ (-24 : ℤ)) := by norm_num
  have h2 : (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 511 ≤
      1 / (1 - (511 : ℚ) * (3 * (2 : ℚ) ^ (-24 : ℤ))) := by
    rw [le_div_iff₀ hpos]
    simpa using key
  have h3 : (1:ℚ) / (1 - (511 : ℚ) * (3 * (2 : ℚ) ^ (-24 : ℤ))) ≤
      1 + (3/4) * (2 : ℚ) ^ (-13 : ℤ) := by
    norm_num
  exact h2.trans h3

/-- `(1 + 3·2⁻²⁴)^k ≤ 1 + (3/4)·2⁻¹³` for `k ≤ 511`. -/
theorem one_add_pow_le_quarter (k : Nat) (hk : k ≤ 511) :
    (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ k ≤ 1 + (3/4) * (2 : ℚ) ^ (-13 : ℤ) := by
  have hmono : (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ k ≤ (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 511 :=
    pow_le_pow_right₀ (by norm_num) hk
  exact hmono.trans one_add_pow_511_le_quarter

/-- `(1 + 3·2⁻²⁴)^k ≤ 1 + 2⁻¹³` for `k ≤ 511`: the tree of at most 511 additions. -/
theorem one_add_pow_le (k : Nat) (hk : k ≤ 511) :
    (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ k ≤ 1 + (2 : ℚ) ^ (-13 : ℤ) := by
  have h := one_add_pow_le_quarter k hk
  have h2 : (1 + (3/4) * (2 : ℚ) ^ (-13 : ℤ) : ℚ) ≤ 1 + (2 : ℚ) ^ (-13 : ℤ) := by norm_num
  exact h.trans h2

/-- The `sumc_err_covers` bound, specialised to `f32` and `n ≤ 512`. -/
theorem errBound_le {n : Nat} (hn : n ≤ 512) {B : ℚ} (hB0 : 0 ≤ B) :
    Tree.errBound (3 * (2 : ℚ) ^ (-24 : ℤ)) (4 * (2 : ℚ) ^ (-150 : ℤ)) (n - 1) (n - 1) B
      ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * B + (2 : ℚ) ^ (-138 : ℤ) := by
  rcases Nat.eq_zero_or_pos n with hn0 | hnpos
  · subst hn0
    simp [Tree.errBound]
    have h1 : (0:ℚ) ≤ (3/4) * (2 : ℚ) ^ (-13 : ℤ) * B :=
      mul_nonneg (by norm_num) hB0
    have h2 : (0:ℚ) < (2 : ℚ) ^ (-138 : ℤ) := by norm_num
    linarith
  · have hd : n - 1 ≤ 511 := by omega
    have hdcast : ((n - 1 : Nat) : ℚ) ≤ 511 := by exact_mod_cast hd
    have hpow := one_add_pow_le_quarter (n - 1) hd
    unfold Tree.errBound
    have hcoef : (1 + 3 * (2:ℚ)^(-24:ℤ)) ^ (n-1) - 1 ≤ (3/4)*(2:ℚ)^(-13:ℤ) := by linarith
    have e1 : ((1 + 3*(2:ℚ)^(-24:ℤ))^(n-1) - 1) * B ≤ ((3/4)*(2:ℚ)^(-13:ℤ)) * B :=
      mul_le_mul_of_nonneg_right hcoef hB0
    have hpowpos : (0:ℚ) ≤ (1 + 3*(2:ℚ)^(-24:ℤ))^(n-1) := by positivity
    have hA : (0:ℚ) ≤ 4 * (2:ℚ)^(-150:ℤ) := by positivity
    have step1 : 4*(2:ℚ)^(-150:ℤ) * ((n-1:Nat):ℚ) ≤ 4*(2:ℚ)^(-150:ℤ) * 511 :=
      mul_le_mul_of_nonneg_left hdcast hA
    have step3 := mul_le_mul step1 hpow hpowpos
      (by positivity : (0:ℚ) ≤ 4*(2:ℚ)^(-150:ℤ)*511)
    have hnum : 4*(2:ℚ)^(-150:ℤ)*511*(1+(3/4)*(2:ℚ)^(-13:ℤ)) ≤ (2:ℚ)^(-138:ℤ) := by
      norm_num
    nlinarith

/-- Compose the tree-sum bound with the per-element rounding of the squares. -/
theorem sum_sq_compose {n : Nat} (hn : n ≤ 512) {a x : Nat → ℝ} {r : ℝ}
    (ha0 : ∀ j, j < n → 0 ≤ a j)
    (hx : ∀ j, j < n → |x j - a j| ≤ (2 : ℝ) ^ (-24 : ℤ) * a j + (2 : ℝ) ^ (-150 : ℤ))
    (hrt : |r - ∑ j ∈ Finset.range n, x j| ≤
      (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (∑ j ∈ Finset.range n, |x j|) +
        (2 : ℝ) ^ (-138 : ℤ)) :
    |r - ∑ j ∈ Finset.range n, a j| ≤
      (2 : ℝ) ^ (-13 : ℤ) * (∑ j ∈ Finset.range n, a j) + (3/4) * (2 : ℝ) ^ (-137 : ℤ) := by
  have hA0 : (0:ℝ) ≤ ∑ j ∈ Finset.range n, a j :=
    Finset.sum_nonneg (fun j hj => ha0 j (Finset.mem_range.mp hj))
  -- (1) bound on the sum of `|x j|`
  have hstep : ∀ j ∈ Finset.range n,
      |x j| ≤ (1 + (2:ℝ)^(-24:ℤ)) * a j + (2:ℝ)^(-150:ℤ) := by
    intro j hj
    have hj' : j < n := Finset.mem_range.mp hj
    have h1 : |x j| ≤ a j + |x j - a j| := by
      calc |x j| = |a j + (x j - a j)| := by ring_nf
        _ ≤ |a j| + |x j - a j| := abs_add_le _ _
        _ = a j + |x j - a j| := by rw [abs_of_nonneg (ha0 j hj')]
    linarith [hx j hj']
  have hsx : ∑ j ∈ Finset.range n, |x j| ≤
      (1 + (2:ℝ)^(-24:ℤ)) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ) := by
    have h := Finset.sum_le_sum hstep
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul] at h
    exact h
  -- (2) bound on the difference of the two sums
  have hdx : |(∑ j ∈ Finset.range n, x j) - (∑ j ∈ Finset.range n, a j)|
      ≤ (2:ℝ)^(-24:ℤ) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ) := by
    have hsub : (∑ j ∈ Finset.range n, x j) - (∑ j ∈ Finset.range n, a j)
        = ∑ j ∈ Finset.range n, (x j - a j) := (Finset.sum_sub_distrib _ _).symm
    rw [hsub]
    calc |∑ j ∈ Finset.range n, (x j - a j)|
        ≤ ∑ j ∈ Finset.range n, |x j - a j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ Finset.range n, ((2:ℝ)^(-24:ℤ) * a j + (2:ℝ)^(-150:ℤ)) :=
          Finset.sum_le_sum (fun j hj => hx j (Finset.mem_range.mp hj))
      _ = (2:ℝ)^(-24:ℤ) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
            nsmul_eq_mul]
  -- (3) the triangle inequality
  have htri : |r - ∑ j ∈ Finset.range n, a j| ≤
      |r - ∑ j ∈ Finset.range n, x j| +
        |(∑ j ∈ Finset.range n, x j) - (∑ j ∈ Finset.range n, a j)| := by
    have heq : r - ∑ j ∈ Finset.range n, a j
        = (r - ∑ j ∈ Finset.range n, x j) +
          ((∑ j ∈ Finset.range n, x j) - (∑ j ∈ Finset.range n, a j)) := by ring
    rw [heq]
    exact abs_add_le _ _
  have hcomb : |r - ∑ j ∈ Finset.range n, a j| ≤
      (3/4) * (2:ℝ)^(-13:ℤ) * (∑ j ∈ Finset.range n, |x j|) + (2:ℝ)^(-138:ℤ) +
        ((2:ℝ)^(-24:ℤ) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ)) := by
    have hh := add_le_add hrt hdx
    linarith [hh, htri]
  have hterm : (3/4) * (2:ℝ)^(-13:ℤ) * (∑ j ∈ Finset.range n, |x j|) ≤
      (3/4) * (2:ℝ)^(-13:ℤ) *
        ((1 + (2:ℝ)^(-24:ℤ)) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ)) :=
    mul_le_mul_of_nonneg_left hsx (by norm_num)
  -- (4) numeric facts
  have hncast : (n:ℝ) ≤ 512 := by exact_mod_cast hn
  have hv0 : (0:ℝ) ≤ (2:ℝ)^(-150:ℤ) := by positivity
  have hnv : (n:ℝ) * (2:ℝ)^(-150:ℤ) ≤ (2:ℝ)^(-141:ℤ) := by
    have h := mul_le_mul_of_nonneg_right hncast hv0
    have h512 : (512:ℝ) * (2:ℝ)^(-150:ℤ) = (2:ℝ)^(-141:ℤ) := by norm_num
    linarith [h, h512]
  have hcoefA : (3/4)*(2:ℝ)^(-13:ℤ) * (1 + (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-24:ℤ) ≤
      (2:ℝ)^(-13:ℤ) := by norm_num
  have hAterm : ((3/4)*(2:ℝ)^(-13:ℤ) * (1 + (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-24:ℤ)) *
      (∑ j ∈ Finset.range n, a j) ≤ (2:ℝ)^(-13:ℤ) * (∑ j ∈ Finset.range n, a j) :=
    mul_le_mul_of_nonneg_right hcoefA hA0
  have hcle : (0:ℝ) ≤ (3/4)*(2:ℝ)^(-13:ℤ) + 1 := by norm_num
  have hnvterm : ((3/4)*(2:ℝ)^(-13:ℤ) + 1) * ((n:ℝ) * (2:ℝ)^(-150:ℤ)) ≤
      ((3/4)*(2:ℝ)^(-13:ℤ) + 1) * (2:ℝ)^(-141:ℤ) :=
    mul_le_mul_of_nonneg_left hnv hcle
  have hconst : (2:ℝ)^(-138:ℤ) + ((3/4)*(2:ℝ)^(-13:ℤ) + 1) * (2:ℝ)^(-141:ℤ)
      ≤ (3/4)*(2:ℝ)^(-137:ℤ) := by norm_num
  -- (5) put the pieces together
  have hmain : |r - ∑ j ∈ Finset.range n, a j| ≤
      (3/4) * (2:ℝ)^(-13:ℤ) * ((1 + (2:ℝ)^(-24:ℤ)) * (∑ j ∈ Finset.range n, a j) +
        (n:ℝ) * (2:ℝ)^(-150:ℤ)) + (2:ℝ)^(-138:ℤ) +
        ((2:ℝ)^(-24:ℤ) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ)) := by
    have hh := add_le_add hterm (le_refl ((2:ℝ)^(-138:ℤ) +
        ((2:ℝ)^(-24:ℤ) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ))))
    linarith [hcomb, hh]
  have hexp : (3/4) * (2:ℝ)^(-13:ℤ) *
        ((1 + (2:ℝ)^(-24:ℤ)) * (∑ j ∈ Finset.range n, a j) +
          (n:ℝ) * (2:ℝ)^(-150:ℤ)) + (2:ℝ)^(-138:ℤ) +
        ((2:ℝ)^(-24:ℤ) * (∑ j ∈ Finset.range n, a j) + (n:ℝ) * (2:ℝ)^(-150:ℤ))
      = ((3/4) * (2:ℝ)^(-13:ℤ) * (1 + (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-24:ℤ)) *
          (∑ j ∈ Finset.range n, a j) +
        ((3/4) * (2:ℝ)^(-13:ℤ) + 1) * ((n:ℝ) * (2:ℝ)^(-150:ℤ)) + (2:ℝ)^(-138:ℤ) := by
    ring
  have hfin : ((3/4) * (2:ℝ)^(-13:ℤ) * (1 + (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-24:ℤ)) *
          (∑ j ∈ Finset.range n, a j) +
        ((3/4) * (2:ℝ)^(-13:ℤ) + 1) * ((n:ℝ) * (2:ℝ)^(-150:ℤ)) + (2:ℝ)^(-138:ℤ)
      ≤ (2:ℝ)^(-13:ℤ) * (∑ j ∈ Finset.range n, a j) + (3/4)*(2:ℝ)^(-137:ℤ) := by
    linarith [hAterm, hnvterm, hconst]
  exact hmain.trans (hexp ▸ hfin)

end Claims.L2NormFwd
