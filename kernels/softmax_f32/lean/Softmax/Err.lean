import Softmax.Model

/-!
# `softmax_f32`: the error of one row

The numerical part of the proof, for one row of the kernel, from what the model's operations allow
(the loaded row, `tl.max`, `row − max`, `tl.exp`, `tl.sum`, `/`) and the three device facts the
claim cites (`Rules.Num.ExpF32`, `ExpF32NegInf`, `DivFull`):

* `exp_lane`: the subtraction `x − m` (one IEEE rounding, `s = x − m ≤ 0`) and `tl.exp`
  (`ExpF32` at `X = |t|`) give `e^s` within `(2^-22 + 2^-33)·e^s + (3·2^-24 + 2^-33)·|s|·e^s +
  2^-139`: relative, with a part that grows with `|s|`;
* `abs_mul_exp_le16`: `|s|·e^s ≤ 16·e^s + 2^-17`, so over a row the lanes' errors add up to at most
  `53·2^-24` of the exact sum (`|s|` above `16` contributes almost nothing);
* `sum_common`: the row sum (`sum_err_covers`, any tree, depth up to `B − 1`) is within
  `(B + 69)·2^-24` of the exact sum, relatively, for `B ≤ 2^14`, so the common factor
  `A / S` is within `(B + 86)·2^-24` of `1`;
* `final_common`: with `DivFull`, each output is within
  `(2^-21 + 2^-30 + (3·2^-24 + 2^-30)·|s|)·σ + 2^-133/A` of `(A / S)·σ`;
* `max_facts`, `lane_facts`, `row_bound`: the same for the model's values, with the masked lanes
  (`-∞`, so `tl.exp` gives `+0`, `ExpF32NegInf`) and the `-∞` inputs.
-/

namespace Kernels.Softmax

open TVA TVA.Math.Float

/-! ## Facts about `exp` -/

theorem le_half_exp {y : ℝ} (hy : 0 ≤ y) : y ≤ Real.exp y / 2 := by
  have h := Real.quadratic_le_exp_of_nonneg hy
  nlinarith [sq_nonneg (y - 1)]

theorem abs_mul_exp_le {t : ℝ} (ht : t ≤ 0) : |t| * Real.exp t ≤ 1 / 2 := by
  rw [abs_of_nonpos ht]
  have h := le_half_exp (neg_nonneg.mpr ht)
  have he : Real.exp (-t) * Real.exp t = 1 := by rw [← Real.exp_add]; simp
  have hpos := Real.exp_pos t
  nlinarith

theorem abs_exp_sub_one_le' (d : ℝ) : |Real.exp d - 1| ≤ |d| * Real.exp |d| := by
  rcases le_or_gt 0 d with hd | hd
  · have h0 := Real.add_one_le_exp d
    rw [abs_of_nonneg hd, abs_of_nonneg (by linarith : 0 ≤ Real.exp d - 1)]
    have h1 := Real.add_one_le_exp (-d)
    have he : Real.exp (-d) * Real.exp d = 1 := by rw [← Real.exp_add]; simp
    have hpos := Real.exp_pos d
    nlinarith
  · have h0 := Real.add_one_le_exp d
    have hlt : Real.exp d < 1 := Real.exp_lt_one_iff.mpr hd
    rw [abs_of_neg hd, abs_of_neg (by linarith : Real.exp d - 1 < 0)]
    have h1 : 1 ≤ Real.exp (-d) := Real.one_le_exp (by linarith)
    nlinarith

/-- `η = 2^-150`, the absolute rounding error of `float32`. -/
theorem exp_eta_le : Real.exp (1 / 2 ^ 150) ≤ 1 + 2 * (1 / 2 ^ 150) := by
  have hη : (0 : ℝ) ≤ 1 / 2 ^ 150 := by positivity
  have h := Real.abs_exp_sub_one_le (x := 1 / 2 ^ 150) (by rw [abs_of_nonneg hη]; norm_num)
  rw [abs_of_nonneg hη] at h
  have := (abs_le.mp h).2
  linarith

theorem exp_le_two {t : ℝ} (ht : t ≤ 1 / 2 ^ 150) : Real.exp t ≤ 2 := by
  have h1 := Real.exp_le_exp.mpr ht
  have h2 := exp_eta_le
  norm_num at h2 ⊢
  linarith

/-- `e^(-n) ≤ 2^-n`. -/
theorem exp_neg_nat_le (n : ℕ) : Real.exp (-(n : ℝ)) ≤ 1 / 2 ^ n := by
  have h1 : (2 : ℝ) ≤ Real.exp 1 := by have := Real.add_one_le_exp (1 : ℝ); linarith
  have h2 : Real.exp (-(n : ℝ)) = Real.exp (-1) ^ n := by rw [← Real.exp_nat_mul]; ring_nf
  have h3 : Real.exp (-1) ≤ 1 / 2 := by
    rw [Real.exp_neg, inv_eq_one_div]
    exact one_div_le_one_div_of_le (by norm_num) h1
  rw [h2]
  calc Real.exp (-1) ^ n ≤ (1 / 2 : ℝ) ^ n := pow_le_pow_left₀ (Real.exp_pos _).le h3 n
    _ = 1 / 2 ^ n := by rw [one_div_pow]

/-- `e^-16 ≤ 2^-21` (from `e ≥ 5/2`). -/
theorem exp_neg16_le : Real.exp (-16) ≤ 1 / 2 ^ 21 := by
  have h1 : (5 / 2 : ℝ) ≤ Real.exp 1 := by
    have := Real.quadratic_le_exp_of_nonneg (x := (1 : ℝ)) (by norm_num)
    norm_num at this; linarith
  have h2 : Real.exp 16 = Real.exp 1 ^ 16 := by rw [← Real.exp_nat_mul]; norm_num
  have h3 : (2 : ℝ) ^ 21 ≤ Real.exp 16 := by
    rw [h2]
    calc (2 : ℝ) ^ 21 ≤ (5 / 2) ^ 16 := by norm_num
      _ ≤ Real.exp 1 ^ 16 := pow_le_pow_left₀ (by norm_num) h1 16
  rw [Real.exp_neg, inv_eq_one_div]
  exact one_div_le_one_div_of_le (by positivity) h3

/-- `|s|·e^s ≤ 16·e^s + 2^-17` for `s ≤ 0`: below `16` trivially, and beyond `16`,
`|s|·e^s ≤ 16·e^-16`. -/
theorem abs_mul_exp_le16 {s : ℝ} (hs : s ≤ 0) : |s| * Real.exp s ≤ 16 * Real.exp s + 1 / 2 ^ 17 := by
  rw [abs_of_nonpos hs]
  have hE := Real.exp_pos s
  rcases le_or_gt (-s) 16 with h | h
  · nlinarith
  · have h1 : Real.exp s = Real.exp (-16) * Real.exp (-(-s - 16)) := by
      rw [← Real.exp_add]; ring_nf
    have h2 : -s ≤ 16 * Real.exp (-s - 16) := by
      have := Real.add_one_le_exp (-s - 16); nlinarith
    have h3 : Real.exp (-s - 16) * Real.exp (-(-s - 16)) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    have h4 := exp_neg16_le
    have h6 := Real.exp_pos (-(-s - 16))
    have h7 : (-s) * Real.exp s ≤ 16 * Real.exp (-16) := by
      rw [h1]
      calc (-s) * (Real.exp (-16) * Real.exp (-(-s - 16))) ≤
            (16 * Real.exp (-s - 16)) * (Real.exp (-16) * Real.exp (-(-s - 16))) :=
          mul_le_mul_of_nonneg_right h2 (by positivity)
        _ = 16 * Real.exp (-16) * (Real.exp (-s - 16) * Real.exp (-(-s - 16))) := by ring
        _ = 16 * Real.exp (-16) := by rw [h3, mul_one]
    nlinarith

/-- **One lane.** `x − m` rounded (`t`, within `u·|s| + η` of `s = x − m ≤ 0`), then `tl.exp` with
`ExpF32`'s bound at `X = |t|`: the result is within `(2^-22 + 2^-33)·e^s +
(3·2^-24 + 2^-33)·|s|·e^s + 2^-139` of `e^s`. For `|s| ≤ 1024` the rounding moves `e^s` by a
factor `e^(t−s)`, `|t − s| ≤ 2^-13`; beyond, everything is below `2^-139`. -/
theorem exp_lane {s t e : ℝ} (hs : s ≤ 0) (hts : |t - s| ≤ |s| * (1 / 2 ^ 24) + 1 / 2 ^ 150)
    (he : |e - Real.exp t| ≤ (1 / 2 ^ 22 + 1 / 2 ^ 23 * |t|) * Real.exp t + 1 / 2 ^ 140) :
    |e - Real.exp s| ≤ (1 / 2 ^ 22 + 1 / 2 ^ 33) * Real.exp s +
      (3 / 2 ^ 24 + 1 / 2 ^ 33) * (|s| * Real.exp s) + 1 / 2 ^ 139 := by
  have hE := Real.exp_pos s
  have hEt := Real.exp_pos t
  have hsa : |s| = -s := abs_of_nonpos hs
  rw [hsa] at hts ⊢
  have hX : 0 ≤ (-s) * Real.exp s := mul_nonneg (by linarith) hE.le
  rcases le_or_gt (-s) 1024 with h | h
  · -- `|t − s| ≤ 2^-13`
    have hd1 : |t - s| ≤ 1 / 2 ^ 13 := by
      have : -s * (1 / 2 ^ 24) ≤ 1024 * (1 / 2 ^ 24) := by nlinarith
      norm_num at this hts ⊢; linarith
    have hed : Real.exp |t - s| ≤ 1 + 1 / 2 ^ 12 := by
      have h1 := Real.abs_exp_sub_one_le (x := |t - s|) (by rw [abs_abs]; linarith)
      rw [abs_abs] at h1
      have := (abs_le.mp h1).2
      linarith
    have hts' : Real.exp t = Real.exp s * Real.exp (t - s) := by rw [← Real.exp_add]; ring_nf
    -- the rounding's effect on `e^s`
    have h1 : |Real.exp t - Real.exp s| ≤ Real.exp s * ((-s * (1 / 2 ^ 24) + 1 / 2 ^ 150) * (1 + 1 / 2 ^ 12)) := by
      rw [hts', show Real.exp s * Real.exp (t - s) - Real.exp s = Real.exp s * (Real.exp (t - s) - 1) by ring,
        abs_mul, abs_of_pos hE]
      apply mul_le_mul_of_nonneg_left _ hE.le
      exact (abs_exp_sub_one_le' (t - s)).trans
        (mul_le_mul hts hed (Real.exp_pos _).le (by nlinarith))
    have h2 : Real.exp t ≤ Real.exp s * (1 + 1 / 2 ^ 12) := by
      rw [hts']
      exact mul_le_mul_of_nonneg_left ((Real.exp_le_exp.mpr (le_abs_self _)).trans hed) hE.le
    have h3 : |t| ≤ -s * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150 := by
      have := abs_add_le s (t - s)
      rw [add_sub_cancel, hsa] at this
      linarith
    have h4 : (1 / 2 ^ 22 + 1 / 2 ^ 23 * |t|) * Real.exp t ≤
        (1 / 2 ^ 22 + 1 / 2 ^ 23 * (-s * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150)) * (Real.exp s * (1 + 1 / 2 ^ 12)) :=
      mul_le_mul (by linarith) h2 hEt.le (by nlinarith)
    calc |e - Real.exp s| ≤ |e - Real.exp t| + |Real.exp t - Real.exp s| := abs_sub_le _ _ _
      _ ≤ (1 / 2 ^ 22 + 1 / 2 ^ 23 * (-s * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150)) * (Real.exp s * (1 + 1 / 2 ^ 12)) +
          1 / 2 ^ 140 + Real.exp s * ((-s * (1 / 2 ^ 24) + 1 / 2 ^ 150) * (1 + 1 / 2 ^ 12)) := by linarith
      _ ≤ (1 / 2 ^ 22 + 1 / 2 ^ 33) * Real.exp s + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (-s * Real.exp s) +
          1 / 2 ^ 139 := by nlinarith
  · -- `t ≤ −1023`: every term is below `2^-139`
    have ht : t ≤ -1023 := by
      have := (abs_le.mp hts).2
      nlinarith
    have hY : Real.exp (t / 2) ≤ 1 / 2 ^ 150 := by
      calc Real.exp (t / 2) ≤ Real.exp (-((150 : ℕ) : ℝ)) := Real.exp_le_exp.mpr (by push_cast; linarith)
        _ ≤ 1 / 2 ^ 150 := exp_neg_nat_le 150
    have hY0 := Real.exp_pos (t / 2)
    have hte : |t| * Real.exp t ≤ Real.exp (t / 2) := by
      have h1 := abs_mul_exp_le (t := t / 2) (by linarith)
      have h2 : Real.exp t = Real.exp (t / 2) * Real.exp (t / 2) := by rw [← Real.exp_add]; ring_nf
      rw [abs_div, abs_two] at h1
      rw [h2]
      nlinarith
    have het : Real.exp t ≤ Real.exp (t / 2) := Real.exp_le_exp.mpr (by linarith)
    have hes : Real.exp s ≤ 1 / 2 ^ 150 := by
      calc Real.exp s ≤ Real.exp (-((150 : ℕ) : ℝ)) := Real.exp_le_exp.mpr (by push_cast; linarith)
        _ ≤ 1 / 2 ^ 150 := exp_neg_nat_le 150
    have hall : |e - Real.exp s| ≤ 1 / 2 ^ 139 := by
      calc |e - Real.exp s| ≤ |e - Real.exp t| + |Real.exp t - Real.exp s| := abs_sub_le _ _ _
        _ ≤ |e - Real.exp t| + (Real.exp t + Real.exp s) := by
            have := abs_sub (Real.exp t) (Real.exp s)
            rw [abs_of_pos hEt, abs_of_pos hE] at this; linarith
        _ ≤ (1 / 2 ^ 22 * Real.exp t + 1 / 2 ^ 23 * (|t| * Real.exp t) + 1 / 2 ^ 140) +
            (Real.exp t + Real.exp s) := by
            have : (1 / 2 ^ 22 + 1 / 2 ^ 23 * |t|) * Real.exp t =
                1 / 2 ^ 22 * Real.exp t + 1 / 2 ^ 23 * (|t| * Real.exp t) := by ring
            linarith
        _ ≤ 1 / 2 ^ 22 * (1 / 2 ^ 150) + 1 / 2 ^ 23 * (1 / 2 ^ 150) + 1 / 2 ^ 140 +
            (1 / 2 ^ 150 + 1 / 2 ^ 150) := by
            have k1 : Real.exp t ≤ 1 / 2 ^ 150 := het.trans hY
            have k2 : |t| * Real.exp t ≤ 1 / 2 ^ 150 := hte.trans hY
            gcongr
        _ ≤ 1 / 2 ^ 139 := by norm_num
    have : 0 ≤ (1 / 2 ^ 22 + 1 / 2 ^ 33) * Real.exp s + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (-s * Real.exp s) := by
      positivity
    linarith

/-- **The row sum, and the common factor.** With every lane within `105·2^-25·a + 2^-39` of its
exact value `a ≥ 0`, exact values summing to `A ≥ 1`, and the sum `S` within the bound of
`sum_err_covers` (`G` is `(1 + u)^(B−1)`): `S` is within `(B + 69)·2^-24·A` of `A`, so `S > 0`
and the row's common factor `A / S` is within `(B + 86)·2^-24` of `1`. -/
theorem sum_common {B : Nat} (hB1 : 1 ≤ B) (hB : B ≤ 2 ^ 14) (a e : ℕ → ℝ)
    (ha0 : ∀ j ∈ Finset.range B, 0 ≤ a j) (hA : 1 ≤ ∑ j ∈ Finset.range B, a j)
    (hea : ∀ j ∈ Finset.range B, |e j - a j| ≤ 105 / 2 ^ 25 * a j + 1 / 2 ^ 39)
    {S G : ℝ} (hG1 : 1 ≤ G) (hG : G * (1 - ((B - 1 : ℕ) : ℝ) * (1 / 2 ^ 24)) ≤ 1)
    (hS : |S - ∑ j ∈ Finset.range B, e j| ≤
        (G - 1) * (∑ j ∈ Finset.range B, |e j|) + 1 / 2 ^ 150 * ((B - 1 : ℕ) : ℝ) * G) :
    0 < S ∧ |(∑ j ∈ Finset.range B, a j) / S - 1| ≤ ((B : ℝ) + 86) / 2 ^ 24 := by
  set A := ∑ j ∈ Finset.range B, a j with hAdef
  set Es := ∑ j ∈ Finset.range B, e j with hEs
  set M := ∑ j ∈ Finset.range B, |e j| with hM
  have hBk : ((B - 1 : ℕ) : ℝ) = (B : ℝ) - 1 := by rw [Nat.cast_sub hB1]; simp
  have hBr : (B : ℝ) ≤ 2 ^ 14 := by exact_mod_cast hB
  have hB1r : (1 : ℝ) ≤ B := by exact_mod_cast hB1
  rw [hBk] at hG hS
  -- the lanes' errors
  have hEA : |Es - A| ≤ 105 / 2 ^ 25 * A + B * (1 / 2 ^ 39) := by
    rw [hEs, hAdef, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    refine (Finset.sum_le_sum hea).trans (le_of_eq ?_)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hBl : (B : ℝ) * (1 / 2 ^ 39) ≤ 1 / 2 ^ 25 * A := by linarith
  have hEA' : |Es - A| ≤ 53 / 2 ^ 24 * A := by linarith
  have hMb : M ≤ (1 + 53 / 2 ^ 24) * A := by
    have : ∀ j ∈ Finset.range B, |e j| ≤ a j + |e j - a j| := by
      intro j hj
      have := abs_add_le (a j) (e j - a j)
      rwa [add_sub_cancel, abs_of_nonneg (ha0 j hj)] at this
    have h1 : M ≤ A + ∑ j ∈ Finset.range B, |e j - a j| := by
      refine (Finset.sum_le_sum this).trans (le_of_eq ?_)
      rw [Finset.sum_add_distrib]
    have h2 : ∑ j ∈ Finset.range B, |e j - a j| ≤ 53 / 2 ^ 24 * A := by
      refine (Finset.sum_le_sum hea).trans ?_
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      linarith
    linarith
  have hM0 : 0 ≤ M := Finset.sum_nonneg fun j _ => abs_nonneg _
  clear_value A Es M
  -- `G − 1 ≤ k·u·G` and `G ≤ 1024/1023`
  have hk0 : (0 : ℝ) ≤ ((B : ℝ) - 1) * (1 / 2 ^ 24) := by
    have : (0 : ℝ) ≤ (B : ℝ) - 1 := by linarith
    positivity
  have hku : ((B : ℝ) - 1) * (1 / 2 ^ 24) ≤ 1 / 1024 := by linarith
  have hG2 : G ≤ 1024 / 1023 := by
    have := mul_le_mul_of_nonneg_left hku (by linarith : (0 : ℝ) ≤ G)
    linarith
  have hG3 : G - 1 ≤ ((B : ℝ) - 1) * (1 / 2 ^ 24) * (1024 / 1023) := by
    have := mul_le_mul_of_nonneg_left hG2 hk0
    linarith
  have e1 : (G - 1) * M ≤ ((B : ℝ) - 1) * (1 / 2 ^ 24) * (1024 / 1023) * ((1 + 53 / 2 ^ 24) * A) :=
    mul_le_mul hG3 hMb hM0 (by positivity)
  have e2 : 1 / 2 ^ 150 * ((B : ℝ) - 1) * G ≤ (1 / 2 ^ 130) * A := by
    have h1 : ((B : ℝ) - 1) * G ≤ 2 ^ 14 * (1024 / 1023) :=
      mul_le_mul (by linarith) hG2 (by linarith) (by norm_num)
    have h2 : 1 / 2 ^ 150 * ((B : ℝ) - 1) * G = 1 / 2 ^ 150 * (((B : ℝ) - 1) * G) := by ring
    rw [h2]
    have h3 := mul_le_mul_of_nonneg_left h1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)
    have h4 : 1 / 2 ^ 150 * (2 ^ 14 * (1024 / 1023) : ℝ) ≤ 1 / 2 ^ 130 := by norm_num
    linarith
  have hfin : |S - A| ≤ (G - 1) * M + 1 / 2 ^ 150 * ((B : ℝ) - 1) * G + 53 / 2 ^ 24 * A := by
    calc |S - A| = |(S - Es) + (Es - A)| := by ring_nf
      _ ≤ |S - Es| + |Es - A| := abs_add_le _ _
      _ ≤ _ := add_le_add hS hEA'
  have hA0 : 0 ≤ A := by linarith
  have key : ((B : ℝ) - 1) * (1 / 2 ^ 24) * (1024 / 1023) * (1 + 53 / 2 ^ 24) + 1 / 2 ^ 130 +
      53 / 2 ^ 24 ≤ ((B : ℝ) + 69) / 2 ^ 24 := by
    linarith
  have hSA : |S - A| ≤ ((B : ℝ) + 69) / 2 ^ 24 * A := by
    have := mul_le_mul_of_nonneg_right key hA0
    linarith
  set δ : ℝ := ((B : ℝ) + 69) / 2 ^ 24 with hδ
  have hδ1 : δ ≤ 1 / 1000 := by rw [hδ]; linarith
  have hS1 : (1 - δ) * A ≤ S := by have := (abs_le.mp hSA).1; linarith
  have hS0 : 0 < S := by
    have := mul_le_mul_of_nonneg_right (by linarith : (1 - 1 / 1000 : ℝ) ≤ 1 - δ) hA0
    linarith
  refine ⟨hS0, ?_⟩
  have h1 : A / S - 1 = (A - S) / S := by rw [sub_div, div_self hS0.ne']
  rw [h1, abs_div, abs_of_pos hS0, div_le_iff₀ hS0, abs_sub_comm]
  -- `δ·A ≤ ((B + 86)/2^24)·(1 − δ)·A ≤ ((B + 86)/2^24)·S`
  have h2 : δ ≤ ((B : ℝ) + 86) / 2 ^ 24 * (1 - δ) := by
    rw [hδ]
    have : ((B : ℝ) + 86) * ((B : ℝ) + 69) ≤ (2 ^ 14 + 86) * (2 ^ 14 + 69) :=
      mul_le_mul (by linarith) (by linarith) (by linarith) (by norm_num)
    linarith
  have h3 := mul_le_mul_of_nonneg_left hS1 (by positivity : (0 : ℝ) ≤ ((B : ℝ) + 86) / 2 ^ 24)
  have h4 := mul_le_mul_of_nonneg_right h2 hA0
  linarith

/-- **One output.** From one lane's error (`w = |s|`), the common factor `K = A / S ≤ 1 + 10^-3`
and `DivFull`'s bound for the division: the output is within
`(2^-21 + 2^-30 + (3·2^-24 + 2^-30)·w)·σ + 2^-133/A` of `K·σ`, `σ = a / A`. -/
theorem final_common {a e S y A w : ℝ} (ha0 : 0 ≤ a) (hA : 1 ≤ A) (hAB : A ≤ 2 ^ 14) (hw : 0 ≤ w)
    (hS0 : 0 < S) (hK : A / S ≤ 1 + 1 / 1000)
    (hea : |e - a| ≤ (1 / 2 ^ 22 + 1 / 2 ^ 33) * a + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (w * a) + 1 / 2 ^ 139)
    (hy : |y - e / S| ≤ 1 / 2 ^ 22 * |e / S| + 1 / 2 ^ 148) :
    |y - A / S * (a / A)| ≤ ((1 / 2 ^ 21 + 1 / 2 ^ 30) + (3 / 2 ^ 24 + 1 / 2 ^ 30) * w) * (a / A) +
      1 / 2 ^ 133 * (1 / A) := by
  have hA0 : 0 < A := by linarith
  set K := A / S with hKdef
  set σ := a / A with hσ
  set q := 1 / A with hq
  have hK0 : 0 < K := div_pos hA0 hS0
  have hσ0 : 0 ≤ σ := div_nonneg ha0 hA0.le
  have hq0 : 0 < q := by rw [hq]; exact div_pos one_pos hA0
  have hqA : 1 / 2 ^ 14 ≤ q := by rw [hq]; exact one_div_le_one_div_of_le hA0 hAB
  have haS : a / S = K * σ := by rw [hKdef, hσ]; field_simp
  have h1S : 1 / S = K * q := by rw [hKdef, hq]; field_simp
  have hwσ : 0 ≤ w * σ := mul_nonneg hw hσ0
  rw [show A / S * (a / A) = a / S by field_simp]
  -- `|e − a| / S`
  have hE : |e - a| / S ≤ K * ((1 / 2 ^ 22 + 1 / 2 ^ 33) * σ + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (w * σ) +
      1 / 2 ^ 139 * q) := by
    calc |e - a| / S ≤ ((1 / 2 ^ 22 + 1 / 2 ^ 33) * a + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (w * a) + 1 / 2 ^ 139) / S :=
          div_le_div_of_nonneg_right hea hS0.le
      _ = (1 / 2 ^ 22 + 1 / 2 ^ 33) * (a / S) + (3 / 2 ^ 24 + 1 / 2 ^ 33) * w * (a / S) +
          1 / 2 ^ 139 * (1 / S) := by ring
      _ = _ := by rw [haS, h1S]; ring
  have heS : |e / S| ≤ a / S + |e - a| / S := by
    rw [abs_div, abs_of_pos hS0, ← add_div]
    apply div_le_div_of_nonneg_right _ hS0.le
    have := abs_add_le a (e - a)
    rwa [add_sub_cancel, abs_of_nonneg ha0] at this
  have hyS : |y - a / S| ≤ |y - e / S| + |e - a| / S := by
    have : y - a / S = (y - e / S) + (e - a) / S := by field_simp; ring
    rw [this]
    refine (abs_add_le _ _).trans (le_of_eq ?_)
    rw [abs_div, abs_of_pos hS0]
  set T := (1 / 2 ^ 22 + 1 / 2 ^ 33) * σ + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (w * σ) + 1 / 2 ^ 139 * q with hT
  have hT0 : 0 ≤ T := by positivity
  have hEK : |e - a| / S ≤ K * T := hE
  -- `|y − Kσ| ≤ K·(2^-22·σ + (1 + 2^-22)·T) + 2^-148`
  have h2 : |y - a / S| ≤ K * (1 / 2 ^ 22 * σ + (1 + 1 / 2 ^ 22) * T) + 1 / 2 ^ 148 := by
    have h3 : |e - a| / S ≥ 0 := div_nonneg (abs_nonneg _) hS0.le
    calc |y - a / S| ≤ 1 / 2 ^ 22 * |e / S| + 1 / 2 ^ 148 + |e - a| / S := by linarith
      _ ≤ 1 / 2 ^ 22 * (a / S + |e - a| / S) + 1 / 2 ^ 148 + |e - a| / S := by
          have := mul_le_mul_of_nonneg_left heS (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22); linarith
      _ = 1 / 2 ^ 22 * (a / S) + (1 + 1 / 2 ^ 22) * (|e - a| / S) + 1 / 2 ^ 148 := by ring
      _ ≤ 1 / 2 ^ 22 * (K * σ) + (1 + 1 / 2 ^ 22) * (K * T) + 1 / 2 ^ 148 := by
          rw [haS]; have := mul_le_mul_of_nonneg_left hEK (by norm_num : (0 : ℝ) ≤ 1 + 1 / 2 ^ 22); linarith
      _ = K * (1 / 2 ^ 22 * σ + (1 + 1 / 2 ^ 22) * T) + 1 / 2 ^ 148 := by ring
  have hU0 : 0 ≤ 1 / 2 ^ 22 * σ + (1 + 1 / 2 ^ 22) * T := by positivity
  have h4 := mul_le_mul_of_nonneg_right hK hU0
  rw [hT] at h4
  nlinarith

/-! ## One row -/

theorem list_sum_range {β : Type} [AddCommMonoid β] (f : ℕ → β) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

theorem negInf_isNaN32 : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN rfl

theorem negInf_isFinite32 : (FloatV.negInf : F32).isFinite = false := by
  rw [← Rules.Num.negInfF32_eq_negInf]; decide

theorem isNaN_of_inputOk {a : F32} (h : InputOk a) : a.isNaN = false := by
  rcases h with ⟨hf, -⟩ | rfl
  · exact (FloatV.isFinite_iff.mp hf).1
  · exact negInf_isNaN32

/-- **The row maximum.** `tl.max` of the loaded row (inputs below `n`, `-∞` beyond) is a finite
input of magnitude at most `2^126`, at least every finite input. -/
theorem max_facts {B n : Nat} (hnB : n ≤ B) (xs : Nat → F32) (hxs : ∀ k, k < n → InputOk (xs k))
    (hfin : ∃ k, k < n ∧ (xs k).isFinite = true) (row : Tensor [B] F32) (mx : F32)
    (hrow_in : ∀ i : Fin B, i.1 < n → row.get (i, ()) = xs i)
    (hrow_out : ∀ i : Fin B, n ≤ i.1 → row.get (i, ()) = FloatV.negInf)
    (hmx : ∃ t : Tree, t.Covers B ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < B then row.get1 ⟨j, h⟩ else default) mx) :
    ∃ i0, i0 < n ∧ mx = xs i0 ∧ (xs i0).isFinite = true ∧ |(xs i0).toRat| ≤ 2 ^ 126 ∧
      ∀ j, j < n → (xs j).isFinite = true → (xs j).toRat ≤ mx.toRat := by
  obtain ⟨t, hcov, hev⟩ := hmx
  set lf : Nat → F32 := fun j => if h : j < B then row.get1 ⟨j, h⟩ else default with hlf
  have hmem : ∀ j, j ∈ t.leaves ↔ j < B := fun j => by
    rw [hcov.mem_iff, List.mem_range]
  have hlf_in : ∀ j, j < n → lf j = xs j := fun j hj => by
    rw [hlf]; simp only [dif_pos (lt_of_lt_of_le hj hnB)]; exact hrow_in ⟨j, _⟩ hj
  have hlf_out : ∀ j, n ≤ j → j < B → lf j = FloatV.negInf := fun j hj hjB => by
    rw [hlf]; simp only [dif_pos hjB]; exact hrow_out ⟨j, hjB⟩ hj
  have hnan : ∀ j ∈ t.leaves, (lf j).isNaN = false := by
    intro j hj
    have hjB := (hmem j).mp hj
    by_cases hjn : j < n
    · rw [hlf_in j hjn]; exact isNaN_of_inputOk (hxs j hjn)
    · rw [hlf_out j (by omega) hjB]; exact negInf_isNaN32
  obtain ⟨⟨i0, hi0, hmxeq⟩, hle⟩ := Tree.eval_maxRel hev hnan
  have hle' : ∀ j, j < B → FloatV.le (lf j) mx = true := fun j hj => by
    have := hle j ((hmem j).mpr hj); simpa using this
  have hmxnan : mx.isNaN = false := by rw [hmxeq]; exact hnan i0 hi0
  have hval : ∀ j, j < n → (xs j).isFinite = true → (xs j).toRat ≤ mx.toRat := by
    intro j hj hf
    have := hle' j (lt_of_lt_of_le hj hnB)
    rw [hlf_in j hj] at this
    exact (FloatV.le_iff (FloatV.isFinite_iff.mp hf).1 hmxnan).mp this
  obtain ⟨k0, hk0, hk0f⟩ := hfin
  have hi0B := (hmem i0).mp hi0
  -- `mx` is not `-∞`: it is at least the finite `xs k0`
  have hnot : mx ≠ FloatV.negInf := by
    intro h
    have h1 := hval k0 hk0 hk0f
    obtain ⟨h2, h3⟩ := FloatV.negInf_toRat_le (f := Format.binary32) Format.binary32_regular rfl
      (FloatV.isFinite_iff.mp hk0f).1
    rw [h] at h1
    have := h3 (le_antisymm h2 h1)
    rw [this] at hk0f
    rw [negInf_isFinite32] at hk0f
    cases hk0f
  have hi0n : i0 < n := by
    by_contra hc
    exact hnot (by rw [hmxeq]; exact hlf_out i0 (by omega) hi0B)
  have hxi0 : mx = xs i0 := by rw [hmxeq]; exact hlf_in i0 hi0n
  rcases hxs i0 hi0n with ⟨hf, hb⟩ | hninf
  · exact ⟨i0, hi0n, hxi0, hf, hb, hval⟩
  · exact absurd (hxi0.trans hninf) hnot

theorem zeroF32_isFinite : (Rules.Num.zeroF32).isFinite = true := by decide

theorem zeroF32_toRat : (Rules.Num.zeroF32).toRat = 0 :=
  FloatV.toRat_of_isZero (by decide)

theorem toReal_abs_le_of_toRat {x : F32} {c : ℚ} (h : |x.toRat| ≤ c) : |x.toReal| ≤ (c : ℝ) := by
  unfold FloatV.toReal; exact_mod_cast h

/-- **One lane.** A live lane (`j < n`, `xs j` finite) gives a finite `tl.exp` result within
`(2^-22 + 2^-33)·a + (3·2^-24 + 2^-33)·(w·a) + 2^-139` of its exact value `a = e^(xs j − mx)`,
with `w = mx − xs j = |s|` (`exp_lane`); a dead lane (`-∞`, loaded or masked) gives `+0`, its exact
value. -/
theorem lane_facts {P : Platform} (hexp : Rules.Num.ExpF32 P) (hninf : Rules.Num.ExpF32NegInf P)
    {B n : Nat} (xs : Nat → F32) (hxs : ∀ k, k < n → InputOk (xs k))
    (row rmm num : Tensor [B] F32) (mx : F32) (hmxf : mx.isFinite = true)
    (hmxb : |mx.toRat| ≤ 2 ^ 126)
    (hle : ∀ j, j < n → (xs j).isFinite = true → (xs j).toRat ≤ mx.toRat)
    (hrow_in : ∀ i : Fin B, i.1 < n → row.get (i, ()) = xs i)
    (hrow_out : ∀ i : Fin B, n ≤ i.1 → row.get (i, ()) = FloatV.negInf)
    (hrmm : ∀ i, FloatV.NaNEq (FloatV.sub (row.get i) mx) (rmm.get i))
    (hnum : ∀ i, P.math .exp .binary32 (rmm.get i) (num.get i)) (j : Nat) (hjB : j < B) :
    (num.get (⟨j, hjB⟩, ())).isFinite = true ∧
      |(num.get (⟨j, hjB⟩, ())).toReal -
          (if j < n ∧ (xs j).isFinite = true then Real.exp ((xs j).toReal - mx.toReal) else 0)| ≤
        (1 / 2 ^ 22 + 1 / 2 ^ 33) *
            (if j < n ∧ (xs j).isFinite = true then Real.exp ((xs j).toReal - mx.toReal) else 0) +
          (3 / 2 ^ 24 + 1 / 2 ^ 33) *
            ((if j < n ∧ (xs j).isFinite = true then mx.toReal - (xs j).toReal else 0) *
              (if j < n ∧ (xs j).isFinite = true then Real.exp ((xs j).toReal - mx.toReal) else 0)) +
          1 / 2 ^ 139 := by
  by_cases hlv : j < n ∧ (xs j).isFinite = true
  · rw [if_pos hlv, if_pos hlv]
    obtain ⟨hjn, hjf⟩ := hlv
    have hjb : |(xs j).toRat| ≤ 2 ^ 126 := by
      rcases hxs j hjn with ⟨-, hb⟩ | h
      · exact hb
      · rw [h, negInf_isFinite32] at hjf; cases hjf
    have hrow : row.get (⟨j, hjB⟩, ()) = xs j := hrow_in ⟨j, hjB⟩ hjn
    have hmax : |(xs j).toRat - mx.toRat| ≤ Format.binary32.maxVal := by
      rw [Format.binary32_maxVal]
      have := abs_sub (xs j).toRat mx.toRat
      norm_num at this ⊢
      linarith
    obtain ⟨hsf, hserr⟩ := FloatV.sub_err Format.binary32_regular hjf hmxf hmax
    have hrmm_eq : rmm.get (⟨j, hjB⟩, ()) = FloatV.sub (xs j) mx := by
      have := hrmm (⟨j, hjB⟩, ()); rw [hrow] at this
      exact this.eq (FloatV.isFinite_iff.mp hsf).1
    set t := FloatV.sub (xs j) mx with ht
    have hs : (xs j).toReal - mx.toReal ≤ 0 := by
      have := hle j hjn hjf
      unfold FloatV.toReal
      have h' : ((xs j).toRat : ℝ) ≤ (mx.toRat : ℝ) := by exact_mod_cast this
      linarith
    have hts : |t.toReal - ((xs j).toReal - mx.toReal)| ≤
        |(xs j).toReal - mx.toReal| * (1 / 2 ^ 24) + 1 / 2 ^ 150 := by
      rw [Format.binary32_u, Format.binary32_eta] at hserr
      have h' : ((|t.toRat - ((xs j).toRat - mx.toRat)| : ℚ) : ℝ) ≤
          ((|(xs j).toRat - mx.toRat| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) : ℚ) : ℝ) := by
        exact_mod_cast hserr
      unfold FloatV.toReal
      push_cast at h'
      norm_num at h' ⊢
      linarith
    have htη : t.toReal ≤ 1 / 2 ^ 150 := by
      have := (abs_le.mp hts).2; rw [abs_of_nonpos hs] at this; nlinarith
    obtain ⟨ref, href, hacc⟩ := hexp |t.toReal| (abs_nonneg _)
    have hrefe : ref = Real.exp := (Option.some.inj href).symm
    subst hrefe
    have hnumj := hnum (⟨j, hjB⟩, ())
    rw [hrmm_eq] at hnumj
    have htb : |t.toReal| ≤ 2 ^ 128 := by
      have h1 := FloatV.abs_toRat_le_maxVal Format.binary32_regular hsf
      rw [Format.binary32_maxVal] at h1
      have h2 := toReal_abs_le_of_toRat h1
      push_cast at h2
      linarith
    have hE2 := exp_le_two htη
    have hEp := Real.exp_pos t.toReal
    have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t.toReal|)) * |Real.exp t.toReal| +
        2 ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [abs_of_pos hEp, Format.binary32_maxVal]
      push_cast
      have h3 : (2 : ℝ) ^ (-23 : ℤ) * |t.toReal| ≤ 2 ^ 105 := by
        have : (2 : ℝ) ^ (-23 : ℤ) * 2 ^ 128 = 2 ^ 105 := by norm_num
        nlinarith [abs_nonneg t.toReal]
      have h4 : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
      have h5 : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 := by norm_num
      nlinarith
    obtain ⟨hef, heerr⟩ := hacc t (num.get (⟨j, hjB⟩, ())) hnumj hsf
      ⟨neg_abs_le _, le_abs_self _⟩ (Set.mem_univ _) hroom
    have hl := exp_lane hs hts (e := (num.get (⟨j, hjB⟩, ())).toReal) (by
      rw [abs_of_pos hEp] at heerr
      norm_num at heerr ⊢
      linarith)
    rw [abs_of_nonpos hs, neg_sub] at hl
    exact ⟨hef, hl⟩
  · rw [if_neg hlv, if_neg hlv]
    have hrow : row.get (⟨j, hjB⟩, ()) = FloatV.negInf := by
      by_cases hjn : j < n
      · have hnf : (xs j).isFinite = false := by
          cases h : (xs j).isFinite
          · rfl
          · exact absurd ⟨hjn, h⟩ hlv
        rw [hrow_in ⟨j, hjB⟩ hjn]
        rcases hxs j hjn with ⟨hf, -⟩ | h
        · rw [hf] at hnf; cases hnf
        · exact h
      · exact hrow_out ⟨j, hjB⟩ (show n ≤ j by omega)
    have hrmm_eq : rmm.get (⟨j, hjB⟩, ()) = FloatV.negInf := by
      have := hrmm (⟨j, hjB⟩, ())
      rw [hrow, FloatV.sub_negInf_left rfl hmxf] at this
      exact this.eq negInf_isNaN32
    have hnumj := hnum (⟨j, hjB⟩, ())
    rw [hrmm_eq, ← Rules.Num.negInfF32_eq_negInf] at hnumj
    have hz := hninf _ hnumj
    rw [hz]
    refine ⟨zeroF32_isFinite, ?_⟩
    unfold FloatV.toReal
    rw [zeroF32_toRat]
    norm_num

/-- `(1 + u)^(B−1)` for `B ≤ 2^14`: `G·(1 − (B−1)·u) ≤ 1`, so `G ≤ 1024/1023`. -/
theorem pow_u_facts {B : Nat} (hB : B ≤ 2 ^ 14) :
    (1 + (1 / 2 ^ 24 : ℚ)) ^ (B - 1) * (1 - ((B - 1 : ℕ) : ℚ) * (1 / 2 ^ 24)) ≤ 1 ∧
      (1 + (1 / 2 ^ 24 : ℚ)) ^ (B - 1) ≤ 1024 / 1023 ∧ 1 ≤ (1 + (1 / 2 ^ 24 : ℚ)) ^ (B - 1) := by
  have hk : ((B - 1 : ℕ) : ℚ) ≤ 2 ^ 14 := by
    have : B - 1 ≤ 2 ^ 14 := by omega
    exact_mod_cast this
  have h1 := DotLoop.pow_mul_le (x := (1 / 2 ^ 24 : ℚ)) (by norm_num) (B - 1) (by nlinarith)
  have h3 : 1 ≤ (1 + (1 / 2 ^ 24 : ℚ)) ^ (B - 1) := one_le_pow₀ (by norm_num)
  refine ⟨h1, ?_, h3⟩
  nlinarith

/-- **One row.** The loaded row, its maximum, `row − max`, `tl.exp`, the sum and the division,
each as the model's operations allow: there is one factor `sc = A / S` for the row, within
`sumTol B` of `1`, and every output below `n` is finite and within
`(elemTol + gapTol · softmaxGap) · σ + floorTol · softmaxMax` of `sc · σ`, for the exact softmax
`σ`. -/
theorem row_bound {P : Platform} (hexp : Rules.Num.ExpF32 P) (hninf : Rules.Num.ExpF32NegInf P)
    (hdiv : Rules.Num.DivFull P)
    {B n : Nat} (hB : B ≤ 2 ^ 14) (hnB : n ≤ B)
    (xs : Nat → F32) (hxs : ∀ k, k < n → InputOk (xs k)) (hfin : ∃ k, k < n ∧ (xs k).isFinite = true)
    (row rmm num out : Tensor [B] F32) (mx den : F32)
    (hrow_in : ∀ i : Fin B, i.1 < n → row.get (i, ()) = xs i)
    (hrow_out : ∀ i : Fin B, n ≤ i.1 → row.get (i, ()) = FloatV.negInf)
    (hmx : ∃ t : Tree, t.Covers B ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < B then row.get1 ⟨j, h⟩ else default) mx)
    (hrmm : ∀ i, FloatV.NaNEq (FloatV.sub (row.get i) mx) (rmm.get i))
    (hnum : ∀ i, P.math .exp .binary32 (rmm.get i) (num.get i))
    (hden : ∃ t : Tree, t.Covers B ∧
      t.Eval (AddRel false) (fun j => if h : j < B then num.get1 ⟨j, h⟩ else default) den)
    (hout : ∀ i, P.div .binary32 (num.get i) den (out.get i)) :
    ∃ sc : ℝ, |sc - 1| ≤ (sumTol B : ℝ) ∧ ∀ i : Fin B, i.1 < n → (out.get (i, ())).isFinite = true ∧
      |(out.get (i, ())).toReal - sc * softmaxRef xs n i| ≤
        ((elemTol : ℝ) + (gapTol : ℝ) * softmaxGap xs n i) * softmaxRef xs n i +
          (floorTol : ℝ) * softmaxMax xs n := by
  obtain ⟨i0, hi0n, hmxeq, hxi0f, hxi0b, hle⟩ := max_facts hnB xs hxs hfin row mx hrow_in hrow_out hmx
  have hmxf : mx.isFinite = true := hmxeq ▸ hxi0f
  have hmxb : |mx.toRat| ≤ 2 ^ 126 := hmxeq ▸ hxi0b
  have hB1 : 1 ≤ B := by omega
  have hi0B : i0 < B := by omega
  -- the exact values `a j`, the gaps `w j = mx − xs j`, and the computed values `e j`
  set a : Nat → ℝ := fun j =>
    if j < n ∧ (xs j).isFinite = true then Real.exp ((xs j).toReal - mx.toReal) else 0 with ha
  set w : Nat → ℝ := fun j =>
    if j < n ∧ (xs j).isFinite = true then mx.toReal - (xs j).toReal else 0 with hw
  set ev : Nat → F32 := fun j => if h : j < B then num.get1 ⟨j, h⟩ else default with hev
  have hev_eq : ∀ j (hj : j < B), ev j = num.get (⟨j, hj⟩, ()) := fun j hj => by
    rw [hev]; simp only [hj, dite_true]; rfl
  have hlane : ∀ j, j < B → (ev j).isFinite = true ∧ |(ev j).toReal - a j| ≤
      (1 / 2 ^ 22 + 1 / 2 ^ 33) * a j + (3 / 2 ^ 24 + 1 / 2 ^ 33) * (w j * a j) + 1 / 2 ^ 139 := by
    intro j hj
    rw [hev_eq j hj]
    exact lane_facts hexp hninf xs hxs row rmm num mx hmxf hmxb hle hrow_in hrow_out hrmm hnum j hj
  have hsle : ∀ j, j < n → (xs j).isFinite = true → (xs j).toReal - mx.toReal ≤ 0 := by
    intro j hj hf
    have := hle j hj hf
    unfold FloatV.toReal
    have h' : ((xs j).toRat : ℝ) ≤ (mx.toRat : ℝ) := by exact_mod_cast this
    linarith
  have ha0 : ∀ j, 0 ≤ a j := fun j => by
    rw [ha]; dsimp only; split_ifs
    · exact (Real.exp_pos _).le
    · exact le_rfl
  have ha1 : ∀ j, a j ≤ 1 := fun j => by
    rw [ha]; dsimp only; split_ifs with h
    · exact Real.exp_le_one_iff.mpr (hsle j h.1 h.2)
    · norm_num
  have hw0 : ∀ j, 0 ≤ w j := fun j => by
    rw [hw]; dsimp only; split_ifs with h
    · linarith [hsle j h.1 h.2]
    · exact le_rfl
  -- `w·a = |s|·e^s` for a live lane
  have hwa : ∀ j, w j * a j ≤ 1 / 2 ∧ w j * a j ≤ 16 * a j + 1 / 2 ^ 17 := fun j => by
    rw [hw, ha]; dsimp only; split_ifs with h
    · have hs := hsle j h.1 h.2
      have e1 : (mx.toReal - (xs j).toReal) = |(xs j).toReal - mx.toReal| := by
        rw [abs_of_nonpos hs]; ring
      rw [e1]
      exact ⟨abs_mul_exp_le hs, abs_mul_exp_le16 hs⟩
    · norm_num
  have hlane3 : ∀ j, j < B → |(ev j).toReal - a j| ≤ 105 / 2 ^ 25 * a j + 1 / 2 ^ 39 := by
    intro j hj
    have h1 := (hlane j hj).2
    have h2 := (hwa j).2
    have h3 := ha0 j
    have h4 := mul_le_mul_of_nonneg_left h2 (by norm_num : (0 : ℝ) ≤ 3 / 2 ^ 24 + 1 / 2 ^ 33)
    linarith
  have hai0 : a i0 = 1 := by
    rw [ha]; dsimp only; rw [if_pos ⟨hi0n, hxi0f⟩, ← hmxeq, sub_self, Real.exp_zero]
  set A := ∑ j ∈ Finset.range B, a j with hA
  have hA1 : 1 ≤ A := by
    rw [← hai0]
    exact Finset.single_le_sum (fun j _ => ha0 j) (Finset.mem_range.mpr hi0B)
  have hAB : A ≤ 2 ^ 14 := by
    have h1 : A ≤ ∑ _j ∈ Finset.range B, (1 : ℝ) := Finset.sum_le_sum fun j _ => ha1 j
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h1
    have : (B : ℝ) ≤ 2 ^ 14 := by exact_mod_cast hB
    linarith
  -- each computed lane is at most `2`
  have hev2 : ∀ j, j < B → |(ev j).toRat| ≤ 2 := by
    intro j hj
    obtain ⟨-, h⟩ := hlane j hj
    have h1 := ha1 j
    have h0 := ha0 j
    have h5 := (hwa j).1
    have h2 : |(ev j).toReal| ≤ 2 := by
      have := abs_add_le (a j) ((ev j).toReal - a j)
      rw [add_sub_cancel, abs_of_nonneg h0] at this
      have h6 := mul_le_mul_of_nonneg_left h5 (by norm_num : (0 : ℝ) ≤ 3 / 2 ^ 24 + 1 / 2 ^ 33)
      have h7 := mul_le_mul_of_nonneg_left h1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22 + 1 / 2 ^ 33)
      linarith
    unfold FloatV.toReal at h2
    exact_mod_cast h2
  -- the sum
  obtain ⟨t2, hcov2, hev2'⟩ := hden
  obtain ⟨hG, hG2, hG1⟩ := pow_u_facts hB
  have hsumq : ((List.range B).map fun i => |(ev i).toRat|).sum ≤ 2 * B := by
    rw [list_sum_range]
    calc ∑ i ∈ Finset.range B, |(ev i).toRat| ≤ ∑ _i ∈ Finset.range B, (2 : ℚ) :=
          Finset.sum_le_sum fun i hi => hev2 i (Finset.mem_range.mp hi)
      _ = 2 * B := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_comm]
  have hM : ((List.range B).map fun i => |(ev i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (B - 1) (B - 1)
        ((List.range B).map fun i => |(ev i).toRat|).sum ≤ Format.binary32.maxVal := by
    unfold Tree.errBound
    rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
    have hs0 : 0 ≤ ((List.range B).map fun i => |(ev i).toRat|).sum :=
      List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨i, -, rfl⟩; exact abs_nonneg _)
    have hBq : (B : ℚ) ≤ 2 ^ 14 := by exact_mod_cast hB
    have hk : ((B - 1 : ℕ) : ℚ) ≤ 2 ^ 14 := by
      have : B - 1 ≤ 2 ^ 14 := by omega
      exact_mod_cast this
    norm_num at hG2 ⊢
    have e1 : ((1 + 1 / 16777216 : ℚ) ^ (B - 1) - 1) * ((List.range B).map fun i => |(ev i).toRat|).sum ≤
        1 * (2 * 2 ^ 14) := by
      apply mul_le_mul (by linarith) (by linarith) hs0 (by norm_num)
    nlinarith
  obtain ⟨hdenf, hdenerr⟩ := sum_err_covers Format.binary32_regular ev hcov2 hev2'
    (fun i hi => (hlane i hi).1) hM
  -- in the reals
  have hS : |den.toReal - ∑ j ∈ Finset.range B, (ev j).toReal| ≤
      ((1 + 1 / 2 ^ 24 : ℝ) ^ (B - 1) - 1) * (∑ j ∈ Finset.range B, |(ev j).toReal|) +
        1 / 2 ^ 150 * ((B - 1 : ℕ) : ℝ) * (1 + 1 / 2 ^ 24 : ℝ) ^ (B - 1) := by
    rw [list_sum_range, list_sum_range] at hdenerr
    unfold Tree.errBound at hdenerr
    rw [Format.binary32_u, Format.binary32_eta] at hdenerr
    have h' := (Rat.cast_le (K := ℝ)).mpr hdenerr
    push_cast at h'
    unfold FloatV.toReal
    norm_num at h' ⊢
    simpa [abs_sub_comm] using h'
  obtain ⟨hden0, hsc⟩ := sum_common hB1 hB a (fun j => (ev j).toReal) (fun j _ => ha0 j) hA1
    (fun j hj => hlane3 j (Finset.mem_range.mp hj))
    (G := (1 + 1 / 2 ^ 24 : ℝ) ^ (B - 1))
    (by have := (Rat.cast_le (K := ℝ)).mpr hG1; push_cast at this; exact this)
    (by have := (Rat.cast_le (K := ℝ)).mpr hG; push_cast at this; exact this) hS
  -- the softmax reference is `a i / A`
  have hexpIn : ∀ k, k < n → expIn (xs k) = Real.exp mx.toReal * a k := by
    intro k hk
    rw [ha]; dsimp only
    by_cases hkf : (xs k).isFinite = true
    · rw [if_pos ⟨hk, hkf⟩, ← Real.exp_add, add_sub_cancel]
      unfold expIn
      rw [if_neg]
      intro h; rw [h, negInf_isFinite32] at hkf; cases hkf
    · rw [if_neg (fun h => hkf h.2), mul_zero]
      unfold expIn
      rcases hxs k hk with ⟨hf, -⟩ | h
      · exact absurd hf hkf
      · rw [if_pos h]
  have hsumA : ∑ k ∈ Finset.range n, expIn (xs k) = Real.exp mx.toReal * A := by
    rw [hA, Finset.mul_sum]
    rw [Finset.sum_congr rfl fun k hk => hexpIn k (Finset.mem_range.mp hk)]
    refine Finset.sum_subset ((fun k hk => Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hk) hnB))) ?_
    intro k hkB hkn
    have : ¬ (k < n) := fun h => hkn (Finset.mem_range.mpr h)
    rw [ha]; dsimp only; rw [if_neg (fun h => this h.1), mul_zero]
  have href : ∀ k, k < n → softmaxRef xs n k = a k / A := by
    intro k hk
    unfold softmaxRef
    rw [hsumA, hexpIn k hk, mul_div_mul_left _ _ (Real.exp_pos _).ne']
  have hA0 : 0 < A := by linarith
  -- the row's largest output is at least `1 / A` (the maximum's own lane)
  have hσmax : 1 / A ≤ softmaxMax xs n := by
    have h1 := le_ciSup (f := fun k : Fin n => softmaxRef xs n k) (Set.finite_range _).bddAbove
      ⟨i0, hi0n⟩
    simp only at h1
    rw [href i0 hi0n, hai0] at h1
    exact h1
  -- the common factor
  have hK : |A / den.toReal - 1| ≤ ((B : ℝ) + 86) / 2 ^ 24 := hsc
  have hK1 : A / den.toReal ≤ 1 + 1 / 1000 := by
    have h1 := (abs_le.mp hK).2
    have : ((B : ℝ) + 86) / 2 ^ 24 ≤ 1 / 1000 := by
      have : (B : ℝ) ≤ 2 ^ 14 := by exact_mod_cast hB
      linarith
    linarith
  refine ⟨A / den.toReal, ?_, fun i hi => ?_⟩
  · unfold sumTol
    push_cast
    rw [show ((2 : ℝ) ^ (-24 : ℤ)) = 1 / 2 ^ 24 by norm_num]
    linarith
  -- each output
  have hden_pos : 1 / 2 ≤ den.toReal := by
    have h1 : A ≤ (1 + 1 / 1000) * den.toReal := (div_le_iff₀ hden0).mp hK1
    linarith
  have hev_i := hlane i i.2
  rw [hev_eq i i.2] at hev_i
  obtain ⟨hnf, hnerr⟩ := hev_i
  have hdenq : 1 / 2 ≤ den.toRat := by
    have h : (((1 / 2 : ℚ) : ℝ)) ≤ (den.toRat : ℝ) := by push_cast; exact hden_pos
    exact_mod_cast h
  have hq : |(num.get (i, ())).toRat / den.toRat| ≤ 2 ^ 126 := by
    have h2 := hev2 i i.2
    rw [hev_eq i i.2] at h2
    rw [abs_div, abs_of_pos (by linarith : (0 : ℚ) < den.toRat), div_le_iff₀ (by linarith)]
    nlinarith
  obtain ⟨hof, hoerr⟩ := hdiv (num.get (i, ())) den (out.get (i, ())) (hout (i, ())) hnf hdenf
    (by linarith) hq
  refine ⟨hof, ?_⟩
  have hy : |(out.get (i, ())).toReal - (num.get (i, ())).toReal / den.toReal| ≤
      1 / 2 ^ 22 * |(num.get (i, ())).toReal / den.toReal| + 1 / 2 ^ 148 := by
    have h' := (Rat.cast_le (K := ℝ)).mpr hoerr
    push_cast at h'
    unfold FloatV.toReal
    norm_num at h' ⊢
    exact h'
  have hfc := final_common (ha0 i) hA1 hAB (hw0 i) hden0 hK1 hnerr hy
  -- the gap: `w i ≤ softmaxGap` on a live lane; nothing to show on a dead one (`a i = 0`)
  have hgap : w i * (a i / A) ≤ softmaxGap xs n i * (a i / A) := by
    by_cases hlv : i.1 < n ∧ (xs i).isFinite = true
    · have hai : a i = Real.exp ((xs i).toReal - mx.toReal) := by rw [ha]; dsimp only; rw [if_pos hlv]
      have hwi : w i = mx.toReal - (xs i).toReal := by rw [hw]; dsimp only; rw [if_pos hlv]
      have hapos : 0 < a i := by rw [hai]; exact Real.exp_pos _
      have hσpos : 0 < a i / A := div_pos hapos hA0
      apply mul_le_mul_of_nonneg_right _ hσpos.le
      unfold softmaxGap
      rw [href i hi]
      have h1 : (1 / A) / (a i / A) = Real.exp (mx.toReal - (xs i).toReal) := by
        rw [hai]; field_simp; rw [← Real.exp_add]; ring_nf; simp
      have h2 : Real.log ((1 / A) / (a i / A)) ≤ Real.log (softmaxMax xs n / (a i / A)) :=
        Real.log_le_log (div_pos (by positivity) hσpos) (div_le_div_of_nonneg_right hσmax hσpos.le)
      rw [h1, Real.log_exp] at h2
      rw [hwi]; exact h2
    · have hai : a i = 0 := by rw [ha]; dsimp only; rw [if_neg hlv]
      rw [hai, zero_div, mul_zero, mul_zero]
  have hfloor : 1 / 2 ^ 133 * (1 / A) ≤ (floorTol : ℝ) * softmaxMax xs n := by
    have : (floorTol : ℝ) = 1 / 2 ^ 133 := by unfold floorTol; push_cast; norm_num
    rw [this]
    exact mul_le_mul_of_nonneg_left hσmax (by norm_num)
  have hel : (elemTol : ℝ) = 1 / 2 ^ 21 + 1 / 2 ^ 30 := by unfold elemTol; push_cast; norm_num
  have hga : (gapTol : ℝ) = 3 / 2 ^ 24 + 1 / 2 ^ 30 := by unfold gapTol; push_cast; norm_num
  rw [href i hi, hel, hga]
  have hσ0 : 0 ≤ a i / A := div_nonneg (ha0 i) hA0.le
  have h5 := mul_le_mul_of_nonneg_left hgap (by norm_num : (0 : ℝ) ≤ 3 / 2 ^ 24 + 1 / 2 ^ 30)
  have h6 : ((1 / 2 ^ 21 + 1 / 2 ^ 30) + (3 / 2 ^ 24 + 1 / 2 ^ 30) * w i) * (a i / A) =
      (1 / 2 ^ 21 + 1 / 2 ^ 30) * (a i / A) + (3 / 2 ^ 24 + 1 / 2 ^ 30) * (w i * (a i / A)) := by ring
  have h7 : ((1 / 2 ^ 21 + 1 / 2 ^ 30) + (3 / 2 ^ 24 + 1 / 2 ^ 30) * softmaxGap xs n i) * (a i / A) =
      (1 / 2 ^ 21 + 1 / 2 ^ 30) * (a i / A) + (3 / 2 ^ 24 + 1 / 2 ^ 30) * (softmaxGap xs n i * (a i / A)) := by
    ring
  linarith

end Kernels.Softmax
