import Proof.RowVar2

/-!
# The row's cutoff

`cut_stage'`: the `bfloat16` roundings of the mean, the standard deviation, their product with the
multiplier and the sum (lines 79–82), step by step, against `7/8·cutTol − 2^-131`. To first order
the error is `2^-7·|mu| + 3·2^-8·|m|·sd`; the `float32` terms are `O(u)·(1 + |m|)·aMean`, and the
absolute terms are absorbed by `aMean ≥ 2^-70`.
-/

namespace Claims.GeluAndMulSparse

open TVA

namespace RowP

variable (r : RowP)

/-- `stdErr ≤ 27·u·aMean + 40·u·sd`. -/
theorem stdErr_le : r.stdErr ≤ 27 * u32 * r.aMean + 40 * u32 * r.sd := by
  unfold stdErr
  have := r.L_le'; have := r.aMean_nonneg'
  have : ((r.L : ℝ) + 12) * u32 * r.aMean ≤ 27 * u32 * r.aMean := by
    gcongr
    · unfold u32; positivity
    · linarith
  linarith

/-- **The cutoff**, the arithmetic: within `7/8·cutTol − 2^-131` of `mu + m·sd`. -/
theorem cut_stage' (hok : r.Ok) {mean0 std32 m32 mb sb P scb Cs ck : ℝ}
    (hm0 : |mean0 - r.mu| ≤ r.meanErr) (hsd : |std32 - r.sd| ≤ r.stdErr)
    (hm32 : |m32 - r.m| ≤ u32 * |r.m| + eta32)
    (hmb : |mb - mean0| ≤ ub * |mean0| + etab)
    (hsb : |sb - std32| ≤ ub * |std32| + etab)
    (hP : |P - sb * m32| ≤ u32 * |sb * m32| + eta32)
    (hscb : |scb - P| ≤ ub * |P| + etab)
    (hCs : |Cs - (mb + scb)| ≤ u32 * |mb + scb| + eta32)
    (hck : |ck - Cs| ≤ ub * |Cs| + etab) :
    |ck - r.cRef| ≤ 7 / 8 * r.cutTol - 2 ^ (-131 : ℤ) := by
  have hA := r.aMean_nonneg'
  have hA32 := r.aMean_le'
  have hAge := hok.aMean_ge
  have hS := r.sd_nonneg'
  have hS33 := r.sd_le'
  have hmu := r.mu_le'
  have hμ := r.m_le
  have hme := r.meanErr_le
  have hse := r.stdErr_le
  have hU : u32 = 1 / 16777216 := u32_eq
  have hH : eta32 = 1 / 2 ^ 150 := eta32_eq
  have hUB : ub = 1 / 256 := by unfold ub; norm_num
  have hZ : etab = 1 / 2 ^ 134 := by unfold etab; norm_num
  have hZ' : (2 : ℝ) ^ (-134 : ℤ) = 1 / 2 ^ 134 := by norm_num
  rw [hU, hZ'] at hme
  rw [hU] at hse
  rw [hU, hH] at hm32 hP hCs
  rw [hUB, hZ] at hmb hsb hscb hck
  obtain ⟨M, hM⟩ : ∃ M, M = |r.mu| := ⟨_, rfl⟩
  obtain ⟨μ, hμd⟩ : ∃ μ, μ = |r.m| := ⟨_, rfl⟩
  obtain ⟨S, hSd⟩ : ∃ S, S = r.sd := ⟨_, rfl⟩
  obtain ⟨A, hAd⟩ : ∃ A, A = r.aMean := ⟨_, rfl⟩
  rw [← hM] at hmu
  rw [← hμd] at hμ hm32
  rw [← hSd] at hsd hS hS33 hse
  rw [← hAd] at hA hA32 hAge hmu hme hse
  have hM0 : 0 ≤ M := hM ▸ abs_nonneg _
  have hμ0 : 0 ≤ μ := hμd ▸ abs_nonneg _
  obtain ⟨μS, hμS⟩ : ∃ μS, μS = μ * S := ⟨_, rfl⟩
  obtain ⟨μA, hμA⟩ : ∃ μA, μA = μ * A := ⟨_, rfl⟩
  have hμS0 : 0 ≤ μS := hμS ▸ mul_nonneg hμ0 hS
  have hμA0 : 0 ≤ μA := hμA ▸ mul_nonneg hμ0 hA
  have hAge' : (1 / 2 ^ 70 : ℝ) ≤ A := by
    have : (2 : ℝ) ^ (-70 : ℤ) = 1 / 2 ^ 70 := by norm_num
    rw [this] at hAge; exact hAge
  -- step 1–2: the mean in `bfloat16`
  have e1 : |mean0 - r.mu| ≤ 20 * (1 / 16777216) * A + 1 / 2 ^ 134 := hm0.trans hme
  have s1 : |mean0| ≤ M + (20 * (1 / 16777216) * A + 1 / 2 ^ 134) := by
    have := abs_sub_abs_le_abs_sub mean0 r.mu; rw [← hM] at this; linarith
  have s2 : |mb - r.mu| ≤ 1 / 256 * (M + (20 * (1 / 16777216) * A + 1 / 2 ^ 134)) + 1 / 2 ^ 134 +
      (20 * (1 / 16777216) * A + 1 / 2 ^ 134) := by
    have := abs_sub_le mb mean0 r.mu
    have := mul_le_mul_of_nonneg_left s1 (show (0 : ℝ) ≤ 1 / 256 by norm_num)
    linarith
  -- step 3–4: the standard deviation in `bfloat16`
  have e2 : |std32 - S| ≤ 27 * (1 / 16777216) * A + 40 * (1 / 16777216) * S := hsd.trans hse
  have s3 : |std32| ≤ S + (27 * (1 / 16777216) * A + 40 * (1 / 16777216) * S) := by
    have := abs_sub_abs_le_abs_sub std32 S; rw [abs_of_nonneg hS] at this; linarith
  obtain ⟨e3, he3⟩ : ∃ e3 : ℝ, e3 = 1 / 256 * (S + (27 * (1 / 16777216) * A + 40 * (1 / 16777216) * S)) +
      1 / 2 ^ 134 + (27 * (1 / 16777216) * A + 40 * (1 / 16777216) * S) := ⟨_, rfl⟩
  have s4 : |sb - S| ≤ e3 := by
    have := abs_sub_le sb std32 S
    have := mul_le_mul_of_nonneg_left s3 (show (0 : ℝ) ≤ 1 / 256 by norm_num)
    rw [he3]; linarith
  have he30 : 0 ≤ e3 := le_trans (abs_nonneg _) s4
  have he3b : e3 ≤ 2 ^ 35 := by rw [he3]; linarith
  -- step 5–6: the product with the multiplier
  have s5 : |m32| ≤ μ + 1 / 16777216 * μ + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub m32 r.m; rw [← hμd] at this; linarith
  have s6a : |sb - S| * |m32| ≤ e3 * (μ + 1 / 16777216 * μ + 1 / 2 ^ 150) :=
    mul_le_mul s4 s5 (abs_nonneg _) he30
  have s6b : e3 * (μ + 1 / 16777216 * μ + 1 / 2 ^ 150) =
      (1 + 1 / 16777216) * (1 / 256 * μS + (1 / 256 + 1) * (27 * (1 / 16777216)) * μA +
        (1 / 256 + 1) * (40 * (1 / 16777216)) * μS + 1 / 2 ^ 134 * μ) + e3 * (1 / 2 ^ 150) := by
    rw [he3, hμS, hμA]; ring
  have s6c : e3 * (1 / 2 ^ 150) ≤ 2 ^ 35 * (1 / 2 ^ 150) :=
    mul_le_mul_of_nonneg_right he3b (by norm_num)
  have s6d : S * |m32 - r.m| ≤ S * (1 / 16777216 * μ + 1 / 2 ^ 150) :=
    mul_le_mul_of_nonneg_left hm32 hS
  have s6e : S * (1 / 16777216 * μ + 1 / 2 ^ 150) = 1 / 16777216 * μS + S * (1 / 2 ^ 150) := by
    rw [hμS]; ring
  have s6f : S * (1 / 2 ^ 150 : ℝ) ≤ 2 ^ 33 * (1 / 2 ^ 150) :=
    mul_le_mul_of_nonneg_right hS33 (by norm_num)
  have s6 : |sb * m32 - S * r.m| ≤ |sb - S| * |m32| + S * |m32 - r.m| := by
    have e : sb * m32 - S * r.m = (sb - S) * m32 + S * (m32 - r.m) := by ring
    rw [e]
    calc _ ≤ |(sb - S) * m32| + |S * (m32 - r.m)| := abs_add_le _ _
      _ = _ := by rw [abs_mul, abs_mul, abs_of_nonneg hS]
  have hSm : |S * r.m| = μS := by rw [abs_mul, abs_of_nonneg hS, hμS, hμd, mul_comm]
  obtain ⟨X1, hX1⟩ : ∃ X1, X1 = |sb * m32 - S * r.m| := ⟨_, rfl⟩
  have s6g : X1 ≤ (1 + 1 / 16777216) * (1 / 256 * μS + (1 / 256 + 1) * (27 * (1 / 16777216)) * μA +
      (1 / 256 + 1) * (40 * (1 / 16777216)) * μS + 1 / 2 ^ 134 * μ) + 2 ^ 35 * (1 / 2 ^ 150) +
      (1 / 16777216 * μS + 2 ^ 33 * (1 / 2 ^ 150)) := by
    rw [hX1]; linarith
  -- step 7–10: `P` and its `bfloat16` rounding
  have s7 : |sb * m32| ≤ μS + X1 := by
    have := abs_sub_abs_le_abs_sub (sb * m32) (S * r.m); rw [hSm, ← hX1] at this; linarith
  have s8 : |P - S * r.m| ≤ 1 / 16777216 * (μS + X1) + 1 / 2 ^ 150 + X1 := by
    have := abs_sub_le P (sb * m32) (S * r.m)
    have := mul_le_mul_of_nonneg_left s7 (show (0 : ℝ) ≤ 1 / 16777216 by norm_num)
    rw [← hX1] at *; linarith
  have s9 : |P| ≤ μS + (1 / 16777216 * (μS + X1) + 1 / 2 ^ 150 + X1) := by
    have := abs_sub_abs_le_abs_sub P (S * r.m); rw [hSm] at this; linarith
  have s10 : |scb - S * r.m| ≤ 1 / 256 * (μS + (1 / 16777216 * (μS + X1) + 1 / 2 ^ 150 + X1)) +
      1 / 2 ^ 134 + (1 / 16777216 * (μS + X1) + 1 / 2 ^ 150 + X1) := by
    have := abs_sub_le scb P (S * r.m)
    have := mul_le_mul_of_nonneg_left s9 (show (0 : ℝ) ≤ 1 / 256 by norm_num)
    linarith
  -- step 11–13: the sum and the cutoff
  obtain ⟨Emb, hEmb⟩ : ∃ Emb : ℝ, Emb = 1 / 256 * (M + (20 * (1 / 16777216) * A + 1 / 2 ^ 134)) +
      1 / 2 ^ 134 + (20 * (1 / 16777216) * A + 1 / 2 ^ 134) := ⟨_, rfl⟩
  obtain ⟨Esc, hEsc⟩ : ∃ Esc : ℝ, Esc = 1 / 256 * (μS + (1 / 16777216 * (μS + X1) + 1 / 2 ^ 150 + X1)) +
      1 / 2 ^ 134 + (1 / 16777216 * (μS + X1) + 1 / 2 ^ 150 + X1) := ⟨_, rfl⟩
  rw [← hEmb] at s2
  rw [← hEsc] at s10
  have hcR : r.cRef = r.mu + S * r.m := by rw [cRef, hSd]; ring
  have s11 : |mb + scb - r.cRef| ≤ Emb + Esc := by
    rw [hcR]
    have e : mb + scb - (r.mu + S * r.m) = (mb - r.mu) + (scb - S * r.m) := by ring
    rw [e]; exact (abs_add_le _ _).trans (by linarith)
  have hcabs : |r.cRef| ≤ M + μS := by
    rw [hcR]; exact (abs_add_le _ _).trans (by rw [hSm, ← hM])
  have s11b : |mb + scb| ≤ M + μS + (Emb + Esc) := by
    have := abs_sub_abs_le_abs_sub (mb + scb) r.cRef; linarith
  have s12 : |Cs - r.cRef| ≤ 1 / 16777216 * (M + μS + (Emb + Esc)) + 1 / 2 ^ 150 + (Emb + Esc) := by
    have := abs_sub_le Cs (mb + scb) r.cRef
    have := mul_le_mul_of_nonneg_left s11b (show (0 : ℝ) ≤ 1 / 16777216 by norm_num)
    linarith
  have s12b : |Cs| ≤ M + μS + (1 / 16777216 * (M + μS + (Emb + Esc)) + 1 / 2 ^ 150 + (Emb + Esc)) := by
    have := abs_sub_abs_le_abs_sub Cs r.cRef; linarith
  have s13 : |ck - r.cRef| ≤ 1 / 256 * (M + μS + (1 / 16777216 * (M + μS + (Emb + Esc)) +
      1 / 2 ^ 150 + (Emb + Esc))) + 1 / 2 ^ 134 +
      (1 / 16777216 * (M + μS + (Emb + Esc)) + 1 / 2 ^ 150 + (Emb + Esc)) := by
    have := abs_sub_le ck Cs r.cRef
    have := mul_le_mul_of_nonneg_left s12b (show (0 : ℝ) ≤ 1 / 256 by norm_num)
    linarith
  -- the target
  have hT : 7 / 8 * r.cutTol - 2 ^ (-131 : ℤ) =
      7 / 8 * ((1 / 64) * (M + μS) + (1 / 2 ^ 17) * (A + μA) + 1 / 2 ^ 130) - 1 / 2 ^ 131 := by
    unfold cutTol
    rw [← hM, ← hμd, ← hSd, ← hAd, hμS, hμA]
    norm_num; ring
  rw [hT]
  have hμc : 1 / 2 ^ 134 * μ ≤ 1 / 2 ^ 134 * 64 := mul_le_mul_of_nonneg_left hμ (by norm_num)
  rw [hEmb, hEsc] at s13
  norm_num at s13 s6g hμc ⊢
  linarith

end RowP

end Claims.GeluAndMulSparse
