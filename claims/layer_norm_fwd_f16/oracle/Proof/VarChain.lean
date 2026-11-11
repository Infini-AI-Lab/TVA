import Proof.Mean

namespace Claims.LayerNormFwdF16

namespace RowP

variable (r : RowP)

/-- The arithmetic of the variance's coefficient of `V0`. -/
theorem var_coef {u E D : ℝ} (hu0 : 0 < u) (hu : u ≤ 1 / 10 ^ 7)
    (hEu : E ≤ 100002 / 100000 * D * u) (hD0 : 0 ≤ D) (hD : D ≤ 15) :
    (3 * u + E * (1 + 3 * u)) + 4 * u * (1 + (3 * u + E * (1 + 3 * u))) ≤ (D + 8) * u := by
  have h1 : E * ((1 + 3 * u) * (1 + 4 * u)) ≤
      100002 / 100000 * D * u * ((1 + 3 * u) * (1 + 4 * u)) := by gcongr
  have h2 : (1 + 3 * u) * (1 + 4 * u) ≤ 100001 / 100000 := by nlinarith
  have h3 : 100002 / 100000 * D * u * ((1 + 3 * u) * (1 + 4 * u)) ≤
      100002 / 100000 * D * u * (100001 / 100000) := by gcongr
  have h4 : u * u ≤ u / 10 ^ 7 := by nlinarith
  nlinarith [mul_nonneg (sub_nonneg.2 hD) hu0.le]

/-- Sum of the squares over the whole block is the sum over the row. -/
theorem sum_block_sq (d : ℕ → ℝ) (hd0 : ∀ j, r.N ≤ j → j < r.B → d j = 0) :
    ∑ j ∈ Finset.range r.B, d j ^ 2 = ∑ j ∈ Finset.range r.N, d j ^ 2 := by
  rw [← Finset.sum_range_add_sum_Ico _ r.n_le_block]
  rw [Finset.sum_eq_zero (s := Finset.Ico r.N r.B) (fun j hj => by
    simp [hd0 j (Finset.mem_Ico.1 hj).1 (Finset.mem_Ico.1 hj).2]), add_zero]

/-- **From the squares to the stored variance**: each square rounds (`v`), the `2^D` squares are
summed along a tree of depth `D` (`T`), and `T / N` is divided (`var`). -/
theorem var_chain (d v : ℕ → ℝ) (T var : ℝ)
    (hd0 : ∀ j, r.N ≤ j → j < r.B → d j = 0)
    (hv : ∀ j, j < r.B → |v j - d j ^ 2| ≤ 3 * u32 * d j ^ 2 + 4 * eta32)
    (hT : |T - ∑ j ∈ Finset.range (r.B), v j| ≤
      ((1 + u32) ^ r.D - 1) * ∑ j ∈ Finset.range (r.B), |v j| +
        eta32 * ((r.B - 1 : ℕ) : ℝ) * (1 + u32) ^ r.D)
    (hvar : |var - T / r.N| ≤ 2 ^ (-22 : ℤ) * |T / r.N| + 2 ^ (-148 : ℤ)) :
    |var - (∑ j ∈ Finset.range r.N, d j ^ 2) / r.N| ≤
      ((r.D : ℝ) + 8) * u32 * ((∑ j ∈ Finset.range r.N, d j ^ 2) / r.N) + 16 * eta32 := by
  have hN : (0 : ℝ) < r.N := by have := r.n_ge; positivity
  have hB1 : (r.N : ℝ) ≤ r.B := by exact_mod_cast r.n_le_block
  have hB2 : (r.B : ℝ) < 2 * r.N := by exact_mod_cast r.block_lt
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hu0 : (0 : ℝ) < u32 := by unfold u32; positivity
  have hu1 : u32 = 2 ^ (-24 : ℤ) := rfl
  have hη0 : (0 : ℝ) < eta32 := by unfold eta32; positivity
  have hQ := r.sum_block_sq d hd0
  set Q := ∑ j ∈ Finset.range r.N, d j ^ 2 with hQdef
  have hQ0 : 0 ≤ Q := Finset.sum_nonneg fun j _ => sq_nonneg _
  set n : ℝ := (r.N : ℝ) with hn
  set B : ℝ := (r.B : ℝ) with hB
  set E := (1 + u32) ^ r.D - 1 with hE
  have hE0 : 0 ≤ E := by
    have := one_le_pow₀ (n := r.D) (a := 1 + u32) (by unfold u32; norm_num)
    linarith
  have hEu : E ≤ 100002 / 100000 * r.D * u32 := pow_sub_one_le r.D_le
  have hE1 : E ≤ 1 / 10 ^ 5 := by
    have : 100002 / 100000 * (r.D : ℝ) * u32 ≤ 100002 / 100000 * 15 * u32 := by gcongr
    have h7 : 100002 / 100000 * (15 : ℝ) * u32 ≤ 1 / 10 ^ 5 := by rw [hu1]; norm_num
    linarith
  have hP : (1 + u32) ^ r.D = 1 + E := by rw [hE]; ring
  have hcast : ((r.B - 1 : ℕ) : ℝ) ≤ B := by rw [hB]; exact_mod_cast Nat.sub_le r.B 1
  have hv1 : ∑ j ∈ Finset.range r.B, |v j| ≤ (1 + 3 * u32) * Q + 4 * B * eta32 := by
    calc ∑ j ∈ Finset.range r.B, |v j|
        ≤ ∑ j ∈ Finset.range r.B, ((1 + 3 * u32) * d j ^ 2 + 4 * eta32) := by
          refine Finset.sum_le_sum fun j hj => ?_
          have := hv j (Finset.mem_range.1 hj)
          have := abs_sub_abs_le_abs_sub (v j) (d j ^ 2)
          rw [abs_of_nonneg (sq_nonneg (d j))] at this
          linarith
      _ = (1 + 3 * u32) * Q + 4 * B * eta32 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, hQ]
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; ring
  have hv2 : |∑ j ∈ Finset.range r.B, v j - Q| ≤ 3 * u32 * Q + 4 * B * eta32 := by
    rw [← hQ, ← Finset.sum_sub_distrib]
    calc _ ≤ ∑ j ∈ Finset.range r.B, |v j - d j ^ 2| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ Finset.range r.B, (3 * u32 * d j ^ 2 + 4 * eta32) :=
          Finset.sum_le_sum fun j hj => hv j (Finset.mem_range.1 hj)
      _ = _ := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, hQ]
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; ring
  rw [hP] at hT
  have hT1 : |T - Q| ≤ Q * (3 * u32 + E * (1 + 3 * u32)) + 5 * B * eta32 * (1 + E) := by
    have h1 : E * ∑ j ∈ Finset.range r.B, |v j| ≤ E * ((1 + 3 * u32) * Q + 4 * B * eta32) :=
      mul_le_mul_of_nonneg_left hv1 hE0
    have h2 : eta32 * ((r.B - 1 : ℕ) : ℝ) * (1 + E) ≤ eta32 * B * (1 + E) := by gcongr
    have h3 : |T - Q| ≤ |T - ∑ j ∈ Finset.range r.B, v j| +
        |∑ j ∈ Finset.range r.B, v j - Q| := abs_sub_le _ _ _
    linarith
  have hBn : B ≤ 2 * n := hB2.le
  have hT2 : |T / n - Q / n| ≤ (Q / n) * (3 * u32 + E * (1 + 3 * u32)) + 10 * eta32 * (1 + E) := by
    have : T / n - Q / n = (T - Q) / n := by ring
    rw [this, abs_div, abs_of_pos hN, div_le_iff₀ hN]
    have h4 : 5 * B * eta32 * (1 + E) ≤ (10 * eta32 * (1 + E)) * n := by
      have : 0 ≤ eta32 * (1 + E) := by positivity
      nlinarith
    have : (Q / n) * (3 * u32 + E * (1 + 3 * u32)) * n = Q * (3 * u32 + E * (1 + 3 * u32)) := by
      field_simp
    linarith
  have hT3 : |T / n| ≤ (1 + (3 * u32 + E * (1 + 3 * u32))) * (Q / n) + 10 * eta32 * (1 + E) := by
    have := abs_sub_abs_le_abs_sub (T / n) (Q / n)
    rw [abs_of_nonneg (div_nonneg hQ0 hN.le)] at this
    linarith
  have hvar' : |var - T / n| ≤ 4 * u32 * |T / n| + 4 * eta32 := by
    have e1 : (2 : ℝ) ^ (-22 : ℤ) = 4 * u32 := by unfold u32; norm_num
    have e2 : (2 : ℝ) ^ (-148 : ℤ) = 4 * eta32 := by unfold eta32; norm_num
    rw [e1, e2] at hvar; exact hvar
  have hV0 : 0 ≤ Q / n := div_nonneg hQ0 hN.le
  have hfin : |var - Q / n| ≤ (Q / n) * ((3 * u32 + E * (1 + 3 * u32)) +
      4 * u32 * (1 + (3 * u32 + E * (1 + 3 * u32)))) +
      (10 * eta32 * (1 + E) * (1 + 4 * u32) + 4 * eta32) := by
    have := abs_sub_le var (T / n) (Q / n)
    have h5 : 4 * u32 * |T / n| ≤ 4 * u32 * ((1 + (3 * u32 + E * (1 + 3 * u32))) * (Q / n) +
        10 * eta32 * (1 + E)) := mul_le_mul_of_nonneg_left hT3 (by positivity)
    nlinarith
  have hc := var_coef hu0 (by rw [hu1]; norm_num) hEu (Nat.cast_nonneg _) hD
  have hc2 : 10 * eta32 * (1 + E) * (1 + 4 * u32) + 4 * eta32 ≤ 16 * eta32 := by
    have hu2 : u32 ≤ 1 / 100 := by rw [hu1]; norm_num
    nlinarith [mul_nonneg hη0.le hE0, mul_nonneg hη0.le hu0.le,
      mul_nonneg (mul_nonneg hη0.le hu0.le) (sub_nonneg.2 hE1),
      mul_nonneg hη0.le (sub_nonneg.2 hE1)]
  have hA2 := mul_le_mul_of_nonneg_left hc hV0
  linarith

/-- **`var + eps`**: the host's `eps` as a `float32` (`epsf`) and the fusable addition (`s`). -/
theorem s_chain (V0 var epsf s : ℝ) (hV0 : 0 ≤ V0)
    (hvar : |var - V0| ≤ ((r.D : ℝ) + 8) * u32 * V0 + 16 * eta32)
    (hε : |epsf - r.eps| ≤ u32 * r.eps + eta32)
    (hs : |s - (var + epsf)| ≤ 3 * u32 * (|var| + |epsf|) + 4 * eta32) :
    |s - (V0 + r.eps)| ≤ ((r.D : ℝ) + 12) * u32 * V0 + 41 / 10 * u32 * r.eps + 22 * eta32 := by
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hD0 : (0 : ℝ) ≤ r.D := Nat.cast_nonneg _
  have hu0 : (0 : ℝ) < u32 := by unfold u32; positivity
  have hu2 : u32 ≤ 1 / 10 ^ 7 := by unfold u32; norm_num
  have hη0 : (0 : ℝ) < eta32 := by unfold eta32; positivity
  have hη2 : eta32 ≤ 1 / 10 ^ 7 := by unfold eta32; norm_num
  have hε0 : 0 < r.eps := lt_of_lt_of_le (by positivity) r.eps_ge
  have hε' : |epsf| ≤ (1 + u32) * r.eps + eta32 := by
    have := abs_sub_abs_le_abs_sub epsf r.eps
    rw [abs_of_pos hε0] at this
    linarith
  have hvar' : |var| ≤ (1 + ((r.D : ℝ) + 8) * u32) * V0 + 16 * eta32 := by
    have := abs_sub_abs_le_abs_sub var V0
    rw [abs_of_nonneg hV0] at this
    linarith
  have h1 : 3 * u32 * (|var| + |epsf|) ≤ 3 * u32 * ((1 + ((r.D : ℝ) + 8) * u32) * V0 + 16 * eta32 +
      ((1 + u32) * r.eps + eta32)) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have htri : |s - (V0 + r.eps)| ≤ |s - (var + epsf)| + |var - V0| + |epsf - r.eps| := by
    have h2 := abs_sub_le s (var + epsf) (V0 + r.eps)
    have h3 : |var + epsf - (V0 + r.eps)| ≤ |var - V0| + |epsf - r.eps| := by
      have := abs_add_le (var - V0) (epsf - r.eps)
      rw [show var - V0 + (epsf - r.eps) = var + epsf - (V0 + r.eps) by ring] at this
      exact this
    linarith
  have hW : 0 ≤ u32 * V0 := by positivity
  have hX : 0 ≤ u32 * r.eps := by positivity
  nlinarith [mul_nonneg hW (by nlinarith : (0 : ℝ) ≤ 1 - 3 * ((r.D : ℝ) + 8) * u32),
    mul_nonneg hX (by nlinarith : (0 : ℝ) ≤ 1 / 1000 - 3 * u32),
    mul_nonneg hη0.le (by nlinarith : (0 : ℝ) ≤ 1 / 1000 - 3 * u32),
    abs_nonneg var, abs_nonneg epsf]

end RowP

end Claims.LayerNormFwdF16
