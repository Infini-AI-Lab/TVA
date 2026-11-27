import Proof.Defs

/-!
# A row's exact statistics: basic facts

Bounds on `mu`, `aMean`, `sd` and `cutTol`, the block sums of `gl`, the growth of `(1 + δ)^L`, and
what `RowP.Ok` gives: `2^-62 ≤ sd ≤ 2^8·aMean`, `2^-70 ≤ aMean ≤ 2^18·sd`.
-/

namespace Claims.GeluAndMulSparse

open TVA

/-- `(1 + x)^n − 1 ≤ n·x·(1 + x)^n` for `x ≥ 0`. -/
theorem one_add_pow_sub_one_le {x : ℝ} (hx : 0 ≤ x) (n : ℕ) :
    (1 + x) ^ n - 1 ≤ n * x * (1 + x) ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h1 : 1 ≤ (1 + x) ^ n := one_le_pow₀ (by linarith)
    have e : (1 + x) ^ (n + 1) - 1 = (1 + x) * ((1 + x) ^ n - 1) + x := by ring
    rw [e]
    push_cast
    have h2 : (1 + x) * ((1 + x) ^ n - 1) ≤ (1 + x) * (n * x * (1 + x) ^ n) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have h3 : x ≤ x * (1 + x) ^ (n + 1) := by
      have : 1 ≤ (1 + x) ^ (n + 1) := one_le_pow₀ (by linarith)
      nlinarith
    have e2 : (1 + x) * (n * x * (1 + x) ^ n) = n * x * (1 + x) ^ (n + 1) := by ring
    nlinarith

/-- `(1 + u)^L ≤ 1 + 2^-9` for `L ≤ 15`. -/
theorem pow_u_le {L : ℕ} (hL : L ≤ 15) : (1 + u32) ^ L ≤ 1 + 2 ^ (-9 : ℤ) := by
  calc (1 + u32) ^ L ≤ (1 + u32) ^ 15 := pow_le_pow_right₀ (by unfold u32; norm_num) hL
    _ ≤ _ := by unfold u32; norm_num

/-- `(1 + 3u)^L ≤ 1 + 2^-8` for `L ≤ 15`. -/
theorem pow_3u_le {L : ℕ} (hL : L ≤ 15) : (1 + 3 * u32) ^ L ≤ 1 + 2 ^ (-8 : ℤ) := by
  calc (1 + 3 * u32) ^ L ≤ (1 + 3 * u32) ^ 15 := pow_le_pow_right₀ (by unfold u32; norm_num) hL
    _ ≤ _ := by unfold u32; norm_num

/-- `(1 + u)^L − 1 ≤ L·u·(1 + 2^-9)`. -/
theorem pow_u_sub_le {L : ℕ} (hL : L ≤ 15) :
    (1 + u32) ^ L - 1 ≤ L * u32 * (1 + 2 ^ (-9 : ℤ)) := by
  have h := one_add_pow_sub_one_le (x := u32) (by unfold u32; positivity) L
  have h2 := pow_u_le hL
  have : (0 : ℝ) ≤ L * u32 := by unfold u32; positivity
  nlinarith

/-- `(1 + 3u)^L − 1 ≤ 3·L·u·(1 + 2^-8)`. -/
theorem pow_3u_sub_le {L : ℕ} (hL : L ≤ 15) :
    (1 + 3 * u32) ^ L - 1 ≤ 3 * L * u32 * (1 + 2 ^ (-8 : ℤ)) := by
  have h := one_add_pow_sub_one_le (x := 3 * u32) (by unfold u32; positivity) L
  have h2 := pow_3u_le hL
  have : (0 : ℝ) ≤ L * u32 := by unfold u32; positivity
  nlinarith

namespace RowP

variable (r : RowP)

/-- `d` as a positive real. -/
theorem d_pos' : (0 : ℝ) < r.d := by have := r.d_pos; exact_mod_cast (by omega : 0 < r.d)

/-- `d ≤ 2^L` as reals. -/
theorem d_le' : (r.d : ℝ) ≤ 2 ^ r.L := by exact_mod_cast r.d_le

/-- `2^L < 2d` as reals. -/
theorem b_lt' : (2 : ℝ) ^ r.L < 2 * r.d := by exact_mod_cast r.b_lt

/-- `d ≤ 2^15`. -/
theorem d_le15 : (r.d : ℝ) ≤ 2 ^ 15 := by
  have h : 2 ^ r.L ≤ 2 ^ 15 := Nat.pow_le_pow_right (by norm_num) r.L_le
  have : r.d ≤ 2 ^ 15 := le_trans r.d_le h
  exact_mod_cast this

/-- `L ≤ 15` as a real. -/
theorem L_le' : (r.L : ℝ) ≤ 15 := by exact_mod_cast r.L_le

/-- `(2^L − 1)/d ≤ 2`, as `(2^L − 1 : ℕ) ≤ 2d`. -/
theorem bm1_le : ((2 ^ r.L - 1 : ℕ) : ℝ) ≤ 2 * r.d := by
  have : 2 ^ r.L - 1 ≤ 2 * r.d := by have := r.b_lt; omega
  exact_mod_cast this

/-- `aMean ≥ 0`. -/
theorem aMean_nonneg' : 0 ≤ r.aMean :=
  div_nonneg (Finset.sum_nonneg fun _ _ => abs_nonneg _) r.d_pos'.le

/-- `Σ_{j<d} |g j| = d·aMean`. -/
theorem sum_abs_eq : ∑ j ∈ Finset.range r.d, |r.g j| = r.d * r.aMean := by
  unfold aMean; field_simp [r.d_pos'.ne']

/-- `Σ_{j<d} g j = d·mu`. -/
theorem sum_eq : ∑ j ∈ Finset.range r.d, r.g j = r.d * r.mu := by
  unfold mu; field_simp [r.d_pos'.ne']

/-- `aMean ≤ 2^32`. -/
theorem aMean_le' : r.aMean ≤ 2 ^ 32 := by
  unfold aMean
  rw [div_le_iff₀ r.d_pos']
  calc ∑ j ∈ Finset.range r.d, |r.g j| ≤ ∑ j ∈ Finset.range r.d, (2 : ℝ) ^ 32 :=
        Finset.sum_le_sum fun j hj => r.g_le j (Finset.mem_range.mp hj)
    _ = 2 ^ 32 * r.d := by simp; ring

/-- `|mu| ≤ aMean`. -/
theorem mu_le' : |r.mu| ≤ r.aMean := by
  unfold mu aMean
  rw [abs_div, abs_of_pos r.d_pos']
  exact div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _) r.d_pos'.le

/-- `sd ≥ 0`. -/
theorem sd_nonneg' : 0 ≤ r.sd := Real.sqrt_nonneg _

/-- The variance `sd² = Σ (g − mu)² / d`. -/
theorem sd_sq : r.sd ^ 2 = (∑ j ∈ Finset.range r.d, (r.g j - r.mu) ^ 2) / r.d :=
  Real.sq_sqrt (div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) r.d_pos'.le)

/-- `Σ (g − mu)² ≤ Σ g²`. -/
theorem sum_dev_le : ∑ j ∈ Finset.range r.d, (r.g j - r.mu) ^ 2 ≤
    ∑ j ∈ Finset.range r.d, r.g j ^ 2 := by
  have e : ∑ j ∈ Finset.range r.d, (r.g j - r.mu) ^ 2 =
      ∑ j ∈ Finset.range r.d, r.g j ^ 2 - r.d * r.mu ^ 2 := by
    have : ∀ j, (r.g j - r.mu) ^ 2 = r.g j ^ 2 - 2 * r.mu * r.g j + r.mu ^ 2 := fun j => by ring
    simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      Finset.sum_const, Finset.card_range, nsmul_eq_mul, r.sum_eq]
    ring
  rw [e]; nlinarith [r.d_pos', sq_nonneg r.mu]

/-- `Σ g² ≤ (Σ |g|)²`. -/
theorem sum_sq_le : ∑ j ∈ Finset.range r.d, r.g j ^ 2 ≤ (r.d * r.aMean) ^ 2 := by
  rw [← r.sum_abs_eq, sq]
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun j hj => ?_
  rw [sq, ← abs_mul_abs_self]
  exact mul_le_mul_of_nonneg_left
    (Finset.single_le_sum (f := fun j => |r.g j|) (fun _ _ => abs_nonneg _) hj) (abs_nonneg _)

/-- `sd ≤ 2^8·aMean`. -/
theorem sd_le_aMean : r.sd ≤ 2 ^ 8 * r.aMean := by
  have hd := r.d_pos'
  have h1 : r.sd ^ 2 ≤ r.d * r.aMean ^ 2 := by
    rw [r.sd_sq, div_le_iff₀ hd]
    have := r.sum_dev_le.trans r.sum_sq_le
    nlinarith
  have h2 : r.sd ^ 2 ≤ (2 ^ 8 * r.aMean) ^ 2 := by
    have := r.d_le15
    nlinarith [sq_nonneg r.aMean]
  have h3 := r.sd_nonneg'
  have h4 : 0 ≤ 2 ^ 8 * r.aMean := by have := r.aMean_nonneg'; positivity
  nlinarith

/-- `sd ≤ 2^33`. -/
theorem sd_le' : r.sd ≤ 2 ^ 33 := by
  have hd := r.d_pos'
  have h1 : r.sd ^ 2 ≤ 2 ^ 64 := by
    rw [r.sd_sq, div_le_iff₀ hd]
    refine r.sum_dev_le.trans ?_
    calc ∑ j ∈ Finset.range r.d, r.g j ^ 2 ≤ ∑ j ∈ Finset.range r.d, (2 : ℝ) ^ 64 :=
          Finset.sum_le_sum fun j hj => by
            have h := r.g_le j (Finset.mem_range.mp hj)
            have : r.g j ^ 2 = |r.g j| ^ 2 := (sq_abs _).symm
            rw [this]
            calc |r.g j| ^ 2 ≤ (2 ^ 32) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) h 2
              _ = 2 ^ 64 := by norm_num
      _ = 2 ^ 64 * r.d := by simp; ring
  have := r.sd_nonneg'
  nlinarith

/-- `cutTol ≥ 2^-130`. -/
theorem cutTol_ge' : 2 ^ (-130 : ℤ) ≤ r.cutTol := by
  unfold cutTol
  have := r.aMean_nonneg'; have := r.sd_nonneg'
  have h1 : 0 ≤ 2 ^ (-6 : ℤ) * (|r.mu| + |r.m| * r.sd) := by positivity
  have h2 : 0 ≤ 2 ^ (-17 : ℤ) * (1 + |r.m|) * r.aMean := by positivity
  linarith

/-- Summing a function that vanishes from `d` on over the block is summing it over `d`. -/
theorem sum_block {f : ℕ → ℝ} (hf : ∀ j, r.d ≤ j → f j = 0) :
    ∑ j ∈ Finset.range (2 ^ r.L), f j = ∑ j ∈ Finset.range r.d, f j := by
  refine (Finset.sum_subset (Finset.range_subset_range.mpr r.d_le) fun j _ hj => ?_).symm
  exact hf j (by simpa using hj)

/-- The block sum of `gl` is `d·mu`. -/
theorem sum_gl' : ∑ j ∈ Finset.range (2 ^ r.L), r.gl j = r.d * r.mu := by
  rw [r.sum_block (f := r.gl) fun j hj => by simp [gl, not_lt.mpr hj], ← r.sum_eq]
  exact Finset.sum_congr rfl fun j hj => by simp [gl, Finset.mem_range.mp hj]

/-- The block sum of `|gl|` is `d·aMean`. -/
theorem sum_abs_gl' : ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| = r.d * r.aMean := by
  rw [r.sum_block (f := fun j => |r.gl j|) fun j hj => by simp [gl, not_lt.mpr hj], ← r.sum_abs_eq]
  exact Finset.sum_congr rfl fun j hj => by simp [gl, Finset.mem_range.mp hj]

/-- On a row with `Ok`: `2^-62 ≤ sd`. -/
theorem Ok.sd_ge {r : RowP} (h : r.Ok) : 2 ^ (-62 : ℤ) ≤ r.sd := by
  have := r.aMean_nonneg'
  unfold Ok at h
  have : 0 ≤ 2 ^ (-18 : ℤ) * r.aMean := by positivity
  linarith

/-- On a row with `Ok`: `aMean ≤ 2^18·sd`. -/
theorem Ok.aMean_le_sd {r : RowP} (h : r.Ok) : r.aMean ≤ 2 ^ 18 * r.sd := by
  unfold Ok at h
  have e : r.aMean = 2 ^ 18 * (2 ^ (-18 : ℤ) * r.aMean) := by
    rw [← mul_assoc]; norm_num
  rw [e]
  have : (0 : ℝ) < 2 ^ (-62 : ℤ) := by positivity
  nlinarith

/-- On a row with `Ok`: `2^-70 ≤ aMean`. -/
theorem Ok.aMean_ge {r : RowP} (h : r.Ok) : 2 ^ (-70 : ℤ) ≤ r.aMean := by
  have h1 := h.sd_ge
  have h2 := r.sd_le_aMean
  have : (2 : ℝ) ^ (-62 : ℤ) = 2 ^ 8 * 2 ^ (-70 : ℤ) := by norm_num
  nlinarith

end RowP

end Claims.GeluAndMulSparse
