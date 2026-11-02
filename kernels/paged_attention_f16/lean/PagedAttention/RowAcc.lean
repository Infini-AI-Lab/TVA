import PagedAttention.RowMax

/-!
# `paged_attention_f16`: one row, the weights, `p` and `alpha`

* `psum`: a sum over the sequence's tokens among the first `t` pages; `wt R m j = exp(c·sv_j − m)`,
  the kernel's weight of token `j` against the running maximum `m`.
* the constants of a page (`psiPr`, `psiLr`, `psiAr`, `stepR`) and `ffac`, the relative bound after
  `t` pages, which grows by a factor `stepR` per page and by `exp(2^-22·Δm)` when the maximum moves.
* `lane_p`: one lane's `p = exp(s − m_new)` (line 65); `alpha_bound`, `alpha_zero`:
  `alpha = exp(m_i − m_new)` (line 64) against `exp(m_i − m_new)`, and `+0` on the first page.
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-! ## Sums over the sequence's tokens -/

/-- `Σ_{j < P·t, j < n} f j`. -/
noncomputable def psum (R : RowCtx) (t : ℕ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (R.P * t), if j < R.n then f j else 0

/-- The kernel's weight of token `j` against the running maximum `m`: `exp(c·sv_j − m)`. -/
noncomputable def wt (R : RowCtx) (m : ℝ) (j : ℕ) : ℝ := Real.exp (R.c * R.sv j - m)

theorem wt_pos (R : RowCtx) (m : ℝ) (j : ℕ) : 0 < wt R m j := Real.exp_pos _

theorem wt_rescale (R : RowCtx) (m m' : ℝ) (j : ℕ) : wt R m' j = Real.exp (m - m') * wt R m j := by
  unfold wt; rw [← Real.exp_add]; congr 1; ring

theorem psum_succ (R : RowCtx) (t : ℕ) (f : ℕ → ℝ) :
    psum R (t + 1) f = psum R t f +
      ∑ r : Fin R.P, (if R.P * t + r < R.n then f (R.P * t + r) else 0) := by
  unfold psum
  rw [show R.P * (t + 1) = R.P * t + R.P by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => if R.P * t + k < R.n then f (R.P * t + k) else 0) R.P]

theorem psum_zero (R : RowCtx) (f : ℕ → ℝ) : psum R 0 f = 0 := by simp [psum]

theorem psum_nonneg {R : RowCtx} {t : ℕ} {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) : 0 ≤ psum R t f :=
  Finset.sum_nonneg fun j _ => by split_ifs <;> simp [hf j]

theorem psum_mul (R : RowCtx) (t : ℕ) (k : ℝ) (f : ℕ → ℝ) :
    psum R t (fun j => k * f j) = k * psum R t f := by
  unfold psum; rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by split_ifs <;> simp

theorem abs_psum_le (R : RowCtx) (t : ℕ) (f : ℕ → ℝ) : |psum R t f| ≤ psum R t (fun j => |f j|) := by
  unfold psum
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j _ => ?_)
  split_ifs <;> simp

theorem psum_le {R : RowCtx} {t : ℕ} {f g : ℕ → ℝ}
    (h : ∀ j, j < R.P * t → j < R.n → f j ≤ g j) : psum R t f ≤ psum R t g :=
  Finset.sum_le_sum fun j hj => by
    split_ifs with ha
    · exact h j (Finset.mem_range.mp hj) ha
    · exact le_rfl

theorem psum_le_card {R : RowCtx} {t : ℕ} {f : ℕ → ℝ} {B : ℝ} (hB : 0 ≤ B)
    (h : ∀ j, j < R.P * t → j < R.n → f j ≤ B) : psum R t f ≤ R.P * t * B := by
  unfold psum
  calc ∑ j ∈ Finset.range (R.P * t), (if j < R.n then f j else 0) ≤ ∑ _j ∈ Finset.range (R.P * t), B :=
        Finset.sum_le_sum fun j hj => by
          split_ifs with ha
          · exact h j (Finset.mem_range.mp hj) ha
          · exact hB
    _ = R.P * t * B := by simp

theorem psum_ge_single {R : RowCtx} {t : ℕ} {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) {j : ℕ} (hj : j < R.P * t)
    (ha : j < R.n) : f j ≤ psum R t f := by
  unfold psum
  have := Finset.single_le_sum (f := fun i => if i < R.n then f i else 0)
    (fun i _ => by split_ifs <;> simp [hf i]) (Finset.mem_range.mpr hj)
  simpa [ha] using this

/-- The weights after a page: `Σ wt m'·g = exp(m − m')·Σ wt m·g + (page t)`. -/
theorem psum_step (R : RowCtx) (m m' : ℝ) (t : ℕ) (g : ℕ → ℝ) :
    psum R (t + 1) (fun j => wt R m' j * g j) =
      Real.exp (m - m') * psum R t (fun j => wt R m j * g j) +
        ∑ r : Fin R.P, (if R.P * t + r < R.n then wt R m' (R.P * t + r) * g (R.P * t + r) else 0) := by
  rw [psum_succ, ← psum_mul]
  congr 1
  unfold psum
  refine Finset.sum_congr rfl fun j _ => ?_
  dsimp only
  rw [wt_rescale R m m' j, mul_assoc]

/-! ## The constants of a page -/

/-- `Num.ExpF32`'s relative error on the exponents (`|x| ≤ 2Y + 1`). -/
noncomputable def relE (R : RowCtx) : ℝ := 1 / 2 ^ 22 + (2 * R.Y + 1) / 2 ^ 23

/-- `p`'s relative error: the exponent's error `E` and `exp`'s. -/
noncomputable def psiPr (R : RowCtx) : ℝ := (1 + R.E + R.E ^ 2) * (1 + relE R) - 1

/-- A page's `tl.sum(p)`'s relative error (`P − 1` IEEE additions). -/
noncomputable def psiLr (R : RowCtx) : ℝ := (1 + 1 / 2 ^ 24) ^ (R.P - 1) * (1 + psiPr R) - 1

/-- A page's `tl.sum(p[:, None] * v, axis=0)`'s relative error. -/
noncomputable def psiAr (R : RowCtx) : ℝ := (1 + 3 / 2 ^ 24) ^ (R.P - 1) * (1 + 1 / 2 ^ 24) * (1 + psiPr R) - 1

/-- The factor by which a page scales the earlier error. -/
noncomputable def stepR : ℝ := (1 + 3 / 2 ^ 24) * (1 + 1 / 2 ^ 140) * (1 + 1 / 2 ^ 22)

/-- `l_i`'s absolute error per page. -/
noncomputable def absLr : ℝ := 1 / 2 ^ 100

/-- The accumulator's absolute error per page. -/
noncomputable def absAr : ℝ := 1 / 2 ^ 90

/-- The relative bound after `t` pages, with running maximum `m`. -/
noncomputable def ffac (R : RowCtx) (psi : ℝ) (t : ℕ) (m : ℝ) : ℝ :=
  (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ t

theorem relE_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ relE R := by
  unfold relE; have := hR.Y_nonneg; positivity

theorem relE_le {R : RowCtx} (hR : R.Ok) : relE R ≤ 1 / 4000 := by
  have h : (2 * R.Y + 1) / 2 ^ 23 ≤ 2049 / 2 ^ 23 :=
    div_le_div_of_nonneg_right (by linarith [hR.Y_le]) (by norm_num)
  calc relE R = 1 / 2 ^ 22 + (2 * R.Y + 1) / 2 ^ 23 := rfl
    _ ≤ 1 / 2 ^ 22 + 2049 / 2 ^ 23 := by linarith
    _ ≤ 1 / 4000 := by norm_num

theorem psiPr_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ psiPr R := by
  unfold psiPr; have := relE_nonneg hR; have := hR.E_nonneg
  have : 0 ≤ R.E ^ 2 := sq_nonneg _
  nlinarith

theorem psiPr_le {R : RowCtx} (hR : R.Ok) : psiPr R ≤ 1 / 7 - 1 / 2 ^ 10 := by
  unfold psiPr
  have h1 := relE_le hR
  have h2 := relE_nonneg hR
  have h3 := hR.E_le
  have h4 := hR.E_nonneg
  have h5 : 1 + R.E + R.E ^ 2 ≤ 73 / 64 := by nlinarith
  have h6 : 0 ≤ 1 + R.E + R.E ^ 2 := by nlinarith
  have := mul_le_mul h5 (by linarith : 1 + relE R ≤ 1 + 1 / 4000) (by linarith) (by norm_num)
  norm_num at this ⊢
  linarith

theorem psiLr_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ psiLr R := by
  unfold psiLr; have := psiPr_nonneg hR
  have : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ (R.P - 1) := one_le_pow₀ (by norm_num)
  nlinarith

theorem psiAr_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ psiAr R := by
  unfold psiAr; have := psiPr_nonneg hR
  have : (1 : ℝ) ≤ (1 + 3 / 2 ^ 24) ^ (R.P - 1) := one_le_pow₀ (by norm_num)
  have h2 : (1 : ℝ) ≤ (1 + 3 / 2 ^ 24) ^ (R.P - 1) * (1 + 1 / 2 ^ 24) := by nlinarith
  nlinarith

theorem psiLr_le {R : RowCtx} (hR : R.Ok) : psiLr R ≤ 1 / 7 := by
  unfold psiLr
  have h1 := psiPr_le hR
  have h0 := psiPr_nonneg hR
  have hK := pow255_le' hR.P_le
  have hK1 : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ (R.P - 1) := one_le_pow₀ (by norm_num)
  have := mul_le_mul hK (by linarith : 1 + psiPr R ≤ 1 + (1 / 7 - 1 / 2 ^ 10)) (by linarith) (by norm_num)
  norm_num at this ⊢
  linarith

theorem psiAr_le {R : RowCtx} (hR : R.Ok) : psiAr R ≤ 1 / 7 := by
  unfold psiAr
  have h1 := psiPr_le hR
  have h0 := psiPr_nonneg hR
  have hK := pow255_le hR.P_le
  have hK1 : (1 : ℝ) ≤ (1 + 3 / 2 ^ 24) ^ (R.P - 1) := one_le_pow₀ (by norm_num)
  have h2 : (1 + 3 / 2 ^ 24) ^ (R.P - 1) * (1 + 1 / 2 ^ 24) ≤ (1 + 1 / 2 ^ 12) * (1 + 1 / 2 ^ 24 : ℝ) :=
    mul_le_mul_of_nonneg_right hK (by norm_num)
  have := mul_le_mul h2 (by linarith : 1 + psiPr R ≤ 1 + (1 / 7 - 1 / 2 ^ 10)) (by linarith) (by norm_num)
  norm_num at this ⊢
  linarith

theorem stepR_one : 1 ≤ stepR := by unfold stepR; norm_num

theorem stepR_le : stepR ≤ 1 + 1 / 2 ^ 21 := by unfold stepR; norm_num

theorem stepR_ge : 1 + 3 / 2 ^ 24 ≤ stepR := by unfold stepR; norm_num

theorem m_nonneg {R : RowCtx} {m : ℝ} (h : |m| ≤ R.Y + R.E) : 0 ≤ m + R.Y + R.E := by
  have := neg_abs_le m; linarith

/-- `ffac` at least `(1 + ρ)(1 + ψ)` after a page. -/
theorem ffac_ge {R : RowCtx} {psi rho : ℝ} {t : ℕ} {m : ℝ} (hpsi : 0 ≤ psi) (hrho : 0 ≤ rho)
    (hstep : 1 + rho ≤ stepR) (hm : 0 ≤ m + R.Y + R.E) :
    (1 + rho) * (1 + psi) ≤ ffac R psi (t + 1) m := by
  unfold ffac
  have he : 1 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 22) := Real.one_le_exp (div_nonneg hm (by norm_num))
  have hs : 1 + rho ≤ stepR ^ (t + 1) := by
    calc 1 + rho ≤ stepR := hstep
      _ = stepR ^ 1 := (pow_one _).symm
      _ ≤ stepR ^ (t + 1) := pow_le_pow_right₀ stepR_one (by omega)
  have h0 : 0 ≤ 1 + psi := by linarith
  calc (1 + rho) * (1 + psi) ≤ stepR ^ (t + 1) * (1 + psi) := mul_le_mul_of_nonneg_right hs h0
    _ ≤ (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ (t + 1) := by
        have : (1 + psi) * 1 ≤ (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) :=
          mul_le_mul_of_nonneg_left he h0
        have hs0 : 0 ≤ stepR ^ (t + 1) := pow_nonneg (by linarith [stepR_one]) _
        nlinarith

theorem ffac_one_le {R : RowCtx} {psi : ℝ} {t : ℕ} {m : ℝ} (hpsi : 0 ≤ psi)
    (hm : 0 ≤ m + R.Y + R.E) : 1 ≤ ffac R psi t m := by
  unfold ffac
  have he : 1 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 22) := Real.one_le_exp (div_nonneg hm (by norm_num))
  have hs : 1 ≤ stepR ^ t := one_le_pow₀ stepR_one
  have := mul_le_mul (by linarith : (1 : ℝ) ≤ 1 + psi) he (by norm_num) (by linarith)
  nlinarith

/-- **The step of `ffac`**: with `m ≤ m'`,
`(1 + 3·2^-24)·((1 + dAl)(1 + 2^-22))·ffac t m ≤ ffac (t + 1) m'`, `dAl = (m' − m)/2^22 + 2^-140`. -/
theorem ffac_step {R : RowCtx} {psi : ℝ} {t : ℕ} {m m' : ℝ} (hpsi : 0 ≤ psi) (hmm : m ≤ m') :
    (1 + 3 / 2 ^ 24) * ((1 + ((m' - m) / 2 ^ 22 + 1 / 2 ^ 140)) * (1 + 1 / 2 ^ 22)) * ffac R psi t m ≤
      ffac R psi (t + 1) m' := by
  unfold ffac
  have ha : 0 ≤ (m' - m) / 2 ^ 22 := by apply div_nonneg <;> linarith
  have h1 : 1 + ((m' - m) / 2 ^ 22 + 1 / 2 ^ 140) ≤ Real.exp ((m' - m) / 2 ^ 22) * (1 + 1 / 2 ^ 140) := by
    have := Real.add_one_le_exp ((m' - m) / 2 ^ 22)
    nlinarith
  have hexp : Real.exp ((m' - m) / 2 ^ 22) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) =
      Real.exp ((m' + R.Y + R.E) / 2 ^ 22) := by rw [← Real.exp_add]; congr 1; ring
  have hs0 : 0 ≤ stepR := by linarith [stepR_one]
  have hst : 0 ≤ stepR ^ t := pow_nonneg hs0 t
  have hE1 : 0 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 22) := (Real.exp_pos _).le
  have hP : 0 ≤ (1 + 3 / 2 ^ 24) * (1 + 1 / 2 ^ 22) * (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) *
      stepR ^ t := by positivity
  calc (1 + 3 / 2 ^ 24) * ((1 + ((m' - m) / 2 ^ 22 + 1 / 2 ^ 140)) * (1 + 1 / 2 ^ 22)) *
        ((1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ t)
      = (1 + ((m' - m) / 2 ^ 22 + 1 / 2 ^ 140)) *
          ((1 + 3 / 2 ^ 24) * (1 + 1 / 2 ^ 22) * (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ t) := by
        ring
    _ ≤ (Real.exp ((m' - m) / 2 ^ 22) * (1 + 1 / 2 ^ 140)) *
          ((1 + 3 / 2 ^ 24) * (1 + 1 / 2 ^ 22) * (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ t) :=
        mul_le_mul_of_nonneg_right h1 hP
    _ = (1 + psi) * (Real.exp ((m' - m) / 2 ^ 22) * Real.exp ((m + R.Y + R.E) / 2 ^ 22)) *
          (stepR ^ t * stepR) := by unfold stepR; ring
    _ = (1 + psi) * Real.exp ((m' + R.Y + R.E) / 2 ^ 22) * stepR ^ (t + 1) := by rw [hexp, pow_succ stepR t]

/-- **`ffac ≤ 5/4`** over the covered loop (`t ≤ 2^16` pages, `|m| ≤ Y + E`, `ψ ≤ 1/7`). -/
theorem ffac_le {R : RowCtx} (hR : R.Ok) {psi : ℝ} {t : ℕ} {m : ℝ} (hpsi : psi ≤ 1 / 7)
    (ht : t ≤ 2 ^ 16) (hm : |m| ≤ R.Y + R.E) : ffac R psi t m ≤ 5 / 4 := by
  unfold ffac
  have hY := hR.Y_le
  have hE := hR.E_le
  have hE0 := hR.E_nonneg
  have hm0 := m_nonneg hm
  have hm1 : m + R.Y + R.E ≤ 2050 := by have := le_abs_self m; linarith
  have h1 : Real.exp ((m + R.Y + R.E) / 2 ^ 22) ≤ 1 + 1 / 2 ^ 9 := by
    have h1a := exp_le_two (y := (m + R.Y + R.E) / 2 ^ 22) (by positivity)
      (by rw [div_le_iff₀ (by norm_num)]; linarith)
    have h1b : (m + R.Y + R.E) / 2 ^ 22 ≤ 2050 / 2 ^ 22 := div_le_div_of_nonneg_right hm1 (by norm_num)
    norm_num at h1a h1b ⊢
    linarith
  have h2 : stepR ^ t ≤ 1 + 2 * ((2 ^ 16 : ℕ) * (1 / 2 ^ 21)) :=
    pow_le_of_step stepR_one stepR_le ht (by norm_num)
  have hst : 0 ≤ stepR ^ t := pow_nonneg (by linarith [stepR_one]) t
  have hE1 : 0 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 22) := (Real.exp_pos _).le
  have hpsi0 : (0 : ℝ) ≤ 1 + psi ∨ 1 + psi < 0 := le_or_gt 0 (1 + psi)
  rcases hpsi0 with hp | hp
  · calc (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ t
        ≤ (1 + 1 / 7) * (1 + 1 / 2 ^ 9) * (1 + 2 * ((2 ^ 16 : ℕ) * (1 / 2 ^ 21))) := by
          apply mul_le_mul (mul_le_mul (by linarith) h1 hE1 (by norm_num)) h2 hst (by norm_num)
      _ ≤ 5 / 4 := by norm_num
  · have : (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ t ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hp.le hE1
      exact mul_nonpos_of_nonpos_of_nonneg this hst
    linarith

/-- `ffac` against its value at the end of the loop: `exp` bounded by the drift. -/
theorem ffac_le_drift {R : RowCtx} (hR : R.Ok) {psi : ℝ} {T nb : ℕ} {m : ℝ} (hpsi : 0 ≤ psi)
    (hT : T ≤ nb) (hm : |m| ≤ R.Y + R.E) :
    ffac R psi T m ≤ (1 + psi) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ nb := by
  unfold ffac
  have hY := hR.Y_le
  have hE := hR.E_le
  have hm0 := m_nonneg hm
  have hm1 : m + R.Y + R.E ≤ 2 * (R.Y + R.E) := by have := le_abs_self m; linarith
  have h1 : Real.exp ((m + R.Y + R.E) / 2 ^ 22) ≤ 1 + 4 / 2 ^ 22 * (R.Y + R.E) := by
    have := exp_le_two (y := (m + R.Y + R.E) / 2 ^ 22) (by positivity) (by
      rw [div_le_iff₀ (by norm_num)]; linarith)
    have : 2 * ((m + R.Y + R.E) / 2 ^ 22) ≤ 4 / 2 ^ 22 * (R.Y + R.E) := by
      rw [show 2 * ((m + R.Y + R.E) / 2 ^ 22) = 2 / 2 ^ 22 * (m + R.Y + R.E) by ring]
      nlinarith
    linarith
  have h2 : stepR ^ T ≤ stepR ^ nb := pow_le_pow_right₀ stepR_one hT
  have h0 : 0 ≤ stepR ^ T := pow_nonneg (by linarith [stepR_one]) T
  have := mul_le_mul_of_nonneg_left h1 (by linarith : (0 : ℝ) ≤ 1 + psi)
  have h3 : 0 ≤ (1 + psi) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) := by
    have := hR.Y_nonneg; have := hR.E_nonneg; positivity
  calc (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 22) * stepR ^ T
      ≤ (1 + psi) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ T := mul_le_mul_of_nonneg_right this h0
    _ ≤ (1 + psi) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ nb := mul_le_mul_of_nonneg_left h2 h3

/-! ## Facts about the weights -/

/-- A token's weight against a running maximum that bounds it: at most `2`. -/
theorem wt_le_two {R : RowCtx} (hR : R.Ok) {t : ℕ} {m : F32} (hm : MInv R t m) {j : ℕ} (hj : j < R.P * t)
    (ha : j < R.n) : wt R m.toReal j ≤ 2 := by
  have h1 := hm.above j hj ha
  have h2 := hR.E_le
  unfold wt
  have h3 : Real.exp (R.c * R.sv j - m.toReal) ≤ Real.exp (1 / 8) := Real.exp_le_exp.mpr (by linarith)
  have h4 := exp_le_two (y := 1 / 8) (by norm_num) (by norm_num)
  linarith

/-- The sum of weights over `t ≤ 2^16` pages: at most `2^25`. -/
theorem psum_wt1_le {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 2 ^ 16) {m : F32} (hm : MInv R t m) :
    psum R t (wt R m.toReal) ≤ 2 ^ 25 := by
  have h := psum_le_card (R := R) (t := t) (f := wt R m.toReal) (B := 2) (by norm_num)
    fun j hj ha => wt_le_two hR hm hj ha
  have hP : (R.P : ℝ) ≤ 256 := by exact_mod_cast hR.P_le
  have htr : (t : ℝ) ≤ 2 ^ 16 := by exact_mod_cast ht
  have : (R.P : ℝ) * t ≤ 256 * 2 ^ 16 := mul_le_mul hP htr (by positivity) (by norm_num)
  nlinarith

/-- The sum of a row's weights against its maximum is at least `1 − E`. -/
theorem W_ge {R : RowCtx} (hR : R.Ok) {t : ℕ} {m : F32} (hm : MInv R t m) :
    1 - R.E ≤ psum R t (wt R m.toReal) := by
  obtain ⟨j, hj, ha, hd⟩ := hm.attained
  have h1 := psum_ge_single (f := wt R m.toReal) (fun i => (wt_pos _ _ _).le) hj ha
  have h2 : Real.exp (-R.E) ≤ wt R m.toReal j := by
    unfold wt; apply Real.exp_le_exp.mpr
    have := le_abs_self (m.toReal - R.c * R.sv j); linarith
  have := Real.add_one_le_exp (-R.E)
  linarith

/-! ## One lane: `p = exp(s − m_new)` (line 65) -/

/-- **One token's `p`**: within `psiPr·wt + 2^-140` of its weight against the new maximum. -/
theorem lane_p {P : Platform} (hexp : Rules.Num.ExpF32 P) {R : RowCtx} (hR : R.Ok) {m' x p : F32}
    {j : ℕ} (hj : j < R.n) (hx : x.isFinite = true)
    (hxe : |x.toReal - (R.c * R.sv j - m'.toReal)| ≤ R.E)
    (habove : R.c * R.sv j ≤ m'.toReal + R.E) (hm' : |m'.toReal| ≤ R.Y + R.E)
    (hp : P.math .exp .binary32 x p) :
    p.isFinite = true ∧ |p.toReal - wt R m'.toReal j| ≤ psiPr R * wt R m'.toReal j + 1 / 2 ^ 140 := by
  have hE0 := hR.E_nonneg
  have hE8 := hR.E_le
  have hY := hR.Y_le
  have hY0 := hR.Y_nonneg
  have hcs := hR.csv_le hj
  have hx1 : x.toReal ≤ 1 := by have := le_abs_self (x.toReal - (R.c * R.sv j - m'.toReal)); linarith
  have hxa : |x.toReal| ≤ 2 * R.Y + 1 := by
    have := abs_sub_abs_le_abs_sub x.toReal (R.c * R.sv j - m'.toReal)
    have := abs_sub (R.c * R.sv j) m'.toReal
    linarith
  obtain ⟨hpf, hpe⟩ := exp_acc hexp hp hx hx1 (by linarith)
  refine ⟨hpf, ?_⟩
  set G := wt R m'.toReal j with hG
  have hGp : 0 < G := wt_pos _ _ _
  set δ := x.toReal - (R.c * R.sv j - m'.toReal) with hδ
  have hsplit : Real.exp x.toReal = Real.exp δ * G := by
    rw [hG, wt, ← Real.exp_add]; congr 1; rw [hδ]; ring
  have h1 := abs_exp_sub_one' (show |δ| ≤ 1 by linarith)
  have hδ2 : δ ^ 2 ≤ R.E ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hxe 2
  have h2 : |Real.exp x.toReal - G| ≤ (R.E + R.E ^ 2) * G := by
    rw [hsplit, show Real.exp δ * G - G = (Real.exp δ - 1) * G by ring, abs_mul, abs_of_pos hGp]
    exact mul_le_mul_of_nonneg_right (by linarith) hGp.le
  have h3 : Real.exp x.toReal ≤ (1 + R.E + R.E ^ 2) * G := by
    have := le_abs_self (Real.exp x.toReal - G); linarith
  have hrel : 1 / 2 ^ 22 + |x.toReal| / 2 ^ 23 ≤ relE R := by
    unfold relE; have := div_le_div_of_nonneg_right hxa (by norm_num : (0 : ℝ) ≤ 2 ^ 23); linarith
  have hexp0 := (Real.exp_pos x.toReal).le
  have h4 : (1 / 2 ^ 22 + |x.toReal| / 2 ^ 23) * Real.exp x.toReal ≤ relE R * ((1 + R.E + R.E ^ 2) * G) :=
    mul_le_mul hrel h3 hexp0 (relE_nonneg hR)
  have := abs_sub_le p.toReal (Real.exp x.toReal) G
  have e : psiPr R * G = relE R * ((1 + R.E + R.E ^ 2) * G) + (R.E + R.E ^ 2) * G := by unfold psiPr; ring
  linarith

/-- **A masked lane's `p`** (line 65): `exp(−∞) = +0`. -/
theorem lane_p_masked {P : Platform} (hni : Rules.Num.ExpF32NegInf P) {p : F32}
    (hp : P.math .exp .binary32 FloatV.negInf p) : p = Rules.Num.zeroF32 := by
  rw [← Rules.Num.negInfF32_eq_negInf] at hp
  exact hni p hp

theorem zero32 : (FloatV.ofPat 0 : F32).isFinite = true ∧ (FloatV.ofPat 0 : F32).toRat = 0 :=
  DotLoop.val?_f32.mp DotLoop.zero_val

theorem zeroF32_facts : Rules.Num.zeroF32.isFinite = true ∧ Rules.Num.zeroF32.toReal = 0 := by
  refine ⟨zero32.1, ?_⟩
  rw [toReal_def, show (Rules.Num.zeroF32 : F32) = FloatV.ofPat 0 from rfl, zero32.2]; simp

/-! ## `alpha` (line 64) -/

/-- The relative error of `alpha`'s input, besides `exp`'s `2^-22`: `(m' − m)·2^-22 + 2^-140`. -/
noncomputable def dAl (m m' : ℝ) : ℝ := (m' - m) / 2 ^ 22 + 1 / 2 ^ 140

theorem dAl_nonneg {m m' : ℝ} (h : m ≤ m') : 0 ≤ dAl m m' := by
  unfold dAl; have : 0 ≤ m' - m := by linarith
  positivity

/-- **`alpha` after the first page**: within `((1 + dAl)(1 + 2^-22) − 1)·exp(m − m') + 2^-140` of
`exp(m − m')`. -/
theorem alpha_bound {P : Platform} (hexp : Rules.Num.ExpF32 P) {R : RowCtx} (hR : R.Ok) {t : ℕ}
    {m m' dm al : F32} (hin : AlphaIn t m m' dm) (ht : 0 < t) (hm : |m.toReal| ≤ R.Y + R.E)
    (hm' : |m'.toReal| ≤ R.Y + R.E) (ha : P.math .exp .binary32 dm al) :
    al.isFinite = true ∧ m.toReal ≤ m'.toReal ∧
      |al.toReal - Real.exp (m.toReal - m'.toReal)| ≤
        ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * Real.exp (m.toReal - m'.toReal) + 1 / 2 ^ 140 := by
  rcases hin with ⟨h0, -⟩ | ⟨-, -, hle, hdf, hde⟩
  · omega
  have hY := hR.Y_le
  have hE := hR.E_le
  have hE0 := hR.E_nonneg
  set Δ := m'.toReal - m.toReal with hΔ
  have hΔ0 : 0 ≤ Δ := by rw [hΔ]; linarith
  have hΔ1 : Δ ≤ 2050 := by
    rw [hΔ]; have := neg_abs_le m.toReal; have := le_abs_self m'.toReal; linarith
  have hdm1 : dm.toReal ≤ 1 := by
    have := le_abs_self (dm.toReal - (m.toReal - m'.toReal))
    have : Δ / 2 ^ 24 ≤ 2050 / 2 ^ 24 := div_le_div_of_nonneg_right hΔ1 (by norm_num)
    norm_num at this hde ⊢; linarith
  have hdma : |dm.toReal| ≤ Δ * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub dm.toReal (m.toReal - m'.toReal)
    rw [abs_of_nonpos (by linarith : m.toReal - m'.toReal ≤ 0)] at this
    have e : -(m.toReal - m'.toReal) = Δ := by rw [hΔ]; ring
    rw [e] at this
    nlinarith
  obtain ⟨haf, hae⟩ := exp_acc hexp ha hdf hdm1 (by nlinarith)
  refine ⟨haf, hle, ?_⟩
  set lam := Real.exp (m.toReal - m'.toReal) with hlam
  have hlp : 0 < lam := Real.exp_pos _
  set δ := dm.toReal - (m.toReal - m'.toReal) with hδ
  set δ0 := Δ / 2 ^ 24 + 1 / 2 ^ 150 with hδ0d
  have hδd : |δ| ≤ δ0 := by rw [hδ, hδ0d, hΔ]; exact hde
  have hδ01 : δ0 ≤ 1 / 2 ^ 12 := by
    rw [hδ0d]; have : Δ / 2 ^ 24 ≤ 2050 / 2 ^ 24 := div_le_div_of_nonneg_right hΔ1 (by norm_num)
    norm_num at this ⊢; linarith
  have hδ00 : 0 ≤ δ0 := (abs_nonneg _).trans hδd
  have hsplit : Real.exp dm.toReal = Real.exp δ * lam := by
    rw [hlam, ← Real.exp_add]; congr 1; rw [hδ]; ring
  have h1 := abs_exp_sub_one' (show |δ| ≤ 1 by linarith)
  have hδ2 : δ ^ 2 ≤ δ0 * (1 / 2 ^ 12) := by
    rw [← sq_abs, sq]; exact mul_le_mul hδd (hδd.trans hδ01) (abs_nonneg _) hδ00
  have h2 : |Real.exp dm.toReal - lam| ≤ δ0 * (1 + 1 / 2 ^ 12) * lam := by
    rw [hsplit, show Real.exp δ * lam - lam = (Real.exp δ - 1) * lam by ring, abs_mul, abs_of_pos hlp]
    exact mul_le_mul_of_nonneg_right (by linarith) hlp.le
  have h3 : Real.exp dm.toReal ≤ (1 + δ0 * (1 + 1 / 2 ^ 12)) * lam := by
    have := le_abs_self (Real.exp dm.toReal - lam); linarith
  set ρ := 1 / 2 ^ 22 + |dm.toReal| / 2 ^ 23 with hρ
  have hρ0 : 0 ≤ ρ := by rw [hρ]; positivity
  have hρ1 : ρ ≤ 1 / 2 ^ 10 := by
    rw [hρ]; have : |dm.toReal| ≤ 2 ^ 12 := by nlinarith
    have := div_le_div_of_nonneg_right this (by norm_num : (0 : ℝ) ≤ 2 ^ 23)
    norm_num at this ⊢; linarith
  have hρ2 : ρ ≤ 1 / 2 ^ 22 + (Δ * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150) / 2 ^ 23 := by
    rw [hρ]; have := div_le_div_of_nonneg_right hdma (by norm_num : (0 : ℝ) ≤ 2 ^ 23); linarith
  have hexp0 := (Real.exp_pos dm.toReal).le
  have h4 : ρ * Real.exp dm.toReal ≤ ρ * ((1 + δ0 * (1 + 1 / 2 ^ 12)) * lam) :=
    mul_le_mul_of_nonneg_left h3 hρ0
  -- the coefficient
  have hcoef : ρ * (1 + δ0 * (1 + 1 / 2 ^ 12)) + δ0 * (1 + 1 / 2 ^ 12) ≤
      (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 := by
    have k1 : ρ * (δ0 * (1 + 1 / 2 ^ 12)) ≤ 1 / 2 ^ 10 * (δ0 * (1 + 1 / 2 ^ 12)) :=
      mul_le_mul_of_nonneg_right hρ1 (by positivity)
    unfold dAl
    rw [← hΔ]
    rw [hδ0d] at k1 ⊢
    nlinarith
  have := abs_sub_le al.toReal (Real.exp dm.toReal) lam
  have k2 := mul_le_mul_of_nonneg_right hcoef hlp.le
  have e : (ρ * (1 + δ0 * (1 + 1 / 2 ^ 12)) + δ0 * (1 + 1 / 2 ^ 12)) * lam =
      ρ * ((1 + δ0 * (1 + 1 / 2 ^ 12)) * lam) + δ0 * (1 + 1 / 2 ^ 12) * lam := by ring
  linarith

/-- **`alpha` on the first page**: `exp(-inf) = +0`. -/
theorem alpha_zero {P : Platform} (hni : Rules.Num.ExpF32NegInf P) {m m' dm al : F32}
    (hin : AlphaIn 0 m m' dm) (ha : P.math .exp .binary32 dm al) :
    al.isFinite = true ∧ al.toReal = 0 := by
  rcases hin with ⟨-, rfl⟩ | ⟨h, -⟩
  · rw [lane_p_masked hni ha]; exact zeroF32_facts
  · omega

/-- `alpha` is finite and at most `2`. -/
theorem alpha_le_two {P : Platform} (hexp : Rules.Num.ExpF32 P) (hni : Rules.Num.ExpF32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} {m m' dm al : F32} (hM' : MInv R (t + 1) m')
    (hin : AlphaIn t m m' dm) (ha : P.math .exp .binary32 dm al) (hm : 0 < t → MInv R t m) :
    al.isFinite = true ∧ |al.toReal| ≤ 2 := by
  rcases Nat.eq_zero_or_pos t with h0 | ht0
  · subst h0; have := alpha_zero hni hin ha; exact ⟨this.1, by rw [this.2]; norm_num⟩
  · have hmi := hm ht0
    obtain ⟨haf, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    refine ⟨haf, ?_⟩
    have hl1 : Real.exp (m.toReal - m'.toReal) ≤ 1 := by
      rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr (by linarith)
    have hl0 := Real.exp_pos (m.toReal - m'.toReal)
    have hd : dAl m.toReal m'.toReal ≤ 1 / 2 ^ 10 := by
      unfold dAl
      have h1 := hmi.bound
      have h2 := hM'.bound
      have h3 := hR.Y_le
      have h4 := hR.E_le
      have h5 := neg_abs_le m.toReal
      have h6 := le_abs_self m'.toReal
      have hΔ : m'.toReal - m.toReal ≤ 2050 := by linarith
      have h7 : (m'.toReal - m.toReal) / 2 ^ 22 ≤ 2050 / 2 ^ 22 := div_le_div_of_nonneg_right hΔ (by norm_num)
      have h8 : (2050 : ℝ) / 2 ^ 22 + 1 / 2 ^ 140 ≤ 1 / 2 ^ 10 := by norm_num
      linarith
    have hd0 := dAl_nonneg hle
    have := abs_sub_abs_le_abs_sub al.toReal (Real.exp (m.toReal - m'.toReal))
    rw [abs_of_pos hl0] at this
    have k : ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * Real.exp (m.toReal - m'.toReal) ≤ 1 / 2 := by
      have : (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 ≤ 1 / 2 := by nlinarith
      have h0 : 0 ≤ (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 := by nlinarith
      nlinarith
    linarith

end Kernels.PagedAttention
