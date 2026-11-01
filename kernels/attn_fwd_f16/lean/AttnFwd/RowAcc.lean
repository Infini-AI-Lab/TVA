import AttnFwd.RowMax

/-!
# `attn_fwd_f16`: one row, the sums `l_i` and the accumulator `acc`

* `psum`: a sum over the attended keys of the first `t` blocks; `wt R m j = 2^(c·s_j − m)`, the
  kernel's weight of key `j` against the running maximum `m`.
* `LInv`, `AInv`: after `t ≥ 1` blocks, `l_i` and each element of a row of `acc` are within a
  relative error `ffac − 1` (which grows by a factor `step` per block) and an absolute error of
  their exact values `Σ wt` and `Σ wt·v`.
* `lane_p`, `lane_p16`: one lane's `p = exp2(qk)` (line 82) and `p.to(float16)` (line 101).
* `alpha_bound`: `alpha = exp2(m_i − m_ij)` (line 84) against `2^(m_i − m_ij)`.
* `lstep` (lines 85, 106) and `astep` (lines 95, 103): one block.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

/-! ## Sums over the attended keys -/

/-- `Σ_{j < 64t, al j} f j`. -/
noncomputable def psum (al : ℕ → Bool) (t : ℕ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (64 * t), if al j = true then f j else 0

/-- The kernel's weight of key `j` against the running maximum `m`: `2^(c·s_j − m)`. -/
noncomputable def wt (R : RowCtx) (m : ℝ) (j : ℕ) : ℝ := (2 : ℝ) ^ (R.c * R.sv j - m)

theorem wt_pos (R : RowCtx) (m : ℝ) (j : ℕ) : 0 < wt R m j := two_rpow_pos _

theorem wt_rescale (R : RowCtx) (m m' : ℝ) (j : ℕ) : wt R m' j = (2 : ℝ) ^ (m - m') * wt R m j := by
  unfold wt; rw [← two_rpow_add]; congr 1; ring

theorem psum_succ (al : ℕ → Bool) (t : ℕ) (f : ℕ → ℝ) :
    psum al (t + 1) f = psum al t f + ∑ jj : Fin 64, (if al (64 * t + jj) = true then f (64 * t + jj) else 0) := by
  unfold psum
  rw [show 64 * (t + 1) = 64 * t + 64 by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => if al (64 * t + k) = true then f (64 * t + k) else 0) 64]

theorem psum_zero (al : ℕ → Bool) (f : ℕ → ℝ) : psum al 0 f = 0 := by simp [psum]

theorem psum_nonneg {al : ℕ → Bool} {t : ℕ} {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) : 0 ≤ psum al t f :=
  Finset.sum_nonneg fun j _ => by split_ifs <;> simp [hf j]

theorem psum_mul (al : ℕ → Bool) (t : ℕ) (k : ℝ) (f : ℕ → ℝ) :
    psum al t (fun j => k * f j) = k * psum al t f := by
  unfold psum; rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by split_ifs <;> simp

theorem abs_psum_le (al : ℕ → Bool) (t : ℕ) (f : ℕ → ℝ) : |psum al t f| ≤ psum al t (fun j => |f j|) := by
  unfold psum
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j _ => ?_)
  split_ifs <;> simp

theorem psum_le {al : ℕ → Bool} {t : ℕ} {f g : ℕ → ℝ}
    (h : ∀ j, j < 64 * t → al j = true → f j ≤ g j) : psum al t f ≤ psum al t g :=
  Finset.sum_le_sum fun j hj => by
    split_ifs with ha
    · exact h j (Finset.mem_range.mp hj) ha
    · exact le_rfl

theorem psum_le_card {al : ℕ → Bool} {t : ℕ} {f : ℕ → ℝ} {B : ℝ} (hB : 0 ≤ B)
    (h : ∀ j, j < 64 * t → al j = true → f j ≤ B) : psum al t f ≤ 64 * t * B := by
  unfold psum
  calc ∑ j ∈ Finset.range (64 * t), (if al j = true then f j else 0) ≤ ∑ _j ∈ Finset.range (64 * t), B :=
        Finset.sum_le_sum fun j hj => by
          split_ifs with ha
          · exact h j (Finset.mem_range.mp hj) ha
          · exact hB
    _ = 64 * t * B := by simp

/-! ## The constants of a step -/

/-- `p`'s relative error: `exp2`'s `2^-22` and the exponent's error `E`. -/
noncomputable def psiPr (E : ℝ) : ℝ := (1 + E) * (1 + 1 / 2 ^ 22) - 1

/-- The same after `p.to(float16)`. -/
noncomputable def psi16r (E : ℝ) : ℝ := (1 + psiPr E) * (1 + 1 / 2 ^ 11) - 1

/-- `l_ij`'s relative error (a sum of `64` terms). -/
noncomputable def psiLr (E : ℝ) : ℝ := (1 + 1 / 2 ^ 24) ^ 63 * (1 + psiPr E) - 1

/-- The factor by which a step scales the accumulator's earlier error. -/
noncomputable def stepAr : ℝ := (1 + 66 / 2 ^ 22) * (1 + 1 / 2 ^ 150) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24))

/-- The same for `l_i`. -/
noncomputable def stepLr : ℝ := (1 + 3 / 2 ^ 24) * (1 + 1 / 2 ^ 150) * (1 + 1 / 2 ^ 22)

/-- `l_i`'s absolute error per block. -/
noncomputable def absLr : ℝ := 1 / 2 ^ 100

/-- The accumulator's absolute error per block, besides `om0·|v|` per key and `kapAr` per unit of
weight: mainly the `p·v` `tl.dot`'s `66·2^-22` times the rest of the lift of the block's operands
(`2^-14·omR + 2^-28` per key). -/
noncomputable def absAr : ℝ := 1 / 2 ^ 37

/-- The accumulator's absolute error per unit of weight: the `p·v` `tl.dot`'s `66·2^-22` times the
lift `2^-14·|p|` of a subnormal `v`, with `|p| ≤ (1 + psi16r)·wt + omR`. -/
noncomputable def kapAr : ℝ := 1 / 2 ^ 29

/-- The absolute error of a key's `p.to(float16)`. -/
noncomputable def omR : ℝ := 1 / 2 ^ 25 + 1 / 2 ^ 125

/-- The accumulator's absolute error per key, times `|v|`: `omR` and the `p·v` `tl.dot`'s
`66·2^-22` times the lift `2^-14` of a subnormal `p`. -/
noncomputable def om0 : ℝ := 1 / 2 ^ 25 + 1 / 2 ^ 125 + 66 / 2 ^ 36

/-- The relative bound after `t` blocks, with running maximum `m`. -/
noncomputable def ffac (R : RowCtx) (psi step : ℝ) (t : ℕ) (m : ℝ) : ℝ :=
  (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * step ^ t

theorem psiPr_nonneg {E : ℝ} (h : 0 ≤ E) : 0 ≤ psiPr E := by unfold psiPr; nlinarith

theorem psi16r_nonneg {E : ℝ} (h : 0 ≤ E) : 0 ≤ psi16r E := by
  unfold psi16r; have := psiPr_nonneg h; nlinarith

theorem psiLr_nonneg {E : ℝ} (h : 0 ≤ E) : 0 ≤ psiLr E := by
  unfold psiLr; have := psiPr_nonneg h
  have : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ 63 := one_le_pow₀ (by norm_num)
  nlinarith

theorem psiPr_le {E : ℝ} (h : E ≤ 1 / 8) : psiPr E ≤ 1 / 8 + 1 / 2 ^ 21 := by
  unfold psiPr; nlinarith

theorem psi16r_le {E : ℝ} (h0 : 0 ≤ E) (h : E ≤ 1 / 8) : psi16r E ≤ 1 / 7 := by
  unfold psi16r; have := psiPr_le h; have := psiPr_nonneg h0; nlinarith

theorem one_add_pow_le_exp {x : ℝ} (hx : 0 ≤ x) (n : ℕ) : (1 + x) ^ n ≤ Real.exp (n * x) := by
  rw [Real.exp_nat_mul]
  exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp x]) n

theorem exp_le_two {y : ℝ} (h0 : 0 ≤ y) (h1 : y ≤ 1) : Real.exp y ≤ 1 + 2 * y := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := y) (by rw [abs_of_nonneg h0]; exact h1)
  have := le_abs_self (Real.exp y - 1 - y)
  nlinarith

theorem pow63_le : ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 ≤ 1 + 1 / 2 ^ 17 := by
  have h1 := one_add_pow_le_exp (x := (1 / 2 ^ 24 : ℝ)) (by norm_num) 63
  have h2 := exp_le_two (y := ((63 : ℕ) : ℝ) * (1 / 2 ^ 24)) (by norm_num) (by norm_num)
  have : 1 + 2 * (((63 : ℕ) : ℝ) * (1 / 2 ^ 24)) ≤ 1 + 1 / 2 ^ 17 := by norm_num
  linarith

theorem psiLr_le {E : ℝ} (h0 : 0 ≤ E) (h : E ≤ 1 / 8) : psiLr E ≤ 1 / 7 := by
  unfold psiLr; have := psiPr_le h; have := psiPr_nonneg h0; have := pow63_le; nlinarith

theorem stepAr_ge : 1 + 66 / 2 ^ 22 ≤ stepAr := by
  unfold stepAr
  have h0 : (1 : ℝ) ≤ (1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24) := by nlinarith
  have h1 : (1 : ℝ) ≤ (1 + 1 / 2 ^ 150) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) := by nlinarith
  nlinarith

theorem stepLr_ge : 1 + 3 / 2 ^ 24 ≤ stepLr := by
  unfold stepLr
  have : (1 : ℝ) ≤ (1 + 1 / 2 ^ 150) * (1 + 1 / 2 ^ 22) := by nlinarith
  nlinarith

theorem stepAr_le : stepAr ≤ 1 + 1 / 2 ^ 14 := by unfold stepAr; norm_num

theorem stepLr_le : stepLr ≤ 1 + 1 / 2 ^ 14 := by unfold stepLr; norm_num

/-- `ffac` at least `(1 + ρ)(1 + ψ)` after a step. -/
theorem ffac_ge {R : RowCtx} {psi step rho : ℝ} {t : ℕ} {m : ℝ} (hpsi : 0 ≤ psi) (hrho : 0 ≤ rho)
    (hstep : 1 + rho ≤ step) (hm : 0 ≤ m + R.Y + R.E) :
    (1 + rho) * (1 + psi) ≤ ffac R psi step (t + 1) m := by
  unfold ffac
  have he : 1 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 24) := Real.one_le_exp (div_nonneg hm (by norm_num))
  have hs : 1 + rho ≤ step ^ (t + 1) := by
    have h1 : (1 : ℝ) ≤ step := by linarith
    calc 1 + rho ≤ step := hstep
      _ = step ^ 1 := (pow_one _).symm
      _ ≤ step ^ (t + 1) := pow_le_pow_right₀ h1 (by omega)
  have h0 : 0 ≤ 1 + psi := by linarith
  calc (1 + rho) * (1 + psi) ≤ step ^ (t + 1) * (1 + psi) := mul_le_mul_of_nonneg_right hs h0
    _ ≤ (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * step ^ (t + 1) := by
        have : (1 + psi) * 1 ≤ (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) :=
          mul_le_mul_of_nonneg_left he h0
        have hs0 : 0 ≤ step ^ (t + 1) := pow_nonneg (by linarith) _
        nlinarith

theorem ffac_one_le {R : RowCtx} {psi step : ℝ} {t : ℕ} {m : ℝ} (hpsi : 0 ≤ psi) (hstep : 1 ≤ step)
    (hm : 0 ≤ m + R.Y + R.E) : 1 ≤ ffac R psi step t m := by
  unfold ffac
  have he : 1 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 24) := Real.one_le_exp (div_nonneg hm (by norm_num))
  have hs : 1 ≤ step ^ t := one_le_pow₀ hstep
  have := mul_le_mul (by linarith : (1 : ℝ) ≤ 1 + psi) he (by norm_num) (by linarith)
  nlinarith

/-- **The step of `ffac`**: with `m ≤ m'` and `d = (m' − m)/2^24 + 2^-150`,
`(1 + ρ)·(1 + d)·B0·ffac t m ≤ ffac (t+1) m'` when `step = (1 + ρ)(1 + 2^-150)·B0`. -/
theorem ffac_step {R : RowCtx} {psi rho B0 : ℝ} {t : ℕ} {m m' : ℝ} (hpsi : 0 ≤ psi) (hrho : 0 ≤ rho)
    (hB0 : 0 ≤ B0) (hmm : m ≤ m') :
    (1 + rho) * ((1 + ((m' - m) / 2 ^ 24 + 1 / 2 ^ 150)) * B0) *
        ffac R psi ((1 + rho) * (1 + 1 / 2 ^ 150) * B0) t m ≤
      ffac R psi ((1 + rho) * (1 + 1 / 2 ^ 150) * B0) (t + 1) m' := by
  unfold ffac
  have ha : 0 ≤ (m' - m) / 2 ^ 24 := by apply div_nonneg <;> linarith
  have h1 : 1 + ((m' - m) / 2 ^ 24 + 1 / 2 ^ 150) ≤ Real.exp ((m' - m) / 2 ^ 24) * (1 + 1 / 2 ^ 150) := by
    have := Real.add_one_le_exp ((m' - m) / 2 ^ 24)
    nlinarith
  have hexp : Real.exp ((m' - m) / 2 ^ 24) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) =
      Real.exp ((m' + R.Y + R.E) / 2 ^ 24) := by rw [← Real.exp_add]; congr 1; ring
  set s := (1 + rho) * (1 + 1 / 2 ^ 150) * B0 with hs
  have hs0 : 0 ≤ s := by positivity
  have hst : 0 ≤ s ^ t := pow_nonneg hs0 t
  have hE1 : 0 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 24) := (Real.exp_pos _).le
  have hP : 0 ≤ (1 + rho) * B0 * (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * s ^ t := by positivity
  calc (1 + rho) * ((1 + ((m' - m) / 2 ^ 24 + 1 / 2 ^ 150)) * B0) *
        ((1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * s ^ t)
      = (1 + ((m' - m) / 2 ^ 24 + 1 / 2 ^ 150)) *
          ((1 + rho) * B0 * (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * s ^ t) := by ring
    _ ≤ (Real.exp ((m' - m) / 2 ^ 24) * (1 + 1 / 2 ^ 150)) *
          ((1 + rho) * B0 * (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * s ^ t) :=
        mul_le_mul_of_nonneg_right h1 hP
    _ = (1 + psi) * (Real.exp ((m' - m) / 2 ^ 24) * Real.exp ((m + R.Y + R.E) / 2 ^ 24)) *
          (s ^ t * s) := by rw [hs]; ring
    _ = (1 + psi) * Real.exp ((m' + R.Y + R.E) / 2 ^ 24) * s ^ (t + 1) := by rw [hexp, pow_succ s t]

/-- **`ffac ≤ 2`** over the covered loop (`t ≤ 1024` blocks, `|m| ≤ Y + E`). -/
theorem ffac_le_two {R : RowCtx} (hR : R.Ok) {psi step : ℝ} {t : ℕ} {m : ℝ} (hpsi : psi ≤ 1 / 7)
    (hstep1 : 1 ≤ step) (hstep : step ≤ 1 + 1 / 2 ^ 14) (ht : t ≤ 1024) (hm : |m| ≤ R.Y + R.E) :
    ffac R psi step t m ≤ 2 := by
  unfold ffac
  have hY := hR.Y_le
  have hE := hR.E_le
  have hE0 := hR.E_nonneg
  have hm1 : m + R.Y + R.E ≤ 3074 := by have := le_abs_self m; linarith
  have h1 : Real.exp ((m + R.Y + R.E) / 2 ^ 24) ≤ Real.exp (3074 / 2 ^ 24) :=
    Real.exp_le_exp.mpr (by apply div_le_div_of_nonneg_right hm1 (by norm_num))
  have h2 : step ^ t ≤ Real.exp (1024 * (1 / 2 ^ 14)) := by
    have hx : 0 ≤ step - 1 := by linarith
    have := one_add_pow_le_exp hx t
    rw [show 1 + (step - 1) = step by ring] at this
    refine this.trans (Real.exp_le_exp.mpr ?_)
    have : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
    nlinarith
  have h3 : Real.exp (3074 / 2 ^ 24) * Real.exp (1024 * (1 / 2 ^ 14)) ≤ 1 + 2 * (3074 / 2 ^ 24 + 1024 * (1 / 2 ^ 14)) := by
    rw [← Real.exp_add]; exact exp_le_two (by norm_num) (by norm_num)
  have hst : 0 ≤ step ^ t := pow_nonneg (by linarith) t
  have hE1 : 0 ≤ Real.exp ((m + R.Y + R.E) / 2 ^ 24) := (Real.exp_pos _).le
  have h4 : Real.exp ((m + R.Y + R.E) / 2 ^ 24) * step ^ t ≤ Real.exp (3074 / 2 ^ 24) * Real.exp (1024 * (1 / 2 ^ 14)) :=
    mul_le_mul h1 h2 hst (Real.exp_pos _).le
  calc (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * step ^ t
      = (1 + psi) * (Real.exp ((m + R.Y + R.E) / 2 ^ 24) * step ^ t) := by ring
    _ ≤ (1 + 1 / 7) * (1 + 2 * (3074 / 2 ^ 24 + 1024 * (1 / 2 ^ 14))) := by
        apply mul_le_mul (by linarith) (h4.trans h3) (by positivity) (by norm_num)
    _ ≤ 2 := by norm_num

/-! ## Facts about the weights -/

theorem two_rpow_tiny {x : ℝ} (h : x ≤ -2 ^ 19) : (2 : ℝ) ^ x ≤ 1 / 2 ^ 200 := by
  have h1 := two_rpow_le (show x ≤ -((200 : ℕ) : ℝ) by push_cast; linarith)
  rw [Real.rpow_neg (by norm_num), Real.rpow_natCast] at h1
  simpa using h1

theorem two_rpow_le_two {x : ℝ} (h : x ≤ 1) : (2 : ℝ) ^ x ≤ 2 := by
  have := two_rpow_le h; rwa [Real.rpow_one] at this

/-- An attended key's weight against a running maximum that bounds it: at most `2`. -/
theorem wt_le_two {R : RowCtx} (hR : R.Ok) {t : ℕ} {m : F32} (hm : MInv R t m) {j : ℕ} (hj : j < 64 * t)
    (ha : R.al j = true) : wt R m.toReal j ≤ 2 := by
  have := hm.above j hj ha
  have := hR.E_le
  exact two_rpow_le_two (by linarith)

/-- A sum of weighted values over the first `t` blocks, against a bounding maximum. -/
theorem psum_wt_le {R : RowCtx} (hR : R.Ok) {t : ℕ} {m : F32} (hm : MInv R t m) {g : ℕ → ℝ} {B : ℝ}
    (hg0 : ∀ j, 0 ≤ g j) (hgB : ∀ j, g j ≤ B) :
    psum R.al t (fun j => wt R m.toReal j * g j) ≤ 64 * t * (2 * B) := by
  have hB : 0 ≤ B := (hg0 0).trans (hgB 0)
  refine psum_le_card (by positivity) fun j hj ha => ?_
  have h1 := wt_le_two hR hm hj ha
  have h2 := (wt_pos R m.toReal j).le
  exact mul_le_mul h1 (hgB j) (hg0 j) (by norm_num)

/-- The weights after a block: `Σ_{j < 64(t+1)} wt m'·g = 2^(m − m')·Σ_{j < 64t} wt m·g + (block t)`. -/
theorem psum_step (R : RowCtx) (m m' : ℝ) (t : ℕ) (g : ℕ → ℝ) :
    psum R.al (t + 1) (fun j => wt R m' j * g j) =
      (2 : ℝ) ^ (m - m') * psum R.al t (fun j => wt R m j * g j) +
        ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then wt R m' (64 * t + jj) * g (64 * t + jj) else 0) := by
  rw [psum_succ, ← psum_mul]
  congr 1
  unfold psum
  refine Finset.sum_congr rfl fun j _ => ?_
  dsimp only
  rw [wt_rescale R m m' j, mul_assoc]

theorem psum_one (R : RowCtx) (m' : ℝ) (g : ℕ → ℝ) :
    psum R.al (0 + 1) (fun j => wt R m' j * g j) =
      ∑ jj : Fin 64, (if R.al (64 * 0 + jj) = true then wt R m' (64 * 0 + jj) * g (64 * 0 + jj) else 0) := by
  rw [psum_succ, psum_zero, zero_add]

/-! ## Sums over the lanes of a block -/

theorem lane_sum_err {a b c : Fin 64 → ℝ} {k e : ℝ} (h : ∀ i, |a i - b i| ≤ k * c i + e) :
    |∑ i, a i - ∑ i, b i| ≤ k * ∑ i, c i + 64 * e := by
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  calc ∑ i, |a i - b i| ≤ ∑ i, (k * c i + e) := Finset.sum_le_sum fun i _ => h i
    _ = k * ∑ i, c i + 64 * e := by rw [Finset.sum_add_distrib, Finset.mul_sum]; simp

theorem lane_sum_le {a c : Fin 64 → ℝ} {k e : ℝ} (h : ∀ i, a i ≤ k * c i + e) :
    ∑ i, a i ≤ k * ∑ i, c i + 64 * e := by
  calc ∑ i, a i ≤ ∑ i, (k * c i + e) := Finset.sum_le_sum fun i _ => h i
    _ = k * ∑ i, c i + 64 * e := by rw [Finset.sum_add_distrib, Finset.mul_sum]; simp

theorem list_sum_range {n : ℕ} (f : ℕ → ℚ) : ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

/-- A block's lanes as a sequence (`tl.sum`'s leaves). -/
def laneSeq (p : Fin 64 → F32) (j : ℕ) : F32 := if h : j < 64 then p ⟨j, h⟩ else default

theorem sum_laneSeq (p : Fin 64 → F32) (g : F32 → ℚ) :
    ((List.range 64).map fun i => g (laneSeq p i)).sum = ∑ jj : Fin 64, g (p jj) := by
  rw [list_sum_range, Finset.sum_range (fun i => g (laneSeq p i))]
  refine Finset.sum_congr rfl fun jj _ => ?_
  simp [laneSeq, jj.2]

/-! ## One lane: `p = exp2(qk)` (line 82) and `p.to(float16)` (line 101) -/

/-- **One lane's `p`.** An attended key's `p` is within `psiP·G + 2^-126` of its weight `G`
against the new maximum; a masked key's is below `2^-125`. -/
theorem lane_p {P : Platform} (hexp : Rules.Num.Exp2F32 P) {R : RowCtx} (hR : R.Ok) {m' x p : F32}
    {j : ℕ} (hx : x.isFinite = true)
    (hatt : R.al j = true → |x.toReal - (R.c * R.sv j - m'.toReal)| ≤ R.E)
    (hmask : R.al j = false → x.toReal ≤ -2 ^ 19)
    (habove : R.al j = true → R.c * R.sv j ≤ m'.toReal + R.E)
    (hp : P.math .exp2 .binary32 x p) :
    p.isFinite = true ∧
      (R.al j = true → |p.toReal - wt R m'.toReal j| ≤ psiPr R.E * wt R m'.toReal j + 1 / 2 ^ 126) ∧
      (R.al j = false → |p.toReal| ≤ 1 / 2 ^ 125) := by
  have hE0 := hR.E_nonneg
  have hE8 := hR.E_le
  cases hal : R.al j
  · have hxm := hmask hal
    obtain ⟨hpf, hpe⟩ := exp2_acc hexp hp hx (by linarith)
    have ht := two_rpow_tiny hxm
    have hpos := two_rpow_pos x.toReal
    refine ⟨hpf, fun h => absurd h (by simp), fun _ => ?_⟩
    have := abs_sub_abs_le_abs_sub p.toReal ((2 : ℝ) ^ x.toReal)
    rw [abs_of_pos hpos] at this
    norm_num at ht ⊢
    linarith
  · have hxa := hatt hal
    have hab := habove hal
    have hx1 : x.toReal ≤ 100 := by have := le_abs_self (x.toReal - (R.c * R.sv j - m'.toReal)); linarith
    obtain ⟨hpf, hpe⟩ := exp2_acc hexp hp hx hx1
    refine ⟨hpf, fun _ => ?_, fun h => absurd h (by simp)⟩
    set G := wt R m'.toReal j with hG
    have hGp : 0 < G := wt_pos _ _ _
    set δ := x.toReal - (R.c * R.sv j - m'.toReal) with hδ
    have hsplit : (2 : ℝ) ^ x.toReal = (2 : ℝ) ^ δ * G := by
      rw [hG, wt, ← two_rpow_add]; congr 1; rw [hδ]; ring
    have h1 := two_rpow_sub_one (show |δ| ≤ 1 / 2 by linarith)
    have h2 : |(2 : ℝ) ^ x.toReal - G| ≤ R.E * G := by
      rw [hsplit, show (2 : ℝ) ^ δ * G - G = ((2 : ℝ) ^ δ - 1) * G by ring, abs_mul, abs_of_pos hGp]
      exact mul_le_mul_of_nonneg_right (h1.trans hxa) hGp.le
    have h3 : (2 : ℝ) ^ x.toReal ≤ (1 + R.E) * G := by
      have := le_abs_self ((2 : ℝ) ^ x.toReal - G); linarith
    have := abs_sub_le p.toReal ((2 : ℝ) ^ x.toReal) G
    unfold psiPr
    nlinarith

/-- **One lane's `p.to(float16)`.** An attended key's is within `psi16·G + omR` of its weight; a
masked key's is `0`. -/
theorem lane_p16 {R : RowCtx} (hR : R.Ok) {m' p : F32} {q : F16} {j : ℕ} (hpf : p.isFinite = true)
    (hpa : R.al j = true → |p.toReal - wt R m'.toReal j| ≤ psiPr R.E * wt R m'.toReal j + 1 / 2 ^ 126)
    (hpm : R.al j = false → |p.toReal| ≤ 1 / 2 ^ 125) (hG2 : R.al j = true → wt R m'.toReal j ≤ 2)
    (hq : FloatV.NaNEq (FloatV.cvt p) q) :
    q.isFinite = true ∧
      (R.al j = true → |q.toReal - wt R m'.toReal j| ≤ psi16r R.E * wt R m'.toReal j + omR) ∧
      (R.al j = false → q.toRat = 0) := by
  have hE0 := hR.E_nonneg
  have hE8 := hR.E_le
  have hps0 := psiPr_nonneg hE0
  have hps1 := psiPr_le hE8
  cases hal : R.al j
  · have hp := hpm hal
    obtain ⟨hqf, hqe⟩ := f16_cvt hq hpf (by norm_num at hp ⊢; linarith)
    refine ⟨hqf, fun h => absurd h (by simp), fun _ => f16_tiny ?_⟩
    have := abs_sub_abs_le_abs_sub q.toReal p.toReal
    have h3 : |q.toReal| < 1 / 2 ^ 24 := by norm_num at hp hqe ⊢; linarith
    rw [toReal_def] at h3
    have h4 : ((|q.toRat| : ℚ) : ℝ) < ((1 / 2 ^ 24 : ℚ) : ℝ) := by push_cast; exact h3
    exact Rat.cast_lt.mp h4
  · have hp := hpa hal
    have hG := hG2 hal
    have hGp : 0 < wt R m'.toReal j := wt_pos _ _ _
    have hpa' : |p.toReal| ≤ (1 + psiPr R.E) * wt R m'.toReal j + 1 / 2 ^ 126 := by
      have := abs_sub_abs_le_abs_sub p.toReal (wt R m'.toReal j)
      rw [abs_of_pos hGp] at this; linarith
    obtain ⟨hqf, hqe⟩ := f16_cvt hq hpf (by nlinarith)
    refine ⟨hqf, fun _ => ?_, fun h => absurd h (by simp)⟩
    have := abs_sub_le q.toReal p.toReal (wt R m'.toReal j)
    unfold psi16r omR
    nlinarith

/-! ## `alpha` (line 84) -/

/-- The relative error of `alpha`'s input, besides its `2^-22`: `(m' − m)·2^-24 + 2^-150`. -/
noncomputable def dAl (m m' : ℝ) : ℝ := (m' - m) / 2 ^ 24 + 1 / 2 ^ 150

/-- **`alpha` after the first block**: within `((1 + d)(1 + 2^-22) − 1)·2^(m − m') + 2^-126` of
`2^(m − m')`. -/
theorem alpha_bound {P : Platform} (hexp : Rules.Num.Exp2F32 P) {R : RowCtx} (hR : R.Ok) {t : ℕ}
    {m m' dm al : F32} (hin : AlphaIn t m m' dm) (ht : 0 < t) (hm : |m.toReal| ≤ R.Y + R.E)
    (hm' : |m'.toReal| ≤ R.Y + R.E) (ha : P.math .exp2 .binary32 dm al) :
    al.isFinite = true ∧ m.toReal ≤ m'.toReal ∧
      |al.toReal - (2 : ℝ) ^ (m.toReal - m'.toReal)| ≤
        ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) + 1 / 2 ^ 126 := by
  rcases hin with ⟨h0, -⟩ | ⟨-, -, hle, hdf, hde⟩
  · omega
  have hY := hR.Y_le
  have hE := hR.E_le
  have hE0 := hR.E_nonneg
  have hmm : m'.toReal - m.toReal ≤ 2 * (R.Y + R.E) := by
    have := neg_abs_le m.toReal; have := le_abs_self m'.toReal; linarith
  have hd0 : 0 ≤ dAl m.toReal m'.toReal := by
    unfold dAl
    have : 0 ≤ m'.toReal - m.toReal := by linarith
    linarith [div_nonneg this (by norm_num : (0 : ℝ) ≤ 2 ^ 24), (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)]
  have hd1 : dAl m.toReal m'.toReal ≤ 1 / 2 := by unfold dAl; linarith
  have hdm1 : dm.toReal ≤ 100 := by
    have := le_abs_self (dm.toReal - (m.toReal - m'.toReal)); unfold dAl at hd1; linarith
  obtain ⟨haf, hae⟩ := exp2_acc hexp ha hdf hdm1
  refine ⟨haf, hle, ?_⟩
  set lam := (2 : ℝ) ^ (m.toReal - m'.toReal) with hlam
  have hlp : 0 < lam := two_rpow_pos _
  set δ := dm.toReal - (m.toReal - m'.toReal) with hδ
  have hδd : |δ| ≤ dAl m.toReal m'.toReal := hde
  have hsplit : (2 : ℝ) ^ dm.toReal = (2 : ℝ) ^ δ * lam := by
    rw [hlam, ← two_rpow_add]; congr 1; rw [hδ]; ring
  have h1 := two_rpow_sub_one (hδd.trans hd1)
  have h2 : |(2 : ℝ) ^ dm.toReal - lam| ≤ dAl m.toReal m'.toReal * lam := by
    rw [hsplit, show (2 : ℝ) ^ δ * lam - lam = ((2 : ℝ) ^ δ - 1) * lam by ring, abs_mul, abs_of_pos hlp]
    exact mul_le_mul_of_nonneg_right (h1.trans hδd) hlp.le
  have h3 : (2 : ℝ) ^ dm.toReal ≤ (1 + dAl m.toReal m'.toReal) * lam := by
    have := le_abs_self ((2 : ℝ) ^ dm.toReal - lam); linarith
  have := abs_sub_le al.toReal ((2 : ℝ) ^ dm.toReal) lam
  nlinarith

/-- **`alpha` before the first block**: `exp2(-inf) = +0`. -/
theorem alpha_zero {P : Platform} (hni : Rules.Num.Exp2F32NegInf P) {m m' dm al : F32}
    (hin : AlphaIn 0 m m' dm) (ha : P.math .exp2 .binary32 dm al) :
    al.isFinite = true ∧ al.toReal = 0 := by
  rcases hin with ⟨-, rfl⟩ | ⟨h, -⟩
  · rw [← Rules.Num.negInfF32_eq_negInf] at ha
    have := hni al ha
    subst this
    have h := DotLoop.val?_f32.mp DotLoop.zero_val
    exact ⟨h.1, by rw [toReal_def, show (Rules.Num.zeroF32 : F32) = FloatV.ofPat 0 from rfl, h.2]; simp⟩
  · omega

/-- **Rescaling by `alpha`**: the product's error against `λ·a` (`acc * alpha[:, None]`, line 95,
one rounding). -/
theorem scale_err {a α a1 lam d : ℝ} (hlam : 0 ≤ lam)
    (hα : |α - lam| ≤ ((1 + d) * (1 + 1 / 2 ^ 22) - 1) * lam + 1 / 2 ^ 126)
    (ha1 : |a1 - a * α| ≤ |a * α| / 2 ^ 24 + 1 / 2 ^ 150) :
    |a1 - lam * a| ≤ ((1 + d) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) - 1) * (lam * |a|) +
      (|a| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150) := by
  have hA := abs_nonneg a
  have h1 : |a * α - lam * a| = |a| * |α - lam| := by
    rw [show a * α - lam * a = a * (α - lam) by ring, abs_mul]
  have h2 : |a| * |α - lam| ≤ |a| * (((1 + d) * (1 + 1 / 2 ^ 22) - 1) * lam + 1 / 2 ^ 126) :=
    mul_le_mul_of_nonneg_left hα hA
  have h3 : |a * α| ≤ |a| * ((1 + d) * (1 + 1 / 2 ^ 22) * lam + 1 / 2 ^ 126) := by
    rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ hA
    have := abs_sub_abs_le_abs_sub α lam
    rw [abs_of_nonneg hlam] at this; linarith
  have := abs_sub_le a1 (a * α) (lam * a)
  have e : ((1 + d) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) - 1) * (lam * |a|) +
      (|a| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150) =
      |a| * ((1 + d) * (1 + 1 / 2 ^ 22) * lam + 1 / 2 ^ 126) / 2 ^ 24 + 1 / 2 ^ 150 +
        |a| * (((1 + d) * (1 + 1 / 2 ^ 22) - 1) * lam + 1 / 2 ^ 126) := by ring
  rw [e]
  linarith

end Kernels.AttnFwd
