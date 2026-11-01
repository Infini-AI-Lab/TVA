import AttnFwdE5m2.RowMax

/-!
# `attn_fwd_e5m2`: one row, sums over the keys, `p`, `p.to(float8e5)` and `alpha`

* `psum`: a sum over the attended keys of the first `t` blocks; `wt R m j = 2^(w·s_j − m)`, the
  exact weight of key `j` against the running maximum `m`, and `ew R j = 2^(w·s_j)`, its exact
  weight `exp(sm_scale·s_j)`.
* the constants of a step as real numbers (`pER = 2^E`, `psiPR`, `psi8R`, `dmaxR`, `stepLR`,
  `qLR`, …, and the `p·v` `tl.dot`'s `cS8`, `cC8`, `cP8`, `cA8`) and their sizes on a covered row
  (`RowCtx.Ok`);
* `lane_p`, `lane_p8`: one lane's `p = exp2(qk)` (line 82) and `p.to(float8e5)` (line 101);
* `alpha_bound`, `alpha_zero`: `alpha = exp2(m_i − m_ij)` (line 84);
* `scale_err`: `acc * alpha[:, None]` (line 95); `sum64_err`: `tl.sum(p, 1)` (line 85).
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## Sums over the attended keys -/

/-- `Σ_{j < 64t, al j} f j`. -/
noncomputable def psum (al : ℕ → Bool) (t : ℕ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (64 * t), if al j = true then f j else 0

/-- The exact weight of key `j` against the running maximum `m`: `2^(w·s_j − m)`. -/
noncomputable def wt (R : RowCtx) (m : ℝ) (j : ℕ) : ℝ := (2 : ℝ) ^ (R.w * R.sv j - m)

/-- The exact weight of key `j`: `2^(w·s_j)`. -/
noncomputable def ew (R : RowCtx) (j : ℕ) : ℝ := (2 : ℝ) ^ (R.w * R.sv j)

theorem wt_pos (R : RowCtx) (m : ℝ) (j : ℕ) : 0 < wt R m j := two_rpow_pos _

theorem ew_pos (R : RowCtx) (j : ℕ) : 0 < ew R j := two_rpow_pos _

theorem wt_eq (R : RowCtx) (m : ℝ) (j : ℕ) : wt R m j = (2 : ℝ) ^ (-m) * ew R j := by
  unfold wt ew; rw [← two_rpow_add]; congr 1; ring

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

theorem psum_ge_single {al : ℕ → Bool} {t : ℕ} {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) {j : ℕ} (hj : j < 64 * t)
    (ha : al j = true) : f j ≤ psum al t f := by
  unfold psum
  have := Finset.single_le_sum (f := fun i => if al i = true then f i else 0)
    (fun i _ => by split_ifs <;> simp [hf i]) (Finset.mem_range.mpr hj)
  simpa [ha] using this

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

theorem lane_sum_ge {a c : Fin 64 → ℝ} {k e : ℝ} (h : ∀ i, k * c i - e ≤ a i) :
    k * ∑ i, c i - 64 * e ≤ ∑ i, a i := by
  calc k * ∑ i, c i - 64 * e = ∑ i, (k * c i - e) := by rw [Finset.sum_sub_distrib, Finset.mul_sum]; simp
    _ ≤ ∑ i, a i := Finset.sum_le_sum fun i _ => h i

theorem lane_sum_err2 {a b c e : Fin 64 → ℝ} {k : ℝ} (h : ∀ i, |a i - b i| ≤ k * c i + e i) :
    |∑ i, a i - ∑ i, b i| ≤ k * ∑ i, c i + ∑ i, e i := by
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  calc ∑ i, |a i - b i| ≤ ∑ i, (k * c i + e i) := Finset.sum_le_sum fun i _ => h i
    _ = k * ∑ i, c i + ∑ i, e i := by rw [Finset.sum_add_distrib, Finset.mul_sum]

theorem lane_sum_le2 {a c e : Fin 64 → ℝ} {k : ℝ} (h : ∀ i, a i ≤ k * c i + e i) :
    ∑ i, a i ≤ k * ∑ i, c i + ∑ i, e i := by
  calc ∑ i, a i ≤ ∑ i, (k * c i + e i) := Finset.sum_le_sum fun i _ => h i
    _ = k * ∑ i, c i + ∑ i, e i := by rw [Finset.sum_add_distrib, Finset.mul_sum]

theorem list_sum_range {n : ℕ} (f : ℕ → ℚ) : ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

/-- A block's lanes as a sequence (`tl.sum`'s and `tl.max`'s leaves). -/
def laneSeq (p : Fin 64 → F32) (j : ℕ) : F32 := if h : j < 64 then p ⟨j, h⟩ else default

theorem sum_laneSeq (p : Fin 64 → F32) (g : F32 → ℚ) :
    ((List.range 64).map fun i => g (laneSeq p i)).sum = ∑ jj : Fin 64, g (p jj) := by
  rw [list_sum_range, Finset.sum_range (fun i => g (laneSeq p i))]
  refine Finset.sum_congr rfl fun jj _ => ?_
  simp [laneSeq, jj.2]

theorem z24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
theorem z150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num

theorem pow63_le : ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 ≤ 1 + 1 / 2 ^ 17 := by
  have h1 := one_add_pow_le_exp (x := (1 / 2 ^ 24 : ℝ)) (by norm_num) 63
  have h2 := exp_le_two (y := ((63 : ℕ) : ℝ) * (1 / 2 ^ 24)) (by norm_num) (by norm_num)
  have : 1 + 2 * (((63 : ℕ) : ℝ) * (1 / 2 ^ 24)) ≤ 1 + 1 / 2 ^ 17 := by norm_num
  linarith

/-- **`tl.sum` of a block's row of `p`** (line 85): `63` additions in some tree. -/
theorem sum64_err {p : Fin 64 → F32} {r : F32}
    (h : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) r)
    (hf : ∀ jj, (p jj).isFinite = true) (hA : ∑ jj, |(p jj).toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - ∑ jj, (p jj).toReal| ≤
      ((1 + 1 / 2 ^ 24) ^ 63 - 1) * ∑ jj, |(p jj).toReal| + 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24) ^ 63 := by
  obtain ⟨tr, hcov, hev⟩ := h
  have hx : ∀ i, i < 64 → (laneSeq p i).isFinite = true := fun i hi => by simp [laneSeq, hi, hf]
  have hS1 := sum_laneSeq p (fun x => |x.toRat|)
  have hS2 := sum_laneSeq p (fun x => x.toRat)
  have hK : ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 ≤ 2 := pow63_le.trans (by norm_num)
  have hM : ((List.range 64).map fun i => |(laneSeq p i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (64 - 1) (64 - 1)
        ((List.range 64).map fun i => |(laneSeq p i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [hS1]
    have : (((∑ jj, |(p jj).toRat|) + Tree.errBound Format.binary32.u Format.binary32.eta (64 - 1) (64 - 1)
        (∑ jj, |(p jj).toRat|) : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; unfold Tree.errBound; rw [u32_eq, eta32_eq]
      push_cast
      rw [z24, z150]
      simp only [← toReal_def]
      have hA0 : 0 ≤ ∑ jj, |(p jj).toReal| := Finset.sum_nonneg fun _ _ => abs_nonneg _
      generalize ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 = K at hK ⊢
      norm_num
      nlinarith
    exact_mod_cast this
  obtain ⟨hrf, herr⟩ := sum_err_covers Format.binary32_regular (laneSeq p) hcov hev hx hM
  refine ⟨hrf, ?_⟩
  rw [hS1, hS2] at herr
  unfold Tree.errBound at herr
  rw [u32_eq, eta32_eq] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  rw [z24, z150] at this
  simp only [← toReal_def] at this
  linarith

/-! ## The constants of a step, as real numbers -/

/-- `2^E`: the factor by which a `p` may differ from its exact weight. -/
noncomputable def pER (R : RowCtx) : ℝ := (2 : ℝ) ^ R.E

/-- `p`'s relative error (upwards): `2^E·(1 + 2^-22) − 1`. -/
noncomputable def psiPR (R : RowCtx) : ℝ := pER R * (1 + 1 / 2 ^ 22) - 1

/-- The same after `p.to(float8e5)`. -/
noncomputable def psi8R (R : RowCtx) : ℝ := (1 + psiPR R) * (9 / 8) - 1

/-- The absolute error of a key's `p.to(float8e5)`. -/
noncomputable def om8r : ℝ := 1 / 2 ^ 17 + 9 / 8 * (1 / 2 ^ 126)

/-- The bound on `alpha`'s relative error besides `2^-22`. -/
noncomputable def dmaxR (R : RowCtx) : ℝ := 2 * (R.Y + R.E) / 2 ^ 24 + 1 / 2 ^ 150

/-- A block sum's relative error. -/
noncomputable def gamr : ℝ := (1 + 1 / 2 ^ 24) ^ 63 - 1

/-- `l_i`'s per-block factor from above. -/
noncomputable def stepLR (R : RowCtx) : ℝ := (1 + 3 / 2 ^ 24) * ((1 + dmaxR R) * (1 + 1 / 2 ^ 22))

/-- `l_i`'s per-block factor from below. -/
noncomputable def qLR (R : RowCtx) : ℝ := (1 - 3 / 2 ^ 24) * (2 - (1 + dmaxR R) * (1 + 1 / 2 ^ 22))

/-- `l_i` after `t` blocks, from above (relative). -/
noncomputable def FhR (R : RowCtx) (t : ℕ) : ℝ := (1 + gamr) * (1 + psiPR R) * stepLR R ^ t

/-- `l_i` after `t` blocks, from below (relative). -/
noncomputable def FlR (R : RowCtx) (t : ℕ) : ℝ :=
  (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) / pER R * qLR R ^ t

/-- `l_i`'s absolute error per block (before the factor `FhR`). -/
noncomputable def absLr : ℝ := 1 / 2 ^ 80

/-- The accumulator's crude bound after `t` blocks. -/
noncomputable def crA (t : ℕ) : ℝ := 2 ^ 23 * t * (61 / 60) ^ t

theorem pER_ge {R : RowCtx} (hR : R.Ok) : 1 ≤ pER R := one_le_two_rpow hR.E_nonneg

theorem pER_le {R : RowCtx} (hR : R.Ok) : pER R ≤ 2 ^ 13 := by
  have := two_rpow_le (show R.E ≤ ((13 : ℕ) : ℝ) by push_cast; exact hR.E_le)
  rwa [two_rpow_nat] at this

theorem psiPR_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ psiPR R := by
  unfold psiPR; have := pER_ge hR; nlinarith

theorem psi8R_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ psi8R R := by
  unfold psi8R; have := psiPR_nonneg hR; nlinarith

theorem dmaxR_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ dmaxR R := by
  unfold dmaxR; have := hR.Y_nonneg; have := hR.E_nonneg; positivity

theorem dmaxR_le {R : RowCtx} (hR : R.Ok) : dmaxR R ≤ 1 / 2 ^ 13 := by
  unfold dmaxR; have := hR.Y_le; have := hR.E_le; norm_num; linarith

/-- The `p·v` `tl.dot`'s constants as real numbers. -/
theorem cS8_r : ((cS8 : ℚ) : ℝ) = 271425 / 2 ^ 24 := by
  unfold cS8 Rules.Num.rhoChunk8 Rules.Num.sigmaChunk8; push_cast; norm_num

theorem cC8_r : ((cC8 : ℚ) : ℝ) = 139297 / 2 ^ 24 := by
  unfold cC8 Rules.Num.rhoChunk8 Rules.Num.sigmaChunk8; push_cast; norm_num

theorem cP8_r : ((cP8 : ℚ) : ℝ) = 132128 / 2 ^ 24 := by
  unfold cP8 Rules.Num.rhoChunk8 Rules.Num.sigmaChunk8; push_cast; norm_num

theorem cA8_r : ((cA8 : ℚ) : ℝ) = 8225 / 2 ^ 12 * (1 / 2 ^ 126) := by
  unfold cA8 Rules.Num.rhoChunk8 Rules.Num.sigmaChunk8 Rules.Num.alphaChunk8; push_cast; norm_num

/-- The loss arm's constants as real numbers. -/
theorem lS8_r : ((lS8 : ℚ) : ℝ) = 4097 / 2 ^ 22 := by
  unfold lS8 Rules.Num.rhoLossChunk8 Rules.Num.sigmaChunk8; push_cast; norm_num

theorem lC8_r : ((lC8 : ℚ) : ℝ) = 16791553 / 2 ^ 23 := by
  unfold lC8 Rules.Num.rhoLossChunk8 Rules.Num.sigmaChunk8 Rules.Num.kappaChunk8; push_cast; norm_num

theorem lP8_r : ((lP8 : ℚ) : ℝ) = 2049 / 2 ^ 23 := by
  unfold lP8 Rules.Num.rhoLossChunk8 Rules.Num.sigmaChunk8; push_cast; norm_num

theorem lA8_r : ((lA8 : ℚ) : ℝ) = 4097 / 2 ^ 11 * (1 / 2 ^ 126) := by
  unfold lA8 Rules.Num.rhoLossChunk8 Rules.Num.sigmaChunk8 Rules.Num.alphaChunk8; push_cast; norm_num

theorem eps8_r : ((eps8 : ℚ) : ℝ) = 1 / 2 ^ 17 + 1 / 2 ^ 125 := by unfold eps8; push_cast; ring

theorem gam63_r : ((gam63 : ℚ) : ℝ) = gamr := by unfold gam63 gamr u32; push_cast; ring

/-- `(1 + cS8)·Φ ≤ 61/60`: the growth of the accumulator's crude bound. -/
theorem crA_factor {R : RowCtx} (hR : R.Ok) :
    (1 + (cS8 : ℝ)) * ((1 + dmaxR R) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24))) ≤ 61 / 60 := by
  rw [cS8_r]; have := dmaxR_le hR; have := dmaxR_nonneg hR; nlinarith

theorem gamr_nonneg : 0 ≤ gamr := by
  unfold gamr; have : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ 63 := one_le_pow₀ (by norm_num); linarith

theorem gamr_le : gamr ≤ 1 / 2 ^ 17 := by unfold gamr; have := pow63_le; linarith

theorem stepLR_ge {R : RowCtx} (hR : R.Ok) : 1 + 3 / 2 ^ 24 ≤ stepLR R := by
  unfold stepLR; have := dmaxR_nonneg hR
  have : (1 : ℝ) ≤ (1 + dmaxR R) * (1 + 1 / 2 ^ 22) := by nlinarith
  nlinarith

theorem stepLR_le {R : RowCtx} (hR : R.Ok) : stepLR R ≤ 1 + 1 / 2 ^ 12 := by
  unfold stepLR; have := dmaxR_le hR; have := dmaxR_nonneg hR; nlinarith

theorem qLR_le {R : RowCtx} (hR : R.Ok) : qLR R ≤ 1 := by
  unfold qLR; have := dmaxR_nonneg hR
  have : (1 : ℝ) ≤ (1 + dmaxR R) * (1 + 1 / 2 ^ 22) := by nlinarith
  nlinarith

theorem qLR_ge {R : RowCtx} (hR : R.Ok) : 1 - 1 / 2 ^ 12 ≤ qLR R := by
  unfold qLR; have := dmaxR_le hR; have := dmaxR_nonneg hR; nlinarith

theorem crA_succ (t : ℕ) : (61 / 60 : ℝ) * crA t + 2 ^ 23 ≤ crA (t + 1) := by
  unfold crA
  have h1 : (1 : ℝ) ≤ (61 / 60) ^ (t + 1) := one_le_pow₀ (by norm_num)
  have e : (2 : ℝ) ^ 23 * ((t + 1 : ℕ) : ℝ) * (61 / 60) ^ (t + 1) =
      61 / 60 * (2 ^ 23 * t * (61 / 60) ^ t) + 2 ^ 23 * (61 / 60) ^ (t + 1) := by
    push_cast; rw [pow_succ]; ring
  rw [e]
  nlinarith

theorem crA_le {t : ℕ} (ht : t ≤ 1024) : crA t ≤ 2 ^ 58 := by
  unfold crA
  have h1 := pow6160_le ht
  have h2 : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have h0 : (0 : ℝ) ≤ (61 / 60) ^ t := by positivity
  have := mul_le_mul h2 h1 h0 (by norm_num)
  nlinarith

theorem crA_nonneg (t : ℕ) : 0 ≤ crA t := by unfold crA; positivity

/-! ## Facts about the weights -/

/-- An attended key's weight against a running maximum that bounds it: at most `2^E`. -/
theorem wt_le {R : RowCtx} {t : ℕ} {m : F32} (hm : MInv R t m) {j : ℕ} (hj : j < 64 * t)
    (ha : R.al j = true) : wt R m.toReal j ≤ pER R := by
  have := hm.above j hj ha
  exact two_rpow_le (by linarith)

/-- The sum of a row's weights against its maximum is at least `2^-E`. -/
theorem psum_wt_ge {R : RowCtx} {t : ℕ} {m : F32} (hm : MInv R t m) :
    1 / pER R ≤ psum R.al t (wt R m.toReal) := by
  obtain ⟨j, hj, ha, hd⟩ := hm.attained
  have h1 := psum_ge_single (f := wt R m.toReal) (fun i => (wt_pos _ _ _).le) hj ha
  have h2 : (2 : ℝ) ^ (-R.E) ≤ wt R m.toReal j := by
    unfold wt; apply two_rpow_le
    have := le_abs_self (m.toReal - R.w * R.sv j); linarith
  rw [two_rpow_neg] at h2
  unfold pER; linarith

/-- The sum of a row's weights over `t` blocks against a maximum that bounds them: at most
`64t·2^E`. -/
theorem psum_wt_le {R : RowCtx} (hR : R.Ok) {t : ℕ} {m : F32} (hm : MInv R t m) :
    psum R.al t (wt R m.toReal) ≤ 64 * t * pER R :=
  psum_le_card ((zero_le_one).trans (pER_ge hR)) fun _ hj ha => wt_le hm hj ha

/-! ## One lane: `p = exp2(qk)` (line 82) and `p.to(float8e5)` (line 101) -/

theorem two_rpow_tiny {x : ℝ} (h : x ≤ -2 ^ 19) : (2 : ℝ) ^ x ≤ 1 / 2 ^ 200 := by
  have h1 := two_rpow_le (show x ≤ -((200 : ℕ) : ℝ) by push_cast; linarith)
  rw [Real.rpow_neg (by norm_num), Real.rpow_natCast] at h1
  simpa using h1

/-- `2^x ≤ 1 + 2^-10` for `x ≤ 2^-10`. -/
theorem two_rpow_small {x : ℝ} (h : x ≤ 1 / 2 ^ 10) : (2 : ℝ) ^ x ≤ 1 + 1 / 2 ^ 10 := by
  have h1 := two_rpow_le h
  have h2 := two_rpow_sub_one (δ := 1 / 2 ^ 10) (by rw [abs_of_pos (by norm_num)]; norm_num)
  rw [abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2 ^ 10)] at h2
  have := le_abs_self ((2 : ℝ) ^ (1 / 2 ^ 10 : ℝ) - 1)
  linarith

/-- `2 ≤ q(1 + u) + (1 − u)/q` for `q ≥ 1`, `0 ≤ u ≤ 1`. -/
theorem amgm_aux {q u : ℝ} (hq : 1 ≤ q) (hu : 0 ≤ u) : 1 - (1 - u) / q ≤ q * (1 + u) - 1 := by
  have hq0 : 0 < q := by linarith
  rw [div_eq_mul_inv]
  have hinv : q⁻¹ * q = 1 := inv_mul_cancel₀ hq0.ne'
  have hi0 : 0 < q⁻¹ := inv_pos.mpr hq0
  have hi1 : q⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hq
  nlinarith [mul_nonneg (sub_nonneg.mpr hq) (sub_nonneg.mpr hq), mul_nonneg hu (sub_nonneg.mpr hq),
    mul_nonneg hu (sub_nonneg.mpr hi1)]

/-- **One lane's `p`.** Every `p` is finite and at most `1 + 2^-9`; an attended key's is within
`psiP·G + 2^-126` of its weight `G` against the new maximum, and at least `G(1 − 2^-22)/2^E −
2^-126`; a masked key's is below `2^-125`. -/
theorem lane_p {P : Platform} (hexp : Rules.Num.Exp2F32 P) {R : RowCtx} (hR : R.Ok) {m' x p : F32}
    {j : ℕ} (hx : x.isFinite = true) (hxle : x.toReal ≤ 1 / 2 ^ 10)
    (hatt : R.al j = true → |x.toReal - (R.w * R.sv j - m'.toReal)| ≤ R.E)
    (hmask : R.al j = false → x.toReal ≤ -2 ^ 19)
    (hp : P.math .exp2 .binary32 x p) :
    p.isFinite = true ∧ |p.toReal| ≤ 1 + 1 / 2 ^ 9 ∧
      (R.al j = true → |p.toReal - wt R m'.toReal j| ≤ psiPR R * wt R m'.toReal j + 1 / 2 ^ 126 ∧
        wt R m'.toReal j * (1 - 1 / 2 ^ 22) / pER R - 1 / 2 ^ 126 ≤ p.toReal) ∧
      (R.al j = false → |p.toReal| ≤ 1 / 2 ^ 125) := by
  obtain ⟨hpf, hpe⟩ := exp2_acc hexp hp hx (by linarith)
  have hpos := two_rpow_pos x.toReal
  have hsm := two_rpow_small hxle
  have hpb : |p.toReal| ≤ 1 + 1 / 2 ^ 9 := by
    rw [abs_le]; constructor
    · have := neg_abs_le (p.toReal - (2 : ℝ) ^ x.toReal); nlinarith
    · have := le_abs_self (p.toReal - (2 : ℝ) ^ x.toReal); nlinarith
  refine ⟨hpf, hpb, fun hal => ?_, fun hal => ?_⟩
  · have hxa := hatt hal
    set G := wt R m'.toReal j with hG
    have hGp : 0 < G := wt_pos _ _ _
    set δ := x.toReal - (R.w * R.sv j - m'.toReal) with hδ
    have hsplit : (2 : ℝ) ^ x.toReal = (2 : ℝ) ^ δ * G := by
      rw [hG, wt, ← two_rpow_add]; congr 1; rw [hδ]; ring
    obtain ⟨h1, h2⟩ := two_rpow_mem hxa
    have hq := pER_ge hR
    have hq0 : 0 < pER R := by linarith
    have hup : (2 : ℝ) ^ x.toReal ≤ pER R * G := by
      rw [hsplit]; unfold pER; exact mul_le_mul_of_nonneg_right h2 hGp.le
    have hlo : G / pER R ≤ (2 : ℝ) ^ x.toReal := by
      rw [hsplit]; unfold pER
      rw [div_eq_mul_one_div, mul_comm G]
      exact mul_le_mul_of_nonneg_right h1 hGp.le
    have hp1 := le_abs_self (p.toReal - (2 : ℝ) ^ x.toReal)
    have hp2 := neg_abs_le (p.toReal - (2 : ℝ) ^ x.toReal)
    have hlow : G * (1 - 1 / 2 ^ 22) / pER R - 1 / 2 ^ 126 ≤ p.toReal := by
      have : G * (1 - 1 / 2 ^ 22) / pER R = G / pER R * (1 - 1 / 2 ^ 22) := by ring
      rw [this]; nlinarith
    refine ⟨?_, hlow⟩
    rw [abs_le]; constructor
    · have ham := amgm_aux hq (show (0 : ℝ) ≤ 1 / 2 ^ 22 by norm_num)
      have hk : G * (1 - 1 / 2 ^ 22) / pER R = G * ((1 - 1 / 2 ^ 22) / pER R) := by ring
      rw [hk] at hlow
      unfold psiPR
      nlinarith
    · unfold psiPR; nlinarith
  · have hxm := hmask hal
    have ht := two_rpow_tiny hxm
    have := abs_sub_abs_le_abs_sub p.toReal ((2 : ℝ) ^ x.toReal)
    rw [abs_of_pos hpos] at this
    norm_num at ht ⊢
    linarith

/-- **One lane's `p.to(float8e5)`** (`Rules.Num.CvtSatF32E5m2`). An attended key's is within
`psi8·G + om8` of its weight; a masked key's is `0`; every one is finite and at most `2`. -/
theorem lane_p8 {P : Platform} (hcvt : Rules.Num.CvtSatF32E5m2 P) {R : RowCtx}
    {m' p : F32} {q : F8E5} {j : ℕ} (hpf : p.isFinite = true) (hpb : |p.toReal| ≤ 1 + 1 / 2 ^ 9)
    (hpa : R.al j = true → |p.toReal - wt R m'.toReal j| ≤ psiPR R * wt R m'.toReal j + 1 / 2 ^ 126)
    (hpm : R.al j = false → |p.toReal| ≤ 1 / 2 ^ 125)
    (hq : P.cvt Format.binary32 Format.e5m2 .rn p q) :
    q.isFinite = true ∧ |q.toReal| ≤ 2 ∧
      (R.al j = true → |q.toReal - wt R m'.toReal j| ≤ psi8R R * wt R m'.toReal j + om8r) ∧
      (R.al j = false → q.toRat = 0) := by
  obtain ⟨hqf, hqe⟩ := f8_cvt hcvt hq hpf (by linarith)
  have hqb : |q.toReal| ≤ 2 := by
    have := abs_sub_abs_le_abs_sub q.toReal p.toReal; norm_num at hqe hpb ⊢; linarith
  refine ⟨hqf, hqb, fun hal => ?_, fun hal => ?_⟩
  · have h1 := hpa hal
    have hGp := wt_pos R m'.toReal j
    have hpa' : |p.toReal| ≤ (1 + psiPR R) * wt R m'.toReal j + 1 / 2 ^ 126 := by
      have := abs_sub_abs_le_abs_sub p.toReal (wt R m'.toReal j)
      rw [abs_of_pos hGp] at this; linarith
    have := abs_sub_le q.toReal p.toReal (wt R m'.toReal j)
    unfold psi8R om8r
    nlinarith
  · have hp := hpm hal
    refine f8_tiny ?_
    have := abs_sub_abs_le_abs_sub q.toReal p.toReal
    have h3 : |q.toReal| < 1 / 2 ^ 16 := by norm_num at hp hqe ⊢; linarith
    rw [toReal_def] at h3
    have h4 : ((|q.toRat| : ℚ) : ℝ) < ((1 / 2 ^ 16 : ℚ) : ℝ) := by push_cast; exact h3
    exact Rat.cast_lt.mp h4

/-! ## `alpha` (line 84) -/

/-- The relative error of `alpha`'s input, besides its `2^-22`: `(m' − m)·2^-24 + 2^-150`. -/
noncomputable def dAl (m m' : ℝ) : ℝ := (m' - m) / 2 ^ 24 + 1 / 2 ^ 150

theorem dAl_nonneg {m m' : ℝ} (h : m ≤ m') : 0 ≤ dAl m m' := by
  unfold dAl; have : 0 ≤ m' - m := by linarith
  linarith [div_nonneg this (by norm_num : (0 : ℝ) ≤ 2 ^ 24), (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)]

theorem dAl_le {R : RowCtx} {m m' : ℝ} (hm : |m| ≤ R.Y + R.E) (hm' : |m'| ≤ R.Y + R.E) :
    dAl m m' ≤ dmaxR R := by
  unfold dAl dmaxR
  have := neg_abs_le m; have := le_abs_self m'
  have : m' - m ≤ 2 * (R.Y + R.E) := by linarith
  have := div_le_div_of_nonneg_right this (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
  linarith

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
  have hd0 := dAl_nonneg hle
  have hd1 : dAl m.toReal m'.toReal ≤ 1 / 2 := (dAl_le hm hm').trans ((dmaxR_le hR).trans (by norm_num))
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

/-- `alpha` is finite, at most `2` in magnitude; after the first block it is within
`((1 + dmax)(1 + 2^-22) − 1)·λ + 2^-126` of `λ = 2^(m − m')`. -/
theorem alpha_facts {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} {m m' dm al : F32} (hM' : MInv R (t + 1) m')
    (hin : AlphaIn t m m' dm) (ha : P.math .exp2 .binary32 dm al) (hm : 0 < t → MInv R t m) :
    al.isFinite = true ∧ |al.toReal| ≤ 2 ∧ (t = 0 → al.toReal = 0) ∧
      (0 < t → m.toReal ≤ m'.toReal ∧
        |al.toReal - (2 : ℝ) ^ (m.toReal - m'.toReal)| ≤
          ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) + 1 / 2 ^ 126) := by
  rcases Nat.eq_zero_or_pos t with h0 | ht0
  · subst h0
    have := alpha_zero hni hin ha
    exact ⟨this.1, by rw [this.2]; norm_num, fun _ => this.2, fun h => absurd h (lt_irrefl 0)⟩
  · have hmi := hm ht0
    obtain ⟨haf, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
    have hl0 := two_rpow_pos (m.toReal - m'.toReal)
    have hd := dAl_le hmi.bound hM'.bound
    have hd0 := dAl_nonneg hle
    have hdm := dmaxR_le hR
    have hae' : |al.toReal - (2 : ℝ) ^ (m.toReal - m'.toReal)| ≤
        ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) + 1 / 2 ^ 126 := by
      have : ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) ≤
          ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) :=
        mul_le_mul_of_nonneg_right (by nlinarith) hl0.le
      linarith
    refine ⟨haf, ?_, fun h => absurd h (by omega), fun _ => ⟨hle, hae'⟩⟩
    have := abs_sub_abs_le_abs_sub al.toReal ((2 : ℝ) ^ (m.toReal - m'.toReal))
    rw [abs_of_pos hl0] at this
    have k : ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 / 2 := by
      have : (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 ≤ 1 / 2 := by nlinarith
      have h0 : 0 ≤ (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 := by nlinarith [dmaxR_nonneg hR]
      nlinarith
    linarith

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

end Kernels.AttnFwdE5m2
