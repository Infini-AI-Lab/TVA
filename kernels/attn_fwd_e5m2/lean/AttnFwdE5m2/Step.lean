import AttnFwdE5m2.RowAcc

/-!
# `attn_fwd_e5m2`: one row, one block of the loop for `l_i` and `acc`

* `LInv`: after `t ≥ 1` blocks with running maximum `m`, `l_i` lies between `FlR t·W − FhR t·t·absL`
  and `FhR t·(W + t·absL)`, `W = Σ wt` over the attended keys so far: a two-sided relative bound,
  since each `p` may be off its exact weight by a factor up to `2^±E`;
* `AInv`: an element of the row's `acc` is within `2^-m·ES + EB` of `2^-m·Σ_j ew_j·v_j`, the exact
  weighted sum of the attended keys so far, for the bounds `ES` (in units of the exact weights
  `ew = exp(sm_scale·s)`) and `EB` (absolute) the recursion of `Model.errS` and `Model.errB` gives;
* `lstep` (lines 82–85, 106) and `astep` (lines 82–84, 95, 98–103): one block keeps them.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The factors of `l_i` -/

theorem FhR_succ (R : RowCtx) (t : ℕ) : FhR R (t + 1) = FhR R t * stepLR R := by
  unfold FhR; rw [pow_succ]; ring

theorem FlR_succ (R : RowCtx) (t : ℕ) : FlR R (t + 1) = FlR R t * qLR R := by
  unfold FlR; rw [pow_succ]; ring

theorem stepLR_one {R : RowCtx} (hR : R.Ok) : 1 ≤ stepLR R := le_trans (by norm_num) (stepLR_ge hR)

theorem FhR_ge {R : RowCtx} (hR : R.Ok) (t : ℕ) : 1 ≤ FhR R t := by
  unfold FhR
  have h1 := gamr_nonneg
  have h2 := psiPR_nonneg hR
  have h3 : 1 ≤ stepLR R ^ t := one_le_pow₀ (stepLR_one hR)
  have : 1 ≤ (1 + gamr) * (1 + psiPR R) := by nlinarith
  nlinarith

/-- `(1 + gam)(1 + psiP) ≤ FhR t`. -/
theorem FhR_ge_hiN {R : RowCtx} (hR : R.Ok) (t : ℕ) : (1 + gamr) * (1 + psiPR R) ≤ FhR R t := by
  unfold FhR
  have h1 := gamr_nonneg
  have h2 := psiPR_nonneg hR
  have h3 : 1 ≤ stepLR R ^ t := one_le_pow₀ (stepLR_one hR)
  have : 0 ≤ (1 + gamr) * (1 + psiPR R) := by positivity
  nlinarith

theorem stepLR_pow_le {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) : stepLR R ^ t ≤ 3 / 2 := by
  have := pow_le_of_step (c := 1 / 2 ^ 12) (stepLR_one hR) (stepLR_le hR) ht (by norm_num)
  norm_num at this ⊢; linarith

theorem FhR_le {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) : FhR R t ≤ 2 ^ 14 := by
  unfold FhR psiPR
  have h1 := gamr_le
  have h0 := gamr_nonneg
  have h2 := pER_le hR
  have h2' := pER_ge hR
  have h3 := stepLR_pow_le hR ht
  have h3' : 0 ≤ stepLR R ^ t := pow_nonneg (le_trans zero_le_one (stepLR_one hR)) t
  have k1 : (1 + gamr) * (1 + (pER R * (1 + 1 / 2 ^ 22) - 1)) ≤ (1 + 1 / 2 ^ 17) * (2 ^ 13 * (1 + 1 / 2 ^ 22)) := by
    have : 1 + (pER R * (1 + 1 / 2 ^ 22) - 1) ≤ 2 ^ 13 * (1 + 1 / 2 ^ 22) := by nlinarith
    exact mul_le_mul (by linarith) this (by nlinarith) (by norm_num)
  calc (1 + gamr) * (1 + (pER R * (1 + 1 / 2 ^ 22) - 1)) * stepLR R ^ t
      ≤ (1 + 1 / 2 ^ 17) * (2 ^ 13 * (1 + 1 / 2 ^ 22)) * (3 / 2) := mul_le_mul k1 h3 h3' (by positivity)
    _ ≤ 2 ^ 14 := by norm_num

theorem qLR_nonneg {R : RowCtx} (hR : R.Ok) : 0 ≤ qLR R := le_trans (by norm_num) (qLR_ge hR)

theorem qLR_pow_ge {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) : 3 / 4 ≤ qLR R ^ t := by
  have := pow_ge_of_step (c := 1 / 2 ^ 12) (qLR_ge hR) (by norm_num) (by norm_num) ht
  norm_num at this ⊢; linarith

theorem FlR_eq (R : RowCtx) (t : ℕ) :
    FlR R t = (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) * qLR R ^ t / pER R := by
  unfold FlR; ring

theorem FlR_nonneg {R : RowCtx} (hR : R.Ok) (t : ℕ) : 0 ≤ FlR R t := by
  rw [FlR_eq]
  have := gamr_le
  have := pER_ge hR
  have : 0 ≤ qLR R ^ t := pow_nonneg (qLR_nonneg hR) t
  apply div_nonneg _ (by linarith)
  apply mul_nonneg _ this
  apply mul_nonneg (mul_nonneg (by norm_num) (by linarith)) (by norm_num)

theorem FlR_le_one {R : RowCtx} (hR : R.Ok) (t : ℕ) : FlR R t ≤ 1 := by
  rw [FlR_eq]
  have := gamr_nonneg
  have := gamr_le
  have hp := pER_ge hR
  have h1 : qLR R ^ t ≤ 1 := pow_le_one₀ (qLR_nonneg hR) (qLR_le hR)
  have h0 : 0 ≤ qLR R ^ t := pow_nonneg (qLR_nonneg hR) t
  rw [div_le_iff₀ (by linarith)]
  have : (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) ≤ 1 := by nlinarith
  have : 0 ≤ (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) := by nlinarith
  nlinarith

theorem FlR_ge {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) : 1 / 2 / pER R ≤ FlR R t := by
  rw [FlR_eq]
  have hq := qLR_pow_ge hR ht
  have hp := pER_ge hR
  have hg := gamr_le
  apply div_le_div_of_nonneg_right _ (by linarith)
  have : (1 / 2 : ℝ) ≤ (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) * (3 / 4) := by nlinarith
  have h0 : 0 ≤ (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) := by nlinarith
  nlinarith

/-- `FlR (t + 1)` against the first block's factor. -/
theorem FlR_qL {R : RowCtx} (hR : R.Ok) (t : ℕ) :
    FlR R t * qLR R ≤ (1 - 3 / 2 ^ 24) * ((1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R)) := by
  rw [← FlR_succ, FlR_eq]
  have hp := pER_ge hR
  have hg := gamr_le
  have h1 : qLR R ^ (t + 1) ≤ 1 := pow_le_one₀ (qLR_nonneg hR) (qLR_le hR)
  rw [div_le_iff₀ (by linarith)]
  have e : (1 - 3 / 2 ^ 24) * ((1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R)) * pER R =
      (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) := by field_simp
  rw [e]
  have : 0 ≤ (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) := by nlinarith
  nlinarith

/-! ## `l_i` -/

/-- `l_i` after `t ≥ 1` blocks with running maximum `m`. -/
def LInv (R : RowCtx) (t : ℕ) (m l : F32) : Prop :=
  l.isFinite = true ∧ 0 < l.toReal ∧
    l.toReal ≤ FhR R t * (psum R.al t (wt R m.toReal) + t * absLr) ∧
    FlR R t * psum R.al t (wt R m.toReal) - FhR R t * (t * absLr) ≤ l.toReal

theorem LInv.bound {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) {m l : F32} (hm : MInv R t m)
    (hl : LInv R t m l) : |l.toReal| ≤ 2 ^ 44 := by
  obtain ⟨-, hl0, hlu, -⟩ := hl
  rw [abs_of_pos hl0]
  have hW := psum_wt_le hR hm
  have hp := pER_le hR
  have hF := FhR_le hR ht
  have htr : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have hW' : psum R.al t (wt R m.toReal) + t * absLr ≤ 2 ^ 30 := by
    unfold absLr
    have : 64 * (t : ℝ) * pER R ≤ 64 * 1024 * 2 ^ 13 :=
      mul_le_mul (by linarith) hp (by linarith [pER_ge hR]) (by norm_num)
    nlinarith
  have hW0 : 0 ≤ psum R.al t (wt R m.toReal) + t * absLr := by
    have := psum_nonneg (al := R.al) (t := t) (f := wt R m.toReal) fun j => (wt_pos _ _ _).le
    unfold absLr; positivity
  have := mul_le_mul hF hW' hW0 (by norm_num)
  linarith

/-- Lane `jj`'s exact weight in block `t` (`0` for a masked key). -/
noncomputable def bw1 (R : RowCtx) (m' : ℝ) (t : ℕ) (jj : Fin 64) : ℝ :=
  if R.al (64 * t + jj) = true then wt R m' (64 * t + jj) else 0

theorem bw1_nonneg (R : RowCtx) (m' : ℝ) (t : ℕ) (jj : Fin 64) : 0 ≤ bw1 R m' t jj := by
  unfold bw1; split_ifs
  · exact (wt_pos _ _ _).le
  · exact le_rfl

theorem psum_step1 (R : RowCtx) (m m' : ℝ) (t : ℕ) :
    psum R.al (t + 1) (wt R m') = (2 : ℝ) ^ (m - m') * psum R.al t (wt R m) + ∑ jj, bw1 R m' t jj := by
  have := psum_step R m m' t (fun _ => 1)
  simp only [mul_one] at this
  exact this

/-- **A block's `p` and `l_ij`** (lines 82, 85): every lane's `p` is finite, and `l_ij` lies within
the factors `(1 + gam)(1 + psiP)` and `(1 − gam)(1 − 2^-22)/2^E` of the block's exact weights. -/
theorem block_l {P : Platform} (hexp : Rules.Num.Exp2F32 P) {R : RowCtx} (hR : R.Ok) {t : ℕ}
    {m' lij : F32} {x p : Fin 64 → F32} (hX : LaneX R t m' x)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij) :
    lij.isFinite = true ∧
      lij.toReal ≤ (1 + gamr) * (1 + psiPR R) * ∑ jj, bw1 R m'.toReal t jj + 1 / 2 ^ 118 ∧
      (1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R) * ∑ jj, bw1 R m'.toReal t jj - 1 / 2 ^ 118 ≤ lij.toReal ∧
      |lij.toReal| ≤ 2 ^ 7 := by
  have hq := pER_ge hR
  have hq0 : 0 < pER R := by linarith
  have hps0 := psiPR_nonneg hR
  have hlane : ∀ jj, (p jj).isFinite = true ∧
      |(p jj).toReal| ≤ (1 + psiPR R) * bw1 R m'.toReal t jj + 1 / 2 ^ 125 ∧
      (1 - 1 / 2 ^ 22) / pER R * bw1 R m'.toReal t jj - 1 / 2 ^ 125 ≤ (p jj).toReal ∧
      |(p jj).toReal| ≤ 1 * (p jj).toReal + 2 / 2 ^ 125 ∧
      |(p jj).toReal| ≤ 0 * (p jj).toReal + (1 + 1 / 2 ^ 9) := by
    intro jj
    obtain ⟨hxf, hxle, hxa, hxm⟩ := hX jj
    obtain ⟨hpf, hpb, hpa, hpm⟩ := lane_p hexp hR hxf hxle hxa hxm (hp jj)
    have hlow : -(1 / 2 ^ 125) ≤ (p jj).toReal := by
      cases hal : R.al (64 * t + jj)
      · have := hpm hal; have := neg_abs_le (p jj).toReal; linarith
      · obtain ⟨-, h2⟩ := hpa hal
        have : 0 ≤ wt R m'.toReal (64 * t + jj) * (1 - 1 / 2 ^ 22) / pER R := by
          have := wt_pos R m'.toReal (64 * t + jj); positivity
        linarith
    refine ⟨hpf, ?_, ?_, ?_, by linarith⟩
    · unfold bw1
      cases hal : R.al (64 * t + jj)
      · have := hpm hal; simp only [Bool.false_eq_true, ↓reduceIte, mul_zero, zero_add]; linarith
      · obtain ⟨h1, -⟩ := hpa hal
        have hGp := wt_pos R m'.toReal (64 * t + jj)
        have := abs_sub_abs_le_abs_sub (p jj).toReal (wt R m'.toReal (64 * t + jj))
        rw [abs_of_pos hGp] at this
        simp only [↓reduceIte]; linarith
    · unfold bw1
      cases hal : R.al (64 * t + jj)
      · simp only [Bool.false_eq_true, ↓reduceIte, mul_zero, zero_sub]; linarith
      · obtain ⟨-, h2⟩ := hpa hal
        simp only [↓reduceIte]
        have e : (1 - 1 / 2 ^ 22) / pER R * wt R m'.toReal (64 * t + jj) =
            wt R m'.toReal (64 * t + jj) * (1 - 1 / 2 ^ 22) / pER R := by ring
        rw [e]; linarith
    · rcases le_or_gt 0 (p jj).toReal with h | h
      · rw [abs_of_nonneg h]; linarith
      · rw [abs_of_neg h]; linarith
  have hS1 := lane_sum_le (fun jj => (hlane jj).2.1)
  have hS2 := lane_sum_ge (fun jj => (hlane jj).2.2.1)
  have hS3 := lane_sum_le (fun jj => (hlane jj).2.2.2.1)
  have hS4 := lane_sum_le (fun jj => (hlane jj).2.2.2.2)
  simp only [one_mul, zero_mul, zero_add] at hS3 hS4
  have hB0 : 0 ≤ ∑ jj, bw1 R m'.toReal t jj := Finset.sum_nonneg fun jj _ => bw1_nonneg _ _ _ _
  have hA0 : 0 ≤ ∑ jj, |(p jj).toReal| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hP : ∑ jj, (p jj).toReal ≤ ∑ jj, |(p jj).toReal| := Finset.sum_le_sum fun _ _ => le_abs_self _
  obtain ⟨hlf, hle⟩ := sum64_err hsum (fun jj => (hlane jj).1) (by norm_num at hS4; linarith)
  have hK := pow63_le
  have hK1 : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ 63 := one_le_pow₀ (by norm_num)
  have hgam : (1 + 1 / 2 ^ 24 : ℝ) ^ 63 - 1 = gamr := rfl
  rw [hgam] at hle
  have hK63 : 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 ≤ 1 / 2 ^ 142 := by nlinarith
  have hg0 := gamr_nonneg
  have hg1 := gamr_le
  have hl1 := le_abs_self (lij.toReal - ∑ jj, (p jj).toReal)
  have hl2 := neg_abs_le (lij.toReal - ∑ jj, (p jj).toReal)
  have hgA := mul_le_mul_of_nonneg_left hS1 hg0
  have hgA' := mul_le_mul_of_nonneg_left hS3 hg0
  refine ⟨hlf, ?_, ?_, ?_⟩
  · have : (1 + gamr) * (1 + psiPR R) * ∑ jj, bw1 R m'.toReal t jj =
        (1 + psiPR R) * ∑ jj, bw1 R m'.toReal t jj + gamr * ((1 + psiPR R) * ∑ jj, bw1 R m'.toReal t jj) := by ring
    rw [this]; norm_num at hS1 hgA ⊢; nlinarith
  · have hc0 : 0 ≤ (1 - 1 / 2 ^ 22) / pER R := by positivity
    have k1 := mul_le_mul_of_nonneg_left hS2 (by linarith : (0 : ℝ) ≤ 1 - gamr)
    have e : (1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R) * ∑ jj, bw1 R m'.toReal t jj =
        (1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R * ∑ jj, bw1 R m'.toReal t jj) := by ring
    rw [e]; norm_num at k1 hgA' ⊢; nlinarith
  · have := abs_sub_abs_le_abs_sub lij.toReal (∑ jj, (p jj).toReal)
    have : |∑ jj, (p jj).toReal| ≤ ∑ jj, |(p jj).toReal| := Finset.abs_sum_le_sum_abs _ _
    norm_num at hS4 ⊢
    nlinarith

/-- **One block of `l_i`, in real numbers** (line 106: `l_i·alpha + l_ij`, one contractible
rounding). The old `l` lies in `[Fl·L − Fh·t·absL, Fh·(L + t·absL)]`, `alpha` within
`eps·λ + 2^-126` of `λ`, and `l_ij` in `[loN·B − 2^-118, hiN·B + 2^-118]`; the new `l'` then lies in
`[Fl·q·(λL + B) − Fh·s·(t+1)·absL, Fh·s·(λL + B + (t+1)·absL)]`. -/
theorem l_step_real {l α lij l' L B lam eps Fh Fl s q hiN loN absL t : ℝ}
    (hl0 : 0 < l) (hlu : l ≤ Fh * (L + t * absL)) (hll : Fl * L - Fh * (t * absL) ≤ l) (hlb : l ≤ 2 ^ 44)
    (hL : 0 ≤ L) (hB : 0 ≤ B) (hlam0 : 0 ≤ lam) (hlam1 : lam ≤ 1) (heps0 : 0 ≤ eps) (heps1 : eps ≤ 1)
    (hα : |α - lam| ≤ eps * lam + 1 / 2 ^ 126)
    (hlij1 : lij ≤ hiN * B + 1 / 2 ^ 118) (hlij2 : loN * B - 1 / 2 ^ 118 ≤ lij) (hloN : 0 ≤ loN)
    (hhiN : 0 ≤ hiN)
    (hl' : |l' - (l * α + lij)| ≤ 3 * (|l * α| + |lij|) / 2 ^ 24 + 4 / 2 ^ 150)
    (hs : (1 + 3 / 2 ^ 24) * (1 + eps) ≤ s) (hs1 : 1 + 3 / 2 ^ 24 ≤ s)
    (hq : q ≤ (1 - 3 / 2 ^ 24) * (1 - eps)) (_hq0 : 0 ≤ q)
    (hFh1 : 1 ≤ Fh) (hFh2 : hiN ≤ Fh) (hFl0 : 0 ≤ Fl) (hFlq : Fl * q ≤ (1 - 3 / 2 ^ 24) * loN)
    (habsL : 1 / 2 ^ 80 ≤ absL) (ht0 : 0 ≤ t) :
    l' ≤ Fh * s * (lam * L + B + (t + 1) * absL) ∧
      Fl * q * (lam * L + B) - Fh * s * ((t + 1) * absL) ≤ l' := by
  have hα1 : α ≤ (1 + eps) * lam + 1 / 2 ^ 126 := by have := le_abs_self (α - lam); linarith
  have hα2 : (1 - eps) * lam - 1 / 2 ^ 126 ≤ α := by have := neg_abs_le (α - lam); linarith
  have hel : 0 ≤ (1 - eps) * lam := mul_nonneg (by linarith) hlam0
  have hla1 : l * α ≤ l * ((1 + eps) * lam) + l * (1 / 2 ^ 126) := by
    rw [← mul_add]; exact mul_le_mul_of_nonneg_left hα1 hl0.le
  have hla2 : l * ((1 - eps) * lam) - l * (1 / 2 ^ 126) ≤ l * α := by
    rw [← mul_sub]; exact mul_le_mul_of_nonneg_left hα2 hl0.le
  have hlt : l * (1 / 2 ^ 126) ≤ 1 / 2 ^ 82 := by
    have := mul_le_mul_of_nonneg_right hlb (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 126)
    norm_num at this ⊢; linarith
  have hlt0 : 0 ≤ l * (1 / 2 ^ 126) := by positivity
  have hle0 : 0 ≤ l * ((1 - eps) * lam) := mul_nonneg hl0.le hel
  have habs1 : |l * α| ≤ l * α + 2 * (1 / 2 ^ 82) := by
    have : -(1 / 2 ^ 82) ≤ l * α := by linarith
    rcases le_or_gt 0 (l * α) with h | h
    · rw [abs_of_nonneg h]; linarith
    · rw [abs_of_neg h]; linarith
  have hlij0 : -(1 / 2 ^ 118) ≤ lij := by have := mul_nonneg hloN hB; linarith
  have habs2 : |lij| ≤ lij + 2 * (1 / 2 ^ 118) := by
    rcases le_or_gt 0 lij with h | h
    · rw [abs_of_nonneg h]; linarith
    · rw [abs_of_neg h]; linarith
  set X := l * α + lij with hX
  have hr : 3 * (|l * α| + |lij|) / 2 ^ 24 + 4 / 2 ^ 150 ≤ 3 / 2 ^ 24 * X + 1 / 2 ^ 100 := by
    have h1 : 3 * (|l * α| + |lij|) / 2 ^ 24 ≤ 3 * (X + 2 * (1 / 2 ^ 82) + 2 * (1 / 2 ^ 118)) / 2 ^ 24 :=
      div_le_div_of_nonneg_right (by linarith) (by norm_num)
    have h2 : 3 * (X + 2 * (1 / 2 ^ 82) + 2 * (1 / 2 ^ 118)) / 2 ^ 24 + 4 / 2 ^ 150 ≤
        3 / 2 ^ 24 * X + 1 / 2 ^ 100 := by
      have e : 3 * (X + 2 * (1 / 2 ^ 82) + 2 * (1 / 2 ^ 118)) / 2 ^ 24 = 3 / 2 ^ 24 * X +
          3 * (2 * (1 / 2 ^ 82) + 2 * (1 / 2 ^ 118)) / 2 ^ 24 := by ring
      rw [e]
      have : 3 * (2 * (1 / 2 ^ 82) + 2 * (1 / 2 ^ 118)) / 2 ^ 24 + 4 / 2 ^ 150 ≤ (1 : ℝ) / 2 ^ 100 := by norm_num
      linarith
    linarith
  have hup : l' ≤ (1 + 3 / 2 ^ 24) * X + 1 / 2 ^ 100 := by
    have := le_abs_self (l' - X); linarith
  have hlo : (1 - 3 / 2 ^ 24) * X - 1 / 2 ^ 100 ≤ l' := by
    have := neg_abs_le (l' - X); linarith
  have hA0 : 0 ≤ absL := le_trans (by norm_num) habsL
  have hs0 : 0 ≤ s := by linarith
  have hFs1 : 1 ≤ Fh * s := one_le_mul_of_one_le_of_one_le hFh1 (by linarith)
  have hFsA : 1 / 2 ^ 80 ≤ Fh * s * absL :=
    calc 1 / 2 ^ 80 ≤ 1 * absL := by linarith
      _ ≤ Fh * s * absL := mul_le_mul_of_nonneg_right hFs1 hA0
  have hLt : 0 ≤ L + t * absL := by positivity
  have htA : 0 ≤ Fh * s * (t * absL) := by positivity
  constructor
  · -- from above
    have k1 : l * ((1 + eps) * lam) ≤ (1 + eps) * lam * (Fh * (L + t * absL)) := by
      rw [mul_comm l]; exact mul_le_mul_of_nonneg_left hlu (by positivity)
    have hX1 : X ≤ (1 + eps) * lam * (Fh * (L + t * absL)) + 1 / 2 ^ 82 + hiN * B + 1 / 2 ^ 118 := by
      rw [hX]; linarith
    have k2 : (1 + 3 / 2 ^ 24) * ((1 + eps) * lam * (Fh * (L + t * absL))) ≤ s * lam * (Fh * (L + t * absL)) := by
      have h0 : 0 ≤ lam * (Fh * (L + t * absL)) := mul_nonneg hlam0 (by positivity)
      have := mul_le_mul_of_nonneg_right hs h0
      linarith
    have k3 : s * lam * (Fh * (L + t * absL)) ≤ Fh * s * (lam * L) + Fh * s * (t * absL) := by
      have e : s * lam * (Fh * (L + t * absL)) = Fh * s * (lam * L) + lam * (Fh * s * (t * absL)) := by ring
      rw [e]
      have := mul_le_of_le_one_left htA hlam1
      linarith
    have k4 : (1 + 3 / 2 ^ 24) * (hiN * B) ≤ Fh * s * B := by
      have h1 : (1 + 3 / 2 ^ 24) * hiN ≤ s * Fh := mul_le_mul hs1 hFh2 hhiN hs0
      have := mul_le_mul_of_nonneg_right h1 hB
      linarith
    have k5 : (1 + 3 / 2 ^ 24) * X ≤ (1 + 3 / 2 ^ 24) * ((1 + eps) * lam * (Fh * (L + t * absL))) +
        (1 + 3 / 2 ^ 24) * (hiN * B) + (1 + 3 / 2 ^ 24) * (1 / 2 ^ 82 + 1 / 2 ^ 118) := by
      have := mul_le_mul_of_nonneg_left hX1 (by norm_num : (0 : ℝ) ≤ 1 + 3 / 2 ^ 24)
      linarith
    have k6 : (1 + 3 / 2 ^ 24) * (1 / 2 ^ 82 + 1 / 2 ^ 118) + 1 / 2 ^ 100 ≤ (1 : ℝ) / 2 ^ 80 := by norm_num
    have e : Fh * s * (lam * L + B + (t + 1) * absL) = Fh * s * (lam * L) + Fh * s * (t * absL) +
        Fh * s * B + Fh * s * absL := by ring
    rw [e]
    linarith
  · -- from below
    have k1 : (1 - eps) * lam * (Fl * L - Fh * (t * absL)) ≤ l * ((1 - eps) * lam) := by
      rw [mul_comm l]; exact mul_le_mul_of_nonneg_left hll hel
    have hX1 : (1 - eps) * lam * (Fl * L) - (1 - eps) * lam * (Fh * (t * absL)) - 1 / 2 ^ 82 +
        (loN * B - 1 / 2 ^ 118) ≤ X := by
      rw [hX]
      have e : (1 - eps) * lam * (Fl * L - Fh * (t * absL)) =
          (1 - eps) * lam * (Fl * L) - (1 - eps) * lam * (Fh * (t * absL)) := by ring
      linarith
    have k2 : q * lam * (Fl * L) ≤ (1 - 3 / 2 ^ 24) * ((1 - eps) * lam * (Fl * L)) := by
      have h0 : 0 ≤ lam * (Fl * L) := mul_nonneg hlam0 (mul_nonneg hFl0 hL)
      have := mul_le_mul_of_nonneg_right hq h0
      linarith
    have k3 : (1 - 3 / 2 ^ 24) * ((1 - eps) * lam * (Fh * (t * absL))) ≤ Fh * s * (t * absL) := by
      have h0 : 0 ≤ Fh * (t * absL) := by positivity
      have h1 : (1 - 3 / 2 ^ 24) * ((1 - eps) * lam) ≤ 1 := by
        have h3 : (1 - eps) * lam ≤ 1 * 1 := mul_le_mul (by linarith) hlam1 hlam0 (by norm_num)
        have := mul_le_mul (show (1 - 3 / 2 ^ 24 : ℝ) ≤ 1 by norm_num) h3 hel (by norm_num)
        linarith
      have h2 : Fh * (t * absL) ≤ Fh * s * (t * absL) := by
        have := mul_le_mul_of_nonneg_left (show (1 : ℝ) ≤ s by linarith) h0
        linarith
      have := mul_le_mul_of_nonneg_right h1 h0
      linarith
    have k4 : Fl * q * B ≤ (1 - 3 / 2 ^ 24) * (loN * B) := by
      have := mul_le_mul_of_nonneg_right hFlq hB; linarith
    have k5 : (1 - 3 / 2 ^ 24) * ((1 - eps) * lam * (Fl * L)) - (1 - 3 / 2 ^ 24) * ((1 - eps) * lam * (Fh * (t * absL))) -
        (1 - 3 / 2 ^ 24) * (1 / 2 ^ 82 + 1 / 2 ^ 118) + (1 - 3 / 2 ^ 24) * (loN * B) ≤ (1 - 3 / 2 ^ 24) * X := by
      have := mul_le_mul_of_nonneg_left hX1 (by norm_num : (0 : ℝ) ≤ 1 - 3 / 2 ^ 24)
      linarith
    have k6 : (1 - 3 / 2 ^ 24) * (1 / 2 ^ 82 + 1 / 2 ^ 118) + 1 / 2 ^ 100 ≤ (1 : ℝ) / 2 ^ 80 := by norm_num
    have e : Fl * q * (lam * L + B) - Fh * s * ((t + 1) * absL) = q * lam * (Fl * L) + Fl * q * B -
        Fh * s * (t * absL) - Fh * s * absL := by ring
    rw [e]
    linarith

/-- **One block for `l_i`** (lines 82–85, 106). `x` are the block's exponents, `p` their `exp2`,
`lij` the row sum of `p`, `dm` and `al` alpha's input and `alpha`, `l'` the new `l_i`. Before the
first block `l_i` is `1.0`. -/
theorem lstep {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool}
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hin : AlphaIn t m m' dm) (ha : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : (t = 0 ∧ l.isFinite = true ∧ l.toReal = 1) ∨ (0 < t ∧ MInv R t m ∧ LInv R t m l)) :
    LInv R (t + 1) m' l' := by
  obtain ⟨hlijf, hlij1, hlij2, hlijb⟩ := block_l hexp hR hX hp hsum
  have hmi : 0 < t → MInv R t m := fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi
  obtain ⟨haf, hab, ha0, hae⟩ := alpha_facts hexp hni hR hM' hin ha hmi
  have hq := pER_ge hR
  have hq0 : 0 < pER R := by linarith
  have hloN : 0 ≤ (1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R) := by
    have := gamr_le; apply mul_nonneg (by linarith); positivity
  have hhiN : 0 ≤ (1 + gamr) * (1 + psiPR R) := by
    have := gamr_nonneg; have := psiPR_nonneg hR; positivity
  -- the old `l_i`
  have hlb : l.isFinite = true ∧ 0 < l.toReal ∧ |l.toReal| ≤ 2 ^ 44 := by
    rcases hold with ⟨-, hf, h1⟩ | ⟨ht0, hmi', hL⟩
    · exact ⟨hf, by rw [h1]; norm_num, by rw [h1]; norm_num⟩
    · exact ⟨hL.1, hL.2.1, hL.bound hR (by omega) hmi'⟩
  -- line 106
  have hroom : |l.toReal * al.toReal| + |lij.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]; have := mul_le_mul hlb.2.2 hab (abs_nonneg _) (by norm_num); linarith
  obtain ⟨hl'f, hl'e⟩ := f32_mulAdd hl' hlb.1 haf hlijf hroom
  have heps0 : 0 ≤ (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 := by have := dmaxR_nonneg hR; nlinarith
  have heps1 : (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 ≤ 1 := by have := dmaxR_le hR; nlinarith
  have hs : (1 + 3 / 2 ^ 24) * (1 + ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1)) ≤ stepLR R := by
    unfold stepLR; ring_nf; exact le_rfl
  have hqq : qLR R ≤ (1 - 3 / 2 ^ 24) * (1 - ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1)) := by
    unfold qLR; ring_nf; exact le_rfl
  have hW1 := psum_wt_ge hM'
  have hsplit := psum_step1 R m.toReal m'.toReal t
  have habsL : (1 : ℝ) / 2 ^ 80 ≤ absLr := by unfold absLr; exact le_rfl
  -- the bounds
  have key : l'.toReal ≤ FhR R (t + 1) * (psum R.al (t + 1) (wt R m'.toReal) + ((t + 1 : ℕ) : ℝ) * absLr) ∧
      FlR R (t + 1) * psum R.al (t + 1) (wt R m'.toReal) - FhR R (t + 1) * (((t + 1 : ℕ) : ℝ) * absLr) ≤
        l'.toReal := by
    rw [FhR_succ, FlR_succ]
    push_cast
    rcases hold with ⟨rfl, -, hl1⟩ | ⟨ht0, hmi', hL⟩
    · -- the first block: `l_i = 1`, `alpha = 0`
      have hal0 := ha0 rfl
      rw [hsplit, psum_zero, mul_zero, zero_add]
      have h := l_step_real (l := l.toReal) (α := al.toReal) (L := 1) (lam := 0) (t := 0)
        (eps := (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1)
        (Fh := FhR R 0) (Fl := FlR R 0) (s := stepLR R) (q := qLR R) (hiN := (1 + gamr) * (1 + psiPR R))
        (loN := (1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R)) (absL := absLr) (B := ∑ jj, bw1 R m'.toReal 0 jj)
        hlb.2.1 (by rw [hl1]; simp; exact FhR_ge hR 0) (by rw [hl1]; simp; exact FlR_le_one hR 0)
        (by rw [hl1]; norm_num) zero_le_one (Finset.sum_nonneg fun jj _ => bw1_nonneg _ _ _ _) le_rfl
        zero_le_one heps0 heps1 (by rw [hal0]; simp) hlij1 hlij2 hloN hhiN hl'e hs (stepLR_ge hR) hqq
        (qLR_nonneg hR) (FhR_ge hR 0) (FhR_ge_hiN hR 0) (FlR_nonneg hR 0) (FlR_qL hR 0) habsL le_rfl
      simp only [zero_mul, zero_add, Nat.cast_zero] at h ⊢
      exact h
    · obtain ⟨hle, hae'⟩ := hae ht0
      have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
      have hl0 := two_rpow_pos (m.toReal - m'.toReal)
      have hL0 : 0 ≤ psum R.al t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
      rw [hsplit]
      exact l_step_real (l := l.toReal) (α := al.toReal) (lam := (2 : ℝ) ^ (m.toReal - m'.toReal))
        (eps := (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) (t := (t : ℝ))
        (Fh := FhR R t) (Fl := FlR R t) (s := stepLR R) (q := qLR R) (hiN := (1 + gamr) * (1 + psiPR R))
        (loN := (1 - gamr) * ((1 - 1 / 2 ^ 22) / pER R)) (absL := absLr) (B := ∑ jj, bw1 R m'.toReal t jj)
        hlb.2.1 hL.2.2.1 hL.2.2.2 (le_of_abs_le hlb.2.2) hL0
        (Finset.sum_nonneg fun jj _ => bw1_nonneg _ _ _ _) hl0.le hl1 heps0 heps1 hae' hlij1 hlij2 hloN hhiN hl'e
        hs (stepLR_ge hR) hqq (qLR_nonneg hR) (FhR_ge hR t) (FhR_ge_hiN hR t) (FlR_nonneg hR t) (FlR_qL hR t)
        habsL (Nat.cast_nonneg t)
  refine ⟨hl'f, ?_, key.1, key.2⟩
  -- `l_i > 0`
  have hFl := FlR_ge hR (t := t + 1) (by omega)
  have hFh := FhR_le hR (t := t + 1) (by omega)
  have hp := pER_le hR
  have htr : ((t + 1 : ℕ) : ℝ) ≤ 1024 := by exact_mod_cast ht
  have h1 : 1 / 2 / pER R * (1 / pER R) ≤ FlR R (t + 1) * psum R.al (t + 1) (wt R m'.toReal) :=
    mul_le_mul hFl hW1 (by positivity) (FlR_nonneg hR _)
  have h2 : (1 : ℝ) / 2 ^ 27 ≤ 1 / 2 / pER R * (1 / pER R) := by
    rw [div_div, div_mul_div_comm, div_le_div_iff₀ (by norm_num) (by positivity)]
    nlinarith
  have h3 : FhR R (t + 1) * (((t + 1 : ℕ) : ℝ) * absLr) ≤ 1 / 2 ^ 56 := by
    calc FhR R (t + 1) * (((t + 1 : ℕ) : ℝ) * absLr) ≤ 2 ^ 14 * (1024 * (1 / 2 ^ 80)) := by
          unfold absLr
          exact mul_le_mul hFh (mul_le_mul_of_nonneg_right htr (by norm_num)) (by positivity) (by norm_num)
      _ = 1 / 2 ^ 56 := by norm_num
  have := key.2
  have : (0 : ℝ) < 1 / 2 ^ 27 - 1 / 2 ^ 56 := by norm_num
  linarith

/-! ## `acc` -/

/-- The exact weighted sum of the attended keys of the first `t` blocks, in the units of the
exact weights: `Σ ew_j·ve_j`. -/
noncomputable def Sx (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : ℝ := psum R.al t (fun j => ew R j * ve j)

/-- `Σ ew_j·|ve_j|` over the attended keys of block `t`. -/
noncomputable def Mg (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ jj : Fin 64, if R.al (64 * t + jj) = true then ew R (64 * t + jj) * |ve (64 * t + jj)| else 0

/-- `Σ |ve_j|` over the attended keys of block `t`. -/
noncomputable def AbV (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ jj : Fin 64, if R.al (64 * t + jj) = true then |ve (64 * t + jj)| else 0

/-- An element of the row's `acc` after `t` blocks (`t ≥ 1`), `ve` the values of its column of
`v`, `ES` and `EB` the bounds of `Model.errS` and `Model.errB` after `t` blocks. -/
def AInv (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) (m a : F32) (ES EB : ℝ) : Prop :=
  a.isFinite = true ∧ |a.toReal| ≤ crA t ∧
    |a.toReal - (2 : ℝ) ^ (-m.toReal) * Sx R ve t| ≤ (2 : ℝ) ^ (-m.toReal) * ES + EB

theorem Sx_succ (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) :
    Sx R ve (t + 1) = Sx R ve t +
      ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then ew R (64 * t + jj) * ve (64 * t + jj) else 0) :=
  psum_succ _ _ _

theorem Mg_nonneg (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : 0 ≤ Mg R ve t :=
  Finset.sum_nonneg fun jj _ => by
    split_ifs
    · exact mul_nonneg (ew_pos _ _).le (abs_nonneg _)
    · exact le_rfl

theorem AbV_nonneg (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : 0 ≤ AbV R ve t :=
  Finset.sum_nonneg fun jj _ => by
    split_ifs
    · exact abs_nonneg _
    · exact le_rfl

/-- **One block of an element of `acc`, in real numbers** (lines 95 and 103). The old value `a`
is within `K·ES + EB` of `K·S`; `a1` (`acc·alpha`) is within `(Φ − 1)·λ|a| + τ1` of `λ·a`; the
block's `Σ p·v` (`cn`) is within `psi·K'·Mn + om·Vn` of `K'·Sn`, with magnitudes `Tq`; the `tl.dot`
adds them within `ρ·(|a1| + Tq) + absD`. With `K' = λ·K`, the new value is within
`K'·((1 + rA)·ES + rA·|S| + ((1+ρ)(1+psi) − 1)·Mn) + ((1 + rA)·EB + (1+ρ)·om·Vn + (1+ρ)τ1 + absD)` of
`K'·(S + Sn)`, `rA = (1 + ρ)Φ − 1`. -/
theorem acc_step8 {a a1 a' S Sn Mn Vn ES EB K K' lam Phi tau1 rho psi om Tq cn absD : ℝ}
    (hK : 0 ≤ K) (hlam0 : 0 ≤ lam) (hlam1 : lam ≤ 1) (hK' : K' = lam * K)
    (hEB : 0 ≤ EB) (hold : |a - K * S| ≤ K * ES + EB) (hPhi : 1 ≤ Phi)
    (ha1 : |a1 - lam * a| ≤ (Phi - 1) * (lam * |a|) + tau1)
    (hcn : |cn - K' * Sn| ≤ psi * (K' * Mn) + om * Vn) (hTq : Tq ≤ (1 + psi) * (K' * Mn) + om * Vn)
    (hrho : 0 ≤ rho) (hcomb : |a' - (a1 + cn)| ≤ rho * (|a1| + Tq) + absD) :
    |a' - K' * (S + Sn)| ≤
      K' * ((1 + rho) * Phi * ES + ((1 + rho) * Phi - 1) * |S| + ((1 + rho) * (1 + psi) - 1) * Mn) +
        ((1 + rho) * Phi * EB + (1 + rho) * om * Vn + ((1 + rho) * tau1 + absD)) := by
  have hK'0 : 0 ≤ K' := by rw [hK']; positivity
  have hla : lam * |a| ≤ K' * |S| + K' * ES + EB := by
    have h1 : |a| ≤ K * |S| + (K * ES + EB) := by
      have := abs_sub_abs_le_abs_sub a (K * S)
      rw [abs_mul, abs_of_nonneg hK] at this; linarith
    have := mul_le_mul_of_nonneg_left h1 hlam0
    rw [hK']
    have : lam * EB ≤ EB := by nlinarith
    nlinarith
  have ha1b : |a1| ≤ Phi * (lam * |a|) + tau1 := by
    have := abs_sub_abs_le_abs_sub a1 (lam * a)
    rw [abs_mul, abs_of_nonneg hlam0] at this; linarith
  have e : a' - K' * (S + Sn) = (a' - (a1 + cn)) + (a1 - lam * a) + lam * (a - K * S) + (cn - K' * Sn) := by
    rw [hK']; ring
  rw [e]
  have t1 := abs_add_le ((a' - (a1 + cn)) + (a1 - lam * a) + lam * (a - K * S)) (cn - K' * Sn)
  have t2 := abs_add_le ((a' - (a1 + cn)) + (a1 - lam * a)) (lam * (a - K * S))
  have t3 := abs_add_le (a' - (a1 + cn)) (a1 - lam * a)
  have t4 : |lam * (a - K * S)| ≤ lam * (K * ES + EB) := by
    rw [abs_mul, abs_of_nonneg hlam0]; exact mul_le_mul_of_nonneg_left hold hlam0
  have t5 : lam * (K * ES + EB) ≤ K' * ES + EB := by
    rw [hK']; have : lam * EB ≤ EB := by nlinarith
    nlinarith
  have hpsi : 0 ≤ 1 + psi ∨ True := Or.inr trivial
  clear hpsi
  have k1 : rho * |a1| ≤ rho * (Phi * (lam * |a|) + tau1) := mul_le_mul_of_nonneg_left ha1b hrho
  have k2 : rho * Tq ≤ rho * ((1 + psi) * (K' * Mn) + om * Vn) := mul_le_mul_of_nonneg_left hTq hrho
  have k3 : ((1 + rho) * Phi - 1) * (lam * |a|) ≤ ((1 + rho) * Phi - 1) * (K' * |S| + K' * ES + EB) :=
    mul_le_mul_of_nonneg_left hla (by nlinarith)
  have e2 : rho * (Phi * (lam * |a|) + tau1) + (Phi - 1) * (lam * |a|) =
      ((1 + rho) * Phi - 1) * (lam * |a|) + rho * tau1 := by ring
  nlinarith

theorem relDot8_64r : ((Rules.Num.relDot8 64 : ℚ) : ℝ) = rho8r := by
  unfold Rules.Num.relDot8 Rules.Num.chunks8 rho8r; push_cast; norm_num

theorem absDot8_64r : ((Rules.Num.absDot8 64 : ℚ) : ℝ) = 16641 / 2 ^ 139 := by
  unfold Rules.Num.absDot8 Rules.Num.chunks8; push_cast; norm_num

set_option maxHeartbeats 1000000 in
/-- **One block for an element of `acc`** (lines 82–84, 95, 98–103): `q` the block's row of
`p.to(float8e5)`, `vt` its column of `v` (the values `ve` of keys `64t + jj`), `a1` the element of
`acc * alpha[:, None]`, `a'` that of `tl.dot(p, v, acc)` (`hdot`: `Num.DotE5m2` at this element).
Before the first block the element is `0.0`. Also the dot's error in real numbers (for
`Row.mag_tail`). -/
theorem astep {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {ve : ℕ → ℝ} (hve : ∀ j, |ve j| ≤ 57344)
    {ES EB : ℝ} (hES : 0 ≤ ES) (hEB : 0 ≤ EB)
    {m m' dm al a a1 a' : F32} {x p : Fin 64 → F32} {q vt : Fin 64 → F8E5}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (hin : AlphaIn t m m' dm) (ha : P.math .exp2 .binary32 dm al)
    (ha1 : FloatV.NaNEq (FloatV.mul a al) a1)
    (hvt : ∀ jj, (vt jj).isFinite = true ∧ (vt jj).toReal = ve (64 * t + jj))
    (hdot : a1.isFinite = true → (∀ k, (q k).isFinite = true) →
      |a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat| +
        (Rules.Num.relDot8 64 * (|a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      a'.isFinite = true ∧ |a'.toRat - (a1.toRat + ∑ k, (q k).toRat * (vt k).toRat)| ≤
        Rules.Num.relDot8 64 * (|a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat|) + Rules.Num.absDot8 64)
    (hold : (t = 0 ∧ a.isFinite = true ∧ a.toReal = 0) ∨ (0 < t ∧ MInv R t m ∧ AInv R ve t m a ES EB)) :
    AInv R ve (t + 1) m' a'
      ((1 + rhoAR R) * ES + rhoAR R * |Sx R ve t| + psiAR R * Mg R ve t)
      ((1 + rhoAR R) * EB + (1 + rho8r) * om8r * AbV R ve t + absAr) ∧
    |a'.toReal - (a1.toReal + ∑ jj, (q jj).toReal * (vt jj).toReal)| ≤
      rho8r * (|a1.toReal| + ∑ jj, |(q jj).toReal * (vt jj).toReal|) + 16641 / 2 ^ 139 := by
  have hps0 := psiPR_nonneg hR
  have hp80 := psi8R_nonneg hR
  set K' := (2 : ℝ) ^ (-m'.toReal) with hK'd
  have hK'0 : 0 < K' := two_rpow_pos _
  -- the lanes
  have hlane : ∀ jj, (q jj).isFinite = true ∧
      |(q jj).toReal * (vt jj).toReal -
          (if R.al (64 * t + jj) = true then K' * (ew R (64 * t + jj) * ve (64 * t + jj)) else 0)| ≤
        psi8R R * (if R.al (64 * t + jj) = true then K' * (ew R (64 * t + jj) * |ve (64 * t + jj)|) else 0) +
          om8r * (if R.al (64 * t + jj) = true then |ve (64 * t + jj)| else 0) ∧
      |(q jj).toReal * (vt jj).toReal| ≤
        (1 + psi8R R) * (if R.al (64 * t + jj) = true then K' * (ew R (64 * t + jj) * |ve (64 * t + jj)|) else 0) +
          om8r * (if R.al (64 * t + jj) = true then |ve (64 * t + jj)| else 0) ∧
      |(q jj).toReal * (vt jj).toReal| ≤ 0 * (q jj).toReal + 2 * 57344 := by
    intro jj
    obtain ⟨hxf, hxle, hxa, hxm⟩ := hX jj
    obtain ⟨hpf, hpb, hpa, hpm⟩ := lane_p hexp hR hxf hxle hxa hxm (hp jj)
    obtain ⟨hqf, hqb, hqa, hqm⟩ := lane_p8 hcvt hpf hpb (fun h => (hpa h).1) hpm (hq jj)
    obtain ⟨-, hvv⟩ := hvt jj
    rw [hvv]
    have hv := hve (64 * t + jj)
    have hva := abs_nonneg (ve (64 * t + jj))
    refine ⟨hqf, ?_, ?_, ?_⟩
    · cases hal : R.al (64 * t + jj)
      · rw [toReal_def, hqm hal]; simp
      · have h1 := hqa hal
        simp only [↓reduceIte]
        have hG : wt R m'.toReal (64 * t + jj) = K' * ew R (64 * t + jj) := wt_eq _ _ _
        rw [hG] at h1
        rw [show (q jj).toReal * ve (64 * t + jj) - K' * (ew R (64 * t + jj) * ve (64 * t + jj)) =
          ((q jj).toReal - K' * ew R (64 * t + jj)) * ve (64 * t + jj) by ring, abs_mul]
        have := mul_le_mul_of_nonneg_right h1 hva
        have e : psi8R R * (K' * (ew R (64 * t + jj) * |ve (64 * t + jj)|)) + om8r * |ve (64 * t + jj)| =
            (psi8R R * (K' * ew R (64 * t + jj)) + om8r) * |ve (64 * t + jj)| := by ring
        rw [e]; exact this
    · cases hal : R.al (64 * t + jj)
      · rw [toReal_def, hqm hal]; simp
      · have h1 := hqa hal
        simp only [↓reduceIte]
        have hG : wt R m'.toReal (64 * t + jj) = K' * ew R (64 * t + jj) := wt_eq _ _ _
        rw [hG] at h1
        have hGp : 0 < K' * ew R (64 * t + jj) := mul_pos hK'0 (ew_pos _ _)
        have h2 : |(q jj).toReal| ≤ (1 + psi8R R) * (K' * ew R (64 * t + jj)) + om8r := by
          have := abs_sub_abs_le_abs_sub (q jj).toReal (K' * ew R (64 * t + jj))
          rw [abs_of_pos hGp] at this; linarith
        rw [abs_mul]
        have := mul_le_mul_of_nonneg_right h2 hva
        have e : (1 + psi8R R) * (K' * (ew R (64 * t + jj) * |ve (64 * t + jj)|)) + om8r * |ve (64 * t + jj)| =
            ((1 + psi8R R) * (K' * ew R (64 * t + jj)) + om8r) * |ve (64 * t + jj)| := by ring
        rw [e]; exact this
    · rw [abs_mul]; have := mul_le_mul hqb hv hva (by norm_num); linarith
  have hqf : ∀ jj, (q jj).isFinite = true := fun jj => (hlane jj).1
  have hcn := lane_sum_err2 (fun jj => (hlane jj).2.1)
  have hTn := lane_sum_le2 (fun jj => (hlane jj).2.2.1)
  have hTb := lane_sum_le (fun jj => (hlane jj).2.2.2)
  simp only [zero_mul, zero_add] at hTb
  rw [← Finset.mul_sum] at hcn hTn
  have hSn : ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then K' * (ew R (64 * t + jj) * ve (64 * t + jj)) else 0) =
      K' * ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then ew R (64 * t + jj) * ve (64 * t + jj) else 0) := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun jj _ => by split_ifs <;> simp
  have hMn : ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then K' * (ew R (64 * t + jj) * |ve (64 * t + jj)|) else 0) =
      K' * Mg R ve t := by
    unfold Mg; rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun jj _ => by split_ifs <;> simp
  have hVn : ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then |ve (64 * t + jj)| else 0) = AbV R ve t := rfl
  rw [hSn, hMn, hVn] at hcn
  rw [hMn, hVn] at hTn
  -- `acc` and `alpha`
  have hmi : 0 < t → MInv R t m := fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi
  obtain ⟨haf, hab, ha0, hae⟩ := alpha_facts hexp hni hR hM' hin ha hmi
  have hacc : a.isFinite = true ∧ |a.toReal| ≤ crA t := by
    rcases hold with ⟨-, hf, hz⟩ | ⟨-, -, hA⟩
    · exact ⟨hf, by rw [hz, abs_zero]; exact crA_nonneg t⟩
    · exact ⟨hA.1, hA.2.1⟩
  have hcr := crA_le (t := t) (by omega)
  -- line 95
  have hroom1 : |a.toReal * al.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]; have := mul_le_mul hacc.2 hab (abs_nonneg _) (crA_nonneg t); linarith
  obtain ⟨ha1f, ha1e⟩ := f32_mul ha1 hacc.1 haf hroom1
  have hrho0 : (0 : ℝ) ≤ rho8r := by unfold rho8r; positivity
  have hrhoA := rhoAR_nonneg hR
  have hrhoA1 := rhoAR_le hR
  -- the case split: before the first block `λ = 1`, `K = K'`, `a = 0`
  obtain ⟨K, lam, hK0, hlam0, hlam1, hKK, hold', ha1'⟩ : ∃ K lam : ℝ, 0 ≤ K ∧ 0 ≤ lam ∧ lam ≤ 1 ∧
      K' = lam * K ∧ |a.toReal - K * Sx R ve t| ≤ K * ES + EB ∧
      |a1.toReal - lam * a.toReal| ≤ ((1 + dmaxR R) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) - 1) *
        (lam * |a.toReal|) + (|a.toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150) := by
    rcases hold with ⟨rfl, -, hz⟩ | ⟨ht0, hmi', hA⟩
    · refine ⟨K', 1, hK'0.le, zero_le_one, le_rfl, by ring, ?_, ?_⟩
      · rw [hz, show Sx R ve 0 = 0 by simp [Sx, psum_zero], mul_zero, sub_zero, abs_zero]; positivity
      · rw [ha0 rfl, mul_zero, sub_zero, abs_zero] at ha1e
        rw [hz]; simp only [mul_zero, sub_zero, abs_zero, zero_mul, zero_add]
        norm_num at ha1e ⊢; linarith
    · obtain ⟨hle, hae'⟩ := hae ht0
      have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
      have hl0 := two_rpow_pos (m.toReal - m'.toReal)
      refine ⟨(2 : ℝ) ^ (-m.toReal), (2 : ℝ) ^ (m.toReal - m'.toReal), (two_rpow_pos _).le, hl0.le, hl1,
        ?_, hA.2.2, ?_⟩
      · rw [hK'd, ← two_rpow_add]; congr 1; ring
      · exact scale_err hl0.le hae' ha1e
  set Phi := (1 + dmaxR R) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) with hPhid
  set tau1 := |a.toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150 with htau1d
  have hPhi1 : (1 : ℝ) ≤ Phi := by rw [hPhid]; have := dmaxR_nonneg hR; nlinarith
  have hrA : (1 + rho8r) * Phi = 1 + rhoAR R := by rw [hPhid]; unfold rhoAR; ring
  have htau1 : tau1 ≤ 1 / 2 ^ 66 := by
    rw [htau1d]
    have h1 : |a.toReal| * (1 + 1 / 2 ^ 24) ≤ 2 ^ 58 * 2 := mul_le_mul (hacc.2.trans hcr) (by norm_num)
      (by norm_num) (by norm_num)
    have h2 : |a.toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 ≤ 2 ^ 58 * 2 / 2 ^ 126 :=
      div_le_div_of_nonneg_right h1 (by norm_num)
    norm_num at h2 ⊢; linarith
  have htau0 : 0 ≤ tau1 := by rw [htau1d]; positivity
  have ha1b : |a1.toReal| ≤ Phi * crA t + tau1 := by
    have h1 := abs_sub_abs_le_abs_sub a1.toReal (lam * a.toReal)
    rw [abs_mul, abs_of_nonneg hlam0] at h1
    have h2 : lam * |a.toReal| ≤ crA t := by
      have := mul_le_mul hlam1 hacc.2 (abs_nonneg _) zero_le_one; linarith
    have h3 : (Phi - 1) * (lam * |a.toReal|) ≤ (Phi - 1) * crA t := mul_le_mul_of_nonneg_left h2 (by linarith)
    linarith
  have ha1b' : |a1.toReal| ≤ 2 ^ 60 := by
    have h1 : Phi ≤ 2 := by rw [hPhid]; have := dmaxR_le hR; nlinarith
    have := mul_le_mul h1 hcr (crA_nonneg t) (by norm_num)
    norm_num at htau1; nlinarith
  -- line 103
  have hdroom : |a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat| +
      (Rules.Num.relDot8 64 * (|a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat|) + Rules.Num.absDot8 64) ≤
        Format.binary32.maxVal := by
    have : (((|a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat| +
        (Rules.Num.relDot8 64 * (|a1.toRat| + ∑ k, |(q k).toRat * (vt k).toRat|) + Rules.Num.absDot8 64)) : ℚ) : ℝ) ≤
        (Format.binary32.maxVal : ℝ) := by
      push_cast
      rw [relDot8_64r, absDot8_64r, maxVal32]
      simp only [← toReal_def]
      have h1 : |a1.toReal| + ∑ k, |(q k).toReal * (vt k).toReal| ≤ 2 ^ 61 := by
        have : (64 : ℝ) * (2 * 57344) ≤ 2 ^ 60 := by norm_num
        linarith
      have h0 : 0 ≤ |a1.toReal| + ∑ k, |(q k).toReal * (vt k).toReal| :=
        add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
      have h2 : rho8r * (|a1.toReal| + ∑ k, |(q k).toReal * (vt k).toReal|) ≤ 1 * 2 ^ 61 :=
        mul_le_mul (by unfold rho8r; norm_num) h1 h0 (by norm_num)
      have h3 : (2 : ℝ) ^ 61 + 1 * 2 ^ 61 + 16641 / 2 ^ 139 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith
    exact_mod_cast this
  obtain ⟨ha'f, ha'e⟩ := hdot ha1f hqf hdroom
  have hcomb : |a'.toReal - (a1.toReal + ∑ jj, (q jj).toReal * (vt jj).toReal)| ≤
      rho8r * (|a1.toReal| + ∑ jj, |(q jj).toReal * (vt jj).toReal|) + 16641 / 2 ^ 139 := by
    have := (Rat.cast_le (K := ℝ)).mpr ha'e
    push_cast at this
    rw [relDot8_64r, absDot8_64r] at this
    simp only [← toReal_def] at this
    exact this
  refine ⟨⟨ha'f, ?_, ?_⟩, hcomb⟩
  · -- the crude bound
    have h1 : |a'.toReal| ≤ (1 + rho8r) * (|a1.toReal| + ∑ jj, |(q jj).toReal * (vt jj).toReal|) + 16641 / 2 ^ 139 := by
      have := abs_sub_abs_le_abs_sub a'.toReal (a1.toReal + ∑ jj, (q jj).toReal * (vt jj).toReal)
      have := abs_add_le a1.toReal (∑ jj, (q jj).toReal * (vt jj).toReal)
      have : |∑ jj, (q jj).toReal * (vt jj).toReal| ≤ ∑ jj, |(q jj).toReal * (vt jj).toReal| :=
        Finset.abs_sum_le_sum_abs _ _
      linarith
    have h2 : (1 + rho8r) * (|a1.toReal| + ∑ jj, |(q jj).toReal * (vt jj).toReal|) ≤
        (1 + rho8r) * Phi * crA t + (1 + rho8r) * (tau1 + 64 * (2 * 57344)) := by
      have := mul_le_mul_of_nonneg_left (add_le_add ha1b hTb) (by linarith : (0 : ℝ) ≤ 1 + rho8r)
      linarith
    have h3 : (1 + rho8r) * Phi * crA t ≤ 64 / 63 * crA t := by
      rw [hrA]; exact mul_le_mul_of_nonneg_right hrhoA1 (crA_nonneg t)
    have h4 : (1 + rho8r) * (tau1 + 64 * (2 * 57344)) + 16641 / 2 ^ 139 ≤ 2 ^ 23 := by
      have k1 : (1 + rho8r) * (tau1 + 64 * (2 * 57344)) ≤ (1 + rho8r) * (1 / 2 ^ 66 + 64 * (2 * 57344)) :=
        mul_le_mul_of_nonneg_left (by linarith) (by linarith)
      have k2 : (1 + rho8r) * (1 / 2 ^ 66 + 64 * (2 * 57344)) + 16641 / 2 ^ 139 ≤ 2 ^ 23 := by
        unfold rho8r; norm_num
      linarith
    have h5 := crA_succ t
    linarith
  · -- the error bound
    rw [Sx_succ]
    have key := acc_step8 (a := a.toReal) (a1 := a1.toReal) (a' := a'.toReal) (S := Sx R ve t)
      (Sn := ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then ew R (64 * t + jj) * ve (64 * t + jj) else 0))
      (Mn := Mg R ve t) (Vn := AbV R ve t) (ES := ES) (EB := EB) (K := K) (K' := K') (lam := lam)
      (Phi := Phi) (tau1 := tau1) (rho := rho8r) (psi := psi8R R)
      (om := om8r) (Tq := ∑ jj, |(q jj).toReal * (vt jj).toReal|)
      (cn := ∑ jj, (q jj).toReal * (vt jj).toReal) (absD := 16641 / 2 ^ 139)
      hK0 hlam0 hlam1 hKK hEB hold' hPhi1 ha1' hcn hTn hrho0 hcomb
    rw [hrA, show 1 + rhoAR R - 1 = rhoAR R by ring, show (1 + rho8r) * (1 + psi8R R) - 1 = psiAR R by
      unfold psiAR; ring] at key
    have htau : (1 + rho8r) * tau1 + 16641 / 2 ^ 139 ≤ absAr := by
      have : (1 + rho8r) * tau1 ≤ 2 * (1 / 2 ^ 66) :=
        mul_le_mul (by unfold rho8r; norm_num) htau1 htau0 (by norm_num)
      unfold absAr; norm_num at this ⊢; linarith
    linarith

end Kernels.AttnFwdE5m2
