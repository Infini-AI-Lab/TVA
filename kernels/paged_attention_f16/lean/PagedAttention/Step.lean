import PagedAttention.RowAcc

/-!
# `paged_attention_f16`: one row, one page of the loop for `l_i` and `acc`

* `LInv`, `AInv`: after `t ≥ 1` pages with running maximum `m`, `l_i` and an element of `acc`
  (column values `ve`) are within a relative error `ffac − 1` and an absolute error of
  `Σ wt` and `Σ wt·ve` over the sequence's tokens so far.
* `lstep` (lines 64–66) and `astep` (lines 64–68): one page keeps them.
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-! ## Page terms -/

/-- Lane `r`'s weight in page `t` (`0` for a masked lane). -/
noncomputable def bw1 (R : RowCtx) (m' : ℝ) (t : ℕ) (r : Fin R.P) : ℝ :=
  if R.P * t + r < R.n then wt R m' (R.P * t + r) else 0

/-- Lane `r`'s weight times `g` in page `t`. -/
noncomputable def bwg (R : RowCtx) (m' : ℝ) (t : ℕ) (g : ℕ → ℝ) (r : Fin R.P) : ℝ :=
  if R.P * t + r < R.n then wt R m' (R.P * t + r) * g (R.P * t + r) else 0

theorem psum_step1 (R : RowCtx) (m m' : ℝ) (t : ℕ) :
    psum R (t + 1) (wt R m') = Real.exp (m - m') * psum R t (wt R m) + ∑ r, bw1 R m' t r := by
  have := psum_step R m m' t (fun _ => 1)
  simp only [mul_one] at this
  exact this

theorem psum_stepg (R : RowCtx) (m m' : ℝ) (t : ℕ) (g : ℕ → ℝ) :
    psum R (t + 1) (fun j => wt R m' j * g j) =
      Real.exp (m - m') * psum R t (fun j => wt R m j * g j) + ∑ r, bwg R m' t g r :=
  psum_step R m m' t g

theorem bw1_nonneg (R : RowCtx) (m' : ℝ) (t : ℕ) (r : Fin R.P) : 0 ≤ bw1 R m' t r := by
  unfold bw1; split_ifs
  · exact (wt_pos _ _ _).le
  · exact le_rfl

theorem bwg_abs_nonneg (R : RowCtx) (m' : ℝ) (t : ℕ) (g : ℕ → ℝ) (r : Fin R.P) :
    0 ≤ bwg R m' t (fun j => |g j|) r := by
  unfold bwg; split_ifs
  · exact mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
  · exact le_rfl

theorem abs_bwg_le (R : RowCtx) (m' : ℝ) (t : ℕ) (g : ℕ → ℝ) (r : Fin R.P) :
    |bwg R m' t g r| ≤ bwg R m' t (fun j => |g j|) r := by
  unfold bwg; split_ifs
  · rw [abs_mul, abs_of_pos (wt_pos _ _ _)]
  · simp

theorem psum_abs_wt (R : RowCtx) (m : ℝ) (t : ℕ) (ve : ℕ → ℝ) :
    |psum R t (fun j => wt R m j * ve j)| ≤ psum R t (fun j => wt R m j * |ve j|) := by
  refine (abs_psum_le _ _ _).trans (le_of_eq ?_)
  unfold psum
  refine Finset.sum_congr rfl fun j _ => ?_
  dsimp only
  rw [abs_mul, abs_of_pos (wt_pos R m j)]

theorem sum_fin_const (R : RowCtx) (c : ℝ) : ∑ _r : Fin R.P, c = R.P * c := by
  simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-! ## The invariants -/

/-- `l_i` after `t` pages (`t ≥ 1`) with running maximum `m`. -/
def LInv (R : RowCtx) (t : ℕ) (m l : F32) : Prop :=
  l.isFinite = true ∧
    |l.toReal - psum R t (wt R m.toReal)| ≤
      (ffac R (psiLr R) t m.toReal - 1) * psum R t (wt R m.toReal) +
        ffac R (psiLr R) t m.toReal * (t * absLr)

/-- An element of `acc` after `t` pages (`t ≥ 1`), `ve` the values of its column of `v`. -/
def AInv (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) (m a : F32) : Prop :=
  a.isFinite = true ∧
    |a.toReal - psum R t (fun j => wt R m.toReal j * ve j)| ≤
      (ffac R (psiAr R) t m.toReal - 1) * psum R t (fun j => wt R m.toReal j * |ve j|) +
        ffac R (psiAr R) t m.toReal * (t * absAr)

theorem LInv.bound {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 2 ^ 16) {m l : F32} (hm : MInv R t m)
    (hl : LInv R t m l) : |l.toReal| ≤ 2 ^ 26 := by
  have hF := ffac_le hR (psiLr_le hR) ht hm.bound
  have hF1 := ffac_one_le (t := t) (psiLr_nonneg hR) (m_nonneg hm.bound)
  have hW := psum_wt1_le hR ht hm
  have hW0 : 0 ≤ psum R t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
  have htA : (t : ℝ) * absLr ≤ 1 := by
    have : (t : ℝ) ≤ 2 ^ 16 := by exact_mod_cast ht
    unfold absLr; nlinarith
  have htA0 : 0 ≤ (t : ℝ) * absLr := by unfold absLr; positivity
  obtain ⟨-, he⟩ := hl
  have := abs_sub_abs_le_abs_sub l.toReal (psum R t (wt R m.toReal))
  rw [abs_of_nonneg hW0] at this
  have k1 := mul_le_mul_of_nonneg_right (by linarith : ffac R (psiLr R) t m.toReal - 1 ≤ 1 / 4) hW0
  have k2 := mul_le_mul_of_nonneg_right hF htA0
  nlinarith

theorem AInv.bound {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 2 ^ 16) {ve : ℕ → ℝ}
    (hve : ∀ j, |ve j| ≤ 2 ^ 15) {m a : F32} (hm : MInv R t m) (ha : AInv R ve t m a) :
    |a.toReal| ≤ 2 ^ 41 := by
  have hF := ffac_le hR (psiAr_le hR) ht hm.bound
  have hF1 := ffac_one_le (t := t) (psiAr_nonneg hR) (m_nonneg hm.bound)
  have hW := psum_wt1_le hR ht hm
  have hMg : psum R t (fun j => wt R m.toReal j * |ve j|) ≤ psum R t (fun j => wt R m.toReal j * 2 ^ 15) :=
    psum_le fun j _ _ => mul_le_mul_of_nonneg_left (hve j) (wt_pos _ _ _).le
  have e15 : psum R t (fun j => wt R m.toReal j * 2 ^ 15) = 2 ^ 15 * psum R t (wt R m.toReal) := by
    rw [← psum_mul]; congr 1; ext j; ring
  rw [e15] at hMg
  have hMg0 : 0 ≤ psum R t (fun j => wt R m.toReal j * |ve j|) :=
    psum_nonneg fun j => mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
  have hS := psum_abs_wt R m.toReal t ve
  have htA : (t : ℝ) * absAr ≤ 1 := by
    have : (t : ℝ) ≤ 2 ^ 16 := by exact_mod_cast ht
    unfold absAr; nlinarith
  have htA0 : 0 ≤ (t : ℝ) * absAr := by unfold absAr; positivity
  obtain ⟨-, he⟩ := ha
  have := abs_sub_abs_le_abs_sub a.toReal (psum R t (fun j => wt R m.toReal j * ve j))
  have k1 := mul_le_mul_of_nonneg_right (by linarith : ffac R (psiAr R) t m.toReal - 1 ≤ 1 / 4) hMg0
  have k2 := mul_le_mul_of_nonneg_right hF htA0
  linarith

/-! ## The lanes of a page -/

/-- **A page's `p`** (line 65): every lane's `p` is within `psiPr·bw1 + 2^-140` of its weight
(`0` on a masked lane, where `p = +0`), and the weights are at most `2`. -/
theorem lanes_p {P : Platform} (hexp : Rules.Num.ExpF32 P) (hni : Rules.Num.ExpF32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} {m' : F32} {x p : Fin R.P → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hp : ∀ r, P.math .exp .binary32 (x r) (p r))
    (r : Fin R.P) :
    (p r).isFinite = true ∧
      |(p r).toReal - bw1 R m'.toReal t r| ≤ psiPr R * bw1 R m'.toReal t r + 1 / 2 ^ 140 ∧
      bw1 R m'.toReal t r ≤ 2 ∧
      (¬ R.P * t + r < R.n → (p r).toReal = 0) := by
  unfold bw1
  by_cases hr : R.P * t + r < R.n
  · obtain ⟨hxf, hxe⟩ := (hX r).1 hr
    have hkey : R.P * t + r < R.P * (t + 1) := by rw [Nat.mul_succ]; omega
    obtain ⟨hpf, hpe⟩ := lane_p hexp hR hr hxf hxe (hM'.above _ hkey hr) hM'.bound (hp r)
    simp only [hr, ite_true]
    exact ⟨hpf, hpe, wt_le_two hR hM' hkey hr, fun h => (h trivial).elim⟩
  · have hx := (hX r).2 hr
    have hp' := hp r
    rw [hx] at hp'
    rw [lane_p_masked hni hp']
    simp only [hr, ite_false, mul_zero, sub_zero]
    obtain ⟨h1, h2⟩ := zeroF32_facts
    refine ⟨h1, by rw [h2]; norm_num, by norm_num, fun _ => h2⟩

/-! ## `l_i` -/

/-- **One page for `l_i`** (lines 64–66). `x` are the page's exponents, `p` their `exp`, `lsum`
the sum of `p`, `dm` and `al` alpha's input and `alpha`, `l'` the new `l_i`. Before the first page
`l_i` is `+0`. -/
theorem lstep {P : Platform} (hexp : Rules.Num.ExpF32 P) (hni : Rules.Num.ExpF32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 2 ^ 16) {fusion : Bool}
    {m m' dm al l lsum l' : F32} {x p : Fin R.P → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x)
    (hp : ∀ r, P.math .exp .binary32 (x r) (p r))
    (hin : AlphaIn t m m' dm) (ha : P.math .exp .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (AddRel false) (laneSeq p) lsum)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lsum) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lsum) l'))
    (hold : (t = 0 ∧ l.isFinite = true ∧ l.toReal = 0) ∨ (0 < t ∧ MInv R t m ∧ LInv R t m l)) :
    LInv R (t + 1) m' l' := by
  have hpsP0 := psiPr_nonneg hR
  have hpsP1 := psiPr_le hR
  have hpsL0 := psiLr_nonneg hR
  have hPle : (R.P : ℝ) ≤ 256 := by exact_mod_cast hR.P_le
  have hlane := lanes_p hexp hni hR hM' hX hp
  set W1 := ∑ r, bw1 R m'.toReal t r with hW1
  have hW10 : 0 ≤ W1 := Finset.sum_nonneg fun r _ => bw1_nonneg _ _ _ _
  have hW1le : W1 ≤ 512 := by
    have := Finset.sum_le_sum fun r (_ : r ∈ Finset.univ) => (hlane r).2.2.1
    rw [sum_fin_const] at this
    nlinarith
  -- the sum of `p`
  have hpa : ∀ r, |(p r).toReal| ≤ (1 + psiPr R) * bw1 R m'.toReal t r + 1 / 2 ^ 140 := fun r => by
    have := abs_sub_abs_le_abs_sub (p r).toReal (bw1 R m'.toReal t r)
    rw [abs_of_nonneg (bw1_nonneg _ _ _ _)] at this
    have := (hlane r).2.1; linarith
  have hsA : ∑ r, |(p r).toReal| ≤ (1 + psiPr R) * W1 + R.P / 2 ^ 140 := by
    calc ∑ r, |(p r).toReal| ≤ ∑ r, ((1 + psiPr R) * bw1 R m'.toReal t r + 1 / 2 ^ 140) :=
          Finset.sum_le_sum fun r _ => hpa r
      _ = (1 + psiPr R) * W1 + R.P / 2 ^ 140 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_fin_const]; ring
  have hsP : |∑ r, (p r).toReal - W1| ≤ psiPr R * W1 + R.P / 2 ^ 140 := by
    rw [hW1, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ r, |(p r).toReal - bw1 R m'.toReal t r| ≤ ∑ r, (psiPr R * bw1 R m'.toReal t r + 1 / 2 ^ 140) :=
          Finset.sum_le_sum fun r _ => (hlane r).2.1
      _ = psiPr R * W1 + R.P / 2 ^ 140 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_fin_const]; ring
  have hsA1 : ∑ r, |(p r).toReal| ≤ 2 ^ 100 := by
    have : (1 + psiPr R) * W1 ≤ 2 * 512 := mul_le_mul (by linarith) hW1le hW10 (by norm_num)
    have : (R.P : ℝ) / 2 ^ 140 ≤ 256 / 2 ^ 140 := div_le_div_of_nonneg_right hPle (by norm_num)
    norm_num at this ⊢; linarith
  obtain ⟨hlsf, hlse⟩ := laneSum_err hR.P_le hsum (fun r => (hlane r).1) hsA1
  set KL := ((1 : ℝ) + 1 / 2 ^ 24) ^ (R.P - 1) with hKL
  have hKL1 : 1 ≤ KL := one_le_pow₀ (by norm_num)
  have hKL2 : KL ≤ 1 + 1 / 2 ^ 12 := pow255_le' hR.P_le
  have hn1 : ((R.P - 1 : ℕ) : ℝ) ≤ 256 := by exact_mod_cast (show R.P - 1 ≤ 256 by have := hR.P_le; omega)
  have hcn : |lsum.toReal - W1| ≤ psiLr R * W1 + 1 / 2 ^ 130 := by
    have t1 := abs_sub_le lsum.toReal (∑ r, (p r).toReal) W1
    have k1 : (KL - 1) * ∑ r, |(p r).toReal| ≤ (KL - 1) * ((1 + psiPr R) * W1 + R.P / 2 ^ 140) :=
      mul_le_mul_of_nonneg_left hsA (by linarith)
    have k2 : ((R.P - 1 : ℕ) : ℝ) / 2 ^ 150 * KL ≤ 256 / 2 ^ 150 * (1 + 1 / 2 ^ 12) :=
      mul_le_mul (div_le_div_of_nonneg_right hn1 (by norm_num)) hKL2 (by linarith) (by positivity)
    have k3 : (KL - 1) * (R.P / 2 ^ 140) ≤ 1 / 2 ^ 12 * (256 / 2 ^ 140) :=
      mul_le_mul (by linarith) (div_le_div_of_nonneg_right hPle (by norm_num)) (by positivity) (by norm_num)
    have k4 : (R.P : ℝ) / 2 ^ 140 ≤ 256 / 2 ^ 140 := div_le_div_of_nonneg_right hPle (by norm_num)
    have e : (KL - 1) * ((1 + psiPr R) * W1 + R.P / 2 ^ 140) + psiPr R * W1 =
        psiLr R * W1 + (KL - 1) * (R.P / 2 ^ 140) := by unfold psiLr; rw [← hKL]; ring
    norm_num at k2 k3 k4 ⊢
    linarith
  have hTn : |lsum.toReal| ≤ (1 + psiLr R) * W1 + 1 / 2 ^ 130 := by
    have := abs_sub_abs_le_abs_sub lsum.toReal W1
    rw [abs_of_nonneg hW10] at this; linarith
  -- `l_i` and `alpha` are finite and bounded
  have hlb : l.isFinite = true ∧ |l.toReal| ≤ 2 ^ 26 := by
    rcases hold with ⟨-, hf, hz⟩ | ⟨ht0, hmi, hL⟩
    · exact ⟨hf, by rw [hz]; norm_num⟩
    · exact ⟨hL.1, hL.bound hR (by omega) hmi⟩
  have halb := alpha_le_two hexp hni hR hM' hin ha (fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi)
  -- line 66: `l_i * alpha + l_ij`
  have hroom : |l.toReal * al.toReal| + |lsum.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]
    have := mul_le_mul hlb.2 halb.2 (abs_nonneg _) (by norm_num)
    have : (1 + psiLr R) * W1 ≤ 2 * 512 := mul_le_mul (by linarith [psiLr_le hR]) hW1le hW10 (by norm_num)
    norm_num at hTn ⊢
    linarith
  obtain ⟨hl'f, hl'e⟩ := f32_mulAdd hl' hlb.1 halb.1 hlsf hroom
  have hcomb : |l'.toReal - (l.toReal * al.toReal + lsum.toReal)| ≤
      3 / 2 ^ 24 * (|l.toReal * al.toReal| + |lsum.toReal|) + 4 / 2 ^ 150 := by linarith
  have hF2 : (1 + 3 / 2 ^ 24) * (1 + psiLr R) ≤ ffac R (psiLr R) (t + 1) m'.toReal :=
    ffac_ge hpsL0 (by positivity) stepR_ge (m_nonneg hM'.bound)
  have hF'1 : 1 ≤ ffac R (psiLr R) (t + 1) m'.toReal :=
    ffac_one_le hpsL0 (m_nonneg hM'.bound)
  refine ⟨hl'f, ?_⟩
  push_cast
  rcases hold with ⟨ht0, -, hl0⟩ | ⟨ht0, hmi, hL⟩
  · -- the first page: `l_i = +0`
    subst ht0
    have key := acc_step (a := l.toReal) (S := 0) (Mg := 0) (Ab := 0) (F := 1) (lam := 0) (Phi := 1)
      (tau1 := 0) (Mn := W1) (An := 1 / 2 ^ 130) (psi := psiLr R) (rho := 3 / 2 ^ 24)
      (tau2 := 4 / 2 ^ 150) (F' := ffac R (psiLr R) (0 + 1) m'.toReal)
      le_rfl (by simp) le_rfl (by rw [hl0]; simp) le_rfl zero_le_one le_rfl (by rw [hl0]; simp)
      (abs_of_nonneg hW10).le hpsL0 (by positivity) hcn hTn (by positivity) le_rfl (by positivity) hcomb
      (by have := mul_le_mul_of_nonneg_left (by linarith : (1 : ℝ) ≤ 1 + psiLr R)
            (by norm_num : (0 : ℝ) ≤ 1 + 3 / 2 ^ 24); linarith) hF2
    have hF0 : 0 ≤ ffac R (psiLr R) (0 + 1) m'.toReal := by linarith
    have : 1 / 2 ^ 130 + 4 / 2 ^ 150 ≤ (absLr : ℝ) := by unfold absLr; norm_num
    have k := mul_le_mul_of_nonneg_left this hF0
    rw [psum_step1 R 0 m'.toReal 0, psum_zero, ← hW1]
    simp only [mul_zero, zero_add, add_zero, Nat.cast_zero, one_mul] at key k ⊢
    linarith
  · obtain ⟨-, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    have hl1 : Real.exp (m.toReal - m'.toReal) ≤ 1 := by
      rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr (by linarith)
    have hl0 := Real.exp_pos (m.toReal - m'.toReal)
    have hd0 := dAl_nonneg hle
    have hW0 : 0 ≤ psum R t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
    have hF1 := ffac_one_le (t := t) hpsL0 (m_nonneg hmi.bound)
    have ha1 : |l.toReal * al.toReal - Real.exp (m.toReal - m'.toReal) * l.toReal| ≤
        ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (Real.exp (m.toReal - m'.toReal) * |l.toReal|) +
          |l.toReal| / 2 ^ 140 := by
      rw [show l.toReal * al.toReal - Real.exp (m.toReal - m'.toReal) * l.toReal =
        l.toReal * (al.toReal - Real.exp (m.toReal - m'.toReal)) by ring, abs_mul]
      have := mul_le_mul_of_nonneg_left hae (abs_nonneg l.toReal)
      have e : |l.toReal| * (((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) *
          Real.exp (m.toReal - m'.toReal) + 1 / 2 ^ 140) =
          ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (Real.exp (m.toReal - m'.toReal) * |l.toReal|) +
            |l.toReal| / 2 ^ 140 := by ring
      linarith
    have hFs := ffac_step (R := R) (psi := psiLr R) (t := t) hpsL0 hle
    have key := acc_step (a := l.toReal) (S := psum R t (wt R m.toReal)) (Mg := psum R t (wt R m.toReal))
      (Ab := t * absLr) (F := ffac R (psiLr R) t m.toReal)
      (lam := Real.exp (m.toReal - m'.toReal)) (Phi := (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22))
      (tau1 := |l.toReal| / 2 ^ 140)
      (Mn := W1) (An := 1 / 2 ^ 130) (psi := psiLr R) (rho := 3 / 2 ^ 24)
      (tau2 := 4 / 2 ^ 150) (F' := ffac R (psiLr R) (t + 1) m'.toReal)
      hF1 (abs_of_nonneg hW0).le (by unfold absLr; positivity) hL.2 hl0.le hl1
      (one_le_mul_of_one_le_of_one_le (by linarith) (by norm_num)) ha1
      (abs_of_nonneg hW10).le hpsL0 (by positivity) hcn hTn (by positivity) (by positivity) (by positivity) hcomb
      (by unfold dAl; exact hFs) hF2
    rw [psum_step1 R m.toReal m'.toReal t, ← hW1]
    have hlb2 := hlb.2
    have : 1 / 2 ^ 130 + |l.toReal| / 2 ^ 140 + 4 / 2 ^ 150 ≤ (absLr : ℝ) := by
      unfold absLr
      have := div_le_div_of_nonneg_right hlb2 (by norm_num : (0 : ℝ) ≤ 2 ^ 140)
      norm_num at this ⊢; linarith
    have hsum' : (t : ℝ) * absLr + 1 / 2 ^ 130 + |l.toReal| / 2 ^ 140 + 4 / 2 ^ 150 ≤ (t + 1) * absLr := by
      linarith
    have := mul_le_mul_of_nonneg_left hsum' (by linarith : (0 : ℝ) ≤ ffac R (psiLr R) (t + 1) m'.toReal)
    linarith

/-! ## `acc` -/

/-- **One page for an element of `acc`** (lines 64–68): `p` the page's `p` (line 65), `vt` its
column of `v` (a token's value `ve`, `0` on a masked lane), `pv` the rounded products
`p[:, None] * v`, `vsum` their column sum (a tree of fusable additions), `a'` the new element
(`tl.mul_add`). Before the first page the element is `+0`. -/
theorem astep {P : Platform} (hexp : Rules.Num.ExpF32 P) (hni : Rules.Num.ExpF32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 2 ^ 16) {fusion fusion' : Bool} {ve : ℕ → ℝ}
    (hve : ∀ j, |ve j| ≤ 2 ^ 15)
    {m m' dm al a vsum a' : F32} {x p vt pv : Fin R.P → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x)
    (hp : ∀ r, P.math .exp .binary32 (x r) (p r))
    (hin : AlphaIn t m m' dm) (ha : P.math .exp .binary32 dm al)
    (hvt : ∀ r, (vt r).isFinite = true ∧
      (R.P * t + r < R.n → (vt r).toReal = ve (R.P * t + r)) ∧ (¬ R.P * t + r < R.n → (vt r).toReal = 0))
    (hpv : ∀ r, FloatV.NaNEq (FloatV.mul (p r) (vt r)) (pv r))
    (hsum : ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (AddRel fusion') (laneSeq pv) vsum)
    (ha' : FloatV.NaNEq (FloatV.add (FloatV.mul a al) vsum) a' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma a al vsum) a'))
    (hold : (t = 0 ∧ a.isFinite = true ∧ a.toReal = 0) ∨ (0 < t ∧ MInv R t m ∧ AInv R ve t m a)) :
    AInv R ve (t + 1) m' a' := by
  have hpsP0 := psiPr_nonneg hR
  have hpsP1 := psiPr_le hR
  have hpsA0 := psiAr_nonneg hR
  have hpsA1 := psiAr_le hR
  have hPle : (R.P : ℝ) ≤ 256 := by exact_mod_cast hR.P_le
  have hlane := lanes_p hexp hni hR hM' hX hp
  set Mn := ∑ r, bwg R m'.toReal t (fun j => |ve j|) r with hMnd
  set Sn := ∑ r, bwg R m'.toReal t ve r with hSnd
  have hMn0 : 0 ≤ Mn := Finset.sum_nonneg fun r _ => bwg_abs_nonneg _ _ _ _ _
  have hSn : |Sn| ≤ Mn := (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun r _ => abs_bwg_le _ _ _ _ _)
  -- the lanes
  have hl : ∀ r, (pv r).isFinite = true ∧
      |(pv r).toReal - bwg R m'.toReal t ve r| ≤
        ((1 + 1 / 2 ^ 24) * (1 + psiPr R) - 1) * bwg R m'.toReal t (fun j => |ve j|) r + 1 / 2 ^ 120 ∧
      |(pv r).toReal| ≤ (1 + 1 / 2 ^ 24) * (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r + 1 / 2 ^ 120 ∧
      bwg R m'.toReal t (fun j => |ve j|) r ≤ 2 ^ 16 := by
    intro r
    obtain ⟨hpf, hpe, hbw2, hpm⟩ := hlane r
    obtain ⟨hvf, hvv, hvm⟩ := hvt r
    -- `p·v` against `bw·ve`
    have hpvx : |(p r).toReal * (vt r).toReal - bwg R m'.toReal t ve r| ≤
        psiPr R * bwg R m'.toReal t (fun j => |ve j|) r + 2 ^ 15 / 2 ^ 140 ∧
        |(p r).toReal * (vt r).toReal| ≤ (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r + 2 ^ 15 / 2 ^ 140 ∧
        bwg R m'.toReal t (fun j => |ve j|) r ≤ 2 ^ 16 := by
      unfold bwg; unfold bw1 at hpe hbw2
      by_cases hr : R.P * t + r < R.n
      · simp only [hr, ite_true] at hpe hbw2 ⊢
        rw [hvv hr]
        have hv := hve (R.P * t + r)
        have hva := abs_nonneg (ve (R.P * t + r))
        have hG := wt_pos R m'.toReal (R.P * t + r)
        refine ⟨?_, ?_, ?_⟩
        · rw [← sub_mul, abs_mul]
          have := mul_le_mul hpe hv hva (by positivity)
          have := mul_le_mul_of_nonneg_right hpe hva
          nlinarith
        · rw [abs_mul]
          have h1 := abs_sub_abs_le_abs_sub (p r).toReal (wt R m'.toReal (R.P * t + r))
          rw [abs_of_pos hG] at h1
          have h2 : |(p r).toReal| ≤ (1 + psiPr R) * wt R m'.toReal (R.P * t + r) + 1 / 2 ^ 140 := by linarith
          have := mul_le_mul_of_nonneg_right h2 hva
          have : 1 / 2 ^ 140 * |ve (R.P * t + r)| ≤ 1 / 2 ^ 140 * 2 ^ 15 := mul_le_mul_of_nonneg_left hv (by norm_num)
          nlinarith
        · have := mul_le_mul hbw2 hv hva (by norm_num); linarith
      · simp only [hr, ite_false]
        rw [hpm hr]
        refine ⟨?_, ?_, by norm_num⟩ <;> norm_num
    obtain ⟨hx1, hx2, hx3⟩ := hpvx
    have hroom : |(p r).toReal * (vt r).toReal| ≤ 2 ^ 120 := by
      have : (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r ≤ 2 * 2 ^ 16 :=
        mul_le_mul (by linarith) hx3 (bwg_abs_nonneg _ _ _ _ _) (by norm_num)
      norm_num at hx2 ⊢; linarith
    obtain ⟨hpvf, hpve⟩ := f32_mul (hpv r) hpf hvf hroom
    refine ⟨hpvf, ?_, ?_, hx3⟩
    · have := abs_sub_le (pv r).toReal ((p r).toReal * (vt r).toReal) (bwg R m'.toReal t ve r)
      have k := div_le_div_of_nonneg_right hx2 (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
      have e : ((1 + 1 / 2 ^ 24) * (1 + psiPr R) - 1) * bwg R m'.toReal t (fun j => |ve j|) r =
          psiPr R * bwg R m'.toReal t (fun j => |ve j|) r +
            (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r / 2 ^ 24 := by ring
      have e2 : ((1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r + 2 ^ 15 / 2 ^ 140) / 2 ^ 24 =
          (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r / 2 ^ 24 + 2 ^ 15 / 2 ^ 140 / 2 ^ 24 := by ring
      have n : (2 : ℝ) ^ 15 / 2 ^ 140 / 2 ^ 24 + 1 / 2 ^ 150 + 2 ^ 15 / 2 ^ 140 ≤ 1 / 2 ^ 120 := by norm_num
      linarith
    · have := abs_sub_abs_le_abs_sub (pv r).toReal ((p r).toReal * (vt r).toReal)
      have k := div_le_div_of_nonneg_right hx2 (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
      have e : (1 + 1 / 2 ^ 24) * (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r =
          (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r +
            (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r / 2 ^ 24 := by ring
      have e2 : ((1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r + 2 ^ 15 / 2 ^ 140) / 2 ^ 24 =
          (1 + psiPr R) * bwg R m'.toReal t (fun j => |ve j|) r / 2 ^ 24 + 2 ^ 15 / 2 ^ 140 / 2 ^ 24 := by ring
      have n : (2 : ℝ) ^ 15 / 2 ^ 140 / 2 ^ 24 + 1 / 2 ^ 150 + 2 ^ 15 / 2 ^ 140 ≤ 1 / 2 ^ 120 := by norm_num
      linarith
  have hMnle : Mn ≤ 2 ^ 24 := by
    have := Finset.sum_le_sum fun r (_ : r ∈ Finset.univ) => (hl r).2.2.2
    rw [sum_fin_const] at this
    nlinarith
  set c1 := (1 + 1 / 2 ^ 24) * (1 + psiPr R) with hc1
  have hc10 : 1 ≤ c1 := by rw [hc1]; nlinarith
  have hc1le : c1 ≤ 2 := by rw [hc1]; nlinarith
  have hsA : ∑ r, |(pv r).toReal| ≤ c1 * Mn + R.P / 2 ^ 120 := by
    calc ∑ r, |(pv r).toReal| ≤ ∑ r, (c1 * bwg R m'.toReal t (fun j => |ve j|) r + 1 / 2 ^ 120) :=
          Finset.sum_le_sum fun r _ => (hl r).2.2.1
      _ = c1 * Mn + R.P / 2 ^ 120 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_fin_const]; ring
  have hsP : |∑ r, (pv r).toReal - Sn| ≤ (c1 - 1) * Mn + R.P / 2 ^ 120 := by
    rw [hSnd, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ r, |(pv r).toReal - bwg R m'.toReal t ve r|
        ≤ ∑ r, ((c1 - 1) * bwg R m'.toReal t (fun j => |ve j|) r + 1 / 2 ^ 120) :=
          Finset.sum_le_sum fun r _ => (hl r).2.1
      _ = (c1 - 1) * Mn + R.P / 2 ^ 120 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_fin_const]; ring
  have hsA1 : ∑ r, |(pv r).toReal| ≤ 2 ^ 100 := by
    have : c1 * Mn ≤ 2 * 2 ^ 24 := mul_le_mul hc1le hMnle hMn0 (by norm_num)
    have : (R.P : ℝ) / 2 ^ 120 ≤ 256 / 2 ^ 120 := div_le_div_of_nonneg_right hPle (by norm_num)
    norm_num at this ⊢; linarith
  obtain ⟨hvsf, hvse⟩ := laneSumc_err hR.P_le hsum (fun r => (hl r).1) hsA1
  set KA := ((1 : ℝ) + 3 / 2 ^ 24) ^ (R.P - 1) with hKA
  have hKA1 : 1 ≤ KA := one_le_pow₀ (by norm_num)
  have hKA2 : KA ≤ 1 + 1 / 2 ^ 12 := pow255_le hR.P_le
  have hn1 : ((R.P - 1 : ℕ) : ℝ) ≤ 256 := by exact_mod_cast (show R.P - 1 ≤ 256 by have := hR.P_le; omega)
  have hcn : |vsum.toReal - Sn| ≤ psiAr R * Mn + 1 / 2 ^ 110 := by
    have t1 := abs_sub_le vsum.toReal (∑ r, (pv r).toReal) Sn
    have k1 : (KA - 1) * ∑ r, |(pv r).toReal| ≤ (KA - 1) * (c1 * Mn + R.P / 2 ^ 120) :=
      mul_le_mul_of_nonneg_left hsA (by linarith)
    have k2 : 4 * ((R.P - 1 : ℕ) : ℝ) / 2 ^ 150 * KA ≤ 4 * 256 / 2 ^ 150 * (1 + 1 / 2 ^ 12) := by
      apply mul_le_mul _ hKA2 (by linarith) (by positivity)
      apply div_le_div_of_nonneg_right _ (by norm_num); linarith
    have k3 : (KA - 1) * (R.P / 2 ^ 120) ≤ 1 / 2 ^ 12 * (256 / 2 ^ 120) :=
      mul_le_mul (by linarith) (div_le_div_of_nonneg_right hPle (by norm_num)) (by positivity) (by norm_num)
    have k4 : (R.P : ℝ) / 2 ^ 120 ≤ 256 / 2 ^ 120 := div_le_div_of_nonneg_right hPle (by norm_num)
    have e : (KA - 1) * (c1 * Mn + R.P / 2 ^ 120) + (c1 - 1) * Mn =
        psiAr R * Mn + (KA - 1) * (R.P / 2 ^ 120) := by unfold psiAr; rw [← hKA, hc1]; ring
    norm_num at k2 k3 k4 ⊢
    linarith
  have hTn : |vsum.toReal| ≤ (1 + psiAr R) * Mn + 1 / 2 ^ 110 := by
    have := abs_sub_abs_le_abs_sub vsum.toReal Sn
    linarith
  -- `acc` and `alpha` are finite and bounded
  have hab : a.isFinite = true ∧ |a.toReal| ≤ 2 ^ 41 := by
    rcases hold with ⟨-, hf, hz⟩ | ⟨ht0, hmi, hA⟩
    · exact ⟨hf, by rw [hz]; norm_num⟩
    · exact ⟨hA.1, hA.bound hR (by omega) hve hmi⟩
  have halb := alpha_le_two hexp hni hR hM' hin ha (fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi)
  -- line 68: `acc * alpha + …`
  have hroom : |a.toReal * al.toReal| + |vsum.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]
    have := mul_le_mul hab.2 halb.2 (abs_nonneg _) (by norm_num)
    have : (1 + psiAr R) * Mn ≤ 2 * 2 ^ 24 := mul_le_mul (by linarith) hMnle hMn0 (by norm_num)
    norm_num at hTn ⊢
    linarith
  obtain ⟨ha'f, ha'e⟩ := f32_mulAdd ha' hab.1 halb.1 hvsf hroom
  have hcomb : |a'.toReal - (a.toReal * al.toReal + vsum.toReal)| ≤
      3 / 2 ^ 24 * (|a.toReal * al.toReal| + |vsum.toReal|) + 4 / 2 ^ 150 := by linarith
  have hF2 : (1 + 3 / 2 ^ 24) * (1 + psiAr R) ≤ ffac R (psiAr R) (t + 1) m'.toReal :=
    ffac_ge hpsA0 (by positivity) stepR_ge (m_nonneg hM'.bound)
  have hF'1 : 1 ≤ ffac R (psiAr R) (t + 1) m'.toReal :=
    ffac_one_le hpsA0 (m_nonneg hM'.bound)
  refine ⟨ha'f, ?_⟩
  push_cast
  rw [psum_stepg R m.toReal m'.toReal t ve, psum_stepg R m.toReal m'.toReal t (fun j => |ve j|), ← hSnd, ← hMnd]
  rcases hold with ⟨ht0, -, ha0⟩ | ⟨ht0, hmi, hA⟩
  · -- the first page: `acc = +0`
    subst ht0
    have key := acc_step (a := a.toReal) (S := 0) (Mg := 0) (Ab := 0) (F := 1) (lam := 0) (Phi := 1)
      (tau1 := 0) (Mn := Mn) (An := 1 / 2 ^ 110) (psi := psiAr R) (rho := 3 / 2 ^ 24)
      (tau2 := 4 / 2 ^ 150) (F' := ffac R (psiAr R) (0 + 1) m'.toReal)
      le_rfl (by simp) le_rfl (by rw [ha0]; simp) le_rfl zero_le_one le_rfl (by rw [ha0]; simp)
      hSn hpsA0 (by positivity) hcn hTn (by positivity) le_rfl (by positivity) hcomb
      (by have := mul_le_mul_of_nonneg_left (by linarith : (1 : ℝ) ≤ 1 + psiAr R)
            (by norm_num : (0 : ℝ) ≤ 1 + 3 / 2 ^ 24); linarith) hF2
    have hF0 : 0 ≤ ffac R (psiAr R) (0 + 1) m'.toReal := by linarith
    have : 1 / 2 ^ 110 + 4 / 2 ^ 150 ≤ (absAr : ℝ) := by unfold absAr; norm_num
    have k := mul_le_mul_of_nonneg_left this hF0
    simp only [psum_zero, mul_zero, zero_add, add_zero, Nat.cast_zero, one_mul] at key k ⊢
    linarith
  · obtain ⟨-, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    have hl1 : Real.exp (m.toReal - m'.toReal) ≤ 1 := by
      rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr (by linarith)
    have hl0 := Real.exp_pos (m.toReal - m'.toReal)
    have hd0 := dAl_nonneg hle
    have hMg0 : 0 ≤ psum R t (fun j => wt R m.toReal j * |ve j|) :=
      psum_nonneg fun j => mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
    have hF1 := ffac_one_le (t := t) hpsA0 (m_nonneg hmi.bound)
    have ha1 : |a.toReal * al.toReal - Real.exp (m.toReal - m'.toReal) * a.toReal| ≤
        ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (Real.exp (m.toReal - m'.toReal) * |a.toReal|) +
          |a.toReal| / 2 ^ 140 := by
      rw [show a.toReal * al.toReal - Real.exp (m.toReal - m'.toReal) * a.toReal =
        a.toReal * (al.toReal - Real.exp (m.toReal - m'.toReal)) by ring, abs_mul]
      have := mul_le_mul_of_nonneg_left hae (abs_nonneg a.toReal)
      have e : |a.toReal| * (((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) *
          Real.exp (m.toReal - m'.toReal) + 1 / 2 ^ 140) =
          ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (Real.exp (m.toReal - m'.toReal) * |a.toReal|) +
            |a.toReal| / 2 ^ 140 := by ring
      linarith
    have hFs := ffac_step (R := R) (psi := psiAr R) (t := t) hpsA0 hle
    have key := acc_step (a := a.toReal) (S := psum R t (fun j => wt R m.toReal j * ve j))
      (Mg := psum R t (fun j => wt R m.toReal j * |ve j|))
      (Ab := t * absAr) (F := ffac R (psiAr R) t m.toReal)
      (lam := Real.exp (m.toReal - m'.toReal)) (Phi := (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22))
      (tau1 := |a.toReal| / 2 ^ 140)
      (Mn := Mn) (An := 1 / 2 ^ 110) (psi := psiAr R) (rho := 3 / 2 ^ 24)
      (tau2 := 4 / 2 ^ 150) (F' := ffac R (psiAr R) (t + 1) m'.toReal)
      hF1 (psum_abs_wt R m.toReal t ve) (by unfold absAr; positivity) hA.2 hl0.le hl1
      (one_le_mul_of_one_le_of_one_le (by linarith) (by norm_num)) ha1
      hSn hpsA0 (by positivity) hcn hTn (by positivity) (by positivity) (by positivity) hcomb
      (by unfold dAl; exact hFs) hF2
    have hab2 := hab.2
    have : 1 / 2 ^ 110 + |a.toReal| / 2 ^ 140 + 4 / 2 ^ 150 ≤ (absAr : ℝ) := by
      unfold absAr
      have := div_le_div_of_nonneg_right hab2 (by norm_num : (0 : ℝ) ≤ 2 ^ 140)
      norm_num at this ⊢; linarith
    have hsum' : (t : ℝ) * absAr + 1 / 2 ^ 110 + |a.toReal| / 2 ^ 140 + 4 / 2 ^ 150 ≤ (t + 1) * absAr := by
      linarith
    have := mul_le_mul_of_nonneg_left hsum' (by linarith : (0 : ℝ) ≤ ffac R (psiAr R) (t + 1) m'.toReal)
    linarith

end Kernels.PagedAttention
