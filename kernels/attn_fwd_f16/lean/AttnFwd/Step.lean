import AttnFwd.RowAcc

/-!
# `attn_fwd_f16`: one row, one block of the loop for `l_i` and `acc`

* `LInv`, `AInv`: after `t ≥ 1` blocks with running maximum `m`, `l_i` and an element of the
  row's `acc` (column values `ve`) are within a relative error `ffac − 1` and an absolute error of
  `Σ wt` and `Σ wt·ve` over the attended keys so far (for `acc`: `om0` per key times `|ve|`,
  `kapAr` per unit of weight, `absAr` per block).
* `lstep` (lines 82–85, 106) and `astep` (lines 82–84, 95, 101–103): one block keeps them.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

theorem z24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
theorem z150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num

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

/-! ## Block terms -/

/-- Lane `jj`'s weight in block `t` (`0` for a masked key). -/
noncomputable def bw1 (R : RowCtx) (m' : ℝ) (t : ℕ) (jj : Fin 64) : ℝ :=
  if R.al (64 * t + jj) = true then wt R m' (64 * t + jj) else 0

/-- Lane `jj`'s weight times `g` in block `t`. -/
noncomputable def bwg (R : RowCtx) (m' : ℝ) (t : ℕ) (g : ℕ → ℝ) (jj : Fin 64) : ℝ :=
  if R.al (64 * t + jj) = true then wt R m' (64 * t + jj) * g (64 * t + jj) else 0

/-- Lane `jj`'s `g` in block `t`, if attended. -/
noncomputable def bg (R : RowCtx) (t : ℕ) (g : ℕ → ℝ) (jj : Fin 64) : ℝ :=
  if R.al (64 * t + jj) = true then g (64 * t + jj) else 0

theorem psum_step1 (R : RowCtx) (m m' : ℝ) (t : ℕ) :
    psum R.al (t + 1) (wt R m') = (2 : ℝ) ^ (m - m') * psum R.al t (wt R m) + ∑ jj, bw1 R m' t jj := by
  have := psum_step R m m' t (fun _ => 1)
  simp only [mul_one] at this
  exact this

theorem psum_stepg (R : RowCtx) (m m' : ℝ) (t : ℕ) (g : ℕ → ℝ) :
    psum R.al (t + 1) (fun j => wt R m' j * g j) =
      (2 : ℝ) ^ (m - m') * psum R.al t (fun j => wt R m j * g j) + ∑ jj, bwg R m' t g jj :=
  psum_step R m m' t g

theorem psum_stepv (R : RowCtx) (t : ℕ) (g : ℕ → ℝ) :
    psum R.al (t + 1) g = psum R.al t g + ∑ jj, bg R t g jj := psum_succ R.al t g

theorem bw1_nonneg (R : RowCtx) (m' : ℝ) (t : ℕ) (jj : Fin 64) : 0 ≤ bw1 R m' t jj := by
  unfold bw1; split_ifs
  · exact (wt_pos _ _ _).le
  · exact le_rfl

/-! ## The invariants -/

/-- `l_i` after `t` blocks (`t ≥ 1`) with running maximum `m`. -/
def LInv (R : RowCtx) (t : ℕ) (m l : F32) : Prop :=
  l.isFinite = true ∧
    |l.toReal - psum R.al t (wt R m.toReal)| ≤
      (ffac R (psiLr R.E) stepLr t m.toReal - 1) * psum R.al t (wt R m.toReal) +
        ffac R (psiLr R.E) stepLr t m.toReal * (t * absLr)

/-- An element of the row's `acc` after `t` blocks (`t ≥ 1`), `ve` the values of its column of `v`. -/
def AInv (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) (m a : F32) : Prop :=
  a.isFinite = true ∧
    |a.toReal - psum R.al t (fun j => wt R m.toReal j * ve j)| ≤
      (ffac R (psi16r R.E) stepAr t m.toReal - 1) * psum R.al t (fun j => wt R m.toReal j * |ve j|) +
        ffac R (psi16r R.E) stepAr t m.toReal *
          (om0 * psum R.al t (fun j => |ve j|) + kapAr * psum R.al t (wt R m.toReal) + t * absAr)

theorem stepLr_one : 1 ≤ stepLr := le_trans (by norm_num) stepLr_ge

theorem stepAr_one : 1 ≤ stepAr := le_trans (by norm_num) stepAr_ge

theorem m_nonneg {R : RowCtx} {m : ℝ} (h : |m| ≤ R.Y + R.E) : 0 ≤ m + R.Y + R.E := by
  have := neg_abs_le m; linarith

/-- The sum of weights over `t ≤ 1024` blocks: at most `2^17`. -/
theorem psum_wt1_le {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) {m : F32} (hm : MInv R t m) :
    psum R.al t (wt R m.toReal) ≤ 2 ^ 17 := by
  have h := psum_wt_le hR hm (g := fun _ => 1) (fun _ => zero_le_one) (fun _ => le_rfl)
  simp only [mul_one] at h
  have : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  nlinarith

theorem LInv.bound {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) {m l : F32} (hm : MInv R t m)
    (hl : LInv R t m l) : |l.toReal| ≤ 2 ^ 19 := by
  have hF := ffac_le_two hR (psiLr_le hR.E_nonneg hR.E_le) stepLr_one stepLr_le ht hm.bound
  have hF1 := ffac_one_le (t := t) (psiLr_nonneg hR.E_nonneg) stepLr_one (m_nonneg hm.bound)
  have hW := psum_wt1_le hR ht hm
  have hW0 : 0 ≤ psum R.al t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
  have htA : (t : ℝ) * absLr ≤ 1 := by
    have : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
    unfold absLr; nlinarith
  have htA0 : 0 ≤ (t : ℝ) * absLr := by unfold absLr; positivity
  obtain ⟨-, he⟩ := hl
  have := abs_sub_abs_le_abs_sub l.toReal (psum R.al t (wt R m.toReal))
  rw [abs_of_nonneg hW0] at this
  have k1 := mul_le_mul_of_nonneg_right (by linarith : ffac R (psiLr R.E) stepLr t m.toReal - 1 ≤ 1) hW0
  have k2 := mul_le_mul_of_nonneg_right hF htA0
  linarith

/-! ## `l_i` -/

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
    (hold : (t = 0 ∧ l.isFinite = true ∧ |l.toReal| ≤ 1) ∨ (0 < t ∧ MInv R t m ∧ LInv R t m l)) :
    LInv R (t + 1) m' l' := by
  have hE0 := hR.E_nonneg
  have hE8 := hR.E_le
  have hpsP0 := psiPr_nonneg hE0
  have hpsP1 := psiPr_le hE8
  have hpsL0 := psiLr_nonneg hE0
  -- the lanes
  have hlane : ∀ jj, (p jj).isFinite = true ∧
      |(p jj).toReal - bw1 R m'.toReal t jj| ≤ psiPr R.E * bw1 R m'.toReal t jj + 1 / 2 ^ 125 ∧
      |(p jj).toReal| ≤ (1 + psiPr R.E) * bw1 R m'.toReal t jj + 1 / 2 ^ 125 ∧
      bw1 R m'.toReal t jj ≤ 0 * bw1 R m'.toReal t jj + 2 := by
    intro jj
    obtain ⟨hxf, hxa, hxm⟩ := hX jj
    obtain ⟨hpf, hpa, hpm⟩ := lane_p hexp hR hxf hxa hxm (fun h => hM'.above _ (by omega) h) (hp jj)
    refine ⟨hpf, ?_⟩
    unfold bw1
    cases hal : R.al (64 * t + jj)
    · have := hpm hal
      simp only [Bool.false_eq_true, ↓reduceIte, sub_zero, mul_zero, zero_add]
      refine ⟨by linarith, by linarith, by norm_num⟩
    · have h1 := hpa hal
      have hGp := wt_pos R m'.toReal (64 * t + jj)
      have hG2 := wt_le_two hR hM' (j := 64 * t + jj) (by omega) hal
      have := abs_sub_abs_le_abs_sub (p jj).toReal (wt R m'.toReal (64 * t + jj))
      rw [abs_of_pos hGp] at this
      simp only [↓reduceIte, zero_mul, zero_add]
      refine ⟨by linarith, by linarith, hG2⟩
  have hsP := lane_sum_err (fun jj => (hlane jj).2.1)
  have hsA := lane_sum_le (fun jj => (hlane jj).2.2.1)
  have hsG := lane_sum_le (fun jj => (hlane jj).2.2.2)
  have hSn0 : 0 ≤ ∑ jj, bw1 R m'.toReal t jj := Finset.sum_nonneg fun jj _ => bw1_nonneg _ _ _ _
  obtain ⟨hlijf, hlije⟩ := sum64_err hsum (fun jj => (hlane jj).1) (by nlinarith)
  -- `l_ij` against the block's exact weights
  have hcn : |lij.toReal - ∑ jj, bw1 R m'.toReal t jj| ≤
      psiLr R.E * ∑ jj, bw1 R m'.toReal t jj + 1 / 2 ^ 117 := by
    unfold psiLr
    have hK := pow63_le
    have hK1 : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ 63 := one_le_pow₀ (by norm_num)
    have := abs_sub_le lij.toReal (∑ jj, (p jj).toReal) (∑ jj, bw1 R m'.toReal t jj)
    generalize ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 = K at hK hK1 hlije ⊢
    have k1 := mul_le_mul_of_nonneg_left hsA (by linarith : (0 : ℝ) ≤ K - 1)
    have e : (K * (1 + psiPr R.E) - 1) * ∑ jj, bw1 R m'.toReal t jj =
        (K - 1) * ((1 + psiPr R.E) * ∑ jj, bw1 R m'.toReal t jj) + psiPr R.E * ∑ jj, bw1 R m'.toReal t jj := by
      ring
    rw [e]
    norm_num at k1 hlije hsP ⊢
    nlinarith
  have hTn : |lij.toReal| ≤ (1 + psiLr R.E) * ∑ jj, bw1 R m'.toReal t jj + 1 / 2 ^ 117 := by
    have := abs_sub_abs_le_abs_sub lij.toReal (∑ jj, bw1 R m'.toReal t jj)
    rw [abs_of_nonneg hSn0] at this; linarith
  -- `l_i` and `alpha` are finite and bounded
  have hlb : l.isFinite = true ∧ |l.toReal| ≤ 2 ^ 19 := by
    rcases hold with ⟨-, hf, hb⟩ | ⟨ht0, hmi, hL⟩
    · exact ⟨hf, by linarith⟩
    · exact ⟨hL.1, hL.bound hR (by omega) hmi⟩
  have halb : al.isFinite = true ∧ |al.toReal| ≤ 2 := by
    rcases Nat.eq_zero_or_pos t with h0 | ht0
    · subst h0; have := alpha_zero hni hin ha; exact ⟨this.1, by rw [this.2]; norm_num⟩
    · have hmi : MInv R t m := by
        rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
        · omega
        · exact hmi
      obtain ⟨haf, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
      refine ⟨haf, ?_⟩
      have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
      have hl0 := two_rpow_pos (m.toReal - m'.toReal)
      have hd : dAl m.toReal m'.toReal ≤ 1 / 2 ^ 12 := by
        unfold dAl; have := hmi.bound; have := hM'.bound; have := hR.Y_le
        have := neg_abs_le m.toReal; have := le_abs_self m'.toReal; linarith
      have hd0 : 0 ≤ dAl m.toReal m'.toReal := by
        unfold dAl; have : 0 ≤ m'.toReal - m.toReal := by linarith
        linarith [div_nonneg this (by norm_num : (0 : ℝ) ≤ 2 ^ 24), (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)]
      have := abs_sub_abs_le_abs_sub al.toReal ((2 : ℝ) ^ (m.toReal - m'.toReal))
      rw [abs_of_pos hl0] at this
      have k : ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 / 2 := by
        have : (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 ≤ 1 / 2 := by nlinarith
        have h0 : 0 ≤ (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 := by nlinarith
        nlinarith
      linarith
  -- line 106: `l_i * alpha + l_ij`
  have hroom : |l.toReal * al.toReal| + |lij.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]
    have := mul_le_mul hlb.2 halb.2 (abs_nonneg _) (by norm_num)
    have hpsL1 := psiLr_le hE0 hE8
    have : (1 + psiLr R.E) * ∑ jj, bw1 R m'.toReal t jj ≤ 2 * 128 :=
      mul_le_mul (by linarith) (by linarith) hSn0 (by norm_num)
    have : |lij.toReal| ≤ 2 ^ 100 := by linarith
    linarith
  obtain ⟨hl'f, hl'e⟩ := f32_mulAdd hl' hlb.1 halb.1 hlijf hroom
  have hcomb : |l'.toReal - (l.toReal * al.toReal + lij.toReal)| ≤
      3 / 2 ^ 24 * (|l.toReal * al.toReal| + |lij.toReal|) + 4 / 2 ^ 150 := by linarith
  have hF2 : (1 + 3 / 2 ^ 24) * (1 + psiLr R.E) ≤ ffac R (psiLr R.E) stepLr (t + 1) m'.toReal :=
    ffac_ge hpsL0 (by positivity) stepLr_ge (m_nonneg hM'.bound)
  have hF'1 : 1 ≤ ffac R (psiLr R.E) stepLr (t + 1) m'.toReal :=
    ffac_one_le hpsL0 stepLr_one (m_nonneg hM'.bound)
  refine ⟨hl'f, ?_⟩
  push_cast
  rcases hold with ⟨ht0, -, -⟩ | ⟨ht0, hmi, hL⟩
  · -- the first block: `alpha = 0`
    subst ht0
    have hal0 := (alpha_zero hni hin ha).2
    have key := acc_step (a := 0) (S := 0) (Mg := 0) (Ab := 0) (F := 1) (lam := 0) (Phi := 1) (tau1 := 0)
      (Mn := ∑ jj, bw1 R m'.toReal 0 jj) (An := 1 / 2 ^ 117) (psi := psiLr R.E) (rho := 3 / 2 ^ 24)
      (tau2 := 4 / 2 ^ 150) (F' := ffac R (psiLr R.E) stepLr (0 + 1) m'.toReal)
      le_rfl (by simp) le_rfl (by simp) le_rfl zero_le_one le_rfl (by rw [hal0]; simp)
      (abs_of_nonneg hSn0).le hpsL0 (by positivity) hcn hTn (by positivity) le_rfl (by positivity) hcomb
      (by nlinarith) hF2
    rw [psum_step1 R 0 m'.toReal 0, psum_zero]
    simp only [mul_zero, zero_add, add_zero, Nat.cast_zero] at key ⊢
    have : 1 / 2 ^ 117 + 4 / 2 ^ 150 ≤ (absLr : ℝ) := by unfold absLr; norm_num
    nlinarith
  · obtain ⟨-, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
    have hl0 := two_rpow_pos (m.toReal - m'.toReal)
    have hd0 : 0 ≤ dAl m.toReal m'.toReal := by
      unfold dAl; have : 0 ≤ m'.toReal - m.toReal := by linarith
      linarith [div_nonneg this (by norm_num : (0 : ℝ) ≤ 2 ^ 24), (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)]
    have hW0 : 0 ≤ psum R.al t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
    have hF1 := ffac_one_le (t := t) hpsL0 stepLr_one (m_nonneg hmi.bound)
    have ha1 : |l.toReal * al.toReal - (2 : ℝ) ^ (m.toReal - m'.toReal) * l.toReal| ≤
        ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * ((2 : ℝ) ^ (m.toReal - m'.toReal) * |l.toReal|) +
          |l.toReal| / 2 ^ 126 := by
      rw [show l.toReal * al.toReal - (2 : ℝ) ^ (m.toReal - m'.toReal) * l.toReal =
        l.toReal * (al.toReal - (2 : ℝ) ^ (m.toReal - m'.toReal)) by ring, abs_mul]
      have := mul_le_mul_of_nonneg_left hae (abs_nonneg l.toReal)
      have e : |l.toReal| * (((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) *
          (2 : ℝ) ^ (m.toReal - m'.toReal) + 1 / 2 ^ 126) =
          ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * ((2 : ℝ) ^ (m.toReal - m'.toReal) * |l.toReal|) +
            |l.toReal| / 2 ^ 126 := by ring
      linarith
    have hFs := ffac_step (R := R) (psi := psiLr R.E) (rho := 3 / 2 ^ 24) (B0 := 1 + 1 / 2 ^ 22) (t := t)
      hpsL0 (by positivity) (by positivity) hle
    have key := acc_step (a := l.toReal) (S := psum R.al t (wt R m.toReal)) (Mg := psum R.al t (wt R m.toReal))
      (Ab := t * absLr) (F := ffac R (psiLr R.E) stepLr t m.toReal)
      (lam := (2 : ℝ) ^ (m.toReal - m'.toReal)) (Phi := (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22))
      (tau1 := |l.toReal| / 2 ^ 126)
      (Mn := ∑ jj, bw1 R m'.toReal t jj) (An := 1 / 2 ^ 117) (psi := psiLr R.E) (rho := 3 / 2 ^ 24)
      (tau2 := 4 / 2 ^ 150) (F' := ffac R (psiLr R.E) stepLr (t + 1) m'.toReal)
      hF1 (abs_of_nonneg hW0).le (by unfold absLr; positivity) hL.2 hl0.le hl1 (by nlinarith) ha1
      (abs_of_nonneg hSn0).le hpsL0 (by positivity) hcn hTn (by positivity) (by positivity) (by positivity) hcomb
      hFs hF2
    rw [psum_step1 R m.toReal m'.toReal t]
    have hlb2 := hlb.2
    have : 1 / 2 ^ 117 + |l.toReal| / 2 ^ 126 + 4 / 2 ^ 150 ≤ (absLr : ℝ) := by
      unfold absLr; norm_num at hlb2 ⊢; linarith
    have hsum' : (t : ℝ) * absLr + 1 / 2 ^ 117 + |l.toReal| / 2 ^ 126 + 4 / 2 ^ 150 ≤ (t + 1) * absLr := by
      linarith
    have := mul_le_mul_of_nonneg_left hsum' (by linarith : (0 : ℝ) ≤ ffac R (psiLr R.E) stepLr (t + 1) m'.toReal)
    linarith

/-! ## `acc` -/

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

theorem psum_abs_wt (R : RowCtx) (m : ℝ) (t : ℕ) (ve : ℕ → ℝ) :
    |psum R.al t (fun j => wt R m j * ve j)| ≤ psum R.al t (fun j => wt R m j * |ve j|) := by
  refine (abs_psum_le _ _ _).trans (le_of_eq ?_)
  unfold psum
  refine Finset.sum_congr rfl fun j _ => ?_
  dsimp only
  rw [abs_mul, abs_of_pos (wt_pos R m j)]

theorem AInv.bound {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t ≤ 1024) {ve : ℕ → ℝ}
    (hve : ∀ j, |ve j| ≤ 2 ^ 15) {m a : F32} (hm : MInv R t m) (ha : AInv R ve t m a) :
    |a.toReal| ≤ 2 ^ 34 := by
  have hF := ffac_le_two hR (psi16r_le hR.E_nonneg hR.E_le) stepAr_one stepAr_le ht hm.bound
  have hF1 := ffac_one_le (t := t) (psi16r_nonneg hR.E_nonneg) stepAr_one (m_nonneg hm.bound)
  have htr : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have hMg := psum_wt_le hR hm (g := fun j => |ve j|) (fun j => abs_nonneg _) hve
  have hMg0 : 0 ≤ psum R.al t (fun j => wt R m.toReal j * |ve j|) :=
    psum_nonneg fun j => mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
  have hV := psum_le_card (al := R.al) (t := t) (f := fun j => |ve j|) (B := 2 ^ 15) (by norm_num)
    fun j _ _ => hve j
  have hV0 : 0 ≤ psum R.al t (fun j => |ve j|) := psum_nonneg fun j => abs_nonneg _
  have hS := psum_abs_wt R m.toReal t ve
  have hW := psum_wt1_le hR ht hm
  have hW0 : 0 ≤ psum R.al t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
  have hAb : om0 * psum R.al t (fun j => |ve j|) + kapAr * psum R.al t (wt R m.toReal) + t * absAr ≤
      2 ^ 8 := by
    unfold om0 kapAr absAr
    have : (1 / 2 ^ 25 + 1 / 2 ^ 125 + 66 / 2 ^ 36 : ℝ) * psum R.al t (fun j => |ve j|) ≤
        (1 / 2 ^ 24) * (64 * 1024 * 2 ^ 15) := by
      apply mul_le_mul (by norm_num) (by nlinarith) hV0 (by norm_num)
    nlinarith
  have hAb0 : 0 ≤ om0 * psum R.al t (fun j => |ve j|) + kapAr * psum R.al t (wt R m.toReal) + t * absAr := by
    unfold om0 kapAr absAr; positivity
  obtain ⟨-, he⟩ := ha
  have := abs_sub_abs_le_abs_sub a.toReal (psum R.al t (fun j => wt R m.toReal j * ve j))
  have k1 := mul_le_mul_of_nonneg_right (by linarith : ffac R (psi16r R.E) stepAr t m.toReal - 1 ≤ 1) hMg0
  have k2 := mul_le_mul_of_nonneg_right hF hAb0
  nlinarith

/-- `alpha` is finite and at most `2`. -/
theorem alpha_le_two {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} {m m' dm al : F32} (hM' : MInv R (t + 1) m')
    (hin : AlphaIn t m m' dm) (ha : P.math .exp2 .binary32 dm al) (hm : 0 < t → MInv R t m) :
    al.isFinite = true ∧ |al.toReal| ≤ 2 := by
  rcases Nat.eq_zero_or_pos t with h0 | ht0
  · subst h0; have := alpha_zero hni hin ha; exact ⟨this.1, by rw [this.2]; norm_num⟩
  · have hmi := hm ht0
    obtain ⟨haf, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    refine ⟨haf, ?_⟩
    have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
    have hl0 := two_rpow_pos (m.toReal - m'.toReal)
    have hd : dAl m.toReal m'.toReal ≤ 1 / 2 ^ 12 := by
      unfold dAl; have := hmi.bound; have := hM'.bound; have := hR.Y_le; have := hR.E_le
      have := neg_abs_le m.toReal; have := le_abs_self m'.toReal; linarith
    have hd0 : 0 ≤ dAl m.toReal m'.toReal := by
      unfold dAl; have : 0 ≤ m'.toReal - m.toReal := by linarith
      linarith [div_nonneg this (by norm_num : (0 : ℝ) ≤ 2 ^ 24), (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)]
    have := abs_sub_abs_le_abs_sub al.toReal ((2 : ℝ) ^ (m.toReal - m'.toReal))
    rw [abs_of_pos hl0] at this
    have k : ((1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 / 2 := by
      have : (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 ≤ 1 / 2 := by nlinarith
      have h0 : 0 ≤ (1 + dAl m.toReal m'.toReal) * (1 + 1 / 2 ^ 22) - 1 := by nlinarith
      nlinarith
    linarith

theorem relDot16_64 : ((Rules.Num.relDot16 64 : ℚ) : ℝ) = 66 / 2 ^ 22 := by
  unfold Rules.Num.relDot16; push_cast; norm_num

theorem absDot16_64 : ((Rules.Num.absDot16 64 : ℚ) : ℝ) = 66 / 2 ^ 126 := by
  unfold Rules.Num.absDot16; push_cast; norm_num

/-- **One block for an element of `acc`** (lines 82–84, 95, 101–103): `q` the block's row of
`p.to(float16)`, `vt` its column of `v` (the values `ve` of keys `64t + jj`), `a1` the element of
`acc * alpha[:, None]`, `a'` that of `tl.dot(p, v, acc)` (`hdot`: `Num.DotF16` at this element,
relative to the lifted magnitudes `|a1| + Σ liftMag p·liftMag v`: a subnormal `p` adds at most
`2^-14·|v|` per key, kept in `om0`, and a subnormal `v` at most `2^-14·|p|`, with
`|p| ≤ (1 + psi16r)·wt + omR`: the weight part, summed with the earlier blocks' rescaled through
`psum_step1`, is kept in `kapAr`, the rest (`2^-14·omR + 2^-28` per key) in `absAr`, all times
`66·2^-22`). Before the first block the element is `0.0`. -/
theorem astep {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {ve : ℕ → ℝ} (hve : ∀ j, |ve j| ≤ 2 ^ 15)
    {m m' dm al a a1 a' : F32} {x p : Fin 64 → F32} {q vt : Fin 64 → F16}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hq : ∀ jj, FloatV.NaNEq (FloatV.cvt (p jj)) (q jj))
    (hin : AlphaIn t m m' dm) (ha : P.math .exp2 .binary32 dm al)
    (ha1 : FloatV.NaNEq (FloatV.mul a al) a1)
    (hvt : ∀ jj, (vt jj).isFinite = true ∧ (vt jj).toReal = ve (64 * t + jj))
    (hdot : a1.isFinite = true → (∀ k, (q k).isFinite = true) →
      |a1.toRat| + ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat +
        (Rules.Num.relDot16 64 * (|a1.toRat| +
          ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat) +
          Rules.Num.absDot16 64) ≤ Format.binary32.maxVal →
      a'.isFinite = true ∧ |a'.toRat - (a1.toRat + ∑ k, (q k).toRat * (vt k).toRat)| ≤
        Rules.Num.relDot16 64 * (|a1.toRat| +
          ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat) +
          Rules.Num.absDot16 64)
    (hold : (t = 0 ∧ a.isFinite = true ∧ a.toReal = 0) ∨ (0 < t ∧ MInv R t m ∧ AInv R ve t m a)) :
    AInv R ve (t + 1) m' a' := by
  have hE0 := hR.E_nonneg
  have hE8 := hR.E_le
  have hps0 := psi16r_nonneg hE0
  have hps1 := psi16r_le hE0 hE8
  -- the lanes
  have hlane : ∀ jj, |(q jj).toReal * (vt jj).toReal - bwg R m'.toReal t ve jj| ≤
        psi16r R.E * bwg R m'.toReal t (fun j => |ve j|) jj + omR * bg R t (fun j => |ve j|) jj ∧
      |(q jj).toReal * (vt jj).toReal| ≤
        (1 + psi16r R.E) * bwg R m'.toReal t (fun j => |ve j|) jj + omR * bg R t (fun j => |ve j|) jj ∧
      |bwg R m'.toReal t ve jj| ≤ 0 * bwg R m'.toReal t ve jj + bwg R m'.toReal t (fun j => |ve j|) jj ∧
      bwg R m'.toReal t (fun j => |ve j|) jj ≤ 0 * bwg R m'.toReal t ve jj + 2 ^ 16 ∧
      bg R t (fun j => |ve j|) jj ≤ 0 * bwg R m'.toReal t ve jj + 2 ^ 15 ∧
      0 ≤ bwg R m'.toReal t (fun j => |ve j|) jj ∧ 0 ≤ bg R t (fun j => |ve j|) jj := by
    intro jj
    obtain ⟨hxf, hxa, hxm⟩ := hX jj
    obtain ⟨hpf, hpa, hpm⟩ := lane_p hexp hR hxf hxa hxm (fun h => hM'.above _ (by omega) h) (hp jj)
    obtain ⟨hqf, hqa, hqm⟩ := lane_p16 hR hpf hpa hpm (fun h => wt_le_two hR hM' (by omega) h) (hq jj)
    obtain ⟨-, hvv⟩ := hvt jj
    rw [hvv]
    have hv := hve (64 * t + jj)
    unfold bwg bg
    cases hal : R.al (64 * t + jj)
    · have h0 := hqm hal
      rw [toReal_def, h0]
      simp only [Bool.false_eq_true, ↓reduceIte, Rat.cast_zero, zero_mul, sub_zero, abs_zero, mul_zero,
        add_zero, zero_add, le_refl, true_and]
      norm_num
    · have h1 := hqa hal
      have hGp := wt_pos R m'.toReal (64 * t + jj)
      have hG2 := wt_le_two hR hM' (j := 64 * t + jj) (by omega) hal
      simp only [↓reduceIte, zero_mul, zero_add]
      have hva := abs_nonneg (ve (64 * t + jj))
      refine ⟨?_, ?_, ?_, ?_, hv, by positivity, hva⟩
      · rw [← sub_mul, abs_mul]
        have := mul_le_mul_of_nonneg_right h1 hva
        linarith
      · rw [abs_mul]
        have := abs_sub_abs_le_abs_sub (q jj).toReal (wt R m'.toReal (64 * t + jj))
        rw [abs_of_pos hGp] at this
        have := mul_le_mul_of_nonneg_right (show |(q jj).toReal| ≤ (1 + psi16r R.E) * wt R m'.toReal (64 * t + jj) + omR by linarith) hva
        linarith
      · rw [abs_mul, abs_of_pos hGp]
      · have := mul_le_mul hG2 hv hva (by norm_num); linarith
  have hqf : ∀ jj, (q jj).isFinite = true := fun jj => by
    obtain ⟨hxf, hxa, hxm⟩ := hX jj
    obtain ⟨hpf, hpa, hpm⟩ := lane_p hexp hR hxf hxa hxm (fun h => hM'.above _ (by omega) h) (hp jj)
    exact (lane_p16 hR hpf hpa hpm (fun h => wt_le_two hR hM' (by omega) h) (hq jj)).1
  have hcn := lane_sum_err2 (fun jj => (hlane jj).1)
  have hTn := lane_sum_le2 (fun jj => (hlane jj).2.1)
  have hMn' := lane_sum_le2 (fun jj => (hlane jj).2.2.1)
  have hMb := lane_sum_le2 (fun jj => (hlane jj).2.2.2.1)
  have hVb := lane_sum_le2 (fun jj => (hlane jj).2.2.2.2.1)
  have hMn0 : 0 ≤ ∑ jj, bwg R m'.toReal t (fun j => |ve j|) jj :=
    Finset.sum_nonneg fun jj _ => (hlane jj).2.2.2.2.2.1
  have hVn0 : 0 ≤ ∑ jj, bg R t (fun j => |ve j|) jj :=
    Finset.sum_nonneg fun jj _ => (hlane jj).2.2.2.2.2.2
  simp only [zero_mul, zero_add, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hMn' hMb hVb
  rw [← Finset.mul_sum] at hcn hTn
  have hMn : |∑ jj, bwg R m'.toReal t ve jj| ≤ ∑ jj, bwg R m'.toReal t (fun j => |ve j|) jj :=
    (Finset.abs_sum_le_sum_abs _ _).trans hMn'
  -- `acc` and `alpha` are finite and bounded
  have hab : a.isFinite = true ∧ |a.toReal| ≤ 2 ^ 34 := by
    rcases hold with ⟨-, hf, hz⟩ | ⟨ht0, hmi, hA⟩
    · exact ⟨hf, by rw [hz]; norm_num⟩
    · exact ⟨hA.1, hA.bound hR (by omega) hve hmi⟩
  have halb := alpha_le_two hexp hni hR hM' hin ha (fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi)
  -- line 95
  have hroom1 : |a.toReal * al.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]; have := mul_le_mul hab.2 halb.2 (abs_nonneg _) (by norm_num); linarith
  obtain ⟨ha1f, ha1e⟩ := f32_mul ha1 hab.1 halb.1 hroom1
  have ha1b : |a1.toReal| ≤ 2 ^ 36 := by
    have := abs_sub_abs_le_abs_sub a1.toReal (a.toReal * al.toReal)
    rw [abs_mul] at ha1e this
    have := mul_le_mul hab.2 halb.2 (abs_nonneg _) (by norm_num)
    norm_num at ha1e this ⊢; linarith
  -- line 103: the lifted magnitudes of the `tl.dot` (a subnormal `p` or `v` counts at `2^-14`)
  have hbw : ∀ jj, bw1 R m'.toReal t jj ≤ 2 := by
    intro jj
    unfold bw1
    split_ifs with hal
    · exact wt_le_two hR hM' (j := 64 * t + jj) (by omega) hal
    · norm_num
  have hW1 : ∑ jj, bw1 R m'.toReal t jj ≤ 128 := by
    have := Finset.sum_le_sum fun jj (_ : jj ∈ Finset.univ) => hbw jj
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at this
    norm_num at this; exact this
  have hW0 : 0 ≤ ∑ jj, bw1 R m'.toReal t jj := Finset.sum_nonneg fun jj _ => bw1_nonneg _ _ _ _
  have hlift1 : ∀ jj, (Rules.Num.liftMag .f16 (q jj).toRat : ℝ) * (Rules.Num.liftMag .f16 (vt jj).toRat : ℝ) ≤
      |(q jj).toReal * (vt jj).toReal| + 1 / 2 ^ 14 * bg R t (fun j => |ve j|) jj +
        (1 + psi16r R.E) / 2 ^ 14 * bw1 R m'.toReal t jj + (1 / 2 ^ 28 + 1 / 2 ^ 38) := by
    intro jj
    obtain ⟨hxf, hxa, hxm⟩ := hX jj
    obtain ⟨hpf, hpa, hpm⟩ := lane_p hexp hR hxf hxa hxm (fun h => hM'.above _ (by omega) h) (hp jj)
    obtain ⟨-, hqa, hqm⟩ := lane_p16 hR hpf hpa hpm (fun h => wt_le_two hR hM' (by omega) h) (hq jj)
    obtain ⟨-, hvv⟩ := hvt jj
    unfold bg bw1
    cases hal : R.al (64 * t + jj)
    · rw [hqm hal, Dot8Chunk.liftMag_zero]
      simp only [Rat.cast_zero, zero_mul, Bool.false_eq_true, ↓reduceIte, mul_zero, add_zero]
      positivity
    · simp only [↓reduceIte]
      have h1 := hqa hal
      have hGp := wt_pos R m'.toReal (64 * t + jj)
      have hqw : |(q jj).toReal| ≤ (1 + psi16r R.E) * wt R m'.toReal (64 * t + jj) + omR := by
        have := abs_sub_abs_le_abs_sub (q jj).toReal (wt R m'.toReal (64 * t + jj))
        rw [abs_of_pos hGp] at this
        linarith
      have hm := DotLift.liftMag_mul_le .f16 .f16 (q jj).toRat (vt jj).toRat
      rw [DotLift.minNormal8_f16] at hm
      have hm' := (Rat.cast_le (K := ℝ)).mpr hm
      push_cast at hm'
      simp only [← toReal_def] at hm'
      rw [hvv] at hm' ⊢
      have e14 : (2 : ℝ) ^ (-14 : ℤ) = 1 / 2 ^ 14 := by norm_num
      rw [e14] at hm'
      have := mul_le_mul_of_nonneg_left hqw (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 14)
      have e : 1 / 2 ^ 14 * ((1 + psi16r R.E) * wt R m'.toReal (64 * t + jj) + omR) =
          (1 + psi16r R.E) / 2 ^ 14 * wt R m'.toReal (64 * t + jj) + omR / 2 ^ 14 := by ring
      have hom : omR / 2 ^ 14 + 1 / 2 ^ 14 * (1 / 2 ^ 14) ≤ 1 / 2 ^ 28 + 1 / 2 ^ 38 := by
        unfold omR; norm_num
      linarith
  have hL : ∑ jj, (Rules.Num.liftMag .f16 (q jj).toRat : ℝ) * (Rules.Num.liftMag .f16 (vt jj).toRat : ℝ) ≤
      ∑ jj, |(q jj).toReal * (vt jj).toReal| + 1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj +
        (1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj + (1 / 2 ^ 22 + 1 / 2 ^ 32) := by
    calc ∑ jj, (Rules.Num.liftMag .f16 (q jj).toRat : ℝ) * (Rules.Num.liftMag .f16 (vt jj).toRat : ℝ)
        ≤ ∑ jj, (|(q jj).toReal * (vt jj).toReal| + 1 / 2 ^ 14 * bg R t (fun j => |ve j|) jj +
            (1 + psi16r R.E) / 2 ^ 14 * bw1 R m'.toReal t jj + (1 / 2 ^ 28 + 1 / 2 ^ 38)) :=
          Finset.sum_le_sum fun jj _ => hlift1 jj
      _ = _ := by
        simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
          Fintype.card_fin, nsmul_eq_mul]
        ring
  have hTb : ∑ jj, |(q jj).toReal * (vt jj).toReal| ≤ 2 ^ 24 := by
    have : (1 + psi16r R.E) * ∑ jj, bwg R m'.toReal t (fun j => |ve j|) jj ≤ 2 * (64 * 2 ^ 16) :=
      mul_le_mul (by linarith) hMb hMn0 (by norm_num)
    have : omR * ∑ jj, bg R t (fun j => |ve j|) jj ≤ 1 := by
      unfold omR; have := mul_le_mul_of_nonneg_left hVb (by positivity : (0 : ℝ) ≤ 1 / 2 ^ 25 + 1 / 2 ^ 125)
      norm_num at this ⊢; linarith
    linarith
  have hLb : ∑ jj, (Rules.Num.liftMag .f16 (q jj).toRat : ℝ) * (Rules.Num.liftMag .f16 (vt jj).toRat : ℝ) ≤
      2 ^ 25 := by
    have : 1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj ≤ 1 / 2 ^ 14 * (64 * 2 ^ 15) :=
      mul_le_mul_of_nonneg_left hVb (by norm_num)
    have : (1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj ≤ 2 / 2 ^ 14 * 128 :=
      mul_le_mul (by linarith) hW1 hW0 (by norm_num)
    norm_num at this ⊢; linarith
  have hdroom : |a1.toRat| + ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat +
      (Rules.Num.relDot16 64 * (|a1.toRat| +
        ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat) +
        Rules.Num.absDot16 64) ≤ Format.binary32.maxVal := by
    have : (((|a1.toRat| + ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat +
        (Rules.Num.relDot16 64 * (|a1.toRat| +
          ∑ k, Rules.Num.liftMag .f16 (q k).toRat * Rules.Num.liftMag .f16 (vt k).toRat) +
          Rules.Num.absDot16 64)) : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      push_cast
      rw [relDot16_64, absDot16_64, maxVal32]
      simp only [← toReal_def]
      have hL0 : 0 ≤ ∑ jj, (Rules.Num.liftMag .f16 (q jj).toRat : ℝ) * (Rules.Num.liftMag .f16 (vt jj).toRat : ℝ) :=
        Finset.sum_nonneg fun jj _ => mul_nonneg (by exact_mod_cast Dot8Chunk.liftMag_nonneg _ _)
          (by exact_mod_cast Dot8Chunk.liftMag_nonneg _ _)
      nlinarith
    exact_mod_cast this
  obtain ⟨ha'f, ha'e⟩ := hdot ha1f hqf hdroom
  have hcomb : |a'.toReal - (a1.toReal + ∑ jj, (q jj).toReal * (vt jj).toReal)| ≤
      66 / 2 ^ 22 * (|a1.toReal| + ∑ jj, |(q jj).toReal * (vt jj).toReal|) +
        (66 / 2 ^ 22 * (1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj +
          (1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj + (1 / 2 ^ 22 + 1 / 2 ^ 32)) + 66 / 2 ^ 126) := by
    have := (Rat.cast_le (K := ℝ)).mpr ha'e
    push_cast at this
    rw [relDot16_64, absDot16_64] at this
    simp only [← toReal_def] at this
    have := mul_le_mul_of_nonneg_left hL (by norm_num : (0 : ℝ) ≤ 66 / 2 ^ 22)
    linarith
  have htau2 : 0 ≤ 66 / 2 ^ 22 * (1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj +
      (1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj + (1 / 2 ^ 22 + 1 / 2 ^ 32)) + 66 / 2 ^ 126 := by
    have : 0 ≤ 1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj := mul_nonneg (by norm_num) hVn0
    have : 0 ≤ (1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj :=
      mul_nonneg (div_nonneg (by linarith) (by norm_num)) hW0
    positivity
  have hF2 : (1 + 66 / 2 ^ 22) * (1 + psi16r R.E) ≤ ffac R (psi16r R.E) stepAr (t + 1) m'.toReal :=
    ffac_ge hps0 (by positivity) stepAr_ge (m_nonneg hM'.bound)
  have hF'1 : 1 ≤ ffac R (psi16r R.E) stepAr (t + 1) m'.toReal :=
    ffac_one_le hps0 stepAr_one (m_nonneg hM'.bound)
  have htau : |a.toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150 + 66 / 2 ^ 22 * (1 / 2 ^ 22 + 1 / 2 ^ 32) +
      66 / 2 ^ 126 ≤ (absAr : ℝ) := by
    have := hab.2; unfold absAr; norm_num at this ⊢; linarith
  -- what the step adds, per key, per unit of weight and per block
  have hadd : ∀ tau1 : ℝ, tau1 ≤ |a.toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150 →
      omR * ∑ jj, bg R t (fun j => |ve j|) jj + tau1 +
        (66 / 2 ^ 22 * (1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj +
          (1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj + (1 / 2 ^ 22 + 1 / 2 ^ 32)) + 66 / 2 ^ 126) ≤
      om0 * ∑ jj, bg R t (fun j => |ve j|) jj + kapAr * ∑ jj, bw1 R m'.toReal t jj + absAr := by
    intro tau1 h
    have e : om0 * ∑ jj, bg R t (fun j => |ve j|) jj =
        omR * ∑ jj, bg R t (fun j => |ve j|) jj + 66 / 2 ^ 22 * (1 / 2 ^ 14 * ∑ jj, bg R t (fun j => |ve j|) jj) := by
      unfold om0 omR; ring
    have hk : 66 / 2 ^ 22 * ((1 + psi16r R.E) / 2 ^ 14) ≤ kapAr := by
      unfold kapAr; linarith
    have hk' := mul_le_mul_of_nonneg_right hk hW0
    have e2 : 66 / 2 ^ 22 * ((1 + psi16r R.E) / 2 ^ 14) * ∑ jj, bw1 R m'.toReal t jj =
        66 / 2 ^ 22 * ((1 + psi16r R.E) / 2 ^ 14 * ∑ jj, bw1 R m'.toReal t jj) := by ring
    rw [e]; linarith
  refine ⟨ha'f, ?_⟩
  push_cast
  rw [psum_stepg R m.toReal m'.toReal t ve, psum_stepg R m.toReal m'.toReal t (fun j => |ve j|),
    psum_stepv R t (fun j => |ve j|), psum_step1 R m.toReal m'.toReal t]
  rcases hold with ⟨ht0, -, ha0⟩ | ⟨ht0, hmi, hA⟩
  · -- the first block: `acc = 0`
    subst ht0
    have ha1' : |a1.toReal - 0 * a.toReal| ≤ (1 - 1) * (0 * |a.toReal|) +
        (|a.toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150) := by
      rw [ha0] at ha1e ⊢; simp only [zero_mul, abs_zero, sub_zero, mul_zero] at ha1e ⊢; linarith
    have key := acc_step (a := a.toReal) (S := 0) (Mg := 0) (Ab := 0) (F := 1) (lam := 0) (Phi := 1)
      (rho := 66 / 2 ^ 22) (psi := psi16r R.E)
      (F' := ffac R (psi16r R.E) stepAr (0 + 1) m'.toReal)
      le_rfl (by simp) le_rfl (by rw [ha0]; simp) le_rfl zero_le_one le_rfl ha1'
      hMn hps0 (by unfold omR; positivity) hcn hTn (by positivity) (by positivity) htau2 hcomb
      (by nlinarith) hF2
    simp only [psum_zero, mul_zero, zero_add, Nat.cast_zero] at key ⊢
    have h2 := mul_le_mul_of_nonneg_left (hadd _ le_rfl)
      (by linarith : (0 : ℝ) ≤ ffac R (psi16r R.E) stepAr (0 + 1) m'.toReal)
    nlinarith
  · obtain ⟨-, hle, hae⟩ := alpha_bound hexp hR hin ht0 hmi.bound hM'.bound ha
    have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
    have hl0 := two_rpow_pos (m.toReal - m'.toReal)
    have hd0 : 0 ≤ dAl m.toReal m'.toReal := by
      unfold dAl; have : 0 ≤ m'.toReal - m.toReal := by linarith
      linarith [div_nonneg this (by norm_num : (0 : ℝ) ≤ 2 ^ 24), (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)]
    have hMg0 : 0 ≤ psum R.al t (fun j => wt R m.toReal j * |ve j|) :=
      psum_nonneg fun j => mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
    have hV0 : 0 ≤ om0 * psum R.al t (fun j => |ve j|) := by
      have : 0 ≤ psum R.al t (fun j => |ve j|) := psum_nonneg fun j => abs_nonneg _
      unfold om0; positivity
    have hWt0 : 0 ≤ kapAr * psum R.al t (wt R m.toReal) := by
      have : 0 ≤ psum R.al t (wt R m.toReal) := psum_nonneg fun j => (wt_pos _ _ _).le
      unfold kapAr; positivity
    have htA0 : 0 ≤ (t : ℝ) * absAr := by unfold absAr; positivity
    have hAb0 : 0 ≤ om0 * psum R.al t (fun j => |ve j|) + kapAr * psum R.al t (wt R m.toReal) + t * absAr := by
      linarith
    have hF1 := ffac_one_le (t := t) hps0 stepAr_one (m_nonneg hmi.bound)
    have hsc := scale_err hl0.le hae ha1e
    have hFs := ffac_step (R := R) (psi := psi16r R.E) (rho := 66 / 2 ^ 22)
      (B0 := (1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) (t := t) hps0 (by positivity) (by positivity) hle
    have key := acc_step' (S := psum R.al t (fun j => wt R m.toReal j * ve j))
      (Mg := psum R.al t (fun j => wt R m.toReal j * |ve j|))
      (Ab := om0 * psum R.al t (fun j => |ve j|) + kapAr * psum R.al t (wt R m.toReal) + t * absAr)
      (F := ffac R (psi16r R.E) stepAr t m.toReal)
      (lam := (2 : ℝ) ^ (m.toReal - m'.toReal))
      (Phi := (1 + dAl m.toReal m'.toReal) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)))
      (rho := 66 / 2 ^ 22) (psi := psi16r R.E) (F' := ffac R (psi16r R.E) stepAr (t + 1) m'.toReal)
      hF1 (psum_abs_wt R m.toReal t ve) hAb0 hA.2 hl0.le hl1
      (one_le_mul_of_one_le_of_one_le (by linarith) (by norm_num)) hsc
      hMn hps0 (by unfold omR; positivity) hcn hTn (by positivity) (by positivity) htau2 hcomb
      hFs hF2
    refine key.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ (le_trans zero_le_one hF'1)))
    have h2 := hadd _ le_rfl
    have k1 := mul_le_mul_of_nonneg_right hl1 hV0
    have k2 := mul_le_mul_of_nonneg_right hl1 htA0
    have e : ((t : ℝ) + 1) * absAr = t * absAr + absAr := by ring
    rw [e, mul_add om0, mul_add kapAr]
    linarith

end Kernels.AttnFwd
