import Claim
import TVA.Proof
import Proof.Setup
import Proof.Cast
import Proof.ExpNeg
import Proof.WeightDen

/-!
# The second routing weight, against the reference softmax

The block computes, with `t1 = top1v`, `t2 = top2v` and `inv = qinv.item`,

  `w = tl.sub (scalar t2) (scalar t1)`, `e = exp w`, `w1 = e * inv`

and the claim is `|w1 - softRef (wC capped) i0 i1| <= wRel1 * softRef ... + wFloor`.

## Numbers

`c0 = wC capped i0`, `c1 = wC capped i1`, `d1 = c1 - c0 <= 0` (`hle1`), `|c0|, |c1| <= 30.00003`
(`hsmall`), `D = wD capped i0` with `1 <= D <= 8`, and `softRef (wC capped) i0 i1` is
definitionally `exp d1 / D`.

1. `|t2 - c1| <= 2^-15` (`htop2val`), `t1 = c0` (`htopval`), so `|t2 - t1 - d1| <= 2^-15` and
   `|t2| + |t1| <= 2 * 30.00003 + 2^-15 = 60.0000916`.
2. `w` (`FloatV.sub_err`): `|w' - d1| <= 2^-15 + u * 60.0000916 + eta <= 3.41e-5 =: δw`.
3. `e` (`ExpF32` with `X = |w'| <= 61`, `rho = 2^-22 + 2^-23 |w'| <= 7.4e-6`):
   `|e' - exp d1| <= Ew * exp d1 + 2^-140` with
   `Ew = rho * exp δw + (exp δw - 1) <= 7.4e-6 * 1.0000341 + 3.41012e-5 <= 4.151e-5`
   (using `exp δw - 1 <= δw / (1 - δw)`).
4. `w1` (`FloatV.mul_err`): `|w1 - e' * inv| <= u |e' * inv| + eta`.
5. `|inv * D - 1| <= 1.5e-5` (`den_qinv_bound`), so `|inv - 1/D| <= 1.5e-5 / D`.
   Together: with `S = exp d1 / D <= 1`,
   `|w1 - S| <= (Ew + 1.5e-5 + u) * S + Ew * 1.5e-5 * S + (absolute terms)`
   and `Ew + 1.5e-5 + u <= 4.151e-5 + 1.5e-5 + 5.96e-8 <= 5.66e-5 < wRel1 = 2^-14 =
   6.1035156e-5` (7 per cent margin); the absolute terms (`2^-140`, `eta`, their products)
   are below `wFloor = 2^-130`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

private theorem abs_exp_sub_one_le {t δ : ℝ} (h : |t| ≤ δ) :
    |Real.exp t - 1| ≤ Real.exp δ - 1 := by
  have ht := abs_le.mp h
  have hupper : Real.exp t - 1 ≤ Real.exp δ - 1 := by linarith [Real.exp_le_exp.mpr ht.2]
  have hlower : -(Real.exp δ - 1) ≤ Real.exp t - 1 := by
    have h1 : Real.exp (-δ) ≤ Real.exp t := Real.exp_le_exp.mpr (by linarith [ht.1])
    have h2 : 1 - δ ≤ Real.exp (-δ) := Real.one_sub_le_exp_neg δ
    have h3 : δ ≤ Real.exp δ - 1 := by linarith [Real.add_one_le_exp δ]
    linarith
  exact abs_le.mpr ⟨hlower, hupper⟩

private theorem abs_exp_sub_exp_le {a b δ : ℝ} (h : |a - b| ≤ δ) :
    |Real.exp a - Real.exp b| ≤ Real.exp b * (Real.exp δ - 1) := by
  have hb : 0 < Real.exp b := Real.exp_pos b
  have hinner : |Real.exp (a - b) - 1| ≤ Real.exp δ - 1 := abs_exp_sub_one_le h
  have heq : Real.exp a - Real.exp b = Real.exp b * (Real.exp (a - b) - 1) := by
    rw [Real.exp_sub]; field_simp
  rw [heq, abs_mul, abs_of_pos hb]
  exact mul_le_mul_of_nonneg_left hinner hb.le

set_option maxHeartbeats 4000000 in
/-- **The second weight.** `topk_weights[t, 1]` is finite and within `wRel1` of the softmax weight of the second index. -/
theorem weight_second {d : Inst} {P : Platform} {t : Nat} (ht : t < d.bs)
    {capped : Tensor [8 - 0] F32}
    (hfin : ∀ e : Fin (8 - 0), (capped.get1 e).isFinite = true)
    (hsmall : ∀ e : Fin (8 - 0), |(capped.get1 e).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)))
    (hsum : Device.SumDepth P)
    (hexpacc : Device.ExpF32 P)
    (hdivfull : Device.DivFull P)
    {i0 : Nat} (hi0 : i0 < 8 - 0)
    (hle0 : ∀ e : Fin (8 - 0), (capped.get1 e).toReal ≤ (capped.get1 ⟨i0, hi0⟩).toReal)
    {top1v : F32} (htopfin : top1v.isFinite = true)
    (htopval : top1v.toReal = (capped.get1 ⟨i0, hi0⟩).toReal)
    {diffs exps : Tensor [8 - 0] F32}
    (hdiffs : ∀ j : Idx [8 - 0],
      AddRel ((launch d).env P).fusion (capped.get j) ((bc top1v).get j).neg (diffs.get j))
    (hexps : ∀ j : Idx [8 - 0],
      P.math MathFn.exp Format.binary32 (diffs.get j) (exps.get j))
    {den : F32}
    (hden : ∃ tr, Tree.Covers (8 - 0) tr ∧ tr.depth ≤ tl.sumBound P (8 - 0) ∧
      Tree.Eval (AddRel false)
        (fun i => if h : i < 8 - 0 then exps.get1 ⟨i, h⟩ else default) tr den)
    {qinv : Tensor [] F32}
    (hinv : ∀ i : Idx [], P.div Format.binary32
      ((Tensor.scalar (c32 0x3FF0000000000000)).get i) ((Tensor.scalar den).get i)
      (qinv.get i))
    {i1 : Nat} (hi1 : i1 < 8 - 0) (_hne1 : i1 ≠ i0)
    (_hle1 : ∀ e : Fin (8 - 0), e.val ≠ i0 →
      (capped.get1 e).toReal ≤ (capped.get1 ⟨i1, hi1⟩).toReal)
    {top2v : F32} (htop2fin : top2v.isFinite = true)
    (htop2val : |top2v.toReal - (capped.get1 ⟨i1, hi1⟩).toReal| ≤ (2 : ℝ) ^ (-15 : ℤ))
    {wdiff w2e : Tensor [] F32} {w1v : Tensor [] F32}
    (hwdiff : ∀ i : Idx [], FloatV.NaNEq
      (FloatV.sub ((Tensor.scalar top2v).get i) ((Tensor.scalar top1v).get i)) (wdiff.get i))
    (hw2e : ∀ i : Idx [],
      P.math MathFn.exp Format.binary32 (wdiff.get i) (w2e.get i))
    (hw1v : ∀ i : Idx [], FloatV.NaNEq
      (FloatV.mul (w2e.get i) ((Tensor.scalar qinv.item).get i)) (w1v.get i)) :
    w1v.item.isFinite = true ∧
      |w1v.item.toReal - softRef (wC capped) i0 i1| ≤
        wRel1 * softRef (wC capped) i0 i1 + wFloor := by
  -- ## 0. the denominator / inverse pipeline
  obtain ⟨hdenfin, hdenne, hqinvfin, hqinvD⟩ :=
    den_qinv_bound ht hfin hsmall hsum hexpacc hdivfull hi0 hle0 htopfin htopval
      hdiffs hexps hden hinv
  -- ## 1. the two capped logits
  have hwC : ∀ (k : Nat) (hk : k < 8 - 0), wC capped k = (capped.get1 ⟨k, hk⟩).toReal := by
    intro k hk
    unfold wC
    rw [dite_eq_left hk]
  set c0 : ℝ := (capped.get1 ⟨i0, hi0⟩).toReal with hc0
  set c1 : ℝ := (capped.get1 ⟨i1, hi1⟩).toReal with hc1
  have hwC0 : wC capped i0 = c0 := by rw [hwC i0 hi0, ← hc0]
  have hwC1 : wC capped i1 = c1 := by rw [hwC i1 hi1, ← hc1]
  have hc0le : |c0| ≤ 30 * (1 + 2 ^ (-20 : ℤ)) := hsmall ⟨i0, hi0⟩
  have hc1le : |c1| ≤ 30 * (1 + 2 ^ (-20 : ℤ)) := hsmall ⟨i1, hi1⟩
  have hd1 : c1 - c0 ≤ 0 := by
    have h := hle0 ⟨i1, hi1⟩
    linarith
  have htopval' : top1v.toReal = c0 := htopval
  have htop2val' : |top2v.toReal - c1| ≤ (2 : ℝ) ^ (-15 : ℤ) := htop2val
  -- ## 2. `|t2 - t1|` is small
  have hdiff : |top2v.toReal - top1v.toReal| ≤ 600001 / 10000 := by
    have ht2 : |top2v.toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)) + (2 : ℝ) ^ (-15 : ℤ) := by
      have h1 := abs_add_le c1 (top2v.toReal - c1)
      have h2 : c1 + (top2v.toReal - c1) = top2v.toReal := by ring
      rw [h2] at h1
      linarith [htop2val', hc1le, h1]
    have h3 : |top2v.toReal - top1v.toReal| ≤ |top2v.toReal| + |top1v.toReal| := by
      have h := abs_add_le top2v.toReal (-top1v.toReal)
      have he : top2v.toReal + -top1v.toReal = top2v.toReal - top1v.toReal := by ring
      rw [he, abs_neg] at h
      exact h
    have h4 : |top1v.toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)) := by rw [htopval']; exact hc0le
    have h5 : (30 * (1 + 2 ^ (-20 : ℤ)) + (2 : ℝ) ^ (-15 : ℤ)) + 30 * (1 + 2 ^ (-20 : ℤ))
        ≤ 600001 / 10000 := by norm_num
    exact le_trans h3 (le_trans (add_le_add ht2 h4) h5)
  -- ## 3. the fused difference `w'`
  have hsubmax : |top2v.toRat - top1v.toRat| ≤ Format.binary32.maxVal := by
    refine Rat.cast_le (K := ℝ).mp ?_
    rw [Rat.cast_abs, Rat.cast_sub]
    calc |(top2v.toRat : ℝ) - (top1v.toRat : ℝ)| = |top2v.toReal - top1v.toReal| := rfl
      _ ≤ 600001 / 10000 := hdiff
      _ ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by rw [maxVal32_eq]; norm_num
  obtain ⟨hsubfin, hsuberr⟩ :=
    FloatV.sub_err (f := Format.binary32) (x := top2v) (y := top1v)
      bin32_regular htop2fin htopfin hsubmax
  have hwdiff' : FloatV.NaNEq (FloatV.sub top2v top1v) wdiff.item := hwdiff ()
  have hwdiff_eq : wdiff.item = FloatV.sub top2v top1v :=
    hwdiff'.eq (Format.not_isNaN_of_isFinite hsubfin)
  have hsubR : |wdiff.item.toReal - (top2v.toReal - top1v.toReal)| ≤
      |top2v.toReal - top1v.toReal| * u32 + eta32 := by
    have hcast : (((|(FloatV.sub top2v top1v).toRat - (top2v.toRat - top1v.toRat)| : ℚ) : ℝ))
        ≤ (((|top2v.toRat - top1v.toRat| : ℚ) *
            (Format.binary32.u : ℚ) + (Format.binary32.eta : ℚ) : ℚ) : ℝ) :=
      Rat.cast_le (K := ℝ).mpr hsuberr
    rw [hwdiff_eq]
    push_cast at hcast
    rw [u32_eq, eta32_eq] at hcast
    exact hcast
  set dw : ℝ := 600001 / 10000 * u32 + eta32 + (2 : ℝ) ^ (-15 : ℤ) with hdwdef
  have hdw : |wdiff.item.toReal - (c1 - c0)| ≤ dw := by
    rw [hdwdef]
    have h1 : |wdiff.item.toReal - (c1 - c0)| ≤
        |wdiff.item.toReal - (top2v.toReal - top1v.toReal)|
          + |(top2v.toReal - top1v.toReal) - (c1 - c0)| := abs_sub_le _ _ _
    have h2 : |(top2v.toReal - top1v.toReal) - (c1 - c0)| ≤ (2 : ℝ) ^ (-15 : ℤ) := by
      have he : (top2v.toReal - top1v.toReal) - (c1 - c0) = top2v.toReal - c1 := by
        rw [htopval']; ring
      rw [he]; exact htop2val'
    have h3 : |top2v.toReal - top1v.toReal| * u32 ≤ 600001 / 10000 * u32 :=
      mul_le_mul_of_nonneg_right hdiff u32_pos.le
    have h4 : |wdiff.item.toReal - (top2v.toReal - top1v.toReal)|
        + |(top2v.toReal - top1v.toReal) - (c1 - c0)| ≤
        (|top2v.toReal - top1v.toReal| * u32 + eta32) + (2 : ℝ) ^ (-15 : ℤ) :=
      add_le_add hsubR h2
    have h5 : |top2v.toReal - top1v.toReal| * u32 + eta32
        ≤ 600001 / 10000 * u32 + eta32 := add_le_add h3 (le_refl _)
    have h6 : (|top2v.toReal - top1v.toReal| * u32 + eta32) + (2 : ℝ) ^ (-15 : ℤ)
        ≤ (600001 / 10000 * u32 + eta32) + (2 : ℝ) ^ (-15 : ℤ) :=
      add_le_add h5 (le_refl _)
    exact le_trans h1 (le_trans h4 h6)
  have hdwpos : 0 < dw := by rw [hdwdef]; unfold u32 eta32; norm_num
  have hdwle : dw ≤ 341 / 10000000 := by rw [hdwdef]; unfold u32 eta32; norm_num
  have hdwlt : dw < 1 := by rw [hdwdef]; unfold u32 eta32; norm_num
  -- ## 4. `|w'| <= 61`
  have hwabs : |wdiff.item.toReal| ≤ 61 := by
    have h1 : |wdiff.item.toReal| ≤ |wdiff.item.toReal - (c1 - c0)| + |c1 - c0| := by
      have h := abs_sub_le wdiff.item.toReal (c1 - c0) 0
      simpa using h
    have h2 : |c1 - c0| ≤ 2 * (30 * (1 + (2 : ℝ) ^ (-20 : ℤ))) := by
      have h3 : |c1 - c0| ≤ |c1| + |c0| := by
        have h := abs_add_le c1 (-c0)
        have he : c1 + -c0 = c1 - c0 := by ring
        rw [he, abs_neg] at h
        exact h
      linarith [h3, hc0le, hc1le]
    have hnum : 2 * (30 * (1 + (2 : ℝ) ^ (-20 : ℤ))) + 1 / 1000 ≤ 61 := by norm_num
    have hsmall' : dw ≤ 1 / 1000 := by linarith [hdwle]
    have h4 : |wdiff.item.toReal - (c1 - c0)| + |c1 - c0|
        ≤ dw + 2 * (30 * (1 + (2 : ℝ) ^ (-20 : ℤ))) := add_le_add hdw h2
    have h5 : dw + 2 * (30 * (1 + (2 : ℝ) ^ (-20 : ℤ)))
        ≤ 1 / 1000 + 2 * (30 * (1 + (2 : ℝ) ^ (-20 : ℤ))) := by linarith [hsmall']
    have h6 : 1 / 1000 + 2 * (30 * (1 + (2 : ℝ) ^ (-20 : ℤ))) ≤ 61 := by
      rw [add_comm]; exact hnum
    exact le_trans h1 (le_trans h4 (le_trans h5 h6))
  -- ## 5. `exp w'`
  obtain ⟨ref, href, hacc⟩ := hexpacc 61 (by norm_num)
  have href' : ref = Real.exp := by
    have h : (some Real.exp : Option (ℝ → ℝ)) = some ref := href
    exact (Option.some.inj h).symm
  have hw2e' : P.math MathFn.exp Format.binary32 wdiff.item w2e.item := hw2e ()
  have hwdifffin : wdiff.item.isFinite = true := by rw [hwdiff_eq]; exact hsubfin
  have hmem : wdiff.item.toReal ∈ Set.Icc (-61) 61 :=
    ⟨by linarith [neg_abs_le wdiff.item.toReal],
     by linarith [le_abs_self wdiff.item.toReal]⟩
  have hnat : wdiff.item.toReal ∈ MathFn.exp.natDom := Set.mem_univ _
  have hexp61 : Real.exp 61 ≤ (3 : ℝ) ^ 61 := by
    have h : Real.exp 61 = Real.exp 1 ^ 61 := by
      rw [← Real.exp_nat_mul]; norm_num
    calc Real.exp 61 = Real.exp 1 ^ 61 := h
      _ ≤ 3 ^ 61 := pow_le_pow_left₀ (le_of_lt (Real.exp_pos 1)) (le_of_lt Real.exp_one_lt_three) 61
  have hexpw_le : Real.exp wdiff.item.toReal ≤ (3 : ℝ) ^ 61 :=
    le_trans (Real.exp_le_exp.mpr (by linarith [le_abs_self wdiff.item.toReal])) hexp61
  have hrel_nn : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61 := by norm_num
  have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61)) * |ref wdiff.item.toReal| +
      2 ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [href', Real.abs_exp]
    have h1 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61)) * Real.exp wdiff.item.toReal ≤
        (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61)) * (3 : ℝ) ^ 61 :=
      mul_le_mul_of_nonneg_left hexpw_le (by norm_num)
    calc (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61)) * Real.exp wdiff.item.toReal
          + 2 ^ (-140 : ℤ)
        ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61)) * (3 : ℝ) ^ 61 + 2 ^ (-140 : ℤ) := by
          linarith
      _ ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by rw [maxVal32_eq]; norm_num
  obtain ⟨hw2efin, hw2eerr⟩ := hacc wdiff.item w2e.item hw2e' hwdifffin hmem hnat hroom
  have hw2eerr' : |w2e.item.toReal - Real.exp wdiff.item.toReal| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61) * Real.exp wdiff.item.toReal +
        2 ^ (-140 : ℤ) := by
    rw [href', Real.abs_exp] at hw2eerr
    exact hw2eerr
  -- ## 6. from `exp w'` to `exp d1`
  have hexpdiff : |Real.exp wdiff.item.toReal - Real.exp (c1 - c0)| ≤
      Real.exp (c1 - c0) * (Real.exp dw - 1) := abs_exp_sub_exp_le hdw
  set Ew : ℝ := (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61) * Real.exp dw + (Real.exp dw - 1)
    with hEwdef
  have hexpw_le' : Real.exp wdiff.item.toReal ≤ Real.exp dw * Real.exp (c1 - c0) := by
    have h := abs_le.mp hdw
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [h.2])
  have hEwbound : |w2e.item.toReal - Real.exp (c1 - c0)| ≤
      Ew * Real.exp (c1 - c0) + 2 ^ (-140 : ℤ) := by
    have h1 : |w2e.item.toReal - Real.exp (c1 - c0)| ≤
        |w2e.item.toReal - Real.exp wdiff.item.toReal|
          + |Real.exp wdiff.item.toReal - Real.exp (c1 - c0)| := abs_sub_le _ _ _
    have h2 : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61) * Real.exp wdiff.item.toReal ≤
        (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61) * (Real.exp dw * Real.exp (c1 - c0)) :=
      mul_le_mul_of_nonneg_left hexpw_le' hrel_nn
    have he : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61) * Real.exp dw * Real.exp (c1 - c0)
          + (Real.exp dw - 1) * Real.exp (c1 - c0) + 2 ^ (-140 : ℤ)
        = Ew * Real.exp (c1 - c0) + 2 ^ (-140 : ℤ) := by
      rw [hEwdef]; ring
    have h3 := abs_le.mp hexpdiff
    have h4 := abs_le.mp hw2eerr'
    nlinarith [h1, h2, h3.1, h3.2, h4.1, h4.2, he]
  have hexp_dw_le : Real.exp dw ≤ 1 / (1 - dw) :=
    Real.exp_bound_div_one_sub_of_interval hdwpos.le hdwlt
  have h1mdw : 0 < 1 - dw := by linarith
  have hexp_dw_le' : Real.exp dw ≤ 100004 / 100000 := by
    have h2 : (1 : ℝ) / (1 - dw) ≤ 100004 / 100000 := by
      rw [div_le_iff₀ h1mdw]
      nlinarith [hdwle]
    linarith
  have hexp_dw_sub : Real.exp dw - 1 ≤ 342 / 10000000 := by
    have h2 : dw / (1 - dw) ≤ 342 / 10000000 := by
      rw [div_le_iff₀ h1mdw]
      nlinarith [hdwle, hdwpos]
    have h3 : Real.exp dw - 1 ≤ 1 / (1 - dw) - 1 := by linarith
    have h4 : (1 : ℝ) / (1 - dw) - 1 = dw / (1 - dw) := by field_simp; try ring
    linarith
  have hrel_le : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61 ≤ 76 / 10000000 := by
    norm_num
  have hEwnn : 0 ≤ Ew := by
    rw [hEwdef]
    have h := Real.one_le_exp hdwpos.le
    have hpos := (Real.exp_pos dw).le
    nlinarith [hrel_nn, h, hpos]
  have hEwle : Ew ≤ 42 / 1000000 := by
    rw [hEwdef]
    have h := mul_le_mul hrel_le hexp_dw_le' (Real.exp_pos dw).le
      (by norm_num : (0 : ℝ) ≤ 76 / 10000000)
    have hnum : 76 / 10000000 * (100004 / 100000) + 342 / 10000000 ≤ 42 / 1000000 := by
      norm_num
    linarith [h, hexp_dw_sub, hnum]
  -- ## 7. the denominator `D` and the reference weight `S`
  set D : ℝ := wD capped i0 with hDdef
  have hD1 : 1 ≤ D := by
    rw [hDdef]
    have hterm : ∀ k ∈ Finset.range 8, (0 : ℝ) ≤ Real.exp (wC capped k - wC capped i0) :=
      fun k _ => (Real.exp_pos _).le
    have hmem : i0 ∈ Finset.range 8 := Finset.mem_range.mpr hi0
    have h := Finset.single_le_sum hterm hmem
    simpa [wD] using h
  have hD8 : D ≤ 8 := by
    rw [hDdef]
    have hle : ∀ k, k < 8 → wC capped k ≤ wC capped i0 := by
      intro k hk
      have h := hle0 ⟨k, hk⟩
      rw [hwC k hk, hwC0]
      exact h
    have h : ∑ k ∈ Finset.range 8, Real.exp (wC capped k - wC capped i0)
        ≤ ∑ k ∈ Finset.range 8, (1 : ℝ) := by
      refine Finset.sum_le_sum fun k hk => ?_
      rw [Real.exp_le_one_iff]
      linarith [hle k (Finset.mem_range.mp hk)]
    simpa [wD] using h
  have hDpos : 0 < D := by linarith
  set S : ℝ := Real.exp (c1 - c0) / D with hSdef
  have hSnn : 0 ≤ S := by
    rw [hSdef]; exact div_nonneg (Real.exp_pos _).le hDpos.le
  have hSle1 : S ≤ 1 := by
    rw [hSdef, div_le_one hDpos]
    have h := Real.exp_le_one_iff.mpr hd1
    linarith
  have hDc : D = ∑ k ∈ Finset.range 8, Real.exp (wC capped k - c0) := by
    rw [hDdef]
    unfold wD
    rw [hwC i0 hi0, ← hc0]
  have hsoft : softRef (wC capped) i0 i1 = S := by
    rw [hSdef, hDc]
    unfold softRef
    rw [hwC i1 hi1, hwC i0 hi0, ← hc1, ← hc0]
  have hqinvD' : |qinv.item.toReal * D - 1| ≤ (3 : ℝ) / 200000 := hqinvD
  have hZ : |qinv.item.toReal - 1 / D| ≤ ((3 : ℝ) / 200000) / D := by
    have h2 : qinv.item.toReal * D - 1 = (qinv.item.toReal - 1 / D) * D := by
      field_simp
    rw [h2, abs_mul, abs_of_pos hDpos] at hqinvD'
    rwa [le_div_iff₀ hDpos]
  have hqinv2 : |qinv.item.toReal| ≤ 2 := by
    have h5 : |1 / D| ≤ 1 := by
      rw [abs_of_pos (by positivity), div_le_one hDpos]
      linarith
    have h4 : qinv.item.toReal = (qinv.item.toReal - 1 / D) + 1 / D := by ring
    have h3 : ((3 : ℝ) / 200000) / D ≤ 1 := by
      rw [div_le_one hDpos]; linarith
    calc |qinv.item.toReal| = |(qinv.item.toReal - 1 / D) + 1 / D| := congrArg abs h4
      _ ≤ |qinv.item.toReal - 1 / D| + |1 / D| := abs_add_le _ _
      _ ≤ 1 + 1 := by linarith
      _ = 2 := by norm_num
  -- ## 8. the product `e' * inv`
  have hpow : (3 : ℝ) ^ 61 ≤ 2 ^ (97 : ℤ) := by norm_num
  have hpow2 : (2 : ℝ) ^ (97 : ℤ) + 2 ^ (97 : ℤ) + 2 ^ (-140 : ℤ) ≤ 2 ^ (100 : ℤ) := by
    norm_num
  have hw2eabs : |w2e.item.toReal| ≤ 2 ^ (100 : ℤ) := by
    have h1 : |w2e.item.toReal| ≤ Real.exp wdiff.item.toReal
        + |w2e.item.toReal - Real.exp wdiff.item.toReal| := by
      have h := abs_add_le (Real.exp wdiff.item.toReal)
        (w2e.item.toReal - Real.exp wdiff.item.toReal)
      have he : Real.exp wdiff.item.toReal
          + (w2e.item.toReal - Real.exp wdiff.item.toReal) = w2e.item.toReal := by ring
      rw [he, Real.abs_exp] at h
      exact h
    have h2 : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 61) * Real.exp wdiff.item.toReal ≤
        (3 : ℝ) ^ 61 := by
      have hle1 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61) ≤ 1 := by norm_num
      have hm := mul_le_mul hle1 hexpw_le (Real.exp_pos _).le (by norm_num : (0 : ℝ) ≤ 1)
      simpa using hm
    have hpos := (Real.exp_pos wdiff.item.toReal).le
    have h140 : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) := by norm_num
    have h4 := abs_le.mp hw2eerr'
    linarith [h1, h2, hexpw_le, hpow, hpow2, h4.1, h4.2, hpos, h140]
  have hmulmax : |w2e.item.toRat * qinv.item.toRat| ≤ Format.binary32.maxVal := by
    refine Rat.cast_le (K := ℝ).mp ?_
    rw [Rat.cast_abs, Rat.cast_mul]
    calc |(w2e.item.toRat : ℝ) * (qinv.item.toRat : ℝ)|
        = |w2e.item.toReal| * |qinv.item.toReal| := by rw [← abs_mul]; rfl
      _ ≤ 2 ^ (100 : ℤ) * 2 := by
          have h2' : (0 : ℝ) ≤ (2 : ℝ) ^ (100 : ℤ) := by norm_num
          exact mul_le_mul hw2eabs hqinv2 (abs_nonneg _) h2'
      _ ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by rw [maxVal32_eq]; norm_num
  obtain ⟨hw1fin, hw1err⟩ :=
    FloatV.mul_err (f := Format.binary32) (x := w2e.item) (y := qinv.item)
      bin32_regular hw2efin hqinvfin hmulmax
  have hw1v' : FloatV.NaNEq (FloatV.mul w2e.item qinv.item) w1v.item := hw1v ()
  have hw1_eq : w1v.item = FloatV.mul w2e.item qinv.item :=
    hw1v'.eq (Format.not_isNaN_of_isFinite hw1fin)
  have hw1err' : |w1v.item.toReal - w2e.item.toReal * qinv.item.toReal| ≤
      |w2e.item.toReal * qinv.item.toReal| * u32 + eta32 := by
    have hcast : (((|(FloatV.mul w2e.item qinv.item).toRat -
          w2e.item.toRat * qinv.item.toRat| : ℚ) : ℝ))
        ≤ (((|w2e.item.toRat * qinv.item.toRat| : ℚ) *
            (Format.binary32.u : ℚ) + (Format.binary32.eta : ℚ) : ℚ) : ℝ) :=
      Rat.cast_le (K := ℝ).mpr hw1err
    rw [hw1_eq]
    push_cast at hcast
    rw [u32_eq, eta32_eq] at hcast
    exact hcast
  -- ## 9. `w1` against `S`
  have hPbound : |w2e.item.toReal * qinv.item.toReal - S| ≤
      S * (Ew + 3 / 200000 + Ew * (3 / 200000)) +
        ((2 : ℝ) ^ (-140 : ℤ) / D) * (1 + 3 / 200000) := by
    have hEq : w2e.item.toReal * qinv.item.toReal - S
        = (w2e.item.toReal - Real.exp (c1 - c0)) / D
          + Real.exp (c1 - c0) * (qinv.item.toReal - 1 / D)
          + (w2e.item.toReal - Real.exp (c1 - c0)) * (qinv.item.toReal - 1 / D) := by
      rw [hSdef]; field_simp; try ring
    rw [hEq]
    have ht1 : |(w2e.item.toReal - Real.exp (c1 - c0)) / D| ≤
        Ew * S + (2 : ℝ) ^ (-140 : ℤ) / D := by
      rw [abs_div, abs_of_pos hDpos]
      have h := div_le_div_of_nonneg_right hEwbound hDpos.le
      have he : (Ew * Real.exp (c1 - c0) + 2 ^ (-140 : ℤ)) / D
          = Ew * S + 2 ^ (-140 : ℤ) / D := by
        rw [hSdef]; field_simp
      linarith
    have ht2 : |Real.exp (c1 - c0) * (qinv.item.toReal - 1 / D)| ≤ (3 / 200000) * S := by
      rw [abs_mul, abs_of_pos (Real.exp_pos (c1 - c0))]
      have h := mul_le_mul_of_nonneg_left hZ (Real.exp_pos (c1 - c0)).le
      have he : Real.exp (c1 - c0) * (((3 : ℝ) / 200000) / D) = (3 / 200000) * S := by
        rw [hSdef]; field_simp
      linarith
    have ht3 : |(w2e.item.toReal - Real.exp (c1 - c0)) * (qinv.item.toReal - 1 / D)| ≤
        Ew * (3 / 200000) * S + (2 : ℝ) ^ (-140 : ℤ) * (3 / 200000) / D := by
      rw [abs_mul]
      have hnn : 0 ≤ Ew * Real.exp (c1 - c0) + 2 ^ (-140 : ℤ) := by
        have h := (Real.exp_pos (c1 - c0)).le
        have h2 : (0 : ℝ) < 2 ^ (-140 : ℤ) := by norm_num
        nlinarith [hEwnn, h, h2]
      have h := mul_le_mul hEwbound hZ (abs_nonneg _) hnn
      have he : (Ew * Real.exp (c1 - c0) + 2 ^ (-140 : ℤ)) * (((3 : ℝ) / 200000) / D)
          = Ew * (3 / 200000) * S + (2 : ℝ) ^ (-140 : ℤ) * (3 / 200000) / D := by
        rw [hSdef]; field_simp
      linarith
    have hparts : |(w2e.item.toReal - Real.exp (c1 - c0)) / D
          + Real.exp (c1 - c0) * (qinv.item.toReal - 1 / D)
          + (w2e.item.toReal - Real.exp (c1 - c0)) * (qinv.item.toReal - 1 / D)| ≤
        (Ew * S + (2 : ℝ) ^ (-140 : ℤ) / D) + (3 / 200000) * S
          + (Ew * (3 / 200000) * S + (2 : ℝ) ^ (-140 : ℤ) * (3 / 200000) / D) := by
      have ha := abs_add_le ((w2e.item.toReal - Real.exp (c1 - c0)) / D
          + Real.exp (c1 - c0) * (qinv.item.toReal - 1 / D))
        ((w2e.item.toReal - Real.exp (c1 - c0)) * (qinv.item.toReal - 1 / D))
      have hb := abs_add_le ((w2e.item.toReal - Real.exp (c1 - c0)) / D)
        (Real.exp (c1 - c0) * (qinv.item.toReal - 1 / D))
      linarith [ha, hb, ht1, ht2, ht3]
    have hcomb : (Ew * S + (2 : ℝ) ^ (-140 : ℤ) / D) + (3 / 200000) * S
          + (Ew * (3 / 200000) * S + (2 : ℝ) ^ (-140 : ℤ) * (3 / 200000) / D)
        = S * (Ew + 3 / 200000 + Ew * (3 / 200000))
          + ((2 : ℝ) ^ (-140 : ℤ) / D) * (1 + 3 / 200000) := by ring
    linarith [hparts, hcomb]
  set K1 : ℝ := Ew + 3 / 200000 + Ew * (3 / 200000) with hK1def
  set T : ℝ := ((2 : ℝ) ^ (-140 : ℤ) / D) * (1 + 3 / 200000) with hTdef
  have hK1le : K1 ≤ 571 / 10000000 := by
    rw [hK1def]
    have hnum : 42 / 1000000 + 3 / 200000 + 42 / 1000000 * (3 / 200000) ≤ 571 / 10000000 := by
      norm_num
    nlinarith [hEwle, hEwnn, hnum]
  have hTle : T ≤ 2 ^ (-139 : ℤ) := by
    rw [hTdef]
    have h140 : (0 : ℝ) < 2 ^ (-140 : ℤ) := by norm_num
    have h1 : (2 : ℝ) ^ (-140 : ℤ) / D ≤ 2 ^ (-140 : ℤ) := by
      rw [div_le_iff₀ hDpos]
      nlinarith [hD1, h140]
    have h2 : (2 : ℝ) ^ (-140 : ℤ) * (1 + 3 / 200000) ≤ (2 : ℝ) ^ (-139 : ℤ) := by
      have h3 : (2 : ℝ) ^ (-139 : ℤ) = 2 * 2 ^ (-140 : ℤ) := by
        rw [show (-139 : ℤ) = -140 + 1 by norm_num, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
        norm_num
      rw [h3]; nlinarith
    nlinarith [h1, h2, h140]
  have hPbound' : |w2e.item.toReal * qinv.item.toReal - S| ≤ S * K1 + T := by
    have h := hPbound
    have he : S * (Ew + 3 / 200000 + Ew * (3 / 200000)) +
        ((2 : ℝ) ^ (-140 : ℤ) / D) * (1 + 3 / 200000) = S * K1 + T := by
      rw [hK1def, hTdef]
    linarith [h, he]
  -- ## 10. assemble
  have hPabs : |w2e.item.toReal * qinv.item.toReal| ≤ S + (S * K1 + T) := by
    have h1 : |w2e.item.toReal * qinv.item.toReal| ≤
        |S| + |w2e.item.toReal * qinv.item.toReal - S| := by
      have h := abs_add_le S (w2e.item.toReal * qinv.item.toReal - S)
      have he : S + (w2e.item.toReal * qinv.item.toReal - S)
          = w2e.item.toReal * qinv.item.toReal := by ring
      rw [he] at h
      exact h
    rw [abs_of_nonneg hSnn] at h1
    linarith [h1, hPbound']
  have hstep1 : |w1v.item.toReal - w2e.item.toReal * qinv.item.toReal| ≤
      (S + (S * K1 + T)) * u32 + eta32 := by
    have h := hw1err'
    have h2 : |w2e.item.toReal * qinv.item.toReal| * u32 ≤
        (S + (S * K1 + T)) * u32 := mul_le_mul_of_nonneg_right hPabs u32_pos.le
    linarith [h, h2]
  have hcoef : u32 + u32 * (571 / 10000000) + 571 / 10000000 ≤ wRel1 := by
    unfold u32 wRel1; norm_num
  have hcoef' : u32 + u32 * K1 + K1 ≤ wRel1 := by
    have h2 : u32 * K1 ≤ u32 * (571 / 10000000) := mul_le_mul_of_nonneg_left hK1le u32_pos.le
    linarith [h2, hcoef]
  have hfloor : T + u32 * T + eta32 ≤ wFloor := by
    have hnum : (2 : ℝ) ^ (-139 : ℤ) + u32 * 2 ^ (-139 : ℤ) + eta32 ≤ wFloor := by
      unfold u32 eta32 wFloor; norm_num
    have h2 : u32 * T ≤ u32 * (2 : ℝ) ^ (-139 : ℤ) := mul_le_mul_of_nonneg_left hTle u32_pos.le
    linarith [h2, hnum]
  have hmain : |w1v.item.toReal - S| ≤ S * wRel1 + wFloor := by
    have h1 := abs_sub_le w1v.item.toReal (w2e.item.toReal * qinv.item.toReal) S
    have hS := mul_le_mul_of_nonneg_left hcoef' hSnn
    have h3 : (S + (S * K1 + T)) * u32 + eta32 + (S * K1 + T)
        = S * (u32 + u32 * K1 + K1) + (T + u32 * T + eta32) := by ring
    have h2 : (S + (S * K1 + T)) * u32 + eta32 + (S * K1 + T) ≤ S * wRel1 + wFloor := by
      rw [h3]
      linarith [hS, hfloor]
    linarith [h1, hstep1, hPbound', h2]
  refine ⟨?_, ?_⟩
  · rw [hw1_eq]; exact hw1fin
  · rw [hsoft]
    simpa only [mul_comm] using hmain

end Claims.MoeRouter
