import Claim
import TVA.Proof
import TVA
import Proof.NumFacts
import Proof.NumWide
import Proof.NumSlack
import Proof.NumMerge

/-!
# The lse stored by the merge branch

`log o`, the addition of the maximum, and the `−∞` clamp.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## Elementary bounds on the reference values -/

/-- A side's weight is nonnegative. -/
theorem side_nonneg (x : F32) (m : ℝ) : 0 ≤ side x m := by
  unfold side
  split_ifs with h
  · norm_num
  · exact (Real.exp_pos _).le

/-- With a nonempty side, the reference sum is at least `1`. -/
theorem one_le_refSum {p s : F32} (hne : ¬ (p = negInfC ∧ s = negInfC)) :
    1 ≤ refSum p s := by
  have hpn : 0 ≤ side p (refMax p s) := side_nonneg p _
  have hsn : 0 ≤ side s (refMax p s) := side_nonneg s _
  rcases eq_or_ne p negInfC with hp | hp
  · have hs : s ≠ negInfC := fun h => hne ⟨hp, h⟩
    have hm : refMax p s = s.toReal := by rw [hp]; exact refMax_negInf_left s
    have hside : side s (refMax p s) = 1 := by
      rw [side_of_ne hs, hm]; simp
    have hzero : side p (refMax p s) = 0 := by rw [hp]; exact side_negInf _
    unfold refSum
    rw [hzero, hside]
    norm_num
  · rcases eq_or_ne s negInfC with hs | hs
    · have hm : refMax p s = p.toReal := by
        simp [refMax, hp, hs]
      have hside : side p (refMax p s) = 1 := by
        rw [side_of_ne hp, hm]; simp
      have hzero : side s (refMax p s) = 0 := by rw [hs]; exact side_negInf _
      unfold refSum
      rw [hzero, hside]
      norm_num
    · have hm : refMax p s = max p.toReal s.toReal := refMax_eq_max hp hs
      rcases le_total p.toReal s.toReal with h | h
      · have hside : side s (refMax p s) = 1 := by
          rw [side_of_ne hs, hm, max_eq_right h]; simp
        unfold refSum
        rw [hside]
        linarith
      · have hside : side p (refMax p s) = 1 := by
          rw [side_of_ne hp, hm, max_eq_left h]; simp
        unfold refSum
        rw [hside]
        linarith

/-! ## The abstract triangle bound -/

/-- The merged lse's final error bound: a relative error `u` of a sum with offset `eta`, and a constant `d`, stay within `T · (1 + |c|)`. -/
theorem merge_lse_final {x y c u eta d T : ℝ}
    (hu0 : 0 ≤ u) (hu : u ≤ 1 / 2) (_heta : 0 ≤ eta) (_hd : 0 ≤ d)
    (hcoef : 2 * u ≤ T) (hconst : (eta + d) * (1 + 2 * u) ≤ T)
    (h1 : |x - y| ≤ |x| * u + eta) (h2 : |y - c| ≤ d) :
    |x - c| ≤ T * (1 + |c|) := by
  have htri : |x - c| ≤ |x| * u + eta + d := by
    have he : x - c = (x - y) + (y - c) := by ring
    calc |x - c| = |(x - y) + (y - c)| := by rw [he]
      _ ≤ |x - y| + |y - c| := abs_add_le _ _
      _ ≤ (|x| * u + eta) + d := add_le_add h1 h2
  have hxle : |x| ≤ |x| * u + eta + |c| + d := by
    have he : x = (x - y) + (y - c) + c := by ring
    have h4 : |x| = |(x - y) + (y - c) + c| := congrArg (fun z : ℝ => |z|) he
    have h5 : |(x - y) + (y - c) + c| ≤ |(x - y) + (y - c)| + |c| := abs_add_le _ _
    have h6 : |(x - y) + (y - c)| ≤ |x - y| + |y - c| := abs_add_le _ _
    linarith
  have hx2 : |x| ≤ 2 * |c| + 2 * (eta + d) := by
    have hfac : |x| * (1 - u) ≤ eta + |c| + d := by nlinarith [hxle]
    have hhalf : (1 : ℝ) / 2 ≤ 1 - u := by linarith
    have hnn : 0 ≤ |x| := abs_nonneg x
    nlinarith [mul_le_mul_of_nonneg_left hhalf hnn, hfac]
  have hstep : |x| * u ≤ T * |c| + 2 * u * (eta + d) := by
    have hm1 : |x| * u ≤ (2 * |c| + 2 * (eta + d)) * u :=
      mul_le_mul_of_nonneg_right hx2 hu0
    have h2u : 2 * u * |c| ≤ T * |c| := by
      have h := mul_le_mul_of_nonneg_right hcoef (abs_nonneg c)
      linarith
    nlinarith [hm1]
  calc |x - c| ≤ |x| * u + eta + d := htri
    _ ≤ (T * |c| + 2 * u * (eta + d)) + eta + d := by linarith
    _ = T * |c| + (eta + d) * (1 + 2 * u) := by ring
    _ ≤ T * |c| + T := by linarith
    _ = T * (1 + |c|) := by ring

/-- **The lse stored by the merge branch.** -/
theorem merge_lse_out {P : Platform} (hL : Device.LogF32 P)
    {p s M u v pe se o l cl0 cl : F32}
    (hm : LseMid p s M u v pe se o) (hne : ¬ (p = negInfC ∧ s = negInfC))
    (hl : P.math .log .binary32 o l)
    (hc : FloatV.NaNEq (FloatV.add l M) cl0)
    (hcl : cl = if FloatV.eq M negInfC then negInfC else cl0) :
    cl.isFinite = true ∧
      |cl.toReal - refLse p s| ≤ (lseTol : ℝ) * (1 + |refLse p s|) := by
  have hMne : M ≠ negInfC := by
    intro h
    have hfin := hm.hMfin
    rw [h, negInfC_isFinite] at hfin
    exact Bool.false_ne_true hfin
  have heq_false : FloatV.eq M negInfC = false := by
    cases h : FloatV.eq M negInfC with
    | false => rfl
    | true => exact absurd ((eq_negInfC_iff hm.hMn).mp h) hMne
  have hcl_eq : cl = cl0 := by
    rw [hcl, heq_false]
    simp
  rw [hcl_eq]
  have hTol : (lseTol : ℝ) = (2 : ℝ) ^ (-16 : ℤ) := by
    norm_num [lseTol]
  have u_val : (Format.binary32.u : ℝ) = (2 : ℝ) ^ (-24 : ℤ) := by
    rw [Format.binary32_u]
    norm_num
  have eta_val : (Format.binary32.eta : ℝ) = (2 : ℝ) ^ (-150 : ℤ) := by
    rw [Format.binary32_eta]
    norm_num
  have hmax3q : (3 : ℚ) ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  have hmax3 : (3 : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast hmax3q
  have hMV0 : (0 : ℚ) ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  -- the reference sum is positive
  have hSpos : 0 < refSum p s := by
    have h := one_le_refSum hne
    linarith
  have ho_pos : (0 : ℝ) < o.toReal := by
    have h0 : (0 : ℝ) < 1 - (2 : ℝ) ^ (-18 : ℤ) := by norm_num
    linarith [hm.ho0]
  -- the logarithm of `o` is small
  have hLabs : |Real.log o.toReal| ≤ 3 / 4 := by
    have hupper : Real.log o.toReal ≤ 3 / 4 := by
      have hop : (0 : ℝ) < 1 + (2 : ℝ) ^ (-17 : ℤ) := by norm_num
      have he : (2 : ℝ) * (1 + (2 : ℝ) ^ (-17 : ℤ)) = 2 + (2 : ℝ) ^ (-16 : ℤ) := by
        norm_num
      have h1 : o.toReal ≤ 2 * (1 + (2 : ℝ) ^ (-17 : ℤ)) := by linarith [hm.ho1, he]
      have hlog := Real.log_le_log ho_pos h1
      have hsplit : Real.log (2 * (1 + (2 : ℝ) ^ (-17 : ℤ))) =
          Real.log 2 + Real.log (1 + (2 : ℝ) ^ (-17 : ℤ)) := by
        rw [Real.log_mul (by norm_num : (2 : ℝ) ≠ 0) (by linarith : (1 + (2 : ℝ) ^ (-17 : ℤ)) ≠ 0)]
      have hlog1 : Real.log (1 + (2 : ℝ) ^ (-17 : ℤ)) ≤ (2 : ℝ) ^ (-17 : ℤ) := by
        have h := Real.log_le_sub_one_of_pos hop
        linarith
      have hlog2 : Real.log 2 ≤ 3 / 4 - (2 : ℝ) ^ (-17 : ℤ) := by
        have h17 : (2 : ℝ) ^ (-17 : ℤ) ≤ 1 / 64 := by norm_num
        linarith [Real.log_two_lt_d9, h17]
      calc Real.log o.toReal ≤ Real.log (2 * (1 + (2 : ℝ) ^ (-17 : ℤ))) := hlog
        _ = Real.log 2 + Real.log (1 + (2 : ℝ) ^ (-17 : ℤ)) := hsplit
        _ ≤ (3 / 4 - (2 : ℝ) ^ (-17 : ℤ)) + (2 : ℝ) ^ (-17 : ℤ) := add_le_add hlog2 hlog1
        _ = 3 / 4 := by ring
    have hlower : -(3 / 4 : ℝ) ≤ Real.log o.toReal := by
      have h1 : (0 : ℝ) < 1 - (2 : ℝ) ^ (-18 : ℤ) := by norm_num
      have hstep1 : Real.log (1 - (2 : ℝ) ^ (-18 : ℤ)) ≤ Real.log o.toReal :=
        Real.log_le_log h1 hm.ho0
      have hhalf : (1 / 2 : ℝ) ≤ 1 - (2 : ℝ) ^ (-18 : ℤ) := by norm_num
      have hstep2 : Real.log (1 / 2 : ℝ) ≤ Real.log (1 - (2 : ℝ) ^ (-18 : ℤ)) :=
        Real.log_le_log (by norm_num) hhalf
      have hlog2 : Real.log (1 / 2 : ℝ) = -Real.log 2 := by
        rw [show (1 / 2 : ℝ) = 2⁻¹ by norm_num, Real.log_inv]
      have hlt : Real.log 2 ≤ 3 / 4 :=
        le_of_lt (lt_of_lt_of_le Real.log_two_lt_d9 (by norm_num))
      have hval : -(3 / 4 : ℝ) ≤ Real.log (1 / 2 : ℝ) := by
        rw [hlog2]
        exact neg_le_neg hlt
      calc -(3 / 4 : ℝ) ≤ Real.log (1 / 2 : ℝ) := hval
        _ ≤ Real.log (1 - (2 : ℝ) ^ (-18 : ℤ)) := hstep2
        _ ≤ Real.log o.toReal := hstep1
    exact abs_le.mpr ⟨hlower, hupper⟩
  -- the accuracy of the logarithm
  obtain ⟨ref, href, hacc⟩ := hL
  have href' : ref = Real.log := by
    have h := href
    simp only [TVA.MathFn.ref] at h
    exact (Option.some.inj h).symm
  rw [href'] at hacc
  have hmem : o.toReal ∈ TVA.MathFn.natDom .log := by
    change (0 : ℝ) < o.toReal
    exact ho_pos
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log o.toReal| + (2 : ℝ) ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    have hc2 : (1 + (2 : ℝ) ^ (-22 : ℤ)) ≤ 2 := by norm_num
    have hmul : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log o.toReal| ≤ 2 * (3 / 4) :=
      mul_le_mul hc2 hLabs (abs_nonneg _) (by norm_num)
    calc (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log o.toReal| + (2 : ℝ) ^ (-149 : ℤ)
        ≤ 2 * (3 / 4) + (2 : ℝ) ^ (-149 : ℤ) := add_le_add hmul (le_refl _)
      _ ≤ 3 := by norm_num
      _ ≤ (Format.binary32.maxVal : ℝ) := hmax3
  obtain ⟨hlfin, hlerr⟩ := hacc o l hl hm.hofin trivial hmem hroom
  have hlerr2 : |l.toReal - Real.log o.toReal| ≤ (2 : ℝ) ^ (-21 : ℤ) := by
    have hmul : (2 : ℝ) ^ (-22 : ℤ) * |Real.log o.toReal| ≤ (2 : ℝ) ^ (-22 : ℤ) * (3 / 4) :=
      mul_le_mul_of_nonneg_left hLabs (by norm_num)
    calc |l.toReal - Real.log o.toReal|
        ≤ (2 : ℝ) ^ (-22 : ℤ) * |Real.log o.toReal| + (2 : ℝ) ^ (-149 : ℤ) := hlerr
      _ ≤ (2 : ℝ) ^ (-22 : ℤ) * (3 / 4) + (2 : ℝ) ^ (-149 : ℤ) := add_le_add hmul (le_refl _)
      _ ≤ (2 : ℝ) ^ (-21 : ℤ) := by norm_num
  have hl_abs : |l.toReal| ≤ 1 := by
    have he : l.toReal = (l.toReal - Real.log o.toReal) + Real.log o.toReal := by ring
    have h4 : |l.toReal| = |(l.toReal - Real.log o.toReal) + Real.log o.toReal| :=
      congrArg (fun z : ℝ => |z|) he
    have htri : |(l.toReal - Real.log o.toReal) + Real.log o.toReal| ≤
        |l.toReal - Real.log o.toReal| + |Real.log o.toReal| := abs_add_le _ _
    calc |l.toReal| = |(l.toReal - Real.log o.toReal) + Real.log o.toReal| := h4
      _ ≤ |l.toReal - Real.log o.toReal| + |Real.log o.toReal| := htri
      _ ≤ (2 : ℝ) ^ (-21 : ℤ) + 3 / 4 := add_le_add hlerr2 hLabs
      _ ≤ 1 := by norm_num
  have hl1q : |l.toRat| ≤ 1 := by
    have hcast : ((|l.toRat| : ℚ) : ℝ) ≤ ((1 : ℚ) : ℝ) := by
      rw [Rat.cast_abs, Rat.cast_one]
      simpa [FloatV.toReal] using hl_abs
    exact Rat.cast_le.mp hcast
  -- the exact sum against the reference lse
  have hchain : |(l.toReal + M.toReal) - refLse p s| ≤
      (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-17 : ℤ) := by
    have hS : |Real.log (refSum p s) - Real.log o.toReal| ≤ (2 : ℝ) ^ (-17 : ℤ) :=
      log_close hSpos hm.ho0 ((abs_sub_comm _ _).trans_le hm.hpo)
    have hs2 : |Real.log o.toReal - Real.log (refSum p s)| ≤ (2 : ℝ) ^ (-17 : ℤ) :=
      (abs_sub_comm _ _).trans_le hS
    have he : (l.toReal + M.toReal) - refLse p s =
        (l.toReal - Real.log o.toReal) + (Real.log o.toReal - Real.log (refSum p s)) := by
      unfold refLse
      rw [← hm.hM]
      ring
    rw [he]
    calc |(l.toReal - Real.log o.toReal) + (Real.log o.toReal - Real.log (refSum p s))|
        ≤ |l.toReal - Real.log o.toReal| + |Real.log o.toReal - Real.log (refSum p s)| :=
          abs_add_le _ _
      _ ≤ (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-17 : ℤ) := add_le_add hlerr2 hs2
  by_cases hq0 : l.toRat + M.toRat = 0
  · -- exact cancellation: the sum is zero
    have hmax0 : |l.toRat + M.toRat| ≤ Format.binary32.maxVal := by
      rw [hq0]
      simpa using hMV0
    obtain ⟨hfin_add, herr_add⟩ :=
      FloatV.add_err (f := Format.binary32) Format.binary32_regular hlfin hm.hMfin hmax0
    have herr_add' : |(FloatV.add l M).toRat| ≤ Format.binary32.eta := by
      have h := herr_add
      rw [hq0] at h
      simpa using h
    have hnan_add : (FloatV.add l M).isNaN = false := (FloatV.isFinite_iff.mp hfin_add).1
    have hcl0eq : cl0 = FloatV.add l M := FloatV.NaNEq.eq hc hnan_add
    have hfin0 : cl0.isFinite = true := by rw [hcl0eq]; exact hfin_add
    have hsmall : |cl0.toReal| ≤ (2 : ℝ) ^ (-150 : ℤ) := by
      have hcast : ((|(FloatV.add l M).toRat| : ℚ) : ℝ) ≤ ((Format.binary32.eta : ℚ) : ℝ) :=
        Rat.cast_le.mpr herr_add'
      rw [eta_val] at hcast
      rw [hcl0eq]
      simpa [FloatV.toReal] using hcast
    have hysum : l.toReal + M.toReal = 0 := by
      have hq : ((l.toRat + M.toRat : ℚ) : ℝ) = 0 := by rw [hq0]; norm_num
      push_cast at hq
      simpa [FloatV.toReal] using hq
    have h1 : |cl0.toReal - (l.toReal + M.toReal)| ≤
        |cl0.toReal| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
      rw [hysum, sub_zero]
      have hnn : (0 : ℝ) ≤ |cl0.toReal| * (2 : ℝ) ^ (-24 : ℤ) :=
        mul_nonneg (abs_nonneg _) (by norm_num)
      calc |cl0.toReal| ≤ (2 : ℝ) ^ (-150 : ℤ) := hsmall
        _ ≤ (2 : ℝ) ^ (-150 : ℤ) + |cl0.toReal| * (2 : ℝ) ^ (-24 : ℤ) :=
            le_add_of_nonneg_right hnn
        _ = |cl0.toReal| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by ring
    refine ⟨hfin0, ?_⟩
    have hmain := merge_lse_final (x := cl0.toReal) (y := l.toReal + M.toReal)
      (c := refLse p s) (u := (2 : ℝ) ^ (-24 : ℤ)) (eta := (2 : ℝ) ^ (-150 : ℤ))
      (d := (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-17 : ℤ)) (T := (2 : ℝ) ^ (-16 : ℤ))
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      h1 hchain
    rw [hTol]
    exact hmain
  · -- the sum is nonzero: the slack rounding bound
    have hq : |l.toRat + M.toRat| ≤ Format.binary32.maxVal + 1 := by
      have h1 : |l.toRat + M.toRat| ≤ |l.toRat| + |M.toRat| := abs_add_le _ _
      have h2 : |M.toRat| ≤ Format.binary32.maxVal := finite_abs_le_maxVal hm.hMfin
      linarith
    obtain ⟨hfin0, herr0⟩ :=
      add_err_slack (l := l) (M := M) (r := cl0) hlfin hm.hMfin hc hq0 hq
    have h1 : |cl0.toReal - (l.toReal + M.toReal)| ≤
        |cl0.toReal| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
      have hcast : ((|cl0.toRat - (l.toRat + M.toRat)| : ℚ) : ℝ) ≤
          ((|cl0.toRat| * Format.binary32.u + Format.binary32.eta : ℚ) : ℝ) :=
        Rat.cast_le.mpr herr0
      push_cast at hcast
      rw [u_val, eta_val] at hcast
      simpa [FloatV.toReal] using hcast
    refine ⟨hfin0, ?_⟩
    have hmain := merge_lse_final (x := cl0.toReal) (y := l.toReal + M.toReal)
      (c := refLse p s) (u := (2 : ℝ) ^ (-24 : ℤ)) (eta := (2 : ℝ) ^ (-150 : ℤ))
      (d := (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-17 : ℤ)) (T := (2 : ℝ) ^ (-16 : ℤ))
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      h1 hchain
    rw [hTol]
    exact hmain

end Claims.MergeAttnStates
