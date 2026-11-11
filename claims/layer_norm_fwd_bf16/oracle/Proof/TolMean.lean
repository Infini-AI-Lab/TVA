import Proof.TolBase

/-! The stored mean's tolerance: `tl.sum1` over the row's `BD` lanes (zeros past `D`), the
division by `D`, and the input's magnitude bound. This is README `## Tolerances`, step 1. -/

namespace Claims.LayerNormFwd

open TVA
open scoped BigOperators

noncomputable section

/-- `(1 + u)^L - 1 ≤ 2·L·u` for `L ≤ 9` and `18u ≤ 1`. -/
theorem mean_tol_pow_bound {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K] {u : K}
    (hu0 : 0 ≤ u) (hu : 18 * u ≤ 1) :
    ∀ L : ℕ, L ≤ 9 → (1 + u) ^ L - 1 ≤ (L : K) * 2 * u := by
  intro L hL
  induction L with
  | zero => simp
  | succ n ih =>
    have hn : n ≤ 9 := by omega
    have ih' := ih hn
    have hnn : (n : K) ≤ 9 := by exact_mod_cast hn
    have hstep : (1 + u) ^ (n + 1) - 1 = (1 + u) * ((1 + u) ^ n - 1) + u := by
      rw [pow_succ]; ring
    rw [hstep]
    have h1 : (1 + u) * ((1 + u) ^ n - 1) ≤ (1 + u) * ((n : K) * 2 * u) := by
      apply mul_le_mul_of_nonneg_left ih'
      linarith
    have h2 : (1 + u) * ((n : K) * 2 * u) + u ≤ ((n + 1 : ℕ) : K) * 2 * u := by
      have hnu : 2 * (n : K) * u ≤ 1 := by nlinarith [hnn, hu]
      have hnu2 : 2 * (n : K) * u * u ≤ u := by nlinarith [hnu, hu0]
      push_cast
      nlinarith [hnu2]
    push_cast at h2 ⊢
    linarith [h1, h2]


/-- The final real-valued assembly of the mean's error bound: a quotient `Q` known to be
within `L·2^-23·A` of the exact mean, and within `A`-scale of it, plus the division's own
error `2^-22·|Q|`. -/
lemma mean_tol_final (Q m b A L : ℝ) (hA0 : 0 ≤ A)
    (hLk : L * (2:ℝ)^(-23:ℤ) ≤ 1)
    (hQm : |Q - m| ≤ L * (2:ℝ)^(-23:ℤ) * A + (2:ℝ)^(-148:ℤ))
    (hQ_abs : |Q| ≤ A + (L * (2:ℝ)^(-23:ℤ) * A + (2:ℝ)^(-148:ℤ)))
    (hbmerrR : |b - Q| ≤ (2:ℝ)^(-22:ℤ) * |Q| + (2:ℝ)^(-148:ℤ))
    (htri : |b - m| ≤ |b - Q| + |Q - m|) :
    |b - m| ≤ (L + 4) * (2:ℝ)^(-23:ℤ) * A + (2:ℝ)^(-140:ℤ) := by
  have h22 : (2:ℝ)^(-22:ℤ) = 2 * (2:ℝ)^(-23:ℤ) := by norm_num
  have hk : (0:ℝ) < (2:ℝ)^(-23:ℤ) := by norm_num
  have hc2 : (1 + (2:ℝ)^(-22:ℤ)) * (2:ℝ)^(-148:ℤ) + (2:ℝ)^(-148:ℤ) ≤ (2:ℝ)^(-140:ℤ) := by
    norm_num
  have hkA : (0:ℝ) ≤ 2 * (2:ℝ)^(-23:ℤ) * A := mul_nonneg (by positivity) hA0
  have hprod : 2 * (2:ℝ)^(-23:ℤ) * A * (L * (2:ℝ)^(-23:ℤ)) ≤ 2 * (2:ℝ)^(-23:ℤ) * A := by
    simpa using mul_le_mul_of_nonneg_left hLk hkA
  have hA3' : (2:ℝ)^(-22:ℤ) * |Q| ≤
      (2:ℝ)^(-22:ℤ) * (A + (L * (2:ℝ)^(-23:ℤ) * A + (2:ℝ)^(-148:ℤ))) :=
    mul_le_mul_of_nonneg_left hQ_abs (by norm_num)
  have hcoefA : ((2:ℝ)^(-22:ℤ) + (1 + (2:ℝ)^(-22:ℤ)) * (L * (2:ℝ)^(-23:ℤ))) * A
      ≤ (L + 4) * (2:ℝ)^(-23:ℤ) * A := by
    have hc1 : (2:ℝ)^(-22:ℤ) + (1 + (2:ℝ)^(-22:ℤ)) * (L * (2:ℝ)^(-23:ℤ))
        ≤ (L + 4) * (2:ℝ)^(-23:ℤ) := by
      nlinarith [h22, hk, hLk]
    exact mul_le_mul_of_nonneg_right hc1 hA0
  linarith [htri, hbmerrR, hQm, hA3', hcoefA, hc2, hprod]

set_option maxHeartbeats 800000 in
/-- **The `mean` tolerance**: the stored mean is within `meanTol` of `meanRef`, from the tree sum and the device fact for `/`. -/
theorem mean_tol {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    {P : Platform} (hdivf : Device.DivFull P)
    {bx : Nat → F32} {dv bsum bmean : F32}
    (hdv : dv.toReal = (d.D : ℝ)) (hdvfin : dv.isFinite = true)
    (hfin : ∀ j, j < d.BD → (bx j).isFinite = true)
    (hval : ∀ j, j < d.D → (bx j).toReal = sR d init t j)
    (hzero : ∀ j, j < d.BD → d.D ≤ j → (bx j).toReal = 0)
    (hbdd : ∀ j, j < d.D → |sR d init t j| ≤ 2 ^ 57)
    (hcover : ∃ tr : Tree, Tree.Covers d.BD tr ∧ tr.depth ≤ Nat.clog 2 d.BD ∧
      Tree.Eval (AddRel false) bx tr bsum)
    (hdiv : P.div .binary32 bsum dv bmean) :
    bmean.isFinite = true ∧ |bmean.toReal - meanRef d init t| ≤ meanTol d init t := by
  -- basic facts about the instance
  have hDpos : 0 < d.D := hd.d_pos
  have hBDpos : 0 < d.BD := lt_of_lt_of_le hDpos hd.d_le_bd
  obtain ⟨e, he⟩ := hd.bd_pow2
  have he_le : e ≤ 9 := by
    have hlt : (2 : ℕ) ^ e < 2 ^ 10 := by
      rw [← he]
      have h1 := hd.bd_lt
      have h2 := hd.d_le
      norm_num
      omega
    have h10 : e < 10 := (Nat.pow_lt_pow_iff_right (by norm_num)).mp hlt
    omega
  have hBD_le : d.BD ≤ 512 := by
    rw [he]
    calc 2 ^ e ≤ 2 ^ 9 := Nat.pow_le_pow_right (by norm_num) he_le
      _ = 512 := by norm_num
  have hdepth_eq : depth d = e := by
    unfold depth
    rw [he, Nat.clog_pow 2 e (by norm_num)]
  have hdepth_le : depth d ≤ 9 := hdepth_eq ▸ he_le
  -- the zeros past `D` at the level of `toRat`
  have hzeroQ : ∀ j, j < d.BD → d.D ≤ j → (bx j).toRat = 0 := by
    intro j hj hDj
    have h : ((bx j).toRat : ℝ) = 0 := by
      have h' := hzero j hj hDj
      simpa [FloatV.toReal] using h'
    exact_mod_cast h
  -- the sums of `sum_err_depth`
  have hTq_range : ((List.range d.BD).map fun i => (bx i).toRat).sum
      = ∑ j ∈ Finset.range d.BD, (bx j).toRat := by
    rw [TVA.sum_range_fin]
    exact Fin.sum_univ_eq_sum_range (fun i => (bx i).toRat) d.BD
  have hSq_range : ((List.range d.BD).map fun i => |(bx i).toRat|).sum
      = ∑ j ∈ Finset.range d.BD, |(bx j).toRat| := by
    rw [TVA.sum_range_fin]
    exact Fin.sum_univ_eq_sum_range (fun i => |(bx i).toRat|) d.BD
  have hsplitT : ∑ j ∈ Finset.range d.BD, (bx j).toRat
      = ∑ j ∈ Finset.range d.D, (bx j).toRat := by
    rw [show d.BD = d.D + (d.BD - d.D) from (Nat.add_sub_cancel' hd.d_le_bd).symm]
    rw [Finset.sum_range_add]
    have h0 : ∀ x ∈ Finset.range (d.BD - d.D), (bx (d.D + x)).toRat = 0 := by
      intro x hx
      rw [Finset.mem_range] at hx
      exact hzeroQ (d.D + x) (by omega) (by omega)
    rw [Finset.sum_eq_zero h0, add_zero]
  have hsplitS : ∑ j ∈ Finset.range d.BD, |(bx j).toRat|
      = ∑ j ∈ Finset.range d.D, |(bx j).toRat| := by
    rw [show d.BD = d.D + (d.BD - d.D) from (Nat.add_sub_cancel' hd.d_le_bd).symm]
    rw [Finset.sum_range_add]
    have h0 : ∀ x ∈ Finset.range (d.BD - d.D), |(bx (d.D + x)).toRat| = 0 := by
      intro x hx
      rw [Finset.mem_range] at hx
      rw [hzeroQ (d.D + x) (by omega) (by omega), abs_zero]
    rw [Finset.sum_eq_zero h0, add_zero]
  -- the real value of the two sums
  have hT_cast : (((List.range d.BD).map fun i => (bx i).toRat).sum : ℝ)
      = ∑ j ∈ Finset.range d.D, sR d init t j := by
    rw [hTq_range, hsplitT]
    push_cast
    exact Finset.sum_congr rfl (fun j hj => by
      rw [Finset.mem_range] at hj
      exact hval j hj)
  have hS_cast : (((List.range d.BD).map fun i => |(bx i).toRat|).sum : ℝ)
      = ∑ j ∈ Finset.range d.D, |sR d init t j| := by
    rw [hSq_range, hsplitS]
    push_cast
    refine Finset.sum_congr rfl (fun j hj => ?_)
    rw [Finset.mem_range] at hj
    rw [show ((bx j).toRat : ℝ) = sR d init t j from hval j hj]

  -- the magnitude bounds
  have htermQ : ∀ j, j < d.D → |(bx j).toRat| ≤ (2:ℚ)^57 := by
    intro j hj
    have h1 : ((bx j).toRat : ℝ) = sR d init t j := hval j hj
    have h2 : |((bx j).toRat : ℝ)| ≤ (2:ℝ)^57 := by rw [h1]; exact hbdd j hj
    have h3 : ((|(bx j).toRat| : ℚ) : ℝ) ≤ ((2:ℚ)^57 : ℝ) := by
      rw [Rat.cast_abs]; push_cast; exact h2
    exact_mod_cast h3
  have hSq0 : 0 ≤ ((List.range d.BD).map fun i => |(bx i).toRat|).sum :=
    List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨i, _, rfl⟩; exact abs_nonneg _)
  have hSq_le : ((List.range d.BD).map fun i => |(bx i).toRat|).sum ≤ (2:ℚ)^66 := by
    rw [hSq_range]
    calc ∑ j ∈ Finset.range d.BD, |(bx j).toRat|
        ≤ ∑ _j ∈ Finset.range d.BD, (2:ℚ)^57 := by
          apply Finset.sum_le_sum
          intro j hj
          rw [Finset.mem_range] at hj
          by_cases hjd : j < d.D
          · exact htermQ j hjd
          · push Not at hjd
            rw [hzeroQ j hj hjd, abs_zero]; norm_num
      _ = (d.BD : ℚ) * (2:ℚ)^57 := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      _ ≤ 512 * (2:ℚ)^57 := by
          have hb : (d.BD : ℚ) ≤ 512 := by exact_mod_cast hBD_le
          nlinarith [hb]
      _ = (2:ℚ)^66 := by norm_num
  -- the room for `sum_err_depth`
  have hpow2 : (1 + Format.binary32.u) ^ (Nat.clog 2 d.BD) ≤ 2 := by
    have hu0 : (0:ℚ) ≤ Format.binary32.u := by rw [Format.binary32_u]; norm_num
    have hu18 : 18 * Format.binary32.u ≤ 1 := by rw [Format.binary32_u]; norm_num
    have hp := mean_tol_pow_bound hu0 hu18 (Nat.clog 2 d.BD) hdepth_le
    have hLu : ((Nat.clog 2 d.BD : ℕ) : ℚ) * 2 * Format.binary32.u ≤ 1 := by
      have hL9 : ((Nat.clog 2 d.BD : ℕ) : ℚ) ≤ 9 := by exact_mod_cast hdepth_le
      rw [Format.binary32_u] at *
      nlinarith
    linarith
  have hηval : Format.binary32.eta = (2:ℚ)^(-150:ℤ) := Format.binary32_eta
  have huval : Format.binary32.u = (2:ℚ)^(-24:ℤ) := Format.binary32_u
  have herr_le : Tree.errBound Format.binary32.u Format.binary32.eta (Nat.clog 2 d.BD) (d.BD - 1)
      ((List.range d.BD).map fun i => |(bx i).toRat|).sum
      ≤ ((List.range d.BD).map fun i => |(bx i).toRat|).sum + (2:ℚ)^(-140:ℤ) := by
    set Sq : ℚ := ((List.range d.BD).map fun i => |(bx i).toRat|).sum with hSqdef
    set p : ℚ := (1 + Format.binary32.u) ^ (Nat.clog 2 d.BD) with hpdef
    have hp1 : 1 ≤ p := by
      rw [hpdef]
      exact one_le_pow₀ (by rw [huval]; norm_num)
    have hp2 : p ≤ 2 := hpow2
    have hbq : ((d.BD - 1 : ℕ) : ℚ) ≤ 511 := by
      have : d.BD - 1 ≤ 511 := by omega
      exact_mod_cast this
    have hSq0' : 0 ≤ Sq := hSq0
    have hstep1 : (p - 1) * Sq ≤ Sq := mul_le_of_le_one_left hSq0' (by linarith)
    have hstep2 : Format.binary32.eta * ((d.BD - 1 : ℕ) : ℚ) * p ≤ (2:ℚ)^(-140:ℤ) := by
      have hb0 : (0:ℚ) ≤ ((d.BD - 1 : ℕ) : ℚ) := Nat.cast_nonneg _
      have hp0 : 0 ≤ p := by linarith
      have hbp : ((d.BD - 1 : ℕ) : ℚ) * p ≤ 511 * 2 := by nlinarith
      calc Format.binary32.eta * ((d.BD - 1 : ℕ) : ℚ) * p
          = (2:ℚ)^(-150:ℤ) * (((d.BD - 1 : ℕ) : ℚ) * p) := by rw [hηval]; ring
        _ ≤ (2:ℚ)^(-150:ℤ) * (511 * 2) := by
            apply mul_le_mul_of_nonneg_left hbp; norm_num
        _ ≤ (2:ℚ)^(-140:ℤ) := by norm_num
    have heq : Tree.errBound Format.binary32.u Format.binary32.eta (Nat.clog 2 d.BD) (d.BD - 1) Sq
        = (p - 1) * Sq + Format.binary32.eta * ((d.BD - 1 : ℕ) : ℚ) * p := by
      unfold Tree.errBound
      rw [← hpdef]
    rw [heq]
    linarith
  have hM : ((List.range d.BD).map fun i => |(bx i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (Nat.clog 2 d.BD) (d.BD - 1)
        ((List.range d.BD).map fun i => |(bx i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    have hbound : ((List.range d.BD).map fun i => |(bx i).toRat|).sum +
        Tree.errBound Format.binary32.u Format.binary32.eta (Nat.clog 2 d.BD) (d.BD - 1)
          ((List.range d.BD).map fun i => |(bx i).toRat|).sum
        ≤ (2:ℚ)^66 + ((2:ℚ)^66 + (2:ℚ)^(-140:ℤ)) := by linarith [hSq_le, herr_le]
    have hnum : (2:ℚ)^66 + ((2:ℚ)^66 + (2:ℚ)^(-140:ℤ)) ≤ (2^24 - 1) * 2^104 := by norm_num
    linarith

  -- the reduction's error bound
  obtain ⟨tr, hcov, hdepth, heval⟩ := hcover
  have hsum := TVA.sum_err_depth (f := Format.binary32)
      Format.binary32_regular bx hcov hdepth heval hfin hM
  -- the divisor and the quotient room
  have hdvQ : dv.toRat = (d.D : ℚ) := by
    have h : ((dv.toRat : ℚ) : ℝ) = ((d.D : ℚ) : ℝ) := hdv
    exact_mod_cast h
  have hdv0 : dv.toRat ≠ 0 := by
    rw [hdvQ, Nat.cast_ne_zero]
    exact ne_of_gt hDpos
  have hTq_abs : |((List.range d.BD).map fun i => (bx i).toRat).sum| ≤ (2:ℚ)^66 := by
    have heq : ((List.range d.BD).map fun i => (bx i).toRat).sum
        = ∑ j ∈ Finset.range d.D, (bx j).toRat := by rw [hTq_range, hsplitT]
    rw [heq]
    have h1 : |∑ j ∈ Finset.range d.D, (bx j).toRat| ≤ ∑ j ∈ Finset.range d.D, |(bx j).toRat| :=
      Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ j ∈ Finset.range d.D, |(bx j).toRat| ≤ ∑ _j ∈ Finset.range d.D, (2:ℚ)^57 :=
      Finset.sum_le_sum (fun j hj => by rw [Finset.mem_range] at hj; exact htermQ j hj)
    have h3 : ∑ _j ∈ Finset.range d.D, (2:ℚ)^57 = (d.D : ℚ) * (2:ℚ)^57 := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    have h4 : (d.D : ℚ) * (2:ℚ)^57 ≤ (2:ℚ)^66 := by
      have hb : (d.D : ℚ) ≤ 512 := by exact_mod_cast hd.d_le
      calc (d.D:ℚ) * (2:ℚ)^57 ≤ 512 * (2:ℚ)^57 := by
              apply mul_le_mul_of_nonneg_right hb; positivity
        _ = (2:ℚ)^66 := by norm_num
    linarith
  have hbsum_abs : |bsum.toRat| ≤ (2:ℚ)^68 := by
    have step : bsum.toRat
        = (bsum.toRat - ((List.range d.BD).map fun i => (bx i).toRat).sum)
          + ((List.range d.BD).map fun i => (bx i).toRat).sum := by ring
    rw [step]
    have h3 : Tree.errBound Format.binary32.u Format.binary32.eta
        (Nat.clog 2 d.BD) (d.BD - 1)
        ((List.range d.BD).map fun i => |(bx i).toRat|).sum
        ≤ (2:ℚ)^66 + (2:ℚ)^(-140:ℤ) := by linarith [herr_le, hSq_le]
    have h2 : |bsum.toRat - ((List.range d.BD).map fun i => (bx i).toRat).sum|
        ≤ (2:ℚ)^66 + (2:ℚ)^(-140:ℤ) := le_trans hsum.2 h3
    calc |(bsum.toRat - ((List.range d.BD).map fun i => (bx i).toRat).sum)
          + ((List.range d.BD).map fun i => (bx i).toRat).sum|
        ≤ |bsum.toRat - ((List.range d.BD).map fun i => (bx i).toRat).sum|
          + |((List.range d.BD).map fun i => (bx i).toRat).sum| := abs_add_le _ _
      _ ≤ ((2:ℚ)^66 + (2:ℚ)^(-140:ℤ)) + (2:ℚ)^66 := add_le_add h2 hTq_abs
      _ ≤ (2:ℚ)^68 := by norm_num
  have hqroom : |bsum.toRat / dv.toRat| ≤ (2:ℚ)^126 := by
    have hD0 : (0:ℚ) < (d.D : ℚ) := by exact_mod_cast hDpos
    rw [abs_div, hdvQ, abs_of_pos hD0]
    have hD1 : (1:ℚ) ≤ (d.D:ℚ) := by exact_mod_cast hd.d_pos
    have h1 : |bsum.toRat| / (d.D:ℚ) ≤ (2:ℚ)^68 := by
      rw [div_le_iff₀ hD0]
      have hmul : (2:ℚ)^68 * 1 ≤ (2:ℚ)^68 * (d.D:ℚ) := by
        apply mul_le_mul_of_nonneg_left hD1; positivity
      linarith
    have h2 : (2:ℚ)^68 ≤ (2:ℚ)^126 := by norm_num
    linarith
  have hdivacc : Device.DivAcc P Format.binary32 (2^126) (2^(-22:ℤ)) (2^(-148:ℤ)) := hdivf
  obtain ⟨hbmfin, hbmerr⟩ := hdivacc bsum dv bmean hdiv hsum.1 hdvfin hdv0 hqroom

  -- real-valued facts about the pieces
  have huR : ((Format.binary32.u : ℚ) : ℝ) = (2:ℝ)^(-24:ℤ) := by
    rw [huval]; norm_num
  have hηR : ((Format.binary32.eta : ℚ) : ℝ) = (2:ℝ)^(-150:ℤ) := by
    rw [hηval]; norm_num
  have hD0R : (0:ℝ) < (d.D : ℝ) := by exact_mod_cast hDpos
  have hSqR : (((List.range d.BD).map fun i => |(bx i).toRat|).sum : ℝ)
      = (d.D : ℝ) * absMean d init t := by
    rw [hS_cast]
    unfold absMean
    have hDne : (d.D : ℝ) ≠ 0 := ne_of_gt hD0R
    field_simp
  have hA0 : 0 ≤ absMean d init t := by
    unfold absMean
    apply div_nonneg
    · exact Finset.sum_nonneg (fun j hj => abs_nonneg _)
    · exact le_of_lt hD0R
  -- the reduction error, divided by `D`
  have herrD : ((Tree.errBound Format.binary32.u Format.binary32.eta
      (Nat.clog 2 d.BD) (d.BD - 1)
      ((List.range d.BD).map fun i => |(bx i).toRat|).sum : ℚ) : ℝ) / (d.D : ℝ)
      ≤ (Nat.clog 2 d.BD : ℝ) * (2:ℝ)^(-23:ℤ) * absMean d init t + (2:ℝ)^(-148:ℤ) := by
    set p : ℝ := (1 + (2:ℝ)^(-24:ℤ)) ^ (Nat.clog 2 d.BD) with hpdef
    have hp1 : 1 ≤ p := by
      rw [hpdef]
      exact one_le_pow₀ (by norm_num)
    have hp0 : 0 ≤ p := le_trans zero_le_one hp1
    have hp2 : p ≤ 2 := by
      rw [hpdef]
      have h' : (((1 + Format.binary32.u) ^ (Nat.clog 2 d.BD) : ℚ) : ℝ)
          ≤ ((2:ℚ) : ℝ) := by exact_mod_cast hpow2
      push_cast at h'
      rw [huR] at h'
      simpa using h'
    have hE0 : p - 1 ≤ (Nat.clog 2 d.BD : ℝ) * 2 * (2:ℝ)^(-24:ℤ) := by
      have h := mean_tol_pow_bound (K := ℚ)
        (show (0:ℚ) ≤ Format.binary32.u by rw [huval]; norm_num)
        (show 18 * Format.binary32.u ≤ 1 by rw [huval]; norm_num)
        (Nat.clog 2 d.BD) hdepth_le
      have h' : (((1 + Format.binary32.u) ^ (Nat.clog 2 d.BD) - 1 : ℚ) : ℝ)
          ≤ (((Nat.clog 2 d.BD : ℚ) * 2 * Format.binary32.u : ℚ) : ℝ) := by
        exact_mod_cast h
      push_cast at h'
      rw [huR] at h'
      rw [hpdef]
      simpa using h'
    have hsumcast : ∀ l : List ℚ, (l.map Rat.cast).sum = ((l.sum : ℚ) : ℝ) := by
      intro l
      induction l with
      | nil => simp
      | cons a as ih =>
          rw [List.map_cons, List.sum_cons, List.sum_cons, ih]
          push_cast
          ring
    have hcast : ((Tree.errBound Format.binary32.u Format.binary32.eta
        (Nat.clog 2 d.BD) (d.BD - 1)
        ((List.range d.BD).map fun i => |(bx i).toRat|).sum : ℚ) : ℝ)
        = (p - 1) * ((d.D:ℝ) * absMean d init t)
          + (2:ℝ)^(-150:ℤ) * ((d.BD - 1 : ℕ) : ℝ) * p := by
      unfold Tree.errBound
      push_cast
      rw [hsumcast (List.map (fun i => |(bx i).toRat|) (List.range d.BD)), hSqR, huR, hηR, ← hpdef]
    rw [hcast]
    have hbd1 : ((d.BD - 1 : ℕ) : ℝ) ≤ 2 * (d.D:ℝ) := by
      have hnat : d.BD - 1 ≤ 2 * d.D := by have := hd.bd_lt; omega
      exact_mod_cast hnat
    have hstep1 : (p - 1) * ((d.D:ℝ) * absMean d init t) / (d.D:ℝ)
        = (p - 1) * absMean d init t := by
      have hDne : (d.D:ℝ) ≠ 0 := ne_of_gt hD0R
      field_simp
    have hstep2 : (2:ℝ)^(-150:ℤ) * ((d.BD - 1 : ℕ) : ℝ) * p / (d.D:ℝ)
        ≤ (2:ℝ)^(-148:ℤ) := by
      have hmul1 : ((d.BD - 1 : ℕ) : ℝ) * p ≤ (2 * (d.D:ℝ)) * p :=
        mul_le_mul_of_nonneg_right hbd1 hp0
      have hmul2 : (2 * (d.D:ℝ)) * p ≤ (2 * (d.D:ℝ)) * 2 :=
        mul_le_mul_of_nonneg_left hp2 (by linarith)
      have hbp : ((d.BD - 1 : ℕ) : ℝ) * p ≤ 4 * (d.D:ℝ) := by linarith
      have hc : (0:ℝ) ≤ (2:ℝ)^(-150:ℤ) := by norm_num
      calc (2:ℝ)^(-150:ℤ) * ((d.BD - 1 : ℕ) : ℝ) * p / (d.D:ℝ)
          = (2:ℝ)^(-150:ℤ) * (((d.BD - 1 : ℕ) : ℝ) * p / (d.D:ℝ)) := by ring
        _ ≤ (2:ℝ)^(-150:ℤ) * 4 := by
            apply mul_le_mul_of_nonneg_left _ hc
            rw [div_le_iff₀ hD0R]
            linarith [hbp]
        _ = (2:ℝ)^(-148:ℤ) := by norm_num
    have hcoef : p - 1 ≤ (Nat.clog 2 d.BD : ℝ) * (2:ℝ)^(-23:ℤ) := by
      have h23 : (2:ℝ)^(-23:ℤ) = 2 * (2:ℝ)^(-24:ℤ) := by norm_num
      rw [h23]
      nlinarith [hE0]
    have hsplit : ((p - 1) * ((d.D:ℝ) * absMean d init t)
          + (2:ℝ)^(-150:ℤ) * ((d.BD - 1 : ℕ) : ℝ) * p) / (d.D:ℝ)
        = (p - 1) * absMean d init t
          + (2:ℝ)^(-150:ℤ) * ((d.BD - 1 : ℕ) : ℝ) * p / (d.D:ℝ) := by
      rw [add_div, hstep1]
    rw [hsplit]
    exact add_le_add (mul_le_mul_of_nonneg_right hcoef hA0) hstep2
  -- the quotient's error against the exact mean
  have hQm : |bsum.toReal / dv.toReal - meanRef d init t| ≤
      (Nat.clog 2 d.BD : ℝ) * (2:ℝ)^(-23:ℤ) * absMean d init t + (2:ℝ)^(-148:ℤ) := by
    have hdiff : bsum.toReal / dv.toReal - meanRef d init t
        = (bsum.toReal - (((List.range d.BD).map fun i => (bx i).toRat).sum : ℝ)) / (d.D:ℝ) := by
      have hmr : meanRef d init t = (∑ j ∈ Finset.range d.D, sR d init t j) / (d.D:ℝ) := rfl
      rw [hmr, hdv, ← hT_cast]
      have hDne : (d.D:ℝ) ≠ 0 := ne_of_gt hD0R
      field_simp
    rw [hdiff, abs_div, abs_of_pos hD0R]
    have hB : |bsum.toReal - (((List.range d.BD).map fun i => (bx i).toRat).sum : ℝ)|
        ≤ ((Tree.errBound Format.binary32.u Format.binary32.eta
            (Nat.clog 2 d.BD) (d.BD - 1)
            ((List.range d.BD).map fun i => |(bx i).toRat|).sum : ℚ) : ℝ) := by
      have h' : ((|bsum.toRat - ((List.range d.BD).map fun i => (bx i).toRat).sum| : ℚ) : ℝ)
          ≤ ((Tree.errBound Format.binary32.u Format.binary32.eta
              (Nat.clog 2 d.BD) (d.BD - 1)
              ((List.range d.BD).map fun i => |(bx i).toRat|).sum : ℚ) : ℝ) := by
        exact_mod_cast hsum.2
      push_cast at h'
      simpa [FloatV.toReal] using h'
    exact le_trans (div_le_div_of_nonneg_right hB (le_of_lt hD0R)) herrD
  -- the quotient's magnitude
  have hQ_abs : |bsum.toReal / dv.toReal| ≤
      absMean d init t
        + ((Nat.clog 2 d.BD : ℝ) * (2:ℝ)^(-23:ℤ) * absMean d init t + (2:ℝ)^(-148:ℤ)) := by
    have h1 : |bsum.toReal / dv.toReal| ≤
        |bsum.toReal / dv.toReal - meanRef d init t| + |meanRef d init t| := by
      have h := abs_add_le (bsum.toReal / dv.toReal - meanRef d init t) (meanRef d init t)
      rwa [sub_add_cancel] at h
    have h2 : |meanRef d init t| ≤ absMean d init t := abs_meanRef_le_absMean hDpos
    linarith [h1, h2, hQm]
  -- the division error, cast to the reals
  have hbmerrR : |bmean.toReal - bsum.toReal / dv.toReal| ≤
      (2:ℝ)^(-22:ℤ) * |bsum.toReal / dv.toReal| + (2:ℝ)^(-148:ℤ) := by
    have h' : ((|bmean.toRat - bsum.toRat / dv.toRat| : ℚ) : ℝ)
        ≤ ((2^(-22:ℤ) * |bsum.toRat / dv.toRat| + 2^(-148:ℤ) : ℚ) : ℝ) := by
      exact_mod_cast hbmerr
    push_cast at h'
    simpa [FloatV.toReal] using h'
  -- the triangle inequality and the final arithmetic
  apply And.intro hbmfin
  have htri : |bmean.toReal - meanRef d init t| ≤
      |bmean.toReal - bsum.toReal / dv.toReal|
        + |bsum.toReal / dv.toReal - meanRef d init t| := by
    have h := abs_add_le (bmean.toReal - bsum.toReal / dv.toReal)
      (bsum.toReal / dv.toReal - meanRef d init t)
    have he : (bmean.toReal - bsum.toReal / dv.toReal)
        + (bsum.toReal / dv.toReal - meanRef d init t) = bmean.toReal - meanRef d init t := by ring
    rwa [he] at h
  have hmain : |bmean.toReal - meanRef d init t| ≤
      ((Nat.clog 2 d.BD : ℝ) + 4) * (2:ℝ)^(-23:ℤ) * absMean d init t + (2:ℝ)^(-140:ℤ) := by
    have hk : (0:ℝ) < (2:ℝ)^(-23:ℤ) := by norm_num
    have hL9 : (Nat.clog 2 d.BD : ℝ) ≤ 9 := by exact_mod_cast hdepth_le
    have hLk : (Nat.clog 2 d.BD : ℝ) * (2:ℝ)^(-23:ℤ) ≤ 1 := by
      have h1 : (Nat.clog 2 d.BD : ℝ) * (2:ℝ)^(-23:ℤ) ≤ 9 * (2:ℝ)^(-23:ℤ) :=
        mul_le_mul_of_nonneg_right hL9 (le_of_lt hk)
      have h2 : (9:ℝ) * (2:ℝ)^(-23:ℤ) ≤ 1 := by norm_num
      linarith
    exact mean_tol_final (bsum.toReal / dv.toReal) (meanRef d init t) bmean.toReal
      (absMean d init t) ((Nat.clog 2 d.BD : ℝ)) hA0 hLk hQm hQ_abs hbmerrR htri
  simpa [meanTol, depth] using hmain

end

end Claims.LayerNormFwd
