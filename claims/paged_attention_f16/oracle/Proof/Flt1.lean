import Proof.Flt2

/-!
# Float facts, part 1: scores, the running maximum, the exponentials

Each lemma turns the results a `tl.*` operation may give (the hypotheses are exactly what the
rules `safe_*` hand the rest of a walk) into a statement about real numbers.
-/
namespace Claims.PagedAttention

namespace Num

open TVA Finset

/-- **A converted `float16` value is exact**: the `float32` result of converting a finite
`float16` has the same value. -/
theorem cvt16_exact {x : F16} {y : F32} (hx : x.isFinite = true) (rnd : RoundMode)
    (h : FloatV.NaNEq (FloatV.cvtWith rnd x) y) : y.isFinite = true ∧ y.toReal = x.toReal := by
  obtain ⟨hf, hv⟩ := FloatV.toRat_cvt_of_wider (g := Format.binary32) Format.binary16_regular
    Format.binary32_regular Format.wider_binary16_binary32 rnd hx
  have hy : y = FloatV.cvtWith rnd x := h.eq_of_isFinite hf
  subst hy
  exact ⟨hf, by unfold FloatV.toReal; rw [hv]⟩


/-- `float32` `+0` converts to `float16` `+0`. -/
theorem cvt0 : (FloatV.cvt (FloatV.ofPat 0 : F32) : F16) = FloatV.ofPat 0 := by decide +kernel

/-- The `float32` literal `0.0` is the pattern `0`. -/
theorem lit0 : FloatV.litF32 (FloatV.ofPat 0) = (FloatV.ofPat 0 : F32) := by decide +kernel

/-- **The `other` of the masked loads is a zero**: converting the `float32` constant `+0` to
`float16` gives a finite `float16` of value `0` (any platform result of `ConvRel`). -/
theorem conv_zero_f16 (P : Platform) {r : F16}
    (h : ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0)) r) :
    r.isFinite = true ∧ r.toReal = 0 := by
  rw [lit0] at h
  have hne : Format.binary32 ≠ Format.binary16 := by decide
  simp only [ConvRel, ofFloatRel, FloatRel, hne, ite_false] at h
  rcases h with ⟨-, -, h⟩ | ⟨-, hF8, -⟩ | ⟨hF8, -⟩
  · rw [cvt0] at h
    have hz : (FloatV.ofPat 0 : F16).isFinite = true := by decide +kernel
    have := h.eq_of_isFinite hz
    subst this
    exact ⟨hz, by unfold FloatV.toReal; norm_num⟩
  · exact absurd hF8 (by decide)
  · exact absurd hF8 (by decide)

/-- **One row of scores.** `q` (length `Dm ≤ 256`) and the row `i` of `kk` hold finite values of
magnitude at most `65504` (`qr`, `kr` their values); `prod` holds their products
(`tl.mul`), `sums.get1 i` is the sum of the row along a tree (`tl.sum1c`, fusable additions of
depth at most `tl.sumBound`), `sc` is that sum times the `float32` scale `σh` (`tl.mul`), where
`σh` is the nearest `float32` to the double `σr ∈ [2^-64, 2^64]` (`|σh − σr| ≤ σr·u + 2^-150`).
If `σr·Σ|q·k| ≤ X ≤ 1024`, then `sc` is finite and within `28·u·(σr·Σ|q·k|) + 2^-60` of the
exact scaled score `σr·Σ q·k`. -/
theorem scores_row {E : Env} (hP : Device E.P) {Pg Dm : ℕ} (hD : Dm ≤ 256) (hDpos : 0 < Dm)
    (q : Tensor [Dm] F32) (kk prod : Tensor [Pg, Dm] F32) (sums : Tensor [Pg] F32) (i : Fin Pg)
    (σh sc : F32) {qr kr : ℕ → ℝ} {σr X : ℝ}
    (hq : ∀ (e : ℕ) (he : e < Dm), (q.get1 ⟨e, he⟩).isFinite = true ∧
      (q.get1 ⟨e, he⟩).toReal = qr e ∧ |qr e| ≤ 65504)
    (hk : ∀ (e : ℕ) (he : e < Dm), (kk.get2 i ⟨e, he⟩).isFinite = true ∧
      (kk.get2 i ⟨e, he⟩).toReal = kr e ∧ |kr e| ≤ 65504)
    (hprod : ∀ e : Fin Dm, FloatV.NaNEq (FloatV.mul (kk.get2 i e) (q.get1 e)) (prod.get2 i e))
    (hsum : ∃ t : Tree, t.Covers Dm ∧ t.depth ≤ tl.sumBound E.P Dm ∧
      t.Eval (AddRel E.fusion) (fun j => if h : j < Dm then prod.get2 i ⟨j, h⟩ else default)
        (sums.get1 i))
    (hsc : FloatV.NaNEq (FloatV.mul (sums.get1 i) σh) sc)
    (hσ : σh.isFinite = true ∧ |σh.toReal - σr| ≤ σr * u + 1 / 2 ^ 150)
    (hσr : 1 / 2 ^ 64 ≤ σr ∧ σr ≤ 2 ^ 64)
    (hX : σr * ∑ e ∈ range Dm, |qr e * kr e| ≤ X) (hX1 : X ≤ 1024) :
    sc.isFinite = true ∧
      |sc.toReal - σr * ∑ e ∈ range Dm, qr e * kr e| ≤
        28 * u * (σr * ∑ e ∈ range Dm, |qr e * kr e|) + 1 / 2 ^ 60 := by
  obtain ⟨t, hcov, hdep, heval⟩ := hsum
  have hd8 := depth_le_eight hP hD hdep
  have hu := u_pos
  set x : ℕ → F32 := fun j => if h : j < Dm then prod.get2 i ⟨j, h⟩ else default with hx
  set w : ℕ → ℝ := fun e => qr e * kr e with hw
  have hwb : ∀ e, e < Dm → |w e| ≤ 2 ^ 32 := by
    intro e he
    simp only [hw]; rw [abs_mul]
    calc |qr e| * |kr e| ≤ 65504 * 65504 :=
          mul_le_mul (hq e he).2.2 (hk e he).2.2 (abs_nonneg _) (by norm_num)
      _ ≤ 2 ^ 32 := by norm_num
  have hxe : ∀ e, e < Dm → (x e).isFinite = true ∧ |(x e).toReal - w e| ≤ u * |w e| + 1 / 2 ^ 150 := by
    intro e he
    obtain ⟨hqf, hqv, -⟩ := hq e he
    obtain ⟨hkf, hkv, -⟩ := hk e he
    have hmax : |(kk.get2 i ⟨e, he⟩).toReal * (q.get1 ⟨e, he⟩).toReal| ≤ (Format.binary32.maxVal : ℝ) := by
      rw [hkv, hqv, max32_real, mul_comm]
      exact (hwb e he).trans (by norm_num)
    obtain ⟨hmf, hme⟩ := FloatV.mul_err_real Format.binary32_regular hkf hqf hmax
    have hxm : prod.get2 i ⟨e, he⟩ = FloatV.mul (kk.get2 i ⟨e, he⟩) (q.get1 ⟨e, he⟩) :=
      (hprod ⟨e, he⟩).eq_of_isFinite hmf
    have hxi : x e = prod.get2 i ⟨e, he⟩ := by simp only [hx, dite_eq_left he]
    rw [hxi, hxm]
    refine ⟨hmf, ?_⟩
    rw [u32_real, eta32_real, hkv, hqv] at hme
    simp only [hw]
    rw [mul_comm (qr e) (kr e), mul_comm u]
    exact hme
  have hxfin : ∀ j, j < Dm → (x j).isFinite = true := fun j hj => (hxe j hj).1
  have hs : Format.binary32.specials ≠ .finite := by decide
  have hu32 : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hDR : (Dm : ℝ) ≤ 256 := by exact_mod_cast hD
  have hD1 : ((Dm - 1 : ℕ) : ℝ) ≤ 255 := by
    have : Dm - 1 ≤ 255 := by omega
    exact_mod_cast this
  set W := ∑ e ∈ range Dm, |w e| with hWdef
  have hW0 : 0 ≤ W := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hW40 : W ≤ 2 ^ 40 := by
    calc W ≤ ∑ e ∈ range Dm, (2 : ℝ) ^ 32 := Finset.sum_le_sum fun e he => hwb e (Finset.mem_range.mp he)
      _ = Dm * 2 ^ 32 := by simp
      _ ≤ 256 * 2 ^ 32 := mul_le_mul_of_nonneg_right hDR (by norm_num)
      _ = 2 ^ 40 := by norm_num
  have hX1' : ∑ e ∈ range Dm, |(x e).toReal| ≤ (1 + u) * W + Dm * (1 / 2 ^ 150) := by
    calc ∑ e ∈ range Dm, |(x e).toReal| ≤ ∑ e ∈ range Dm, ((1 + u) * |w e| + 1 / 2 ^ 150) := by
          refine Finset.sum_le_sum fun e he => ?_
          have h := (hxe e (Finset.mem_range.mp he)).2
          have := abs_sub_abs_le_abs_sub (x e).toReal (w e)
          linarith
      _ = (1 + u) * W + Dm * (1 / 2 ^ 150) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp [hWdef]
  have hX2 : |∑ e ∈ range Dm, (x e).toReal - ∑ e ∈ range Dm, w e| ≤ u * W + Dm * (1 / 2 ^ 150) := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ e ∈ range Dm, |(x e).toReal - w e| ≤ ∑ e ∈ range Dm, (u * |w e| + 1 / 2 ^ 150) :=
          Finset.sum_le_sum fun e he => (hxe e (Finset.mem_range.mp he)).2
      _ = u * W + Dm * (1 / 2 ^ 150) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp [hWdef]
  set Aq : ℚ := ∑ e ∈ range Dm, |(x e).toRat| with hAq
  have hAeq : (Aq : ℝ) = ∑ e ∈ range Dm, |(x e).toReal| := by
    rw [hAq]; push_cast; rfl
  have hu24 : u * 2 ^ 24 = 1 := by unfold u; norm_num
  have hDη : (Dm : ℝ) * (1 / 2 ^ 150) ≤ 1 / 2 ^ 142 := by
    have := mul_le_mul_of_nonneg_right hDR (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)
    norm_num at this ⊢; linarith
  have hAr : (Aq : ℝ) ≤ 2 ^ 42 := by
    rw [hAeq]
    have h2 : u * W ≤ u * 2 ^ 40 := mul_le_mul_of_nonneg_left hW40 hu.le
    have h3 : u * 2 ^ 40 = 2 ^ 16 := by unfold u; norm_num
    have h4 : (1 : ℝ) / 2 ^ 142 ≤ 1 := by norm_num
    have h5 : (2 : ℝ) ^ 40 + 2 ^ 16 + 1 ≤ 2 ^ 42 := by norm_num
    nlinarith
  have hAq0 : (0 : ℚ) ≤ Aq := Finset.sum_nonneg fun i _ => abs_nonneg _
  have hEB : ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Dm - 1) Aq : ℚ) : ℝ) ≤
      241 / 10 * u * (Aq : ℝ) + 1 / 2 ^ 138 := by
    unfold Tree.errBound
    push_cast
    rw [u32_real, eta32_real]
    have hp8 := pow8_3_le
    have h1 : ((1 + 3 * u) ^ 8 - 1) * (Aq : ℝ) ≤ 241 / 10 * u * (Aq : ℝ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by exact_mod_cast hAq0)
    have h8 : (1 + 3 * u) ^ 8 ≤ 2 := by unfold u; norm_num
    have h2 : 4 * (1 / 2 ^ 150 : ℝ) * ((Dm - 1 : ℕ) : ℝ) * (1 + 3 * u) ^ 8 ≤ 1 / 2 ^ 138 := by
      have : 4 * (1 / 2 ^ 150 : ℝ) * ((Dm - 1 : ℕ) : ℝ) ≤ 4 * (1 / 2 ^ 150) * 255 :=
        mul_le_mul_of_nonneg_left hD1 (by norm_num)
      calc 4 * (1 / 2 ^ 150 : ℝ) * ((Dm - 1 : ℕ) : ℝ) * (1 + 3 * u) ^ 8 ≤ (4 * (1 / 2 ^ 150) * 255) * 2 :=
            mul_le_mul this h8 (by positivity) (by positivity)
        _ ≤ 1 / 2 ^ 138 := by norm_num
    linarith
  have hM : ((List.range Dm).map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Dm - 1)
        ((List.range Dm).map fun i => |(x i).toRat|).sum ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [list_range_sum, ← hAq]
    rw [← Rat.cast_le (K := ℝ)]
    push_cast
    have := hEB
    rw [max32_real, eta32_real]
    have hu8 : (241 / 10 * u * 2 ^ 42 : ℝ) ≤ 2 ^ 23 := by unfold u; norm_num
    have hA2 : 241 / 10 * u * (Aq : ℝ) ≤ 241 / 10 * u * 2 ^ 42 :=
      mul_le_mul_of_nonneg_left hAr (by positivity)
    have h140 : (1 / 2 ^ 138 : ℝ) ≤ 1 := by norm_num
    have : (2 : ℝ) ^ 42 + 2 ^ 23 + 1 + 2 * (1 / 2 ^ 150) ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 := by norm_num
    linarith
  obtain ⟨hrfin, herr⟩ := sumc_err_depth Format.binary32_regular hs hu32 E.fusion x hcov hd8 heval hxfin hM
  rw [list_range_sum, list_range_sum] at herr
  have herr' : |(sums.get1 i).toReal - ∑ e ∈ range Dm, (x e).toReal| ≤
      ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Dm - 1) Aq : ℚ) : ℝ) := by
    have h1 : ((|(sums.get1 i).toRat - ∑ e ∈ range Dm, (x e).toRat| : ℚ) : ℝ) ≤
        ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Dm - 1) Aq : ℚ) : ℝ) := by
      exact_mod_cast herr
    push_cast at h1
    unfold FloatV.toReal
    exact h1
  have herrB := herr'.trans hEB
  -- the sum's error
  have hu1 : u ≤ 1 / 1000 := by unfold u; norm_num
  have hAr2 : 241 / 10 * u * (Aq : ℝ) ≤ 241 / 10 * u * ((1 + u) * W + Dm * (1 / 2 ^ 150)) := by
    have : (Aq : ℝ) ≤ (1 + u) * W + Dm * (1 / 2 ^ 150) := by rw [hAeq]; exact hX1'
    exact mul_le_mul_of_nonneg_left this (by positivity)
  have hsumsT : |(sums.get1 i).toReal - ∑ e ∈ range Dm, w e| ≤ 252 / 10 * u * W + 1 / 2 ^ 130 := by
    have h1 : |(sums.get1 i).toReal - ∑ e ∈ range Dm, w e| ≤
        |(sums.get1 i).toReal - ∑ e ∈ range Dm, (x e).toReal| +
          |∑ e ∈ range Dm, (x e).toReal - ∑ e ∈ range Dm, w e| := by
      have := abs_add_le ((sums.get1 i).toReal - ∑ e ∈ range Dm, (x e).toReal)
        (∑ e ∈ range Dm, (x e).toReal - ∑ e ∈ range Dm, w e)
      rwa [show (sums.get1 i).toReal - ∑ e ∈ range Dm, (x e).toReal +
        (∑ e ∈ range Dm, (x e).toReal - ∑ e ∈ range Dm, w e) =
          (sums.get1 i).toReal - ∑ e ∈ range Dm, w e by ring] at this
    have hc1 : 241 / 10 * u * ((1 + u) * W + Dm * (1 / 2 ^ 150)) ≤
        (241 / 10 * u + 1 / 10 * u) * W + 241 / 10 * u * (1 / 2 ^ 142) := by
      have e : 241 / 10 * u * ((1 + u) * W + Dm * (1 / 2 ^ 150)) =
          241 / 10 * u * W + 241 / 10 * u * u * W + 241 / 10 * u * (Dm * (1 / 2 ^ 150)) := by ring
      have h2 : 241 / 10 * u * u * W ≤ 1 / 10 * u * W := by
        have : 241 / 10 * u * u ≤ 1 / 10 * u := by nlinarith
        exact mul_le_mul_of_nonneg_right this hW0
      have h3 : 241 / 10 * u * (Dm * (1 / 2 ^ 150)) ≤ 241 / 10 * u * (1 / 2 ^ 142) :=
        mul_le_mul_of_nonneg_left hDη (by positivity)
      linarith
    have hsmall : 241 / 10 * u * (1 / 2 ^ 142) + 1 / 2 ^ 138 + (Dm : ℝ) * (1 / 2 ^ 150) ≤ 1 / 2 ^ 130 := by
      unfold u; norm_num; linarith
    nlinarith
  -- the scaled score
  set S := (sums.get1 i).toReal with hS
  set σt := σh.toReal with hσt
  set V := σr * W with hV
  have hσr0 : 0 < σr := lt_of_lt_of_le (by norm_num) hσr.1
  have hV0 : 0 ≤ V := mul_nonneg hσr0.le hW0
  have hVX : V ≤ 1024 := by
    have : σr * W ≤ X := hX
    linarith
  have hT0 : |∑ e ∈ range Dm, w e| ≤ W := Finset.abs_sum_le_sum_abs _ _
  have hSb : |S| ≤ (1 + 26 * u) * W + 1 / 2 ^ 130 := by
    have h1 := abs_sub_abs_le_abs_sub S (∑ e ∈ range Dm, w e)
    have h2 : 252 / 10 * u * W ≤ 26 * u * W := mul_le_mul_of_nonneg_right (by linarith) hW0
    linarith
  have hσb : |σt| ≤ σr * (1 + u) + 1 / 2 ^ 150 := by
    have h1 := abs_sub_abs_le_abs_sub σt σr
    rw [abs_of_pos hσr0] at h1
    have := hσ.2
    linarith
  have hSB0 : 0 ≤ (1 + 26 * u) * W + 1 / 2 ^ 130 := by positivity
  have hσB0 : 0 ≤ σr * (1 + u) + 1 / 2 ^ 150 := by positivity
  have hprod : |S * σt| ≤ ((1 + 26 * u) * W + 1 / 2 ^ 130) * (σr * (1 + u) + 1 / 2 ^ 150) := by
    rw [abs_mul]; exact mul_le_mul hSb hσb (abs_nonneg _) hSB0
  have hu1 : u ≤ 1 / 1000 := by unfold u; norm_num
  have hσ64 : σr ≤ 2 ^ 64 := hσr.2
  have hW40' : W ≤ 2 ^ 40 := hW40
  have hexp1 : ((1 + 26 * u) * W + 1 / 2 ^ 130) * (σr * (1 + u) + 1 / 2 ^ 150) =
      (1 + 26 * u) * (1 + u) * V + (1 + 26 * u) * W * (1 / 2 ^ 150) +
        1 / 2 ^ 130 * (σr * (1 + u)) + 1 / 2 ^ 130 * (1 / 2 ^ 150) := by
    rw [hV]; ring
  have hc1 : (1 + 26 * u) * W * (1 / 2 ^ 150) ≤ 1 / 2 ^ 108 := by
    have h1 : (1 + 26 * u) * W ≤ 2 * 2 ^ 40 :=
      mul_le_mul (by unfold u; norm_num) hW40' hW0 (by norm_num)
    calc (1 + 26 * u) * W * (1 / 2 ^ 150) ≤ (2 * 2 ^ 40) * (1 / 2 ^ 150) :=
          mul_le_mul_of_nonneg_right h1 (by norm_num)
      _ ≤ 1 / 2 ^ 108 := by norm_num
  have hc2 : 1 / 2 ^ 130 * (σr * (1 + u)) ≤ 1 / 2 ^ 65 := by
    have h1 : σr * (1 + u) ≤ 2 ^ 64 * 2 :=
      mul_le_mul hσ64 (by unfold u; norm_num) (by unfold u; norm_num) (by positivity)
    calc 1 / 2 ^ 130 * (σr * (1 + u)) ≤ 1 / 2 ^ 130 * (2 ^ 64 * 2) :=
          mul_le_mul_of_nonneg_left h1 (by norm_num)
      _ = 1 / 2 ^ 65 := by norm_num
  have hSσ : |S * σt| ≤ (1 + 26 * u) * (1 + u) * V + 1 / 2 ^ 64 := by
    have : (1 / 2 ^ 130 * (1 / 2 ^ 150) : ℝ) ≤ 1 / 2 ^ 70 := by norm_num
    have h3 : (1 : ℝ) / 2 ^ 108 + 1 / 2 ^ 65 + 1 / 2 ^ 70 ≤ 1 / 2 ^ 64 := by norm_num
    linarith
  -- the product and the rounding
  have hmax : |S * σt| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [max32_real]
    refine hSσ.trans ?_
    have h1 : (1 + 26 * u) * (1 + u) * V ≤ 2 * 1024 := by
      have : (1 + 26 * u) * (1 + u) ≤ 2 := by unfold u; norm_num
      calc (1 + 26 * u) * (1 + u) * V ≤ 2 * V := mul_le_mul_of_nonneg_right this hV0
        _ ≤ 2 * 1024 := by linarith
    have h2 : (1 : ℝ) / 2 ^ 64 ≤ 1 := by norm_num
    have h3 : (2 : ℝ) * 1024 + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hscf, hsce⟩ := FloatV.mul_err_real Format.binary32_regular
    hrfin hσ.1 hmax
  have hscE : sc = FloatV.mul (sums.get1 i) σh := hsc.eq_of_isFinite hscf
  rw [u32_real, eta32_real] at hsce
  refine ⟨by rw [hscE]; exact hscf, ?_⟩
  rw [hscE]
  change |(FloatV.mul (sums.get1 i) σh).toReal - σr * ∑ e ∈ range Dm, w e| ≤ 28 * u * V + 1 / 2 ^ 60
  set T0 := ∑ e ∈ range Dm, w e with hT0def
  have e1 : (FloatV.mul (sums.get1 i) σh).toReal - σr * T0 =
      ((FloatV.mul (sums.get1 i) σh).toReal - S * σt) + S * (σt - σr) + σr * (S - T0) := by ring
  have t1 := abs_add_le ((FloatV.mul (sums.get1 i) σh).toReal - S * σt + S * (σt - σr)) (σr * (S - T0))
  have t2 := abs_add_le ((FloatV.mul (sums.get1 i) σh).toReal - S * σt) (S * (σt - σr))
  rw [← e1] at t1
  have d1 : |(FloatV.mul (sums.get1 i) σh).toReal - S * σt| ≤
      ((1 + 26 * u) * (1 + u) * V + 1 / 2 ^ 64) * u + 1 / 2 ^ 150 := by
    refine hsce.trans ?_
    have := mul_le_mul_of_nonneg_right hSσ hu.le
    linarith
  have d2 : |S * (σt - σr)| ≤ ((1 + 26 * u) * W + 1 / 2 ^ 130) * (σr * u + 1 / 2 ^ 150) := by
    rw [abs_mul]
    exact mul_le_mul hSb hσ.2 (abs_nonneg _) hSB0
  have d3 : |σr * (S - T0)| ≤ σr * (252 / 10 * u * W + 1 / 2 ^ 130) := by
    rw [abs_mul, abs_of_pos hσr0]
    exact mul_le_mul_of_nonneg_left hsumsT hσr0.le
  have x2 : ((1 + 26 * u) * W + 1 / 2 ^ 130) * (σr * u + 1 / 2 ^ 150) =
      (1 + 26 * u) * u * V + (1 + 26 * u) * W * (1 / 2 ^ 150) + 1 / 2 ^ 130 * (σr * u) +
        1 / 2 ^ 130 * (1 / 2 ^ 150) := by rw [hV]; ring
  have x3 : σr * (252 / 10 * u * W + 1 / 2 ^ 130) = 252 / 10 * u * V + σr * (1 / 2 ^ 130) := by
    rw [hV]; ring
  have c3 : 1 / 2 ^ 130 * (σr * u) ≤ 1 / 2 ^ 90 := by
    have h1 : σr * u ≤ 2 ^ 64 * u := mul_le_mul_of_nonneg_right hσ64 hu.le
    calc 1 / 2 ^ 130 * (σr * u) ≤ 1 / 2 ^ 130 * (2 ^ 64 * u) := mul_le_mul_of_nonneg_left h1 (by norm_num)
      _ ≤ 1 / 2 ^ 90 := by unfold u; norm_num
  have c4 : σr * (1 / 2 ^ 130) ≤ 1 / 2 ^ 66 := by
    calc σr * (1 / 2 ^ 130) ≤ 2 ^ 64 * (1 / 2 ^ 130) := mul_le_mul_of_nonneg_right hσ64 (by norm_num)
      _ = 1 / 2 ^ 66 := by norm_num
  have c5 : (1 + 26 * u) * (1 + u) * V * u ≤ (1 + 26 * u) * (1 + u) * V * u := le_rfl
  have c6 : 1 / 2 ^ 64 * u ≤ 1 / 2 ^ 88 := by unfold u; norm_num
  have c7 : (1 / 2 ^ 130 * (1 / 2 ^ 150) : ℝ) ≤ 1 / 2 ^ 100 := by norm_num
  have c8 : (1 / 2 ^ 150 : ℝ) ≤ 1 / 2 ^ 100 := by norm_num
  have c9 : (1 / 2 ^ 108 + 1 / 2 ^ 100 + 1 / 2 ^ 100 + 1 / 2 ^ 90 + 1 / 2 ^ 66 + 1 / 2 ^ 88 + 1 / 2 ^ 100 : ℝ) ≤ 1 / 2 ^ 60 := by norm_num
  have hq1 : (1 + 26 * u) * (1 + u) * V * u ≤ (1 + 271 / 10 * u) * (u * V) := by
    have e : (1 + 26 * u) * (1 + u) * V * u = ((1 + 26 * u) * (1 + u)) * (u * V) := by ring
    rw [e]
    exact mul_le_mul_of_nonneg_right (by unfold u; norm_num) (mul_nonneg hu.le hV0)
  have hq2 : (1 + 26 * u) * u * V ≤ (1 + 271 / 10 * u) * (u * V) := by
    have e : (1 + 26 * u) * u * V = (1 + 26 * u) * (u * V) := by ring
    rw [e]
    exact mul_le_mul_of_nonneg_right (by unfold u; norm_num) (mul_nonneg hu.le hV0)
  have hq3 : (1 + 271 / 10 * u) * (u * V) ≤ (1001 / 1000) * (u * V) :=
    mul_le_mul_of_nonneg_right (by unfold u; norm_num) (mul_nonneg hu.le hV0)
  have hq4 : 0 ≤ u * V := mul_nonneg hu.le hV0
  have e3 : 28 * u * V = 28 * (u * V) := by ring
  rw [e3]
  have e4 : 252 / 10 * u * V = 252 / 10 * (u * V) := by ring
  linarith

/-- `float32` has infinities and NaN. -/
theorem b32_ieee : Format.binary32.specials = .ieee := rfl

/-- `−∞` is not finite. -/
theorem negInf_notFinite : (FloatV.negInf : F32).isFinite = false := by
  by_contra h
  have h' : (FloatV.negInf : F32).isFinite = true := by simpa using h
  have := (FloatV.isFinite_iff.mp h').2
  rw [FloatV.negInf_isInf b32_ieee] at this
  cases this

/-- `−∞` is not NaN. -/
theorem negInf_nan : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN b32_ieee

/-- **The running maximum.** `sp` holds the masked scores of one page: lanes `i < c` (`1 ≤ c ≤
Pg`) are finite, the others `−∞`. `mx` is their maximum along a tree (`tl.max`), `mn` the maximum
of `mx` and the old maximum `mi` (`−∞` or finite; `tl.maximum`). Then `mn` is finite, at least
`mi` and every valid lane, and equal to `mi` or to a valid lane. -/
theorem max_new {Pg c : ℕ} (hc : 1 ≤ c) (hcP : c ≤ Pg) (sp : Tensor [Pg] F32)
    (hfin : ∀ i : Fin Pg, i.val < c → (sp.get1 i).isFinite = true)
    (hneg : ∀ i : Fin Pg, c ≤ i.val → sp.get1 i = FloatV.negInf) (mi mx mn : F32)
    (hmi : mi = FloatV.negInf ∨ mi.isFinite = true)
    (hmx : ∃ t : Tree, t.Covers Pg ∧ t.Eval (FloatV.MaxRel true)
      (fun j => if h : j < Pg then sp.get1 ⟨j, h⟩ else default) mx)
    (hmn : FloatV.MaxRel true mi mx mn) :
    mn.isFinite = true ∧ (mi.isFinite = true → mi.toReal ≤ mn.toReal) ∧
      (∀ i : Fin Pg, i.val < c → (sp.get1 i).toReal ≤ mn.toReal) ∧
      ((∃ i : Fin Pg, i.val < c ∧ mn.toReal = (sp.get1 i).toReal) ∨
        (mi.isFinite = true ∧ mn = mi)) := by
  obtain ⟨t, hcov, heval⟩ := hmx
  have hPg : 0 < Pg := by omega
  set x : ℕ → F32 := fun j => if h : j < Pg then sp.get1 ⟨j, h⟩ else default with hx
  have hnan : ∀ i ∈ t.leaves, (x i).isNaN = false := by
    intro i hi
    have hi' : i < Pg := List.mem_range.mp (hcov.subset hi)
    simp only [hx, dite_eq_left hi']
    by_cases hic : i < c
    · exact (FloatV.isFinite_iff.mp (hfin ⟨i, hi'⟩ hic)).1
    · rw [hneg ⟨i, hi'⟩ (by simpa using hic)]; exact negInf_nan
  obtain ⟨⟨i0, hi0, hmxeq⟩, hle⟩ := Tree.eval_maxRel_toReal heval hnan
  have hi0' : i0 < Pg := List.mem_range.mp (hcov.subset hi0)
  have hleaf : ∀ i, i < Pg → i ∈ t.leaves := fun i hi => hcov.mem_iff.mpr (List.mem_range.mpr hi)
  -- mx is finite and equal to a valid lane
  have hx0 : (x 0).isFinite = true := by
    simp only [hx, dite_eq_left hPg]; exact hfin ⟨0, hPg⟩ (show 0 < c by omega)
  have hx0le := hle 0 (hleaf 0 hPg)
  have hi0c : i0 < c := by
    by_contra hic
    have hx0n : x i0 = FloatV.negInf := by
      simp only [hx, dite_eq_left hi0']; exact hneg ⟨i0, hi0'⟩ (by simpa using hic)
    rw [hmxeq, hx0n] at hx0le
    have hxn := (FloatV.isFinite_iff.mp hx0).1
    obtain ⟨hl, heq⟩ := FloatV.negInf_toRat_le Format.binary32_regular b32_ieee hxn
    have h1 : (x 0).toRat ≤ (FloatV.negInf : F32).toRat := by
      unfold FloatV.toReal at hx0le; exact_mod_cast hx0le
    have := heq (le_antisymm hl h1)
    rw [this] at hx0
    rw [negInf_notFinite] at hx0; cases hx0
  have hmxfin : mx.isFinite = true := by
    rw [hmxeq]; simp only [hx, dite_eq_left hi0']; exact hfin ⟨i0, hi0'⟩ hi0c
  have hmxn := (FloatV.isFinite_iff.mp hmxfin).1
  have hmin : mi.isNaN = false := by
    rcases hmi with h | h
    · rw [h]; exact negInf_nan
    · exact (FloatV.isFinite_iff.mp h).1
  have hmem := hmn.mem hmin hmxn
  have hmnfin : mn.isFinite = true := by
    rcases hmi with h | h
    · rw [h] at hmn
      have := FloatV.MaxRel.negInf_left Format.binary32_regular b32_ieee hmn hmxn
      rw [this]; exact hmxfin
    · rcases hmem with h' | h' <;> rw [h'] <;> assumption
  have hleR : ∀ y, y.isNaN = false → FloatV.le y mn = true → y.toReal ≤ mn.toReal := by
    intro y hy hh
    have hmnn := (FloatV.isFinite_iff.mp hmnfin).1
    have := (FloatV.le_iff hy hmnn).mp hh
    unfold FloatV.toReal; exact_mod_cast this
  obtain ⟨hl1, hl2⟩ := hmn.le hmin hmxn
  have hmx_le : mx.toReal ≤ mn.toReal := hleR mx hmxn hl2
  refine ⟨hmnfin, ?_, ?_, ?_⟩
  · intro hfi; exact hleR mi hmin hl1
  · intro i hi
    have := hle i (hleaf i i.2)
    simp only [hx, dite_eq_left i.2] at this
    exact this.trans hmx_le
  · rcases hmi with h | h
    · rw [h] at hmn
      have := FloatV.MaxRel.negInf_left Format.binary32_regular b32_ieee hmn hmxn
      left
      refine ⟨⟨i0, hi0'⟩, hi0c, ?_⟩
      rw [this, hmxeq]; simp only [hx, dite_eq_left hi0']
    · rcases hmem with h' | h'
      · right; exact ⟨h, h'⟩
      · left
        refine ⟨⟨i0, hi0'⟩, hi0c, ?_⟩
        rw [h', hmxeq]; simp only [hx, dite_eq_left hi0']

/-- **`tl.exp` of a finite `float32` of magnitude at most `4200` (and at most `1`)**: finite and
within `(2^-22 + 2^-23·|x|)·e^x + 2^-140` of `e^x` (the device fact, with `X := |x|`). -/
theorem exp_fact {E : Env} (hP : Device E.P) {xh ph : F32} (hx : xh.isFinite = true)
    (hup : xh.toReal ≤ 1) (hbig : |xh.toReal| ≤ 4200) (hexp : E.P.math .exp .binary32 xh ph) :
    ph.isFinite = true ∧
      |ph.toReal - Real.exp xh.toReal| ≤
        (4 * u + 2 * u * |xh.toReal|) * Real.exp xh.toReal + 1 / 2 ^ 140 := by
  obtain ⟨ref, href, hmain⟩ := hP.expF32 |xh.toReal| (abs_nonneg _)
  have hr : ref = Real.exp := by
    simp only [MathFn.ref, Option.some.injEq] at href; exact href.symm
  subst hr
  have hd : xh.toReal ∈ Set.Icc (-|xh.toReal|) |xh.toReal| := ⟨neg_abs_le _, le_abs_self _⟩
  have he3 : Real.exp xh.toReal ≤ 3 := by
    have := Real.exp_le_exp.mpr hup
    have h1 : Real.exp 1 < 3 := by
      have := Real.exp_one_lt_d9; linarith
    linarith
  have hu := u_pos
  have hrel : (2 : ℝ) ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |xh.toReal| ≤ 1 := by
    have : (2 : ℝ) ^ (-23 : ℤ) * |xh.toReal| ≤ 2 ^ (-23 : ℤ) * 4200 :=
      mul_le_mul_of_nonneg_left hbig (by positivity)
    have h2 : (2 : ℝ) ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 4200 ≤ 1 := by norm_num
    linarith
  have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |xh.toReal|)) * |Real.exp xh.toReal| +
      2 ^ (-140 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [abs_of_pos (Real.exp_pos _), max32_real]
    have h1 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |xh.toReal|)) * Real.exp xh.toReal ≤ 2 * 3 :=
      mul_le_mul (by linarith) he3 (Real.exp_pos _).le (by norm_num)
    have h2 : (2 : ℝ) * 3 + 2 ^ (-140 : ℤ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hf, he⟩ := hmain xh ph hexp hx hd (Set.mem_univ _) hroom
  refine ⟨hf, ?_⟩
  rw [abs_of_pos (Real.exp_pos _)] at he
  have e1 : (2 : ℝ) ^ (-22 : ℤ) = 4 * u := by unfold u; norm_num
  have e2 : (2 : ℝ) ^ (-23 : ℤ) = 2 * u := by unfold u; norm_num
  have e3 : (2 : ℝ) ^ (-140 : ℤ) = 1 / 2 ^ 140 := by norm_num
  rw [e1, e2, e3] at he
  exact he

/-- The IEEE difference of two finite `float32` of moderate size. -/
theorem sub_fact {a b xh : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : |a.toReal - b.toReal| ≤ 2 ^ 20) (h : FloatV.NaNEq (FloatV.sub a b) xh) :
    xh.isFinite = true ∧ |xh.toReal - (a.toReal - b.toReal)| ≤ |a.toReal - b.toReal| * u + 1 / 2 ^ 150 := by
  obtain ⟨hf, he⟩ := FloatV.sub_err_real Format.binary32_regular ha hb (by
    rw [max32_real]; exact hab.trans (by norm_num))
  have hx : xh = FloatV.sub a b := h.eq_of_isFinite hf
  subst hx
  rw [u32_real, eta32_real] at he
  exact ⟨hf, he⟩

/-- **The rescaling factor** `α̂ = exp (m_old − m_new)` (`tl.sub`, then `tl.exp`) between finite
references `μ ≤ μ'` with `μ' − μ ≤ 2050`: finite, within `(aA0 + aA1·(μ' − μ))·e^(μ−μ') + zeta0`
of the exact factor. -/
theorem alpha_ok {E : Env} (hP : Device E.P) {mi mn xh ah : F32} {μ μ' : ℝ}
    (hmi : mi.isFinite = true) (hmn : mn.isFinite = true) (hμ : mi.toReal = μ)
    (hμ' : mn.toReal = μ') (hle : μ ≤ μ') (hK : μ' - μ ≤ 2050)
    (hsub : FloatV.NaNEq (FloatV.sub mi mn) xh) (hexp : E.P.math .exp .binary32 xh ah) :
    ah.isFinite = true ∧
      |ah.toReal - Real.exp (μ - μ')| ≤
        (aA0 + aA1 * (μ' - μ)) * Real.exp (μ - μ') + zeta0 := by
  have hu := u_pos
  have hd0 : 0 ≤ μ' - μ := by linarith
  obtain ⟨hxf, hxe⟩ := sub_fact hmi hmn (by
    rw [hμ, hμ']; rw [abs_of_nonpos (by linarith)]; norm_num; linarith) hsub
  rw [hμ, hμ'] at hxe
  have habs : |μ - μ'| = μ' - μ := by rw [abs_of_nonpos (by linarith)]; ring
  rw [habs] at hxe
  set d := μ' - μ with hd
  set δ := d * u + 1 / 2 ^ 150 with hδ
  have hδ0 : 0 ≤ δ := by positivity
  have hdu : d * u ≤ 2050 * u := mul_le_mul_of_nonneg_right hK hu.le
  have hδ1 : δ ≤ 13 / 100000 := by
    have : (2050 : ℝ) * u + 1 / 2 ^ 150 ≤ 13 / 100000 := by unfold u; norm_num
    linarith
  have hxlo : |xh.toReal - (μ - μ')| ≤ δ := hxe
  have hxup : xh.toReal ≤ 1 := by
    have := (abs_le.mp hxlo).2
    have : δ ≤ 1 := by linarith
    linarith
  have hxabs : |xh.toReal| ≤ d + δ := by
    have h1 := abs_sub_abs_le_abs_sub xh.toReal (μ - μ')
    rw [abs_of_nonpos (by linarith : μ - μ' ≤ 0)] at h1
    have : -(μ - μ') = d := by rw [hd]; ring
    linarith
  obtain ⟨hf, he⟩ := exp_fact hP hxf hxup (by linarith) hexp
  refine ⟨hf, ?_⟩
  have hexpd := TVA.Softmax.abs_exp_sub_exp_le_mul (x := μ - μ') (y := xh.toReal) hxlo
    (by linarith)
  set ex := Real.exp (μ - μ') with hex
  have hex0 : 0 < ex := Real.exp_pos _
  set eh := Real.exp xh.toReal with heh
  have hq : δ * (1 + δ) ≤ 100013 / 100000 * δ := by nlinarith
  have heh1 : eh ≤ ex * (1 + 100013 / 100000 * δ - 100000 / 100000 * 0 + 0) := by
    have := (abs_le.mp hexpd).2
    nlinarith
  have hq2 : δ * (1 + δ) ≤ 131 / 1000000 := by nlinarith
  have hq3 : δ * (1 + δ) ≤ 100013 / 100000 * δ := hq
  have hη : (1 / 2 ^ 150 : ℝ) ≤ 1 / 1000000 * u := by unfold u; norm_num
  have hdu0 : 0 ≤ d * u := mul_nonneg hd0 hu.le
  have hexd : |eh - ex| ≤ δ * (1 + δ) * ex := hexpd
  have heh : eh ≤ ex * (1 + δ * (1 + δ)) := by
    have := (abs_le.mp hexd).2
    nlinarith
  have hA0 : 0 ≤ 4 * u + 2 * u * |xh.toReal| := by positivity
  have hs1 : (4 * u + 2 * u * |xh.toReal|) * eh ≤
      (4 * u + 2 * u * (d + δ)) * (ex * (1 + δ * (1 + δ))) := by
    have h1 : 4 * u + 2 * u * |xh.toReal| ≤ 4 * u + 2 * u * (d + δ) := by
      have := mul_le_mul_of_nonneg_left hxabs (by positivity : (0 : ℝ) ≤ 2 * u)
      linarith
    exact mul_le_mul h1 heh (Real.exp_pos _).le (by positivity)
  have hbracket : (4 * u + 2 * u * (d + δ)) * (1 + δ * (1 + δ)) + δ * (1 + δ) ≤ aA0 + aA1 * d := by
    have hdd : d * δ * u ≤ d * (13 / 100000) * u := by
      have : d * δ ≤ d * (13 / 100000) := mul_le_mul_of_nonneg_left hδ1 hd0
      exact mul_le_mul_of_nonneg_right this hu.le
    have h1 : (4 * u + 2 * u * (d + δ)) * (δ * (1 + δ)) ≤
        (4 * u + 2 * u * (d + δ)) * (131 / 1000000) :=
      mul_le_mul_of_nonneg_left hq2 (by positivity)
    have h2 : u * δ ≤ u * (13 / 100000) := mul_le_mul_of_nonneg_left hδ1 hu.le
    have h3 : 2 * u * (d + δ) * (1 + δ * (1 + δ)) = 2 * u * (d + δ) + 2 * u * (d + δ) * (δ * (1 + δ)) := by ring
    unfold aA0 aA1
    have hδeq : δ = d * u + 1 / 2 ^ 150 := rfl
    linarith [hdu0, h1, h2, hq3, hη, hu, hδeq]
  have h5 : |ah.toReal - ex| ≤ (aA0 + aA1 * d) * ex + zeta0 := by
    have h6 : |ah.toReal - ex| ≤ |ah.toReal - eh| + |eh - ex| := by
      have := abs_add_le (ah.toReal - eh) (eh - ex)
      rwa [show ah.toReal - eh + (eh - ex) = ah.toReal - ex by ring] at this
    have h7 : δ * (1 + δ) * ex ≤ δ * (1 + δ) * ex := le_rfl
    have h8 : (aA0 + aA1 * d) * ex ≥ ((4 * u + 2 * u * (d + δ)) * (1 + δ * (1 + δ)) + δ * (1 + δ)) * ex :=
      mul_le_mul_of_nonneg_right hbracket hex0.le
    have h9 : (1 / 2 ^ 140 : ℝ) ≤ zeta0 := by unfold zeta0; norm_num
    have e : (4 * u + 2 * u * (d + δ)) * (ex * (1 + δ * (1 + δ))) =
        ((4 * u + 2 * u * (d + δ)) * (1 + δ * (1 + δ))) * ex := by ring
    have e2 : ((4 * u + 2 * u * (d + δ)) * (1 + δ * (1 + δ)) + δ * (1 + δ)) * ex =
        ((4 * u + 2 * u * (d + δ)) * (1 + δ * (1 + δ))) * ex + δ * (1 + δ) * ex := by ring
    linarith
  exact h5

/-- **The first rescaling factor** (`m_old = −∞`) is `+0`. -/
theorem alpha_first {E : Env} (hP : Device E.P) {mn xh ah : F32} (hmn : mn.isFinite = true)
    (hsub : FloatV.NaNEq (FloatV.sub FloatV.negInf mn) xh)
    (hexp : E.P.math .exp .binary32 xh ah) : ah = FloatV.ofPat 0 := by
  have h1 : FloatV.sub FloatV.negInf mn = FloatV.negInf := FloatV.sub_negInf_left rfl hmn
  rw [h1] at hsub
  have hx : xh = FloatV.negInf := hsub.eq (FloatV.negInf_isNaN rfl)
  subst hx
  exact hP.expF32NegInf ah hexp

/-- **A masked lane's weight** (`ŝ = −∞`) is `+0`. -/
theorem lane_masked {E : Env} (hP : Device E.P) {mn xh ph : F32} (hmn : mn.isFinite = true)
    (hsub : FloatV.NaNEq (FloatV.sub FloatV.negInf mn) xh)
    (hexp : E.P.math .exp .binary32 xh ph) : ph = FloatV.ofPat 0 :=
  alpha_first hP hmn hsub hexp

/-- **A valid lane's weight** `p̂ = exp (ŝ − m_new)` (`tl.sub`, then `tl.exp`): the computed
score `sh` is finite, within `dS X` of the exact `Sk` (`|Sk| ≤ X ≤ 1024`), at most the finite new
reference `mn` (value `μ'`, `|μ'| ≤ X + 1`). Then `p̂` is finite and within
`rhoP X·e^(Sk−μ') + etaP` of `e^(Sk − μ')`. -/
theorem lane_ok {E : Env} (hP : Device E.P) {X : ℝ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024)
    {sh mn xh ph : F32} {Sk μ' : ℝ} (hsh : sh.isFinite = true) (hmn : mn.isFinite = true)
    (hμ' : mn.toReal = μ') (hS : |Sk| ≤ X) (hd : |sh.toReal - Sk| ≤ dS X)
    (hle : sh.toReal ≤ μ') (hμ'K : |μ'| ≤ X + 1)
    (hsub : FloatV.NaNEq (FloatV.sub sh mn) xh) (hexp : E.P.math .exp .binary32 xh ph) :
    ph.isFinite = true ∧ |ph.toReal - Real.exp (Sk - μ')| ≤ rhoP X * Real.exp (Sk - μ') + etaP := by
  have hu := u_pos
  have hdS0 : 0 ≤ dS X := by unfold dS; positivity
  have hdS : dS X ≤ 2 / 1000 := by unfold dS u; norm_num; nlinarith
  have hshabs : |sh.toReal| ≤ X + dS X := by
    have := abs_sub_abs_le_abs_sub sh.toReal Sk
    linarith
  set y := sh.toReal - μ' with hy
  have hy0 : y ≤ 0 := by linarith
  have hyabs : |y| ≤ 2 * X + 1002 / 1000 := by
    have : |y| ≤ |sh.toReal| + |μ'| := by
      rw [hy]; exact abs_sub _ _
    linarith
  obtain ⟨hxf, hxe⟩ := sub_fact hsh hmn (by
    rw [hμ']; show |y| ≤ _; refine hyabs.trans ?_; norm_num; linarith) hsub
  rw [hμ'] at hxe
  change |xh.toReal - y| ≤ |y| * u + 1 / 2 ^ 150 at hxe
  set δ := |y| * u + 1 / 2 ^ 150 + dS X with hδ
  have hη : (1 / 2 ^ 150 : ℝ) ≤ 1 / 1000000 * u := by unfold u; norm_num
  have hη' : (1 / 2 ^ 150 : ℝ) ≤ 1 / 1000000 := by norm_num
  have h60 : (2 : ℝ) ^ (-60 : ℤ) ≤ 1 / 1000000 * u := by unfold u; norm_num
  have hyu : |y| * u ≤ (2 * X + 1002 / 1000) * u := mul_le_mul_of_nonneg_right hyabs hu.le
  have huX : u * X ≤ u * 1024 := mul_le_mul_of_nonneg_left hX1 hu.le
  have hu1024 : u * 1024 ≤ 1 / 16000 := by unfold u; norm_num
  have hδ0 : 0 ≤ δ := by positivity
  have hδb : δ ≤ 30 * u * X + 1003 / 1000 * u + 1 / 1000000 * u + 1 / 1000000 * u := by
    have hdSdef : dS X = 28 * u * X + (2 : ℝ) ^ (-60 : ℤ) := rfl
    rw [hδ, hdSdef]; linarith [hyu, hη, h60]
  have hδ1 : δ ≤ 19 / 10000 := by nlinarith
  have hxlo : |xh.toReal - (Sk - μ')| ≤ δ := by
    have h1 : |xh.toReal - (Sk - μ')| ≤ |xh.toReal - y| + |y - (Sk - μ')| := by
      have := abs_add_le (xh.toReal - y) (y - (Sk - μ'))
      rwa [show xh.toReal - y + (y - (Sk - μ')) = xh.toReal - (Sk - μ') by ring] at this
    have h2 : y - (Sk - μ') = sh.toReal - Sk := by rw [hy]; ring
    rw [h2] at h1
    linarith
  have hxup : xh.toReal ≤ 1 := by
    have := (abs_le.mp hxe).2
    have h3 : |y| * u = -y * u := by rw [abs_of_nonpos hy0]
    nlinarith [hη']
  have hxabs : |xh.toReal| ≤ 2 * X + 1003 / 1000 := by
    have h1 := abs_sub_abs_le_abs_sub xh.toReal y
    have h2 : |y| * u ≤ 1 / 1000 := by nlinarith
    linarith
  obtain ⟨hf, he⟩ := exp_fact hP hxf hxup (by nlinarith) hexp
  refine ⟨hf, ?_⟩
  have hexpd := TVA.Softmax.abs_exp_sub_exp_le_mul (x := Sk - μ') (y := xh.toReal) hxlo
    (by linarith)
  set ex := Real.exp (Sk - μ') with hex
  have hex0 : 0 < ex := Real.exp_pos _
  set eh := Real.exp xh.toReal with heh
  have hδδ : δ * δ ≤ δ * (19 / 10000) := mul_le_mul_of_nonneg_left hδ1 hδ0
  have hq : δ * (1 + δ) ≤ 2 / 1000 := by linarith
  have hq3 : δ * (1 + δ) ≤ 10019 / 10000 * δ := by linarith
  have hehb : eh ≤ ex * (1 + δ * (1 + δ)) := by
    have := (abs_le.mp hexpd).2
    linarith
  have hA' : 4 * u + 2 * u * |xh.toReal| ≤ 6006 / 1000 * u + 4 * (u * X) := by
    have := mul_le_mul_of_nonneg_left hxabs (by positivity : (0 : ℝ) ≤ 2 * u)
    linarith
  have hA0 : 0 ≤ 4 * u + 2 * u * |xh.toReal| := by positivity
  have hs1 : (4 * u + 2 * u * |xh.toReal|) * eh ≤
      (6006 / 1000 * u + 4 * (u * X)) * (ex * (1 + δ * (1 + δ))) :=
    mul_le_mul hA' hehb (Real.exp_pos _).le (by positivity)
  have hbr : (6006 / 1000 * u + 4 * (u * X)) * (1 + δ * (1 + δ)) + δ * (1 + δ) ≤ rhoP X := by
    have h1 : (6006 / 1000 * u + 4 * (u * X)) * (δ * (1 + δ)) ≤
        (6006 / 1000 * u + 4 * (u * X)) * (2 / 1000) :=
      mul_le_mul_of_nonneg_left hq (by positivity)
    unfold rhoP
    have huX0 : 0 ≤ u * X := mul_nonneg hu.le hX0
    linarith [h1, hq3, hδb, hu, huX0]
  have h6 : |ph.toReal - ex| ≤ |ph.toReal - eh| + |eh - ex| := by
    have := abs_add_le (ph.toReal - eh) (eh - ex)
    rwa [show ph.toReal - eh + (eh - ex) = ph.toReal - ex by ring] at this
  have h9 : (1 / 2 ^ 140 : ℝ) ≤ etaP := by unfold etaP; norm_num
  have h8 : rhoP X * ex ≥ ((6006 / 1000 * u + 4 * (u * X)) * (1 + δ * (1 + δ)) + δ * (1 + δ)) * ex :=
    mul_le_mul_of_nonneg_right hbr hex0.le
  have e : (6006 / 1000 * u + 4 * (u * X)) * (ex * (1 + δ * (1 + δ))) =
      ((6006 / 1000 * u + 4 * (u * X)) * (1 + δ * (1 + δ))) * ex := by ring
  have e2 : ((6006 / 1000 * u + 4 * (u * X)) * (1 + δ * (1 + δ)) + δ * (1 + δ)) * ex =
      ((6006 / 1000 * u + 4 * (u * X)) * (1 + δ * (1 + δ))) * ex + δ * (1 + δ) * ex := by ring
  have hexd : |eh - ex| ≤ δ * (1 + δ) * ex := hexpd
  linarith

end Num

end Claims.PagedAttention
