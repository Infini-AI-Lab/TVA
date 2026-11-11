import TVA.Device.Ref.Ops

/-!
# The chunk facts on the reference platform

`Platform.ref` rounds the exact `acc + Σ a·b` of a `tl.dot` once. Its partial values for the
chunk facts (`TVA.Device.Dot8Chunk`, `TVA.Device.Dot16Chunk`) are the exact partial sums
`acc + Σ_{i<j} c_i` before the last chunk and the rounded result after it: every step but the
last is exact, and the last is one rounding of `s j + c_j`, within `u·|s j + c_j| + η`. So the
chunk shapes hold with `sigma ≥ u`, `alpha ≥ η` and any nonnegative other constants
(`dotChunkAcc`, `dotChunkLoss`, `dotChunk16Loss`).
-/
namespace TVA.Platform.ref

open Device

/-- The exact partial sums of the chunks from `v`: `v + Σ_{i<j} c i`. -/
private def psum (v : ℚ) (c : ℕ → ℚ) (j : ℕ) : ℚ := v + ∑ i ∈ Finset.range j, c i

private theorem psum_zero (v : ℚ) (c : ℕ → ℚ) : psum v c 0 = v := by simp [psum]

private theorem psum_succ (v : ℚ) (c : ℕ → ℚ) (j : ℕ) :
    psum v c (j + 1) = psum v c j + c j := by
  simp [psum, Finset.sum_range_succ, add_assoc]

/-- The partial values of the reference platform: the exact partial sums for `j < nC`, then the
result `v`. Every step but the last is exact; the last is from `vc + Σ va·vb` to `v`. -/
private theorem steps {nC : ℕ} (hn1 : 1 ≤ nC) (vc v : ℚ) (c : ℕ → ℚ) :
    let s : ℕ → ℚ := fun j => if j < nC then psum vc c j else v
    s 0 = vc ∧ s nC = v ∧ ∀ j < nC,
      (j + 1 < nC ∧ s (j + 1) - (s j + c j) = 0) ∨
      (j + 1 = nC ∧ s (j + 1) = v ∧ s j + c j = psum vc c nC) := by
  intro s
  refine ⟨?_, ?_, fun j hj => ?_⟩
  · show (if 0 < nC then _ else v) = vc
    rw [ite_eq_left (by omega : 0 < nC), psum_zero]
  · show (if nC < nC then _ else v) = v
    rw [ite_eq_right (lt_irrefl nC)]
  · have hsj : s j = psum vc c j := ite_eq_left hj
    by_cases hj1 : j + 1 < nC
    · refine Or.inl ⟨hj1, ?_⟩
      have hs1 : s (j + 1) = psum vc c (j + 1) := ite_eq_left hj1
      rw [hs1, hsj, psum_succ]; ring
    · have hjn : j + 1 = nC := by omega
      refine Or.inr ⟨hjn, ite_eq_right hj1, ?_⟩
      rw [hsj, ← psum_succ, hjn]

/-- **`DotChunkAcc` holds of `Platform.ref`** for a float accumulator, with nonnegative `rho`,
`sigma` at least one rounding's `u` and `alpha` at least its `η`. -/
theorem dotChunkAcc {cfg : DotCfg} {g : Format} (hg : cfg.acc.format? = some g)
    {rho sigma alpha : ℚ} (hrho : 0 ≤ rho) (hsigma : g.u ≤ sigma) (halpha : g.eta ≤ alpha) :
    DotChunkAcc Platform.ref cfg rho sigma alpha := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hu : 0 ≤ g.u := Format.u_nonneg
  have he : 0 ≤ g.eta := Format.eta_nonneg
  by_cases hK : K = 0
  · subst hK
    have hri : r.get2 i n = acc.get2 i n := by rw [hr i n, ite_eq_left rfl]
    refine ⟨fun _ => vc, vc, by rw [hri]; exact hvc, rfl, rfl, fun j hj => ?_⟩
    exact absurd hj (by simp [chunks8])
  have hex : cfg.exact a b acc i n = vc + ∑ k, va k * vb k := by
    simp only [DotCfg.exact, hva, hvb, hvc, Option.getD_some]
  have hri : r.get2 i n = cfg.acc.ofRat (vc + ∑ k, va k * vb k) := by
    rw [hr i n, ite_eq_right hK, hex]
  have hmag : 0 ≤ |vc| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hrel : 0 ≤ relDot8 K := by
    unfold relDot8
    have : (1 : ℚ) ≤ (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K :=
      one_le_pow₀ (by have : (0 : ℚ) ≤ 2 ^ (-7 : ℤ) := by positivity
                      linarith)
    linarith
  have habs : 0 ≤ absDot8 K := by unfold absDot8; positivity
  have hqmax : |vc + ∑ k, va k * vb k| ≤ g.maxVal := by
    rw [← DType.bound_of_format? hg]
    have := mul_nonneg hrel hmag
    have := (abs_add_le vc _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs
      (fun k => va k * vb k) Finset.univ))
    linarith
  obtain ⟨v, hv, herr⟩ := DType.val?_ofRat_float hg hqmax
  have hn1 : 1 ≤ chunks8 K := by unfold chunks8; omega
  obtain ⟨h0, hn, hstep⟩ := steps hn1 vc v (chunkSum8 va vb)
  refine ⟨fun j => if j < chunks8 K then psum vc (chunkSum8 va vb) j else v, v,
    by rw [hri]; exact hv, h0, hn, fun j hj Mj hM _ => ?_⟩
  have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
  have := mul_nonneg hrho hM0
  rcases hstep j hj with ⟨_, h0'⟩ | ⟨_, hs1, hsum⟩
  · rw [h0', abs_zero]
    have := mul_nonneg (hu.trans hsigma) (abs_nonneg ((if j < chunks8 K then psum vc
      (chunkSum8 va vb) j else v) + chunkSum8 va vb j))
    linarith
  · rw [hs1, hsum, psum, sum_chunkSum8]
    have h1 : |vc + ∑ k, va k * vb k| * g.u ≤ sigma * |vc + ∑ k, va k * vb k| := by
      rw [mul_comm]; exact mul_le_mul_of_nonneg_right hsigma (abs_nonneg _)
    linarith

/-- **`DotChunkLoss` holds of `Platform.ref`** for a float accumulator, with nonnegative `rho`,
`kappa`, `rho'`, `sigma` at least one rounding's `u` and `alpha` at least its `η`. -/
theorem dotChunkLoss {cfg : DotCfg} {g : Format} (hg : cfg.acc.format? = some g)
    {rho sigma alpha kappa rho' : ℚ} (hrho : 0 ≤ rho) (hkappa : 0 ≤ kappa) (hrho' : 0 ≤ rho')
    (hsigma : g.u ≤ sigma) (halpha : g.eta ≤ alpha) :
    DotChunkLoss Platform.ref cfg rho sigma alpha kappa rho' :=
  DotChunkLoss.of_chunkAcc_zero (dotChunkAcc hg le_rfl hsigma halpha) hrho hkappa hrho'

/-- **`DotChunk16Loss` holds of `Platform.ref`** for a float accumulator, with nonnegative `rho`,
`kappa`, `rho'`, `sigma` at least one rounding's `u` and `alpha` at least its `η`. -/
theorem dotChunk16Loss {cfg : DotCfg} {g : Format} (hg : cfg.acc.format? = some g)
    {rho sigma alpha kappa rho' : ℚ} (hrho : 0 ≤ rho) (hkappa : 0 ≤ kappa) (hrho' : 0 ≤ rho')
    (hsigma : g.u ≤ sigma) (halpha : g.eta ≤ alpha) :
    DotChunk16Loss Platform.ref cfg rho sigma alpha kappa rho' := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hu : 0 ≤ g.u := Format.u_nonneg
  have he : 0 ≤ g.eta := Format.eta_nonneg
  by_cases hK : K = 0
  · subst hK
    have hri : r.get2 i n = acc.get2 i n := by rw [hr i n, ite_eq_left rfl]
    refine ⟨fun _ => vc, vc, by rw [hri]; exact hvc, rfl, rfl, fun j hj => ?_⟩
    exact absurd hj (by simp [chunks16])
  have hex : cfg.exact a b acc i n = vc + ∑ k, va k * vb k := by
    simp only [DotCfg.exact, hva, hvb, hvc, Option.getD_some]
  have hri : r.get2 i n = cfg.acc.ofRat (vc + ∑ k, va k * vb k) := by
    rw [hr i n, ite_eq_right hK, hex]
  have hmag : 0 ≤ |vc| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hrel : 0 ≤ relChunk16 K := by
    unfold relChunk16 rhoChunk16
    have : (1 : ℚ) ≤ (1 + 2 ^ (-13 : ℤ)) ^ chunks16 K :=
      one_le_pow₀ (by have : (0 : ℚ) ≤ 2 ^ (-13 : ℤ) := by positivity
                      linarith)
    linarith
  have habs : 0 ≤ absChunk16 K := by unfold absChunk16 rhoChunk16 alphaChunk16; positivity
  have hqmax : |vc + ∑ k, va k * vb k| ≤ g.maxVal := by
    rw [← DType.bound_of_format? hg]
    have := mul_nonneg hrel hmag
    have := (abs_add_le vc _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs
      (fun k => va k * vb k) Finset.univ))
    linarith
  obtain ⟨v, hv, herr⟩ := DType.val?_ofRat_float hg hqmax
  have hn1 : 1 ≤ chunks16 K := by unfold chunks16; omega
  obtain ⟨h0, hn, hstep⟩ := steps hn1 vc v (chunkSum16 va vb)
  refine ⟨fun j => if j < chunks16 K then psum vc (chunkSum16 va vb) j else v, v,
    by rw [hri]; exact hv, h0, hn, fun j hj Mj hM _ => ?_⟩
  have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
  have hmin : 0 ≤ min (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj) :=
    le_min (mul_nonneg hrho hM0)
      (add_nonneg (mul_nonneg hkappa (Finset.sum_nonneg fun _ _ => abs_nonneg _))
        (mul_nonneg hrho' hM0))
  rcases hstep j hj with ⟨_, h0'⟩ | ⟨_, hs1, hsum⟩
  · rw [h0', abs_zero]
    have := mul_nonneg (hu.trans hsigma) (abs_nonneg ((if j < chunks16 K then psum vc
      (chunkSum16 va vb) j else v) + chunkSum16 va vb j))
    linarith
  · rw [hs1, hsum, psum, sum_chunkSum16]
    have h1 : |vc + ∑ k, va k * vb k| * g.u ≤ sigma * |vc + ∑ k, va k * vb k| := by
      rw [mul_comm]; exact mul_le_mul_of_nonneg_right hsigma (abs_nonneg _)
    linarith

end TVA.Platform.ref
