import TVA.Device.Dot16Chunk
import TVA.Proof.Float.Error
import TVA.Proof.Int.Basic

/-!
# Weakening a device fact

A proof may use a fact with larger constants or a smaller domain than the device fact states,
or in a weaker shape. These lemmas derive the weaker statement; they state nothing new about the
device, so they are proved, not trusted.

* `MathAcc.mono`, `DotAcc.mono`, `DotLiftAcc.mono`, `DotScaledAcc.mono`, `DotChunkAcc.mono`,
  `DotChunkLoss.mono`, `DotChunk16Loss.mono`: larger constants, smaller domains.
* `CvtSat.cvtRn` (saturation is rounding to nearest in range), `CvtRn.cvtAcc` (one rounding's
  bound), `DivRn.divAcc` and `DivFull.of_divRn` (IEEE division meets `DivFull`),
  `DotAcc.liftAcc` (a bound on every value is a bound relative to lifted magnitudes),
  `DotChunkLoss.chunkAcc`, `DotChunk16Loss.step_left` (the first arm of the smaller of two
  bounds), `DotChunkLoss.of_chunkAcc_zero`.
* `liftMag_nonneg`, `abs_le_liftMag`, `abs_mul_le_liftMag`: a lifted magnitude is at least the
  magnitude.

`TVA.Device` uses them to derive the facts implied by others, and the reference platform
(`TVA.Device.Ref`) to meet the facts.
-/
namespace TVA.Device

variable {P : Platform}

/-! ## Lifted magnitudes -/

/-- A lifted magnitude is nonnegative. -/
theorem liftMag_nonneg (d : DType) (q : ℚ) : 0 ≤ liftMag d q := by
  unfold liftMag
  split_ifs
  · exact le_rfl
  · exact (abs_nonneg q).trans (le_max_left _ _)

/-- A lifted magnitude is at least the magnitude. -/
theorem abs_le_liftMag (d : DType) (q : ℚ) : |q| ≤ liftMag d q := by
  unfold liftMag; split_ifs with h
  · rw [h, abs_zero]
  · exact le_max_left _ _

/-- `|x·y| ≤ liftMag da x · liftMag db y`: a product's magnitude is at most its lifted one. -/
theorem abs_mul_le_liftMag (da db : DType) (x y : ℚ) :
    |x * y| ≤ liftMag da x * liftMag db y := by
  rw [abs_mul]
  exact mul_le_mul (abs_le_liftMag da x) (abs_le_liftMag db y) (abs_nonneg y)
    (liftMag_nonneg da x)

/-! ## Larger constants, smaller domains -/

/-- `MathAcc` on a smaller domain with a bound that is at least as large at every point. -/
theorem MathAcc.mono {fn : MathFn} {f : Format} {dom dom' : Set ℝ} {rel abs rel' abs' : ℝ}
    (h : MathAcc P fn f dom rel abs) (hdom : dom' ⊆ dom)
    (hb : ∀ g, fn.ref = some g → ∀ x ∈ dom', x ∈ fn.natDom →
      rel * |g x| + abs ≤ rel' * |g x| + abs') :
    MathAcc P fn f dom' rel' abs' := by
  obtain ⟨g, hg, h⟩ := h
  refine ⟨g, hg, fun x r hr hx hd hn hroom => ?_⟩
  have hb' := hb g hg _ hd hn
  obtain ⟨hfin, herr⟩ := h x r hr hx (hdom hd) hn (by linarith)
  exact ⟨hfin, herr.trans hb'⟩

/-- `DotAcc` on a smaller domain with constants at least as large. -/
theorem DotAcc.mono {cfg : DotCfg} {dom dom' : ℚ → Prop} {rel abs rel' abs' : ℕ → ℚ}
    (h : DotAcc P cfg dom rel abs) (hdom : ∀ q, dom' q → dom q) (hrel : ∀ K, rel K ≤ rel' K)
    (habs : ∀ K, abs K ≤ abs' K) : DotAcc P cfg dom' rel' abs' := by
  intro M N K a b acc r hr i n va vb vc hva hvb hd hvc hroom
  have hmag : 0 ≤ |vc| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hm := mul_le_mul_of_nonneg_right (hrel K) hmag
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb
    (fun k => ⟨hdom _ (hd k).1, hdom _ (hd k).2⟩) hvc (by linarith [habs K])
  exact ⟨vr, hvr, by linarith [habs K]⟩

/-- `DotLiftAcc` with constants at least as large. -/
theorem DotLiftAcc.mono {cfg : DotCfg} {rel abs rel' abs' : ℕ → ℚ} (h : DotLiftAcc P cfg rel abs)
    (hrel : ∀ K, rel K ≤ rel' K) (habs : ∀ K, abs K ≤ abs' K) : DotLiftAcc P cfg rel' abs' := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hL : 0 ≤ |vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ =>
      mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _))
  have hm := mul_le_mul_of_nonneg_right (hrel K) hL
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb hvc (by linarith [habs K])
  exact ⟨vr, hvr, by linarith [habs K]⟩

/-- `DotScaledAcc` on smaller domains with constants at least as large. -/
theorem DotScaledAcc.mono {cfg : ScaledCfg} {dom dom' : ℚ → Prop} {sdom sdom' : Nat → Prop}
    {rel abs rel' abs' : ℕ → ℚ} (h : DotScaledAcc P cfg dom sdom rel abs)
    (hdom : ∀ q, dom' q → dom q) (hsdom : ∀ b, sdom' b → sdom b) (hrel : ∀ K, rel K ≤ rel' K)
    (habs : ∀ K, abs K ≤ abs' K) : DotScaledAcc P cfg dom' sdom' rel' abs' := by
  intro M N K a sa b sb acc r hr i n va vb hva hvb hd hsa hsb hacc hroom
  have hmag : 0 ≤ |(acc.get2 i n).toRat| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hm := mul_le_mul_of_nonneg_right (hrel K) hmag
  obtain ⟨hfin, herr⟩ := h a sa b sb acc r hr i n va vb hva hvb
    (fun k => ⟨hdom _ (hd k).1, hdom _ (hd k).2⟩) (fun t ht k => hsdom _ (hsa t ht k))
    (fun t ht k => hsdom _ (hsb t ht k)) hacc (by linarith [habs K])
  exact ⟨hfin, by linarith [habs K]⟩

/-- `DotChunkAcc` with constants at least as large. -/
theorem DotChunkAcc.mono {cfg : DotCfg} {rho sigma alpha rho' sigma' alpha' : ℚ}
    (h : DotChunkAcc P cfg rho sigma alpha) (hr : rho ≤ rho') (hs : sigma ≤ sigma')
    (ha : alpha ≤ alpha') : DotChunkAcc P cfg rho' sigma' alpha' := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
  have e := hstep j hj Mj hM hk
  have e1 := mul_le_mul_of_nonneg_right hr hM0
  have e2 := mul_le_mul_of_nonneg_right hs (abs_nonneg (s j + chunkSum8 va vb j))
  linarith

/-- `DotChunkLoss` with constants at least as large. -/
theorem DotChunkLoss.mono {cfg : DotCfg}
    {rho sigma alpha kappa rho' rho₂ sigma₂ alpha₂ kappa₂ rho'₂ : ℚ}
    (h : DotChunkLoss P cfg rho sigma alpha kappa rho') (hr : rho ≤ rho₂) (hs : sigma ≤ sigma₂)
    (ha : alpha ≤ alpha₂) (hκ : kappa ≤ kappa₂) (hr' : rho' ≤ rho'₂) :
    DotChunkLoss P cfg rho₂ sigma₂ alpha₂ kappa₂ rho'₂ := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
  have hA0 : 0 ≤ chunkAbs8 va vb j := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have e := hstep j hj Mj hM hk
  have e1 : min (rho * Mj) (kappa * chunkAbs8 va vb j + rho' * Mj) ≤
      min (rho₂ * Mj) (kappa₂ * chunkAbs8 va vb j + rho'₂ * Mj) :=
    min_le_min (mul_le_mul_of_nonneg_right hr hM0)
      (add_le_add (mul_le_mul_of_nonneg_right hκ hA0) (mul_le_mul_of_nonneg_right hr' hM0))
  have e2 := mul_le_mul_of_nonneg_right hs (abs_nonneg (s j + chunkSum8 va vb j))
  linarith

/-- `DotChunk16Loss` with constants at least as large. -/
theorem DotChunk16Loss.mono {cfg : DotCfg}
    {rho sigma alpha kappa rho' rho₂ sigma₂ alpha₂ kappa₂ rho'₂ : ℚ}
    (h : DotChunk16Loss P cfg rho sigma alpha kappa rho') (hr : rho ≤ rho₂) (hs : sigma ≤ sigma₂)
    (ha : alpha ≤ alpha₂) (hκ : kappa ≤ kappa₂) (hr' : rho' ≤ rho'₂) :
    DotChunk16Loss P cfg rho₂ sigma₂ alpha₂ kappa₂ rho'₂ := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
  have hA0 : 0 ≤ chunkAbs16 va vb j := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have e := hstep j hj Mj hM hk
  have e1 : min (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj) ≤
      min (rho₂ * Mj) (kappa₂ * chunkAbs16 va vb j + rho'₂ * Mj) :=
    min_le_min (mul_le_mul_of_nonneg_right hr hM0)
      (add_le_add (mul_le_mul_of_nonneg_right hκ hA0) (mul_le_mul_of_nonneg_right hr' hM0))
  have e2 := mul_le_mul_of_nonneg_right hs (abs_nonneg (s j + chunkSum16 va vb j))
  linarith

/-! ## Weaker shapes -/

/-- A saturating conversion rounds to nearest even in range: `CvtSat` gives `CvtRn`. -/
theorem CvtSat.cvtRn {f g : Format} (hg : g.Regular) (h : CvtSat P f g) : CvtRn P f g := by
  intro x r hr hx hmax
  have := h x r hr
  rwa [FloatV.cvtSat_of_le hg .rn hx hmax] at this

/-- Rounding to nearest in range is within one rounding's bound: `CvtRn` gives `CvtAcc`. -/
theorem CvtRn.cvtAcc {f g : Format} (hg : g.Regular) (h : CvtRn P f g) : CvtAcc P f g := by
  intro x r hr hx hmax
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err hg hx hmax
  have hne := h x r hr hx hmax
  rw [FloatV.cvtWith_rn] at hne
  rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
  exact ⟨hfin, herr⟩

/-- IEEE division meets `DivAcc` with one rounding's bound (`u`, `η`) over the whole range of
the format. -/
theorem DivRn.divAcc {f : Format} (hf : f.Regular) (h : DivRn P f) :
    DivAcc P f f.maxVal f.u f.eta := by
  intro x y r hr hx hy hy0 hmax
  have hz : f.isZero y.pat = false := by
    cases hzy : f.isZero y.pat
    · rfl
    · exact absurd (FloatV.toRat_of_isZero hzy) hy0
  obtain ⟨hfin, herr⟩ := FloatV.divRn_err hf hx hy hz hmax
  rw [(h x y r hr).eq (FloatV.isFinite_iff.mp hfin).1]
  exact ⟨hfin, by rw [mul_comm]; exact herr⟩

/-- IEEE `float32` division meets `DivFull`: a proof that cites `DivFull` also holds when `/` is
IEEE. -/
theorem DivFull.of_divRn (h : DivRn P .binary32) : DivFull P := by
  intro x y r hr hx hy hy0 hq
  have hmax : (2 : ℚ) ^ 126 ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hfin, herr⟩ := DivRn.divAcc Format.binary32_regular h x y r hr hx hy hy0 (hq.trans hmax)
  refine ⟨hfin, herr.trans ?_⟩
  have hu : Format.binary32.u ≤ 2 ^ (-22 : ℤ) := by
    rw [Format.binary32_u]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num)
  have he : Format.binary32.eta ≤ 2 ^ (-148 : ℤ) := by
    rw [Format.binary32_eta]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num)
  exact add_le_add (mul_le_mul_of_nonneg_right hu (abs_nonneg _)) he

/-- A `DotAcc` fact on every value gives `DotLiftAcc` with the same constants, since
`|va·vb| ≤ liftMag va · liftMag vb`. -/
theorem DotAcc.liftAcc {cfg : DotCfg} {rel abs : ℕ → ℚ} (h : DotAcc P cfg (fun _ => True) rel abs)
    (hrel : ∀ K, 0 ≤ rel K) : DotLiftAcc P cfg rel abs := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hle : ∑ k, |va k * vb k| ≤ ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) :=
    Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _
  have hm := mul_le_mul_of_nonneg_left (add_le_add_right hle |vc|) (hrel K)
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb (fun _ => ⟨trivial, trivial⟩) hvc
    (by linarith)
  exact ⟨vr, hvr, by linarith⟩

/-- `DotChunkLoss` gives `DotChunkAcc` with the same `rho`, `sigma`, `alpha` (its first arm), on
the same partial values. -/
theorem DotChunkLoss.chunkAcc {cfg : DotCfg} {rho sigma alpha kappa rho' : ℚ}
    (h : DotChunkLoss P cfg rho sigma alpha kappa rho') : DotChunkAcc P cfg rho sigma alpha := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have e := hstep j hj Mj hM hk
  have := min_le_left (rho * Mj) (kappa * chunkAbs8 va vb j + rho' * Mj)
  linarith

/-- `DotF8ChunkLoss` gives `DotF8Chunk`: `DotE4m3ChunkLoss` gives `DotE4m3Chunk`, and likewise
for the other pairs. -/
theorem DotF8ChunkLoss.chunk {a b : DType} (h : DotF8ChunkLoss P a b) : DotF8Chunk P a b :=
  fun p => (h p).chunkAcc

/-- `DotChunkAcc` with `rho = 0` gives `DotChunkLoss` for any nonnegative `rho`, `kappa`,
`rho'`: both arms of the minimum are then at least `0`. -/
theorem DotChunkLoss.of_chunkAcc_zero {cfg : DotCfg} {sigma alpha rho kappa rho' : ℚ}
    (h : DotChunkAcc P cfg 0 sigma alpha) (hρ : 0 ≤ rho) (hκ : 0 ≤ kappa) (hρ' : 0 ≤ rho') :
    DotChunkLoss P cfg rho sigma alpha kappa rho' := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
  have e := hstep j hj Mj hM hk
  have h1 : 0 ≤ min (rho * Mj) (kappa * chunkAbs8 va vb j + rho' * Mj) :=
    le_min (mul_nonneg hρ hM0)
      (add_nonneg (mul_nonneg hκ (Finset.sum_nonneg fun _ _ => abs_nonneg _)) (mul_nonneg hρ' hM0))
  linarith

/-- The first arm of `DotChunk16Loss` alone: each step within
`rho·Mj + sigma·|s j + c_j| + alpha`. -/
theorem DotChunk16Loss.step_left {cfg : DotCfg} {rho sigma alpha kappa rho' : ℚ}
    (h : DotChunk16Loss P cfg rho sigma alpha kappa rho') ⦃M N K : Nat⦄
    (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val) (acc r : Tensor [M, N] cfg.acc.Val)
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| +
        (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤ cfg.acc.bound) :
    ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧
      s (chunks16 K) = vr ∧
      ∀ j < chunks16 K, ∀ Mj : ℚ, |s j| ≤ Mj →
        (∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
        |s (j + 1) - (s j + chunkSum16 va vb j)| ≤
          rho * Mj + sigma * |s j + chunkSum16 va vb j| + alpha := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have e := hstep j hj Mj hM hk
  have := min_le_left (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj)
  linarith

end TVA.Device
