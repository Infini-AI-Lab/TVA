import TVA.Rules.Num.Dot16Chunk
import TVA.Lemmas.Dot8Chunk

/-!
# Using the 16-bit chunk facts (`DotF16Chunk`, `DotBF16Chunk`)

Page: `docs/rules/num.md` ("`tl.dot` of `float16` and `bfloat16`, chunk by chunk"). The facts
(`TVA.Rules.Num.Dot16Chunk`, library issue L150) give, per output element, partial values
`s 0 = acc, …, s n = result` (`n = chunks16 K`) and a bound on each chunk's step. These lemmas
compose the steps into a bound from the exact inputs alone, and show the facts satisfiable. The
recursions and their closed forms are the fp8 ones (`TVA.Lemmas.Dot8Chunk`: `psum`, `errB`,
`chain_err`, `errB_le`, `errM`, `chain_errM`, `errM_le_errB`, `errL`, `errM_le_errL`, `errL_le`),
which take any sequence of chunk sums, magnitudes and bounds.

* `Dot16Chunk.sum_chunkSum16`, `sum_chunkAbs16`, `psum_chunks16`: the chunks add up to the dot.
* `Dot16Chunk.chunkLift16`: the largest lifted product of a chunk (`le_chunkLift16`,
  `chunkLift16_nonneg`), a choice of `m j`.
* `Dot16Chunk.DotChunk16Loss.err`, `Dot16Chunk.Dot16BitChunk.err`: the result is within
  `Dot8Chunk.errM ρ σ α κ ρ' vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K)` of
  `acc + Σ a·b`: `errM (j+1) = (1+σ)·errM j + min (ρ·(X j + errM j)) (κ·A_j + ρ'·(X j + errM j)) +
  σ·|S (j+1)| + α`, `X j = max |S j| (m j)`, `S j` the exact partial sums. It is at most
  `Dot8Chunk.errB` with the first arm alone (`errM_le_errB`) and at most `errL`, the second arm alone,
  growing by `1 + 2^-22 + 2^-20` per chunk of 64 (`errM_le_errL`, `errL_le`).
* `Dot16Chunk.DotChunk16Loss.errB`: the first arm alone, `errB`.
* `Dot16Chunk.Dot16BitChunk.liftAcc`: the chunk facts give `Rules.Num.DotLiftAcc` (a whole `tl.dot`
  within `rel K·L + abs K` of `acc + Σ a·b`, `L = |acc| + Σ liftMag a_k · liftMag b_k`, with room on
  `L`) with `relLift16 K = (1 + 2^-13 + 2^-20)^⌈K/64⌉ − 1` and
  `absLift16 K = ⌈K/64⌉·(1 + 2^-13 + 2^-20)^⌈K/64⌉·2^-143`.
* `Platform.ref.dotChunk16Loss`, `Platform.sm90ref.dotF16Chunk`, `.dotBF16Chunk`: the facts hold of
  the reference platforms (one rounding of the exact result), so a bundle citing them proves
  `Claim.Consistent` with `Platform.sm90ref`, as with the other sm_90 facts.
-/
namespace TVA

open TVA.Math.Float Rules.Num Dot8Chunk

namespace Dot16Chunk

/-- Every `k < K` lies in a chunk `j < chunks16 K`. -/
theorem div_lt_chunks16 {K : ℕ} (k : Fin K) : k.val / 64 < chunks16 K := by
  unfold chunks16; have := k.isLt; omega

/-- The chunks' exact sums add up to the whole sum. -/
theorem sum_chunkSum16 {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks16 K), chunkSum16 va vb j = ∑ k, va k * vb k := by
  unfold chunkSum16 chunk16
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 64)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks16 k)) _

/-- The chunks' exact magnitudes add up to `Σ|a·b|`. -/
theorem sum_chunkAbs16 {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks16 K), chunkAbs16 va vb j = ∑ k, |va k * vb k| := by
  unfold chunkAbs16 chunk16
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 64)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks16 k)) _

/-- The partial sum over all chunks of a dot is `acc + Σ a·b`. -/
theorem psum_chunks16 {K : ℕ} (v : ℚ) (va vb : Fin K → ℚ) :
    psum v (chunkSum16 va vb) (chunks16 K) = v + ∑ k, va k * vb k := by
  rw [psum, sum_chunkSum16]

theorem chunkAbs16_nonneg {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : 0 ≤ chunkAbs16 va vb j :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

theorem abs_chunkSum16_le {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    |chunkSum16 va vb j| ≤ chunkAbs16 va vb j :=
  Finset.abs_sum_le_sum_abs _ _

/-- The largest lifted product of chunk `j` (`0` for an empty chunk). -/
def chunkLift16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  (chunk16 K j).fold max 0 fun k => liftMag da (va k) * liftMag db (vb k)

theorem le_chunkLift16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) {j : ℕ} {k : Fin K}
    (hk : k ∈ chunk16 K j) : liftMag da (va k) * liftMag db (vb k) ≤ chunkLift16 da db va vb j := by
  unfold chunkLift16
  exact (Finset.le_fold_max _).mpr (Or.inr ⟨k, hk, le_rfl⟩)

theorem chunkLift16_nonneg (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    0 ≤ chunkLift16 da db va vb j := by
  unfold chunkLift16
  exact (Finset.le_fold_max _).mpr (Or.inl le_rfl)

theorem rhoChunk16_nonneg : 0 ≤ rhoChunk16 := by unfold rhoChunk16; positivity
theorem sigmaChunk16_nonneg : 0 ≤ sigmaChunk16 := by unfold sigmaChunk16; positivity
theorem alphaChunk16_nonneg : 0 ≤ alphaChunk16 := by unfold alphaChunk16; positivity
theorem kappaChunk16_nonneg : 0 ≤ kappaChunk16 := by unfold kappaChunk16; norm_num
theorem rhoLossChunk16_nonneg : 0 ≤ rhoLossChunk16 := by unfold rhoLossChunk16; positivity

/-- **A 16-bit `tl.dot` from its exact inputs.** Under `DotChunk16Loss`, for any bounds `m j` on
the lifted products of each chunk (`chunkLift16` will do), the result is within
`errM ρ σ α κ ρ' vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K)` of `vc + Σ va·vb`. -/
theorem DotChunk16Loss.err {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunk16Loss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hρ' : 0 ≤ ρ') {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤
      cfg.acc.bound)
    (m : ℕ → ℚ) (hm : ∀ j < chunks16 K, ∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errM ρ σ α κ ρ' vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_errM (s := s) (c := chunkSum16 va vb) (a := chunkAbs16 va vb) (m := m)
    (n := chunks16 K) hρ hσ hρ'
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj) (chunks16 K) le_rfl
  rw [hn, h0, psum_chunks16] at e
  exact e

/-- **The first arm alone, composed**: the result is within
`errB ρ σ α vc (chunkSum16 va vb) m (chunks16 K)` of `vc + Σ va·vb`. -/
theorem DotChunk16Loss.errB {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunk16Loss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤
      cfg.acc.bound)
    (m : ℕ → ℚ) (hm : ∀ j < chunks16 K, ∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤ Dot8Chunk.errB ρ σ α vc (chunkSum16 va vb) m (chunks16 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h.step_left a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_err (s := s) (c := chunkSum16 va vb) (m := m) (n := chunks16 K) hρ hσ
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj) (chunks16 K) le_rfl
  rw [hn, h0, psum_chunks16] at e
  exact e

/-- **`DotF16Chunk` and `DotBF16Chunk`, from the exact inputs**: the result is within
`errM 2^-13 2^-20 2^-143 2 2^-22 vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K)` of
`vc + Σ va·vb`. -/
theorem Dot16BitChunk.err {P : Platform} {da db : DType} (h : Dot16BitChunk P da db) (p : InputPrecision)
    {M N K : ℕ} {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {acc r : Tensor [M, N] F32}
    (hd : P.dot ⟨da, db, .f32, p, 0⟩ a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ)
    (vc : ℚ) (hva : ∀ k, da.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k))
    (hvc : DType.val? .f32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤
      DType.bound .f32)
    (m : ℕ → ℚ) (hm : ∀ j < chunks16 K, ∀ k ∈ chunk16 K j, liftMag da (va k) * liftMag db (vb k) ≤ m j) :
    ∃ vr, DType.val? .f32 (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errM rhoChunk16 sigmaChunk16 alphaChunk16 kappaChunk16 rhoLossChunk16 vc (chunkSum16 va vb)
          (chunkAbs16 va vb) m (chunks16 K) :=
  DotChunk16Loss.err (h p) rhoChunk16_nonneg sigmaChunk16_nonneg rhoLossChunk16_nonneg hd i n va vb vc
    hva hvb hvc hroom m hm

/-! ## A bound on the whole dot, relative to lifted magnitudes -/

/-- The lifted magnitude of chunk `j`: `Σ_{k ∈ chunk16 K j} liftMag da (va k) · liftMag db (vb k)`. -/
def chunkLiftSum16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  ∑ k ∈ chunk16 K j, liftMag da (va k) * liftMag db (vb k)

theorem sum_chunkLiftSum16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks16 K), chunkLiftSum16 da db va vb j =
      ∑ k, liftMag da (va k) * liftMag db (vb k) := by
  unfold chunkLiftSum16 chunk16
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 64)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks16 k)) _

/-- **A whole 16-bit `tl.dot` from the chunk steps.** Under `DotChunk16Loss`, the result is within
`DotLoop.bound (ρ + σ) α ⌈K/64⌉ L` of `vc + Σ va·vb`, `L = |vc| + Σ liftMag a_k · liftMag b_k`
(each chunk taken with `Mj = |s j| + A j`, the first arm, composed by `DotLoop.accum_step`). -/
theorem DotChunk16Loss.liftErr {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunk16Loss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤
      cfg.acc.bound) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        DotLoop.bound (ρ + σ) α (chunks16 K) (|vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k)) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h.step_left a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have hA0 : ∀ j, 0 ≤ chunkLiftSum16 cfg.a cfg.b va vb j := fun j =>
    Finset.sum_nonneg fun k _ => mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _)
  have hcA : ∀ j, |chunkSum16 va vb j| ≤ chunkLiftSum16 cfg.a cfg.b va vb j := fun j =>
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _)
  have hρσ : 0 ≤ ρ + σ := add_nonneg hρ hσ
  have key : ∀ j ≤ chunks16 K, |s j - psum vc (chunkSum16 va vb) j| ≤
      DotLoop.bound (ρ + σ) α j (psum |vc| (chunkLiftSum16 cfg.a cfg.b va vb) j) := by
    intro j
    induction j with
    | zero => intro _; simp [h0, psum_zero, DotLoop.bound]
    | succ j ih =>
      intro hj
      have e := ih (by omega)
      have hk : ∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤
          |s j| + chunkLiftSum16 cfg.a cfg.b va vb j := fun k hk =>
        (Finset.single_le_sum (f := fun k => liftMag cfg.a (va k) * liftMag cfg.b (vb k))
          (fun k _ => mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _)) hk).trans
          (le_add_of_nonneg_left (abs_nonneg _))
      have hs := hstep j (by omega) (|s j| + chunkLiftSum16 cfg.a cfg.b va vb j)
        (le_add_of_nonneg_right (hA0 j)) hk
      have hsc : |s j + chunkSum16 va vb j| ≤ |s j| + chunkLiftSum16 cfg.a cfg.b va vb j :=
        (abs_add_le _ _).trans (by linarith [hcA j])
      have hs' : |s (j + 1) - (s j + chunkSum16 va vb j)| ≤
          (ρ + σ) * (|s j| + chunkLiftSum16 cfg.a cfg.b va vb j) + α := by
        have := mul_le_mul_of_nonneg_left hsc hσ
        linarith
      have hSt : |psum vc (chunkSum16 va vb) j| ≤ psum |vc| (chunkLiftSum16 cfg.a cfg.b va vb) j := by
        unfold psum
        exact (abs_add_le _ _).trans (add_le_add le_rfl
          ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => hcA i)))
      rw [psum_succ, psum_succ]
      exact DotLoop.accum_step hρσ hα hSt (hcA j) e hs'
  have e := key (chunks16 K) le_rfl
  rw [hn, psum_chunks16, psum, sum_chunkLiftSum16] at e
  exact e

/-- The relative constant of the whole-dot bound from the chunk facts:
`(1 + 2^-13 + 2^-20)^⌈K/64⌉ − 1`. -/
def relLift16 (K : ℕ) : ℚ := (1 + (rhoChunk16 + sigmaChunk16)) ^ chunks16 K - 1

/-- The absolute constant of the whole-dot bound from the chunk facts:
`⌈K/64⌉·(1 + 2^-13 + 2^-20)^⌈K/64⌉·2^-143`. -/
def absLift16 (K : ℕ) : ℚ := chunks16 K * (1 + (rhoChunk16 + sigmaChunk16)) ^ chunks16 K * alphaChunk16

private theorem base_le16 : (1 : ℚ) + rhoChunk16 ≤ 1 + (rhoChunk16 + sigmaChunk16) := by
  have := sigmaChunk16_nonneg
  linarith

theorem relChunk16_nonneg (K : ℕ) : 0 ≤ relChunk16 K := by
  unfold relChunk16
  have : (1 : ℚ) ≤ (1 + rhoChunk16) ^ chunks16 K := one_le_pow₀ (by linarith [rhoChunk16_nonneg])
  linarith

theorem absChunk16_nonneg (K : ℕ) : 0 ≤ absChunk16 K := by
  unfold absChunk16 rhoChunk16 alphaChunk16; positivity

theorem relChunk16_le_relLift16 (K : ℕ) : relChunk16 K ≤ relLift16 K := by
  unfold relChunk16 relLift16
  have := pow_le_pow_left₀ (by linarith [rhoChunk16_nonneg]) base_le16 (chunks16 K)
  linarith

theorem absChunk16_le_absLift16 (K : ℕ) : absChunk16 K ≤ absLift16 K := by
  unfold absChunk16 absLift16
  have hp := pow_le_pow_left₀ (by linarith [rhoChunk16_nonneg]) base_le16 (chunks16 K)
  have := mul_le_mul_of_nonneg_left hp (Nat.cast_nonneg (α := ℚ) (chunks16 K))
  exact mul_le_mul_of_nonneg_right this alphaChunk16_nonneg

theorem relLift16_nonneg (K : ℕ) : 0 ≤ relLift16 K :=
  (relChunk16_nonneg K).trans (relChunk16_le_relLift16 K)

/-- **A whole 16-bit `tl.dot` from `DotF16Chunk` or `DotBF16Chunk`**: `DotLiftAcc` with
`relLift16`, `absLift16`. The room on lifted magnitudes covers the facts' room
(`relChunk16 ≤ relLift16`, `absChunk16 ≤ absLift16`, `|a·b| ≤ liftMag a · liftMag b`). -/
theorem Dot16BitChunk.liftAcc {P : Platform} {da db : DType} (h : Dot16BitChunk P da db)
    (p : InputPrecision) : DotLiftAcc P ⟨da, db, .f32, p, 0⟩ relLift16 absLift16 := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  dsimp only at hroom ⊢
  have hT : ∑ k, |va k * vb k| ≤ ∑ k, liftMag da (va k) * liftMag db (vb k) :=
    Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _
  have hT0 : 0 ≤ |vc| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hr := relChunk16_le_relLift16 K
  have ha := absChunk16_le_absLift16 K
  have hm : relChunk16 K * (|vc| + ∑ k, |va k * vb k|) ≤
      relLift16 K * (|vc| + ∑ k, liftMag da (va k) * liftMag db (vb k)) :=
    mul_le_mul hr (by linarith) hT0 (relLift16_nonneg K)
  obtain ⟨vr, hvr, herr⟩ := DotChunk16Loss.liftErr (h p) rhoChunk16_nonneg sigmaChunk16_nonneg
    alphaChunk16_nonneg hd i n va vb vc hva hvb hvc (by dsimp only; linarith)
  refine ⟨vr, hvr, herr.trans (le_of_eq ?_)⟩
  unfold DotLoop.bound relLift16 absLift16
  ring

end Dot16Chunk

/-! ## Satisfiability -/

namespace Platform.ref

variable {rel abs : MathFn → TVA.Math.Float.Format → ℝ}
  {sp : (fn : MathFn) → (f : TVA.Math.Float.Format) → FloatV f → Option (FloatV f)}

/-- **`DotChunk16Loss` holds of `Platform.ref`** for a float accumulator, with nonnegative `rho`,
`kappa`, `rho'`, `sigma` at least one rounding's `u` and `alpha` at least its `η`: the partial
values are the exact partial sums, and the last step is the one rounding of the exact result. -/
theorem dotChunk16Loss {cfg : DotCfg} {g : TVA.Math.Float.Format} (hg : cfg.acc.format? = some g)
    {rho sigma alpha kappa rho' : ℚ} (hrho : 0 ≤ rho) (hkappa : 0 ≤ kappa) (hrho' : 0 ≤ rho')
    (hsigma : g.u ≤ sigma) (halpha : g.eta ≤ alpha) :
    Rules.Num.DotChunk16Loss (Platform.ref rel abs sp) cfg rho sigma alpha kappa rho' := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hu : 0 ≤ g.u := TVA.Math.Float.Format.u_nonneg
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
  have hqm : |vc + ∑ k, va k * vb k| ≤ |vc| + ∑ k, |va k * vb k| :=
    (abs_add_le _ _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs _ _))
  have hrel := Dot16Chunk.relChunk16_nonneg K
  have habs := Dot16Chunk.absChunk16_nonneg K
  have hqmax : |vc + ∑ k, va k * vb k| ≤ g.maxVal := by
    rw [← DType.bound_of_format? hg]
    have := mul_nonneg hrel hmag
    linarith
  obtain ⟨v, hv, herr⟩ := DType.val?_ofRat_float hg hqmax
  have hn1 : 1 ≤ chunks16 K := by unfold chunks16; omega
  set nC := chunks16 K with hnC
  let s : ℕ → ℚ := fun j => if j < nC then Dot8Chunk.psum vc (chunkSum16 va vb) j else v
  refine ⟨s, v, by rw [hri]; exact hv, ?_, ?_, fun j hj Mj hM _ => ?_⟩
  · show (if 0 < nC then _ else v) = vc
    rw [ite_eq_left (by omega : 0 < nC), Dot8Chunk.psum_zero]
  · show (if nC < nC then _ else v) = v
    rw [ite_eq_right (lt_irrefl nC)]
  · have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
    have hmin : 0 ≤ min (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj) :=
      le_min (mul_nonneg hrho hM0)
        (add_nonneg (mul_nonneg hkappa (Dot16Chunk.chunkAbs16_nonneg va vb j)) (mul_nonneg hrho' hM0))
    have hR : 0 ≤ min (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj) +
        sigma * |s j + chunkSum16 va vb j| + alpha := by
      have := mul_nonneg (hu.trans hsigma) (abs_nonneg (s j + chunkSum16 va vb j))
      have he : 0 ≤ g.eta := TVA.Math.Float.Format.eta_nonneg
      linarith
    have hsj : s j = Dot8Chunk.psum vc (chunkSum16 va vb) j := ite_eq_left hj
    by_cases hj1 : j + 1 < nC
    · have hs1 : s (j + 1) = Dot8Chunk.psum vc (chunkSum16 va vb) (j + 1) := ite_eq_left hj1
      have h0 : s (j + 1) - (s j + chunkSum16 va vb j) = 0 := by
        rw [hs1, hsj, Dot8Chunk.psum_succ]; ring
      rw [h0, abs_zero]
      exact hR
    · have hjn : j + 1 = nC := by omega
      have hs1 : s (j + 1) = v := ite_eq_right hj1
      have hsum : s j + chunkSum16 va vb j = vc + ∑ k, va k * vb k := by
        rw [hsj, ← Dot8Chunk.psum_succ, hjn, hnC, Dot16Chunk.psum_chunks16]
      rw [hs1, hsum]
      have h1 : |vc + ∑ k, va k * vb k| * g.u ≤ sigma * |vc + ∑ k, va k * vb k| := by
        rw [mul_comm]; exact mul_le_mul_of_nonneg_right hsigma (abs_nonneg _)
      linarith

end Platform.ref

namespace Platform.sm90ref

/-- The 16-bit chunk facts hold of `Platform.sm90ref` (one `float32` rounding of the exact result). -/
theorem dot16BitChunk (a b : DType) : Dot16BitChunk Platform.sm90ref a b := fun _ =>
  Platform.ref.dotChunk16Loss (g := Format.binary32) rfl Dot16Chunk.rhoChunk16_nonneg
    Dot16Chunk.kappaChunk16_nonneg Dot16Chunk.rhoLossChunk16_nonneg
    (by rw [Format.binary32_u]; unfold sigmaChunk16; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (by rw [Format.binary32_eta]; unfold alphaChunk16; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))

theorem dotF16Chunk : DotF16Chunk Platform.sm90ref := dot16BitChunk _ _
theorem dotBF16Chunk : DotBF16Chunk Platform.sm90ref := dot16BitChunk _ _

end Platform.sm90ref

end TVA
