import TVA.Proof.Dot.Chunk8

/-!
# A 16-bit `tl.dot`, chunks of 64

A `tl.dot` of `float16` or `bfloat16` operands into `float32` adds its `K` products in chunks of
64 consecutive `k` (`Device.chunk16`), each to the accumulator as it stands. The device facts
`Device.DotF16Chunk`, `Device.DotBF16Chunk` (`Device.Dot16BitChunk`, shape
`Device.DotChunk16Loss`) bound each chunk's step by the smaller of two charges. This file composes
the steps (`TVA.Proof.Dot.Chunk`, the same recursions as the fp8 chunks) into a bound on one
`tl.dot` from its exact inputs. The whole-dot facts `Device.DotF16`, `Device.DotBF16`
(`TVA.Proof.Dot.Lift`) bound the same dot more coarsely; these are for claims whose tolerance
follows the chunks.

* `Device.DotChunk16Loss.err`, `Device.Dot16BitChunk.err`: the result is within
  `Dot8Chunk.errM ρ σ α κ ρ' vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K)` of
  `vc + Σ va·vb`, for any bounds `m j` on each chunk's lifted products (`chunkLift16` will do);
  at most `errB` (`Dot8Chunk.errM_le_errB`) and `errL` (`Dot8Chunk.errM_le_errL`, growing by
  `1 + 2^-22 + 2^-20` per chunk, `Dot8Chunk.errL_le`).
* `Device.DotChunk16Loss.errB`: the first arm alone, within `Dot8Chunk.errB`.
* `Device.DotChunk16Loss.liftErr`, `Device.Dot16BitChunk.liftAcc`: the whole dot is a
  `Device.DotLiftAcc` fact with `Dot16Chunk.relLift16 K = (1 + 2^-13 + 2^-20)^⌈K/64⌉ − 1` and
  `Dot16Chunk.absLift16 K = ⌈K/64⌉·(1 + 2^-13 + 2^-20)^⌈K/64⌉·2^-143`.
* The pieces: `psum_chunks16`, `sum_chunkAbs16`, `chunkAbs16_nonneg`, `abs_chunkSum16_le`,
  `chunkLift16`, `chunkLiftSum16`, and the constants' signs.
-/

namespace TVA

open Device Dot8Chunk

namespace Dot16Chunk

/-- The partial sum over all chunks of a dot is `acc + Σ a·b`. -/
theorem psum_chunks16 {K : ℕ} (v : ℚ) (va vb : Fin K → ℚ) :
    psum v (chunkSum16 va vb) (chunks16 K) = v + ∑ k, va k * vb k := by
  rw [psum, sum_chunkSum16]

/-- The chunks' magnitudes add up to `Σ|a·b|`. -/
theorem sum_chunkAbs16 {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks16 K), chunkAbs16 va vb j = ∑ k, |va k * vb k| := by
  unfold chunkAbs16 chunk16
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 64)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks16 k)) _

/-- A chunk's magnitude is nonnegative. -/
theorem chunkAbs16_nonneg {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : 0 ≤ chunkAbs16 va vb j :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- A chunk's sum is at most its magnitude. -/
theorem abs_chunkSum16_le {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    |chunkSum16 va vb j| ≤ chunkAbs16 va vb j :=
  Finset.abs_sum_le_sum_abs _ _

/-- The largest lifted product of chunk `j` (`0` for an empty chunk): a choice of the bounds
`m j` of `DotChunk16Loss.err`. -/
def chunkLift16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  (chunk16 K j).fold max 0 fun k => liftMag da (va k) * liftMag db (vb k)

/-- Each lifted product of a chunk is at most `chunkLift16`. -/
theorem le_chunkLift16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) {j : ℕ} {k : Fin K}
    (hk : k ∈ chunk16 K j) :
    liftMag da (va k) * liftMag db (vb k) ≤ chunkLift16 da db va vb j := by
  unfold chunkLift16
  exact (Finset.le_fold_max _).mpr (Or.inr ⟨k, hk, le_rfl⟩)

/-- `chunkLift16` is nonnegative. -/
theorem chunkLift16_nonneg (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    0 ≤ chunkLift16 da db va vb j := by
  unfold chunkLift16
  exact (Finset.le_fold_max _).mpr (Or.inl le_rfl)

/-- `2^-13 ≥ 0`. -/
theorem rhoChunk16_nonneg : 0 ≤ rhoChunk16 := by unfold rhoChunk16; positivity
/-- `2^-20 ≥ 0`. -/
theorem sigmaChunk16_nonneg : 0 ≤ sigmaChunk16 := by unfold sigmaChunk16; positivity
/-- `2^-143 ≥ 0`. -/
theorem alphaChunk16_nonneg : 0 ≤ alphaChunk16 := by unfold alphaChunk16; positivity
/-- `2 ≥ 0`. -/
theorem kappaChunk16_nonneg : 0 ≤ kappaChunk16 := by unfold kappaChunk16; norm_num
/-- `2^-22 ≥ 0`. -/
theorem rhoLossChunk16_nonneg : 0 ≤ rhoLossChunk16 := by unfold rhoLossChunk16; positivity

/-- `relChunk16 K ≥ 0`. -/
theorem relChunk16_nonneg (K : ℕ) : 0 ≤ relChunk16 K := by
  unfold relChunk16
  have : (1 : ℚ) ≤ (1 + rhoChunk16) ^ chunks16 K := one_le_pow₀ (by linarith [rhoChunk16_nonneg])
  linarith

/-- `absChunk16 K ≥ 0`. -/
theorem absChunk16_nonneg (K : ℕ) : 0 ≤ absChunk16 K := by
  unfold absChunk16 rhoChunk16 alphaChunk16; positivity

end Dot16Chunk

open Dot16Chunk

/-! ## One 16-bit `tl.dot` from its exact inputs -/

/-- **A 16-bit `tl.dot` from its exact inputs.** Under `DotChunk16Loss`, for any bounds `m j` on
the lifted products of each chunk (`chunkLift16` will do), the result is within
`errM ρ σ α κ ρ' vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K)` of `vc + Σ va·vb`. -/
theorem Device.DotChunk16Loss.err {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunk16Loss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hρ' : 0 ≤ ρ') {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| +
        (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤ cfg.acc.bound)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks16 K, ∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errM ρ σ α κ ρ' vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_errM (s := s) (c := chunkSum16 va vb) (a := chunkAbs16 va vb) (m := m)
    (n := chunks16 K) hρ hσ hρ'
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj)
    (chunks16 K) le_rfl
  rw [hn, h0, psum_chunks16] at e
  exact e

/-- **A 16-bit `tl.dot` from its exact inputs, first arm alone**: the result is within
`errB ρ σ α vc (chunkSum16 va vb) m (chunks16 K)` of `vc + Σ va·vb`. -/
theorem Device.DotChunk16Loss.errB {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunk16Loss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| +
        (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤ cfg.acc.bound)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks16 K, ∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤ Dot8Chunk.errB ρ σ α vc (chunkSum16 va vb) m (chunks16 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h.step_left a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_err (s := s) (c := chunkSum16 va vb) (m := m) (n := chunks16 K) hρ hσ
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj)
    (chunks16 K) le_rfl
  rw [hn, h0, psum_chunks16] at e
  exact e

/-- **`DotF16Chunk` and `DotBF16Chunk`, from the exact inputs**: the result is within
`errM 2^-13 2^-20 2^-143 2 2^-22 vc (chunkSum16 va vb) (chunkAbs16 va vb) m (chunks16 K)` of
`vc + Σ va·vb`. -/
theorem Device.Dot16BitChunk.err {P : Platform} {da db : DType} (h : Dot16BitChunk P da db)
    (p : InputPrecision) {M N K : ℕ} {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val}
    {acc r : Tensor [M, N] F32} (hd : P.dot ⟨da, db, .f32, p, 0⟩ a b acc r) (i : Fin M)
    (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ) (hva : ∀ k, da.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k)) (hvc : DType.val? .f32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| +
        (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤ DType.bound .f32)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks16 K, ∀ k ∈ chunk16 K j, liftMag da (va k) * liftMag db (vb k) ≤ m j) :
    ∃ vr, DType.val? .f32 (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errM rhoChunk16 sigmaChunk16 alphaChunk16 kappaChunk16 rhoLossChunk16 vc (chunkSum16 va vb)
          (chunkAbs16 va vb) m (chunks16 K) :=
  (h p).err rhoChunk16_nonneg sigmaChunk16_nonneg rhoLossChunk16_nonneg hd i n va vb vc hva hvb
    hvc hroom m hm

/-! ## The whole dot, relative to lifted magnitudes -/

namespace Dot16Chunk

/-- The lifted magnitude of chunk `j`:
`Σ_{k ∈ chunk16 K j} liftMag da (va k) · liftMag db (vb k)`. -/
def chunkLiftSum16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  ∑ k ∈ chunk16 K j, liftMag da (va k) * liftMag db (vb k)

/-- The chunks' lifted magnitudes add up to `Σ liftMag a·liftMag b`. -/
theorem sum_chunkLiftSum16 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks16 K), chunkLiftSum16 da db va vb j =
      ∑ k, liftMag da (va k) * liftMag db (vb k) := by
  unfold chunkLiftSum16 chunk16
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 64)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks16 k)) _

/-- The relative constant of a whole 16-bit `tl.dot` from the chunk facts:
`(1 + 2^-13 + 2^-20)^⌈K/64⌉ − 1`. -/
def relLift16 (K : ℕ) : ℚ := (1 + (rhoChunk16 + sigmaChunk16)) ^ chunks16 K - 1

/-- The absolute constant of a whole 16-bit `tl.dot` from the chunk facts:
`⌈K/64⌉·(1 + 2^-13 + 2^-20)^⌈K/64⌉·2^-143`. -/
def absLift16 (K : ℕ) : ℚ :=
  chunks16 K * (1 + (rhoChunk16 + sigmaChunk16)) ^ chunks16 K * alphaChunk16

private theorem base_le16 : (1 : ℚ) + rhoChunk16 ≤ 1 + (rhoChunk16 + sigmaChunk16) := by
  have := sigmaChunk16_nonneg
  linarith

/-- The facts' room constant is at most the whole-dot one: `relChunk16 ≤ relLift16`. -/
theorem relChunk16_le_relLift16 (K : ℕ) : relChunk16 K ≤ relLift16 K := by
  unfold relChunk16 relLift16
  have := pow_le_pow_left₀ (by linarith [rhoChunk16_nonneg]) base_le16 (chunks16 K)
  linarith

/-- `absChunk16 ≤ absLift16`. -/
theorem absChunk16_le_absLift16 (K : ℕ) : absChunk16 K ≤ absLift16 K := by
  unfold absChunk16 absLift16
  have hp := pow_le_pow_left₀ (by linarith [rhoChunk16_nonneg]) base_le16 (chunks16 K)
  have := mul_le_mul_of_nonneg_left hp (Nat.cast_nonneg (α := ℚ) (chunks16 K))
  exact mul_le_mul_of_nonneg_right this alphaChunk16_nonneg

/-- `relLift16 K ≥ 0`. -/
theorem relLift16_nonneg (K : ℕ) : 0 ≤ relLift16 K :=
  (relChunk16_nonneg K).trans (relChunk16_le_relLift16 K)

/-- `absLift16 K ≥ 0`. -/
theorem absLift16_nonneg (K : ℕ) : 0 ≤ absLift16 K :=
  (absChunk16_nonneg K).trans (absChunk16_le_absLift16 K)

end Dot16Chunk

/-- **A whole 16-bit `tl.dot` from the chunk steps.** Under `DotChunk16Loss` (its first arm), the
result is within `DotLoop.bound (ρ + σ) α ⌈K/64⌉ L` of `vc + Σ va·vb`, with
`L = |vc| + Σ liftMag a_k·liftMag b_k` (each chunk taken with `Mj = |s j| + ` its lifted
magnitude, composed by `DotLoop.accum_step`). Use `Dot16BitChunk.liftAcc` for the device's
constants. -/
theorem Device.DotChunk16Loss.liftErr {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunk16Loss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| +
        (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤ cfg.acc.bound) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        DotLoop.bound (ρ + σ) α (chunks16 K)
          (|vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k)) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h.step_left a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have hA0 : ∀ j, 0 ≤ chunkLiftSum16 cfg.a cfg.b va vb j := fun j =>
    Finset.sum_nonneg fun k _ => mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _)
  have hcA : ∀ j, |chunkSum16 va vb j| ≤ chunkLiftSum16 cfg.a cfg.b va vb j := fun j =>
    (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _)
  have hρσ : 0 ≤ ρ + σ := add_nonneg hρ hσ
  have key : ∀ j ≤ chunks16 K, |s j - psum vc (chunkSum16 va vb) j| ≤
      DotLoop.bound (ρ + σ) α j (psum |vc| (chunkLiftSum16 cfg.a cfg.b va vb) j) := by
    intro j
    induction j with
    | zero => intro _; simp [h0, DotLoop.bound]
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
      have hSt : |psum vc (chunkSum16 va vb) j| ≤
          psum |vc| (chunkLiftSum16 cfg.a cfg.b va vb) j := by
        unfold psum
        exact (abs_add_le _ _).trans (add_le_add le_rfl
          ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => hcA i)))
      rw [psum_succ, psum_succ]
      exact DotLoop.accum_step hρσ hα hSt (hcA j) e hs'
  have e := key (chunks16 K) le_rfl
  rw [hn, psum_chunks16, psum, sum_chunkLiftSum16] at e
  exact e

/-- **A whole 16-bit `tl.dot` from `DotF16Chunk` or `DotBF16Chunk`**: `DotLiftAcc` with
`relLift16`, `absLift16`. Use it when the chunk facts give a tighter whole-dot bound than
`DotF16`'s `relDot16` (large `K`). -/
theorem Device.Dot16BitChunk.liftAcc {P : Platform} {da db : DType} (h : Dot16BitChunk P da db)
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
  obtain ⟨vr, hvr, herr⟩ := (h p).liftErr rhoChunk16_nonneg sigmaChunk16_nonneg
    alphaChunk16_nonneg hd i n va vb vc hva hvb hvc (by dsimp only; linarith)
  refine ⟨vr, hvr, herr.trans (le_of_eq ?_)⟩
  unfold DotLoop.bound relLift16 absLift16
  ring

end TVA
