import TVA.Proof.Dot.Chunk

/-!
# An fp8 `tl.dot`, chunks of 32

A `tl.dot` of fp8 operands (`float8e4nv`, `float8e5`) into `float32` adds its `K` products in
chunks of 32 consecutive `k` (`Device.chunk8`), each to the accumulator as it stands. The device
facts `Device.DotE4m3ChunkLoss`, … (`Device.DotF8ChunkLoss`, shape `Device.DotChunkLoss`) bound
each chunk's step, and their first arm `Device.DotF8Chunk` (shape `Device.DotChunkAcc`) follows
(`Device.DotE4m3Chunk`, …). This file composes the steps (`TVA.Proof.Dot.Chunk`) into a bound on
one `tl.dot` from its exact inputs.

* `Device.DotChunkAcc.err`, `Device.DotF8Chunk.err`: the result is within
  `Dot8Chunk.errB ρ σ α vc (chunkSum8 va vb) m (chunks8 K)` of `vc + Σ va·vb`, for any bounds
  `m j` on each chunk's lifted products (`Dot8Chunk.chunkLift8` will do).
* `Device.DotChunkLoss.err`, `Device.DotF8ChunkLoss.err`: within
  `Dot8Chunk.errM … (chunkSum8 va vb) (chunkAbs8 va vb) m (chunks8 K)` (each chunk the smaller of
  two charges; at most `errB` by `Dot8Chunk.errM_le_errB`).
* `Device.DotChunkAcc.liftErr`, `Device.DotF8Chunk.liftAcc`: the whole dot is a
  `Device.DotLiftAcc` fact with `Dot8Chunk.relLift8 K = (1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1` and
  `Dot8Chunk.absLift8 K = ⌈K/32⌉·(1 + 2^-7 + 2^-12)^⌈K/32⌉·2^-126`, so an fp8 GEMM loop
  `acc += tl.dot(a, b)` uses `DotLift.dotAdd_step` and `DotLoop.accum` as a 16-bit one does.
* The pieces: `psum_chunks8`, `sum_chunkAbs8`, `chunkAbs8_nonneg`, `abs_chunkSum8_le`,
  `chunkLift8` (`le_chunkLift8`, `chunkLift8_nonneg`), `chunkLiftSum8` (`sum_chunkLiftSum8`), and
  the constants' signs (`rhoChunk8_nonneg`, …, `relDot8_nonneg`, `absDot8_nonneg`).
-/

namespace TVA

open Device

namespace Dot8Chunk

/-- The partial sum over all chunks of a dot is `acc + Σ a·b`. -/
theorem psum_chunks8 {K : ℕ} (v : ℚ) (va vb : Fin K → ℚ) :
    psum v (chunkSum8 va vb) (chunks8 K) = v + ∑ k, va k * vb k := by
  rw [psum, sum_chunkSum8]

/-- A chunk's magnitude is nonnegative. -/
theorem chunkAbs8_nonneg {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : 0 ≤ chunkAbs8 va vb j :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- A chunk's sum is at most its magnitude. -/
theorem abs_chunkSum8_le {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    |chunkSum8 va vb j| ≤ chunkAbs8 va vb j :=
  Finset.abs_sum_le_sum_abs _ _

/-- The chunks' magnitudes add up to `Σ|a·b|`. -/
theorem sum_chunkAbs8 {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks8 K), chunkAbs8 va vb j = ∑ k, |va k * vb k| := by
  unfold chunkAbs8 chunk8
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 32)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks8 k)) _

/-- The largest lifted product `liftMag da (va k) · liftMag db (vb k)` of chunk `j` (`0` for an
empty chunk): a choice of the bounds `m j` of `DotChunkAcc.err` and `DotChunkLoss.err`. -/
def chunkLift8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  (chunk8 K j).fold max 0 fun k => liftMag da (va k) * liftMag db (vb k)

/-- Each lifted product of a chunk is at most `chunkLift8`. -/
theorem le_chunkLift8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) {j : ℕ} {k : Fin K}
    (hk : k ∈ chunk8 K j) : liftMag da (va k) * liftMag db (vb k) ≤ chunkLift8 da db va vb j := by
  unfold chunkLift8
  exact (Finset.le_fold_max _).mpr (Or.inr ⟨k, hk, le_rfl⟩)

/-- `chunkLift8` is nonnegative. -/
theorem chunkLift8_nonneg (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    0 ≤ chunkLift8 da db va vb j := by
  unfold chunkLift8
  exact (Finset.le_fold_max _).mpr (Or.inl le_rfl)

/-- `2^-7 ≥ 0`. -/
theorem rhoChunk8_nonneg : 0 ≤ rhoChunk8 := by unfold rhoChunk8; positivity
/-- `2^-12 ≥ 0`. -/
theorem sigmaChunk8_nonneg : 0 ≤ sigmaChunk8 := by unfold sigmaChunk8; positivity
/-- `2^-126 ≥ 0`. -/
theorem alphaChunk8_nonneg : 0 ≤ alphaChunk8 := by unfold alphaChunk8; positivity
/-- `2 ≥ 0`. -/
theorem kappaChunk8_nonneg : 0 ≤ kappaChunk8 := by unfold kappaChunk8; norm_num
/-- `2^-12 ≥ 0`. -/
theorem rhoLossChunk8_nonneg : 0 ≤ rhoLossChunk8 := by unfold rhoLossChunk8; positivity

/-- `relDot8 K ≥ 0`. -/
theorem relDot8_nonneg (K : ℕ) : 0 ≤ relDot8 K := by
  unfold relDot8
  have h7 : (0 : ℚ) ≤ 2 ^ (-7 : ℤ) := by positivity
  have : (1 : ℚ) ≤ (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K := one_le_pow₀ (by linarith)
  linarith

/-- `absDot8 K ≥ 0`. -/
theorem absDot8_nonneg (K : ℕ) : 0 ≤ absDot8 K := by unfold absDot8; positivity

end Dot8Chunk

open Dot8Chunk

/-! ## One fp8 `tl.dot` from its exact inputs -/

/-- **An fp8 `tl.dot` from its exact inputs, first arm.** Under `DotChunkAcc`, for any bounds
`m j` on the lifted products of each chunk (`chunkLift8` will do), the result is within
`errB ρ σ α vc (chunkSum8 va vb) m (chunks8 K)` of `vc + Σ va·vb`. -/
theorem Device.DotChunkAcc.err {P : Platform} {cfg : DotCfg} {ρ σ α : ℚ}
    (h : DotChunkAcc P cfg ρ σ α) (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      cfg.acc.bound)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks8 K, ∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤ errB ρ σ α vc (chunkSum8 va vb) m (chunks8 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_err (s := s) (c := chunkSum8 va vb) (m := m) (n := chunks8 K) hρ hσ
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj)
    (chunks8 K) le_rfl
  rw [hn, h0, psum_chunks8] at e
  exact e

/-- **`DotE5m2Chunk` and the other first-arm fp8 facts, from the exact inputs**: the result is
within `errB 2^-7 2^-12 2^-126 vc (chunkSum8 va vb) m (chunks8 K)` of `vc + Σ va·vb`. -/
theorem Device.DotF8Chunk.err {P : Platform} {da db : DType} (h : DotF8Chunk P da db)
    (p : InputPrecision) {M N K : ℕ} {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val}
    {acc r : Tensor [M, N] F32} (hd : P.dot ⟨da, db, .f32, p, 2 ^ 30⟩ a b acc r) (i : Fin M)
    (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ) (hva : ∀ k, da.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k)) (hvc : DType.val? .f32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      DType.bound .f32)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks8 K, ∀ k ∈ chunk8 K j, liftMag da (va k) * liftMag db (vb k) ≤ m j) :
    ∃ vr, DType.val? .f32 (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errB rhoChunk8 sigmaChunk8 alphaChunk8 vc (chunkSum8 va vb) m (chunks8 K) :=
  (h p).err rhoChunk8_nonneg sigmaChunk8_nonneg hd i n va vb vc hva hvb hvc hroom m hm

/-- **An fp8 `tl.dot` from its exact inputs, each chunk within the smaller of two bounds.** Under
`DotChunkLoss`, for any bounds `m j` on the lifted products of each chunk (`chunkLift8` will do),
the result is within `errM ρ σ α κ ρ' vc (chunkSum8 va vb) (chunkAbs8 va vb) m (chunks8 K)` of
`vc + Σ va·vb`. -/
theorem Device.DotChunkLoss.err {P : Platform} {cfg : DotCfg} {ρ σ α κ ρ' : ℚ}
    (h : DotChunkLoss P cfg ρ σ α κ ρ') (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hρ' : 0 ≤ ρ') {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      cfg.acc.bound)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks8 K, ∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errM ρ σ α κ ρ' vc (chunkSum8 va vb) (chunkAbs8 va vb) m (chunks8 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_errM (s := s) (c := chunkSum8 va vb) (a := chunkAbs8 va vb) (m := m)
    (n := chunks8 K) hρ hσ hρ'
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj)
    (chunks8 K) le_rfl
  rw [hn, h0, psum_chunks8] at e
  exact e

/-- **`DotE5m2ChunkLoss` and the other fp8 facts, from the exact inputs**: the result is within
`errM 2^-7 2^-12 2^-126 2 2^-12 vc (chunkSum8 va vb) (chunkAbs8 va vb) m (chunks8 K)` of
`vc + Σ va·vb`. This is at most `DotF8Chunk.err`'s `errB` (`errM_le_errB`), and grows by
`1 + 2^-11` per chunk where the second arm is the smaller (`errM_le_errL`, `errL_le`). -/
theorem Device.DotF8ChunkLoss.err {P : Platform} {da db : DType} (h : DotF8ChunkLoss P da db)
    (p : InputPrecision) {M N K : ℕ} {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val}
    {acc r : Tensor [M, N] F32} (hd : P.dot ⟨da, db, .f32, p, 2 ^ 30⟩ a b acc r) (i : Fin M)
    (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ) (hva : ∀ k, da.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k)) (hvc : DType.val? .f32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      DType.bound .f32)
    (m : ℕ → ℚ)
    (hm : ∀ j < chunks8 K, ∀ k ∈ chunk8 K j, liftMag da (va k) * liftMag db (vb k) ≤ m j) :
    ∃ vr, DType.val? .f32 (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errM rhoChunk8 sigmaChunk8 alphaChunk8 kappaChunk8 rhoLossChunk8 vc (chunkSum8 va vb)
          (chunkAbs8 va vb) m (chunks8 K) :=
  (h p).err rhoChunk8_nonneg sigmaChunk8_nonneg rhoLossChunk8_nonneg hd i n va vb vc hva hvb hvc
    hroom m hm

/-! ## The whole dot, relative to lifted magnitudes -/

namespace Dot8Chunk

/-- The lifted magnitude of chunk `j`: `Σ_{k ∈ chunk8 K j} liftMag da (va k) · liftMag db (vb k)`. -/
def chunkLiftSum8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  ∑ k ∈ chunk8 K j, liftMag da (va k) * liftMag db (vb k)

/-- The chunks' lifted magnitudes add up to `Σ liftMag a·liftMag b`. -/
theorem sum_chunkLiftSum8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks8 K), chunkLiftSum8 da db va vb j =
      ∑ k, liftMag da (va k) * liftMag db (vb k) := by
  unfold chunkLiftSum8 chunk8
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 32)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks8 k)) _

/-- The relative constant of a whole fp8 `tl.dot` from the chunk facts:
`(1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1`. -/
def relLift8 (K : ℕ) : ℚ := (1 + (rhoChunk8 + sigmaChunk8)) ^ chunks8 K - 1

/-- The absolute constant of a whole fp8 `tl.dot` from the chunk facts:
`⌈K/32⌉·(1 + 2^-7 + 2^-12)^⌈K/32⌉·2^-126`. -/
def absLift8 (K : ℕ) : ℚ := chunks8 K * (1 + (rhoChunk8 + sigmaChunk8)) ^ chunks8 K * alphaChunk8

private theorem base_le : (1 : ℚ) + 2 ^ (-7 : ℤ) ≤ 1 + (rhoChunk8 + sigmaChunk8) := by
  unfold rhoChunk8 sigmaChunk8
  have : (0 : ℚ) ≤ 2 ^ (-12 : ℤ) := by positivity
  linarith

/-- The facts' room constant is at most the whole-dot one: `relDot8 ≤ relLift8`. -/
theorem relDot8_le_relLift8 (K : ℕ) : relDot8 K ≤ relLift8 K := by
  unfold relDot8 relLift8
  have := pow_le_pow_left₀ (by positivity) base_le (chunks8 K)
  linarith

/-- `absDot8 ≤ absLift8`. -/
theorem absDot8_le_absLift8 (K : ℕ) : absDot8 K ≤ absLift8 K := by
  unfold absDot8 absLift8 alphaChunk8
  have hp := pow_le_pow_left₀ (by positivity) base_le (chunks8 K)
  have := mul_le_mul_of_nonneg_left hp (Nat.cast_nonneg (α := ℚ) (chunks8 K))
  exact mul_le_mul_of_nonneg_right this (by positivity)

/-- `relLift8 K ≥ 0`. -/
theorem relLift8_nonneg (K : ℕ) : 0 ≤ relLift8 K :=
  (relDot8_nonneg K).trans (relDot8_le_relLift8 K)

/-- `absLift8 K ≥ 0`. -/
theorem absLift8_nonneg (K : ℕ) : 0 ≤ absLift8 K := by
  unfold absLift8 rhoChunk8 sigmaChunk8 alphaChunk8; positivity

end Dot8Chunk

/-- **A whole fp8 `tl.dot` from the chunk steps.** Under `DotChunkAcc` with constants `ρ σ α`,
the result is within `DotLoop.bound (ρ + σ) α ⌈K/32⌉ L` of `vc + Σ va·vb`, with
`L = |vc| + Σ liftMag a_k·liftMag b_k`: each chunk, taken with `Mj = |s j| + ` its lifted
magnitude, adds at most `(ρ + σ)·Mj + α`, and `DotLoop.accum_step` composes the chunks. Use
`DotF8Chunk.liftAcc` for the device's constants. -/
theorem Device.DotChunkAcc.liftErr {P : Platform} {cfg : DotCfg} {ρ σ α : ℚ}
    (h : DotChunkAcc P cfg ρ σ α) (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) {M N K : ℕ}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hd : P.dot cfg a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      cfg.acc.bound) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        DotLoop.bound (ρ + σ) α (chunks8 K)
          (|vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k)) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have hA0 : ∀ j, 0 ≤ chunkLiftSum8 cfg.a cfg.b va vb j := fun j =>
    Finset.sum_nonneg fun k _ => mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _)
  have hcA : ∀ j, |chunkSum8 va vb j| ≤ chunkLiftSum8 cfg.a cfg.b va vb j := fun j =>
    (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _)
  have hρσ : 0 ≤ ρ + σ := add_nonneg hρ hσ
  have key : ∀ j ≤ chunks8 K, |s j - psum vc (chunkSum8 va vb) j| ≤
      DotLoop.bound (ρ + σ) α j (psum |vc| (chunkLiftSum8 cfg.a cfg.b va vb) j) := by
    intro j
    induction j with
    | zero => intro _; simp [h0, DotLoop.bound]
    | succ j ih =>
      intro hj
      have e := ih (by omega)
      have hk : ∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤
          |s j| + chunkLiftSum8 cfg.a cfg.b va vb j := fun k hk =>
        (Finset.single_le_sum (f := fun k => liftMag cfg.a (va k) * liftMag cfg.b (vb k))
          (fun k _ => mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _)) hk).trans
          (le_add_of_nonneg_left (abs_nonneg _))
      have hs := hstep j (by omega) (|s j| + chunkLiftSum8 cfg.a cfg.b va vb j)
        (le_add_of_nonneg_right (hA0 j)) hk
      have hsc : |s j + chunkSum8 va vb j| ≤ |s j| + chunkLiftSum8 cfg.a cfg.b va vb j :=
        (abs_add_le _ _).trans (by linarith [hcA j])
      have hs' : |s (j + 1) - (s j + chunkSum8 va vb j)| ≤
          (ρ + σ) * (|s j| + chunkLiftSum8 cfg.a cfg.b va vb j) + α := by
        have := mul_le_mul_of_nonneg_left hsc hσ
        linarith
      have hSt : |psum vc (chunkSum8 va vb) j| ≤
          psum |vc| (chunkLiftSum8 cfg.a cfg.b va vb) j := by
        unfold psum
        exact (abs_add_le _ _).trans (add_le_add le_rfl
          ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => hcA i)))
      rw [psum_succ, psum_succ]
      exact DotLoop.accum_step hρσ hα hSt (hcA j) e hs'
  have e := key (chunks8 K) le_rfl
  rw [hn, psum_chunks8, psum, sum_chunkLiftSum8] at e
  exact e

/-- **A whole fp8 `tl.dot` from `DotE5m2Chunk` and the other first-arm facts**: `DotLiftAcc`
with `relLift8`, `absLift8`. Use it to treat an fp8 GEMM loop as a 16-bit one
(`DotLift.dotAdd_step`, `DotLoop.accum`). -/
theorem Device.DotF8Chunk.liftAcc {P : Platform} {da db : DType} (h : DotF8Chunk P da db)
    (p : InputPrecision) : DotLiftAcc P ⟨da, db, .f32, p, 2 ^ 30⟩ relLift8 absLift8 := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  dsimp only at hroom ⊢
  have hT : ∑ k, |va k * vb k| ≤ ∑ k, liftMag da (va k) * liftMag db (vb k) :=
    Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _
  have hT0 : 0 ≤ |vc| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hr := relDot8_le_relLift8 K
  have ha := absDot8_le_absLift8 K
  have hm : relDot8 K * (|vc| + ∑ k, |va k * vb k|) ≤
      relLift8 K * (|vc| + ∑ k, liftMag da (va k) * liftMag db (vb k)) :=
    mul_le_mul hr (by linarith) hT0 (relLift8_nonneg K)
  obtain ⟨vr, hvr, herr⟩ := (h p).liftErr rhoChunk8_nonneg sigmaChunk8_nonneg
    alphaChunk8_nonneg hd i n va vb vc hva hvb hvc (by dsimp only; linarith)
  refine ⟨vr, hvr, herr.trans (le_of_eq ?_)⟩
  unfold DotLoop.bound relLift8 absLift8
  ring

end TVA
