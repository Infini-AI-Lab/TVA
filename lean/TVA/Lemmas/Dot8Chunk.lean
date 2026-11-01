import TVA.Rules.Num.Dot8Chunk
import TVA.Lemmas.Sm90
import TVA.Lemmas.DotLoop

/-!
# Using the fp8 chunk facts (`DotE5m2Chunk`, …)

Page: `docs/rules/num.md` ("`tl.dot` of fp8, chunk by chunk"). The facts (`TVA.Rules.Num.Dot8Chunk`)
give, per output element, partial values `s 0 = acc, …, s n = result` (`n = chunks8 K`) and a bound
on each chunk's step. These lemmas compose the steps into a bound from the exact inputs alone, and
show the facts satisfiable.

* `Dot8Chunk.sum_chunkSum8`: the chunks' exact sums add up to `Σ a·b`.
* `Dot8Chunk.psum v c j = v + Σ_{i<j} c i`: the exact partial sums.
* `Dot8Chunk.errB ρ σ α v c m j`: the composed bound after `j` chunks, from the exact partial sums
  and bounds `m i` on each chunk's lifted products:
  `errB (j+1) = (1+ρ+σ)·errB j + ρ·max |psum j| (m j) + σ·|psum (j+1)| + α`, `errB 0 = 0`.
* `Dot8Chunk.chain_err`: any chain of steps bounded as the facts bound them stays within `errB` of
  the exact partial sums.
* `Dot8Chunk.errB_le`: `errB j ≤ (1+ρ+σ)^j · Σ_{i<j} (ρ·max |psum i| (m i) + σ·|psum (i+1)| + α)`.
* `Dot8Chunk.chunkLift8`: the largest lifted product of a chunk (`le_chunkLift8`, `chunkLift8_nonneg`).
* `Dot8Chunk.DotChunkAcc.err`, `Dot8Chunk.DotF8Chunk.err`: an fp8 `tl.dot`'s result is within
  `errB` (at `n = chunks8 K`) of `acc + Σ a·b`, for any bounds `m j` on the chunks' lifted
  products (`chunkLift8` will do).
* `Dot8Chunk.DotF8Chunk.liftAcc`: the chunk facts give `Rules.Num.DotLiftAcc` (a whole `tl.dot`
  within `rel K·L + abs K` of `acc + Σ a·b`, `L = |acc| + Σ liftMag a_k · liftMag b_k`, with room on
  `L`; round 24 called this shape `Dot8Chunk.LiftAcc`) with
  `relLift8 K = (1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1` and
  `absLift8 K = ⌈K/32⌉·(1 + 2^-7 + 2^-12)^⌈K/32⌉·2^-126` (`DotChunkAcc.liftErr` for any constants,
  as `DotLoop.bound`). This replaces the withdrawn `DotE4m3`, … (library issue L125).
* For any dtype (the 16-bit facts' `liftMag` too, `TVA.Lemmas.DotLift`): for zero and normal
  operands `liftMag` is `|q|` (`liftMag_of_le`, `liftMag_zero`); in general
  `|q| ≤ liftMag d q ≤ |q| + minNormal8 d` (`abs_le_liftMag`, `liftMag_le`), and
  `|x·y| ≤ liftMag da x · liftMag db y` (`abs_mul_le_liftMag`).
* `Platform.ref.dotChunkAcc`, `Platform.sm90ref.dotE4m3Chunk`, …: the facts hold of the reference
  platforms (one rounding of the exact result), so a bundle citing them proves `Claim.Consistent`
  with `Platform.sm90ref`, as with the other sm_90 facts.
-/
namespace TVA

open TVA.Math.Float Rules.Num

namespace Dot8Chunk

/-- Every `k < K` lies in a chunk `j < chunks8 K`. -/
theorem div_lt_chunks8 {K : ℕ} (k : Fin K) : k.val / 32 < chunks8 K := by
  unfold chunks8; have := k.isLt; omega

/-- The chunks' exact sums add up to the whole sum. -/
theorem sum_chunkSum8 {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks8 K), chunkSum8 va vb j = ∑ k, va k * vb k := by
  unfold chunkSum8 chunk8
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 32)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks8 k)) _

/-- The exact partial sums from `v`: `v + Σ_{i<j} c i`. -/
def psum (v : ℚ) (c : ℕ → ℚ) (j : ℕ) : ℚ := v + ∑ i ∈ Finset.range j, c i

theorem psum_zero (v : ℚ) (c : ℕ → ℚ) : psum v c 0 = v := by simp [psum]

theorem psum_succ (v : ℚ) (c : ℕ → ℚ) (j : ℕ) : psum v c (j + 1) = psum v c j + c j := by
  simp [psum, Finset.sum_range_succ, add_assoc]

/-- The partial sum over all chunks of a dot is `acc + Σ a·b`. -/
theorem psum_chunks8 {K : ℕ} (v : ℚ) (va vb : Fin K → ℚ) :
    psum v (chunkSum8 va vb) (chunks8 K) = v + ∑ k, va k * vb k := by
  rw [psum, sum_chunkSum8]

/-- **The composed bound** after `j` chunks: `errB 0 = 0` and
`errB (j+1) = (1+ρ+σ)·errB j + ρ·max |psum v c j| (m j) + σ·|psum v c (j+1)| + α`. -/
def errB (ρ σ α v : ℚ) (c m : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | j + 1 => (1 + ρ + σ) * errB ρ σ α v c m j + ρ * max |psum v c j| (m j) +
      σ * |psum v c (j + 1)| + α

theorem errB_nonneg {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) (v : ℚ) (c m : ℕ → ℚ) :
    ∀ j, 0 ≤ errB ρ σ α v c m j
  | 0 => le_rfl
  | j + 1 => by
    have ih := errB_nonneg hρ hσ hα v c m j
    have h1 : 0 ≤ max |psum v c j| (m j) := (abs_nonneg _).trans (le_max_left _ _)
    simp only [errB]
    have : 0 ≤ (1 + ρ + σ) * errB ρ σ α v c m j := mul_nonneg (by linarith) ih
    have : 0 ≤ ρ * max |psum v c j| (m j) := mul_nonneg hρ h1
    have : 0 ≤ σ * |psum v c (j + 1)| := mul_nonneg hσ (abs_nonneg _)
    linarith

/-- **A chain of chunk steps.** If each step `j < n` satisfies the facts' bound (for every `Mj` at
least `|s j|` and `m j`), then after `j ≤ n` steps `s j` is within `errB` of the exact partial sum
`partial (s 0) c j`. -/
theorem chain_err {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) {s c m : ℕ → ℚ} {n : ℕ}
    (hstep : ∀ j < n, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + c j)| ≤ ρ * Mj + σ * |s j + c j| + α) :
    ∀ j ≤ n, |s j - psum (s 0) c j| ≤ errB ρ σ α (s 0) c m j := by
  intro j
  induction j with
  | zero => intro _; simp [psum_zero, errB]
  | succ j ih =>
    intro hj
    have e := ih (by omega)
    set S := psum (s 0) c j with hS
    set E := errB ρ σ α (s 0) c m j with hE
    have hE0 : 0 ≤ E := (abs_nonneg _).trans e
    have hsj : |s j| ≤ |S| + E := by
      have := abs_sub_abs_le_abs_sub (s j) S
      linarith
    have hstep' := hstep j (by omega) (max (|S| + E) (m j)) (hsj.trans (le_max_left _ _))
      (le_max_right _ _)
    have hsc : |s j + c j| ≤ |S + c j| + E := by
      have h1 : s j + c j = (S + c j) + (s j - S) := by ring
      rw [h1]
      exact (abs_add_le _ _).trans (by linarith)
    have hmax : max (|S| + E) (m j) ≤ max |S| (m j) + E := by
      apply max_le
      · linarith [le_max_left |S| (m j)]
      · linarith [le_max_right |S| (m j)]
    have hS1 : psum (s 0) c (j + 1) = S + c j := psum_succ _ _ _
    have htri : |s (j + 1) - (S + c j)| ≤ |s (j + 1) - (s j + c j)| + |s j - S| := by
      have h1 : s (j + 1) - (S + c j) = (s (j + 1) - (s j + c j)) + (s j - S) := by ring
      rw [h1]; exact abs_add_le _ _
    rw [hS1]
    show |s (j + 1) - (S + c j)| ≤ (1 + ρ + σ) * E + ρ * max |S| (m j) + σ * |psum (s 0) c (j + 1)| + α
    rw [hS1]
    have h2 := mul_le_mul_of_nonneg_left hmax hρ
    have h3 := mul_le_mul_of_nonneg_left hsc hσ
    nlinarith

/-- **The closed form.** `errB j ≤ (1+ρ+σ)^j · Σ_{i<j} (ρ·max |psum i| (m i) + σ·|psum (i+1)| + α)`. -/
theorem errB_le {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) (v : ℚ) (c m : ℕ → ℚ) :
    ∀ j, errB ρ σ α v c m j ≤ (1 + ρ + σ) ^ j *
      ∑ i ∈ Finset.range j, (ρ * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α)
  | 0 => by simp [errB]
  | j + 1 => by
    have ih := errB_le hρ hσ hα v c m j
    set T := ∑ i ∈ Finset.range j, (ρ * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α)
    set t := ρ * max |psum v c j| (m j) + σ * |psum v c (j + 1)| + α
    have ht : 0 ≤ t := by
      have h1 : 0 ≤ max |psum v c j| (m j) := (abs_nonneg _).trans (le_max_left _ _)
      have := mul_nonneg hρ h1
      have := mul_nonneg hσ (abs_nonneg (psum v c (j + 1)))
      linarith
    have hq : 1 ≤ 1 + ρ + σ := by linarith
    have hqj : 1 ≤ (1 + ρ + σ) ^ (j + 1) := one_le_pow₀ hq
    have hT : 0 ≤ T := Finset.sum_nonneg fun i _ => by
      have h1 : 0 ≤ max |psum v c i| (m i) := (abs_nonneg _).trans (le_max_left _ _)
      have := mul_nonneg hρ h1
      have := mul_nonneg hσ (abs_nonneg (psum v c (i + 1)))
      linarith
    rw [Finset.sum_range_succ]
    rw [show errB ρ σ α v c m (j + 1) = (1 + ρ + σ) * errB ρ σ α v c m j + t by
      simp only [errB, t]; ring]
    have e1 : (1 + ρ + σ) * errB ρ σ α v c m j ≤ (1 + ρ + σ) * ((1 + ρ + σ) ^ j * T) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have e2 : (1 + ρ + σ) * ((1 + ρ + σ) ^ j * T) = (1 + ρ + σ) ^ (j + 1) * T := by ring
    have e3 : t ≤ (1 + ρ + σ) ^ (j + 1) * t := le_mul_of_one_le_left ht hqj
    nlinarith

/-- The largest lifted product of chunk `j` (`0` for an empty chunk). -/
def chunkLift8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  (chunk8 K j).fold max 0 fun k => liftMag da (va k) * liftMag db (vb k)

theorem le_chunkLift8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) {j : ℕ} {k : Fin K}
    (hk : k ∈ chunk8 K j) : liftMag da (va k) * liftMag db (vb k) ≤ chunkLift8 da db va vb j := by
  unfold chunkLift8
  exact (Finset.le_fold_max _).mpr (Or.inr ⟨k, hk, le_rfl⟩)

theorem chunkLift8_nonneg (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) :
    0 ≤ chunkLift8 da db va vb j := by
  unfold chunkLift8
  exact (Finset.le_fold_max _).mpr (Or.inl le_rfl)

/-- **An fp8 `tl.dot` from its exact inputs.** Under `DotChunkAcc`, for any bounds `m j` on the
lifted products of each chunk (`chunkLift8` will do), the result is within
`errB ρ σ α vc (chunkSum8 va vb) m (chunks8 K)` of `vc + Σ va·vb`. -/
theorem DotChunkAcc.err {P : Platform} {cfg : DotCfg} {ρ σ α : ℚ} (h : DotChunkAcc P cfg ρ σ α)
    (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) {M N K : ℕ} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val} (hd : P.dot cfg a b acc r)
    (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      cfg.acc.bound)
    (m : ℕ → ℚ) (hm : ∀ j < chunks8 K, ∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ m j) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤ errB ρ σ α vc (chunkSum8 va vb) m (chunks8 K) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have e := chain_err (s := s) (c := chunkSum8 va vb) (m := m) (n := chunks8 K) hρ hσ
    (fun j hj Mj hM hmj => hstep j hj Mj hM fun k hk => (hm j hj k hk).trans hmj) (chunks8 K) le_rfl
  rw [hn, h0, psum_chunks8] at e
  exact e

theorem rhoChunk8_nonneg : 0 ≤ rhoChunk8 := by unfold rhoChunk8; positivity
theorem sigmaChunk8_nonneg : 0 ≤ sigmaChunk8 := by unfold sigmaChunk8; positivity
theorem alphaChunk8_nonneg : 0 ≤ alphaChunk8 := by unfold alphaChunk8; positivity

/-- **`DotE5m2Chunk` and the other chunk facts, from the exact inputs**: the result is within
`errB 2^-7 2^-12 2^-126 vc (chunkSum8 va vb) m (chunks8 K)` of `vc + Σ va·vb`. -/
theorem DotF8Chunk.err {P : Platform} {da db : DType} (h : DotF8Chunk P da db) (p : InputPrecision)
    {M N K : ℕ} {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {acc r : Tensor [M, N] F32}
    (hd : P.dot ⟨da, db, .f32, p, 2 ^ 30⟩ a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ)
    (vc : ℚ) (hva : ∀ k, da.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k))
    (hvc : DType.val? .f32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      DType.bound .f32)
    (m : ℕ → ℚ) (hm : ∀ j < chunks8 K, ∀ k ∈ chunk8 K j, liftMag da (va k) * liftMag db (vb k) ≤ m j) :
    ∃ vr, DType.val? .f32 (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        errB rhoChunk8 sigmaChunk8 alphaChunk8 vc (chunkSum8 va vb) m (chunks8 K) :=
  DotChunkAcc.err (h p) rhoChunk8_nonneg sigmaChunk8_nonneg hd i n va vb vc hva hvb hvc hroom m hm

/-! ## A bound on the whole dot, relative to lifted magnitudes -/

theorem liftMag_nonneg (d : DType) (q : ℚ) : 0 ≤ liftMag d q := by
  unfold liftMag
  split_ifs
  · exact le_rfl
  · exact (abs_nonneg q).trans (le_max_left _ _)

theorem abs_le_liftMag (d : DType) (q : ℚ) : |q| ≤ liftMag d q := by
  unfold liftMag
  split_ifs with h
  · rw [h, abs_zero]
  · exact le_max_left _ _

theorem liftMag_zero (d : DType) : liftMag d 0 = 0 := by simp [liftMag]

/-- A value at least the smallest normal magnitude (a normal value of the dtype) counts as `|q|`. -/
theorem liftMag_of_le {d : DType} {q : ℚ} (h : minNormal8 d ≤ |q|) : liftMag d q = |q| := by
  unfold liftMag
  split_ifs with h0
  · rw [h0, abs_zero]
  · exact max_eq_left h

theorem minNormal8_nonneg (d : DType) : 0 ≤ minNormal8 d := by
  cases d <;> simp only [minNormal8] <;> positivity

theorem liftMag_le (d : DType) (q : ℚ) : liftMag d q ≤ |q| + minNormal8 d := by
  have h0 := minNormal8_nonneg d
  unfold liftMag
  split_ifs
  · exact add_nonneg (abs_nonneg q) h0
  · exact max_le (by linarith) (by linarith [abs_nonneg q])

theorem abs_mul_le_liftMag (da db : DType) (x y : ℚ) : |x * y| ≤ liftMag da x * liftMag db y := by
  rw [abs_mul]
  exact mul_le_mul (abs_le_liftMag da x) (abs_le_liftMag db y) (abs_nonneg y) (liftMag_nonneg da x)

/-- The lifted magnitude of chunk `j`: `Σ_{k ∈ chunk8 K j} liftMag da (va k) · liftMag db (vb k)`. -/
def chunkLiftSum8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ :=
  ∑ k ∈ chunk8 K j, liftMag da (va k) * liftMag db (vb k)

theorem sum_chunkLiftSum8 (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks8 K), chunkLiftSum8 da db va vb j =
      ∑ k, liftMag da (va k) * liftMag db (vb k) := by
  unfold chunkLiftSum8 chunk8
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 32)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks8 k)) _

/- The shape of a whole `tl.dot` relative to lifted magnitudes, `LiftAcc` here in round 24, is
`Rules.Num.DotLiftAcc` since round 25 (the same statement), which the 16-bit facts `DotF16` and
`DotBF16` also have (library issue L126). For the fp8 dot it stays a lemma's statement, not a device
fact: `DotF8Chunk.liftAcc` proves it from the chunk facts. -/

/-- **A whole fp8 `tl.dot` from the chunk steps.** Under `DotChunkAcc` with constants `ρ σ α`, the
result is within `DotLoop.bound (ρ + σ) α ⌈K/32⌉ L` of `vc + Σ va·vb`,
`L = |vc| + Σ liftMag cfg.a (va k) · liftMag cfg.b (vb k)`. Each chunk `j` is taken with
`Mj = |s j| + A j` (`A j` its lifted magnitude, `chunkLiftSum8`), so it adds at most
`(ρ + σ)·(|s j| + A j) + α`, and `DotLoop.accum_step` composes the chunks. -/
theorem DotChunkAcc.liftErr {P : Platform} {cfg : DotCfg} {ρ σ α : ℚ} (h : DotChunkAcc P cfg ρ σ α)
    (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) {M N K : ℕ} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val} (hd : P.dot cfg a b acc r)
    (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
      cfg.acc.bound) :
    ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        DotLoop.bound (ρ + σ) α (chunks8 K) (|vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k)) := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨vr, hvr, ?_⟩
  have hA0 : ∀ j, 0 ≤ chunkLiftSum8 cfg.a cfg.b va vb j := fun j =>
    Finset.sum_nonneg fun k _ => mul_nonneg (liftMag_nonneg _ _) (liftMag_nonneg _ _)
  have hcA : ∀ j, |chunkSum8 va vb j| ≤ chunkLiftSum8 cfg.a cfg.b va vb j := fun j =>
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _)
  have hρσ : 0 ≤ ρ + σ := add_nonneg hρ hσ
  have key : ∀ j ≤ chunks8 K, |s j - psum vc (chunkSum8 va vb) j| ≤
      DotLoop.bound (ρ + σ) α j (psum |vc| (chunkLiftSum8 cfg.a cfg.b va vb) j) := by
    intro j
    induction j with
    | zero => intro _; simp [h0, psum_zero, DotLoop.bound]
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
      have hSt : |psum vc (chunkSum8 va vb) j| ≤ psum |vc| (chunkLiftSum8 cfg.a cfg.b va vb) j := by
        unfold psum
        exact (abs_add_le _ _).trans (add_le_add le_rfl
          ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => hcA i)))
      rw [psum_succ, psum_succ]
      exact DotLoop.accum_step hρσ hα hSt (hcA j) e hs'
  have e := key (chunks8 K) le_rfl
  rw [hn, psum_chunks8, psum, sum_chunkLiftSum8] at e
  exact e

theorem relDot8_nonneg (K : ℕ) : 0 ≤ relDot8 K := by
  unfold relDot8
  have h7 : (0 : ℚ) ≤ 2 ^ (-7 : ℤ) := by positivity
  have : (1 : ℚ) ≤ (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K := one_le_pow₀ (by linarith)
  linarith

/-- The relative constant of the whole-dot bound from the chunk facts:
`(1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1`. -/
def relLift8 (K : ℕ) : ℚ := (1 + (rhoChunk8 + sigmaChunk8)) ^ chunks8 K - 1

/-- The absolute constant of the whole-dot bound from the chunk facts:
`⌈K/32⌉·(1 + 2^-7 + 2^-12)^⌈K/32⌉·2^-126`. -/
def absLift8 (K : ℕ) : ℚ := chunks8 K * (1 + (rhoChunk8 + sigmaChunk8)) ^ chunks8 K * alphaChunk8

private theorem base_le : (1 : ℚ) + 2 ^ (-7 : ℤ) ≤ 1 + (rhoChunk8 + sigmaChunk8) := by
  unfold rhoChunk8 sigmaChunk8
  have : (0 : ℚ) ≤ 2 ^ (-12 : ℤ) := by positivity
  linarith

theorem relDot8_le_relLift8 (K : ℕ) : relDot8 K ≤ relLift8 K := by
  unfold relDot8 relLift8
  have := pow_le_pow_left₀ (by positivity) base_le (chunks8 K)
  linarith

theorem absDot8_le_absLift8 (K : ℕ) : absDot8 K ≤ absLift8 K := by
  unfold absDot8 absLift8 alphaChunk8
  have hp := pow_le_pow_left₀ (by positivity) base_le (chunks8 K)
  have := mul_le_mul_of_nonneg_left hp (Nat.cast_nonneg (α := ℚ) (chunks8 K))
  exact mul_le_mul_of_nonneg_right this (by positivity)

theorem relLift8_nonneg (K : ℕ) : 0 ≤ relLift8 K := (relDot8_nonneg K).trans (relDot8_le_relLift8 K)

theorem absLift8_nonneg (K : ℕ) : 0 ≤ absLift8 K := by
  unfold absLift8 rhoChunk8 sigmaChunk8 alphaChunk8; positivity

/-- **A whole fp8 `tl.dot` from `DotE5m2Chunk` and the other chunk facts**: `DotLiftAcc` with
`relLift8`, `absLift8`. The room on lifted magnitudes covers the facts' room (`relDot8 ≤ relLift8`,
`absDot8 ≤ absLift8`, `|a·b| ≤ liftMag a · liftMag b`). -/
theorem DotF8Chunk.liftAcc {P : Platform} {da db : DType} (h : DotF8Chunk P da db)
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
  obtain ⟨vr, hvr, herr⟩ := DotChunkAcc.liftErr (h p) rhoChunk8_nonneg sigmaChunk8_nonneg
    alphaChunk8_nonneg hd i n va vb vc hva hvb hvc (by dsimp only; linarith)
  refine ⟨vr, hvr, herr.trans (le_of_eq ?_)⟩
  unfold DotLoop.bound relLift8 absLift8
  ring

end Dot8Chunk

/-! ## Satisfiability -/

namespace Platform.ref

variable {rel abs : MathFn → TVA.Math.Float.Format → ℝ}
  {sp : (fn : MathFn) → (f : TVA.Math.Float.Format) → FloatV f → Option (FloatV f)}

/-- **`DotChunkAcc` holds of `Platform.ref`** for a float accumulator, with `sigma` at least one
rounding's `u` and `alpha` at least its `η`: the partial values are the exact partial sums, and the
last step is the one rounding of the exact result. -/
theorem dotChunkAcc {cfg : DotCfg} {g : TVA.Math.Float.Format} (hg : cfg.acc.format? = some g)
    {rho sigma alpha : ℚ} (hrho : 0 ≤ rho) (hsigma : g.u ≤ sigma) (halpha : g.eta ≤ alpha) :
    Rules.Num.DotChunkAcc (Platform.ref rel abs sp) cfg rho sigma alpha := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hu : 0 ≤ g.u := TVA.Math.Float.Format.u_nonneg
  have he : 0 ≤ g.eta := TVA.Math.Float.Format.eta_nonneg
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
  have hqm : |vc + ∑ k, va k * vb k| ≤ |vc| + ∑ k, |va k * vb k| :=
    (abs_add_le _ _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs _ _))
  have hrel : 0 ≤ relDot8 K := by
    unfold relDot8
    have h7 : (0 : ℚ) ≤ 2 ^ (-7 : ℤ) := by positivity
    have : (1 : ℚ) ≤ (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K := one_le_pow₀ (by linarith)
    linarith
  have habs : 0 ≤ absDot8 K := by unfold absDot8; positivity
  have hqmax : |vc + ∑ k, va k * vb k| ≤ g.maxVal := by
    rw [← DType.bound_of_format? hg]
    have := mul_nonneg hrel hmag
    linarith
  obtain ⟨v, hv, herr⟩ := DType.val?_ofRat_float hg hqmax
  have hn1 : 1 ≤ chunks8 K := by unfold chunks8; omega
  set nC := chunks8 K with hnC
  let s : ℕ → ℚ := fun j => if j < nC then Dot8Chunk.psum vc (chunkSum8 va vb) j else v
  refine ⟨s, v, by rw [hri]; exact hv, ?_, ?_, fun j hj Mj hM _ => ?_⟩
  · show (if 0 < nC then _ else v) = vc
    rw [ite_eq_left (by omega : 0 < nC), Dot8Chunk.psum_zero]
  · show (if nC < nC then _ else v) = v
    rw [ite_eq_right (lt_irrefl nC)]
  · have hM0 : 0 ≤ Mj := (abs_nonneg _).trans hM
    have hR : 0 ≤ rho * Mj + sigma * |s j + chunkSum8 va vb j| + alpha := by
      have := mul_nonneg hrho hM0
      have := mul_nonneg (hu.trans hsigma) (abs_nonneg (s j + chunkSum8 va vb j))
      linarith
    have hsj : s j = Dot8Chunk.psum vc (chunkSum8 va vb) j := ite_eq_left hj
    by_cases hj1 : j + 1 < nC
    · have hs1 : s (j + 1) = Dot8Chunk.psum vc (chunkSum8 va vb) (j + 1) := ite_eq_left hj1
      have h0 : s (j + 1) - (s j + chunkSum8 va vb j) = 0 := by
        rw [hs1, hsj, Dot8Chunk.psum_succ]; ring
      rw [h0, abs_zero]
      exact hR
    · have hjn : j + 1 = nC := by omega
      have hs1 : s (j + 1) = v := ite_eq_right hj1
      have hsum : s j + chunkSum8 va vb j = vc + ∑ k, va k * vb k := by
        rw [hsj, ← Dot8Chunk.psum_succ, hjn, hnC, Dot8Chunk.psum_chunks8]
      rw [hs1, hsum]
      have h1 : |vc + ∑ k, va k * vb k| * g.u ≤ sigma * |vc + ∑ k, va k * vb k| := by
        rw [mul_comm]; exact mul_le_mul_of_nonneg_right hsigma (abs_nonneg _)
      have := mul_nonneg hrho hM0
      linarith

end Platform.ref

namespace Platform.sm90ref

/-- The chunk facts hold of `Platform.sm90ref` (one `float32` rounding of the exact result). -/
theorem dotF8Chunk (a b : DType) : DotF8Chunk Platform.sm90ref a b := fun _ =>
  Platform.ref.dotChunkAcc (g := Format.binary32) rfl Dot8Chunk.rhoChunk8_nonneg
    (by rw [Format.binary32_u]; unfold sigmaChunk8; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (by rw [Format.binary32_eta]; unfold alphaChunk8; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))

theorem dotE4m3Chunk : DotE4m3Chunk Platform.sm90ref := dotF8Chunk _ _
theorem dotE5m2Chunk : DotE5m2Chunk Platform.sm90ref := dotF8Chunk _ _
theorem dotE4m3E5m2Chunk : DotE4m3E5m2Chunk Platform.sm90ref := dotF8Chunk _ _
theorem dotE5m2E4m3Chunk : DotE5m2E4m3Chunk Platform.sm90ref := dotF8Chunk _ _

end Platform.sm90ref

end TVA
