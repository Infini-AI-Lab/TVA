import TVA.Rules.Num.Sm90

/-!
# An fp8 `tl.dot` chunk by chunk (sm_90, Triton 3.8.0)

Page: `docs/rules/num.md` ("`tl.dot` of fp8, chunk by chunk"). Evidence: `evidence/num-dot-chunk`.

These are the library's fp8 `tl.dot` facts. The whole-dot facts `DotE4m3`, …, `DotE5m2E4m3`, which
bounded the error relative to `|acc| + Σ|a·b|`, were withdrawn in round 24 (library issue L125): when
a subnormal operand sets the window, the device's error exceeds them (evidence Part F). The facts
here (library issue L124) state the accumulation itself, per output element: the `K` products in chunks of `32` consecutive `k` (`chunk8 K j`: the `k` with
`k / 32 = j`), added in the order of the chunks, each to the accumulator as it stands, with an error
bounded by the largest term the chunk's instruction aligns to, the accumulator included:

* there are partial values `s 0 = acc, s 1, …, s n = result`, `n = ⌈K/32⌉` (`chunks8 K`);
* for each chunk `j < n` and every `Mj` that bounds `|s j|` and every lifted product
  `liftMag a_k · liftMag b_k` of the chunk, `s (j+1)` is within
  `rho·Mj + sigma·|s j + c_j| + alpha` of `s j + c_j`, `c_j` the chunk's exact sum (`chunkSum8`).

`liftMag d q` (`TVA.Rules.Num`) is `0` for `q = 0` and `max |q| (2^emin)` otherwise (`minNormal8`:
`2^-14` for `float8e5`, `2^-6` for `float8e4nv`): the tensor core aligns a product by its operands'
exponent fields, and a subnormal operand's field reads as the smallest normal's.

The constants are `rhoChunk8 = 2^-7`, `sigmaChunk8 = 2^-12`, `alphaChunk8 = 2^-126`. On the `wgmma`
path (the evidence's bit-level model, equal to the device on every element tested) a chunk's error is
below `33·(1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| + 34·2^-139`: each of the 33 terms loses less than
the window's unit `u = 2^(E-13) ≤ 2^-13·Mj` (with `E` the largest exponent field), and the sum is
then cut to 14 significant bits. `rho` and `sigma` keep a factor of about 2 over that. On the
`mma.sync` path (fp8 converted to `float16`, two `float16` instructions per chunk, each aligned to
its largest field with 25 bits below it, `evidence/num-dot-sub16-round25`) the error is far smaller.

A bound on the whole dot follows by a lemma, relative to `|acc| + Σ liftMag a_k · liftMag b_k`:
`Dot8Chunk.DotF8Chunk.liftAcc` (`TVA.Lemmas.Dot8Chunk`), the shape `DotLiftAcc` with
`(1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1`.

**The loss bound** (round 26, library issue L124). Each step is relative to `Mj ≥ |s j|`, so an
accumulator fed back through many chunks is charged `2^-7 + 2^-12` of itself per chunk, and a bound
composed from these steps grows as `(1 + 2^-7 + 2^-12)^n`. The device's error does not compound:
by the same model every term and the sum are truncated toward zero, so no partial value exceeds
`|acc| + Σ_{i<j} A_i` in magnitude, and a product loses at most `min(|p|, u)`, so a chunk's error is
also below `(1 + 2^-13)·(A_j + u) + 2^-13·|s j + c_j|`, `A_j` the chunk's `Σ|a·b|` (`chunkAbs8`), with
`u ≤ 2^-13·Mj + 2^-139`. The facts `DotE4m3ChunkLoss`, …, `DotE5m2E4m3ChunkLoss` (`DotChunkLoss`,
`DotF8ChunkLoss`) state both bounds of the same partial values: each step within
`min (2^-7·Mj) (2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126` (`kappaChunk8 = 2`,
`rhoLossChunk8 = 2^-12`, each about twice the derivation's `1 + 2^-13` and `(1 + 2^-13)·2^-13`). They
give the facts above (`DotF8ChunkLoss.chunk`); a bound composed from them grows by `1 + 2^-11` per
chunk where the second arm is the smaller (`Dot8Chunk.errM`, `Dot8Chunk.chain_errM`).
-/
namespace TVA.Rules.Num

open TVA.Math.Float

/-- The number of `32`-product chunks of an fp8 `tl.dot` with `K` products: `⌈K/32⌉`. -/
def chunks8 (K : ℕ) : ℕ := (K + 31) / 32

/-- The relative constant of the chunk facts' room: `(1 + 2^-7)^⌈K/32⌉ − 1`. Before round 24 it was
also the constant of the whole-dot facts `DotE4m3`, …, withdrawn (L125). -/
def relDot8 (K : ℕ) : ℚ := (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K - 1

/-- The absolute constant of the chunk facts' room: `⌈K/32⌉·(1 + 2^-7)^⌈K/32⌉·2^-126`. -/
def absDot8 (K : ℕ) : ℚ := chunks8 K * (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K * 2 ^ (-126 : ℤ)

/- `minNormal8` and `liftMag`, defined here in round 24, are in `TVA.Rules.Num` (`Rules/Num.lean`)
since round 25, with the same names, where the 16-bit facts use them too (library issue L126). -/

/-- Chunk `j` of `K` products: the `k < K` with `k / 32 = j`, that is the `32` consecutive `k`
from `32·j` (fewer in a last, partial chunk). -/
def chunk8 (K j : ℕ) : Finset (Fin K) := Finset.univ.filter fun k => k.val / 32 = j

/-- The exact sum of chunk `j`: `Σ_{k ∈ chunk8 K j} va k · vb k`. -/
def chunkSum8 {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ := ∑ k ∈ chunk8 K j, va k * vb k

/-- The exact magnitude of chunk `j`: `Σ_{k ∈ chunk8 K j} |va k · vb k|`, the products' own
magnitudes (not lifted). -/
def chunkAbs8 {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ := ∑ k ∈ chunk8 K j, |va k * vb k|

variable (P : Platform)

/-- **A `tl.dot` chunk by chunk**, per output element `(i, n)`. Suppose row `i` of `a` and
column `n` of `b` have values `va`, `vb` (`DType.val?`), and the accumulator element has value `vc`,
with room (`mag + relDot8 K·mag + absDot8 K` at most the accumulator's `DType.bound`,
`mag = |vc| + Σ|va·vb|`). Then the result element has a value `vr`, and there are
partial values `s 0 = vc, …, s (chunks8 K) = vr` such that, for each chunk `j < chunks8 K` and every
`Mj` at least `|s j|` and every `liftMag cfg.a (va k) · liftMag cfg.b (vb k)` of the chunk,
`|s (j+1) − (s j + chunkSum8 va vb j)| ≤ rho·Mj + sigma·|s j + chunkSum8 va vb j| + alpha`. The
chunks are added in increasing order of `j`. -/
def DotChunkAcc (cfg : DotCfg) (rho sigma alpha : ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
        cfg.acc.bound →
      ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧ s (chunks8 K) = vr ∧
        ∀ j < chunks8 K, ∀ Mj : ℚ, |s j| ≤ Mj →
          (∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
          |s (j + 1) - (s j + chunkSum8 va vb j)| ≤
            rho * Mj + sigma * |s j + chunkSum8 va vb j| + alpha

/-- The chunk facts' relative constant on the largest term: `2^-7`. -/
def rhoChunk8 : ℚ := 2 ^ (-7 : ℤ)

/-- The chunk facts' relative constant on the new value: `2^-12`. -/
def sigmaChunk8 : ℚ := 2 ^ (-12 : ℤ)

/-- The chunk facts' absolute constant: `2^-126` per chunk. -/
def alphaChunk8 : ℚ := 2 ^ (-126 : ℤ)

/-- `tl.dot` of fp8 operands `a`, `b` into `float32` with Triton's default `max_num_imprecise_acc`
on sm_90 (`2^30`), any `input_precision`, chunk by chunk (`DotChunkAcc`) with `rhoChunk8`,
`sigmaChunk8`, `alphaChunk8`. -/
def DotF8Chunk (a b : DType) : Prop :=
  ∀ p : InputPrecision, DotChunkAcc P ⟨a, b, .f32, p, 2 ^ 30⟩ rhoChunk8 sigmaChunk8 alphaChunk8

/-- **`tl.dot` of `float8e4nv` operands into `float32`, chunk by chunk** (`DotF8Chunk`): partial
values `acc = s 0, …, s ⌈K/32⌉ = result`, chunk `j` (the `k` with `k / 32 = j`, in increasing `j`)
within `2^-7·Mj + 2^-12·|s j + c_j| + 2^-126` of `s j + c_j`, for every `Mj` bounding `|s j|` and the
chunk's lifted products (`liftMag`: a subnormal operand counts as `2^-6`). On `wgmma` each
instruction aligns the accumulator and its 32 products to the largest exponent field and keeps 13
bits below it, truncating toward zero, then cuts the sum to 14 significant bits; on `mma.sync`
Triton converts to `float16`. Evidence: `evidence/num-dot-chunk`. -/
def DotE4m3Chunk : Prop := DotF8Chunk P .f8e4nv .f8e4nv

/-- **`tl.dot` of `float8e5` operands into `float32`, chunk by chunk** (`DotF8Chunk`, as
`DotE4m3Chunk`; a subnormal operand counts as `2^-14`). Evidence: `evidence/num-dot-chunk`. -/
def DotE5m2Chunk : Prop := DotF8Chunk P .f8e5 .f8e5

/-- **`tl.dot` of `float8e4nv` by `float8e5` into `float32`, chunk by chunk** (`DotF8Chunk`).
Evidence: `evidence/num-dot-chunk`. -/
def DotE4m3E5m2Chunk : Prop := DotF8Chunk P .f8e4nv .f8e5

/-- **`tl.dot` of `float8e5` by `float8e4nv` into `float32`, chunk by chunk** (`DotF8Chunk`).
Evidence: `evidence/num-dot-chunk`. -/
def DotE5m2E4m3Chunk : Prop := DotF8Chunk P .f8e5 .f8e4nv

/-! ## Each chunk's error, the smaller of two bounds (round 26, library issue L124) -/

/-- **A `tl.dot` chunk by chunk, each step within the smaller of two bounds**, per output element
`(i, n)`. The values, the room and the partial values `s 0 = vc, …, s (chunks8 K) = vr` are as in
`DotChunkAcc`, and for each chunk `j < chunks8 K`, every `Mj` at least `|s j|` and every
`liftMag cfg.a (va k) · liftMag cfg.b (vb k)` of the chunk,
`|s (j+1) − (s j + c_j)| ≤ min (rho·Mj) (kappa·A_j + rho'·Mj) + sigma·|s j + c_j| + alpha`, with
`c_j = chunkSum8 va vb j` and `A_j = chunkAbs8 va vb j = Σ_chunk |va k · vb k|` (not lifted). The
first arm is `DotChunkAcc`'s bound (`DotChunkLoss.chunkAcc`). The second holds because a product
truncated toward zero to a multiple of the window's unit loses at most its own magnitude, and the
accumulator less than one unit (`docs/rules/num.md`). Both arms bound the same partial values. -/
def DotChunkLoss (cfg : DotCfg) (rho sigma alpha kappa rho' : ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
        cfg.acc.bound →
      ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧ s (chunks8 K) = vr ∧
        ∀ j < chunks8 K, ∀ Mj : ℚ, |s j| ≤ Mj →
          (∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
          |s (j + 1) - (s j + chunkSum8 va vb j)| ≤
            min (rho * Mj) (kappa * chunkAbs8 va vb j + rho' * Mj) +
              sigma * |s j + chunkSum8 va vb j| + alpha

/-- The loss bound's constant on the chunk's exact magnitude: `2`. -/
def kappaChunk8 : ℚ := 2

/-- The loss bound's constant on the largest term: `2^-12`. -/
def rhoLossChunk8 : ℚ := 2 ^ (-12 : ℤ)

/-- `tl.dot` of fp8 operands `a`, `b` into `float32` with Triton's default `max_num_imprecise_acc`
on sm_90 (`2^30`), any `input_precision`, chunk by chunk with each step within the smaller of two
bounds (`DotChunkLoss`): `min (2^-7·Mj) (2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126`. It gives
`DotF8Chunk` (`DotF8ChunkLoss.chunk`). -/
def DotF8ChunkLoss (a b : DType) : Prop :=
  ∀ p : InputPrecision, DotChunkLoss P ⟨a, b, .f32, p, 2 ^ 30⟩ rhoChunk8 sigmaChunk8 alphaChunk8
    kappaChunk8 rhoLossChunk8

/-- **`tl.dot` of `float8e4nv` operands into `float32`, chunk by chunk, each step within the smaller
of two bounds** (`DotF8ChunkLoss`): partial values as `DotE4m3Chunk`, chunk `j` within
`min (2^-7·Mj) (2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126` of `s j + c_j`, `A_j` the chunk's
`Σ|a·b|`. On `wgmma` each product truncated toward zero to a multiple of the window's unit `u`
loses at most its own magnitude, the accumulator less than `u ≤ 2^-13·Mj + 2^-139`, and the cut to
14 bits less than `2^-13` of the truncated sum. It gives `DotE4m3Chunk`. Evidence:
`evidence/num-dot-chunk`, `evidence/num-dot-chunk-round26`. -/
def DotE4m3ChunkLoss : Prop := DotF8ChunkLoss P .f8e4nv .f8e4nv

/-- **`tl.dot` of `float8e5` operands into `float32`, chunk by chunk, each step within the smaller
of two bounds** (`DotF8ChunkLoss`, as `DotE4m3ChunkLoss`; a subnormal operand counts as `2^-14` in
`Mj`). It gives `DotE5m2Chunk`. Evidence: `evidence/num-dot-chunk`,
`evidence/num-dot-chunk-round26`. -/
def DotE5m2ChunkLoss : Prop := DotF8ChunkLoss P .f8e5 .f8e5

/-- **`tl.dot` of `float8e4nv` by `float8e5` into `float32`, chunk by chunk, each step within the
smaller of two bounds** (`DotF8ChunkLoss`). It gives `DotE4m3E5m2Chunk`. Evidence:
`evidence/num-dot-chunk`, `evidence/num-dot-chunk-round26`. -/
def DotE4m3E5m2ChunkLoss : Prop := DotF8ChunkLoss P .f8e4nv .f8e5

/-- **`tl.dot` of `float8e5` by `float8e4nv` into `float32`, chunk by chunk, each step within the
smaller of two bounds** (`DotF8ChunkLoss`). It gives `DotE5m2E4m3Chunk`. Evidence:
`evidence/num-dot-chunk`, `evidence/num-dot-chunk-round26`. -/
def DotE5m2E4m3ChunkLoss : Prop := DotF8ChunkLoss P .f8e5 .f8e4nv

/-! ## Weakening -/

section Mono
variable {P}

/-- `DotChunkAcc` with constants at least as large. It states nothing new about the device. -/
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

/-- `DotChunkLoss` gives `DotChunkAcc` with the same `rho`, `sigma`, `alpha` (its first arm), on
the same partial values. It states nothing new about the device. -/
theorem DotChunkLoss.chunkAcc {cfg : DotCfg} {rho sigma alpha kappa rho' : ℚ}
    (h : DotChunkLoss P cfg rho sigma alpha kappa rho') : DotChunkAcc P cfg rho sigma alpha := by
  intro M N K a b acc r hd i n va vb vc hva hvb hvc hroom
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have e := hstep j hj Mj hM hk
  have := min_le_left (rho * Mj) (kappa * chunkAbs8 va vb j + rho' * Mj)
  linarith

/-- `DotChunkLoss` with constants at least as large. It states nothing new about the device. -/
theorem DotChunkLoss.mono {cfg : DotCfg} {rho sigma alpha kappa rho' rho₂ sigma₂ alpha₂ kappa₂ rho'₂ : ℚ}
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

/-- `DotF8ChunkLoss` gives `DotF8Chunk` (`DotChunkLoss.chunkAcc`): `DotE4m3ChunkLoss` gives
`DotE4m3Chunk`, and likewise for the other pairs. -/
theorem DotF8ChunkLoss.chunk {a b : DType} (h : DotF8ChunkLoss P a b) : DotF8Chunk P a b :=
  fun p => (h p).chunkAcc

end Mono

end TVA.Rules.Num
