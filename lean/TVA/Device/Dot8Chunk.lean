import TVA.Device.Sm90

/-!
# An fp8 `tl.dot`, chunk by chunk (sm_90, Triton 3.8.0)

These are the device facts for `tl.dot` of fp8 operands (`float8e4nv`, `float8e5`) into
`float32`. A bound on the whole dot relative to `|acc| + Σ|a·b|` does not hold on the device: when
a subnormal operand sets the instruction's alignment window, the error exceeds it. The facts
state the accumulation itself, per output element: the `K` products in chunks of `32`
consecutive `k` (`chunk8 K j`: the `k` with `k / 32 = j`), added in the order of the chunks, each
to the accumulator as it stands, with an error bounded by the largest term the chunk's
instruction aligns to, the accumulator included:

* there are partial values `s 0 = acc, s 1, …, s n = result`, `n = ⌈K/32⌉` (`chunks8 K`);
* for each chunk `j < n`, every `Mj` that bounds `|s j|` and every lifted product
  `liftMag a_k · liftMag b_k` of the chunk (a subnormal operand counts as `2^-14` for `float8e5`,
  `2^-6` for `float8e4nv`), `s (j+1)` is within
  `min (2^-7·Mj) (2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126` of `s j + c_j`, with `c_j` the
  chunk's exact sum (`chunkSum8`) and `A_j` its exact magnitude `Σ |a_k·b_k|` (`chunkAbs8`).

**Why the constants** (`evidence/num-dot-chunk`, a bit-level model of `wgmma` equal to the device
on every element tested). Each instruction aligns the accumulator and its `32` products to the
largest exponent field `E`, keeps 13 bits below it (truncating toward zero, unit `u = 2^(E-13)`,
`u ≤ 2^-13·Mj + 2^-139`), and cuts the sum to 14 significant bits. Each of the 33 terms loses
less than `u`, so a chunk's error is below `33·(1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| +
34·2^-139`: the first arm keeps a factor of about 2. A product truncated toward zero loses at most
its own magnitude, and the accumulator less than `u`, so the error is also below
`(1 + 2^-13)·(A_j + u) + 2^-13·|s j + c_j|`: the second arm keeps a factor of about 2
(`evidence/num-dot-chunk-loss`). On `mma.sync` (fp8 converted to `float16`, two `float16`
instructions per chunk) the error is far smaller. Both arms bound the same partial values.

The first arm alone is `DotChunkAcc` (`DotE4m3Chunk`, …), which follows from the facts here
(`DotChunkLoss.chunkAcc`).
-/
namespace TVA.Device

/-- The number of `32`-product chunks of an fp8 `tl.dot` with `K` products: `⌈K/32⌉`. -/
def chunks8 (K : ℕ) : ℕ := (K + 31) / 32

/-- The relative constant of the chunk facts' room: `(1 + 2^-7)^⌈K/32⌉ − 1`. -/
def relDot8 (K : ℕ) : ℚ := (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K - 1

/-- The absolute constant of the chunk facts' room: `⌈K/32⌉·(1 + 2^-7)^⌈K/32⌉·2^-126`. -/
def absDot8 (K : ℕ) : ℚ := chunks8 K * (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K * 2 ^ (-126 : ℤ)

/-- Chunk `j` of `K` products: the `k < K` with `k / 32 = j`, that is the `32` consecutive `k`
from `32·j` (fewer in a last, partial chunk). -/
def chunk8 (K j : ℕ) : Finset (Fin K) := Finset.univ.filter fun k => k.val / 32 = j

/-- The exact sum of chunk `j`: `Σ_{k ∈ chunk8 K j} va k · vb k`. -/
def chunkSum8 {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ := ∑ k ∈ chunk8 K j, va k * vb k

/-- The exact magnitude of chunk `j`: `Σ_{k ∈ chunk8 K j} |va k · vb k|` (not lifted). -/
def chunkAbs8 {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ := ∑ k ∈ chunk8 K j, |va k * vb k|

/-- The chunk facts' relative constant on the largest term: `2^-7`. -/
def rhoChunk8 : ℚ := 2 ^ (-7 : ℤ)

/-- The chunk facts' relative constant on the new value: `2^-12`. -/
def sigmaChunk8 : ℚ := 2 ^ (-12 : ℤ)

/-- The chunk facts' absolute constant: `2^-126` per chunk. -/
def alphaChunk8 : ℚ := 2 ^ (-126 : ℤ)

/-- The second arm's constant on the chunk's exact magnitude: `2`. -/
def kappaChunk8 : ℚ := 2

/-- The second arm's constant on the largest term: `2^-12`. -/
def rhoLossChunk8 : ℚ := 2 ^ (-12 : ℤ)

/-- Every `k < K` lies in a chunk `j < chunks8 K`. -/
theorem div_lt_chunks8 {K : ℕ} (k : Fin K) : k.val / 32 < chunks8 K := by
  unfold chunks8; have := k.isLt; omega

/-- The chunks' exact sums add up to the whole sum `Σ va·vb`. -/
theorem sum_chunkSum8 {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ j ∈ Finset.range (chunks8 K), chunkSum8 va vb j = ∑ k, va k * vb k := by
  unfold chunkSum8 chunk8
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin K => k.val / 32)
    (fun k _ => Finset.mem_range.mpr (div_lt_chunks8 k)) _

variable (P : Platform)

/-- **A `tl.dot` chunk by chunk**, per output element `(i, n)`. Suppose row `i` of `a` and
column `n` of `b` have values `va`, `vb` (`DType.val?`), and the accumulator element has value
`vc`, with room (`mag + relDot8 K·mag + absDot8 K` at most the accumulator's `DType.bound`,
`mag = |vc| + Σ|va·vb|`). Then the result element has a value `vr`, and there are partial values
`s 0 = vc, …, s (chunks8 K) = vr` such that, for each chunk `j < chunks8 K`, every `Mj` at least
`|s j|` and every `liftMag cfg.a (va k) · liftMag cfg.b (vb k)` of the chunk,
`|s (j+1) − (s j + chunkSum8 va vb j)| ≤ rho·Mj + sigma·|s j + chunkSum8 va vb j| + alpha`. -/
def DotChunkAcc (cfg : DotCfg) (rho sigma alpha : ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
        cfg.acc.bound →
      ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧
        s (chunks8 K) = vr ∧
        ∀ j < chunks8 K, ∀ Mj : ℚ, |s j| ≤ Mj →
          (∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
          |s (j + 1) - (s j + chunkSum8 va vb j)| ≤
            rho * Mj + sigma * |s j + chunkSum8 va vb j| + alpha

/-- **A `tl.dot` chunk by chunk, each step within the smaller of two bounds**, per output
element `(i, n)`. The values, the room and the partial values are as in `DotChunkAcc`, and each
step is within `min (rho·Mj) (kappa·A_j + rho'·Mj) + sigma·|s j + c_j| + alpha` of
`s j + c_j`, with `c_j = chunkSum8 va vb j` and `A_j = chunkAbs8 va vb j`. -/
def DotChunkLoss (cfg : DotCfg) (rho sigma alpha kappa rho' : ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (relDot8 K * (|vc| + ∑ k, |va k * vb k|) + absDot8 K) ≤
        cfg.acc.bound →
      ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧
        s (chunks8 K) = vr ∧
        ∀ j < chunks8 K, ∀ Mj : ℚ, |s j| ≤ Mj →
          (∀ k ∈ chunk8 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
          |s (j + 1) - (s j + chunkSum8 va vb j)| ≤
            min (rho * Mj) (kappa * chunkAbs8 va vb j + rho' * Mj) +
              sigma * |s j + chunkSum8 va vb j| + alpha

/-- `tl.dot` of fp8 operands `a`, `b` into `float32` with Triton's default
`max_num_imprecise_acc` on sm_90 (`2^30`), any `input_precision`, chunk by chunk with each step
within `min (2^-7·Mj) (2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126` (`DotChunkLoss`). -/
def DotF8ChunkLoss (a b : DType) : Prop :=
  ∀ p : InputPrecision, DotChunkLoss P ⟨a, b, .f32, p, 2 ^ 30⟩ rhoChunk8 sigmaChunk8 alphaChunk8
    kappaChunk8 rhoLossChunk8

/-- **`tl.dot` of `float8e4nv` operands into `float32`, chunk by chunk** (`DotF8ChunkLoss`; a
subnormal operand counts as `2^-6` in `Mj`). On `wgmma` each instruction aligns the accumulator
and its 32 products to the largest exponent field and keeps 13 bits below it, truncating toward
zero, then cuts the sum to 14 significant bits; on `mma.sync` Triton converts to `float16`.
Evidence: `evidence/num-dot-chunk`, `evidence/num-dot-chunk-loss`. -/
def DotE4m3ChunkLoss : Prop := DotF8ChunkLoss P .f8e4nv .f8e4nv

/-- **`tl.dot` of `float8e5` operands into `float32`, chunk by chunk** (`DotF8ChunkLoss`, as
`DotE4m3ChunkLoss`; a subnormal operand counts as `2^-14`). Evidence: `evidence/num-dot-chunk`,
`evidence/num-dot-chunk-loss`. -/
def DotE5m2ChunkLoss : Prop := DotF8ChunkLoss P .f8e5 .f8e5

/-- **`tl.dot` of `float8e4nv` by `float8e5` into `float32`, chunk by chunk**
(`DotF8ChunkLoss`). Evidence: `evidence/num-dot-chunk`, `evidence/num-dot-chunk-loss`. -/
def DotE4m3E5m2ChunkLoss : Prop := DotF8ChunkLoss P .f8e4nv .f8e5

/-- **`tl.dot` of `float8e5` by `float8e4nv` into `float32`, chunk by chunk**
(`DotF8ChunkLoss`). Evidence: `evidence/num-dot-chunk`, `evidence/num-dot-chunk-loss`. -/
def DotE5m2E4m3ChunkLoss : Prop := DotF8ChunkLoss P .f8e5 .f8e4nv

/-- `tl.dot` of fp8 operands `a`, `b` into `float32`, chunk by chunk with each step within
`2^-7·Mj + 2^-12·|s j + c_j| + 2^-126` (`DotChunkAcc`, the first arm of `DotF8ChunkLoss`). -/
def DotF8Chunk (a b : DType) : Prop :=
  ∀ p : InputPrecision, DotChunkAcc P ⟨a, b, .f32, p, 2 ^ 30⟩ rhoChunk8 sigmaChunk8 alphaChunk8

/-- **`tl.dot` of `float8e4nv` operands, chunk by chunk, first arm** (`DotF8Chunk`). It follows
from `DotE4m3ChunkLoss`. -/
def DotE4m3Chunk : Prop := DotF8Chunk P .f8e4nv .f8e4nv

/-- **`tl.dot` of `float8e5` operands, chunk by chunk, first arm** (`DotF8Chunk`). It follows
from `DotE5m2ChunkLoss`. -/
def DotE5m2Chunk : Prop := DotF8Chunk P .f8e5 .f8e5

/-- **`tl.dot` of `float8e4nv` by `float8e5`, chunk by chunk, first arm** (`DotF8Chunk`). It
follows from `DotE4m3E5m2ChunkLoss`. -/
def DotE4m3E5m2Chunk : Prop := DotF8Chunk P .f8e4nv .f8e5

/-- **`tl.dot` of `float8e5` by `float8e4nv`, chunk by chunk, first arm** (`DotF8Chunk`). It
follows from `DotE5m2E4m3ChunkLoss`. -/
def DotE5m2E4m3Chunk : Prop := DotF8Chunk P .f8e5 .f8e4nv

end TVA.Device
