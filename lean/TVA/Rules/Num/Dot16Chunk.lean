import TVA.Rules.Num.Sm90

/-!
# A 16-bit `tl.dot` chunk by chunk (sm_90, Triton 3.8.0)

Page: `docs/rules/num.md` ("`tl.dot` of `float16` and `bfloat16`, chunk by chunk"). Evidence:
`evidence/num-dot-chunk16-round33` (the chunks of 64 and the instructions' `k`), and
`evidence/num-dot-chunk16`, `evidence/num-dot-sub16-round25` (the per-instruction model; library
issue L150, rounds 32 and 33).

`DotF16` and `DotBF16` (`TVA.Rules.Num.Sm90`) bound a whole `tl.dot` of `float16` (`bfloat16`)
operands into `float32` by `relDot16 K = (K+2)·2^-22` of `|acc| + Σ liftMag a·liftMag b`, and a `K`
loop that feeds its accumulator back (`acc = tl.dot(a, b, acc)`) is charged that on the whole `|acc|`
at every step. The facts here state the accumulation itself, per output element, as the fp8 chunk
facts do (`TVA.Rules.Num.Dot8Chunk`): the `K` products in chunks of `64` consecutive `k`
(`chunk16 K j`: the `k` with `k / 64 = j`; all `K` of them when `K < 64`), added in the order of the
chunks, each to the accumulator as it stands:

* there are partial values `s 0 = acc, s 1, …, s n = result`, `n = ⌈K/64⌉` (`chunks16 K`);
* for each chunk `j < n`, every `Mj` that bounds `|s j|` and every lifted product
  `liftMag a_k · liftMag b_k` of the chunk, `s (j+1)` is within
  `min (2^-13·Mj) (2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| + 2^-143` of `s j + c_j`, `c_j` the
  chunk's exact sum (`chunkSum16`), `A_j` its exact magnitude `Σ_chunk |a_k·b_k|` (`chunkAbs16`).

**Which `k` an instruction adds** (`evidence/num-dot-chunk16-round33`, parts W and P). Each
`mma.sync…m16n8k16` or `wgmma…m64nNk16` adds 16 products to the accumulator the previous one left.
On `wgmma` they are 16 consecutive `k`. On `mma.sync` they depend on the operands' layout, whose
`kWidth` Triton 3.8.0 sets from the narrowest dtype the operands were loaded in
(`AccelerateMatmul.cpp:385-427, 561-568`; `kWidth = max(32 / bitwidth, 1)`,
`TritonGPUAttrDefs.td:1478-1488`, with `bitwidth` at most the operands' own 16): `kWidth = 2`
(both loaded in 16 bits or wider), 16 consecutive `k`; `kWidth = 4` (an operand converted in
registers from an 8-bit load, `fp8` or `int8`), the `k ≡ 0, 1 (mod 4)` of a block of 32, then the
`k ≡ 2, 3`; `kWidth = 8` (an 8-bit load unpacked by `tl.join` into twice as many `k`), the
`k ≡ 2i, 2i+1 (mod 8)` of a block of 64 for `i = 0, …, 3` (at `K = 32`, two instructions within
the 32). In every case each instruction's `k` lie in one aligned block of 64, the blocks are taken
in increasing order, and the instructions of a block one after the other: a chunk of 64 is at most
four instructions, in some order, whatever the partition.

**Why the constants** (the per-instruction model of `evidence/num-dot-sub16-round25`, equal to the
device on every element tested, every partition). Each instruction's terms are the accumulator `t`
and its 16 exact products; with `E` the largest exponent field (`e(a) + e(b)` for a product, a
subnormal operand read as the smallest normal; `max(⌊log₂|t|⌋, −126)` for the accumulator) and
`u = 2^(E−25)`, every term is truncated toward zero to a multiple of `u`, the truncated terms are
summed exactly (`S'`), and `S'` is truncated toward zero to `float32` (losing less than
`2^-23·|S'| + 2^-149`). Truncation never increases a magnitude, so before the chunk's `i`-th
instruction `|t| ≤ |s j| + Σ_{earlier} |p| ≤ (16i − 15)·Mj`, and its `u ≤ 2^-25·max(|t|, Mj) + 2^-151`.
* Each of an instruction's 17 terms loses less than `u`, and `|S'|` exceeds `|s j + c_j|` by at most
  the products still to come and the losses: a chunk's error is below
  `17·(1 + 17 + 33 + 49)·2^-25·Mj + 2^-23·(48 + 32 + 16)·Mj + 4·2^-23·|s j + c_j| + 21·2^-149` up to
  terms `2^-21` smaller, that is `2084·2^-25·Mj + 2^-21·|s j + c_j| + 21·2^-149`: the first arm keeps
  `2^-13 = 4096·2^-25` (a factor `1.97`), `2^-20` (2) and `2^-143` (3.05).
* A product loses at most `min(|p|, u)` and the accumulator less than `u`, with
  `u ≤ 2^-25·(Mj + Σ_{earlier} |p|) + 2^-151`: a chunk's error is also below
  `(1 + 2^-21 + 2^-23)·A_j + (1 + 2^-21)·2^-23·Mj + (1 + 2^-21)·2^-21·|s j + c_j| + 5·2^-149`:
  the second arm keeps `2·A_j` and `2^-22·Mj` (factors about 2). An accumulator that dominates its
  chunk is charged about `2^-22 + 2^-20` of itself per chunk of 64, as four chunks of 16 were charged
  before (round 32).

The room (`mag + relChunk16 K·mag + absChunk16 K` at most the `float32` maximum,
`mag = |acc| + Σ|a·b|`) is that of the fp8 facts with the 16-bit constants; on the device no partial
value exceeds `mag` (every term and the sum are truncated toward zero).

Until round 33 the chunks were 16 consecutive `k` with `2^-20`, `2^-22`, `2^-145`, `2^-24`: false
on `mma.sync` with `kWidth = 4` (library issue L150, round 32's review: a `K = 32` launch whose
device result no partial values of 16-chunks reach).
-/
namespace TVA.Rules.Num

open TVA.Math.Float

/-- The number of chunks of a 16-bit `tl.dot` with `K` products: `⌈K/64⌉`. -/
def chunks16 (K : ℕ) : ℕ := (K + 63) / 64

/-- Chunk `j` of `K` products: the `k < K` with `k / 64 = j`, the `64` consecutive `k` from
`64·j` (all `K` of them when `K < 64`: `K` is a power of two at least `16`). -/
def chunk16 (K j : ℕ) : Finset (Fin K) := Finset.univ.filter fun k => k.val / 64 = j

/-- The exact sum of chunk `j`: `Σ_{k ∈ chunk16 K j} va k · vb k`. -/
def chunkSum16 {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ := ∑ k ∈ chunk16 K j, va k * vb k

/-- The exact magnitude of chunk `j`: `Σ_{k ∈ chunk16 K j} |va k · vb k|` (not lifted). -/
def chunkAbs16 {K : ℕ} (va vb : Fin K → ℚ) (j : ℕ) : ℚ := ∑ k ∈ chunk16 K j, |va k * vb k|

/-- The chunk facts' relative constant on the largest term: `2^-13`. -/
def rhoChunk16 : ℚ := 2 ^ (-13 : ℤ)

/-- The chunk facts' relative constant on the new value: `2^-20`. -/
def sigmaChunk16 : ℚ := 2 ^ (-20 : ℤ)

/-- The chunk facts' absolute constant: `2^-143` per chunk. -/
def alphaChunk16 : ℚ := 2 ^ (-143 : ℤ)

/-- The loss bound's constant on the chunk's exact magnitude: `2`. -/
def kappaChunk16 : ℚ := 2

/-- The loss bound's constant on the largest term: `2^-22`. -/
def rhoLossChunk16 : ℚ := 2 ^ (-22 : ℤ)

/-- The relative constant of the room: `(1 + 2^-13)^⌈K/64⌉ − 1`. -/
def relChunk16 (K : ℕ) : ℚ := (1 + rhoChunk16) ^ chunks16 K - 1

/-- The absolute constant of the room: `⌈K/64⌉·(1 + 2^-13)^⌈K/64⌉·2^-143`. -/
def absChunk16 (K : ℕ) : ℚ := chunks16 K * (1 + rhoChunk16) ^ chunks16 K * alphaChunk16

variable (P : Platform)

/-- **A 16-bit `tl.dot` chunk by chunk**, per output element `(i, n)`. Suppose row `i` of `a` and
column `n` of `b` have values `va`, `vb` (`DType.val?`) and the accumulator element has value `vc`,
with room (`mag + relChunk16 K·mag + absChunk16 K` at most the accumulator's `DType.bound`,
`mag = |vc| + Σ|va·vb|`). Then the result element has a value `vr`, and there are partial values
`s 0 = vc, …, s (chunks16 K) = vr` such that, for each chunk `j < chunks16 K`, every `Mj` at least
`|s j|` and every `liftMag cfg.a (va k) · liftMag cfg.b (vb k)` of the chunk,
`|s (j+1) − (s j + c_j)| ≤ min (rho·Mj) (kappa·A_j + rho'·Mj) + sigma·|s j + c_j| + alpha`, with
`c_j = chunkSum16 va vb j` and `A_j = chunkAbs16 va vb j`. The chunks are added in increasing
order of `j`. -/
def DotChunk16Loss (cfg : DotCfg) (rho sigma alpha kappa rho' : ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤
        cfg.acc.bound →
      ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧ s (chunks16 K) = vr ∧
        ∀ j < chunks16 K, ∀ Mj : ℚ, |s j| ≤ Mj →
          (∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
          |s (j + 1) - (s j + chunkSum16 va vb j)| ≤
            min (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj) +
              sigma * |s j + chunkSum16 va vb j| + alpha

/-- `tl.dot` of 16-bit operands `a`, `b` into `float32`, `max_num_imprecise_acc = 0` (the default),
any `input_precision`, chunk by chunk with the constants above: each chunk of 64 consecutive `k`
within `min (2^-13·Mj) (2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| + 2^-143`. -/
def Dot16BitChunk (a b : DType) : Prop :=
  ∀ p : InputPrecision, DotChunk16Loss P ⟨a, b, .f32, p, 0⟩ rhoChunk16 sigmaChunk16 alphaChunk16
    kappaChunk16 rhoLossChunk16

/-- **`tl.dot` of `float16` operands into `float32`, chunk by chunk** (`Dot16BitChunk`): partial
values `acc = s 0, …, s ⌈K/64⌉ = result`, chunk `j` (the 64 `k` with `k / 64 = j`, in increasing
`j`) within `min (2^-13·Mj) (2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| + 2^-143` of `s j + c_j`, for
every `Mj` bounding `|s j|` and the chunk's lifted products (a subnormal operand counts as `2^-14`),
`A_j` the chunk's `Σ|a·b|`. At most four `mma.sync…m16n8k16` or `wgmma…m64nNk16` per chunk, each
adding 16 of its `k` (which ones depends on the operands' layout, above): the accumulator and 16
products aligned to the largest exponent field, 25 bits kept below it (truncated toward zero), the
sum truncated toward zero to `float32`. Evidence: `evidence/num-dot-chunk16-round33`,
`evidence/num-dot-chunk16`, `evidence/num-dot-sub16-round25`. -/
def DotF16Chunk : Prop := Dot16BitChunk P .f16 .f16

/-- **`tl.dot` of `bfloat16` operands into `float32`, chunk by chunk** (`Dot16BitChunk`, as
`DotF16Chunk`; a subnormal operand counts as `2^-126`). Evidence:
`evidence/num-dot-chunk16-round33`, `evidence/num-dot-chunk16`, `evidence/num-dot-sub16-round25`. -/
def DotBF16Chunk : Prop := Dot16BitChunk P .bf16 .bf16

/-! ## Weakening -/

section Mono
variable {P}

/-- `DotChunk16Loss` with constants at least as large. It states nothing new about the device. -/
theorem DotChunk16Loss.mono {cfg : DotCfg} {rho sigma alpha kappa rho' rho₂ sigma₂ alpha₂ kappa₂ rho'₂ : ℚ}
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

/-- The first arm alone: each step within `rho·Mj + sigma·|s j + c_j| + alpha`. -/
theorem DotChunk16Loss.step_left {cfg : DotCfg} {rho sigma alpha kappa rho' : ℚ}
    (h : DotChunk16Loss P cfg rho sigma alpha kappa rho') ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val)
    (b : Tensor [K, N] cfg.b.Val) (acc r : Tensor [M, N] cfg.acc.Val) (hd : P.dot cfg a b acc r)
    (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ)
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relChunk16 K * (|vc| + ∑ k, |va k * vb k|) + absChunk16 K) ≤
        cfg.acc.bound) :
    ∃ (s : ℕ → ℚ) (vr : ℚ), cfg.acc.val? (r.get2 i n) = some vr ∧ s 0 = vc ∧ s (chunks16 K) = vr ∧
      ∀ j < chunks16 K, ∀ Mj : ℚ, |s j| ≤ Mj →
        (∀ k ∈ chunk16 K j, liftMag cfg.a (va k) * liftMag cfg.b (vb k) ≤ Mj) →
        |s (j + 1) - (s j + chunkSum16 va vb j)| ≤ rho * Mj + sigma * |s j + chunkSum16 va vb j| + alpha := by
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := h a b acc r hd i n va vb vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, hn, fun j hj Mj hM hk => ?_⟩
  have e := hstep j hj Mj hM hk
  have := min_le_left (rho * Mj) (kappa * chunkAbs16 va vb j + rho' * Mj)
  linarith

end Mono

end TVA.Rules.Num
