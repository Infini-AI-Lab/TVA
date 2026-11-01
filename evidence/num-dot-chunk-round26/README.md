# An fp8 `tl.dot` chunk by chunk on sm_90: each chunk's loss bounded two ways

Library evidence (not a kernel's), for library issues L124 (round 26: a sharper per-chunk bound
for the fp8 chunk facts) and L128 (`input_precision` on fp8 operands). Triton 3.8.0 (TVA's
`.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-01. Script: [run.py](run.py);
output: [out.txt](out.txt) (run time about 2 minutes). It builds on
[evidence/num-dot-chunk](../num-dot-chunk/README.md), whose bit-level model of a `wgmma` chunk it
reuses.

**Question.**
1. Do the facts `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss` and
   `DotE5m2E4m3ChunkLoss` ([rules/num.md](../../docs/rules/num.md), "`tl.dot` of fp8, chunk by
   chunk") hold? They state, per output element, partial values `s 0 = acc, …, s n = result`
   (`n = ⌈K/32⌉`), one step per chunk of 32 consecutive `k` in increasing order, with

   `|s (j+1) − (s j + c_j)| ≤ min(2^-7·Mj, 2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126`

   for every `Mj` at least `|s j|` and every lifted product `L(a_k)·L(b_k)` of the chunk. Here `c_j`
   is the chunk's exact sum, `A_j = Σ_chunk |a_k·b_k|` its exact magnitude (not lifted), and `L(x)` is
   `0` for `x = 0` and `max(|x|, 2^emin)` otherwise (`2^-14` for `float8e5`, `2^-6` for
   `float8e4nv`). The first arm alone is the round-24 facts `DotE4m3Chunk`, …; the new facts state
   both arms of the same partial values.
2. Does `input_precision` change an fp8 `tl.dot`, for each of the four operand pairs?

**What Triton runs** (evidence/num-dot-chunk, checked again here in every case from the PTX). With
`M ≥ 64` and at least 4 warps, one `wgmma.mma_async…m64nNk32.f32` per chunk of 32. Otherwise Triton
converts each fp8 operand to `float16` (exact) and runs two `mma.sync…m16n8k16.f32.f16.f16.f32` per
chunk of 32.

**The model** (evidence/num-dot-chunk, equal to the device bit for bit on about `2·10^8` results
there, and on every `wgmma` element here). One `wgmma` chunk, per output element: the terms are the
accumulator `s` and the 32 exact products. Each nonzero term has an exponent field:
`max(⌊log₂|s|⌋, −126)` for the accumulator, `e(a) + e(b)` for a product, with
`e(x) = max(⌊log₂|x|⌋, emin)` (a subnormal operand reads as the smallest normal). With `E` the
largest field and `u = 2^(E−13)`, every term is truncated toward zero to a multiple of `u`, the
truncated terms are summed exactly, and the sum is truncated toward zero to 14 significant bits.

**Derivation** (the new constants from the model).
- **A product** truncated toward zero to a multiple of `u` loses at most `min(|p|, u)`: all of it
  when `|p| < u`, less than `u ≤ |p|` otherwise. **The accumulator** loses less than `u`. So the
  truncated sum `S'` is within `A_j + u` of `s j + c_j`.
- **The cut to 14 bits** loses less than `2^-13·|S'|` (the kept grid is `2^(⌊log₂|S'|⌋−13)`), and
  `|S'| ≤ |s j + c_j| + A_j + u`. So a chunk's error is below
  `(1 + 2^-13)·(A_j + u) + 2^-13·|s j + c_j|`.
- **The unit against `Mj`.** `2^e(x) ≤ L(x)` for `x ≠ 0`, so a product's `2^(e(a)+e(b))` is at most
  its lifted product, hence at most `Mj`; the accumulator's `2^max(⌊log₂|s|⌋, −126)` is at most
  `max(|s|, 2^-126) ≤ Mj + 2^-126`. So `u ≤ 2^-13·Mj + 2^-139`.
- **So a chunk's error is below**
  `(1 + 2^-13)·A_j + (1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| + (1 + 2^-13)·2^-139`.
  The facts keep `2·A_j` (a factor `2/(1 + 2^-13) = 1.9998`), `2^-12·Mj` (`1.9998`),
  `2^-12·|s j + c_j|` (`2`) and `2^-126` (about `8191`). The first arm, `2^-7·Mj`, is the round-24
  derivation's (evidence/num-dot-chunk, conclusion 2), on the same partial values.
- **The worst case of the second arm** is an accumulator that sets the window with every bit below
  `u` set and a magnitude just above `2^E` (so `Mj` is smallest against `u`), and 32 products of its
  sign just below `u`, each lost whole: an error of about `33u` against `2·32u + 2u + 2u`, a ratio
  of about `0.485` (kind `lossacc` below). A product that sets the window has `|p| ≥ 2^(E−3)` (a
  subnormal operand is at least `2^-3` of the smallest normal), so there `2·A_j` alone is far above
  `u`.
- **`mma.sync`** (two `float16` instructions per chunk, each aligned to its largest exponent field
  with 25 bits below it, terms and sum truncated toward zero:
  [evidence/num-dot-sub16-round25](../num-dot-sub16-round25/README.md)). Each instruction's error is
  below the products' magnitudes plus `2^-25` of its largest lifted term (at most `Mj` for the
  first, `Mj + A_j` for the second, whose accumulator is at most `|s j| + A_j`), plus
  `2^-23·(|s j| + A_j)` and `2^-149` for the cut to `float32`. In all, below
  `(1 + 2^-25 + 2^-22)·A_j + (2^-24 + 2^-22)·Mj + 2·2^-151 + 2·2^-149`, far inside the second arm.
- **No growth.** Every term and the sum are truncated toward zero on both paths, so no partial value
  exceeds `|acc| + Σ_{i<j} A_i` in magnitude: the device's own error grows at most linearly in the
  chunks. The facts do not state this; the evidence checks the result against it (`magnitude`).

**Method.**
- **Part P, `input_precision`** (L128). For each operand pair, on two `wgmma` blocks (`64 × 64`, 4
  warps; `128 × 64`, 8 warps) and two `mma.sync` blocks (`16 × 16`, 1 warp; `32 × 32`, 2 warps),
  `K = 64`: the PTX of `input_precision` `tf32`, `tf32x3`, `ieee`, `bf16x3` and `bf16x6` against
  that of the default (`None`), and the default's against a kernel that omits the argument (both
  without their debug line information, which names source lines); and the results, bit for bit,
  on one draw each of `normal`, `sub`, `subwin` and `bits` inputs.
- **Part S, single chunks on the device** (`K = 32`, both paths). With one chunk the partial values
  are the accumulator and the device's result, so each element is checked against the step bound
  with no model, at the smallest `Mj` (the larger of `|acc|` and the chunk's largest lifted
  product). Blocks: `64 × 64` (4 warps), `128 × 64` (8 warps), and `128 × 64` (4 warps) with `b`
  transposed in the kernel and with `a` converted to fp8 in registers, as tutorial 06's `_attn_fwd`
  does (`wgmma`); `16 × 16` (1 warp), `32 × 32` (2 warps) and `16 × 64` (4 warps) (`mma.sync`).
  Eighteen kinds of inputs, two draws each: evidence/num-dot-chunk's thirteen (`normal`, `spread`,
  `accbig`, `allones`, `onebig`, `cancel`, `tiny`, `huge`, `sub`, `subwin`, `winworst`, `trunc14`,
  `bits`) and five where the accumulator sets the window: `accone` (an accumulator
  `(2 − 2^-23)·2^e` and 32 products of its sign just below `u`, the case the library review of
  round 24 measured), `accdom` (products just below `u/2`, random signs), `accsmall` (products far
  below the accumulator, as in attention after many blocks), `lossacc` (the worst case above) and
  `lossmix` (as `lossacc`, random signs, a quarter of the products zero). Each row reports the
  largest error over the new bound, over its second arm alone, and over the round-24 bound.
- **Part B, every `K` that compiles** (64 doubling up to 2048 on `64 × 64` with 4 warps, 512 on
  `128 × 128` with 8, 2048 on `16 × 16` with 1, 1024 on `32 × 32` with 2; Triton rejects `K = 2048`
  on `64 × 64` on the host, "out of resources", recorded). All eighteen kinds, one draw each. On
  `wgmma`, the model against the device bit for bit, and each chunk of the model's chain against the
  step bound. On both paths, every element against the interval the facts imply: the step bound
  composed over the chunks with the exact partial sums `S j = acc + Σ_{i<j} c_i` and the worst case
  of the unknown partial values. The step `s ↦ s + c_j + B(s)` is increasing in `s` (`B` moves by at
  most `2^-7 + 2^-12 < 1` per unit of `s`, and a minimum of increasing functions is increasing), so
  the interval is exact: `hi (j+1) = hi j + c_j + B(hi j)`, and likewise `lo`. `half-width`: the new
  composed bound over the round-24 one.
- **Part C, chains** as tutorial 06's `_attn_fwd`:
  `for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t.to(fp8), v_t, acc)`, `p_t` `[M, 64]`
  converted in registers, `v_t` `[64, N]`, every `acc_t` stored; `128 × 64` and `128 × 128` with 4
  warps (`wgmma`), `32 × 64` with 2 (`mma.sync`). Inputs: `softmax` (`p` the `exp2` of random scores
  minus the row maximum, scale 1, 4 or 16; `v` from `randn`; `T` up to 256, a `K` of 16384),
  `accdom` (after a `randn·2^6` accumulator, products just below the accumulator's window unit) and
  `subwin`; with and without a rescale `alpha_t ∈ [0.25, 1)`. Each dot as in Part B, from the stored
  accumulator before it (times `alpha_t` in `float32`); without the rescale, the final result against
  the facts composed over all `2T` chunks from the initial accumulator, and the model run over the
  concatenated operands.
- **Arithmetic.** `float64` tensors: the products of fp8 values are exact, a sum of `n` terms is
  within `n·2^-53` of their magnitudes' sum, and every step bound is at least `2^-12·Mj`, with `Mj` at
  least each term, so a step ratio computed in `float64` is within `2^-30` of the exact one, and an
  interval ratio within `10^-4` for the longest chains. The largest element of every row, and any
  element above `0.9` of a bound, is recomputed in exact rationals.

**Result** (`out.txt`).

Part P: for every operand pair and block, every `input_precision` gives the default's PTX, the
default's PTX equals that of a kernel that omits the argument, and every result equals the default's
bit for bit (80 rows).

Totals (every row is in `out.txt`; `new`: the largest error over the new facts' bound, the step
bound in Part S and the interval in Parts B and C; `chunk`: each chunk of the model's chain over the
new step bound; `half-width`: the smallest ratio of the new composed bound to the round-24 one):

| Operands | Path | Part | Elements | Outside | Model = device | New | Round 24 | Chunk | Half-width |
|---|---|---|---|---|---|---|---|---|---|
| `float8e4nv × float8e4nv` | `wgmma` | S | 1032192 | 0 | — | 0.4843 | 0.4696 | — | 0.0609 |
| `float8e4nv × float8e4nv` | `wgmma` | B | 1548288 | 0 | all | 0.4872 | 0.4855 | 0.4843 | 0.0615 |
| `float8e4nv × float8e4nv` | `wgmma` | C, per dot | 18210816 | 0 | all | 0.4591 | 0.4591 | 0.4677 | 0.0604 |
| `float8e4nv × float8e4nv` | `wgmma` | C, whole chain | 221184 | 0 | all | 0.2191 | 0.2191 | — | 0.319 |
| `float8e5 × float8e5` | `wgmma` | S | 1032192 | 0 | — | 0.4833 | 0.4393 | — | 0.0632 |
| `float8e5 × float8e5` | `wgmma` | B | 1548288 | 0 | all | 0.4750 | 0.4548 | 0.4833 | 0.0685 |
| `float8e5 × float8e5` | `wgmma` | C, per dot | 18210816 | 0 | all | 0.4265 | 0.4265 | 0.4358 | 0.0604 |
| `float8e5 × float8e5` | `wgmma` | C, whole chain | 221184 | 0 | all | 0.1333 | 0.1333 | — | 0.294 |
| `float8e4nv × float8e5` | `wgmma` | S, B, C | 21012480 | 0 | all of B and C | 0.4872 | 0.4855 | 0.4843 | 0.0605 |
| `float8e5 × float8e4nv` | `wgmma` | S, B, C | 21012480 | 0 | all of B and C | 0.4833 | 0.4548 | 0.4833 | 0.0604 |
| each pair | `mma.sync` | S, B, C | 1738752 | 0 | — | at most 0.0005 | at most 0.0002 | — | 0.0574 |

The largest ratio to the new bound, `0.4872`, is a whole dot of kind `lossacc` at `K = 1024`
(`float8e4nv` left operand, `64 × 64`, 4 warps). Per chunk it is `0.4843` with a `float8e4nv` left
operand (Part S, kind `lossacc`, `128 × 64` with `a` converted in registers: accumulator
`262175.96875 = 2^18 + 2^5 − 2^-5`, window unit `u = 2^5`, 32 products `30 = 2^5·(1 − 2^-4)` lost
whole, exact `263135.96875`, device `262144`), and `0.4833` with a `float8e5` one. Kind `accone`
reaches `0.4558` (the round-24 bound: `0.2348`; on such inputs the round-24 library review measured
`0.82` of its `B_alt`, which has `1 + 2^-13` where the facts keep `2`). No result exceeds
`|acc| + Σ|a·b|` in magnitude (`magnitude`, every row).

The sharpening on chains: for tutorial 06's softmax chains at `T = 256`, the new composed bound is
`0.29` to `0.78` of the round-24 one, element by element; for the `accdom` chains at `T = 64`,
`0.35` to `1`. Where a chunk's products are far below the accumulator,
the second arm is the smaller and the composed bound grows by `1 + 2^-11` per chunk instead of
`1 + 2^-7 + 2^-12`; elsewhere the first arm is, and the bound is the round-24 one.

**Conclusions.**
1. **The facts hold** (`DotE4m3ChunkLoss`, …) on every element tested, on both paths, for every pair,
   every `K` that compiles and the chains (about `9.1·10^7` elements): no error above `0.49` of the
   bound, as the derivation's factor of 2 predicts.
2. **The model is exact** on every `wgmma` element here, chains included, as in evidence/num-dot-chunk.
3. **`input_precision` changes nothing** for fp8 operands, for each of the four pairs, on both
   paths: the PTX is the same for every value and for an omitted argument, and the results are the
   same bit for bit. So the fp8 facts' `∀ p` holds for the mixed pairs too (evidence/num-dot Part 3
   compared only `float8e4nv × float8e4nv` and `float8e5 × float8e5`, bit for bit on random inputs).
4. **The device does not compound**: its partial values stay within `|acc| + Σ|a·b|`, so its error
   grows at most linearly in the chunks. The round-24 facts' bound compounds because each step is
   relative to `Mj ≥ |s j|`; the new facts' second arm charges a dominating accumulator only
   `2^-12 + 2^-12` of itself per chunk.
