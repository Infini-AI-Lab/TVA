# A 16-bit `tl.dot` on sm_90, a subnormal operand included

Library evidence (not a kernel's), for library issue L126 (round 25, the user's decision of
2026-11-01: "please fix L126 after the current run"). Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-01. Script: [run.py](run.py); output:
[out.txt](out.txt) (run time about 7 minutes). The coordinator's inputs that found the failure are in
[evidence/num-dot-sub16](../num-dot-sub16/README.md), kept as they are.

**Question.** How does sm_90 accumulate a `float16` × `float16` and a `bfloat16` × `bfloat16`
`tl.dot` into `float32` (`max_num_imprecise_acc = 0`, Triton's default for these operands), a
subnormal operand included, and do the facts `DotF16` and `DotBF16` as restated in round 25
([rules/num.md](../../docs/rules/num.md), "`tl.dot` (the chunk model)") hold? They state, per
output element with values `a_k`, `b_k`, `acc`, for every `input_precision`,

`|r − (acc + Σ a_k·b_k)| ≤ relDot16 K·L + absDot16 K`, with `L = |acc| + Σ liftMag a_k·liftMag b_k`,

`relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`, when `L + relDot16 K·L + absDot16 K` is at
most the `float32` maximum. `liftMag x` is `0` for `x = 0` and `max(|x|, 2^emin)` otherwise, `2^emin`
the smallest normal (`2^-14` for `float16`, `2^-126` for `bfloat16`). Until round 25 the facts said
the same with `Σ|a_k·b_k|` in place of the lifted sum (the "old bound" below). Also: how wide is the
window, measured against the field a subnormal operand reads as; are the accumulator and the
products aligned to it as for fp8 (evidence/num-dot-chunk); in what order are the chunks added; and
does any other listed fact have the same gap?

**What Triton runs** (the PTX, Part A lists it). With `M ≥ 64` and at least 4 warps,
`wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16` (`.bf16.bf16`), one per chunk of 16 along `K`,
each from the accumulator the previous one left; otherwise
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (`.bf16.bf16`), one per chunk of 16.

**The model** (the module docstring of `run.py`; Part A establishes each rule, Parts B and D check
it). One instruction, per output element: the terms are the accumulator `s` and the 16 exact products
of the chunk. Each nonzero term has an exponent field: `e(a) + e(b)` for a product, with
`e(x) = max(⌊log₂|x|⌋, emin)` (a subnormal operand reads as the smallest normal), and
`max(⌊log₂|s|⌋, −126)` for the accumulator. With `E` the largest field and `u = 2^(E−25)`, every term
is truncated toward zero to a multiple of `u`, the truncated terms are summed exactly, and the sum is
truncated toward zero to `float32` (24 significant bits; the grid `2^-149` below `2^-126`). The
instructions run in increasing order of the chunks, 16 consecutive `k` each. Both paths compute the
same.

**Method.**
- **Part A, the window.** Single elements of a `K = 16` dot with chosen products and accumulators, on
  both paths and both formats, each against the model in exact rationals: the window's width against
  a product's field (`1 − 1 + 2^-25` keeps `2^-25`, `2^-26` is lost; the same for `1.5·1.5` and for
  the largest significands' product, whose fields are `0` though their values reach almost `4`),
  truncation toward zero of each term, of either sign, and no partial sums (14 terms `u/2` are all
  lost), the accumulator in the same window with its own field (`1` and `3`), the final truncation
  toward zero to `float32` (`1 + 3·2^-25` gives `1`), a subnormal operand read as the smallest normal,
  in `a` and in `b` (`float16` `2^-23·2^14`: `E = 0`, `2^-28` lost, where the same product from
  normal operands keeps it; `bfloat16` `2^-133·2^124`: `E = −2`), L126's two cases, and subnormal
  accumulators and results (`bfloat16`: an accumulator's field is `−126`, subnormal results are kept
  on the grid `2^-149`, a product `2^-200` gives `0`).
- **Parts B and C, the model and the facts.** Both formats; the seven block shapes of
  evidence/num-dot Part 2 (`16 × 16` with 1 warp to `128 × 128` with 8); `K` from 16 doubling to the
  largest that compiles (2048 on `16 × 16`, 256 on `128 × 128`; Triton rejects the next on the host,
  "out of resources", recorded). Fifteen kinds of inputs: `normal` (`randn` cast to the format, as
  the tutorials' inputs); `softmax` (`exp2` of random scores minus the row maximum, cast: tutorial
  06's `p.to(float16)`, many subnormal and zero values); `spread`; `accbig`; `allones` (same-sign
  products with every fraction bit set against an accumulator with every bit set); `onebig`;
  `cancel`; `tiny` (multiples of the smallest subnormals, subnormal accumulators); `huge`; `sub` (many
  subnormal operands on both sides); `subwin` (per chunk one product of a subnormal operand, the
  smallest or a random one, in `a` or in `b`, and a large power of two, which sets the field `E`, the
  15 other products just below `u = 2^(E−25)` with every fraction bit set, so they are lost whole,
  every product of one sign per row, and an accumulator `0`, just below `u`, or near the large
  product's value, of either sign); `winworst` (the same with a normal large product: the window's
  worst case relative to `|a·b|`); `subrand` (random subnormal operands against random large partners
  among random products near the window's unit: a randomized sweep); `trunc24` (sums just above a
  power of two with bits just below the result's 24-bit unit); `bits` (every finite value of the
  format, by bit pattern). Each kind is drawn 16 times per configuration when it has subnormal
  operands or the window's worst case (`softmax`, `tiny`, `sub`, `subwin`, `winworst`, `subrand`,
  `bits`), 6 times otherwise. Every element is compared with the model (`model = device`, as
  values) and checked against the new bound and the old one. The exact sum and `L` are in `float64`
  where it holds every partial sum exactly (all terms multiples of `2^q` with their magnitudes' sum
  below `2^(q+53)`, checked per element); elsewhere `(K+2)·2^-52·Σ|terms|` is added to the error and
  `L` is reduced by the same factor, so the ratio is an upper bound; the largest element of each run,
  and every element above `0.9` of the bound, is recomputed in exact rationals. Each chunk of the
  model's chain is also checked against the derivation's per-chunk bound (below; `chunk ratio`).
- **Part D, the kernels' shapes.** D1: tutorial 06's `_attn_fwd` and `_attn_bwd` (`[128, D] × [D, 64]`
  with `b` transposed in the kernel, as `k.T`, with and without an accumulator; `[128, 64] × [64, D]`
  with `a` converted to the format in registers, as `p.to(…)`), `D ∈ {64, 128, 256}`, and tutorials
  03 and 08's tiles (`128 × 128 × 64`, `128 × 256 × 64`), each with 4 and 8 warps; the kinds and
  checks of Parts B and C. D2, chains: `for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t,
  v_t, acc)`, every `acc_t` stored (the result without the stores checked equal), with `p_t`
  converted in registers from a softmax (tutorial 06) or `subwin` blocks, and with `p_t` loaded in the
  format (tutorial 03's `K` loop) from `normal` or `subwin` blocks; `128 × {64, 128, 256}` with 4
  warps, `128 × {64, 128}` with 8, `32 × 64` with 2 (`mma.sync`); `T` up to 256 (a `K` of 16384); with
  and without a rescale `alpha_t ∈ [0.25, 1)`. Each dot against the facts and the model from the
  stored accumulator before it; without the rescale, the final result against the facts for one dot
  of `K = 64·T` and against the model run over the concatenated operands.
- **Part E, the order of the chunks.** For chunks `x < y` and `k`, all distinct: `+B` in chunk `x`,
  `−B` in chunk `y`, a small product `s` in chunk `k`, `B/s = 2^28`. `s` survives exactly when chunk
  `k` is added after both. 8 chunks (all 168 triples) and 32 chunks (3000 sampled), both paths, both
  formats.
- **Part F, L126's inputs.** evidence/num-dot-sub16's `probe.py` cases at `K = 16` (the large product
  at `k = 0, 7, 15`), and repeated at `K = 64` and `256`: against the old and the new bound, and the
  model.
- **Part G, other facts.** G1: every `input_precision` (`tf32x3`, `ieee`, `bf16x3`, `bf16x6`)
  against the default, bit for bit, on the kinds with subnormal operands, both paths, `K ∈ {16, 64,
  256}`. G2: the fp8 chunk facts on `mma.sync`, where Triton converts fp8 to `float16` (a subnormal
  `float8e5` value is a subnormal `float16` value): per chunk of 16 one product of a subnormal
  `float8e5` operand and `2^15`, the others just below that `float16` window's unit, against
  `DotE5m2Chunk`'s implied interval (as evidence/num-dot-chunk computes it). G3: `tl.dot_scaled`
  (`"e4m3"` by `"e4m3"`, both `fast_math`), whose emulated `bfloat16` dot has only normal operands on
  `scaledDomBF16`, on window inputs (per chunk one product `448·448`, 15 products of normal and
  subnormal `e4m3` elements lost below the window), against `DotScaledE4m3`'s bound.

**Result** (`out.txt`).

Part A: every device result equals the model (both paths, both formats). Part E: on both paths and
in both formats every outcome is that of increasing `k`, chunk by chunk (168 of 168, 3000 of 3000).

Parts B and C (totals; every row is in `out.txt`):

| Operands | Path | Elements in range | Outside the new bound | Model = device | Largest new ratio | Largest chunk ratio | Outside the old bound | Largest old ratio |
|---|---|---|---|---|---|---|---|---|
| `float16` | `mma.sync` | 2621440 | 0 | all | 0.1111 | 0.8456 | 33404 | 113.67 |
| `float16` | `wgmma` | 32768000 | 0 | all | 0.1111 | 0.8456 | 489494 | 113.67 |
| `bfloat16` | `mma.sync` | 2621440 | 0 | all | 0.1107 | 0.9020 | 60832 | 14.17 |
| `bfloat16` | `wgmma` | 32768000 | 0 | all | 0.1107 | 0.9020 | 908610 | 14.17 |

Every element outside the old bound is of kind `subwin`, except 49 of kind `subrand` (`float16`): the
randomized inputs reach it too. The largest new ratio is reached by `winworst` at `K = 16` (normal
operands: a large product, 15 products just below `u`, lost whole); `subwin` stays below it, since a
subnormal product's lifted magnitude is the window's own scale.

Part D. D1, the kernels' single dots (all on `wgmma`): 49807360 elements per format, none outside the
new bound, every one equal to the model; the largest new ratio is `0.0897` (`float16`, the
`128 × 128 × 64` tile, `accbig`) and `0.0868` (`bfloat16`); the old bound fails on 483328 (`float16`,
up to `30.1` times) and 1574888 (`bfloat16`, up to `3.87` times) elements, all of kind `subwin`. D2,
264 chains: every dot of every chain equals the model (169951232 element results), and so does each
whole chain run as one dot (3022848); none is outside the new bound, the largest ratio is `0.0489`
per dot (a softmax chain) and `0.0292` for a whole chain. The softmax chains (tutorial 06's
`p.to(float16)`) stay inside the old bound too (at most `0.0489`): a subnormal `p`'s lifted product,
at most `2^-14·|v|`, is far below the row's largest product, `1·|v|`, which sets the window. The `subwin` chains exceed the old bound up to
`29.1` times per dot.

Part F: L126's `float16` case is at `53.181` times the old bound and `0.1040` of the new one, its
`bfloat16` case at `13.122` and `0.1025`, on both paths and at every position; repeated at `K = 64`
and `256` the old bound fails less (`14.5`, `3.71` for `float16`) and the new bound is at `0.028` and
`0.007`. The normal-operand versions of the same products are at most `0.036` of either bound. The
model equals every device result.

Part G: G1, every `input_precision` gives the default's bits on every element (48 configurations).
G2, the fp8 chunk facts on `mma.sync` with a subnormal `float8e5` operand: the products below the
`float16` window are lost (relative error about `10^-6`), at most `0.0001` of the facts' bound. G3,
`tl.dot_scaled` on window inputs: at most `0.0057` of `DotScaledE4m3`'s bound, every scaled value in
`scaledDomBF16`.

Every zero result equal to the model's (577048 of them, all parts) has the model's sign too: the model
is exact bit for bit.

**Derivation** (the constants from the model; `docs/rules/num.md` gives it too).
- **A chunk.** Each of the at most 17 nonzero terms loses less than `u` and no more than its own
  magnitude (truncation toward zero), so the truncated sum `S'` is within `17u` of `s + c` (`c` the
  chunk's exact sum) and `|S'| ≤ |s| + Σ_chunk|a·b|`. The cut to `float32` loses less than
  `2^(⌊log₂|S'|⌋−23) ≤ 2^-23·|S'|` for a normal `S'` and less than `2^-149` below `2^-126`, so less than
  `2^-23·|S'| + 2^-149`.
- **`u` against lifted magnitudes.** `2^e(x) = max(2^⌊log₂|x|⌋, 2^emin) ≤ max(|x|, 2^emin) = liftMag x`
  for `x ≠ 0`, so a product's `2^(e(a)+e(b)) ≤ liftMag a·liftMag b`; the accumulator's
  `2^max(⌊log₂|s|⌋, −126) ≤ max(|s|, 2^-126)`. So `u ≤ 2^-25·Mj + 2^-151`, with `Mj` the larger of `|s|`
  and the chunk's largest `liftMag a·liftMag b`, and `Mj ≤ |s| + A`, `A` the chunk's sum of
  `liftMag a·liftMag b` (at least `Σ_chunk|a·b|`).
- **The error of a chunk** is below `17·(2^-25·(|s| + A) + 2^-151) + 2^-23·(|s| + A) + 2^-149
  = 5.25·2^-23·(|s| + A) + 5.25·2^-149` (the `chunk ratio` column checks each chunk of the model's chain
  against this: at most `0.902`).
- **No growth.** Truncation toward zero never increases a magnitude, so the accumulator before chunk
  `j` is at most `|acc| + Σ_{i<j} Σ_chunk|a·b|`, and `|s| + A_j ≤ L = |acc| + Σ liftMag a·liftMag b`.
- **The whole dot.** The result's error is the sum of its `⌈K/16⌉` chunks' errors, below
  `⌈K/16⌉·(5.25·2^-23·L + 5.25·2^-149)`. Since `⌈K/16⌉ ≤ (K+15)/16` and `5.25·(K+15)/16 ≤ 2·(K+2)` for
  every `K ≥ 1`, this is below `relDot16 K·L + absDot16 K` (`relDot16 K = 2·(K+2)·2^-23`): a factor
  `6.9` at `K = 16` and about `6.1` for large `K`, on the relative term; the absolute term is far
  larger than `⌈K/16⌉·5.25·2^-149`. No `K` is excluded. The measured ratios (at most `0.111`) agree
  with these margins: the derivation allows at most `5.25/36 = 0.146` of the bound at `K = 16` and
  `5.25/32 = 0.164` for large `K` (`K` a multiple of 16).
- **Room.** Every partial accumulator and every `|S'|` is at most `|acc| + Σ|a·b| ≤ L`, so with room
  nothing overflows and the result is finite.

**Conclusions.**
1. **The model is exact** on both paths and in both formats: every element of every kind, shape, `K`
   and chain equals it. The window keeps **25 bits below the largest exponent field**, the
   accumulator included and aligned to the same window, each term truncated toward zero; the sum is
   truncated toward zero to `float32`. num-dot Part 1's "at least `2^-23` of the largest term" was the
   final `float32` cut seen from normal operands.
2. **A subnormal operand reads as the smallest normal** (`2^-14` for `float16`, `2^-126` for
   `bfloat16`), in `a` and in `b`, on both paths. So the old facts (relative to `Σ|a·b|`) fail, up to
   `113.7` times for `float16` and `14.2` times for `bfloat16` (`subwin`, `K = 16`), and the restated
   facts (relative to `Σ liftMag a·liftMag b`) hold on every element, about `3.4·10^8` element results
   with the chains, at most `0.111` of the bound; the derivation above extends them to every `K`.
3. **The order** is increasing `k`, 16 consecutive `k` per instruction, on both paths.
4. **`input_precision`** changes nothing for these operands, subnormal ones included: the facts hold
   for every value.
5. **Other facts.** The fp8 chunk facts hold on `mma.sync` with a subnormal `float8e5` operand: the
   `float16` instructions lose the products below their window, as the 16-bit facts say, but the fp8
   bound charges `2^-7` of the chunk's largest lifted term, far above (G2). `DotScaledE4m3` and
   `DotScaledE4m3E2m1` ask for scaled values in `scaledDomBF16` (`0` or a normal `bfloat16`), so their
   `bfloat16` dot has no subnormal operand and its lifted magnitudes are `|a·b|`: no gap (G3).
   `DotF32Ieee` uses no tensor core (`fma.rn.f32`, which keeps subnormals: evidence/num-dot Part 4,
   bit for bit, `tiny` inputs included), so no window. A `tf32` dot is not stated by the library.
