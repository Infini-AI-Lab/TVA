# An fp8 `tl.dot` chunk by chunk on sm_90

Library evidence (not a kernel's), for library issue L124 (the user's request of 2026-10-31,
"please tighten the fp8 kernel error bound"). Triton 3.8.0 (TVA's `.venv`, torch 2.11.0+cu130),
NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-31. Script: [run.py](run.py); output: [out.txt](out.txt)
(run time about 21 minutes).

**Question.** How does sm_90 accumulate an fp8 × fp8 `tl.dot` into `float32` with Triton's default
`max_num_imprecise_acc` (`2^30`), and do the facts `DotE4m3Chunk`, `DotE5m2Chunk`,
`DotE4m3E5m2Chunk` and `DotE5m2E4m3Chunk` ([rules/num.md](../../docs/rules/num.md), "`tl.dot` of
fp8, chunk by chunk") hold? They state, per output element, partial values
`s 0 = acc, …, s n = result` (`n = ⌈K/32⌉`), one step per chunk of 32 consecutive `k` (chunk `j`:
`32j ≤ k < 32j + 32`), in increasing order of `j`, with

`|s (j+1) − (s j + c_j)| ≤ 2^-7·Mj + 2^-12·|s j + c_j| + 2^-126`

for every `Mj` at least `|s j|` and every lifted product `L(a_k)·L(b_k)` of the chunk, where `c_j` is
the chunk's exact sum and `L(x)` is `0` for `x = 0` and `max(|x|, 2^emin)` otherwise (`2^-14` for
`float8e5`, `2^-6` for `float8e4nv`). In what order are the chunks added? Are the facts `DotE4m3` …
(`relDot8`), which the library stated when this ran, true on the same inputs (they are not, and were
withdrawn: conclusion 6)?

**What Triton runs** (the PTX, `run.py` checks it in every case). With `M ≥ 64` and at least 4
warps, one `wgmma.mma_async.sync.aligned.m64nNk32.f32.<a>.<b>` per chunk of 32 along `K`, each from
the accumulator the previous one left. Otherwise Triton converts each fp8 operand to `float16`
(`cvt.rn.f16x2.e5m2x2`, `cvt.rn.f16x2.e4m3x2`, exact) and runs
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, two per chunk of 32. An fp8 `tl.dot` with
`K = 16` is rejected on the host, so `K` is a power of two of at least 32. `input_precision` makes no
difference for fp8 operands, so the script uses the default. [evidence/num-dot](../num-dot/README.md)
Part 3 compared the results bit for bit for `float8e4nv × float8e4nv` and `float8e5 × float8e5` only
(random inputs, both paths). For all four pairs, the two mixed ones included, on both paths, every
`input_precision` gives the default's PTX and the default's bits:
[evidence/num-dot-chunk-loss](../num-dot-chunk-loss/README.md) Part P (round 26, library issue
L128).

**The model** (the module docstring of `run.py`). One `wgmma` chunk, per output element: the terms
are the accumulator `s` and the 32 exact products. Each nonzero term has an exponent field:
`max(⌊log₂|s|⌋, −126)` for the accumulator, `e(a) + e(b)` for a product, with
`e(x) = max(⌊log₂|x|⌋, emin)` (a subnormal operand reads as the smallest normal). With `E` the
largest field and `u = 2^(E−13)`, every term is truncated toward zero to a multiple of `u`, the
truncated terms are summed exactly, and the sum is truncated toward zero to 14 significant bits.

**Method.**
- **Part A, the window.** Single elements with chosen products and accumulators, against the model,
  to establish each of its rules: the unit `u` against the largest term (a product or the
  accumulator), truncation toward zero, one window for the whole chunk (no partial sums: 31 terms
  `u/2` against a term `1` are all lost; `1 − 1 + 2^-16` loses `2^-16`), the accumulator in the same
  window and itself cut, the final cut to 14 bits (`1 + 1 + 2^-13` gives `2`), a subnormal operand read
  as the smallest normal (the product `2^-16 · 2^15 = 2^-1` of `float8e5` values sets `u = 2^-12`,
  the same product of normal values `u = 2^-14`; likewise for `float8e4nv` and the mixed pairs), and
  subnormal accumulators (`e = −126`).
- **Part A2, the `mma.sync` path.** The accumulator alone passes unchanged (subnormals included), a
  term `2^-23` of the largest is kept and `2^-24` is not, the smallest products are kept.
- **Part B, the model against the device, bit for bit** (`wgmma`): every element of Part C's `wgmma`
  runs.
- **Part C, the facts.** Every operand pair; the seven block shapes of evidence/num-dot Part 2
  (`16 × 16` with 1 warp to `128 × 128` with 8), `K` from 32 doubling to the largest that compiles
  (8192 on `16 × 16`; Triton rejects the rest on the host, "out of resources", recorded); and the
  shapes of tutorial 06's `_attn_fwd` with `num_warps = 4`: `[128, D] × [D, 64]` for
  `D ∈ {64, 128, 256}` with `b` transposed in the kernel (`k.T`), with and without an accumulator, and
  `[128, 64] × [64, D]` with `a` converted to fp8 in registers (`p.to(…)`). Thirteen kinds of
  inputs, two draws each: `normal`; `spread` (operands over 8 binades); `accbig` (an accumulator with
  all bits set, up to `2^20`); `allones` (same-sign products with all fraction bits set against an
  accumulator with all bits set); `onebig` (one large term per chunk among small ones); `cancel`
  (pairwise cancellation, exact or near); `tiny` (multiples of the smallest subnormals, subnormal
  accumulators); `huge` (operands near the formats' maxima); `sub` (many subnormal operands);
  `subwin` and `winworst` (the window's worst case: per chunk the largest exponent field on one
  product, with a subnormal operand or a power of two, the other 31 products just below `u` with all
  fraction bits set, of one sign, and an accumulator just below `u` or near the large product, of
  either sign); `trunc14` (sums just above a power of two with bits just below the result's 14-bit
  unit); `bits` (every finite value of the formats, by bit pattern; on blocks up to `32 × 32` and `K`
  up to 1024). Every element is checked against the interval the facts imply for it: the per-chunk
  bound composed over the chunks, with the exact partial sums `S j = acc + Σ_{i<j} c_i` and the worst
  case of the unknown partial values. The step is monotone in `s j`, so the interval is exact:
  `hi (j+1) = hi j + c_j + 2^-7·max(|hi j|, m_j) + 2^-12·|hi j + c_j| + 2^-126` and likewise `lo`, with
  `m_j` the chunk's largest lifted product. `ratio` is the error over the half-width of the interval on
  its side (1 is on the bound), also against the closed form `errB` of
  `TVA.Lemmas.Dot8Chunk` (`ratio (closed form)`), and, on `wgmma`, each chunk of the model's chain
  against the per-chunk bound (`chunk ratio`). Exact values: `float64` where it holds every partial
  sum exactly (all terms multiples of `2^q` and their magnitudes' sum below `2^(q+53)`, checked per
  element), and otherwise rationals (`exact_ratio`: the products as integers, the rest as
  fractions); every element within `0.01` of the bound, and the largest of each run, is recomputed in
  rationals. The existing facts' bound `relDot8 K·(|acc| + Σ|a·b|) + absDot8 K` is checked too.
- **Part D, chains.** `for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t.to(fp8), v_t, acc)`
  as `_attn_fwd`'s loop does, `p_t` `[M, 64]` converted in registers, `v_t` `[64, N]`, every `acc_t`
  stored (the result without the stores is checked equal): `128 × {64, 128, 256}` with 4 warps
  (`wgmma`) and `32 × 64` with 2 (`mma.sync`); `p` from a softmax of random scores (`exp2` of scores
  minus the row maximum, scale 1, 4 or 16), or `allones`, or `subwin` blocks; `T` up to 256 (a `K` of
  16384); with and without a rescale `alpha_t ∈ [0.25, 1)`. Each dot against the facts and the model
  from the stored accumulator before it (times `alpha_t` in `float32`), and, without the rescale, the
  final result against the facts composed over all `2T` chunks from the initial accumulator and against
  the model run over the concatenated operands.
- **Part E, the order of the chunks.** For chunks `x < y` and `k`, all distinct: `+B` at a random
  position of chunk `x`, `−B` in chunk `y`, a small product `s` in chunk `k`, `B/s` beyond the window.
  `s` survives exactly when chunk `k` is added after both `x` and `y`. 8 chunks (all 168 triples) and
  32 chunks (3000 sampled triples), every operand pair, both paths; on `mma.sync` also with chunks of
  16 (one instruction each).
- **Part F, the existing facts on a subnormal operand.** One chunk as `subwin` builds it, repeated, at
  `K = 32, 64, 256`, both paths: the error against `relDot8`'s bound and against the new facts'.

**Result** (`out.txt`).

Part A: every device result equals the model (27 cases). Part A2: every accumulator alone passes
unchanged (`2^-149` included); a term `2^-23` of the largest is kept and `2^-24` is lost; 16 terms `1`
survive against `2^25` and not against `2^26`; the smallest products, `2^-18` (`float8e4nv`) against `1`
and `2^-32` (`float8e5`) alone, are kept.

Parts B and C (totals per operand pair and path; every row is in `out.txt`):

| Operands | Path | Elements | Outside the bound | Model = device | Largest ratio | Largest chunk ratio | relDot8: elements outside (largest ratio) |
|---|---|---|---|---|---|---|---|
| `float8e5 × float8e5` | `wgmma` | 7471104 | 0 | all | 0.4360 | 0.4360 | 36610 (1.7340) |
| `float8e5 × float8e5` | `mma.sync` | 451584 | 0 | — | 0.0002 | — | 0 |
| `float8e4nv × float8e4nv` | `wgmma` | 7471104 | 0 | all | 0.4678 | 0.4678 | 294912 (3.6506) |
| `float8e4nv × float8e4nv` | `mma.sync` | 451584 | 0 | — | 0.0002 | — | 0 |
| `float8e4nv × float8e5` | `wgmma` | 7471104 | 0 | all | 0.4678 | 0.4678 | 294912 (3.6506) |
| `float8e4nv × float8e5` | `mma.sync` | 451584 | 0 | — | 0.0002 | — | 0 |
| `float8e5 × float8e4nv` | `wgmma` | 7471104 | 0 | all | 0.4360 | 0.4360 | 37010 (1.7340) |
| `float8e5 × float8e4nv` | `mma.sync` | 451584 | 0 | — | 0.0002 | — | 0 |

The largest ratio on `wgmma` is reached by `subwin` at `K = 32` (one product with the smallest
subnormal operand sets the window, the other 31 products, just below `u`, are lost whole, and so is an
accumulator just below `u` of the products' sign), on `mma.sync` by `allones`. Every element outside
`relDot8`'s bound is of kind `subwin`, at `K = 32` or `64` (`out.txt`, "by kind of input").

Part D: 304 chains; no element outside the bound; every dot of every `wgmma` chain equal to the model
(`1.7·10^8` element results), and so is the whole chain run as one dot; the largest ratio is
`0.4591` per dot and `0.4514` for a whole chain (`subwin`); for softmax chains it stays below `0.13`
per dot and `0.012` for the whole chain at `T = 256`.

Part E: on both paths, with chunks of 32, every outcome is that of increasing `k` (168 of 168 with
8 chunks, 3000 of 3000 sampled triples with 32). With chunks of 16 on `mma.sync`, 155 to 158 of 168:
the two instructions of a chunk of 32 interleave their `k`.

Part F: on `wgmma`, the error reaches `3.65` times `relDot8 32`'s bound with a `float8e4nv` left
operand (both pairs; `1.79` times at `K = 64`) and `1.73` times with a `float8e5` one (`K = 32`); the
new facts' ratio stays below `0.47`. On `mma.sync` the errors are below `10^-4` of either bound.

**Conclusions.**
1. **The model is exact** on `wgmma`: every element of every kind, pair, shape and `K` (about
   `2·10^8` results with the chains) equals it bit for bit.
2. **The constants.** By the model, each term loses less than `u` and no more than itself, so the
   truncated sum is within `33u` of `s j + c_j`; the cut to 14 bits loses less than `2^-13` of the
   truncated sum, which is within `|s j + c_j| + 33u`. Since `2^e(x) ≤ L(x)` and
   `2^max(⌊log₂|s|⌋, −126) ≤ max(|s|, 2^-126)`, `u ≤ 2^-13·Mj + 2^-139` (the `2^-139` only when a
   subnormal accumulator sets `E`). So a chunk's error is below
   `33·(1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| + 34·2^-139`. The facts keep `2^-7` (a factor `1.94`
   over `33·(1 + 2^-13)·2^-13`, the margin `relDot8` kept over its per-chunk `33·2^-13`), `2^-12` (a
   factor 2) and `2^-126`. The measured chunk ratio reaches `0.468`: the bound is about twice the
   largest error, as the margins say. (Round 26, library issues L124 and L127: since every step is
   relative to `Mj ≥ |s j|`, a bound composed from these steps compounds by up to about
   `1 + 2^-7 + 2^-12` per chunk over a fed-back accumulator. That is the facts' form and margin, not
   the device: by the same model every term and the sum are truncated toward zero, so no partial
   value exceeds `|acc| + Σ_{i<j} Σ_chunk |a·b|` in magnitude, and a product loses at most
   `min(|p|, u)`, so the device's error grows at most linearly in the chunks. The loss facts
   `DotE4m3ChunkLoss`, … add the second bound `2·A_j + 2^-12·Mj + 2^-12·|s j + c_j| + 2^-126`,
   `A_j = Σ_chunk |a·b|`, of the same partial values:
   [evidence/num-dot-chunk-loss](../num-dot-chunk-loss/README.md).)
3. **The order** is increasing `k` on both paths, chunk by chunk (Part E), and the model, run in that
   order, equals the device (conclusion 1).
4. **`mma.sync`.** Each `float16` instruction keeps every term's bits down to at least `2^-23` of its
   largest term and rounds into `float32` (evidence/num-dot Part 1; Part A2), and accumulators pass
   unchanged; a chunk of two is within about `360·2^-23·Mj`, far below the facts (ratio `0.0002`).
   (Round 25, library issue L126: precisely, each `float16` instruction keeps 25 bits below its
   largest exponent field, a subnormal `float16` operand, such as a subnormal `float8e5` value, read
   as `2^-14`, and truncates its sum toward zero into `float32`; so "its largest term" is its largest
   lifted term, which `Mj` bounds, and the derivation stands. A subnormal `float8e5` operand with a
   large partner stays below `10^-4` of the facts' bound on `mma.sync`:
   [evidence/num-dot16-model](../num-dot16-model/README.md) Parts A and G2.)
5. **The facts hold** on every element tested, on both paths, for every pair and every `K` that
   compiles, the chains included.
6. **The existing facts `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3` do not hold** for a chunk
   whose largest exponent field comes from a product with a subnormal operand (Part F, and the
   `subwin` rows of Part C): their derivation (`33·2^-13` of the magnitudes' sum per chunk) assumed the
   window follows the largest term, and a subnormal operand's field reads as the smallest normal. They
   were withdrawn in round 24 (library issue L125). A bound on the whole dot is now the lemma
   `Dot8Chunk.DotF8Chunk.liftAcc` (`TVA.Lemmas.Dot8Chunk`), relative to `|acc| + Σ L(a_k)·L(b_k)`: a
   theorem of the chunk facts, so it holds wherever they do and needs no evidence of its own.
