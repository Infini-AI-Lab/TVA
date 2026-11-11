# A 16-bit `tl.dot` on sm_90, chunk by chunk

Library evidence (not a kernel's), for library issue L150 (round 32: sharper per-chunk facts for
`float16` and `bfloat16` `tl.dot`). Triton 3.8.0 (TVA's `.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL
(sm_90), GPU 2, 2026-11-08. Script: [run.py](run.py); output: [out.txt](out.txt) (run time about 16 minutes). It builds on
[evidence/num-dot16-model](../num-dot16-model/README.md), whose bit-level model of one
instruction and whose input kinds it copies.

**Question.** Do the facts `DotF16Chunk` and `DotBF16Chunk` (`lean/TVA/Rules/Num/Dot16Chunk.lean`)
hold? For a `float16` × `float16` (`bfloat16` × `bfloat16`) `tl.dot` into `float32` with
`max_num_imprecise_acc = 0` (the default for these operands) and any `input_precision`, they state,
per output element whose row of `a`, column of `b` and accumulator have values `a_k`, `b_k`, `acc`, with
room (`mag + ((1 + 2^-20)^n − 1)·mag + n·(1 + 2^-20)^n·2^-145` at most the `float32` maximum,
`mag = |acc| + Σ|a_k·b_k|`, `n = ⌈K/16⌉`), that there are partial values `s 0 = acc, …, s n = result`,
one step per chunk of 16 consecutive `k` in increasing order, with

`|s (j+1) − (s j + c_j)| ≤ min(2^-20·Mj, 2·A_j + 2^-24·Mj) + 2^-22·|s j + c_j| + 2^-145`

for every `Mj` at least `|s j|` and every lifted product `L(a_k)·L(b_k)` of the chunk. `c_j` is the
chunk's exact sum, `A_j = Σ_chunk |a_k·b_k|` its exact magnitude (not lifted), and `L(x)` is `0` for
`x = 0` and `max(|x|, 2^emin)` otherwise (`2^-14` for `float16`, `2^-126` for `bfloat16`). They stand
beside `DotF16` and `DotBF16` (a whole dot within `(K+2)·2^-22` of `|acc| + Σ L(a)·L(b)`), which stay.

**What Triton runs** (evidence/num-dot16-model; every configuration here names its path from
the PTX). With `M ≥ 64` and at least 4 warps, one `wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16`
(`.bf16.bf16`) per chunk of 16 along `K`, each from the accumulator the previous one left; otherwise
one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (`.bf16.bf16`) per chunk.

**The model** (evidence/num-dot16-model, equal to the device bit for bit on about `3.4·10^8`
element results there, and on every element here). One instruction, per output element: the terms
are the accumulator `s` and the 16 exact products. Each nonzero term has an exponent field:
`e(a) + e(b)` for a product, with `e(x) = max(⌊log₂|x|⌋, emin)` (a subnormal operand reads as the
smallest normal), and `max(⌊log₂|s|⌋, −126)` for the accumulator. With `E` the largest field and
`u = 2^(E−25)`, every term is truncated toward zero to a multiple of `u`, the truncated terms are
summed exactly (`S'`), and `S'` is truncated toward zero to `float32` (24 significant bits; the grid
`2^-149` below `2^-126`). The chunks are 16 consecutive `k`, in increasing order, on both paths.

**Derivation** (the constants from the model).
- **The unit against `Mj`.** `2^e(x) ≤ L(x)` for `x ≠ 0`, so a product's `2^(e(a)+e(b))` is at most
  its lifted product, hence at most `Mj`; the accumulator's `2^max(⌊log₂|s|⌋, −126)` is at most
  `max(|s|, 2^-126) ≤ Mj + 2^-126`. So `u ≤ 2^-25·Mj + 2^-151`.
- **The cut to `float32`** loses less than `2^(⌊log₂|S'|⌋−23) ≤ 2^-23·|S'|` for a normal `S'`, and less
  than `2^-149` below `2^-126`: less than `2^-23·|S'| + 2^-149`.
- **First arm.** Each of the at most 17 terms loses less than `u`, so `|S' − (s + c)| < 17u` and
  `|S'| ≤ |s + c| + 17u`. A chunk's error is below
  `17·(1 + 2^-23)·u + 2^-23·|s + c| + 2^-149 ≤ 17·(1 + 2^-23)·2^-25·Mj + 2^-23·|s + c| + 5.25·2^-149`.
  The fact keeps `2^-20 = 32·2^-25` (a factor `1.88`), `2^-22` (`2`) and `2^-145` (`3.05`).
- **Second arm.** A product truncated toward zero to a multiple of `u` loses at most `min(|p|, u) ≤ |p|`,
  and the accumulator less than `u`, so `|S' − (s + c)| ≤ A_j + u` and `|S'| ≤ |s + c| + A_j + u`. A
  chunk's error is below `(1 + 2^-23)·A_j + (1 + 2^-23)·2^-25·Mj + 2^-23·|s + c| + 5.25·2^-149`. The fact
  keeps `2·A_j` (`2/(1 + 2^-23)`), `2^-24·Mj` (`2/(1 + 2^-23)`), `2^-22` and `2^-145`.
- **What the second arm buys.** An accumulator that dominates its chunk (its field sets the window, the
  products far below it) is charged `2·A_j + (2^-24 + 2^-22)·|s|` per chunk, not `relDot16 K·|s|` per
  `tl.dot`: with `K = 64`, `66·2^-22·|s|` under `DotF16` against `4·1.25·2^-22·|s|` plus the products'
  own magnitudes.
- **The worst cases.** For the first arm, a product that sets the window at `Mj = 2^E` and 15 others
  just below `u` (lost whole, kind `winworst`, or `subwin` with a subnormal operand setting the window):
  an error of about `15u` plus the cut, against `32u + 2^-22·|s + c|`, a ratio of about `0.5`. For the
  second, an accumulator `2^E` (times `1`, `1 + 2^-23` or `2 − 2^-23`) with 16 products just below `u`
  (lost whole, kind `accdom`): an error of about `16u` against `34u + 8u`, a ratio of about `0.4`; with
  products just below `2u` (each truncated to `u`, kind `accdom2`) the second arm grows with `A_j`.
- **Room.** No partial value exceeds `mag` in magnitude (every term and the sum are truncated toward
  zero), so with room nothing overflows and the result is finite.

**Method.**
- **Part S, single chunks on the device** (`K = 16`). With one chunk the partial values are the
  accumulator and the device's result, so each element is checked against the step bound with no
  model, at the smallest `Mj` (the larger of `|acc|` and the chunk's largest lifted product), where the
  bound is tightest. Blocks `16 × 16` (1 warp), `32 × 32` (2), `64 × 64` (2) on `mma.sync`, and
  `64 × 64` (4), `128 × 64` (4), `128 × 128` (8) on `wgmma`; with an accumulator, without one, and with
  `a` converted in registers and `b` transposed (tutorial 06's `p.to(…)` and `k.T`). Kinds: the 15 of
  evidence/num-dot16-model (`normal`, `softmax`, `spread`, `accbig`, `allones`, `onebig`,
  `cancel`, `tiny`, `huge`, `sub`, `subwin`, `winworst`, `subrand`, `trunc24`, `bits`) and three new
  ones: `accdom` and `accdom2` (above) and `subacc` (`bfloat16`: a subnormal `float32` accumulator sets
  the window, field `−126`, with products just below `u = 2^-151`; for `float16` it draws `tiny`). Each
  kind 12 times per configuration where it targets the window or subnormals, 6 times otherwise. The
  ratio is computed in `float64` with `2^-48` of the magnitudes added to the error (an upper bound);
  every element above `0.95` and each draw's largest are recomputed in exact rationals. The model is
  also compared with the device.
- **Part C, chains at every `K` that compiles** (`K = 32` to 2048 on blocks up to `32 × 32`, to 1024
  otherwise; a launch Triton rejects on the host is recorded). The device result against the model
  (bit for bit as values); each step of the model's chain (whose last value is the device's result)
  against the step bound; and the device result against `errM`, the bound
  `Dot16Chunk.DotChunk16Loss.err` composes from the exact inputs (`errM (j+1) = (1 + 2^-22)·errM j +
  min(2^-20·(X j + errM j), 2·A_j + 2^-24·(X j + errM j)) + 2^-22·|S (j+1)| + 2^-145`, `S j` the exact
  partial sums, `X j = max(|S j|, m j)`, `m j` the chunk's largest lifted product), with the rounding of
  the `float64` partial sums added; next to the ratio to `DotF16`'s whole-dot bound. Each kind 6 or 3
  times per configuration.
- **Part L, fed-back accumulators.** `for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t, v_t,
  acc)`, `K = 64` per dot, every `acc_t` stored: tutorial 03's loop (`p_t` loaded in the format, `normal`
  blocks, and `accdom` blocks: an accumulator that dominates every chunk) and tutorial 06's (`p_t`
  converted in registers from a softmax, with and without a rescale `alpha_t ∈ [0.25, 1)`), `T` up to
  256 (`K = 16384`), on `wgmma` (`128 × 64`, 4 warps; `128 × 128`, 8; `64 × 64`, 4) and `mma.sync`
  (`32 × 64`, 2; `16 × 64`, 1). Each dot from the stored accumulator before it: the model, each step,
  and `errM`; the whole chain (no rescale) against `errM` over `4T` chunks from the exact inputs, and
  that bound divided by the bound a kernel gets by composing `DotF16` step by step
  (`e (t+1) = e t + relDot16 64·(|S t| + e t + L t) + absDot16 64`).

**Result** (`out.txt`).

- **Part S** (single chunks, the device's own partial values; totals):

  | Operands | Path | Elements in range | Outside | Model = device | Largest ratio | First arm alone | Second arm alone |
  |---|---|---|---|---|---|---|---|
  | `float16` | `mma.sync` | 2709504 | 0 | all | 0.4996 | 0.4996 | 0.3809 |
  | `float16` | `wgmma` | 14450688 | 0 | all | 0.4996 | 0.4996 | 0.3809 |
  | `bfloat16` | `mma.sync` | 2709504 | 0 | all | 0.4972 | 0.4972 | 0.3806 |
  | `bfloat16` | `wgmma` | 14450688 | 0 | all | 0.4972 | 0.4972 | 0.3806 |

  The largest ratio is reached by `subwin` (a subnormal operand setting the window, 15 products lost),
  as the derivation's `17·2^-25/2^-20 = 0.53` allows; `accdom` reaches `0.400` (`float16`) and `0.398`
  (`bfloat16`) of the bound, on its second arm. 6048 elements were recomputed in exact rationals.
- **Part C** (chains, `K = 32` to the largest that compiles; the next is rejected on the host, out of
  resources, recorded): 26087424 elements in range, none outside; every device result equals the model;
  the largest step ratio is `0.4996` (`float16`) and `0.4972` (`bfloat16`) and the largest ratio to the
  composed `errM` `0.4933` (`float16`, `mma.sync`), `0.4891` (`float16`, `wgmma`), `0.4972`
  (`bfloat16`). The same errors are at most `0.125` of `DotF16`'s whole-dot bound.
- **Part L** (fed-back accumulators, `T` up to 256): 57647104 element results (every dot of every
  chain), none outside; every dot equals the model from the stored accumulator; the largest step
  ratio and per-dot `errM` ratio are `0.3998` (`accdom`), and the largest whole-chain `errM` ratio is
  `0.369`. Composed over the whole chain, `errM` is at most `0.30` of what composing `DotF16` step by
  step gives (`0.21` to `0.27` for tutorial 03's loop and tutorial 06's softmax chains at `T = 32` to
  `256`, `0.11` for `bfloat16` `accdom`). In `K loop accdom` only the first block is drawn against the
  initial accumulator, so after it the accumulator no longer sets every window, and the whole-chain
  ratio falls (`0.02` to `0.06` at `T = 256`).
- Every element on every path came out within the facts' bound; the largest ratio over all parts is
  `0.4996`, below the derivation's `0.53` (the margin of `1.88` in `2^-20`).

**Conclusions.**
1. **The facts hold** on both paths, in both formats, on about `1.2·10^8` element results: single
   chunks against the device's own partial values, chains at every `K` that compiles, and fed-back
   accumulators up to `K = 16384`. The derivation from the per-instruction model extends them to every
   `K` and every input with room; the largest measured ratio, `0.4996`, is within the derivation's
   worst case (`0.53` of the first arm).
2. **The accumulator's share** is bounded per chunk: an accumulator that dominates its chunk is charged
   `2·A_j + (2^-24 + 2^-22)·|s|` per chunk by the second arm, where `DotF16` charges `relDot16 K·|acc|`
   per `tl.dot`; composed over the tutorials' loops the new bound is `0.2` to `0.3` of the old one.
3. **`input_precision`** changes nothing for these operands (evidence/num-dot16-model Part G1),
   so the facts hold for every value.
