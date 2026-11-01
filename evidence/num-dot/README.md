# `tl.dot` into `float32` on sm_90

Library evidence (not a kernel's), for library issue L4. Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28. Script: [run.py](run.py); output:
[out.txt](out.txt). For library issue L66 (2026-10-28): [ieee_ratio.py](ieee_ratio.py), output
[ieee_ratio.out](ieee_ratio.out) ("The ratio of `DotF32Ieee`", below).

**Question.** How accurate is `tl.dot(a, b, acc)` into `float32` for `float16`, `bfloat16` and
fp8 operands (with Triton's default `max_num_imprecise_acc`: `0` for 16-bit operands, `2^30` for
fp8 on sm_90), and for `float32` operands with `input_precision="ieee"`? The library states
([rules/num.md](../../docs/rules/num.md), "`tl.dot`"), per output element, with
`mag = |acc| + Σ|a·b|` and room in `float32`:

| Fact | Operands | Bound on `|r − (acc + Σ a·b)|` |
|---|---|---|
| `DotF16`, `DotBF16` | `float16`, `bfloat16`; any `input_precision` | `(K+2)·2^-22·mag + (K+2)·2^-126` |
| `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3` | fp8; any `input_precision` | `((1+2^-7)^n − 1)·mag + n·(1+2^-7)^n·2^-126`, `n = ⌈K/32⌉` |
| `DotF32Ieee` | `float32`, `"ieee"` | `(K+1)·2^-23·mag + (K+1)·2^-149` |

The facts are for every `K`. What does the device do, and do the facts hold?

**Withdrawn (round 24, library issue L125).** The fp8 facts `DotE4m3`, `DotE5m2`, `DotE4m3E5m2` and
`DotE5m2E4m3` in the table above are false of the device, and the library no longer states them.
Part 2 below had no input where a product with a subnormal operand sets the alignment window (it
has the largest exponent field of its chunk although it is not the largest term). On such inputs
`wgmma` exceeds their bound, up to `3.65` times at `K = 32`: see
[evidence/num-dot-chunk](../num-dot-chunk/README.md), Part F. The fp8 facts are now the chunk facts
of that folder. This folder's fp8 rows stay as the record of what was measured; its Part 1 (the
13-bit window) still stands.

**What Triton runs.** With `M ≥ 64` and `num_warps ≥ 4`, `wgmma` (16 products per instruction
for 16-bit operands, 32 for fp8); otherwise `mma.sync`, with fp8 operands upcast to `float16`
first. `float32` with `"ieee"` uses no tensor cores: `fma.rn.f32` along `K`. The script reads
the path from each kernel's PTX.

**Method.**
- **Part 1, the accumulation window.** One output element, one chunk (`K = 16` or `32`), on both
  paths: a term `1` with `K − 1` terms `2^-t` (which `t` are kept fully?), `K` terms `1` against
  `acc = 2^s` (which `s`?), and `acc = 1 + 2^-23` with no products (is the accumulator itself
  cut?).
- **Part 2, the facts.** Every operand pair, seven block shapes (`16×16` with 1 warp to `128×128`
  with 8), `K` from 16 (fp8: 32; `float32`: 8) doubling up to 4096 (fp8: 8192; `float32`: 1024),
  or up to the largest that fits in shared memory (Triton rejects the rest on the host: "out of
  resources"; for blocks above `32×32` the script stops at `K = 2048`). Eight kinds of inputs,
  two draws each: normal values, values spread over many binades, a large accumulator with all
  bits set, same-sign products with all fraction bits set (the truncation's worst case), one
  large term among small ones, pairwise cancellation, tiny values (products below `float32`'s
  normal range), huge values (products near the format's limits). Every element with room is
  checked: finite, and `err + (K+1)·2^-52·mag ≤ bound` (the second term covers the `float64`
  reference, which is exact for each product).
- **Part 3.** For 16-bit and fp8 operands: every `input_precision` against the default, bit for
  bit.
- **Part 4.** For `float32` with `"ieee"`: each sampled element against the exact sequential
  `fma` over `k = 0, …, K−1` from `acc` (computed with fractions), bit for bit.

**Result** (`out.txt`).

Part 1:

| Operands | Path | Small terms kept fully | Terms against `acc = 2^s` kept | `acc = 1 + 2^-23` |
|---|---|---|---|---|
| `float16`, `bfloat16` | both | `t ≤ 23` | `s ≤ 25` | kept |
| `float8e4nv` | `mma.sync` | `t ≤ 9` (the smallest e4m3 value) | `s ≤ 25` | kept |
| `float8e5` | `mma.sync` | `t ≤ 16` (the smallest e5m2 value) | `s ≤ 25` | kept |
| `float8e4nv` | `wgmma` | `t ≤ 9` (the smallest e4m3 value) | `s ≤ 13` | cut to `1` |
| `float8e5` | `wgmma` | `t ≤ 13` | `s ≤ 13` | cut to `1` |

Part 2 (rows per path; every row is in `out.txt`):

| Operands (fact) | Path | `K` tested | Outside the bound | Largest error / bound |
|---|---|---|---|---|
| `float16` (`DotF16`) | `mma.sync` | 16 – 2048 | 0 | 0.124 |
| `float16` (`DotF16`) | `wgmma` | 16 – 512 | 0 | 0.123 |
| `bfloat16` (`DotBF16`) | `mma.sync` | 16 – 2048 | 0 | 0.124 |
| `bfloat16` (`DotBF16`) | `wgmma` | 16 – 512 | 0 | 0.122 |
| `float8e4nv` (`DotE4m3`) | `mma.sync` | 32 – 8192 | 0 | 0.0001 |
| `float8e4nv` (`DotE4m3`) | `wgmma` | 32 – 1024 | 0 | 0.388 (`K = 256`) |
| `float8e5` (`DotE5m2`) | `mma.sync` / `wgmma` | 32 – 8192 / 32 – 1024 | 0 | 0.0001 / 0.336 |
| `float8e4nv × float8e5`, `float8e5 × float8e4nv` | `mma.sync` / `wgmma` | 32 – 8192 / 32 – 1024 | 0 | 0.0001 / 0.361 |
| `float32`, `"ieee"` (`DotF32Ieee`) | `fma` | 8 – 1024 | 0 | 0.499 (`allones`, below) |

In `out.txt`, Part 2's last column, "err/u (worst input kind)", gives the largest error relative to
`|acc| + Σ|a·b|` (in units of `2^-24`) and the input kind with that largest *relative error*, which
need not be the kind with the largest ratio. For `float32` it names `tiny`, whose relative error is
large (products below the subnormal range are lost) while its ratio is `0`: those errors are far
below the absolute term `(K+1)·2^-149`.

Part 3: every `input_precision` gives the same bits as the default for 16-bit and fp8 operands.
Part 4: every compared element (576) equals the sequential `fma`.

**Conclusions.**
1. Every fact holds on every element tested, on both paths, at every `K` that compiles.
2. **16-bit operands.** Each instruction keeps every term down to at least `2^-23` of the
   largest (truncating below) and keeps the accumulator; a chunk of `j` nonzero terms is within
   `j·2^-23` times their magnitudes' sum. Summed over `⌈K/16⌉` chunks this is about
   `(K + ⌈K/16⌉)·2^-23·mag`, below `(K+2)·2^-22·mag` for every `K`; the measurements reach `0.12`
   of the bound, nearly constant in `K`, so the fact extends past the largest `K` tested.
3. **fp8 on `wgmma`** (Triton's default `max_num_imprecise_acc = 2^30` on sm_90) keeps only 13
   bits below the largest term, the accumulator included. A chunk of 32 products and the
   accumulator is within `33·2^-13 < 2^-7` times its magnitudes' sum, and the error compounds over
   the `⌈K/32⌉` chunks: `relDot8`. The relative error reaches `2.5%` at `K = 256` (`0.39` of the
   bound) and the ratio falls for larger `K`, as the compounded bound grows faster than the
   error. On `mma.sync` the fp8 dot is a `float16` dot and far inside the bound.
4. **`float32` with `"ieee"`** is exactly the sequential `fma` (Part 4): `K` roundings, each
   within half an ulp of its partial sum. The classical bound for them is
   `γ_K = K·2^-24/(1 − K·2^-24) ≈ K·2^-24` times `|acc| + Σ|a·b|`; the fact's `(K+1)·2^-23` is
   about twice it at every `K` that compiles. Its largest ratio, `0.499` at `K = 1024`, is the
   relative term on the `allones` input, not the absolute term (below).
5. `input_precision` does not change 16-bit or fp8 dots, so their facts hold for every value.

**The ratio of `DotF32Ieee`** ([ieee_ratio.py](ieee_ratio.py), [ieee_ratio.out](ieee_ratio.out);
library issue L66). The same kernel and input kinds, on `16 × 16` with one warp, `K = 8 … 1024`,
with the largest ratio per kind and the worst element recomputed exactly with fractions. The
largest ratio is always on `allones`, and lies below `(K−1)/(2(K+1))` and within `3.1·10^-5` of
it: the exact table's `error / bound` is `0.388889` against `0.388889` at `K = 8`, `0.498035`
against `0.498051` at `K = 512` (so the four-digit table prints `0.4980` against `0.4981`), and
`0.498994` against `0.499024` at `K = 1024`, the gap growing with `K`; every other kind stays
below `0.30`, and `tiny` is at `0.0000`.
The worst element has an accumulator with all significand bits set (`acc = 2^25 − 2` at `K = 8`)
and `K` equal products just below one ulp of it (`≈ 2`): the first `fma` rounds up to the next
power of two (`2^25`), where the ulp doubles, and each of the `K − 1` later products, now just
below half an ulp, is lost. The error is just below `(K−1)·2^-24·mag` (exactly `6.999996`, …,
`1022.94` in units of `2^-24·mag`), against the bound's relative term
`(K+1)·2^-23·mag`, so the ratio `(K−1)/(2(K+1))` approaches `1/2` from below and never reaches
it. It is the factor `2` the fact keeps over half an ulp per rounding; the absolute term plays no
part. (The account in the earlier version of this README, "the absolute term by construction",
was wrong.)
