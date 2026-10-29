# Reductions of broadcast products that become a `tl.dot`

Library evidence (not a kernel's), for library issue L26. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Since round 6** (library issue L36), the fact is stated in [SEMANTICS.md](../../docs/SEMANTICS.md) ("The combine pass's
rewrites"); TRANSCRIPTION.md is guidance.

**Question.** The transcription rule `[sum-bcast-mul]` named one spelling,
`tl.sum(x[:, :, None] * y[None, :, :], axis=1)`. Which spellings does Triton's combine pass rewrite
into a `tl.dot`, and does `tl.sum_bcast_mul` (which allows both the rewrite and the plain reduction)
cover every spelling of the computation?

**What the library reads.** `CombineBroadcastMulReducePattern`
(`primitives/triton-3.8.0-Combine.cpp:112-198`) matches the IR: a rank-3 `tt.reduce` on axis 1
whose combine region is one `arith.addf` of at most 32 bits, of an `arith.mulf` whose left operand
is a `tt.broadcast` of a `tt.expand_dims` on axis 2 and whose right operand is a `tt.broadcast`
of a `tt.expand_dims` on axis 0, broadcast to `M, N ≥ 16`. It rewrites it to `tt.dot` with IEEE
input precision, whatever `enable_fp_fusion` says.

**Method.** `r[i, j] = Σ_k x[i, k] y[k, j]`, `float32`, `M = N = 16`, `K = 2`, in eleven
spellings, each with `enable_fp_fusion` True and False. Reported: whether the TTIR (after the
combine pass) has a `tt.dot`, and how many of the 256 outputs no IEEE tree gives (for `K = 2`, any
value other than `round(round(x0 y0) + round(x1 y1))`). Also the first spelling on `float16` and
`bfloat16`, and with `M = 8` or `N = 8`.

**Result** (`out.txt`).

| Spelling | `tt.dot` | outputs no IEEE tree gives, fusion on / off |
|---|---|---|
| `tl.sum(x[:, :, None] * y[None, :, :], axis=1)` | yes | 49 / 49 |
| `tl.sum(tl.expand_dims(x, 2) * tl.expand_dims(y, 0), axis=1)` | yes | 49 / 49 |
| both factors through `tl.broadcast_to(…, (M, K, N))` | yes | 49 / 49 |
| one factor through `tl.broadcast_to` | yes | 49 / 49 |
| `tl.reduce(x[:, :, None] * y[None, :, :], 1, f)` with `f` returning `a + b` | yes | 49 / 49 |
| the method `(…).sum(axis=1)` | yes | 49 / 49 |
| `keep_dims=True` | yes | 49 / 49 |
| `dtype=tl.float32` on `float32` inputs (no conversion) | yes | 49 / 49 |
| the factors swapped, `y[None, :, :] * x[:, :, None]` | no | 56 / 0 |
| `tl.reshape(x, (M, K, 1)) * tl.reshape(y, (1, K, N))` | no | 56 / 0 |
| `tl.reduce` with `f` returning `a + b + 0.0` | no | 56 / 0 |

`float16` and `bfloat16` inputs with `M = N = 16`: the rewrite builds an ill-typed accumulator and
compilation fails on the host (a rejected launch, no outcome). With `M = 8` or `N = 8`: no rewrite.

**Conclusions.**
1. The pass rewrites every spelling whose IR is the pattern, not one spelling: `tl.expand_dims`,
   `tl.broadcast_to`, `tl.reduce` with one addition, the method form and `keep_dims` are rewritten,
   with fusion on or off, and give results no IEEE tree gives (49 of 256). So a rule that names one
   spelling lets a reviewer read the others by `[reduce]`/`[fusion]`, which is wrong.
2. Spellings the pass does not match (swapped factors, `tl.reshape`, a combine function with more
   than one operation) are plain reductions of products: IEEE with fusion off, fusable with it on.
3. `tl.sum_bcast_mul` allows both outcomes (the tree of fusable additions of the IEEE products, and
   the IEEE `tl.dot` when the additions are at most 32 bits and `M, N ≥ 16`). So the rule can be
   stated by the computation, every spelling of it is `tl.sum_bcast_mul`, and the reviewer need not
   know which spellings the pass matches.
