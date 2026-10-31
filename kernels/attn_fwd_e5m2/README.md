# attn_fwd_e5m2

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

**Read this first.** The kernel as given reads `v` for the first `⌈Z·H/HEAD_DIM⌉` heads only. For
every other head (`bh·HEAD_DIM ≥ Z·H`), line 98 reads the zero padding of `v`'s descriptor, and the
head's output is `±0`. On the host's shapes that is every head of a launch but 1 to 3. `Post`
compares `o` with the attention of the values the kernel uses (`vEff`, Specification), so for those
heads it certifies `±0`. For the other heads its accuracy part is loose. On the test's and the
benchmark's data it does not tell `o` from `0` (Specification, "Where `Post` does not tell `o` from
`0`"). Part of that is the fp8 rounding itself; the rest comes from the library's `Num.DotE5m2`
(library issue [L124](../../docs/library-review/ISSUES.md#l124)).

Source: [`source/06-fused-attention.py`](source/06-fused-attention.py) (tutorial 06 as set up in
[SETUP.md](SETUP.md): the `@triton.autotune` decorator removed, the config `BLOCK_M=128,
BLOCK_N=64, num_stages=2, num_warps=4` passed by the launch, the config's `pre_hook` called by the
host; the same file as `attn_fwd_f16`'s). Lean: [`lean/`](lean/), the module `AttnFwdE5m2`
(namespace `Kernels.AttnFwdE5m2`) and its parts:

| Module | What it holds |
|---|---|
| `AttnFwdE5m2.Model` | the model (`attn_fwd_inner`, `maybe_make_tensor_desc`, `attn_fwd`, `model`, `launch`), the conditions (`Inst`, `Inst.ok`, `Pre`), the specification (`Post`, `Spec`, the references and tolerances), the assumptions |
| `AttnFwdE5m2.Real`, `.Lane`, `.RowMax`, `.RowAcc`, `.Step` | one query row through the loop, in real numbers: the running maximum (`MInv`), `l_i` (`LInv`) and `acc` (`AInv`) |
| `AttnFwdE5m2.Final` | the end of a row (lines 241–245: `final_o`, `final_M`), the `Model` constants as real numbers, their numeric facts (`model_factors`) |
| `AttnFwdE5m2.Mag` | the second bound on the output, over the computed weights: `l_i ≥ lminB t` (`lmin_real`), one block of `\|acc − r·l_i\| ≤ magA·l_i + magD` (`estep_real`), the end of a row (`final_v`) |
| `AttnFwdE5m2.Facts` | the grid, `int32` arithmetic, the descriptors (`descOf`, the transposed `descV`), the constants, `qk_scale` (`qk_scale_facts`, `qk_scale_w`) |
| `AttnFwdE5m2.Row`, `.Tile`, `.Inner` | a row's state (`RowSt`, `mag_tail`, `row_step_plain`, `row_step_masked`), the tile loads and stores (`safe_load_v` for `v`), the loop of `_attn_fwd_inner` and its three calls |
| `AttnFwdE5m2.Program` | the row's sums in the specification's terms (`esR_eq`, `ebR_eq`), one row's end (`row_out`), one program (`safe_program`) |
| `AttnFwdE5m2` | `main`, `consistent`, `satisfiable` |

**The kernel** is `_attn_fwd` (source lines 176–245) with its two `@triton.jit` helpers
`_attn_fwd_inner` (lines 47–110) and `_maybe_make_tensor_desc` (lines 168–173), in the `float8e5`
run (`FP8_OUTPUT = True`). It reads the module-level names `triton` (the decorators) and `tl`,
bound by the imports of lines 20–21, which are kernel lines and are mapped (dropped: they add no
step), and the two helpers, bound by their `def`s. As given, it gets `v` only for the first
`⌈Z·H/HEAD_DIM⌉` heads and writes `±0` for every other head (`vEff`, Specification).

**The host.** `_attention.forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True)` (lines
505–580) asserts one `HEAD_DIM` in `{16, 32, 64, 128, 256}` (lines 508–512; `v.shape[-1]` is
`HEAD_DIM` for the transposed `v`), allocates `o = torch.empty_like(q)` (`float8e5`) and the
`float32` `M = torch.empty((Z, H, N_CTX))`, and passes `STAGE = 3` when `causal` and `1` otherwise.
On the H200, when `warp_specialize` is false it builds host `TensorDescriptor`s: over `q`, `k`, `o`
with `shape = [Z·H·N_CTX, HEAD_DIM]`, `strides = [HEAD_DIM, 1]` (lines 528, 535, 536), and, since
`q.dtype == torch.float8_e5m2`, over `v` with `shape = [HEAD_DIM, Z·H·N_CTX]`,
`strides = [N_CTX, 1]` (lines 529–531); `_host_descriptor_pre_hook` (line 557) sets their block
shapes to `[128, HEAD_DIM]` (`q`, `o`), `[64, HEAD_DIM]` (`k`) and `[HEAD_DIM, 64]` (`v`, line 121
with `FP8_OUTPUT`). When `warp_specialize` is true it passes the tensors (lines 538–541) and the
kernel builds the same descriptors (lines 195–206, 173), with the allocator of line 546. It launches
the grid `(cdiv(N_CTX, 128), Z·H, 1)` (lines 548–549) with `BLOCK_M=128`, `BLOCK_N=64`,
`FP8_OUTPUT=True` (line 569), `STAGE`, `warp_specialize`, `IS_HOPPER=is_hopper()` (true) and
`num_stages=2, num_warps=4` (`extra_kern_args` is empty on the H200).

The script calls it on `float8e5` inputs in the test (`test_op`, lines 631–695, the `fwd` mode of
the `triton-fp8` provider: `Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`,
`HEAD_DIM ∈ {64, 128}`, both `causal`, `warp_specialize = False` off Blackwell, `normal(0, 0.5)`
inputs cast to `float8_e5m2`, `v` made the `(Z, H, N_CTX, HEAD_DIM)` view of a contiguous
`(Z, H, HEAD_DIM, N_CTX)` tensor (lines 672–676: `.to` keeps the dense strides), `sm_scale = 0.5`,
checked with `atol = 3`) and in the benchmark (lines 698–780, the `triton-fp8` line of the `fwd`
mode: `Z = 4`, `H = 32`, `N_CTX = 2^10 … 2^14`, `HEAD_DIM ∈ {64, 128}`, both `causal`,
`warp_specialize ∈ {False, True}` for the non-causal forward on Hopper, `torch.randn` inputs cast
the same way, `sm_scale = 1.3`). Its `float16` run is `attn_fwd_f16` (Narrowings).

**Host lines and unused functions** are those of `attn_fwd_f16` (the same file): every
module-level line outside the kernel lines is a host line (`source.host`: the imports, `DEVICE`,
the device predicates, `_host_descriptor_pre_hook`, the config list, the class `_attention`,
`attention`, the test, the benchmark, the `__main__` block), and `keep`, `prune_invalid_configs`
(unused once the decorator is removed, SETUP.md) and the backward kernels `_attn_bwd_preprocess`,
`_attn_bwd_dkdv`, `_attn_bwd_dq`, `_attn_bwd` (other folders' kernels; `_attn_fwd` does not call
them) are `source.unused`, their headers host lines.

**Taken from `attn_fwd_f16`, and what changed.** The model is `attn_fwd_f16`'s with the
`FP8_OUTPUT = True` branches: `float8e5` tiles, `cfg8` for both `tl.dot`s (fp8 operands, the sm_90
default `max_num_imprecise_acc = 2^30`), line 65's `offsetv_y = offset_y·HEAD_DIM + lo`, line 98's
transposed load of `v` through the `[HEAD_DIM, y_dim]` descriptor of lines 198–199, and the fp8
conversions of lines 101 and 245 (`tl.cvt8`, the platform's conversion). The model map is
`attn_fwd_f16`'s with these lines' entries changed (48–53, 64–67, 72–73, 97–103, 169, 177–187,
197–204, 245). From
`attn_fwd_f16` the proof keeps its plan and most of its program-level files (`Facts`, `Row`, `Tile`,
`Inner`, `Program` and the root are adapted copies: the grid, the `int32` facts, the descriptor
lemmas, the loop invariant `StInv`, `safe_inner1/2/3`, `safe_epi`, `safe_program`, `main`), the
running-maximum analysis (`RowMax`) and the shape of the `l_i` analysis (`lstep`). What changed:

- the accuracy facts are `Num.DotE5m2` (`relDot8 K = (1 + 2^-7)^⌈K/32⌉ − 1`, about `1.57%` at
  `K = 64`, against `attn_fwd_f16`'s `relDot16 K = (K + 2)·2^-22`) and `Num.CvtSatF32E5m2` (one
  saturating rounding to `float8e5`, `2^-3` relative, `2^-17` absolute);
- with these constants the accumulator's error compounds by `1 + relDot8 64` per block, so `acc` is
  bounded block by block in the order the loop runs (`errS`, `errB`, below) instead of by a closed
  factor, and `l_i` from both sides (`Fhi`, `Flo`);
- each row's exact exponents are the base-2 `w·s_j` with `w = sm_scale/ln 2` (in `RowCtx`, with
  `|qk_scale − w| ≤ 2^-22·w`), so the weights are `exp(sm_scale·s_j)` directly;
- the computed exponents are kept at most `2^-10` (so `p ≤ 1 + 2^-9`, far inside `float8e5`), and
  the coverage cap on the scaled scores is `Xh D = 2^10/⌈D/32⌉`, which keeps each exponent's error
  `expErr ≤ 13`;
- `v` is read through the transposed descriptor (`descV`, `safe_load_v`), which reads zeros for most
  heads (`vEff`, Specification);
- `Inst.ok.elems_lt` (`Z·H·N_CTX·HEAD_DIM < 2^31`, for `offsetv_y`) replaces `rows_lt`.

## Specification

Write `D = HEAD_DIM`, `N = N_CTX`, `σ = sm_scale` (the double, as a rational: `sigma`). Element
`(bh, n, e)` of `q`, `k`, `o` (for the head `bh = z·H + h`) is element `idx d bh n e =
(bh·N + n)·D + e` of its buffer (contiguous); element `(bh, n, e)` of the transposed `v` is element
`vIdx d bh n e = (bh·D + e)·N + n`; element `(bh, n)` of `M` is element `bh·N + n`. For query `n`,
its keys are `keys d n`: every `j < N`, and only `j ≤ n` when `causal`. With the input values in
`init` (`val8`: a `float8e5` element's value), the exact score is `score = Σ_e q[bh, n, e]·k[bh, j, e]`,
its magnitude `scoreAbs = Σ_e |q·k|`, and the exact weight `weight = exp(σ·score)` (real `exp`).

**The values of `v` the kernel uses (`vIn`, `vEff`).** Line 98 loads `v` through the descriptor of
shape `[HEAD_DIM, y_dim]`, `y_dim = Z·H·N` (lines 198–199 and the host's 530), at column
`offsetv_y = bh·N·D + 64t` (line 65 with `offset_y = bh·N`). The addresses are those of
`v[bh, j, e]`, but the columns of head `bh`'s keys, `bh·N·D + j`, are inside the descriptor's
`y_dim` exactly when `bh·D < Z·H` (`vIn`). For every other head every lane of every `v` block is
outside, and the descriptor's padding `"zero"` gives `+0` (SEMANTICS.md, tensor descriptors). So
`vEff d init bh j e` is `v[bh, j, e]` when `bh·D < Z·H` and `0` otherwise, and the kernel computes
the attention of `q`, `k`, `vEff`. On the host's shapes only the first heads get their `v`
(`bh < Z·H/D`: heads `0` to `2` of `192` at `D = 64` in the test, only head `0` of `128` at
`D = 128` in the benchmark); every other head's output is `0`. The test does not see this: a
head's reference is a weighted mean of `normal(0, 0.5)` values, far below the test's `atol = 3`.
The claim states it.

- `attnRef d init bh n e = Σ_j weight_j·vEff[bh, j, e] / Σ_j weight_j` (over `keys d n`): the
  exact attention of the values the kernel uses (for a head with `vIn`, the attention of `q`, `k`,
  `v`; for the others `0`).
- `lseRef d init bh n = log₂ Σ_j weight_j`, the exact log-sum-exp in base 2, which `M` holds.

A head is covered (`HeadOk d init bh`) when every element of `q` and `k` in it is a finite
`float8e5`, so is every element of `v` in it when `vIn`, and every scaled magnitude
`σ·scoreAbs ≤ Xh D = 2^10/⌈D/32⌉` (`512`, `256`, `128` at `D = 64, 128, 256`).

`Post d init fin` says three things.

- **The output `o`.** Every element `(bh, n, e)` of `o` (`bh < Z·H`, `n < N`, `e < D`) holds some
  `float8e5` value `c` in `fin`. When head `bh` is covered, `c` is finite, and two bounds hold.
  First, for every `X ≤ Xh D` that bounds the row's scaled magnitudes (`σ·scoreAbs(n, j) ≤ X` for
  every key `j` of `n`), `|c − attnRef| ≤ tolO d init bh n e X`. Second, for every `K` that bounds
  `|vEff[bh, j, e] − attnRef|` over the keys `j` of `n`, `|c − attnRef| ≤ tolV d init bh n e K`.
- **The log-sum-exp `M`.** Every element `(bh, n)` of `M` holds some `float32` value `c`. When head
  `bh` is covered, `c` is finite and, for every such `X`, `|c − lseRef| ≤ tolM D N X + 2^-24·|lseRef|`.
- **Nothing else changes.** Every byte outside the elements of `o` and `M` (`outBytes`: the first
  `Z·H·N·D` elements of `o`, the first `Z·H·N` of `M`) is the same in `fin` as in `init`.

**The tolerances** (`AttnFwdE5m2.Model`, "The constants") are explicit expressions of `D`, `N`, `X`
and of the row's exact data, built only from the library's constants: `relDot8 D` (the score's
`tl.dot`, `Num.DotE5m2`), `relDot8 64 = 257/2^14` (the `p·v` `tl.dot`), `absDot8`, `2^-22`
(`Exp2F32`, `DivFull`, `Log2F32`), the `float32` roundings (`2^-24`, `2^-150`) and the `float8e5`
rounding (`2^-3`, `2^-17`):

- each computed exponent is within `expErr D X = (relDot8 D + 2^-19)·(3X/2) + 2^-50` of the exact
  `w·s_j − m_i` (the score's `tl.dot` error relative to `Σ|q·k|`, times `qk_scale ≤ 3σ/2`), so
  each `p` is within a factor `pE = 2^expErr` (and `exp2`'s `2^-22`) of its exact weight;
- the accumulator's error is the recursion the library's facts give block by block, over the
  row's blocks `t = 0 … nblk − 1` in the loop's order: `errS (t+1) = (1 + rhoA)·errS t +
  rhoA·|partS t| + psiA·blkMag t` against the exact partial sums `partS t = Σ_{j < 64t} w_j·vEff_j`
  (`blkMag t = Σ_{j in block t} w_j·|vEff_j|`), and the absolute `errB (t+1) = (1 + rhoA)·errB t +
  (1 + relDot8 64)·omega8·Σ_{block t}|vEff_j| + 2^-64` in the running maximum's units; `rhoA`
  (about `1.57%`) is `relDot8 64` with `alpha`'s and `acc·alpha`'s roundings, `psiA` (about `14%`
  times `pE`) the error of a block's `Σ p·v` (the fp8 rounding of `p` and the `tl.dot`), `omega8` the
  fp8 rounding's absolute `2^-17`;
- `l_i` is between `Flo·Σ − aL` and `Fhi·Σ + aL` (`Σ` the weights in the running maximum's units),
  `Fhi`, `Flo` products of per-block factors `stepL`, `qL` over `N/64` blocks;
- `tolO = 9/8·(1 + 2^-22)·(errS/Σw/Fp + pE·errB/Fp + betaL/Fp·|attnRef|) + (9/8·2^-22 + 1/8)·|attnRef|
  + 9/8·2^-148 + 2^-17`: the accumulator over `l_i` (`Fp = Flo − aL·pE`), `l_i`'s own error `betaL`,
  then the division and the rounding of the result to `float8e5`;
- `tolM` is `log₂`'s error of `l_i` against `Σ` (`lgB`), `Log2F32`'s, and the final rounding;
- `tolV d init bh n e K` (`Model.magA`, `magD`, `tolV`) compares `acc` with `r·l_i`, `r = attnRef`,
  over the computed `p`: lines 95 and 106 scale both by the same `alpha`, so no comparison of `p`
  with the exact weights is needed, and `pE` does not enter. After `t` blocks
  `|acc − r·l_i| ≤ magA t·l_i + magD t`, with `magA (t+1)` the larger of
  `((1 + ρ)(1 + 2^-24)·magA t + |r|·(4·2^-24 + ρ(1 + 2^-24)))/(1 − 3·2^-24)` (the old `acc`, and the
  dot's error `ρ = relDot8 64` on it) and `magCp/(1 − 2^-17 − 3(1 + 2^-17)·2^-24)`, where
  `magCp = 9/8·K(1 + ρ) + |r|·(9/8·ρ + 1/8 + 2^-17 + 3(1 + 2^-17)·2^-24)` (the block's `Σ q·v − r·Σ p`
  with `p`'s fp8 rounding `q`, the dot's error on it, `tl.sum(p)`'s and line 106's roundings);
  `magD` collects the absolute parts (`2^-17` per fp8 rounding, `2^-11` per block). With
  `l_i ≥ 1/2` (`lminB`: the lane that sets the running maximum has `p ≈ 1`, and a kept maximum has
  `alpha ≈ 1`), the division and the rounding to `float8e5` give
  `tolV = 9/8·((magA + 2·magD)(1 + 2^-22) + 2^-22·|attnRef| + 2^-148) + |attnRef|/8 + 2^-17` at the
  row's `nblk`. It is a magnitude bound: for a short row about `1.3·K + 0.3·|attnRef|`; over `T`
  blocks both parts grow as `(1 + ρ)^T`, and the dot's error on `acc` adds about
  `|attnRef|·((1 + ρ)^T − 1)`. **What `tolV` checks.** `tolV ≥ 9/8·K + |attnRef|/8 + 2^-17` at
  every `nblk ≥ 1` (`magA ≥ magCp ≥ 9/8·K + |attnRef|/8`). The `float8e5` rounding of any average
  of the row's own values `vEff[bh, j, e]` (keys `j` of `n`) is within `K + (K + |attnRef|)/8 + 2^-17`
  of `attnRef`. So any such average meets the `tolV` clause, whatever `q`, `k` and `sm_scale` are:
  the plain mean of the row's values does, and so does one key's value. By itself `tolV` checks the
  key set (the causal mask) and the range of the row's values; the weights are checked only through
  `tolO`.

Some values, computed from Model.lean's expressions. For random data, each figure is the median
over sampled rows of one head with `vIn` (at `N = 128`, every row of the head), non-causal, each
row at its own smallest `X`. The causal rows are alike. The figures agree with the reviewers'
round-2 evaluations within sampling noise; where samples differ, a range is given.

| data | `D` | `N` | `tolO` | `tolV` | `tolM` | `0` rejected |
|---|---|---|---|---|---|---|
| one key (a causal row's first query: `attnRef = v[0, e]`) | 64, 128 | any | `0.52·\|attnRef\|` to `2.1·\|attnRef\|` (test), `6.1·\|attnRef\|` to `9.3·\|attnRef\|` (benchmark, `D = 64`), `1.1·10^3·\|attnRef\|` to `2.4·10^3·\|attnRef\|` (`D = 128`) | `0.31·\|attnRef\|` | — | yes, by `tolV` |
| `q = 0`, `v = 1` on the last 64 keys | 64 | `2^16` | `8.5·10^-4` (`attnRef = 9.8·10^-4`) | `1.1·10^7` | `6.3·10^-4` | yes, by `tolO` |
| `q = 0`, `v = 1` everywhere | 64 | 1024 | `0.46` | `0.64` | `1.9·10^-5` | yes |
| `q = 0`, `v = 1` everywhere | 64 | 2048 | `0.68` | `1.13` | `2.9·10^-5` | yes, by `tolO` |
| `q = 0`, `v = 1` everywhere | 64 | 4096 | `1.25` | `2.55` | `4.8·10^-5` | no |
| `q = 0`, `v = 1` everywhere | 64 | `2^14` | `17.8` | `71` | `1.6·10^-4` | no |
| `q = 0`, `v = 1` everywhere | 64 | `2^16` | `9.7·10^5` | `1.1·10^7` | `6.3·10^-4` | no |
| test-like (`normal(0, 0.5)`, `σ = 0.5`) | 64 | 128, 1024, 4096 | `0.15`, `0.17`, `0.30` | `1.9`, `2.9`, `6.4` to `7.1` | `0.16`, `0.17`, `0.18` | no (`\|attnRef\|` median `0.049`, `0.017`, `0.009`) |
| test-like | 128 | 128, 1024, 4096 | `0.54`, `0.64`, `1.01` | `1.9`, `2.9`, `6.4` to `7.1` | `0.58`, `0.62`, `0.63` | no (`0.064`, `0.026`, `0.013`) |
| test-like | 64, 128 | `2^14` | `3.0`, `9.7` | `144`, `145` | `0.17`, `0.66` | no |
| benchmark-like (`randn`, `σ = 1.3`) | 64 | 1024 | `13.9` | `6.5` | `1.8` | no (`\|attnRef\|` median `0.50`) |
| benchmark-like | 128 | 1024 | `1.2·10^4` | `6.7` | `6.4` | no (`0.58`) |
| benchmark-like | 64, 128 | `2^14` | `90` to `140`, `1.0·10^5` to `1.4·10^5` | about `360`, about `360` | `1.7` to `2.0`, `6.7` to `6.9` | no |
| any, `vEff = 0` (a head without `vIn`) | any | `≤ 2^14` | `7.6·10^-6` (`≈ 2^-17`) | `7.6·10^-6` | as above | `o = ±0` |
| any, `vEff = 0` | 256 | `2^16` | `9.2·10^-4` | `7.6·10^-6` | as above | `o = ±0` |

**Where `Post` does not tell `o` from `0`.** `Post` holds both bounds, so it accepts `o = 0`
exactly when `|attnRef|` is at most both `tolO` and `tolV`. That is the case:

- **On the test's data** (`normal(0, 0.5)`, `σ = 0.5`): on 87% to 100% of the sampled elements,
  at every shape of `test_op`. The rejected elements are mostly on early causal rows (few keys).
  The fp8 floor is what `p`'s fp8 rounding alone allows (line 101, `2^-3` of each weight,
  `CvtSatF32E5m2`): `Σ p·|v| / (8·Σ p)`, about `0.05`. Per `N_CTX` (both `HEAD_DIM`s; the ranges
  cover this write-up's and both reviewers' round-2 evaluations):

  | `N_CTX` | rows | `0` meets `Post` | median `\|attnRef\|` | fp8 floor `≥ \|attnRef\|` |
  |---|---|---|---|---|
  | 128 | non-causal | 99.4% to 100% | `0.044` to `0.066` | 37% to 54% |
  | 128 | causal | 87% to 89.5% (`D = 64`), 98.7% to 99.1% (`D = 128`) | `0.064` to `0.094` | 26% to 38% |
  | 1024 | non-causal | 100% | `0.017` to `0.027` | 78% to 95% |
  | 1024 | causal | 96.5% to 99.7% | `0.026` to `0.044` | 54% to 76% |
  | 4096 | non-causal | 100% | `0.008` to `0.014` | 98% to 100% |
  | 4096 | causal | 96.8% to 99.7% | `0.013` to `0.032` | 64% to 93% |

  The fp8 floor explains most of the non-causal rows at `N_CTX ≥ 1024` (78% to 100% of their
  elements): there `|attnRef|`, a mean of values of both signs, is mostly below what any bound
  built from the per-lane facts can give. The floor
  alone would reject `0` on 46% to 74% of the elements at `N_CTX = 128`, where `Post` rejects at
  most 13%. On causal rows at 1024 and 4096 it would reject `0` on 7% to 46%, where `Post` rejects
  at most 3.5%. There most of the gap is `Num.DotE5m2`'s (L124): the score's `relDot8 D` in `pE` (about `1.1` at `D = 64`, `1.5` at `D = 128`) and
  `relDot8 64` in `psiA`. `tolO` is 3 to 20 times the floor.
- **On the benchmark's data** (`randn`, `σ = 1.3`): on every sampled element; `tolV` is `5.3` to
  `6.8` at `N = 1024` against `|attnRef|` of median `0.5`. The library's score error,
  `relDot8 D·Σ|q·k|` inside each exponent, lets each weight move by a factor up to
  `pE = 2^expErr`. In the sampled data its median is about `3.4` at `D = 64` and `87` to `100` at
  `D = 128`; the samples' maxima are `4.5` to `5.5` and `190` to `270`. With such factors
  the weighted mean can lie almost anywhere between the row's values, so a magnitude bound (`tolV`)
  is what these facts give. A sharper fact for the score's `tl.dot` would let `tolO` resolve it.
- **On rows spread over many blocks**: `Num.DotE5m2` allows each `tl.dot(p, v, acc)` to change its
  accumulator by `relDot8 64 = 257/2^14 ≈ 1.57%` (two `wgmma` instructions, each keeping the terms'
  bits only down to `2^-13` of the largest, the accumulator included). The accumulator is fed back
  once per block, so both tolerances grow as `(1 + relDot8 64)^nblk`. For `v = 1` everywhere `0` is
  accepted from `N = 4096` on (`tolO` `1.25`), and at `N = 2^16` the tolerances are about `10^6`
  and `10^7`. With the accumulation imprecise throughout (`2^30`), the real hardware loses accuracy
  this way too, though less: once `acc ≥ 2^13·p·v`, a block's products fall below the kept bits.

The library's part is library issue [L124](../../docs/library-review/ISSUES.md#l124) (a sharper
`DotE5m2`, for example relative to each chunk's largest product, would tighten `pE` and the
compounding). `Post` rejects `o = 0` only where one tolerance is below `|attnRef|`: a row's values
all close to `attnRef` (`tolV < |attnRef|` needs `max_j |vEff_j − attnRef|` below about
`0.53·|attnRef|` at 2 blocks and `0.22·|attnRef|` at 16, and never holds from 32 blocks on), as on
a causal row's first query (one key: `tolV = 0.31·|attnRef|`); `v = 1` rows at `N ≤ 2048`; and a
row whose values sit in few blocks (the first table row). What `Post` pins at every covered size
is the rest: no fault, every element written and finite, nothing else written, every head without
`vIn` at `±0` (a `float8e5` within `7.6·10^-6` of `0` is `±0`), and `M` (`tolM` about `0.2` at
`D = 64` on the test's data).

On the host's inputs the cap `Xh` is met: `σ·Σ_e|q_e·k_e|` is, by `attn_fwd_f16`'s Chernoff
estimate for the same distributions, below about `23` (test) and `240` (benchmark) at `D = 128`
except with probability below `10^-6`, against `Xh 128 = 256`; at `D = 64` the cap is `512` and the
magnitudes are half.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `Z`, `H`, `N_CTX`, `HEAD_DIM`, `sm_scale` (the Python
`float`, a double), `causal`, `warp_specialize`, and the storage of `q`, `k`, `v`, `o`
(`Buf .f8e5`; `v` the contiguous `[Z, H, HEAD_DIM, N_CTX]` storage) and `M` (`Buf .f32`).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `head_dim` | `HEAD_DIM ∈ {64, 128, 256}` | the wrapper allows `16` and `32` too (line 512), which line 188's `tl.static_assert(BLOCK_N <= HEAD_DIM)` rejects on the host (no outcome); the host uses `64` and `128` |
| `n_mul` | `0 < N_CTX`, `N_CTX % 128 = 0` | otherwise the last query tile of a head runs into the next head's rows (a race on `o`) and its last key block reads the next head's keys; every host size is a multiple of `128` |
| `n_le` | `N_CTX ≤ 2^16` | the proof's numeric bounds (`N/64 ≤ 1024` blocks); the host's largest is `2^14` |
| `zh_pos` | `0 < Z`, `0 < H` | `off_hz // H` (line 191) needs `H ≠ 0`; an empty batch is not covered |
| `zh_lt` | `Z·H < 2^16` | the grid's `y` is `Z·H` (line 549), within CUDA's limit; the host's largest is `192` |
| `elems_lt` | `Z·H·N_CTX·HEAD_DIM < 2^31` | line 65's `offsetv_y = offset_y·HEAD_DIM + lo` and its steps (line 109) are `int32` values that do not wrap; with it `y_dim` (line 194), every row offset (lines 208–209, 243) and the descriptors' shape entries are below `2^31`; the host's largest is `2^28` |
| `scale` | `sm_scale` finite, in `[2^-64, 2^64]` | keeps `qk_scale` (line 219) a finite positive `float32` within `2^-22` relative of `sm_scale·log₂e`; the host passes `0.5` and `1.3` |
| `lens` | `Z·H·N_CTX·HEAD_DIM` elements in `q`, `k`, `v`, `o`, `Z·H·N_CTX` in `M` | the kernel's accesses (`torch.empty_like`, `torch.empty`, the `.contiguous()` `v`) |
| `align` | the four tensors' base addresses are multiples of 16 | TMA's requirement for a descriptor's base (`TensorDesc.Valid`); PyTorch's allocations are |
| `disj` | `o` shares no byte with `q`, `k`, `v`, `M`, and `M` none with `q`, `k`, `v` | the outputs are fresh tensors (lines 513, 521) |

The descriptors: `descArg` (for `q`, `k`, `o`) is the host `TensorDescriptor`
`⟨b.ptr 0, [Z·H·N_CTX, HEAD_DIM], [HEAD_DIM, 1], block_shape, "zero"⟩` when `warp_specialize` is
false and the tensor's pointer when it is true; `descArgV` (for `v`) is
`⟨v.ptr 0, [HEAD_DIM, Z·H·N_CTX], [N_CTX, 1], [HEAD_DIM, 64], "zero"⟩` or the pointer. The model's
`maybe_make_tensor_desc` returns the host descriptor, or builds the same one (`make_desc`,
`make_descV`), so both kinds are covered.

`Pre d init`: the five buffers are valid in the initial memory (`Buf.Valid`). Any input values are
allowed: `Post` says what holds for covered heads, and only that `o` and `M` hold values for the
others.

**The host's use is covered.** Every test and benchmark shape meets `Inst.ok`: `HEAD_DIM ∈ {64,
128}`, `N_CTX ∈ {128, 1024, 4096, 2^10, …, 2^14}`, `Z·H ≤ 192`, `Z·H·N_CTX·HEAD_DIM ≤ 2^28`,
`sm_scale ∈ {0.5, 1.3}`, fresh tensors (16-byte aligned, disjoint; `v`'s storage the contiguous
transposed tensor), both values of `causal` and of `warp_specialize`. Their values are finite
`float8e5` (casts of finite `float16` values below `57344`) with `σ·Σ|q·k|` below the cap except
with negligible probability (Specification), so the heads the script runs are covered.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md)), as for `attn_fwd_f16` (the same host code):
every launch is the direct JIT launch `_attn_fwd[grid](…)` of lines 562–574 through `attention(…)`
(line 677 in the test; line 754, timed by `triton.testing.do_bench`, in the benchmark). Point 4:
the launch gives the constexprs and `num_stages=2`, `num_warps=4`; `extra_kern_args` is empty on
the H200 (`is_hip()`, `is_blackwell()` false); no `warmup`, `triton.compile`, `triton.Config` or
heuristics; a bare `@triton.jit`, so `launch` keeps the defaults (`enable_fp_fusion = true`,
`debug = false`, `num_ctas = 1`, no `launch_pdl`). Point 5: when `warp_specialize` is true the
kernel builds descriptors; the allocator (lines 543–546) returns a fresh `torch.empty` scratch that
holds no byte of the tensors. Point 6: no `CompiledKernel`. Point 8: the host descriptors leave
`round_f32_to_tf32` false. Point 9: no stream capture or `do_bench_cudagraph`. **Part (B)**, the
environment, is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md)): as for `attn_fwd_f16`. The lines
the front end does not visit on a covered launch are 66–67, 88–93, 99–100 and 200–202 (the
compile-time `else` branches of `dtype == tl.float8e5` and `FP8_OUTPUT`, and line 87's Blackwell
branch).

## Assumptions

Twelve assumptions, all of kind `[rule]`, each cited as the library states it
([num.md](../../docs/rules/num.md)):

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d` (within
  its scope: part (A), which the launches meet, and part (B), which the claim relies on).
- `dot` — `Rules.Num.DotE5m2 P`: `tl.dot` of `float8e5` operands into `float32` with
  `maxImpreciseAcc = 2^30`, any `input_precision` (lines 73, 103): each output element within
  `relDot8 K·(|acc| + Σ|a·b|) + absDot8 K` of `acc + Σ a·b`, with room in `float32`.
- `dot_total` — `Rules.Num.DotTotal P` at the configuration `cfg8` (written out).
- `cvt` — `Rules.Num.CvtSatF32E5m2 P`: `x.to(tl.float8e5)` of a `float32` (lines 101, 245) is the
  saturating rounding to nearest even, up to NaN payload.
- `cvt_total` — `Rules.Num.CvtTotal P binary32 e5m2 .rn`.
- `exp2`, `exp2_ninf`, `exp2_total` — `Rules.Num.Exp2F32 P` (within `2^-22` relative and `2^-126`
  absolute of `2^x`, lines 82, 84), `Rules.Num.Exp2F32NegInf P` (`exp2(−∞) = +0`, line 84 at the
  first block), `Rules.Num.MathTotal P .exp2 .binary32`.
- `log2`, `log2_total` — `Rules.Num.Log2F32 P` (line 241, `2^-22` relative, `2^-149` absolute),
  `Rules.Num.MathTotal P .log2 .binary32`.
- `div_full`, `div_total` — `Rules.Num.DivFull P` (line 242), `Rules.Num.DivTotal P .binary32`.

There is no `[kernel]` assumption. The IEEE operations, the constants and the descriptors'
padding are the library's definitions.

## Proof

`main` applies `TVA.meets_independent`, as `attn_fwd_f16`'s:

- `hdebug` by `launch`'s definition; `hvalid` (`launch_valid`): the grid `(N/128, Z·H, 1)` within
  CUDA's limits with fewer than `2^31` programs, the eight block shapes powers of two.
- Footprints: every program reads `q`, `k`, `v` (`Rd`); program `i`, the tile
  `(start_m, off_hz) = (i mod g0, i / g0)`, writes rows `128·i …` of the `[Z·H·N, D]` view of `o`
  and the same elements of `M` (`Wr`, `row0`); tiles are disjoint (`Wr_disj`) and inside
  `outBytes`, which is disjoint from `Rd` (`outBytes_disj`).
- `hsafe` (`safe_program`): lines 189–192 (`safe_idiv1`, `safe_irem1`); lines 195–206
  (`safe_maybe_desc` for `q`, `k`, `o`, giving `descOf`, and `safe_maybe_descV` for `v`, giving
  `descV`; a built descriptor passes `safe_make_tensor_descriptor` and equals the host one, the
  `int64` stride `N_CTX` by `IntV.toInt_cast`); lines 208–219 (`offset_y = bh·N`,
  `qo_offset_y = 128·i`, `m_i = −∞`, `l_i = 1.0`, `qk_scale_facts`); line 221 (`safe_load_rows`,
  `QReads`); lines 225–239 (`safe_inner1`, `safe_inner2`, `safe_inner3`: the loops by
  `safe_forRange` with `offsetk_y = bh·N + 64t` and `offsetv_y = bh·N·D + 64t`, which `elems_lt`
  keeps from wrapping, keeping `StInv`); lines 241–245 (`safe_epi`: `log2`, `+`, `/`, the store of
  `M`, `safe_cvt8`, `safe_store_rows` for `o`).
- One block (`row_step_plain`, `row_step_masked`, for every row of a covered head and every
  `X ≤ Xh`, `RowSt`): `Num.DotE5m2` at each score (`qk_lane`) and each `p·v` element (`pv_lane`); line 98's
  lanes (`safe_load_v`, `v_lanes`: inside the descriptor exactly when `bh·D < Z·H`, then
  `v[bh, j, e]`, otherwise the padding `+0`, so each lane is `vEff`); the running maximum
  (`mstep_plain`, `mstep_masked`: `MInv`, each exponent within `E = expErr` of `w·s_j − m_ij` and at
  most `2^-10`, a masked lane's below `−2^19`); `p` (`lane_p`: within `psiP·wt + 2^-126` of the
  exact `wt = 2^(w·s_j − m_ij)`, and at least `wt·(1 − 2^-22)/pE − 2^-126`), its fp8 rounding
  (`lane_p8`, from `CvtSatF32E5m2`: `p ≤ 1 + 2^-9` needs no saturation), `alpha`
  (`alpha_facts`); `l_i` (`block_l`, `lstep`: `LInv`, two-sided) and each element of `acc` (`astep`,
  `acc_step8`: `AInv` with the bounds `esR`, `ebR`, which are `errS`, `errB` in the row's terms).
  The second bound (`mag_tail`, module `Mag`): `mstep_plain` and `mstep_masked` also give that the
  maximum is kept or some lane's exponent is within `2^-11` of `0`, so `l_i ≥ lminB t =
  1 − 2^-9 − t·2^-12 ≥ 1/2` (`p_big`, `lmin_real`); each lane's `q·v − r·p` (`lane_pq`, `lane_E`),
  `tl.sum(p)` (`sum64_err`), line 106 (`f32_mulAdd`), line 95 (`f32_mul`) and the dot's error (from
  `astep`) give one block of `|acc − r·l_i| ≤ magA·l_i + magD` for every `r` and `K` (`estep_real`).
- The end of a row (`row_out`): after the row's `nblk` blocks (`N/64`, or `2·start_m + 2` when
  causal), `final_o` (`DivFull`, `CvtSatF32E5m2`, `ratio8`, `l_final`), `final_v` (`tolV`, at
  `r = attnRef`, with `|attnRef| ≤ 57344` by `ref_le` and `K ≥ |vEff_j − attnRef|` over the keys,
  `veOf_key`) and `final_M` (`Log2F32`)
  give `|o − Sx/W|` and `|M − log₂ W|` with `W = Σ_j 2^(w·s_j)`, which is `wsum` (`ew_weight`:
  `2^(σ/ln 2·s) = exp(σ·s)`), `Sx/W = attnRef` (`Sx_partS`, `veOf_key`), `esR = errS`, `ebR = errB`
  (`Mg_blkMag`, `AbV_blkAbs`, `psum_below`, `blk_sum`), and the constants equal to `Model`'s
  (`rhoA_r`, `psiA_r`, `pE_r`, `Fhi_r`, `Flo_r`, …); `model_factors` checks the numeric side
  conditions (`Fp ≥ 2^-15`, `Fhi + aL·pE ≤ 2^15`, …) at `N ≤ 2^16`.
- `hpost`: element `(bh, n, e)` of `o` and `(bh, n)` of `M` lie in program `bh·g0 + n/128`'s tile
  (`prog_of`), whose `Qp` gives them; a byte outside `outBytes` is in no write footprint.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref`, and `Platform.sm90ref.dotE5m2`,
`cvtSatF32E5m2`, `exp2F32`, `exp2F32NegInf`, `log2F32`, `divFull`, `Platform.ref.dotTotal`,
`cvtTotal`, `mathTotal`, `divTotal`.

`satisfiable`: the instance `Z = H = 1`, `N_CTX = 128`, `HEAD_DIM = 64`, `sm_scale = 0.5`, not
causal, host descriptors, the tensors at `2^21`, `2·2^21`, `3·2^21`, `4·2^21` and `M` at `5·2^21`
(`testInst`), in a memory where every byte is present (`testMem`).

`main`, `consistent` and `satisfiable` use only `propext`, `Classical.choice` and `Quot.sound`.

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float8e5` only.** This folder is the `float8e5` run (`FP8_OUTPUT = True`): `q`, `k`, `v`, `o`
  are `float8e5` (`v` transposed in storage) and `M` is `float32`. The `float16` run
  (`FP8_OUTPUT = False`, the `triton-fp16` provider) is excluded here; it is `attn_fwd_f16`. The
  backward kernels are host lines here (`source.unused`), other folders' kernels.
- **The H200.** `IS_HOPPER = True`; other devices are outside `Compiler.Faithful`'s point 7.
- **The launch's configuration.** `BLOCK_M = 128`, `BLOCK_N = 64`, `num_stages = 2`,
  `num_warps = 4` (SETUP.md).
- **`HEAD_DIM ∈ {64, 128, 256}`, `N_CTX` a positive multiple of `128` at most `2^16`,
  `Z·H < 2^16`, `Z·H·N_CTX·HEAD_DIM < 2^31`, `sm_scale ∈ [2^-64, 2^64]`** (Conditions): the host's
  shapes are inside.
- **Covered heads** (`HeadOk`): finite inputs and `σ·Σ_e|q·k| ≤ Xh D`. The cap keeps each
  exponent's error below `13`, the causal mask's `−10^6` below every attended exponent, and the
  numeric side conditions; for another head the claim says only that `o` and `M` hold some value.
  The benchmark's inputs at `D = 128` are near the cap (about `240` against `256`).
- **The reference is `vEff`'s attention**, not `v`'s, for the heads the kernel reads as zeros
  (Specification). This is the kernel's behaviour, not a narrowing of the inputs.
- **The tolerances' form** (Specification): `tolO` is data-dependent (`errS`, `errB`, `blkMag` of
  the row's values). `tolV` is a magnitude bound in `K`, which any average of the row's own values
  meets: alone it checks the key set and the range of the values, not the weights. On the test's
  and the benchmark's data, `Post`'s accuracy part does not tell `o` from `0`. On the test's data
  `0` meets `Post` on 87% to 100% of the sampled elements, and on the benchmark's data on all of
  them (Specification, "Where `Post` does not tell `o` from `0`"). On the test's non-causal rows at
  `N_CTX ≥ 1024`, that is mostly the fp8 rounding of `p` itself (78% to 100% of the elements). At `N_CTX = 128` and on causal rows, most
  of the gap is the library's `Num.DotE5m2`. The same holds on the benchmark's data and on rows
  spread over many blocks (`(1 + relDot8 64)^nblk`, about `10^6` at `N = 2^16`); the library's part
  is library issue [L124](../../docs/library-review/ISSUES.md#l124). Narrowing `N_CTX` would not
  help: the bounds are loose at the host's sizes already.
- **The environment** (part (B) of `Compiler.Faithful`'s scope) is relied on, not checked.

Modelling decisions (as `attn_fwd_f16`'s, with the fp8 changes):

- `Z`, `H`, `N_CTX` are `int32` (`IntV.ofNat`; at `1` the constant `1`, the same model);
  `sm_scale` is the `float32` rounding of the double (`FloatV.hostF32`).
- `dtype` (`tl.float8e5`), `FP8_OUTPUT` (`True`), `IS_HOPPER` (`True`) are fixed on covered launches
  and are no parameters; `warp_specialize` only chooses the descriptor kind (`descArg`, `descArgV`).
- The compile-time `if`s on `dtype`, `FP8_OUTPUT`, `IS_HOPPER`, `STAGE`, `isinstance` are decided
  as Python decides them; the visited branches are the model.
- Line 98 is a descriptor load of the `[HEAD_DIM, BLOCK_N]` block at `[0, offsetv_y]` followed by
  `.T`; lines 198–199 build `desc_v` with the `int64` stride `N_CTX` (`IntV.cast`).
- `.to(tl.float8e5)` (lines 101, 245) is `tl.cvt8`, the platform's conversion, which `CvtSatF32E5m2`
  describes; both `tl.dot`s are `cfg8` (`tf32`, `2^30`, the defaults for fp8 operands on sm_90).
- Floating-point fusion as in `attn_fwd_f16`: lines 76 and 106 are `tl.mul_add`, line 81 `tl.subc`.
- `tl.max(qk, 1)`, `tl.sum(p, 1)` are row reductions in any tree; `/` is `P.div`.
- The hint of line 59 is a check at the definition of `lo`; line 70's hint is dropped.
