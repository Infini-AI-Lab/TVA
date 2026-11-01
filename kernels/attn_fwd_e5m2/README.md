# attn_fwd_e5m2

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

**Read this first.** The kernel as given reads `v` for the first `⌈Z·H/HEAD_DIM⌉` heads only. For
every other head (`bh·HEAD_DIM ≥ Z·H`), line 98 reads the zero padding of `v`'s descriptor, and the
head's output is `±0`. On the host's shapes that is every head of a launch but 1 to 3. `Post`
compares `o` with the attention of the values the kernel uses (`vEff`, Specification), so for those
heads it certifies `±0`. For the other heads its accuracy part rests on the library's chunk facts
for the fp8 `tl.dot` (`Num.DotE5m2Chunk`, library round 24, "What changed (library round 24)"
below). The first tolerance is about 1.75 to 5.5 times the fp8 floor (what the fp8 rounding of `p`
alone allows) on the test's data and about 5 to 18.5 times on the benchmark's. `Post` now rejects
`o = 0` on many elements. On the test's non-causal rows at `N_CTX ≥ 1024`, `|attnRef|` is mostly
below the fp8 floor itself, and there `Post` accepts `0` (Specification, "Where `Post` does not tell `o`
from `0`"). What remains of the library's part is the compounding of the `p·v` accumulator over the
key blocks. It is the bound of the cited fact, `DotE5m2Chunk`, not the device's behaviour (library
issue [L127](../../docs/library-review/ISSUES.md#l127)).

Source: [`source/06-fused-attention.py`](source/06-fused-attention.py) (tutorial 06 as set up in
[SETUP.md](SETUP.md): the `@triton.autotune` decorator removed, the config `BLOCK_M=128,
BLOCK_N=64, num_stages=2, num_warps=4` passed by the launch, the config's `pre_hook` called by the
host; the same file as `attn_fwd_f16`'s). Lean: [`lean/`](lean/), the module `AttnFwdE5m2`
(namespace `Kernels.AttnFwdE5m2`) and its parts:

| Module | What it holds |
|---|---|
| `AttnFwdE5m2.Model` | the model (`attn_fwd_inner`, `maybe_make_tensor_desc`, `attn_fwd`, `model`, `launch`), the conditions (`Inst`, `Inst.ok`, `Pre`), the specification (`Post`, `Spec`, the references and tolerances), the assumptions |
| `AttnFwdE5m2.Real`, `.Lane`, `.RowMax`, `.RowAcc`, `.Step` | one query row through the loop, in real numbers: the running maximum (`MInv`), `l_i` (`LInv`), the magnitude of `acc` (`ACr`) and the `p·v` `tl.dot`'s two chunks (`errB2_le`, `astep`) |
| `AttnFwdE5m2.Final` | `l_i` after the loop (`l_final`), the log-sum-exp (`final_M`), the `Model` constants as real numbers, their numeric facts (`model_factors`) |
| `AttnFwdE5m2.Mag` | the second bound on the output, over the computed weights: `l_i ≥ lminB t` (`lmin_real`), one block of `\|acc − r·l_i\| ≤ magA·l_i + magD` (`estep_real`), the end of a row (`final_v`) |
| `AttnFwdE5m2.Cen` | the first bound, centred: one block of `acc − r·l_i` against `kap` times the centred sum of the exact weights (`cstep_real`, `lane_cen`, `block_cen`, `cl2_cen`), the row's recursions `acR`, `ecR` and the invariant `CInv` |
| `AttnFwdE5m2.Facts` | the grid, `int32` arithmetic, the descriptors (`descOf`, the transposed `descV`), the constants, `qk_scale` (`qk_scale_facts`, `qk_scale_w`) |
| `AttnFwdE5m2.Row`, `.Tile`, `.Inner` | a row's state (`RowSt`, `mag_tail`, `cen_tail`, `row_step_plain`, `row_step_masked`), the two `tl.dot`s' lanes (`qk_lane`, `pv_lane`), the tile loads and stores (`safe_load_v` for `v`), the loop of `_attn_fwd_inner` and its three calls |
| `AttnFwdE5m2.Program` | the row's sums in the specification's terms (`acR_accA`, `ecR_errC`, `centC_final`), one row's end (`row_out`), one program (`safe_program`) |
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

- the accuracy facts are the chunk facts `Num.DotE5m2Chunk` (library round 24: each chunk of `32`
  products within `2^-7` of its largest lifted term, the accumulator included, plus `2^-12` of the
  new sum and `2^-126`) and `Num.CvtSatF32E5m2` (one saturating rounding to `float8e5`, `2^-3`
  relative, `2^-17` absolute);
- with these constants the accumulator's error compounds by about `1.62%` per block, so `acc` is
  bounded block by block in the order the loop runs (`errC`, `accA`, below) instead of by a closed
  factor, and `l_i` from both sides (`Fhi`, `Flo`);
- each row's exact exponents are the base-2 `w·s_j` with `w = sm_scale/ln 2` (in `RowCtx`, with
  `|qk_scale − w| ≤ 2^-22·w`), so the weights are `exp(sm_scale·s_j)` directly;
- the computed exponents are kept at most `2^-10` (so `p ≤ 1 + 2^-9`, far inside `float8e5`), and
  the coverage cap on the scaled lifted scores is `Xh D = 2^10/⌈D/32⌉`, which keeps each exponent's
  error `expErr ≤ 13`;
- `v` is read through the transposed descriptor (`descV`, `safe_load_v`), which reads zeros for most
  heads (`vEff`, Specification);
- `Inst.ok.elems_lt` (`Z·H·N_CTX·HEAD_DIM < 2^31`, for `offsetv_y`) replaces `rows_lt`.

## What changed (library round 24)

- **The fact.** Round 24 withdrew `Num.DotE5m2` (L125), which this claim cited for both `tl.dot`s,
  and added the chunk facts (L124). `Num.DotE5m2Chunk` describes each output element as its
  `⌈K/32⌉` chunks of `32` products in order, each step within
  `2^-7·M_c + 2^-12·|s_c + c_c| + 2^-126` of the exact step. Here `M_c` is the larger of the running
  sum and the chunk's largest lifted product (`liftMag`: a subnormal operand counts as `2^-14`). The
  claim cites it as `dot` for lines 73 and 103.
- **The score.** Each key's score error is now the chunk facts' own bound from the exact `q` and `k`
  (`scoreErr`: the chunks' largest lifted products and exact partial sums), or the whole-dot bound
  `relLift8 D·scoreLift` if that is smaller. Before, it was `relDot8 D·Σ|q·k|`. `Post` quantifies a
  bound `Z ≥ σ·scoreErr` next to `X ≥ σ·scoreLift`. Each weight's factor `pE` (median) drops from
  about `100` to about `2.3` on the benchmark's data at `D = 128`, and from `3.4` to `1.3` at `D = 64`.
- **`tolO`, centred.** `tolO` now follows `acc − attnRef·l_i` over the computed `p` (lines 95 and
  106 rescale both by the same `alpha`) against the centred sum `Σ w_j·(vEff_j − attnRef)` of the
  exact weights. Since that sum is `0` exactly after the last block, the score error enters once,
  as the spread `hw ≈ 1/8 + (pE − 1)` of the weights' ratios times `Σ w·|vEff − attnRef|` (and
  through `l_i ≥ Fp·Σw`, `Fp ≈ 1/pE`). It no longer enters as `pE²·Σ w·|v|`.
- **The `p·v` `tl.dot`** is bounded over its two chunks: `cS8·|acc| + cC8·Σ|p·v| + cP8·(m₀ + m₁) +
  cA8`, with `m_c` the chunk's largest lifted product. Its share on the accumulator, `cS8 ≈ 1.62%`
  per block, compounds over the blocks. That is `DotE5m2Chunk`'s bound (each step relative to
  `M_c ≥ |acc|`, with its constant `2^-7`, about twice the window's worst chunk), not the device's
  behaviour, whose error grows at most
  linearly in the chunks (L127). It is what remains: it is slightly above `DotE5m2`'s `1.57%`, so
  rows spread over many blocks are a little looser than before (`v = 1` at `N = 2^14`: `tolO`
  `19.3` against `17.8`). Round 26's `DotE5m2ChunkLoss` gives a sharper bound per chunk, which this
  claim does not cite.
- **Subnormal operands** of `q`, `k`, `v` and of `p` after its fp8 conversion count as `liftMag`
  wherever the fact does (`scoreLift`, `scoreErr`, `blkLW`, `blkVL`, and `p`'s lift in `accA`'s
  `eps8 + 2^-14`). `tolV`'s lifted products carry
  `iotaV = min(1, 2^16·(K + |attnRef|))`, so a head without `vIn` stays at `±0` at every covered
  size. This uses the fact that a nonzero `float8e5` is at least `2^-16` (`f8_abs_ge`, by evaluating
  the `256` patterns).
- **One condition changed.** `HeadOk` caps the scaled lifted magnitude `σ·scoreLift` where it capped
  `σ·Σ|q·k|`. This is needed because the whole-dot bound that the cap's numeric side conditions use
  (`relLift8 D·scoreLift + absLift8`, `Dot8Chunk.DotF8Chunk.liftAcc`) is relative to lifted
  magnitudes. The two agree when no operand is subnormal, and the host's data meet the new cap as
  they met the old one (Specification). The other conditions are unchanged.
- **`tolV` is kept**, adapted to the chunk facts (`cS8`, `cP8·iotaV`). It is the tighter bound
  where `pE` is large (rows near the cap) and for a head without `vIn` at large `N`.

The tolerances, before and after, are in the Specification's table.

## Specification

Write `D = HEAD_DIM`, `N = N_CTX`, `σ = sm_scale` (the double, as a rational: `sigma`). Element
`(bh, n, e)` of `q`, `k`, `o` (for the head `bh = z·H + h`) is element `idx d bh n e =
(bh·N + n)·D + e` of its buffer (contiguous); element `(bh, n, e)` of the transposed `v` is element
`vIdx d bh n e = (bh·D + e)·N + n`; element `(bh, n)` of `M` is element `bh·N + n`. For query `n`,
its keys are `keys d n`: every `j < N`, and only `j ≤ n` when `causal`. With the input values in
`init` (`val8`: a `float8e5` element's value), the exact score is `score = Σ_e q[bh, n, e]·k[bh, j, e]`
and the exact weight is `weight = exp(σ·score)` (real `exp`). Two quantities come from the library's
chunk facts for the score's `tl.dot` (`Num.DotE5m2Chunk`), on the exact `q` and `k`:

- the lifted magnitude `scoreLift = Σ_e liftMag(q[bh, n, e])·liftMag(k[bh, j, e])`. Here
  `liftMag x` is `0` for `x = 0` and `max(|x|, 2^-14)` otherwise: a subnormal operand counts as the
  smallest normal, as the tensor core aligns it. For normal values `scoreLift = Σ_e|q·k|`;
- the error bound `scoreErr`: the chunk facts' bound over the `⌈D/32⌉` chunks of `32` products from
  a zero accumulator (`Dot8Chunk.errB`). It starts at `err_0 = 0`, and each chunk gives
  `err_{c+1} = (1 + 2^-7 + 2^-12)·err_c + 2^-7·max(|s_c|, m_c) + 2^-12·|s_{c+1}| + 2^-126`, with
  `s_c` the exact sum of the first `c` chunks and `m_c` chunk `c`'s largest lifted product
  `liftMag(q_e)·liftMag(k_e)`. It is each key's own bound, from the chunks' largest terms and partial
  sums, not from `Σ|q·k|`.

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
`float8e5`, so is every element of `v` in it when `vIn`, and every scaled lifted magnitude
`σ·scoreLift ≤ Xh D = 2^10/⌈D/32⌉` (`512`, `256`, `128` at `D = 64, 128, 256`).

`Post d init fin` says three things.

- **The output `o`.** Every element `(bh, n, e)` of `o` (`bh < Z·H`, `n < N`, `e < D`) holds some
  `float8e5` value `c` in `fin`. When head `bh` is covered, `c` is finite, and two bounds hold.
  First, take any `X ≤ Xh D` that bounds the row's scaled lifted magnitudes
  (`σ·scoreLift(n, j) ≤ X` for every key `j` of `n`) and any `Z` that bounds its scaled score
  errors (`σ·scoreErr(n, j) ≤ Z`). Then `|c − attnRef| ≤ tolO d init bh n e X Z`. Second, for every
  `K` that bounds `|vEff[bh, j, e] − attnRef|` over the keys `j` of `n`,
  `|c − attnRef| ≤ tolV d init bh n e K`.
- **The log-sum-exp `M`.** Every element `(bh, n)` of `M` holds some `float32` value `c`. When head
  `bh` is covered, `c` is finite and, for every such `X` and `Z`,
  `|c − lseRef| ≤ tolM D N X Z + 2^-24·|lseRef|`.
- **Nothing else changes.** Every byte outside the elements of `o` and `M` (`outBytes`: the first
  `Z·H·N·D` elements of `o`, the first `Z·H·N` of `M`) is the same in `fin` as in `init`.

**The tolerances** (`AttnFwdE5m2.Model`: "The constants", "The output's tolerance", "The second
bound on the output") are explicit expressions of `D`, `N`, `X`, `Z` and of the row's exact data.
They are built only from the library's constants: the chunk facts' `2^-7`, `2^-12`, `2^-126` (both
`tl.dot`s) and their whole-dot bound `relLift8 D = (1 + 2^-7 + 2^-12)^⌈D/32⌉ − 1` relative to
`scoreLift` (`Dot8Chunk.DotF8Chunk.liftAcc`), `2^-22` (`Exp2F32`, `DivFull`, `Log2F32`), the
`float32` roundings (`2^-24`, `2^-150`) and the `float8e5` rounding (`2^-3`, `2^-17`):

- each computed exponent is within `expErr D X Z = scoreE + 3X/2^20 + 2^-50` of the exact
  `w·s_j − m_i`. Here `scoreE = min(3Z/2, relLift8 D·3X/2 + 2^-55)` is `qk_scale ≤ 3σ/2` times the
  smaller of the key's `scoreErr` and the whole-dot bound `relLift8 D·scoreLift`. So each `p` is
  within a factor `pE = 2^expErr` (and `exp2`'s `2^-22`) of its exact weight. Its fp8 rounding is
  between `a8 = 7/8·(1 − 2^-22)/pE` and `b8 = 9/8·pE·(1 + 2^-22)` times that weight, up to
  `eps8 = 2^-17 + 2^-125`;
- `l_i` is between `Flo·Σ − aL` and `Fhi·Σ + aL` (`Σ` the weights in the running maximum's units),
  `Fhi`, `Flo` products of per-block factors `stepL`, `qL` over `N/64` blocks (`FhT t` after `t`
  blocks);
- the `p·v` `tl.dot` (line 103: `64` products in two chunks) is within
  `cS8·|acc| + cC8·Σ|p·v| + cP8·(m₀ + m₁) + cA8` of `acc + Σ p·v` (`Dot8Chunk.errB` at two chunks).
  Here `m_c` is chunk `c`'s largest lifted product, `cS8 = (2^-7 + 2^-12)(2 + 2^-7 + 2^-12) ≈ 1.62%`,
  `cC8 ≈ 0.83%`, `cP8 ≈ 0.79%` and `cA8 ≈ 2^-125`;
- **`tolO`, centred** (`errC`, `accA`). Lines 95 and 106 rescale `acc` and `l_i` by the same
  `alpha`, so the proof follows `acc − r·l_i` for `r = attnRef`. Write `kap = (a8 + b8)/2` and
  `hw = (b8 − a8)/2`. `hw` is about `1/8 + (pE − 1)`: the spread of the keys' ratios of fp8 `p` to
  their exact weights. After `t` blocks, `acc − r·l_i` is within `2^-m_i·errC t + accA t·l_i` of
  `kap·2^-m_i·centC t`, where `centC t = Σ w_j·(vEff_j − attnRef)` over the keys before block `t`
  (`CInv`). One block gives
  `errC (t+1) = (1 + gA)·errC t + gA·kap·|centC t| + (gA + etaA)·(|r| + accA t)·FhT t·partW t +
  (hw + cC8·b8)·blkC t + cP8·b8·blkLW t`.
  Here `blkC t = Σ w_j·|vEff_j − attnRef|` over block `t`, `blkLW t` is the sum over the block's two
  chunks of the largest `w_j·liftMag(vEff_j)`, and `partW t` is the weights' sum before block `t`.
  `etaA = (1 + dmax)(1 + 2^-22) − 1` is `alpha`'s error, and `gA ≈ cS8` is how much what `acc`
  carries grows per block (`alpha`, `acc·alpha`'s rounding, the dot's `cS8`). `accA (t+1)` is the
  larger of `(accA t + 3(1 + etaA)·|r|/2^24)/denX` and `|r|·coefS/denS`, plus twice the block's
  absolute parts (through `l_i ≥ 1/2`). `coefS ≈ 1/8 + 9/8·cC8` is `p`'s fp8 rounding against `r`,
  the dot, and `l_ij`'s and `l_i`'s roundings. The absolute parts are `(1 + cC8)·eps8·Σ|vEff_j − r|`
  over the block, `cP8·(eps8 + 2^-14)·blkVL t` (`blkVL`: the chunks' largest `liftMag(vEff_j)`),
  `|r|·(1 + cC8)/2^11` (`2^-17` per key against `r`) and roundings near `0`. After the last block,
  `centC = 0` exactly (`attnRef` is the weights' mean), and `l_i ≥ Fp·Σ` with `Fp = Flo − aL·pE`. So
  `tolO = 9/8·((errC/(Σw·Fp) + accA)·(1 + 2^-22) + 2^-22·|attnRef| + 2^-148) + |attnRef|/8 + 2^-17` at
  the row's `nblk` (the division, then the rounding to `float8e5`). The score's error enters once,
  through `hw` on `Σ w·|vEff − attnRef|` and through `Fp ≈ 1/pE`;
- `tolM` is `log₂`'s error of `l_i` against `Σ` (`lgB`), `Log2F32`'s, and the final rounding;
- `tolV d init bh n e K` (`Model.magA`, `magD`, `tolV`) compares `acc` with `r·l_i`, `r = attnRef`,
  over the computed `p`: lines 95 and 106 scale both by the same `alpha`, so no comparison of `p`
  with the exact weights is needed, and `pE` does not enter. After `t` blocks
  `|acc − r·l_i| ≤ magA t·l_i + magD t`. `magA (t+1)` is the larger of
  `((1 + cS8)(1 + 2^-24)·magA t + |r|·(4·2^-24 + cS8(1 + 2^-24)))/(1 − 3·2^-24)` (the old `acc`, and
  the dot's `cS8` on it) and `magCp/(1 − 2^-17 − 3(1 + 2^-17)·2^-24)`. Here
  `magCp = 9/8·K(1 + cS8) + |r|·(9/8·cS8 + 1/8 + 2^-17 + 3(1 + 2^-17)·2^-24) + 9/8·cP8·iotaV/2^14`:
  the block's `Σ q·v − r·Σ p` with `p`'s fp8 rounding `q`, the dot's error on it, `tl.sum(p)`'s and
  line 106's roundings. The dot's lifted products cost `liftMag`'s `2^-14` times
  `iotaV = min(1, 2^16·(K + |r|))`, which is `0` when every value of the row is `0` (a nonzero
  `float8e5` is at least `2^-16`). `magD` collects the absolute parts (`2^-17` per fp8 rounding,
  `2^-11` per block, `cP8·iotaV` on the lifted values). With `l_i ≥ 1/2` (`lminB`: the lane that sets
  the running maximum has `p ≈ 1`, and a kept maximum has `alpha ≈ 1`), the division and the rounding
  to `float8e5` give
  `tolV = 9/8·((magA + 2·magD)(1 + 2^-22) + 2^-22·|attnRef| + 2^-148) + |attnRef|/8 + 2^-17` at the
  row's `nblk`. It is a magnitude bound: for a short row about `1.3·K + 0.3·|attnRef|`. Over `T`
  blocks both parts grow as `(1 + cS8)^T`, and the dot's error on `acc` adds about
  `|attnRef|·((1 + cS8)^T − 1)`. **What `tolV` checks.** `tolV ≥ 9/8·K + |attnRef|/8 + 2^-17` at
  every `nblk ≥ 1` (`magA ≥ magCp ≥ 9/8·K + |attnRef|/8`). The `float8e5` rounding of any average
  of the row's own values `vEff[bh, j, e]` (keys `j` of `n`) is within `K + (K + |attnRef|)/8 + 2^-17`
  of `attnRef`. So any such average meets the `tolV` clause, whatever `q`, `k` and `sm_scale` are:
  the plain mean of the row's values does, and so does one key's value. By itself `tolV` checks the
  key set (the causal mask) and the range of the row's values; the weights are checked only through
  `tolO`.

Some values, computed from Model.lean's expressions. For random data, each figure is the median
(or share) over all rows of a head with `vIn`, each row at its own smallest `X` and `Z`, non-causal
unless stated. The ranges come from all rows of three such heads, widened to the reviewers'
independent samples; other samples may differ by a few percent at the ends of a range. "Before" is
the claim on `Num.DotE5m2` (before library round 24), as this write-up gave it.

| data | `D` | `N` | `tolO` before → now | `tolV` before → now | `tolM` before → now | `0` rejected (now) |
|---|---|---|---|---|---|---|
| one key (a causal row's first query: `attnRef = v[0, e]`) | 64, 128 | any | test: `0.52` to `2.1` → `0.31` to `0.33`; benchmark `D = 64`: `6.1` to `9.3` → `0.31` to `0.33`; `D = 128`: `1.1·10^3` to `2.4·10^3` → `0.32` to `0.39` (times `\|attnRef\|`) | `0.31·\|attnRef\|` → `0.31` to `0.32` times `\|attnRef\|` | — | yes, by both |
| `q = 0`, `v = 1` everywhere | 64 | 1024 | `0.46` → `0.46` | `0.64` → `0.66` | `1.9·10^-5` → `2.0·10^-5` | yes |
| `q = 0`, `v = 1` everywhere | 64 | 4096 | `1.25` → `1.31` | `2.55` → `2.67` | `4.8·10^-5` → `4.8·10^-5` | no |
| `q = 0`, `v = 1` everywhere | 64 | `2^14` | `17.8` → `19.3` | `71` → `81` | `1.6·10^-4` → `1.6·10^-4` | no |
| `q = 0`, `v = 1` everywhere | 64 | `2^16` | `9.7·10^5` → `1.2·10^6` | `1.1·10^7` → `1.9·10^7` | `6.3·10^-4` → `6.3·10^-4` | no |
| test-like (`normal(0, 0.5)`, `σ = 0.5`) | 64 | 128, 1024, 4096 | `0.15`, `0.17`, `0.30` → `0.085` to `0.091`, `0.097` to `0.104`, `0.19` to `0.20` (causal: `0.089` to `0.093`, `0.096`, `0.13`) | `1.9`, `2.9`, `6.4` to `7.1` → `1.9`, `2.9`, `6.6` to `7.3` | `0.16`, `0.17`, `0.18` → `0.030` to `0.032`, `0.037` to `0.038`, `0.041` to `0.042` | on 12% to 19%, 0%, 0% (causal rows: 33% to 39%, 4.5% to 6.6%, 1% to 2.2%) |
| test-like | 128 | 128, 1024, 4096 | `0.54`, `0.64`, `1.01` → `0.117` to `0.119`, `0.134` to `0.146`, `0.27` (causal: `0.12`, `0.133` to `0.134`, `0.19`) | `1.9`, `2.9`, `6.4` to `7.1` → `1.9`, `2.9`, `6.6` to `7.3` | `0.58`, `0.62`, `0.63` → `0.093` to `0.099`, `0.116` to `0.120`, `0.126` to `0.130` | as at `D = 64` |
| benchmark-like (`randn`, `σ = 1.3`) | 64 | 1024 | `13.9` → `0.56` to `0.60` (causal `0.42` to `0.46`) | `6.5` → `6.7` (causal `5.6`) | `1.8` → `0.39` | on 45% to 57% (causal: 64% to 69%) |
| benchmark-like | 128 | 1024 | `1.2·10^4` → `1.16` to `1.54` (causal `0.83` to `1.01`) | `6.7` → `6.7` to `6.8` (causal `5.6` to `5.7`) | `6.4` → `1.2` | on 9.5% to 19% (causal: 28% to 38%) |
| any, `vEff = 0` (a head without `vIn`) | any | `≤ 2^14` | `7.6·10^-6` → `7.6·10^-6` at `X = 0`; at the cap `1.0·10^-5` (`N = 4096`), `1.0·10^-4` to `1.4·10^-4` (`2^14`, `D = 64` to `256`) | `7.6·10^-6` → `7.6·10^-6` | as above | `o = ±0` |
| any, `vEff = 0` | 256 | `2^16` | `9.2·10^-4` → `7.8·10^-6` at `X = 0`, `10` at the cap | `7.6·10^-6` → `7.6·10^-6` | as above | `o = ±0` |

"At the cap" is `X = Xh D` with the whole-dot score bound (`pE` about `5.5·10^3` to `6.8·10^3`).
The test-like (non-causal) `|attnRef|` medians are `0.044` to `0.068`, `0.017` to `0.026`, `0.008` to
`0.014` at `N = 128, 1024, 4096`; the benchmark's are `0.51` to `0.61`.

**Where `Post` does not tell `o` from `0`.** `Post` holds both bounds, so it accepts `o = 0`
exactly when `|attnRef|` is at most both `tolO` and `tolV`. The fp8 floor is what `p`'s fp8 rounding
alone allows (line 101, `2^-3` of each weight, `CvtSatF32E5m2`): `Σ w·|vEff| / (8·Σ w)`, about `0.05`
on the test's data and `0.09` on the benchmark's. A bound built from the per-lane facts cannot go
below it.

- **On the test's data** (`normal(0, 0.5)`, `σ = 0.5`): `tolO` is `0.085` to `0.12` at
  `N_CTX = 128`, `1.75` to `2.5` times the floor. It is `0.096` to `0.146` at `1024` and `0.13` to
  `0.27` at `4096` (both `HEAD_DIM`s, causal and not; before: `0.15` to `1.01`), at most `5.5` times
  the floor. Where `0` meets `Post` (both `HEAD_DIM`s; all rows of three heads, widened to the reviewers'
  samples; the "before" ranges are this write-up's and both reviewers' round-2 evaluations):

  | `N_CTX` | rows | `0` meets `Post`, before → now | median `\|attnRef\|` | fp8 floor `≥ \|attnRef\|` |
  |---|---|---|---|---|
  | 128 | non-causal | 99.4% to 100% → 81% to 88% | `0.044` to `0.068` | 36% to 54% |
  | 128 | causal | 87% to 99.1% → 61% to 67% | `0.065` to `0.096` | 25% to 38% |
  | 1024 | non-causal | 100% → 100% | `0.017` to `0.026` | 79% to 95% |
  | 1024 | causal | 96.5% to 99.7% → 93.4% to 95.5% | `0.025` to `0.039` | 59% to 78% |
  | 4096 | non-causal | 100% → 100% | `0.008` to `0.014` | 98% to 100% |
  | 4096 | causal | 96.8% to 99.7% → 97.8% to 99% | `0.013` to `0.021` | 83% to 93% |

  On causal rows the first query has one key, and `Post` rejects `0` on all of its elements
  (`tolO ≈ tolV ≈ 0.31·|attnRef|`); it is `1/N_CTX` of the rows, so small samples over-weight it.

  On the non-causal rows at `N_CTX ≥ 1024`, `|attnRef|` (a mean of values of both signs) is mostly
  below the floor itself (79% to 100% of the elements), and no such bound can reject `0` there.
  Elsewhere `Post` now rejects `0` on 1% to 39% of the elements, where the floor alone would reject
  it on 7% to 75%. The rest of the gap is the output's own fp8 rounding (`9/8` and `|attnRef|/8`),
  `coefS·|attnRef|` (`p`'s rounding against `r`), and the per-block terms that grow with `N_CTX`
  (below).
- **On the benchmark's data** (`randn`, `σ = 1.3`, `N = 1024`): `pE` is now `1.27` to `1.32`
  (median; a head's largest `1.46` to `1.65`) at `D = 64` and `2.13` to `2.37` (largest `3.05` to
  `3.87`) at `D = 128`, where it was `3.4` and `87` to `100`. `tolO` is `0.42` to `0.60` and `0.83`
  to `1.54` (before `13.9` and `1.2·10^4`), against `|attnRef|` of median `0.51` to `0.61` and a
  floor of `0.09`. `0` meets `Post` on 31% to 55% of the elements at `D = 64` and on 62% to 90.5% at
  `D = 128`; before, it met `Post` on all of them. At
  `D = 128` each weight may still move by a factor `pE ≈ 2.3`, so `hw ≈ 1.4` multiplies
  `Σ w·|vEff − attnRef|` and `1/Fp ≈ pE` the whole of `errC`.
- **On rows spread over many blocks**: the chunk facts bound each chunk relative to its largest
  term, the accumulator included (`M_c ≥ |acc|`, with the constant `2^-7`, about twice the window's
  worst chunk). So under the cited fact the
  `p·v` `tl.dot` may change `acc` by `cS8 ≈ 1.62%` per block, and the accumulator is fed back once
  per block. That compounding is `DotE5m2Chunk`'s bound, from the fact's form, not the device's
  behaviour: the device truncates every term and the sum toward zero, so its error grows at most
  linearly in the chunks (library issue [L127](../../docs/library-review/ISSUES.md#l127), round 26).
  Round 26's `DotE5m2ChunkLoss` gives a sharper bound per chunk, which this claim does not cite.
  The compounding is slightly larger than `DotE5m2`'s `1.57%` (the chunk facts' `2^-12` on each new
  sum). It is what remains of the library's part. Through `gA` in `errC` and the
  `2^-17`-per-key terms in `accA` (against `l_i ≥ 1/2`), `tolO` grows with `N_CTX` on the test's data
  (`0.09` to `0.19` from `128` to `4096` at `D = 64`). For `v = 1` everywhere, `0` is accepted from
  `N = 4096` on (`tolO` `1.31`), and at `N = 2^16` the tolerances are about `10^6` and `10^7`
  (before: `9.7·10^5` and `1.1·10^7`).

What `Post` pins at every covered size: no fault, every element written and finite, nothing else
written, every head without `vIn` at `±0` (`tolV = 7.6·10^-6` at `vEff = 0`, and a `float8e5`
within it of `0` is `±0`), and `M` (`tolM` `0.03` to `0.04` at `D = 64`, about `0.1` at `D = 128` on
the test's data). It rejects `o = 0` where `|attnRef|` is above both tolerances: on a causal row's
first query (one key: `tolO ≈ tolV ≈ 0.31·|attnRef|`), on the elements counted above, and on
`v = 1` rows at `N ≤ 2048`.

On the host's inputs the cap `Xh` is met. `σ·scoreLift` is at most `σ·Σ_e|q_e·k_e|` plus
`2^-14·σ·Σ_e(|q_e| + |k_e| + 2^-14)` (from `liftMag x ≤ |x| + 2^-14`; below `0.02` on the host's
data, and `0` when no operand is subnormal). By `attn_fwd_f16`'s
Chernoff estimate for the same distributions, `σ·Σ_e|q_e·k_e|` is below about `23` (test) and `240`
(benchmark) at `D = 128` except with probability below `10^-6`, against `Xh 128 = 256`. At `D = 64`
the cap is `512` and the magnitudes are half. The largest `X` over the rows evaluated were `17`
(test) and `175` (benchmark) at `D = 128`.

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
`float8e5` (casts of finite `float16` values below `57344`) with `σ·scoreLift` below the cap except
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
- `dot` — `Rules.Num.DotE5m2Chunk P`: `tl.dot` of `float8e5` operands into `float32` with
  `maxImpreciseAcc = 2^30`, any `input_precision` (lines 73, 103), chunk by chunk. Each output
  element is its chunks of `32` consecutive products in increasing order, and each step is within
  `2^-7·M_c + 2^-12·|s_c + c_c| + 2^-126` of `s_c + c_c` (`M_c` the larger of `|s_c|` and the
  chunk's largest lifted product), with room in `float32`. Library round 24 added it (L124) and
  withdrew `Num.DotE5m2` (L125), which the claim cited before.
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
- One block (`row_step_plain`, `row_step_masked`, `row_tail`, for every row of a covered head and
  every `X ≤ Xh` and `Z`, `RowSt`). The score: `Num.DotE5m2Chunk` at each score (`qk_lane`).
  `Dot8Chunk.DotF8Chunk.err`, with the chunks' largest lifted products, gives `scoreErr`;
  `DotF8Chunk.liftAcc` gives `relLift8 D·scoreLift + absLift8`; the smaller bounds the score's
  error. Each `p·v` element (`pv_lane`): `DotF8Chunk.err` over its two chunks, `errB2_le` and
  `abs_chunk2_le`. Line 98's lanes (`safe_load_v`, `v_lanes`): inside the descriptor exactly when
  `bh·D < Z·H`, then `v[bh, j, e]`, otherwise the padding `+0`, so each lane is `vEff` and its lifted
  magnitude `vlOf`. The running maximum (`mstep_plain`, `mstep_masked`): `MInv`, each exponent
  within `E = expErr` of `w·s_j − m_ij` and at most `2^-10`, a masked lane's below `−2^19`. `p`
  (`lane_p`): within `psiP·wt + 2^-126` of the exact `wt = 2^(w·s_j − m_ij)`, and at least
  `wt·(1 − 2^-22)/pE − 2^-126`. Its fp8 rounding (`lane_p8`, from `CvtSatF32E5m2`: `p ≤ 1 + 2^-9`
  needs no saturation; `lane_q8`: between `a8·wt − eps8` and `b8·wt + eps8`). `alpha`
  (`alpha_facts`); `l_i` (`block_l`, `lstep`: `LInv`, two-sided); each element of `acc` (`astep`:
  finite, `|acc| ≤ crA t`, `ACr`).
  The first bound (`cen_tail`, module `Cen`): for every `r`, one block of
  `|acc − r·l_i − kap·2^-m_i·Cx| ≤ 2^-m_i·ecR + acR·l_i` (`cstep_real`). The lanes' `Σ q·v − r·Σ p`
  is within `hw·Σ w·|v − r| + eps8·Σ|v − r| + |r|·(Σ|p|/8 + 2^-11)` of `kap·Σ w·(v − r)` (`lane_cen`,
  `block_cen`); the dot's chunks' largest lifted products are at most `b8` times the chunks' largest
  `w·liftMag(v)`, plus `eps8 + 2^-14` times their largest `liftMag(v)` (`cl2_cen`); and
  `l_i ≤ FhT·Σ` from `LInv`.
  The second bound (`mag_tail`, module `Mag`): `mstep_plain` and `mstep_masked` also give that the
  maximum is kept or some lane's exponent is within `2^-11` of `0`, so `l_i ≥ lminB t =
  1 − 2^-9 − t·2^-12 ≥ 1/2` (`p_big`, `lmin_real`). Each lane's `q·v − r·p` (`lane_pq`, `lane_E`),
  each lifted product (`lane_lift`: `0` when `v = 0`, otherwise `iotaV = 1` since a nonzero
  `float8e5` is at least `2^-16`, `f8_abs_ge`), `tl.sum(p)` (`sum64_err`), line 106 (`f32_mulAdd`),
  line 95 (`f32_mul`) and the dot's error (from `pv_lane`) give one block of
  `|acc − r·l_i| ≤ magA·l_i + magD` for every `r` and `K` (`estep_real`).
- The end of a row (`row_out`), after the row's `nblk` blocks (`N/64`, or `2·start_m + 2` when
  causal). `CInv` at `r = attnRef`, where the centred sum is `0` (`centC_final`), with
  `l_i ≥ Fp·Σ` (`l_final`), gives `|acc − attnRef·l_i| ≤ (errC/(Σw·Fp) + accA)·l_i`. Then
  `final_v` (`DivFull`, `CvtSatF32E5m2`, with no absolute part) gives `tolO`. `final_v` with
  `RowSt`'s magnitude bound at `X = Xh` (and `Z = Σ_j|σ·scoreErr_j|`) gives `tolV`, with
  `|attnRef| ≤ 57344` by `ref_le` and `K ≥ |vEff_j − attnRef|` over the keys (`veOf_key`).
  `final_M` (`Log2F32`) gives `tolM`. The row's sums are the specification's: `W = Σ_j 2^(w·s_j)`
  is `wsum` (`ew_weight`: `2^(σ/ln 2·s) = exp(σ·s)`), `Sx/W = attnRef` (`veOf_key`), `acR = accA` and
  `ecR = errC` (`acR_accA`, `ecR_errC`, from `Cx_centC`, `bC_blkC`, `bA_blkA`, `bLW_blkLW`,
  `bVL_blkVL`, `psum_partW`, `psum_below`, `blk_sum`), and the constants equal `Model`'s
  (`pE_r`, `dmax_r`, `FhT_r`, `Fhi_r`, `Flo_r`, …). `model_factors` checks the numeric side
  conditions (`Fp ≥ 2^-15`, `Fhi + aL·pE ≤ 2^15`, …) at `N ≤ 2^16`.
- `hpost`: element `(bh, n, e)` of `o` and `(bh, n)` of `M` lie in program `bh·g0 + n/128`'s tile
  (`prog_of`), whose `Qp` gives them; a byte outside `outBytes` is in no write footprint.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref`, and `Platform.sm90ref.dotE5m2Chunk`,
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
- **Covered heads** (`HeadOk`): finite inputs and `σ·scoreLift ≤ Xh D`. The cap keeps each
  exponent's error below `13`, the causal mask's `−10^6` below every attended exponent, and the
  numeric side conditions; for another head the claim says only that `o` and `M` hold some value.
  The benchmark's inputs at `D = 128` are near the cap (about `240` against `256`).
- **The reference is `vEff`'s attention**, not `v`'s, for the heads the kernel reads as zeros
  (Specification). This is the kernel's behaviour, not a narrowing of the inputs.
- **The tolerances' form** (Specification): `tolO` is data-dependent (`errC`, `accA`: `blkC`,
  `blkLW`, `centC` of the row's values and weights) and centred on `attnRef`. `tolV` is a magnitude
  bound in `K`, which any average of the row's own values meets: alone it checks the key set and
  the range of the values, not the weights. On the test's data `0` meets `Post` on 61% to 100% of
  the elements, and on the benchmark's data on 31% to 90.5% (Specification, "Where `Post` does
  not tell `o` from `0`"). On the test's non-causal rows at `N_CTX ≥ 1024`, that is mostly the fp8
  rounding of `p` itself (79% to 100% of the elements). What remains of the library's part is the
  `p·v` accumulator's compounding over the blocks, `(1 + cS8)^nblk` (about `10^6` at `N = 2^16` for
  `v = 1`). It is the cited fact's (`DotE5m2Chunk`'s) bound, not the device's behaviour (library
  issue [L127](../../docs/library-review/ISSUES.md#l127)).
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
