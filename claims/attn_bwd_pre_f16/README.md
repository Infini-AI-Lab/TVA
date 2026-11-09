# `attn_bwd_pre_f16`: the prepared claim

Kernel: `source/06-fused-attention.py`, `_attn_bwd_preprocess` (lines 248–262; the kernel lines
are 20–21 and 248–262, the host lines as [`claim.json`](claim.json) gives them). Claim: [`Claim.lean`](Claim.lean), namespace `Claims.AttnBwdPre`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py). The launches to cover: [`SETUP.md`](SETUP.md).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.AttnBwdPre.main : Statement` (`Statement`:
`Claim.lean` line 279). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 109), and every initial memory `init` that `Pre` accepts (line 171), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 213).

**What `Post` asks** (line 213), two parts:
1. for every `bh < Z·H` and `m < N_CTX`, `OutSpec d init fin (bh·N_CTX + m)` (line 200): element
   `r = bh·N_CTX + m` of `Delta` reads in `fin` as a `float32` `c`, for every such `r`. Only when
   row `r` of `O` and of `DO` (elements `r·HEAD_DIM + k`, `k < HEAD_DIM`) read in `init` as finite
   `float16` values `vo k`, `vdo k` is more asked: `c` is finite and
   `|c.toRat − dotSum vo vdo HEAD_DIM| ≤ tol HEAD_DIM (dotAbs vo vdo HEAD_DIM)`. `dotSum` and
   `dotAbs` (lines 177, 180) are the exact sums, in the rationals, of `vo k · vdo k` and of their
   magnitudes; `tol` is defined at line 191. With a non-finite input in the row only "holds a
   `float32` value" is asked;
2. the frame: every byte outside `Delta.elems (outSet d)` (line 100) is unchanged.

**What you are given** (`Assumptions`, line 221), two fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `sum_depth` (`Rules.Num.SumDepth P`): every `tl.sum` tree over `n` positions has depth at most
  `⌈log₂ n⌉`; it is used for the `tl.sum1c` of line 70 (source line 260), with
  `SumDepth.depth_le` and `clog_two_pow`.

**The kernel, as the model runs it** (`_attn_bwd_preprocess` line 42, `model` line 150, `launch`
line 161): a 2-D grid `(N_CTX / BLOCK_M, Z·H, 1)`. Program `(pm, bh)` loads, with no mask, the
`[BLOCK_M, HEAD_DIM]` blocks of `O` and `DO` at rows `bh·N_CTX + pm·BLOCK_M + [0, BLOCK_M)`
(`int32` offsets), widens `do` with `tl.cvt` and `o` with `tl.convert .f16 .f32`, multiplies
(`tl.mul`), sums each row with `tl.sum1c`, and stores the `BLOCK_M` sums to `Delta` at
`bh·N_CTX + pm·BLOCK_M + [0, BLOCK_M)`. No loop, no atomics; programs write disjoint rows: the route
is `TVA.meets_independent` (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`launch` sets `debug := false`; `Launch.Valid`
  from `bh_lt`, `grid_lt`, `ctx_lt`, and the block shapes from `block_pow2`, `head_pow2`,
  `block_le`);
- each program is `Safe`: its loads and its store are in bounds (`O_len`, `DO_len`, `Delta_len`,
  `ctx_dvd`), and the `int32` products `off_hz·HEAD_DIM·N_CTX`, `off_m·HEAD_DIM`, `off_hz·N_CTX`
  do not wrap (`row_fit`, `head_fit`; `TVA.Lemmas.Int`); `safe_load`, `safe_cvt`, `safe_convert`,
  `safe_mul`, `safe_sum1c`, `safe_store`;
- the programs' write rows are disjoint (from the grid coordinates and `ctx_dvd`), and no program
  writes what any program reads (`Delta_O`, `Delta_DO`);
- every `r < Z·H·N_CTX` is written by one program (`ctx_dvd`: the grid covers every position);
- the value part of part 1: the widenings are exact (`FloatV.toRat_cvt_of_wider`), the products are
  one `float32` rounding each, and the fused sum is bounded by `sumc_err_depth` at depth
  `⌈log₂ HEAD_DIM⌉`, composed as in `sum_bcast_mul_tree_err` (README `## Tolerances`);
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `canon_ok` and the
`setup_*_ok` theorems; `tol_le_bcast` (line 236), `tol_mono` (line 251), `tol_le` (line 267) and
`elems_disjoint_of_lt` (line 288) can be reused.

## Specification

For every `(batch, head)` pair `bh < Z·H` and position `m < N_CTX`, with `r = bh·N_CTX + m`
(element `(b, h, m)` of the contiguous `delta`, `bh = b·H + h`), `Post` (`OutSpec`) says:

* element `r` of `Delta` holds some `float32` value `c` in the final memory;
* whenever row `r` of `O` and of `DO` (elements `r·HEAD_DIM + k`, `k < HEAD_DIM`) hold finite
  `float16` values `o_k`, `do_k` in the initial memory, `c` is finite and
  `|c − Σ_k o_k·do_k| ≤ tol(HEAD_DIM, Σ_k |o_k·do_k|)`, the sums exact (`dotSum`, `dotAbs`).

And every byte outside the elements `r < Z·H·N_CTX` of `Delta` is unchanged (`O`, `DO`, the rest
of `Delta`'s storage, and every other byte the launch is given).

`tol(D, B) = errBound(3u, 4η, ⌈log₂ D⌉, D − 1, (1 + u)·B + D·η) + (u·B + D·η)` with `u = 2^-24`,
`η = 2^-150`, `Tree.errBound δ η d N A = ((1 + δ)^d − 1)·A + η·N·(1 + δ)^d`: the library's
`bcastTreeBound D B` with the reduction's depth `⌈log₂ D⌉` (`Rules.Num.SumDepth`) in place of
`D − 1` (`tol_le_bcast`: `tol ≤ bcastTreeBound`). To first order `(3·log₂ D + 1)·2^-24·B`:

| `HEAD_DIM` | `tol / B` |
|---|---|
| 16 | `7.75·10^-7` |
| 64 | `1.13·10^-6` |
| 128 | `1.31·10^-6` |
| 256 | `1.49·10^-6` |

`tol_le` proves `tol(D, B) ≤ 2^-19·B + 2^-138` for every covered `D ≤ 256`. The bound is relative
to `Σ|o·do|`, not to `|Σ o·do|` (a sum with cancellation can lose every digit in `float32`). It
rejects a `float16` result or accumulation (`2^-11` relative), a missing or doubled column, or the
wrong row of `do` (`check_gpu.py`, negatives: on `randn` inputs at `HEAD_DIM` 64 and 128, `Post`
fails on `90–93%` of the rows of a result rounded to `float16`, and on `99.9–100%` of the rows of
the others).

## Conditions

The model is `_attn_bwd_preprocess` line by line (lines 254–262): the `int32` products and pointer
additions left to right with the constexprs built in `int32`, no mask and no `other` (every lane is
in bounds on a covered instance), `.to(tl.float32)` as `tl.cvt`, the `float16 × float32` product
computed in `float32` (`tl.convert`, then `tl.mul`), and `tl.sum(o * do, axis=1)` as `tl.sum1c`
(a sum of products, whose additions may be contracted).

`Inst`: `Z`, `H`, `N_CTX` (the three integer arguments), the constexprs `HEAD_DIM`, `BLOCK_M`, and
the buffers `O`, `DO` (`float16`) and `Delta` (`float32`), laid out as the contiguous tensors the
host passes (`(b·H + h)·N_CTX + m` rows of `HEAD_DIM`). `Inst.ok`:

* `BLOCK_M` and `HEAD_DIM` powers of two, `HEAD_DIM ≤ 256`, `BLOCK_M·HEAD_DIM ≤ 2^20` (Triton's
  blocks; the host's `128` and `16 … 256`);
* `BLOCK_M ∣ N_CTX` (the host asserts `N_CTX % 128 == 0`, line 599); `N_CTX < 2^31` (an `int32`);
* the grid `(N_CTX // BLOCK_M, Z·H)` within CUDA's limits (`Z·H < 2^16`, fewer than `2^31`
  programs);
* `(N_CTX − 1)·HEAD_DIM < 2^31` and `(Z·H − 1)·HEAD_DIM·N_CTX < 2^31`: the kernel's `int32`
  products do not wrap (a limit of the kernel, see Exclusions);
* each storage holds its tensor; the written elements of `Delta` share no byte with the read
  elements of `O` or `DO` (a fresh `delta`).

`Pre`: the three buffers are valid. Any `float16` values, NaN and infinities included.

The launch: the grid `(N_CTX // BLOCK_M, Z·H, 1)`, the source's block shapes, every option at
Triton's default (`fusion = true`, `debug = false`, `num_ctas = 1`, no `launch_pdl`): the launch of
lines 600–607 gives no option, the decorator is a bare `@triton.jit`, no `@triton.autotune` or
`@triton.heuristics` on this kernel, no `warmup`/`triton.compile`, no descriptors (the forward's
allocator and host descriptors go to `_attn_fwd`): within part (A) of `Compiler.Faithful`'s scope. The claim relies on part (B)
(the environment) as the rule states it.

## Assumptions

Two `[rule]` fields: `faithful` (`Compiler.Faithful`) and `sum_depth` (`Num.SumDepth`: the
`tl.sum` of line 260 along a tree of depth at most `⌈log₂ HEAD_DIM⌉`; the kernel has no loop, so
not the loop form the library excludes). Every other operation of the model is
exact or has its meaning in the library without a platform fact (exact `float16 → float32`
widenings, IEEE `float32` products, the sum's `AddRel` additions). `consistent` uses
`Platform.sm90ref`.

## Coverage

`canon Z H N D` is the host's launch (`BLOCK_M = 128`, contiguous `o`, `do`, fresh `delta`);
`canon_ok` proves `Inst.ok` from numeric facts `norm_num` checks. `SETUP.md`'s launches:
`setup_test_ok` (all 24 shapes of the test, both `causal` modes being the same launch of this
kernel), `setup_bench_ok` (all 10 shapes of the benchmark), `setup_test_large_ok` (`4·48` heads of
`4096 × 128`), `setup_bench_large_ok` (`4·32` heads of `16384 × 128`), and `setup_headdims_ok`
(`HEAD_DIM` 16, 32 and 256 at those two shapes).

## Tolerances

From the library (`docs/rules/num.md`, "Exact operations need no rule"): the widenings are exact
(`FloatV.toRat_cvt_of_wider`), each product is one IEEE `float32` rounding (`u·|a·b| + η`; the
product of two `float16` values is in fact exact), and `tl.sum1c` is a sum along a tree covering
the `D` products with `AddRel fusion` nodes (`3u`, `4η` per addition, `FloatV.AddRel.err`) of depth
at most `tl.sumBound P D`, which `SumDepth` with `SumDepth.depth_le` bounds by `⌈log₂ D⌉`
(`log₂ D`: `clog_two_pow`). `sumc_err_depth` (`TVA.Lemmas.SumFused`) bounds that tree's sum of the
rounded products by `errBound(3u, 4η, ⌈log₂ D⌉, D − 1, Σ|p_k|)`, with `Σ|p_k| ≤ (1 + u)·B + D·η`
(`Tree.errBound_mono`), and the products' own roundings add `u·B + D·η`: the composition is that of
`sum_bcast_mul_tree_err`, with `sumc_err_depth` for `sumc_err_covers`. The room hypothesis holds on
every input: a finite `float16` is at most `65504`, so `B ≤ 256·2^32 = 2^40`. So the bound is
provable on every row with finite inputs. Meaningful: `tol/B ≤ 1.5·10^-6`, about `25` `float32`
roundings; on the host's inputs the kernel's error reaches `8%` of it (`14%` at `HEAD_DIM = 16`).

## Exclusions

* **`int32` offsets (a limit of the kernel).** `off_hz * HEAD_DIM * N_CTX` (lines 258–259) and
  `off_m * HEAD_DIM` are `int32` products; once `(Z·H − 1)·HEAD_DIM·N_CTX ≥ 2^31` they wrap and the
  kernel reads the wrong rows. The first host launch that hits it has `Z·H·N_CTX·HEAD_DIM` a little
  above `2^31` elements per tensor (for example `Z·H = 1025` heads of `16384 × 128`, a 4 GiB `o`).
  The file's launches are far below (`191·2^19` and `127·2^21`); real use of the tutorial does not
  reach it.
* **`N_CTX` not a multiple of `BLOCK_M`** (the last positions get no program, the kernel has no
  mask): the host asserts it away (line 599).
* **`HEAD_DIM > 256`**, other dtypes: the forward rejects them (line 512); the backward in
  `float8e5` is not run (the test skips it, line 643; the benchmark's `triton-fp8` provider runs the
  `float16` backward).
* **The grid** beyond CUDA's limits (`Z·H ≥ 2^16`, `2^31` programs): no launch.
* **`delta` overlapping `o` or `do`**: the host allocates a fresh `delta`.
* **Non-finite inputs** stay covered, with a weaker `Post`: a row with a NaN or `∞` in `o` or `do`
  has no exact finite sum, and `Post` says only that a `float32` value is stored there.

## Library requests

None.

## Checks

* **Builds**: `lake env lean ../claims/attn_bwd_pre_f16/Claim.lean`: no errors or warnings, no
  `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (the test's `Z = 1`, `H = 2`,
  `N_CTX = 128`, `HEAD_DIM = 64`, a memory of zeros).
* **Covers the setup**: `canon_ok`, `setup_test_ok`, `setup_bench_ok`, `setup_test_large_ok`,
  `setup_bench_large_ok`, `setup_headdims_ok`.
* **Model map**: every code line of the kernel lines covered, every model text found in
  `Claim.lean` (the gate's `map_problems`, run on this folder).
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 48 test launches through the host's
  `attention(...).backward(dout)` (the kernel's tensors recorded at its launch), the benchmark's 10
  sizes, `HEAD_DIM` 16, 32, 256, `Z·H = 1`, `N_CTX = 0`, values near the `float16` limit (`2^13`
  scale), exact cancellation, subnormal-sized values, and rows with `∞`/NaN. Worst error /
  tolerance `0.138` at `HEAD_DIM = 16` (margin 7×); `0.079` on the test's launches, `0.084` on
  the benchmark's sizes; every row with finite inputs finite; `o`, `do` unchanged bit for bit and
  the bytes past `delta` keep their sentinel. Wrong outputs (a `float16` result, the last column
  dropped or doubled, the next row of `do`) fail `Post`.
