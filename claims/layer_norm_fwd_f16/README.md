# `layer_norm_fwd_f16`: the prepared claim

Kernel: `source/05-layer-norm.py` (Triton 3.8.0's tutorial 05, unchanged), `_layer_norm_fwd_fused`
(lines 48–95; the kernel lines are 34–35 and 48–95), launched by `LayerNorm.forward` (lines
235–260) with one program per row, the grid `(M,)`, on `float16` `x`, `weight`, `bias` and `y`,
`float32` `mean` and `rstd`, `BLOCK_SIZE = next_power_of_2(N)` (`N ≤ 32768`). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.LayerNormFwdF16`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json).
Launches: [`SETUP.md`](SETUP.md). GPU check: [`check_gpu.py`](check_gpu.py). The host lines, the
two backward kernels (unused here) and their headers are classed in [`claim.json`](claim.json) and
the model map.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.LayerNormFwdF16.main : Statement` (`Statement`:
`Claim.lean` line 413). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 173), and every initial memory `init` that `Pre` accepts (line 362), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 383).

**What `Post` asks** (line 383): for every row `t < M`, `RowDone d init fin t` (line 373), and the
frame. The reference is the exact real LayerNorm of row `t` of `x`, read from `init` as reals
(`meanRef` line 272, `varRef` 280, `rstdRef` 287, `zRef` 291, `yRef` 295, with `weight[j]` and
`bias[j]`); element `(t, j)` of `x` and `y` is `idx d t j = t·N + j` (line 153). `RowDone` has three
parts:
1. `mean[t]` reads in `fin` as a finite `float32` `c` with
   `|c.toReal - meanRef d init t| ≤ meanTol d init t` (`meanTol`, line 303:
   `(D + 4)·2^-23·A + 2^-140`, `A = absMean`);
2. `rstd[t]` reads as a finite `float32` `c` with `|c.toReal - rstdRef d init t| ≤ rstdTol d init t`
   (`rstdTol`, line 326, with `cond` 308): `(D·2^-21 + 2^-140/eps + cond/4)·rstdRef + 2^-140` when
   `cond ≤ 1/2`, and `2/sqrt(eps)` otherwise;
3. for every `j < N`: `y[idx t j]` reads as a finite `float16` `c` with
   `|c.toReal - yRef d init t j| ≤ yTol d init t j` (`yTol`, line 350:
   `2^-9·yMag + 2·|w|·zTol + 2^-24`, with `yMag = |w|·|zRef| + |b|` line 345 and `zTol` line 335,
   built from `gam` 313 and `mTerm` 320, which has the same two cases).

These hold on every row; the two cases are inside the tolerances, not guards on what is asked.
`depth` (line 300) is `Nat.clog 2 BLOCK`. The frame: every byte outside `outBytes d` (line 169:
the `M·N` elements of `y`, the `M` elements of `mean` and `rstd`) is unchanged.

**What you are given** (`Assumptions`, line 390):
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `sqrt_total` (`Num.MathTotal P .sqrt .binary32`): `tl.sqrt` (source line 81, `Claim.lean`
  line 93) has a result;
- `sqrt_acc` (`Num.SqrtF32`): on `0` and inputs `≥ 2^-126`, `tl.sqrt` is within `2^-22·√x + 2^-149`;
- `div_total` (`Num.DivTotal P .binary32`): `/` (source lines 72, 80, 81; `Claim.lean` lines 69,
  88, 92) has a result;
- `div_full` (`Num.DivFull`): `/` is within `2^-22·|q| + 2^-148` of the quotient `q`, for
  `|q| ≤ 2^126`;
- `sum_depth` (`Num.SumDepth`): the trees of the two `tl.sum`s (source lines 72, 80; `Claim.lean`
  lines 69, 88) over their `BLOCK_SIZE` positions have depth at most `Nat.clog 2 BLOCK`
  (`SumDepth.depth_le`).

**The kernel, as the model runs it** (`layer_norm_fwd_fused` line 42, `model` line 226, `launch`
line 233): a 1-D grid of `M` programs, one per row. Program `row` advances `X` and `Y` by the `int32`
product `row * stride` (`stride = N`), then runs three `tl.forRange` loops over `[0, N)` with step
`BLOCK_SIZE`; since `N ≤ BLOCK_SIZE` each makes one trip (`off = 0`, `cols = [0, BLOCK_SIZE)`, mask
`cols < N`). Loop 1 loads `x` (`other = 0`) and adds it to a zero block; `tl.sum`, then `/` by
`tl.itof N`, gives `mean`. Loop 2 loads `x` again, forms `tl.subc x mean` (zero outside the mask),
and `tl.mul_add x x _var`; `tl.sum`, `/` by `N`, `tl.addc` with `eps`, `tl.sqrt` and `1 / ·` give
`rstd`. `tl.store1` writes `mean[row]` and `rstd[row]`. Loop 3 loads `w`, `b` (no `other`) and `x`,
forms `tl.subc`, `tl.mul` by `rstd`, `tl.mul_add` with `w`, `b`, converts to `float16` and stores
the row of `y` under the mask. Programs write disjoint rows, all read the same `weight` and `bias`,
and use no atomics: the route is `TVA.meets_independent`. Each loop takes `safe_forRange` with an
invariant on the carried block; loops 1 and 2 write nothing (`fun _ => ∅`, `iterW_empty`), loop 3's
one trip writes row `row` of `y` (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: `M < 2^31` from `rows_lt` and
  `n_ge`; the block shape `[BLOCK]` from `block_pow2`, `block_lt`, `n_le`);
- each program is `Safe`: `row * stride` does not wrap (`rows_lt`), nor do `cols` (`BLOCK ≤ 32768`);
  the loop runs one trip (`n_le_block`); every enabled lane of `x` and `y` is below `M·N` (`x_len`,
  `y_len`), of `w`, `b` below `N` (`w_len`, `b_len`), and `mean[row]`, `rstd[row]` are in bounds
  (`mean_len`, `rstd_len`); every float step has a result (`sqrt_total`, `div_total`);
- the write footprints of distinct programs (row `row` of `y`, `mean[row]`, `rstd[row]`) are
  disjoint, and no program's reads meet any program's writes (`out_in`, `y_mean`, `y_rstd`,
  `mean_rstd`); this also gives `hRW`;
- parts 1–3: the rounding of each step, the sums by `sum_err_depth` with `SumDepth.depth_le`, `N`
  exact through `tl.ItofRel` (`safe_itof`), `safe_mul_add`, `div_full`, `sqrt_acc` (its input is at
  least `2^-126` from `eps_ge`), the conversion of the host's `eps` (`FloatV.hostF32`), and `Pre`'s
  `YRoom` (line 355), which keeps `y`'s `float32` value below `2^15·(1 + 3·2^-24)` so the `float16`
  conversion does not overflow; README `## Tolerances` gives the derivation of the fine bounds,
  steps 1–5, then relax them to `rstdTol` and `yTol` as its step 6 shows (`meanTol` is the fine
  bound itself);
- the frame: the bytes no program writes are `init`'s.

`n_dim` and `m_ctx` (real model shapes) are not needed by the derivation; they give `N ≥ 128`
(so `D ≥ 7`) and at most `2^22` programs.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mkInst_ok`,
`eps1e5_facts`, `setup_test_ok` and `setup_bench_ok`; `bytes_disj` (line 455) and `elems_disj`
(line 463) can be reused.

## Specification

For each row `t < M`, with `x j = x[t, j]` (the `float16` input as a real, exactly), the reference
is the exact real LayerNorm: `meanRef = Σ x / N`, `varRef = Σ (x − meanRef)² / N`,
`rstdRef = 1/sqrt(varRef + eps)` (the host's double `eps`), `zRef j = (x j − meanRef)·rstdRef`,
`yRef j = zRef j·w j + b j`. `Post` says:

- `mean[t]` is a finite `float32` within `meanTol` of `meanRef`;
- `rstd[t]` is a finite `float32` within `rstdTol` of `rstdRef`;
- `y[t, j]` is a finite `float16` within `yTol j` of `yRef j`;
- every byte outside the `M·N` elements of `y` and the `M` elements of `mean` and `rstd` is
  unchanged (frame).

The reference reads the host's tensors at the host's layout (`t·N + j`), never through the kernel's
indexing. With `D = log₂ BLOCK_SIZE` (`depth`: the depth of the reductions' trees over their
`BLOCK_SIZE` leaves, `Rules.Num.SumDepth`), `A = Σ|x|/N` and the row's condition number
`cond = meanTol·rstdRef`:

| Output | Tolerance |
|---|---|
| `mean` | `meanTol = (D + 4)·2^-23·A + 2^-140` |
| `rstd` | `cond ≤ 1/2`: `rstdTol = (D·2^-21 + 2^-140/eps + cond/4)·rstdRef + 2^-140`; `cond > 1/2`: `2/sqrt(eps)` |
| `y` | `yTol = 2^-9·yMag + 2·|w|·zTol + 2^-24`, `yMag = |w|·|zRef| + |b|`, with `zTol = (1 + 2·gam)·(|zRef|·(gam + mTerm) + cond) + 2^-140·(1 + rstdRef)` when `cond ≤ 1/2`, and `zTol = |zRef| + (1 + 2·gam)·(|x − meanRef| + meanTol)/sqrt(eps) + 2^-140·(1 + 1/sqrt(eps))` when `cond > 1/2`; `gam = (D + 8)·(2^-23 + 2^-145/eps)`, `mTerm = cond/8 + cond²/4` |

In words: the mean is within `(D + 4)·2^-23` of the row's mean magnitude; `rstd` is within
`D·2^-21` relatively, plus `cond/4` for the mean's error (and a term in `1/eps` that is negligible
at real `eps`); `y` is within `2^-9` of its terms' magnitude `yMag`, plus twice the standardised
value's error `zTol` (about `cond` on a well-conditioned row) scaled by `|w|`. `zTol` is also what
`Pre`'s `YRoom` uses.

They hold on every row, with no condition on the data's spread. On the file's rows (`x ≈ −2.3 ±
0.5`, `A ≈ 2.3`, `rstdRef ≈ 2`) at `BLOCK_SIZE = 8192` (`D = 13`, the test) they are `2.0·10^-6`
relative for the mean, `8.5·10^-6` relative for `rstd` (`cond ≈ 9·10^-6`), and for `y`
`2^-9·yMag` plus about `2·10^-5·|w|`: `tol/|y|` has median `2.0·10^-3` (larger only where `yRef`
is near `0`); at `N = 15872` (`D = 14`) `2.1·10^-6`, `9.1·10^-6` and the same for `y`. The `y`
tolerance is about four times the output's `float16` rounding. A kernel that drops one term
of the mean's sum, one term of the variance's sum, or divides the variance by `N − 1` fails `Post`
on these rows (`check_gpu.py`, negatives: error / tolerance `106`, `87`, `7.2` at `N = 8192`;
`49`, `34`, `3.5` at `N = 15872`). The tolerances grow with `cond`, which is at most `meanTol/sqrt(eps)`:
a row whose values share a large offset and differ by little loses the variance to the computed
mean's rounding (the two-pass algorithm in `float32`). Up to `cond = 1/2` the tolerances stay
relative (`rstd` within `D·2^-21 + cond/4` of `rstdRef`, relatively, at real `eps`); beyond it
they say only what `eps` bounds: `rstd` and `rstdRef` are both at most about `1/sqrt(eps)`, and the standardised value
is at most about `(|x − meanRef| + meanTol)/sqrt(eps)`. Such a row has its values within about
`2·meanTol` of one another (a constant row of `1000` at `eps = 1e-5` is one).

## Conditions

`Inst`: `M`, `N`, `BLOCK` (`BLOCK_SIZE`), `eps` (the Python double), the storages of `x`, `weight`,
`bias`, `y` (`Buf .f16`), `mean`, `rstd` (`Buf .f32`). `Inst.ok`:

- `2 ≤ N ≤ 32768` (the host raises for `N > 32768`; `N = 1`: Exclusions); `BLOCK` a power of two
  with `N ≤ BLOCK < 2N` (`next_power_of_2(N)`, line 246);
- `1 ≤ M`, `(M − 1)·N < 2^31` (line 63's `int32` product `row * stride`: Exclusions);
- `N % 128 = 0`, `N ≤ 2^17` (`n_dim`: `N` is a model dimension, the hidden size) and `M ≤ 2^22`
  (`m_ctx`: `M` counts the batch's tokens, one row each): real model shapes only (Exclusions);
- `eps` finite, `2^-125 ≤ eps ≤ 2^126` (Exclusions);
- every tensor holds what the launch touches (`M·N`, `N`, `M` elements); the outputs share no byte
  with the inputs or with each other (fresh `torch.empty` tensors).

`Pre`: the six buffers valid (aligned, present); every element of `x`, `weight`, `bias` the launch
reads finite; every row's output has room in `float16` (`YRoom`:
`|w j|·(|zRef j| + zTol j) + |b j| ≤ 2^15` for every column; on the file's data `|y| < 4`).

Scope of `Compiler.Faithful`, part (A): the launch passes `num_warps` and `num_ctas=1` besides the
kernel's arguments (no other option, no autotuning or heuristics), the decorator has no `debug`
(`Launch.debug = false`), no descriptors, no `CompiledKernel`, no host `TensorDescriptor`. Part (B), the environment, is assumed: the claim relies on it.

## Assumptions

- `[rule] Compiler.Faithful`, as `docs/rules/compiler-faithful.md` states it.
- `[rule] Num.MathTotal P .sqrt .binary32` and `Num.SqrtF32 P`: `tl.sqrt` on `float32` (line 81).
- `[rule] Num.DivTotal P .binary32` and `Num.DivFull P`: `/` on `float32` (lines 72, 80, 81).
- `[rule] Num.SumDepth P`: the trees of the two float `tl.sum`s (lines 72, 80) over their
  `BLOCK_SIZE` positions have depth at most `⌈log₂ BLOCK_SIZE⌉ = D`. Neither is in the loop form the
  library excludes: each reduces a loop's carried block once, after its loop, and no
  loop updates a carried value with a reduction.

`consistent` witnesses them with `Platform.sm90ref`.

## The model

`model-map.json` maps every code line. `_mean += a` is the IEEE `+`; `_var += x * x` is
`tl.mul_add`; both reductions are `tl.sum` (their inputs are not products); `/ N` converts the
runtime `int32` `N` to `float32` (`tl.itof`, exact) and is the platform's division. A `float32`
quotient is a product for the fusion rule (SEMANTICS.md, "Float `+` and `-`"), so the additions it
feeds are the fusable forms: `x - mean` (lines 78 and 92) is `tl.subc` and `var + eps` (line 81) is
`tl.addc`, `AddRel` on the quotient `tl.div` picks; `x_hat * w + b` is `tl.mul_add` after the exact widening of `w`
and `b`; the store converts `y` to `float16` (one rounding to nearest even). `other=0.` is the
`float32` `+0` converted to `float16`. `eps` is the host's double, `FloatV.hostF32` in the model;
`stride` and `N` are runtime `int32` scalars (`stride = N`, contiguous rows). Since `N ≤ BLOCK`,
each of the three loops makes one trip on every covered launch; the model keeps the loops.

## Coverage

`mkInst M N BLOCK` is the host's launch (contiguous `x`, fresh `y`, `mean`, `rstd`, `eps = 1e-5`,
the six tensors at `k·2^36`). `mkInst_ok` proves `Inst.ok` for every `2 ≤ N ≤ 32768` that is a
multiple of `128`, `BLOCK` its next power of two, `1 ≤ M ≤ 2^22` with `M·N ≤ 2^30`. The launches
of [`SETUP.md`](SETUP.md):

| Launch | Instance |
|---|---|
| the test, `M = 1151`, `N = 8192` | `setup_test` (`setup_test_ok`) |
| the benchmark, `M = 4096`, `N = 512·i`, `2 ≤ i ≤ 31` | `setup_bench i` (`setup_bench_ok i`, `BLOCK = benchBlock i`) |

`satisfiable` uses `setup_test` in a memory of zeros (every row constant `0`, `y = 0`).

## Tolerances

Notation: `u = 2^-24`, `η = 2^-150` (`float32`; `FloatV.add_err`, `sub_err`, `mul_err`, `cvt_err`:
`u·|q| + η`), `u16 = 2^-11`, `η16 = 2^-25` (`float16`), `n = BLOCK`, `D = log₂ n`,
`e_m = mean' − meanRef` (the kernel's mean `mean'`). The facts used:
`SumDepth` with `SumDepth.depth_le` (each `tl.sum`'s tree over its `n` leaves has depth at most
`Nat.clog 2 n = D`, `clog_two_pow`) and `sum_err_depth` (`tl.sum` along such a tree:
`((1+u)^D − 1)·Σ|leaf| + η(n−1)(1+u)^D`), the `tl.mul_add`
relation (one rounding of `a·b + c`, or a rounded product then a rounded sum), `DivFull`
(`2^-22·|q| + 2^-148`, `|q| ≤ 2^126`), `SqrtF32` (`2^-22`, `2^-149`, on `0` and `x ≥ 2^-126`),
`ItofRel` (`N` exactly), and `AddRel` for the fused `tl.subc`, `tl.addc` of `x` and a quotient `q`:
the IEEE result, or the rounding of an exact sum in which one operand is replaced by an exact value
that rounds to it, `p` for `q` (`|p − q| ≤ u·|q| + η`, `RoundsTo.err`) or `ξ` for `x`
(`|ξ − x| ≤ u·|x| + η`), chosen lane by lane. The masked lanes `j ≥ N` are `+0` (`other=0.`, and `tl.where` puts `0.` in
the deviations); each loop makes one trip, so `_mean = +0 + a` and `_var = mul_add(x, x, +0)` are
`a` and `fl(x²)` exactly.

1. **Mean.** `|S' − S| ≤ E0·N·A + η(n−1)(1+u)^D` with `E0 = (1+u)^D − 1`, then the division:
   `|mean' − meanRef| ≤ (E0 + 4u(1 + E0))·A + 2^-148 + 2^-148`. Composed exactly, the relative part
   is at most `0.501` of `(D + 4)·2^-23` at every `1 ≤ D ≤ 15` (`E0 ≤ D·u·(1 + 2^-19)`); the
   absolute part is below `2^-140` (`(n − 1)/N < 2`).
2. **Variance.** Lane `j` of line 78 rounds `x j − mean' − δ_j` with `δ_j = 0` (the IEEE `-`),
   `δ_j = p_j − mean'` (the mean replaced: `|δ_j| ≤ u·|mean'| + η`) or `δ_j = x j − ξ_j` (`x`
   replaced: `|δ_j| ≤ u·|x j| + η`); in every branch `δ_j = α_j·(x j − meanRef) + β_j` with
   `|α_j| ≤ u` and `|β_j| ≤ δ = u·(|meanRef| + |e_m|) + η` (`u·|x j| ≤ u·|x j − meanRef| + u·|meanRef|`).
   So its exact deviation is `d_j = (1 − α_j)(x j − meanRef) − (e_m + β_j)`; with
   `Σ_j (x j − meanRef) = 0`, `s = sqrt(varRef)` and Cauchy–Schwarz on the cross terms,
   `S/N = Σ_j d_j²/N` is `(1 ± u)²·S₀/N` with `S₀/N` in `[s² − 2δ'·s, s² + 2δ'·s + E²]`,
   `δ' = ((1 + u)·δ + u·|e_m|)/(1 − u)²`, `E = (|e_m| + δ)/(1 − u)`. Every factor
   after that is relative on nonnegative terms: the replacement's `(1 ± u)²` above, the subtraction's rounding `(1 ± u)` squared, the
   square `(1 ± u)`, the sum `(1 ± E1)` with `E1 = (1+u)^D − 1`, the division `(1 ± 4u)`, the fused
   addition's replaced operand (`var`'s `p` or `eps`'s `ξ`) `(1 ± u)`, the conversion of `eps` to `float32` `(1 ± u)`, the addition's
   rounding `(1 ± u)`, and the subtraction's absolute error `(1 ± u/64)`. So the input of `tl.sqrt`
   is `(S₀/N + eps)·(1 + Θ)` with `1 + Θ` between `(1−u)^7(1−u/64)(1−E1)(1−4u)` and
   `(1+u)^7(1+u/64)(1+E1)(1+4u)`, plus absolute terms below `11·2^-150 ≤ (D + 8)·2^-149` (after the
   division by `N`: the `η` of the `N` squares and the sum's `η(n − 1)(1+u)^D`, at most `3.01η`; the
   division's `2^-148`; the fused `p`'s, the conversion's and the addition's `η`; the subtraction's
   `η` enters only through `2η·Σ|dev| ≤ (u/64)·Σdev² + 64·n·η²/u`, a relative `u/64` on the sum of
   squares, kept in `Θ`, and a negligible absolute part). Relative to the input, at least
   `eps·(1 − 3u)`, they are below `2^-146.5/eps`. Room: `|x| ≤ 65504`, each squared deviation below
   `2^35`, the sum below `2^50`; the input of `tl.sqrt` is at least `eps(1 − 10u) ≥ 2^-126`
   (`eps ≥ 2^-125`; `DivFull` may give `var' = −2^-148`).
3. **rstd.** `tl.sqrt` and `1/·` add `(1 ± 4u)` each (their absolute terms are below `2^-84`
   relative, since `sqrt(var + eps) ≥ 2^-63` and `rstd ≥ 2^-64`). With `r* = 1/sqrt(S₀/N + eps)`,
   `|rstd' − r*| ≤ ρ·r*`; composed exactly the rounding part of `ρ` is at most `0.78` of
   `(D + 8)·2^-23` (`0.46` at `D = 15`), and the absolute part below `2^-147.4/eps`, under a fortieth
   of `gam`'s `(D + 8)·2^-145/eps` (`D ≥ 1`). (At `eps = 2^-125` that part of `gam` is
   `(D + 8)·2^-20`, about `2·10^-5` at `D = 15`; at `eps = 1e-5` it is negligible.) The mean's error:
   `|e_m| ≤ 0.501·(D + 4)·2^-23·A + 2^-147` (step 1) and `δ ≤ 1.00001·u·A + 2^-149.9`, so
   `δ' ≤ 0.102·meanTol` and `E ≤ 0.62·meanTol`. Case `cond ≤ 1/2`: `2δ'·s/(s² + eps) ≤ 2δ'·rstdRef ≤
   0.204·cond ≤ 0.102` and `E²/(s² + eps) ≤ 0.3844·cond²`, so `r*/rstdRef = sqrt((s² + eps)/(S₀/N + eps))`
   is at most `1/sqrt(1 − 0.204·cond) ≤ 1 + 0.111·cond` and at least
   `1/sqrt(1 + 0.204·cond + 0.3844·cond²) ≥ 1 − 0.102·cond − 0.193·cond²`: within `mTerm` of `1`
   (at most `0.82·mTerm`). Then `|rstd' − rstdRef| ≤ rstdRef·(ρ(1 + mTerm) + mTerm) ≤ rstdRef·(gam + mTerm) + 2^-140`
   (`ρ·(1 + 1/8) ≤ 0.90·gam`, the absolute part included). Case `cond > 1/2`: both are positive, `rstdRef ≤ 1/sqrt(eps)`, and the
   input of `tl.sqrt` is at least `eps(1 − 10u)`, so `rstd' ≤ (1 + 14u)/sqrt(eps) + 2^-148`;
   `|rstd' − rstdRef| ≤ max(rstd', rstdRef) ≤ (1 + gam)/sqrt(eps) + 2^-140` (`gam ≥ 18u`).
4. **Standardised value.** Lane `j` of line 92 rounds `(1 − α'_j)(x j − meanRef) − e'_j` with
   `|α'_j| ≤ u` and `|e'_j| ≤ E` (as line 78, the branch chosen anew), so
   `z' = fl(fl((1 − α'_j)(x j − meanRef) − e'_j)·rstd')` and
   `z' − zRef = (x j − meanRef)·((1 − α'_j)(1+a)(1+c)·rstd' − rstdRef) − e'_j(1+a)(1+c)·rstd'`
   (`|a|, |c| ≤ u`). Case `cond ≤ 1/2`: the first term is at most
   `|zRef|·((1+u)³(ρ(1 + mTerm) + mTerm) + 3.0002u)`; the second at most
   `0.62·meanTol·(1+u)²·rstdRef·(1 + 0.13) ≤ (1 + 2·gam)·cond`; the absolute terms are below
   `2^-140·(1 + rstdRef)`. Since `(1+u)³ρ + 3.0002u ≤ 0.97·gam` (`17.4u` of `gam = 18u` at `D = 1`, `0.53` at `D = 15`;
   the absolute part included), `(1+u)³ρ·mTerm ≤ 2·gam·mTerm` and `(1+u)³·mTerm ≤ (1 + 2·gam)·mTerm`, `zTol` bounds `|z' − zRef|`.
   Case `cond > 1/2`: `|z' − zRef| ≤ |zRef| + |z'|`, and
   `|z'| ≤ (1+u)³(|x j − meanRef| + E)·rstd' + (1+u)²·η·rstd' + η`, within
   `(1 + 2·gam)(|x j − meanRef| + meanTol)/sqrt(eps) + 2^-140·(1 + 1/sqrt(eps))` (`(1+u)³(1 + 14u) ≤
   1 + 2·gam` keeps room for `rstd'`'s `2^-148`).
5. **Output.** `y32 = mul_add(z', w, b)`: `|y32 − (z'w + b)| ≤ 2u·|z'w| + u·|b| + 2η` (fused or not).
   `y' = float16(y32)`: `u16·|y32| + η16`, and no overflow: `|y32| ≤ |w|(|zRef| + zTol) + |b| +
   (rounding) ≤ 2^15·(1 + 3u) < 65504` (`YRoom`). So
   `|y' − yRef| ≤ 2^-11·|yRef| + (1 + 2^-11)((1 + 2u)·|w|·zTol + 2u·|w||zRef| + u·|b| + 2η) + 2^-25`,
   within `2^-10·|yRef| + (1 + 2^-10)·|w|·zTol + 2^-21·(|w|·|zRef| + |b|) + 2^-24`
   (`(1 + 2^-11)(1 + 2u) ≤ 1 + 2^-10`; `2u(1 + 2^-11) ≤ 2^-21`; the absolute part
   below `2^-24`). The relative `2^-10` is twice the rounding's `2^-11`.
6. **The stated tolerances.** Steps 1–5 give the fine bounds: `meanTol` itself; for `rstd`,
   `rstdRef·(gam + mTerm) + 2^-140` (`cond ≤ 1/2`) or `(1 + gam)/sqrt(eps) + 2^-140`
   (`cond > 1/2`); for `y`, `2^-10·|yRef| + (1 + 2^-10)·|w|·zTol + 2^-21·(|w|·|zRef| + |b|) +
   2^-24`. Each is at most the stated one on every covered instance (`N ≥ 128`, so
   `7 ≤ D ≤ 15`; `eps ≥ 2^-125`):
   - `rstd`, `cond ≤ 1/2`: `gam = (D + 8)·2^-23 + (D + 8)·2^-145/eps ≤ D·2^-21 + 2^-140/eps`
     (`D + 8 ≤ 4·D` for `D ≥ 3`; `D + 8 ≤ 23 < 2^5`), and `mTerm = cond/8 + cond²/4 ≤ cond/4`
     (`cond ≤ 1/2`).
   - `rstd`, `cond > 1/2`: `gam ≤ 23·(2^-23 + 2^-20) < 2^-15` and `1/sqrt(eps) ≥ 2^-63`, so
     `(1 + gam)/sqrt(eps) + 2^-140 ≤ (1 + 2^-14)/sqrt(eps) ≤ 2/sqrt(eps)`.
   - `y`: `|yRef| ≤ yMag`, so `2^-10·|yRef| + 2^-21·yMag ≤ (2^-10 + 2^-21)·yMag ≤ 2^-9·yMag`;
     `(1 + 2^-10)·|w|·zTol ≤ 2·|w|·zTol` (`zTol ≥ 0`); `2^-24` is kept.

Against the kernel's errors on the GPU: the `y` tolerance is about four times the output's
`float16` rounding, which dominates the kernel's error (ratio `0.250`); `meanTol` and `rstdTol` are
about ten and thirty times the device's errors (ratios up to `0.10` and `0.053`): they take every
rounding of a depth-`D` tree at its worst, where the device's partly cancel. Against the fine
bounds of steps 1–5, on the file's launches (the test and the 30 benchmark launches): `rstdTol` is
`2.3×` larger (`tol/rstd` median `8.6·10^-6`, worst `9.2·10^-6`, against `3.7·10^-6` and
`3.9·10^-6`); `yTol` is `2×` larger at the median (`tol/|y|` median `2.0·10^-3` against
`9.9·10^-4`), more where `yRef` cancels to near `0` (`yMag ≫ |yRef|`); `meanTol` is unchanged.

## Exclusions

- **Dimensions that are not multiples of 128 or exceed 2^17, and contexts longer than 2^22
  tokens** — out of scope: the claim covers real model shapes only. The fields: `N` (`n_dim`: a
  multiple of `128`, at most `2^17`) and `M`, the batch's tokens (`m_ctx`: at most `2^22`). Every
  launch of the file is covered (`N = 8192` and `N = 512·i`; `M = 1151`, `4096`).
- **`(M − 1)·N ≥ 2^31` (a limit of the kernel).** `row * stride` (lines 63–64) is the product of two
  `int32` values and wraps, so the last rows read and write the wrong elements. The first launch is
  `M = 65537`, `N = 32768` (`2^31` elements, `4 GiB` per tensor); the file's largest is `2^26`
  elements. Real use with large batches of long rows reaches it, and the host does not check.
- **`eps < 2^-125`.** Below `2^-126` this is a **limit of the kernel**: the kernel's `float32` `eps`
  is subnormal, so on a row of zero variance `var + eps` is subnormal, `tl.sqrt`
  (`sqrt.approx.ftz`) flushes it to `0`, `rstd = +inf` and `y = 0·inf = NaN`. The first launch is
  `M = 1`, `N = 2`, a constant row, `eps = 1e-39`; `check_gpu.py` shows `rstd = inf`, `y = NaN`.
  In `[2^-126, 2^-125)` the library's semantics do not exclude a subnormal `var + eps` (`DivFull`
  allows `−2^-148` for a zero variance); on the device the kernel is right there (`check_gpu.py`,
  `eps = 1.5·2^-126`). Real use (`eps` from `1e-5` to `1e-12`) reaches neither. Library gap.
- **`eps > 2^126`**: outside real use (a stabiliser far below 1). The kernel is right there on
  `float16` inputs up to the `float32` maximum (`check_gpu.py`, `eps = 3.4e38`).
- **Outputs without room in `float16`** (`YRoom` false): `|y|` near `65504` overflows the output's
  format (a limit of the output dtype, not of the kernel's arithmetic). Outside real use (the file's
  outputs are below `4`). `YRoom` uses `zTol`, so it also fails on a launch with a row of
  `cond > 1/2` whose `|w|·(|x − meanRef| + meanTol)/sqrt(eps)` reaches `2^15`: there the library's
  fused subtractions (lines 78 and 92, each lane its own value within `u·|mean|` of the stored
  mean) allow a variance near `0` with deviations near `u·|mean|`, hence a standardised value near
  `u·|mean|/sqrt(eps)`. This part is a library gap, not a limit of the kernel: on the device the two
  subtractions use the same fused product, and `y` stays finite (`check_gpu.py`: constant rows of
  `1000` at `eps = 2^-125` are outside `Pre`, and the kernel writes `y = b` or `b ± w` there). It is
  reached at `eps = 2^-125` by a constant row of `1000`, and at `eps = 1e-12` by a constant row of
  `2·10^4` with `|w| = 1` (`N = 2048`); at `eps = 1e-5` a constant row of `1000` has room.
- **Non-finite inputs**: no reference value.
- **`N = 1`**: the host passes `N` (and `stride`) equal to `1` as compile-time constants, a
  different compiled kernel whose `/ N` the compiler may fold; not part of the setup (the file's
  smallest `N` is `1024`), and a LayerNorm of one column is the constant `b`.
- **`N > 32768`**: the host raises `RuntimeError` (no launch).
- **The backward kernels, other dtypes**: not part of the setup.

## Library requests

- **One fused product per quotient.** `tl.subc`/`tl.addc` on a quotient let each lane and each use
  replace the quotient by its own exact value; on the device a broadcast quotient's fused product is
  one value, the same at lines 78 and 92. Stating that would keep `YRoom` on rows with
  `cond > 1/2` (Exclusions).

## Checks

1. **Builds**: `lake env lean ../claims/layer_norm_fwd_f16/Claim.lean` from `lean/` gives no error
   or warning, no `sorry`, no `axiom`. `consistent`, `satisfiable`, `mkInst_ok`, `setup_test_ok` and
   `setup_bench_ok` depend only on `propext`, `Classical.choice` and `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (`setup_test`, zeros).
3. **Covers the setup**: `setup_test_ok` and `setup_bench_ok` (all 30 benchmark sizes).
4. **GPU** (`check_gpu.py`, H200, Triton 3.8.0): the test's launch and the 30 benchmark launches
   with the file's input distributions, and edges: `N` = 2, 3, 1000, 32767, 32768; `M = 1` and
   `M = 2^16` (`N = 4096`); `eps = 2^-125`, `1`, `2^126`; constant rows (`1000`, also with
   `eps = 2^-125`) and near-constant rows; `|x|` up to `65504`; subnormal `x`; `w = b = 0`; spike
   rows with `|w| = 960`, `|b| = 1000`; `|w| = |b|` up to `2^10`. Every row checked. The frame holds
   everywhere (guard bands around `y`, `mean`, `rstd`; inputs unchanged), every element written and
   finite. Worst error / tolerance: `y` `0.250` (the `float16` rounding; margin `4.0×`), `rstd`
   `0.053`, `mean` `0.101` (`N = 3`); on the file's launches `0.250`, `0.050`, `0.092`. The
   constant rows with `eps = 2^-125` are outside `Pre` (`YRoom`, Exclusions); their `mean` and
   `rstd` are checked and within tolerance. Negatives
   (wrong outputs at the test's shape and at `N = 15872`): a mean that drops one term, a variance
   that drops one term, the variance divided by `N − 1`, an all-zero `y`, `y` with the weights
   shifted by one column; `Post` rejects each (error / tolerance at least `3.4`).
   The edges `N = 2, 3, 1000, 32767` are outside `n_dim`: they check `Post` beyond the claim.
   The two launches outside `Inst.ok` behave as Exclusions says. PASS.
