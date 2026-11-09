# `attn_fwd_f16`: the prepared claim

Kernel: `source/06-fused-attention.py` (tutorial 06 as set up in [`SETUP.md`](SETUP.md)), `_attn_fwd` (lines 176–245) with its `@triton.jit` helpers
`_attn_fwd_inner` (lines 47–110) and `_maybe_make_tensor_desc` (lines 168–173), in the `float16`
run (`FP8_OUTPUT = False`), with `BLOCK_M = 128`, `BLOCK_N = 64`, `num_stages = 2`, `num_warps = 4`,
on the launches with `warp_specialize = False`, where the host passes host `TensorDescriptor`s.
The kernel lines are 20–21, 47–110, 168–173 and 176–245; the host lines, the unused functions
(`keep`, `prune_invalid_configs` and the four backward kernels) and their headers are classed in
[`claim.json`](claim.json) and the model map. Claim: [`Claim.lean`](Claim.lean), namespace
`Claims.AttnFwd`, `Statement := ∀ K P, Assumptions K P → Spec K`. Model map:
[`model-map.json`](model-map.json). Launches: [`SETUP.md`](SETUP.md). GPU check:
[`check_gpu.py`](check_gpu.py).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.AttnFwd.main : Statement` (`Statement`:
`Claim.lean` line 569). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 253), and every initial memory `init` that `Pre` accepts (line 315), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 518).

**What `Post` asks** (line 518), three parts:
1. `o`: for every `bh < Z·H`, `n < N_CTX`, `e < HEAD_DIM`, element `idx d bh n e` (line 247) of `o`
   reads in `fin` as a `float16` `c`, for every such element. Only when `HeadOk d init bh` holds
   (line 364: the head's `q`, `k`, `v` finite, `|v| ≤ 2^15`, `sigma d * scoreMag ≤ 2^10`) is more
   asked: `c` is finite and, for every `X ≤ 2^10` with `sigma d * scoreMag d init bh n j ≤ X` for
   every `j ∈ keys d n` (line 334),
   `|c.toReal − attnRef| ≤ tolO D N X · attnAbs + tolS D N X · vAbsSum + tolF D N X`
   (`D = HEAD_DIM`, `N = N_CTX`). `attnRef` (line 375) is the exact attention in the reals, with
   weights `weight` (line 371) `= exp(σ·score)` from the values in `init` (`score` line 338, `val16`
   line 321); `attnAbs` (line 380) and `vAbsSum` (line 385) are the scales. `tolO`, `tolS`, `tolF`
   are defined at lines 487, 493, 498;
2. `M`: for every `bh < Z·H`, `n < N_CTX`, element `bh·N_CTX + n` of `M` reads in `fin` as a
   `float32` `c`. Under `HeadOk d init bh` only: `c` is finite and, for every such `X`,
   `|c.toReal − lseRef| ≤ tolM D N X + u32·|lseRef|`, `lseRef` (line 389) `= log₂ Σ_j weight`,
   `tolM` at line 504, `u32` at line 400;
3. the frame: every byte outside `outBytes d` (line 508: the first `rows d · HEAD_DIM` elements of
   `o` and the first `rows d` of `M`) is unchanged.

Outside `HeadOk` only "holds a value" is asked, in parts 1 and 2.

**What you are given** (`Assumptions`, line 538), ten fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotF16`): each `tl.dot` (model lines 93, 129) is within `relDot16 K` times
  the lifted magnitude `|acc| + Σ liftMag a·liftMag b`, plus `absDot16 K` (a `DotLiftAcc`);
- `dot_total` (`Rules.Num.DotTotal` at `cfg16`, line 50): both `tl.dot`s have a result;
- `exp2` (`Rules.Num.Exp2F32`, a `MathAcc`, `2^-22` relative, `2^-126` absolute): model lines
  117, 119;
- `exp2_ninf` (`Rules.Num.Exp2F32NegInf`, a `MathAt`): `exp2(−∞) = +0`, model line 119 at the first
  block (`m_i = −∞`);
- `exp2_total`, `log2_total` (`Rules.Num.MathTotal`): `exp2`, `log2` have a result;
- `log2` (`Rules.Num.Log2F32`, `2^-22` relative, `2^-149` absolute): model line 200;
- `div_full` (`Rules.Num.DivFull`, `2^-22` relative, `2^-148` absolute, for `|x/y| ≤ 2^126`):
  model line 202; `div_total` (`Rules.Num.DivTotal`): `/` has a result.

**The kernel, as the model runs it** (`attn_fwd` line 149, `attn_fwd_inner` line 61, `model`
line 299, `launch` line 308): a 2-D grid `((N_CTX + 127)/128, Z·H, 1)`, `BLOCK_M = 128`,
`BLOCK_N = 64`, `STAGE = 3` when causal and `1` otherwise. The descriptors are host descriptors
(`hostDesc`, line 291). Program `(start_m, off_hz)` splits `off_hz` by `H` (`tl.idiv`, `tl.irem`),
loads its `[128, HEAD_DIM]` query tile at row `qo_offset_y = off_hz·N_CTX + start_m·128`
(`tl.desc_load`), and runs `attn_fwd_inner`, a `tl.forRange lo hi 64` loop: non-causal one call over
`[0, N_CTX)`; causal two calls, over `[0, start_m·128)` (STAGE 1) and then the diagonal block
`[start_m·128, (start_m + 1)·128)` (STAGE 2, with the mask and `tl.multiple_of1`). Each trip loads
a key and a value tile (`tl.desc_load`), computes `qk` (`tl.dot`), the running max (`tl.max1`,
`tl.maximum`), `p = exp2(·)`, `alpha`, the row sum (`tl.sum1`), rescales `acc`, adds
`tl.dot(p.to(f16), v, acc)` and updates `l_i` (`tl.mul_add`). After the loops it computes
`m_i + log₂ l_i`, `acc / l_i`, stores `M` (`tl.store`, `[128]` lanes) and the `float16` output tile
(`tl.desc_store`). Programs write disjoint tiles and use no atomics: the route is
`TVA.meets_independent`; each loop is `safe_forRange` with an invariant (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (the `Launch` default; `Launch.Valid` from
  `zh_lt`, `rows_lt`, `n_mul`, `n_le`, and the block shapes from `head_dim`);
- each program is `Safe`: the divisor `H` is nonzero (`zh_pos`); the `int32` offsets
  (`y_dim`, `offset_y`, `qo_offset_y`, the loop bounds and `offsetk_y`, `offsetv_y`, the `M`
  pointer) do not wrap (`rows_lt`); every descriptor access is allowed (`safe_desc_load`,
  `safe_desc_store`: `D.Access` from `align`, `lens`) and in bounds of its buffer; the hint of
  line 75 holds; `safe_dot` (with `dot_total`), `safe_math` (with `exp2_total`, `log2_total`),
  `safe_div` (with `div_total`), `safe_max1`, `safe_maximum`, `safe_sum1`, `safe_mul_add`,
  `safe_subc`, `safe_cvt`, `safe_store`; the loop bodies write nothing;
- the programs' write tiles of `o` and `M` are disjoint (distinct `(start_m, off_hz)`; `n_mul`),
  and no program writes what any program reads (`disj`);
- the value parts of parts 1 and 2 under `HeadOk`: loop invariants on `m_i`, `l_i`, `acc` (README
  `## Tolerances`, steps 1–5: the exponents, the weights from `exp2` and `exp2_ninf`, the row sums
  with `sum_err_covers`, the accumulator from `dot` with `DotLiftAcc.mono`, the end from `div_full`
  and `log2`), then the change from `2^(qk_scale·s)` to `exp(σ·s)`;
- the bytes no program writes give part 3.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `canon_ok`,
`smTest_ok`, `smBench_ok` and the `setup_*_ok` theorems; `disj_of_le` (line 580) can be reused.

## Specification

Write `D = HEAD_DIM`, `N = N_CTX`, `σ = sm_scale` (the double, as a rational: `sigma`). Element
`(bh, n, e)` of `q`, `k`, `v`, `o` (for the head `bh = z·H + h`) is element
`idx d bh n e = (bh·N + n)·D + e` of its buffer (contiguous), and element `(bh, n)` of `M` is element
`bh·N + n`. Query `n` attends to `keys d n`: every `j < N`, and only `j ≤ n` when `causal`. From the
input values in `init` (`val16`):

- `score = Σ_e q[bh, n, e]·k[bh, j, e]`, `scoreMag = Σ_e liftMag q·liftMag k`
  (`Rules.Num.liftMag .f16`: `|x|`, a subnormal counted at `2^-14`; for normal values `Σ_e |q·k|`);
- `weight = exp(σ·score)` (real `exp`);
- `attnRef = Σ_j weight_j·v[bh, j, e] / Σ_j weight_j` (the exact attention), `attnAbs` the same
  weighted mean of `|v|`, `vAbsSum = Σ_j |v[bh, j, e]|`;
- `lseRef = log₂ Σ_j weight_j` (the exact log-sum-exp in base 2, which `M` holds: the kernel
  computes it as `m_i + log₂ l_i` with base-2 exponents).

A head is covered (`HeadOk d init bh`) when every element of `q`, `k`, `v` in it is a finite
`float16`, every `|v| ≤ 2^15`, and every `σ·scoreMag ≤ 2^10` (all `n, j < N`).

`Post d init fin`:

- **`o`.** Every element `(bh, n, e)` (`bh < Z·H`, `n < N`, `e < D`) holds a `float16` `c`. For a
  covered head, `c` is finite and, for every `X ≤ 2^10` bounding the row (`σ·scoreMag(n, j) ≤ X`
  for every key `j` of `n`),
  `|c − attnRef| ≤ tolO D N X · attnAbs + tolS D N X · vAbsSum + tolF D N X`.
- **`M`.** Every element `(bh, n)` holds a `float32` `c`. For a covered head, `c` is finite and,
  for every such `X`, `|c − lseRef| ≤ tolM D N X + 2^-24·|lseRef|`.
- **Frame.** Every byte outside the first `Z·H·N·D` elements of `o` and the first `Z·H·N` of `M`
  is unchanged.

For a head that is not covered the claim says only that its elements hold values. On the file's
launches every head is covered: their inputs are finite, `|v|` is far below `2^15`, and
`σ·Σ_e|q_e·k_e| = σ·s²·Σ_e|X_e·Y_e|` with `X`, `Y` independent standard normals (test: `s = 0.5`,
`σ = 0.5`; benchmark: `s = 1`, `σ = 1.3`): a Chernoff bound on the `128` terms with a union bound
over every pair of the largest shape keeps it below `22.7` (test) and `239.9` (benchmark) except
with probability below `10^-6`; the lift adds below `0.006` and `0.031` (a `χ²` tail of `|q|²`). The
GPU check measured at most `17.4` (test) and `185.6` (benchmark) over the rows it read.

| `D` | `N` | `X` | `tolO` | `tolS` | `tolF` | `tolM` |
|---|---|---|---|---|---|---|
| 64 | 128 | 16 | `1.8·10^-3` | `3.1·10^-8` | `3.2·10^-8` | `8.3·10^-4` |
| 128 | 4096 | 23 (test) | `4.3·10^-3` | `3.1·10^-8` | `3.2·10^-8` | `2.3·10^-3` |
| 128 | 2^14 | 240 (benchmark) | `2.9·10^-2` | `3.2·10^-8` | `3.4·10^-8` | `2.3·10^-2` |
| 256 | 2^16 | 1024 | `0.24` | `4.2·10^-8` | `4.2·10^-8` | `0.19` |

## Conditions

`Inst`: `Z`, `H`, `N_CTX`, `HEAD_DIM`, `sm_scale` (the double), `causal`, the
storages of `q`, `k`, `v`, `o` (`Buf .f16`) and `M` (`Buf .f32`). `Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `head_dim` | `HEAD_DIM ∈ {64, 128, 256}` | `16`, `32` are rejected on the host (line 188); the file uses `64`, `128` |
| `n_mul` | `0 < N_CTX`, `N_CTX % 128 = 0` | a ragged last tile is a kernel limit (Exclusions); every file size is a multiple of `128` |
| `n_le` | `N_CTX ≤ 2^16` | keeps the tolerance below `1`; the file's largest is `2^14` |
| `zh_pos`, `zh_lt` | `0 < Z`, `0 < H`, `Z·H < 2^16` | `off_hz // H`; the grid's `y` limit; the file's largest is `192` |
| `rows_lt` | `Z·H·N_CTX < 2^31` | `y_dim` (line 194) and the row offsets (lines 208–209, 243) are `int32`: a kernel limit; the file's largest is `2^21` |
| `scale` | `sm_scale` finite in `[2^-64, 2^64]` | `qk_scale` (line 219) a finite positive `float32`; the file passes `0.5`, `1.3` |
| `lens`, `align`, `disj` | the buffers hold their tensors, 16-byte aligned (TMA), `o` and `M` disjoint from each other and the inputs | fresh PyTorch tensors |

Every covered launch has `warp_specialize = False`: the host (`supports_host_descriptor()` and not
`is_hopper() and warp_specialize`, line 523) passes the descriptor arguments `hostDesc`, host
`TensorDescriptor`s `⟨b.ptr 0, [Z·H·N_CTX, HEAD_DIM], [HEAD_DIM, 1], block_shape, "zero"⟩` (the
pre-hook's block shapes), so `_maybe_make_tensor_desc` returns them (line 171) and never builds one
in the kernel (lines 172–173 are not visited).

`Pre`: the five buffers `Valid`. Any input values (`Post` is conditional on `HeadOk`).

The launch: grid `(N_CTX/128, Z·H, 1)`, the source's block shapes, `fusion = true`,
`debug = false`, `num_ctas = 1`, no `launch_pdl`; only `num_warps`/`num_stages` among the other
options (`extra_kern_args` is empty on the H200), no heuristics or autotuning (SETUP.md), a
`JITFunction` launch; the host sets an allocator with `triton.set_allocator` (lines 543–546) on
every launch, and a launch with host descriptors never calls it, so the allocator plays no part in the launch;
the host descriptors leave `round_f32_to_tf32` false.
That is part (A) of `Compiler.Faithful`'s scope; the claim relies on part (B) (the environment) as
the rule states it.

## Assumptions

`[rule]` fields (`docs/rules/num.md`, "Facts the library states"): `faithful`
(`Compiler.Faithful`), `dot` (`DotF16`, both `tl.dot`s, lines 73 and 103), `dot_total`
(`DotTotal P cfg16`, written out), `exp2` (`Exp2F32`, lines 82, 84), `exp2_ninf`
(`Exp2F32NegInf`, line 84 at the first block), `exp2_total` (`MathTotal P .exp2 .binary32`), `log2`
(`Log2F32`, line 241), `log2_total`, `div_full` (`DivFull`, line 242), `div_total`
(`DivTotal P .binary32`). No `[kernel]` field. `consistent` uses `Platform.sm90ref`.

## Coverage

Every launch of [`SETUP.md`](SETUP.md) is `canon Z H N D sm causal` (contiguous `q`, `k`, `v`,
fresh `o`, `M`, at fixed aligned addresses). `canon_ok` proves `Inst.ok` for every
`D ∈ {64, 128, 256}`, `N` a positive multiple of `128` up to `2^16`, `1 ≤ Z·H < 2^16` with
`Z·H·N < 2^31`, every `sm_scale` in range, both `causal`. Then:

- `setup_test_ok Z H N D causal`: every test launch (`Z ∈ {1, 4}`, `H ∈ {2, 48}`,
  `N ∈ {128, 1024, 4096}`, `D ∈ {64, 128}`, both modes, `sm_scale = 0.5`, host descriptors);
- `setup_bench_ok i D causal`: every `float16` benchmark launch with `warp_specialize = False`
  (`Z = 4`, `H = 32`, `N = 2^i` for `10 ≤ i ≤ 14`, `D ∈ {64, 128}`, both modes, `sm_scale = 1.3`);
- the named instances `setup_test_small` (`1, 2, 128, 64`, causal), `setup_test_large`
  (`4, 48, 4096, 128`), `setup_bench_causal` (`4, 32, 2^14, 128`, causal) and
  `setup_bench_noncausal` (`4, 32, 2^14, 128`), each with its `_ok`.

`satisfiable` uses `setup_test_small` in a memory where every byte is present.

## Tolerances

The tolerances (`Claim.lean`, "The tolerances") compose only library constants: `relDot16 D = (D + 2)·2^-22` (`DotF16`, the score's `tl.dot`) and
`relDot16 64` (the `p·v` `tl.dot`), `2^-22` (`Exp2F32`, `DivFull`, `Log2F32`), one `float32`
rounding `2^-24` and one `float16` rounding `2^-11`. Their derivation, step by step (each step a library fact or an IEEE rounding lemma, so the
bound is provable from the cited facts):

1. **Exponents.** `qk_scale = fl(fl32(σ)·fl32(1.44269504))` is within `2^-22·σ` of `σ·log₂e`
   relative and at most `3σ/2`. Each computed exponent `qk·qk_scale − m_i` is within
   `expErr D X = (relDot16 D + 2^-20)·(3X/2) + 2^-50` of the exact one: `DotF16` relative to
   `scoreMag`, the roundings of the products and of `−m_i`. The running maximum stays within
   `expErr` of an attended exponent (`MInv`); a masked lane's exponent is below `−2^19` (the
   `−10^6`), so its `p` enters only through absolute terms.
2. **Weights.** `p = exp2(·)`: `Exp2F32` (`2^-22`, `2^-126`); `alpha = exp2(m_i − m_ij)`, and
   `exp2(−∞) = +0` at the first block (`Exp2F32NegInf`); `p.to(float16)`: `2^-11` relative,
   `2^-25` absolute (`psiP`, `psi16`).
3. **Row sums.** `l_ij = tl.sum(p)` (63 additions, any tree: `(1 + u)^63`), `l_i = mul_add(l_i,
   alpha, l_ij)` (`3u` and alpha's error per block: `stepL^(N/64)`), so `l_i` is within `fl − 1`
   relative and `N/64·2^-100` absolute of `Σ 2^(exponent)` (`LInv`).
4. **Accumulator.** `acc·alpha` (`u`), `tl.dot(p, v, acc)` (`DotF16` at `K = 64`: `relDot16 64` of
   `|acc| + Σ liftMag p·liftMag v`), compounding as `stepAcc^(N/64)` (`AInv`). A subnormal `p` adds
   `66·2^-36·|v|` per key (`omega0`, the `tolS·vAbsSum` term); a subnormal `v` adds `66·2^-36·|p|`,
   at most `kapA = 2^-29` per unit of weight and `absA = 2^-37` per block (in `tolF`).
5. **End.** `acc / l_i` (`DivFull`), `relQ = (facc − 1 + betaL)/(1 − betaL)` against the kernel's
   own weighted mean; the change from `2^(qk_scale·s)` to `exp(σ·s)` (`thetaW = 2^-20·X`); the
   rounding to `float16` (`lastRel`, `2^-25` near zero). `M = m_i + log₂ l_i`: `Log2F32` with
   `|log₂ l_i| ≤ 17`, `log₂` of `l_i`'s error (`2·betaL`), `2^-21·X`, one rounding (`tolM`).

**Meaningful.** At the test's sizes (`D = 128`, `N = 4096`, `X = 23`) `tolO` is `0.43%` of the
weighted mean of `|v|` (about nine `float16` roundings); at the benchmark's (`N = 2^14`, `X = 240`)
`2.9%`; at the largest covered values `24%`. The GPU check reaches `0.52` of the bound on the
test's launches, `0.18` on the benchmark's, and `0.84` on an adversarial input (subnormal `q`, `v`),
so the bound is within a factor of about 2 of the kernel's error on the file's launches. `tolS` is
about `2^-25` per key of `Σ|v|`; `tolF` (below `4.3·10^-8`) is mostly the output's own `float16`
rounding near zero. The bound is relative to `attnAbs`, not to `|attnRef|` (for zero-mean `v` the
output can be much smaller than `attnAbs`); it grows with `X` because the library's `tl.dot` fact
bounds each score's error relative to `scoreMag`.

## Exclusions

- **`N_CTX` not a multiple of `128` (a limit of the kernel).** The query tile of the last program of
  a head runs past the head's rows: through the descriptors over the `[Z·H·N_CTX, HEAD_DIM]` view it
  reads the next head's queries and keys and stores into the next head's rows of `o`, a race with
  that head's first program (and the last key block attends to the next head's keys). The first such
  launch is `N_CTX = 129` with `Z·H ≥ 2`; with `Z·H = 1` the descriptor's bounds clip the stores but
  the padding keys (zeros) are attended in the non-causal mode, so the output is wrong too. The file's
  launches are all multiples of `128` (and `prune_invalid_configs`, unused here, assumed
  `N_CTX ≥ BLOCK_M`); real use outside the file would reach it.
- **`Z·H·N_CTX ≥ 2^31` (a limit of the kernel).** `y_dim = Z * H * N_CTX` (line 194) and the row
  offsets (lines 208–209, 243) are `int32` and wrap. The first launch is `Z·H·N_CTX = 2^31`, e.g.
  `Z = 4`, `H = 32`, `N_CTX = 2^24` (far beyond the file's `2^21`, and beyond the device's memory at
  any useful `HEAD_DIM`).
- **`N_CTX > 2^16`.** The kernel is correct there (up to the `int32` limit), but the bound grows
  with the blocks (`stepAcc^(N/64)`) and `X`; the file's largest is `2^14`. Not a kernel limit.
- **`Z·H ≥ 2^16`**: the grid's `y` limit, no launch. `HEAD_DIM ∈ {16, 32}`: rejected on the host by
  line 188 (no outcome); other head dims are rejected by the wrapper's assert.
- **Input values** outside `HeadOk` (non-finite values, `|v| > 2^15`, `σ·scoreMag > 2^10`): `Post`
  says only that `o` and `M` hold values there. A head with `|v|` near `2^16` can overflow `o`'s
  `float16` (a limit of the output format); scores beyond `2^10` are outside real use and would make
  the bound vacuous.
- **`sm_scale` outside `[2^-64, 2^64]`**, zero or negative: outside real use.
- **`warp_specialize = True`** (out of scope: a tensor descriptor built inside the kernel): on the H200 the host then passes the raw tensors (lines
  538–541) and the kernel builds its descriptors with `tl.make_tensor_descriptor` (line 173). The benchmark makes such
  launches: the non-causal `fwd`-mode forward at both `HEAD_DIM` and every `N_CTX` (`enable_ws`,
  line 713, runs it with `warp_specialize` `False` and `True`); its `False` twin is covered. The test
  passes `False`. (`attention`'s own default is `True`.)
- **The `float8e5` run, other configs, other devices**: not this folder's (`attn_fwd_e5m2`;
  SETUP.md).

## Library requests

None. (The tolerance's growth with `N/64` comes from `DotF16`'s `relDot16 64` on the accumulator per
block.)

## Checks

- **Builds**: `lake env lean ../claims/attn_fwd_f16/Claim.lean`: no errors or warnings, no `sorry`,
  no `axiom`; `consistent`, `satisfiable`, `setup_test_ok`, `setup_bench_ok` use only `propext`,
  `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (`setup_test_small`).
- **Covers the setup**: `canon_ok`, `setup_test_ok`, `setup_bench_ok` and the four `setup_*_ok`.
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 48 test launches (every head of the small
  ones, four heads of the large ones), all 20 benchmark launches with `warp_specialize = False` (`Z = 4`, `H = 32`,
  `N = 2^10 … 2^14`, both `HEAD_DIM`, both modes, `warp_specialize = False`; three or four heads
  each, every row), and `Inst.ok`/`HeadOk` edges: `HEAD_DIM = 256` (causal and not),
  `N_CTX = 2^16` (`D = 256`, and causal at `D = 64`), `Z·H = 4096`, `sm_scale = 2^-60`, `X = 1000`
  (causal and not, and at `D = 256`, `N = 4096`), `|v| = 2^15`, uniform attention (`q = 0`,
  `v = 1` and `v = 1/3`, `N = 2^14`), subnormal `q` and `v`. Worst error / tolerance: `o` `0.842`
  (the subnormal edge; `0.52` on the test's launches, `0.18` on the benchmark's), `M` `0.048`
  (`0.007` on the file's launches). Every element of `o` and `M` written and finite; the frame
  holds (`o` and `M` are allocated inside larger NaN-filled buffers whose guard bands stay unchanged,
  and every input is unchanged, and no other tensor is allocated: the allocator the host sets is
  never called).
