# `block_scaled_matmul_mxfp8`: the prepared claim

Kernel: `source/10-block-scaled-matmul.py` (Triton 3.8.0's tutorial 10, `SETUP.md`),
`block_scaled_matmul_kernel` (lines 174–240; the kernel lines are 123–124 and 174–240, the host
lines as in [`claim.json`](claim.json); `block_scaled_matmul_kernel_cdna4`, the AMD kernel,
is unused). Claim: [`Claim.lean`](Claim.lean), namespace `Claims.BlockScaledMxfp8`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Launches to cover: [`SETUP.md`](SETUP.md). GPU
check: [`check_gpu.py`](check_gpu.py).

The covered paths are the `mxfp8` and `mixed` ones on the H200 (`Inst.mixed`): a `float8e4nv` `a`
(`M × K`) and a `b` (`N × K`) of `float8e4nv` (`mxfp8`) or of `e2m1` packed two per byte along `K`
(`mixed`, `uint8` `N × K / 2`), with `e8m0` scales in groups of 32 along `K`;
`tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)` (line 234) or, when
`MIXED_PREC`, `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)` (line 230)
into `float32` over `K / 128` steps, the tile rounded to `float16` and stored through a host
descriptor; `BLOCK_M = 128`, `BLOCK_N = 256`, `BLOCK_K = 128`, `b`'s block `[256, 128]` or
`[256, 64]`, a 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.BlockScaledMxfp8.main : Statement` (`Statement`:
`Claim.lean` line 505). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 246), and every initial memory `init` that `Pre` accepts (line 449), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 469).

**What `Post` asks** (line 469), two parts:
1. for every `i < M`, `j < N`, `OutSpec d init fin i j` (line 460): element `i·N + j` of `c` reads
   in `fin` as a `float16` `c`, and there is a finite `float32` `v` with
   `|v.toRat − ref d init i j| ≤ vTol d init i j` and `FloatV.NaNEq (FloatV.cvt v) c`; and when
   `|ref| + tolOut ≤ binary16.maxVal`, `c` is finite and `|c.toRat − ref| ≤ cTol d init i j`.
   `ref` (line 417) is the exact sum, in the rationals, of the `K` scaled products `prods`
   (line 413: `xa`, `bVal`, the scale bytes `sa`, `sb` at `sIdx` (line 327), all read from `init`).
   `vTol` (line 428) is `2^-14·μ + 2^-108` and `cTol` (line 432) is `2^-11·|ref| + 2^-14·μ + 2^-24`,
   with `μ = magOut d init i j` (line 424), `stepMag` (line 369) over the `K / 128` steps:
   `Σ_t (|partSum p t| + chunkAbs p t)` (lines 343, 346). `tolOut` (line 421), the step recursion
   `tol` (line 362, with `rho`, `alpha`, lines 351, 354), enters only the `float16` range condition
   (and `Pre`'s room). This holds for every element: `Pre`'s `ValsOk` (line 439) puts every input in
   the domain;
2. the frame: every byte outside `cEl d` (line 241, the first `M·N` elements of `c`) is unchanged.

**What you are given** (`Assumptions`, line 476), five fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotScaledE4m3`, a `DotScaledAcc`): the `tl.dot_scaled` of model line 147
  (`cfgMx`, `mxfp8`) is within `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K` of `acc + Σ a·b` when
  the scaled values are in `scaledDomBF16`, the scale bits in `e8m0Dom`, the accumulator finite,
  with room in `float32`;
- `dot_mixed` (`Rules.Num.DotScaledE4m3E2m1`): the same for model line 141 (`cfgMixed`, `mixed`);
- `dot_total`, `dot_total_mixed` (`Rules.Num.DotScaledTotal` at `cfgMx`, `cfgMixed`): those
  `tl.dot_scaled` always have a result. The `cfgFp4` branch (line 144) is never reached.

**The kernel, as the model runs it** (`block_scaled_matmul_kernel` line 73, `model` line 288,
`launch` line 301): a 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs. Program `pid` takes
`pid_m = pid % num_pid_m`, `pid_n = pid / num_pid_m` (`tl.irem`, `tl.idiv`), then runs
`tl.forRange 0 (K/128) 1`: each trip loads (`tl.desc_load`, host descriptors `aDesc`, `bDesc`,
`aScaleDesc`, `bScaleDesc`, lines 210–228) the `a` tile `[128, 128]`, the `b` tile
`[256, 128 / epbB]` and the two 5-D scale blocks, reshapes and permutes the scales (pure), and
accumulates with `tl.dot_scaled`. After the loop it rounds the accumulator to `float16` (`tl.cvt`)
and stores the `[128, 256]` tile through `cDesc` (`tl.desc_store`, lanes outside `c` not stored).
Programs write disjoint tiles and use no atomics: the route is `TVA.meets_independent`; the loop is
`safe_forRange` with an invariant (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1` with `grid_lt`; the block
  shapes are fixed sizes);
- each program is `Safe`: `num_pid_m > 0` (`M_ok`); the `int32` offsets `offs_am`, `offs_bn`,
  `offs_k_a`, `offs_k_b`, `offs_scale_*` do not wrap (`M_ok`, `N_ok`, `K_ok`); each descriptor
  access is allowed (`safe_desc_load`, `safe_desc_store`: alignment from `base_al`) and in bounds
  (`a_len`, `b_len`, `as_len`, `bs_len`, `c_len`); `safe_dot_scaled` (shape and types of `cfgMx` or
  `cfgMixed`, with `dot_total` or `dot_total_mixed`), `safe_cvt`; the loop body writes nothing;
- the programs' write tiles of `c` are disjoint (distinct `(pid_m, pid_n)`), and no program writes
  what any program reads (`c_in`);
- the value part of part 1, `|v − S| ≤ 2^-14·μ + 2^-108` and `|c − S| ≤ 2^-11·|S| + 2^-14·μ + 2^-24`:
  derive the fine bound of README Tolerances (an invariant over trips `t`,
  `|acc_t − partSum p t| ≤ tol p t` with `acc_t` finite; each step from `dot` or `dot_mixed`
  (domain from `ValsOk.a_dom`, `b_dom`; the permuted scale byte is `sIdx K r k`; for `mixed`,
  `rhsBits` reads nibble `k % 2` of byte `k / 2`, which is `xbBits`; room from `ValsOk.room`); the
  padded lanes of a half tile hold `+0` and are not stored; then `FloatV.cvt_err` for the store),
  then relax it by the closed form given there (`tolOut ≤ 1.0323·(ρ·μ + T·α)`, `T = K/128 ≤ 2^10`);
- the bytes no program writes give part 2.

The real-shape conditions (`M_ctx`: `M ≤ 2^22`; `N_dim`: `N ≤ 2^17`; with `K_ok`, `K ≤ 2^17`) keep
every `int32` offset and descriptor shape entry far below `2^31`: `offs_am < M ≤ 2^22`,
`offs_bn < N + 128 ≤ 2^17 + 128`, `offs_k < K ≤ 2^17`, and the grid below `2^15 · 2^9 = 2^24`
programs.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `satInst_ok`,
`hostInst_ok` and the `setup_*_ok` theorems; `bytes_disjoint` (line 537), `elems_lt_sub` (line 546),
`bDT_bytes` (line 551) and `div_epbB_le` (line 560) can be reused.

## Specification

Element `(i, k)` of `a` is element `i·K + k` of its storage (`xa`). Element `(j, k)` of `b`
(`xbBits`, read in `b`'s format `bFmt`: `bFin`, `bVal`) is byte `j·K + k` (`mxfp8`, `e4m3`), or
nibble `k % 2` of byte `j·(K / 2) + k / 2` (`mixed`, `e2m1`: the low nibble for an even `k`, as
`MXFP4Tensor.to_packed_tensor` packs `(high << 4) | low`). The scale byte of `a`'s element `(i, k)` is byte `sIdx K i k` of `a_scale` (`sa`,
and `sb` for `b`), with

`sIdx K r k = (r / 128·(K / 128) + k / 128)·512 + 16·(r % 32) + 4·((r % 128) / 32) + (k % 128) / 32`,

the place the host's own reference reads it (`unpack_scale`, lines 411–417: `reshape(…, 32, 4, 4)`,
`permute(0, 3, 2, 1, 4)`, `reshape(R, K / 32)`, `repeat_interleave(32)`), so the reference does not
follow the kernel's 5-D loads and permutes. The scaled value of an element `x` with scale byte `s`
is `x·2^(s − 127)`; the products of output `(i, j)` are
`p k = scaled(xa i k, sa i k)·scaled(bVal j k, sb j k)` (`prods`), its exact value `S = Σ_{k<K} p k`
(`ref`), and the magnitude the loop's `T = K / 128` steps carry (`magOut`, `stepMag`):

`μ = Σ_{t<T} (|partSum p t| + chunkAbs p t) = Σ_k |p k| + Σ_{t<T} |partSum p t|`,

`partSum p t = Σ_{k < 128t} p k`, `chunkAbs p t = Σ_{128t ≤ k < 128t+128} |p k|`: each step's error
is relative to the accumulator's magnitude plus its products' magnitudes.

`Post`: for every `i < M`, `j < N`, element `(i, j)` of `c` (element `i·N + j`) holds a `float16`
value `c` that is the rounding to nearest even (up to NaN payload) of a finite `float32` `v` with

`|v − S| ≤ 2^-14·μ + 2^-108` (`vTol`);

and when `|S| + tolOut ≤ 65504`, `c` is finite and

`|c − S| ≤ 2^-11·|S| + 2^-14·μ + 2^-24` (`cTol`): one `float16` rounding of the result plus `v`'s
tolerance.

`tolOut = tol p T` is the step recursion `tol p 0 = 0`,
`tol p (t + 1) = tol p t + ρ·(|partSum p t| + tol p t + chunkAbs p t) + α`
(`ρ = relDot16 128 = 130·2^-22 ≈ 3.10·10^-5`, `α = absDot16 128 = 130·2^-126`); it enters only the
range conditions (`OutSpec`'s `float16` condition and `Pre`'s room). Every byte outside the matrix
`c` (its first `M·N` elements) is unchanged.

## Conditions

`Inst`: the format (`mixed`: `true` for `"mixed"`, `false` for `"mxfp8"`), `M`, `N`, `K` and the
five storages (`a` `float8e4nv`; `b` of `bDT mixed`, `uint8` or `float8e4nv`; `a_scale`, `b_scale`
`uint8`; `c` `float16`). The descriptors are the host's (`aDesc`, …, `cDesc`: the contiguous
shapes and strides, block shapes `[128, 128]`, `[256, 128 / ELEM_PER_BYTE_B]`, `[1, 1, 1, 2, 256]`,
`[1, 2, 1, 2, 256]`, `[128, 256]`, padding `"zero"`). The model passes `ELEM_PER_BYTE_A = 1` and
`ELEM_PER_BYTE_B = epbB mixed` (`1` or `2`) and keeps the source's three-way compile-time branch
(lines 229–234) as a Lean `if` on them. `Inst.ok`:

* `M`, `N` positive multiples of `128` below `2^31` (the host's scale layout has `M // 128` and
  `N // 128` blocks; descriptor shape entries are `int32`); the last column of tiles may be half
  outside `c` (`N % 256 = 128`), whose lanes read the padding and are not stored;
* `K` a positive multiple of `128` (the benchmark asserts it), `K ≤ 2^17` (Exclusions): a model
  dimension (the input features);
* `M ≤ 2^22` (`M_ctx`): `M` counts tokens (the rows of the activations `a`), a context length;
* `N ≤ 2^17` (`N_dim`): `N` is a model dimension (the output features), with `N_ok` a positive
  multiple of `128` at most `2^17`;
* the grid has fewer than `2^31` programs;
* every storage 16-byte aligned (TMA) and long enough for its tensor (`b`: `N·K` or `N·K / 2`
  bytes);
* `c`'s matrix shares no byte with the inputs (the inputs may overlap each other: they are only
  read).

`Pre`: the five buffers `Valid`, and `ValsOk`: every element of `a` and `b` with its scale in the
library facts' domain (`InDom`: finite, the scale byte in `1 … 254`, the scaled value `0` or of a
normal `bfloat16` magnitude, `[2^-126, bfloat16.maxVal]`; every `e2m1` pattern is finite), and
room for every output element, `Σ_k |p k| + tol ≤ 2^127`. The host's values always meet it, in
both formats: scaled values are `0` or at least `2^-28`, at most `6·2^-1`, `Σ|p| ≤ 9K`.

The launch: grid `(cdiv(M, 128)·cdiv(N, 256), 1)`, the source's eleven block shapes,
`fusion = true`, `debug = false`, `num_ctas = 1`, no `launch_pdl`; the launch gives no option
(`NUM_STAGES` and `disallow_acc_multi_buffer` are kernel arguments); no autotuning or heuristics, a
`JITFunction` launch (`launch_metadata=` only names a host function, `_matmul_launch_metadata`,
lines 155–171); the kernel builds no descriptor (no allocator needed); the host descriptors keep
`round_f32_to_tf32 = False`: within part (A) of `Compiler.Faithful`'s scope. The claim relies on
part (B) (the environment). Point 7 is the H200, where Triton emulates `tl.dot_scaled`
(`docs/rules/num.md`, "`tl.dot_scaled` on sm_90").

## Assumptions

`[rule]` fields: `faithful` (`Compiler.Faithful`), `dot` (`Rules.Num.DotScaledE4m3 P`: `"e4m3"` by
`"e4m3"`, `e8m0` scales, groups of 32, `K`-packed, either `fast_math`; line 234), `dot_mixed`
(`Rules.Num.DotScaledE4m3E2m1 P`: `"e4m3"` by `"e2m1"`, otherwise the same; line 230), `dot_total`
and `dot_total_mixed` (`Rules.Num.DotScaledTotal P` for the model's configurations `cfgMx` and
`cfgMixed`). No `[kernel]` field. The branch of line 232 (`cfgFp4`) is reached by no covered
instance (`ELEM_PER_BYTE_A = 1`), so it needs no fact.
`consistent` uses `Platform.sm90ref`.

## Coverage

`hostInst mixed M N K` is the host's instance (contiguous tensors, fresh `c`); `hostInst_ok`
proves `Inst.ok` for either format and every size meeting `Inst.ok`'s bounds (each tensor below
`2^40` bytes). SETUP.md's launches: for `mxfp8`, `setup_test_8192` (the test, `8192³`),
`setup_bench_k512` (the default `-K`), `setup_bench_k8192`, `setup_bench_k10240`,
`setup_bench_k12288`, `setup_bench_k14336`, `setup_bench_k16384` (the docstring's range) and
`setup_bench_k131072` (the largest covered `K`, `M = N = 8192`); for `mixed`,
`setup_mixed_test_8192`, `setup_mixed_bench_k512` and `setup_mixed_bench_k131072`; each with its
`_ok`. Every other `-K` the benchmark accepts (a multiple of `128`) up to `2^17` is covered by
`hostInst_ok`, in both formats. The host's values meet `Pre` (above). Every launch meets the
real-shape conditions (`M = N = 8192`, `K ≤ 2^17`), so none is left out.

## Tolerances

The bound is provable from `DotScaledE4m3` (`mxfp8`) or `DotScaledE4m3E2m1` (`mixed`), which have
the same constants and domain, and the library's rounding lemma for the store, by an induction
over the loop's `K / 128` steps; `K` enters only through the `int32` loop bounds and
offsets (`offs_k_a ≤ K`, `K / 128` trips, `K < 2^31`) and the room condition, so it holds at every
covered `K`. Derivation, per output element and step `t` (`acc_t` the accumulator before step `t`):

1. **One `tl.dot_scaled` step** of 128 products: `DotScaledE4m3` or `DotScaledE4m3E2m1` (relative constant
   `relDot16 128 = 130·2^-22`, absolute `absDot16 128 = 130·2^-126`), applicable since the row of
   `a` and the column of `b.T` are in `InDom` and `acc_t` is finite: `acc_{t+1}` is finite and
   within `ρ·(|acc_t| + chunkAbs p t) + α` of `acc_t + Σ_{chunk} p`. The scale's permute and
   reshape (lines 226–227) put the byte `sIdx K r k` at entry `(r, k / 32)` exactly. For `mixed`,
   step `t` loads bytes `64t … 64t + 63` of each row of `b` (`offs_k_b += 64`), and the library's
   `rhsBits` reads element `k` of the `K`-packed `b.T` as nibble `k % 2` of byte `k / 2`, which is
   `xbBits`.
2. **Induction**: with `|acc_t − partSum p t| ≤ tol p t`, `|acc_t| ≤ |partSum p t| + tol p t`, so
   `|acc_{t+1} − partSum p (t+1)| ≤ tol p (t + 1)`. The room condition keeps every step's bound
   below `2^127`, where the fact applies. `acc_0 = +0` exactly.
3. **The store**: `accumulator.to(float16)` is one rounding to nearest even (`tl.cvt`); with
   `FloatV.cvt_err`, `|c − v| ≤ |v|·2^-11 + 2^-25` when `|v| ≤ 65504`, which `|S| + tol ≤ 65504`
   gives.

So the fine bounds are `|v − S| ≤ tolOut = tol p T` (`T = K / 128`) and
`|c − S| ≤ tolOut + (|S| + tolOut)·2^-11 + 2^-25`.

**Relaxation to `vTol`, `cTol`.** Write `P_t = partSum p t`, `C_t = chunkAbs p t`, so
`tol p (t+1) = (1 + ρ)·tol p t + ρ·(|P_t| + C_t) + α`. Unrolled from `tol p 0 = 0`:
`tol p T = Σ_{t<T} (1 + ρ)^(T−1−t)·(ρ·(|P_t| + C_t) + α) ≤ (1 + ρ)^T·(ρ·μ + T·α)`, with
`μ = Σ_{t<T} (|P_t| + C_t) = stepMag p T`. `K ≤ 2^17` (`K_ok`) gives `T ≤ 2^10` and
`T·ρ ≤ 1024·130·2^-22 ≤ 0.0318`, so `(1 + ρ)^T ≤ e^0.0318 ≤ 1.0323`; then
`1.0323·ρ = 134.2·2^-22 ≤ 256·2^-22 = 2^-14` and `1.0323·T·α ≤ 1.0323·1024·130·2^-126 ≤ 2^-108.9`.
Hence `tolOut ≤ 2^-14·μ + 2^-108 = vTol`. For `c`:
`tolOut + (|S| + tolOut)·2^-11 + 2^-25 = 2^-11·|S| + (1 + 2^-11)·tolOut + 2^-25
≤ 2^-11·|S| + 134.3·2^-22·μ + 2^-108.8 + 2^-25 ≤ 2^-11·|S| + 2^-14·μ + 2^-24 = cTol`.
The `float16` rounding stays relative to `|S|`, not folded into `μ`: on the host's data `μ` is
about `100` to `170` times `|S|` at `K = 8192 … 16384` (medians; the products cancel), so
`2^-11·μ` would be `5%` to `8%` of the output.

Meaningful: on the host's data (GPU check, median over outputs, the same for both formats),
`vTol / |S|` is `1.07·10^-3` at `K = 512`, `6.4·10^-3` at `8192`, `1.06·10^-2` at `16384`,
`5.8·10^-2` at `2^17` (the fine bound `tolOut / |S|`: `5.4·10^-4`, `3.3·10^-3`, `5.4·10^-3`,
`3.0·10^-2`; since `tolOut ≥ ρ·μ`, `vTol ≤ 1.97·tolOut + 2^-108`); `cTol / |S|` is `1.6·10^-3`, `6.9·10^-3`,
`1.1·10^-2`, `5.9·10^-2` (the fine bound: `1.0·10^-3`, `3.7·10^-3`, `5.9·10^-3`, `3.0·10^-2`). The
device's worst error is `0.69`, `0.22`, `0.14` and `0.02` of `cTol` at these `K` (mostly the
`float16` rounding at small `K`). The tolerance is relative to `|acc| + Σ|a·b|` per step, so on
cancelling products it is looser than the device's error; it still rejects an omitted or
repeated group of 32 products, a wrong scale or scale layout, swapped nibbles, or another tile's
element (`check_gpu.py`'s negative tests: each breaks `Post` on `77%` or more of the elements at
`K = 512` and `8192`).

## Exclusions

* **`K > 2^17`: outside real use.** Not a kernel limit (the kernel is correct up to its `int32`
  offsets, `K < 2^31`), and the bound stays provable; it is left out because the tolerance grows
  with the number of steps (closed form `(1 + ρ)^(K/128) − 1`: `0.13` of `Σ|p|` at `K = 2^19`, and
  about `25%` of the typical output at `2^20` on the host's data). The file's launches use
  `512 … 16384` (SETUP.md); `2^17` is eight times the largest.
* **`M`, `N`, `K` not multiples of `128`, or zero.** Outside the file's launches: the host's scale
  tensors have `M // 128`, `N // 128`, `K // 128` blocks (a ragged `M` or `N` would leave rows
  without scales), the benchmark asserts `K % 128 == 0`, and a zero size gives a zero descriptor
  shape entry, which the host rejects.
* **out of scope**: dimensions that are not multiples of 128 or exceed 2^17, contexts longer
  than 2^22 tokens, and batches of more than 8192 requests: the claim covers real model shapes
  only. The fields: `N` (`N_dim`, with `N_ok`) and `K` (`K_ok`), model dimensions, and `M`
  (`M_ctx`), a token count; the kernel has no batch-size field.
* **Sizes at or above `2^31`** (descriptor shape entries and the `int32` offsets `pid_m * 128`,
  `pid_n * 256`, `offs_k_a`) and grids of `2^31` programs or more: a limit of the kernel and of
  TMA; real use (`8192`) is far below.
* **Input values outside `InDom` or without room.** NaN elements, scale bytes `0x00` (which the
  sm_90 emulation reads as `0`, not `2^-127`) and `0xFF` (NaN), scaled values below `2^-126` or
  above `bfloat16`'s range (the emulation multiplies by the scale in `bfloat16`, which may flush or
  overflow; for `e2m1`, a scale byte `1` with the value `0.5`), and products whose magnitudes reach
  `2^127`: the library states nothing about `tl.dot_scaled` there, and the host never produces them
  (scale bytes `100 … 126`, `e2m1` values).
* **`c` overlapping an input**, other output dtypes (`output_type` `0`, `2`), the `mxfp4` and `nvfp4`
  configurations (rejected on the H200 at the tutorial's config, out of shared memory) and the AMD
  kernel: not asked by SETUP.md.

## Library requests

None.

## Checks

* **Builds**: `lake env lean ../claims/block_scaled_matmul_mxfp8/Claim.lean`: no errors or
  warnings, no `sorry`, no `axiom`; `consistent`, `satisfiable`, `hostInst_ok` and the `setup_*_ok`
  use only `propext`, `Classical.choice`, `Quot.sound`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (`mxfp8`, `M = N = K = 128`,
  every byte `0x38`).
* **Covers the setup**: `hostInst_ok` and the eleven `setup_*_ok` above.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0; worst error / `cTol`, all elements in
  `float16`'s range unless said; `mxfp8` / `mixed`): the claim's float64 reference against the
  host's own reference (`compute_reference=True`) at `256 × 512 × 512`: `5·10^-8` relative for
  both; the test's launch `8192³` (inputs drawn from the test's distribution, not its seed-42
  bytes) `0.220` / `0.220`; the benchmark `K = 512` `0.692` / `0.692` (on the host's data the two
  formats have the same values), `K = 16384` `0.135` (`mxfp8`; at `8192³` and `K = 16384` the
  inputs are built by `host_like`, the host builder's steps in slices of rows, to stay under
  4 GB); `Inst.ok` edges: `M = N = K = 128` (half a tile) `0.809` / `0.809`, `128 × 384 × 256`
  `0.698` / `0.742`, `K = 2^17` `0.023` / `0.021`, `1152 × 640 × 1408` `0.457` / `0.498`; `Pre`
  edges: same-sign products `0.130` / `0.181` (and `0.014` / `0.014` at `K = 2^17`), cancelling
  partial sums `0.020` / `0.014`, every finite `e4m3` pattern and every `e2m1` code with some
  outputs beyond `float16` (the rounding clause checked for `±∞`) `0.645` / `0.613`, extreme scale
  bytes (`254` on `a`; `1` on an `e4m3` `b`, `2` on an `e2m1` `b`) `0.298` / `0.265`, subnormal
  elements `0` (exact). Worst `0.809` (margin `1.24×`, the `float16` rounding of a small-`K`
  output). Median `vTol / |S|` and `cTol / |S|` per case are printed (`## Tolerances`). Every case
  meets `Pre`; every element satisfies `Post`; the inputs are unchanged byte for byte
  (`torch.equal` against full copies) and the guards around `c` keep their bits; every element of
  `c` was written (it starts as a NaN sentinel). Negative tests at `1024 × 1024 × 512` and
  `× 8192`, both formats: all zero, a group of 32 products left out, one group's scale doubled, the
  scale layout with `(r % 128) // 32` and `(k % 128) // 32` exchanged, the neighbouring column, and
  for `mixed` the two nibbles of each byte swapped: each breaks `Post` (on `77%` to `100%` of the
  elements). Peak device memory 2.9 GiB.
