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
(`ref`), and its tolerance `tol = tol p (K / 128)` (`tolOut`), with

`tol p 0 = 0`, `tol p (t + 1) = tol p t + ρ·(|partSum p t| + tol p t + chunkAbs p t) + α`,

`partSum p t = Σ_{k < 128t} p k`, `chunkAbs p t = Σ_{128t ≤ k < 128t+128} |p k|`,
`ρ = relDot16 128 = 130·2^-22 ≈ 3.10·10^-5`, `α = absDot16 128 = 130·2^-126`.

`Post`: for every `i < M`, `j < N`, element `(i, j)` of `c` (element `i·N + j`) holds a `float16`
value `c` that is the rounding to nearest even (up to NaN payload) of a finite `float32` `v` with
`|v − S| ≤ tol`; and when `|S| + tol ≤ 65504`, `c` is finite and
`|c − S| ≤ tol + (|S| + tol)·2^-11 + 2^-25`. Every byte outside the matrix `c` (its first `M·N`
elements) is unchanged.

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
* `K` a positive multiple of `128` (the benchmark asserts it), `K ≤ 2^17` (Exclusions);
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
`hostInst_ok`, in both formats. The host's values meet `Pre` (above).

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

Closed form (not used by the proof): `tol ≤ ((1 + ρ)^T − 1)·Σ|p| + T·(1 + ρ)^T·α`, `T = K / 128`:
`0.20%` of `Σ|p|` at `K = 8192`, `0.40%` at `16384`, `3.2%` at `2^17`, plus at most `2^-109`.
The recursion is tighter when the partial sums cancel. On the host's data (GPU check, median over
outputs of `tol / |S|`, the same for both formats): `5.4·10^-4` at `K = 512`, `3.3·10^-3` at
`8192`, `5.4·10^-3` at `16384`, `3.0·10^-2` at `2^17`; the device's worst error is `0.80`, `0.36`, `0.23` and `0.04` of the whole
bound at these `K` (it is mostly the `float16` rounding at small `K`). The tolerance is relative to
`|acc| + Σ|a·b|` per step, so on cancelling products it is looser than the device's error (the
products cancel by a factor of 60 to 90 at `K = 8192 … 16384`); it still rejects an omitted or
repeated group of 32 products, a wrong scale or scale layout, swapped nibbles, or another tile's
element (`check_gpu.py`'s negative tests: each breaks `Post` on `85%` or more of the elements at
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
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0; worst error / whole bound, all elements in
  `float16`'s range unless said; `mxfp8` / `mixed`): the claim's float64 reference against the
  host's own reference (`compute_reference=True`) at `256 × 512 × 512`: `5·10^-8` relative for
  both; the test's launch `8192³` (inputs drawn from the test's distribution, not its seed-42
  bytes) `0.355` / `0.355`; the benchmark `K = 512` `0.800` / `0.800` (on the host's data the two
  formats have the same values), `K = 16384` `0.233` (`mxfp8`; at `8192³` and `K = 16384` the
  inputs are built by `host_like`, the host builder's steps in slices of rows, to stay under
  4 GB); `Inst.ok` edges: `M = N = K = 128` (half a tile) `0.887` / `0.890`, `128 × 384 × 256`
  `0.815` / `0.847`, `K = 2^17` `0.044` / `0.039`, `1152 × 640 × 1408` `0.622` / `0.646`; `Pre`
  edges: same-sign products `0.215` / `0.298` (and `0.027` / `0.027` at `K = 2^17`), cancelling
  partial sums `0.038` / `0.027`, every finite `e4m3` pattern and every `e2m1` code with some
  outputs beyond `float16` (the rounding clause checked for `±∞`) `0.752` / `0.722`, extreme scale
  bytes (`254` on `a`; `1` on an `e4m3` `b`, `2` on an `e2m1` `b`) `0.454` / `0.412`, subnormal
  elements `0` (exact). Worst `0.890` (margin `1.12×`, the `float16` rounding of a small-`K`
  output). Every case meets `Pre`; every element satisfies `Post`; the inputs are unchanged byte
  for byte (`torch.equal` against full copies) and the guards around `c` keep their bits; every
  element of `c` was written (it starts as a NaN sentinel). Negative tests at
  `1024 × 1024 × 512` and `× 8192`, both formats: all zero, a group of 32 products left out, one
  group's scale doubled, the scale layout with `(r % 128) // 32` and `(k % 128) // 32` exchanged,
  the neighbouring column, and for `mixed` the two nibbles of each byte swapped: each breaks
  `Post` (on `85%` to `100%` of the elements). Peak device memory 2.9 GiB.
