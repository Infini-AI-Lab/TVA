# Prepared claim: `recompute_wu_bf16`

Kernel: `source/wy_fast.py`, `recompute_w_u_fwd_kernel` (lines 69–133) with the helper `exp2`
(lines 65–66); the kernel lines are 61–62, 65–66 and 69–133, the host lines 1–57, 59 and 136–220.
Claim: [`Claim.lean`](Claim.lean), namespace `Claims.RecomputeWU`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

The covered path is `SETUP.md`'s: `USE_G = True` (`g` given), `bfloat16` `k`, `v`, `A`, `w`, `u`,
`beta` `float32` or `bfloat16`, `float32` `g`, chunk size `BT = 64` with `BK = BV = 64`, padded or packed (`IS_VARLEN`, with
`int32` or `int64` `cu_seqlens`), on an H200.

## Specification

A sequence is its first flat token `s0` and its length `len`: padded, `(b·T, T)` for `b < B`; packed,
`(cu_seqlens[n], cu_seqlens[n+1] − cu_seqlens[n])` for `n < N`, read from the initial memory. Token
`t < len` is the flat token `s0 + t`; its chunk starts at `t // 64 · 64`, and column `j < 64` of its row
of `A` (`A[s0 + t, h, j]`) belongs to the chunk's token `f_j = s0 + t // 64 · 64 + j`, which is inside the
sequence when `t // 64 · 64 + j < len` (`inSeq`). Every value is read from the initial memory, by the
tensors' contiguous layouts (`kIdx`, `vIdx`, `wIdx`, `hIdx`, `aIdx`); value head `h` reads key head
`h // (HV // H)` (`kHead`). For every token, value head `h < HV`, and the sums over the columns inside
the sequence:

* **`u[s0 + t, h, c]`** (`c < V`) is a finite `bfloat16` `o` with `|o − uRef| ≤ tolU(uRef, uMag)`:
  `vb_j` is the kernel's line 115 operand, an exact function of the inputs (`vbV`): with a `float32`
  `beta`, `bf16(fl32(v[f_j, h, c] · beta[f_j, h]))`, the `float32` IEEE product rounded to nearest
  `bfloat16`; with a `bfloat16` `beta`, the IEEE `bfloat16` product `v[f_j, h, c] · beta[f_j, h]`; `uRef = Σ_j A_j·vb_j`,
  `uMag = Σ_j liftMag(A_j)·liftMag(vb_j)` (`Rules.Num.liftMag .bf16`: `|q|`, a subnormal counted at
  `2^-126`), and

  `tolU = 2^-8·|uRef| + (1 + 2^-8)·(relDot16 64·uMag + absDot16 64) + 2^-134`
  (`relDot16 64 = 66·2^-22 ≈ 1.6·10^-5`, `absDot16 64 = 66·2^-126`);
* **`w[s0 + t, h, c]`** (`c < K`) is a finite `bfloat16` `o` with `|o − wRef| ≤ tolW`: with
  `x_j = k[f_j, h // (HV//H), c]·beta[f_j, h]` and `e_j = 2^(g[f_j, h])`, exactly (in `ℝ`),
  `wRef = Σ_j A_j·x_j·e_j`, `wMag = Σ_j liftMag(A_j)·|x_j|·e_j`, `wFloor = Σ_j liftMag(A_j)·(|x_j| + 1)`,
  and

  `tolW = 2^-8·|wRef| + c·wMag + 2^-124·wFloor + 2^-119`, with `c = cW = 2^-8 + 2^-14` for a `float32`
  `beta` and `2^-7 + 2^-13` for a `bfloat16` one (one more rounding per operand, line 129);
* every byte outside the elements of `u` and `w` is unchanged.

So `u` is the chunk's `A` times the `bfloat16` products `beta ⊙ v` the kernel forms, up to the dot
product's error and one rounding to `bfloat16`; `w` is the exact `A·(beta·2^g ⊙ k)` up to one rounding
of the result and one (`float32` `beta`) or two (`bfloat16` `beta`) of each operand. The reference reads `A`, `v`, `k`, `beta`, `g` by the host's
layouts and the sequences from `cu_seqlens`, never from `chunk_indices` or from the kernel's own
offsets.

## Conditions

`Inst` records the index dtype `idx`, `beta`'s dtype (`bbf`), `B`, `T`, `H`, `HV`, `K`, `V`, `N`, `NT`,
`isVarlen` and the ten buffers (`cu`, `ci` empty on the padded path, where the host passes `None`;
`beta` (`float32`) and `betab` (`bfloat16`), the one of the other dtype empty). `Inst.ok` (each field
documented in `Claim.lean`):
- the host's own requirements: `H, HV, K, V ≥ 1`, `HV` a multiple of `H`; `idx ∈ {int32, int64}`;
  packed batches `[1, T, ·, ·]`; buffers holding their tensors; `w`, `u` fresh (disjoint from every
  input and from each other);
- bounds outside real use: `T < 2^31` (an `int32` argument), `K + 64 ≤ 2^31` and `V + 64 ≤ 2^31` (the
  `int32` column offsets and comparisons, lines 110–111, 124–125), and
  `(B·T + 64)·HV·(K + V + 64) < 2^62` (every element offset, masked lanes included, far inside `int64`);
- kernel limits (Exclusions): `B·HV < 2^16` (grid axis 1), the grid's axis 0 below `2^31` and fewer
  than `2^31` programs, `N < 2^31`.

`Pre`: the seven tensors' buffers valid; every element of `k`, `v` finite with `|·| ≤ 2^32`, of `A`
finite with `|·| ≤ 2^64` (all 64 columns of every token's row: the kernel reads them all), of `beta` (of
its dtype) finite with `|·| ≤ 2^8`, of `g` finite with `g ≤ 8` (any negative value); packed: `SeqOk`, i.e.
`cu_seqlens` valid, non-negative, non-decreasing from `0` to `T`, and `chunk_indices` listing each
(sequence, chunk of 64 tokens) pair exactly once, as `prepare_chunk_indices` builds it (any order).

Part (A) of `Compiler.Faithful`'s scope, point by point: (4) the launch passes only `num_warps = 4` and
`num_stages = 3`; no `@triton.heuristics`, no `@triton.autotune` (removed in the setup, the flags
passed explicitly); the decorator gives no `debug`, so `debug = false`; (5) no descriptors; (6) a
direct launch of the JIT function, not a `CompiledKernel`; (8) no host descriptors. The claim relies on part (B) (the environment: H200, Triton 3.8.0 as installed, default
knobs, `FLA_USE_FAST_OPS` irrelevant since the source fixes `exp2`, the kernel new), which is assumed,
not checked.

## Assumptions

`[rule]` fields (`docs/rules/num.md`'s table): `faithful` (`Compiler.Faithful`), `dot` (`DotBF16`, for
both dots: `input_precision` `ieee` at line 116, `tf32` at line 132, `max_num_imprecise_acc = 0`),
`dot_total_u`, `dot_total_w` (`DotTotal P cfgU`, `DotTotal P cfgW`), `exp2` (`Exp2F32`), `exp2_total`
(`MathTotal P .exp2 .binary32`). `consistent` uses `Platform.sm90ref`.

## Model notes

The model is the source line by line (`model-map.json` maps every kernel line), once per dtype of
`beta`: Triton compiles the kernel for its arguments' pointer dtypes, and `model` picks
`recompute_w_u_fwd_kernel_b32` (`float32` `beta`) or `recompute_w_u_fwd_kernel_b16` (`bfloat16`
`beta`), which differ only at lines 103–104, 115, 129 and 131. Points to note:
- index arithmetic: the program ids are converted to `int64` (line 91), so `i_b`, `i_h`, `bos`, `eos`,
  `T` (packed), `o_t` and every element offset are `int64`; only `o_A`, `o_v`, `o_k` and the loop
  variables are `int32` (below `K + 64`, `V + 64`); `beta + bos*HV + i_h + o_t*HV` is added left to
  right, `g + (bos*HV + i_h) + …` sums first, as in Python;
- `i_bh // HV`, `i_bh % HV` and `i_h // (HV // H)` are the truncating `tl.idiv`, `tl.irem` on
  non-negative values;
- the `chunk_indices` loads are converted with `tl.convert sD .i32` / `.i64` (`.to(tl.int32)` truncates
  an `int64` sequence index; `N < 2^31` keeps it);
- `other=0.0` is the `float32` `+0`, converted to `bfloat16` for the `bfloat16` loads;
- with a `float32` `beta`, `b_v * b_b[:, None]`, `b_k * b_b[:, None]` and `b_kb *= b_g[:, None]` are IEEE
  `float32` products of the operands converted to `float32` (`tl.mul`; no addition consumes them, so
  nothing contracts), and the `.to(…)`s round to nearest `bfloat16` (`tl.cvt`); with a `bfloat16`
  `beta`, `b_v * b_b[:, None]` and `b_k * b_b[:, None]` are IEEE `bfloat16` products (both operands
  `bfloat16`; line 115's `.to(b_v.dtype)` is then the identity), and line 131 converts `b_kb` to
  `float32` (exact) for the `float32` product with `b_g`;
- both dots have the default `float32` `+0` accumulator and no addition after them (`tl.dot`, not
  `tl.dot_add`);
- the helper `exp2` is inlined: `x.to(tl.float32)` of a `float32` block is the identity, then
  `tl.math.exp2` (`tl.exp2`, `P.math .exp2`);
- the two loops carry no name; the `if USE_G:` lines are compile-time and taken (`config` drops).

## Coverage

`canon idx bbf B T H HV K V N NT varlen` is the host's launch for those sizes (contiguous tensors, the
ten storages `2^40` bytes apart). Each launch of `SETUP.md` is a `setup_*` instance with `setup_*_ok`:
`setup_chunk_b4_t1024_h4_d128`, `setup_chunk_b2_t75_h4_d64`, `setup_chunk_b2_t500_h3_d60`,
`setup_chunk_b2_t1000_h3_d64`, `setup_chunk_b3_t1024_h4_d100`, `setup_chunk_b2_t1500_h4_d128`,
`setup_chunk_b4_t2048_h8_d64`, `setup_chunk_b2_t256_h2_hv4_d64`, `setup_chunk_b2_t512_h2_hv8_d64`,
`setup_chunk_b2_t1024_h4_hv8_d128` (`test_chunk`, `float32` `beta`); `setup_varlen_h4_d60_n1`, `setup_varlen_h4_d64_n3`,
`setup_varlen_h4_d100_n5`, `setup_varlen_h2_hv4_d64_n3`, `setup_varlen_h2_hv8_d64_n3`
(`test_chunk_varlen`, `float32` `beta`); `setup_wu_b2_t128_h2_d64`, `setup_wu_b2_t128_h2_hv4_d64`,
`setup_wu_b1_t256_h4_d32`, `setup_wu_varlen_h2_d64`, `setup_wu_varlen_h2_hv4_d64`,
`setup_wu_varlen_h1_d64` (the kernel's own tests, `bfloat16` `beta`, with `g`);
`setup_qwen3next_b2_t4096`, `setup_qwen3next_varlen_i32`, `setup_fla_gdn_b2_t4096` (`float32` `beta`);
`setup_doc_b4_t2048_h4_hv8_d512`, `setup_doc_varlen_h4_hv8_d512`, `setup_gdp_b2_t4096_h6_k256_v512`,
`setup_gdp_varlen_h6_k256_v512`, `setup_precond_b2_t2048_h6_k256_v512`, `setup_mom_varlen_h4_k256_v512`
(the docstring, `GatedDeltaProduct`, `PrecondGatedDeltaNet`, `MoM`: `bfloat16` `beta`);
`setup_edge_t1_k1_v1`, `setup_edge_bhv65535`, `setup_edge_varlen_empty`, `setup_edge_varlen_132x1`
(edges). `satisfiable` uses `setup_edge_t1_k1_v1` with every byte `0`.

## Tolerances

Derived from the library's facts (`docs/rules/num.md`): `u = 2^-24`, `η = 2^-150` for `float32`;
`u_b = 2^-8`, `η_b = 2^-134` for `bfloat16` (`FloatV.cvt_err`); `DotBF16`: `relDot16 64 = 66·2^-22`,
`absDot16 64 = 66·2^-126` of `L = |acc| + Σ liftMag a·liftMag b` (`acc = +0`); `Exp2F32`: `2^-22·2^x +
2^-126`. Fix a token (row `r` of its program), head `h`, column `c`.

**`u`.** The dot's operands (line 116) are the row's `A_j` (all 64 loaded, finite) and the column of
`b_vb`: inside the sequence `vb_j` (with a `float32` `beta` the IEEE product of the exact widening of
`v` and `beta`, then `tl.cvt`; with a `bfloat16` `beta` the IEEE `bfloat16` product; finite since
`|v·beta| ≤ 2^40`); outside, `v` and `beta` load as `+0`, so `b_vb = +0` and the
term is `0` exactly. `DotBF16`: the result `ρ` is within `E = relDot16 64·uMag + absDot16 64` of
`uRef` (room: `uMag ≤ 64·2^64·2^41 = 2^111`). The store (line 117): `|o − ρ| ≤ 2^-8|ρ| + 2^-134`,
`|ρ| ≤ |uRef| + E`, so `|o − uRef| ≤ 2^-8|uRef| + (1 + 2^-8)·E + 2^-134 = tolU`, and `o` is finite
(`|ρ| < 2^113`).

**`w`.** Per column inside the sequence, with `y_j = |x_j|·e_j`: `p1 = fl32(k·beta)` (`|p1 − x| ≤
u|x| + η`), `e = exp2(g)` (`|e − 2^g| ≤ 2^-22·2^g + 2^-126`; room since `g ≤ 8`), `p2 = fl32(p1·e)`
(`u|p1·e| + η`), `kb = bf16(p2)` (`2^-8|p2| + 2^-134`; no overflow, `|p2| < 2^49`). Composing,
`|kb − x·2^g| ≤ (2^-8 + 2^-21)·y + 1.01·2^-126·|x| + 2^-133 ≤ τ_j := (2^-8 + 2^-20)·y + 2^-125(|x| + 1)`.
Outside the sequence `kb = +0`. The dot (line 132, `DotBF16`, room as for `u`) is within
`relDot16 64·L + absDot16 64` of `Σ A_j·kb_j`, with `liftMag(kb_j) ≤ (1 + 2^-7)·y_j + 2^-124(|x_j| + 1)`, so
`L ≤ (1 + 2^-7)·wMag + 2^-124·wFloor`; and `|Σ A_j·kb_j − wRef| ≤ Σ liftMag(A_j)·τ_j`. So the dot's
result is within `E ≤ (2^-8 + 2^-20 + 66.52·2^-22)·wMag + (2^-125 + 66·2^-146)·wFloor + 66·2^-126` of
`wRef`, and after the store `|o − wRef| ≤ 2^-8|wRef| + (1 + 2^-8)·E + 2^-134`, where
`(1 + 2^-8)(2^-8 + 2^-20 + 66.52·2^-22) = 0.0039384 ≤ 2^-8 + 2^-14 = 0.0039673`,
`(1 + 2^-8)(2^-125 + 66·2^-146) ≤ 2^-124` and `(1 + 2^-8)·66·2^-126 + 2^-134 ≤ 2^-119`: that is
`tolW` for a `float32` `beta`. The output is finite (`|wRef| + E < 2^120`).

**`w`, `bfloat16` `beta`.** Line 129 rounds `k·beta` to `bfloat16`: `p0 = bf16(x)`
(`|p0 − x| ≤ 2^-8|x| + 2^-134`, exact when the product fits), and line 131 converts it to `float32`
exactly; then `e`, `p2 = fl32(p0·e)` and `kb = bf16(p2)` as above. Composing (the `2^-134` of `p0`
scaled by `2^g ≤ 2^8` gives `2^-126`), `|kb − x·2^g| ≤ (2^-7 + 2^-16 + 2^-21)·y + 1.02·2^-126·|x| +
1.01·2^-126 ≤ τ'_j := (2^-7 + 2^-15)·y + 2^-125(|x| + 1)`, and `liftMag(kb_j) ≤ (1 + 2^-6)·y_j +
2^-124(|x_j| + 1)`. So `E ≤ (2^-7 + 2^-15 + 66·2^-22·(1 + 2^-6))·wMag + (2^-125 + 66·2^-146)·wFloor +
66·2^-126`, and `(1 + 2^-8)(2^-7 + 2^-15 + 67.04·2^-22) = 0.0078897 ≤ 2^-7 + 2^-13 = 0.0079346`; the
floors as above: that is `tolW` with `cW = 2^-7 + 2^-13`.

The bounds are provable from `DotBF16`, `Exp2F32` and the library's IEEE rounding lemmas
(`FloatV.mul_err`, `cvt_err`); the dots' room and the outputs' finiteness come from `Pre`'s magnitudes.
Meaningful: `tolU` is one `bfloat16` rounding of `uRef` plus `1.6·10^-5` of `uMag`, and the device
reaches `0.992` of it; `tolW` is one rounding of `wRef` plus one rounding of each operand
(`0.40%` of `wMag`; two roundings, `0.79%`, with a `bfloat16` `beta`), and the device reaches `0.968`
of it (`0.948` with a `bfloat16` `beta`). These are the largest values measured on `check_gpu.py`'s
inputs, not the worst case: an output whose roundings all reach half an ulp in the same direction
comes within about `2^-6` of the bound (`1 − 2^-6` of `tolW`). Both bounds reject a 2% error in one
element and `w` scaled by `1 + 2^-6` (`check_gpu.py`).

## Exclusions

* **`B·HV ≥ 65536` on a padded batch (a limit of the kernel).** The host's grid is `(NT, B·HV)`, and
  CUDA allows at most `65535` on axis 1: the launch fails with `invalid argument` (`check_gpu.py` at
  `B = 16384`, `HV = 4`), no outcome. `Inst.ok.grid_bh` keeps `B·HV < 2^16`. The first launch that hits
  it is a padded batch of `65536 / HV` sequences, e.g. `B = 2048` at Qwen3-Next's 32 value heads, or
  `B = 10923` at fla's default 6 heads. Training batches are far smaller, and packed batches
  (`B = 1`) never reach it.
* **`2^31` sequences or more (a limit of the kernel).** Line 94 truncates the sequence index to
  `int32`; `Inst.ok.n_lt` keeps `N < 2^31`. Not reached by real use.
* **No `int32` offset limit.** The program ids are converted to `int64` (line 91), so `bos·HV·V`,
  `bos·H·K`, `o_t·(HV·K)` and every other element offset are `int64`; the claim covers tensors up to
  `2^62` elements (`size_lt`, far beyond any device). The GPU check stays below 4 GB of device memory,
  so its largest tensors have `2^26` elements.
* **Input values.** `|A| > 2^64`, `|k|` or `|v| > 2^32`, `|beta| > 2^8`, `g > 8`, and non-finite values
  are outside real use (`k` L2-normalised, `beta` in `(0, 2)`, `g ≤ 0`, `A` the inverse of a unit
  lower-triangular matrix with entries of order 1); they keep the `float32` products, the dots and
  the outputs inside their formats.
* **`T ≥ 2^31`, `K` or `V` above `2^31 − 64`, tensors of `2^62` elements:** outside real use.
* **`g = None` (`USE_G = False`), `float16` `beta`, `float16`/`float32` `k`, `v`, `A`, chunk sizes 16 and
  32:** not part of `SETUP.md`'s launches (the gated callers pass `g`; this claim is the `bfloat16`
  kernel at chunk size 64).
* **Overlapping outputs, non-contiguous tensors:** the host allocates `w`, `u` fresh, and
  `chunk_gated_delta_rule` makes its inputs contiguous.

## Library requests

None.

## Checks

* `Claim.lean` builds with no output (no `sorry`, no `axiom`); `consistent`, `satisfiable` and the 34
  `setup_*_ok` are proved.
* `check_gpu.py` on the H200: 47 launches through the host (`beta` `float32` and `bfloat16`) and one
  direct launch with guard regions, all passing; largest measured error / tolerance `0.9921` (`u`),
  `0.9676` (`w`, `float32` `beta`) and `0.9483` (`w`, `bfloat16` `beta`); every output finite, the
  inputs and the guards unchanged; every negative test rejected; `B·HV = 65536` fails to launch.
