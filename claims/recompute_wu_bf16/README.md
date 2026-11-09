# Prepared claim: `recompute_wu_bf16`

Kernel: `source/wy_fast.py`, `recompute_w_u_fwd_kernel` (lines 69–133) with the helper `exp2`
(lines 65–66); the kernel lines are 61–62, 65–66 and 69–133, the host lines 1–57, 59 and 136–220.
Claim: [`Claim.lean`](Claim.lean), namespace `Claims.RecomputeWU`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

The covered path is `SETUP.md`'s: `USE_G = True` (`g` given), `bfloat16` `k`, `v`, `A`, `w`, `u`,
`beta` `float32` or `bfloat16`, `float32` `g`, chunk size `BT = 64` with `BK = BV = 64`, padded or packed (`IS_VARLEN`, with
`int32` or `int64` `cu_seqlens`), on an H200.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.RecomputeWU.main : Statement` (`Statement`:
`Claim.lean` line 752). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 407), and every initial memory `init` that `Pre` accepts (line 598), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 722).

**What `Post` asks** (line 722), three parts:
1. padded (`isVarlen = false`): for every `b < B`, `t < T`, `h < HV`,
   `TokOk d init fin (b * T) T t h`: the sequence starts at flat token `b·T` and has length `T`;
2. packed (`isVarlen = true`): for every `n < N`, `t < seqLen d init n`, `h < HV`,
   `TokOk d init fin (cuN d init n) (seqLen d init n) t h`: the sequence starts at `cu_seqlens[n]`
   (`cuN`, line 537) and has length `cu_seqlens[n+1] − cu_seqlens[n]` (`seqLen`, line 540), both
   read from `init`;
3. the frame: every byte outside `u.elems (uSet d)` and `w.elems (wSet d)` (lines 720, 723: the first
   `nTok·HV·V` elements of `u`, the first `nTok·HV·K` of `w`) is unchanged.

`TokOk d init fin s0 len t h` (line 710) asks, for token `t` of the sequence at `s0`, head `h`:
- `CellU` (line 659) for every `c < V`: element `vIdx (s0 + t) h c` of `u` reads in `fin` as a
  finite `bfloat16` `o` with `|o.toRat − uRef| ≤ tolU uRef uMag`. `uRef` (line 641) is, in `ℚ`,
  `Σ_{j<64, inSeq len t j} (aV (s0+t) h j).toRat · (vbV (colTok s0 t j) h c).toRat`: `aV` (line 631)
  is `A`'s column `j`, `colTok` (line 628) the column's flat token, `inSeq` (line 625) whether it is
  inside the sequence; `vbV` (line 635) is source line 115's operand, an exact IEEE function of `init`
  (`bf16(fl32(v·beta))` for a `float32` `beta`, the `bfloat16` product for a `bfloat16` one).
  `uMag` (line 647) is the same sum of `liftMag .bf16 A_j · liftMag .bf16 vb_j`;
  `tolU ref L = 2^-8·|ref| + 2^-15·L + 2^-119` (line 655);
- `CellW` (line 704) for every `c < K`: element `wIdx (s0 + t) h c` of `w` reads in `fin` as a
  finite `bfloat16` `o` with `|o.toReal − wRef| ≤ tolW wMag wFloor`. `wRef` (line 679)
  is exact, in `ℝ`: `Σ_{j<64, inSeq} A_j · kbX · gE`, with `kbX` (line 671) `k[f, kHead h, c] · beta[f, h]`
  (`kHead`, line 664: `h / (HV / H)`; `betaR`, line 667) and `gE` (line 675) `2^(g[f, h])`;
  `wMag` (line 685) and `wFloor` (line 692) the sums of `liftMag A_j · |kbX| · gE` and
  `liftMag A_j · (|kbX| + 1)`; `tolW M F = 2^-6·M + 2^-124·F + 2^-119` (line 700), for either
  dtype of `beta`.

Every element of every sequence's rows is constrained (no guard). Columns `j` outside the sequence
add nothing to the sums.

**What you are given** (`Assumptions`, line 733):
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotBF16`): every `bfloat16` `tl.dot` into `float32`, at any input precision, is a
  `DotLiftAcc` with `relDot16`, `absDot16`: within `relDot16 K·L + absDot16 K` of the exact
  `acc + Σ a·b`, `L = |acc| + Σ liftMag a · liftMag b`, given room (source lines 116 and 132);
- `dot_total_u`, `dot_total_w` (`Rules.Num.DotTotal P cfgU`, `cfgW`; lines 44, 49): source line 116's and
  source line 132's `tl.dot` have a result for all operands;
- `exp2` (`Rules.Num.Exp2F32`): `tl.math.exp2` on `float32` is within `2^-22·2^x + 2^-126` of `2^x`,
  given room (source line 121);
- `exp2_total` (`Rules.Num.MathTotal P .exp2 .binary32`): it has a result for every input.

**The kernel, as the model runs it** (`model` line 493 picks `recompute_w_u_fwd_kernel_b16`, line 212,
when `bbf`, else `recompute_w_u_fwd_kernel_b32`, line 62; `launch` line 511): the grid
`(g0 d, B·HV, 1)` (`g0`, line 403: `NT` packed, `cdiv(T, 64)` padded). Program `(i_t, i_bh)` takes
`i_b = i_bh / HV`, `i_h = i_bh % HV` (source line 92). Packed, it loads `chunk_indices[2·i_t]` (its
sequence `i_n`) and `[2·i_t + 1]` (its chunk), then `cu_seqlens[i_n]`, `[i_n + 1]` (`bos`, `eos`);
padded, `bos = i_b·T` and the length is `T` (source lines 94–98). Its rows are `i_t·64 + [0, 64)`, masked by
the sequence's length. It loads its rows of `beta` (source line 104) and all 64 columns of `A` (source line 107).
Loop 1 (source line 109, `cdiv(V, 64)` trips): load a `v` tile (source line 114), form `b_vb` (source line 115), dot with `cfgU`
(source line 116), round to `bfloat16`, store the `u` tile (source line 117). Then it loads `g` and takes `exp2` (source line 121).
Loop 2 (source line 123, `cdiv(K, 64)` trips): load a `k` tile of key head `i_h / (HV/H)` (source line 128), the
products with `beta` and `exp2(g)` (source lines 129, 131), round, dot with `cfgW` (source line 132), store the `w` tile (source line 133).
Programs write disjoint tiles of `u` and `w`, use no atomics, and share nothing: the route is
`TVA.meets_independent`. Each loop is `safe_forRange` with an invariant and per-trip write
footprints (the trip's 64 columns of `u` or `w`), disjoint across trips (PROVER_GUIDE §4). Packed,
a program's footprints depend on `init` (its sequence and chunk come from `chunk_indices`).

**What the proof must establish.**
- each launch has `debug = false` and is `Launch.Valid`: axis 0 below `2^31` (`grid_t`), axis 1 below
  `2^16` (`grid_bh`), fewer than `2^31` programs (`grid_size`), and `blockShapes` (line 506) accepted;
- each program is `Safe`: `i_b`, `i_h` (`safe_idiv`, `safe_irem`; `hv_pos`, `h_pos`, `hv_div`);
  packed, the four scalar loads (`safe_load1`) in bounds (`ci_len`, `cu_len`, `SeqOk.ci_ok`) with
  non-negative values (`ci_some`, `cu_some`) that the conversions keep (`safe_convert`; `n_lt`), and
  `bos + len ≤ T` (`cu_mono`, `cu_last`); every tile load and store in bounds
  (`safe_load_buf`, `safe_store_buf`; `k_len`, `v_len`, `beta_len` / `betab_len`, `A_len`, `g_len`,
  `w_len`, `u_len`; `Pre`'s `*v` fields), the `int32` `o_v`, `o_k` and comparisons not wrapping
  (`V_lt`, `K_lt`, `t_lt`; `t_ctx` gives `T ≤ 2^22`, so every token index is far below `2^31`, and
  `b_batch`, `n_batch` keep the sequence indices `≤ 8192`), the `int64` offsets not wrapping (`size_lt`; `TVA.Lemmas.Int`); the
  operations by `safe_cvt`, `safe_mul`, `safe_math` (`exp2_total`), `safe_dot` (`dot_total_u`,
  `dot_total_w`);
- the programs' write footprints are disjoint (padded: distinct `(i_t, i_bh)`; packed: `ci_once`,
  `cu_mono`, rows masked inside the sequence), and no program reads what any program writes
  (`w_k` … `w_u`, `u_k` … `u_ci`; `Buf.elems_disjoint`, `Buf.elems_disjoint_of`);
- every `(sequence, t, h)` is written by one program: padded `(t/64, b·HV + h)`, packed the `i` that
  `SeqOk.ci_all` gives for chunk `t/64` of sequence `n`;
- the values: for `u`, the row's `A_j` and the column `vbV` (`+0` outside the sequence), `dot` at
  `.ieee` with a `+0` accumulator, then `FloatV.cvt_err` (source line 117); for `w`, `FloatV.mul_err`
  (source lines 129, 131), `exp2` (`g ≤ 8` gives room), `cvt_err` (source line 132), `dot` at `.tf32`, `cvt_err`
  (source line 133); room and finiteness from `Pre`'s magnitudes (`KVOk`, `AOk`, `BetaOk`, `BetaOkB`, `GOk`).
  Derive the fine bounds of README `## Tolerances` (`2^-8·|uRef| + (1 + 2^-8)·(66·2^-22·uMag +
  66·2^-126) + 2^-134` for `u`; `2^-8·|wRef| + c·wMag + 2^-124·wFloor + 2^-119` with `c = 2^-8 + 2^-14`
  or `2^-7 + 2^-13` for `w`), then relax them to `tolU` and `tolW` by the steps given there (for `w`,
  `|wRef| ≤ wMag`);
- the frame: every written element lies in `uSet d` or `wSet d`, and bytes no program writes are
  unchanged.

**Already in `Claim.lean`** (do not prove again): `consistent` (line 756), `satisfiable` (line 1118)
and the 33 `setup_*_ok` theorems; `bytes_disj` (line 763) can be reused.

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

  `tolU = 2^-8·|uRef| + 2^-15·uMag + 2^-119`: one `bfloat16` rounding of the result, the dot
  product's error (`2^-15 ≈ 3.1·10^-5` of the lifted magnitude) and an absolute floor;
* **`w[s0 + t, h, c]`** (`c < K`) is a finite `bfloat16` `o` with `|o − wRef| ≤ tolW`: with
  `x_j = k[f_j, h // (HV//H), c]·beta[f_j, h]` and `e_j = 2^(g[f_j, h])`, exactly (in `ℝ`),
  `wRef = Σ_j A_j·x_j·e_j`, `wMag = Σ_j liftMag(A_j)·|x_j|·e_j`, `wFloor = Σ_j liftMag(A_j)·(|x_j| + 1)`,
  and

  `tolW = 2^-6·wMag + 2^-124·wFloor + 2^-119` for either dtype of `beta`: `2^-6` of the magnitude
  `wMag ≥ |wRef|` (the result's rounding and each operand's one or two roundings), and absolute
  floors;
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
  than `2^31` programs, `N < 2^31`;
- real model shapes (Exclusions): `T ≤ 2^22` (`t_ctx`: `T` is a context length, the sequence length,
  or the packed batch's total tokens); `B ≤ 8192` (`b_batch`) and, packed, `N ≤ 8192` (`n_batch`):
  batch sizes, the sequences of the batch.

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
`setup_edge_t1_k1_v1`, `setup_edge_varlen_empty`, `setup_edge_varlen_132x1` (edges). Every one has
`T ≤ 16384 ≤ 2^22` (`t_ctx`) and at most `132` sequences (`b_batch`, `n_batch`). `SETUP.md`'s edge
`B·HV = 65535` (`B = 13107`, `HV = 5`) has more than `8192` sequences and is not covered (Exclusions);
`check_gpu.py` runs it outside `Inst.ok`. `satisfiable` uses `setup_edge_t1_k1_v1` with every byte `0`.

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
`|ρ| ≤ |uRef| + E`, so `|o − uRef| ≤ 2^-8|uRef| + (1 + 2^-8)·E + 2^-134`, the fine bound of `u`,
and `o` is finite (`|ρ| < 2^113`).

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
`(1 + 2^-8)(2^-125 + 66·2^-146) ≤ 2^-124` and `(1 + 2^-8)·66·2^-126 + 2^-134 ≤ 2^-119`: the fine
bound of `w` for a `float32` `beta` is `2^-8·|wRef| + c·wMag + 2^-124·wFloor + 2^-119` with
`c = 2^-8 + 2^-14`. The output is finite (`|wRef| + E < 2^120`).

**`w`, `bfloat16` `beta`.** Line 129 rounds `k·beta` to `bfloat16`: `p0 = bf16(x)`
(`|p0 − x| ≤ 2^-8|x| + 2^-134`, exact when the product fits), and line 131 converts it to `float32`
exactly; then `e`, `p2 = fl32(p0·e)` and `kb = bf16(p2)` as above. Composing (the `2^-134` of `p0`
scaled by `2^g ≤ 2^8` gives `2^-126`), `|kb − x·2^g| ≤ (2^-7 + 2^-16 + 2^-21)·y + 1.02·2^-126·|x| +
1.01·2^-126 ≤ τ'_j := (2^-7 + 2^-15)·y + 2^-125(|x| + 1)`, and `liftMag(kb_j) ≤ (1 + 2^-6)·y_j +
2^-124(|x_j| + 1)`. So `E ≤ (2^-7 + 2^-15 + 66·2^-22·(1 + 2^-6))·wMag + (2^-125 + 66·2^-146)·wFloor +
66·2^-126`, and `(1 + 2^-8)(2^-7 + 2^-15 + 67.04·2^-22) = 0.0078897 ≤ 2^-7 + 2^-13 = 0.0079346`; the
floors as above: the fine bound of `w` with `c = 2^-7 + 2^-13`.

**From the fine bounds to `tolU` and `tolW`.** For `u`: `(1 + 2^-8)·66·2^-22 = 66.26·2^-22 ≤ 128·2^-22
= 2^-15` and `(1 + 2^-8)·66·2^-126 + 2^-134 ≤ 67·2^-126 ≤ 2^-119`, so the fine bound is at most
`2^-8·|uRef| + 2^-15·uMag + 2^-119 = tolU`; the rounding term `2^-8·|uRef|` stays as it is (folding it
into `uMag`, `2^-7·uMag`, would make the tolerance about `6` times larger on the check's data, since
`uMag` is about `3·|uRef|` there). For `w`: `|wRef| ≤ wMag` (each term `|A_j·x_j·e_j| ≤
liftMag(A_j)·|x_j|·e_j`), so `2^-8·|wRef| + c·wMag ≤ (2^-8 + c)·wMag ≤ (2^-8 + 2^-7 + 2^-13)·wMag =
0.011841·wMag ≤ 2^-6·wMag` for either `c` (`2^-8 + 2^-14 ≤ 2^-7 + 2^-13`), and the floors are the
same: the fine bound is at most `2^-6·wMag + 2^-124·wFloor + 2^-119 = tolW`.

The bounds are provable from `DotBF16`, `Exp2F32` and the library's IEEE rounding lemmas
(`FloatV.mul_err`, `cvt_err`); the dots' room and the outputs' finiteness come from `Pre`'s magnitudes.
Meaningful: `tolU` is one `bfloat16` rounding of `uRef` plus `3.1·10^-5` of `uMag` (median
`tolU / |uRef|` `0.40%` on the check's launches, the same as the fine bound's), and the device reaches
`0.988` of it. `tolW` is `1.56%` of `wMag`: one rounding of the result and one (or two, with a
`bfloat16` `beta`) of each operand would be `0.79%` (`1.18%`); its median `tolW / |wRef|` is `2.2%` to
`2.6%` on the check's launches (the fine bound's: `0.96%` with a `float32` `beta`, `1.5%` to `1.7%`
with a `bfloat16` one, so `tolW` is at most `2.3` times looser), and the device reaches `0.488` of it
(`0.719` with a `bfloat16` `beta`). These are the largest values measured on `check_gpu.py`'s inputs,
not the worst case: an output whose roundings all reach half an ulp in the same direction comes within
about `2^-6` of the fine bound, which is at most `0.50` of `tolW` (`float32` `beta`) or `0.76`
(`bfloat16` `beta`). Both bounds reject a 2% error in one element and `w` scaled by `1 + 2^-6`
(`check_gpu.py`).

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
* **Out of scope: dimensions that are not multiples of `128` or exceed `2^17`, contexts longer than
  `2^22` tokens, and batches of more than `8192` requests: the claim covers real model shapes
  only.** The fields are `T` (`t_ctx`: the sequence length, or the packed batch's total tokens),
  `B` (`b_batch`) and, packed, `N` (`n_batch`), the batch's sequences. `SETUP.md`'s edge
  `B·HV = 65535` at `B = 13107` is outside. `K` and `V` are head dimensions (`60`, `100` in the
  setup), not model dimensions, and stay as they are.
* **`g = None` (`USE_G = False`), `float16` `beta`, `float16`/`float32` `k`, `v`, `A`, chunk sizes 16 and
  32:** not part of `SETUP.md`'s launches (the gated callers pass `g`; this claim is the `bfloat16`
  kernel at chunk size 64).
* **Overlapping outputs, non-contiguous tensors:** the host allocates `w`, `u` fresh, and
  `chunk_gated_delta_rule` makes its inputs contiguous.

## Library requests

None.

## Checks

* `Claim.lean` builds with no output (no `sorry`, no `axiom`); `consistent`, `satisfiable` and the 33
  `setup_*_ok` are proved.
* `check_gpu.py` on the H200: 47 launches through the host (`beta` `float32` and `bfloat16`; one,
  `B = 13107`, outside `Inst.ok`'s `b_batch`) and one
  direct launch with guard regions, all passing; largest measured error / tolerance `0.9884` (`u`),
  `0.4876` (`w`, `float32` `beta`) and `0.7186` (`w`, `bfloat16` `beta`); every output finite, the
  inputs and the guards unchanged; every negative test rejected; `B·HV = 65536` fails to launch.
