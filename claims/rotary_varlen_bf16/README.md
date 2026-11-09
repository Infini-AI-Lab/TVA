# Prepared claim: `rotary_varlen_bf16`

`Claim.lean` (namespace `Claims.RotaryVarlen`) states the claim `Statement : ∀ K P, Assumptions K P → Spec K`
for `rotary_embedding_kernel` (`source/rotary.py`, lines 63–154) as its host
`rotary_embedding_fwd` (lines 200–272) launches it: the forward of the rotary embedding on `bfloat16` queries
or keys, padded or packed (`cu_seqlens` `int32` or `int64`), interleaved (GPT-J) or not (GPT-NeoX), with one
integer offset or a tensor of one offset per sequence (`int32` or `int64`, negative under left padding). The file
proves `consistent`, `satisfiable` and the coverage facts `setup_*_ok`; it does not prove `Statement`.

## Specification

`Post d init fin`: for every sequence `n` (`nSeq`: the batch entries `b < B` when padded, `n < N` when
packed), token `l` of it (`lenAt`: `l < T` padded, `l < cu_seqlens[n+1] − cu_seqlens[n]` packed), head `h < H`
and rotated channel `c < 2 R`, with the flat token `bosAt n + l` (`n·T + l` padded, `cu_seqlens[n] + l`
packed) and the position `p = l + offAt n` (`offAt n`: the one `int` offset, or `seqlen_offsets[n]` from the
tensor):
- **`p ≥ 0`**: `CellOk` at that flat token and table position `p`;
- **`p < 0`** (a left-padding token under a tensor of offsets, source line 106's `o_cs >= 0`): the element's
  bytes are unchanged (the host's `torch.zeros_like`, line 237, made them `0`).
The sequence bounds and offsets are read from the host's `cu_seqlens` and `seqlen_offsets` in `init`; nothing
in `Post` depends on `chunk_indices` or on how the kernel indexes.
- `CellOk`: with `a = x[t, h, c]`, `b = x[t, h, partner(c)]`, `cs = cos[pos, pair(c)]`,
  `sn = sin[pos, pair(c)]` (not interleaved: pair `c mod R`, partner `c ± R`; interleaved: pair `c // 2`,
  partner `c ± 1` by parity), the final `y[t, h, c]` is a finite `bfloat16` `v` with
  `|v − ref| ≤ 2^-8·|ref| + 2^-22·(|a·cs| + |b·sn|) + 2^-132`, where `ref = a·cs − b·sn` for the first channel
  of a pair (`c < R`, or `c` even) and `a·cs + b·sn` for the second: the rotation of the source's `reference`
  (lines 275–300), in exact rational arithmetic on the initial `bfloat16` values.
- **Frame**: every byte outside the rotated channels of `y` (the set `outSet`) is unchanged. `y`'s channels from
  `2 R` on keep what the host put there (`x`'s values), and no input is written.

Every output the kernel writes is constrained: every rotated channel of every token of `y` (the host covers
every token: `cu_seqlens` runs from `0` to `T`).

## Conditions

`Inst` records the shapes, `BT`, `BD`, the flags (`offT` is `IS_SEQLEN_OFFSETS_TENSOR`), the `int` offset
`off`, the index dtype `idx`, the offsets' dtype `offD`, and the seven buffers (`cu`, `ci` empty on the padded
path, where the host passes `None`; `offs` empty with an `int` offset). `Inst.ok` (each field documented in
`Claim.lean`):
- the host's own requirements: `D ≤ 256`, `2 R ≤ D`, `T ≤ TR` padded, `off + T ≤ TR` for an `int` offset, a
  tensor of offsets `int32` or `int64` with one entry per sequence (asserts, lines 217–234); `R ≥ 1`,
  `idx ∈ {int32, int64}`, packed batches `[1, T, H, D]` (SETUP.md); `BT`, `BD` powers of two with
  `2 R ≤ BD` and a tile of at most `2^20` elements (wider than the host's `BT ≤ 128`, `BD = next_pow2(2R)`);
  buffers holding their tensors; `y` fresh (disjoint from every input, the offsets included);
- bounds outside real use: `T < 2^31` and `off < 2^31` (`int32` arguments), `TR·R < 2^62` and
  `B·T·H·D < 2^62`;
- kernel limits (Exclusions): `B < 2^16`, `H < 2^16`, fewer than `2^31` programs, `N < 2^31`.

The integer arguments: `T` is in `do_not_specialize`, so a runtime `int32` even at `T = 1`; the offset `1` is a
compile-time constant, whose only use gives the same `int64` value (`model`'s docstring).

`Pre`: `x`, `y`, `cos`, `sin` valid; only what the kernel reads is constrained: every rotated channel
(`c < 2 R`) of every token and head of `x` finite with `|x| ≤ 2^126` (channels from `2 R` on may hold
anything, NaN included), and every entry of the table rows the launch reads, the positions `l + offAt n ≥ 0` of
the tokens `l` of every sequence `n`, of `cos` and `sin` finite with `|·| ≤ 1` (they are cosine and sine tables, `cos_sin_cache`; other rows may hold
anything); a tensor of offsets: `offs_vals`, valid, every sequence's offset `v` with `len + v ≤ TR` (the
host's `torch._assert_async`, lines 228–232) and `v ≥ −2^31`; packed: `SeqOk`, i.e.
`cu_seqlens` valid, non-negative, non-decreasing from `0` to `T`, and `chunk_indices` listing each (sequence,
chunk of `BT` tokens) pair exactly once, as `prepare_chunk_indices` builds it (any order is covered).

Part (A) of `Compiler.Faithful`'s scope, point by point: (4) the launch passes only `num_warps = 4` and
`num_stages = 2`; no `@triton.heuristics`, no `@triton.autotune` (removed in the setup); the decorator gives no
`debug`, so `debug = false`; (5) no descriptors; (6) a direct launch of the JIT function, not a
`CompiledKernel`; (8) no host descriptors. The claim relies on part (B) (the
environment: H200, Triton 3.8.0 as installed, default knobs, the kernel new), which is assumed, not checked.

## Assumptions

One `[rule]` field, `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`. The kernel
uses only operations that are exact up to NaN payload in the library (`tl.mul`, the fusable `tl.addc` /
`tl.subc`, `tl.cvt`, `tl.convert`, integer arithmetic), so no `Num.*` fact is needed. `consistent` uses
`exists_total_faithful Platform.any`.

## Model notes

The model is the source line by line (`model-map.json` maps every line 1–354). Points to note:
- index arithmetic: `i_t`, `i_b`, `bos`, `eos`, `o_t`, `o_cs` are `int64`; `i_h * D`, `o_r`, `o_d`,
  `o_d_swap`, `o_d_repeat` are `int32` (no wrap: `< 2^24`, `≤ 256`); `x + a + b` is added left to right as
  in Python, while `cos + (… + …)` and `y + (… + …)` sum in `int64` first; `i_h * D` is written as the
  source's two multiplications;
- the `chunk_indices` loads are converted with `tl.convert sD .i32` / `.i64` (`.to(tl.int32)` truncates an
  `int64` sequence index; `N < 2^31` keeps it);
- `other=1.0`, `other=0.0` are `float32` constants converted to `bfloat16` (exact);
- both rotations' additions are fed by products, so `tl.subc` / `tl.addc` (IEEE or one `fma` contraction);
  the products are IEEE `tl.mul`; the stores round `float32` to `bfloat16` (`tl.cvt`, nearest even);
- `IS_SEQLEN_OFFSETS_TENSOR` is a `Bool` parameter (both branches modelled): line 105 loads the sequence's
  offset (`seq_offsets + i_n`, the `int32` `i_n` packed, the `int64` `i_b` padded) and promotes it to `int64`
  (`tl.convert`, the identity on `int64`); the source's one argument `seq_offsets` is the model's `I32`
  `seq_offsets` or its pointer `seq_offsets_t`, the path reading only its own;
- the branches of `CONJUGATE` are compile-time and not taken (`config` drops).

## Coverage

35 instances, each with `setup_*_ok : (setup_*).ok` proved by evaluation:
- the host's 22 test cases (lines 323–337), both values of `INTERLEAVED`: padded `(B, T, H, D, rotary_dim)` =
  `(2, 2048, 4, 128, 128)`, `(2, 4096, 4, 256, 256)`, `(1, 1, 1, 64, 64)`, `(3, 1000, 8, 64, 64)` with offset 7,
  `(2, 256, 4, 128, 64)` (partial rotary), `(4, 8192, 32, 128, 128)`; packed `(T, H, D, rotary_dim, N, dtype)` =
  `(2048, 4, 128, 128, 4, int64)`, `(4096, 4, 256, 256, 4, int32)`, `(777, 2, 64, 32, 9, int32)`,
  `(5, 2, 128, 128, 5, int64)`, `(32768, 32, 128, 128, 16, int32)`. `BT`, `BD` and the grid are the host's
  formulas (132 SMs); each packed `NT` is the number of chunks of the host's own `cu_seqlens` (its generator,
  seed 0);
- the other configurations SETUP.md names: decoding (`B = 8`, `T = 1`, `H = 32`, `D = 128`, offset 4095, both
  layouts), head dim 64 with 32 heads, one key head, partial rotary 64 of `D = 256` (both layouts), and one
  packed sequence with `int64` `cu_seqlens`;
- a tensor of offsets (`IS_SEQLEN_OFFSETS_TENSOR = True`), the batched-generation path of fla's attention
  layers (`fla/layers/attn.py:117-119`: a cache and an `attention_mask`, no `cu_seqlens`; `int32` offsets from
  `prepare_lens_from_mask`): the prefill under left padding (`B = 4`, `T = 512`, `H = 32`, `D = 64`,
  `TR = 2048`, offsets `−pad`) and the decoding step (`B = 8`, `T = 1`, offsets the cache's length minus the
  padding), both layouts; `int64` offsets on a padded batch (`B = 2`, `T = 2048`, `D = 128`, `TR = 4096`); and
  packed sequences with `int64` offsets (`T = 777`, `N = 3`).

All flag combinations (padded/packed × GPT-NeoX/GPT-J × `int`/tensor offsets), both index dtypes, both offset
dtypes, negative offsets, partial rotary, and `T` from 1 to 32768 are covered. Every padded batch SETUP.md asks for (`1 ≤ B < 65536`) is inside `Inst.ok` (`grid_b : B < 2^16`); `check_gpu.py` runs the largest, `B = 65535`. `satisfiable` uses the one-token instance with a zero memory.

## Tolerances

`tol = 2^-8·|ref| + 2^-22·S + 2^-132`, `S = |a·cs| + |b·sn|`, composed from the float layer's proved bounds
(`FloatV.mul_err`, `add_err`, `fma_err`, `cvt_err`, `TVA.Lemmas.FloatErr`: one rounding is within `|q|·u + η`,
`u = 2^-24`, `η = 2^-150` for `float32`; `u = 2^-8`, `η = 2^-134` for `bfloat16`):
1. The products `a·cs`, `b·sn` of `bfloat16` values (converted exactly) are `float32` roundings:
   `|p − a·cs| ≤ 2^-24·|a·cs| + 2^-150`.
2. The addition (`AddRel`): IEEE, `r = rn(p + q)`; or fused, `r = rn(p' + q')` with `p'`, `q'` any values that
   round to the products' results, so within one more half-ulp each: `|p' − a·cs| ≤ 2^-23·|a·cs|(1 + 2^-24) +
   2^-149`. Then `|r − ref| ≤ (2^-23 + 2^-24)·S·(1 + 2^-22) + 2^-146 =: e32` in both cases.
3. The store: `|v − r| ≤ 2^-8·|r| + 2^-134 ≤ 2^-8·|ref| + 2^-8·e32 + 2^-134`.
So `|v − ref| ≤ 2^-8·|ref| + (1 + 2^-8)·e32 + 2^-134 ≤ 2^-8·|ref| + 1.51·2^-23·S + 2^-133`, inside `tol`
(`2^-22 = 2·2^-23`). No overflow: `|a| ≤ 2^126`, `|cs|, |sn| ≤ 1` give `S ≤ 2^127`, `|r| ≤ 2^127·(1 + 2^-22)`,
below both formats' largest finite value (`≈ 2^128`), so every step is finite. The tolerance is meaningful under cancellation
(`|ref| ≪ S`): there `S` enters only through `2^-22·S`.
The interleaved path computes the same two products and one fusable addition per channel (selected by
`tl.where`), and a tensor of offsets only changes the table row read, so the same derivation applies. At a
real shape (the left-padded prefill, `B = 4`, `T = 512`, `H = 32`, `D = 64`, `bfloat16` normal inputs and fla's
tables) `tol / |ref|` is `0.0039` at the median and `0.025` at the worst nonzero `ref` (larger only under
cancellation, where `|ref| ≪ S`).

The kernel's real error reaches `0.996` of `tol`: the `bfloat16` rounding's worst case, half an ulp just below
a binade boundary, is `2^-8·|r|/(1 + 2^-8)`, that is `0.9961` of the `2^-8·|ref|` term. The margin is
structural (the term is the one-rounding bound), not a measurement margin.

## Exclusions

Launches SETUP.md allows that `Inst.ok` / `Pre` leave out:
- **`B ≥ 65536` on the padded path: a limit of the kernel.** The grid's axis 1 is `B`, and CUDA allows at most
  65535 there. First launch that hits it: `B = 65536, T = 1, H = 1, D = 64` (an 8 MB `x`): on the H200 the launch
  fails (`Triton Error [CUDA]: invalid argument`, `check_gpu.py`); `B = 65535` runs and passes. SETUP.md asks
  for padded batches `1 ≤ B < 65536` only, so none of its launches reaches this limit;
  `Inst.ok.grid_b` (`B < 2^16`) is exactly SETUP's bound. The packed path (`B = 1`) does not have the limit.
- **`H ≥ 65536`: a limit of the kernel** (grid axis 2). First launch: `H = 65536, B = T = 1, D = 64`. Not
  reached by real use (heads ≤ 128).
- **`NT·B·H ≥ 2^31` programs: a limit of the kernel** (Triton's launcher runs no program). Each program covers
  at least one token of one head, so this needs at least `2^31·D` elements of `x` (≥ 256 GB): not reached.
- **`N ≥ 2^31` sequences: a limit of the kernel** (line 88 truncates the sequence index to `int32`). It needs a
  `cu_seqlens` of more than `2^31` entries (≥ 8 GB): not reached.
- `T ≥ 2^31` or `off ≥ 2^31` (`int64` arguments), `TR·R ≥ 2^62`, `B·T·H·D ≥ 2^62`: outside real use (tensors or
  tables of ≥ 2^31 rows); the kernel's `int64` indexing would still be right below `2^63`.
- A tensor offset below `−2^31` (an `int64` entry): outside real use (an offset is `−pad` at its lowest, with
  fewer than `2^31` padding tokens).
- Inputs: non-finite values in a rotated channel of `x` or in a table row the launch reads, `|x| > 2^126`
  there, or such a table entry above 1 in magnitude (what the kernel does not read is unconstrained). Real
  tables are cosines and sines, and activations are finite and far below `2^126`. Beyond these bounds a rotated
  value can exceed `bfloat16`'s range (`x0 = x1 = 2^127`, a 45° angle), where no tolerance holds; the bounds are
  a sufficient no-overflow condition, not the exact frontier.
- Not SETUP's launches: `conjugate = True` (the backward), the in-place path, a negative `int` offset.

No condition is there only to ease a proof: both values of `isVarlen` and `interleaved`, both metadata dtypes
and inputs up to `2^126` are covered.

## Checks

- **Builds**: `Claim.lean` builds with no errors, no warnings, no
  `sorry`, no `axiom`; `consistent`, `satisfiable` and all 35 `setup_*_ok` depend only on `propext`,
  `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent`, `satisfiable` proved.
- **Coverage**: 35 setup instances, each `ok` proved.
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 63 launches (the 22 host cases with their own inputs, the 13
  other setup configurations, 28 edges: with a tensor of offsets, `BT = 128` with padding across chunks and a
  whole row padded, offsets `−2^31` (`int32` and `int64`) and `TR − len`, packed with empty sequences, 132
  one-token sequences; NaN in `x`'s channels from `2R` on and in the unread table rows, `B = 65535`, `R = 1` (`BD = 2`), `R = 3` (`BD = 8 > 2R`), offset `2^20`
  with `BT = 128`, packed with empty sequences, 132 one-token sequences (`BT = 1`), `|x| = 2^126` with unit
  tables, subnormal inputs, exact cancellation; each in both layouts). All pass: worst `|y − ref|/tol = 0.9960`
  (margin 0.0040, the structural one above); every rotated value at a non-negative position finite; the frame holds (inputs unchanged,
  `y`'s channels from `2R` on equal `x`'s, the left-padding tokens' rotated channels and the guard bytes around
  `y` unchanged; with the host's own `torch.zeros_like` those tokens' channels are `0`). Five wrong outputs
  fail `Post`: the kernel run with the tensor offsets ignored, with `conjugate = True`, with offset `1` for
  `0`, all-zero rotated channels, and the left-padding tokens written. `B = 65536` fails to launch, as stated
  in Exclusions.

## Library requests

None.
