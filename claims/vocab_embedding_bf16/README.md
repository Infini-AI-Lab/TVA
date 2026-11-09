# Prepared claim: `vocab_embedding_bf16`

Kernel: `source/vocab_parallel_embedding.py`, SGLang's
`_vocab_parallel_embedding_kernel` (lines 54-99), launched by `vocab_parallel_embedding` (lines
102-143). Claim: `Claim.lean`, namespace `Claims.VocabEmbedding`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. The model map is `model-map.json`.

One program per (token, block of `BLOCK_H` columns): the program loads the token's id, maps it to
the shard's local row (original range: `id - org_vocab_start_index`; added range:
`id - added_offset`), and copies that row's columns to the output, or stores zeros when the id is
in neither range. The model writes the addresses in the source's order,
`(weight_ptr + local_id * stride) + cols` and `(out_ptr + row * hidden_dim) + cols`, with `valid`
an `int1` tensor.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.VocabEmbedding.main : Statement` (`Statement`:
`Claim.lean` line 243). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 133), and every initial memory `init` that `Pre` accepts (line 183), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 210).

**What `Post` asks** (line 210), two parts:
1. for every `r < n_tokens` and `col < hidden_dim`: element `r` of `input_` reads in `init` as an id
   `v` (`tokAt`, line 126: the signed value `ivOf`, line 119), element `r * hidden_dim + col` of
   `output` reads in `fin` as some `c`, and `outVal d init v col c` holds (line 196):
   - if `org_vocab_start_index ≤ v < org_vocab_end_index`: `init` reads the weight's element
     `(v - org_vocab_start_index) * weight_stride0 + col` as `c`;
   - else, if `added_vocab_start_index ≤ v < added_vocab_end_index`: `init` reads the weight's
     element `(v - addedOffset d) * weight_stride0 + col` as `c` (`addedOffset`, line 113);
   - else `c` is the `bfloat16` `+0` (`FloatV.ofPat 0`).

   It is bit-exact: a copy, no tolerance, no NaN equivalence;
2. the frame: every byte outside the output's first `n_tokens * hidden_dim` elements
   (`outElems`, line 190) is unchanged.

**What you are given** (`Assumptions`, line 234): one field, `faithful` (`Compiler.Faithful`): the
device's outcomes `K d` are outcomes of the model's launch, when the launch is valid, has
`debug = false`, and the model's launch is well behaved from `init` (no out-of-bounds access, no race).
No numeric fact: the kernel only moves values (the one conversion is the `float32` `+0` to
`bfloat16`, exact).

**The kernel, as the model runs it** (`vocab_parallel_embedding_kernel`, line 31; `model` line 221,
`launch` line 229): a 2-D grid `(n_tokens, nColBlocks d, 1)` (`nColBlocks`, line 130), block shape
`[BLOCK_H]`. Program `(row, col_block)` loads one id (`tl.load1` of `input_` element `row`), converts
it to `int64`, forms the two range masks and `local_id` (`int64` subtractions, `tl.where`), then does
one masked `tl.load` of the weight at `local_id * weight_stride0 + cols` (mask
`col_mask & valid`, `other` the converted `+0`) and one masked `tl.store` of output row `row`,
columns `cols = col_block * BLOCK_H + [0, BLOCK_H)` below `hidden_dim`. No loop. Programs write
disjoint column blocks of the output and use no atomics; several may read the same weight row,
which is only read. The route is `TVA.meets_independent` (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`launch` sets no `debug`). The grid is 2-D, so
  `Launch.Valid` is proved directly: `x < 2^31` from `tokens_lt`, `y < 2^16` from `cols_lt`, the
  program count from `grid_lt`, the shape `[BLOCK_H]` from `block` (`Shape.TritonOk`);
- each program is `Safe`: the id load is in bounds (`input_len`, `safe_load1`); `int32` columns
  do not wrap (`hidden_lt`), and `ltS` against `hidden_dim` is the integer comparison; the
  `int64` subtractions and `local_id * weight_stride0 + col` do not wrap (`org_start` … `offset`,
  `stride_lt`, and `TokOk`'s `< 2^63`, through `Pre`'s last conjunct); the enabled weight lanes
  are in bounds (`TokOk`'s `≤ weight.len`; the added range also needs `0 ≤ v - addedOffset d`);
  the store is in bounds (`out_len`), with distinct lanes (`safe_load`, `safe_store`,
  `safe_convert`, `safe_cvt`);
- footprints: program `(row, col_block)` writes output elements `row * hidden_dim + col` for its
  columns; these are disjoint across programs, and disjoint from every read (`out_input`,
  `out_weight`);
- the stored lanes give part 1: an enabled lane holds the weight's element as `init` reads it; a
  lane disabled by `valid` holds `other`, the `bfloat16` `+0`. The `tl.where` order matches
  `outVal`'s: the original range wins;
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `bytes_disjoint_of_le` (line 252) and `zeroMem_read` (line 275) can be
reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`. `Post d init fin`, for the id `v` at element `r` of
`input_` (its signed value, `int64` or `int32`):

- element `r * hidden_dim + col` of the output (`r < n_tokens`, `col < hidden_dim`) is, bit for
  bit,
  - the weight's storage element `(v - org_vocab_start_index) * weight_stride0 + col` if
    `org_vocab_start_index ≤ v < org_vocab_end_index`;
  - else the element `(v - added_offset) * weight_stride0 + col` if
    `added_vocab_start_index ≤ v < added_vocab_end_index`, with
    `added_offset = added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) -
    num_org_vocab_padding`;
  - else the `bfloat16` `+0` (pattern `0x0000`);
- every byte outside the output's first `n_tokens * hidden_dim` elements is unchanged.

The reference is computed from the host's inputs (the ids, the weight as a `[rows, hidden_dim]`
view with row stride `weight.stride(0)`, the shard's ranges), as the host's `reference` does
(`weight[local]`, zero rows outside the shard). Exact: a copy, no arithmetic on the values; NaN
payloads, infinities and `-0` are copied as they are.

## Conditions

`Inst`: the ids' dtype `dI`, `n_tokens = input_.numel()`, `hidden_dim`, `BLOCK_H`,
`weight_stride0`, the five vocabulary values, and the buffers `input_` (contiguous ids), `weight`
(its storage from its first element) and `output` (contiguous `bfloat16`).

`Inst.ok`:
- `dI` is `int64` (SGLang's `input_ids`) or `int32` (the host asserts one of the two; the test
  passes both);
- `BLOCK_H` a power of two of at most 1024 (the host's `min(1024, next_power_of_2(hidden_dim))`
  is one; any other is covered too);
- `hidden_dim + BLOCK_H ≤ 2^31`: the `int32` columns `col_block * BLOCK_H + l` do not wrap
  (a limit of the kernel, Exclusions);
- the grid `(n_tokens, cdiv(hidden_dim, BLOCK_H))` within CUDA's limits (`n_tokens < 2^31`,
  `cdiv < 2^16`) with fewer than `2^31` programs (`Launch.Valid`; Exclusions);
- `n_tokens ≤ input_.len`, `n_tokens * hidden_dim ≤ output.len`;
- the five vocabulary values in `[0, 2^63)` and `added_offset` in `[-2^63, 2^63)` (a rank's
  indices and counts; each constexpr is then an `int32` or `int64` constant and the model's `int64`
  constants are exact); `weight_stride0 < 2^63`;
- the output shares no byte with `input_` or `weight` (`torch.empty`).

`hidden_dim = 0` and `weight_stride0 = 0` are covered (the kernel is right at both: an empty grid,
or rows that alias, which are only read).

`Pre d init`: the three buffers valid; every id the launch reads that lies in a range names a
local row whose `hidden_dim` columns are inside the weight's storage (below `2^63`): the shard's
ids map to rows of the weight (SETUP.md). Ids outside both ranges are unconstrained (`-1`, other
ranks' ids, ids past the vocabulary, `2^30`, the `int64` extremes), and so are the weight's and the
output's values.

`launch d`: grid `(n_tokens, cdiv(hidden_dim, BLOCK_H), 1)`, block shape `[BLOCK_H]`,
`enable_fp_fusion` Triton's default; `num_warps = 8`, `num_stages = 3` change no result;
`num_ctas = 1`, no `launch_pdl`, `debug = false`.

`Compiler.Faithful`'s scope, part (A), checked against the source: (4) the launch passes only
`BLOCK_H`, `num_warps`, `num_stages`; no `@triton.heuristics` or `@triton.autotune`; (5) no
descriptor; (6) no `CompiledKernel`; (8) no host `TensorDescriptor`. Part (B),
the environment, is assumed and not checked.

## Assumptions

One `[rule]` field, `faithful` (`Compiler.Faithful`, as the library states it). No approximate
operation, no `Rules.Num` fact (`Platform.any`), no `[kernel]` field.

## Coverage

Each launch of the host's test (source lines 199-226; ranges from `shard_indices`, weight
contiguous) has an instance with `Inst.ok` proved:

| Launch | `dI`, ids, hidden, `BLOCK_H`, rows, ranges | Instance |
|---|---|---|
| vocab 128256, TP 8 rank 3 | `int64`, 8192, 4096, 1024, 16032, org `[48096, 64128)` | `setup_v128256_tp8` |
| vocab 129280, TP 8 rank 7 | `int64`, 1, 7168, 1024, 16160, org `[113120, 129280)` | `setup_v129280_tp8` |
| vocab 151936, TP 4 rank 0, ids `(3, 77)` | `int64`, 231, 2048, 1024, 37984, org `[0, 37984)` | `setup_v151936_tp4` |
| vocab 152064, TP 2 rank 1 | `int32`, 513, 5120, 1024, 76032, org `[76032, 152064)` | `setup_v152064_tp2_i32` |
| vocab 32000 + 256, TP 2 rank 1 | `int64`, 1000, 257, 512, 16128, org `[16000, 32000)`, added `[32128, 32256)` | `setup_v32000_added_tp2` |
| vocab 50257 + 3, TP 8 rank 7 | `int64`, 300, 6144, 1024, 6296, org `[44016, 50257)`, 47 padding rows | `setup_v50257_added_tp8` |
| vocab 1000, TP 2 rank 0 | `int32`, 64, 7, 8, 512, org `[0, 512)` | `setup_v1000_tp2_i32` |
| empty input `(0,)` | no launch (the host returns first) | (none needed) |
| a column slice (row stride above `hidden_dim`, the code base's test) | `int64`, 64, 128, 128, stride 256 | `setup_column_slice` |
| serving: hidden 8192, vocab 151936 over TP 2 | `int64`, 65536, 8192, 1024, 75968 | `setup_serving_h8192` |

`Inst.ok` bounds no size beyond the `int32` column limit and the grid, so every vocabulary,
tensor-parallel split, hidden size (7 to 8192, any value) and number of ids SETUP.md names is
covered, with ids of any value.

## Tolerances

None: `Post` is bit-exact.

## Exclusions

- **`hidden_dim + BLOCK_H > 2^31`: a limit of the kernel.** `cols = col_block * BLOCK_H +
  arange` is `int32` and wraps; the first launch that hits it has `hidden_dim > 2^31 - 1024`
  (one weight row of 4 GiB). Real use (hidden sizes up to 8192, `2^13`) never reaches it.
- **Grids beyond CUDA's limits or with `2^31` programs or more** (`n_tokens ≥ 2^31`, or
  `n_tokens * cdiv(hidden_dim, BLOCK_H) ≥ 2^31`): CUDA rejects the first, and Triton 3.8.0's
  launcher runs no program for the second without reporting an error, so the output would be
  left unwritten; a limit of the launch, not of the kernel's code. The first launch that hits it
  at hidden 8192 (8 column blocks) has `2^28` ids (a 4 TiB output); real use never reaches it.
- **Vocabulary values outside `[0, 2^63)`, `added_offset` outside `[-2^63, 2^63)`,
  `weight_stride0 ≥ 2^63`**: outside real use (a rank's indices, with `org_vocab_start_index ≤
  org_vocab_end_index`, so `added_offset ≤ added_vocab_start_index`); at such values a constexpr
  would be a `uint64` constant (`added_offset` reaches `[2^63, 2^64)` only when
  `org_vocab_end_index < org_vocab_start_index`) or rejected on the host.
- **An id of a range whose local row is not in the weight** (`Pre`): outside real use (the layer
  allocates the rows of its ranges); the kernel would read outside the weight.
- **An output that overlaps an input**: outside real use (fresh allocation).

## Library requests

None.

## Checks

- **Builds**: `Claim.lean` builds with no error or warning, no
  `sorry`, no `axiom`; the theorems use only `propext`, `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent` (`Platform.any`); `satisfiable` (one `int64` id `0`, hidden 7, on
  the all-zero memory).
- **Coverage**: the 9 instances above (`setup_*_ok`).
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 42 host calls: the 8 setup cases (same
  generator and ids) and edges: every rank of 8 shardings (with added vocabularies and padding
  rows, row strides above `hidden_dim`, hidden 1, 7, 257, 1000, 1025, 6144, 7168, 8192, `int32`
  and `int64` ids), each with every boundary id (`start - 1`, `start`, `end - 1`, `end` of both
  ranges, `-1`, `2^30`, `±2^31`, the `int64` extremes) and a weight of random bit patterns (NaN
  payloads included); 32768 ids at hidden 8192; overlapping ranges (the original range wins); two
  direct launches with the host's launch line on an output inside a random-filled guarded pool.
  377,579,531 output elements: every one bit-exact as `Post` states (error 0, no tolerance);
  every input and every guard unchanged.
