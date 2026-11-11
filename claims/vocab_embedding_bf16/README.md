# `vocab_embedding_bf16`

**The kernel.** SGLang's `_vocab_parallel_embedding_kernel` (`source/vocab_parallel_embedding.py`,
kernel lines 54-99; host `vocab_parallel_embedding` lines 102-143). A grid `(n_tokens,
cdiv(hidden_dim, BLOCK_H))`; program `(r, b)` reads token `r`'s id, maps it to the shard's local
row (original range or added range, else no row), and copies columns `b·BLOCK_H + [0, BLOCK_H)` of
that weight row (or zeros) into output row `r`. The claim is `Claims.VocabEmbedding.claim` in
`Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for the
id `v` at element `r` of `input_`, element `r·hidden_dim + col` of the output is, bit for bit:
- the weight's storage element `(v − org_vocab_start_index)·weight_stride0 + col` if
  `org_vocab_start_index ≤ v < org_vocab_end_index`;
- else the element `(v − added_offset)·weight_stride0 + col` if
  `added_vocab_start_index ≤ v < added_vocab_end_index`, with `added_offset =
  added_vocab_start_index − (org_vocab_end_index − org_vocab_start_index) − num_org_vocab_padding`;
- else the `bfloat16` `+0`.

Every byte outside the output's first `n_tokens·hidden_dim` elements is unchanged. The lookup is a
copy: no tolerance.

**The conditions** (`Inst.ok`, `Pre`): ids `int64` or `int32`; `BLOCK_H` a power of two at most
`1024`; `hidden_dim + BLOCK_H ≤ 2^31`; `n_tokens < 2^31`, `cdiv(hidden_dim, BLOCK_H) < 2^16`,
fewer than `2^31` programs; `hidden_dim` a positive multiple of `128`, at most `2^17`;
`n_tokens ≤ 2^22`; the buffers hold the ids and the output rows; the five vocabulary values in
`[0, 2^63)`, `added_offset` in `[-2^63, 2^63)`, `weight_stride0 < 2^63`; the output shares no byte
with the ids or the weight; the three buffers are valid; an id the launch reads that lies in a
range names a local row whose `hidden_dim` columns are inside the weight (`TokOk`). Ids outside
both ranges and every weight value are covered; `weight_stride0 = 0` is covered.

## Coverage

The launches of `SETUP.md` (`setup_*_ok`): the host's test cases (source lines 199-226) whose
`hidden_dim` is a model dimension (vocabularies 128256, 129280, 151936, 152064 with `int32` ids,
50257 + 3 added), a column slice (row stride 256, hidden 128), and serving at hidden 8192 with
65536 ids. The test cases with hidden 257 and 7 are not covered (not multiples of `128`); the empty
input makes no launch. `mk_ok` covers the host's layout at every shape `Inst.ok` allows.

## Exclusions

- `hidden_dim + BLOCK_H > 2^31`: a kernel limit (the `int32` columns wrap).
- Grids beyond CUDA's limits or with `2^31` programs or more: a launch limit.
- Vocabulary values outside `[0, 2^63)`, `added_offset` outside `[-2^63, 2^63)`,
  `weight_stride0 ≥ 2^63`: outside real use (a constexpr would be a `uint64` constant).
- An id of a range whose local row is not in the weight: outside real use (`Pre`).
- An output that overlaps an input: outside real use (fresh allocation).
- `hidden_dim` not a positive multiple of `128` or above `2^17`, and `n_tokens > 2^22`: the claim
  covers real model shapes only (this leaves out the test sizes hidden 7 and 257).

`check_gpu.py` (one H200) checks `Post` bit for bit on the setup's launches and on edge cases.
