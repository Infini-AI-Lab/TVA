# `slot_mappings_i64`

**The kernel.** `_compute_slot_mappings_kernel` of vLLM's GPU model runner
(`source/block_table.py`, kernel lines 73–154 with its helper `_load_ptr`, lines 66–70; host
`BlockTables.compute_slot_mappings`, lines 233–264). A grid of `(num_groups, num_reqs + 1)`
programs: program `(g, b)` with `b < num_reqs` computes, for KV-cache group `g`, the slot of every
token of request `b` (`query_start_loc[b] ≤ t < query_start_loc[b + 1]`) and stores it in
`slot_mappings[g, t]` (`int64`); program `(g, num_reqs)` stores `-1` (`PAD_SLOT_ID`) from
`query_start_loc[num_reqs]` to the end of row `g`. The block tables are read through their
addresses (`block_table_ptrs`, `uint64`). Both settings of the constexpr `CP_SIZE` are covered:
`CP_SIZE = 1` (no context parallelism) and `CP_SIZE > 1` (decode context parallelism, with any
`CP_INTERLEAVE` and rank). The claim is `Claims.SlotMappings.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and, with
`B` the group's block size, `p = positions[t]` and `p'` its local position (`cpPos`: `p` itself
without context parallelism or for an unsharded group; otherwise, with `V = B·CP_SIZE` and
`I = CP_INTERLEAVE`, `(p / V)·B + ((p % V) / (I·CP_SIZE))·I + (p % V) % I`):
- for every group `g`, request `b` and token `t` of `b`, element `(g, t)` (at
  `g·stride + t`) holds `block_table[g][idx_mapping[b], p' / B] · B + p' % B` when the group uses
  slot mappings (`slot_mapping_enabled[g]`), the request is real (`idx_mapping[b] ≥ 0`) and the
  token is the rank's (`cpLocal`: always without context parallelism or for an unsharded group;
  otherwise `(p % V) / I % CP_SIZE = cp_rank`), and `-1` otherwise (`slot`);
- every element `(g, t)` with `query_start_loc[num_reqs] ≤ t < max_num_tokens` holds `-1`;
- every other byte is unchanged (the inputs, the elements before `query_start_loc[0]`, the columns
  of a wider row past `max_num_tokens`).

It is exact: integers only, no tolerance.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ num_groups < 2^31`, `num_reqs ≤ 8192`, fewer than `2^31` programs;
- `max_num_tokens ≤ 2^22`, at most the output's row stride; the stride below `2^31` with
  `(num_groups − 1)·stride < 2^31` (a limit of the kernel, see Exclusions); the output holds
  `(num_groups − 1)·stride + max_num_tokens` elements;
- `CP_SIZE ≥ 1`, `CP_INTERLEAVE ≥ 1`, `CP_INTERLEAVE·CP_SIZE < 2^31`, `cp_rank < 2^31`;
- every block size at least `1`, with `block_size·CP_SIZE < 2^31` (a limit of the kernel); every
  row stride below `2^63`;
- the per-group tensors hold each group's block-table address (a multiple of 16 bytes, which the
  kernel's `tl.multiple_of` promises), row stride, block size and flags; `idx_mapping` holds
  `num_reqs` entries, `query_start_loc` `num_reqs + 1`; all buffers valid; the output shares no
  byte with any input;
- `query_start_loc` is nondecreasing over the batch from a nonnegative start, its last entry at
  most `max_num_tokens` and the number of positions;
- a real request with at least one token has a row in each group's block table
  (`idx_mapping[b]·stride_g` below the table's length: the kernel reads it even for a token
  without a slot) (`RowOk`);
- for each token of a real request in a group that uses slot mappings: its position is
  nonnegative and, when the token gets a slot, its block `idx_mapping[b]·stride_g + p' / B` lies
  in the block table and `block_number·B` fits `int32` (a limit of the kernel) (`TokOk`).

Dummy requests (`idx_mapping[b] < 0`) may have any positions; block-table entries the kernel does
not turn into a slot are arbitrary; the output's initial contents are arbitrary.

**The launches covered** (`SETUP.md`; `setup_*_ok`, 14 instances): vLLM's model runner with one
attention group and with a hybrid model's three groups (one without slot mappings), decode context
parallelism (2 ranks at rank 0 and 1, 8 ranks with interleave 64 and an unsharded group, 4 ranks
with interleave 16), a dummy batch and a batch with dummy requests, a speculator's wider output,
the source's own test at `CP_SIZE` 1, 2 and 4, and the real ranges' edge (8 groups, 8192
requests, `2^22` tokens). `mk_ok` covers the host's layout for every shape within `Inst.ok`; the
batch's token ranges, positions and block tables are `Pre`'s.

**Device facts.** None: the kernel has no float operation.

## Exclusions

- **`block_number·block_size ≥ 2^31` (or below `-2^31`): a limit of the kernel.** Line 149
  multiplies the `int32` block number by the `int32` block size in `int32`, which wraps: a slot
  past `2^31 − 1` comes out negative. A KV-cache group with `2^31` slots holds the keys and values
  of `2^31` tokens in each of its layers: 512 GiB for one key-value head of 64 `bfloat16` values,
  more than one GPU's memory.
- **`block_size·CP_SIZE ≥ 2^31`: a limit of the kernel.** Line 131's virtual block size is an
  `int32` product.
- **`(num_groups − 1)·stride ≥ 2^31`, or a row stride or `max_num_tokens` of `2^31` or more: a
  limit of the kernel.** Line 95's row offset is an `int32` product (and a runtime integer from
  `2^31` on is an `int64`, another kernel).
- **More than 8192 requests, more than `2^22` tokens, `CP_INTERLEAVE·CP_SIZE ≥ 2^31`**: the
  claim covers real batch sizes, token counts and context-parallel sizes only.
- **A decreasing `query_start_loc`, a negative position of a real request, a real request without
  a row in a block table, an output overlapping an input**: outside the host's launches (its
  token ranges are running sums, its positions token positions, its tables have a row per request
  and the output is a separate allocation).
- **`_load_ptr` from another file.** vLLM imports the helper from `buffer_utils.py`; helpers in
  another file are outside the language TVA covers, so `source/` holds it in the kernel's file,
  verbatim. The host's other kernel in the same vLLM file (`_gather_block_tables_kernel`) is not
  part of this claim.

`check_gpu.py` (one H200) checks `Post` exactly on the `SETUP.md` launches and on edges of
`Inst.ok` and `Pre`.
