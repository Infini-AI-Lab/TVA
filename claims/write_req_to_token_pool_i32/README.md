# `write_req_to_token_pool_i32`

**The kernel.** SGLang's `write_req_to_token_pool_triton` (`source/write_req_to_token_pool.py`,
kernel lines 56–104; host `write_cache_indices` lines 107–136), which the scheduler runs when it
allocates an extend (prefill) batch. A 1-D grid of `bs` programs, one per request; program `b`
reads its row `req_pool_indices[b]`, its prefix length `prefix_lens[b]`, its length `seq_lens[b]`
and the data pointer of its prefix tensor (from the `uint64` tensor `prefix_pointers`), then, in
blocks of `512`:
- copies its prefix tensor's first `prefix_lens[b]` `int64` locations to positions
  `0 … prefix_lens[b] − 1` of its row of the `int32` table `req_to_token`;
- sums `extend_lens[0 … b − 1]` into `c[b]` (an `int64` loop over the preceding requests), and
  copies `out_cache_loc[c[b] …]` to positions `prefix_lens[b] … seq_lens[b] − 1` of its row.

Each `int64` location is stored as an `int32` (`tl.store` converts: the low 32 bits). The row
stride `max_context_len` is a compile-time argument. The claim is
`Claims.WriteReqToTokenPool.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and, with
`r[b] = req_pool_indices[b] · max_context_len` (the row's first element of the contiguous table):
- for every request `b < bs` and `j < prefix_lens[b]`, element `r[b] + j` of the table holds the
  `int32` cast of the initial `prefix_b[j]`;
- for every request `b < bs` and `prefix_lens[b] ≤ j < seq_lens[b]`, element `r[b] + j` holds the
  `int32` cast of the initial `out_cache_loc[c[b] + j − prefix_lens[b]]`;
- every byte outside those elements is unchanged: the rest of the table, and every input.

It is exact: no float operation, no tolerance, no device fact. Any `int64` locations are covered
(the cast keeps the low 32 bits, exactly as stated).

**The conditions** (`Inst.ok`, `Pre`):
- `bs < 2^31`, `bs ≤ 8192`; `max_context_len ≤ 2^22`;
- `req_pool_indices`, `prefix_pointers`, `prefix_lens`, `seq_lens`, `extend_lens` have at least `bs`
  elements; all buffers are valid in the initial memory;
- the table shares no byte with any input buffer, nor with any request's prefix tensor;
- every request has `0 ≤ prefix_lens[b] ≤ seq_lens[b] ≤ max_context_len`, a nonnegative row, and
  its row's first `seq_lens[b]` elements in the table's storage (`ReqOk`);
- a request with a prefix (`prefix_lens[b] > 0`) has its pointer `prefix_pointers[b]` at its prefix
  tensor, which is valid and holds at least `prefix_lens[b]` elements (`PrefixOk`); the pointer of
  a request without a prefix is never followed, and any value is covered;
- every `extend_lens[b]` is in `[0, 2^22]`, and each request's window
  `c[b] … c[b] + seq_lens[b] − prefix_lens[b] − 1` lies in `out_cache_loc`; `extend_lens` need not
  equal `seq_lens − prefix_lens` (the kernel reads `c[b]` from `extend_lens` only);
- two requests that write (`seq_lens[b] > 0`) have distinct rows.

`Inst` gives each request's prefix tensor as a buffer (`prefixT b`); for a request without a
prefix it may be any buffer (for instance an empty one), and `Post` says nothing about it.

**The launches covered** (`SETUP.md`): the host test's five cases and four serving shapes
(`setup_*_ok`): pools of 64, 256 and 4096 requests at a 4096-token context, a 32768-token context,
a 131072-token context, a chunked prefill over 1024 requests of a 128K-context pool, a 1M-token
context, 8192 requests, and the row length of speculative decoding. `mk_ok` covers the host's
layout for every batch and table shape; the lengths, rows and pointers are `Pre`'s.

## Exclusions

- **`bs > 8192`, `max_context_len > 2^22`, an `extend_lens[b]` above `2^22`**: the claim covers
  real batch and context sizes only (they also keep every `int64` index far from wrapping).
- **`seq_lens[b] > max_context_len`**: the window would run into the next row; a request's tokens
  fit its row.
- **`prefix_lens[b] > seq_lens[b]`, negative lengths**: not made by the host.
- **Two writing requests on one row**: both programs write the row's first elements, a race; the
  scheduler gives each request its own row.
- **A table overlapping an input or a prefix tensor**: outside real use (separate allocations; a
  request's own row is copied to a new `int64` tensor before it is passed as its prefix).
- **`bs ≥ 2^31`**: beyond CUDA's grid.

`check_gpu.py` (one H200) checks `Post` on the setup's launches.
