# `prepare_pos_seq_lens_i32`

**The kernel.** vLLM's `_prepare_pos_seq_lens_kernel` (`source/prepare_pos_seq_lens.py`, kernel
lines 54–87; host `prepare_pos_seq_lens` lines 90–106). A 1-D grid of `num_reqs + 1` programs.
Program `r < num_reqs` serves request `r`: it reads the request's state slot `idx_mapping[r]`
(`int64`), the slot's `num_computed_tokens` (`int32`) and the request's `query_start_loc[r]`,
`query_start_loc[r + 1]` (`int32`), stores `seq_lens[r]` (`int32`), and stores the positions of
the request's tokens to `pos` (`int64`) in blocks of `1024`. The last program
(`program_id == num_programs − 1`) stores `0` to `seq_lens[num_reqs .. max_num_reqs)`, also in
blocks of `1024`, where `max_num_reqs` is the host's `seq_lens.shape[0]`. The claim is
`Claims.PreparePosSeqLens.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and, with
`c_r = num_computed_tokens[idx_mapping[r]]`, `s_r = query_start_loc[r]` and
`q_r = query_start_loc[r + 1] − s_r` (all read from the initial memory):
- for every request `r < num_reqs`, `seq_lens[r]` holds the `int32` value `c_r + q_r`;
- for every request `r < num_reqs` and `j < q_r`, `pos[s_r + j]` holds the `int64` value
  `c_r + j`;
- for every `num_reqs ≤ r < max_num_reqs`, `seq_lens[r]` holds `0`;
- every byte outside those elements of `seq_lens` and `pos` is unchanged (the inputs, the other
  elements of both outputs, and all other memory).

It is exact: no float operation, no tolerance, no device fact. The `int32` sums do not wrap on
the covered inputs (`c_r + q_r ≤ 2^23`), and the `int32` positions are sign-extended to `int64`
on the store, so each position is its exact value.

The programs write disjoint elements: request `r` writes `seq_lens[r]` (`r < num_reqs`) and
`pos[s_r .. s_r + q_r)`, the last program `seq_lens[num_reqs .. max_num_reqs)`; the requests'
windows of `pos` are disjoint because `query_start_loc` is nondecreasing. No program reads what
any program writes (the outputs share no byte with the inputs), so the kernel is not an in-place
update.

**The conditions** (`Inst.ok`, `Pre`):
- `num_reqs + 1 < 2^31` (the grid) and `max_num_reqs + 1024 ≤ 2^31` (the `int32` bounds of the
  zeroing loop): limits of the kernel;
- `idx_mapping` has `num_reqs` elements, `query_start_loc` at least `num_reqs + 1`, `seq_lens`
  at least `num_reqs` and at least `max_num_reqs`; all five buffers are valid in the initial
  memory;
- `pos` and `seq_lens` share no byte with each other or with an input;
- `query_start_loc[0 .. num_reqs]` is nondecreasing from a nonnegative start, and its last entry
  is at most `pos`'s length and at most `2^22` (the batch's tokens);
- every request's slot is an element of `num_computed_tokens`, and its value there is in
  `[0, 2^22]` (a context length).

The inputs may overlap one another and slots may repeat; the other entries of
`num_computed_tokens`, the entries of `query_start_loc` past `num_reqs`, and the initial contents
of the outputs are arbitrary.

**The launches covered** (`SETUP.md`): the host's test cases (decode, a prefill with lengths
around the block size, a full batch, one request with `max_num_reqs = 1`, a long zeroing loop,
speculative decoding, a serving batch, a `2^20`-token prefill) and large serving sizes (8192
requests, `2^22` positions), each a `setup_*_ok` theorem. `mk_ok` covers the host's layout for
every `num_reqs ≤ max_num_reqs` within the kernel's limits; the requests' lengths and computed
tokens are `Pre`'s.

## Exclusions

- **More than `2^22` tokens in a batch, or more than `2^22` computed tokens in a request**: the
  claim covers real model sizes only (the kernel itself is exact while `c_r + q_r < 2^31` and
  `q_r + 1023 < 2^31`).
- **`max_num_reqs + 1024 > 2^31`, `num_reqs + 1 ≥ 2^31`**: limits of the kernel (the zeroing
  loop's `int32` bound; the grid); also far outside real use.
- **A `query_start_loc` that decreases within the batch, a negative start, or a window past the
  end of `pos`**: the windows of two requests would overlap (a race) or leave `pos`; the host's
  running sums from `0` never do.
- **A slot outside `num_computed_tokens`**: an access outside the buffer.
- **An output overlapping an input or the other output**: outside real use (vLLM's separate
  persistent buffers).

`check_gpu.py` (one H200) checks `Post` on the setup's launches and on edges of the conditions.
