# `compute_position_i64`

**The kernel.** SGLang's `compute_position_kernel` (`source/position.py`, kernel lines 64–91;
host `compute_position_triton` lines 38–61). A 1-D grid of `bs` programs, one per request of an
extend batch; program `b` reads its prefix `extend_prefix_lens[b]` (when the host passes one per
request, `has_prefix`) and its length `extend_seq_lens[b]`, sums the lengths of the requests
before it in a loop (`start(b)`, an `int64`), writes the `int64` positions
`p(b), p(b) + 1, …, p(b) + extend_seq_lens[b] − 1` to `positions[start(b) …]` in blocks of `512`,
and stores `start(b)` in `extend_start_loc[b]` (`int32`). The claim is
`Claims.ComputePosition.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and, with
`start(b) = extend_seq_lens[0] + … + extend_seq_lens[b − 1]` and `p(b)` the prefix
`extend_prefix_lens[b]` (`0` without prefixes):
- for every request `b < bs` and position `j < extend_seq_lens[b]`, element `start(b) + j` of
  `positions` holds the `int64` value `p(b) + j`;
- for every request `b < bs`, element `b` of `extend_start_loc` holds the `int32` value
  `start(b)`;
- every byte outside the first `start(bs)` elements of `positions` and the first `bs` elements of
  `extend_start_loc` is unchanged.

It is exact: no float operation, no tolerance, no device fact.

**The conditions** (`Inst.ok`, `Pre`):
- `extend_prefix_lens` and `extend_seq_lens` are `int32` (every caller's);
- `bs < 2^31` (the grid); `extend_seq_lens` and `extend_start_loc` have `bs` elements, and
  `extend_prefix_lens` too with prefixes; all buffers are valid in the initial memory;
- `positions` and `extend_start_loc` share no byte with each other or with an input;
- every length is nonnegative with `extend_seq_lens[b] + 511 < 2^31`; with prefixes,
  `p(b) + extend_seq_lens[b] ≤ 2^31`; every `start(b) < 2^31` (`ReqOk`);
- `positions` holds at least `start(bs)` elements (the host's `extend_seq_lens_sum`).

Any `int32` prefix within that bound is covered, negative ones included, and any initial contents
of the outputs.

**The launches covered** (`SETUP.md`): the code base's test (`bs` 1, 3 and 129), prefill batches
(64 requests in an 8192-token budget, one request of `2^20` tokens, 4096 requests), speculative
draft extends (256 requests of 8 tokens), the host without prefixes, and an empty batch
(`setup_*_ok`). `mk_ok` covers the host's layout for every `bs < 2^31` with or without prefixes;
the lengths and prefixes are `Pre`'s.

**Device facts.** None: the kernel has no float operation.

## Exclusions

- **`extend_seq_lens[b] + 511 ≥ 2^31`: a limit of the kernel.** The `int32` `tl.cdiv` wraps to a
  negative trip count and the program writes none of its positions.
- **With prefixes, `p(b) + extend_seq_lens[b] > 2^31`: a limit of the kernel.** The `int32` sum
  `prefix_len + offset` wraps before it is widened to `int64`.
- **`start(b) ≥ 2^31`: a limit of the kernel.** The `int32` `extend_start_loc[b]` wraps. Such a
  batch has at least `2^31` tokens (a `positions` of 16 GiB).
- **`int64` lengths or prefixes**: no caller passes them (`ForwardBatch.init_new`, the speculative
  workers and the code base's test build both as `int32`); the model reads `int32` elements.
- **Negative lengths, a `positions` shorter than the sum of the lengths, an output overlapping an
  input**: outside the host's use (lengths of new tokens, `extend_seq_lens_sum` their sum, fresh
  `torch.empty` outputs).
- **`bs ≥ 2^31`**: beyond CUDA's grid.

`check_gpu.py` (one H200) checks `Post` exactly on the `SETUP.md` launches and on edges of
`ReqOk` (a prefix with `p(b) + extend_seq_lens[b] = 2^31`, negative prefixes, 8192 requests,
lengths at the block edges, direct launches inside guarded buffers). It cannot reach
`start(b)` near `2^31` within 4 GB.
