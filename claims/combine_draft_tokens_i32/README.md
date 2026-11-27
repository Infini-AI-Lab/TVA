# `combine_draft_tokens_i32`

**The kernel.** vLLM's `_combine_sampled_and_draft_tokens_kernel` (`source/combine_draft_tokens.py`,
kernel lines 60–118; host `combine_sampled_and_draft_tokens` lines 121–159). A 1-D grid of
`num_reqs` programs, one per request, with `BLOCK_SIZE = next_power_of_2(S + N)` (`S` the width of
the draft table, `N = NUM_NEW_SAMPLED_TOKENS`, `0` or `1`). Program `b` reads its request-state row
`i = idx_mapping[b]`, its number of logits `nl = cu_num_logits[b + 1] − cu_num_logits[b]`, its
query end `qe = query_start_loc[b + 1]`, `seq_lens[b]` and `prefill_len[i]`; it writes the
request's `logits_indices`, and, for a request past its prefill, the last sampled token and the
draft tokens into its last query slots of `input_ids`. The claim is
`Claims.CombineDraftTokens.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
request `b < num_reqs`, with `nd = nl − N`:
- `logits_indices[cu_num_logits[b] + j] = qe − nl + j` for every `j < nl` (an `int64`);
- if `seq_lens[b] > prefill_len[i]` (`Decode`):
  - when `N = 1` and `seq_lens[b] − nl ≥ prefill_len[i]` (`LastOn`), `input_ids[qe − nl]` holds the
    `int32` truncation of `last_sampled_tokens[i]` (`rdt64`, the value modulo `2^32`, as the store
    into the `int32` pointer converts);
  - `input_ids[qe − nd + j]` holds the `int32` truncation of `draft_tokens[i · stride + j]` for
    every `j < nd` (`stride = draft_tokens.stride(0)`);
- every other byte (every other `input_ids` slot, every input) is unchanged.

A request still in its prefill (`seq_lens[b] ≤ prefill_len[i]`) writes only its logits indices; a
request whose first logit is a prompt position (`seq_lens[b] − nl < prefill_len[i]`) keeps that
slot. It is exact: integers only, no tolerance, no device fact.

**The conditions** (`Inst.ok`, `Pre`):
- `N ≤ 1` (the host's assert); `BLOCK_SIZE` a power of two of at most `2^20` lanes (Triton's
  limit); `num_reqs < 2^31`; the stride below `2^31`;
- `idx_mapping` and `seq_lens` have `num_reqs` entries, `cu_num_logits` and `query_start_loc`
  `num_reqs + 1`; all nine buffers are valid in the initial memory;
- `input_ids` and `logits_indices` share no byte with each other or with an input (the inputs may
  overlap one another);
- per request (`ReqOk`): `0 ≤ i` below the lengths of `prefill_len` and `last_sampled_tokens`;
  `seq_lens[b] ≥ 0`; `N ≤ nl ≤ BLOCK_SIZE`; `query_start_loc[b] + nl ≤ query_start_loc[b + 1]`
  (the logits are among the request's query rows, the host's assert at `model_runner.py:1437`);
  with `nd > 0`, `i · stride + nd` within `draft_tokens`;
- `cu_num_logits[0] ≥ 0` and `cu_num_logits[num_reqs]` within `logits_indices`;
  `query_start_loc[0] ≥ 0` and `query_start_loc[num_reqs]` within `input_ids`.

Token values (any `int64`), `prefill_len` entries (any `int32`), repeated request-state rows and the
outputs' initial contents are arbitrary. `N ≤ nl` makes both running sums nondecreasing, so the
programs' slots and logits never meet, and every `int32` offset (`qe − nl + j`, `cu_num_logits[b] +
j`, `qe − nd + j`) lies in `[0, 2^31)` on the lanes that access memory.

**The launches covered** (`SETUP.md`): decoding without drafts (`S = 0`, `BLOCK_SIZE = 1`; 1, 256
and 1024 requests), MTP (`S = 1`), EAGLE (`S = 3`, `7`), n-gram (`S = 8`), long drafts
(`S = 128`), `N = 0` (`S = 4`, `7`), a large deployment (8192 requests over a `2^20`-token budget)
and an empty batch (`setup_*_ok`; 12 instances). `mk_ok` covers the host's layout for every batch,
row count, draft width, `N ≤ 1` and `BLOCK_SIZE = 2^e`, `e ≤ 20`; the batch's contents are `Pre`'s.

**Device facts.** None: the kernel has no float operation.

## Exclusions

- **`nl > BLOCK_SIZE`, or `nl < N`**: the host never makes them (`nl = nd + N` with `nd ≤ S`, and
  `BLOCK_SIZE = next_power_of_2(S + N)`). The kernel writes only `BLOCK_SIZE` logits indices and
  drafts, and with `N = 1`, `nl = 0` its last sampled token may land on the next request's first
  slot.
- **Logits outside the query rows** (`nl > query_start_loc[b + 1] − query_start_loc[b]`): the host
  asserts against it; such a launch writes another request's slots.
- **`S = 0` with `N = 0`**: `BLOCK_SIZE = next_power_of_2(0) = 0`, and `tl.arange(0, 0)` is rejected
  on the host, so the launch has no outcome.
- **A negative `seq_lens` entry**, **outputs overlapping each other or an input**: outside real
  use (`seq_lens` holds lengths; `logits_indices` is a fresh `torch.empty`, `input_ids` its own
  buffer).
- **The kernel parameter's default** `NUM_NEW_SAMPLED_TOKENS = 1`: outside TVA's subset; the source
  drops it, and the host passes the parameter on every launch.
- **`prefill_len`** lives in host memory mapped for the device (`UvaBackedTensor`); like every
  input, nothing may write it while the launch runs.

`check_gpu.py` (one H200) checks `Post` exactly on the setup's launches and on edges of `Inst.ok`
and `Pre`, inside guarded buffers.
