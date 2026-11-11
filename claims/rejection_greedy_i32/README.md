# `rejection_greedy_i32`

**The kernel.** `rejection_greedy_sample_kernel` of vLLM's rejection sampler
(`source/rejection_greedy.py`, kernel lines 67–122; host lines 164–177, launched on a 1-D grid of
`batch_size` programs). Program `r` handles request `r`: if the request is not greedy it returns;
otherwise it walks the request's packed draft tokens `[cu[r-1], cu[r])`, writing the target
model's argmax (truncated to `int32`) at each position and stopping after the first position whose
draft token differs, and if every draft matched it writes the bonus token after them. The covered
launches have `SYNTHETIC_MODE = False`, with `is_greedy = None` or an `is_greedy` mask. The claim
is `Claims.RejectionGreedy.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). With
`start_r` (`0`, or `cu[r-1]`), `n_r = cu[r] − start_r`, `t(p)` the `int32` truncation of
`target_argmax[p]`, `k_r` the number of leading positions whose draft equals `t` (`matchLen`) and
`greedy_r` the request's flag (`greedyAt`), for each request `r < B` and column `j < S + 1`,
output element `r·(S + 1) + j` is:
- its initial value if the request is not greedy;
- `t(start_r + j)` if `j < min(k_r + 1, n_r)` (every column up to and including the first
  mismatch);
- `bonus[r]` if `j = n_r` and `k_r = n_r` (every draft matched);
- its initial value otherwise (the host's `-1`).

Every byte outside the output's first `B·(S + 1)` elements is unchanged. It is exact: integers
only, no tolerance. On vLLM's token ids the truncation is the identity.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ B ≤ 8192` requests (one program each);
- `B·(S + 1) ≤ 2^31`: the `int32` offsets `req_idx·(max_spec_len + 1) + pos` do not wrap;
- `cu` and `bonus` hold `B` entries, a mask `B` bytes, the output `B·(S + 1)` elements; all valid
  in the initial memory;
- the output shares no byte with any input (the inputs may overlap one another);
- `cu` holds the inclusive running sums of the draft counts (`cu[0] ≥ 0`, nondecreasing), its last
  entry at most `2^22` and at most the lengths of `draft` and `target`; every `n_r ≤ S`.

Draft, target, bonus and mask values are arbitrary (`-1` drafts, any `int64` target, any mask
byte, any initial output).

**The launches covered** (`SETUP.md`): vLLM's five hand-written cases, the eight random batches
each with `is_greedy = None` and with a mask, and the serving sizes (`MAX_SPEC_LEN = 128`, 1024
requests, all-greedy and mixed) (`setup_*_ok`; 23 instances). `mk_ok` covers every `B ≤ 8192` and
`S` within the offset limit.

**Device facts.** None: the kernel has no float operation.

## Exclusions

- **`B·(S + 1) > 2^31`: a limit of the kernel.** The `int32` output offset wraps once a written
  slot's offset reaches `2^31`, and the store goes below the output buffer (at `S = 128`, from
  about 16.6 million requests). Each such launch needs an output of at least 8 GiB.
- **More than 8192 requests, or more than `2^22` packed drafts**: the claim covers real batch
  sizes and context lengths only. Within `B ≤ 8192` the offset limit leaves out only
  `S ≥ 262,144`.
- **An empty batch (`B = 0`)**, **an output overlapping an input**, and **synthetic mode**:
  outside the file's launches.

`check_gpu.py` (one H200) checks `Post` exactly on the `SETUP.md` launches and on edges (high bits
in the `int64` targets, masks of every byte value, direct launches inside guarded buffers), and
runs two launches beyond `Inst.ok`.
