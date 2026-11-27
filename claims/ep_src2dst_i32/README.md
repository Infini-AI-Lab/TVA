# `ep_src2dst_i32`

**The kernel.** SGLang's `compute_src2dst_triton_kernel` (`source/ep_src2dst.py`, kernel lines
37–59; host `cutlass_w4_run_moe_ep_preproess` lines 62–74). The host sorts the flat `int32` expert
ids `topk_ids` stably (`sorted_topk_ids`, ascending; `reorder_ids`, the `int64` permutation) and
launches a 1-D grid of `cdiv(n, 512)` programs, `n = topk_ids.numel()`. Every program first finds
`first_nonnegative`, the first sorted position whose id is not negative: `0` when
`sorted_topk_ids[0] ≥ 0`, else a binary search over the `n` sorted ids. Program `p` then handles
the sorted positions `i = 512·p + l < n`: it stores `i − first_nonnegative`, or `−1` when
`i < first_nonnegative`, at `src2dst[reorder_ids[i]]`. The claim is `Claims.EpSrc2Dst.claim` in
`Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- for every sorted position `i < n`, element `reorder_ids[i]` of `src2dst` holds the `int32` `−1`
  when `sorted_topk_ids[i]` is negative, and `i − nneg` otherwise, where `nneg` is the number of
  negative ids among the first `n` sorted ids (the slot's row among the slots routed to a local
  expert, in sorted order);
- every byte outside the elements `reorder_ids[i]`, `i < n`, of `src2dst` is unchanged.

It is exact: no float operation, no tolerance, no device fact.

**The conditions** (`Inst.ok`, `Pre`):
- `n < 2^31` (`num_toks` is an `int32`; at `n = 1` the compile-time constant `1`, which gives the same
  values: `model`'s docstring);
- `sorted_topk_ids` and `reorder_ids` hold at least `n` entries; all three buffers are valid in
  the initial memory;
- `src2dst` shares no byte with either input;
- the negative ids among the first `n` sorted ids come first (an ascending sort has this; the ids
  after them may be in any order);
- the first `n` entries of `reorder_ids` are distinct elements of `src2dst` (a permutation of
  `[0, n)` into the host's `src2dst` of `n` elements has this).

Any `int32` id values are covered, negative ids below `−1` included.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the host's test cases (1 to 24576 slots,
from none to all of them negative), decode (one token, top-8), a 16384-token prefill chunk,
`2^22` tokens of top-8 (`2^25` slots), and `n = 2^31 − 1`. `mk_ok` covers the host's layout for
every `n < 2^31`; the contents are `Pre`'s.

**Device facts.** None: the kernel is integer-only.

## Exclusions

- **`n ≥ 2^31`**: Triton passes `num_toks` as an `int64`, which the model does not describe; the
  `int32` positions `pid * 512 + arange` would also wrap. Real use stays far below (`2^25` slots
  at `2^22` tokens of top-8).
- **`int64` ids**: the caller makes `topk_ids` contiguous `int32` (`cutlass_w4a8_moe.py:131-132`);
  `sorted_topk_ids` is then `int32`.
- **Sorted ids whose negatives do not come first**: outside the host, which sorts them.
- **Repeated or out-of-range entries of `reorder_ids`**: outside the host, whose `reorder_ids` is
  the sort's permutation; two programs would store to one element (a race), or a store would leave
  the buffer.
- **An output overlapping an input**: outside real use (a fresh `torch.empty`).
- **`n = 0`** is covered (an empty grid writes nothing); the host returns before launching.

## Modelling decisions

- `first_nonnegative = 0`, `left, right = 0, num_toks` and `left = middle + 1` hold `int32`
  values (a name assigned a Python number holds an `int32` tensor); `(right - left) // 2` is the
  truncating `int32` `//` (`tl.idiv`) of the constant `2` built in `int32`.
- `while left < right:` carries `(left, right)` (`tl.whileLoop`); `middle` and `expert_id` are new
  to the body; `first_nonnegative = left` uses a name defined before the loop.
- `dst_id >= first_nonnegative` is `Tensor.le (bc first_nonnegative) dst_id`; the `−1` of
  `tl.where` is built in `int32`.
- Every program reads the sorted ids it needs (only reads), so the programs share no written byte.

`check_gpu.py` (one H200) checks `Post` exactly on the host's launches and on edges of `Inst.ok`
and `Pre`.
