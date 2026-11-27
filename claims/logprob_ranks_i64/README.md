# `logprob_ranks_i64`

**The kernel.** vLLM's `_ranks_kernel` (`source/logprob_ranks.py`, kernel lines 47–67; host
`compute_token_ranks` lines 70–85, the ranks part of `compute_topk_scores`). A 1-D grid of
`batch_size` programs; program `r` loads row `r`'s sampled token id `t_r` and its logit
`x_r = logits[r, t_r]`, walks the row in blocks of `BLOCK_SIZE = 8192` lanes (lanes past
`vocab_size` hold `-inf`), counts in an `int32` the lanes with `logits >= x_r`, and stores the
count at `token_ranks[r]` (`int64`). The claim is `Claims.LogprobRanks.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and in the
final memory:
- for every row `r < batch_size`, element `r` of `token_ranks` is the `int64` `rank d r`: the
  number of entries `j < vocab_size` with `x_r ≤ logits[r, j]` (`countGe`), plus, when
  `x_r ≤ -inf`, the `padLanes = ⌈vocab_size / BLOCK_SIZE⌉ · BLOCK_SIZE − vocab_size` padding lanes
  of the last block. The comparison is ordered (`FloatV.le`): false when either side is NaN. So
  ties count and the token counts itself (the rank is `1` for a unique maximum); a NaN `x_r`
  gives `0`; a NaN entry never counts; `x_r = -inf` counts every entry that is not NaN and the
  padding lanes as well (the rank may then exceed `vocab_size`). Exact: no tolerance;
- every byte outside the output's first `batch_size` elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- the logits are `float32`, `bfloat16` or `float16`;
- `BLOCK_SIZE` a power of two `≤ 2^20` (the host's is `8192`);
- `vocab_size + BLOCK_SIZE ≤ 2^31`; the row stride below `2^31`; `batch_size < 2^31`;
- `logits` holds element `r · stride + j` for every `r < batch_size`, `j < vocab_size`; the token
  ids and the output hold `batch_size` elements; all valid and holding the instance's contents in
  the initial memory;
- the output shares no byte with `logits` or `sampled_token_ids`;
- every token id is in `[0, vocab_size)`.

Every logit value (NaN, infinities, signed zeros) is covered, every stride below `2^31` (`0`, and
strides below `vocab_size`, whose rows overlap, included), and `batch_size = 0`.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the host's six test cases, the sampler's raw
(`bfloat16`, `float16`) and processed (`float32`) logits, the slice of a padded vocabulary
(`64 × 50257`, stride `50304`), a speculative batch (`2560 × 128256`), prompt-logprob chunks,
vLLM's `16448 × 131072` regression shape, and an empty batch (15 instances). `mk_ok` covers every
dtype, `B · 8 ≤ 2^32` rows, `V + 2^e ≤ 2^31` and any stride below `2^31`.

**Device facts.** None: the kernel compares and counts (comparisons, integer sums, the exact
conversions `int32 → int64`, `int1 → int32`, and the conversion of the `float32` `-inf` to the
logits' dtype, which is exact).

## Modelling decisions

- `logits >= x` is the ordered `x <= logits` (`Tensor.fle`, `geX`), in the logits' own dtype (a
  float compared with its own dtype is not promoted).
- `other=float("-inf")` is the `float32` constant of the double `-inf`, converted to the logits'
  dtype by `tl.convert` (the identity for `float32`).
- `req_idx * logits_stride` sign-extends the `int32` stride (`tl.convert .i32 .i64`) and multiplies
  in `int64`; the row offset never wraps (`batch_size · stride < 2^62`).
- `tl.sum` of the `int32` block is `Tensor.foldAll (+)` (an order-independent wrapping sum); the
  final `int32` count is sign-extended into the `int64` output.
- At the value `1`, `vocab_size` and `logits_stride` are compile-time constants (`vocab_size = 1`;
  a stride of `1`); every use makes them `int32` tensors first (`range`'s bounds, `block <
  vocab_size`) or multiplies them into the `int64` `req_idx`, so the values are the model's.

## Exclusions

- **`vocab_size + BLOCK_SIZE > 2^31`**: a limit of the kernel. The `int32` loop variable's last
  increment `i + BLOCK_SIZE` overflows (`tl.RangeOk`); at `vocab_size ≥ 2^31` Triton passes an
  `int64`. No vocabulary is near it.
- **Strides `≥ 2^31`**: outside real use (Triton would pass an `int64`; the model takes `int32`).
- **Token ids outside `[0, vocab_size)`**: the kernel would read outside the row; vLLM's sampled
  tokens, prompt tokens and the rejection sampler's padding `0` are vocabulary indices.
- **`int32` token ids**, **dtypes other than `float32`, `bfloat16`, `float16`**, **an output
  overlapping an input**: outside the code base's launches.

`check_gpu.py` (one H200) checks `Post` exactly on the setup's launches and on edges of `Inst.ok`
and `Pre`.
