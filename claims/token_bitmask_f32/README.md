# `token_bitmask_f32`

**The kernel.** SGLang's `apply_token_bitmask_inplace_kernel` (`source/bitmask_ops.py`, kernel
lines 56–124; host `apply_token_bitmask_inplace_triton` lines 127–184, with
`get_device_core_count` lines 49–53), used by grammar-constrained decoding (xgrammar) on the
sampler's `float32` logits. A persistent loop: `NUM_SMS` programs (the device's multiprocessor
count, `132` on the H200) walk the `num_rows · cdiv(vocab_size, 4096)` work items with stride
`NUM_SMS`; item `w` is processed row `w / nb`, block `w % nb` (`nb = cdiv(vocab_size, 4096)`). An
item loads its block's `128` bitmask words, expands them to `4096` bits (`>>` by `0..31`,
`& 1`, `== 0`, row-major reshape) and stores `-inf` where the token is below `vocab_size` and its
bit is `0`. The processed rows are `0 … num_rows − 1`, or `indices[0 … num_rows − 1]` when the
host is given indices. The claim is `Claims.TokenBitmask.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and, with
`r` the `k`-th processed row (`rowOf`) and token `c`'s bit bit `c % 32` (least significant first)
of the `int32` bitmask element `r · bitmask_strides + c / 32` (`bitOf`):
- for every `k < num_rows` and `c < vocab_size` whose bit is `0`, the logit
  `r · logits_strides + c` holds the `float32` `-inf` (`0xff800000`);
- every other byte is unchanged: the logits whose bit is `1`, the logits past `vocab_size` (a
  padded row), the rows not processed, the bitmask, the indices.

It is exact, bit for bit.

**The conditions** (`Inst.ok`, `Pre`):
- at most `NUM_SMS` programs, and enough of them to reach every item
  (`min(NUM_SMS, items) ≤ grid ≤ NUM_SMS`): the host's `grid = NUM_SMS` on a CUDA device, and its
  `grid = items`, `NUM_SMS = next_power_of_2(items)` for a device that reports no multiprocessors;
  `NUM_SMS` positive (the loop's step) unless there is no item, and below `2^31`;
- `vocab_size + 4095 < 2^31`, `num_rows < 2^31`, `items + NUM_SMS ≤ 2^31`, both strides below
  `2^31`;
- `vocab_size ≤ logits_strides` and `vocab_size ≤ 32 · bitmask_strides` (the host's
  `vocab_size = min(width, 32 · bitmask width)`);
- the logits hold `logits_rows` rows of `logits_strides` and the bitmask `bitmask_rows` rows of
  `bitmask_strides`, each at most `2^31` elements;
- without indices, `num_rows` is at most both row counts; with indices, the indices buffer holds
  `num_rows` entries, each a row of both tensors, all distinct (the kernel's "unique indices");
- the logits share no byte with the bitmask or the indices; the buffers are valid in the initial
  memory.

Any logits values (NaNs, infinities) and any bitmask words are covered.

**The launches covered** (`SETUP.md`): SGLang's calls, `[bs, vocab]` logits and
`[bs, cdiv(vocab, 32)]` bitmasks without indices on the H200, at vocabularies 32000, 128256,
129280, 151936, 152064, 202048 and 262144 and batches 1 to 8192 (8192 rows at 262144 is `2^31`
logits); a padded row (width 32064, vocabulary 32000); the host with a list of indices; the host's
branch without multiprocessors, with work and empty (`setup_*_ok`; 13 instances). `mk_ok`,
`mkG_ok` and `mkIdx_ok` cover the host's layout at every shape `Inst.ok` allows.

**Tolerances.** None: the only float value is the stored constant `-inf` (the `float32` constant
Triton builds for `-float("inf")`), and the comparison and the masks are integer operations. No
device fact is used.

## Exclusions

- **A tensor of more than `2^31` elements** (logits or bitmask): a limit of the kernel. The row
  offset `batch_id · logits_strides` is an `int32` product, which wraps for a row whose offset
  reaches `2^31`, and the store goes outside the logits. Within SGLang's sizes this leaves out
  more than 8192 rows at vocabulary 262144 (an 8 GiB `float32` tensor).
- **`vocab_size + 4095 ≥ 2^31`, `items + NUM_SMS > 2^31`**: limits of the kernel (the `int32`
  `cdiv` addition, the product `num_rows · num_blocks` and the loop's last increment wrap).
- **Duplicate indices** or an index outside either tensor: outside the kernel's contract (two
  programs would store into the same logits).
- **A grid of more than `NUM_SMS` programs** (two programs would serve the same items), **or too
  few to reach every item**: outside the host's launches.
- **Logits that overlap the bitmask or the indices**, **a stride or count of `2^31` or more**
  (an `int64` argument): outside real use.
- **Logits of other dtypes** (`bfloat16`, `float16`, which the host accepts with fewer warps):
  not on SGLang's path, which applies the mask to the logits processor's `float32` logits.

`check_gpu.py` (one H200) checks `Post` bit for bit on the setup's launches (the two largest cut to
fit 4 GB) and on edges, and checks that plausible wrong outputs fail `Post`.
