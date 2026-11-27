# `num_kv_splits_i32`

**The kernel.** SGLang's `get_num_kv_splits_triton` (`source/metadata.py`, kernel lines 56–108;
host `get_num_kv_splits` lines 111–149), the Triton attention backend's choice of the number of
KV splits for split-K decode attention. One program (grid `(1,)`) loads the batch's KV lengths
`seq_lens` (`int64` or `int32`) in a block of `MAX_NUM_SEQ` lanes, takes their maximum `L` and
minimum (raised to `L` when it is above 80 % of `L`), and computes two chunk sizes: one from
`⌈L / min⌉` splits, one from `ext_device_core_count`, a `float32` expression of the core count
`C` and `L`, divided by the attention grid. It stores, for every request, the larger of the two
split counts at each of its `num_group` token positions of the `int32` output `num_kv_splits`,
and, when `uneven_batch` is a tensor, whether the batch is uneven. The claim is
`Claims.NumKvSplits.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and there is
one integer `E` (the kernel's `ext_device_core_count`) with

- `C ≤ E`, `E ≤ v + C·2^-17` and `v − C·2^-17 < E + 1`, where `v = C · max(log₂(L / 64), 1)`
  (`CoreOk`),

such that

- element `j < num_seq · num_group` of `num_kv_splits` is the `int32` value
  `numSplits E (j / num_group)`: for request `s` of length `ℓ`,
  `max(⌈ℓ / chunk1⌉, ⌈ℓ / chunk2 E⌉)`, with `chunk1 = ⌈L / min(⌈L / M⌉, max_kv_splits)⌉`
  (`M` the smallest length after the 80 % rule) and
  `chunk2 E = ⌈L / min(⌈E / token_grid⌉, max_kv_splits)⌉`, `token_grid` the source's
  `num_seq · num_group · num_head` (one head per KV head) or
  `num_seq · num_group · ⌈num_head / min(16, num_head // num_kv_head)⌉`. Exact;
- with `uneven_batch`, its element `0` is `1` when `M < L` and `0` otherwise. Exact;
- every byte outside those elements is unchanged.

`E` is the only value the device facts do not fix: it is the truncation of a `float32` value
computed with `/` and `tl.log2`. The set `CoreOk` allows is the integers in
`[max(C, ⌊v − τ⌋), ⌊v + τ⌋]`, `τ = C·2^-17 < 1`: a single integer, `⌊v⌋`, unless `v` is within `τ`
of an integer, and then the two integers around it. For `L ≤ 128` (`v = C`) it is exactly `C`.
For `L = 64·2^k`, `k ≥ 2`, `v = C·k` is an integer and the facts allow `C·k − 1` or `C·k`.
On the H200 every launch tried gives outputs equal to those of `E = ⌊v⌋`.

**The conditions** (`Inst.ok`, `Pre`):
- `seq_lens` is `int64` or `int32`;
- `1 ≤ num_seq ≤ 8192`; `MAX_NUM_SEQ` a power of two, `num_seq ≤ MAX_NUM_SEQ ≤ 2^20`;
- `num_group ≥ 1`, `MAX_NUM_SEQ · num_group ≤ 2^31`;
- `1 ≤ num_kv_head ≤ num_head`, `num_seq · num_group · num_head ≤ 2^30`;
- `1 ≤ max_kv_splits ≤ 2^22`; `1 ≤ device_core_count ≤ 2^16`;
- `seq_lens` holds `num_seq` lengths, each in `[1, 2^22]`; the output holds
  `num_seq · num_group` elements, `uneven_batch` (when given) one; all are valid in the initial
  memory;
- neither output shares a byte with `seq_lens` or with the other output.

Any lengths in `[1, 2^22]` are covered.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the backend's call sites — decode
with `uneven_batch` (into fresh or preallocated buffers), sliding-window lengths, the
speculative draft's `int32` window lengths, the multi-step draft with `num_group = topk` — at
model shapes (Llama-3-8B, Llama-3-70B at TP 8, DeepSeek-V3 MLA at TP 8 with 256 splits, Qwen2.5-7B
at batch 8192, a split tile of 256), and the source's five test cases. `mk_ok` covers the host's
layout for every shape in `Inst.ok`; the lengths are `Pre`'s.

## Tolerances

Only `E` carries an error; the tolerance `C·2^-17` on the `float32` value before truncation rests
on two device facts, `Device.DivFull` (`/` on `float32`: within `2^-22·|x/y| + 2^-148`) and
`Device.Log2F32` (`tl.math.log2`: within `2^-22·|log₂ x| + 2^-149`), and on IEEE rounding of the
product:

- `L ≤ 2^22` and `C ≤ 2^16` convert to `float32` exactly, and `64.0`, `1.0` are exact;
- the quotient `q` of `L` by `64` is within `(2^-22 + 2^-142)·L/64` of `L/64` (`L ≥ 1`), so
  `|log₂ q − log₂(L/64)| ≤ 3·(2^-22 + 2^-142)`;
- `log₂(L/64) ∈ [−6, 16]`, so `|log₂ q| ≤ 17` and `tl.log2` adds at most `17·2^-22 + 2^-149`: in
  all, `|lg − log₂(L/64)| ≤ 21·2^-22`; `tl.maximum` with `1` keeps that bound and gives
  `m ∈ [1, 17]`;
- the product `C·m` rounds once: within `17·C·2^-24 + 2^-150`. Altogether
  `|p − v| ≤ 101·C·2^-24 + 2^-150 ≤ 128·C·2^-24 = C·2^-17`;
- `p ≥ C`: `C·m ≥ C`, and `C` is a multiple of the quantum of every `float32` value below `2^24`,
  so rounding to nearest does not go below it.

`E = ⌊p⌋` (`p ≥ 1`; truncation is the floor), which gives `CoreOk`. At the largest covered core
count (`2^16`) the bound is `τ = 1/2`; at the H200's `132`, `τ ≈ 0.001`.

## Exclusions

- **A length `0`**: outside real use (a decode request has at least its new token; the backend
  clamps the DCP lengths to `1` and pads preallocated batches with length `1`). With a `0` length the
  minimum is `0` and `tl.cdiv(max_seq_len, min_seq_len)` divides by zero.
- **Lengths above `2^22`, `num_seq > 8192`, `max_kv_splits > 2^22`, `device_core_count > 2^16`**:
  the claim covers real model shapes and devices only.
- **`MAX_NUM_SEQ · num_group > 2^31`**: a limit of the kernel; the `int32` `offs_seq * num_group`
  wraps for lanes beyond the batch, which can then pass the mask `offs_token < num_seq * num_group`
  and store outside the output.
- **`num_seq · num_group · num_head > 2^30`**: a limit of the kernel (with slack); the `int32`
  `token_grid` wraps at `2^31`, and `tl.cdiv(ext_device_core_count, token_grid)` adds up to `2^21`
  to it.
- **`num_kv_head > num_head`, `num_kv_head = 0`, `num_seq = 0`**: outside real use; the kernel
  divides by zero (`block_h = 0`, `num_head // 0`, a minimum of `0`).
- **An output overlapping an input or the other output**: outside real use (fresh or separate
  buffers).
- **The host's other branches** (`static_kv_splits`, a non-positive core count, deterministic
  inference with a split tile): they do not launch the kernel.
- **The AMD Wave backend** (`srt/layers/attention/wave_backend.py`), which also launches the
  kernel: another target than the H200.

## Modelling decisions

- `seq_lens` may be `int64` or `int32`: the model computes in the signed integer type of the
  dtype's width (`wOf`, `toW`, `ofW`, identities for both); `max_kv_splits` and
  `max_kv_splits_2` are sign-extended where they meet the lengths (`IntV.cast`), as Triton
  promotes them; the split counts are converted to `int32` by the store (`Tensor.castInt`).
- `tl.max`/`tl.min` of an integer block are `Tensor.foldAll IntV.max` / `IntV.min` from the type's
  smallest / largest value; the masked lanes hold `0` and `max_seq_len` as the `other=` of each
  load says.
- At the value `1`, `num_seq`, `num_group`, `num_head`, `num_kv_head`, `max_kv_splits` and
  `device_core_count` are compile-time constants. Every use gives the model's value: in a product
  or comparison with a tensor the constant is built as an `int32` (or `float32` `1.0` for the core
  count), `tl.minimum` makes it an `int32` tensor, `range(0, 1)` runs once with `i = 0`, and
  `num_head = num_kv_head = 1` makes `num_kv_group == 1` a compile-time `True`, the branch the
  model's runtime `if` takes; Python's arithmetic on the constants (`1 * 1 * 1`, `1 // 1`) gives
  the same values as the wrapping `int32` arithmetic of the model, which does not wrap there.
- `block_h, num_kv_group = 16, …` makes `block_h` an `int32` scalar tensor; both branches of the
  runtime `if` at line 90 define `token_grid`, an `int32`.

`check_gpu.py` (one H200) checks `Post` on the setup's launches and on edges of `Inst.ok`;
`MAX_NUM_SEQ` is cut to `2^16` there (a block of `2^20` lanes compiles for many minutes).
