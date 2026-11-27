# Setup of `num_kv_splits_i32`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/attention/metadata.py:8`,
the kernel `get_num_kv_splits_triton`, launched by `TritonAttnBackend.get_num_kv_splits`
(`python/sglang/srt/layers/attention/triton_backend.py:431-480`), which the backend calls when it
builds the decode metadata.

## The source

`source/metadata.py` holds:
- `get_num_kv_splits_triton`, the `@triton.jit` kernel, copied verbatim but for its parameter
  default (`uneven_batch_ptr=None` becomes `uneven_batch_ptr`, `metadata.py:19`): TVA's subset has
  no kernel parameter defaults, and the host passes the parameter at every launch;
- `get_num_kv_splits`, the host function, adapted from `TritonAttnBackend.get_num_kv_splits`: the
  backend's attributes `num_head`, `num_kv_head`, `max_kv_splits` and `device_core_count` become
  arguments, and the two branches that return before the launch (`static_kv_splits` or a
  non-positive core count; deterministic inference with a split tile) are left out;
- `reference`, a test helper (the kernel's computation on the host, `ext_device_core_count` in
  `float64`);
- the test under `if __name__ == "__main__":`.

The other kernels and functions of `metadata.py` are not copied. The module docstring lists the
changes with their source lines.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. The configuration is that of the
backend's launch: grid `(1,)`; `MAX_NUM_SEQ = SCHEDULE_SEQ`, `256` for fewer than 256 requests and
`next_power_of_2(num_seq)` otherwise; the integer arguments as runtime `int32`. The launch passes
no options; the host passes `num_warps=4, num_stages=3`, Triton 3.8.0's defaults.

## The launches

The backend's launches (`triton_backend.py`), all with grid `(1,)`, an `int32` output
`num_kv_splits` of `num_seq · num_group` elements (a fresh tensor or the first elements of a
preallocated buffer), and `num_head`, `num_kv_head` the model's heads per rank:
- eager decode (`:973`): `forward_batch.seq_lens` (`int64`), or with DCP its per-rank lengths
  clamped to `1` (`int64`); `uneven_batch` a fresh `int32` `[1]` tensor;
- eager sliding-window layers (`:936`): `torch.minimum(seq_lens, sliding_window_size)` (`int64`),
  no `uneven_batch`;
- speculative draft windows (`:948`): `window_kv_indptr[1:] - window_kv_indptr[:-1]` (`int32`), no
  `uneven_batch`;
- decode into preallocated buffers (`:1415`): the first `bs` elements of a preallocated output,
  the decode lengths (`int64`; padded requests have length `1`), a preallocated `int32` `[1]`
  `uneven_batch`;
- sliding-window layers into preallocated buffers (`:1421`): no `uneven_batch`;
- the multi-step draft (`:2827`): `num_group = topk`, the first `bs * topk` elements of a
  preallocated output,
  `seq_lens[:bs]` (`int64`), no `uneven_batch`.

`max_kv_splits` is the server's `triton_attention_num_kv_splits` (`8` by default), raised for MLA
to at most `next_power_of_2(device_core_count)` (`:83-90`), or `⌈context_len / split_tile_size⌉`
with a split tile (`:369-372`); `device_core_count` is the device's SM count (`132` on the H200),
and the kernel is launched only when it is positive.

The setup theorems (`setup_*_ok`) cover: eager decode for Llama-3-8B (32/8 heads) at batches 1 and
7, Llama-3-70B at TP 8 (8/1) at batch 256, DeepSeek-V3 MLA at TP 8 (16/1) with 256 splits at batch
300, Qwen2.5-7B (28/4) at batch 8192, a split tile of 256 over 131072 tokens (512 splits) at batch
64; eager sliding-window lengths at batch 100; `int32` draft window lengths at batch 33; decode and
sliding-window lengths into preallocated buffers at batch 160; the multi-step draft with `topk = 4` at batch 64
and `topk = 8` at batch 1024; and the source's five test cases. Every one has the H200's 132
cores.

Not covered: the backend's branches that do not launch the kernel, and the AMD Wave backend's
launches (`srt/layers/attention/wave_backend.py:146`), whose target is not the H200.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 54 launches: the call sites
above at model shapes (10, through the host), the source's five test cases, and 39 direct launches
of the host's launch line on outputs carved from random-filled pools with guards: `L = 64·2^k` for
`k = 0 … 15` (where `v` is an integer), `L` around `64` and `128` and at `2^22 − 1`, `2^22`, the
80 % rule on both sides, every integer argument `1`, `device_core_count = 1` and `2^16`,
`max_kv_splits = 1` and `2^22`, grouped heads (`128/2`, `20/2`, `3/2`), `num_seq = 8192`,
`MAX_NUM_SEQ = 2^16` with `MAX_NUM_SEQ · num_group = 2^31`, `num_seq · num_group · num_head = 2^30`,
and an uneven batch. Every output is the one `Post` requires for some `E` of `CoreOk` (in every
launch the `E` whose outputs match is `⌊v⌋`), the flag is exact, and every input, guard and other
output element is unchanged; 87 wrong outputs (every split count plus one, the 80 % rule ignored, a
floor in place of a ceiling, `E` far above the set) all fail `Post`.
