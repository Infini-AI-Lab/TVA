# Setup of `extend_attn_bf16`

From SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/attention/extend_attention.py:375` `_fwd_kernel`, launched by
`extend_attention_fwd` (`extend_attention.py:916`), which SGLang's Triton attention backend calls for
the extend (prefill) step and for the target verification of speculative decoding,
`TritonAttnBackend.forward_extend` (`python/sglang/srt/layers/attention/triton_backend.py:1669`).

## The source

`source/extend_attention.py` holds:
- `tanh` (lines 93-96, the source's lines 207-210), a `@triton.jit` helper the kernel calls only
  under `logit_cap > 0`;
- `_fwd_kernel` (lines 99-638), the `@triton.jit` kernel, verbatim from the source's lines
  374-913 but for its parameter defaults (below): one program per (request, query head, block of
  `BLOCK_M` new tokens) returns if its block starts past the request's new tokens; otherwise it
  runs an online softmax (running maximum `e_max`, unnormalised sum `deno`, accumulator `acc`)
  over the request's cached prefix in blocks of `BLOCK_N_PREFIX` slots gathered through
  `kv_indices`, then over the request's new tokens up to the end of its block in blocks of
  `BLOCK_N`, causally or with the custom (tree) mask, each stage with the sliding-window mask when
  there is one (skipping tiles the masks empty), with `tl.dot` in `float32` and the probabilities
  cast to `bfloat16` before the second `tl.dot`, and stores `acc / deno`. The kernel's other
  constexpr paths (logit cap, xai temperature, sinks, LSE, the compact tile grid, `exp2`, fp8, the
  RoPE part `BLOCK_DPE`, pages larger than 1, `IDENTITY_KV_INDICES`, `score_mod`, the transposed
  store, non-causal layers, the custom mask over the prefix) stay in the source;
- `_get_block_sizes_for_extend_attention` and `_extract_kv_strides`, host helpers;
- `extend_attention_fwd`, the host: the source's wrapper with the backend's flags;
- `reference` (attention in `float64`, request by request, causal or with the custom mask, and the
  window) and the test under `if __name__ == "__main__":` (the code base's test shapes, a window of
  127, and target verifications with and without a window).

Changes from the source (the file's docstring lists each with its `file:line`):
- `_fwd_kernel`'s last eight parameters lose their defaults (`extend_attention.py:437-444`:
  `IS_GFX1250`, `PAGE_SIZE`, `IDENTITY_KV_INDICES`, `SCORE_MOD`, `Aux0`, `aux0_stride_t`,
  `aux0_stride_h`, `aux0_len`): TVA's subset has no kernel parameter defaults; the host passes each,
  as the wrapper does. The kernel's body is unchanged.
- The imports of `sglang` and the module's device globals (`extend_attention.py:19-45`) are replaced
  by locals of the host functions with their values on an H200 (`_is_cuda = True`,
  `_is_hip = _is_gfx95 = _is_gfx1250 = False`, `CUDA_CAPABILITY = (9, 0)`); the file has no
  module-level assignments. `_extract_kv_strides` is copied from `decode_attention.py:212-225`;
  `unpack_aux_tensors(None, None)` is `(None, 0, 0, 0)` (`score_mod.py:30-32`).
- `_get_block_sizes_for_extend_attention` (`extend_attention.py:48-142`) keeps the branches the
  H200 takes (capability 9).
- `extend_attention_fwd` (`extend_attention.py:916-1168`) takes the arguments that differ between
  the backend's extend batches and target verifications, with or without a sliding window
  (`custom_mask`, `mask_indptr`, `sliding_window_size`, `window_kv_offsets`, keyword arguments
  after `sm_scale`), and fixes the rest at the backend's values (below); it allocates the output
  `o_extend = torch.empty_like(q_extend)` (as `forward_extend` does for equal head dims,
  `triton_backend.py:1690-1698`) and returns it, where the source's wrapper takes it from the caller.
  The fp8 zero-prefix cast, the compact tile grid and the HIP launch options
  (`extend_attention.py:989-1010, 1056-1074, 1080-1082`), all off on an H200, are left out; the
  launch itself (`extend_attention.py:1095-1168`) is unchanged.

## The config

No autotuning and no heuristics: the wrapper picks the launch itself. On an H200 (capability 9),
`_get_block_sizes_for_extend_attention` gives, for head dim `D = Lq = Lv`:
- `BLOCK_DMODEL = BLOCK_DV = next_power_of_2(D)`, `BLOCK_DPE = 0` (for every `D` but 192, 288 and
  576, the MLA head dims, which are not part of the setup);
- `BLOCK_M = 128` for `D <= 128`, `BLOCK_M = 64` for `128 < D <= 256`; `BLOCK_N = 64`; and
  `BLOCK_N_PREFIX = BLOCK_N` (no fp8);
- `num_warps = 4` for `D <= 64`, else `8`; `num_stages = 1`;
- grid `(batch, num_heads, cdiv(max_len_extend, BLOCK_M))`; `kv_group_num = num_heads //
  num_kv_heads`; `sm_scale = 1 / sqrt(D)` by default (the backend passes `layer.scaling`).

The flags, as `forward_extend` passes them (`triton_backend.py:1888-1915`: a causal decoder layer,
`logit_cap = 0`, no sinks, no `score_mod`, a `bfloat16` KV cache without scales, not deterministic
mode, no DCP):
- an extend batch: `custom_mask = None` (`USE_CUSTOM_MASK = False`; `forward_metadata.custom_mask`
  is `None` for an extend batch, `triton_backend.py:1118`), `qo_indptr` `int64`
  (`triton_backend.py:398-402, 1114-1116`);
- a target verification (`forward_mode.is_target_verify()`, `triton_backend.py:1006-1061`; on an
  H200 the split-KV verify kernels are off, they need gfx95, `:238-252`): the speculative
  algorithm's tree mask `custom_mask` (`bool`; `USE_CUSTOM_MASK = True`) with `mask_indptr`
  (`int64`, `:404-406`, the cumulative `ext_b·(seq_b + ext_b)`), `qo_indptr` `int32` (`:1009-1015`,
  `num_tokens_per_req` draft tokens per request), `max_len_extend = num_tokens_per_req`;
- a layer with a sliding window (`layer.sliding_window_size > -1`, `:1812-1832`):
  `SLIDING_WINDOW_SIZE` the layer's window (Gemma 3: `sliding_window - 1`,
  `gemma3_causal.py:65`), `kv_indptr`, `kv_indices` the window's (`window_kv_indptr`,
  `window_kv_indices`: the last `min(prefix, W)` cached tokens, `update_sliding_window_buffer`,
  `:2845-2902`) and `window_kv_offsets` (`int64`, the window's first position, read only with the
  custom mask); without one `SLIDING_WINDOW_SIZE = -1`;
- every launch: `is_causal = True`, `k_scale = v_scale = 1.0` (`k_descale`, `v_descale`,
  `triton_backend.py:1835-1840`), `logit_cap = 0.0`, `HAS_SINK = False`,
  `xai_temperature_len = -1`, `STORE_LSE = False`,
`SKIP_PREFIX = SKIP_EXTEND = False`, `PAGE_SIZE = 1` (the server's default page size),
`IDENTITY_KV_INDICES = False`, `SCORE_MOD = None`, `SKIP_PREFIX_CUSTOM_MASK = True`; on an H200
`IS_GFX1250 = STORE_TRANSPOSE = USE_COMPACT_TILE_GRID = USE_EXP2 = USE_FP8_PREFIX = USE_FP8_EXTEND
= False` and `FP8_MAX = 1.0`.

## The launches

The claim covers the host's launches with, as variables:
- `q_extend`: `bfloat16`, `[T, num_heads, D]`; `k_extend`, `v_extend`: `bfloat16`,
  `[T, num_kv_heads, D]`; `k_buffer`, `v_buffer`: `bfloat16`, `[num_slots, num_kv_heads, D]`; last
  dimension contiguous, token and head strides as the tensors give them (`q_extend` may be a view of a
  fused projection; the backend makes `k_extend`, `v_extend` contiguous); `num_heads` a multiple of
  `num_kv_heads`; `1 <= D <= 256`;
- `qo_indptr`: `int64` (an extend batch) or `int32` (a target verification), `kv_indptr`: `int32`,
  both `[batch + 1]`, `kv_indices`: `int64` (the backend's dtypes, `triton_backend.py:381-383,
  398-402, 1009-1015, 1076-1080`): request `b`'s new tokens are rows
  `qo_indptr[b] … qo_indptr[b + 1] - 1` (nondecreasing, within `T`, at most `max_len_extend` of
  them; `0` allowed), its prefix the cache slots `kv_indices[kv_indptr[b] + j]`,
  `j < kv_indptr[b + 1] - kv_indptr[b]` (prefix length `0` allowed; a sliding-window layer's window
  prefix, or the whole prefix as in the code base's window test), each a slot of the cache; prefix
  and new tokens of a request at most `2^18` positions in all;
- a sliding window `W = SLIDING_WINDOW_SIZE` from `1` to `2^18`, or none (`-1`);
- with the custom mask (target verification): `custom_mask`, `bool`, request `b`'s block of
  `ext_b` rows of `pre_b + ext_b + window_kv_offsets[b]` (offset `0` without a window) at
  `mask_indptr[b]` (`int64`), a tree mask over the new tokens (each token attends itself and some
  earlier ones; the prefix columns are not read), at most `2^12` draft tokens per request, and
  `T·num_heads·D ≤ 2^31`; `window_kv_offsets` (`int64`, with a window) the window's first position;
- the code base's shapes: the unit test (`test/registered/attention/test_triton_attention_kernels.py:172-325`,
  its first launch, line 245) runs 19 requests, 12 heads on 4 KV heads, head dims 256, 128, 96, 80
  and 13, prefix and extend lengths drawn from `[1, 6165)`, values `N(0.1, 0.2)`; its window test
  (`:533-643`) runs 19 requests, 64 heads on 8, head dim 128, window 127, lengths from
  `[1, 6165)`; models run head dims 64 to 256 (Qwen2.5-0.5B 64, the Llama and Qwen families 128,
  Gemma 3's layers 256), 1 to 128 heads per rank with grouped KV heads (down to one KV head per
  rank), chunked-prefill batches of thousands of new tokens over prefixes up to a 128K context
  (Llama 3.1) and a 256K context (the Qwen3-2507 family, head dim 128, 262,144 tokens); Gemma 3's
  local layers have a window of 1023 (`sliding_window = 1024`, five of every six layers); EAGLE's
  target verification runs 4 to 64 draft tokens per request (`--speculative-num-draft-tokens`).

Not part of the setup: the unit tests' `int32` `qo_indptr` without a custom mask and their `int32`
`kv_indices` (the backend passes an `int32` `qo_indptr` only with the custom mask, except on the
dLLM FA4 path, and `kv_indices` as `int64`, `triton_backend.py:398-402, 1009-1015, 1076-1080`), so
their launches as written (`test_triton_attention_kernels.py:245, 282, 602`); the other paths of the
kernel (logit caps, sinks, xai temperature, the LSE and skip flags of DCP, pages larger than 1,
`identity_kv_indices`, `score_mod`, non-causal layers and bidirectional windows,
`skip_prefix_custom_mask = False`, `k_scale`/`v_scale` other than `1.0`, an fp8 KV cache, the MLA
head dims 192/288/576, `Lq != Lv`), and the deterministic mode's other kernel
(`_fwd_kernel_unified`).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs the code base's test at its full size for each of its head dims
(19 requests, 12/4 heads, lengths up to about 6100; the backend's index dtypes), SETUP.md's model shapes
(Llama-3-8B 32/8 and Qwen2.5-7B 28/4 at head dim 128, Qwen2.5-0.5B 14/2 at 64, Gemma 3 16/8 at 256,
8/1 heads; requests with prefix 0 and extend 1 included; prefix slots scattered through the cache), a
strided `q_extend` (a view of a fused projection), long requests (a 128K context, `2^17 - 8192` cached
and 8192 new tokens, and the longest covered, `2^18 - 2048` and 2048, also with the prefix's first
quarter of values shifted), sliding-window layers (the window test's shape at window 127 with the
whole prefix, Gemma 3's local layers with window 1023 over window prefixes and an 8K chunk, windows
64, 1 and 4096), target verifications (`int32` `qo_indptr`, random tree masks of 64, 4 and 8 draft
tokens, head dims 128 and 256, with window 1023 and window 60 and their mask offsets), `Inst.ok`
edges (head dims 9 and 200, `sm_scale = 0.5`, requests with no new tokens, a grid longer than the
longest request) and `Pre` edges (large scores, `|v| ≈ 10^30`, `|v| ≈ 10^-20`, subnormal `v`,
uniform attention, and an input on which every `bfloat16` rounding pushes the same way, at head dim
128 and at head dim 16 with one trip). The worst error / tolerance is `0.980` (margin `1.02`; the
second adversarial input; `0.802` on random inputs (window 1), `0.784` on a target verification,
`0.461` on the code base's test); every written element is finite, every input is unchanged, and
every element of `o_extend` outside the requests' new tokens keeps its bits. Head dims 1 and 8 are
rejected on the host (`CompilationError`). Wrong outputs are rejected: no prefix, not causal, heads
swapped, one element off by `0.25`, a NaN, a write outside the requests; with a window, the output
without it; with the custom mask, the causal output, the output without the prefix, and with a
window the output without it or with the mask read without the window offset; the all-zero output
at 128K (`2.6` times the bound) and at `2^18` (`1.3`, only by the tail: `524,125` of its `524,288`
elements are within the bound, so the shifted-prefix test, `6.0`–`6.6`, is the real check there);
the output without the prefix's first quarter at 128K (`1.5`) and, on the shifted input, at `2^18`
(`6.0`). On random values at `2^18` that output is within the bound (`0.83`), as dropping a quarter
of so many independent positions changes the exact attention by about `attnAbs/sqrt(n)`.
