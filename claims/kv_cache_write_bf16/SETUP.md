# Setup of `kv_cache_write_bf16`

From vLLM v0.33.0 (commit `4ba787ec99`),
`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`, the kernel
`reshape_and_cache_kernel_flash`, launched by the wrapper `triton_reshape_and_cache_flash`.

## The source

`source/reshape_and_cache_flash.py` holds:
- `reshape_and_cache_kernel_flash` (lines 73-195), the `@triton.jit` kernel, verbatim from the
  source's lines 32-154: one program per (token, tile of `TILE_SIZE` of the token's
  `num_heads * head_size` values) stores the token's key and value rows at the cache slot
  `slot_mapping[token]`, and does nothing for a slot below 0 (padding);
- `reshape_and_cache_flash`, the host wrapper, from `triton_reshape_and_cache_flash` (source lines
  377-483), with `_is_supported_kv_cache_dtype` and `_is_quantized_kv_cache` (from source lines
  16-29);
- `make_caches` and `reference`, test helpers (caches laid out as vLLM's backends lay them out,
  and the expected caches after the launch);
- the test under `if __name__ == "__main__":`.

The changes from vLLM's source, all on the host (the module docstring lists each with
its line): the vLLM imports are removed; `current_platform.fp8_dtype()` is
`torch.float8_e4m3fn`; the fp8 cache dtypes recognised are `fp8` and `fp8_e4m3` (vLLM's
`is_quantized_kv_cache` asks a registry); the platform branch of the wrapper is the CUDA one for
capability 9; the module-level dtype sets are locals of the functions (the file has no
module-level assignments). The kernel is unchanged. The caches are the caller's tensors, as in vLLM: the
kernel only stores into them (it reads `key`, `value`, `slot_mapping`, and, with an fp8 cache,
`k_scale` and `v_scale`), and the test allocates them.

## The config

No autotuning and no heuristics in the source: the wrapper computes the launch itself ("heuristics
instead of autotuning", source lines 437-448), and the host keeps that computation, on its CUDA
branch for an H200 (capability 9, so `TILE_SIZE` is not capped at 512):
- `TILE_SIZE = min(2048, next_power_of_2(num_heads * head_size))`;
- `num_warps = 16`, `num_stages = 10`;
- grid `(num_tokens, cdiv(num_heads * head_size, TILE_SIZE))`.

The kernel declares no parameter defaults; the host passes each parameter.

Flags as the wrapper sets them for this path: `USE_HEAD_MAJOR_LAYOUT = False` (a 4-D key
cache), `INTERLEAVED_V_PACK_FACTOR = 0`, `x = 1`, `dim_stride_k = dim_stride_v = 0`;
`FP8_KV_CACHE = False` with `kv_cache_dtype = "auto"` (a `bfloat16` cache) and `True` with
`kv_cache_dtype = "fp8"` (a `float8e4nv` cache: the host views vLLM's `uint8` cache as
`torch.float8_e4m3fn`, its `current_platform.fp8_dtype()` on an H200).

## The launches

The claim covers the wrapper's launches with a `bfloat16` or a `float8e4nv` cache in the flash
(NHD) layout, the paths of vLLM's Triton attention backend
(`vllm/v1/attention/backends/triton_attn.py:828-843`, which views a quantized cache as
`self.fp8_dtype`) and of the code base's test (`tests/kernels/attention/test_cache.py:181`,
`implementation="triton"`, `kv_cache_dtype` `"auto"` and `"fp8"`, layout `NHD`, the per-tensor
scales `kv_scale_type="tensor"`), with shapes as variables:
- `key`, `value`: `bfloat16` (with either cache), or `float8e4nv` with the fp8 cache (the kernel's
  `is_fp8()` side, which stores them as they are), `[num_tokens, num_heads, head_size]`, each token's
  `num_heads * head_size` values contiguous, any token stride (vLLM passes slices of the fused
  `qkv` projection: stride `(num_q_heads + 2 * num_heads) * head_size`);
- `key_cache`, `value_cache`: `bfloat16` or `float8e4nv`, `[num_blocks, block_size, num_heads, head_size]`, last
  dimension contiguous, any other strides (the flash backends' `kv[:, 0]`/`kv[:, 1]` of one
  `[num_blocks, 2, block_size, num_heads, head_size]` allocation; the Triton backend's
  `kv_cache.transpose(1, 2).split(head_size, -1)` of a `[num_blocks, num_heads, block_size,
  2 * head_size]` allocation, where keys and values interleave in one buffer); the value cache's
  strides those of the key cache, as the kernel assumes;
- `slot_mapping`: `int64`, `[num_tokens]`, each slot `-1` or in `[0, num_blocks * block_size)`;
  the non-negative slots distinct (vLLM gives each token its own slot);
- `k_scale`, `v_scale`: one-element `float32` tensors (vLLM's `layer._k_scale`), read with an fp8
  cache and `bfloat16` inputs (the test's `key.amax() / 64`, `value.amax() / 64`; calibrated
  scales in serving);
- the code base's shapes: `num_heads` 8 in the test (a rank's KV heads: 1 to 32 in models, e.g.
  8 for Llama-3), `head_size` 64, 80 and 256 in the test (128 in most models), `block_size` 8,
  16 and 32 in the test (vLLM's default is 16), `num_blocks` 512 and 5000 in the test (up to the
  cache's size in serving), `num_tokens` 21 in the test, up to `max_num_batched_tokens` in
  serving (thousands).

The head-major layout stays in the source; it is not part of the setup. Neither are `float16` or
`float32` inputs, a `float8e5` cache, per-head scales (`kv_scale_type="attn_head"`, which the
test runs only with the CUDA implementation) and the per-token-head quantised path (another
kernel).

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 33 launches through the host,
each with the caches carved from one random-bit allocation between guard regions. With a
`bfloat16` cache (19 launches, random 16-bit patterns, NaN payloads included): the host's 8 test
shapes in their layouts (1 to 4096 tokens, up to 100 padding slots, 4 to 32 heads, head sizes 64 to
256, blocks of 8 to 64, one and two tiles) and 11 edges (two and four tiles in both layouts,
`n = 400`, one head of size 1, block size 1, every slot used, only padding, one block, a token
stride `2n`, 4096 tokens in the Triton backend's layout); every written element is bit-exact.
With a `float8e4nv` cache (14 launches): vLLM's fp8 test shapes (21 tokens, 8 heads of 64, 80 and
256) and 256 and 4096 tokens in both layouts with the test's scales `amax / 64`; inputs with NaN,
infinities, saturating and fp8-subnormal values; other scales (`1`, `0.0137`, `−3.5`, `10^-30`,
`2^-126`, `0`, `∞`); and `float8e4nv` inputs (random bytes), stored bit for bit. Every constrained
element is one of the fp8 values `Post` allows (the allowed deviation from the clamped quotient is
`0.021` of the output at the median and at most `0.0625` in fp8's normal range). Every other
element of each allocation, every guard, every input and the scales are unchanged. Four wrong fp8
caches (the scale left out, the scale multiplied, rounding toward zero, the value cache divided
by `k_scale`) are rejected on two launches.
