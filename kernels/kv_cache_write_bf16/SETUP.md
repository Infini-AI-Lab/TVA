# Setup of `kv_cache_write_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (`sandbox/TESTSET.md`, kernel
11), from vLLM v0.33.0 (commit `4ba787ec99`),
`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33` `reshape_and_cache_kernel_flash`.

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

What changed from the code base's source, all on the host (the module docstring lists each with
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

Flags as the wrapper sets them for the proved path: `USE_HEAD_MAJOR_LAYOUT = False` (a 4-D key
cache), `INTERLEAVED_V_PACK_FACTOR = 0`, `x = 1`, `dim_stride_k = dim_stride_v = 0`,
`FP8_KV_CACHE = False` (`kv_cache_dtype = "auto"`).

## The launches

The claim should cover the wrapper's launches with a `bfloat16` cache in the flash (NHD) layout,
the default path of vLLM's Triton attention backend (`vllm/v1/attention/backends/triton_attn.py:828-843`)
and of the code base's test (`tests/kernels/attention/test_cache.py:181`, `implementation="triton"`,
`kv_cache_dtype="auto"`, layout `NHD`), with shapes as variables:
- `key`, `value`: `bfloat16`, `[num_tokens, num_heads, head_size]`, each token's
  `num_heads * head_size` values contiguous, any token stride (vLLM passes slices of the fused
  `qkv` projection: stride `(num_q_heads + 2 * num_heads) * head_size`);
- `key_cache`, `value_cache`: `bfloat16`, `[num_blocks, block_size, num_heads, head_size]`, last
  dimension contiguous, any other strides (the flash backends' `kv[:, 0]`/`kv[:, 1]` of one
  `[num_blocks, 2, block_size, num_heads, head_size]` allocation; the Triton backend's
  `kv_cache.transpose(1, 2).split(head_size, -1)` of a `[num_blocks, num_heads, block_size,
  2 * head_size]` allocation, where keys and values interleave in one buffer); the value cache's
  strides those of the key cache, as the kernel assumes;
- `slot_mapping`: `int64`, `[num_tokens]`, each slot `-1` or in `[0, num_blocks * block_size)`;
  the non-negative slots distinct (vLLM gives each token its own slot);
- `k_scale`, `v_scale`: one-element `float32` tensors (vLLM's `layer._k_scale`), not read on this
  path;
- the code base's shapes: `num_heads` 8 in the test (a rank's KV heads: 1 to 32 in models, e.g.
  8 for Llama-3), `head_size` 64, 80 and 256 in the test (128 in most models), `block_size` 8,
  16 and 32 in the test (vLLM's default is 16), `num_blocks` 512 and 5000 in the test (up to the
  cache's size in serving), `num_tokens` 21 in the test, up to `max_num_batched_tokens` in
  serving (thousands).

The head-major layout and the fp8 cache stay in the source, unproved.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes: for 8 cases (1 to 4096
tokens, up to 100 padding slots, 4 to 32 heads, head sizes 64, 80, 96, 128 and 256, blocks of 8 to
64, both cache layouts above, `num_heads * head_size` from 384 to 4096, so one and two tiles), the
caches after the launch are bit-for-bit equal to the reference: the new tokens' rows at their
slots, every other element unchanged.
