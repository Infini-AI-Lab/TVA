"""
Reshape and cache (flash layout)
================================

Writes the keys and values of a step's new tokens into the paged KV cache. Token ``t`` goes to
slot ``s = slot_mapping[t]``: page ``s // block_size``, row ``s % block_size`` of that page. A
slot of ``-1`` marks a padding token, which writes nothing. One program handles one token and
one tile of ``TILE_SIZE`` consecutive elements of its ``num_heads * head_size`` values (grid
``(num_tokens, cdiv(num_heads * head_size, TILE_SIZE))``). The kernel only stores into the
caches; it never reads them.

Layouts (the default, flash layout, ``USE_HEAD_MAJOR_LAYOUT = False``):

* ``key``, ``value``: ``[num_tokens, num_heads, head_size]``, ``bfloat16``; each token's
  ``num_heads * head_size`` values contiguous (``stride(1) = head_size``, ``stride(2) = 1``),
  the token stride ``key.stride(0)`` free (vLLM passes slices of the fused ``qkv`` projection);
* ``key_cache``, ``value_cache``: ``[num_blocks, block_size, num_heads, head_size]``,
  ``bfloat16``, last dimension contiguous, the other strides free (``block_stride``,
  ``page_stride``, ``head_stride``); the kernel addresses the value cache with the key cache's
  strides;
* ``slot_mapping``: ``[num_tokens]``, ``int64``;
* ``k_scale``, ``v_scale``: one-element ``float32`` tensors, read only with an fp8 cache.

The head-major layout (5-D key cache ``[num_blocks, num_heads, head_size // x, block_size, x]``)
and the fp8 cache (``FP8_KV_CACHE``) are other compile-time paths of the same kernel, kept as
in the source.

Provenance: vLLM v0.33.0, commit ``4ba787ec99``,
``vllm/v1/attention/ops/triton_reshape_and_cache_flash.py``:

* ``reshape_and_cache_kernel_flash``: lines 32-154, copied verbatim;
* ``triton_reshape_and_cache_flash`` (the host wrapper): lines 377-483, adapted below as
  ``reshape_and_cache_flash``;
* ``_NATIVE_KV_CACHE_DTYPES`` and ``_is_supported_kv_cache_dtype``: lines 16-29, adapted (the set
  moved into the function).

Changes from the source:

* ``triton_reshape_and_cache_flash.py:4-14``: the imports from vLLM (``FP8_DTYPE``,
  ``get_fp8_min_max``, ``current_platform``, ``KVQuantMode``, ``is_quantized_kv_cache``) are
  removed; the file imports ``torch``, ``triton`` and ``triton.language`` directly.
* ``triton_reshape_and_cache_flash.py:19-29``: ``_is_supported_kv_cache_dtype`` keeps the native
  dtypes and the per-tensor fp8 cache dtypes (``fp8``, ``fp8_e4m3``) in place of
  ``is_quantized_kv_cache`` (``vllm/v1/kv_cache_interface.py:152``), and the device checks for an
  H200 (capability 9.0: both hold). The module-level set ``_NATIVE_KV_CACHE_DTYPES`` (line 16)
  is a local of ``_is_supported_kv_cache_dtype``, and ``FP8_MIN, FP8_MAX`` (line 14, unused by
  this kernel) are dropped: the file has no module-level assignments.
* ``triton_reshape_and_cache_flash.py:423-427``: ``current_platform.fp8_dtype()`` is
  ``torch.float8_e4m3fn``, its value on an H200.
* ``triton_reshape_and_cache_flash.py:440-448``: the CUDA branch of the wrapper's choice is taken
  (the H200 is neither ROCm nor XPU, and its capability is 9, so ``TILE_SIZE`` is not capped at
  512): ``TILE_SIZE = min(2048, next_power_of_2(num_heads * head_size))``, ``num_warps = 16``,
  ``num_stages = 10``. There is no autotuning in the source.
* The wrapper's per-token-head quantised path and the other kernels of the module
  (``_reshape_cache_per_token_head``, ``reshape_and_cache_kernel_flash_diffkv``) are not copied.
"""

import torch

import triton
import triton.language as tl


def _is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in {"fp8", "fp8_e4m3"}


def _is_supported_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    _NATIVE_KV_CACHE_DTYPES = {"auto", "float16", "bfloat16", "float32", "half", "float"}
    return kv_cache_dtype in _NATIVE_KV_CACHE_DTYPES or _is_quantized_kv_cache(kv_cache_dtype)


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    dim_stride_k: tl.int64,
    dim_stride_v: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    x: tl.constexpr,
    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
    # >0 writes V interleaved as [Block, Head, Slot//kx, Dim, kx] (head-major
    # layout only); 0 keeps the standard [Block, Head, Dim, Slot] V layout.
    INTERLEAVED_V_PACK_FACTOR: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    if USE_HEAD_MAJOR_LAYOUT:
        # Decompose the tile index back into head and dim coordinates.
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        # Value addressing: standard 4D [Block, Head, Dim, Slot], or the
        # interleaved 5D [Block, Head, Slot//kx, Dim, kx] the fused AITER
        # writer emits and the interleaved read path expects.
        if INTERLEAVED_V_PACK_FACTOR > 0:
            tgt_idx_v = (
                block_idx * block_stride
                + cur_head * head_stride
                + (block_offset // INTERLEAVED_V_PACK_FACTOR)
                * (head_size * INTERLEAVED_V_PACK_FACTOR)
                + cur_dim * INTERLEAVED_V_PACK_FACTOR
                + (block_offset % INTERLEAVED_V_PACK_FACTOR)
            )
        else:
            tgt_idx_v = (
                block_idx * block_stride
                + cur_head * head_stride
                + cur_dim * dim_stride_v
                + block_offset * 1
            )
        # Key addressing (5D): [Block, Head, Dim//8, Slot, 8]
        tgt_idx_k = (
            block_idx * block_stride
            + cur_head * head_stride
            + (cur_dim // x) * dim_stride_k
            + block_offset * x
            + (cur_dim % x)
        )
    else:
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        tgt_idx_k = (
            block_idx * block_stride
            + block_offset * page_stride
            + cur_head * head_stride
            + cur_dim
        )
        tgt_idx_v = tgt_idx_k

    # [TILE_SIZE]
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        # tl.store will do the correct implicit cast to fp8,
        # based on the key_cache_ptr.dtype.element_ty
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            # tl.store will do the correct implicit cast to fp8,
            #  based on the value_cache_ptr.dtype.element_ty
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    tl.store(
        key_cache_ptr + tgt_idx_k,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx_v,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


def reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # [num_blocks, block_size, num_heads, head_size]
    key_cache: torch.Tensor,
    # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
    interleaved_v_pack_factor: int = 0,  # >0: interleave V (head-major); 0: standard
):
    """The host wrapper, from ``triton_reshape_and_cache_flash`` (lines 377-483). The caches are
    the caller's tensors: the kernel stores the new tokens' rows into them and reads nothing
    else of them."""
    num_heads = key.shape[1]
    head_size = key.shape[2]

    use_head_major_layout = key_cache.ndim == 5
    if interleaved_v_pack_factor and not use_head_major_layout:
        raise NotImplementedError(
            "Interleaved V-cache write is only implemented for the head-major "
            "(5D) cache layout; got a non-head-major cache with "
            f"interleaved_v_pack_factor={interleaved_v_pack_factor}."
        )
    if use_head_major_layout:
        block_size = key_cache.shape[3]
        x = key_cache.shape[4]
        head_stride = key_cache.stride(1)
        dim_stride_k = key_cache.stride(2)
        dim_stride_v = value_cache.stride(2)
    else:
        block_size = key_cache.shape[1]
        x = 1
        dim_stride_k = 0
        dim_stride_v = 0
        head_stride = key_cache.stride()[2]
    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"Triton reshape-and-cache cannot store kv_cache_dtype={kv_cache_dtype} "
        f"on this device: an FP8 KV cache needs native fp8e4nv (SM89+). Use "
        f"--kv-cache-dtype bfloat16 (or float16 on SM75)."
    )
    kv_cache_torch_dtype = (
        torch.float8_e4m3fn  # current_platform.fp8_dtype() on an H200
        if _is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and _is_quantized_kv_cache(
        kv_cache_dtype
    ):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    FP8_KV_CACHE = _is_quantized_kv_cache(kv_cache_dtype)
    # heuristics instead of autotuning; the CUDA branch, capability 9 (H200)
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    num_stages = 10
    num_warps = 16

    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=use_head_major_layout,
        INTERLEAVED_V_PACK_FACTOR=interleaved_v_pack_factor,
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def make_caches(layout, num_blocks, block_size, num_heads, head_size, device, generator):
    """Key and value caches as vLLM's backends allocate them, filled with random values.

    * ``"flash"``: one ``[num_blocks, 2, block_size, num_heads, head_size]`` allocation, split
      into ``[:, 0]`` and ``[:, 1]`` (``create_kv_caches_with_random_flash``,
      ``vllm/utils/torch_utils.py:598``, the NHD layout);
    * ``"triton_attn"``: one ``[num_blocks, num_heads, block_size, 2 * head_size]`` allocation,
      transposed to ``[num_blocks, block_size, num_heads, 2 * head_size]`` and split along the
      last dimension (``vllm/v1/attention/backends/triton_attn.py:828``).
    """
    scale = head_size**-0.5
    if layout == "flash":
        kv = torch.rand((num_blocks, 2, block_size, num_heads, head_size), generator=generator)
        kv = (kv * 2 * scale - scale).to(device=device, dtype=torch.bfloat16)
        return kv[:, 0], kv[:, 1]
    assert layout == "triton_attn"
    kv = torch.rand((num_blocks, num_heads, block_size, 2 * head_size), generator=generator)
    kv = (kv * 2 * scale - scale).to(device=device, dtype=torch.bfloat16)
    key_cache, value_cache = kv.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache


def reference(key, value, key_cache, value_cache, slot_mapping):
    """The caches after the launch, from copies of the caches before it: each non-padding
    token's rows written at its slot, every other element unchanged."""
    block_size = key_cache.shape[1]
    ref_k = key_cache.clone()
    ref_v = value_cache.clone()
    for t, s in enumerate(slot_mapping.tolist()):
        if s < 0:
            continue
        ref_k[s // block_size, s % block_size] = key[t]
        ref_v[s // block_size, s % block_size] = value[t]
    return ref_k, ref_v


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (num_tokens, num_heads, head_size, block_size, num_blocks, layout, padding tokens)
    # tests/kernels/attention/test_cache.py:19-36 (42 // 2 tokens, 8 heads, head sizes 64, 80,
    # 256, block sizes 8, 16, 32, 1024 // 2 and 10000 // 2 blocks), and model shapes
    # (8 KV heads of 128: Llama-3; 32 KV heads of 128: Llama-2-7B; 4 of 96).
    for num_tokens, num_heads, head_size, block_size, num_blocks, layout, num_pad in [
        (21, 8, 64, 16, 512, "flash", 0),
        (21, 8, 80, 8, 512, "flash", 0),
        (21, 8, 256, 32, 5000, "flash", 0),
        (1, 8, 128, 16, 64, "flash", 0),
        (1, 8, 128, 16, 64, "triton_attn", 1),
        (256, 8, 128, 16, 64, "triton_attn", 64),
        (300, 4, 96, 64, 20, "triton_attn", 17),
        (4096, 32, 128, 16, 300, "flash", 100),
    ]:
        num_slots = num_blocks * block_size
        slots = torch.randperm(num_slots, generator=gen)[:num_tokens]
        pad = torch.randperm(num_tokens, generator=gen)[:num_pad]
        slots[pad] = -1
        slot_mapping = slots.to(device=DEVICE, dtype=torch.long)
        # key and value are slices of the fused qkv projection, as in vLLM's models
        qkv = torch.randn((num_tokens, 3, num_heads, head_size), generator=gen)
        qkv = qkv.to(device=DEVICE, dtype=torch.bfloat16)
        _, key, value = qkv.unbind(dim=1)
        key_cache, value_cache = make_caches(layout, num_blocks, block_size, num_heads, head_size,
                                             DEVICE, gen)
        ref_k, ref_v = reference(key, value, key_cache, value_cache, slot_mapping)
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
        reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", k_scale,
                                v_scale)
        ok = torch.equal(key_cache, ref_k) and torch.equal(value_cache, ref_v)
        print(f"tokens {num_tokens} (padding {num_pad}) heads {num_heads} head_size {head_size} "
              f"block {block_size} blocks {num_blocks} layout {layout}: caches "
              f"{'equal' if ok else 'DIFFER'} to the reference")
        assert ok
    print("✅ reshape_and_cache_flash matches the reference (bit-exact)")
