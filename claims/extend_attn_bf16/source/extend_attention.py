"""
Extend attention over a paged KV cache (bfloat16)
=================================================

Attention for the new ("extend") tokens of a batch of requests, each over its cached prefix and
its own new tokens. Request ``b`` has ``ext_b = qo_indptr[b + 1] - qo_indptr[b]`` new tokens,
rows ``qo_indptr[b] ... qo_indptr[b + 1] - 1`` of ``q_extend``, ``k_extend``, ``v_extend`` and
the output, and a prefix of ``pre_b = kv_indptr[b + 1] - kv_indptr[b]`` cached tokens whose slots
in ``k_buffer``, ``v_buffer`` are ``kv_indices[kv_indptr[b] + j]`` (``j < pre_b``; page size 1).
The new token ``i`` of request ``b`` (sequence position ``pre_b + i``) attends to the prefix and
to the new tokens ``0 ... i`` (with a custom mask, target verification: those its row of the mask
allows); with a sliding window ``W > 0``, only to positions ``p`` with ``pre_b + i <= p + W``.
Query head ``h`` reads key/value head ``h // kv_group_num`` (grouped-query attention).

One program handles one request, one query head and one block of ``BLOCK_M`` new tokens (grid
``(batch, num_heads, cdiv(max_len_extend, BLOCK_M))``; a program whose block starts past its
request's new tokens returns at once). It runs an online softmax (running row maximum
``e_max``, unnormalised sum ``deno`` and accumulator ``acc``) first over the prefix in blocks of
``BLOCK_N`` cached tokens, gathered through ``kv_indices`` (stage 1), then over the request's new
tokens up to the end of its own block, causally or with the custom mask (stage 2), with the
window mask, skipping tiles it empties; ``tl.dot`` in ``float32``, the probabilities cast to
``bfloat16`` before the second ``tl.dot``; finally ``acc / deno``.

Layouts (last dimension contiguous): ``q_extend`` ``[total_extend_tokens, num_heads, head_dim]``;
``k_extend``, ``v_extend`` ``[total_extend_tokens, num_kv_heads, head_dim]``; ``k_buffer``,
``v_buffer`` (the layer's KV cache) ``[num_slots, num_kv_heads, head_dim]``; all ``bfloat16``,
token and head strides free; ``qo_indptr`` ``[batch + 1]``, ``int64`` (an extend batch) or
``int32`` (target verification); ``kv_indptr`` ``[batch + 1]``, ``int32``, and ``kv_indices``,
``int64`` (a sliding-window layer's window); ``custom_mask`` ``bool``, request ``b``'s
``[ext_b, pre_b + ext_b + window_kv_offsets[b]]`` block (offset ``0`` without a window) at
``mask_indptr[b]`` (``int64``); ``window_kv_offsets`` ``[batch]``, ``int64``; the output
``o_extend`` ``[total_extend_tokens, num_heads, head_dim]``, ``torch.empty_like(q_extend)``.

Provenance: SGLang v0.5.23, commit ``297e54fd10`` (``297e54fd105e974cdf401e5b4d20253a22c16f90``),
``python/sglang/kernels/ops/attention/extend_attention.py``:

* ``tanh`` (a ``@triton.jit`` helper the kernel names only under ``logit_cap > 0``): lines
  207-210, copied verbatim;
* ``_fwd_kernel``: lines 374-913, copied verbatim except for the parameter defaults (below);
* ``_get_block_sizes_for_extend_attention``: lines 48-142; ``extend_attention_fwd`` (the host
  wrapper): lines 916-1168, both adapted below;
* ``_extract_kv_strides``: ``python/sglang/kernels/ops/attention/decode_attention.py:212-225``;
* the caller, ``TritonAttnBackend.forward_extend``:
  ``python/sglang/srt/layers/attention/triton_backend.py:1669-1916``, and the metadata it reads
  (``triton_backend.py:1006-1128``: ``kv_indices`` ``int64``, ``qo_indptr`` ``int64``
  (``:398-402``; ``int32`` for target verification, ``:1009-1015``), ``kv_indptr`` ``int32``).

Changes from the source:

* ``extend_attention.py:437-444``: the defaults of ``_fwd_kernel``'s last eight parameters
  (``IS_GFX1250: tl.constexpr = False``, ``PAGE_SIZE: tl.constexpr = 1``,
  ``IDENTITY_KV_INDICES: tl.constexpr = False``, ``SCORE_MOD: tl.constexpr = None``,
  ``Aux0=None``, ``aux0_stride_t=0``, ``aux0_stride_h=0``, ``aux0_len=0``) are removed: TVA's
  subset has no kernel parameter defaults. The host passes each of them, as the wrapper does.
* ``extend_attention.py:19-45``: the imports of ``sglang`` and the module's device globals are
  replaced by locals of the host functions with their values on an H200: ``_is_cuda = True``,
  ``_is_hip = _is_gfx95 = _is_gfx1250 = False``, ``CUDA_CAPABILITY = (9, 0)``; the file has no
  module-level assignments. ``_extract_kv_strides`` is copied as a host function;
  ``unpack_aux_tensors(None, None)`` is ``(None, 0, 0, 0)`` (``score_mod.py:30-32``).
* ``extend_attention.py:48-142``: ``_get_block_sizes_for_extend_attention`` keeps only the CUDA
  branches the H200 takes (capability 9): ``BLOCK_M, BLOCK_N = (128, 64)`` for ``Lq <= 128`` and
  ``(64, 64)`` for ``Lq <= 256``, ``num_warps = 4 if Lq <= 64 else 8``; ``BLOCK_DMODEL`` and
  ``BLOCK_DPE`` as the source computes them.
* ``extend_attention.py:916-1168``: ``extend_attention_fwd`` takes the arguments that differ
  between the backend's extend batches and target verifications, with or without a sliding
  window (``custom_mask``, ``mask_indptr``, ``sliding_window_size``, ``window_kv_offsets``,
  keyword arguments after ``sm_scale``; ``triton_backend.py:1888-1915``), and fixes the others at
  the backend's values: ``is_causal = True``, ``k_scale = v_scale = 1.0`` (a ``bfloat16`` cache
  with no scales, ``triton_backend.py:1835-1840``), ``logit_cap = 0.0``, ``sinks = None``,
  ``xai_temperature_len = -1``, ``skip_prefix_custom_mask = True`` (the wrapper's default),
  ``lse_extend = None``, ``skip_prefix = skip_extend = False``, ``page_size = 1``,
  ``score_mod = aux_tensors = None``, ``identity_kv_indices = False``; and it allocates the output
  ``o_extend = torch.empty_like(q_extend)`` and returns it, where the source's wrapper takes it from
  the caller (``forward_extend`` allocates it so, ``triton_backend.py:1690-1698``, for equal head
  dims). The fp8 zero-prefix cast (``extend_attention.py:989-1010``), the compact tile grid (HIP
  only, ``:1056-1074``) and the HIP launch options (``:1080-1082``) are not taken on an H200 and
  are left out; ``USE_FP8_PREFIX``, ``USE_FP8_EXTEND``, ``USE_EXP2`` and ``use_compact_tile_grid``
  are ``False`` (each needs ``_is_gfx95`` or ``_is_hip``), ``FP8_MAX = 1.0``, and
  ``num_stages = 1`` (``kimi_k3_shape`` needs 12 heads with head dims 192/128 or 576/512, which
  this file does not launch). The kernel launch itself (``extend_attention.py:1095-1168``) is
  unchanged.
* ``reference`` (attention in ``float64``, request by request, causal or with the custom mask,
  and the window) and the test under ``if __name__ == "__main__":`` (after
  ``test/registered/attention/test_triton_attention_kernels.py:172-325, 533-643``) are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    LSE_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    k_scale,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    compact_batch_size,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_LSE: tl.constexpr,
    SKIP_PREFIX: tl.constexpr,
    SKIP_EXTEND: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_COMPACT_TILE_GRID: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_FP8_PREFIX: tl.constexpr,
    USE_FP8_EXTEND: tl.constexpr,
    FP8_MAX: tl.constexpr,
    IS_GFX1250: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IDENTITY_KV_INDICES: tl.constexpr,
    SCORE_MOD: tl.constexpr,
    Aux0,
    aux0_stride_t,
    aux0_stride_h,
    aux0_len,
):
    if USE_COMPACT_TILE_GRID:
        output_tile = tl.program_id(0)
        cur_head = tl.program_id(1)

        cur_seq = tl.full((), 0, tl.int64)
        cum_tiles = tl.full((), 0, tl.int64)
        found = tl.full((), 0, tl.int32)
        while (cur_seq < compact_batch_size) & (found == 0):
            seq_q_start = tl.load(qo_indptr + cur_seq)
            seq_q_end = tl.load(qo_indptr + cur_seq + 1)
            seq_q_len = seq_q_end - seq_q_start
            seq_tiles = (seq_q_len + BLOCK_M - 1) // BLOCK_M
            next_cum_tiles = cum_tiles + seq_tiles
            if next_cum_tiles > output_tile:
                found = 1
            else:
                cum_tiles = next_cum_tiles
                cur_seq = cur_seq + 1

        if found == 0:
            return

        cur_block_m = output_tile - cum_tiles
    else:
        cur_seq = tl.program_id(0)
        cur_head = tl.program_id(1)
        cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    # Grid axis 2 spans the batch-max extend length; all stores are masked by mask_m.
    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # For SWA, we should only load the mask in the sliding window
    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)
    # The FP8 prefix sweep can use a wider tile than the current-token sweep.
    offs_n_prefix = tl.arange(0, BLOCK_N_PREFIX)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    prefix_end = 0 if SKIP_PREFIX else cur_seq_len_prefix
    for start_n in range(0, prefix_end, BLOCK_N_PREFIX):
        start_n = tl.multiple_of(start_n, BLOCK_N_PREFIX)
        mask_n = (start_n + offs_n_prefix) < cur_seq_len_prefix

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + start_n
                + offs_n_prefix[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            # The mask buffer is int8; narrow it to i1, since a score_mod hands
            # final_mask to tl.load, which rejects an i8 mask.
            final_mask &= custom_mask != 0
        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            # q_id = prefix_len + cur_m, kv_id = cur_n
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n_prefix[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            if IDENTITY_KV_INDICES:
                offs_kv_loc = cur_seq_kv_start_idx + start_n + offs_n_prefix
            else:
                offs_kv_loc = tl.load(
                    kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix,
                    mask=mask_n,
                    other=0,
                )

            # Page-aware KV address math. At PAGE_SIZE==1
            # (legacy / non-shared / shared-at-ps=1), Triton specializes
            # the else-branch away — byte-identical SASS to today.
            if PAGE_SIZE == 1:
                # load k in transposed way
                offs_buf_k = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            else:
                page_id = offs_kv_loc // PAGE_SIZE
                tok_in_p = offs_kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )
            # gfx1250: triton tl.dot(fp8, fp8) returns garbage (~1e34+) for contraction
            # dim K>=128 (K=64 ok). This prefix read fires when a radix-cache prefix is
            # reused (prefill reads the cached fp8 KV), and the MLA nope dot has K=512,
            # so we must upcast the fp8 K to q's dtype and dot in bf16 rather than
            # downcasting q to fp8. No-op for a bf16 cache. (Do NOT revert to q.to(fp8).)
            # On all other platforms keep the original q.to(k.dtype) downcast.
            # TODO: remove this branch once the gfx1250 fp8 tl.dot issue is resolved.
            if IS_GFX1250:
                qk = tl.dot(q, k.to(q.dtype))
            else:
                qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_kpe = (
                        offs_kv_loc[None, :] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                else:
                    offs_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                if IS_GFX1250:
                    qk += tl.dot(qpe, kpe.to(qpe.dtype))
                else:
                    qk += tl.dot(qpe.to(kpe.dtype), kpe)
            if USE_EXP2:
                qk *= sm_scale * k_scale * LOG2E
            else:
                qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m)[:, None],
                    start_n + offs_n_prefix[None, :],
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m)[
                        :, None
                    ],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            if USE_EXP2:
                re_scale = tl.exp2(e_max - n_e_max)
                p = tl.exp2(qk - n_e_max[:, None])
            else:
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if PAGE_SIZE == 1:
                offs_buf_v = (
                    offs_kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            else:
                offs_buf_v = (
                    page_id[:, None] * stride_buf_vpage
                    + tok_in_p[:, None] * stride_buf_vtok
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            if USE_FP8_PREFIX:
                p_dot = (p * FP8_MAX).to(v.dtype)
                acc = acc * re_scale[:, None] + tl.dot(p_dot, v) * (v_scale / FP8_MAX)
            else:
                # keep softmax weights p in fp32 for the P·V dot (do not downcast to bf16)
                # on gfx1250; on other platforms restore the original p.to(v.dtype) cast.
                # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
                if IS_GFX1250:
                    dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
                else:
                    dot = tl.dot(p.to(v.dtype), v)
                acc = acc * re_scale[:, None] + dot * v_scale

            e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    extend_end = 0 if SKIP_EXTEND else cur_block_m_end
    # Causal or not, the mask below keeps (q, kv) only if q <= kv + SLIDING_WINDOW_SIZE,
    # so no tile under this floor holds an unmasked element -- tight for any
    # BLOCK_M/BLOCK_N. SKIP_TILE already zeroes them; bounding the loop also drops
    # their cross-wave tl.max.
    extend_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        extend_start = (
            tl.maximum(cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0) // BLOCK_N
        ) * BLOCK_N
    for start_n in range(extend_start, extend_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            # int8 -> i1 before the &=; see the prefix loop above.
            custom_mask = (custom_mask != 0) & mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            if not IS_CAUSAL:
                window_mask = (
                    (cur_block_m * BLOCK_M + offs_m[:, None])
                    <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
                ) & (
                    (start_n + offs_n[None, :])
                    <= (cur_block_m * BLOCK_M + offs_m[:, None] + SLIDING_WINDOW_SIZE)
                )
            else:
                window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                    start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
                )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Extend + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe)

            if USE_EXP2:
                qk *= sm_scale * LOG2E
            else:
                qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m)[:, None],
                    cur_seq_len_prefix + start_n + offs_n[None, :],
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m)[
                        :, None
                    ],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            if USE_EXP2:
                re_scale = tl.exp2(e_max - n_e_max)
                p = tl.exp2(qk - n_e_max[:, None])
            else:
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            if USE_FP8_EXTEND:
                p_dot = (p * FP8_MAX).to(v.dtype)
                acc = acc * re_scale[:, None] + tl.dot(p_dot, v) * (1.0 / FP8_MAX)
            else:
                # keep softmax weights p in fp32 for the P·V dot (do not downcast to bf16)
                # on gfx1250; on other platforms restore the original p.to(v.dtype) cast.
                # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
                if IS_GFX1250:
                    dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
                else:
                    dot = tl.dot(p.to(v.dtype), v)
                acc = acc * re_scale[:, None] + dot

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        if USE_EXP2:
            deno += tl.exp2(cur_sink * LOG2E - e_max)
        else:
            deno += tl.exp(cur_sink - e_max)

    # A ragged prefix chunk can be empty for some requests. Represent an empty
    # partial as output=0 and LSE=-inf so merge_state ignores it exactly.
    no_kv = deno == 0.0

    if STORE_LSE:
        offs_lse = (
            cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m
        ) * stride_lse_bs + cur_head * stride_lse_h
        if USE_EXP2:
            lse = tl.log(deno) + e_max * LN2
        else:
            lse = tl.log(deno) + e_max
        lse = tl.where(no_kv, float("-inf"), lse)
        tl.store(LSE_Extend + offs_lse, lse, mask=mask_m)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    deno_safe = tl.where(no_kv, 1.0, deno)
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno_safe[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno_safe[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """extend_attention.py:48-142 on an H200 (CUDA capability 9): the block sizes and
    num_warps."""
    _is_cuda = True
    CUDA_CAPABILITY = (9, 0)  # torch.cuda.get_device_capability() on an H200
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(Lv)

    if _is_cuda and CUDA_CAPABILITY[0] >= 9:
        # Hopper architecture (H100, etc.)
        if Lq <= 128:
            BLOCK_M, BLOCK_N = (128, 64)
        elif Lq <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)

    num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


def _extract_kv_strides(buf, page_size: int):
    """decode_attention.py:212-225."""
    if buf.ndim != 3:
        raise ValueError(f"unexpected KV buffer ndim={buf.ndim}, shape={buf.shape}")
    slot_stride = buf.stride(0)
    head_stride = buf.stride(1)
    page_stride = slot_stride * page_size
    tok_stride = slot_stride
    return slot_stride, head_stride, page_stride, tok_stride


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
    sm_scale=None,
    custom_mask=None,
    mask_indptr=None,
    sliding_window_size=-1,
    window_kv_offsets=None,
):
    """
    q_extend, k_extend, v_extend: [total_extend_tokens, heads, head_dim], the new tokens
    k_buffer, v_buffer: [num_slots, num_kv_heads, head_dim], the KV cache
    qo_indptr (int64 for an extend batch, int32 for target verification), kv_indptr (int32):
    [batch + 1]; kv_indices (int64): the prefix slots
    custom_mask (bool), mask_indptr (int64): target verification's tree mask, request b's
    [extend_b, prefix_b + extend_b (+ its window offset)] block at mask_indptr[b], or None
    sliding_window_size: the layer's window (> 0), or -1; window_kv_offsets (int64): the first
    position of each request's window, read only with a custom mask
    returns o_extend: [total_extend_tokens, heads, head_dim], allocated here

    The other flags of SGLang's Triton attention backend (triton_backend.py:1888-1915): causal,
    no logit cap, no sinks, no xai temperature, no LSE, page size 1, no score_mod,
    k_scale = v_scale = 1.0.
    """
    # extend_attention.py:31-37, on an H200
    _is_hip = False
    _is_gfx95 = False
    _is_gfx1250 = False
    is_causal = True
    k_scale = 1.0
    v_scale = 1.0
    logit_cap = 0.0
    skip_prefix_custom_mask = True
    sinks = None
    xai_temperature_len = -1
    lse_extend = None
    skip_prefix = False
    skip_extend = False
    page_size = 1
    score_mod = None
    identity_kv_indices = False
    # triton_backend.py:1698 (equal head dims)
    o_extend = torch.empty_like(q_extend)

    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    # Get block sizes and configuration for the generic fallback.
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    HAS_SINK = sinks is not None
    USE_FP8_PREFIX = False  # needs _is_gfx95
    USE_FP8_EXTEND = False  # needs _is_gfx95
    FP8_MAX = 1.0
    BLOCK_N_ARCH = BLOCK_N
    FP8_BLOCK_N = 128 if BLOCK_N_ARCH < 128 and Lq <= 192 else BLOCK_N_ARCH
    BLOCK_N = FP8_BLOCK_N if USE_FP8_EXTEND else BLOCK_N_ARCH
    BLOCK_N_PREFIX = FP8_BLOCK_N if USE_FP8_PREFIX else BLOCK_N_ARCH
    USE_EXP2 = False  # needs _is_gfx95
    STORE_LSE = lse_extend is not None
    stride_lse_bs = lse_extend.stride(0) if STORE_LSE else 0
    stride_lse_h = lse_extend.stride(1) if STORE_LSE else 0

    use_compact_tile_grid = False  # needs _is_hip
    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    # score_mod.py:30-32 (score_mod is None)
    aux0, aux0_stride_t, aux0_stride_h, aux0_len = None, 0, 0, 0

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        lse_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        k_scale,
        v_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        stride_lse_bs,
        stride_lse_h,
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        batch_size,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_N_PREFIX=BLOCK_N_PREFIX,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        STORE_LSE=STORE_LSE,
        SKIP_PREFIX=skip_prefix,
        SKIP_EXTEND=skip_extend,
        HAS_SINK=HAS_SINK,
        IS_GFX1250=_is_gfx1250,
        STORE_TRANSPOSE=_is_hip,
        USE_COMPACT_TILE_GRID=use_compact_tile_grid,
        USE_EXP2=USE_EXP2,
        USE_FP8_PREFIX=USE_FP8_PREFIX,
        USE_FP8_EXTEND=USE_FP8_EXTEND,
        FP8_MAX=FP8_MAX,
        PAGE_SIZE=page_size,
        IDENTITY_KV_INDICES=identity_kv_indices,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o_extend


def reference(q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, sm_scale=None,
              custom_mask=None, mask_indptr=None, sliding_window_size=-1,
              window_kv_offsets=None):
    """Attention in float64, request by request: each new token over the request's prefix (its
    slots kv_indices[kv_indptr[b]:kv_indptr[b+1]]) and, causally (or as its row of the custom
    mask allows), its own new tokens; with a window W > 0, only positions p >= pre + i - W."""
    H, HKV, D = q.shape[1], k.shape[1], q.shape[2]
    group = H // HKV
    sigma = sm_scale or D ** -0.5
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for b in range(qo_indptr.shape[0] - 1):
        q0, q1 = int(qo_indptr[b]), int(qo_indptr[b + 1])
        k0, k1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
        n, pre = q1 - q0, k1 - k0
        if n == 0:
            continue
        slots = kv_indices[k0:k1]
        kf = torch.cat([k_buffer[slots], k[q0:q1]]).double().repeat_interleave(group, dim=1)
        vf = torch.cat([v_buffer[slots], v[q0:q1]]).double().repeat_interleave(group, dim=1)
        qs = q[q0:q1].double().transpose(0, 1)  # [H, n, D]
        s = qs @ kf.permute(1, 2, 0) * sigma  # [H, n, pre + n]
        pos = torch.arange(pre + n, device=q.device)
        row = pre + torch.arange(n, device=q.device)
        if custom_mask is None:
            allowed = pos[None, :] <= row[:, None]
        else:
            off = int(window_kv_offsets[b]) if sliding_window_size > 0 else 0
            m0 = int(mask_indptr[b])
            m = custom_mask[m0:m0 + n * (pre + n + off)].view(n, pre + n + off)[:, off + pre:]
            allowed = torch.cat([torch.ones(n, pre, dtype=torch.bool, device=q.device), m], 1)
        if sliding_window_size > 0:
            allowed &= row[:, None] <= pos[None, :] + sliding_window_size
        s = s.masked_fill(~allowed, float("-inf"))
        out[q0:q1] = (torch.softmax(s, dim=-1) @ vf.transpose(0, 1)).transpose(0, 1)
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # test/registered/attention/test_triton_attention_kernels.py:172-325 runs 19 requests,
    # 12 heads on 4 KV heads, head dims 256, 128, 96, 80 and 13, prefix and extend lengths
    # drawn from [1, 6165); here with fewer, shorter requests, the backend's index dtypes, and
    # each request's prefix slots scattered through the cache
    for H, HKV, D in [(12, 4, 256), (12, 4, 128), (12, 4, 96), (12, 4, 80), (12, 4, 13),
                      (32, 8, 128), (8, 1, 64)]:
        B = 5
        pre = torch.randint(0, 300, (B,), generator=gen)
        ext = torch.randint(1, 300, (B,), generator=gen)
        pre[0], ext[1] = 0, 1
        slots_total = int(pre.sum()) + 7
        qo_indptr = torch.zeros(B + 1, dtype=torch.int64)
        qo_indptr[1:] = torch.cumsum(ext, 0)
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(pre, 0)
        kv_indices = torch.randperm(slots_total, generator=gen)[: int(pre.sum())]
        T = int(ext.sum())
        q = torch.randn((T, H, D), generator=gen).to(DEVICE, torch.bfloat16)
        k = torch.randn((T, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        v = torch.randn((T, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        kb = torch.randn((slots_total, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        vb = torch.randn((slots_total, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        qo_indptr, kv_indptr = qo_indptr.to(DEVICE), kv_indptr.to(DEVICE)
        kv_indices = kv_indices.to(DEVICE, torch.int64)
        o = extend_attention_fwd(q, k, v, kb, vb, qo_indptr, kv_indptr, kv_indices,
                                 int(ext.max()))
        ref = reference(q, k, v, kb, vb, qo_indptr, kv_indptr, kv_indices)
        err = (o.double() - ref).abs().max().item()
        print(f"heads {H}/{HKV} head_dim {D} prefix {pre.tolist()} extend {ext.tolist()}: "
              f"max abs error {err:.2e}")
        assert err < 2e-2, err
    # a sliding-window layer (the code base's test_extend_attention_sliding_window, window 127,
    # test_triton_attention_kernels.py:533-643), and target verification: int32 qo_indptr,
    # 8 draft tokens per request with a tree mask (each token sees its ancestors and itself)
    for W, verify in [(127, False), (-1, True), (127, True)]:
        H, HKV, D, B = 16, 8, 128, 4
        seq = torch.randint(1, 600, (B,), generator=gen)
        n = 8 if verify else 200
        pre = torch.minimum(seq, torch.tensor(W)) if W > 0 else seq
        ext = torch.full((B,), n)
        qo_indptr = torch.zeros(B + 1, dtype=torch.int32 if verify else torch.int64)
        qo_indptr[1:] = torch.cumsum(ext, 0)
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(pre, 0)
        slots_total = int(pre.sum()) + 7
        kv_indices = torch.randperm(slots_total, generator=gen)[: int(pre.sum())]
        T = int(ext.sum())
        q = torch.randn((T, H, D), generator=gen).to(DEVICE, torch.bfloat16)
        k = torch.randn((T, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        v = torch.randn((T, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        kb = torch.randn((slots_total, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        vb = torch.randn((slots_total, HKV, D), generator=gen).to(DEVICE, torch.bfloat16)
        offs = (seq - pre).to(torch.int64)
        mask = mask_indptr = None
        if verify:
            parent = [-1] + [int(torch.randint(0, j, (1,), generator=gen)) for j in range(1, n)]
            tree = torch.zeros(n, n, dtype=torch.bool)
            for j in range(n):
                a = j
                while a >= 0:
                    tree[j, a] = True
                    a = parent[a]
            rows = [torch.cat([torch.ones(n, int(seq[b]), dtype=torch.bool), tree], 1).flatten()
                    for b in range(B)]
            mask = torch.cat(rows).to(DEVICE)
            mask_indptr = torch.zeros(B + 1, dtype=torch.int64)
            mask_indptr[1:] = torch.cumsum(n * (seq + n), 0)
            mask_indptr = mask_indptr.to(DEVICE)
        qo_indptr, kv_indptr = qo_indptr.to(DEVICE), kv_indptr.to(DEVICE)
        kv_indices, offs = kv_indices.to(DEVICE, torch.int64), offs.to(DEVICE)
        kw = dict(custom_mask=mask, mask_indptr=mask_indptr, sliding_window_size=W,
                  window_kv_offsets=offs if W > 0 else None)
        o = extend_attention_fwd(q, k, v, kb, vb, qo_indptr, kv_indptr, kv_indices, n, **kw)
        ref = reference(q, k, v, kb, vb, qo_indptr, kv_indptr, kv_indices, **kw)
        err = (o.double() - ref).abs().max().item()
        print(f"window {W} {'verify' if verify else 'extend'}: max abs error {err:.2e}")
        assert err < 2e-2, err
    print("extend_attention_fwd matches the float64 reference")
