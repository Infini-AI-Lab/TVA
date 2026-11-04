"""
Grouped-query decode attention, stage 1 (split KV)
==================================================

The first stage of decode attention for grouped queries over SGLang's token-indexed KV pool:
one query token per request, the request's keys and values split along the sequence. Request
``b`` holds the ``L = kv_indptr[b + 1] - kv_indptr[b]`` tokens whose pool slots are
``kv_indices[kv_indptr[b] : kv_indptr[b + 1]]``. With ``S = num_kv_splits[b]`` and
``L' = cdiv(cdiv(L, S), MIN_BLOCK_KV) * MIN_BLOCK_KV``, split ``s`` covers the tokens
``[s * L', min((s + 1) * L', L))``. One program handles one request, one block of up to
``BLOCK_H`` query heads that share a KV head, and one split (grid ``(batch, cdiv(num_heads,
min(BLOCK_H, kv_group_num)), max_kv_splits)``). It runs an online softmax over blocks of
``BLOCK_N`` tokens with ``tl.dot``, and, for a non-empty split, writes the split's normalised
partial output and its log-sum-exp; it writes nothing for an empty split. The second stage
(``_fwd_kernel_stage2``, not in this file) merges the splits.

Layouts:

* ``q``: ``[batch, num_heads, Lk]``, ``bfloat16``, last dimension contiguous;
* ``k_buffer``: ``[num_slots, num_kv_heads, Lk]``, ``v_buffer``: ``[num_slots, num_kv_heads, Lv]``,
  ``bfloat16``, last dimension contiguous, slot and head strides as given;
* ``kv_indptr``: ``[batch + 1]``, ``int32``; ``kv_indices``: ``int64``, the requests' slots;
* ``num_kv_splits``: ``[batch]``, ``int32``, each in ``[1, max_kv_splits]``;
* ``att_out``: ``[batch, num_heads, max_kv_splits, Lv]`` and ``att_lse``:
  ``[batch, num_heads, max_kv_splits]``, ``float32``, allocated by the host (contiguous).

``Lk`` and ``Lv`` need not be powers of two. The logit cap (``logit_cap > 0``:
``cap * tanh(qk / cap)``), the xAI temperature, the MLA layout (``HAS_MLA``, ``BLOCK_DPE``), a
score modifier, forced splits, batch rotation, PDL and gfx1250 are other compile-time paths of
the same kernel, kept as in the source.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/attention/decode_attention.py``:

* ``_extract_kv_strides`` (a host helper): lines 212-225, copied verbatim;
* ``tanh`` (a ``@triton.jit`` helper): lines 228-231, copied verbatim;
* ``_fwd_grouped_kernel_stage1``: lines 529-793, copied verbatim except for its parameters'
  defaults (below);
* ``_decode_grouped_att_m_fwd`` (the stage-1 launcher): lines 796-931, with the entry points'
  checks, defaults and dispatch (``decode_attention_fwd``, lines 1191-1316, and
  ``decode_attention_fwd_grouped``, lines 1129-1188), adapted below as
  ``decode_attention_fwd_grouped_stage1``; ``_grouped_head_tiles`` (lines 123-125) inlined.

Changes from the source:

* ``decode_attention.py:23-43``: the imports from SGLang (``unpack_aux_tensors``, ``envs``,
  ``sglang.srt.utils``), ``logging``, ``math`` and ``typing`` are removed; the module-level
  ``_is_hip`` and ``_is_gfx1250`` (lines 40-41) are ``False`` (an H200), written into the host.
* ``decode_attention.py:46, 58``: the module-level constants ``_MIN_BLOCK_KV = 32`` and
  ``_GROUPED_BLOCK_H = 16`` are locals of the host (the file has no module-level assignments),
  and ``_grouped_head_tiles`` (line 123) is inlined as
  ``triton.cdiv(head_num, min(BLOCK_H, kv_group_num))``.
* ``decode_attention.py:566-578``: the defaults of the kernel's parameters are removed
  (``HAS_MLA: tl.constexpr = False`` is now ``HAS_MLA: tl.constexpr``, ``Aux0=None`` is ``Aux0``,
  and so on for ``USE_PDL``, ``IS_GFX1250``, ``PAGE_SIZE``, ``SCORE_MOD``, ``aux0_stride_t``,
  ``aux0_stride_h``, ``aux0_len``, ``forced_kv_splits``, ``USE_FORCED``, ``uneven_batch``,
  ``ROTATE_BATCH``): TVA's subset has no kernel parameter defaults, and the host passes each
  parameter.
* ``decode_attention.py:796-931``: the host allocates ``att_out`` and ``att_lse`` fresh
  (``torch.empty``, as SGLang's Triton backend allocates ``attn_logits`` and ``attn_lse``,
  ``python/sglang/srt/layers/attention/triton_backend.py:953-970``) and returns them, where the
  source's launcher takes them from the caller. It fixes the paths the task does not take, at
  the values of the default CUDA path: no MLA (``has_mla=False``), no PDL (``use_pdl=False``), no
  score modifier (``score_mod=None``, so ``unpack_aux_tensors`` gives ``(None, 0, 0, 0)``,
  ``score_mod.py:30-32``), no forced splits and no tuned MLA geometry (``_mla_launch_plan`` gives
  ``(False, 0)`` off ROCm, lines 181-191), no batch rotation (``uneven_batch=None``, so
  ``_rotate_batch`` is ``False``, lines 49-54), ``IS_GFX1250=False``, no ROCm ``extra_kargs``.
  From the entry points it takes ``sm_scale_withk = sm_scale * k_scale`` (line 1305), the
  asserts of lines 1220-1222 (the first and the third hold by the allocation), the Lean path's
  ROCm-only branch not taken, and the dispatch of lines 1269-1293: a call with ``kv_group_num == 1`` goes
  to the other kernel, ``_fwd_kernel_stage1``, so the host asserts ``kv_group_num != 1``.
* No autotuning in the source: the launcher's own choice is kept, on the CUDA branch:
  ``BLOCK_N = 32``, ``BLOCK_H = 16``, ``MIN_BLOCK_KV = 32``, ``num_warps = 4``,
  ``num_stages = 2``.
* The stage-2 kernel, the MHA kernel and the Lean kernel of the module are not copied.
"""

import torch

import triton
import triton.language as tl


def _extract_kv_strides(buf, page_size: int):
    """Extract (slot_stride, head_stride, page_stride, tok_stride) for a 3-D
    ``[max_slots, head_num, head_dim]`` KV buffer.

    Returns a 4-tuple of ints suitable for passing as ``stride_buf_*bs``,
    ``stride_buf_*h``, ``stride_buf_*page``, ``stride_buf_*tok``.
    """
    if buf.ndim != 3:
        raise ValueError(f"unexpected KV buffer ndim={buf.ndim}, shape={buf.shape}")
    slot_stride = buf.stride(0)
    head_stride = buf.stride(1)
    page_stride = slot_stride * page_size
    tok_stride = slot_stride
    return slot_stride, head_stride, page_stride, tok_stride


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    HAS_MLA: tl.constexpr,
    USE_PDL: tl.constexpr,
    IS_GFX1250: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCORE_MOD: tl.constexpr,
    Aux0,
    aux0_stride_t,
    aux0_stride_h,
    aux0_len,
    forced_kv_splits,
    USE_FORCED: tl.constexpr,
    uneven_batch,
    ROTATE_BATCH: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    if ROTATE_BATCH:
        # Only the split rotates: the head tiles of one (request, split) read the same
        # kv_indices (and the same latent KV under MLA), so they stay on one XCD.
        if tl.load(uneven_batch) != 0:
            cur_batch = (cur_batch + split_kv_id) % tl.num_programs(0)
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = cur_batch.to(tl.int64)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    # runtime, not constexpr: it only feeds the kv_len_per_split arithmetic below, so
    # a constexpr buys nothing and costs one stage-1 variant per cuda-graph ladder
    # rung (stage-2 does need it at compile time). Any count covers any length since
    # kv_len_per_split rounds cdiv(L, S) up; short sequences leave the tail empty.
    if USE_FORCED:
        kv_splits = forced_kv_splits
    else:
        kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # Hoist loop-invariant base offsets
    base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
    if BLOCK_DPE > 0:
        base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
    if not HAS_MLA:
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        # gfx1250: triton tl.dot(fp8, fp8) returns garbage (~1e34+) for contraction
        # dim K>=128 (verified K=64 ok, K>=128 broken; bf16 fine at all K). The MLA
        # nope QK dot has K=512, so an fp8 KV cache MUST NOT be consumed as an fp8 dot
        # here: keep q in bf16 and upcast the fp8 K to bf16 for the dot. No-op for a
        # bf16 cache. (Do NOT "optimize" this back to q.to(fp8) on gfx1250.)
        # On all other platforms keep the original downcast of q to the KV dtype.
        # TODO: remove this branch once the gfx1250 fp8 tl.dot issue is resolved.
        if IS_GFX1250:
            q_k = q
        else:
            q_k = q.to(K_Buffer.dtype.element_ty)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Page-aware KV address math (see _fwd_kernel_stage1).
            if PAGE_SIZE == 1:
                offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + base_offs_k
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            if IS_GFX1250:
                qk = tl.dot(q_k, k.to(q_k.dtype))
            else:
                qk = tl.dot(q_k, k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                else:
                    offs_buf_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + base_offs_kpe
                    )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    cur_batch_seq_len - 1,
                    offs_n[None, :],
                    cur_batch,
                    cur_head[:, None],
                    mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )
            if HAS_MLA:
                v = tl.trans(k)
            else:
                if PAGE_SIZE == 1:
                    offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                else:
                    offs_buf_v = (
                        page_id[:, None] * stride_buf_vpage
                        + tok_in_p[:, None] * stride_buf_vtok
                        + base_offs_v
                    )
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # Keep the softmax weights p in fp32 for the P·V dot (do NOT downcast p to
            # bf16) on gfx1250. The bf16 downcast of p was the accuracy loss vs a torch
            # fp32 SDPA reference (recovers gfx1250 R1 GSM8K ~0.82 -> ~0.92 with
            # attention idealized). On other platforms restore the p.to(v.dtype) cast.
            # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
            if IS_GFX1250:
                acc += tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
            else:
                acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def decode_attention_fwd_grouped_stage1(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale=1.0,
    logit_cap=0.0,
    xai_temperature_len=-1,
    page_size: int = 1,
):
    """Stage 1 of ``decode_attention_fwd`` for ``kv_group_num > 1`` on the default CUDA path: the
    launch of ``_decode_grouped_att_m_fwd`` (lines 796-931). Returns ``att_out``,
    ``[batch, num_heads, max_kv_splits, Lv]``, and ``att_lse``, ``[batch, num_heads,
    max_kv_splits]``."""
    # module constants of decode_attention.py (lines 40-41, 46, 58), on an H200
    _is_hip = False
    _is_gfx1250 = False
    _MIN_BLOCK_KV = 32
    _GROUPED_BLOCK_H = 16

    att_out = torch.empty(
        (q.shape[0], q.shape[1], max_kv_splits, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    att_lse = torch.empty(
        (q.shape[0], q.shape[1], max_kv_splits), dtype=torch.float32, device=q.device
    )

    # decode_attention_fwd, lines 1220-1222 and 1269-1293
    assert max_kv_splits == att_out.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= att_out.shape[0]
    assert q.shape[1] // v_buffer.shape[-2] != 1, "kv_group_num == 1 runs _fwd_kernel_stage1"
    sm_scale_withk = sm_scale * k_scale

    # _decode_grouped_att_m_fwd, with has_mla=False, use_pdl=False, score_mod=None,
    # aux_tensors=None, tune_mla=False, forced_kv_splits=0, uneven_batch=None
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    # 4-D view exposes head_num at dim 2; legacy 3-D exposes
    # it at dim 1.
    kv_head_num = k_buffer.shape[-2]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // kv_head_num

    BLOCK_H = _GROUPED_BLOCK_H
    MAX_KV_SPLITS = max_kv_splits
    head_tiles = triton.cdiv(head_num, min(_GROUPED_BLOCK_H, kv_group_num))

    num_stages = 2
    num_warps = 4

    # Blocks at or above the split count return immediately, so the grid shrinks too.
    forced_kv_splits = 0
    grid = (batch, head_tiles, forced_kv_splits or MAX_KV_SPLITS)

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = None, 0, 0, 0  # score_mod is None

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        HAS_MLA=False,
        USE_PDL=False,
        IS_GFX1250=_is_gfx1250,
        PAGE_SIZE=page_size,
        SCORE_MOD=None,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        forced_kv_splits=forced_kv_splits,
        USE_FORCED=forced_kv_splits > 0,
        uneven_batch=None,
        ROTATE_BATCH=False,
    )
    return att_out, att_lse


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, max_kv_splits, sm_scale):
    """Each non-empty split's output and lse in float64; ``None`` for an empty split."""
    num_heads, num_kv_heads = q.shape[1], k_buffer.shape[-2]
    group = num_heads // num_kv_heads
    indptr = kv_indptr.tolist()
    out = {}
    for b in range(q.shape[0]):
        n = indptr[b + 1] - indptr[b]
        per = triton.cdiv(triton.cdiv(n, int(num_kv_splits[b])), 32) * 32
        slots = kv_indices[indptr[b]:indptr[b + 1]].long()
        k = k_buffer[slots].double().repeat_interleave(group, dim=1)  # [n, H, Lk]
        v = v_buffer[slots].double().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b].double(), k) * sm_scale
        for s in range(max_kv_splits):
            lo, hi = per * s, min(per * s + per, n)
            if hi <= lo:
                out[b, s] = None
                continue
            sc = scores[:, lo:hi]
            o = torch.einsum("hn,nhd->hd", torch.softmax(sc, dim=-1), v[lo:hi])
            out[b, s] = (o, torch.logsumexp(sc, dim=-1))
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # test/registered/attention/test_triton_attention_kernels.py:855-868 (batch 2, lengths 5,
    # 100, 128 and 500, heads 16/1, 64/1, 128/1, 128/2, head dims 64, 13, 80, 512, 4 splits of
    # max 8), and model shapes: Llama-3-8B (32 / 8, 128), Qwen2.5-7B (28 / 4, 128), Llama-3-70B
    # (64 / 8), mixed lengths, pages of 16, a split count per request.
    # (num_heads, num_kv_heads, Lk, Lv, page_size, seq_lens, num_kv_splits, max_kv_splits)
    for num_heads, num_kv_heads, lk, lv, page_size, seq_lens, splits, max_splits in [
        (16, 1, 64, 64, 1, [5, 100], [4, 4], 8),
        (64, 1, 13, 13, 1, [128, 500], [4, 4], 8),
        (128, 1, 80, 80, 1, [5, 500], [4, 4], 8),
        (128, 2, 512, 512, 1, [100, 128], [4, 4], 8),
        (32, 8, 128, 128, 1, [1, 31, 32, 33, 1000], [1, 8, 8, 2, 8], 8),
        (28, 4, 128, 128, 16, [2048, 3, 700], [8, 1, 5], 8),
        (64, 8, 128, 128, 16, [4000, 64], [8, 8], 8),
        (16, 2, 96, 96, 1, [300], [16], 16),
    ]:
        batch = len(seq_lens)
        num_slots = sum(seq_lens) + 1000
        k_buffer = torch.randn((num_slots, num_kv_heads, lk), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        v_buffer = torch.randn((num_slots, num_kv_heads, lv), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        q = torch.randn((batch, num_heads, lk), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        indptr = [0]
        for n in seq_lens:
            indptr.append(indptr[-1] + n)
        kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=DEVICE)
        # distinct random slots, as SGLang's req_to_token pool hands out (int64)
        kv_indices = torch.randperm(num_slots, generator=gen)[: sum(seq_lens)].to(device=DEVICE, dtype=torch.int64)
        num_kv_splits = torch.tensor(splits, dtype=torch.int32, device=DEVICE)
        sm_scale = 1.0 / (lk**0.5)
        att_out, att_lse = decode_attention_fwd_grouped_stage1(
            q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, max_splits, sm_scale, 1.0,
            0.0, -1, page_size)
        ref = reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, max_splits, sm_scale)
        err = lse_err = 0.0
        for (b, s), r in ref.items():
            if r is None:
                continue
            o, lse = r
            err = max(err, (att_out[b, :, s].double() - o).abs().max().item())
            lse_err = max(lse_err, (att_lse[b, :, s].double() - lse).abs().max().item())
        print(f"heads {num_heads}/{num_kv_heads} Lk {lk} Lv {lv} page {page_size} seq_lens {seq_lens} "
              f"splits {splits} (max {max_splits}): max abs error {err:.2e}, lse {lse_err:.2e}")
        assert err < 1e-2 and lse_err < 1e-3, (err, lse_err)
    print("✅ _fwd_grouped_kernel_stage1 matches the float64 reference on every non-empty split")
