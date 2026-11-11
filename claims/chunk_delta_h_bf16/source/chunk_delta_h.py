"""
Gated delta rule: the state recurrence across chunks (forward)
==============================================================

For each sequence and value head, the state ``h`` (``[K, V]``, ``float32``) is carried across the
chunks of ``BT`` tokens. In chunk ``t`` the kernel stores the state entering it into ``h[t]``,
computes ``v_new = u - w @ h`` (stored when ``SAVE_NEW_VALUE``), and updates
``h <- h * exp2(g_last) + k^T @ (v_new * exp2(g_last - g))`` (``USE_G``; ``g`` is the gate's
chunk-local cumulative sum in log2 units, ``g_last`` its value at the chunk's last token). The
state starts from ``h0`` (``USE_INITIAL_STATE``) or zero and the last one goes to ``ht``
(``STORE_FINAL_STATE``). One program per (block of ``BV`` value columns, sequence, value head).

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:
``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``: ``fla/ops/common/chunk_delta_h.py:40-349``;
``exp2``: ``fla/ops/utils/op.py:46-47``; ``chunk_gated_delta_rule_fwd_h``:
``fla/ops/common/chunk_delta_h.py:687-748``; ``prepare_lens``, ``_segmented_arange``,
``prepare_chunk_indices``, ``prepare_chunk_offsets``: ``fla/ops/utils/index.py:40-42, 113-134,
155-172``.

Changes from the source (every other line of the kernel is unchanged, at its upstream line number):

* ``chunk_delta_h.py:1-37``: the license header, the imports of fla's modules and the module-level
  values ``NUM_WARPS`` and ``GATED_DELTA_RULE_FWD_H_NUM_WARPS`` replaced by this docstring, the
  imports of ``torch``, ``triton`` and ``triton.language as tl``, and the helper ``exp2``.
* ``op.py:46-47``: the helper ``exp2`` copied verbatim (fla's default branch, taken when
  ``FLA_USE_FAST_OPS`` is not ``'1'``); the kernel calls it.
* ``chunk_delta_h.py:40-57``: the decorators ``@triton.heuristics({...})`` (``USE_G``, ``USE_GK``,
  ``USE_INITIAL_STATE``, ``STORE_FINAL_STATE``, ``SAVE_NEW_VALUE``, ``IS_VARLEN``) and
  ``@fla_cache_autotune(configs=[triton.Config({'BV': BV}, num_warps=w, num_stages=s) ...],
  key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'])`` removed (blank lines here);
  ``@triton.jit(do_not_specialize=['T'])`` kept (line 58).
* ``chunk_delta_h.py:687-748``: the host ``chunk_gated_delta_rule_fwd_h`` without its
  ``@dispatch('common')`` decorator (backend selection; Triton on NVIDIA GPUs); it passes the flags
  the removed heuristics computed and one configuration of the removed list (``BV=32``,
  ``num_warps=4``, ``num_stages=3``) as launch arguments, and the grid as a tuple. Its checks,
  allocations and arguments are the source's.
* ``index.py``: the torch helpers copied without their ``@tensor_cache`` decorator (a memo by
  argument identity); ``F.pad`` written ``torch.nn.functional.pad``.
* The backward kernel and host (``chunk_delta_h.py:352-822``) are not copied; ``make_inputs``,
  ``reference`` and the test under ``__main__`` are new.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))









@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
            m_h0_1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
            m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                m_h0_2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                m_h0_3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                m_h0_3 = m_k3[:, None] & m_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                m_h0_4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                m_h0_4 = m_k4[:, None] & m_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV*K*V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV*K*V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV*K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v = tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype), b_v)
            else:
                b_v = tl.dot(b_w, b_h2.to(b_w.dtype), b_v)
        if K > 128:
            p_w = w + o_t[:, None] * (HV*K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v = tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype), b_v)
            else:
                b_v = tl.dot(b_w, b_h3.to(b_w.dtype), b_v)
        if K > 192:
            p_w = w + o_t[:, None] * (HV*K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v = tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype), b_v)
            else:
                b_v = tl.dot(b_w, b_h4.to(b_w.dtype), b_v)
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 = tl.dot(b_k, b_v, b_h1)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 = tl.dot(b_k, b_v, b_h2)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 = tl.dot(b_k, b_v, b_h3)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 = tl.dot(b_k, b_v, b_h4)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + o_v[:, None] * K + o_k1[None, :]
            m_ht = m_v[:, None] & m_k1[None, :]
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
            m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                m_ht = m_v[:, None] & m_k2[None, :]
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                m_ht = m_k2[:, None] & m_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                m_ht = m_v[:, None] & m_k3[None, :]
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                m_ht = m_k3[:, None] & m_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                m_ht = m_v[:, None] & m_k4[None, :]
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                m_ht = m_k4[:, None] & m_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


def _segmented_arange(counts: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Expand per-segment counts into flat per-slot index tensors.

    Given segment sizes ``counts = [c0, c1, ...]``, return two 1-D tensors of
    length ``counts.sum()`` that together label every slot with its segment and
    its position within that segment.

    Example -- ``counts = [2, 3]`` (segment 0 spans 2 slots, segment 1 spans 3)::

        seg_id    = [0, 0, 1, 1, 1]   # which segment each slot belongs to
        intra_idx = [0, 1, 0, 1, 2]   # running index within that segment

    With CUDA ``counts``, ``repeat_interleave`` reads ``counts.sum()`` on the
    host (one device sync). Pass host-side counts to avoid it.
    """
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype),
        counts,
    )
    seg_start = torch.nn.functional.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra_idx


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode='floor')
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return torch.nn.functional.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    BV = 32  # one configuration of the removed autotune list (with num_warps, num_stages below)
    grid = (triton.cdiv(V, BV) * N * HV,)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=g is not None,  # the removed @triton.heuristics
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        SAVE_NEW_VALUE=v_new is not None,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return h, v_new, final_state


RCP_LN2 = 1.4426950216  # fla/ops/utils/constant.py:10


def make_inputs(lens, H, HV, K, V, varlen=True, gate_scale=1.0, beta_scale=1.0, h0_scale=1.0,
                seed=0, device="cuda"):
    """Inputs as the chunked gated delta rule's forward builds them (chunk size 64): ``k``
    L2-normalized per token (``bfloat16``); ``g`` the chunk-local cumulative sum of a per-token
    ``logsigmoid`` gate, times ``1/ln 2`` (``float32``); ``w``, ``u`` the WY representation of each
    chunk (``w = A·(beta·k·2^g)``, ``u = A·(beta·v)``, ``A = (I + tril(beta·k·k^T·2^(g_i-g_j), -1))^-1``,
    computed in ``float64``, then ``bfloat16``); ``h0`` ``float32``. ``lens`` are the sequences'
    lengths: packed when ``varlen`` (``B = 1``, ``cu_seqlens`` ``int64``), else ``B = len(lens)``
    rows of one common length."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    BT = 64
    if varlen:
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        B, T = 1, cu[-1]
        bounds = [(0, cu[i], cu[i + 1]) for i in range(len(lens))]
    else:
        assert len(set(lens)) == 1
        B, T = len(lens), lens[0]
        bounds = [(b, 0, T) for b in range(B)]
    k = torch.randn(B, T, H, K, generator=gen, dtype=torch.float64)
    k = torch.nn.functional.normalize(k, p=2, dim=-1).to(torch.bfloat16).double()
    v = torch.randn(B, T, HV, V, generator=gen, dtype=torch.float64).to(torch.bfloat16).double()
    beta = (torch.rand(B, T, HV, generator=gen, dtype=torch.float64).sigmoid() * beta_scale)
    gt = torch.nn.functional.logsigmoid(torch.rand(B, T, HV, generator=gen, dtype=torch.float64)) * gate_scale
    g = torch.zeros(B, T, HV, dtype=torch.float32)
    w = torch.zeros(B, T, HV, K, dtype=torch.float64)
    u = torch.zeros(B, T, HV, V, dtype=torch.float64)
    G = HV // H
    k, v, beta, gt, w, u = (t.to(device) for t in (k, v, beta, gt, w, u))
    g = g.to(device)
    hk = torch.arange(HV, device=device) // G
    for b, bos, eos in bounds:
        for s0 in range(bos, eos, BT):
            s1 = min(s0 + BT, eos)
            L = s1 - s0
            gc = torch.cumsum(gt[b, s0:s1].float(), 0) * RCP_LN2  # [L, HV], float32 like fla
            g[b, s0:s1] = gc
            kc = k[b, s0:s1][:, hk].permute(1, 0, 2)               # [HV, L, K]
            bc = beta[b, s0:s1].T                                   # [HV, L]
            gg = gc.T.double()                                      # [HV, L]
            M = (bc[:, :, None] * (kc @ kc.transpose(1, 2))) * torch.exp2(gg[:, :, None] - gg[:, None, :])
            A = torch.linalg.inv(torch.eye(L, dtype=torch.float64, device=device) + torch.tril(M, -1))
            w[b, s0:s1] = (A @ (kc * (bc * torch.exp2(gg))[:, :, None])).permute(1, 0, 2)
            u[b, s0:s1] = (A @ (v[b, s0:s1].permute(1, 0, 2) * bc[:, :, None])).permute(1, 0, 2)
    N = len(lens) if varlen else B
    h0 = torch.randn(N, HV, K, V, generator=gen, dtype=torch.float32) * h0_scale
    out = dict(k=k.to(torch.bfloat16).to(device), w=w.to(torch.bfloat16).to(device),
               u=u.to(torch.bfloat16).to(device), g=g.to(device), h0=h0.to(device))
    out["cu_seqlens"] = torch.tensor(cu, dtype=torch.int64, device=device) if varlen else None
    return out


def reference(k, w, u, g, h0=None, cu_seqlens=None, BT=64):
    """The recurrence in float64 from the inputs: per sequence, value head and chunk, ``h[t] = S``,
    ``v_new = u - w @ S``, ``S <- 2^g_last · S + k^T @ (v_new · 2^(g_last - g))``; returns ``h``
    (``[B, NT, HV, K, V]``), ``v_new`` and the final states (``[N, HV, K, V]``)."""
    B, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[3]
    G = HV // H
    k, w, u, g = k.double().cpu(), w.double().cpu(), u.double().cpu(), g.double().cpu()
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    nts = [(e - s + BT - 1) // BT for _, s, e in bounds]
    NT = nts[0] if cu_seqlens is None else sum(nts)
    h = torch.zeros(B, NT, HV, K, V, dtype=torch.float64)
    vn = torch.zeros(B, T, HV, V, dtype=torch.float64)
    ht = torch.zeros(len(bounds), HV, K, V, dtype=torch.float64)
    c0 = 0
    for n, (b, bos, eos) in enumerate(bounds):
        c = 0 if cu_seqlens is None else c0
        for hv in range(HV):
            S = torch.zeros(K, V, dtype=torch.float64) if h0 is None else h0[n, hv].double().cpu()
            for i, s0 in enumerate(range(bos, eos, BT)):
                s1 = min(s0 + BT, eos)
                h[b, c + i, hv] = S
                vv = u[b, s0:s1, hv] - w[b, s0:s1, hv] @ S
                vn[b, s0:s1, hv] = vv
                gl = g[b, s1 - 1, hv]
                S = torch.exp2(gl) * S + k[b, s0:s1, hv // G].T @ (vv * torch.exp2(gl - g[b, s0:s1, hv])[:, None])
            ht[n, hv] = S
        c0 += nts[n]
    return h, vn, ht


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = [
        # (lens, H, HV, K, V, varlen)
        ([1024] * 2, 4, 4, 128, 128, False),
        ([75] * 2, 4, 4, 64, 64, False),
        ([500] * 2, 3, 3, 60, 60, False),
        ([256] * 2, 2, 4, 64, 64, False),
        ([1024] * 2, 4, 8, 128, 128, False),
        ([15], 4, 4, 60, 60, True),
        ([256, 244, 500], 4, 4, 64, 64, True),
        ([15, 85, 200, 900, 800], 4, 4, 100, 100, True),
        ([256, 244, 500], 2, 8, 64, 64, True),
        ([1000, 3000], 16, 32, 128, 128, True),
    ]
    for lens, H, HV, K, V, varlen in cases:
        x = make_inputs(lens, H, HV, K, V, varlen=varlen)
        h, vn, ht = chunk_gated_delta_rule_fwd_h(
            x["k"], x["w"], x["u"], g=x["g"], initial_state=x["h0"], output_final_state=True,
            cu_seqlens=x["cu_seqlens"])
        rh, rv, rt = reference(x["k"], x["w"], x["u"], x["g"], x["h0"], x["cu_seqlens"])
        eh = ((h.double().cpu() - rh).abs().max() / rh.abs().max()).item()
        ev = ((vn.double().cpu() - rv).abs().max() / rv.abs().max()).item()
        et = ((ht.double().cpu() - rt).abs().max() / rt.abs().max()).item()
        print(f"lens={lens} H={H} HV={HV} K={K} V={V} varlen={varlen}: "
              f"rel. max error h {eh:.2e}, v_new {ev:.2e}, ht {et:.2e}")
        assert eh < 2e-2 and ev < 2e-2 and et < 2e-2
    print("ok")
