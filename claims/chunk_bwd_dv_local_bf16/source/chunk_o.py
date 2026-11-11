"""
Intra-chunk value gradient of chunked linear attention
======================================================

The backward of the intra-chunk term of chunked (gated) linear attention, as fla's gated DeltaNet,
DeltaNet, Comba and preconditioned gated DeltaNet backward passes call it: for every chunk of
``BT`` tokens of a sequence, every value head ``hv`` and every token ``i`` of the chunk,

    dv[i, hv, :] = sum over the chunk's tokens j >= i of
                   scale * (k[i, h, :] . q[j, h, :]) * 2^(g[j, hv] - g[i, hv]) * do[j, hv, :]

with ``h = hv // (HV // H)`` (grouped value heads) and the gate factor present only when a gate
``g`` is given (``USE_G``; fla passes the chunk-local cumulative log decay in base 2). The kernel
forms the causal-masked ``[BT, BT]`` matrix ``A`` of ``k @ q^T`` (times the gate) in ``float32``,
rounds it to the dtype of ``do`` and multiplies it with ``do``.

Layouts (contiguous):

* ``q``, ``k``: ``[B, T, H, K]``, ``bfloat16``;
* ``do``, ``dv`` (the output, ``empty_like(do)``): ``[B, T, HV, V]``, ``bfloat16``;
* ``g`` (optional): ``[B, T, HV]``, ``float32``;
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int32`` or ``int64``), the
  sequences' boundaries in the packed time axis; ``chunk_indices``: ``[NT, 2]``, of the dtype of
  ``cu_seqlens``, one row ``(sequence, chunk within it)`` per chunk of the batch
  (``prepare_chunk_indices``).

One program per (chunk, batch row and value head): ``grid = (NT, B * HV)``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``chunk_bwd_kernel_dv_local``: ``fla/ops/common/chunk_o.py:450-540``;
* ``exp2``: ``fla/ops/utils/op.py:46-47`` (the default branch, ``FLA_USE_FAST_OPS`` unset);
* ``chunk_bwd_dv_local``: ``fla/ops/common/chunk_o.py:653-706``;
* ``prepare_lens``: ``fla/ops/utils/index.py:40-42``; ``_segmented_arange``:
  ``fla/ops/utils/index.py:113-134``; ``prepare_chunk_indices``: ``fla/ops/utils/index.py:155-164``.

Changes from the source:

* ``fla/ops/common/chunk_o.py:450-464``: the decorators ``@triton.heuristics({'USE_G': ...,
  'USE_G_GAMMA': ..., 'USE_A': ..., 'IS_VARLEN': ...})`` and ``@fla_cache_autotune(configs=[
  triton.Config({}, num_warps=w, num_stages=s) for w in CHUNK_BWD_DV_LOCAL_NUM_WARPS for s in
  [2, 3, 4]], key=[...])`` removed; ``@triton.jit(do_not_specialize=['T'])`` (line 465) kept. The
  kernel's signature and body (lines 466-540) are unchanged.
* ``fla/ops/utils/op.py:46-47``: the helper ``exp2`` copied into this file (the kernel calls it),
  from the module's default branch (``FLA_USE_FAST_OPS`` not ``'1'``), written as a plain ``def``
  at the top level instead of inside the module's ``if``/``else``.
* ``fla/ops/common/chunk_o.py:653-706``: the host ``chunk_bwd_dv_local``: the decorator
  ``@dispatch('common')`` (backend selection, ``fla/backends``; the Triton default on NVIDIA GPUs)
  removed; the device query ``check_shared_mem('hopper', ...)`` replaced by its value on an H200,
  so ``CONST_TILING = 128`` (lines 670-679); the launch passes, as the removed decorators did,
  ``USE_G = g is not None``, ``USE_G_GAMMA = g_gamma is not None``, ``USE_A = A is not None``,
  ``IS_VARLEN = cu_seqlens is not None``, and one configuration of the removed autotune list,
  ``num_warps=4, num_stages=3`` (the fastest of the list on an H200 at gated DeltaNet's default
  head dims, ``K = 256``, ``V = 512``, 64-token chunks, with a gate; at head dims 128 the six
  configurations are within 1% of one another). Otherwise its argument handling, allocation, grid and
  arguments are the source's.
* ``fla/ops/utils/index.py``: ``prepare_lens``, ``_segmented_arange`` and
  ``prepare_chunk_indices`` copied without their ``@tensor_cache`` decorator (a memo of recent
  results by argument identity, ``fla/utils/decorators.py``); ``F.pad`` (``index.py:132``, with
  ``import torch.nn.functional as F``) is written ``torch.nn.functional.pad``.
* The reference (``float64``) and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_A: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_A:
        p_A = A + (bos * HV + i_h) * BT + tl.arange(0, BT)[:, None] + o_t[None, :] * (HV*BT)
        b_A = tl.load(p_A, mask=m_t[None, :], other=0.0)
    else:
        if USE_G:
            g += bos * HV + i_h
            p_g = g + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0)
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
            p_q = q + o_k[:, None] + o_t[None, :] * (H*K)

            b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_q = tl.load(p_q, mask=m_k[:, None] & m_t[None, :], other=0.0)
            b_A += tl.dot(b_k, b_q) * scale
        if USE_G or USE_G_GAMMA:
            b_A *= exp2(b_g[None, :] - b_g[:, None])
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


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


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # the source's device query (check_shared_mem('hopper', ...)) on an H200
    CONST_TILING = 128
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    grid = (NT, B * HV)
    chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        A=A,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,  # the removed @triton.heuristics
        USE_G_GAMMA=g_gamma is not None,  # the removed @triton.heuristics
        USE_A=A is not None,  # the removed @triton.heuristics
        IS_VARLEN=cu_seqlens is not None,  # the removed @triton.heuristics
        num_warps=4,  # one configuration of the removed autotune list
        num_stages=3,
    )
    return dv


def reference(q, k, do, g=None, scale=1.0, cu_seqlens=None, chunk_size=64):
    """dv in float64, chunk by chunk of each sequence (padded rows or cu_seqlens)."""
    B, T, H, K = k.shape
    HV, V = do.shape[2], do.shape[3]
    G = HV // H
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
    out = torch.zeros(do.shape, dtype=torch.float64, device=do.device)
    for b, bos, eos in bounds:
        for c in range(bos, eos, chunk_size):
            e = min(c + chunk_size, eos)
            kk = k[b, c:e].double().repeat_interleave(G, dim=1)  # [L, HV, K]
            qq = q[b, c:e].double().repeat_interleave(G, dim=1)
            dd = do[b, c:e].double()  # [L, HV, V]
            a = torch.einsum('ihk,jhk->hij', kk, qq) * scale  # [HV, L, L]
            if g is not None:
                gg = g[b, c:e].double().transpose(0, 1)  # [HV, L]
                a = a * torch.exp2(gg[:, None, :] - gg[:, :, None])
            a = torch.triu(a)
            out[b, c:e] = torch.einsum('hij,jhv->ihv', a, dd)
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    RCP_LN2 = 1.4426950216  # fla/ops/utils/constant.py:10
    gen = torch.Generator().manual_seed(0)
    cases = [
        # (B, T, H, HV, K, V, chunk_size, cu_seqlens): shapes of fla's tests/ops/test_gdn.py
        # (test_chunk, test_chunk_with_chunk_size, test_chunk_varlen), gated DeltaNet's default heads,
        # Qwen3-Next's grouped value heads, and short and chunk-sized sequences
        (1, 63, 1, 1, 64, 64, 64, None),
        (2, 500, 4, 4, 60, 60, 64, None),
        (2, 1000, 2, 8, 128, 128, 64, None),
        (3, 1024, 4, 4, 100, 100, 64, None),
        (1, 64, 2, 4, 32, 32, 16, None),
        (1, 64, 2, 4, 32, 32, 32, None),
        (2, 1024, 6, 6, 256, 512, 64, None),
        (1, 2048, 16, 32, 128, 128, 64, None),
        (1, 15, 4, 4, 60, 60, 64, [0, 15]),
        (1, 1000, 4, 4, 64, 64, 64, [0, 256, 500, 1000]),
        (1, 2000, 4, 4, 100, 100, 64, [0, 15, 100, 300, 1200, 2000]),
        (1, 1000, 2, 8, 64, 64, 64, [0, 256, 500, 1000]),
        (1, 4096, 4, 8, 64, 64, 64, [0, 1, 63, 64, 65, 1000, 4096]),
    ]
    for B, T, H, HV, K, V, BT, cu in cases:
        q = torch.randn((B, T, H, K), generator=gen).to(DEVICE, torch.bfloat16)
        k = torch.randn((B, T, H, K), generator=gen).to(DEVICE, torch.bfloat16)
        do = torch.randn((B, T, HV, V), generator=gen).to(DEVICE, torch.bfloat16)
        g0 = torch.nn.functional.logsigmoid(torch.randn((B, T, HV), generator=gen)).to(DEVICE)
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.long, device=DEVICE)
        scale = K ** -0.5
        worst = 0.0
        for gate in (False, True):
            if gate:  # the chunk-local cumulative log2 decay, as gated DeltaNet's forward builds it
                g = torch.zeros_like(g0)
                for b, bos, eos in ([(b, 0, T) for b in range(B)] if cu is None
                                    else [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]):
                    for c in range(bos, eos, BT):
                        e = min(c + BT, eos)
                        g[b, c:e] = g0[b, c:e].cumsum(0) * RCP_LN2
            else:
                g = None
            dv = chunk_bwd_dv_local(q, k, do, g=g, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT)
            ref = reference(q, k, do, g=g, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT)
            assert dv.dtype == torch.bfloat16
            err = ((dv.double() - ref).abs().max() / ref.abs().max()).item()
            assert err < 2 ** -6, err
            worst = max(worst, err)
        print(f"B {B} T {T} H {H} HV {HV} K {K} V {V} chunk {BT} cu_seqlens {cu}: without and with a gate: "
              f"max error {worst:.2e} (relative to max |ref|)")
    print("chunk_bwd_dv_local matches the float64 reference")
