"""
Chunk-scaled K·Kᵀ, forward
==========================

Within each chunk of ``BT`` tokens, the strictly lower triangular matrix of DeltaNet's WY
representation: for rows ``i`` and columns ``j`` of one chunk (positions ``t_i``, ``t_j`` of a
sequence), and value head ``h`` reading key head ``h // (HV // H)``,

    A[t_i, h, j] = beta[t_i, h] * (k[t_i] · k[t_j]) * exp2(g[t_i, h] - g[t_j, h])   if j < i and t_j < T,
    A[t_i, h, j] = 0                                                                 otherwise,

with the factor ``exp2(...)`` only when a gate ``g`` is given. ``g`` is the within-chunk
cumulative log2 decay (``chunk_local_cumsum`` of the log decay with ``scale = 1/ln 2``), as gated
DeltaNet computes it. The dot products are ``tl.dot`` of ``bfloat16`` blocks into ``float32``,
over blocks of ``BK`` key channels.

Layouts (contiguous):

* ``k``: ``[B, T, H, K]``, ``bfloat16``;
* ``g``: ``[B, T, HV]``, ``float32``; ``beta``: ``[B, T, HV]``, ``bfloat16``;
* ``A``: ``[B, T, HV, BT]``, ``float32`` (``output_dtype``): row ``t`` holds the row of ``t``'s chunk;
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int32`` or ``int64``), the
  sequences' boundaries; ``chunk_indices``: ``[NT, 2]``, of the dtype of ``cu_seqlens``, one row
  ``(sequence, chunk within it)`` per chunk (``prepare_chunk_indices``).

One program per (chunk, batch row and value head): ``grid = (NT, B * HV)``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``chunk_scaled_dot_kkt_fwd_kernel``: ``fla/ops/common/chunk_scaled_dot_kkt.py:32-80``;
* ``exp2``: ``fla/ops/utils/op.py:46-47`` (the default branch of ``op.py:32-53``);
* ``chunk_scaled_dot_kkt_fwd``: ``fla/ops/common/chunk_scaled_dot_kkt.py:83-135``;
* ``prepare_lens``: ``fla/ops/utils/index.py:40-42``; ``_segmented_arange``:
  ``fla/ops/utils/index.py:113-134``; ``prepare_chunk_indices``: ``fla/ops/utils/index.py:155-164``.

Changes from the source:

* ``fla/ops/common/chunk_scaled_dot_kkt.py:18-31``: the decorators ``@triton.heuristics({'USE_G':
  ..., 'IS_VARLEN': ...})`` and ``@triton.autotune(configs=[triton.Config({'BK': BK},
  num_warps=w, num_stages=s) for BK in [32, 64, 128] for w in [2, 4, 8] for s in [2, 3, 4]],
  key=['H', 'HV', 'K', 'BT', 'IS_VARLEN'])`` removed; ``@triton.jit(do_not_specialize=['T'])``
  kept. The kernel's signature and body are unchanged.
* ``fla/ops/utils/op.py:46-47``: the helper ``exp2`` copied verbatim. ``op.py`` defines it by the
  environment variable ``FLA_USE_FAST_OPS``: unset (the default, so on an H200 too), it is
  ``tl.math.exp2`` of the argument in ``float32``; with ``FLA_USE_FAST_OPS=1`` it would be
  libdevice's ``exp2`` (``op.py:36``), which this file does not take.
* ``fla/ops/common/chunk_scaled_dot_kkt.py:83-135``: the host ``chunk_scaled_dot_kkt_fwd`` passes,
  as the removed heuristics did, ``USE_G = g is not None`` and ``IS_VARLEN = cu_seqlens is not
  None``, and one configuration of the removed autotune list, ``BK = 64``, ``num_warps=2``,
  ``num_stages=3`` (its eleventh entry, the autotuner's choice on an H200 for 16 key heads, 32
  value heads, ``K = 128`` and chunks of 32 and 64 tokens), in place of the autotuner's choice. Its decorator ``@dispatch('common')``
  (backend selection, ``fla/backends.py``; the Triton default on NVIDIA GPUs) is removed. Otherwise
  its argument handling, allocation, grid and arguments are the source's.
* ``fla/ops/utils/index.py``: ``prepare_lens``, ``_segmented_arange`` and
  ``prepare_chunk_indices`` copied without their ``@tensor_cache`` decorator (a memo of recent
  results by argument identity, ``fla/utils/decorators.py:33-80``); ``F.pad`` (``index.py:132``,
  with ``import torch.nn.functional as F``) is written ``torch.nn.functional.pad``.
* The reference (``float64``) and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = beta + bos*HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0.0)
        b_A = tl.dot(b_k, tl.trans(b_k), b_A)

    if USE_G:
        p_g = g + bos*HV + i_h + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp2(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = A + (bos*HV + i_h) * BT + o_t[:, None] * (BT*HV) + tl.arange(0, BT)[None, :]
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None])


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


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]` where `H` is the number of query/key heads.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, HV]` where `HV` is the number of value/output heads.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`
        chunk_indices (torch.LongTensor):
            The chunk indices of the input tensor. Default: None.
    Returns:
        beta * K * K^T of shape `[B, T, HV, BT]` where `BT` is the chunk size.
        For GVA, H < HV and HV % H == 0. For standard attention, H == HV.
    """
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * HV)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BK=64,  # one configuration of the removed autotune list
        IS_VARLEN=cu_seqlens is not None,  # the removed @triton.heuristics
        USE_G=g is not None,  # the removed @triton.heuristics
        num_warps=2,
        num_stages=3,
    )
    return A


def reference(k, g, beta, chunk_size, cu_seqlens=None):
    """beta * K * K^T within each chunk, gated by exp2(g_i - g_j), strictly lower triangular, in float64."""
    B, T, H, K = k.shape
    HV = beta.shape[2]
    G = HV // H
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
    out = torch.zeros((B, T, HV, chunk_size), dtype=torch.float64, device=k.device)
    for b, bos, eos in bounds:
        for c in range(bos, eos, chunk_size):
            L = min(c + chunk_size, eos) - c
            kc = k[b, c:c + L].double().repeat_interleave(G, dim=1)  # [L, HV, K]
            a = torch.einsum("ihd,jhd->hij", kc, kc)  # [HV, L, L]
            if g is not None:
                gc = g[b, c:c + L].double().T  # [HV, L]
                a = a * torch.exp2(gc[:, :, None] - gc[:, None, :])
            a = a * beta[b, c:c + L].double().T[:, :, None]
            a = torch.tril(a, diagonal=-1)
            out[b, c:c + L, :, :L] = a.permute(1, 0, 2)
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    RCP_LN2 = 1.4426950216  # fla/ops/utils/constant.py:10
    gen = torch.Generator().manual_seed(0)

    def gate_cumsum(B, T, HV, chunk_size, cu):
        """The log2 decay's within-chunk cumulative sum, as gated DeltaNet passes g (float32)."""
        g = torch.nn.functional.logsigmoid(torch.randn((B, T, HV), generator=gen)) * RCP_LN2
        bounds = [(b, 0, T) for b in range(B)] if cu is None else [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        for b, bos, eos in bounds:
            for c in range(bos, eos, chunk_size):
                g[b, c:min(c + chunk_size, eos)] = g[b, c:min(c + chunk_size, eos)].cumsum(0)
        return g.to(DEVICE)

    cases = [
        # (B, T, H, HV, K, chunk_size, cu_seqlens): gated DeltaNet's unfused path (chunk 16, 32),
        # gated DeltaProduct (chunk 64); fla's tests/ops/test_solve_tril.py and test_gdn.py shapes
        (1, 63, 1, 1, 64, 16, None),
        (2, 500, 4, 4, 64, 32, None),
        (2, 1000, 2, 8, 128, 64, None),
        (3, 1024, 2, 2, 128, 16, None),
        (2, 1024, 4, 8, 256, 32, None),
        (1, 1, 2, 4, 128, 64, None),
        (4, 2048, 16, 32, 128, 64, None),
        (1, 15, 4, 4, 64, 16, [0, 15]),
        (1, 1000, 4, 4, 64, 32, [0, 256, 500, 1000]),
        (1, 2000, 4, 8, 100, 64, [0, 15, 100, 300, 1200, 2000]),
        (1, 2048, 4, 4, 64, 16, [0, 1, 100, 300, 1200, 2048]),
    ]
    for B, T, H, HV, K, chunk_size, cu in cases:
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEVICE)
        k = torch.nn.functional.normalize(torch.randn((B, T, H, K), generator=gen), dim=-1).to(DEVICE, torch.bfloat16)
        beta = torch.rand((B, T, HV), generator=gen).sigmoid().to(DEVICE, torch.bfloat16)
        g = gate_cumsum(B, T, HV, chunk_size, cu)
        worst = 0.0
        for gg in (g, None):
            A = chunk_scaled_dot_kkt_fwd(k, g=gg, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
            ref = reference(k, gg, beta, chunk_size, cu_seqlens=cu_seqlens)
            assert A.dtype == torch.float32 and A.shape == (B, T, HV, chunk_size)
            err = (A.double() - ref).abs().max().item()
            assert err < 1e-5, err
            worst = max(worst, err)
        print(f"B {B} T {T} H {H} HV {HV} K {K} chunk {chunk_size} cu_seqlens {cu}: with g and without: "
              f"max abs error {worst:.2e}")
    print("✅ chunk-scaled K·Kᵀ matches the float64 reference")
