"""
Chunked linear-attention output
===============================

The output of chunked linear attention, chunk by chunk of ``BT = 64`` tokens: for a token ``t`` of
a chunk whose state (the recurrent memory at the chunk's start) is ``S``, value head ``hv`` and
query/key head ``hq = hv // (HV // H)``,

    o[t] = scale * ( exp2(g[t]) * (q[t] @ S)
                     + sum over s in the chunk, s <= t, of (q[t] . k[s]) * exp2(g[t] - g[s]) * v[s] ),

with the gates ``g`` (``float32`` base-2 log decays, cumulative in the chunk) when ``USE_G``, with
``g[t] = g_gamma[hv] * (i + 1)`` (``i``: ``t``'s position in its chunk) when ``USE_G_GAMMA``, else
none. ``S`` is ``h[chunk, hv]``, ``[K, V]``, or ``[V, K]`` when ``STATE_V_FIRST``. Gated DeltaNet
(``fla/ops/gated_delta_rule/chunk.py:111``), the delta rule, comba, preconditioned GDN and simple
GLA (``fla/ops/simple_gla/chunk.py:45``; RetNet, lightning attention) call it on the states.

Layouts (contiguous): ``q``, ``k``: ``[B, T, H, K]``; ``v``, ``o``: ``[B, T, HV, V]``; ``g``:
``[B, T, HV]`` (``float32``); ``h``: ``[B, NT, HV, K, V]`` (``[B, NT, HV, V, K]`` when
``STATE_V_FIRST``); ``q``, ``k``, ``v``, ``h``, ``o`` ``bfloat16``. A packed variable-length
batch has ``B = 1``, ``cu_seqlens`` (``[N + 1]``, ``int32`` or ``int64``) and ``chunk_indices``
(``[NT, 2]``, the dtype of ``cu_seqlens``: one row ``(sequence, chunk within it)`` per chunk,
``prepare_chunk_indices``), and ``h`` is ``[1, NT, HV, ...]`` with chunk ``r`` the ``r``-th row's.

One program per (block of ``BV`` value columns, chunk, batch row and value head):
``grid = (cdiv(V, BV) * NT, B * HV)``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``chunk_fwd_kernel_o``: ``fla/ops/common/chunk_o.py:51-155``;
* ``unflatten_program_id``: ``fla/ops/utils/op.py:22-29``;
* ``exp2``: ``fla/ops/utils/op.py:46-47``;
* ``chunk_fwd_o``: ``fla/ops/common/chunk_o.py:543-586``;
* ``prepare_lens``: ``fla/ops/utils/index.py:40-42``; ``_segmented_arange``:
  ``fla/ops/utils/index.py:113-134``; ``prepare_chunk_indices``: ``fla/ops/utils/index.py:155-164``.

Changes from the source:

* ``fla/ops/common/chunk_o.py:51-56``: the decorators ``@triton.heuristics({'USE_G': ...,
  'USE_G_GAMMA': ..., 'IS_VARLEN': ...})`` and ``@fla_cache_autotune(configs=_O_CONFIGS,
  key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'], ...)`` removed;
  ``@triton.jit(do_not_specialize=['T'])`` kept. The kernel's signature and body are unchanged.
* ``fla/ops/common/chunk_o.py:27-48``: the module-level config lists (``_O_CONFIGS`` and the values
  it reads) removed; the host passes one of its configurations (below).
* ``fla/ops/utils/op.py:22-29``: the helper ``unflatten_program_id`` copied verbatim (the kernel
  calls it).
* ``fla/ops/utils/op.py:46-47``: the helper ``exp2`` copied as fla defines it when
  ``FLA_USE_FAST_OPS`` is unset (its default, ``op.py:32,43``): ``tl.math.exp2(x.to(tl.float32))``.
* ``fla/ops/common/chunk_o.py:543-586``: the host ``chunk_fwd_o`` passes, as the removed decorators
  did, ``USE_G = g is not None``, ``USE_G_GAMMA = g_gamma is not None`` and ``IS_VARLEN =
  cu_seqlens is not None``, and, for the autotuner's choice, the entry of the removed list on Hopper
  (``chunk_o.py:34-42``) its new argument ``BK`` names (default 64): ``BK = BV = 128``, 8 warps;
  ``BK = BV = 64``, 4 warps; ``BK = BV = 32``, 2 warps; all with ``num_stages=3``; its decorator
  ``@dispatch('common')`` (backend selection, ``fla/backends``; the Triton default on NVIDIA GPUs)
  removed. Otherwise its argument handling, allocation, grid and arguments are the source's.
* ``fla/ops/utils/index.py``: ``prepare_lens``, ``_segmented_arange`` and
  ``prepare_chunk_indices`` copied without their ``@tensor_cache`` decorator (a memo of recent
  results by argument identity, ``fla/utils/decorators.py:33-80``); ``F.pad`` (``index.py:132``,
  with ``import torch.nn.functional as F``) is written ``torch.nn.functional.pad``.
* The reference (``float64``) and the test under ``__main__`` are new; they cover ``g`` only, not ``g_gamma``.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def unflatten_program_id(X: tl.constexpr, Y: tl.constexpr = None):
    """Recover int64 x-fastest 2D/3D indices without using CUDA's limited y/z grid dimensions."""
    pid = tl.program_id(0).to(tl.int64)
    if Y is None:
        return pid % X, pid // X
    else:
        return pid % X, (pid // X) % Y, pid // (X * Y)


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
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
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H*K)
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if STATE_V_FIRST:
            b_o = tl.dot(b_q, tl.trans(b_h), b_o)
        else:
            b_o = tl.dot(b_q, b_h, b_o)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A = tl.dot(b_q, b_k, b_A)

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV*V) + o_v[None, :]

    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


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


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    BK: int = 64,
) -> torch.Tensor:
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT, B * HV)
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,  # one configuration of the removed autotune list: BK = BV, num_warps by BK
        BV=BK,
        USE_G=g is not None,  # the removed @triton.heuristics
        USE_G_GAMMA=g_gamma is not None,
        IS_VARLEN=cu_seqlens is not None,
        STATE_V_FIRST=state_v_first,
        num_warps={32: 2, 64: 4, 128: 8}[BK],
        num_stages=3,
    )
    return o


def reference(q, k, v, h, g=None, scale=None, state_v_first=False, cu_seqlens=None, chunk_size=64,
              chunk_indices=None):
    """The output in float64, chunk by chunk: the inter-chunk term ``exp2(g[t]) * q[t] @ S`` and the
    causal intra-chunk term, with ``S`` the chunk's state in ``h`` (the ``r``-th chunk of the batch;
    for a packed batch, the ``r``-th row of ``chunk_indices``)."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    BT = chunk_size
    if scale is None:
        scale = K ** -0.5
    out = torch.zeros(v.shape, dtype=torch.float64, device=v.device)
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT)
        chunks = [(b, b * NT + c, c * BT, min(c * BT + BT, T)) for b in range(B) for c in range(NT)]
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        cu = cu_seqlens.tolist()
        chunks = [(0, r, cu[n] + c * BT, min(cu[n] + c * BT + BT, cu[n + 1]))
                  for r, (n, c) in enumerate(chunk_indices.tolist())]
    qd = q.double().repeat_interleave(G, dim=2)
    kd = k.double().repeat_interleave(G, dim=2)
    vd, hd = v.double(), h.double()
    if state_v_first:
        hd = hd.transpose(-1, -2)
    gd = g.double() if g is not None else None
    for b, r, lo, hi in chunks:
        n = hi - lo
        qc, kc, vc = qd[b, lo:hi].transpose(0, 1), kd[b, lo:hi].transpose(0, 1), vd[b, lo:hi].transpose(0, 1)
        S = hd.reshape(-1, HV, K, V)[r]                       # [HV, K, V]
        inter = qc @ S                                        # [HV, n, V]
        A = qc @ kc.transpose(-1, -2)                         # [HV, n, n]
        if gd is not None:
            gc = gd[b, lo:hi].transpose(0, 1)                 # [HV, n]
            inter = inter * torch.exp2(gc)[..., None]
            A = A * torch.exp2(gc[:, :, None] - gc[:, None, :])
        A = A * torch.tril(torch.ones(n, n, dtype=torch.bool, device=A.device))
        out[b, lo:hi] = (scale * (inter + A @ vc)).transpose(0, 1)
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    cases = [
        # (B, T, H, HV, K, V, use_g, state_v_first, cu_seqlens): fla's tests/ops/test_gdn_kernels.py
        # (test_chunk_fwd_o, bfloat16), then gated DeltaNet's uses
        (2, 128, 2, 2, 64, 64, True, False, None),
        (2, 128, 2, 4, 64, 64, True, False, None),
        (2, 128, 2, 2, 64, 64, False, False, None),
        (1, 8192, 8, 8, 128, 128, True, False, [0, 5000, 8192]),
        (1, 3000, 16, 32, 128, 128, True, True, [0, 1, 64, 65, 1000, 3000]),
        (1, 1000, 6, 6, 256, 512, True, True, [0, 100, 1000]),
        (2, 500, 4, 4, 60, 100, True, False, None),
    ]
    for B, T, H, HV, K, V, use_g, svf, cu in cases:
        q = torch.randn((B, T, H, K), generator=gen).to(DEVICE, torch.bfloat16)
        k = torch.randn((B, T, H, K), generator=gen).to(DEVICE, torch.bfloat16)
        v = torch.randn((B, T, HV, V), generator=gen).to(DEVICE, torch.bfloat16)
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEVICE)
        NT = triton.cdiv(T, 64) if cu is None else len(prepare_chunk_indices(cu_seqlens, 64))
        shape = (V, K) if svf else (K, V)
        h = (0.1 * torch.randn((B, NT, HV, *shape), generator=gen)).to(DEVICE, torch.bfloat16)
        g = None
        if use_g:
            g = -torch.rand((B, T, HV), generator=gen).to(DEVICE)
            # cumulative within each 64-token chunk of each sequence, as the gate cumsum makes it
            bounds = [(b, 0, T) for b in range(B)] if cu is None else [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
            for b, lo, hi in bounds:
                for c in range(lo, hi, 64):
                    g[b, c:min(c + 64, hi)] = g[b, c:min(c + 64, hi)].cumsum(0)
        o = chunk_fwd_o(q, k, v, h, g=g, state_v_first=svf, cu_seqlens=cu_seqlens)
        ref = reference(q, k, v, h, g=g, state_v_first=svf, cu_seqlens=cu_seqlens)
        assert o.dtype == torch.bfloat16
        err = ((o.double() - ref).abs() / (ref.abs().max() + 1e-30)).max().item()
        assert err < 1e-2, err
        print(f"B {B} T {T} H {H} HV {HV} K {K} V {V} use_g {use_g} state_v_first {svf} cu_seqlens {cu}: "
              f"max error {err:.2e} (relative to max |ref|)")
    print("✅ chunk_fwd_o matches the float64 reference")
