"""
WY representation of a chunk: recompute ``w`` and ``u`` (gated DeltaNet, forward)
=================================================================================

The sequence is cut into chunks of ``BT = 64`` tokens. For each chunk, value head ``h`` and token
``t`` of the chunk, with ``A`` the chunk's precomputed ``BT x BT`` matrix (row ``t`` of it stored at
``A[b, t, h, :]``, column ``s`` the chunk's token ``s``):

* ``u[t] = sum_s A[t, s] * (beta[s] * v[s])``, with ``beta[s] * v[s]`` rounded to ``v``'s dtype;
* ``w[t] = sum_s A[t, s] * (beta[s] * exp2(g[s]) * k[s])``, rounded to ``k``'s dtype before the
  product (``g`` is the gate's chunk-local cumulative sum in base 2).

``s`` runs over the chunk's tokens inside the sequence. Value head ``h`` reads key head
``h // (HV // H)`` (grouped value attention when ``HV > H``). With ``cu_seqlens`` the batch is packed:
``B = 1`` and sequence ``n`` is the tokens ``cu_seqlens[n] .. cu_seqlens[n + 1] - 1``, cut into its own
chunks; ``chunk_indices`` lists the (sequence, chunk) pair of each program along grid axis 0.
A program handles one chunk of one value head (grid ``(NT, B * HV)``).

Layouts (contiguous):

* ``k``: ``[B, T, H, K]``; ``v``: ``[B, T, HV, V]`` (``bfloat16``);
* ``beta``, ``g``: ``[B, T, HV]`` (``float32``);
* ``A``: ``[B, T, HV, BT]`` (``bfloat16``, the dtype of ``k``);
* ``cu_seqlens`` (optional): ``[N + 1]``, ``int32`` or ``int64``; ``chunk_indices``: ``[NT, 2]``,
  the dtype of ``cu_seqlens``;
* ``w``: ``[B, T, HV, K]``, ``u``: ``[B, T, HV, V]``, freshly allocated (the dtypes of ``k``, ``v``).

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``.

* ``exp2``: ``fla/ops/utils/op.py:46-47`` (the branch taken when ``FLA_USE_FAST_OPS`` is not ``1``,
  its default).
* ``recompute_w_u_fwd_kernel``: ``fla/ops/gated_delta_rule/wy_fast.py:29-106`` (the kernel's
  ``def`` at line 43).
* ``prepare_lens``, ``_segmented_arange``, ``prepare_chunk_indices``: ``fla/ops/utils/index.py:40-42``,
  ``113-134`` and ``155-164``.
* ``recompute_w_u_fwd`` (the host): ``fla/ops/gated_delta_rule/wy_fast.py:254-294``.

Changes from the source:

* ``fla/ops/utils/op.py:46-47``: ``exp2`` is defined at the top level of this file (upstream it is
  defined under ``if os.environ.get('FLA_USE_FAST_OPS', '0') == '1': ... else:`` and imported by the
  kernel's module); its body is verbatim.
* ``fla/ops/gated_delta_rule/wy_fast.py:29-41``: the decorators ``@triton.heuristics({'USE_G': ...,
  'IS_VARLEN': ...})`` and ``@fla_cache_autotune(configs=[triton.Config({}, num_warps=num_warps,
  num_stages=num_stages) for num_warps in RECOMPUTE_W_U_NUM_WARPS for num_stages in [2, 3, 4]],
  key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'], **autotune_cache_kwargs)`` are removed
  (``RECOMPUTE_W_U_NUM_WARPS = [2, 4, 8]`` on NVIDIA GPUs, line 26). ``@triton.jit(do_not_specialize=['T'])``
  and the kernel's signature and body (lines 42-106) are verbatim.
* ``fla/ops/gated_delta_rule/wy_fast.py:254-294`` (``recompute_w_u_fwd``): the decorator ``@dispatch``
  (fla's backend registry, which falls back to this function on NVIDIA GPUs) is removed; the launch
  passes the flags the heuristics computed (``USE_G = g is not None``, ``IS_VARLEN = cu_seqlens is not
  None``) and one configuration of the autotune list, ``num_warps=4`` and ``num_stages=3``, as
  explicit arguments. The rest (the shapes, ``BT = A.shape[-1]``, ``BK = BV = 64``, ``chunk_indices``,
  ``NT``, the outputs' allocation, the grid and the other keywords) is the wrapper's.
* ``fla/ops/utils/index.py``: the decorator ``@tensor_cache`` is removed from ``prepare_lens`` and
  ``prepare_chunk_indices``, and ``_segmented_arange`` calls ``torch.nn.functional.pad`` for ``F.pad``.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
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
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_b = beta + bos*HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    p_A = A + (bos*HV + i_h) * BT + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_u = u + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    if USE_G:
        p_g = g + (bos*HV + i_h) + o_t * HV
        b_g = exp2(tl.load(p_g, mask=m_t, other=0.0))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_w = w + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


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


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    # One configuration of the autotune list (fla/ops/gated_delta_rule/wy_fast.py:33-41):
    # num_warps = 4, num_stages = 3; the flags the heuristics computed (lines 29-32).
    recompute_w_u_fwd_kernel[(NT, B*HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return w, u
