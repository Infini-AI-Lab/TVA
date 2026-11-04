"""
Causal depthwise convolution, forward
=====================================

The short convolution of linear-attention layers (fla's ``ShortConvolution``, used by Mamba,
DeltaNet, gated DeltaNet and others): each channel ``d`` of a sequence is convolved along time
with its own filter of width ``W``, causally,

    y[t, d] = act( bias[d] + sum_{w=0}^{W-1} weight[d, w] * x[t - (W - 1) + w, d] ) (+ residual[t, d]),

where ``x`` at a position before the sequence's start is ``0``, or, with an initial state (a
cache from earlier tokens), ``initial_state[n, d, W + t']`` for ``t' = t - (W - 1) + w < 0``. The
activation is SiLU (``x * sigmoid(x)``, named ``'silu'`` or ``'swish'``) or none. Sums are in
``float32``; the result is stored in ``y``'s dtype, rounded to nearest even.

Layouts (``x`` may be strided; the others contiguous):

* ``x``: ``[B, T, D]``, ``bfloat16``, strides ``(stride_x_n, stride_x_t, stride_x_d)``;
* ``weight``: ``[D, W]``; ``bias``: ``[D]``; ``residual``: ``[B, T, D]``; all ``bfloat16``;
* ``initial_state``: ``[N, D, W]``, ``bfloat16``, one per sequence (its last ``W - 1`` columns are
  the tokens before the sequence, oldest first; column ``0`` is not read);
* ``y``: ``[B, T, D]``, contiguous, of ``x``'s dtype;
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int32`` or ``int64``), the
  sequences' boundaries; ``chunk_indices``: ``[NT, 2]``, of the dtype of ``cu_seqlens``, one row
  ``(sequence, chunk within it)`` per chunk of ``BT`` tokens (``prepare_chunk_indices``).

One program per (block of ``BD`` channels, chunk of ``BT`` tokens, batch row): ``grid =
(cdiv(D, BD), NT, B)``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``causal_conv1d_fwd_kernel``: ``fla/modules/causal_conv1d/ops.py:50-147``;
* ``causal_conv1d_fwd``: ``fla/modules/causal_conv1d/ops.py:604-662``;
* ``causal_conv1d``: ``fla/modules/causal_conv1d/ops.py:995-1115`` (the ``backend='triton'``
  branch), through ``CausalConv1dFunction.forward``: ``fla/modules/causal_conv1d/ops.py:934-972``;
* ``prepare_lens``: ``fla/ops/utils/index.py:40-42``; ``_segmented_arange``:
  ``fla/ops/utils/index.py:113-134``; ``prepare_chunk_indices``: ``fla/ops/utils/index.py:155-164``.

Changes from the source:

* ``fla/modules/causal_conv1d/ops.py:34-49``: the decorators ``@triton.heuristics({'HAS_WEIGHT':
  ..., 'HAS_BIAS': ..., 'HAS_RESIDUAL': ..., 'USE_INITIAL_STATE': ..., 'IS_VARLEN': ...})`` and
  ``@fla_cache_autotune(configs=[triton.Config({'BD': BD}, num_warps=w) for BD in [16, 32, 64, 128]
  for w in NUM_WARPS_AUTOTUNE], key=['D', 'W'])`` (``NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]`` on
  NVIDIA, ``ops.py:30``) removed; ``@triton.jit`` kept. The kernel's signature and body are
  unchanged.
* ``fla/modules/causal_conv1d/ops.py:604-662``: the host ``causal_conv1d_fwd`` passes, as the
  removed heuristics did, ``HAS_WEIGHT``, ``HAS_BIAS``, ``HAS_RESIDUAL``, ``USE_INITIAL_STATE`` and
  ``IS_VARLEN`` (each ``True`` when its tensor is not ``None``), and one configuration of the
  removed autotune list, ``BD = 64`` with ``num_warps=8`` (its tenth entry, the autotuner's choice
  on an H200 for ``D`` = 2048, 4096 and 8192, ``W = 4``), and ``num_stages=3`` (a
  ``triton.Config``'s default), in place of the autotuner's choice (the grid uses that ``BD``). Its argument ``output_final_state`` is dropped (the final state is
  computed by another kernel, ``causal_conv1d_update_states``, ``ops.py:817-863``), and so is
  ``layout_fallback`` (unused by the Triton path). ``einops.rearrange(x, 'b t ... -> b t (...)')``
  (``ops.py:622``) is written ``x.reshape(x.shape[0], x.shape[1], -1)``. Its decorators
  ``@deprecate_kwarg('BT', ...)`` (a renamed keyword), ``@dispatch`` (backend selection,
  ``fla/backends.py``; the Triton default on NVIDIA GPUs) and ``@input_guard(no_guard_contiguous=["x"])``
  (every tensor argument but ``x`` made contiguous, ``fla/utils/decorators.py:97-157``) are replaced
  by ``.contiguous()`` calls on those arguments.
* ``fla/modules/causal_conv1d/ops.py:995-1115``: the entry point ``causal_conv1d`` keeps the path
  ``ShortConvolution.forward`` takes by default (``backend='triton'``, no context parallelism,
  ``ops.py:1118-1324``): ``chunk_indices`` prepared as ``CausalConv1dFunction.forward`` does, then
  ``causal_conv1d_fwd``. The ``'cuda'`` and ``'mix'`` backends and context parallelism call other
  code and are not kept.
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
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_y = (o_t < T)[:, None] & m_d[None, :]

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            # [BT, BD]
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # separate branches avoid a Triton compiler issue.
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            # [BT, BD]
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d, mask=m_x, other=0).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = residual + bos * D + o_t[:, None] * D + o_d[None, :]
        b_residual = tl.load(p_residual, mask=m_y, other=0.0)
        b_y += b_residual

    p_y = y + bos * D + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)


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


def _contiguous(t):
    # the source's @input_guard: tensor arguments made contiguous
    return t.contiguous() if isinstance(t, torch.Tensor) else t


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    weight, bias, residual, initial_state = map(_contiguous, (weight, bias, residual, initial_state))
    cu_seqlens, cu_seqlens_cpu, chunk_indices = map(_contiguous, (cu_seqlens, cu_seqlens_cpu, chunk_indices))
    BT = chunk_size
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = x.reshape(x.shape[0], x.shape[1], -1)
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]
    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    BD = 64  # one configuration of the removed autotune list

    def grid(meta) -> tuple[int, ...]:

        return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        ACTIVATION=activation,
        HAS_WEIGHT=weight is not None,  # the removed @triton.heuristics
        HAS_BIAS=bias is not None,
        HAS_RESIDUAL=residual is not None,
        USE_INITIAL_STATE=initial_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=8,
        num_stages=3,
    )
    return y.view(shape)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """The ``backend='triton'`` path of fla's ``causal_conv1d``, as ``ShortConvolution`` calls it."""
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
    return causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


def reference(x, weight, bias=None, initial_state=None, activation=None, cu_seqlens=None):
    """The causal depthwise convolution in float64, sequence by sequence."""
    B, T, D = x.shape
    W = weight.shape[1]
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
    w = weight.double()
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for n, (b, bos, eos) in enumerate(bounds):
        xs = x[b, bos:eos].double()  # [L, D]
        if initial_state is None:
            prev = torch.zeros((W - 1, D), dtype=torch.float64, device=x.device)
        else:
            prev = initial_state[n, :, 1:].double().T  # [W - 1, D], oldest first
        xf = torch.cat([prev, xs], 0)
        L = eos - bos
        acc = torch.zeros((L, D), dtype=torch.float64, device=x.device)
        for j in range(W):
            acc += w[:, j] * xf[j:j + L]
        if bias is not None:
            acc += bias.double()
        if activation in ('silu', 'swish'):
            acc = acc * torch.sigmoid(acc)
        out[b, bos:eos] = acc
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(shape, generator=gen).to(device=DEVICE, dtype=torch.bfloat16)

    cases = [
        # (B, T, D, W, cu_seqlens, bias, initial state, activation): the proved path first
        # (fla's tests/modules/test_conv.py test_conv_varlen_initial_state), then the defaults of
        # ShortConvolution (no bias, no cache) and fixed-length batches.
        (1, 256, 128, 4, [0, 100, 256], True, True, 'silu'),
        (1, 1, 128, 4, [0, 1], True, True, 'silu'),
        (1, 4096 + 8192, 2048, 4, [0] + [32 * (i + 1) for i in range(128)] + [4096 + 8192], True, True, 'silu'),
        (1, 3000, 8192, 4, [0, 1, 2, 3, 5, 67, 1000, 3000], True, True, 'swish'),
        (1, 500, 200, 3, [0, 63, 64, 65, 500], True, True, 'silu'),
        (1, 300, 256, 2, [0, 1, 300], True, True, 'silu'),
        (1, 1024, 4096, 4, [0, 17, 1024], False, False, 'silu'),
        (1, 2048, 6144, 4, [0, 1000, 2048], True, True, 'silu'),
        (2, 300, 1536, 4, None, False, True, 'silu'),
        (2, 300, 3072, 4, None, False, False, 'silu'),
        (2, 128, 128, 4, None, False, False, 'silu'),
        (3, 130, 1024, 4, None, True, True, 'silu'),
        (2, 64, 100, 3, None, True, True, None),
    ]
    for B, T, D, W, cu, has_bias, has_h0, activation in cases:
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEVICE)
        N = B if cu is None else len(cu) - 1
        x = randn(B, T, D)
        weight = randn(D, W)
        bias = randn(D) if has_bias else None
        h0 = randn(N, D, W) if has_h0 else None
        y = causal_conv1d(x, weight, bias=bias, initial_state=h0, activation=activation, cu_seqlens=cu_seqlens)
        ref = reference(x, weight, bias=bias, initial_state=h0, activation=activation, cu_seqlens=cu_seqlens)
        assert y.dtype == torch.bfloat16 and y.shape == x.shape
        err = ((y.double() - ref).abs() / (1 + ref.abs())).max().item()
        print(f"B {B} T {T} D {D} W {W} sequences {N if cu is not None else None} bias {has_bias} "
              f"initial state {has_h0} activation {activation}: max error {err:.2e} (relative to 1 + |ref|)")
        assert err < 1e-2, err
    print("✅ causal conv1d matches the float64 reference")
