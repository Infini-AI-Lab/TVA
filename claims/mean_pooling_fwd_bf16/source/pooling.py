"""
Mean pooling over chunks
========================

The mean of every chunk of ``BT`` consecutive tokens, per head and feature: for a sequence of
length ``T`` (one row of a batch, or one sequence of a packed variable-length batch), chunk ``c``
holds the tokens ``c * BT .. min(c * BT + BT, T) - 1`` (chunks counted from the sequence's start,
the last one possibly shorter), and ``o[c, h, d]`` is the mean of ``x[t, h, d]`` over them, summed
in ``float32`` and stored in ``x``'s dtype. Native sparse attention (NSA) uses it to compress the
keys and values of each block of ``block_size`` tokens.

Layouts (contiguous):

* ``x``: ``[B, T, H, D]`` (the input, ``bfloat16`` in the claim);
* ``o``: ``[B, NT, H, D]`` of ``x``'s dtype, ``NT = cdiv(T, BT)`` (fixed length) or the number of
  chunks of the batch (variable length);
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int32`` or ``int64``), the
  sequences' boundaries in the packed time axis; ``chunk_indices``: ``[NT, 2]``, of the dtype of
  ``cu_seqlens``, one row ``(sequence, chunk within it)`` per chunk of the batch
  (``prepare_chunk_indices``); output row ``r`` is the chunk row ``r`` names.

One program per (feature block, chunk) and (batch row, head): ``grid = (cdiv(D, BD) * NT, B * H)``;
axis 0 is split by ``unflatten_program_id(cdiv(D, BD))`` into ``(i_d, i_t)``, ``i_d`` fastest.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``mean_pooling_fwd_kernel``: ``fla/ops/utils/pooling.py:17-65``;
* ``mean_pooling_fwd``: ``fla/ops/utils/pooling.py:119-143``;
* ``mean_pooling``: ``fla/ops/utils/pooling.py:207-224``;
* ``unflatten_program_id``: ``fla/ops/utils/op.py:22-29``;
* ``prepare_lens``: ``fla/ops/utils/index.py:40-42``; ``_segmented_arange``:
  ``fla/ops/utils/index.py:113-134``; ``prepare_chunk_indices``: ``fla/ops/utils/index.py:155-164``.

Changes from the source:

* ``fla/ops/utils/pooling.py:17-28``: the decorators ``@triton.heuristics({'IS_VARLEN': lambda args:
  args['cu_seqlens'] is not None})`` and ``@triton.autotune(configs=[triton.Config({'BD': BD},
  num_warps=num_warps) for BD in [16, 32, 64, 128] for num_warps in [1, 2, 4, 8]], key=['BT'],
  **autotune_cache_kwargs)`` removed; ``@triton.jit(do_not_specialize=['T'])`` kept. The kernel's
  signature and body are unchanged.
* ``fla/ops/utils/op.py:22-29``: the helper ``unflatten_program_id`` copied verbatim (the kernel
  calls it).
* ``fla/ops/utils/pooling.py:119-143``: the host ``mean_pooling_fwd`` passes, as the removed
  decorators did, ``IS_VARLEN = cu_seqlens is not None``, and one configuration of the removed
  autotune list, ``BD=64, num_warps=2`` (the autotuner's choice on an H200 for a packed batch of
  65,536 tokens, 4 heads of 128 features, 64-token chunks, ``bfloat16``), with ``num_stages=3`` (a
  ``triton.Config``'s default), in place of the autotuner's choice; its grid uses that ``BD``.
  Otherwise its argument handling, allocation, grid and arguments are the source's.
* ``fla/ops/utils/pooling.py:207-224``: the entry point ``mean_pooling`` keeps its checks and calls
  ``mean_pooling_fwd`` directly, in place of ``MeanPoolingFunction.apply``
  (``pooling.py:175-204``, the autograd function, whose forward is ``mean_pooling_fwd`` under the
  decorators ``@input_guard``, which makes every tensor argument contiguous,
  ``fla/utils/decorators.py:97-157``, and ``@autocast_custom_fwd``, ``torch.amp.custom_fwd``, which
  changes nothing outside an autocast region); the tensor arguments are made contiguous by
  ``.contiguous()`` calls. The backward kernel ``mean_pooling_bwd_kernel``
  (``pooling.py:68-116``), its host ``mean_pooling_bwd`` (``pooling.py:146-172``) and the autograd
  function are not kept.
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
def unflatten_program_id(X: tl.constexpr, Y: tl.constexpr = None):
    """Recover int64 x-fastest 2D/3D indices without using CUDA's limited y/z grid dimensions."""
    pid = tl.program_id(0).to(tl.int64)
    if Y is None:
        return pid % X, pid // X
    else:
        return pid % X, (pid // X) % Y, pid // (X * Y)


@triton.jit(do_not_specialize=['T'])
def mean_pooling_fwd_kernel(
    x,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t = unflatten_program_id(tl.cdiv(D, BD))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
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

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_x = (o_t[:, None] < T) & (o_d[None, :] < D)
    p_x = x + (bos * H + i_h) * D + o_t[:, None] * (H*D) + o_d[None, :]
    p_o = o + (i_tg * H + i_h) * D + o_d
    # [BT, BD]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    # [BD]
    b_o = tl.sum(b_x, axis=0) / min(BT, T - i_t * BT)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=o_d < D)


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


def mean_pooling_fwd(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, D = x.shape
    BT = chunk_size
    BD = 64  # one configuration of the removed autotune list
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = x.new_empty(B, NT, H, D)
    def grid(meta): return (triton.cdiv(D, meta['BD']) * NT, B * H)
    mean_pooling_fwd_kernel[grid](
        x,
        o,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
        BD=BD,  # one configuration of the removed autotune list
        IS_VARLEN=cu_seqlens is not None,  # the removed @triton.heuristics
        num_warps=2,
        num_stages=3,
    )
    return o


def mean_pooling(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    # the source's @input_guard (on MeanPoolingFunction.forward): tensor arguments made contiguous
    x = x.contiguous()
    cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {x.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
    o = mean_pooling_fwd(x, chunk_size, cu_seqlens)  # the source's MeanPoolingFunction.apply(x, chunk_size, cu_seqlens)
    return o


def reference(x, chunk_size, cu_seqlens=None):
    """The chunk means in float64, sequence by sequence, in the output's row order."""
    B, T, H, D = x.shape
    if cu_seqlens is None:
        rows = [[x[b, c:min(c + chunk_size, T)].double().mean(0) for c in range(0, T, chunk_size)]
                for b in range(B)]
        return torch.stack([torch.stack(r) for r in rows]) if T > 0 else x.new_zeros(B, 0, H, D).double()
    cu = cu_seqlens.tolist()
    rows = [x[0, c:min(c + chunk_size, cu[n + 1])].double().mean(0)
            for n in range(len(cu) - 1) for c in range(cu[n], cu[n + 1], chunk_size)]
    return torch.stack(rows)[None]


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    cases = [
        # (B, T, H, chunk_size, D, cu_seqlens): fla's tests/ops/utils/test_pooling.py, in bfloat16
        (1, 63, 1, 16, 30, None),
        (2, 500, 4, 32, 60, None),
        (2, 1000, 5, 64, 128, None),
        (3, 1024, 6, 64, 500, None),
        (4, 2048, 8, 128, 1024, None),
        (1, 15, 2, 32, 60, [0, 15]),
        (1, 1000, 3, 64, 100, [0, 256, 500, 1000]),
        (1, 2000, 4, 64, 256, [0, 15, 100, 300, 1200, 2000]),
        (1, 2048, 4, 128, 500, [0, 1, 100, 300, 1200, 2048]),
        (1, 2048, 2, 128, 1024, [0, 200, 512, 1200, 2048]),
    ]
    for B, T, H, chunk_size, D, cu in cases:
        x = torch.randn((B, T, H, D), generator=gen).to(DEVICE, torch.bfloat16)
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEVICE)
        out = mean_pooling(x, chunk_size, cu_seqlens=cu_seqlens)
        ref = reference(x, chunk_size, cu_seqlens=cu_seqlens)
        assert out.dtype == torch.bfloat16 and out.shape == ref.shape
        err = ((out.double() - ref).abs() / (2.0**-8 * ref.abs() + 1e-6)).max().item()
        assert err <= 1.0, err
        print(f"B {B} T {T} H {H} chunk {chunk_size} D {D} cu_seqlens {cu}: "
              f"max error {err:.3f} (relative to 2^-8 |ref| + 1e-6)")
    print("✅ mean pooling matches the float64 reference")
