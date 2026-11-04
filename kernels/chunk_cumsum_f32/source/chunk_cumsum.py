"""
Chunk-local cumulative sum
==========================

The cumulative sum of a gate along time, restarted at every chunk of ``BT`` tokens: for a sequence
of length ``T`` (one row of a batch, or one sequence of a packed variable-length batch), position
``t`` of head ``h`` gets the sum of ``s[t0 .. t, h]`` with ``t0 = (t // BT) * BT`` the start of
its chunk (forward), or of ``s[t .. min(t0 + BT, T) - 1, h]`` (reverse), in ``float32``, times
``scale`` when one is given. Chunks are counted from each sequence's start. Linear-attention
layers (gated DeltaNet, gated DeltaProduct, ...) use it to turn per-token log decays into within-chunk cumulative
decays; gated DeltaNet's forward calls it with ``scale = 1/ln 2`` (``RCP_LN2``) so that later
kernels can use ``exp2``.

Layouts (contiguous):

* ``s``: ``[B, T, H]``, ``float32`` (the input gate);
* ``o``: ``[B, T, H]``, ``float32`` (``output_dtype``, by default ``torch.float``);
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int32`` or ``int64``), the
  sequences' boundaries in the packed time axis; ``chunk_indices``: ``[NT, 2]``, of the dtype of
  ``cu_seqlens``, one row ``(sequence, chunk within it)`` per chunk of the batch
  (``prepare_chunk_indices``).

One program per (chunk, batch row, head): ``grid = (NT * B * H,)``; the program id is split by
``unflatten_program_id(B * H)`` into ``(i_bh, i_t)``, ``i_bh`` fastest.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``chunk_local_cumsum_scalar_kernel``: ``fla/ops/utils/cumsum.py:33-73``;
* ``unflatten_program_id``: ``fla/ops/utils/op.py:22-29``;
* ``chunk_local_cumsum_scalar``: ``fla/ops/utils/cumsum.py:253-293``;
* ``chunk_local_cumsum``: ``fla/ops/utils/cumsum.py:451-496``;
* ``prepare_lens``: ``fla/ops/utils/index.py:40-42``; ``_segmented_arange``:
  ``fla/ops/utils/index.py:113-134``; ``prepare_chunk_indices``: ``fla/ops/utils/index.py:155-164``.

Changes from the source:

* ``fla/ops/utils/cumsum.py:21-32``: the decorators ``@triton.heuristics({'HAS_SCALE': ...,
  'IS_VARLEN': ...})`` and ``@fla_cache_autotune(configs=[triton.Config({}, num_warps=w) for w in
  [1, 2, 4, 8]], key=[...])`` removed; ``@triton.jit(do_not_specialize=['T'])`` kept.
* ``fla/ops/utils/cumsum.py:47``: the default of the kernel's last parameter removed,
  ``USE_GRAPH: tl.constexpr = False`` written ``USE_GRAPH: tl.constexpr`` (the host passes it, as
  fla's host already does). The rest of the kernel's signature and its body are unchanged.
* ``fla/ops/utils/op.py:22-29``: the helper ``unflatten_program_id`` copied verbatim (the kernel
  calls it).
* ``fla/ops/utils/cumsum.py:253-293``: the host ``chunk_local_cumsum_scalar`` passes, as the
  removed decorators did, ``HAS_SCALE = scale is not None`` and ``IS_VARLEN = cu_seqlens is not
  None``, and one configuration of the removed autotune list, ``num_warps=2`` (its second entry, the
  autotuner's choice on an H200 for 32 heads and 64-token chunks), with ``num_stages=3`` (a
  ``triton.Config``'s default), in place of the autotuner's choice. Otherwise its argument handling, allocation, grid and arguments are the
  source's.
* ``fla/ops/utils/cumsum.py:451-496``: the entry point ``chunk_local_cumsum`` keeps its checks and
  its dispatch, except that its 4-D branch, which launches another kernel
  (``chunk_local_cumsum_vector_kernel``), raises ``NotImplementedError``; its decorators ``@input_guard`` (which makes every tensor
  argument contiguous, ``fla/utils/decorators.py:97-157``) and ``@dispatch`` (backend selection,
  ``fla/backends.py``; the Triton default on NVIDIA GPUs) are replaced by ``.contiguous()`` calls on
  the tensor arguments.
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
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr,
):
    i_bh, i_t = unflatten_program_id(B * H)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_s = s + bos*H + i_h + o_t * H
    p_o = o + bos*H + i_h + o_t * H
    # [BT]
    b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


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


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
    **kwargs,
) -> torch.Tensor:
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if use_graph:
        # zero inactive rows: kda_gate_bwd reduces the entire output buffer
        o = torch.zeros_like(g, dtype=output_dtype or g.dtype)
    else:
        o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT * B * H,)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,  # the removed @triton.heuristics
        IS_VARLEN=cu_seqlens is not None,  # the removed @triton.heuristics
        USE_GRAPH=use_graph,
        num_warps=2,  # one configuration of the removed autotune list
        num_stages=3,
    )
    return o


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
    **kwargs,
) -> torch.Tensor:
    # the source's @input_guard: every tensor argument made contiguous
    g = g.contiguous()
    cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None
    chunk_indices = chunk_indices.contiguous() if chunk_indices is not None else None
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
            use_graph=use_graph,
        )
    elif len(g.shape) == 4:
        raise NotImplementedError("the (B, T, H, D) path launches chunk_local_cumsum_vector_kernel, not kept here")
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H) or (B, T, H, D)",
        )


def reference(g, chunk_size, reverse=False, scale=None, cu_seqlens=None):
    """The chunk-local cumulative sum in float64, sequence by sequence."""
    B, T, H = g.shape
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
    out = torch.zeros(g.shape, dtype=torch.float64, device=g.device)
    for b, bos, eos in bounds:
        for c in range(bos, eos, chunk_size):
            x = g[b, c:min(c + chunk_size, eos)].double()
            y = x.flip(0).cumsum(0).flip(0) if reverse else x.cumsum(0)
            out[b, c:min(c + chunk_size, eos)] = y if scale is None else y * scale
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    RCP_LN2 = 1.4426950216  # fla/ops/utils/constant.py:10
    gen = torch.Generator().manual_seed(0)
    cases = [
        # (B, T, H, chunk_size, cu_seqlens): fla's tests/ops/utils/test_cumsum.py, then gated DeltaNet's use
        (1, 63, 1, 16, None),
        (2, 500, 4, 32, None),
        (2, 1000, 5, 64, None),
        (3, 1024, 6, 64, None),
        (4, 2048, 8, 128, None),
        (1, 15, 2, 16, [0, 15]),
        (1, 15, 2, 32, [0, 15]),
        (1, 1000, 3, 64, [0, 256, 500, 1000]),
        (1, 2000, 4, 64, [0, 15, 100, 300, 1200, 2000]),
        (1, 2048, 4, 128, [0, 1, 100, 300, 1200, 2048]),
        (1, 1, 32, 64, [0, 1]),
        (1, 65536, 32, 64, [0, 1, 64, 65, 4096, 30000, 65536]),
    ]
    for B, T, H, chunk_size, cu in cases:
        g = torch.nn.functional.logsigmoid(torch.randn((B, T, H), generator=gen)).to(DEVICE)
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEVICE)
        worst = 0.0
        for reverse in (False, True):
            for scale in (None, RCP_LN2):
                out = chunk_local_cumsum(g, chunk_size, reverse=reverse, scale=scale, cu_seqlens=cu_seqlens)
                ref = reference(g, chunk_size, reverse=reverse, scale=scale, cu_seqlens=cu_seqlens)
                assert out.dtype == torch.float32
                err = ((out.double() - ref).abs() / (1 + ref.abs())).max().item()
                assert err < 1e-5, err
                worst = max(worst, err)
        print(f"B {B} T {T} H {H} chunk {chunk_size} cu_seqlens {cu}: forward/reverse, scale None/RCP_LN2: "
              f"max error {worst:.2e} (relative to 1 + |ref|)")
    print("✅ chunk-local cumsum matches the float64 reference")
