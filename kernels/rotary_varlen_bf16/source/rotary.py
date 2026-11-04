"""
Rotary position embedding, forward (out of place)
=================================================

Each head vector ``x[b, t, h, :]`` is rotated by the angles of its position: with ``R`` the number
of rotated pairs (``2 R <= D``), pair ``i`` of the vector at position ``p`` becomes
``(x0 cos - x1 sin, x0 sin + x1 cos)`` with ``cos = cos[p, i]`` and ``sin = sin[p, i]``. Not
interleaved (GPT-NeoX), pair ``i`` is channels ``(i, i + R)``; interleaved (GPT-J), channels
``(2i, 2i + 1)``. The channels from ``2 R`` on are copied (by the host). The position is
``p = t + offset``: one offset for every sequence (an ``int``) or one per sequence (a tensor). With
``cu_seqlens``, the batch is packed: ``x`` is ``[1, T, H, D]`` and sequence ``n`` is the tokens
``cu_seqlens[n] .. cu_seqlens[n + 1] - 1``, each with its own positions from 0. The computation is
in ``float32``. A program handles ``BT`` tokens of one head of one sequence; with ``cu_seqlens``,
``chunk_indices`` lists the (sequence, chunk) pair of each program along grid dimension 0.

Layouts (contiguous):

* ``x``: ``[B, T, H, D]``, ``bfloat16`` (``B = 1`` with ``cu_seqlens``);
* ``cos``, ``sin``: ``[TR, R]``, the dtype of ``x``;
* ``seqlen_offsets``: an ``int``, or ``[N]`` ``int32``/``int64``;
* ``cu_seqlens`` (optional): ``[N + 1]``, ``int32`` or ``int64``;
* ``chunk_indices``: ``[NT, 2]``, the dtype of ``cu_seqlens``;
* ``y``: ``[B, T, H, D]``, the dtype of ``x``, freshly allocated.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the kernel is lines 63-154.

* ``rotary_embedding_kernel``: ``fla/modules/rotary/ops.py:21-121`` (the kernel's ``def`` at
  line 31).
* ``rotary_embedding_fwd`` (the host): adapted from ``rotary_embedding_fwdbwd``,
  ``fla/modules/rotary/ops.py:144-241``.
* ``get_multiprocessor_count``: from ``fla/utils/hardware.py:30-43``.
* ``prepare_lens``, ``_segmented_arange``, ``prepare_chunk_indices``: from
  ``fla/ops/utils/index.py:40-42``, ``113-134`` and ``155-164``.

Changes from the source:

* ``fla/modules/rotary/ops.py:21-29``: the decorator ``@triton.autotune(...)`` (over ``num_warps``
  in ``[2, 4, 8, 16, 32]`` and ``num_stages`` in ``[2, 3, 4]``, key
  ``['B', 'H', 'D', 'INTERLEAVED']``) is removed. ``@triton.jit(do_not_specialize=['T'])`` and the
  kernel's signature and body are verbatim.
* ``fla/modules/rotary/ops.py:144-241`` (``rotary_embedding_fwdbwd``): the decorator ``@dispatch``
  (fla's backend registry, which falls back to this function on NVIDIA GPUs) is removed; the
  function is renamed ``rotary_embedding_fwd``; the argument ``inplace`` and its path
  (``y = x``) are removed, so ``y`` is always ``torch.zeros_like(x)`` (TVA does not cover in-place
  updates); the launch passes one configuration of the autotune list, ``num_warps=4`` and
  ``num_stages=2``, as explicit arguments. The rest (the asserts, ``y``'s allocation and the copy
  of the channels past ``2 R``, ``BD``, ``BT``, ``chunk_indices``, the grid, the flags and the
  launch's keywords) is the wrapper's.
* ``fla/utils/hardware.py:30-43`` (``get_multiprocessor_count``): the decorator ``@cache`` and the
  fallback for NPU devices are removed.
* ``fla/ops/utils/index.py``: the decorator ``@tensor_cache`` is removed from ``prepare_lens`` and
  ``prepare_chunk_indices``, and ``_segmented_arange`` calls ``torch.nn.functional.pad`` for
  ``F.pad``.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def rotary_embedding_kernel(
    x,
    cos,
    sin,
    y,
    cu_seqlens,
    chunk_indices,
    seq_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    TR: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    i_t, i_b, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        x = x + bos * H*D + i_h * D
        y = y + bos * H*D + i_h * D
    else:
        i_n = i_b
        x = x + i_n * T*H*D + i_h * D
        y = y + i_n * T*H*D + i_h * D

    if i_t * BT >= T:
        return

    o_t = i_t * BT + tl.arange(0, BT)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        o_cs = o_t + seq_offsets
    else:
        o_cs = o_t + tl.load(seq_offsets + i_n)
    m_t = (o_t >= 0) & (o_t < T) & (o_cs >= 0) & (o_cs < TR)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of x, do calculation, then store to 1st and 2nd halves of out
        o_r = tl.arange(0, BD // 2)
        p_x = x + o_t[:, None] * H*D + o_r[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_r[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_r[None, :])
        mask = m_t[:, None] & (o_r < R)[None, :]

        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x + R, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos - b_x1 * b_sin
        b_o1 = b_x0 * b_sin + b_x1 * b_cos
        p_y = y + (o_t[:, None] * H*D + o_r[None, :])
        tl.store(p_y, b_o0, mask=mask)
        tl.store(p_y + R, b_o1, mask=mask)
    else:
        # We don't want to load x[0, 2, 4, ...] and x[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = x[0, 1, 2, 3, ...] and x1 = x[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = cos[0, 0, 1, 1, ...] and sin = sin[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        o_d = tl.arange(0, BD)
        # pair adjacent channels: 1, 0, 3, 2, 5, 4, ...
        o_d_swap = o_d + ((o_d + 1) % 2) * 2 - 1
        o_d_repeat = tl.arange(0, BD) // 2
        p_x0 = x + o_t[:, None] * H*D + o_d[None, :]
        p_x1 = x + o_t[:, None] * H*D + o_d_swap[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_d_repeat[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_d_repeat[None, :])
        mask = m_t[:, None] & (o_d_repeat < R)[None, :]

        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x0, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x1, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos
        b_o1 = b_x1 * b_sin
        b_y = tl.where(o_d[None, :] % 2 == 0, b_o0 - b_o1, b_o0 + b_o1)
        p_y = y + (o_t[:, None] * H*D + o_d[None, :])
        tl.store(p_y, b_y, mask=mask)


def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']


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


def rotary_embedding_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    interleaved: bool = False,
    conjugate: bool = False,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None

    B, T, H, D = x.shape
    N = B if not is_varlen else cu_seqlens.shape[0] - 1
    TR, R = cos.shape
    R2 = R * 2

    assert D <= 256, "Only support D <= 256"
    assert R2 <= D, f"Rotary dimension must not exceed head dimension, got rotary_dim={R2} and head_dim={D}"
    if not is_varlen:
        assert TR >= T, f"TR must be >= T, got {TR} and {T}"

    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert x.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (N,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        sequence_lengths = T if not is_varlen else cu_seqlens[1:] - cu_seqlens[:-1]
        torch._assert_async(
            torch.all(sequence_lengths + seqlen_offsets <= TR),
            f"Rotary cache is too short for tensor offsets, got {TR} positions",
        )
    else:
        assert seqlen_offsets + T <= TR

    # zeros_like: rows the kernel skips (negative o_cs under left-padding) must be defined, not uninitialized.
    y = torch.zeros_like(x)
    if R2 < D:
        y[..., R2:].copy_(x[..., R2:])

    BD = triton.next_power_of_2(R2)
    BT = min(128, triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index))))
    if chunk_indices is None and is_varlen:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if is_varlen else triton.cdiv(T, BT)

    grid = (NT, B, H)
    # One configuration of the autotune list at fla/modules/rotary/ops.py:22-26: num_warps = 4, num_stages = 2.
    rotary_embedding_kernel[grid](
        x=x,
        cos=cos,
        sin=sin,
        y=y,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        seq_offsets=seqlen_offsets,
        T=T,
        B=B,
        H=H,
        D=D,
        R=R,
        TR=TR,
        BT=BT,
        BD=BD,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=is_varlen,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        num_warps=4,
        num_stages=2,
    )
    return y


def reference(x, cos, sin, seqlen_offsets=0, cu_seqlens=None, interleaved=False):
    """The rotation in float64, sequence by sequence."""
    B, T, H, D = x.shape
    R = cos.shape[1]
    xd = x.double().reshape(B * T, H, D)
    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = list(zip(cu[:-1], cu[1:]))
    out = xd.clone()
    for n, (bos, eos) in enumerate(bounds):
        off = int(seqlen_offsets[n]) if isinstance(seqlen_offsets, torch.Tensor) else seqlen_offsets
        pos = torch.arange(eos - bos, device=x.device) + off
        c = cos[pos].double()[:, None, :]  # [len, 1, R]
        s = sin[pos].double()[:, None, :]
        seg = xd[bos:eos]
        if not interleaved:
            x0, x1 = seg[..., :R], seg[..., R:2 * R]
            out[bos:eos, :, :R] = x0 * c - x1 * s
            out[bos:eos, :, R:2 * R] = x0 * s + x1 * c
        else:
            x0, x1 = seg[..., 0:2 * R:2], seg[..., 1:2 * R:2]
            out[bos:eos, :, 0:2 * R:2] = x0 * c - x1 * s
            out[bos:eos, :, 1:2 * R:2] = x0 * s + x1 * c
    return out.reshape(B, T, H, D)


def cos_sin_cache(TR, rotary_dim, device, base=10000.0):
    """The tables of fla's RotaryEmbedding (fla/modules/rotary/ops.py:435-468), in bfloat16."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(TR, dtype=torch.float32), inv_freq)
    return (torch.cos(freqs).to(device=device, dtype=torch.bfloat16),
            torch.sin(freqs).to(device=device, dtype=torch.bfloat16))


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)

    def random_cu_seqlens(T, N, dtype):
        """N sequences packed in T tokens, as fla's tests/modules/test_rotary.py:102-109 draws them."""
        cuts = torch.arange(1, T)[torch.randperm(T - 1, generator=gen)[:N - 1]]
        return torch.cat([torch.tensor([0]), cuts, torch.tensor([T])]).sort()[0].to(device=DEVICE, dtype=dtype)

    # (B, T, H, D, rotary_dim, offsets, cu_seqlens): fla's tests (tests/modules/test_rotary.py: B = 2
    # or 4, T = 2048 or 4096, H = 4, D = 128 or 256, partial rotary_dim = 64, N = 4 packed
    # sequences), head dim 64, one token, offsets, and a long batch.
    cases = []
    for interleaved in (False, True):
        cases += [
            (2, 2048, 4, 128, 128, 0, None, interleaved),
            (2, 4096, 4, 256, 256, 0, None, interleaved),
            (1, 1, 1, 64, 64, 0, None, interleaved),
            (3, 1000, 8, 64, 64, 7, None, interleaved),
            (2, 256, 4, 128, 64, 0, None, interleaved),
            (1, 2048, 4, 128, 128, 0, (4, torch.int64), interleaved),
            (1, 4096, 4, 256, 256, 0, (4, torch.int32), interleaved),
            (1, 777, 2, 64, 32, 0, (9, torch.int32), interleaved),
            (1, 5, 2, 128, 128, 0, (5, torch.int64), interleaved),
            (4, 8192, 32, 128, 128, 0, None, interleaved),
            (1, 32768, 32, 128, 128, 0, (16, torch.int32), interleaved),
        ]
    for B, T, H, D, rd, off, varlen, interleaved in cases:
        cu_seqlens = None if varlen is None else random_cu_seqlens(T, varlen[0], varlen[1])
        TR = T + off
        cos, sin = cos_sin_cache(TR, rd, DEVICE)
        x = torch.randn((B, T, H, D), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        y = rotary_embedding_fwd(x, cos, sin, seqlen_offsets=off, cu_seqlens=cu_seqlens, interleaved=interleaved)
        ref = reference(x, cos, sin, seqlen_offsets=off, cu_seqlens=cu_seqlens, interleaved=interleaved)
        assert y.dtype == torch.bfloat16 and y.shape == x.shape and y.data_ptr() != x.data_ptr()
        assert torch.equal(y[..., rd:], x[..., rd:])
        err = (y.double() - ref).abs().max().item()
        rel = ((y.double() - ref).abs() / (ref.abs() + 1e-3)).max().item()
        n = "-" if cu_seqlens is None else f"{varlen[0]} seqs ({varlen[1]})".replace("torch.", "")
        print(f"interleaved {interleaved} B {B} T {T} H {H} D {D} rotary_dim {rd} offset {off} varlen {n}: "
              f"max abs error {err:.2e}, max rel error {rel:.2e}")
        # bfloat16 rounds to 2^-9 relative.
        assert rel < 5e-3, rel
    print("rotary_embedding_fwd matches the float64 reference")
