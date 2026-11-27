"""
Global cumulative sum
=====================

The cumulative sum of a gate along time over each whole sequence: for a sequence of length ``T``
(one row of a batch, or one sequence of a packed variable-length batch), position ``t`` of head
``h`` gets the sum of ``s[0 .. t, h]`` (forward), or of ``s[t .. T - 1, h]`` (reverse), in
``float32``, times ``scale`` when one is given. One program per (sequence, head) walks the sequence
in blocks of ``BT`` tokens (from the last block backwards when reversed), takes ``tl.cumsum``
within each block and adds the running sum ``b_z`` of the blocks before it, which it then grows by
the block's ``tl.sum``. Attention layers with a forget gate (forgetting attention, PaTH attention,
WALL attention, simple GLA's backward, ...) use it for cumulative log decays, some with
``scale = 1/ln 2`` (``RCP_LN2``) so that later kernels can use ``exp2``.

Layouts (contiguous):

* ``s``: ``[B, T, H]``, ``float32`` (the input gate);
* ``z``: ``[B, T, H]``, ``float32`` (``output_dtype``, by default ``torch.float``);
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int32`` or ``int64``), the
  sequences' boundaries in the packed time axis.

One program per (sequence, head): ``grid = (N * H,)``, the program id split into
``(i_n, i_h) = (pid // H, pid % H)``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``chunk_global_cumsum_scalar_kernel``: ``fla/ops/utils/cumsum.py:150-191``;
* ``chunk_global_cumsum_scalar``: ``fla/ops/utils/cumsum.py:347-375``;
* ``chunk_global_cumsum``: ``fla/ops/utils/cumsum.py:412-448``.

Changes from the source:

* ``fla/ops/utils/cumsum.py:136-149``: the decorators ``@triton.heuristics({'HAS_SCALE': ...,
  'IS_VARLEN': ...})`` and ``@triton.autotune(configs=[triton.Config({'BT': BT},
  num_warps=num_warps, num_stages=num_stages) for BT in [32, 64, 128, 256] for num_warps in [2, 4,
  8] for num_stages in [1, 2, 3, 4]], key=['B', 'H', 'IS_VARLEN', 'REVERSE'])`` removed;
  ``@triton.jit(do_not_specialize=['T'])`` kept. The kernel's signature and body are unchanged.
* ``fla/ops/utils/cumsum.py:347-375``: the host ``chunk_global_cumsum_scalar`` passes, as the
  removed decorators did, ``HAS_SCALE = scale is not None`` and ``IS_VARLEN = cu_seqlens is not
  None``, and one configuration of the removed autotune list, ``BT=256``, ``num_warps=8``,
  ``num_stages=3`` (the autotuner's choice on an H200 for a packed batch of 65,536 tokens with 32
  heads, forward and reverse), in place of the autotuner's choice; its decorator ``@input_guard``
  (which makes every tensor argument contiguous, ``fla/utils/decorators.py:97-157``) is replaced by
  ``.contiguous()`` calls. Otherwise its argument handling, allocation, grid and arguments are the
  source's.
* ``fla/ops/utils/cumsum.py:412-448``: the entry point ``chunk_global_cumsum`` keeps its checks and
  its dispatch, except that its 4-D branch, which launches another kernel
  (``chunk_global_cumsum_vector_kernel``), raises ``NotImplementedError``; its decorators
  ``@input_guard`` and ``@dispatch`` (backend selection, ``fla/backends.py``; the Triton default on
  NVIDIA GPUs) are replaced by ``.contiguous()`` calls on the tensor arguments.
* The reference (``float64``) and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_s = s + bos*H + i_h + o_t * H
        p_o = o + bos*H + i_h + o_t * H
        b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
        if REVERSE:
            b_o = tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    # the source's @input_guard: every tensor argument made contiguous
    s = s.contiguous()
    cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (N * H,)
    chunk_global_cumsum_scalar_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,  # the removed @triton.heuristics
        IS_VARLEN=cu_seqlens is not None,  # the removed @triton.heuristics
        BT=256,  # one configuration of the removed autotune list
        num_warps=8,
        num_stages=3,
    )
    return z


def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    # the source's @input_guard: every tensor argument made contiguous
    s = s.contiguous()
    cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_dtype=output_dtype,
        )
    elif len(s.shape) == 4:
        raise NotImplementedError("the [B, T, H, D] path launches chunk_global_cumsum_vector_kernel, not kept here")
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H] or [B, T, H, D]",
        )


def reference(s, reverse=False, scale=None, cu_seqlens=None):
    """The global cumulative sum in float64, sequence by sequence."""
    B, T, H = s.shape
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
    out = torch.zeros(s.shape, dtype=torch.float64, device=s.device)
    for b, bos, eos in bounds:
        x = s[b, bos:eos].double()
        y = x.flip(0).cumsum(0).flip(0) if reverse else x.cumsum(0)
        out[b, bos:eos] = y if scale is None else y * scale
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    RCP_LN2 = 1.4426950216  # fla/ops/utils/constant.py:10
    gen = torch.Generator().manual_seed(0)
    cases = [
        # (B, T, H, cu_seqlens): fixed-length rows, then packed batches as fla's attention layers pass them
        (1, 63, 1, None),
        (2, 500, 4, None),
        (2, 1000, 5, None),
        (3, 1024, 6, None),
        (4, 2048, 8, None),
        (1, 15, 2, [0, 15]),
        (1, 1000, 3, [0, 256, 500, 1000]),
        (1, 2000, 4, [0, 15, 100, 300, 1200, 2000]),
        (1, 2048, 4, [0, 1, 100, 300, 1200, 2048]),
        (1, 1, 32, [0, 1]),
        (1, 65536, 32, [0, 1, 64, 65, 4096, 30000, 65536]),
    ]
    for B, T, H, cu in cases:
        s = torch.nn.functional.logsigmoid(torch.randn((B, T, H), generator=gen)).to(DEVICE)
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32, device=DEVICE)
        worst = 0.0
        for reverse in (False, True):
            for scale in (None, RCP_LN2):
                out = chunk_global_cumsum(s, reverse=reverse, scale=scale, cu_seqlens=cu_seqlens)
                ref = reference(s, reverse=reverse, scale=scale, cu_seqlens=cu_seqlens)
                assert out.dtype == torch.float32
                err = ((out.double() - ref).abs() / (1 + ref.abs())).max().item()
                assert err < 1e-4, err
                worst = max(worst, err)
        print(f"B {B} T {T} H {H} cu_seqlens {cu}: forward/reverse, scale None/RCP_LN2: "
              f"max error {worst:.2e} (relative to 1 + |ref|)")
    print("✅ global cumsum matches the float64 reference")
