"""
Mamba-2 SSD per-chunk C.B^T (SGLang's ``_bmm_chunk_fwd_kernel``), bfloat16 inputs
================================================================================

Step 4 of Mamba-2's chunked scan (SSD). The prefill tokens of all requests are packed into one
sequence of ``seqlen`` tokens (``batch = 1``) and cut into ``nchunks = cdiv(seqlen, chunk_size)``
chunks, which may cross request boundaries. For chunk ``c``, group ``g`` and positions
``0 <= m, n < chunk_size`` (tokens ``tm = c * chunk_size + m``, ``tn = c * chunk_size + n``):

    out[0, c, g, m, n] = sum_k a[0, tm, g, k] * b[0, tn, g, k]
                           if tm, tn < seqlen and seq_idx[0, tm] == seq_idx[0, tn]
                       = 0 otherwise (another request's token, or past the end)

``ssd_combined.py`` calls it as ``CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx,
output_dtype=torch.float32)`` (``causal = False``): ``a`` is ``C``, ``b`` is ``B``, both
``[1, seqlen, ngroups, dstate]`` ``bfloat16`` with ``stride(-1) = 1``; ``seq_idx`` is
``[1, seqlen]`` ``int32`` (the request of each token); ``out`` is allocated fresh,
``[1, nchunks, ngroups, chunk_size, chunk_size]`` ``float32``. Programs:
``(cdiv(chunk_size, BLOCK_SIZE_M) * cdiv(chunk_size, BLOCK_SIZE_N), batch, nchunks * ngroups)``.

Provenance: SGLang v0.5.23 (commit ``297e54fd10``),
``python/sglang/kernels/ops/mamba/triton_ops/ssd_bmm.py``:

* ``_bmm_chunk_fwd_kernel``: ``ssd_bmm.py:20-190`` (``@triton.autotune`` at 20-79, ``@triton.jit``
  at 80, ``def`` at 81);
* ``_bmm_chunk_fwd`` (host wrapper): ``ssd_bmm.py:193-276``;
* its call: ``python/sglang/kernels/ops/mamba/triton_ops/ssd_combined.py:165``;
* ``seq_idx``: ``python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:265-275``.

Changes from the source (every other line of ``ssd_bmm.py:80-190`` is copied verbatim, at the
same line numbers):

* ``ssd_bmm.py:1-19``: the header comments and imports are replaced by this docstring and the
  imports below; ``from .autotune import autotune_cache_kwargs, prune_oversized_tiles``
  (``ssd_bmm.py:17``) is removed (nothing here uses it).
* ``ssd_bmm.py:20-79``: ``@triton.autotune`` removed (nine configs, ``key=["chunk_size", "K",
  "IS_CAUSAL"]``, pruned by ``prune_oversized_tiles``, with SGLang's ``autotune_cache_kwargs``).
  The host passes one of them: ``BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
  num_stages=4, num_warps=4`` (the fifth, ``ssd_bmm.py:42-46``).
* ``ssd_bmm.py:111-113``: the defaults ``= 16`` of ``BLOCK_SIZE_M``, ``BLOCK_SIZE_N`` and
  ``BLOCK_SIZE_K`` are removed (``BLOCK_SIZE_M: tl.constexpr,`` ...); the host passes them.
* ``ssd_bmm.py:193-276``: the host ``bmm_chunk_fwd``, adapted from ``_bmm_chunk_fwd``: the grid is
  the tuple for the fixed configuration instead of a function of ``META``, and the launch passes
  the configuration as keyword arguments; the checks, the ``.contiguous()`` fallbacks, the
  allocation of ``out``, ``dot_dtype``, the argument order, the device context and the flags
  (``IS_CAUSAL = causal``, ``HAS_SEQ_IDX = seq_idx is not None``) are as in the wrapper.
* ``make_inputs``, ``reference`` and the ``__main__`` test are new: ``make_inputs`` builds
  ``seq_idx`` as SGLang does (``repeat_interleave`` of the request indices) and ``C``, ``B`` either
  contiguous (as SGLang's tests) or as the views of a packed ``xBC`` tensor that the Mamba-2
  mixer passes (``python/sglang/srt/layers/attention/mamba/mamba.py:499-507, 615-617``).
"""

import math

import torch

import triton
import triton.language as tl


# (blank lines: the kernel below keeps its upstream line numbers, ssd_bmm.py:80-190)


















@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    seq_idx_ptr,
    # Matrix dimensions
    seqlen,
    chunk_size,
    K,
    ngroups,
    stride_a_batch,
    stride_a_seqlen,
    stride_a_head,
    stride_ak,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_bk,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_outm,
    stride_outn,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += (
        pid_b * stride_a_batch
        + pid_c * chunk_size * stride_a_seqlen
        + pid_h * stride_a_head
    )
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + pid_h * stride_b_head
    )
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(
            seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit,
            other=-1,
        )
        seq_idx_n = tl.load(
            seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit,
            other=-2,
        )
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += (
        pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),
    )


def bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    # Check constraints.
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty(
        (
            (batch, nchunks, chunk_size, chunk_size)
            if not has_groups
            else (batch, nchunks, ngroups, chunk_size, chunk_size)
        ),
        device=a.device,
        dtype=out_dtype,
    )
    dot_dtype = (
        tl.bfloat16
        if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16
            if a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32
        )
    )
    # the fixed configuration (the fifth of the autotune list, ssd_bmm.py:42-46)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 32
    grid = (
        triton.cdiv(chunk_size, BLOCK_SIZE_M) * triton.cdiv(chunk_size, BLOCK_SIZE_N),
        batch,
        nchunks if not has_groups else nchunks * ngroups,
    )
    with torch.get_device_module(a.device).device(a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a,
            b,
            out,
            seq_idx,
            seqlen,
            chunk_size,
            k,
            ngroups if has_groups else 1,
            a.stride(0),
            a.stride(1),
            0 if not has_groups else a.stride(2),
            a.stride(-1),
            b.stride(0),
            b.stride(1),
            0 if not has_groups else b.stride(2),
            b.stride(-1),
            out.stride(0),
            out.stride(1),
            0 if not has_groups else out.stride(2),
            out.stride(-2),
            out.stride(-1),
            *(
                (seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            causal,
            dot_dtype,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_stages=4,
            num_warps=4,
        )
    return out


def make_inputs(seqlens, ngroups, dstate, gen, conv_dim=None, device="cuda"):
    """``C``, ``B`` (``[1, seqlen, ngroups, dstate]``, ``bfloat16``) and ``seq_idx`` (``[1, seqlen]``,
    ``int32``) for requests of the given lengths, packed one after another.

    ``conv_dim=None``: ``C`` and ``B`` contiguous, as SGLang's tests make them. Otherwise they are
    the views the Mamba-2 mixer passes: one packed ``xBC`` tensor ``[seqlen, conv_dim]``
    (``conv_dim = intermediate + 2 * ngroups * dstate``) split along the last axis into ``x``,
    ``B``, ``C`` and viewed as ``[1, seqlen, ngroups, dstate]`` (strides
    ``(seqlen * conv_dim, conv_dim, dstate, 1)``)."""
    seqlen = sum(seqlens)
    gd = ngroups * dstate
    if conv_dim is None:
        C = torch.randn((1, seqlen, ngroups, dstate), generator=gen).to(torch.bfloat16).to(device)
        B = torch.randn((1, seqlen, ngroups, dstate), generator=gen).to(torch.bfloat16).to(device)
    else:
        xBC = torch.randn((seqlen, conv_dim), generator=gen).to(torch.bfloat16).to(device)
        inter = conv_dim - 2 * gd
        _, B_, C_ = torch.split(xBC, [inter, gd, gd], dim=-1)
        B = B_.view(1, seqlen, ngroups, -1)
        C = C_.view(1, seqlen, ngroups, -1)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(seqlens), dtype=torch.int32), torch.tensor(seqlens), output_size=seqlen
    ).unsqueeze(0).to(device)
    return C, B, seq_idx


def reference(a, b, seq_idx, chunk_size):
    """The per-chunk product of the docstring, in float64, from the inputs alone."""
    _, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)
    pad = nchunks * chunk_size - seqlen
    A = torch.nn.functional.pad(a[0].double(), (0, 0, 0, 0, 0, pad))
    Bm = torch.nn.functional.pad(b[0].double(), (0, 0, 0, 0, 0, pad))
    A = A.view(nchunks, chunk_size, ngroups, k).permute(0, 2, 1, 3)
    Bm = Bm.view(nchunks, chunk_size, ngroups, k).permute(0, 2, 1, 3)
    S = A @ Bm.transpose(-1, -2)
    sid = torch.full((nchunks * chunk_size,), -1, dtype=torch.int64, device=a.device)
    sid[:seqlen] = seq_idx[0].long()
    valid = torch.zeros(nchunks * chunk_size, dtype=torch.bool, device=a.device)
    valid[:seqlen] = True
    sid = sid.view(nchunks, chunk_size)
    valid = valid.view(nchunks, chunk_size)
    same = (sid[:, :, None] == sid[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    return torch.where(same[:, None], S, 0.0).unsqueeze(0)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # (request lengths, ngroups, dstate, chunk_size, conv_dim)
    cases = [
        ([112], 3, 5, 16, None),
        ([112], 32, 128, 16, None),
        ([32, 32, 32, 32, 16, 8, 8], 4, 32, 8, None),
        ([5, 1, 1, 1, 30, 2, 2, 2], 13, 5, 256, None),
        ([138, 225, 138, 225, 42], 13, 32, 128, None),
        ([270, 88, 212, 203], 16, 64, 256, None),
        ([16, 2, 8, 13], 16, 64, 8, None),
        ([3000, 1, 700, 4096, 383], 8, 128, 256, 10240),
        ([8192, 5, 1, 8190], 1, 256, 256, 8704),
    ]
    for sl, G, K, cs, cd in cases:
        C, B, seq_idx = make_inputs(sl, G, K, gen, conv_dim=cd)
        CB = bmm_chunk_fwd(C, B, cs, seq_idx=seq_idx, output_dtype=torch.float32)
        ref = reference(C, B, seq_idx, cs)
        err = (CB.double() - ref).abs().max().item()
        print(f"requests {len(sl)} tokens {sum(sl)} groups {G} dstate {K} chunk {cs}: max |err| {err:.2e}")
        assert err < 1e-3 * max(1.0, ref.abs().max().item())
