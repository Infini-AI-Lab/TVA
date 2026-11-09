"""
Mamba-2 SSD chunk states, bfloat16 activations (SGLang)
=======================================================

The per-chunk state of Mamba-2's chunked scan (SSD). The prefill tokens of all requests are packed
into one sequence of ``seqlen`` tokens (``batch = 1``), cut into physical chunks of ``chunk_size``
tokens that may cross request boundaries; ``seq_idx[0, t]`` is the request of token ``t``. For
chunk ``c``, head ``h`` (in group ``g = h // (nheads // ngroups)``), ``p < headdim`` and
``n < dstate``, with ``last = min((c + 1) * chunk_size, seqlen) - 1`` the chunk's last token and
``i = t - c * chunk_size`` a token's position in its chunk:

    states[0, c, h, p, n] = sum over the tokens t of chunk c with seq_idx[0, t] == seq_idx[0, last] of
        x[0, t, h, p] * B[0, t, g, n] * exp(dA_cumsum[0, h, c, chunk_size - 1] - dA_cumsum[0, h, c, i])
                                      * dt[0, h, c, i]

that is, the contribution of the chunk's own tokens of the request running at its end, each decayed
to the end of the chunk. The kernel scales ``B`` in ``float32``, rounds the scaled ``B`` to the
dtype of ``x`` (``bfloat16``), and accumulates ``x^T @ B_scaled`` in ``float32`` with ``tl.dot``
over blocks of ``BLOCK_SIZE_K`` tokens. Programs: ``(cdiv(headdim, BLOCK_SIZE_M) *
cdiv(dstate, BLOCK_SIZE_N), batch * nchunks, nheads)``.

Layouts and dtypes (as SGLang's Mamba-2 prefill passes them, through ``_mamba_chunk_scan_combined_fwd``):

* ``x``: ``[1, seqlen, nheads, headdim]``, ``bfloat16``, any strides (SGLang passes a view of the
  convolution's output, rows ``conv_dim = nheads * headdim + 2 * ngroups * dstate`` apart);
* ``B``: ``[1, seqlen, ngroups, dstate]``, ``bfloat16``, last stride ``1`` (the caller makes it so),
  another view of the same rows;
* ``dt``: ``[1, nheads, nchunks, chunk_size]``, ``float32``, contiguous (``_chunk_cumsum_fwd``'s
  ``dt_out``: the softplus of ``dt + dt_bias``, ``0`` past ``seqlen``);
* ``dA_cumsum``: ``[1, nheads, nchunks, chunk_size]``, ``float32``, contiguous (the cumulative sum
  of ``A * dt`` within each physical chunk, across request boundaries, constant past ``seqlen``);
* ``seq_idx``: ``[1, seqlen]``, ``int32``, non-decreasing from ``0``;
* ``states`` (output, allocated fresh): ``[1, nchunks, nheads, headdim, dstate]``, ``float32``
  (``states_in_fp32=True``).

Provenance: SGLang v0.5.23 (commit ``297e54fd10``),
``python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_state.py``:

* ``_chunk_state_fwd_kernel``: ``ssd_chunk_state.py:128-363`` (``@triton.autotune`` at 128-215,
  ``@triton.jit`` at 216, ``def`` at 217);
* ``_chunk_state_fwd`` (host wrapper): ``ssd_chunk_state.py:686-755``;
* its caller ``_mamba_chunk_scan_combined_fwd``: ``ssd_combined.py:55-224`` (line 133).

Changes from the source:

* ``ssd_chunk_state.py:128-215``: ``@triton.autotune`` removed (14 configs of ``BLOCK_SIZE_M``,
  ``BLOCK_SIZE_N``, ``BLOCK_SIZE_K``, ``num_stages``, ``num_warps``, ``key=["hdim", "dstate",
  "chunk_size"]``, pruned by ``prune_oversized_tiles`` to blocks no wider than the next power of two
  of the dimension they tile, with SGLang's ``autotune_cache_kwargs``).
* ``ssd_chunk_state.py:258-260``: the defaults of the parameters ``BLOCK_SIZE_M: tl.constexpr = 16``,
  ``BLOCK_SIZE_N: tl.constexpr = 16``, ``BLOCK_SIZE_K: tl.constexpr = 16`` are removed (``BLOCK_SIZE_M:
  tl.constexpr,`` ...); the host passes all three. Every other line of the kernel
  (``ssd_chunk_state.py:216-363``), its ``@triton.jit`` included, is copied verbatim; a kernel line
  here is the upstream line minus 139.
* ``ssd_chunk_state.py:686-755``: the host ``chunk_state_fwd``, adapted from ``_chunk_state_fwd``:
  - it passes one configuration of the autotune list: ``BLOCK_SIZE_M=64``, ``BLOCK_SIZE_N=128``,
    ``BLOCK_SIZE_K=64``, ``num_stages=2``, ``num_warps=4`` (the fifth,
    ``ssd_chunk_state.py:152-156``), so the grid is the tuple
    ``(cdiv(headdim, 64) * cdiv(dstate, 128), batch * nchunks, nheads)`` instead of a function of
    ``META``;
  - the starred argument group of the launch (``ssd_chunk_state.py:748-752``: ``seq_idx``'s strides,
    or zeros) is written out as two arguments, computed before the launch;
  - the asserts, the allocation of ``states``, the argument order (with the ``dt`` and
    ``dA_cumsum`` strides ``(0, 2, 1, 3)`` for batch, chunk, head, chunk position, and
    ``B.stride(-1)``), the device context and the flag ``HAS_SEQ_IDX`` are as in the wrapper.
* ``make_inputs``, ``reference`` and the ``__main__`` test are new (after SGLang's
  ``test/registered/layers/mamba/test_mamba_ssm_ssd.py`` and the layer's call,
  ``python/sglang/srt/layers/attention/mamba/mamba.py:577-644``).
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    # Matrix dimensions
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(
            seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen
        )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(
                seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1
            )
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        else:
            scale = tl.where(
                seq_idx_k == seq_idx_last, tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0
            )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += (
        pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def chunk_state_fwd(
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True
):
    """SGLang's `_chunk_state_fwd` (`ssd_chunk_state.py:686`) with one configuration; returns `states`."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=x.device,
            dtype=states_dtype,
        )
    if seq_idx is not None:
        stride_seq_idx_batch, stride_seq_idx_seqlen = seq_idx.stride(0), seq_idx.stride(1)
    else:
        stride_seq_idx_batch, stride_seq_idx_seqlen = 0, 0
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 64
    grid = (
        triton.cdiv(headdim, BLOCK_SIZE_M) * triton.cdiv(dstate, BLOCK_SIZE_N),
        batch * nchunks,
        nheads,
    )
    with torch.get_device_module(x.device).device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x,
            B,
            states,
            dt,
            dA_cumsum,
            seq_idx,
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(-1),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            stride_seq_idx_batch,
            stride_seq_idx_seqlen,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_stages=2,
            num_warps=4,
        )
    return states


def make_inputs(seqlens, nheads, headdim, dstate, ngroups, chunk_size, device, generator, rows=True):
    """A packed prefill batch as SGLang's Mamba-2 layer passes it (mamba.py:577-644,
    ssd_combined.py:127-133): x and B as views of one bfloat16 row per token (the convolution's
    output, `rows=True`) or as contiguous tensors (SGLang's tests, `rows=False`); dt the softplus of
    a random pre-activation plus a bias (0 past seqlen) and dA_cumsum the per-chunk cumulative sum of
    A * dt (A < 0), both [1, nheads, nchunks, chunk_size] float32, as `_chunk_cumsum_fwd` lays
    them out; seq_idx the request of each token. Built on the CPU with `generator`, then moved."""
    seqlen = sum(seqlens)
    nchunks = triton.cdiv(seqlen, chunk_size)
    if rows:
        conv_dim = nheads * headdim + 2 * ngroups * dstate
        hbc = torch.randn((seqlen, conv_dim), generator=generator).to(torch.bfloat16).to(device)
        x = hbc[:, : nheads * headdim].view(1, seqlen, nheads, headdim)
        B = hbc[:, nheads * headdim : nheads * headdim + ngroups * dstate].view(1, seqlen, ngroups, dstate)
    else:
        x = torch.randn((1, seqlen, nheads, headdim), generator=generator).to(torch.bfloat16).to(device)
        B = torch.randn((1, seqlen, ngroups, dstate), generator=generator).to(torch.bfloat16).to(device)
    A = -torch.exp(torch.rand(nheads, generator=generator))
    dt_bias = torch.rand(nheads, generator=generator) - 4.0
    dt = torch.nn.functional.softplus(torch.randn((nheads, nchunks * chunk_size), generator=generator)
                                      + dt_bias[:, None])
    dt[:, seqlen:] = 0.0
    dt = dt.reshape(1, nheads, nchunks, chunk_size)
    dA_cumsum = (A[None, :, None, None] * dt).cumsum(-1)
    seq_idx = torch.repeat_interleave(torch.arange(len(seqlens), dtype=torch.int32),
                                      torch.tensor(seqlens)).unsqueeze(0)
    return x, B, dt.to(device), dA_cumsum.to(device), seq_idx.to(device)


def reference(x, B, dt, dA_cumsum, seqlens, chunk_size):
    """The chunk states in float64 from the inputs, with the request boundaries from `seqlens`:
    the chunk's tokens of the request running at its last token, each decayed to the chunk's end."""
    _, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2], B.shape[3]
    nchunks = dt.shape[2]
    starts = torch.tensor([0] + list(seqlens)).cumsum(0).tolist()
    xs = x[0].double()
    Bs = B[0].double().repeat_interleave(nheads // ngroups, dim=1)       # [seqlen, nheads, dstate]
    out = torch.empty((nchunks, nheads, headdim, dstate), dtype=torch.float64, device=x.device)
    for c in range(nchunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, seqlen)
        r = max(j for j in range(len(seqlens)) if starts[j] <= hi - 1)   # the request at the end
        t0 = max(lo, starts[r])
        i = torch.arange(t0 - lo, hi - lo, device=x.device)
        dA = dA_cumsum[0, :, c].double()
        w = torch.exp(dA[:, -1:] - dA[:, i]) * dt[0, :, c].double()[:, i]  # [nheads, tokens]
        out[c] = torch.einsum("thp,thn,ht->hpn", xs[t0:hi], Bs[t0:hi], w)
    return out.unsqueeze(0)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (request lengths, nheads, headdim, dstate, ngroups, chunk_size, rows): SGLang's tests
    # (test/registered/layers/mamba/test_mamba_ssm_ssd.py: one request of 112 or 128 tokens with
    # chunks of 16 or 32, nheads 3 to 32, headdim = dstate 5 to 128, ngroups = nheads; packed
    # batches with chunks of 8, 128 and 256), and the Mamba-2 layer's shapes per tensor-parallel
    # rank (headdim 64 or 80, dstate 128 or 256, ngroups 1 or 8, chunk_size 256).
    cases = [
        ([112], 3, 5, 5, 3, 16, False),
        ([128], 32, 128, 128, 32, 32, False),
        ([16, 20], 16, 64, 64, 16, 8, False),
        ([270, 88, 212, 203], 16, 64, 64, 16, 256, False),
        ([138, 225], 13, 32, 32, 13, 128, False),
        ([5, 30], 4, 21, 21, 4, 256, False),
        ([1000, 1, 2049, 64, 64, 300], 80, 64, 128, 1, 256, True),
        ([8192], 128, 64, 128, 8, 256, True),
        ([3] * 40 + [500], 128, 80, 128, 8, 256, True),
        ([300, 1700], 64, 64, 256, 1, 256, True),
    ]
    for seqlens, nheads, headdim, dstate, ngroups, chunk_size, rows in cases:
        x, B, dt, dA_cumsum, seq_idx = make_inputs(seqlens, nheads, headdim, dstate, ngroups, chunk_size,
                                                   DEVICE, gen, rows)
        states = chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
        ref = reference(x, B, dt, dA_cumsum, seqlens, chunk_size)
        err = ((states.double() - ref).abs().max() / ref.abs().max()).item()
        print(f"requests {len(seqlens)} tokens {sum(seqlens)} chunks {states.shape[1]} heads {nheads} "
              f"headdim {headdim} dstate {dstate} groups {ngroups} chunk {chunk_size}: "
              f"max error / max |state| {err:.1e}")
        assert err < 1e-2
    print("✅ SSD chunk states match the float64 reference")
