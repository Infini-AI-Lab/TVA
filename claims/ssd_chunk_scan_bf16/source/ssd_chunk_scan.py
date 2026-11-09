"""
Mamba-2 SSD chunk scan, continuous batching (SGLang), bfloat16
===============================================================

The output of Mamba-2's chunked scan (SSD). The prefill tokens of all requests are packed into one
sequence of ``seqlen`` tokens (``batch = 1``), cut into physical chunks of ``chunk_size`` tokens
that may cross request boundaries. A *logical chunk* is the part of a physical chunk that belongs
to one request; ``chunk_indices[k]`` and ``chunk_offsets[k]`` give logical chunk ``k``'s physical
chunk and its start offset in it. For a token ``t`` of request ``r`` in physical chunk
``c = t // chunk_size`` (position ``i = t - c * chunk_size``), head ``h`` (group
``g = h // (nheads // ngroups)``) and channel ``p``, with ``l`` the first token of ``t``'s logical
chunk (``max(c * chunk_size, query_start_loc[r])``, at position ``o = l - c * chunk_size``):

    S   = initial_states[r, h]        if r >= 1 and request r starts in chunk c (0 without them)
          states[0, c, h]             otherwise (the state entering chunk c; for c = 0 the
                                       first request's initial state, as state passing stores it)
    dB  = dA_cumsum[0, h, c, o - 1]   if o > 0, else 0
    out[0, t, h, p] = exp(dA_cumsum[0, h, c, i] - dB) * sum_n C[0, t, g, n] * S[p, n]
                    + sum_{s = l .. t} cb[0, c, g, i, s - c*chunk_size]
                          * exp(dA_cumsum[0, h, c, i] - dA_cumsum[0, h, c, s - c*chunk_size])
                          * dt[0, h, c, s - c*chunk_size] * x[0, s, h, p]
                    + D[h] * x[0, t, h, p]

Programs: ``(cdiv(chunk_size, BLOCK_SIZE_M) * cdiv(headdim, BLOCK_SIZE_N), N, nheads)``, ``N`` the
number of logical chunks (without ``initial_states``, ``nchunks``: ``cb`` is zero across requests
and ``seq_idx`` selects the rows that continue a request); a program computes one tile.

Layouts and dtypes (as SGLang's Mamba-2 prefill passes them, ``mamba.py:606-642`` through
``mamba_chunk_scan_combined`` and ``_mamba_chunk_scan_combined_fwd``):

* ``cb``: ``[1, nchunks, ngroups, chunk_size, chunk_size]``, ``float32`` (``C_i . B_j`` per chunk,
  ``_bmm_chunk_fwd(..., output_dtype=torch.float32)``);
* ``x``: ``[1, seqlen, nheads, headdim]``, ``bfloat16``, a slice of the convolution's output (row
  stride ``conv_dim``, last dimension contiguous);
* ``dt``: ``[1, nheads, nchunks, chunk_size]``, ``float32`` (``_chunk_cumsum_fwd``'s ``dt``,
  after ``softplus(dt + dt_bias)``, ``0`` past ``seqlen``);
* ``dA_cumsum``: ``[1, nheads, nchunks, chunk_size]``, ``float32`` (cumulative ``A * dt`` within
  each physical chunk);
* ``C``: ``[1, seqlen, ngroups, dstate]``, ``bfloat16``, a slice of the convolution's output;
* ``states``: ``[1, nchunks, nheads, headdim, dstate]``, ``float32`` (state passing's output,
  ``state_dtype = ssm_state.dtype``);
* ``D``: ``[nheads]``, ``bfloat16`` (the layer's parameter, in the model's dtype);
* ``seq_idx``: ``[1, seqlen]``, ``int32``; ``chunk_indices``, ``chunk_offsets``: ``[N]``, ``int32``;
* ``initial_states``: ``[num_requests, nheads, headdim, dstate]``, ``float32`` (the cached state, or
  zero for a request without a cached prefix);
* ``out`` (output, given by the caller): ``[1, seqlen, nheads, headdim]``, ``bfloat16``.

Provenance: SGLang v0.5.23 (commit ``297e54fd10``),
``python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_scan.py``:

* ``_chunk_scan_fwd_kernel``: ``ssd_chunk_scan.py:21-557`` (``@triton.autotune`` at 21-160,
  ``@triton.jit`` at 161, ``def`` at 162);
* ``_chunk_scan_fwd`` (host wrapper): ``ssd_chunk_scan.py:560-700``;
* ``_query_start_loc_to_chunk_indices_offsets``:
  ``python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:111-192``.

Changes from the source:

* ``ssd_chunk_scan.py:21-160``: ``@triton.autotune`` removed (24 configs, ``key=["chunk_size",
  "hdim", "dstate", "IS_CAUSAL"]``, pruned by ``prune_oversized_tiles`` to blocks no wider than
  ``next_power_of_2`` of ``chunk_size`` (``BLOCK_SIZE_M``, ``BLOCK_SIZE_K``) and ``hdim``
  (``BLOCK_SIZE_N``), with SGLang's ``autotune_cache_kwargs``).
* ``ssd_chunk_scan.py:237-239``: the defaults of ``BLOCK_SIZE_M``, ``BLOCK_SIZE_N``,
  ``BLOCK_SIZE_K`` (``tl.constexpr = 16``) are removed; the host passes them. Every other line of
  the kernel (``ssd_chunk_scan.py:161-557``), its ``@triton.jit`` included, is copied verbatim.
* ``ssd_chunk_scan.py:14, 16, 18``: the module's imports of ``packaging.version`` and of the
  autotune helpers, and ``TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")``,
  are removed; the host passes ``IS_TRITON_22=True`` (the value of ``TRITON_22`` under
  Triton 3.8.0).
* ``ssd_chunk_scan.py:560-700``: the host ``chunk_scan_fwd``, adapted from ``_chunk_scan_fwd``:
  - it passes one configuration of the autotune list (``ssd_chunk_scan.py:61-65``):
    ``BLOCK_SIZE_M=64``, ``BLOCK_SIZE_N=64``, ``BLOCK_SIZE_K=32``, ``num_warps=4``,
    ``num_stages=1``, so the grid is a tuple instead of a function of ``META``;
  - the starred stride groups of the launch (``ssd_chunk_scan.py:670, 680-689``: ``seq_idx``'s and
    the initial states' strides, or zeros) are written out as one argument each;
  - the asserts, the argument order (with the ``dt`` and ``dA_cumsum`` strides ``(0, 2, 1, 3)`` for
    batch, chunk, head, chunk position), the flags and ``out_x`` are as in the wrapper.
* ``mamba2_metadata.py:111-192``: ``query_start_loc_to_chunk_indices_offsets``, copied (a
  torch-only helper; SGLang's static method, with its docstring shortened) to build
  ``chunk_indices`` and ``chunk_offsets``; ``make_inputs``, ``reference`` and the ``__main__``
  test are new.
"""

import math

import torch

import triton
import triton.language as tl


@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    out_x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    chunk_indices_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_z_batch,
    stride_z_seqlen,
    stride_z_head,
    stride_z_hdim,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
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
    stride_C_batch,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    stride_D_head,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    if not HAS_INITSTATES:
        c_idx = pid_c
        c_off = 0
    else:
        c_idx = tl.load(chunk_indices_ptr + pid_c, mask=pid_c > -1, other=0)
        c_off = tl.load(chunk_offsets_ptr + pid_c, mask=pid_c > -1, other=0)

    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += (
        pid_b * stride_cb_batch
        + c_idx * stride_cb_chunk
        + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    )
    x_ptr += (
        pid_b * stride_x_batch
        + c_idx * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + c_idx * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + c_idx * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    C_ptr += (
        pid_b * stride_C_batch
        + c_idx * chunk_size * stride_C_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    offs_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    prev_states_ptr = (
        states_ptr
        + pid_b * stride_states_batch
        + c_idx * stride_states_chunk
        + pid_h * stride_states_head
    )
    prev_states_hdim = stride_states_hdim
    prev_states_dstate = stride_states_dstate

    chunk_size_limit = min(chunk_size, seqlen - c_idx * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + c_idx * chunk_size * stride_seq_idx_seqlen
        )

        # - we only need seq_idx_prev to be aligned to chunk boundary
        seq_idx_prev = tl.load(
            seq_idx_ptr - stride_seq_idx_seqlen, mask=c_idx >= 1, other=0
        )

        if HAS_INITSTATES:
            # if there are init states, we only need seq_idx_m to point
            # what is the current seq_idx

            # get current seq idx
            if (pid_m * BLOCK_SIZE_M + c_off) < chunk_size_limit:
                seq_idx_m = tl.load(
                    seq_idx_ptr
                    + (pid_m * BLOCK_SIZE_M + c_off) * stride_seq_idx_seqlen,
                )

                # - recall that in ssd_state_passing, for the case c_off == 0
                # i.e., the very first sequence, we made states_ptr hold its initial state
                # so this edge case is taken care of
                if (
                    (c_off == 0)
                    and (
                        seq_idx_prev != seq_idx_m
                    )  # if a seq is changed exactly on boundary
                    or (c_off > 0)  # implies a new example (pseudo chunk)
                ):
                    # - replace prev_states_ptr with init_states
                    prev_states_ptr = (
                        initstates_ptr
                        + seq_idx_m * stride_init_states_batch
                        + pid_h * stride_init_states_head
                    )
                    prev_states_hdim = stride_init_states_hdim  # override strides
                    prev_states_dstate = stride_init_states_dstate

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    # - handle chunk state limit
    if HAS_INITSTATES:
        # have to split this if otherwise compilation will have problems
        dA_cs_m_boundary = 0.0

        # get the c_idx for the next (logica) chunk
        c_idx_n = tl.load(
            chunk_indices_ptr + (pid_c + 1),
            mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,
            other=-1,  # to trigger different chunk
        )

        # - there are things to consider
        # A. if c_off > 0 then we need to move the dA_cs boundary to ensure correct
        #    contribution of past states
        # B. if c_off_n < chunk_size_limit, then we need to adjust this so as not to
        #    encroach into the next sequence, where c_off_n is the offset of the next
        #    (logical) chunk.
        # An equivalent check for B is c_idx == c_idx_n, where there is repetition in
        # (logical) chunk indices.

        if (c_idx == c_idx_n) or c_off > 0:
            # get the next offset
            c_off_n = tl.load(
                chunk_offsets_ptr + (pid_c + 1),
                mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,
                other=chunk_size,
            )

            # in this case, adjust down the chunk_size_limit
            if c_idx == c_idx_n:
                chunk_size_limit = min(c_off_n, chunk_size_limit)

            # get the cs at the offset boundary
            # - c_off == 0 is a passthrough
            # - We need dA_cs at the boundary, defined by c_off - no need
            #   to increase pointer by pid_m (it is a constant offset,
            #   i.e. the same for all blocks)
            dA_cs_m_boundary = tl.load(
                dA_cumsum_ptr + (c_off - 1) * stride_dA_cs_csize,
                mask=(((c_off - 1) > -1) and ((c_off) < chunk_size)),
                other=0.0,
            ).to(tl.float32)

    if HAS_SEQ_IDX:
        # - handle seq idx when HAS_INITSTATES==False
        if not HAS_INITSTATES:
            seq_idx_m = tl.load(
                seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
                mask=offs_m < chunk_size_limit,
                other=-1,
            )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or c_idx > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(
            0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
        )
        C_ptrs = C_ptr + (
            offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate
        )

        prev_states_ptrs = prev_states_ptr + (
            offs_n[None, :] * prev_states_hdim
            + offs_k_dstate[:, None] * prev_states_dstate
        )
        if HAS_SEQ_IDX:
            if not HAS_INITSTATES:
                # - this is for continuous batching where there is no init states
                scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
            else:
                # - if there is initstates, we will rely on prev_states, no zeroing
                #   required.
                scale_m = tl.exp(dA_cs_m - dA_cs_m_boundary)
        else:
            scale_m = tl.exp(dA_cs_m)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k_dstate[None, :] < dstate),
                other=0.0,
            )

            prev_states = tl.load(
                prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(
                    C_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit)
                    & (offs_k_dstate[None, :] < dstate - k),
                    other=0.0,
                )
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(
                    prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k)
                    & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K) + c_off
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    x_ptrs = x_ptr + (
        offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = (
        chunk_size_limit
        if not IS_CAUSAL
        else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    )
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
            tl.float32
        )
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr
            + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        out_x_ptr += (
            pid_b * stride_out_batch
            + c_idx * chunk_size * stride_out_seqlen
            + pid_h * stride_out_head
        )
        out_x_ptrs = out_x_ptr + (
            stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :]
        )
        tl.store(
            out_x_ptrs,
            acc,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
        )

        z_ptr += (
            pid_b * stride_z_batch
            + c_idx * chunk_size * stride_z_seqlen
            + pid_h * stride_z_head
        )
        z_ptrs = z_ptr + (
            stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :]
        )
        z = tl.load(
            z_ptrs,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += (
        pid_b * stride_out_batch
        + c_idx * chunk_size * stride_out_seqlen
        + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
    )


def chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    D=None,
    z=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    initial_states=None,
    out=None,
):
    """SGLang's `_chunk_scan_fwd` (`ssd_chunk_scan.py:560`); writes `out`, returns `out_x`."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)

    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

        if initial_states is not None:
            # with initial states, we need to take care of how
            # seq_idx crosses the boundaries
            assert batch == 1, "chunk scan only supports initial states with batch 1"
            assert chunk_indices is not None and chunk_offsets is not None, (
                "chunk_indices and chunk_offsets should have been set"
            )
        else:
            chunk_indices, chunk_offsets = None, None
    else:
        chunk_indices, chunk_offsets = None, None

    assert out.shape == x.shape

    if z is not None:
        out_x = torch.empty_like(x)
        assert out_x.stride() == out.stride()
    else:
        out_x = None

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    grid = (
        triton.cdiv(chunk_size, BLOCK_SIZE_M) * triton.cdiv(headdim, BLOCK_SIZE_N),
        batch * nchunks if chunk_offsets is None else len(chunk_offsets),
        nheads,
    )
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )
    if seq_idx is not None:
        stride_seq_idx_batch, stride_seq_idx_seqlen = seq_idx.stride(0), seq_idx.stride(1)
    else:
        stride_seq_idx_batch, stride_seq_idx_seqlen = 0, 0
    if initial_states is not None:
        stride_init_states_batch = initial_states.stride(0)
        stride_init_states_head = initial_states.stride(1)
        stride_init_states_hdim = initial_states.stride(2)
        stride_init_states_dstate = initial_states.stride(3)
    else:
        stride_init_states_batch, stride_init_states_head = 0, 0
        stride_init_states_hdim, stride_init_states_dstate = 0, 0
    _chunk_scan_fwd_kernel[grid](
        cb,
        x,
        z,
        out,
        out_x,
        dt,
        dA_cumsum,
        seq_idx,
        C,
        states,
        D,
        initial_states,
        chunk_indices,
        chunk_offsets,
        len(chunk_indices) if chunk_indices is not None else 0,
        chunk_size,
        headdim,
        dstate,
        batch,
        seqlen,
        nheads // ngroups,
        cb.stride(0),
        cb.stride(1),
        cb.stride(2),
        cb.stride(3),
        cb.stride(4),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        z_strides[3],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
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
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        stride_init_states_batch,
        stride_init_states_head,
        stride_init_states_hdim,
        stride_init_states_dstate,
        D.stride(0) if D is not None else 0,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=True,
        HAS_INITSTATES=initial_states is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
        num_stages=1,
    )
    return out_x


def query_start_loc_to_chunk_indices_offsets(query_start_loc, chunk_size, total_seqlens):
    """SGLang's `Mamba2Metadata._query_start_loc_to_chunk_indices_offsets`
    (`mamba2_metadata.py:111-192`): for each logical chunk (the part of a physical chunk that
    belongs to one request), its physical chunk and its start offset in it."""

    cu_seqlens = query_start_loc[1:]  # remove prepended 0

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = (
        math.ceil(total_seqlens / chunk_size)
        + (cu_seqlens[:-1] % chunk_size > 0).sum()
    )
    chunk_indices = torch.arange(N, dtype=torch.int, device=query_start_loc.device)
    chunk_offsets = torch.zeros(
        (N,), dtype=torch.int, device=query_start_loc.device
    )

    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        # if does not divide chunk_size, then there is one chunk insertion
        p += s % chunk_size > 0

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)

        # adjust indices and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return chunk_indices, chunk_offsets


def make_inputs(seqlens, nheads, headdim, dstate, ngroups, chunk_size, generator, device):
    """A packed prefill batch as SGLang's Mamba-2 layer builds it (`mamba.py:571-642`,
    `mamba2_metadata.py:247-286`, `ssd_combined.py:127-191`): `x`, `B`, `C` slices of one
    convolution output `[seqlen, conv_dim]` in `bfloat16`; `dt = softplus(.)` and the per-chunk
    cumulative sums of `A * dt` (`A < 0`) in `float32`, `0` past `seqlen`; `cb = C . B^T` per chunk
    and group in `float32`, zero across requests (`_bmm_chunk_fwd` with `seq_idx`) and NaN past
    `seqlen` (never written there); random `float32` chunk states; initial states (zero for
    requests without a cached prefix); `D` in `bfloat16`; the metadata."""
    seqlen = sum(seqlens)
    nchunks = triton.cdiv(seqlen, chunk_size)
    conv_dim = nheads * headdim + 2 * ngroups * dstate
    conv = torch.randn((seqlen, conv_dim), generator=generator).to(torch.bfloat16).to(device)
    x_flat, B_flat, C_flat = torch.split(
        conv, [nheads * headdim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = x_flat.view(1, seqlen, nheads, headdim)
    B = B_flat.view(1, seqlen, ngroups, dstate)
    C = C_flat.view(1, seqlen, ngroups, dstate)
    query_start_loc = torch.tensor([0] + list(seqlens)).cumsum(0)
    seq_idx = torch.repeat_interleave(torch.arange(len(seqlens), dtype=torch.int32),
                                      torch.tensor(seqlens)).unsqueeze(0)
    chunk_indices, chunk_offsets = query_start_loc_to_chunk_indices_offsets(
        query_start_loc, chunk_size, seqlen)
    A = -torch.exp(torch.rand(nheads, generator=generator) * 2.0)
    dt_tok = torch.nn.functional.softplus(
        torch.randn((nheads, nchunks * chunk_size), generator=generator) - 3.0)
    dt_tok[:, seqlen:] = 0.0
    dt = dt_tok.reshape(1, nheads, nchunks, chunk_size)
    dA_cumsum = (A[:, None] * dt_tok).reshape(1, nheads, nchunks, chunk_size).cumsum(-1)
    pad = nchunks * chunk_size - seqlen
    Bp = torch.nn.functional.pad(B[0].float().cpu(), (0, 0, 0, 0, 0, pad))
    Cp = torch.nn.functional.pad(C[0].float().cpu(), (0, 0, 0, 0, 0, pad))
    Bc = Bp.reshape(nchunks, chunk_size, ngroups, dstate)
    Cc = Cp.reshape(nchunks, chunk_size, ngroups, dstate)
    cb = torch.einsum("cign,cjgn->cgij", Cc.double(), Bc.double()).float()
    sid = torch.nn.functional.pad(seq_idx[0], (0, pad), value=-1).reshape(nchunks, chunk_size)
    same = (sid[:, :, None] == sid[:, None, :]) & (sid[:, :, None] >= 0)
    cb = torch.where(same[:, None], cb, 0.0)
    valid = (sid >= 0)
    cb = torch.where(valid[:, None, :, None] & valid[:, None, None, :], cb, float("nan"))
    cb = cb.contiguous().unsqueeze(0)
    states = torch.randn((1, nchunks, nheads, headdim, dstate), generator=generator)
    has_prefix = torch.rand(len(seqlens), generator=generator) < 0.7
    initial_states = (torch.randn((len(seqlens), nheads, headdim, dstate), generator=generator)
                      * has_prefix[:, None, None, None])
    D = (torch.rand(nheads, generator=generator) + 0.5).to(torch.bfloat16)
    return dict(
        cb=cb.to(device), x=x, dt=dt.to(device), dA_cumsum=dA_cumsum.to(device), C=C,
        states=states.to(device), D=D.to(device), seq_idx=seq_idx.to(device),
        chunk_indices=chunk_indices.to(torch.int32).to(device),
        chunk_offsets=chunk_offsets.to(torch.int32).to(device),
        initial_states=initial_states.to(device), query_start_loc=query_start_loc)


def reference(inp, chunk_size, nheads_per_group, has_init=True):
    """The output in float64 from the host's inputs, with the request boundaries taken from
    `query_start_loc` (not from the metadata); without initial states (`has_init=False`) a
    request that starts in a chunk starts there from the zero state."""
    x = inp["x"][0].double()
    seqlen, nheads, headdim = x.shape
    C = inp["C"][0].double()
    cb = inp["cb"][0].double()
    dt = inp["dt"][0].double()
    dA = inp["dA_cumsum"][0].double()
    states = inp["states"][0].double()
    init = inp["initial_states"].double()
    D = inp["D"].double()
    qsl = inp["query_start_loc"].tolist()
    out = torch.empty((seqlen, nheads, headdim), dtype=torch.float64, device=x.device)
    for r in range(len(qsl) - 1):
        start, end = qsl[r], qsl[r + 1]
        c0, c1 = start // chunk_size, (end - 1) // chunk_size
        for c in range(c0, c1 + 1):
            lo, hi = max(start, c * chunk_size), min(end, (c + 1) * chunk_size)
            o, i0, i1 = lo - c * chunk_size, lo - c * chunk_size, hi - c * chunk_size
            if r >= 1 and start >= c * chunk_size:
                S = init[r] if has_init else torch.zeros_like(states[c])    # [h, p, n]
            else:
                S = states[c]
            dB = dA[:, c, o - 1] if o > 0 else torch.zeros_like(dA[:, c, 0])
            dAi = dA[:, c, i0:i1]                                              # [h, L]
            g = torch.arange(nheads, device=x.device) // nheads_per_group
            Ct = C[lo:hi][:, g]                                                # [L, h, n]
            y = torch.exp(dAi - dB[:, None]).T[:, :, None] * torch.einsum("lhn,hpn->lhp", Ct, S)
            decay = torch.exp(dAi[:, :, None] - dAi[:, None, :])               # [h, L, L]
            causal = torch.ones(i1 - i0, i1 - i0, dtype=torch.bool, device=x.device).tril()
            w = cb[c][g][:, i0:i1, i0:i1] * torch.where(causal, decay, 0.0) * dt[:, c, i0:i1][:, None, :]
            y = y + torch.einsum("hls,shp->lhp", w, x[lo:hi])
            out[lo:hi] = y + D[None, :, None] * x[lo:hi]
    return out.unsqueeze(0)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (request lengths, nheads, headdim, dstate, ngroups, chunk_size): SGLang's tests
    # (test/registered/layers/mamba/test_mamba_ssm_ssd.py: chunk sizes 8, 128 and 256, the
    # continuous-batching and prefill-chunking lengths, headdim = dstate 5 to 32 with ngroups =
    # nheads) and the Mamba-2 hybrids' layers per tensor-parallel rank (headdim 64, dstate 128:
    # Mamba-2, Nemotron-H, Granite 4.0-H; headdim 80: Nemotron Nano 2; ngroups 1 or 8).
    cases = [
        ([1], 8, 64, 128, 8, 256),
        ([256], 8, 64, 128, 8, 256),
        ([16, 2, 8, 13], 16, 64, 64, 16, 8),
        ([270, 88, 212, 203], 16, 64, 64, 16, 256),
        ([16, 20], 4, 32, 32, 4, 8),
        ([5, 1, 1, 1], 13, 5, 5, 13, 256),
        ([138, 225, 138, 225], 8, 21, 21, 8, 128),
        ([64, 32, 16, 8, 8], 4, 16, 16, 4, 8),
        ([1000, 1, 2049, 64, 64, 300], 32, 80, 128, 8, 128),
        ([4096, 4096], 64, 64, 128, 8, 256),
        ([3] * 40 + [500], 16, 64, 128, 1, 256),
    ]
    for seqlens, nheads, headdim, dstate, ngroups, chunk_size in cases:
        inp = make_inputs(seqlens, nheads, headdim, dstate, ngroups, chunk_size, gen, DEVICE)
        # with initial states (a batch with a cached prefix) and without (SGLang passes None)
        for has_init in (True, False):
            out = torch.empty_like(inp["x"])
            chunk_scan_fwd(inp["cb"], inp["x"], inp["dt"], inp["dA_cumsum"], inp["C"],
                           inp["states"], D=inp["D"], z=None, seq_idx=inp["seq_idx"],
                           chunk_indices=inp["chunk_indices"], chunk_offsets=inp["chunk_offsets"],
                           initial_states=inp["initial_states"] if has_init else None, out=out)
            ref = reference(inp, chunk_size, nheads // ngroups, has_init)
            err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
            print(f"requests {len(seqlens)} tokens {sum(seqlens)} heads {nheads} headdim {headdim} "
                  f"dstate {dstate} groups {ngroups} chunk {chunk_size} initial states {has_init}: "
                  f"max error / max |out| {err:.1e}")
            assert err < 2e-2
    print("✅ SSD chunk scan matches the float64 reference")
