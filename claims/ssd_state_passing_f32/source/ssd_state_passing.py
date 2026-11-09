"""
Mamba-2 SSD state passing across chunks, continuous batching (SGLang)
=====================================================================

The inter-chunk recurrence of Mamba-2's chunked scan (SSD). The prefill tokens of all requests are
packed into one sequence of ``seqlen`` tokens (``batch = 1``), cut into physical chunks of
``chunk_size`` tokens that may cross request boundaries; ``seq_idx[0, t]`` is the request of token
``t``. For each head ``h`` and state element ``m`` (``dim = headdim * dstate`` per head), with
``e_c = seq_idx[0, min((c + 1) * chunk_size, seqlen) - 1]`` the request at the end of chunk ``c``:

    s = initial_states[0, h, m]  (0 without initial states);  out[0, 0, h, m] = s
    for c in 0 .. nchunks - 1:
        if e_c differs from e_{c-1} (e_{-1} = 0):    # a request starts inside chunk c
            s = initial_states[e_c, h, m]  (without initial states: s = 0, through scale = 0)
            dA = dA_cumsum[0, h, c, chunk_size - 1] - dA_cumsum[0, h, c, off - 1]   (0 if off = 0)
        else:
            dA = dA_cumsum[0, h, c, chunk_size - 1]
        s = exp(dA) * s + states[0, c, h, m]
        out[0, c + 1, h, m] = s   if c < nchunks - 1,   final_states[0, h, m] = s   otherwise

where ``off`` is the offset in chunk ``c`` at which request ``e_c`` starts, read from
``chunk_offsets`` at the logical chunk the kernel counts (a logical chunk is the part of a physical
chunk that belongs to one request). So ``out[0, c]`` is the state entering chunk ``c`` of the request
running there. Programs: ``(cdiv(dim, BLOCK_SIZE), batch, nheads)``.

Layouts and dtypes (as SGLang's Mamba-2 prefill passes them):

* ``states``: ``[1, nchunks, nheads, dim]``, ``float32`` (the chunk states, computed in ``float32``);
* ``dA_cumsum``: ``[1, nheads, nchunks, chunk_size]``, ``float32`` (cumulative sum within each
  physical chunk, across request boundaries, constant past ``seqlen``);
* ``initial_states``: ``[num_requests, nheads, dim]``, ``float32`` (the state cache's dtype), or ``None``;
* ``seq_idx``: ``[1, seqlen]``, ``int32``, non-decreasing from ``0``;
* ``chunk_offsets``: ``[N]``, ``int32``, each logical chunk's offset in its chunk, or ``None``;
* ``out`` (output, allocated fresh): ``[1, nchunks, nheads, dim]``, ``out_dtype`` (``float32``);
* ``final_states`` (output, allocated fresh): ``[1, nheads, dim]``, ``float32``.

Provenance: SGLang v0.5.23 (commit ``297e54fd10``),
``python/sglang/kernels/ops/mamba/triton_ops/ssd_state_passing.py``:

* ``_state_passing_fwd_kernel``: ``ssd_state_passing.py:18-185`` (``@triton.autotune`` at 18-32,
  ``@triton.jit`` at 33, ``def`` at 34);
* ``_state_passing_fwd`` (host wrapper): ``ssd_state_passing.py:188-281``;
* ``_query_start_loc_to_chunk_indices_offsets``:
  ``python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:111-192``.

Changes from the source:

* ``ssd_state_passing.py:18-32``: ``@triton.autotune`` removed (configs ``BLOCK_SIZE`` 64, 128, 256,
  512, 1024, 2048, ``key=["dim"]``, pruned by ``prune_oversized_tiles`` to blocks no wider than
  ``next_power_of_2(dim)``, with SGLang's ``autotune_cache_kwargs``).
* ``ssd_state_passing.py:74``: the default of the parameter ``BLOCK_SIZE: tl.constexpr = 16`` is
  removed (``BLOCK_SIZE: tl.constexpr,``); the host passes ``BLOCK_SIZE``. Every other line of the
  kernel (``ssd_state_passing.py:33-185``), its ``@triton.jit`` included, is copied verbatim.
* ``ssd_state_passing.py:188-281``: the host ``state_passing_fwd``, adapted from
  ``_state_passing_fwd``:
  - it passes one configuration of the autotune list: ``BLOCK_SIZE=1024`` (the fifth),
    ``num_warps=4``, ``num_stages=3`` (``triton.Config``'s defaults), so the grid is the tuple
    ``(triton.cdiv(dim, 1024), batch, nheads)`` instead of a function of ``META``;
  - the starred argument groups of the launch (``ssd_state_passing.py:263-276``: the initial
    states' and ``seq_idx``'s strides, or zeros) are written out as one argument each, computed
    before the launch;
  - the asserts, the allocations of ``out`` and ``final_states``, the argument order (with the
    ``dA_cumsum`` strides ``(0, 2, 1, 3)`` for batch, chunk, head, chunk position), the device
    context and the flags ``HAS_INITSTATES``, ``HAS_SEQ_IDX``, ``IS_CONT_BATCHED`` are as in the
    wrapper.
* ``mamba2_metadata.py:111-192``: ``query_start_loc_to_chunk_indices_offsets``, copied (a torch-only
  helper; SGLang's static method, with its docstring shortened) to build ``chunk_offsets`` in the
  test; ``make_inputs``, ``reference`` and the ``__main__`` test are new.
"""

import math

import torch

import triton
import triton.language as tl


@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    final_states_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    # Matrix dimensions
    dim,
    nchunks,
    seqlen,
    chunk_size,
    # Strides
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_final_states_batch,
    stride_final_states_head,
    stride_final_states_dim,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_initstates_batch,
    stride_initstates_head,
    stride_initstates_dim,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_CONT_BATCHED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_h * stride_dA_cs_head
        + (chunk_size - 1) * stride_dA_cs_csize
    )
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += (
        pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    )
    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_initstates_head
        if not IS_CONT_BATCHED:
            initstates_ptr += pid_b * stride_initstates_batch

    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    # - states will be the past state of the sequence that continues on the current check
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        initstates_ptr += offs_m * stride_initstates_dim
        initstates_ptrs = initstates_ptr
        # - for cont batches, for the first chunk mean it will be the first batch's
        #   init state
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    prev_seq_idx_chunk_end = 0
    logical_chunk_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale_mask = True
        if HAS_SEQ_IDX:
            # - the seq to pass forward is the one that is flushed to the right
            #   boundary.
            # - that is given by seq_idx_chunk_end below: the sequence index at the end of the chunk.
            seq_idx_chunk_end = tl.load(
                seq_idx_ptr
                + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen
            )
            if HAS_INITSTATES:
                if IS_CONT_BATCHED and prev_seq_idx_chunk_end != seq_idx_chunk_end:
                    # this means in the current chunk the rightmost flushed seq
                    # has changed.
                    # - so we do not propagate the state from previous chunk
                    # - but rather we load that sequence's init state
                    initstates_ptrs = (
                        initstates_ptr + seq_idx_chunk_end * stride_initstates_batch
                    )

                    # - update state with seq_idx_new's init state
                    states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(
                        tl.float32
                    )

                    # - we need to consider the cumsum only of the last sequence in the chunk
                    # - find its starting position (given by c_off of the logical chunk index)
                    # - and subtract the cumsum just before that position from the total cumsum
                    # - first, update the logical chunk index (add the number of sequences in the current physical chunk):
                    # sequence index at the start of the current chunk
                    seq_idx_chunk_start = tl.load(
                        seq_idx_ptr
                        + min(c * chunk_size, seqlen) * stride_seq_idx_seqlen
                    )
                    logical_chunk_idx += seq_idx_chunk_end - seq_idx_chunk_start
                    # - load the chunk offset:
                    c_off = tl.load(
                        chunk_offsets_ptr + logical_chunk_idx,
                        mask=logical_chunk_idx < chunk_meta_num,
                        other=0,
                    )
                    # - if offset is 0, then the sequence starts at the beginning of the chunk, and we don't need to subtract anything
                    if c_off > 0:
                        # - dA_cs_ptr currently points to the cumsum at the end of the chunk - subtract the chunk size and add the offset
                        dA_cs_boundary = tl.load(
                            dA_cs_ptr
                            - (chunk_size - 1) * stride_dA_cs_csize
                            + (c_off - 1) * stride_dA_cs_csize,
                            mask=(c_off - 1) > -1 and c_off < chunk_size,
                            other=0.0,
                        )
                        dA_cs -= dA_cs_boundary

                # - increment logical chunk index for every physical chunk
                logical_chunk_idx += 1
            else:
                scale_mask = seq_idx_chunk_end == prev_seq_idx_chunk_end
            prev_seq_idx_chunk_end = seq_idx_chunk_end

        scale = tl.where(scale_mask, tl.exp(dA_cs), 0.0)
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def state_passing_fwd(
    states,
    dA_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
    chunk_offsets=None,
):
    """SGLang's `_state_passing_fwd` (`ssd_state_passing.py:188`); returns `(out, final_states)`."""
    batch, nchunks, nheads, dim = states.shape
    if chunk_size is None:
        chunk_size = dA_cumsum.shape[-1]
    else:
        assert chunk_size == dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if initial_states is not None:
        if is_cont_batched:
            # - if cu_seqlens is provided, then the initial states
            #   are used for continuous batching. In which case we
            #   require seq_idx to be provided
            assert seq_idx is not None, (
                "seq_idx must be provided for continuous batching"
            )
            # - we also need chunk_offsets to be provided, to account
            #   for computation of dA_cumsum from the start of the
            #   sequence
            assert chunk_offsets is not None, (
                "chunk_offsets must be provided for continuous batching"
            )
        else:
            # - this is the regular batching case, where initial
            #   states are used are for each example of the batch.
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty(
        (batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype
    )
    final_states = torch.empty(
        (batch, nheads, dim), device=states.device, dtype=torch.float32
    )
    if initial_states is not None:
        stride_initstates_batch = initial_states.stride(0)
        stride_initstates_head = initial_states.stride(1)
        stride_initstates_dim = initial_states.stride(2)
    else:
        stride_initstates_batch, stride_initstates_head, stride_initstates_dim = 0, 0, 0
    if seq_idx is not None:
        stride_seq_idx_batch, stride_seq_idx_seqlen = seq_idx.stride(0), seq_idx.stride(1)
    else:
        stride_seq_idx_batch, stride_seq_idx_seqlen = 0, 0
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(dim, BLOCK_SIZE), batch, nheads)
    with torch.get_device_module(states.device).device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            final_states,
            dA_cumsum,
            initial_states,
            seq_idx,
            chunk_offsets,
            len(chunk_offsets) if chunk_offsets is not None else 0,
            dim,
            nchunks,
            seqlen if seq_idx is not None else 0,
            chunk_size,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            final_states.stride(0),
            final_states.stride(1),
            final_states.stride(2),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            stride_initstates_batch,
            stride_initstates_head,
            stride_initstates_dim,
            stride_seq_idx_batch,
            stride_seq_idx_seqlen,
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_CONT_BATCHED=is_cont_batched,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=3,
        )
    return out, final_states


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


def make_inputs(seqlens, nheads, dim, chunk_size, generator):
    """A packed prefill batch as SGLang's Mamba-2 layer builds it (mamba2_metadata.py:261-286): seq_idx,
    chunk_offsets, random chunk states, and the per-chunk cumulative sums of dA = A * dt (A < 0,
    dt = softplus(.) > 0, 0 past seqlen), as `_chunk_cumsum_fwd` gives them. On the CPU."""
    seqlen = sum(seqlens)
    nchunks = triton.cdiv(seqlen, chunk_size)
    query_start_loc = torch.tensor([0] + seqlens).cumsum(0)
    seq_idx = torch.repeat_interleave(torch.arange(len(seqlens), dtype=torch.int32),
                                      torch.tensor(seqlens)).unsqueeze(0)
    _, chunk_offsets = query_start_loc_to_chunk_indices_offsets(query_start_loc, chunk_size, seqlen)
    A = -torch.exp(torch.rand(nheads, generator=generator) * 2.0)
    dt = torch.nn.functional.softplus(torch.randn((nheads, nchunks * chunk_size), generator=generator) - 6.0)
    dt[:, seqlen:] = 0.0
    dA_tok = A[:, None] * dt                                       # [nheads, nchunks * chunk_size]
    dA_cumsum = dA_tok.reshape(nheads, nchunks, chunk_size).cumsum(-1).unsqueeze(0)
    states = torch.randn((1, nchunks, nheads, dim), generator=generator)
    return states, dA_cumsum, seq_idx, chunk_offsets.to(torch.int32), dA_tok


def reference(states, dA_tok, seq_idx, initial_states, chunk_size):
    """The recurrence in float64 from the per-token dA (a request's decay counts only its own
    tokens), with the request starts taken from seq_idx, each request starting from its initial
    state (from 0 when initial_states is None); returns out and final_states."""
    states = states[0].double()
    _, nchunks, nheads, dim = (1,) + tuple(states.shape)
    dA_tok = dA_tok.double()
    sid = seq_idx[0].tolist()
    seqlen = len(sid)
    if initial_states is None:
        initial_states = torch.zeros((sid[-1] + 1, nheads, dim))
    init = initial_states.double()
    out = torch.empty((nchunks, nheads, dim), dtype=torch.float64)
    s = init[0].clone()
    out[0] = s
    prev = 0
    final = None
    for c in range(nchunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, seqlen)
        e = sid[hi - 1]
        if e != prev:
            start = sid.index(e)  # e starts inside this chunk
            s = init[e].clone()
            dA = dA_tok[:, start:hi].sum(-1)
        else:
            dA = dA_tok[:, lo:hi].sum(-1)
        s = torch.exp(dA)[:, None] * s + states[c]
        if c < nchunks - 1:
            out[c + 1] = s
        else:
            final = s
        prev = e
    return out.unsqueeze(0), final.unsqueeze(0)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (request lengths, nheads, headdim * dstate, chunk_size): SGLang's tests
    # (test/registered/layers/mamba/test_mamba_ssm_ssd.py: chunk sizes 8 to 256, prefill-chunking
    # lengths (16, 20) and (270, 88, 212, 203)), and Mamba-2 / Nemotron-H heads (headdim 64,
    # dstate 128: dim 8192; Nemotron Nano 2: headdim 80, dim 10240), nheads after tensor parallelism.
    cases = [
        ([1], 8, 8192, 256),
        ([256], 8, 8192, 256),
        ([16, 20], 4, 32 * 16, 8),
        ([270, 88, 212, 203], 4, 5 * 16, 8),
        ([270, 88, 212, 203], 16, 8192, 256),
        ([5, 1, 1, 1, 300, 7, 128, 129], 16, 8192, 128),
        ([1000, 1, 2049, 64, 64, 300], 32, 10240, 128),
        ([4096, 4096], 128, 8192, 256),
        ([3] * 40 + [500], 8, 8192, 64),
    ]
    for seqlens, nheads, dim, chunk_size in cases:
        states, dA_cumsum, seq_idx, chunk_offsets, dA_tok = make_inputs(seqlens, nheads, dim, chunk_size, gen)
        # initial states: zero for requests without a cached prefix, as SGLang's torch.where gives
        has_prefix = torch.rand(len(seqlens), generator=gen) < 0.7
        init = torch.randn((len(seqlens), nheads, dim), generator=gen) * has_prefix[:, None, None]
        # with initial states (some request has a cached prefix), and without (no request has one:
        # SGLang passes initial_states = None and builds no chunk_offsets)
        for has_init in (True, False):
            out, final = state_passing_fwd(states.to(DEVICE), dA_cumsum.to(DEVICE),
                                           initial_states=init.to(DEVICE) if has_init else None,
                                           seq_idx=seq_idx.to(DEVICE), chunk_size=chunk_size,
                                           out_dtype=torch.float32, is_cont_batched=True,
                                           chunk_offsets=chunk_offsets.to(DEVICE) if has_init else None)
            ref_out, ref_final = reference(states, dA_tok, seq_idx, init if has_init else None, chunk_size)
            scale = max(ref_out.abs().max().item(), ref_final.abs().max().item())
            err_out = (out.double().cpu() - ref_out).abs().max().item() / scale
            err_fin = (final.double().cpu() - ref_final).abs().max().item() / scale
            print(f"requests {len(seqlens)} tokens {sum(seqlens)} chunks {states.shape[1]} heads {nheads} dim {dim} "
                  f"chunk {chunk_size} initial states {has_init}: max error / max |state| {err_out:.1e} (out), "
                  f"{err_fin:.1e} (final)")
            assert err_out < 1e-5 and err_fin < 1e-5
    print("✅ SSD state passing matches the float64 reference")
