"""
KV indices for FlashInfer (a batch's KV token indices gathered into one flat list)
=================================================================================

The request-to-token table ``req_to_token`` holds, for each request slot, the KV-cache location
of each of its tokens. For a batch of requests, FlashInfer's paged attention wants their
locations in one flat list in CSR form: request ``b``'s ``seq_lens[b]`` locations, starting at
position ``kv_start_idx[b]`` of its row (``0`` without ``kv_start_idx``), are written at
``kv_indices[kv_indptr[b] : kv_indptr[b] + seq_lens[b]]``. One program per request copies them in
blocks of 512 (``BLOCK_SIZE``).

Layouts:

* ``req_to_token``: ``[num_slots, max_context_len]``, ``int32``, row stride
  ``req_to_token.shape[1]``;
* ``req_pool_indices``: ``[bs]``, ``int64`` (or ``int32``), each request's row of
  ``req_to_token``;
* ``seq_lens`` (the kernel's ``page_kernel_lens``): ``[bs]``, ``int64`` (or ``int32``);
* ``kv_start_idx`` (optional): ``[bs]``, each request's first position (a sliding window's start);
* ``kv_indptr``: ``[bs + 1]``, ``int32``, the exclusive prefix sums of ``seq_lens``;
* ``kv_indices`` (the output): ``[sum(seq_lens)]``, ``int32``.

The kernel also has paths the proved configuration does not take: a page-granular source table
(``ENTRY_PAGE_SIZE > 1``), several programs per request (``TOKEN_BLOCK_PARALLEL``) and a remapping
of the gathered ids (``token_mapping``).

Provenance: SGLang v0.5.23, commit ``297e54fd10``:

* ``create_flashinfer_kv_indices_triton``:
  ``python/sglang/kernels/ops/attention/kv_indices.py:59``;
* the host ``create_kv_indices`` is adapted from the launch in
  ``ForwardBatchDeepSeekMHAMixin.fetch_mha_one_shot_kv_indices``
  (``python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py:265-296``).
  FlashInfer's decode and prefill planners build ``kv_indptr`` and ``kv_indices`` the same way
  (``python/sglang/srt/layers/attention/flashinfer_backend.py:1788-1809``), with their
  ``kv_start_idx``, and launch the kernel through ``KVIndexTranslator.pack_read_stream``
  (``python/sglang/srt/mem_cache/kv_index_translator.py:600``), whose source table, grid and
  constexpr values depend on the KV pool.

Changes from the source:

* ``create_flashinfer_kv_indices_triton`` (``kv_indices.py:59``) is copied verbatim, except
  for its parameter defaults (below), with its ``@triton.jit`` (the source has no autotuning
  or heuristics on it).
* The launched kernel's parameter defaults are removed, in its signature only
  (``kv_indices.py:71-73``): ``ENTRY_PAGE_SIZE: tl.constexpr = 1``,
  ``TOKEN_BLOCK_PARALLEL: tl.constexpr = False`` and ``token_mapping=None`` become
  ``ENTRY_PAGE_SIZE: tl.constexpr``, ``TOKEN_BLOCK_PARALLEL: tl.constexpr`` and ``token_mapping``.
  TVA's subset has no kernel parameter defaults; the host passes each parameter.
* The host ``create_kv_indices`` follows ``forward_batch_deepseek_mha_mixin.py:265-296``:
  ``kv_indptr`` zeroed and filled with the cumulative sums of ``seq_lens``, ``kv_indices``
  allocated with ``torch.empty`` (``int32``, ``sum(seq_lens)`` elements), the grid ``(bs,)`` and
  the arguments in the same order. It takes ``kv_start_idx`` (``None`` there; the FlashInfer
  planners pass theirs, ``flashinfer_backend.py:1807``), and computes ``bs`` and the sum of the
  lengths from ``seq_lens`` instead of the batch's fields. It passes the constexpr parameters
  ``ENTRY_PAGE_SIZE=1``, ``TOKEN_BLOCK_PARALLEL=False`` and ``token_mapping=None`` explicitly (the
  source kernel's defaults, which that launch gets) and the launch options ``num_warps=4,
  num_stages=3`` (the launch passes none, so these are Triton 3.8.0's defaults). It returns
  ``kv_indptr`` and ``kv_indices``.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len] token table; at
    # ENTRY_PAGE_SIZE > 1 a PAGE-granular table (the unified pool's read table)
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    # Runtime, not constexpr: the translator's eager table is allocated at the
    # batch's live width, so a constexpr stride would JIT-specialize per width
    # (a recompile every few decode steps at small page sizes).
    req_to_token_ptr_stride,
    ENTRY_PAGE_SIZE: tl.constexpr,
    TOKEN_BLOCK_PARALLEL: tl.constexpr,
    token_mapping,
):
    """Gather per-request token ids into a flat CSR kv_indices stream.

    ``ENTRY_PAGE_SIZE == 1`` (default): the source table is token-granular and
    entries are emitted verbatim -- byte-identical to the historical kernel.
    ``ENTRY_PAGE_SIZE == ps``: the source is the translator's PAGE-granular
    read table (entries already physical page ids); token ids are rebuilt
    as ``token = entry * ps + pos % ps``, exact because converting an id keeps
    its offset inside the page.
    ``TOKEN_BLOCK_PARALLEL`` (default False): launched on a 2D grid
    ``(batch, num_blocks)``, the programs of a request stride over its copy
    loop together instead of one program crawling the whole context serially
    (which bottlenecks long-context spec decode, where this kernel runs every
    iteration). With the default, the kernel is the historical
    one-program-per-request loop and 1D launch sites are unaffected.
    ``token_mapping`` remaps gathered token ids in the same pass, preserving -1.
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    if TOKEN_BLOCK_PARALLEL:
        blk = tl.program_id(axis=1)
        num_blk = tl.num_programs(axis=1)
    else:
        blk = 0
        num_blk = 1

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    if TOKEN_BLOCK_PARALLEL:
        # Blocks with no copy work exit early.
        if blk >= num_loop:
            return
    for i in range(blk, num_loop, num_blk):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        if ENTRY_PAGE_SIZE == 1:
            data = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + kv_start
                + offset,
                mask=mask,
            )
        else:
            pos = kv_start + offset
            entry = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + pos // ENTRY_PAGE_SIZE,
                mask=mask,
            )
            data = entry.to(tl.int64) * ENTRY_PAGE_SIZE + pos % ENTRY_PAGE_SIZE
        if token_mapping is not None:
            data = tl.load(token_mapping + data, mask=mask & (data >= 0), other=-1)
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def create_kv_indices(req_to_token, req_pool_indices, seq_lens, kv_start_idx=None):
    """``kv_indptr`` and ``kv_indices``: request ``b``'s ``seq_lens[b]`` entries of row
    ``req_pool_indices[b]`` of ``req_to_token``, from position ``kv_start_idx[b]`` (or 0).

    Adapted from ``forward_batch_deepseek_mha_mixin.py:265-296``.
    """
    batch_size = seq_lens.numel()
    paged_kernel_lens_sum = int(seq_lens.sum())
    kv_indices = torch.empty(
        paged_kernel_lens_sum,
        dtype=torch.int32,
        device=req_pool_indices.device,
    )
    kv_indptr = torch.zeros(
        batch_size + 1,
        dtype=torch.int32,
        device=req_pool_indices.device,
    )
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    create_flashinfer_kv_indices_triton[(batch_size,)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
        req_to_token.shape[1],
        ENTRY_PAGE_SIZE=1,
        TOKEN_BLOCK_PARALLEL=False,
        token_mapping=None,
        num_warps=4,
        num_stages=3,
    )
    return kv_indptr, kv_indices


def reference(req_to_token, req_pool_indices, seq_lens, kv_start_idx=None):
    """The rows' slices, concatenated (exact)."""
    parts = []
    for b in range(seq_lens.numel()):
        s = 0 if kv_start_idx is None else int(kv_start_idx[b])
        parts.append(req_to_token[int(req_pool_indices[b]), s:s + int(seq_lens[b])])
    return torch.cat(parts) if parts else req_to_token.new_empty(0)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (num_slots, max_context_len, batch, lengths or None for random in [0, max_context_len),
    #  index dtype, with kv_start_idx)
    cases = [
        (4096, 4096, 1, None, torch.int32, False),
        (4096, 4096, 37, None, torch.int32, False),
        (4096, 4096, 1786, None, torch.int64, False),
        (513, 8192, 6, [1, 511, 512, 513, 1025, 8192], torch.int64, False),
        (9, 131072, 8, [131072, 1, 70000, 4097, 0, 33, 99999, 512], torch.int64, False),
        (2049, 32768, 256, None, torch.int64, False),
        (300, 4096, 100, None, torch.int64, True),
    ]
    for num_slots, ctx, bs, lens, idt, with_start in cases:
        # the pool's table: each slot's token locations, distinct across the pool
        req_to_token = torch.randperm(num_slots * ctx, generator=gen).to(torch.int32).reshape(num_slots, ctx)
        req_pool_indices = torch.randperm(num_slots, generator=gen)[:bs]
        if lens is None:
            lens = torch.randint(0, ctx, (bs,), generator=gen)
        else:
            lens = torch.tensor(lens)
        start = None
        if with_start:  # a sliding window: the last `window` positions of each request
            window = torch.randint(1, 1024, (bs,), generator=gen)
            gather = torch.minimum(lens, window)
            start = (lens - gather).to(idt).to(DEVICE)
            lens = gather
        req_to_token = req_to_token.to(DEVICE)
        req_pool_indices = req_pool_indices.to(device=DEVICE, dtype=idt)
        lens = lens.to(device=DEVICE, dtype=idt)
        kv_indptr, kv_indices = create_kv_indices(req_to_token, req_pool_indices, lens, start)
        ref = reference(req_to_token, req_pool_indices, lens, start)
        assert torch.equal(kv_indices, ref)
        assert torch.equal(kv_indptr[1:].long(), torch.cumsum(lens, dim=0).long())
        print(f"slots {num_slots} context {ctx} batch {bs} {str(idt).split('.')[-1]} "
              f"start {'yes' if with_start else 'no'}: {kv_indices.numel()} indices, max length "
              f"{int(lens.max())}: exact")
    print("create_kv_indices matches the reference exactly")
