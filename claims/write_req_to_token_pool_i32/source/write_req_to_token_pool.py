"""
Write a batch's KV indices into the request-to-token table (SGLang's extend step)
================================================================================

SGLang's request-to-token table ``req_to_token`` holds, for each request slot (a row), the
KV-cache location of each of the request's tokens. When a batch of requests is extended
(prefill), each request ``b`` gets its row ``req_pool_indices[b]``; its first ``prefix_lens[b]``
tokens are a cached prefix whose locations are in a tensor of its own (``prefix_tensors[b]``),
and its next ``seq_lens[b] - prefix_lens[b]`` tokens get the newly allocated locations, the
batch's ``out_cache_loc`` taken in batch order (request ``b``'s start the sum of the preceding
requests' ``extend_lens``). One program per request writes both into its row, in blocks of 512
(``BLOCK_SIZE``).

Layouts:

* ``req_to_token``: ``[size + 1, max_context_len]``, ``int32``, row stride
  ``req_to_token.shape[1]`` (a compile-time argument of the kernel);
* ``req_pool_indices``, ``prefix_lens``, ``seq_lens``, ``extend_lens``: ``[bs]``, ``int64``;
* ``prefix_tensors``: a list of ``bs`` ``int64`` tensors, request ``b``'s holding at least
  ``prefix_lens[b]`` locations (an empty one when ``prefix_lens[b] = 0``); the kernel gets their
  data pointers as a ``[bs]`` ``uint64`` tensor;
* ``out_cache_loc``: ``[sum(extend_lens)]``, ``int64``, the new locations, stored into the
  ``int32`` table (``tl.store`` converts).

Provenance: SGLang v0.5.23, commit ``297e54fd10``:

* ``write_req_to_token_pool_triton``: ``python/sglang/kernels/ops/memory/common.py:9``;
* the host ``write_cache_indices`` is adapted from ``write_cache_indices`` in
  ``python/sglang/srt/mem_cache/allocation.py:56-103``, whose one caller is ``alloc_for_extend``
  (``allocation.py:374-488``), the scheduler's allocation for an extend batch
  (``ScheduleBatch.prepare_for_extend``).

Changes from the source:

* ``write_req_to_token_pool_triton`` (``common.py:8-56``) is copied verbatim, with its
  ``@triton.jit`` and the module's imports (``common.py:1-5``); the module's other kernels
  (``get_last_loc_*``, ``common.py:59-163``) are left out.
* The host ``write_cache_indices`` keeps only the Triton branch of ``allocation.py:56-90`` (the
  attention-backend test and the per-request fallback, lines 70-71 and 91-103, are dropped). It
  takes the table ``req_to_token`` instead of the pool object, and the device tensors only (the
  CPU mirrors were used by the fallback only). The pointer tensor ``prefix_pointers`` is built
  as there, ``uint64`` data pointers moved to the table's device, but without pinned memory and
  with a blocking copy. The launch is the same (grid ``(bs,)``, the arguments in the same order,
  ``req_to_token.shape[1]`` for the stride), with the launch options ``num_warps=4,
  num_stages=3``: the source passes none, and these are Triton 3.8.0's defaults.
* ``reference`` and the ``__main__`` test are new.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    seq_lens_tensor,
    extend_lens_tensor,
    prefix_tensors,
    req_to_token,
):
    """Write each request's prefix locations and new locations into its row of `req_to_token`.

    Adapted from ``allocation.py:56-90`` (the Triton branch).
    """
    prefix_pointers = torch.tensor(
        [t.data_ptr() for t in prefix_tensors],
        dtype=torch.uint64,
    ).to(req_to_token.device)
    write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
        req_to_token,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor,
        extend_lens_tensor,
        out_cache_loc,
        req_to_token.shape[1],
        num_warps=4,
        num_stages=3,
    )
    return prefix_pointers


def reference(req_to_token, req_pool_indices, prefix_lens, seq_lens, extend_lens,
              prefix_tensors, out_cache_loc):
    """The per-request writes of the fallback (``allocation.py:91-103``), on a copy (exact: the
    ``int64`` locations stored into the ``int32`` table keep their low 32 bits)."""
    out = req_to_token.clone()
    pt = 0
    for b in range(req_pool_indices.numel()):
        row, pre, seq = int(req_pool_indices[b]), int(prefix_lens[b]), int(seq_lens[b])
        out[row, :pre] = prefix_tensors[b][:pre].to(out.dtype)
        out[row, pre:seq] = out_cache_loc[pt:pt + seq - pre].to(out.dtype)
        pt += int(extend_lens[b])
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    empty = torch.empty((0,), dtype=torch.int64)  # the host's prefix of a request without one
    # (pool size, model context length, batch, largest prefix, largest extend)
    cases = [
        (64, 4096, 1, 0, 4096),
        (256, 4096, 37, 2048, 2048),
        (4096, 4096, 512, 1024, 32),
        (64, 32768, 8, 20000, 8192),
        (8, 131072, 4, 100000, 16384),
    ]
    for size, ctx, bs, max_pre, max_ext in cases:
        mcl = ctx + 4  # ReqToTokenPool's max_context_len: the context length + 4
        req_to_token = torch.zeros((size + 1, mcl), dtype=torch.int32)
        rows = (torch.randperm(size, generator=gen) + 1)[:bs]  # row 0 is the padding row
        pre = torch.randint(0, max_pre + 1, (bs,), generator=gen)
        ext = torch.randint(1, max_ext + 1, (bs,), generator=gen)
        ext = torch.minimum(ext, mcl - pre)
        seq = pre + ext
        slots = torch.randperm(int(ext.sum()) + 1000, generator=gen)[:int(ext.sum())] + 1
        prefixes = [torch.randint(1, 2 ** 31, (int(p),), generator=gen).to(DEVICE)
                    if int(p) > 0 else empty for p in pre]
        args = [t.to(DEVICE) for t in (rows, pre, seq, ext)]
        out_cache_loc = slots.to(DEVICE)
        req_to_token = req_to_token.to(DEVICE)
        ref = reference(req_to_token, *args, prefixes, out_cache_loc)
        write_cache_indices(out_cache_loc, *args, prefixes, req_to_token)
        assert torch.equal(req_to_token, ref)
        print(f"pool {size} x {mcl}, batch {bs}: {int(seq.sum())} locations written, "
              f"longest {int(seq.max())}: exact")
    print("write_cache_indices matches the reference exactly")
