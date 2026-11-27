"""
Number of KV splits for split-K decode attention (SGLang's Triton attention backend)
====================================================================================

For a decode batch of ``num_seq`` requests, the Triton backend's split-K decode attention splits
each request's KV sequence into chunks processed by separate programs. This kernel chooses, from
the requests' KV lengths ``seq_lens``, the number of splits of each request: a single program
reads all lengths, takes their maximum and minimum, derives two chunk sizes (one from the ratio
of the longest to the shortest request, one from the device's core count scaled by
``log2(max_seq_len / 64)`` and divided by the attention grid), and stores, for each request,
``max(cdiv(seq_len, chunk_1), cdiv(seq_len, chunk_2))`` at its ``num_group`` token positions.
With ``uneven_batch``, it also stores whether the batch is uneven (``min < max`` after the 80 %
rule).

Layouts:

* ``seq_lens``: ``[num_seq]``, ``int64`` (the batch's lengths) or ``int32`` (lengths computed from
  an ``int32`` index pointer);
* ``num_kv_splits`` (the output): ``[num_seq * num_group]``, ``int32``; token ``s * num_group + g``
  is request ``s``'s ``g``-th draft token (``num_group > 1`` in speculative decoding);
* ``uneven_batch`` (optional output): ``[1]``, ``int32``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``:

* ``get_num_kv_splits_triton``: ``python/sglang/kernels/ops/attention/metadata.py:8-60``;
* the host ``get_num_kv_splits`` is adapted from ``TritonAttnBackend.get_num_kv_splits``
  (``python/sglang/srt/layers/attention/triton_backend.py:431-480``).

Changes from the source:

* ``get_num_kv_splits_triton`` (``metadata.py:8-60``) is copied verbatim, with its
  ``@triton.jit`` (the source has no autotuning or heuristics on it), except for its parameter
  default: ``uneven_batch_ptr=None`` (``metadata.py:19``) becomes ``uneven_batch_ptr``. TVA's
  subset has no kernel parameter defaults; the host passes the parameter (``None`` or a tensor)
  at every launch, as the source's host does. The other kernels and functions of
  ``metadata.py`` are not copied.
* The host ``get_num_kv_splits`` follows ``triton_backend.py:431-480``: ``num_group`` from the
  shapes, the assert, ``SCHEDULE_SEQ`` (``256`` below 256 requests, else the next power of two),
  the grid ``(1,)`` and the arguments in the same order. It takes the backend's attributes
  ``num_head``, ``num_kv_head``, ``max_kv_splits`` and ``device_core_count`` as arguments, and
  leaves out the two branches that return before the launch (``static_kv_splits`` or a
  non-positive core count: ``num_kv_splits.fill_(max_kv_splits)``; deterministic inference with a
  ``split_tile_size``: a torch expression); neither launches the kernel. It passes the launch
  options ``num_warps=4, num_stages=3`` (the source passes none, so these are Triton 3.8.0's
  defaults).
* ``reference`` and the ``__main__`` test are new.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
    uneven_batch_ptr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    if uneven_batch_ptr is not None:
        tl.store(uneven_batch_ptr, (min_seq_len < max_seq_len).to(tl.int32))
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def get_num_kv_splits(
    num_kv_splits: torch.Tensor,
    seq_lens: torch.Tensor,
    num_head: int,
    num_kv_head: int,
    max_kv_splits: int,
    device_core_count: int,
    uneven_batch: Optional[torch.Tensor] = None,
):
    """Fill ``num_kv_splits`` (and ``uneven_batch``) for the batch's ``seq_lens``.

    Adapted from ``TritonAttnBackend.get_num_kv_splits`` (``triton_backend.py:431-480``).
    """
    num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
    num_group = num_token // num_seq

    assert num_group * num_seq == num_token, (
        f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"
    )

    if num_seq < 256:
        SCHEDULE_SEQ = 256
    else:
        SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        num_seq,
        num_group,
        num_head,
        num_kv_head,
        max_kv_splits,
        device_core_count,
        MAX_NUM_SEQ=SCHEDULE_SEQ,
        uneven_batch_ptr=uneven_batch,
        num_warps=4,
        num_stages=3,
    )


def _cdiv(a, b):
    return (a + b - 1) // b


def reference(seq_lens, num_group, num_head, num_kv_head, max_kv_splits, device_core_count):
    """The kernel's computation on the host, with ``ext_device_core_count`` computed in float64
    (``int(C * max(log2(L / 64), 1))``); returns the ``num_kv_splits`` list and the uneven flag."""
    import math

    lens = [int(x) for x in seq_lens.tolist()]
    num_seq = len(lens)
    mx = max(lens)
    mn = min(lens)
    if mx * 8 < mn * 10:
        mn = mx
    uneven = 1 if mn < mx else 0
    k1 = min(_cdiv(mx, mn), max_kv_splits)
    c1 = _cdiv(mx, k1)
    ext = int(device_core_count * max(math.log2(mx / 64.0), 1.0))
    g = num_head // num_kv_head
    if g == 1:
        tg = num_seq * num_group * num_head
    else:
        tg = num_seq * num_group * _cdiv(num_head, min(16, g))
    k2 = min(_cdiv(ext, tg), max_kv_splits)
    c2 = _cdiv(mx, k2)
    out = []
    for x in lens:
        out += [max(_cdiv(x, c1), _cdiv(x, c2))] * num_group
    return out, uneven


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (lengths or (batch, max length) for random lengths, num_group, num_head, num_kv_head,
    #  max_kv_splits, dtype, with uneven_batch)
    cases = [
        ([1], 1, 32, 8, 8, torch.int64, True),
        ([4096] * 8, 1, 32, 32, 8, torch.int64, True),
        ((64, 32768), 1, 128, 1, 8, torch.int64, True),
        ((300, 131072), 1, 64, 8, 16, torch.int64, False),
        ((7, 2000), 4, 32, 8, 8, torch.int32, False),
    ]
    for lens, ng, nh, nkv, mks, dt, with_uneven in cases:
        if isinstance(lens, tuple):
            bs, top = lens
            lens = torch.randint(1, top + 1, (bs,), generator=gen)
        else:
            lens = torch.tensor(lens)
        seq_lens = lens.to(device=DEVICE, dtype=dt)
        out = torch.full((lens.numel() * ng,), -1, dtype=torch.int32, device=DEVICE)
        uneven = torch.zeros((1,), dtype=torch.int32, device=DEVICE) if with_uneven else None
        cores = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        get_num_kv_splits(out, seq_lens, nh, nkv, mks, cores, uneven)
        ref, ref_uneven = reference(seq_lens.cpu(), ng, nh, nkv, mks, cores)
        assert out.tolist() == ref
        if with_uneven:
            assert int(uneven[0]) == ref_uneven
        print(f"batch {lens.numel()} group {ng} heads {nh}/{nkv} max splits {mks} "
              f"{str(dt).split('.')[-1]}: {out.numel()} splits, exact")
    print("get_num_kv_splits matches the reference")
