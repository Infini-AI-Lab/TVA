"""
Expert-parallel MoE source-to-destination map (``src2dst``) after sorting the routed expert ids
================================================================================================

Each token of a MoE layer is routed to ``topk`` experts; ``topk_ids`` (``[num_tokens, topk]``,
``int32``) holds the expert of each routing slot, with a negative id for a slot that goes to no
local expert. The host sorts the flat ids stably (``torch.sort``), so ``sorted_topk_ids`` is
ascending and ``reorder_ids[i]`` (``int64``) is the slot that lands at sorted position ``i``. The
kernel writes, for each sorted position ``i``, ``src2dst[reorder_ids[i]]``: the slot's row in the
packed per-expert buffer, ``i - first_nonnegative``, where ``first_nonnegative`` is the number of
negative ids (found by a binary search over the sorted ids, in every program); a slot with a
negative id gets ``-1``. One program handles 512 sorted positions.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/moe/ep_moe_kernels.py``:

* ``compute_src2dst_triton_kernel``: lines 146-168;
* the host ``cutlass_w4_run_moe_ep_preproess``: lines 205-217, called by ``cutlass_w4a8_moe``
  (``python/sglang/srt/layers/moe/cutlass_w4a8_moe.py:166``, on the path without the CUDA
  ``cutlass_prepare_moe_input``) with the contiguous ``int32`` ``topk_ids`` of the layer.

Changes from the source:

* ``compute_src2dst_triton_kernel`` and ``cutlass_w4_run_moe_ep_preproess`` are copied verbatim,
  with the kernel's ``@triton.jit`` (the source has no autotuning or heuristics on it).
* The source module's other kernels and hosts, its logging, its ``sglang`` imports and its device
  checks are dropped: neither copied function uses them.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def compute_src2dst_triton_kernel(
    sorted_topk_ids, reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    first_nonnegative = 0
    if tl.load(sorted_topk_ids) < 0:
        left, right = 0, num_toks
        while left < right:
            middle = left + (right - left) // 2
            expert_id = tl.load(sorted_topk_ids + middle)
            if expert_id < 0:
                left = middle + 1
            else:
                right = middle
        first_nonnegative = left

    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    # Native expert offsets omit negative IDs, so packed rows must omit that prefix.
    packed_id = tl.where(dst_id >= first_nonnegative, dst_id - first_nonnegative, -1)
    tl.store(src2dst + src_id, packed_id, mask=mask)


def cutlass_w4_run_moe_ep_preproess(topk_ids: torch.Tensor):
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    if topk_ids.numel() == 0:
        return src2dst
    sorted_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)

    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    compute_src2dst_triton_kernel[grid](
        sorted_topk_ids, reorder_ids, src2dst, topk_ids.numel(), BLOCK_SIZE
    )

    return src2dst


def reference(topk_ids):
    """``src2dst`` from its definition (exact): the stable sort's permutation, each slot's packed
    row ``i - (number of negative ids)`` at sorted position ``i``, ``-1`` for a negative id."""
    flat = topk_ids.view(-1).long().cpu()
    order = torch.sort(flat, stable=True).indices
    num_neg = int((flat < 0).sum())
    out = torch.empty(flat.numel(), dtype=torch.int64)
    pos = torch.arange(flat.numel())
    out[order] = torch.where(pos >= num_neg, pos - num_neg, -1)
    return out.to(torch.int32)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (num_tokens, topk, num_local_experts, share of slots routed to no local expert)
    cases = [
        (1, 1, 8, 0.0),
        (7, 8, 16, 0.5),
        (512, 8, 32, 0.0),
        (1000, 8, 64, 0.75),
        (4096, 6, 160, 0.25),
        (300, 4, 8, 1.0),
    ]
    for num_tokens, topk, num_experts, neg_share in cases:
        topk_ids = torch.randint(0, num_experts, (num_tokens, topk), generator=gen, dtype=torch.int32)
        drop = torch.rand((num_tokens, topk), generator=gen) < neg_share
        topk_ids[drop] = -1
        topk_ids = topk_ids.to(DEVICE)
        src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)
        assert torch.equal(src2dst.cpu(), reference(topk_ids))
        print(f"tokens {num_tokens} topk {topk} experts {num_experts} "
              f"negative {int((topk_ids < 0).sum())}: exact")
    print("cutlass_w4_run_moe_ep_preproess matches the reference exactly")
