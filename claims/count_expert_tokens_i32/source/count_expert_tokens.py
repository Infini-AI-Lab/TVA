"""
Expert token counts for MoE (vLLM)
==================================

For each local expert ``e``, the number of entries of ``topk_ids`` routed to it: the count of
positions ``i < topk_ids.numel()`` with ``expert(topk_ids[i]) == e``, where ``expert`` is the
identity without an expert map, and ``expert_map[id]`` (``-1`` for ``id < 0``) with one (expert
parallelism: global expert ids are mapped to this rank's local ids, ``-1`` for experts on other
ranks). One program per local expert scans all of ``topk_ids`` in blocks of ``BLOCK_SIZE``.

Layouts and dtypes:

* ``topk_ids``: ``[num_tokens, topk]``, contiguous, a signed integer dtype (``int32`` from vLLM's
  routers by default, ``int64`` from DeepEP's dispatch and in vLLM's test); ``-1`` marks an
  invalid entry; read as a flat array of ``topk_ids.numel()`` elements;
* ``expert_map`` (optional): ``[num_global_experts]``, ``int32``, global id to local id or ``-1``;
* ``expert_num_tokens`` (output, allocated fresh): ``[num_local_experts]``, ``int32``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``), ``vllm/model_executor/layers/fused_moe/utils.py``:

* ``_count_expert_num_tokens``: ``utils.py:170-200`` (``@triton.jit`` at line 170, ``def`` at 171);
* ``count_expert_num_tokens`` (host wrapper): ``utils.py:202-238``.

Changes from the source:

* ``utils.py:170-200``: the kernel, copied verbatim with its ``@triton.jit``; it has no autotuning
  or heuristics and calls no helper.
* ``utils.py:202-238``: the host ``count_expert_num_tokens`` copied with the same assert,
  allocation, grid and arguments; its docstring is shortened, and the launch options the wrapper
  leaves to Triton's defaults are written out (``num_warps=4``, ``num_stages=3``, Triton 3.8.0's
  defaults), which changes nothing.
* ``reference``, ``make_topk_ids``, ``make_expert_map`` and the ``__main__`` test are new; the
  expert map follows vLLM's ``determine_expert_map``
  (``vllm/model_executor/layers/fused_moe/expert_map_manager.py:60-112``, the "linear" and
  "round_robin" placements), and the test's shapes follow
  ``tests/kernels/moe/test_count_expert_num_tokens.py``.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        if HAS_EXPERT_MAP:
            expert_map_ptrs = expert_map + expert_ids
            expert_map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptrs, mask=expert_map_mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))



def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    """Count the number to tokens assigned to each expert (vLLM's wrapper, `utils.py:202`).

    Returns a tensor of size num_local_experts, where tensor[i] holds the number of tokens
    assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    return expert_num_tokens


def make_topk_ids(num_tokens, topk, num_experts, dtype, generator, invalid_frac=0.0):
    """Each token's top-k distinct experts (as a router gives), some entries set to -1."""
    ids = torch.stack([torch.randperm(num_experts, generator=generator)[:topk] for _ in range(num_tokens)])
    if invalid_frac > 0:
        drop = torch.rand(ids.shape, generator=generator) < invalid_frac
        ids[drop] = -1
    return ids.to(dtype)


def make_expert_map(num_global_experts, ep_size, ep_rank, placement="linear"):
    """vLLM's `determine_expert_map` (expert_map_manager.py:60-112): global id -> local id or -1."""
    base = num_global_experts // ep_size
    rem = num_global_experts % ep_size
    num_local = base + 1 if ep_rank < rem else base
    expert_map = torch.full((num_global_experts,), -1, dtype=torch.int32)
    if placement == "linear":
        start = ep_rank * base + min(ep_rank, rem)
        expert_map[start:start + num_local] = torch.arange(0, num_local, dtype=torch.int32)
    else:
        local = torch.arange(ep_rank, num_global_experts, ep_size)
        expert_map[local] = torch.arange(0, num_local, dtype=torch.int32)
    return num_local, expert_map


def reference(topk_ids, num_local_experts, expert_map):
    """Exact counts on the CPU."""
    ids = topk_ids.flatten().cpu().long()
    if expert_map is not None:
        emap = expert_map.cpu().long()
        ids = torch.where(ids >= 0, emap[ids.clamp(min=0)], torch.full_like(ids, -1))
    counts = torch.zeros(num_local_experts, dtype=torch.int32)
    for e in range(num_local_experts):
        counts[e] = int((ids == e).sum())
    return counts


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    cases = []
    # vLLM's test: num_tokens x topk x 64 experts, ep_size 1/2/4, int64 ids.
    for num_tokens in [1, 4, 11, 127, 128, 3333, 7317]:
        for topk, ep_size in [(2, 1), (6, 2), (8, 4)]:
            cases.append((num_tokens, topk, 64, ep_size, torch.int64, "linear", 0.0))
    # vLLM's numel test: topk 1, 32 experts, ep_size 2, numel 1 .. 8191 (a few of them).
    for numel in [1, 112, 1024, 1025, 2047, 8191]:
        cases.append((numel, 1, 32, 2, torch.int64, "linear", 0.0))
    # Model configs, int32 ids (the routers' default): DeepSeek-V3 (256 experts, top-8),
    # Qwen3-MoE (128, top-8), Mixtral (8, top-2); invalid (-1) entries; uneven EP split.
    cases += [
        (4096, 8, 256, 8, torch.int32, "linear", 0.0),
        (16384, 8, 256, 16, torch.int32, "round_robin", 0.0),
        (2000, 8, 128, 1, torch.int32, "linear", 0.05),
        (513, 8, 128, 3, torch.int32, "linear", 0.05),
        (777, 2, 8, 1, torch.int32, "linear", 0.0),
        (1, 8, 256, 8, torch.int32, "round_robin", 0.0),
    ]
    for num_tokens, topk, num_experts, ep_size, dtype, placement, invalid in cases:
        topk_ids = make_topk_ids(num_tokens, topk, num_experts, dtype, gen, invalid)
        for ep_rank in sorted({0, ep_size - 1}):
            num_local, emap = make_expert_map(num_experts, ep_size, ep_rank, placement)
            ref = reference(topk_ids, num_local, emap)
            got_map = count_expert_num_tokens(topk_ids.to(DEVICE), num_local, emap.to(DEVICE)).cpu()
            # Without a map: the ids already mapped to local ids (as vLLM's test does).
            local_ids = torch.where(topk_ids >= 0, emap.long()[topk_ids.long().clamp(min=0)], -1).to(dtype)
            got_nomap = count_expert_num_tokens(local_ids.to(DEVICE), num_local, None).cpu()
            ok = torch.equal(got_map, ref) and torch.equal(got_nomap, ref)
            print(f"tokens {num_tokens} topk {topk} experts {num_experts} ep {ep_rank}/{ep_size} {placement} "
                  f"{str(dtype)[6:]} invalid {invalid}: {'exact' if ok else 'MISMATCH'}")
            assert ok
    print("✅ expert token counts match the reference exactly, with and without an expert map")
