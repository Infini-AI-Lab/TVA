"""
Route histogram (pairs counted per LoRA adapter, or per adapter and expert)
===========================================================================

The first step of building a LoRA route: each (token, group) pair, a group being a token's
expert among its top-k (MoE LoRA) or one of its heads, goes to the bucket of its adapter slot
and group, ``slot * GROUPS_PER_SLOT + group`` (or ``slot`` alone without groups). Pairs with an
invalid slot or group go to the last bucket, ``NUM_BUCKETS - 1``. The kernel counts the pairs of
each bucket into ``counts``, which the host allocates zeroed. Each program takes ``BLOCK`` pairs;
with ``BINS > 0`` it counts them with ``tl.histogram`` over ``BINS`` bins (masked-off lanes go
to bin ``BINS - 1``, which is at least ``NUM_BUCKETS`` and not added) and adds its nonzero bins
to ``counts`` with ``tl.atomic_add``; with ``BINS == 0`` each pair adds 1 with its own atomic.

Layouts:

* ``token_slots``: ``[num_tokens]``, ``int32``, each token's adapter slot (``-1`` or a value at
  least ``MAX_LORAS`` is invalid);
* ``group_ids`` (optional): ``[num_tokens, WIDTH]``, ``int32``, contiguous, the pairs' groups
  (MoE LoRA: ``topk_ids``, ``-1`` for an expert not on this rank); pair ``p`` is token
  ``p // WIDTH``'s group ``group_ids.view(-1)[p]``. Without groups, ``WIDTH = 1`` and pair ``p``
  is token ``p``;
* ``counts`` (the output): ``[NUM_BUCKETS]``, ``int32``, ``NUM_BUCKETS = GROUPS_PER_SLOT *
  MAX_LORAS + 1``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/lora/common/routing.py``:

* ``route_bucket_ids``: line 40;
* ``_add_counts``: line 88;
* ``_route_histogram_kernel``: line 105;
* the host ``route_histogram`` is adapted from ``_build_large_route``: line 416 (the histogram
  launch, lines 455-469), with ``_histogram_bins``: line 364.

Changes from the source:

* ``route_bucket_ids`` (``routing.py:40``), ``_add_counts`` (``routing.py:88``) and
  ``_route_histogram_kernel`` (``routing.py:105``) are copied verbatim, with their
  ``@triton.jit`` (the source has no autotuning or heuristics on them).
* ``_histogram_bins`` (``routing.py:364``) is copied with the module constants it reads inlined:
  ``_COUNT_MAX_BINS = 512`` (``routing.py:34``) and ``_COUNT_MIN_PAIRS = 16384``
  (``routing.py:35``).
* The host ``route_histogram`` is adapted from ``_build_large_route`` (``routing.py:416``),
  keeping only its first launch, ``_route_histogram_kernel`` (``routing.py:455-469``): the same
  placeholder pointer for ``group_ids``, ``num_pairs``, ``WIDTH``, bucket count, grid and
  meta-parameters, with ``_HIST_BLOCK = 512`` and ``_HIST_WARPS = 8`` (``routing.py:26-27``)
  inlined. ``counts`` is allocated here with ``torch.zeros``, as ``_aligned_route_scratch``
  allocates it without a workspace (``routing.py:393-406``: ``zero=True``), and returned. The
  later launches (scan and place), the capacity of their buffers and its ``2**31`` check are
  dropped; the bucket count's ``2**31`` check is kept. ``USE_PDL=False``: the source passes
  ``USE_PDL=is_arch_support_pdl()`` (``routing.py:453, 467``), and programmatic dependent launch
  is outside TVA's launches. The launch option ``num_stages=3`` is passed explicitly: the source
  passes none, so this is Triton 3.8.0's default, which its launches get.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def route_bucket_ids(
    group_ids_ptr,
    token_slots_ptr,
    pair_ids,
    pair_mask,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
):
    # Invalid slots/groups map to -1. With one group per slot, any nonnegative
    # group folds into that slot; otherwise bound it to the bucket table.
    token_ids = pair_ids // WIDTH
    slots = tl.load(token_slots_ptr + token_ids, mask=pair_mask, other=-1)
    live = (slots >= 0) & (slots < MAX_LORAS)
    slots = slots.to(tl.int32)
    if HAS_GROUPS:
        groups = tl.load(group_ids_ptr + pair_ids, mask=pair_mask, other=-1)
        live = live & (groups >= 0)
        if GROUPS_PER_SLOT == 1:
            buckets = slots
        else:
            live = live & (groups < GROUPS_PER_SLOT)
            buckets = slots * GROUPS_PER_SLOT + groups.to(tl.int32)
    else:
        buckets = slots
    return tl.where(live, buckets, -1)


@triton.jit
def _add_counts(
    counts_ptr,
    bucket_ids,
    mask,
    NUM_BUCKETS: tl.constexpr,
    BINS: tl.constexpr,
):
    buckets = tl.where(bucket_ids < 0, NUM_BUCKETS - 1, bucket_ids)
    if BINS == 0:
        tl.atomic_add(counts_ptr + buckets, 1, mask=mask)
    else:
        mine = tl.histogram(tl.where(mask, buckets, BINS - 1), BINS)
        bins = tl.arange(0, BINS)
        tl.atomic_add(counts_ptr + bins, mine, mask=(bins < NUM_BUCKETS) & (mine > 0))


@triton.jit
def _route_histogram_kernel(
    group_ids_ptr,
    token_slots_ptr,
    counts_ptr,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # The previous scan (or first allocation) cleared counts.
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    _add_counts(
        counts_ptr,
        route_bucket_ids(
            group_ids_ptr,
            token_slots_ptr,
            pair_ids,
            pair_mask,
            GROUPS_PER_SLOT=GROUPS_PER_SLOT,
            MAX_LORAS=MAX_LORAS,
            WIDTH=WIDTH,
            HAS_GROUPS=HAS_GROUPS,
        ),
        pair_mask,
        NUM_BUCKETS=NUM_BUCKETS,
        BINS=BINS,
    )
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _histogram_bins(num_buckets: int, num_pairs: int) -> int:
    # 0 = one atomic per pair; otherwise in-block counting over 2^k bins.
    if num_buckets >= 512 or num_pairs < 16384:
        return 0
    return 1 << num_buckets.bit_length()  # the extra bin holds the masked-off lanes


def route_histogram(token_slots, group_ids=None, *, groups_per_slot=1, max_loras):
    """The number of pairs in each bucket, ``[groups_per_slot * max_loras + 1]`` ``int32``.

    Adapted from ``_build_large_route`` (``routing.py:416``), its histogram launch only.
    """
    _HIST_BLOCK = 512
    _HIST_WARPS = 8
    groups = token_slots if group_ids is None else group_ids  # placeholder pointer
    num_pairs = groups.numel()
    width = 1 if group_ids is None else group_ids.shape[1]
    num_route_buckets = groups_per_slot * max_loras
    num_buckets = num_route_buckets + 1
    if num_route_buckets + 1 >= 2**31:
        raise ValueError(f"aligned routes use int32 plan math: {num_route_buckets + 1} buckets must be < 2**31")
    counts = torch.zeros((num_buckets,), dtype=torch.int32, device=token_slots.device)
    _route_histogram_kernel[(triton.cdiv(max(num_pairs, 1), _HIST_BLOCK),)](
        groups,
        token_slots,
        counts,
        num_pairs,
        NUM_BUCKETS=num_buckets,
        GROUPS_PER_SLOT=groups_per_slot,
        MAX_LORAS=max_loras,
        WIDTH=width,
        HAS_GROUPS=group_ids is not None,
        BLOCK=_HIST_BLOCK,
        BINS=_histogram_bins(num_buckets, num_pairs),
        USE_PDL=False,
        num_warps=_HIST_WARPS,
        num_stages=3,
    )
    return counts


def reference(token_slots, group_ids, groups_per_slot, max_loras):
    """The bucket of each pair, counted with ``torch.bincount`` (exact)."""
    num_buckets = groups_per_slot * max_loras + 1
    if group_ids is None:
        slots = token_slots.long()
        live = (slots >= 0) & (slots < max_loras)
        buckets = slots
    else:
        width = group_ids.shape[1]
        slots = token_slots.long().repeat_interleave(width)
        groups = group_ids.reshape(-1).long()
        live = (slots >= 0) & (slots < max_loras) & (groups >= 0)
        if groups_per_slot == 1:
            buckets = slots
        else:
            live = live & (groups < groups_per_slot)
            buckets = slots * groups_per_slot + groups
    buckets = torch.where(live, buckets, num_buckets - 1)
    return torch.bincount(buckets, minlength=num_buckets).to(torch.int32)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (route, num_tokens, width, groups_per_slot, max_loras, groups drawn from [0, num_groups))
    cases = [
        ("dense", 16384, None, 1, 1, None),
        ("dense", 16385, None, 1, 4, None),
        ("dense", 100000, None, 1, 8, None),
        ("dense", 32768, None, 1, 16, None),
        ("moe per expert", 2048, 8, 64, 4, 128),
        ("moe per expert", 8193, 2, 8, 8, 8),
        ("moe per expert", 16384, 8, 32, 8, 32),
        ("moe shared outer", 3000, 6, 1, 8, 64),
        ("pairs (heads)", 1024, 1, 32, 4, 32),
        ("dense, one atomic per pair", 1000, None, 1, 8, None),
        ("moe, one atomic per pair", 4096, 8, 128, 4, 128),
    ]
    for name, num_tokens, width, gps, max_loras, num_groups in cases:
        slots = torch.randint(-1, max_loras + 1, (num_tokens,), generator=gen, dtype=torch.int32)
        if name.startswith("pairs"):
            # one pair per (token, head): the token's slot, repeated, and the head's index
            slots = slots.repeat_interleave(gps)
            groups = torch.arange(gps, dtype=torch.int32).repeat(num_tokens).reshape(-1, 1)
        elif width is None:
            groups = None
        else:
            groups = torch.randint(-1, num_groups, (num_tokens, width), generator=gen, dtype=torch.int32)
        slots = slots.to(DEVICE)
        groups = None if groups is None else groups.to(DEVICE)
        counts = route_histogram(slots, groups, groups_per_slot=gps, max_loras=max_loras)
        ref = reference(slots, groups, gps, max_loras)
        num_pairs = slots.numel() if groups is None else groups.numel()
        bins = _histogram_bins(gps * max_loras + 1, num_pairs)
        assert torch.equal(counts, ref)
        assert int(counts.sum()) == num_pairs
        print(f"{name}: pairs {num_pairs}, buckets {gps * max_loras + 1}, BINS {bins}, "
              f"invalid {int(counts[-1])}: exact")
    print("route_histogram matches the reference exactly")
