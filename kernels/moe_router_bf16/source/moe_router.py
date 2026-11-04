"""
MoE router (CUDA cores)
=======================

The router of a mixture-of-experts layer, for one token per program (grid ``(bs,)``): the
token's hidden state times each expert's gate weights, summed in ``float32`` (no ``tl.dot``),
the logits optionally soft-capped (``c * tanh(logit / c)``, written with ``exp``) and shifted by
a correction bias, then the ``topk`` experts by repeated ``argmax``, each with its softmax
probability over all the experts (``exp(logit - max) / sum exp(logit - max)``; no
renormalisation over the chosen ones).

Layouts (all contiguous):

* ``x`` (``input_ptr``): ``[bs, hidden_dim]``, ``bfloat16``;
* ``router_weight``: ``[num_experts, hidden_dim]``, ``float32`` (the gate's weight);
* ``correction_bias``: ``[num_experts]``, ``float32``, or ``None``;
* ``topk_weights``: ``[bs, topk]``, ``float32``, and ``topk_ids``: ``[bs, topk]``, ``int32``,
  allocated by the host.

``num_experts`` is a power of two (``tl.arange(0, num_experts)``), and ``BLOCK_SIZE`` the power
of two at or above ``hidden_dim``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/moe/router.py``:

* ``fused_moe_router_cudacore_kernel``: lines 12-113, copied verbatim;
* ``fused_moe_router_cudacore`` (the host wrapper): lines 116-155, copied with the changes below.

Changes from the source:

* ``router.py:1``: ``from typing import Optional`` is removed, and the wrapper's
  ``Optional[torch.Tensor]`` (``router.py:121``) is written ``torch.Tensor | None``.
* ``router.py:7-9``: ``from sglang.srt.utils import is_hip`` and the module-level
  ``_is_hip = is_hip()`` are replaced by a local of the host, ``_is_hip = False``, its value on
  an H200 (so ``max_warps = 32``): the file has no module-level assignments.
* No autotuning in the source: the wrapper's own choice (``router.py:132-138``) is kept,
  ``BLOCK_SIZE = next_power_of_2(hidden_dim)`` and
  ``num_warps = max(min(next_power_of_2(cdiv(hidden_dim, 256)), 32), 4)``.
* The tensor-core kernel (``fused_moe_router_tensorcore_kernel``, ``router.py:158``) and the
  dispatcher ``fused_moe_router_shim`` (``router.py:338-387``), which sends a call here when
  ``not ((bs >= 512 or num_experts > 8) and hidden_dim % BLOCK_SIZE_K == 0 and not
  enable_deterministic_inference)``, are not copied.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def fused_moe_router_cudacore_kernel(
    input_ptr,  # input (bs, hidden_dim)
    moe_router_weight_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    correction_bias_ptr,
    is_correction_bias: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # moe_router_weight is k major
    expert_offsets = tl.arange(0, num_experts)[:, None]
    router_mask = mask[None, :]
    w_router = tl.load(
        moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :],
        mask=router_mask,
        other=0.0,
    )

    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)

    # todo: tl.dot?
    logits = tl.sum((w_router.to(tl.float32) * x[None, :].to(tl.float32)), axis=-1)

    # logit softcap
    if moe_softcapping == 0:
        logits_softcapped = logits
    else:
        logits_scaled = logits / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        top = exped - 1
        bottom = exped + 1
        logits_softcapped = top / bottom * moe_softcapping

    # Add bias after softcapping
    if is_correction_bias:
        bias = tl.load(correction_bias_ptr + tl.arange(0, num_experts))
        logits_softcapped = logits_softcapped + bias

    # topk
    # assert 1 <= topk <= num_experts

    # 5.38 us

    top1 = tl.argmax(logits_softcapped, axis=0)
    tl.store(topk_ids_ptr + pid * topk + 0, top1)  # 5.63 us

    top1_v = tl.max(logits_softcapped, axis=0)
    invsumexp = 1.0 / tl.sum(tl.exp(logits_softcapped - top1_v), axis=0)

    tl.store(
        topk_weights_ptr + pid * topk + 0,
        invsumexp,
    )  # 5.73 us

    if topk >= 2:
        top2 = tl.argmax(
            tl.where(
                tl.arange(0, num_experts) != top1, logits_softcapped, float("-inf")
            ),
            axis=0,
        )
        tl.store(topk_ids_ptr + pid * topk + 1, top2)
        top2_v = tl.sum(logits_softcapped * (tl.arange(0, num_experts) == top2), axis=0)
        tl.store(
            topk_weights_ptr + pid * topk + 1,
            tl.exp(top2_v - top1_v) * invsumexp,
        )  # 5.95us

    # probably slow
    if topk > 2:
        topk_mask = tl.full(logits_softcapped.shape, 1.0, dtype=logits_softcapped.dtype)
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top1, topk_mask, float("-inf")
        )
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top2, topk_mask, float("-inf")
        )
        for i in range(2, topk):
            topi = tl.argmax(logits_softcapped + topk_mask, axis=0)
            topk_mask = tl.where(
                tl.arange(0, num_experts) != topi, topk_mask, float("-inf")
            )
            tl.store(topk_ids_ptr + pid * topk + i, topi)
            topi_v = tl.sum(
                logits_softcapped * (tl.arange(0, num_experts) == topi), axis=0
            )
            tl.store(
                topk_weights_ptr + pid * topk + i,
                tl.exp(topi_v - top1_v) * invsumexp,
            )
    # assert not moe_renormalize, "moe weight renormalization not implemented"


def fused_moe_router_cudacore(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: torch.Tensor | None = None,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    # router_logits = torch.empty((bs, num_experts), dtype=torch.float32, device=x.device)
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    is_correction_bias = correction_bias is not None

    _is_hip = False  # sglang.srt.utils.is_hip() on an H200 (router.py:9)
    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_moe_router_cudacore_kernel[(bs,)](
        x,
        router_weight,
        topk_weights,
        topk_ids,
        correction_bias,
        is_correction_bias=is_correction_bias,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        hidden_dim=hidden_dim,
        **config,
    )

    return topk_weights, topk_ids


def reference(x, router_weight, topk, moe_softcapping, correction_bias):
    """The router in float64: the (capped, biased) logits, and their softmax."""
    logits = x.double() @ router_weight.double().T
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    if correction_bias is not None:
        logits = logits + correction_bias.double()
    return logits, torch.softmax(logits, dim=-1)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # Grok-1 (hidden 6144) and Grok-2 (hidden 8192): 8 experts, top-2, soft cap 30, no bias,
    # python/sglang/srt/models/grok.py:139-152; the cudacore path takes bs < 512 there.
    # Then the grid of test/registered/cpu/test_topk.py:412-427 (the CPU router's test): 17 or
    # 128 tokens, hidden 64 or 257, 8 to 32 experts, top-1, 2 or 4, cap 0 or 30, bias or not.
    cases = [(bs, h, 8, 2, 30.0, False) for bs in (1, 7, 128, 511) for h in (6144, 8192)]
    cases += [(17, 64, 8, 1, 0.0, False), (128, 257, 16, 4, 30.0, True), (17, 257, 32, 2, 0.0, True),
              (128, 64, 8, 4, 0.0, True), (1, 4096, 8, 8, 30.0, False)]
    for bs, hidden_dim, num_experts, topk, softcap, with_bias in cases:
        x = (torch.randn((bs, hidden_dim), generator=gen) / 16).to(device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn((num_experts, hidden_dim), generator=gen) / 16
        w[:, 0] += torch.linspace(-0.03, 0.03, num_experts)
        w = w.to(DEVICE)
        bias = torch.linspace(-0.05, 0.05, num_experts).to(DEVICE) if with_bias else None
        weights, ids = fused_moe_router_cudacore(x, w, topk, softcap, bias)
        logits, probs = reference(x, w, topk, softcap, bias)
        ref_v, _ = torch.topk(logits, topk, dim=-1)
        ids_l = ids.long()
        # each chosen expert is distinct and has the rank-k logit, up to float32 rounding
        assert all(len(set(r)) == topk for r in ids.tolist())
        rank_gap = (logits.gather(1, ids_l) - ref_v).abs().max().item()
        assert rank_gap < 1e-5, rank_gap
        werr = (weights.double() - probs.gather(1, ids_l)).abs().max().item()
        assert werr < 1e-5, werr
        print(f"bs {bs} hidden {hidden_dim} experts {num_experts} top-{topk} cap {softcap} "
              f"bias {with_bias}: ids' rank gap {rank_gap:.1e}, weights max abs error {werr:.2e}")
    print("✅ fused_moe_router_cudacore matches the float64 reference")
