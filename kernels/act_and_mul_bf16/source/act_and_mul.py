"""
MoE activation and multiply (SiLU or GELU of the gate, times the up projection)
================================================================================

The MoE expert GEMM ``gate_up`` writes, for each routed row (a token's copy for one of its top-k
experts), the gate projection and the up projection side by side. This kernel computes
``act(gate) * up`` for every row whose expert is local, and leaves the other rows unwritten. One
program per row; each program walks the row's ``hidden_size // 2`` columns in blocks of
``BLOCK_SIZE``.

Layouts (contiguous):

* ``gateup_output``: ``[rows, hidden_size]``, ``bfloat16`` (``hidden_size = 2 * I``, ``I`` the
  expert's intermediate size on this rank): columns ``[0, I)`` hold the gate, ``[I, 2I)`` the up
  projection;
* ``topk_ids``: ``[num_tokens, topk]``, ``int32``; row ``r`` of ``gateup_output`` belongs to
  expert ``topk_ids.view(-1)[r]``, ``-1`` for an expert not on this rank (expert parallelism);
* ``expert_ids`` (the sorted, TMA layout): ``[num_blocks]``, ``int32``, one expert per block of
  ``BLOCK_SIZE_M`` rows;
* ``down_input`` (the output): ``[rows, I]``, ``bfloat16``.

``act`` is SiLU (``x * sigmoid(x)``) or GELU (tanh form, with ``tanh`` built from
``tl.sigmoid``), computed in ``float32`` and rounded to the input dtype before the product. With
``swiglu_limit = L``, the gate is first clamped to ``(-inf, L]`` and the up projection to
``[-L, L]``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py``:

* ``tanh``: line 1114;
* ``_apply_activation``: line 1119;
* ``act_and_mul_kernel``: line 1141;
* the host ``act_and_mul`` is adapted from ``act_and_mul_triton``: line 1192.

Changes from the source:

* ``tanh`` (``fused_moe_triton_kernels.py:1114``), ``_apply_activation`` (``:1119``) and
  ``act_and_mul_kernel`` (``:1141``) are copied verbatim, except for the kernel's parameter
  defaults (below), with their ``@triton.jit`` decorators (the source has no autotuning or
  heuristics on them).
* The launched kernel's parameter defaults are removed, in its signature only
  (``fused_moe_triton_kernels.py:1149-1151``): ``SWIGLU_LIMIT: tl.constexpr = 0.0``,
  ``HAS_SWIGLU_LIMIT: tl.constexpr = False`` and ``HAS_EXPERT_FILTER: tl.constexpr = True``
  become ``tl.constexpr`` parameters without defaults. TVA's subset has no kernel parameter
  defaults; the host passes each parameter (as the wrapper already does).
* The host ``act_and_mul`` is adapted from ``act_and_mul_triton``
  (``fused_moe_triton_kernels.py:1192``): it allocates the output ``down_input`` fresh, with the
  shape and dtype the caller ``fused_moe.py`` gives it
  (``python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:713``:
  ``torch.empty((total_tokens, N // 2), dtype=hidden_states.dtype)``), instead of taking it as an
  argument, and returns it. It passes the launch options ``num_warps=4, num_stages=3``
  explicitly: the wrapper passes none, so these are Triton 3.8.0's defaults, which its launches
  get. The grid, the expert-filter dispatch, ``BLOCK_SIZE=512`` and the other meta-parameters are
  the wrapper's.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input
    """
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f"Unsupported activation: {ACTIVATION_TYPE}")


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    HAS_SWIGLU_LIMIT: tl.constexpr,
    HAS_EXPERT_FILTER: tl.constexpr,
):
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.
    """
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    if HAS_EXPERT_FILTER:
        expert_id = tl.load(expert_ids_ptr + pid // expert_step)
        if expert_id == -1:
            return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)

        if HAS_SWIGLU_LIMIT:
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)

        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)
        gate_output_activated = gate_output_activated.to(InDtype)

        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def act_and_mul(
    gateup_output,
    config,
    topk_ids=None,
    expert_ids=None,
    down_moe_use_tma=False,
    activation="silu",
    swiglu_limit=None,
):
    """``act(gate) * up`` for each row of ``gateup_output`` whose expert is not ``-1``.

    Adapted from ``act_and_mul_triton`` (``fused_moe_triton_kernels.py:1192``); the output is
    allocated here, as ``fused_moe.py:713`` allocates it, and returned. Rows whose expert is
    ``-1`` are not written.
    """
    down_input = torch.empty(
        (gateup_output.shape[0], gateup_output.shape[1] // 2),
        device=gateup_output.device,
        dtype=gateup_output.dtype,
    )
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    has_expert_filter = topk_ids is not None or expert_ids is not None
    if has_expert_filter:
        expert_ids_row = topk_ids.view(-1) if not down_moe_use_tma else expert_ids
        expert_step = 1 if not down_moe_use_tma else config["BLOCK_SIZE_M"]
    else:
        expert_ids_row = None
        expert_step = 1
    has_swiglu_limit = swiglu_limit is not None
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        expert_ids_row,
        expert_step,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
        SWIGLU_LIMIT=float(swiglu_limit) if has_swiglu_limit else 0.0,
        HAS_SWIGLU_LIMIT=has_swiglu_limit,
        HAS_EXPERT_FILTER=has_expert_filter,
        num_warps=4,
        num_stages=3,
    )
    return down_input


def reference(gateup_output, activation="silu", swiglu_limit=None):
    """``act(gate)`` in ``float64``, rounded to ``bfloat16``, times ``up``, in ``float64``."""
    half = gateup_output.shape[1] // 2
    gate = gateup_output[:, :half]
    up = gateup_output[:, half:]
    if swiglu_limit is not None:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    g = gate.double()
    if activation == "silu":
        act = g * torch.sigmoid(g)
    else:
        act = 0.5 * g * (1 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
    act = act.to(gateup_output.dtype).double()
    return act * up.double()


def live_rows(rows, topk_ids, expert_ids, block_size_m, down_moe_use_tma):
    """The rows the kernel writes: those whose expert is not ``-1``."""
    if topk_ids is None and expert_ids is None:
        return torch.ones(rows, dtype=torch.bool, device=DEVICE)
    if not down_moe_use_tma:
        return topk_ids.view(-1) != -1
    return (expert_ids != -1).repeat_interleave(block_size_m)[:rows]


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (num_tokens, topk, I, experts, local experts, activation, swiglu_limit, TMA layout)
    cases = [
        (1, 8, 256, 256, 128, "silu", 10.0, False),
        (7, 8, 2048, 256, 32, "silu", 10.0, False),
        (33, 6, 1408, 64, 16, "silu", 10.0, False),
        (512, 8, 768, 128, 128, "silu", 10.0, False),
        (8192, 8, 256, 256, 32, "silu", 10.0, False),
        (300, 8, 1536, 128, 16, "silu", 10.0, True),
        (100, 8, 640, 64, 64, "silu", 10.0, None),
        (100, 8, 640, 64, 8, "silu", None, False),
        (100, 8, 640, 64, 8, "gelu", None, False),
    ]
    for num_tokens, topk, inter, experts, local, activation, limit, tma in cases:
        block_size_m = 64
        topk_ids = expert_ids = None
        if tma is None:
            rows = num_tokens * topk
        elif not tma:
            rows = num_tokens * topk
            ids = torch.randint(0, experts, (num_tokens, topk), generator=gen, dtype=torch.int32)
            topk_ids = torch.where(ids < local, ids, -1).to(torch.int32).to(DEVICE)
        else:
            num_blocks = triton.cdiv(num_tokens * topk, block_size_m) + min(num_tokens * topk, experts + 1)
            rows = num_blocks * block_size_m
            ids = torch.randint(0, experts, (num_blocks,), generator=gen, dtype=torch.int32)
            expert_ids = torch.where(ids < local, ids, -1).to(torch.int32).to(DEVICE)
        x = (torch.randn((rows, 2 * inter), generator=gen) * 6).to(device=DEVICE, dtype=torch.bfloat16)
        out = act_and_mul(x, {"BLOCK_SIZE_M": block_size_m}, topk_ids, expert_ids, bool(tma), activation, limit)
        ref = reference(x, activation, limit)
        live = live_rows(rows, topk_ids, expert_ids, block_size_m, bool(tma))
        o, r = out[live].double(), ref[live]
        err = (o - r).abs()
        rel = (err / r.abs().clamp_min(1e-3)).max().item() if live.any() else 0.0
        print(f"tokens {num_tokens} topk {topk} I {inter} {activation} limit {limit} "
              f"layout {'tma' if tma else ('none' if tma is None else 'topk')}: rows {rows}, live {int(live.sum())}, "
              f"max abs error {err.max().item() if live.any() else 0.0:.2e}, max rel error {rel:.2e}")
        assert torch.allclose(o, r, rtol=1e-2, atol=1e-5)
    print("act_and_mul matches the float64 reference")
