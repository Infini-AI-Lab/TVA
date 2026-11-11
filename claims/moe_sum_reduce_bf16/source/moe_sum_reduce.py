"""
MoE sum-reduce (the combine of the experts' outputs)
====================================================

After the down GEMM, each token has ``topk`` output rows, one per expert it was routed to (already
weighted by the router's weights). This kernel sums them in ``float32``, multiplies the sum by
``routed_scaling_factor`` (a compile-time value) and stores it in the input's dtype. Each program
handles ``BLOCK_M`` tokens and ``BLOCK_DIM`` hidden columns.

Layouts (contiguous):

* ``input``: ``[token_num, topk_num, hidden_dim]``, ``bfloat16``;
* ``output``: ``[token_num, hidden_dim]``, ``bfloat16``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py``:

* ``_moe_sum_reduce_kernel``: line 1240 (decorator line 1239; the comment above it, line 1238,
  says it was modified from LightLLM's ``moe_sum_reduce.py``);
* the host ``moe_sum_reduce`` is adapted from ``moe_sum_reduce_triton``: line 1296.

Changes from the source:

* ``_moe_sum_reduce_kernel`` (``fused_moe_triton_kernels.py:1240``) is copied verbatim, except
  for its parameter defaults (below), with its ``@triton.jit`` (the source has no autotuning
  or heuristics on it).
* The launched kernel's parameter defaults are removed, in its signature only
  (``fused_moe_triton_kernels.py:1255``): ``USE_GDC: tl.constexpr = False`` becomes
  ``USE_GDC: tl.constexpr``. TVA's subset has no kernel parameter defaults; the host passes each
  parameter.
* The host ``moe_sum_reduce`` is adapted from ``moe_sum_reduce_triton``
  (``fused_moe_triton_kernels.py:1296``): it allocates ``output`` fresh
  (``torch.empty((token_num, hidden_dim))`` in the input's dtype, the shape the wrapper asserts)
  instead of taking it as an argument, and returns it. The wrapper passes
  ``USE_GDC=True, launch_pdl=True`` when ``is_arch_support_pdl()`` holds (``:1315``); this host
  passes ``USE_GDC=False`` and no ``launch_pdl`` (programmatic dependent launch is outside TVA's
  launches). It passes ``num_stages=3`` explicitly: the wrapper passes no launch ``num_stages``,
  so this is Triton 3.8.0's default, which its launches get. ``BLOCK_M = 1``,
  ``BLOCK_DIM = 2048``, ``NUM_STAGE = 1``, ``num_warps = 16``, the grid and the strides are the
  wrapper's.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce(input: torch.Tensor, routed_scaling_factor: float):
    """``output[t, :] = routed_scaling_factor * sum_k input[t, k, :]`` (sum in ``float32``).

    Adapted from ``moe_sum_reduce_triton`` (``fused_moe_triton_kernels.py:1296``); the output is
    allocated here and returned, and the launch is without programmatic dependent launch.
    """
    assert input.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    output = torch.empty((token_num, hidden_dim), dtype=input.dtype, device=input.device)
    assert output.is_contiguous()
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        USE_GDC=False,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


def reference(input, routed_scaling_factor):
    """The sum over the top-k axis and the scaling, in ``float64``."""
    return input.double().sum(dim=1) * routed_scaling_factor


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (token_num, topk_num, hidden_dim, routed_scaling_factor)
    cases = [
        (1, 8, 7168, 2.5),
        (7, 9, 4096, 0.3),
        (32, 8, 2048, 1.0),
        (33, 6, 5120, 1.0),
        (17, 8, 2880, 1.0),
        (1, 1, 1, 1.0),
        (300, 3, 100, 2.5),
        (1024, 9, 4096, 0.3),
        (8192, 8, 7168, 2.5),
    ]
    for token_num, topk, hidden, scale in cases:
        x = torch.randn((token_num, topk, hidden), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        out = moe_sum_reduce(x, scale)
        ref = reference(x, scale)
        err = (out.double() - ref).abs()
        bound = 2.0**-7 * ref.abs() + 1e-5 * topk * abs(scale)
        print(f"tokens {token_num} topk {topk} hidden {hidden} scale {scale}: max abs error {err.max().item():.2e}, "
              f"max error / bound {(err / bound).max().item():.2f}")
        assert (err <= bound).all()
    print("moe_sum_reduce matches the float64 reference")
