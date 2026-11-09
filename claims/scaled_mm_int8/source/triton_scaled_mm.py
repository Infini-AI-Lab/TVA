"""
Scaled int8 GEMM (``scaled_mm_kernel``)
=======================================

``C = (scale_a · scale_bᵀ) ⊙ (A @ B) (+ bias)``: ``A`` an ``int8`` ``[M, K]`` matrix (the quantised
activations), ``B`` an ``int8`` ``[K, N]`` matrix (the quantised weights), the products summed in an
``int32`` accumulator; ``scale_a`` ``float32``, one value per row (``[M, 1]``, per-token) or one for
all (``[1, 1]``, per-tensor); ``scale_b`` ``float32``, one value per column (``[N, 1]``,
per-channel) or one for all; ``bias`` (optional) ``[N]`` of the output's dtype; ``C`` ``bfloat16`` or
``float16`` ``[M, N]``. Each program computes one ``BLOCK_SIZE_M × BLOCK_SIZE_N`` tile of ``C``; the
host picks the tile from ``M`` and ``N``.

Provenance: vLLM v0.33.0, commit ``4ba787ec99`` (``4ba787ec9959e566d3b587a796dff07673324da5``),
``vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py``:

* ``is_weak_contiguous``: lines 13-18;
* ``scaled_mm_kernel``: lines 21-172;
* ``triton_scaled_mm`` (the host): lines 175-284.

Changes from the source:

* lines 1-10: the imports ``from vllm.triton_utils import tl, triton, use_tensor_descriptor`` and
  ``from vllm.triton_utils.allocation import set_triton_allocator`` become ``import triton`` and
  ``import triton.language as tl`` (``vllm.triton_utils`` re-exports Triton's modules); the
  module-level set ``_TD_ALLOCATOR_DEVICES`` (line 10), which only the host's tensor-descriptor
  path uses, is removed.
* lines 44-45: the defaults of the kernel's last two parameters are removed
  (``USE_TD: tl.constexpr = False`` and ``B_T: tl.constexpr = False`` become ``USE_TD: tl.constexpr``
  and ``B_T: tl.constexpr``): TVA's subset has no kernel parameter defaults; the host passes both.
  The kernel's other lines are unchanged.
* lines 188, 237-254: the host's parameter ``use_td`` is removed and ``use_td = False``:
  ``use_tensor_descriptor(None)`` (``vllm/triton_utils/tensor_descriptor.py:5-14``) is
  ``current_platform.is_xpu()`` when ``VLLM_TRITON_USE_TD`` is unset, so ``False`` on CUDA, and no
  caller of the W8A8 path passes ``use_td``; the allocator set-up for descriptors (lines 252-254)
  goes with it. ``b_t`` is still computed and passed as ``B_T`` (the kernel reads it only under
  ``USE_TD``).
* ``reference`` and the test under ``if __name__ == "__main__":`` are new (after vLLM's
  ``tests/kernels/quantization/test_triton_scaled_mm.py``).
"""

import torch

import triton
import triton.language as tl


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    USE_TD: tl.constexpr,
    B_T: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    if USE_TD:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        if B_T:
            # Checkpoint weights arrive as a transposed [N, K] view, so describe
            # the underlying N-major buffer and transpose each tile.
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[N, K],
                strides=[stride_bn, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
        else:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[K, N],
                strides=[stride_bk, 1],
                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if USE_TD:
            a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
            if B_T:
                b = tl.trans(b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]))
            else:
                b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
        else:
            masks_k = offsets_k < K
            masks_a = masks_am[:, None] & masks_k[None, :]
            a = tl.load(a_ptrs, mask=masks_a)

            masks_b = masks_k[:, None] & masks_bn[None, :]
            b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Add bias in accumulator dtype for less rounding error.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        accumulator += bias

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# input   - [M, K]
# weight - [K, N]
def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)
    else:
        tile_shape = (block_size_m, block_size_n, block_size_k)

    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # TD operand loads are off on CUDA (use_tensor_descriptor(None) is False there).
    b_t = weight.stride(1) != 1 and weight.stride(0) == 1
    use_td = False

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        USE_TD=use_td,
        B_T=b_t,
    )

    return result.to(out_dtype)


def reference(a, b, scale_a, scale_b, bias=None):
    """``scale_a · scale_bᵀ ⊙ (a @ b) + bias`` in ``float64`` (the ``int8`` products summed exactly:
    every partial sum is an integer below ``2^53``)."""
    out = a.double() @ b.double()
    out = scale_a.double().reshape(-1, 1) * out
    out = scale_b.double().reshape(1, -1) * out
    if bias is not None:
        out = out + bias.double()
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (M, N, K), from vLLM's tests/kernels/quantization/test_triton_scaled_mm.py:66-73 (those
    # whose K is a multiple of the tile's BLOCK_SIZE_K) and Llama-3-8B's layers
    # (benchmarks/kernels/weight_shapes.py:28-33) at TP1: qkv (4096 -> 6144), o (4096 -> 4096),
    # gate_up (4096 -> 28672), down (14336 -> 4096).
    cases = [
        (64, 971, 1024),
        (512, 20486, 1024),
        (1, 6144, 4096),
        (16, 4096, 4096),
        (100, 28672, 4096),
        (256, 4096, 14336),
    ]
    for M, N, K in cases:
        for out_dtype in (torch.bfloat16, torch.float16):
            for per_token, per_channel, use_bias, b_transposed in (
                (True, True, False, True),
                (False, False, True, False),
                (True, False, True, True),
                (False, True, False, False),
            ):
                a = torch.randint(-128, 128, (M, K), dtype=torch.int8, generator=gen).to(DEVICE)
                if b_transposed:
                    # the W8A8 linear's weight: the checkpoint's [N, K] transposed to a [K, N] view
                    b = torch.randint(-128, 128, (N, K), dtype=torch.int8, generator=gen).to(DEVICE).t()
                else:
                    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, generator=gen).to(DEVICE)
                scale_a = (torch.rand((M, 1) if per_token else (1, 1), generator=gen) * 1e-2).to(DEVICE)
                scale_b = (torch.rand((N, 1) if per_channel else (1, 1), generator=gen) * 1e-3).to(DEVICE)
                bias = torch.randn((N,), generator=gen).to(device=DEVICE, dtype=out_dtype) if use_bias else None
                out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
                ref = reference(a, b, scale_a, scale_b, bias)
                err = (out.double() - ref).abs()
                rel = (err / (ref.abs() + ref.abs().mean())).max().item()
                print(f"M {M} N {N} K {K} {out_dtype} per-token {per_token} per-channel {per_channel} "
                      f"bias {use_bias} b transposed {b_transposed}: max error / (|ref| + mean |ref|) {rel:.2e}")
                assert out.shape == (M, N) and out.dtype == out_dtype
                assert rel < (2 ** -7 if out_dtype == torch.bfloat16 else 2 ** -10), rel
    print("✅ scaled int8 GEMM matches the float64 reference")
