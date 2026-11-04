"""
Block-scaled fp8 GEMM (w8a8, DeepSeek-V3's format)
==================================================

``C = (A · As) @ (B · Bs)ᵀ``: an ``e4m3`` activation ``A`` with one scale per token and per group of
``block_k`` columns, times an ``e4m3`` weight ``B`` with one scale per ``block_n × block_k`` block,
accumulated in ``float32`` and written in the output dtype. Each program computes one
``BLOCK_SIZE_M × BLOCK_SIZE_N`` tile of ``C`` (programs grouped along M by ``GROUP_SIZE_M``); along K,
each ``BLOCK_SIZE_K`` slice's ``tl.dot`` is scaled by the slice's ``As`` column and ``Bs`` entry before
it is added.

Layouts:

* ``A``: ``[..., K]``, ``float8_e4m3fn`` (Triton's ``float8e4nv``), contiguous; ``M`` is the product of
  the leading dimensions;
* ``B``: ``[N, K]``, ``float8_e4m3fn``;
* ``As``: ``[..., cdiv(K, block_k)]``, ``float32`` (the leading dimensions of ``A``);
* ``Bs``: ``[cdiv(N, block_n), cdiv(K, block_k)]``, ``float32``;
* ``C``: ``[..., N]``, ``bfloat16`` (or ``float16``, ``float32``: the output dtype).

Provenance: vLLM v0.33.0, commit ``4ba787ec99``
(``4ba787ec9959e566d3b587a796dff07673324da5``), package ``vllm/``:

* ``_w8a8_triton_block_scaled_mm``: ``model_executor/layers/quantization/utils/fp8_utils.py:840-921``;
* ``_upcast_e8m0_to_fp32``: ``model_executor/layers/quantization/utils/fp8_utils.py:1157-1166``;
* ``w8a8_triton_block_scaled_mm`` (the host wrapper): ``model_executor/layers/quantization/utils/fp8_utils.py:961-1093``.

Changes from the source:

* ``fp8_utils.py:840-921``: the kernel is copied verbatim (it has no autotune or heuristics decorator).
* ``fp8_utils.py:1157-1166``: the torch helper ``_upcast_e8m0_to_fp32`` is copied verbatim.
* ``fp8_utils.py:961-1093``: the host ``w8a8_triton_block_scaled_mm`` is the wrapper, adapted:
  - ``fp8_utils.py:987-1019``: the ROCm ``gfx1250`` branch (a torch fallback that launches no kernel) is
    removed; on an H200 ``current_platform.is_rocm()`` is false and the branch is not taken;
  - ``fp8_utils.py:1044-1062``: the configuration is always the wrapper's default configuration
    (``fp8_utils.py:1052-1062``), the one it takes when ``get_w8a8_block_fp8_configs``
    (``fp8_utils.py:924-958``) finds no tuned JSON file for ``(N, K, device, block_size)`` and the one it
    takes under ``VLLM_BATCH_INVARIANT``: ``BLOCK_SIZE_M = 64``, ``BLOCK_SIZE_N = block_size[0]``,
    ``BLOCK_SIZE_K = block_size[1]``, ``GROUP_SIZE_M = 32``, ``num_warps = 4``, ``num_stages = 2``;
    the tuned-file lookup is removed;
  - ``fp8_utils.py:1064-1067``: the grid function of ``META`` is computed directly with the same value;
  - the launch's arguments and strides are those of ``fp8_utils.py:1069-1091``, unchanged.
* ``native_w8a8_block_matmul``-style reference and the test are new (after vLLM's
  ``tests/kernels/quantization/test_block_fp8.py`` and ``tests/kernels/quant_utils.py:92``).
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _w8a8_triton_block_scaled_mm(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Upcast E8M0 (exponent-only) scale to float32.

    E8M0 stores only the 8-bit biased exponent (bias=127). To convert
    to float32 we place those 8 bits into the exponent field of an
    IEEE-754 float32 (bits 23-30) with sign=0 and mantissa=0.
    """
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    fp32_bits = exp_bits << 23
    return fp32_bits.view(torch.float32)


def w8a8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """The block-scaled GEMM ``C = (A · As) @ (B · Bs)ᵀ`` in ``output_dtype`` (vLLM's wrapper, with its
    default configuration)."""
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    # Triton cannot currently bind E8M0 scale tensors directly. Checkpoints
    # with exponent-only UE8M0 scales (e.g. DeepSeek-V4) are decoded to fp32
    # before launching the kernel.
    if As.dtype == torch.float8_e8m0fnu:
        As = _upcast_e8m0_to_fp32(As).contiguous()
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = _upcast_e8m0_to_fp32(Bs).contiguous()

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Default config
    # Block-wise quant: BLOCK_SIZE_N must be divisible by block_size[0]
    # BLOCK_SIZE_K must be divisible by block_size[1]
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_size[0],
        "BLOCK_SIZE_K": block_size[1],
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }

    grid = (triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]), )

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def make_inputs(M, N, K, block_size, device, generator):
    """Inputs as vLLM's test makes them (``test_block_fp8.py:139-153``)."""
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    A = ((torch.rand(M, K, generator=generator) - 0.5) * 2 * fp8_max).clamp(min=fp8_min, max=fp8_max)
    B = ((torch.rand(N, K, generator=generator) - 0.5) * 2 * fp8_max).clamp(min=fp8_min, max=fp8_max)
    block_n, block_k = block_size
    As = torch.rand(M, triton.cdiv(K, block_k), generator=generator) * 1e-2
    Bs = torch.rand(triton.cdiv(N, block_n), triton.cdiv(K, block_k), generator=generator) * 1e-2
    return (A.to(device=device, dtype=torch.float8_e4m3fn), B.to(device=device, dtype=torch.float8_e4m3fn),
            As.to(device), Bs.to(device))


def reference(A, B, As, Bs, block_size, dtype=torch.float64):
    """The dequantised product ``(A · As) @ (B · Bs)ᵀ`` in ``dtype``."""
    block_n, block_k = block_size
    M, K = A.numel() // A.shape[-1], A.shape[-1]
    N = B.shape[0]
    a = A.reshape(M, K).to(dtype) * As.reshape(M, -1).to(dtype).repeat_interleave(block_k, dim=1)[:, :K]
    bs = Bs.to(dtype).repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)[:N, :K]
    b = B.to(dtype) * bs
    return (a @ b.t()).reshape(*A.shape[:-1], N)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    block_size = [128, 128]
    # M, N, K from vLLM's test_block_fp8.py:48-58 (M 1..4096, N 128..13824, K 256..16384, K = 3884 not a
    # multiple of 128) and DeepSeek-V3's shapes (hidden 7168; N = 4608, 2112, 24576 at TP8 and below).
    cases = [
        (1, 128, 256),
        (7, 576, 3884),
        (8, 512, 4096),
        (83, 7168, 256),
        (83, 13824, 16384),
        (100, 2112, 7168),
        (257, 4608, 7168),
        (4096, 7168, 2304),
        (4096, 4608, 7168),
    ]
    for M, N, K in cases:
        A, B, As, Bs = make_inputs(M, N, K, block_size, DEVICE, gen)
        out = w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, torch.bfloat16)
        ref_dtype = torch.float64 if M * N * K <= 2**34 else torch.float32
        ref = reference(A, B, As, Bs, block_size, ref_dtype).double()
        err = (out.double() - ref).abs()
        rel_mean = (err.mean() / ref.abs().mean()).item()
        rel_max = (err / (ref.abs() + ref.abs().mean())).max().item()
        print(f"M {M} N {N} K {K}: mean rel error {rel_mean:.2e}, max error / (|ref| + mean |ref|) {rel_max:.2e}")
        assert out.dtype == torch.bfloat16 and out.shape == (M, N)
        assert rel_mean < 3e-3, rel_mean  # bfloat16 rounding alone gives about 1.4e-3
        assert rel_max < 1e-2, rel_max
    print("✅ block-scaled fp8 GEMM matches the dequantised reference")
