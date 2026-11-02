"""
Stream-K matrix multiplication
==============================

``C = A @ B`` with ``A`` (``M x K``) and ``B`` (``K x N``) in ``float16`` and ``C`` in ``float32``.

A data-parallel GEMM gives each program whole output tiles, so when the number of tiles is not a
multiple of the number of SMs, the last wave leaves SMs idle. Stream-K instead splits the total
work, ``num_tiles * iters_per_tile`` iterations of the K loop, into equal contiguous ranges, one per
program. A program's range may start or end in the middle of a tile. For each tile its range
touches, the program computes the partial product over the K iterations it owns:

* if it owns all of the tile's K iterations, it stores the tile;
* otherwise the tile is split across programs, and each of them adds its partial product into
  ``C`` with ``tl.atomic_add``. ``C`` starts at zero.

So every element of ``C`` is written by one store, or accumulated by atomic adds, never both.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def streamk_matmul_kernel(
    a_ptr, b_ptr, c_ptr,  #
    M, N, K,  #
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,  #
    iters_per_program, total_iters,  #
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    it = pid * iters_per_program
    end = tl.minimum(it + iters_per_program, total_iters)
    while it < end:
        tile = it // iters_per_tile
        # This program's K iterations of the tile: k_first ... k_last - 1.
        k_first = it - tile * iters_per_tile
        k_last = tl.minimum(iters_per_tile, k_first + (end - it))
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(k_first, k_last):
            offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if (k_first == 0) & (k_last == iters_per_tile):
            tl.store(c_ptrs, acc, mask=c_mask)
        else:
            tl.atomic_add(c_ptrs, acc, mask=c_mask)
        it += k_last - k_first


BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64


def streamk_matmul(a, b, num_programs=None):
    assert a.dtype == b.dtype == torch.float16
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    assert M > 0 and N > 0 and K > 0
    if num_programs is None:
        num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    total_iters = num_tiles * triton.cdiv(K, BLOCK_K)
    iters_per_program = triton.cdiv(total_iters, num_programs)
    grid = (num_programs, )
    streamk_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        iters_per_program, total_iters,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        num_warps=8, num_stages=3,  #
    )
    return c


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    for M, N, K, num_programs in [
        (512, 512, 512, None),
        (1000, 1000, 1000, None),
        (3072, 4096, 1024, None),
        (257, 129, 4097, 7),
        (64, 64, 64, 3),
        (8192, 8192, 4096, None),
    ]:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        c = streamk_matmul(a, b, num_programs)
        ref = a.double() @ b.double()
        err = ((c.double() - ref).abs() / (a.double().abs() @ b.double().abs())).max().item()
        print(f"M={M} N={N} K={K} programs={num_programs}: max error relative to |A| @ |B|: {err:.2e}")
        assert err < 1e-5, err
    print("✅ stream-K matmul matches the float64 reference")
