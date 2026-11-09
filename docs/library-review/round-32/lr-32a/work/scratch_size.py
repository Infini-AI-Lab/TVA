"""lr-32a: which kernels does Triton 3.8.0 give a global (or profile) scratch buffer on the H200?

Library issue L146 says that within Compiler.Faithful's scope 5 (no `tl.make_tensor_descriptor` visited)
the global scratch size is 0, so the launcher calls no allocator. This compiles kernels that use the
features most likely to need a workspace (host descriptors in a pipelined and a warp-specialised loop,
descriptor gather and scatter, block pointers, warp-specialised pointer loops with tl.dot, a gather,
histogram, sort, scan, atomics, tl.dot_scaled, a 3-D dot, inline asm) and prints each kernel's
`global_scratch_size` and `profile_scratch_size`, with the null allocator in place (a kernel that
needs scratch would fail to launch). The control builds a descriptor in the kernel.
Run: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I scratch_size.py
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEV = "cuda"


@triton.jit
def k_desc_loop(a_desc, b_desc, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                WS: tl.constexpr):
    acc = tl.zeros([BM, BN], tl.float32)
    for k in tl.range(0, K, BK, num_stages=3, warp_specialize=WS):
        a = a_desc.load([0, k])
        b = b_desc.load([k, 0])
        acc = tl.dot(a, b, acc)
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)


@triton.jit
def k_ptr_loop(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, WS: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], tl.float32)
    for k in tl.range(0, K, BK, num_stages=3, warp_specialize=WS):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * BN + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)


@triton.jit
def k_block_ptr(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros([BM, BN], tl.float32)
    pa = tl.make_block_ptr(a_ptr, (BM, K), (K, 1), (0, 0), (BM, BK), (1, 0))
    pb = tl.make_block_ptr(b_ptr, (K, BN), (BN, 1), (0, 0), (BK, BN), (1, 0))
    for k in tl.range(0, K, BK, num_stages=3):
        acc = tl.dot(tl.load(pa), tl.load(pb), acc)
        pa = tl.advance(pa, (0, BK))
        pb = tl.advance(pb, (BK, 0))
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)


@triton.jit
def k_desc_gather_scatter(src_desc, dst_desc, idx_ptr, R: tl.constexpr, C: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, R))
    x = src_desc.gather(idx, 0)
    dst_desc.scatter(x, idx, 0)


@triton.jit
def k_misc(x_ptr, i_ptr, o_ptr, h_ptr, n, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    i = tl.load(i_ptr + r)
    g = tl.gather(x, i % N, 0)
    s = tl.sort(x)
    c = tl.cumsum(x, 0)
    h = tl.histogram(i % 16, 16)
    tl.store(h_ptr + tl.arange(0, 16), h)
    tl.atomic_add(o_ptr + r, g + s + c)
    tl.atomic_cas(i_ptr + r, i, i + 1)
    y = tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(x_ptr + r, y, mask=r < n)


@triton.jit
def k_dot_scaled(a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    sa = tl.load(as_ptr + rm[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    sb = tl.load(bs_ptr + rn[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    c = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def k_dot3d(a_ptr, b_ptr, c_ptr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rb = tl.arange(0, B)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rb[:, None, None] * M * K + rm[None, :, None] * K + rk[None, None, :])
    b = tl.load(b_ptr + rb[:, None, None] * K * N + rk[None, :, None] * N + rn[None, None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + rb[:, None, None] * M * N + rm[None, :, None] * N + rn[None, None, :], c)


@triton.jit
def k_make_desc(a_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr):
    d = tl.make_tensor_descriptor(a_ptr, [BM, BN], [BN, 1], [BM, BN])
    x = d.load([0, 0])
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], x)


def show(name, h):
    md = h.metadata
    print(f"{name:40s} global_scratch_size={md.global_scratch_size} profile_scratch_size={md.profile_scratch_size}")


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    BM, BN, BK, K = 64, 64, 32, 256
    a = torch.randn(BM, K, device=DEV, dtype=torch.float16)
    b = torch.randn(K, BN, device=DEV, dtype=torch.float16)
    c = torch.empty(BM, BN, device=DEV, dtype=torch.float32)
    ad = TensorDescriptor(a, shape=[BM, K], strides=[K, 1], block_shape=[BM, BK])
    bd = TensorDescriptor(b, shape=[K, BN], strides=[BN, 1], block_shape=[BK, BN])
    for ws in (False, True):
        try:
            show(f"host descriptors, pipelined, WS={ws}", k_desc_loop[(1,)](ad, bd, c, K, BM, BN, BK, ws, num_warps=4))
        except Exception as e:  # noqa: BLE001
            print(f"host descriptors, WS={ws}: {type(e).__name__}: {str(e)[:120]}")
        try:
            show(f"pointer loads + dot, pipelined, WS={ws}", k_ptr_loop[(1,)](a, b, c, K, BM, BN, BK, ws, num_warps=4))
        except Exception as e:  # noqa: BLE001
            print(f"pointer loads, WS={ws}: {type(e).__name__}: {str(e)[:120]}")
    show("block pointers, pipelined", k_block_ptr[(1,)](a, b, c, K, BM, BN, BK, num_warps=4))
    R, C = 8, 64
    src = torch.randn(64, C, device=DEV, dtype=torch.float16)
    dst = torch.zeros(64, C, device=DEV, dtype=torch.float16)
    sd = TensorDescriptor(src, shape=[64, C], strides=[C, 1], block_shape=[1, C])
    dd = TensorDescriptor(dst, shape=[64, C], strides=[C, 1], block_shape=[1, C])
    idx = torch.arange(R, device=DEV, dtype=torch.int32) * 3
    try:
        show("host descriptor gather/scatter", k_desc_gather_scatter[(1,)](sd, dd, idx, R, C, num_warps=4))
    except Exception as e:  # noqa: BLE001
        print(f"host descriptor gather/scatter: {type(e).__name__}: {str(e)[:160]}")
    N = 128
    x = torch.randn(N, device=DEV)
    i = torch.randint(0, 1000, (N,), device=DEV, dtype=torch.int32)
    o = torch.zeros(N, device=DEV)
    hh = torch.zeros(16, device=DEV, dtype=torch.int32)
    show("gather, sort, cumsum, histogram, atomics, asm", k_misc[(1,)](x, i, o, hh, N, N, num_warps=4))
    M2, N2, K2 = 64, 64, 64
    a8 = torch.randn(M2, K2, device=DEV).to(torch.float8_e4m3fn)
    b8 = torch.randn(K2, N2, device=DEV).to(torch.float8_e4m3fn)
    sa = torch.full((M2, K2 // 32), 127, device=DEV, dtype=torch.uint8)
    sb = torch.full((N2, K2 // 32), 127, device=DEV, dtype=torch.uint8)
    c2 = torch.empty(M2, N2, device=DEV)
    show("tl.dot_scaled e4m3", k_dot_scaled[(1,)](a8, b8, sa, sb, c2, M2, N2, K2, num_warps=4))
    a3 = torch.randn(2, 16, 16, device=DEV, dtype=torch.float16)
    b3 = torch.randn(2, 16, 16, device=DEV, dtype=torch.float16)
    c3 = torch.empty(2, 16, 16, device=DEV)
    show("3-D tl.dot", k_dot3d[(1,)](a3, b3, c3, 2, 16, 16, 16, num_warps=4))
    try:
        show("CONTROL tl.make_tensor_descriptor", k_make_desc[(1,)](a, c, BM, BN, num_warps=4))
    except Exception as e:  # noqa: BLE001
        print(f"CONTROL tl.make_tensor_descriptor with the null allocator: {type(e).__name__}: {str(e)[:120]}")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
