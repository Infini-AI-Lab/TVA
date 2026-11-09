"""TVA library evidence (library issue L146, round 33): a host that sets an allocator with `triton.set_allocator`
for a launch within Compiler.Faithful's scope 5 (no `tl.make_tensor_descriptor` visited).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/set-allocator/run.py
Output: evidence/set-allocator/output.txt (README.md says what it shows).

Each kernel (the in-scope features of round 32's library review, lr-32a's survey of scratch sizes, whose kernels are
copied here) is launched twice on the same inputs: first with Triton's null allocator in place (a launch that needed
global scratch would raise), then after `triton.set_allocator` installs an allocator that counts its calls and
returns zero-filled buffers. The line gives the kernel's global and profile scratch sizes, the allocator's calls,
and whether every output byte of the second launch equals the first's. The control builds a descriptor in the
kernel.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEV = "cuda"
OUT = []
CALLS = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


def counting_alloc(size, alignment, stream):
    CALLS.append((size, alignment))
    return torch.zeros(size, dtype=torch.int8, device=DEV)


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



def cases():
    BM, BN, BK, K = 64, 64, 32, 256
    g = torch.Generator(device=DEV).manual_seed(20261108)
    a = torch.randn(BM, K, device=DEV, generator=g).to(torch.float16)
    b = torch.randn(K, BN, device=DEV, generator=g).to(torch.float16)
    ad = TensorDescriptor(a, shape=[BM, K], strides=[K, 1], block_shape=[BM, BK])
    bd = TensorDescriptor(b, shape=[K, BN], strides=[BN, 1], block_shape=[BK, BN])
    N = 128
    x = torch.randn(N, device=DEV, generator=g)
    i = torch.randint(0, 1000, (N,), device=DEV, dtype=torch.int32, generator=g)
    M2, N2, K2 = 64, 64, 64
    a8 = torch.randn(M2, K2, device=DEV, generator=g).to(torch.float8_e4m3fn)
    b8 = torch.randn(K2, N2, device=DEV, generator=g).to(torch.float8_e4m3fn)
    sa = torch.full((M2, K2 // 32), 127, device=DEV, dtype=torch.uint8)
    sb = torch.full((N2, K2 // 32), 127, device=DEV, dtype=torch.uint8)
    a3 = torch.randn(2, 16, 16, device=DEV, generator=g).to(torch.float16)
    b3 = torch.randn(2, 16, 16, device=DEV, generator=g).to(torch.float16)

    def c_desc(ws):
        c = torch.zeros(BM, BN, device=DEV)
        return (lambda: k_desc_loop[(1,)](ad, bd, c, K, BM, BN, BK, ws, num_warps=4)), [c]

    def c_ptr():
        c = torch.zeros(BM, BN, device=DEV)
        return (lambda: k_ptr_loop[(1,)](a, b, c, K, BM, BN, BK, False, num_warps=4)), [c]

    def c_bp():
        c = torch.zeros(BM, BN, device=DEV)
        return (lambda: k_block_ptr[(1,)](a, b, c, K, BM, BN, BK, num_warps=4)), [c]

    def c_misc():
        xx, ii = x.clone(), i.clone()
        o = torch.zeros(N, device=DEV)
        hh = torch.zeros(16, device=DEV, dtype=torch.int32)

        def run():
            xx.copy_(x); ii.copy_(i); o.zero_(); hh.zero_()
            return k_misc[(1,)](xx, ii, o, hh, N, N, num_warps=4)
        return run, [xx, ii, o, hh]

    def c_scaled():
        c2 = torch.zeros(M2, N2, device=DEV)
        return (lambda: k_dot_scaled[(1,)](a8, b8, sa, sb, c2, M2, N2, K2, num_warps=4)), [c2]

    def c_dot3():
        c3 = torch.zeros(2, 16, 16, device=DEV)
        return (lambda: k_dot3d[(1,)](a3, b3, c3, 2, 16, 16, 16, num_warps=4)), [c3]

    def c_make():
        c = torch.zeros(BM, BN, device=DEV)
        af = torch.randn(BM, BN, device=DEV, generator=g).to(torch.float16)
        return (lambda: k_make_desc[(1,)](af, c, BM, BN, num_warps=4)), [c]

    return [("host descriptors, pipelined loop", lambda: c_desc(False)),
            ("host descriptors, warp-specialised loop", lambda: c_desc(True)),
            ("pointer loads and tl.dot, pipelined loop", c_ptr),
            ("block pointers, pipelined loop", c_bp),
            ("gather, sort, cumsum, histogram, atomics, inline asm", c_misc),
            ("tl.dot_scaled e4m3", c_scaled),
            ("3-D tl.dot", c_dot3),
            ("CONTROL tl.make_tensor_descriptor in the kernel", c_make)]


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for name, make in cases():
        run, outs = make()
        try:
            h = run()
            torch.cuda.synchronize()
            first = [t.clone() for t in outs]
            gs, ps = h.metadata.global_scratch_size, h.metadata.profile_scratch_size
            null = "ran"
        except RuntimeError as e:
            first, gs, ps = None, None, None
            null = f"{type(e).__name__}: {str(e)[:70]}"
        triton.set_allocator(counting_alloc)
        CALLS.clear()
        h = run()
        torch.cuda.synchronize()
        n = len(CALLS)
        same = None if first is None else all(torch.equal(u.view(torch.uint8), v.view(torch.uint8))
                                              for u, v in zip(first, outs))
        if gs is None:
            gs, ps = h.metadata.global_scratch_size, h.metadata.profile_scratch_size
        out(f"{name}: global scratch {gs}, profile scratch {ps}; with the null allocator: {null}; with "
            f"triton.set_allocator: {n} allocator calls {CALLS}, outputs equal to the first launch's: {same}")
        triton.set_allocator(triton.runtime._allocation.NullAllocator())
    with open(__file__.rsplit("/", 1)[0] + "/output.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
