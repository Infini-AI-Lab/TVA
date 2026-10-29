"""lr-11a: which operations make Triton 3.8.0 allocate launch scratch (global_scratch_size > 0) on
the H200? Faithful's frame (L73) says a valid, well-behaved launch writes no global byte outside
`init` "except the scratch of the descriptors the kernel builds (scope 5)", and scope 5 asks only of
kernels that call tl.make_tensor_descriptor. This compiles one small kernel per operation and prints
the compiled kernel's global_scratch_size and profile_scratch_size.
Run: CUDA_VISIBLE_DEVICES=0 <venv>/bin/python -I scratch_sweep.py
"""
import torch
import triton
import triton.language as tl

triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))


@triton.jit
def k_sort(x, y):
    i = tl.arange(0, 128)
    tl.store(y + i, tl.sort(tl.load(x + i)))


@triton.jit
def k_topk(x, y):
    v = tl.topk(tl.load(x + tl.arange(0, 128)[None, :]), 8, dim=1)
    tl.store(y + tl.arange(0, 8)[None, :], v)


@triton.jit
def k_hist(x, y):
    i = tl.arange(0, 128)
    tl.store(y + tl.arange(0, 16), tl.histogram(tl.load(x + i).to(tl.int32) % 16, 16))


@triton.jit
def k_gather(x, y):
    i = tl.arange(0, 128)
    v = tl.load(x + i)
    idx = (127 - i)
    tl.store(y + i, tl.gather(v, idx, 0))


@triton.jit
def k_scan(x, y):
    i = tl.arange(0, 128)
    tl.store(y + i, tl.cumsum(tl.load(x + i), 0))


@triton.jit
def k_dot(a, b, c):
    r = tl.arange(0, 64)
    A = tl.load(a + r[:, None] * 64 + r[None, :])
    B = tl.load(b + r[:, None] * 64 + r[None, :])
    tl.store(c + r[:, None] * 64 + r[None, :], tl.dot(A, B))


@triton.jit
def k_dot_loop(a, b, c, K):
    r = tl.arange(0, 64)
    acc = tl.zeros([64, 64], tl.float32)
    for k in tl.range(0, K, 64, num_stages=3):
        A = tl.load(a + r[:, None] * K + (k + r[None, :]))
        B = tl.load(b + (k + r[:, None]) * 64 + r[None, :])
        acc = tl.dot(A, B, acc)
    tl.store(c + r[:, None] * 64 + r[None, :], acc)


@triton.jit
def k_dot_ws(a, b, c, K):
    r = tl.arange(0, 64)
    acc = tl.zeros([64, 64], tl.float32)
    for k in tl.range(0, K, 64, warp_specialize=True):
        A = tl.load(a + r[:, None] * K + (k + r[None, :]))
        B = tl.load(b + (k + r[:, None]) * 64 + r[None, :])
        acc = tl.dot(A, B, acc)
    tl.store(c + r[:, None] * 64 + r[None, :], acc)


@triton.jit
def k_blockptr(a, c, M, N):
    p = tl.make_block_ptr(a, (M, N), (N, 1), (0, 0), (64, 64), (1, 0))
    q = tl.make_block_ptr(c, (M, N), (N, 1), (0, 0), (64, 64), (1, 0))
    tl.store(q, tl.load(p, boundary_check=(0, 1)), boundary_check=(0, 1))


@triton.jit
def k_print(x):
    tl.device_print("v", tl.load(x))


@triton.jit
def k_desc(a, c, M, N):
    d = tl.make_tensor_descriptor(a, [M, N], [N, 1], [64, 64])
    e = tl.make_tensor_descriptor(c, [M, N], [N, 1], [64, 64])
    e.store([0, 0], d.load([0, 0]))


@triton.jit
def k_atomic(x, y):
    i = tl.arange(0, 128)
    tl.atomic_add(y + i % 4, tl.load(x + i))
    tl.atomic_cas(y + 4, 0.0, 1.0)


def show(name, ck):
    md = ck.metadata
    print(f"{name}: global_scratch_size {md.global_scratch_size}, profile_scratch_size {md.profile_scratch_size}")


def main():
    print(f"Triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
    f = torch.rand(128, device="cuda")
    o = torch.zeros(128, device="cuda")
    h = torch.rand(64 * 64, device="cuda", dtype=torch.float16)
    hb = torch.rand(256 * 64, device="cuda", dtype=torch.float16)
    of = torch.zeros(64 * 64, device="cuda")
    show("sort", k_sort[(1,)](f, o))
    show("topk", k_topk[(1,)](f, o))
    show("histogram", k_hist[(1,)](f * 100, torch.zeros(16, device="cuda", dtype=torch.int32)))
    show("gather", k_gather[(1,)](f, o))
    show("cumsum", k_scan[(1,)](f, o))
    show("dot fp16", k_dot[(1,)](h, h, of))
    show("dot loop num_stages=3", k_dot_loop[(1,)](hb, hb, of, 256))
    # warp_specialize with pointer loads aborts the process in the compiler
    # ("LLVM ERROR: unsupported load type for producer commit"), so it is not run here.
    show("block pointer load/store", k_blockptr[(1,)](h, torch.zeros_like(h), 64, 64))
    show("device_print", k_print[(1,)](f))
    show("make_tensor_descriptor", k_desc[(1,)](h, torch.zeros_like(h), 64, 64))
    show("atomics", k_atomic[(1,)](f, torch.zeros(8, device="cuda")))
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
