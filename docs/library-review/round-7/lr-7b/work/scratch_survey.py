"""Which constructs make Triton 3.8.0 ask the launcher for global scratch memory on the H200 (sm_90)?
Such memory comes from `triton.set_allocator`'s allocator, which `Compiler.Faithful`'s scope does not
mention; the device writes it during the launch. Compile-only (`warmup`), so no launch needs an allocator.

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-7/lr-7b/work/scratch_survey.py
"""
import datetime

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def plain(x_ptr, y_ptr, c_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    x = tl.load(x_ptr + o)
    tl.atomic_add(c_ptr, tl.sum(x))
    tl.store(y_ptr + o, tl.sort(x))


@triton.jit
def gather_hist(x_ptr, i_ptr, y_ptr, h_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    x = tl.load(x_ptr + o)
    i = tl.load(i_ptr + o)
    tl.store(y_ptr + o, tl.gather(x, i, 0))
    tl.store(h_ptr + o, tl.histogram(i, B))


@triton.jit
def dot_ws(a_ptr, b_ptr, c_ptr, K, B: tl.constexpr):
    r = tl.arange(0, B)
    acc = tl.zeros((B, B), dtype=tl.float32)
    for k in tl.range(0, K, B, warp_specialize=True):
        a = tl.load(a_ptr + r[:, None] * K + k + r[None, :])
        b = tl.load(b_ptr + (k + r[:, None]) * B + r[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + r[:, None] * B + r[None, :], acc)


@triton.jit
def host_desc(a_desc, c_desc, B: tl.constexpr):
    t = a_desc.load([0, 0])
    c_desc.store([0, 0], t)


@triton.jit
def device_desc(a_ptr, c_ptr, M, N, B: tl.constexpr):
    a = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1], block_shape=[B, B])
    c = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[B, B])
    c.store([0, 0], a.load([0, 0]))


def scratch(f):
    try:
        k = f()
    except Exception as e:  # a construct this configuration rejects
        return f"rejected at compile time ({type(e).__name__}: {str(e).splitlines()[0][:80]})"
    m = k.metadata
    return f"global_scratch_size {m.global_scratch_size}, profile_scratch_size {m.profile_scratch_size}"


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}, "
          f"{datetime.date.today()}")
    d = "cuda"
    x = torch.randn(64, device=d)
    i = torch.randint(0, 64, (64,), device=d, dtype=torch.int32)
    c = torch.zeros(1, device=d)
    y = torch.empty(64, device=d)
    h = torch.empty(64, device=d, dtype=torch.int32)
    print("load, sum, atomic_add, sort, store:", scratch(lambda: plain.warmup(x, y, c, B=64, grid=(1,))))
    print("gather, histogram:", scratch(lambda: gather_hist.warmup(x, i, y, h, B=64, grid=(1,))))
    a = torch.randn(64, 256, device=d, dtype=torch.float16)
    b = torch.randn(256, 64, device=d, dtype=torch.float16)
    cc = torch.empty(64, 64, device=d)
    print("tl.dot in tl.range(warp_specialize=True):", scratch(lambda: dot_ws.warmup(a, b, cc, 256, B=64, grid=(1,))))
    a2 = torch.randn(64, 64, device=d, dtype=torch.float16)
    c2 = torch.empty(64, 64, device=d, dtype=torch.float16)
    da = TensorDescriptor.from_tensor(a2, [64, 64])
    dc = TensorDescriptor.from_tensor(c2, [64, 64])
    print("host TensorDescriptor load/store:", scratch(lambda: host_desc.warmup(da, dc, B=64, grid=(1,))))
    print("tl.make_tensor_descriptor load/store:", scratch(lambda: device_desc.warmup(a2, c2, 64, 64, B=64, grid=(1,))))


if __name__ == "__main__":
    main()
