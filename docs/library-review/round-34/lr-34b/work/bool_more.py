"""lr-34b (L158): more `torch.bool` stores: float32 and int64 values into a bool tensor, and a bool
loaded with a mask whose `other` is a float. Prints the stored bytes and the TTIR conversion ops."""
import re

import torch
import triton
import triton.language as tl


@triton.jit
def kst(v_ptr, o_ptr, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(o_ptr + j, tl.load(v_ptr + j))


@triton.jit
def kld(b_ptr, o_ptr, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(o_ptr + j, tl.load(b_ptr + j, mask=j < 2, other=0.25).to(tl.int32))


print(f"Triton {triton.__version__}, {torch.cuda.get_device_name()}")
N = 8
for vals, dt in (([0.0, 0.5, -0.5, 1.5, 2.0, 256.0, -1.0, 255.0], torch.float32),
                 ([0, 1, 2 ** 32, 2 ** 32 + 1, 256, -256, 2 ** 40 + 3, -1], torch.int64)):
    v = torch.tensor(vals, dtype=dt, device="cuda")
    o = torch.zeros(N, dtype=torch.bool, device="cuda")
    h = kst[(1,)](v, o, N)
    ops = sorted(set(re.findall(r"arith\.\w+", h.asm["ttir"])))
    print(f"store {dt} {vals} -> bytes {o.view(torch.uint8).tolist()}  TTIR ops {ops}")
b = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.uint8, device="cuda").view(torch.bool)
o = torch.zeros(N, dtype=torch.int32, device="cuda")
kld[(1,)](b, o, N)
print(f"load bool bytes [0,1,0,...] mask j<2 other=0.25 -> {o.tolist()}")
