"""Library review round 6 (lr-6b), survey area `semantics`: is a float64 division by a power-of-two constant
a "product" that a following addition may fuse with?

SEMANTICS.md ("Float `+` and `-`: IEEE or fusable") calls an addition fusable when a product feeds it,
"an operand computed by `*`, `tl.dot` or `tl.rand`"; `/` is `P.div`, and for float64 `Rules.Num.DivRn`
(IEEE). Triton lowers float64 `/` to `llvm.nvvm.div.rn.d`, which LLVM's NVPTX instcombine turns into a
plain `fdiv` (`primitives/llvm-NVPTXTargetTransformInfo.cpp:298-299`), and LLVM rewrites `fdiv x, C` with an
exactly invertible `C` into `fmul x, 1/C`. With `enable_fp_fusion` the backend may then contract that
product with the addition.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-6/lr-6b/work/div_fusion.py
Output: div_fusion.out.txt next to this file.

Cases, float64, one program:
  D1  `x / 4.0 + y`, default fusion: the PTX instructions, and the result for x = 11·2^-1074,
      y = 2^-1021, where the IEEE division then the IEEE addition give y + 2·2^-1073 (a tie, to even) and
      one fma gives y + 2^-1073.
  D2  the same kernel with `enable_fp_fusion=False`.
  D3  `x / y + z` with a runtime divisor (no rewrite expected), the same inputs with y = 4.0.
  D4  float32 `x / 4.0 + y` (`div.full.f32`) and `tl.div_rn(x, 4.0) + y`, to see whether the rewrite
      reaches float32.
"""
import os
import re
import struct
import tempfile
from fractions import Fraction

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr6b-div-")

import torch
import triton
import triton.language as tl

print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")


def ptx_arith(kernel):
    cache = kernel.device_caches[0][0]
    ptx = list(cache.values())[-1].asm["ptx"]
    ops = set()
    for l in ptx.splitlines():
        m = re.match(r"^\s*(fma|add|sub|mul|div|rcp)\.([a-z0-9.]+)", l)
        if m and ("f64" in m.group(2) or "f32" in m.group(2)):
            ops.add(m.group(1) + "." + m.group(2))
    return sorted(ops)


@triton.jit
def d1_kernel(x_ptr, y_ptr, o_ptr):
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    tl.store(o_ptr, x / 4.0 + y)


@triton.jit
def d3_kernel(x_ptr, d_ptr, y_ptr, o_ptr):
    x = tl.load(x_ptr)
    d = tl.load(d_ptr)
    y = tl.load(y_ptr)
    tl.store(o_ptr, x / d + y)


@triton.jit
def d4_kernel(x_ptr, y_ptr, o_ptr):
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    tl.store(o_ptr, x / 4.0 + y)
    tl.store(o_ptr + 1, tl.div_rn(x, 4.0) + y)


def f64(v):
    return struct.unpack("<d", struct.pack("<d", v))[0]


tiny = 2.0 ** -1074
x_v, y_v = 11 * tiny, 2.0 ** -1021
g = 2.0 ** -1073
ieee = x_v / 4.0 + y_v           # Python: IEEE division, then IEEE addition
exact = Fraction(x_v) / 4 + Fraction(y_v)
print(f"inputs x = 11*2^-1074, y = 2^-1021; IEEE / then IEEE +: y + {round((ieee - y_v) / g)}*2^-1073; "
      f"exact x/4 + y = y + {float((exact - Fraction(y_v)) / Fraction(g))}*2^-1073 (one fma rounds it to y + 1*2^-1073)")


def run_d1(fusion):
    x = torch.tensor([x_v], dtype=torch.float64, device="cuda")
    y = torch.tensor([y_v], dtype=torch.float64, device="cuda")
    o = torch.zeros(1, dtype=torch.float64, device="cuda")
    d1_kernel[(1,)](x, y, o, enable_fp_fusion=fusion)
    torch.cuda.synchronize()
    return ptx_arith(d1_kernel), round((o.item() - y_v) / g)


ops, k = run_d1(True)
print(f"D1 x / 4.0 + y, fusion on:  PTX {ops}; result y + {k}*2^-1073")
ops, k = run_d1(False)
print(f"D2 x / 4.0 + y, fusion off: PTX {ops}; result y + {k}*2^-1073")

x = torch.tensor([x_v], dtype=torch.float64, device="cuda")
d = torch.tensor([4.0], dtype=torch.float64, device="cuda")
y = torch.tensor([y_v], dtype=torch.float64, device="cuda")
o = torch.zeros(1, dtype=torch.float64, device="cuda")
d3_kernel[(1,)](x, d, y, o)
torch.cuda.synchronize()
print(f"D3 x / d + y (runtime d = 4.0): PTX {ptx_arith(d3_kernel)}; result y + {round((o.item() - y_v) / g)}*2^-1073")

xs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
ys = torch.tensor([1.0], dtype=torch.float32, device="cuda")
os_ = torch.zeros(2, dtype=torch.float32, device="cuda")
d4_kernel[(1,)](xs, ys, os_)
torch.cuda.synchronize()
print(f"D4 float32 x / 4.0 + y and tl.div_rn(x, 4.0) + y: PTX {ptx_arith(d4_kernel)}")
