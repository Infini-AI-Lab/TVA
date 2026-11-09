"""lr-34b: does an addition after `/` or a math function fuse with the function's own last multiply?
For each `f(a, b) + c` in float32, list the PTX and the SASS arithmetic (L155 survey)."""
import os
import re
import subprocess

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")


@triton.jit
def k(a_ptr, b_ptr, c_ptr, o_ptr, FN: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    a = tl.load(a_ptr + j)
    b = tl.load(b_ptr + j)
    c = tl.load(c_ptr + j)
    if FN == 0:
        v = a / b
    elif FN == 1:
        v = tl.log(a)
    elif FN == 2:
        v = tl.sqrt(a)
    elif FN == 3:
        v = tl.rsqrt(a)
    elif FN == 4:
        v = tl.exp(a)
    elif FN == 5:
        v = tl.sigmoid(a)
    elif FN == 6:
        v = tl.erf(a)
    elif FN == 7:
        v = tl.log2(a)
    elif FN == 8:
        v = tl.sin(a)
    elif FN == 9:
        v = tl.exp2(a)
    elif FN == 10:
        v = tl.fdiv(a, b)
    elif FN == 11:
        v = tl.div_rn(a, b)
    elif FN == 12:
        v = tl.sqrt_rn(a)
    elif FN == 13:
        v = tl.cos(a)
    else:
        v = a
    tl.store(o_ptr + j, v + c)


NAMES = ["a / b", "tl.log", "tl.sqrt", "tl.rsqrt", "tl.exp", "tl.sigmoid", "tl.erf", "tl.log2", "tl.sin", "tl.exp2",
         "tl.fdiv", "tl.div_rn", "tl.sqrt_rn", "tl.cos"]

if __name__ == "__main__":
    N = 1
    x = torch.ones(N, device="cuda")
    for fn, name in enumerate(NAMES):
        h = k[(1,)](x, x, x, torch.empty_like(x), fn, N, num_warps=1)
        ptx = h.asm["ptx"].split(".entry", 1)[1]
        pops = [m for m in re.findall(r"\b((?:add|sub|fma|mul|div|rcp|sqrt|rsqrt|lg2|ex2|sin|cos|tanh)\.[a-z0-9.]+)", ptx)
                if "s64" not in m and "u32" not in m and "s32" not in m]
        p = os.path.join(HERE, "_k.cubin")
        open(p, "wb").write(h.asm["cubin"])
        s = subprocess.run([TOOL, "-sass", p], capture_output=True, text=True, timeout=60).stdout
        os.remove(p)
        sops = [re.sub(r"\s+", " ", re.sub(r"/\*.*?\*/", "", ln)).strip() for ln in s.splitlines()
                if re.search(r"\b(FADD|FFMA|FMUL|MUFU)", ln)]
        print(f"## {name} + c\n  PTX: {pops}\n  SASS (last 4): {sops[-4:]}", flush=True)
