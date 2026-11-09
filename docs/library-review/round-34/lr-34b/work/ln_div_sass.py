"""lr-34b: in the layer-norm pattern (Triton tutorial 05's forward body, written here), does ptxas
fuse the multiply of `tl.sum(...) / N` (div.full.f32) into `x - mean` and `var + eps`?  Prints the
SASS around each MUFU.RCP (new issue, round 34)."""
import os
import re
import subprocess

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")


@triton.jit
def ln(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


M, N = 4, 1024
x = torch.randn(M, N, device="cuda", dtype=torch.float16)
w = torch.ones(N, device="cuda", dtype=torch.float16)
y = torch.empty_like(x)
mean = torch.empty(M, device="cuda")
rstd = torch.empty(M, device="cuda")
h = ln[(M,)](x, y, w, w, mean, rstd, N, N, 1e-5, BLOCK_SIZE=1024, num_warps=8)
p = os.path.join(HERE, "_k.cubin")
open(p, "wb").write(h.asm["cubin"])
s = subprocess.run([TOOL, "-sass", p], capture_output=True, text=True, timeout=60).stdout
os.remove(p)
lines = [re.sub(r"\s+", " ", re.sub(r"/\*.*?\*/", "", ln_)).strip() for ln_ in s.splitlines()]
lines = [l_ for l_ in lines if l_ and not l_.startswith(".") and "{" not in l_]
for i, l_ in enumerate(lines):
    if "MUFU.RCP" in l_:
        print("----")
        for l2 in lines[i:i + 40]:
            if re.search(r"FFMA|FADD|FMUL|MUFU|STG|FSEL", l2):
                print("  ", l2)
