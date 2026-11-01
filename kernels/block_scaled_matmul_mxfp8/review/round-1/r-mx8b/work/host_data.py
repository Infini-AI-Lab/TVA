"""Reviewer r-mx8b: the host's data, on the CPU (no GPU).

1. What `MXFP4Tensor(...).random()` and `MXScaleTensor(torch.rand(...) + 1e-8)` produce
   (initialize_block_scaled, lines 340-389), on CPU tensors.
2. The claim's tolerance `tol p (K / 128)` (Defs.lean `tol`, recursion over steps of 128) on
   sampled output elements with the host's distributions, at K = 512, 8192, 16384, against the
   README's table.

Run with CUDA hidden: CUDA_VISIBLE_DEVICES="" python -I host_data.py
"""
import math
import sys

import torch
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

torch.manual_seed(42)

# --- 1. element and scale distributions ---------------------------------------------------
x = MXFP4Tensor(size=(256, 4096), device="cpu").random()
xf = x.to(torch.float32)
vals, counts = torch.unique(xf, return_counts=True)
print("e2m1 values (as float32) and counts:")
print([(float(v), int(c)) for v, c in zip(vals, counts)])
# exact in e4m3?
x8 = xf.to(torch.float8_e4m3fn)
print("exact in e4m3:", bool((x8.to(torch.float32) == xf).all()))

s = torch.rand([64, 64, 32, 16]) + 1e-8
st = MXScaleTensor(s)
b = st.data
print("scale dtype:", b.dtype, "shape:", tuple(b.shape), "contiguous:", b.is_contiguous())
print("scale bytes min/max:", int(b.min()), int(b.max()))
sv = st.to(torch.float32)
# value of a byte: 2^(b - 127)?
ref = torch.pow(2.0, b.to(torch.float32) - 127.0)
print("value == 2^(b-127):", bool((sv == ref).all()))
# floor or round of log2?
fl = torch.floor(torch.log2(s)) + 127
print("byte == floor(log2(rand+1e-8)) + 127:", bool((b.to(torch.float32) == fl).all()),
      " mismatches:", int((b.to(torch.float32) != fl).sum()))
hist = torch.bincount(b.flatten().to(torch.int64), minlength=256)
print("byte histogram (nonzero):", [(i, int(hist[i])) for i in range(256) if hist[i] > 0][-8:])

# --- 2. the tolerance on sampled elements --------------------------------------------------
RHO = 130 / 2 ** 22
ALPHA = 130 / 2 ** 126
U16 = 2.0 ** -11
ETA16 = 2.0 ** -25


def tol_of(p):
    """Defs.lean `tol p (K/128)`: tol 0 = 0; tol (t+1) = tol t + rho (|partSum t| + tol t + chunkAbs t) + alpha."""
    K = len(p)
    t_ = 0.0
    part = 0.0
    parts = []
    for t in range(K // 128):
        chunk = p[128 * t:128 * t + 128]
        ca = math.fsum(abs(v) for v in chunk)
        t_ = t_ + RHO * (abs(part) + t_ + ca) + ALPHA
        part = math.fsum(p[:128 * (t + 1)])
    return t_


def sample_row(K, gen):
    xa = MXFP4Tensor(size=(1, K), device="cpu").random().to(torch.float32)[0]
    xb = MXFP4Tensor(size=(1, K), device="cpu").random().to(torch.float32)[0]
    # one scale per 32 elements, from torch.rand + 1e-8 through MXScaleTensor
    sa = MXScaleTensor(torch.rand(K // 32, generator=gen) + 1e-8).to(torch.float32)
    sb = MXScaleTensor(torch.rand(K // 32, generator=gen) + 1e-8).to(torch.float32)
    va = (xa * sa.repeat_interleave(32)).double()
    vb = (xb * sb.repeat_interleave(32)).double()
    return [float(u) * float(w) for u, w in zip(va.tolist(), vb.tolist())]


gen = torch.Generator().manual_seed(7)
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for K in (512, 8192, 16384):
    rows = []
    for _ in range(NS):
        p = sample_row(K, gen)
        S = math.fsum(p)
        A = math.fsum(abs(v) for v in p)
        T = tol_of(p)
        whole = T + (abs(S) + T) * U16 + ETA16
        rows.append((abs(S), A, T, T / A, T / abs(S) if S != 0 else float("inf"),
                     whole / abs(S) if S != 0 else float("inf")))

    def med(i):
        v = sorted(r[i] for r in rows)
        return v[len(v) // 2]

    print(f"K={K}: median |S|={med(0):.4g} median sum|p|={med(1):.4g} median tol={med(2):.4g} "
          f"max tol/sum|p|={max(r[3] for r in rows):.3g} median tol/|S|={100 * med(4):.3g}% "
          f"median whole/|S|={100 * med(5):.3g}%  (assert_close bound at median |S|: "
          f"{1e-3 + 1e-3 * med(0):.4g})")
