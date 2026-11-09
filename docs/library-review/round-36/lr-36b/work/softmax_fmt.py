"""lr-36b, library issue L166: tl.softmax and tl.fdiv(ieee_rounding=True) on every float format,
mixed operands and constants, and a fused following addition with the default.

  A. tl.softmax(x, dim=1, ieee_rounding=r) on float16, bfloat16, float32, float64 tiles (results stored as float32,
     float64 for float64): compiles?
     the PTX division; on float32/float64, results bit-exact against num/den computed in the
     same kernel with tl.div_rn (r=True, float32) or the format's IEEE division (float64, numpy).
  B. tl.fdiv(x, y, ieee_rounding=True) with a float32 x and a Python constant y (3.0, 1.0), with
     float32 / float16 mixed operands, and with float32 x / float64 y.
  C. tl.softmax(x, ieee_rounding=True) + c, enable_fp_fusion on: bit-exact against the IEEE sum.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import re

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def ksm(x, o, nm, dn, R: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(x + r)
    s = tl.softmax(v, dim=1, keep_dims=True, ieee_rounding=R)
    tl.store(o + r, s)
    z = v - tl.max(v, 1, keep_dims=True)
    num = tl.exp(z)
    tl.store(nm + r, num)
    tl.store(dn + r, tl.broadcast_to(tl.sum(num, 1, keep_dims=True), (M, N)))


@triton.jit
def kc(x, o, Y: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(o + j, tl.fdiv(tl.load(x + j), Y, ieee_rounding=True))


@triton.jit
def km(x, y, o, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(o + j, tl.fdiv(tl.load(x + j), tl.load(y + j), ieee_rounding=True))


@triton.jit
def ksa(x, c, o, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    s = tl.softmax(tl.load(x + r), dim=1, keep_dims=True, ieee_rounding=True)
    tl.store(o + r, s - tl.load(c + r))


def divs(ptx):
    return sorted(set(re.findall(r"\bdiv\.[a-z0-9.]+", ptx)))


def last(e):
    return type(e).__name__ + ": " + str(e).strip().splitlines()[-1][:110] if str(e).strip() else type(e).__name__


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    g = torch.Generator(device="cuda").manual_seed(166)
    M, N = 64, 16
    print("## A. tl.softmax by format")
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for R in (False, True):
            x = (torch.randn(M * N, device="cuda", generator=g, dtype=torch.float32) * 8).to(dt)
            od = torch.float64 if dt == torch.float64 else torch.float32   # tl.max widens 16-bit floats
            o, nm, dn = (torch.empty(M * N, device="cuda", dtype=od) for _ in range(3))
            try:
                h = ksm[(1,)](x, o, nm, dn, R, M, N)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{dt} ieee_rounding={R}: rejected ({last(e)})")
                continue
            ref = (nm.cpu().numpy() / dn.cpu().numpy())   # numpy IEEE division in the format
            got = o.cpu().numpy()
            diff = int(np.sum(got.view(f"u{got.itemsize}") != ref.view(f"u{ref.itemsize}")))
            print(f"{dt} ieee_rounding={R}: {divs(h.asm['ptx'])}; results != IEEE num/den: {diff} of {M * N}")
    print("## B. tl.fdiv(ieee_rounding=True) with constants and mixed operands")
    x = torch.randn(256, device="cuda", generator=g)
    for Y in (3.0, 1.0):
        o = torch.empty_like(x)
        try:
            h = kc[(1,)](x, o, Y, 256)
            torch.cuda.synchronize()
            ref = (x.cpu().numpy() / np.float32(Y))
            d = int(np.sum(o.cpu().numpy().view("u4") != ref.view("u4")))
            print(f"float32 / constant {Y}: {divs(h.asm['ptx'])}; != IEEE: {d} of 256")
        except Exception as e:
            print(f"float32 / constant {Y}: rejected ({last(e)})")
    for dy in (torch.float16, torch.float64):
        y = torch.randn(256, device="cuda", generator=g).to(dy)
        o = torch.empty_like(x)
        try:
            h = km[(1,)](x, y, o, 256)
            torch.cuda.synchronize()
            print(f"float32 / {dy}: compiles: {divs(h.asm['ptx'])}")
        except Exception as e:
            print(f"float32 / {dy}: rejected ({last(e)})")
    print("## C. tl.softmax(ieee_rounding=True) - c, fusion on")
    bad, tot = 0, 0
    for t in range(16):
        x = torch.randn(M * N, device="cuda", generator=g) * 4
        o0, nm, dn = (torch.empty_like(x) for _ in range(3))
        ksm[(1,)](x, o0, nm, dn, True, M, N)
        c = o0 * (1 + torch.randn(M * N, device="cuda", generator=g) * 2 ** -20)
        o = torch.empty_like(x)
        ksa[(1,)](x, c, o, M, N)
        torch.cuda.synchronize()
        ref = o0.cpu().numpy() - c.cpu().numpy()
        bad += int(np.sum(o.cpu().numpy().view("u4") != ref.view("u4")))
        tot += M * N
    print(f"{bad} of {tot} differ from the IEEE difference of the stored softmax")


if __name__ == "__main__":
    main()
