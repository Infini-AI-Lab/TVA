"""Library issue L168 (1), part E of evidence/fdiv-ieee: tl.softmax on every float format.
Adapted from library review lr-36b's work/softmax_fmt.py (round 36).

  A. tl.softmax(x, dim=1, ieee_rounding=r) on float16, bfloat16, float32, float64 tiles: does it
     compile, the result's dtype (a static assertion: float32 for the 16-bit formats and float32,
     float64 for float64), the PTX division, and the results bit for bit against num/den with the
     IEEE division in the result's format (numpy), num and den stored by the same kernel.
  B. tl.fdiv(x, y, ieee_rounding=True) with a float32 x and a Python constant y (3.0, 1.0), with
     float32 / float16 mixed operands, and with float32 x / float64 y.
  C. tl.softmax(x, ieee_rounding=True) - c, enable_fp_fusion on: bit for bit against the IEEE
     difference.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I evidence/fdiv-ieee/softmax_fmt.py
"""
import contextlib
import os
import re
import sys

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def ksm(x, o, nm, dn, R: tl.constexpr, M: tl.constexpr, N: tl.constexpr, OD: tl.constexpr = tl.float32):
    r = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(x + r)
    s = tl.softmax(v, dim=1, keep_dims=True, ieee_rounding=R)
    tl.static_assert(s.dtype == OD, "the result of tl.softmax is not of the expected dtype")
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


@contextlib.contextmanager
def quiet():
    """Drop what the compiler writes to file descriptors 1 and 2 (a rejected module's MLIR dump,
    which names this file's path); the script's own lines are printed outside."""
    sys.stdout.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for f in saved + [null]:
            os.close(f)


def divs(ptx):
    return sorted(set(re.findall(r"\bdiv\.[a-z0-9.]+", ptx)))


def last(e):
    s = str(e).strip()
    s = re.sub(r"(/[^\s:]+)+", "<path>", s.splitlines()[-1]) if s else ""
    return f"{type(e).__name__}: {s[:110]}"


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
                with quiet():
                    h = ksm[(1,)](x, o, nm, dn, R, M, N, tl.float64 if dt == torch.float64 else tl.float32)
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"{dt} ieee_rounding={R}: rejected ({last(e)})")
                continue
            ref = (nm.cpu().numpy() / dn.cpu().numpy())   # numpy IEEE division in the format
            got = o.cpu().numpy()
            diff = int(np.sum(got.view(f"u{got.itemsize}") != ref.view(f"u{ref.itemsize}")))
            print(f"{dt} ieee_rounding={R}: result {od}; {divs(h.asm['ptx'])}; results != IEEE num/den: {diff} of {M * N}")
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
            with quiet():
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
