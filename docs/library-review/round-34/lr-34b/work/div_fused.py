"""lr-34b (round 34, new issue): `x / y` on float32 followed by an addition is contracted by ptxas.

`div.full.f32` is expanded by ptxas into `MUFU.RCP` and a multiply by the dividend; with
`enable_fp_fusion` (ptxas `--fmad=true`) that multiply fuses with a following `add.f32` into one
`FFMA` (fn_add_sass.txt). The library models `/` as `tl.div` (`P.div`, any value within `DivFull`) and
the addition, which no `*`/`tl.dot`/`tl.rand` feeds, as IEEE `tl.add`/`tl.sub` (docs/SEMANTICS.md,
"Float `+` and `-`"). Test: launch 1 stores q = x / y; launch 2 (same x, y) stores x / y - q.
The model gives `m - q` for floats m within DivFull of x/y: an exact difference of two floats of
q's binade (Sterbenz), a multiple of half an ulp of q. Count the elements where the device result is
nonzero and not such a multiple (so no Platform's `P.div` value can give it).
Also: `tl.div` with `enable_fp_fusion=False` (expected: 0 everywhere).
"""
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl


@triton.jit
def kq(x_ptr, y_ptr, q_ptr, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(q_ptr + j, tl.load(x_ptr + j) / tl.load(y_ptr + j))


@triton.jit
def kr(x_ptr, y_ptr, q_ptr, r_ptr, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(r_ptr + j, tl.load(x_ptr + j) / tl.load(y_ptr + j) - tl.load(q_ptr + j))


def ulp(v):
    q = abs(Fr(v))
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fr(2) ** e > q:
        e -= 1
    return Fr(2) ** (max(e, -126) - 23)


def main():
    torch.manual_seed(1)
    N = 1 << 16
    x = (torch.rand(N, device="cuda") + 0.5)
    y = (torch.rand(N, device="cuda") + 0.5)
    for fusion in (True, False):
        q = torch.empty_like(x)
        r = torch.empty_like(x)
        kq[(1,)](x, y, q, N, enable_fp_fusion=fusion)
        kr[(1,)](x, y, q, r, N, enable_fp_fusion=fusion)
        torch.cuda.synchronize()
        Q, R = q.cpu().tolist(), r.cpu().tolist()
        nz = sum(1 for v in R if v != 0)
        unreach = sum(1 for qq, rr in zip(Q, R) if rr != 0 and (Fr(rr) / (ulp(qq) / 2)).denominator != 1)
        ex = next(((xx, yy, qq, rr) for xx, yy, qq, rr in zip(x.cpu().tolist(), y.cpu().tolist(), Q, R) if rr != 0), None)
        print(f"enable_fp_fusion={fusion}: {N} elements, x/y - q nonzero on {nz}, of which not a multiple of "
              f"ulp(q)/2 (no model outcome) {unreach}; example (x, y, q, device x/y - q): {ex}")


if __name__ == "__main__":
    print(f"Triton {triton.__version__}, {torch.cuda.get_device_name()}")
    main()
