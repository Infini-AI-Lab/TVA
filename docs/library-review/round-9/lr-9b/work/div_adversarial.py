"""lr-9b, library review round 9 (L4): `DivFull` where its bound is tightest in ulps.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-9/lr-9b/work/div_adversarial.py

DivFull: |r - x/y| <= 2^-22 |x/y| + 2^-148 for |x/y| <= 2^126. Just above a power of two the bound
is 2 ulp of the quotient (it is almost 4 ulp just below one, where the evidence's largest ulp
errors are). For every divisor mantissa (2^23 values, divisor exponents 0 and 100 and -100), the
dividend is the float nearest to y * (1 + j * 2^-23) for j in a list (quotients just above 1),
and the float nearest to y * (2 - j * 2^-23) (just below 2), and the float nearest to y * 2^-140
(subnormal quotients). The reference is the float64 quotient (IEEE), whose own error is added.
"""
import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def div(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x / y, mask=m)


def check(x, y):
    o = torch.empty_like(x)
    div[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), B=1024)
    q = x.double() / y.double()
    keep = torch.isfinite(x) & torch.isfinite(y) & (y != 0) & (q.abs() <= 2.0 ** 126)
    err = (o.double() - q).abs()
    lhs = err * (1 + 2.0 ** -52) + 2.0 ** -52 * q.abs()
    bound = 2.0 ** -22 * q.abs() + 2.0 ** -148
    fin = torch.isfinite(o)
    viol = int((keep & (~fin | (lhs > bound))).sum())
    r = torch.where(keep & fin, lhs / bound, torch.zeros_like(lhs))
    return int(keep.sum()), viol, float(r.max())


def main():
    mant = torch.arange(2 ** 23, device=DEV, dtype=torch.int64)
    print("| divisor exponent | quotients | pairs | outside DivFull | largest error / bound |")
    print("|---|---|---|---|---|")
    tot = 0
    for e in (0, 100, -100, -126):
        y = ((mant + ((e + 127) << 23)).to(torch.int32)).view(torch.float32)
        for name, fac in (("just above 1", [1 + j * 2.0 ** -23 for j in (0, 1, 2, 3, 5, 8, 13)]),
                          ("just below 2", [2 - j * 2.0 ** -23 for j in (1, 2, 3, 5, 8, 13)]),
                          ("subnormal", [2.0 ** -140, 1.5 * 2.0 ** -145, 2.0 ** -149 * 3])):
            P = V = 0
            R = 0.0
            for f in fac:
                x = (y.double() * f).float()
                for sx in (1.0, -1.0):
                    p, v, r = check(x * sx, y)
                    P += p; V += v; R = max(R, r)
            tot += V
            print(f"| {e} | {name} | {P} | {V} | {R:.4f} |")
    print(f"\nTotal outside DivFull: {tot}")


if __name__ == "__main__":
    main()
