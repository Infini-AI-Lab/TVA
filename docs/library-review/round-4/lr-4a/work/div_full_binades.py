"""lr-4a: the error of float32 `/` (div.full.f32) per binade of the quotient, against DivFull.

div_full.py found errors above the PTX ISA's 2 ulp (2.085 ulp at x/y ~ 0.995 * 2^126), still inside
DivFull's 2^-22*|x/y| + 2^-148 because that bound is up to 4 ulp at the top of a binade. This script
looks for the largest error per binade of the quotient, and splits the binade into the bottom
eighth of mantissas (where 2^-22*|q| is about 2 ulp) and the rest, to see whether the bound
itself can fail. For each binade 2^E <= |q| < 2^(E+1) and each exponent of y, it builds y at random,
q = y * m * 2^E with a random mantissa m in [1, 2), x = round(q*y), so the quotient lands in the binade.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/div_full_binades.py
"""
from fractions import Fraction

import torch
import triton
import triton.language as tl


@triton.jit
def k_div(x, y, o, n, B: tl.constexpr):
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    m = j < n
    tl.store(o + j, tl.load(x + j, mask=m) / tl.load(y + j, mask=m), mask=m)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    B = 1024
    n = 1 << 24
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    rows = []
    worst_ratio = (0.0, None)
    for E in [125, 124, 120, 64, 1, 0, -1, -64, -120, -125, -126, -127, -130, -140, -148, -149]:
        mx_ulp = mx_ulp_bottom = mx_ratio = 0.0
        cnt = 0
        for ylo, yhi in [(-149, -126), (-126, -64), (-64, 0), (0, 64), (64, 126), (126, 128)]:
            # y = s * 2^e * (1 + f), x = round(y * q)
            ey = torch.randint(ylo, yhi, (n,), generator=gen, device="cuda").double()
            fy = torch.rand(n, generator=gen, device="cuda", dtype=torch.float64)
            y = (torch.pow(2.0, ey) * (1 + fy)).float()
            mq = 1 + torch.rand(n, generator=gen, device="cuda", dtype=torch.float64)
            mq = torch.where(torch.arange(n, device="cuda") % 2 == 0, 1 + (mq - 1) / 8, mq)  # half in the bottom eighth
            x = (y.double() * mq * 2.0 ** E).float()
            ok = torch.isfinite(x) & (y != 0) & (x != 0) & torch.isfinite(y)
            q = x.double() / y.double()
            ok &= (q.abs() <= 2.0 ** 126) & (q.abs() >= 2.0 ** E) & (q.abs() < 2.0 ** (E + 1))
            o = torch.empty(n, dtype=torch.float32, device="cuda")
            k_div[(n // B,)](x, y, o, n, B=B)
            od = o.double()
            fin = torch.isfinite(od)
            err = (od - q).abs()
            ulp = 2.0 ** (max(E, -126) - 23)
            ue = torch.where(ok & fin, err / ulp, torch.zeros_like(err))
            bottom = (q.abs() < 2.0 ** E * 1.125)
            ratio = torch.where(ok & fin, err / (2.0 ** -22 * q.abs() + 2.0 ** -148), torch.zeros_like(err))
            cnt += int(ok.sum())
            mx_ulp = max(mx_ulp, float(ue.max()))
            mx_ulp_bottom = max(mx_ulp_bottom, float(torch.where(bottom, ue, torch.zeros_like(ue)).max()))
            r = float(ratio.max())
            if r > mx_ratio:
                mx_ratio = r
            if r > worst_ratio[0]:
                i = int(ratio.argmax())
                worst_ratio = (r, (float(x[i]), float(y[i]), float(o[i])))
            if int((ok & ~fin).sum()):
                print(f"  E={E}: {int((ok & ~fin).sum())} covered pairs with a non-finite result")
        rows.append((E, cnt, mx_ulp, mx_ulp_bottom, mx_ratio))
        print(f"q in [2^{E}, 2^{E+1}): {cnt} pairs; max error {mx_ulp:.4f} ulp; in the bottom eighth {mx_ulp_bottom:.4f} ulp;"
              f" max error / (2^-22|q| + 2^-148) = {mx_ratio:.4f}", flush=True)
    r, (x, y, o) = worst_ratio
    qq = Fraction(x) / Fraction(y)
    exact = abs(Fraction(o) - qq) / (Fraction(1, 2**22) * abs(qq) + Fraction(1, 2**148))
    print(f"largest error / bound: {r:.6f} (exactly {float(exact):.6f}) at x={x!r}, y={y!r}, result {o!r}")


if __name__ == "__main__":
    main()
