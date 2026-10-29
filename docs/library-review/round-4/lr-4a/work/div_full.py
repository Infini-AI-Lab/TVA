"""lr-4a: is `Rules.Num.DivFull` true of `/` on float32 on the device?

`DivFull P := DivAcc P binary32 2^126 2^-22 2^-148`: for finite x, y, y != 0, |x/y| <= 2^126, the
result of `x / y` is finite and within 2^-22*|x/y| + 2^-148 of x/y. Triton 3.8.0 compiles float32
`/` to `div.full.f32` (rules/num.md "Division"; this script checks the PTX too).

Method: 2^30 pairs in 16 batches, six kinds of adversarial pairs (any exponents, subnormals included;
|y| >= 2^126; subnormal x; quotients near 2^126; subnormal quotients; quotients near 1). The
quotient is computed in float64 (relative error 2^-53); a pair is a violation when the float64
error exceeds the bound by more than that, and every pair within 2^-40 of the bound is re-checked
exactly with Python fractions. Also reports the largest error in ulps of the quotient.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/div_full.py
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


def bits_to_f32(bits):
    bits = bits & 0xFFFFFFFF
    return (bits - (bits >= 2**31).long() * 2**32).to(torch.int32).view(torch.float32)


def make(n, gen, kind):
    r = lambda lo, hi: torch.randint(lo, hi, (n,), generator=gen, device="cuda", dtype=torch.int64)
    sx, sy = r(0, 2) << 31, r(0, 2) << 31
    mx, my = r(0, 1 << 23), r(0, 1 << 23)
    if kind == 0:
        ex, ey = r(0, 255), r(0, 255)
    elif kind == 1:      # |y| >= 2^126
        ex, ey = r(1, 255), r(253, 255)
    elif kind == 2:      # subnormal x
        ex, ey = torch.zeros(n, dtype=torch.int64, device="cuda"), r(100, 155)
    elif kind == 3:      # quotient near 2^126 (biased exponent of q about 252..253)
        ey = r(0, 130)
        ex = (ey + r(125, 127)).clamp(max=254)
    elif kind == 4:      # subnormal quotient
        ex = r(1, 60)
        ey = ex + 127 + r(-1, 30)
        ey = ey.clamp(min=1, max=254)
    else:                # quotient near 1
        ex = r(1, 255)
        ey = ex
    x = bits_to_f32(sx | (ex << 23) | mx)
    y = bits_to_f32(sy | (ey << 23) | my)
    return x, y


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    B = 1024
    n = 1 << 26
    gen = torch.Generator(device="cuda")
    gen.manual_seed(126)
    x0 = torch.ones(B, device="cuda"); y0 = torch.ones(B, device="cuda"); o0 = torch.empty(B, device="cuda")
    h = k_div[(1,)](x0, y0, o0, B, B=B)
    ins = sorted({l.split()[0] for l in h.asm["ptx"].splitlines() if l.strip().startswith(("div", "rcp"))})
    print("PTX division instructions:", ins)
    total = covered = viol = recheck = nonfinite = 0
    worst = (0.0, None)
    per_kind = [0] * 6
    for b in range(16):
        kind = b % 6
        x, y = make(n, gen, kind)
        o = torch.empty(n, dtype=torch.float32, device="cuda")
        k_div[(n // B,)](x, y, o, n, B=B)
        xd, yd, od = x.double(), y.double(), o.double()
        q = xd / yd
        ok = torch.isfinite(xd) & torch.isfinite(yd) & (yd != 0) & (q.abs() <= 2.0 ** 126)
        total += n
        covered += int(ok.sum())
        fin = torch.isfinite(od)
        nonfinite += int((ok & ~fin).sum())
        err = (od - q).abs()
        bound = 2.0 ** -22 * q.abs() + 2.0 ** -148
        v = ok & fin & (err > bound * (1 + 2.0 ** -50))
        viol += int(v.sum())
        per_kind[kind] += int(v.sum())
        near = ok & fin & (err > bound * (1 - 2.0 ** -40)) & ~v
        idx = torch.nonzero(near).flatten()[:2000].tolist()
        for i in idx:
            recheck += 1
            fx, fy, fo = Fraction(float(xd[i])), Fraction(float(yd[i])), Fraction(float(od[i]))
            qq = fx / fy
            if abs(fo - qq) > Fraction(1, 2**22) * abs(qq) + Fraction(1, 2**148):
                viol += 1
                per_kind[kind] += 1
        # error in ulps of the quotient
        e = torch.floor(torch.log2(q.abs().clamp(min=2.0 ** -149))).clamp(min=-126)
        ulp = torch.pow(2.0, e - 23)
        ue = torch.where(ok & fin, err / ulp, torch.zeros_like(err))
        m = float(ue.max())
        if m > worst[0]:
            i = int(ue.argmax())
            worst = (m, (float(xd[i]), float(yd[i]), float(od[i]), float(q[i])))
        del x, y, o, xd, yd, od, q, ok, err, bound, ue
        torch.cuda.empty_cache()
    print(f"pairs {total}, covered by DivFull (finite, y != 0, |x/y| <= 2^126) {covered}")
    print(f"covered pairs with a non-finite result: {nonfinite}")
    print(f"covered pairs outside 2^-22*|x/y| + 2^-148: {viol} (per kind {per_kind}); re-checked exactly near the bound: {recheck}")
    print(f"largest error: {worst[0]:.4f} ulp of the quotient at x={worst[1][0]!r}, y={worst[1][1]!r}, result {worst[1][2]!r}, x/y={worst[1][3]!r}")


if __name__ == "__main__":
    main()
