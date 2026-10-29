"""lr-14a: `/` and `tl.fdiv` on float64 against an independent IEEE reference (L67).

The reference is the host CPU's float64 division through numpy (x86 `divpd`, IEEE division with
one rounding to nearest even, subnormals kept: numpy sets no flush-to-zero; checked below on
subnormal cases with exact fractions), not TVA's Lean definition, so it checks the evidence's
reference too. Families of my own (fixed seed), 2^24 pairs per chunk:
  uniform   both operands uniform 64-bit patterns
  spread    random significands, exponents of x and y anywhere in [-1074, 1023] (subnormal and
            overflowing quotients, subnormal operands)
  near1     y within 2^22 ulps of +-1, x random finite
  mid       x near m*y for m a midpoint between consecutive doubles (a random double q plus half
            its ulp h): x = RN(m*y) exactly (fractions) on a subsample of 2^16 pairs per chunk,
            x = RN(q*y) + RN(h*y) in float64 elsewhere, so the quotient is near a midpoint
  shortsig  x, y with short significands (<= 8 bits) and random exponents: exact quotients and
            ties, also in the subnormal range
Results compared bit for bit, NaN against NaN up to payload.
Run: CUDA_VISIBLE_DEVICES=0 python -I div_f64.py
"""
from fractions import Fraction

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def kdiv(x_ptr, y_ptr, o_ptr, o2_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float64, bitcast=True)
    y = tl.load(y_ptr + offs, mask=m).to(tl.float64, bitcast=True)
    tl.store(o_ptr + offs, (x / y).to(tl.int64, bitcast=True), mask=m)
    tl.store(o2_ptr + offs, tl.fdiv(x, y).to(tl.int64, bitcast=True), mask=m)


N = 1 << 24
rng = np.random.default_rng(1414)


def pats(a):
    return a.view(np.uint64)


def from_parts(sign, e, sig):
    """sign 0/1, biased exponent field 0..2046, 52-bit fraction."""
    return ((sign.astype(np.uint64) << np.uint64(63)) | (e.astype(np.uint64) << np.uint64(52)) | sig.astype(np.uint64)).view(np.float64)


def fam_uniform():
    return rng.integers(0, 1 << 64, N, dtype=np.uint64, endpoint=False).view(np.float64), \
        rng.integers(0, 1 << 64, N, dtype=np.uint64, endpoint=False).view(np.float64)


def fam_spread():
    def one():
        sig = rng.integers(0, 1 << 52, N, dtype=np.uint64)
        e = rng.integers(0, 2047, N)  # 0: subnormal or zero
        return from_parts(rng.integers(0, 2, N), e, sig)
    return one(), one()


def fam_near1():
    k = rng.integers(-(1 << 22), (1 << 22) + 1, N)
    one = np.float64(1.0).view(np.uint64)
    y = (np.uint64(one) + k.astype(np.int64).view(np.uint64)).view(np.float64)
    y = np.where(rng.integers(0, 2, N) == 1, -y, y)
    x = from_parts(rng.integers(0, 2, N), rng.integers(1, 2047, N), rng.integers(0, 1 << 52, N, dtype=np.uint64))
    return x, y


def fam_mid():
    y = from_parts(rng.integers(0, 2, N), rng.integers(1023 - 200, 1023 + 200, N), rng.integers(0, 1 << 52, N, dtype=np.uint64))
    q = from_parts(rng.integers(0, 2, N), rng.integers(1023 - 200, 1023 + 200, N), rng.integers(0, 1 << 52, N, dtype=np.uint64))
    # m = q + ulp(q)/2; x = RN(q*y) + RN(h*y) in float64, and RN(m*y) exactly on a subsample
    h = np.spacing(np.abs(q)) / 2
    p = q * y
    x = p + h * y
    sub = rng.choice(N, 1 << 16, replace=False)
    for i in sub:
        exact = (Fraction(float(q[i])) + Fraction(float(h[i]))) * Fraction(float(y[i]))
        x[i] = float(exact)  # Python's float(Fraction) rounds to nearest even
    return x, y


def fam_shortsig():
    def one():
        sig = (rng.integers(0, 256, N, dtype=np.uint64) << np.uint64(44))
        e = rng.integers(0, 2047, N)
        return from_parts(rng.integers(0, 2, N), e, sig)
    return one(), one()


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    # the reference keeps subnormals and rounds to nearest even
    tiny = np.array([5e-324, 1.5e-323, 2.2250738585072014e-308, 1e-310], dtype=np.float64)
    dv = np.array([2.0, 2.0, 3.0, 7.0])
    for a, b in zip(tiny, dv):
        exact = Fraction(float(a)) / Fraction(float(b))
        assert float(exact) == float(a / b), (a, b)
    print("reference check: numpy float64 division on subnormals equals exact rounding (fractions)")
    print("| family | pairs | x / y differ | tl.fdiv differ | NaN results |")
    print("|---|---|---|---|---|")
    total = 0
    for name, fam, reps in [("uniform", fam_uniform, 4), ("spread", fam_spread, 4), ("near1", fam_near1, 2),
                            ("mid", fam_mid, 1), ("shortsig", fam_shortsig, 2)]:
        nd = nf = nn = cnt = 0
        for _ in range(reps):
            x, y = fam()
            with np.errstate(all="ignore"):
                ref = x / y
            xd = torch.from_numpy(pats(x).view(np.int64)).cuda()
            yd = torch.from_numpy(pats(y).view(np.int64)).cuda()
            o = torch.empty_like(xd); o2 = torch.empty_like(xd)
            kdiv[(N // 1024,)](xd, yd, o, o2, N, B=1024)
            torch.cuda.synchronize()
            for out, tag in ((o, "div"), (o2, "fdiv")):
                got = out.cpu().numpy().view(np.float64)
                same = (got.view(np.uint64) == ref.view(np.uint64)) | (np.isnan(got) & np.isnan(ref))
                bad = int((~same).sum())
                if tag == "div":
                    nd += bad
                    nn += int(np.isnan(ref).sum())
                else:
                    nf += bad
                if bad:
                    k = int(np.nonzero(~same)[0][0])
                    print(f"  {name} {tag}: first difference x={x[k]!r} y={y[k]!r} device={got[k]!r} ref={ref[k]!r}")
            cnt += N
        total += cnt
        print(f"| {name} | {cnt} | {nd} | {nf} | {nn} |")
    print(f"\nTotal pairs: {total} (each through x / y and tl.fdiv)")


if __name__ == "__main__":
    main()
