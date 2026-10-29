"""lr-14b, round 14 (L67): float64 `/` on the device against IEEE division on the host.

An independent check of `DivRn P .binary64`, with inputs and seeds of my own: the device's
`x / y` (Triton 3.8.0, a kernel of my own with a different block size and warp count from
evidence/num-div-f64) against numpy's float64 division on the host CPU (SSE2 `divsd`: IEEE,
round to nearest even, subnormals kept), bit for bit, NaN up to payload. The evidence compares
with the library's own `Format.div .rn`; this compares the device with IEEE.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I div64_own.py   (output: div64_own.out)
"""
import sys

import numpy as np
import torch
import triton
import triton.language as tl

FRAC = np.uint64((1 << 52) - 1)


@triton.jit
def kdiv(o_ptr, x_ptr, y_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.float64, bitcast=True)
    y = tl.load(y_ptr + offs, mask=m, other=0).to(tl.float64, bitcast=True)
    tl.store(o_ptr + offs, (x / y).to(tl.int64, bitcast=True), mask=m)


@triton.jit
def kdiv_where(o_ptr, x_ptr, y_ptr, n, B: tl.constexpr):
    # `/` inside an expression the compiler may rewrite: (-x) / (-y) and the quotient reused
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.float64, bitcast=True)
    y = tl.load(y_ptr + offs, mask=m, other=0).to(tl.float64, bitcast=True)
    q = (-x) / (-y)
    tl.store(o_ptr + offs, q.to(tl.int64, bitcast=True), mask=m)


def dev(kernel, x, y, nw):
    n = len(x)
    xs = torch.from_numpy(x.view(np.int64)).cuda()
    ys = torch.from_numpy(y.view(np.int64)).cuda()
    out = torch.empty(n, dtype=torch.int64, device="cuda")
    B = 256
    h = kernel[(triton.cdiv(n, B),)](out, xs, ys, n, B=B, num_warps=nw)
    return out.cpu().numpy().view(np.uint64), h


def host(x, y):
    with np.errstate(all="ignore"):
        return (x.view(np.float64) / y.view(np.float64)).view(np.uint64)


def isnan(p):
    return ((p >> np.uint64(52)) & np.uint64(0x7FF) == np.uint64(0x7FF)) & (p & FRAC != np.uint64(0))


def pat(s, e, f):
    return (s.astype(np.uint64) << np.uint64(63)) | (e.astype(np.uint64) << np.uint64(52)) | (f.astype(np.uint64) & FRAC)


def rb(r, n):
    return r.integers(0, 1 << 63, n, dtype=np.uint64) * np.uint64(2) + r.integers(0, 2, n, dtype=np.uint64)


def rf(r, n):
    f = rb(r, n) & FRAC
    k = r.integers(0, 53, n).astype(np.uint64)
    return np.where(r.integers(0, 3, n) == 0, (f >> (np.uint64(52) - k)) << (np.uint64(52) - k),
                    np.where(r.integers(0, 2, n) == 0, f, FRAC - (f & np.uint64(0xFF))))


def families(r, n):
    s = lambda: r.integers(0, 2, n)
    yield "uniform patterns", rb(r, n), rb(r, n)
    e = r.integers(1, 2047, n)
    yield "equal exponents", pat(s(), e, rf(r, n)), pat(s(), e, rf(r, n))
    xe = r.integers(1, 1000, n)
    yield "quotient at 2^-1022 (+-2 binades)", pat(s(), xe, rf(r, n)), pat(s(), xe + r.integers(1020, 1025, n), rf(r, n))
    xe = r.integers(1, 2047 - 1080, n)
    yield "quotient deep subnormal / underflow", pat(s(), xe, rf(r, n)), pat(s(), xe + r.integers(1060, 1080, n), rf(r, n))
    ye = r.integers(1, 1000, n)
    yield "quotient at maxVal (+-1 binade)", pat(s(), np.minimum(ye + r.integers(1022, 1026, n), 2046), rf(r, n)), pat(s(), ye, rf(r, n))
    y = pat(s(), r.integers(1, 2047, n), rf(r, n))
    k = r.integers(-64, 65, n).astype(np.int64)
    x = (y.astype(np.int64) + k).astype(np.uint64)
    yield "x within 64 ulps of y", x, y
    yield "all-ones divisor significand", pat(s(), r.integers(1, 2047, n), rf(r, n)), pat(s(), r.integers(1, 2047, n), np.full(n, FRAC))
    yield "subnormal by subnormal", pat(s(), np.zeros(n, dtype=np.int64), rf(r, n) | np.uint64(1)), pat(s(), np.zeros(n, dtype=np.int64), rf(r, n) | np.uint64(1))
    yield "subnormal by normal", pat(s(), np.zeros(n, dtype=np.int64), rf(r, n) | np.uint64(1)), pat(s(), r.integers(1, 2047, n), rf(r, n))
    q = pat(s(), r.integers(1023 - 500, 1023 + 500, n), rf(r, n)).view(np.float64)
    yv = pat(s(), r.integers(1023 - 500, 1023 + 500, n), rf(r, n)).view(np.float64)
    with np.errstate(all="ignore"):
        xv = q * yv
    yield "x = RN(q*y), wide exponents", xv.view(np.uint64).copy(), yv.view(np.uint64).copy()


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    # the host reference keeps subnormals
    assert np.float64(5e-324) / np.float64(2.0) == 0.0 and np.float64(1e-310) / 2 == 5e-311
    rows = []
    total = 0
    bad_total = 0
    for rep in range(8):
        r = np.random.default_rng((0x14B, rep))
        for name, x, y in families(r, 1 << 22):
            for kname, kern, nw in (("x / y", kdiv, 2), ("(-x) / (-y)", kdiv_where, 8)):
                d, h = dev(kern, x, y, nw)
                hst = host(x, y)
                same = (d == hst) | (isnan(d) & isnan(hst))
                bad = int((~same).sum())
                total += len(x)
                bad_total += bad
                rows.append((rep, name, kname, len(x), bad))
                if bad:
                    i = int(np.nonzero(~same)[0][0])
                    print(f"  DIFF {name} {kname}: x={x[i]:016x} y={y[i]:016x} dev={d[i]:016x} host={hst[i]:016x}")
                if rep == 0 and name == "uniform patterns":
                    ptx = h.asm["ptx"]
                    ins = sorted({l.split()[0] for l in ptx.splitlines() if l.strip().startswith(("div.", "rcp.", "fma.rn.f64", "mul.rn.f64"))})
                    print(f"  PTX of {kname}: {', '.join(ins)}")
    agg = {}
    for rep, name, kname, n, bad in rows:
        a = agg.setdefault((name, kname), [0, 0])
        a[0] += n
        a[1] += bad
    print("\n| inputs | kernel | pairs | differ from IEEE (host) |")
    print("|---|---|---|---|")
    for (name, kname), (n, bad) in agg.items():
        print(f"| {name} | `{kname}` | {n} | {bad} |")
    print(f"\nTotal pairs: {total}; differing: {bad_total}")


if __name__ == "__main__":
    sys.exit(main())
