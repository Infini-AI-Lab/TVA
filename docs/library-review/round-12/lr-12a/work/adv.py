"""lr-12a: adversarial inputs for the float layer (TVA.Math.Defs) against Triton 3.8.0 on GPU 0.

The library side is evidence/float-layer's native driver (built by its build.sh from
lean/TVA/Math/Defs.lean), run in `dump-file` mode on records written here; the device side is a
Triton kernel that loads the same bit patterns, runs the operation, and stores the result's bits.
Both sides replace a NaN result by the format's canonical NaN (the library is up to NaN payload).

Inputs are constructed, not uniform: ties and near-ties at every rounding boundary, gradual
underflow (subnormal results of normal operands, ties at half the smallest subnormal), overflow
boundaries (maxVal + half an ulp), exact cancellation in fma and the sign of an exact zero,
products that overflow or underflow before an fma adds them, double-rounding traps for
float64 -> float32/float16/bfloat16 and for 64-bit integers -> float16/bfloat16/float32.

Usage: CUDA_VISIBLE_DEVICES=0 python -I adv.py <build folder of evidence/float-layer> [test ...]
"""
import os
import random
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl

FMT = {"f16": (5, 10), "bf16": (8, 7), "f32": (8, 23), "f64": (11, 52)}
TL = {"f16": tl.float16, "bf16": tl.bfloat16, "f32": tl.float32, "f64": tl.float64,
      "s64": tl.int64, "u64": tl.uint64, "s32": tl.int32, "u32": tl.uint32}
UT = {16: tl.uint16, 32: tl.uint32, 64: tl.uint64}


def W(t):
    return sum(FMT[t]) + 1 if t in FMT else int(t[1:])


@triton.jit
def _val(b, SRC: tl.constexpr, W_: tl.constexpr, U: tl.constexpr, ISINT: tl.constexpr):
    if ISINT:
        return b.to(SRC)
    else:
        if W_ == 64:
            return b.to(tl.float64, bitcast=True)
        else:
            return b.to(U).to(SRC, bitcast=True)


@triton.jit
def kern(out_ptr, xs, ys, zs, n, OP: tl.constexpr, SRC: tl.constexpr, W_: tl.constexpr,
         U: tl.constexpr, ISINT: tl.constexpr, DST: tl.constexpr, OW: tl.constexpr,
         OU: tl.constexpr, RTZ: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    m = offs < n
    x = _val(tl.load(xs + offs, mask=m, other=0), SRC, W_, U, ISINT)
    y = _val(tl.load(ys + offs, mask=m, other=0), SRC, W_, U, ISINT)
    z = _val(tl.load(zs + offs, mask=m, other=0), SRC, W_, U, ISINT)
    if OP == "add":
        r = x + y
    elif OP == "sub":
        r = x - y
    elif OP == "mul":
        r = x * y
    elif OP == "fma":
        r = tl.fma(x, y, z)
    elif OP == "div":
        if SRC == tl.float32:
            r = tl.div_rn(x, y)
        else:
            r = x / y
    elif OP == "max":
        r = tl.maximum(x, y)
    elif OP == "min":
        r = tl.minimum(x, y)
    else:  # cvt / itof
        if RTZ:
            r = x.to(DST, fp_downcast_rounding="rtz")
        else:
            r = x.to(DST)
    if OW == 64:
        rb = r.to(tl.int64, bitcast=True)
    else:
        rb = r.to(OU, bitcast=True).to(tl.int64)
    tl.store(out_ptr + offs, rb, mask=m)


def canon_np(o, g):
    e, mb = FMT[g]
    o = o & ((1 << (e + mb + 1)) - 1) if e + mb + 1 < 64 else o
    ex = (o >> np.uint64(mb)) & np.uint64((1 << e) - 1)
    fr = o & np.uint64((1 << mb) - 1)
    nan = (ex == np.uint64((1 << e) - 1)) & (fr != np.uint64(0))
    return np.where(nan, np.uint64((1 << (e + mb)) - 1), o)


def device(op, src, dst, x, y, z, rtz=False):
    isint = src not in FMT
    w = W(src)
    ow = W(dst)
    n = len(x)
    out = torch.empty(n, dtype=torch.int64, device="cuda")
    xs, ys, zs = (torch.from_numpy(a.astype(np.uint64).view(np.int64)).cuda() for a in (x, y, z))
    B = 1024
    kern[(triton.cdiv(n, B),)](out, xs, ys, zs, n, OP=op, SRC=TL[src], W_=w, U=UT.get(w, tl.uint64),
                               ISINT=isint, DST=TL[dst], OW=ow, OU=UT.get(ow, tl.uint64), RTZ=rtz, B=B)
    o = out.cpu().numpy().view(np.uint64)
    if ow < 64:
        o = o & np.uint64((1 << ow) - 1)
    return canon_np(o, dst)


def library(driver, spec, x, y, z, tmp):
    p = os.path.join(tmp, "adv.bin")
    np.stack([x, y, z], axis=1).astype("<u8").tofile(p)
    r = subprocess.run([driver, spec, "dump-file", p, "0"], capture_output=True, text=True, check=True)
    os.remove(p)
    out = np.zeros(len(x), dtype=np.uint64)
    for line in r.stdout.splitlines():
        a, b = line.split()
        out[int(a)] = int(b, 16)
    return out


# ------------------------------------------------------------------------------------------------
# constructing patterns

def pat(t, s, E, F):
    e, mb = FMT[t]
    return (s << (e + mb)) | (E << mb) | F


def from_frac(t, q):
    """The exact pattern of a rational q that is representable in t (asserted)."""
    from fractions import Fraction
    e, mb = FMT[t]
    bias = (1 << (e - 1)) - 1
    s = 1 if q < 0 else 0
    q = abs(Fraction(q))
    if q == 0:
        return s << (e + mb)
    # find E, F
    k = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** k > q:
        k -= 1
    if k < 1 - bias:
        F = q / Fraction(2) ** (1 - bias - mb)
        assert F.denominator == 1, q
        return pat(t, s, 0, int(F))
    N = q / Fraction(2) ** (k - mb)
    assert N.denominator == 1 and (1 << mb) <= N < (2 << mb), q
    return pat(t, s, k + bias, int(N) - (1 << mb))


def rnd_pat(t, rng, Elo, Ehi, short=False):
    e, mb = FMT[t]
    E = rng.randint(Elo, Ehi)
    if short:
        k = rng.randint(0, mb)
        F = (rng.getrandbits(mb) >> (mb - k)) << (mb - k) if k else 0
    else:
        F = rng.getrandbits(mb)
    return pat(t, rng.getrandbits(1), E, F)


def flip(t, p):
    e, mb = FMT[t]
    return p ^ (1 << (e + mb))


def gen_arith(t, op, n, rng):
    """Adversarial (x, y, z) for op on format t."""
    from fractions import Fraction as Q
    e, mb = FMT[t]
    bias = (1 << (e - 1)) - 1
    emax = (1 << e) - 2
    maxpat = pat(t, 0, emax, (1 << mb) - 1)
    xs, ys, zs = [], [], []
    for i in range(n):
        kind = rng.randrange(8)
        x = y = z = 0
        if op in ("add", "sub"):
            if kind == 0:   # both tiny: subnormal results
                x, y = rnd_pat(t, rng, 0, 2), rnd_pat(t, rng, 0, 2)
            elif kind == 1:  # tie or near-tie: y about half an ulp of x
                x = rnd_pat(t, rng, mb + 3, emax)
                Ex = (x >> mb) & ((1 << e) - 1)
                d = rng.choice([-2, -1, 0, 0, 0, 1])
                Fy = rng.choice([0, 0, (1 << mb) - 1, 1 << (mb - 1)])
                y = pat(t, rng.getrandbits(1), max(Ex - mb - 1 + d, 0), Fy)
            elif kind == 2:  # top of the range: maxVal plus about half an ulp
                x = maxpat | (rng.getrandbits(1) << (e + mb))
                d = rng.choice([-1, 0, 0, 1])
                y = pat(t, (x >> (e + mb)) & 1, emax - mb - 1 + d, rng.choice([0, 0, 1, (1 << mb) - 1]))
            elif kind == 3:  # cancellation near the bottom of the normal range
                x = rnd_pat(t, rng, 1, 3)
                y = flip(t, (x + rng.randint(-5, 5)) & ((1 << (e + mb)) - 1) | (x & (1 << (e + mb))))
            elif kind == 4:  # exact zero: x + (-x), and signed zeros
                x = rnd_pat(t, rng, 0, emax)
                y = flip(t, x) if op == "add" else x
                if rng.random() < 0.3:
                    x, y = rng.choice([0, 1 << (e + mb)]), rng.choice([0, 1 << (e + mb)])
            else:  # random exponents within mb + 3 of each other
                x = rnd_pat(t, rng, 0, emax, short=rng.random() < 0.5)
                Ex = (x >> mb) & ((1 << e) - 1)
                y = rnd_pat(t, rng, max(0, Ex - mb - 3), min(emax, Ex + 2), short=rng.random() < 0.5)
        elif op == "mul":
            if kind <= 2:  # product near or below the subnormal threshold
                P = rng.randint(1 - bias - mb - 3, 2 - bias)  # the product's unbiased exponent
                Ex = rng.randint(max(1, P + 2 * bias - emax), min(emax, P + 2 * bias - 1))
                Ey = min(max(P + 2 * bias - Ex, 1), emax)
                x = rnd_pat(t, rng, Ex, Ex, short=kind == 0)
                y = rnd_pat(t, rng, Ey, Ey, short=kind <= 1)
            elif kind == 3:  # product near the overflow threshold
                Ex = rng.randint(bias, emax)
                Ey = 3 * bias - Ex + rng.choice([-1, 0, 0, 1])
                Ey = min(max(Ey, 1), emax)
                x = pat(t, rng.getrandbits(1), Ex, (1 << mb) - 1 - rng.randint(0, 3))
                y = pat(t, rng.getrandbits(1), Ey, rng.choice([0, 1, (1 << mb) - 1, rng.getrandbits(mb)]))
            elif kind == 4:  # a subnormal times a power of two (exact or a tie)
                x = rnd_pat(t, rng, 0, 0)
                y = pat(t, rng.getrandbits(1), bias - rng.randint(1, 3), 0)
            elif kind == 5:  # zeros and signs
                x = rng.choice([0, 1 << (e + mb)])
                y = rnd_pat(t, rng, 0, emax)
            else:
                x, y = rnd_pat(t, rng, 0, emax, short=True), rnd_pat(t, rng, 0, emax, short=True)
        elif op == "fma":
            if kind == 0:  # exact cancellation: z = -(x*y) when representable
                x = rnd_pat(t, rng, bias - 10, bias + 10, short=True)
                y = rnd_pat(t, rng, bias - 10, bias + 10, short=True)
                z = 0
                xv, yv = val(t, x), val(t, y)
                try:
                    z = from_frac(t, -(xv * yv))
                except AssertionError:
                    z = flip(t, x)
            elif kind == 1:  # signed zeros everywhere
                x = rng.choice([0, 1 << (e + mb), rnd_pat(t, rng, 0, emax)])
                y = rng.choice([0, 1 << (e + mb), rnd_pat(t, rng, 0, emax)])
                z = rng.choice([0, 1 << (e + mb)])
            elif kind == 2:  # a product far below the range, z = +-0 or tiny: sign of a nonzero q
                x = rnd_pat(t, rng, 1, bias // 2)
                y = rnd_pat(t, rng, 1, bias // 2)
                z = rng.choice([0, 1 << (e + mb), rnd_pat(t, rng, 0, 0)])
            elif kind == 3:  # a product beyond maxVal brought back by z
                x = pat(t, rng.getrandbits(1), emax - rng.randint(0, 2), rng.getrandbits(mb))
                y = pat(t, 0, bias + rng.randint(1, 3), rng.getrandbits(mb))
                z = pat(t, 1 - ((x >> (e + mb)) & 1), emax, rng.choice([(1 << mb) - 1, rng.getrandbits(mb)]))
            elif kind == 4:  # near cancellation: z = -round(x*y) +- a few ulps
                x = rnd_pat(t, rng, 1, emax)
                Ex = (x >> mb) & ((1 << e) - 1)
                y = rnd_pat(t, rng, max(1, 2 * bias - Ex - 5), min(emax, 2 * bias - Ex + 5))
                xv, yv = val(t, x), val(t, y)
                pv = xv * yv
                try:
                    zz = round_pat(t, -pv)
                    z = (zz + rng.randint(-3, 3)) % (1 << (e + mb + 1))
                except Exception:
                    z = rnd_pat(t, rng, 0, emax)
            elif kind == 5:  # subnormal results by cancellation
                x = rnd_pat(t, rng, 1, 4)
                y = pat(t, rng.getrandbits(1), bias + rng.randint(-2, 2), rng.getrandbits(mb))
                xv, yv = val(t, x), val(t, y)
                try:
                    zz = round_pat(t, -(xv * yv))
                    z = (zz + rng.randint(-40, 40)) % (1 << (e + mb + 1))
                except Exception:
                    z = 0
            else:
                x, y, z = (rnd_pat(t, rng, 0, emax, short=True) for _ in range(3))
        elif op == "div":
            if kind <= 1:  # a subnormal or small value divided by a power of two (ties)
                x = rnd_pat(t, rng, 0, 2)
                y = pat(t, rng.getrandbits(1), bias + rng.randint(1, 4), 0)
            elif kind == 2:  # a quotient in the subnormal range
                x = rnd_pat(t, rng, 1, bias)
                Ex = (x >> mb) & ((1 << e) - 1)
                Ey = min(emax, Ex + bias - 1 + rng.randint(0, mb + 2))
                y = rnd_pat(t, rng, Ey, Ey)
            elif kind == 3:  # a quotient near maxVal
                x = pat(t, rng.getrandbits(1), emax, rng.getrandbits(mb))
                y = pat(t, 0, bias - rng.randint(0, 1), rng.choice([0, 1, rng.getrandbits(mb), (1 << mb) - 1]))
            elif kind == 4:  # exact quotients of short significands
                x = rnd_pat(t, rng, 1, emax, short=True)
                y = rnd_pat(t, rng, 1, emax, short=True)
            elif kind == 5:
                x = rng.choice([0, 1 << (e + mb), rnd_pat(t, rng, 0, emax)])
                y = rng.choice([0, 1 << (e + mb), rnd_pat(t, rng, 0, emax)])
            else:
                x, y = rnd_pat(t, rng, 0, emax), rnd_pat(t, rng, 0, emax)
        elif op in ("max", "min"):
            x = rng.choice([0, 1 << (e + mb), rnd_pat(t, rng, 0, emax)])
            y = rng.choice([0, 1 << (e + mb), x, flip(t, x), rnd_pat(t, rng, 0, emax)])
        xs.append(x); ys.append(y); zs.append(z)
    return (np.array(xs, dtype=np.uint64), np.array(ys, dtype=np.uint64), np.array(zs, dtype=np.uint64))


def val(t, p):
    from fractions import Fraction as Q
    e, mb = FMT[t]
    bias = (1 << (e - 1)) - 1
    s = (p >> (e + mb)) & 1
    E = (p >> mb) & ((1 << e) - 1)
    F = p & ((1 << mb) - 1)
    assert E != (1 << e) - 1
    v = Q(F, 1 << mb) * Q(2) ** (1 - bias) if E == 0 else Q((1 << mb) + F, 1 << mb) * Q(2) ** (E - bias)
    return -v if s else v


def round_pat(t, q):
    """Nearest-even pattern of q (finite range assumed), by brute force over the binade."""
    from fractions import Fraction as Q
    e, mb = FMT[t]
    bias = (1 << (e - 1)) - 1
    s = 1 if q < 0 else 0
    a = abs(q)
    if a == 0:
        return s << (e + mb)
    k = a.numerator.bit_length() - a.denominator.bit_length()
    if Q(2) ** k > a:
        k -= 1
    k = max(k, 1 - bias)
    quantum = Q(2) ** (k - mb)
    N = a / quantum
    fl = N.numerator // N.denominator
    r = N - fl
    if r > Q(1, 2) or (r == Q(1, 2) and fl % 2 == 1):
        fl += 1
    if fl == 0:
        return s << (e + mb)
    return from_frac(t, (-1 if s else 1) * fl * quantum)


def gen_cvt(src, dst, n, rng):
    """float64/float32 inputs at and near the rounding boundaries of dst, incl. double-rounding
    traps through float32 (for float16/bfloat16) and the subnormal and overflow ranges of dst."""
    from fractions import Fraction as Q
    es, ms = FMT[src]
    ed, md = FMT[dst]
    bd = (1 << (ed - 1)) - 1
    xs = []
    for i in range(n):
        kind = rng.randrange(6)
        # an exponent in dst's range (subnormal to top), or just outside
        k = rng.randint(1 - bd - md - 2, bd + 1)
        N = rng.getrandbits(md) | (1 << md)          # a dst significand
        mid = (Q(2 * N + 1, 2)) * Q(2) ** (k - md)    # midpoint between two dst neighbours
        if k < 1 - bd:                                # subnormal range: quantum 2^(1-bd-md)
            q0 = Q(2) ** (1 - bd - md)
            mid = (Q(rng.getrandbits(md)) + Q(1, 2)) * q0
        if kind == 0:
            v = mid
        elif kind == 1:   # just above or below the midpoint by one ulp of src
            v = mid
        elif kind == 2:   # midpoint +- 2^-(m32+2) relative: a float32 rounding lands on the midpoint
            v = mid * (1 + rng.choice([-1, 1]) * Q(1, 1 << (24 + rng.randint(1, 20))))
        elif kind == 3:   # the overflow boundary of dst: maxVal + half an ulp, +- a little
            top = Q((1 << (md + 1)) - 1) * Q(2) ** (bd - md)
            half = Q(2) ** (bd - md - 1)
            v = (top + half) * (1 + rng.choice([-1, 0, 1]) * Q(1, 1 << rng.randint(20, 50)))
        elif kind == 4:   # half the smallest subnormal, +- a little
            v = Q(2) ** (1 - bd - md - 1) * (1 + rng.choice([-1, 0, 1]) * Q(1, 1 << rng.randint(1, 50)))
        else:
            v = Q(N) * Q(2) ** (k - md) * (1 + Q(rng.getrandbits(30), 1 << 40))
        if rng.getrandbits(1):
            v = -v
        try:
            p = round_pat(src, v)
        except AssertionError:
            continue
        if kind == 1:
            p = p + rng.choice([-1, 1])
        xs.append(p % (1 << (es + ms + 1)))
    x = np.array(xs, dtype=np.uint64)
    return x, np.zeros_like(x), np.zeros_like(x)


def gen_itof(src, dst, n, rng):
    """64-bit integers at dst's midpoints, +-1, and near float32's midpoints (double rounding)."""
    ed, md = FMT[dst]
    sg = src[0] == "s"
    w = int(src[1:])
    xs = []
    for i in range(n):
        bd = (1 << (ed - 1)) - 1
        k = rng.randint(md + 1, min(w - 1, bd + 1))           # leading bit position
        N = (1 << md) | rng.getrandbits(md)
        base = N << (k - md)                                  # a dst value (k >= md)
        if k - md - 1 < 0:
            continue
        mid = base + (1 << (k - md - 1))
        kind = rng.randrange(4)
        if kind == 0:
            v = mid + rng.choice([-1, 0, 1])
        elif kind == 1 and k - 24 > 0:                        # a float32 rounding lands on mid
            v = mid + rng.choice([-1, 1]) * rng.randint(1, (1 << (k - 24)) - 1 if k > 24 else 1)
        elif kind == 2 and k - 53 > 0:                        # a float64 rounding lands on mid
            v = mid + rng.choice([-1, 1]) * rng.randint(1, (1 << (k - 53)) - 1 if k > 53 else 1)
        else:
            v = base + rng.getrandbits(max(1, k - md))
        if sg and rng.getrandbits(1):
            v = -v
        lim = (1 << (w - 1)) if sg else (1 << w)
        if not (-lim <= v < lim):
            continue
        xs.append(v % (1 << w))
    x = np.array(xs, dtype=np.uint64)
    return x, np.zeros_like(x), np.zeros_like(x)


TESTS = {}
for t in ("f32", "f64", "f16", "bf16"):
    for op in ("add", "sub", "mul", "fma") + (("max", "min") if t != "bf16" else ()):
        TESTS[f"{op}.{t}"] = (op, t, t, f"{op}:{t}", "arith")
    if t in ("f32", "f64"):
        TESTS[f"div.{t}"] = ("div", t, t, f"div:{t}", "arith")
for src, dst in (("f64", "f32"), ("f64", "f16"), ("f64", "bf16"), ("f32", "f16"), ("f32", "bf16")):
    TESTS[f"cvt.{src}.{dst}"] = ("cvt", src, dst, f"cvt:{src}:{dst}:rn:0", "cvt")
for src in ("s64", "u64"):
    for dst in ("f16", "bf16", "f32", "f64"):
        TESTS[f"itof.{src}.{dst}"] = ("itof", src, dst, f"itof:{src}:{dst}", "itof")


def main():
    build = os.path.abspath(sys.argv[1])
    driver = os.path.join(build, "driver")
    names = sys.argv[2:] or list(TESTS)
    n = int(os.environ.get("ADV_N", "200000"))
    for name in names:
        op, src, dst, spec, kind = TESTS[name]
        rng = random.Random(name)
        if kind == "arith":
            x, y, z = gen_arith(src, op, n, rng)
        elif kind == "cvt":
            x, y, z = gen_cvt(src, dst, n, rng)
        else:
            x, y, z = gen_itof(src, dst, n, rng)
        dv = device(op, src, dst, x, y, z)
        lv = library(driver, spec, x, y, z, build)
        lv = canon_np(lv, dst)
        bad = np.nonzero(dv != lv)[0]
        line = f"{name:16s} {spec:20s} {len(x):8d} inputs  " + ("agree" if len(bad) == 0 else f"DIFFER on {len(bad)}")
        # control: on how many of these inputs would a double rounding (through float32, or
        # float64 for integers) have given another result? (the inputs discriminate)
        ctl = ""
        if kind == "cvt" and src == "f64" and dst in ("f16", "bf16"):
            mid = library(driver, "cvt:f64:f32:rn:0", x, y, z, build)
            two = canon_np(library(driver, f"cvt:f32:{dst}:rn:0", mid, mid, mid, build), dst)
            ctl = f"  [control: twice through float32 differs on {int((two != lv).sum())}]"
        if kind == "itof":
            for via in ("f32", "f64"):
                if via != dst and not (via == "f32" and dst == "f64"):
                    two = canon_np(library(driver, f"itof2:{src}:{via}:{dst}", x, y, z, build), dst)
                    ctl += f"  [control: twice through {via} differs on {int((two != lv).sum())}]"
        print(line + ctl, flush=True)
        for i in bad[:8]:
            print(f"    input {int(x[i]):x},{int(y[i]):x},{int(z[i]):x}: library {int(lv[i]):x}, device {int(dv[i]):x}")
        if kind == "itof" and len(bad):
            # is the device's value one that ItofRel allows (twice through float64 or float32)?
            alts = []
            for via in ("f64", "f32"):
                if via == dst or (via == "f32" and dst not in ("f16", "bf16")):
                    continue
                av = canon_np(library(driver, f"itof2:{src}:{via}:{dst}", x, y, z, build), dst)
                alts.append(av)
            ok = np.zeros(len(bad), dtype=bool)
            for av in alts:
                ok |= dv[bad] == av[bad]
            print(f"    of which allowed by tl.ItofRel's double roundings: {int(ok.sum())}")


if __name__ == "__main__":
    main()
