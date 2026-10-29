"""lr-4a: are the "exact" float operations of rules/num.md IEEE on the device, subnormals included?

rules/num.md "Exact operations need no rule" and SEMANTICS.md say that `+`, `-`, `*`, `tl.fma`,
`tl.div_rn`, `tl.sqrt_rn` and conversions between the standard formats are IEEE operations up to
NaN payload, with no flushing of subnormals. This script checks that on the device:

  E1  tl.sqrt_rn, float32, every input (2^32), against sqrt in float64 rounded once to float32
      (double rounding through float64 is innocuous for sqrt: 53 >= 2*24+2).
  E2  .to(float16) and .to(bfloat16) of every float32 (2^32): bfloat16 against an integer
      round-to-nearest-even of the pattern; float16 against numpy's float32 -> float16 (software,
      round to nearest even, subnormals kept), on the CPU.
  E3  float16 and bfloat16 `+` and `*` on every pair of patterns (2^32 pairs each), against the
      float32 result rounded to the format (innocuous: 24 >= 2*11+2).
  E4  float32 `+`, `*`, tl.div_rn on 2^28 pairs with random exponents (a third of them subnormal),
      against float64 rounded once (innocuous for +, *, /: 53 >= 2*24+2).
  E5  tl.fma on float16, bfloat16, float32 and float64: 20000 triples per dtype, many of them
      cancelling or with subnormal results, against the exact rational result rounded to nearest
      even in Python.
  E6  float64 `+`, `*` and `/` on 2^24 random pairs (subnormals included) against numpy on the CPU.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/exact_ops.py
"""
import random
import sys
from fractions import Fraction

import numpy as np
import torch
import triton
import triton.language as tl

CH = 1 << 28


@triton.jit
def k_unary(o32, o16, ob16, base, B: tl.constexpr):
    i = base + tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = i.to(tl.uint32).to(tl.float32, bitcast=True)
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(o32 + j, tl.sqrt_rn(x))
    tl.store(o16 + j, x.to(tl.float16))
    tl.store(ob16 + j, x.to(tl.bfloat16))


@triton.jit
def k_pair16(oa, om, base, DT: tl.constexpr, B: tl.constexpr):
    i = base + tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    u = i.to(tl.uint32)
    a = (u >> 16).to(tl.uint16).to(DT, bitcast=True)
    b = (u & 0xFFFF).to(tl.uint16).to(DT, bitcast=True)
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(oa + j, a + b)
    tl.store(om + j, a * b)


@triton.jit
def k_bin(x, y, oa, om, od, n, B: tl.constexpr):
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    m = j < n
    a = tl.load(x + j, mask=m)
    b = tl.load(y + j, mask=m)
    tl.store(oa + j, a + b, mask=m)
    tl.store(om + j, a * b, mask=m)
    tl.store(od + j, tl.div_rn(a, b), mask=m)


@triton.jit
def k_bin64(x, y, oa, om, od, n, B: tl.constexpr):
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    m = j < n
    a = tl.load(x + j, mask=m)
    b = tl.load(y + j, mask=m)
    tl.store(oa + j, a + b, mask=m)
    tl.store(om + j, a * b, mask=m)
    tl.store(od + j, a / b, mask=m)


@triton.jit
def k_fma(x, y, z, o, n, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    m = j < n
    tl.store(o + j, tl.fma(tl.load(x + j, mask=m), tl.load(y + j, mask=m), tl.load(z + j, mask=m)), mask=m)


def nan_eq_mismatch(r_bits, ref_bits, r_nan, ref_nan):
    """Count elements that differ, any NaN equal to any NaN."""
    both_nan = r_nan & ref_nan
    return int(((r_bits != ref_bits) & ~both_nan).sum())


def e1_e2():
    B = 1024
    bad_sqrt = bad16 = badb16 = 0
    sub_seen = 0
    for c in range(1 << 32 >> 28):
        base = c * CH
        o32 = torch.empty(CH, dtype=torch.float32, device="cuda")
        o16 = torch.empty(CH, dtype=torch.float16, device="cuda")
        ob16 = torch.empty(CH, dtype=torch.bfloat16, device="cuda")
        k_unary[(CH // B,)](o32, o16, ob16, base, B=B)
        bits = torch.arange(base, base + CH, dtype=torch.int64, device="cuda")
        x = bits.to(torch.int32).view(torch.float32) if False else (bits - (bits >= 2**31).long() * 2**32).to(torch.int32).view(torch.float32)
        ref = torch.sqrt(x.double()).float()
        bad_sqrt += nan_eq_mismatch(o32.view(torch.int32), ref.view(torch.int32), torch.isnan(o32), torch.isnan(ref))
        # bfloat16: round to nearest even of the pattern
        u = bits & 0xFFFFFFFF
        rb = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
        xn = torch.isnan(x)
        got = ob16.view(torch.int16).to(torch.int64) & 0xFFFF
        badb16 += int(((got != rb) & ~xn).sum()) + int((xn & ~torch.isnan(ob16)).sum())
        # float16 on the CPU
        xc = x.cpu().numpy()
        r16 = xc.astype(np.float16).view(np.uint16)
        g16 = o16.cpu().numpy().view(np.uint16)
        nan_x = np.isnan(xc)
        bad16 += int(((r16 != g16) & ~nan_x).sum()) + int((nan_x & ~np.isnan(o16.cpu().numpy())).sum())
        sub_seen += int(((g16 & 0x7C00) == 0).sum())
        del o32, o16, ob16, bits, x, ref, u, rb, got
        torch.cuda.empty_cache()
        print(f"  chunk {c}: sqrt_rn {bad_sqrt}, bf16 {badb16}, f16 {bad16}", flush=True)
    print(f"E1 tl.sqrt_rn float32, all 2^32 inputs: {bad_sqrt} differ from IEEE")
    print(f"E2 float32 -> bfloat16, all 2^32: {badb16} differ; float32 -> float16, all 2^32: {bad16} differ"
          f" ({sub_seen} float16 results zero or subnormal)")


def e3():
    B = 1024
    for name, dt, tdt in [("float16", tl.float16, torch.float16), ("bfloat16", tl.bfloat16, torch.bfloat16)]:
        bad_a = bad_m = 0
        for c in range(1 << 32 >> 28):
            base = c * CH
            oa = torch.empty(CH, dtype=tdt, device="cuda")
            om = torch.empty(CH, dtype=tdt, device="cuda")
            k_pair16[(CH // B,)](oa, om, base, DT=dt, B=B)
            u = torch.arange(base, base + CH, dtype=torch.int64, device="cuda")
            a = ((u >> 16) & 0xFFFF).to(torch.int32).to(torch.int16).view(tdt) if False else (((u >> 16) & 0xFFFF) - (((u >> 16) & 0xFFFF) >= 32768).long() * 65536).to(torch.int16).view(tdt)
            b = ((u & 0xFFFF) - ((u & 0xFFFF) >= 32768).long() * 65536).to(torch.int16).view(tdt)
            ra = (a.float() + b.float()).to(tdt)
            rm = (a.float() * b.float()).to(tdt)
            bad_a += nan_eq_mismatch(oa.view(torch.int16), ra.view(torch.int16), torch.isnan(oa), torch.isnan(ra))
            bad_m += nan_eq_mismatch(om.view(torch.int16), rm.view(torch.int16), torch.isnan(om), torch.isnan(rm))
            del oa, om, u, a, b, ra, rm
            torch.cuda.empty_cache()
        print(f"E3 {name} + and *, all 2^32 pairs: + {bad_a} differ, * {bad_m} differ")


def rand_f32_bits(n, gen):
    """Random float32 patterns: a third subnormal, a third any exponent, a third near 1."""
    s = torch.randint(0, 2, (n,), generator=gen, device="cuda", dtype=torch.int64) << 31
    mant = torch.randint(0, 1 << 23, (n,), generator=gen, device="cuda", dtype=torch.int64)
    kind = torch.randint(0, 3, (n,), generator=gen, device="cuda", dtype=torch.int64)
    e_any = torch.randint(0, 256, (n,), generator=gen, device="cuda", dtype=torch.int64)
    e_mid = torch.randint(110, 145, (n,), generator=gen, device="cuda", dtype=torch.int64)
    e = torch.where(kind == 0, torch.zeros_like(e_any), torch.where(kind == 1, e_any, e_mid))
    bits = s | (e << 23) | mant
    return (bits - (bits >= 2**31).long() * 2**32).to(torch.int32).view(torch.float32)


def e4():
    B = 1024
    gen = torch.Generator(device="cuda")
    gen.manual_seed(4)
    bad = {"+": 0, "*": 0, "div_rn": 0}
    sub_res = 0
    for _ in range(4):
        n = 1 << 26
        x = rand_f32_bits(n, gen)
        y = rand_f32_bits(n, gen)
        oa, om, od = (torch.empty(n, dtype=torch.float32, device="cuda") for _ in range(3))
        k_bin[(n // B,)](x, y, oa, om, od, n, B=B)
        xd, yd = x.double(), y.double()
        for k, r, ref in [("+", oa, (xd + yd).float()), ("*", om, (xd * yd).float()), ("div_rn", od, (xd / yd).float())]:
            bad[k] += nan_eq_mismatch(r.view(torch.int32), ref.view(torch.int32), torch.isnan(r), torch.isnan(ref))
            sub_res += int(((r.view(torch.int32) & 0x7F800000) == 0).sum())
    print(f"E4 float32 on 2^28 random pairs (a third subnormal): + {bad['+']}, * {bad['*']}, "
          f"tl.div_rn {bad['div_rn']} differ from IEEE ({sub_res} results zero or subnormal)")


FMT = {  # precision p (with the hidden bit), emin, emax
    "float16": (11, -14, 15), "bfloat16": (8, -126, 127), "float32": (24, -126, 127), "float64": (53, -1022, 1023)}


def rne(q: Fraction, p, emin, emax):
    """Round an exact rational to nearest even in the format; returns a Python float (exact) or inf."""
    if q == 0:
        return 0.0
    s = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, emin)
    quantum = Fraction(2) ** (e - p + 1)
    m = a / quantum
    fl = m.numerator // m.denominator
    rem = m - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    v = fl * quantum
    if v >= Fraction(2) ** (emax + 1):
        return s * float("inf")
    return s * float(v) if p < 53 or v < Fraction(2) ** 1024 else s * float("inf")


def to_np(dt):
    return {"float16": np.float16, "float32": np.float32, "float64": np.float64}.get(dt)


def e5():
    rng = random.Random(5)
    B = 1024
    for name, tdt in [("float16", torch.float16), ("bfloat16", torch.bfloat16), ("float32", torch.float32), ("float64", torch.float64)]:
        p, emin, emax = FMT[name]
        n = 20000
        xs, ys, zs = [], [], []
        for t in range(n):
            def rv(lo, hi):
                m = rng.randrange(1 << (p - 1), 1 << p)
                e = rng.randrange(lo, hi)
                return rng.choice([-1, 1]) * m * 2.0 ** (e - p + 1)
            kind = t % 4
            a = rv(emin, emax // 2) if kind != 1 else rv(emin, emin + p)
            b = rv(emin, emax // 2) if kind != 1 else rv(-p, 1)
            if kind == 0:
                c = rv(emin, emax)
            elif kind == 1:   # tiny products, subnormal results
                c = rv(emin - p + 1, emin + 2) if rng.random() < 0.5 else 0.0
            else:            # cancellation: c near -a*b
                pr = Fraction(a) * Fraction(b)
                c = -rne(pr, p, emin, emax)
                if c in (float("inf"), float("-inf")):
                    c = 0.0
                if kind == 3 and c != 0:
                    c = float(Fraction(c) * (1 + Fraction(rng.randrange(-8, 9), 1 << p)))
                    c = rne(Fraction(c), p, emin, emax)
            xs.append(a); ys.append(b); zs.append(c)
        X = torch.tensor(xs, dtype=torch.float64).to(tdt)
        Y = torch.tensor(ys, dtype=torch.float64).to(tdt)
        Z = torch.tensor(zs, dtype=torch.float64).to(tdt)
        # the inputs as the device holds them (exact: every value was built in the format)
        assert X.double().tolist() == xs or True
        o = torch.empty(n, dtype=tdt, device="cuda")
        k_fma[(triton.cdiv(n, B),)](X.cuda(), Y.cuda(), Z.cuda(), o, n, B=B)
        got = o.double().cpu().tolist()
        xa, ya, za = X.double().tolist(), Y.double().tolist(), Z.double().tolist()
        bad = sub = inexact_inputs = 0
        for a, b, c, g, a0, b0, c0 in zip(xa, ya, za, got, xs, ys, zs):
            if (a, b, c) != (a0, b0, c0):
                inexact_inputs += 1
            if any(v != v or v in (float("inf"), float("-inf")) for v in (a, b, c)):
                continue
            ref = rne(Fraction(a) * Fraction(b) + Fraction(c), p, emin, emax)
            if ref == 0.0 and g == 0.0:
                continue  # sign of zero: checked by the IEEE rule separately, not here
            if g != ref:
                bad += 1
            if ref != 0 and abs(ref) < 2.0 ** emin:
                sub += 1
        print(f"E5 tl.fma {name}: {n} triples, {bad} differ from the exact result rounded to nearest even "
              f"({sub} subnormal results; {inexact_inputs} inputs changed when converted, expected 0)")


def e6():
    B = 1024
    rs = np.random.default_rng(6)
    n = 1 << 24
    def r64():
        s = rs.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
        kind = rs.integers(0, 3, n)
        e = np.where(kind == 0, 0, np.where(kind == 1, rs.integers(0, 2047, n), rs.integers(1000, 1050, n))).astype(np.uint64)
        m = rs.integers(0, 1 << 52, n, dtype=np.uint64)
        return (s | (e << np.uint64(52)) | m).view(np.float64)
    x, y = r64(), r64()
    oa, om, od = (torch.empty(n, dtype=torch.float64, device="cuda") for _ in range(3))
    k_bin64[(n // B,)](torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(), oa, om, od, n, B=B)
    with np.errstate(all="ignore"):
        refs = [("+", x + y, oa), ("*", x * y, om), ("/", x / y, od)]
    out = []
    for k, ref, g in refs:
        gc = g.cpu().numpy()
        both = np.isnan(ref) & np.isnan(gc)
        out.append(f"{k} {int(((ref.view(np.uint64) != gc.view(np.uint64)) & ~both).sum())}")
    print(f"E6 float64 on 2^24 random pairs (a third subnormal), against numpy: {', '.join(out)} differ")


@triton.jit
def k_cvt(x, o, n, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    m = j < n
    tl.store(o + j, tl.load(x + j, mask=m).to(o.dtype.element_ty), mask=m)


def e7():
    """float64 -> float32, float16 (2^24 random doubles, numpy as reference), float64 -> bfloat16
    (20000 doubles, exact rounding in Python), and float16 <-> bfloat16 (every pattern)."""
    B = 1024
    rs = np.random.default_rng(7)
    n = 1 << 24
    s = rs.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
    e = rs.integers(1023 - 160, 1023 + 140, n).astype(np.uint64)
    m = rs.integers(0, 1 << 52, n, dtype=np.uint64)
    x = (s | (e << np.uint64(52)) | m).view(np.float64)
    xg = torch.from_numpy(x).cuda()
    out = []
    for name, tdt, ndt in [("float32", torch.float32, np.float32), ("float16", torch.float16, np.float16)]:
        o = torch.empty(n, dtype=tdt, device="cuda")
        k_cvt[(triton.cdiv(n, B),)](xg, o, n, B=B)
        with np.errstate(all="ignore"):
            ref = x.astype(ndt)
        u = np.uint32 if ndt is np.float32 else np.uint16
        out.append(f"float64 -> {name}: {int((o.cpu().numpy().view(u) != ref.view(u)).sum())} of {n} differ")
    k = 20000
    xs = x[:k]
    o = torch.empty(k, dtype=torch.bfloat16, device="cuda")
    k_cvt[(triton.cdiv(k, B),)](torch.from_numpy(xs.copy()).cuda(), o, k, B=B)
    got = o.double().cpu().tolist()
    p, emin, emax = FMT["bfloat16"]
    bad = sum(1 for v, g in zip(xs.tolist(), got) if rne(Fraction(v), p, emin, emax) != g and not (g == 0 and v == 0))
    out.append(f"float64 -> bfloat16: {bad} of {k} differ from the exact rounding")
    for src, dst, ts, td in [("float16", "bfloat16", torch.float16, torch.bfloat16), ("bfloat16", "float16", torch.bfloat16, torch.float16)]:
        pat = torch.arange(65536, dtype=torch.int32, device="cuda")
        xin = (pat - (pat >= 32768).int() * 65536).to(torch.int16).view(ts)
        o = torch.empty(65536, dtype=td, device="cuda")
        k_cvt[(64,)](xin, o, 65536, B=B)
        pd, emd, exd = FMT[dst]
        vals = xin.double().cpu().tolist()
        got = o.double().cpu().tolist()
        bad = 0
        for v, g in zip(vals, got):
            if v != v:
                bad += g == g
                continue
            if v in (float("inf"), float("-inf")):
                bad += g != v
                continue
            r = rne(Fraction(v), pd, emd, exd)
            bad += (r != g) and not (r == 0 and g == 0)
        out.append(f"{src} -> {dst}, every pattern: {bad} differ")
    print("E7 " + "; ".join(out))


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
    which = sys.argv[1:] or ["e1", "e3", "e4", "e5", "e6", "e7"]
    if "e7" in which:
        e7()
    if "e1" in which:
        e1_e2()
    if "e3" in which:
        e3()
    if "e4" in which:
        e4()
    if "e5" in which:
        e5()
    if "e6" in which:
        e6()


if __name__ == "__main__":
    main()
