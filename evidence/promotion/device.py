"""What mixed-dtype operators, integer division, comparisons with Python constants, `tl.where`,
`and`/`or`/`not`, `min`/`max`, shifts and unary minus compute on the device (library issue L45,
round 8). Companion of `table.py` (the front end's types); this script checks values.

Each case runs in its own process on GPU 0 (a rejected or faulting kernel cannot disturb the
others) and prints its lines; the parent process prints them in order. A kernel checks the dtype
the front end gave its result with `tl.static_assert(z.dtype == Z)`, so a store cannot hide a
conversion.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/promotion/device.py > evidence/promotion/device-output.txt
(from TVA's root)
"""
import os
import random
import struct
import subprocess
import sys

import torch
import triton
import triton.language as tl

DEV = "cuda"
BLOCK = 1024


@triton.jit
def k_bin(x_ptr, y_ptr, z_ptr, n, OP: tl.constexpr, Z: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    if OP == "add":
        z = x + y
    elif OP == "mul":
        z = x * y
    elif OP == "div":
        z = x / y
    elif OP == "floordiv":
        z = x // y
    elif OP == "mod":
        z = x % y
    elif OP == "shl":
        z = x << y
    elif OP == "shr":
        z = x >> y
    elif OP == "lt":
        z = x < y
    elif OP == "eq":
        z = x == y
    elif OP == "minimum":
        z = tl.minimum(x, y)
    elif OP == "maximum":
        z = tl.maximum(x, y)
    elif OP == "minimum_all":
        z = tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "builtin_min":
        z = min(x, y)
    elif OP == "and":
        z = x and y
    elif OP == "or":
        z = x or y
    elif OP == "bitand":
        z = x & y
    tl.static_assert(z.dtype == Z)
    tl.store(z_ptr + offs, z, mask=m)


@triton.jit
def k_un(x_ptr, z_ptr, n, OP: tl.constexpr, Z: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if OP == "neg":
        z = -x
    elif OP == "invert":
        z = ~x
    elif OP == "not":
        z = not x
    elif OP == "and_true":
        z = x and True
    elif OP == "lt_m1":
        z = x < -1
    elif OP == "lt_200":
        z = x < 200
    elif OP == "eq_m1":
        z = x == -1
    elif OP == "lt_2p31":
        z = x < 2**31
    elif OP == "eq_f":
        z = x == 16777216.0
    elif OP == "lt_01":
        z = x < 0.1
    elif OP == "mul_01":
        z = x * 0.1
    elif OP == "add_m1":
        z = x + (-1)
    elif OP == "shl_32":
        z = x << 32
    elif OP == "min_m1":
        z = tl.minimum(x, -1)
    tl.static_assert(z.dtype == Z)
    tl.store(z_ptr + offs, z, mask=m)


@triton.jit
def k_where(c_ptr, x_ptr, y_ptr, z_ptr, n, Z: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    c = tl.load(c_ptr + offs, mask=m)
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    z = tl.where(c, x, y)
    tl.static_assert(z.dtype == Z)
    tl.store(z_ptr + offs, z, mask=m)


@triton.jit
def k_scalar_min(a, b, z_ptr, OP: tl.constexpr, Z: tl.constexpr):
    if OP == "builtin_min":
        z = min(a, b)
    elif OP == "builtin_max":
        z = max(a, b)
    elif OP == "minimum":
        z = tl.minimum(a, b)
    tl.static_assert(z.dtype == Z)
    tl.store(z_ptr, z)


TORCH = {"int1": torch.bool, "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
         "int64": torch.int64, "uint8": torch.uint8, "uint16": torch.uint16, "uint32": torch.uint32,
         "uint64": torch.uint64, "fp8e5": torch.float8_e5m2, "fp8e4nv": torch.float8_e4m3fn,
         "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
TL = {"int1": tl.int1, "int8": tl.int8, "int16": tl.int16, "int32": tl.int32, "int64": tl.int64,
      "uint8": tl.uint8, "uint16": tl.uint16, "uint32": tl.uint32, "uint64": tl.uint64,
      "fp8e5": tl.float8e5, "fp8e4nv": tl.float8e4nv, "fp16": tl.float16, "bf16": tl.bfloat16,
      "fp32": tl.float32, "fp64": tl.float64}
WIDTH = {"int8": 8, "int16": 16, "int32": 32, "int64": 64, "uint8": 8, "uint16": 16, "uint32": 32,
         "uint64": 64, "int1": 1}


def signed(d):
    return d.startswith("int") and d != "int1"


def rng(d):
    w = WIDTH[d]
    return (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed(d) else (0, (1 << w) - 1)


def wrap(v, d):
    w = WIDTH[d]
    v &= (1 << w) - 1
    if signed(d) and v >= 1 << (w - 1):
        v -= 1 << w
    return v


def ints_tensor(vals, d):
    """A device tensor of integer dtype d holding the Python ints vals (two's complement)."""
    if d in ("uint16", "uint32", "uint64"):
        sd = {"uint16": "int16", "uint32": "int32", "uint64": "int64"}[d]
        t = torch.tensor([wrap(v, sd) for v in vals], dtype=TORCH[sd])
        return t.view(TORCH[d]).to(DEV)
    if d == "int1":
        return torch.tensor([bool(v) for v in vals], dtype=torch.bool, device=DEV)
    return torch.tensor(vals, dtype=TORCH[d], device=DEV)


def ints_list(t, d):
    t = t.cpu()
    if d in ("uint16", "uint32", "uint64"):
        sd = {"uint16": "int16", "uint32": "int32", "uint64": "int64"}[d]
        return [v & ((1 << WIDTH[d]) - 1) for v in t.view(TORCH[sd]).tolist()]
    return [int(v) for v in t.tolist()]


def edges(d):
    lo, hi = rng(d)
    s = {0, 1, 2, 3, 7, hi, hi - 1, lo, lo + 1, hi // 2, hi // 3}
    if signed(d):
        s |= {-1, -2, -3, -7, lo // 2, lo // 3}
    random.seed(WIDTH[d] * (3 if signed(d) else 5))
    s |= {random.randint(lo, hi) for _ in range(40)}
    return sorted(s)


def launch_bin(op, x, y, zd):
    n = x.numel()
    z = torch.empty(n, dtype=TORCH[zd], device=DEV)
    k_bin[(triton.cdiv(n, BLOCK),)](x, y, z, n, OP=op, Z=TL[zd], BLOCK=BLOCK)
    torch.cuda.synchronize()
    return z


def launch_un(op, x, zd):
    n = x.numel()
    z = torch.empty(n, dtype=TORCH[zd], device=DEV)
    k_un[(triton.cdiv(n, BLOCK),)](x, z, n, OP=op, Z=TL[zd], BLOCK=BLOCK)
    torch.cuda.synchronize()
    return z


def tdiv(a, b):          # C's division: toward zero
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def tmod(a, b):
    return a - b * tdiv(a, b)


def pairs_all(d):
    lo, hi = rng(d)
    vals = list(range(lo, hi + 1))
    return [a for a in vals for _ in vals], [b for _ in vals for b in vals]


def pairs_edges(d):
    e = edges(d)
    return [a for a in e for _ in e], [b for _ in e for b in e]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def same_float(a, b):
    """Bitwise equal, or both NaN (tensors of the same float dtype)."""
    ia = a.view(torch.int16 if a.element_size() == 2 else torch.int32 if a.element_size() == 4 else torch.int8)
    ib = b.view(ia.dtype)
    return (ia == ib) | (torch.isnan(a.float()) & torch.isnan(b.float()))


# ---------------------------------------------------------------------------------------------- cases

def P1():
    """int / int: both operands converted to float32 first, then float32 division."""
    for d in ("int32", "int64", "uint32", "int8"):
        xs = {"int32": [16777217, 16777219, 2**31 - 1, -2**31, 7, -7],
              "int64": [2**53 + 1, 16777217, 2**62 + 2**40 + 1, -2**63, 7],
              "uint32": [2**32 - 1, 16777217, 7],
              "int8": [127, -128, 7]}[d]
        ys = [1] * len(xs)
        z = launch_bin("div", ints_tensor(xs, d), ints_tensor(ys, d), "fp32").tolist()
        exp = [f32(float(f32(float(v)))) for v in xs]   # the float32 nearest to x, divided by 1.0
        ok = all(a == b for a, b in zip(z, exp))
        print(f"  {d} / {d} -> float32: x/1 for x in {xs}: device {z}; the float32 nearest x: "
              f"{'equal' if ok else 'DIFFER ' + str(exp)}")
    z = launch_bin("div", ints_tensor([7, -7, 1, 0], "int32"), ints_tensor([2, 2, 3, 0], "int32"), "fp32").tolist()
    print(f"  int32: 7/2, -7/2, 1/3, 0/0 -> {z}")


def P2():
    """`//` and `%` on integers: C's truncation, signed or unsigned by the promoted type."""
    for d, gen in (("int8", pairs_all), ("uint8", pairs_all), ("int16", pairs_edges), ("int32", pairs_edges),
                   ("int64", pairs_edges), ("uint32", pairs_edges), ("uint64", pairs_edges)):
        xs, ys = gen(d)
        x, y = ints_tensor(xs, d), ints_tensor(ys, d)
        q = ints_list(launch_bin("floordiv", x, y, d), d)
        r = ints_list(launch_bin("mod", x, y, d), d)
        lo, hi = rng(d)
        nd = nq = nr = 0
        ovf = []
        for a, b, qq, rr in zip(xs, ys, q, r):
            if b == 0:
                continue
            if signed(d) and a == lo and b == -1:
                ovf.append((qq, rr))
                continue
            nd += 1
            if qq != wrap(tdiv(a, b), d):
                nq += 1
            if rr != wrap(tmod(a, b), d):
                nr += 1
        print(f"  {d} ({'all' if gen is pairs_all else 'edge'} pairs): {nd} pairs with b != 0"
              f"{' (and not MIN // -1)' if signed(d) else ''}: `//` differs from C's truncation in {nq}, "
              f"`%` in {nr}")
        xb0 = [(a, qq, rr) for a, b, qq, rr in zip(xs, ys, q, r) if b == 0]
        qset = sorted(set(qq for _, qq, _ in xb0))
        r_is_x = sum(1 for a, _, rr in xb0 if rr == a)
        ex = [(a, qq, rr) for a, qq, rr in xb0][:: max(1, len(xb0) // 5)][:6]
        print(f"    b = 0 ({len(xb0)} x): x // 0 takes the values {qset[:6]}{' ...' if len(qset) > 6 else ''}; "
              f"x % 0 = x for {r_is_x}; examples (x, x // 0, x % 0) {ex}")
        if ovf:
            print(f"    MIN // -1 = {ovf[0][0]}, MIN % -1 = {ovf[0][1]} (MIN = {lo})")


def P3():
    """Integer promotion: widths, signedness, int1."""
    x = ints_tensor([-5, -1, 7], "int32")
    y = ints_tensor([2**40, 0, -2**40], "int64")
    print(f"  int32 + int64 -> int64: [-5, -1, 7] + [2^40, 0, -2^40] = "
          f"{ints_list(launch_bin('add', x, y, 'int64'), 'int64')} (sign extension: [2^40-5, -1, 7-2^40])")
    x = ints_tensor([2**32 - 1, 2**31, 5], "uint32")
    y = ints_tensor([1, -1, -6], "int32")
    print(f"  uint32 + int32 -> uint32: [2^32-1, 2^31, 5] + [1, -1, -6] = "
          f"{ints_list(launch_bin('add', x, y, 'uint32'), 'uint32')}")
    print(f"  uint32 < int32 (compared as uint32): [2^32-1, 2^31, 5] < [1, -1, -6] = "
          f"{launch_bin('lt', x, y, 'int1').tolist()}")
    xs, ys = [], []
    for a in range(-128, 128):
        for b in range(256):
            xs.append(a)
            ys.append(b)
    z = ints_list(launch_bin("add", ints_tensor(xs, "int8"), ints_tensor(ys, "uint8"), "uint8"), "uint8")
    bad = sum(1 for a, b, c in zip(xs, ys, z) if c != (a + b) % 256)
    print(f"  int8 + uint8 -> uint8: all 65536 pairs, (a + b) mod 256 differs in {bad}")
    lt = launch_bin("lt", ints_tensor(xs, "int8"), ints_tensor(ys, "uint8"), "int1").tolist()
    bad = sum(1 for a, b, c in zip(xs, ys, lt) if c != ((a % 256) < b))
    print(f"  int8 < uint8 (as uint8): all 65536 pairs, (a mod 256 < b) differs in {bad}")
    b1 = ints_tensor([0, 0, 1, 1], "int1")
    b2 = ints_tensor([0, 1, 0, 1], "int1")
    print(f"  int1 + int1 -> int1: [F,F,T,T] + [F,T,F,T] = {launch_bin('add', b1, b2, 'int1').tolist()}")
    print(f"  int1 * int1 -> int1: {launch_bin('mul', b1, b2, 'int1').tolist()}")
    x = ints_tensor([1, 0, 1], "int1")
    y = ints_tensor([-1, -1, 127], "int8")
    print(f"  int1 + int8 -> int8 (int1 zero-extended): [T,F,T] + [-1,-1,127] = "
          f"{ints_list(launch_bin('add', x, y, 'int8'), 'int8')}")


def P4():
    """Mixed float dtypes: exact conversion to the promoted type, then one IEEE operation in it."""
    allh = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16).to(DEV)
    random.seed(4)
    bf = [random.randint(0, 65535) for _ in range(62)] + [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x0001]
    bad = 0
    for v in bf:
        yb = torch.tensor([v - 65536 if v >= 32768 else v], dtype=torch.int16).view(torch.bfloat16)
        y = yb.to(DEV).expand(allh.numel()).contiguous()
        z = launch_bin("add", allh, y, "fp16")
        ref = (allh.float() + y.float().half().float()).half()   # y rounded to fp16, then one fp16 addition
        bad += int((~same_float(z, ref)).sum())
    print(f"  fp16 + bf16 -> fp16 (not fp32): all 65536 fp16 x 68 bf16 (random and special): "
          f"differs from fp16(x + fp16(y)) in {bad}")
    yb = torch.tensor([1.0e38, 3.0e-39, 1.0 + 2**-7], dtype=torch.bfloat16).to(DEV)
    xh = torch.zeros(3, dtype=torch.float16, device=DEV)
    print(f"  fp16 0.0 + bf16 [1e38, 3e-39, 1+2^-7] -> fp16: {launch_bin('add', xh, yb, 'fp16').tolist()}")
    print(f"  fp16 0.0 / bf16 [1e38, 3e-39, 1+2^-7] -> fp32 (`/` widens): "
          f"{launch_bin('div', xh + 1, yb, 'fp32').tolist()}")
    ys = torch.tensor([1.0, -0.0, 3.0e-39, 1.0e38, 0.1, float('nan'), float('inf'), 65504.0, 1/3], dtype=torch.float32)
    bad = 0
    for v in ys:
        y = v.to(DEV).expand(allh.numel()).contiguous()
        z = launch_bin("mul", allh, y, "fp32")
        bad += int((~same_float(z, allh.float() * y)).sum())
    print(f"  fp16 * fp32 -> fp32: all fp16 x 9 fp32 values: differs from float32(x) * y in {bad}")
    xi = ints_tensor([16777217, 16777219, -16777217, 2**31 - 1, 3], "int32")
    yf = torch.zeros(5, dtype=torch.float32, device=DEV)
    print(f"  int32 + fp32 -> fp32: [2^24+1, 2^24+3, -(2^24+1), 2^31-1, 3] + 0.0 = "
          f"{launch_bin('add', xi, yf, 'fp32').tolist()}")
    xi = ints_tensor([65504, 65519, 65520, 2049, 2051, -65520], "int32")
    yh = torch.zeros(6, dtype=torch.float16, device=DEV)
    print(f"  int32 + fp16 -> fp16: [65504, 65519, 65520, 2049, 2051, -65520] + 0.0 = "
          f"{launch_bin('add', xi, yh, 'fp16').tolist()}")
    for f8 in ("fp8e5", "fp8e4nv"):
        a8 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(TORCH[f8]).to(DEV)
        xs = a8.repeat_interleave(256)
        ys16 = torch.arange(-32768, 32768, 256, dtype=torch.int32).to(torch.int16).view(torch.float16)
        ys16 = torch.cat([ys16, torch.tensor([1.0, -0.0, 0.5, 2.0**-24], dtype=torch.float16)])
        bad = 0
        for v in ys16:
            y = v.to(DEV).expand(256).contiguous()
            z = launch_bin("add", a8, y, "fp16")
            ref = (a8.float() + y.float()).half()
            bad += int((~same_float(z, ref)).sum())
        print(f"  {f8} + fp16 -> fp16: 256 {f8} x {ys16.numel()} fp16: differs from fp16(float32(x) + float32(y)) in {bad}")
    xs = a8 = None
    for f8a, f8b, zd in (("fp8e5", "fp8e4nv", "fp16"), ("fp8e5", "fp8e5", "fp8e5"), ("fp8e4nv", "fp8e4nv", "fp8e4nv")):
        a = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(TORCH[f8a])
        b = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(TORCH[f8b])
        xs = a.repeat_interleave(256).to(DEV)
        ys = b.repeat(256).to(DEV)
        try:
            z = launch_bin("add", xs, ys, zd)
        except Exception as e:
            print(f"  {f8a} + {f8b} -> {zd}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            continue
        exact = xs.float() + ys.float()
        if zd == "fp16":
            ref = exact.half()
            bad = int((~same_float(z, ref)).sum())
            print(f"  {f8a} + {f8b} -> fp16: all 65536 pairs: differs from fp16(float32(x) + float32(y)) in {bad}")
        else:
            ref_rne = exact.to(TORCH[zd])          # torch: to nearest even
            bad = int((~same_float(z, ref_rne)).sum())
            finite = torch.isfinite(exact)
            zf = z.float()
            # where does the device differ from round-to-nearest of the exact float32 sum?
            dif = (~same_float(z, ref_rne)) & finite
            ex = [(float(xs[i].float()), float(ys[i].float()), float(zf[i]), float(ref_rne[i].float()))
                  for i in torch.nonzero(dif).flatten()[:4].tolist()]
            print(f"  {f8a} + {f8b} -> {zd}: all 65536 pairs: differs from {zd}(float32(x) + float32(y)) "
                  f"(torch's rounding) in {bad}; examples (x, y, device, torch) {ex}")


def P5():
    """tl.where with a condition that is not int1."""
    cs = list(range(-128, 128))
    c = ints_tensor(cs, "int8")
    x = torch.ones(256, dtype=torch.float32, device=DEV)
    y = torch.zeros(256, dtype=torch.float32, device=DEV)
    z = torch.empty(256, dtype=torch.float32, device=DEV)
    k_where[(1,)](c, x, y, z, 256, Z=tl.float32, BLOCK=256)
    bad = sum(1 for cv, zv in zip(cs, z.tolist()) if zv != (1.0 if cv != 0 else 0.0))
    print(f"  int8 condition, all 256 values: differs from c != 0 in {bad}")
    cs = [0, 1, -1, 2, 256, 2**31 - 1, -2**31, 65536]
    c = ints_tensor(cs, "int32")
    k_where[(1,)](c, x, y, z, len(cs), Z=tl.float32, BLOCK=256)
    print(f"  int32 condition {cs}: {z[:len(cs)].tolist()}")
    cf = [0.0, -0.0, 1.0, float('nan'), 1e-45, -1e-45, float('inf'), 0.5]
    c = torch.tensor(cf, dtype=torch.float32, device=DEV)
    k_where[(1,)](c, x, y, z, len(cf), Z=tl.float32, BLOCK=256)
    print(f"  float32 condition [0.0, -0.0, 1.0, nan, 1e-45, -1e-45, inf, 0.5]: {z[:len(cf)].tolist()}")
    xh = torch.ones(256, dtype=torch.float16, device=DEV)
    zz = torch.empty(256, dtype=torch.float32, device=DEV)
    c = torch.ones(256, dtype=torch.bool, device=DEV)
    k_where[(1,)](c, xh, y, zz, 256, Z=tl.float32, BLOCK=256)
    print(f"  tl.where(c, fp16, fp32) -> fp32: {zz[:2].tolist()}")


def P6():
    """and / or / not."""
    x = ints_tensor([0, 1, 2, -1], "int32")
    for op in ("and", "or"):
        try:
            launch_bin(op, x, x, "int32")
            print(f"  int32 {op} int32: accepted")
        except Exception as e:
            print(f"  int32 {op} int32: rejected: {type(e).__name__}: {_first(e)}")
    b1 = ints_tensor([0, 0, 1, 1], "int1")
    b2 = ints_tensor([0, 1, 0, 1], "int1")
    print(f"  int1 and int1: [F,F,T,T] and [F,T,F,T] = {launch_bin('and', b1, b2, 'int1').tolist()}")
    print(f"  int1 or int1: {launch_bin('or', b1, b2, 'int1').tolist()}")
    print(f"  not int1: not [F,F,T,T] = {launch_un('not', b1, 'int1').tolist()}")
    try:
        launch_un("not", x, "int1")
        print("  not int32: accepted")
    except Exception as e:
        print(f"  not int32: rejected: {type(e).__name__}: {_first(e)}")
    print(f"  int32 and True (the constant is dropped; the result is x): [0, 1, 2, -1] -> "
          f"{ints_list(launch_un('and_true', x, 'int32'), 'int32')}")


def P7():
    """min / max: builtin on scalars, integer signedness, NaN and signed zeros."""
    for a, b in ((3, -5), (-2**31, 2**31 - 1), (7, 7)):
        z = torch.empty(1, dtype=torch.int32, device=DEV)
        k_scalar_min[(1,)](a, b, z, OP="builtin_min", Z=tl.int32)
        z2 = torch.empty(1, dtype=torch.int32, device=DEV)
        k_scalar_min[(1,)](a, b, z2, OP="minimum", Z=tl.int32)
        print(f"  builtin min({a}, {b}) on int32 arguments = {z.item()}; tl.minimum = {z2.item()}")
    z = torch.empty(1, dtype=torch.int32, device=DEV)
    k_scalar_min[(1,)](3, -5, z, OP="builtin_max", Z=tl.int32)
    print(f"  builtin max(3, -5) = {z.item()}")
    x = ints_tensor([2**32 - 1, 0, 5], "uint32")
    y = ints_tensor([1, 2**31, 5], "uint32")
    print(f"  tl.minimum on uint32: [2^32-1, 0, 5], [1, 2^31, 5] -> {ints_list(launch_bin('minimum', x, y, 'uint32'), 'uint32')}")
    x = ints_tensor([-1, -2**31, 5], "int32")
    y = ints_tensor([1, 2**31 - 1, 5], "int32")
    print(f"  tl.minimum on int32: [-1, -2^31, 5], [1, 2^31-1, 5] -> {ints_list(launch_bin('minimum', x, y, 'int32'), 'int32')}")
    x = ints_tensor([2**32 - 1, 7], "uint32")
    y = ints_tensor([1, -1], "int32")
    print(f"  tl.minimum(uint32, int32) -> uint32: [2^32-1, 7], [1, -1] -> {ints_list(launch_bin('minimum', x, y, 'uint32'), 'uint32')}")
    xs = ints_tensor(list(range(256)), "uint8")
    print(f"  tl.minimum(uint8 x, -1) -> int32 (the constant keeps its own type): x = 0..255 -> "
          f"{sorted(set(ints_list(launch_un('min_m1', xs, 'int32'), 'int32')))}")
    nan = float("nan")
    xf = torch.tensor([nan, 1.0, nan, -0.0, 0.0, -0.0, 0.0], dtype=torch.float32, device=DEV)
    yf = torch.tensor([1.0, nan, nan, 0.0, -0.0, -0.0, 0.0], dtype=torch.float32, device=DEV)
    for op in ("minimum", "maximum", "minimum_all"):
        z = launch_bin(op, xf, yf, "fp32")
        sg = [("-" if struct.pack("<f", v)[3] & 0x80 else "+") + repr(abs(v)) for v in z.tolist()]
        print(f"  tl.{op.replace('_all', '(propagate_nan=ALL)')} fp32: (nan,1) (1,nan) (nan,nan) (-0,+0) (+0,-0) (-0,-0) (+0,+0) -> {sg}")
    xb = xf.to(torch.bfloat16)
    yb = yf.to(torch.bfloat16)
    z = launch_bin("minimum", xb, yb, "fp32")
    print(f"  tl.minimum(bf16, bf16) -> fp32: {z.tolist()}")


def P8():
    """Comparisons with a Python int constant: the constant keeps its own type (to_tensor first)."""
    xs = ints_tensor(list(range(256)), "uint8")
    print(f"  uint8 x < -1 (as int32), x = 0..255: values {sorted(set(launch_un('lt_m1', xs, 'int1').tolist()))}")
    try:
        launch_un("add_m1", xs, "uint8")
        print("  uint8 x + (-1): accepted")
    except Exception as e:
        print(f"  uint8 x + (-1): rejected: {type(e).__name__}: {_first(e)}")
    xs = ints_tensor(list(range(-128, 128)), "int8")
    print(f"  int8 x < 200 (as int32), x = -128..127: values {sorted(set(launch_un('lt_200', xs, 'int1').tolist()))}")
    xs = ints_tensor([2**32 - 1, 0, 2**31], "uint32")
    print(f"  uint32 x == -1 (as uint32: -1 is 2^32-1): x = [2^32-1, 0, 2^31] -> {launch_un('eq_m1', xs, 'int1').tolist()}")
    xs = ints_tensor([-1, 0, 2**31 - 1, -2**31], "int32")
    print(f"  int32 x < 2**31 (the constant is uint32, so as uint32): x = [-1, 0, 2^31-1, -2^31] -> "
          f"{launch_un('lt_2p31', xs, 'int1').tolist()}")
    xs = ints_tensor([16777217, 16777216, 16777218], "int32")
    print(f"  int32 x == 16777216.0 (as float32): x = [2^24+1, 2^24, 2^24+2] -> {launch_un('eq_f', xs, 'int1').tolist()}")
    xh = torch.tensor([0.0999755859375, 0.10009765625], dtype=torch.float16, device=DEV)
    print(f"  fp16 x < 0.1 (as float32): x = [fp16 0.0999755859375, 0.10009765625] -> {launch_un('lt_01', xh, 'int1').tolist()}")


def P9():
    """Shifts by the width or more, and by a negative amount (a tensor amount)."""
    for d in ("int8", "uint8", "int32", "uint32"):
        w = WIDTH[d]
        xs_v = [1, -1 if signed(d) else rng(d)[1], 5, rng(d)[0] if signed(d) else 1 << (w - 1)]
        amts = list(range(0, w + 3)) + [-1, 2 * w, rng(d)[1]]
        if not signed(d):
            amts = [a for a in amts if a >= 0]
        amts = sorted(set(amts))
        xs = [a for a in xs_v for _ in amts]
        ys = [b for _ in xs_v for b in amts]
        x, y = ints_tensor(xs, d), ints_tensor(ys, d)
        shl = ints_list(launch_bin("shl", x, y, d), d)
        shr = ints_list(launch_bin("shr", x, y, d), d)
        inr = [(a, b, l, r) for a, b, l, r in zip(xs, ys, shl, shr) if 0 <= b < w]
        bad = sum(1 for a, b, l, r in inr if l != wrap(a << b, d) or r != (a >> b if signed(d) else a >> b))
        print(f"  {d}: amounts 0..{w-1}: differs from (x << s mod 2^{w}) and (x >> s, "
              f"{'arithmetic' if signed(d) else 'logical'}) in {bad} of {len(inr)}")
        out = {}
        for a, b, l, r in zip(xs, ys, shl, shr):
            if not 0 <= b < w:
                out.setdefault(b, []).append((a, l, r))
        for b in sorted(out):
            print(f"    shift amount {b}: (x, x << s, x >> s) = {out[b]}")
    x = ints_tensor([-1, -128, 64], "int8")
    y = ints_tensor([1, 1, 1], "uint8")
    print(f"  int8 x >> uint8 y -> uint8 (x converted to uint8; the shift is arithmetic because x's own "
          f"dtype is signed): [-1, -128, 64] >> 1 = {ints_list(launch_bin('shr', x, y, 'uint8'), 'uint8')} "
          f"(logical would be [127, 64, 32])")
    x = ints_tensor([1, 5], "int32")
    print(f"  int32 x << 32 (a constant amount; the front end warns): x = [1, 5] -> "
          f"{ints_list(launch_un('shl_32', x, 'int32'), 'int32')}")


def P10():
    """Unary minus on unsigned and on the minimum; `~` on int1."""
    xs = ints_tensor(list(range(256)), "uint8")
    z = ints_list(launch_un("neg", xs, "uint8"), "uint8")
    bad = sum(1 for a, b in zip(range(256), z) if b != (-a) % 256)
    print(f"  -x on uint8, all 256: differs from (-x) mod 256 in {bad}")
    xs = ints_tensor([-2**31, 5, 0], "int32")
    print(f"  -x on int32 [-2^31, 5, 0] = {ints_list(launch_un('neg', xs, 'int32'), 'int32')}")
    xs = ints_tensor([-128, 127], "int8")
    print(f"  -x on int8 [-128, 127] = {ints_list(launch_un('neg', xs, 'int8'), 'int8')}")
    b = ints_tensor([0, 1], "int1")
    print(f"  ~x on int1 [F, T] = {launch_un('invert', b, 'int1').tolist()}")
    print(f"  -x on int1 [F, T] = {launch_un('neg', b, 'int1').tolist()}")
    xf = torch.tensor([0.0, -0.0, float('nan'), 1.0], dtype=torch.float32, device=DEV)
    z = launch_un("neg", xf, "fp32")
    sg = [("-" if struct.pack("<f", v)[3] & 0x80 else "+") + repr(abs(v)) for v in z.tolist()]
    print(f"  -x on fp32 [+0, -0, nan, 1] = {sg} (sign flipped: fneg)")


def P11():
    """A Python float constant with an integer tensor: the float type wins (float32 here)."""
    vals = [16777217, 16777219, 3, -7, 2**31 - 1, -2**31, 10]
    x = ints_tensor(vals, "int32")
    z = launch_un("mul_01", x, "fp32").tolist()
    ref = [f32(f32(float(v)) * f32(0.1)) for v in vals]
    print(f"  int32 x * 0.1 -> fp32: x = {vals}: device {z}; float32(x) * float32(0.1) rounded: "
          f"{'equal' if z == ref else 'DIFFER ' + str(ref)}")


def _first(e):
    lines = [l for l in str(e).splitlines() if l.strip()]
    for l in lines:
        if "Error" in l and ":" in l:
            return l.strip()[:200]
    return (lines[-1] if lines else "")[:200]


CASES = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6, "P7": P7, "P8": P8, "P9": P9,
         "P10": P10, "P11": P11}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        CASES[sys.argv[2]]()
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, fn in CASES.items():
        print(f"\n{name}: {fn.__doc__}")
        sys.stdout.flush()
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        sys.stdout.write(p.stdout)
        if p.returncode != 0:
            err = [l for l in p.stderr.splitlines() if l.strip()]
            print(f"  (process failed: {err[-1][:200] if err else p.returncode})")


if __name__ == "__main__":
    main()
