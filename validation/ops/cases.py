#!/usr/bin/env python3
"""Write the single-operation cases (cases.json) for Ops.lean and device.py.

  cases.py [OUT.json] [--small]

A case is one tl.* operation on one dtype, with inputs as bit patterns: the edge values of the
dtype (zeros, subnormals, extremes, infinities, NaNs with payloads; for integers 0, ±1, the
extremes, shift amounts around the width), all pairs of them for binary operations, and random
values. Lane counts are powers of two (padded with the first value), as tl.arange needs.
--small: fewer random values (for the interpreter while developing). Standard library only."""
import json, random, struct, sys

R = random.Random(20261113)
SMALL = "--small" in sys.argv
NRAND = 64 if SMALL else 1024

def f32(x): return struct.unpack("<I", struct.pack("<f", x))[0]

# Edge bit patterns per float format: (exponent bits, mantissa bits).
FMT = {"f16": (5, 10), "bf16": (8, 7), "f32": (8, 23), "f64": (11, 52)}

def float_edges(d):
    e, m = FMT[d]
    w = 1 + e + m; sign = 1 << (w - 1); emax = (1 << e) - 1
    one = ((1 << (e - 1)) - 1) << m
    pos = [0, 1, 2, (1 << m) - 1, 1 << m, (1 << m) + 1,     # 0, subnormals, smallest normals
           one, one + 1, one - 1, one + (1 << (m - 1)),         # 1, 1 ± ulp, 1.5
           one + (1 << m), one - (1 << m),                      # 2, 0.5
           (emax << m) - 1, (emax - 1) << m,                    # largest finite, a large power of 2
           emax << m,                                           # +inf
           (emax << m) | (1 << (m - 1)), (emax << m) | 1,       # quiet NaN, signalling NaN (payload 1)
           (emax << m) | ((1 << m) - 1)]                        # NaN, all-ones payload
    if d == "f32":
        pos += [f32(3.0), f32(1 / 3), f32(0.1), f32(88.72283935546875), f32(1e-40)]
    return pos + [p | sign for p in pos]

def int_edges(d):
    sg = d[0] == "i"; w = int(d[1:])
    vals = [0, 1, 2, 3, 7, w - 1, w, w + 1, (1 << (w - 1)) - 1, 1 << (w - 1), (1 << w) - 1,
            (1 << w) - 2, (1 << (w - 1)) + 1]
    if sg:
        vals += [(1 << w) - 3, (1 << w) - 7]                   # -3, -7
    return sorted(set(v % (1 << w) for v in vals))

def width(d): return {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}.get(d) or int(d[1:])

def rand_bits(d, k):
    w = width(d)
    return [R.getrandbits(w) for _ in range(k)]

def pad(cols, gens):
    """Pad each column to a power of two with fresh values from its generator."""
    n = len(cols[0]); p = 1
    while p < n: p *= 2
    return [c + g(p - n) for c, g in zip(cols, gens)], p

def case(id, op, dtypes, dims, inputs):
    return {"id": id, "op": op, "dtypes": dtypes, "dims": dims, "inputs": inputs}

def unary(op, d):
    xs = float_edges(d) + rand_bits(d, NRAND)
    if d in ("f16", "bf16") and not SMALL:
        xs = list(range(1 << 16))                               # every pattern
    (xs,), n = pad([xs], [lambda k: rand_bits(d, k)])
    return case(f"{op}.{d}", op, [d], [n], [xs])

def shift_amounts(d, k):
    return [R.randrange(width(d)) for _ in range(k)]

def binary(op, d, edges):
    es = edges(d)
    ry = (lambda k: shift_amounts(d, k)) if op in ("shl", "ashr", "lshr") else (lambda k: rand_bits(d, k))
    rx = lambda k: rand_bits(d, k)
    xs = [a for a in es for b in es] + rx(NRAND)
    ys = [b for a in es for b in es] + ry(NRAND)
    (xs, ys), n = pad([xs, ys], [rx, ry])
    return case(f"{op}.{d}", op, [d], [n], [xs, ys])

def ternary(op, d):
    es = float_edges(d)[:12] + float_edges(d)[18:30]
    t = [(a, b, c) for a in es for b in es for c in es[::3]]
    t += [tuple(rand_bits(d, 3)) for _ in range(NRAND)]
    rx = lambda k: rand_bits(d, k)
    (xs, ys, zs), n = pad([[a for a, _, _ in t], [b for _, b, _ in t], [c for _, _, c in t]], [rx, rx, rx])
    return case(f"{op}.{d}", op, [d], [n], [xs, ys, zs])

def small_ints(d, k):
    """Floats with small integer values (exact sums in any order)."""
    e, m = FMT[d]; bias = (1 << (e - 1)) - 1
    def enc(v):
        if v == 0: return 0
        s = 1 << (1 + e + m - 1) if v < 0 else 0; v = abs(v)
        ex = v.bit_length() - 1
        return s | ((ex + bias) << m) | ((v - (1 << ex)) << (m - ex) if ex <= m else 0)
    return [enc(R.randint(-8, 8)) for _ in range(k)]

def moderate(d, k):
    """Random finite floats of moderate size (`[-4, 4]`, full mantissas): rounding happens."""
    def enc(v):
        if d == "f32": return f32(v)
        if d == "f16": return struct.unpack("<H", struct.pack("<e", v))[0]
        return f32(v) >> 16                                      # bf16: truncated f32
    return [enc(R.uniform(-4, 4)) for _ in range(k)]

def pow2s(d, k, choices):
    """Floats drawn from `choices` (powers of two and their negatives): exact products."""
    def enc(v):
        if d == "f32": return f32(v)
        if d == "f16": return struct.unpack("<H", struct.pack("<e", v))[0]
        return f32(v) >> 16
    return [enc(R.choice(choices)) for _ in range(k)]

SCAN_OUT = {"bf16": "f32", "i8": "i32", "u8": "u32"}             # tl.cumsum's promotions

def scans():
    """Scans. Axes of at most 8 elements: random values, every tree searched (Ops.lean). Longer
    axes: values whose partial sums or products are exact, so that every tree agrees."""
    out = []
    def add(op, d, dims, xs, w=4):
        d2 = SCAN_OUT.get(d, d) if op.startswith("cumsum") else ("f32" if d == "bf16" else d)
        if op.startswith(("cummax", "cummin")): d2 = d
        c = case(f"{op}.{d}.{'x'.join(map(str, dims))}.w{w}", op, [d, d2], dims, [xs])
        c["num_warps"] = w; out.append(c)
    for kind in ("cumsum", "cumprod"):
        for rev in ("", "_rev"):
            for d in ("f32", "f16", "bf16"):
                for n in (4, 8):
                    for w in (1, 4):
                        add(kind + rev, d, [n], moderate(d, n), w)
                for ax, (m, n) in (("", (8, 4)), ("_ax1", (4, 8))):
                    add(kind + ax + rev, d, [m, n], moderate(d, m * n))
            for n in (64, 1024):
                xs = small_ints("f32", n) if kind == "cumsum" else pow2s("f32", n, [1, -1, 2, 0.5])
                add(kind + rev, "f32", [n], xs, 8)
            add(kind + "_ax1" + rev, "f16", [4, 64],
                small_ints("f16", 256) if kind == "cumsum" else pow2s("f16", 256, [1, -1]))
            for d in ("i32", "i8", "u8", "i64"):
                add(kind + rev, d, [64], rand_bits(d, 64))
                add(kind + "_ax1" + rev, d, [4, 16], rand_bits(d, 64))
    # Running maxima and minima (`tl.associative_scan` with `tl.maximum` / `tl.minimum`): exact,
    # so any tree agrees, and longer axes need no search. Moderate values with infinities and
    # NaNs, no zeros (`MaxRel` lets either zero of a `-0`/`+0` pair win).
    special = {"f32": [0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001],
               "f16": [0x7c00, 0xfc00, 0x7e00, 0xfe01]}
    def mixed(d, k):
        return [R.choice(special[d]) if R.random() < 0.2 else v for v in moderate(d, k)]
    for kind in ("cummax", "cummin"):
        for d in ("f32", "f16"):
            for rev in ("", "_rev"):
                add(kind + rev, d, [8], mixed(d, 8), 1)
                add(kind + rev, d, [1024], mixed(d, 1024), 8)
            add(kind + "_ax1", d, [4, 64], mixed(d, 256))
    return out

def cases():
    out = []
    for d in ("f16", "bf16", "f32"):
        for op in ("add", "sub", "mul", "div_rn", "maximum", "minimum", "div"):
            if op == "div" and d != "f32": continue
            out.append(binary(op, d, float_edges))
        for op in ("neg", "abs", "sqrt_rn"):
            out.append(unary(op, d))
        out.append(ternary("fma", d))
    for op in ("exp", "exp2", "log", "log2", "sqrt", "rsqrt"):
        out.append(unary(op, "f32"))
    for d in ("i8", "i32", "u32", "i64"):
        for op in ("idiv", "irem", "shl"):
            out.append(binary(op, d, int_edges))
        out.append(binary("ashr" if d[0] == "i" else "lshr", d, int_edges))
    for d, d2 in (("f32", "f16"), ("f32", "bf16"), ("f16", "f32"), ("f32", "i32"), ("i32", "f32"),
                  ("f32", "f8e4nv"), ("f32", "f8e5"), ("f16", "f8e4nv"), ("i32", "i8"),
                  ("f32", "i1"),
                  ("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("i32", "i32"),
                  ("f8e5", "f32"), ("f8e4nv", "f16")):
        if d.startswith("f8"):
            xs, n = list(range(256)), 256                        # every fp8 pattern
        else:
            xs = (float_edges(d) if d in FMT else int_edges(d)) + rand_bits(d, NRAND)
            (xs,), n = pad([xs], [lambda k, d=d: rand_bits(d, k)])
        out.append(case(f"convert.{d}.{d2}", "convert", [d, d2], [n], [xs]))
    # Reductions: random values (rounding depends on the tree) and small integers (it does not).
    for d in ("f32", "f16"):
        for n in (8, 64, 1024):
            xs = rand_bits(d, n)
            for w in (1, 4, 8):                                  # num_warps changes the tree
                c = case(f"sum.{d}.{n}.w{w}", "sum", [d], [n], [xs]); c["num_warps"] = w
                out.append(c)
    out += scans()
    for M, N, K in ((16, 16, 16), (32, 16, 64)):
        a, b, c = rand_bits("f16", M * K), rand_bits("f16", K * N), small_ints("f32", M * N)
        out.append(case(f"dot_ieee.f16.{M}x{N}x{K}", "dot_ieee", ["f16", "f16", "f32"], [M, N, K], [a, b, c]))
        out.append(case(f"dot_add_ieee.f16.{M}x{N}x{K}", "dot_add_ieee", ["f16", "f16", "f32"], [M, N, K], [a, b, c]))
    return out

if __name__ == "__main__":
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), "cases.json")
    cs = cases()
    with open(path, "w") as f:
        json.dump(cs, f)
    print(f"{path}: {len(cs)} cases, {sum(len(c['inputs'][0]) for c in cs)} lanes")
