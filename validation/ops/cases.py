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
                  ("f32", "f8e4nv"), ("f32", "f8e5"), ("f16", "f8e4nv"), ("i32", "i8"), ("f32", "i1")):
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
    for M, N, K in ((16, 16, 16), (32, 16, 64)):
        a, b, c = rand_bits("f16", M * K), rand_bits("f16", K * N), small_ints("f32", M * N)
        out.append(case(f"dot_ieee.f16.{M}x{N}x{K}", "dot_ieee", ["f16", "f16", "f32"], [M, N, K], [a, b, c]))
        out.append(case(f"dot_add_ieee.f16.{M}x{N}x{K}", "dot_add_ieee", ["f16", "f16", "f32"], [M, N, K], [a, b, c]))
    return out

if __name__ == "__main__":
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), "cases.json")
    cs = cases()
    json.dump(cs, open(path, "w"))
    print(f"{path}: {len(cs)} cases, {sum(len(c['inputs'][0]) for c in cs)} lanes")
