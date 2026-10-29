"""evidence/float-layer: the library's float definitions against the device.

The definitions (lean/TVA/Math/Defs.lean) run as native code (build.sh compiles them with
Driver.lean). The device runs the same operation through Triton 3.8.0 on the same inputs. Both
sides reduce the outputs of each chunk of 2^CB consecutive indices to
    h = sum over the chunk of out(i) * (2*i*K + 1) mod 2^64,   K = 0x9E3779B97F4A7C15,
with every NaN output first replaced by the format's canonical NaN (the library's operations are
up to NaN payload). The multiplier is odd, so one differing output always changes h. A chunk whose
hashes differ is dumped on both sides and its differing inputs are listed.

Inputs come from the index (`all`: one pattern, or a pair x = i >> w, y = i & (2^w - 1), so every
pattern or pair once) or from files of records written here from a fixed seed (`file`).

Run from this folder, after `sh build.sh <build folder>`:
    CUDA_VISIBLE_DEVICES=0 python -I run.py <build folder> <test or prefix>... [--jobs N]
`python -I run.py <build folder> --list` lists the tests. One line per test is appended to
results.out (and printed).
"""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import triton
import triton.language as tl

K = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1
C2 = (2 * K) & M64  # below 2^63, a valid int64
CB = 20             # chunk: 2^20 indices
OUT_OF_RANGE = (1 << 63) + 12345  # ftoi of a non-finite or out-of-range input (as Driver.lean)
SLAB = 1 << 26      # indices per device launch

FMT = {"f16": (5, 10, "ieee"), "bf16": (8, 7, "ieee"), "f32": (8, 23, "ieee"),
       "f64": (11, 52, "ieee"), "e5m2": (5, 2, "ieee"), "e4m3": (4, 3, "nanOnly")}
TL = {"f16": tl.float16, "bf16": tl.bfloat16, "f32": tl.float32, "f64": tl.float64,
      "e5m2": tl.float8e5, "e4m3": tl.float8e4nv}
UT = {8: tl.uint8, 16: tl.uint16, 32: tl.uint32, 64: tl.uint64}
ITY = {"s8": tl.int8, "s16": tl.int16, "s32": tl.int32, "s64": tl.int64,
       "u8": tl.uint8, "u16": tl.uint16, "u32": tl.uint32, "u64": tl.uint64}


def width(t):
    return FMT[t][0] + FMT[t][1] + 1 if t in FMT else int(t[1:])


# ---------------------------------------------------------------------------------------------
# device

@triton.jit
def _val(b, SRC: tl.constexpr, W: tl.constexpr, U: tl.constexpr, ISINT: tl.constexpr):
    if ISINT:
        return b.to(SRC)
    else:
        if W == 64:
            return b.to(tl.float64, bitcast=True)
        else:
            return b.to(U).to(SRC, bitcast=True)


@triton.jit
def _bits(r, OW: tl.constexpr, OU: tl.constexpr, OBOOL: tl.constexpr):
    if OBOOL:
        return r.to(tl.int64)
    else:
        if OW == 64:
            return r.to(tl.int64, bitcast=True)
        else:
            return r.to(OU, bitcast=True).to(tl.int64)


@triton.jit
def kernel(out_ptr, xs_ptr, ys_ptr, zs_ptr, lo, n,
           FROMFILE: tl.constexpr, ARITY: tl.constexpr, W: tl.constexpr, SRC: tl.constexpr,
           U: tl.constexpr, ISINT: tl.constexpr, OP: tl.constexpr, DST: tl.constexpr,
           RND: tl.constexpr, OW: tl.constexpr, OU: tl.constexpr, OBOOL: tl.constexpr,
           B: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    m = offs < n
    if FROMFILE:
        xb = tl.load(xs_ptr + offs, mask=m, other=0)
        yb = tl.load(ys_ptr + offs, mask=m, other=0)
        zb = tl.load(zs_ptr + offs, mask=m, other=0)
    else:
        idx = lo + offs
        if ARITY == 1:
            xb = idx
            yb = idx
            zb = idx
        else:
            xb = idx >> W
            yb = idx & ((1 << W) - 1)
            zb = idx
    x = _val(xb, SRC, W, U, ISINT)
    y = _val(yb, SRC, W, U, ISINT)
    z = _val(zb, SRC, W, U, ISINT)
    if OP == "cvt":
        if RND == "rtz":
            r = x.to(DST, fp_downcast_rounding="rtz")
        else:
            r = x.to(DST)
    elif OP == "sqrt":
        r = tl.sqrt_rn(x)
    elif OP == "neg":
        r = -x
    elif OP == "abs":
        r = tl.abs(x)
    elif OP == "add":
        r = x + y
    elif OP == "sub":
        r = x - y
    elif OP == "mul":
        r = x * y
    elif OP == "div":
        if SRC == tl.float32:
            r = tl.div_rn(x, y)
        else:
            r = x / y
    elif OP == "fma":
        r = tl.fma(x, y, z)
    elif OP == "max":
        r = tl.maximum(x, y)
    elif OP == "min":
        r = tl.minimum(x, y)
    elif OP == "maxnan":
        r = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "minnan":
        r = tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "lt":
        r = x < y
    elif OP == "le":
        r = x <= y
    elif OP == "eq":
        r = x == y
    elif OP == "ne":
        r = x != y
    else:
        tl.static_assert(False, "unknown op")
    tl.store(out_ptr + offs, _bits(r, OW, OU, OBOOL), mask=m)


def canon(o, g):
    """NaN outputs of format g replaced by its canonical NaN (as Driver.lean's canon)."""
    e, mb, sp = FMT[g]
    if sp == "ieee":
        nan = (((o >> mb) & ((1 << e) - 1)) == (1 << e) - 1) & ((o & ((1 << mb) - 1)) != 0)
    else:
        nan = (o & ((1 << (e + mb)) - 1)) == (1 << (e + mb)) - 1
    return torch.where(nan, torch.full_like(o, (1 << (e + mb)) - 1), o)


def float_of_bits(b, t):
    """Exact float64 values of patterns of a standard format (torch's own decoding)."""
    if t == "f64":
        return b.view(torch.float64)
    dt = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[t]
    it = {16: torch.int16, 32: torch.int32}[width(t)]
    return b.to(it).view(dt).to(torch.float64)


def is_finite_bits(b, t):
    e, mb, sp = FMT[t]
    mag = b & ((1 << (e + mb)) - 1)
    if sp == "ieee":
        return (mag >> mb) != (1 << e) - 1
    return mag != (1 << (e + mb)) - 1


class Test:
    def __init__(self, name, spec, op, src, dst, arity=1, rnd="rtne", domain="all", gen=None,
                 n=None, post=None, note=""):
        self.name, self.spec, self.op, self.src, self.dst = name, spec, op, src, dst
        self.arity, self.rnd, self.domain, self.gen, self.post, self.note = arity, rnd, domain, gen, post, note
        w = width(src)
        self.n = n if n is not None else (1 << (w * arity))

    def device(self, lo, cnt, files=None):
        """Device outputs (int64) for indices lo .. lo+cnt (from index), or for given records."""
        src, dst = self.src, self.dst
        isint = src not in FMT
        w = width(src)
        obool = self.op in ("lt", "le", "eq", "ne")
        ow = 1 if obool else width(dst)
        out = torch.empty(cnt, dtype=torch.int64, device="cuda")
        if files is not None:
            xs, ys, zs = (torch.from_numpy(a.view(np.int64)).cuda() for a in files)
        else:
            xs = ys = zs = out
        B = 1024
        self.compiled = kernel[(triton.cdiv(cnt, B),)](
            out, xs, ys, zs, lo, cnt, FROMFILE=files is not None, ARITY=self.arity, W=w,
            SRC=ITY[src] if isint else TL[src], U=UT.get(w, tl.uint64), ISINT=isint, OP=self.op,
            DST=ITY[dst] if dst in ITY else TL[dst], RND=self.rnd, OW=ow,
            OU=UT.get(ow, tl.uint64) if not obool else tl.uint8, OBOOL=obool, B=B)
        if not obool and dst in FMT:
            out = canon(out, dst)
        if self.post is not None:
            if files is not None:
                xb = xs
            else:
                xb = torch.arange(lo, lo + cnt, dtype=torch.int64, device="cuda")
                if self.arity == 2:
                    xb = xb >> w
            out = self.post(out, xb)
        return out


def dev_hashes(out, lo):
    """Chunk hashes of outputs `out` of indices lo .. lo+len (lo a multiple of 2^CB)."""
    n = out.numel()
    idx = torch.arange(lo, lo + n, dtype=torch.int64, device=out.device)
    prod = out * (idx * C2 + 1)
    pad = (-n) % (1 << CB)
    if pad:
        prod = torch.cat([prod, torch.zeros(pad, dtype=torch.int64, device=out.device)])
    hs = prod.view(-1, 1 << CB).sum(1).tolist()
    c0 = lo >> CB
    return {c0 + k: h & M64 for k, h in enumerate(hs)}


# ---------------------------------------------------------------------------------------------
# inputs written to files

def specials(t):
    e, mb, sp = FMT[t]
    w = e + mb + 1
    s = 1 << (e + mb)
    emax = (1 << e) - 1
    mags = [0, 1, 2, 3, (1 << mb) - 1, 1 << mb, (1 << mb) + 1, (2 << mb) - 1,
            ((1 << (e - 1)) - 1) << mb,                         # 1
            (((1 << (e - 1)) - 1) << mb) + 1,                   # 1 + ulp
            (((1 << (e - 1)) - 2) << mb) | ((1 << mb) - 1),     # 1 - ulp/2
            ((emax - 1) << mb) | ((1 << mb) - 1),               # maxVal (ieee)
            ((emax - 1) << mb) | ((1 << mb) - 2),
            ((emax - 1) << mb),
            emax << mb, (emax << mb) | 1, (emax << mb) | ((1 << mb) - 1)]
    vals = []
    for mg in mags:
        vals += [mg, mg | s]
    return np.array(vals, dtype=np.uint64)


def rand_float(t, n, rng, near=None):
    """Patterns of format t: a mix of uniform patterns, short significands, values with an exponent
    near `near`'s (a pattern array), and special values."""
    e, mb, sp = FMT[t]
    w = e + mb + 1
    kind = rng.integers(0, 4, n)
    sign = rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(e + mb)
    if w == 64:
        uni = rng.integers(0, 1 << 63, n, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, n, dtype=np.uint64)
    else:
        uni = rng.integers(0, 1 << w, n, dtype=np.uint64)
    expo = rng.integers(0, (1 << e), n, dtype=np.uint64)
    frac = rng.integers(0, 1 << mb, n, dtype=np.uint64) if mb < 63 else rng.integers(0, 1 << 62, n, dtype=np.uint64)
    keep = rng.integers(0, mb + 1, n, dtype=np.uint64)
    short = (frac >> (np.uint64(mb) - keep)) << (np.uint64(mb) - keep)
    if near is not None:
        ne = (near >> np.uint64(mb)) & np.uint64((1 << e) - 1)
        d = rng.integers(-(mb + 3), mb + 4, n)
        ex2 = np.clip(ne.astype(np.int64) + d, 0, (1 << e) - 1).astype(np.uint64)
        expo = np.where(rng.integers(0, 2, n) == 1, ex2, expo)
    struct = sign | (expo << np.uint64(mb)) | np.where(kind == 1, short, frac)
    sp_vals = specials(t)
    spec = sp_vals[rng.integers(0, len(sp_vals), n)]
    return np.where(kind == 0, uni, np.where(kind == 3, spec, struct)).astype(np.uint64)


def rand_int(t, n, rng):
    """Integers of type t with a random number of significant bits, either sign."""
    w = int(t[1:])
    v = (rng.integers(0, 1 << 32, n, dtype=np.uint64) << np.uint64(32)) | rng.integers(0, 1 << 32, n, dtype=np.uint64)
    k = rng.integers(1, w + 1, n).astype(np.uint64)
    v = v & (np.uint64(M64) >> (np.uint64(64) - k))
    neg = (rng.integers(0, 2, n) == 1) & (t[0] == "s")
    mask = np.uint64(M64) >> np.uint64(64 - w)
    return np.where(neg, (~v + np.uint64(1)) & mask, v).astype(np.uint64)


def gen_records(test, n, seed):
    rng = np.random.default_rng(seed)
    src = test.src
    if src in FMT:
        x = rand_float(src, n, rng)
        y = rand_float(src, n, rng, near=x)
        z = rand_float(src, n, rng, near=x)
    else:
        x = rand_int(src, n, rng)
        y = z = np.zeros(n, dtype=np.uint64)
    return x, y, z


# ---------------------------------------------------------------------------------------------
# the Lean side

def lean_hashes_all(driver, test, jobs):
    T = 1 << (CB + 3)
    tasks = [(lo, min(test.n, lo + T)) for lo in range(0, test.n, T)]

    def one(t):
        r = subprocess.run(["nice", "-n", "10", driver, test.spec, "all", str(t[0]), str(t[1]), str(CB)],
                           capture_output=True, text=True, check=True)
        return r.stdout

    hs = {}
    with ThreadPoolExecutor(jobs) as ex:
        for out in ex.map(one, tasks):
            for line in out.splitlines():
                c, h = line.split()
                hs[int(c)] = (hs.get(int(c), 0) + int(h)) & M64
    return hs


def lean_hashes_files(driver, test, paths, jobs):
    def one(pb):
        p, base = pb
        r = subprocess.run(["nice", "-n", "10", driver, test.spec, "file", p, str(base), str(CB)],
                           capture_output=True, text=True, check=True)
        return r.stdout

    hs = {}
    with ThreadPoolExecutor(jobs) as ex:
        for out in ex.map(one, paths):
            for line in out.splitlines():
                c, h = line.split()
                hs[int(c)] = (hs.get(int(c), 0) + int(h)) & M64
    return hs


def lean_dump(driver, test, lo, hi, path=None, base=None):
    if path is None:
        args = [driver, test.spec, "dump-all", str(lo), str(hi)]
    else:
        args = [driver, test.spec, "dump-file", path, str(base)]
    r = subprocess.run(args, capture_output=True, text=True, check=True)
    return {int(a): int(b, 16) for a, b in (l.split() for l in r.stdout.splitlines())}


# ---------------------------------------------------------------------------------------------

def ftoi_post(t, w, sg):
    # the range as lo <= trunc(x) < top, both powers of two (or 0), exact in float64
    lo, top = (-(1 << (w - 1)), 1 << (w - 1)) if sg else (0, 1 << w)

    def post(out, xb):
        v = float_of_bits(xb, t)
        tr = torch.trunc(v)
        ok = torch.isfinite(v) & (tr >= float(lo)) & (tr < float(top))
        return torch.where(ok, out, torch.full_like(out, OUT_OF_RANGE - (1 << 64)))
    return post


def fin_post(t, g):
    def post(out, xb):
        return torch.where(is_finite_bits(xb, t), out, torch.full_like(out, 1 << width(g)))
    return post


def tests():
    T = []
    # A: decoding and conversions
    for f in ("f16", "bf16"):
        for g in ("f32", "f64"):
            T.append(Test(f"A.cvt.{f}.{g}", f"cvt:{f}:{g}:rn:0", "cvt", f, g))
        h = "bf16" if f == "f16" else "f16"
        T.append(Test(f"A.cvt.{f}.{h}", f"cvt:{f}:{h}:rn:0", "cvt", f, h))
        for g in ("e4m3", "e5m2"):
            for sat in (1, 0):
                T.append(Test(f"A.cvt.{f}.{g}.sat{sat}", f"cvt:{f}:{g}:rn:{sat}", "cvt", f, g))
    for e in ("e5m2", "e4m3"):
        for g in ("f16", "bf16", "f32"):
            T.append(Test(f"A.cvt.{e}.{g}", f"cvtfin:{e}:{g}", "cvt", e, g, post=fin_post(e, g),
                          note="finite inputs"))
    T.append(Test("A.cvt.f32.f64", "cvt:f32:f64:rn:0", "cvt", "f32", "f64"))
    for g in ("f16", "bf16"):
        T.append(Test(f"A.cvt.f32.{g}.rn", f"cvt:f32:{g}:rn:0", "cvt", "f32", g))
        T.append(Test(f"A.cvt.f32.{g}.rz", f"cvt:f32:{g}:rz:0", "cvt", "f32", g, rnd="rtz"))
    for g in ("e4m3", "e5m2"):
        T.append(Test(f"A.cvt.f32.{g}.sat1", f"cvt:f32:{g}:rn:1", "cvt", "f32", g))
    for sat in (1, 0):
        T.append(Test(f"A.cvt.f32.e5m2.rz.sat{sat}", f"cvt:f32:e5m2:rz:{sat}", "cvt", "f32", "e5m2", rnd="rtz"))
    for g in ("f32", "f16", "bf16"):
        T.append(Test(f"A.cvt.f64.{g}", f"cvt:f64:{g}:rn:0", "cvt", "f64", g, domain="file", n=1 << 22))
    for f in ("e4m3", "e5m2", "f32"):
        for g in ("e4m3", "e5m2"):
            if f in ("f32",):
                T.append(Test(f"A.cvt.{f}.{g}.sat0.sample", f"cvt:{f}:{g}:rn:0", "cvt", f, g,
                              domain="file", n=1 << 22))
    # B: arithmetic
    for f in ("f16", "bf16"):
        for op in ("add", "sub", "mul"):
            T.append(Test(f"B.{op}.{f}", f"{op}:{f}", op, f, f, arity=2))
        T.append(Test(f"B.fma.{f}", f"fma:{f}", "fma", f, f, arity=3, domain="file", n=1 << 22))
    for op in ("add", "sub", "mul", "div", "fma"):
        T.append(Test(f"B.{op}.f32", f"{op}:f32", op, "f32", "f32", arity=3 if op == "fma" else 2,
                      domain="file", n=1 << 24))
        T.append(Test(f"B.{op}.f64", f"{op}:f64", op, "f64", "f64", arity=3 if op == "fma" else 2,
                      domain="file", n=1 << 22))
    T.append(Test("B.sqrt.f32", "sqrt:f32", "sqrt", "f32", "f32"))
    for op in ("lt", "le", "eq", "ne", "max", "min", "maxnan", "minnan"):
        T.append(Test(f"B.{op}.f16", f"{op}:f16", op, "f16", "f16" if "m" in op else "u8", arity=2))
        T.append(Test(f"B.{op}.f32", f"{op}:f32", op, "f32", "f32" if "m" in op else "u8", arity=2,
                      domain="file", n=1 << 24))
        T.append(Test(f"B.{op}.f64", f"{op}:f64", op, "f64", "f64" if "m" in op else "u8", arity=2,
                      domain="file", n=1 << 22))
    # C: sign operations
    for f in ("f16", "bf16", "f32"):
        for op in ("neg", "abs"):
            T.append(Test(f"C.{op}.{f}", f"{op}:{f}", op, f, f))
    for op in ("neg", "abs"):
        T.append(Test(f"C.{op}.f64", f"{op}:f64", op, "f64", "f64", domain="file", n=1 << 22))
    # D: integers
    for t in ("s8", "u8", "s16", "u16"):
        for g in ("f16", "bf16", "f32"):
            T.append(Test(f"D.itof.{t}.{g}", f"itof:{t}:{g}", "cvt", t, g))
    for t in ("s32", "u32"):
        T.append(Test(f"D.itof.{t}.f32", f"itof:{t}:f32", "cvt", t, "f32"))
        for g in ("f16", "bf16", "f64"):
            T.append(Test(f"D.itof.{t}.{g}", f"itof:{t}:{g}", "cvt", t, g, domain="file", n=1 << 24))
    for t in ("s64", "u64"):
        for g in ("f16", "bf16", "f32", "f64"):
            T.append(Test(f"D.itof.{t}.{g}", f"itof:{t}:{g}", "cvt", t, g, domain="file", n=1 << 22))
    for f in ("f16", "bf16"):
        for t in ("s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64"):
            T.append(Test(f"D.ftoi.{f}.{t}", f"ftoi:{f}:{t}", "cvt", f, t,
                          post=ftoi_post(f, int(t[1:]), t[0] == "s"), note="in-range inputs"))
    for t in ("s32", "u32"):
        T.append(Test(f"D.ftoi.f32.{t}", f"ftoi:f32:{t}", "cvt", "f32", t,
                      post=ftoi_post("f32", 32, t[0] == "s"), note="in-range inputs"))
    for t in ("s64", "u64"):
        T.append(Test(f"D.ftoi.f32.{t}", f"ftoi:f32:{t}", "cvt", "f32", t, domain="file", n=1 << 24,
                      post=ftoi_post("f32", 64, t[0] == "s"), note="in-range inputs"))
        T.append(Test(f"D.ftoi.f64.{t}", f"ftoi:f64:{t}", "cvt", "f64", t, domain="file", n=1 << 22,
                      post=ftoi_post("f64", 64, t[0] == "s"), note="in-range inputs"))
    # a structured sample (2^26 inputs from files) of each test over 2^32 inputs
    for t in list(T):
        if t.domain == "all" and t.n == 1 << 32:
            u = Test(t.name + ".sample", t.spec, t.op, t.src, t.dst, arity=t.arity, rnd=t.rnd,
                     domain="file", n=1 << 26, post=t.post, note=t.note)
            T.append(u)
    return {t.name: t for t in T}


def run_test(driver, test, jobs, work):
    t0 = time.time()
    dev = {}
    files = []
    if test.domain == "all":
        for lo in range(0, test.n, SLAB):
            dev.update(dev_hashes(test.device(lo, min(SLAB, test.n - lo)), lo))
        lean = lean_hashes_all(driver, test, jobs)
    else:
        seed = int.from_bytes(test.name.encode()[:8].ljust(8, b"\0"), "little")
        per = 1 << CB
        for k, base in enumerate(range(0, test.n, per)):
            x, y, z = gen_records(test, min(per, test.n - base), (seed, k))
            p = os.path.join(work, f"{test.name}.{k}.bin")
            np.stack([x, y, z], axis=1).astype("<u8").tofile(p)
            files.append((p, base))
            dev.update(dev_hashes(test.device(0, len(x), files=(x, y, z)), base))
        lean = lean_hashes_files(driver, test, files, jobs)
    bad = sorted(c for c in dev if dev[c] != lean.get(c))
    missing = sorted(set(dev) ^ set(lean))
    lines = []
    nbad = 0
    for c in bad[:4]:
        lo, hi = c << CB, min(test.n, (c + 1) << CB)
        if test.domain == "all":
            lv = lean_dump(driver, test, lo, hi)
            dv = test.device(lo, hi - lo).tolist()
        else:
            p, base = files[c]
            lv = lean_dump(driver, test, 0, 0, path=p, base=base)
            rec = np.fromfile(p, dtype="<u8").reshape(-1, 3)
            dv = test.device(0, len(rec), files=(rec[:, 0].copy(), rec[:, 1].copy(), rec[:, 2].copy())).tolist()
        diffs = [i for i in range(lo, hi) if (lv[i] & M64) != (dv[i - lo] & M64)]
        nbad += len(diffs)
        if test.dst in FMT:
            e, mb, sp = FMT[test.dst]
            mx = ((1 << e) - 1 << mb) - 1 if sp == "ieee" else (1 << (e + mb)) - 2
            sb = 1 << (e + mb)
            nf = lambda v: (sp == "ieee" and (v % sb) >> mb == (1 << e) - 1) or (sp != "ieee" and v % sb == sb - 1)
            ovf = sum(1 for i in diffs if nf(lv[i]) and (dv[i - lo] & M64) % sb == mx)
            if ovf:
                lines.append(f"    chunk {c}: {len(diffs)} inputs differ, {ovf} of them where the library "
                             f"gives an infinity or NaN and the device the largest finite magnitude")
        for i in diffs[:6]:
            if test.domain == "all":
                w = width(test.src)
                inp = f"{i:x}" if test.arity == 1 else f"{i >> w:x},{i & ((1 << w) - 1):x}"
            else:
                inp = ",".join(f"{v:x}" for v in rec[i - lo][:test.arity])
            lines.append(f"    input {inp}: library {lv[i]:x}, device {dv[i - lo] & M64:x}")
    for p, _ in files:
        os.remove(p)
    verdict = "agree" if not bad and not missing else f"DIFFER in {len(bad)} chunks ({nbad}+ inputs)"
    head = (f"{test.name:28s} {test.spec:24s} {test.domain:4s} {test.n:>12d} inputs  {verdict}"
            f"{'  (' + test.note + ')' if test.note else ''}  [{time.time() - t0:.0f} s]")
    return "\n".join([head] + lines)


def main():
    args = sys.argv[1:]
    jobs = 40
    if "--jobs" in args:
        k = args.index("--jobs")
        jobs = int(args[k + 1])
        del args[k:k + 2]
    build = args[0]
    driver = os.path.join(os.path.abspath(build), "driver")
    T = tests()
    if "--list" in args:
        for t in T.values():
            print(f"{t.name:28s} {t.spec:24s} {t.domain:4s} {t.n:>12d}")
        return
    if "--ptx" in args:
        # the floating-point and conversion instructions of each test's kernel, as compiled
        pat = re.compile(r"^\s*((?:add|sub|mul|fma|div|sqrt|rcp|neg|abs|max|min|setp|selp|cvt|testp|and|or|prmt|shl|shr)"
                         r"\.[\w.]+)")
        with open("ptx.out", "w") as out:
            for n in T:
                if not any(n == a or n.startswith(a) for a in args[1:] if a != "--ptx"):
                    continue
                t = T[n]
                try:
                    if t.domain == "all":
                        t.device(0, 1024)
                    else:
                        z = np.zeros(1024, dtype=np.uint64)
                        t.device(0, 1024, files=(z, z, z))
                    ins = sorted({m.group(1) for l in t.compiled.asm["ptx"].splitlines() if (m := pat.match(l))})
                    ins = [i for i in ins if not re.match(r"(add|mul|shl|shr|and|or|setp\.\w+)\.(s|u|b)(32|64)$", i)]
                    line = f"{n:28s} {' '.join(ins)}"
                except Exception as ex:
                    line = f"{n:28s} does not compile: {type(ex).__name__}"
                print(line, flush=True)
                out.write(line + "\n")
        return
    names = []
    for a in args[1:]:  # in the order given; a prefix selects its tests in definition order
        names += [n for n in ([a] if a in T else [n for n in T if n.startswith(a)]) if n not in names]
    work = os.path.join(os.path.abspath(build), "work")
    os.makedirs(work, exist_ok=True)
    with open("results.out", "a") as out:
        for n in names:
            try:
                line = run_test(driver, T[n], jobs, work)
            except Exception as ex:  # a conversion Triton does not compile, for one
                msg = str(ex).strip().splitlines()[-1] if str(ex).strip() else type(ex).__name__
                line = f"{n:28s} {T[n].spec:24s} {T[n].domain:4s} {T[n].n:>12d} inputs  NOT RUN: {type(ex).__name__}: {msg[:120]}"
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()


if __name__ == "__main__":
    main()
