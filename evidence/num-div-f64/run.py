"""evidence/num-div-f64: `/` on float64 against the library's IEEE division.

The library states `DivRn P .binary64`: every result of `x / y` on `float64` is
`FloatV.divRn x y` (`Format.div .rn`, IEEE division, round to nearest even) up to NaN payload.
The reference is that definition itself: evidence/float-layer's driver (lean/TVA/Math/Defs.lean
compiled to native code, spec `div:f64`). The device runs `x / y` (and `tl.fdiv(x, y)`, and
`x / c` for constant divisors `c`) through Triton 3.8.0 on the same inputs. Both sides reduce the
outputs of each chunk of 2^20 records to
    h = sum over the chunk of out(i) * (2*i*K + 1) mod 2^64,   K = 0x9E3779B97F4A7C15,
NaN outputs replaced by the canonical NaN first. A chunk whose hashes differ is dumped on both
sides and its differing inputs are listed.

Run from this folder, after building the driver (from evidence/float-layer: sh build.sh <B>):
    CUDA_VISIBLE_DEVICES=0 python -I run.py <B> [--jobs N] [--quick]
Output: out.txt (one line per family, then the PTX division instructions of each kernel); the
time per family goes to standard error. `--quick` is a smoke test (2^20 records per family at
most, into <B>/quick.txt).
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import torch
import triton
import triton.language as tl

K = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1
C2 = (2 * K) & M64
CB = 20
PER = 1 << CB
SIGN = np.uint64(1 << 63)
FRAC = np.uint64((1 << 52) - 1)
NAN = (1 << 63) - 1  # the canonical NaN magnitude of binary64 (Format.nanMag)


@triton.jit
def div_kernel(out_ptr, xs_ptr, ys_ptr, n, MODE: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    m = offs < n
    x = tl.load(xs_ptr + offs, mask=m, other=0).to(tl.float64, bitcast=True)
    y = tl.load(ys_ptr + offs, mask=m, other=0).to(tl.float64, bitcast=True)
    if MODE == "div":
        r = x / y
    elif MODE == "fdiv":
        r = tl.fdiv(x, y)
    else:
        r = x / C
    tl.store(out_ptr + offs, r.to(tl.int64, bitcast=True), mask=m)


COMPILED = {}


def device(x, y, mode="div", c=0.0):
    n = len(x)
    xs = torch.from_numpy(x.view(np.int64)).cuda()
    ys = torch.from_numpy(y.view(np.int64)).cuda()
    out = torch.empty(n, dtype=torch.int64, device="cuda")
    B = 1024
    h = div_kernel[(triton.cdiv(n, B),)](out, xs, ys, n, MODE=mode, C=c, B=B)
    COMPILED[(mode, c)] = h
    # canonical NaN: exponent all ones and a nonzero fraction
    e = (out >> 52) & 0x7FF
    f = out & ((1 << 52) - 1)
    return torch.where((e == 0x7FF) & (f != 0), torch.full_like(out, NAN), out)


def dev_hashes(out, base):
    n = out.numel()
    idx = torch.arange(base, base + n, dtype=torch.int64, device=out.device)
    prod = out * (idx * C2 + 1)
    pad = (-n) % PER
    if pad:
        prod = torch.cat([prod, torch.zeros(pad, dtype=torch.int64, device=out.device)])
    hs = prod.view(-1, PER).sum(1).tolist()
    return {(base >> CB) + k: h & M64 for k, h in enumerate(hs)}


def lean_hashes(driver, paths, jobs):
    def one(pb):
        p, base = pb
        r = subprocess.run(["nice", "-n", "10", driver, "div:f64", "file", p, str(base), str(CB)],
                           capture_output=True, text=True, check=True)
        return r.stdout

    hs = {}
    with ThreadPoolExecutor(jobs) as ex:
        for out in ex.map(one, paths):
            for line in out.splitlines():
                c, h = line.split()
                hs[int(c)] = (hs.get(int(c), 0) + int(h)) & M64
    return hs


def lean_dump(driver, path, base):
    r = subprocess.run([driver, "div:f64", "dump-file", path, str(base)], capture_output=True, text=True, check=True)
    return {int(a): int(b, 16) for a, b in (l.split() for l in r.stdout.splitlines())}


# ---------------------------------------------------------------------------------------------
# input families (bit patterns of float64, as uint64)

def pat(sign, expo, frac):
    return (sign.astype(np.uint64) << np.uint64(63)) | (expo.astype(np.uint64) << np.uint64(52)) | (frac.astype(np.uint64) & FRAC)


def rbits(rng, n):
    return (rng.integers(0, 1 << 32, n, dtype=np.uint64) << np.uint64(32)) | rng.integers(0, 1 << 32, n, dtype=np.uint64)


def rfrac(rng, n):
    """Random fractions: uniform, or with only their top bits set (short significands)."""
    f = rbits(rng, n) & FRAC
    keep = rng.integers(0, 53, n).astype(np.uint64)
    short = (f >> (np.uint64(52) - keep)) << (np.uint64(52) - keep)
    return np.where(rng.integers(0, 2, n) == 1, short, f)


def rsign(rng, n):
    return rng.integers(0, 2, n)


def specials():
    mags = [0, 1, 2, 3, (1 << 52) - 1, 1 << 52, (1 << 52) + 1, (2 << 52) - 1,
            1023 << 52, (1023 << 52) + 1, (1022 << 52) | ((1 << 52) - 1), 1022 << 52,
            (2046 << 52) | ((1 << 52) - 1), (2046 << 52) | ((1 << 52) - 2), 2046 << 52,
            2047 << 52, (2047 << 52) | 1, (2047 << 52) | ((1 << 52) - 1)]
    vals = []
    for m in mags:
        vals += [m, m | (1 << 63)]
    return np.array(vals, dtype=np.uint64)


def fam_uniform(rng, n):
    return rbits(rng, n), rbits(rng, n)


def fam_struct(rng, n):
    """As evidence/float-layer's records: a quarter uniform patterns, a quarter short significands,
    a quarter structured (random exponents, the divisor's near the dividend's), a quarter specials."""
    sp = specials()
    out = []
    xe = rng.integers(0, 2048, n)
    for which in range(2):
        kind = rng.integers(0, 4, n)
        e = xe if which == 0 else np.clip(xe + rng.integers(-56, 57, n), 0, 2047)
        e = np.where(rng.integers(0, 2, n) == 1, e, rng.integers(0, 2048, n))
        f = rbits(rng, n) & FRAC
        keep = rng.integers(0, 53, n).astype(np.uint64)
        short = (f >> (np.uint64(52) - keep)) << (np.uint64(52) - keep)
        struct = pat(rsign(rng, n), e, np.where(kind == 1, short, f))
        out.append(np.where(kind == 0, rbits(rng, n), np.where(kind == 3, sp[rng.integers(0, len(sp), n)], struct)))
    return out[0], out[1]


def fam_near1(rng, n):
    """Divisors within 2^20 ulps of 1 (both sides of 1, and of -1); dividends finite, any exponent."""
    k = rng.integers(-(1 << 20), (1 << 20) + 1, n)
    one = np.int64(1023 << 52)
    y = (one + k).astype(np.uint64) | (rsign(rng, n).astype(np.uint64) << np.uint64(63))
    x = pat(rsign(rng, n), rng.integers(0, 2047, n), rfrac(rng, n))
    return x, y


def fam_binade1(rng, n):
    """Divisors in [1/2, 2) with random significands; dividends finite, any exponent."""
    y = pat(rsign(rng, n), rng.integers(1022, 1024, n), rfrac(rng, n))
    x = pat(rsign(rng, n), rng.integers(0, 2047, n), rfrac(rng, n))
    return x, y


def fam_subnormal_dividend(rng, n):
    """Subnormal dividends; divisors with an exponent near 1's (half), or any finite one."""
    x = pat(rsign(rng, n), np.zeros(n, dtype=np.int64), rfrac(rng, n) | np.uint64(1))
    ye = np.where(rng.integers(0, 2, n) == 1, rng.integers(1023 - 60, 1023 + 61, n), rng.integers(0, 2047, n))
    y = pat(rsign(rng, n), ye, rfrac(rng, n))
    return x, y


def fam_subnormal_quotient(rng, n):
    """Normal dividends and divisors whose quotient is subnormal or just above: the divisor's
    exponent is the dividend's plus 1021 to 1075."""
    xe = rng.integers(1, 2047 - 1075, n)
    ye = xe + rng.integers(1021, 1076, n)
    x = pat(rsign(rng, n), xe, rfrac(rng, n))
    y = pat(rsign(rng, n), ye, rfrac(rng, n))
    return x, y


def fam_subnormal_divisor(rng, n):
    """Subnormal divisors (quotients large, up to overflow); dividends finite."""
    y = pat(rsign(rng, n), np.zeros(n, dtype=np.int64), rfrac(rng, n) | np.uint64(1))
    xe = np.where(rng.integers(0, 2, n) == 1, rng.integers(0, 60, n), rng.integers(0, 2047, n))
    x = pat(rsign(rng, n), xe, rfrac(rng, n))
    return x, y


def fam_pow2(rng, n):
    """Divisors that are powers of two (exact quotients, or a shift into the subnormal range with
    ties); dividends finite, any exponent."""
    ye = rng.integers(1, 2047, n)
    y = pat(rsign(rng, n), ye, np.zeros(n, dtype=np.uint64))
    x = pat(rsign(rng, n), rng.integers(0, 2047, n), rfrac(rng, n))
    return x, y


def fam_overflow(rng, n):
    """Quotients near the top of the range and beyond: the dividend's exponent minus the divisor's
    is 1021 to 1026."""
    ye = rng.integers(0, 2047 - 1026 - 1, n)
    xe = np.clip(ye + rng.integers(1021, 1027, n), 0, 2046)
    x = pat(rsign(rng, n), xe, rfrac(rng, n))
    y = pat(rsign(rng, n), ye, rfrac(rng, n))
    return x, y


def fam_near_exact(rng, n):
    """Dividends x = RN(q·y) for random q and y (numpy's float64 product, round to nearest): x/y
    lies within about an ulp of q, so the rounding decides between q and its neighbours."""
    q = pat(rsign(rng, n), rng.integers(1023 - 200, 1023 + 200, n), rfrac(rng, n)).view(np.float64)
    y = pat(rsign(rng, n), rng.integers(1023 - 200, 1023 + 200, n), rfrac(rng, n)).view(np.float64)
    with np.errstate(all="ignore"):
        x = q * y
    return x.view(np.uint64).copy(), y.view(np.uint64).copy()


def rn_frac(q):
    """q (a positive Fraction in the normal range of float64) rounded to nearest even, as a float."""
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    if Fraction(2) ** (e + 1) <= q:
        e += 1
    t = q / Fraction(2) ** (e - 52)
    m = t.numerator // t.denominator
    r = t - m
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and m % 2 == 1):
        m += 1
    return float(Fraction(m) * Fraction(2) ** (e - 52))


def fam_midpoint(rng, n):
    """Quotients next to a midpoint: x = RN(mid·y) for mid halfway between two consecutive
    float64 values (in exact arithmetic), so x/y is within a fraction of an ulp of the midpoint,
    on either side: the case a division algorithm is most likely to round wrongly."""
    xs = np.empty(n, dtype=np.uint64)
    ys = np.empty(n, dtype=np.uint64)
    mant = (rbits(rng, n) & FRAC) | np.uint64(1 << 52)
    ymant = (rbits(rng, n) & FRAC) | np.uint64(1 << 52)
    qe = rng.integers(-40, 41, n)
    ye = rng.integers(-40, 41, n)
    sg = rsign(rng, n)
    for i in range(n):
        mid = Fraction(2 * int(mant[i]) + 1, 1 << 53) * Fraction(2) ** int(qe[i])
        y = Fraction(int(ymant[i]), 1 << 52) * Fraction(2) ** int(ye[i])
        x = rn_frac(mid * y)
        xs[i] = np.float64(-x if sg[i] else x).view(np.uint64)
        ys[i] = np.float64(float(y)).view(np.uint64)
    return xs, ys


def fam_specials(rng, n):
    sp = specials()
    x, y = np.meshgrid(sp, sp, indexing="ij")
    return x.ravel().copy(), y.ravel().copy()


CONSTS = [1.0, -1.0, 2.0, 0.5, 3.0, 0.1, 7.0, 2.0 ** -1074, 2.0 ** 1023, 1e308, 1.0 / 3.0]

FAMILIES = [
    ("uniform", fam_uniform, 1 << 26, "div"),
    ("struct", fam_struct, 1 << 26, "div"),
    ("near1", fam_near1, 1 << 25, "div"),
    ("binade1", fam_binade1, 1 << 25, "div"),
    ("subnormal-dividend", fam_subnormal_dividend, 1 << 25, "div"),
    ("subnormal-quotient", fam_subnormal_quotient, 1 << 26, "div"),
    ("subnormal-divisor", fam_subnormal_divisor, 1 << 24, "div"),
    ("pow2", fam_pow2, 1 << 24, "div"),
    ("overflow", fam_overflow, 1 << 24, "div"),
    ("near-exact", fam_near_exact, 1 << 25, "div"),
    ("midpoint", fam_midpoint, 1 << 22, "div"),
    ("specials", fam_specials, None, "div"),
    ("struct, tl.fdiv", fam_struct, 1 << 24, "fdiv"),
]


def run_family(driver, name, gen, n, mode, work, jobs, c=None):
    t0 = time.time()
    seed = int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") + (0 if c is None else int(np.float64(c).view(np.uint64)))
    files, dev = [], {}
    if n is None:
        x, y = gen(np.random.default_rng(seed), 0)
        n = len(x)
        chunks = [(x, y)]
    else:
        chunks = None
    k = 0
    for base in range(0, n, PER):
        if chunks is not None:
            x, y = chunks[0]
        else:
            x, y = gen(np.random.default_rng((seed, k)), min(PER, n - base))
        if c is not None:
            y = np.full(len(x), np.float64(c).view(np.uint64), dtype=np.uint64)
        p = os.path.join(work, f"rec.{k}.bin")
        np.stack([x, y, np.zeros(len(x), dtype=np.uint64)], axis=1).astype("<u8").tofile(p)
        files.append((p, base))
        out = device(x, y, "const" if c is not None else mode, 0.0 if c is None else c)
        dev.update(dev_hashes(out, base))
        k += 1
    lean = lean_hashes(driver, files, jobs)
    bad = sorted(ch for ch in dev if dev[ch] != lean.get(ch))
    lines = []
    for ch in bad[:4]:
        p, base = files[ch]
        lv = lean_dump(driver, p, base)
        rec = np.fromfile(p, dtype="<u8").reshape(-1, 3)
        dv = device(rec[:, 0].copy(), rec[:, 1].copy(), "const" if c is not None else mode,
                    0.0 if c is None else c).tolist()
        diffs = [i for i in range(base, base + len(rec)) if (lv[i] & M64) != (dv[i - base] & M64)]
        for i in diffs[:6]:
            lines.append(f"    input {int(rec[i - base][0]):016x},{int(rec[i - base][1]):016x}: "
                         f"library {lv[i]:016x}, device {dv[i - base] & M64:016x}")
        lines.append(f"    chunk {ch}: {len(diffs)} inputs differ")
    for p, _ in files:
        os.remove(p)
    label = name if c is None else f"x / {c!r} (constant)"
    verdict = "agree" if not bad and set(dev) == set(lean) else f"DIFFER in {len(bad)} chunks"
    print(f"  ({label}: {time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    return f"| {label} | {mode if c is None else 'const'} | {n} | {verdict} |", lines


def main():
    args = sys.argv[1:]
    jobs = 32
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i:i + 2]
    quick = "--quick" in args  # a smoke test: every family at 2^20 records at most, into <B>/quick.txt
    if quick:
        args.remove("--quick")
    driver = os.path.join(os.path.abspath(args[0]), "driver")
    work = os.path.join(os.path.abspath(args[0]), "work-div64")
    os.makedirs(work, exist_ok=True)
    fams = [(a, b, None if n is None else (min(n, PER) if quick else n), d) for a, b, n, d in FAMILIES]
    with open(os.path.join(args[0], "quick.txt") if quick else "out.txt", "w") as out:
        def emit(s):
            print(s, flush=True)
            out.write(s + "\n")
            out.flush()
        emit(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
        emit("| inputs | kernel | records | against Format.div .rn (FloatV.divRn) |")
        emit("|---|---|---|---|")
        total = 0
        for name, gen, n, mode in fams:
            line, extra = run_family(driver, name, gen, n, mode, work, jobs)
            emit(line)
            for l in extra:
                emit(l)
            total += int(line.split("|")[3])
        for c in CONSTS:
            line, extra = run_family(driver, "const", fam_struct, PER if quick else 1 << 22, "const", work, jobs, c=c)
            emit(line)
            for l in extra:
                emit(l)
            total += int(line.split("|")[3])
        emit(f"\nTotal records: {total}")
        emit("\n## Division instructions in the PTX")
        for (mode, c), h in sorted(COMPILED.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            ins = sorted({l.strip().split()[0] for l in h.asm["ptx"].splitlines()
                          if l.strip().split() and l.strip().split()[0].split(".")[0] in ("div", "rcp", "mul", "fma")
                          and l.strip().split()[0].endswith("f64")})
            emit(f"- `{mode}`{'' if mode != 'const' else f' by {c!r}'}: " +
                 (", ".join(f"`{i}`" for i in ins) if ins else "no division instruction (the compiler folds it)"))


if __name__ == "__main__":
    main()
