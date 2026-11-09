"""TVA library evidence (library issue L163): a float32 `/` is a product for the fusion rule.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/div-fusion/run.py
Output: evidence/div-fusion/out.txt (README.md says what it shows).

`div.full.f32` (what `/` on float32 lowers to, docs/rules/num.md "Division") is a reciprocal and a
multiply by the (scaled) dividend in ptxas; with `enable_fp_fusion` that multiply may fuse with a
following addition into one `FFMA`. For each spelling of a quotient followed by an addition (FORMS),
a first launch stores the quotient alone (`q`, the value the model's `tl.div` picks), and a second
launch, with the same inputs, stores the spelling's result `r`. Each `r` is compared, exactly (Python
`Fraction`s, rounding to nearest even in the format), with what `AddRel true u v` allows
(`lean/TVA/Lang/Ops.lean`) for the operands `u`, `v` the model gives the addition from `q` and the
addend: the IEEE sum, the rounding of `p + v` for an exact `p` rounding to `u`, or of `u + p` for an
exact `p` rounding to `v`, an infinite operand added to an exact other operand. Counted: results
outside it (none expected), and results that differ from the IEEE sum of the operands (those where the
fusion shows). Inputs: quotients nearly cancelled by the addend (the addend is minus the quotient moved
by up to 3 units in the last place, so the unrounded product shows), random patterns, divisors beyond
2^126 and subnormal ones (the two scaled paths of `div.full.f32`), subnormal quotients, and every
combination of special values. A control runs the first spelling with `enable_fp_fusion=False`. The
PTX and SASS instructions are listed.
"""
import itertools
import os
import random
import re
import struct
import subprocess
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

DEV = "cuda"
OUT = []
HERE = os.path.dirname(os.path.abspath(__file__))
CUOBJDUMP = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
INF = float("inf")


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


class Fmt:
    def __init__(self, name, p, emin, emax):
        self.name, self.p, self.emin, self.emax = name, p, emin, emax
        self.maxf = (2 - Fr(2) ** (1 - p)) * Fr(2) ** emax
        self.thr = self.maxf + Fr(2) ** (emax - p)      # the overflow threshold (a tie there goes to infinity)
        self.minsub = Fr(2) ** (emin - p + 1)
        self.top = Fr(2) ** (emax + 1)                  # toRat of an infinity (TVA.Math.Defs.toRat)

    def quantum(self, a):
        """The spacing of the format at a > 0."""
        e = a.numerator.bit_length() - a.denominator.bit_length()
        if Fr(2) ** e > a:
            e -= 1
        return Fr(2) ** (max(e, self.emin) - self.p + 1)

    def rnd(self, t):
        """Round an exact t to nearest even; returns a Fraction (0 for a zero; its sign apart) or +-INF."""
        if t == 0:
            return Fr(0)
        s = -1 if t < 0 else 1
        a = abs(t)
        if a >= self.thr:
            return s * INF
        qu = self.quantum(a)
        m = a / qu
        n = m.numerator // m.denominator
        rem = m - n
        if rem > Fr(1, 2) or (rem == Fr(1, 2) and n % 2 == 1):
            n += 1
        return s * n * qu

    def interval(self, v):
        """The exact values that round to the float v (RoundsTo): (lo, lo_in, hi, hi_in), with -INF/INF ends."""
        if v == INF:
            return (self.thr, True, INF, False)
        if v == -INF:
            return (-INF, False, -self.thr, True)
        v = Fr(v)
        if v == 0:
            return (-self.minsub / 2, True, self.minsub / 2, True)
        a = abs(v)
        qu = self.quantum(a)
        below = qu / 2 if (a / qu == 2 ** (self.p - 1) and a > Fr(2) ** self.emin) else qu
        even = (a / qu).numerator % 2 == 0
        lo, hi = a - below / 2, a + qu / 2
        if v > 0:
            return (lo, even, hi, even)
        return (-hi, even, -lo, even)


F32, F64, F16, BF16 = Fmt("float32", 24, -126, 127), Fmt("float64", 53, -1022, 1023), Fmt("float16", 11, -14, 15), \
    Fmt("bfloat16", 8, -126, 127)
EPS = Fr(1, 2 ** 3000)


def is_nan(v):
    return v != v


def neg_zero(v):
    return v == 0 and struct.pack(">d", v)[0] >= 0x80


def ieee_add(f, u, v):
    """The IEEE sum as a Python float (NaN, an infinity, or a finite value with its zero sign)."""
    if is_nan(u) or is_nan(v) or (abs(u) == INF and abs(v) == INF and u != v):
        return float("nan")
    if abs(u) == INF:
        return u
    if abs(v) == INF:
        return v
    t = Fr(u) + Fr(v)
    if t == 0:
        return -0.0 if (neg_zero(u) and neg_zero(v)) else 0.0
    r = f.rnd(t)
    if r == 0:
        return -0.0 if t < 0 else 0.0
    return float(r)


def same(a, b):
    """Equal as floats of the format, up to NaN payload, zeros by sign."""
    if is_nan(a) or is_nan(b):
        return is_nan(a) and is_nan(b)
    return a == b and neg_zero(a) == neg_zero(b)


def image_allows(f, lo, lo_in, hi, hi_in, r):
    """Whether r is the rounding of some t in the interval (rounding is monotone, so the image is every
    float between the images of the ends; a zero result needs a t of its sign within half the least
    subnormal, or t = 0, whose zero has either sign)."""
    if lo == -INF:
        rmin = -INF
    else:
        rmin = f.rnd(lo if lo_in else lo + EPS)
    if hi == INF:
        rmax = INF
    else:
        rmax = f.rnd(hi if hi_in else hi - EPS)
    if r == 0:
        h = f.minsub / 2
        if neg_zero(r):   # some t in [-h, 0]
            return (lo == -INF or lo < 0 or (lo == 0 and lo_in)) and (hi == INF or hi > -h or (hi == -h and hi_in))
        return (hi == INF or hi > 0 or (hi == 0 and hi_in)) and (lo == -INF or lo < h or (lo == h and lo_in))
    rv = r if abs(r) == INF else Fr(r)
    return rmin <= rv <= rmax


def shift(iv, c):
    lo, lo_in, hi, hi_in = iv
    return (lo + c if lo != -INF else -INF, lo_in, hi + c if hi != INF else INF, hi_in)


def addrel(f, u, v, r):
    """AddRel true u v r (lean/TVA/Lang/Ops.lean): the IEEE sum up to NaN payload, or the rounding of
    p + q with RoundsTo p u, RoundsTo q v and p or q the operand's own value (toRat; +-2^(emax+1) for an
    infinity), or an infinite operand returned when the other has some exact value rounding to it."""
    if same(ieee_add(f, u, v), r):
        return True
    if is_nan(r) or is_nan(u) or is_nan(v):
        return False
    if abs(v) == INF and r == v:
        return True
    if abs(u) == INF and r == u:
        return True
    own = lambda w: (f.top if w > 0 else -f.top) if abs(w) == INF else Fr(w)  # noqa: E731
    # p = u's own value, q ranges over v's interval; and the other way round
    return (image_allows(f, *shift(f.interval(v), own(u)), r) or
            image_allows(f, *shift(f.interval(u), own(v)), r))


# ---------------------------------------------------------------- kernels

@triton.jit
def k_quot(x_ptr, y_ptr, z_ptr, w_ptr, m_ptr, q_ptr, q2_ptr, FORM: tl.constexpr, N: tl.constexpr):
    """The quotient alone, as the spelling computes it (`q`), and the second quotient (`q2`) where there is one."""
    j = tl.arange(0, N)
    x = tl.load(x_ptr + j)
    y = tl.load(y_ptr + j)
    if FORM == 4:
        q = tl.where(tl.load(m_ptr + j) != 0, x / y, 0.0)
    elif FORM == 6:
        q = tl.fdiv(x, y)
    elif FORM == 7:
        q = tl.sigmoid(x)
    elif FORM == 8:
        q = x / 3.0
    elif FORM == 9:
        q = 1.0 / y
    else:
        q = x / y
    tl.store(q_ptr + j, q)
    if FORM == 10 or FORM == 11:
        tl.store(q2_ptr + j, tl.load(z_ptr + j) / tl.load(w_ptr + j))


@triton.jit
def k_form(x_ptr, y_ptr, c_ptr, z_ptr, w_ptr, m_ptr, o_ptr, FORM: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(x_ptr + j)
    y = tl.load(y_ptr + j)
    c = tl.load(c_ptr + j)
    if FORM == 0:
        r = x / y + c
    elif FORM == 1:
        r = c + x / y
    elif FORM == 2:
        r = x / y - c
    elif FORM == 3:
        r = c - x / y
    elif FORM == 4:
        r = tl.where(tl.load(m_ptr + j) != 0, x / y, 0.0) + c
    elif FORM == 5:
        r = (x / y).to(tl.float32) + c
    elif FORM == 6:
        r = tl.fdiv(x, y) + c
    elif FORM == 7:
        r = tl.sigmoid(x) + c
    elif FORM == 8:
        r = x / 3.0 + c
    elif FORM == 9:
        r = 1.0 / y + c
    elif FORM == 10:
        r = x / y + tl.load(z_ptr + j) / tl.load(w_ptr + j)
    elif FORM == 11:
        r = tl.sum(tl.join(x / y, tl.load(z_ptr + j) / tl.load(w_ptr + j)), axis=1)
    elif FORM == 12:
        r = c
        for t in range(1):
            r += x / y
    elif FORM == 13:
        r = -(x / y) + c
    elif FORM == 14:
        r = (x / y).to(tl.float64) + c     # c is float64
    elif FORM == 15:
        r = (x / y).to(tl.float16) + c     # c is float16
    else:
        r = x / y + c                      # FORM 16, 17: x, y, c are float16 / bfloat16, the quotient float32
    tl.store(o_ptr + j, r)


FORMS = {0: "x / y + c", 1: "c + x / y", 2: "x / y - c", 3: "c - x / y", 4: "tl.where(m, x / y, 0.0) + c",
         5: "(x / y).to(tl.float32) + c", 6: "tl.fdiv(x, y) + c", 7: "tl.sigmoid(x) + c", 8: "x / 3.0 + c",
         9: "1.0 / y + c", 10: "x / y + z / w", 11: "tl.sum(tl.join(x / y, z / w), axis=1)",
         12: "r = c; loop: r += x / y", 13: "-(x / y) + c", 14: "(x / y).to(tl.float64) + c (float64 c)",
         15: "(x / y).to(tl.float16) + c (float16 c)", 16: "x / y + c, all float16 (quotient float32)",
         17: "x / y + c, all bfloat16 (quotient float32)"}
IN_DT = {16: torch.float16, 17: torch.bfloat16}
C_DT = {14: torch.float64, 15: torch.float16, 16: torch.float16, 17: torch.bfloat16}
OUT_DT = {14: torch.float64, 15: torch.float16}
RES_FMT = {14: F64, 15: F16}


def cvt(f, v):
    """v rounded to the format f (to nearest even), keeping a zero's sign."""
    if is_nan(v) or abs(v) == INF:
        return v
    r = f.rnd(Fr(v))
    if r == 0:
        return -0.0 if (v < 0 or neg_zero(v)) else 0.0
    return float(r)


def operands(form, q, q2, c):
    """The addition's operands (u, v), as the model writes them (`tl.addc u v`; `tl.subc` as `u + (-v)`)."""
    if form == 2:
        return q, -c
    if form == 3:
        return c, -q
    if form == 13:
        return -q, c
    if form in (10, 11):
        return q, q2
    if form == 15:
        return cvt(F16, q), c                                                       # `.to(float16)`: rounding
    return q, c                                                                     # 4: q is the where's value


# ---------------------------------------------------------------- inputs

def f32_pattern(rng):
    while True:
        v = struct.unpack("<f", struct.pack("<I", rng.getrandbits(32)))[0]
        if v == v and abs(v) != INF:
            return v


def to_f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def nudge(f, v, k):
    """v moved by k units in its last place (in the format f)."""
    if v == 0 or abs(v) == INF or is_nan(v):
        return v
    return float(Fr(v) + k * f.quantum(abs(Fr(v))))


SPECIALS = [0.0, -0.0, INF, -INF, float("nan"), 1.0, -1.0, 3.0, 2.0 ** -149, 2.0 ** 127, 2.0 ** 126 * 1.5,
            float(F32.maxf), 2.0 ** -126, 0.75]


def regimes(form, N, rng):
    """(name, x, y, z, w, m, c-maker) per regime; c is made from the quotient for the cancelling regimes."""
    R = []
    if form == 7:
        mk = lambda: [rng.uniform(-8, 8) for _ in range(N)]      # noqa: E731  sigmoid's input
        R.append(("cancel", mk(), None, "cancel"))
        R.append(("random", [f32_pattern(rng) / 2 ** 100 if rng.random() < 0.5 else rng.uniform(-90, 90)
                             for _ in range(N)], None, "random"))
    else:
        R.append(("cancel", [rng.uniform(0.5, 2) * rng.choice((1, -1)) for _ in range(N)],
                  [rng.uniform(0.5, 2) for _ in range(N)], "cancel"))
        R.append(("random", [f32_pattern(rng) for _ in range(N)], [f32_pattern(rng) for _ in range(N)], "random"))
        R.append(("divisor > 2^126", [rng.uniform(1, 2) * 2.0 ** rng.randint(100, 127) * rng.choice((1, -1))
                                      for _ in range(N)],
                  [rng.uniform(1, 2) * 2.0 ** rng.choice((126, 127)) for _ in range(N)], "cancel"))
        R.append(("subnormal divisor", [rng.uniform(1, 2) * 2.0 ** rng.randint(-149, -100) for _ in range(N)],
                  [rng.randint(1, 2 ** 23 - 1) * 2.0 ** -149 for _ in range(N)], "cancel"))
        R.append(("subnormal quotient", [rng.uniform(1, 2) * 2.0 ** rng.randint(-149, -126) for _ in range(N)],
                  [rng.uniform(0.5, 4) for _ in range(N)], "cancel"))
    return R


# ---------------------------------------------------------------- instructions

def instructions(h):
    ptx = h.asm["ptx"].split(".entry", 1)[1]
    pops = sorted({m for m in re.findall(r"\b((?:add|sub|fma|mul|div|rcp)\.[a-z0-9.]+)", ptx)
                   if any(x in m for x in ("f32", "f64", "f16", "bf16"))})
    path = os.path.join(HERE, "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(h.asm["cubin"])
    try:
        s = subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)
    sops = sorted({m for m in re.findall(r"\b((?:[DHF](?:ADD|FMA|MUL)2?|MUFU)(?:\.[A-Z0-9_]+)*)\b", s)})
    return pops, sops


# ---------------------------------------------------------------- main

def run_form(form, rng, N, fusion=True):
    rows = []
    in_dt = IN_DT.get(form, torch.float32)
    c_dt = C_DT.get(form, torch.float32)
    o_dt = OUT_DT.get(form, torch.float32)
    f_res = RES_FMT.get(form, F32)
    f_c = {torch.float64: F64, torch.float16: F16, torch.bfloat16: BF16}.get(c_dt, F32)
    f_in = {torch.float16: F16, torch.bfloat16: BF16}.get(in_dt, F32)
    h = None
    for name, xs, ys, cmode in regimes(form, N, rng) + [("special", None, None, "special")]:
        if name == "special":
            combos = list(itertools.product(SPECIALS, SPECIALS, SPECIALS))
            combos += [combos[i] for i in range(N - len(combos))]   # padded to N (a power of two)
            n = len(combos)
            xs = [a for a, _, _ in combos]
            ys = [b for _, b, _ in combos]
            cs_fixed = [c for _, _, c in combos]
        else:
            n = len(xs)
            cs_fixed = None
        if ys is None:
            ys = [1.0] * n
        x = torch.tensor(xs, dtype=torch.float64).to(in_dt)
        y = torch.tensor(ys, dtype=torch.float64).to(in_dt)
        # the second quotient z / w: the first's dividend negated and moved, so that the two nearly cancel
        z = torch.tensor([nudge(f_in, -float(v), rng.randint(-3, 3)) for v in x.double().tolist()],
                         dtype=torch.float64).to(in_dt) if name != "special" else torch.flip(x, [0])
        w = y.clone() if name != "special" else torch.roll(y, 1)
        m = torch.tensor([rng.random() < 0.75 for _ in range(n)], dtype=torch.int8)
        q = torch.empty(n, dtype=torch.float32, device=DEV)
        q2 = torch.empty(n, dtype=torch.float32, device=DEV)
        dx, dy, dz, dw, dm = (t.to(DEV) for t in (x, y, z, w, m))
        k_quot[(1,)](dx, dy, dz, dw, dm, q, q2, form, n, enable_fp_fusion=fusion)
        torch.cuda.synchronize()
        Q, Q2 = q.cpu().double().tolist(), q2.cpu().double().tolist()
        # the addend
        if cs_fixed is not None:
            cs = cs_fixed
        elif cmode == "random":
            cs = [(abs(v) * 2.0 ** rng.randint(-30, 30) if v == v and abs(v) not in (0, INF) else rng.uniform(-1, 1))
                  * rng.choice((1, -1)) for v in Q]
        else:
            base = [cvt(f_c, v) if (v == v and abs(v) != INF) else 0.0 for v in Q]
            sign = 1 if form in (2, 3, 13) else -1
            cs = [nudge(f_c, sign * b, rng.randint(-3, 3)) for b in base]
        c = torch.tensor(cs, dtype=torch.float64).to(c_dt)
        o = torch.empty(n, dtype=o_dt, device=DEV)
        h = k_form[(1,)](dx, dy, c.to(DEV), dz, dw, dm, o, form, n, enable_fp_fusion=fusion)
        torch.cuda.synchronize()
        C, O = c.double().tolist(), o.cpu().double().tolist()
        n_out = n_fused = 0
        example = None
        for i in range(n):
            u, v = operands(form, Q[i], Q2[i], C[i])
            if form == 14:  # the float64 addition of the widened quotient
                u = Q[i]
            if not addrel(f_res, u, v, O[i]):
                n_out += 1
                if example is None:
                    example = (xs[i], ys[i], C[i], Q[i], O[i])
            if not same(ieee_add(f_res, u, v), O[i]):
                n_fused += 1
        rows.append((name, n, n_out, n_fused, example))
    return rows, h


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    rng = random.Random(163)
    N = 4096
    out(f"N = {N} elements per regime (special: every combination of {len(SPECIALS)} values for x, y and c)")
    out("")
    out("| form | regime | elements | outside `AddRel` | not the IEEE sum (fused) | example outside (x, y, c, q, r) |")
    out("|---|---|---|---|---|---|")
    total = 0
    instr = {}
    for form in FORMS:
        rows, h = run_form(form, rng, N)
        instr[form] = instructions(h)
        for name, n, n_out, n_fused, ex in rows:
            total += n_out
            out(f"| `{FORMS[form]}` | {name} | {n} | {n_out} | {n_fused} | {ex if ex else ''} |")
    rows, h = run_form(0, rng, N, fusion=False)
    for name, n, n_out, n_fused, ex in rows:
        total += n_out
        out(f"| `x / y + c`, `enable_fp_fusion=False` | {name} | {n} | {n_out} | {n_fused} | {ex if ex else ''} |")
    instr["off"] = instructions(h)
    out("")
    out(f"results outside AddRel, all cases: {total}")
    out("")
    out("| form | PTX | SASS |")
    out("|---|---|---|")
    for form, (p, s) in instr.items():
        label = FORMS[form] if form != "off" else "x / y + c, enable_fp_fusion=False"
        out(f"| `{label}` | {', '.join('`' + x + '`' for x in p)} | {', '.join('`' + x + '`' for x in s)} |")
    with open(os.path.join(HERE, "out.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
