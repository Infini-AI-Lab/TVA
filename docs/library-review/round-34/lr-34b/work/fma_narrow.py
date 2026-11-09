"""lr-34b (round 34, L155): try to break the narrowed `AddRel` on the device.

For every one-product spelling Triton contracts (a*b + c, c + a*b, a*b - c, c - a*b, -(a*b) + c, a
product through `tl.where`, a `tl.dot` result scaled then added, a product widened by `.to` then
added), in float32, float64, float16 and bfloat16, on input regimes chosen to show the product's
rounding (near cancellation, subnormal products and results, a product that overflows when rounded
but whose fma is finite, infinite and NaN addends, random), each device result is checked exactly
(Fractions) against the narrow relation of `lean/TVA/Lang/Ops.lean` (`AddRel true x y r`, x and y
the operands as the model computes them: the rounded product, the addend):
  - the IEEE sum (NaN results equal up to payload), or
  - the rounding of p + y for some exact p rounding to x, or of x + q for some q rounding to y
    (an exact zero sum may take either sign), or
  - an infinite operand and an exact other operand: that infinity.
The set {rnd(p + y) : p rounds to x} is the float interval [rnd(lo + y), rnd(hi + y)] (rounding is
monotone); the check uses the closed interval (an over-approximation only at an excluded endpoint).
Prints one line per (dtype, form, regime): elements, outside the relation, device = IEEE, device =
fma (the exact product rounded once with the addend).
"""
import math
import random
import sys
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

FMT = {torch.float32: (24, -126, 127), torch.float64: (53, -1022, 1023),
       torch.float16: (11, -14, 15), torch.bfloat16: (8, -126, 127)}
INF = float("inf")


class F:
    """A float value: ('fin', Fraction, signbit) | ('inf', sign) | ('nan',)."""


def rnd(q, dt):
    """Round a Fraction to nearest even; returns a Fraction or +-INF (float). Zero returned as Fr(0)."""
    p, emin, emax = FMT[dt]
    if q == 0:
        return Fr(0)
    s = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a:
        e -= 1
    e = max(e, emin)
    quantum = Fr(2) ** (e - p + 1)
    m = a / quantum
    n = m.numerator // m.denominator
    rem = m - n
    if rem > Fr(1, 2) or (rem == Fr(1, 2) and n % 2 == 1):
        n += 1
    v = n * quantum
    if v >= Fr(2) ** (emax + 1):
        return s * INF
    return s * v


def maxf(dt):
    p, emin, emax = FMT[dt]
    return Fr(2) ** emax * (2 - Fr(2) ** (1 - p))


def interval(x, dt):
    """The exact values that round to x (x a Fraction or +-INF): (lo, hi), hi may be +-INF (unbounded)."""
    p, emin, emax = FMT[dt]
    if isinstance(x, float):
        thr = Fr(2) ** emax * (2 - Fr(2) ** (-p))
        return (thr, INF) if x > 0 else (-INF, -thr)
    a = abs(x)
    if a == 0:
        h = Fr(2) ** (emin - p + 1) / 2
        return (-h, h)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a:
        e -= 1
    eq = max(e, emin)
    ulp = Fr(2) ** (eq - p + 1)
    up = ulp / 2
    down = (ulp / 4) if (a == Fr(2) ** e and e > emin) else ulp / 2
    lo, hi = a - down, a + up
    if a == maxf(dt):
        hi = Fr(2) ** emax * (2 - Fr(2) ** (-p))
    return (lo, hi) if x > 0 else (-hi, -lo)


def val(t):
    """Device tensor -> list of (kind, value): ('n',) NaN, else Fraction / +-INF, with sign bit for zeros."""
    out = []
    for v in t.double().tolist():
        if math.isnan(v):
            out.append(("nan", None, None))
        elif math.isinf(v):
            out.append(("v", v, v < 0))
        else:
            out.append(("v", Fr(v), math.copysign(1.0, v) < 0))
    return out


def add_ieee(x, y, dt):
    """IEEE x + y on (kind, value) pairs; returns (kind, value, zero_signs_allowed set or None)."""
    if x[0] == "nan" or y[0] == "nan":
        return ("nan", None)
    xv, yv = x[1], y[1]
    if isinstance(xv, float) or isinstance(yv, float):
        if isinstance(xv, float) and isinstance(yv, float):
            return ("nan", None) if xv != yv else ("v", xv)
        return ("v", xv if isinstance(xv, float) else yv)
    s = xv + yv
    r = rnd(s, dt)
    return ("v", r)


def ieee_zero_sign(x, y):
    """Sign bit of an exact-zero IEEE sum (RN): negative only if both operands are -0 ... or nonzero exact cancel -> +0."""
    if x[1] == 0 and y[1] == 0:
        return x[2] and y[2]
    return False


def in_narrow(x, y, r, dt):
    """Is r allowed by AddRel true x y r? x, y, r are (kind, value, signbit)."""
    if r[0] == "nan":
        ie = add_ieee(x, y, dt)
        return ie[0] == "nan"
    ie = add_ieee(x, y, dt)
    if ie[0] == "v" and ie[1] == r[1]:
        if r[1] != 0 or isinstance(r[1], float) or ieee_zero_sign(x, y) == r[2]:
            return True
    # fusion branch 1: one operand replaced by an exact value rounding to it, the other as it is
    for a, b in ((x, y), (y, x)):
        if a[0] == "nan" or b[0] == "nan" or isinstance(b[1], float):
            continue
        lo, hi = interval(a[1], dt)
        rl = rnd(lo + b[1], dt) if not isinstance(lo, float) else lo
        rh = rnd(hi + b[1], dt) if not isinstance(hi, float) else hi
        if rl <= r[1] <= rh:
            if r[1] == 0 and not isinstance(r[1], float):
                # an exact zero sum may take either sign; a nonzero sum that underflows keeps its sign
                exact_zero_possible = (lo <= -b[1] <= hi)
                if exact_zero_possible:
                    return True
                # some nonzero sum rounds to zero: its sign
                if (r[2] and rl <= 0 and lo + b[1] < 0) or ((not r[2]) and rh >= 0 and hi + b[1] > 0):
                    return True
                continue
            return True
    # fusion branches 2, 3: an infinite operand, the other roundable: that infinity
    for a, b in ((x, y), (y, x)):
        if b[0] == "v" and isinstance(b[1], float) and a[0] == "v" and r[1] == b[1]:
            return True
    return False


@triton.jit
def k(a_ptr, b_ptr, c_ptr, m_ptr, o_ptr, FORM: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    a = tl.load(a_ptr + j)
    b = tl.load(b_ptr + j)
    c = tl.load(c_ptr + j)
    m = tl.load(m_ptr + j) != 0
    if FORM == 0:
        r = a * b + c
    elif FORM == 1:
        r = c + a * b
    elif FORM == 2:
        r = a * b - c
    elif FORM == 3:
        r = c - a * b
    elif FORM == 4:
        r = -(a * b) + c
    elif FORM == 5:
        r = tl.where(m, a * b, 0.0) + c
    elif FORM == 6:  # widened product (a, b narrower than c)
        r = (a * b).to(c.dtype) + c
    elif FORM == 7:
        r = (a * b) * a + c
    tl.store(o_ptr + j, r)


@triton.jit
def kdot(a_ptr, b_ptr, s_ptr, c_ptr, acc_ptr, o_ptr, M: tl.constexpr):
    i = tl.arange(0, M)
    A = tl.load(a_ptr + i[:, None] * M + i[None, :])
    B = tl.load(b_ptr + i[:, None] * M + i[None, :])
    S = tl.load(s_ptr + i[:, None] * M + i[None, :])
    C = tl.load(c_ptr + i[:, None] * M + i[None, :])
    acc = tl.dot(A, B)
    tl.store(acc_ptr + i[:, None] * M + i[None, :], acc)
    tl.store(o_ptr + i[:, None] * M + i[None, :], acc * S - C)


FORMS = {0: "a*b + c", 1: "c + a*b", 2: "a*b - c", 3: "c - a*b", 4: "-(a*b) + c", 5: "where(m, a*b, 0) + c",
         6: "(a*b).to(wide) + c", 7: "(a*b)*a + c"}


def ulp_nudge(v, dt, kk):
    p, emin, emax = FMT[dt]
    q = Fr(v)
    if q == 0:
        return float(kk * Fr(2) ** (emin - p + 1))
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fr(2) ** e > abs(q):
        e -= 1
    return float(q + kk * Fr(2) ** (max(e, emin) - p + 1))


def regime_inputs(reg, dt, N):
    p, emin, emax = FMT[dt]
    a = torch.rand(N, dtype=torch.float64) * 2 + 0.5
    b = torch.rand(N, dtype=torch.float64) * 2 + 0.5
    sgn = torch.where(torch.rand(N) < 0.5, -1.0, 1.0).double()
    a = a * sgn
    if reg == "subnormal":  # a*b near the smallest normal and below; c of the same size
        a = a * 2.0 ** (emin // 2)
        b = b * 2.0 ** (emin - emin // 2 - random.choice([0, 2, 5, p // 2]))
    elif reg == "overflow":  # a*b just above the largest float, c = -(a large float)
        a = (a * 2.0 ** (emax // 2)).clamp(-float(maxf(dt)), float(maxf(dt)))
        b = b.abs() * 2.0 ** (emax - emax // 2) + 2.0 ** (emax - emax // 2)
    a, b = a.to(dt), b.to(dt)
    ab = [Fr(x) * Fr(y) for x, y in zip(a.double().tolist(), b.double().tolist())]
    if reg in ("cancel", "subnormal"):
        c = [ulp_nudge(float(rnd(q, dt)) if not isinstance(rnd(q, dt), float) else 0.0, dt, random.randint(-2, 2)) for q in ab]
        c = torch.tensor(c, dtype=torch.float64)
    elif reg == "overflow":
        c = torch.tensor([-float(maxf(dt)) * (1 if q > 0 else -1) * random.choice([1, 0.75, 0.5]) for q in ab],
                         dtype=torch.float64)
    elif reg == "infnan":
        c = torch.tensor([random.choice([INF, -INF, float("nan"), 1.0, 0.0, -0.0]) for _ in range(N)],
                         dtype=torch.float64)
        a = torch.tensor([random.choice([float(x), INF, -INF, 0.0, -0.0, float(maxf(dt))]) for x in a.double().tolist()],
                         dtype=torch.float64).to(dt)
    else:
        c = (torch.rand(N, dtype=torch.float64) - 0.5) * 8
    return a, b, c.to(dt)


def model_operands(form, A, B, C, M, dt_ab, dt_c):
    """(x, y) as the model computes them, with `.to`s exact (a widening)."""
    def prod(a, b, dt):
        if a[0] == "nan" or b[0] == "nan":
            return ("nan", None, None)
        av, bv = a[1], b[1]
        if isinstance(av, float) or isinstance(bv, float):
            if av == 0 or bv == 0:
                return ("nan", None, None)
            s = (1 if av > 0 else -1) * (1 if bv > 0 else -1)
            return ("v", s * INF, s < 0)
        q = av * bv
        r = rnd(q, dt)
        neg = (a[2] != b[2])
        return ("v", r, neg)
    def neg(x):
        if x[0] == "nan":
            return x
        return ("v", -x[1] if x[1] != 0 else x[1], not x[2])
    P = prod(A, B, dt_ab)
    if form == 0:
        return P, C
    if form == 1:
        return C, P
    if form == 2:
        return P, neg(C)
    if form == 3:
        return C, neg(P)
    if form == 4:
        return neg(P), C
    if form == 5:
        return (P if M else ("v", Fr(0), False)), C
    if form == 6:
        return P, C
    if form == 7:
        return prod(P, A, dt_ab), C


def fma_value(form, A, B, C, M, dt):
    """The exact product rounded once with the addend (finite operands only), else None."""
    vals = [A[1], B[1], C[1]]
    if any(v is None or isinstance(v, float) for v in vals):
        return None
    a, b, c = vals
    if form == 0 or form == 1 or form == 6:
        return rnd(a * b + c, dt)
    if form == 2:
        return rnd(a * b - c, dt)
    if form == 3 or form == 4:
        return rnd(c - a * b, dt)
    if form == 5:
        return rnd(a * b + c, dt) if M else rnd(c, dt)
    return None


def run():
    random.seed(34)
    torch.manual_seed(34)
    N = 2048
    T = 2
    total = bad = 0
    print(f"Triton {triton.__version__}, {torch.cuda.get_device_name()}")
    print("| a,b dtype | add dtype | form | regime | elements | outside narrow AddRel | = IEEE | = fma |")
    print("|---|---|---|---|---|---|---|---|")
    for dt in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        for form in FORMS:
            if form == 6:
                pairs = {torch.float16: torch.float32, torch.bfloat16: torch.float32, torch.float32: torch.float64,
                         torch.float64: None}[dt]
                if pairs is None:
                    continue
                dc = pairs
            else:
                dc = dt
            for reg in ["cancel", "subnormal", "overflow", "infnan", "random"]:
                n_out = n_ieee = n_fma = n = 0
                examples = []
                for t in range(T):
                    a, b, c = regime_inputs(reg, dt, N)
                    if form == 6:
                        c = c.double().to(dc)
                    m = (torch.rand(N) < 0.5).int()
                    o = torch.empty(N, dtype=dc, device="cuda")
                    k[(1,)](a.cuda(), b.cuda(), c.cuda(), m.cuda(), o, form, N)
                    torch.cuda.synchronize()
                    A, B, C, O = val(a), val(b), val(c), val(o.cpu())
                    Ml = m.tolist()
                    for i in range(N):
                        x, y = model_operands(form, A[i], B[i], C[i], Ml[i], dt, dc)
                        n += 1
                        ok = in_narrow(x, y, O[i], dc)
                        ie = add_ieee(x, y, dc)
                        if ie[0] == O[i][0] and (ie[0] == "nan" or ie[1] == O[i][1]):
                            n_ieee += 1
                        fv = fma_value(form, A[i], B[i], C[i], Ml[i], dc)
                        if fv is not None and O[i][0] == "v" and fv == O[i][1]:
                            n_fma += 1
                        if not ok:
                            n_out += 1
                            if len(examples) < 3:
                                examples.append((A[i], B[i], C[i], Ml[i], x, y, O[i]))
                total += n
                bad += n_out
                print(f"| {str(dt)[6:]} | {str(dc)[6:]} | `{FORMS[form]}` | {reg} | {n} | {n_out} | {n_ieee} | {n_fma} |",
                      flush=True)
                for ex in examples:
                    print("    example:", ex)
    # tl.dot epilogue: acc * S - C, f16 inputs, f32 accumulator, nearly cancelling C
    M = 32
    n_out = n = 0
    for t in range(4):
        A = (torch.randn(M, M) / 4).half()
        B = (torch.randn(M, M) / 4).half()
        S = (torch.rand(M, M) + 0.5).float()
        acc_ref = (A.float() @ B.float())
        C = (acc_ref * S).float()
        acc = torch.empty(M, M, device="cuda")
        o = torch.empty(M, M, device="cuda")
        kdot[(1,)](A.cuda(), B.cuda(), S.cuda(), C.cuda(), acc, o, M)
        torch.cuda.synchronize()
        AC, SS, CC, OO = val(acc.cpu().flatten()), val(S.flatten()), val(C.flatten()), val(o.cpu().flatten())
        for i in range(M * M):
            pr = rnd(AC[i][1] * SS[i][1], torch.float32)
            x = ("v", pr, pr < 0)
            y = ("v", -CC[i][1], not CC[i][2])
            n += 1
            if not in_narrow(x, y, OO[i], torch.float32):
                n_out += 1
    print(f"| float16 dot | float32 | `tl.dot(A, B) * S - C` | cancel | {n} | {n_out} | | |")
    total += n
    bad += n_out
    print(f"TOTAL elements {total}, outside the narrow relation {bad}")


if __name__ == "__main__":
    run()
