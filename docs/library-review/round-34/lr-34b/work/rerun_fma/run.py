"""TVA library evidence (library issue L155): with `enable_fp_fusion`, an addition takes at most one
operand unrounded.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/fma-one-product/run.py
Output: evidence/fma-one-product/out.txt (README.md says what it shows).

For each float dtype (float32, float64, float16, bfloat16) and each form of an addition of two
products (`a*b + c*d`, `a*b - c*d`, `tl.sum` of a two-element row of products, `acc += c*d` after
`acc = a*b` in a loop, `(a*b + c*d) + e*f`), with Triton's default `enable_fp_fusion=True`, inputs are
chosen so that `a*b` and `c*d` nearly cancel (c = -a and d = b, each moved by a few units in the last
place), so that the rounding of each product matters. Each device result is compared, exactly (Python
`Fraction`s), with:
- the values with at most one operand of each addition unrounded (`AddRel` as library issue L155
  narrows it): the IEEE sum of the rounded products, `rnd(a*b + rnd(c*d))`, `rnd(rnd(a*b) + c*d)`
  (and, for the three-product form, every choice of at most one unrounded operand per addition);
- the value with both products unrounded, `rnd(a*b + c*d)`, which only the wide relation (before L155)
  allows.
A result outside the first set is counted; so are the inputs where the wide-only value differs from
every value of the first set (the inputs that can tell the two relations apart). The PTX and SASS
instructions that add are listed.
"""
import os
import random
import re
import shutil
import subprocess
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

DEV = "cuda"
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


FMT = {  # (mantissa bits incl. hidden, min normal exponent, max exponent)
    torch.float32: (24, -126, 127),
    torch.float64: (53, -1022, 1023),
    torch.float16: (11, -14, 15),
    torch.bfloat16: (8, -126, 127),
}


def rnd(q, dt):
    """Round a Fraction to nearest even in the format (subnormals kept; no overflow in these runs)."""
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
    return s * n * quantum


def fr(t):
    return [Fr(v) for v in t.double().tolist()]


@triton.jit
def k_forms(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr, o_ptr, FORM: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    a = tl.load(a_ptr + j)
    b = tl.load(b_ptr + j)
    c = tl.load(c_ptr + j)
    d = tl.load(d_ptr + j)
    if FORM == 0:
        r = a * b + c * d
    elif FORM == 1:
        r = a * b - c * d
    elif FORM == 2:
        x = tl.join(a, c)
        y = tl.join(b, d)
        r = tl.sum(x * y, axis=1)
    elif FORM == 3:
        r = a * b
        for t in range(1):
            r += c * d
    else:
        e = tl.load(e_ptr + j)
        f = tl.load(f_ptr + j)
        r = (a * b + c * d) + e * f
    tl.store(o_ptr + j, r)


FORMS = {0: "a*b + c*d", 1: "a*b - c*d", 2: "tl.sum(join(a,c) * join(b,d), axis=1)", 3: "r = a*b; loop: r += c*d",
         4: "(a*b + c*d) + e*f"}


def nudge(x, dt, k):
    """x moved by k units in its last place (in the format)."""
    p, emin, emax = FMT[dt]
    xs = []
    for v, kk in zip(x.double().tolist(), k):
        q = Fr(v)
        e = q.numerator.bit_length() - q.denominator.bit_length() if q != 0 else emin
        if q != 0 and Fr(2) ** e > abs(q):
            e -= 1
        xs.append(float(q + kk * Fr(2) ** (max(e, emin) - p + 1)))
    return torch.tensor(xs, dtype=torch.float64).to(dt)


def candidates(form, a, b, c, d, e, f, dt):
    R = lambda q: rnd(q, dt)  # noqa: E731
    ab, cd = a * b, (-c * d if form == 1 else c * d)
    rab, rcd = R(ab), R(cd)
    if form in (0, 1, 2, 3):
        narrow = {R(rab + rcd), R(ab + rcd), R(rab + cd)}
        wide_only = R(ab + cd)
        return narrow, wide_only
    ef = e * f
    ref = R(ef)
    narrow = set()
    firsts = {(R(rab + rcd), R(rab + rcd)), (R(ab + rcd), R(ab + rcd)), (R(rab + cd), R(rab + cd))}
    for s, _ in firsts:  # first addition's result s (rounded), then s + ef with at most one unrounded
        narrow |= {R(s + ref), R(s + ef)}
    # an unrounded first sum cannot enter the second addition (it is not a product)
    wide = {R(ab + cd + ef), R(R(ab + cd) + ef)}
    wide_only = next(iter(wide - narrow), None)
    return narrow, wide_only


def sass_adds(h):
    cub = h.asm.get("cubin")
    tool = shutil.which("cuobjdump") or os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin",
                                                      "cuobjdump")
    if not cub or not os.path.exists(tool):
        return "cuobjdump not found"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(cub)
    try:
        s = subprocess.run([tool, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)
    ops = sorted({m for m in re.findall(r"\b([DHF](?:ADD|FMA|MUL)2?(?:\.[A-Z0-9_]+)*)\b", s)})
    return ops


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    random.seed(155)
    torch.manual_seed(155)
    N = 1024
    T = 8
    out(f"{T} launches of N = {N} elements per dtype and form")
    out("| dtype | form | elements | outside the narrow set | inputs where the wide-only value differs | "
        "device = wide-only value there | PTX add/fma | SASS |")
    out("|---|---|---|---|---|---|---|---|")
    total_out = 0
    for dt in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        for form in FORMS:
            n_out = n_disc = n_wide = 0
            h = None
            for t in range(T):
                if dt == torch.float16:
                    a = (torch.rand(N) * 2 + 0.5).to(dt)
                    b = (torch.rand(N) * 2 + 0.5).to(dt)
                else:
                    a = (torch.rand(N, dtype=torch.float64) * 2 + 0.5).to(dt)
                    b = (torch.rand(N, dtype=torch.float64) * 2 + 0.5).to(dt)
                c = -nudge(a, dt, [random.randint(-3, 3) for _ in range(N)])
                d = nudge(b, dt, [random.randint(-3, 3) for _ in range(N)])
                if form == 1:
                    c = -c
                e = (torch.rand(N, dtype=torch.float64) * 2 ** -6).to(dt)
                f = (torch.rand(N, dtype=torch.float64) * 2 ** -6 + 2 ** -7).to(dt)
                o = torch.empty(N, dtype=dt, device=DEV)
                args = [x.to(DEV) for x in (a, b, c, d, e, f)]
                h = k_forms[(1,)](*args, o, form, N)
                torch.cuda.synchronize()
                A, B, C, D, E, F, O = map(fr, (a, b, c, d, e, f, o.cpu()))
                for i in range(N):
                    narrow, wide_only = candidates(form, A[i], B[i], C[i], D[i], E[i], F[i], dt)
                    if O[i] not in narrow:
                        n_out += 1
                    if wide_only is not None and wide_only not in narrow:
                        n_disc += 1
                        if O[i] == wide_only:
                            n_wide += 1
            ptx = h.asm["ptx"].split(".entry", 1)[1]
            pops = sorted({m for m in re.findall(r"\b((?:add|sub|fma|mul)\.[a-z0-9.]+)", ptx)
                           if any(x in m for x in ("f32", "f64", "f16", "bf16"))})
            total_out += n_out
            out(f"| {str(dt)[6:]} | `{FORMS[form]}` | {T * N} | {n_out} | {n_disc} | {n_wide} | "
                f"{', '.join('`' + p + '`' for p in pops)} | {', '.join('`' + p + '`' for p in sass_adds(h))} |")
    out(f"results outside the narrow set, all cases: {total_out}")
    with open(__file__.rsplit("/", 1)[0] + "/out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
