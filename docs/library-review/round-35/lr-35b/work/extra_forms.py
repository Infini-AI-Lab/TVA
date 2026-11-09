"""lr-35b (library issue L163): more spellings of a float32 quotient feeding an addition.

Each form's quotient(s) are stored by a first launch (q = a / b, q2 = b / a); a second launch stores
the form's result r. Checked exactly (the AddRel checker of evidence/div-fusion, copied to
rerun_divfusion/run.py):
  E0 `(a / b) * d + c`, E1 `d * (a / b) + c`: `tl.mul_add q d c` (RN(RN(q*d) + c) or RN(q*d + c)), or
     the fusable `AddRel` at the rounded product (the model may also write `tl.addc (q*d) c`);
  E2 `tl.where(m, a / b, b / a) + c`: AddRel at the where's value;
  E3 `(a / b) + (a / b)`: AddRel q q;
  E4 `i / j + c` (int32 i, j): AddRel q c;
  E5 `tl.maximum(a / b, 0.0) + c`: AddRel at the maximum;
  E6 `tl.where(FLAG, a / b, 0.0) + c` with FLAG a constexpr True: AddRel q c.
Regimes: nearly cancelling (c = -(the addend's base) moved by up to 3 ulp), random patterns, special values.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
Output: extra_forms.txt beside this file."""
import importlib.util
import itertools
import os
import random
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dfrun", os.path.join(HERE, "rerun_divfusion", "run.py"))
DF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DF)
OUT = []
F = DF.F32


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def k_q(a_ptr, b_ptr, i_ptr, j_ptr, q_ptr, q2_ptr, FORM: tl.constexpr, N: tl.constexpr):
    k = tl.arange(0, N)
    a = tl.load(a_ptr + k)
    b = tl.load(b_ptr + k)
    if FORM == 4:
        q = tl.load(i_ptr + k) / tl.load(j_ptr + k)
    else:
        q = a / b
    tl.store(q_ptr + k, q)
    tl.store(q2_ptr + k, b / a)


@triton.jit
def k_r(a_ptr, b_ptr, i_ptr, j_ptr, d_ptr, c_ptr, m_ptr, r_ptr, FORM: tl.constexpr, FLAG: tl.constexpr,
        N: tl.constexpr):
    k = tl.arange(0, N)
    a = tl.load(a_ptr + k)
    b = tl.load(b_ptr + k)
    c = tl.load(c_ptr + k)
    d = tl.load(d_ptr + k)
    if FORM == 0:
        r = (a / b) * d + c
    elif FORM == 1:
        r = d * (a / b) + c
    elif FORM == 2:
        r = tl.where(tl.load(m_ptr + k) != 0, a / b, b / a) + c
    elif FORM == 3:
        r = (a / b) + (a / b)
    elif FORM == 4:
        r = tl.load(i_ptr + k) / tl.load(j_ptr + k) + c
    elif FORM == 5:
        r = tl.maximum(a / b, 0.0) + c
    else:
        r = tl.where(FLAG, a / b, 0.0) + c
    tl.store(r_ptr + k, r)


NAMES = {0: "(a / b) * d + c", 1: "d * (a / b) + c", 2: "tl.where(m, a / b, b / a) + c", 3: "(a / b) + (a / b)",
         4: "i / j + c (int32)", 5: "tl.maximum(a / b, 0.0) + c", 6: "tl.where(True, a / b, 0.0) + c (constexpr)"}
SPEC = [0.0, -0.0, DF.INF, -DF.INF, float("nan"), 1.0, -1.0, 3.0, 2.0 ** -149, 2.0 ** 127, float(F.maxf), 2.0 ** -126]


def rnd(t):
    """RN to float32 of an exact Fraction, as a float (NaN/inf handled by the caller)."""
    r = F.rnd(t)
    return r if abs(r) == DF.INF else float(r)


def mul_add_ok(q, d, c, r):
    """tl.mul_add q d c: two roundings or one (finite operands); else AddRel at the rounded product."""
    vals = [float(x) for x in (q, d, c)]
    if all(v == v and abs(v) != DF.INF for v in vals):
        p = Fr(q) * Fr(d)
        two = DF.ieee_add(F, rnd(p) if abs(rnd(p)) == DF.INF else float(rnd(p)), c)
        one = rnd(p + Fr(c)) if (p + Fr(c)) != 0 else 0.0
        if DF.same(two, r) or (one == r and r != 0) or (r == 0 and p + Fr(c) == 0):
            return True
    qd = q * d   # Python's double product of two floats is exact enough here only as a fallback
    u = DF.cvt(F, qd) if qd == qd else qd
    return DF.addrel(F, u, c, r)


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    rng = random.Random(3536)
    N = 4096
    out("| form | regime | elements | outside the model | not the IEEE sum |")
    out("|---|---|---|---|---|")
    total_out = 0
    for form in NAMES:
        for regime in ("cancel", "random", "special"):
            if regime == "special":
                combos = list(itertools.product(SPEC, SPEC, SPEC))
                combos = (combos * (N // len(combos) + 1))[:N]
                A = [x for x, _, _ in combos]
                B = [y for _, y, _ in combos]
                Cfix = [z for _, _, z in combos]
            elif regime == "cancel":
                A = [rng.uniform(0.5, 2) * rng.choice((1, -1)) for _ in range(N)]
                B = [rng.uniform(0.5, 2) for _ in range(N)]
                Cfix = None
            else:
                A = [DF.f32_pattern(rng) for _ in range(N)]
                B = [DF.f32_pattern(rng) for _ in range(N)]
                Cfix = None
            I = [rng.randint(-2 ** 24, 2 ** 24) for _ in range(N)]
            J = [rng.choice((1, -1)) * rng.randint(1, 2 ** 12) for _ in range(N)]
            D = [rng.uniform(0.5, 2) * rng.choice((1, -1)) for _ in range(N)]
            M = [rng.random() < 0.5 for _ in range(N)]
            t = lambda xs, dt=torch.float32: torch.tensor(xs, dtype=torch.float64).to(dt).cuda()  # noqa: E731
            da, db, dd = t(A), t(B), t(D)
            di = torch.tensor(I, dtype=torch.int32).cuda()
            dj = torch.tensor(J, dtype=torch.int32).cuda()
            dm = torch.tensor(M, dtype=torch.int8).cuda()
            q = torch.empty(N, dtype=torch.float32, device="cuda")
            q2 = torch.empty(N, dtype=torch.float32, device="cuda")
            k_q[(1,)](da, db, di, dj, q, q2, form, N)
            torch.cuda.synchronize()
            Q, Q2 = q.double().cpu().tolist(), q2.double().cpu().tolist()
            Dv = dd.double().cpu().tolist()
            # the addition's first operand as the model writes it
            if form in (0, 1):
                base = [DF.cvt(F, x * y) for x, y in zip(Q, Dv)]
            elif form == 2:
                base = [x if mm else y for x, y, mm in zip(Q, Q2, M)]
            elif form == 5:
                base = [x if x >= 0 else 0.0 for x in Q]   # maximum(q, 0) is maxnum (SEMANTICS.md, `FloatV.MaxRel`): a NaN q gives 0
            else:
                base = Q
            if Cfix is not None:
                C = Cfix
            elif regime == "cancel":
                C = [DF.nudge(F, -u, rng.randint(-3, 3)) if (u == u and abs(u) != DF.INF) else 1.0 for u in base]
            else:
                C = [(abs(u) * 2.0 ** rng.randint(-30, 30) if (u == u and abs(u) not in (0, DF.INF)) else 1.0)
                     * rng.choice((1, -1)) for u in base]
            dc = t(C)
            C = dc.double().cpu().tolist()
            r = torch.empty(N, dtype=torch.float32, device="cuda")
            k_r[(1,)](da, db, di, dj, dd, dc, dm, r, form, True, N)
            torch.cuda.synchronize()
            R = r.double().cpu().tolist()
            n_out = n_f = 0
            for i in range(N):
                if form in (0, 1):
                    ok = mul_add_ok(Q[i], Dv[i], C[i], R[i])
                    ieee = DF.ieee_add(F, base[i], C[i])
                elif form == 3:
                    ok = DF.addrel(F, Q[i], Q[i], R[i])
                    ieee = DF.ieee_add(F, Q[i], Q[i])
                elif form == 5:
                    # tl.maximum may return either zero for zeros; take both
                    ok = DF.addrel(F, base[i], C[i], R[i]) or (base[i] == 0 and DF.addrel(F, -base[i], C[i], R[i]))
                    ieee = DF.ieee_add(F, base[i], C[i])
                else:
                    ok = DF.addrel(F, base[i], C[i], R[i])
                    ieee = DF.ieee_add(F, base[i], C[i])
                n_out += not ok
                n_f += not DF.same(ieee, R[i])
            total_out += n_out
            out(f"| `{NAMES[form]}` | {regime} | {N} | {n_out} | {n_f} |")
    out("")
    out(f"outside the model, all: {total_out}")
    with open(os.path.join(HERE, "extra_forms.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
