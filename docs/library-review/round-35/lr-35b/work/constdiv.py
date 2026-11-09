"""lr-35b (survey, `DivFull`; library issue L163): float32 `x / D` with `D` a compile-time constant
(a `tl.constexpr` float or int), which ptxas may fold into a multiply by a constant reciprocal.

For each D: the quotient alone `q = x / D` against the card `DivFull` (within 2^-22 |x/D| + 2^-148 of
x/D when |x/D| <= 2^126), and `x / D + c` (c nearly cancelling q) against `AddRel true q c`, using
the exact checker of evidence/div-fusion (copied to rerun_divfusion/run.py). The SASS of `x / D` is
listed. Also D passed at runtime, for comparison.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
Output: constdiv.txt beside this file."""
import importlib.util
import os
import random
import re
import subprocess
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dfrun", os.path.join(HERE, "rerun_divfusion", "run.py"))
DF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DF)
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def k_q(x_ptr, d_ptr, q_ptr, D: tl.constexpr, RT: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(x_ptr + j)
    if RT:
        q = x / tl.load(d_ptr + j)
    else:
        q = x / D
    tl.store(q_ptr + j, q)


@triton.jit
def k_r(x_ptr, d_ptr, c_ptr, r_ptr, D: tl.constexpr, RT: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(x_ptr + j)
    if RT:
        q = x / tl.load(d_ptr + j)
    else:
        q = x / D
    tl.store(r_ptr + j, q + tl.load(c_ptr + j))


def sass_ops(h):
    path = os.path.join(HERE, "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(h.asm["cubin"])
    try:
        s = subprocess.run([DF.CUOBJDUMP, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)
    return [ln.split(";")[0].split("*/")[-1].strip() for ln in s.splitlines()
            if re.search(r"\b(FMUL|FFMA|FADD|MUFU|FSETP|FCHK)\b", ln)]


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    rng = random.Random(3535)
    N = 4096
    DS = [3.0, 7, 0.1, 1.5 * 2.0 ** 126, 3e38, 2.0 ** 127, float(DF.F32.maxf), 2.0 ** -126, 1e-39, 1e-40,
          2.0 ** -130, 2.0 ** -149, -3.0]
    out("| D | how | elements (|x/D| <= 2^126) | outside DivFull | x/D + c outside AddRel | not the IEEE sum | worst err / card |")
    out("|---|---|---|---|---|---|---|")
    sass_rows = []
    for D in DS:
        Df = DF.to_f32(float(D))
        # dividends: quotient magnitudes spread over the whole range, and some beyond
        xs = []
        for _ in range(N):
            e = rng.randint(-149, 127)
            v = rng.uniform(1, 2) * 2.0 ** e * rng.choice((1, -1))
            xs.append(DF.to_f32(v) if abs(v) < 3.4e38 else 3.0e38)
        x = torch.tensor(xs, dtype=torch.float32).cuda()
        d = torch.full((N,), Df, dtype=torch.float32).cuda()
        for rt in (False, True):
            q = torch.empty(N, dtype=torch.float32, device="cuda")
            h = k_q[(1,)](x, d, q, D, rt, N)
            torch.cuda.synchronize()
            if not rt:
                sass_rows.append((D, sass_ops(h)))
            Q = q.double().cpu().tolist()
            X = x.double().cpu().tolist()
            n_in = n_out = 0
            worst = 0.0
            for xi, qi in zip(X, Q):
                ex = Fr(xi) / Fr(Df)
                if abs(ex) > 2 ** 126:
                    continue
                n_in += 1
                card = Fr(1, 2 ** 22) * abs(ex) + Fr(1, 2 ** 148)
                if qi != qi or abs(qi) == float("inf"):
                    n_out += 1
                    worst = float("inf")
                    continue
                err = abs(Fr(qi) - ex)
                if err > card:
                    n_out += 1
                worst = max(worst, float(err / card))
            # the following addition, c nearly cancelling q
            cs = [DF.nudge(DF.F32, -qi, rng.randint(-3, 3)) if (qi == qi and abs(qi) != float("inf")) else 1.0
                  for qi in Q]
            c = torch.tensor(cs, dtype=torch.float64).to(torch.float32).cuda()
            C = c.double().cpu().tolist()
            r = torch.empty(N, dtype=torch.float32, device="cuda")
            k_r[(1,)](x, d, c, r, D, rt, N)
            torch.cuda.synchronize()
            R = r.double().cpu().tolist()
            a_out = sum(1 for qi, ci, ri in zip(Q, C, R) if not DF.addrel(DF.F32, qi, ci, ri))
            fused = sum(1 for qi, ci, ri in zip(Q, C, R) if not DF.same(DF.ieee_add(DF.F32, qi, ci), ri))
            out(f"| {D!r} | {'runtime' if rt else 'constexpr'} | {n_in} | {n_out} | {a_out} | {fused} | {worst:.3f} |")
    out("")
    out("SASS of `x / D` (constexpr D), arithmetic only, distinct instructions (registers written R):")
    for D, ops in sass_rows:
        uniq = sorted({re.sub(r"\bR\d+\b", "R", o) for o in ops})
        out(f"- {D!r}: " + " ; ".join(uniq))
    with open(os.path.join(HERE, "constdiv.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
