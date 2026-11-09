"""lr-36a, round 36 (L166): is `tl.fdiv(x, D, ieee_rounding=True) + c` (and `tl.div_rn(x, D) + c`) with a
compile-time divisor `D` still the IEEE quotient followed by the IEEE sum? A divisor that ptxas could turn
into a multiply (powers of two) is the case the evidence's runtime divisors do not reach.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -I <this file>
Output: divrn_const.out.txt beside this file."""
import os
import re
import subprocess

import numpy as np
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
CUOBJDUMP = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def k_fdiv(X, C, O, D: tl.constexpr, SUB: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * N + tl.arange(0, N)
    x = tl.load(X + i)
    c = tl.load(C + i)
    q = tl.fdiv(x, D, ieee_rounding=True)
    if SUB:
        r = c - q
    else:
        r = q + c
    tl.store(O + i, r)


@triton.jit
def k_divrn(X, C, O, D: tl.constexpr, SUB: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * N + tl.arange(0, N)
    x = tl.load(X + i)
    c = tl.load(C + i)
    q = tl.div_rn(x, tl.full((N,), D, tl.float32))
    if SUB:
        r = c - q
    else:
        r = q + c
    tl.store(O + i, r)


def sass_ops(h):
    path = os.path.join(HERE, "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(h.asm["cubin"])
    try:
        s = subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)
    ops = re.findall(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?P\w+\s+)?([A-Z][A-Z0-9_.]*)", s, re.M)
    return {op: ops.count(op) for op in ("FFMA", "FADD", "FMUL", "MUFU.RCP", "CALL.REL.NOINC") if op in ops}


def inputs(D, rng, n):
    """Adversarial pairs: x whose quotient by D is inexact in float32 (subnormal results, overflow), and c
    nearly cancelling the quotient, so an unrounded quotient fused into the addition shows."""
    f32 = np.float32
    xs, cs = [], []
    # subnormal-result quotients: x = odd multiples of the smallest subnormal, scaled so x/D is inexact
    tiny = np.arange(1, 2 * 4096, 2, dtype=np.int64)
    base = (tiny * 2.0 ** -149).astype(f32)
    for x in base:
        q = f32(x) / f32(D)  # IEEE quotient (numpy float32)
        xs.append(x)
        cs.append(-q + f32(np.nextafter(f32(0), f32(1))) * rng.choice([-1, 0, 1]))
    # large x near the top, for divisors below 1 (overflowing quotients)
    big = (rng.uniform(1.0, 2.0, 4096) * 2.0 ** 127).astype(f32)
    for x in big:
        xs.append(x)
        cs.append(f32(-x / f32(D) if abs(float(D)) >= 1 else -float(x) * 1.5))
    # random normal pairs with cancellation
    r = (rng.standard_normal(n - len(xs)) * 2.0 ** rng.integers(-100, 100, n - len(xs))).astype(f32)
    for x in r:
        xs.append(x)
        cs.append(-(f32(x) / f32(D)) * f32(1 + 2 ** -20 * rng.standard_normal()))
    return np.array(xs[:n], dtype=f32), np.array(cs[:n], dtype=f32)


def main():
    rng = np.random.default_rng(36)
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    N, n = 1024, 1 << 16
    for D in (2.0, 4.0, 0.5, 0.25, -2.0, 1.0, 3.0, 2.0 ** -126, 2.0 ** 100):
        x, c = inputs(D, rng, n)
        X = torch.from_numpy(x).cuda()
        Cc = torch.from_numpy(c).cuda()
        with np.errstate(all="ignore"):
            q = x / np.float32(D)
            ref_add = q + c
            ref_sub = c - q
        for name, kern in (("tl.fdiv(ieee_rounding=True)", k_fdiv), ("tl.div_rn", k_divrn)):
            for sub in (False, True):
                O = torch.empty_like(X)
                h = kern[(n // N,)](X, Cc, O, D, sub, N)
                torch.cuda.synchronize()
                o = O.cpu().numpy()
                ref = ref_sub if sub else ref_add
                ob, rb = o.view(np.uint32), ref.view(np.uint32)
                bad = int(np.sum((ob != rb) & ~(np.isnan(o) & np.isnan(ref))))
                divs = sorted(set(re.findall(r"\b(div\.[a-z0-9.]+|rcp\.[a-z0-9.]+|mul\.[a-z0-9.]+)", h.asm["ptx"])))
                form = "c - q" if sub else "q + c"
                out(f"- D = {D!r}, {name}, {form}: {bad} of {n} differ from the IEEE result; PTX {divs}; "
                    f"SASS {sass_ops(h)}")
    with open(os.path.join(HERE, "divrn_const.out.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
