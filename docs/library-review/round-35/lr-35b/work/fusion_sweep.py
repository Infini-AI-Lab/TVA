"""lr-35b (library issue L163): which operations does ptxas fuse into a following addition?

For each operation `v = op(a, b)` below (FORMS), three launches with the same inputs store
  A: `v` alone,   B: `v + c` with `c = -(v moved by k ulp)`,   C: `c2 - v` with `c2 = v moved by k ulp`,
|k| <= 3. By Sterbenz the IEEE sum is exact there: B must be `v + c` and C `c2 - v` exactly, at the `v`
that A stores, if the addition is IEEE on the operation's value. A difference means the addition got
something else (an unrounded product: fusion). Counted per form: elements of B and C not equal to the
exact sum. The SASS of A and B is compared: the change in the number of FADD, FMUL, FFMA (and D*, H*)
instructions shows whether the addition became an FFMA. A control row repeats each form with
`enable_fp_fusion=False`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
Output: fusion_sweep.txt beside this file."""
import collections
import os
import random
import re
import struct
import subprocess

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
def val(a, b, ai, FORM: tl.constexpr):
    if FORM == 0:
        v = tl.exp(a)
    elif FORM == 1:
        v = tl.math.exp2(a)
    elif FORM == 2:
        v = tl.log(a)
    elif FORM == 3:
        v = tl.math.log2(a)
    elif FORM == 4:
        v = tl.sqrt(a)
    elif FORM == 5:
        v = tl.sqrt_rn(a)
    elif FORM == 6:
        v = tl.math.rsqrt(a)
    elif FORM == 7:
        v = tl.sin(a)
    elif FORM == 8:
        v = tl.cos(a)
    elif FORM == 9:
        v = tl.math.erf(a)
    elif FORM == 10:
        v = tl.fdiv(a, b, ieee_rounding=True)
    elif FORM == 11:
        v = tl.div_rn(a, b)
    elif FORM == 12:
        v = tl.reshape(tl.softmax(tl.reshape(a, (a.shape[0] // 4, 4)), dim=1, keep_dims=True, ieee_rounding=True), (a.shape[0],))
    elif FORM == 13:
        v = tl.fma(a, b, -0.0)
    elif FORM == 14:
        v = tl.fma(a, b, 0.0)
    elif FORM == 15:
        v = a + a
    elif FORM == 16:
        v = tl.maximum(a, b)
    elif FORM == 17:
        v = tl.floor(a)
    elif FORM == 18:
        v = ai.to(tl.float32)
    elif FORM == 19:
        v = a.to(tl.float16).to(tl.float32)
    elif FORM == 20:
        v = tl.clamp(a, -1.0, 1.0)
    elif FORM == 21:
        v = tl.abs(a)
    elif FORM == 22:
        v = a / b                     # float64 operands: div.rn.f64
    elif FORM == 23:
        v = tl.sqrt(a)                # float64
    elif FORM == 24:
        v = tl.fma(a, b, -0.0)        # float64
    elif FORM == 25:
        v = tl.ceil(a)
    elif FORM == 26:
        v = tl.cumsum(a, 0)
    elif FORM == 27:
        v = tl.math.exp(a) * 1.0      # `*` (a product): control, expected fusable
    elif FORM == 28:
        v = a / b                     # float32: a product (control, expected fused)
    elif FORM == 29:
        v = tl.sigmoid(a)             # listed as a product (control)
    elif FORM == 30:
        v = a * b                     # a product (control, expected fused)
    elif FORM == 31:
        v = tl.math.div_rn(a, b) * 1.0
    elif FORM == 32:
        v = tl.sum(tl.reshape(a, (a.shape[0] // 2, 2)), axis=1)
    elif FORM == 33:
        v = tl.math.sqrt_rn(a) - 0.0
    else:
        v = a
    return v


@triton.jit
def k_alone(a_ptr, b_ptr, ai_ptr, o_ptr, FORM: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    v = val(tl.load(a_ptr + j), tl.load(b_ptr + j), tl.load(ai_ptr + j), FORM)
    tl.store(o_ptr + j, v)


@triton.jit
def k_alone_half(a_ptr, b_ptr, ai_ptr, o_ptr, FORM: tl.constexpr, N: tl.constexpr):
    """For the forms with N // 2 results (the pair sum)."""
    j = tl.arange(0, N)
    v = val(tl.load(a_ptr + j), tl.load(b_ptr + j), tl.load(ai_ptr + j), FORM)
    tl.store(o_ptr + tl.arange(0, N // 2), v)


@triton.jit
def k_add(a_ptr, b_ptr, ai_ptr, c_ptr, o_ptr, SIDE: tl.constexpr, FORM: tl.constexpr, N: tl.constexpr,
          M: tl.constexpr):
    j = tl.arange(0, N)
    v = val(tl.load(a_ptr + j), tl.load(b_ptr + j), tl.load(ai_ptr + j), FORM)
    c = tl.load(c_ptr + tl.arange(0, M))
    if SIDE == 0:
        r = v + c
    else:
        r = c - v
    tl.store(o_ptr + tl.arange(0, M), r)


NAMES = {0: "tl.exp(a)", 1: "tl.math.exp2(a)", 2: "tl.log(a)", 3: "tl.math.log2(a)", 4: "tl.sqrt(a)",
         5: "tl.sqrt_rn(a)", 6: "tl.math.rsqrt(a)", 7: "tl.sin(a)", 8: "tl.cos(a)", 9: "tl.math.erf(a)",
         10: "tl.fdiv(a, b, ieee_rounding=True)", 11: "tl.div_rn(a, b)", 12: "tl.softmax(a, ieee_rounding=True)",
         13: "tl.fma(a, b, -0.0)", 14: "tl.fma(a, b, 0.0)", 15: "a + a", 16: "tl.maximum(a, b)",
         17: "tl.floor(a)", 18: "i.to(tl.float32) (int32 i)", 19: "a.to(tl.float16).to(tl.float32)",
         20: "tl.clamp(a, -1.0, 1.0)", 21: "tl.abs(a)", 22: "a / b (float64)", 23: "tl.sqrt(a) (float64)",
         24: "tl.fma(a, b, -0.0) (float64)", 25: "tl.ceil(a)", 26: "tl.cumsum(a, 0)",
         27: "tl.exp(a) * 1.0 [a product]", 28: "a / b [a product]", 29: "tl.sigmoid(a) [a product]",
         30: "a * b [a product]", 31: "tl.div_rn(a, b) * 1.0 [a product]", 32: "pair sum tl.sum(.., axis=1)",
         33: "tl.sqrt_rn(a) - 0.0"}
F64 = {22, 23, 24}
HALF = {32}


def to_f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ulp32(x):
    x = abs(x)
    if x == 0 or x < 2.0 ** -126:
        return 2.0 ** -149
    e = struct.unpack("<I", struct.pack("<f", x))[0] >> 23
    return 2.0 ** (e - 127 - 23)


def ulp64(x):
    x = abs(x)
    if x == 0 or x < 2.0 ** -1022:
        return 2.0 ** -1074
    e = struct.unpack("<Q", struct.pack("<d", x))[0] >> 52
    return 2.0 ** (e - 1023 - 52)


def inputs(form, n, rng):
    """(a, b, ai) lists for a form."""
    U = rng.uniform
    if form in (0, 27):
        a = [U(-87, 88) if rng.random() < 0.8 else U(-103.9, -87.4) for _ in range(n)]   # incl. subnormal results
    elif form == 1:
        a = [U(-125, 127) for _ in range(n)]
    elif form in (2, 3, 4, 5, 6, 23, 33):
        a = [U(1, 2) * 2.0 ** rng.randint(-126, 126) for _ in range(n)]
    elif form in (7, 8):
        a = [U(-100, 100) for _ in range(n)]
    elif form == 9:
        a = [U(-4, 4) for _ in range(n)]
    elif form == 15:
        a = [U(1, 2) * 2.0 ** rng.randint(-149, 127) * rng.choice((1, -1)) for _ in range(n)]
    elif form in (29,):
        a = [U(-20, 20) for _ in range(n)]
    else:
        a = [U(0.5, 2) * rng.choice((1, -1)) * 2.0 ** rng.randint(-20, 20) for _ in range(n)]
    b = [U(0.5, 2) * rng.choice((1, -1)) * 2.0 ** rng.randint(-20, 20) for _ in range(n)]
    ai = [rng.randint(-2 ** 31, 2 ** 31 - 1) for _ in range(n)]
    return a, b, ai


def sass_counts(h):
    path = os.path.join(HERE, "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(h.asm["cubin"])
    try:
        s = subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)
    ops = re.findall(r"\b([DHF](?:ADD|MUL|FMA)2?)(?:\.[A-Z0-9_]+)*\b", s)
    return collections.Counter(ops)


def run(form, rng, n, fusion):
    dt = torch.float64 if form in F64 else torch.float32
    ulp = ulp64 if form in F64 else ulp32
    m = n // 2 if form in HALF else n
    a, b, ai = inputs(form, n, rng)
    da = torch.tensor(a, dtype=torch.float64).to(dt).cuda()
    db = torch.tensor(b, dtype=torch.float64).to(dt).cuda()
    dai = torch.tensor(ai, dtype=torch.int64).to(torch.int32).cuda()
    o = torch.empty(m, dtype=dt, device="cuda")
    kern = k_alone_half if form in HALF else k_alone
    hA = kern[(1,)](da, db, dai, o, form, n, enable_fp_fusion=fusion)
    torch.cuda.synchronize()
    V = o.double().cpu().tolist()
    bad = 0
    total = 0
    example = None
    hB = None
    for side in (0, 1):
        cs = []
        for v in V:
            if v != v or abs(v) == float("inf"):
                cs.append(0.0)
                continue
            k = rng.randint(-3, 3)
            w = v + k * ulp(v)
            cs.append(-w if side == 0 else w)
        dc = torch.tensor(cs, dtype=torch.float64).to(dt).cuda()
        C = dc.double().cpu().tolist()
        r = torch.empty(m, dtype=dt, device="cuda")
        h = k_add[(1,)](da, db, dai, dc, r, side, form, n, m, enable_fp_fusion=fusion)
        torch.cuda.synchronize()
        if side == 0:
            hB = h
        R = r.double().cpu().tolist()
        for i in range(m):
            v = V[i]
            if v != v or abs(v) == float("inf"):
                continue
            want = v + C[i] if side == 0 else C[i] - v    # exact (Sterbenz), in the format
            if dt == torch.float32:
                assert to_f32(want) == want
            total += 1
            if R[i] != want:
                bad += 1
                if example is None:
                    example = (side, a[i % n], b[i % n], v, C[i], R[i], want)
    return total, bad, example, sass_counts(hA), sass_counts(hB)


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    rng = random.Random(35)
    n = 4096
    out(f"{n} elements per form and side (B: v + c, C: c2 - v), |k| <= 3 ulp")
    out("")
    out("| form | fusion | elements | sums not exact | SASS change A -> B (v + c) | example (side, a, b, v, c, got, IEEE) |")
    out("|---|---|---|---|---|---|")
    for form in NAMES:
        for fusion in (True, False):
            total, bad, ex, cA, cB = run(form, rng, n, fusion)
            keys = sorted(set(cA) | set(cB))
            delta = ", ".join(f"{k} {cB[k] - cA[k]:+d}" for k in keys if cB[k] != cA[k]) or "none"
            out(f"| `{NAMES[form]}` | {fusion} | {total} | {bad} | {delta} | {ex if ex else ''} |")
    with open(os.path.join(HERE, "fusion_sweep.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
