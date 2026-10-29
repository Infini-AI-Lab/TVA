"""TVA library evidence (library issue L4): the approximate float32 math functions on sm_90.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-math-f32/run.py
Output: evidence/num-math-f32/out.txt (README.md says what it shows).

For every float32 bit pattern x (all 2^32, in 64 chunks of 2^26) the device computes `tl.exp`,
`tl.math.exp2`, `tl.log`, `tl.math.log2`, `tl.sqrt` and `tl.math.rsqrt`. Each result is compared
with the real function, computed in float64 by numpy on the host (glibc's `exp`, `exp2`, `log`,
`log2` are within one float64 ulp, `sqrt` is correctly rounded), and checked against the fact the
library states (`TVA.Rules.Num`, docs/rules/num.md):

  ExpF32   for every X >= 0, x in [-X, X]:   |r - e^x|      <= (2^-22 + 2^-23 X) e^x + 2^-140
  Exp2F32  every x:                          |r - 2^x|      <= 2^-22 2^x + 2^-126
  LogF32   x > 0:                            |r - ln x|     <= 2^-22 |ln x| + 2^-149
  Log2F32  x > 0:                            |r - log2 x|   <= 2^-22 |log2 x| + 2^-149
  SqrtF32  x = 0 or x >= 2^-126:             |r - sqrt x|   <= 2^-22 sqrt x + 2^-149
  RsqrtF32 x >= 2^-126:                      |r - 1/sqrt x| <= 2^-22 / sqrt x + 2^-149

each for finite x whose reference value leaves room in float32, (1 + rel)|ref| + abs <= maxVal
(the condition of `MathAcc`), and the result must then be finite. For ExpF32 the bound grows
with X, so the tightest instance for x is X = |x|. The comparison is in float64: `ref` is within
2^-52 |ref| of the real value and the difference is rounded once, so the script checks
err (1 + 2^-52) + 2^-51 |ref| <= bound. It also prints the special values (`exp(-inf)`, `exp2(-inf)`
are the `MathAt` facts) and the instructions in the PTX.
"""
import re
import sys
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def funcs(x_ptr, e_ptr, e2_ptr, l_ptr, l2_ptr, s_ptr, rs_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32, bitcast=True)
    tl.store(e_ptr + offs, tl.exp(x), mask=m)
    tl.store(e2_ptr + offs, tl.math.exp2(x), mask=m)
    tl.store(l_ptr + offs, tl.log(x), mask=m)
    tl.store(l2_ptr + offs, tl.math.log2(x), mask=m)
    tl.store(s_ptr + offs, tl.sqrt(x), mask=m)
    tl.store(rs_ptr + offs, tl.math.rsqrt(x), mask=m)


@triton.jit
def one(x_ptr, o_ptr, n, FN: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if FN == 0:
        y = tl.exp(x)
    elif FN == 1:
        y = tl.math.exp2(x)
    elif FN == 2:
        y = tl.log(x)
    elif FN == 3:
        y = tl.math.log2(x)
    elif FN == 4:
        y = tl.sqrt(x)
    else:
        y = tl.math.rsqrt(x)
    tl.store(o_ptr + offs, y, mask=m)


NAMES = ["exp", "exp2", "log", "log2", "sqrt", "rsqrt"]
FACT = {"exp": "ExpF32", "exp2": "Exp2F32", "log": "LogF32", "log2": "Log2F32", "sqrt": "SqrtF32",
        "rsqrt": "RsqrtF32"}
MAXV = float(np.finfo(np.float32).max)
N126 = 2.0 ** -126


def reference(name, x):
    with np.errstate(all="ignore"):
        return {"exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2, "sqrt": np.sqrt,
                "rsqrt": lambda v: 1.0 / np.sqrt(v)}[name](x)


def bound_terms(name, x):
    """(dom, rel, abs) of the fact, per element."""
    fin = np.isfinite(x)
    if name == "exp":
        return fin, 2.0 ** -22 + 2.0 ** -23 * np.abs(x), 2.0 ** -140
    if name == "exp2":
        return fin, np.full_like(x, 2.0 ** -22), 2.0 ** -126
    if name in ("log", "log2"):
        return fin & (x > 0), np.full_like(x, 2.0 ** -22), 2.0 ** -149
    if name == "sqrt":
        return fin & ((x == 0) | (x >= N126)), np.full_like(x, 2.0 ** -22), 2.0 ** -149
    return fin & (x >= N126), np.full_like(x, 2.0 ** -22), 2.0 ** -149


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    CH = 1 << 26
    st = {n: dict(checked=0, viol=0, nonfin=0, ratio=0.0, at=None, relmax=0.0, relat=None) for n in NAMES}
    for c in range(64):
        lo = c * CH
        b64 = torch.arange(lo, lo + CH, dtype=torch.int64, device="cuda")
        bits = torch.where(b64 >= 2 ** 31, b64 - 2 ** 32, b64).to(torch.int32)
        outs = [torch.empty(CH, dtype=torch.float32, device="cuda") for _ in NAMES]
        funcs[(CH // 1024,)](bits, *outs, CH, B=1024)
        x = bits.view(torch.float32).cpu().numpy().astype(np.float64)
        for name, o in zip(NAMES, outs):
            r = o.cpu().numpy().astype(np.float64)
            ref = reference(name, x)
            dom, rel, ab = bound_terms(name, x)
            aref = np.abs(ref)
            room = dom & np.isfinite(ref) & ((1 + rel) * aref + ab <= MAXV)
            s = st[name]
            s["checked"] += int(room.sum())
            s["nonfin"] += int((room & ~np.isfinite(r)).sum())
            with np.errstate(all="ignore"):
                err = np.abs(r - ref)
                lhs = err * (1 + 2.0 ** -52) + 2.0 ** -51 * aref
                bound = rel * aref + ab
                ratio = np.where(room & np.isfinite(r), lhs / bound, 0.0)
                relerr = np.where(room & np.isfinite(r) & (aref >= N126), err / np.where(aref > 0, aref, 1), 0.0)
            s["viol"] += int((room & np.isfinite(r) & (lhs > bound)).sum())
            i = int(np.argmax(ratio))
            if ratio[i] > s["ratio"]:
                s["ratio"], s["at"] = float(ratio[i]), float(x[i])
            j = int(np.argmax(relerr))
            if relerr[j] > s["relmax"]:
                s["relmax"], s["relat"] = float(relerr[j]), float(x[j])
        print(f"chunk {c} done", file=sys.stderr, flush=True)
    print("\n## Every float32 input, against the stated fact")
    print("| function | fact | inputs where the fact applies | outside the bound | non-finite results | largest error / bound (at x) | largest relative error, normal reference (at x) |")
    print("|---|---|---|---|---|---|---|")
    for n in NAMES:
        s = st[n]
        print(f"| {n} | {FACT[n]} | {s['checked']} | {s['viol']} | {s['nonfin']} | {s['ratio']:.4f} ({s['at']!r}) | "
              f"2^{np.log2(s['relmax']):.3f} ({s['relat']!r}) |")
    print("\n## Special values (result bits)")
    sp = torch.tensor([float("-inf"), float("inf"), float("nan"), 0.0, -0.0, -1.0, 2.0 ** -149, 2.0 ** -127],
                      dtype=torch.float32, device="cuda")
    outs = [torch.empty(8, dtype=torch.float32, device="cuda") for _ in NAMES]
    funcs[(1,)](sp.view(torch.int32), *outs, 8, B=8)
    hdr = ["-inf", "+inf", "nan", "+0", "-0", "-1", "2^-149", "2^-127"]
    print("| function | " + " | ".join(hdr) + " |")
    print("|---" * (len(hdr) + 1) + "|")
    for n, o in zip(NAMES, outs):
        print(f"| {n} | " + " | ".join(f"`{b & 0xffffffff:08x}` ({v:g})" for b, v in zip(o.view(torch.int32).tolist(), o.tolist())) + " |")
    print("\n## Instructions in the PTX (one function per kernel)")
    xx = torch.ones(16, dtype=torch.float32, device="cuda")
    oo = torch.empty_like(xx)
    for k, n in enumerate(NAMES):
        h = one[(1,)](xx, oo, 16, FN=k, B=16)
        ops = sorted(set(re.findall(r"^\s*([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)\s", h.asm["ptx"], re.M)))
        keep = [o for o in ops if o.split(".")[0] in ("ex2", "lg2", "sqrt", "rsqrt", "rcp", "fma", "mul", "add", "sub", "div", "selp", "setp")]
        print(f"- `{n}`: " + ", ".join(f"`{o}`" for o in keep))


if __name__ == "__main__":
    main()
