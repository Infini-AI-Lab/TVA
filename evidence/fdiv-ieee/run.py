"""Library issue L166: `tl.fdiv(x, y, ieee_rounding=True)` and `tl.softmax(..., ieee_rounding=True)`.

Triton 3.8.0: `semantic.py:331-341` gives `ieee_rounding=True` `tt.precise_divf`, the op `tl.div_rn`
emits (`math.py:203-210`), lowered to `llvm.nvvm.div.rn.f` (`nvidia-ElementwiseOpToLLVM.cpp:886-887`);
`standard.py:54-64` ends `tl.softmax` in `math.fdiv(num, den, ieee_rounding)`. This script checks:
  A. the PTX of `tl.fdiv(x, y, ieee_rounding=True)` on float32, float16, bfloat16 and float64
     operands, of `tl.div_rn` and of `tl.softmax` with each `ieee_rounding`;
  B. the float32 results: every pair of a set of special and random bit patterns, and 2^24 random
     pairs, bit for bit equal to `tl.div_rn`'s and to the IEEE quotient (numpy float32 division),
     NaNs compared as NaN;
  C. `tl.fdiv(x, y, ieee_rounding=True) + c` and `c - tl.fdiv(...)` with `enable_fp_fusion` on: every
     result equals the IEEE sum of the IEEE quotient (not fused), on inputs where a fused multiply
     would differ; the SASS of the division's final step;
  D. `tl.softmax(x, ieee_rounding=True)` on rows of 16: bit for bit `num / den` with the IEEE
     division, `num` and `den` read back from the same kernel.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/fdiv-ieee/run.py
Output: out.txt beside this file."""
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
def k_div(x, y, o, r, N: tl.constexpr, MODE: tl.constexpr):
    j = tl.program_id(0) * N + tl.arange(0, N)
    a = tl.load(x + j)
    b = tl.load(y + j)
    if MODE == 0:
        v = tl.fdiv(a, b, ieee_rounding=True)
    elif MODE == 1:
        v = tl.div_rn(a, b)
    else:
        v = tl.fdiv(a, b)
    tl.store(o + j, v)


@triton.jit
def k_add(x, y, c, o, N: tl.constexpr, SUB: tl.constexpr):
    j = tl.program_id(0) * N + tl.arange(0, N)
    a = tl.load(x + j)
    b = tl.load(y + j)
    d = tl.load(c + j)
    if SUB:
        v = d - tl.fdiv(a, b, ieee_rounding=True)
    else:
        v = tl.fdiv(a, b, ieee_rounding=True) + d
    tl.store(o + j, v)


@triton.jit
def k_softmax(x, o, num_o, den_o, R: tl.constexpr, C: tl.constexpr, IEEE: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    a = tl.load(x + r * C + c)
    tl.store(o + r * C + c, tl.softmax(a, dim=1, keep_dims=True, ieee_rounding=IEEE))
    z = a - tl.max(a, 1, keep_dims=True)
    num = tl.exp(z)
    tl.store(num_o + r * C + c, num)
    tl.store(den_o + r * C + c, tl.broadcast_to(tl.sum(num, 1, keep_dims=True), (R, C)))


def divs(ptx):
    return sorted(set(re.findall(r"\b(div\.[a-z0-9.]+|rcp\.[a-z0-9.]+)", ptx)))


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


def same_bits(a, b):
    a = a.view(np.uint32)
    b = b.view(np.uint32)
    an = np.isnan(a.view(np.float32))
    bn = np.isnan(b.view(np.float32))
    return int(np.count_nonzero(~((a == b) | (an & bn))))


def patterns(rng):
    sp = np.array([0.0, -0.0, 1.0, -1.0, 3.0, 0.1, -7.0, np.inf, -np.inf, np.nan, 2.0 ** 126, 2.0 ** 127,
                   2.0 ** -126, 2.0 ** -149, -(2.0 ** -149), 3.4e38, 1.0 + 2.0 ** -23, 2.0 ** 100, 2.0 ** -100],
                  dtype=np.float32).view(np.uint32)
    return np.concatenate([sp, rng.integers(0, 2 ** 32, 256 - sp.size, dtype=np.uint64).astype(np.uint32)])


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, numpy {np.__version__}, {torch.cuda.get_device_name()}")
    out("")
    rng = np.random.default_rng(166)
    N = 1024
    # A. PTX.
    out("## A. PTX division instructions")
    for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        x = torch.ones(N, dtype=dt, device="cuda")
        odt = torch.float64 if dt == torch.float64 else torch.float32
        o = torch.empty(N, dtype=odt, device="cuda")
        for mode, name in ((0, "tl.fdiv(ieee_rounding=True)"), (1, "tl.div_rn"), (2, "tl.fdiv")):
            try:
                h = k_div[(1,)](x, x, o, None, N, mode)
                torch.cuda.synchronize()
                out(f"- {name}, {str(dt)[6:]} operands: {', '.join(divs(h.asm['ptx']))}; result {h.asm['ttir'].count('tt.precise_divf')} `tt.precise_divf`")
            except Exception as e:   # noqa: BLE001
                out(f"- {name}, {str(dt)[6:]} operands: rejected ({type(e).__name__}: {str(e).splitlines()[0][:100]})")
    xs = torch.randn(16, 16, device="cuda")
    for ieee in (False, True):
        h = k_softmax[(1,)](xs, torch.empty_like(xs), torch.empty_like(xs), torch.empty_like(xs), 16, 16, ieee)
        out(f"- tl.softmax(ieee_rounding={ieee}), float32: {', '.join(divs(h.asm['ptx']))}")
    out("")
    # B. float32 results.
    out("## B. float32 quotients, bit for bit")
    pat = patterns(rng)
    xa = np.repeat(pat, pat.size)
    ya = np.tile(pat, pat.size)
    xr = rng.integers(0, 2 ** 32, 1 << 24, dtype=np.uint64).astype(np.uint32)
    yr = rng.integers(0, 2 ** 32, 1 << 24, dtype=np.uint64).astype(np.uint32)
    for label, xb, yb in (("all pairs of 256 patterns", xa, ya), ("2^24 random pairs", xr, yr)):
        x = torch.from_numpy(xb.view(np.int32)).cuda().view(torch.float32)
        y = torch.from_numpy(yb.view(np.int32)).cuda().view(torch.float32)
        res = {}
        for mode in (0, 1, 2):
            o = torch.empty_like(x)
            k_div[(x.numel() // N,)](x, y, o, None, N, mode)
            torch.cuda.synchronize()
            res[mode] = o.cpu().numpy()
        with np.errstate(all="ignore"):
            ieee = xb.view(np.float32) / yb.view(np.float32)
        out(f"- {label} ({xb.size}): ieee_rounding=True vs tl.div_rn differ in {same_bits(res[0], res[1])}; "
            f"vs the IEEE quotient in {same_bits(res[0], ieee)}; the default tl.fdiv vs the IEEE quotient in "
            f"{same_bits(res[2], ieee)}")
    out("")
    # C. a following addition.
    out("## C. a following addition, enable_fp_fusion on (the default)")
    M = 1 << 20
    a = rng.uniform(1, 2, M).astype(np.float32)
    b = rng.uniform(1, 2, M).astype(np.float32)
    q = a / b
    c = (-q * (1 + rng.uniform(-2 ** -20, 2 ** -20, M))).astype(np.float32)   # nearly cancelling
    for sub in (False, True):
        cc = -c if sub else c
        x, y, d = (torch.from_numpy(v).cuda() for v in (a, b, cc))
        o = torch.empty_like(x)
        h = k_add[(M // N,)](x, y, d, o, N, sub)
        torch.cuda.synchronize()
        got = o.cpu().numpy()
        want = (cc - q) if sub else (q + cc)
        # what a fused multiply-add of the exact quotient would give differs from want on most inputs
        exact = (a.astype(np.float64) / b.astype(np.float64))
        fused = ((cc.astype(np.float64) - exact) if sub else (exact + cc.astype(np.float64))).astype(np.float32)
        name = "c - tl.fdiv(x, y, ieee_rounding=True)" if sub else "tl.fdiv(x, y, ieee_rounding=True) + c"
        out(f"- {name}: {M} results, {same_bits(got, want)} differ from the IEEE sum of the IEEE quotient "
            f"(an unrounded quotient would differ on {same_bits(fused, want)}); SASS {sass_ops(h)}")
    out("")
    # D. softmax.
    out("## D. tl.softmax(ieee_rounding=True), rows of 16")
    R, C = 64, 16
    bad, tot = 0, 0
    for t in range(64):
        xs = torch.from_numpy((rng.standard_normal((R, C)) * (4 if t % 2 else 30)).astype(np.float32)).cuda()
        o, num, den = torch.empty_like(xs), torch.empty_like(xs), torch.empty_like(xs)
        k_softmax[(1,)](xs, o, num, den, R, C, True)
        torch.cuda.synchronize()
        want = num.cpu().numpy() / den.cpu().numpy()
        bad += same_bits(o.cpu().numpy(), want)
        tot += R * C
    out(f"- {tot} results, {bad} differ from `num / den` with the IEEE division")
    with open(os.path.join(HERE, "out.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
