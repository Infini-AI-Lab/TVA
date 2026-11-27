"""TVA library evidence, L40 (sm_89): float32 into fp8 is two roundings.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/l40/num-cvt8/double.py
Output (stdout): kept as evidence/l40/num-cvt8/double.out.

On sm_89 Triton 3.8.0 lowers `x.to(tl.float8e4nv)` / `x.to(tl.float8e5)` of a float32 `x` to
`cvt.rz.f16.f32` (float16, rounding toward zero) and then `cvt.rn.satfinite.e4m3x2.f16x2` /
`e5m2x2.f16x2` (`evidence/num-cvt8/run.py` on the L40 lists the instructions). For every float32
bit pattern (all 2^32) the device's result is compared, bit for bit, with that composition computed
exactly here:

* `h = RZ16(x)`: float16 rounding toward zero (a finite value beyond 65504 gives +-65504; infinities
  and NaN stay);
* the fp8 result is `cvtSat(h)`: round to nearest even when `|h| <= maxVal`, `+-maxVal` above it
  and for infinities (`.satfinite`), the NaN `0x7f` for a NaN.

It also counts the inputs where the composition differs from one rounding (`CvtSat` of `x`), to
show what the H200's fact would get wrong here.
"""
import os
import sys
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "num-cvt8"))
from run import cvt, rne_bits, FMT  # noqa: E402  (the H200 script's kernel and exact rounding)

SAT = {"float8e4nv": 0x7E, "float8e5": 0x7B}


def rz16(xf):
    """float16 rounding toward zero of the float64 values xf (exact: float64 holds every step)."""
    a = xf.abs()
    nz = (a > 0) & torch.isfinite(a)
    lg = torch.floor(torch.log2(a.clamp(min=1e-300))).to(torch.int64)
    lg = torch.where(nz & (torch.ldexp(torch.ones_like(a), lg) > a), lg - 1, lg)
    lg = torch.where(nz & (torch.ldexp(torch.ones_like(a), lg + 1) <= a), lg + 1, lg)
    q = torch.maximum(lg, torch.full_like(lg, -14)) - 10          # float16's quantum at |x|
    t = torch.ldexp(torch.floor(torch.ldexp(a, -q)), q)           # truncated: exact in float64
    t = torch.where(nz, torch.clamp(t, max=65504.0), a)
    return torch.copysign(t, xf)


def sat_bits(h, fmt):
    """`cvtSat` of float16 values h (as float64): fp8 bit patterns."""
    maxv = FMT[fmt][2]
    fin = torch.isfinite(h)
    inr = fin & (h.abs() <= maxv)
    b = rne_bits(torch.where(inr, h, torch.zeros_like(h)), fmt)
    neg = torch.signbit(h)
    over = ~torch.isnan(h) & ~inr
    b = torch.where(over, torch.where(neg, SAT[fmt] | 0x80, SAT[fmt]), b)
    b = torch.where(torch.isnan(h), torch.full_like(b, 0x7F), b)
    return b


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    total, CH = 1 << 32, 1 << 26
    st = {f: dict(diff=0, ex=None, one=0, onex=None, nan=0) for f in FMT}
    for lo in range(0, total, CH):
        b64 = torch.arange(lo, lo + CH, dtype=torch.int64, device="cuda")
        bits = torch.where(b64 >= (1 << 31), b64 - total, b64).to(torch.int32)
        e4 = torch.empty(CH, dtype=torch.int8, device="cuda")
        e5 = torch.empty(CH, dtype=torch.int8, device="cuda")
        cvt[(CH // 1024,)](bits, e4, e5, CH, SRC=0, B=1024)
        xf = bits.view(torch.float32).to(torch.float64)
        h = rz16(xf)
        for fmt, out in (("float8e4nv", e4), ("float8e5", e5)):
            got = out.to(torch.int64) & 0xFF
            ref = sat_bits(h, fmt)
            isnan = torch.isnan(xf)
            # a NaN input: any NaN result (up to payload), as CvtSat states
            nanok = isnan & ((got & 0x7F) == 0x7F) if fmt == "float8e4nv" else isnan & ((got & 0x7F) > 0x7C)
            bad = (got != ref) & ~nanok
            s = st[fmt]
            s["diff"] += int(bad.sum())
            s["nan"] += int(isnan.sum())
            if s["ex"] is None and bool(bad.any()):
                k = int(bad.nonzero()[0]); s["ex"] = (float(xf[k]), int(got[k]), int(ref[k]))
            one = sat_bits(xf, fmt)
            d1 = (one != ref) & ~isnan
            s["one"] += int(d1.sum())
            if s["onex"] is None and bool(d1.any()):
                k = int(d1.nonzero()[0]); s["onex"] = (float(xf[k]), int(one[k]), int(ref[k]))
    print("\n| to | inputs | NaN inputs | differ from cvtSat(RZ16(x)) | first difference (x, device, ref) |"
          " differ: one rounding vs two | first (x, one rounding, two) |")
    print("|---|---|---|---|---|---|---|")
    for fmt in FMT:
        s = st[fmt]
        print(f"| {fmt} | {total} | {s['nan']} | {s['diff']} | {s['ex']} | {s['one']} | {s['onex']} |")


if __name__ == "__main__":
    main()
