"""lr-14b, round 14 (L64): constants converted into fp8.

docs/rules/num.md ("Conversions into fp8") says a constant converted into fp8 is folded by the
compiler only when the conversion is exact, so the `CvtSat…` facts cover constants too. This test
builds fp8 values from Python constants in three ways and compares each with the device's
run-time conversion of the same `float32` value loaded from memory, which evidence/float-layer
(part A) ties to `FloatV.cvtSat .rn`:
  K1  tl.full([B], c, tl.float8e4nv / tl.float8e5)
  K2  tl.full([B], c, tl.float32).to(fp8)
  K3  (tl.zeros([B], tl.float32) + c).to(fp8)          (an arithmetic result the compiler may fold)
  K4  tl.store of a float32 constant tensor into an fp8 pointer (the implicit conversion)
  RT  tl.load(p).to(fp8) of the float32 pattern of c (run time)
and with a saturating round-to-nearest-even reference written here. A kernel Triton rejects on
the host (a compile error: no outcome, outside every claim) is listed as such. It also prints whether the
PTX of K1/K2 converts at run time (a `cvt…e4m3`/`e5m2` instruction) or stores a folded byte.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I cvt8_const.py   (output: cvt8_const.out)
"""
import math
import struct
from fractions import Fraction

import torch
import triton
import triton.language as tl

B = 128
CONSTS = [0.0, -0.0, 1.0, 0.1, -0.3, 1.0 / 3.0, 448.0, 449.0, 464.0, 470.0, 480.0, 1000.0, -1000.0,
          57344.0, 60000.0, 61440.0, 65504.0, 1e6, -1e30, 3.0e38, 2.0 ** -9, 2.0 ** -10, 2.0 ** -17,
          1e-8, float("inf"), float("-inf"), float("nan")]
FMTS = {"e4m3": (tl.float8e4nv, 4, 3, 448.0, True), "e5m2": (tl.float8e5, 5, 2, 57344.0, False)}


@triton.jit
def k_full8(o, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    r = tl.full([B], C, DT)
    tl.store(o + tl.arange(0, B), r.to(tl.uint8, bitcast=True))


@triton.jit
def k_full32(o, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    r = tl.full([B], C, tl.float32).to(DT)
    tl.store(o + tl.arange(0, B), r.to(tl.uint8, bitcast=True))


@triton.jit
def k_arith(o, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    r = (tl.zeros([B], tl.float32) + C).to(DT)
    tl.store(o + tl.arange(0, B), r.to(tl.uint8, bitcast=True))


@triton.jit
def k_store(o8, C: tl.constexpr, B: tl.constexpr):
    tl.store(o8 + tl.arange(0, B), tl.full([B], C, tl.float32))


@triton.jit
def k_rt(o, p, DT: tl.constexpr, B: tl.constexpr):
    x = tl.load(p + tl.arange(0, B))
    tl.store(o + tl.arange(0, B), x.to(DT).to(tl.uint8, bitcast=True))


def ref(c, e, m, maxv, fn):
    """Saturating round to nearest even of the float32 value of c into the fp8 format."""
    if math.isnan(c):
        return "nan"
    f32 = struct.unpack("<f", struct.pack("<f", c))[0]  # the float32 constant
    s = 0x80 if math.copysign(1.0, f32) < 0 else 0
    if math.isinf(f32) or abs(f32) > maxv:
        q = Fraction(maxv)
    else:
        q = abs(Fraction(f32))
    bias = 2 ** (e - 1) - 1
    emin = 1 - bias
    if q == 0:
        return s
    ex = max(math.floor(math.log2(q)), emin)
    while Fraction(2) ** ex > q and ex > emin:
        ex -= 1
    while Fraction(2) ** (ex + 1) <= q:
        ex += 1
    quantum = Fraction(2) ** (ex - m)
    t = q / quantum
    n = t.numerator // t.denominator
    r = t - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    v = n * quantum
    if v > maxv:
        v = Fraction(maxv)
    # encode v
    if v == 0:
        return s
    ev = math.floor(math.log2(v)) if v >= Fraction(2) ** emin else emin - 1
    if v < Fraction(2) ** emin:
        return s | int(v / Fraction(2) ** (emin - m))
    frac = int((v / Fraction(2) ** ev - 1) * 2 ** m)
    return s | ((ev + bias) << m) | frac


def isnan8(b, fn):
    if fn:
        return (b & 0x7F) == 0x7F
    return ((b >> 2) & 0x1F) == 0x1F and (b & 3) != 0


def main():
    lines = [f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}", ""]
    lines.append("| fmt | c | library (litF32, +0 for a zero, then cvtSat) | K1 full8 | K2 full32.to | K3 (0+c).to | K4 store | RT run time | all = library |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    bad = 0
    ptx = {}
    for name, (dt, e, m, maxv, fn) in FMTS.items():
        for c in CONSTS:
            out = {}
            for kn, kern in (("full8", k_full8), ("full32", k_full32), ("arith", k_arith)):
                o = torch.zeros(B, dtype=torch.uint8, device="cuda")
                try:
                    h = kern[(1,)](o, C=c, DT=dt, B=B)
                except Exception as ex:  # rejected on the host: no outcome
                    out[kn] = "host error: " + type(ex).__name__
                    continue
                out[kn] = int(o[0].item())
                assert (o == o[0]).all()
                if c in (1000.0, 0.1, 1.0):
                    ins = sorted({l.split()[0] for l in h.asm["ptx"].splitlines() if l.strip().startswith("cvt") and ("e4m3" in l or "e5m2" in l)})
                    ptx[(name, kn, c)] = ", ".join(ins) if ins else "no fp8 cvt (folded)"
            o8 = torch.zeros(B, dtype=torch.float8_e4m3fn if fn else torch.float8_e5m2, device="cuda")
            k_store[(1,)](o8, C=c, B=B)
            out["store"] = int(o8.view(torch.uint8)[0].item())
            p = torch.full((B,), c, dtype=torch.float32, device="cuda")
            o = torch.zeros(B, dtype=torch.uint8, device="cuda")
            k_rt[(1,)](o, p, DT=dt, B=B)
            out["rt"] = int(o[0].item())
            # the library: a constant zero is +0 (`-0.0` included), then `tl.cvt8` (`CvtSat`); the
            # run-time load of `-0.0` keeps its sign
            r = ref(0.0 if c == 0 else c, e, m, maxv, fn)
            r_rt = ref(c, e, m, maxv, fn)
            vals = [out["full8"], out["full32"], out["arith"], out["store"], out["rt"]]
            refs = [r, r, r, r, r_rt]
            pairs = [(v, q) for v, q in zip(vals, refs) if isinstance(v, int)]
            ok = all(isnan8(v, fn) if q == "nan" else v == q for v, q in pairs)
            rs = "NaN" if r == "nan" else f"{r:02x}" + ("" if r_rt == r else f" (run time {r_rt:02x})")
            bad += 0 if ok else 1
            cell = lambda v: f"{v:02x}" if isinstance(v, int) else v
            lines.append(f"| {name} | {c!r} | {rs} | " + " | ".join(cell(v) for v in vals) +
                         f" | {'yes' if ok else 'NO'}{'' if len(pairs) == len(vals) else ' (of those that compile)'} |")
    lines.append("")
    lines.append("PTX fp8 conversions (K1 full8, K2 full32, K3 arith):")
    for k, v in sorted(ptx.items(), key=lambda kv: str(kv[0])):
        lines.append(f"- {k[0]} {k[1]} c={k[2]!r}: {v}")
    lines.append(f"\nrows that differ: {bad}")
    text = "\n".join(lines)
    print(text)
    with open("cvt8_const.out", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
