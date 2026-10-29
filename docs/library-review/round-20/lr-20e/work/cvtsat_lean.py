"""lr-20e: the six `CvtSat…` facts (docs/rules/num.md) against the library's own definition, evaluated by Lean.

`CvtSat P f g`: the device's `x.to(fp8)` (default rounding) is `FloatV.cvtSat .rn x` up to NaN payload.
evidence/float-layer part A compares a compiled copy of the definition with the device; this script evaluates
the definition itself (`lake env lean`, `#eval`) on the inputs where a conversion is easiest to get wrong, and
compares bit for bit (both NaN counts as equal) with what Triton 3.8.0 stores on the H200:
- for each finite fp8 value `v` of the target, `v` in the source format, the midpoint between `v` and the next
  fp8 value (a tie) when the source holds it, and the source's neighbours of that midpoint;
- `maxVal`, the tie above it (464, 61440), its neighbours, values far above, the source's largest finite value;
- half the smallest fp8 subnormal (a tie with zero), its neighbours, the smallest source subnormals;
- `±0`, `±inf`, NaNs (canonical and with payloads), every value with both signs.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I cvtsat_lean.py
"""
import os
import struct
import subprocess
import sys
import tempfile

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, "../../../../.."))
LAKE = os.path.expanduser("~/.elan/bin/lake")

# source: (torch dtype, int dtype, bits, exponent bits, fraction bits, Lean format)
SRC = {"float32": (torch.float32, torch.int32, 32, 8, 23, "Math.Float.Format.binary32"),
       "float16": (torch.float16, torch.int16, 16, 5, 10, "Math.Float.Format.binary16"),
       "bfloat16": (torch.bfloat16, torch.int16, 16, 8, 7, "Math.Float.Format.bfloat16")}
# target: (Triton dtype, exponent bits, fraction bits, ieee specials, Lean format)
DST = {"float8e4nv": ("tl.float8e4nv", 4, 3, False, "Math.Float.Format.e4m3"),
       "float8e5": ("tl.float8e5", 5, 2, True, "Math.Float.Format.e5m2")}


def fp8_values(e, m, ieee):
    out = []
    for p in range(128):
        ex, fr = p >> m, p & ((1 << m) - 1)
        bias = (1 << (e - 1)) - 1
        if ieee and ex == (1 << e) - 1:
            continue
        if not ieee and ex == (1 << e) - 1 and fr == (1 << m) - 1:
            continue
        out.append(fr * 2.0 ** (1 - bias - m) if ex == 0 else ((1 << m) + fr) * 2.0 ** (ex - bias - m))
    return sorted(out)


def to_bits(v, sname):
    """The source pattern nearest to the double v (round to nearest even), as an unsigned int."""
    if sname == "float32":
        try:
            return struct.unpack("<I", struct.pack("<f", v))[0]
        except OverflowError:
            return 0x7f800000 if v > 0 else 0xff800000
    if sname == "float16":
        try:
            return struct.unpack("<H", struct.pack("<e", v))[0]
        except OverflowError:
            return 0x7c00 if v > 0 else 0xfc00
    t = torch.tensor([v], dtype=torch.float64).to(torch.bfloat16)   # round to nearest even
    return int(t.view(torch.int16).item()) & 0xFFFF


def inputs(sname, dname):
    _, _, nb, se, sm, _ = SRC[sname]
    _, e, m, ieee, _ = DST[dname]
    vals = fp8_values(e, m, ieee)
    pats = set()
    above = vals[-1] + (vals[-1] - vals[-2])      # the next value at maxVal's spacing (480, 65536)
    for a, b in zip(vals, vals[1:] + [above]):
        for v in (a, (a + b) / 2):
            p = to_bits(v, sname)
            for d in (-2, -1, 0, 1, 2):
                q = p + d
                if 0 <= q < (1 << (nb - 1)):
                    pats.add(q)
    mx = vals[-1]
    for v in (mx * 1.01, mx * 1.5, mx * 4, 1e5, 65504.0, 3.0e38, vals[1] / 2, vals[1] / 4, vals[1] * 0.75):
        p = to_bits(v, sname)
        for d in (-1, 0, 1):
            q = p + d
            if 0 <= q < (1 << (nb - 1)):
                pats.add(q)
    pats |= {0, 1, 2, 3, (1 << (nb - 1)) - (1 << sm) - 1}                    # zero, subnormals, max finite
    inf = ((1 << se) - 1) << sm
    pats |= {inf, inf | 1, inf | (1 << (sm - 1)), inf | ((1 << sm) - 1), inf | (1 << (sm - 1)) | 1}
    out = sorted(pats)
    return out + [p | (1 << (nb - 1)) for p in out]


@triton.jit
def cvt(x_ptr, o4_ptr, o5_ptr, n, SRC: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    raw = tl.load(x_ptr + offs, mask=m)
    if SRC == 0:
        x = raw.to(tl.float32, bitcast=True)
    elif SRC == 1:
        x = raw.to(tl.float16, bitcast=True)
    else:
        x = raw.to(tl.bfloat16, bitcast=True)
    tl.store(o4_ptr + offs, x.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m)
    tl.store(o5_ptr + offs, x.to(tl.float8e5).to(tl.uint8, bitcast=True), mask=m)


def device(sname, pats):
    _, idt, nb, *_ = SRC[sname]
    signed = [p - (1 << nb) if p >= 1 << (nb - 1) else p for p in pats]
    x = torch.tensor(signed, dtype=torch.int64, device="cuda").to(idt)
    n = len(pats)
    o4 = torch.empty(n, dtype=torch.uint8, device="cuda")
    o5 = torch.empty(n, dtype=torch.uint8, device="cuda")
    cvt[(triton.cdiv(n, 256),)](x, o4, o5, n, SRC=list(SRC).index(sname), B=256)
    return {"float8e4nv": o4.tolist(), "float8e5": o5.tolist()}


def lean(jobs):
    src = ["import TVA.Lang.Scalar", "open TVA", "set_option maxRecDepth 200000", ""]
    for k, (sname, dname, pats) in enumerate(jobs):
        sf, df = SRC[sname][5], DST[dname][4]
        src.append(f"def ps{k} : List Nat := [{', '.join(map(str, pats))}]")
        src.append(f"#eval IO.println (String.intercalate \" \" (ps{k}.map fun n => "
                   f"let r : FloatV {df} := FloatV.cvtSat .rn (FloatV.ofPat (f := {sf}) n); "
                   f"toString r.pat ++ (if r.isNaN then \"n\" else \"\")))")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "CvtSatCheck.lean")
        with open(path, "w") as fh:
            fh.write("\n".join(src) + "\n")
        r = subprocess.run([LAKE, "env", "lean", path], cwd=os.path.join(TVA, "lean"), capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise SystemExit("lean failed:\n" + (r.stdout[-1500:] + r.stderr[-1500:]).replace(tmp, "<tmp>"))
    return [l.split() for l in r.stdout.strip().splitlines()]


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    jobs = [(s, d, inputs(s, d)) for s in SRC for d in DST]
    lib = lean(jobs)
    total = 0
    for (sname, dname, pats), row in zip(jobs, lib):
        dev = device(sname, pats)[dname]
        e, m, ieee = DST[dname][1], DST[dname][2], DST[dname][3]

        def isnan8(b):
            ex, fr = (b >> m) & ((1 << e) - 1), b & ((1 << m) - 1)
            return ex == (1 << e) - 1 and (fr != 0 if ieee else fr == (1 << m) - 1)
        bad = []
        for p, d, w in zip(pats, dev, row):
            wn = w.endswith("n")
            wv = int(w.rstrip("n"))
            if not (d == wv or (wn and isnan8(d))):
                bad.append(f"{p:x}: device {d:02x}, cvtSat {wv:02x}")
        total += len(bad)
        print(f"{sname:8s} -> {dname:10s}: {len(pats)} inputs, {len(bad)} differ from FloatV.cvtSat .rn "
              f"(up to NaN payload)" + "".join(f"\n    {b}" for b in bad[:10]))
    print(f"differences in all: {total}")


if __name__ == "__main__":
    main()
