"""lr-20e: fp8 constants beyond evidence/fp8-constants' cases, against `FloatV.litF8` evaluated by Lean.

Cases: doubles whose `float32` is a zero though the double is not (`1e-46`, `-1e-50`), `float32` subnormals,
Python `int`s (small, exact in fp8 or not, beyond `float32`, beyond `float64`) and `bool`s, as literals in
`tl.full([1], c, d)` and in `tl.where(i < 0, x, c)` beside an fp8 tensor `x`. Each in a kernel of its own (a
file of its own). The library side: `Operand.constIn? c d` (`TVA.Lang.Promote`) for the `Operand` the literal
is (`.float`, `.int`, `.bool`), evaluated by `lake env lean` on a scratch file.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I fp8_const_edges.py
"""
import importlib.util
import os
import struct
import subprocess
import sys
import tempfile

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, "../../../../.."))
LAKE = os.path.expanduser("~/.elan/bin/lake")

# (source text, Lean Operand)
def fop(v):
    return f".float (.ofPat {struct.unpack('<Q', struct.pack('<d', v))[0]})"


CASES = [("1e-46", fop(1e-46)), ("-1e-46", fop(-1e-46)), ("1e-50", fop(1e-50)), ("-1e-50", fop(-1e-50)),
         ("2.0**-149", fop(2.0 ** -149)), ("-(2.0**-149)", fop(-(2.0 ** -149))),
         ("3", ".int 3"), ("-3", ".int (-3)"), ("448", ".int 448"), ("449", ".int 449"),
         ("16777217", ".int 16777217"), ("2**200", ".int (2^200)"), ("-(2**200)", ".int (-(2^200))"),
         ("10**400", ".int (10^400)"), ("True", ".bool true"), ("False", ".bool false"),
         ("0.25", fop(0.25)), ("-0.0", fop(-0.0))]
FMTS = [("tl.float8e5", ".f8e5", torch.float8_e5m2), ("tl.float8e4nv", ".f8e4nv", torch.float8_e4m3fn)]


def lean_side():
    src = ["import TVA.Lang.Promote", "open TVA", ""]
    for _, d, _ in FMTS:
        for k, (_, op) in enumerate(CASES):
            src.append(f"#eval IO.println (match Operand.constIn? ({op}) {d} with "
                       f"| some y => toString y.pat | none => \"-1\")")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "Fp8Edges.lean")
        with open(path, "w") as fh:
            fh.write("\n".join(src) + "\n")
        r = subprocess.run([LAKE, "env", "lean", path], cwd=os.path.join(TVA, "lean"), capture_output=True,
                           text=True)
    if r.returncode != 0:
        raise SystemExit("lean failed:\n" + r.stdout[-1500:] + r.stderr[-1500:])
    vals = [int(l) for l in r.stdout.split()]
    n = len(CASES)
    return {d: vals[i * n:(i + 1) * n] for i, (_, d, _) in enumerate(FMTS)}


def load(tmp, name, body):
    src = "\n".join(["import triton", "import triton.language as tl", "", "@triton.jit",
                     f"def {name}(x_ptr, o_ptr):", "    i = tl.arange(0, 1)",
                     "    x = tl.load(x_ptr + i)"] + ["    " + b for b in body]) + "\n"
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    lib = lean_side()
    tmp = tempfile.mkdtemp(prefix="lr20e-fp8e-")
    bad = 0
    for dt, d, tdt in FMTS:
        for form in ("full", "where"):
            for k, (c, _) in enumerate(CASES):
                e = f"tl.full([1], {c}, {dt})" if form == "full" else f"tl.where(i < 0, x, {c})"
                kern = load(tmp, f"k_{d[1:]}_{form}_{k}", [f"tl.store(o_ptr + i, {e})"])
                x = torch.zeros(1, dtype=torch.uint8, device="cuda").view(tdt)
                o = torch.full((1,), 0x55, dtype=torch.uint8, device="cuda").view(tdt)
                try:
                    kern[(1,)](x, o)
                    torch.cuda.synchronize()
                    dev = o.view(torch.uint8).item()
                except Exception as ex:
                    dev = None
                    err = type(ex).__name__
                want = lib[d][k]
                want = None if want < 0 else want
                ok = dev == want
                bad += 0 if ok else 1
                ds = "rejected" if dev is None else f"{dev:02x}"
                ls = "none" if want is None else f"{want:02x}"
                harmless = (" (device rejects: no outcome, a model value there is never used)"
                            if dev is None and want is not None else "")
                print(f"{dt:14s} {form:5s} {c:14s} device {ds:8s} constIn? {ls:5s} "
                      f"{'same' if ok else 'DIFFER' + harmless}")
    print(f"differences: {bad}")


if __name__ == "__main__":
    main()
