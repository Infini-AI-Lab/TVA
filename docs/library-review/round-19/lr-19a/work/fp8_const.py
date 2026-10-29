"""lr-19a: a Python float constant built in an fp8 dtype (`scalar_constant`: `get_fp32`, then
`create_fp_trunc`, an `arith.truncf`; primitives/triton-3.8.0-semantic.py:601-613). SEMANTICS
"Constants" says it is `FloatV.litF32`, then `tl.cvt8`; rules/num.md ("Conversions into fp8") that
the compiler folds the conversion only when it is exact, "so the facts cover it too". Check, one
kernel per constant and dtype: does `tl.full([1], c, d)` (and `tl.where(<false>, x8, c)`) compile,
and is its value the run-time conversion `x.to(d)` of the `float32` constant loaded from memory (the
operation the `CvtSat…` facts are about), bit for bit up to NaN payload, and torch's saturating
conversion of that `float32` (`FloatV.cvtSat .rn`)?

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I fp8_const.py
"""
import importlib.util
import os
import struct
import tempfile

import torch
import triton

CONSTS = ["0.3", "1.0", "-1.0", "448.0", "449.0", "464.0", "465.0", "480.0", "500.0", "1e6", "-1e6",
          "57344.0", "61439.0", "61440.0", "65536.0", "1e-9", "2.0**-9", "2.0**-10", "2.0**-17", "2.0**-16",
          "-2.0**-17", "1.0 + 2.0**-4", "1.0 + 3 * 2.0**-4", "float('inf')", "-float('inf')", "float('nan')",
          "-float('nan')", "3.0e38", "1e-40"]
DT = {"e4m3": ("tl.float8e4nv", torch.float8_e4m3fn, 448.0), "e5m2": ("tl.float8e5", torch.float8_e5m2, 57344.0)}

SRC = '''import triton
import triton.language as tl


@triton.jit
def k(src, full_o, where_o, cvt_o):
    i = tl.arange(0, 1)
    tl.store(cvt_o + i, tl.load(src + i).to({dt}))
    x8 = tl.load(cvt_o + 1 + i)
    tl.store(full_o + i, tl.full([1], {c}, {dt}))
    tl.store(where_o + i, tl.where(i < 0, x8, {c}))
'''


def isnan(name, b):
    return (b & 0x7f) == 0x7f if name == "e4m3" else (b & 0x7c) == 0x7c and (b & 3) != 0


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    tmp = tempfile.mkdtemp(prefix="lr19a-fp8-")
    for name, (dt, ttd, mx) in DT.items():
        print(f"{name}:")
        counts = {"compiled, as the run-time conversion": 0, "rejected on the host": 0, "DIFFERS": 0}
        for n, c in enumerate(CONSTS):
            f32 = torch.tensor([eval(c)], dtype=torch.float32)
            ref = torch.where(f32.isnan(), f32, f32.clamp(-mx, mx)).to(ttd).view(torch.uint8).item()
            path = os.path.join(tmp, f"k_{name}_{n}.py")
            with open(path, "w") as fh:
                fh.write(SRC.format(dt=dt, c=c))
            spec = importlib.util.spec_from_file_location(f"k_{name}_{n}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            outs = [torch.zeros(2, dtype=torch.uint8, device="cuda") for _ in range(3)]
            fbits = struct.unpack('<I', struct.pack('<f', f32.item()))[0]
            try:
                mod.k[(1,)](f32.cuda(), *[o.view(ttd) for o in outs])
                torch.cuda.synchronize()
            except Exception as e:  # compile error: rejected on the host
                msg = str(e).strip().splitlines()[-1][:80] if str(e).strip() else type(e).__name__
                counts["rejected on the host"] += 1
                print(f"  {c:20s} f32 {fbits:08x}: rejected on the host ({type(e).__name__}: {msg}); "
                      f"cvtSat .rn {ref:02x}")
                continue
            full, where, cvt = outs[0][0].item(), outs[1][0].item(), outs[2][0].item()
            same = lambda a, b: a == b or (isnan(name, a) and isnan(name, b))
            ok = same(full, cvt) and same(where, cvt) and same(cvt, ref)
            counts["compiled, as the run-time conversion" if ok else "DIFFERS"] += 1
            print(f"  {c:20s} f32 {fbits:08x}: tl.full {full:02x}, tl.where {where:02x}, run-time x.to {cvt:02x}, "
                  f"cvtSat .rn {ref:02x}" + ("" if ok else "  DIFFERS"))
        print(f"  {name}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
