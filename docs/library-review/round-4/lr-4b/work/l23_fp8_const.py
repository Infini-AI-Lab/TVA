"""lr-4b, library review round 4, L23: the fp8 constant of a Python number.

[literal] says a Python number in an fp8 operation (`tl.full(shape, c, fp8)`, `tl.where` with an fp8
tensor) is `FloatV.litF32 c`, then `tl.cvt8` (the platform's conversion into fp8, the one a runtime
`x.to(fp8)` uses). Triton 3.8.0 builds it with `get_fp32(c)` then `create_fp_trunc` (arith.truncf,
semantic.py:607-609), while a runtime `.to(fp8)` is `tt.fp_to_fp` (semantic.py:835-840). MLIR
folds the truncf only when it is exact (llvm-ArithOps.cpp:1591-1611, 1811-1824). This compares,
for float8e5 and float8e4nv, per value `c`:
  const:  tl.full([1], c, fp8)
  where:  tl.where(m, x_fp8, c) with m false (an operand of tl.where)
  rt:     the float32 constant loaded from memory at run time, then `.to(fp8)` (tl.cvt8 on the device)
Values: exact fp8 values, ties, values between fp8 neighbours, values beyond the largest finite
value, values in the subnormal range and below it, and both signs.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import struct

import torch
import triton
import triton.language as tl

VALUES = [0.5, 0.3, 0.1, 1.0 / 3, 1.125, 1.0625, 1.1875, 2.5, 3.0, 3.5, 100.0, 240.0, 248.0, 256.0, 448.0,
          464.0, 480.0, 500.0, 1000.0, 57344.0, 60000.0, 61440.0, 65536.0, 1e6, 1e30, 1e-2, 1e-3,
          2.0 ** -9, 1.5 * 2.0 ** -9, 2.0 ** -10, 2.0 ** -16, 1.5 * 2.0 ** -16, 2.0 ** -17, 3 * 2.0 ** -18, 1e-7, 1e-9,
          float("inf")]


def make(dt, cs):
    lines = ["import triton", "import triton.language as tl", "", "@triton.jit",
             "def k(src_ptr, x_ptr, c_ptr, w_ptr, r_ptr):",
             "    x = tl.load(x_ptr + tl.arange(0, 1))",
             "    m = tl.arange(0, 1) < 0"]
    for i, c in enumerate(cs):
        lit = repr(c) if abs(c) != float("inf") else ("float('inf')" if c > 0 else "-float('inf')")
        lines.append(f"    tl.store(c_ptr + {i} + tl.arange(0, 1), tl.full([1], {lit}, {dt}))")
        lines.append(f"    tl.store(w_ptr + {i} + tl.arange(0, 1), tl.where(m, x, {lit}))")
        lines.append(f"    tl.store(r_ptr + {i} + tl.arange(0, 1), tl.load(src_ptr + {i} + tl.arange(0, 1)).to({dt}))")
    return "\n".join(lines) + "\n"


def exact_in(c, tdt):
    t = torch.tensor([c], dtype=torch.float64)
    return t.to(tdt).to(torch.float64).item() == c if abs(c) != float("inf") else tdt == torch.float8_e5m2


def compiles(dt, c, tmp, mod_name):
    import importlib.util
    import os
    path = os.path.join(tmp, mod_name + ".py")
    lit = repr(c) if abs(c) != float("inf") else ("float('inf')" if c > 0 else "-float('inf')")
    open(path, "w").write("import triton\nimport triton.language as tl\n\n@triton.jit\ndef k(o_ptr):\n"
                          f"    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], {lit}, {dt}))\n")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    o = torch.zeros(1, dtype=torch.float8_e5m2 if dt.endswith("e5") else torch.float8_e4m3fn, device="cuda")
    try:
        mod.k[(1,)](o)
        torch.cuda.synchronize()
        return "compiles, stores %#04x" % o.view(torch.uint8).item()
    except Exception as e:
        return "rejected on the host (" + type(e).__name__ + ")"


def main():
    import importlib.util
    import os
    import tempfile
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    with tempfile.TemporaryDirectory() as tmp:
      for name, dt, tdt in (("float8e5", "tl.float8e5", torch.float8_e5m2), ("float8e4nv", "tl.float8e4nv", torch.float8_e4m3fn)):
        print(f"== {name}: values not exactly representable (and inf for float8e4nv), one kernel each, tl.full([1], c, {dt})")
        for j, v in enumerate(VALUES):
            for c in (v, -v):
                if not exact_in(c, tdt):
                    print(f"  c = {c!r:>24}: " + compiles(dt, c, tmp, f"one_{name}_{j}_{int(c < 0)}"))
      for name, dt, tdt in (("float8e5", "tl.float8e5", torch.float8_e5m2), ("float8e4nv", "tl.float8e4nv", torch.float8_e4m3fn)):
            cs = []
            for v in VALUES:
                cs += [c for c in (v, -v) if exact_in(c, tdt)]
            path = os.path.join(tmp, f"k_{name}.py")
            open(path, "w").write(make(dt, cs))
            spec = importlib.util.spec_from_file_location(f"k_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            src = torch.tensor(cs, dtype=torch.float32, device="cuda")
            x = torch.zeros(1, dtype=tdt, device="cuda")
            outs = [torch.zeros(len(cs), dtype=tdt, device="cuda") for _ in range(3)]
            h = mod.k[(1,)](src, x, *outs)
            torch.cuda.synchronize()
            const, where, rt = [o.view(torch.uint8).tolist() for o in outs]
            ttir = h.asm["ttir"]
            print(f"== {name}, exactly representable values: TTIR arith.truncf {ttir.count('arith.truncf')}, tt.fp_to_fp {ttir.count('tt.fp_to_fp')}; "
                  f"PTX cvt lines {h.asm['ptx'].count('cvt.rn')}")
            nd = 0
            for i, c in enumerate(cs):
                f32 = struct.unpack('<f', struct.pack('<f', c))[0] if abs(c) < 3.5e38 else c
                tag = "" if const[i] == rt[i] and where[i] == rt[i] else "   <-- differs"
                nd += tag != ""
                print(f"  c = {c!r:>24} (f32 {f32!r}): const {const[i]:#04x}, where {where[i]:#04x}, run-time .to {rt[i]:#04x}{tag}")
            print(f"  values where the constant differs from the run-time conversion: {nd} of {len(cs)}")


if __name__ == "__main__":
    main()
