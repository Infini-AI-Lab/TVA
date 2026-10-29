"""TVA library evidence (library issue L23): the constant Triton 3.8.0 builds for a Python number in a
float operation, per dtype, against the library's definitions `FloatV.litF64`, `litF32`, `litF16`
and `litBF16` (`lean/TVA/Lang/Scalar.lean`).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/literals/run.py
Output: evidence/literals/out.txt (README.md says what it shows).

The literals: an adversarial list (values around the six-decimal cut of `bfloat16` constants,
ties of that cut, values next to `bfloat16` rounding midpoints, zeros, values outside the
`float32` range, which overflow to infinities there, integers) and 4000 log-uniform random values
in `[1e-9, 1e6]`, each with both signs. Literals are written with `repr`, so NaN and the infinities,
which have no literal, are in `nan.py` (README.md, "NaN constants").

For each literal `c` and dtype `d` in `bfloat16`, `float16`, `float32`, `float64`, the device
stores, in one kernel per dtype and chunk of literals:
- `full`: `tl.full([1], c, d)`;
- `mul`: `x * c`, with `x` a `d` tensor holding `1` (exactly `c` for every `c`; `NaN` for a NaN);
- `other`: `tl.load(p, mask=False, other=c)` from a `d` pointer, where the number becomes a
  tensor of its own type first (`float32` in that type's normal range, `float64` outside it) and
  is then converted to `d` (the library's `[literal]` rule says so; here it is compared with the
  number's own-type constant rounded to `d`).
The script writes a Lean file that evaluates the library's functions on the same doubles, runs
it with `lake env lean`, and compares the stored bits with Lean's: `full` and `mul` bit for bit (a
constant is an exact pattern), `other` up to NaN payload (a conversion, `tl.convert`). It also
checks the six-decimal text against the C library's own `snprintf("%f")`.
"""
import ctypes
import importlib.util
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

import torch
import triton
import triton.language as tl

TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAKE = os.path.expanduser("~/.elan/bin/lake")
CHUNK = 256


def f64bits(c):
    return struct.unpack("<Q", struct.pack("<d", c))[0]


def literals():
    out = []
    base = [1e-7, 4.9e-7, 5e-7, 5.0000001e-7, 1e-6, 1.4e-6, 1.5e-6, 2.5e-6, 9.99999e-6, 1e-5, 1.23456e-5,
            1e-4, 1.2345e-4, 1e-3, 0.1, 0.3, 1 / 3, 0.5, 1.0, 1.5, 2.0, 3.14159265, 2.718281828, 10.0, 100.5,
            65504.0, 65519.0, 65520.0, 1e5, 123456.789, 1e10, 1e30, 3e38, 3.3895313892515355e38,
            3.4e38, 3.4028234663852886e38, 1e39, 1e300, 1e-38, 1.1754943508222875e-38, 1e-40, 1e-45, 1e-310,
            0.0, -0.0, 3, 7, 1000001, 2 ** 24 + 1, 2 ** 31 - 1, 2 ** 53 + 1]
    # ties of the six-decimal cut: odd multiples of 2^-7 (exact in float32)
    base += [j / 128 for j in (1, 3, 5, 7, 129, 131, 1001, 12345)]
    # values next to bfloat16 rounding midpoints, where the six-decimal cut may cross the midpoint
    for e in range(-20, 6):
        for mant in (1, 37, 64, 101):
            mid = (1 + (2 * mant + 1) / 256) * 2.0 ** e   # a bfloat16 midpoint (8 significant bits)
            for d in (-6e-7, -2e-7, -1e-9, 1e-9, 2e-7, 6e-7):
                base.append(mid + d)
    rnd = random.Random(20261027)
    base += [10 ** rnd.uniform(-9, 6) for _ in range(4000)]
    for c in base:
        out.append(c)
        if not (isinstance(c, float) and math.isnan(c)):
            out.append(-c)
    seen, uniq = set(), []
    for c in out:
        key = (type(c), f64bits(float(c)) if not isinstance(c, int) else c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


DTYPES = [("bf16", torch.bfloat16, "tl.bfloat16", torch.int16), ("f16", torch.float16, "tl.float16", torch.int16),
          ("f32", torch.float32, "tl.float32", torch.int32), ("f64", torch.float64, "tl.float64", torch.int64)]


def kernel_source(name, cs, dt):
    lines = ["import triton", "import triton.language as tl", "", "@triton.jit",
             f"def {name}(x_ptr, p_ptr, full_ptr, mul_ptr, other_ptr):", "    x = tl.load(x_ptr)"]
    for i, c in enumerate(cs):
        lit = repr(c)
        lines.append(f"    tl.store(full_ptr + {i} + tl.arange(0, 1), tl.full([1], {lit}, {dt}))")
        lines.append(f"    tl.store(mul_ptr + {i}, x * {lit})")
        lines.append(f"    tl.store(other_ptr + {i} + tl.arange(0, 1), tl.load(p_ptr + tl.arange(0, 1), "
                     f"mask=tl.arange(0, 1) < 0, other={lit}))")
    return "\n".join(lines) + "\n"


def device(cs, tmp):
    res = {}
    for tag, tdt, dt, idt in DTYPES:
        full, mul, other = [], [], []
        for k in range(0, len(cs), CHUNK):
            chunk = cs[k:k + CHUNK]
            name = f"lit_{tag}_{k // CHUNK}"
            path = os.path.join(tmp, name + ".py")
            open(path, "w").write(kernel_source(name, chunk, dt))
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            x = torch.ones(1, dtype=tdt, device="cuda")
            p = torch.zeros(1, dtype=tdt, device="cuda")
            bufs = [torch.empty(len(chunk), dtype=tdt, device="cuda") for _ in range(3)]
            getattr(mod, name)[(1,)](x, p, *bufs)
            torch.cuda.synchronize()
            for lst, b in zip((full, mul, other), bufs):
                lst += [v & ((1 << (8 * b.element_size())) - 1) for v in b.view(idt).tolist()]
        res[tag] = (full, mul, other)
    return res


def lean_values(cs, tmp):
    """By the library, for each literal and dtype `d`: the operation's constant (`FloatV.litBF16`,
    `litF16`, `litF32`, `litF64`), and the conversion to `d` (`FloatV.cvt`, one rounding to nearest
    even) of the number's own-type constant, `litF32` or `litF64`, for `other=`. Evaluated by Lean in
    chunks of 500 literals."""
    out = {tag: [[], [], []] for tag in ("bf16", "f16", "f32", "f64")}
    for k in range(0, len(cs), 500):
        src = ["import TVA.Lang.Scalar", "open TVA", "", "def cs : List Nat := ["]
        src.append(",\n".join(f"  {f64bits(float(c))}" for c in cs[k:k + 500]) + "]")
        src.append("def row {f : Math.Float.Format} (g : Nat → FloatV f) : IO Unit :=")
        src.append("  IO.println (String.intercalate \" \" (cs.map fun n => toString (g n).pat))")
        for tag, ty in (("bf16", "BF16"), ("f16", "F16"), ("f32", "F32"), ("f64", "F64")):
            src.append(f"#eval row (fun n => FloatV.lit{ty} (.ofPat n))")
            src.append(f"#eval row (fun n => (FloatV.cvt (FloatV.litF32 (.ofPat n)) : {ty}))")
            src.append(f"#eval row (fun n => (FloatV.cvt (FloatV.litF64 (.ofPat n)) : {ty}))")
        path = os.path.join(tmp, "LitCheck.lean")
        open(path, "w").write("\n".join(src) + "\n")
        r = subprocess.run([LAKE, "env", "lean", path], cwd=os.path.join(TVA, "lean"), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("lean failed: " + (r.stderr[-500:] + r.stdout[-500:]).replace(tmp, "<tmp>"))
        rows = [list(map(int, l.split())) for l in r.stdout.strip().splitlines()]
        for j, tag in enumerate(("bf16", "f16", "f32", "f64")):
            for q in range(3):
                out[tag][q] += rows[3 * j + q]
    return out


def is_nan(tag, v):
    e, m = {"bf16": (8, 7), "f16": (5, 10), "f32": (8, 23), "f64": (11, 52)}[tag]
    return (v >> m) & ((1 << e) - 1) == (1 << e) - 1 and v & ((1 << m) - 1) != 0


def own_f32(c):
    """The number's own type (`to_tensor_type`): for a float, float32 when 0, inf, NaN or in
    float32's normal range, float64 otherwise. An int is an int32 or int64 constant, whose conversion
    rounds once; it is compared here through the double, which holds the listed integers exactly
    (`2^53 + 1` rounds to the same value either way)."""
    if isinstance(c, int):
        return False
    x = float(c)
    return x == 0 or math.isinf(x) or math.isnan(x) or 2 ** -126 <= abs(x) <= (2 - 2 ** -23) * 2 ** 127


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    cs = literals()
    print(f"{len(cs)} literals")
    libc = ctypes.CDLL(None)
    buf = ctypes.create_string_buffer(512)
    text_bad = 0
    for c in cs:
        x = struct.unpack("<f", struct.pack("<f", float(c)))[0] if abs(float(c)) < 3.5e38 or not math.isfinite(float(c)) \
            else float(c)
        if math.isfinite(x):
            libc.snprintf(buf, 512, b"%f", ctypes.c_double(x))
            text_bad += buf.value.decode() != ("%f" % x)
    print(f"six-decimal text: C's snprintf(\"%f\") and Python's \"%f\" differ on {text_bad} of the float32 values")
    with tempfile.TemporaryDirectory() as tmp:
        dev = device(cs, tmp)
        lean = lean_values(cs, tmp)
    for tag, _, _, _ in DTYPES:
        full, mul, other = dev[tag]
        lib, via32, via64 = lean[tag]

        def same(a, b):
            return a == b or (is_nan(tag, a) and is_nan(tag, b))
        bad_full = [i for i in range(len(cs)) if full[i] != lib[i]]   # bit for bit
        bad_mul = [i for i in range(len(cs)) if mul[i] != lib[i]]
        own = [via32[i] if own_f32(c) else via64[i] for i, c in enumerate(cs)]
        bad_other = [i for i in range(len(cs)) if not same(other[i], own[i])]
        lit_vs_own = sum(1 for i in range(len(cs)) if not same(lib[i], own[i]))
        print(f"{tag:>4}: tl.full differs from FloatV.lit{tag.upper() if tag != 'bf16' else 'BF16'}: {len(bad_full)}; "
              f"x * c differs from it: {len(bad_mul)}; other= differs from own type then convert: {len(bad_other)}; "
              f"literals where the operation's constant and other='s differ: {lit_vs_own}")
        for i in (bad_full + bad_mul + bad_other)[:5]:
            print(f"      e.g. c = {cs[i]!r}: full {full[i]:#x}, mul {mul[i]:#x}, other {other[i]:#x}, "
                  f"library {lib[i]:#x}, own-type {own[i]:#x}")
    # a few named examples
    names = [1e-7, -1e-7, 1.23456e-5, 0.1, 3.14159265]
    for c in names:
        i = cs.index(c)
        print(f"c = {c!r}: " + ", ".join(f"{tag} {lean[tag][0][i]:#x}" for tag, _, _, _ in DTYPES))


if __name__ == "__main__":
    main()
