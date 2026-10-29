"""lr-20d, library review round 20 (survey area: rules).

Question: Compiler.Faithful's scope 7 names the host (an x86-64 machine running CPython 3.13, library
issue L107) because some values SEMANTICS.md states are the host's. Does the host process's own state
change those values on that same host: its C numeric locale, or its floating-point environment
(rounding mode, flush-to-zero / denormals-are-zero)? Each is process state that host code sets with no
Triton knob, hook, option or kernel statement:
- the C locale: `locale.setlocale(locale.LC_NUMERIC, ...)` (a program that calls
  `locale.setlocale(locale.LC_ALL, "")` under such a `LANG` does the same). `get_bf16`
  (`primitives/triton-3.8.0-ir.cc:952-957`) builds a `bfloat16` constant from `std::to_string(v)`,
  C's `%f`, whose decimal point is the locale's; SEMANTICS "Constants" states `FloatV.litBF16`, the
  `bfloat16` nearest to the six-decimal text;
- the rounding mode: `fesetround` (glibc sets the x87 and SSE modes); `get_fp32` (`ir.cc:963-967`)
  takes a C++ `float` from the Python float, the launcher converts a `float` argument with C's
  `(float)` (SEMANTICS "Scalar arguments": `FloatV.hostF32`, round to nearest even), and the front end
  computes compile-time float arithmetic with Python's floats on the host's FPU (SEMANTICS: "the
  operator is Python's on their values");
- flush-to-zero and denormals-are-zero (MXCSR bits 15 and 6): set here by a two-line C function, as a
  shared library built with `-ffast-math` (crtfastmath) does when a Python package loads it.

Each case runs in a process of its own with a fresh Triton cache; the kernel stores
`tl.full([1], 0.1, tl.float32)`, `tl.full([1], 1e-40, tl.float32)`, the `float32` scalar argument
`y = 0.1` (as `y + zeros`), `tl.full([1], A + B, tl.float64)` for the constexprs `A = 0.1`, `B = 0.2`,
`tl.full([1], C * 1e300, tl.float64)` for `C = 1e-310`, `tl.full([1], 1.5, tl.bfloat16)`,
`x * 0.75` for a `bfloat16` `x = 2.0`, and `tl.full([1], 1.5e-6, tl.bfloat16)` (`litBF16`: the
`bfloat16` nearest to the `%f` text `0.000002` of its `float32`; glibc's `%f` rounds by the current
mode). The model's values (SEMANTICS) are those of the plain process.

Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
No path is printed.
"""
import json
import os
import subprocess
import sys
import tempfile

KSRC = '''import triton
import triton.language as tl


@triton.jit
def envk(o32_ptr, o64_ptr, obf_ptr, xbf_ptr, y, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o32_ptr + i, tl.full([1], 0.1, tl.float32))
    tl.store(o32_ptr + 1 + i, tl.full([1], 1e-40, tl.float32))
    tl.store(o32_ptr + 2 + i, y + tl.zeros([1], tl.float32))
    tl.store(o64_ptr + i, tl.full([1], A + B, tl.float64))
    tl.store(o64_ptr + 1 + i, tl.full([1], C * 1e300, tl.float64))
    tl.store(obf_ptr + i, tl.full([1], 1.5, tl.bfloat16))
    tl.store(obf_ptr + 1 + i, tl.load(xbf_ptr + i) * 0.75)
    tl.store(obf_ptr + 2 + i, tl.full([1], 1.5e-6, tl.bfloat16))
'''

CSRC = '''#include <xmmintrin.h>
void set_ftz_daz(void) { _mm_setcsr(_mm_getcsr() | 0x8040); }
'''


def child(mode, kdir):
    import ctypes
    import locale
    import torch
    sys.path.insert(0, kdir)
    import kmod_lr20d_env as kmod
    if mode == "locale_en_DK":
        locale.setlocale(locale.LC_NUMERIC, "en_DK.utf8")
    if mode == "round_toward_zero":
        assert ctypes.CDLL("libm.so.6").fesetround(0xC00) == 0
    if mode == "ftz_daz":
        ctypes.CDLL(os.path.join(kdir, "libftz.so")).set_ftz_daz()
    o32 = torch.zeros(3, dtype=torch.float32, device="cuda")
    o64 = torch.zeros(2, dtype=torch.float64, device="cuda")
    obf = torch.zeros(3, dtype=torch.bfloat16, device="cuda")
    xbf = torch.full((1,), 2.0, dtype=torch.bfloat16, device="cuda")
    kmod.envk[(1,)](o32, o64, obf, xbf, 0.1, 0.1, 0.2, 1e-310)
    torch.cuda.synchronize()
    w32 = [format(v, "08x") for v in o32.cpu().view(torch.int32).tolist()]
    w64 = [format(v & (2**64 - 1), "016x") for v in o64.cpu().view(torch.int64).tolist()]
    wbf = [format(v & 0xFFFF, "04x") for v in obf.cpu().view(torch.int16).tolist()]
    return {"0.1 f32": w32[0], "1e-40 f32": w32[1], "arg y=0.1": w32[2], "A+B f64": w64[0],
            "C*1e300 f64": w64[1], "1.5 bf16": wbf[0], "2.0*0.75 bf16": wbf[1], "1.5e-6 bf16": wbf[2]}


def run(mode, kdir):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="c-", dir=kdir)
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "child", mode, kdir],
                       env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    import re
    last = (p.stderr.strip().splitlines() or ["?"])[-1]
    # keep no path of the build or the machine in the output (the audit refuses absolute paths)
    last = re.sub(r"(?<![\w.])/[^\s:]+", "<path>", last)
    return f"ERROR (exit {p.returncode}) " + last[:200]


def main():
    import shutil
    import torch
    import triton
    kdir = tempfile.mkdtemp(prefix="lr20d-env-")
    with open(os.path.join(kdir, "kmod_lr20d_env.py"), "w") as fh:
        fh.write(KSRC)
    with open(os.path.join(kdir, "ftz.c"), "w") as fh:
        fh.write(CSRC)
    subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", os.path.join(kdir, "libftz.so"),
                    os.path.join(kdir, "ftz.c")], check=True)
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"python {sys.version.split()[0]}, machine {os.uname().machine}")
    base = run("plain", kdir)
    print(f"plain (the model's values): {base}")
    differ = 0
    for mode in ["locale_en_DK", "round_toward_zero", "ftz_daz"]:
        r = run(mode, kdir)
        if isinstance(r, dict):
            d = {k: f"{v} (plain {base[k]})" for k, v in r.items() if v != base[k]}
            differ += len(d)
            print(f"{mode}: differs from plain in {len(d)} of {len(r)} words: {d}")
        else:
            print(f"{mode}: {r}")
    print(f"words that differ from the plain process's: {differ}")
    shutil.rmtree(kdir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print("RESULT " + json.dumps(child(sys.argv[2], sys.argv[3])))
    else:
        main()
