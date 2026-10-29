"""lr-18b: host code that changes what the launch compiles without a knob, a hook, an option, a decorator or a
statement of the kernel's module (`Compiler.Faithful`'s scope; SEMANTICS "The supported subset", Names).

Each case writes a kernel module to a temporary file (`@triton.jit` reads its function's source file), imports it
from this script (another module, as a host program or test harness would), changes one thing from here, and
launches the kernel once on 16 values. Every line of the kernel module is in the supported subset: imports of
`triton` and `triton.language as tl`, `@triton.jit` functions, and the kernel's body. What the kernel module says
is the model a reviewer would write from it; what does the device store?

- R1: `kmod.double = kmod.triple` (this script rebinds the kernel module's JIT helper `double`, which the kernel
  calls; the module's own statements bind `double` only by its `def`). Model: 2*x.
- R2: `kmod.k.fn.__globals__["double"] = kmod.triple` (the same through the kernel's globals). Model: 2*x.
- D1: `kmod.k.debug = True` (an attribute of the kernel's JITFunction; `JITFunction.run` reads `self.debug`,
  `triton-3.8.0-jit.py:727`), on `int32` `a + b` with `a = 2^31 - 1`, `b = 1` in lane 0. Model (`debug = false`,
  as `Launch.debug`'s docstring gives it for this host: no launch keyword, decorator, environment variable or knob
  asks for it): wraps to `-2^31`. D0 is the control without the attribute.
- T1: `tl.exp = tl.log` (an attribute of Triton's module `triton.language`, which the kernel reads as `tl.exp`).
  Model: `exp(x)`.
- S1: `kmod.k._unsafe_update_src(<a def of k that stores 5*x>)` (a method of the kernel's JITFunction,
  `triton-3.8.0-jit.py:561-569`). Model: 2*x.

Run: `CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I host_mutation.py > host_mutation.out` (each case runs in its
own process, so a fault in one leaves the others alone).
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

KMOD = '''import triton
import triton.language as tl


@triton.jit
def double(x):
    return x * 2


@triton.jit
def triple(x):
    return x * 3


@triton.jit
def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, double(x))


@triton.jit
def add_i32(a_ptr, b_ptr, o_ptr):
    i = tl.arange(0, 16)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(o_ptr + i, a + b)


@triton.jit
def expk(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.exp(x))
'''

NEW_K = '''def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * 5)
'''


def load(d):
    p = os.path.join(d, "kmod.py")
    open(p, "w").write(KMOD)
    spec = importlib.util.spec_from_file_location("kmod", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["kmod"] = m
    spec.loader.exec_module(m)
    return m


def case(name):
    import torch
    import triton.language as tl
    d = tempfile.mkdtemp()
    m = load(d)
    x = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    o = torch.full((16,), -7.0, dtype=torch.float32, device="cuda")
    if name in ("R1", "R2", "S1"):
        if name == "R1":
            m.double = m.triple
        elif name == "R2":
            m.k.fn.__globals__["double"] = m.triple
        else:
            m.k._unsafe_update_src(NEW_K)
        m.k[(1,)](x, o)
        torch.cuda.synchronize()
        print(f"- {name}: stored {o[:4].tolist()} …; o == 2·x (the model): {bool(torch.equal(o, 2 * x))}, "
              f"o == 3·x: {bool(torch.equal(o, 3 * x))}, o == 5·x: {bool(torch.equal(o, 5 * x))}")
    elif name in ("D0", "D1"):
        a = torch.zeros(16, dtype=torch.int32, device="cuda")
        b = torch.zeros(16, dtype=torch.int32, device="cuda")
        a[0], b[0] = 2 ** 31 - 1, 1
        oi = torch.full((16,), 7, dtype=torch.int32, device="cuda")
        if name == "D1":
            m.add_i32.debug = True
        try:
            m.add_i32[(1,)](a, b, oi)
            torch.cuda.synchronize()
            print(f"- {name}: returned; lane 0 stored {int(oi[0])} (the model, wrapping: {-2 ** 31})")
        except Exception as e:  # a device assert reaches the host as an error once the stream syncs
            print(f"- {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    elif name == "T1":
        tl.exp = tl.log
        m.expk[(1,)](x, o)
        torch.cuda.synchronize()
        print(f"- T1: stored {[round(v, 6) for v in o[:4].tolist()]} …; o ≈ exp(x) (the model): "
              f"{bool(torch.allclose(o, torch.exp(x)))}, o ≈ log(x): {bool(torch.allclose(o, torch.log(x)))}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        case(sys.argv[1])
        sys.exit(0)
    import torch
    import triton
    print(f"Triton {triton.__version__}, {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"Python {sys.version.split()[0]}")
    for c in ("R1", "R2", "D0", "D1", "T1", "S1"):
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), c], capture_output=True, text=True)
        out = r.stdout.strip() or f"- {c}: exit {r.returncode}"
        err = [l for l in r.stderr.splitlines() if "Assertion" in l or "Error" in l][:2]
        err = [re.sub(r"\S*kmod\.py", "kmod.py", e) for e in err]  # the temporary folder's name is not printed
        print(out + ("" if not err else "  [stderr: " + " | ".join(e.strip()[:140] for e in err) + "]"))
