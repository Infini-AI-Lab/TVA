"""TVA library evidence (library issues L91, L104, L105): host code that changes what a launch runs, or whether
it runs, with no knob, hook, launch option or statement of the kernel's module.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/host-round20/run.py
Output: evidence/host-round20/out.txt (README.md says what it shows).

Each case writes the kernel module to a temporary file (`@triton.jit` reads its function's source file),
imports it from this script (another module, as a host program would) and runs in a process of its own
with an empty Triton cache. Every line of the kernel module is in the supported subset.

G0-G3 (L91) CUDA graph capture of `o = x + 1` (compiled before the capture): G0 the call made on the
    capturing stream (`torch.cuda.graph`'s); G1, G2, G3 the call made on another stream
    (`with torch.cuda.stream(s2)`) while a capture is in progress, in the capture modes "global",
    "thread_local" and "relaxed" (the capture also records a `torch` operation on its own stream). What
    the output holds after the capture, before any replay, and after it is reset and the graph replayed.
M0-M4 (L104) the kernel stores `double(x)` (a JIT helper, `x * 2`), or `tl.exp(x)`:
    M0 the control; M1 `kmod.double = kmod.triple` (the helper's module-level name rebound from this
    script); M2 `kmod.k.fn.__globals__["double"] = kmod.triple`; M3 `tl.exp = tl.log` (an attribute of
    Triton's module); M4 `kmod.k._unsafe_update_src(<a def of k that stores 5·x>)`
    (`primitives/triton-3.8.0-jit.py:561-569`).
D0, D1 (L105) an `int32` `a + b` that overflows in lane 0 (`2^31 - 1 + 1`): D0 plain; D1 after
    `kmod.add_i32.debug = True`, the kernel's `JITFunction` attribute (`JITFunction.run` takes
    `kwargs.get("debug", self.debug)`, `primitives/triton-3.8.0-jit.py:727`), with no launch keyword,
    decorator argument, environment variable or knob.
P0, P1 (L105) `JITFunction.preload` (`primitives/triton-3.8.0-jit.py:823-875`): a first process launches
    `y = tl.math.exp2(x)` with `enable_reflect_ftz=False` and records its specialisation data with the key
    of a plain launch (through a `jit_cache_hook` that lets the compile go on, in that process only); P1,
    a second process with an empty cache and no hook, calls `k.preload(<the data>)` and then launches
    `k[(1,)](x, y)` with no option, on `x = -130`; P0, the control, launches plainly. The model's
    `exp2(-130)` is `0` (libdevice's flush to zero at Triton's default `enable_reflect_ftz=True`).
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

KMOD = '''import triton
import triton.language as tl


@triton.jit
def add1(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1, mask=m)


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
def expk(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.exp(x))


@triton.jit
def add_i32(a_ptr, b_ptr, o_ptr):
    i = tl.arange(0, 16)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(o_ptr + i, a + b)


@triton.jit
def exp2k(x_ptr, y_ptr):
    i = tl.arange(0, 16)
    tl.store(y_ptr + i, tl.math.exp2(tl.load(x_ptr + i)))
'''

NEW_K = '''def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * 5)
'''


def load(d):
    p = os.path.join(d, "kmod.py")
    if not os.path.exists(p):
        with open(p, "w") as fh:
            fh.write(KMOD)
    spec = importlib.util.spec_from_file_location("kmod", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["kmod"] = m
    spec.loader.exec_module(m)
    return m


def case_g(name, m):
    import torch
    n = 4096
    x = torch.ones(n, device="cuda")
    o = torch.full((n,), -7.0, device="cuda")
    y = torch.zeros(n, device="cuda")
    m.add1[(n // 1024,)](x, o, n, BLOCK=1024)   # compiles before the capture
    torch.cuda.synchronize()
    o.fill_(-7.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s2 = torch.cuda.Stream()
    mode = {"G0": "global", "G1": "global", "G2": "thread_local", "G3": "relaxed"}[name]
    err = None
    try:
        with torch.cuda.graph(g, capture_error_mode=mode):
            y.add_(1.0)
            if name == "G0":
                m.add1[(n // 1024,)](x, o, n, BLOCK=1024)
            else:
                with torch.cuda.stream(s2):
                    m.add1[(n // 1024,)](x, o, n, BLOCK=1024)
    except Exception as e:
        err = type(e).__name__
    s2.synchronize()
    torch.cuda.synchronize()
    res = (f"returned {'with ' + err if err else 'without an error'}; after the capture, before any replay, "
           f"o[0], o[-1] = {float(o[0])}, {float(o[-1])}")
    if err is None:
        o.fill_(-7.0)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        res += f"; reset to -7 and replayed once: {float(o[0])}, {float(o[-1])}"
    return res


def case(name, d):
    import torch
    import triton
    import triton.language as tl
    m = load(d)
    if name.startswith("G"):
        return case_g(name, m)
    x = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    o = torch.full((16,), -7.0, dtype=torch.float32, device="cuda")
    if name in ("M0", "M1", "M2", "M4"):
        if name == "M1":
            m.double = m.triple
        elif name == "M2":
            m.k.fn.__globals__["double"] = m.triple
        elif name == "M4":
            m.k._unsafe_update_src(NEW_K)
        m.k[(1,)](x, o)
        torch.cuda.synchronize()
        what = "2·x (the source's)" if torch.equal(o, 2 * x) else "3·x" if torch.equal(o, 3 * x) else \
            "5·x" if torch.equal(o, 5 * x) else "something else"
        return f"stored {o[:3].tolist()} …: {what}"
    if name == "M3":
        tl.exp = tl.log
        m.expk[(1,)](x, o)
        torch.cuda.synchronize()
        what = "exp x (the source's)" if torch.allclose(o, torch.exp(x)) else \
            "log x" if torch.allclose(o, torch.log(x)) else "something else"
        return f"stored {[round(v, 6) for v in o[:3].tolist()]} …: {what}"
    if name in ("D0", "D1"):
        a = torch.zeros(16, dtype=torch.int32, device="cuda")
        b = torch.zeros(16, dtype=torch.int32, device="cuda")
        a[0], b[0] = 2 ** 31 - 1, 1
        oi = torch.full((16,), 7, dtype=torch.int32, device="cuda")
        if name == "D1":
            m.add_i32.debug = True
        try:
            m.add_i32[(1,)](a, b, oi)
            torch.cuda.synchronize()
            return f"returned; lane 0 stored {int(oi[0])} (wrapping: {-2 ** 31})"
        except Exception as e:   # a device assert reaches the host when the stream synchronises
            return f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
    xs = torch.full((16,), -130.0, dtype=torch.float32, device="cuda")
    ys = torch.full((16,), -7.0, dtype=torch.float32, device="cuda")
    if name == "Pgen":
        seen = []

        def hook(*args, **kw):
            seen.append((kw["key"], kw["compile"]["specialization_data"]))
            return False
        triton.knobs.runtime.jit_cache_hook = hook
        m.exp2k[(1,)](xs, ys, enable_reflect_ftz=False)
        m.exp2k[(1,)](xs, ys)
        torch.cuda.synchronize()
        triton.knobs.runtime.jit_cache_hook = None
        data = json.loads(seen[0][1])
        data["key"] = seen[1][0]
        with open(os.path.join(d, "spec.json"), "w") as fh:
            fh.write(json.dumps(data))
        return (f"recorded {len(seen)} compiles; the first's options give enable_reflect_ftz = "
                f"{data['options'].get('enable_reflect_ftz')}; the data now names the plain launch's key")
    if name == "P1":
        with open(os.path.join(d, "spec.json")) as fh:
            m.exp2k.preload(fh.read())
    m.exp2k[(1,)](xs, ys)
    torch.cuda.synchronize()
    v = float(ys[0])
    return f"a plain launch stored exp2(-130) = {v!r} ({'2^-130' if v == 2.0 ** -130 else '0, the model' if v == 0 else '?'})"


def run(name, d):
    env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="tva-host20-cache-"))
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name, d],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            out = json.loads(line[7:])
            break
    else:
        out = f"exit {r.returncode}"
    err = [re.sub(r"\S*kmod\.py", "kmod.py", l).strip()[:120] for l in r.stderr.splitlines() if "Assertion" in l][:1]
    return out + ("" if not err else f"  [device: {err[0]}]")


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    labels = {"G0": "the call on the capturing stream", "G1": "the call on another stream, capture mode global",
              "G2": "the call on another stream, capture mode thread_local",
              "G3": "the call on another stream, capture mode relaxed",
              "M0": "control", "M1": "kmod.double = kmod.triple", "M2": "kmod.k.fn.__globals__['double'] = kmod.triple",
              "M3": "tl.exp = tl.log", "M4": "kmod.k._unsafe_update_src(<k storing 5·x>)",
              "D0": "plain", "D1": "kmod.add_i32.debug = True",
              "Pgen": "first process", "P1": "second process: preload, then a plain launch",
              "P0": "control: a plain launch, no preload"}
    for name in ("G0", "G1", "G2", "G3", "M0", "M1", "M2", "M3", "M4", "D0", "D1"):
        d = tempfile.mkdtemp(prefix="tva-host20-")
        print(f"{name} ({labels[name]}): {run(name, d)}")
    d = tempfile.mkdtemp(prefix="tva-host20-")
    for name in ("Pgen", "P1", "P0"):
        print(f"{name} ({labels[name]}): {run(name, d)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--case":
        print("RESULT " + json.dumps(case(sys.argv[2], sys.argv[3])))
    else:
        main()
