"""lr-20a probes (library review round 20), run from the TVA root:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-20/lr-20a/work/probes.py

Output: probes.out beside this file (the run's stdout). Every case runs in a process of its own with an
empty TRITON_CACHE_DIR of its own (a fresh cache directory), made under TMPDIR.

P1 (L64, SEMANTICS "Values", fp8 row: "a Python number built in them is its exact value, or the kernel is
    rejected"): fp8 constants that are neither the number's exact value nor rejected, and integer and
    bool constants beside an fp8 tensor. Each stored byte, against FloatV.litF8 (Fp8Probe.lean).
P2 (L103, REVIEWER_GUIDE item 11 and compiler-faithful.md checklist 11: "A fresh cache directory, or values
    Triton keys apart, meets it"; "each compile-time parameter given values that Triton keys apart"):
    - P2a one process, a fresh cache directory, a kernel launched with C = 1 and then C = 1.0;
    - P2b one process, a fresh cache directory, two constexpr parameters whose values are keyed apart
      parameter by parameter (A in {"a-b", "a"}, B in {"c", "b-c"}): (A, B) = ("a-b", "c"), then
      ("a", "b-c");
    each against the second launch alone in a fresh process and cache.
"""
import json
import os
import subprocess
import sys
import tempfile

SRC = '''import triton
import triton.language as tl


@triton.jit
def full_e5(o_ptr, C: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], C, tl.float8e5))


@triton.jit
def full_e4(o_ptr, C: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], C, tl.float8e4nv))


@triton.jit
def where_e5(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.where(i < 0, tl.load(x_ptr + i), C))


@triton.jit
def where_e4(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.where(i < 0, tl.load(x_ptr + i), C))


@triton.jit
def kc(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * C)


@triton.jit
def kab(o_ptr, A: tl.constexpr, B: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], len(A) * 10 + len(B), tl.int32))
'''

# P1: (kernel, value as Python source, Lean double pattern)
P1 = [("full_e5", "1.0000000009313226", "1 + 2^-30: float32 constant 1.0"),
      ("full_e4", "1.0000000009313226", "1 + 2^-30: float32 constant 1.0"),
      ("full_e5", "-1e-50", "float32 constant -0.0"),
      ("full_e4", "-1e-50", "float32 constant -0.0"),
      ("full_e5", "1e39", "float32 constant +inf"),
      ("full_e5", "3", "an int"),
      ("full_e4", "448", "an int"),
      ("where_e4", "448", "an int beside a float8e4nv tensor"),
      ("where_e5", "True", "a bool beside a float8e5 tensor"),
      ("where_e5", "-1e-50", "float32 constant -0.0, beside a float8e5 tensor")]


def load(tmp):
    path = os.path.join(tmp, "lr20a_mod.py")
    with open(path, "w") as fh:
        fh.write(SRC)
    import importlib.util
    spec = importlib.util.spec_from_file_location("lr20a_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def child(plan):
    import torch
    mod = load(tempfile.mkdtemp())
    out = []
    for kern, val in json.loads(plan):
        try:
            if kern.startswith("full") or kern.startswith("where"):
                tdt = torch.float8_e5m2 if kern.endswith("e5") else torch.float8_e4m3fn
                o = torch.full((1,), 0x55, dtype=torch.uint8, device="cuda").view(tdt)
                x = torch.zeros(1, dtype=torch.uint8, device="cuda").view(tdt)
                c = eval(val)
                if kern.startswith("full"):
                    getattr(mod, kern)[(1,)](o, C=c)
                else:
                    getattr(mod, kern)[(1,)](o, x, C=c)
                torch.cuda.synchronize()
                out.append(f"{o.view(torch.uint8).item():02x}")
            elif kern == "kc":
                o = torch.zeros(1, dtype=torch.int32, device="cuda")
                x = torch.full((1,), 2 ** 24 + 1, dtype=torch.int32, device="cuda")
                mod.kc[(1,)](o, x, C=eval(val))
                torch.cuda.synchronize()
                out.append(o.item())
            else:
                a, b = eval(val)
                o = torch.zeros(1, dtype=torch.int32, device="cuda")
                mod.kab[(1,)](o, A=a, B=b)
                torch.cuda.synchronize()
                out.append(o.item())
        except Exception as e:
            out.append(f"rejected ({type(e).__name__})")
    print("RESULT " + json.dumps(out))


def run(plan):
    env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp())
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", json.dumps(plan)],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise SystemExit("child failed: " + r.stderr[-1500:])


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    print("P1. fp8 constants (stored byte; FloatV.litF8 in Fp8Probe.out):")
    for kern, val, note in P1:
        print(f"  {kern}, C = {val} ({note}): {run([[kern, val]])[0]}")
    print("P2a. one process, fresh cache directory: kc (x * C, x = 2^24 + 1) with C = 1, then C = 1.0")
    seq = run([["kc", "1"], ["kc", "1.0"]])
    print(f"  C = 1 -> {seq[0]}; then C = 1.0 -> {seq[1]}; C = 1.0 alone (fresh process and cache) -> "
          f"{run([['kc', '1.0']])[0]}")
    print("P2b. one process, fresh cache directory: kab (10*len(A) + len(B)) with (A, B) = ('a-b', 'c'), "
          "then ('a', 'b-c'); A's two values and B's two values are each keyed apart")
    seq = run([["kab", "('a-b', 'c')"], ["kab", "('a', 'b-c')"]])
    print(f"  ('a-b', 'c') -> {seq[0]}; then ('a', 'b-c') -> {seq[1]}; ('a', 'b-c') alone -> "
          f"{run([['kab', repr(('a', 'b-c'))]])[0]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
