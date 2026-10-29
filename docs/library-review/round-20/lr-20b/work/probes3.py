"""lr-20b probes, part 3: the host's Python run with `-O` (or PYTHONOPTIMIZE set), which removes Python `assert`
statements, among them the checks by which Triton's front end rejects some kernels on the host
(`primitives/triton-3.8.0-code-generator.py:497, 1020, 1117-1121`). SEMANTICS states these kernels as rejected.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-20/lr-20b/work/probes3.py
Each case runs in a child process of its own, with an empty TRITON_CACHE_DIR, once as `python -I` and once as
`python -I -O`. No path is printed.

- D. A name holding a dtype before a `for` loop whose body assigns it another dtype (SEMANTICS "Loops and unrolling",
  "Names after a loop": "a body that assigns a name holding a compile-time value (a `tl.constexpr`, `None`, a dtype)
  ... is rejected on the host"): `d = tl.float16; for i in range(n): d = tl.float32`, then
  `tl.store(o, tl.full([1], 1 + 2**-11, d).to(tl.float32))` with `n = 0` (the loop runs no iteration) and `n = 3`.
  A `float16` `d` stores 1.0 (the constant rounds to even), a `float32` `d` stores 1.00048828125.
- E. A runtime conditional expression whose sides have two types (SEMANTICS "Order": rejected on the host):
  `tl.full([1], 1 if c > 0 else 1.0, tl.float32)` with `c = 5`.
"""
import json
import os
import subprocess
import sys
import tempfile

SRC = '''import triton
import triton.language as tl


@triton.jit
def kd(o_ptr, n):
    d = tl.float16
    for i in range(n):
        d = tl.float32
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], 1 + 2**-11, d).to(tl.float32))


@triton.jit
def ke(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], 1 if c > 0 else 1.0, tl.float32))
'''


def child(name, arg):
    import importlib.util
    import torch
    tmp = tempfile.mkdtemp(prefix="lr20b3-")
    path = os.path.join(tmp, "lr20b3_mod.py")
    with open(path, "w") as fh:
        fh.write(SRC)
    spec = importlib.util.spec_from_file_location("lr20b3_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    o = torch.full((1,), -7.0, dtype=torch.float32, device="cuda")
    try:
        getattr(mod, name)[(1,)](o, arg)
        torch.cuda.synchronize()
        print(json.dumps(repr(o.item())))
    except Exception as ex:
        print(json.dumps(f"rejected on the host ({type(ex).__name__})"))


def run(name, arg, opt):
    env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="lr20b3-cache-"))
    env.pop("PYTHONOPTIMIZE", None)
    cmd = [sys.executable, "-I"] + (["-O"] if opt else []) + [os.path.abspath(__file__), "--child", name, str(arg)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith('"')]
    return json.loads(lines[-1]) if lines else f"child failed (exit {r.returncode})"


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    for name, arg, what in (("kd", 0, "D, n = 0"), ("kd", 3, "D, n = 3"), ("ke", 5, "E, c = 5")):
        print(f"  {what}: python -I -> {run(name, arg, False)}; python -I -O -> {run(name, arg, True)}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]))
    else:
        main()
