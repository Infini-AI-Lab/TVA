"""lr-20b probes, part 2 (library issue L64): fp8 constants whose `float32` is a zero although the Python number
is not (`litF8` gives the signed zero: `Check.lean` evaluates `constIn?` of `-1e-50` in `float8e5` as `0x80`),
one whose `float32` is the smallest subnormal (`litF8` gives `none`: rejected on the host), and `1 + 2^-40`,
which no fp8 format holds but whose `float32` is `1.0` (`litF8`: `1.0`, `0x3c` in `float8e5`).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-20/lr-20b/work/probes2.py
Each value is a `tl.constexpr` argument in a process of its own with an empty TRITON_CACHE_DIR. No path is printed.
"""
import json
import os
import subprocess
import sys
import tempfile

SRC = '''import triton
import triton.language as tl


@triton.jit
def full8(o_ptr, C: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], C, tl.float8e5))


@triton.jit
def where8(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, tl.where(i < 0, x, C))
'''


def child(name, c):
    import importlib.util
    import torch
    tmp = tempfile.mkdtemp(prefix="lr20b2-")
    path = os.path.join(tmp, "lr20b2_mod.py")
    with open(path, "w") as fh:
        fh.write(SRC)
    spec = importlib.util.spec_from_file_location("lr20b2_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    o = torch.full((1,), 0x55, dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    try:
        if name == "full8":
            mod.full8[(1,)](o, C=c)
        else:
            x = torch.zeros(1, dtype=torch.float8_e5m2, device="cuda")
            mod.where8[(1,)](o, x, C=c)
        torch.cuda.synchronize()
        print(json.dumps(f"{o.view(torch.uint8).item():02x}"))
    except Exception as ex:
        print(json.dumps(f"rejected on the host ({type(ex).__name__})"))


def run(name, c):
    env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="lr20b2-cache-"))
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", name, repr(c)],
                       env=env, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith('"')]
    return json.loads(lines[-1]) if lines else f"child failed (exit {r.returncode})"


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    for c, lib in ((-1e-50, "80"), (1e-50, "00"), (-1e-45, "none (rejected)"), (1 + 2 ** -40, "3c (1.0)")):
        print(f"  c = {c!r}: tl.full -> {run('full8', c)}, tl.where -> {run('where8', c)} (litF8: {lib})")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--child":
        child(sys.argv[2], float(sys.argv[3]))
    else:
        main()
