"""lr-17a: two `tl.constexpr` arguments of different types with one `str` share a disk-cache entry.

Survey, area `semantics` (what a launch means). The subset admits a `tl.constexpr` parameter whose
argument is a Python `int`, `float`, `bool`, `str` or `None` (SEMANTICS.md "The supported subset",
"Parameters and arguments"). Triton's in-process cache keys the launch by the `repr` of each
constexpr (`primitives/triton-3.8.0-jit.py:592-613`, a `str` of tuples), but the disk cache keys
the compiled kernel by `ASTSource.hash`, which writes each constexpr with `str`
(`primitives/triton-3.8.0-compiler.py:71-76`; `get_cache_key`, `triton/runtime/cache.py:319-321`
of the installed 3.8.0). So the string `"1"` and the integer `1` (and `"None"` and `None`,
`"True"` and `True`, `"2.5"` and `2.5`) give one hash, and a launch with one of them can load the
kernel compiled for the other.

Kernel: `if MODE == "1"` (resp. `MODE is None`, `MODE == "True"`, `MODE == "2.5"`): store 10, else 20.

  X0  fresh cache, one process: MODE = "1" alone (control: 10)
  X1  shared cache, two processes: the first launches MODE = 1 (20), the second MODE = "1"
  X2  shared cache, one process: MODE = 1, then MODE = "1"
  X3  shared cache, two processes: MODE = "1" first (10), then MODE = 1
  X4  shared cache, two processes: MODE = None, then MODE = "None" (kernel tests `MODE == "None"`)
  X5  shared cache, two processes: MODE = 2.5, then MODE = "2.5"

GPU 0. Run from TVA's root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-17/lr-17a/work/constexpr_cache.py
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_one(out_ptr, MODE: tl.constexpr):
    if MODE == "1":
        tl.store(out_ptr, 10)
    else:
        tl.store(out_ptr, 20)


@triton.jit
def k_none(out_ptr, MODE: tl.constexpr):
    if MODE == "None":
        tl.store(out_ptr, 10)
    else:
        tl.store(out_ptr, 20)


@triton.jit
def k_flt(out_ptr, MODE: tl.constexpr):
    if MODE == "2.5":
        tl.store(out_ptr, 10)
    else:
        tl.store(out_ptr, 20)


KERNELS = {"one": k_one, "none": k_none, "flt": k_flt}
VALUES = {"int1": 1, "str1": "1", "None": None, "strNone": "None", "f2.5": 2.5, "str2.5": "2.5"}


def launch(kname, vname):
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    KERNELS[kname][(1,)](out, MODE=VALUES[vname])
    torch.cuda.synchronize()
    return int(out.item())


def child(cache, steps):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--steps", json.dumps(steps)],
                       capture_output=True, text=True, env=env)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line:
        return json.loads(line[len("RESULT "):])
    err = [l for l in p.stderr.strip().splitlines() if l.strip()]
    return "process failed: " + (err[-1][:160] if err else "(no output)")


CASES = {
    "X0": ("MODE = '1' alone, fresh cache (expected 10)", [[["one", "str1"]]]),
    "X1": ("process 1: MODE = 1; process 2: MODE = '1' (expected 20, then 10)", [[["one", "int1"]], [["one", "str1"]]]),
    "X2": ("one process: MODE = 1, then MODE = '1' (expected 20, then 10)", [[["one", "int1"], ["one", "str1"]]]),
    "X3": ("process 1: MODE = '1'; process 2: MODE = 1 (expected 10, then 20)", [[["one", "str1"]], [["one", "int1"]]]),
    "X4": ("process 1: MODE = None; process 2: MODE = 'None' (expected 20, then 10)", [[["none", "None"]], [["none", "strNone"]]]),
    "X5": ("process 1: MODE = 2.5; process 2: MODE = '2.5' (expected 20, then 10)", [[["flt", "f2.5"]], [["flt", "str2.5"]]]),
}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--steps":
        steps = json.loads(sys.argv[2])
        res = []
        for kname, vname in steps:
            try:
                res.append(launch(kname, vname))
            except Exception as e:
                res.append("rejected on the host: " + type(e).__name__)
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (desc, procs) in CASES.items():
        with tempfile.TemporaryDirectory() as cache:
            stored = [child(cache, steps) for steps in procs]
        print(f"{name}: {desc}\n  stored {stored}")


if __name__ == "__main__":
    main()
