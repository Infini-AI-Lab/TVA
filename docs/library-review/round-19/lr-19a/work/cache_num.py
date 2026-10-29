"""lr-19a: the in-process kernel key of a `tl.constexpr` argument. `compute_cache_key`
(primitives/triton-3.8.0-jit.py:592-611) looks the specialization up in `kernel_key_cache`, a dict
keyed by the tuple of `("constexpr", value)` pairs (`:409`). Python's `1 == 1.0 == True` (and
`0 == 0.0 == -0.0 == False`) with equal hashes, so a launch with `C=1.0` or `C=True`, after one with
`C=1` in the same process, gets the cache key, and so the kernel, compiled for `C=1`.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I cache_num.py
(each order runs in a child process of its own, with an empty cache folder).
"""
import json
import os
import subprocess
import sys
import tempfile

KERNEL = '''import triton
import triton.language as tl


@triton.jit
def kind(o, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    if isinstance(C, float):
        tl.store(o + i, tl.full([1], 2, tl.int32))
    else:
        tl.store(o + i, tl.full([1], 1, tl.int32))
    x = tl.load(x_ptr + i)
    tl.store(o + 1 + i, x * C)
'''

MEANING = {"1": (1, 16777217), "1.0": (2, 16777216), "True": (1, 16777217),
           "0": (1, 0), "0.0": (2, 0), "-0.0": (2, 0), "False": (1, 0)}


def child(order_json):
    import importlib.util
    import torch
    tmp = tempfile.mkdtemp(prefix="lr19a-")
    path = os.path.join(tmp, "cache_num_kernel.py")
    with open(path, "w") as fh:
        fh.write(KERNEL)
    spec = importlib.util.spec_from_file_location("cache_num_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.tensor([2 ** 24 + 1], dtype=torch.int32, device="cuda")
    res = []
    for text in json.loads(order_json):
        o = torch.zeros(2, dtype=torch.int32, device="cuda")
        mod.kind[(1,)](o, x, C=eval(text))
        torch.cuda.synchronize()
        res.append(o.tolist())
    print("RESULT", json.dumps(res))


def run(order):
    env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="lr19a-cache-"))
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", json.dumps(order)],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[7:])
    raise SystemExit("child failed: " + r.stderr[-1500:])


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("kernel: stores 2 if isinstance(C, float), else 1; then x * C, x = 2^24 + 1 "
          "(int32), stored into int32 (a float C computes in float32)")
    for order in (["1", "1.0"], ["1.0", "1"], ["1.0", "True"], ["0", "0.0", "False"], ["-0.0", "0"]):
        got = run(order)
        cells = []
        for text, g in zip(order, got):
            want = list(MEANING[text])
            cells.append(f"C={text}: stored {g}, Python's meaning {want}" + ("" if g == want else " DIFFERS"))
        print("one process, in this order: " + "; ".join(cells))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
