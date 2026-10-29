"""lr-19a: the disk cache key of a `tl.constexpr` argument is `str(value)`
(primitives/triton-3.8.0-compiler.py:71-76, `get_key`), which writes the int `1` and the string
`"1"` alike (and `True`/`"True"`, `None`/`"None"`). The in-process key is the `repr`
(primitives/triton-3.8.0-jit.py:409, 611), which tells them apart. So a launch with `MODE="1"`, in a
process whose cache folder holds the kernel compiled for `MODE=1`, may run that kernel.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I cache_str.py
(the script makes its own cache folders under the system's temporary folder and runs each launch in
a child process of its own).
"""
import json
import os
import subprocess
import sys
import tempfile

KERNEL = '''import triton
import triton.language as tl


@triton.jit
def pick(o, MODE: tl.constexpr):
    if MODE == 1:
        tl.store(o + tl.arange(0, 1), tl.full([1], 1, tl.int32))
    else:
        tl.store(o + tl.arange(0, 1), tl.full([1], 2, tl.int32))
'''


def child(val_json):
    import importlib.util
    import torch
    val = json.loads(val_json)
    tmp = tempfile.mkdtemp(prefix="lr19a-")
    path = os.path.join(tmp, "cache_str_kernel.py")
    with open(path, "w") as fh:
        fh.write(KERNEL)
    spec = importlib.util.spec_from_file_location("cache_str_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    o = torch.zeros(1, dtype=torch.int32, device="cuda")
    mod.pick[(1,)](o, MODE=val)
    torch.cuda.synchronize()
    print("RESULT", o.item())


def run(val, cache):
    env = dict(os.environ, TRITON_CACHE_DIR=cache)
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", json.dumps(val)],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return int(line.split()[1])
    raise SystemExit("child failed: " + r.stderr[-1500:])


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("Python's meaning: MODE=1 stores 1; MODE='1' stores 2 ('1' == 1 is False)")
    fresh = tempfile.mkdtemp(prefix="lr19a-cache-")
    print(f"K1 empty cache, MODE='1': stores {run('1', fresh)}")
    shared = tempfile.mkdtemp(prefix="lr19a-cache-")
    a = run(1, shared)
    b = run("1", shared)
    print(f"K2 one cache folder, a process with MODE=1 stores {a}, then a process with MODE='1' stores {b}")
    shared2 = tempfile.mkdtemp(prefix="lr19a-cache-")
    c = run("1", shared2)
    d = run(1, shared2)
    print(f"K3 one cache folder, a process with MODE='1' stores {c}, then a process with MODE=1 stores {d}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
