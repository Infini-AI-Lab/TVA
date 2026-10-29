"""lr-20d, library review round 20 (survey area: rules): two more checks of Compiler.Faithful's scope 11.

- Y1: the in-process key equates every pair of values Python's `==` equates, not only those the
  reviewer's checklist lists (`1`, `1.0`, `True`; `0`, `0.0`, `-0.0`, `False`): one process launches
  with `C = 2` and then `C = 2.0` (and the reverse), `C = 3` then `C = 3.0`; the kernel stores `x * C`
  for the `int32` `x = 2^24 + 1` and whether `isinstance(C, float)`.
- Y2: `TRITON_ALWAYS_COMPILE=1` (an exception of scope 2) in the later process: does it keep a later
  process from running code an earlier process put in the shared cache folder? (a) `MODE = 1` in a
  first process, then `MODE = "1"` in a second on the same folder with `TRITON_ALWAYS_COMPILE=1`;
  (b) a first process with `tl.exp = tl.log`, then a plain second process on the same folder with
  `TRITON_ALWAYS_COMPILE=1`.

Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
No path is printed.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

KSRC = '''import triton
import triton.language as tl


@triton.jit
def k1(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * C)
    if isinstance(C, float):
        tl.store(o_ptr + 1 + i, tl.full([1], 1, tl.int32))


@triton.jit
def k3(o_ptr, MODE: tl.constexpr):
    i = tl.arange(0, 1)
    if MODE == "1":
        tl.store(o_ptr + i, tl.full([1], 20, tl.int32))
    else:
        tl.store(o_ptr + i, tl.full([1], 10, tl.int32))


@triton.jit
def expk(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    tl.store(o_ptr + i, tl.exp(tl.load(x_ptr + i)))
'''


def child(plan, kdir):
    import torch
    import triton.language as tl
    plan = json.loads(plan)
    if plan.get("patch"):
        tl.exp = tl.log
    sys.path.insert(0, kdir)
    import kmod_lr20d_keys as kmod
    out = []
    for kern, val in plan["launches"]:
        if kern == "k1":
            o = torch.zeros(2, dtype=torch.int32, device="cuda")
            x = torch.full((1,), 2**24 + 1, dtype=torch.int32, device="cuda")
            kmod.k1[(1,)](o, x, val)
            out.append(o.tolist())
        elif kern == "k3":
            o = torch.zeros(1, dtype=torch.int32, device="cuda")
            kmod.k3[(1,)](o, val)
            out.append(o.tolist())
        else:
            x = torch.full((16,), 2.0, device="cuda")
            o = torch.zeros(16, device="cuda")
            kmod.expk[(1,)](x, o)
            out.append(round(o[0].item(), 6))
    return out


def run(plan, kdir, cache, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
    env["TRITON_CACHE_DIR"] = cache
    env.update(extra or {})
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "child", json.dumps(plan), kdir],
                       env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return "ERROR " + (p.stderr.strip().splitlines() or ["?"])[-1][:120]


def main():
    import torch
    import triton
    kdir = tempfile.mkdtemp(prefix="lr20d-keys-")
    with open(os.path.join(kdir, "kmod_lr20d_keys.py"), "w") as fh:
        fh.write(KSRC)
    fresh = lambda: tempfile.mkdtemp(prefix="c-", dir=kdir)
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"python {sys.version.split()[0]}")
    wrong = 0
    print("Y1. one process, fresh cache, k1 (x * C, isinstance(C, float)); each value alone as the control:")
    for a, b in [(2, 2.0), (2.0, 2), (3, 3.0)]:
        got = run({"launches": [["k1", a], ["k1", b]]}, kdir, fresh())
        alone = run({"launches": [["k1", b]]}, kdir, fresh())[0]
        bad = got[1] != alone
        wrong += bad
        print(f"  {a!r} then {b!r}: {got[0]}, {got[1]} (alone: {alone}){' DIFFERS' if bad else ''}")
    print("Y2. a first process, then a second on the same cache folder with TRITON_ALWAYS_COMPILE=1:")
    s = fresh()
    r1 = run({"launches": [["k3", 1]]}, kdir, s)
    r2 = run({"launches": [["k3", "1"]]}, kdir, s, {"TRITON_ALWAYS_COMPILE": "1"})
    r2n = run({"launches": [["k3", "1"]]}, kdir, s)
    print(f"  (a) MODE = 1 -> {r1}; then MODE = '1' with TRITON_ALWAYS_COMPILE=1 -> {r2} "
          f"(the string's own code: [[20]]); then MODE = '1' without it -> {r2n}")
    s = fresh()
    r1 = run({"patch": True, "launches": [["expk", None]]}, kdir, s)
    r2 = run({"launches": [["expk", None]]}, kdir, s, {"TRITON_ALWAYS_COMPILE": "1"})
    print(f"  (b) tl.exp = tl.log in the first -> {r1}; plain second with TRITON_ALWAYS_COMPILE=1 -> {r2} "
          f"(exp(2) = 7.389056)")
    print(f"Y1 launches that ran code compiled for another value: {wrong} of 3")
    shutil.rmtree(kdir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print("RESULT " + json.dumps(child(sys.argv[2], sys.argv[3])))
    else:
        main()
