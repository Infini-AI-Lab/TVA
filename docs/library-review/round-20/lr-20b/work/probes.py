"""lr-20b probes (library review round 20, survey area `transcription`).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-20/lr-20b/work/probes.py
Output: probes.out beside this file (written by redirecting stdout).

Every case that compiles more than one value of a `tl.constexpr` runs in a process of its own with an
empty TRITON_CACHE_DIR, so that no launch of another case shares a key with it. No path is printed.

T. A helper whose visited `return`s give equal tuples of compile-time values (SEMANTICS "The supported
   subset", `return`: "`return`s that all give one compile-time value, equal values of one Python type").
   `(1, 2) == (1.0, 2)` and both are of Python type `tuple`. `c` is a runtime `int` (5 or -5, never 1).
   - T1 `if c > 0: return (1, 2)` then `return (1.0, 2)`; the kernel stores `x * th(c)[0]` with the
     `int32` `x = 2^24 + 1`. The value of the `return` the program takes, as Python reads it, gives
     16777217 for c = 5 (integer product) and 16777216 for c = -5 (float32 product, stored as int32).
   - T2 the same through `a, b = th(c)` (a tuple assignment of a helper's tuple result) and `x * a`.
   - T3 the order reversed (`(1.0, 2)` first).
K. Scope 11, in one process on a fresh cache: `C = 2` then `C = 2.0` (Python's `==` equates them; the
   REVIEWER_GUIDE item 11 and the rule page's checklist list only `1`/`1.0`/`True` and `0`/`0.0`/`-0.0`/
   `False`). The kernel stores `x * C` with the `int32` `x = 2^24 + 1`: 33554434 for the integer `2`,
   33554432 for the float `2.0` (float32 product). Each value alone is the control.
B. A conditional expression whose runtime condition is a one-element block (SEMANTICS "Order": "a block
   `c` is rejected too"): `x + 1 if m else x - 1` with `m` a loaded `[1]` block of int32, and with a
   `[2]` block.
O. The scalar argument equal to 1 (TRANSCRIPTION's pitfall): `(-1 if c > 0 else -3) // 2` with c = 1.
"""
import json
import os
import subprocess
import sys
import tempfile

SRC = '''import triton
import triton.language as tl


@triton.jit
def th1(c):
    if c > 0:
        return (1, 2)
    return (1.0, 2)


@triton.jit
def th3(c):
    if c > 0:
        return (1.0, 2)
    return (1, 2)


@triton.jit
def t1(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * th1(c)[0])


@triton.jit
def t2(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    a, b = th1(c)
    tl.store(o_ptr + i, x * a)


@triton.jit
def t3(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * th3(c)[0])


@triton.jit
def k(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * C)


@triton.jit
def b1(o_ptr, x_ptr, m_ptr):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    m = tl.load(m_ptr + i)
    tl.store(o_ptr + i, x + 1 if m else x - 1)


@triton.jit
def b2(o_ptr, x_ptr, m_ptr):
    i = tl.arange(0, 2)
    x = tl.load(x_ptr + i)
    m = tl.load(m_ptr + i)
    tl.store(o_ptr + i, x + 1 if m else x - 1)


@triton.jit
def o1(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], (-1 if c > 0 else -3) // 2, tl.int32))
'''


def load_mod(tmp):
    import importlib.util
    path = os.path.join(tmp, "lr20b_mod.py")
    with open(path, "w") as fh:
        fh.write(SRC)
    spec = importlib.util.spec_from_file_location("lr20b_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def child(plan):
    import torch
    mod = load_mod(tempfile.mkdtemp(prefix="lr20b-"))
    dev = "cuda"
    out = []
    for step in plan:
        name, args = step[0], step[1:]
        kern = getattr(mod, name)
        o = torch.zeros(2, dtype=torch.int32, device=dev)
        x = torch.full((2,), 2 ** 24 + 1, dtype=torch.int32, device=dev)
        try:
            if name in ("t1", "t2", "t3"):
                kern[(1,)](o, x, args[0])
            elif name == "k":
                kern[(1,)](o, x, C=args[0])
            elif name in ("b1", "b2"):
                m = torch.ones(2, dtype=torch.int32, device=dev)
                x7 = torch.full((2,), 7, dtype=torch.int32, device=dev)
                kern[(1,)](o, x7, m)
            elif name == "o1":
                kern[(1,)](o, args[0])
            torch.cuda.synchronize()
            out.append(o[0].item() if name != "b2" else o.tolist())
        except Exception as ex:
            out.append(f"rejected on the host ({type(ex).__name__})")
    print(json.dumps(out))


def run(plan):
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr20b-cache-")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", json.dumps(plan)],
                       env=env, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("[")]
    if not lines:
        return [f"child failed (exit {r.returncode})"]
    return json.loads(lines[-1])


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    print("T. equal tuples of compile-time values returned on two paths (x = 2^24 + 1, int32)")
    for name, what, py in (("t1", "x * th(c)[0], (1, 2) then (1.0, 2)", (16777217, 16777216)),
                           ("t2", "a, b = th(c); x * a, (1, 2) then (1.0, 2)", (16777217, 16777216)),
                           ("t3", "x * th(c)[0], (1.0, 2) then (1, 2)", (16777216, 16777217))):
        r = run([[name, 5], [name, -5]])
        print(f"  {name} {what}: c = 5 -> {r[0]}, c = -5 -> {r[1]} "
              f"(the taken return's value as Python reads it: {py[0]}, {py[1]})")
    print("K. scope 11 in one process, fresh cache: C = 2 then C = 2.0 (x = 2^24 + 1, int32; x * C)")
    alone2 = run([["k", 2]])[0]
    alone2f = run([["k", 2.0]])[0]
    seq = run([["k", 2], ["k", 2.0]])
    rev = run([["k", 2.0], ["k", 2]])
    print(f"  alone: C = 2 -> {alone2}, C = 2.0 -> {alone2f}")
    print(f"  2 then 2.0: {seq[0]}, {seq[1]}{'  DIFFERS from 2.0 alone' if seq[1] != alone2f else ''}")
    print(f"  2.0 then 2: {rev[0]}, {rev[1]}{'  DIFFERS from 2 alone' if rev[1] != alone2 else ''}")
    print("B. a conditional expression whose runtime condition is a block (x = 7, m = 1)")
    print(f"  b1 [1] block condition: {run([['b1']])[0]}")
    print(f"  b2 [2] block condition: {run([['b2']])[0]}")
    print("O. the scalar argument equal to 1: (-1 if c > 0 else -3) // 2")
    r = run([["o1", 1], ["o1", 5]])
    print(f"  c = 1 -> {r[0]} (a compile-time condition: Python's -1 // 2 = -1); c = 5 -> {r[1]} (int32 //)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(json.loads(sys.argv[2]))
    else:
        main()
