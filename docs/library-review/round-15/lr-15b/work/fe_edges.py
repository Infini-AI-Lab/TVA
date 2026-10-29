"""lr-15b: edge cases of the front end around the supported subset (library round 15).

Each case is compiled and run on GPU 0 in its own process with its own empty Triton cache; a case
the front end refuses is reported with its error ("rejected on the host").

  K1  for i in range(0, n, step=2): s += 1          (a keyword argument to Python's range), n = 8
  K2  for i in range(n, step=2): s += 1             (one positional argument and a keyword), n = 8
  K3  for i in tl.range(0, n, step=2): s += 1       (control: tl.range takes step=), n = 8
  K4  for i in range(0, n, 2, 99): s += 1            (four positional arguments), n = 8
  A1  a: tl.int32 = tl.atomic_add(c, 1)              (an annotation other than tl.constexpr), 8 programs
  A2  c: tl.constexpr = 1 / 2 in both branches of a runtime if, then c used after the if
  A3  c: tl.constexpr = 4 in a for (range) body, used in the body (dry visit and real visit)
  S1  for j in tl.static_range(3): pass; then store j after the loop
  S2  for j in tl.static_range(3): t = j * 10; then store t after the loop
  H1  helper def h(x, mode="a"): ...; h(x, "b")        (a helper default that is a string)
  H2  helper def h(x, e=None): if e is None: ... ; h(x)
  W1  while i < n: if FLAG: return; i += 1            (FLAG a tl.constexpr True; return in a compile-time if)
  R1  for i in range(lo, hi) where lo, hi = tl.atomic_add(c, 1) * 0, 4   (two arguments: once)
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>   (Triton 3.8.0 from TVA's .venv)
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_range_kw(out_ptr, n, CASE: tl.constexpr):
    s = 0
    if CASE == 1:
        for i in range(0, n, step=2):
            s += 1
    elif CASE == 2:
        for i in range(n, step=2):
            s += 1
    elif CASE == 3:
        for i in tl.range(0, n, step=2):
            s += 1
    else:
        for i in range(0, n, 2, 99):
            s += 1
    tl.store(out_ptr, s)


@triton.jit
def k_ann_other(c_ptr, out_ptr):
    a: tl.int32 = tl.atomic_add(c_ptr, 1)
    tl.store(out_ptr + tl.program_id(0), a)


@triton.jit
def k_ann_if(out_ptr, n):
    if n > 0:
        c: tl.constexpr = 1
    else:
        c: tl.constexpr = 2
    tl.store(out_ptr, c)


@triton.jit
def k_ann_for(out_ptr, n):
    s = 0
    for i in range(0, n):
        c: tl.constexpr = 4
        s += c
    tl.store(out_ptr, s)


@triton.jit
def k_static_var(out_ptr, CASE: tl.constexpr):
    if CASE == 1:
        for j in tl.static_range(3):
            pass
        tl.store(out_ptr, j)
    else:
        for j in tl.static_range(3):
            t = j * 10
        tl.store(out_ptr, t)


@triton.jit
def h_str(x, mode="a"):
    if mode == "a":
        return x + 1
    else:
        return x + 2


@triton.jit
def h_none(x, e=None):
    if e is None:
        return x + 3
    else:
        return x + e


@triton.jit
def k_helper_default(out_ptr, CASE: tl.constexpr):
    x = tl.program_id(0) * 0
    if CASE == 1:
        y = h_str(x, "b")
    else:
        y = h_none(x)
    tl.store(out_ptr, y)


@triton.jit
def k_while_ret(out_ptr, n, FLAG: tl.constexpr):
    i = 0
    while i < n:
        if FLAG:
            return
        i += 1
    tl.store(out_ptr, i)


@triton.jit
def k_range2(c_ptr, out_ptr):
    s = 0
    for i in range(tl.atomic_add(c_ptr, 1) * 0, 4):
        s += 1
    tl.store(out_ptr + tl.program_id(0), s)


CASES = {
    "K1": "for i in range(0, n, step=2), n = 8: trips",
    "K2": "for i in range(n, step=2), n = 8: trips",
    "K3": "for i in tl.range(0, n, step=2), n = 8: trips (control)",
    "K4": "for i in range(0, n, 2, 99), n = 8: trips (four positional arguments)",
    "A1": "a: tl.int32 = tl.atomic_add(c, 1), 8 programs: atomics",
    "A2": "c: tl.constexpr = 1 / 2 in the branches of a runtime if; c after the if",
    "A3": "c: tl.constexpr = 4 in a range(0, n) body, n = 3; s += c",
    "S1": "for j in tl.static_range(3): pass; store j after the loop",
    "S2": "for j in tl.static_range(3): t = j * 10; store t after the loop",
    "H1": "def h(x, mode='a'); h(x, 'b'), x = 0",
    "H2": "def h(x, e=None); h(x), x = 0",
    "W1": "i = 0; while i < n: if FLAG (constexpr True): return; i += 1; store i (n = 4)",
    "R1": "for i in range(tl.atomic_add(c, 1) * 0, 4), 8 programs: atomics",
}


def run_case(name):
    dev = "cuda"
    if name in ("K1", "K2", "K3", "K4"):
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_range_kw[(1,)](out, 8, CASE=int(name[1]))
        torch.cuda.synchronize()
        return f"trips {out.item()}"
    if name == "A1":
        c = torch.zeros(1, device=dev, dtype=torch.int32)
        out = torch.full((8,), -1, device=dev, dtype=torch.int32)
        k_ann_other[(8,)](c, out)
        torch.cuda.synchronize()
        return f"atomics {c.item()} in 8 programs; a = {sorted(out.tolist())}"
    if name == "A2":
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_ann_if[(1,)](out, 5)
        torch.cuda.synchronize()
        return f"compiled and ran; stored {out.item()}"
    if name == "A3":
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_ann_for[(1,)](out, 3)
        torch.cuda.synchronize()
        return f"compiled and ran; stored {out.item()}"
    if name in ("S1", "S2"):
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_static_var[(1,)](out, CASE=int(name[1]))
        torch.cuda.synchronize()
        return f"compiled and ran; stored {out.item()}"
    if name in ("H1", "H2"):
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_helper_default[(1,)](out, CASE=int(name[1]))
        torch.cuda.synchronize()
        return f"compiled and ran; stored {out.item()}"
    if name == "W1":
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_while_ret[(1,)](out, 4, FLAG=True)
        torch.cuda.synchronize()
        return f"compiled and ran; stored {out.item()}"
    if name == "R1":
        c = torch.zeros(1, device=dev, dtype=torch.int32)
        out = torch.full((8,), -1, device=dev, dtype=torch.int32)
        k_range2[(8,)](c, out)
        torch.cuda.synchronize()
        return f"atomics {c.item()} in 8 programs; trips {out.tolist()}"
    raise SystemExit("unknown case " + name)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:200]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = f"process failed (exit {p.returncode}): " + (err[-1][:200] if err else "(no output)")
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()
