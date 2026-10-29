"""Python numbers in a kernel: assigned to a name, and given to Python's `min` and `max`.

Library evidence for library issues L59 and L61 (round 11).

L59. `visit_Assign` (`primitives/triton-3.8.0-code-generator.py` lines 727-760) turns a Python `int`,
`float` or `bool` that a statement assigns to a name (a literal, the value of a `tl.constexpr`, an
element of a tuple) into a scalar tensor of its own type (`to_tensor`, `semantic.py` lines 118-158),
unless the target is annotated `: tl.constexpr`; `None`, dtypes and tuples are kept, and another
Python value is refused. A Python number passed directly to a helper stays a compile-time value there
(`call_JitFunction`, `normalize_value`, lines 95-100, 1358-1365). So a name that holds an assigned
number is a tensor, not a Python number, in every rule that depends on it.

L61. Python's `min` and `max` of compile-time values refuse NaN and `-0.0`
(`primitives/triton-3.8.0-core.py` lines 3882-3888, 3906-3912).

Each case is compiled and run on GPU 0 in its own process; a case that the front end refuses is
reported with its error. "in place" is the same expression with the number written where it is used.

  A1   e = 1e-7; y = x * e (x a bfloat16 block)       vs  x * 1e-7 in place
  A2   e = 1e-300; y = x * e (x a float32 block)      vs  x * 1e-300 in place   (the result's dtype)
  A3   a = -7; b = 2; a // b                            vs  -7 // 2 in place
  A4   lo, hi = -7, 2; lo // hi (a tuple)               vs  the annotated  q: tl.constexpr = -7 // 2
  A5   k = 0; k -= 7; k // 2 (an augmented assignment)
  A6   for j in tl.static_range(1): m = j - 7; m // 2   (j is a Python integer; m is assigned)
  A7   q: tl.constexpr = -7; q // 2 (annotated)
  A8   h(-7) with h(n) returning n // 2 (a literal passed to a helper)  vs  a = -7; h(a)
  A9   F = False; if F and tl.atomic_add(c, 1) > 0: ...   vs  False and ... in place (8 programs)
  A10  F = False; if F: tl.static_assert(False)          vs  if False: tl.static_assert(False)
  A11  B = BLOCK (a tl.constexpr parameter); tl.arange(0, B)   vs  B: tl.constexpr = BLOCK
  A12  z = -0.0; store z (a float32 buffer)
  A13  n = None; if n is None: ...; d = tl.float16; tl.full(…, 1, d)   (None and dtypes are kept)
  A14  s = "abc" (a string assigned to a name)
  B1   max(1.0, F) with the constexpr F = nan          B2  min(1.0, -0.0) (literals)
  B3   max(1.0, 2.0) (control)                         B4  a = 3; min(a, 5) (a is a tensor: tl.minimum)

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/python-numbers/run.py  (from TVA's root)
"""
import json, os, struct, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k_mul(x_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    if CASE == 1:
        e = 1e-7
        y = x * e
    elif CASE == 2:
        y = x * 1e-7
    elif CASE == 3:
        e = 1e-300
        y = x * e
        tl.static_assert(y.dtype == tl.float64)
    else:
        y = x * 1e-300
        tl.static_assert(y.dtype == tl.float32)
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def half(n):
    return n // 2


@triton.jit
def k_int(out_ptr, CASE: tl.constexpr):
    if CASE == 1:
        a = -7
        b = 2
        q = a // b
    elif CASE == 2:
        q = -7 // 2
    elif CASE == 3:
        lo, hi = -7, 2
        q = lo // hi
    elif CASE == 4:
        k = 0
        k -= 7
        q = k // 2
    elif CASE == 5:
        q = 0
        for j in tl.static_range(1):
            m = j - 7
            q = m // 2
    elif CASE == 6:
        c: tl.constexpr = -7
        q = c // 2
    elif CASE == 7:
        q = half(-7)
    else:
        a = -7
        q = half(a)
    tl.store(out_ptr, q)


@triton.jit
def k_and(c_ptr, CASE: tl.constexpr):
    if CASE == 1:
        F = False
        if F and tl.atomic_add(c_ptr, 1) > 0:
            tl.atomic_add(c_ptr + 1, 1)
    else:
        if False and tl.atomic_add(c_ptr, 1) > 0:
            tl.atomic_add(c_ptr + 1, 1)


@triton.jit
def k_static(out_ptr, CASE: tl.constexpr):
    if CASE == 1:
        F = False
        if F:
            tl.static_assert(False)
    else:
        if False:
            tl.static_assert(False)
    tl.store(out_ptr, 1)


@triton.jit
def k_block(out_ptr, BLOCK: tl.constexpr, CASE: tl.constexpr):
    if CASE == 1:
        B = BLOCK
        tl.store(out_ptr + tl.arange(0, B), 1)
    else:
        B: tl.constexpr = BLOCK
        tl.store(out_ptr + tl.arange(0, B), 1)


@triton.jit
def k_misc(out_ptr, CASE: tl.constexpr):
    if CASE == 1:
        z = -0.0
        tl.store(out_ptr, z)
    elif CASE == 2:
        n = None
        d = tl.float16
        if n is None:
            tl.store(out_ptr + tl.arange(0, 1), tl.full([1], 1, d).to(tl.float32))
    else:
        s = "abc"
        tl.store(out_ptr, 1.0)


@triton.jit
def k_minmax(out_ptr, F: tl.constexpr, CASE: tl.constexpr):
    if CASE == 1:
        tl.store(out_ptr, max(1.0, F))
    elif CASE == 2:
        tl.store(out_ptr, min(1.0, -0.0))
    elif CASE == 3:
        tl.store(out_ptr, max(1.0, 2.0))
    else:
        a = 3
        m = min(a, 5)
        tl.static_assert(m.dtype == tl.int32)
        tl.store(out_ptr, m.to(tl.float32))


CASES = {
    "A1": "e = 1e-7; x * e, x bfloat16 [1, 2, 3, 1000]",
    "A1p": "x * 1e-7 in place",
    "A2": "e = 1e-300; x * e, x float32: dtype float64 (static_assert), value",
    "A2p": "x * 1e-300 in place: dtype float32 (static_assert), value",
    "A3": "a = -7; b = 2; a // b",
    "A3p": "-7 // 2 in place",
    "A4": "lo, hi = -7, 2; lo // hi",
    "A5": "k = 0; k -= 7; k // 2",
    "A6": "static_range: m = j - 7; m // 2 (j = 0)",
    "A7": "c: tl.constexpr = -7; c // 2",
    "A8": "half(-7), a literal passed to a helper",
    "A8a": "a = -7; half(a)",
    "A9": "F = False; if F and tl.atomic_add(c, 1) > 0 (8 programs): count of atomics",
    "A9p": "if False and tl.atomic_add(c, 1) > 0 in place: count of atomics",
    "A10": "F = False; if F: tl.static_assert(False)",
    "A10p": "if False: tl.static_assert(False) in place",
    "A11": "B = BLOCK; tl.arange(0, B)",
    "A11c": "B: tl.constexpr = BLOCK; tl.arange(0, B)",
    "A12": "z = -0.0; store z: the bits stored",
    "A13": "n = None; d = tl.float16; if n is None: store tl.full([1], 1, d)",
    "A14": 's = "abc"',
    "B1": "max(1.0, F) with the constexpr F = nan",
    "B2": "min(1.0, -0.0)",
    "B3": "max(1.0, 2.0) (control)",
    "B4": "a = 3; min(a, 5): an int32 tensor",
}


def run_case(name):
    dev = "cuda"
    if name in ("A1", "A1p", "A2", "A2p"):
        if name.startswith("A1"):
            x = torch.tensor([1.0, 2.0, 3.0, 1000.0], device=dev, dtype=torch.bfloat16)
        else:
            x = torch.tensor([1.0, 2.0, 3.0, 1000.0], device=dev, dtype=torch.float32)
        out = torch.zeros(4, device=dev)
        case = {"A1": 1, "A1p": 2, "A2": 3, "A2p": 4}[name]
        k_mul[(1,)](x, out, CASE=case)
        torch.cuda.synchronize()
        return ("the static_assert on y.dtype held; " if name.startswith("A2") else "") + "stored " + str(out.tolist())
    if name in ("A3", "A3p", "A4", "A5", "A6", "A7", "A8", "A8a"):
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        case = {"A3": 1, "A3p": 2, "A4": 3, "A5": 4, "A6": 5, "A7": 6, "A8": 7, "A8a": 8}[name]
        k_int[(1,)](out, CASE=case)
        torch.cuda.synchronize()
        return "stored " + str(out.item())
    if name in ("A9", "A9p"):
        c = torch.zeros(2, device=dev, dtype=torch.int32)
        k_and[(8,)](c, CASE=1 if name == "A9" else 2)
        torch.cuda.synchronize()
        return f"atomics in the condition: {c[0].item()}, in the branch: {c[1].item()}"
    if name in ("A10", "A10p"):
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_static[(1,)](out, CASE=1 if name == "A10" else 2)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.item())
    if name in ("A11", "A11c"):
        out = torch.zeros(8, device=dev, dtype=torch.int32)
        k_block[(1,)](out, BLOCK=8, CASE=1 if name == "A11" else 2)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.tolist())
    if name in ("A12", "A13", "A14"):
        out = torch.full((1,), 7.0, device=dev)
        k_misc[(1,)](out, CASE={"A12": 1, "A13": 2, "A14": 3}[name])
        torch.cuda.synchronize()
        bits = struct.unpack("<I", struct.pack("<f", out.item()))[0]
        return f"stored {out.item()!r} (bits 0x{bits:08x})"
    if name.startswith("B"):
        out = torch.zeros(1, device=dev)
        case = int(name[1:])
        k_minmax[(1,)](out, F=float("nan"), CASE=case)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.item())
    raise SystemExit("unknown case " + name)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = str(e).strip().splitlines()
            # the front end's own message is the last line of a CompilationError
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1] if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = "process failed: " + (err[-1][:160] if err else "(no output)")
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()
