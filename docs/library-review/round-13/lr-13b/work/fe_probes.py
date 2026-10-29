"""lr-13b, round 13, survey area `transcription`: probes of the front-end facts that
docs/TRANSCRIPTION.md's patterns and pitfalls rest on (docs/SEMANTICS.md, "How the front end
evaluates the source", "Constants", "Operators and promotion").

Each case is compiled and run on GPU 0 in its own process; a case the front end refuses is reported
with the first line of its error.

  W1   tl.where(keep, x, 0.0), keep an int32 mask, x an int32 tensor: result dtype and values
       (TRANSCRIPTION section 2 row "tl.where(keep, x, 0.0) with keep an int32 tensor")
  W2   the same with x a float32 tensor (tutorial 04's use; control)
  C1   BLOCK: tl.constexpr parameter, then BLOCK = BLOCK * 1, then tl.arange(0, BLOCK)
  C2   c: tl.constexpr = -7; c += 0; c // 2           (reassigning an annotated name)
  C3   c: tl.constexpr = -7; c // 2                     (control: -4)
  C4   e: tl.constexpr (a bare annotation); e = 1e-7; x * e, x bfloat16
  C5   e = 1e-7; x * e, x bfloat16                      (control: float32 product)
  C6   a: tl.int64 = 2**31 - 1; b = a + 1 (an annotation other than tl.constexpr), stored int64
  C7   a = 2**31 - 1; b = a.to(tl.int64) + 1 (control: 2**31)
  D1   a helper h(x, e=1e-7) returning x * e, called h(x), x bfloat16 (a default argument)
  D2   h(x, 1e-7) (control: a literal passed directly)
  G1   a module global E = tl.constexpr(1e-7); x * E, x bfloat16
  G2   a module global E2 = 1e-7 (not constexpr) read in the kernel
  R1   if FLAG: return (FLAG a constexpr, True), then a store: does the store run?
  L1   for i in range(0, n): tl.atomic_add(c, 1), n = 0 at run time (the body's dry visit)
  L2   the same with n = 1 (one atomic per program, not two)
  L3   i = 0; while i < n: tl.atomic_add(c, 1); i += 1, n = 0 (the while body's dry visit)
  L4   for i in range(tl.atomic_add(c, 1) * 0): ... (an operation in range's single argument), 8 programs
  L5   for i in range(0, tl.atomic_add(c, 1) * 0): ... (the same operation as the upper of two bounds)
  L6   for i in tl.range(tl.atomic_add(c, 1) * 0): ... (tl.range with one argument)
  O1   tl.load(p, mask=all false, other=1 + 2**-11 + 2**-30), p a float16 pointer: other's bits
       (float32 first, then float16: two roundings give 1.0 = 0x3c00; one rounding gives 0x3c01)
  F1   a @triton.constexpr_function f(n) = 2 * n, then tl.arange(0, f(HALF)) (a compile-time value)
  F2   a @triton.constexpr_function called on a tensor argument
"""
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import triton
import triton.language as tl

E = tl.constexpr(1e-7)
E2 = 1e-7


@triton.jit
def k_where(m_ptr, x_ptr, out_ptr, FLOAT: tl.constexpr):
    i = tl.arange(0, 4)
    keep = tl.load(m_ptr + i)
    x = tl.load(x_ptr + i)
    y = tl.where(keep, x, 0.0)
    if FLOAT:
        tl.static_assert(y.dtype == tl.float32)
    else:
        tl.static_print("W1 dtype", y.dtype)
    tl.store(out_ptr + i, y)


@triton.jit
def k_c1(out_ptr, BLOCK: tl.constexpr):
    BLOCK = BLOCK * 1
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, i)


@triton.jit
def k_c2(out_ptr, CASE: tl.constexpr):
    c: tl.constexpr = -7
    if CASE == 2:
        c += 0
    y = c // 2
    tl.store(out_ptr, y)


@triton.jit
def k_c4(x_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    if CASE == 4:
        e: tl.constexpr
        e = 1e-7
        y = x * e
    else:
        e = 1e-7
        y = x * e
    tl.static_print("C4/C5 dtype", y.dtype)
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def k_c6(out_ptr, CASE: tl.constexpr):
    if CASE == 6:
        a: tl.int64 = 2147483647
        tl.static_print("C6 a dtype", a.dtype)
        b = a + 1
    else:
        a = 2147483647
        b = a.to(tl.int64) + 1
    tl.store(out_ptr, b.to(tl.int64))


@triton.jit
def h_mul(x, e=1e-7):
    return x * e


@triton.jit
def k_d(x_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    if CASE == 1:
        y = h_mul(x)
    else:
        y = h_mul(x, 1e-7)
    tl.static_print("D dtype", y.dtype)
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def k_g1(x_ptr, out_ptr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    y = x * E
    tl.static_print("G1 dtype", y.dtype)
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def k_g2(x_ptr, out_ptr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    y = x * E2
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def k_r1(out_ptr, FLAG: tl.constexpr):
    tl.store(out_ptr, 1)
    if FLAG:
        return
    tl.store(out_ptr + 1, 1)


@triton.jit
def k_l(c_ptr, n):
    for i in range(0, n):
        tl.atomic_add(c_ptr, 1)


@triton.jit
def k_l3(c_ptr, n):
    i = 0
    while i < n:
        tl.atomic_add(c_ptr, 1)
        i += 1


@triton.jit
def k_l4(c_ptr, d_ptr, CASE: tl.constexpr):
    if CASE == 4:
        for i in range(tl.atomic_add(c_ptr, 1) * 0):
            tl.store(d_ptr, i)
    elif CASE == 5:
        for i in range(0, tl.atomic_add(c_ptr, 1) * 0):
            tl.store(d_ptr, i)
    else:
        for i in tl.range(tl.atomic_add(c_ptr, 1) * 0):
            tl.store(d_ptr, i)


@triton.jit
def k_o1(p_ptr, out_ptr):
    i = tl.arange(0, 4)
    x = tl.load(p_ptr + i, mask=i < 0, other=1.0004882821813226)
    tl.store(out_ptr + i, x)


@triton.constexpr_function
def twice(n):
    return 2 * n


@triton.jit
def k_f1(out_ptr, HALF: tl.constexpr):
    i = tl.arange(0, twice(HALF))
    tl.store(out_ptr + i, i)


@triton.jit
def k_f2(out_ptr):
    p = tl.program_id(0)
    y = twice(p)
    tl.store(out_ptr, y)


def bits(t):
    return [hex(v) for v in t.view(torch.int32).tolist()]


def run(case):
    dev = "cuda"
    if case in ("W1", "W2"):
        m = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device=dev)
        if case == "W1":
            x = torch.tensor([16777217, 5, -3, 2**31 - 1], dtype=torch.int32, device=dev)
            out = torch.zeros(4, dtype=torch.int32, device=dev)
        else:
            x = torch.tensor([1.5, 5.0, -3.0, 0.1], dtype=torch.float32, device=dev)
            out = torch.zeros(4, dtype=torch.float32, device=dev)
        k_where[(1,)](m, x, out, FLOAT=(case == "W2"))
        print(case, "stored", out.tolist())
    elif case == "C1":
        out = torch.zeros(8, dtype=torch.int32, device=dev)
        k_c1[(1,)](out, BLOCK=8)
        print(case, "stored", out.tolist())
    elif case in ("C2", "C3"):
        out = torch.zeros(1, dtype=torch.int32, device=dev)
        k_c2[(1,)](out, CASE=2 if case == "C2" else 3)
        print(case, "stored", out.tolist())
    elif case in ("C4", "C5"):
        x = torch.tensor([1, 2, 3, 1000], dtype=torch.bfloat16, device=dev)
        out = torch.zeros(4, dtype=torch.float32, device=dev)
        k_c4[(1,)](x, out, CASE=4 if case == "C4" else 5)
        print(case, "stored", out.tolist())
    elif case in ("C6", "C7"):
        out = torch.zeros(1, dtype=torch.int64, device=dev)
        k_c6[(1,)](out, CASE=6 if case == "C6" else 7)
        print(case, "stored", out.tolist())
    elif case in ("D1", "D2"):
        x = torch.tensor([1, 2, 3, 1000], dtype=torch.bfloat16, device=dev)
        out = torch.zeros(4, dtype=torch.float32, device=dev)
        k_d[(1,)](x, out, CASE=1 if case == "D1" else 2)
        print(case, "stored", out.tolist())
    elif case in ("G1", "G2"):
        x = torch.tensor([1, 2, 3, 1000], dtype=torch.bfloat16, device=dev)
        out = torch.zeros(4, dtype=torch.float32, device=dev)
        (k_g1 if case == "G1" else k_g2)[(1,)](x, out)
        print(case, "stored", out.tolist())
    elif case == "R1":
        out = torch.zeros(2, dtype=torch.int32, device=dev)
        k_r1[(1,)](out, FLAG=True)
        print(case, "stored", out.tolist())
    elif case in ("L1", "L2"):
        c = torch.zeros(1, dtype=torch.int32, device=dev)
        k_l[(8,)](c, 0 if case == "L1" else 1)
        print(case, "counter after 8 programs", c.tolist())
    elif case == "L3":
        c = torch.zeros(1, dtype=torch.int32, device=dev)
        k_l3[(8,)](c, 0)
        print(case, "counter after 8 programs", c.tolist())
    elif case in ("L4", "L5", "L6"):
        c = torch.zeros(1, dtype=torch.int32, device=dev)
        d = torch.zeros(1, dtype=torch.int32, device=dev)
        k_l4[(8,)](c, d, CASE=int(case[1]))
        print(case, "counter after 8 programs", c.tolist())
    elif case == "O1":
        p = torch.zeros(4, dtype=torch.float16, device=dev)
        out = torch.zeros(4, dtype=torch.float16, device=dev)
        k_o1[(1,)](p, out)
        print(case, "other's bits", [hex(v & 0xFFFF) for v in out.view(torch.int16).tolist()])
    elif case == "F1":
        out = torch.zeros(8, dtype=torch.int32, device=dev)
        k_f1[(1,)](out, HALF=4)
        print(case, "stored", out.tolist())
    elif case == "F2":
        out = torch.zeros(1, dtype=torch.int32, device=dev)
        k_f2[(1,)](out)
        print(case, "stored", out.tolist())
    torch.cuda.synchronize()


CASES = ["W1", "W2", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "D1", "D2", "G1", "G2", "R1", "L1", "L2",
         "L3", "L4", "L5", "L6", "O1", "F1", "F2"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
        sys.exit(0)
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    cache = tempfile.mkdtemp()  # a fresh Triton cache, so that every case compiles (and prints its dtypes)
    env = dict(os.environ, TRITON_CACHE_DIR=cache)
    for case in CASES:
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), case], capture_output=True, text=True,
                           env=env)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        if r.returncode:
            err = [l for l in r.stderr.splitlines() if l.strip()]
            # the root cause: the first "<Name>Error: …" line of the traceback chain
            def head(l):
                return l.replace("(", ":").split(":")[0]
            msg = next((l for l in err if head(l).endswith("Error") and " " not in head(l)
                        and "CompilationError" not in head(l)), err[-1] if err else "?")
            print(f"{case}: rejected on the host: {msg.strip()[:240]}")
            for l in lines:
                print(f"  {l}")
        else:
            for l in lines:
                print(f"{case}: {l}")
    shutil.rmtree(cache, ignore_errors=True)
