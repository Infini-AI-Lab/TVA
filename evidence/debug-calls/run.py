"""TVA library evidence: what `tl.static_assert`, `tl.static_print`, `tl.device_print`, `tl.device_assert`,
Python `assert` and Python `print` leave in a kernel compiled without `debug`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/debug-calls/run.py
Output: evidence/debug-calls/out.txt (README.md says what it shows).

Part P compiles a kernel with and without such calls on pure arguments and compares its TTIR and PTX,
with source locations and the kernel's name removed. Part E puts an operation with an effect (an
atomic increment of a counter, directly or in a `@triton.jit` helper) into an argument and counts how
many programs ran it. Part V shows where the front end evaluates a `tl.static_assert`.
"""
import os
import re
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
B = 64
N = 200


@triton.jit
def base(x_ptr, out_ptr, n, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(out_ptr + offs, x + 1.0, mask=m)


@triton.jit
def pure_checks(x_ptr, out_ptr, n, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    m = offs < n
    tl.static_assert(B % 16 == 0, "B is a multiple of 16")
    tl.static_assert(x_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(offs.shape[0] == B)
    tl.static_assert(offs is not None)
    tl.static_print("offs", offs, B)
    tl.device_assert(offs >= 0, "offsets are nonnegative")
    tl.device_assert(offs < n + B, "offsets are below n + B", mask=m)
    assert pid * B >= 0, "pid * B is nonnegative"
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(out_ptr + offs, x + 1.0, mask=m)


@triton.jit
def pure_prints(x_ptr, out_ptr, n, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    m = offs < n
    tl.device_print("n", n)
    print("pid", pid)
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(out_ptr + offs, x + 1.0, mask=m)


@triton.jit
def bump(c_ptr):
    tl.atomic_add(c_ptr, 1)


@triton.jit
def bump_ret(c_ptr):
    return tl.atomic_add(c_ptr, 1)


@triton.jit
def e1(c_ptr):
    tl.static_assert(tl.atomic_add(c_ptr, 1) is not None)


@triton.jit
def e2(c_ptr):
    tl.static_assert(bump(c_ptr) is None)


@triton.jit
def e3(c_ptr):
    tl.static_print("old", tl.atomic_add(c_ptr, 1))


@triton.jit
def e4(c_ptr):
    tl.device_print("old", tl.atomic_add(c_ptr, 1))


@triton.jit
def e5(c_ptr):
    tl.device_assert(tl.atomic_add(c_ptr, 1) >= 0, "the old count is nonnegative")


@triton.jit
def e6(c_ptr):
    assert bump_ret(c_ptr) >= 0, "the old count is nonnegative"


@triton.jit
def e7(c_ptr):
    print("old", tl.atomic_add(c_ptr, 1))


@triton.jit
def e8(c_ptr):
    tl.static_assert(True, tl.atomic_add(c_ptr, 1))


@triton.jit
def e9(c_ptr):
    tl.device_assert(c_ptr is not None, "pointer", mask=tl.atomic_add(c_ptr, 1) >= 0)


@triton.jit
def v1(c_ptr, B: tl.constexpr):
    if tl.program_id(0) < 0:
        tl.static_assert(B == 3, "inside a runtime branch no program takes")
    tl.atomic_add(c_ptr, 1)


@triton.jit
def v2(c_ptr, B: tl.constexpr):
    for i in tl.static_range(3):
        tl.static_assert(i < 2, "in the unrolled iteration i = 2")
    tl.atomic_add(c_ptr, 1)


@triton.jit
def v3(c_ptr, n, B: tl.constexpr):
    for i in range(0, n):
        tl.static_assert(B == 3, "inside a loop body that runs zero times")
    tl.atomic_add(c_ptr, 1)


@triton.jit
def v4(c_ptr, B: tl.constexpr):
    if B == 3:
        tl.static_assert(B == 4, "inside a constexpr branch the configuration does not take")
    tl.atomic_add(c_ptr, 1)


def normal(text, kind):
    out = []
    for line in text.splitlines():
        if kind == "ttir":
            if line.startswith("#loc"):
                continue
            line = re.sub(r"\s*loc\([^()]*(\([^()]*\))?[^()]*\)", "", line)
        else:
            if re.match(r"\s*\.(loc|file)\b", line) or line.strip().startswith("//"):
                continue
        out.append(line.replace("pure_checks", "K").replace("pure_prints", "K").replace("base", "K"))
    return "\n".join(l for l in out if l.strip())


def compiled(kern):
    x = torch.randn(N, device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float32)
    ck = kern[(triton.cdiv(N, B),)](x, out, N, B)
    torch.cuda.synchronize()
    return ck, bool(torch.equal(out, x + 1.0))


def run_count(name, kern, grid=4):
    c = torch.zeros(1, device=DEV, dtype=torch.int32)
    try:
        kern[(grid,)](c)
        torch.cuda.synchronize()
        res = f"compiles; counter = {int(c.item())} after {grid} programs"
    except Exception as e:  # noqa: BLE001
        lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
        res = f"{type(e).__name__}: {lines[-1][:120] if lines else ''}"
    print(f"{name}\n    {res}", flush=True)


def run_visit(name, launch):
    c = torch.zeros(1, device=DEV, dtype=torch.int32)
    try:
        launch(c)
        torch.cuda.synchronize()
        res = f"compiles; counter = {int(c.item())}"
    except Exception as e:  # noqa: BLE001
        lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
        res = f"{type(e).__name__}: {lines[-1][:120] if lines else ''}"
    print(f"{name}\n    {res}", flush=True)


def main():
    # Device printf and compile-time prints go to file descriptor 1. Send them to /dev/null, and print
    # this report on a copy of the original descriptor, so that the output holds only the report.
    global print
    report = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    print = lambda *a, **k: (report.write(" ".join(map(str, a)) + k.get("end", "\n")), report.flush())  # noqa: E731
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}, "
          f"TRITON_DEBUG={os.environ.get('TRITON_DEBUG')!r}", flush=True)

    print("\n== P. Pure arguments ==", flush=True)
    k0, ok0 = compiled(base)
    k1, ok1 = compiled(pure_checks)
    k2, ok2 = compiled(pure_prints)
    torch.cuda.synchronize()
    for kind in ("ttir", "ptx"):
        same = normal(k0.asm[kind], kind) == normal(k1.asm[kind], kind)
        print(f"P1 static_assert x4, static_print, device_assert x2, assert: {kind.upper()} identical to the kernel "
              f"without them: {same}", flush=True)
    print(f"P1 outputs correct: without {ok0}, with {ok1}", flush=True)
    ptx2 = k2.asm["ptx"]
    print(f"P2 device_print and print of pure values: PTX calls vprintf: {'vprintf' in ptx2}; "
          f"TTIR has tt.print: {'tt.print' in k2.asm['ttir']}; outputs correct: {ok2}", flush=True)
    stripped = [l for l in normal(k2.asm["ttir"], "ttir").splitlines() if "tt.print" not in l]
    print(f"P2 TTIR without its tt.print lines identical to the kernel without them: "
          f"{chr(10).join(stripped) == normal(k0.asm['ttir'], 'ttir')}", flush=True)

    print("\n== E. Operations with an effect inside the arguments (grid of 4 programs) ==", flush=True)
    run_count("E1 tl.static_assert(tl.atomic_add(c, 1) is not None)", e1)
    run_count("E2 tl.static_assert(bump(c) is None), bump a @triton.jit helper that adds 1", e2)
    run_count("E3 tl.static_print('old', tl.atomic_add(c, 1))", e3)
    run_count("E4 tl.device_print('old', tl.atomic_add(c, 1))", e4)
    run_count("E5 tl.device_assert(tl.atomic_add(c, 1) >= 0, ...)", e5)
    run_count("E6 assert bump_ret(c) >= 0, ... (a Python assert; bump_ret returns tl.atomic_add(c, 1))", e6)
    run_count("E7 print('old', tl.atomic_add(c, 1)) (Python print)", e7)
    run_count("E8 tl.static_assert(True, tl.atomic_add(c, 1)): an atomic in the message of a holding assertion", e8)
    run_count("E9 tl.device_assert(c is not None, ..., mask=tl.atomic_add(c, 1) >= 0)", e9)

    print("\n== V. Where the front end evaluates tl.static_assert (B = 16) ==", flush=True)
    run_visit("V1 in a runtime `if` branch that no program takes", lambda c: v1[(1,)](c, 16))
    run_visit("V2 in a tl.static_range(3) loop, assertion i < 2", lambda c: v2[(1,)](c, 16))
    run_visit("V3 in a `for` body that runs zero times (n = 0)", lambda c: v3[(1,)](c, 0, 16))
    run_visit("V4 in a constexpr `if` branch that is not taken (B == 3 is false)", lambda c: v4[(1,)](c, 16))


if __name__ == "__main__":
    main()
