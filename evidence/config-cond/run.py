"""TVA library evidence (library issue L25): the condition of a branch the configuration decides, and
which operations of an expression the front end evaluates.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/config-cond/run.py
Output: evidence/config-cond/out.txt (README.md says what it shows).

Each kernel has an `if`, a conditional expression or a call whose condition or argument holds
`tl.atomic_add(cnt_ptr, 1)` (or `2`). Grid of 4 programs; the counter shows how many atomics ran,
the outputs which branch was taken, and the TTIR whether the branch is static (no `scf.if`).

Round 4:
- C1 `if <atomic> is not None:` (an operation inside a constexpr condition)
- C2 `v = 1 if <atomic> is None else 2`
- C3 `elif <atomic> is not None:` after a false `if FLAG:` (the `elif` condition)
- C4 `if FLAG and <atomic> is not None:` with `FLAG = False` (`and` on constexprs)
- P1 `if FLAG:`, P2 `if N > 4:`, P3 `v = 1 if FLAG else 2` (pure conditions: controls)
Round 5 (L25 reopened): what the front end evaluates, by `visit_BoolOp`, `visit_IfExp` and `visit_If`
- S1 `if FLAG and <atomic> is not None:` with `FLAG = True` (a compile-time true operand of `and`)
- S2, S3 `if FLAG or <atomic> is None:` with `FLAG = True`, `False`
- S4 `if (<atomic> is None) and (<atomic> is None):` (the first operand, a compile-time `False`, ends it)
- S5 `if (<atomic> is not None) or (<atomic> is None):` (the first operand, a compile-time `True`, ends it)
- S6, S7 `v = 1 if (<atomic> if FLAG else None) is None else 2` with `FLAG = False`, `True`
- S8 `if FLAG: … elif <atomic> is not None:` with `FLAG = True` (an `elif` after a taken `if`)
- S9 `if (tl.load(x) > 0) and (<atomic> > 0):` with `x = 0` (a tensor operand does not end `and`)
- S10 `tl.static_assert(FLAG or <atomic> is None)` with `FLAG = True`
- S11 `tl.static_print(FLAG and <atomic>)` with `FLAG = False`
- S12, S13 `tl.static_print(<atomic> if FLAG else 0)` with `FLAG = False`, `True`
- S14 `tl.device_print("v", <atomic> if FLAG else 0)` with `FLAG = False`
- S15 `if (FLAG and <atomic +1> is None) or <atomic +2> is not None:` with `FLAG = False` (nesting)
- S16 `if <atomic> is not None and FLAG:` with `FLAG = False` (the operation before the deciding operand)
"""
import ctypes
import os
import sys
import tempfile

# a fresh compile cache, so that every kernel compiles here and `tl.static_print` prints
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="tva-config-cond-")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def c1(cnt_ptr, o_ptr, x_ptr):
    if tl.atomic_add(cnt_ptr, 1) is not None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def c2(cnt_ptr, o_ptr, x_ptr):
    v = 1 if tl.atomic_add(cnt_ptr, 1) is None else 2
    tl.store(o_ptr + tl.program_id(0), v)


@triton.jit
def c3(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    if FLAG:
        tl.store(o_ptr + tl.program_id(0), 1)
    elif tl.atomic_add(cnt_ptr, 1) is not None:
        tl.store(o_ptr + tl.program_id(0), 2)
    else:
        tl.store(o_ptr + tl.program_id(0), 3)


@triton.jit
def c4(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    if FLAG and tl.atomic_add(cnt_ptr, 1) is not None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def p1(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    if FLAG:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def p2(cnt_ptr, o_ptr, x_ptr, N: tl.constexpr):
    if N > 4:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def p3(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    v = 1 if FLAG else 2
    tl.store(o_ptr + tl.program_id(0), v)


@triton.jit
def s_or(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    if FLAG or tl.atomic_add(cnt_ptr, 1) is None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def s4(cnt_ptr, o_ptr, x_ptr):
    if (tl.atomic_add(cnt_ptr, 1) is None) and (tl.atomic_add(cnt_ptr, 1) is None):
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def s5(cnt_ptr, o_ptr, x_ptr):
    if (tl.atomic_add(cnt_ptr, 1) is not None) or (tl.atomic_add(cnt_ptr, 1) is None):
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def s_ifexp(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    v = 1 if (tl.atomic_add(cnt_ptr, 1) if FLAG else None) is None else 2
    tl.store(o_ptr + tl.program_id(0), v)


@triton.jit
def s9(cnt_ptr, o_ptr, x_ptr):
    if (tl.load(x_ptr) > 0) and (tl.atomic_add(cnt_ptr, 1) > 0):
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def s10(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    tl.static_assert(FLAG or tl.atomic_add(cnt_ptr, 1) is None)
    tl.store(o_ptr + tl.program_id(0), 1)


@triton.jit
def s11(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    tl.static_print(FLAG and tl.atomic_add(cnt_ptr, 1))
    tl.store(o_ptr + tl.program_id(0), 1)


@triton.jit
def s_print(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    tl.static_print(tl.atomic_add(cnt_ptr, 1) if FLAG else 0)
    tl.store(o_ptr + tl.program_id(0), 1)


@triton.jit
def s14(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    tl.device_print("v", tl.atomic_add(cnt_ptr, 1) if FLAG else 0)
    tl.store(o_ptr + tl.program_id(0), 1)


@triton.jit
def s15(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    if (FLAG and tl.atomic_add(cnt_ptr, 1) is None) or tl.atomic_add(cnt_ptr, 2) is not None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def s16(cnt_ptr, o_ptr, x_ptr, FLAG: tl.constexpr):
    if tl.atomic_add(cnt_ptr, 1) is not None and FLAG:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


CASES = (
    ("C1 if <atomic> is not None", c1, {}),
    ("C2 1 if <atomic> is None else 2", c2, {}),
    ("C3 elif <atomic> is not None (FLAG=False)", c3, {"FLAG": False}),
    ("C4 if FLAG and <atomic> is not None (FLAG=False)", c4, {"FLAG": False}),
    ("P1 if FLAG (FLAG=True)", p1, {"FLAG": True}),
    ("P2 if N > 4 (N=8)", p2, {"N": 8}),
    ("P3 1 if FLAG else 2 (FLAG=False)", p3, {"FLAG": False}),
    ("S1 if FLAG and <atomic> is not None (FLAG=True)", c4, {"FLAG": True}),
    ("S2 if FLAG or <atomic> is None (FLAG=True)", s_or, {"FLAG": True}),
    ("S3 if FLAG or <atomic> is None (FLAG=False)", s_or, {"FLAG": False}),
    ("S4 if (<atomic> is None) and (<atomic> is None)", s4, {}),
    ("S5 if (<atomic> is not None) or (<atomic> is None)", s5, {}),
    ("S6 1 if (<atomic> if FLAG else None) is None else 2 (FLAG=False)", s_ifexp, {"FLAG": False}),
    ("S7 1 if (<atomic> if FLAG else None) is None else 2 (FLAG=True)", s_ifexp, {"FLAG": True}),
    ("S8 if FLAG: ... elif <atomic> is not None (FLAG=True)", c3, {"FLAG": True}),
    ("S9 if (tl.load(x) > 0) and (<atomic> > 0) (x=0)", s9, {}),
    ("S10 tl.static_assert(FLAG or <atomic> is None) (FLAG=True)", s10, {"FLAG": True}),
    ("S11 tl.static_print(FLAG and <atomic>) (FLAG=False)", s11, {"FLAG": False}),
    ("S12 tl.static_print(<atomic> if FLAG else 0) (FLAG=False)", s_print, {"FLAG": False}),
    ("S13 tl.static_print(<atomic> if FLAG else 0) (FLAG=True)", s_print, {"FLAG": True}),
    ("S14 tl.device_print('v', <atomic> if FLAG else 0) (FLAG=False)", s14, {"FLAG": False}),
    ("S15 if (FLAG and <atomic +1> is None) or <atomic +2> is not None (FLAG=False)", s15, {"FLAG": False}),
    ("S16 if <atomic> is not None and FLAG (FLAG=False)", s16, {"FLAG": False}),
)


def captured(fn):
    """Run `fn` with the process's standard output (the compiler's `static_print` and the device's
    `printf`) sent to a pipe; return fn's result and the lines printed."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    r, w = os.pipe()
    saved = os.dup(1)
    os.dup2(w, 1)
    try:
        out = fn()
    finally:
        sys.stdout.flush()
        libc.fflush(None)
        os.dup2(saved, 1)
        os.close(w)
        os.close(saved)
    with os.fdopen(r) as f:
        return out, f.read().splitlines()


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, k, kw in CASES:
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        o = torch.zeros(4, dtype=torch.int32, device="cuda")
        x = torch.zeros(1, dtype=torch.int32, device="cuda")

        def go():
            h = k[(4,)](cnt, o, x, **kw)
            torch.cuda.synchronize()
            return h
        try:
            h, printed = captured(go)
            ttir = h.asm["ttir"]
            shown = f"; printed {len(printed)} line(s), first {printed[0]!r}" if printed else ""
            print(f"{name}: counter {cnt.item()} (each atomic that runs adds its operand in each of 4 programs); "
                  f"outputs {o.tolist()}; tt.atomic_rmw in TTIR: {ttir.count('tt.atomic_rmw')}; "
                  f"scf.if in TTIR: {ttir.count('scf.if')}{shown}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: rejected on the host: {type(e).__name__}: {str(e).splitlines()[0][:100]}", flush=True)


if __name__ == "__main__":
    main()
