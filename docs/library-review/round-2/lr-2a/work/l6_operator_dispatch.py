"""lr-2a, round 2, L6/L10: can an expression that TRANSCRIPTION.md §10 calls pure run an operation?

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-2/lr-2a/work/l6_operator_dispatch.py

§10 lists `+`, the comparisons, `is not`, names and attributes as pure, with "no call of any kind".
But the front end applies a binary operator on a value that is not a tensor by looking up the
operator's method on it and calling that method (`triton-3.8.0-code-generator.py:795-808`). For an
aggregate (`triton.language.core._aggregate`) the method may be a `@triton.jit` function, which is
inlined. Each case counts the programs (grid of 4) that ran an atomic increment.
"""
import os
import time

import torch
import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

DEV = "cuda"


@aggregate
class Counter:
    ptr: tl.tensor

    @triton.jit
    def __add__(self, other):
        tl.atomic_add(self.ptr, 1)
        return self

    @triton.jit
    def __eq__(self, other):
        tl.atomic_add(self.ptr, 1)
        return True

    @triton.jit
    def __lt__(self, other):
        tl.atomic_add(self.ptr, 1)
        return self.ptr is not None


@triton.jit
def make(c_ptr):
    return Counter(c_ptr)


@triton.jit
def g1(c_ptr):
    a = make(c_ptr)
    # names, `+` and `is not`: pure by §10, so `compile-check` would drop the line
    tl.static_assert((a + a) is not None)


@triton.jit
def g2(c_ptr):
    a = make(c_ptr)
    # names and `==`: pure by §10, so `no-op:print` would drop the line
    tl.device_assert(a == a, "equal")


@triton.jit
def g3(c_ptr):
    a = make(c_ptr)
    # a Python `assert` on a comparison: pure by §10
    assert a < a, "less"


@triton.jit
def g4(c_ptr):
    a = make(c_ptr)
    # `dead`: a value no path uses, computed by a "pure" expression
    unused = a + a


def run(name, kern, grid=4):
    c = torch.zeros(1, device=DEV, dtype=torch.int32)
    try:
        kern[(grid,)](c)
        torch.cuda.synchronize()
        res = f"compiles; counter = {int(c.item())} after {grid} programs"
    except Exception as e:  # noqa: BLE001
        lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
        res = f"{type(e).__name__}: {lines[-1][:160] if lines else ''}"
    print(f"{name}\n    {res}", flush=True)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}, "
          f"TRITON_DEBUG={os.environ.get('TRITON_DEBUG')!r}", flush=True)
    run("G1 tl.static_assert((a + a) is not None), a an aggregate whose jitted __add__ adds 1", g1)
    run("G2 tl.device_assert(a == a, 'equal'), jitted __eq__ adds 1", g2)
    run("G3 assert a < a, 'less', jitted __lt__ adds 1", g3)
    run("G4 unused = a + a (a `dead` value), jitted __add__ adds 1", g4)


if __name__ == "__main__":
    main()
