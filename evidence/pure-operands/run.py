"""TVA library evidence (library issue L14): which operands keep an expression free of effects.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/pure-operands/run.py
Output: evidence/pure-operands/out.txt (README.md says what it shows).

Part A gives an aggregate (a class decorated with `triton.language.core._aggregate`) operator
methods that are `@triton.jit` functions with an effect (an atomic increment of a counter), uses
the aggregate as the operand of an operator, a comparison, a subscript or a unary operator inside
lines the transcription rules could drop, and counts how many programs ran the increment.

Part B uses the operand types the purity rule allows (tensors of integers, floats, pointers and
booleans, `tl.constexpr` values, dtypes, and tuples of these) with every operator the rule lists,
inside `tl.static_assert`, `tl.device_assert`, `assert`, `tl.static_print` and as unused (`dead`)
values. It compares the kernel's TTIR and PTX with those of the same kernel without these lines
(locations and the kernel's name removed) and checks the output.
"""
import os
import re
import time

import torch
import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

DEV = "cuda"
B = 64
N = 200


# ---------------------------------------------------------------- part A: aggregates

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
    def __getitem__(self, idx):
        tl.atomic_add(self.ptr, 1)
        return idx

    @triton.jit
    def __neg__(self):
        tl.atomic_add(self.ptr, 1)
        return self


@triton.jit
def make(c_ptr):
    return Counter(c_ptr)


@triton.jit
def a1(c_ptr):
    a = make(c_ptr)
    tl.static_assert((a + a) is not None)


@triton.jit
def a2(c_ptr):
    a = make(c_ptr)
    tl.device_assert(a == a, "equal")


@triton.jit
def a3(c_ptr):
    a = make(c_ptr)
    assert a < a, "less"


@triton.jit
def a4(c_ptr):
    a = make(c_ptr)
    unused = a + a  # noqa: F841


@triton.jit
def a5(c_ptr):
    a = make(c_ptr)
    tl.static_assert(a[0] is not None)


@triton.jit
def a6(c_ptr):
    a = make(c_ptr)
    tl.static_assert((-a) is not None)


@triton.jit
def a7(c_ptr):
    a = make(c_ptr)
    tl.static_assert((a, ) == (a, ))


@triton.jit
def a8(c_ptr):
    a = make(c_ptr)
    tl.static_assert(((a, ) + (a, )) is not None)


@triton.jit
def a0(c_ptr):
    # control: the aggregate is made and nothing else
    a = make(c_ptr)  # noqa: F841


# ---------------------------------------------------------------- part B: allowed operand types

@triton.jit
def base(x_ptr, i_ptr, out_ptr, iout_ptr, n, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    i = tl.load(i_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, x * 2.0 + 1.0, mask=m)
    tl.store(iout_ptr + offs, i * 3 + offs, mask=m)


@triton.jit
def with_pure(x_ptr, i_ptr, out_ptr, iout_ptr, n, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    i = tl.load(i_ptr + offs, mask=m, other=0)
    p = x_ptr + offs
    # integer tensors: + - * ~ & | ^ and comparisons
    tl.static_assert((i + offs - 2 * i * offs) is not None)
    tl.static_assert(((~i) & offs | (i ^ offs)) is not None)
    tl.static_assert(((i < offs) == (i <= offs)) is not None)
    tl.static_assert(((i > 0) != (i >= 0)) is not None)
    # float tensors: + - * and comparisons, unary minus
    tl.static_assert((x + 1.0 - x * 2.0 - (-x)) is not None)
    tl.static_assert(((x < 0.0) | (x == x) & (x != 1.0)) is not None)
    # pointer tensors: + (a comparison of two pointer tensors does not compile in 3.8.0)
    tl.static_assert((p + 1 + offs) is not None)
    tl.static_assert((p + 1).dtype == p.dtype)
    # boolean masks: & | ^ ~, not, and, or
    tl.static_assert(((m & m) | (~m) ^ m) is not None)
    tl.static_assert((not (pid < 0)) is not None)
    tl.static_assert(((pid >= 0) and (pid < 1000000)) is not None)
    tl.static_assert(((pid < 0) or (pid >= 0)) is not None)
    # constexprs and Python values: / // % ** << >> between constexprs only
    tl.static_assert((B / 2) * 2 == B)
    tl.static_assert(B // 2 * 2 + B % 2 == B)
    tl.static_assert(B ** 1 == B and (B << 1) >> 1 == B)
    tl.static_assert(not (B < 0) or B is None)
    # dtypes, attributes, subscripts that add axes, shapes and tuples
    tl.static_assert(x.dtype == tl.float32)
    tl.static_assert(x_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(i.dtype != x.dtype)
    tl.static_assert(offs.shape[0] == B)
    tl.static_assert(offs.numel == B)
    tl.static_assert(offs[:, None].shape[1] == 1)
    tl.static_assert(offs[None, :] is not None)
    tl.static_assert((offs[:, None] < offs[None, :]) is not None)
    tl.static_assert(offs.shape == (B, ))
    tl.static_assert((x, offs) is not None)
    tl.static_assert(((x, offs) + (m, )) is not None)
    tl.static_assert(((x, offs) == (x, offs)))
    # the other calls
    tl.device_assert((offs >= 0) & m | ~m, "offsets are nonnegative")
    tl.device_assert((x == x) | (x != x), "x is a value", mask=m)
    assert pid * B >= 0 and pid >= 0, "pid * B is nonnegative"
    tl.static_print("pure", offs + 1, x * 2.0, x.dtype, offs.shape, B // 2)
    # unused values
    dead1 = offs * 3 + 1  # noqa: F841
    dead2 = (x - 1.0) * x  # noqa: F841
    dead3 = offs[:, None] < offs[None, :]  # noqa: F841
    dead4 = (i, x)  # noqa: F841
    tl.store(out_ptr + offs, x * 2.0 + 1.0, mask=m)
    tl.store(iout_ptr + offs, i * 3 + offs, mask=m)


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
        out.append(line.replace("with_pure", "K").replace("base", "K"))
    return "\n".join(l for l in out if l.strip())


def last_line(e):
    lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
    msg = lines[-1] if lines else ""
    return re.sub(r"(/[^\s:'\"]+)+", "<path>", msg)[:140]


def run_count(name, kern, grid=4):
    c = torch.zeros(1, device=DEV, dtype=torch.int32)
    try:
        kern[(grid, )](c)
        torch.cuda.synchronize()
        res = f"compiles; counter = {int(c.item())} after {grid} programs"
    except Exception as e:  # noqa: BLE001
        res = f"rejected on the host: {type(e).__name__}: {last_line(e)}"
    print(f"{name}\n    {res}", flush=True)


def compiled(kern):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(N, generator=g).to(DEV)
    i = torch.randint(-1000, 1000, (N, ), generator=g, dtype=torch.int32).to(DEV)
    out = torch.zeros(N, device=DEV, dtype=torch.float32)
    iout = torch.zeros(N, device=DEV, dtype=torch.int32)
    ck = kern[(triton.cdiv(N, B), )](x, i, out, iout, N, B)
    torch.cuda.synchronize()
    offs = torch.arange(N, device=DEV, dtype=torch.int32)
    ok = bool(torch.equal(out, x * 2.0 + 1.0)) and bool(torch.equal(iout, i * 3 + offs))
    return ck, ok


def main():
    # Compile-time prints go to file descriptor 1: send them to /dev/null and print the report on a
    # copy of the original descriptor, so that the output holds only the report.
    global print
    report = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    print = lambda *a, **k: (report.write(" ".join(map(str, a)) + k.get("end", "\n")), report.flush())  # noqa: E731
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}, "
          f"TRITON_DEBUG={os.environ.get('TRITON_DEBUG')!r}")

    print("\n== A. An aggregate as an operand (grid of 4 programs; each jitted method adds 1) ==")
    run_count("A0 control: a = make(c) only", a0)
    run_count("A1 tl.static_assert((a + a) is not None)  [__add__]", a1)
    run_count("A2 tl.device_assert(a == a, 'equal')  [__eq__]", a2)
    run_count("A3 assert a < a, 'less'  [__lt__]", a3)
    run_count("A4 unused = a + a  (a dead value)  [__add__]", a4)
    run_count("A5 tl.static_assert(a[0] is not None)  [__getitem__]", a5)
    run_count("A6 tl.static_assert((-a) is not None)  [__neg__]", a6)
    run_count("A7 tl.static_assert((a,) == (a,))  [a tl.tuple comparison]", a7)
    run_count("A8 tl.static_assert(((a,) + (a,)) is not None)  [a tl.tuple concatenation]", a8)

    print("\n== B. The allowed operand types, every listed operator ==")
    k0, ok0 = compiled(base)
    k1, ok1 = compiled(with_pure)
    for kind in ("ttir", "ptx"):
        same = normal(k0.asm[kind], kind) == normal(k1.asm[kind], kind)
        print(f"B1 {kind.upper()} identical to the kernel without the lines: {same}")
    ops = lambda k: sorted(re.sub(r"%[\w.#]+", "%v", l) for l in normal(k.asm["ttir"], "ttir").splitlines())
    print(f"B1 TTIR identical up to value names and the order of the lines (same operations): {ops(k0) == ops(k1)}")
    print(f"B1 outputs correct: without the lines {ok0}, with the lines {ok1}")
    ttir = k1.asm["ttir"]
    for op in ("tt.atomic_rmw", "tt.atomic_cas", "tt.print", "tt.assert", "tt.call"):
        print(f"B1 TTIR of the kernel with the lines contains {op}: {op in ttir}")


if __name__ == "__main__":
    main()
