"""A name assigned a Python number inside a kernel (lr-8b, round 8).

`visit_Assign` (code-generator.py:727-738) passes an assigned value through `_sanitize_value`,
which turns a Python number into a scalar tensor (`semantic.to_tensor`) unless the target is
annotated `tl.constexpr`. These cases compare a literal written in place with the same literal
assigned to a name first. Each case runs in its own process on GPU 0.
"""
import os, subprocess, sys
import torch
import triton
import triton.language as tl


@triton.jit
def k_dtype(x_ptr, out_ptr, NAMED: tl.constexpr):
    # bfloat16 tensor times 0.5: the result's dtype
    x = tl.load(x_ptr + tl.arange(0, 4))
    if NAMED:
        s = 0.5
        y = x * s
        tl.static_assert(y.dtype == tl.float32)
    else:
        y = x * 0.5
        tl.static_assert(y.dtype == tl.bfloat16)
    tl.store(out_ptr + tl.arange(0, 4), y.to(tl.float32))


@triton.jit
def k_small(x_ptr, out_ptr, NAMED: tl.constexpr):
    # bfloat16 tensor times 1e-7: a literal is built in bfloat16 (+0); a named one is a float32 tensor
    x = tl.load(x_ptr + tl.arange(0, 4))
    if NAMED:
        e = 1e-7
        y = x * e
    else:
        y = x * 1e-7
    tl.store(out_ptr + tl.arange(0, 4), y.to(tl.float32))


@triton.jit
def k_and(c_ptr, NAMED: tl.constexpr):
    # `F and <atomic>` with F False: a compile-time False ends the `and`; a named False does not
    if NAMED:
        F = False
        if F and tl.atomic_add(c_ptr, 1) > 0:
            tl.atomic_add(c_ptr + 1, 1)
    else:
        if False and tl.atomic_add(c_ptr, 1) > 0:
            tl.atomic_add(c_ptr + 1, 1)


@triton.jit
def k_branch(out_ptr, NAMED: tl.constexpr):
    # `if F:` with F False: a compile-time condition visits one branch; a named False is a runtime if
    if NAMED:
        F = False
        if F:
            tl.static_assert(False, "branch visited")
    else:
        if False:
            tl.static_assert(False, "branch visited")
    tl.store(out_ptr, 1)


@triton.jit
def k_div(out_ptr, NAMED: tl.constexpr):
    # -7 // 2: Python's floor division (-4) at compile time; C's truncation (-3) on tensors
    if NAMED:
        a = -7
        b = 2
        tl.store(out_ptr, a // b)
    else:
        tl.store(out_ptr, -7 // 2)


@triton.jit
def _scale(x, s):
    return x * s


@triton.jit
def k_helper(x_ptr, out_ptr, NAMED: tl.constexpr):
    # a literal passed to a helper is a compile-time value inside it; a named one is a tensor
    x = tl.load(x_ptr + tl.arange(0, 4))
    if NAMED:
        e = 1e-7
        y = _scale(x, e)
    else:
        y = _scale(x, 1e-7)
    tl.store(out_ptr + tl.arange(0, 4), y.to(tl.float32))


@triton.jit
def k_tuple(x_ptr, out_ptr, NAMED: tl.constexpr):
    # a tuple assignment: each Python number becomes a tensor
    x = tl.load(x_ptr + tl.arange(0, 4))
    if NAMED:
        e, z = 1e-7, 0
        y = x * e + z
    else:
        y = x * 1e-7 + 0
    tl.store(out_ptr + tl.arange(0, 4), y.to(tl.float32))


def run(name):
    dev = "cuda"
    kind, named = name[:-1], name.endswith("n")
    try:
        if kind == "dtype":
            x = torch.tensor([1.0, 3.0, 1.0 / 3.0, 1e-30], device=dev, dtype=torch.bfloat16)
            out = torch.zeros(4, device=dev)
            k_dtype[(1,)](x, out, NAMED=named)
            print(name, "result", out.cpu().tolist())
        elif kind == "small":
            x = torch.tensor([1.0, 3.0, 100.0, -2.0], device=dev, dtype=torch.bfloat16)
            out = torch.zeros(4, device=dev)
            k_small[(1,)](x, out, NAMED=named)
            print(name, "result", out.cpu().tolist())
        elif kind in ("helper", "tuple"):
            x = torch.tensor([1.0, 3.0, 100.0, -2.0], device=dev, dtype=torch.bfloat16)
            out = torch.zeros(4, device=dev)
            (k_helper if kind == "helper" else k_tuple)[(1,)](x, out, NAMED=named)
            print(name, "result", out.cpu().tolist())
        elif kind == "and":
            c = torch.zeros(2, device=dev, dtype=torch.int32)
            k_and[(64,)](c, NAMED=named)
            torch.cuda.synchronize()
            print(name, "atomics run (first, second)", c.cpu().tolist(), "of 64 programs")
        elif kind == "branch":
            out = torch.zeros(1, device=dev, dtype=torch.int32)
            k_branch[(1,)](out, NAMED=named)
            print(name, "compiled; stored", out.cpu().tolist())
        elif kind == "div":
            out = torch.zeros(1, device=dev, dtype=torch.int32)
            k_div[(1,)](out, NAMED=named)
            print(name, "stored", out.cpu().tolist())
    except Exception as e:
        msg = [l for l in str(e).splitlines() if l.strip()]
        print(name, "rejected:", type(e).__name__, (msg[-1] if msg else "")[:160])


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("triton", triton.__version__, torch.cuda.get_device_name(0), torch.__version__)
        for k in ("dtype", "small", "helper", "tuple", "and", "branch", "div"):
            for suf in ("l", "n"):   # l: literal in place, n: literal assigned to a name first
                p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), k + suf],
                                   capture_output=True, text=True)
                sys.stdout.write(p.stdout or (k + suf + " no output: " + p.stderr[-300:] + "\n"))
