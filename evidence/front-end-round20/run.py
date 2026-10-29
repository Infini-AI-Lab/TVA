"""TVA library evidence (library issue L102): what a conditional expression with a runtime condition, and a
helper's `return`s, give when a side or a returned value is a Python number.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round20/run.py
Output: evidence/front-end-round20/out.txt (README.md says what it shows).

`c` is an `int` argument, `5` or `-5`: never `1`, which Triton passes as a compile-time constant
(`docs/SEMANTICS.md`, "Scalar arguments"), so that every condition below is a runtime one.

E1. `a if c else b` with `c` a runtime scalar (`visit_IfExp`, `primitives/triton-3.8.0-code-generator.py:998-1036`:
    each side is `to_tensor`ed, then both must have one type):
    - E1a `(-1 if c > 0 else -3) // 2`, stored for `c = 5` and `c = -5`: the `int32` tensor `//`, which
      truncates (Python's floor division would give `-1` and `-2`);
    - E1b `x * (0.1 if c > 0 else 0.2)` on every finite `bfloat16` `x`, stored as `bfloat16`: against the
      `float32` product of `x` and `float32(0.1)` rounded to `bfloat16` (the side a `float32` tensor), and
      against the `bfloat16` product with `bfloat16(0.1)` (the side a Python number, which a `bfloat16`
      operand would keep in `bfloat16`);
    - E1c `1 if c > 0 else 1.0` and `x if c > 0 else 1.0` with an `int32` `x`: rejected on the host;
    - E1d `x + 1 if c > 0 else x - 1` with an `int32` `x` (both sides tensors of one type): the taken side.
E2. A helper whose `return`s the front end visits give other kinds of values (`decide_return_type`,
    `cast_to`, `handle_returns`, `primitives/triton-3.8.0-code-generator.py:556-626`), with `c` a runtime
    scalar:
    - E2a `if c > 0: return -7` then `return -9`; `h(c) // 2` stored: scalar `int32` tensors;
    - E2b `if c > 0: return 1` then `return 1.0`; `x * h(c)` with the `int32` `x = 2^24 + 1`: the first
      `return`'s `1` on both paths (`16777217`), where `1.0` gives `16777216` through `float32`;
    - E2c `if c > 0: return 5` and nothing after; `h(c)` stored for `c = -5`: `5`, where Python's
      function returns `None`;
    - E2d (admitted) `return 5` on both paths: `5`; E2e (admitted) `return x + 1` / `return x - 1`: the
      taken path's tensor.
"""
import os
import struct
import sys
import tempfile

SRC = '''import triton
import triton.language as tl


@triton.jit
def e1a(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], (-1 if c > 0 else -3) // 2, tl.int32))


@triton.jit
def e1b(o_ptr, x_ptr, n, c, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x * (0.1 if c > 0 else 0.2), mask=m)


@triton.jit
def e1c_num(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], 1 if c > 0 else 1.0, tl.float32))


@triton.jit
def e1c_mix(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x if c > 0 else 1.0)


@triton.jit
def e1d(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x + 1 if c > 0 else x - 1)


@triton.jit
def h_a(c):
    if c > 0:
        return -7
    return -9


@triton.jit
def e2a(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], h_a(c) // 2, tl.int32))


@triton.jit
def h_b(c):
    if c > 0:
        return 1
    return 1.0


@triton.jit
def e2b(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * h_b(c))


@triton.jit
def h_c(c):
    if c > 0:
        return 5


@triton.jit
def e2c(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], h_c(c), tl.int32))


@triton.jit
def h_d(c):
    if c > 0:
        return 5
    return 5


@triton.jit
def e2d(o_ptr, c):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], h_d(c), tl.int32))


@triton.jit
def h_e(x, c):
    if c > 0:
        return x + 1
    return x - 1


@triton.jit
def e2e(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, h_e(tl.load(x_ptr + i), c))
'''


def main():
    import torch
    import triton
    tmp = tempfile.mkdtemp(prefix="tva-fe20-")
    path = os.path.join(tmp, "fe20_mod.py")
    with open(path, "w") as fh:
        fh.write(SRC)
    import importlib.util
    spec = importlib.util.spec_from_file_location("fe20_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    dev = "cuda"

    def i32(kern, *args, extra=()):
        o = torch.zeros(1, dtype=torch.int32, device=dev)
        try:
            kern[(1,)](o, *extra, *args)
            torch.cuda.synchronize()
        except Exception as ex:
            return f"rejected on the host ({type(ex).__name__})"
        return o.item()

    xi = torch.full((1,), 2 ** 24 + 1, dtype=torch.int32, device=dev)
    x7 = torch.full((1,), 7, dtype=torch.int32, device=dev)
    print("E1. a conditional expression with a runtime condition")
    print(f"  E1a (-1 if c > 0 else -3) // 2: c = 5 -> {i32(mod.e1a, 5)}, c = -5 -> {i32(mod.e1a, -5)} "
          f"(Python's floor division: -1, -2)")
    # E1b: every finite bfloat16
    pats = torch.arange(0, 1 << 16, dtype=torch.int32, device=dev).to(torch.int16)
    xb = pats.view(torch.bfloat16)
    fin = torch.isfinite(xb)
    xb = xb[fin].contiguous()
    for c, k in ((5, 0.1), (-5, 0.2)):
        o = torch.empty_like(xb)
        n = xb.numel()
        mod.e1b[(triton.cdiv(n, 1024),)](o, xb, n, c, B=1024)
        torch.cuda.synchronize()
        kf = struct.unpack("<f", struct.pack("<f", k))[0]
        ref32 = (xb.float() * torch.tensor(kf, dtype=torch.float32, device=dev)).to(torch.bfloat16)
        refbf = xb * torch.tensor(k, dtype=torch.bfloat16, device=dev)
        same = lambda a, b: ((a.view(torch.int16) == b.view(torch.int16)) | (torch.isnan(a) & torch.isnan(b)))
        print(f"  E1b x * ({k} if c > 0 else ...) with c = {c}, on all {n} finite bfloat16 x: "
              f"{int((~same(o, ref32)).sum())} differ from the float32 product rounded to bfloat16, "
              f"{int((~same(o, refbf)).sum())} differ from the bfloat16 product")
    print(f"  E1c 1 if c > 0 else 1.0: {i32(mod.e1c_num, 5)}; x if c > 0 else 1.0 (int32 x): "
          f"{i32(mod.e1c_mix, 5, extra=(x7,))}")
    print(f"  E1d x + 1 if c > 0 else x - 1 (int32 x = 7): c = 5 -> {i32(mod.e1d, 5, extra=(x7,))}, "
          f"c = -5 -> {i32(mod.e1d, -5, extra=(x7,))}")
    print("E2. a helper's returns")
    print(f"  E2a return -7 / return -9; h(c) // 2: c = 5 -> {i32(mod.e2a, 5)}, c = -5 -> {i32(mod.e2a, -5)} "
          f"(Python: -4, -5)")
    print(f"  E2b return 1 / return 1.0; x * h(c), x = 2^24 + 1: c = 5 -> {i32(mod.e2b, 5, extra=(xi,))}, "
          f"c = -5 -> {i32(mod.e2b, -5, extra=(xi,))} (Python: 16777217, 16777216)")
    print(f"  E2c return 5 in the if only; h(c): c = 5 -> {i32(mod.e2c, 5)}, c = -5 -> {i32(mod.e2c, -5)} "
          f"(Python: 5, None)")
    print(f"  E2d return 5 on both paths: c = 5 -> {i32(mod.e2d, 5)}, c = -5 -> {i32(mod.e2d, -5)}")
    print(f"  E2e return x + 1 / return x - 1 (x = 7): c = 5 -> {i32(mod.e2e, 5, extra=(x7,))}, "
          f"c = -5 -> {i32(mod.e2e, -5, extra=(x7,))}")


if __name__ == "__main__":
    main()
