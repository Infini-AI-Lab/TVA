"""lr-20e: the NaN patterns `FloatV.nan`'s docstring (lean/TVA/Lang/Scalar.lean) says the device gives.

The docstring (library issue L108): the library's canonical NaN (`0x7fffffff`, `0x7fff`) for every NaN result of
float32, float16 and bfloat16 arithmetic (`+ - *`, `tl.fma`, `/`, `tl.maximum`/`tl.minimum` with
`propagate_nan=ALL`, and on float32 `tl.div_rn`, `tl.sqrt_rn`), and of the conversions between those formats
other than `bfloat16 → float32`; `float64` arithmetic gives `0xfff8000000000000` for an invalid operation and
keeps a NaN operand's quieted payload; float64 `maximum`/`minimum` with `propagate_nan=ALL` give
`0x7ff8000000000000`; `bfloat16 → float32` and conversions into float64 keep the payload, those out of float64
the sign. Each operation runs on NaN operands with payloads (both signs, quiet and signalling) and on invalid
operations (`inf - inf`, `0 * inf`), and the distinct result patterns are printed.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I nan_results.py
"""
import sys

import torch
import triton
import triton.language as tl

PN = tl.PropagateNan.ALL


@triton.jit
def opk(x_ptr, y_ptr, o_ptr, n, OP: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    if OP == "add":
        r = x + y
    elif OP == "sub":
        r = x - y
    elif OP == "mul":
        r = x * y
    elif OP == "fma":
        r = tl.fma(x, y, y)
    elif OP == "div":
        r = x / y
    elif OP == "max":
        r = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "min":
        r = tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)
    elif OP == "div_rn":
        r = tl.div_rn(x, y)
    else:
        r = tl.sqrt_rn(x)
    tl.store(o_ptr + offs, r, mask=m)


@triton.jit
def cvk(x_ptr, o_ptr, n, D: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x.to(D), mask=m)


FMT = {torch.float32: (torch.int32, 32, 8, 23), torch.float16: (torch.int16, 16, 5, 10),
       torch.bfloat16: (torch.int16, 16, 8, 7), torch.float64: (torch.int64, 64, 11, 52)}


def pats(dt):
    _, nb, e, m = FMT[dt]
    inf = ((1 << e) - 1) << m
    p = [inf | (1 << (m - 1)), inf | (1 << (m - 1)) | 5, inf | 1, inf | 3, inf | ((1 << m) - 1)]
    return p + [q | (1 << (nb - 1)) for q in p]


def tensor_from(dt, ps):
    idt, nb, *_ = FMT[dt]
    s = [q - (1 << nb) if q >= 1 << (nb - 1) else q for q in ps]
    return torch.tensor(s, dtype=torch.int64, device="cuda").to(idt).view(dt)


def bits(t):
    idt, nb, *_ = FMT[t.dtype]
    return sorted(set(int(v) & ((1 << nb) - 1) for v in t.view(idt).tolist()))


def run_op(op, dt, x, y):
    o = torch.empty_like(x)
    opk[(1,)](x, y, o, x.numel(), OP=op, B=32)
    return o


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        ps = pats(dt)
        nan = tensor_from(dt, ps)
        one = torch.ones_like(nan)
        inf = torch.full_like(nan, float("inf"))
        zero = torch.zeros_like(nan)
        ops = ["add", "sub", "mul", "fma", "div", "max", "min"] + (["div_rn", "sqrt_rn"] if dt == torch.float32 else [])
        print(f"\n{dt}: NaN operands {', '.join(f'{p:x}' for p in ps)}")
        for op in ops:
            try:
                r1 = run_op(op, dt, nan, one)           # NaN operand
                r2 = run_op(op, dt, one, nan)
                rinv = run_op(op, dt, inf, inf) if op == "sub" else (run_op(op, dt, zero, inf) if op == "mul" else None)
                torch.cuda.synchronize()
            except Exception as ex:
                print(f"  {op:8s} rejected: {type(ex).__name__}")
                continue
            line = f"  {op:8s} NaN op x: {[hex(b) for b in bits(r1)]}; NaN op y: {[hex(b) for b in bits(r2)]}"
            if rinv is not None:
                line += f"; invalid ({'inf-inf' if op == 'sub' else '0*inf'}): {[hex(b) for b in bits(rinv)]}"
            if op == "sqrt_rn":
                rneg = run_op(op, dt, -one, one)
                line += f"; sqrt_rn(-1): {[hex(b) for b in bits(rneg)]}"
            print(line)
    print("\nconversions of NaN operands (default rounding):")
    for s, d in ((torch.float16, torch.float32), (torch.float32, torch.float16), (torch.float32, torch.bfloat16),
                 (torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16), (torch.bfloat16, torch.float32),
                 (torch.float32, torch.float64), (torch.float16, torch.float64), (torch.bfloat16, torch.float64),
                 (torch.float64, torch.float32), (torch.float64, torch.float16), (torch.float64, torch.bfloat16)):
        x = tensor_from(s, pats(s))
        o = torch.empty(x.numel(), dtype=d, device="cuda")
        D = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16,
             torch.float64: tl.float64}[d]
        try:
            cvk[(1,)](x, o, x.numel(), D=D, B=32)
            torch.cuda.synchronize()
            print(f"  {str(s):15s} -> {str(d):15s}: " + ", ".join(f"{a:x}->{b:x}" for a, b in zip(
                pats(s), [int(v) & ((1 << FMT[d][1]) - 1) for v in o.view(FMT[d][0]).tolist()])))
        except Exception as ex:
            print(f"  {s} -> {d}: rejected: {type(ex).__name__}")


if __name__ == "__main__":
    main()
