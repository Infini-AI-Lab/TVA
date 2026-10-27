"""lr-2b, library review round 2, semantics survey: is the result of a `tl.*` math function a
*product* in the compiled dataflow, so that a following `+` may be contracted into an `fma`?

The library models `tl.log(x) + y` as `P.math .log` followed by `tl.add` (IEEE) unless a product
feeds the addition (TRANSCRIPTION.md §4 lists `*`, `tl.dot` and `tl.rand` as products). If the
lowering of `tl.log` ends with a multiplication, LLVM's contraction (enable_fp_fusion=True, the
default) can turn `log(x) + y` into `fma(.., .., y)`, and then the device's result is not the IEEE
sum of any float the function could have returned.

Method. For each function `f` and dtype, kernel `alone` stores `f(x)`, kernel `added` stores
`f(x) + y`. On the host, the IEEE sum `f(x) + y` of `alone`'s output is computed (torch, same dtype,
round to nearest even), and compared bitwise with `added`. `y` is chosen to cancel: `y = -f(x)`
rounded in a coarser way, so that the exact sum is tiny and the IEEE sum of *any* float near
`f(x)` with `y` lies on the grid of `ulp(y)` (Sterbenz): a result off that grid can come from no
float returned by `f`. The same is run with enable_fp_fusion=False as a control. The PTX of
`added` is searched for `fma` instructions.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-2/lr-2b/work/fusion_math.py
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
N = 1 << 16
B = 1024


def make_kernels(fn):
    @triton.jit
    def alone(x_ptr, out_ptr, n, B: tl.constexpr):
        offs = tl.program_id(0) * B + tl.arange(0, B)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        tl.store(out_ptr + offs, fn(x), mask=m)

    @triton.jit
    def added(x_ptr, y_ptr, out_ptr, n, B: tl.constexpr):
        offs = tl.program_id(0) * B + tl.arange(0, B)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        y = tl.load(y_ptr + offs, mask=m)
        tl.store(out_ptr + offs, fn(x) + y, mask=m)

    return alone, added


@triton.jit
def f_log(x):
    return tl.log(x)


@triton.jit
def f_log2(x):
    return tl.math.log2(x)


@triton.jit
def f_exp(x):
    return tl.exp(x)


@triton.jit
def f_exp2(x):
    return tl.math.exp2(x)


@triton.jit
def f_sin(x):
    return tl.sin(x)


@triton.jit
def f_cos(x):
    return tl.cos(x)


@triton.jit
def f_sqrt(x):
    return tl.sqrt(x)


@triton.jit
def f_rsqrt(x):
    return tl.math.rsqrt(x)


@triton.jit
def f_erf(x):
    return tl.math.erf(x)


@triton.jit
def f_sigmoid(x):
    return tl.sigmoid(x)


FNS = [("tl.log", f_log, (0.01, 100.0)), ("tl.math.log2", f_log2, (0.01, 100.0)),
       ("tl.exp", f_exp, (-10.0, 10.0)), ("tl.math.exp2", f_exp2, (-10.0, 10.0)),
       ("tl.sin", f_sin, (-3.0, 3.0)), ("tl.cos", f_cos, (-3.0, 3.0)),
       ("tl.sqrt", f_sqrt, (0.01, 100.0)), ("tl.math.rsqrt", f_rsqrt, (0.01, 100.0)),
       ("tl.math.erf", f_erf, (-2.0, 2.0)), ("tl.sigmoid", f_sigmoid, (-5.0, 5.0))]


def coarse(v, dtype):
    """-v rounded to about half the format's precision, so that v + coarse(v) cancels."""
    keep = 12 if dtype == torch.float32 else 26
    m, e = torch.frexp(v)
    return -(torch.round(m * 2.0 ** keep) / 2.0 ** keep) * torch.pow(torch.full_like(v, 2.0), e.to(v.dtype))


def off_grid(z, y, dtype):
    """z is not a multiple of half the ulp of y: no exact sum r' + y (r' a float near -y) gives it."""
    ibits = 23 if dtype == torch.float32 else 52
    _, e = torch.frexp(y)
    grid = torch.pow(torch.full_like(y, 2.0), (e - 1 - ibits - 1).to(y.dtype))
    q = z / grid
    return (q != torch.round(q)) & torch.isfinite(z) & (y != 0)


def bits_equal(a, b):
    nan = torch.isnan(a) & torch.isnan(b)
    ia = a.view(torch.int32 if a.dtype == torch.float32 else torch.int64)
    ib = b.view(torch.int32 if b.dtype == torch.float32 else torch.int64)
    return (ia == ib) | nan


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    gen = torch.Generator(device="cpu").manual_seed(1)
    for dtype in (torch.float32, torch.float64):
        print(f"\n== {str(dtype).replace('torch.', '')} ==", flush=True)
        for name, fn, (lo, hi) in FNS:
            alone, added = make_kernels(fn)
            x = (torch.rand(N, generator=gen, dtype=torch.float64) * (hi - lo) + lo).to(dtype).to(DEV)
            r = torch.empty_like(x)
            alone[(triton.cdiv(N, B),)](x, r, N, B)
            y = coarse(r, dtype)
            for fusion in (True, False):
                z = torch.empty_like(x)
                k = added[(triton.cdiv(N, B),)](x, y, z, N, B, enable_fp_fusion=fusion)
                torch.cuda.synchronize()
                ieee = r + y
                diff = ~bits_equal(z, ieee)
                imposs = diff & off_grid(z, y, dtype)
                nfma = sum(1 for l in k.asm["ptx"].splitlines() if l.strip().startswith("fma."))
                ex = ""
                if imposs.any():
                    i = int(torch.nonzero(imposs)[0])
                    ex = (f"; e.g. x={x[i].item()!r}: f(x)={r[i].item()!r}, y={y[i].item()!r}, "
                          f"IEEE f(x)+y={ieee[i].item()!r}, device={z[i].item()!r}")
                print(f"{name:14s} fusion={str(fusion):5s}: {int(diff.sum()):6d} of {N} differ from the IEEE sum, "
                      f"{int(imposs.sum()):6d} off the grid of y (no float result gives them); "
                      f"fma in PTX: {nfma}{ex}", flush=True)


if __name__ == "__main__":
    main()
