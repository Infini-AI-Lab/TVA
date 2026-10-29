"""lr-12a: which NaN patterns Triton 3.8.0 produces on the H200 (GPU 0).

`FloatV.nan`'s docstring says the canonical NaN NVIDIA arithmetic produces is positive with every
exponent and fraction bit set (0x7fffffff, 0x7fff). The library's exact operations are up to NaN
payload, but the constants Triton builds for a Python NaN (`FloatV.litF32` ... `litBF16`) and
`FloatV.nan` are exact patterns. This prints the device's pattern for each case.

Usage: CUDA_VISIBLE_DEVICES=0 python -I nan_probe.py
"""
import torch
import triton
import triton.language as tl

U = {16: tl.uint16, 32: tl.uint32, 64: tl.uint64}


@triton.jit
def arith(xb_ptr, out_ptr, CASE: tl.constexpr, DT: tl.constexpr, UB: tl.constexpr, W: tl.constexpr):
    i = tl.arange(0, 4)
    xb = tl.load(xb_ptr + i)
    if W == 64:
        x = xb.to(tl.float64, bitcast=True)
    else:
        x = xb.to(UB).to(DT, bitcast=True)
    # x = [+inf, 0, nan(a payload), -nan(a payload)]
    inf = tl.sum(tl.where(i == 0, x, 0.0).to(DT), 0)
    if CASE == 0:
        r = x - inf            # inf - inf, 0 - inf, nan - inf
    elif CASE == 1:
        r = x * 0.0            # inf * 0, 0, nan * 0 (computed: x is a runtime value)
    elif CASE == 2:
        r = x + 1.0            # inf, 1, payload NaN + 1
    elif CASE == 3:
        r = tl.fma(x, 0.0 * x, x)
    else:
        r = -x
    if W == 64:
        rb = r.to(tl.int64, bitcast=True)
    else:
        rb = r.to(UB, bitcast=True).to(tl.int64)
    tl.store(out_ptr + i, rb)


@triton.jit
def consts(out_ptr, DT: tl.constexpr, UB: tl.constexpr, W: tl.constexpr, NEG: tl.constexpr):
    i = tl.arange(0, 4)
    if NEG:
        r = tl.full((4,), -float("nan"), DT)
    else:
        r = tl.full((4,), float("nan"), DT)
    if W == 64:
        rb = r.to(tl.int64, bitcast=True)
    else:
        rb = r.to(UB, bitcast=True).to(tl.int64)
    tl.store(out_ptr + i, rb)


@triton.jit
def cvt(xb_ptr, out_ptr, SRC: tl.constexpr, SU: tl.constexpr, SW: tl.constexpr, DST: tl.constexpr,
        DU: tl.constexpr, DW: tl.constexpr):
    i = tl.arange(0, 4)
    xb = tl.load(xb_ptr + i)
    if SW == 64:
        x = xb.to(tl.float64, bitcast=True)
    else:
        x = xb.to(SU).to(SRC, bitcast=True)
    r = x.to(DST)
    if DW == 64:
        rb = r.to(tl.int64, bitcast=True)
    elif DW == 8:
        rb = r.to(tl.uint8, bitcast=True).to(tl.int64)
    else:
        rb = r.to(DU, bitcast=True).to(tl.int64)
    tl.store(out_ptr + i, rb)


@triton.jit
def divsqrt(xb_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(xb_ptr + i).to(tl.uint32).to(tl.float32, bitcast=True)
    if CASE == 0:
        r = tl.div_rn(x * 0.0, x * 0.0)
    elif CASE == 1:
        r = tl.sqrt_rn(-tl.abs(x) - 1.0)
    else:
        r = x / (x * 0.0)
    tl.store(out_ptr + i, r.to(tl.uint32, bitcast=True).to(tl.int64))


FM = {"f16": (tl.float16, 16, (0x7c00, 0, 0x7d23, 0xfe01)),
      "bf16": (tl.bfloat16, 16, (0x7f80, 0, 0x7fa3, 0xffc1)),
      "f32": (tl.float32, 32, (0x7f800000, 0, 0x7f812345, 0xffc00001)),
      "f64": (tl.float64, 64, (0x7ff0000000000000, 0, 0x7ff0000000012345, 0xfff8000000000001 - (1 << 64)))}


def main():
    for name, (dt, w, pats) in FM.items():
        xb = torch.tensor(list(pats), dtype=torch.int64, device="cuda")
        out = torch.empty(4, dtype=torch.int64, device="cuda")
        for case, what in enumerate(["x - inf", "x * 0", "x + 1", "fma(x, 0*x, x)", "-x"]):
            arith[(1,)](xb, out, CASE=case, DT=dt, UB=U.get(w, tl.uint64), W=w)
            o = [f"{int(v) & ((1 << w) - 1):x}" for v in out.tolist()]
            print(f"{name:5s} {what:16s} x = [+inf, 0, nan {pats[2]:x}, -nan {pats[3] & ((1 << w) - 1):x}] -> {o}")
        for neg in (False, True):
            consts[(1,)](out, DT=dt, UB=U.get(w, tl.uint64), W=w, NEG=neg)
            print(f"{name:5s} tl.full({'-' if neg else ''}nan){'' if neg else ' '}     -> {int(out[0].item()) & ((1 << w) - 1):x}")
    xb = torch.tensor([0x7f800000, 0x3f800000, 0x7f812345, 0xffc00001 - (1 << 32)], dtype=torch.int64, device="cuda")
    out = torch.empty(4, dtype=torch.int64, device="cuda")
    for case, what in enumerate(["div_rn(0, 0)", "sqrt_rn(-|x|-1)", "x / (0*x)"]):
        divsqrt[(1,)](xb, out, CASE=case)
        print(f"f32   {what:16s} -> {[f'{int(v) & 0xffffffff:x}' for v in out.tolist()]}")
    for src, dst in (("f32", "f16"), ("f32", "bf16"), ("f32", "f64"), ("f64", "f32"), ("f16", "f32"),
                     ("bf16", "f32"), ("f16", "bf16"), ("bf16", "f16")):
        sdt, sw, pats = FM[src]
        ddt, dw, _ = FM[dst]
        xb = torch.tensor(list(pats), dtype=torch.int64, device="cuda")
        cvt[(1,)](xb, out, SRC=sdt, SU=U.get(sw, tl.uint64), SW=sw, DST=ddt, DU=U.get(dw, tl.uint64), DW=dw)
        print(f"cvt {src}->{dst}: [+inf, 0, nan {pats[2] & ((1 << sw) - 1):x}, -nan {pats[3] & ((1 << sw) - 1):x}] -> "
              f"{[f'{int(v) & ((1 << dw) - 1):x}' for v in out.tolist()]}")
    for src in ("f32", "f16", "bf16"):
        for dst, ddt in (("e4m3", tl.float8e4nv), ("e5m2", tl.float8e5)):
            sdt, sw, pats = FM[src]
            xb = torch.tensor(list(pats), dtype=torch.int64, device="cuda")
            cvt[(1,)](xb, out, SRC=sdt, SU=U.get(sw, tl.uint64), SW=sw, DST=ddt, DU=tl.uint8, DW=8)
            print(f"cvt {src}->{dst}: [+inf, 0, nan, -nan] -> {[f'{int(v) & 0xff:x}' for v in out.tolist()]}")


if __name__ == "__main__":
    main()
