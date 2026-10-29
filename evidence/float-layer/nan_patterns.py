"""TVA library evidence, part F of evidence/float-layer (library issue L99): which NaN patterns the
device's float operations produce, against the library's canonical NaN `FloatV.nan`
(`Format.nanMag`: positive, every exponent and fraction bit set).

Run from this folder:  CUDA_VISIBLE_DEVICES=0 python -I nan_patterns.py
Output: nan_patterns.out (README.md, part F).

For each standard format, a kernel computes, on operands that give a NaN (a NaN operand with a
payload, of either sign, or an invalid operation: `inf - inf`, `0 * inf`, `0 / 0`, `fma(inf, 0, x)`,
`sqrt(-1)`), the operations the library takes as exact (`+ - *`, `tl.fma`, `tl.div_rn`, `/`,
`tl.sqrt_rn`, `tl.maximum`/`tl.minimum` with `propagate_nan=ALL`, `-x`, `tl.abs`; `tl.div_rn` and
`tl.sqrt_rn` take `float32` only, so the other formats run `/` and `tl.sqrt` there; on the
16-bit formats these compute in `float32` (`tl.sqrt` of the operand widened) and convert back) and the
conversions between the standard formats and into fp8, and prints every NaN result's pattern.
"""
import struct

import torch
import triton
import triton.language as tl

FMT = {  # name: (dtype, torch dtype, width, +inf, a payload NaN, its negation with another payload)
    "f16": (tl.float16, torch.float16, 16, 0x7c00, 0x7d23, 0xfe01),
    "bf16": (tl.bfloat16, torch.bfloat16, 16, 0x7f80, 0x7fa3, 0xffc1),
    "f32": (tl.float32, torch.float32, 32, 0x7f800000, 0x7f812345, 0xffc00001),
    "f64": (tl.float64, torch.float64, 64, 0x7ff0000000000000, 0x7ff0000000012345, 0xfff8000000000001),
}
UINT = {16: tl.uint16, 32: tl.uint32, 64: tl.uint64}
SINT = {16: torch.int16, 32: torch.int32, 64: torch.int64}
OPS = ["inf - inf", "0 * inf", "0 / 0 (div_rn)", "0 / 0 (/)", "fma(inf, 0, 1)", "sqrt_rn(-1)",
       "nan + 1", "-nan + 1", "nan * 1", "fma(nan, 1, 1)", "div_rn(nan, 1)", "nan / 1", "sqrt_rn(nan)",
       "maximum(nan, 1, ALL)", "minimum(-nan, 1, ALL)", "-nan", "-(-nan)", "abs(-nan)"]


@triton.jit
def ld(xb_ptr, k: tl.constexpr, DT: tl.constexpr, UB: tl.constexpr, W: tl.constexpr):
    xb = tl.load(xb_ptr + k)
    if W == 64:
        x = xb.to(tl.float64, bitcast=True)
    else:
        x = xb.to(UB).to(DT, bitcast=True)
    return x


@triton.jit
def ops_kernel(xb_ptr, out_ptr, DT: tl.constexpr, UB: tl.constexpr, W: tl.constexpr):
    # xb = [+inf, 0, 1, -1, nan, -nan] as bit patterns, loaded at run time
    inf = ld(xb_ptr, 0, DT, UB, W)
    zero = ld(xb_ptr, 1, DT, UB, W)
    one = ld(xb_ptr, 2, DT, UB, W)
    mone = ld(xb_ptr, 3, DT, UB, W)
    nan = ld(xb_ptr, 4, DT, UB, W)
    nnan = ld(xb_ptr, 5, DT, UB, W)
    for k in tl.static_range(18):
        if k == 0:
            r = inf - inf
        elif k == 1:
            r = zero * inf
        elif k == 2:
            if W == 32:
                r = tl.div_rn(zero, zero)
            else:
                r = zero / zero
        elif k == 3:
            r = zero / zero
        elif k == 4:
            r = tl.fma(inf, zero, one)
        elif k == 5:
            if W == 32:
                r = tl.sqrt_rn(mone)
            elif W == 64:
                r = tl.sqrt(mone)
            else:
                r = tl.sqrt(mone.to(tl.float32))
        elif k == 6:
            r = nan + one
        elif k == 7:
            r = nnan + one
        elif k == 8:
            r = nan * one
        elif k == 9:
            r = tl.fma(nan, one, one)
        elif k == 10:
            if W == 32:
                r = tl.div_rn(nan, one)
            else:
                r = nan / one
        elif k == 11:
            r = nan / one
        elif k == 12:
            if W == 32:
                r = tl.sqrt_rn(nan)
            elif W == 64:
                r = tl.sqrt(nan)
            else:
                r = tl.sqrt(nan.to(tl.float32))
        elif k == 13:
            r = tl.maximum(nan, one, propagate_nan=tl.PropagateNan.ALL)
        elif k == 14:
            r = tl.minimum(nnan, one, propagate_nan=tl.PropagateNan.ALL)
        elif k == 15:
            r = -nnan
        elif k == 16:
            r = -(-nnan)
        else:
            r = tl.abs(nnan)
        r = r.to(DT)   # `/` and `tl.sqrt` on 16-bit formats compute in float32
        if W == 64:
            tl.store(out_ptr + k, r.to(tl.int64, bitcast=True))
        else:
            tl.store(out_ptr + k, r.to(UB, bitcast=True).to(tl.int64))


@triton.jit
def cvt_kernel(xb_ptr, out_ptr, SRC: tl.constexpr, SU: tl.constexpr, SW: tl.constexpr, DST: tl.constexpr,
               DU: tl.constexpr, DW: tl.constexpr):
    i = tl.arange(0, 2)
    xb = tl.load(xb_ptr + i)
    if SW == 64:
        x = xb.to(tl.float64, bitcast=True)
    else:
        x = xb.to(SU).to(SRC, bitcast=True)
    r = x.to(DST)
    if DW == 64:
        tl.store(out_ptr + i, r.to(tl.int64, bitcast=True))
    elif DW == 8:
        tl.store(out_ptr + i, r.to(tl.uint8, bitcast=True).to(tl.int64))
    else:
        tl.store(out_ptr + i, r.to(DU, bitcast=True).to(tl.int64))


def canonical(w):
    return {8: 0x7f, 16: 0x7fff, 32: 0x7fffffff, 64: 0x7fffffffffffffff}[w]


def is_nan(name, v):
    e, m = {"f16": (5, 10), "bf16": (8, 7), "f32": (8, 23), "f64": (11, 52), "e4m3": (4, 3), "e5m2": (5, 2)}[name]
    if name == "e4m3":
        return v & 0x7f == 0x7f
    return (v >> m) & ((1 << e) - 1) == (1 << e) - 1 and v & ((1 << m) - 1) != 0


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (dt, td, w, inf, nan, nnan) in FMT.items():
        one = struct.unpack("<Q", struct.pack("<d", 1.0))[0] if w == 64 else \
            (struct.unpack("<I", struct.pack("<f", 1.0))[0] if w == 32 else (0x3c00 if name == "f16" else 0x3f80))
        sign = 1 << (w - 1)
        pats = [inf, 0, one, one | sign, nan, nnan]
        xb = torch.tensor([p - (1 << 64) if p >= 1 << 63 else p for p in pats], dtype=torch.int64, device="cuda")
        out = torch.zeros(len(OPS), dtype=torch.int64, device="cuda")
        ops_kernel[(1,)](xb, out, DT=dt, UB=UINT.get(w, tl.uint64), W=w)
        torch.cuda.synchronize()
        res = [v & ((1 << w) - 1) for v in out.tolist()]
        noncanon = [(op, r) for op, r in zip(OPS, res) if is_nan(name, r) and r != canonical(w)]
        notnan = [op for op, r in zip(OPS, res) if not is_nan(name, r)]
        print(f"{name}: operands nan {nan:x}, -nan {nnan:x}")
        print("  " + ", ".join(f"{op} -> {r:x}" for op, r in zip(OPS, res)))
        print(f"  NaN results other than the canonical {canonical(w):x}: "
              + (", ".join(f"{op} ({r:x})" for op, r in noncanon) or "none")
              + (f"; not NaN: {', '.join(notnan)}" if notnan else ""))
    print("conversions of [nan, -nan] (the payload NaNs above)")
    dst_all = list(FMT) + ["e4m3", "e5m2"]
    for src, (sdt, _, sw, _, nan, nnan) in FMT.items():
        for dst in dst_all:
            if dst == src or (dst in ("e4m3", "e5m2") and src == "f64"):
                continue
            if dst in FMT:
                ddt, _, dw = FMT[dst][0], None, FMT[dst][2]
                du = UINT.get(dw, tl.uint64)
            else:
                ddt, dw, du = (tl.float8e4nv if dst == "e4m3" else tl.float8e5), 8, tl.uint8
            xb = torch.tensor([p - (1 << 64) if p >= 1 << 63 else p for p in (nan, nnan)], dtype=torch.int64,
                              device="cuda")
            out = torch.zeros(2, dtype=torch.int64, device="cuda")
            cvt_kernel[(1,)](xb, out, SRC=sdt, SU=UINT.get(sw, tl.uint64), SW=sw, DST=ddt, DU=du, DW=dw)
            torch.cuda.synchronize()
            res = [v & ((1 << dw) - 1) for v in out.tolist()]
            tag = "canonical" if all(r == canonical(dw) for r in res) else \
                ("NaN, not canonical" if all(is_nan(dst, r) for r in res) else "NOT NaN")
            print(f"  {src} -> {dst}: {res[0]:x}, {res[1]:x} ({tag})")


if __name__ == "__main__":
    main()
