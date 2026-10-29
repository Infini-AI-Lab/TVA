"""lr-9a: the "exact operations" of docs/rules/num.md (`+`, `-`, `*`, `tl.div_rn`, `tl.sqrt_rn`)
on float16, bfloat16 and float32, against IEEE rounding to nearest even, on sampled operands
(random bit patterns, subnormals, near-overflow), bit for bit up to NaN payload.

The reference is the operation in float64 followed by one rounding into the format (exact
integer rounding, as in cvt_std.py). Rounding twice, first to float64, is innocuous for +, -, *,
/ and sqrt into a format of p bits when 53 >= 2p + 2 (p = 11, 8, 24 here), so the reference is
the correctly rounded result.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I exact_ops.py
"""
import os
import sys
sys.dont_write_bytecode = True  # import nothing into a cache next to TVA's files
import torch
import triton
import triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvt_std import round_bits, is_nan_bits  # noqa: E402

torch.manual_seed(930)


@triton.jit
def ops(x_ptr, y_ptr, a_ptr, s_ptr, m_ptr, d_ptr, q_ptr, n, SRC: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    msk = offs < n
    xr = tl.load(x_ptr + offs, mask=msk)
    yr = tl.load(y_ptr + offs, mask=msk)
    if SRC == 0:
        x = xr.to(tl.float16, bitcast=True); y = yr.to(tl.float16, bitcast=True); it = tl.int16
    elif SRC == 1:
        x = xr.to(tl.bfloat16, bitcast=True); y = yr.to(tl.bfloat16, bitcast=True); it = tl.int16
    else:
        x = xr.to(tl.float32, bitcast=True); y = yr.to(tl.float32, bitcast=True); it = tl.int32
    tl.store(a_ptr + offs, (x + y).to(it, bitcast=True), mask=msk)
    tl.store(s_ptr + offs, (x - y).to(it, bitcast=True), mask=msk)
    tl.store(m_ptr + offs, (x * y).to(it, bitcast=True), mask=msk)
    if SRC == 2:
        tl.store(d_ptr + offs, tl.div_rn(x, y).to(it, bitcast=True), mask=msk)
        tl.store(q_ptr + offs, tl.sqrt_rn(x).to(it, bitcast=True), mask=msk)


def gen(fmt, nbits, kind, n):
    if kind == "random":
        b = torch.randint(0, 2 ** nbits, (n,), device="cuda", dtype=torch.int64)
    elif kind == "subnormal":
        p = {"f16": 10, "bf16": 7, "f32": 23}[fmt]
        b = torch.randint(0, 2 ** (p + 2), (n,), device="cuda", dtype=torch.int64)   # subnormal and small normal
        b = b | (torch.randint(0, 2, (n,), device="cuda", dtype=torch.int64) << (nbits - 1))
    else:  # near the top of the range
        top = {"f16": 0x7c00, "bf16": 0x7f80, "f32": 0x7f800000}[fmt]
        b = top - torch.randint(1, 2 ** (nbits - 6 if fmt != "bf16" else 9), (n,), device="cuda", dtype=torch.int64)
        b = b | (torch.randint(0, 2, (n,), device="cuda", dtype=torch.int64) << (nbits - 1))
    return b


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| format | operation | pairs (random, subnormal, near-overflow) | differ from IEEE rounding to nearest even (NaN: not a NaN) | first difference (x, y, device, reference) |")
    print("|---|---|---|---|---|")
    n = 1 << 24
    for src, fmt, nbits, idt, tdt in [(0, "f16", 16, torch.int16, torch.float16), (1, "bf16", 16, torch.int16, torch.bfloat16),
                                      (2, "f32", 32, torch.int32, torch.float32)]:
        names = ["+", "-", "*"] + (["div_rn", "sqrt_rn"] if src == 2 else [])
        stats = {k: [0, 0, None] for k in names}
        for kind in ["random", "subnormal", "edge"]:
            for _ in range(4):
                xb = gen(fmt, nbits, kind, n); yb = gen(fmt, nbits, kind, n)
                half = 2 ** (nbits - 1)
                xi = torch.where(xb >= half, xb - 2 ** nbits, xb).to(idt); yi = torch.where(yb >= half, yb - 2 ** nbits, yb).to(idt)
                outs = [torch.zeros(n, dtype=idt, device="cuda") for _ in range(5)]
                ops[(n // 1024,)](xi, yi, *outs, n, SRC=src, B=1024)
                x = xi.view(tdt).double(); y = yi.view(tdt).double()
                with torch.no_grad():
                    refs = {"+": x + y, "-": x - y, "*": x * y, "div_rn": x / y, "sqrt_rn": torch.sqrt(x)}
                for k, o in zip(names, outs):
                    r = refs[k]
                    got = o.to(torch.int64) & (2 ** nbits - 1)
                    fin = torch.isfinite(r)
                    rb = round_bits(torch.where(fin, r, torch.zeros_like(r)), fmt, False)
                    infb = {"f16": 0x7c00, "bf16": 0x7f80, "f32": 0x7f800000}[fmt]
                    sgn = 1 << (nbits - 1)
                    exp_inf = torch.where(r > 0, torch.full_like(got, infb), torch.full_like(got, infb | sgn))
                    bad = (fin & (got != rb)) | (torch.isnan(r) & ~is_nan_bits(got, fmt)) | (torch.isinf(r) & (got != exp_inf))
                    s = stats[k]
                    s[0] += n
                    nb = int(bad.sum())
                    s[1] += nb
                    if nb and s[2] is None:
                        i = int(bad.nonzero()[0])
                        s[2] = (float(x[i]), float(y[i]), hex(int(got[i])), hex(int(rb[i])))
        for k in names:
            print(f"| {fmt} | {k} | {stats[k][0]} | {stats[k][1]} | {stats[k][2]} |")


if __name__ == "__main__":
    main()
