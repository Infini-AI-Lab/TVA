"""lr-9a: the standard float conversions that docs/rules/num.md ("Exact operations need no rule")
and `tl.cvt` (TVA/Lang/Ops.lean) take as IEEE, on the device through Triton 3.8.0.

`tl.cvt` says `x.to(dtype, fp_downcast_rounding=rnd)` between float16, bfloat16, float32 and
float64 is one rounding (`.rn`, or `.rz` from float32 into float16/bfloat16), up to NaN payload,
and widening is exact. This checks, bit for bit (NaN: any NaN):
- every float32 pattern into float16 and bfloat16, with "rtne" (default) and "rtz";
- every float16 and bfloat16 pattern into float32, float64 and each other;
- every float32 pattern into float64 and back (2^32 patterns, through float64 and back).
The reference is the exact rounding in integer arithmetic on the GPU (the input is M·2^E with an
integer M; quantum 2^(max(floor(log2|x|), emin) − m); ties to even; overflow gives ∞ under rn
and the largest finite value under rz; the sign is kept, also for results that round to zero),
the definition `Format.cvt … .rn/.rz` and `FloatV.cvtWith` state.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I cvt_std.py
"""
import sys
import torch
import triton
import triton.language as tl

FMT = {"f16": (10, -14, 15), "bf16": (7, -126, 127), "f32": (23, -126, 127)}


@triton.jit
def from32(x_ptr, hn, hz, bn, bz, d_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32, bitcast=True)
    tl.store(hn + offs, x.to(tl.float16).to(tl.int16, bitcast=True), mask=m)
    tl.store(hz + offs, x.to(tl.float16, fp_downcast_rounding="rtz").to(tl.int16, bitcast=True), mask=m)
    tl.store(bn + offs, x.to(tl.bfloat16).to(tl.int16, bitcast=True), mask=m)
    tl.store(bz + offs, x.to(tl.bfloat16, fp_downcast_rounding="rtz").to(tl.int16, bitcast=True), mask=m)
    tl.store(d_ptr + offs, x.to(tl.float64).to(tl.float32).to(tl.int32, bitcast=True), mask=m)


@triton.jit
def from16(x_ptr, f32_ptr, f64_ptr, oth_ptr, n, SRC: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    raw = tl.load(x_ptr + offs, mask=m)
    if SRC == 0:
        x = raw.to(tl.float16, bitcast=True)
        o = x.to(tl.bfloat16)
    else:
        x = raw.to(tl.bfloat16, bitcast=True)
        o = x.to(tl.float16)
    tl.store(f32_ptr + offs, x.to(tl.float32).to(tl.int32, bitcast=True), mask=m)
    tl.store(f64_ptr + offs, x.to(tl.float64).to(tl.int64, bitcast=True), mask=m)
    tl.store(oth_ptr + offs, o.to(tl.int16, bitcast=True), mask=m)


def decompose(xf):
    m, e = torch.frexp(xf.abs())
    M = torch.ldexp(m, torch.full_like(e, 53)).to(torch.int64)
    return M, e.to(torch.int64) - 53


def round_bits(xf, fmt, rz):
    """Exact rounding of finite float64 values xf into fmt: bit pattern (int64)."""
    p, emin, emax = FMT[fmt]
    bias = -emin + 1
    ebits = {"f16": 5, "bf16": 8, "f32": 8}[fmt]
    M, E = decompose(xf)
    nz = M > 0
    lg = torch.where(nz, torch.floor(torch.log2(xf.abs().clamp(min=1e-300))).to(torch.int64), torch.zeros_like(E))
    lg = torch.where(nz & (torch.ldexp(torch.ones_like(xf), lg) > xf.abs()), lg - 1, lg)
    lg = torch.where(nz & (torch.ldexp(torch.ones_like(xf), lg + 1) <= xf.abs()), lg + 1, lg)
    q = torch.maximum(lg, torch.full_like(lg, emin)) - p
    sh = q - E
    pos = sh > 0
    shc = sh.clamp(min=0, max=62)
    Q = torch.where(pos, M >> shc, M << (-sh).clamp(min=0, max=62))
    R = torch.where(pos, M - (Q << shc), torch.zeros_like(M))
    half = torch.where(pos, torch.ones_like(M) << (shc - 1).clamp(min=0), torch.zeros_like(M))
    if not rz:
        up = pos & ((R > half) | ((R == half) & (Q % 2 == 1)))
        Q = Q + up.to(torch.int64)
    carry = Q >= (1 << (p + 1))
    Q = torch.where(carry, Q >> 1, Q)
    q = torch.where(carry, q + 1, q)
    normal = Q >= (1 << p)
    expf = torch.where(normal, q + p + bias, torch.zeros_like(q))
    frac = torch.where(normal, Q - (1 << p), Q)
    mag = (expf << p) | frac
    mag = torch.where(nz, mag, torch.zeros_like(mag))
    inf = ((1 << ebits) - 1) << p
    maxf = inf - 1
    over = nz & (expf > (1 << ebits) - 2)
    mag = torch.where(over, torch.full_like(mag, maxf if rz else inf), mag)
    sign = (xf < 0) | ((xf == 0) & (torch.copysign(torch.ones_like(xf), xf) < 0))
    return mag | (sign.to(torch.int64) << (p + ebits))


def is_nan_bits(b, fmt):
    p, _, _ = FMT[fmt]
    ebits = {"f16": 5, "bf16": 8, "f32": 8}[fmt]
    e = (b >> p) & ((1 << ebits) - 1)
    return (e == (1 << ebits) - 1) & ((b & ((1 << p) - 1)) != 0)


def inf_bits(fmt, neg):
    p, _, _ = FMT[fmt]
    ebits = {"f16": 5, "bf16": 8, "f32": 8}[fmt]
    return (((1 << ebits) - 1) << p) | ((1 << (p + ebits)) if neg else 0)


def compare(name, src_vals, got, fmt, rz, stats):
    """src_vals: float64 values (NaN, inf allowed); got: int64 bit patterns in fmt."""
    nbits = {"f16": 16, "bf16": 16, "f32": 32}[fmt]
    got = got & ((1 << nbits) - 1)
    fin = torch.isfinite(src_vals)
    ref = round_bits(torch.where(fin, src_vals, torch.zeros_like(src_vals)), fmt, rz)
    nanin = torch.isnan(src_vals)
    infp = torch.isinf(src_vals) & (src_vals > 0)
    infn = torch.isinf(src_vals) & (src_vals < 0)
    bad = (fin & (got != ref)) | (nanin & ~is_nan_bits(got, fmt)) | (infp & (got != inf_bits(fmt, False))) \
        | (infn & (got != inf_bits(fmt, True)))
    s = stats.setdefault(name, [0, 0, None])
    s[0] += int(src_vals.numel())
    nb = int(bad.sum())
    s[1] += nb
    if nb and s[2] is None:
        k = int(bad.nonzero()[0])
        s[2] = (float(src_vals[k]), hex(int(got[k])), hex(int(ref[k])))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    stats = {}
    CH = 1 << 26
    for c in range(64):
        b64 = torch.arange(c * CH, (c + 1) * CH, dtype=torch.int64, device="cuda")
        bits = torch.where(b64 >= 2 ** 31, b64 - 2 ** 32, b64).to(torch.int32)
        outs = [torch.empty(CH, dtype=torch.int16, device="cuda") for _ in range(4)]
        d = torch.empty(CH, dtype=torch.int32, device="cuda")
        from32[(CH // 1024,)](bits, *outs, d, CH, B=1024)
        xf = bits.view(torch.float32).double()
        for name, o, fmt, rz in [("f32 -> f16, rtne", outs[0], "f16", False), ("f32 -> f16, rtz", outs[1], "f16", True),
                                 ("f32 -> bf16, rtne", outs[2], "bf16", False), ("f32 -> bf16, rtz", outs[3], "bf16", True)]:
            compare(name, xf, o.to(torch.int64), fmt, rz, stats)
        compare("f32 -> f64 -> f32", xf, d.to(torch.int64), "f32", False, stats)
        print(f"chunk {c}", file=sys.stderr, flush=True)
    for src, sname, tdt in [(0, "f16", torch.float16), (1, "bf16", torch.bfloat16)]:
        b64 = torch.arange(0, 1 << 16, dtype=torch.int64, device="cuda")
        bits = torch.where(b64 >= 2 ** 15, b64 - 2 ** 16, b64).to(torch.int16)
        f32 = torch.empty(1 << 16, dtype=torch.int32, device="cuda")
        f64 = torch.empty(1 << 16, dtype=torch.int64, device="cuda")
        oth = torch.empty(1 << 16, dtype=torch.int16, device="cuda")
        from16[(64,)](bits, f32, f64, oth, 1 << 16, SRC=src, B=1024)
        xf = bits.view(tdt).double()
        compare(f"{sname} -> f32", xf, f32.to(torch.int64), "f32", False, stats)
        # float64 result: compare its value with the input (exact widening), NaN to NaN
        v = f64.view(torch.float64)
        bad = ~((v == xf) & (torch.signbit(v) == torch.signbit(xf)) | (torch.isnan(v) & torch.isnan(xf)))
        s = stats.setdefault(f"{sname} -> f64", [0, 0, None]); s[0] += xf.numel(); s[1] += int(bad.sum())
        other = "bf16" if sname == "f16" else "f16"
        compare(f"{sname} -> {other}, rtne", xf, oth.to(torch.int64), other, False, stats)
    print("| conversion | inputs (every pattern) | differ from the exact rounding (NaN: not a NaN) | first difference (input, device, reference) |")
    print("|---|---|---|---|")
    for k, (n, nb, ex) in stats.items():
        print(f"| {k} | {n} | {nb} | {ex} |")


if __name__ == "__main__":
    main()
