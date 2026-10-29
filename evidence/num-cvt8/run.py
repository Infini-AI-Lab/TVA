"""TVA library evidence (library issue L4): conversions into fp8 on sm_90.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-cvt8/run.py
Output: evidence/num-cvt8/out.txt (README.md says what it shows).

For every input bit pattern of `float32` (all 2^32, in chunks), `float16` and `bfloat16` (all
2^16), the device computes `x.to(tl.float8e4nv)` and `x.to(tl.float8e5)` (the default rounding,
`"rtne"`). Each result is compared, bit for bit, with round-to-nearest-even into the fp8 format,
computed exactly in integer arithmetic on the GPU: the fact `CvtRn` (docs/rules/num.md) for finite
inputs with |x| <= maxVal of the fp8 format (448 for float8e4nv, 57344 for float8e5). Outside
that range the script reports what the device gives (saturation), which the library does not
state.
"""
import re
import torch
import triton
import triton.language as tl


@triton.jit
def cvt(x_ptr, e4_ptr, e5_ptr, n, SRC: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    raw = tl.load(x_ptr + offs, mask=m)
    if SRC == 0:
        x = raw.to(tl.float32, bitcast=True)
    elif SRC == 1:
        x = raw.to(tl.float16, bitcast=True)
    else:
        x = raw.to(tl.bfloat16, bitcast=True)
    tl.store(e4_ptr + offs, x.to(tl.float8e4nv).to(tl.int8, bitcast=True), mask=m)
    tl.store(e5_ptr + offs, x.to(tl.float8e5).to(tl.int8, bitcast=True), mask=m)


# (mantissa bits, minimum normal exponent, maxVal, exponent bias)
FMT = {"float8e4nv": (3, -6, 448.0, 7), "float8e5": (2, -14, 57344.0, 15)}


def decompose(xf):
    """|x| = M * 2^E exactly, M < 2^24 an integer (from the float64 value of a float32/16 input)."""
    m, e = torch.frexp(xf.abs())                     # |x| = m 2^e, m in [1/2, 1)
    M = torch.ldexp(m, torch.full_like(e, 53)).to(torch.int64)   # exact: at most 24 significant bits
    E = e.to(torch.int64) - 53
    return M, E


def rne_bits(xf, fmt):
    """Round-to-nearest-even of the finite values xf (float64, |x| <= maxVal) into fp8: bit pattern."""
    p, emin, _, bias = FMT[fmt]
    M, E = decompose(xf)
    nz = M > 0
    lg = torch.where(nz, torch.floor(torch.log2(xf.abs().clamp(min=1e-300))).to(torch.int64), torch.zeros_like(E))
    # exact floor(log2 |x|): fix possible off-by-one of log2
    lg = torch.where(nz & (torch.ldexp(torch.ones_like(xf), lg) > xf.abs()), lg - 1, lg)
    lg = torch.where(nz & (torch.ldexp(torch.ones_like(xf), lg + 1) <= xf.abs()), lg + 1, lg)
    q = torch.maximum(lg, torch.full_like(lg, emin)) - p          # quantum exponent
    sh = q - E                                                     # value / 2^q = M / 2^sh
    pos = sh > 0
    shc = sh.clamp(min=0, max=62)
    Q = torch.where(pos, M >> shc, M << (-sh).clamp(min=0, max=62))
    R = torch.where(pos, M - (Q << shc), torch.zeros_like(M))
    half = torch.where(pos, torch.ones_like(M) << (shc - 1).clamp(min=0), torch.zeros_like(M))
    up = pos & ((R > half) | ((R == half) & (Q % 2 == 1)))
    Q = Q + up.to(torch.int64)                                    # |result| = Q 2^q
    # encode: Q in [2^p, 2^(p+1)] normal (Q = 2^(p+1) carries into the next binade)
    carry = Q >= (1 << (p + 1))
    Q = torch.where(carry, Q >> 1, Q)
    q = torch.where(carry, q + 1, q)
    normal = Q >= (1 << p)
    expf = torch.where(normal, q + p + bias, torch.zeros_like(q))
    frac = torch.where(normal, Q - (1 << p), Q)
    mag = (expf << p) | frac
    mag = torch.where(nz, mag, torch.zeros_like(mag))
    sign = (xf < 0) | ((xf == 0) & (torch.copysign(torch.ones_like(xf), xf) < 0))
    return (mag | (sign.to(torch.int64) << 7)).to(torch.int64)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    rows = []
    for src, sname, nbits, tdt, idt in [(0, "float32", 32, torch.float32, torch.int32),
                                        (1, "float16", 16, torch.float16, torch.int16),
                                        (2, "bfloat16", 16, torch.bfloat16, torch.int16)]:
        total = 1 << nbits
        CH = min(total, 1 << 26)
        stats = {f: dict(inrange=0, diff=0, ex=None, sat=0, satbits=set(), outr=0) for f in FMT}
        for lo in range(0, total, CH):
            b64 = torch.arange(lo, lo + CH, dtype=torch.int64, device="cuda")
            half = 1 << (nbits - 1)
            bits = torch.where(b64 >= half, b64 - total, b64).to(idt)
            e4 = torch.empty(CH, dtype=torch.int8, device="cuda")
            e5 = torch.empty(CH, dtype=torch.int8, device="cuda")
            cvt[(CH // 1024,)](bits, e4, e5, CH, SRC=src, B=1024)
            xf = bits.view(tdt).to(torch.float64)
            fin = torch.isfinite(xf)
            for fmt, out in (("float8e4nv", e4), ("float8e5", e5)):
                maxv = FMT[fmt][2]
                inr = fin & (xf.abs() <= maxv)
                got = out.to(torch.int64) & 0xFF
                ref = rne_bits(torch.where(inr, xf, torch.zeros_like(xf)), fmt)
                bad = inr & (got != ref)
                s = stats[fmt]
                s["inrange"] += int(inr.sum())
                s["diff"] += int(bad.sum())
                if s["ex"] is None and bool(bad.any()):
                    k = int(bad.nonzero()[0])
                    s["ex"] = (float(xf[k]), int(got[k]), int(ref[k]))
                outr = fin & (xf.abs() > maxv)
                s["outr"] += int(outr.sum())
                if bool(outr.any()):
                    s["satbits"] |= set(torch.unique(got[outr]).tolist())
        for fmt in FMT:
            s = stats[fmt]
            rows.append((sname, fmt, s["inrange"], s["diff"], s["ex"], s["outr"], sorted(s["satbits"])))
    print("\n| from | to | finite inputs with abs(x) <= maxVal | differ from round-to-nearest-even | first difference | finite inputs above maxVal | their results (bits) |")
    print("|---|---|---|---|---|---|---|")
    for sname, fmt, inr, diff, ex, outr, sat in rows:
        print(f"| {sname} | {fmt} | {inr} | {diff} | {ex} | {outr} | {', '.join(f'`{b:02x}`' for b in sat)} |")
    # special inputs
    print("\n## Special inputs from float32 (result bits)")
    sp = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 448.0, 464.0, 57344.0, 61440.0],
                      dtype=torch.float32, device="cuda")
    e4 = torch.empty(9, dtype=torch.int8, device="cuda"); e5 = torch.empty(9, dtype=torch.int8, device="cuda")
    cvt[(1,)](sp.view(torch.int32), e4, e5, 9, SRC=0, B=16)
    names = ["+inf", "-inf", "nan", "+0", "-0", "448", "464", "57344", "61440"]
    print("| to | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for fmt, o in (("float8e4nv", e4), ("float8e5", e5)):
        print(f"| {fmt} | " + " | ".join(f"`{b & 0xff:02x}`" for b in o.tolist()) + " |")
    print("\n## Instructions in the PTX")
    for src, sname, idt in [(0, "float32", torch.int32), (1, "float16", torch.int16), (2, "bfloat16", torch.int16)]:
        xx = torch.zeros(16, dtype=idt, device="cuda")
        a = torch.empty(16, dtype=torch.int8, device="cuda"); b = torch.empty(16, dtype=torch.int8, device="cuda")
        h = cvt[(1,)](xx, a, b, 16, SRC=src, B=16)
        ops = sorted(set(re.findall(r"\bcvt\.[a-z0-9.]+", h.asm["ptx"])))
        print(f"- from `{sname}`: " + ", ".join(f"`{o}`" for o in ops))


if __name__ == "__main__":
    main()
