"""TVA library evidence (library issues L4, L101): conversions of infinities and NaNs into fp8.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-cvt8/specials.py
Output: evidence/num-cvt8/specials.out (README.md, "Infinities and NaNs").

run.py compares the finite inputs; of the others it reports only float32's `+inf`, `-inf` and
the canonical NaN. This script converts, on the device, every non-finite bit pattern of
`float32` (both infinities and all 16,777,214 NaN patterns), `float16` and `bfloat16` (both
infinities and every NaN pattern) into `float8e4nv` and `float8e5` with the default rounding,
and checks what `CvtSat` (docs/rules/num.md) says of them: an infinity gives the largest finite
value of its sign, and a NaN gives a NaN of the fp8 format (the fact holds up to NaN payload).
"""
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


# source: (kernel SRC, bits, exponent bits, fraction bits, int dtype)
SRC = {"float32": (0, 32, 8, 23, torch.int32), "float16": (1, 16, 5, 10, torch.int16),
       "bfloat16": (2, 16, 8, 7, torch.int16)}
# fp8: (largest finite magnitude bits, is-NaN test on the 8-bit pattern)
FP8 = {"float8e4nv": (0x7e, lambda b: (b & 0x7f) == 0x7f),
       "float8e5": (0x7b, lambda b: ((b & 0x7c) == 0x7c) & ((b & 0x03) != 0))}


def patterns(nbits, ebits, fbits):
    """Every non-finite bit pattern (exponent field all ones), as signed integers of nbits."""
    frac = torch.arange(0, 1 << fbits, dtype=torch.int64, device="cuda")
    out = []
    for sign in (0, 1):
        u = (sign << (nbits - 1)) | (((1 << ebits) - 1) << fbits) | frac
        out.append(u)
    u = torch.cat(out)
    return torch.where(u >= (1 << (nbits - 1)), u - (1 << nbits), u)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("\n| from | to | input | patterns | results (bits, count) | as CvtSat says |")
    print("|---|---|---|---|---|---|")
    bad_all = 0
    for sname, (src, nbits, ebits, fbits, idt) in SRC.items():
        u = patterns(nbits, ebits, fbits)
        n = u.numel()
        bits = u.to(idt)
        e4 = torch.empty(n, dtype=torch.int8, device="cuda")
        e5 = torch.empty(n, dtype=torch.int8, device="cuda")
        B = 1024
        cvt[(triton.cdiv(n, B),)](bits, e4, e5, n, SRC=src, B=B)
        frac = u & ((1 << fbits) - 1)
        neg = u < 0
        for fname, out in (("float8e4nv", e4), ("float8e5", e5)):
            maxbits, isnan = FP8[fname]
            got = out.to(torch.int64) & 0xFF
            for kind, sel, want in (
                    ("+inf", (frac == 0) & ~neg, lambda g: g == maxbits),
                    ("-inf", (frac == 0) & neg, lambda g: g == (0x80 | maxbits)),
                    ("NaN, sign 0", (frac != 0) & ~neg, isnan),
                    ("NaN, sign 1", (frac != 0) & neg, isnan)):
                g = got[sel]
                vals, cnt = torch.unique(g, return_counts=True)
                ok = bool(want(g).all())
                bad_all += 0 if ok else 1
                res = ", ".join(f"`{v:02x}` × {c}" for v, c in zip(vals.tolist(), cnt.tolist()))
                print(f"| {sname} | {fname} | {kind} | {int(sel.sum())} | {res} | {'yes' if ok else 'NO'} |")
    print(f"\nrows not as CvtSat says: {bad_all}")


if __name__ == "__main__":
    main()
