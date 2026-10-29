"""Is a float64 -> bfloat16 / float16 conversion one rounding (SEMANTICS "Float conversions")?

lr-8a, round 8. Values just above a bf16 (f16) midpoint by less than half a float32 ulp: one
rounding goes up; a rounding through float32 lands on the midpoint and ties to even (down).
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>   (from TVA's root)
"""
import struct
import torch
import triton
import triton.language as tl


@triton.jit
def k_bf16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(x_ptr + o, mask=m)
    tl.store(y_ptr + o, x.to(tl.bfloat16), mask=m)


@triton.jit
def k_f16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(x_ptr + o, mask=m)
    tl.store(y_ptr + o, x.to(tl.float16), mask=m)


@triton.jit
def k_store_bf16(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(x_ptr + o, mask=m)
    tl.store(y_ptr + o, x, mask=m)          # implicit conversion of the stored value


def one_rounding(vals, dt):
    # exact round-to-nearest-even of each double to the target's precision (normal range only):
    # torch's own CPU conversion of float64 to bfloat16/float16 rounds through float32 (printed below)
    from fractions import Fraction
    import math
    prec = 8 if dt == torch.bfloat16 else 11
    out = []
    for v in vals:
        f = Fraction(v)
        s = -1 if f < 0 else 1
        a = abs(f)
        e = math.floor(math.log2(a))
        while Fraction(2) ** e > a:
            e -= 1
        while Fraction(2) ** (e + 1) <= a:
            e += 1
        q = a / Fraction(2) ** (e - prec + 1)
        n = math.floor(q)
        r = q - n
        if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
            n += 1
        out.append(float(s * n * Fraction(2) ** (e - prec + 1)))
    return torch.tensor(out, dtype=torch.float64).to(dt)


def main():
    base = [1.0, 1.5, 3.0, 1000.0, 2.0 ** -100, 1.0 + 2.0 ** -7]
    vb, vh = [], []
    for b in base:
        # bf16 ulp at b, midpoint, plus a tiny amount below f32's resolution
        e = struct.unpack('<q', struct.pack('<d', b))[0] >> 52 & 0x7FF
        ulp_bf = 2.0 ** (e - 1023 - 7)
        ulp_h = 2.0 ** (e - 1023 - 10)
        vb += [b + ulp_bf / 2 + ulp_bf * 2.0 ** -20, -(b + ulp_bf / 2 + ulp_bf * 2.0 ** -20)]
        vh += [b + ulp_h / 2 + ulp_h * 2.0 ** -20]
    vh = [v for v in vh if abs(v) < 60000 and abs(v) > 2.0 ** -14]
    for name, k, vals, dt in [("x.to(bf16)", k_bf16, vb, torch.bfloat16), ("x.to(f16)", k_f16, vh, torch.float16),
                              ("store f64 into bf16", k_store_bf16, vb, torch.bfloat16)]:
        x = torch.tensor(vals, dtype=torch.float64, device="cuda")
        y = torch.empty(len(vals), dtype=dt, device="cuda")
        k[(1,)](x, y, len(vals), BLOCK=16)
        ref1 = one_rounding(vals, dt)
        ref2 = torch.tensor(vals, dtype=torch.float64).to(torch.float32).to(dt)
        dev = y.cpu()
        n1 = int((dev.view(torch.int16) == ref1.view(torch.int16)).sum())
        n2 = int((dev.view(torch.int16) == ref2.view(torch.int16)).sum())
        print(f"{name}: {len(vals)} values; equal to one rounding: {n1}; equal to rounding through f32: {n2}")
        for v, d, r1, r2 in zip(vals, dev.tolist(), ref1.tolist(), ref2.tolist()):
            print(f"   {v!r:>26}  device {d!r:<22} one-rounding {r1!r:<22} via-f32 {r2!r}")
    print("torch CPU check: 1+2^-8+2^-28 -> bf16 =", torch.tensor([1 + 2.0 ** -8 + 2.0 ** -28], dtype=torch.float64).to(torch.bfloat16).item())
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
