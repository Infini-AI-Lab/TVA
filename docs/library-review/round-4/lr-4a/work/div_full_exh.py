"""lr-4a: float32 `/` (div.full.f32) against DivFull, every mantissa of the divisor.

The error of a reciprocal-based division depends mostly on the divisor's mantissa, so this script
takes every one of the 2^23 mantissas of y, at the divisor exponents where div.full.f32 scales
(|y| >= 2^126, subnormal y) and at ordinary ones, each with 16 dividends x = y * r (r random in
[0.25, 2), or [0.25, 1) where y * r could overflow). It reports pairs outside
2^-22*|x/y| + 2^-148, and the largest error relative to that bound.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/div_full_exh.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_div(x, y, o, n, B: tl.constexpr):
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    m = j < n
    tl.store(o + j, tl.load(x + j, mask=m) / tl.load(y + j, mask=m), mask=m)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    B = 1024
    gen = torch.Generator(device="cuda")
    gen.manual_seed(23)
    mant = torch.arange(1 << 23, dtype=torch.int64, device="cuda")
    total_bad = 0
    overall = 0.0
    for name, ebits in [("subnormal y (every pattern)", 0), ("y in [2^-126, 2^-125)", 1), ("y in [1, 2)", 127),
                        ("y in [2^125, 2^126)", 252), ("y in [2^126, 2^127)", 253), ("y in [2^127, 2^128)", 254)]:
        ybits = (ebits << 23) | mant
        y = ybits.to(torch.int32).view(torch.float32)
        worst = 0.0
        bad = 0
        for rep in range(16):
            hi = 1.0 if ebits == 254 else 2.0
            r = 0.25 + (hi - 0.25) * torch.rand(1 << 23, generator=gen, device="cuda", dtype=torch.float64)
            x = (y.double() * r).float()
            ok = torch.isfinite(x) & (x != 0) & (y != 0)
            o = torch.empty_like(y)
            n = y.numel()
            k_div[(n // B,)](x, y, o, n, B=B)
            q = x.double() / y.double()
            ok &= q.abs() <= 2.0 ** 126
            err = (o.double() - q).abs()
            bound = 2.0 ** -22 * q.abs() + 2.0 ** -148
            ratio = torch.where(ok, err / bound, torch.zeros_like(err))
            bad += int((ok & (~torch.isfinite(o) | (ratio > 1 + 2.0 ** -40))).sum())
            worst = max(worst, float(ratio.max()))
        total_bad += bad
        overall = max(overall, worst)
        print(f"{name}: 2^23 divisors x 16 dividends; outside the bound {bad}; largest error / bound {worst:.4f}", flush=True)
    print(f"all: outside the bound {total_bad}; largest error / bound {overall:.4f}")


if __name__ == "__main__":
    main()
