"""lr-20e: `DivFull` (float32 `/`) and `DivRn P .binary64` (float64 `/`) on the H200, sampled adversarially.

`DivFull`: for finite `x`, `y`, `y ≠ 0`, `|x/y| ≤ 2^126`, the result is finite and within `2^-22·|x/y| + 2^-148`
of `x/y`. Pairs: random bit patterns; divisors in `[2^126, maxVal]` (where a plain reciprocal underflows) with
dividends that keep `|x/y|` in range; subnormal dividends and divisors; quotients near `2^126` and in the
subnormal range; divisor mantissas just below a power of two. Reference: float64 `x/y` (exact to 2^-53, margin
2^-50·|q| added).
`DivRn P .binary64`: `x / y` on float64 equals IEEE division, up to NaN payload; the reference is the host's
float64 division (numpy on x86-64, IEEE), on random patterns, subnormals and special values.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I div_probe.py
"""
import sys

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def divk(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x / y, mask=m)


def f32_from_bits(b):
    return b.to(torch.int32).view(torch.float32)


def rand_bits(n, g):
    return torch.randint(-(1 << 31), 1 << 31, (n,), generator=g, dtype=torch.int64).cuda()


def pairs32(g, n):
    out = []
    out.append((f32_from_bits(rand_bits(n, g)), f32_from_bits(rand_bits(n, g))))            # random
    big = torch.rand(n, generator=g).cuda() * (float.fromhex("0x1.fffffep+127") - 2.0 ** 126) + 2.0 ** 126
    xs = big.float() * torch.exp2(torch.randint(-250, 0, (n,), generator=g).cuda().float() * 0.5).float()
    out.append((xs.float(), big.float()))                                                    # huge divisors
    sub = f32_from_bits(torch.randint(1, 1 << 23, (n,), generator=g, dtype=torch.int64).cuda())
    out.append((sub, f32_from_bits(rand_bits(n, g))))                                        # subnormal x
    out.append((f32_from_bits(rand_bits(n, g)), sub))                                        # subnormal y
    q = torch.exp2(torch.empty(n).uniform_(120, 126, generator=g)).cuda().float()
    y = torch.exp2(torch.empty(n).uniform_(-30, 0, generator=g)).cuda().float()
    out.append(((q * y).float(), y))                                                         # |q| near 2^126
    q = torch.exp2(torch.empty(n).uniform_(-150, -120, generator=g)).cuda().float()
    y = torch.exp2(torch.empty(n).uniform_(0, 100, generator=g)).cuda().float()
    out.append(((q * y).float(), y))                                                         # subnormal q
    mant = (torch.randint(0x7f0000, 0x800000, (n,), generator=g, dtype=torch.int64) |
            (torch.randint(1, 254, (n,), generator=g, dtype=torch.int64) << 23)).cuda()
    out.append((f32_from_bits(rand_bits(n, g)), f32_from_bits(mant)))                        # y below 2^k
    return out


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    g = torch.Generator().manual_seed(20261028)
    n = 1 << 22
    checked = viol = 0
    worst = 0.0
    for rep in range(8):
        for x, y in pairs32(g, n):
            o = torch.empty_like(x)
            divk[(triton.cdiv(n, 1024),)](x, y, o, n, B=1024)
            x64, y64, o64 = x.double(), y.double(), o.double()
            q = x64 / y64
            sel = torch.isfinite(x64) & torch.isfinite(y64) & (y64 != 0) & (q.abs() <= 2.0 ** 126)
            bound = 2.0 ** -22 * q.abs() + 2.0 ** -148
            err = (o64 - q).abs()
            bad = sel & (~torch.isfinite(o64) | (err > bound + 2.0 ** -50 * q.abs()))
            checked += int(sel.sum())
            viol += int(bad.sum())
            r = torch.where(sel & torch.isfinite(o64), err / bound, torch.zeros_like(err))
            worst = max(worst, float(r.max()))
    print(f"DivFull: {checked} pairs in its domain, {viol} outside the fact, largest error / bound {worst:.4f}")
    # float64
    rng = np.random.default_rng(20261028)
    m = 1 << 22
    xb = rng.integers(0, 1 << 64, m, dtype=np.uint64)
    yb = rng.integers(0, 1 << 64, m, dtype=np.uint64)
    sub = rng.integers(1, 1 << 52, m // 4, dtype=np.uint64)
    xb[: m // 4] = sub                                                 # subnormal dividends
    yb[m // 4: m // 2] = sub                                           # subnormal divisors
    yb[m // 2: m // 2 + m // 8] = (np.uint64(0x3ff) << np.uint64(52)) | (yb[m // 2: m // 2 + m // 8] & np.uint64((1 << 20) - 1))
    specials = np.array([0, 1 << 63, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000, 1,
                         0x7fefffffffffffff, 0x0010000000000000, 0x3ff0000000000000], dtype=np.uint64)
    sx, sy = np.meshgrid(specials, specials)
    xb = np.concatenate([xb, sx.ravel()])
    yb = np.concatenate([yb, sy.ravel()])
    x = xb.view(np.float64)
    y = yb.view(np.float64)
    with np.errstate(all="ignore"):
        ref = x / y
    xt = torch.from_numpy(x.copy()).cuda()
    yt = torch.from_numpy(y.copy()).cuda()
    o = torch.empty_like(xt)
    divk[(triton.cdiv(len(x), 1024),)](xt, yt, o, len(x), B=1024)
    got = o.cpu().numpy()
    same = (got.view(np.uint64) == ref.view(np.uint64)) | (np.isnan(got) & np.isnan(ref))
    print(f"DivRn P .binary64: {len(x)} pairs, {int((~same).sum())} differ from IEEE division (up to NaN payload)")
    print(f"outside the facts in all: {viol + int((~same).sum())}")


if __name__ == "__main__":
    main()
