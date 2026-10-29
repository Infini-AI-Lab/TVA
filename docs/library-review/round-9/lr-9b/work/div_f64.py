"""lr-9b, library review round 9 (L4, rules/num.md): `DivRn P .binary64` rests on the lowering
(`div.rn.f64`); this checks the device against IEEE division on the host.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-9/lr-9b/work/div_f64.py

The host's float64 division (numpy, x86-64 SSE2) is IEEE round-to-nearest-even with subnormals.
Inputs: random bit patterns, operands and quotients near the subnormal range, and divisors with
every low-mantissa pattern near 1. Compared bit for bit (NaN results compared as NaN).
"""
import numpy as np
import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def div(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x / y, mask=m)


def run(xb, yb):
    x = torch.from_numpy(xb.view(np.float64)).to(DEV)
    y = torch.from_numpy(yb.view(np.float64)).to(DEV)
    o = torch.empty_like(x)
    div[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), B=1024)
    got = o.cpu().numpy()
    with np.errstate(all="ignore"):
        ref = xb.view(np.float64) / yb.view(np.float64)
    same = (got.view(np.uint64) == ref.view(np.uint64)) | (np.isnan(got) & np.isnan(ref))
    return x.numel(), int((~same).sum())


def main():
    rng = np.random.default_rng(9)
    print("| inputs | pairs | differ from IEEE |")
    print("|---|---|---|")
    tot = 0
    n = 1 << 24
    for name in ("random patterns", "subnormal dividends", "subnormal quotients", "divisors near 1"):
        P = D = 0
        for _ in range(4):
            if name == "random patterns":
                xb = rng.integers(0, 2 ** 64, n, dtype=np.uint64)
                yb = rng.integers(0, 2 ** 64, n, dtype=np.uint64)
            elif name == "subnormal dividends":
                xb = rng.integers(0, 2 ** 52, n, dtype=np.uint64) | (rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63))
                yb = rng.integers(0, 2 ** 64, n, dtype=np.uint64)
            elif name == "subnormal quotients":
                x = rng.uniform(1, 2, n) * 2.0 ** rng.integers(-1022, -960, n)
                y = rng.uniform(1, 2, n) * 2.0 ** rng.integers(40, 100, n)
                xb, yb = x.view(np.uint64), y.view(np.uint64)
            else:
                y = (np.uint64(0x3FF0000000000000) | rng.integers(0, 2 ** 20, n, dtype=np.uint64)) ^ np.uint64(0)
                y = y | (rng.integers(0, 2, n, dtype=np.uint64) * np.uint64(0x000FFFFFFFF00000))
                xb = rng.integers(0x3FF0000000000000, 0x4000000000000000, n, dtype=np.uint64)
                yb = y
            p, d = run(xb, yb)
            P += p; D += d
        tot += D
        print(f"| {name} | {P} | {D} |")
    print(f"\nTotal differing from IEEE: {tot}")


if __name__ == "__main__":
    main()
