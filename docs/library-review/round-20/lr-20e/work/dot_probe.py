"""lr-20e: adversarial checks of the `tl.dot` facts (docs/rules/num.md, `TVA.Rules.Num.Sm90`) on the H200.

Inputs that sit just below the accumulation window of each instruction (products just under the last bit
kept relative to the accumulator), against the fact's bound for the element:
`|r − (acc + Σ a·b)| ≤ rel K·(|acc| + Σ|a·b|) + abs K`, the exact value computed in float64 (every input is a
power of two times a short significand, so the sums are exact).
- `DotF16`/`DotBF16` (`relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`);
- `DotE4m3`/`DotE5m2` (`relDot8`, `absDot8`, `maxImpreciseAcc = 2^30`, the default);
- `DotF32Ieee` (`(K+1)·2^-23`, `(K+1)·2^-149`), with products and accumulators below float32's normal range
  (a flush to zero would exceed `(K+1)·2^-149`).
Each row `i` of `a` holds one value along `K`, `b` one value, the accumulator one value per row: one
combination per row. Both instruction paths: `M = 64`, 4 warps (`wgmma`) and `M = 16`, 1 warp (`mma.sync`).

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I dot_probe.py [nofusion]
With `nofusion`, every launch passes `enable_fp_fusion=False` (`Launch.fusion = false`, which scope 4 admits).
"""
import math
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def dk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, IEEE: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    if IEEE:
        r = tl.dot(a, b, c, input_precision="ieee")
    else:
        r = tl.dot(a, b, c)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], r)


FUSION = not (len(sys.argv) > 1 and sys.argv[1] == "nofusion")


def rel16(K):
    return (K + 2) * 2.0 ** -22, (K + 2) * 2.0 ** -126


def rel8(K):
    ch = (K + 31) // 32
    return (1 + 2.0 ** -7) ** ch - 1, ch * (1 + 2.0 ** -7) ** ch * 2.0 ** -126


def relieee(K):
    return (K + 1) * 2.0 ** -23, (K + 1) * 2.0 ** -149


def combos16():
    """(acc, a, b) per row: products just below 2^-23, 2^-22, … of the accumulator, with signs."""
    out = []
    for acc in (1.0, 2.0 - 2.0 ** -23, -1.0, 1.5, 2.0 ** 20 - 1):
        for j in (22, 23, 24, 25):
            a = (2 - 2.0 ** -10) * 2.0 ** -12
            b = 2.0 ** -(j - 12) * (abs(acc) if acc != 1.5 else 1)
            out.append((acc, a, b))
    return out


def combos8(fmt):
    out = []
    # float8e4nv: normals 2^-6 .. 448 (3 fraction bits); float8e5: 2^-14 .. 57344 (2 bits)
    sig = 1.875 if fmt == "e4m3" else 1.75
    for acc in (1.0, 2.0 - 2.0 ** -23, -1.0, 1.5):
        for j in (10, 11, 12, 13, 14, 15, 16):
            a = sig * 2.0 ** -4
            b = 2.0 ** -(j - 4)
            out.append((acc, a, b))
    return out


def combosieee():
    out = []
    for acc in (0.0, 2.0 ** -140, -(2.0 ** -145), 2.0 ** -126, 1.0):
        for a, b in ((2.0 ** -70, 2.0 ** -70), (2.0 ** -75, 2.0 ** -70), (2.0 ** -60, 2.0 ** -88), (1.5, 2.0 ** -25)):
            out.append((acc, a, b))
    return out


def run(dt, combos, Ms, Ks, relf, ieee=False):
    worst, viol, n = 0.0, 0, 0
    N = 16
    for M, warps in Ms:
        for K in Ks:
            rows = [combos[i % len(combos)] for i in range(M)]
            a = torch.tensor([[r[1]] * K for r in rows], dtype=torch.float32, device="cuda").to(dt)
            bv = [r[2] for r in rows]
            # b: one value per column n; row i uses column i % N, so pair row i with its own b via a per-row
            # scale: put b's value into column 0 and read the element (i, 0) only for row-specific b.
            b = torch.zeros((K, N), dtype=torch.float32, device="cuda")
            c = torch.tensor([[r[0]] * N for r in rows], dtype=torch.float32, device="cuda")
            out = []
            for col_b in sorted(set(bv)):
                b[:, :] = col_b
                bb = b.to(dt)
                o = torch.empty((M, N), dtype=torch.float32, device="cuda")
                dk[(1,)](a, bb, c, o, M=M, N=N, K=K, IEEE=ieee, num_warps=warps, enable_fp_fusion=FUSION)
                torch.cuda.synchronize()
                out.append((col_b, o.cpu()))
            rel, ab = relf(K)
            av = a.float().cpu()
            for i, (acc, _, bval) in enumerate(rows):
                o = next(t for v, t in out if v == bval)
                aa = float(av[i, 0])
                bq = float(torch.tensor([bval]).to(dt).float())
                exact = acc + K * aa * bq
                mag = abs(acc) + K * abs(aa * bq)
                bound = rel * mag + ab
                r = float(o[i, 0])
                err = abs(r - exact) if math.isfinite(r) else math.inf
                n += 1
                if err > bound:
                    viol += 1
                    print(f"  OUTSIDE: {dt} M={M} K={K} acc={acc!r} a={aa!r} b={bq!r}: r={r!r} exact={exact!r} "
                          f"err/bound={err / bound:.4f}")
                worst = max(worst, err / bound)
    return n, viol, worst


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}; enable_fp_fusion={FUSION}")
    paths = [(64, 4), (16, 1)]
    total = 0
    for name, dt, combos, Ks, relf, ieee in (
            ("DotF16", torch.float16, combos16(), (16, 32, 64, 256, 1024), rel16, False),
            ("DotBF16", torch.bfloat16, combos16(), (16, 32, 64, 256, 1024), rel16, False),
            ("DotE4m3", torch.float8_e4m3fn, combos8("e4m3"), (32, 64, 128, 512, 1024), rel8, False),
            ("DotE5m2", torch.float8_e5m2, combos8("e5m2"), (32, 64, 128, 512, 1024), rel8, False),
            ("DotF32Ieee", torch.float32, combosieee(), (16, 32, 64), relieee, True)):
        n, v, w = run(dt, combos, paths, Ks, relf, ieee)
        total += v
        print(f"{name}: {n} elements (both paths, K in {Ks}), {v} outside the fact, largest error / bound {w:.4f}")
    print(f"outside the facts in all: {total}")


if __name__ == "__main__":
    main()
