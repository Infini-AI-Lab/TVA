"""lr-20e: `DotScaledE4m3` and `DotScaledE4m3E2m1` (docs/rules/num.md) on the H200, sampled.

`DotScaledAcc P cfg scaledDomBF16 e8m0Dom relDot16 absDot16`: per output element whose scaled values
(element · 2^(scale bits − 127)) are `0` or of a normal `bfloat16` magnitude, with scale bits in `1 … 254` and a
finite accumulator, with room in float32, the result is finite and within
`relDot16 K·(|acc| + Σ|va·vb|) + absDot16 K` of `acc + Σ va·vb` (`relDot16 K = (K+2)·2^-22`,
`absDot16 K = (K+2)·2^-126`). Random fp8/fp4 elements, random scales over a wide range, and rows with extreme
scales (`2^±100`, the smallest normal scaled values), both `fast_math` values; the exact sum is `math.fsum`
of the exact products (its error, at most `2^-53` of the sum, is added to the bound's margin).

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I dot_scaled_probe.py
"""
import math
import random
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def dsk(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        RF: tl.constexpr, FM: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    rg = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    sa = tl.load(as_ptr + rm[:, None] * (K // 32) + rg[None, :])
    if RF == "e2m1":
        rk2 = tl.arange(0, K // 2)
        b = tl.load(b_ptr + rk2[:, None] * N + rn[None, :])
    else:
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    sb = tl.load(bs_ptr + rn[:, None] * (K // 32) + rg[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    r = tl.dot_scaled(a, sa, "e4m3", b, sb, RF, acc=c, fast_math=FM)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], r)


def e4m3_val(p):
    s = -1.0 if p & 0x80 else 1.0
    e, m = (p >> 3) & 0xF, p & 7
    if e == 15 and m == 7:
        return None
    return s * (m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10))


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_val(p):
    return (-1.0 if p & 8 else 1.0) * E2M1[p & 7]


BF16_MAX = float.fromhex("0x1.fep+127")
F32_MAX = float.fromhex("0x1.fffffep+127")


def in_dom(v):
    return v == 0 or (2.0 ** -126 <= abs(v) <= BF16_MAX)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    rnd = random.Random(20261028)
    M = N = 32
    total_checked = total_viol = 0
    worst = 0.0
    for RF in ("e4m3", "e2m1"):
        for K in (64, 256):
            for FM in (False, True):
                for trial in range(6):
                    fin = [p for p in range(256) if e4m3_val(p) is not None]
                    A = [[rnd.choice(fin) for _ in range(K)] for _ in range(M)]
                    if trial == 0:      # extreme scales
                        SA = [[rnd.choice([227, 27, 133, 7 + 6]) for _ in range(K // 32)] for _ in range(M)]
                    else:
                        SA = [[rnd.randint(100, 154) for _ in range(K // 32)] for _ in range(M)]
                    if RF == "e2m1":
                        Bv = [[rnd.randint(0, 15) for _ in range(N)] for _ in range(K)]
                        Bp = [[(Bv[2 * k][n] & 15) | ((Bv[2 * k + 1][n] & 15) << 4) for n in range(N)]
                              for k in range(K // 2)]
                        bval = e2m1_val
                    else:
                        Bv = [[rnd.choice(fin) for _ in range(N)] for _ in range(K)]
                        Bp = Bv
                        bval = e4m3_val
                    if trial == 0:
                        SB = [[rnd.choice([27, 227, 121, 140]) for _ in range(K // 32)] for _ in range(N)]
                    else:
                        SB = [[rnd.randint(100, 154) for _ in range(K // 32)] for _ in range(N)]
                    C = [[rnd.choice([0.0, 1.0, -3.5, rnd.uniform(-1e3, 1e3)]) for _ in range(N)] for _ in range(M)]
                    dev = lambda x, dt: torch.tensor(x, dtype=dt, device="cuda")
                    a = dev(A, torch.int32).to(torch.uint8)
                    sa = dev(SA, torch.int32).to(torch.uint8)
                    b = dev(Bp, torch.int32).to(torch.uint8)
                    sb = dev(SB, torch.int32).to(torch.uint8)
                    c = dev(C, torch.float32)
                    o = torch.empty((M, N), dtype=torch.float32, device="cuda")
                    dsk[(1,)](a, sa, b, sb, c, o, M=M, N=N, K=K, RF=RF, FM=FM, num_warps=4)
                    torch.cuda.synchronize()
                    O = o.cpu().tolist()
                    Cf = c.cpu().tolist()
                    rel, ab = (K + 2) * 2.0 ** -22, (K + 2) * 2.0 ** -126
                    for i in range(M):
                        va = [e4m3_val(A[i][k]) * 2.0 ** (SA[i][k // 32] - 127) for k in range(K)]
                        if not all(in_dom(v) for v in va):
                            continue
                        for n in range(N):
                            vb = [bval(Bv[k][n]) * 2.0 ** (SB[n][k // 32] - 127) for k in range(K)]
                            if not all(in_dom(v) for v in vb):
                                continue
                            prods = [x * y for x, y in zip(va, vb)]
                            acc = Cf[i][n]
                            mag = abs(acc) + math.fsum(abs(p) for p in prods)
                            bound = rel * mag + ab
                            if mag + bound > F32_MAX:
                                continue
                            exact = math.fsum([acc] + prods)
                            r = O[i][n]
                            err = abs(r - exact) if math.isfinite(r) else math.inf
                            total_checked += 1
                            if err > bound + 2.0 ** -52 * mag:
                                total_viol += 1
                                if total_viol <= 5:
                                    print(f"  OUTSIDE: {RF} K={K} fast_math={FM}: r={r!r} exact={exact!r} "
                                          f"err/bound={err / bound:.4g}")
                            worst = max(worst, err / bound)
                print(f"lhs e4m3, rhs {RF}, K={K}, fast_math={FM}: checked so far {total_checked}, "
                      f"outside {total_viol}")
    print(f"DotScaledE4m3 / DotScaledE4m3E2m1: {total_checked} elements in the facts' domain, {total_viol} outside, "
          f"largest error / bound {worst:.4f}")


if __name__ == "__main__":
    main()
