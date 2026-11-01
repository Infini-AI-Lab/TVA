"""Does a subnormal 16-bit operand set the tl.dot window on sm_90, as a subnormal fp8 operand does
(library issues L125, L126)? Each trial is one diagonal element c[i, i] of a K-product tl.dot: one
product a[i, k0]*b[k0, i] with a subnormal operand and a large partner, and K - 1 same-sign products
with all fraction bits set just below the unit the window would have if the subnormal's exponent field
read as the smallest normal's. The result is compared with the exact sum and with DotF16's bound
(K+2)*2^-22*(|acc| + sum|a*b|) + (K+2)*2^-126."""
import random
from fractions import Fraction as Fr
import torch, triton, triton.language as tl

@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))

FMT = {"float16": (torch.float16, -14, 10, 15), "bfloat16": (torch.bfloat16, -126, 7, 127)}  # min normal exp, fraction bits, max exp

def trial(rng, fmt, K, side):
    _, emin, frac, emax = FMT[fmt]
    t = rng.randint(1, frac)                    # subnormal 2^(emin - t)
    sub = Fr(2) ** (emin - t) * (1 + Fr(rng.randrange(0, 2 ** max(frac - t, 0)), 2 ** max(frac - t, 1)) if frac - t > 0 else 1)
    big_e = rng.randint(max(emin + 2, -2), min(emax - 1, 15))
    big = Fr(2) ** big_e
    lifted = Fr(2) ** (emin + big_e)            # the product's magnitude if the subnormal read as 2^emin
    unit = lifted * Fr(2) ** -23                # the window's unit then (23 bits kept)
    allset = 2 - Fr(1, 2 ** frac)               # significand with every fraction bit set
    small = []
    for _ in range(K - 1):                      # x*y just below the unit: x = allset*2^ex, y = 2^ey
        ex = rng.randint(-8, 0)
        small.append((allset * Fr(2) ** ex, unit / 2 / Fr(2) ** ex))
    k0 = rng.randrange(K)
    av, bv = [], []
    j = 0
    for k in range(K):
        if k == k0:
            av.append(sub if side == "a" else big); bv.append(big if side == "a" else sub)
        else:
            x, y = small[j]; j += 1
            av.append(x); bv.append(y)
    return av, bv

def run(fmt, K, M, nw, side, rng, reps):
    dt = FMT[fmt][0]
    worst = (0.0, None); kept = 0; n = 0
    for _ in range(reps):
        a = torch.zeros(M, K, dtype=torch.float64); b = torch.zeros(K, M, dtype=torch.float64)
        rows = []
        for i in range(M):
            av, bv = trial(rng, fmt, K, side)
            a[i] = torch.tensor([float(x) for x in av], dtype=torch.float64)
            b[:, i] = torch.tensor([float(y) for y in bv], dtype=torch.float64)
        a2, b2 = a.to(dt).cuda(), b.to(dt).cuda()
        c = torch.empty(M, M, dtype=torch.float32, device="cuda")
        h = dot_kernel[(1,)](a2, b2, c, M, M, K, num_warps=nw)
        path = "wgmma" if "wgmma" in h.asm["ptx"] else ("mma.sync" if "mma.sync" in h.asm["ptx"] else "other")
        A = a2.double().cpu(); B = b2.double().cpu(); C = c.double().cpu()
        for i in range(M):
            prods = [Fr(float(A[i, k])) * Fr(float(B[k, i])) for k in range(K)]
            exact = sum(prods); mag = sum(abs(p) for p in prods)
            err = abs(Fr(float(C[i, i])) - exact)
            bound = (K + 2) * Fr(2) ** -22 * mag + (K + 2) * Fr(2) ** -126
            r = float(err / bound); n += 1
            if err * 2 ** 10 < mag * Fr(2) ** -23 * 32:  # the small products were not lost whole
                kept += 1
            if r > worst[0]:
                worst = (r, (float(A[i, :].abs().min()), float(err), float(mag)))
    return path, n, kept, worst

if __name__ == "__main__":
    rng = random.Random(20261101)
    print("fmt K M warps side path trials small-kept worst-ratio-to-DotF16 (min|a|, err, mag)")
    for fmt in ("float16", "bfloat16"):
        for K in (16, 32, 64):
            for M, nw in ((64, 4), (16, 1)):
                for side in ("a", "b"):
                    path, n, kept, worst = run(fmt, K, M, nw, side, rng, reps=4)
                    print(fmt, K, M, nw, side, path, n, kept, f"{worst[0]:.6f}", worst[1])
