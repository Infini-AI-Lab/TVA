"""Targeted probes for library issue L126: one product with a subnormal 16-bit operand and a large partner,
and 15 products with normal operands just below the unit of a window that follows the subnormal's exponent
field read as the smallest normal's. Each case is also run with normal operands of the same product.
K = 16, row 0 of a and column 0 of b, on wgmma (64x64 block, 4 warps) and mma.sync (16x64, 1 warp)."""
from fractions import Fraction as Fr
import torch, triton, triton.language as tl

@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))

def run(dt, av, bv, M, nw, K=16):
    a = torch.zeros(M, K, dtype=torch.float64); b = torch.zeros(K, 64, dtype=torch.float64)
    a[0] = torch.tensor(av, dtype=torch.float64); b[:, 0] = torch.tensor(bv, dtype=torch.float64)
    a2, b2 = a.to(dt).cuda(), b.to(dt).cuda()
    c = torch.empty(M, 64, dtype=torch.float32, device="cuda")
    h = dot_kernel[(1,)](a2, b2, c, M, 64, K, num_warps=nw)
    path = "wgmma" if "wgmma" in h.asm["ptx"] else "mma.sync"
    A = [Fr(float(x)) for x in a2[0].double().cpu()]; B = [Fr(float(x)) for x in b2[:, 0].double().cpu()]
    prods = [x * y for x, y in zip(A, B)]; exact = sum(prods); mag = sum(abs(p) for p in prods)
    r = Fr(float(c[0, 0])); err = abs(r - exact)
    bound = (K + 2) * Fr(2) ** -22 * mag + (K + 2) * Fr(2) ** -126   # DotF16 / DotBF16
    return path, float(r), float(exact), float(err / bound)

CASES = [  # (dtype, label, big a, big b, small a, small b)
    (torch.float16, "float16, subnormal a = 2^-23, b = 2^14", 2.0 ** -23, 2.0 ** 14, (2 - 2 ** -10) * 2 ** -12, 2.0 ** -12),
    (torch.float16, "float16, normal a = 2^-14, b = 2^5 (same product)", 2.0 ** -14, 2.0 ** 5, (2 - 2 ** -10) * 2 ** -12, 2.0 ** -12),
    (torch.bfloat16, "bfloat16, subnormal a = 2^-133, b = 2^124", 2.0 ** -133, 2.0 ** 124, (2 - 2 ** -7) * 2 ** -13, 2.0 ** -13),
    (torch.bfloat16, "bfloat16, normal a = 2^-126, b = 2^117 (same product)", 2.0 ** -126, 2.0 ** 117, (2 - 2 ** -7) * 2 ** -13, 2.0 ** -13),
]
if __name__ == "__main__":
    print("case | k of the large product | path | result | exact | error / DotF16's bound")
    for dt, label, ba, bb, sa, sb in CASES:
        for M, nw in ((64, 4), (16, 1)):
            for pos in (0, 7, 15):
                av = [sa] * 16; bv = [sb] * 16; av[pos] = ba; bv[pos] = bb
                path, r, ex, ratio = run(dt, av, bv, M, nw)
                print(f"{label} | {pos} | {path} | {r!r} | {ex!r} | {ratio:.3f}")
