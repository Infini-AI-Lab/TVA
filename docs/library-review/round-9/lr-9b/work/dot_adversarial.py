"""lr-9b, library review round 9 (L4): adversarial tests of the tl.dot facts.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-9/lr-9b/work/dot_adversarial.py

Cases the evidence (evidence/num-dot) does not name explicitly:
  S  subnormal operands times large ones (does the tensor core keep subnormal inputs?);
  W  the truncation worst case: an accumulator 1 and products just below the window, so each
     instruction may drop every product (ratio near 1/2 of the fact's bound if the window is as
     the evidence says);
  Z  products whose exact value is subnormal in float32, with the ieee float32 dot.
Every element of the output is checked against the fact (exact reference with fractions).
"""
from fractions import Fraction
import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    o = tl.dot(a, b, c, input_precision=PREC)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float8e4nv": torch.float8_e4m3fn,
      "float8e5": torch.float8_e5m2, "float32": torch.float32}
MAXF32 = Fraction((2 ** 24 - 1) * 2 ** 104)


def bound(dt, K):
    if dt == "float32":
        return Fraction(K + 1, 2 ** 23), Fraction(K + 1, 2 ** 149)
    if dt in ("float16", "bfloat16"):
        return Fraction(K + 2, 2 ** 22), Fraction(K + 2, 2 ** 126)
    n = (K + 31) // 32
    g = (1 + Fraction(1, 2 ** 7)) ** n
    return g - 1, n * g * Fraction(1, 2 ** 126)


def run(dt, a, b, c, nw, prec):
    M, K = a.shape
    N = b.shape[1]
    o = torch.empty(M, N, dtype=torch.float32, device=DEV)
    h = dotk[(1,)](a, b, c, o, M, N, K, prec, num_warps=nw)
    ptx = h.asm["ptx"]
    path = "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "fma")
    return o, path


def check(name, dt, a, b, c, nw, prec="tf32"):
    o, path = run(dt, a, b, c, nw, prec)
    M, K = a.shape
    N = b.shape[1]
    rel, ab = bound(dt, K)
    A = [[Fraction(float(x)) for x in row] for row in a.float().cpu().tolist()]
    B = [[Fraction(float(x)) for x in row] for row in b.float().cpu().tolist()]
    C = [[Fraction(float(x)) for x in row] for row in c.cpu().tolist()]
    O = o.cpu().tolist()
    worst, bad, room = Fraction(0), 0, 0
    for i in range(M):
        for j in range(N):
            ex = C[i][j] + sum(A[i][k] * B[k][j] for k in range(K))
            mag = abs(C[i][j]) + sum(abs(A[i][k] * B[k][j]) for k in range(K))
            bd = rel * mag + ab
            if mag + bd > MAXF32:
                continue
            room += 1
            r = O[i][j]
            if r != r or r in (float("inf"), float("-inf")):
                bad += 1
                continue
            err = abs(Fraction(r) - ex)
            if err > bd:
                bad += 1
            if bd > 0 and err / bd > worst:
                worst = err / bd
    print(f"| {name} | {dt} | {M}x{N}x{K}, {nw} warps | {path} | {room} | {bad} | {float(worst):.4f} |")


def main():
    torch.manual_seed(9)
    print("| case | dtype | shape | path | elements with room | outside the fact | largest error / bound |")
    print("|---|---|---|---|---|---|---|")
    shapes = [(16, 16, 1), (64, 64, 4)]
    # S: subnormal times large
    sub = {"float16": (2.0 ** -24, 2.0 ** -15, 2.0 ** 15), "bfloat16": (2.0 ** -133, 2.0 ** -127, 2.0 ** 100),
           "float8e4nv": (2.0 ** -9, 2.0 ** -7, 448.0), "float8e5": (2.0 ** -16, 2.0 ** -15, 57344.0)}
    for dt, (lo, hi, big) in sub.items():
        for K in (32, 256):
            for (M, N, nw) in shapes:
                # a: subnormal magnitudes (random in [lo, hi]) with random sign; b: big times random sign
                t = torch.rand(M, K, device=DEV, dtype=torch.float64)
                a = (lo + (hi - lo) * t) * torch.where(torch.rand(M, K, device=DEV) < 0.5, -1.0, 1.0).double()
                a = a.to(torch.float32).to(DT[dt])
                b = (torch.full((K, N), big, device=DEV) * torch.where(torch.rand(K, N, device=DEV) < 0.5, -1.0, 1.0)).to(DT[dt])
                c = torch.zeros(M, N, device=DEV, dtype=torch.float32)
                check("S subnormal x large", dt, a, b, c, nw)
                # one row of plain values in a, the rest subnormal, so the products differ in size
                a2 = a.clone()
                a2[:, 0] = torch.ones(M, device=DEV).to(DT[dt])
                check("S subnormal x large, one unit term", dt, a2, b, c, nw)
    # W: accumulator 1, products just below 2^-23 (16-bit) or 2^-13 (fp8)
    for dt in ("float16", "bfloat16"):
        for K in (16, 64, 512, 2048):
            for (M, N, nw) in shapes:
                if K > 512 and M == 64:
                    continue
                av = 2.0 ** -12 * (2 - 2.0 ** -7)  # representable in both; product with 2^-12 is just below 2^-23
                a = torch.full((M, K), av, device=DEV).to(DT[dt])
                b = torch.full((K, N), 2.0 ** -12, device=DEV).to(DT[dt])
                c = torch.ones(M, N, device=DEV, dtype=torch.float32)
                check("W window, acc 1", dt, a, b, c, nw)
                c2 = -torch.ones(M, N, device=DEV, dtype=torch.float32)
                check("W window, acc -1", dt, a, b, c2, nw)
    for dt in ("float8e4nv", "float8e5"):
        for K in (32, 256, 1024):
            for (M, N, nw) in shapes:
                if dt == "float8e4nv":
                    av, bv = 2.0 ** -7 * 1.75, 2.0 ** -7  # e4m3: 1.75*2^-7 is subnormal-free? (min normal 2^-6) -> product 1.75*2^-14 < 2^-13
                else:
                    av, bv = 2.0 ** -7 * 1.75, 2.0 ** -7
                a = torch.full((M, K), av, device=DEV).to(DT[dt])
                b = torch.full((K, N), bv, device=DEV).to(DT[dt])
                c = torch.ones(M, N, device=DEV, dtype=torch.float32)
                check("W window, acc 1", dt, a, b, c, nw)
    # Z: ieee float32 dot with products in float32's subnormal range
    for K in (16, 64):
        for (M, N, nw) in shapes:
            a = (torch.rand(M, K, device=DEV) * 2.0 ** -60 + 2.0 ** -61) * torch.where(torch.rand(M, K, device=DEV) < 0.5, -1.0, 1.0)
            b = (torch.rand(K, N, device=DEV) * 2.0 ** -70 + 2.0 ** -71)
            c = torch.zeros(M, N, device=DEV, dtype=torch.float32)
            check("Z subnormal products", "float32", a.float(), b.float(), c, nw, "ieee")


if __name__ == "__main__":
    main()
