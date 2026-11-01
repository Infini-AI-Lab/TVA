"""lr-25b: does a zero operand with a large partner set the tensor core's alignment window?
A zero is encoded with exponent bits 0, like a subnormal, which the tensor core reads at the smallest
normal's field. If a zero also did, a product 0 * big would set the window while liftMag 0 = 0 in the facts.
K = 16 (16-bit) or 32 (fp8), row 0 of a, column 0 of b; both paths. The control `sub` puts the
format's smallest subnormal in place of the zero: there the other products lose bits, as the window
read at the smallest normal's field makes them, so the case would show a zero read that way."""
from fractions import Fraction as Fr
import torch, triton, triton.language as tl

@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))

def run(dt, av, bv, M, nw, K):
    a = torch.zeros(M, K, dtype=torch.float32); b = torch.zeros(K, 64, dtype=torch.float32)
    a[0] = torch.tensor(av, dtype=torch.float32); b[:, 0] = torch.tensor(bv, dtype=torch.float32)
    a2, b2 = a.to(dt).cuda(), b.to(dt).cuda()
    assert torch.equal(a2.float().cpu(), a) and torch.equal(b2.float().cpu(), b)
    c = torch.empty(M, 64, dtype=torch.float32, device="cuda")
    h = dot_kernel[(1,)](a2, b2, c, M, 64, K, num_warps=nw)
    path = "wgmma" if "wgmma" in h.asm["ptx"] else "mma.sync"
    A = [Fr(float(x)) for x in a2[0].float().cpu()]; B = [Fr(float(x)) for x in b2[:, 0].float().cpu()]
    exact = sum(x * y for x, y in zip(A, B))
    return path, Fr(float(c[0, 0])), exact

if __name__ == "__main__":
    print("dtype | zero side | big partner | path | device | exact | error")
    cases = [
        (torch.float16, 16, (2 - 2 ** -10) * 2 ** -12, 2.0 ** -12, 2.0 ** 15),
        (torch.bfloat16, 16, (2 - 2 ** -7) * 2 ** -12, 2.0 ** -12, 2.0 ** 127),
        (torch.float8_e4m3fn, 32, 1.875 * 2 ** -6, 2.0 ** -6, 448.0),
        (torch.float8_e5m2, 32, 1.75 * 2 ** -8, 2.0 ** -8, 57344.0),
    ]
    SUB = {torch.float16: 2.0 ** -24, torch.bfloat16: 2.0 ** -133, torch.float8_e4m3fn: 2.0 ** -9,
           torch.float8_e5m2: 2.0 ** -16}
    for dt, K, sa, sb, big in cases:
        for zside in ("a", "b", "both", "negzero-a", "none", "sub"):
            for M, nw in ((64, 4), (16, 1)):
                av = [sa] * K; bv = [sb] * K
                if zside == "a": av[0], bv[0] = 0.0, big
                elif zside == "b": av[0], bv[0] = big, 0.0
                elif zside == "both": av[0], bv[0] = 0.0, 0.0
                elif zside == "negzero-a": av[0], bv[0] = -0.0, big
                elif zside == "sub": av[0], bv[0] = SUB[dt], big
                else: av[0], bv[0] = 0.0, 0.0
                path, r, ex = run(dt, av, bv, M, nw, K)
                print(f"{str(dt).split('.')[-1]} | {zside} | {big} | {path} | {float(r)!r} | {float(ex)!r} | {float(abs(r - ex)):.3e}")
