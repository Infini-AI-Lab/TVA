"""Count float16 subnormal operands in torch.randn data of the test's shapes (CPU generator;
the device's generator differs, but the distribution is the same), and how much they raise
A = dotMag above the test's normaliser (|A| @ |B|)[i, j]."""
import math
import torch

torch.manual_seed(0)
SUB = 2.0 ** -14  # smallest normal float16
p = math.erf(SUB / math.sqrt(2))  # P(|X| < 2^-14), X ~ N(0, 1)
print(f"P(|x| < 2^-14) for N(0,1): {p:.3e}")
for M, N, K in [(512, 512, 512), (1000, 1000, 1000), (3072, 4096, 1024), (257, 129, 4097),
                (64, 64, 64), (8192, 8192, 4096)]:
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    sa = ((a != 0) & (a.abs() < SUB)).sum().item()
    sb = ((b != 0) & (b.abs() < SUB)).sum().item()
    line = f"M={M} N={N} K={K}: subnormal a={sa} b={sb} (expected {p*M*K:.1f}, {p*K*N:.1f})"
    if M * N * K <= 2 ** 31:
        ad, bd = a.double(), b.double()
        lift = lambda x: torch.where(x == 0, torch.zeros_like(x), torch.clamp(x.abs(), min=SUB))
        A = lift(ad) @ lift(bd)
        norm = ad.abs() @ bd.abs()
        line += f"; max A/(|A|@|B|) - 1 = {((A / norm) - 1).max().item():.2e}"
    print(line)
