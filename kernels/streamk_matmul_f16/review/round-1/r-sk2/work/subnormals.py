# Does torch.randn in float16 give subnormal operands (|x| < 2^-14, x != 0) at the test's sizes?
# CPU generator (the test runs on the GPU with another generator; the rate is the same law).
import math
import torch

torch.manual_seed(0)
p = math.erf(2 ** -14 / math.sqrt(2))  # P(|x| < 2^-14) for a standard normal
for M, N, K in [(512, 512, 512), (1000, 1000, 1000), (3072, 4096, 1024), (257, 129, 4097),
                (64, 64, 64), (8192, 8192, 4096)]:
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    sa = int(((a != 0) & (a.abs() < 2 ** -14)).sum())
    sb = int(((b != 0) & (b.abs() < 2 ** -14)).sum())
    print(f"M={M} N={N} K={K}: subnormal a {sa}, b {sb}; expected about {p*M*K:.1f}, {p*K*N:.1f}")
