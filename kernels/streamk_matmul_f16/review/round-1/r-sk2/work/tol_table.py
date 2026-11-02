# Evaluate the claim's tolerance tol K A / A (A = 1) and its absolute part, exactly, at the
# README's K values; and the schedule facts of the test shapes.
from fractions import Fraction as F

def bound(rho, alpha, T, B):
    return ((1 + rho) ** T - 1) * B + T * (1 + rho) ** T * alpha

def steps(K):
    return (K + 63) // 64

rho = F(66, 2 ** 22)
alpha = F(66, 2 ** 126)
ra = F(1, 2 ** 23)
aa = F(1, 2 ** 124)

def tol(K, A):
    td = bound(rho, alpha, steps(K), A)
    return td + bound(ra, aa, steps(K), A + td)

print("K, T, rel part tol(K,1)-tol(K,0), abs part tol(K,0), tolDot rel")
for K in [64, 512, 1000, 1024, 4096, 4097, 8192]:
    rel = tol(K, F(1)) - tol(K, F(0))
    ab = tol(K, F(0))
    tdrel = bound(rho, alpha, steps(K), F(1)) - bound(rho, alpha, steps(K), F(0))
    print(K, steps(K), "%.4e" % float(rel), "%.3e" % float(ab), "%.4e" % float(tdrel))

# bf16 / fp16 rounding: largest relative error of round-to-nearest
print("bf16 max rel rounding err", float(F(1, 2**8) / (1 + F(1, 2**8))))
print("fp16 max rel rounding err", float(F(1, 2**11) / (1 + F(1, 2**11))))
