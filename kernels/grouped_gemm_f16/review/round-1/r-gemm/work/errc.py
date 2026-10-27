from fractions import Fraction as F
import math
u = F(1, 2**24); eta = F(1, 2**150)
def rel(K): return (K+2)*F(1,2**22)
def ab(K): return (K+2)*F(1,2**126)
def rho(BK): return rel(BK) + 3*u*(1+rel(BK))
def alpha(BK): return ab(BK)*(1+3*u) + 4*eta
def errc_rel(BK, K):
    T = K//BK; r = float(rho(BK))
    return (1+r)**T - 1
for BK in [16,32,64,128]:
    r = float(rho(BK))
    Tmax = math.floor(0.5/r)
    Kmax = Tmax*BK
    print(f"BK={BK} rho={r:.3e} Tmax={Tmax} Kmax={Kmax} errC/S at Kmax={errc_rel(BK,Kmax):.3f}")
    for K in [1024, 4096, 8192, 16384, 65536]:
        T = K//BK
        e = errc_rel(BK, K)
        print(f"   K={K} T={T} errC/S={e:.3e}  1/T={1/T:.3e}  dropped-step-of-positive-data-within-bound={1/T <= e}")
# fp16 output resolution relative: 2^-11
print("fp16 half-ulp rel", 2**-11, 2**-12)
