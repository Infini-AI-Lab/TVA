"""Reviewer r-fwd, round 3 (constants of round 25: absA = 2^-22, omega0 + 66*2^-36): evaluate the claim's tolerances (Model.lean, `tolO`, `tolS`, `tolF`, `tolM`)
exactly, with Python's fractions, at the test's, the benchmark's and the largest covered sizes.
`relDot16 K = (K + 2) * 2^-22` (lean/TVA/Rules/Num/Sm90.lean)."""

from fractions import Fraction as Fr

def relDot16(K): return Fr(K + 2, 2**22)
u32 = Fr(1, 2**24)
def ybound(X): return Fr(3, 2) * X
def expErr(D, X): return (relDot16(D) + Fr(1, 2**20)) * ybound(X) + Fr(1, 2**50)
def psiP(D, X): return (1 + expErr(D, X)) * (1 + Fr(1, 2**22)) - 1
def psi16(D, X): return (1 + psiP(D, X)) * (1 + Fr(1, 2**11)) - 1
def psiL(D, X): return (1 + u32) ** 63 * (1 + psiP(D, X)) - 1
def drift(D, X): return 1 + 4 * u32 * (ybound(X) + expErr(D, X))
stepAcc = (1 + relDot16(64)) * (1 + Fr(1, 2**150)) * ((1 + Fr(1, 2**22)) * (1 + u32))
stepL = (1 + 3 * u32) * (1 + Fr(1, 2**150)) * (1 + Fr(1, 2**22))
def nblocks(N): return N // 64
def facc(D, N, X): return (1 + psi16(D, X)) * drift(D, X) * stepAcc ** nblocks(N)
def fl(D, N, X): return (1 + psiL(D, X)) * drift(D, X) * stepL ** nblocks(N)
absL = Fr(1, 2**100)
absA = Fr(1, 2**22)
omega0 = Fr(1, 2**25) + Fr(1, 2**125) + Fr(66, 2**36)
def wlow(D, X): return 1 - expErr(D, X)
def betaL(D, N, X): return (fl(D, N, X) - 1) + fl(D, N, X) * (nblocks(N) * absL) / wlow(D, X)
def denQ(D, N, X): return 1 - betaL(D, N, X)
def relQ(D, N, X): return (facc(D, N, X) - 1 + betaL(D, N, X)) / denQ(D, N, X)
def absQS(D, N, X): return facc(D, N, X) * omega0 / (wlow(D, X) * denQ(D, N, X))
def absQ0(D, N, X): return facc(D, N, X) * (nblocks(N) * absA) / (wlow(D, X) * denQ(D, N, X))
def thetaW(X): return Fr(1, 2**20) * X
lastRel = Fr(1, 2**11) * (1 + Fr(1, 2**22)) + Fr(1, 2**22)
def tolO(D, N, X): return lastRel * (1 + relQ(D, N, X)) * (1 + thetaW(X)) + relQ(D, N, X) * (1 + thetaW(X)) + thetaW(X)
def tolS(D, N, X): return absQS(D, N, X) * (1 + lastRel)
def tolF(D, N, X): return absQ0(D, N, X) * (1 + lastRel) + Fr(1, 2**11) * Fr(1, 2**148) + Fr(1, 2**25) + Fr(1, 2**148)
def tolM(D, N, X): return (1 + u32) * (Fr(17, 2**22) + 2 * betaL(D, N, X) + Fr(1, 2**21) * X + Fr(1, 2**149)) + Fr(1, 2**150)

def row(D, N, X):
    return (f"D={D:4d} N={N:6d} X={float(X):7.1f}  tolO={float(tolO(D, N, X)):.4%}  "
            f"tolS={float(tolS(D, N, X)):.3e}  tolS*N={float(tolS(D, N, X) * N):.3e}  "
            f"tolF={float(tolF(D, N, X)):.3e}  tolM={float(tolM(D, N, X)):.4f}  "
            f"expErr={float(expErr(D, X)):.4f}  stepAcc^nb-1={float(stepAcc ** nblocks(N) - 1):.4f}")

print("test (normal(0,0.5), sm_scale 0.5): X about 30")
for D in (64, 128):
    for N in (128, 1024, 4096):
        print(row(D, N, 30))
print("benchmark (randn, sm_scale 1.3): X about 150 (D=64) and 300 (D=128)")
for D, X in ((64, 150), (128, 300)):
    for N in (1024, 16384):
        print(row(D, N, X))
print("largest covered: N = 2^16, X = 2^10")
for D in (64, 128, 256):
    print(row(D, 2**16, 1024))
print("largest N with realistic X")
for D in (128, 256):
    for X in (30, 300):
        print(row(D, 2**16, X))
# where the drift alone (X small) reaches 1%: the loop factor stepAcc^nblocks
for N in (2**14, 2**15, 2**16):
    print("N", N, "stepAcc^nblocks - 1 =", float(stepAcc ** nblocks(N) - 1), " drift-only tolO at X=1 D=64:", float(tolO(64, N, 1)))

print("README table check")
for D, N, X in ((64, 128, 16), (128, 4096, 16), (128, 2**14, 256), (256, 2**16, 16), (256, 2**16, 1024)):
    print(D, N, X, f"tolO={float(tolO(D, N, X)):.3e} tolS={float(tolS(D, N, X)):.3e} tolF={float(tolF(D, N, X)):.3e} tolM={float(tolM(D, N, X)):.3e}")
