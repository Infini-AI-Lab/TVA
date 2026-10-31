"""Reviewer r-fwd2: evaluate the claim's tolerances (AttnFwd.Model, "The tolerances") exactly,
with the definitions transcribed from the Lean, at the host's sizes and at the largest covered
ones. Run: python3 -I tol.py"""
from fractions import Fraction as Q

u32 = Q(1, 2**24)


def relDot16(K):
    return (K + 2) * Q(1, 2**22)


def ybound(X):
    return Q(3, 2) * X


def expErr(D, X):
    return (relDot16(D) + Q(1, 2**20)) * ybound(X) + Q(1, 2**50)


def psiP(D, X):
    return (1 + expErr(D, X)) * (1 + Q(1, 2**22)) - 1


def psi16(D, X):
    return (1 + psiP(D, X)) * (1 + Q(1, 2**11)) - 1


def psiL(D, X):
    return (1 + u32) ** 63 * (1 + psiP(D, X)) - 1


def drift(D, X):
    return 1 + 4 * u32 * (ybound(X) + expErr(D, X))


stepAcc = (1 + relDot16(64)) * (1 + Q(1, 2**150)) * ((1 + Q(1, 2**22)) * (1 + u32))
stepL = (1 + 3 * u32) * (1 + Q(1, 2**150)) * (1 + Q(1, 2**22))


def nblocks(N):
    return N // 64


def facc(D, N, X):
    return (1 + psi16(D, X)) * drift(D, X) * stepAcc ** nblocks(N)


def fl(D, N, X):
    return (1 + psiL(D, X)) * drift(D, X) * stepL ** nblocks(N)


absL = Q(1, 2**100)
absA = Q(1, 2**85)
omega0 = Q(1, 2**25) + Q(1, 2**125)


def wlow(D, X):
    return 1 - expErr(D, X)


def betaL(D, N, X):
    return (fl(D, N, X) - 1) + fl(D, N, X) * (nblocks(N) * absL) / wlow(D, X)


def denQ(D, N, X):
    return 1 - betaL(D, N, X)


def relQ(D, N, X):
    return (facc(D, N, X) - 1 + betaL(D, N, X)) / denQ(D, N, X)


def absQS(D, N, X):
    return facc(D, N, X) * omega0 / (wlow(D, X) * denQ(D, N, X))


def absQ0(D, N, X):
    return facc(D, N, X) * (nblocks(N) * absA) / (wlow(D, X) * denQ(D, N, X))


def thetaW(X):
    return Q(1, 2**20) * X


lastRel = Q(1, 2**11) * (1 + Q(1, 2**22)) + Q(1, 2**22)


def tolO(D, N, X):
    return lastRel * (1 + relQ(D, N, X)) * (1 + thetaW(X)) + relQ(D, N, X) * (1 + thetaW(X)) + thetaW(X)


def tolS(D, N, X):
    return absQS(D, N, X) * (1 + lastRel)


def tolF(D, N, X):
    return absQ0(D, N, X) * (1 + lastRel) + Q(1, 2**11) * Q(1, 2**148) + Q(1, 2**25) + Q(1, 2**148)


def tolM(D, N, X):
    return (1 + u32) * (Q(17, 2**22) + 2 * betaL(D, N, X) + Q(1, 2**21) * X + Q(1, 2**149)) + Q(1, 2**150)


rows = [
    (64, 128, 16), (128, 4096, 16), (128, 4096, 32), (128, 2**14, 256), (128, 2**14, 300),
    (64, 2**14, 256), (256, 2**16, 16), (256, 2**16, 1024), (128, 2**16, 1024), (64, 2**16, 1024),
    (128, 2**14, 0), (256, 2**16, 0),
]
print("D N X | tolO tolS tolF tolM | expErr facc-1 betaL relQ | tolS*N")
for D, N, X in rows:
    X = Q(X)
    print(D, N, float(X), "|", "%.3g %.3g %.3g %.3g" % (float(tolO(D, N, X)), float(tolS(D, N, X)),
          float(tolF(D, N, X)), float(tolM(D, N, X))), "|",
          "%.3g %.3g %.3g %.3g" % (float(expErr(D, X)), float(facc(D, N, X) - 1), float(betaL(D, N, X)),
                                   float(relQ(D, N, X))), "|", "%.3g" % float(tolS(D, N, X) * N))

# the worst absolute contribution of tolS * vAbsSum with |v| <= 2^15 over N keys
D, N, X = 256, 2**16, Q(1024)
print("tolS*N*2^15 at the largest case:", float(tolS(D, N, X) * N * 2**15))
