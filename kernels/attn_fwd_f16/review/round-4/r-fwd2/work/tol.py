"""Reviewer r-fwd2, round 4: evaluate the claim's tolerances (AttnFwd.Model, "The tolerances")
exactly, with the definitions transcribed from the Lean at gate 7c52bcaa1a441cb2 (absA = 2^-37, the
new kapA = 2^-29, absQ0 with the term facc*kapA/denQ), at the host's sizes and at the largest covered
ones; compare with the round-3 constants (absA = 2^-22, no kapA); check absA's and kapA's docstring
figures, and the threshold below which tolF outweighs tolO*attnAbs.
Run: python3 -I tol.py"""
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
omega0 = Q(1, 2**25) + Q(1, 2**125) + Q(66, 2**36)
# (absA, kapA)
CONST = {"round3": (Q(1, 2**22), Q(0)), "round4": (Q(1, 2**37), Q(1, 2**29))}


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


def absQ0(D, N, X, c):
    absA, kapA = CONST[c]
    return facc(D, N, X) * (nblocks(N) * absA) / (wlow(D, X) * denQ(D, N, X)) + facc(D, N, X) * kapA / denQ(D, N, X)


def thetaW(X):
    return Q(1, 2**20) * X


lastRel = Q(1, 2**11) * (1 + Q(1, 2**22)) + Q(1, 2**22)


def tolO(D, N, X):
    return lastRel * (1 + relQ(D, N, X)) * (1 + thetaW(X)) + relQ(D, N, X) * (1 + thetaW(X)) + thetaW(X)


def tolS(D, N, X):
    return absQS(D, N, X) * (1 + lastRel)


def tolF(D, N, X, c):
    return absQ0(D, N, X, c) * (1 + lastRel) + Q(1, 2**11) * Q(1, 2**148) + Q(1, 2**25) + Q(1, 2**148)


def tolM(D, N, X):
    return (1 + u32) * (Q(17, 2**22) + 2 * betaL(D, N, X) + Q(1, 2**21) * X + Q(1, 2**149)) + Q(1, 2**150)


rows = [
    (64, 128, 16), (128, 4096, 16), (128, 4096, 23), (128, 2**14, 240), (128, 2**14, 256),
    (256, 2**16, 16), (256, 2**16, 1024), (64, 2**16, 1024), (256, 2**16, 0),
]
print("D N X | tolO tolS tolF tolM (round 4) | tolF (round 3) | tolF - 2^-25 | attnAbs* = tolF/(1-tolO)")
worst = Q(0)
for D, N, X in rows:
    X = Q(X)
    f4, f3 = tolF(D, N, X, "round4"), tolF(D, N, X, "round3")
    thr = f4 / (1 - tolO(D, N, X))
    worst = max(worst, thr)
    print(D, N, int(X), "|", "%.3g %.3g %.4g %.3g" % (float(tolO(D, N, X)), float(tolS(D, N, X)), float(f4),
          float(tolM(D, N, X))), "|", "%.3g" % float(f3), "|", "%.3g" % float(f4 - Q(1, 2**25)), "|",
          "%.4g" % float(thr))
print("largest attnAbs* over the rows: %.4g; float16 smallest subnormal 2^-24 = %.4g; below: %s"
      % (float(worst), 2.0**-24, worst < Q(1, 2**24)))
# at that threshold the bound tolO*attnAbs + tolF equals attnAbs, so it is below 2^-24 as well
D, N, X = 256, 2**16, Q(1024)
a = tolF(D, N, X, "round4") / (1 - tolO(D, N, X))
print("at D=256, N=2^16, X=2^10: tolO*a + tolF = %.4g (< 2^-24: %s)"
      % (float(tolO(D, N, X) * a + tolF(D, N, X, "round4")), tolO(D, N, X) * a + tolF(D, N, X, "round4") < Q(1, 2**24)))

print()
# docstring figures
perkey_rest = Q(1, 2**14) * (Q(1, 2**25) + Q(1, 2**125)) + Q(1, 2**28)
block_rest = 64 * perkey_rest
print("absA: per block rest 64*(2^-14*(2^-25+2^-125) + 2^-28) = 2^-22 + %.4g*2^-33 (<= 2^-22 + 2^-32: %s)"
      % (float((block_rest - Q(1, 2**22)) * 2**33), block_rest <= Q(1, 2**22) + Q(1, 2**32)))
print("absA: relDot16 64 * (2^-22 + 2^-32) = %.4f * 2^-37; relDot16 64 * block rest = %.4f * 2^-37; absA = 2^-37"
      % (float(relDot16(64) * (Q(1, 2**22) + Q(1, 2**32)) * 2**37), float(relDot16(64) * block_rest * 2**37)))
print("kapA: 66*2^-36*8/7 = %.4f * 2^-29 (<= 2^-29: %s)"
      % (float(Q(66, 2**36) * Q(8, 7) * 2**29), Q(66, 2**36) * Q(8, 7) <= Q(1, 2**29)))
print("psi16 at D=256, X=2^10 = %.4f (<= 1/7: %s)" % (float(psi16(256, Q(1024))), psi16(256, Q(1024)) <= Q(1, 7)))
