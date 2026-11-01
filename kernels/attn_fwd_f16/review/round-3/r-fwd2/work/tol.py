"""Reviewer r-fwd2, round 3: evaluate the claim's tolerances (AttnFwd.Model, "The tolerances")
exactly, with the definitions transcribed from the Lean at gate df22d153664804d2 (library round 25:
omega0 gains 66*2^-36, absA is 2^-22), at the host's sizes and at the largest covered ones; compare
with the round-1 constants; and bound the lift scoreMag - sum|q.k| on the host's inputs.
Run: python3 -I tol.py"""
from fractions import Fraction as Q
import math

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
CONST = {
    "round1": (Q(1, 2**85), Q(1, 2**25) + Q(1, 2**125)),
    "round25": (Q(1, 2**22), Q(1, 2**25) + Q(1, 2**125) + Q(66, 2**36)),
}


def wlow(D, X):
    return 1 - expErr(D, X)


def betaL(D, N, X):
    return (fl(D, N, X) - 1) + fl(D, N, X) * (nblocks(N) * absL) / wlow(D, X)


def denQ(D, N, X):
    return 1 - betaL(D, N, X)


def relQ(D, N, X):
    return (facc(D, N, X) - 1 + betaL(D, N, X)) / denQ(D, N, X)


def thetaW(X):
    return Q(1, 2**20) * X


lastRel = Q(1, 2**11) * (1 + Q(1, 2**22)) + Q(1, 2**22)


def tolO(D, N, X):
    return lastRel * (1 + relQ(D, N, X)) * (1 + thetaW(X)) + relQ(D, N, X) * (1 + thetaW(X)) + thetaW(X)


def tolS(D, N, X, c):
    absA, omega0 = CONST[c]
    return facc(D, N, X) * omega0 / (wlow(D, X) * denQ(D, N, X)) * (1 + lastRel)


def tolF(D, N, X, c):
    absA, omega0 = CONST[c]
    absQ0 = facc(D, N, X) * (nblocks(N) * absA) / (wlow(D, X) * denQ(D, N, X))
    return absQ0 * (1 + lastRel) + Q(1, 2**11) * Q(1, 2**148) + Q(1, 2**25) + Q(1, 2**148)


def tolM(D, N, X):
    return (1 + u32) * (Q(17, 2**22) + 2 * betaL(D, N, X) + Q(1, 2**21) * X + Q(1, 2**149)) + Q(1, 2**150)


rows = [
    (64, 128, 16), (128, 4096, 16), (128, 4096, 23), (128, 2**14, 240), (128, 2**14, 256),
    (256, 2**16, 16), (256, 2**16, 1024), (64, 2**16, 1024),
]
print("D N X | tolO tolS tolF tolM (round 25) | tolS tolF (round-1 constants) | tolS growth")
for D, N, X in rows:
    X = Q(X)
    s25, f25 = tolS(D, N, X, "round25"), tolF(D, N, X, "round25")
    s1, f1 = tolS(D, N, X, "round1"), tolF(D, N, X, "round1")
    print(D, N, int(X), "|", "%.3g %.3g %.3g %.3g" % (float(tolO(D, N, X)), float(s25), float(f25),
          float(tolM(D, N, X))), "|", "%.3g %.3g" % (float(s1), float(f1)), "|", "%.4f" % float(s25 / s1))

print()
print("tolF against N/64*2^-22, and tolS*N (round 25):")
for D, N, X in rows:
    X = Q(X)
    print(D, N, int(X), "tolF/(N/64*2^-22) = %.4f" % float(tolF(D, N, X, "round25") / (nblocks(N) * Q(1, 2**22))),
          " tolS*N = %.3g" % float(tolS(D, N, X, "round25") * N))

print()
print("Where tolF matters: the weighted mean |v| (attnAbs) at which tolF equals tolO*attnAbs:")
for D, N, X in rows:
    X = Q(X)
    print(D, N, int(X), "attnAbs* = %.3g" % float(tolF(D, N, X, "round25") / tolO(D, N, X)))
print("float16 half-ulp at 1: %.3g; smallest normal float16: %.3g" % (2.0**-11, 2.0**-14))

print()
print("The p.v lift if charged relative to p (2^-14*p_k summed to 2^-14*l_i): 66*2^-36 = %.3g" % (66 * 2.0**-36))

# Host coverage of HeadOk / Post's X premise with scoreMag:
# scoreMag <= sum|q.k| + 2^-14*(sum|q| + sum|k|) + D*2^-28 (DotLift.sum_liftMag_mul_le),
# sum|q| <= sqrt(D)*|q|, |q|^2/s^2 ~ chi2_D; Chernoff: P(chi2_k > x) <= (x/k)^(k/2) exp(-(x-k)/2).
print()
rowsmax = 2 * 2**21  # q and k rows, Z*H*N_CTX <= 2^21
for name, D, s, sig in [("test", 128, 0.5, 0.5), ("benchmark", 128, 1.0, 1.3), ("test D=64", 64, 0.5, 0.5)]:
    x = 288.0
    logp = (D / 2) * math.log(x / D) - (x - D) / 2
    p = rowsmax * math.exp(logp)
    lift = sig * (2.0**-14 * 2 * math.sqrt(D) * math.sqrt(x) * s + D * 2.0**-28)
    print("%s: P(some row |row|^2 > 288 s^2) <= %.3g; lift on sigma*scoreMag <= %.4f" % (name, p, lift))
