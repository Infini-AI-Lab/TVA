"""The tolerances of Model.lean (tolO, tolF), transcribed definition by definition, evaluated
at the README's rows and at other covered sizes, in 80-digit decimal arithmetic (exact rationals
make step^(2^16) too large to evaluate quickly)."""
from decimal import Decimal, getcontext

getcontext().prec = 80


def F(a, b=1):
    return Decimal(a) / Decimal(b)


u32 = F(1, 2**24)


def rhoD(D):
    return (1 + u32) * (1 + 3 * u32) ** (D - 1) - 1


def expErr(D, X):
    return (rhoD(D) + 8 * u32) * X + F(1, 2**50)


def relExp(X):
    return F(1, 2**22) + (2 * X + 1) / F(2**23)


def psiP(D, X):
    e = expErr(D, X)
    return (1 + e + e**2) * (1 + relExp(X)) - 1


def psiL(D, P, X):
    return (1 + u32) ** (P - 1) * (1 + psiP(D, X)) - 1


def psiA(D, P, X):
    return (1 + 3 * u32) ** (P - 1) * (1 + u32) * (1 + psiP(D, X)) - 1


step = (1 + 3 * u32) * (1 + F(1, 2**140)) * (1 + F(1, 2**22))


def drift(D, X):
    return 1 + F(4, 2**22) * (X + expErr(D, X))


def fL(D, P, T, X):
    return (1 + psiL(D, P, X)) * drift(D, X) * step**T


def fA(D, P, T, X):
    return (1 + psiA(D, P, X)) * drift(D, X) * step**T


absL = F(1, 2**100)
absA = F(1, 2**90)


def wlow(D, X):
    return 1 - expErr(D, X)


def betaL(D, P, T, X):
    return (fL(D, P, T, X) - 1) + fL(D, P, T, X) * (T * absL) / wlow(D, X)


def denQ(D, P, T, X):
    return 1 - betaL(D, P, T, X)


def relQ(D, P, T, X):
    return (fA(D, P, T, X) - 1 + betaL(D, P, T, X)) / denQ(D, P, T, X)


def absQ(D, P, T, X):
    return fA(D, P, T, X) * (T * absA) / (wlow(D, X) * denQ(D, P, T, X))


lastRel = F(1, 2**11) * (1 + F(1, 2**22)) + F(1, 2**22)


def tolO(D, P, T, X):
    return lastRel * (1 + relQ(D, P, T, X)) + relQ(D, P, T, X)


def tolF(D, P, T, X):
    return absQ(D, P, T, X) * (1 + lastRel) + F(1, 2**11) * F(1, 2**148) + F(1, 2**148) + F(1, 2**25)


rows = [
    (128, 16, 256, 11),
    (128, 32, 256, 11),
    (64, 16, 129, 10),
    (128, 16, 512, 20),
    (256, 256, 2**16, 2**10),
    # further covered points
    (128, 16, 1, 11),
    (128, 16, 2**16, 11),
    (256, 256, 2**16, 64),
    (256, 16, 2**16, 2**10),
    (64, 16, 2**16, 2**10),
]
print("D P T X | expErr psiP relQ tolO tolF/2^-25")
for D, P, T, X in rows:
    X = F(X)
    print(D, P, T, float(X), "|", f"{float(expErr(D, X)):.4e}", f"{float(psiP(D, X)):.4e}",
          f"{float(relQ(D, P, T, X)):.4e}", f"{float(tolO(D, P, T, X)):.4e}",
          f"{float(tolF(D, P, T, X) / F(1, 2**25)):.6f}")
