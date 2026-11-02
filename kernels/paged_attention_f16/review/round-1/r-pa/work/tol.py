"""The tolerances of Model.lean (tolO, tolF, and their parts), evaluated with 80-digit decimals.

Each definition is transcribed from lean/PagedAttention/Model.lean, in the same order.
Run: python3 -I tol.py
"""
from decimal import Decimal as Dm, getcontext

getcontext().prec = 80
one = Dm(1)
two = Dm(2)
u32 = one / two ** 24


def rhoD(D):
    return (one + u32) * (one + 3 * u32) ** (D - 1) - one


def expErr(D, X):
    return (rhoD(D) + 8 * u32) * X + one / two ** 50


def relExp(X):
    return one / two ** 22 + (2 * X + 1) / two ** 23


def psiP(D, X):
    e = expErr(D, X)
    return (one + e + e ** 2) * (one + relExp(X)) - one


def psiL(D, P, X):
    return (one + u32) ** (P - 1) * (one + psiP(D, X)) - one


def psiA(D, P, X):
    return (one + 3 * u32) ** (P - 1) * (one + u32) * (one + psiP(D, X)) - one


step = (one + 3 * u32) * (one + one / two ** 140) * (one + one / two ** 22)


def drift(D, X):
    return one + 4 / two ** 22 * (X + expErr(D, X))


def fL(D, P, T, X):
    return (one + psiL(D, P, X)) * drift(D, X) * step ** T


def fA(D, P, T, X):
    return (one + psiA(D, P, X)) * drift(D, X) * step ** T


absL = one / two ** 100
absA = one / two ** 90


def wlow(D, X):
    return one - expErr(D, X)


def betaL(D, P, T, X):
    return (fL(D, P, T, X) - one) + fL(D, P, T, X) * (T * absL) / wlow(D, X)


def denQ(D, P, T, X):
    return one - betaL(D, P, T, X)


def relQ(D, P, T, X):
    return (fA(D, P, T, X) - one + betaL(D, P, T, X)) / denQ(D, P, T, X)


def absQ(D, P, T, X):
    return fA(D, P, T, X) * (T * absA) / (wlow(D, X) * denQ(D, P, T, X))


lastRel = one / two ** 11 * (one + one / two ** 22) + one / two ** 22


def tolO(D, P, T, X):
    return lastRel * (one + relQ(D, P, T, X)) + relQ(D, P, T, X)


def tolF(D, P, T, X):
    return absQ(D, P, T, X) * (one + lastRel) + one / two ** 11 * (one / two ** 148) + one / two ** 148 + one / two ** 25


def show(D, P, T, X):
    X = Dm(X)
    print(f"D={D:4d} P={P:4d} T={T:6d} X={float(X):8.3f}  tolO={float(tolO(D, P, T, X)):.4e}  "
          f"tolF={float(tolF(D, P, T, X)):.4e}  relQ={float(relQ(D, P, T, X)):.4e}  "
          f"expErr={float(expErr(D, X)):.3e}  step^T-1={float(step ** T - 1):.3e}")


if __name__ == "__main__":
    print("README table rows:")
    show(128, 16, 256, 11)
    show(128, 32, 256, 11)
    show(64, 16, 129, 10)
    show(128, 16, 512, 20)
    show(256, 256, 2 ** 16, 2 ** 10)
    print("The test's rows at the measured X (see test_data.out):")
    for (D, P, T, X) in [(128, 16, 256, "10.96"), (64, 16, 129, "9.04"), (128, 32, 256, "10.49")]:
        show(D, P, T, X)
    print("Realistic larger scores (D=128, P=16, T=4096 pages, i.e. 65536 tokens):")
    for X in [20, 50, 100, 200, 500, 1024]:
        show(128, 16, 4096, X)
    print("The fp16 output rounding alone, 2^-11 =", float(one / two ** 11))
