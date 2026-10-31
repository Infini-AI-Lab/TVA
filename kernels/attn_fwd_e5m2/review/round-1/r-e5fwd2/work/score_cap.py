"""Reviewer r-e5fwd2: do the host's inputs meet HeadOk's cap `sm_scale * sum_e |q_e k_e| <= Xh D`
(Model.lean: `Xh D = 2^10 / ceil(D/32)`: 512 at D = 64, 256 at D = 128)?

A Chernoff bound on one pair (n, j): with z, z' iid standard normal rounded to float16 and then to
float8e5 (round to nearest even), S = sum_{e < D} |r(z_e) r(z'_e)| (times std^2) and
P(S >= s) <= exp(-(lam*s - D*log M(lam))) for lam in (0, 1), M the exact moment generating function
of |r(z) r(z')|, computed by quadrature over the normal density.
"""
import numpy as np


def e5m2(x):
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    ulp = np.ldexp(1.0, np.maximum(e - 3, -16))
    return np.clip(np.round(x / ulp) * ulp, -57344.0, 57344.0)


x = np.linspace(-14, 14, 400001)
dx = x[1] - x[0]
phi = np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * dx
r = np.abs(e5m2(x.astype(np.float16).astype(np.float64)))
vals, idx = np.unique(r, return_inverse=True)
wts = np.bincount(idx, weights=phi)


def logM(lam):
    inner = np.array([np.sum(wts * np.exp(lam * v * vals)) for v in vals])
    return np.log(np.sum(wts * inner))


print("E|r(z) r(z')| =", np.sum(wts * vals) ** 2)
for label, D, sig, std in [("bench", 128, 1.3, 1.0), ("bench", 64, 1.3, 1.0),
                           ("test", 128, 0.5, 0.5), ("test", 64, 0.5, 0.5)]:
    cap = {64: 512.0, 128: 256.0}[D]
    s = cap / (sig * std * std)
    best = max((lam * s - D * logM(lam), lam) for lam in np.linspace(0.02, 0.98, 97))
    print(f"{label} D={D} sm_scale={sig} std={std}: cap on sum|z z'| = {s:.1f}; "
          f"log10 P(one pair exceeds) <= {-best[0] / np.log(10):.1f}")
pairs = 4 * 32 * 16384 ** 2
print(f"benchmark's largest launch (Z=4, H=32, N=2^14): {pairs:.3g} pairs, log10 = {np.log10(pairs):.1f}")
