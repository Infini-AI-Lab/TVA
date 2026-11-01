"""Reviewer r-e5fwd2, re-review: do the host's inputs meet HeadOk's new cap
`sm_scale * scoreLift <= Xh D` (Model.lean: `scoreLift = sum_e liftMag(q_e) liftMag(k_e)`,
`liftMag x = 0` for `x = 0`, else `max(|x|, 2^-14)`; `Xh D = 2^10 / ceil(D/32)`: 512 at D = 64,
256 at D = 128)?

A Chernoff bound on one pair (n, j): q_e, k_e iid normal(0, std) rounded to float16 and then to
float8e5 (round to nearest even), L = liftMag of that value, S = sum_{e < D} L(q_e) L(k_e) and
P(S >= s) <= exp(-(lam*s - D*log M(lam))), M the exact moment generating function of L(q) L(k).
The distribution of L is computed from the normal CDF over a grid that is uniform on [-14 std,
14 std] and logarithmic near 0 (so the subnormal range of float8e5 and float16 is resolved). The
same bound for the old cap (|q_e k_e|) is printed beside it.
"""
import math
import numpy as np


def e5m2(x):
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    ulp = np.ldexp(1.0, np.maximum(e - 3, -16))
    return np.clip(np.round(x / ulp) * ulp, -57344.0, 57344.0)


def dist(std):
    u = np.linspace(-14 * std, 14 * std, 400001)
    lg = np.logspace(-30, -2, 200001)
    x = np.unique(np.concatenate([u, lg, -lg, [0.0]]))
    mid = np.concatenate([[-np.inf], (x[1:] + x[:-1]) / 2, [np.inf]])
    cdf = 0.5 * (1 + np.array([math.erf(t / (std * math.sqrt(2))) if np.isfinite(t) else (1.0 if t > 0 else -1.0)
                               for t in mid]))
    p = np.diff(cdf)
    r = e5m2(x.astype(np.float16).astype(np.float64))
    a = np.abs(r)
    L = np.where(r == 0, 0.0, np.maximum(a, 2.0 ** -14))
    out = {}
    for name, vals in (("abs", a), ("lift", L)):
        uv, idx = np.unique(vals, return_inverse=True)
        out[name] = (uv, np.bincount(idx, weights=p))
    out["psub"] = p[(r != 0) & (a < 2.0 ** -14)].sum()
    return out


def logM(vals, wts, lam):
    inner = np.array([np.sum(wts * np.exp(lam * v * vals)) for v in vals])
    return math.log(np.sum(wts * inner))


for label, D, sig, std in [("bench", 128, 1.3, 1.0), ("bench", 64, 1.3, 1.0),
                           ("test", 128, 0.5, 0.5), ("test", 64, 0.5, 0.5)]:
    dd = dist(std)
    cap = {64: 512.0, 128: 256.0}[D]
    s = cap / sig
    res = {}
    for name in ("abs", "lift"):
        vals, wts = dd[name]
        best = max((lam * s - D * logM(vals, wts, lam), lam) for lam in np.linspace(0.02, 0.98, 49) / std ** 2)
        res[name] = -best[0] / math.log(10)
    print(f"{label} D={D} sm_scale={sig} std={std}: P(subnormal fp8 operand) = {dd['psub']:.3g}; cap on the sum "
          f"= {s:.1f}; log10 P(one pair exceeds) <= {res['lift']:.2f} (new cap, lifted), {res['abs']:.2f} (old cap)")
pairs = 4 * 32 * 16384 ** 2
print(f"benchmark's largest launch (Z=4, H=32, N=2^14): {pairs:.3g} pairs, log10 = {np.log10(pairs):.1f}")
