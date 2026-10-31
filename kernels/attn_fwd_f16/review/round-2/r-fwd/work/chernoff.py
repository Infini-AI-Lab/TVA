"""Reviewer r-fwd, round 2: check the prover's revised score estimate (HeadOk's docstring, README):
sigma*s^2*sum_e |X_e Y_e| below 23 (test) and 240 (benchmark) for every pair, except with
probability below 1e-6. Chernoff: P(S > t) <= M(l)^128 exp(-l t), M(l) = E exp(l |X Y|) =
E_X[2 exp(l^2 X^2 / 2) Phi(l |X|)] for standard normals X, Y (trapezoid rule), then a union bound
over the pairs."""

import math

def Phi(a):
    return 0.5 * (1 + math.erf(a / math.sqrt(2)))

def M(l, n=20000, L=12.0):
    h = 2 * L / n
    s = 0.0
    for i in range(n + 1):
        x = -L + i * h
        w = 0.5 if i in (0, n) else 1.0
        s += w * 2 * math.exp(l * l * x * x / 2) * Phi(l * abs(x)) * math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    return s * h

for name, sigma_s2, limit, pairs in (("test", 0.5 * 0.25, 23.0, 3.2e9), ("bench", 1.3, 240.0, 2.0**35)):
    t = limit / sigma_s2
    lp, l = min((128 * math.log(M(l)) - l * t, l) for l in [i / 100 for i in range(5, 90, 2)])
    print(f"{name}: S > {t:.1f} has P <= {math.exp(lp):.2e} (lambda {l}); over {pairs:.2e} pairs <= {pairs * math.exp(lp):.2e}")
