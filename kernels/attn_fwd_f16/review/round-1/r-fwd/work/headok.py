"""Reviewer r-fwd: do the host's inputs meet `HeadOk`'s score premise
`sigma * sum_e |q[n,e] k[j,e]| <= 2^10`?

By Cauchy-Schwarz, sum_e |q_e k_e| <= |q| |k|. For entries N(0, s^2), |row|^2 / s^2 is chi^2 with
D degrees of freedom. Chernoff: P(chi^2_D > D (1 + t)) <= exp(-D/2 (t - ln(1 + t))). A union bound
over every q and k row of the run bounds the chance that some row exceeds the threshold; below
it, every scaled score magnitude is at most sigma * s^2 * D (1 + t)."""

import math

def tail(D, t):
    return math.exp(-D / 2 * (t - math.log1p(t)))

cases = [
    # name, D, entry std, sigma, rows of q and k together
    ("test D=128 (Z=4, H=48, N=4096)", 128, 0.5, 0.5, 2 * 4 * 48 * 4096),
    ("test D=64", 64, 0.5, 0.5, 2 * 4 * 48 * 4096),
    ("bench D=128 (Z=4, H=32, N=2^14)", 128, 1.0, 1.3, 2 * 4 * 32 * 2**14),
    ("bench D=64", 64, 1.0, 1.3, 2 * 4 * 32 * 2**14),
]
for name, D, s, sigma, rows in cases:
    t = 1.25 if D == 128 else 3.0  # threshold D * (1 + t) on chi^2
    p = rows * tail(D, t)
    bound = sigma * s * s * D * (1 + t)
    print(f"{name}: P(some row norm^2 > {D * (1 + t):.0f} s^2) <= {p:.2e}; then sigma*sum|qk| <= {bound:.1f} (premise 1024)")
