"""Reviewer r-fwd2: how large sigma * sum_e |q_e k_e| gets on the host's inputs (HeadOk needs
<= 2^10 on every pair of a head, Post's tolerance is evaluated at the row's bound X).

q, k have iid entries s * N(0, 1) (the test: s = 0.5, sigma = 0.5; the benchmark: s = 1,
sigma = 1.3), so sigma * sum_e |q_e k_e| = sigma * s^2 * S with S = sum of D iid |X Y|, X, Y
standard normal. A Chernoff bound P(S >= t) <= exp(-(theta t - D log M(theta))) with
M(theta) = E exp(theta |X Y|), times the number of pairs, bounds the chance that any pair of any
launch exceeds t. M is computed by quadrature: M(theta) = E_X[2 exp(theta^2 X^2 / 2) Phi(theta |X|)].
Run: python3 -I scores.py"""
import math


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def M(theta, steps=4000, L=40.0):
    # the integrand decays like exp(-(1 - theta^2) x^2 / 2); L = 40 covers theta <= 0.95
    h = 2 * L / steps
    tot = 0.0
    for i in range(steps + 1):
        x = -L + i * h
        w = 0.5 if i in (0, steps) else 1.0
        tot += w * math.exp(-x * x / 2 + theta * theta * x * x / 2) * 2 * Phi(theta * abs(x))
    return tot * h / math.sqrt(2 * math.pi)


THETAS = [k / 100 for k in range(1, 96)]
LOGM = [math.log(M(th)) for th in THETAS]


def log_tail(D, t):
    best = 0.0
    for th, lm in zip(THETAS, LOGM):
        best = max(best, th * t - D * lm)
    return -best  # log P(S >= t) <= this


def smallest_t(D, pairs, target=1e-6):
    # the smallest t (step 1) with pairs * P(S >= t) <= target
    t = D * 2 / math.pi
    while math.log(pairs) + log_tail(D, t) > math.log(target):
        t += 1.0
    return t


cases = [
    # name, D, sigma, s, pairs (all launches of the largest shape, summed over heads)
    ("test D=128 (Z=4, H=48, N=4096)", 128, 0.5, 0.5, 4 * 48 * 4096 ** 2),
    ("test D=64  (Z=4, H=48, N=4096)", 64, 0.5, 0.5, 4 * 48 * 4096 ** 2),
    ("bench D=128 (Z=4, H=32, N=2^14)", 128, 1.3, 1.0, 4 * 32 * (2 ** 14) ** 2),
    ("bench D=64  (Z=4, H=32, N=2^14)", 64, 1.3, 1.0, 4 * 32 * (2 ** 14) ** 2),
]
for name, D, sigma, s, pairs in cases:
    t = smallest_t(D, pairs)
    print("%s: mean %.1f, bound with prob >= 1 - 1e-6 over %.2g pairs: %.1f (2^10 = 1024)"
          % (name, sigma * s * s * D * 2 / math.pi, pairs, sigma * s * s * t))
