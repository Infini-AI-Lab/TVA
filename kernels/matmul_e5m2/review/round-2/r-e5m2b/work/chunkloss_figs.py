"""Independent evaluation of the claim's tolerance `tol` (the minimum of the two arms, as in
`errAt`) and of the round-24 tolerance (the first arm alone), with exact rationals, on the README's
inputs.  Run with the TVA virtual environment's Python; the `torch.randn` part uses the GPU."""
from fractions import Fraction as F
import statistics
import sys

RHO, SIG, ALPHA = F(1, 2**7), F(1, 2**12), F(1, 2**126)
KAP, RHOL = F(2), F(1, 2**12)
MIN_NORMAL = F(1, 2**14)


def lift(x):
    return F(0) if x == 0 else max(abs(x), MIN_NORMAL)


def tols(a, b, K):
    """(new, old, arms): the tolerance by errAt (minimum) and by the round-24 recursion, and how
    many chunks took the second arm."""
    n = 2 * ((K + 63) // 64)
    p = [a[e] * b[e] if e < K else F(0) for e in range(32 * n)]
    q = [lift(a[e]) * lift(b[e]) if e < K else F(0) for e in range(32 * n)]
    E, Eo, S, second = F(0), F(0), F(0), 0
    for j in range(n):
        ch = range(32 * j, 32 * j + 32)
        m = max(q[e] for e in ch)
        A = sum(abs(p[e]) for e in ch)
        S1 = S + sum(p[e] for e in ch)
        X = max(abs(S), m)
        a1, a2 = RHO * (X + E), KAP * A + RHOL * (X + E)
        second += a2 < a1
        E = (1 + SIG) * E + min(a1, a2) + SIG * abs(S1) + ALPHA
        Eo = (1 + RHO + SIG) * Eo + RHO * X + SIG * abs(S1) + ALPHA
        S = S1
    return E, Eo, second, S


def pct(x):
    return f"{100 * float(x):.4g}%"


def spike(K, idx, x, y):
    a = [F(0)] * K
    b = [F(0)] * K
    a[idx], b[idx] = x, y
    return a, b


def main():
    for K in (512, 4096):
        print(f"--- K = {K}")
        a, b = spike(K, K - 1, F(1), F(1))
        E, Eo, s, _ = tols(a, b, K)
        print("one product 1 at K-1:", pct(Eo), "->", pct(E), "second-arm chunks", s)
        a, b = spike(K, 0, F(1), F(1))
        E, Eo, s, _ = tols(a, b, K)
        print("one product 1 at 0:", pct(Eo), "->", pct(E), "second-arm chunks", s)
        a = [F(0)] * K
        a[0], a[64] = F(1), F(-1)
        b = [F(1)] * K
        E, Eo, s, S = tols(a, b, K)
        print("drop s=0 input:", float(Eo), "->", float(E), "S =", S)
        a = [F(1)] * K
        E, Eo, s, S = tols(a, a, K)
        print("all ones rel S:", pct(Eo / S), "->", pct(E / S), "second-arm chunks", s)
        a = [F(1)] * 32 + [F(1, 16)] * (K - 32)
        b = [F(1)] * K
        E, Eo, s, S = tols(a, b, K)
        print("first chunk 1, then products 1/16, rel S:", pct(Eo / S), "->", pct(E / S), "second", s)
        a = [F(1)] * 32 + [F(1, 16)] * (K - 32)
        E, Eo, s, S = tols(a, a, K)
        print("first chunk 1, then products 1/256, rel S:", pct(Eo / S), "->", pct(E / S), "second", s)
    # tol_spike worst case: K mod 64 in 1..32, the product in the step's first chunk
    worst = F(0)
    for K in range(1, 4097):
        a, b = spike(K, K - 1, F(1), F(1))
        E, _, _, _ = tols(a, b, K)
        worst = max(worst, E)
    print("tol_spike, max over covered K of tol / L:", pct(worst), "bound 71713/2^23 =", pct(F(71713, 2**23)))
    # x = y = 1.25 at K - 1: the float16 clause's bound, worst covered K
    xb = F(5, 4)
    w16 = F(0)
    for K in range(1, 4097):
        a, b = spike(K, K - 1, xb, xb)
        E, _, _, S = tols(a, b, K)
        w16 = max(w16, E + (abs(S) + E) * F(1, 2**11) + F(1, 2**25))
    print("x = y = 1.25, float16 bound, worst covered K:", float(w16))
    a, b = spike(512, 511, xb, xb)
    E, _, _, S = tols(a, b, 512)
    print("  at K = 512:", float(E + (abs(S) + E) * F(1, 2**11) + F(1, 2**25)))

    if len(sys.argv) > 1 and sys.argv[1] == "randn":
        import torch
        for K in (512, 4096):
            torch.manual_seed(0)
            A = torch.randn((K, K), device="cuda", dtype=torch.float16).to(torch.float8_e5m2)
            B = torch.randn((K, K), device="cuda", dtype=torch.float16).T.to(torch.float8_e5m2)
            A = A.float().cpu()
            B = B.float().cpu()
            news, olds, Ss, ratio_lt, sec = [], [], [], 0, 0
            draws = 20
            for i in range(draws):
                a = [F(float(v)) for v in A[i, :K].tolist()]
                b = [F(float(v)) for v in B[:K, i].tolist()]
                E, Eo, s, S = tols(a, b, K)
                news.append(float(E)); olds.append(float(Eo)); Ss.append(abs(float(S)))
                ratio_lt += E < abs(S)
                sec += s
            print(f"randn K = {K}: median tol old {statistics.median(olds):.3g} new "
                  f"{statistics.median(news):.3g}, median |S| {statistics.median(Ss):.3g}, "
                  f"tol < |S| on {ratio_lt}/{draws}, second-arm chunks {sec}, "
                  f"max new/old {max(n / o for n, o in zip(news, olds)):.6f}")


main()
