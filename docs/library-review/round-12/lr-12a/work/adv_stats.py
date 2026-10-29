"""lr-12a: what adv.py's inputs exercise (its first N inputs of each arithmetic test, same seeds).

For each test, with exact rational arithmetic: how many exact results are ties of the rounding
(exactly halfway between two neighbours), how many round to a nonzero subnormal, how many are
nonzero but round to zero, how many overflow, and how many are exact zeros (whose sign the
library's rule fixes). No device here.

Usage: python -I adv_stats.py [N]
"""
import os
import sys
from fractions import Fraction as Q

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adv  # noqa: E402


def classify(t, q):
    e, mb = adv.FMT[t]
    bias = (1 << (e - 1)) - 1
    if q == 0:
        return "zero"
    a = abs(q)
    maxv = Q((1 << (mb + 1)) - 1) * Q(2) ** (bias - mb)
    k = a.numerator.bit_length() - a.denominator.bit_length()
    if Q(2) ** k > a:
        k -= 1
    kk = max(k, 1 - bias)
    N = a / Q(2) ** (kk - mb)
    tie = (N - (N.numerator // N.denominator)) == Q(1, 2)
    r = adv.round_pat(t, q) if a <= maxv * 2 else None
    out = []
    if tie:
        out.append("tie")
    if a > maxv + Q(2) ** (bias - mb - 1) or (a == maxv + Q(2) ** (bias - mb - 1)):
        out.append("overflow")
    elif k < 1 - bias:
        rv = r & ((1 << (e + mb)) - 1)
        out.append("subnormal" if rv else "to-zero")
    return "+".join(out) or "plain"


def main():
    import random
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    for name in ("add.f32", "sub.f32", "mul.f32", "fma.f32", "div.f32", "add.f64", "mul.f64", "fma.f64", "div.f64"):
        op, src, dst, spec, kind = adv.TESTS[name]
        rng = random.Random(name)
        x, y, z = adv.gen_arith(src, op, n, rng)
        e, mb = adv.FMT[src]
        cnt = {}
        for i in range(n):
            ps = [int(x[i]), int(y[i]), int(z[i])]
            try:
                xv, yv, zv = (adv.val(src, p) for p in ps)
            except AssertionError:
                cnt["special operand"] = cnt.get("special operand", 0) + 1
                continue
            if op == "add":
                q = xv + yv
            elif op == "sub":
                q = xv - yv
            elif op == "mul":
                q = xv * yv
            elif op == "fma":
                q = xv * yv + zv
            else:
                if yv == 0:
                    cnt["zero divisor"] = cnt.get("zero divisor", 0) + 1
                    continue
                q = xv / yv
            c = classify(src, q)
            cnt[c] = cnt.get(c, 0) + 1
        print(f"{name:8s} " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())), flush=True)


if __name__ == "__main__":
    main()
