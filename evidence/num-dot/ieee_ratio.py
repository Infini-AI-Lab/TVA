"""TVA library evidence (library issue L66): which input gives `DotF32Ieee` its largest ratio.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot/ieee_ratio.py
Output: evidence/num-dot/ieee_ratio.out (README.md, "The ratio of `DotF32Ieee`").

Part 2 of run.py reports, for the float32 "ieee" dot, a largest ratio of error to bound near
(K-1)/(2(K+1)) and, in its last column, the input kind with the largest error relative to
|acc| + sum |a b| (`tiny`), which is not the kind with the largest ratio. This script uses run.py's
kernel and input generator (the same eight kinds, two draws each, on 16 x 16 with one warp; fresh
draws from run.py's seed) and reports, per K and kind, the largest ratio as run.py computes it.
For the element with the largest ratio over all kinds it recomputes the error exactly with
fractions, in units of 2^-24 mag (mag = |acc| + sum |a b|) and against the bound
(K+1) 2^-23 mag + (K+1) 2^-149, and prints the element's accumulator, products and result.
"""
import os
import sys
from fractions import Fraction

sys.dont_write_bytecode = True  # no __pycache__ in the evidence folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as nd  # noqa: E402  (run.py of this folder: its kernel, inputs and seed)

import torch  # noqa: E402
import triton  # noqa: E402


def main():
    out = []

    def emit(s=""):
        print(s, flush=True)
        out.append(s)

    emit(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    emit("## The largest ratio per input kind (float32 x float32, ieee, 16 x 16, 1 warp)")
    emit("`ratio` as run.py's Part 2 computes it: (error + (K+1) 2^-52 mag) / ((K+1) 2^-23 mag + (K+1) 2^-149).\n")
    emit("| K | " + " | ".join(nd.KINDS) + " | (K-1)/(2(K+1)) |")
    emit("|---" * (len(nd.KINDS) + 2) + "|")
    worst = {}
    for K in [8, 16, 32, 64, 128, 256, 512, 1024]:
        name, rel, ab = nd.fact("float32", "float32", K)
        row = []
        best = (0.0, None)
        for kind in nd.KINDS:
            r_kind = 0.0
            for rep in range(2):
                a, b, c = nd.gen(kind, 16, 16, K, "float32", "float32")
                o, path = nd.run(a, b, c, 1, "ieee")
                assert path == "fma"
                ad, bd, cd = a.double(), b.double(), c.double()
                exact = ad @ bd + cd
                mag = ad.abs() @ bd.abs() + cd.abs()
                err = (o.double() - exact).abs() + (K + 1) * 2.0 ** -52 * mag
                ratio = err / (rel * mag + ab)
                r = ratio.max().item()
                r_kind = max(r_kind, r)
                if r > best[0]:
                    i, j = divmod(int(ratio.argmax().item()), 16)
                    best = (r, (kind, a[i].tolist(), b[:, j].tolist(), c[i, j].item(), o[i, j].item()))
            row.append(f"{r_kind:.4f}")
        worst[K] = best
        emit(f"| {K} | " + " | ".join(row) + f" | {(K - 1) / (2 * (K + 1)):.4f} |")

    emit("\n## The element with the largest ratio, exactly")
    emit("| K | kind | acc | products a·b (distinct values) | result | exact acc + sum a·b | error / (2^-24 mag) | error / bound |")
    emit("|---|---|---|---|---|---|---|---|")
    for K, (r, (kind, arow, bcol, acc, res)) in worst.items():
        prods = [Fraction(x) * Fraction(y) for x, y in zip(arow, bcol)]
        exact = Fraction(acc) + sum(prods)
        mag = abs(Fraction(acc)) + sum(abs(p) for p in prods)
        err = abs(Fraction(res) - exact)
        bound = (K + 1) * Fraction(1, 2 ** 23) * mag + (K + 1) * Fraction(1, 2 ** 149)
        dist = sorted({float(p) for p in prods})
        emit(f"| {K} | {kind} | {acc!r} | {', '.join(repr(p) for p in dist[:3])}{' …' if len(dist) > 3 else ''} "
             f"| {res!r} | {float(exact)!r} | {float(err / (mag * Fraction(1, 2 ** 24))):.6f} | {float(err / bound):.6f} |")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ieee_ratio.out"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
