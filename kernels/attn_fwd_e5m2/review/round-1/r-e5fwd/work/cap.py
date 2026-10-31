"""Reviewer r-e5fwd: the largest scaled score magnitude sigma * sum_e |q_e k_e| over every pair
(n, j) of a head, on the benchmark's inputs (randn float16 cast to float8e5, sm_scale = 1.3) at
N_CTX = 2^14 and on the test's (normal(0, 0.5), sm_scale = 0.5) at N_CTX = 4096, against the
cap Xh HEAD_DIM = 2^10 / ceil(HEAD_DIM / 32) of `HeadOk` (512 at 64, 256 at 128).
Run: python3 -I cap.py
"""
import numpy as np


def e5m2(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.zeros_like(x)
    nz = a > 0
    e = np.maximum(np.floor(np.log2(a[nz])), -14.0)
    q = 2.0 ** (e - 2)
    out[nz] = np.round(a[nz] / q) * q
    return np.sign(x) * out


def head_max(rng, N, D, kind):
    if kind == 'bench':
        g = lambda: np.abs(e5m2(np.asarray(rng.standard_normal((N, D)), np.float16).astype(np.float64)))
        s = float(np.float32(1.3))
    else:
        g = lambda: np.abs(e5m2(np.asarray(rng.normal(0.0, 0.5, (N, D)), np.float16).astype(np.float64)))
        s = 0.5
    q = g().astype(np.float32)
    k = g().astype(np.float32)
    m = 0.0
    for n0 in range(0, N, 2048):
        m = max(m, float((q[n0:n0 + 2048] @ k.T).max()))
    return s * m


def main():
    rng = np.random.default_rng(7)
    for kind, N, heads in (('test', 4096, 8), ('bench', 2 ** 14, 6)):
        for D, cap in ((64, 512), (128, 256)):
            ms = [head_max(rng, N, D, kind) for _ in range(heads)]
            print(f"{kind}: D={D} N={N}: per-head max of sigma*sum|q k| over {heads} heads: "
                  + ", ".join(f"{x:.1f}" for x in ms) + f"  (cap Xh = {cap})")


main()
