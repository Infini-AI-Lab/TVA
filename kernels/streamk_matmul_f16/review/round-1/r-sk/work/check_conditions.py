"""Check each launch the test makes against Inst.ok, and the README's tolerance table.

Exact rational arithmetic (fractions) for the tolerance; integers for the conditions.
"""
from fractions import Fraction as Fr

def cdiv(a, b):
    return (a + b - 1) // b

def steps(K):
    return (K + 63) // 64

def inst_ok(M, N, K, np_, sam, sak, sbk, sbn, scm, scn):
    T = steps(K)
    tiles = cdiv(M, 128) * cdiv(N, 128)
    total = tiles * T
    ipp = cdiv(total, np_)
    L = 2 ** 31
    conds = {
        "pos": M > 0 and N > 0 and K > 0,
        "M_le": M + 128 <= L, "N_le": N + 128 <= L, "K_le": K <= 2 ** 13,
        "strides": all(s < L for s in (sam, sak, sbk, sbn, scm)) and 0 < scn < L,
        "am_fit": (M - 1) * sam < L, "ak_fit": (K - 1) * sak < L,
        "bk_fit": (K - 1) * sbk < L, "bn_fit": (N - 1) * sbn < L,
        "cm_fit": (M - 1) * scm < L, "c_rows": N * scn <= scm,
        "cover": total <= np_ * ipp, "progs_lt": np_ * ipp < L,
    }
    return conds, T, tiles, total, ipp

def bound(rho, alpha, t, B):
    return ((1 + rho) ** t - 1) * B + t * (1 + rho) ** t * alpha

rho = Fr(66, 2 ** 22)
alpha = Fr(66, 2 ** 126)
ra, aa = Fr(1, 2 ** 23), Fr(1, 2 ** 124)

def tol(K, A):
    T = steps(K)
    td = bound(rho, alpha, T, A)
    return td + bound(ra, aa, T, A + td)

shapes = [(512, 512, 512, 132), (1000, 1000, 1000, 132), (3072, 4096, 1024, 132),
          (257, 129, 4097, 7), (64, 64, 64, 3), (8192, 8192, 4096, 132)]
for (M, N, K, np_) in shapes:
    conds, T, tiles, total, ipp = inst_ok(M, N, K, np_, K, 1, N, 1, N, 1)
    bad = [k for k, v in conds.items() if not v]
    # which tiles are whole (owned by one program) and which split
    whole = sum(1 for t in range(tiles)
                if (t * T) // ipp == (t * T + T - 1) // ipp and (t * T) // ipp < np_)
    print(f"M={M} N={N} K={K} np={np_}: T={T} tiles={tiles} total={total} ipp={ipp} "
          f"np*ipp={np_*ipp} whole={whole} split={tiles-whole} failing={bad}")

print()
print("tol K A / A (absolute part at A = 1 given separately):")
for K in [64, 512, 1000, 1024, 4096, 4097, 8192, 16384, 65536]:
    rel = tol(K, Fr(1)) - tol(K, Fr(0))
    absp = tol(K, Fr(0))
    print(f"K={K:6d} T={steps(K):5d} tol/A={float(rel):.4e} abs={float(absp):.2e}")

# bfloat16's largest relative rounding error
print("bf16 max rel err:", float(Fr(1, 2 ** 8) / (1 + Fr(1, 2 ** 8))))
