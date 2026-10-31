"""Reviewer r-e5fwd2, round 2: Post's two output tolerances, `tolO` and the new `tolV`
(Model.lean, "The constants"), evaluated on data like the host's and compared with the exact
attention `attnRef` of the same rows.

`tolO` is transcribed as in round 1 (`relDot8`, `expErr`, `pE`, `dmax`, `rhoA`, `psiP`, `psi8`,
`psiA`, `omega8`, `absA`, `errS`, `errB`, `gam63`, `stepL`, `qL`, `Fhi`, `Flo`, `aL`, `Fp`, `betaL`,
`tolO`, `lgB`, `tolM`), with `X` the row's own `max_j sm_scale * scoreAbs(n, j)` over its keys.

`tolV` is transcribed from `magCp`, `magNext`, `magA`, `magD`, `tolV`, with `rho8 = relDot8 64`,
`R = |attnRef|`, the recursion run over the row's `nblk` blocks (`nblk`: `2 * (n / 128) + 2` when
causal, `N / 64` otherwise), and `K` the smallest bound Post allows:
`K = max_j |vEff_j - attnRef|` over the row's keys, per column `e`.

Data: `q`, `k`, `v` drawn as the test does (`normal(0, 0.5)` in float16, `sm_scale = 0.5`) or as the
benchmark does (`randn` in float16, `sm_scale = 1.3`), then rounded to float8e5 (round to nearest
even, 2 fraction bits). One head with `vIn` (so `vEff = v`), a sample of rows; every column `e`.

Printed per configuration: the median `X` and `pE`, the medians of `tolO`, `tolV` and `|attnRef|`,
and the share of the sampled elements on which the output `0` meets each bound and Post (both):
`|attnRef| <= tolO`, `|attnRef| <= tolV`, both. `p8` is the share on which `p`'s fp8 rounding alone,
`sum_j w_j |v_j| / (8 sum_j w_j)`, is at least `|attnRef|` (the README's per-lane remark).

The last part checks the README's linear form: `tolV = a_T * K + b_T * R` (up to `2^-17`-sized
terms), the largest `K / R` with `tolV < R` at `T` blocks, and the convex-hull property: any
float8e5 rounding `c` of an average of the row's values with any nonnegative weights has
`|c - attnRef| <= K + (K + R) / 8`, below `tolV` (`a_T > 9/8`, `b_T > 1/8`).
"""
import math
import numpy as np


def e5m2(x):
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)                      # x = m * 2^e, 1/2 <= |m| < 1
    ulp = np.ldexp(1.0, np.maximum(e - 3, -16))
    return np.clip(np.round(x / ulp) * ulp, -57344.0, 57344.0)


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


u32 = 2.0 ** -24


def relDot8(K):
    return (1 + 2.0 ** -7) ** ((K + 31) // 32) - 1


rho8 = relDot8(64)


def consts(D, N, X):
    yb = 1.5 * X
    expErr = (relDot8(D) + 2.0 ** -19) * yb + 2.0 ** -50
    pE = 2.0 ** expErr
    dmax = 2 * (yb + expErr) / 2 ** 24 + 2.0 ** -150
    rhoA = (1 + rho8) * ((1 + dmax) * ((1 + 2.0 ** -22) * (1 + u32))) - 1
    psiP = pE * (1 + 2.0 ** -22) - 1
    psi8 = (1 + psiP) * 9 / 8 - 1
    psiA = (1 + rho8) * (1 + psi8) - 1
    omega8 = 2.0 ** -17 + 9 / 8 * 2.0 ** -126
    absA = 2.0 ** -64
    nb = N // 64
    gam63 = (1 + u32) ** 63 - 1
    stepL = (1 + 3 * u32) * ((1 + dmax) * (1 + 2.0 ** -22))
    qL = (1 - 3 * u32) * (2 - (1 + dmax) * (1 + 2.0 ** -22))
    Fhi = (1 + gam63) * (1 + psiP) * stepL ** nb
    Flo = (1 - 3 * u32) * (1 - gam63) * (1 - 2.0 ** -22) / pE * qL ** nb
    aL = nb * 2.0 ** -64
    Fp = Flo - aL * pE
    betaL = max(Fhi - 1, 1 - Flo) + aL * pE
    lgB = max(math.log2(Fhi + aL * pE), -math.log2(Fp))
    tolM = (1 + u32) * (lgB + (lgB + expErr + 16) / 2 ** 22 + 2.0 ** -149) + 2.0 ** -150
    return dict(pE=pE, rhoA=rhoA, psiA=psiA, omega8=omega8, absA=absA, Fp=Fp,
                betaL=betaL, tolM=tolM)


def magCp(K, R):
    return 9 / 8 * K * (1 + rho8) + R * (9 / 8 * rho8 + 1 / 8 + 2.0 ** -17 + 3 * (1 + 2.0 ** -17) / 2 ** 24)


def magNext(K, R, A):
    return np.maximum(((1 + rho8) * (1 + u32) * A + R * (4 / 2 ** 24 + rho8 * (1 + u32))) / (1 - 3 / 2 ** 24),
                      magCp(K, R) / (1 - 2.0 ** -17 - 3 * (1 + 2.0 ** -17) / 2 ** 24))


def magAD(K, R, T):
    A = np.zeros_like(np.asarray(K, dtype=np.float64) + R)
    Dm = np.zeros_like(A)
    for _ in range(T):
        A1 = magNext(K, R, A)
        Dm = (1 + rho8) * (1 + u32) * (1 + 2.0 ** -12) * Dm + (K + R) * (1 + rho8) / 2 ** 11 + (A1 + R + 1) / 2 ** 80
        A = A1
    return A, Dm


def tolV(K, R, T):
    A, Dm = magAD(K, R, T)
    return 9 / 8 * ((A + 2 * Dm) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148) + R / 8 + 2.0 ** -17


def rows_eval(q, k, v, sigma, causal, rows):
    N, D = q.shape
    out = dict(tolO=[], tolV=[], ref=[], p8=[], X=[], pE=[], tolM=[])
    for n in rows:
        keys = np.arange(N) if not causal else np.arange(n + 1)
        s = q[n] @ k[keys].T
        X = sigma * (np.abs(q[n]) @ np.abs(k[keys]).T).max()
        c = consts(D, N, X)
        w = np.exp(sigma * (s - s.max()))
        wv = w[:, None] * v[keys]
        wav = w[:, None] * np.abs(v[keys])
        av = np.abs(v[keys])
        nblk = (2 * (n // 128) + 2) if causal else N // 64
        errS = np.zeros(D); errB = np.zeros(D); part = np.zeros(D)
        for t in range(nblk):
            sel = (keys >= 64 * t) & (keys < 64 * t + 64)
            errS = (1 + c['rhoA']) * errS + c['rhoA'] * np.abs(part) + c['psiA'] * wav[sel].sum(0)
            errB = (1 + c['rhoA']) * errB + (1 + rho8) * c['omega8'] * av[sel].sum(0) + c['absA']
            part = part + wv[sel].sum(0)
        wsum = w.sum()
        ref = part / wsum
        R = np.abs(ref)
        tO = (9 / 8 * (1 + 2.0 ** -22) * (errS / wsum / c['Fp'] + c['pE'] * errB / c['Fp']
                                         + c['betaL'] / c['Fp'] * R)
              + (9 / 8 * 2.0 ** -22 + 1 / 8) * R + 9 / 8 * 2.0 ** -148 + 2.0 ** -17)
        K = np.abs(v[keys] - ref[None, :]).max(0)
        tV = tolV(K, R, nblk)
        out['tolO'].append(tO); out['tolV'].append(tV); out['ref'].append(ref)
        out['p8'].append(wav.sum(0) / wsum / 8); out['X'].append(X); out['pE'].append(c['pE'])
        out['tolM'].append(c['tolM'])
    return {key: np.array(val) for key, val in out.items()}


def run(label, D, N, causal, std, sigma, nrows=96, seed=20, gen=None, rows=None):
    rng = np.random.default_rng(seed)
    if gen is None:
        q, k, v = (e5m2(f16(rng.normal(0, std, (N, D)))) for _ in range(3))
    else:
        q, k, v = gen(N, D)
    if rows is None:
        rows = np.unique(np.linspace(0, N - 1, nrows).astype(int))
    r = rows_eval(q, k, v, sigma, causal, rows)
    a = np.abs(r['ref'])
    okO = a <= r['tolO']
    okV = a <= r['tolV']
    print(f"{label:<9} D={D:<3} N={N:<5} c={int(causal)} X~{np.median(r['X']):6.1f} "
          f"pE med {np.median(r['pE']):6.3g} max {r['pE'].max():6.3g} | tolO {np.median(r['tolO']):8.3g} "
          f"tolV {np.median(r['tolV']):8.3g} |ref| {np.median(a):8.3g} | 0 meets: tolO {100 * okO.mean():5.1f}% "
          f"tolV {100 * okV.mean():5.1f}% Post {100 * (okO & okV).mean():5.1f}% | p8 {100 * np.mean(r['p8'] >= a):5.1f}% "
          f"| tolM {np.median(r['tolM']):.3g}", flush=True)
    return r


if __name__ == "__main__":
    print("test-like: normal(0, 0.5) -> float16 -> float8e5, sm_scale = 0.5 (test_op's fwd / triton-fp8), 96 rows")
    for D in (64, 128):
        for N in (128, 1024, 4096):
            for causal in (False, True):
                run("test", D, N, causal, 0.5, 0.5)
    print("test-like, causal, the first 64 rows only (where the README says the rejected elements are)")
    for D in (64, 128):
        run("test r<64", D, 128, True, 0.5, 0.5, rows=np.arange(64))
    print("benchmark-like: randn -> float16 -> float8e5, sm_scale = 1.3 (bench_flash_attention, triton-fp8)")
    for D in (64, 128):
        for N in (1024, 4096):
            run("bench", D, N, False, 1.0, 1.3)
    for D in (64, 128):
        run("bench", D, 16384, False, 1.0, 1.3, nrows=6)
    print("larger N")
    for D in (64, 128):
        run("test", D, 16384, False, 0.5, 0.5, nrows=6)
    ones = lambda N, D: (np.zeros((N, D)), np.zeros((N, D)), np.ones((N, D)))
    for N in (1024, 2048, 4096, 16384, 65536):
        run("q=0,v=1", 64, N, False, 0, 0.5, nrows=2, gen=ones)

    def last64(N, D):
        v = np.zeros((N, D)); v[N - 64:] = 1.0
        return np.zeros((N, D)), np.zeros((N, D)), v
    run("v=1 last", 64, 65536, False, 0, 0.5, nrows=2, gen=last64)

    print("one key (causal row 0, attnRef = v[0, e], K = 0): tolV / |attnRef|")
    for R in (0.25, 1.0, 3.0):
        print(f"  R = {R}: {tolV(0.0, R, 2) / R:.4f}")
    print("vEff = 0 (K = 0, R = 0): tolV at nblk 2, 256, 1024:",
          ", ".join(f"{tolV(0.0, 0.0, T):.3g}" for T in (2, 256, 1024)))

    print("tolV at K = 1, R = 0 (a_T) and K = 0, R = 1 (b_T); the largest K/R with tolV < R (R = 1, by"
          " bisection); min over K/R in [0, 100] of tolV - (9/8 K + R/8) (the hull margin, R = 1)")
    ratios = np.concatenate([np.linspace(0, 1, 101), np.linspace(1, 100, 100)])
    for T in (1, 2, 4, 8, 16, 24, 31, 32, 64, 256, 1024):
        a_T = tolV(1.0, 0.0, T)
        b_T = tolV(0.0, 1.0, T)
        if tolV(0.0, 1.0, T) < 1:
            lo, hi = 0.0, 10.0
            for _ in range(80):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if tolV(mid, 1.0, T) < 1 else (lo, mid)
            kr = f"{lo:.3f}"
        else:
            kr = "none"
        margin = min(float(tolV(x, 1.0, T) - (9 / 8 * x + 1 / 8)) for x in ratios)
        print(f"  T={T:<5} a_T {a_T:10.4g} b_T {b_T:10.4g}  K/R < {kr:>6}  hull margin {margin:.4g}")

    print("test_op's N_CTX = 128 and 1024, three seeds each (every row at N = 128, 128 rows at 1024)")
    for N in (128, 1024):
        for D in (64, 128):
            for causal in (False, True):
                for seed in (1, 2, 3):
                    run("test all", D, N, causal, 0.5, 0.5, seed=seed, nrows=128)
