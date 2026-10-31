"""Reviewer r-e5fwd2: Post's output tolerance `tolO` (Model.lean, "The constants") evaluated on data
like the host's, and compared with the exact attention `attnRef` of the same rows.

The formulas are transcribed from `Kernels.AttnFwdE5m2` (Model.lean): `relDot8`, `expErr`, `pE`,
`dmax`, `rhoA`, `psiP`, `psi8`, `psiA`, `omega8`, `absA`, `errS`, `errB`, `gam63`, `stepL`, `qL`,
`Fhi`, `Flo`, `aL`, `Fp`, `betaL`, `tolO`, `lgB`, `tolM`, with `X` the row's own
`max_j sm_scale * scoreAbs(n, j)` over its keys (the smallest `X` Post allows), `nblk` the row's
block count, and the row's weights `exp(sm_scale * score)` (normalised by their maximum: `errS / wsum`
and `attnRef` do not depend on the scale).

Data: `q`, `k`, `v` drawn as the test does (`normal(0, 0.5)` in float16, `sm_scale = 0.5`) or as the
benchmark does (`randn` in float16, `sm_scale = 1.3`), then rounded to float8e5 (round to nearest
even, 2 fraction bits). One head, a sample of rows; every column `e`.

Printed per configuration: the median `X`, the median and largest `tolO`, the median and largest
`|attnRef|`, and the share of the sampled elements on which the output `0` meets Post
(`|0 - attnRef| <= tolO`), i.e. on which Post does not reject a kernel that writes zeros.

`floor`: the same share for a bound that keeps only the two fp8 roundings the kernel cannot avoid
(`p.to(float8e5)`, `1/8` of each weight, and the output's `1/8`), everything else exact:
`9/8 * (sum_j w_j |v_j| / sum_j w_j) / 8 + |attnRef| / 8`. It shows how much of the gap is the fp8
format and how much the `tl.dot` constants.
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


def consts(D, N, X):
    yb = 1.5 * X
    expErr = (relDot8(D) + 2.0 ** -19) * yb + 2.0 ** -50
    pE = 2.0 ** expErr
    dmax = 2 * (yb + expErr) / 2 ** 24 + 2.0 ** -150
    rho8 = relDot8(64)
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
    return dict(pE=pE, rhoA=rhoA, psiA=psiA, rho8=rho8, omega8=omega8, absA=absA, Fp=Fp,
                betaL=betaL, tolM=tolM)


def rows_eval(q, k, v, sigma, causal, rows):
    N, D = q.shape
    T, R, F, Xs, TM = [], [], [], [], []
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
            errB = (1 + c['rhoA']) * errB + (1 + c['rho8']) * c['omega8'] * av[sel].sum(0) + c['absA']
            part = part + wv[sel].sum(0)
        wsum = w.sum()
        ref = part / wsum
        tolO = (9 / 8 * (1 + 2.0 ** -22) * (errS / wsum / c['Fp'] + c['pE'] * errB / c['Fp']
                                           + c['betaL'] / c['Fp'] * np.abs(ref))
                + (9 / 8 * 2.0 ** -22 + 1 / 8) * np.abs(ref) + 9 / 8 * 2.0 ** -148 + 2.0 ** -17)
        floor = 9 / 8 * (wav.sum(0) / wsum) / 8 + np.abs(ref) / 8
        T.append(tolO); R.append(ref); F.append(floor); Xs.append(X); TM.append(c['tolM'])
    return np.array(T), np.array(R), np.array(F), np.array(Xs), np.array(TM)


def run(label, D, N, causal, std, sigma, nrows=96, seed=20, gen=None):
    rng = np.random.default_rng(seed)
    if gen is None:
        q, k, v = (e5m2(f16(rng.normal(0, std, (N, D)))) for _ in range(3))
    else:
        q, k, v = gen(N, D)
    rows = np.unique(np.linspace(0, N - 1, nrows).astype(int))
    tol, ref, floor, X, tolM = rows_eval(q, k, v, sigma, causal, rows)
    a = np.abs(ref)
    zero_ok = np.mean(a <= tol)
    zero_ok_floor = np.mean(a <= floor)
    print(f"{label:<10} D={D:<3} N={N:<5} causal={int(causal)} X~{np.median(X):6.1f} | "
          f"tolO med {np.median(tol):9.3g} max {tol.max():9.3g} | |attnRef| med {np.median(a):8.3g} "
          f"max {a.max():7.3g} | 0 meets Post on {100 * zero_ok:5.1f}% (floor: {100 * zero_ok_floor:5.1f}%) | "
          f"tolM med {np.median(tolM):.3g}", flush=True)


if __name__ == "__main__":
    print("test-like: normal(0, 0.5) -> float16 -> float8e5, sm_scale = 0.5 (test_op's fwd / triton-fp8)")
    for D in (64, 128):
        for N in (128, 1024, 4096):
            for causal in (False, True):
                run("test", D, N, causal, 0.5, 0.5)
    print("benchmark-like: randn -> float16 -> float8e5, sm_scale = 1.3 (bench_flash_attention, triton-fp8)")
    for D in (64, 128):
        for N in (1024, 4096):
            run("bench", D, N, False, 1.0, 1.3)
    print("larger N (covered: N <= 2^16; the host's largest is 2^14)")
    run("test", 64, 16384, False, 0.5, 0.5, nrows=8)
    run("test", 128, 16384, False, 0.5, 0.5, nrows=8)
    run("test", 64, 65536, False, 0.5, 0.5, nrows=3)
    ones = lambda N, D: (np.zeros((N, D)), np.zeros((N, D)), np.ones((N, D)))
    for N in (1024, 4096, 16384, 65536):
        run("q=0,v=1", 64, N, False, 0, 0.5, nrows=2, gen=ones)
