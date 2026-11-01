"""Reviewer r-e5fwd2, re-review after library round 24: Post's tolerances `tolO`, `tolV` and `tolM`
as Model.lean now defines them (gate 4740b3814ba92693), evaluated on data like the host's.

Transcribed from Model.lean ("The constants", "The output's tolerance", "The second bound on the
output") and the library (`Rules.Num.liftMag`, `chunk8`, `chunkSum8`, `rhoChunk8 = 2^-7`,
`sigmaChunk8 = 2^-12`, `alphaChunk8 = 2^-126`; `Dot8Chunk.errB`, `chunkLift8`, `relLift8`):

* per key: `scoreLift = sum_e liftMag(q_e) liftMag(k_e)`; `scoreErr = errB` over the ceil(D/32)
  chunks from a zero accumulator (`errB(c+1) = (1 + rho + sig) errB(c) + rho max(|s_c|, m_c)
  + sig |s_(c+1)| + alpha`, `s_c` the exact partial sums, `m_c` the chunk's largest lifted product);
* per row: `X = max_j sigma scoreLift`, `Z = max_j sigma scoreErr` over the row's keys (the smallest
  bounds Post allows); `scoreE`, `expErr`, `pE`, `dmax`, `psiP`, `gam63`, `stepL`, `qL`, `FhT`, `Fhi`,
  `Flo`, `aL`, `Fp`, `cS8`, `cC8`, `cP8`, `eps8`, `etaA`, `gA`, `a8`, `b8`, `kap`, `hw`, `denX`,
  `denS`, `coefS`, `accNext`, `errNext`, `accA`, `errC`, `tolO`, `lgB`, `tolM`;
* per element: `centC`, `blkC`, `blkA`, `blkLW`, `blkVL`, `partW` over the row's blocks (`nblk`:
  `2 (n / 128) + 2` causal, `N / 64` otherwise); `tolV` with `iotaV`, `magCp`, `magNext`, `magA`,
  `magD` at `K = max_j |vEff_j - attnRef|`.

The weights are `exp(sigma (s - max s))`: `errC / wsum` and `attnRef` do not depend on the scale
(`errNext` is homogeneous in its centred terms, `Cen.errNext_scale`).

Data: `normal(0, 0.5)` (test) or `randn` (benchmark) in float16, then float8e5 (round to nearest
even, 2 fraction bits, subnormals in steps of 2^-16); `sm_scale` 0.5 or 1.3; one head with `vIn`.
`floor` is the share of elements with `sum w |v| / (8 sum w) >= |attnRef|`.
"""
import math
import numpy as np


def e5m2(x):
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    ulp = np.ldexp(1.0, np.maximum(e - 3, -16))
    return np.clip(np.round(x / ulp) * ulp, -57344.0, 57344.0)


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def lift(x):
    a = np.abs(x)
    return np.where(x == 0, 0.0, np.maximum(a, 2.0 ** -14))


u32 = 2.0 ** -24
rho, sig, alp = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126
cS8 = (rho + sig) * (2 + rho + sig)
cC8 = rho + sig * (2 + rho + sig)
cP8 = rho * (1 + rho + sig)
eps8 = 2.0 ** -17 + 2.0 ** -125
gam63 = (1 + u32) ** 63 - 1
denS = 1 - gam63 - 3 * (1 + gam63) / 2 ** 24
coefS = 1 / 8 + 9 / 8 * cC8 + gam63 + 3 * (1 + gam63) / 2 ** 24


def chunks8(D):
    return (D + 31) // 32


def relLift8(D):
    return (1 + (rho + sig)) ** chunks8(D) - 1


def score_parts(qn, kk):
    """scoreLift and scoreErr of row qn against every row of kk."""
    D = qn.shape[0]
    nc = chunks8(D)
    prod = qn[None, :] * kk                       # exact in float64 (fp8 values)
    lp = lift(qn)[None, :] * lift(kk)
    sl = lp.sum(1)
    cs = prod.reshape(len(kk), nc, 32).sum(2)
    cm = lp.reshape(len(kk), nc, 32).max(2)
    err = np.zeros(len(kk)); ps = np.zeros(len(kk))
    for c in range(nc):
        ps1 = ps + cs[:, c]
        err = (1 + rho + sig) * err + rho * np.maximum(np.abs(ps), cm[:, c]) + sig * np.abs(ps1) + alp
        ps = ps1
    return sl, err, prod.sum(1)


def consts(D, N, X, Z):
    yb = 1.5 * X
    scoreE = min(1.5 * Z, relLift8(D) * yb + 2.0 ** -55)
    expErr = scoreE + yb / 2 ** 19 + 2.0 ** -50
    pE = 2.0 ** expErr
    dmax = 2 * (yb + expErr) / 2 ** 24 + 2.0 ** -150
    psiP = pE * (1 + 2.0 ** -22) - 1
    stepL = (1 + 3 * u32) * ((1 + dmax) * (1 + 2.0 ** -22))
    qL = (1 - 3 * u32) * (2 - (1 + dmax) * (1 + 2.0 ** -22))
    nb = N // 64
    FhT = lambda t: (1 + gam63) * (1 + psiP) * stepL ** t
    Fhi = FhT(nb)
    Flo = (1 - 3 * u32) * (1 - gam63) * (1 - 2.0 ** -22) / pE * qL ** nb
    aL = nb * 2.0 ** -64
    Fp = Flo - aL * pE
    lgB = max(math.log2(Fhi + aL * pE), -math.log2(Fp))
    tolM = (1 + u32) * (lgB + (lgB + expErr + 16) / 2 ** 22 + 2.0 ** -149) + 2.0 ** -150
    etaA = (1 + dmax) * (1 + 2.0 ** -22) - 1
    gA = (1 + cS8) * ((1 + dmax) * (1 + 2.0 ** -22) * (1 + 2.0 ** -24)) - 1
    a8 = 7 / 8 * (1 - 2.0 ** -22) / pE
    b8 = 9 / 8 * pE * (1 + 2.0 ** -22)
    return dict(pE=pE, dmax=dmax, FhT=FhT, Fp=Fp, tolM=tolM, etaA=etaA, gA=gA, a8=a8, b8=b8,
                kap=(a8 + b8) / 2, hw=(b8 - a8) / 2, denX=1 - etaA - 3 * (1 + etaA) / 2 ** 24)


def accNext(c, R, A, Ba, Bv):
    return ((1 + 2.0 ** -50) * np.maximum((A + 3 * (1 + c['etaA']) * R / 2 ** 24) / c['denX'], R * coefS / denS)
            + 2 * ((1 + cC8) * eps8 * Ba + cP8 * (eps8 + 2.0 ** -14) * Bv + R * (1 + cC8) / 2 ** 11
                   + (1 + R + A) / 2 ** 50))


def errNext(c, EC, C, R, A, Lw, Bc, Bl):
    return ((1 + c['gA']) * EC + c['gA'] * c['kap'] * np.abs(C) + (c['gA'] + c['etaA']) * (R + A) * Lw
            + (c['hw'] + cC8 * c['b8']) * Bc + cP8 * c['b8'] * Bl)


def iotaV(K, R):
    return np.minimum(1.0, 2.0 ** 16 * (K + R))


def magCp(K, R):
    return (9 / 8 * K * (1 + cS8) + R * (9 / 8 * cS8 + 1 / 8 + 2.0 ** -17 + 3 * (1 + 2.0 ** -17) / 2 ** 24)
            + 9 / 8 * cP8 * iotaV(K, R) / 2 ** 14)


def tolV(K, R, T):
    K = np.asarray(K, dtype=np.float64); R = np.asarray(R, dtype=np.float64)
    A = np.zeros(np.broadcast(K, R).shape); Dm = np.zeros_like(A)
    for _ in range(T):
        A1 = np.maximum(((1 + cS8) * (1 + u32) * A + R * (4 / 2 ** 24 + cS8 * (1 + u32))) / (1 - 3 / 2 ** 24),
                        magCp(K, R) / (1 - 2.0 ** -17 - 3 * (1 + 2.0 ** -17) / 2 ** 24))
        Dm = ((1 + cS8) * (1 + u32) * (1 + 2.0 ** -12) * Dm + (K + R) * (1 + cS8) / 2 ** 11
              + cP8 * iotaV(K, R) * (K + R + 2.0 ** -13) / 2 ** 8 + (A1 + R + 1) / 2 ** 80)
        A = A1
    return 9 / 8 * ((A + 2 * Dm) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148) + R / 8 + 2.0 ** -17


def chunkmax(x, sel):
    """sum over the block's two chunks of 32 keys of the largest x over the attended keys."""
    out = 0.0
    for c in range(2):
        s = sel[32 * c:32 * c + 32]
        xs = x[32 * c:32 * c + 32]
        out = out + np.where(s[:, None], xs, 0.0).max(0) if s.any() else out + 0.0
    return out


def row_eval(q, k, v, sigma, causal, n, XZ=None):
    N, D = q.shape
    nkeys = N if not causal else n + 1
    sl, se, s = score_parts(q[n], k[:nkeys])
    X = sigma * sl.max(); Z = sigma * se.max()
    if XZ is not None:
        X, Z = XZ
    c = consts(D, N, X, Z)
    w = np.exp(sigma * (s - s.max()))
    vv = v[:nkeys]
    wsum = w.sum()
    ref = (w[:, None] * vv).sum(0) / wsum
    R = np.abs(ref)
    dv = vv - ref[None, :]
    nblk = (2 * (n // 128) + 2) if causal else N // 64
    A = np.zeros(D); EC = np.zeros(D); C = np.zeros(D); pw = 0.0
    for t in range(nblk):
        idx = np.arange(64 * t, 64 * t + 64)
        sel = idx < nkeys
        ii = idx[sel]
        wb = np.zeros(64); wb[sel] = w[ii]
        vb = np.zeros((64, D)); vb[sel] = vv[ii]
        db = np.zeros((64, D)); db[sel] = dv[ii]
        Bc = (wb[:, None] * np.abs(db)).sum(0)
        Ba = np.abs(db).sum(0)
        Bl = chunkmax(wb[:, None] * lift(vb), sel)
        Bv = chunkmax(lift(vb), sel)
        EC = errNext(c, EC, C, R, A, c['FhT'](t) * pw, Bc, Bl)
        A = accNext(c, R, A, Ba, Bv)
        C = C + (wb[:, None] * db).sum(0)
        pw = pw + wb.sum()
    tolO = (9 / 8 * ((EC / wsum / c['Fp'] + A) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148)
            + R / 8 + 2.0 ** -17)
    K = np.abs(dv).max(0)
    tV = tolV(K, R, nblk)
    floor = (w[:, None] * np.abs(vv)).sum(0) / wsum / 8
    return dict(tolO=tolO, tolV=tV, ref=ref, floor=floor, X=X, Z=Z, pE=c['pE'], tolM=c['tolM'])


def gen_normal(N, D, std, seed):
    rng = np.random.default_rng(seed)
    return [e5m2(f16(rng.normal(0, std, (N, D)))) for _ in range(3)]


def run(label, D, N, causal, std, sigma, rows, seed, gen=None):
    q, k, v = gen(N, D) if gen is not None else gen_normal(N, D, std, seed)
    rs = [row_eval(q, k, v, sigma, causal, n) for n in rows]
    g = lambda key: np.array([r[key] for r in rs])
    tO, tV, ref, fl = g('tolO'), g('tolV'), g('ref'), g('floor')
    a = np.abs(ref)
    ok = (a <= tO) & (a <= tV)
    print(f"{label:<8} D={D:<3} N={N:<5} c={int(causal)} s={seed} X~{np.median(g('X')):6.1f} max {g('X').max():6.1f} "
          f"pE med {np.median(g('pE')):6.3g} max {g('pE').max():6.3g} | tolO {np.median(tO):8.3g} "
          f"tolV {np.median(tV):7.3g} tolM {np.median(g('tolM')):7.3g} |ref| {np.median(a):7.3g} "
          f"floor {np.median(fl):6.3g} | 0 meets Post {100 * ok.mean():5.1f}% (tolO {100 * (a <= tO).mean():5.1f}%) "
          f"| floor>=|ref| {100 * (fl >= a).mean():5.1f}% | tolO/floor {np.median(tO / fl):5.2f}", flush=True)
    return rs


if __name__ == "__main__":
    print("test-like (normal(0, 0.5), sm_scale 0.5); all rows at N = 128, 96 rows otherwise; seeds 1, 2")
    for D in (64, 128):
        for N in (128, 1024, 4096):
            for causal in (False, True):
                rows = np.arange(N) if N == 128 else np.unique(np.linspace(0, N - 1, 96).astype(int))
                for seed in (1, 2):
                    run("test", D, N, causal, 0.5, 0.5, rows, seed)
    print("benchmark-like (randn, sm_scale 1.3), N = 1024, 96 rows; seeds 1, 2")
    for D in (64, 128):
        for causal in (False, True):
            for seed in (1, 2):
                run("bench", D, 1024, causal, 1.0, 1.3, np.unique(np.linspace(0, 1023, 96).astype(int)), seed)
    print("one key (causal row 0): tolO / |attnRef| and tolV / |attnRef| (median, range over e and seeds)")
    for label, D, std, s_ in (("test", 64, 0.5, 0.5), ("test", 128, 0.5, 0.5), ("bench", 64, 1.0, 1.3),
                              ("bench", 128, 1.0, 1.3)):
        ratO, ratV = [], []
        for seed in (1, 2, 3):
            q, k, v = gen_normal(128, D, std, seed)
            r = row_eval(q, k, v, s_, True, 0)
            m = np.abs(r['ref']) > 0
            ratO += list(r['tolO'][m] / np.abs(r['ref'][m])); ratV += list(r['tolV'][m] / np.abs(r['ref'][m]))
        print(f"  {label} D={D}: tolO/|r| med {np.median(ratO):.3f} [{min(ratO):.3f}, {max(ratO):.3f}]; "
              f"tolV/|r| med {np.median(ratV):.3f} [{min(ratV):.3f}, {max(ratV):.3f}]")
    print("q = 0, v = 1 everywhere, D = 64")
    ones = lambda N, D: (np.zeros((N, D)), np.zeros((N, D)), np.ones((N, D)))
    for N in (1024, 2048, 4096, 16384, 65536):
        r = row_eval(*ones(N, 64), 0.5, False, 0)
        print(f"  N={N:<6} tolO {np.median(r['tolO']):.4g} tolV {np.median(r['tolV']):.4g} tolM {r['tolM']:.3g}")
    print("vEff = 0 (a head without vIn): tolV at K = R = 0; tolO with q = k = 0 (uniform weights) at X = Z = 0,"
          " and at the cap X = Xh with the whole-dot bound (Z large; allowed, as Post holds for every larger X, Z);"
          " also at the cap with test-like q, k (row 0)")
    for D, N in ((64, 4096), (128, 4096), (64, 16384), (128, 16384), (256, 65536)):
        Xh = 2 ** 10 / chunks8(D)
        z = np.zeros((N, D))
        r0 = row_eval(z, z, z, 0.5, False, 0)
        rc = row_eval(z, z, z, 0.5, False, 0, XZ=(Xh, 1e9))
        qq, kk, _ = gen_normal(N, D, 0.5, 5)
        rr = row_eval(qq, kk, z, 0.5, False, 0, XZ=(Xh, 1e9))
        cc = consts(D, N, Xh, 1e9)
        print(f"  D={D:<3} N={N:<6} tolV {np.max(r0['tolV']):.4g} | tolO, q = k = 0: at X=0 {np.max(r0['tolO']):.4g}, "
              f"at the cap {np.max(rc['tolO']):.4g}; test-like q, k at the cap {np.max(rr['tolO']):.4g} (pE {cc['pE']:.4g})")
    print("2^-16 (the smallest nonzero float8e5) against tolV at vEff = 0:", 2.0 ** -16)
    print("test-like causal rows at N = 1024 and 4096 with 24 rows (as the README samples), and row 0 alone")
    for D in (64, 128):
        for N in (1024, 4096):
            for seed in (1, 2):
                run("test 24r", D, N, True, 0.5, 0.5, np.unique(np.linspace(0, N - 1, 24).astype(int)), seed)
            run("row 0", D, N, True, 0.5, 0.5, [0], 1)
