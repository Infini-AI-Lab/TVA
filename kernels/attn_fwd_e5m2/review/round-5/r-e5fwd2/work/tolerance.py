"""Reviewer r-e5fwd2, re-review of the claim on `Num.DotE5m2ChunkLoss` (gate 3d7848ee51b5e6a6):
Post's `tolO` as Model.lean now defines it, against the claim before (on `Num.DotE5m2Chunk`), on
data like the host's.

Transcribed from Model.lean and the library:
* `scoreErr` = `Dot8Chunk.errM` (new) or `errB` (before) over the ceil(D/32) chunks of q.k from a
  zero accumulator: `errM(j+1) = (1 + sig) errM(j) + min(rho (X_j + errM(j)), kap a_j + rho' (X_j +
  errM(j))) + sig |s_(j+1)| + alpha`, `X_j = max(|s_j|, m_j)`, `a_j = chunkAbs8` (the chunk's exact
  sum |q k|), `m_j = chunkLift8`; rho = 2^-7, sig = 2^-12, alpha = 2^-126, kap = 2, rho' = 2^-12;
* the constants `scoreE`, `expErr`, `pE`, `dmax`, `psiP`, `gam63`, `stepL`, `qL`, `FhT`, `Flo`, `aL`,
  `Fp`, `cS8`, `cC8`, `cP8`, `lS8`, `lC8`, `lP8`, `eps8`, `etaA`, `gD`, `a8`, `b8`, `kap`, `hw`,
  `denX`, `denS`, `coefD`, `accStep`, `errStep`, `stepC`, `stC`, `tolOW`, `tolO = min(tolOW(wsum Fp),
  tolOW(0))`; `lgB`, `tolM`; `tolV` (unchanged, chunk facts' constants, `iotaV`).
* per element: `centC`, `blkC`, `blkV` (sum w |v|), `blkLW`, `blkA`, `blkAbsV` (sum |v|), `blkVL`,
  `partW` over the row's blocks (`nblk`: 2 (n / 128) + 2 causal, N / 64 otherwise).

`old` is `tolOW(0)` with `scoreErr = errB` (the claim before, which my round-3 run evaluated).
Weights `exp(sigma (s - max s))`: `errC / wsum`, `attnRef` and `stepC`'s choice do not depend on the
scale (each is homogeneous in the weights). `floor` = `sum w |v| / (8 sum w)`.
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
    return np.where(x == 0, 0.0, np.maximum(np.abs(x), 2.0 ** -14))


u32 = 2.0 ** -24
rho, sig, alp, kapC, rhoL = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126, 2.0, 2.0 ** -12
cS8 = (rho + sig) * (2 + rho + sig)
cC8 = rho + sig * (2 + rho + sig)
cP8 = rho * (1 + rho + sig)
lS8 = (rhoL + sig) * (2 + rhoL + sig)
lC8 = (1 + rhoL + sig) * (kapC + sig) + rhoL + sig
lP8 = rhoL * (1 + rhoL + sig)
eps8 = 2.0 ** -17 + 2.0 ** -125
gam63 = (1 + u32) ** 63 - 1
denS = 1 - gam63 - 3 * (1 + gam63) / 2 ** 24


def coefD(cr):
    return 1 / 8 + 9 / 8 * cr + gam63 + 3 * (1 + gam63) / 2 ** 24


def chunks8(D):
    return (D + 31) // 32


def relLift8(D):
    return (1 + (rho + sig)) ** chunks8(D) - 1


def score_parts(qn, kk):
    D = qn.shape[0]
    nc = chunks8(D)
    prod = qn[None, :] * kk
    lp = lift(qn)[None, :] * lift(kk)
    sl = lp.sum(1)
    cs = prod.reshape(len(kk), nc, 32).sum(2)
    ca = np.abs(prod).reshape(len(kk), nc, 32).sum(2)
    cm = lp.reshape(len(kk), nc, 32).max(2)
    eB = np.zeros(len(kk)); eM = np.zeros(len(kk)); ps = np.zeros(len(kk)); first = True
    for c in range(nc):
        ps1 = ps + cs[:, c]
        Xj = np.maximum(np.abs(ps), cm[:, c])
        eB = (1 + rho + sig) * eB + rho * Xj + sig * np.abs(ps1) + alp
        arm1 = rho * (Xj + eM); arm2 = kapC * ca[:, c] + rhoL * (Xj + eM)
        first = first and bool(np.all(arm1 <= arm2))
        eM = (1 + sig) * eM + np.minimum(arm1, arm2) + sig * np.abs(ps1) + alp
        ps = ps1
    return sl, eB, eM, prod.sum(1), first


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
    Flo = (1 - 3 * u32) * (1 - gam63) * (1 - 2.0 ** -22) / pE * qL ** nb
    aL = nb * 2.0 ** -64
    Fp = Flo - aL * pE
    lgB = max(math.log2(FhT(nb) + aL * pE), -math.log2(Fp))
    tolM = (1 + u32) * (lgB + (lgB + expErr + 16) / 2 ** 22 + 2.0 ** -149) + 2.0 ** -150
    etaA = (1 + dmax) * (1 + 2.0 ** -22) - 1
    a8 = 7 / 8 * (1 - 2.0 ** -22) / pE
    b8 = 9 / 8 * pE * (1 + 2.0 ** -22)
    return dict(pE=pE, dmax=dmax, FhT=FhT, Fp=Fp, tolM=tolM, etaA=etaA, a8=a8, b8=b8,
                kap=(a8 + b8) / 2, hw=(b8 - a8) / 2, denX=1 - etaA - 3 * (1 + etaA) / 2 ** 24)


def gD(cs, c):
    return (1 + cs) * ((1 + c['dmax']) * (1 + 2.0 ** -22) * (1 + 2.0 ** -24)) - 1


def errStep(c, cs, cc, cp, EC, C, R, A, Lw, Bc, Bq, Bl):
    g = gD(cs, c)
    return ((1 + g) * EC + g * c['kap'] * np.abs(C) + (g + c['etaA']) * (R + A) * Lw + c['hw'] * Bc
            + cc * c['b8'] * Bq + cp * c['b8'] * Bl)


def accStep(c, cc, cp, cr, R, A, Ba, Bq, Bv):
    return ((1 + 2.0 ** -50) * np.maximum((A + 3 * (1 + c['etaA']) * R / 2 ** 24) / c['denX'], R * coefD(cr) / denS)
            + 2 * (eps8 * Ba + cc * eps8 * Bq + cp * (eps8 + 2.0 ** -14) * Bv + R * (1 + cr) / 2 ** 11
                   + (1 + R + A) / 2 ** 50))


def iotaV(K, R):
    return np.minimum(1.0, 2.0 ** 16 * (K + R))


def tolV(K, R, T):
    A = np.zeros(np.broadcast(K, R).shape); Dm = np.zeros_like(A)
    magCp = (9 / 8 * K * (1 + cS8) + R * (9 / 8 * cS8 + 1 / 8 + 2.0 ** -17 + 3 * (1 + 2.0 ** -17) / 2 ** 24)
             + 9 / 8 * cP8 * iotaV(K, R) / 2 ** 14)
    for _ in range(T):
        A1 = np.maximum(((1 + cS8) * (1 + u32) * A + R * (4 / 2 ** 24 + cS8 * (1 + u32))) / (1 - 3 / 2 ** 24),
                        magCp / (1 - 2.0 ** -17 - 3 * (1 + 2.0 ** -17) / 2 ** 24))
        Dm = ((1 + cS8) * (1 + u32) * (1 + 2.0 ** -12) * Dm + (K + R) * (1 + cS8) / 2 ** 11
              + cP8 * iotaV(K, R) * (K + R + 2.0 ** -13) / 2 ** 8 + (A1 + R + 1) / 2 ** 80)
        A = A1
    return 9 / 8 * ((A + 2 * Dm) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148) + R / 8 + 2.0 ** -17


def chunkmax(x, sel):
    out = 0.0
    for c in range(2):
        s = sel[32 * c:32 * c + 32]
        if s.any():
            out = out + np.where(s[:, None], x[32 * c:32 * c + 32], 0.0).max(0)
    return out


def tolOW(c, blocks, R, wsum, W):
    D = R.shape[0]
    EC = np.zeros(D); A = np.zeros(D); nloss = np.zeros(D)
    for b in blocks:
        eC = errStep(c, cS8, cC8, cP8, EC, b['C'], R, A, b['Lw'], b['Bc'], b['Bc'], b['Bl'])
        aC = accStep(c, cC8, cP8, cC8, R, A, b['Ba'], b['Ba'], b['Bv'])
        if W > 0:
            eL = errStep(c, lS8, lC8, lP8, EC, b['C'], R, A, b['Lw'], b['Bc'], b['BV'], b['Bl'])
            aL_ = accStep(c, lC8, lP8, 0.0, R, A, b['Ba'], b['BVa'], b['Bv'])
            take = eL / W + aL_ < eC / W + aC
            EC = np.where(take, eL, eC); A = np.where(take, aL_, aC); nloss += take
        else:
            EC, A = eC, aC
    tol = 9 / 8 * ((EC / wsum / c['Fp'] + A) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148) + R / 8 + 2.0 ** -17
    return tol, nloss


def row_eval(q, k, v, sigma, causal, n, XZ=None):
    N, D = q.shape
    nkeys = N if not causal else n + 1
    sl, eB, eM, s, first = score_parts(q[n], k[:nkeys])
    X = sigma * sl.max(); Z = sigma * eM.max(); Zold = sigma * eB.max()
    if XZ is not None:
        X, Z = XZ; Zold = Z
    c = consts(D, N, X, Z); cold = consts(D, N, X, Zold)
    w = np.exp(sigma * (s - s.max()))
    vv = v[:nkeys]
    wsum = w.sum()
    ref = (w[:, None] * vv).sum(0) / wsum
    R = np.abs(ref)
    dv = vv - ref[None, :]
    nblk = (2 * (n // 128) + 2) if causal else N // 64

    def blocks(cc):
        out = []; C = np.zeros(D); pw = 0.0
        for t in range(nblk):
            idx = np.arange(64 * t, 64 * t + 64)
            sel = idx < nkeys
            ii = idx[sel]
            wb = np.zeros(64); wb[sel] = w[ii]
            vb = np.zeros((64, D)); vb[sel] = vv[ii]
            db = np.zeros((64, D)); db[sel] = dv[ii]
            out.append(dict(C=C.copy(), Lw=cc['FhT'](t) * pw, Bc=(wb[:, None] * np.abs(db)).sum(0),
                            BV=(wb[:, None] * np.abs(vb)).sum(0), Bl=chunkmax(wb[:, None] * lift(vb), sel),
                            Ba=np.abs(db).sum(0), BVa=np.abs(vb).sum(0), Bv=chunkmax(lift(vb), sel)))
            C = C + (wb[:, None] * db).sum(0)
            pw = pw + wb.sum()
        return out

    bl = blocks(c)
    tW, nloss = tolOW(c, bl, R, wsum, wsum * c['Fp'])
    t0, _ = tolOW(c, bl, R, wsum, 0.0)
    old, _ = tolOW(cold, blocks(cold), R, wsum, 0.0)
    tolO = np.minimum(tW, t0)
    K = np.abs(dv).max(0)
    return dict(tolO=tolO, old=old, tolV=tolV(K, R, nblk), ref=ref, floor=(w[:, None] * np.abs(vv)).sum(0) / wsum / 8,
                X=X, pE=c['pE'], pEold=cold['pE'], tolM=c['tolM'], tolMold=cold['tolM'], first=first,
                frac_loss=nloss.mean() / max(nblk, 1), Zr=Z / max(Zold, 1e-300))


def gen_normal(N, D, std, seed):
    rng = np.random.default_rng(seed)
    return [e5m2(f16(rng.normal(0, std, (N, D)))) for _ in range(3)]


def run(label, D, N, causal, std, sigma, rows, seed):
    q, k, v = gen_normal(N, D, std, seed)
    rs = [row_eval(q, k, v, sigma, causal, n) for n in rows]
    g = lambda key: np.array([r[key] for r in rs])
    tO, old, tV, ref, fl = g('tolO'), g('old'), g('tolV'), g('ref'), g('floor')
    a = np.abs(ref)
    ok = (a <= tO) & (a <= tV); okold = (a <= old) & (a <= tV)
    print(f"{label:<6} D={D:<3} N={N:<5} c={int(causal)} s={seed} | tolO {np.median(old):7.3g} -> {np.median(tO):7.3g} "
          f"(max new/old {np.max(tO / old):.4f}) | x floor {np.median(old / fl):5.2f} -> {np.median(tO / fl):5.2f} "
          f"| 0 meets Post {100 * okold.mean():5.1f}% -> {100 * ok.mean():5.1f}% | pE {np.median(g('pEold')):.4g} -> "
          f"{np.median(g('pE')):.4g}, tolM {np.median(g('tolMold')):.4g} -> {np.median(g('tolM')):.4g}, "
          f"scoreErr first arm always: {bool(g('first').all())}, min Z ratio {g('Zr').min():.6f} | "
          f"loss-arm blocks {100 * np.mean(g('frac_loss')):.1f}%", flush=True)
    return rs


if __name__ == "__main__":
    print("test-like (normal(0, 0.5), sm_scale 0.5): all rows at N = 128, 96 rows otherwise; seeds 1, 2")
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
    print("benchmark-like at N = 2^14, non-causal, 6 rows")
    for D in (64, 128):
        run("bench", D, 16384, False, 1.0, 1.3, np.unique(np.linspace(0, 16383, 6).astype(int)), 1)
    print("one key (causal row 0): tolO / |attnRef|, before -> now (median and range over e and three seeds)")
    for label, D, std, s_ in (("test", 64, 0.5, 0.5), ("test", 128, 0.5, 0.5), ("bench", 64, 1.0, 1.3),
                              ("bench", 128, 1.0, 1.3)):
        ro, rn = [], []
        for seed in (1, 2, 3):
            q, k, v = gen_normal(128, D, std, seed)
            r = row_eval(q, k, v, s_, True, 0)
            m = np.abs(r['ref']) > 0
            ro += list(r['old'][m] / np.abs(r['ref'][m])); rn += list(r['tolO'][m] / np.abs(r['ref'][m]))
        print(f"  {label} D={D}: {np.median(ro):.3f} [{min(ro):.3f}, {max(ro):.3f}] -> {np.median(rn):.3f} "
              f"[{min(rn):.3f}, {max(rn):.3f}]")
    print("q = 0, v = 1 everywhere, D = 64: tolO before -> now")
    for N in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        z = np.zeros((N, 64)); r = row_eval(z, z, np.ones((N, 64)), 0.5, False, 0)
        print(f"  N={N:<6} {np.median(r['old']):.4g} -> {np.median(r['tolO']):.4g} (tolV {np.median(r['tolV']):.4g})")
    print("vEff = 0 (q = k = 0, uniform weights): tolO at X = 0 and at the cap X = Xh (Z large), before -> now")
    for D, N in ((64, 4096), (256, 4096), (64, 16384), (256, 16384), (64, 65536), (256, 65536)):
        z = np.zeros((N, D)); Xh = 2 ** 10 / chunks8(D)
        r0 = row_eval(z, z, z, 0.5, False, 0)
        rc = row_eval(z, z, z, 0.5, False, 0, XZ=(Xh, 1e9))
        print(f"  D={D:<3} N={N:<6} X=0: {np.max(r0['old']):.4g} -> {np.max(r0['tolO']):.4g}; cap: "
              f"{np.max(rc['old']):.4g} -> {np.max(rc['tolO']):.4g}; tolV {np.max(r0['tolV']):.4g} (2^-16 = {2.0 ** -16:.4g})")
