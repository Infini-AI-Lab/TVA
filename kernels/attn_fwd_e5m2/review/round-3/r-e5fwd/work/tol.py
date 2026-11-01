"""Reviewer r-e5fwd, the re-proof on the chunk facts (gate 4740b3814ba92693): evaluate the
specification's tolerances tolO, tolV, tolM exactly as Model.lean now writes them, on data like
the host's.

For one head with vIn and sampled rows n, every column e:
  - X = max over the row's keys of sigma * scoreLift (lifted magnitudes, liftMag: 0 for 0, else
    max(|x|, 2^-14)); Z = max over the keys of sigma * scoreErr (Dot8Chunk.errB over the score's
    chunks of 32, from the exact chunk sums and each chunk's largest lifted product). Both are the
    smallest values Post admits, and every tolerance grows with them.
  - pE, dmax, FhT, Flo, Fp; accA, errC (centC, blkC, blkA, blkLW, blkVL, partW) at the row's nblk;
    tolO; tolM; K = max over keys |vEff_j - attnRef| and tolV (cS8, cP8, iotaV).
It reports medians, the share where Post accepts o = 0 (|attnRef| <= tolO and <= tolV), the fp8
floor sum w|v| / (8 sum w), causal row 0 (one key), and the largest sigma*scoreLift against Xh.
Then v = 1 / q = 0 rows, and the heads without vIn (vEff = 0) at the cap.
Data: test = normal(0, 0.5) through float16 to float8e5, sm_scale 0.5; bench = randn, sm_scale 1.3.
Run: python3 -I tol.py
"""
import numpy as np

U = 2.0 ** -24
RHO, SIG, ALP = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126
CS8 = (RHO + SIG) * (2 + RHO + SIG)
CC8 = RHO + SIG * (2 + RHO + SIG)
CP8 = RHO * (1 + RHO + SIG)
EPS8 = 2.0 ** -17 + 2.0 ** -125
GAM63 = (1 + U) ** 63 - 1
DENS = 1 - GAM63 - 3 * (1 + GAM63) / 2 ** 24
COEFS = 1 / 8 + 9 / 8 * CC8 + GAM63 + 3 * (1 + GAM63) / 2 ** 24


def e5m2(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.zeros_like(x)
    nz = a > 0
    e = np.maximum(np.floor(np.log2(a[nz])), -14.0)
    q = 2.0 ** (e - 2)
    out[nz] = np.round(a[nz] / q) * q
    return np.sign(x) * out


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def lift(x):
    a = np.abs(x)
    return np.where(a == 0, 0.0, np.maximum(a, 2.0 ** -14))


def chunks8(K):
    return (K + 31) // 32


def relLift8(K):
    return (1 + RHO + SIG) ** chunks8(K) - 1


class Const:
    def __init__(self, D, N, X, Zs):
        scoreE = min(1.5 * Zs, relLift8(D) * 1.5 * X + 2.0 ** -55)
        self.expErr = scoreE + 1.5 * X / 2 ** 19 + 2.0 ** -50
        pe = 2.0 ** self.expErr
        self.pE = pe
        self.dm = dm = 2 * (1.5 * X + self.expErr) / 2 ** 24 + 2.0 ** -150
        psiP = pe * (1 + 2.0 ** -22) - 1
        self.stepL = (1 + 3 * U) * ((1 + dm) * (1 + 2.0 ** -22))
        qL = (1 - 3 * U) * (2 - (1 + dm) * (1 + 2.0 ** -22))
        self.FhT0 = (1 + GAM63) * (1 + psiP)
        nb = N // 64
        Fhi = self.FhT0 * self.stepL ** nb
        Flo = (1 - 3 * U) * (1 - GAM63) * (1 - 2.0 ** -22) / pe * qL ** nb
        aL = nb * 2.0 ** -64
        self.Fp = Flo - aL * pe
        lgB = max(np.log2(Fhi + aL * pe), -np.log2(self.Fp))
        self.tolM = (1 + U) * (lgB + (lgB + self.expErr + 16) / 2 ** 22 + 2.0 ** -149) + 2.0 ** -150
        self.etaA = (1 + dm) * (1 + 2.0 ** -22) - 1
        self.gA = (1 + CS8) * ((1 + dm) * (1 + 2.0 ** -22) * (1 + 2.0 ** -24)) - 1
        a8 = 7 / 8 * (1 - 2.0 ** -22) / pe
        self.b8 = b8 = 9 / 8 * pe * (1 + 2.0 ** -22)
        self.kap = (a8 + b8) / 2
        self.hw = (b8 - a8) / 2
        self.denX = 1 - self.etaA - 3 * (1 + self.etaA) / 2 ** 24


def tolV(K, R, T):
    iota = np.minimum(1.0, 2.0 ** 16 * (K + R))
    magCp = (9 / 8 * K * (1 + CS8) + R * (9 / 8 * CS8 + 1 / 8 + 2.0 ** -17 + 3 * (1 + 2.0 ** -17) / 2 ** 24)
             + 9 / 8 * CP8 * iota / 2 ** 14)
    A = 0.0 * (K + R)
    Dd = 0.0 * (K + R)
    for _ in range(T):
        A = np.maximum(((1 + CS8) * (1 + 2.0 ** -24) * A + R * (4 / 2 ** 24 + CS8 * (1 + 2.0 ** -24))) / (1 - 3 / 2 ** 24),
                       magCp / (1 - 2.0 ** -17 - 3 * (1 + 2.0 ** -17) / 2 ** 24))
        Dd = ((1 + CS8) * (1 + 2.0 ** -24) * (1 + 2.0 ** -12) * Dd + (K + R) * (1 + CS8) / 2 ** 11
              + CP8 * iota * (K + R + 2.0 ** -13) / 2 ** 8 + (A + R + 1) / 2 ** 80)
    return 9 / 8 * ((A + 2 * Dd) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148) + R / 8 + 2.0 ** -17


def tolO_row(c, w, keymask, v, r, nblk, N):
    """tolO for one row: w (N,) exact weights (any common scale), keymask (N,), v (N, D), r (D,)."""
    D = v.shape[1]
    nb = N // 64
    wk = np.where(keymask, w, 0.0)
    dv = v - r[None, :]
    blkA = (keymask[:, None] * np.abs(dv)).reshape(nb, 64, D).sum(1)
    blkC = (wk[:, None] * np.abs(dv)).reshape(nb, 64, D).sum(1)
    blkS = (wk[:, None] * dv).reshape(nb, 64, D).sum(1)
    centC = np.vstack([np.zeros((1, D)), np.cumsum(blkS, axis=0)])      # centC t = sum over blocks < t
    partW = np.concatenate([[0.0], np.cumsum(wk.reshape(nb, 64).sum(1))])
    lv = lift(v)
    blkLW = (wk[:, None] * lv).reshape(nb, 2, 32, D).max(axis=2).sum(axis=1)
    blkVL = (keymask[:, None] * lv).reshape(nb, 2, 32, D).max(axis=2).sum(axis=1)
    R = np.abs(r)
    A = np.zeros(D)
    EC = np.zeros(D)
    FhT = c.FhT0
    for t in range(nblk):
        EC = ((1 + c.gA) * EC + c.gA * c.kap * np.abs(centC[t]) + (c.gA + c.etaA) * (R + A) * (FhT * partW[t])
              + (c.hw + CC8 * c.b8) * blkC[t] + CP8 * c.b8 * blkLW[t])
        A = ((1 + 2.0 ** -50) * np.maximum((A + 3 * (1 + c.etaA) * R / 2 ** 24) / c.denX, R * COEFS / DENS)
             + 2 * ((1 + CC8) * EPS8 * blkA[t] + CP8 * (EPS8 + 2.0 ** -14) * blkVL[t]
                    + R * (1 + CC8) / 2 ** 11 + (1 + R + A) / 2 ** 50))
        FhT = FhT * c.stepL
    wsum = wk.sum()
    return (9 / 8 * ((EC / wsum / c.Fp + A) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148)
            + R / 8 + 2.0 ** -17)


def score_bounds(q, k, sigma, rows, keymask):
    """X and Z per row (max over keys of sigma*scoreLift and sigma*scoreErr), and scores."""
    N, D = q.shape
    nc = chunks8(D)
    qr = q[rows]
    cs = [qr[:, 32 * i:32 * i + 32] @ k[:, 32 * i:32 * i + 32].T for i in range(nc)]
    lq, lk = lift(qr).astype(np.float32), lift(k).astype(np.float32)
    sl = lift(qr) @ lift(k).T
    E = np.zeros_like(cs[0])
    ps = np.zeros_like(cs[0])
    for i in range(nc):
        m = np.empty_like(cs[0])
        for j0 in range(0, N, 1024):
            m[:, j0:j0 + 1024] = (lq[:, None, 32 * i:32 * i + 32] * lk[None, j0:j0 + 1024, 32 * i:32 * i + 32]).max(axis=2)
        nxt = ps + cs[i]
        E = (1 + RHO + SIG) * E + RHO * np.maximum(np.abs(ps), m) + SIG * np.abs(nxt) + ALP
        ps = nxt
    S = sum(cs)
    X = np.where(keymask, sigma * sl, -np.inf).max(axis=1)
    Zs = np.where(keymask, sigma * E, -np.inf).max(axis=1)
    return S, X, Zs, (sigma * sl).max()


def report(name, q, k, v, sigma, causal, rows):
    N, D = q.shape
    rows = np.asarray(rows)
    j = np.arange(N)
    keymask = np.ones((len(rows), N), bool) if not causal else (j[None, :] <= rows[:, None])
    S, X, Zs, slmax = score_bounds(q, k, sigma, rows, keymask)
    tO, tV, tM, att, fl, pes = [], [], [], [], [], []
    for i, n in enumerate(rows):
        km = keymask[i]
        z = np.where(km, sigma * S[i], -np.inf)
        w = np.exp(z - z.max())
        r = (w[:, None] * v).sum(0) / w.sum()
        nblk = 2 * (n // 128) + 2 if causal else N // 64
        c = Const(D, N, X[i], Zs[i])
        tO.append(tolO_row(c, w, km, v, r, nblk, N))
        K = np.abs(v[km] - r[None, :]).max(axis=0)
        tV.append(tolV(K, np.abs(r), nblk))
        tM.append(c.tolM)
        att.append(r)
        fl.append((w[:, None] * np.abs(v)).sum(0) / w.sum() / 8)
        pes.append(c.pE)
    tO, tV, att, fl = map(np.array, (tO, tV, att, fl))
    a = np.abs(att)
    acc = np.mean((tO >= a) & (tV >= a))
    line = (f"{name}: D={D} N={N} causal={causal} rows={len(rows)}  max sigma*scoreLift {slmax:.1f} (Xh {2 ** 10 / chunks8(D):.0f})"
            f"  pE median {np.median(pes):.3g} max {np.max(pes):.3g}\n"
            f"    tolO median {np.median(tO):.3g}  tolV median {np.median(tV):.3g}  tolM median {np.median(tM):.3g}"
            f"  |attnRef| median {np.median(a):.3g}  floor median {np.median(fl):.3g}\n"
            f"    0 meets Post on {100 * acc:.1f}% (rejected {100 * (1 - acc):.1f}%);  tolO/floor median {np.median(tO / fl):.3g};"
            f"  floor >= |attnRef| on {100 * np.mean(fl >= a):.1f}%")
    if causal and rows[0] == 0:
        line += (f"\n    row 0 (one key): tolO/|attnRef| median {np.median(tO[0] / a[0]):.3g} max {(tO[0] / a[0]).max():.3g},"
                 f" tolV/|attnRef| median {np.median(tV[0] / a[0]):.3g}")
    print(line, flush=True)


def data(kind, N, D, rng):
    if kind == 'test':
        g = lambda: e5m2(f16(rng.normal(0.0, 0.5, size=(N, D))))
        return g(), g(), g(), 0.5
    g = lambda: e5m2(f16(rng.standard_normal(size=(N, D))))
    return g(), g(), g(), float(np.float32(1.3))


def main():
    rng = np.random.default_rng(31)
    for D in (64, 128):
        for N in (128, 1024, 4096):
            for causal in (False, True):
                q, k, v, s = data('test', N, D, rng)
                rows = np.arange(N) if N == 128 else np.concatenate([np.arange(0, 64), np.arange(N // 2, N // 2 + 64)])
                report('test ', q, k, v, s, causal, rows)
    for D in (64, 128):
        for causal in (False, True):
            q, k, v, s = data('bench', 1024, D, rng)
            report('bench', q, k, v, s, causal, np.concatenate([np.arange(0, 64), np.arange(512, 576)]))
    # q = 0, v = 1 everywhere: scores 0, scoreLift 0, scoreErr = alpha terms only
    for N in (1024, 4096, 2 ** 14, 2 ** 16):
        nb = N // 64
        c = Const(64, N, 0.0, 0.5 * 2 * ALP * 3)
        w = np.ones(N)
        v = np.ones((N, 1))
        r = np.ones(1)
        t_o = tolO_row(c, w, np.ones(N, bool), v, r, nb, N)[0]
        print(f"ones : D=64 N={N}  tolO {t_o:.4g}  tolV {float(tolV(0.0, 1.0, nb)):.4g}  tolM {c.tolM:.3g}  (attnRef = 1)", flush=True)
    # heads without vIn: vEff = 0, attnRef = 0; tolV at K = R = 0, tolO at the cap X = Xh (whole-dot branch)
    for D, N in ((64, 4096), (64, 2 ** 14), (128, 2 ** 14), (256, 2 ** 16)):
        nb = N // 64
        X = 2 ** 10 / chunks8(D)
        c = Const(D, N, X, 1e9)
        t_o = tolO_row(c, np.ones(N), np.ones(N, bool), np.zeros((N, 1)), np.zeros(1), nb, N)[0]
        print(f"vEff0: D={D} N={N}  at the cap X={X:.0f}: pE {c.pE:.3g}, tolO {t_o:.3g};  tolV(K=R=0) {float(tolV(0.0, 0.0, nb)):.4g}"
              f"  (a float8e5 below 2^-16 = {2.0 ** -16:.3g} in magnitude is 0)", flush=True)


def uniform():
    """Rows spread evenly over the head (every row at N = 128, every 8th at 1024, every 32nd at
    4096), three seeds: the shares as a uniform sample of the head's rows would give them."""
    for seed in (41, 42, 43):
        rng = np.random.default_rng(seed)
        for kind, Ns in (('test', (128, 1024, 4096)), ('bench', (1024,))):
            for D in (64, 128):
                for N in Ns:
                    for causal in (False, True):
                        q, k, v, s = data(kind, N, D, rng)
                        step = max(1, N // 128)
                        report(f"{kind:5s} seed {seed} uniform", q, k, v, s, causal, np.arange(0, N, step))


if __name__ == '__main__':
    import sys
    uniform() if sys.argv[1:] == ['uniform'] else main()
