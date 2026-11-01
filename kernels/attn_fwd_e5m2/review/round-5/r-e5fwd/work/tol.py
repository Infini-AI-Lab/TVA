"""Reviewer r-e5fwd, the re-proof on the loss facts (DotE5m2ChunkLoss, gate 3d7848ee51b5e6a6):
evaluate tolO (stepC, tolOW, the minimum with W = 0), tolV and tolM exactly as Model.lean now
writes them, on data like the host's, and the old tolO (the chunk facts' recursion, tolOW at W = 0
with the old scoreErr = errB) for comparison.

For one head with vIn and sampled rows n, every column e:
  - X = max over keys of sigma*scoreLift; Z = max over keys of sigma*scoreErr, scoreErr now
    Dot8Chunk.errM (each chunk the smaller arm: min(rho*(X_j + E), kappa*A_j + rho'*(X_j + E))),
    Zold the same with errB;
  - tolO = min(tolOW(W = wsum*Fp), tolOW(0)); stepC picks per block and per column the arm whose
    errStep/W + accStep is smaller (loss arm: lS8, lC8, lP8, Bq = blkV, Bqa = blkAbsV, cr = 0);
  - tolV (unchanged), tolM (now at the new Z).
Reports medians, the share where Post accepts o = 0, the share of rows where the new tolO is below
the old, the share of blocks on the loss arm, causal row 0, then v = 1 rows and heads without vIn.
Data: test = normal(0, 0.5) through float16 to float8e5, sm_scale 0.5; bench = randn, sm_scale 1.3.
Run: python3 -I tol.py          (N <= 4096, the benchmark at N = 1024, v = 1, vEff = 0)
     python3 -I tol.py big      (the benchmark at N = 2^14, one head, every 97th row)
"""
import sys
import numpy as np

U = 2.0 ** -24
RHO, SIG, ALP = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126
KAP, RHOL = 2.0, 2.0 ** -12
CS8 = (RHO + SIG) * (2 + RHO + SIG)
CC8 = RHO + SIG * (2 + RHO + SIG)
CP8 = RHO * (1 + RHO + SIG)
LS8 = (RHOL + SIG) * (2 + RHOL + SIG)
LC8 = (1 + RHOL + SIG) * (KAP + SIG) + RHOL + SIG
LP8 = RHOL * (1 + RHOL + SIG)
EPS8 = 2.0 ** -17 + 2.0 ** -125
GAM63 = (1 + U) ** 63 - 1
DENS = 1 - GAM63 - 3 * (1 + GAM63) / 2 ** 24


def coefD(cr):
    return 1 / 8 + 9 / 8 * cr + GAM63 + 3 * (1 + GAM63) / 2 ** 24


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
        a8 = 7 / 8 * (1 - 2.0 ** -22) / pe
        self.b8 = 9 / 8 * pe * (1 + 2.0 ** -22)
        self.kap = (a8 + self.b8) / 2
        self.hw = (self.b8 - a8) / 2
        self.denX = 1 - self.etaA - 3 * (1 + self.etaA) / 2 ** 24

    def gD(self, cs):
        return (1 + cs) * ((1 + self.dm) * (1 + 2.0 ** -22) * (1 + 2.0 ** -24)) - 1

    def errStep(self, cs, cc, cp, EC, C, R, A, Lw, Bc, Bq, Bl):
        g = self.gD(cs)
        return ((1 + g) * EC + g * self.kap * np.abs(C) + (g + self.etaA) * (R + A) * Lw
                + self.hw * Bc + cc * self.b8 * Bq + cp * self.b8 * Bl)

    def accStep(self, cc, cp, cr, R, A, Ba, Bq, Bv):
        return ((1 + 2.0 ** -50) * np.maximum((A + 3 * (1 + self.etaA) * R / 2 ** 24) / self.denX, R * coefD(cr) / DENS)
                + 2 * (EPS8 * Ba + cc * EPS8 * Bq + cp * (EPS8 + 2.0 ** -14) * Bv + R * (1 + cr) / 2 ** 11
                       + (1 + R + A) / 2 ** 50))


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


def block_sums(w, keymask, v, r, N):
    D = v.shape[1]
    nb = N // 64
    wk = np.where(keymask, w, 0.0)
    dv = v - r[None, :]
    s = {}
    s['A'] = (keymask[:, None] * np.abs(dv)).reshape(nb, 64, D).sum(1)
    s['C'] = (wk[:, None] * np.abs(dv)).reshape(nb, 64, D).sum(1)
    s['V'] = (wk[:, None] * np.abs(v)).reshape(nb, 64, D).sum(1)
    s['AbsV'] = (keymask[:, None] * np.abs(v)).reshape(nb, 64, D).sum(1)
    blkS = (wk[:, None] * dv).reshape(nb, 64, D).sum(1)
    s['cent'] = np.vstack([np.zeros((1, D)), np.cumsum(blkS, axis=0)])
    s['partW'] = np.concatenate([[0.0], np.cumsum(wk.reshape(nb, 64).sum(1))])
    lv = lift(v)
    s['LW'] = (wk[:, None] * lv).reshape(nb, 2, 32, D).max(axis=2).sum(axis=1)
    s['VL'] = (keymask[:, None] * lv).reshape(nb, 2, 32, D).max(axis=2).sum(axis=1)
    s['wsum'] = wk.sum()
    return s


def tolOW(c, s, r, nblk, W):
    """tolOW along stepC at W; returns the tolerance and the share of (block, column) on the loss arm."""
    D = r.shape[0]
    R = np.abs(r)
    EC = np.zeros(D)
    A = np.zeros(D)
    FhT = c.FhT0
    nloss = 0
    for t in range(nblk):
        Lw = FhT * s['partW'][t]
        ec_c = c.errStep(CS8, CC8, CP8, EC, s['cent'][t], R, A, Lw, s['C'][t], s['C'][t], s['LW'][t])
        ac_c = c.accStep(CC8, CP8, CC8, R, A, s['A'][t], s['A'][t], s['VL'][t])
        if W > 0:
            ec_l = c.errStep(LS8, LC8, LP8, EC, s['cent'][t], R, A, Lw, s['C'][t], s['V'][t], s['LW'][t])
            ac_l = c.accStep(LC8, LP8, 0.0, R, A, s['A'][t], s['AbsV'][t], s['VL'][t])
            pick = ec_l / W + ac_l < ec_c / W + ac_c
            nloss += pick.sum()
            EC, A = np.where(pick, ec_l, ec_c), np.where(pick, ac_l, ac_c)
        else:
            EC, A = ec_c, ac_c
        FhT = FhT * c.stepL
    tol = 9 / 8 * ((EC / s['wsum'] / c.Fp + A) * (1 + 2.0 ** -22) + R / 2 ** 22 + 2.0 ** -148) + R / 8 + 2.0 ** -17
    return tol, nloss / max(1, nblk * D)


def tolO_both(D, N, X, Z, Zold, s, r, nblk):
    """(new tolO, old tolO, share of loss blocks, new Const)."""
    c = Const(D, N, X, Z)
    tW, share = tolOW(c, s, r, nblk, s['wsum'] * c.Fp)
    t0, _ = tolOW(c, s, r, nblk, 0.0)
    co = Const(D, N, X, Zold)
    told, _ = tolOW(co, s, r, nblk, 0.0)
    return np.minimum(tW, t0), told, share, c


def score_bounds(q, k, sigma, rows, keymask):
    N, D = q.shape
    nc = chunks8(D)
    qr = q[rows]
    cs = [qr[:, 32 * i:32 * i + 32] @ k[:, 32 * i:32 * i + 32].T for i in range(nc)]
    ab = [np.abs(qr[:, 32 * i:32 * i + 32]) @ np.abs(k[:, 32 * i:32 * i + 32]).T for i in range(nc)]
    lq, lk = lift(qr).astype(np.float32), lift(k).astype(np.float32)
    sl = lift(qr) @ lift(k).T
    EB = np.zeros_like(cs[0])
    EM = np.zeros_like(cs[0])
    ps = np.zeros_like(cs[0])
    for i in range(nc):
        m = np.empty_like(cs[0])
        for j0 in range(0, N, 1024):
            m[:, j0:j0 + 1024] = (lq[:, None, 32 * i:32 * i + 32] * lk[None, j0:j0 + 1024, 32 * i:32 * i + 32]).max(axis=2)
        nxt = ps + cs[i]
        Xj = np.maximum(np.abs(ps), m)
        EB = (1 + RHO + SIG) * EB + RHO * Xj + SIG * np.abs(nxt) + ALP
        EM = (1 + SIG) * EM + np.minimum(RHO * (Xj + EM), KAP * ab[i] + RHOL * (Xj + EM)) + SIG * np.abs(nxt) + ALP
        ps = nxt
    S = sum(cs)
    X = np.where(keymask, sigma * sl, -np.inf).max(axis=1)
    Z = np.where(keymask, sigma * EM, -np.inf).max(axis=1)
    Zold = np.where(keymask, sigma * EB, -np.inf).max(axis=1)
    return S, X, Z, Zold, float(np.max(np.where(keymask, EM < EB - 1e-30, False).mean(axis=1)))


def report(name, q, k, v, sigma, causal, rows):
    N, D = q.shape
    rows = np.asarray(rows)
    j = np.arange(N)
    keymask = np.ones((len(rows), N), bool) if not causal else (j[None, :] <= rows[:, None])
    S, X, Z, Zold, zless = score_bounds(q, k, sigma, rows, keymask)
    tN, tOld, tV, tM, att, fl, shares, pes = [], [], [], [], [], [], [], []
    for i, n in enumerate(rows):
        km = keymask[i]
        z = np.where(km, sigma * S[i], -np.inf)
        w = np.exp(z - z.max())
        r = (w[:, None] * v).sum(0) / w.sum()
        nblk = 2 * (n // 128) + 2 if causal else N // 64
        s = block_sums(w, km, v, r, N)
        tn, to, sh, c = tolO_both(D, N, X[i], Z[i], Zold[i], s, r, nblk)
        tN.append(tn); tOld.append(to); shares.append(sh); tM.append(c.tolM); pes.append(c.pE)
        K = np.abs(v[km] - r[None, :]).max(axis=0)
        tV.append(tolV(K, np.abs(r), nblk))
        att.append(r)
        fl.append((w[:, None] * np.abs(v)).sum(0) / w.sum() / 8)
    tN, tOld, tV, att, fl = map(np.array, (tN, tOld, tV, att, fl))
    a = np.abs(att)
    accN = np.mean((tN >= a) & (tV >= a))
    accO = np.mean((tOld >= a) & (tV >= a))
    line = (f"{name}: D={D} N={N} causal={causal} rows={len(rows)}  max X {X.max():.1f}  pE median {np.median(pes):.3g}"
            f"  Z<Zold (beyond rounding) on {100 * np.mean(Z < Zold * (1 - 1e-9)):.0f}% of rows\n"
            f"    tolO old median {np.median(tOld):.3g} -> new {np.median(tN):.3g} (new/old median {np.median(tN / tOld):.3g},"
            f" max {np.max(tN / tOld):.4g}; below old on {100 * np.mean(tN < tOld * (1 - 1e-12)):.1f}% of elements)"
            f"  loss-arm blocks {100 * np.mean(shares):.0f}%\n"
            f"    tolV median {np.median(tV):.3g}  tolM median {np.median(tM):.3g}  |attnRef| median {np.median(a):.3g}"
            f"  tolO/floor median {np.median(tN / fl):.3g}\n"
            f"    0 meets Post: old {100 * accO:.1f}% -> new {100 * accN:.1f}%")
    if causal and rows[0] == 0:
        line += f"\n    row 0 (one key): tolO/|attnRef| old {np.median(tOld[0] / a[0]):.3g} -> new {np.median(tN[0] / a[0]):.3g}"
    print(line, flush=True)


def data(kind, N, D, rng):
    if kind == 'test':
        g = lambda: e5m2(f16(rng.normal(0.0, 0.5, size=(N, D))))
        return g(), g(), g(), 0.5
    g = lambda: e5m2(f16(rng.standard_normal(size=(N, D))))
    return g(), g(), g(), float(np.float32(1.3))


def main():
    for seed in (41, 42):
        rng = np.random.default_rng(seed)
        for kind, Ns in (('test', (128, 1024, 4096)), ('bench', (1024,))):
            for D in (64, 128):
                for N in Ns:
                    for causal in (False, True):
                        q, k, v, s = data(kind, N, D, rng)
                        report(f"{kind:5s} seed {seed}", q, k, v, s, causal, np.arange(0, N, max(1, N // 128)))
    # q = 0, v = 1 everywhere
    for N in (1024, 2048, 4096, 8192, 2 ** 14, 2 ** 15, 2 ** 16):
        nb = N // 64
        w = np.ones(N)
        v = np.ones((N, 1))
        r = np.ones(1)
        s = block_sums(w, np.ones(N, bool), v, r, N)
        tn, to, sh, c = tolO_both(64, N, 0.0, 3 * ALP, 3 * ALP, s, r, nb)
        print(f"ones : D=64 N={N}  tolO old {to[0]:.4g} -> new {tn[0]:.4g}  (loss-arm blocks {100 * sh:.0f}%)", flush=True)
    # heads without vIn at the cap (whole-dot score bound), and tolV
    for D, N in ((64, 4096), (64, 2 ** 14), (128, 2 ** 14), (256, 2 ** 16)):
        nb = N // 64
        X = 2 ** 10 / chunks8(D)
        s = block_sums(np.ones(N), np.ones(N, bool), np.zeros((N, 1)), np.zeros(1), N)
        tn, to, sh, c = tolO_both(D, N, X, 1e9, 1e9, s, np.zeros(1), nb)
        print(f"vEff0: D={D} N={N} at the cap: tolO old {to[0]:.3g} -> new {tn[0]:.3g};  tolV(K=R=0) {float(tolV(0.0, 0.0, nb)):.4g}", flush=True)


def big():
    rng = np.random.default_rng(97)
    N = 2 ** 14
    for D in (64, 128):
        q, k, v, s = data('bench', N, D, rng)
        for causal in (False, True):
            report("bench N=2^14", q, k, v, s, causal, np.arange(0, N, 97))


def spread():
    """The benchmark at N = 1024 over six more seeds (every 8th row), for the spread of the medians."""
    for seed in range(50, 56):
        rng = np.random.default_rng(seed)
        for D in (64, 128):
            for causal in (False, True):
                q, k, v, s = data('bench', 1024, D, rng)
                report(f"bench seed {seed}", q, k, v, s, causal, np.arange(0, 1024, 8))


if __name__ == '__main__':
    mode = sys.argv[1:]
    big() if mode == ['big'] else spread() if mode == ['spread'] else main()
