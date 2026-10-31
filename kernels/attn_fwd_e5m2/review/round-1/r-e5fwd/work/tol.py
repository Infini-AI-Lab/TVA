"""Reviewer r-e5fwd: evaluate the specification's output tolerance `tolO` (Model.lean, "The
constants") on data like the host's, with the formulas exactly as Model.lean writes them.

For one head whose `v` is read (`vIn`), each row `n` and column `e`:
  - X = the row's smallest admissible bound, max over the row's keys of sigma * sum_e |q k|;
  - attnRef, errS (at the row's `nblk`), errB, tolO, all as in Model.lean.
It reports the ratio tolO / |attnRef| and the fraction of elements where tolO >= |attnRef|, i.e.
where `Post` accepts the output 0 (|0 - attnRef| <= tolO). For causal runs it also reports row 0,
whose only key is key 0 (attnRef = v[0, e]). It compares tolO with a magnitude bound of the shape
the same library facts give when acc and l_i are compared over the same computed p (an estimate
of that shape, not a proof): |attnRef| + (1 + rhoA)^nblk * (9/8 + nkeys * 2^-17) * max_j |v_j| *
9/8 * (1 + 2^-20) + 2^-17.

Data:
  test   q, k, v ~ normal(0, 0.5) in float16, cast to float8e5 (round to nearest even), sm_scale 0.5
  bench  q, k, v ~ randn in float16, cast the same way, sm_scale 1.3
  ones   q = 0, v = 1 everywhere (X = 0)
Run: python3 -I tol.py
"""
import math
import numpy as np

U32 = 2.0 ** -24


def e5m2(x):
    """Round float64 values to float8e5 (E5M2), nearest even, no saturation needed here."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.zeros_like(x)
    nz = a > 0
    e = np.floor(np.log2(a[nz]))
    e = np.maximum(e, -14.0)          # subnormals below 2^-14 share the quantum 2^-16
    q = 2.0 ** (e - 2)                # 2 fraction bits
    out[nz] = np.round(a[nz] / q) * q  # numpy rounds half to even
    return np.sign(x) * out


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def relDot8(K):
    return (1 + 2.0 ** -7) ** ((K + 31) // 32) - 1


class Const:
    def __init__(self, D, N, X):
        self.expErr = (relDot8(D) + 2.0 ** -19) * (1.5 * X) + 2.0 ** -50
        self.pE = 2.0 ** self.expErr
        dmax = 2 * (1.5 * X + self.expErr) / 2 ** 24 + 2.0 ** -150
        self.rho8 = relDot8(64)
        self.rhoA = (1 + self.rho8) * ((1 + dmax) * ((1 + 2.0 ** -22) * (1 + U32))) - 1
        psiP = self.pE * (1 + 2.0 ** -22) - 1
        psi8 = (1 + psiP) * (9 / 8) - 1
        self.psiA = (1 + self.rho8) * (1 + psi8) - 1
        self.omega8 = 2.0 ** -17 + 9 / 8 * 2.0 ** -126
        self.absA = 2.0 ** -64
        nb = N // 64
        gam63 = (1 + U32) ** 63 - 1
        stepL = (1 + 3 * U32) * ((1 + dmax) * (1 + 2.0 ** -22))
        qL = (1 - 3 * U32) * (2 - (1 + dmax) * (1 + 2.0 ** -22))
        Fhi = (1 + gam63) * (1 + psiP) * stepL ** nb
        Flo = (1 - 3 * U32) * (1 - gam63) * (1 - 2.0 ** -22) / self.pE * qL ** nb
        aL = nb * 2.0 ** -64
        self.Fp = Flo - aL * self.pE
        self.betaL = max(Fhi - 1, 1 - Flo) + aL * self.pE
        lgB = max(math.log2(Fhi + aL * self.pE), -math.log2(self.Fp))
        self.tolM = (1 + U32) * (lgB + (lgB + self.expErr + 16) / 2 ** 22 + 2.0 ** -149) + 2.0 ** -150


def rows_tol(q, k, v, sigma, causal, n0, n1):
    """tolO and attnRef for rows n0..n1-1 of one head: arrays of shape (rows, D)."""
    N, D = q.shape
    nb = N // 64
    S = q[n0:n1] @ k.T                          # exact scores (float64 is exact for e5m2 data)
    SA = np.abs(q[n0:n1]) @ np.abs(k).T
    n_idx = np.arange(n0, n1)[:, None]
    j_idx = np.arange(N)[None, :]
    key = np.ones_like(S, dtype=bool) if not causal else (j_idx <= n_idx)
    X = np.where(key, sigma * SA, -np.inf).max(axis=1)        # the row's smallest admissible X
    Z = np.where(key, sigma * S, -np.inf)
    W = np.exp(Z - Z.max(axis=1, keepdims=True))              # weights, scaled per row
    wsum = W.sum(axis=1)
    attn = (W @ v) / wsum[:, None]
    Wb = W.reshape(len(W), nb, 64)
    vb = v.reshape(nb, 64, D)
    blkS = np.einsum('rbk,bkd->rbd', Wb, vb)                  # sum w v per block
    blkMag = np.einsum('rbk,bkd->rbd', Wb, np.abs(vb))        # sum w |v| per block
    keyb = key.reshape(len(W), nb, 64).astype(np.float64)
    blkAbs = np.einsum('rbk,bkd->rbd', keyb, np.abs(vb))      # sum |v| per block over keys
    nblk = (2 * (np.arange(n0, n1) // 128) + 2) if causal else np.full(n1 - n0, nb)
    tol = np.empty((n1 - n0, D))
    hull = np.empty((n1 - n0, D))
    out = {'tolM': np.empty(n1 - n0), 'X': X}
    for r in range(n1 - n0):
        c = Const(D, N, X[r])
        eS = np.zeros(D)
        eB = np.zeros(D)
        part = np.zeros(D)
        for t in range(nblk[r]):
            eS = (1 + c.rhoA) * eS + c.rhoA * np.abs(part) + c.psiA * blkMag[r, t]
            eB = (1 + c.rhoA) * eB + (1 + c.rho8) * c.omega8 * blkAbs[r, t] + c.absA
            part = part + blkS[r, t]
        a = np.abs(attn[r])
        tol[r] = (9 / 8 * (1 + 2.0 ** -22) * (eS / wsum[r] / c.Fp + c.pE * eB / c.Fp + c.betaL / c.Fp * a)
                  + (9 / 8 * 2.0 ** -22 + 1 / 8) * a + 9 / 8 * 2.0 ** -148 + 2.0 ** -17)
        out['tolM'][r] = c.tolM
        # A magnitude bound of the shape the same facts give when acc and l_i are compared over the
        # same computed p (both are scaled by the same alpha): |o| <= (1 + rhoA)^nblk * (9/8 +
        # nkeys * 2^-17) * max_j |v_j| * 9/8 * (1 + 2^-20) + 2^-17, so |o - attnRef| <= |attnRef| + that.
        nkeys = int(key[r].sum())
        vmax = np.abs(v[key[r]]).max(axis=0)
        hull[r] = a + (1 + c.rhoA) ** nblk[r] * (9 / 8 + nkeys * 2.0 ** -17) * vmax * 9 / 8 * (1 + 2.0 ** -20) + 2.0 ** -17
    out['hull'] = hull
    out['tol'] = tol
    out['attn'] = attn
    return out


def report(name, q, k, v, sigma, causal, rows):
    N, D = q.shape
    res = [rows_tol(q, k, v, sigma, causal, n0, min(n0 + 256, N)) for n0 in rows]
    tol = np.concatenate([r['tol'] for r in res])
    attn = np.concatenate([r['attn'] for r in res])
    X = np.concatenate([r['X'] for r in res])
    tolM = np.concatenate([r['tolM'] for r in res])
    hull = np.concatenate([r['hull'] for r in res])
    acc0 = np.mean(tol >= np.abs(attn))
    hullr = f"  magnitude bound median {np.median(hull):.3g}, tolO/bound median {np.median(tol / hull):.3g} max {(tol / hull).max():.3g}"
    row0 = ''
    if causal and rows[0] == 0:
        # row 0 attends to key 0 only: attnRef = v[0, e]
        row0 = f"  row 0 (one key): X {X[0]:.1f}, tolO/|attnRef| median {np.median(tol[0] / np.abs(attn[0])):.3g}"
    print(f"{name}: D={D} N={N} causal={causal} rows={len(tol)}  X max {X.max():.1f}  "
          f"tolO median {np.median(tol):.3g} max {tol.max():.3g}  |attnRef| median {np.median(np.abs(attn)):.3g} "
          f"max {np.abs(attn).max():.3g}  max|v| {np.abs(v).max():.3g}  "
          f"zero accepted on {100 * acc0:.1f}% of elements  tolM max {tolM.max():.3g}" + row0 + hullr)


def data(kind, N, D, rng):
    if kind == 'test':
        g = lambda: e5m2(f16(rng.normal(0.0, 0.5, size=(N, D))))
        return g(), g(), g(), 0.5
    g = lambda: e5m2(f16(rng.standard_normal(size=(N, D))))
    return g(), g(), g(), float(np.float32(1.3))


def main():
    rng = np.random.default_rng(20)
    for D in (64, 128):
        for N in (1024, 4096):
            for causal in (False, True):
                q, k, v, s = data('test', N, D, rng)
                rows = [0, N // 2] if N > 1024 else [0, 512]
                report('test ', q, k, v, s, causal, rows)
    for D in (64, 128):
        for N in (1024,):
            for causal in (False, True):
                q, k, v, s = data('bench', N, D, rng)
                report('bench', q, k, v, s, causal, [0, 512])
    # q = 0, v = 1 everywhere: every weight 1, X = 0
    for N in (1024, 4096, 2 ** 14, 2 ** 16):
        c = Const(64, N, 0.0)
        nb = N // 64
        eS = 0.0
        eB = 0.0
        part = 0.0
        for t in range(nb):
            eS = (1 + c.rhoA) * eS + c.rhoA * abs(part) + c.psiA * 64
            eB = (1 + c.rhoA) * eB + (1 + c.rho8) * c.omega8 * 64 + c.absA
            part += 64
        tol = (9 / 8 * (1 + 2.0 ** -22) * (eS / N / c.Fp + c.pE * eB / c.Fp + c.betaL / c.Fp)
               + (9 / 8 * 2.0 ** -22 + 1 / 8) + 9 / 8 * 2.0 ** -148 + 2.0 ** -17)
        print(f"ones : D=64 N={N} not causal  tolO {tol:.4g}  (attnRef = 1)")


main()
